import { useState } from 'react';
import {
  VizPanel,
  Segmented,
  Slider,
  Button,
  Legend,
  Stats,
  Note,
  TooltipHost,
  useTip,
  fmtNum,
  useSize,
} from './Viz';

/**
 * The buffer pool as three cooperating structures, stepped one micro-operation at a time.
 *
 * Models, at the granularity PostgreSQL's bufmgr.c works at:
 *  - a fixed array of frames, each with a descriptor holding tag / refcount (pin) /
 *    usage count / BM_DIRTY / BM_VALID / BM_IO_IN_PROGRESS,
 *  - a hash table keyed by BufferTag (relation, fork, block), bucketed, with the
 *    partition lock each bucket falls under,
 *  - the miss path: probe -> victim (free list, then clock sweep) -> write back if
 *    dirty -> swap the tag under both partition locks -> mark BM_IO_IN_PROGRESS ->
 *    read -> mark BM_VALID and wake waiters.
 *
 * Two backends share the pool, so a second reader of the same missing page finds the
 * hash entry already inserted, pins the frame, sees BM_IO_IN_PROGRESS and blocks in
 * WaitIO instead of issuing a second read of the same block.
 *
 * Deterministic by construction: the page table hashes the tag with FNV-1a, and there is no
 * randomness anywhere in the component.
 */

const NBUCKETS = 8;
const NPART = 4; // real PostgreSQL: NUM_BUFFER_PARTITIONS = 128
const MAX_USAGE = 5; // BM_MAX_USAGE_COUNT
const MAX_FRAMES = 8;

/* ------------------------------------------------------------------- pages */

type Rel = 'orders' | 'orders_pkey';
type Fork = 'main' | 'fsm';
type Tag = { rel: Rel; fork: Fork; block: number };

const PAGES: Tag[] = [
  { rel: 'orders', fork: 'main', block: 0 },
  { rel: 'orders', fork: 'main', block: 1 },
  { rel: 'orders', fork: 'main', block: 2 },
  { rel: 'orders', fork: 'main', block: 3 },
  { rel: 'orders', fork: 'main', block: 4 },
  { rel: 'orders', fork: 'main', block: 5 },
  { rel: 'orders', fork: 'fsm', block: 0 },
  { rel: 'orders_pkey', fork: 'main', block: 0 },
  { rel: 'orders_pkey', fork: 'main', block: 1 },
  { rel: 'orders_pkey', fork: 'main', block: 2 },
];

const key = (t: Tag) => `${t.rel}/${t.fork}/${t.block}`;
const short = (t: Tag) => `${t.rel === 'orders_pkey' ? 'pk' : t.fork === 'fsm' ? 'fsm' : 'ord'} ${t.block}`;
const long = (t: Tag) => `${t.rel}/${t.fork}/${t.block}`;
const compact = (t: Tag) => `${t.rel === 'orders_pkey' ? 'pk' : 'ord'}/${t.fork}/${t.block}`;

/** FNV-1a over the tag — a real hash of a real key, identical on the server and in the
 *  browser on every render. Nothing in this component is random. */
function fnv1a(str: string) {
  let h = 0x811c9dc5;
  for (let i = 0; i < str.length; i++) {
    h ^= str.charCodeAt(i);
    h = Math.imul(h, 0x01000193) >>> 0;
  }
  return h >>> 0;
}

const HASH: Record<string, number> = (() => {
  const m: Record<string, number> = {};
  for (const t of PAGES) m[key(t)] = fnv1a(key(t));
  return m;
})();

const bucketOf = (t: Tag) => HASH[key(t)] % NBUCKETS;
/** Both the bucket and the partition come from the same hash code, so partition is a
 *  function of the bucket — exactly as in PostgreSQL, where nbuckets and
 *  NUM_BUFFER_PARTITIONS are both powers of two. */
const partOf = (b: number) => b % NPART;

/* ------------------------------------------------------------------- state */

type Frame = {
  id: number;
  tag: Tag | null;
  pin: number;
  usage: number;
  dirty: boolean;
  valid: boolean;
  io: boolean;
};

type Phase = 'idle' | 'lookup' | 'victim' | 'writeback' | 'swap' | 'io' | 'wait' | 'done' | 'error';

type Backend = {
  id: 'A' | 'B';
  pid: number;
  phase: Phase;
  want: Tag | null;
  buf: number | null;
  victim: number | null;
  pins: number[];
  line: string;
};

type LogRow = {
  n: number;
  who: string;
  op: string;
  detail: string;
  hits: number;
  reads: number;
  evict: number;
  written: number;
};

type S = {
  frames: Frame[];
  hand: number;
  A: Backend;
  B: Backend;
  hits: number;
  reads: number;
  evict: number;
  written: number;
  waits: number;
  head: string;
  body: string;
  log: LogRow[];
};

const PHASE_LINE: Record<Phase, string> = {
  idle: 'idle — no request in flight',
  lookup: 'BufTableLookup() — probing the bucket',
  victim: 'StrategyGetBuffer() — choosing a victim',
  writeback: 'FlushBuffer() — victim is dirty',
  swap: 'BufTableDelete + BufTableInsert',
  io: 'smgrread() — BM_IO_IN_PROGRESS',
  wait: 'WaitIO() — blocked on another backend',
  done: 'holds the buffer pinned',
  error: 'ERROR: no unpinned buffers available',
};

function mkFrames(n: number): Frame[] {
  return Array.from({ length: n }, (_, i) => ({
    id: i,
    tag: null,
    pin: 0,
    usage: 0,
    dirty: false,
    valid: false,
    io: false,
  }));
}

function mkBackend(id: 'A' | 'B', pid: number): Backend {
  return { id, pid, phase: 'idle', want: null, buf: null, victim: null, pins: [], line: PHASE_LINE.idle };
}

const INITIAL: S = {
  frames: mkFrames(MAX_FRAMES),
  hand: 0,
  A: mkBackend('A', 4711),
  B: mkBackend('B', 4712),
  hits: 0,
  reads: 0,
  evict: 0,
  written: 0,
  waits: 0,
  head: 'An empty pool: eight frames, every descriptor tagless and on the free list.',
  body:
    'Click a page in the "on disk" strip to make the active backend call ReadBuffer() on it, then press ' +
    'Step to walk the request one micro-operation at a time. Fill the pool, dirty a page, then request ' +
    'something new and watch the victim get written back before its frame can be reused.',
  log: [],
};

function clone(s: S): S {
  return {
    ...s,
    frames: s.frames.map((f) => ({ ...f })),
    A: { ...s.A, pins: [...s.A.pins] },
    B: { ...s.B, pins: [...s.B.pins] },
    log: [...s.log],
  };
}

function push(s: S, who: string, op: string, detail: string) {
  s.log.push({
    n: s.log.length + 1,
    who,
    op,
    detail,
    hits: s.hits,
    reads: s.reads,
    evict: s.evict,
    written: s.written,
  });
}

/* ------------------------------------------------------------- transitions */

function request(prev: S, who: 'A' | 'B', t: Tag): S {
  const s = clone(prev);
  const be = who === 'A' ? s.A : s.B;
  if (be.phase !== 'idle' && be.phase !== 'done' && be.phase !== 'error') return prev;
  be.want = t;
  be.buf = null;
  be.victim = null;
  be.phase = 'lookup';
  be.line = PHASE_LINE.lookup;
  const b = bucketOf(t);
  s.head = `Backend ${who}: ReadBuffer(${long(t)}).`;
  s.body =
    `The tag is the key: relation, fork, block number — nothing about the page's contents. Hashing it gives ` +
    `bucket ${b}, which lives under partition lock ${partOf(b)}. The backend takes that one lock in shared ` +
    `mode; the other ${NPART - 1} partitions stay open to everybody else.`;
  push(s, who, 'ReadBuffer', long(t));
  return s;
}

function step(prev: S, who: 'A' | 'B'): S {
  const s = clone(prev);
  const be = who === 'A' ? s.A : s.B;
  const other = who === 'A' ? s.B : s.A;
  const t = be.want;
  if (!t) return prev;

  switch (be.phase) {
    case 'lookup': {
      const b = bucketOf(t);
      const hit = s.frames.find((f) => f.tag && key(f.tag) === key(t));
      if (hit) {
        hit.pin += 1;
        if (hit.usage < MAX_USAGE) hit.usage += 1;
        be.buf = hit.id;
        be.pins.push(hit.id);
        if (hit.io || !hit.valid) {
          be.phase = 'wait';
          s.waits += 1;
          s.hits += 1;
          s.head = `Backend ${who} found the entry — and the page is not there yet.`;
          s.body =
            `Buffer ${hit.id} is already tagged ${long(t)} but carries BM_IO_IN_PROGRESS: backend ${other.id} ` +
            `inserted the hash entry before starting the read. ${who} pins the frame and blocks in WaitIO() on the ` +
            `buffer's condition variable rather than issuing a second read of the same block. Note what the counter ` +
            `does: this is charged as a *hit*, because the entry was found — so an EXPLAIN plan can say ` +
            `"shared hit" for a page it waited on the disk for.`;
          push(s, who, 'WaitIO', `buf ${hit.id} — BM_IO_IN_PROGRESS`);
        } else {
          be.phase = 'done';
          s.hits += 1;
          s.head = `Hit: ${long(t)} is in buffer ${hit.id}.`;
          s.body =
            `One probe of bucket ${b} under partition lock ${partOf(b)}, then PinBuffer() bumps the refcount to ` +
            `${hit.pin} and the usage count to ${hit.usage} (capped at BM_MAX_USAGE_COUNT = ${MAX_USAGE}) with a ` +
            `single compare-and-swap on the descriptor's packed state word. The partition lock is released ` +
            `immediately; the pin is what keeps the frame from moving under the reader's feet.`;
          push(s, who, 'PinBuffer', `buf ${hit.id} — hit, pin=${hit.pin} usage=${hit.usage}`);
        }
      } else {
        s.reads += 1;
        be.phase = 'victim';
        s.head = `Miss: no entry for ${long(t)} in bucket ${b}.`;
        s.body =
          `The probe came back empty, so this counts as a read (EXPLAIN's "shared read", pg_stat_io's reads). ` +
          `Before anything can be read in, the backend needs a frame to read into — and it has to find one ` +
          `that nobody has pinned.`;
        push(s, who, 'BufTableLookup', `bucket ${b} — miss`);
      }
      be.line = PHASE_LINE[be.phase];
      return s;
    }

    case 'victim': {
      const free = s.frames.find((f) => f.tag === null && f.pin === 0 && !f.io);
      if (free) {
        be.victim = free.id;
        be.phase = 'swap';
        be.line = PHASE_LINE.swap;
        s.head = `Victim: buffer ${free.id}, straight off the free list.`;
        s.body =
          `While the pool has never been filled, StrategyGetBuffer() pops a frame from the free list and skips ` +
          `the clock sweep entirely. This is why a cold pool has a completely different cost profile from a warm ` +
          `one: no victim search, no write-back, just an empty frame waiting.`;
        push(s, who, 'StrategyGetBuffer', `buf ${free.id} from the free list`);
        return s;
      }
      const n = s.frames.length;
      let hand = s.hand;
      let scanned = 0;
      let chosen: Frame | null = null;
      let decremented = 0;
      while (scanned < n * (MAX_USAGE + 1)) {
        const f = s.frames[hand % n];
        hand = (hand + 1) % n;
        scanned += 1;
        if (f.pin > 0 || f.io) continue;
        if (f.usage === 0) {
          chosen = f;
          break;
        }
        f.usage -= 1;
        decremented += 1;
      }
      s.hand = hand;
      if (!chosen) {
        be.phase = 'error';
        be.line = PHASE_LINE.error;
        s.head = 'ERROR: no unpinned buffers available';
        s.body =
          `Every frame in the pool is pinned, so there is nothing the sweep is allowed to take. PostgreSQL raises ` +
          `exactly this error from StrategyGetBuffer() after a full circuit finds no candidate. In production you ` +
          `only see it with an absurdly small shared_buffers or a backend that has leaked pins — but it is the ` +
          `proof that the pin contract is absolute: the pool would rather fail the query than evict a pinned frame.`;
        push(s, who, 'StrategyGetBuffer', 'no unpinned buffers available');
        return s;
      }
      be.victim = chosen.id;
      be.phase = chosen.dirty ? 'writeback' : 'swap';
      be.line = PHASE_LINE[be.phase];
      s.head = `Victim: buffer ${chosen.id}${chosen.dirty ? ' — and it is dirty.' : '.'}`;
      s.body =
        `The clock hand swept ${scanned} descriptor${scanned === 1 ? '' : 's'}, decrementing the usage count of ` +
        `${decremented} frame${decremented === 1 ? '' : 's'} it passed over and skipping every pinned one, until ` +
        `it landed on a frame with usage_count = 0 and refcount = 0. ` +
        (chosen.dirty
          ? 'Its page has been modified since it was read in, so the frame cannot simply be reused: the old contents have to reach disk first.'
          : 'The page is clean — an identical copy is already on disk — so it can be overwritten with no I/O at all.');
      push(s, who, 'StrategyGetBuffer', `buf ${chosen.id} (${chosen.dirty ? 'dirty' : 'clean'})`);
      return s;
    }

    case 'writeback': {
      const v = s.frames[be.victim ?? -1];
      if (!v) {
        be.phase = 'victim';
        be.line = PHASE_LINE.victim;
        return s;
      }
      v.dirty = false;
      s.written += 1;
      be.phase = 'swap';
      be.line = PHASE_LINE.swap;
      s.head = `Write-back: buffer ${v.id} flushed before it may be reused.`;
      s.body =
        `FlushBuffer() sets BM_IO_IN_PROGRESS for a write, flushes the WAL up to the page's LSN first, writes the ` +
        `page through smgrwrite(), then clears BM_DIRTY. The query that caused the miss pays for it — this is the ` +
        `"buffers written" in EXPLAIN (ANALYZE, BUFFERS) and the reason a read-only SELECT can show write I/O.`;
      push(s, who, 'FlushBuffer', `buf ${v.id} written`);
      return s;
    }

    case 'swap': {
      const v = s.frames[be.victim ?? -1];
      if (!v || v.pin > 0) {
        be.phase = 'victim';
        be.line = PHASE_LINE.victim;
        return s;
      }
      // BufferAlloc re-probes with the partition lock held exclusively before inserting:
      // if another backend inserted this tag while we were hunting for a victim, its entry
      // wins and we give our victim back rather than tagging a second frame with the same page.
      const raced = s.frames.find((f) => f.tag && key(f.tag) === key(t) && f.id !== v.id);
      if (raced) {
        raced.pin += 1;
        if (raced.usage < MAX_USAGE) raced.usage += 1;
        be.buf = raced.id;
        be.pins.push(raced.id);
        be.victim = null;
        s.reads -= 1;
        s.hits += 1;
        const blocking = raced.io || !raced.valid;
        if (blocking) {
          be.phase = 'wait';
          s.waits += 1;
        } else {
          be.phase = 'done';
        }
        be.line = PHASE_LINE[be.phase];
        s.head = `Lost the race: ${long(t)} is already in buffer ${raced.id}.`;
        s.body =
          `Backend ${other.id} inserted the entry while this backend was searching for a victim. ` +
          `BufTableInsert reports the existing buffer instead of adding a second entry for the same tag, ` +
          `so the victim is handed back untouched and this backend pins the winner's frame` +
          (blocking
            ? `, then blocks in WaitIO() until the read completes. The counter moves from read to hit: the ` +
              `entry was found in the end.`
            : ` — counted as a hit, because the entry was found.`);
        push(s, who, 'BufTableInsert', `existing entry for ${long(t)} → buf ${raced.id}`);
        return s;
      }
      const old = v.tag;
      if (old) s.evict += 1;
      v.tag = t;
      v.valid = false;
      v.io = true;
      v.dirty = false;
      v.pin = 1;
      v.usage = 1;
      be.buf = v.id;
      be.pins.push(v.id);
      be.phase = 'io';
      be.line = PHASE_LINE.io;
      s.head = `Tag swap: buffer ${v.id} now answers to ${long(t)}.`;
      s.body =
        (old
          ? `The entry for ${long(old)} is deleted and an entry for ${long(t)} is inserted, which means both ` +
            `partition locks are held at once — taken in a fixed order so two backends swapping in opposite ` +
            `directions cannot deadlock. `
          : 'The new entry is inserted into its bucket under that bucket\'s partition lock. ') +
        `The frame is pinned, marked BM_IO_IN_PROGRESS and left *invalid*: the tag says which page this frame ` +
        `is for, BM_VALID says whether the bytes have arrived. Anyone probing for this page from now on finds ` +
        `the entry and waits instead of starting a second read.`;
      push(s, who, 'BufTableInsert', `buf ${v.id} ← ${long(t)}${old ? `, evicted ${long(old)}` : ''}`);
      return s;
    }

    case 'io': {
      const f = s.frames[be.buf ?? -1];
      if (!f) return prev;
      f.io = false;
      f.valid = true;
      be.phase = 'done';
      be.line = PHASE_LINE.done;
      let woke = 0;
      if (other.phase === 'wait' && other.buf === f.id) {
        other.phase = 'done';
        other.line = PHASE_LINE.done;
        woke += 1;
      }
      s.head = `Read complete: buffer ${f.id} is BM_VALID.`;
      s.body =
        `TerminateBufferIO() clears BM_IO_IN_PROGRESS, sets BM_VALID and broadcasts on the buffer's condition ` +
        `variable. ` +
        (woke
          ? `Backend ${other.id} wakes up already holding its pin and returns the buffer — one physical read served ` +
            `two readers. Had the read failed, the frame would be left invalid with BM_IO_ERROR and the waiter would ` +
            `see the failure rather than a half-filled page.`
          : `Only now may anyone read the bytes: a frame with a tag but without BM_VALID is a promise, not a page.`);
      push(s, who, 'TerminateBufferIO', `buf ${f.id} valid${woke ? `, woke ${other.id}` : ''}`);
      return s;
    }

    case 'wait': {
      s.head = `Backend ${who} is blocked and cannot make progress on its own.`;
      s.body =
        `It is parked in WaitIO() until whoever owns the read finishes it. Step the other backend to completion ` +
        `and this one wakes. In a real server this shows up as a wait event — BufferIO, or IO/DataFileRead on the ` +
        `backend actually doing the read.`;
      return s;
    }

    default:
      return prev;
  }
}

function finish(prev: S, who: 'A' | 'B'): S {
  let s = prev;
  for (let i = 0; i < 10; i++) {
    const be = who === 'A' ? s.A : s.B;
    if (be.phase === 'idle' || be.phase === 'done' || be.phase === 'wait' || be.phase === 'error') break;
    s = step(s, who);
  }
  return s;
}

function unpin(prev: S, who: 'A' | 'B', all: boolean): S {
  const s = clone(prev);
  const be = who === 'A' ? s.A : s.B;
  if (be.pins.length === 0 || be.phase === 'wait') return prev;
  const drop = all ? be.pins.slice() : [be.pins[be.pins.length - 1]];
  be.pins = all ? [] : be.pins.slice(0, -1);
  for (const id of drop) {
    const f = s.frames[id];
    if (f && f.pin > 0) f.pin -= 1;
  }
  if (be.pins.length === 0 && be.phase === 'done') {
    be.phase = 'idle';
    be.buf = null;
    be.line = PHASE_LINE.idle;
  }
  s.head = `Backend ${who} released ${drop.length} pin${drop.length === 1 ? '' : 's'}.`;
  s.body =
    `ReleaseBuffer() decrements the refcount in the descriptor's state word. The page stays in the pool and stays ` +
    `dirty if it was dirty — dropping a pin says "I am no longer looking at this frame", never "write this out". ` +
    `Only at refcount 0 does the frame become a legal victim again. A backend that forgets this is where ` +
    `"buffer refcount leak" warnings come from.`;
  push(s, who, 'ReleaseBuffer', drop.map((d) => `buf ${d}`).join(', '));
  return s;
}

function markDirty(prev: S, who: 'A' | 'B'): S {
  const s = clone(prev);
  const be = who === 'A' ? s.A : s.B;
  const id = be.pins[be.pins.length - 1];
  const f = id === undefined ? undefined : s.frames[id];
  if (!f || !f.valid || f.dirty) return prev;
  f.dirty = true;
  s.head = `Buffer ${f.id} is now dirty.`;
  s.body =
    `MarkBufferDirty() sets BM_DIRTY in the state word while the backend holds both the pin and an exclusive ` +
    `content lock. The in-memory page and the on-disk page now differ, and that difference is the pool's only copy ` +
    `until a checkpoint or a victim search writes it out. From here, evicting this frame costs a write.`;
  push(s, who, 'MarkBufferDirty', `buf ${f.id}`);
  return s;
}

function resize(prev: S, n: number): S {
  const s = clone(prev);
  if (n > s.frames.length) {
    const added = n - s.frames.length;
    while (s.frames.length < n) {
      s.frames.push({ id: s.frames.length, tag: null, pin: 0, usage: 0, dirty: false, valid: false, io: false });
    }
    s.head = `Pool grown to ${s.frames.length} frames.`;
    s.body =
      `${added} new frame${added === 1 ? '' : 's'} went straight onto the free list, so the next misses cost no ` +
      `victim search at all. Growing is the easy direction. InnoDB does it online in innodb_buffer_pool_chunk_size ` +
      `units; PostgreSQL's shared_buffers is fixed at postmaster start, so this slider is a restart.`;
    push(s, 'admin', 'resize', `${s.frames.length} frames`);
    return s;
  }
  let blocked: Frame | null = null;
  let flushed = 0;
  let dropped = 0;
  while (s.frames.length > n) {
    const last = s.frames[s.frames.length - 1];
    if (last.pin > 0 || last.io) {
      blocked = last;
      break;
    }
    if (last.dirty) {
      s.written += 1;
      flushed += 1;
    }
    if (last.tag) s.evict += 1;
    s.frames.pop();
    dropped += 1;
  }
  s.hand = s.frames.length ? s.hand % s.frames.length : 0;
  for (const b of [s.A, s.B]) {
    b.pins = b.pins.filter((id) => id < s.frames.length);
    if (b.buf !== null && b.buf >= s.frames.length) b.buf = null;
    if (b.victim !== null && b.victim >= s.frames.length) b.victim = null;
  }
  s.head = blocked
    ? `Shrink stalled at ${s.frames.length} frames: buffer ${blocked.id} is pinned.`
    : `Pool shrunk to ${s.frames.length} frames.`;
  s.body = blocked
    ? `The frame being withdrawn has refcount ${blocked.pin}${blocked.io ? ' and an I/O in progress' : ''}, and the ` +
      `pin contract does not bend for a resize either. InnoDB's online shrink has exactly this shape: it withdraws ` +
      `chunks, relocating or flushing the pages inside them, and waits on pages it cannot move — the error log ` +
      `reports the resize starting and completing, and in between the pool is smaller than you asked for.`
    : `${dropped} frame${dropped === 1 ? '' : 's'} withdrawn; ${flushed} dirty page${flushed === 1 ? '' : 's'} had ` +
      `to be written out first, and every page in them is gone from the cache. Shrinking a pool is not free memory, ` +
      `it is a burst of write I/O followed by a colder cache.`;
  push(s, 'admin', 'resize', `${s.frames.length} frames${blocked ? ' (stalled: pinned)' : ''}`);
  return s;
}

/* ----------------------------------------------------------------- drawing */

const LEFT_W = 310;
const GRID_X = LEFT_W + 16;
const CARD_W = 120;
const CARD_H = 100;
const CARD_GAP = 8;
const COLS = 4;
const CHIP_W = 66;
const CHIP_H = 34;
const MIN_W = 846;

const DISK_Y = 22;
const SEC_Y = 78;
const BODY_Y = 86;
const ROW_H = 26;
const LANE_Y = 318;
const LANE_H = 38;
const HEIGHT = LANE_Y + 2 * (LANE_H + 6) + 6;

function frameColor(f: Frame) {
  if (!f.tag) return 'var(--viz-ink-muted)';
  if (f.io || !f.valid) return 'var(--viz-warning)';
  return f.dirty ? 'var(--viz-dirty)' : 'var(--viz-clean)';
}

function flagsOf(f: Frame) {
  const out: string[] = [];
  if (f.valid) out.push('V');
  if (f.dirty) out.push('D');
  if (f.io) out.push('IO');
  return out.length ? out.join(' ') : '—';
}

function phaseColor(p: Phase) {
  if (p === 'idle') return 'var(--viz-ink-muted)';
  if (p === 'done') return 'var(--viz-good)';
  if (p === 'wait' || p === 'error') return 'var(--viz-critical)';
  return 'var(--viz-warning)';
}

/* --------------------------------------------------------------- component */

export default function BufferFrameDescriptorLab() {
  const [s, setS] = useState<S>(INITIAL);
  const [who, setWho] = useState<'A' | 'B'>('A');
  const [ref, width] = useSize(MIN_W);
  const tip = useTip();

  const be = who === 'A' ? s.A : s.B;
  const svgW = Math.max(width, MIN_W);
  const used = s.frames.filter((f) => f.tag).length;
  const pinned = s.frames.filter((f) => f.pin > 0).length;
  const total = s.hits + s.reads;

  const cardXY = (i: number) => ({
    x: GRID_X + (i % COLS) * (CARD_W + CARD_GAP),
    y: BODY_Y + Math.floor(i / COLS) * (CARD_H + 12),
  });

  const buckets = Array.from({ length: NBUCKETS }, (_, b) =>
    s.frames.filter((f) => f.tag && bucketOf(f.tag) === b),
  );
  const probing = be.phase === 'lookup' && be.want ? bucketOf(be.want) : -1;

  return (
    <VizPanel
      title="Frames, the page table and the pin contract"
      subtitle="Two backends share one pool. Click a page to call ReadBuffer(), then step through the probe, the victim search, the write-back, the tag swap and the I/O — and request the same missing page from the other backend to watch it block instead of double-reading."
      controls={
        <>
          <Segmented
            label="Active backend"
            value={who}
            onChange={setWho}
            options={[
              { value: 'A', label: 'A · pid 4711' },
              { value: 'B', label: 'B · pid 4712' },
            ]}
          />
          <Button
            onClick={() => setS((cur) => step(cur, who))}
            disabled={be.phase === 'idle' || be.phase === 'done' || be.phase === 'error'}
            primary
          >
            Step
          </Button>
          <Button
            onClick={() => setS((cur) => finish(cur, who))}
            disabled={be.phase === 'idle' || be.phase === 'done' || be.phase === 'wait' || be.phase === 'error'}
            title="Run this backend's request to completion"
          >
            Finish request
          </Button>
          <Button
            onClick={() => setS((cur) => markDirty(cur, who))}
            disabled={be.pins.length === 0 || be.phase === 'wait'}
            title="MarkBufferDirty() on the most recently pinned buffer"
          >
            Mark dirty
          </Button>
          <Button onClick={() => setS((cur) => unpin(cur, who, false))} disabled={be.pins.length === 0 || be.phase === 'wait'}>
            Unpin last
          </Button>
          <Button onClick={() => setS((cur) => unpin(cur, who, true))} disabled={be.pins.length === 0 || be.phase === 'wait'}>
            Unpin all
          </Button>
          <Slider
            label="Pool size"
            min={2}
            max={MAX_FRAMES}
            value={s.frames.length}
            onChange={(n) => setS((cur) => resize(cur, n))}
            format={(n) => `${n} frames`}
          />
          <Button onClick={() => setS(INITIAL)}>Reset</Button>
        </>
      }
      legend={
        <Legend
          items={[
            { label: 'V — valid, clean', color: 'var(--viz-clean)' },
            { label: 'D — valid, dirty', color: 'var(--viz-dirty)' },
            { label: 'IO — tagged, BM_IO_IN_PROGRESS, not yet valid', color: 'var(--viz-warning)' },
            { label: 'free — no tag, on the free list', color: 'var(--viz-ink-muted)' },
            { label: 'pin > 0 — heavy outline, never a victim', color: 'var(--viz-ink)', shape: 'line' },
            { label: '▶ clock sweep hand', color: 'var(--viz-ink-2)' },
          ]}
        />
      }
      stats={
        <Stats
          items={[
            { label: 'shared hit', value: fmtNum(s.hits), hint: 'Found in the page table — what EXPLAIN (ANALYZE, BUFFERS) calls a hit' },
            { label: 'shared read', value: fmtNum(s.reads), hint: 'Buffer-pool miss: a read was issued to the smgr layer' },
            { label: 'Hit ratio', value: total ? `${((s.hits / total) * 100).toFixed(0)}%` : '—' },
            { label: 'Evictions', value: fmtNum(s.evict), hint: 'Frames whose tag was swapped out from under a resident page' },
            { label: 'Buffers written', value: fmtNum(s.written), hint: 'Dirty victims flushed by the backend that needed the frame' },
            { label: 'Blocked on I/O', value: fmtNum(s.waits), hint: 'Times a backend found BM_IO_IN_PROGRESS and waited instead of re-reading' },
            { label: 'Frames used', value: `${used} / ${s.frames.length}`, hint: `${pinned} currently pinned` },
          ]}
        />
      }
      note={
        <Note>
          <strong>{s.head}</strong> {s.body}
        </Note>
      }
      table={
        <>
          <table className="viz-table">
            <thead>
              <tr>
                <th>Buffer</th>
                <th>BufferTag</th>
                <th>Bucket</th>
                <th>Partition lock</th>
                <th>refcount (pin)</th>
                <th>usage_count</th>
                <th>Flags</th>
              </tr>
            </thead>
            <tbody>
              {s.frames.map((f) => (
                <tr key={f.id}>
                  <td>{f.id}</td>
                  <td>{f.tag ? long(f.tag) : '— free —'}</td>
                  <td>{f.tag ? bucketOf(f.tag) : '—'}</td>
                  <td>{f.tag ? partOf(bucketOf(f.tag)) : '—'}</td>
                  <td>{f.pin}</td>
                  <td>{f.usage}</td>
                  <td>{flagsOf(f)}</td>
                </tr>
              ))}
            </tbody>
          </table>
          <table className="viz-table">
            <thead>
              <tr>
                <th>#</th>
                <th>Actor</th>
                <th>Operation</th>
                <th>Detail</th>
                <th>hit</th>
                <th>read</th>
                <th>evicted</th>
                <th>written</th>
              </tr>
            </thead>
            <tbody>
              {s.log.length === 0 ? (
                <tr>
                  <td colSpan={8}>Nothing requested yet.</td>
                </tr>
              ) : (
                s.log.map((r) => (
                  <tr key={r.n}>
                    <td>{r.n}</td>
                    <td>{r.who}</td>
                    <td>{r.op}</td>
                    <td>{r.detail}</td>
                    <td>{r.hits}</td>
                    <td>{r.reads}</td>
                    <td>{r.evict}</td>
                    <td>{r.written}</td>
                  </tr>
                ))
              )}
            </tbody>
          </table>
        </>
      }
    >
      <div ref={ref}>
        <TooltipHost>
          <svg
            width={svgW}
            height={HEIGHT}
            role="img"
            aria-label="A buffer pool: pages on disk, a partitioned page table, eight frame descriptors and two backends"
          >
            {/* ------------------------------------------------ pages on disk */}
            <text x={0} y={14} fill="var(--viz-ink)" fontWeight={600}>
              On disk — click a page to ReadBuffer() it as backend {who}
            </text>
            {PAGES.map((t, i) => {
              const x = 0 + i * (CHIP_W + 6);
              const f = s.frames.find((fr) => fr.tag && key(fr.tag) === key(t));
              const resident = !!f;
              return (
                <g
                  key={key(t)}
                  {...tip(
                    <>
                      <strong>{long(t)}</strong>
                      <br />
                      hash → bucket {bucketOf(t)}, partition lock {partOf(bucketOf(t))}
                      <br />
                      {f ? `resident in buffer ${f.id}` : 'not in the pool — requesting it is a miss'}
                    </>,
                  )}
                  role="button"
                  tabIndex={0}
                  aria-label={`Request ${long(t)}`}
                  style={{ cursor: 'pointer' }}
                  onClick={() => setS((cur) => request(cur, who, t))}
                  onKeyDown={(e) => {
                    if (e.key === 'Enter' || e.key === ' ') {
                      e.preventDefault();
                      setS((cur) => request(cur, who, t));
                    }
                  }}
                >
                  <rect
                    x={x}
                    y={DISK_Y}
                    width={CHIP_W}
                    height={CHIP_H}
                    rx={6}
                    fill="var(--viz-plane)"
                    stroke={resident ? frameColor(f!) : 'var(--viz-border)'}
                    strokeWidth={resident ? 2 : 1}
                  />
                  <text x={x + CHIP_W / 2} y={DISK_Y + 15} textAnchor="middle" fill="var(--viz-ink)">
                    {short(t)}
                  </text>
                  <text x={x + CHIP_W / 2} y={DISK_Y + 27} textAnchor="middle" fill="var(--viz-ink-muted)">
                    {f ? `buf ${f.id}` : 'on disk'}
                  </text>
                </g>
              );
            })}

            {/* ----------------------------------------------- the page table */}
            <text x={0} y={SEC_Y} fill="var(--viz-ink)" fontWeight={600}>
              Page table — hash on BufferTag, {NPART} partition locks
            </text>
            {buckets.map((entries, b) => {
              const y = BODY_Y + b * ROW_H;
              const active = b === probing;
              return (
                <g key={b}>
                  <rect
                    x={0}
                    y={y}
                    width={LEFT_W}
                    height={ROW_H - 4}
                    rx={5}
                    fill="var(--viz-plane)"
                    stroke={active ? 'var(--viz-1)' : 'var(--viz-border)'}
                    strokeWidth={active ? 2 : 1}
                  />
                  <text x={8} y={y + 15} fill={active ? 'var(--viz-ink)' : 'var(--viz-ink-2)'}>
                    b{b} · p{partOf(b)}
                  </text>
                  {entries.slice(0, 3).map((f, k) => (
                    <g
                      key={f.id}
                      {...tip(
                        <>
                          <strong>{long(f.tag!)} → buffer {f.id}</strong>
                          <br />
                          The hash entry is inserted before the read starts, which is what lets a second reader
                          find the frame and wait on it.
                        </>,
                      )}
                    >
                      <rect
                        x={62 + k * 80}
                        y={y + 2}
                        width={76}
                        height={ROW_H - 8}
                        rx={4}
                        fill="var(--viz-surface)"
                        stroke={frameColor(f)}
                        strokeWidth={1.5}
                      />
                      <text x={62 + k * 80 + 38} y={y + 15} textAnchor="middle" fill="var(--viz-ink)">
                        {short(f.tag!)} → {f.id}
                      </text>
                    </g>
                  ))}
                  {entries.length === 0 ? (
                    <text x={68} y={y + 15} fill="var(--viz-ink-muted)">
                      empty
                    </text>
                  ) : null}
                </g>
              );
            })}

            {/* ----------------------------------------------- the descriptors */}
            <text x={GRID_X} y={SEC_Y} fill="var(--viz-ink)" fontWeight={600}>
              Frames + descriptors — {s.frames.length} × 8 KB
            </text>
            {s.frames.map((f, i) => {
              const { x, y } = cardXY(i);
              const col = frameColor(f);
              const isHand = i === s.hand % Math.max(s.frames.length, 1);
              return (
                <g
                  key={f.id}
                  {...tip(
                    <>
                      <strong>buffer {f.id}</strong>
                      <br />
                      {f.tag ? long(f.tag) : 'no tag — on the free list'}
                      <br />
                      refcount {f.pin}, usage_count {f.usage}, flags {flagsOf(f)}
                      <br />
                      {f.pin > 0
                        ? 'Pinned: the clock sweep will skip it, and its tag cannot be swapped.'
                        : 'Unpinned: a legal victim once its usage count reaches 0.'}
                    </>,
                  )}
                >
                  <rect
                    x={x}
                    y={y}
                    width={CARD_W}
                    height={CARD_H}
                    rx={8}
                    fill="var(--viz-plane)"
                    stroke={f.pin > 0 ? 'var(--viz-ink)' : col}
                    strokeWidth={f.pin > 0 ? 3 : 1.5}
                  />
                  <rect x={x + 1} y={y + 8} width={5} height={CARD_H - 16} rx={2.5} fill={col} />
                  <text x={x + 14} y={y + 18} fill="var(--viz-ink)" fontWeight={600}>
                    buf {f.id}
                  </text>
                  <text x={x + CARD_W - 8} y={y + 18} textAnchor="end" fill={col} fontWeight={600}>
                    {flagsOf(f)}
                  </text>
                  <text x={x + 14} y={y + 36} fill={f.tag ? 'var(--viz-ink)' : 'var(--viz-ink-muted)'}>
                    {f.tag ? compact(f.tag) : '— free —'}
                  </text>
                  <text x={x + 14} y={y + 54} fill="var(--viz-ink-2)">
                    refcount {f.pin}
                  </text>
                  <text x={x + 14} y={y + 72} fill="var(--viz-ink-2)">
                    usage
                  </text>
                  {Array.from({ length: MAX_USAGE }, (_, u) => (
                    <rect
                      key={u}
                      x={x + 52 + u * 12}
                      y={y + 64}
                      width={9}
                      height={9}
                      rx={2}
                      fill={u < f.usage ? 'var(--viz-seq-400)' : 'var(--viz-neutral)'}
                      stroke="var(--viz-border)"
                    />
                  ))}
                  <text x={x + 14} y={y + 90} fill="var(--viz-ink-muted)">
                    {f.tag ? `bucket ${bucketOf(f.tag)} · part ${partOf(bucketOf(f.tag))}` : 'free list'}
                  </text>
                  {isHand ? (
                    <polygon
                      points={`${x - 11},${y + CARD_H / 2 - 6} ${x - 2},${y + CARD_H / 2} ${x - 11},${y + CARD_H / 2 + 6}`}
                      fill="var(--viz-ink-2)"
                    />
                  ) : null}
                </g>
              );
            })}

            {/* ------------------------------------------------- the backends */}
            {[s.A, s.B].map((b, i) => {
              const y = LANE_Y + i * (LANE_H + 6);
              const active = b.id === who;
              return (
                <g key={b.id}>
                  <rect
                    x={0}
                    y={y}
                    width={svgW - 4}
                    height={LANE_H}
                    rx={7}
                    fill="var(--viz-plane)"
                    stroke={active ? 'var(--viz-1)' : 'var(--viz-border)'}
                    strokeWidth={active ? 2 : 1}
                  />
                  <text x={10} y={y + 23} fill="var(--viz-ink)" fontWeight={600}>
                    Backend {b.id} · pid {b.pid}
                  </text>
                  <circle cx={140} cy={y + 19} r={5} fill={phaseColor(b.phase)} />
                  <text x={152} y={y + 23} fill={phaseColor(b.phase)} fontWeight={600}>
                    {b.phase}
                  </text>
                  <text x={232} y={y + 23} fill="var(--viz-ink-2)">
                    {b.line}
                    {b.want ? ` · wants ${long(b.want)}` : ''}
                    {b.pins.length ? ` · pins held: ${b.pins.map((p) => `buf ${p}`).join(', ')}` : ' · no pins held'}
                  </text>
                </g>
              );
            })}
          </svg>
        </TooltipHost>
      </div>
    </VizPanel>
  );
}
