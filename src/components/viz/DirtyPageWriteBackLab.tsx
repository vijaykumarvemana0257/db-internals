import { useRef, useState } from 'react';
import {
  VizPanel,
  Segmented,
  Slider,
  Check,
  Button,
  Legend,
  Stats,
  Note,
  TooltipHost,
  useTip,
  useTicker,
  useSize,
  makeRng,
  fmtBytes,
  fmtNum,
} from './Viz';

/**
 * Dirty pages, write-back and the WAL-before-data rule.
 *
 * The model is the real one, shrunk: a pool of frames whose descriptors carry a dirty bit,
 * an oldest-modification LSN (recLSN) and a newest-modification LSN (pageLSN); a WAL whose
 * insert pointer runs ahead of its flushed pointer; and the rule that a frame may not be
 * written until the WAL is durable through its pageLSN.
 *
 * Two engines, because the interesting difference is structural:
 *   - InnoDB keeps a flush list ordered by oldest_modification, so the checkpoint LSN — and
 *     therefore where recovery starts — is whatever sits at the old end of that list, and it
 *     advances continuously as the page cleaner drains it.
 *   - PostgreSQL keeps no flush list. Recovery starts at the redo point of the last *completed*
 *     checkpoint, so it moves in steps, and a checkpoint is a full scan of the pool whose
 *     victims are sorted by file and block and written out over checkpoint_completion_target.
 *
 * Scaled down (24 frames, a redo file measured in KB) so a whole checkpoint cycle fits in a
 * few hundred ticks. Every ratio is real; the absolute byte counts are not.
 */

const FRAMES = 24;
const PAGES = 64; // distinct blocks in the relation
const HOT = 16; // the hot set
const COLS = 12;
const HIST = 48; // ticks kept on the I/O timeline
const FPI_BYTES = 8192; // a full-page image, logged on first touch after the redo point

type Engine = 'pg' | 'innodb';

type Frame = {
  id: number;
  page: number | null;
  dirty: boolean;
  recLSN: number; // oldest_modification: the LSN of the change that first dirtied it
  pageLSN: number; // newest_modification: pd_lsn / FIL_PAGE_LSN
  usage: number; // clock-sweep usage count, capped at 5
  fpi: boolean; // a full-page image has already been logged since the redo point
  ckpt: boolean; // BM_CHECKPOINT_NEEDED
  justWritten: number; // tick it was last written, for the flash
};

type Row = {
  tick: number;
  cleaner: number; // checkpointer (pg) / page cleaner (innodb)
  bgw: number; // background writer (pg) / LRU-list flush (innodb)
  backend: number; // a backend wrote its own victim
  forced: number; // XLogFlush calls forced by the rule
  forcedBytes: number;
  dirty: number;
  age: number;
  state: 'ok' | 'async' | 'sync' | 'ckpt';
  mods: number;
};

type S = {
  tick: number;
  frames: Frame[];
  hand: number; // clock sweep position
  lsn: number; // WAL insert pointer
  flushed: number; // WAL durable pointer
  prevLsn: number; // where the insert pointer stood one tick ago (the WAL writer's lag)
  ckptRedo: number; // redo point of the running checkpoint
  ckptDone: number; // redo point of the last COMPLETED checkpoint (pg recovery start)
  ckptActive: boolean;
  ckptStart: number;
  ckptTodo: number[]; // frame ids marked BM_CHECKPOINT_NEEDED, in sorted-write order
  ckptWritten: number;
  lastCkptEnd: number;
  ckptCount: number;
  writes: { cleaner: number; bgw: number; backend: number };
  forced: number;
  forcedBytes: number;
  walBytes: number;
  fpiBytes: number;
  mods: number;
  stalls: number;
  head: string;
  body: string;
  hist: Row[];
};

type Cfg = {
  engine: Engine;
  rate: number; // page modifications offered per tick
  io: number; // flush budget in pages/tick (innodb_io_capacity, checkpointer+bgwriter)
  cap: number; // redo capacity in bytes (innodb_redo_log_capacity / max_wal_size)
  interval: number; // checkpoint_timeout, in ticks
  spread: boolean; // checkpoint_completion_target 0.9 / innodb_adaptive_flushing
};

const MAX_DIRTY_PCT = 90; // innodb_max_dirty_pages_pct
const LWM_DIRTY_PCT = 10; // innodb_max_dirty_pages_pct_lwm
const ADAPTIVE_LWM = 0.1; // innodb_adaptive_flushing_lwm, as a fraction of redo capacity
const ASYNC_AT = 7 / 8; // classic async-preflush water mark, as a fraction of max checkpoint age
const SYNC_AT = 15 / 16; // classic sync-flush water mark: user threads wait in log_free_check()
// PostgreSQL requests a checkpoint once WAL since the last one passes
// max_wal_size / (2 + checkpoint_completion_target) — CalculateCheckpointSegments().
const PG_CKPT_DIVISOR = 2.9;

function blank(): S {
  return {
    tick: 0,
    frames: Array.from({ length: FRAMES }, (_, i) => ({
      id: i,
      page: null,
      dirty: false,
      recLSN: 0,
      pageLSN: 0,
      usage: 0,
      fpi: false,
      ckpt: false,
      justWritten: -99,
    })),
    hand: 0,
    lsn: 4096,
    flushed: 4096,
    prevLsn: 4096,
    ckptRedo: 4096,
    ckptDone: 4096,
    ckptActive: false,
    ckptStart: 0,
    ckptTodo: [],
    ckptWritten: 0,
    lastCkptEnd: 0,
    ckptCount: 0,
    writes: { cleaner: 0, bgw: 0, backend: 0 },
    forced: 0,
    forcedBytes: 0,
    walBytes: 0,
    fpiBytes: 0,
    mods: 0,
    stalls: 0,
    head: 'Idle pool, empty log.',
    body:
      'Press Run. Frames turn orange as they are modified, each stamped with the LSN of the record that ' +
      'changed it. Nothing may be written back to disk until the WAL is durable through that LSN.',
    hist: [],
  };
}

/** Where crash recovery would have to start reading, right now. */
function recoveryStart(s: S, engine: Engine) {
  if (engine === 'pg') return s.ckptDone;
  let min = Infinity;
  for (const f of s.frames) if (f.dirty && f.recLSN < min) min = f.recLSN;
  return min === Infinity ? s.lsn : min;
}

function dirtyCount(s: S) {
  return s.frames.reduce((n, f) => n + (f.dirty ? 1 : 0), 0);
}

/**
 * Write one frame back. This is the whole rule: if the log is not durable through the
 * frame's pageLSN, the writer must flush the WAL first, synchronously, before the 8 KB
 * write may be issued.
 */
function writeBack(s: S, f: Frame, who: 'cleaner' | 'bgw' | 'backend') {
  if (f.pageLSN > s.flushed) {
    s.forced++;
    s.forcedBytes += f.pageLSN - s.flushed;
    s.flushed = f.pageLSN; // XLogFlush(BufferGetLSN(buf)) / log_write_up_to(newest_modification)
  }
  f.dirty = false;
  f.recLSN = 0;
  f.ckpt = false;
  f.justWritten = s.tick;
  s.writes[who]++;
}

/** Clock sweep, exactly as in the structure page: decrement usage, take the first zero. */
function pickVictim(s: S) {
  for (let n = 0; n < FRAMES * 6; n++) {
    const f = s.frames[s.hand];
    s.hand = (s.hand + 1) % FRAMES;
    if (f.page === null) return f;
    if (f.usage === 0) return f;
    f.usage--;
  }
  return s.frames[s.hand];
}

function step(prev: S, cfg: Cfg): S {
  const s: S = {
    ...prev,
    frames: prev.frames.map((f) => ({ ...f })),
    writes: { ...prev.writes },
    ckptTodo: [...prev.ckptTodo],
    hist: prev.hist,
    tick: prev.tick + 1,
  };
  const rng = makeRng(((s.tick + 1) * 2654435761) >>> 0);
  const before = { ...s.writes };
  const forced0 = s.forced;
  const forcedB0 = s.forcedBytes;

  /* ---- 1. how much redo is outstanding, and is the engine throttling for it? ---- */
  const age0 = s.lsn - recoveryStart(prev, cfg.engine);
  const ratio0 = age0 / cfg.cap;
  const sync = cfg.engine === 'innodb' && ratio0 >= SYNC_AT;
  const asyncFlush = cfg.engine === 'innodb' && ratio0 >= ASYNC_AT && !sync;

  /* ---- 2. the workload ---- */
  // InnoDB throttles user threads in log_free_check() once the checkpoint age nears capacity.
  // PostgreSQL never throttles for WAL volume; it requests a checkpoint instead.
  let mods = cfg.rate;
  if (sync) mods = Math.max(1, Math.round(cfg.rate * 0.15));
  else if (asyncFlush) mods = Math.max(1, Math.round(cfg.rate * 0.6));

  let evictionsBlocked = 0;
  for (let i = 0; i < mods; i++) {
    const page = rng() < 0.72 ? Math.floor(rng() * HOT) : Math.floor(rng() * PAGES);
    let f = s.frames.find((x) => x.page === page);
    if (!f) {
      const v = pickVictim(s);
      if (v.dirty) {
        // No clean victim: this backend writes the page itself, and eats the WAL flush.
        if (v.pageLSN > s.flushed) evictionsBlocked++;
        writeBack(s, v, 'backend');
      }
      v.page = page;
      v.usage = 0;
      v.fpi = false;
      v.pageLSN = 0;
      f = v;
    }
    const logsFpi = cfg.engine === 'pg' && !f.fpi;
    const bytes = (logsFpi ? FPI_BYTES : 0) + 220 + Math.floor(rng() * 200);
    const start = s.lsn;
    s.lsn += bytes;
    s.walBytes += bytes;
    if (logsFpi) {
      s.fpiBytes += FPI_BYTES;
      f.fpi = true;
    }
    if (!f.dirty) {
      f.dirty = true;
      f.recLSN = start; // oldest_modification
    }
    f.pageLSN = s.lsn; // the stamp is the END LSN of the record
    f.usage = Math.min(5, f.usage + 1);
    s.mods++;
  }

  /* ---- 3. the WAL writer, running one tick behind the inserters ---- */
  s.flushed = Math.max(s.flushed, prev.prevLsn);
  s.prevLsn = s.lsn;

  /* ---- 4. the flushers ---- */
  let budget = cfg.io;
  let state: Row['state'] = 'ok';

  if (cfg.engine === 'innodb') {
    if (sync) {
      budget = cfg.io * 2; // innodb_io_capacity_max: aggressive "sync" flushing
      state = 'sync';
      s.stalls++;
    } else if (asyncFlush) {
      budget = Math.ceil(cfg.io * 1.5);
      state = 'async';
    }
    const dirtyPct = (dirtyCount(s) / FRAMES) * 100;
    let want: number;
    if (cfg.spread) {
      // Adaptive flushing: rate rises with checkpoint age past innodb_adaptive_flushing_lwm,
      // and with the dirty-page ratio past innodb_max_dirty_pages_pct_lwm.
      const byAge = ratio0 <= ADAPTIVE_LWM ? 0 : Math.pow((ratio0 - ADAPTIVE_LWM) / (1 - ADAPTIVE_LWM), 1.4);
      const byDirty =
        dirtyPct <= LWM_DIRTY_PCT ? 0 : Math.min(1, (dirtyPct - LWM_DIRTY_PCT) / (MAX_DIRTY_PCT - LWM_DIRTY_PCT));
      want = Math.ceil(Math.max(byAge, byDirty) * budget);
    } else {
      // Adaptive flushing off: idle until the dirty ratio blows through the limit, then
      // flush flat out. This is what a burst looks like.
      want = dirtyPct >= MAX_DIRTY_PCT ? budget : dirtyPct > 60 ? 1 : 0;
    }
    if (sync) want = budget;
    // Drain the flush list from the old end: lowest oldest_modification first. That is the
    // only order that moves the checkpoint LSN.
    const list = s.frames.filter((f) => f.dirty).sort((a, b) => a.recLSN - b.recLSN);
    for (const f of list.slice(0, Math.max(0, want))) writeBack(s, f, 'cleaner');
    // The LRU-list cleaner keeps free frames available so user threads never write their own.
    const lru = s.frames
      .filter((f) => f.dirty && f.usage === 0)
      .sort((a, b) => a.usage - b.usage)
      .slice(0, Math.max(0, Math.floor(cfg.io * 0.25)));
    for (const f of lru) writeBack(s, f, 'bgw');
  } else {
    /* PostgreSQL: a background writer that runs all the time, and a checkpointer that runs
       on a schedule and writes a snapshot of the dirty set. */
    const bgBudget = Math.max(1, Math.round(cfg.io * 0.3));
    const candidates = s.frames
      .filter((f) => f.dirty && f.usage === 0)
      .sort((a, b) => a.recLSN - b.recLSN)
      .slice(0, bgBudget);
    for (const f of candidates) writeBack(s, f, 'bgw');

    const walSince = s.lsn - s.ckptDone;
    if (!s.ckptActive && (s.tick - s.lastCkptEnd >= cfg.interval || walSince > cfg.cap / PG_CKPT_DIVISOR)) {
      s.ckptActive = true;
      s.ckptStart = s.tick;
      s.ckptRedo = s.lsn; // the redo point is taken now, before any buffer is written
      // BufferSync marks every dirty buffer, then writes them in file/block order.
      s.ckptTodo = s.frames
        .filter((f) => f.dirty)
        .sort((a, b) => (a.page ?? 0) - (b.page ?? 0))
        .map((f) => f.id);
      for (const id of s.ckptTodo) s.frames[id].ckpt = true;
      s.ckptWritten = 0;
      for (const f of s.frames) f.fpi = false; // a new checkpoint interval: FPIs start again
    }
    if (s.ckptActive) {
      state = 'ckpt';
      const target = cfg.spread ? 0.9 : 0.2; // checkpoint_completion_target
      // IsCheckpointOnSchedule(): keep to whichever is further along, elapsed time as a
      // fraction of checkpoint_timeout or WAL consumed as a fraction of max_wal_size.
      const byTime = (s.tick - s.ckptStart) / Math.max(1, cfg.interval * target);
      const byWal = (s.lsn - s.ckptRedo) / Math.max(1, cfg.cap * target);
      const frac = Math.min(1, Math.max(byTime, byWal));
      const should = Math.ceil(s.ckptTodo.length * frac);
      let allowed = Math.min(budget, Math.max(0, should - s.ckptWritten));
      while (allowed > 0 && s.ckptWritten < s.ckptTodo.length) {
        const f = s.frames[s.ckptTodo[s.ckptWritten]];
        s.ckptWritten++;
        if (f.dirty && f.ckpt) {
          writeBack(s, f, 'cleaner');
          allowed--;
        }
      }
      if (s.ckptWritten >= s.ckptTodo.length) {
        // fsync the files, then update pg_control: this is the moment recovery's start moves.
        s.ckptActive = false;
        s.ckptDone = s.ckptRedo;
        s.lastCkptEnd = s.tick;
        s.ckptCount++;
      }
    }
  }

  /* ---- 5. narrate ---- */
  const age = s.lsn - recoveryStart(s, cfg.engine);
  const d = dirtyCount(s);
  const nForced = s.forced - forced0;
  const nBackend = s.writes.backend - before.backend;

  if (sync) {
    s.head = 'Sync flush: user threads are being throttled.';
    s.body =
      `The checkpoint age is ${fmtBytes(age)}, ${Math.round((age / cfg.cap) * 100)}% of the redo capacity. ` +
      'InnoDB cannot overwrite redo that the checkpoint still needs, so log_free_check() makes the ' +
      'writers wait while the page cleaners flush flat out. Throughput collapses and no query is slow ' +
      'for any reason a plan would show you.';
  } else if (nBackend > 0) {
    s.head = `${nBackend} page${nBackend === 1 ? '' : 's'} written by a backend, not by a flusher.`;
    s.body =
      'The clock sweep found no clean victim, so the thread that wanted to read a page had to write ' +
      "somebody else's dirty one first" +
      (evictionsBlocked > 0 ? ', and flush the WAL through its pageLSN before it was allowed to.' : '.') +
      ' Every one of these is a foreground read that turned into a write plus (sometimes) a log flush.';
  } else if (nForced > 0) {
    s.head = `${nForced} write${nForced === 1 ? '' : 's'} had to flush the WAL first.`;
    s.body =
      `${fmtBytes(s.forcedBytes - forcedB0)} of log were forced out ahead of the data write because the ` +
      'pages being written carried a pageLSN past the flushed pointer. This is the rule that makes ' +
      'steal safe: the log describing the change is on disk before the changed page is.';
  } else if (state === 'ckpt') {
    s.head = `Checkpoint running — ${s.ckptWritten}/${s.ckptTodo.length} buffers written.`;
    s.body =
      'The redo point was taken when the checkpoint started; recovery will not move to it until every ' +
      'marked buffer is written and fsynced. Until then the recovery-start marker stays where the ' +
      'previous checkpoint left it.';
  } else {
    s.head = `${d} of ${FRAMES} frames dirty, checkpoint age ${fmtBytes(age)}.`;
    s.body =
      cfg.engine === 'innodb'
        ? 'The head of the flush list — the oldest oldest_modification in the pool — is exactly where ' +
          'recovery would start. Every page the cleaner drains from that end moves it forward.'
        : 'Recovery starts at the redo point of the last completed checkpoint, and moves only when the ' +
          'next one finishes. Dirty buffers written in between buy nothing until then.';
  }

  const row: Row = {
    tick: s.tick,
    cleaner: s.writes.cleaner - before.cleaner,
    bgw: s.writes.bgw - before.bgw,
    backend: s.writes.backend - before.backend,
    forced: nForced,
    forcedBytes: s.forcedBytes - forcedB0,
    dirty: d,
    age,
    state,
    mods,
  };
  s.hist = [...prev.hist, row].slice(-HIST);
  return s;
}

/** The "evict a dirty frame now" button: the rule, on demand, on the worst possible page. */
function forceEvict(prev: S, cfg: Cfg): S {
  const s: S = {
    ...prev,
    frames: prev.frames.map((f) => ({ ...f })),
    writes: { ...prev.writes },
    ckptTodo: [...prev.ckptTodo],
  };
  const victim = s.frames.filter((f) => f.dirty).sort((a, b) => b.pageLSN - a.pageLSN)[0];
  if (!victim) {
    return { ...prev, head: 'No dirty frame to evict.', body: 'Run the workload for a few ticks first.' };
  }
  const gap = victim.pageLSN - s.flushed;
  writeBack(s, victim, 'backend');
  s.head =
    gap > 0
      ? `Blocked: XLogFlush(${fmtBytes(victim.pageLSN)}) before the 8 KB write.`
      : 'Allowed straight through: the log was already durable past this page.';
  s.body =
    gap > 0
      ? `Block ${victim.page} carried a pageLSN ${fmtBytes(gap)} past the flushed pointer, so ` +
        (cfg.engine === 'pg'
          ? 'FlushBuffer() called XLogFlush() and waited for the WAL fdatasync before it handed the block to smgrwrite(). '
          : 'the page cleaner called log_write_up_to(newest_modification) and waited before issuing the write. ') +
        'The eviction cost a synchronous log flush on top of the page write — and the WAL now covers ' +
        'every other dirty page stamped below that LSN too, so the next few writes are free of it.'
      : `Block ${victim.page} was last changed at ${fmtBytes(victim.pageLSN)} and the log is durable to ` +
        `${fmtBytes(s.flushed)}. XLogNeedsFlush() returns false and the write goes out with no log I/O at ` +
        'all — which is the common case for a page that has been sitting dirty for a while, and the whole ' +
        'case for a page dirtied only by hint bits.';
  return s;
}

/* ------------------------------------------------------------------ drawing */

const C = {
  clean: 'var(--viz-clean)',
  dirty: 'var(--viz-dirty)',
  empty: 'var(--viz-neutral)',
  // The timeline's three writer bands. --viz-1 is taken by --viz-clean (clean frames) and
  // --viz-2 by --viz-dirty, so the bands start at slot 6; every band is also readable by
  // tooltip and in the "Show the numbers" table.
  cleaner: 'var(--viz-6)',
  bgw: 'var(--viz-4)',
  backend: 'var(--viz-7)',
};

function Figure({ s, cfg, width }: { s: S; cfg: Cfg; width: number }) {
  const tip = useTip();
  const svgW = Math.max(width, 760);
  const cellW = Math.floor((svgW - 8) / COLS);
  const cellH = 46;

  const gridTop = 24;
  const gridH = 2 * cellH;
  const walTop = gridTop + gridH + 42;
  const walH = 26;
  const listTop = walTop + walH + 52;
  const listH = 22;
  const tlTop = listTop + listH + 40;
  const tlH = 96;
  const height = tlTop + tlH + 26;

  const start = recoveryStart(s, cfg.engine);
  const age = s.lsn - start;
  const barW = svgW - 16;
  const pos = (lsn: number) => 8 + barW * Math.min(1, Math.max(0, (lsn - start) / cfg.cap));
  const ageFrac = Math.min(1, age / cfg.cap);
  const ageColor =
    cfg.engine === 'innodb'
      ? ageFrac >= SYNC_AT
        ? 'var(--viz-critical)'
        : ageFrac >= ASYNC_AT
          ? 'var(--viz-warning)'
          : C.dirty
      : ageFrac >= 1
        ? 'var(--viz-serious)'
        : C.dirty;

  const list = s.frames
    .filter((f) => f.dirty)
    .sort((a, b) => (cfg.engine === 'innodb' ? a.recLSN - b.recLSN : (a.page ?? 0) - (b.page ?? 0)));

  const maxIo = Math.max(4, ...s.hist.map((r) => r.cleaner + r.bgw + r.backend));
  const colW = (svgW - 16) / HIST;

  return (
    <svg
      width={svgW}
      height={height}
      role="img"
      aria-label="Buffer pool frames with dirty bits and pageLSNs, the redo capacity bar, the flush list and an I/O timeline"
    >
      {/* ---- pool ---- */}
      <text x={0} y={14} fill="var(--viz-ink)" fontWeight={600}>
        Buffer pool — {FRAMES} frames
      </text>
      <text x={svgW - 4} y={14} textAnchor="end" fill="var(--viz-ink-muted)">
        tick {s.tick} · WAL insert {fmtBytes(s.lsn)} · flushed {fmtBytes(s.flushed)}
      </text>
      {s.frames.map((f, i) => {
        const x = 4 + (i % COLS) * cellW;
        const y = gridTop + Math.floor(i / COLS) * cellH;
        const blocked = f.dirty && f.pageLSN > s.flushed;
        const fill = f.page === null ? C.empty : f.dirty ? C.dirty : C.clean;
        const glyph = f.page === null ? '' : f.dirty ? (blocked ? 'D!' : 'D') : '·';
        return (
          <g
            key={f.id}
            style={{ cursor: 'help' }}
            {...tip(
              f.page === null ? (
                <>
                  <strong>frame {f.id}</strong>
                  <br />
                  empty
                </>
              ) : (
                <>
                  <strong>
                    frame {f.id} — block {f.page}
                  </strong>
                  <br />
                  {f.dirty ? 'BM_DIRTY set' : 'clean'} · usage_count {f.usage}
                  <br />
                  pageLSN {fmtBytes(f.pageLSN)}
                  {f.dirty ? <> · recLSN {fmtBytes(f.recLSN)}</> : null}
                  <br />
                  {blocked
                    ? `Cannot be written yet: the WAL is durable only to ${fmtBytes(
                        s.flushed,
                      )}. Writing it would force an XLogFlush first.`
                    : f.dirty
                      ? 'Writable now — the log already covers this page.'
                      : 'Clean: evictable with no I/O at all.'}
                  {f.ckpt ? <br /> : null}
                  {f.ckpt ? 'BM_CHECKPOINT_NEEDED' : ''}
                </>
              ),
            )}
          >
            <rect
              x={x}
              y={y}
              width={cellW - 6}
              height={26}
              rx={5}
              fill={fill}
              stroke={
                blocked
                  ? 'var(--viz-critical)'
                  : s.tick - f.justWritten <= 1
                    ? 'var(--viz-good)'
                    : 'var(--viz-border)'
              }
              strokeWidth={blocked || s.tick - f.justWritten <= 1 ? 2 : 1}
              strokeDasharray={f.ckpt ? '4 2' : undefined}
            />
            <text x={x + (cellW - 6) / 2} y={y + 18} textAnchor="middle" fill="var(--viz-surface)" fontWeight={700}>
              {glyph}
            </text>
            <text x={x + (cellW - 6) / 2} y={y + 40} textAnchor="middle" fill="var(--viz-ink-muted)">
              {f.page === null ? '—' : `b${f.page}`}
            </text>
          </g>
        );
      })}

      {/* ---- redo capacity ---- */}
      <text x={0} y={walTop - 22} fill="var(--viz-ink)" fontWeight={600}>
        {cfg.engine === 'innodb' ? 'Redo log capacity' : 'WAL since the last completed checkpoint'}
      </text>
      <text x={svgW - 4} y={walTop - 22} textAnchor="end" fill="var(--viz-ink-muted)">
        checkpoint age {fmtBytes(age)} of {fmtBytes(cfg.cap)} ({Math.round(ageFrac * 100)}%)
      </text>
      <rect x={8} y={walTop} width={barW} height={walH} rx={5} fill="var(--viz-plane)" stroke="var(--viz-border)" />
      <rect x={8} y={walTop} width={Math.max(1, barW * ageFrac)} height={walH} rx={5} fill={ageColor} opacity={0.85} />
      {(cfg.engine === 'innodb' ? [ASYNC_AT, SYNC_AT] : [1 / PG_CKPT_DIVISOR]).map((t) => (
        <g key={t}>
          <line
            x1={8 + barW * t}
            x2={8 + barW * t}
            y1={walTop - 4}
            y2={walTop + walH + 4}
            stroke="var(--viz-critical)"
            strokeWidth={1.5}
            strokeDasharray="4 3"
          />
          <text x={8 + barW * t - 4} y={walTop + walH + 16} textAnchor="end" fill="var(--viz-critical)">
            {cfg.engine === 'innodb' ? (t === SYNC_AT ? 'sync flush' : 'async preflush') : 'checkpoint requested'}
          </text>
        </g>
      ))}
      <text x={8} y={walTop + walH + 16} fill="var(--viz-ink-2)">
        ◤ recovery starts here — {cfg.engine === 'innodb' ? 'checkpoint LSN' : 'last checkpoint’s redo point'}{' '}
        {fmtBytes(start)}
      </text>
      <line
        x1={pos(s.flushed)}
        x2={pos(s.flushed)}
        y1={walTop - 6}
        y2={walTop + walH}
        stroke="var(--viz-ink)"
        strokeWidth={1.5}
      />
      <text x={pos(s.flushed)} y={walTop - 10} textAnchor="middle" fill="var(--viz-ink-2)">
        flushed
      </text>

      {/* ---- flush list ---- */}
      <text x={0} y={listTop - 22} fill="var(--viz-ink)" fontWeight={600}>
        {cfg.engine === 'innodb'
          ? 'Flush list — dirty frames in oldest_modification order'
          : 'Dirty frames — PostgreSQL keeps no flush list; a checkpoint sorts them by file and block'}
      </text>
      <text x={svgW - 4} y={listTop - 22} textAnchor="end" fill="var(--viz-ink-muted)">
        {cfg.engine === 'innodb'
          ? 'the cleaner drains the left end, and that is what moves the checkpoint LSN'
          : `${s.ckptActive ? `checkpoint in progress: ${s.ckptWritten}/${s.ckptTodo.length}` : `${s.ckptCount} checkpoints complete`}`}
      </text>
      {list.length === 0 ? (
        <text x={8} y={listTop + 16} fill="var(--viz-ink-muted)">
          nothing dirty
        </text>
      ) : null}
      {list.slice(0, 24).map((f, i) => {
        const w = Math.min(46, (svgW - 16) / 24 - 4);
        const x = 8 + i * (w + 4);
        return (
          <g
            key={f.id}
            style={{ cursor: 'help' }}
            {...tip(
              <>
                <strong>block {f.page}</strong>
                <br />
                recLSN {fmtBytes(f.recLSN)} · pageLSN {fmtBytes(f.pageLSN)}
                <br />
                {i === 0 && cfg.engine === 'innodb'
                  ? 'Head of the list: until this page is written, the checkpoint LSN cannot pass its recLSN.'
                  : 'Written in this order, the checkpoint LSN advances monotonically.'}
              </>,
            )}
          >
            <rect
              x={x}
              y={listTop}
              width={w}
              height={listH}
              rx={4}
              fill={C.dirty}
              opacity={i === 0 && cfg.engine === 'innodb' ? 1 : 0.62}
              stroke={f.pageLSN > s.flushed ? 'var(--viz-critical)' : 'var(--viz-border)'}
              strokeWidth={f.pageLSN > s.flushed ? 1.5 : 1}
            />
            <text x={x + w / 2} y={listTop + 15} textAnchor="middle" fill="var(--viz-surface)">
              b{f.page}
            </text>
          </g>
        );
      })}

      {/* ---- I/O timeline ---- */}
      <text x={0} y={tlTop - 22} fill="var(--viz-ink)" fontWeight={600}>
        Page writes per tick
      </text>
      <text x={svgW - 4} y={tlTop - 22} textAnchor="end" fill="var(--viz-ink-muted)">
        peak {maxIo} pages/tick · line = checkpoint age
      </text>
      <line x1={8} x2={svgW - 8} y1={tlTop + tlH} y2={tlTop + tlH} stroke="var(--viz-axis)" />
      {s.hist.map((r, i) => {
        const x = 8 + i * colW;
        const h = (v: number) => (v / maxIo) * tlH;
        const hb = h(r.backend);
        const hg = h(r.bgw);
        const hc = h(r.cleaner);
        const yb = tlTop + tlH - hb;
        const yg = yb - hg;
        const yc = yg - hc;
        return (
          <g
            key={r.tick}
            style={{ cursor: 'help' }}
            {...tip(
              <>
                <strong>tick {r.tick}</strong>
                <br />
                {r.mods} modifications · {r.cleaner + r.bgw + r.backend} page writes
                <br />
                cleaner {r.cleaner} · background {r.bgw} · backend {r.backend}
                <br />
                {r.forced} forced WAL flushes ({fmtBytes(r.forcedBytes)})
                <br />
                {r.dirty} dirty · age {fmtBytes(r.age)}
              </>,
            )}
          >
            {r.state === 'sync' || r.state === 'async' ? (
              <rect
                x={x}
                y={tlTop}
                width={colW}
                height={tlH}
                fill={r.state === 'sync' ? 'var(--viz-critical)' : 'var(--viz-warning)'}
                opacity={0.14}
              />
            ) : null}
            <rect x={x} y={yb} width={colW - 1} height={hb} fill={C.backend} />
            <rect x={x} y={yg} width={colW - 1} height={hg} fill={C.bgw} />
            <rect x={x} y={yc} width={colW - 1} height={hc} fill={C.cleaner} />
            {r.forced > 0 ? (
              <circle cx={x + colW / 2} cy={tlTop + 6} r={2.5} fill="var(--viz-critical)" />
            ) : null}
          </g>
        );
      })}
      <polyline
        fill="none"
        stroke={C.dirty}
        strokeWidth={1.75}
        points={s.hist
          .map((r, i) => `${8 + i * colW + colW / 2},${tlTop + tlH - Math.min(1, r.age / cfg.cap) * tlH}`)
          .join(' ')}
      />
    </svg>
  );
}

/* --------------------------------------------------------------- component */

export default function DirtyPageWriteBackLab() {
  const [engine, setEngine] = useState<Engine>('innodb');
  const [rate, setRate] = useState(6);
  const [io, setIo] = useState(4);
  const [capKb, setCapKb] = useState(64);
  const [ckptTicks, setCkptTicks] = useState(60);
  const [spread, setSpread] = useState(true);
  const [running, setRunning] = useState(false);
  const [s, setS] = useState<S>(blank);
  const [ref, width] = useSize(860);
  const acc = useRef(0);

  const cfg: Cfg = { engine, rate, io, cap: capKb * 1024, interval: ckptTicks, spread };

  useTicker((dt) => {
    acc.current += dt;
    if (acc.current < 240) return;
    acc.current = 0;
    setS((cur) => step(cur, cfg));
  }, running);

  const start = recoveryStart(s, engine);
  const age = s.lsn - start;
  const dirty = dirtyCount(s);
  const totalWrites = s.writes.cleaner + s.writes.bgw + s.writes.backend;
  const backendPct = totalWrites ? (s.writes.backend / totalWrites) * 100 : 0;

  return (
    <VizPanel
      title="Dirty frames, the flush list and the WAL barrier"
      subtitle="Drive a write workload and watch frames dirty, the log's insert pointer run ahead of its flushed pointer, and the flushers race the checkpoint age. Scaled down — 24 frames and a redo file measured in KB — so a whole checkpoint cycle fits on screen; the real defaults are innodb_redo_log_capacity = 100 MB and max_wal_size = 1 GB."
      controls={
        <>
          <Segmented
            label="Engine"
            value={engine}
            onChange={(v) => {
              setEngine(v);
              // PostgreSQL logs a full-page image on each block's first touch after the redo
              // point, so it burns through the scaled-down log far faster than InnoDB does.
              setCapKb(v === 'innodb' ? 64 : 512);
              setS(blank());
              setRunning(false);
            }}
            options={[
              { value: 'innodb', label: 'InnoDB', title: 'Flush list ordered by oldest_modification; continuous checkpoint' },
              { value: 'pg', label: 'PostgreSQL', title: 'No flush list; recovery starts at the last completed checkpoint’s redo point' },
            ]}
          />
          <Slider label="Write rate" min={1} max={16} value={rate} onChange={setRate} format={(n) => `${n} mods/tick`} />
          <Slider
            label={engine === 'innodb' ? 'innodb_io_capacity' : 'Flush budget'}
            min={1}
            max={14}
            value={io}
            onChange={setIo}
            format={(n) => `${n} pages/tick`}
          />
          <Slider
            label={engine === 'innodb' ? 'Redo capacity' : 'max_wal_size'}
            min={16}
            max={1024}
            step={16}
            value={capKb}
            onChange={setCapKb}
            format={(n) => fmtBytes(n * 1024)}
          />
          <Slider
            label="checkpoint_timeout"
            min={20}
            max={200}
            step={10}
            value={ckptTicks}
            onChange={setCkptTicks}
            format={(n) => `${n} ticks`}
            disabled={engine === 'innodb'}
          />
          <Check
            label={engine === 'innodb' ? 'innodb_adaptive_flushing' : 'checkpoint_completion_target 0.9'}
            checked={spread}
            onChange={setSpread}
          />
          <Button onClick={() => setRunning((r) => !r)} primary>
            {running ? 'Pause' : 'Run'}
          </Button>
          <Button onClick={() => setS((cur) => step(cur, cfg))} disabled={running}>
            Step
          </Button>
          <Button
            onClick={() => setS((cur) => forceEvict(cur, cfg))}
            title="Make a thread evict the most recently dirtied frame, right now"
          >
            Evict a dirty frame
          </Button>
          <Button
            onClick={() => {
              setRunning(false);
              setS(blank());
            }}
          >
            Reset
          </Button>
        </>
      }
      legend={
        <Legend
          items={[
            { label: 'clean frame (·)', color: C.clean },
            { label: 'dirty frame (D)', color: C.dirty },
            { label: 'D! — pageLSN past the flushed LSN: a write forces XLogFlush', color: 'var(--viz-critical)' },
            { label: 'checkpointer / page cleaner write', color: C.cleaner },
            { label: 'background writer / LRU flush', color: C.bgw },
            { label: 'backend wrote its own victim', color: C.backend },
            { label: 'checkpoint age', color: C.dirty, shape: 'line' },
          ]}
        />
      }
      stats={
        <Stats
          items={[
            { label: 'Dirty frames', value: `${dirty} / ${FRAMES}`, hint: 'InnoDB compares this ratio to innodb_max_dirty_pages_pct' },
            {
              label: 'Checkpoint age',
              value: `${fmtBytes(age)} (${Math.round((age / cfg.cap) * 100)}%)`,
              hint: 'Log that recovery would have to replay — the distance from the recovery start to the insert pointer',
            },
            {
              label: 'Forced WAL flushes',
              value: `${fmtNum(s.forced)} · ${fmtBytes(s.forcedBytes)}`,
              hint: 'Page writes that had to flush the log through their pageLSN first',
            },
            {
              label: 'Written by backends',
              value: `${Math.round(backendPct)}%`,
              hint: 'Foreground threads that had to write a dirty victim before they could read — pg_stat_io / Innodb_buffer_pool_wait_free',
            },
            { label: 'Pages written', value: fmtNum(totalWrites) },
            {
              label: engine === 'innodb' ? 'Throttled ticks' : 'FPI bytes logged',
              value: engine === 'innodb' ? fmtNum(s.stalls) : fmtBytes(s.fpiBytes),
              hint:
                engine === 'innodb'
                  ? 'Ticks spent in sync flush, with user threads waiting in log_free_check()'
                  : 'Full-page images: the first change to each block after a checkpoint logs the whole 8 KB page',
            },
          ]}
        />
      }
      note={
        <Note>
          <strong>{s.head}</strong> {s.body}
        </Note>
      }
      table={
        <table className="viz-table">
          <thead>
            <tr>
              <th>Tick</th>
              <th>Mods</th>
              <th>Cleaner</th>
              <th>Bg writer</th>
              <th>Backend</th>
              <th>Forced flushes</th>
              <th>Forced bytes</th>
              <th>Dirty</th>
              <th>Checkpoint age</th>
              <th>State</th>
            </tr>
          </thead>
          <tbody>
            {s.hist.length === 0 ? (
              <tr>
                <td colSpan={10}>No ticks yet.</td>
              </tr>
            ) : (
              [...s.hist]
                .reverse()
                .slice(0, 24)
                .map((r) => (
                  <tr key={r.tick}>
                    <td>{r.tick}</td>
                    <td>{r.mods}</td>
                    <td>{r.cleaner}</td>
                    <td>{r.bgw}</td>
                    <td>{r.backend}</td>
                    <td>{r.forced}</td>
                    <td>{fmtBytes(r.forcedBytes)}</td>
                    <td>{r.dirty}</td>
                    <td>{fmtBytes(r.age)}</td>
                    <td>
                      {r.state === 'sync'
                        ? 'sync flush (throttled)'
                        : r.state === 'async'
                          ? 'async preflush'
                          : r.state === 'ckpt'
                            ? 'checkpoint running'
                            : 'steady'}
                    </td>
                  </tr>
                ))
            )}
          </tbody>
        </table>
      }
    >
      <div ref={ref}>
        <TooltipHost>
          <Figure s={s} cfg={cfg} width={width} />
        </TooltipHost>
      </div>
    </VizPanel>
  );
}
