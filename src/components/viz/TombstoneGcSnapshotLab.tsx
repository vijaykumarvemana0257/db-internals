import { useMemo, useState } from 'react';
import {
  VizPanel,
  Choice,
  Slider,
  Button,
  Legend,
  Stats,
  Note,
  TooltipHost,
  useTip,
  fmtNum,
  makeRng,
  useSize,
} from './Viz';

/**
 * Tombstone garbage collection under snapshots — the RocksDB CompactionIterator rules,
 * run by hand.
 *
 * Every version of every key is an internal key (user key, sequence number, type). A
 * compaction may drop a version only when no reader can still be entitled to it:
 *
 *  - shadowed: an older version in the same *snapshot stripe* as a newer one is invisible
 *    to every possible reader, so it goes.
 *  - deletion markers: droppable only when no live snapshot sits BELOW the tombstone's
 *    sequence number (RocksDB: `ikey_.sequence <= earliest_snapshot_`) AND the key exists
 *    nowhere beyond the output level (`KeyNotExistsBeyondOutputLevel`).
 *  - SingleDelete: cancels against its matching Put at any level — and, if the key was
 *    written twice, exposes the older Put. That is the documented undefined behaviour.
 *  - TTL / compaction filter: an expired Put is dropped only if no snapshot can see it.
 */

/* ----------------------------------------------------------------- the model */

const KEYS = 8;
const keyName = (k: number) => `k${String(k + 1).padStart(2, '0')}`;

type Kind = 'put' | 'del' | 'sdel';

type Entry = { id: number; key: number; seq: number; kind: Kind; level: number; born: number };
type RangeTomb = { id: number; lo: number; hi: number; seq: number; level: number; born: number };
type Snap = { id: number; seq: number };

const LEVELS = [
  { name: 'Memtable', sub: 'skip list, not yet an SSTable' },
  { name: 'L0', sub: 'flushed files, overlapping ranges' },
  { name: 'L1', sub: 'sorted runs' },
  { name: 'L2', sub: 'bottommost — nothing below it' },
];

type JobId = 'flush' | 'c01' | 'c12';
type Job = { id: JobId; label: string; inLevels: number[]; out: number };

const JOBS: Record<JobId, Job> = {
  flush: { id: 'flush', label: 'Flush memtable → L0', inLevels: [0], out: 1 },
  c01: { id: 'c01', label: 'Compact L0 → L1', inLevels: [1, 2], out: 2 },
  c12: { id: 'c12', label: 'Compact L1 → L2', inLevels: [2, 3], out: 3 },
};

const jobForLevel = (l: number): JobId => (l === 0 ? 'flush' : l === 1 ? 'c01' : 'c12');

type LogRow = {
  n: number;
  op: string;
  dropped: number;
  purged: number;
  pinned: number;
  live: number;
  internal: number;
};

type ScanResult = { at: string; internal: number; tombs: number; obsolete: number; live: number };

type S = {
  entries: Entry[];
  ranges: RangeTomb[];
  snaps: Snap[];
  nextSeq: number;
  nextId: number;
  nextSnap: number;
  now: number;
  head: string;
  body: string;
  log: LogRow[];
  scan: ScanResult | null;
};

const INITIAL: S = {
  entries: [],
  ranges: [],
  snaps: [],
  nextSeq: 1,
  nextId: 1,
  nextSnap: 1,
  now: 0,
  head: 'An empty LSM tree.',
  body:
    'Put a few keys, delete some of them, then step a compaction and watch which internal keys it is ' +
    'allowed to throw away. Take a snapshot first and the same compaction has to keep them.',
  log: [],
  scan: null,
};

/* ------------------------------------------------------------- the GC rules */

type FateTag = 'keep' | 'shadowed' | 'purged' | 'ttl' | 'pair' | 'pinned' | 'overlap' | 'tip';
type Fate = { drop: boolean; tag: FateTag; why: string };

type Item =
  | { kind: 'put' | 'del' | 'sdel'; seq: number; ref: string; entry: Entry }
  | { kind: 'rdel'; seq: number; ref: string; range: RangeTomb };

type Plan = {
  job: Job;
  fate: Record<string, Fate>;
  dropped: number;
  purged: number;
  pinned: number;
  resurrected: string[];
};

/** How many live snapshots can see a version with this sequence number. */
const stripeOf = (snaps: Snap[], seq: number) => snaps.filter((s) => s.seq >= seq).length;

function planCompaction(s: S, job: Job, ttl: number): Plan {
  const fate: Record<string, Fate> = {};
  const resurrected: string[] = [];
  const inLv = (l: number) => job.inLevels.includes(l);

  /** RocksDB's KeyNotExistsBeyondOutputLevel: is there data for this key we are not merging? */
  const beyond = (k: number) =>
    s.entries.some((e) => e.key === k && !inLv(e.level) && e.level >= job.out) ||
    s.ranges.some((r) => r.lo <= k && k < r.hi && !inLv(r.level) && r.level >= job.out);

  const inputEntries = s.entries.filter((e) => inLv(e.level));
  const inputRanges = s.ranges.filter((r) => inLv(r.level));
  const keptOlder: Record<number, number[]> = {};

  for (let k = 0; k < KEYS; k++) {
    const items: Item[] = [
      ...inputEntries
        .filter((e) => e.key === k)
        .map((e): Item => ({ kind: e.kind, seq: e.seq, ref: `e${e.id}`, entry: e })),
      ...inputRanges
        .filter((r) => r.lo <= k && k < r.hi)
        .map((r): Item => ({ kind: 'rdel', seq: r.seq, ref: `r${r.id}`, range: r })),
    ].sort((a, b) => b.seq - a.seq);

    let lastStripe = -1;
    let pairedStripe = -1;

    for (let i = 0; i < items.length; i++) {
      const it = items[i];
      const st = stripeOf(s.snaps, it.seq);

      if (st === lastStripe) {
        fate[it.ref] = {
          drop: true,
          tag: 'shadowed',
          why:
            `Shadowed: a newer version of ${keyName(k)} sits in the same snapshot stripe, so no reader — ` +
            `at the tip or at any live snapshot — can ever be entitled to this one.`,
        };
        continue;
      }
      lastStripe = st;

      if (it.kind === 'rdel') {
        // The fragment's own fate is decided per range below; here it only shadows.
        continue;
      }

      const e = it.entry;

      if (e.kind === 'sdel') {
        const nx = items[i + 1];
        if (nx && nx.kind === 'put' && stripeOf(s.snaps, nx.seq) === st) {
          const why =
            `SingleDelete met the Put it was promised (#${nx.seq}) with no snapshot between them, so the ` +
            `pair cancels and both disappear — at this level, with no need to reach the bottom.`;
          fate[it.ref] = { drop: true, tag: 'pair', why };
          fate[nx.ref] = { drop: true, tag: 'pair', why };
          i++;
          lastStripe = -1;
          pairedStripe = st;
          continue;
        }
      }

      if (e.kind === 'put') {
        if (pairedStripe === st) {
          resurrected.push(keyName(k));
          pairedStripe = -1;
        }
        const expired = ttl > 0 && s.now - e.born >= ttl;
        if (expired && st === 0) {
          fate[it.ref] = {
            drop: true,
            tag: 'ttl',
            why:
              `Expired: written ${s.now - e.born} operations ago, past the ${ttl}-operation TTL, and no live ` +
              `snapshot can see it. The compaction filter returns kRemove and the row never reaches the output file.`,
          };
          continue;
        }
        fate[it.ref] = {
          drop: false,
          tag: expired ? 'pinned' : 'keep',
          why: expired
            ? `Expired, but a live snapshot still sees it — RocksDB does not run the compaction filter on a version a snapshot is entitled to.`
            : `The newest version of ${keyName(k)} in its stripe. It is the answer to a read, so it is rewritten into the output file.`,
        };
        (keptOlder[k] ||= []).push(e.seq);
        continue;
      }

      // A Delete, or a SingleDelete that found no matching Put.
      const snapBelow = s.snaps.some((sn) => sn.seq < e.seq);
      const over = beyond(k);
      if (!snapBelow && !over) {
        fate[it.ref] = {
          drop: true,
          tag: 'purged',
          why:
            `Purged: no live snapshot sits below sequence ${e.seq}, and ${keyName(k)} exists nowhere beyond ` +
            `${LEVELS[job.out].name}. There is no older version left for this tombstone to hide, so the marker itself goes.`,
        };
        continue;
      }
      fate[it.ref] = {
        drop: false,
        tag: snapBelow ? 'pinned' : 'overlap',
        why: snapBelow
          ? `Pinned: snapshot @${Math.max(...s.snaps.filter((sn) => sn.seq < e.seq).map((sn) => sn.seq))} is older ` +
            `than this delete and must still see ${keyName(k)} as it was before it. Drop the marker and that reader sees a resurrected row.`
          : `${keyName(k)} still has data below ${LEVELS[job.out].name}. Drop the marker here and the older version ` +
            `underneath becomes the newest thing in the tree — the row comes back from the dead.`,
      };
      (keptOlder[k] ||= []).push(e.seq);
    }
  }

  for (const r of inputRanges) {
    const snapBelow = s.snaps.some((sn) => sn.seq < r.seq);
    let over = false;
    for (let k = r.lo; k < r.hi; k++) if (beyond(k)) over = true;
    const shadowingKept = Object.entries(keptOlder).some(
      ([k, seqs]) => Number(k) >= r.lo && Number(k) < r.hi && seqs.some((q) => q < r.seq),
    );
    if (!snapBelow && !over && !shadowingKept) {
      fate[`r${r.id}`] = {
        drop: true,
        tag: 'purged',
        why: `Purged: the fragmented range tombstone [${keyName(r.lo)}, ${keyName(r.hi)}) covers nothing that survives below ${LEVELS[job.out].name}, and no snapshot predates it.`,
      };
    } else {
      fate[`r${r.id}`] = {
        drop: false,
        tag: snapBelow ? 'pinned' : 'overlap',
        why: snapBelow
          ? `Pinned by a snapshot older than sequence ${r.seq}: that reader must still see every key in [${keyName(r.lo)}, ${keyName(r.hi)}).`
          : `Still covering live data below ${LEVELS[job.out].name}, so the fragment is rewritten into the output file's range_del block.`,
      };
    }
  }

  let dropped = 0;
  let purged = 0;
  let pinned = 0;
  for (const f of Object.values(fate)) {
    if (f.drop) dropped++;
    if (f.tag === 'purged' || f.tag === 'pair') purged++;
    if (f.tag === 'pinned') pinned++;
  }
  return { job, fate, dropped, purged, pinned, resurrected };
}

/* --------------------------------------------------------------- the reader */

function resolve(s: S, readSeq: number) {
  let internal = 0;
  let tombs = 0;
  let obsolete = 0;
  let live = 0;
  const winner: Record<number, Item | null> = {};

  for (let k = 0; k < KEYS; k++) {
    const items: Item[] = [
      ...s.entries
        .filter((e) => e.key === k && e.seq <= readSeq)
        .map((e): Item => ({ kind: e.kind, seq: e.seq, ref: `e${e.id}`, entry: e })),
      ...s.ranges
        .filter((r) => r.lo <= k && k < r.hi && r.seq <= readSeq)
        .map((r): Item => ({ kind: 'rdel', seq: r.seq, ref: `r${r.id}`, range: r })),
    ].sort((a, b) => b.seq - a.seq);

    internal += items.length;
    tombs += items.filter((it) => it.kind !== 'put').length;
    const top = items[0] ?? null;
    winner[k] = top;
    if (top && top.kind === 'put') live++;
    obsolete += Math.max(0, items.filter((it) => it.kind === 'put').length - (top && top.kind === 'put' ? 1 : 0));
  }
  return { internal, tombs, obsolete, live, winner };
}

/* -------------------------------------------------------------- chip drawing */

type ChipKind = 'live' | 'obsolete' | 'del' | 'sdel' | 'expired';

const CHIP: Record<ChipKind, { fill: string; glyph: string }> = {
  live: { fill: 'var(--viz-clean)', glyph: 'P' },
  obsolete: { fill: 'var(--viz-stale)', glyph: 'p' },
  del: { fill: 'var(--viz-8)', glyph: 'D' },
  sdel: { fill: 'var(--viz-7)', glyph: 'S' },
  expired: { fill: 'var(--viz-4)', glyph: 'T' },
};

const FATE_GLYPH: Record<FateTag, string> = {
  keep: '',
  tip: '',
  shadowed: '✕',
  purged: '✕',
  ttl: '✕',
  pair: '✕',
  pinned: '◆',
  overlap: '▼',
};

/* ------------------------------------------------------------- the component */

export default function TombstoneGcSnapshotLab() {
  const [key, setKey] = useState('0');
  const [ttl, setTtl] = useState(0);
  const [readAt, setReadAt] = useState('tip');
  const [s, setS] = useState<S>(INITIAL);
  const [ref, width] = useSize(780);
  const tip = useTip();

  const plans = useMemo(
    () => ({
      flush: planCompaction(s, JOBS.flush, ttl),
      c01: planCompaction(s, JOBS.c01, ttl),
      c12: planCompaction(s, JOBS.c12, ttl),
    }),
    [s, ttl],
  );

  const fateOf = (level: number, refId: string): Fate | undefined => plans[jobForLevel(level)].fate[refId];

  const readSeq =
    readAt === 'tip' ? s.nextSeq : (s.snaps.find((x) => String(x.id) === readAt)?.seq ?? s.nextSeq);
  const view = resolve(s, readSeq);
  const tipView = resolve(s, s.nextSeq);

  /* ------------------------------------------------------------- operations */

  const write = (kind: Kind) =>
    setS((cur) => {
      const k = Number(key);
      const e: Entry = { id: cur.nextId, key: k, seq: cur.nextSeq, kind, level: 0, born: cur.now + 1 };
      const label = kind === 'put' ? 'Put' : kind === 'del' ? 'Delete' : 'SingleDelete';
      return {
        ...cur,
        entries: [...cur.entries, e],
        nextId: cur.nextId + 1,
        nextSeq: cur.nextSeq + 1,
        now: cur.now + 1,
        head: `${label}(${keyName(k)}) → internal key (${keyName(k)}, ${cur.nextSeq}, ${
          kind === 'put' ? 'kTypeValue' : kind === 'del' ? 'kTypeDeletion' : 'kTypeSingleDeletion'
        }).`,
        body:
          kind === 'put'
            ? 'One skip-list insert in the memtable, one WAL record. Nothing older was touched or even looked at — the old version is still sitting wherever it was.'
            : 'A delete is a write. Nothing was searched for, nothing was overwritten: a marker went into the memtable at a higher sequence number than everything it hides, and every read of this key from now on has to walk down to it first.',
        log: [...cur.log],
        scan: null,
      };
    });

  const delRange = () =>
    setS((cur) => {
      const lo = Number(key);
      const hi = Math.min(KEYS, lo + 3);
      if (hi <= lo) return cur;
      const r: RangeTomb = { id: cur.nextId, lo, hi, seq: cur.nextSeq, level: 0, born: cur.now + 1 };
      return {
        ...cur,
        ranges: [...cur.ranges, r],
        nextId: cur.nextId + 1,
        nextSeq: cur.nextSeq + 1,
        now: cur.now + 1,
        head: `DeleteRange([${keyName(lo)}, ${keyName(hi)})) at sequence ${cur.nextSeq}.`,
        body:
          'One internal key covers three user keys. It is not written inline with the data — it goes into the ' +
          "file's range_del meta-block, is fragmented into non-overlapping intervals at flush, and is consulted " +
          'by every read of that file. Deleting a million keys costs one record instead of a million.',
        scan: null,
      };
    });

  const snapshot = () =>
    setS((cur) => ({
      ...cur,
      snaps: [...cur.snaps, { id: cur.nextSnap, seq: cur.nextSeq - 1 }],
      nextSnap: cur.nextSnap + 1,
      now: cur.now + 1,
      head: `GetSnapshot() pinned sequence ${cur.nextSeq - 1}.`,
      body:
        'No data was copied. The snapshot is one number in an in-memory list, and its whole effect is on ' +
        'compaction: every version at or below that sequence number is now un-droppable until it is released. ' +
        'A forgotten snapshot is a storage leak with no error message.',
      scan: null,
    }));

  const release = () =>
    setS((cur) => {
      if (cur.snaps.length === 0) return cur;
      const oldest = cur.snaps.reduce((a, b) => (a.seq <= b.seq ? a : b));
      return {
        ...cur,
        snaps: cur.snaps.filter((x) => x.id !== oldest.id),
        now: cur.now + 1,
        head: `ReleaseSnapshot(@${oldest.seq}) — the stripe below it just collapsed.`,
        body:
          'Nothing on disk changed. But the next compaction that touches these files may now drop everything ' +
          'that snapshot was holding: releasing a snapshot is how the space actually comes back.',
        scan: null,
      };
    });

  const runJob = (id: JobId) =>
    setS((cur) => {
      const p = planCompaction(cur, JOBS[id], ttl);
      const job = JOBS[id];
      const survivors = cur.entries.map((e) =>
        job.inLevels.includes(e.level) ? { ...e, level: job.out } : e,
      );
      const entries = survivors.filter((e) => !p.fate[`e${e.id}`]?.drop);
      const ranges = cur.ranges
        .map((r) => (job.inLevels.includes(r.level) ? { ...r, level: job.out } : r))
        .filter((r) => !p.fate[`r${r.id}`]?.drop);

      const after = { ...cur, entries, ranges };
      const t = resolve(after, after.nextSeq);
      const shadow = Object.values(p.fate).filter((f) => f.tag === 'shadowed').length;
      const ttlDrop = Object.values(p.fate).filter((f) => f.tag === 'ttl').length;

      return {
        ...after,
        now: cur.now + 1,
        head:
          p.dropped === 0
            ? `${job.label}: rewrote every input key. Nothing was droppable.`
            : `${job.label}: dropped ${p.dropped} internal key${p.dropped === 1 ? '' : 's'}${
                p.pinned > 0 ? `, kept ${p.pinned} pinned by a snapshot` : ''
              }.`,
        body:
          `${shadow} shadowed version${shadow === 1 ? '' : 's'}, ${p.purged} tombstone${
            p.purged === 1 ? '' : 's'
          } purged, ${ttlDrop} expired by TTL. ` +
          (p.pinned > 0
            ? `${p.pinned} internal key${p.pinned === 1 ? ' was' : 's were'} rewritten unchanged because a live snapshot is entitled to ${
                p.pinned === 1 ? 'it' : 'them'
              } — this is the space a long-running transaction costs you. `
            : '') +
          (job.out === 3
            ? 'This is the bottommost level: a tombstone that reaches here with no snapshot below it finally stops existing.'
            : 'Tombstones for keys that still have data further down had to be rewritten — they only die at a level where nothing below overlaps them.') +
          (p.resurrected.length > 0
            ? ` SingleDelete on ${p.resurrected.join(', ')} cancelled the newest Put and exposed an older one: the row is back.`
            : ''),
        log: [
          ...cur.log,
          {
            n: cur.log.length + 1,
            op: job.label,
            dropped: p.dropped,
            purged: p.purged,
            pinned: p.pinned,
            live: t.live,
            internal: t.internal,
          },
        ],
        scan: null,
      };
    });

  const scan = () =>
    setS((cur) => {
      const rs = readAt === 'tip' ? cur.nextSeq : (cur.snaps.find((x) => String(x.id) === readAt)?.seq ?? cur.nextSeq);
      const r = resolve(cur, rs);
      const ratio = r.live > 0 ? r.tombs / r.live : r.tombs;
      return {
        ...cur,
        scan: { at: readAt === 'tip' ? 'tip' : `@${rs}`, internal: r.internal, tombs: r.tombs, obsolete: r.obsolete, live: r.live },
        head:
          `Range scan over [${keyName(0)}, ${keyName(KEYS - 1)}]: ${r.live} live row${r.live === 1 ? '' : 's'} returned, ` +
          `${r.internal} internal key${r.internal === 1 ? '' : 's'} stepped over.`,
        body:
          `${r.tombs} of them were deletion markers and ${r.obsolete} were obsolete versions — ` +
          `${ratio.toFixed(1)} dead entries per live row. The iterator has to merge and skip every one of them before it ` +
          `can hand you a row, in every SSTable that overlaps the range, on every scan, until a compaction removes them. ` +
          `This is the queue anti-pattern: the reads get slower even though the table looks empty.`,
      };
    });

  const preset = () =>
    setS(() => {
      const rng = makeRng(20180813);
      let st: S = { ...INITIAL, log: [] };
      const push = (kind: Kind, k: number) => {
        st = {
          ...st,
          entries: [...st.entries, { id: st.nextId, key: k, seq: st.nextSeq, kind, level: 0, born: st.now + 1 }],
          nextId: st.nextId + 1,
          nextSeq: st.nextSeq + 1,
          now: st.now + 1,
        };
      };
      for (let k = 0; k < KEYS; k++) push('put', k);
      st = { ...st, entries: st.entries.map((e) => ({ ...e, level: 1 })) };
      for (let k = 0; k < KEYS; k++) if (rng() < 0.75) push('del', k);
      return {
        ...st,
        head: 'A queue table: eight rows written, six of them consumed and deleted.',
        body:
          'Flush, then compact, and watch how far down the tree the tombstones have to travel before they are ' +
          'allowed to disappear. Then press Range scan and read the dead-entries-per-live-row number.',
      };
    });

  /* ---------------------------------------------------------------- geometry */

  const labelW = 104;
  const colW = 74;
  const svgW = Math.max(width, labelW + KEYS * colW + 12);
  const chipH = 20;
  const bandH = 16;

  const perLevel = LEVELS.map((_, l) => {
    let maxChips = 0;
    for (let k = 0; k < KEYS; k++) {
      const n = s.entries.filter((e) => e.level === l && e.key === k).length;
      if (n > maxChips) maxChips = n;
    }
    const rs = s.ranges.filter((r) => r.level === l);
    return { maxChips, ranges: rs };
  });

  const rowH = perLevel.map((p) => 26 + p.ranges.length * (bandH + 3) + Math.max(1, p.maxChips) * (chipH + 3) + 8);
  const rowY: number[] = [];
  let acc = 8;
  for (let l = 0; l < LEVELS.length; l++) {
    rowY.push(acc);
    acc += rowH[l] + 8;
  }
  const rulerY = acc + 10;
  const svgH = rulerY + 54;

  const colX = (k: number) => labelW + k * colW;
  const seqX = (q: number) => labelW + ((q - 0.5) / Math.max(1, s.nextSeq - 0.5)) * (svgW - labelW - 16);

  const ratio = view.live > 0 ? tipView.tombs / view.live : tipView.tombs;
  const oldest = s.snaps.length ? Math.min(...s.snaps.map((x) => x.seq)) : null;

  const allRows = [
    ...s.entries.map((e) => ({
      what: e.kind === 'put' ? 'Put' : e.kind === 'del' ? 'Delete' : 'SingleDelete',
      key: keyName(e.key),
      seq: e.seq,
      level: LEVELS[e.level].name,
      age: s.now - e.born,
      stripe: stripeOf(s.snaps, e.seq),
      fate: fateOf(e.level, `e${e.id}`),
      job: JOBS[jobForLevel(e.level)].label,
    })),
    ...s.ranges.map((r) => ({
      what: `DeleteRange [${keyName(r.lo)}, ${keyName(r.hi)})`,
      key: '—',
      seq: r.seq,
      level: LEVELS[r.level].name,
      age: s.now - r.born,
      stripe: stripeOf(s.snaps, r.seq),
      fate: fateOf(r.level, `r${r.id}`),
      job: JOBS[jobForLevel(r.level)].label,
    })),
  ].sort((a, b) => b.seq - a.seq);

  return (
    <VizPanel
      title="Tombstones, snapshot stripes and what a compaction may throw away"
      subtitle="Write and delete keys, hold a snapshot, then run compactions one at a time. Every chip is one internal key; the mark on its right edge is what the next compaction of its level will do with it."
      controls={
        <>
          <Choice
            label="Key"
            value={key}
            onChange={setKey}
            options={Array.from({ length: KEYS }, (_, i) => ({ value: String(i), label: keyName(i) }))}
          />
          <Button onClick={() => write('put')} primary>
            Put
          </Button>
          <Button onClick={() => write('del')} title="kTypeDeletion — the ordinary tombstone">
            Delete
          </Button>
          <Button onClick={() => write('sdel')} title="kTypeSingleDeletion — valid only if the key was written exactly once">
            SingleDelete
          </Button>
          <Button onClick={delRange} title="One record covering three keys, stored in the range_del block">
            DeleteRange ×3
          </Button>
          <Button onClick={snapshot}>Take snapshot</Button>
          <Button onClick={release} disabled={s.snaps.length === 0}>
            Release oldest
          </Button>
          <Slider
            label="TTL"
            min={0}
            max={16}
            value={ttl}
            onChange={setTtl}
            format={(n) => (n === 0 ? 'off' : `${n} ops`)}
          />
          <Button onClick={() => runJob('flush')} disabled={s.entries.every((e) => e.level !== 0) && s.ranges.every((r) => r.level !== 0)}>
            Flush
          </Button>
          <Button onClick={() => runJob('c01')}>Compact L0→L1</Button>
          <Button onClick={() => runJob('c12')}>Compact L1→L2</Button>
          <Choice
            label="Read at"
            value={readAt}
            onChange={setReadAt}
            options={[{ value: 'tip', label: 'tip (latest)' }, ...s.snaps.map((x) => ({ value: String(x.id), label: `snapshot @${x.seq}` }))]}
          />
          <Button onClick={scan}>Range scan</Button>
          <Button onClick={preset} title="Eight rows written, six consumed and deleted">
            Queue workload
          </Button>
          <Button onClick={() => setS(INITIAL)}>Reset</Button>
        </>
      }
      legend={
        <Legend
          items={[
            { label: 'P — newest value (what a read returns)', color: CHIP.live.fill },
            { label: 'p — obsolete version', color: CHIP.obsolete.fill },
            { label: 'D — Delete tombstone', color: CHIP.del.fill },
            { label: 'S — SingleDelete', color: CHIP.sdel.fill },
            { label: 'T — TTL-expired value', color: CHIP.expired.fill },
            { label: 'R — range tombstone fragment', color: 'var(--viz-5)' },
            { label: '✕ next compaction drops it', color: 'var(--viz-ink-2)', shape: 'dot' },
            { label: '◆ pinned by a snapshot', color: 'var(--viz-warning)', shape: 'dot' },
            { label: '▼ key still exists below', color: 'var(--viz-serious)', shape: 'dot' },
          ]}
        />
      }
      stats={
        <Stats
          items={[
            { label: 'Live rows', value: fmtNum(view.live), hint: `Keys whose newest visible version at ${readAt === 'tip' ? 'the tip' : `sequence ${readSeq}`} is a value` },
            { label: 'Internal keys', value: fmtNum(tipView.internal), hint: 'Every version and every marker still stored across all levels' },
            { label: 'Tombstones', value: fmtNum(tipView.tombs), hint: 'Deletion markers a scan must step over' },
            { label: 'Dead per live row', value: view.live > 0 ? ratio.toFixed(1) : '—', hint: 'The number that decides whether a scan of this key range is fast' },
            { label: 'Snapshots held', value: fmtNum(s.snaps.length), hint: 'rocksdb.num-snapshots' },
            { label: 'Oldest snapshot', value: oldest === null ? 'none' : `@${oldest}`, hint: 'rocksdb.oldest-snapshot-sequence — nothing at or below this can be dropped' },
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
                <th>Seq</th>
                <th>Internal key</th>
                <th>User key</th>
                <th>Level</th>
                <th>Age (ops)</th>
                <th>Snapshot stripe</th>
                <th>Next compaction</th>
                <th>Fate</th>
              </tr>
            </thead>
            <tbody>
              {allRows.length === 0 ? (
                <tr>
                  <td colSpan={8}>Nothing written yet.</td>
                </tr>
              ) : (
                allRows.map((r) => (
                  <tr key={`${r.what}-${r.seq}`}>
                    <td>{r.seq}</td>
                    <td>{r.what}</td>
                    <td>{r.key}</td>
                    <td>{r.level}</td>
                    <td>{r.age}</td>
                    <td>{r.stripe}</td>
                    <td>{r.job}</td>
                    <td>{r.fate ? `${r.fate.drop ? 'dropped' : 'kept'} — ${r.fate.tag}` : '—'}</td>
                  </tr>
                ))
              )}
            </tbody>
          </table>
          <table className="viz-table">
            <thead>
              <tr>
                <th>#</th>
                <th>Operation</th>
                <th>Internal keys dropped</th>
                <th>Tombstones removed</th>
                <th>Pinned by snapshot</th>
                <th>Live rows after</th>
                <th>Internal keys after</th>
              </tr>
            </thead>
            <tbody>
              {s.log.length === 0 ? (
                <tr>
                  <td colSpan={7}>No compactions run yet.</td>
                </tr>
              ) : (
                s.log.map((r) => (
                  <tr key={r.n}>
                    <td>{r.n}</td>
                    <td>{r.op}</td>
                    <td>{r.dropped}</td>
                    <td>{r.purged}</td>
                    <td>{r.pinned}</td>
                    <td>{r.live}</td>
                    <td>{r.internal}</td>
                  </tr>
                ))
              )}
            </tbody>
          </table>
          {s.scan ? (
            <table className="viz-table">
              <thead>
                <tr>
                  <th>Scan at</th>
                  <th>Internal keys stepped</th>
                  <th>Tombstones skipped</th>
                  <th>Obsolete versions skipped</th>
                  <th>Live rows returned</th>
                </tr>
              </thead>
              <tbody>
                <tr>
                  <td>{s.scan.at}</td>
                  <td>{s.scan.internal}</td>
                  <td>{s.scan.tombs}</td>
                  <td>{s.scan.obsolete}</td>
                  <td>{s.scan.live}</td>
                </tr>
              </tbody>
            </table>
          ) : null}
        </>
      }
    >
      <div ref={ref}>
        <TooltipHost>
          <svg width={svgW} height={svgH} role="img" aria-label="Versions of eight keys across memtable, L0, L1 and L2, with the fate of each under the next compaction">
            {LEVELS.map((lv, l) => (
              <g key={lv.name}>
                <rect
                  x={labelW}
                  y={rowY[l]}
                  width={svgW - labelW - 12}
                  height={rowH[l]}
                  rx={8}
                  fill="var(--viz-plane)"
                  stroke="var(--viz-border)"
                />
                <text x={0} y={rowY[l] + 20} fill="var(--viz-ink)" fontWeight={600}>
                  {lv.name}
                </text>
                <text x={0} y={rowY[l] + 36} fill="var(--viz-ink-muted)" fontSize={11}>
                  {lv.sub}
                </text>
                {l === 0
                  ? Array.from({ length: KEYS }, (_, k) => (
                      <text key={k} x={colX(k) + 6} y={rowY[0] - 12} fill="var(--viz-ink-2)" fontSize={11}>
                        {keyName(k)}
                      </text>
                    ))
                  : null}

                {perLevel[l].ranges.map((r, ri) => {
                  const f = fateOf(l, `r${r.id}`);
                  const x = colX(r.lo) + 4;
                  const w = (r.hi - r.lo) * colW - 10;
                  return (
                    <g key={r.id} {...tip(<><strong>DeleteRange [{keyName(r.lo)}, {keyName(r.hi)}) @{r.seq}</strong><br />{f?.why}</>)}>
                      <rect
                        x={x}
                        y={rowY[l] + 22 + ri * (bandH + 3)}
                        width={Math.max(24, w)}
                        height={bandH}
                        rx={4}
                        fill="var(--viz-5)"
                        opacity={f?.drop ? 0.35 : 0.85}
                        stroke={f?.drop ? 'var(--viz-critical)' : 'var(--viz-surface)'}
                        strokeDasharray={f?.drop ? '3 2' : undefined}
                      />
                      <text x={x + 6} y={rowY[l] + 22 + ri * (bandH + 3) + 12} fontSize={11} fill="var(--viz-surface)">
                        R{r.seq} [{keyName(r.lo)},{keyName(r.hi)}) {f ? FATE_GLYPH[f.tag] : ''}
                      </text>
                    </g>
                  );
                })}

                {Array.from({ length: KEYS }, (_, k) => {
                  const list = s.entries
                    .filter((e) => e.level === l && e.key === k)
                    .sort((a, b) => b.seq - a.seq);
                  const base = rowY[l] + 22 + perLevel[l].ranges.length * (bandH + 3);
                  return list.map((e, ci) => {
                    const f = fateOf(l, `e${e.id}`);
                    const top = tipView.winner[k];
                    const isTop = top !== null && top.ref === `e${e.id}`;
                    const expired = ttl > 0 && s.now - e.born >= ttl;
                    const ck: ChipKind =
                      e.kind === 'del' ? 'del' : e.kind === 'sdel' ? 'sdel' : expired ? 'expired' : isTop ? 'live' : 'obsolete';
                    const y = base + ci * (chipH + 3);
                    return (
                      <g
                        key={e.id}
                        {...tip(
                          <>
                            <strong>
                              ({keyName(k)}, {e.seq},{' '}
                              {e.kind === 'put' ? 'kTypeValue' : e.kind === 'del' ? 'kTypeDeletion' : 'kTypeSingleDeletion'})
                            </strong>
                            <br />
                            {LEVELS[l].name} · stripe {stripeOf(s.snaps, e.seq)} · written {s.now - e.born} ops ago
                            <br />
                            {JOBS[jobForLevel(l)].label}: {f?.why}
                          </>,
                        )}
                      >
                        <rect
                          x={colX(k) + 4}
                          y={y}
                          width={colW - 10}
                          height={chipH}
                          rx={5}
                          fill={CHIP[ck].fill}
                          opacity={f?.drop ? 0.4 : 1}
                          stroke={f?.drop ? 'var(--viz-critical)' : f?.tag === 'pinned' ? 'var(--viz-warning)' : 'var(--viz-surface)'}
                          strokeWidth={f?.drop || f?.tag === 'pinned' ? 2 : 1.5}
                          strokeDasharray={f?.drop ? '3 2' : undefined}
                        />
                        <text x={colX(k) + 10} y={y + 14} fontSize={11} fill="var(--viz-surface)" fontWeight={600}>
                          {CHIP[ck].glyph}
                          {e.seq}
                        </text>
                        <text x={colX(k) + colW - 12} y={y + 14} fontSize={11} textAnchor="end" fill="var(--viz-surface)">
                          {f ? FATE_GLYPH[f.tag] : ''}
                        </text>
                      </g>
                    );
                  });
                })}
              </g>
            ))}

            {/* sequence ruler: the snapshot stripes, drawn on the axis they actually live on */}
            <text x={0} y={rulerY + 14} fill="var(--viz-ink)" fontWeight={600}>
              Sequence
            </text>
            <text x={0} y={rulerY + 30} fill="var(--viz-ink-muted)" fontSize={11}>
              snapshot stripes
            </text>
            <line x1={labelW} y1={rulerY + 24} x2={svgW - 16} y2={rulerY + 24} stroke="var(--viz-axis)" />
            {s.snaps.map((sn) => (
              <g key={sn.id} {...tip(<>Snapshot @{sn.seq}: every version at or below sequence {sn.seq} must survive every compaction until this snapshot is released.</>)}>
                <line x1={seqX(sn.seq + 0.5)} y1={rulerY + 6} x2={seqX(sn.seq + 0.5)} y2={rulerY + 38} stroke="var(--viz-warning)" strokeWidth={2} />
                <text x={seqX(sn.seq + 0.5) + 4} y={rulerY + 14} fontSize={11} fill="var(--viz-warning)">
                  @{sn.seq}
                </text>
              </g>
            ))}
            {[...s.entries.map((e) => ({ q: e.seq, t: e.kind })), ...s.ranges.map((r) => ({ q: r.seq, t: 'rdel' as const }))].map((m) => (
              <circle
                key={`${m.t}-${m.q}`}
                cx={seqX(m.q)}
                cy={rulerY + 24}
                r={4}
                fill={m.t === 'put' ? 'var(--viz-clean)' : m.t === 'del' ? 'var(--viz-8)' : m.t === 'sdel' ? 'var(--viz-7)' : 'var(--viz-5)'}
              />
            ))}
            <line
              x1={seqX(readSeq)}
              y1={rulerY + 2}
              x2={seqX(readSeq)}
              y2={rulerY + 44}
              stroke="var(--viz-ink-2)"
              strokeWidth={1.5}
              strokeDasharray="4 3"
            />
            <text x={seqX(readSeq) + 4} y={rulerY + 50} fontSize={11} fill="var(--viz-ink-2)">
              reading at {readAt === 'tip' ? 'tip' : `@${readSeq}`}
            </text>
          </svg>
        </TooltipHost>
      </div>
    </VizPanel>
  );
}
