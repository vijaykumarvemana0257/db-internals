import { useMemo, useState, type ReactNode } from 'react';
import { VizPanel, Segmented, Choice, Slider, Button, Legend, Stats, Note, fmtNum } from './Viz';

/**
 * Building an index without blocking writers, in two engines.
 *
 * POSTGRESQL — CREATE INDEX CONCURRENTLY as DefineIndex() in indexcmds.c runs it (REL_18_STABLE):
 *  - txn 1 inserts the pg_index row with indisready = indisvalid = false and commits.
 *  - WaitForLockers(ShareLock) waits for every transaction holding a lock that conflicts with ShareLock
 *    (i.e. every writer of the table) at that moment. Then snapshot S1; index_build() indexes the tuples
 *    live in S1; index_set_state_flags(INDEX_CREATE_SET_READY); commit.
 *  - WaitForLockers again. Reference snapshot S2; validate_index() collects the index's TIDs, sorts them,
 *    and merge-joins a heap scan under S2 against them, inserting every visible tuple whose HOT-chain root
 *    TID is absent. Commit, then WaitForOlderSnapshots(S2.xmin) waits for every transaction whose
 *    advertised xmin is <= S2's xmin, then sets indisvalid.
 *
 * Model assumptions (labelled in the UI):
 *  - Time is in abstract ticks. At one tick: CIC's commits, then transactions' writes, then their commits,
 *    then CIC's snapshot / wait checks.
 *  - A writer's view of the table's index list is fixed when it first writes (takes RowExclusiveLock).
 *    That is the worst case the waits are designed for; real backends can notice sooner.
 *    Writers that first wrote before txn 1 committed do HOT updates that change the new column;
 *    writers that first wrote after indisready committed insert into the new index.
 *  - Every heap tuple lives on one page, so TIDs are (0, n). Updates change the indexed column.
 *  - xids are assigned at a transaction's first write. A snapshot's xmin is the oldest running xid.
 *  - "Skip a wait" is hypothetical — PostgreSQL never does it — and shows what the wait prevents.
 *
 * INNODB — online ADD INDEX (MySQL 8.4): the SQL layer upgrades the metadata lock to EXCLUSIVE for
 * prepare (row log allocated, read view assigned), downgrades to SHARED_UPGRADABLE for the build, and
 * upgrades to EXCLUSIVE again to commit. Concurrent DML is appended to the index's row log in blocks of
 * innodb_sort_buffer_size; row_log_apply() replays it after the build, taking the index X latch for the
 * last block. If the log reaches innodb_online_alter_log_max_size the index is marked corrupt and the
 * statement fails with ER_INNODB_ONLINE_LOG_TOO_BIG when the log is applied.
 * Model assumptions: 100 bytes per logged index record, a constant apply rate, one DML statement per
 * row change, and innodb_sort_buffer_size at its 1 MiB default.
 */

/* =============================================================== PostgreSQL */

export type OpKind = 'insert' | 'update' | 'delete';
export type Op = { t: number; kind: OpKind; row: string; key?: number };
export type Txn = {
  id: string;
  pid: number;
  label: string;
  start: number;
  end: number;
  ops: Op[];
  /** Holds one snapshot from this tick until it ends (REPEATABLE READ, or one long statement). */
  snapshotAt?: number;
  /** Works on a different table in the same database: never locks ours. */
  otherTable?: boolean;
};
export type Scenario = {
  id: string;
  label: string;
  rows: { row: string; key: number }[];
  txns: Txn[];
  long: { txn: string; min: number; max: number; def: number; label: string };
};
export type SkipWait = 'none' | 'w1' | 'w2' | 'w3';

export const CIC_START = 4;
export const BUILD_TICKS = 7;
export const VALIDATE_TICKS = 5;
const NEVER = Number.POSITIVE_INFINITY;

const PRE_ROWS = [10, 20, 30, 40, 50, 60].map((key, i) => ({ row: `r${i + 1}`, key }));

export const SCENARIOS: Scenario[] = [
  {
    id: 'busy',
    label: 'Busy table: writers, a HOT update, a long report',
    rows: PRE_ROWS,
    txns: [
      { id: 'T1', pid: 4101, label: 'writer', start: 1, end: 6, ops: [{ t: 2, kind: 'insert', row: 'r7', key: 70 }, { t: 3, kind: 'update', row: 'r3', key: 35 }] },
      { id: 'T5', pid: 4105, label: 'writer', start: 7, end: 8, ops: [{ t: 7, kind: 'insert', row: 'r12', key: 120 }] },
      { id: 'T2', pid: 4102, label: 'writer', start: 7, end: 11, ops: [{ t: 8, kind: 'insert', row: 'r8', key: 80 }, { t: 10, kind: 'update', row: 'r5', key: 55 }] },
      { id: 'R1', pid: 4106, label: 'report, other table', start: 10, end: 26, ops: [], snapshotAt: 10, otherTable: true },
      { id: 'T3', pid: 4103, label: 'writer', start: 11, end: 17, ops: [{ t: 12, kind: 'insert', row: 'r9', key: 90 }, { t: 13, kind: 'delete', row: 'r12' }, { t: 15, kind: 'insert', row: 'r10', key: 100 }] },
      { id: 'T4', pid: 4104, label: 'writer', start: 14, end: 19, ops: [{ t: 14, kind: 'insert', row: 'r11', key: 110 }, { t: 16, kind: 'delete', row: 'r2' }] },
    ],
    long: { txn: 'R1', min: 11, max: 40, def: 26, label: 'R1 (REPEATABLE READ report) commits at t =' },
  },
  {
    id: 'idle',
    label: 'A writer left idle in transaction',
    rows: PRE_ROWS,
    txns: [
      { id: 'T1', pid: 4201, label: 'writer, then idle', start: 1, end: 18, ops: [{ t: 2, kind: 'insert', row: 'r7', key: 70 }] },
      { id: 'Q1', pid: 4202, label: 'idle after a SELECT', start: 2, end: 40, ops: [] },
      { id: 'T2', pid: 4203, label: 'writer', start: 8, end: 10, ops: [{ t: 9, kind: 'insert', row: 'r8', key: 80 }] },
      { id: 'T3', pid: 4204, label: 'writer', start: 21, end: 24, ops: [{ t: 22, kind: 'insert', row: 'r9', key: 90 }] },
      { id: 'T4', pid: 4205, label: 'writer', start: 27, end: 29, ops: [{ t: 28, kind: 'insert', row: 'r10', key: 100 }] },
    ],
    long: { txn: 'T1', min: 3, max: 36, def: 18, label: 'T1 (idle writer) commits at t =' },
  },
];

export type Version = {
  n: number;
  row: string;
  key: number;
  by: string;
  createdAt: number;
  commitAt: number;
  deadAt: number;
  deadBy?: string;
  root: number;
  hot: boolean;
};
export type EntrySrc = 'build' | 'writer' | 'validate';
export type Entry = { key: number; root: number; src: EntrySrc; at: number; v: number };
export type Flag = { v: number; kind: 'missing' | 'wrong' | 'oldsnap'; txn?: string };
export type PhaseKind = 'step' | 'wait' | 'build' | 'validate';
export type Phase = { id: string; from: number; to: number; kind: PhaseKind; label: string; lockers: Txn[] };

export const visibleAt = (v: Version, s: number) => v.commitAt <= s && v.deadAt > s;

export function simulateCic(sc: Scenario, longEnd: number, skip: SkipWait = 'none', failAt: number | null = null) {
  const txns: Txn[] = sc.txns.map((t) => {
    if (t.id !== sc.long.txn) return { ...t };
    const lastOp = t.ops.reduce((m, o) => Math.max(m, o.t), t.snapshotAt ?? t.start);
    return { ...t, end: Math.max(longEnd, lastOp + 1) };
  });
  const firstWrite = new Map(txns.map((t) => [t.id, t.ops.length ? Math.min(...t.ops.map((o) => o.t)) : NEVER]));
  const fw = (id: string) => firstWrite.get(id) ?? NEVER;
  const writers = txns
    .filter((t) => fw(t.id) < NEVER)
    .sort((a, b) => fw(a.id) - fw(b.id) || txns.indexOf(a) - txns.indexOf(b));
  const xid = new Map(writers.map((t, i) => [t.id, 1001 + i]));
  const nextXidAt = (s: number) => 1001 + writers.filter((t) => fw(t.id) <= s).length;
  const xminAt = (s: number) => {
    let m = nextXidAt(s);
    for (const t of writers) if (fw(t.id) <= s && t.end > s) m = Math.min(m, xid.get(t.id) ?? m);
    return m;
  };
  const lockersAt = (w: number) => txns.filter((t) => !t.otherTable && fw(t.id) <= w && t.end > w);
  const waitEnd = (w: number, ls: Txn[]) => ls.reduce((m, t) => Math.max(m, t.end), w);

  const start = CIC_START;
  const createCommit = start + 1;
  const lockers1 = lockersAt(createCommit);
  const wait1End = skip === 'w1' ? createCommit : waitEnd(createCommit, lockers1);
  const S1 = wait1End;
  const readyCommit = S1 + BUILD_TICKS;
  const lockers2 = lockersAt(readyCommit);
  const wait2End = skip === 'w2' ? readyCommit : waitEnd(readyCommit, lockers2);
  const S2 = wait2End;
  const S2xmin = xminAt(S2);
  const valEnd = S2 + VALIDATE_TICKS;
  const lockers3 = txns.filter((t) => t.snapshotAt !== undefined && t.snapshotAt <= valEnd && t.end > valEnd && xminAt(t.snapshotAt) <= S2xmin);
  const wait3End = skip === 'w3' ? valEnd : waitEnd(valEnd, lockers3);
  const validCommit = wait3End + 1;

  const failed = failAt !== null && failAt < validCommit;
  const fa = failed ? (failAt as number) : NEVER;
  const exists = fa >= createCommit;
  const readyDone = fa >= readyCommit;
  const valDone = fa >= valEnd;

  // heap: every version of every row, in TID order
  const versions: Version[] = [];
  const tip = new Map<string, number>();
  for (const r of sc.rows) {
    const n = versions.length + 1;
    versions.push({ n, row: r.row, key: r.key, by: 'pre', createdAt: -1, commitAt: -1, deadAt: NEVER, root: n, hot: false });
    tip.set(r.row, n - 1);
  }
  const ops = txns
    .flatMap((t, ti) => t.ops.map((o, oi) => ({ ...o, txn: t, ti, oi })))
    .sort((a, b) => a.t - b.t || a.ti - b.ti || a.oi - b.oi);
  for (const o of ops) {
    const awareOfEntry = fw(o.txn.id) >= createCommit;
    if (o.kind === 'insert') {
      const n = versions.length + 1;
      versions.push({ n, row: o.row, key: o.key ?? 0, by: o.txn.id, createdAt: o.t, commitAt: o.txn.end, deadAt: NEVER, root: n, hot: false });
      tip.set(o.row, n - 1);
      continue;
    }
    const i = tip.get(o.row);
    if (i === undefined) continue;
    const old = versions[i];
    old.deadAt = o.txn.end;
    old.deadBy = o.txn.id;
    if (o.kind === 'update') {
      const n = versions.length + 1;
      const hot = !awareOfEntry; // it does not know the new index's column is indexed
      versions.push({ n, row: o.row, key: o.key ?? old.key, by: o.txn.id, createdAt: o.t, commitAt: o.txn.end, deadAt: NEVER, root: hot ? old.root : n, hot });
      tip.set(o.row, n - 1);
    } else {
      tip.delete(o.row);
    }
  }

  // index entries
  const entries: Entry[] = [];
  if (readyDone) for (const v of versions) if (visibleAt(v, S1)) entries.push({ key: v.key, root: v.root, src: 'build', at: readyCommit, v: v.n });
  const readyAware = (id: string) => readyDone && fw(id) >= readyCommit;
  for (const v of versions) if (v.by !== 'pre' && !v.hot && readyAware(v.by)) entries.push({ key: v.key, root: v.root, src: 'writer', at: v.createdAt, v: v.n });
  if (valDone) {
    const have = new Set(entries.filter((e) => e.at <= S2).map((e) => e.root));
    for (const v of versions) {
      if (visibleAt(v, S2) && !have.has(v.root)) {
        entries.push({ key: v.key, root: v.root, src: 'validate', at: valEnd, v: v.n });
        have.add(v.root);
      }
    }
  }

  const lastEnd = txns.reduce((m, t) => Math.max(m, t.end), 0);
  const horizon = Math.max(validCommit + 3, lastEnd + 2, 30);

  // correctness once the planner may use the index
  const flags: Flag[] = [];
  if (!failed) {
    const has = (v: Version) => entries.some((e) => e.root === v.root && e.key === v.key);
    for (const v of versions) {
      if (visibleAt(v, horizon) && !has(v)) flags.push({ v: v.n, kind: entries.some((e) => e.root === v.root) ? 'wrong' : 'missing' });
    }
    for (const t of txns) {
      if (t.snapshotAt === undefined || t.snapshotAt >= validCommit || t.end <= validCommit) continue;
      for (const v of versions) {
        if (visibleAt(v, t.snapshotAt) && !has(v) && !flags.some((f) => f.v === v.n)) flags.push({ v: v.n, kind: 'oldsnap', txn: t.id });
      }
    }
  }

  const phases: Phase[] = [
    { id: 'create', from: start, to: createCommit, kind: 'step', label: 'catalog row', lockers: [] },
    { id: 'wait1', from: createCommit, to: wait1End, kind: 'wait', label: 'wait 1', lockers: lockers1 },
    { id: 'build', from: S1, to: readyCommit, kind: 'build', label: 'build from S1', lockers: [] },
    { id: 'wait2', from: readyCommit, to: wait2End, kind: 'wait', label: 'wait 2', lockers: lockers2 },
    { id: 'validate', from: S2, to: valEnd, kind: 'validate', label: 'validate with S2', lockers: [] },
    { id: 'wait3', from: valEnd, to: wait3End, kind: 'wait', label: 'wait 3', lockers: lockers3 },
    { id: 'mark', from: wait3End, to: validCommit, kind: 'step', label: 'valid', lockers: [] },
  ];

  const events = new Set<number>([0, start, createCommit, wait1End, readyCommit, wait2End, valEnd, wait3End, validCommit, horizon]);
  for (const t of txns) {
    events.add(t.start);
    events.add(t.end);
    for (const o of t.ops) events.add(o.t);
  }
  if (failed) events.add(fa);

  return {
    txns, versions, entries, flags, phases, horizon,
    times: { start, createCommit, wait1End, S1, readyCommit, wait2End, S2, valEnd, wait3End, validCommit, S2xmin },
    lockers: { w1: lockers1, w2: lockers2, w3: lockers3 },
    xid, firstWrite: fw, xminAt,
    failed, failAt: failed ? fa : null, exists, readyDone, valDone, skip,
    events: [...events].filter((e) => e <= horizon && (!failed || e <= fa || e === horizon)).sort((a, b) => a - b),
  };
}
export type CicSim = ReturnType<typeof simulateCic>;

/** What pg_stat_progress_create_index and pg_index show at tick t. */
export function cicStateAt(sim: CicSim, t: number) {
  const T = sim.times;
  const failedNow = sim.failed && t >= (sim.failAt as number);
  const exists = t >= T.createCommit && (!sim.failed || (sim.failAt as number) >= T.createCommit);
  const indisready = t >= T.readyCommit && sim.readyDone;
  const indisvalid = !sim.failed && t >= T.validCommit;
  let phase: string | null = null;
  let wait: { id: string; lockers: Txn[]; done: number; current?: Txn } | null = null;
  if (failedNow) phase = null;
  else if (t < T.start) phase = null;
  else if (t < T.createCommit) phase = 'initializing';
  else if (t < T.wait1End) phase = 'waiting for writers before build';
  else if (t < T.readyCommit) {
    const k = t - T.S1;
    phase = k < 4 ? 'building index: scanning table' : k < 5 ? 'building index: sorting live tuples' : 'building index: loading tuples in tree';
  } else if (t < T.wait2End) phase = 'waiting for writers before validation';
  else if (t < T.valEnd) {
    const k = t - T.S2;
    phase = k < 1 ? 'index validation: scanning index' : k < 2 ? 'index validation: sorting tuples' : 'index validation: scanning table';
  } else if (t < T.validCommit) phase = 'waiting for old snapshots';

  if (!failedNow) {
    const w = t >= T.createCommit && t < T.wait1End ? 'w1' : t >= T.readyCommit && t < T.wait2End ? 'w2' : t >= T.valEnd && t < T.wait3End ? 'w3' : null;
    if (w) {
      const lockers = sim.lockers[w as 'w1' | 'w2' | 'w3'];
      wait = { id: w, lockers, done: lockers.filter((x) => x.end <= t).length, current: lockers.find((x) => x.end > t) };
    }
  }
  const live = sim.entries.filter((e) => e.at <= t);
  const count = (s: EntrySrc) => live.filter((e) => e.src === s).length;
  return {
    phase, wait, exists, indisready, indisvalid, failedNow,
    counts: { build: count('build'), writer: count('writer'), validate: count('validate') },
    flagsShown: !sim.failed && t >= T.validCommit ? sim.flags : [],
  };
}

export type Fate = 'build' | 'writer' | 'validate' | 'dead' | 'pending' | 'bad' | 'unborn';

/** How the new index covers one heap tuple version, as of tick t. */
export function fateAt(sim: CicSim, v: Version, t: number): { fate: Fate; entry?: Entry; flag?: Flag } {
  if (v.createdAt > t) return { fate: 'unborn' };
  const entry = sim.entries.find((e) => e.v === v.n);
  if (entry && entry.at <= t) return { fate: entry.src, entry };
  const flag = sim.flags.find((f) => f.v === v.n);
  if (flag && !sim.failed && t >= sim.times.validCommit) return { fate: 'bad', flag };
  if (!entry && v.deadAt <= t) return { fate: 'dead' };
  return { fate: 'pending', entry };
}

/* ==================================================================== InnoDB */

export type InnoOpen = 'none' | 'before' | 'during';
export type InnoInput = { dmlRate: number; buildSec: number; logMaxMiB: number; applyRate: number; open: InnoOpen; openSec: number };
export const INNO_REC_BYTES = 100;
export const INNO_SORT_BUF = 1 << 20;
const MiB = 1 << 20;
const COMMIT_SEC = 1;

export function simulateInnodb(inp: InnoInput) {
  const failBytes = Math.ceil((inp.logMaxMiB * MiB) / INNO_SORT_BUF) * INNO_SORT_BUF;
  const recPerBlock = INNO_SORT_BUF / INNO_REC_BYTES;
  const P = inp.open === 'before' ? inp.openSec : 0; // exclusive MDL for prepare granted
  const B = P + inp.buildSec; // scan + sort + load done, log apply starts
  const bps = inp.dmlRate * INNO_REC_BYTES;
  const L = inp.dmlRate * inp.buildSec; // records logged during the build
  let t1 = B; // reader is within one block of the writer: take index X latch
  let lastBlock = L;
  if (L > recPerBlock) {
    const gap = inp.applyRate - inp.dmlRate;
    t1 = gap > 0 ? B + (L - recPerBlock) / gap : NEVER;
    lastBlock = recPerBlock;
  }
  const hitAt = bps > 0 ? P + failBytes / bps : NEVER;
  const failed = hitAt <= t1 && hitAt < NEVER;
  const errorAt = failed ? Math.max(hitAt, B) : null;
  const latchEnd = failed ? NEVER : t1 + lastBlock / inp.applyRate;
  const openStart = inp.open === 'before' ? -Math.max(30, inp.openSec * 0.15) : inp.open === 'during' ? P + inp.buildSec / 2 : null;
  const openEnd = inp.open === 'before' ? inp.openSec : openStart !== null ? openStart + inp.openSec : null;
  const commitWaitEnd = failed ? NEVER : inp.open === 'during' && openEnd !== null && openEnd > latchEnd ? openEnd : latchEnd;
  const done = failed ? (errorAt as number) : commitWaitEnd + COMMIT_SEC;
  const horizon = Math.max(done, openEnd ?? 0, B) * 1.06 + 5;

  const tailAt = (t: number) => {
    if (t < P) return 0;
    if (failed) return t >= (errorAt as number) ? 0 : Math.min(failBytes, bps * (t - P));
    if (t >= latchEnd) return 0;
    return bps * (Math.min(t, t1) - P);
  };
  const appliedAt = (t: number) => {
    if (t < B) return 0;
    if (failed ? t >= (errorAt as number) : t >= latchEnd) return 0;
    const tail = tailAt(t);
    return Math.min(tail, inp.applyRate * INNO_REC_BYTES * (t - B));
  };
  /** Intervals in which new DML on the table cannot proceed. */
  const blocked: { from: number; to: number; why: string }[] = [];
  if (P > 0) blocked.push({ from: 0, to: P, why: 'prepare: waiting for exclusive MDL' });
  if (!failed) {
    if (latchEnd > t1) blocked.push({ from: t1, to: latchEnd, why: 'last log block: index X latch' });
    if (commitWaitEnd > latchEnd) blocked.push({ from: latchEnd, to: commitWaitEnd, why: 'commit: waiting for exclusive MDL' });
    blocked.push({ from: commitWaitEnd, to: done, why: 'commit: exclusive MDL held' });
  }
  const blockedUntil = (t: number) => blocked.reduce((s, b) => s + Math.max(0, Math.min(t, b.to) - b.from), 0);

  const stateAt = (t: number) => {
    if (failed && t >= (errorAt as number)) return { stmt: 'failed: log too big', online: 'ONLINE_INDEX_ABORTED', mdl: '—' };
    if (t < P) return { stmt: 'prepare: waiting for exclusive metadata lock', online: '—', mdl: 'SHARED_UPGRADABLE, waiting for EXCLUSIVE' };
    if (t < B) return { stmt: failed && t >= hitAt ? 'building; log overflowed, index marked corrupt' : 'building: scan, sort, load', online: 'ONLINE_INDEX_CREATION', mdl: 'SHARED_UPGRADABLE' };
    if (t < t1) return { stmt: failed && t >= hitAt ? 'applying log; log overflowed' : 'applying the row log', online: 'ONLINE_INDEX_CREATION', mdl: 'SHARED_UPGRADABLE' };
    if (t < latchEnd) return { stmt: 'applying the last block under the index X latch', online: 'ONLINE_INDEX_CREATION', mdl: 'SHARED_UPGRADABLE' };
    if (t < commitWaitEnd) return { stmt: 'commit: waiting for exclusive metadata lock', online: 'ONLINE_INDEX_COMPLETE', mdl: 'SHARED_UPGRADABLE, waiting for EXCLUSIVE' };
    if (t < done) return { stmt: 'commit: new table definition', online: 'ONLINE_INDEX_COMPLETE', mdl: 'EXCLUSIVE' };
    return { stmt: 'done', online: 'ONLINE_INDEX_COMPLETE', mdl: '—' };
  };

  return { failBytes, recPerBlock, P, B, t1, latchEnd, hitAt, failed, errorAt, openStart, openEnd, commitWaitEnd, done, horizon, L, tailAt, appliedAt, blocked, blockedUntil, stateAt, peakBytes: failed ? failBytes : bps * (Math.min(t1, NEVER) - P) };
}
export type InnoSim = ReturnType<typeof simulateInnodb>;

/* ======================================================================= UI */

const FATE_STYLE: Record<Fate, { stroke: string; label: string; dash?: string }> = {
  build: { stroke: 'var(--viz-1)', label: 'build (S1)' },
  writer: { stroke: 'var(--viz-2)', label: 'by its writer' },
  validate: { stroke: 'var(--viz-3)', label: 'validation (S2)' },
  dead: { stroke: 'var(--viz-stale)', label: 'dead, no entry' },
  pending: { stroke: 'var(--viz-axis)', label: 'no entry yet', dash: '4 3' },
  bad: { stroke: 'var(--viz-critical)', label: 'MISSING' },
  unborn: { stroke: 'var(--viz-grid)', label: 'not written yet', dash: '2 3' },
};
const PHASE_STROKE: Record<PhaseKind, string> = {
  step: 'var(--viz-ink-muted)',
  wait: 'var(--viz-warning)',
  build: 'var(--viz-1)',
  validate: 'var(--viz-3)',
};
const tid = (n: number) => `(0,${n})`;
/** InnoDB sizes its log settings in MiB; show them that way. */
const fmtMiB = (b: number) => (b <= 0 ? '0 MiB' : b < 10 * MiB ? `${(b / MiB).toFixed(1)} MiB` : `${fmtNum(b / MiB)} MiB`);

function badLabel(f: Flag | undefined) {
  if (!f) return 'MISSING';
  return f.kind === 'wrong' ? 'WRONG KEY' : f.kind === 'oldsnap' ? `${f.txn} misses it` : 'MISSING';
}

function pgNote(sim: CicSim, t: number, sc: Scenario) {
  const T = sim.times;
  const st = cicStateAt(sim, t);
  const names = (ls: Txn[]) => ls.map((x) => x.id).join(', ');
  if (sim.failed && t >= (sim.failAt as number)) {
    const fa = sim.failAt as number;
    if (fa < T.createCommit)
      return <><strong>Cancelled at t = {fa}, before the first commit.</strong> The pg_index row was never committed, so nothing is left behind.</>;
    if (fa < T.readyCommit)
      return <><strong>Cancelled at t = {fa}: an INVALID index is left behind with indisready = false.</strong> Writers do not insert into it, but it still counts in HOT-safety decisions, so updates to its column stop being HOT. <code>DROP INDEX</code> it, or <code>REINDEX INDEX CONCURRENTLY</code>.</>;
    return <><strong>Cancelled at t = {fa}: an INVALID index is left behind with indisready = true.</strong> The planner ignores it, yet every insert and non-HOT update after t = {T.readyCommit} still pays to maintain it ({st.counts.writer} entr{st.counts.writer === 1 ? 'y' : 'ies'} so far). Drop it or rebuild it with <code>REINDEX INDEX CONCURRENTLY</code>.</>;
  }
  if (t < T.start) return <><strong>t = {t}. CREATE INDEX CONCURRENTLY has not started.</strong> Pre-existing rows sit in the heap; press <em>Next event</em> to step.</>;
  if (t < T.createCommit) return <><strong>t = {t}. Transaction 1</strong> takes ShareUpdateExclusiveLock (writers keep going), inserts a pg_index row with indisready and indisvalid both false, and commits at t = {T.createCommit}.</>;
  if (st.wait?.id === 'w1')
    return <><strong>t = {t}. Waiting for writers before build.</strong> {st.wait.current ? `${st.wait.current.id} (pid ${st.wait.current.pid}) was already writing before the catalog row committed, so it may still HOT-update rows without knowing the new column is indexed.` : ''} The build cannot take its snapshot until {names(sim.lockers.w1)} end{sim.lockers.w1.length === 1 ? 's' : ''}. Writers that start now see the catalog row and are not waited for.</>;
  if (t < T.readyCommit)
    return <><strong>t = {t}. Building from snapshot S1 (taken at t = {T.S1}).</strong> The scan indexes exactly the tuples live in S1. Rows written from now on are invisible to it, and indisready is still false, so their writers do not insert them either — they are left for validation.</>;
  if (st.wait?.id === 'w2')
    return <><strong>t = {t}. indisready committed at t = {T.readyCommit}; waiting for writers before validation.</strong> {names(sim.lockers.w2)} opened the table for writing before that commit, so {sim.lockers.w2.length === 1 ? 'it is' : 'they are'} not inserting into the index. The reference snapshot S2 has to see everything {sim.lockers.w2.length === 1 ? 'it writes' : 'they write'}.</>;
  if (t < T.valEnd)
    return <><strong>t = {t}. Validating with reference snapshot S2 (t = {T.S2}).</strong> validate_index collects every TID already in the index, sorts them, and merge-joins a heap scan under S2 against the list. Tuples visible in S2 whose root TID is absent are inserted; anything committed after S2 is inserted by its own writer.</>;
  if (st.wait?.id === 'w3')
    return <><strong>t = {t}. Waiting for old snapshots.</strong> {names(sim.lockers.w3)} hold{sim.lockers.w3.length === 1 ? 's' : ''} a snapshot whose xmin is not newer than S2's ({T.S2xmin}) — {sim.lockers.w3.some((x) => x.otherTable) ? 'even though it never touches this table.' : 'so it could still see tuples the index skipped.'}</>;
  if (t < T.validCommit) return <><strong>t = {t}. Setting indisvalid</strong> and sending a relcache invalidation for the table so cached plans are rebuilt.</>;
  const flags = st.flagsShown;
  if (flags.length)
    return <><strong>t = {t}. The index is valid — and wrong.</strong> Skipping {sim.skip === 'w1' ? 'wait 1 let a writer that did not know about the index HOT-update a row and commit after S1; the build indexed the old key under the chain root, and validation skipped that root because its TID was already present' : sim.skip === 'w2' ? 'wait 2 let S2 be taken while a writer that never saw indisready was still inserting; its rows are in neither pass' : 'wait 3 let the index become valid while a transaction with an older snapshot was still running; it can see a row that was deleted before S2 and never indexed'}. {flags.length} tuple{flags.length === 1 ? '' : 's'} flagged below.</>;
  return <><strong>t = {t}. indisvalid = true: every tuple any running snapshot can see has an entry.</strong> {st.counts.build} came from the build, {st.counts.writer} from writers after indisready, {st.counts.validate} from validation. {sc.id === 'idle' ? 'Q1 sat idle in transaction the whole time and was never waited for: it holds no write lock and, in READ COMMITTED, no snapshot between statements.' : ''}</>;
}

function PgPanel({ engine }: { engine: ReactNode }) {
  const [scId, setScId] = useState('busy');
  const sc = SCENARIOS.find((s) => s.id === scId) ?? SCENARIOS[0];
  const [longEnd, setLongEnd] = useState(sc.long.def);
  const [skip, setSkip] = useState<SkipWait>('none');
  const [failAt, setFailAt] = useState<number | null>(null);
  const [cursor, setCursor] = useState<number | null>(null);

  const sim = useMemo(() => simulateCic(sc, longEnd, skip, failAt), [sc, longEnd, skip, failAt]);
  const t = Math.min(cursor ?? sim.horizon, sim.horizon);
  const st = cicStateAt(sim, t);
  const T = sim.times;

  const W = 680;
  const LEFT = 136;
  const RIGHT = 16;
  const xs = (v: number) => LEFT + (Math.max(0, Math.min(v, sim.horizon)) / sim.horizon) * (W - LEFT - RIGHT);
  const lanes = [...sim.txns].sort((a, b) => a.start - b.start || a.id.localeCompare(b.id));
  const yCic = 30;
  const yT0 = yCic + 58;
  const laneH = 26;
  const laneY = (i: number) => yT0 + i * laneH;
  const axisY = laneY(lanes.length) + 4;
  const heapTop = axisY + 40;
  const cellW = 98;
  const cellH = 50;
  const cols = Math.floor((W - 10) / (cellW + 8));
  const heapRows = Math.ceil(sim.versions.length / cols);
  const H = heapTop + 14 + heapRows * (cellH + 8) + 4;
  const endShown = sim.failed ? (sim.failAt as number) : sim.horizon;

  const reset = (patch: () => void) => {
    patch();
    setFailAt(null);
    setCursor(null);
  };
  const nextEvent = sim.events.find((e) => e > t);
  const vById = new Map(sim.versions.map((v) => [v.n, v]));

  const legend = (
    <Legend
      items={[
        { label: 'Indexed by the build scan (S1)', color: 'var(--viz-1)' },
        { label: 'Inserted by its writer (indisready)', color: 'var(--viz-2)' },
        { label: 'Back-filled by validation (S2)', color: 'var(--viz-3)' },
        { label: 'Dead before an entry was needed', color: 'var(--viz-stale)' },
        { label: 'Wait / transaction it waits for', color: 'var(--viz-warning)' },
        { label: 'Missing entry or failed build', color: 'var(--viz-critical)' },
        { label: 'Valid', color: 'var(--viz-good)' },
      ]}
    />
  );
  return (
    <VizPanel
      title="CREATE INDEX CONCURRENTLY, tick by tick"
      subtitle="Four transactions, two table scans, three waits. Step through time and watch each tuple get its index entry from the build, from its own writer, or from validation — and which open transaction each wait is stuck on."
      controls={engine}
      legend={
        <>
          {legend}
          <p style={{ fontSize: '0.75rem', color: 'var(--viz-ink-2)', margin: '0.2rem 0 0' }}>
            ● insert &nbsp; ■ update of the indexed column &nbsp; ✕ delete &nbsp; ◆ snapshot taken &nbsp; | commit. Model: ticks, one heap page; a writer&rsquo;s view of the index flags is fixed at its first write (the worst case).
          </p>
        </>
      }
      stats={
        <Stats
        items={[
          { label: 'Progress phase', hint: 'pg_stat_progress_create_index.phase', value: st.phase ?? (st.failedNow ? 'ERROR' : t >= T.validCommit ? '(finished)' : '(not started)') },
          { label: 'pg_index', value: st.exists ? `indisready ${st.indisready ? 't' : 'f'} · indisvalid ${st.indisvalid ? 't' : 'f'}` : 'no row' },
          {
            label: 'Waiting on',
            value: st.wait && st.wait.lockers.length ? `${st.wait.current ? `${st.wait.current.id} (pid ${st.wait.current.pid})` : '—'} · ${st.wait.done}/${st.wait.lockers.length} done` : '—',
            hint: 'lockers_done / lockers_total and current_locker_pid in pg_stat_progress_create_index',
          },
          { label: 'Entries: build · writers · validation', value: `${st.counts.build} · ${st.counts.writer} · ${st.counts.validate}` },
          { label: 'Tuples missing or wrong', value: t >= T.validCommit && !sim.failed ? fmtNum(st.flagsShown.length) : '—', hint: 'Checked once indisvalid is set: every tuple a running snapshot can see needs an entry with its key and its HOT-chain root TID.' },
        ]}
      />
      }
      note={<Note>{pgNote(sim, t, sc)}</Note>}
      table={
        <table className="viz-table">
          <thead>
            <tr>
              <th>TID</th>
              <th>Row</th>
              <th>Key</th>
              <th>Written by (commit t)</th>
              <th>Dead at t</th>
              <th>Index entry</th>
            </tr>
          </thead>
          <tbody>
            {sim.versions.map((v) => {
              const e = sim.entries.find((x) => x.v === v.n);
              const fl = sim.flags.find((x) => x.v === v.n);
              return (
                <tr key={v.n}>
                  <td>{tid(v.n)}{v.hot ? ` (heap-only, root ${tid(v.root)})` : ''}</td>
                  <td>{v.row}</td>
                  <td>{v.key}</td>
                  <td>{v.by === 'pre' ? 'pre-existing' : `${v.by} (${v.commitAt})`}</td>
                  <td>{v.deadAt === NEVER ? '—' : `${v.deadAt} (${v.deadBy})`}</td>
                  <td>
                    {e ? `${e.src === 'build' ? 'build' : e.src === 'writer' ? 'writer' : 'validation'} at t = ${e.at}, key ${e.key} → ${tid(e.root)}` : 'none'}
                    {fl && !sim.failed ? ` — ${badLabel(fl)}` : ''}
                    {!e && vById.get(v.n) && sim.entries.some((x) => x.root === v.root && x.v !== v.n) ? ` (root has key ${sim.entries.find((x) => x.root === v.root)?.key})` : ''}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      }
    >
      <div className="viz-controls">
        <Choice
          label="Scenario"
          value={scId}
          onChange={(v) => reset(() => {
            setScId(v);
            setLongEnd((SCENARIOS.find((s) => s.id === v) ?? SCENARIOS[0]).long.def);
          })}
          options={SCENARIOS.map((s) => ({ value: s.id, label: s.label }))}
        />
        <Slider label={sc.long.label} min={sc.long.min} max={sc.long.max} value={longEnd} onChange={(v) => reset(() => setLongEnd(v))} />
        <Choice
          label="Skip a wait (hypothetical)"
          value={skip}
          onChange={(v) => reset(() => setSkip(v))}
          options={[
            { value: 'none', label: 'No — PostgreSQL always waits' },
            { value: 'w1', label: 'Skip wait 1 (writers before build)' },
            { value: 'w2', label: 'Skip wait 2 (writers before validation)' },
            { value: 'w3', label: 'Skip wait 3 (old snapshots)' },
          ]}
        />
      </div>
      <div className="viz-controls">
        <Slider label="Time t" min={0} max={sim.horizon} value={t} onChange={(v) => setCursor(v)} />
        <Button onClick={() => setCursor(0)}>Start</Button>
        <Button primary onClick={() => setCursor(nextEvent ?? sim.horizon)} disabled={nextEvent === undefined}>
          Next event
        </Button>
        {sim.failed ? (
          <Button onClick={() => setFailAt(null)}>Undo the failure</Button>
        ) : (
          <Button onClick={() => setFailAt(t)} disabled={t < T.start || t >= T.validCommit} title="Error or pg_cancel_backend() at the current tick">
            Fail at t = {t}
          </Button>
        )}
      </div>

      <svg viewBox={`0 0 ${W} ${H}`} width={W} height={H} style={{ minWidth: 600 }} role="img" aria-label={`CREATE INDEX CONCURRENTLY timeline at t = ${t}: ${st.phase ?? (sim.failed && t >= (sim.failAt as number) ? 'failed, index invalid' : t >= T.validCommit ? 'index valid' : 'not started')}`}>
        {/* snapshots */}
        {[
          { at: T.S1, label: 'S1', show: T.S1 <= endShown },
          { at: T.S2, label: 'S2', show: T.S2 <= endShown && (!sim.failed || (sim.failAt as number) >= T.S2) },
        ].map((s) =>
          s.show ? (
            <g key={s.label}>
              <line x1={xs(s.at)} x2={xs(s.at)} y1={yCic - 12} y2={axisY} stroke="var(--viz-ink-2)" strokeDasharray="3 3" />
              <text x={xs(s.at)} y={yCic - 16} textAnchor="middle" fontSize={11} fontWeight={600} fill="var(--viz-ink)">
                {s.label}
              </text>
            </g>
          ) : null,
        )}

        {/* CIC lane */}
        <text x={8} y={yCic + 15} fontSize={11} fontWeight={600} fill="var(--viz-ink)">
          CREATE INDEX CONC.
        </text>
        {sim.phases.map((p) => {
          const to = Math.min(p.to, endShown);
          if (p.from >= to && !(p.kind === 'wait' && p.lockers.length && sim.skip === p.id.replace('wait', 'w'))) return null;
          const w = Math.max(0, xs(to) - xs(p.from));
          if (w <= 0) return null;
          return (
            <g key={p.id}>
              <rect x={xs(p.from)} y={yCic} width={w} height={22} rx={3} fill="var(--viz-surface)" stroke={PHASE_STROKE[p.kind]} strokeWidth={2.5} />
              {(() => {
                const text = w > p.label.length * 5.6 + 6 ? p.label : p.kind === 'build' && w > 36 ? 'build' : p.kind === 'validate' && w > 51 ? 'validate' : null;
                return text ? (
                  <text x={xs(p.from) + w / 2} y={yCic + 15} textAnchor="middle" fontSize={10} fill="var(--viz-ink)">
                    {text}
                  </text>
                ) : null;
              })()}
            </g>
          );
        })}
        {!sim.failed ? (
          <g>
            <rect x={xs(T.validCommit)} y={yCic} width={Math.max(0, xs(sim.horizon) - xs(T.validCommit))} height={22} rx={3} fill="var(--viz-surface)" stroke="var(--viz-good)" strokeWidth={2.5} />
            {xs(sim.horizon) - xs(T.validCommit) > 36 ? (
              <text x={xs(T.validCommit) + 6} y={yCic + 15} fontSize={10} fill="var(--viz-ink)">
                {xs(sim.horizon) - xs(T.validCommit) > 60 ? 'indisvalid' : 'valid'}
              </text>
            ) : null}
          </g>
        ) : (
          <g>
            <rect x={xs(sim.failAt as number)} y={yCic} width={Math.max(0, xs(sim.horizon) - xs(sim.failAt as number))} height={22} rx={3} fill="var(--viz-surface)" stroke="var(--viz-critical)" strokeWidth={2.5} strokeDasharray="5 3" />
            <text x={xs(sim.failAt as number) + 6} y={yCic + 15} fontSize={10} fill="var(--viz-ink)">
              {sim.exists ? 'ERROR: INVALID index left' : 'ERROR: rolled back'}
            </text>
          </g>
        )}
        {/* writers maintain the index once indisready is committed */}
        {sim.readyDone ? (
          <g>
            <rect x={xs(T.readyCommit)} y={yCic + 27} width={Math.max(0, xs(sim.horizon) - xs(T.readyCommit))} height={5} rx={2} fill="var(--viz-2)" />
            <text x={xs(T.readyCommit) + 2} y={yCic + 45} fontSize={10} fill="var(--viz-ink-2)">
              indisready: writers insert
            </text>
          </g>
        ) : null}

        {/* transactions */}
        {lanes.map((tx, i) => {
          const y = laneY(i);
          const blockingNow = st.wait && st.wait.lockers.includes(tx) && tx.end > t;
          const lockerOf = (['w1', 'w2', 'w3'] as const).filter((w) => sim.lockers[w].includes(tx));
          return (
            <g key={tx.id}>
              <text x={8} y={y + 16} fontSize={10.5} fill="var(--viz-ink)">
                <tspan fontWeight={600}>{tx.id}</tspan> {tx.label}
              </text>
              <line x1={LEFT} x2={W - RIGHT} y1={y + 12} y2={y + 12} stroke="var(--viz-grid)" />
              <rect x={xs(tx.start)} y={y + 5} width={Math.max(3, xs(tx.end) - xs(tx.start))} height={14} rx={4} fill="var(--viz-plane)" stroke={blockingNow ? 'var(--viz-warning)' : 'var(--viz-ink-muted)'} strokeWidth={blockingNow ? 3 : 1} />
              <line x1={xs(tx.end)} x2={xs(tx.end)} y1={y + 2} y2={y + 22} stroke="var(--viz-ink)" strokeWidth={1.5} />
              {tx.snapshotAt !== undefined ? (
                <path d={`M ${xs(tx.snapshotAt)} ${y + 6} l 6 6 l -6 6 l -6 -6 z`} fill="var(--viz-ink-2)" />
              ) : null}
              {lockerOf.map((w) => {
                const skipped = sim.skip === w;
                const ph = sim.phases.find((p) => p.id === w.replace('w', 'wait'));
                if (!ph || (sim.failed && ph.from > (sim.failAt as number))) return null;
                if (!skipped && sim.failed && tx.end > (sim.failAt as number)) return null;
                const x = skipped ? xs(ph.from) : xs(tx.end);
                return (
                  <line key={w} x1={x} x2={x} y1={yCic + 22} y2={y + 5} stroke={skipped ? 'var(--viz-critical)' : 'var(--viz-warning)'} strokeWidth={1.5} strokeDasharray="4 3" />
                );
              })}
              {tx.ops.map((o, oi) => {
                const future = o.t > t;
                const cx = xs(o.t);
                const cy = y + 12;
                if (o.kind === 'delete') {
                  return (
                    <path key={oi} d={`M ${cx - 5} ${cy - 5} L ${cx + 5} ${cy + 5} M ${cx + 5} ${cy - 5} L ${cx - 5} ${cy + 5}`} stroke="var(--viz-ink)" strokeWidth={2} opacity={future ? 0.3 : 1} />
                  );
                }
                const v = sim.versions.find((vv) => vv.by === tx.id && vv.createdAt === o.t && vv.row === o.row);
                const f = v ? fateAt(sim, v, t) : { fate: 'pending' as Fate };
                const fill = future || f.fate === 'pending' || f.fate === 'unborn' ? 'var(--viz-surface)' : FATE_STYLE[f.fate].stroke;
                const stroke = f.fate === 'pending' || future ? 'var(--viz-ink-muted)' : 'var(--viz-surface)';
                return o.kind === 'insert' ? (
                  <circle key={oi} cx={cx} cy={cy} r={6} fill={fill} stroke={stroke} strokeWidth={1.5} opacity={future ? 0.4 : 1} />
                ) : (
                  <rect key={oi} x={cx - 6} y={cy - 6} width={12} height={12} fill={fill} stroke={stroke} strokeWidth={1.5} opacity={future ? 0.4 : 1} />
                );
              })}
            </g>
          );
        })}

        {/* axis + cursor */}
        <line x1={LEFT} x2={W - RIGHT} y1={axisY} y2={axisY} stroke="var(--viz-axis)" />
        {Array.from({ length: Math.floor(sim.horizon / 5) + 1 }, (_, k) => k * 5).map((v) => (
          <g key={v}>
            <line x1={xs(v)} x2={xs(v)} y1={axisY} y2={axisY + 4} stroke="var(--viz-axis)" />
            <text x={xs(v)} y={axisY + 15} textAnchor="middle" fontSize={10} fill="var(--viz-ink-2)">
              {v}
            </text>
          </g>
        ))}
        <text x={8} y={axisY + 15} fontSize={10} fill="var(--viz-ink-2)">
          time (ticks)
        </text>
        <line x1={xs(t)} x2={xs(t)} y1={yCic - 6} y2={axisY} stroke="var(--viz-ink)" strokeWidth={1.5} />

        {/* heap vs index */}
        <text x={8} y={heapTop} fontSize={11} fontWeight={600} fill="var(--viz-ink)">
          Heap page 0 — each tuple version, and how the new index covers it at t = {t}
        </text>
        {sim.versions.map((v, i) => {
          const x = 8 + (i % cols) * (cellW + 8);
          const y = heapTop + 10 + Math.floor(i / cols) * (cellH + 8);
          const f = fateAt(sim, v, t);
          const style = FATE_STYLE[f.fate];
          const who = v.by === 'pre' ? 'pre-existing' : `${v.by}${v.hot ? `, HOT → ${tid(v.root)}` : ''}`;
          const failedUnused = sim.failed && t >= (sim.failAt as number) && f.fate === 'pending';
          const label = f.fate === 'bad' ? badLabel(f.flag) : failedUnused ? 'no entry (failed)' : f.fate === 'dead' && v.hot === false && sim.versions.some((w) => w.hot && w.root === v.root && w.n !== v.n) ? 'dead, HOT root' : style.label;
          return (
            <g key={v.n} opacity={f.fate === 'unborn' ? 0.45 : 1}>
              <rect x={x} y={y} width={cellW} height={cellH} rx={5} fill="var(--viz-surface)" stroke={style.stroke} strokeWidth={f.fate === 'bad' ? 3 : 2} strokeDasharray={style.dash} />
              <text x={x + 6} y={y + 15} fontSize={10.5} fontWeight={600} fill="var(--viz-ink)">
                {tid(v.n)} key {v.key}
              </text>
              <text x={x + 6} y={y + 29} fontSize={9.5} fill="var(--viz-ink-2)">
                {who}
              </text>
              <text x={x + 6} y={y + 43} fontSize={9.5} fill="var(--viz-ink)">
                {label}
              </text>
            </g>
          );
        })}
      </svg>

    </VizPanel>
  );
}

function InnoPanel({ engine }: { engine: ReactNode }) {
  const [dmlRate, setDmlRate] = useState(1500);
  const [buildSec, setBuildSec] = useState(600);
  const [logMax, setLogMax] = useState('128');
  const [applyRate, setApplyRate] = useState(20000);
  const [open, setOpen] = useState<InnoOpen>('during');
  const [openSec, setOpenSec] = useState(450);
  const [cursor, setCursor] = useState<number | null>(null);

  const sim = useMemo(() => simulateInnodb({ dmlRate, buildSec, logMaxMiB: Number(logMax), applyRate, open, openSec }), [dmlRate, buildSec, logMax, applyRate, open, openSec]);
  const tStep = Math.max(1, Math.round(sim.horizon / 200));
  const horizon = Math.ceil(sim.horizon / tStep) * tStep;
  const t = Math.min(cursor ?? horizon, horizon);
  const st = sim.stateAt(t);
  const touch = (f: () => void) => {
    f();
    setCursor(null);
  };

  const W = 680;
  const LEFT = 136;
  const RIGHT = 16;
  const xs = (v: number) => LEFT + (Math.max(0, Math.min(v, horizon)) / horizon) * (W - LEFT - RIGHT);
  const yAlter = 26;
  const yOpen = yAlter + 38;
  const yDml = yOpen + 30;
  const chartTop = yDml + 52;
  const chartH = 150;
  const chartBottom = chartTop + chartH;
  const H = chartBottom + 30;
  const yMax = Math.max(sim.failBytes, sim.peakBytes) * 1.12;
  const ys = (b: number) => chartBottom - (b / yMax) * chartH;
  const fin = (v: number) => Math.min(v, horizon);

  const segs: { from: number; to: number; stroke: string; label: string }[] = [];
  if (sim.P > 0) segs.push({ from: 0, to: sim.P, stroke: 'var(--viz-warning)', label: 'wait for MDL X' });
  segs.push({ from: sim.P, to: fin(sim.B), stroke: 'var(--viz-1)', label: 'scan · sort · load (read view)' });
  if (sim.failed) {
    if ((sim.errorAt as number) > sim.B) segs.push({ from: sim.B, to: sim.errorAt as number, stroke: 'var(--viz-3)', label: 'apply log' });
  } else {
    segs.push({ from: sim.B, to: sim.t1, stroke: 'var(--viz-3)', label: 'apply row log' });
    segs.push({ from: sim.t1, to: sim.latchEnd, stroke: 'var(--viz-warning)', label: 'last block' });
    if (sim.commitWaitEnd > sim.latchEnd) segs.push({ from: sim.latchEnd, to: sim.commitWaitEnd, stroke: 'var(--viz-warning)', label: 'wait for MDL X' });
    segs.push({ from: sim.commitWaitEnd, to: sim.done, stroke: 'var(--viz-ink-muted)', label: 'commit' });
  }

  const N = 240;
  const pts = Array.from({ length: N + 1 }, (_, i) => (i / N) * horizon);
  const extra = [sim.P, sim.B, sim.t1, sim.latchEnd, sim.hitAt, sim.errorAt ?? NEVER].filter((v) => v > 0 && v < horizon);
  const times = [...new Set([...pts, ...extra, ...extra.map((v) => v - 1e-6)])].sort((a, b) => a - b);
  const pathOf = (f: (x: number) => number) => times.map((x, i) => `${i ? 'L' : 'M'} ${xs(x).toFixed(1)} ${ys(f(x)).toFixed(1)}`).join(' ');

  const dmlSegs: { from: number; to: number; kind: 'log' | 'direct' | 'blocked' | 'plain' }[] = [];
  {
    const cuts = [...new Set([0, horizon, sim.P, sim.B, sim.t1, sim.latchEnd, sim.commitWaitEnd, sim.done, sim.hitAt, sim.errorAt ?? NEVER].filter((v) => v >= 0 && v <= horizon))].sort((a, b) => a - b);
    for (let i = 0; i < cuts.length - 1; i++) {
      const mid = (cuts[i] + cuts[i + 1]) / 2;
      const blocked = sim.blocked.some((b) => mid >= b.from && mid < b.to);
      const logging = !blocked && mid >= sim.P && (sim.failed ? mid < sim.hitAt : mid < sim.t1);
      const afterFailure = sim.failed && mid >= sim.hitAt;
      dmlSegs.push({ from: cuts[i], to: cuts[i + 1], kind: blocked ? 'blocked' : logging ? 'log' : afterFailure ? 'plain' : 'direct' });
    }
  }
  const stalled = sim.blockedUntil(t);
  const currentBlock = sim.blocked.find((b) => t >= b.from && t < b.to);
  const fmtS = (s: number) => (s >= 100 ? `${fmtNum(s)} s` : `${fmtNum(s, 1)} s`);

  return (
    <VizPanel
      title="InnoDB online ADD INDEX: the row log"
      subtitle="InnoDB builds from a read view while concurrent changes are appended to a row log, replays the log, and takes an exclusive metadata lock at both ends. Push DML up or shrink the log limit to make it fail; leave a transaction open to see the lock queue."
      controls={engine}
      legend={
        <>
        <Legend
        items={[
          { label: 'Build from the read view', color: 'var(--viz-1)' },
          { label: 'DML appended to the row log', color: 'var(--viz-7)' },
          { label: 'Row log applied (dashed line)', color: 'var(--viz-3)' },
          { label: 'DML written straight into the index', color: 'var(--viz-2)' },
          { label: 'Waiting on a metadata lock or latch', color: 'var(--viz-warning)' },
          { label: 'Log limit / failure', color: 'var(--viz-critical)' },
          ...(sim.failed ? [{ label: 'DML after the log overflowed (not logged)', color: 'var(--viz-stale)' }] : []),
        ]}
      />
      <p style={{ fontSize: '0.75rem', color: 'var(--viz-ink-2)', margin: '0.2rem 0 0' }}>
        Model: {INNO_REC_BYTES} bytes per logged index record, innodb_sort_buffer_size = 1 MiB (one log block), a constant apply rate, one statement per row change.
      </p>
        </>
      }
      stats={
        <Stats
        items={[
          { label: 'Statement', value: st.stmt },
          { label: 'online_status', value: st.online.replace('ONLINE_INDEX_', ''), hint: 'dict_index_t online_status: ONLINE_INDEX_CREATION, _COMPLETE or _ABORTED' },
          { label: 'ALTER’s metadata lock', value: st.mdl },
          { label: 'Row log written', value: `${fmtMiB(sim.tailAt(t))} of ${fmtMiB(sim.failBytes)}`, hint: 'The writer offset in the temporary log file; it only resets when the apply catches up.' },
          { label: 'DML stalled so far', value: fmtS(stalled) },
        ]}
      />
      }
      note={
        <Note>
        {sim.failed ? (
          <>
            <strong>The row log reaches {fmtMiB(sim.failBytes)} at t = {fmtNum(sim.hitAt)} s{sim.hitAt < sim.B ? ', during the build' : ', while the apply is still chasing the writers'}.</strong> InnoDB marks the half-built index corrupt and stops logging; the statement reports ER_INNODB_ONLINE_LOG_TOO_BIG at t = {fmtNum(sim.errorAt as number)} s{sim.hitAt < sim.B ? ', when the build reaches the log-apply step' : ''}, and the index is dropped. Lower the DML rate, shorten the build, or raise innodb_online_alter_log_max_size.
          </>
        ) : currentBlock ? (
          <>
            <strong>t = {fmtNum(t)} s: DML on the table is waiting — {currentBlock.why}.</strong>{' '}
            {currentBlock.why.includes('waiting for exclusive MDL')
              ? `The open transaction still holds a shared metadata lock, and the ALTER's pending exclusive request queues every new statement behind it: about ${fmtNum(dmlRate * (t - currentBlock.from))} row changes are stuck so far.`
              : currentBlock.why.includes('latch')
                ? `Writers cannot append, so the apply can finish the last ${fmtNum(sim.recPerBlock, 0)} records or fewer and flip the index to ONLINE_INDEX_COMPLETE.`
                : 'The exclusive lock is held only long enough to commit the new definition.'}
          </>
        ) : (
          <>
            <strong>t = {fmtNum(t)} s: {st.stmt}.</strong> The build scans a read view taken at t = {fmtNum(sim.P)} s while {fmtNum(dmlRate)} index changes/s go to the row log; it holds {fmtMiB(dmlRate * buildSec * INNO_REC_BYTES)} when the build ends at {fmtNum(sim.B)} s. The apply catches up at {fmtNum(sim.t1)} s, and in total DML stalls for {fmtS(sim.blockedUntil(horizon))}{open !== 'none' && sim.blocked.some((b) => b.why.includes('waiting')) ? ', almost all of it queued behind the open transaction' : ''}.
          </>
        )}
      </Note>
      }
      table={
        <table className="viz-table">
          <thead>
            <tr>
              <th>Step</th>
              <th>Starts</th>
              <th>Ends</th>
            </tr>
          </thead>
          <tbody>
            <tr><td>Prepare: exclusive MDL, row log allocated, read view</td><td>0 s</td><td>{fmtNum(sim.P)} s</td></tr>
            <tr><td>Scan, sort, load (row log grows {fmtMiB(dmlRate * INNO_REC_BYTES)}/s)</td><td>{fmtNum(sim.P)} s</td><td>{fmtNum(sim.B)} s</td></tr>
            {sim.failed ? (
              <tr><td>Log reaches the limit / statement fails</td><td>{fmtNum(sim.hitAt)} s</td><td>{fmtNum(sim.errorAt as number)} s</td></tr>
            ) : (
              <>
                <tr><td>Apply row log without the index latch ({fmtNum(sim.L)} records logged during the build)</td><td>{fmtNum(sim.B)} s</td><td>{fmtNum(sim.t1, 1)} s</td></tr>
                <tr><td>Last block under index X latch</td><td>{fmtNum(sim.t1, 1)} s</td><td>{fmtNum(sim.latchEnd, 1)} s</td></tr>
                <tr><td>Commit: wait for exclusive MDL</td><td>{fmtNum(sim.latchEnd, 1)} s</td><td>{fmtNum(sim.commitWaitEnd, 1)} s</td></tr>
                <tr><td>Commit the new definition</td><td>{fmtNum(sim.commitWaitEnd, 1)} s</td><td>{fmtNum(sim.done, 1)} s</td></tr>
              </>
            )}
            <tr><td>Peak row log size</td><td colSpan={2}>{fmtMiB(sim.peakBytes)} (limit {fmtMiB(sim.failBytes)})</td></tr>
          </tbody>
        </table>
      }
    >
      <div className="viz-controls">
        <Slider label="Concurrent DML (index changes/s)" min={0} max={6000} step={100} value={dmlRate} onChange={(v) => touch(() => setDmlRate(v))} format={fmtNum} />
        <Slider label="Scan + sort + load time" min={60} max={1800} step={30} value={buildSec} onChange={(v) => touch(() => setBuildSec(v))} format={(v) => `${v} s`} />
        <Choice
          label="innodb_online_alter_log_max_size"
          value={logMax}
          onChange={(v) => touch(() => setLogMax(v))}
          options={[
            { value: '64', label: '64 MiB' },
            { value: '128', label: '128 MiB (default)' },
            { value: '256', label: '256 MiB' },
            { value: '512', label: '512 MiB' },
            { value: '1024', label: '1 GiB' },
          ]}
        />
        <Slider label="Log apply rate (model, records/s)" min={2000} max={50000} step={1000} value={applyRate} onChange={(v) => touch(() => setApplyRate(v))} format={fmtNum} />
      </div>
      <div className="viz-controls">
        <Choice
          label="An open transaction that read the table"
          value={open}
          onChange={(v) => touch(() => setOpen(v))}
          options={[
            { value: 'none', label: 'None' },
            { value: 'before', label: 'Opened before the ALTER' },
            { value: 'during', label: 'Opened halfway through the build' },
          ]}
        />
        <Slider label={open === 'before' ? 'It commits after (s)' : 'It stays open for (s)'} min={0} max={1200} step={30} value={openSec} onChange={(v) => touch(() => setOpenSec(v))} format={(v) => `${v} s`} />
        <Slider label="Time t" min={0} max={horizon} step={tStep} value={t} onChange={(v) => setCursor(v)} format={(v) => `${fmtNum(v)} s`} />
      </div>

      <svg viewBox={`0 0 ${W} ${H}`} width={W} height={H} style={{ minWidth: 600 }} role="img" aria-label={`InnoDB online ADD INDEX at t = ${fmtNum(t)} s: ${st.stmt}; row log ${fmtMiB(sim.tailAt(t))}`}>
        <text x={8} y={yAlter + 15} fontSize={11} fontWeight={600} fill="var(--viz-ink)">
          ALTER … ADD INDEX
        </text>
        {segs.map((s, i) => {
          const w = Math.max(0, xs(s.to) - xs(s.from));
          if (w <= 0) return null;
          return (
            <g key={i}>
              <rect x={xs(s.from)} y={yAlter} width={Math.max(2, w)} height={22} rx={3} fill="var(--viz-surface)" stroke={s.stroke} strokeWidth={2.5} />
              {w > s.label.length * 5.6 + 6 ? (
                <text x={xs(s.from) + w / 2} y={yAlter + 15} textAnchor="middle" fontSize={10} fill="var(--viz-ink)">
                  {s.label}
                </text>
              ) : null}
            </g>
          );
        })}
        {sim.failed ? (
          <g>
            <path d={`M ${xs(sim.errorAt as number) - 7} ${yAlter + 4} l 14 14 M ${xs(sim.errorAt as number) + 7} ${yAlter + 4} l -14 14`} stroke="var(--viz-critical)" strokeWidth={3} />
            <text x={W - RIGHT} y={yAlter - 5} textAnchor="end" fontSize={10} fill="var(--viz-ink)">
              ER_INNODB_ONLINE_LOG_TOO_BIG: index dropped
            </text>
          </g>
        ) : null}

        <text x={8} y={yOpen + 14} fontSize={10.5} fill="var(--viz-ink)">
          open transaction
        </text>
        <line x1={LEFT} x2={W - RIGHT} y1={yOpen + 10} y2={yOpen + 10} stroke="var(--viz-grid)" />
        {sim.openStart !== null && sim.openEnd !== null ? (
          <g>
            <rect x={xs(sim.openStart)} y={yOpen + 3} width={Math.max(3, xs(sim.openEnd) - xs(sim.openStart))} height={14} rx={4} fill="var(--viz-plane)" stroke={currentBlock && currentBlock.why.includes('MDL') && sim.openEnd > t ? 'var(--viz-warning)' : 'var(--viz-ink-muted)'} strokeWidth={currentBlock && currentBlock.why.includes('MDL') && sim.openEnd > t ? 3 : 1} />
            <line x1={xs(sim.openEnd)} x2={xs(sim.openEnd)} y1={yOpen} y2={yOpen + 20} stroke="var(--viz-ink)" strokeWidth={1.5} />
          </g>
        ) : null}

        <text x={8} y={yDml + 14} fontSize={10.5} fill="var(--viz-ink)">
          application DML
        </text>
        {dmlSegs.map((s, i) => (
          <rect key={i} x={xs(s.from)} y={yDml + (s.kind === 'blocked' ? 2 : 5)} width={Math.max(1, xs(s.to) - xs(s.from))} height={s.kind === 'blocked' ? 16 : 10} rx={2} fill={s.kind === 'blocked' ? 'var(--viz-warning)' : s.kind === 'log' ? 'var(--viz-7)' : s.kind === 'plain' ? 'var(--viz-stale)' : 'var(--viz-2)'} opacity={0.9} />
        ))}

        {/* row log gauge */}
        <text x={8} y={chartTop - 10} fontSize={11} fontWeight={600} fill="var(--viz-ink)">
          Row log size (writer offset in the temporary file)
        </text>
        <line x1={LEFT} x2={W - RIGHT} y1={chartBottom} y2={chartBottom} stroke="var(--viz-axis)" />
        <line x1={LEFT} x2={LEFT} y1={chartTop} y2={chartBottom} stroke="var(--viz-axis)" />
        {[0, 0.5, 1].map((f) => (
          <text key={f} x={LEFT - 6} y={ys(yMax * f * 0.89) + 4} textAnchor="end" fontSize={10} fill="var(--viz-ink-2)">
            {fmtMiB(yMax * f * 0.89)}
          </text>
        ))}
        <line x1={LEFT} x2={W - RIGHT} y1={ys(sim.failBytes)} y2={ys(sim.failBytes)} stroke="var(--viz-critical)" strokeDasharray="6 4" strokeWidth={1.5} />
        <text x={W - RIGHT - 4} y={ys(sim.failBytes) - 5} textAnchor="end" fontSize={10} fill="var(--viz-ink)">
          innodb_online_alter_log_max_size ({fmtMiB(sim.failBytes)})
        </text>
        <path d={pathOf(sim.tailAt)} fill="none" stroke="var(--viz-7)" strokeWidth={2.5} />
        <path d={pathOf(sim.appliedAt)} fill="none" stroke="var(--viz-3)" strokeWidth={2.5} strokeDasharray="6 3" />
        {Array.from({ length: 5 }, (_, k) => Math.round((k * horizon) / 4 / 10) * 10).map((v) => (
          <g key={v}>
            <line x1={xs(v)} x2={xs(v)} y1={chartBottom} y2={chartBottom + 4} stroke="var(--viz-axis)" />
            <text x={xs(v)} y={chartBottom + 16} textAnchor="middle" fontSize={10} fill="var(--viz-ink-2)">
              {fmtNum(v)} s
            </text>
          </g>
        ))}
        <line x1={xs(t)} x2={xs(t)} y1={yAlter - 8} y2={chartBottom} stroke="var(--viz-ink)" strokeWidth={1.5} />
        <circle cx={xs(t)} cy={ys(sim.tailAt(t))} r={4} fill="var(--viz-7)" stroke="var(--viz-surface)" strokeWidth={1.5} />
      </svg>

    </VizPanel>
  );
}

export default function CreateIndexConcurrentlyLab() {
  const [panel, setPanel] = useState<'pg' | 'innodb'>('pg');
  const engine = (
    <Segmented
      label="Engine"
      value={panel}
      onChange={setPanel}
      options={[
        { value: 'pg', label: 'PostgreSQL CREATE INDEX CONCURRENTLY' },
        { value: 'innodb', label: 'InnoDB online ADD INDEX' },
      ]}
    />
  );
  return panel === 'pg' ? <PgPanel engine={engine} /> : <InnoPanel engine={engine} />;
}
