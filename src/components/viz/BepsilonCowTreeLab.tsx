import { useMemo, useRef, useState } from 'react';
import { VizPanel, Segmented, Choice, Slider, Button, Legend, Stats, Note, TooltipHost, useTip, makeRng, fmtNum } from './Viz';

/**
 * Two write paths that avoid rewriting a B+tree leaf in place for every change.
 *
 * Bε-tree mode (message buffers, after Bender et al., "An Introduction to Bε-trees and Write-Optimization",
 * ;login: 2015, and PerconaFT's ft-ops.cc / ft-flusher.cc):
 * - A node holds B units. An internal node spends `fanout` of them on pivots and the rest on a message buffer,
 *   so ε = log_B(fanout). Every insert, upsert or delete becomes a message with a sequence number (PerconaFT's MSN)
 *   and lands in the root's buffer. (PerconaFT also promotes a message up to two levels lower past empty buffers, and
 *   to a leaf at the tree's left and right edges; the lab keeps the textbook root-only rule.)
 * - When a buffer holds more than its capacity, the node flushes the whole buffer segment of its heaviest child
 *   (PerconaFT's find_heaviest_child). If that child is internal and is now over capacity, it flushes too — a cascade.
 *   A leaf applies the messages it receives. The lab repeats flushes until the node is back under capacity; PerconaFT
 *   may leave that to a later flush or its cleaner thread.
 * - A lookup reads every node on the root-to-leaf path and applies the pending messages for its key in MSN order.
 * Model assumptions: the tree shape is fixed (64 leaves, no splits or merges on either side); the tree is far larger
 * than memory, so a B+tree insert dirties a different leaf nearly every time and costs one leaf write; the Bε-tree's
 * root buffer lives in memory (made durable by a log, as PerconaFT does) and a flush writes every node it modified
 * once. Checkpoint coalescing of writes is not modelled for either tree.
 *
 * Copy-on-write mode (after LMDB 0.9 mdb.c and bbolt tx.go / freelist):
 * - A commit copies every page it changes plus all their ancestors to newly allocated page numbers (a page is copied
 *   once per transaction, as mdb_page_touch does), writes them, syncs, then writes meta page (txnid & 1) and syncs.
 *   Reopen uses the meta page with the higher transaction id that validates (bbolt checksums its meta; LMDB 0.9 does not).
 * - Freed pages are reused under each engine's rule. LMDB (mdb_find_oldest/mdb_page_alloc): a page freed by txn t is
 *   reusable only when t < min(oldest reader's txn, last committed txn). bbolt (ReleasePendingPages): reusable when no
 *   reader is open, or t < oldest reader's txn, or the page was allocated after that reader started.
 * - The lab's data tree has three levels and fanout 3. It omits LMDB's free-page B-tree and bbolt's freelist page, which
 *   every real commit also writes. It reuses the lowest reusable page number. A crash is a machine crash: open readers
 *   die with it.
 */

/* ============================================================ Bε-tree model */

export type MsgType = 'insert' | 'upsert' | 'delete';
export type Msg = { key: number; msn: number; type: MsgType };
export type Workload = 'random' | 'hot' | 'mixed';
export type BeConfig = { nodeSize: number; fanout: number; workload: Workload };

export const LEAVES = 64;
export const KEYS_PER_LEAF = 100;
export const KEY_SPACE = LEAVES * KEYS_PER_LEAF;
export const MAX_OPS = 10_000;
/** Sixteen counters spread across the key space, one every fourth leaf. */
export const HOT_KEYS = Array.from({ length: 16 }, (_, i) => (i * 4 + 1) * KEYS_PER_LEAF + 55);

export const internalLevels = (fanout: number) => Math.round(Math.log(LEAVES) / Math.log(fanout));
export const bufferCapacity = (c: BeConfig) => c.nodeSize - c.fanout;
export const epsilonOf = (c: BeConfig) => Math.log(c.fanout) / Math.log(c.nodeSize);
/** Nodes a B+tree of the same node size reads for a lookup: its fanout is the full node size. */
export const bplusLookupReads = (nodeSize: number) => Math.ceil(Math.log(LEAVES) / Math.log(nodeSize) - 1e-9) + 1;

const opsCache = new Map<Workload, Msg[]>();

export function workloadOps(w: Workload): Msg[] {
  const hit = opsCache.get(w);
  if (hit) return hit;
  const rng = makeRng(w === 'random' ? 11 : w === 'hot' ? 23 : 37);
  const out: Msg[] = [];
  for (let i = 0; i < MAX_OPS; i++) {
    const msn = i + 1;
    if (w === 'random') out.push({ key: Math.floor(rng() * KEY_SPACE), msn, type: 'insert' });
    else if (w === 'hot') out.push({ key: HOT_KEYS[Math.floor(rng() * HOT_KEYS.length)], msn, type: 'upsert' });
    else {
      const r = rng();
      if (r < 0.6) out.push({ key: Math.floor(rng() * KEY_SPACE), msn, type: 'insert' });
      else if (r < 0.85) out.push({ key: HOT_KEYS[Math.floor(rng() * HOT_KEYS.length)], msn, type: 'upsert' });
      else out.push({ key: Math.floor(rng() * LEAVES) * KEYS_PER_LEAF + 5 * Math.floor(rng() * 20), msn, type: 'delete' });
    }
  }
  opsCache.set(w, out);
  return out;
}

/** Each leaf starts with 20 keys (every fifth key in its range), all with value 0. */
export function initialLeaves(): Map<number, number>[] {
  return Array.from({ length: LEAVES }, (_, l) => {
    const m = new Map<number, number>();
    for (let j = 0; j < 20; j++) m.set(l * KEYS_PER_LEAF + j * 5, 0);
    return m;
  });
}

/** insert writes the op number as the value; upsert adds 1 (a counter); delete removes the key. */
export function applyMsg(v: number | undefined, m: Msg): number | undefined {
  if (m.type === 'insert') return m.msn;
  if (m.type === 'upsert') return (v ?? 0) + 1;
  return undefined;
}

export type FlushEdge = { level: number; index: number; child: number; moved: number; toLeaf: boolean };
export type BeEvent = { op: number; edges: FlushEdge[]; written: string[]; reachedLeaf: boolean };
export type BeState = {
  cfg: BeConfig;
  levels: number;
  cap: number;
  buffers: Msg[][][];
  leaves: Map<number, number>[];
  leafWrites: number[];
  ops: number;
  nodeWrites: number;
  flushes: number;
  leafFlushes: number;
  writesByLevel: number[];
  flushesFromLevel: number[];
  movedFromLevel: number[];
  cumWrites: number[];
  eventKind: number[];
  lastEvent: BeEvent | null;
  lastOp: Msg | null;
};

export const beNodeId = (levels: number, level: number, index: number) => (level === levels ? `leaf:${index}` : `n${level}:${index}`);

export function simulateBe(cfg: BeConfig, n: number): BeState {
  const levels = internalLevels(cfg.fanout);
  const cap = bufferCapacity(cfg);
  const f = cfg.fanout;
  const s: BeState = {
    cfg,
    levels,
    cap,
    buffers: Array.from({ length: levels }, (_, d) => Array.from({ length: f ** d }, () => [] as Msg[])),
    leaves: initialLeaves(),
    leafWrites: new Array(LEAVES).fill(0),
    ops: 0,
    nodeWrites: 0,
    flushes: 0,
    leafFlushes: 0,
    writesByLevel: new Array(levels + 1).fill(0),
    flushesFromLevel: new Array(levels).fill(0),
    movedFromLevel: new Array(levels).fill(0),
    cumWrites: [0],
    eventKind: [0],
    lastEvent: null,
    lastOp: null,
  };
  const ops = workloadOps(cfg.workload);
  const count = Math.max(0, Math.min(MAX_OPS, Math.floor(n)));

  const flush = (d: number, idx: number, written: Map<string, number>, edges: FlushEdge[]) => {
    const buf = s.buffers[d][idx];
    const span = f ** (levels - d - 1);
    const counts = new Array(f).fill(0);
    const pos = (m: Msg) => Math.floor(Math.floor(m.key / KEYS_PER_LEAF) / span) - idx * f;
    for (const m of buf) counts[pos(m)]++;
    let best = 0;
    for (let c = 1; c < f; c++) if (counts[c] > counts[best]) best = c;
    const moved: Msg[] = [];
    const rest: Msg[] = [];
    for (const m of buf) (pos(m) === best ? moved : rest).push(m);
    s.buffers[d][idx] = rest;
    written.set(beNodeId(levels, d, idx), d);
    const childIdx = idx * f + best;
    s.flushes++;
    s.flushesFromLevel[d]++;
    s.movedFromLevel[d] += moved.length;
    if (d + 1 === levels) {
      const leaf = s.leaves[childIdx];
      for (const m of moved) {
        const v = applyMsg(leaf.get(m.key), m);
        if (v === undefined) leaf.delete(m.key);
        else leaf.set(m.key, v);
      }
      written.set(beNodeId(levels, levels, childIdx), levels);
      s.leafFlushes++;
      s.leafWrites[childIdx]++;
      edges.push({ level: d, index: idx, child: childIdx, moved: moved.length, toLeaf: true });
    } else {
      const child = s.buffers[d + 1][childIdx];
      for (const m of moved) child.push(m);
      written.set(beNodeId(levels, d + 1, childIdx), d + 1);
      edges.push({ level: d, index: idx, child: childIdx, moved: moved.length, toLeaf: false });
      while (s.buffers[d + 1][childIdx].length > cap) flush(d + 1, childIdx, written, edges);
    }
  };

  for (let i = 0; i < count; i++) {
    const m = ops[i];
    s.buffers[0][0].push(m);
    const written = new Map<string, number>();
    const edges: FlushEdge[] = [];
    while (s.buffers[0][0].length > cap) flush(0, 0, written, edges);
    s.nodeWrites += written.size;
    for (const lvl of written.values()) s.writesByLevel[lvl]++;
    const reachedLeaf = edges.some((e) => e.toLeaf);
    s.lastEvent = edges.length ? { op: m.msn, edges, written: [...written.keys()], reachedLeaf } : null;
    s.lastOp = m;
    s.ops++;
    s.cumWrites.push(s.nodeWrites);
    s.eventKind.push(edges.length === 0 ? 0 : reachedLeaf ? 2 : 1);
  }
  return s;
}

export function bufferedTotal(s: BeState) {
  return s.buffers.reduce((a, lvl) => a + lvl.reduce((b, buf) => b + buf.length, 0), 0);
}

export type Lookup = {
  key: number;
  leaf: number;
  path: { level: number; index: number; msgs: Msg[] }[];
  leafValue: number | undefined;
  pending: Msg[];
  result: number | undefined;
  nodesRead: number;
  bplusReads: number;
};

export function lookupBe(s: BeState, key: number): Lookup {
  const f = s.cfg.fanout;
  const leaf = Math.floor(key / KEYS_PER_LEAF);
  const path = [];
  for (let d = 0; d < s.levels; d++) {
    const index = Math.floor(leaf / f ** (s.levels - d));
    path.push({ level: d, index, msgs: s.buffers[d][index].filter((m) => m.key === key) });
  }
  const leafValue = s.leaves[leaf].get(key);
  const pending = path.flatMap((p) => p.msgs).sort((a, b) => a.msn - b.msn);
  let result = leafValue;
  for (const m of pending) result = applyMsg(result, m);
  return { key, leaf, path, leafValue, pending, result, nodesRead: s.levels + 1, bplusReads: bplusLookupReads(s.cfg.nodeSize) };
}

export type LookupPick = 'busiest' | 'last' | 'tomb' | 'cold';

export function pickLookupKey(s: BeState, which: LookupPick): number {
  const counts = new Map<number, number>();
  let newestDelete: Msg | null = null;
  for (const lvl of s.buffers)
    for (const buf of lvl)
      for (const m of buf) {
        counts.set(m.key, (counts.get(m.key) ?? 0) + 1);
        if (m.type === 'delete' && (!newestDelete || m.msn > newestDelete.msn)) newestDelete = m;
      }
  const busiest = () => {
    let best = -1;
    let bestN = 0;
    for (const [k, c] of counts) if (c > bestN || (c === bestN && k < best)) (best = k), (bestN = c);
    return best >= 0 ? best : s.lastOp?.key ?? HOT_KEYS[0];
  };
  if (which === 'last') return s.lastOp?.key ?? HOT_KEYS[0];
  if (which === 'tomb') return newestDelete ? newestDelete.key : busiest();
  if (which === 'cold') {
    for (let l = 0; l < LEAVES; l++) {
      const keys = [...s.leaves[l].keys()].sort((a, b) => a - b);
      for (const k of keys) if (!counts.has(k)) return k;
    }
    return 0;
  }
  return busiest();
}

/* ============================================================ copy-on-write model */

export type Engine = 'lmdb' | 'bbolt';
export const COW_BRANCHES = 3;
export const COW_LEAVES = 9;
export const MAX_TXN = 25;

export type Snapshot = { txn: number; root: number; branches: number[]; leaves: number[] };
export type FreeEntry = { pgno: number; freedBy: number; alloc: number };
/** Oldest open reader: started after commit `start` (0 = none) and stays open through write txn `end`. */
export type Reader = { start: number; end: number };
export type PlanEntry = { kind: 'root' | 'branch' | 'leaf'; pos: number; old: number; pgno: number; reused: boolean };
export type CommitPlan = { txn: number; leaves: number[]; entries: PlanEntry[]; next: Snapshot; fileEndAfter: number };
export type CommitRec = { txn: number; leaves: number[]; copied: number; reused: number; extended: number; fileEnd: number; root: number };

export const snapshotPages = (s: Snapshot) => (s.root < 0 ? [] : [s.root, ...s.branches, ...s.leaves]);

export function readerPresent(r: Reader, writeTxn: number) {
  return r.start >= 1 && r.start < writeTxn && writeTxn <= r.end;
}

/** May write transaction `writeTxn` reuse this freed page? */
export function reusable(engine: Engine, e: FreeEntry, writeTxn: number, r: Reader) {
  const present = readerPresent(r, writeTxn);
  if (engine === 'lmdb') {
    const oldest = present ? Math.min(r.start, writeTxn - 1) : writeTxn - 1;
    return e.freedBy < oldest;
  }
  if (!present) return true;
  return e.freedBy < r.start || e.alloc > r.start;
}

export function leavesForTxn(txn: number, keys: number): number[] {
  const rng = makeRng(txn * 7919 + 17);
  const picked: number[] = [];
  while (picked.length < keys) {
    const l = Math.floor(rng() * COW_LEAVES);
    if (!picked.includes(l)) picked.push(l);
  }
  return picked.sort((a, b) => a - b);
}

export type CowRun = {
  engine: Engine;
  keys: number;
  txn: number;
  snapshots: Snapshot[];
  commits: CommitRec[];
  free: FreeEntry[];
  alloc: Map<number, number>;
  fileEnd: number;
  violations: string[];
};

export function planCommit(run: CowRun, reader: Reader): CommitPlan {
  const t = run.txn + 1;
  const prev = run.snapshots[run.txn];
  const leaves = leavesForTxn(t, run.keys);
  const branches = [...new Set(leaves.map((l) => Math.floor(l / 3)))].sort((a, b) => a - b);
  const order: Omit<PlanEntry, 'pgno' | 'reused'>[] = [
    { kind: 'root', pos: 0, old: prev.root },
    ...branches.map((b) => ({ kind: 'branch' as const, pos: b, old: prev.branches[b] })),
    ...leaves.map((l) => ({ kind: 'leaf' as const, pos: l, old: prev.leaves[l] })),
  ];
  const candidates = run.free.filter((e) => reusable(run.engine, e, t, reader)).sort((a, b) => a.pgno - b.pgno);
  let fileEnd = run.fileEnd;
  const entries: PlanEntry[] = order.map((o) => {
    const c = candidates.shift();
    return c ? { ...o, pgno: c.pgno, reused: true } : { ...o, pgno: fileEnd++, reused: false };
  });
  const next: Snapshot = { txn: t, root: prev.root, branches: prev.branches.slice(), leaves: prev.leaves.slice() };
  for (const e of entries) {
    if (e.kind === 'root') next.root = e.pgno;
    else if (e.kind === 'branch') next.branches[e.pos] = e.pgno;
    else next.leaves[e.pos] = e.pgno;
  }
  return { txn: t, leaves, entries, next, fileEndAfter: fileEnd };
}

function applyPlan(run: CowRun, plan: CommitPlan, reader: Reader) {
  const t = plan.txn;
  const latest = new Set(snapshotPages(run.snapshots[run.txn]));
  const readerPages = readerPresent(reader, t) ? new Set(snapshotPages(run.snapshots[reader.start])) : new Set<number>();
  const otherMeta = t >= 3 ? new Set(snapshotPages(run.snapshots[t - 2])) : new Set<number>();
  for (const e of plan.entries) {
    if (!e.reused) continue;
    if (latest.has(e.pgno)) run.violations.push(`txn ${t} reused live page ${e.pgno}`);
    if (readerPages.has(e.pgno)) run.violations.push(`txn ${t} reused page ${e.pgno} still read by the reader at txn ${reader.start}`);
    if (run.engine === 'lmdb' && otherMeta.has(e.pgno)) run.violations.push(`txn ${t} reused page ${e.pgno} of the other meta page's tree`);
  }
  const used = new Set(plan.entries.filter((e) => e.reused).map((e) => e.pgno));
  run.free = run.free.filter((e) => !used.has(e.pgno));
  for (const e of plan.entries) {
    run.free.push({ pgno: e.old, freedBy: t, alloc: run.alloc.get(e.old) ?? 0 });
    run.alloc.delete(e.old);
    run.alloc.set(e.pgno, t);
  }
  run.fileEnd = plan.fileEndAfter;
  run.snapshots[t] = plan.next;
  run.commits[t] = {
    txn: t,
    leaves: plan.leaves,
    copied: plan.entries.length,
    reused: plan.entries.filter((e) => e.reused).length,
    extended: plan.entries.filter((e) => !e.reused).length,
    fileEnd: run.fileEnd,
    root: plan.next.root,
  };
  run.txn = t;
}

export function runCow(engine: Engine, keys: number, txn: number, reader: Reader): CowRun {
  const boot: Snapshot = { txn: 1, root: 2, branches: [3, 4, 5], leaves: [6, 7, 8, 9, 10, 11, 12, 13, 14] };
  const run: CowRun = {
    engine,
    keys,
    txn: 1,
    snapshots: [{ txn: 0, root: -1, branches: [], leaves: [] }, boot],
    commits: [],
    free: [],
    alloc: new Map(snapshotPages(boot).map((p) => [p, 1])),
    fileEnd: 15,
    violations: [],
  };
  run.commits[1] = { txn: 1, leaves: [0, 1, 2, 3, 4, 5, 6, 7, 8], copied: 13, reused: 0, extended: 13, fileEnd: 15, root: 2 };
  const target = Math.max(1, Math.min(MAX_TXN, Math.floor(txn)));
  while (run.txn < target) applyPlan(run, planCommit(run, reader), reader);
  return run;
}

export type PageState = 'meta' | 'live' | 'inflight' | 'garbage' | 'pinned' | 'held' | 'free';

/**
 * State of every page in the file as the next write transaction (txn + 1) sees it.
 * `phase` 1..4 is an in-flight commit (1 copy, 2 write, 3 fsync, 4 meta write); `crashed` shows the reopened file.
 * Step 1 copies pages in memory only, so the copies reach the file (and can be left behind by a crash) from step 2 on.
 */
export function classifyPages(run: CowRun, reader: Reader, plan: CommitPlan | null, phase: number, crashed: boolean): PageState[] {
  const T = run.txn + 1;
  const end = Math.max(run.fileEnd, plan && phase >= 2 ? plan.fileEndAfter : 0);
  const out: PageState[] = new Array(end).fill('free');
  out[0] = 'meta';
  out[1] = 'meta';
  const live = new Set(snapshotPages(run.snapshots[run.txn]));
  const present = !crashed && readerPresent(reader, T);
  const readerPages = present ? new Set(snapshotPages(run.snapshots[reader.start])) : new Set<number>();
  const noReader: Reader = { start: 0, end: 0 };
  for (const p of live) out[p] = 'live';
  for (const e of run.free) {
    if (readerPages.has(e.pgno)) out[e.pgno] = 'pinned';
    else out[e.pgno] = reusable(run.engine, e, T, crashed ? noReader : reader) ? 'free' : 'held';
  }
  if (plan && phase >= 2) for (const e of plan.entries) out[e.pgno] = crashed ? 'garbage' : 'inflight';
  return out;
}

export function countStates(states: PageState[]) {
  const c: Record<PageState, number> = { meta: 0, live: 0, inflight: 0, garbage: 0, pinned: 0, held: 0, free: 0 };
  for (const s of states) c[s]++;
  return c;
}

export type MetaView = { slot: number; txn: number; root: number; status: 'newest' | 'older' | 'writing' | 'torn' | 'unwritten' };

export function metaView(run: CowRun, plan: CommitPlan | null, phase: number, crashed: boolean): MetaView[] {
  const n = run.txn;
  return [0, 1].map((slot) => {
    const committedTxn = n % 2 === slot ? n : n - 1;
    const snap = run.snapshots[committedTxn];
    const base: MetaView = { slot, txn: committedTxn, root: snap ? snap.root : -1, status: committedTxn === n ? 'newest' : 'older' };
    if (plan && phase === 4 && plan.txn % 2 === slot) {
      if (!crashed) return { slot, txn: plan.txn, root: plan.next.root, status: 'writing' };
      return run.engine === 'bbolt' ? { slot, txn: plan.txn, root: plan.next.root, status: 'torn' } : { ...base, status: 'unwritten' };
    }
    return base;
  });
}

/* ============================================================ component */

const LOOKUP_OPTIONS: Record<Workload, { value: LookupPick; label: string }[]> = {
  random: [
    { value: 'busiest', label: 'Key with the most pending messages' },
    { value: 'last', label: 'Key of the last operation' },
    { value: 'cold', label: 'A key with nothing pending' },
  ],
  hot: [
    { value: 'busiest', label: 'Busiest counter' },
    { value: 'last', label: 'Counter of the last operation' },
    { value: 'cold', label: 'A key with nothing pending' },
  ],
  mixed: [
    { value: 'busiest', label: 'Key with the most pending messages' },
    { value: 'tomb', label: 'Newest pending tombstone' },
    { value: 'last', label: 'Key of the last operation' },
    { value: 'cold', label: 'A key with nothing pending' },
  ],
};

const MSG_COLOR: Record<MsgType, string> = { insert: 'var(--viz-3)', upsert: 'var(--viz-4)', delete: 'var(--viz-5)' };
const MSG_LABEL: Record<MsgType, string> = { insert: 'insert', upsert: 'upsert', delete: 'tombstone' };

const STATE_COLOR: Record<PageState, string> = {
  meta: 'var(--viz-3)',
  live: 'var(--viz-clean)',
  inflight: 'var(--viz-dirty)',
  pinned: 'var(--viz-warning)',
  held: 'var(--viz-5)',
  free: 'var(--viz-stale)',
  garbage: 'var(--viz-stale)',
};

const PHASES = [
  'Copy the path in memory: each changed leaf, its branch and the root get new page numbers; the old pages go on the free list for this transaction. The file is untouched.',
  'Write the copied pages to the file. Nothing points at them yet.',
  'fsync the file: the new pages are durable but still unreferenced.',
  'Write meta page (txn & 1) with the new root and transaction id, and sync it. This one page write is the commit.',
];

const fmtVal = (v: number | undefined) => (v === undefined ? 'absent' : fmtNum(v));

function summarizeMsgs(msgs: Msg[]) {
  const parts = (['insert', 'upsert', 'delete'] as MsgType[])
    .map((t) => {
      const k = msgs.filter((m) => m.type === t).length;
      return k ? `${k} ${MSG_LABEL[t]}${k === 1 ? '' : 's'}` : '';
    })
    .filter(Boolean);
  const range = msgs.length <= 3 ? ` (${msgs.map((m) => `#${m.msn}`).join(', ')})` : ` (#${msgs[0].msn} … #${msgs[msgs.length - 1].msn})`;
  return parts.join(' + ') + range;
}

function BeFigure({ s, look, W }: { s: BeState; look: Lookup; W: number }) {
  const tip = useTip();
  const f = s.cfg.fanout;
  const pad = 10;
  const rowH = s.levels > 3 ? 46 : 64;
  const top = 26;
  const leafY = top + s.levels * rowH;
  const H = leafY + 44;
  const written = new Set(s.lastEvent?.written ?? []);
  const onPath = new Set([...look.path.map((p) => beNodeId(s.levels, p.level, p.index)), beNodeId(s.levels, s.levels, look.leaf)]);
  const slot = (level: number) => (W - 2 * pad) / (level === s.levels ? LEAVES : f ** level);
  const nodeBox = (level: number, index: number) => {
    const sw = slot(level);
    const gap = Math.min(8, sw * 0.18);
    const h = level === s.levels ? 24 : 32;
    return { x: pad + index * sw + gap / 2, y: level === s.levels ? leafY : top + level * rowH, w: sw - gap, h };
  };

  return (
    <svg viewBox={`0 0 ${W} ${H}`} width={W} height={H} role="img" aria-label={`Bε-tree with ${s.levels} internal levels, fanout ${f}, ${fmtNum(bufferedTotal(s))} buffered messages after ${fmtNum(s.ops)} operations`}>
      {(() => {
        const edges = s.lastEvent?.edges ?? [];
        const groups = new Map<string, FlushEdge[]>();
        for (const e of edges) {
          const k = `${e.level}:${e.index}`;
          groups.set(k, [...(groups.get(k) ?? []), e]);
        }
        const line = (e: FlushEdge) => {
          const a = nodeBox(e.level, e.index);
          const b = nodeBox(e.level + 1, e.child);
          return { x1: a.x + a.w / 2, y1: a.y + a.h, x2: b.x + b.w / 2, y2: b.y };
        };
        return (
          <g>
            {edges.map((e, i) => {
              const l = line(e);
              return <line key={`e${i}`} x1={l.x1} y1={l.y1} x2={l.x2} y2={l.y2 - 2} stroke="var(--viz-ink)" strokeWidth={2} />;
            })}
            {[...groups.values()].map((g, i) => {
              const l = line(g[0]);
              const label = `${g.map((e) => e.moved).join(' + ')} msgs`;
              const px = l.x1 + 0.4 * (l.x2 - l.x1);
              const py = l.y1 + 0.4 * (l.y2 - l.y1) + 4;
              const goesLeft = l.x2 < l.x1 - 2;
              const textW = label.length * 6.2;
              let anchor: 'start' | 'end' = goesLeft ? 'end' : 'start';
              let x = goesLeft ? px - 8 : px + 8;
              if (anchor === 'start' && x + textW > W - 4) (anchor = 'end'), (x = Math.min(px - 8, W - 4));
              if (anchor === 'end' && x - textW < 4) (anchor = 'start'), (x = Math.max(px + 8, 4));
              const dx = l.x2 - l.x1;
              let y = py;
              if (Math.abs(dx) >= 2) {
                const lineY = Math.min(l.y2, Math.max(l.y1, l.y1 + ((x - l.x1) * (l.y2 - l.y1)) / dx));
                const above = (anchor === 'start') === dx > 0;
                y = Math.min(l.y2 + 10, Math.max(l.y1 + 10, above ? lineY - 3 : lineY + 12));
              }
              return (
                <text key={`t${i}`} x={x} y={y} textAnchor={anchor} fontSize={11} fill="var(--viz-ink)" stroke="var(--viz-surface)" strokeWidth={3} paintOrder="stroke">
                  {label}
                </text>
              );
            })}
          </g>
        );
      })()}
      {s.buffers.map((lvl, d) =>
        lvl.map((buf, i) => {
          const b = nodeBox(d, i);
          const id = beNodeId(s.levels, d, i);
          const counts: Record<MsgType, number> = { insert: 0, upsert: 0, delete: 0 };
          for (const m of buf) counts[m.type]++;
          let x = b.x + 2;
          const track = b.w - 4;
          return (
            <g key={id} {...tip(<><strong>{d === 0 ? 'Root' : `Level ${d}, node ${i}`}</strong><br />Buffer {buf.length} / {s.cap} messages<br />{counts.insert} insert · {counts.upsert} upsert · {counts.delete} tombstone</>)}>
              {onPath.has(id) ? <rect x={b.x - 3} y={b.y - 3} width={b.w + 6} height={b.h + 6} rx={5} fill="none" stroke="var(--viz-ink-2)" strokeDasharray="4 3" /> : null}
              <rect x={b.x} y={b.y} width={b.w} height={b.h} rx={3} fill="var(--viz-surface)" stroke={written.has(id) ? 'var(--viz-ink)' : 'var(--viz-ink-muted)'} strokeWidth={written.has(id) ? 2.5 : 1} />
              <rect x={b.x + 2} y={b.y + b.h - 11} width={track} height={8} fill="none" stroke="var(--viz-grid)" />
              {(['insert', 'upsert', 'delete'] as MsgType[]).map((t) => {
                const w = (counts[t] / s.cap) * track;
                const el = w > 0 ? <rect key={t} x={x} y={b.y + b.h - 11} width={w} height={8} fill={MSG_COLOR[t]} /> : null;
                x += w;
                return el;
              })}
              {b.w >= 46 ? (
                <text x={b.x + b.w / 2} y={b.y + 13} textAnchor="middle" fontSize={10} fill="var(--viz-ink)">
                  {d === 0 ? `root ${buf.length}/${s.cap}` : `${buf.length}/${s.cap}`}
                </text>
              ) : null}
            </g>
          );
        }),
      )}
      {s.leaves.map((leaf, i) => {
        const b = nodeBox(s.levels, i);
        const id = beNodeId(s.levels, s.levels, i);
        return (
          <g key={id} {...tip(<><strong>Leaf {i}</strong> (keys {i * KEYS_PER_LEAF}–{i * KEYS_PER_LEAF + KEYS_PER_LEAF - 1})<br />{leaf.size} items · written {s.leafWrites[i]} times</>)}>
            {onPath.has(id) ? <rect x={b.x - 2} y={b.y - 3} width={b.w + 4} height={b.h + 6} rx={3} fill="none" stroke="var(--viz-ink-2)" strokeDasharray="4 3" /> : null}
            <rect x={b.x} y={b.y} width={Math.max(2, b.w)} height={b.h} rx={2} fill="var(--viz-neutral)" stroke={written.has(id) ? 'var(--viz-ink)' : 'var(--viz-ink-muted)'} strokeWidth={written.has(id) ? 2.5 : 0.75} />
          </g>
        );
      })}
      <text x={pad} y={14} fontSize={11} fill="var(--viz-ink-2)">
        Internal nodes: pivots + message buffer (fill bar)
      </text>
      <text x={pad} y={leafY + 40} fontSize={11} fill="var(--viz-ink-2)">
        {LEAVES} leaves · dashed outline = path of the lookup below · dark outline = written by the last flush
      </text>
    </svg>
  );
}

function RaceChart({ s, W }: { s: BeState; W: number }) {
  const H = 116;
  const left = 10;
  const right = 200;
  const plotW = W - left - right;
  const top = 12;
  const plotH = H - 34;
  const n = Math.max(1, s.ops);
  const X = (i: number) => left + (i / n) * plotW;
  const Y = (v: number) => top + plotH - (v / n) * plotH;
  const step = Math.max(1, Math.floor(s.ops / 300));
  const pts: string[] = [];
  for (let i = 0; i <= s.ops; i += step) pts.push(`${X(i).toFixed(1)},${Y(s.cumWrites[i]).toFixed(1)}`);
  pts.push(`${X(s.ops).toFixed(1)},${Y(s.cumWrites[s.ops]).toFixed(1)}`);
  return (
    <svg viewBox={`0 0 ${W} ${H}`} width={W} height={H} role="img" aria-label={`After ${fmtNum(s.ops)} operations the Bε-tree has written ${fmtNum(s.nodeWrites)} nodes and a B+tree ${fmtNum(s.ops)} leaves`} style={{ marginTop: 8 }}>
      <line x1={left} x2={left + plotW} y1={top + plotH} y2={top + plotH} stroke="var(--viz-axis)" />
      <line x1={left} x2={left} y1={top} y2={top + plotH} stroke="var(--viz-axis)" />
      <line x1={X(0)} y1={Y(0)} x2={X(s.ops)} y2={Y(s.ops)} stroke="var(--viz-2)" strokeWidth={2.5} />
      <polyline points={pts.join(' ')} fill="none" stroke="var(--viz-1)" strokeWidth={2.5} />
      <text x={left + plotW + 8} y={Y(s.ops) + 4} fontSize={11} fill="var(--viz-ink)">
        B+tree: {fmtNum(s.ops)} leaf writes
      </text>
      <text x={left + plotW + 8} y={Math.min(top + plotH, Math.max(Y(s.nodeWrites) + 4, Y(s.ops) + 22))} fontSize={11} fill="var(--viz-ink)">
        Bε-tree: {fmtNum(s.nodeWrites)} node writes
      </text>
      <text x={left} y={H - 6} fontSize={11} fill="var(--viz-ink-2)">
        Cumulative disk writes over the same {fmtNum(s.ops)} operations
      </text>
    </svg>
  );
}

function CowFigure({
  run,
  plan,
  phase,
  crashed,
  reader,
  states,
  metas,
  onDragReader,
  W,
}: {
  run: CowRun;
  plan: CommitPlan | null;
  phase: number;
  crashed: boolean;
  reader: Reader;
  states: PageState[];
  metas: MetaView[];
  onDragReader: (txn: number) => void;
  W: number;
}) {
  const tip = useTip();
  const dragging = useRef(false);
  const showPlan = plan && phase >= 1 && !crashed;
  const current: Snapshot = showPlan ? plan.next : run.snapshots[run.txn];
  const inflightPages = new Set(showPlan ? plan.entries.map((e) => e.pgno) : []);
  const T = run.txn + 1;
  const readerOpen = !crashed && readerPresent(reader, T);
  const readerSnap = readerOpen ? run.snapshots[reader.start] : null;
  const currentPages = new Set(snapshotPages(current));

  const metaY = 8;
  const treeTop = 78;
  const timelineTop = 252;
  const stripTop = timelineTop + 58;
  const cols = 30;
  const cellW = (W - 20) / cols;
  const rows = Math.max(1, Math.ceil(states.length / cols));
  const H = stripTop + 22 + rows * 26 + 6;
  const tlLeft = 60;
  const tlRight = W - 20;
  const tx = (t: number) => tlLeft + ((t - 1) / (MAX_TXN - 1)) * (tlRight - tlLeft);

  const txnFromClient = (e: React.PointerEvent<SVGRectElement>) => {
    const svg = (e.currentTarget as SVGRectElement).ownerSVGElement;
    if (!svg) return reader.start;
    const box = svg.getBoundingClientRect();
    const x = ((e.clientX - box.left) / box.width) * W;
    if (x < tlLeft - 10) return 0;
    const t = Math.round(1 + ((x - tlLeft) / (tlRight - tlLeft)) * (MAX_TXN - 1));
    return Math.max(1, Math.min(run.txn, t));
  };

  const tree = (snap: Snapshot, x0: number, width: number, title: string, mode: 'current' | 'reader') => {
    const leafSlot = width / COW_LEAVES;
    const leafX = (i: number) => x0 + i * leafSlot + leafSlot / 2;
    const branchX = (b: number) => leafX(b * 3 + 1);
    const rootX = x0 + width / 2;
    const yRoot = treeTop + 22;
    const yBranch = treeTop + 70;
    const yLeaf = treeTop + 118;
    const stroke = (pg: number) => {
      if (mode === 'current') return inflightPages.has(pg) ? STATE_COLOR.inflight : STATE_COLOR.live;
      return currentPages.has(pg) ? STATE_COLOR.live : STATE_COLOR.pinned;
    };
    const box = (pg: number, cx: number, cy: number, w: number, label: string) => (
      <g key={`${mode}${label}`} {...tip(<><strong>Page {pg}</strong> — {label}<br />{mode === 'reader' ? (currentPages.has(pg) ? 'shared with the newest tree' : 'only the reader can reach it: cannot be freed') : inflightPages.has(pg) ? 'copied by the commit in progress' : 'live in the newest tree'}</>)}>
        <rect x={cx - w / 2} y={cy - 11} width={w} height={22} rx={3} fill="var(--viz-surface)" stroke={stroke(pg)} strokeWidth={2.5} />
        <text x={cx} y={cy + 4} textAnchor="middle" fontSize={11} fill="var(--viz-ink)">
          {pg}
        </text>
      </g>
    );
    return (
      <g>
        <text x={x0} y={treeTop + 2} fontSize={11} fill="var(--viz-ink-2)">
          {title}
        </text>
        {snap.branches.map((_, b) => (
          <line key={`rb${b}`} x1={rootX} y1={yRoot + 11} x2={branchX(b)} y2={yBranch - 11} stroke="var(--viz-grid)" strokeWidth={1.5} />
        ))}
        {snap.leaves.map((_, l) => (
          <line key={`bl${l}`} x1={branchX(Math.floor(l / 3))} y1={yBranch + 11} x2={leafX(l)} y2={yLeaf - 11} stroke="var(--viz-grid)" strokeWidth={1.5} />
        ))}
        {box(snap.root, rootX, yRoot, 44, 'root')}
        {snap.branches.map((pg, b) => box(pg, branchX(b), yBranch, 40, `branch ${b}`))}
        {snap.leaves.map((pg, l) => box(pg, leafX(l), yLeaf, Math.min(32, leafSlot - 4), `leaf ${l}`))}
      </g>
    );
  };

  const half = (W - 30) / 2;

  return (
    <svg viewBox={`0 0 ${W} ${H}`} width={W} height={H} role="img" aria-label={`Copy-on-write B+tree after txn ${run.txn}; data file of ${states.length} pages`}>
      {metas.map((m, i) => {
        const x = 10 + i * 200;
        const dashed = m.status === 'writing' || m.status === 'torn' || m.status === 'unwritten';
        const color = m.status === 'torn' ? STATE_COLOR.garbage : m.status === 'writing' ? STATE_COLOR.inflight : STATE_COLOR.meta;
        const statusText =
          m.status === 'newest'
            ? crashed
              ? 'highest valid txn: opened'
              : 'newest: new readers start here'
            : m.status === 'older'
              ? 'previous commit'
              : m.status === 'writing'
                ? 'being written now'
                : m.status === 'torn'
                  ? 'torn: checksum fails, ignored'
                  : 'write never landed';
        return (
          <g key={`meta${m.slot}`}>
            <rect x={x} y={metaY} width={188} height={56} rx={4} fill="var(--viz-surface)" stroke={color} strokeWidth={m.status === 'newest' ? 3 : 2} strokeDasharray={dashed ? '5 3' : undefined} />
            <text x={x + 8} y={metaY + 16} fontSize={11} fill="var(--viz-ink)" fontWeight={600}>
              Meta page {m.slot}
            </text>
            <text x={x + 8} y={metaY + 32} fontSize={11} fill="var(--viz-ink)">
              {m.txn <= 0 ? 'txn 0 · empty tree' : `txn ${m.txn} · root page ${m.root}`}
            </text>
            <text x={x + 8} y={metaY + 48} fontSize={10} fill="var(--viz-ink-2)">
              {statusText}
            </text>
          </g>
        );
      })}
      <text x={420} y={metaY + 16} fontSize={11} fill="var(--viz-ink-2)">
        {crashed ? 'Crashed and reopened' : showPlan ? `Committing txn ${plan.txn}: step ${phase} of 4` : `Txn ${run.txn} committed`}
      </text>
      <text x={420} y={metaY + 32} fontSize={11} fill="var(--viz-ink-2)">
        {run.engine === 'lmdb' ? 'LMDB: no meta checksum (0.9)' : 'bbolt: FNV-64a meta checksum'}
      </text>

      {tree(current, 10, half, showPlan ? `Tree being committed (txn ${plan.txn})` : `Newest tree (txn ${run.txn})`, 'current')}
      {readerSnap ? (
        tree(readerSnap, 20 + half, half, `Oldest open reader's snapshot (txn ${reader.start})`, 'reader')
      ) : (
        <text x={20 + half} y={treeTop + 2} fontSize={11} fill="var(--viz-ink-2)">
          {crashed ? 'The crash ended every reader.' : 'No reader open: drag the handle below to open one.'}
        </text>
      )}

      <text x={10} y={stripTop + 12} fontSize={11} fill="var(--viz-ink-2)">
        Data file: {states.length} pages (cell = page number)
      </text>
      {states.map((st, pg) => {
        const col = pg % cols;
        const row = Math.floor(pg / cols);
        const x = 10 + col * cellW;
        const y = stripTop + 20 + row * 26;
        const label =
          st === 'meta' ? 'meta page' : st === 'live' ? 'live in the newest tree' : st === 'inflight' ? 'written by the commit in progress' : st === 'garbage' ? 'written by the crashed commit; unreferenced' : st === 'pinned' ? 'old page the open reader still reads' : st === 'held' ? 'freed, but the engine will not reuse it yet' : 'free: the next commit may reuse it';
        return (
          <g key={`pg${pg}`} {...tip(<><strong>Page {pg}</strong><br />{label}</>)}>
            <rect x={x + 1} y={y} width={cellW - 2} height={22} rx={2} fill="var(--viz-surface)" stroke={STATE_COLOR[st]} strokeWidth={st === 'free' ? 1.2 : 2} strokeDasharray={st === 'garbage' ? '3 2' : undefined} />
            <rect x={x + 3} y={y + 16} width={cellW - 6} height={4} fill={STATE_COLOR[st]} opacity={st === 'free' || st === 'garbage' ? 0.5 : 1} />
            <text x={x + cellW / 2} y={y + 12} textAnchor="middle" fontSize={9} fill="var(--viz-ink)">
              {st === 'meta' ? `M${pg}` : pg}
            </text>
          </g>
        );
      })}

      <text x={10} y={timelineTop + 26} fontSize={11} fill="var(--viz-ink-2)">
        Commits
      </text>
      <line x1={tlLeft} x2={tlRight} y1={timelineTop + 22} y2={timelineTop + 22} stroke="var(--viz-axis)" />
      {Array.from({ length: MAX_TXN }, (_, i) => i + 1).map((t) => (
        <circle key={`t${t}`} cx={tx(t)} cy={timelineTop + 22} r={t <= run.txn ? 4 : 3} fill={t <= run.txn ? 'var(--viz-ink-2)' : 'var(--viz-surface)'} stroke="var(--viz-ink-muted)" />
      ))}
      <text x={tx(1)} y={timelineTop + 44} fontSize={10} textAnchor="middle" fill="var(--viz-ink-2)">
        1
      </text>
      <text x={tx(run.txn)} y={timelineTop + 44} fontSize={10} textAnchor="middle" fill="var(--viz-ink)">
        {run.txn}
      </text>
      {readerOpen ? (
        <g>
          <line x1={tx(reader.start)} x2={tx(reader.start)} y1={timelineTop + 4} y2={timelineTop + 34} stroke="var(--viz-warning)" strokeWidth={3} />
          <path d={`M ${tx(reader.start) - 7} ${timelineTop} L ${tx(reader.start) + 7} ${timelineTop} L ${tx(reader.start)} ${timelineTop + 9} Z`} fill="var(--viz-warning)" />
          <text x={Math.min(tlRight - 60, tx(reader.start) + 10)} y={timelineTop + 8} fontSize={11} fill="var(--viz-ink)">
            oldest reader (txn {reader.start})
          </text>
        </g>
      ) : (
        <text x={tlLeft} y={timelineTop + 8} fontSize={11} fill="var(--viz-ink-2)">
          drag along the commits to open a long-running reader
        </text>
      )}
      <rect
        x={tlLeft - 44}
        y={timelineTop - 4}
        width={tlRight - tlLeft + 58}
        height={46}
        fill="transparent"
        style={{ cursor: 'ew-resize', touchAction: 'none' }}
        onPointerDown={(e) => {
          dragging.current = true;
          (e.currentTarget as SVGRectElement).setPointerCapture?.(e.pointerId);
          onDragReader(txnFromClient(e));
        }}
        onPointerMove={(e) => {
          if (dragging.current) onDragReader(txnFromClient(e));
        }}
        onPointerUp={() => {
          dragging.current = false;
        }}
        onPointerCancel={() => {
          dragging.current = false;
        }}
      />
    </svg>
  );
}

export default function BepsilonCowTreeLab() {
  const [mode, setMode] = useState<'betree' | 'cow'>('betree');

  // Bε-tree
  const [workload, setWorkload] = useState<Workload>('random');
  const [nodeSize, setNodeSize] = useState(64);
  const [fanout, setFanout] = useState(8);
  const [ops, setOps] = useState(446);
  const [pick, setPick] = useState<LookupPick>('busiest');
  const cfg = useMemo<BeConfig>(() => ({ nodeSize, fanout, workload }), [nodeSize, fanout, workload]);
  const full = useMemo(() => simulateBe(cfg, MAX_OPS), [cfg]);
  const be = useMemo(() => simulateBe(cfg, ops), [cfg, ops]);
  const lookupKey = pickLookupKey(be, LOOKUP_OPTIONS[workload].some((o) => o.value === pick) ? pick : 'busiest');
  const look = lookupBe(be, lookupKey);
  const nextEvent = (kind: 1 | 2) => {
    for (let i = ops + 1; i <= MAX_OPS; i++) if (full.eventKind[i] >= kind) return i;
    return MAX_OPS;
  };

  // copy-on-write
  const [engine, setEngine] = useState<Engine>('lmdb');
  const [keys, setKeys] = useState(1);
  const [txn, setTxn] = useState(8);
  const [reader, setReader] = useState<Reader>({ start: 3, end: Infinity });
  const [phase, setPhase] = useState(0);
  const [crashed, setCrashed] = useState(false);
  const run = useMemo(() => runCow(engine, keys, txn, reader), [engine, keys, txn, reader]);
  const plan = useMemo(() => (phase >= 1 && run.txn < MAX_TXN ? planCommit(run, reader) : null), [run, reader, phase]);
  const states = classifyPages(run, reader, plan, phase, crashed);
  const counts = countStates(states);
  const metas = metaView(run, plan, phase, crashed);
  const lastCommit = run.commits[run.txn];
  const readerOpenNow = !crashed && readerPresent(reader, run.txn + 1);

  const setReaderStart = (t: number) => {
    const start = Math.max(0, Math.min(run.txn, t));
    if (start !== reader.start || reader.end !== Infinity) setReader({ start, end: Infinity });
  };
  const resetCow = () => {
    setPhase(0);
    setCrashed(false);
  };

  const W = 680;

  if (mode === 'cow') {
    const writtenThisCommit = plan ? plan.entries.length + 1 : lastCommit.copied + 1;
    return (
      <VizPanel
        title="Copy-on-write B+tree: commit by flipping a meta page"
        subtitle="Every commit copies the changed leaf and its ancestors to fresh pages, syncs them, then writes one of two meta pages. Step through a commit and crash it anywhere; drag the oldest reader back in time and watch which old pages the engine refuses to reuse."
        controls={
          <>
            <Segmented label="Structure" value={mode} onChange={setMode} options={[{ value: 'betree', label: 'Bε-tree buffers' }, { value: 'cow', label: 'Copy-on-write tree' }]} />
            <Segmented
              label="Engine"
              value={engine}
              onChange={(v) => {
                setEngine(v);
                resetCow();
              }}
              options={[
                { value: 'lmdb', label: 'LMDB' },
                { value: 'bbolt', label: 'bbolt' },
              ]}
            />
            <Segmented
              label="Keys changed per commit"
              value={String(keys) as '1' | '3'}
              onChange={(v) => {
                setKeys(Number(v));
                resetCow();
              }}
              options={[
                { value: '1', label: '1' },
                { value: '3', label: '3' },
              ]}
            />
            <Slider label="Oldest reader started at txn (0 = none)" min={0} max={run.txn} value={readerOpenNow ? reader.start : 0} onChange={(v) => { resetCow(); setReaderStart(v); }} />
          </>
        }
        legend={
          <Legend
            items={[
              { label: 'Live page (newest tree)', color: STATE_COLOR.live },
              { label: 'Copied by the commit in progress', color: STATE_COLOR.inflight },
              { label: 'Meta page', color: STATE_COLOR.meta },
              { label: 'Old page the open reader still reads', color: STATE_COLOR.pinned },
              { label: 'Freed, held back by the engine’s rule', color: STATE_COLOR.held },
              { label: 'Free for reuse (dashed: crashed commit’s pages or torn meta page)', color: STATE_COLOR.free },
            ]}
          />
        }
        stats={
          <Stats
            items={[
              { label: plan ? `Page writes, txn ${plan.txn}` : `Page writes, txn ${run.txn}`, value: `${writtenThisCommit}`, hint: 'Copied data pages plus the meta page. Real LMDB and bbolt also rewrite their free-page structure on every commit.' },
              { label: 'Data file', value: `${states.length} pages` },
              { label: 'Read by the old reader', value: fmtNum(counts.pinned) },
              { label: 'Held by the rule', value: fmtNum(counts.held), hint: 'Freed pages no reader needs, which the engine will still not hand to the next commit.' },
              { label: 'Free for next commit', value: fmtNum(counts.free + counts.garbage) },
            ]}
          />
        }
        note={
          <Note>
            {crashed ? (
              <>
                <strong>Crash during step {phase}, then reopen.</strong>{' '}
                {phase === 4 && engine === 'bbolt'
                  ? `The half-written meta page ${plan ? plan.txn % 2 : 0} fails its checksum, so bbolt opens the other one: txn ${run.txn}. `
                  : phase === 4
                    ? `The meta write for txn ${plan ? plan.txn : ''} never reached the disk, so meta page ${run.txn % 2} still has the highest txnid: LMDB opens txn ${run.txn}. `
                    : `No meta page mentions txn ${plan ? plan.txn : ''}, so the file opens at txn ${run.txn} from meta page ${run.txn % 2}. `}
                {phase === 1
                  ? `Its ${plan ? plan.entries.length : 0} copied pages existed only in memory, so nothing reached the file. `
                  : `Its ${plan ? plan.entries.length : 0} copied pages are unreferenced: they sit on pages the reopened version counts as free or past its last page. `}
                There is no log to replay and nothing to undo. The crash also ended the reader.
              </>
            ) : plan ? (
              <>
                <strong>Step {phase}: </strong>
                {PHASES[phase - 1]} {phase < 4 ? `A crash now reopens at txn ${run.txn}.` : engine === 'bbolt' ? 'If this write tears, the checksum exposes it and bbolt falls back to the other meta page.' : 'LMDB 0.9 rewrites only the last 120 bytes of the meta structure here, with no checksum.'}
              </>
            ) : (
              <>
                <strong>Txn {run.txn} is committed on meta page {run.txn % 2}.</strong> Changing {keys === 1 ? 'one key' : 'three keys'} copied {lastCommit.copied} pages ({lastCommit.reused} reused, {lastCommit.extended} appended) plus the meta page.{' '}
                {readerOpenNow
                  ? `The reader that started at txn ${reader.start} still reads ${counts.pinned} old page${counts.pinned === 1 ? '' : 's'}, and ${engine === 'lmdb' ? `LMDB holds ${counts.held} more that no one reads, because they were freed by txn ${reader.start} or later` : `bbolt holds ${counts.held} more, and reuses pages that were both allocated and freed after the reader started`}. The file is ${states.length} pages.`
                  : engine === 'lmdb'
                    ? `No reader is open, yet LMDB still holds the ${counts.held} pages txn ${run.txn} freed: they belong to txn ${run.txn - 1}'s tree, which meta page ${(run.txn - 1) % 2} still points at.`
                    : `No reader is open, so bbolt will reuse every freed page, including the ${lastCommit.copied} that txn ${run.txn} just replaced.`}
              </>
            )}
          </Note>
        }
        table={
          <table className="viz-table">
            <thead>
              <tr>
                <th>Txn</th>
                <th>Leaves changed</th>
                <th>New root page</th>
                <th>Pages copied</th>
                <th>Reused / appended</th>
                <th>File after (pages)</th>
              </tr>
            </thead>
            <tbody>
              {run.commits.slice(1).map((c) => (
                <tr key={c.txn}>
                  <td>{c.txn}</td>
                  <td>{c.txn === 1 ? 'initial load' : c.leaves.join(', ')}</td>
                  <td>{c.root}</td>
                  <td>{c.copied}</td>
                  <td>
                    {c.reused} / {c.extended}
                  </td>
                  <td>{c.fileEnd}</td>
                </tr>
              ))}
            </tbody>
          </table>
        }
      >
        <div className="viz-controls">
          <Button primary onClick={() => { resetCow(); setTxn((t) => Math.min(MAX_TXN, t + 1)); }} disabled={phase > 0 || run.txn >= MAX_TXN}>
            Commit
          </Button>
          <Button onClick={() => { setCrashed(false); setPhase(1); }} disabled={phase > 0 || run.txn >= MAX_TXN}>
            Step through a commit
          </Button>
          <Button
            onClick={() => {
              if (phase >= 4) {
                resetCow();
                setTxn((t) => Math.min(MAX_TXN, t + 1));
              } else setPhase((p) => p + 1);
            }}
            disabled={phase === 0 || crashed}
          >
            {phase >= 4 ? 'Finish commit' : 'Next step'}
          </Button>
          <Button onClick={() => setCrashed(true)} disabled={phase === 0 || crashed}>
            Crash now
          </Button>
          <Button
            onClick={() => {
              if (reader.start >= 1) setReader({ start: reader.start, end: run.txn });
              resetCow();
            }}
            disabled={!crashed}
          >
            Continue after reopen
          </Button>
          <Button
            onClick={() => {
              resetCow();
              setEngine('lmdb');
              setKeys(1);
              setTxn(8);
              setReader({ start: 3, end: Infinity });
            }}
            title="Back to the opening state: LMDB, one key per commit, txn 8, reader from txn 3"
          >
            Reset
          </Button>
        </div>
        <TooltipHost>
          <CowFigure run={run} plan={plan} phase={phase} crashed={crashed} reader={reader} states={states} metas={metas} onDragReader={(t) => { resetCow(); setReaderStart(t); }} W={W} />
        </TooltipHost>
        {plan ? (
          <ol style={{ margin: '0.5rem 0 0', paddingLeft: '1.2rem', fontSize: '0.8rem', display: 'grid', gap: 4 }}>
            {PHASES.map((p, i) => (
              <li key={i} style={{ color: i + 1 <= phase ? 'var(--viz-ink)' : 'var(--viz-ink-muted)', borderLeft: i + 1 === phase ? `3px ${crashed ? 'dashed var(--viz-ink)' : 'solid var(--viz-dirty)'}` : '3px solid transparent', paddingLeft: 6 }}>
                {p}
              </li>
            ))}
          </ol>
        ) : null}
      </VizPanel>
    );
  }

  const eps = epsilonOf(cfg);
  const perOp = be.ops ? be.nodeWrites / be.ops : 0;
  const ev = be.lastEvent;
  const buffered = bufferedTotal(be);
  const levelName = (d: number, index: number) => (d === 0 ? 'root' : `level-${d} node ${index}`);

  return (
    <VizPanel
      title="Bε-tree: buffer messages, flush them in batches"
      subtitle="Every insert, upsert and delete is a message dropped into the root's buffer. A full buffer flushes the batch bound for its busiest child, which may overflow and flush in turn. Race the disk writes against a B+tree taking the same operations, then look up a key through the pending messages."
      controls={
        <>
          <Segmented label="Structure" value={mode} onChange={setMode} options={[{ value: 'betree', label: 'Bε-tree buffers' }, { value: 'cow', label: 'Copy-on-write tree' }]} />
          <Choice
            label="Workload"
            value={workload}
            onChange={(v) => {
              setWorkload(v);
              setPick('busiest');
            }}
            options={[
              { value: 'random', label: 'Random inserts' },
              { value: 'hot', label: 'Upserts to 16 hot counters' },
              { value: 'mixed', label: 'Mixed: inserts, upserts, deletes' },
            ]}
          />
          <Segmented label="Node size B (messages)" value={String(nodeSize) as '64' | '256'} onChange={(v) => setNodeSize(Number(v))} options={[{ value: '64', label: '64' }, { value: '256', label: '256' }]} />
          <Segmented label="Fanout" value={String(fanout) as '2' | '4' | '8'} onChange={(v) => setFanout(Number(v))} options={[{ value: '2', label: '2' }, { value: '4', label: '4' }, { value: '8', label: '8' }]} />
          <Slider label="Operations applied" min={0} max={MAX_OPS} value={ops} onChange={setOps} format={fmtNum} />
          <Choice label="Look up" value={LOOKUP_OPTIONS[workload].some((o) => o.value === pick) ? pick : 'busiest'} onChange={setPick} options={LOOKUP_OPTIONS[workload]} />
        </>
      }
      legend={
        <Legend
          items={[
            { label: 'Insert message', color: MSG_COLOR.insert },
            { label: 'Upsert message', color: MSG_COLOR.upsert },
            { label: 'Tombstone (delete) message', color: MSG_COLOR.delete },
            { label: 'Bε-tree node writes', color: 'var(--viz-1)', shape: 'line' },
            { label: 'B+tree leaf writes', color: 'var(--viz-2)', shape: 'line' },
          ]}
        />
      }
      stats={
        <Stats
          items={[
            { label: 'ε = log_B(fanout)', value: eps.toFixed(2), hint: `Internal nodes use ${fanout} of ${nodeSize} slots for pivots and ${be.cap} for the buffer.` },
            { label: 'Disk writes: Bε vs B+tree', value: `${fmtNum(be.nodeWrites)} vs ${fmtNum(be.ops)}`, hint: 'Model: a B+tree insert into a tree far larger than memory writes one leaf; a Bε flush writes each node it changed once. Splits and checkpoint coalescing are not modelled.' },
            { label: 'Writes per operation', value: be.ops ? `${perOp.toFixed(2)} vs 1` : '—' },
            { label: 'Messages still buffered', value: fmtNum(buffered) },
            { label: 'Lookup reads: Bε vs B+tree', value: `${look.nodesRead} vs ${look.bplusReads} nodes`, hint: 'Same node size. The B+tree spends the whole node on pivots, so it is shallower.' },
          ]}
        />
      }
      note={
        <Note>
          {be.ops === 0 ? (
            <>No operations yet: every buffer is empty. Press <strong>+10 ops</strong> or drag the slider. </>
          ) : ev ? (
            <>
              <strong>Op #{fmtNum(ev.op)} overflowed the root.</strong>{' '}
              {ev.edges.map((e, i) => `${i === 0 ? 'The' : 'then'} ${levelName(e.level, e.index)} flushed ${e.moved} messages to ${e.toLeaf ? `leaf ${e.child}` : `its child ${e.child}`}`).join(', ')}
              {` — ${ev.written.length} node writes for this operation. `}
            </>
          ) : (
            <>
              <strong>Op #{fmtNum(be.ops)} ({be.lastOp ? MSG_LABEL[be.lastOp.type] : ''} on key {be.lastOp?.key}) just joined the root buffer</strong> ({be.buffers[0][0].length} of {be.cap} slots): no disk write. {' '}
            </>
          )}
          So far {fmtNum(be.flushes)} flushes, {fmtNum(be.leafFlushes)} of them into leaves, have written {fmtNum(be.nodeWrites)} nodes; a B+tree would have written {fmtNum(be.ops)} leaves.
        </Note>
      }
      table={
        <table className="viz-table">
          <thead>
            <tr>
              <th>Level</th>
              <th>Nodes</th>
              <th>Buffer capacity</th>
              <th>Messages buffered</th>
              <th>Flushes out</th>
              <th>Messages per flush</th>
              <th>Node writes</th>
            </tr>
          </thead>
          <tbody>
            {be.buffers.map((lvl, d) => (
              <tr key={d}>
                <td>{d === 0 ? 'root' : `level ${d}`}</td>
                <td>{lvl.length}</td>
                <td>{be.cap}</td>
                <td>{fmtNum(lvl.reduce((a, b) => a + b.length, 0))}</td>
                <td>{fmtNum(be.flushesFromLevel[d])}</td>
                <td>{be.flushesFromLevel[d] ? (be.movedFromLevel[d] / be.flushesFromLevel[d]).toFixed(1) : '—'}</td>
                <td>{fmtNum(be.writesByLevel[d])}</td>
              </tr>
            ))}
            <tr>
              <td>leaves</td>
              <td>{LEAVES}</td>
              <td>—</td>
              <td>—</td>
              <td>—</td>
              <td>—</td>
              <td>{fmtNum(be.writesByLevel[be.levels])}</td>
            </tr>
          </tbody>
        </table>
      }
    >
      <div className="viz-controls">
        <Button onClick={() => setOps((n) => Math.min(MAX_OPS, n + 1))} disabled={ops >= MAX_OPS}>
          +1 op
        </Button>
        <Button onClick={() => setOps((n) => Math.min(MAX_OPS, n + 10))} disabled={ops >= MAX_OPS}>
          +10 ops
        </Button>
        <Button primary onClick={() => setOps(nextEvent(1))} disabled={ops >= MAX_OPS}>
          Run to next flush
        </Button>
        <Button onClick={() => setOps(nextEvent(2))} disabled={ops >= MAX_OPS}>
          Run to next leaf flush
        </Button>
        <Button
          onClick={() => {
            setOps(446);
            setNodeSize(64);
            setFanout(8);
            setWorkload('random');
            setPick('busiest');
          }}
          disabled={ops === 446 && nodeSize === 64 && fanout === 8 && workload === 'random' && pick === 'busiest'}
          title="Back to the opening state: operation 446, node size 64, fanout 8, random inserts"
        >
          Reset
        </Button>
      </div>
      <TooltipHost>
        <BeFigure s={be} look={look} W={W} />
      </TooltipHost>
      <div style={{ fontSize: '0.8rem', color: 'var(--viz-ink)', margin: '0.25rem 0 0' }}>
        <strong>Lookup key {look.key}</strong> (leaf {look.leaf}): reads {look.nodesRead} nodes; a B+tree with the same node size reads {look.bplusReads}.{' '}
        {look.path
          .filter((p) => p.msgs.length)
          .map((p) => `${p.level === 0 ? 'The root' : `Level-${p.level} node ${p.index}`} buffers ${summarizeMsgs(p.msgs)} for it.`)
          .join(' ')}{' '}
        Leaf says {fmtVal(look.leafValue)}
        {look.pending.length ? `; applying ${look.pending.length} pending message${look.pending.length === 1 ? '' : 's'} in order gives ${look.result === undefined ? 'NOT FOUND' : fmtNum(look.result)}.` : ' and nothing is pending, so that is the answer.'}
      </div>
      <RaceChart s={be} W={W} />
    </VizPanel>
  );
}
