import { useDeferredValue, useMemo, useState } from 'react';
import { VizPanel, Segmented, Choice, Button, Legend, Stats, Note, makeRng, fmtBytes, fmtNum } from './Viz';

/**
 * A list endpoint over a multi-tenant `tickets` table, graded shape by shape against a candidate index set.
 *
 * What is executed, not estimated:
 * - A deterministic 100,000-row table (10 tenants; tenant 1 holds ~20%), inserted in created_at order,
 *   20 rows per 8 KB heap page, so heap block = floor(row / 20).
 * - Every candidate index is materialized as its real sort order (per-column ASC/DESC, default NULL
 *   placement, heap TID as the final tiebreaker) and every plan is run against it:
 *   ordered index scans (forward or backward, stopping at LIMIT, or an Incremental Sort when the index
 *   delivers only a prefix of the ORDER BY), bitmap heap scans (one index, or a BitmapAnd of two) followed
 *   by a sort, and a sequential scan.
 * - Buffers follow PostgreSQL's access pattern: the descent pages above the leaf, each leaf page walked,
 *   and a heap buffer read each time the fetched TID's block differs from the previous one
 *   (heapam_index_fetch_tuple). A bitmap heap scan reads each distinct heap block once.
 *
 * Planner rules taken from PostgreSQL:
 * - Equality on leading key columns bounds the scan; equality on later key columns is checked inside the
 *   index; everything else is a Filter applied after the heap fetch.
 * - A column fixed by an equality clause drops out of the ordering comparison (redundant pathkey). A
 *   backward scan inverts every column, so an ORDER BY with mixed directions needs an index declared that way.
 * - A partial index is usable only if the query's WHERE implies its predicate at planning time, so
 *   `WHERE status = 'open'` is unusable by a generic plan whose clause is `status = $2`.
 * - `(col = $n OR $n IS NULL)`: a custom plan folds the parameter (PARAM_FLAG_CONST + eval_const_expressions),
 *   leaving `col = value` or nothing. A generic plan keeps the OR, which no index arm can match, so it is a
 *   heap Filter, and one plan must serve every value of $n.
 *
 * Model assumptions (labelled in the lab):
 * - The planner is an oracle: it picks the path with the fewest buffers (ties: fewer rows sorted). A generic
 *   plan gets the best case — the single plan with the fewest buffers summed over the shapes it serves.
 * - No skip scan (PostgreSQL 18 can skip a low-cardinality gap column), no index-only scans (the endpoint
 *   returns columns no index holds), no deduplication, freshly built indexes (leaf fillfactor 90, internal 70).
 * - Write mix per 1,000 writes: 200 inserts, 800 updates; every update sets updated_at, 25% change status,
 *   10% reassign, 5% change priority; pages have room for HOT. A non-HOT update adds an entry to every index;
 *   updates to open tickets move a quarter of them out of `status = 'open'`.
 */

/* ------------------------------------------------------------------ table */

export const N_ROWS = 100_000;
export const ROWS_PER_PAGE = 20;
export const HEAP_PAGES = N_ROWS / ROWS_PER_PAGE;
export const LIMIT = 25;
export const TENANT = 1;
export const OPEN = 0;
export const ME = 107; // tenant 1, agent 7
export const ACME = 10_000; // tenant 1, customer 0
const NULL_VAL = 1 << 30; // NULL sorts after every value: ASC NULLS LAST, DESC NULLS FIRST
const EPOCH_MS = Date.UTC(2025, 0, 1);
const TICK_MS = 5 * 60 * 1000; // created_at advances ~5 minutes per row

export type Col = 'tenant_id' | 'status' | 'assignee_id' | 'customer_id' | 'priority' | 'created_at' | 'updated_at' | 'id';
export const KEY_COLS: Col[] = ['tenant_id', 'status', 'assignee_id', 'customer_id', 'priority', 'created_at', 'updated_at', 'id'];
const WIDTH: Record<Col, [number, number]> = {
  // [bytes, alignment]
  tenant_id: [8, 8],
  status: [4, 4], // enum
  assignee_id: [8, 8],
  customer_id: [8, 8],
  priority: [2, 2], // smallint
  created_at: [8, 8],
  updated_at: [8, 8],
  id: [8, 8],
};

type Table = Record<Col, Int32Array> & { deleted: Uint8Array };
let TABLE: Table | null = null;

export function table(): Table {
  if (TABLE) return TABLE;
  const rng = makeRng(20260914);
  const t = {
    tenant_id: new Int32Array(N_ROWS),
    status: new Int32Array(N_ROWS),
    assignee_id: new Int32Array(N_ROWS),
    customer_id: new Int32Array(N_ROWS),
    priority: new Int32Array(N_ROWS),
    created_at: new Int32Array(N_ROWS),
    updated_at: new Int32Array(N_ROWS),
    id: new Int32Array(N_ROWS),
    deleted: new Uint8Array(N_ROWS),
  };
  for (let i = 0; i < N_ROWS; i++) {
    const tenant = rng() < 0.2 ? 1 : 2 + Math.floor(rng() * 9);
    t.tenant_id[i] = tenant;
    const pOpen = 0.1 + 0.3 * (i / N_ROWS); // newer tickets are more likely still open
    const s = rng();
    t.status[i] = s < pOpen ? 0 : s < pOpen + 0.15 ? 1 : 2;
    t.deleted[i] = rng() < 0.04 ? 1 : 0;
    const a = rng();
    t.assignee_id[i] = a < 0.15 ? NULL_VAL : a < 0.18 ? tenant * 100 + 7 : tenant * 100 + [0, 1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20][Math.floor(rng() * 20)];
    const c = rng();
    t.customer_id[i] = tenant * 10_000 + (c < 0.02 ? 0 : 1 + Math.floor(rng() * 1999));
    const p = rng();
    t.priority[i] = p < 0.05 ? 4 : p < 0.2 ? 3 : p < 0.7 ? 2 : 1;
    t.created_at[i] = i;
    t.updated_at[i] = i + Math.floor(rng() * rng() * 30_000);
    t.id[i] = i + 1;
  }
  TABLE = t;
  return t;
}

export const heapBlock = (row: number) => Math.floor(row / ROWS_PER_PAGE);

/* ------------------------------------------------------------ indexes */

export type PartialPred = 'none' | 'live' | 'live_open';
export const PARTIAL_SQL: Record<PartialPred, string> = {
  none: '',
  live: 'WHERE deleted_at IS NULL',
  live_open: "WHERE deleted_at IS NULL AND status = 'open'",
};
export type IndexCol = { col: Col; desc: boolean };
export type IndexDef = { name: string; cols: IndexCol[]; partial: PartialPred };

const sig = (ix: IndexDef) => `${ix.cols.map((c) => `${c.col}${c.desc ? '-' : '+'}`).join(',')}|${ix.partial}`;

const inPartial = (t: Table, r: number, p: PartialPred) => (p === 'none' ? true : p === 'live' ? !t.deleted[r] : !t.deleted[r] && t.status[r] === OPEN);

const SORTED = new Map<string, Int32Array>();
/** The index's entries in key order: row numbers sorted by (cols with directions, heap TID). */
export function sortedEntries(ix: IndexDef): Int32Array {
  const key = sig(ix);
  const hit = SORTED.get(key);
  if (hit) return hit;
  const t = table();
  const rows: number[] = [];
  for (let r = 0; r < N_ROWS; r++) if (inPartial(t, r, ix.partial)) rows.push(r);
  const arrs = ix.cols.map((c) => t[c.col]);
  const desc = ix.cols.map((c) => c.desc);
  rows.sort((a, b) => {
    for (let j = 0; j < arrs.length; j++) {
      const va = arrs[j][a];
      const vb = arrs[j][b];
      if (va !== vb) return desc[j] ? vb - va : va - vb;
    }
    return a - b;
  });
  const out = Int32Array.from(rows);
  if (SORTED.size > 24) SORTED.clear();
  SORTED.set(key, out);
  return out;
}

export function indexGeometry(ix: IndexDef) {
  let off = 0;
  for (const c of ix.cols) {
    const [w, al] = WIDTH[c.col];
    off = Math.ceil(off / al) * al + w;
  }
  const tuple = Math.ceil((8 + off) / 8) * 8; // IndexTupleData header + MAXALIGN
  const item = tuple + 4; // + line pointer
  const usable = 8192 - 24 - 16; // page header, BTPageOpaque
  const leafCap = Math.max(1, Math.floor((usable - 819) / item)); // fillfactor 90
  const fanout = Math.max(2, Math.floor((usable - 2457) / item)); // internal fillfactor 70
  const entries = sortedEntries(ix).length;
  const leaves = Math.max(1, Math.ceil(entries / leafCap));
  let internal = 0;
  let height = 0;
  for (let level = leaves; level > 1; level = Math.ceil(level / fanout)) {
    height++;
    internal += Math.ceil(level / fanout);
  }
  return { leafCap, fanout, entries, leaves, height, pages: leaves + internal + 1, bytes: (leaves + internal + 1) * 8192 };
}

/* ------------------------------------------------------------ the query */

export type SortId = 'newest' | 'oldest' | 'updated' | 'prio_newest' | 'prio_queue';
export type SortKey = { col: Col; desc: boolean };
export const SORTS: { id: SortId; label: string; short: string; keys: SortKey[] }[] = [
  { id: 'newest', label: 'Newest first', short: 'newest', keys: [{ col: 'created_at', desc: true }, { col: 'id', desc: true }] },
  { id: 'oldest', label: 'Oldest first', short: 'oldest', keys: [{ col: 'created_at', desc: false }, { col: 'id', desc: false }] },
  { id: 'updated', label: 'Recently updated', short: 'updated', keys: [{ col: 'updated_at', desc: true }, { col: 'id', desc: true }] },
  { id: 'prio_newest', label: 'Priority, newest first', short: 'priority ↓ newest', keys: [{ col: 'priority', desc: true }, { col: 'created_at', desc: true }, { col: 'id', desc: true }] },
  { id: 'prio_queue', label: 'Priority, oldest first', short: 'priority ↓ oldest', keys: [{ col: 'priority', desc: true }, { col: 'created_at', desc: false }, { col: 'id', desc: false }] },
];
export const sortById = (id: SortId) => SORTS.find((s) => s.id === id) ?? SORTS[0];

export type Opt = 'assignee' | 'customer';
export type Shape = { assignee: boolean; customer: boolean };
export const SHAPES: { key: string; label: string; shape: Shape }[] = [
  { key: '--', label: 'no optional filter', shape: { assignee: false, customer: false } },
  { key: 'a-', label: 'assignee = me', shape: { assignee: true, customer: false } },
  { key: '-c', label: 'customer = Acme', shape: { assignee: false, customer: true } },
  { key: 'ac', label: 'assignee + customer', shape: { assignee: true, customer: true } },
];
const OPT_COL: Record<Opt, Col> = { assignee: 'assignee_id', customer: 'customer_id' };
const OPT_VAL: Record<Opt, number> = { assignee: ME, customer: ACME };
const EQ_VAL: Partial<Record<Col, number>> = { tenant_id: TENANT, status: OPEN, assignee_id: ME, customer_id: ACME };

export type Style = 'dynamic' | 'ornull';
export type PlanMode = 'custom' | 'generic';
export type Ctx = { mode: PlanMode; style: Record<Opt, Style> };

/** Clauses the planner sees for one shape. */
export function clausesFor(shape: Shape, ctx: Ctx) {
  const sargable: Col[] = ['tenant_id', 'status']; // equality clauses an index can use (and that fix the column's order)
  const opaque: Col[] = []; // OR-null clauses kept by a generic plan: evaluated after the heap fetch
  for (const o of ['assignee', 'customer'] as Opt[]) {
    const present = shape[o];
    if (ctx.style[o] === 'dynamic' || ctx.mode === 'custom') {
      if (present) sargable.push(OPT_COL[o]);
    } else if (present) {
      opaque.push(OPT_COL[o]);
    }
  }
  return { sargable, opaque };
}

/** Cells that share one statement text and therefore one generic plan. */
export function planUnitKey(shape: Shape, sort: SortId, ctx: Ctx) {
  if (ctx.mode === 'custom') return `${sort}|${shape.assignee}${shape.customer}`;
  const part = (o: Opt) => (ctx.style[o] === 'ornull' ? '*' : shape[o] ? '1' : '0');
  return `${sort}|${part('assignee')}${part('customer')}`;
}

/* ------------------------------------------------------------ paths */

export type PathSpec =
  | { kind: 'index'; ix: number; backward: boolean; presorted: number }
  | { kind: 'bitmap'; ixs: number[] }
  | { kind: 'seq' };

type Prep = { ix: IndexDef; usable: boolean; prefix: Col[]; implied: Col[] };

function prepIndex(ix: IndexDef, sargable: Col[], ctx: Ctx): Prep {
  const usable = ix.cols.length > 0 && (ix.partial !== 'live_open' || ctx.mode === 'custom');
  const implied: Col[] = ix.partial === 'live_open' ? ['status'] : [];
  const prefix: Col[] = [];
  for (const c of ix.cols) {
    if (sargable.includes(c.col) || implied.includes(c.col)) prefix.push(c.col);
    else break;
  }
  return { ix, usable, prefix, implied };
}

/** How many leading ORDER BY keys the index delivers, and in which direction. */
export function orderMatch(ix: IndexDef, sort: SortKey[], fixed: Col[]) {
  const idx = ix.cols.filter((c) => !fixed.includes(c.col));
  const want = sort.filter((k) => !fixed.includes(k.col));
  if (!idx.length || !want.length) return { n: 0, backward: false, want: want.length };
  const backward = idx[0].desc !== want[0].desc;
  let n = 0;
  while (n < want.length && n < idx.length && idx[n].col === want[n].col && (idx[n].desc !== backward) === want[n].desc) n++;
  return { n, backward, want: want.length };
}

export function candidatePaths(indexes: IndexDef[], shape: Shape, sort: SortId, ctx: Ctx): PathSpec[] {
  const { sargable } = clausesFor(shape, ctx);
  const keys = sortById(sort).keys;
  const preps = indexes.map((ix) => prepIndex(ix, sargable, ctx));
  const out: PathSpec[] = [];
  preps.forEach((p, i) => {
    if (!p.usable) return;
    const m = orderMatch(p.ix, keys, sargable);
    if (m.n > 0) out.push({ kind: 'index', ix: i, backward: m.backward, presorted: m.n === m.want ? -1 : m.n });
    if (p.prefix.length > 0) out.push({ kind: 'bitmap', ixs: [i] });
  });
  const quals = (p: Prep) => p.ix.cols.map((c) => c.col).filter((c) => sargable.includes(c));
  preps.forEach((a, i) =>
    preps.forEach((b, j) => {
      if (j <= i || !a.usable || !b.usable || !a.prefix.length || !b.prefix.length) return;
      const qa = quals(a);
      const qb = quals(b);
      if (qb.some((c) => !qa.includes(c)) && qa.some((c) => !qb.includes(c))) out.push({ kind: 'bitmap', ixs: [i, j] });
    }),
  );
  out.push({ kind: 'seq' });
  return out;
}

export type Exec = {
  buffers: number;
  indexBuffers: number;
  heapBuffers: number;
  examined: number; // index entries (or heap rows, for seq) read
  removedInIndex: number;
  removedByFilter: number;
  sorted: number; // rows fed to a Sort / Incremental Sort
  returned: number;
  seekCols: Col[];
  inIndexCols: Col[];
  filterCols: string[];
  impliedCols: Col[];
};

const STAMP = new Int32Array(HEAP_PAGES);
let stampGen = 1;

function rowPasses(t: Table, r: number, cols: Col[]) {
  for (const c of cols) if (t[c][r] !== EQ_VAL[c]) return false;
  return true;
}

function bounds(arr: Int32Array, ix: IndexDef, prefix: Col[]) {
  const t = table();
  const cols = ix.cols.slice(0, prefix.length);
  const cmp = (r: number) => {
    for (const c of cols) {
      const v = t[c.col][r];
      const want = EQ_VAL[c.col] as number;
      if (v !== want) return c.desc ? (v > want ? -1 : 1) : v < want ? -1 : 1;
    }
    return 0;
  };
  let lo = 0;
  let hi = arr.length;
  while (lo < hi) {
    const mid = (lo + hi) >> 1;
    if (cmp(arr[mid]) < 0) lo = mid + 1;
    else hi = mid;
  }
  const start = lo;
  hi = arr.length;
  while (lo < hi) {
    const mid = (lo + hi) >> 1;
    if (cmp(arr[mid]) <= 0) lo = mid + 1;
    else hi = mid;
  }
  return [start, lo] as const;
}

const presortKey = (t: Table, r: number, keys: SortKey[], n: number) => {
  let k = 0;
  for (let j = 0; j < n; j++) k = k * 1048576 + t[keys[j].col][r];
  return k;
};

export function execute(path: PathSpec, indexes: IndexDef[], shape: Shape, sort: SortId, ctx: Ctx): Exec {
  const t = table();
  const { sargable, opaque } = clausesFor(shape, ctx);
  const keys = sortById(sort).keys;
  const base = { removedInIndex: 0, removedByFilter: 0, sorted: 0, returned: 0, examined: 0 };

  if (path.kind === 'seq') {
    let match = 0;
    let removed = 0;
    for (let r = 0; r < N_ROWS; r++) {
      if (!t.deleted[r] && rowPasses(t, r, sargable) && rowPasses(t, r, opaque)) match++;
      else removed++;
    }
    return {
      ...base,
      buffers: HEAP_PAGES,
      indexBuffers: 0,
      heapBuffers: HEAP_PAGES,
      examined: N_ROWS,
      removedByFilter: removed,
      sorted: match,
      returned: Math.min(LIMIT, match),
      seekCols: [],
      inIndexCols: [],
      filterCols: [...sargable, ...opaque, 'deleted_at'],
      impliedCols: [],
    };
  }

  if (path.kind === 'index') {
    const ix = indexes[path.ix];
    const p = prepIndex(ix, sargable, ctx);
    const g = indexGeometry(ix);
    const arr = sortedEntries(ix);
    const [lo, hi] = bounds(arr, ix, p.prefix);
    const keyCols = ix.cols.map((c) => c.col);
    const inIndex = sargable.filter((c) => keyCols.includes(c) && !p.prefix.includes(c) && !p.implied.includes(c));
    const heapEq = [...sargable.filter((c) => !keyCols.includes(c) && !p.implied.includes(c)), ...opaque];
    const checkDeleted = ix.partial === 'none';
    let emitted = 0;
    let heap = 0;
    let lastBlock = -1;
    let examined = 0;
    let removedInIndex = 0;
    let removedByFilter = 0;
    let limitKey = -1;
    let lastPos = path.backward ? hi - 1 : lo;
    const step = path.backward ? -1 : 1;
    for (let pos = path.backward ? hi - 1 : lo; path.backward ? pos >= lo : pos < hi; pos += step) {
      const r = arr[pos];
      examined++;
      lastPos = pos;
      if (!rowPasses(t, r, inIndex)) {
        removedInIndex++;
        continue;
      }
      const b = heapBlock(r);
      if (b !== lastBlock) {
        heap++;
        lastBlock = b;
      }
      if ((checkDeleted && t.deleted[r]) || !rowPasses(t, r, heapEq)) {
        removedByFilter++;
        continue;
      }
      if (path.presorted < 0) {
        emitted++;
        if (emitted === LIMIT) break;
      } else {
        const k = presortKey(t, r, keys, path.presorted);
        if (emitted >= LIMIT && k !== limitKey) {
          break; // the first row of the next presorted group ends the last batch
        }
        emitted++;
        if (emitted === LIMIT) limitKey = k;
      }
    }
    const startPos = path.backward ? hi - 1 : lo;
    const leaves = examined === 0 ? 1 : Math.abs(Math.floor(lastPos / g.leafCap) - Math.floor(startPos / g.leafCap)) + 1;
    return {
      buffers: g.height + leaves + heap,
      indexBuffers: g.height + leaves,
      heapBuffers: heap,
      examined,
      removedInIndex,
      removedByFilter,
      sorted: path.presorted < 0 ? 0 : emitted,
      returned: Math.min(LIMIT, emitted),
      seekCols: p.prefix.filter((c) => !p.implied.includes(c)),
      inIndexCols: inIndex,
      filterCols: [...heapEq, ...(checkDeleted ? ['deleted_at'] : [])],
      impliedCols: p.implied,
    };
  }

  // bitmap heap scan: one index, or a BitmapAnd of two
  const parts = path.ixs.map((i) => {
    const ix = indexes[i];
    const p = prepIndex(ix, sargable, ctx);
    const g = indexGeometry(ix);
    const arr = sortedEntries(ix);
    const [lo, hi] = bounds(arr, ix, p.prefix);
    const keyCols = ix.cols.map((c) => c.col);
    const quals = sargable.filter((c) => keyCols.includes(c) && !p.implied.includes(c));
    const leaves = hi > lo ? Math.floor((hi - 1) / g.leafCap) - Math.floor(lo / g.leafCap) + 1 : 1;
    return { ix, p, arr, lo, hi, quals, pages: g.height + leaves, keyCols };
  });
  parts.sort((a, b) => a.hi - a.lo - (b.hi - b.lo));
  const [first, ...rest] = parts;
  const allQuals = Array.from(new Set(parts.flatMap((x) => x.quals)));
  const implied = Array.from(new Set(parts.flatMap((x) => x.p.implied)));
  const heapEq = [...sargable.filter((c) => !allQuals.includes(c) && !implied.includes(c)), ...opaque];
  const checkDeleted = parts.every((x) => x.ix.partial === 'none');
  stampGen++;
  let heap = 0;
  let examined = 0;
  let removedInIndex = 0;
  let removedByFilter = 0;
  let match = 0;
  for (const x of parts) examined += x.hi - x.lo;
  for (let pos = first.lo; pos < first.hi; pos++) {
    const r = first.arr[pos];
    if (!rowPasses(t, r, first.quals) || rest.some((x) => !inPartial(t, r, x.ix.partial) || !rowPasses(t, r, x.quals))) {
      removedInIndex++;
      continue;
    }
    const b = heapBlock(r);
    if (STAMP[b] !== stampGen) {
      STAMP[b] = stampGen;
      heap++;
    }
    if ((checkDeleted && t.deleted[r]) || !rowPasses(t, r, heapEq)) {
      removedByFilter++;
      continue;
    }
    match++;
  }
  const indexBuffers = parts.reduce((s, x) => s + x.pages, 0);
  return {
    buffers: indexBuffers + heap,
    indexBuffers,
    heapBuffers: heap,
    examined,
    removedInIndex,
    removedByFilter,
    sorted: match,
    returned: Math.min(LIMIT, match),
    seekCols: Array.from(new Set(parts.flatMap((x) => x.p.prefix.filter((c) => !x.p.implied.includes(c))))),
    inIndexCols: allQuals.filter((c) => !parts.some((x) => x.p.prefix.includes(c))),
    filterCols: [...heapEq, ...(checkDeleted ? ['deleted_at'] : [])],
    impliedCols: implied,
  };
}

export type Grade = 'seek' | 'filter' | 'seq';
export type Cell = { shapeKey: string; shape: Shape; sort: SortId; path: PathSpec; exec: Exec; grade: Grade; unit: string; unitSize: number };

const kindRank = (p: PathSpec) => (p.kind === 'index' ? 0 : p.kind === 'bitmap' ? 1 : 2);

export function gradeOf(path: PathSpec, exec: Exec, shape: Shape): Grade {
  if (path.kind === 'seq') return 'seq';
  if (path.kind === 'bitmap') return 'filter';
  const wanted: Col[] = ['tenant_id', 'status', ...(shape.assignee ? (['assignee_id'] as Col[]) : []), ...(shape.customer ? (['customer_id'] as Col[]) : [])];
  const ok = wanted.every((c) => exec.seekCols.includes(c) || exec.impliedCols.includes(c));
  return ok && exec.sorted <= 2 * LIMIT ? 'seek' : 'filter';
}

/** Plan and run all twenty shapes. Cells sharing a statement text under a generic plan share one path. */
export function scoreboard(indexes: IndexDef[], ctx: Ctx): Cell[] {
  const units = new Map<string, { shapeKey: string; shape: Shape; sort: SortId }[]>();
  for (const s of SORTS)
    for (const sh of SHAPES) {
      const k = planUnitKey(sh.shape, s.id, ctx);
      const list = units.get(k) ?? [];
      list.push({ shapeKey: sh.key, shape: sh.shape, sort: s.id });
      units.set(k, list);
    }
  const cells: Cell[] = [];
  for (const [unit, members] of units) {
    // Planning sees only what is sargable for the whole unit: the member with no optional filter present.
    const planShape = members[0].shape;
    const specs = candidatePaths(indexes, planShape, members[0].sort, ctx);
    let best: { spec: PathSpec; runs: Exec[]; total: number; sorted: number } | null = null;
    for (const spec of specs) {
      const runs = members.map((m) => execute(spec, indexes, m.shape, m.sort, ctx));
      const total = runs.reduce((s, r) => s + r.buffers, 0);
      const sorted = runs.reduce((s, r) => s + r.sorted, 0);
      if (!best || total < best.total || (total === best.total && (sorted < best.sorted || (sorted === best.sorted && kindRank(spec) < kindRank(best.spec))))) best = { spec, runs, total, sorted };
    }
    members.forEach((m, i) => {
      const exec = best!.runs[i];
      cells.push({ ...m, path: best!.spec, exec, grade: gradeOf(best!.spec, exec, m.shape), unit, unitSize: members.length });
    });
  }
  const order = (c: Cell) => SORTS.findIndex((s) => s.id === c.sort) * 10 + SHAPES.findIndex((s) => s.key === c.shapeKey);
  return cells.sort((a, b) => order(a) - order(b));
}

/* ------------------------------------------------------------ writes */

export const WRITE_MIX = { inserts: 200, updates: 800, status: 0.25, assignee: 0.1, priority: 0.05 };

export function writeCost(indexes: IndexDef[]) {
  const blocking = new Set<string>();
  for (const ix of indexes) {
    ix.cols.forEach((c) => blocking.add(c.col));
    if (ix.partial !== 'none') blocking.add('deleted_at');
    if (ix.partial === 'live_open') blocking.add('status');
  }
  const pHot = indexes.length === 0 ? 1 : (blocking.has('updated_at') ? 0 : 1) * (blocking.has('status') ? 1 - WRITE_MIX.status : 1) * (blocking.has('assignee_id') ? 1 - WRITE_MIX.assignee : 1) * (blocking.has('priority') ? 1 - WRITE_MIX.priority : 1);
  const nonHot = WRITE_MIX.updates * (1 - pHot);
  const perIndex = indexes.map((ix) => WRITE_MIX.inserts + nonHot * (ix.partial === 'live_open' ? 1 - WRITE_MIX.status : 1));
  const bytes = indexes.reduce((s, ix) => s + indexGeometry(ix).bytes, 0);
  return { pHot, nonHot, perIndex, total: perIndex.reduce((s, v) => s + v, 0), bytes, blocking: Array.from(blocking) };
}

/* ------------------------------------------------------------ results and cursors */

/** The true answer for one shape and sort: every matching row in ORDER BY order (independent of the plan). */
export function answer(shape: Shape, sort: SortId): number[] {
  const t = table();
  const rows: number[] = [];
  for (let r = 0; r < N_ROWS; r++) {
    if (t.tenant_id[r] !== TENANT || t.status[r] !== OPEN || t.deleted[r]) continue;
    if (shape.assignee && t.assignee_id[r] !== ME) continue;
    if (shape.customer && t.customer_id[r] !== ACME) continue;
    rows.push(r);
  }
  const keys = sortById(sort).keys;
  rows.sort((a, b) => compareTuple(tupleOf(a, keys), tupleOf(b, keys), keys));
  return rows;
}

export const tupleOf = (r: number, keys: SortKey[]) => keys.map((k) => table()[k.col][r]);
export function compareTuple(a: number[], b: number[], keys: SortKey[]) {
  for (let j = 0; j < keys.length; j++) if (a[j] !== b[j]) return keys[j].desc ? b[j] - a[j] : a[j] - b[j];
  return 0;
}

export type CursorToken = { sort: SortId; filters: string; after: number[] };

export function issueCursor(shapeKey: string, shape: Shape, sort: SortId): { page: number[]; token: CursorToken } {
  const rows = answer(shape, sort);
  const page = rows.slice(0, LIMIT);
  const keys = sortById(sort).keys;
  const last = page[page.length - 1];
  return { page, token: { sort, filters: shapeKey, after: last === undefined ? [] : tupleOf(last, keys) } };
}

export type NextResult =
  | { ok: true; page: number[]; token: CursorToken }
  | { ok: false; reason: 'sort' | 'filters'; naive: null | { page: number[]; duplicates: number; skipped: number } };

/**
 * Serve the next page for a cursor. A cursor issued for another sort or filter set is rejected; `naive` is what a
 * server that ignored the token's sort and filter tags would return, binding the tuple to the new sort's columns.
 * `seen` is the ids the client has already been shown in this session.
 */
export function nextPage(token: CursorToken, seen: number[], shapeKey: string, shape: Shape, sort: SortId): NextResult {
  const keys = sortById(sort).keys;
  const rows = answer(shape, sort);
  const after = (vals: number[]) => rows.filter((r) => compareTuple(tupleOf(r, keys), vals, keys) > 0);
  if (token.sort === sort && token.filters === shapeKey) {
    const page = after(token.after).slice(0, LIMIT);
    const last = page[page.length - 1];
    return { ok: true, page, token: { sort, filters: shapeKey, after: last === undefined ? token.after : tupleOf(last, keys) } };
  }
  const reason = token.sort !== sort ? 'sort' : 'filters';
  if (token.after.length !== keys.length) return { ok: false, reason, naive: null };
  const seenSet = new Set(seen);
  const t = table();
  const page = after(token.after).slice(0, LIMIT);
  const firstPos = page.length ? rows.indexOf(page[0]) : rows.length;
  const duplicates = page.filter((r) => seenSet.has(t.id[r])).length;
  const skipped = rows.slice(0, firstPos).filter((r) => !seenSet.has(t.id[r])).length;
  return { ok: false, reason, naive: { page, duplicates, skipped } };
}

export const fmtTs = (v: number) => {
  const d = new Date(EPOCH_MS + v * TICK_MS);
  return d.toISOString().slice(0, 16).replace('T', ' ');
};

/* ------------------------------------------------------------ SQL and plan text */

const LITERAL: Partial<Record<Col, string>> = { tenant_id: String(TENANT), status: "'open'", assignee_id: String(ME), customer_id: String(ACME) };
const fmtKeyVal = (col: Col, v: number) => (col === 'created_at' || col === 'updated_at' ? `'${fmtTs(v)}'` : String(v));
const keysSql = (keys: SortKey[]) => keys.map((k) => `${k.col}${k.desc ? ' DESC' : ''}`).join(', ');

/** The statement the endpoint's query builder emits for one shape (optionally with a keyset predicate). */
export function sqlFor(shape: Shape, sort: SortId, ctx: Ctx, keyset: number[] | null) {
  const lines = ['SELECT id, subject, priority, created_at, updated_at', 'FROM tickets', 'WHERE tenant_id = $1', '  AND status = $2', '  AND deleted_at IS NULL'];
  const params = [`$1 = ${TENANT}`, "$2 = 'open'"];
  const paramOf: Partial<Record<Col, string>> = { tenant_id: '$1', status: '$2' };
  let n = 3;
  for (const o of ['assignee', 'customer'] as Opt[]) {
    const col = OPT_COL[o];
    if (ctx.style[o] === 'ornull') {
      lines.push(`  AND (${col} = $${n} OR $${n} IS NULL)`);
      params.push(`$${n} = ${shape[o] ? OPT_VAL[o] : 'NULL'}`);
    } else if (shape[o]) {
      lines.push(`  AND ${col} = $${n}`);
      params.push(`$${n} = ${OPT_VAL[o]}`);
    } else continue;
    paramOf[col] = `$${n}`;
    n++;
  }
  const keys = sortById(sort).keys;
  if (keyset && keyset.length === keys.length) {
    const ps = keys.map((k, j) => {
      params.push(`$${n + j} = ${fmtKeyVal(k.col, keyset[j])}`);
      return `$${n + j}`;
    });
    const uniform = keys.every((k) => k.desc === keys[0].desc);
    if (uniform) {
      lines.push(`  AND (${keys.map((k) => k.col).join(', ')}) ${keys[0].desc ? '<' : '>'} (${ps.join(', ')})`);
    } else {
      const [k0, ...rest] = keys;
      const [p0, ...pr] = ps;
      lines.push(`  AND ${k0.col} ${k0.desc ? '<=' : '>='} ${p0}`);
      lines.push(`  AND (${k0.col} ${k0.desc ? '<' : '>'} ${p0}`);
      lines.push(`       OR (${rest.map((k) => k.col).join(', ')}) ${rest[0].desc ? '<' : '>'} (${pr.join(', ')}))`);
    }
  }
  lines.push(`ORDER BY ${keysSql(keys)}`);
  lines.push(`LIMIT ${LIMIT};`);
  return { text: lines.join('\n'), params, paramOf };
}

/** EXPLAIN-shaped description of the path the model chose for one cell. */
export function planText(cell: Cell, indexes: IndexDef[], ctx: Ctx) {
  const { exec, path, shape, sort } = cell;
  const keys = sortById(sort).keys;
  const { paramOf } = sqlFor(shape, sort, ctx, null);
  const isOrNull = (c: string) => (c === 'assignee_id' && ctx.style.assignee === 'ornull') || (c === 'customer_id' && ctx.style.customer === 'ornull');
  const clause = (c: string) => {
    if (c === 'deleted_at') return 'deleted_at IS NULL';
    const col = c as Col;
    if (ctx.mode === 'custom') return `${col} = ${LITERAL[col]}`;
    const p = paramOf[col] ?? '$?';
    return isOrNull(col) ? `(${col} = ${p} OR ${p} IS NULL)` : `${col} = ${p}`;
  };
  // A generic plan carries every OR-null clause as a Filter, whether or not this execution's parameter is NULL.
  const filters = [...exec.filterCols];
  if (ctx.mode === 'generic' && path.kind !== 'seq') {
    (['assignee_id', 'customer_id'] as Col[]).forEach((c) => {
      if (isOrNull(c) && !filters.includes(c)) filters.splice(filters.length - (filters.includes('deleted_at') ? 1 : 0), 0, c);
    });
  }
  const out: string[] = [`Limit  (rows returned: ${exec.returned})`];
  const filterLines = (pad: string) => {
    if (!filters.length) return;
    out.push(`${pad}Filter: ${filters.map(clause).join(' AND ')}`);
    out.push(`${pad}Rows Removed by Filter: ${fmtNum(exec.removedByFilter)}`);
  };
  if (path.kind === 'seq') {
    out.push(`  ->  Sort  (Sort Key: ${keysSql(keys)}; ${fmtNum(exec.sorted)} rows in)`);
    out.push('        ->  Seq Scan on tickets');
    filterLines('              ');
  } else if (path.kind === 'index') {
    const ix = indexes[path.ix];
    let pad = '  ';
    if (path.presorted >= 0) {
      out.push(`  ->  Incremental Sort  (Sort Key: ${keysSql(keys)}; Presorted Key: ${keysSql(keys.slice(0, path.presorted))}; ${fmtNum(exec.sorted)} rows in)`);
      pad = '        ';
    }
    out.push(`${pad}->  Index Scan${path.backward ? ' Backward' : ''} using ${ix.name} on tickets`);
    const inner = `${pad}      `;
    if (exec.seekCols.length) out.push(`${inner}Index Cond, bounds the range: ${exec.seekCols.map(clause).join(' AND ')}`);
    if (exec.inIndexCols.length) {
      out.push(`${inner}Index Cond, checked inside the range: ${exec.inIndexCols.map(clause).join(' AND ')}`);
      out.push(`${inner}(entries rejected in the index: ${fmtNum(exec.removedInIndex)})`);
    }
    if (!exec.seekCols.length) out.push(`${inner}(no usable Index Cond: a full walk of the index, for its order)`);
    if (exec.impliedCols.length) out.push(`${inner}(status = 'open' is implied by the index predicate)`);
    filterLines(inner);
  } else {
    out.push(`  ->  Sort  (Sort Key: ${keysSql(keys)}; ${fmtNum(exec.sorted)} rows in)`);
    out.push('        ->  Bitmap Heap Scan on tickets');
    filterLines('              ');
    out.push(`              Heap Blocks: ${fmtNum(exec.heapBuffers)}`);
    const { sargable } = clausesFor(shape, ctx);
    const scanLine = (i: number, pad: string) => {
      const ix = indexes[i];
      const quals = ix.cols.map((c) => c.col).filter((c) => sargable.includes(c) && !(ix.partial === 'live_open' && c === 'status'));
      out.push(`${pad}->  Bitmap Index Scan on ${ix.name}  (Index Cond: ${quals.map(clause).join(' AND ')})`);
    };
    if (path.ixs.length > 1) {
      out.push('              ->  BitmapAnd');
      path.ixs.forEach((i) => scanLine(i, '                    '));
    } else scanLine(path.ixs[0], '              ');
  }
  out.push(`Buffers (model): ${fmtNum(exec.buffers)} = ${fmtNum(exec.indexBuffers)} index + ${fmtNum(exec.heapBuffers)} heap`);
  return out.join('\n');
}

/* ------------------------------------------------------------ presets */

const ic = (col: Col, desc = false): IndexCol => ({ col, desc });
const IX = {
  open: { name: 'ix_open', cols: [ic('tenant_id'), ic('status')], partial: 'live' as PartialPred },
  created: { name: 'ix_created', cols: [ic('tenant_id'), ic('status'), ic('created_at'), ic('id')], partial: 'live' as PartialPred },
  assignee: { name: 'ix_assignee', cols: [ic('tenant_id'), ic('status'), ic('assignee_id'), ic('created_at'), ic('id')], partial: 'live' as PartialPred },
  customer: { name: 'ix_customer', cols: [ic('customer_id')], partial: 'none' as PartialPred },
  priority: { name: 'ix_priority', cols: [ic('tenant_id'), ic('status'), ic('priority', true), ic('created_at'), ic('id')], partial: 'live' as PartialPred },
  updated: { name: 'ix_updated', cols: [ic('tenant_id'), ic('status'), ic('updated_at'), ic('id')], partial: 'live' as PartialPred },
  myOpen: { name: 'ix_my_open', cols: [ic('tenant_id'), ic('assignee_id'), ic('created_at'), ic('id')], partial: 'live_open' as PartialPred },
};
export const PRESETS: { id: string; label: string; indexes: IndexDef[] }[] = [
  { id: 'none', label: 'No secondary index', indexes: [] },
  { id: 'prefix', label: '(tenant_id, status)', indexes: [IX.open] },
  { id: 'newest', label: '(tenant_id, status, created_at, id)', indexes: [IX.created] },
  { id: 'optfirst', label: 'Optional column before the sort', indexes: [IX.assignee] },
  { id: 'three', label: 'Three indexes', indexes: [IX.created, IX.assignee, IX.customer] },
  { id: 'four', label: 'Four: + priority queue', indexes: [IX.created, IX.assignee, IX.customer, IX.priority] },
  { id: 'five', label: 'Five: + recently updated', indexes: [IX.created, IX.assignee, IX.customer, IX.priority, IX.updated] },
  { id: 'partial_open', label: "Partial index WHERE status = 'open'", indexes: [IX.created, IX.myOpen] },
];
const clonePreset = (id: string) => (PRESETS.find((p) => p.id === id) ?? PRESETS[0]).indexes.map((ix) => ({ ...ix, cols: ix.cols.map((c) => ({ ...c })) }));

export const indexSql = (ix: IndexDef) =>
  `CREATE INDEX ${ix.name} ON tickets (${ix.cols.map((c) => `${c.col}${c.desc ? ' DESC' : ''}`).join(', ')})${ix.partial === 'none' ? '' : ` ${PARTIAL_SQL[ix.partial]}`}`;

/* ------------------------------------------------------------ component */

const GRADE_COLOR: Record<Grade, string> = { seek: 'var(--viz-good)', filter: 'var(--viz-warning)', seq: 'var(--viz-critical)' };
const GRADE_LABEL: Record<Grade, string> = { seek: 'index-served', filter: 'filtered after seek', seq: 'sequential scan' };
const SORT_HEAD: Record<SortId, [string, string]> = {
  newest: ['Newest', 'first'],
  oldest: ['Oldest', 'first'],
  updated: ['Recently', 'updated'],
  prio_newest: ['Priority,', 'newest first'],
  prio_queue: ['Priority,', 'oldest first'],
};

/** At most ~18 characters, for the scoreboard cell. */
function cellPath(cell: Cell, indexes: IndexDef[]) {
  const p = cell.path;
  if (p.kind === 'seq') return 'Seq Scan';
  if (p.kind === 'bitmap') return p.ixs.length > 1 ? 'BitmapAnd' : `bitmap ${indexes[p.ixs[0]].name}`.slice(0, 18);
  const name = indexes[p.ix].name.slice(0, 12);
  return p.backward && p.presorted < 0 ? `${name} bwd` : name;
}

function shortPath(cell: Cell, indexes: IndexDef[]) {
  const p = cell.path;
  if (p.kind === 'seq') return 'Seq Scan + Sort';
  if (p.kind === 'bitmap') return p.ixs.length > 1 ? 'BitmapAnd + Sort' : `bitmap ${indexes[p.ixs[0]].name} + Sort`;
  return `${indexes[p.ix].name}${p.presorted >= 0 ? ' + incr. sort' : p.backward ? ', backward' : ''}`;
}

type Session = {
  token: CursorToken;
  seen: number[];
  pages: number;
  last: { kind: 'first' | 'ok'; count: number } | { kind: 'rejected'; result: Extract<NextResult, { ok: false }> };
};

export default function ListEndpointIndexSetLab() {
  const [presetId, setPresetId] = useState('four');
  const [indexes, setIndexes] = useState<IndexDef[]>(() => clonePreset('four'));
  const [mode, setMode] = useState<PlanMode>('generic');
  const [styleA, setStyleA] = useState<Style>('dynamic');
  const [styleC, setStyleC] = useState<Style>('dynamic');
  const [sel, setSel] = useState<{ shapeKey: string; sort: SortId }>({ shapeKey: 'a-', sort: 'newest' });
  const [session, setSession] = useState<Session | null>(null);

  const ctx = useMemo<Ctx>(() => ({ mode, style: { assignee: styleA, customer: styleC } }), [mode, styleA, styleC]);
  const input = useMemo(() => ({ indexes, ctx }), [indexes, ctx]);
  const deferred = useDeferredValue(input);
  const cells = useMemo(() => scoreboard(deferred.indexes, deferred.ctx), [deferred]);
  const writes = useMemo(() => writeCost(deferred.indexes), [deferred]);
  const stale = deferred !== input;
  const dIx = deferred.indexes;
  const dCtx = deferred.ctx;

  const cell = cells.find((c) => c.shapeKey === sel.shapeKey && c.sort === sel.sort) ?? cells[0];
  const shapeOf = (k: string) => (SHAPES.find((s) => s.key === k) ?? SHAPES[0]).shape;
  const shapeLabel = (k: string) => (SHAPES.find((s) => s.key === k) ?? SHAPES[0]).label;
  const grades = { seek: 0, filter: 0, seq: 0 };
  cells.forEach((c) => grades[c.grade]++);
  const totalBuffers = cells.reduce((s, c) => s + c.exec.buffers, 0);

  // Generic plans shared by several cells get a common tag.
  const unitTags = new Map<string, number>();
  if (dCtx.mode === 'generic') cells.forEach((c) => c.unitSize > 1 && !unitTags.has(c.unit) && unitTags.set(c.unit, unitTags.size + 1));

  const edit = (next: IndexDef[]) => {
    setIndexes(next);
    setPresetId('edited');
  };
  const updIx = (i: number, f: (ix: IndexDef) => IndexDef) => edit(indexes.map((ix, j) => (j === i ? f({ ...ix, cols: ix.cols.map((c) => ({ ...c })) }) : ix)));

  const cursorMatches = session && session.token.sort === sel.sort && session.token.filters === sel.shapeKey && session.last.kind !== 'rejected';
  const sql = sqlFor(shapeOf(sel.shapeKey), sel.sort, ctx, cursorMatches ? session!.token.after : null);

  const fetchFirst = () => {
    const { page, token } = issueCursor(sel.shapeKey, shapeOf(sel.shapeKey), sel.sort);
    setSession({ token, seen: page.map((r) => table().id[r]), pages: 1, last: { kind: 'first', count: page.length } });
  };
  const fetchNext = () => {
    if (!session) return;
    const r = nextPage(session.token, session.seen, sel.shapeKey, shapeOf(sel.shapeKey), sel.sort);
    if (r.ok) setSession({ token: r.token, seen: [...session.seen, ...r.page.map((x) => table().id[x])], pages: session.pages + 1, last: { kind: 'ok', count: r.page.length } });
    else setSession({ ...session, last: { kind: 'rejected', result: r } });
  };

  const W = 720;
  const LEFTW = 128;
  const TOP = 40;
  const CW = (W - LEFTW) / SORTS.length;
  const CH = 62;
  const H = TOP + SHAPES.length * CH + 4;

  const MW = 720;
  const barX = 150;
  const barMax = 5000;
  const bw = (v: number) => (v / barMax) * (MW - barX - 90);

  const tokenView = session
    ? JSON.stringify({
        sort: session.token.sort,
        filters: shapeLabel(session.token.filters),
        after: session.token.after.map((v, j) => {
          const col = sortById(session.token.sort).keys[j].col;
          return col === 'created_at' || col === 'updated_at' ? fmtTs(v) : v;
        }),
      })
    : '';

  const noteGrade = (() => {
    const e = cell.exec;
    const where = `${shapeLabel(cell.shapeKey)}, ${sortById(cell.sort).label.toLowerCase()}`;
    const shared = dCtx.mode === 'generic' && cell.unitSize > 1;
    const lead = (
      <strong>
        {where}: {GRADE_LABEL[cell.grade]}, {fmtNum(e.buffers)} buffers.
      </strong>
    );
    let body = '';
    if (cell.path.kind === 'seq') body = `No index has a usable leading column, so every one of the table's ${fmtNum(HEAP_PAGES)} heap pages is read and ${fmtNum(e.sorted)} matching rows are sorted to return ${e.returned}.`;
    else if (cell.path.kind === 'bitmap')
      body = `The ${cell.path.ixs.length > 1 ? 'BitmapAnd of two indexes' : 'bitmap scan'} finds all ${fmtNum(e.sorted)} matching rows, but a bitmap visits the heap in physical order, so every one of them is fetched (${fmtNum(e.heapBuffers)} heap pages) and sorted before the first ${LIMIT} can be returned.`;
    else if (cell.grade === 'seek')
      body = `The scan seeks ${e.seekCols.join(', ')}${e.impliedCols.length ? ' (status is implied by the partial predicate)' : ''}, reads rows already in ORDER BY order and stops after ${e.returned}: ${fmtNum(e.indexBuffers)} index pages and ${fmtNum(e.heapBuffers)} heap pages.`;
    else
      body = `The index ${e.seekCols.length ? `seeks only ${e.seekCols.join(', ')}` : 'has no usable leading column and is walked for its order'}; ${fmtNum(e.examined)} entries are read${e.removedInIndex ? `, ${fmtNum(e.removedInIndex)} rejected inside the index` : ''}${e.removedByFilter ? `, ${fmtNum(e.removedByFilter)} heap rows discarded by Filter` : ''}${e.sorted ? `, ${fmtNum(e.sorted)} rows sorted` : ''} to return ${e.returned}.`;
    const sharedText = shared ? ` This statement text also serves ${cell.unitSize - 1} other shape${cell.unitSize > 2 ? 's' : ''} (tag P${unitTags.get(cell.unit)}), so one generic plan is used for all of them and the OR-null clause is only a Filter.` : '';
    return (
      <>
        {lead} {body}
        {sharedText}
      </>
    );
  })();

  const cursorMessage = (() => {
    if (!session) return `Press "Fetch page 1" to issue a cursor for the selected cell, then change the sort or filters and ask for the next page.`;
    const l = session.last;
    if (l.kind !== 'rejected')
      return l.kind === 'first' ? `Page 1: ${l.count} rows. The cursor records the sort, the filters and the last row's full sort tuple.` : `Page ${session.pages}: ${l.count} rows, read with the keyset predicate shown in the SQL.`;
    const r = l.result;
    const head = `Rejected: this cursor was issued for ${r.reason === 'sort' ? `sort "${sortById(session.token.sort).label}"` : `filters "${shapeLabel(session.token.filters)}"`}. Restart at page 1.`;
    if (!r.naive) return `${head} A server that ignored the tag could not even bind it: the cursor holds ${session.token.after.length} values and this sort needs ${sortById(sel.sort).keys.length}.`;
    return `${head} A server that trusted the tuple would bind it to the new sort's columns and return ${r.naive.page.length} rows, ${r.naive.duplicates} of them already shown, and silently skip ${fmtNum(r.naive.skipped)} rows the client has never seen.`;
  })();

  return (
    <VizPanel
      title="One list endpoint, twenty query shapes"
      subtitle="tenant_id and status are always filtered, assignee and customer are optional, and five sort orders are offered. Every shape is planned and run, LIMIT 25, against a 100,000-row tickets table. Edit the index set and click any cell for its SQL and plan."
      controls={
        <>
          <Choice
            label="Index set"
            value={presetId}
            onChange={(v) => {
              if (v === 'edited') return;
              setPresetId(v);
              setIndexes(clonePreset(v));
            }}
            options={[...PRESETS.map((p) => ({ value: p.id, label: p.label })), ...(presetId === 'edited' ? [{ value: 'edited', label: 'Edited set' }] : [])]}
          />
          <Segmented
            label="Plan built"
            value={mode}
            onChange={setMode}
            options={[
              { value: 'generic', label: 'once, for any values (generic)' },
              { value: 'custom', label: 'per execution, with values (custom)' },
            ]}
          />
          <Segmented
            label="assignee filter"
            value={styleA}
            onChange={setStyleA}
            options={[
              { value: 'dynamic', label: 'added when present' },
              { value: 'ornull', label: 'OR $n IS NULL' },
            ]}
          />
          <Segmented
            label="customer filter"
            value={styleC}
            onChange={setStyleC}
            options={[
              { value: 'dynamic', label: 'added when present' },
              { value: 'ornull', label: 'OR $n IS NULL' },
            ]}
          />
        </>
      }
      legend={
        <Legend
          items={[
            { label: 'Index-served: work grows with LIMIT', color: GRADE_COLOR.seek },
            { label: 'Filtered after seek: work grows with rows matched', color: GRADE_COLOR.filter },
            { label: 'Sequential scan: work grows with the table', color: GRADE_COLOR.seq },
            { label: 'Heap row versions written', color: 'var(--viz-1)' },
            { label: 'Index entries written', color: 'var(--viz-7)' },
          ]}
        />
      }
      stats={
        <Stats
          items={[
            { label: 'Index-served', value: `${grades.seek} / 20` },
            { label: 'Filtered after seek', value: `${grades.filter} / 20` },
            { label: 'Sequential scans', value: `${grades.seq} / 20` },
            { label: 'Buffers, all 20 shapes', value: fmtNum(totalBuffers), hint: 'Model buffers: descent pages, each leaf walked, and each heap page visit (a new block for an index scan; each distinct block for a bitmap or seq scan).' },
            { label: 'Index entries per 1,000 writes', value: fmtNum(writes.total), hint: 'Assumed mix: 200 inserts and 800 updates. Every update sets updated_at; 25% change status, 10% reassign, 5% change priority. A non-HOT update adds an entry to every index.' },
            { label: 'Index size', value: fmtBytes(writes.bytes), hint: 'Freshly built: leaf pages 90% full, internal pages 70%, no deduplication.' },
          ]}
        />
      }
      note={<Note>{noteGrade}</Note>}
      table={
        <table className="viz-table">
          <thead>
            <tr>
              <th>Filters</th>
              <th>Sort</th>
              <th>Grade</th>
              <th>Path</th>
              <th>Buffers</th>
              <th>Entries or rows read</th>
              <th>Rejected in index</th>
              <th>Removed by Filter</th>
              <th>Rows sorted</th>
            </tr>
          </thead>
          <tbody>
            {cells.map((c) => (
              <tr key={`${c.shapeKey}${c.sort}`}>
                <td>{shapeLabel(c.shapeKey)}</td>
                <td>{sortById(c.sort).label}</td>
                <td>{GRADE_LABEL[c.grade]}</td>
                <td>{shortPath(c, dIx)}</td>
                <td>{fmtNum(c.exec.buffers)}</td>
                <td>{fmtNum(c.exec.examined)}</td>
                <td>{fmtNum(c.exec.removedInIndex)}</td>
                <td>{fmtNum(c.exec.removedByFilter)}</td>
                <td>{fmtNum(c.exec.sorted)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      }
    >
      {/* ---- index editor ---- */}
      <div style={{ display: 'grid', gap: 4, marginBottom: 8, minWidth: 0 }} role="group" aria-label="Candidate index set">
        {indexes.length === 0 ? <div style={{ color: 'var(--viz-ink-2)', fontSize: '0.8rem' }}>No secondary indexes: only the primary key on id, which no shape can use.</div> : null}
        {indexes.map((ix, i) => (
          <div key={i} data-ix={ix.name} style={{ display: 'flex', flexWrap: 'wrap', alignItems: 'center', gap: 3, padding: '3px 5px', border: '1px solid var(--viz-border)', borderRadius: 6, background: 'var(--viz-plane)', fontSize: '0.74rem', lineHeight: 1.3 }}>
            <code style={{ color: 'var(--viz-ink)', minWidth: '5.6rem', fontSize: '0.74rem' }}>{ix.name}</code>
            {ix.cols.map((c, j) => (
              <span key={j} style={{ display: 'inline-flex', alignItems: 'center', gap: 2, border: '1px solid var(--viz-border)', borderRadius: 5, padding: '0 2px 0 5px', background: 'var(--viz-surface)', color: 'var(--viz-ink)' }}>
                {c.col}
                <button type="button" title="Toggle ASC / DESC" aria-label={`${ix.name} ${c.col} direction`} style={{ padding: '0 4px', fontSize: '0.66rem', lineHeight: 1.4 }} onClick={() => updIx(i, (x) => ({ ...x, cols: x.cols.map((y, k) => (k === j ? { ...y, desc: !y.desc } : y)) }))}>
                  {c.desc ? 'DESC' : 'ASC'}
                </button>
                <button type="button" aria-label={`Remove ${c.col} from ${ix.name}`} style={{ display: 'inline-flex', alignItems: 'center', height: 16, padding: '0 3px 2px', fontSize: '0.8rem', lineHeight: 1, border: 0, background: 'transparent', color: 'var(--viz-ink-2)' }} onClick={() => updIx(i, (x) => ({ ...x, cols: x.cols.filter((_, k) => k !== j) }))}>
                  ×
                </button>
              </span>
            ))}
            {ix.cols.length < 5 ? (
              <select
                aria-label={`Add a column to ${ix.name}`}
                value=""
                style={{ padding: '0 2px', fontSize: '0.7rem' }}
                onChange={(e) => {
                  const col = e.currentTarget.value as Col;
                  if (col) updIx(i, (x) => ({ ...x, cols: [...x.cols, { col, desc: false }] }));
                }}
              >
                <option value="">+ column</option>
                {KEY_COLS.filter((k) => !ix.cols.some((y) => y.col === k)).map((k) => (
                  <option key={k} value={k}>
                    {k}
                  </option>
                ))}
              </select>
            ) : null}
            <select aria-label={`${ix.name} partial predicate`} value={ix.partial} style={{ padding: '0 2px', fontSize: '0.7rem' }} onChange={(e) => updIx(i, (x) => ({ ...x, partial: e.currentTarget.value as PartialPred }))}>
              <option value="none">all rows</option>
              <option value="live">WHERE deleted_at IS NULL</option>
              <option value="live_open">WHERE deleted_at IS NULL AND status = &apos;open&apos;</option>
            </select>
            {ix.cols.length === 0 ? <span style={{ color: 'var(--viz-ink-2)' }}>add a column</span> : null}
            <button type="button" aria-label={`Drop ${ix.name}`} style={{ padding: '0 6px', fontSize: '0.7rem', marginLeft: 'auto' }} onClick={() => edit(indexes.filter((_, k) => k !== i))}>
              drop
            </button>
          </div>
        ))}
        <div>
          <Button
            disabled={indexes.length >= 5}
            onClick={() => {
              let n = indexes.length + 1;
              while (indexes.some((x) => x.name === `ix_new${n}`)) n++;
              edit([...indexes, { name: `ix_new${n}`, cols: [ic('tenant_id'), ic('status')], partial: 'live' }]);
            }}
          >
            + Add index
          </Button>
        </div>
      </div>

      {/* ---- scoreboard ---- */}
      <div style={{ overflowX: 'auto', overscrollBehaviorX: 'contain' }}>
      <svg viewBox={`0 0 ${W} ${H}`} width={W} height={H} role="group" aria-label={`Scoreboard: ${grades.seek} shapes index-served, ${grades.filter} filtered after seek, ${grades.seq} sequential scans`} style={{ minWidth: 600, opacity: stale ? 0.6 : 1 }}>
        {SORTS.map((s, i) => (
          <g key={s.id}>
            <text x={LEFTW + i * CW + CW / 2} y={15} textAnchor="middle" style={{ fontSize: 12, fill: 'var(--viz-ink)', fontWeight: 600 }}>
              {SORT_HEAD[s.id][0]}
            </text>
            <text x={LEFTW + i * CW + CW / 2} y={30} textAnchor="middle" style={{ fontSize: 11, fill: 'var(--viz-ink-2)' }}>
              {SORT_HEAD[s.id][1]}
            </text>
          </g>
        ))}
        {SHAPES.map((sh, r) => (
          <text key={sh.key} x={4} y={TOP + r * CH + CH / 2 + 4} style={{ fontSize: 12, fill: 'var(--viz-ink)' }}>
            {sh.label}
          </text>
        ))}
        {cells.map((c) => {
          const col = SORTS.findIndex((s) => s.id === c.sort);
          const row = SHAPES.findIndex((s) => s.key === c.shapeKey);
          const x = LEFTW + col * CW + 3;
          const y = TOP + row * CH + 3;
          const w = CW - 6;
          const h = CH - 6;
          const selected = c.shapeKey === sel.shapeKey && c.sort === sel.sort;
          const tag = unitTags.get(c.unit);
          return (
            <g
              key={`${c.shapeKey}${c.sort}`}
              data-cell={`${c.shapeKey}|${c.sort}`}
              data-grade={c.grade}
              data-buffers={c.exec.buffers}
              role="button"
              tabIndex={0}
              aria-label={`${shapeLabel(c.shapeKey)}, ${sortById(c.sort).label}: ${GRADE_LABEL[c.grade]}, ${c.exec.buffers} buffers`}
              style={{ cursor: 'pointer' }}
              onClick={() => setSel({ shapeKey: c.shapeKey, sort: c.sort })}
              onKeyDown={(e) => {
                if (e.key === 'Enter' || e.key === ' ') {
                  e.preventDefault();
                  setSel({ shapeKey: c.shapeKey, sort: c.sort });
                }
              }}
            >
              <title>{`${shapeLabel(c.shapeKey)}, ${sortById(c.sort).label}: ${shortPath(c, dIx)}, ${fmtNum(c.exec.buffers)} buffers`}</title>
              <rect x={x} y={y} width={w} height={h} rx={6} fill="var(--viz-surface)" stroke={selected ? 'var(--viz-ink)' : GRADE_COLOR[c.grade]} strokeWidth={selected ? 2.5 : 1.5} />
              <rect x={x + 1} y={y + 1} width={6} height={h - 2} rx={3} fill={GRADE_COLOR[c.grade]} />
              <text x={x + 14} y={y + 19} style={{ fontSize: 15, fontWeight: 600, fill: 'var(--viz-ink)' }}>
                {fmtNum(c.exec.buffers)}
                <tspan style={{ fontSize: 10, fontWeight: 400, fill: 'var(--viz-ink-2)' }}> buf</tspan>
              </text>
              <text x={x + 14} y={y + 34} style={{ fontSize: 10.5, fill: 'var(--viz-ink-2)' }}>
                {c.grade === 'seek' ? 'index-served' : c.grade === 'filter' ? (c.exec.sorted > 0 ? 'filtered + sort' : 'filtered') : 'seq scan + sort'}
              </text>
              <text x={x + 14} y={y + 48} style={{ fontSize: 9.5, fill: 'var(--viz-ink-muted)' }}>
                {cellPath(c, dIx)}
              </text>
              {tag ? (
                <text x={x + w - 5} y={y + 14} textAnchor="end" style={{ fontSize: 9.5, fill: 'var(--viz-ink-2)' }}>
                  P{tag}
                </text>
              ) : null}
            </g>
          );
        })}
      </svg>
      </div>
      {unitTags.size ? <div style={{ fontSize: '0.75rem', color: 'var(--viz-ink-2)', margin: '2px 0 0' }}>Cells tagged with the same P-number run one shared generic plan: an OR-null filter makes them the same statement text.</div> : null}

      {/* ---- write cost ---- */}
      <div style={{ overflowX: 'auto', overscrollBehaviorX: 'contain', marginTop: 8 }}>
      <svg viewBox={`0 0 ${MW} 74`} width={MW} height={74} role="img" aria-label={`Write cost: ${Math.round(writes.total)} index entries per 1,000 row writes`} style={{ minWidth: 600 }}>
        <text x={4} y={20} style={{ fontSize: 12, fill: 'var(--viz-ink)' }}>
          heap row versions
        </text>
        <rect x={barX} y={8} width={bw(1000)} height={16} rx={3} fill="var(--viz-1)" />
        <text x={barX + bw(1000) + 6} y={20} style={{ fontSize: 11, fill: 'var(--viz-ink-2)' }}>
          1,000 per 1,000 writes
        </text>
        <text x={4} y={50} style={{ fontSize: 12, fill: 'var(--viz-ink)' }}>
          index entries
        </text>
        {(() => {
          let x = barX;
          return writes.perIndex.map((v, i) => {
            const w = Math.max(2, bw(v));
            const el = (
              <g key={i}>
                <rect x={x} y={38} width={Math.max(1, w - 2)} height={16} rx={3} fill="var(--viz-7)" />
                {w > 56 ? (
                  <text x={x + 2} y={67} style={{ fontSize: 9.5, fill: 'var(--viz-ink-muted)' }}>
                    {dIx[i]?.name}
                  </text>
                ) : null}
              </g>
            );
            x += w;
            return el;
          });
        })()}
        <text x={barX + bw(writes.total) + 6} y={50} style={{ fontSize: 11, fill: 'var(--viz-ink-2)' }}>
          {fmtNum(writes.total)}
        </text>

      </svg>
      </div>
      <div data-role="hot" style={{ fontSize: '0.75rem', color: 'var(--viz-ink-2)' }}>
        {dIx.length === 0
          ? 'No secondary index: every update can be HOT.'
          : `${fmtNum(writes.pHot * 100)}% of updates stay HOT${writes.blocking.includes('updated_at') ? ' (none can: every update changes updated_at, which is indexed)' : ''}; each non-HOT update adds an entry to all ${dIx.length} index${dIx.length === 1 ? '' : 'es'}.`}
      </div>
      {/* ---- SQL + plan for the selected cell ---- */}
      <div style={{ display: 'flex', flexWrap: 'wrap', gap: 8, marginTop: 10 }}>
        <div style={{ flex: '1 1 300px', minWidth: 0 }}>
          <div style={{ fontSize: '0.75rem', color: 'var(--viz-ink-2)', marginBottom: 2 }}>SQL the builder emits</div>
          <pre data-role="sql" style={{ margin: 0, padding: '6px 8px', fontSize: '0.72rem', lineHeight: 1.45, background: 'var(--viz-plane)', color: 'var(--viz-ink)', border: '1px solid var(--viz-border)', borderRadius: 6, overflowX: 'auto' }}>
            {sql.text}
            {'\n-- '}
            {sql.params.join(', ')}
          </pre>
        </div>
        <div style={{ flex: '1 1 300px', minWidth: 0 }}>
          <div style={{ fontSize: '0.75rem', color: 'var(--viz-ink-2)', marginBottom: 2 }}>Plan the model chose (fewest buffers)</div>
          <pre data-role="plan" style={{ margin: 0, padding: '6px 8px', fontSize: '0.72rem', lineHeight: 1.45, background: 'var(--viz-plane)', color: 'var(--viz-ink)', border: '1px solid var(--viz-border)', borderRadius: 6, overflowX: 'auto' }}>
            {planText(cell, dIx, dCtx)}
          </pre>
        </div>
      </div>

      {/* ---- cursor ---- */}
      <div style={{ marginTop: 10, padding: '6px 8px', border: '1px solid var(--viz-border)', borderRadius: 6 }}>
        <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6, alignItems: 'center' }}>
          <span style={{ fontSize: '0.78rem', color: 'var(--viz-ink)', fontWeight: 600 }}>Keyset cursor</span>
          <Button onClick={fetchFirst}>Fetch page 1</Button>
          <Button primary onClick={fetchNext} disabled={!session}>
            Next page
          </Button>
        </div>
        {session ? (
          <code data-role="token" style={{ display: 'block', marginTop: 4, fontSize: '0.72rem', color: 'var(--viz-ink-2)', overflowWrap: 'anywhere' }}>
            cursor (decoded; sign it in production): {tokenView}
          </code>
        ) : null}
        <div data-role="cursor-msg" style={{ marginTop: 4, fontSize: '0.8rem', color: 'var(--viz-ink)', borderLeft: session?.last.kind === 'rejected' ? '3px solid var(--viz-critical)' : '3px solid transparent', paddingLeft: 6 }}>
          {cursorMessage}
        </div>
      </div>

    </VizPanel>
  );

}
