import { useMemo, useState, type ReactNode } from 'react';
import { VizPanel, Segmented, Slider, Choice, Button, Legend, Stats, Note, fmtBytes, fmtNum } from './Viz';

/**
 * Designing an index set for a whole workload.
 *
 * Model (all numbers are estimates in "pages touched per second"):
 * - Four tables with fixed row counts and heap pages of 8 KB. Primary keys and UNIQUE constraints are fixed indexes
 *   the learner cannot drop (DTA refuses to drop them too).
 * - Each query is planned against the current index set: a sequential scan, or any B-tree whose leading column the
 *   query can use. Equality columns from the front of the index form the seek; the next index column serves the
 *   ORDER BY (and lets a LIMIT stop early) or a range; any later index column with a predicate is checked inside the
 *   index before the heap fetch; other predicates are checked on the heap row. Predicates are independent (the
 *   planner's default assumption). No bitmap scans, no index-only scans, no BitmapAnd of two indexes.
 * - Heap pages fetched follow PostgreSQL's cost_index(): interpolate between one page per row (uncorrelated) and
 *   ceil(rows / rows-per-page) (perfectly correlated) by correlation². The plan is chosen by a planner-style I/O cost
 *   (random page = 4, sequential page = 1); the scoreboard counts pages.
 * - Constraint maintenance: the child-side lookups referential-integrity triggers run. DELETE on customers cascades
 *   `DELETE FROM ONLY orders WHERE $1 = customer_id`, and each deleted order cascades into order_items by order_id;
 *   DELETE on products runs `SELECT 1 FROM ONLY order_items x WHERE $1 = product_id FOR KEY SHARE OF x` (the product
 *   was never sold, so nothing matches). Parent-side checks on child INSERT probe a primary key and do not depend on
 *   the learner's choices.
 * - Index maintenance: an INSERT adds an entry to every index (root-to-leaf pages). An UPDATE in PostgreSQL is HOT
 *   only if no index references a changed column (page room assumed); otherwise every index gets a new entry. In
 *   InnoDB only secondary indexes containing a changed column change: the old entry is delete-marked and a new one
 *   inserted. DELETE writes no index entries in PostgreSQL (VACUUM removes them later); InnoDB delete-marks them.
 * - InnoDB requires an index whose leading columns are the foreign key columns and creates one if none exists.
 * - Index sizes: PostgreSQL B-tree leaves at the default 90% fillfactor, 8-byte index tuple header + key, MAXALIGNed,
 *   plus a 4-byte line pointer; no deduplication. Read costs and sizes use this model in both engine modes.
 */

export type Engine = 'postgres' | 'innodb';
const PAGE = 8192;
const RANDOM = 4;
const SEQ = 1;

export type Table = { name: string; rows: number; rowBytes: number; cols: Record<string, { bytes: number; corr: number }> };

export const TABLES: Record<string, Table> = {
  customers: { name: 'customers', rows: 500_000, rowBytes: 200, cols: { id: { bytes: 8, corr: 1 }, email: { bytes: 24, corr: 0 }, region: { bytes: 4, corr: 0 }, created_at: { bytes: 8, corr: 0.98 } } },
  orders: { name: 'orders', rows: 50_000_000, rowBytes: 110, cols: { id: { bytes: 8, corr: 1 }, customer_id: { bytes: 8, corr: 0 }, status: { bytes: 4, corr: 0 }, created_at: { bytes: 8, corr: 0.99 }, total: { bytes: 8, corr: 0 } } },
  order_items: { name: 'order_items', rows: 200_000_000, rowBytes: 64, cols: { id: { bytes: 8, corr: 1 }, order_id: { bytes: 8, corr: 0.98 }, product_id: { bytes: 8, corr: 0 }, qty: { bytes: 4, corr: 0 } } },
  products: { name: 'products', rows: 100_000, rowBytes: 320, cols: { id: { bytes: 8, corr: 1 }, sku: { bytes: 16, corr: 0 }, category: { bytes: 4, corr: 0 }, price: { bytes: 8, corr: 0 } } },
};
export const TABLE_ORDER = ['customers', 'orders', 'order_items', 'products'] as const;

export const rowsPerPage = (t: Table) => Math.floor(8000 / t.rowBytes);
export const heapPages = (t: Table) => Math.ceil(t.rows / rowsPerPage(t));

export type Index = { table: string; cols: string[]; kind?: 'pk' | 'unique' | 'auto' };
export const keyOf = (i: { table: string; cols: string[] }) => `${i.table}(${i.cols.join(', ')})`;

export const CONSTRAINT_INDEXES: Index[] = [
  { table: 'customers', cols: ['id'], kind: 'pk' },
  { table: 'customers', cols: ['email'], kind: 'unique' },
  { table: 'orders', cols: ['id'], kind: 'pk' },
  { table: 'order_items', cols: ['id'], kind: 'pk' },
  { table: 'products', cols: ['id'], kind: 'pk' },
  { table: 'products', cols: ['sku'], kind: 'unique' },
];

/** The set the learner inherits: typical accretion, one index per ticket. */
export const INHERITED: Index[] = [
  { table: 'customers', cols: ['region'] },
  { table: 'orders', cols: ['customer_id'] },
  { table: 'orders', cols: ['customer_id', 'created_at'] },
  { table: 'orders', cols: ['status'] },
  { table: 'orders', cols: ['total'] },
  { table: 'order_items', cols: ['order_id'] },
  { table: 'products', cols: ['category'] },
];

export type Pred = { col: string; op: 'eq' | 'range'; sel: number };
export type Query = { id: string; label: string; sql: string; table: string; preds: Pred[]; order?: string; limit?: number; perSec: number };

export const QUERIES: Query[] = [
  { id: 'Q1', label: 'Recent orders of a customer', sql: 'SELECT * FROM orders WHERE customer_id = $1 ORDER BY created_at DESC LIMIT 20', table: 'orders', preds: [{ col: 'customer_id', op: 'eq', sel: 1 / 500_000 }], order: 'created_at', limit: 20, perSec: 300 },
  { id: 'Q2', label: 'Customer orders by status', sql: 'SELECT * FROM orders WHERE customer_id = $1 AND status = $2', table: 'orders', preds: [{ col: 'customer_id', op: 'eq', sel: 1 / 500_000 }, { col: 'status', op: 'eq', sel: 0.2 }], perSec: 60 },
  { id: 'Q3', label: 'Newest pending orders', sql: "SELECT * FROM orders WHERE status = 'pending' ORDER BY created_at DESC LIMIT 50", table: 'orders', preds: [{ col: 'status', op: 'eq', sel: 0.002 }], order: 'created_at', limit: 50, perSec: 5 },
  { id: 'Q4', label: 'Items of an order', sql: 'SELECT * FROM order_items WHERE order_id = $1', table: 'order_items', preds: [{ col: 'order_id', op: 'eq', sel: 1 / 50_000_000 }], perSec: 800 },
  { id: 'Q5', label: 'Is a product in an order', sql: 'SELECT qty FROM order_items WHERE order_id = $1 AND product_id = $2', table: 'order_items', preds: [{ col: 'order_id', op: 'eq', sel: 1 / 50_000_000 }, { col: 'product_id', op: 'eq', sel: 1 / 100_000 }], perSec: 40 },
  { id: 'Q6', label: 'Log in by email', sql: 'SELECT * FROM customers WHERE email = $1', table: 'customers', preds: [{ col: 'email', op: 'eq', sel: 1 / 500_000 }], perSec: 150 },
  { id: 'Q7', label: 'Regional sign-ups report', sql: 'SELECT * FROM customers WHERE region = $1 AND created_at >= $2', table: 'customers', preds: [{ col: 'region', op: 'eq', sel: 0.02 }, { col: 'created_at', op: 'range', sel: 0.1 }], perSec: 0.05 },
  { id: 'Q8', label: 'Category page', sql: 'SELECT * FROM products WHERE category = $1 ORDER BY price LIMIT 50', table: 'products', preds: [{ col: 'category', op: 'eq', sel: 0.005 }], order: 'price', limit: 50, perSec: 150 },
  { id: 'Q9', label: 'Daily revenue', sql: 'SELECT sum(total) FROM orders WHERE created_at >= $1 AND created_at < $2', table: 'orders', preds: [{ col: 'created_at', op: 'range', sel: 0.001 }], perSec: 0.01 },
  { id: 'Q10', label: 'Product page by SKU', sql: 'SELECT * FROM products WHERE sku = $1', table: 'products', preds: [{ col: 'sku', op: 'eq', sel: 1 / 100_000 }], perSec: 400 },
];

export type Write = { id: string; sql: string; table: string; kind: 'insert' | 'update' | 'delete'; changed?: string[]; perSec: number };
export const WRITES: Write[] = [
  { id: 'W1', sql: 'INSERT INTO orders …', table: 'orders', kind: 'insert', perSec: 40 },
  { id: 'W2', sql: 'INSERT INTO order_items …', table: 'order_items', kind: 'insert', perSec: 160 },
  { id: 'W3', sql: 'UPDATE orders SET status = $2 WHERE id = $1', table: 'orders', kind: 'update', changed: ['status'], perSec: 80 },
  { id: 'W4', sql: 'INSERT INTO customers …', table: 'customers', kind: 'insert', perSec: 2 },
  { id: 'W5', sql: 'UPDATE products SET price = $2 WHERE id = $1', table: 'products', kind: 'update', changed: ['price'], perSec: 5 },
  { id: 'W6', sql: 'DELETE FROM customers WHERE id = $1', table: 'customers', kind: 'delete', perSec: 0.02 },
  { id: 'W7', sql: 'DELETE FROM products WHERE id = $1', table: 'products', kind: 'delete', perSec: 0.01 },
];

export type Fk = { id: string; child: string; col: string; parent: string; action: 'CASCADE' | 'RESTRICT'; childRowsPerParent: number };
export const FKS: Fk[] = [
  { id: 'FK1', child: 'orders', col: 'customer_id', parent: 'customers', action: 'CASCADE', childRowsPerParent: 100 },
  { id: 'FK2', child: 'order_items', col: 'order_id', parent: 'orders', action: 'CASCADE', childRowsPerParent: 4 },
  { id: 'FK3', child: 'order_items', col: 'product_id', parent: 'products', action: 'RESTRICT', childRowsPerParent: 0 },
];

/* ----------------------------------------------------------------- B-tree geometry */

const maxalign = (n: number) => Math.ceil(n / 8) * 8;
export function geometry(ix: { table: string; cols: string[] }) {
  const t = TABLES[ix.table];
  const key = ix.cols.reduce((s, c) => s + t.cols[c].bytes, 0);
  const entry = maxalign(8 + key) + 4;
  const perLeaf = Math.floor(((PAGE - 40) * 0.9) / entry);
  const leaves = Math.max(1, Math.ceil(t.rows / perLeaf));
  const fanout = Math.floor(((PAGE - 40) * 0.7) / entry);
  let height = 1;
  let level = leaves;
  while (level > 1) {
    level = Math.ceil(level / fanout);
    height++;
  }
  const inner = height > 1 ? Math.ceil(leaves / (fanout - 1)) : 0;
  return { entry, perLeaf, leaves, height, bytes: (leaves + inner) * PAGE };
}

/* ----------------------------------------------------------------- planning */

export type Plan = {
  kind: 'seq' | 'index';
  index?: string;
  quality: 'seek' | 'partial' | 'seq';
  pages: number;
  cost: number;
  seekCols: string[];
  inIndexFilter: string[];
  heapFilter: string[];
  sortNeeded: boolean;
  entries: number;
  heapFetches: number;
};

type Lookup = { table: string; preds: Pred[]; order?: string; limit?: number; actualMatches?: number };

function seqPlan(l: Lookup): Plan {
  const t = TABLES[l.table];
  const pages = heapPages(t);
  return { kind: 'seq', quality: 'seq', pages, cost: pages * SEQ, seekCols: [], inIndexFilter: [], heapFilter: l.preds.map((p) => p.col), sortNeeded: !!l.order, entries: 0, heapFetches: 0 };
}

export function indexPlan(l: Lookup, ix: Index): Plan | null {
  const t = TABLES[l.table];
  if (ix.table !== l.table) return null;
  const predOn = (c: string) => l.preds.find((p) => p.col === c);
  const seekCols: string[] = [];
  let seekSel = 1;
  let i = 0;
  while (i < ix.cols.length) {
    const p = predOn(ix.cols[i]);
    if (p && p.op === 'eq') {
      seekSel *= p.sel;
      seekCols.push(ix.cols[i]);
      i++;
    } else break;
  }
  let orderOk = false;
  if (i < ix.cols.length) {
    const c = ix.cols[i];
    const p = predOn(c);
    if (l.order && c === l.order) {
      orderOk = true;
      if (p) {
        seekSel *= p.sel;
        seekCols.push(c);
      }
      i++;
    } else if (p && p.op === 'range') {
      seekSel *= p.sel;
      seekCols.push(c);
      i++;
    }
  }
  // An index can serve a query only by seeking, or (for ORDER BY … LIMIT) by walking in order and stopping early.
  if (seekCols.length === 0 && !(orderOk && l.limit)) return null;
  const rest = ix.cols.slice(i);
  const inIndexFilter = l.preds.filter((p) => !seekCols.includes(p.col) && rest.includes(p.col));
  const heapFilter = l.preds.filter((p) => !seekCols.includes(p.col) && !ix.cols.includes(p.col));
  const fSel = inIndexFilter.reduce((s, p) => s * p.sel, 1);
  const hSel = heapFilter.reduce((s, p) => s * p.sel, 1);
  const matched = t.rows * seekSel;
  let entries = matched;
  if (l.limit && orderOk) entries = Math.min(matched, Math.ceil(l.limit / (fSel * hSel)));
  let fetches = entries * fSel;
  if (l.actualMatches !== undefined) {
    // an existence check that finds nothing reads one leaf and no heap pages
    entries = l.actualMatches;
    fetches = l.actualMatches;
  }
  const g = geometry(ix);
  const indexPages = g.height - 1 + Math.max(1, Math.ceil(entries / g.perLeaf));
  const tp = heapPages(t);
  const corr = t.cols[ix.cols[0]].corr;
  const c2 = corr * corr;
  const maxP = Math.min(fetches, tp);
  const minP = Math.ceil(fetches / rowsPerPage(t));
  const heap = fetches <= 0 ? 0 : maxP + c2 * (minP - maxP);
  const heapCost = fetches <= 0 ? 0 : maxP * RANDOM + c2 * (RANDOM + Math.max(0, minP - 1) * SEQ - maxP * RANDOM);
  const sortNeeded = !!l.order && !orderOk;
  const quality = inIndexFilter.length === 0 && heapFilter.length === 0 && !sortNeeded ? 'seek' : 'partial';
  return {
    kind: 'index',
    index: keyOf(ix),
    quality,
    pages: indexPages + heap,
    cost: indexPages * RANDOM + heapCost,
    seekCols,
    inIndexFilter: inIndexFilter.map((p) => p.col),
    heapFilter: heapFilter.map((p) => p.col),
    sortNeeded,
    entries,
    heapFetches: fetches,
  };
}

export function bestPlan(l: Lookup, set: Index[]): Plan {
  let best = seqPlan(l);
  for (const ix of set) {
    const p = indexPlan(l, ix);
    if (p && p.cost < best.cost) best = p;
  }
  return best;
}

/* ----------------------------------------------------------------- the set as a whole */

export type Opts = { engine: Engine; writeScale: number };

/** InnoDB silently adds an index for any foreign key whose columns do not lead an existing index. */
export function effectiveSet(user: Index[], engine: Engine): Index[] {
  const all = [...CONSTRAINT_INDEXES, ...user];
  if (engine === 'innodb') {
    for (const fk of FKS) {
      if (!all.some((ix) => ix.table === fk.child && ix.cols[0] === fk.col)) all.push({ table: fk.child, cols: [fk.col], kind: 'auto' });
    }
  }
  return all;
}

export const fkIndexed = (fk: Fk, set: Index[]) => set.some((ix) => ix.table === fk.child && ix.cols[0] === fk.col);

export type ConstraintRow = { id: string; table: string; label: string; perSec: number; lookups: number; plan: Plan; pagesPerEvent: number; fk?: string };

export function evaluate(user: Index[], o: Opts) {
  const set = effectiveSet(user, o.engine);
  const queries = QUERIES.map((q) => {
    const plan = bestPlan(q, set);
    return { q, plan, pagesPerSec: plan.pages * q.perSec };
  });
  const read = queries.reduce((s, r) => s + r.pagesPerSec, 0);

  // Child-side referential-integrity lookups (the reads nobody wrote).
  const w = (id: string) => WRITES.find((x) => x.id === id)!.perSec * o.writeScale;
  const custDel = w('W6');
  const prodDel = w('W7');
  const fk1 = bestPlan({ table: 'orders', preds: [{ col: 'customer_id', op: 'eq', sel: 1 / 500_000 }] }, set);
  const fk2 = bestPlan({ table: 'order_items', preds: [{ col: 'order_id', op: 'eq', sel: 1 / 50_000_000 }] }, set);
  const fk3 = bestPlan({ table: 'order_items', preds: [{ col: 'product_id', op: 'eq', sel: 1 / 100_000 }], actualMatches: 0 }, set);
  const constraintRows: ConstraintRow[] = [
    { id: 'FK1', table: 'orders', fk: 'FK1', label: 'DELETE customers → DELETE FROM ONLY orders WHERE $1 = customer_id', perSec: custDel, lookups: 1, plan: fk1, pagesPerEvent: fk1.pages },
    { id: 'FK2', table: 'order_items', fk: 'FK2', label: '… each deleted order → DELETE FROM ONLY order_items WHERE $1 = order_id', perSec: custDel * FKS[0].childRowsPerParent, lookups: FKS[0].childRowsPerParent, plan: fk2, pagesPerEvent: fk2.pages },
    { id: 'FK3', table: 'order_items', fk: 'FK3', label: 'DELETE products → SELECT 1 FROM ONLY order_items x WHERE $1 = product_id FOR KEY SHARE OF x', perSec: prodDel, lookups: 1, plan: fk3, pagesPerEvent: fk3.pages },
  ];
  // Parent-side checks on child inserts always probe a primary key: fixed cost, same for every index set.
  const pkProbe = (table: string) => geometry({ table, cols: ['id'] }).height + 1;
  const parentChecks = w('W1') * pkProbe('customers') + w('W2') * (pkProbe('orders') + pkProbe('products'));
  const childSide = constraintRows.reduce((s, r) => s + r.perSec * r.pagesPerEvent, 0);
  const constraint = childSide + parentChecks;

  // Index maintenance.
  const perIndexWrite = new Map<string, number>();
  const add = (ix: Index, n: number) => perIndexWrite.set(keyOf(ix), (perIndexWrite.get(keyOf(ix)) ?? 0) + n);
  let hotUpdates = 0;
  let nonHotUpdates = 0;
  for (const wr of WRITES) {
    const rate = wr.perSec * o.writeScale;
    const tableSet = set.filter((ix) => ix.table === wr.table);
    if (wr.kind === 'insert') {
      // InnoDB's primary key is the clustered table itself: that write is the row, not index maintenance.
      for (const ix of tableSet) if (!(o.engine === 'innodb' && ix.kind === 'pk')) add(ix, rate * geometry(ix).height);
    } else if (wr.kind === 'update') {
      const touches = (ix: Index) => ix.cols.some((c) => wr.changed!.includes(c));
      if (o.engine === 'postgres') {
        if (tableSet.some(touches)) {
          nonHotUpdates += rate;
          for (const ix of tableSet) add(ix, rate * geometry(ix).height);
        } else hotUpdates += rate;
      } else {
        for (const ix of tableSet) if (touches(ix) && ix.kind !== 'pk') add(ix, rate * 2 * geometry(ix).height);
      }
    } else if (o.engine === 'innodb') {
      // delete-mark every secondary entry of the deleted rows, including cascaded child rows
      const cascade: [string, number][] = wr.table === 'customers' ? [['customers', 1], ['orders', 100], ['order_items', 400]] : [[wr.table, 1]];
      for (const [tbl, rows] of cascade) for (const ix of set.filter((x) => x.table === tbl && x.kind !== 'pk')) add(ix, rate * rows * geometry(ix).height);
    }
  }
  const write = [...perIndexWrite.values()].reduce((s, v) => s + v, 0);
  const bytes = set.reduce((s, ix) => s + geometry(ix).bytes, 0);
  const unindexedFks = FKS.filter((fk) => !fkIndexed(fk, set));
  return { set, queries, read, constraintRows, parentChecks, childSide, constraint, write, perIndexWrite, bytes, total: read + constraint + write, unindexedFks, hotUpdates, nonHotUpdates };
}

/** Redundant prefix: a non-constraint index whose columns lead another index on the same table. */
export function redundantOf(ix: Index, set: Index[]): Index | undefined {
  if (ix.kind === 'pk' || ix.kind === 'unique') return undefined;
  return set.find((o) => o !== ix && o.table === ix.table && o.cols.length > ix.cols.length && ix.cols.every((c, i) => o.cols[i] === c));
}

export type Marginal = { ix: Index; key: string; readSaved: number; constraintSaved: number; writeAdded: number; bytes: number; net: number; serves: string[]; redundantOf?: string; servesNothing: boolean; fkLocked: boolean };

/** Leave-one-out value of each learner-controlled index in the current set. */
export function marginals(user: Index[], o: Opts): Marginal[] {
  const base = evaluate(user, o);
  return user.map((ix, i) => {
    const without = evaluate(user.filter((_, j) => j !== i), o);
    const k = keyOf(ix);
    const serves = base.queries.filter((r) => r.plan.index === k).map((r) => r.q.id);
    base.constraintRows.forEach((r) => {
      if (r.plan.index === k) serves.push(r.id);
    });
    const readSaved = without.read - base.read;
    const constraintSaved = without.constraint - base.constraint;
    const writeAdded = base.write - without.write;
    const red = redundantOf(ix, base.set);
    // In InnoDB the last index leading with a foreign key column cannot be dropped; "without" then holds the index InnoDB would create instead.
    const fkLocked = o.engine === 'innodb' && innodbDropRefusal([ix], user) !== null;
    return { ix, key: k, readSaved, constraintSaved, writeAdded, bytes: geometry(ix).bytes, net: readSaved + constraintSaved - writeAdded, serves, redundantOf: red ? keyOf(red) : undefined, servesNothing: serves.length === 0, fkLocked };
  });
}

/* ----------------------------------------------------------------- candidates, merging, advisor */

/** Order-preserving merge: keep a's columns, append b's columns that a lacks. */
export function merge(a: Index, b: Index): Index | null {
  if (a.table !== b.table) return null;
  return { table: a.table, cols: [...a.cols, ...b.cols.filter((c) => !a.cols.includes(c))] };
}

/** Equality columns, then the sort column, then the range column. */
export function idealFor(q: Query): Index {
  const eq = q.preds.filter((p) => p.op === 'eq').sort((a, b) => a.sel - b.sel).map((p) => p.col);
  const cols = [...eq];
  if (q.order && !cols.includes(q.order)) cols.push(q.order);
  for (const p of q.preds) if (p.op === 'range' && !cols.includes(p.col)) cols.push(p.col);
  return { table: q.table, cols };
}

const sameIndex = (a: Index, b: Index) => keyOf(a) === keyOf(b);

export function candidates(): Index[] {
  const out: Index[] = [];
  const push = (ix: Index) => {
    if (![...out, ...CONSTRAINT_INDEXES].some((o) => sameIndex(o, ix))) out.push(ix);
  };
  QUERIES.forEach((q) => push(idealFor(q)));
  FKS.forEach((fk) => push({ table: fk.child, cols: [fk.col] }));
  const base = out.slice();
  for (const a of base)
    for (const b of base) {
      if (a === b || a.table !== b.table || a.cols[0] !== b.cols[0]) continue;
      const m = merge(a, b);
      if (m) push(m);
    }
  return out;
}

export type AdvisorStep = { added: Index; totalBefore: number; totalAfter: number };

/** Greedy(0, k) in the AutoAdmin sense: repeatedly add the candidate that lowers total cost most; stop when none does. */
export function greedyAdvisor(o: Opts, perTableBudget: number) {
  const cands = candidates();
  let chosen: Index[] = [];
  let cur = evaluate(chosen, o).total;
  const steps: AdvisorStep[] = [];
  for (let guard = 0; guard < 20; guard++) {
    let best: { ix: Index; total: number } | null = null;
    for (const c of cands) {
      if (chosen.some((x) => sameIndex(x, c))) continue;
      const set = effectiveSet([...chosen, c], o.engine);
      if (set.filter((x) => x.table === c.table).length > perTableBudget) continue;
      const total = evaluate([...chosen, c], o).total;
      if (!best || total < best.total) best = { ix: c, total };
    }
    if (!best || best.total >= cur) break;
    steps.push({ added: best.ix, totalBefore: cur, totalAfter: best.total });
    chosen = [...chosen, best.ix];
    cur = best.total;
  }
  return { chosen, steps, total: cur, candidates: cands.length };
}


/* ----------------------------------------------------------------- UI */

export const WRITE_SCALES = [0.25, 0.5, 1, 2, 5, 10, 25, 50, 100];
export const fmtRate = (r: number) => (r >= 1 ? `${fmtNum(r, r < 10 ? 1 : 0)}/s` : `${fmtNum(r * 3600, 0)}/h`);
/** 1.3B, 2.97M, 23.5k — for the scoreboard, where exact digits would not fit. */
export const fmtCompact = (n: number) => {
  const a = Math.abs(n);
  const sign = n < 0 ? '−' : '';
  if (a >= 1e9) return `${sign}${fmtNum(a / 1e9, a >= 1e10 ? 1 : 2)}B`;
  if (a >= 1e6) return `${sign}${fmtNum(a / 1e6, a >= 1e7 ? 1 : 2)}M`;
  if (a >= 1e4) return `${sign}${fmtNum(a / 1e3, 1)}k`;
  return `${sign}${fmtNum(a)}`;
};
const fmtPages = (n: number) => (n >= 100 ? fmtNum(n) : fmtNum(n, n < 10 ? 2 : 1));
const colsOf = (ix: { cols: string[] }) => `(${ix.cols.join(', ')})`;
const EDGE: Record<Plan['quality'], string> = { seek: 'var(--viz-1)', partial: 'var(--viz-2)', seq: 'var(--viz-critical)' };
const RI_SHORT: Record<string, string> = {
  FK1: 'customers → orders.customer_id',
  FK2: '… each order → order_items.order_id',
  FK3: 'products → order_items.product_id',
};

/** Why InnoDB would refuse to drop this index: it is the last one that leads with a foreign key's column. */
export function innodbDropRefusal(dropping: Index[], user: Index[]): string | null {
  const keep = [...CONSTRAINT_INDEXES, ...user.filter((u) => !dropping.some((d) => sameIndex(d, u)))];
  for (const d of dropping) {
    for (const fk of FKS) {
      if (d.table === fk.child && d.cols[0] === fk.col && !keep.some((k) => k.table === fk.child && k.cols[0] === fk.col)) {
        return `ERROR 1553 (HY000): Cannot drop index '${keyOf(d)}': needed in a foreign key constraint. No other index would lead with ${fk.child}.${fk.col}, the foreign key to ${fk.parent}.`;
      }
    }
  }
  return null;
}

export default function WorkloadIndexSetLab() {
  const [engine, setEngine] = useState<Engine>('postgres');
  const [scaleIdx, setScaleIdx] = useState(2);
  const [replicas, setReplicas] = useState(1);
  const [budget, setBudget] = useState(4);
  const [user, setUser] = useState<Index[]>(INHERITED);
  const [selected, setSelected] = useState<string[]>([]);
  const [bTable, setBTable] = useState<string>('orders');
  const [bCols, setBCols] = useState<string[]>([]);
  const [idealQ, setIdealQ] = useState('Q3');
  const [focus, setFocus] = useState<string | null>(null);
  const [advisorOn, setAdvisorOn] = useState(false);
  const [message, setMessage] = useState<string | null>(null);

  const o: Opts = useMemo(() => ({ engine, writeScale: WRITE_SCALES[scaleIdx] }), [engine, scaleIdx]);
  const ev = useMemo(() => evaluate(user, o), [user, o]);
  const ms = useMemo(() => marginals(user, o), [user, o]);
  const adv = useMemo(() => (advisorOn ? greedyAdvisor(o, budget) : null), [advisorOn, o, budget]);
  const mByKey = new Map(ms.map((m) => [m.key, m]));

  const exists = (ix: Index) => [...CONSTRAINT_INDEXES, ...user].some((x) => sameIndex(x, ix));
  const addIndex = (ix: Index) => {
    if (ix.cols.length === 0) return;
    if (exists(ix)) {
      setMessage(`${keyOf(ix)} already exists.`);
      return;
    }
    setUser([...user, ix]);
    setMessage(`Created ${keyOf(ix)}.`);
  };
  const selectedIx = selected.map((k) => user.find((u) => keyOf(u) === k)).filter((x): x is Index => !!x);
  const dropSelected = () => {
    if (selectedIx.length === 0) return;
    if (engine === 'innodb') {
      const refusal = innodbDropRefusal(selectedIx, user);
      if (refusal) {
        setMessage(refusal);
        return;
      }
    }
    setUser(user.filter((u) => !selectedIx.some((s) => sameIndex(s, u))));
    setSelected([]);
    setMessage(`Dropped ${selectedIx.map(keyOf).join(' and ')}.`);
  };
  const mergeSelected = () => {
    if (selectedIx.length !== 2 || selectedIx[0].table !== selectedIx[1].table) return;
    const m = merge(selectedIx[0], selectedIx[1])!;
    if (engine === 'innodb') {
      const refusal = innodbDropRefusal(selectedIx.filter((s) => !sameIndex(s, m)), [...user, m]);
      if (refusal) {
        setMessage(refusal);
        return;
      }
    }
    const rest = user.filter((u) => !selectedIx.some((s) => sameIndex(s, u)));
    setUser(rest.some((r) => sameIndex(r, m)) || CONSTRAINT_INDEXES.some((c) => sameIndex(c, m)) ? rest : [...rest, m]);
    setSelected([]);
    setMessage(`Merged ${keyOf(selectedIx[0])} and ${keyOf(selectedIx[1])} into ${keyOf(m)}.`);
  };
  const toggleSel = (k: string) => setSelected(selected.includes(k) ? selected.filter((x) => x !== k) : [...selected, k]);

  const perTable = (t: string) => ev.set.filter((x) => x.table === t).length;
  const overBudget = TABLE_ORDER.filter((t) => perTable(t) > budget);
  const flagged = ms.filter((m) => m.redundantOf || m.servesNothing);
  const seqQueries = ev.queries.filter((r) => r.plan.kind === 'seq');
  const worst = [...ev.queries].sort((a, b) => b.pagesPerSec - a.pagesPerSec)[0];
  const ordersStatusIx = ev.set.filter((x) => x.table === 'orders' && x.cols.includes('status'));

  /* ------------------------------------------------ SVG geometry */
  const W = 600;
  const NODE_X: Record<string, number> = { customers: 50, orders: 214, order_items: 382, products: 550 };
  const NODE_W = 88;
  const rowH = 20;
  const LX = 246; // left column: dot position
  const RX = 352; // right column start
  const qy = (i: number) => 132 + i * rowH;
  const riHeaderY = qy(QUERIES.length) + 8;
  const ry = (i: number) => riHeaderY + 18 + i * rowH;
  const leftBottom = ry(3) + 8;
  type Row = { kind: 'header'; table: string; y: number } | { kind: 'scan'; table: string; y: number } | { kind: 'index'; ix: Index; y: number };
  const layout: Row[] = [];
  let yy = 128;
  for (const t of TABLE_ORDER) {
    layout.push({ kind: 'header', table: t, y: yy });
    yy += 17;
    layout.push({ kind: 'scan', table: t, y: yy });
    yy += 19;
    for (const ix of ev.set.filter((x) => x.table === t)) {
      layout.push({ kind: 'index', ix, y: yy });
      yy += 19;
    }
    yy += 5;
  }
  const H = Math.max(leftBottom, yy) + 4;
  const targetY = new Map<string, number>();
  for (const l of layout) {
    if (l.kind === 'index') targetY.set(keyOf(l.ix), l.y);
    if (l.kind === 'scan') targetY.set(`scan:${l.table}`, l.y);
  }
  const dim = (keys: string[]) => (focus && !keys.includes(focus) ? 0.12 : 1);
  const targetOf = (plan: Plan, table: string) => plan.index ?? `scan:${table}`;

  const edge = (y1: number, target: string, quality: Plan['quality'], ids: string[]) => {
    const y2 = targetY.get(target);
    if (y2 === undefined) return null;
    const on = focus !== null && (focus === target || ids.includes(focus));
    return (
      <path
        d={`M ${LX + 5} ${y1} C ${LX + 55} ${y1}, ${RX - 55} ${y2}, ${RX - 3} ${y2}`}
        fill="none"
        stroke={EDGE[quality]}
        strokeWidth={on ? 3 : 1.5}
        strokeDasharray={quality === 'partial' ? '5 3' : undefined}
        opacity={dim([target, ...ids])}
      />
    );
  };

  const flagText = (ix: Index) => {
    if (ix.kind === 'pk') return 'PK';
    if (ix.kind === 'unique') return 'UNIQUE';
    if (ix.kind === 'auto') return 'auto FK';
    const m = mByKey.get(keyOf(ix));
    if (m?.redundantOf) return 'redundant';
    if (m?.servesNothing) return 'unused';
    return '';
  };

  const narrate = () => {
    const parts: ReactNode[] = [];
    if (message) parts.push(<span key="m"><strong>{message}</strong> </span>);
    const fkRow = ev.constraintRows.find((r) => r.plan.kind === 'seq');
    if (ev.unindexedFks.length > 0 && fkRow) {
      const fk = FKS.find((f) => f.id === fkRow.fk)!;
      parts.push(
        <span key="fk">
          <strong>{fk.child}.{fk.col} leads no index.</strong> Every {fk.parent === 'customers' ? 'cascaded delete' : `DELETE on ${fk.parent}`} makes the trigger scan all {fmtNum(fkRow.pagesPerEvent)} pages of {fk.child} — {fmtNum(fkRow.perSec * fkRow.pagesPerEvent)} pages/s from {fmtRate(fkRow.perSec)} of events that appear in no query inventory.{' '}
        </span>,
      );
    } else if (flagged.length > 0) {
      const f0 = flagged[0];
      parts.push(
        <span key="fl">
          <strong>{f0.key} {f0.redundantOf ? `is a redundant prefix of ${f0.redundantOf}` : 'serves no query and no constraint lookup'}.</strong> Dropping it removes {fmtNum(f0.writeAdded)} pages/s of index maintenance and {fmtBytes(f0.bytes)}, and costs {fmtNum(Math.max(0, f0.readSaved + f0.constraintSaved))} pages/s of reads.{' '}
        </span>,
      );
    } else if (seqQueries.length > 0) {
      const s = seqQueries[0];
      parts.push(
        <span key="sq">
          <strong>{s.q.id} ({s.q.label}) still runs a sequential scan</strong> of {fmtNum(s.plan.pages)} pages, {fmtNum(s.pagesPerSec)} pages/s at {fmtRate(s.q.perSec)}.{' '}
        </span>,
      );
    } else {
      parts.push(
        <span key="ok">
          <strong>Every query and every constraint lookup uses an index.</strong> The most expensive statement is {worst.q.id} at {fmtNum(worst.pagesPerSec)} pages/s.{' '}
        </span>,
      );
    }
    if (engine === 'postgres') {
      parts.push(
        <span key="hot">
          {ordersStatusIx.length > 0
            ? `${fmtRate(WRITE_SCALES[scaleIdx] * 80)} status updates are non-HOT because ${ordersStatusIx.map(keyOf).join(' and ')} contain${ordersStatusIx.length === 1 ? 's' : ''} status, so every one of orders' ${perTable('orders')} indexes gets a new entry.`
            : `No orders index contains status, so status updates can stay HOT and write no index entries.`}
        </span>,
      );
    } else {
      parts.push(
        <span key="inno">
          {ordersStatusIx.length > 0
            ? `InnoDB rewrites only the ${ordersStatusIx.length} orders ${ordersStatusIx.length === 1 ? 'index that contains' : 'indexes that contain'} status on a status update: delete-mark the old entry, insert a new one.`
            : 'No orders index contains status, so InnoDB touches no secondary index on a status update.'}
        </span>,
      );
    }
    return parts;
  };

  const tableCols = Object.keys(TABLES[bTable].cols);

  return (
    <VizPanel
      title="One index set, the whole workload"
      subtitle="Ten queries with their rates, a write stream and three foreign keys run against the index set on the right. Add, merge and drop indexes; the planner re-plans every query and every constraint lookup, and the scoreboard re-totals pages touched per second."
      controls={
        <>
          <Segmented
            label="Engine"
            value={engine}
            onChange={(v) => {
              setEngine(v);
              setMessage(null);
            }}
            options={[
              { value: 'postgres', label: 'PostgreSQL' },
              { value: 'innodb', label: 'InnoDB' },
            ]}
          />
          <Slider label="Write rate" min={0} max={WRITE_SCALES.length - 1} value={scaleIdx} onChange={setScaleIdx} format={(v) => `×${WRITE_SCALES[v]}`} />
          <Slider label="Index budget per table" min={2} max={6} value={budget} onChange={setBudget} />
          <Slider label="Physical standbys" min={0} max={4} value={replicas} onChange={setReplicas} />
        </>
      }
      legend={
        <Legend
          items={[
            { label: 'Index seek answers it', color: 'var(--viz-1)', shape: 'line' },
            { label: 'Seek, then filter or sort (dashed)', color: 'var(--viz-2)', shape: 'line' },
            { label: 'Scans the whole table', color: 'var(--viz-critical)', shape: 'dot' },
            { label: 'Index flagged: redundant or unused', color: 'var(--viz-warning)' },
            { label: 'Selected index', color: 'var(--viz-7)' },
          ]}
        />
      }
      stats={
        <Stats
          items={[
            { label: 'Query reads, pages/s', value: fmtNum(ev.read), hint: 'Pages each query touches with its chosen plan, times its rate.' },
            { label: 'Constraint lookups, pages/s', value: fmtNum(ev.constraint), hint: `Child-side RI lookups ${fmtNum(ev.childSide)} + parent-key probes on child inserts ${fmtNum(ev.parentChecks)} (the probes use primary keys and never change).` },
            { label: 'Index maintenance, pages/s', value: fmtNum(ev.write), hint: 'Root-to-leaf pages touched to add (or delete-mark) index entries.' },
            { label: 'Total, pages/s', value: fmtNum(ev.total) },
            { label: `Index storage × ${1 + replicas} cop${replicas === 0 ? 'y' : 'ies'}`, value: fmtBytes(ev.bytes * (1 + replicas)), hint: `${fmtBytes(ev.bytes)} on the primary. A physical standby replays the same WAL, so it stores and maintains the same indexes.` },
            { label: 'Tables over budget', value: overBudget.length === 0 ? 'none' : overBudget.join(', ') },
          ]}
        />
      }
      note={<Note>{narrate()}</Note>}
      table={
        <>
          <table className="viz-table">
            <thead>
              <tr>
                <th>Query</th>
                <th>Rate</th>
                <th>Plan</th>
                <th>Seek on</th>
                <th>Filter in index</th>
                <th>Filter on heap</th>
                <th>Sort</th>
                <th>Pages per call</th>
                <th>Pages/s</th>
              </tr>
            </thead>
            <tbody>
              {ev.queries.map((r) => (
                <tr key={r.q.id}>
                  <td>
                    {r.q.id} <code>{r.q.sql}</code>
                  </td>
                  <td>{fmtRate(r.q.perSec)}</td>
                  <td>{r.plan.index ?? 'Seq Scan'}</td>
                  <td>{r.plan.seekCols.join(', ') || '—'}</td>
                  <td>{r.plan.inIndexFilter.join(', ') || '—'}</td>
                  <td>{r.plan.heapFilter.join(', ') || '—'}</td>
                  <td>{r.plan.sortNeeded ? 'yes' : 'no'}</td>
                  <td>{fmtPages(r.plan.pages)}</td>
                  <td>{fmtNum(r.pagesPerSec)}</td>
                </tr>
              ))}
              {ev.constraintRows.map((r) => (
                <tr key={r.id}>
                  <td>{r.label}</td>
                  <td>{fmtRate(r.perSec)}</td>
                  <td>{r.plan.index ?? 'Seq Scan'}</td>
                  <td>{r.plan.seekCols.join(', ') || '—'}</td>
                  <td>—</td>
                  <td>{r.plan.heapFilter.join(', ') || '—'}</td>
                  <td>no</td>
                  <td>{fmtPages(r.pagesPerEvent)}</td>
                  <td>{fmtNum(r.perSec * r.pagesPerEvent)}</td>
                </tr>
              ))}
            </tbody>
          </table>
          <table className="viz-table">
            <thead>
              <tr>
                <th>Index</th>
                <th>Height</th>
                <th>Size</th>
                <th>Maintenance pages/s</th>
              </tr>
            </thead>
            <tbody>
              {ev.set.map((ix) => (
                <tr key={keyOf(ix)}>
                  <td>
                    {keyOf(ix)} {ix.kind ? `(${flagText(ix)})` : ''}
                  </td>
                  <td>{geometry(ix).height}</td>
                  <td>{fmtBytes(geometry(ix).bytes)}</td>
                  <td>{fmtNum(ev.perIndexWrite.get(keyOf(ix)) ?? 0)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </>
      }
    >
      <svg
        viewBox={`0 0 ${W} ${H}`}
        width={W}
        height={H}
        role="img"
        aria-label={`Workload map: ${ev.queries.filter((r) => r.plan.kind === 'index').length} of 10 queries use an index, ${ev.unindexedFks.length} foreign key columns unindexed, ${ev.set.length} indexes.`}
        style={{ minWidth: 560 }}
        onMouseLeave={() => setFocus(null)}
      >
        {/* foreign-key graph */}
        <text x={4} y={11} fontSize={11} fill="var(--viz-ink)" fontWeight={600}>
          Foreign keys: deleting a parent row runs a lookup on the child column
        </text>
        {TABLE_ORDER.map((t) => (
          <g key={t}>
            <rect x={NODE_X[t] - NODE_W / 2} y={46} width={NODE_W} height={24} rx={5} fill="var(--viz-surface)" stroke="var(--viz-ink-muted)" />
            <text x={NODE_X[t]} y={62} textAnchor="middle" fontSize={11} fill="var(--viz-ink)">
              {t}
            </text>
          </g>
        ))}
        {FKS.map((fk) => {
          const px = NODE_X[fk.parent];
          const cx = NODE_X[fk.child];
          const dir = cx > px ? 1 : -1;
          const x1 = px + (dir * NODE_W) / 2;
          const x2 = cx - (dir * NODE_W) / 2;
          const mid = (x1 + x2) / 2;
          const bad = !fkIndexed(fk, ev.set);
          const stroke = bad ? 'var(--viz-critical)' : 'var(--viz-ink-muted)';
          return (
            <g key={fk.id} opacity={dim([fk.id])} onMouseEnter={() => setFocus(fk.id)}>
              <title>{`${fk.child}.${fk.col} REFERENCES ${fk.parent}(id) ON DELETE ${fk.action}${bad ? ' — no index leads with this column' : ''}`}</title>
              <line x1={x1} x2={x2 - dir * 6} y1={58} y2={58} stroke={stroke} strokeWidth={bad ? 2.5 : 1.5} />
              <path d={`M ${x2} 58 L ${x2 - dir * 7} 54 L ${x2 - dir * 7} 62 Z`} fill={stroke} />
              <text x={mid} y={29} textAnchor="middle" fontSize={10} fill="var(--viz-ink)">
                {fk.col}
              </text>
              <text x={mid} y={40} textAnchor="middle" fontSize={9}>
                {fk.action}
              </text>
              <text x={mid} y={84} textAnchor="middle" fontSize={9.5} fill={bad ? 'var(--viz-ink)' : undefined} fontWeight={bad ? 600 : undefined}>
                {bad ? 'unindexed: seq scan' : 'indexed'}
              </text>
            </g>
          );
        })}

        <text x={4} y={114} fontSize={11} fill="var(--viz-ink)" fontWeight={600}>
          Queries and rates
        </text>
        <text x={RX} y={114} fontSize={11} fill="var(--viz-ink)" fontWeight={600}>
          Index set (click an index to select it)
        </text>

        {/* edges first, so labels sit on top */}
        {ev.queries.map((r, i) => (
          <g key={`e${r.q.id}`}>{edge(qy(i), targetOf(r.plan, r.q.table), r.plan.quality, [r.q.id])}</g>
        ))}
        {ev.constraintRows.map((r, i) => (
          <g key={`e${r.id}`}>{edge(ry(i), targetOf(r.plan, r.table), r.plan.quality, [r.id])}</g>
        ))}

        {ev.queries.map((r, i) => {
          const y = qy(i);
          const t = targetOf(r.plan, r.q.table);
          return (
            <g key={r.q.id} onMouseEnter={() => setFocus(r.q.id)} opacity={focus && focus !== r.q.id && focus !== t ? 0.4 : 1}>
              <title>{`${r.q.sql}\n${r.plan.index ? `Index ${r.plan.index}` : 'Seq Scan'} · ${fmtPages(r.plan.pages)} pages per call · ${fmtNum(r.pagesPerSec)} pages/s`}</title>
              <rect x={0} y={y - 9} width={LX + 8} height={rowH - 2} fill="transparent" />
              <text x={4} y={y + 4} fontSize={10.5} fill="var(--viz-ink)" fontWeight={600}>
                {r.q.id}
              </text>
              <text x={30} y={y + 4} fontSize={10.5} fill="var(--viz-ink)">
                {r.q.label}
              </text>
              <text x={LX - 8} y={y + 4} textAnchor="end" fontSize={10}>
                {fmtRate(r.q.perSec)}
              </text>
              <circle cx={LX} cy={y} r={3.5} fill={EDGE[r.plan.quality]} />
            </g>
          );
        })}
        <text x={4} y={riHeaderY + 4} fontSize={11} fill="var(--viz-ink)" fontWeight={600}>
          FK lookups on parent DELETE
        </text>
        {ev.constraintRows.map((r, i) => {
          const y = ry(i);
          const t = targetOf(r.plan, r.table);
          return (
            <g key={r.id} onMouseEnter={() => setFocus(r.id)} opacity={focus && focus !== r.id && focus !== t ? 0.4 : 1}>
              <title>{`${r.label}\n${r.plan.index ? `Index ${r.plan.index}` : 'Seq Scan'} · ${fmtPages(r.pagesPerEvent)} pages per lookup · ${fmtNum(r.perSec * r.pagesPerEvent)} pages/s`}</title>
              <rect x={0} y={y - 9} width={LX + 8} height={rowH - 2} fill="transparent" />
              <text x={4} y={y + 4} fontSize={10} fill="var(--viz-ink)">
                {RI_SHORT[r.id]}
              </text>
              <text x={LX - 8} y={y + 4} textAnchor="end" fontSize={10}>
                {fmtRate(r.perSec)}
              </text>
              <circle cx={LX} cy={y} r={3.5} fill={EDGE[r.plan.quality]} />
            </g>
          );
        })}

        {/* right: index set, one block per table */}
        {layout.map((l) => {
          if (l.kind === 'header') {
            const n = perTable(l.table);
            const over = n > budget;
            return (
              <text key={`h${l.table}`} x={RX} y={l.y + 4} fontSize={10.5} fill="var(--viz-ink)" fontWeight={600}>
                {l.table}
                <tspan fontWeight={over ? 600 : 400} fill={over ? 'var(--viz-ink)' : 'var(--viz-ink-2)'}>
                  {`  ${n} of ${budget} indexes${over ? ' · over budget' : ''}`}
                </tspan>
              </text>
            );
          }
          if (l.kind === 'scan') {
            const k = `scan:${l.table}`;
            const used = ev.queries.some((r) => r.q.table === l.table && r.plan.kind === 'seq') || ev.constraintRows.some((r) => r.plan.kind === 'seq' && targetOf(r.plan, r.table) === k);
            return (
              <g key={k} onMouseEnter={() => setFocus(k)} opacity={focus && focus !== k ? 0.4 : 1}>
                <title>{`Sequential scan of ${l.table}: ${fmtNum(heapPages(TABLES[l.table]))} heap pages`}</title>
                <rect x={RX} y={l.y - 8} width={W - RX - 2} height={16} rx={3} fill="var(--viz-surface)" stroke={used ? 'var(--viz-critical)' : 'var(--viz-border)'} strokeDasharray="3 2" />
                <text x={RX + 8} y={l.y + 4} fontSize={9.5} fill={used ? 'var(--viz-ink)' : undefined}>
                  full table scan · {fmtNum(heapPages(TABLES[l.table]))} pages
                </text>
              </g>
            );
          }
          const k = keyOf(l.ix);
          const m = mByKey.get(k);
          const isUser = !l.ix.kind;
          const sel = selected.includes(k);
          const warn = !!(m && (m.redundantOf || m.servesNothing));
          const tag = flagText(l.ix);
          return (
            <g
              key={k}
              onMouseEnter={() => setFocus(k)}
              onClick={isUser ? () => toggleSel(k) : undefined}
              style={{ cursor: isUser ? 'pointer' : 'default' }}
              opacity={focus && focus !== k && !ev.queries.some((r) => r.q.id === focus && r.plan.index === k) && !ev.constraintRows.some((r) => r.id === focus && r.plan.index === k) ? 0.4 : 1}
            >
              <title>{`${k} · ${fmtBytes(geometry(l.ix).bytes)}${m ? ` · serves ${m.serves.join(', ') || 'nothing'}; saves ${fmtNum(m.readSaved + m.constraintSaved)} pages/s, costs ${fmtNum(m.writeAdded)} pages/s to maintain` : ''}`}</title>
              <rect x={RX} y={l.y - 8} width={W - RX - 2} height={16} rx={3} fill="var(--viz-surface)" stroke={sel ? 'var(--viz-7)' : 'var(--viz-border)'} strokeWidth={sel ? 2 : 1} />
              {warn ? <rect x={RX} y={l.y - 8} width={5} height={16} rx={1} fill="var(--viz-warning)" /> : null}
              <text x={RX + 10} y={l.y + 4} fontSize={10} fill="var(--viz-ink)">
                {colsOf(l.ix)}
              </text>
              {tag ? (
                <text x={W - 8} y={l.y + 4} textAnchor="end" fontSize={9} fill={warn ? 'var(--viz-ink)' : undefined} fontWeight={warn ? 600 : undefined}>
                  {tag}
                </text>
              ) : null}
            </g>
          );
        })}
      </svg>

      <div className="viz-controls" style={{ marginTop: '0.75rem' }}>
        <Choice
          label="Build an index on"
          value={bTable}
          onChange={(v) => {
            setBTable(v);
            setBCols([]);
          }}
          options={TABLE_ORDER.map((t) => ({ value: t, label: t }))}
        />
        <div className="viz-control">
          <span>Columns, in key order</span>
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: 4 }}>
            {tableCols.map((c) => (
              <Button key={c} onClick={() => setBCols(bCols.includes(c) ? bCols.filter((x) => x !== c) : [...bCols, c])} title={bCols.includes(c) ? 'Remove from the key' : 'Append to the key'}>
                {bCols.includes(c) ? `${bCols.indexOf(c) + 1}. ${c}` : c}
              </Button>
            ))}
          </div>
        </div>
        <Button primary disabled={bCols.length === 0} onClick={() => { addIndex({ table: bTable, cols: bCols }); setBCols([]); }}>
          {bCols.length ? `Create ${bTable}(${bCols.join(', ')})` : 'Create index'}
        </Button>
      </div>
      <div className="viz-controls">
        <Choice
          label="Ideal index for one query (equality, sort, range)"
          value={idealQ}
          onChange={setIdealQ}
          options={QUERIES.map((q) => ({ value: q.id, label: `${q.id} → ${keyOf(idealFor(q))}` }))}
        />
        <Button onClick={() => addIndex(idealFor(QUERIES.find((q) => q.id === idealQ)!))}>Add it</Button>
        <Button onClick={dropSelected} disabled={selectedIx.length === 0}>
          Drop selected
        </Button>
        <Button onClick={mergeSelected} disabled={!(selectedIx.length === 2 && selectedIx[0].table === selectedIx[1].table)} title="Keep the first selected index's columns and append the second's missing columns">
          Merge 2 selected
        </Button>
        <Button
          onClick={() => {
            setUser(INHERITED);
            setSelected([]);
            setMessage(null);
          }}
        >
          Reset to inherited set
        </Button>
      </div>

      <div style={{ overflowX: 'auto' }}>
        <table className="viz-table" aria-label="Leave-one-out value of each index in the current set, in pages per second">
          <thead>
            <tr>
              <th aria-label="Select" />
              <th>Index</th>
              <th>Serves</th>
              <th title="Query pages per second that would come back if this index were dropped">Reads saved</th>
              <th title="Constraint-lookup pages per second that would come back if this index were dropped">Lookups saved</th>
              <th title="Index maintenance pages per second this index adds">Upkeep</th>
              <th>Net</th>
              <th>Size</th>
            </tr>
          </thead>
          <tbody>
            {ms.map((m) => (
              <tr key={m.key} onMouseEnter={() => setFocus(m.key)} onMouseLeave={() => setFocus(null)} data-index={m.key}>
                <td>
                  <input type="checkbox" aria-label={`Select ${m.key}`} checked={selected.includes(m.key)} onChange={() => toggleSel(m.key)} />
                </td>
                <td>
                  <code>{m.key}</code>
                  {m.redundantOf ? <div>redundant prefix of {m.redundantOf}</div> : m.servesNothing ? <div>serves nothing</div> : null}
                  {m.fkLocked ? <div>last index on a foreign key: InnoDB refuses the drop, and these numbers compare it with the index InnoDB would create instead</div> : null}
                </td>
                <td>{m.serves.join(' ') || '—'}</td>
                <td>{fmtCompact(m.readSaved)}</td>
                <td>{fmtCompact(m.constraintSaved)}</td>
                <td>{fmtCompact(m.writeAdded)}</td>
                <td style={{ fontWeight: 600 }}>{fmtCompact(m.net)}</td>
                <td>{fmtBytes(m.bytes)}</td>
              </tr>
            ))}
            {ev.set
              .filter((x) => x.kind === 'auto')
              .map((x) => (
                <tr key={keyOf(x)}>
                  <td />
                  <td>
                    <code>{keyOf(x)}</code>
                    <div>created by InnoDB for the foreign key</div>
                  </td>
                  <td colSpan={5}>InnoDB will not let the last index on a foreign key column be dropped.</td>
                  <td>{fmtBytes(geometry(x).bytes)}</td>
                </tr>
              ))}
          </tbody>
        </table>
        <p style={{ margin: '0.25rem 0 0', fontSize: '0.75rem', color: 'var(--viz-ink-2)' }}>Pages per second, measured by dropping each index alone and re-planning everything.</p>
      </div>

      <div className="viz-controls" style={{ marginTop: '0.75rem', marginBottom: '0.25rem' }}>
        <Button onClick={() => setAdvisorOn(!advisorOn)}>{advisorOn ? 'Hide greedy advisor' : 'Run greedy advisor'}</Button>
        {adv ? (
          <Button
            primary
            onClick={() => {
              setUser(adv.chosen);
              setSelected([]);
              setMessage(`Applied the advisor's ${adv.chosen.length} indexes.`);
            }}
          >
            Apply its set
          </Button>
        ) : null}
      </div>
      {adv ? (
        <div style={{ fontSize: '0.8rem', color: 'var(--viz-ink)' }}>
          <p style={{ margin: '0.25rem 0' }}>
            Starting from the constraint indexes alone, add whichever of {adv.candidates} candidates lowers the total most; stop when nothing lowers it or a table reaches {budget} indexes.
          </p>
          <ol style={{ margin: 0, paddingLeft: '1.2rem', display: 'grid', gap: 2 }}>
            {adv.steps.map((s) => (
              <li key={keyOf(s.added)}>
                + {keyOf(s.added)}: total {fmtNum(s.totalBefore)} → {fmtNum(s.totalAfter)} pages/s
              </li>
            ))}
          </ol>
          <p style={{ margin: '0.25rem 0' }}>
            Advisor total {fmtNum(adv.total)} pages/s with {adv.chosen.length} indexes; your set {fmtNum(ev.total)} pages/s with {user.length}.
          </p>
        </div>
      ) : null}
    </VizPanel>
  );
}
