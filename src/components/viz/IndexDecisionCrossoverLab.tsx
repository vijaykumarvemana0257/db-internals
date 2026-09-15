import { useDeferredValue, useMemo, useRef, useState } from 'react';
import { VizPanel, Segmented, Choice, Slider, Check, Button, Legend, Stats, Note, makeRng, useTicker, fmtBytes, fmtNum } from './Viz';

/**
 * Two labs for "should this index exist?".
 *
 * THE BET — a range or equality predicate on a 1,000,000-row `orders` table.
 * - Rows are placed on heap pages by a deterministic simulation: a fraction `p` of rows sits in index-key order,
 *   the rest are shuffled among themselves. pg_stats.correlation is then measured the way ANALYZE does it
 *   (rank of value vs rank of physical position, over a 30,000-row sample) — it comes out ≈ p.
 * - Plan costs follow PostgreSQL 17's formulas: cost_seqscan; btcostestimate + genericcostestimate + cost_index
 *   (Mackert–Lohman heap pages, interpolated by correlation²); compute_bitmap_pages + cost_bitmap_heap_scan
 *   (per-page cost interpolated by √(pages/T)). Defaults: seq_page_cost 1, cpu_tuple_cost 0.01,
 *   cpu_index_tuple_cost 0.005, cpu_operator_cost 0.0025, effective_cache_size 4 GB (larger than these tables,
 *   so the cache term never binds). Estimates equal the true row counts; SELECT * so no index-only scan;
 *   the bitmap always fits in work_mem (no lossy pages). Serial plans only: with bitmap scans off, PostgreSQL 17's
 *   default max_parallel_workers_per_gather = 2 lets a Parallel Seq Scan beat the Index Scan earlier than shown.
 * - B-tree size: 8-byte key, 20 bytes per leaf item, leaves built at fillfactor 90 and internal pages at 70,
 *   no deduplication. The table already has a primary key, so a new index is the second B-tree per INSERT.
 * - "Pages read" are simulated with a cold cache: distinct heap pages holding a match, plus index pages.
 * - CLUSTER needs the index; after it the table is in key order. Rows changed since CLUSTER are modeled as
 *   landing on random pages, so correlation decays to ≈ 1 − fraction changed.
 *
 * FOREIGN-KEY LOOKUPS — DELETE on customers(10k) → orders(1M, ON DELETE action chosen) → order_items(4M, CASCADE).
 * - PostgreSQL issues one RI query per affected parent row (AFTER ROW triggers in ri_triggers.c): NO ACTION runs
 *   SELECT 1 … FOR KEY SHARE with LIMIT 1 semantics, CASCADE a DELETE, SET NULL an UPDATE. The lab deletes
 *   customers with no orders under NO ACTION (the only way that delete succeeds), so every check finds nothing.
 * - Without an index each lookup reads the whole child heap; with one, a 3-page descent plus one heap page per
 *   matching row (matches assumed on distinct pages). Oracle takes a full table lock on an unindexed child once per
 *   parent row modified. InnoDB requires the index and creates it automatically.
 * - 100 orders per customer and 4 items per order, 100 rows per heap page.
 */

/* ================================================================ model: the bet */

export const N_ROWS = 1_000_000;
export const ROWS_PER_PAGE = 100;
export const PAGE_BYTES = 8192;
export const COSTS = { seq: 1.0, cpuTuple: 0.01, cpuIndexTuple: 0.005, cpuOperator: 0.0025, effectiveCacheSizePages: (4 * 1024 * 1024 * 1024) / PAGE_BYTES };
export type PlanName = 'Seq Scan' | 'Index Scan' | 'Bitmap Heap Scan';
export type Bet = 'index' | 'leave' | 'cluster';

/** Size of a B-tree built by CREATE INDEX over `entries` 8-byte keys (no deduplication). */
export function btreeGeometry(entries: number) {
  const usable = PAGE_BYTES - 24 - 16; // page header, BTPageOpaqueData
  const item = 20; // IndexTupleData (8) + int8 key (8) + line pointer (4)
  const leafCap = Math.floor((usable - PAGE_BYTES * 0.1) / item); // leaves left 10% free (fillfactor 90)
  const innerCap = Math.floor((usable - PAGE_BYTES * 0.3) / item); // internal pages at fillfactor 70
  const leaves = Math.max(1, Math.ceil(entries / leafCap));
  let level = leaves;
  let pages = leaves;
  let height = 0;
  while (level > 1) {
    level = Math.ceil(level / innerCap);
    pages += level;
    height++;
  }
  pages += 1; // metapage
  return { leafCap, innerCap, leaves, height, pages, bytes: pages * PAGE_BYTES };
}

/** costsize.c index_pages_fetched (Mackert–Lohman), one table plus one index competing for the cache. */
export function indexPagesFetched(tuples: number, T: number, indexPages: number, ecsPages = COSTS.effectiveCacheSizePages) {
  const totalPages = Math.max(1, T + indexPages);
  let b = (ecsPages * T) / totalPages;
  b = b <= 1 ? 1 : Math.ceil(b);
  if (T <= b) {
    const pf = (2 * T * tuples) / (2 * T + tuples);
    return pf >= T ? T : Math.ceil(pf);
  }
  const lim = (2 * T * b) / (2 * T - b);
  const pf = tuples <= lim ? (2 * T * tuples) / (2 * T + tuples) : b + ((tuples - lim) * (T - b)) / T;
  return Math.ceil(pf);
}

export type PlanCosts = {
  seq: number;
  index: number;
  bitmap: number;
  indexPart: number;
  tuples: number;
  indexLeafPages: number;
  mlPages: number;
  bitmapPages: number;
  best: PlanName;
  bestCost: number;
  bestIndexPath: PlanName;
  bestIndexPathCost: number;
};

/** PostgreSQL 17 cost estimates for the three access paths over `orders`. */
export function planCosts(sel: number, corr: number, rpc: number, quals: number, bitmapOn = true): PlanCosts {
  const N = N_ROWS;
  const T = Math.ceil(N / ROWS_PER_PAGE);
  const { cpuTuple, cpuIndexTuple, cpuOperator } = COSTS;
  const spc = COSTS.seq;
  const qualCost = quals * cpuOperator;
  const geo = btreeGeometry(N);
  const seq = spc * T + N * (cpuTuple + qualCost);

  const tuples = Math.max(1, Math.round(sel * N)); // clamp_row_est
  const indexLeafPages = geo.pages > 1 ? Math.ceil((tuples * geo.pages) / N) : 1;
  const indexPart =
    indexLeafPages * rpc +
    tuples * (cpuIndexTuple + qualCost) +
    Math.ceil(Math.log2(N)) * cpuOperator +
    (geo.height + 1) * 50 * cpuOperator;

  const mlPages = indexPagesFetched(tuples, T, geo.pages);
  const maxIO = mlPages * rpc;
  const corrPages = Math.ceil(sel * T);
  const minIO = corrPages > 0 ? rpc + Math.max(0, corrPages - 1) * spc : 0;
  const index = indexPart + maxIO + corr * corr * (minIO - maxIO) + tuples * cpuTuple;

  let bitmapPages = (2 * T * tuples) / (2 * T + tuples);
  bitmapPages = bitmapPages >= T ? T : Math.ceil(bitmapPages);
  const perPage = bitmapPages >= 2 ? rpc - (rpc - spc) * Math.sqrt(bitmapPages / T) : rpc;
  const bitmap = indexPart + 0.1 * cpuOperator * tuples + bitmapPages * perPage + tuples * (cpuTuple + qualCost);

  const bestIndexPath: PlanName = bitmapOn && bitmap < index ? 'Bitmap Heap Scan' : 'Index Scan';
  const bestIndexPathCost = bestIndexPath === 'Index Scan' ? index : bitmap;
  const best: PlanName = bestIndexPathCost < seq ? bestIndexPath : 'Seq Scan';
  const bestCost = Math.min(seq, bestIndexPathCost);
  return { seq, index, bitmap, indexPart, tuples, indexLeafPages, mlPages, bitmapPages, best, bestCost, bestIndexPath, bestIndexPathCost };
}

const costOf = (c: PlanCosts, p: PlanName) => (p === 'Seq Scan' ? c.seq : p === 'Index Scan' ? c.index : c.bitmap);
export type Crossover = { from: PlanName; to: PlanName; sel: number };

/** Selectivities where the cheapest plan changes, for a given correlation. */
export function crossovers(corr: number, rpc: number, quals: number, bitmapOn = true): Crossover[] {
  const lo = -5;
  const steps = 240;
  const out: Crossover[] = [];
  let prevS = Math.pow(10, lo);
  let prev = planCosts(prevS, corr, rpc, quals, bitmapOn).best;
  for (let i = 1; i <= steps; i++) {
    const s = Math.pow(10, lo + (-lo * i) / steps);
    const cur = planCosts(s, corr, rpc, quals, bitmapOn).best;
    if (cur !== prev) {
      let a = Math.log10(prevS);
      let b = Math.log10(s);
      for (let k = 0; k < 30; k++) {
        const m = (a + b) / 2;
        const c = planCosts(Math.pow(10, m), corr, rpc, quals, bitmapOn);
        if (costOf(c, prev) <= costOf(c, cur)) a = m;
        else b = m;
      }
      out.push({ from: prev, to: cur, sel: Math.pow(10, (a + b) / 2) });
    }
    prev = cur;
    prevS = s;
  }
  return out;
}

export type Placement = { pageOfRank: Int32Array; T: number; correlation: number };

/** Put N rows on heap pages: a fraction `inOrder` keeps key order, the rest are shuffled among themselves. */
export function placeRows(inOrder: number, seed = 7): Placement {
  const N = N_ROWS;
  const T = Math.ceil(N / ROWS_PER_PAGE);
  const rng = makeRng(seed);
  const slot = new Int32Array(N);
  const displaced = new Int32Array(N);
  let d = 0;
  for (let r = 0; r < N; r++) {
    slot[r] = r;
    if (rng() >= inOrder) displaced[d++] = r;
  }
  const values = displaced.slice(0, d);
  for (let i = d - 1; i > 0; i--) {
    const j = Math.min(i, Math.floor((rng() + rng() / 1e6) * (i + 1)));
    const t = values[i];
    values[i] = values[j];
    values[j] = t;
  }
  for (let i = 0; i < d; i++) slot[displaced[i]] = values[i];
  const pageOfRank = new Int32Array(N);
  for (let r = 0; r < N; r++) pageOfRank[r] = Math.floor(slot[r] / ROWS_PER_PAGE);
  return { pageOfRank, T, correlation: sampleCorrelation(slot) };
}

/** ANALYZE-style correlation: ranks of values vs ranks of physical position, over a systematic sample. */
export function sampleCorrelation(slot: Int32Array, sample = 30_000) {
  const N = slot.length;
  const n = Math.min(sample, N);
  const idx = new Array<number>(n);
  const pos = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    pos[i] = slot[Math.floor((i * N) / n)];
    idx[i] = i;
  }
  idx.sort((a, b) => pos[a] - pos[b]);
  // x = value rank i (the sample is already in value order); y = physical rank
  let xy = 0;
  for (let y = 0; y < n; y++) xy += idx[y] * y;
  const xs = ((n - 1) * n) / 2;
  const x2 = ((n - 1) * n * (2 * n - 1)) / 6;
  return (n * xy - xs * xs) / (n * x2 - xs * xs);
}

export const fmtCorr = (c: number) => (Math.abs(c) < 0.005 ? '0.00' : c.toFixed(2));

export const GRID_COLS = 50;
export const GRID_ROWS = 20;

/** Which heap pages a predicate matching `sel` of the rows touches, in the order an index scan visits them. */
export function touchPages(pl: Placement, sel: number) {
  const N = N_ROWS;
  const m = Math.max(1, Math.round(sel * N));
  const start = Math.floor((N - m) * 0.37);
  const mark = new Uint8Array(pl.T);
  let distinct = 0;
  let jumps = 0;
  let prev = -2;
  const pathCells: number[] = [];
  const cells = GRID_COLS * GRID_ROWS;
  const perCell = pl.T / cells;
  for (let r = start; r < start + m; r++) {
    const p = pl.pageOfRank[r];
    if (!mark[p]) {
      mark[p] = 1;
      distinct++;
    }
    if (p !== prev && p !== prev + 1) jumps++;
    prev = p;
    if (pathCells.length < 40) {
      const c = Math.floor(p / perCell);
      if (pathCells[pathCells.length - 1] !== c) pathCells.push(c);
    }
  }
  let runs = 0;
  const cellHits = new Float64Array(cells);
  for (let p = 0; p < pl.T; p++) {
    if (mark[p]) {
      if (p === 0 || !mark[p - 1]) runs++;
      cellHits[Math.floor(p / perCell)] += 1 / perCell;
    }
  }
  return { matches: m, distinct, jumps, runs, cellHits, pathCells, perCell };
}

export type OptionScore = {
  bet: Bet;
  plan: PlanName;
  cost: number;
  indexPath: PlanName | null;
  indexPathCost: number;
  heapPages: number;
  indexPages: number;
  pagesRead: number;
  randomReads: number;
  indexBytes: number;
  btreesPerInsert: number;
  rewritePages: number;
  correlation: number;
};

export function scoreOption(bet: Bet, sel: number, rpc: number, quals: number, pl: Placement, bitmapOn = true): OptionScore {
  const T = pl.T;
  const geo = btreeGeometry(N_ROWS);
  const c = planCosts(sel, pl.correlation, rpc, quals, bitmapOn);
  if (bet === 'leave') {
    return { bet, plan: 'Seq Scan', cost: c.seq, indexPath: null, indexPathCost: 0, heapPages: T, indexPages: 0, pagesRead: T, randomReads: 1, indexBytes: 0, btreesPerInsert: 1, rewritePages: 0, correlation: pl.correlation };
  }
  const touch = touchPages(pl, sel);
  const plan = c.best;
  const indexPages = plan === 'Seq Scan' ? 0 : geo.height + c.indexLeafPages;
  const heapPages = plan === 'Seq Scan' ? T : touch.distinct;
  const randomReads = plan === 'Seq Scan' ? 1 : (plan === 'Index Scan' ? touch.jumps : touch.runs) + geo.height + 1;
  const pkPages = geo.pages; // the existing primary key is rebuilt too
  return {
    bet,
    plan,
    cost: c.bestCost,
    indexPath: c.bestIndexPath,
    indexPathCost: c.bestIndexPathCost,
    heapPages,
    indexPages,
    pagesRead: heapPages + indexPages,
    randomReads,
    indexBytes: geo.bytes,
    btreesPerInsert: 2,
    rewritePages: bet === 'cluster' ? T + geo.pages + pkPages : geo.pages,
    correlation: pl.correlation,
  };
}

/**
 * The lab's scoring rule: an option wins only if it at least halves the pages read of every option that is cheaper
 * to own AND saves at least 1% of the table's pages per execution. Leaving it < creating the index < clustering.
 * Clustering is never scored best on a key whose values change during a row's life (a status column): the order
 * would decay with the workload itself, which the pages-read score cannot see.
 */
export function bestBet(leave: OptionScore, index: OptionScore, cluster: OptionScore, clusterable = true): Bet {
  const T = Math.ceil(N_ROWS / ROWS_PER_PAGE);
  const wins = (cand: number, incumbent: number) => cand <= incumbent / 2 && incumbent - cand >= T * 0.01;
  let best: Bet = 'leave';
  if (wins(index.pagesRead, leave.pagesRead)) best = 'index';
  if (clusterable && wins(cluster.pagesRead, Math.min(leave.pagesRead, index.pagesRead))) best = 'cluster';
  return best;
}

export type Preset = { value: string; label: string; v: number; corr: number; quals: number; stat: string; volatileKey?: boolean };

/** Selectivity slider position v ∈ [0, 500] ↔ 10^(−5 + v/100). */
export const selOf = (v: number) => Math.pow(10, -5 + v / 100);

export const PRESETS: Preset[] = [
  { value: 'day', label: 'created_at in the last day (append-only)', v: 244, corr: 0.99, quals: 2, stat: 'histogram_bounds: one day out of a year of orders' },
  { value: 'customer', label: 'customer_id = $1', v: 29, corr: 0.01, quals: 1, stat: 'n_distinct and the MCV list: (1 − 0.02) ÷ (50,000 − 100)' },
  { value: 'month', label: 'created_at in one month, after years of updates', v: 392, corr: 0.35, quals: 2, stat: 'histogram_bounds: one month out of a year' },
  { value: 'pending', label: "status = 'pending' (rows scattered)", v: 260, corr: 0, quals: 1, stat: "most_common_freqs for 'pending': 0.004", volatileKey: true },
  { value: 'shipped', label: "status = 'shipped' (rows scattered)", v: 479, corr: 0, quals: 1, stat: "most_common_freqs for 'shipped': 0.62", volatileKey: true },
  { value: 'archived', label: 'is_archived = false', v: 499, corr: 0, quals: 1, stat: 'most_common_freqs for false: 0.98', volatileKey: true },
];

/** Large counts for stat tiles: 4.01 billion, 40.1 million, 5,000. */
export function fmtCount(n: number) {
  if (n >= 1e9) return `${fmtNum(n / 1e9, 2)} billion`;
  if (n >= 1e7) return `${fmtNum(n / 1e6, 1)} million`;
  return fmtNum(n);
}

export function fmtPct(s: number) {
  const p = s * 100;
  if (p >= 10) return `${fmtNum(p, 0)}%`;
  if (p >= 1) return `${fmtNum(p, 1)}%`;
  if (p >= 0.1) return `${fmtNum(p, 2)}%`;
  if (p >= 0.01) return `${fmtNum(p, 3)}%`;
  return `${fmtNum(p, 4)}%`;
}

/* ================================================================ model: FK lookups */

export type Engine = 'postgres' | 'oracle' | 'mysql';
export type FkAction = 'noaction' | 'cascade' | 'setnull';
export const FK = { customers: 10_000, orders: 1_000_000, items: 4_000_000, ordersPerCustomer: 100, itemsPerOrder: 4, rowsPerPage: 100 };

export type FkLevel = {
  table: string;
  statement: string;
  executions: number;
  plan: string;
  rowsExaminedPerExec: number;
  pagesPerExec: number;
  matchesPerExec: number;
  rowsAffected: number;
  rowLocks: string;
  tableLocks: number;
  tableLockNote: string;
};

export function fkDelete(engine: Engine, parents: number, action: FkAction, idxOrdersIn: boolean, idxItemsIn: boolean) {
  const idxOrders = engine === 'mysql' ? true : idxOrdersIn;
  const idxItems = engine === 'mysql' ? true : idxItemsIn;
  const ordersPages = Math.ceil(FK.orders / FK.rowsPerPage);
  const itemsPages = Math.ceil(FK.items / FK.rowsPerPage);
  const hOrders = btreeGeometry(FK.orders).height + 1;
  const hItems = btreeGeometry(FK.items).height + 1;
  const levels: FkLevel[] = [];

  levels.push({
    table: 'customers',
    statement: 'DELETE FROM customers WHERE id = ANY($1)',
    executions: 1,
    plan: 'Index seek on customers primary key',
    rowsExaminedPerExec: parents,
    pagesPerExec: parents * (btreeGeometry(FK.customers).height + 2),
    matchesPerExec: parents,
    rowsAffected: parents,
    rowLocks: `${fmtNum(parents)} rows deleted`,
    tableLocks: 0,
    tableLockNote: '',
  });

  const ordMatches = action === 'noaction' ? 0 : FK.ordersPerCustomer;
  const pgOrdersSql =
    action === 'noaction'
      ? 'SELECT 1 FROM ONLY orders x WHERE $1 = customer_id FOR KEY SHARE OF x'
      : action === 'cascade'
        ? 'DELETE FROM ONLY orders WHERE $1 = customer_id'
        : 'UPDATE ONLY orders SET customer_id = NULL WHERE $1 = customer_id';
  const internal = engine === 'oracle' ? 'recursive referential-integrity work inside Oracle (no user SQL)' : 'performed inside InnoDB, row by row, through the foreign-key index (no SQL)';
  levels.push({
    table: 'orders',
    statement: engine === 'postgres' ? pgOrdersSql : internal,
    executions: parents,
    plan: idxOrders ? 'Index seek on orders(customer_id)' : 'Full scan of orders',
    rowsExaminedPerExec: idxOrders ? ordMatches : FK.orders,
    pagesPerExec: idxOrders ? hOrders + ordMatches : ordersPages,
    matchesPerExec: ordMatches,
    rowsAffected: parents * ordMatches,
    rowLocks:
      action === 'noaction'
        ? engine === 'postgres'
          ? 'FOR KEY SHARE on matches: none found'
          : 'no matches found'
        : action === 'cascade'
          ? `${fmtNum(parents * ordMatches)} rows deleted`
          : `${fmtNum(parents * ordMatches)} rows set to NULL`,
    tableLocks: engine === 'oracle' && !idxOrders ? parents : 0,
    tableLockNote: engine === 'oracle' && !idxOrders ? 'full table lock on orders, taken and released once per customer row' : '',
  });

  if (action === 'cascade') {
    const execs = parents * FK.ordersPerCustomer;
    levels.push({
      table: 'order_items',
      statement: engine === 'postgres' ? 'DELETE FROM ONLY order_items WHERE $1 = order_id' : internal,
      executions: execs,
      plan: idxItems ? 'Index seek on order_items(order_id)' : 'Full scan of order_items',
      rowsExaminedPerExec: idxItems ? FK.itemsPerOrder : FK.items,
      pagesPerExec: idxItems ? hItems + FK.itemsPerOrder : itemsPages,
      matchesPerExec: FK.itemsPerOrder,
      rowsAffected: execs * FK.itemsPerOrder,
      rowLocks: `${fmtNum(execs * FK.itemsPerOrder)} rows deleted`,
      tableLocks: engine === 'oracle' && !idxItems ? execs : 0,
      tableLockNote: engine === 'oracle' && !idxItems ? 'full table lock on order_items, once per orders row deleted' : '',
    });
  }

  const child = levels.slice(1);
  const lookups = child.reduce((s, l) => s + l.executions, 0);
  const rowsExamined = child.reduce((s, l) => s + l.executions * l.rowsExaminedPerExec, 0);
  const pagesRead = child.reduce((s, l) => s + l.executions * l.pagesPerExec, 0);
  const tableLocks = child.reduce((s, l) => s + l.tableLocks, 0);
  const rowsAffected = child.reduce((s, l) => s + l.rowsAffected, 0);
  return { levels, lookups, rowsExamined, pagesRead, tableLocks, rowsAffected, perParent: rowsExamined / parents, idxOrders, idxItems, ordersPages, itemsPages };
}

/* ================================================================ UI */

// Module-wide convention (index-selection-strategy): an index seek is --viz-1 and a read-and-check path is --viz-2.
const PLAN_COLOR: Record<PlanName, string> = { 'Seq Scan': 'var(--viz-2)', 'Index Scan': 'var(--viz-1)', 'Bitmap Heap Scan': 'var(--viz-3)' };
const CLUSTER_COLOR = 'var(--viz-4)';
/** Matching pages are a neutral ink wash: every hue in this figure already names a plan. */
const MATCH_FILL = 'var(--viz-ink-2)';
const BET_LABEL: Record<Bet, string> = { index: 'Create this index', leave: 'Leave it', cluster: 'Cluster the table' };
/** A surface-coloured outline behind chart labels, so curves and guide lines never cut through the text. */
const HALO = { stroke: 'var(--viz-surface)', strokeWidth: 3, strokeLinejoin: 'round' as const, paintOrder: 'stroke' as const };
const xoText = (xo: Crossover[]) => xo.map((x) => `${x.from} → ${x.to} at ${fmtPct(x.sel)}`).join(' · ');

export default function IndexDecisionCrossoverLab() {
  const [tab, setTab] = useState<'bet' | 'fk'>('bet');
  return (
    <VizPanel
      title={tab === 'bet' ? 'Build the index, leave it, or cluster the table?' : 'The reads you never wrote: foreign-key lookups'}
      subtitle={
        tab === 'bet'
          ? 'Set how many rows the predicate matches and how closely the heap follows the key order, then commit to a decision. Only then does the lab show the planner’s cost curves and score your bet.'
          : 'Delete parent rows and follow the lookups the database runs against the child tables on your behalf: one per affected row, each a full scan unless the child’s foreign-key column is indexed.'
      }
      controls={
        <Segmented
          label="Lab"
          value={tab}
          onChange={setTab}
          options={[
            { value: 'bet', label: 'The bet' },
            { value: 'fk', label: 'Foreign-key lookups' },
          ]}
        />
      }
    >
      {tab === 'bet' ? <BetLab /> : <FkLab />}
    </VizPanel>
  );
}

function BetLab() {
  const [preset, setPreset] = useState('month');
  const [v, setV] = useState(392);
  const [corrIn, setCorrIn] = useState(0.35);
  const [rpc, setRpc] = useState(4);
  const [bitmapOn, setBitmapOn] = useState(true);
  const [churn, setChurn] = useState(0);
  const [bet, setBet] = useState<Bet | null>(null);

  const p = PRESETS.find((x) => x.value === preset);
  const quals = p ? p.quals : 2;
  const volatileKey = !!p?.volatileKey;
  const sel = selOf(v);
  const corrD = useDeferredValue(corrIn);
  const churnD = useDeferredValue(churn);
  const selD = useDeferredValue(sel);

  const table = useMemo(() => placeRows(corrD, 7), [corrD]);
  const clustered = useMemo(() => placeRows(1 - churnD, 11), [churnD]);
  const touchNow = useMemo(() => touchPages(table, selD), [table, selD]);
  const touchCl = useMemo(() => touchPages(clustered, selD), [clustered, selD]);

  const scores = useMemo(() => {
    const leave = scoreOption('leave', selD, rpc, quals, table, bitmapOn);
    const index = scoreOption('index', selD, rpc, quals, table, bitmapOn);
    const cluster = scoreOption('cluster', selD, rpc, quals, clustered, bitmapOn);
    return { leave, index, cluster, best: bestBet(leave, index, cluster, !volatileKey) };
  }, [selD, rpc, quals, table, clustered, bitmapOn, volatileKey]);

  const xo = useMemo(() => crossovers(table.correlation, rpc, quals, bitmapOn), [table.correlation, rpc, quals, bitmapOn]);
  const xoCl = useMemo(() => crossovers(clustered.correlation, rpc, quals, bitmapOn), [clustered.correlation, rpc, quals, bitmapOn]);

  const reset = () => setBet(null);
  const choosePreset = (val: string) => {
    const q = PRESETS.find((x) => x.value === val);
    setPreset(val);
    if (q) {
      setV(q.v);
      setCorrIn(q.corr);
    }
    reset();
  };

  const revealed = bet !== null;
  const mine = bet ? scores[bet] : null;
  const geo = btreeGeometry(N_ROWS);
  const shownTouch = bet === 'cluster' ? touchCl : touchNow;

  return (
    <>
      <div className="viz-controls">
        <Choice
          label="Predicate on orders"
          value={preset}
          onChange={choosePreset}
          options={[...PRESETS.map((x) => ({ value: x.value, label: x.label })), { value: 'custom', label: 'Custom range (use the sliders)' }]}
        />
        <Slider
          label="Rows matched"
          min={0}
          max={500}
          value={v}
          onChange={(n) => {
            setV(n);
            setPreset('custom');
            reset();
          }}
          format={(n) => `${fmtPct(selOf(n))} · ${fmtNum(Math.max(1, Math.round(selOf(n) * N_ROWS)))} rows`}
        />
        <Slider
          label="Rows still in key order"
          min={0}
          max={1}
          step={0.01}
          value={corrIn}
          onChange={(n) => {
            setCorrIn(n);
            setPreset('custom');
            reset();
          }}
          format={(n) => `${Math.round(n * 100)}%`}
        />
      </div>

      <div className="viz-controls" role="group" aria-label="Your bet">
        {(['index', 'leave', 'cluster'] as Bet[]).map((b) => (
          <Button key={b} primary={bet === b} onClick={() => setBet(b)} title={b === 'cluster' ? 'CLUSTER needs the index, then rewrites the heap in key order' : undefined}>
            {BET_LABEL[b]}
          </Button>
        ))}
        {revealed ? <Button onClick={reset}>New bet</Button> : null}
      </div>

      {revealed ? (
        <div className="viz-controls">
          <Slider label="random_page_cost" min={1} max={6} step={0.1} value={rpc} onChange={setRpc} format={(n) => n.toFixed(1)} />
          <Check label="enable_bitmapscan" checked={bitmapOn} onChange={setBitmapOn} />
          <Slider label="Rows changed since CLUSTER" min={0} max={0.5} step={0.01} value={churn} onChange={setChurn} format={(n) => `${Math.round(n * 100)}%`} />
        </div>
      ) : null}

      <HeapGrid touch={shownTouch} T={table.T} plan={mine ? mine.plan : null} clusteredView={bet === 'cluster'} />

      {revealed ? (
        <CrossoverChart sel={selD} corr={table.correlation} corrCl={clustered.correlation} rpc={rpc} quals={quals} bitmapOn={bitmapOn} xo={xo} />
      ) : (
        <p className="viz-note" style={{ textAlign: 'center', padding: '1.2rem 0.5rem', border: '1px dashed var(--viz-border)', borderRadius: 8 }}>
          The planner’s cost curves and the scoreboard stay hidden until you commit: <strong>create this index</strong>, <strong>leave it</strong> or <strong>cluster the table</strong>.
        </p>
      )}

      <Legend
        items={[
          { label: 'Heap pages holding a match (darker = more of the cell)', color: MATCH_FILL },
          { label: 'Seq Scan', color: PLAN_COLOR['Seq Scan'], shape: 'line' },
          { label: 'Index Scan', color: PLAN_COLOR['Index Scan'], shape: 'line' },
          { label: 'Bitmap Heap Scan', color: PLAN_COLOR['Bitmap Heap Scan'], shape: 'line' },
          { label: 'Index Scan after CLUSTER', color: CLUSTER_COLOR, shape: 'line' },
        ]}
      />

      <Stats
        items={
          revealed && mine
            ? [
                { label: 'Your bet', value: BET_LABEL[mine.bet] },
                { label: 'Plan it gets', value: mine.plan, hint: 'The cheapest path available to that option, by PostgreSQL 17’s cost formulas' },
                { label: 'Pages read (cold cache)', value: fmtNum(mine.pagesRead), hint: 'Distinct heap pages holding a match plus index pages, from the simulated table' },
                { label: 'Index bytes added', value: mine.indexBytes ? fmtBytes(mine.indexBytes) : '0', hint: '8-byte key, 20 bytes per leaf item, fillfactor 90, no deduplication' },
                { label: 'B-trees written per INSERT', value: `${mine.btreesPerInsert}`, hint: 'The table already has a primary key' },
                { label: 'Best bet here', value: BET_LABEL[scores.best], hint: 'Wins only by halving the pages read of every cheaper-to-own option and saving at least 1% of the table per execution; clustering is not scored best on a status column, whose values change' },
              ]
            : [
                { label: 'Rows matched', value: fmtNum(touchNow.matches) },
                { label: 'pg_stats.correlation', value: fmtCorr(table.correlation), hint: 'Rank correlation of value order vs physical order over a 30,000-row sample, as ANALYZE computes it' },
                { label: 'Heap pages holding a match', value: `${fmtNum(touchNow.distinct)} of ${fmtNum(table.T)}` },
                { label: 'The index, if built', value: `${fmtBytes(geo.bytes)}, height ${geo.height}` },
              ]
        }
      />

      <Note>
        {!revealed ? (
          <>
            <strong>
              {p ? p.label : 'Custom range'}: {fmtNum(touchNow.matches)} of {fmtNum(N_ROWS)} rows ({fmtPct(selD)}).
            </strong>{' '}
            {p ? `The planner gets this from ${p.stat}. ` : ''}Those rows sit on {fmtNum(touchNow.distinct)} of the table’s {fmtNum(table.T)} heap pages; pg_stats.correlation is {fmtCorr(table.correlation)}. Place your bet.
          </>
        ) : (
          <BetNarration bet={bet!} scores={scores} xo={xo} rpc={rpc} churn={churn} bitmapOn={bitmapOn} volatileKey={volatileKey} />
        )}
      </Note>

      {revealed ? (
        <details className="viz-data">
          <summary>Show the numbers</summary>
          <table className="viz-table">
            <thead>
              <tr>
                <th>Option</th>
                <th>Plan</th>
                <th>Est. cost</th>
                <th>Heap pages</th>
                <th>Index pages</th>
                <th>Non-sequential reads</th>
                <th>Index bytes</th>
                <th>B-trees / INSERT</th>
                <th>Pages written up front</th>
              </tr>
            </thead>
            <tbody>
              {(['leave', 'index', 'cluster'] as Bet[]).map((b) => {
                const s = scores[b];
                return (
                  <tr key={b}>
                    <td>
                      {BET_LABEL[b]}
                      {b === bet ? ' (your bet)' : ''}
                      {b === scores.best ? ' ★' : ''}
                    </td>
                    <td>{s.plan}</td>
                    <td>{fmtNum(s.cost)}</td>
                    <td>{fmtNum(s.heapPages)}</td>
                    <td>{fmtNum(s.indexPages)}</td>
                    <td>{fmtNum(s.randomReads)}</td>
                    <td>{s.indexBytes ? fmtBytes(s.indexBytes) : '0'}</td>
                    <td>{s.btreesPerInsert}</td>
                    <td>{fmtNum(s.rewritePages)}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
          <p className="viz-note">
            Crossovers with the index at correlation {fmtCorr(table.correlation)}: {xo.length ? xoText(xo) : 'one plan wins everywhere'}. After CLUSTER (correlation {fmtCorr(clustered.correlation)}): {xoCl.length ? xoText(xoCl) : 'one plan wins everywhere'}. ★ marks the best bet.
          </p>
        </details>
      ) : null}
    </>
  );
}

function BetNarration({ bet, scores, xo, rpc, churn, bitmapOn, volatileKey }: { bet: Bet; scores: { leave: OptionScore; index: OptionScore; cluster: OptionScore; best: Bet }; xo: Crossover[]; rpc: number; churn: number; bitmapOn: boolean; volatileKey: boolean }) {
  const { leave, index, cluster, best } = scores;
  const times = (a: number, b: number) => {
    const r = Math.max(a, b) / Math.max(1, Math.min(a, b));
    return `${fmtNum(r, r < 10 ? 1 : 0)}×`;
  };
  const verdict = bet === best ? `Good bet: ${BET_LABEL[bet].toLowerCase()}.` : `Best bet here: ${BET_LABEL[best].toLowerCase()}.`;
  const readsVs =
    Math.abs(index.pagesRead - leave.pagesRead) < leave.pagesRead * 0.1
      ? 'about the same'
      : `${times(leave.pagesRead, index.pagesRead)} ${index.pagesRead <= leave.pagesRead ? 'fewer' : 'more'}`;
  const idx =
    index.plan === 'Seq Scan'
      ? `With the index, the planner still picks Seq Scan: its best index path (${index.indexPath}) is estimated at ${fmtNum(index.indexPathCost)} against ${fmtNum(leave.cost)}. You would pay ${fmtBytes(index.indexBytes)} and a second B-tree write on every INSERT for nothing.`
      : `With the index, the planner picks ${index.plan} and reads ${fmtNum(index.pagesRead)} pages instead of ${fmtNum(leave.pagesRead)} (${readsVs}), for ${fmtBytes(index.indexBytes)} of index.`;
  const cl =
    cluster.plan === 'Seq Scan'
      ? ' Clustering does not change that: the rows are too many for any index path.'
      : ` After CLUSTER${churn > 0 ? ` and ${Math.round(churn * 100)}% of rows changed` : ''} it is ${cluster.plan} over ${fmtNum(cluster.heapPages)} heap pages — but CLUSTER first rewrites the table and its indexes under an ACCESS EXCLUSIVE lock, and the order starts decaying with the next update.`;
  const volatile =
    volatileKey && cluster.plan !== 'Seq Scan'
      ? ` A cluster order on a status column decays with every status change, so the lab does not score it best here${index.plan !== 'Seq Scan' ? '; and since the index is read only for this rare value, build it as a partial index' : ''}.`
      : '';
  const seqWin = xo.find((x) => x.to === 'Seq Scan');
  const tail = seqWin ? ` At random_page_cost ${rpc.toFixed(1)}${bitmapOn ? '' : ' with bitmap scans off'}, Seq Scan takes over above ${fmtPct(seqWin.sel)}.` : '';
  return (
    <>
      <strong>{verdict}</strong> {idx}
      {cl}
      {volatile}
      {tail}
    </>
  );
}

function HeapGrid({ touch, T, plan, clusteredView }: { touch: ReturnType<typeof touchPages>; T: number; plan: PlanName | null; clusteredView: boolean }) {
  const cell = 12;
  const gap = 1.4;
  const left = 8;
  const top = 26;
  const W = left * 2 + GRID_COLS * (cell + gap);
  const H = top + GRID_ROWS * (cell + gap) + 24;
  const x0 = (i: number) => left + (i % GRID_COLS) * (cell + gap);
  const y0 = (i: number) => top + Math.floor(i / GRID_COLS) * (cell + gap);
  const planColor = plan ? (clusteredView && plan === 'Index Scan' ? CLUSTER_COLOR : PLAN_COLOR[plan]) : null;
  const cells = [];
  for (let i = 0; i < GRID_COLS * GRID_ROWS; i++) {
    const share = touch.cellHits[i];
    cells.push(
      <g key={i}>
        <rect x={x0(i)} y={y0(i)} width={cell} height={cell} rx={2} fill="var(--viz-neutral)" />
        {share > 0 ? <rect x={x0(i)} y={y0(i)} width={cell} height={cell} rx={2} fill={MATCH_FILL} fillOpacity={0.12 + 0.43 * Math.min(1, share)} /> : null}
        {plan && plan !== 'Seq Scan' && share > 0 && planColor ? <rect x={x0(i) + 0.7} y={y0(i) + 0.7} width={cell - 1.4} height={cell - 1.4} rx={2} fill="none" stroke={planColor} strokeWidth={1.4} /> : null}
      </g>,
    );
  }
  const cx = (c: number) => x0(c) + cell / 2;
  const cy = (c: number) => y0(c) + cell / 2;
  const jumps: [number, number][] = [];
  for (let k = 1; k < touch.pathCells.length; k++) {
    const a = touch.pathCells[k - 1];
    const bb = touch.pathCells[k];
    if (bb !== a + 1) jumps.push([a, bb]);
  }
  const label = !plan
    ? `orders heap: ${fmtNum(T)} pages, ${fmtNum(touch.perCell)} pages per cell`
    : plan === 'Seq Scan'
      ? `Seq Scan: reads all ${fmtNum(T)} pages in block order and tests every row`
      : plan === 'Index Scan'
        ? `Index Scan${clusteredView ? ' after CLUSTER' : ''}: heap fetches in index order (first ones traced): ${fmtNum(touch.distinct)} pages, ${fmtNum(touch.jumps)} non-sequential`
        : `Bitmap Heap Scan${clusteredView ? ' after CLUSTER' : ''}: TIDs sorted, ${fmtNum(touch.distinct)} pages read in block order, ${fmtNum(touch.runs)} runs`;
  return (
    <svg viewBox={`0 0 ${W} ${H}`} width={W} height={H} role="img" aria-label={`Heap page grid: ${fmtNum(touch.distinct)} of ${fmtNum(T)} pages hold a matching row${plan ? `; plan ${plan}` : ''}`}>
      <text x={left} y={16} fontSize={12} fill="var(--viz-ink)">
        {label}
      </text>
      {cells}
      {plan === 'Seq Scan' && planColor ? <rect x={left - 4} y={top - 4} width={GRID_COLS * (cell + gap) + 6} height={GRID_ROWS * (cell + gap) + 6} rx={4} fill="none" stroke={planColor} strokeWidth={2.5} /> : null}
      {plan === 'Index Scan' && planColor
        ? jumps.map(([a, bb], k) => <line key={`j${k}`} x1={cx(a)} y1={cy(a)} x2={cx(bb)} y2={cy(bb)} stroke={planColor} strokeWidth={1.6} strokeOpacity={0.9} />)
        : null}
      {plan === 'Index Scan' && planColor && touch.pathCells.length > 0 ? <circle cx={cx(touch.pathCells[0])} cy={cy(touch.pathCells[0])} r={4} fill={planColor} /> : null}
      <text x={left} y={H - 7} fontSize={11} fill="var(--viz-ink-2)">
        page 0 at top left, page {fmtNum(T - 1)} at bottom right · {fmtNum(touch.distinct)} pages hold at least one of the {fmtNum(touch.matches)} matching rows
      </text>
    </svg>
  );
}

function CrossoverChart({ sel, corr, corrCl, rpc, quals, bitmapOn, xo }: { sel: number; corr: number; corrCl: number; rpc: number; quals: number; bitmapOn: boolean; xo: Crossover[] }) {
  const W = 680;
  const H = 290;
  const L = 52;
  const R = 16;
  const Tp = 26;
  const B = 62;
  const xs = (s: number) => L + ((Math.log10(s) + 5) / 5) * (W - L - R);
  const series = useMemo(() => {
    const pts: { s: number; c: PlanCosts; cl: PlanCosts }[] = [];
    for (let i = 0; i <= 120; i++) {
      const s = Math.pow(10, -5 + (5 * i) / 120);
      pts.push({ s, c: planCosts(s, corr, rpc, quals, bitmapOn), cl: planCosts(s, corrCl, rpc, quals, bitmapOn) });
    }
    return pts;
  }, [corr, corrCl, rpc, quals, bitmapOn]);
  let yMin = Infinity;
  let yMax = 0;
  for (const pt of series) {
    yMin = Math.min(yMin, pt.c.index, pt.c.bitmap, pt.cl.index);
    yMax = Math.max(yMax, pt.c.seq, pt.c.index, pt.c.bitmap, pt.cl.index);
  }
  const lo = Math.floor(Math.log10(Math.max(1, yMin)));
  const hi = Math.ceil(Math.log10(yMax));
  const ys = (c: number) => Tp + (1 - (Math.log10(Math.max(1, c)) - lo) / (hi - lo)) * (H - Tp - B);
  const line = (f: (pt: (typeof series)[number]) => number) => series.map((pt) => `${xs(pt.s).toFixed(1)},${ys(f(pt)).toFixed(1)}`).join(' ');
  const here = planCosts(sel, corr, rpc, quals, bitmapOn);
  const bandY = H - B + 20;
  const markX = xs(sel);
  return (
    <svg viewBox={`0 0 ${W} ${H}`} width={W} height={H} role="img" aria-label={`Estimated cost versus rows matched; at ${fmtPct(sel)} the planner picks ${here.best}`} style={{ marginTop: 8 }}>
      {Array.from({ length: hi - lo + 1 }, (_, i) => lo + i).map((e) => (
        <g key={e}>
          <line x1={L} x2={W - R} y1={ys(10 ** e)} y2={ys(10 ** e)} stroke="var(--viz-grid)" />
          <text x={L - 6} y={ys(10 ** e) + 4} fontSize={10} textAnchor="end" fill="var(--viz-ink-2)">
            {e >= 3 ? `${fmtNum(10 ** (e - 3))}k` : fmtNum(10 ** e)}
          </text>
        </g>
      ))}
      {[-5, -4, -3, -2, -1, 0].map((e) => (
        <g key={e}>
          <line x1={xs(10 ** e)} x2={xs(10 ** e)} y1={Tp} y2={H - B} stroke="var(--viz-grid)" />
          <text x={xs(10 ** e)} y={H - B + 13} fontSize={10} textAnchor="middle" fill="var(--viz-ink-2)">
            {fmtPct(10 ** e)}
          </text>
        </g>
      ))}
      <text x={4} y={Tp - 12} fontSize={10} fill="var(--viz-ink-2)">
        estimated cost (log)
      </text>
      <polyline points={line((pt) => pt.c.seq)} fill="none" stroke={PLAN_COLOR['Seq Scan']} strokeWidth={2.4} />
      <polyline points={line((pt) => pt.c.bitmap)} fill="none" stroke={PLAN_COLOR['Bitmap Heap Scan']} strokeWidth={2.4} strokeOpacity={bitmapOn ? 1 : 0.3} />
      <polyline points={line((pt) => pt.c.index)} fill="none" stroke={PLAN_COLOR['Index Scan']} strokeWidth={2.4} />
      <polyline points={line((pt) => pt.cl.index)} fill="none" stroke={CLUSTER_COLOR} strokeWidth={2.2} strokeDasharray="6 4" />

      {xo.map((x, i) => (
        <line key={`xo${i}`} x1={xs(x.sel)} x2={xs(x.sel)} y1={Tp} y2={bandY + 8} stroke="var(--viz-ink-2)" strokeDasharray="3 3" />
      ))}
      {series.slice(1).map((pt, i) => (
        <rect key={i} x={xs(series[i].s)} y={bandY} width={Math.max(0.5, xs(pt.s) - xs(series[i].s) + 0.4)} height={8} fill={PLAN_COLOR[pt.c.best]} />
      ))}
      <text x={L} y={H - 8} fontSize={10.5} fill="var(--viz-ink-2)">
        cheapest plan with the index: {xo.length ? xoText(xo) : `${here.best} everywhere`}
      </text>
      <line x1={markX} x2={markX} y1={Tp - 4} y2={bandY + 8} stroke="var(--viz-ink)" strokeWidth={1.5} />
      <circle cx={markX} cy={ys(here.bestCost)} r={4.5} fill="var(--viz-surface)" stroke="var(--viz-ink)" strokeWidth={1.8} />
      <text x={xs(series[4].s)} y={ys(series[4].c.seq) - 7} fontSize={11} fill="var(--viz-ink)" {...HALO}>
        Seq Scan
      </text>
      <text x={xs(series[40].s)} y={Math.min(ys(series[40].c.index), ys(series[40].c.bitmap)) - 8} fontSize={11} textAnchor="middle" fill="var(--viz-ink)" {...HALO}>
        Index Scan (correlation {fmtCorr(corr)})
      </text>
      <text x={xs(series[74].s)} y={Math.max(ys(series[74].c.bitmap), ys(series[74].c.index)) + 17} fontSize={11} fill="var(--viz-ink)" {...HALO}>
        Bitmap Heap Scan{bitmapOn ? '' : ' (disabled)'}
      </text>
      <text x={xs(series[34].s)} y={ys(series[34].cl.index) + 17} fontSize={11} fill="var(--viz-ink)" {...HALO}>
        Index Scan after CLUSTER (correlation {fmtCorr(corrCl)})
      </text>
      <text x={markX > W - 220 ? markX - 6 : markX + 6} y={Tp - 12} fontSize={11} textAnchor={markX > W - 220 ? 'end' : 'start'} fill="var(--viz-ink)">
        your predicate: {fmtPct(sel)} → {here.best}
      </text>
    </svg>
  );
}

function FkLab() {
  const [engine, setEngine] = useState<Engine>('postgres');
  const [parents, setParents] = useState(10);
  const [action, setAction] = useState<FkAction>('cascade');
  const [idxOrders, setIdxOrders] = useState(false);
  const [idxItems, setIdxItems] = useState(false);
  const [progress, setProgress] = useState(1);
  const [running, setRunning] = useState(false);
  const r = useMemo(() => fkDelete(engine, parents, action, idxOrders, idxItems), [engine, parents, action, idxOrders, idxItems]);

  const progRef = useRef(1);
  useTicker((dt) => {
    const n = Math.min(1, progRef.current + dt / 2500);
    progRef.current = n;
    setProgress(n);
    if (n >= 1) setRunning(false);
  }, running);

  const done = Math.max(0, Math.min(parents, Math.ceil(progress * parents)));
  const frac = parents ? done / parents : 1;
  const W = 680;
  const laneH = 84;
  const H = 20 + r.levels.length * laneH;
  const stripX = 196;
  const cells = 40;
  const cellW = (W - stripX - 16) / cells - 2;

  return (
    <>
      <div className="viz-controls">
        <Segmented
          label="Engine"
          value={engine}
          onChange={setEngine}
          options={[
            { value: 'postgres', label: 'PostgreSQL' },
            { value: 'oracle', label: 'Oracle' },
            { value: 'mysql', label: 'MySQL InnoDB' },
          ]}
        />
        <Slider label="Customers deleted" min={1} max={100} value={parents} onChange={setParents} />
        <Segmented
          label="orders.customer_id ON DELETE"
          value={action}
          onChange={setAction}
          options={[
            { value: 'noaction', label: 'NO ACTION' },
            { value: 'cascade', label: 'CASCADE' },
            { value: 'setnull', label: 'SET NULL' },
          ]}
        />
        <Check label={`Index orders(customer_id)${engine === 'mysql' ? ' (InnoDB requires it)' : ''}`} checked={r.idxOrders} onChange={(b) => engine !== 'mysql' && setIdxOrders(b)} />
        <Check label={`Index order_items(order_id)${engine === 'mysql' ? ' (InnoDB requires it)' : ''}`} checked={r.idxItems} onChange={(b) => engine !== 'mysql' && setIdxItems(b)} />
        <Button
          onClick={() => {
            progRef.current = 0;
            setProgress(0);
            setRunning(true);
          }}
        >
          Replay the delete
        </Button>
      </div>

      <svg viewBox={`0 0 ${W} ${H}`} width={W} height={H} role="img" aria-label={`Foreign-key lookups: ${fmtNum(r.rowsExamined)} child rows examined for ${parents} deleted customers`}>
        {r.levels.map((l, i) => {
          const y = 14 + i * laneH;
          const pagesTotal = i === 0 ? cells : i === 1 ? r.ordersPages : r.itemsPages;
          const seq = l.plan.startsWith('Full');
          const litCells = i === 0 ? 0 : seq ? cells : Math.min(cells, Math.max(1, Math.ceil((l.pagesPerExec / pagesTotal) * cells)));
          const color = seq ? PLAN_COLOR['Seq Scan'] : PLAN_COLOR['Index Scan'];
          const execsDone = i === 0 ? (progress > 0 ? 1 : 0) : Math.round(l.executions * frac);
          return (
            <g key={l.table}>
              {i > 0 ? <line x1={60} x2={60} y1={y - laneH + 62} y2={y - 2} stroke="var(--viz-ink-muted)" strokeDasharray="3 3" /> : null}
              <rect x={6} y={y} width={180} height={62} rx={6} fill="var(--viz-surface)" stroke={l.tableLocks ? 'var(--viz-critical)' : 'var(--viz-border)'} strokeWidth={l.tableLocks ? 2.5 : 1} />
              <text x={16} y={y + 18} fontSize={13} fill="var(--viz-ink)">
                {l.table}
              </text>
              <text x={16} y={y + 34} fontSize={10} fill="var(--viz-ink-2)">
                {i === 0 ? `${fmtNum(FK.customers)} rows` : `${fmtNum(i === 1 ? FK.orders : FK.items)} rows, ${fmtNum(pagesTotal)} pages`}
              </text>
              {l.tableLocks ? (
                <text x={16} y={y + 52} fontSize={10} fill="var(--viz-ink)">
                  ⚠ full table lock ×{fmtNum(Math.round(l.tableLocks * frac))}
                </text>
              ) : null}
              {i > 0 ? (
                <>
                  {Array.from({ length: cells }, (_, c) => (
                    <rect key={c} x={stripX + c * (cellW + 2)} y={y + 4} width={cellW} height={16} rx={2} fill={c < litCells && execsDone > 0 ? color : 'var(--viz-neutral)'} fillOpacity={c < litCells && execsDone > 0 ? 0.85 : 1} />
                  ))}
                  <text x={stripX} y={y + 36} fontSize={11} fill="var(--viz-ink)">
                    {seq ? `Full scan: all ${fmtNum(pagesTotal)} pages, every lookup` : `Index seek: ${fmtNum(l.pagesPerExec)} pages per lookup`} · lookup {fmtNum(execsDone)} of {fmtNum(l.executions)}
                  </text>
                  <text x={stripX} y={y + 51} fontSize={10.5} fill="var(--viz-ink-2)">
                    rows examined so far: {fmtNum(execsDone * l.rowsExaminedPerExec)} · {l.rowLocks}
                  </text>
                </>
              ) : (
                <text x={stripX} y={y + 24} fontSize={11} fill="var(--viz-ink)">
                  {fmtNum(parents)} parent rows deleted through the primary key
                </text>
              )}
            </g>
          );
        })}
      </svg>

      <ol style={{ margin: '0.4rem 0 0', paddingLeft: '1.2rem', fontSize: '0.8rem', display: 'grid', gap: 4 }}>
        {r.levels.map((l, i) => (
          <li key={l.table} style={{ color: 'var(--viz-ink)' }}>
            {i === 0 ? 'Statement: ' : `Runs ×${fmtNum(l.executions)}: `}
            <code>{l.statement}</code>
            {l.tableLockNote ? ` — ${l.tableLockNote}` : ''}
          </li>
        ))}
      </ol>

      <Legend
        items={[
          { label: 'Pages read by one lookup: full scan', color: PLAN_COLOR['Seq Scan'] },
          { label: 'Pages read by one lookup: index seek', color: PLAN_COLOR['Index Scan'] },
          { label: 'Full table lock held (Oracle)', color: 'var(--viz-critical)', shape: 'line' },
        ]}
      />

      <Stats
        items={[
          { label: 'Child lookups issued', value: fmtNum(r.lookups), hint: 'One per affected parent row, at every level of the cascade' },
          { label: 'Child rows examined', value: fmtCount(r.rowsExamined) },
          { label: 'Rows examined per customer', value: fmtCount(r.perParent), hint: 'Collapses to the matching rows once the child FK columns are indexed' },
          { label: 'Heap + index pages read', value: fmtCount(r.pagesRead), hint: 'Cold cache; a full scan reads every page of the child table' },
          { label: 'Child table locks', value: engine === 'oracle' ? fmtNum(r.tableLocks) : '0', hint: 'Oracle only: a full table lock on an unindexed child, once per parent key modified' },
        ]}
      />

      <Note>
        <FkNarration engine={engine} parents={parents} action={action} r={r} />
      </Note>

      <details className="viz-data">
        <summary>Show the numbers</summary>
        <table className="viz-table">
          <thead>
            <tr>
              <th>Table</th>
              <th>Executions</th>
              <th>Plan</th>
              <th>Rows examined / execution</th>
              <th>Pages / execution</th>
              <th>Rows affected</th>
              <th>Table locks</th>
            </tr>
          </thead>
          <tbody>
            {r.levels.map((l) => (
              <tr key={l.table}>
                <td>{l.table}</td>
                <td>{fmtNum(l.executions)}</td>
                <td>{l.plan}</td>
                <td>{fmtNum(l.rowsExaminedPerExec)}</td>
                <td>{fmtNum(l.pagesPerExec)}</td>
                <td>{fmtNum(l.rowsAffected)}</td>
                <td>{fmtNum(l.tableLocks)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </details>
    </>
  );
}

function FkNarration({ engine, parents, action, r }: { engine: Engine; parents: number; action: FkAction; r: ReturnType<typeof fkDelete> }) {
  const unindexed = !r.idxOrders || (action === 'cascade' && !r.idxItems);
  const head = `Deleting ${fmtNum(parents)} customer${parents === 1 ? '' : 's'} issues ${fmtNum(r.lookups)} child lookup${r.lookups === 1 ? '' : 's'} and examines ${fmtNum(r.rowsExamined)} child rows.`;
  let body = '';
  if (engine === 'mysql') body = 'InnoDB would not let the foreign key exist without an index on the referencing columns — it created one when the constraint was declared — so every lookup is a seek.';
  else if (!unindexed) body = `Every lookup is an index seek: ${fmtNum(r.perParent)} rows examined per customer, exactly the rows the delete has to touch anyway.`;
  else if (action === 'cascade' && r.idxOrders && !r.idxItems) body = `orders is indexed, but each of the ${fmtNum(r.levels[2].executions)} deleted orders fires its own lookup on order_items, and each of those is a full scan of ${fmtNum(FK.items)} rows. Index the grandchild too.`;
  else if (action === 'cascade' && !r.idxOrders && r.idxItems) body = `order_items is indexed, but finding the orders to delete is still a full scan of ${fmtNum(FK.orders)} rows per customer.`;
  else if (action === 'noaction') body = `These customers have no orders, so each check finds nothing — and proving “nothing” without an index means reading all ${fmtNum(FK.orders)} rows of orders, once per customer. A customer with orders would stop at the first match and fail the delete.`;
  else body = `No child index: each customer costs a full scan of orders${action === 'cascade' ? ', and each deleted order a full scan of order_items' : ''} — ${fmtNum(r.perParent)} rows examined per customer.`;
  const lock = engine === 'oracle' && r.tableLocks ? ` Oracle also takes a full table lock on the unindexed child ${fmtNum(r.tableLocks)} times: other sessions can still query it but cannot modify it while each lock is held.` : '';
  return (
    <>
      <strong>{head}</strong> {body}
      {lock}
    </>
  );
}
