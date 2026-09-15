import { useMemo, useRef, useState } from 'react';
import {
  VizPanel,
  Segmented,
  Choice,
  Check,
  Slider,
  Button,
  Legend,
  Stats,
  Note,
  TooltipHost,
  useTip,
  useTicker,
  makeRng,
  fmtBytes,
  fmtNum,
} from './Viz';

/**
 * Covering indexes and index-only scans, PostgreSQL and InnoDB.
 *
 * What is modelled from source:
 * - PostgreSQL (nodeIndexonlyscan.c IndexOnlyNext): for every index TID, check the heap page's
 *   visibility-map bit. Set: the row comes from the index tuple. Clear: fetch the heap tuple to test
 *   visibility (EXPLAIN's "Heap Fetches" counts these per TID), skip it if no version is visible.
 *   indxpath.c check_index_only: an index-only scan is possible only if every column in the target list
 *   and the restriction clauses is returnable from the index. SELECT * therefore gets a plain Index Scan.
 * - The VM bit is cleared by any insert, update or delete on the page (heapam.c) and set by VACUUM only
 *   if every tuple's xmin precedes OldestXmin (pruneheap.c), so a long-running snapshot blocks it.
 *   pg_class.relallvisible — what the planner divides by the current page count (tableam.c, costsize.c) —
 *   changes only at VACUUM and when the index definition changes (CREATE INDEX) here.
 * - An INCLUDE column is HOT-blocking (relcache.c): updating it adds a new index entry and leaves a dead one.
 * - InnoDB (row0sel.cc, lock0lock.cc): a consistent read of a secondary record needs no clustered lookup
 *   only if the leaf page's PAGE_MAX_TRX_ID < read view up_limit_id (the oldest active read-write trx id
 *   when the view opened). Otherwise every record on that page goes to the clustered index. Delete-marked
 *   records on a page that passes the check are skipped without a lookup. Secondary records carry the PK.
 *
 * Model assumptions (labelled in the UI):
 * - Lab-scale table: 40 heap pages of 10 rows. Updates keep the new version on the same page (the page
 *   has room), so an update clears exactly one page's VM bit; it is HOT unless `status` is in the index.
 * - InnoDB: customer 42's secondary records share one leaf page with customers 40, 41, 43 and 44.
 *   The clustered index is modelled as 3 levels, so each lookup reads 3 pages.
 * - Size model (PostgreSQL B-tree, 8 KB pages, built by CREATE INDEX): tuple = MAXALIGN(8-byte header +
 *   aligned data) + 4-byte line pointer; leaves filled to fillfactor 90, internal pages to 70; pivots keep
 *   key columns only; deduplication (posting lists capped at 812 bytes at build) only without INCLUDE.
 *   Column widths are averages: status 8 bytes, note 121 bytes (short varlena, unaligned), NOT NULL.
 */

export type Engine = 'pg' | 'innodb';
export type Role = 'off' | 'key' | 'include';
export type Col = 'id' | 'customer_id' | 'created_at' | 'status' | 'amount_cents' | 'note';
export type PayloadCol = Exclude<Col, 'customer_id'>;
export type IndexDef = Record<PayloadCol, Role>;
export type QueryId = 'q1' | 'q2' | 'q3' | 'q4' | 'q5';
export type Status = 'open' | 'shipped';

type ColMeta = { name: Col; type: string; width: number; align: number; short?: boolean };
export const COLS: ColMeta[] = [
  { name: 'id', type: 'bigint', width: 8, align: 8 },
  { name: 'customer_id', type: 'integer', width: 4, align: 4 },
  { name: 'created_at', type: 'timestamptz', width: 8, align: 8 },
  { name: 'status', type: 'text', width: 8, align: 1, short: true },
  { name: 'amount_cents', type: 'bigint', width: 8, align: 8 },
  { name: 'note', type: 'text', width: 121, align: 1, short: true },
];
const META = Object.fromEntries(COLS.map((c) => [c.name, c])) as Record<Col, ColMeta>;
export const PAYLOAD: PayloadCol[] = ['id', 'created_at', 'status', 'amount_cents', 'note'];
/** Key columns follow customer_id in this order: status first, so `status = 'open'` is always part of the seek. */
const KEY_ORDER: PayloadCol[] = ['status', 'id', 'created_at', 'amount_cents', 'note'];
const ALL_COLS: Col[] = COLS.map((c) => c.name);

export const QUERIES: Record<QueryId, { label: string; sql: string; select: Col[]; filterOpen: boolean }> = {
  q1: { label: 'SELECT created_at, amount_cents', sql: 'SELECT created_at, amount_cents FROM orders WHERE customer_id = 42', select: ['created_at', 'amount_cents'], filterOpen: false },
  q2: { label: 'SELECT count(*)', sql: 'SELECT count(*) FROM orders WHERE customer_id = 42', select: [], filterOpen: false },
  q3: { label: 'SELECT id, status', sql: 'SELECT id, status FROM orders WHERE customer_id = 42', select: ['id', 'status'], filterOpen: false },
  q4: { label: 'SELECT *', sql: 'SELECT * FROM orders WHERE customer_id = 42', select: ALL_COLS, filterOpen: false },
  q5: { label: "SELECT created_at … AND status = 'open'", sql: "SELECT created_at FROM orders WHERE customer_id = 42 AND status = 'open'", select: ['created_at'], filterOpen: true },
};

export const DEFAULT_DEF: IndexDef = { id: 'off', created_at: 'off', status: 'off', amount_cents: 'off', note: 'off' };

/* ------------------------------------------------------------- the index */

export function indexColumns(def: IndexDef, engine: Engine) {
  // InnoDB has no INCLUDE clause: payload columns become trailing key columns, and the PK rides along.
  const keys: Col[] = ['customer_id', ...KEY_ORDER.filter((c) => def[c] === 'key' || (engine === 'innodb' && def[c] === 'include'))];
  const include: Col[] = engine === 'pg' ? PAYLOAD.filter((c) => def[c] === 'include') : [];
  const implicitPk = engine === 'innodb' && !keys.includes('id');
  const all: Col[] = [...keys, ...include, ...(implicitPk ? (['id'] as Col[]) : [])];
  return { keys, include, implicitPk, all };
}

export function createIndexSql(def: IndexDef, engine: Engine) {
  const { keys, include, implicitPk } = indexColumns(def, engine);
  const base = `CREATE INDEX orders_customer_idx ON orders (${keys.join(', ')})`;
  if (engine === 'pg') return `${base}${include.length ? ` INCLUDE (${include.join(', ')})` : ''};`;
  return `${base};${implicitPk ? '  -- every record also stores id (the PRIMARY KEY)' : ''}`;
}

export function coverage(def: IndexDef, engine: Engine, q: QueryId) {
  const { all } = indexColumns(def, engine);
  const query = QUERIES[q];
  const needed = Array.from(new Set<Col>(['customer_id', ...query.select, ...(query.filterOpen ? (['status'] as Col[]) : [])]));
  const missing = ALL_COLS.filter((c) => needed.includes(c) && !all.includes(c));
  return { needed, missing, covered: missing.length === 0 };
}

/* ---------------------------------------------------------- size model */

const MAXALIGN = (n: number) => Math.ceil(n / 8) * 8;
const alignTo = (off: number, a: number) => Math.ceil(off / a) * a;
const PAGE = 8192;
const PAGE_HEADER = 24;
const BT_OPAQUE = 16;
const LEAF_FREE_TARGET = Math.floor((PAGE * 10) / 100); // fillfactor 90 at build
const NONLEAF_FREE_TARGET = Math.floor((PAGE * 30) / 100); // BTREE_NONLEAF_FILLFACTOR 70
const MAX_POSTING_AT_BUILD = Math.floor((PAGE * 10) / 100 / 8) * 8 - 4; // nbtsort.c: 812 bytes
export const STATUS_VALUES = 2;

export function dataBytes(cols: Col[]) {
  let off = 0;
  for (const c of cols) {
    const m = META[c];
    if (!m.short) off = alignTo(off, m.align);
    off += m.width;
  }
  return off;
}

export function pgIndexSize(def: IndexDef, rows: number, perCustomer: number) {
  const { keys, include } = indexColumns(def, 'pg');
  const keyData = dataBytes(keys);
  const tupleBytes = MAXALIGN(8 + dataBytes([...keys, ...include]));
  const plainEntry = tupleBytes + 4;
  const extraKeys = keys.filter((k) => k !== 'customer_id');
  let group = 1;
  if (extraKeys.length === 0) group = perCustomer;
  else if (extraKeys.length === 1 && extraKeys[0] === 'status') group = perCustomer / STATUS_VALUES;
  const dedup = include.length === 0 && group >= 2;
  const base = MAXALIGN(8 + keyData);
  const maxTids = Math.floor((Math.floor(MAX_POSTING_AT_BUILD / 8) * 8 - base) / 6);
  let leafBytes = rows * plainEntry;
  if (dedup) {
    const g = Math.max(2, Math.round(group));
    const full = Math.floor(g / maxTids);
    const rem = g % maxTids;
    const perGroup = full * (MAXALIGN(base + 6 * maxTids) + 4) + (rem >= 2 ? MAXALIGN(base + 6 * rem) + 4 : rem === 1 ? plainEntry : 0);
    leafBytes = (rows / g) * perGroup;
  }
  // Pivot tuples keep key columns only; where keys repeat, a heap TID tiebreaker is appended (_bt_truncate).
  const pivot = MAXALIGN(8 + keyData) + (group >= 2 ? 8 : 0) + 4;
  const leafUsable = PAGE - PAGE_HEADER - BT_OPAQUE - LEAF_FREE_TARGET - pivot;
  const leafPages = Math.max(1, Math.ceil(leafBytes / leafUsable));
  const nonleafUsable = PAGE - PAGE_HEADER - BT_OPAQUE - NONLEAF_FREE_TARGET - pivot;
  let level = leafPages;
  let internal = 0;
  let height = 1;
  while (level > 1) {
    level = Math.ceil((level * pivot) / nonleafUsable);
    internal += level;
    height++;
  }
  const pages = leafPages + internal + 1; // + metapage
  return {
    tupleBytes,
    plainEntry,
    dedup,
    group,
    bytesPerRow: leafBytes / rows,
    entriesPerLeaf: rows / leafPages,
    leafPages,
    internal,
    height,
    bytes: pages * PAGE,
  };
}

export function pgHeapSize(rows: number) {
  const tuple = MAXALIGN(24 + dataBytes(ALL_COLS)) + 4;
  const perPage = Math.floor((PAGE - PAGE_HEADER) / tuple);
  const pages = Math.ceil(rows / perPage);
  return { tuple, perPage, pages, bytes: pages * PAGE };
}

/* -------------------------------------------------------- lab-scale world */

export const LAB_PAGES = 40;
export const ROWS_PER_PAGE = 10;
const INITIAL_FULL_PAGES = 36;
const INITIAL_TAIL_ROWS = 3;
export const TARGET = 42;
export const LEAF_NEIGHBOURS = [40, 41, 43, 44];
export const CLUSTERED_LEVELS = 3;
const BASE_XID = 900;
export const MAX_ENTRIES = 120;

export type Row = { id: number; customer: number; page: number; status: Status };
export type Entry = { rowId: number; page: number; slot: number; status: Status; deadXid: number | null };
export type World = {
  perCustomer: number;
  seed: number;
  rows: Row[];
  fill: number[];
  pagesUsed: number;
  vm: boolean[];
  pageXid: number[];
  relallvisible: number;
  relpages: number;
  entries: Entry[];
  leafMaxTrxId: number;
  nextXid: number;
  ltxXid: number | null;
  nextSlot: number;
  lastWrite: { pg: string; innodb: string };
};

function otherCustomer(rng: () => number) {
  const c = 1 + Math.floor(rng() * 199);
  return c >= TARGET ? c + 1 : c;
}

export function makeWorld(perCustomer: number, seed = 7): World {
  const rng = makeRng(seed * 7919 + perCustomer);
  const total = INITIAL_FULL_PAGES * ROWS_PER_PAGE + INITIAL_TAIL_ROWS;
  const picked = new Set<number>();
  while (picked.size < Math.min(perCustomer, total)) picked.add(Math.floor(rng() * total));
  const rows: Row[] = [];
  const fill = Array.from({ length: LAB_PAGES }, () => 0);
  for (let i = 0; i < total; i++) {
    const page = Math.floor(i / ROWS_PER_PAGE);
    const mine = picked.has(i);
    rows.push({ id: i + 1, customer: mine ? TARGET : otherCustomer(rng), page, status: mine && rng() < 0.3 ? 'open' : 'shipped' });
    fill[page]++;
  }
  const pagesUsed = INITIAL_FULL_PAGES + 1;
  const entries: Entry[] = rows.filter((r) => r.customer === TARGET).map((r) => ({ rowId: r.id, page: r.page, slot: (r.id - 1) % ROWS_PER_PAGE, status: r.status, deadXid: null }));
  return {
    perCustomer,
    seed,
    rows,
    fill,
    pagesUsed,
    vm: Array.from({ length: LAB_PAGES }, (_, p) => p < pagesUsed),
    pageXid: Array.from({ length: LAB_PAGES }, () => BASE_XID),
    relallvisible: pagesUsed,
    relpages: pagesUsed,
    entries,
    leafMaxTrxId: BASE_XID,
    nextXid: 1000,
    ltxXid: null,
    nextSlot: ROWS_PER_PAGE,
    lastWrite: {
      pg: 'Table freshly loaded and vacuumed: every heap page is all-visible.',
      innodb: `Table freshly loaded: the last change to customer 42’s secondary leaf page was by trx ${BASE_XID}.`,
    },
  };
}

const clone = (w: World): World => ({
  ...w,
  rows: w.rows.map((r) => ({ ...r })),
  fill: w.fill.slice(),
  vm: w.vm.slice(),
  pageXid: w.pageXid.slice(),
  entries: w.entries.map((e) => ({ ...e })),
});

export type Target = 'random' | 'customer';

/** One transaction: UPDATE status of `count` rows. */
export function applyUpdate(world: World, def: IndexDef, target: Target, count = 5): World {
  const w = clone(world);
  const xid = w.nextXid++;
  const rng = makeRng(w.seed * 104729 + xid);
  const pool = target === 'customer' ? w.rows.filter((r) => r.customer === TARGET) : w.rows;
  const chosen = new Set<number>();
  const want = Math.min(count, pool.length);
  let guard = 0;
  while (chosen.size < want && guard++ < 1000) chosen.add(pool[Math.floor(rng() * pool.length)].id);
  const statusIndexed = def.status !== 'off';
  let hot = 0;
  let nonHot = 0;
  for (const id of chosen) {
    const row = w.rows[id - 1];
    const next: Status = row.status === 'open' ? 'shipped' : 'open';
    w.vm[row.page] = false;
    w.pageXid[row.page] = xid;
    if (statusIndexed) {
      nonHot++;
      if (row.customer === TARGET || LEAF_NEIGHBOURS.includes(row.customer)) w.leafMaxTrxId = xid;
      if (row.customer === TARGET) {
        for (const e of w.entries) if (e.rowId === id && e.deadXid === null) e.deadXid = xid;
        w.entries.push({ rowId: id, page: row.page, slot: w.nextSlot++, status: next, deadXid: null });
      }
    } else {
      hot++;
    }
    row.status = next;
  }
  const pages = new Set([...chosen].map((id) => w.rows[id - 1].page)).size;
  const who = target === 'customer' ? 'of customer 42’s orders' : 'random orders';
  const bumped = w.leafMaxTrxId === xid;
  w.lastWrite = {
    pg: `xid ${xid}: UPDATE status of ${chosen.size} ${who}. ${pages} heap page${pages === 1 ? ' loses its' : 's lose their'} VM bit; ${statusIndexed ? `all ${nonHot} updates are non-HOT because status is in the index` : `all ${hot} updates are HOT because no indexed column changed`}.`,
    innodb: `trx ${xid}: UPDATE status of ${chosen.size} ${who}. ${statusIndexed ? (bumped ? `status is in the index, so each update delete-marks a secondary record and inserts a new one; PAGE_MAX_TRX_ID of customer 42’s leaf page is now ${xid}.` : 'status is in the index, but none of these rows has a record on customer 42’s leaf page.') : 'status is not in the index, so no secondary record changes and PAGE_MAX_TRX_ID stays put.'}`,
  };
  return w;
}

/** One transaction: INSERT `count` new orders at the tail of the heap. */
export function applyInsert(world: World, target: Target, count = 5): World {
  const w = clone(world);
  const xid = w.nextXid++;
  const rng = makeRng(w.seed * 15485863 + xid);
  let done = 0;
  const pages = new Set<number>();
  for (let i = 0; i < count; i++) {
    const page = w.fill.findIndex((f, p) => p >= INITIAL_FULL_PAGES && f < ROWS_PER_PAGE);
    if (page < 0) break;
    const customer = target === 'customer' ? TARGET : otherCustomer(rng);
    const row: Row = { id: w.rows.length + 1, customer, page, status: 'open' };
    w.rows.push(row);
    w.fill[page]++;
    w.pagesUsed = Math.max(w.pagesUsed, page + 1);
    w.vm[page] = false;
    w.pageXid[page] = xid;
    pages.add(page);
    if (customer === TARGET || LEAF_NEIGHBOURS.includes(customer)) w.leafMaxTrxId = xid;
    if (customer === TARGET) w.entries.push({ rowId: row.id, page, slot: w.nextSlot++, status: 'open', deadXid: null });
    done++;
  }
  const bumpedI = w.leafMaxTrxId === xid;
  const who = target === 'customer' ? 'orders for customer 42' : 'orders for other customers';
  w.lastWrite = done
    ? {
        pg: `xid ${xid}: INSERT ${done} ${who} into the last heap page${pages.size === 1 ? '' : 's'}, clearing ${pages.size} VM bit${pages.size === 1 ? '' : 's'}.`,
        innodb: `trx ${xid}: INSERT ${done} ${who}. ${bumpedI ? `A new secondary record landed on customer 42’s leaf page: PAGE_MAX_TRX_ID is now ${xid}.` : 'No new record landed on customer 42’s leaf page.'}`,
      }
    : { pg: 'The lab table is full: reset it to insert more.', innodb: 'The lab table is full: reset it to insert more.' };
  return w;
}

export function tableFull(world: World) {
  return world.fill.every((f, p) => p < INITIAL_FULL_PAGES || f >= ROWS_PER_PAGE);
}

/** VACUUM (PostgreSQL) / purge (InnoDB): remove dead entries and set VM bits below the horizon. */
export function applyVacuum(world: World, engine: Engine): World {
  const w = clone(world);
  const horizon = w.ltxXid ?? w.nextXid;
  const before = w.entries.length;
  w.entries = w.entries.filter((e) => e.deadXid === null || e.deadXid >= horizon);
  const removed = before - w.entries.length;
  if (engine === 'innodb') {
    const kept = w.entries.filter((e) => e.deadXid !== null).length;
    w.lastWrite = { ...w.lastWrite, innodb: `Purge: ${removed} delete-marked secondary record${removed === 1 ? '' : 's'} removed${kept ? `, ${kept} kept for the open transaction` : ''}. PAGE_MAX_TRX_ID is unchanged: purge never lowers it.` };
    return w;
  }
  let set = 0;
  let blocked = 0;
  for (let p = 0; p < w.pagesUsed; p++) {
    if (w.vm[p]) continue;
    if (w.pageXid[p] < horizon) {
      w.vm[p] = true;
      set++;
    } else blocked++;
  }
  w.relallvisible = w.vm.slice(0, w.pagesUsed).filter(Boolean).length;
  w.relpages = w.pagesUsed;
  w.lastWrite = { ...w.lastWrite, pg: `VACUUM: ${set} VM bit${set === 1 ? '' : 's'} set, ${removed} dead index entr${removed === 1 ? 'y' : 'ies'} removed${blocked ? `; ${blocked} page${blocked === 1 ? '' : 's'} left clear because the transaction open since xid ${w.ltxXid} may still need their old row versions` : ''}. relallvisible is now ${w.relallvisible} of ${w.relpages}.` };
  return w;
}

export function setLongTxn(world: World, open: boolean): World {
  const w = clone(world);
  w.ltxXid = open ? w.nextXid++ : null;
  w.lastWrite = open
    ? { pg: `A long-running transaction (xid ${w.ltxXid}) is now open: VACUUM cannot treat anything written after it as visible to everyone.`, innodb: `A long-running read-write transaction (trx ${w.ltxXid}) is now open: read views opened from now on have up_limit_id ${w.ltxXid}.` }
    : { pg: 'The long-running transaction has committed.', innodb: 'The long-running transaction has committed.' };
  return w;
}

/**
 * Changing the definition means building a new index: one entry per live row, no dead entries.
 * CREATE INDEX also refreshes pg_class.relpages and relallvisible from the visibility map (index.c index_update_stats).
 */
export function rebuildIndex(world: World): World {
  const w = clone(world);
  w.entries = w.rows.filter((r) => r.customer === TARGET).map((r) => ({ rowId: r.id, page: r.page, slot: (r.id - 1) % ROWS_PER_PAGE, status: r.status, deadXid: null }));
  w.relallvisible = w.vm.slice(0, w.pagesUsed).filter(Boolean).length;
  w.relpages = w.pagesUsed;
  return w;
}

/* ------------------------------------------------------------- the scan */

export type Action = 'index' | 'heap' | 'heap-dead' | 'skip-dead' | 'clust' | 'clust-dead';
export type ScanStep = { entry: Entry; row: Row; action: Action; returned: boolean; filtered: boolean };

export function runScan(world: World, def: IndexDef, engine: Engine, q: QueryId) {
  const cov = coverage(def, engine, q);
  const query = QUERIES[q];
  const { keys } = indexColumns(def, engine);
  // status as a key column turns `status = 'open'` into part of the index search.
  const statusInCond = query.filterOpen && keys.includes('status');
  const byRow = world.rows;
  let entries = world.entries.slice();
  if (statusInCond) entries = entries.filter((e) => e.status === 'open');
  // Leaf order: status first when it is a key column, then heap TID (PostgreSQL) or the primary key (InnoDB).
  const statusKey = keys.includes('status');
  entries.sort(
    (a, b) =>
      (statusKey ? (a.status === b.status ? 0 : a.status === 'open' ? -1 : 1) : 0) ||
      (engine === 'pg' ? a.page - b.page || a.slot - b.slot : a.rowId - b.rowId || (a.deadXid === null ? 1 : 0) - (b.deadXid === null ? 1 : 0)),
  );
  const upLimit = world.ltxXid ?? world.nextXid;
  const pageCheck = world.leafMaxTrxId < upLimit;
  const steps: ScanStep[] = entries.map((entry) => {
    const row = byRow[entry.rowId - 1];
    const dead = entry.deadXid !== null;
    let action: Action;
    let valueStatus = row.status;
    if (engine === 'pg') {
      if (cov.covered && world.vm[entry.page]) {
        action = dead ? 'heap-dead' : 'index';
        valueStatus = entry.status;
      } else {
        action = dead ? 'heap-dead' : 'heap';
        if (cov.covered) valueStatus = entry.status;
      }
    } else if (pageCheck) {
      action = dead ? 'skip-dead' : cov.covered ? 'index' : 'clust';
      if (cov.covered) valueStatus = entry.status;
    } else {
      action = dead ? 'clust-dead' : 'clust';
    }
    const visible = !dead;
    const passes = !query.filterOpen || statusInCond || valueStatus === 'open';
    return { entry, row, action, returned: visible && passes, filtered: visible && !passes };
  });
  const plan =
    engine === 'pg'
      ? cov.covered
        ? 'Index Only Scan'
        : 'Index Scan'
      : cov.covered
        ? 'Extra: Using index'
        : 'no Using index';
  return { cov, steps, plan, statusInCond, pageCheck, upLimit };
}

export function tally(steps: ScanStep[]) {
  const heap = steps.filter((s) => s.action === 'heap' || s.action === 'heap-dead');
  const clust = steps.filter((s) => s.action === 'clust' || s.action === 'clust-dead');
  return {
    entriesRead: steps.length,
    heapFetches: heap.length,
    heapPages: new Set(heap.map((s) => s.entry.page)).size,
    fromIndex: steps.filter((s) => s.action === 'index').length,
    clustLookups: clust.length,
    clustPages: clust.length * CLUSTERED_LEVELS,
    skippedDead: steps.filter((s) => s.action === 'skip-dead').length,
    returned: steps.filter((s) => s.returned).length,
    filtered: steps.filter((s) => s.filtered).length,
  };
}

export function explainText(scan: ReturnType<typeof runScan>, def: IndexDef, engine: Engine, q: QueryId) {
  const t = tally(scan.steps);
  const query = QUERIES[q];
  if (engine === 'pg') {
    const node = [`${scan.plan} using orders_customer_idx on orders`];
    const detail = [`Index Cond: ${scan.statusInCond ? "((customer_id = 42) AND (status = 'open'::text))" : '(customer_id = 42)'}`];
    if (query.filterOpen && !scan.statusInCond) detail.push(`Filter: (status = 'open'::text)`, `Rows Removed by Filter: ${t.filtered}`);
    if (scan.cov.covered) detail.push(`Heap Fetches: ${t.heapFetches}`);
    if (q === 'q2') return ['Aggregate', `  ->  ${node[0]}`, ...detail.map((l) => `        ${l}`)].join('\n');
    return [...node, ...detail.map((l) => `  ${l}`)].join('\n');
  }
  const ref = scan.statusInCond ? 'const,const' : 'const';
  const extra = scan.cov.covered ? 'Using index' : query.filterOpen && !scan.statusInCond ? 'Using where' : 'NULL';
  return [`table: orders   type: ref   key: orders_customer_idx   ref: ${ref}`, `Extra: ${extra}`].join('\n');
}

/* --------------------------------------------------------------- the UI */

const ROLE_OPTIONS_PG = [
  { value: 'off', label: 'not in index' },
  { value: 'key', label: 'key column' },
  { value: 'include', label: 'INCLUDE' },
] as const;
const ROLE_OPTIONS_INNODB = [
  { value: 'off', label: 'not in index' },
  { value: 'key', label: 'key column' },
] as const;

const W = 680;
const LEFT = 104;
const RIGHT = 8;
const Y_ENTRY = 26;
const ENTRY_H = 26;
const Y_MID = 118;
const MID_H = 16;
const Y_PAGE = 160;
const PAGE_H = 34;
const H = Y_PAGE + PAGE_H + 26;

const pageX = (p: number) => LEFT + (p * (W - LEFT - RIGHT)) / LAB_PAGES;
const pageW = (W - LEFT - RIGHT) / LAB_PAGES;

function actionColor(a: Action) {
  if (a === 'index') return 'var(--viz-good)';
  if (a === 'heap' || a === 'clust') return 'var(--viz-critical)';
  return 'var(--viz-stale)';
}

function actionText(a: Action, engine: Engine) {
  switch (a) {
    case 'index':
      return 'answered from the index alone';
    case 'heap':
      return 'heap fetch (VM bit clear or not covered)';
    case 'heap-dead':
      return 'heap fetch found no visible version: dead entry';
    case 'skip-dead':
      return 'delete-marked, skipped without a lookup';
    case 'clust':
      return `clustered-index lookup (${CLUSTERED_LEVELS} pages)`;
    case 'clust-dead':
      return engine === 'innodb' ? 'clustered lookup: record is an old, delete-marked version' : '';
  }
}

function ScanFigure({ world, engine, steps, cursor, pageCheck, upLimit }: { world: World; engine: Engine; steps: ScanStep[]; cursor: number; pageCheck: boolean; upLimit: number }) {
  const tip = useTip();
  const n = Math.max(steps.length, 1);
  const slotW = Math.min(24, (W - LEFT - RIGHT) / n);
  const boxW = Math.max(2, slotW - (slotW > 6 ? 3 : 1));
  const entryX = (i: number) => LEFT + i * slotW;
  const perPage = Array.from({ length: LAB_PAGES }, () => 0);
  for (const r of world.rows) if (r.customer === TARGET) perPage[r.page]++;
  const done = steps.slice(0, cursor);
  const fetchedPages = new Set(done.filter((s) => s.action === 'heap' || s.action === 'heap-dead' || s.action === 'clust' || s.action === 'clust-dead').map((s) => s.entry.page));
  const current = cursor > 0 && cursor <= steps.length ? steps[cursor - 1] : null;

  return (
    <svg viewBox={`0 0 ${W} ${H}`} width={W} height={H} role="img" aria-label={`${steps.length} index entries scanned; ${done.filter((s) => s.action === 'heap' || s.action === 'heap-dead').length} heap fetches; ${done.filter((s) => s.action === 'clust' || s.action === 'clust-dead').length} clustered lookups`}>
      <text x={0} y={Y_ENTRY + 11} fontSize={11} fill="var(--viz-ink)">
        index entries
      </text>
      <text x={0} y={Y_ENTRY + 24} fontSize={10}>
        customer_id = 42
      </text>
      {steps.length === 0 ? (
        <text x={LEFT} y={Y_ENTRY + 16} fontSize={11}>
          no entries match
        </text>
      ) : null}
      {steps.map((s, i) => {
        const x = entryX(i);
        const seen = i < cursor;
        const cx = x + boxW / 2;
        const px = pageX(s.entry.page) + pageW / 2;
        const toPage = seen && (s.action === 'heap' || s.action === 'heap-dead' || s.action === 'clust' || s.action === 'clust-dead');
        const dead = s.entry.deadXid !== null;
        return (
          <g key={`${s.entry.rowId}-${s.entry.slot}`}>
            {seen ? (
              <line
                x1={cx}
                y1={Y_ENTRY + ENTRY_H}
                x2={px}
                y2={toPage ? Y_PAGE : Y_MID}
                stroke={toPage ? actionColor(s.action === 'heap-dead' || s.action === 'clust-dead' ? 'heap' : s.action) : 'var(--viz-ink-muted)'}
                strokeOpacity={toPage ? 0.75 : 0.45}
                strokeWidth={toPage ? 1.3 : 0.8}
                strokeDasharray={toPage ? undefined : '3 2'}
              />
            ) : null}
            <rect
              x={x}
              y={Y_ENTRY}
              width={boxW}
              height={ENTRY_H}
              rx={2}
              fill={seen ? actionColor(s.action) : 'var(--viz-surface)'}
              fillOpacity={seen ? (s.action === 'index' ? 0.85 : 0.8) : 1}
              stroke={dead ? 'var(--viz-stale)' : 'var(--viz-ink-muted)'}
              strokeWidth={0.8}
              strokeDasharray={dead ? '2 1.5' : undefined}
              {...tip(
                <>
                  <strong>
                    id {s.row.id}
                    {engine === 'pg' ? ` · TID (${s.entry.page},${s.entry.slot + 1})` : ` · on clustered leaf ${s.entry.page}`}
                  </strong>
                  <br />
                  status in {engine === 'pg' ? 'this entry' : 'this record'}: {s.entry.status}
                  {dead ? ` · dead since xid ${s.entry.deadXid}` : ''}
                  <br />
                  {seen ? actionText(s.action, engine) : 'not scanned yet'}
                </>,
              )}
            />
            {dead && boxW > 5 ? <line x1={x} y1={Y_ENTRY + ENTRY_H} x2={x + boxW} y2={Y_ENTRY} stroke="var(--viz-ink-2)" strokeWidth={0.8} pointerEvents="none" /> : null}
          </g>
        );
      })}
      {current && cursor < steps.length ? <rect x={entryX(cursor - 1) - 2} y={Y_ENTRY - 3} width={boxW + 4} height={ENTRY_H + 6} rx={3} fill="none" stroke="var(--viz-ink)" strokeWidth={1.5} strokeDasharray="3 2" /> : null}

      {engine === 'pg' ? (
        <g>
          <text x={0} y={Y_MID + 12} fontSize={11} fill="var(--viz-ink)">
            visibility map
          </text>
          {Array.from({ length: LAB_PAGES }, (_, p) => {
            const exists = p < world.pagesUsed;
            const set = exists && world.vm[p];
            return (
              <rect
                key={p}
                x={pageX(p) + 1}
                y={Y_MID}
                width={pageW - 2}
                height={MID_H}
                rx={2}
                fill={set ? 'var(--viz-clean)' : 'var(--viz-surface)'}
                stroke={exists ? (set ? 'var(--viz-clean)' : 'var(--viz-dirty)') : 'var(--viz-ink-muted)'}
                strokeWidth={set ? 1 : 1.6}
                strokeDasharray={exists ? undefined : '2 2'}
                strokeOpacity={exists ? 1 : 0.5}
                {...tip(
                  <>
                    <strong>heap page {p}</strong>
                    <br />
                    {!exists ? 'not allocated yet' : set ? 'all-visible bit set' : `all-visible bit clear (last written by xid ${world.pageXid[p]})`}
                  </>,
                )}
              />
            );
          })}
        </g>
      ) : (
        <g>
          <text x={0} y={Y_MID + 12} fontSize={11} fill="var(--viz-ink)">
            leaf page header
          </text>
          <rect x={LEFT} y={Y_MID - 2} width={W - LEFT - RIGHT} height={MID_H + 4} rx={3} fill={pageCheck ? 'var(--viz-clean)' : 'var(--viz-surface)'} fillOpacity={pageCheck ? 0.18 : 1} stroke={pageCheck ? 'var(--viz-clean)' : 'var(--viz-dirty)'} strokeWidth={1.6} />
          <text x={LEFT + 8} y={Y_MID + 12} fontSize={11} fill="var(--viz-ink)">
            PAGE_MAX_TRX_ID {world.leafMaxTrxId} {pageCheck ? '<' : '≥'} read view up_limit_id {upLimit} → {pageCheck ? 'all records visible, no lookups' : 'visibility unknown: clustered lookups'}
          </text>
        </g>
      )}

      <text x={0} y={Y_PAGE + 14} fontSize={11} fill="var(--viz-ink)">
        {engine === 'pg' ? 'heap pages' : 'PRIMARY leaves'}
      </text>
      <text x={0} y={Y_PAGE + 27} fontSize={10}>
        {engine === 'pg' ? 'rows of cust. 42' : 'rows of cust. 42'}
      </text>
      {Array.from({ length: LAB_PAGES }, (_, p) => {
        const exists = p < world.pagesUsed;
        const hit = fetchedPages.has(p);
        const dots = Math.min(perPage[p], 6);
        return (
          <g key={p}>
            <rect
              x={pageX(p) + 1}
              y={Y_PAGE}
              width={pageW - 2}
              height={PAGE_H}
              rx={2}
              fill="var(--viz-surface)"
              stroke={hit ? 'var(--viz-critical)' : exists ? 'var(--viz-ink-muted)' : 'var(--viz-ink-muted)'}
              strokeWidth={hit ? 2 : 0.8}
              strokeOpacity={exists ? 1 : 0.4}
              strokeDasharray={exists ? undefined : '2 2'}
            />
            {Array.from({ length: dots }, (_, d) => (
              <circle key={d} cx={pageX(p) + pageW / 2} cy={Y_PAGE + 5 + d * 5} r={1.7} fill="var(--viz-ink-2)" />
            ))}
            {p % 5 === 0 ? (
              <text x={pageX(p) + pageW / 2} y={Y_PAGE + PAGE_H + 13} fontSize={9} textAnchor="middle">
                {p}
              </text>
            ) : null}
          </g>
        );
      })}
    </svg>
  );
}

function SizeBars({ def, rows, perCustomer }: { def: IndexDef; rows: number; perCustomer: number }) {
  const heap = pgHeapSize(rows);
  const mine = pgIndexSize(def, rows, perCustomer);
  const bare = pgIndexSize(DEFAULT_DEF, rows, perCustomer);
  const bars = [
    { label: 'table (heap)', bytes: heap.bytes, fill: 'var(--viz-neutral)', stroke: 'var(--viz-ink-muted)' },
    { label: 'this index', bytes: mine.bytes, fill: 'var(--viz-seq-550)', stroke: 'var(--viz-seq-550)' },
    { label: '(customer_id) alone', bytes: bare.bytes, fill: 'var(--viz-seq-250)', stroke: 'var(--viz-seq-250)' },
  ];
  const max = Math.max(...bars.map((b) => b.bytes));
  const BW = W - 150 - 90;
  return (
    <svg viewBox={`0 0 ${W} 78`} width={W} height={78} role="img" aria-label={`Index size ${fmtBytes(mine.bytes)} against a ${fmtBytes(heap.bytes)} table`} style={{ marginTop: 8 }}>
      {bars.map((b, i) => (
        <g key={b.label}>
          <text x={0} y={16 + i * 24} fontSize={11} fill="var(--viz-ink)">
            {b.label}
          </text>
          <rect x={150} y={5 + i * 24} width={Math.max(2, (b.bytes / max) * BW)} height={15} rx={3} fill={b.fill} stroke={b.stroke} />
          <text x={150 + Math.max(2, (b.bytes / max) * BW) + 6} y={16 + i * 24} fontSize={11}>
            {fmtBytes(b.bytes)}
          </text>
        </g>
      ))}
    </svg>
  );
}

const RATE_MS = 170;

export default function IndexOnlyScanVisibilityLab() {
  const [engine, setEngine] = useState<Engine>('pg');
  const [q, setQ] = useState<QueryId>('q1');
  const [perCustomer, setPerCustomer] = useState(12);
  const [millions, setMillions] = useState(10);
  const [def, setDef] = useState<IndexDef>({ ...DEFAULT_DEF });
  const [target, setTarget] = useState<Target>('random');
  const [world, setWorld] = useState<World>(() => makeWorld(12));
  const [cursor, setCursor] = useState(Number.MAX_SAFE_INTEGER);
  const [running, setRunning] = useState(false);
  const acc = useRef(0);

  const scan = useMemo(() => runScan(world, def, engine, q), [world, def, engine, q]);
  const steps = scan.steps;
  const shownCursor = Math.min(cursor, steps.length);
  const done = steps.slice(0, shownCursor);
  const t = tally(done);
  const full = tally(steps);
  const finished = shownCursor >= steps.length;
  const size = useMemo(() => pgIndexSize(def, millions * 1e6, perCustomer), [def, millions, perCustomer]);
  const bareSize = useMemo(() => pgIndexSize(DEFAULT_DEF, millions * 1e6, perCustomer), [millions, perCustomer]);
  const heapSize = pgHeapSize(millions * 1e6);
  const explain = explainText(scan, def, engine, q);

  const cursorRef = useRef(cursor);
  cursorRef.current = cursor;
  useTicker(
    (dt) => {
      acc.current += dt;
      if (acc.current < RATE_MS) return;
      acc.current = 0;
      const next = Math.min(cursorRef.current, steps.length) + 1;
      cursorRef.current = next;
      setCursor(next);
      if (next >= steps.length) setRunning(false);
    },
    running,
  );

  const settle = () => {
    setRunning(false);
    setCursor(Number.MAX_SAFE_INTEGER);
  };
  const changeWorld = (fn: (w: World) => World) => {
    setWorld((w) => fn(w));
    settle();
  };
  const setRole = (c: PayloadCol, r: Role) => {
    setDef((d) => ({ ...d, [c]: r }));
    setWorld((w) => rebuildIndex(w));
    settle();
  };

  const allVisibleNow = world.vm.slice(0, world.pagesUsed).filter(Boolean).length;
  const blockedPages = world.ltxXid !== null ? world.vm.slice(0, world.pagesUsed).filter((v, p) => !v && world.pageXid[p] >= (world.ltxXid ?? 0)).length : 0;
  const tooManyEntries = world.entries.length >= MAX_ENTRIES;
  const statusHotBlocking = def.status !== 'off';
  const effectiveRoles = engine === 'innodb' ? ROLE_OPTIONS_INNODB : ROLE_OPTIONS_PG;
  const shownDef: IndexDef = engine === 'innodb' ? (Object.fromEntries(PAYLOAD.map((c) => [c, def[c] === 'include' ? 'key' : def[c]])) as IndexDef) : def;

  const missingText = scan.cov.missing.join(', ');
  const pgNote = !scan.cov.covered ? (
    <>
      <strong>Index Scan, not Index Only Scan.</strong> The query needs {missingText}, which the index does not store, so all {full.entriesRead} entries fetch their heap tuple and the visibility map is never consulted{q === 'q4' ? ' — SELECT * needs every column' : ''}.{' '}
      {q !== 'q4' ? `Add ${missingText} to the index to make it covering.` : 'No sensible index covers SELECT *.'}
    </>
  ) : full.heapFetches === 0 ? (
    <>
      <strong>Index Only Scan, Heap Fetches: 0.</strong> Every entry points into a page whose all-visible bit is set, so the {full.entriesRead} rows came from the index and the visibility map alone.{' '}
      Now press <em>UPDATE status of 5 orders</em> and run the query again.
    </>
  ) : (
    <>
      <strong>
        Index Only Scan, but Heap Fetches: {full.heapFetches} of {full.entriesRead}.
      </strong>{' '}
      Those entries point into {full.heapPages} heap page{full.heapPages === 1 ? '' : 's'} whose VM bit was cleared by writes since the last VACUUM, so each row had to be fetched to check its visibility
      {full.entriesRead - full.returned - full.filtered > 0 ? `, and ${full.entriesRead - full.returned - full.filtered} of them turned out to be dead versions left by non-HOT updates` : ''}.{' '}
      {blockedPages > 0 ? `VACUUM cannot set ${blockedPages} of the bits while the transaction open since xid ${world.ltxXid} might still need older versions.` : 'Run VACUUM to set the bits again.'}
    </>
  );
  const innoNote = !scan.cov.covered ? (
    <>
      <strong>Not covering: every record costs a clustered-index lookup.</strong> The query needs {missingText}. Each of the {full.clustLookups} lookups descends the PRIMARY KEY B-tree ({fmtNum(full.clustPages)} pages in all).
      {scan.cov.needed.includes('id') ? '' : ' Note that id would have been free: every secondary record already carries the primary key.'}
    </>
  ) : scan.pageCheck ? (
    <>
      <strong>Using index, and no lookups.</strong> PAGE_MAX_TRX_ID {world.leafMaxTrxId} is below the read view’s up_limit_id {scan.upLimit}, so every record on the leaf page is visible and {full.fromIndex} rows came straight from the secondary index
      {full.skippedDead ? `; ${full.skippedDead} delete-marked records were skipped without a lookup` : ''}.
      {q === 'q3' ? ' id costs nothing here: it is the primary key, stored in every secondary record.' : ''}
    </>
  ) : (
    <>
      <strong>EXPLAIN still says Using index, but all {full.clustLookups} records went to the clustered index.</strong> A transaction that modified this leaf page (trx {world.leafMaxTrxId}) is not older than the oldest transaction active when the read view opened (up_limit_id {scan.upLimit}), so InnoDB cannot use the secondary record as proof and reads the row — {fmtNum(full.clustPages)} pages.{' '}
      {world.ltxXid !== null ? 'Commit the long-running transaction and run again.' : ''}
    </>
  );

  return (
    <VizPanel
      title="Covering an index scan, and what still sends it to the table"
      subtitle="Choose which columns the index stores and run a query for one customer's orders. Every entry either answers from the index or has to visit the table: in PostgreSQL that depends on the heap page's visibility-map bit, in InnoDB on the secondary leaf page's PAGE_MAX_TRX_ID."
      controls={
        <>
          <Segmented
            label="Engine"
            value={engine}
            onChange={(e) => {
              setEngine(e);
              settle();
            }}
            options={[
              { value: 'pg', label: 'PostgreSQL' },
              { value: 'innodb', label: 'MySQL InnoDB' },
            ]}
          />
          <Choice
            label="Query (WHERE customer_id = 42)"
            value={q}
            onChange={(v) => {
              setQ(v);
              settle();
            }}
            options={(Object.keys(QUERIES) as QueryId[]).map((k) => ({ value: k, label: QUERIES[k].label }))}
          />
          <Slider
            label="Orders per customer"
            min={4}
            max={40}
            value={perCustomer}
            onChange={(v) => {
              setPerCustomer(v);
              setWorld(makeWorld(v));
              settle();
            }}
          />
        </>
      }
      legend={
        <Legend
          items={
            engine === 'pg'
              ? [
                  { label: 'Answered from the index', color: 'var(--viz-good)' },
                  { label: 'Heap fetch', color: 'var(--viz-critical)' },
                  { label: 'Dead entry', color: 'var(--viz-stale)' },
                  { label: 'VM bit set (all-visible)', color: 'var(--viz-clean)' },
                  { label: 'VM bit clear (hollow)', color: 'var(--viz-dirty)' },
                ]
              : [
                  { label: 'Answered from the index', color: 'var(--viz-good)' },
                  { label: 'Clustered-index lookup', color: 'var(--viz-critical)' },
                  { label: 'Delete-marked record', color: 'var(--viz-stale)' },
                  { label: 'Page check passes', color: 'var(--viz-clean)' },
                  { label: 'Page check fails (hollow)', color: 'var(--viz-dirty)' },
                ]
          }
        />
      }
      stats={
        engine === 'pg' ? (
          <Stats
            items={[
              { label: 'Plan node', value: scan.plan },
              { label: 'Index entries read', value: fmtNum(t.entriesRead) },
              { label: 'Heap Fetches', value: scan.cov.covered ? fmtNum(t.heapFetches) : 'not printed', hint: 'EXPLAIN ANALYZE prints Heap Fetches only for an Index Only Scan, and it counts index entries, not distinct pages. A plain Index Scan visits the heap for every entry.' },
              { label: 'Heap pages visited', value: fmtNum(t.heapPages) },
              { label: 'Rows returned', value: fmtNum(t.returned) },
              { label: 'All-visible pages: now / planner', value: `${allVisibleNow} / ${world.relallvisible} of ${world.pagesUsed}`, hint: 'The planner divides pg_class.relallvisible by the current page count. VACUUM, ANALYZE and CREATE INDEX update relallvisible; bits cleared since are invisible to it.' },
              { label: `Index size at ${millions}M rows`, value: fmtBytes(size.bytes), hint: `${fmtNum(size.bytesPerRow, 1)} bytes per row in leaf pages${size.dedup ? ', deduplicated' : ''}; height ${size.height}.` },
            ]}
          />
        ) : (
          <Stats
            items={[
              { label: 'EXPLAIN', value: scan.plan },
              { label: 'Secondary records read', value: fmtNum(t.entriesRead) },
              { label: 'PAGE_MAX_TRX_ID check', value: scan.pageCheck ? 'passes' : 'fails', hint: 'lock_sec_rec_cons_read_sees(): PAGE_MAX_TRX_ID < read view up_limit_id' },
              { label: 'Clustered lookups', value: fmtNum(t.clustLookups) },
              { label: 'Clustered pages read', value: fmtNum(t.clustPages), hint: `Model: a ${CLUSTERED_LEVELS}-level clustered index, so each lookup reads ${CLUSTERED_LEVELS} pages.` },
              { label: 'Rows returned', value: fmtNum(t.returned) },
            ]}
          />
        )
      }
      note={
        <Note>
          {engine === 'pg' ? pgNote : innoNote}
          {!finished ? ' (scanning…)' : ''}
        </Note>
      }
      table={
        <>
          <table className="viz-table">
            <thead>
              <tr>
                <th>#</th>
                <th>id</th>
                <th>{engine === 'pg' ? 'TID' : 'PRIMARY leaf'}</th>
                <th>{engine === 'pg' ? 'VM bit' : 'page check'}</th>
                <th>What the scan did</th>
                <th>Returned</th>
              </tr>
            </thead>
            <tbody>
              {steps.map((s, i) => (
                <tr key={`${s.entry.rowId}-${s.entry.slot}`}>
                  <td>{i + 1}</td>
                  <td>{s.row.id}</td>
                  <td>{engine === 'pg' ? `(${s.entry.page},${s.entry.slot + 1})` : s.entry.page}</td>
                  <td>{engine === 'pg' ? (world.vm[s.entry.page] ? '1' : '0') : scan.pageCheck ? 'passes' : 'fails'}</td>
                  <td>{actionText(s.action, engine)}</td>
                  <td>{s.returned ? 'yes' : s.filtered ? 'removed by filter' : 'no'}</td>
                </tr>
              ))}
            </tbody>
          </table>
          <table className="viz-table">
            <thead>
              <tr>
                <th>PostgreSQL B-tree at {millions}M rows</th>
                <th>Leaf tuple</th>
                <th>Bytes per row (leaf)</th>
                <th>Deduplicated</th>
                <th>Leaf pages</th>
                <th>Height</th>
                <th>Size</th>
              </tr>
            </thead>
            <tbody>
              {[
                { name: createIndexSql(def, 'pg'), s: size },
                { name: 'CREATE INDEX … (customer_id);', s: bareSize },
              ].map((r) => (
                <tr key={r.name}>
                  <td>{r.name}</td>
                  <td>{r.s.tupleBytes} B + 4 B line pointer</td>
                  <td>{fmtNum(r.s.bytesPerRow, 1)}</td>
                  <td>{r.s.dedup ? 'yes' : 'no'}</td>
                  <td>{fmtNum(r.s.leafPages)}</td>
                  <td>{r.s.height}</td>
                  <td>{fmtBytes(r.s.bytes)}</td>
                </tr>
              ))}
              <tr>
                <td>the table itself</td>
                <td>{heapSize.tuple - 4} B + 4 B line pointer</td>
                <td>{fmtNum(heapSize.tuple, 0)}</td>
                <td>—</td>
                <td>{fmtNum(heapSize.pages)}</td>
                <td>—</td>
                <td>{fmtBytes(heapSize.bytes)}</td>
              </tr>
            </tbody>
          </table>
        </>
      }
    >
      <div className="viz-controls">
        {PAYLOAD.map((c) => (
          <Choice key={c} label={`${c} (${META[c].type})`} value={shownDef[c]} onChange={(v) => setRole(c, v as Role)} options={effectiveRoles as unknown as { value: Role; label: string }[]} />
        ))}
      </div>
      <pre style={{ margin: '0 0 .5rem', padding: '.45rem .6rem', fontSize: '.78rem', lineHeight: 1.45, color: 'var(--viz-ink)', background: 'var(--viz-plane)', border: '1px solid var(--viz-border)', borderRadius: '6px', overflowX: 'auto', whiteSpace: 'pre-wrap' }}>
        {createIndexSql(def, engine)}
        {'\n'}
        {QUERIES[q].sql};
      </pre>
      <div className="viz-controls">
        <Button
          primary
          onClick={() => {
            acc.current = 0;
            setCursor(0);
            setRunning(true);
          }}
        >
          Run query
        </Button>
        <Segmented
          label="Writes hit"
          value={target}
          onChange={setTarget}
          options={[
            { value: 'random', label: 'random orders' },
            { value: 'customer', label: 'customer 42' },
          ]}
        />
        <Button onClick={() => changeWorld((w) => applyUpdate(w, def, target))} disabled={tooManyEntries} title={statusHotBlocking ? 'status is in the index: these updates are not HOT' : 'status is not indexed: these updates are HOT'}>
          UPDATE status of 5 orders
        </Button>
        <Button onClick={() => changeWorld((w) => applyInsert(w, target))} disabled={tableFull(world) || tooManyEntries}>
          INSERT 5 orders
        </Button>
        <Button onClick={() => changeWorld((w) => applyVacuum(w, engine))}>{engine === 'pg' ? 'VACUUM' : 'Purge'}</Button>
        <Check label="Long-running transaction open" checked={world.ltxXid !== null} onChange={(b) => changeWorld((w) => setLongTxn(w, b))} />
        <Button onClick={() => changeWorld(() => makeWorld(perCustomer))}>Reset table</Button>
      </div>
      <p style={{ margin: '0 0 .35rem', fontSize: '.78rem', color: 'var(--viz-ink-2)' }}>{world.lastWrite[engine]}</p>
      <TooltipHost>
        <ScanFigure world={world} engine={engine} steps={steps} cursor={shownCursor} pageCheck={scan.pageCheck} upLimit={scan.upLimit} />
      </TooltipHost>
      <pre style={{ margin: '.5rem 0 0', padding: '.45rem .6rem', fontSize: '.78rem', lineHeight: 1.45, color: 'var(--viz-ink)', background: 'var(--viz-plane)', border: '1px solid var(--viz-border)', borderRadius: '6px', overflowX: 'auto', whiteSpace: 'pre' }}>
        {finished ? explain : engine === 'pg' ? 'EXPLAIN (ANALYZE) … running' : 'running…'}
      </pre>
      {engine === 'pg' ? (
        <>
          <div className="viz-controls" style={{ marginTop: '.6rem' }}>
            <Slider label="Table rows (size model)" min={1} max={100} value={millions} onChange={setMillions} format={(v) => `${v}M`} />
          </div>
          <SizeBars def={def} rows={millions * 1e6} perCustomer={perCustomer} />
          <p style={{ margin: '.2rem 0 0', fontSize: '.75rem', color: 'var(--viz-ink-2)' }}>
            Size model: 8 KB pages, leaves built at fillfactor 90, {fmtNum(size.bytesPerRow, 1)} bytes per row in this index{size.dedup ? ' with deduplication' : ''}
            {!size.dedup && bareSize.dedup && indexColumns(def, 'pg').include.length ? ' — INCLUDE switched deduplication off' : ''}. Lab assumptions: updates find room on their page; note averages 120 characters.
          </p>
        </>
      ) : (
        <p style={{ margin: '.4rem 0 0', fontSize: '.75rem', color: 'var(--viz-ink-2)' }}>
          MySQL has no INCLUDE clause, so payload columns are key columns. Lab assumptions: customer 42’s records share one leaf page with customers 40–44; the clustered index has {CLUSTERED_LEVELS} levels; the long-running transaction is a read-write one.
        </p>
      )}
    </VizPanel>
  );
}
