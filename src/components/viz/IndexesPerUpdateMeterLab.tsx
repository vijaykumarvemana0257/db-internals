import { useMemo, useRef, useState, type ReactNode } from 'react';
import { VizPanel, Segmented, Choice, Slider, Button, Legend, Stats, Note, fmtBytes, fmtNum, makeRng, siteHref, useTicker } from './Viz';

/**
 * Write cost of an index set, one simulated minute at a time.
 *
 * An orders table (50,000 rows, 124-byte tuples, 8 KB pages) takes 2,500 row writes a minute:
 * 20% INSERT, 70% UPDATE (80% of them on the newest 5,000 orders), 10% DELETE of the oldest row.
 * The learner toggles B-tree indexes; the heap driver (fillfactor, the UPDATE's SET list and the age
 * of the oldest open snapshot) decides HOT exactly as heap_update does:
 *  - HOT iff no index covers a column in the SET list (every SET column here changes value; heap_update
 *    compares values, so a same-value SET would not block HOT) AND PageGetHeapFreeSpace >= the new tuple
 *    (fillfactor is not consulted for the same page; it only keeps inserts from filling it);
 *  - otherwise every index gets a new entry (TU_All), with the executor's indexUnchanged hint for
 *    indexes whose key columns are not in the SET list;
 *  - DELETE writes no index entry at all; dead entries wait for bottom-up deletion or VACUUM.
 * heap_page_prune_opt reclaims versions older than the snapshot horizon when a page has less than
 * Max(fillfactor reserve, BLCKSZ/10) free. At a would-be leaf split, PostgreSQL 14+ tries bottom-up
 * deletion (hinted inserts; frees ~Max(BLCKSZ/16, item) bytes of removable dead duplicates, succeeds
 * at Max(BLCKSZ/24, item)), then deduplication (13+), then splits (fillfactor 90 on the rightmost
 * leaf, else 50/50). Autovacuum runs when chain-ending dead tuples exceed 50 + 0.2 x reltuples,
 * checked once a minute; it scans every index in full.
 *
 * WAL is counted from record struct sizes (XLogRecord 24, block reference 20/8, xl_heap_update 14,
 * xl_heap_insert 3, xl_heap_delete 8, xl_btree_insert 2 + tuple, MAXALIGN'd) plus a full-page image
 * on a page's first modification in each 5-minute checkpoint cycle; VACUUM's own records are kept out of
 * the per-row numbers. "Pages dirtied" counts distinct
 * pages per cycle. Model assumptions: leaves are tracked by counts with keys spread evenly inside a
 * leaf; no internal pages; index scans set no LP_DEAD hints (no simple deletion); no TOAST, no
 * logical decoding; the throughput gauge assumes an I/O-bound write path, where throughput is
 * inversely proportional to WAL bytes plus dirtied pages x 8 KB per row, relative to a shadow copy
 * of the same table with no indexes fed the same operations.
 */
/* ------------------------------------------------------------------ model */

export const BLCKSZ = 8192;
const PAGE_HDR = 24; // PageHeaderData
const LP = 4; // ItemIdData
const T_LEN = 124; // heap tuple t_len for one orders row
export const TUP = 128; // MAXALIGN(t_len)
const TUP_DATA = T_LEN - 23; // bytes after the fixed 23-byte header, as logged in xl_heap_header + data
const BT_USABLE = BLCKSZ - PAGE_HDR - 16; // 8152: B-tree page minus header and BTPageOpaqueData
const BT_FILLFACTOR = 0.9; // BTREE_DEFAULT_FILLFACTOR, used by builds and rightmost splits
const MAX_POSTING = 1352; // Min(BTMaxItemSize / 2, INDEX_SIZE_MASK) on an 8 KB page
const ma = (n: number) => Math.ceil(n / 8) * 8;

/* WAL record pieces (xlogrecord.h): 24-byte XLogRecord, 20-byte first block reference
   (4-byte header + 12-byte RelFileLocator + 4-byte BlockNumber), 8 for a later block of the same
   relation, 5-byte image header, 2-byte short main-data header. */
const REC_HDR = 24;
const BLK_FIRST = 20;
const BLK_SAME = 8;
const IMG_HDR = 5;
const DATA_HDR = 2;

export const ROWS0 = 50_000;
export const WRITES_PER_MIN = 2_500;
export const MIX = { insert: 0.2, update: 0.7, delete: 0.1 };
export const CHECKPOINT_MIN = 5; // checkpoint_timeout default
export const MAX_MIN = 120;
/** Share of UPDATEs that hit one of the newest RECENT_ROWS orders (they change status soon after creation). */
export const RECENT_SHARE = 0.8;
export const RECENT_ROWS = 5_000;
export type Snapshot = 'short' | 's30' | 'm10';
export const SNAPSHOTS: Record<Snapshot, { label: string; seconds: number }> = {
  short: { label: 'none: only short transactions', seconds: 0.5 },
  s30: { label: '30 s (a slow report)', seconds: 30 },
  m10: { label: '10 min (idle in transaction)', seconds: 600 },
};
const lagWrites = (snap: Snapshot) => Math.round((SNAPSHOTS[snap].seconds * WRITES_PER_MIN) / 60);
const CUSTOMERS = 2_000;
const MAX_PAGES = 32_768;
const ROW_CAP = ROWS0 + MAX_MIN * Math.ceil(WRITES_PER_MIN * MIX.insert) + 10;

export type Col = 'id' | 'customer_id' | 'status' | 'created_at' | 'updated_at' | 'total' | 'external_ref';
export type Engine = 'pg17' | 'pg12';
export type Touch = 'status_updated' | 'total_updated' | 'total' | 'status';
export const TOUCHES: Record<Touch, { label: string; cols: Col[] }> = {
  status_updated: { label: 'status, updated_at', cols: ['status', 'updated_at'] },
  total_updated: { label: 'total, updated_at', cols: ['total', 'updated_at'] },
  total: { label: 'total only', cols: ['total'] },
  status: { label: 'status only', cols: ['status'] },
};

export type IndexId = 'pkey' | 'cust' | 'custCreated' | 'status' | 'updated' | 'created' | 'extref';
export type IndexDef = {
  id: IndexId;
  name: string;
  cols: Col[];
  /** IndexTupleData (8 bytes) + MAXALIGN'd key */
  item: number;
  unique: boolean;
  locked?: boolean;
  /** how many heap rows share one key value (1 = unique-ish) */
  dupsPerKey: (rows: number) => number;
};

export const INDEXES: IndexDef[] = [
  { id: 'pkey', name: 'orders_pkey', cols: ['id'], item: 16, unique: true, locked: true, dupsPerKey: () => 1 },
  { id: 'cust', name: 'orders_customer_id_idx', cols: ['customer_id'], item: 16, unique: false, dupsPerKey: (n) => n / CUSTOMERS },
  { id: 'custCreated', name: 'orders_customer_id_created_at_idx', cols: ['customer_id', 'created_at'], item: 24, unique: false, dupsPerKey: () => 1 },
  { id: 'status', name: 'orders_status_idx', cols: ['status'], item: 24, unique: false, dupsPerKey: (n) => n / 5 },
  { id: 'updated', name: 'orders_updated_at_idx', cols: ['updated_at'], item: 16, unique: false, dupsPerKey: () => 1 },
  { id: 'created', name: 'orders_created_at_idx', cols: ['created_at'], item: 16, unique: false, dupsPerKey: () => 1 },
  { id: 'extref', name: 'orders_external_ref_key', cols: ['external_ref'], item: 24, unique: true, dupsPerKey: () => 1 },
];
export const DEF: Record<IndexId, IndexDef> = Object.fromEntries(INDEXES.map((d) => [d.id, d])) as Record<IndexId, IndexDef>;

/** The read workload that decides idx_scan. Each query uses the best index present, else a sequential scan. */
export const QUERIES: { id: string; sql: string; perMin: number; uses: IndexId[]; rows: number }[] = [
  { id: 'byId', sql: 'WHERE id = $1', perMin: 1_500, uses: ['pkey'], rows: 1 },
  { id: 'feed', sql: 'WHERE customer_id = $1 ORDER BY created_at DESC LIMIT 20', perMin: 600, uses: ['custCreated', 'cust'], rows: 20 },
  { id: 'pending', sql: "WHERE status = 'pending' ORDER BY created_at LIMIT 100", perMin: 60, uses: ['status'], rows: 100 },
  { id: 'report', sql: "WHERE created_at >= now() - interval '1 hour'", perMin: 1, uses: ['created'], rows: 2_500 },
];

/** One leaf page, by counts. `fpiEpoch`: checkpoint cycle whose redo pointer the page LSN has passed; `dirtyEpoch`: cycle it was last dirtied in. */
type Leaf = { lo: number; minK: number; maxK: number; bytes: number; live: number; deadVer: number; deadOther: number; fpiEpoch: number; dirtyEpoch: number };

export type IdxCounters = { entries: number; updEntries: number; pagesDirtied: number; wal: number; splits: number; bottomUp: number; bottomUpPasses: number; dedupPasses: number; vacRemoved: number; vacPagesScanned: number; vacWal: number };
const zeroIdx = (): IdxCounters => ({ entries: 0, updEntries: 0, pagesDirtied: 0, wal: 0, splits: 0, bottomUp: 0, bottomUpPasses: 0, dedupPasses: 0, vacRemoved: 0, vacPagesScanned: 0, vacWal: 0 });

type IndexState = {
  def: IndexDef;
  leaves: Leaf[];
  /** write sequence at which each not-yet-removed dead entry died, oldest first (for the removable horizon) */
  deaths: number[];
  deathHead: number;
  deadTotal: number;
  freePages: number;
  scans: number;
  builtAt: number;
  buildWal: number;
  total: IdxCounters;
  minute: IdxCounters;
};

type Heap = {
  used: Int32Array;
  /** per page, flattened [deathSeq, reclaimableBytes, becomesLpDead] triples, oldest first */
  deaths: (number[] | undefined)[];
  lpDead: Int32Array;
  pendingChainEnds: number;
  slots: Int32Array;
  epoch: Int32Array;
  nPages: number;
  target: number;
  fsm: number[];
  fsmPos: number;
  // per-row placement
  page: Int32Array;
  slot: Int32Array;
  heapOnly: Uint8Array;
};

export type HeapCounters = { writes: number; inserts: number; updates: number; deletes: number; hot: number; newPage: number; wal: number; pagesDirtied: number; prunes: number; vacWal: number };
const zeroHeap = (): HeapCounters => ({ writes: 0, inserts: 0, updates: 0, deletes: 0, hot: 0, newPage: 0, wal: 0, pagesDirtied: 0, prunes: 0, vacWal: 0 });

export type MinuteRecord = {
  minute: number;
  heap: HeapCounters;
  shadow: HeapCounters;
  idx: Partial<Record<IndexId, IdxCounters>>;
  vacuumed: boolean;
  seqScans: number;
  readPages: number;
};

export type Sim = {
  engine: Engine;
  fillfactor: number;
  touch: Touch;
  snapshot: Snapshot;
  minute: number;
  rng: () => number;
  nextId: number;
  liveRows: number;
  alive: Uint8Array;
  cust: Int32Array;
  status: Uint8Array;
  updated: Float64Array;
  ext: Float64Array;
  opSeq: number;
  heap: Heap;
  shadow: Heap;
  indexes: Partial<Record<IndexId, IndexState>>;
  vacuums: number;
  delCursor: number;
  history: MinuteRecord[];
  heapMinute: HeapCounters;
  shadowMinute: HeapCounters;
  misses: number;
  reindexWal: number;
};

const reserveFor = (ff: number) => Math.floor((BLCKSZ * (100 - ff)) / 100);
const cycleOf = (minute: number) => Math.floor(minute / CHECKPOINT_MIN);

function newHeap(): Heap {
  return {
    used: new Int32Array(MAX_PAGES),
    deaths: [],
    lpDead: new Int32Array(MAX_PAGES),
    pendingChainEnds: 0,
    slots: new Int32Array(MAX_PAGES),
    epoch: new Int32Array(MAX_PAGES).fill(-1),
    nPages: 0,
    target: -1,
    fsm: [],
    fsmPos: 0,
    page: new Int32Array(ROW_CAP),
    slot: new Int32Array(ROW_CAP),
    heapOnly: new Uint8Array(ROW_CAP),
  };
}

/** PageGetHeapFreeSpace: free bytes between pd_lower and pd_upper, less one new line pointer. */
const heapFree = (h: Heap, p: number) => Math.max(0, BLCKSZ - PAGE_HDR - h.used[p] - LP);

/* ---- index keys: heap TID is the tiebreaker for duplicates (PostgreSQL 12+) ---- */
const tidOf = (h: Heap, r: number) => h.page[r] * 256 + (h.slot[r] % 256);
function keyOf(s: Sim, id: IndexId, r: number): number {
  switch (id) {
    case 'pkey':
    case 'created':
      return r;
    case 'updated':
      return s.updated[r];
    case 'cust':
      return s.cust[r] * 2 ** 24 + tidOf(s.heap, r);
    case 'custCreated':
      return s.cust[r] * 2 ** 20 + r;
    case 'status':
      return s.status[r] * 2 ** 24 + tidOf(s.heap, r);
    case 'extref':
      return s.ext[r];
  }
}

/** Bytes per index entry once duplicates are folded into posting lists (PostgreSQL 13+). */
function postingBytesPerEntry(def: IndexDef, rows: number) {
  const dups = Math.floor(def.dupsPerKey(rows));
  if (dups < 2) return def.item + LP;
  const g = Math.max(2, Math.min(dups, Math.floor((MAX_POSTING - def.item) / 6)));
  return (ma(def.item + 6 * g) + LP) / g;
}

const entriesOf = (l: Leaf) => l.live + l.deadVer + l.deadOther;

/* ---- WAL + dirty-page accounting ---- */

function touchHeapPage(s: Sim, h: Heap, c: HeapCounters, p: number) {
  const cyc = cycleOf(s.minute);
  if (h.epoch[p] !== cyc) {
    h.epoch[p] = cyc;
    c.pagesDirtied++;
    return true; // this modification carries a full-page image
  }
  return false;
}
const heapImage = (h: Heap, p: number) => IMG_HDR + PAGE_HDR + h.used[p];

function touchLeaf(s: Sim, ix: IndexState, leaf: Leaf, willInit = false) {
  const cyc = cycleOf(s.minute);
  if (leaf.dirtyEpoch !== cyc) {
    leaf.dirtyEpoch = cyc;
    ix.minute.pagesDirtied++;
    ix.total.pagesDirtied++;
  }
  if (leaf.fpiEpoch !== cyc) {
    leaf.fpiEpoch = cyc;
    return !willInit;
  }
  return false;
}
const leafImage = (leaf: Leaf) => IMG_HDR + PAGE_HDR + 16 + Math.round(leaf.bytes);
function idxWal(ix: IndexState, bytes: number) {
  ix.minute.wal += bytes;
  ix.total.wal += bytes;
}

/* ---- heap ---- */

const horizonOf = (s: Sim) => s.opSeq - lagWrites(s.snapshot);

function recordDeath(s: Sim, h: Heap, p: number, bytes: number, chainEnd: boolean) {
  const d = h.deaths[p] ?? (h.deaths[p] = []);
  d.push(s.opSeq, bytes, chainEnd ? 1 : 0);
  if (chainEnd) h.pendingChainEnds++;
}

/**
 * heap_page_prune_opt: skip unless the oldest dead version on the page is already removable
 * (pd_prune_xid) and the page is short of space; then reclaim every version that died before the horizon.
 */
function pruneOpt(s: Sim, h: Heap, c: HeapCounters, p: number, force = false) {
  const d = h.deaths[p];
  if (!d || d.length === 0) return;
  const horizon = horizonOf(s);
  if (d[0] > horizon) return;
  const minfree = Math.max(reserveFor(s.fillfactor), Math.floor(BLCKSZ / 10));
  if (!force && heapFree(h, p) >= minfree) return;
  let k = 0;
  let bytes = 0;
  let lps = 0;
  while (k < d.length && d[k] <= horizon) {
    bytes += d[k + 1];
    lps += d[k + 2];
    k += 3;
  }
  d.splice(0, k);
  h.used[p] -= bytes;
  h.lpDead[p] += lps;
  h.pendingChainEnds -= lps;
  c.prunes++;
  if (force) {
    // VACUUM's own writes are counted apart from the foreground per-row cost
    c.vacWal += ma(REC_HDR + BLK_FIRST + 2 * (k / 3) + DATA_HDR + 8);
    return;
  }
  const fpi = touchHeapPage(s, h, c, p);
  c.wal += ma(REC_HDR + BLK_FIRST + (fpi ? heapImage(h, p) : 2 * (k / 3)) + DATA_HDR + 8);
}

function targetPage(s: Sim, h: Heap, len: number) {
  const need = len + reserveFor(s.fillfactor);
  if (h.target >= 0 && heapFree(h, h.target) >= need) return h.target;
  while (h.fsmPos < h.fsm.length) {
    const q = h.fsm[h.fsmPos++];
    if (heapFree(h, q) >= need) return (h.target = q);
  }
  const q = Math.min(h.nPages, MAX_PAGES - 1);
  if (h.nPages < MAX_PAGES) h.nPages++;
  return (h.target = q);
}

function placeTuple(h: Heap, r: number, p: number) {
  h.used[p] += TUP + LP;
  h.page[r] = p;
  h.slot[r] = h.slots[p]++;
}

function heapInsert(s: Sim, h: Heap, c: HeapCounters, r: number) {
  const p = targetPage(s, h, TUP);
  placeTuple(h, r, p);
  h.heapOnly[r] = 0;
  const fpi = touchHeapPage(s, h, c, p);
  c.wal += ma(REC_HDR + BLK_FIRST + (fpi ? heapImage(h, p) : 5 + TUP_DATA) + DATA_HDR + 3);
}

/** Returns 'hot' | 'same' | 'newpage'. `blocked` = an index covers a column in the SET list. */
function heapUpdate(s: Sim, h: Heap, c: HeapCounters, r: number, blocked: boolean) {
  const p = h.page[r];
  pruneOpt(s, h, c, p);
  // a dead heap-only version gives back its line pointer too; a dead root keeps it (redirect or LP_DEAD)
  const deadBytes = h.heapOnly[r] ? TUP + LP : TUP;
  if (heapFree(h, p) >= TUP) {
    placeTuple(h, r, p);
    const fpi = touchHeapPage(s, h, c, p);
    // same page: log_heap_update may send only the changed middle of the tuple (prefix/suffix compression)
    c.wal += ma(REC_HDR + BLK_FIRST + (fpi ? heapImage(h, p) : 5 + 4 + 16) + DATA_HDR + 14);
    if (!blocked) {
      recordDeath(s, h, p, deadBytes, false);
      h.heapOnly[r] = 1;
      c.hot++;
      return 'hot' as const;
    }
    recordDeath(s, h, p, deadBytes, true); // the chain ends: its root line pointer will become LP_DEAD
    h.heapOnly[r] = 0;
    return 'same' as const;
  }
  const q = targetPage(s, h, TUP);
  recordDeath(s, h, p, deadBytes, true);
  placeTuple(h, r, q);
  h.heapOnly[r] = 0;
  const fpiNew = touchHeapPage(s, h, c, q);
  const fpiOld = q !== p && touchHeapPage(s, h, c, p);
  c.wal += ma(REC_HDR + BLK_FIRST + (fpiNew ? heapImage(h, q) : 5 + TUP_DATA) + BLK_SAME + (fpiOld ? heapImage(h, p) : 0) + DATA_HDR + 14);
  c.newPage++;
  return 'newpage' as const;
}

function heapDelete(s: Sim, h: Heap, c: HeapCounters, r: number) {
  const p = h.page[r];
  pruneOpt(s, h, c, p);
  recordDeath(s, h, p, h.heapOnly[r] ? TUP + LP : TUP, true);
  const fpi = touchHeapPage(s, h, c, p);
  c.wal += ma(REC_HDR + BLK_FIRST + (fpi ? heapImage(h, p) : 0) + DATA_HDR + 8);
}

/* ---- B-tree leaves ---- */

function findLeaf(leaves: Leaf[], key: number) {
  let lo = 0;
  let hi = leaves.length - 1;
  while (lo < hi) {
    const mid = (lo + hi + 1) >> 1;
    if (leaves[mid].lo <= key) lo = mid;
    else hi = mid - 1;
  }
  return lo;
}

function removeEntries(ix: IndexState, leaf: Leaf, nVer: number, nOther: number) {
  const n = entriesOf(leaf);
  if (n <= 0) return;
  const avg = leaf.bytes / n;
  leaf.deadVer -= nVer;
  leaf.deadOther -= nOther;
  leaf.bytes = Math.max(0, leaf.bytes - avg * (nVer + nOther));
  ix.deadTotal -= nVer + nOther;
}

/** Fraction of this index's dead entries whose heap tuples died before the removable horizon. */
function removableFraction(s: Sim, ix: IndexState) {
  const horizon = horizonOf(s);
  while (ix.deathHead < ix.deaths.length && ix.deaths[ix.deathHead] <= horizon) ix.deathHead++;
  if (ix.deathHead > 4096) {
    ix.deaths.splice(0, ix.deathHead);
    ix.deathHead = 0;
  }
  const recent = ix.deaths.length - ix.deathHead;
  if (ix.deadTotal <= 0) return 0;
  return Math.max(0, 1 - recent / ix.deadTotal);
}

function splitLeaf(s: Sim, ix: IndexState, i: number) {
  const leaf = ix.leaves[i];
  const rightmost = i === ix.leaves.length - 1;
  const keep = rightmost ? BT_FILLFACTOR : 0.5; // nbtsplitloc: fillfactor on the rightmost page, else ~50/50
  const moveFrac = 1 - keep;
  const mv = (x: number) => Math.round(x * moveFrac);
  const right: Leaf = {
    lo: 0,
    minK: 0,
    maxK: leaf.maxK,
    bytes: leaf.bytes * moveFrac,
    live: mv(leaf.live),
    deadVer: mv(leaf.deadVer),
    deadOther: mv(leaf.deadOther),
    fpiEpoch: -1,
    dirtyEpoch: -1,
  };
  const splitKey = leaf.minK + (leaf.maxK - leaf.minK) * keep;
  right.lo = right.minK = splitKey;
  leaf.maxK = splitKey;
  leaf.live -= right.live;
  leaf.deadVer -= right.deadVer;
  leaf.deadOther -= right.deadOther;
  leaf.bytes -= right.bytes;
  if (ix.freePages > 0) ix.freePages--;
  ix.leaves.splice(i + 1, 0, right);
  ix.minute.splits++;
  ix.total.splits++;
  const fpiLeft = touchLeaf(s, ix, leaf);
  touchLeaf(s, ix, right, true);
  let rec = REC_HDR + BLK_FIRST + (fpiLeft ? leafImage(leaf) : 0) + BLK_SAME + Math.round(right.bytes) + DATA_HDR + 14;
  if (!rightmost) {
    const sib = ix.leaves[i + 2];
    const fpiSib = touchLeaf(s, ix, sib);
    rec += BLK_SAME + (fpiSib ? leafImage(sib) : 0);
  }
  idxWal(ix, ma(rec) + ma(REC_HDR + BLK_FIRST + ix.def.item + LP + DATA_HDR + 2)); // + downlink into the parent
}

function dedupBytes(s: Sim, ix: IndexState, leaf: Leaf) {
  const def = ix.def;
  const n = entriesOf(leaf);
  const per = postingBytesPerEntry(def, s.liveRows);
  if (per < def.item + LP) return Math.min(leaf.bytes, n * per);
  // unique-ish keys: only version duplicates (dead version + its successor) pair up
  const pairs = Math.min(leaf.deadVer, leaf.live);
  const merged = (n - 2 * pairs) * (def.item + LP) + pairs * (ma(def.item + 12) + LP);
  return Math.min(leaf.bytes, merged);
}

/**
 * Insert one index tuple. At the point a leaf would split, PostgreSQL 14+ first tries bottom-up deletion
 * (only for inserts carrying the executor's indexUnchanged hint, or a unique index that saw a duplicate),
 * then deduplication (13+), and only then splits.
 */
function indexInsert(s: Sim, ix: IndexState, key: number, hint: boolean, fromUpdate: boolean) {
  const def = ix.def;
  const need = def.item + LP;
  let i = findLeaf(ix.leaves, key);
  let leaf = ix.leaves[i];
  if (leaf.bytes + need > BT_USABLE) {
    const uniquedup = def.unique && hint;
    if (s.engine === 'pg17' && (hint || uniquedup)) {
      const frac = removableFraction(s, ix);
      const eligibleVer = Math.floor(leaf.deadVer * frac);
      const eligibleOther = def.dupsPerKey(s.liveRows) >= 2 ? Math.floor(leaf.deadOther * frac) : 0;
      if (eligibleVer + eligibleOther > 0) {
        const avg = leaf.bytes / Math.max(1, entriesOf(leaf));
        const want = Math.ceil(Math.max(BLCKSZ / 16, need) / avg);
        const nVer = Math.min(eligibleVer, want);
        const nOther = Math.min(eligibleOther, want - nVer);
        removeEntries(ix, leaf, nVer, nOther);
        ix.minute.bottomUp += nVer + nOther;
        ix.total.bottomUp += nVer + nOther;
        ix.minute.bottomUpPasses++;
        ix.total.bottomUpPasses++;
        const fpi = touchLeaf(s, ix, leaf);
        idxWal(ix, ma(REC_HDR + BLK_FIRST + (fpi ? leafImage(leaf) : 2 * (nVer + nOther)) + DATA_HDR + 8));
      }
    }
    const free = BT_USABLE - leaf.bytes;
    if (!(s.engine === 'pg17' && free >= Math.max(BLCKSZ / 24, need))) {
      if (s.engine === 'pg17' && (!def.unique || uniquedup)) {
        const nb = dedupBytes(s, ix, leaf);
        if (nb < leaf.bytes - 0.5) {
          leaf.bytes = nb;
          ix.minute.dedupPasses++;
          ix.total.dedupPasses++;
          const fpi = touchLeaf(s, ix, leaf);
          idxWal(ix, ma(REC_HDR + BLK_FIRST + (fpi ? leafImage(leaf) : 16) + DATA_HDR + 4));
        }
      }
      if (leaf.bytes + need > BT_USABLE) {
        splitLeaf(s, ix, i);
        i = findLeaf(ix.leaves, key);
        leaf = ix.leaves[i];
      }
    }
  }
  leaf.bytes += need;
  leaf.live++;
  if (key < leaf.minK) leaf.minK = key;
  if (key > leaf.maxK) leaf.maxK = key;
  const fpi = touchLeaf(s, ix, leaf);
  idxWal(ix, ma(REC_HDR + BLK_FIRST + (fpi ? leafImage(leaf) : def.item) + DATA_HDR + 2));
  ix.minute.entries++;
  ix.total.entries++;
  if (fromUpdate) {
    ix.minute.updEntries++;
    ix.total.updEntries++;
  }
}

/** In PostgreSQL an UPDATE or DELETE writes nothing to the old index entry: it just stops pointing at a visible row. */
function markDead(s: Sim, ix: IndexState, key: number, version: boolean) {
  const i = findLeaf(ix.leaves, key);
  // Split keys are interpolated, so the entry may sit a few leaves away; and a run of leaves whose
  // entries are all dead (oldest rows deleted, VACUUM held back) can be long. Scan outward to the nearest live entry.
  for (let d = 0; d < ix.leaves.length; d++) {
    for (const j of d === 0 ? [i] : [i + d, i - d]) {
      const leaf = ix.leaves[j];
      if (leaf && leaf.live > 0) {
        leaf.live--;
        if (version) leaf.deadVer++;
        else leaf.deadOther++;
        ix.deadTotal++;
        ix.deaths.push(s.opSeq);
        return;
      }
    }
  }
  s.misses++;
}

function buildIndex(s: Sim, id: IndexId): IndexState {
  const def = DEF[id];
  const keys = new Float64Array(s.liveRows);
  let k = 0;
  for (let r = 0; r < s.nextId; r++) if (s.alive[r]) keys[k++] = keyOf(s, id, r);
  keys.sort();
  const per = s.engine === 'pg17' ? postingBytesPerEntry(def, s.liveRows) : def.item + LP;
  const perLeaf = Math.max(1, Math.floor((BT_USABLE * BT_FILLFACTOR) / per));
  const leaves: Leaf[] = [];
  for (let start = 0; start < k || leaves.length === 0; start += perLeaf) {
    const end = Math.min(k, start + perLeaf);
    const n = end - start;
    leaves.push({
      lo: leaves.length === 0 ? -Infinity : keys[start],
      minK: n > 0 ? keys[start] : 0,
      maxK: n > 0 ? keys[end - 1] : 0,
      bytes: n * per,
      live: n,
      deadVer: 0,
      deadOther: 0,
      fpiEpoch: -1, // simplification: charge full-page images as if the build predated the last checkpoint
      dirtyEpoch: -1,
    });
    if (end >= k) break;
  }
  return { def, leaves, deaths: [], deathHead: 0, deadTotal: 0, freePages: 0, scans: 0, builtAt: s.minute, buildWal: leaves.length * BLCKSZ, total: zeroIdx(), minute: zeroIdx() };
}

export function createSim(engine: Engine, fillfactor: number, touch: Touch, snapshot: Snapshot, enabled: IndexId[], seed = 7): Sim {
  const rng = makeRng(seed);
  const s: Sim = {
    engine,
    fillfactor,
    touch,
    snapshot,
    minute: 0,
    rng,
    nextId: 0,
    liveRows: 0,
    alive: new Uint8Array(ROW_CAP),
    cust: new Int32Array(ROW_CAP),
    status: new Uint8Array(ROW_CAP),
    updated: new Float64Array(ROW_CAP),
    ext: new Float64Array(ROW_CAP),
    opSeq: 0,
    heap: newHeap(),
    shadow: newHeap(),
    indexes: {},
    vacuums: 0,
    delCursor: 0,
    history: [],
    heapMinute: zeroHeap(),
    shadowMinute: zeroHeap(),
    misses: 0,
    reindexWal: 0,
  };
  for (let r = 0; r < ROWS0; r++) {
    newRowFields(s, r);
    for (const h of [s.heap, s.shadow]) {
      const p = targetPage(s, h, TUP);
      placeTuple(h, r, p);
    }
  }
  for (const h of [s.heap, s.shadow]) h.epoch.fill(-1);
  for (const id of enabled) s.indexes[id] = buildIndex(s, id);
  if (!s.indexes.pkey) s.indexes.pkey = buildIndex(s, 'pkey');
  for (const ix of Object.values(s.indexes)) if (ix) ix.buildWal = 0;
  return s;
}

const STATUS_WEIGHTS = [0.1, 0.1, 0.15, 0.6, 0.05]; // pending, paid, shipped, delivered, cancelled
function newRowFields(s: Sim, r: number) {
  s.alive[r] = 1;
  s.cust[r] = Math.floor(s.rng() * CUSTOMERS);
  let u = s.rng();
  let st = 0;
  while (st < 4 && u >= STATUS_WEIGHTS[st]) u -= STATUS_WEIGHTS[st++];
  s.status[r] = r < ROWS0 ? st : 0; // new orders start as 'pending'
  s.updated[r] = r;
  s.ext[r] = Math.floor(s.rng() * 2 ** 40);
  s.nextId = r + 1;
  s.liveRows++;
}

const enabledList = (s: Sim) => INDEXES.map((d) => s.indexes[d.id]).filter((x): x is IndexState => !!x);
export const touchedCols = (t: Touch) => TOUCHES[t].cols;
export const blocksHot = (id: IndexId, t: Touch) => DEF[id].cols.some((c) => touchedCols(t).includes(c));

function pickUpdateTarget(s: Sim) {
  const recent = s.rng() < RECENT_SHARE;
  for (let tries = 0; tries < 40; tries++) {
    const r = recent ? s.nextId - 1 - Math.floor(s.rng() * Math.min(RECENT_ROWS, s.nextId)) : Math.floor(s.rng() * s.nextId);
    if (s.alive[r]) return r;
  }
  return -1;
}

/** A retention job deletes the oldest live order. */
function pickDeleteTarget(s: Sim) {
  while (s.delCursor < s.nextId && !s.alive[s.delCursor]) s.delCursor++;
  return s.delCursor < s.nextId ? s.delCursor : -1;
}

export function stepMinute(s: Sim) {
  if (s.minute >= MAX_MIN) return;
  const idxs = enabledList(s);
  for (const ix of idxs) ix.minute = zeroIdx();
  s.heapMinute = zeroHeap();
  s.shadowMinute = zeroHeap();
  const hc = s.heapMinute;
  const sc = s.shadowMinute;
  const cols = touchedCols(s.touch);
  const blocked = idxs.some((ix) => ix.def.cols.some((c) => cols.includes(c)));
  const oldKeys = new Float64Array(INDEXES.length);

  for (let w = 0; w < WRITES_PER_MIN; w++) {
    const u = s.rng();
    let op: 'insert' | 'update' | 'delete' = u < MIX.insert ? 'insert' : u < MIX.insert + MIX.update ? 'update' : 'delete';
    if (op === 'insert' && s.nextId >= ROW_CAP) op = 'update';
    hc.writes++;
    sc.writes++;
    s.opSeq++;
    if (op === 'insert') {
      const r = s.nextId;
      newRowFields(s, r);
      s.updated[r] = ROWS0 + s.opSeq;
      heapInsert(s, s.heap, hc, r);
      heapInsert(s, s.shadow, sc, r);
      hc.inserts++;
      sc.inserts++;
      for (const ix of idxs) indexInsert(s, ix, keyOf(s, ix.def.id, r), false, false);
      continue;
    }
    const r = op === 'delete' ? pickDeleteTarget(s) : pickUpdateTarget(s);
    if (r < 0) continue;
    if (op === 'delete') {
      for (const ix of idxs) markDead(s, ix, keyOf(s, ix.def.id, r), false);
      heapDelete(s, s.heap, hc, r);
      heapDelete(s, s.shadow, sc, r);
      s.alive[r] = 0;
      s.liveRows--;
      hc.deletes++;
      sc.deletes++;
      continue;
    }
    // UPDATE ... SET <touched columns>
    idxs.forEach((ix, j) => (oldKeys[j] = keyOf(s, ix.def.id, r)));
    for (const c of cols) {
      if (c === 'status') s.status[r] = (s.status[r] + 1 + Math.floor(s.rng() * 4)) % 5;
      if (c === 'updated_at') s.updated[r] = ROWS0 + s.opSeq;
    }
    hc.updates++;
    sc.updates++;
    const res = heapUpdate(s, s.heap, hc, r, blocked);
    heapUpdate(s, s.shadow, sc, r, false);
    if (res === 'hot') continue;
    idxs.forEach((ix, j) => {
      const hint = !ix.def.cols.some((c) => cols.includes(c));
      markDead(s, ix, oldKeys[j], hint);
      indexInsert(s, ix, keyOf(s, ix.def.id, r), hint, true);
    });
  }

  // read workload -> idx_scan and a rough read cost
  let seqScans = 0;
  let readPages = 0;
  for (const q of QUERIES) {
    const use = q.uses.find((id) => s.indexes[id]);
    if (use) {
      s.indexes[use]!.scans += q.perMin;
      readPages += q.perMin * (3 + q.rows);
    } else {
      seqScans += q.perMin;
      readPages += q.perMin * s.heap.nPages;
    }
  }

  // autovacuum_vacuum_threshold (50) + autovacuum_vacuum_scale_factor (0.2) * reltuples, checked once per naptime (1 min)
  let vacuumed = false;
  const threshold = 50 + 0.2 * s.liveRows;
  if (s.heap.pendingChainEnds + lpDeadCount(s.heap) > threshold) {
    vacuum(s, s.heap, hc, idxs);
    s.vacuums++;
    vacuumed = true;
  }
  if (s.shadow.pendingChainEnds + lpDeadCount(s.shadow) > threshold) vacuum(s, s.shadow, sc, []);

  s.history.push({
    minute: s.minute,
    heap: { ...hc },
    shadow: { ...sc },
    idx: Object.fromEntries(idxs.map((ix) => [ix.def.id, { ...ix.minute }])),
    vacuumed,
    seqScans,
    readPages,
  });
  s.minute++;
}

function lpDeadCount(h: Heap) {
  let n = 0;
  for (let p = 0; p < h.nPages; p++) n += h.lpDead[p];
  return n;
}

function vacuum(s: Sim, h: Heap, c: HeapCounters, idxs: IndexState[]) {
  // heap pass 1: prune every page, collecting LP_DEAD items
  for (let p = 0; p < h.nPages; p++) pruneOpt(s, h, c, p, true);
  // index vacuum: every index is scanned in full, whatever the number of dead tuples
  for (const ix of idxs) {
    const scanned = ix.leaves.length + ix.freePages;
    ix.minute.vacPagesScanned += scanned;
    ix.total.vacPagesScanned += scanned;
    const frac = removableFraction(s, ix);
    for (let i = 0; i < ix.leaves.length; i++) {
      const leaf = ix.leaves[i];
      const nVer = Math.floor(leaf.deadVer * frac);
      const nOther = Math.floor(leaf.deadOther * frac);
      const dead = nVer + nOther;
      if (dead === 0) continue;
      removeEntries(ix, leaf, nVer, nOther);
      ix.minute.vacRemoved += dead;
      ix.total.vacRemoved += dead;
      ix.minute.vacWal += ma(REC_HDR + BLK_FIRST + 2 * dead + DATA_HDR + 4);
      ix.total.vacWal += ma(REC_HDR + BLK_FIRST + 2 * dead + DATA_HDR + 4);
      if (leaf.live === 0 && ix.leaves.length > 1) {
        // an empty leaf is unlinked and later recycled; the index file does not shrink
        const left = ix.leaves[i - 1];
        if (left) left.maxK = Math.max(left.maxK, leaf.maxK);
        else ix.leaves[1].lo = -Infinity;
        ix.leaves.splice(i, 1);
        ix.freePages++;
        ix.minute.vacWal += 2 * ma(REC_HDR + BLK_FIRST + 2 * BLK_SAME + DATA_HDR + 24);
        i--;
      }
    }
  }
  // heap pass 2: LP_DEAD -> LP_UNUSED, then record free space in the FSM
  const reserve = reserveFor(s.fillfactor);
  h.fsm = [];
  h.fsmPos = 0;
  for (let p = 0; p < h.nPages; p++) {
    if (h.lpDead[p] > 0) {
      h.used[p] -= h.lpDead[p] * LP;
      const n = h.lpDead[p];
      h.lpDead[p] = 0;
      c.vacWal += ma(REC_HDR + BLK_FIRST + 2 * n + DATA_HDR + 4);
    }
    if (heapFree(h, p) >= TUP + reserve) h.fsm.push(p);
  }
}

export function setIndex(s: Sim, id: IndexId, on: boolean) {
  if (DEF[id].locked) return;
  if (on && !s.indexes[id]) s.indexes[id] = buildIndex(s, id);
  if (!on) delete s.indexes[id];
}

export function reindexAll(s: Sim) {
  for (const ix of enabledList(s)) {
    const fresh = buildIndex(s, ix.def.id);
    fresh.scans = ix.scans;
    fresh.total = ix.total;
    fresh.builtAt = ix.builtAt;
    s.reindexWal += fresh.buildWal;
    s.indexes[ix.def.id] = fresh;
  }
}

/* ---- derived metrics ---- */

export type IndexReport = {
  id: IndexId;
  name: string;
  cols: Col[];
  unique: boolean;
  leafPages: number;
  filePages: number;
  bloatPct: number;
  densityPct: number;
  deadEntries: number;
  liveEntries: number;
  entriesPerMin: number;
  updEntriesPerUpdate: number;
  pagesPerMin: number;
  walPerRow: number;
  splitsPerMin: number;
  bottomUpPerMin: number;
  scans: number;
  blocksHot: boolean;
  zeroScans: boolean;
  redundantOf: string | null;
  strip: { live: number; dead: number }[];
};

export const WINDOW = 2 * CHECKPOINT_MIN;

function windowSum<T extends Record<string, number>>(recs: T[], zero: () => T): T {
  const out = zero();
  for (const r of recs) for (const k of Object.keys(out) as (keyof T)[]) (out[k] as number) += r[k] as number;
  return out;
}

export function report(s: Sim) {
  const recent = s.history.slice(-WINDOW);
  const lastMin = s.history.slice(-1);
  const heap = windowSum(recent.map((r) => r.heap), zeroHeap);
  const heap1 = windowSum(lastMin.map((r) => r.heap), zeroHeap);
  const shadow = windowSum(recent.map((r) => r.shadow), zeroHeap);
  const writes = Math.max(1, heap.writes);
  const mins = Math.max(1, recent.length);
  const idxs = enabledList(s);
  const reports: IndexReport[] = idxs.map((ix) => {
    const w = windowSum(recent.map((r) => r.idx[ix.def.id] ?? zeroIdx()), zeroIdx);
    const w1 = windowSum(lastMin.map((r) => r.idx[ix.def.id] ?? zeroIdx()), zeroIdx);
    const live = ix.leaves.reduce((a, l) => a + l.live, 0);
    const dead = ix.leaves.reduce((a, l) => a + l.deadVer + l.deadOther, 0);
    const bytes = ix.leaves.reduce((a, l) => a + l.bytes, 0);
    const per = s.engine === 'pg17' ? postingBytesPerEntry(ix.def, s.liveRows) : ix.def.item + LP;
    const needed = Math.max(1, Math.ceil((live * per) / (BT_USABLE * BT_FILLFACTOR)));
    const filePages = ix.leaves.length + ix.freePages;
    const others = idxs.filter((o) => o !== ix);
    const dominant = ix.def.unique
      ? undefined
      : others.find((o) => o.def.cols.length > ix.def.cols.length && ix.def.cols.every((c, k) => o.def.cols[k] === c));
    return {
      id: ix.def.id,
      name: ix.def.name,
      cols: ix.def.cols,
      unique: ix.def.unique,
      leafPages: ix.leaves.length,
      filePages,
      bloatPct: Math.max(0, 100 * (1 - needed / filePages)),
      densityPct: (100 * bytes) / (ix.leaves.length * BT_USABLE),
      deadEntries: dead,
      liveEntries: live,
      entriesPerMin: w.entries / mins,
      updEntriesPerUpdate: w1.updEntries / Math.max(1, heap1.updates),
      pagesPerMin: w.pagesDirtied / mins,
      walPerRow: w.wal / writes,
      splitsPerMin: w.splits / mins,
      bottomUpPerMin: w.bottomUp / mins,
      scans: ix.scans,
      blocksHot: blocksHot(ix.def.id, s.touch),
      zeroScans: ix.scans === 0 && s.minute > ix.builtAt,
      redundantOf: dominant ? dominant.def.name : null,
      strip: stripOf(ix),
    };
  });
  const idxWalPerRow = reports.reduce((a, r) => a + r.walPerRow, 0);
  const idxPages = recent.reduce((a, r) => a + Object.values(r.idx).reduce((b, x) => b + (x ? x.pagesDirtied : 0), 0), 0);
  const io = (wal: number, pages: number) => wal + pages * BLCKSZ;
  const current = io(heap.wal + idxWalPerRow * writes, heap.pagesDirtied + idxPages);
  const base = io(shadow.wal, shadow.pagesDirtied);
  const last = s.history[s.history.length - 1];
  return {
    minute: s.minute,
    windowMinutes: recent.length,
    hotPct: (100 * heap1.hot) / Math.max(1, heap1.updates),
    shadowHotPct: (100 * shadow.hot) / Math.max(1, shadow.updates),
    newPagePct: (100 * heap1.newPage) / Math.max(1, heap1.updates),
    heapPagesPerMin: heap.pagesDirtied / mins,
    idxTuplesPerUpdate: reports.reduce((a, r) => a + r.updEntriesPerUpdate, 0),
    idxTuplesPerRow: reports.reduce((a, r) => a + r.entriesPerMin, 0) / (writes / mins),
    lastMinuteVacuumed: lastMin.length ? lastMin[0].vacuumed : false,
    heapWalPerRow: heap.wal / writes,
    idxWalPerRow,
    walPerRow: heap.wal / writes + idxWalPerRow,
    shadowWalPerRow: shadow.wal / Math.max(1, shadow.writes),
    pagesPerRow: (heap.pagesDirtied + idxPages) / writes,
    shadowPagesPerRow: shadow.pagesDirtied / Math.max(1, shadow.writes),
    throughputPct: recent.length ? (100 * base) / Math.max(1, current) : 100,
    heapPages: s.heap.nPages,
    heapBloatPct: heapBloat(s, s.heap),
    heapStrip: heapStrip(s.heap),
    reindexWal: s.reindexWal,
    engine: s.engine,
    shadowHeapPages: s.shadow.nPages,
    liveRows: s.liveRows,
    vacuums: s.vacuums,
    seqScansPerMin: last ? last.seqScans : 0,
    readPagesPerMin: last ? last.readPages : 0,
    indexes: reports,
    misses: s.misses,
  };
}

function heapBloat(s: Sim, h: Heap) {
  const perPage = Math.floor((BLCKSZ - PAGE_HDR - LP - TUP - reserveFor(s.fillfactor)) / (TUP + LP)) + 1;
  const needed = Math.ceil(s.liveRows / perPage);
  return Math.max(0, 100 * (1 - needed / Math.max(1, h.nPages)));
}

export const STRIP_BINS = 48;

function heapStrip(h: Heap) {
  const bins: { live: number; dead: number }[] = [];
  const per = h.nPages / STRIP_BINS;
  const usable = BLCKSZ - PAGE_HDR;
  for (let b = 0; b < STRIP_BINS; b++) {
    const a = Math.floor(b * per);
    const e = Math.max(a + 1, Math.floor((b + 1) * per));
    let live = 0;
    let dead = 0;
    let pages = 0;
    for (let p = a; p < e && p < h.nPages; p++) {
      pages++;
      const d = h.deaths[p];
      let deadBytes = h.lpDead[p] * LP;
      if (d) for (let k = 1; k < d.length; k += 3) deadBytes += d[k];
      dead += deadBytes / usable;
      live += (h.used[p] - deadBytes) / usable;
    }
    bins.push({ live: pages ? live / pages : 0, dead: pages ? dead / pages : 0 });
  }
  return bins;
}
function stripOf(ix: IndexState) {
  const bins: { live: number; dead: number }[] = [];
  const L = ix.leaves.length + ix.freePages;
  const per = L / STRIP_BINS;
  for (let b = 0; b < STRIP_BINS; b++) {
    const a = Math.floor(b * per);
    const e = Math.max(a + 1, Math.floor((b + 1) * per));
    let live = 0;
    let dead = 0;
    let pages = 0;
    for (let i = a; i < e && i < L; i++) {
      pages++;
      const leaf = ix.leaves[i];
      if (!leaf) continue; // recycled page
      const n = Math.max(1, entriesOf(leaf));
      live += (leaf.bytes / BT_USABLE) * (leaf.live / n);
      dead += (leaf.bytes / BT_USABLE) * ((leaf.deadVer + leaf.deadOther) / n);
    }
    bins.push({ live: pages ? live / pages : 0, dead: pages ? dead / pages : 0 });
  }
  return bins;
}


/** Index tuples each operation writes right now, PostgreSQL versus InnoDB, for an index set and SET list. */
export function perOperationIndexWork(set: IndexId[], touch: Touch) {
  const n = set.length;
  const changed = set.filter((id) => blocksHot(id, touch));
  const secondaryChanged = changed.filter((id) => id !== 'pkey').length;
  return {
    pgInsert: n,
    pgUpdateHot: changed.length === 0 ? 0 : null,
    pgUpdateNonHot: n,
    pgDelete: 0,
    innoInsert: n,
    innoUpdate: 1 + 2 * secondaryChanged,
    innoDelete: n,
  };
}

/* --------------------------------------------------------------------- UI */

export const PRERUN = 15;
const STEP_MS = 650;
const HOT_PAGE = '/p02-storage-engines/pages-tuples-heap-files/04-hot-updates-redirect-line-pointers-and-opportunistic-page-pr/';
const ENGINE_LABEL: Record<Engine, string> = { pg17: 'PostgreSQL 14+', pg12: 'PostgreSQL 12' };

export function freshSim(engine: Engine, ff: number, touch: Touch, snap: Snapshot, set: IndexId[]) {
  const s = createSim(engine, ff, touch, snap, set);
  for (let i = 0; i < PRERUN; i++) stepMinute(s);
  return s;
}

function Strip({ bins, label }: { bins: { live: number; dead: number }[]; label: string }) {
  const W = 120;
  const H = 24;
  const bw = W / bins.length;
  return (
    <svg viewBox={`0 0 ${W} ${H}`} width={W} height={H} role="img" aria-label={label}>
      <rect x={0.5} y={0.5} width={W - 1} height={H - 1} fill="var(--viz-plane)" stroke="var(--viz-grid)" />
      {bins.map((b, i) => {
        const live = Math.max(0, Math.min(1, b.live));
        const dead = Math.max(0, Math.min(1 - live, b.dead));
        const lh = live * (H - 2);
        const dh = dead * (H - 2);
        return (
          <g key={i}>
            {lh > 0 ? <rect x={i * bw + 0.4} y={H - 1 - lh} width={Math.max(0.6, bw - 0.8)} height={lh} fill="var(--viz-clean)" /> : null}
            {dh > 0 ? <rect x={i * bw + 0.4} y={H - 1 - lh - dh} width={Math.max(0.6, bw - 0.8)} height={dh} fill="var(--viz-stale)" /> : null}
          </g>
        );
      })}
    </svg>
  );
}

function Flag({ tone, children }: { tone: 'critical' | 'warning' | 'good'; children: ReactNode }) {
  return (
    <span
      style={{
        display: 'inline-block',
        border: `1.5px solid var(--viz-${tone})`,
        borderRadius: 999,
        padding: '0 0.4rem',
        margin: '1px 3px 1px 0',
        fontSize: '0.6875rem',
        lineHeight: 1.5,
        color: 'var(--viz-ink)',
        whiteSpace: 'nowrap',
      }}
    >
      {children}
    </span>
  );
}

const cellNum: React.CSSProperties = { textAlign: 'right', whiteSpace: 'nowrap', verticalAlign: 'top' };
const cellTop: React.CSSProperties = { verticalAlign: 'top' };
const colList = (cols: Col[]) => `(${cols.join(', ')})`;

export default function IndexesPerUpdateMeterLab() {
  const [engine, setEngine] = useState<Engine>('pg17');
  const [touch, setTouch] = useState<Touch>('status_updated');
  const [ff, setFf] = useState(100);
  const [snap, setSnap] = useState<Snapshot>('short');
  const [running, setRunning] = useState(false);
  const [ver, setVer] = useState(0);
  const [action, setAction] = useState<string | null>(null);
  const simRef = useRef<Sim | null>(null);
  if (!simRef.current) simRef.current = freshSim('pg17', 100, 'status_updated', 'short', INDEXES.map((d) => d.id));
  const sim = simRef.current;
  const acc = useRef(0);

  const bump = () => setVer((v) => v + 1);
  const step = () => {
    if (sim.minute >= MAX_MIN) {
      setRunning(false);
      return;
    }
    stepMinute(sim);
    bump();
  };
  useTicker((dt) => {
    acc.current += dt;
    if (acc.current >= STEP_MS) {
      acc.current = 0;
      step();
    }
  }, running);

  const rep = useMemo(() => report(sim), [ver, sim]);
  const enabled = INDEXES.filter((d) => sim.indexes[d.id]).map((d) => d.id);
  const work = perOperationIndexWork(enabled, touch);
  const blockers = rep.indexes.filter((r) => r.blocksHot);
  const zeroDrop = rep.indexes.filter((r) => r.zeroScans && !r.unique);
  const redundant = rep.indexes.filter((r) => r.redundantOf);

  const reset = (e = engine, f = ff, t = touch, sn = snap, set = enabled) => {
    simRef.current = freshSim(e, f, t, sn, set);
    setAction(null);
    bump();
  };

  const slots = 7;
  const tuples = rep.idxTuplesPerUpdate;
  const thr = Math.max(0, Math.min(100, rep.throughputPct));

  return (
    <VizPanel
      title="The indexes-per-update meter"
      subtitle={
        <>
          A 50,000-row <code>orders</code> table takes 2,500 row writes a minute. Toggle its indexes and run the clock: each index an UPDATE cannot skip is one more B-tree insert, more dirty pages and more WAL, and the dead entries it leaves behind are bloat until something removes them. The heap driver — fillfactor, the SET list and the oldest snapshot — decides HOT exactly as on the{' '}
          <a href={siteHref(HOT_PAGE)}>HOT updates page</a>.
        </>
      }
      controls={
        <>
          <Segmented
            label="Engine"
            value={engine}
            onChange={(e) => {
              setEngine(e);
              reset(e);
            }}
            options={[
              { value: 'pg17', label: 'PostgreSQL 14+', title: 'Deduplication (13) and bottom-up index deletion (14)' },
              { value: 'pg12', label: 'PostgreSQL 12', title: 'Neither deduplication nor bottom-up deletion' },
            ]}
          />
          <Choice
            label="UPDATE orders SET"
            value={touch}
            onChange={(t) => {
              setTouch(t);
              sim.touch = t;
              bump();
            }}
            options={(Object.keys(TOUCHES) as Touch[]).map((k) => ({ value: k, label: TOUCHES[k].label }))}
          />
          <Slider
            label="Heap fillfactor"
            min={70}
            max={100}
            step={5}
            value={ff}
            onChange={(v) => {
              setFf(v);
              sim.fillfactor = v;
              bump();
            }}
          />
          <Choice
            label="Oldest open snapshot"
            value={snap}
            onChange={(v) => {
              setSnap(v);
              sim.snapshot = v;
              bump();
            }}
            options={(Object.keys(SNAPSHOTS) as Snapshot[]).map((k) => ({ value: k, label: SNAPSHOTS[k].label }))}
          />
        </>
      }
      legend={
        <Legend
          items={[
            { label: 'Live entries (tuples on heap pages)', color: 'var(--viz-clean)' },
            { label: 'Dead entries / dead versions awaiting cleanup', color: 'var(--viz-stale)' },
            { label: 'Index tuples written per UPDATE', color: 'var(--viz-7)' },
            { label: 'Write throughput vs no indexes (I/O-bound model)', color: 'var(--viz-3)' },
          ]}
        />
      }
      stats={
        <Stats
          items={[
            { label: 'HOT updates', value: rep.windowMinutes ? `${fmtNum(rep.hotPct, 1)}%` : '—', hint: 'n_tup_hot_upd / n_tup_upd over the last simulated minute.' },
            { label: 'Index tuples per UPDATE', value: fmtNum(tuples, 2), hint: 'New index entries written by UPDATEs, divided by UPDATEs, over the last simulated minute.' },
            { label: 'WAL per row write', value: `${fmtNum(rep.walPerRow)} B`, hint: `Foreground WAL: record bytes plus full-page images over the last 10 minutes, excluding VACUUM's own records. The same table with no indexes: ${fmtNum(rep.shadowWalPerRow)} B.` },
            { label: 'Pages dirtied per 100 writes', value: `${fmtNum(rep.pagesPerRow * 100, 1)}`, hint: `Distinct heap and index pages dirtied per checkpoint cycle. No indexes: ${fmtNum(rep.shadowPagesPerRow * 100, 1)}.` },
            { label: 'Throughput vs no indexes', value: `${fmtNum(thr)}%`, hint: 'Model: an I/O-bound write path, throughput inversely proportional to foreground WAL bytes + dirtied pages x 8 KB per row write, over the last 10 minutes.' },
            { label: 'Autovacuum runs', value: fmtNum(rep.vacuums), hint: 'Triggered when dead tuples exceed 50 + 0.2 x reltuples; each run scans every index in full.' },
          ]}
        />
      }
      note={
        <Note>
          {rep.hotPct < 5 && blockers.length ? (
            <>
              <strong>
                Every UPDATE is non-HOT: {blockers.map((b) => b.name).join(' and ')} cover{blockers.length === 1 ? 's' : ''} a column the UPDATE changes,
              </strong>{' '}
              so each UPDATE writes a new heap tuple plus one entry in all {enabled.length} indexes — including the {enabled.length - blockers.length} whose key did not change.{' '}
            </>
          ) : rep.hotPct < 95 && !blockers.length ? (
            <>
              <strong>{fmtNum(100 - rep.hotPct, 1)}% of UPDATEs found no room on their own page</strong> — no index blocks HOT, but the page was full and pruning found nothing {snap === 'short' ? 'it could remove yet' : 'older than the oldest snapshot'}, so each of those updates still wrote a tuple into every index.{' '}
            </>
          ) : blockers.length === 0 ? (
            <>
              <strong>{fmtNum(rep.hotPct, 1)}% of UPDATEs are HOT:</strong> no index covers {TOUCHES[touch].label}, so they wrote only the heap page. Index writes now come from INSERTs ({fmtNum(rep.idxTuplesPerRow, 2)} tuples per row write).{' '}
            </>
          ) : (
            <>
              <strong>{fmtNum(rep.hotPct, 1)}% HOT.</strong>{' '}
            </>
          )}
          {zeroDrop.length ? `Detector: ${zeroDrop.map((z) => z.name).join(', ')} ${zeroDrop.length === 1 ? 'has' : 'have'} never been scanned. ` : ''}
          {redundant.length ? `${redundant.map((z) => z.name).join(', ')} is a leading prefix of ${redundant[0].redundantOf}. ` : ''}
          {engine === 'pg12' ? 'PostgreSQL 12 has neither deduplication nor bottom-up deletion, so version churn splits leaves that 14+ would have cleaned.' : ''}
          {action ? ` ${action}` : ''}
        </Note>
      }
      table={
        <>
          <table className="viz-table">
            <thead>
              <tr>
                <th>Index</th>
                <th style={cellNum}>Leaf pages (file)</th>
                <th style={cellNum}>Live</th>
                <th style={cellNum}>Dead</th>
                <th style={cellNum}>Leaf density</th>
                <th style={cellNum}>Bloat</th>
                <th style={cellNum}>Entries/min</th>
                <th style={cellNum}>Splits/min</th>
                <th style={cellNum}>Bottom-up deletions/min</th>
                <th style={cellNum}>WAL per row write</th>
              </tr>
            </thead>
            <tbody>
              {rep.indexes.map((r) => (
                <tr key={r.id}>
                  <td>{r.name}</td>
                  <td style={cellNum}>
                    {fmtNum(r.leafPages)} ({fmtNum(r.filePages)})
                  </td>
                  <td style={cellNum}>{fmtNum(r.liveEntries)}</td>
                  <td style={cellNum}>{fmtNum(r.deadEntries)}</td>
                  <td style={cellNum}>{fmtNum(r.densityPct, 1)}%</td>
                  <td style={cellNum}>{fmtNum(r.bloatPct, 1)}%</td>
                  <td style={cellNum}>{fmtNum(r.entriesPerMin)}</td>
                  <td style={cellNum}>{fmtNum(r.splitsPerMin, 1)}</td>
                  <td style={cellNum}>{fmtNum(r.bottomUpPerMin)}</td>
                  <td style={cellNum}>{fmtNum(r.walPerRow)} B</td>
                </tr>
              ))}
            </tbody>
          </table>
          <table className="viz-table">
            <thead>
              <tr>
                <th>Index entries written by one…</th>
                <th style={cellNum}>PostgreSQL</th>
                <th style={cellNum}>InnoDB (clustered + secondary)</th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <td>INSERT</td>
                <td style={cellNum}>{work.pgInsert}</td>
                <td style={cellNum}>{work.innoInsert}</td>
              </tr>
              <tr>
                <td>UPDATE SET {TOUCHES[touch].label}</td>
                <td style={cellNum}>{work.pgUpdateHot === 0 ? `0 if HOT, else ${work.pgUpdateNonHot}` : work.pgUpdateNonHot}</td>
                <td style={cellNum}>{work.innoUpdate} (clustered record in place, plus delete-mark + insert per changed secondary)</td>
              </tr>
              <tr>
                <td>DELETE</td>
                <td style={cellNum}>0 now; VACUUM scans all {enabled.length} later</td>
                <td style={cellNum}>{work.innoDelete} delete-marks; purge removes them later</td>
              </tr>
            </tbody>
          </table>
        </>
      }
    >
      <div className="viz-controls">
        <Button primary onClick={() => setRunning((r) => !r)} disabled={sim.minute >= MAX_MIN}>
          {running ? 'Pause' : 'Run the clock'}
        </Button>
        <Button onClick={step} disabled={running || sim.minute >= MAX_MIN}>
          +1 minute
        </Button>
        <Button
          onClick={() => {
            const before = sim.reindexWal;
            reindexAll(sim);
            setAction(`REINDEX CONCURRENTLY rebuilt ${enabled.length} indexes, writing ${fmtBytes(sim.reindexWal - before)} of WAL; every leaf is packed to fillfactor 90 again.`);
            bump();
          }}
        >
          REINDEX CONCURRENTLY
        </Button>
        <Button
          onClick={() => {
            setRunning(false);
            reset();
          }}
        >
          Reset table
        </Button>
      </div>
      <p style={{ margin: '0.25rem 0 0.5rem', fontSize: '0.8125rem', color: 'var(--viz-ink-2)' }}>
        Minute <strong style={{ color: 'var(--viz-ink)' }}>{rep.minute}</strong> of {MAX_MIN} · checkpoint every {CHECKPOINT_MIN} min · {ENGINE_LABEL[engine]} · {fmtNum(rep.liveRows)} live rows · HOT and tuples per UPDATE: last minute; pages, WAL and throughput: last {rep.windowMinutes} min
      </p>
      <table className="viz-table" style={{ display: 'table', width: '100%', minWidth: 560, marginTop: 0, tableLayout: 'fixed' }}>
        <colgroup>
          <col style={{ width: '1.6rem' }} />
          <col />
          <col style={{ width: '8.4rem' }} />
          <col style={{ width: '4.4rem' }} />
          <col style={{ width: '4.6rem' }} />
          <col style={{ width: '4.6rem' }} />
        </colgroup>
        <thead>
          <tr>
            <th aria-label="Built" />
            <th>Relation · scans · detector</th>
            <th>Pages: live / dead · bloat</th>
            <th style={cellNum}>
              Tuples per
              <br />
              UPDATE
            </th>
            <th style={cellNum}>
              Pages
              <br />
              dirtied/min
            </th>
            <th style={cellNum}>
              WAL per
              <br />
              row write
            </th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td style={cellTop} />
            <td style={cellTop}>
              <strong>orders</strong> (heap)
              <br />
              <span style={{ color: 'var(--viz-ink-2)' }}>
                {fmtNum(rep.heapPages)} pages · {rep.seqScansPerMin ? `${fmtNum(rep.seqScansPerMin)} seq scans/min` : 'no seq scans'}
              </span>{' '}
              {rep.seqScansPerMin ? <Flag tone="critical">a query lost its index</Flag> : null}
            </td>
            <td style={cellTop}>
              <Strip bins={rep.heapStrip} label={`Heap: live and dead tuple space across ${rep.heapPages} pages`} />
              <div style={{ marginTop: 2, fontSize: '0.6875rem', color: 'var(--viz-ink-2)' }}>bloat {fmtNum(rep.heapBloatPct, 1)}%</div>
            </td>
            <td style={cellNum}>{rep.windowMinutes ? `${fmtNum(rep.hotPct, 1)}% HOT` : '—'}</td>
            <td style={cellNum}>{fmtNum(rep.heapPagesPerMin)}</td>
            <td style={cellNum}>{fmtNum(rep.heapWalPerRow)} B</td>
          </tr>
          {INDEXES.map((def) => {
            const r = rep.indexes.find((x) => x.id === def.id);
            const dominant = r && r.redundantOf ? INDEXES.find((d) => d.name === r.redundantOf) : undefined;
            return (
              <tr key={def.id}>
                <td style={cellTop}>
                  <input
                    type="checkbox"
                    aria-label={`${r ? 'Drop' : 'Create'} ${def.name}`}
                    checked={!!r}
                    disabled={def.locked}
                    onChange={(e) => {
                      const on = e.currentTarget.checked;
                      setIndex(sim, def.id, on);
                      setAction(on ? `CREATE INDEX CONCURRENTLY ${def.name}: built packed at fillfactor 90.` : `DROP INDEX CONCURRENTLY ${def.name}.`);
                      bump();
                    }}
                  />
                </td>
                <td style={{ ...cellTop, color: r ? 'var(--viz-ink)' : 'var(--viz-ink-muted)', overflowWrap: 'anywhere' }}>
                  <code style={{ fontSize: '0.68rem' }}>{def.name}</code>
                  <br />
                  <span style={{ color: r ? 'var(--viz-ink-2)' : 'var(--viz-ink-muted)' }}>
                    {def.unique ? 'UNIQUE ' : ''}
                    {colList(def.cols)}
                    {r ? ` · ${fmtNum(r.scans)} scans` : ' · not built'}
                  </span>
                  {r && (r.blocksHot || r.zeroScans || dominant) ? (
                    <div style={{ marginTop: 2 }}>
                      {r.blocksHot ? <Flag tone="critical">blocks HOT</Flag> : null}
                      {r.zeroScans && !r.unique ? <Flag tone="warning">0 scans</Flag> : null}
                      {r.zeroScans && r.unique ? <Flag tone="good">0 scans, but UNIQUE</Flag> : null}
                      {dominant ? <Flag tone="warning">prefix of {colList(dominant.cols)}</Flag> : null}
                    </div>
                  ) : null}
                </td>
                <td style={cellTop}>
                  {r ? (
                    <>
                      <Strip bins={r.strip} label={`${def.name}: live and dead entries across ${r.filePages} leaf pages`} />
                      <div style={{ marginTop: 2, fontSize: '0.6875rem', color: 'var(--viz-ink-2)' }}>
                        bloat {fmtNum(r.bloatPct, 1)}% · {fmtNum(r.filePages)} pg
                      </div>
                    </>
                  ) : null}
                </td>
                <td style={cellNum}>{r ? fmtNum(r.updEntriesPerUpdate, 2) : '—'}</td>
                <td style={cellNum}>{r ? fmtNum(r.pagesPerMin) : '—'}</td>
                <td style={cellNum}>{r ? `${fmtNum(r.walPerRow)} B` : '—'}</td>
              </tr>
            );
          })}
        </tbody>
      </table>

      <div role="img" aria-label={`${fmtNum(tuples, 2)} index tuples per UPDATE; write throughput ${fmtNum(thr)}% of the same table with no indexes`} style={{ marginTop: 12, display: 'grid', gap: 8, fontSize: '0.8125rem', color: 'var(--viz-ink)' }}>
        <div style={{ display: 'flex', flexWrap: 'wrap', alignItems: 'center', gap: '0.25rem 0.75rem' }}>
          <span style={{ flex: '1 1 13rem' }}>Index tuples written per UPDATE</span>
          <span style={{ flex: '2 1 12rem', display: 'flex', gap: 3, alignItems: 'center' }}>
            {Array.from({ length: slots }, (_, i) => {
              const fill = Math.max(0, Math.min(1, tuples - i));
              return (
                <span key={i} style={{ position: 'relative', flex: 1, height: 18, borderRadius: 3, border: '1px solid var(--viz-grid)', background: 'var(--viz-plane)', overflow: 'hidden' }}>
                  <span style={{ position: 'absolute', left: 0, top: 0, bottom: 0, width: `${fill * 100}%`, background: 'var(--viz-7)' }} />
                </span>
              );
            })}
            <span style={{ minWidth: '4.2rem', textAlign: 'right', fontVariantNumeric: 'tabular-nums' }}>
              {fmtNum(tuples, 2)} of {enabled.length}
            </span>
          </span>
        </div>
        <div style={{ display: 'flex', flexWrap: 'wrap', alignItems: 'center', gap: '0.25rem 0.75rem' }}>
          <span style={{ flex: '1 1 13rem' }}>Throughput vs no indexes (I/O-bound model)</span>
          <span style={{ flex: '2 1 12rem', display: 'flex', gap: 3, alignItems: 'center' }}>
            <span style={{ position: 'relative', flex: 1, height: 18, borderRadius: 3, border: '1px solid var(--viz-grid)', background: 'var(--viz-plane)', overflow: 'hidden' }}>
              <span style={{ position: 'absolute', left: 0, top: 0, bottom: 0, width: `${thr}%`, background: 'var(--viz-3)' }} />
            </span>
            <span style={{ minWidth: '4.2rem', textAlign: 'right', fontVariantNumeric: 'tabular-nums' }}>{fmtNum(thr)}%</span>
          </span>
        </div>
      </div>
    </VizPanel>
  );
}
