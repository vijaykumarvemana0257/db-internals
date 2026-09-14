import { useMemo, useState, type ReactNode } from 'react';
import {
  VizPanel,
  Slider,
  Segmented,
  Choice,
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
  fmtTime,
} from './Viz';

/**
 * Two ways to build the same B+tree index, side by side.
 *
 * LEFT  — repeated top-down inserts. A real per-key simulation: keys arrive in the chosen
 *         order, leaves overflow, and `_bt_findsplitloc`'s rule is applied — a rightmost
 *         page splits at the index's fillfactor, any other page splits 50/50. The right
 *         half of a split lands on a freshly extended block, which is what scrambles the
 *         mapping from key order to physical block order.
 *
 * RIGHT — the bottom-up build PostgreSQL's CREATE INDEX actually performs: scan the heap
 *         into tuplesort, spill sorted runs to temp tapes when maintenance_work_mem is
 *         exhausted, merge, then `_bt_load` fills leaf pages left to right to fillfactor
 *         (non-leaf pages to BTREE_NONLEAF_FILLFACTOR = 70) and stacks parent levels.
 *
 * The timeline is modeled wall-clock on a log axis, because the two builds differ by two
 * orders of magnitude and both have to be watchable.
 */

/* ----------------------------------------------------------------- geometry */

const PAGE_HEADER = 24; // PageHeaderData
const BT_SPECIAL = 16; // BTPageOpaqueData
const ITEM_ID = 4; // ItemIdData, the line pointer
const TUPLE_HEADER = 8; // IndexTupleData: t_tid (heap TID or downlink) + t_info
const NONLEAF_FILLFACTOR = 0.7; // BTREE_NONLEAF_FILLFACTOR in nbtree.h
const HEAP_ROW_BYTES = 120; // the table being scanned, stated in the numbers table
const DISORDER_KEYS = 400; // 'near-ascending': keys climb, but arrive out of order within a window

const align8 = (n: number) => Math.ceil(n / 8) * 8;

/** Entries that fit on one index page, exactly as the page arithmetic works out. */
function pageCapacity(pageSize: number, keyBytes: number) {
  const usable = pageSize - PAGE_HEADER - BT_SPECIAL;
  const entry = ITEM_ID + align8(TUPLE_HEADER + keyBytes);
  return Math.max(3, Math.floor(usable / entry));
}

/* ------------------------------------------------------------------ device */

const SEQ_NS_8K = 6_000; // 8 KB inside a large sequential transfer (~1.3 GB/s)
const RAND_NS_8K = 90_000; // one random 8 KB page read, queue depth 1, NVMe
const CPU_NS_INSERT = 2_200; // descent, page search, WAL record assembly
const CMP_NS = 30; // one tuplesort comparison
const FSYNC_NS = 3_000_000; // the flush that ends the build

const seqNs = (pageSize: number) => (SEQ_NS_8K * pageSize) / 8192;
const randNs = (pageSize: number) => RAND_NS_8K + (SEQ_NS_8K * (pageSize - 8192)) / 8192;

/* -------------------------------------------------------------- simulation */

type Order = 'random' | 'clustered' | 'sorted';
type CacheFit = 'fits' | 'quarter' | 'twentieth';

type Leaf = { lo: number; hi: number; mx: number; cnt: number; blk: number };

type Snap = {
  keys: number;
  leaves: number;
  splits: number;
  fill: number; // mean leaf fill, 0..1
  hist: number[]; // 10 buckets of leaf fill
  pts: { x: number; y: number }[]; // key order vs physical block order
  corr: number; // rank correlation of the two
};

function measure(leaves: Leaf[], keys: number, splits: number, cap: number): Snap {
  const P = leaves.length;
  const hist = new Array(10).fill(0);
  let sum = 0;
  for (const l of leaves) {
    const f = Math.min(1, l.cnt / cap);
    sum += f;
    hist[Math.min(9, Math.floor(f * 10))]++;
  }
  const byBlk = leaves.map((l, i) => ({ i, blk: l.blk })).sort((a, b) => a.blk - b.blk);
  const rank = new Array<number>(P);
  byBlk.forEach((o, r) => {
    rank[o.i] = r;
  });
  let corr = 1;
  if (P > 1) {
    const m = (P - 1) / 2;
    let num = 0;
    let dx = 0;
    let dy = 0;
    for (let i = 0; i < P; i++) {
      const a = i - m;
      const b = rank[i] - m;
      num += a * b;
      dx += a * a;
      dy += b * b;
    }
    corr = dx > 0 && dy > 0 ? num / Math.sqrt(dx * dy) : 1;
  }
  const step = Math.max(1, Math.ceil(P / 90));
  const pts: { x: number; y: number }[] = [];
  const d = Math.max(1, P - 1);
  for (let i = 0; i < P; i += step) pts.push({ x: i / d, y: rank[i] / d });
  return { keys, leaves: P, splits, fill: P ? sum / P : 0, hist, pts, corr };
}

/** Key-by-key inserts into a real B+tree, recording ~40 snapshots of its shape. */
function simulateInserts(n: number, cap: number, order: Order, ff: number, seed: number) {
  const rng = makeRng(seed);
  const leaves: Leaf[] = [{ lo: 0, hi: 1, mx: 0, cnt: 0, blk: 0 }];
  const target = Math.max(2, Math.floor((cap * ff) / 100));
  let nextBlk = 1;
  let splits = 0;
  const every = Math.max(1, Math.floor(n / 40));
  const snaps: Snap[] = [measure(leaves, 0, 0, cap)];

  for (let i = 0; i < n; i++) {
    let u: number;
    if (order === 'sorted') u = (i + 0.5) / n;
    else if (order === 'clustered')
      u = Math.min(0.999999, Math.max(0, (i + 0.5 + (rng() - 0.5) * DISORDER_KEYS) / n));
    else u = rng();

    let lo = 0;
    let hi = leaves.length - 1;
    while (lo < hi) {
      const mid = (lo + hi + 1) >> 1;
      if (leaves[mid].lo <= u) lo = mid;
      else hi = mid - 1;
    }
    const lf = leaves[lo];
    lf.cnt++;
    if (u > lf.mx) lf.mx = u;

    if (lf.cnt > cap) {
      splits++;
      // nbtsplitloc.c: the rightmost page of a level splits at the index's fillfactor;
      // every other page splits as close to 50/50 by space as it can.
      const rightmost = lo === leaves.length - 1;
      const leftCnt = rightmost ? Math.min(lf.cnt - 1, target) : Math.ceil(lf.cnt / 2);
      const frac = rightmost ? leftCnt / lf.cnt : 0.5 + (rng() - 0.5) * 0.2;
      // The separator is the key at position leftCnt, so the range splits inside the part
      // of it that actually holds keys — not at the arbitrary upper bound of the page.
      const mid = lf.lo + Math.max(1e-12, lf.mx - lf.lo) * frac;
      const right: Leaf = { lo: mid, hi: lf.hi, mx: lf.mx, cnt: lf.cnt - leftCnt, blk: nextBlk++ };
      lf.hi = mid;
      lf.mx = mid;
      lf.cnt = leftCnt;
      leaves.splice(lo + 1, 0, right);
    }
    if ((i + 1) % every === 0 || i === n - 1) snaps.push(measure(leaves, i + 1, splits, cap));
  }
  return { snaps, splits, leaves: leaves.length, blocks: nextBlk };
}

/* ------------------------------------------------------------------ levels */

function levelsFrom(leafPages: number, cap: number, fillRatio: number) {
  const per = Math.max(2, Math.floor(cap * fillRatio));
  const out = [Math.max(1, leafPages)];
  let cur = out[0];
  while (cur > 1) {
    cur = Math.ceil(cur / per);
    out.push(cur);
  }
  return out; // index 0 = leaf level
}

/* ------------------------------------------------------------- the model */

type Cfg = {
  rows: number;
  keyBytes: number;
  pageSize: number;
  ff: number;
  order: Order;
  memBytes: number;
  workers: number;
  cache: CacheFit;
};

type Phase = { key: string; label: string; ns: number; detail: string };

const CACHE_FRAC: Record<CacheFit, number> = { fits: 1, quarter: 0.25, twentieth: 0.05 };

function model(cfg: Cfg) {
  const { rows, keyBytes, pageSize, ff, order, memBytes, workers, cache } = cfg;
  const cap = pageCapacity(pageSize, keyBytes);
  const seq = seqNs(pageSize);
  const rnd = randNs(pageSize);

  /* ---- left: repeated inserts -------------------------------------------- */
  const sim = simulateInserts(rows, cap, order, ff, 0x5eed1);
  const last = sim.snaps[sim.snaps.length - 1];
  const insLevels = levelsFrom(last.leaves, cap, Math.max(0.4, last.fill));
  const insPages = insLevels.reduce((a, b) => a + b, 0) + 1; // + metapage
  const insBytes = insPages * pageSize;

  const cf = CACHE_FRAC[cache];
  const pMiss =
    order === 'sorted' ? 0 : order === 'clustered' ? 0.25 * (1 - cf) : Math.max(0, 1 - cf);
  const insRandomIO = Math.round(rows * 2 * pMiss);
  const insPageWrites = Math.max(insPages, Math.round(rows * pMiss) + sim.splits * 2);
  const insWalBytes = rows * (align8(TUPLE_HEADER + keyBytes) + 56) + sim.splits * pageSize;
  const insNs = rows * (CPU_NS_INSERT + 2 * pMiss * rnd) + sim.splits * (seq + pMiss * rnd);
  const insScanNs = last.leaves * (last.corr * seq + (1 - last.corr) * rnd);

  /* ---- right: bottom-up build -------------------------------------------- */
  const perLeaf = Math.max(2, Math.floor((cap * ff) / 100));
  const bulkLevels = levelsFrom(Math.ceil(rows / perLeaf), cap, NONLEAF_FILLFACTOR);
  const bulkPages = bulkLevels.reduce((a, b) => a + b, 0) + 1;
  const bulkBytes = bulkPages * pageSize;
  const bulkLeaves = bulkLevels[0];
  const bulkFill = rows / (bulkLeaves * cap);

  const participants = workers + 1;
  const sortTupleBytes = align8(TUPLE_HEADER + keyBytes) + 24; // IndexTuple + SortTuple slot
  const sortBytes = rows * sortTupleBytes;
  const memPer = memBytes / participants;
  const sharePer = sortBytes / participants;
  const spills = workers > 0 || sortBytes > memBytes;
  const runs = spills ? participants * Math.max(1, Math.ceil(sharePer / memPer)) : 0;
  const mergeFanout = Math.max(6, Math.floor(memPer / pageSize));
  const mergePasses = runs > 1 ? Math.max(1, Math.ceil(Math.log(runs) / Math.log(mergeFanout))) : 0;

  const heapPages = Math.ceil((rows * HEAP_ROW_BYTES) / (pageSize - PAGE_HEADER - 80));
  const par = 1 + workers * 0.85;
  const scanNs = (heapPages * seq) / par;
  const sortCpuNs = (rows * Math.log2(Math.max(2, rows)) * CMP_NS) / par;
  const spillNs = runs > 0 ? ((sortBytes / pageSize) * seq) / par : 0;
  const mergeNs = runs > 0 ? (sortBytes / pageSize) * seq * mergePasses : 0;
  const leafNs = bulkLeaves * seq;
  const upperNs = (bulkPages - bulkLeaves) * seq;

  const phases: Phase[] = [
    {
      key: 'scan',
      label: 'Scan the heap',
      ns: scanNs,
      detail: `${fmtNum(heapPages)} heap pages read sequentially and fed to tuplesort${
        workers > 0 ? `, split across ${participants} participants` : ''
      }.`,
    },
    {
      key: 'sort',
      label: runs > 0 ? 'Sort into runs' : 'Quicksort in memory',
      ns: sortCpuNs + spillNs,
      detail:
        runs > 0
          ? `${fmtBytes(sortBytes)} of index tuples does not fit in ${fmtBytes(memPer)} per participant, ` +
            `so tuplesort quicksorts each batch and writes ${fmtNum(runs)} run${runs === 1 ? '' : 's'} out to temp tapes.`
          : `${fmtBytes(sortBytes)} fits inside maintenance_work_mem: one quicksort, no temp files at all.`,
    },
    {
      key: 'merge',
      label: 'Merge',
      ns: mergeNs,
      detail:
        runs > 0
          ? `${fmtNum(runs)} run${runs === 1 ? '' : 's'} merged in ${mergePasses} pass${
              mergePasses === 1 ? '' : 'es'
            } (about ${fmtNum(mergeFanout)} tapes fit in memory at once); the final merge streams straight into the builder.`
          : 'Nothing to merge — the sorted array is handed to the builder directly.',
    },
    {
      key: 'leaves',
      label: 'Write leaves left to right',
      ns: leafNs,
      detail: `_bt_load fills each leaf to ${ff}% and seals it: ${fmtNum(bulkLeaves)} pages, appended in key order.`,
    },
    {
      key: 'upper',
      label: 'Stack parent levels',
      ns: upperNs,
      detail: `Each sealed child hands its high key up; parents are filled to ${Math.round(
        NONLEAF_FILLFACTOR * 100,
      )}% — ${fmtNum(bulkPages - bulkLeaves)} pages, then the metapage on block 0.`,
    },
    { key: 'flush', label: 'Flush', ns: FSYNC_NS, detail: 'One fsync of the finished relation, and the index is visible.' },
  ];
  const bulkNs = phases.reduce((a, p) => a + p.ns, 0);
  const bulkScanNs = bulkLeaves * seq;
  const bulkWal = bulkPages * pageSize;

  return {
    cap,
    sim,
    last,
    insLevels,
    insPages,
    insBytes,
    insRandomIO,
    insPageWrites,
    insWalBytes,
    insNs,
    insScanNs,
    pMiss,
    bulkLevels,
    bulkPages,
    bulkBytes,
    bulkLeaves,
    bulkFill,
    runs,
    mergePasses,
    mergeFanout,
    sortBytes,
    memPer,
    heapPages,
    phases,
    bulkNs,
    bulkScanNs,
    bulkWal,
    seq,
    rnd,
  };
}

type M = ReturnType<typeof model>;

/* ------------------------------------------------------------------ colors */

const C_INS = 'var(--viz-2)';
const C_BULK = 'var(--viz-1)';

const SEQ5 = [
  'var(--viz-seq-100)',
  'var(--viz-seq-250)',
  'var(--viz-seq-400)',
  'var(--viz-seq-550)',
  'var(--viz-seq-700)',
];
const fillColor = (f: number) => SEQ5[Math.max(0, Math.min(4, Math.floor(f * 5)))];

/* ------------------------------------------------------- the right column */

type BulkState = {
  phase: number;
  frac: number;
  runsDone: number;
  leavesDone: number;
  upperDone: boolean;
  done: boolean;
};

function bulkAt(m: M, wall: number): BulkState {
  let t = wall;
  for (let i = 0; i < m.phases.length; i++) {
    const p = m.phases[i];
    if (t < p.ns || (p.ns === 0 && t <= 0)) {
      const frac = p.ns > 0 ? t / p.ns : 1;
      return {
        phase: i,
        frac,
        runsDone: p.key === 'scan' ? 0 : p.key === 'sort' ? Math.floor(frac * m.runs) : m.runs,
        leavesDone: p.key === 'leaves' ? Math.floor(frac * m.bulkLeaves) : i > 3 ? m.bulkLeaves : 0,
        upperDone: i > 4,
        done: false,
      };
    }
    t -= p.ns;
  }
  return {
    phase: m.phases.length - 1,
    frac: 1,
    runsDone: m.runs,
    leavesDone: m.bulkLeaves,
    upperDone: true,
    done: true,
  };
}

/* ------------------------------------------------------------- the drawing */

const COL_H = 352;

export default function IndexBuildRaceLab() {
  const [rowsK, setRowsK] = useState(100);
  const [keyBytes, setKeyBytes] = useState(8);
  const [pageSize, setPageSize] = useState<'8192' | '16384'>('8192');
  const [ff, setFf] = useState(90);
  const [order, setOrder] = useState<Order>('random');
  const [memExp, setMemExp] = useState(6);
  const [workers, setWorkers] = useState(2);
  const [cache, setCache] = useState<CacheFit>('quarter');
  const [t, setT] = useState(1000);
  const [playing, setPlaying] = useState(false);

  const [ref, width] = useSize(860);
  const tip = useTip();

  const cfg: Cfg = {
    rows: rowsK * 1000,
    keyBytes,
    pageSize: Number(pageSize),
    ff,
    order,
    memBytes: (1 << memExp) * 1024 * 1024,
    workers,
    cache,
  };
  const m = useMemo(() => model(cfg), [rowsK, keyBytes, pageSize, ff, order, memExp, workers, cache]);

  const tMax = Math.max(m.insNs, m.bulkNs);
  const tMin = 1e6; // 1 ms
  const wall = tMin * Math.pow(tMax / tMin, t / 1000);

  useTicker((dt) => {
    if (t >= 1000) {
      setPlaying(false);
      return;
    }
    setT((cur) => Math.min(1000, cur + (dt / 11000) * 1000));
  }, playing);

  const bulk = bulkAt(m, wall);
  const insFrac = Math.max(0, Math.min(1, wall / m.insNs));
  const snap = m.sim.snaps[Math.min(m.sim.snaps.length - 1, Math.round(insFrac * (m.sim.snaps.length - 1)))];
  const insLevelsNow = levelsFrom(snap.leaves, m.cap, Math.max(0.4, snap.fill || 0.5));

  /* geometry */
  const gap = 26;
  const svgW = Math.max(width, 760);
  const colW = Math.max(340, (svgW - gap) / 2);
  const totalW = colW * 2 + gap;
  const height = COL_H + 8;

  const speedup = m.insNs / m.bulkNs;
  const scanRatio = m.insScanNs / m.bulkScanNs;

  const bulkHist = new Array<number>(10).fill(0);
  if (bulk.leavesDone > 0) bulkHist[Math.min(9, Math.floor(m.bulkFill * 10))] = bulk.leavesDone;
  const bulkShown = Math.min(90, bulk.leavesDone);
  const bulkPts = Array.from({ length: bulkShown }, (_, i) => {
    const g = (i / Math.max(1, bulkShown - 1)) * (bulk.leavesDone / Math.max(1, m.bulkLeaves));
    return { x: g, y: g };
  });

  const phase = m.phases[bulk.phase];
  const note = bulk.done
    ? insFrac >= 1
      ? `Both builds are finished. The bottom-up index is ${fmtNum(Math.abs(m.insPages - m.bulkPages))} pages ` +
        `${m.bulkPages <= m.insPages ? 'smaller' : 'larger — at this fillfactor its pages are packed looser than random splits leave them'}, ` +
        `its leaves are ${Math.round(m.bulkFill * 100)}% full against ${Math.round(snap.fill * 100)}%, and its key order ` +
        `is its physical order — a full range scan reads it as one sequential file.`
      : `The bottom-up build finished at ${fmtTime(m.bulkNs)}. The insert-by-insert build is ${(insFrac * 100).toFixed(
          insFrac < 0.1 ? 2 : 1,
        )}% done — ${fmtNum(snap.leaves)} leaf pages, ${fmtNum(snap.splits)} splits so far.`
    : `${phase.label}: ${phase.detail}`;

  return (
    <VizPanel
      title="Two ways to build the same index"
      subtitle="Left: one INSERT at a time, top-down, splitting as it goes. Right: what CREATE INDEX actually does — sort everything first, then write leaves left to right and stack the parents. Scrub the modeled clock (log scale) to watch both."
      controls={
        <>
          <Slider
            label="Rows"
            min={10}
            max={200}
            step={10}
            value={rowsK}
            onChange={setRowsK}
            format={(n) => fmtNum(n * 1000)}
          />
          <Slider
            label="Key width"
            min={8}
            max={128}
            step={8}
            value={keyBytes}
            onChange={setKeyBytes}
            format={(n) => `${n} B`}
          />
          <Segmented
            label="Page"
            value={pageSize}
            onChange={setPageSize}
            options={[
              { value: '8192', label: '8 KB', title: 'PostgreSQL BLCKSZ' },
              { value: '16384', label: '16 KB', title: 'InnoDB innodb_page_size' },
            ]}
          />
          <Slider label="fillfactor" min={50} max={100} step={5} value={ff} onChange={setFf} format={(n) => `${n}%`} />
          <Segmented
            label="Key arrival"
            value={order}
            onChange={setOrder}
            options={[
              { value: 'random', label: 'random', title: 'UUIDv4, hashes, natural keys' },
              {
                value: 'clustered',
                label: 'near-ascending',
                title: 'now() from many concurrent sessions: climbing, with a window of local disorder',
              },
              { value: 'sorted', label: 'strictly ascending', title: 'a bigserial read by one writer, or a UUIDv7' },
            ]}
          />
          <Slider
            label="maintenance_work_mem"
            min={0}
            max={10}
            value={memExp}
            onChange={setMemExp}
            format={(n) => fmtBytes((1 << n) * 1024 * 1024)}
          />
          <Slider label="Parallel workers" min={0} max={4} value={workers} onChange={setWorkers} />
          <Choice
            label="Index vs buffer pool"
            value={cache}
            onChange={setCache}
            options={[
              { value: 'fits', label: 'index fits in the pool' },
              { value: 'quarter', label: 'pool holds 25%' },
              { value: 'twentieth', label: 'pool holds 5%' },
            ]}
          />
          <Slider
            label="Clock"
            min={0}
            max={1000}
            value={Math.round(t)}
            onChange={(v) => {
              setPlaying(false);
              setT(v);
            }}
            format={() => fmtTime(wall)}
          />
          <Button onClick={() => setPlaying((p) => !p)} primary>
            {playing ? 'Pause' : 'Run'}
          </Button>
          <Button
            onClick={() => {
              setPlaying(false);
              setT(0);
            }}
          >
            Rewind
          </Button>
        </>
      }
      legend={
        <Legend
          items={[
            { label: 'Repeated INSERT (top-down)', color: C_INS },
            { label: 'Bottom-up build (CREATE INDEX)', color: C_BULK },
            { label: 'page fill: low', color: SEQ5[0] },
            { label: 'page fill: full', color: SEQ5[4] },
            { label: 'target fillfactor (dashed)', color: 'var(--viz-critical)', shape: 'line' },
          ]}
        />
      }
      stats={
        <Stats
          items={[
            { label: 'Bottom-up build', value: fmtTime(m.bulkNs), hint: 'Scan + sort + merge + sequential page writes' },
            {
              label: 'Repeated inserts',
              value: fmtTime(m.insNs),
              hint: 'N descents, each paying a random leaf read and write once the index outgrows the pool',
            },
            { label: 'Speedup', value: `${speedup >= 10 ? Math.round(speedup) : speedup.toFixed(1)}×` },
            {
              label: 'Leaf fill',
              value: `${Math.round(m.bulkFill * 100)}% vs ${Math.round(m.last.fill * 100)}%`,
              hint: 'Bottom-up hits fillfactor exactly; random inserts converge on ln 2 ≈ 69%',
            },
            {
              label: 'Index size',
              value: `${fmtBytes(m.bulkBytes)} vs ${fmtBytes(m.insBytes)}`,
              hint: 'Same keys, same page size — different packing',
            },
            {
              label: 'Key order = block order',
              value: `1.00 vs ${m.last.corr.toFixed(2)}`,
              hint: 'Rank correlation between a leaf page’s key position and its block number',
            },
            {
              label: 'Full index scan',
              value: `${fmtTime(m.bulkScanNs)} vs ${fmtTime(m.insScanNs)}`,
              hint: `Following the leaf right-links: sequential vs ${scanRatio.toFixed(1)}× that`,
            },
            {
              label: 'Sort runs',
              value: m.runs === 0 ? 'none (in memory)' : fmtNum(m.runs),
              hint: 'Runs spilled to temp tapes, then merged',
            },
            { label: 'WAL written', value: `${fmtBytes(m.bulkWal)} vs ${fmtBytes(m.insWalBytes)}` },
          ]}
        />
      }
      note={<Note>{note}</Note>}
      table={
        <>
          <table className="viz-table">
            <thead>
              <tr>
                <th>Bottom-up phase</th>
                <th>Time</th>
                <th>What it does</th>
              </tr>
            </thead>
            <tbody>
              {m.phases.map((p) => (
                <tr key={p.key}>
                  <td>{p.label}</td>
                  <td>{fmtTime(p.ns)}</td>
                  <td>{p.detail}</td>
                </tr>
              ))}
              <tr>
                <td>
                  <strong>Total</strong>
                </td>
                <td>
                  <strong>{fmtTime(m.bulkNs)}</strong>
                </td>
                <td>—</td>
              </tr>
            </tbody>
          </table>
          <table className="viz-table">
            <thead>
              <tr>
                <th>Metric</th>
                <th>Repeated INSERT</th>
                <th>Bottom-up build</th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <td>Leaf pages</td>
                <td>{fmtNum(m.last.leaves)}</td>
                <td>{fmtNum(m.bulkLeaves)}</td>
              </tr>
              <tr>
                <td>Total pages (incl. metapage)</td>
                <td>{fmtNum(m.insPages)}</td>
                <td>{fmtNum(m.bulkPages)}</td>
              </tr>
              <tr>
                <td>Levels</td>
                <td>{m.insLevels.length}</td>
                <td>{m.bulkLevels.length}</td>
              </tr>
              <tr>
                <td>Mean leaf fill</td>
                <td>{Math.round(m.last.fill * 100)}%</td>
                <td>{Math.round(m.bulkFill * 100)}%</td>
              </tr>
              <tr>
                <td>Page splits</td>
                <td>{fmtNum(m.sim.splits)}</td>
                <td>0</td>
              </tr>
              <tr>
                <td>Random page I/Os</td>
                <td>{fmtNum(m.insRandomIO)}</td>
                <td>0</td>
              </tr>
              <tr>
                <td>Page writes</td>
                <td>{fmtNum(m.insPageWrites)}</td>
                <td>{fmtNum(m.bulkPages)}</td>
              </tr>
              <tr>
                <td>WAL bytes</td>
                <td>{fmtBytes(m.insWalBytes)}</td>
                <td>{fmtBytes(m.bulkWal)}</td>
              </tr>
              <tr>
                <td>Key order vs block order</td>
                <td>{m.last.corr.toFixed(3)}</td>
                <td>1.000</td>
              </tr>
              <tr>
                <td>Build time</td>
                <td>{fmtTime(m.insNs)}</td>
                <td>{fmtTime(m.bulkNs)}</td>
              </tr>
              <tr>
                <td>Full index scan afterwards</td>
                <td>{fmtTime(m.insScanNs)}</td>
                <td>{fmtTime(m.bulkScanNs)}</td>
              </tr>
            </tbody>
          </table>
          <p>
            Page arithmetic: usable space {fmtNum(cfg.pageSize - PAGE_HEADER - BT_SPECIAL)} bytes (page size minus a{' '}
            {PAGE_HEADER}-byte page header and a {BT_SPECIAL}-byte B-tree special area), one entry ={' '}
            {ITEM_ID} bytes of line pointer + {TUPLE_HEADER}-byte index-tuple header + a {keyBytes}-byte key aligned to 8
            = {ITEM_ID + align8(TUPLE_HEADER + keyBytes)} bytes, so <strong>{fmtNum(m.cap)} entries per page</strong> and{' '}
            {fmtNum(Math.floor((m.cap * ff) / 100))} at fillfactor {ff}%. Timing model: {fmtTime(m.seq)} per page inside a
            sequential transfer, {fmtTime(m.rnd)} per random page read, {fmtTime(CPU_NS_INSERT)} of CPU per insert,{' '}
            {CMP_NS} ns per sort comparison, heap rows assumed {HEAP_ROW_BYTES} bytes wide. Miss probability on the insert
            path here: {Math.round(m.pMiss * 100)}%.
          </p>
        </>
      }
    >
      <div ref={ref}>
        <TooltipHost>
          <svg
            width={totalW}
            height={height}
            role="img"
            aria-label="Side-by-side simulation of building a B+tree by repeated inserts versus a sorted bottom-up build"
          >
            <Column
              x={0}
              w={colW}
              accent={C_INS}
              title="Repeated INSERT — top-down"
              status={
                insFrac >= 1
                  ? `done · ${fmtNum(m.sim.splits)} splits`
                  : `${fmtNum(snap.keys)} of ${fmtNum(cfg.rows)} keys · ${fmtNum(snap.splits)} splits`
              }
              progress={insFrac}
              hist={snap.hist}
              histTotal={snap.leaves}
              bandTitle="Leaf fill distribution"
              ff={ff}
              levels={insLevelsNow}
              levelFill={[snap.fill || 0, ...insLevelsNow.slice(1).map(() => snap.fill || 0)]}
              pts={snap.pts}
              corr={snap.corr}
              cap={m.cap}
              tip={tip}
              scatterLabel="right half of a split → a new block"
            />
            <Column
              x={colW + gap}
              w={colW}
              accent={C_BULK}
              title="Bottom-up build — CREATE INDEX"
              status={bulk.done ? `done in ${fmtTime(m.bulkNs)}` : m.phases[bulk.phase].label}
              progress={bulk.done ? 1 : Math.min(1, wall / m.bulkNs)}
              hist={bulkHist}
              histTotal={bulk.leavesDone}
              bandTitle={bulk.phase <= 2 ? 'Sorted runs on temp tapes' : 'Leaf fill distribution'}
              runs={bulk.phase <= 2 ? { total: m.runs, done: bulk.runsDone, merging: bulk.phase === 2 } : undefined}
              ff={ff}
              levels={bulk.upperDone ? m.bulkLevels : [bulk.leavesDone]}
              levelFill={[m.bulkFill, ...m.bulkLevels.slice(1).map(() => NONLEAF_FILLFACTOR)]}
              pts={bulkPts}
              corr={1}
              cap={m.cap}
              tip={tip}
              scatterLabel="appended in key order"
            />
          </svg>
        </TooltipHost>
      </div>
    </VizPanel>
  );
}

/* ---------------------------------------------------------- one column */

function Column({
  x,
  w,
  accent,
  title,
  status,
  progress,
  hist,
  histTotal,
  bandTitle,
  runs,
  ff,
  levels,
  levelFill,
  pts,
  corr,
  cap,
  tip,
  scatterLabel,
}: {
  x: number;
  w: number;
  accent: string;
  title: string;
  status: string;
  progress: number;
  hist: number[];
  histTotal: number;
  bandTitle: string;
  runs?: { total: number; done: number; merging: boolean };
  ff: number;
  levels: number[];
  levelFill: number[];
  pts: { x: number; y: number }[];
  corr: number;
  cap: number;
  tip: (c: ReactNode) => Record<string, unknown>;
  scatterLabel: string;
}) {
  const innerW = w - 8;
  const bandY = 56;
  const bandH = 72;
  const lvlY = 152;
  const scatY = 246;
  const scatH = 96;
  const maxHist = Math.max(1, ...hist);

  return (
    <g transform={`translate(${x},0)`}>
      <rect x={0} y={0} width={w} height={COL_H} rx={10} fill="var(--viz-plane)" stroke="var(--viz-border)" />
      <rect x={0} y={0} width={4} height={COL_H} rx={2} fill={accent} />

      <text x={12} y={18} fill="var(--viz-ink)" fontWeight={600}>
        {title}
      </text>
      <text x={w - 12} y={18} textAnchor="end">
        {status}
      </text>

      {/* progress */}
      <rect x={12} y={26} width={innerW - 16} height={8} rx={4} fill="var(--viz-neutral)" />
      <rect x={12} y={26} width={Math.max(0, (innerW - 16) * progress)} height={8} rx={4} fill={accent} />
      <text x={12} y={48}>
        {bandTitle}
      </text>

      {/* band: runs, or the fill histogram */}
      {runs ? (
        <RunBand x={12} y={bandY} w={innerW - 16} h={bandH} runs={runs} accent={accent} tip={tip} />
      ) : (
        <g>
          <line
            className="viz-axis-line"
            x1={12}
            x2={12 + (innerW - 16)}
            y1={bandY + bandH}
            y2={bandY + bandH}
          />
          {hist.map((c, i) => {
            const bw = (innerW - 16) / 10;
            const bx = 12 + i * bw;
            const bh = histTotal > 0 ? (c / maxHist) * (bandH - 12) : 0;
            return (
              <g key={i} {...tip(
                <>
                  <strong>
                    {i * 10}–{i * 10 + 10}% full
                  </strong>
                  <br />
                  {fmtNum(c)} leaf page{c === 1 ? '' : 's'} ({fmtNum(Math.round((i * 10 + 5) * cap / 100))} entries each,
                  roughly)
                </>,
              )}>
                <rect
                  x={bx + 1.5}
                  y={bandY + bandH - bh}
                  width={bw - 3}
                  height={Math.max(c > 0 ? 2 : 0, bh)}
                  fill={fillColor((i + 0.5) / 10)}
                  stroke="var(--viz-border)"
                />
              </g>
            );
          })}
          <line
            x1={12 + ((innerW - 16) * ff) / 100}
            x2={12 + ((innerW - 16) * ff) / 100}
            y1={bandY - 2}
            y2={bandY + bandH}
            stroke="var(--viz-critical)"
            strokeDasharray="4 3"
          />
          <text x={12 + ((innerW - 16) * ff) / 100 - 4} y={bandY + 8} textAnchor="end" fill="var(--viz-critical)">
            fillfactor {ff}%
          </text>
          <text x={12} y={bandY + bandH + 12}>
            0%
          </text>
          <text x={12 + (innerW - 16) / 2} y={bandY + bandH + 12} textAnchor="middle">
            50% full
          </text>
          <text x={12 + (innerW - 16)} y={bandY + bandH + 12} textAnchor="end">
            100%
          </text>
        </g>
      )}

      {/* levels */}
      <text x={12} y={lvlY - 6}>
        Pages per level
      </text>
      {levels
        .map((count, i) => ({ count, i }))
        .reverse()
        .map(({ count, i }, row) => {
          const y = lvlY + row * 18;
          if (y > scatY - 24) return null;
          const cells = Math.min(40, count);
          const cw = (innerW - 96) / 40;
          const fill = levelFill[i] ?? 0.7;
          return (
            <g
              key={i}
              {...tip(
                <>
                  <strong>{i === 0 ? 'Leaf level' : i === levels.length - 1 ? 'Root' : `Internal level ${i}`}</strong>
                  <br />
                  {fmtNum(count)} page{count === 1 ? '' : 's'}, about {Math.round(fill * 100)}% full (
                  {fmtNum(Math.round(cap * fill))} of {fmtNum(cap)} entries)
                </>,
              )}
            >
              <text x={12} y={y + 9}>
                {i === 0 ? 'leaf' : i === levels.length - 1 ? 'root' : `L${i}`}
              </text>
              {Array.from({ length: cells }, (_, k) => (
                <rect
                  key={k}
                  x={44 + k * cw}
                  y={y}
                  width={Math.max(1.5, cw - 1.5)}
                  height={11}
                  rx={1.5}
                  fill={fillColor(fill)}
                  stroke="var(--viz-border)"
                />
              ))}
              <text x={innerW - 4} y={y + 9} textAnchor="end">
                {fmtNum(count)} {count > 40 ? '(40 shown)' : ''}
              </text>
            </g>
          );
        })}

      {/* scatter: key order vs physical block order */}
      <text x={12} y={scatY - 6}>
        Key order → block number{' '}
        <tspan fill={corr > 0.99 ? 'var(--viz-good)' : 'var(--viz-serious)'}>r = {corr.toFixed(2)}</tspan>
      </text>
      <rect x={12} y={scatY} width={innerW - 16} height={scatH} rx={4} fill="var(--viz-surface)" stroke="var(--viz-border)" />
      {pts.map((p, i) => (
        <circle
          key={i}
          cx={12 + 4 + p.x * (innerW - 24)}
          cy={scatY + scatH - 4 - p.y * (scatH - 8)}
          r={2.4}
          fill={accent}
          opacity={0.85}
        />
      ))}
      <text x={innerW - 4} y={scatY + scatH - 6} textAnchor="end" fill="var(--viz-ink-muted)">
        {scatterLabel}
      </text>
    </g>
  );
}

/* --------------------------------------------------------- the run band */

function RunBand({
  x,
  y,
  w,
  h,
  runs,
  accent,
  tip,
}: {
  x: number;
  y: number;
  w: number;
  h: number;
  runs: { total: number; done: number; merging: boolean };
  accent: string;
  tip: (c: ReactNode) => Record<string, unknown>;
}) {
  if (runs.total === 0) {
    return (
      <g>
        <rect x={x} y={y} width={w} height={h} rx={6} fill="var(--viz-surface)" stroke="var(--viz-border)" />
        <text x={x + w / 2} y={y + h / 2 + 4} textAnchor="middle">
          the whole sort fits in maintenance_work_mem — no temp tapes at all
        </text>
      </g>
    );
  }
  const shown = Math.min(28, runs.total);
  const cw = w / shown;
  return (
    <g>
      <rect x={x} y={y} width={w} height={h} rx={6} fill="var(--viz-surface)" stroke="var(--viz-border)" />
      {Array.from({ length: shown }, (_, i) => {
        const written = (i / shown) * runs.total < runs.done;
        return (
          <g
            key={i}
            {...tip(
              <>
                <strong>Sorted run {i + 1}</strong>
                <br />
                {written
                  ? 'Quicksorted in memory and written to a temp tape in base/pgsql_tmp.'
                  : 'Not produced yet — tuplesort is still filling its memory.'}
              </>,
            )}
          >
            <rect
              x={x + i * cw + 1.5}
              y={y + 10}
              width={Math.max(2, cw - 3)}
              height={h - 34}
              rx={2}
              fill={written ? accent : 'var(--viz-neutral)'}
              stroke="var(--viz-border)"
              opacity={written ? 0.9 : 1}
            />
            {runs.merging ? (
              <line
                x1={x + i * cw + cw / 2}
                y1={y + h - 24}
                x2={x + w / 2}
                y2={y + h - 6}
                stroke={accent}
                strokeWidth={0.8}
                opacity={0.6}
              />
            ) : null}
          </g>
        );
      })}
      <text x={x + 6} y={y + h - 6}>
        {fmtNum(runs.done)} / {fmtNum(runs.total)} runs{runs.total > 28 ? ' (28 shown)' : ''}
      </text>
      {runs.merging ? (
        <text x={x + w - 6} y={y + h - 6} textAnchor="end" fill={accent}>
          merging → one sorted stream
        </text>
      ) : null}
    </g>
  );
}
