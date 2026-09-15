import { useDeferredValue, useMemo, useRef, useState } from 'react';
import { VizPanel, Segmented, Choice, Slider, Button, Legend, Stats, Note, makeRng, fmtNum, useSize } from './Viz';

/**
 * Zone-map pruning under different physical orders, and how clustering decays.
 *
 * Model (labelled on the page):
 * - A table of rows with two integer columns A and B on a 64 × 64 domain. 4,096 base rows, uniformly random
 *   (makeRng). "Distinct A values" spreads A over the domain (0, 16, 32, 48 for 4 values), as Delta's
 *   range-partition ids spread a column over its range before interleaving.
 * - The table is stored as parts (a ClickHouse part, a Parquet/Delta file, a Snowflake load). Each part is cut
 *   into blocks of a fixed row count; a block never spans parts. Every block keeps a min/max zone map for
 *   A and for B, so its summary is a bounding box on the (A, B) plane.
 * - A range predicate A ∈ [a0, a1] AND B ∈ [b0, b1] reads a block iff its box intersects the predicate
 *   rectangle — the test Parquet row-group statistics, Delta file stats, Redshift block metadata, Snowflake
 *   micro-partition metadata and a ClickHouse minmax skip index all perform.
 * - Orders: arrival (no key), sort by A, sort by B, compound (A, B), Morton (Z-order, A bit above B bit at each
 *   level) and Hilbert (the standard xy2d construction). Ties keep arrival order.
 * - Inserts append a 512-row part, either in arrival order or sorted by the current key. "Merge inserted parts"
 *   merges every inserted part into one sorted part; "Recluster" rewrites the whole table as one sorted part.
 * - Clustering depth follows Snowflake's definition in spirit, not its implementation: a block's depth is the
 *   largest number of blocks whose boxes cover any one cell inside its own box; overlaps counts the other
 *   blocks whose boxes intersect it. Histogram buckets follow SYSTEM$CLUSTERING_INFORMATION: 1..16, then 32, 64, ….
 * - Curve clusters (Moon, Jagadish, Faloutsos, Saltz 2001): maximal runs of consecutive curve positions that
 *   stay inside the predicate, counted over cells that can hold data.
 */

export const GRID = 64;
export const BASE_ROWS = 4096;
export const BATCH_ROWS = 512;
export const MAX_BATCHES = 12;
const TOTAL_ROWS = BASE_ROWS + BATCH_ROWS * MAX_BATCHES;

export type Order = 'arrival' | 'a' | 'b' | 'ab' | 'z' | 'hilbert';
export type Rect = { a0: number; a1: number; b0: number; b1: number };
export type InsertMode = 'arrival' | 'sorted';
export type Run = { kind: 'base' | 'batch' | 'merged'; ids: number[] };
export type Block = {
  index: number;
  run: number;
  start: number;
  rows: number;
  minA: number;
  maxA: number;
  minB: number;
  maxB: number;
};

/* ------------------------------------------------------------------ data */

const U_A = new Float64Array(TOTAL_ROWS);
const B_VAL = new Uint8Array(TOTAL_ROWS);
{
  const rng = makeRng(20260915);
  for (let i = 0; i < BASE_ROWS; i++) {
    U_A[i] = rng();
    B_VAL[i] = Math.floor(rng() * GRID);
  }
  for (let k = 0; k < MAX_BATCHES; k++) {
    const r = makeRng(9001 + k * 7919);
    for (let j = 0; j < BATCH_ROWS; j++) {
      const i = BASE_ROWS + k * BATCH_ROWS + j;
      U_A[i] = r();
      B_VAL[i] = Math.floor(r() * GRID);
    }
  }
}

export const aOf = (id: number, card: number) => Math.min(card - 1, Math.floor(U_A[id] * card)) * (GRID / card);
export const bOf = (id: number) => B_VAL[id];

/* ---------------------------------------------------------------- curves */

/** Morton code: interleave the 6 bits of A and B, A's bit above B's at every level. */
export function morton(a: number, b: number) {
  let z = 0;
  for (let i = 5; i >= 0; i--) z = z * 4 + ((a >> i) & 1) * 2 + ((b >> i) & 1);
  return z;
}

/** Hilbert index of (x, y) on an n × n grid (n a power of two), the classic xy2d construction. */
export function hilbert(x: number, y: number, n = GRID) {
  let d = 0;
  for (let s = n >> 1; s > 0; s >>= 1) {
    const rx = (x & s) > 0 ? 1 : 0;
    const ry = (y & s) > 0 ? 1 : 0;
    d += s * s * ((3 * rx) ^ ry);
    if (ry === 0) {
      if (rx === 1) {
        x = n - 1 - x;
        y = n - 1 - y;
      }
      const t = x;
      x = y;
      y = t;
    }
  }
  return d;
}

/** Position of a cell along the order's curve; null when the order is not a curve over cells. */
export function cellRank(order: Order, a: number, b: number): number | null {
  if (order === 'ab') return a * GRID + b;
  if (order === 'z') return morton(a, b);
  if (order === 'hilbert') return hilbert(a, b);
  return null;
}

function keyOf(order: Order, id: number, card: number) {
  const a = aOf(id, card);
  const b = bOf(id);
  switch (order) {
    case 'arrival':
      return 0;
    case 'a':
      return a;
    case 'b':
      return b;
    case 'ab':
      return a * GRID + b;
    case 'z':
      return morton(a, b);
    case 'hilbert':
      return hilbert(a, b);
  }
}

export function sortIds(ids: number[], order: Order, card: number) {
  const keyed = ids.map((id) => [keyOf(order, id, card), id] as const);
  keyed.sort((x, y) => x[0] - y[0] || x[1] - y[1]);
  return keyed.map((k) => k[1]);
}

/* ---------------------------------------------------------------- layout */

export type TableState = { order: Order; card: number; batches: number; runs: Run[] };

const baseIds = () => Array.from({ length: BASE_ROWS }, (_, i) => i);
const batchIds = (k: number) => Array.from({ length: BATCH_ROWS }, (_, j) => BASE_ROWS + k * BATCH_ROWS + j);

export function freshTable(order: Order, card: number): TableState {
  return { order, card, batches: 0, runs: [{ kind: 'base', ids: sortIds(baseIds(), order, card) }] };
}

export function insertBatch(t: TableState, mode: InsertMode): TableState {
  if (t.batches >= MAX_BATCHES) return t;
  const ids = batchIds(t.batches);
  const run: Run = { kind: 'batch', ids: mode === 'sorted' ? sortIds(ids, t.order, t.card) : ids };
  return { ...t, batches: t.batches + 1, runs: [...t.runs, run] };
}

export function mergeInserted(t: TableState): TableState {
  const inserted = t.runs.filter((r) => r.kind !== 'base');
  if (inserted.length === 0) return t;
  const base = t.runs.filter((r) => r.kind === 'base');
  const ids = inserted.flatMap((r) => r.ids);
  return { ...t, runs: [...base, { kind: 'merged', ids: sortIds(ids, t.order, t.card) }] };
}

export function recluster(t: TableState): TableState {
  const ids = t.runs.flatMap((r) => r.ids);
  return { ...t, runs: [{ kind: 'base', ids: sortIds(ids, t.order, t.card) }] };
}

/** Rewrite the same rows in a new order (like changing the key and rewriting every file). */
export function reorder(t: TableState, order: Order): TableState {
  return recluster({ ...t, order });
}

export function buildBlocks(t: TableState, blockRows: number) {
  const blocks: Block[] = [];
  const physical: number[] = [];
  t.runs.forEach((run, ri) => {
    for (let s = 0; s < run.ids.length; s += blockRows) {
      const e = Math.min(run.ids.length, s + blockRows);
      let minA = GRID, maxA = -1, minB = GRID, maxB = -1;
      for (let i = s; i < e; i++) {
        const id = run.ids[i];
        const a = aOf(id, t.card);
        const b = bOf(id);
        if (a < minA) minA = a;
        if (a > maxA) maxA = a;
        if (b < minB) minB = b;
        if (b > maxB) maxB = b;
      }
      blocks.push({ index: blocks.length, run: ri, start: physical.length, rows: e - s, minA, maxA, minB, maxB });
      for (let i = s; i < e; i++) physical.push(run.ids[i]);
    }
  });
  return { blocks, physical };
}

/* --------------------------------------------------------------- pruning */

export const intersects = (bl: Block, r: Rect) => bl.minA <= r.a1 && bl.maxA >= r.a0 && bl.minB <= r.b1 && bl.maxB >= r.b0;

export function prune(t: TableState, blocks: Block[], physical: number[], r: Rect, blockRows: number) {
  const read: boolean[] = [];
  const matches: number[] = [];
  let rowsRead = 0;
  let matching = 0;
  let wasted = 0;
  let ranges = 0;
  for (const bl of blocks) {
    const hit = intersects(bl, r);
    read.push(hit);
    let m = 0;
    if (hit) {
      rowsRead += bl.rows;
      for (let i = bl.start; i < bl.start + bl.rows; i++) {
        const id = physical[i];
        const a = aOf(id, t.card);
        const b = bOf(id);
        if (a >= r.a0 && a <= r.a1 && b >= r.b0 && b <= r.b1) m++;
      }
      if (m === 0) wasted++;
      const prev = blocks[bl.index - 1];
      if (!prev || !read[bl.index - 1] || prev.run !== bl.run) ranges++;
    }
    matches.push(m);
    matching += m;
  }
  const blocksRead = read.filter(Boolean).length;
  return {
    read,
    matches,
    blocksRead,
    pruned: blocks.length - blocksRead,
    prunedFraction: blocks.length ? (blocks.length - blocksRead) / blocks.length : 0,
    rowsRead,
    matching,
    wasted,
    ranges,
    lowerBound: Math.ceil(matching / blockRows),
  };
}

/* ------------------------------------------------------ clustering depth */

export function depthBucket(d: number) {
  if (d <= 16) return d;
  let b = 32;
  while (b < d) b *= 2;
  return b;
}

export function clusteringDepth(blocks: Block[]) {
  const n = blocks.length;
  const diff = new Int32Array((GRID + 1) * (GRID + 1));
  const at = (a: number, b: number) => a * (GRID + 1) + b;
  for (const bl of blocks) {
    diff[at(bl.minA, bl.minB)] += 1;
    diff[at(bl.maxA + 1, bl.minB)] -= 1;
    diff[at(bl.minA, bl.maxB + 1)] -= 1;
    diff[at(bl.maxA + 1, bl.maxB + 1)] += 1;
  }
  const cover = new Int32Array(GRID * GRID);
  for (let a = 0; a < GRID; a++) {
    for (let b = 0; b < GRID; b++) {
      const up = a > 0 ? cover[(a - 1) * GRID + b] : 0;
      const left = b > 0 ? cover[a * GRID + b - 1] : 0;
      const diag = a > 0 && b > 0 ? cover[(a - 1) * GRID + b - 1] : 0;
      cover[a * GRID + b] = diff[at(a, b)] + up + left - diag;
    }
  }
  // per-row sparse max would be faster; the grid is 64 × 64 and blocks are capped, so a direct scan stays cheap.
  const depth: number[] = blocks.map((bl) => {
    let m = 0;
    for (let a = bl.minA; a <= bl.maxA; a++) for (let b = bl.minB; b <= bl.maxB; b++) m = Math.max(m, cover[a * GRID + b]);
    return m;
  });
  let overlapsTotal = 0;
  for (let i = 0; i < n; i++) {
    const x = blocks[i];
    for (let j = i + 1; j < n; j++) {
      const y = blocks[j];
      if (x.minA <= y.maxA && x.maxA >= y.minA && x.minB <= y.maxB && x.maxB >= y.minB) overlapsTotal += 2;
    }
  }
  const hist = new Map<number, number>();
  for (const d of depth) hist.set(depthBucket(d), (hist.get(depthBucket(d)) ?? 0) + 1);
  return {
    depth,
    averageDepth: n ? depth.reduce((s, d) => s + d, 0) / n : 0,
    averageOverlaps: n ? overlapsTotal / n : 0,
    hist,
    maxCover: cover.reduce((m, c) => Math.max(m, c), 0),
  };
}

/** Moon et al.'s clusters: runs of consecutive curve positions inside the rectangle, over cells that can hold data. */
export function curveClusters(order: Order, card: number, r: Rect): number | null {
  if (cellRank(order, 0, 0) === null) return null;
  const cells: { rank: number; inside: boolean }[] = [];
  const step = GRID / card;
  for (let a = 0; a < GRID; a += step) {
    for (let b = 0; b < GRID; b++) {
      cells.push({ rank: cellRank(order, a, b) as number, inside: a >= r.a0 && a <= r.a1 && b >= r.b0 && b <= r.b1 });
    }
  }
  cells.sort((x, y) => x.rank - y.rank);
  let clusters = 0;
  for (let i = 0; i < cells.length; i++) if (cells[i].inside && (i === 0 || !cells[i - 1].inside)) clusters++;
  return clusters;
}

/** The same rows rewritten in every order, for the comparison table. */
export function compareOrders(t: TableState, blockRows: number, r: Rect) {
  return (['arrival', 'a', 'b', 'ab', 'z', 'hilbert'] as Order[]).map((order) => {
    const tt = reorder(t, order);
    const b = buildBlocks(tt, blockRows);
    const p = prune(tt, b.blocks, b.physical, r, blockRows);
    return { order, blocks: b.blocks.length, blocksRead: p.blocksRead, rowsRead: p.rowsRead, wasted: p.wasted, clusters: curveClusters(order, tt.card, r) };
  });
}

/* ------------------------------------------------------------------- UI */

export const PRESETS: { value: string; label: string; rect: Rect | null }[] = [
  { value: 'aonly', label: 'A in [16, 23]', rect: { a0: 16, a1: 23, b0: 0, b1: 63 } },
  { value: 'bonly', label: 'B in [40, 47]', rect: { a0: 0, a1: 63, b0: 40, b1: 47 } },
  { value: 'box', label: 'A in [16, 31] and B in [40, 55]', rect: { a0: 16, a1: 31, b0: 40, b1: 55 } },
  { value: 'small', label: 'A in [32, 35] and B in [12, 15]', rect: { a0: 32, a1: 35, b0: 12, b1: 15 } },
  { value: 'custom', label: 'Custom (drag on the plane)', rect: null },
];

export const ORDER_LABEL: Record<Order, string> = {
  arrival: 'Arrival order (no key)',
  a: 'ORDER BY A',
  b: 'ORDER BY B',
  ab: 'ORDER BY (A, B)',
  z: 'Z-order (Morton) on A, B',
  hilbert: 'Hilbert curve on A, B',
};

const clampCell = (v: number) => Math.max(0, Math.min(GRID - 1, Math.round(v)));

export default function ClusteringCurveSkipLab() {
  const [card, setCard] = useState(64);
  const [blockExp, setBlockExp] = useState(7);
  const [preset, setPreset] = useState('aonly');
  const [rect, setRect] = useState<Rect>(PRESETS[0].rect as Rect);
  const [insertMode, setInsertMode] = useState<InsertMode>('arrival');
  const [table, setTable] = useState<TableState>(() => freshTable('hilbert', 64));
  const [hover, setHover] = useState<number | null>(null);
  const drag = useRef<{ a: number; b: number } | null>(null);
  const svgRef = useRef<SVGSVGElement>(null);
  const [sizeRef, figWidth] = useSize(720);

  const blockRows = 1 << blockExp;
  const { blocks, physical } = useMemo(() => buildBlocks(table, blockRows), [table, blockRows]);
  const pr = useMemo(() => prune(table, blocks, physical, rect, blockRows), [table, blocks, physical, rect, blockRows]);
  const cd = useMemo(() => clusteringDepth(blocks), [blocks]);
  const clusters = useMemo(() => curveClusters(table.order, table.card, rect), [table.order, table.card, rect]);
  // The comparison table re-sorts the table six times (~10 ms at the largest size), so it trails a drag instead of blocking it.
  const deferredRect = useDeferredValue(rect);
  const comparison = useMemo(() => compareOrders(table, blockRows, deferredRect), [table, blockRows, deferredRect]);
  const totalRows = physical.length;
  const hovered = hover !== null && hover < blocks.length ? blocks[hover] : null;

  const setOrder = (o: Order) => {
    setHover(null);
    setTable((t) => reorder(t, o));
  };
  const choosePreset = (v: string) => {
    setPreset(v);
    const p = PRESETS.find((x) => x.value === v);
    if (p?.rect) setRect(p.rect);
  };
  const editRect = (patch: Partial<Rect>) => {
    setPreset('custom');
    setRect((r) => {
      const n = { ...r, ...patch };
      return { a0: Math.min(n.a0, n.a1), a1: Math.max(n.a0, n.a1), b0: Math.min(n.b0, n.b1), b1: Math.max(n.b0, n.b1) };
    });
  };

  /* geometry */
  // Wide: block strip and histogram to the right of the plane. Narrow (phones): stacked under it.
  const narrow = figWidth < 480;
  const W = narrow ? 404 : 700;
  const PX = 44;
  const PY = 14;
  const S = 352;
  const cell = S / GRID;
  const panelTop = narrow ? PY + S + 46 : PY;
  const X = (a: number) => PX + a * cell;
  const Y = (b: number) => PY + S - (b + 1) * cell;
  const RX = narrow ? 12 : 432;
  const RW = W - RX - 8;

  const cellFromEvent = (e: React.PointerEvent) => {
    const svg = svgRef.current;
    if (!svg) return null;
    const box = svg.getBoundingClientRect();
    const sx = ((e.clientX - box.left) / box.width) * W;
    const sy = ((e.clientY - box.top) / box.height) * H;
    return { a: clampCell((sx - PX) / cell - 0.5), b: clampCell((PY + S - sy) / cell - 0.5) };
  };

  // Block strip layout: fit every block (plus one gap slot between parts) into a fixed box.
  const stripTop = panelTop + 18;
  const stripH = 130;
  const slots = blocks.length + table.runs.length - 1;
  let cs = 22;
  while (cs > 3 && Math.ceil(slots / Math.floor(RW / cs)) * cs > stripH) cs -= 1;
  const perRow = Math.floor(RW / cs);
  const slotOf: number[] = [];
  {
    let s = 0;
    blocks.forEach((bl, i) => {
      if (i > 0 && blocks[i - 1].run !== bl.run) s += 1;
      slotOf.push(s);
      s += 1;
    });
  }

  // Depth histogram buckets.
  const maxBucket = Math.max(16, depthBucket(Math.max(1, ...cd.depth)));
  const buckets: number[] = [];
  for (let d = 1; d <= 16; d++) buckets.push(d);
  for (let b = 32; b <= maxBucket; b *= 2) buckets.push(b);
  const histTop = stripTop + stripH + 66;
  const histH = narrow ? 120 : PY + S + 40 - histTop - 26;
  const H = narrow ? histTop + histH + 26 : PY + S + 40;
  const bw = RW / buckets.length;
  const maxCount = Math.max(1, ...buckets.map((b) => cd.hist.get(b) ?? 0));
  let lastCount = { x: -Infinity, y: -Infinity };

  // Read boxes share a bounded amount of ink, so many overlapping boxes shade the plane instead of flooding it.
  const readFillOpacity = Math.min(0.1, 0.4 / Math.max(1, pr.blocksRead));
  const blockFill = (i: number) => (!pr.read[i] ? 'none' : pr.matches[i] > 0 ? 'var(--viz-1)' : 'var(--viz-serious)');
  const blockStroke = (i: number) => (!pr.read[i] ? 'var(--viz-ink-muted)' : pr.matches[i] > 0 ? 'var(--viz-1)' : 'var(--viz-serious)');
  const tipText = (bl: Block) =>
    `Block ${bl.index} (part ${bl.run + 1}, ${bl.rows} rows): A ${bl.minA}–${bl.maxA}, B ${bl.minB}–${bl.maxB}; ${pr.read[bl.index] ? `read, ${pr.matches[bl.index]} matching rows` : 'pruned'}; depth ${cd.depth[bl.index]}`;

  const inserted = table.runs.filter((r) => r.kind !== 'base').length;
  const pendingBatches = table.runs.filter((r) => r.kind === 'batch').length;
  const selectivity = totalRows ? pr.matching / totalRows : 0;

  return (
    <VizPanel
      title="Which blocks survive a range predicate"
      subtitle="Each block keeps a min and max for A and for B, so its zone map is a box on the (A, B) plane. A block is read when its box touches the predicate. Change the physical order, the block size and the predicate, then stream inserts and watch clustering depth climb until a recluster restores it."
      controls={
        <>
          <Choice label="Physical order" value={table.order} onChange={(v) => setOrder(v as Order)} options={(Object.keys(ORDER_LABEL) as Order[]).map((o) => ({ value: o, label: ORDER_LABEL[o] }))} />
          <Segmented
            label="Distinct A values"
            value={String(card)}
            onChange={(v) => {
              const c = Number(v);
              setCard(c);
              setHover(null);
              setTable((t) => freshTable(t.order, c));
            }}
            options={[
              { value: '64', label: '64' },
              { value: '16', label: '16' },
              { value: '4', label: '4' },
            ]}
          />
          <Slider label="Rows per block" min={4} max={9} value={blockExp} onChange={(v) => { setBlockExp(v); setHover(null); }} format={(v) => fmtNum(1 << v)} />
          <Choice label="Predicate" value={preset} onChange={choosePreset} options={PRESETS.map((p) => ({ value: p.value, label: p.label }))} />
          <Segmented
            label="Inserted parts are written"
            value={insertMode}
            onChange={setInsertMode}
            options={[
              { value: 'arrival', label: 'in arrival order' },
              { value: 'sorted', label: 'sorted by the key' },
            ]}
          />
        </>
      }
      legend={
        <Legend
          items={[
            { label: 'Block read, holds matching rows', color: 'var(--viz-1)' },
            { label: 'Block read, no matching rows', color: 'var(--viz-serious)' },
            { label: 'Block pruned by its min/max', color: 'var(--viz-ink-muted)', shape: 'line' },
            { label: 'Range predicate', color: 'var(--viz-7)', shape: 'line' },
            { label: 'Blocks at each depth', color: 'var(--viz-3)' },
          ]}
        />
      }
      stats={
        <Stats
          items={[
            { label: 'Blocks read', value: `${fmtNum(pr.blocksRead)} of ${fmtNum(blocks.length)}`, hint: `Lower bound for this predicate at this block size: ${fmtNum(pr.lowerBound)} (matching rows ÷ rows per block).` },
            { label: 'Pruned', value: `${fmtNum(pr.prunedFraction * 100, 1)}%` },
            { label: 'Rows read / matching', value: `${fmtNum(pr.rowsRead)} / ${fmtNum(pr.matching)}` },
            { label: 'Read, no match', value: fmtNum(pr.wasted), hint: 'Blocks whose box touches the predicate but that hold no matching row: the zone map could not rule them out.' },
            { label: 'Read ranges', value: fmtNum(pr.ranges), hint: 'Runs of consecutive surviving blocks within a part, like ClickHouse’s “marks to read from N ranges”.' },
            { label: 'Average depth', value: cd.averageDepth.toFixed(2), hint: 'Model of Snowflake’s average_depth over the (A, B) boxes: each block’s depth is the most boxes covering any one cell inside it.' },
          ]}
        />
      }
      note={
        <Note>
          <strong>
            {ORDER_LABEL[table.order]}, {fmtNum(blockRows)} rows per block: {fmtNum(pr.blocksRead)} of {fmtNum(blocks.length)} blocks read for {fmtNum(pr.matching)} matching rows ({fmtNum(selectivity * 100, 1)}% of the table).
          </strong>{' '}
          {table.order === 'arrival'
            ? pr.pruned === 0
              ? 'With no key, every block holds values from across the whole domain, so every box touches the predicate and nothing can be pruned. '
              : `With no key, almost every block holds values from across the whole domain; only ${fmtNum(pr.pruned)} box${pr.pruned === 1 ? ' misses' : 'es miss'} the predicate. `
            : pr.wasted > 0
              ? pr.wasted === 1
                ? '1 of the blocks read holds no matching row at all: its box stretches across the predicate without any row inside it. '
                : `${fmtNum(pr.wasted)} of the blocks read hold no matching row at all: their boxes stretch across the predicate without any row inside it. `
              : pr.blocksRead === 0
                ? 'No box touches the predicate, so no block is read. '
                : 'Every block read holds at least one matching row. '}
          {table.order !== 'arrival' && clusters !== null ? clusters === 0 ? 'No cell that can hold data lies inside the predicate. ' : clusters === 1 ? 'The curve passes through the predicate in one stretch. ' : `The curve passes through the predicate in ${fmtNum(clusters)} separate stretches. ` : ''}
          {inserted > 0
            ? pendingBatches > 0
              ? `${inserted} inserted part${inserted === 1 ? ' overlaps' : 's overlap'} the original data (dashed outlines), raising average depth to ${cd.averageDepth.toFixed(2)}. Merge or recluster to bring it back down.`
              : `The merged part still overlaps the original data (dashed outlines), so average depth is ${cd.averageDepth.toFixed(2)}. Recluster to rewrite both as one sorted part.`
            : table.order !== 'arrival'
              ? `Average depth is ${cd.averageDepth.toFixed(2)} with ${fmtNum(cd.averageOverlaps, 1)} overlapping boxes per block.`
              : ''}
        </Note>
      }
      table={
        <table className="viz-table">
          <caption>The same rows rewritten in each order, inserted rows included, at this block size and predicate</caption>
          <thead>
            <tr>
              <th>Physical order</th>
              <th>Blocks read</th>
              <th>Rows read</th>
              <th>Read, no match</th>
              <th>Curve clusters in predicate</th>
            </tr>
          </thead>
          <tbody>
            {comparison.map((row) => (
              <tr key={row.order}>
                <td>{ORDER_LABEL[row.order]}</td>
                <td>
                  {fmtNum(row.blocksRead)} of {fmtNum(row.blocks)}
                </td>
                <td>{fmtNum(row.rowsRead)}</td>
                <td>{fmtNum(row.wasted)}</td>
                <td>{row.clusters === null ? '—' : fmtNum(row.clusters)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      }
    >
      <div className="viz-controls">
        <Button primary onClick={() => setTable((t) => insertBatch(t, insertMode))} disabled={table.batches >= MAX_BATCHES}>
          Insert {fmtNum(BATCH_ROWS)} rows
        </Button>
        <Button onClick={() => setTable(mergeInserted)} disabled={!table.runs.some((r) => r.kind === 'batch')}>
          Merge inserted parts
        </Button>
        <Button onClick={() => setTable(recluster)} disabled={table.runs.length === 1}>
          Recluster table
        </Button>
        <Button onClick={() => { setHover(null); setTable((t) => freshTable(t.order, t.card)); }} disabled={table.batches === 0}>
          Reset inserts
        </Button>
      </div>
      <div ref={sizeRef}>
      <svg
        ref={svgRef}
        viewBox={`0 0 ${W} ${H}`}
        width={W}
        height={H}
        role="img"
        aria-label={`${ORDER_LABEL[table.order]}: ${pr.blocksRead} of ${blocks.length} blocks read for predicate A ${rect.a0} to ${rect.a1}, B ${rect.b0} to ${rect.b1}`}
      >
        {/* plane */}
        <rect x={PX} y={PY} width={S} height={S} fill="var(--viz-plane)" stroke="var(--viz-axis)" />
        {card < GRID
          ? Array.from({ length: card }, (_, i) => (
              <rect key={i} x={X(i * (GRID / card))} y={PY} width={cell} height={S} fill="var(--viz-neutral)" stroke="var(--viz-grid)" strokeWidth={0.5} />
            ))
          : null}
        {[0, 16, 32, 48, 63].map((v) => (
          <g key={v}>
            <text x={X(v) + cell / 2} y={PY + S + 13} fontSize={10} textAnchor="middle" fill="var(--viz-ink-2)">
              {v}
            </text>
            <text x={PX - 5} y={Y(v) + cell / 2 + 3} fontSize={10} textAnchor="end" fill="var(--viz-ink-2)">
              {v}
            </text>
          </g>
        ))}
        <text x={PX + S / 2} y={PY + S + 30} fontSize={12} textAnchor="middle" fill="var(--viz-ink)">
          A →
        </text>
        <text x={12} y={PY + S / 2} fontSize={12} textAnchor="middle" fill="var(--viz-ink)" transform={`rotate(-90 12 ${PY + S / 2})`}>
          B →
        </text>

        {blocks.map((bl, i) =>
          pr.read[i] ? null : (
            <rect key={`p${i}`} x={X(bl.minA) + 0.5} y={Y(bl.maxB) + 0.5} width={(bl.maxA - bl.minA + 1) * cell - 1} height={(bl.maxB - bl.minB + 1) * cell - 1} fill="none" stroke="var(--viz-ink-muted)" strokeOpacity={0.35} strokeWidth={0.8} strokeDasharray={table.runs[bl.run].kind === 'base' ? undefined : '4 2'} />
          ),
        )}
        {blocks.map((bl, i) =>
          pr.read[i] ? (
            <rect key={`r${i}`} x={X(bl.minA) + 0.5} y={Y(bl.maxB) + 0.5} width={(bl.maxA - bl.minA + 1) * cell - 1} height={(bl.maxB - bl.minB + 1) * cell - 1} fill={blockFill(i)} fillOpacity={readFillOpacity} stroke={blockStroke(i)} strokeOpacity={0.85} strokeWidth={1} strokeDasharray={table.runs[bl.run].kind === 'base' ? undefined : '4 2'} />
          ) : null,
        )}
        <rect x={X(rect.a0)} y={Y(rect.b1)} width={(rect.a1 - rect.a0 + 1) * cell} height={(rect.b1 - rect.b0 + 1) * cell} fill="var(--viz-7)" fillOpacity={0.12} stroke="var(--viz-7)" strokeWidth={2.5} />
        {hovered ? (
          <rect x={X(hovered.minA)} y={Y(hovered.maxB)} width={(hovered.maxA - hovered.minA + 1) * cell} height={(hovered.maxB - hovered.minB + 1) * cell} fill="none" stroke="var(--viz-ink)" strokeWidth={2.5} strokeDasharray="5 3" />
        ) : null}
        <rect
          x={PX}
          y={PY}
          width={S}
          height={S}
          fill="transparent"
          style={{ cursor: 'crosshair', touchAction: 'none' }}
          onPointerDown={(e) => {
            const c = cellFromEvent(e);
            if (!c) return;
            (e.currentTarget as Element).setPointerCapture?.(e.pointerId);
            drag.current = c;
            setPreset('custom');
            setRect({ a0: c.a, a1: c.a, b0: c.b, b1: c.b });
          }}
          onPointerMove={(e) => {
            const c = cellFromEvent(e);
            if (!c) return;
            if (drag.current) {
              const s = drag.current;
              setRect({ a0: Math.min(s.a, c.a), a1: Math.max(s.a, c.a), b0: Math.min(s.b, c.b), b1: Math.max(s.b, c.b) });
            } else {
              let best: number | null = null;
              let area = Infinity;
              for (const bl of blocks) {
                if (c.a >= bl.minA && c.a <= bl.maxA && c.b >= bl.minB && c.b <= bl.maxB) {
                  const ar = (bl.maxA - bl.minA + 1) * (bl.maxB - bl.minB + 1);
                  if (ar < area) {
                    area = ar;
                    best = bl.index;
                  }
                }
              }
              setHover(best);
            }
          }}
          onPointerUp={() => {
            drag.current = null;
          }}
          onPointerLeave={() => {
            if (!drag.current) setHover(null);
          }}
        />

        {/* block strip */}
        <text x={RX} y={panelTop + 8} fontSize={12} fill="var(--viz-ink)">
          Blocks in physical order ({table.runs.length} part{table.runs.length === 1 ? '' : 's'})
        </text>
        {blocks.map((bl, i) => {
          const s = slotOf[i];
          const x = RX + (s % perRow) * cs;
          const y = stripTop + Math.floor(s / perRow) * cs;
          return (
            <rect
              key={`s${i}`}
              x={x + 0.5}
              y={y + 0.5}
              width={cs - 1.5}
              height={cs - 1.5}
              rx={1}
              fill={pr.read[i] ? blockFill(i) : 'var(--viz-surface)'}
              fillOpacity={pr.read[i] ? 0.85 : 1}
              stroke={hover === i ? 'var(--viz-ink)' : pr.read[i] ? 'none' : 'var(--viz-ink-muted)'}
              strokeWidth={hover === i ? 2 : 0.6}
              onMouseEnter={() => setHover(i)}
              onMouseLeave={() => setHover(null)}
            >
              <title>{tipText(bl)}</title>
            </rect>
          );
        })}
        <text x={RX} y={stripTop + stripH + 16} fontSize={11} fill="var(--viz-ink-2)">
          {hovered ? `Block ${hovered.index} (part ${hovered.run + 1}): A ${hovered.minA}–${hovered.maxA}, B ${hovered.minB}–${hovered.maxB}` : 'Hover a block or a box to see its min/max.'}
        </text>
        <text x={RX} y={stripTop + stripH + 31} fontSize={11} fill="var(--viz-ink-2)">
          {hovered ? `${pr.read[hovered.index] ? `read, ${pr.matches[hovered.index]} matching rows` : 'pruned'} · depth ${cd.depth[hovered.index]}` : ''}
        </text>

        {/* depth histogram */}
        <text x={RX} y={histTop - 10} fontSize={12} fill="var(--viz-ink)">
          Clustering depth histogram (blocks per depth)
        </text>
        <line x1={RX} x2={RX + RW} y1={histTop + histH} y2={histTop + histH} stroke="var(--viz-axis)" />
        {buckets.map((b, i) => {
          const count = cd.hist.get(b) ?? 0;
          const h = (count / maxCount) * (histH - 12);
          const x = RX + i * bw;
          const cx = x + bw / 2;
          const cy = histTop + histH - h - 2;
          // Narrow buckets: skip a count label that would collide with the previous one (the bar's title still carries it).
          const showCount = count > 0 && (cx - lastCount.x >= 16 || Math.abs(cy - lastCount.y) >= 10);
          if (showCount) lastCount = { x: cx, y: cy };
          const last = buckets.length - 1;
          const showTick = b === 1 || b === 4 || b === 8 || b === 12 || (last === 15 ? b === 16 : i >= 15 && (last - i) % 2 === 0);
          return (
            <g key={b}>
              {count > 0 ? (
                <rect x={x + 1} y={histTop + histH - h} width={bw - 2} height={h} fill="var(--viz-3)">
                  <title>{`Depth ${i >= 16 ? `${buckets[i - 1] + 1}–${b}` : b}: ${count} block${count === 1 ? '' : 's'}`}</title>
                </rect>
              ) : null}
              {showCount ? (
                <text x={cx} y={cy} fontSize={8} textAnchor="middle" fill="var(--viz-ink-2)">
                  {count}
                </text>
              ) : null}
              {showTick ? (
                <text x={x + bw / 2} y={histTop + histH + 11} fontSize={8} textAnchor="middle" fill="var(--viz-ink-2)">
                  {b}
                </text>
              ) : null}
            </g>
          );
        })}
      </svg>
      </div>

      <div className="viz-controls">
        <Slider label="A from" min={0} max={63} value={rect.a0} onChange={(v) => editRect({ a0: v })} />
        <Slider label="A to" min={0} max={63} value={rect.a1} onChange={(v) => editRect({ a1: v })} />
        <Slider label="B from" min={0} max={63} value={rect.b0} onChange={(v) => editRect({ b0: v })} />
        <Slider label="B to" min={0} max={63} value={rect.b1} onChange={(v) => editRect({ b1: v })} />
      </div>
    </VizPanel>
  );
}
