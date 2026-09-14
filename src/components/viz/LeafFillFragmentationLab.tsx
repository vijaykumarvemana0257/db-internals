import { useMemo, useState } from 'react';
import {
  VizPanel,
  Segmented,
  Slider,
  Check,
  Button,
  Legend,
  Stats,
  Note,
  TooltipHost,
  useTip,
  makeRng,
  fmtNum,
  useSize,
} from './Viz';

/**
 * What the shape of the key stream does to a B+tree's leaf level.
 *
 * The model is a leaf level only (the internal levels change nothing about the
 * numbers this page is about) and it runs PostgreSQL nbtree's actual rules:
 *
 *  - a leaf overflows at capacity and splits; the new page is allocated at the
 *    END of the file (a growing index has no free pages to hand back), so the
 *    sibling chain and the block order diverge from the first non-rightmost split
 *  - _bt_findsplitloc applies the index's fillfactor ONLY to the rightmost page of
 *    the level — left keeps floor(capacity * fillfactor/100), the remainder plus
 *    the new tuple go right. Every other split is at the midpoint, which is why a
 *    fillfactor set on a randomly-inserted index does nothing.
 *  - pgstatindex's leaf_fragmentation is reproduced exactly: the share of leaf
 *    pages whose btpo_next sits at a LOWER block number than the page itself.
 *  - avg_leaf_density is entries / (pages * capacity).
 *
 * Capacity is scaled down by SCALE so a legible strip of pages fits on screen;
 * the real per-page capacity for an 8 KB page is computed from the real entry
 * width and shown in the note and the table, and the "pages per million rows"
 * column extrapolates the simulated density back onto it.
 */

const BLCKSZ = 8192;
/** 8192 - PageHeaderData (24) - BTPageOpaqueData (16). */
const LEAF_USABLE = BLCKSZ - 24 - 16;
const SCALE = 16;
const MAX_LEAVES = 80;
const BATCH = 250;
/** Rows the index is seeded with, and the floor a control change replays to. */
const SEED_ROWS = 750;
const RECENT = 200;
/** Leaf budget the comparison table fills, to compare streams at equal size. */
const TABLE_LEAVES = 64;

/** Deterministic noise, drawn once. Never Math.random: SSR and hydration must agree. */
const NOISE: number[] = (() => {
  const rng = makeRng(20260913);
  return Array.from({ length: 4096 }, () => rng());
})();

type StreamId = 'serial' | 'uuid4' | 'uuid7' | 'tsrand';

type Stream = {
  id: StreamId;
  label: string;
  /** IndexTupleData header + key, MAXALIGNed, plus the 4-byte line pointer. */
  entryBytes: number;
  decl: string;
};

const STREAMS: Stream[] = [
  { id: 'serial', label: 'bigserial', entryBytes: 20, decl: 'bigint, strictly ascending' },
  { id: 'uuid4', label: 'UUIDv4', entryBytes: 28, decl: 'uuid, 122 random bits' },
  { id: 'uuid7', label: 'UUIDv7 / ULID', entryBytes: 28, decl: 'uuid, 48-bit ms prefix + random' },
  { id: 'tsrand', label: 'timestamp + random', entryBytes: 28, decl: '(created_at, id), second resolution' },
];

/** Key value for the i-th insert of a stream. Pure, so a rerun is identical. */
function keyFor(id: StreamId, i: number): number {
  const r = NOISE[i % NOISE.length];
  switch (id) {
    case 'serial':
      return i + 1;
    case 'uuid4':
      return r * 1e6;
    case 'uuid7':
      // 48-bit millisecond prefix: the bucket advances every 8 rows, and the
      // random tail scrambles order only inside the current millisecond.
      return Math.floor(i / 8) + r;
    case 'tsrand':
      // A one-second clock: ~64 rows land in the same bucket, so the hot band is
      // several leaves wide instead of one.
      return Math.floor(i / 64) + r;
  }
}

type Leaf = { block: number; lo: number; keys: number[] };

type Sim = {
  leaves: Leaf[]; // logical order: the sibling chain
  nextBlock: number;
  inserted: number;
  splits: number;
  rightmostHits: number;
  recent: number[]; // blocks touched by the last RECENT inserts
  rebuilds: number;
  full: boolean;
};

function freshSim(): Sim {
  return {
    leaves: [{ block: 0, lo: -Infinity, keys: [] }],
    nextBlock: 1,
    inserted: 0,
    splits: 0,
    rightmostHits: 0,
    recent: [],
    rebuilds: 0,
    full: false,
  };
}

function capacityFor(entryBytes: number) {
  const real = Math.floor(LEAF_USABLE / entryBytes);
  return { real, sim: Math.max(4, Math.round(real / SCALE)) };
}

/** Last leaf whose separator is <= key — the descent's answer. */
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

function runBatch(
  prev: Sim,
  id: StreamId,
  cap: number,
  fillfactor: number,
  rightmostOpt: boolean,
  count: number,
  maxLeaves: number,
): Sim {
  const s: Sim = {
    ...prev,
    leaves: prev.leaves.map((l) => ({ ...l, keys: l.keys.slice() })),
    recent: prev.recent.slice(),
  };

  for (let k = 0; k < count; k++) {
    if (s.leaves.length >= maxLeaves) {
      s.full = true;
      break;
    }
    const key = keyFor(id, s.inserted);
    s.inserted++;
    const idx = findLeaf(s.leaves, key);
    const leaf = s.leaves[idx];
    const wasRightmost = idx === s.leaves.length - 1;
    if (wasRightmost) s.rightmostHits++;
    s.recent.push(leaf.block);
    if (s.recent.length > RECENT) s.recent.shift();

    // sorted insert
    let p = leaf.keys.length;
    while (p > 0 && leaf.keys[p - 1] > key) p--;
    leaf.keys.splice(p, 0, key);

    if (leaf.keys.length <= cap) continue;

    // overflow: choose the split point
    const n = leaf.keys.length;
    const left =
      wasRightmost && rightmostOpt
        ? Math.max(1, Math.min(n - 1, Math.floor((cap * fillfactor) / 100)))
        : Math.ceil(n / 2);
    const rightKeys = leaf.keys.slice(left);
    leaf.keys = leaf.keys.slice(0, left);
    s.leaves.splice(idx + 1, 0, { block: s.nextBlock++, lo: rightKeys[0], keys: rightKeys });
    s.splits++;
  }
  return s;
}

/** Replay a stream from empty under one set of knobs. Pure — same args, same index. */
function build(id: StreamId, fillfactor: number, rightmostOpt: boolean, rows: number): Sim {
  const cap = capacityFor(STREAMS.find((s) => s.id === id)!.entryBytes).sim;
  return runBatch(freshSim(), id, cap, fillfactor, rightmostOpt, rows, MAX_LEAVES);
}

/** REINDEX / OPTIMIZE TABLE: sorted rebuild, packed to fillfactor, blocks in order. */
function rebuild(prev: Sim, cap: number, fillfactor: number): Sim {
  const all: number[] = [];
  for (const l of prev.leaves) for (const k of l.keys) all.push(k);
  const per = Math.max(1, Math.floor((cap * fillfactor) / 100));
  const leaves: Leaf[] = [];
  for (let i = 0; i < all.length; i += per) {
    leaves.push({ block: leaves.length, lo: i === 0 ? -Infinity : all[i], keys: all.slice(i, i + per) });
  }
  if (leaves.length === 0) leaves.push({ block: 0, lo: -Infinity, keys: [] });
  return { ...prev, leaves, nextBlock: leaves.length, recent: [], rebuilds: prev.rebuilds + 1, full: false };
}

type Metrics = {
  entries: number;
  pages: number;
  density: number;
  leafFragmentation: number;
  backwardLinks: number;
  scanFrom: number;
  scanLeaves: number;
  scanRuns: number;
  scanSpan: number;
  hotShare: number;
  workingSet: number;
};

function metrics(s: Sim, cap: number, scanPct: number): Metrics {
  const leaves = s.leaves;
  const pages = leaves.length;
  let entries = 0;
  for (const l of leaves) entries += l.keys.length;

  // pgstatindex: a leaf counts as fragmented when its right sibling is earlier in the file.
  let backwardLinks = 0;
  for (let i = 0; i < pages - 1; i++) if (leaves[i + 1].block < leaves[i].block) backwardLinks++;

  const scanLeaves = Math.max(1, Math.min(pages, Math.round((pages * scanPct) / 100)));
  const scanFrom = Math.max(0, Math.min(pages - scanLeaves, Math.floor(pages * 0.25)));
  let scanRuns = 1;
  let scanMin = Infinity;
  let scanMax = -Infinity;
  for (let i = scanFrom; i < scanFrom + scanLeaves; i++) {
    scanMin = Math.min(scanMin, leaves[i].block);
    scanMax = Math.max(scanMax, leaves[i].block);
    if (i + 1 < scanFrom + scanLeaves && leaves[i + 1].block !== leaves[i].block + 1) scanRuns++;
  }

  return {
    entries,
    pages,
    density: pages ? entries / (pages * cap) : 0,
    leafFragmentation: pages ? (backwardLinks / pages) * 100 : 0,
    backwardLinks,
    scanFrom,
    scanLeaves,
    scanRuns,
    scanSpan: pages ? scanMax - scanMin + 1 : 0,
    hotShare: s.inserted ? (s.rightmostHits / s.inserted) * 100 : 0,
    workingSet: new Set(s.recent).size,
  };
}

/* ------------------------------------------------------------------ colour */

const DENSITY_BUCKETS = [
  { label: 'under 50% full', color: 'var(--viz-seq-100)' },
  { label: '50–65%', color: 'var(--viz-seq-250)' },
  { label: '65–80%', color: 'var(--viz-seq-400)' },
  { label: '80–95%', color: 'var(--viz-seq-550)' },
  { label: '95%+', color: 'var(--viz-seq-700)' },
];

function bucketOf(d: number) {
  if (d < 0.5) return 0;
  if (d < 0.65) return 1;
  if (d < 0.8) return 2;
  if (d < 0.95) return 3;
  return 4;
}

/* ------------------------------------------------------------------ figure */

/** Lives inside <TooltipHost> so useTip() sees the provider. */
function LeafFigure({ sim, cap, m, avail }: { sim: Sim; cap: number; m: Metrics; avail: number }) {
  const tip = useTip();
  const pages = sim.leaves.length;
  const cellW = Math.max(9, Math.min(22, Math.floor((avail - 70) / Math.max(pages, 1))));
  const gap = cellW > 12 ? 2 : 1;
  const w = cellW - gap;
  const left = 58;
  const svgW = Math.max(340, left + pages * cellW + 12);

  const TOP_Y = 26;
  const BOT_Y = 150;
  const H = 46;
  const height = BOT_Y + H + 30;

  const scanTo = m.scanFrom + m.scanLeaves - 1;
  const x = (i: number) => left + i * cellW;

  const cell = (i: number, l: Leaf, y: number) => {
    const occ = l.keys.length / cap;
    const inScan = i >= m.scanFrom && i <= scanTo;
    const fillH = Math.max(1.5, occ * H);
    return (
      <g
        key={`lb${l.block}`}
        {...tip(
          <>
            <strong>block {l.block}</strong> — position {i + 1} of {pages} in the sibling chain
            <br />
            {l.keys.length} / {cap} entries ({Math.round(occ * 100)}% full)
            <br />
            right sibling: {i + 1 < pages ? `block ${sim.leaves[i + 1].block}` : 'none (rightmost leaf)'}
          </>,
        )}
      >
        <rect x={x(i)} y={y} width={w} height={H} rx={2} fill="var(--viz-neutral)" />
        <rect
          x={x(i)}
          y={y + H - fillH}
          width={w}
          height={fillH}
          rx={2}
          fill={DENSITY_BUCKETS[bucketOf(occ)].color}
        />
        <rect
          x={x(i)}
          y={y}
          width={w}
          height={H}
          rx={2}
          fill="none"
          stroke={inScan ? 'var(--viz-2)' : 'var(--viz-border)'}
          strokeWidth={inScan ? 1.5 : 1}
        />
      </g>
    );
  };

  return (
    <svg
      width={svgW}
      height={height}
      role="img"
      aria-label="Leaf pages in sibling-chain order above, in block-number order below, joined by lines"
    >
      <text x={0} y={TOP_Y - 8} fontSize={11} fill="var(--viz-ink-2)">
        logical
      </text>
      <text x={0} y={TOP_Y + 8} fontSize={11} fill="var(--viz-ink-muted)">
        chain
      </text>
      <text x={0} y={BOT_Y + 18} fontSize={11} fill="var(--viz-ink-2)">
        physical
      </text>
      <text x={0} y={BOT_Y + 34} fontSize={11} fill="var(--viz-ink-muted)">
        block #
      </text>

      {/* logical position i -> physical position (block number, no holes) */}
      {sim.leaves.map((l, i) => {
        const inScan = i >= m.scanFrom && i <= scanTo;
        return (
          <line
            key={`c${l.block}`}
            x1={x(i) + w / 2}
            y1={TOP_Y + H}
            x2={x(l.block) + w / 2}
            y2={BOT_Y}
            stroke={inScan ? 'var(--viz-2)' : 'var(--viz-axis)'}
            strokeWidth={inScan ? 1.4 : 1}
            opacity={inScan ? 0.95 : 0.4}
          />
        );
      })}

      {sim.leaves.map((l, i) => cell(i, l, TOP_Y))}
      {Array.from({ length: pages }, (_, b) => {
        const logical = sim.leaves.findIndex((l) => l.block === b);
        const l = sim.leaves[logical];
        if (!l) return null;
        const occ = l.keys.length / cap;
        const inScan = logical >= m.scanFrom && logical <= scanTo;
        const fillH = Math.max(1.5, occ * H);
        return (
          <g
            key={`pb${b}`}
            {...tip(
              <>
                <strong>block {b}</strong> — position {logical + 1} of {pages} in the sibling chain
                <br />
                {l.keys.length} / {cap} entries ({Math.round(occ * 100)}% full)
              </>,
            )}
          >
            <rect x={x(b)} y={BOT_Y} width={w} height={H} rx={2} fill="var(--viz-neutral)" />
            <rect
              x={x(b)}
              y={BOT_Y + H - fillH}
              width={w}
              height={fillH}
              rx={2}
              fill={DENSITY_BUCKETS[bucketOf(occ)].color}
            />
            <rect
              x={x(b)}
              y={BOT_Y}
              width={w}
              height={H}
              rx={2}
              fill="none"
              stroke={inScan ? 'var(--viz-2)' : 'var(--viz-border)'}
              strokeWidth={inScan ? 1.5 : 1}
            />
          </g>
        );
      })}

      <text x={left} y={BOT_Y + H + 16} fontSize={11} fill="var(--viz-ink-muted)">
        0
      </text>
      <text x={Math.max(left + 20, x(pages - 1))} y={BOT_Y + H + 16} fontSize={11} fill="var(--viz-ink-muted)">
        {pages - 1}
      </text>

      {/* the rightmost leaf, where every ascending insert lands */}
      <text x={x(pages - 1) + w} y={TOP_Y - 6} fontSize={11} textAnchor="end" fill="var(--viz-ink-2)">
        rightmost leaf ▾
      </text>
    </svg>
  );
}

/* ------------------------------------------------------------------- panel */

export default function LeafFillFragmentationLab() {
  const [streamId, setStreamId] = useState<StreamId>('serial');
  const [fillfactor, setFillfactor] = useState(90);
  const [scanPct, setScanPct] = useState(20);
  const [rightmostOpt, setRightmostOpt] = useState(true);
  const [sim, setSim] = useState<Sim>(() => build('serial', 90, true, SEED_ROWS));
  const [ref, avail] = useSize(820);

  const stream = STREAMS.find((s) => s.id === streamId)!;
  const cap = capacityFor(stream.entryBytes);
  const m = metrics(sim, cap.sim, scanPct);

  // Changing a knob replays the same number of rows under the new rule, so the
  // figure answers the question you just asked instead of going blank.
  const replay = (next: Partial<{ id: StreamId; ff: number; opt: boolean }>) => {
    const id = next.id ?? streamId;
    const ff = next.ff ?? fillfactor;
    const opt = next.opt ?? rightmostOpt;
    if (next.id !== undefined) setStreamId(next.id);
    if (next.ff !== undefined) setFillfactor(next.ff);
    if (next.opt !== undefined) setRightmostOpt(next.opt);
    setSim(build(id, ff, opt, Math.max(SEED_ROWS, sim.inserted)));
  };

  const insert = () =>
    setSim((s) => runBatch(s, streamId, cap.sim, fillfactor, rightmostOpt, BATCH, MAX_LEAVES));

  /* ------------------------------------------------- four-stream comparison */
  const comparison = useMemo(
    () =>
      STREAMS.map((st) => {
        const c = capacityFor(st.entryBytes);
        let s = freshSim();
        while (s.leaves.length < TABLE_LEAVES && !s.full) {
          s = runBatch(s, st.id, c.sim, fillfactor, rightmostOpt, 200, TABLE_LEAVES);
        }
        const mm = metrics(s, c.sim, scanPct);
        return { st, c, s, mm, perMillion: mm.density > 0 ? 1e6 / (c.real * mm.density) : 0 };
      }),
    [fillfactor, rightmostOpt, scanPct],
  );

  const pct = (n: number) => `${Math.round(n)}%`;

  return (
    <VizPanel
      title="Leaf occupancy and file order, under four key streams"
      subtitle={`Every leaf of one index. Top row: the sibling chain, left to right in key order. Bottom row: the same pages in block-number order, which is the order a sequential read returns them. Bar height is how full the page is. Leaf capacity is scaled down ${SCALE}× so the whole index fits on screen — the real 8 KB page holds ${cap.real} ${stream.label} entries.`}
      controls={
        <>
          <Segmented
            label="Key stream"
            value={streamId}
            onChange={(v) => replay({ id: v })}
            options={STREAMS.map((s) => ({ value: s.id, label: s.label, title: s.decl }))}
          />
          <Slider
            label="fillfactor"
            min={50}
            max={100}
            step={5}
            value={fillfactor}
            onChange={(n) => replay({ ff: n })}
            format={(n) => `${n}%${n === 90 ? ' (nbtree default)' : ''}`}
          />
          <Slider
            label="Range scan width"
            min={5}
            max={100}
            step={5}
            value={scanPct}
            onChange={setScanPct}
            format={(n) => `${n}% of the index`}
          />
          <Check
            label="Rightmost-page split optimization"
            checked={rightmostOpt}
            onChange={(b) => replay({ opt: b })}
          />
          <Button onClick={insert} primary disabled={sim.full}>
            Insert {fmtNum(BATCH)} rows
          </Button>
          <Button onClick={() => setSim((s) => rebuild(s, cap.sim, fillfactor))} disabled={sim.inserted === 0}>
            REINDEX
          </Button>
          <Button
            onClick={() => setSim(build(streamId, fillfactor, rightmostOpt, SEED_ROWS))}
            disabled={sim.inserted <= SEED_ROWS && sim.rebuilds === 0}
          >
            Reset
          </Button>
        </>
      }
      legend={
        <Legend
          items={[
            ...DENSITY_BUCKETS.map((b) => ({ label: b.label, color: b.color })),
            { label: 'range scan window', color: 'var(--viz-2)', shape: 'line' as const },
          ]}
        />
      }
      stats={
        <Stats
          items={[
            {
              label: 'Leaf pages',
              value: `${m.pages}${sim.full ? ' (capped)' : ''}`,
              hint: 'The simulation stops at 80 leaves so the strip stays readable',
            },
            { label: 'Rows indexed', value: fmtNum(m.entries) },
            {
              label: 'avg_leaf_density',
              value: pct(m.density * 100),
              hint: 'pgstatindex column: how full the leaf level actually is',
            },
            {
              label: 'leaf_fragmentation',
              value: pct(m.leafFragmentation),
              hint: `${m.backwardLinks} leaves whose right sibling sits at a lower block number`,
            },
            { label: 'Page splits', value: fmtNum(sim.splits), hint: 'One page allocation and one WAL record each' },
            {
              label: 'Inserts on the rightmost leaf',
              value: pct(m.hotShare),
              hint: 'The share of all inserts that contended for one buffer',
            },
            {
              label: 'Leaves touched, last 200 inserts',
              value: fmtNum(m.workingSet),
              hint: 'The write working set the buffer pool has to hold',
            },
            {
              label: 'Range scan',
              value: `${m.scanLeaves} leaves · ${m.scanRuns} runs`,
              hint: `A run is a maximal stretch of blocks readahead can carry you through. This window is scattered over ${m.scanSpan} of the file's ${m.pages} blocks.`,
            },
          ]}
        />
      }
      note={
        <Note>
          <strong>
            {streamId === 'serial' && rightmostOpt
              ? `Every insert lands on the rightmost leaf, so every split is a rightmost split at fillfactor: pages settle at ~${fillfactor}% and the chain stays in file order.`
              : streamId === 'serial'
                ? 'With midpoint splits everywhere, ascending keys freeze each page at half full and the index is twice the size it needs to be.'
                : streamId === 'uuid4'
                  ? 'Random keys hit a different leaf every time, so nearly every split is a midpoint split: density converges on ln 2 ≈ 69% and the chain scrambles against the file.'
                  : streamId === 'uuid7'
                    ? fillfactor >= 100
                      ? 'At fillfactor 100 there is no room left for the intra-millisecond jitter, so keys that sort backwards split a full non-rightmost page — the headroom is exactly what this stream needs.'
                      : 'The millisecond prefix pins inserts to the right edge, so the fillfactor rule applies again — the random tail only scrambles order inside the current millisecond.'
                    : 'A one-second clock makes the hot band several leaves wide: the write working set stays small, but splits inside the band are midpoint splits and scramble block order locally.'}
          </strong>{' '}
          {sim.rebuilds > 0
            ? `After ${sim.rebuilds} rebuild${sim.rebuilds > 1 ? 's' : ''}, keep inserting and watch fragmentation come back — a REINDEX buys time, not a fix. `
            : ''}
          Follow the lines: they are the mapping from key order to file order, and every one that crosses another is a
          seek a range scan has to pay.
        </Note>
      }
      table={
        <table className="viz-table">
          <caption>
            Each stream inserted into a fresh index until it reaches {TABLE_LEAVES} leaves, at fillfactor {fillfactor}
            {rightmostOpt ? '' : ' with the rightmost-split optimization off'}.
          </caption>
          <thead>
            <tr>
              <th>Key stream</th>
              <th>Entry bytes</th>
              <th>Entries / 8 KB leaf</th>
              <th>Rows stored</th>
              <th>avg_leaf_density</th>
              <th>leaf_fragmentation</th>
              <th>Splits / 1k rows</th>
              <th>Rightmost-leaf inserts</th>
              <th>Leaf pages per 1M rows</th>
            </tr>
          </thead>
          <tbody>
            {comparison.map(({ st, c, s, mm, perMillion }) => (
              <tr key={st.id}>
                <td>{st.label}</td>
                <td>{st.entryBytes} B</td>
                <td>{c.real}</td>
                <td>{fmtNum(mm.entries)}</td>
                <td>{pct(mm.density * 100)}</td>
                <td>{pct(mm.leafFragmentation)}</td>
                <td>{((s.splits / Math.max(1, s.inserted)) * 1000).toFixed(1)}</td>
                <td>{pct(mm.hotShare)}</td>
                <td>{fmtNum(perMillion)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      }
    >
      <div ref={ref}>
        <TooltipHost>
          <LeafFigure sim={sim} cap={cap.sim} m={m} avail={avail} />
        </TooltipHost>
      </div>
    </VizPanel>
  );
}
