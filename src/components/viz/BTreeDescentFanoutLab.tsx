import { useState } from 'react';
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
  fmtBytes,
  fmtNum,
  makeRng,
  useSize,
} from './Viz';

/**
 * A B+tree descent, built from the numbers that actually decide its shape.
 *
 * Fanout is not a tuning knob — it falls out of (usable bytes per page) / (bytes per
 * entry), and bytes per entry is decided by the key width and by whether the engine
 * puts records in internal nodes. Everything the learner changes here feeds that one
 * division: page size and key width move the divisor, B-tree mode adds the record to
 * every entry, and height, addressable rows and page reads all follow.
 *
 * The descent is simulated exactly: at each level the model runs the same lower_bound
 * binary search an engine runs over the sorted slot array, and records which separators
 * it actually compared, so the drawn probes are the real probes.
 */

/* ------------------------------------------------------------------ sizing */

const align8 = (n: number) => Math.ceil(n / 8) * 8;

/** PageHeaderData (24 B) + the B-tree special area (16 B in nbtree). */
const PAGE_OVERHEAD = 40;
/** IndexTupleData: 6-byte t_tid (heap TID, or the downlink block number) + 2-byte t_info. */
const TUPLE_HEADER = 8;
/** The 4-byte ItemIdData line pointer each entry costs in the slot array. */
const LINE_POINTER = 4;
/** A representative heap row, for the "records in every node" B-tree mode. */
const ROW_BYTES = 120;
/** CREATE INDEX fillfactor for a freshly built nbtree leaf. */
const FILL = 0.9;

const KEY_WIDTHS = [8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512, 768, 1024, 1536, 2048];

const PAGES = [
  { value: '4096', label: '4 KB (SQLite)' },
  { value: '8192', label: '8 KB (PostgreSQL)' },
  { value: '16384', label: '16 KB (InnoDB)' },
  { value: '32768', label: '32 KB' },
] as const;

type Mode = 'bplus' | 'btree';

function fmtKey(n: number) {
  if (n < 100_000) return fmtNum(n);
  if (n < 1e6) return `${(n / 1e3).toFixed(0)}K`;
  if (n < 1e9) return `${(n / 1e6).toFixed(n < 1e7 ? 2 : 1)}M`;
  if (n < 1e12) return `${(n / 1e9).toFixed(n < 1e10 ? 2 : 1)}B`;
  return `${(n / 1e12).toFixed(2)}T`;
}

/* ------------------------------------------------------------------- model */

type Node = {
  level: number;
  isLeaf: boolean;
  nodeIndex: number;
  nodesAtLevel: number;
  slots: number; // separator keys (internal) or index entries (leaf)
  probes: number[]; // slot indices the binary search actually touched, in order
  chosen: number; // child index taken, or the matching entry in the leaf
  first: number; // first key in this node's subtree
  span: number; // keys covered by this node's subtree
  childSpan: number; // keys covered by one child
  block: number;
};

type Plan = {
  entryBytes: number;
  pivotBytes: number;
  usable: number;
  leafCap: number;
  fanout: number;
  leaves: number;
  height: number;
  counts: number[]; // nodes per level, root first
  maxRows: number;
  path: Node[];
  leafIndex: number;
  endLeaf: number;
  leavesTouched: number;
  pagesBplus: number;
  pagesBtree: number;
  maxItem: number;
  oversized: boolean;
};

function plan(pageBytes: number, keyBytes: number, rows: number, mode: Mode, key: number, want: number): Plan {
  const usable = Math.floor((pageBytes - PAGE_OVERHEAD) * FILL);
  // nbtree's BTMaxItemSize: a third of the page after the header (MAXALIGNed with three
  // line pointers) and the special area, MAXALIGN_DOWNed, less one ItemPointerData.
  // 8 KB pages give 2704 B, which is the number in the "index row size ... exceeds
  // btree version 4 maximum" error.
  const maxItem = Math.floor(Math.floor((pageBytes - 56) / 3) / 8) * 8 - 8;

  // A B+tree leaf entry is key + heap TID; a pivot tuple is the same size, because the
  // downlink reuses the t_tid field. A classic B-tree carries the record everywhere.
  const leafPayload = mode === 'btree' ? ROW_BYTES : 0;
  const entryBytes = align8(TUPLE_HEADER + keyBytes + leafPayload) + LINE_POINTER;
  const pivotBytes = align8(TUPLE_HEADER + keyBytes + leafPayload) + LINE_POINTER;

  const leafCap = Math.max(2, Math.floor(usable / entryBytes));
  const fanout = Math.max(2, Math.floor(usable / pivotBytes));

  const leaves = Math.max(1, Math.ceil(rows / leafCap));
  const counts: number[] = [leaves];
  while (counts[0]! > 1) counts.unshift(Math.ceil(counts[0]! / fanout));
  const height = counts.length;

  const maxRows =
    mode === 'btree'
      ? Array.from({ length: height - 1 }, (_, j) => Math.pow(fanout, j) * (fanout - 1)).reduce(
          (a, b) => a + b,
          0,
        ) + Math.pow(fanout, height - 1) * leafCap
      : Math.pow(fanout, height - 1) * leafCap;

  // Block numbers that look like a real index file rather than 0,1,2,3.
  const rng = makeRng(pageBytes ^ (keyBytes << 7) ^ (rows % 100003) ^ (mode === 'btree' ? 8191 : 17));
  const blocks = Array.from({ length: height }, () => 1 + Math.floor(rng() * Math.max(8, leaves)));

  // Descend top-down, running the real lower_bound at every level.
  const path: Node[] = [];
  let nodeIndex = 0;
  for (let j = 0; j < height; j++) {
    const span = leafCap * Math.pow(fanout, height - 1 - j);
    const first = nodeIndex * span + 1;
    const isLeaf = j === height - 1;
    const childSpan = isLeaf ? 1 : leafCap * Math.pow(fanout, height - 2 - j);
    const children = isLeaf
      ? Math.max(1, Math.min(leafCap, rows - nodeIndex * leafCap))
      : Math.max(1, Math.min(fanout, counts[j + 1]! - nodeIndex * fanout));
    const slots = isLeaf ? children : children - 1;
    // slotKey(i): the i-th separator (internal) or the i-th key (leaf).
    const slotKey = (i: number) => first + (isLeaf ? i : (i + 1) * childSpan);

    let lo = 0;
    let hi = slots;
    const probes: number[] = [];
    while (lo < hi) {
      const mid = (lo + hi) >> 1;
      probes.push(mid);
      if (slotKey(mid) <= key) lo = mid + 1;
      else hi = mid;
    }
    const chosen = isLeaf ? Math.max(0, Math.min(slots - 1, lo - 1)) : lo;

    path.push({
      level: j,
      isLeaf,
      nodeIndex,
      nodesAtLevel: counts[j]!,
      slots,
      probes,
      chosen,
      first,
      span,
      childSpan,
      block: blocks[j]!,
    });
    if (!isLeaf) nodeIndex = nodeIndex * fanout + chosen;
  }

  const leafIndex = path[height - 1]!.nodeIndex;
  const endKey = Math.min(rows, key + want - 1);
  const endLeaf = Math.min(leaves - 1, Math.floor((endKey - 1) / leafCap));
  const leavesTouched = Math.max(1, endLeaf - leafIndex + 1);

  const pagesBplus = height - 1 + leavesTouched;
  let pagesBtree = leavesTouched;
  for (let j = 0; j < height - 1; j++) {
    const d = Math.pow(fanout, height - 1 - j);
    pagesBtree += Math.floor(endLeaf / d) - Math.floor(leafIndex / d) + 1;
  }

  return {
    entryBytes,
    pivotBytes,
    usable,
    leafCap,
    fanout,
    leaves,
    height,
    counts,
    maxRows,
    path,
    leafIndex,
    endLeaf,
    leavesTouched,
    pagesBplus,
    pagesBtree,
    maxItem,
    oversized: entryBytes - LINE_POINTER > maxItem,
  };
}

/* ----------------------------------------------------------------- drawing */

const ROW_H = 74;
const GUTTER = 108;
const CELL_W = 62;
const CELL_H = 26;
const GAP = 4;
const ELLIPSIS_W = 18;

type Cell = { kind: 'slot'; slot: number; probe: number } | { kind: 'gap' };

function cellsFor(n: Node): Cell[] {
  if (n.slots <= 0) return [];
  const shown = new Set<number>([0, n.slots - 1, ...n.probes]);
  if (!n.isLeaf) shown.add(Math.max(0, Math.min(n.slots - 1, n.chosen - 1)));
  const sorted = [...shown].sort((a, b) => a - b);
  const out: Cell[] = [];
  let prev = -1;
  for (const s of sorted) {
    if (prev >= 0 && s - prev > 1) out.push({ kind: 'gap' });
    out.push({ kind: 'slot', slot: s, probe: n.probes.indexOf(s) });
    prev = s;
  }
  return out;
}

export default function BTreeDescentFanoutLab() {
  const [pageStr, setPageStr] = useState<string>('8192');
  const [keyIdx, setKeyIdx] = useState(0);
  const [rowsExp, setRowsExp] = useState(9);
  const [wantExp, setWantExp] = useState(0);
  const [keyPct, setKeyPct] = useState(618);
  const [mode, setMode] = useState<Mode>('bplus');
  const [step, setStep] = useState(0);
  const [ref, width] = useSize(760);
  const tip = useTip();

  const pageBytes = Number(pageStr);
  const keyBytes = KEY_WIDTHS[keyIdx]!;
  const rows = Math.round(Math.pow(10, rowsExp));
  const want = Math.round(Math.pow(10, wantExp));
  const searchKey = 1 + Math.floor((keyPct / 1000) * (rows - 1));

  const p = plan(pageBytes, keyBytes, rows, mode, searchKey, want);
  const alt = plan(pageBytes, keyBytes, rows, mode === 'bplus' ? 'btree' : 'bplus', searchKey, want);

  const maxStep = p.height + 1;
  const s = Math.min(step, maxStep);
  const pages = mode === 'bplus' ? p.pagesBplus : p.pagesBtree;

  // Which levels get a row: all of them if the tree is short, otherwise elide the middle.
  const drawn: (Node | 'elide')[] =
    p.height <= 6
      ? p.path
      : [p.path[0]!, p.path[1]!, 'elide', p.path[p.height - 3]!, p.path[p.height - 2]!, p.path[p.height - 1]!];

  const widest = Math.max(...p.path.map((n) => cellsFor(n).length));
  const boxW = Math.max(320, widest * (CELL_W + GAP) + 24);
  const baseW = Math.max(560, Math.min(width, 980));
  const svgW = Math.max(baseW, GUTTER + boxW + 24);
  const chainTop = drawn.length * ROW_H + 26;
  const svgH = chainTop + 96;

  // Leaf-chain window: up to 13 pages starting just before the first leaf touched.
  const winStart = Math.max(0, Math.min(p.leafIndex - 1, p.leaves - 9));
  const winCount = Math.min(9, p.leaves - winStart);
  const chainW = Math.max(34, Math.floor((svgW - GUTTER - 24) / Math.max(winCount, 1)) - 8);

  const levelName = (n: Node) =>
    n.isLeaf ? (p.height === 1 ? 'Root = leaf' : 'Leaf') : n.level === 0 ? 'Root' : `Internal L${n.level}`;

  const narration = () => {
    if (s === 0) {
      return (
        <>
          <strong>Nothing read yet.</strong> The metapage names the root; every lookup starts there.
          Press <em>Step</em> to read one node at a time — the tree is {p.height} level
          {p.height === 1 ? '' : 's'} deep, so a point lookup for key {fmtKey(searchKey)} costs{' '}
          {p.height} page read{p.height === 1 ? '' : 's'} no matter which key you pick.
        </>
      );
    }
    if (s <= p.height) {
      const n = p.path[s - 1]!;
      const hit = n.probes.length ? n.first + (n.isLeaf ? n.probes[n.probes.length - 1]! : (n.probes[n.probes.length - 1]! + 1) * n.childSpan) : n.first;
      return (
        <>
          <strong>
            {levelName(n)} (block {fmtNum(n.block)}): {fmtNum(n.slots)} {n.isLeaf ? 'index entries' : 'separator keys'}, {n.probes.length} comparison
            {n.probes.length === 1 ? '' : 's'}.
          </strong>{' '}
          {n.isLeaf ? (
            <>
              Binary search over the sorted line-pointer array lands on entry {fmtNum(n.chosen)}, key{' '}
              {fmtKey(n.first + n.chosen)}
              {mode === 'bplus'
                ? ' — which holds a heap TID, not the row. The row still has to be fetched.'
                : ' — which holds the record itself.'}
            </>
          ) : (
            <>
              Last probe compared slot {fmtNum(n.probes[n.probes.length - 1] ?? 0)} (separator{' '}
              {fmtKey(hit)}); the search descends into child {fmtNum(n.chosen)} of {fmtNum(n.slots + 1)},
              which covers keys {fmtKey(n.first + n.chosen * n.childSpan)}–
              {fmtKey(n.first + (n.chosen + 1) * n.childSpan - 1)}.
            </>
          )}
        </>
      );
    }
    if (want === 1) {
      return (
        <>
          <strong>Done: {p.height} page reads.</strong> Every key in this index costs exactly the same
          descent, because all {fmtNum(p.leaves)} leaves sit at the same depth. Raise{' '}
          <em>rows returned</em> to turn this into a range scan.
        </>
      );
    }
    return mode === 'bplus' ? (
      <>
        <strong>
          Leaf chain: {fmtNum(p.leavesTouched)} leaf page{p.leavesTouched === 1 ? '' : 's'} walked.
        </strong>{' '}
        After the descent the scan never touches an internal node again — it follows{' '}
        <code>btpo_next</code> sideways, so {fmtNum(want)} rows cost {fmtNum(p.pagesBplus)} pages
        instead of {fmtNum(want)} descents.
      </>
    ) : (
      <>
        <strong>No leaf chain.</strong> With records in every node the traversal has to climb back
        through ancestors to find the next subtree, so the same {fmtNum(want)} rows touch{' '}
        {fmtNum(p.pagesBtree)} distinct nodes against the B+tree&rsquo;s {fmtNum(alt.pagesBplus)}.
      </>
    );
  };

  return (
    <VizPanel
      title="A B+tree descent, sized by real bytes"
      subtitle="Fanout is (usable bytes per page) ÷ (bytes per entry). Move either side of that division and watch height, addressable rows and page reads move with it."
      controls={
        <>
          <Choice
            label="Page size"
            value={pageStr}
            onChange={setPageStr}
            options={PAGES.map((x) => ({ value: x.value, label: x.label }))}
          />
          <Slider
            label="Key width"
            min={0}
            max={KEY_WIDTHS.length - 1}
            value={keyIdx}
            onChange={(n) => setKeyIdx(n)}
            format={() => `${keyBytes} B`}
          />
          <Slider
            label="Table rows"
            min={3}
            max={11}
            value={rowsExp}
            onChange={setRowsExp}
            format={() => fmtKey(rows)}
          />
          <Slider
            label="Rows returned"
            min={0}
            max={6}
            value={wantExp}
            onChange={setWantExp}
            format={() => (want === 1 ? '1 (point lookup)' : fmtNum(want))}
          />
          <Slider
            label="Search key"
            min={0}
            max={1000}
            value={keyPct}
            onChange={setKeyPct}
            format={() => fmtKey(searchKey)}
          />
          <Segmented
            label="Records live in"
            value={mode}
            onChange={(m) => setMode(m)}
            options={[
              { value: 'bplus', label: 'B+tree: leaves only', title: 'Internal nodes hold separator keys and downlinks only' },
              { value: 'btree', label: 'B-tree: every node', title: 'Every entry carries the record, so the pivots get fat' },
            ]}
          />
          <Button onClick={() => setStep(Math.min(s + 1, maxStep))} disabled={s >= maxStep} primary>
            Step
          </Button>
          <Button onClick={() => setStep(0)} disabled={s === 0}>
            Reset
          </Button>
        </>
      }
      legend={
        <Legend
          items={[
            { label: 'Node read on this descent', color: 'var(--viz-1)' },
            { label: 'Separator compared (binary-search probe)', color: 'var(--viz-2)' },
            { label: 'Leaf page walked by the range scan', color: 'var(--viz-3)' },
            { label: 'Not read', color: 'var(--viz-neutral)', shape: 'square' },
          ]}
        />
      }
      stats={
        <Stats
          items={[
            {
              label: 'Fanout',
              value: fmtNum(p.fanout),
              hint: `${fmtBytes(p.usable)} usable per page ÷ ${p.pivotBytes} B per pivot entry`,
            },
            {
              label: 'Entries per leaf',
              value: fmtNum(p.leafCap),
              hint: `${p.entryBytes} B per leaf entry, including the 4-byte line pointer`,
            },
            { label: 'Height', value: fmtNum(p.height), hint: 'Levels, leaf included — every leaf is at this depth' },
            { label: 'Rows addressable', value: fmtKey(p.maxRows), hint: 'With every node packed to the fill factor' },
            {
              label: want === 1 ? 'Pages per lookup' : 'Pages per range scan',
              value: fmtNum(pages),
              hint: mode === 'bplus' ? 'Descent + leaf-chain walk' : 'Distinct nodes visited by the in-order traversal',
            },
          ]}
        />
      }
      note={<Note>{narration()}</Note>}
      table={
        <table className="viz-table">
          <thead>
            <tr>
              <th>Level</th>
              <th>Nodes at level</th>
              <th>Slots in path node</th>
              <th>Comparisons</th>
              <th>Child / entry taken</th>
              <th>Key range covered</th>
            </tr>
          </thead>
          <tbody>
            {p.path.map((n) => (
              <tr key={n.level}>
                <td>
                  {levelName(n)} (blk {fmtNum(n.block)})
                </td>
                <td>{fmtNum(n.nodesAtLevel)}</td>
                <td>{fmtNum(n.slots)}</td>
                <td>{fmtNum(n.probes.length)}</td>
                <td>{fmtNum(n.chosen)}</td>
                <td>
                  {fmtKey(n.first)}–{fmtKey(Math.min(rows, n.first + n.span - 1))}
                </td>
              </tr>
            ))}
            <tr>
              <td>Leaf chain</td>
              <td>{fmtNum(p.leaves)}</td>
              <td>{fmtNum(p.leavesTouched)} walked</td>
              <td>—</td>
              <td>—</td>
              <td>
                {fmtKey(searchKey)}–{fmtKey(Math.min(rows, searchKey + want - 1))}
              </td>
            </tr>
            <tr>
              <td>Total pages</td>
              <td>B+tree {fmtNum(mode === 'bplus' ? p.pagesBplus : alt.pagesBplus)}</td>
              <td>B-tree {fmtNum(mode === 'btree' ? p.pagesBtree : alt.pagesBtree)}</td>
              <td colSpan={3}>
                Height B+tree {fmtNum(mode === 'bplus' ? p.height : alt.height)} · B-tree{' '}
                {fmtNum(mode === 'btree' ? p.height : alt.height)} · leaf entry {p.entryBytes} B
              </td>
            </tr>
          </tbody>
        </table>
      }
    >
      <TooltipHost>
        <div ref={ref}>
          <svg width={svgW} height={svgH} role="img" aria-label="B+tree descent with per-level binary search">
            {drawn.map((d, row) => {
              const y = row * ROW_H + 8;
              if (d === 'elide') {
                return (
                  <g key="elide">
                    <text x={GUTTER} y={y + 26} fill="var(--viz-ink-muted)">
                      ⋮ {fmtNum(p.height - 5)} more internal level{p.height - 5 === 1 ? '' : 's'}, same
                      rule at each
                    </text>
                  </g>
                );
              }
              const read = d.level < s;
              const cells = cellsFor(d);
              const stroke = read ? 'var(--viz-1)' : 'var(--viz-axis)';
              let cx = GUTTER + 12;
              return (
                <g key={d.level}>
                  <text x={0} y={y + 20} fill="var(--viz-ink)" style={{ fontWeight: 600 }}>
                    {levelName(d)}
                  </text>
                  <text x={0} y={y + 36} fill="var(--viz-ink-muted)">
                    blk {fmtNum(d.block)}
                  </text>
                  <text x={0} y={y + 52} fill="var(--viz-ink-muted)">
                    {fmtNum(d.nodesAtLevel)} node{d.nodesAtLevel === 1 ? '' : 's'}
                  </text>
                  <rect
                    x={GUTTER}
                    y={y}
                    width={boxW}
                    height={CELL_H + 20}
                    rx={8}
                    fill={read ? 'var(--viz-plane)' : 'none'}
                    stroke={stroke}
                    strokeWidth={read ? 1.5 : 1}
                    strokeDasharray={read ? undefined : '4 3'}
                  />
                  {cells.map((c, i) => {
                    if (c.kind === 'gap') {
                      const gx = cx;
                      cx += ELLIPSIS_W + GAP;
                      return (
                        <text key={`g${i}`} x={gx} y={y + 28} fill="var(--viz-ink-muted)">
                          ⋯
                        </text>
                      );
                    }
                    const x = cx;
                    cx += CELL_W + GAP;
                    const isProbe = read && c.probe >= 0;
                    const isMatch = read && d.isLeaf && c.slot === d.chosen;
                    const val = d.first + (d.isLeaf ? c.slot : (c.slot + 1) * d.childSpan);
                    return (
                      <g
                        key={`c${c.slot}`}
                        {...tip(
                          <>
                            {d.isLeaf ? 'entry' : 'separator'} [{fmtNum(c.slot)}] = {fmtNum(val)}
                            {c.probe >= 0 ? ` · probe #${c.probe + 1}` : ''}
                          </>,
                        )}
                      >
                        <rect
                          x={x}
                          y={y + 10}
                          width={CELL_W}
                          height={CELL_H}
                          rx={4}
                          fill={isMatch ? 'var(--viz-3)' : isProbe ? 'var(--viz-2)' : 'var(--viz-neutral)'}
                          stroke={isProbe || isMatch ? 'none' : 'var(--viz-border)'}
                        />
                        <text
                          x={x + CELL_W / 2}
                          y={y + 27}
                          textAnchor="middle"
                          fill={isProbe || isMatch ? 'var(--viz-surface)' : 'var(--viz-ink-2)'}
                          style={{ fontVariantNumeric: 'tabular-nums' }}
                        >
                          {fmtKey(val)}
                        </text>
                        {isProbe ? (
                          <text x={x + CELL_W / 2} y={y + 7} textAnchor="middle" fill="var(--viz-2)">
                            #{c.probe + 1}
                          </text>
                        ) : null}
                      </g>
                    );
                  })}
                  {read && !d.isLeaf ? (
                    <text x={GUTTER + boxW - 8} y={y + 46} textAnchor="end" fill="var(--viz-ink-2)">
                      → child {fmtNum(d.chosen)} of {fmtNum(d.slots + 1)}
                    </text>
                  ) : null}
                  {read && !d.isLeaf && row < drawn.length - 1 ? (
                    <line
                      x1={GUTTER + boxW / 2}
                      y1={y + CELL_H + 20}
                      x2={GUTTER + boxW / 2}
                      y2={y + ROW_H}
                      stroke="var(--viz-1)"
                      strokeWidth={1.5}
                    />
                  ) : null}
                </g>
              );
            })}

            {/* leaf chain */}
            <text x={0} y={chainTop + 20} fill="var(--viz-ink)" style={{ fontWeight: 600 }}>
              Leaf chain
            </text>
            <text x={0} y={chainTop + 36} fill="var(--viz-ink-muted)">
              {fmtNum(p.leaves)} pages
            </text>
            {Array.from({ length: winCount }, (_, i) => {
              const idx = winStart + i;
              const x = GUTTER + i * (chainW + 8);
              const touched = s > p.height && idx >= p.leafIndex && idx <= p.endLeaf;
              const isStart = idx === p.leafIndex;
              return (
                <g
                  key={idx}
                  {...tip(
                    <>
                      leaf #{fmtNum(idx)} · keys {fmtKey(idx * p.leafCap + 1)}–
                      {fmtKey(Math.min(rows, (idx + 1) * p.leafCap))}
                    </>,
                  )}
                >
                  <rect
                    x={x}
                    y={chainTop + 4}
                    width={chainW}
                    height={CELL_H}
                    rx={4}
                    fill={touched ? 'var(--viz-3)' : 'var(--viz-neutral)'}
                    stroke={isStart && s >= p.height ? 'var(--viz-1)' : 'var(--viz-border)'}
                    strokeWidth={isStart && s >= p.height ? 2 : 1}
                  />
                  <text
                    x={x + chainW / 2}
                    y={chainTop + 21}
                    textAnchor="middle"
                    fill={touched ? 'var(--viz-surface)' : 'var(--viz-ink-2)'}
                  >
                    {fmtNum(idx)}
                  </text>
                  {i < winCount - 1 && mode === 'bplus' ? (
                    <text x={x + chainW + 4} y={chainTop + 21} fill="var(--viz-ink-muted)">
                      →
                    </text>
                  ) : null}
                </g>
              );
            })}
            <text x={GUTTER} y={chainTop + 54} fill="var(--viz-ink-2)">
              {mode === 'bplus'
                ? `btpo_next / btpo_prev link every leaf — ${fmtNum(p.leavesTouched)} of ${fmtNum(p.leaves)} pages walked, no re-descent`
                : `no sibling pointers — the scan re-ascends through ancestors, ${fmtNum(p.pagesBtree)} distinct nodes touched`}
            </text>
            {p.oversized && mode === 'bplus' ? (
              <text x={GUTTER} y={chainTop + 74} fill="var(--viz-critical)">
                Index tuple is {fmtNum(p.entryBytes - LINE_POINTER)} B — over a third of a{' '}
                {fmtBytes(pageBytes)} page. PostgreSQL rejects this index outright.
              </text>
            ) : (
              <text x={GUTTER} y={chainTop + 74} fill="var(--viz-ink-muted)">
                Leaf entry {p.entryBytes} B ({TUPLE_HEADER} B header + {keyBytes} B key
                {mode === 'btree' ? ` + ${ROW_BYTES} B record` : ' + 6 B TID inside the header'} +{' '}
                {LINE_POINTER} B line pointer), fill factor {Math.round(FILL * 100)}%
              </text>
            )}
          </svg>
        </div>
      </TooltipHost>
    </VizPanel>
  );
}
