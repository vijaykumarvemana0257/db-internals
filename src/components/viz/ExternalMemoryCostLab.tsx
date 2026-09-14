import { useMemo, useState } from 'react';
import {
  VizPanel,
  Slider,
  Segmented,
  Legend,
  Stats,
  Note,
  TooltipHost,
  useTip,
  makeRng,
  fmtBytes,
  fmtNum,
  useSize,
} from './Viz';

/**
 * The DAM / external-memory model, run rather than asserted.
 *
 * One deterministic key workload is replayed over four layouts. Every layout is
 * simulated as real byte offsets into its own address space; a block transfer is
 * counted whenever a touched offset falls in a block that is not in an LRU cache
 * of M/B blocks. So the y-axis is *measured* I/O for the model, and the dashed
 * lines are the analytic Θ(...) each layout is supposed to hit with a cold cache.
 *
 * The x-axis is the engine's configured page size P; the slider is the hardware's
 * true transfer unit B. The B+tree's optimum sits at P ≈ B and moves when B moves.
 * The van Emde Boas layout has no P to configure, so its line is flat in P and
 * still tracks log_B N — which is the whole claim of cache-obliviousness.
 */

const ENTRY = 16; // bytes per index entry: key + child pointer
const HEADER = 24; // per-page header bytes
const OPS = 64; // operations measured per data point
const SCAN_K = 1000; // records touched by one range scan
const T_RATIO = 10; // leveled-LSM size ratio
const FPR = 0.01; // Bloom-filter false-positive rate
const PMA_DENSITY = 0.7; // packed-memory array fill
const ALIGN = 65536; // region alignment, >= the largest B

const PAGE_EXPS = [9, 10, 11, 12, 13, 14, 15, 16];

/** Compact record counts for a control label. */
const short = (n: number) =>
  n >= 1e9 ? `${(n / 1e9).toFixed(n < 1e10 ? 1 : 0)}B` : n >= 1e6 ? `${(n / 1e6).toFixed(n < 1e7 ? 1 : 0)}M` : fmtNum(n);

type LayoutId = 'array' | 'btree' | 'veb' | 'lsm';

const LAYOUTS: { id: LayoutId; label: string; color: string; tag: string }[] = [
  { id: 'array', label: 'Sorted array + binary search', color: 'var(--viz-1)', tag: 'binary search' },
  { id: 'btree', label: 'B+tree, node = page size P', color: 'var(--viz-2)', tag: 'B+tree' },
  { id: 'veb', label: 'van Emde Boas layout (no P)', color: 'var(--viz-3)', tag: 'vEB' },
  { id: 'lsm', label: 'Leveled LSM, T=10, Bloom resident', color: 'var(--viz-4)', tag: 'LSM' },
];

/* ----------------------------------------------------------------- machinery */

/** LRU over block ids; returns true on a hit. Map iteration order is the LRU order. */
function makeCache(capBlocks: number) {
  const m = new Map<number, true>();
  return (id: number) => {
    if (m.has(id)) {
      m.delete(id);
      m.set(id, true);
      return true;
    }
    m.set(id, true);
    if (m.size > capBlocks) {
      const oldest = m.keys().next();
      if (!oldest.done) m.delete(oldest.value);
    }
    return false;
  };
}

type Touch = (id: number) => boolean;

/** Count the block transfers needed to read [off, off+len). */
function touchRange(touch: Touch, off: number, len: number, B: number) {
  const first = Math.floor(off / B);
  const last = Math.floor((off + Math.max(1, len) - 1) / B);
  let misses = 0;
  for (let id = first; id <= last; id++) if (!touch(id)) misses++;
  return misses;
}

const alignUp = (n: number) => Math.ceil(n / ALIGN) * ALIGN;

/** Slot index of the node reached by `path` in a height-h tree laid out van Emde Boas. */
function vebSlot(path: number[], from: number, h: number, base: number): number {
  if (h <= 1 || path.length === from) return base;
  const top = Math.floor(h / 2);
  const bot = h - top;
  if (path.length - from < top) return vebSlot(path, from, top, base);
  let idx = 0;
  for (let i = 0; i < top; i++) idx = idx * 2 + path[from + i];
  const nextBase = base + (2 ** top - 1) + idx * (2 ** bot - 1);
  return vebSlot(path, from + top, bot, nextBase);
}

type Op = { rank: number; leaf: number; bits: number[]; levelDraw: number; fp: number[] };

function makeOps(N: number, count: number, hMax: number, seed: number): Op[] {
  const rng = makeRng(seed);
  const ops: Op[] = [];
  for (let i = 0; i < count; i++) {
    const u = rng();
    const rank = Math.min(N - 1, Math.floor(u * N));
    const bits: number[] = [];
    let x = u;
    for (let d = 0; d < hMax; d++) {
      x *= 2;
      const b = x >= 1 ? 1 : 0;
      if (b) x -= 1;
      bits.push(b);
    }
    ops.push({
      rank,
      leaf: rank,
      bits,
      levelDraw: rng(),
      fp: Array.from({ length: 12 }, () => rng()),
    });
  }
  return ops;
}

function btreeShape(N: number, P: number) {
  const fanout = Math.max(2, Math.floor((P - HEADER) / ENTRY));
  const counts: number[] = [Math.max(1, Math.ceil(N / fanout))]; // counts[0] = leaves
  while (counts[counts.length - 1] > 1) counts.push(Math.ceil(counts[counts.length - 1] / fanout));
  const base: number[] = new Array(counts.length);
  let acc = 0;
  for (let i = counts.length - 1; i >= 0; i--) {
    // root level first in the file, leaves last — how a bulk-loaded index is written
    base[i] = acc;
    acc += counts[i] * P;
  }
  return { fanout, counts, base, depth: counts.length };
}

function lsmShape(N: number, M: number) {
  const memEntries = Math.max(4096, Math.floor(M / ENTRY / 4));
  const levels = Math.max(1, Math.min(10, Math.ceil(Math.log(Math.max(2, N / memEntries)) / Math.log(T_RATIO))));
  const weights: number[] = [];
  let sum = 0;
  for (let i = 1; i <= levels; i++) {
    const w = T_RATIO ** i;
    weights.push(w);
    sum += w;
  }
  const entries = weights.map((w) => Math.max(1, Math.round((w / sum) * N)));
  const base: number[] = [];
  let acc = 0;
  for (let i = 0; i < levels; i++) {
    base.push(acc);
    acc = alignUp(acc + entries[i] * ENTRY + ALIGN);
  }
  return { levels, entries, base, total: acc };
}

type Measured = { transfers: number; bytes: number };

function simulate(
  layout: LayoutId,
  ops: Op[],
  N: number,
  M: number,
  B: number,
  P: number,
  scan: boolean,
): Measured {
  const scanBytes = SCAN_K * ENTRY;
  const bt = btreeShape(N, P);
  const ls = lsmShape(N, M);
  const h = Math.max(1, Math.min(30, Math.ceil(Math.log2(N + 1))));
  const dataBase = alignUp((2 ** h - 1) * ENTRY);
  // Total bytes of this layout's address space — the denominator of the resident fraction.
  const structureBytes =
    layout === 'array'
      ? N * ENTRY
      : layout === 'btree'
        ? bt.counts.reduce((a, c) => a + c * P, 0)
        : layout === 'veb'
          ? dataBase + (N * ENTRY) / PMA_DENSITY
          : ls.total;

  const run = (touch: Touch) => {
    let t = 0;
    for (const op of ops) {
      if (layout === 'array') {
        let lo = 0;
        let hi = N - 1;
        while (lo <= hi) {
          const mid = lo + Math.floor((hi - lo) / 2);
          t += touchRange(touch, mid * ENTRY, ENTRY, B);
          if (mid < op.rank) lo = mid + 1;
          else if (mid > op.rank) hi = mid - 1;
          else break;
        }
        if (scan) t += touchRange(touch, op.rank * ENTRY, scanBytes, B);
      } else if (layout === 'btree') {
        const s = bt;
        const leaf = Math.min(s.counts[0] - 1, Math.floor(op.rank / s.fanout));
        for (let lvl = s.counts.length - 1; lvl >= 0; lvl--) {
          const idx = Math.min(s.counts[lvl] - 1, Math.floor(leaf / s.fanout ** lvl));
          t += touchRange(touch, s.base[lvl] + idx * P, P, B);
        }
        if (scan) {
          const pages = Math.max(1, Math.ceil(SCAN_K / s.fanout));
          const span = Math.min(pages, Math.max(1, s.counts[0] - leaf));
          t += touchRange(touch, s.base[0] + leaf * P, span * P, B);
        }
      } else if (layout === 'veb') {
        for (let d = 0; d < h; d++) {
          const slot = vebSlot(op.bits.slice(0, d), 0, h, 0);
          t += touchRange(touch, slot * ENTRY, ENTRY, B);
        }
        const dataOff = dataBase + Math.floor((op.rank * ENTRY) / PMA_DENSITY);
        t += touchRange(touch, dataOff, ENTRY, B);
        if (scan) t += touchRange(touch, dataOff, scanBytes / PMA_DENSITY, B);
      } else {
        const s = ls;
        if (scan) {
          for (let i = 0; i < s.levels; i++) {
            const share = (s.entries[i] / N) * scanBytes;
            const off = s.base[i] + Math.floor((op.rank / N) * s.entries[i]) * ENTRY;
            t += touchRange(touch, off, Math.max(P, share), B);
          }
        } else {
          // Bloom filters and fence pointers are resident; one data block per probed level.
          let acc = 0;
          let chosen = s.levels - 1;
          for (let i = s.levels - 1; i >= 0; i--) {
            acc += s.entries[i] / N;
            if (op.levelDraw <= acc) {
              chosen = i;
              break;
            }
          }
          for (let i = 0; i < s.levels; i++) {
            if (i !== chosen && op.fp[i % op.fp.length] > FPR) continue;
            const off = s.base[i] + Math.floor((op.rank / N) * s.entries[i]) * ENTRY;
            t += touchRange(touch, Math.floor(off / P) * P, P, B);
          }
        }
      }
    }
    return t;
  };

  // The workload is a sample, so an LRU sized M/B would hold all of it. Size the
  // simulated cache to the same *fraction* of the structure that M is of the real
  // one: LRU then keeps the genuinely hot blocks (upper levels) and misses the rest.
  const seen = new Set<number>();
  run((id) => {
    const had = seen.has(id);
    seen.add(id);
    return had;
  });
  const cap = Math.max(2, Math.round((seen.size * M) / Math.max(1, structureBytes)));
  const touch = makeCache(cap);
  run(touch); // warm to steady state
  const transfers = run(touch);
  return { transfers: transfers / ops.length, bytes: (transfers / ops.length) * B };
}

/* ------------------------------------------------------------------ analytic */

function analytic(layout: LayoutId, N: number, M: number, B: number, P: number, scan: boolean) {
  const perBlock = Math.max(2, B / ENTRY);
  const fanout = Math.max(2, Math.floor((P - HEADER) / ENTRY));
  const depth = Math.max(1, Math.ceil(Math.log(N) / Math.log(fanout)));
  const blocksPerPage = Math.max(1, P / B);
  const scanBlocks = scan ? (SCAN_K * ENTRY) / B : 0;
  const { levels } = lsmShape(N, M);
  switch (layout) {
    case 'array':
      return Math.max(1, Math.log2(N) - Math.log2(perBlock)) + scanBlocks;
    case 'btree':
      return depth * blocksPerPage + scanBlocks;
    case 'veb':
      return Math.max(1, Math.log(N) / Math.log(perBlock)) + 1 + scanBlocks / PMA_DENSITY;
    default:
      return blocksPerPage * (1 + FPR * (levels - 1)) + (scan ? scanBlocks + levels : 0);
  }
}

/* -------------------------------------------------------------------- figure */

type Point = { P: number; measured: number; analytic: number };
type Series = { id: LayoutId; label: string; color: string; tag: string; points: Point[] };

function Chart({
  series,
  avail,
  pageExp,
  scan,
}: {
  series: Series[];
  avail: number;
  pageExp: number;
  scan: boolean;
}) {
  const tip = useTip();
  const W = Math.max(360, Math.min(avail, 780));
  const H = 300;
  const pad = { l: 46, r: 108, t: 14, b: 38 };
  const iw = W - pad.l - pad.r;
  const ih = H - pad.t - pad.b;

  const all = series.flatMap((s) => s.points.flatMap((p) => [p.measured, p.analytic]));
  const hi = Math.max(4, ...all) * 1.35;
  const lo = 0.7;
  const x = (i: number) => pad.l + (i * iw) / (PAGE_EXPS.length - 1);
  const y = (v: number) =>
    pad.t + ih - ((Math.log(Math.max(lo, v)) - Math.log(lo)) / (Math.log(hi) - Math.log(lo))) * ih;

  const ticks = [1, 2, 3, 5, 10, 20, 30, 50, 100, 200, 300, 500, 1000, 2000, 5000].filter(
    (t) => t <= hi,
  );
  const here = PAGE_EXPS.indexOf(pageExp);

  // right-hand labels, nudged apart so they never collide
  const labels = series
    .map((s) => ({ s, y: y(s.points[s.points.length - 1].measured) }))
    .sort((a, b) => a.y - b.y);
  for (let i = 1; i < labels.length; i++) {
    if (labels[i].y - labels[i - 1].y < 14) labels[i].y = labels[i - 1].y + 14;
  }

  return (
    <svg width={W} height={H} role="img" aria-label="Block transfers per operation against the engine's page size">
      {ticks.map((t) => (
        <g key={t}>
          <line className="viz-grid-line" x1={pad.l} x2={pad.l + iw} y1={y(t)} y2={y(t)} />
          <text x={pad.l - 8} y={y(t) + 4} textAnchor="end" fill="var(--viz-ink-muted)">
            {t}
          </text>
        </g>
      ))}
      <line className="viz-axis-line" x1={pad.l} x2={pad.l + iw} y1={pad.t + ih} y2={pad.t + ih} />
      <rect x={x(here) - 9} y={pad.t} width={18} height={ih} fill="var(--viz-neutral)" opacity={0.85} />
      {PAGE_EXPS.map((e, i) => (
        <text
          key={e}
          x={x(i)}
          y={pad.t + ih + 16}
          textAnchor="middle"
          fill={e === pageExp ? 'var(--viz-ink)' : 'var(--viz-ink-muted)'}
        >
          {fmtBytes(2 ** e)}
        </text>
      ))}
      <text x={pad.l} y={H - 6} fill="var(--viz-ink-2)">
        engine page size P
      </text>
      <text x={pad.l - 8} y={pad.t - 2} textAnchor="end" fill="var(--viz-ink-2)">
        I/Os
      </text>

      {series.map((s) => (
        <g key={s.id}>
          <polyline
            fill="none"
            stroke={s.color}
            strokeWidth={1.25}
            strokeDasharray="4 3"
            opacity={0.5}
            points={s.points.map((p, i) => `${x(i)},${y(p.analytic)}`).join(' ')}
          />
          <polyline
            fill="none"
            stroke={s.color}
            strokeWidth={2}
            points={s.points.map((p, i) => `${x(i)},${y(p.measured)}`).join(' ')}
          />
          {s.points.map((p, i) => (
            <circle
              key={p.P}
              cx={x(i)}
              cy={y(p.measured)}
              r={i === here ? 5 : 3.2}
              fill={s.color}
              stroke="var(--viz-surface)"
              strokeWidth={i === here ? 1.5 : 0}
              {...tip(
                <>
                  <strong>{s.label}</strong>
                  <br />
                  page size {fmtBytes(p.P)} — measured {p.measured.toFixed(2)} I/Os per{' '}
                  {scan ? 'scan' : 'lookup'}
                  <br />
                  analytic (cold cache) {p.analytic.toFixed(2)}
                </>,
              )}
            />
          ))}
        </g>
      ))}

      {labels.map(({ s, y: ly }) => (
        <text key={s.id} x={pad.l + iw + 8} y={ly + 4} fill={s.color}>
          {s.tag}
        </text>
      ))}
    </svg>
  );
}

/* --------------------------------------------------------------------- panel */

export default function ExternalMemoryCostLab() {
  const [nExp, setNExp] = useState(8); // 10^8 records
  const [blockExp, setBlockExp] = useState(12); // hardware transfer unit B = 4 KB
  const [pageExp, setPageExp] = useState(13); // engine page size P = 8 KB
  const [memExp, setMemExp] = useState(28); // M = 256 MB
  const [scan, setScan] = useState<'point' | 'scan'>('point');
  const [ref, width] = useSize(760);

  const N = Math.round(10 ** nExp);
  const B = 2 ** blockExp;
  const P = 2 ** pageExp;
  const M = 2 ** memExp;
  const isScan = scan === 'scan';

  const series = useMemo<Series[]>(() => {
    const hMax = 30;
    const ops = makeOps(N, OPS, hMax, 0x5eed);
    return LAYOUTS.map((l) => ({
      id: l.id,
      label: l.label,
      color: l.color,
      tag: l.tag,
      points: PAGE_EXPS.map((e) => {
        const pp = 2 ** e;
        return {
          P: pp,
          measured: simulate(l.id, ops, N, M, B, pp, isScan).transfers,
          analytic: analytic(l.id, N, M, B, pp, isScan),
        };
      }),
    }));
  }, [N, M, B, isScan]);

  const at = (id: LayoutId) => series.find((s) => s.id === id)!.points[PAGE_EXPS.indexOf(pageExp)];
  const bt = at('btree');
  const veb = at('veb');
  const arr = at('array');
  const lsm = at('lsm');
  const btPoints = series.find((s) => s.id === 'btree')!.points;
  const best = btPoints.reduce((a, b) => (b.measured < a.measured ? b : a));
  const fanout = Math.max(2, Math.floor((P - HEADER) / ENTRY));
  const depth = btreeShape(N, P).depth;

  return (
    <VizPanel
      title="The external-memory model, run rather than asserted"
      subtitle="One key workload, four layouts, every byte offset simulated against an LRU cache of M/B blocks. Solid = measured transfers; dashed = the analytic bound with a cold cache."
      controls={
        <>
          <Slider
            label="Records N"
            min={5}
            max={9}
            step={0.5}
            value={nExp}
            onChange={setNExp}
            format={() => short(N)}
          />
          <Slider
            label="Transfer unit B"
            min={9}
            max={16}
            value={blockExp}
            onChange={setBlockExp}
            format={() => fmtBytes(B)}
          />
          <Slider
            label="Engine page P"
            min={9}
            max={16}
            value={pageExp}
            onChange={setPageExp}
            format={() => fmtBytes(P)}
          />
          <Slider
            label="Memory M"
            min={22}
            max={33}
            value={memExp}
            onChange={setMemExp}
            format={() => fmtBytes(M)}
          />
          <Segmented
            label="Workload"
            value={scan}
            onChange={setScan}
            options={[
              { value: 'point', label: 'Point lookup' },
              { value: 'scan', label: `Range scan (${SCAN_K})` },
            ]}
          />
        </>
      }
      legend={<Legend items={LAYOUTS.map((l) => ({ label: l.label, color: l.color, shape: 'line' as const }))} />}
      stats={
        <Stats
          items={[
            {
              label: `B+tree I/Os per ${isScan ? 'scan' : 'lookup'}`,
              value: bt.measured.toFixed(2),
              hint: `${depth} levels, fanout ${fanout}, ${Math.max(1, Math.ceil(P / B))} transfer(s) per node`,
            },
            { label: 'vEB I/Os', value: veb.measured.toFixed(2), hint: 'Cache-oblivious: no page size to configure' },
            { label: 'Binary search I/Os', value: arr.measured.toFixed(2) },
            { label: 'LSM I/Os', value: lsm.measured.toFixed(2) },
            {
              label: 'Best P for this B',
              value: fmtBytes(best.P),
              hint: 'Argmin of the measured B+tree curve',
            },
            { label: 'Blocks resident (M/B)', value: fmtNum(Math.floor(M / B)) },
          ]}
        />
      }
      note={
        <Note>
          <strong>
            {isScan
              ? `Scanning ${SCAN_K} records costs at least ${fmtNum((SCAN_K * ENTRY) / B, 1)} transfers — that is k/B, and no layout beats it.`
              : `At B = ${fmtBytes(B)} the B+tree's cheapest page size is ${fmtBytes(best.P)}.`}
          </strong>{' '}
          The B+tree pays <em>depth × ⌈P/B⌉</em>: pages smaller than B waste the transfer, pages larger
          than B cost several. Its optimum sits at P ≈ B and slides the moment you move the B slider.
          The van Emde Boas line is flat in P because it has no P — the recursive layout puts each
          root-to-leaf path inside ⌈log<sub>B/{ENTRY}</sub> N⌉ blocks for <em>every</em> B at once. Binary
          search over a sorted array is the control: log₂ N probes, of which only the last few land in
          the same block, so it stays several times worse than either tree no matter how big B gets.
        </Note>
      }
      table={
        <table className="viz-table">
          <thead>
            <tr>
              <th>Page size P</th>
              {LAYOUTS.map((l) => (
                <th key={l.id}>{l.tag}</th>
              ))}
              <th>B+tree analytic</th>
            </tr>
          </thead>
          <tbody>
            {PAGE_EXPS.map((e, i) => (
              <tr key={e}>
                <td>{fmtBytes(2 ** e)}</td>
                {series.map((s) => (
                  <td key={s.id}>{s.points[i].measured.toFixed(2)}</td>
                ))}
                <td>{btPoints[i].analytic.toFixed(2)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      }
    >
      <div ref={ref}>
        <TooltipHost>
          <Chart series={series} avail={width} pageExp={pageExp} scan={isScan} />
        </TooltipHost>
      </div>
    </VizPanel>
  );
}
