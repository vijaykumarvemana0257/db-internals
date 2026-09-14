import { useMemo, useState } from 'react';
import {
  VizPanel,
  Choice,
  Slider,
  Check,
  Legend,
  Stats,
  Note,
  TooltipHost,
  useTip,
  makeRng,
  useSize,
} from './Viz';

/**
 * Mattson's stack algorithm, and the policies it does not describe.
 *
 * One pass over a trace records each reference's *stack distance* — the number of
 * distinct pages touched since that page was last touched. For any stack algorithm the
 * contents of a cache of size k are a subset of the contents of size k+1, so a reference
 * with distance d hits at every size k > d and misses at every size k <= d. The whole
 * miss-ratio curve therefore falls out of one histogram. CLOCK, ARC and friends are not
 * stack algorithms: their internal state depends on the cache size, so every size has to
 * be simulated separately, and the simulated curves are not even guaranteed monotone.
 */

type Workload = 'hot' | 'loop' | 'mixed' | 'random';
const TRACE_LEN = 1600;
const MAX_K = 48;

const WORKLOADS: { value: Workload; label: string }[] = [
  { value: 'mixed', label: 'Hot set + looping scan' },
  { value: 'hot', label: 'Zipf-ish hot set' },
  { value: 'loop', label: 'Pure looping scan' },
  { value: 'random', label: 'Uniform random (40 pages)' },
];

function buildTrace(kind: Workload, loopLen: number): number[] {
  const rng = makeRng(0x5ea1 + loopLen * 31 + kind.length);
  const t: number[] = [];
  for (let i = 0; i < TRACE_LEN; i++) {
    if (kind === 'loop') {
      t.push(i % loopLen);
    } else if (kind === 'random') {
      t.push(Math.floor(rng() * 40));
    } else if (kind === 'hot') {
      // Zipf-ish: rank r chosen so that small ranks dominate.
      const u = rng();
      const rank = Math.floor(Math.pow(u, 3) * 60);
      t.push(rank);
    } else {
      // Half the references hit a 6-page hot set; the other half walk a cyclic scan.
      if (rng() < 0.5) t.push(Math.floor(rng() * 6));
      else t.push(100 + (i % loopLen));
    }
  }
  return t;
}

/* ------------------------------------------- one pass: Mattson stack distances */

type Rd = { hist: number[]; cold: number; n: number };

/**
 * Stack distance = number of *distinct* pages referenced since this page's last
 * reference. A first-ever reference has distance infinity (a compulsory miss).
 */
function reuseDistances(trace: number[]): Rd {
  const stack: number[] = []; // index 0 = most recently used
  const hist = new Array<number>(MAX_K + 2).fill(0);
  let cold = 0;
  for (const p of trace) {
    const at = stack.indexOf(p);
    if (at < 0) cold++;
    else {
      hist[Math.min(at, MAX_K + 1)]++;
      stack.splice(at, 1);
    }
    stack.unshift(p);
  }
  return { hist, cold, n: trace.length };
}

/** Miss ratio predicted for LRU at cache size k: everything with distance >= k misses. */
function mrcFromDistances(rd: Rd, k: number): number {
  let hits = 0;
  for (let d = 0; d < k && d < rd.hist.length; d++) hits += rd.hist[d];
  return (rd.n - hits) / rd.n;
}

/* --------------------------------------------- per-size simulation of policies */

function missLru(trace: number[], k: number): number {
  const order: number[] = [];
  let misses = 0;
  for (const p of trace) {
    const at = order.indexOf(p);
    if (at >= 0) order.splice(at, 1);
    else {
      misses++;
      if (order.length >= k) order.pop();
    }
    order.unshift(p);
  }
  return misses / trace.length;
}

function missOpt(trace: number[], k: number): number {
  const next = new Array<number>(trace.length).fill(Infinity);
  const seen = new Map<number, number>();
  for (let i = trace.length - 1; i >= 0; i--) {
    next[i] = seen.has(trace[i]) ? (seen.get(trace[i]) as number) : Infinity;
    seen.set(trace[i], i);
  }
  const res = new Map<number, number>();
  let misses = 0;
  for (let i = 0; i < trace.length; i++) {
    const p = trace[i];
    if (!res.has(p)) {
      misses++;
      if (res.size >= k) {
        let worst = -1;
        let worstAt = -1;
        for (const [q, at] of res) if (at > worstAt) ((worstAt = at), (worst = q));
        res.delete(worst);
      }
    }
    res.set(p, next[i]);
  }
  return misses / trace.length;
}

function missClock(trace: number[], k: number): number {
  const page = new Array<number>(k).fill(-1);
  const ref = new Array<number>(k).fill(0);
  const where = new Map<number, number>();
  let hand = 0;
  let fill = 0;
  let misses = 0;
  for (const p of trace) {
    const at = where.get(p);
    if (at !== undefined) {
      ref[at] = 1;
      continue;
    }
    misses++;
    if (fill < k) {
      page[fill] = p;
      ref[fill] = 1;
      where.set(p, fill);
      fill++;
      continue;
    }
    while (ref[hand] === 1) {
      ref[hand] = 0;
      hand = (hand + 1) % k;
    }
    where.delete(page[hand]);
    page[hand] = p;
    ref[hand] = 1;
    where.set(p, hand);
    hand = (hand + 1) % k;
  }
  return misses / trace.length;
}

/** ARC (Megiddo & Modha, FAST '03). Its state depends on k, so it has no stack property. */
function missArc(trace: number[], k: number): number {
  const T1: number[] = [];
  const T2: number[] = [];
  const B1: number[] = [];
  const B2: number[] = [];
  let p = 0;
  let misses = 0;
  const drop = (l: number[], x: number) => {
    const i = l.indexOf(x);
    if (i >= 0) l.splice(i, 1);
  };
  const replace = (inB2: boolean) => {
    if (T1.length > 0 && (T1.length > p || (inB2 && T1.length === p))) B1.push(T1.shift() as number);
    else if (T2.length > 0) B2.push(T2.shift() as number);
  };
  for (const x of trace) {
    if (T1.includes(x)) {
      drop(T1, x);
      T2.push(x);
    } else if (T2.includes(x)) {
      drop(T2, x);
      T2.push(x);
    } else if (B1.includes(x)) {
      misses++;
      p = Math.min(k, p + (B1.length >= B2.length ? 1 : B2.length / B1.length));
      replace(false);
      drop(B1, x);
      T2.push(x);
    } else if (B2.includes(x)) {
      misses++;
      p = Math.max(0, p - (B2.length >= B1.length ? 1 : B1.length / B2.length));
      replace(true);
      drop(B2, x);
      T2.push(x);
    } else {
      misses++;
      if (T1.length + B1.length === k) {
        if (T1.length < k) {
          B1.shift();
          replace(false);
        } else T1.shift();
      } else if (T1.length + T2.length + B1.length + B2.length >= k) {
        if (T1.length + T2.length + B1.length + B2.length === 2 * k) B2.shift();
        replace(false);
      }
      T1.push(x);
    }
  }
  return misses / trace.length;
}

/* ------------------------------------------------------------- the component */

const SERIES = [
  { id: 'opt', label: 'OPT (Belady MIN)', color: 'var(--viz-1)' },
  { id: 'lru', label: 'LRU, simulated at every size', color: 'var(--viz-2)' },
  { id: 'clock', label: 'CLOCK, simulated at every size', color: 'var(--viz-3)' },
  { id: 'arc', label: 'ARC, simulated at every size', color: 'var(--viz-4)' },
] as const;

type Row = { k: number; opt: number; lru: number; clock: number; arc: number; mrc: number };

export default function ReuseDistanceMrcLab() {
  const [workload, setWorkload] = useState<Workload>('mixed');
  const [loopLen, setLoopLen] = useState(18);
  const [poolK, setPoolK] = useState(12);
  const [showClock, setShowClock] = useState(true);
  const [showArc, setShowArc] = useState(true);
  const [ref, width] = useSize(760);
  const tip = useTip();

  const trace = useMemo(() => buildTrace(workload, loopLen), [workload, loopLen]);
  const rd = useMemo(() => reuseDistances(trace), [trace]);
  const rows: Row[] = useMemo(
    () =>
      Array.from({ length: MAX_K }, (_, j) => {
        const k = j + 1;
        return {
          k,
          opt: missOpt(trace, k),
          lru: missLru(trace, k),
          clock: missClock(trace, k),
          arc: missArc(trace, k),
          mrc: mrcFromDistances(rd, k),
        };
      }),
    [trace, rd],
  );

  const maxDev = rows.reduce((m, r) => Math.max(m, Math.abs(r.mrc - r.lru)), 0);
  /** Belady's anomaly, live: sizes at which one more frame produced *more* misses. */
  const anomalies = rows.reduce(
    (acc, r, j) =>
      j === 0
        ? acc
        : acc +
          (r.clock > rows[j - 1].clock + 1e-9 ? 1 : 0) +
          (r.arc > rows[j - 1].arc + 1e-9 ? 1 : 0),
    0,
  );
  const clockDev = rows.reduce((m, r) => Math.max(m, Math.abs(r.mrc - r.clock)), 0);
  const arcDev = rows.reduce((m, r) => Math.max(m, Math.abs(r.mrc - r.arc)), 0);
  const here = rows[Math.min(poolK, MAX_K) - 1];

  /** The knee: the smallest size that gets within 2 points of the best LRU ever does. */
  const floor = rows[MAX_K - 1].lru;
  const knee = rows.find((r) => r.lru <= floor + 0.02)?.k ?? MAX_K;
  /** The steepest single-frame improvement — the cliff edge. */
  const cliff = rows.reduce(
    (best, r, j) => {
      const drop = j === 0 ? 0 : rows[j - 1].lru - r.lru;
      return drop > best.drop ? { k: r.k, drop } : best;
    },
    { k: 1, drop: 0 },
  );

  /* geometry: histogram on top, MRC below, sharing the x axis */
  const padL = 54;
  const padR = 128;
  const W = Math.max(560, Math.min(width, 900));
  const plotW = W - padL - padR;
  const histH = 76;
  const mrcH = 210;
  const gap = 26;
  const topY = 14;
  const mrcY = topY + histH + gap;
  const H = mrcY + mrcH + 34;
  const xOf = (k: number) => padL + ((k - 0.5) / MAX_K) * plotW;
  const yMrc = (v: number) => mrcY + (1 - v) * mrcH;
  const histMax = Math.max(1, ...rd.hist.slice(0, MAX_K));
  const yHist = (c: number) => topY + histH - (c / histMax) * histH;

  const path = (pick: (r: Row) => number) =>
    rows.map((r, j) => `${j === 0 ? 'M' : 'L'}${xOf(r.k).toFixed(1)},${yMrc(pick(r)).toFixed(1)}`).join(' ');

  const shown = [
    { ...SERIES[0], on: true, pick: (r: Row) => r.opt },
    { ...SERIES[1], on: true, pick: (r: Row) => r.lru },
    { ...SERIES[2], on: showClock, pick: (r: Row) => r.clock },
    { ...SERIES[3], on: showArc, pick: (r: Row) => r.arc },
  ].filter((s) => s.on);

  return (
    <VizPanel
      title="One pass, the whole curve — and the policies it cannot predict"
      subtitle="The reuse-distance histogram (top) is computed in a single sweep of the trace. Its reverse cumulative sum is the LRU miss-ratio curve (bottom), exactly. Simulated CLOCK and ARC are drawn over it; they are not stack algorithms, so they had to be run once per size."
      controls={
        <>
          <Choice
            label="Workload"
            value={workload}
            onChange={setWorkload}
            options={WORKLOADS.map((w) => ({ value: w.value, label: w.label }))}
          />
          <Slider
            label="Loop length"
            min={4}
            max={40}
            value={loopLen}
            onChange={setLoopLen}
            disabled={workload === 'hot' || workload === 'random'}
            format={(n) => `${n} pages`}
          />
          <Slider label="Pool size" min={1} max={MAX_K} value={poolK} onChange={setPoolK} format={(n) => `${n} frames`} />
          <Check label="CLOCK" checked={showClock} onChange={setShowClock} />
          <Check label="ARC" checked={showArc} onChange={setShowArc} />
        </>
      }
      legend={
        <Legend
          items={[
            { label: 'Reuse distances inside the pool (would hit)', color: 'var(--viz-seq-400)' },
            { label: 'Reuse distances beyond it (would miss)', color: 'var(--viz-seq-100)' },
            ...SERIES.map((s) => ({ label: s.label, color: s.color, shape: 'line' as const })),
            { label: 'MRC predicted from distances (○, LRU only)', color: 'var(--viz-ink-2)', shape: 'dot' as const },
          ]}
        />
      }
      stats={
        <Stats
          items={[
            { label: `Miss ratio @ ${here.k} frames`, value: `${(here.lru * 100).toFixed(1)}%`, hint: 'Measured by simulating LRU at exactly this size' },
            { label: 'MRC prediction', value: `${(here.mrc * 100).toFixed(1)}%`, hint: 'Read off the one-pass reuse-distance histogram — no simulation' },
            { label: 'LRU prediction error', value: maxDev < 5e-9 ? 'exactly 0' : maxDev.toExponential(1), hint: 'Largest gap between prediction and measurement across all 48 sizes' },
            { label: 'CLOCK vs the curve', value: `${(clockDev * 100).toFixed(1)} pts`, hint: 'Largest gap: CLOCK has no stack property, so the LRU curve does not describe it' },
            { label: 'ARC vs the curve', value: `${(arcDev * 100).toFixed(1)} pts`, hint: 'ARC keeps size-dependent state (p, B1, B2) — simulate per size or measure' },
            { label: 'Steepest frame', value: `${cliff.k} → ${(cliff.drop * 100).toFixed(1)} pts`, hint: 'The single frame that buys the most: the edge of the cliff' },
            { label: 'Working set (knee)', value: `${knee} frames`, hint: 'Smallest pool within 2 points of the best miss ratio LRU reaches here' },
            {
              label: 'Non-monotone points',
              value: anomalies,
              hint: "Sizes where CLOCK or ARC got worse with one more frame — Belady's anomaly. A stack algorithm can never do this.",
            },
          ]}
        />
      }
      note={
        <Note>
          <strong>
            {maxDev < 5e-9
              ? 'The one-pass prediction and the simulated LRU curve are identical — that is the stack property.'
              : 'Prediction and measurement have diverged.'}
          </strong>{' '}
          {workload === 'loop'
            ? `A cyclic scan over ${loopLen} pages puts every one of its references at exactly the same reuse distance, ${loopLen - 1}, so the histogram is one spike and the curve is a cliff: below ${loopLen} frames LRU evicts each page exactly one reference before it is needed again, at ${loopLen} frames it keeps them all. Drag the loop past the pool size and watch the miss ratio at ${poolK} frames jump.`
            : workload === 'mixed'
              ? `Half the references walk a ${loopLen}-page cycle and half land on a 6-page hot set, so the loop's reuse distances cluster just above ${loopLen} and the cliff only bottoms out around ${loopLen + 6} frames — the loop plus the hot pages it shares the pool with. Drag the loop past the pool marker and watch the miss ratio at ${poolK} frames jump.`
              : `Skewed reuse spreads the histogram out, so the curve bends instead of falling off. The knee at ${knee} frames is the working set; past it you are buying almost nothing.`}{' '}
          {anomalies > 0
            ? `CLOCK and ARC track the curve loosely but never lie on it, and here they are not even monotone: at ${anomalies} size${anomalies === 1 ? '' : 's'} one extra frame produced more misses — Belady's anomaly, which a stack algorithm cannot exhibit.`
            : 'CLOCK and ARC track the curve loosely but never lie on it, which is why their curves have to be measured at the size you actually plan to run.'}
        </Note>
      }
      table={
        <table className="viz-table">
          <thead>
            <tr>
              <th>Frames</th>
              <th>Refs at this reuse distance</th>
              <th>MRC predicted</th>
              <th>LRU measured</th>
              <th>CLOCK measured</th>
              <th>ARC measured</th>
              <th>OPT measured</th>
            </tr>
          </thead>
          <tbody>
            {rows
              .filter((r) => r.k <= 8 || r.k % 4 === 0 || r.k === poolK)
              .map((r) => (
                <tr key={r.k}>
                  <td>{r.k}</td>
                  <td>{rd.hist[r.k - 1]}</td>
                  <td>{(r.mrc * 100).toFixed(2)}%</td>
                  <td>{(r.lru * 100).toFixed(2)}%</td>
                  <td>{(r.clock * 100).toFixed(2)}%</td>
                  <td>{(r.arc * 100).toFixed(2)}%</td>
                  <td>{(r.opt * 100).toFixed(2)}%</td>
                </tr>
              ))}
            <tr>
              <td>—</td>
              <td>{rd.cold} compulsory (distance ∞)</td>
              <td colSpan={5}>First reference to a page: a miss at every cache size, the floor no policy can go below.</td>
            </tr>
          </tbody>
        </table>
      }
    >
      <div ref={ref}>
        <TooltipHost>
          <svg width={W} height={H} role="img" aria-label="Reuse-distance histogram above a miss-ratio curve with four policies">
            {/* ---- histogram ---- */}
            <text x={0} y={topY - 2} fill="var(--viz-ink)" fontWeight={600}>
              References per reuse distance
            </text>
            <line x1={padL} x2={padL + plotW} y1={topY + histH} y2={topY + histH} className="viz-axis-line" />
            {rd.hist.slice(0, MAX_K).map((c, d) => {
              const bw = Math.max(2, plotW / MAX_K - 2);
              return (
                <g key={d} {...tip(<>reuse distance {d}: {c} references — they hit in any LRU pool of {d + 1} frames or more</>)}>
                  <rect
                    x={xOf(d + 1) - bw / 2}
                    y={yHist(c)}
                    width={bw}
                    height={Math.max(0, topY + histH - yHist(c))}
                    fill={d + 1 <= poolK ? 'var(--viz-seq-400)' : 'var(--viz-seq-100)'}
                  />
                </g>
              );
            })}
            <text x={padL + plotW + 8} y={topY + 12} fill="var(--viz-ink-2)">
              {rd.cold} compulsory
            </text>
            <text x={padL + plotW + 8} y={topY + 26} fill="var(--viz-ink-muted)">
              (distance ∞)
            </text>

            {/* ---- MRC ---- */}
            <text x={0} y={mrcY - 8} fill="var(--viz-ink)" fontWeight={600}>
              Miss ratio
            </text>
            {[0, 0.25, 0.5, 0.75, 1].map((v) => (
              <g key={v}>
                <line x1={padL} x2={padL + plotW} y1={yMrc(v)} y2={yMrc(v)} className="viz-grid-line" />
                <text x={padL - 8} y={yMrc(v) + 4} textAnchor="end" fill="var(--viz-ink-muted)">
                  {(v * 100).toFixed(0)}%
                </text>
              </g>
            ))}
            {[1, 8, 16, 24, 32, 40, 48].map((k) => (
              <text key={k} x={xOf(k)} y={mrcY + mrcH + 16} textAnchor="middle" fill="var(--viz-ink-muted)">
                {k}
              </text>
            ))}
            <text x={padL + plotW / 2} y={mrcY + mrcH + 30} textAnchor="middle" fill="var(--viz-ink-2)">
              buffer pool size (frames)
            </text>

            {/* pool-size marker */}
            <line x1={xOf(poolK)} x2={xOf(poolK)} y1={topY} y2={mrcY + mrcH} stroke="var(--viz-ink-2)" strokeDasharray="4 3" />
            <text x={xOf(poolK) + 5} y={mrcY + 12} fill="var(--viz-ink-2)">
              pool = {poolK}
            </text>

            {/* predicted MRC as open circles, drawn under the lines */}
            {rows
              .filter((r) => r.k % 2 === 1)
              .map((r) => (
                <circle
                  key={`p${r.k}`}
                  cx={xOf(r.k)}
                  cy={yMrc(r.mrc)}
                  r={4}
                  fill="none"
                  stroke="var(--viz-ink-2)"
                  strokeWidth={1.2}
                />
              ))}

            {shown.map((s) => (
              <path key={s.id} d={path(s.pick)} fill="none" stroke={s.color} strokeWidth={2} />
            ))}

            {/* direct end labels so identity is never colour alone */}
            {(() => {
              const placed: number[] = [];
              return shown
                .map((s) => ({ s, y: yMrc(s.pick(rows[MAX_K - 1])) }))
                .sort((a, b) => a.y - b.y)
                .map(({ s, y }) => {
                  const last = placed.length ? placed[placed.length - 1] : -Infinity;
                  const ly = Math.max(y, last + 15);
                  placed.push(ly);
                  return (
                    <g key={`l${s.id}`}>
                      <line x1={padL + plotW + 6} x2={padL + plotW + 22} y1={ly} y2={ly} stroke={s.color} strokeWidth={2.5} />
                      <text x={padL + plotW + 27} y={ly + 4} fill="var(--viz-ink)" fontWeight={600}>
                        {s.id.toUpperCase()}
                      </text>
                    </g>
                  );
                });
            })()}

            {/* hover targets over the whole plot */}
            {rows.map((r) => (
              <rect
                key={`h${r.k}`}
                x={xOf(r.k) - plotW / MAX_K / 2}
                y={mrcY}
                width={plotW / MAX_K}
                height={mrcH}
                fill="transparent"
                {...tip(
                  <>
                    <strong>{r.k} frames</strong>
                    <br />
                    predicted {(r.mrc * 100).toFixed(2)}% · LRU {(r.lru * 100).toFixed(2)}%
                    <br />
                    CLOCK {(r.clock * 100).toFixed(2)}% · ARC {(r.arc * 100).toFixed(2)}% · OPT {(r.opt * 100).toFixed(2)}%
                  </>,
                )}
              />
            ))}
          </svg>
        </TooltipHost>
      </div>
    </VizPanel>
  );
}
