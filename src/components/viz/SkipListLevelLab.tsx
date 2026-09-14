import { useMemo, useState } from 'react';
import {
  VizPanel,
  Segmented,
  Slider,
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
 * A skip list, built one coin flip at a time.
 *
 * Every node is a tower: it is always in the level-0 list, and it is promoted to the next
 * level up with probability p. Nothing enforces the shape — there is no invariant to
 * restore, no rotation, no split — so the structure is correct for *any* draw of heights
 * and only its cost depends on them.
 *
 * Search is LevelDB's FindGreaterOrEqual: at each level walk forward while the next key is
 * strictly less than the target, then drop a level. The staircase the learner sees is the
 * whole algorithm.
 *
 * The cost curve measures the same model at sizes up to 65 536 keys and plots it against
 * Pugh's bound L(n)/p + 1/(1-p), so the analytic line and the sampled means sit on the
 * same axes — including the p99, which is where probabilistic balance shows its teeth.
 */

const MAX_H = 12; // LevelDB's kMaxHeight; RocksDB's InlineSkipList default max_height
const KEY = (i: number) => i * 2; // node i holds key 2i, so every odd target is a miss
const CURVE_E = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16];

/* ------------------------------------------------------------------- model */

/** height h with P(h > k) = p^k: flip a biased coin until it comes up tails. */
function drawHeights(n: number, p: number, seed: number) {
  const rng = makeRng(seed);
  const h: number[] = [];
  for (let i = 0; i < n; i++) {
    let k = 1;
    while (k < MAX_H && rng() < p) k++;
    h.push(k);
  }
  return h;
}

/** next[l][i] = the node after i in the level-l list; index 0 is the head, -1 is NULL. */
function forwards(h: number[], top: number) {
  const n = h.length;
  const next: Int32Array[] = [];
  for (let l = 0; l < top; l++) {
    const arr = new Int32Array(n + 1).fill(-1);
    let last = -1;
    for (let i = n; i >= 1; i--) {
      arr[i] = last;
      if (h[i - 1] > l) last = i;
    }
    arr[0] = last;
    next.push(arr);
  }
  return next;
}

function searchCost(next: Int32Array[], top: number, target: number) {
  let x = 0;
  let cmp = 0;
  for (let l = top - 1; l >= 0; l--) {
    for (;;) {
      const nx = next[l][x];
      if (nx !== -1) cmp++;
      if (nx !== -1 && KEY(nx) < target) {
        x = nx;
        continue;
      }
      break;
    }
  }
  return cmp;
}

type Step = {
  kind: 'advance' | 'drop' | 'land';
  level: number;
  from: number;
  to: number; // -1 = NULL
  cmp: number;
  note: string;
};

type Model = {
  n: number;
  p: number;
  h: number[];
  top: number;
  next: Int32Array[];
  ptrPerNode: number;
  perLevel: number[];
  meanCmp: number;
  maxCmp: number;
  worstKey: number;
};

function buildModel(n: number, p: number, seed: number): Model {
  const h = drawHeights(n, p, seed);
  let top = 1;
  let sum = 0;
  for (const k of h) {
    if (k > top) top = k;
    sum += k;
  }
  const next = forwards(h, top);
  const perLevel = Array.from({ length: top }, (_, l) => h.filter((k) => k > l).length);

  let cmpSum = 0;
  let maxCmp = 0;
  let worstKey = 0;
  for (let i = 1; i <= n; i++) {
    const c = searchCost(next, top, KEY(i));
    cmpSum += c;
    if (c > maxCmp) {
      maxCmp = c;
      worstKey = KEY(i);
    }
  }
  return { n, p, h, top, next, ptrPerNode: sum / n, perLevel, meanCmp: cmpSum / n, maxCmp, worstKey };
}

function searchPath(m: Model, target: number) {
  const steps: Step[] = [];
  let x = 0;
  let cmp = 0;
  for (let l = m.top - 1; l >= 0; l--) {
    for (;;) {
      const nx = m.next[l][x];
      if (nx !== -1) cmp++;
      if (nx !== -1 && KEY(nx) < target) {
        steps.push({
          kind: 'advance',
          level: l,
          from: x,
          to: nx,
          cmp,
          note: `L${l}: ${KEY(nx)} < ${target} — follow the forward pointer.`,
        });
        x = nx;
        continue;
      }
      const why =
        nx === -1
          ? `L${l}: no successor at this level`
          : `L${l}: ${KEY(nx)} ≥ ${target}, so the target is in the gap`;
      steps.push({
        kind: l === 0 ? 'land' : 'drop',
        level: l,
        from: x,
        to: nx,
        cmp,
        note: l === 0 ? `${why} — level 0 is the answer.` : `${why} — drop to L${l - 1} without moving.`,
      });
      break;
    }
  }
  const landed = m.next[0][x];
  const found = landed !== -1 && KEY(landed) === target;
  return { steps, cmp, x, landed, found };
}

/** Pugh's expected search cost: L(n)/p + 1/(1-p), with L(n) = log_{1/p} n. */
function analytic(n: number, p: number) {
  return Math.log(Math.max(n, 2)) / Math.log(1 / p) / p + 1 / (1 - p);
}

type CurvePoint = { n: number; top: number; ptr: number; mean: number; p99: number; max: number; model: number };

function costCurve(p: number, seed: number): CurvePoint[] {
  return CURVE_E.map((e) => {
    const n = 2 ** e;
    const h = drawHeights(n, p, seed + e * 1013);
    let top = 1;
    let sum = 0;
    for (const k of h) {
      if (k > top) top = k;
      sum += k;
    }
    const next = forwards(h, top);
    const stride = Math.max(1, Math.floor(n / 128));
    const costs: number[] = [];
    for (let i = 1; i <= n; i += stride) costs.push(searchCost(next, top, KEY(i)));
    costs.sort((a, b) => a - b);
    const mean = costs.reduce((a, b) => a + b, 0) / costs.length;
    return {
      n,
      top,
      ptr: sum / n,
      mean,
      p99: costs[Math.floor(0.99 * (costs.length - 1))],
      max: costs[costs.length - 1],
      model: analytic(n, p),
    };
  });
}

/* ----------------------------------------------------------------- drawing */

const BRANCH = [
  { value: '2', label: 'p = 1/2', title: "Pugh's original: 2 pointers per node" },
  { value: '4', label: 'p = 1/4', title: 'LevelDB kBranching = 4; RocksDB InlineSkipList branching_factor = 4; Redis ZSKIPLIST_P = 0.25' },
  { value: '8', label: 'p = 1/8', title: 'Fewer pointers, taller walks at each level' },
] as const;

export default function SkipListLevelLab() {
  const [branch, setBranch] = useState<'2' | '4' | '8'>('4');
  const [n, setN] = useState(24);
  const [seed, setSeed] = useState(1);
  const [target, setTarget] = useState(35);
  const [step, setStep] = useState(0);
  const [ref, width] = useSize(760);
  const tip = useTip();

  const p = 1 / Number(branch);
  const m = useMemo(() => buildModel(n, p, seed), [n, p, seed]);
  const run = useMemo(() => searchPath(m, target), [m, target]);
  const curve = useMemo(() => costCurve(p, seed), [p, seed]);

  const shown = Math.min(step, run.steps.length);
  const done = shown >= run.steps.length;
  const cur = shown > 0 ? run.steps[shown - 1] : null;
  const here = cur ? (cur.kind === 'advance' ? cur.to : cur.from) : 0;
  const level = cur ? cur.level : m.top - 1;

  const reset = (fn: () => void) => {
    fn();
    setStep(0);
  };

  /* ---- the list figure: a column per node, a lane per level ---- */
  const headW = 54;
  const padL = 8;
  const colW = Math.max(22, Math.min(40, (width - headW - padL - 14) / n));
  const cellW = Math.max(14, colW - 8);
  const cellH = 15;
  const laneH = 20;
  const topPad = 16;
  const listW = headW + padL + n * colW + 12;
  const listH = topPad + m.top * laneH + 34;
  const laneY = (l: number) => topPad + (m.top - 1 - l) * laneH;
  const colX = (i: number) => (i === 0 ? padL : headW + padL + (i - 1) * colW + (colW - cellW) / 2);
  const colMid = (i: number) => colX(i) + (i === 0 ? headW - 10 : cellW) / 2;

  const taken = run.steps.slice(0, shown);
  const walked = new Set(taken.filter((s) => s.kind === 'advance').map((s) => `${s.level}:${s.from}`));
  const dropped = new Map<number, number>(); // level -> node where the search dropped
  for (const s of taken) if (s.kind !== 'advance') dropped.set(s.level, s.from);

  /* ---- the cost curve ---- */
  const chW = Math.max(320, Math.min(width, 620));
  const chH = 190;
  const mL = 42;
  const mR = 58;
  const mT = 14;
  const mB = 30;
  const yMax = Math.max(...curve.map((c) => Math.max(c.p99, c.model))) * 1.1;
  const cx = (e: number) => mL + ((e - CURVE_E[0]) / (CURVE_E[CURVE_E.length - 1] - CURVE_E[0])) * (chW - mL - mR);
  const cy = (v: number) => chH - mB - (v / yMax) * (chH - mT - mB);
  const line = (get: (c: CurvePoint) => number) =>
    curve.map((c, i) => `${i === 0 ? 'M' : 'L'}${cx(CURVE_E[i]).toFixed(1)},${cy(get(c)).toFixed(1)}`).join(' ');

  return (
    <VizPanel
      title="Coin-flip towers, and the staircase they buy"
      subtitle="Every node is in the level-0 list and is promoted one level higher with probability p. Nothing rebalances; the heights are simply whatever the coins said. Then search: walk forward while the next key is smaller, otherwise drop a level."
      controls={
        <>
          <Segmented
            label="Promotion probability"
            value={branch}
            onChange={(v) => reset(() => setBranch(v))}
            options={BRANCH}
          />
          <Slider label="Keys" min={8} max={40} value={n} onChange={(v) => reset(() => setN(v))} />
          <Slider
            label="Search for"
            min={1}
            max={2 * n + 3}
            value={target}
            onChange={(v) => reset(() => setTarget(v))}
            format={(v) => `${v}${v % 2 === 0 && v <= 2 * n ? '' : ' (absent)'}`}
          />
          <Button onClick={() => setStep((s) => Math.min(s + 1, run.steps.length))} disabled={done} primary>
            Step
          </Button>
          <Button onClick={() => setStep(run.steps.length)} disabled={done}>
            Run search
          </Button>
          <Button onClick={() => setStep(0)} disabled={shown === 0}>
            Rewind
          </Button>
          <Button onClick={() => reset(() => setSeed((s) => s + 1))} title="Draw a fresh set of coin flips — same n, same p, different shape">
            Reflip coins
          </Button>
        </>
      }
      legend={
        <Legend
          items={[
            { label: 'Node tower (one cell per level it reaches)', color: 'var(--viz-1)' },
            { label: 'Forward pointer followed by the search', color: 'var(--viz-2)', shape: 'line' },
            { label: 'Level drop (no move, no pointer followed)', color: 'var(--viz-8)', shape: 'line' },
            { label: 'Pugh bound L(n)/p + 1/(1−p)', color: 'var(--viz-1)', shape: 'line' },
            { label: 'Measured mean comparisons', color: 'var(--viz-2)', shape: 'dot' },
            { label: 'Measured p99 comparisons', color: 'var(--viz-3)', shape: 'dot' },
          ]}
        />
      }
      stats={
        <Stats
          items={[
            { label: 'Levels in use', value: `${m.top}`, hint: `Capped at ${MAX_H} — LevelDB's kMaxHeight` },
            {
              label: 'Pointers / node',
              value: `${m.ptrPerNode.toFixed(2)} (E = ${(1 / (1 - p)).toFixed(2)})`,
              hint: 'Expected pointers per node is 1/(1−p): 2.00 at p=1/2, 1.33 at p=1/4',
            },
            {
              label: 'Comparisons so far',
              value: `${cur ? cur.cmp : 0} / ${run.cmp}`,
              hint: 'Key comparisons in this search; the null-successor check is not one',
            },
            {
              label: 'Expected for this n',
              value: analytic(n, p).toFixed(1),
              hint: 'L(n)/p + 1/(1−p), with L(n) = log_{1/p} n',
            },
            {
              label: 'Worst key in this list',
              value: `${m.maxCmp} (key ${m.worstKey})`,
              hint: `Mean over all ${n} keys is ${m.meanCmp.toFixed(1)} — the spread is what probabilistic balance costs you`,
            },
          ]}
        />
      }
      note={
        <Note>
          {cur ? (
            <>
              <strong>{cur.note}</strong>{' '}
              {done
                ? run.found
                  ? `Found key ${target} after ${run.cmp} comparisons — the search always ends by landing on level 0.`
                  : `Key ${target} is absent: level 0's next key is ${run.landed === -1 ? 'NULL' : KEY(run.landed)}, which is where an insert would splice the new node in. ${run.cmp} comparisons.`
                : 'Press Step again.'}
            </>
          ) : (
            <>
              <strong>Start at the head, at the top level.</strong> The search is two rules — advance while the
              next key is strictly smaller, drop a level otherwise — and it is correct no matter how the coins
              fell. Only the cost depends on the heights.
            </>
          )}
        </Note>
      }
      table={
        <>
          <table className="viz-table">
            <thead>
              <tr>
                <th>Level</th>
                <th>Nodes present</th>
                <th>Expected n·p^level</th>
                <th>Share of nodes</th>
              </tr>
            </thead>
            <tbody>
              {m.perLevel.map((c, l) => (
                <tr key={l}>
                  <td>L{l}</td>
                  <td>{c}</td>
                  <td>{(n * p ** l).toFixed(2)}</td>
                  <td>{((c / n) * 100).toFixed(1)}%</td>
                </tr>
              ))}
            </tbody>
          </table>
          <table className="viz-table">
            <thead>
              <tr>
                <th>n</th>
                <th>Levels</th>
                <th>Pointers/node</th>
                <th>Pugh bound</th>
                <th>Mean</th>
                <th>p99</th>
                <th>Worst sampled</th>
              </tr>
            </thead>
            <tbody>
              {curve.map((c) => (
                <tr key={c.n}>
                  <td>{fmtNum(c.n)}</td>
                  <td>{c.top}</td>
                  <td>{c.ptr.toFixed(2)}</td>
                  <td>{c.model.toFixed(1)}</td>
                  <td>{c.mean.toFixed(1)}</td>
                  <td>{c.p99}</td>
                  <td>{c.max}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </>
      }
    >
      <div ref={ref}>
        <TooltipHost>
          <svg width={listW} height={listH} role="img" aria-label="Skip list towers and the search path through them">
            {/* level lanes */}
            {Array.from({ length: m.top }, (_, l) => (
              <g key={`lane${l}`}>
                <line
                  x1={padL}
                  x2={listW - 8}
                  y1={laneY(l) + cellH / 2}
                  y2={laneY(l) + cellH / 2}
                  stroke="var(--viz-grid)"
                />
                <text x={listW - 6} y={laneY(l) + cellH / 2 + 4} textAnchor="start" fill="var(--viz-ink-muted)">
                  L{l}
                </text>
              </g>
            ))}

            {/* forward pointers */}
            {Array.from({ length: m.top }, (_, l) =>
              Array.from({ length: n + 1 }, (_, i) => {
                if (i > 0 && m.h[i - 1] <= l) return null;
                const j = m.next[l][i];
                const y = laneY(l) + cellH / 2;
                const x1 = i === 0 ? padL + headW - 10 : colX(i) + cellW;
                const x2 = j === -1 ? listW - 22 : colX(j);
                const hot = walked.has(`${l}:${i}`);
                return (
                  <g key={`e${l}-${i}`}>
                    <line
                      x1={x1}
                      x2={x2 - 3}
                      y1={y}
                      y2={y}
                      stroke={hot ? 'var(--viz-2)' : 'var(--viz-axis)'}
                      strokeWidth={hot ? 2.5 : 1}
                      strokeDasharray={j === -1 ? '2 3' : undefined}
                    />
                    {j !== -1 ? (
                      <path
                        d={`M${x2 - 5},${y - 3} L${x2},${y} L${x2 - 5},${y + 3} Z`}
                        fill={hot ? 'var(--viz-2)' : 'var(--viz-axis)'}
                      />
                    ) : null}
                  </g>
                );
              }),
            )}

            {/* the head sentinel */}
            <g {...tip(<>The head sentinel has a forward pointer at every level in use.</>)}>
              <rect
                x={padL}
                y={laneY(m.top - 1)}
                width={headW - 10}
                height={(m.top - 1) * laneH + cellH}
                rx={4}
                fill="var(--viz-plane)"
                stroke="var(--viz-ink-2)"
              />
              <text x={padL + (headW - 10) / 2} y={laneY(0) + cellH - 4} textAnchor="middle" fill="var(--viz-ink)">
                head
              </text>
            </g>

            {/* towers */}
            {m.h.map((h, k) => {
              const i = k + 1;
              const isHere = here === i;
              return (
                <g
                  key={`t${i}`}
                  {...tip(
                    <>
                      <strong>key {KEY(i)}</strong>
                      <br />
                      height {h} — promoted {h - 1}× ({((p ** (h - 1)) * 100).toFixed(1)}% of nodes reach this
                      height or more)
                    </>,
                  )}
                  style={{ cursor: 'help' }}
                >
                  {Array.from({ length: h }, (_, l) => (
                    <rect
                      key={l}
                      x={colX(i)}
                      y={laneY(l)}
                      width={cellW}
                      height={cellH}
                      rx={3}
                      fill={isHere ? 'var(--viz-2)' : 'var(--viz-1)'}
                      opacity={l === 0 ? 1 : 0.55 + 0.45 * (1 - l / Math.max(1, m.top))}
                      stroke="var(--viz-surface)"
                    />
                  ))}
                  <text
                    x={colX(i) + cellW / 2}
                    y={laneY(0) + cellH + 13}
                    textAnchor="middle"
                    fill={KEY(i) === target ? 'var(--viz-8)' : 'var(--viz-ink-2)'}
                    fontWeight={KEY(i) === target ? 700 : 400}
                  >
                    {KEY(i)}
                  </text>
                </g>
              );
            })}

            {/* level drops: the vertical part of the staircase */}
            {Array.from(dropped.entries()).map(([l, i]) => (
              <line
                key={`d${l}`}
                x1={colMid(i)}
                x2={colMid(i)}
                y1={laneY(l) + cellH / 2}
                y2={laneY(Math.max(0, l - 1)) + cellH / 2}
                stroke="var(--viz-8)"
                strokeWidth={2}
                strokeDasharray="3 2"
              />
            ))}

            <text x={padL} y={listH - 4} fill="var(--viz-ink-muted)">
              searching for {target} · currently at {here === 0 ? 'head' : `key ${KEY(here)}`}, level L{level}
            </text>
          </svg>

          <svg width={chW} height={chH} role="img" aria-label="Measured search cost against the analytic bound, up to 65536 keys">
            <line x1={mL} x2={chW - mR} y1={chH - mB} y2={chH - mB} stroke="var(--viz-axis)" />
            <line x1={mL} x2={mL} y1={mT} y2={chH - mB} stroke="var(--viz-axis)" />
            {[0, 0.25, 0.5, 0.75, 1].map((f) => (
              <g key={f}>
                <line x1={mL} x2={chW - mR} y1={cy(yMax * f)} y2={cy(yMax * f)} stroke="var(--viz-grid)" />
                <text x={mL - 6} y={cy(yMax * f) + 4} textAnchor="end" fill="var(--viz-ink-muted)">
                  {Math.round(yMax * f)}
                </text>
              </g>
            ))}
            {CURVE_E.filter((e) => e % 3 === 1).map((e) => (
              <text key={e} x={cx(e)} y={chH - mB + 14} textAnchor="middle" fill="var(--viz-ink-muted)">
                {fmtNum(2 ** e)}
              </text>
            ))}
            <text x={mL} y={chH - 4} fill="var(--viz-ink-muted)">
              keys in the list (log scale) · comparisons per point lookup
            </text>

            <path d={line((c) => c.model)} fill="none" stroke="var(--viz-1)" strokeWidth={2} />
            <path d={line((c) => c.mean)} fill="none" stroke="var(--viz-2)" strokeWidth={2} strokeDasharray="5 3" />
            <path d={line((c) => c.p99)} fill="none" stroke="var(--viz-3)" strokeWidth={2} strokeDasharray="2 3" />
            {curve.map((c, i) => (
              <g key={c.n} {...tip(
                <>
                  <strong>n = {fmtNum(c.n)}</strong>
                  <br />
                  bound {c.model.toFixed(1)} · mean {c.mean.toFixed(1)} · p99 {c.p99} · worst {c.max}
                  <br />
                  {c.top} levels, {c.ptr.toFixed(2)} pointers/node
                </>,
              )}>
                <circle cx={cx(CURVE_E[i])} cy={cy(c.mean)} r={3} fill="var(--viz-2)" />
                <circle cx={cx(CURVE_E[i])} cy={cy(c.p99)} r={3} fill="var(--viz-3)" />
                <rect x={cx(CURVE_E[i]) - 8} y={mT} width={16} height={chH - mT - mB} fill="transparent" />
              </g>
            ))}
            <text x={chW - mR + 6} y={cy(curve[curve.length - 1].model) + 4} fill="var(--viz-1)">
              bound
            </text>
            <text x={chW - mR + 6} y={cy(curve[curve.length - 1].mean) + 4} fill="var(--viz-2)">
              mean
            </text>
            <text x={chW - mR + 6} y={cy(curve[curve.length - 1].p99) + 4} fill="var(--viz-3)">
              p99
            </text>
            <line
              x1={cx(Math.log2(n))}
              x2={cx(Math.log2(n))}
              y1={mT}
              y2={chH - mB}
              stroke="var(--viz-ink-muted)"
              strokeDasharray="4 3"
            />
            <text x={cx(Math.log2(n)) + 4} y={mT + 10} fill="var(--viz-ink-muted)">
              this list
            </text>
          </svg>
        </TooltipHost>
      </div>
    </VizPanel>
  );
}
