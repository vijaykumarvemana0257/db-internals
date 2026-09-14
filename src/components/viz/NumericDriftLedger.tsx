import { useMemo, useState } from 'react';
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
  makeRng,
  fmtNum,
  useSize,
} from './Viz';

/**
 * One column of money, three accumulators.
 *
 * The amounts are generated as exact integer minor units (cents), so there is never any
 * doubt about the true answer: `exactCents` is it. The other two columns are what a
 * database would actually compute:
 *
 *   - numeric / DECIMAL      exact base-10 arithmetic — always equals exactCents/100
 *   - float8 / float4        IEEE-754 binary, accumulated in the same width the engine
 *                            uses for its aggregate state (float8pl over float8,
 *                            float4pl over float4)
 *
 * JS numbers *are* IEEE-754 binary64, so the float8 column is not a simulation — the
 * drift you see is the drift Postgres gets. Math.fround() gives real binary32 rounding
 * for the float4 column.
 *
 * The two things that make the float column non-deterministic are modelled directly:
 * reordering the rows (a different scan order, a different plan) and splitting the scan
 * across N parallel workers that each keep their own partial sum and hand it to the
 * leader's combine function.
 */

/* ------------------------------------------------------------------- inputs */

const ROW_STEPS = [100, 300, 1_000, 3_000, 10_000, 30_000, 100_000, 300_000] as const;

type Shape = 'retail' | 'tenths' | 'ledger' | 'tail';

const SHAPES: { value: Shape; label: string }[] = [
  { value: 'retail', label: 'Retail prices, $0.01–$999.99' },
  { value: 'tenths', label: 'Every row exactly $0.10' },
  { value: 'ledger', label: 'Ledger: debits + credits, nets to $0.00' },
  { value: 'tail', label: 'Long tail: micro-charges + $1M–$10M rows' },
];

/** Amounts in exact integer cents. Deterministic for a given (shape, n). */
function generate(shape: Shape, n: number): number[] {
  const rng = makeRng(shape === 'retail' ? 8_675_309 : shape === 'ledger' ? 4_011 : 220_461);
  const out = new Array<number>(n);
  if (shape === 'tenths') {
    // 0.1 has no finite binary expansion, so every single addend is already wrong.
    out.fill(10);
  } else if (shape === 'retail') {
    for (let i = 0; i < n; i++) out[i] = 1 + Math.floor(rng() * 99_999);
  } else if (shape === 'ledger') {
    // Every debit has a matching credit, so the exact total is exactly zero in any order.
    for (let i = 0; i < n; i += 2) {
      const c = 1 + Math.floor(rng() * 500_000);
      out[i] = c;
      if (i + 1 < n) out[i + 1] = -c;
    }
    if (n % 2 === 1) out[n - 1] = 0;
    // ...but they are interleaved, not adjacent. A real ledger never stores +c in the row
    // immediately before -c, and an adjacent pair would cancel *exactly* even in binary
    // float ((0 + v) - v is 0 with no rounding), which would hide the whole effect.
    const perm = shuffled(n, 0);
    const mixed = new Array<number>(n);
    for (let i = 0; i < n; i++) mixed[i] = out[perm[i]!]!;
    return mixed;
  } else {
    for (let i = 0; i < n; i++) {
      out[i] = rng() < 0.004 ? 100_000_000 + Math.floor(rng() * 900_000_000) : 1 + Math.floor(rng() * 999);
    }
  }
  return out;
}

function shuffled(n: number, seed: number): number[] {
  const rng = makeRng(1_000_003 + seed * 7919);
  const o = Array.from({ length: n }, (_, i) => i);
  for (let i = n - 1; i > 0; i--) {
    const j = Math.floor(rng() * (i + 1));
    const t = o[i]!;
    o[i] = o[j]!;
    o[j] = t;
  }
  return o;
}

/* -------------------------------------------------------------- the machine */

type Run = {
  total: number; // the float accumulator's final value, in dollars
  drift: number; // total - exact, in dollars
  curve: { x: number; e: number }[]; // sampled error, in dollars
  partials: { start: number; end: number; f: number; exact: number }[];
  cuts: number[]; // x positions where a new worker's accumulator starts
};

const SAMPLES = 260;

function accumulate(cents: number[], order: number[], f4: boolean, workers: number, exactCents: number): Run {
  const n = order.length;
  const per = Math.ceil(n / workers);
  const curve: { x: number; e: number }[] = [];
  const partials: Run['partials'] = [];
  const cuts: number[] = [];
  const every = Math.max(1, Math.floor(n / SAMPLES));

  for (let w = 0; w < workers; w++) {
    const start = w * per;
    const end = Math.min(n, start + per);
    if (start >= end) break;
    if (w > 0) cuts.push(start);
    let f = 0;
    let ex = 0;
    curve.push({ x: start, e: 0 });
    for (let i = start; i < end; i++) {
      const c = cents[order[i]!]!;
      const v = f4 ? Math.fround(c / 100) : c / 100;
      f = f4 ? Math.fround(f + v) : f + v;
      ex += c;
      if ((i - start) % every === 0 || i === end - 1) curve.push({ x: i + 1, e: f - ex / 100 });
    }
    partials.push({ start, end, f, exact: ex });
  }

  // The leader combines the partial sums with the same float addition, in worker order.
  let total = 0;
  for (const p of partials) total = f4 ? Math.fround(total + p.f) : total + p.f;

  return { total, drift: total - exactCents / 100, curve, partials, cuts };
}

/* ------------------------------------------------------------- formatting */

function money(v: number): string {
  if (v !== 0 && Math.abs(v) < 1e-6) return v.toExponential(3);
  const s = v.toFixed(12);
  if (!s.includes('.')) return s;
  const trimmed = s.replace(/0+$/, '');
  const [int, frac = ''] = trimmed.split('.');
  return `${int}.${frac.padEnd(2, '0')}`;
}

/** Drift is interesting across ~15 orders of magnitude; label it in whatever unit reads. */
function drift(v: number): string {
  const a = Math.abs(v);
  if (a === 0) return 'exactly 0';
  if (a >= 0.01) return `${v < 0 ? '−' : '+'}$${a.toFixed(2)}`;
  if (a >= 1e-4) return `${v < 0 ? '−' : '+'}${(a * 100).toFixed(4)} ¢`;
  return `${v < 0 ? '−' : '+'}${(a * 100).toExponential(2)} ¢`;
}

export default function NumericDriftLedger() {
  const [rowIdx, setRowIdx] = useState(5); // 30 000 rows
  const [shape, setShape] = useState<Shape>('retail');
  const [f4, setF4] = useState<'f8' | 'f4'>('f8');
  const [workers, setWorkers] = useState(1);
  const [seed, setSeed] = useState(1);
  const [ref, width] = useSize(720);
  const tip = useTip();

  const n = ROW_STEPS[rowIdx]!;
  const isF4 = f4 === 'f4';

  const { cents, exactCents, natural, reordered, parallel } = useMemo(() => {
    const cents = generate(shape, n);
    let exactCents = 0;
    for (const c of cents) exactCents += c;
    const idNat = Array.from({ length: n }, (_, i) => i);
    const idShuf = shuffled(n, seed);
    return {
      cents,
      exactCents,
      natural: accumulate(cents, idNat, isF4, 1, exactCents),
      reordered: accumulate(cents, idShuf, isF4, 1, exactCents),
      parallel: accumulate(cents, idNat, isF4, workers, exactCents),
    };
  }, [shape, n, isF4, workers, seed]);

  const exact = exactCents / 100;
  const shown = workers > 1 ? parallel : natural;

  /* ------------------------------------------------------------------ chart */
  const W = Math.max(360, Math.min(width, 860));
  const H = 216;
  const M = { l: 76, r: 74, t: 14, b: 30 };
  const iw = W - M.l - M.r;
  const ih = H - M.t - M.b;

  const curves = [
    { key: 'nat', c: natural, color: 'var(--viz-2)', dash: undefined, label: 'in table order' },
    { key: 'shuf', c: reordered, color: 'var(--viz-3)', dash: '5 3', label: 'shuffled' },
    ...(workers > 1
      ? [{ key: 'par', c: parallel, color: 'var(--viz-4)', dash: '2 3', label: `${workers} workers` }]
      : []),
  ];

  let span = 0;
  for (const { c } of curves) for (const p of c.curve) span = Math.max(span, Math.abs(p.e));
  span = span === 0 ? 1e-15 : span * 1.15;

  const x = (v: number) => M.l + (v / n) * iw;
  const y = (v: number) => M.t + ih / 2 - (v / span) * (ih / 2);
  const path = (pts: { x: number; e: number }[]) =>
    pts.map((p, i) => `${i === 0 ? 'M' : 'L'}${x(p.x).toFixed(2)} ${y(p.e).toFixed(2)}`).join(' ');

  const yTicks = [span, span / 2, 0, -span / 2, -span];
  const tickLabel = (v: number) => {
    if (v === 0) return '0';
    const cts = v * 100;
    return Math.abs(cts) >= 0.01 ? `${cts.toFixed(2)}¢` : `${cts.toExponential(0)}¢`;
  };

  const orderChanges = natural.total !== reordered.total;
  const parallelChanges = parallel.total !== natural.total;
  const floatExact = shown.total === exact;

  return (
    <VizPanel
      title="One column of money, three accumulators"
      subtitle="The amounts are exact integer cents, so the right answer is never in doubt. Watch what the binary-float accumulator does to it — and what reordering the rows or splitting them across parallel workers does to that."
      controls={
        <>
          <Choice
            label="Amounts"
            value={shape}
            onChange={(v) => setShape(v as Shape)}
            options={SHAPES}
          />
          <Slider
            label="Rows"
            min={0}
            max={ROW_STEPS.length - 1}
            value={rowIdx}
            onChange={setRowIdx}
            format={() => fmtNum(n)}
          />
          <Segmented
            label="Float column"
            value={f4}
            onChange={setF4}
            options={[
              { value: 'f8', label: 'float8', title: 'double precision — binary64, 53-bit significand' },
              { value: 'f4', label: 'float4', title: 'real — binary32, 24-bit significand' },
            ]}
          />
          <Slider
            label="Parallel workers"
            min={1}
            max={8}
            value={workers}
            onChange={setWorkers}
            format={(w) => (w === 1 ? 'serial' : `${w} partial sums`)}
          />
          <Button onClick={() => setSeed((s) => s + 1)} title="Same rows, different scan order">
            Shuffle rows
          </Button>
        </>
      }
      legend={
        <Legend
          items={[
            { label: 'numeric / bigint cents — exact, flat on zero', color: 'var(--viz-1)', shape: 'line' },
            { label: `${f4} sum, table order`, color: 'var(--viz-2)', shape: 'line' },
            { label: `${f4} sum, shuffled order (dashed)`, color: 'var(--viz-3)', shape: 'line' },
            ...(workers > 1
              ? [{ label: `${f4} partial sums, ${workers} workers (dotted)`, color: 'var(--viz-4)', shape: 'line' as const }]
              : []),
          ]}
        />
      }
      stats={
        <Stats
          items={[
            { label: 'numeric / cents', value: `$${exact.toFixed(2)}`, hint: 'The exact answer: sum of the integer minor units, divided by 100' },
            { label: `${f4} total`, value: `$${money(shown.total)}`, hint: 'What the binary-float accumulator ended up holding' },
            { label: 'drift', value: drift(shown.drift), hint: 'float total − exact total' },
            { label: 'float = exact?', value: floatExact ? 'yes' : 'no' },
            { label: 'order-stable?', value: orderChanges || parallelChanges ? 'no' : 'yes', hint: 'Does the float total survive a reorder and a parallel split?' },
          ]}
        />
      }
      note={
        <Note>
          <strong>
            {floatExact
              ? 'At this size the float total happens to land on the exact value.'
              : `The ${f4} total is off by ${drift(shown.drift)}.`}
          </strong>{' '}
          {orderChanges
            ? `Shuffling the rows changes it again — table order gives $${money(natural.total)}, the shuffled order gives $${money(reordered.total)}. `
            : 'Shuffling the rows happens to give the same total at this size. '}
          {workers > 1
            ? `Splitting the scan across ${workers} workers gives a third answer, $${money(parallel.total)}: each worker rounds inside its own accumulator and the leader rounds again when it combines them. `
            : 'Push the worker slider up and the leader combines separately-rounded partial sums into a fourth answer. '}
          The numeric and integer-cents columns are on zero at every row and stay there under every reordering, because
          exact decimal addition is associative and binary-float addition is not.
        </Note>
      }
      table={
        <>
          <table className="viz-table">
            <thead>
              <tr>
                <th>Accumulator</th>
                <th>Total</th>
                <th>Difference from exact</th>
                <th>= exact?</th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <td>numeric(14,2)</td>
                <td>{exact.toFixed(2)}</td>
                <td>exactly 0</td>
                <td>yes</td>
              </tr>
              <tr>
                <td>bigint minor units</td>
                <td>{fmtNum(exactCents)} ¢</td>
                <td>exactly 0</td>
                <td>yes</td>
              </tr>
              <tr>
                <td>{f4}, table order</td>
                <td>{money(natural.total)}</td>
                <td>{drift(natural.drift)}</td>
                <td>{natural.total === exact ? 'yes' : 'no'}</td>
              </tr>
              <tr>
                <td>{f4}, shuffled order</td>
                <td>{money(reordered.total)}</td>
                <td>{drift(reordered.drift)}</td>
                <td>{reordered.total === exact ? 'yes' : 'no'}</td>
              </tr>
              <tr>
                <td>
                  {f4}, {workers} worker{workers === 1 ? '' : 's'}
                </td>
                <td>{money(parallel.total)}</td>
                <td>{drift(parallel.drift)}</td>
                <td>{parallel.total === exact ? 'yes' : 'no'}</td>
              </tr>
            </tbody>
          </table>
          <table className="viz-table">
            <thead>
              <tr>
                <th>Row</th>
                <th>Amount</th>
                <th>Running cents (exact)</th>
                <th>Running {f4}</th>
                <th>Drift</th>
              </tr>
            </thead>
            <tbody>
              {(() => {
                const rows = [];
                let f = 0;
                let ex = 0;
                for (let i = 0; i < Math.min(10, n); i++) {
                  const c = cents[i]!;
                  const v = isF4 ? Math.fround(c / 100) : c / 100;
                  f = isF4 ? Math.fround(f + v) : f + v;
                  ex += c;
                  rows.push(
                    <tr key={i}>
                      <td>{i + 1}</td>
                      <td>{(c / 100).toFixed(2)}</td>
                      <td>{fmtNum(ex)}</td>
                      <td>{money(f)}</td>
                      <td>{drift(f - ex / 100)}</td>
                    </tr>,
                  );
                }
                return rows;
              })()}
            </tbody>
          </table>
        </>
      }
    >
      <div ref={ref}>
        <TooltipHost>
          <svg width={W} height={H} role="img" aria-label="Accumulated error of the float sum against row number">
            {/* zero line = numeric and integer cents, at every row */}
            <line className="viz-axis-line" x1={M.l} x2={M.l + iw} y1={y(0)} y2={y(0)} stroke="var(--viz-1)" strokeWidth={2} />
            <text x={M.l + iw + 6} y={y(0) + 4} fill="var(--viz-1)">
              exact
            </text>

            {yTicks.map((t) => (
              <g key={t}>
                {t !== 0 ? <line className="viz-grid-line" x1={M.l} x2={M.l + iw} y1={y(t)} y2={y(t)} /> : null}
                <text x={M.l - 8} y={y(t) + 4} textAnchor="end">
                  {tickLabel(t)}
                </text>
              </g>
            ))}
            <text x={6} y={M.t + 10} fill="var(--viz-ink-2)">
              error
            </text>

            {/* x axis */}
            <line className="viz-axis-line" x1={M.l} x2={M.l + iw} y1={M.t + ih} y2={M.t + ih} />
            {[0, 0.25, 0.5, 0.75, 1].map((f) => (
              <text key={f} x={x(n * f)} y={M.t + ih + 16} textAnchor="middle">
                {fmtNum(Math.round(n * f))}
              </text>
            ))}
            <text x={M.l + iw} y={M.t + ih + 28} textAnchor="end" fill="var(--viz-ink-muted)">
              rows accumulated
            </text>

            {/* worker boundaries: each partial sum restarts its own accumulator at zero */}
            {workers > 1
              ? parallel.cuts.map((c) => (
                  <line
                    key={c}
                    className="viz-grid-line"
                    x1={x(c)}
                    x2={x(c)}
                    y1={M.t}
                    y2={M.t + ih}
                    strokeDasharray="2 4"
                    stroke="var(--viz-4)"
                  />
                ))
              : null}

            {curves.map((cv) => (
              <path
                key={cv.key}
                d={path(cv.c.curve)}
                fill="none"
                stroke={cv.color}
                strokeWidth={cv.key === 'nat' ? 2 : 1.5}
                strokeDasharray={cv.dash}
              />
            ))}

            {/* direct end labels, so identity is never color-alone */}
            {curves.map((cv, i) => {
              const last = cv.c.curve[cv.c.curve.length - 1]!;
              return (
                <text key={cv.key} x={M.l + iw + 6} y={y(last.e) + 4 + (i === 0 ? -10 : i === 1 ? 10 : 22)} fill={cv.color}>
                  {cv.label}
                </text>
              );
            })}

            {/* per-worker partial-sum markers, hoverable */}
            {workers > 1
              ? parallel.partials.map((p, i) => (
                  <circle
                    key={i}
                    cx={x(p.end)}
                    cy={y(p.f - p.exact / 100)}
                    r={4}
                    fill="var(--viz-4)"
                    {...tip(
                      <>
                        <strong>worker {i + 1}</strong>
                        <br />
                        rows {fmtNum(p.start + 1)}–{fmtNum(p.end)}
                        <br />
                        partial {f4}: {money(p.f)}
                        <br />
                        exact: {(p.exact / 100).toFixed(2)}
                        <br />
                        drift: {drift(p.f - p.exact / 100)}
                      </>,
                    )}
                  />
                ))
              : null}

            {/* final markers */}
            {[
              { c: natural, color: 'var(--viz-2)', name: 'table order' },
              { c: reordered, color: 'var(--viz-3)', name: 'shuffled order' },
            ].map((s) => (
              <circle
                key={s.name}
                cx={x(n)}
                cy={y(s.c.curve[s.c.curve.length - 1]!.e)}
                r={4}
                fill={s.color}
                {...tip(
                  <>
                    <strong>{s.name}</strong>
                    <br />
                    {f4} total: {money(s.c.total)}
                    <br />
                    exact: {exact.toFixed(2)}
                    <br />
                    drift: {drift(s.c.drift)}
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
