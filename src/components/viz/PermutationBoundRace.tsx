import { useMemo, useState } from 'react';
import {
  VizPanel,
  Slider,
  Button,
  Legend,
  Stats,
  Note,
  TooltipHost,
  useTip,
  useTicker,
  fmtBytes,
  fmtNum,
  useSize,
} from './Viz';

/**
 * The permutation lower bound, as a race.
 *
 * Building an index is a permutation: N records arrive in one order and must end
 * up in another. Aggarwal and Vitter proved that costs Ω(min(N, (N/B)·log_{M/B}(N/B)))
 * block transfers. The two branches of that min are two real strategies — move one
 * record at a time (N), or sort (the second term) — and the gap between them is a
 * factor of roughly B/(2 log_{M/B}(N/B)). This runs both at the same record rate and
 * accumulates the I/O each actually spends.
 */

const ENTRY = 24; // bytes of one index tuple: bigint key + heap TID + header
const FILL = 0.9; // leaf fill factor for a bulk load
const LIVE_FILL = 0.7; // steady-state fill under random inserts
const STEPS = 240;

/** Compact counts — axis labels have no room for 10,000,000,000. */
const short = (n: number) =>
  n >= 1e9
    ? `${(n / 1e9).toFixed(n < 1e10 ? 1 : 0)}B`
    : n >= 1e6
      ? `${(n / 1e6).toFixed(n < 1e7 ? 1 : 0)}M`
      : n >= 1e3
        ? `${Math.round(n / 1e3)}k`
        : `${Math.round(n)}`;

function passesFor(N: number, M: number, B: number) {
  const dataBytes = N * ENTRY;
  const runs = Math.max(1, Math.ceil(dataBytes / M));
  const fanIn = Math.max(2, Math.floor(M / B) - 1);
  if (runs <= 1) return { runs, fanIn, passes: 0 };
  return { runs, fanIn, passes: Math.ceil(Math.log(runs) / Math.log(fanIn)) };
}

type Model = {
  sortCurve: number[]; // cumulative I/Os after each step
  insertCurve: number[];
  sortTotal: number;
  insertTotal: number;
  passes: number;
  runs: number;
  fanIn: number;
  blocks: number;
  crossover: number; // fraction of N at which random inserts have spent the whole sort budget
};

function buildModel(N: number, M: number, B: number): Model {
  const { runs, fanIn, passes } = passesFor(N, M, B);
  const blocks = (N * ENTRY) / B;
  // read the heap, write initial runs, read+write per extra merge pass, read the
  // final merge into the leaf builder, write the packed leaves.
  const sortTotal = blocks * (2 + 2 * Math.max(0, passes - 1) + (passes > 0 ? 1 : 0)) + blocks / FILL;

  const sortCurve: number[] = [];
  const insertCurve: number[] = [];
  let acc = 0;
  let crossover = 1;
  for (let i = 1; i <= STEPS; i++) {
    const done = (i / STEPS) * N;
    sortCurve.push((i / STEPS) * sortTotal);
    // index bytes so far, and the share of them the buffer pool still holds
    const indexBytes = (done * ENTRY) / LIVE_FILL;
    const resident = Math.min(1, M / Math.max(1, indexBytes));
    acc += (N / STEPS) * 2 * (1 - resident); // one read + one eventual write per miss
    insertCurve.push(acc);
    if (crossover === 1 && acc >= sortTotal) crossover = i / STEPS;
  }
  return {
    sortCurve,
    insertCurve,
    sortTotal,
    insertTotal: acc,
    passes,
    runs,
    fanIn,
    blocks,
    crossover,
  };
}

function Chart({ model, t, avail, N }: { model: Model; t: number; avail: number; N: number }) {
  const tip = useTip();
  const W = Math.max(360, Math.min(avail, 780));
  const H = 270;
  const pad = { l: 48, r: 96, t: 14, b: 36 };
  const iw = W - pad.l - pad.r;
  const ih = H - pad.t - pad.b;
  const hi = Math.max(model.insertTotal, model.sortTotal, 10) * 1.6;
  const lo = Math.max(1, model.sortTotal / 5000);
  const x = (f: number) => pad.l + f * iw;
  const y = (v: number) =>
    pad.t + ih - ((Math.log(Math.max(lo, v)) - Math.log(lo)) / (Math.log(hi) - Math.log(lo))) * ih;

  const ticks: number[] = [];
  for (let e = 0; e <= 14; e++) {
    const v = 10 ** e;
    if (v >= lo && v <= hi) ticks.push(v);
  }

  const path = (curve: number[], upto: number) =>
    curve
      .slice(0, Math.max(1, Math.round(upto * STEPS)))
      .map((v, i) => `${x((i + 1) / STEPS)},${y(v)}`)
      .join(' ');

  const bound = Math.min(N, model.sortTotal);

  return (
    <svg width={W} height={H} role="img" aria-label="Cumulative block transfers for sort-then-build versus random inserts">
      {ticks.map((v) => (
        <g key={v}>
          <line className="viz-grid-line" x1={pad.l} x2={pad.l + iw} y1={y(v)} y2={y(v)} />
          <text x={pad.l - 8} y={y(v) + 4} textAnchor="end" fill="var(--viz-ink-muted)">
            {short(v)}
          </text>
        </g>
      ))}
      <line className="viz-axis-line" x1={pad.l} x2={pad.l + iw} y1={pad.t + ih} y2={pad.t + ih} />
      {[0, 0.25, 0.5, 0.75, 1].map((f) => (
        <text key={f} x={x(f)} y={pad.t + ih + 16} textAnchor="middle" fill="var(--viz-ink-muted)">
          {short(f * N)}
        </text>
      ))}
      <text x={pad.l} y={H - 4} fill="var(--viz-ink-2)">
        records inserted
      </text>
      <text x={pad.l - 8} y={pad.t - 2} textAnchor="end" fill="var(--viz-ink-2)">
        I/Os
      </text>

      {/* the two branches of the permutation lower bound */}
      <line
        x1={pad.l}
        x2={pad.l + iw}
        y1={y(N)}
        y2={y(N)}
        stroke="var(--viz-ink-muted)"
        strokeDasharray="2 4"
      />
      <text x={pad.l + 6} y={y(N) - 5} fill="var(--viz-ink-muted)">
        N — one record at a time
      </text>
      <line
        x1={pad.l}
        x2={pad.l + iw}
        y1={y(bound)}
        y2={y(bound)}
        stroke="var(--viz-ink-muted)"
        strokeDasharray="2 4"
      />
      <text x={pad.l + 6} y={y(bound) - 5} fill="var(--viz-ink-muted)">
        (N/B)·log&#8202;<tspan dy={3} fontSize="0.8em">M/B</tspan>
        <tspan dy={-3}>(N/B) — sort</tspan>
      </text>

      {[
        { c: 'var(--viz-2)', curve: model.insertCurve, label: 'random inserts', total: model.insertTotal },
        { c: 'var(--viz-1)', curve: model.sortCurve, label: 'sort + bulk load', total: model.sortTotal },
      ].map((s) => (
        <g key={s.label}>
          <polyline fill="none" stroke={s.c} strokeWidth={1} opacity={0.28} points={path(s.curve, 1)} />
          <polyline fill="none" stroke={s.c} strokeWidth={2.5} points={path(s.curve, t)} />
          <circle
            cx={x(Math.max(1 / STEPS, t))}
            cy={y(s.curve[Math.max(0, Math.round(t * STEPS) - 1)])}
            r={4}
            fill={s.c}
            {...tip(
              <>
                <strong>{s.label}</strong>
                <br />
                {fmtNum(s.curve[Math.max(0, Math.round(t * STEPS) - 1)])} I/Os after{' '}
                {short(t * N)} records
                <br />
                finishes at {fmtNum(s.total)}
              </>,
            )}
          />
          <text x={pad.l + iw + 6} y={y(s.total) + 4} fill={s.c}>
            {s.label === 'random inserts' ? 'inserts' : 'sort'}
          </text>
        </g>
      ))}
    </svg>
  );
}

export default function PermutationBoundRace() {
  const [nExp, setNExp] = useState(8);
  const [blockExp, setBlockExp] = useState(13); // 8 KB
  const [memExp, setMemExp] = useState(26); // 64 MB — Postgres maintenance_work_mem default
  const [t, setT] = useState(1);
  const [running, setRunning] = useState(false);
  const [ref, width] = useSize(760);

  const N = Math.round(10 ** nExp);
  const B = 2 ** blockExp;
  const M = 2 ** memExp;
  const model = useMemo(() => buildModel(N, M, B), [N, M, B]);

  useTicker(() => {
    setT((v) => {
      const next = v + 0.008;
      if (next >= 1) {
        setRunning(false);
        return 1;
      }
      return next;
    });
  }, running);

  const idx = Math.max(0, Math.round(t * STEPS) - 1);
  const ratio = model.insertTotal / Math.max(1, model.sortTotal);

  return (
    <VizPanel
      title="Sort-then-build versus N random inserts"
      subtitle="Both lanes process the same records in the same order at the same rate; only the I/O accounting differs. The dashed lines are the two branches of the permutation lower bound."
      controls={
        <>
          <Slider
            label="Records N"
            min={6}
            max={10}
            step={0.5}
            value={nExp}
            onChange={setNExp}
            format={() => short(N)}
          />
          <Slider
            label="Page size B"
            min={9}
            max={16}
            value={blockExp}
            onChange={setBlockExp}
            format={() => `${fmtBytes(B)} (${Math.floor(B / ENTRY)} tuples)`}
          />
          <Slider
            label="Sort memory M"
            min={22}
            max={33}
            value={memExp}
            onChange={setMemExp}
            format={() => fmtBytes(M)}
          />
          <Button
            primary
            onClick={() => {
              setT(0);
              setRunning(true);
            }}
          >
            Run the build
          </Button>
          <Button onClick={() => setRunning((r) => !r)} disabled={t >= 1}>
            {running ? 'Pause' : 'Resume'}
          </Button>
        </>
      }
      legend={
        <Legend
          items={[
            { label: 'Sort + bulk load', color: 'var(--viz-1)', shape: 'line' },
            { label: 'N random inserts', color: 'var(--viz-2)', shape: 'line' },
            { label: 'Lower-bound branches', color: 'var(--viz-ink-muted)', shape: 'line' },
          ]}
        />
      }
      stats={
        <Stats
          items={[
            { label: 'Sort + build I/Os', value: fmtNum(model.sortCurve[idx]), hint: `finishes at ${fmtNum(model.sortTotal)}` },
            { label: 'Random-insert I/Os', value: fmtNum(model.insertCurve[idx]), hint: `finishes at ${fmtNum(model.insertTotal)}` },
            { label: 'Final ratio', value: `${ratio < 10 ? ratio.toFixed(1) : fmtNum(ratio)}×` },
            { label: 'Merge passes', value: model.passes, hint: `${fmtNum(model.runs)} runs, ${fmtNum(model.fanIn)}-way merge` },
            {
              label: 'Inserts blow the sort budget at',
              value: model.crossover >= 1 ? 'never' : `${(model.crossover * 100).toFixed(1)}% of N`,
            },
          ]}
        />
      }
      note={
        <Note>
          <strong>
            {model.crossover >= 1
              ? 'At this size the whole index still fits in memory — inserts are free until it does not.'
              : `Random inserts spend the entire sort-and-build budget after ${(model.crossover * 100).toFixed(1)}% of the rows, then keep paying.`}
          </strong>{' '}
          The insert curve is flat while the growing index fits in M and bends the moment it does not:
          from there every record costs a leaf read plus an eventual leaf write, so the cost approaches
          2N. The sort lane pays {model.passes === 0 ? 'no merge pass at all' : `${model.passes} merge pass(es)`}{' '}
          over {fmtNum(model.blocks, 0)} blocks, which is N/B · log<sub>M/B</sub>(N/B) with the constants
          filled in. Shrinking M raises the merge passes by one at a time — a step function, not a slope.
        </Note>
      }
      table={
        <table className="viz-table">
          <thead>
            <tr>
              <th>N</th>
              <th>Sort + build I/Os</th>
              <th>Random-insert I/Os</th>
              <th>Ratio</th>
              <th>Merge passes</th>
            </tr>
          </thead>
          <tbody>
            {[6, 6.5, 7, 7.5, 8, 8.5, 9, 9.5, 10].map((e) => {
              const n = Math.round(10 ** e);
              const m = buildModel(n, M, B);
              return (
                <tr key={e}>
                  <td>{short(n)}</td>
                  <td>{fmtNum(m.sortTotal)}</td>
                  <td>{fmtNum(m.insertTotal)}</td>
                  <td>{(m.insertTotal / Math.max(1, m.sortTotal)).toFixed(1)}×</td>
                  <td>{m.passes}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      }
    >
      <div ref={ref}>
        <TooltipHost>
          <Chart model={model} t={t} avail={width} N={N} />
        </TooltipHost>
      </div>
    </VizPanel>
  );
}
