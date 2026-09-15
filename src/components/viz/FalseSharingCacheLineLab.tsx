import { useMemo, useState } from 'react';
import { VizPanel, Segmented, Slider, Legend, Stats, Note, fmtNum } from './Viz';

/**
 * Per-thread counters and the cache line they live on.
 *
 * Each core loops: do W units of private work, then increment its counter. The counter's 8 bytes live on a
 * 64-byte cache line:
 *   padded  — every counter on its own line (alignas(64) plus padding, as InnoDB's Counter::Shard and
 *             RocksDB's per-core StatisticsData do);
 *   packed  — all counters adjacent, up to 8 × 8 bytes on one 64-byte line;
 *   shared  — one counter that every core increments with fetch_add.
 *
 * Cost model (abstract units, not measurements):
 *   - an increment of a line this core wrote last costs 1 unit;
 *   - an increment of a line another core wrote last first pulls the line over, K units, then costs 1;
 *   - while a line is being written or moved, no other core can write it; cores queue in arrival order.
 * Private work never touches shared lines. Throughput is reported relative to one core doing the same loop.
 * The simulation is deterministic and bounded (fixed horizon, at most 8 cores).
 */

export type Layout = 'padded' | 'packed' | 'shared';
export type Seg = { core: number; start: number; end: number; kind: 'work' | 'wait' | 'move' | 'inc' };

export function simulate(layout: Layout, cores: number, K: number, W: number, horizon = 6000, traceUntil = 0) {
  const n = Math.max(1, Math.min(8, Math.round(cores)));
  const lineOf = (i: number) => (layout === 'padded' ? i : 0);
  const readyAt = Array.from({ length: n }, () => W);
  const freeAt = new Array<number>(n).fill(0);
  const lastWriter = new Array<number>(n).fill(-1);
  const incs = new Array<number>(n).fill(0);
  let moves = 0;
  let waited = 0;
  const trace: Seg[] = [];
  if (traceUntil > 0 && W > 0) for (let i = 0; i < n; i++) trace.push({ core: i, start: 0, end: W, kind: 'work' });
  for (let guard = 0; guard < 400000; guard++) {
    let i = 0;
    for (let j = 1; j < n; j++) if (readyAt[j] < readyAt[i]) i = j;
    if (readyAt[i] >= horizon) break;
    const L = lineOf(i);
    const start = Math.max(readyAt[i], freeAt[L]);
    const move = lastWriter[L] !== -1 && lastWriter[L] !== i;
    const end = start + (move ? K : 0) + 1;
    if (end > horizon) {
      readyAt[i] = horizon;
      continue;
    }
    if (traceUntil > 0 && readyAt[i] < traceUntil) {
      if (start > readyAt[i]) trace.push({ core: i, start: readyAt[i], end: start, kind: 'wait' });
      if (move) trace.push({ core: i, start, end: start + K, kind: 'move' });
      trace.push({ core: i, start: end - 1, end, kind: 'inc' });
      if (W > 0) trace.push({ core: i, start: end, end: end + W, kind: 'work' });
    }
    waited += start - readyAt[i];
    if (move) moves++;
    freeAt[L] = end;
    lastWriter[L] = i;
    incs[i]++;
    readyAt[i] = end + W;
  }
  const total = incs.reduce((a, b) => a + b, 0);
  // one core alone: W units of work then a 1-unit increment, repeated; it never has to pull its line over
  const single = Math.floor(horizon / (W + 1));
  return {
    cores: n,
    total,
    relative: total / Math.max(1, single),
    movesPerInc: total ? moves / total : 0,
    waitShare: waited / (n * horizon),
    trace,
  };
}

export function sweep(K: number, W: number) {
  const layouts: Layout[] = ['padded', 'packed', 'shared'];
  return Object.fromEntries(layouts.map((l) => [l, Array.from({ length: 8 }, (_, k) => simulate(l, k + 1, K, W, 6000).relative)])) as Record<Layout, number[]>;
}

// Curves are told apart by dash pattern and direct labels, so they stay clear of the Gantt's state colors.
const LAYOUT_COLOR: Record<Layout, string> = { padded: 'var(--viz-ink)', packed: 'var(--viz-ink-2)', shared: 'var(--viz-ink-muted)' };
const LAYOUT_DASH: Record<Layout, string | undefined> = { padded: undefined, packed: '7 4', shared: '2 3' };
const LAYOUT_NAME: Record<Layout, string> = { padded: 'padded: a line per counter', packed: 'packed: one 64-byte line', shared: 'one shared counter' };
const KIND_COLOR: Record<Seg['kind'], string> = { work: 'var(--viz-neutral)', wait: 'var(--viz-warning)', move: 'var(--viz-serious)', inc: 'var(--viz-1)' };

export default function FalseSharingCacheLineLab() {
  const [layout, setLayout] = useState<Layout>('padded');
  const [cores, setCores] = useState(4);
  const [K, setK] = useState(10);
  const [W, setW] = useState(2);
  const TRACE = 90;
  const r = useMemo(() => simulate(layout, cores, K, W, 6000, TRACE), [layout, cores, K, W]);
  const curves = useMemo(() => sweep(K, W), [K, W]);

  const Wd = 680;
  // --- memory layout
  const lineW = 152;
  const slot = lineW / 8;
  const lines = layout === 'padded' ? cores : 1;
  const perRow = 4;
  const rows = Math.ceil(lines / perRow);
  const memH = 26 + rows * 34;
  // --- gantt
  const gTop = memH + 26;
  const rowH = 15;
  const gLeft = 64;
  const gX = (t: number) => gLeft + (Math.min(t, TRACE) / TRACE) * (Wd - gLeft - 16);
  const gH = cores * rowH + 22;
  // --- chart
  const cTop = gTop + gH + 26;
  const cH = 150;
  const cLeft = 64;
  const cW = Wd - cLeft - 190;
  const maxY = 8;
  const cx = (k: number) => cLeft + ((k - 1) / 7) * cW;
  const cy = (v: number) => cTop + cH - (Math.min(v, maxY) / maxY) * cH;
  const H = cTop + cH + 30;

  const padLabel = layout === 'shared' ? 'total' : null;

  return (
    <VizPanel
      title="Counters sharing one cache line"
      subtitle="Every core loops over some private work and one increment of its own counter, or of the one shared counter. Move the counters between padded lines and one shared 64-byte line, and watch the line travel between cores."
      controls={
        <>
          <Segmented
            label="Counter placement"
            value={layout}
            onChange={setLayout}
            options={[
              { value: 'padded', label: 'Padded (a line each)' },
              { value: 'packed', label: 'Packed (one line)' },
              { value: 'shared', label: 'One shared counter' },
            ]}
          />
          <Slider label="Cores" min={1} max={8} value={cores} onChange={setCores} />
          <Slider label="Line transfer cost (units)" min={1} max={40} value={K} onChange={setK} />
          <Slider label="Private work per increment (units)" min={0} max={60} value={W} onChange={setW} />
        </>
      }
      legend={
        <Legend
          items={[
            { label: 'increment', color: KIND_COLOR.inc },
            { label: 'private work', color: KIND_COLOR.work },
            { label: 'waiting for the line', color: KIND_COLOR.wait },
            { label: 'pulling the line over', color: KIND_COLOR.move },
          ]}
        />
      }
      stats={
        <Stats
          items={[
            { label: `Throughput vs 1 core`, value: `${fmtNum(r.relative, 2)}×`, hint: 'Increments completed, relative to one core running the same loop alone.' },
            { label: 'Line transfers per increment', value: fmtNum(r.movesPerInc, 2) },
            { label: 'Core time spent waiting', value: `${fmtNum(r.waitShare * 100, 0)}%` },
            { label: 'Cache lines written', value: fmtNum(lines) },
          ]}
        />
      }
      note={
        <Note>
          <strong>
            {cores} core{cores === 1 ? '' : 's'}, {LAYOUT_NAME[layout]}: {fmtNum(r.relative, 2)}× one core.
          </strong>{' '}
          {cores === 1
            ? 'One core never loses its line, so placement does not matter yet — add cores.'
            : layout === 'padded'
              ? 'No core ever writes a line another core wrote, so the counters scale with the cores. Now pack them onto one line: nothing in the code changes.'
              : layout === 'packed'
                ? `The counters are logically independent, but they share a line, so ${fmtNum(r.movesPerInc * 100, 1)}% of increments first pull the line from another core. This is false sharing, and its curve is the same as one truly shared counter. Raise the private work to see the damage shrink as the line cools.`
                : 'fetch_add needs no lock, but every increment still writes the one line every core wants, so the line — not a mutex — serializes the cores. Its curve is identical to the packed layout.'}
        </Note>
      }
      table={
        <table className="viz-table">
          <thead>
            <tr>
              <th>Cores</th>
              <th>Padded</th>
              <th>Packed on one line</th>
              <th>One shared counter</th>
            </tr>
          </thead>
          <tbody>
            {curves.padded.map((_, k) => (
              <tr key={k}>
                <td>{k + 1}</td>
                <td>{fmtNum(curves.padded[k], 2)}×</td>
                <td>{fmtNum(curves.packed[k], 2)}×</td>
                <td>{fmtNum(curves.shared[k], 2)}×</td>
              </tr>
            ))}
          </tbody>
        </table>
      }
    >
      <svg viewBox={`0 0 ${Wd} ${H}`} width={Wd} height={H} role="img" aria-label={`${LAYOUT_NAME[layout]} with ${cores} cores: throughput ${fmtNum(r.relative, 2)} times one core`}>
        <text x={12} y={16} fontSize={11} fill="var(--viz-ink-2)">
          memory: each box is a 64-byte cache line, each cell an 8-byte slot
        </text>
        {Array.from({ length: lines }, (_, li) => {
          const x = 12 + (li % perRow) * (lineW + 14);
          const y = 26 + Math.floor(li / perRow) * 34;
          return (
            <g key={li}>
              {Array.from({ length: 8 }, (_, s) => {
                const used = layout === 'padded' ? s === 0 : layout === 'packed' ? s < cores : s === 0;
                const label = !used ? '' : layout === 'padded' ? `c${li}` : layout === 'packed' ? `c${s}` : padLabel;
                return (
                  <g key={s}>
                    <rect x={x + s * slot} y={y} width={slot} height={22} fill={used ? 'var(--viz-surface)' : 'var(--viz-plane)'} stroke="var(--viz-border)" />
                    {label ? (
                      <text x={x + s * slot + slot / 2} y={y + 15} fontSize={layout === 'shared' ? 8 : 9} textAnchor="middle" fill="var(--viz-ink)">
                        {label}
                      </text>
                    ) : null}
                  </g>
                );
              })}
              <rect x={x} y={y} width={lineW} height={22} fill="none" stroke="var(--viz-ink-2)" strokeWidth={1.5} />
            </g>
          );
        })}

        <text x={12} y={gTop - 8} fontSize={11} fill="var(--viz-ink-2)">
          first {TRACE} time units, per core
        </text>
        {Array.from({ length: cores }, (_, i) => (
          <text key={i} x={12} y={gTop + i * rowH + 11} fontSize={10} fill="var(--viz-ink)">
            core {i}
          </text>
        ))}
        {r.trace
          .filter((sg) => sg.start < TRACE)
          .map((sg, k) => (
            <rect
              key={k}
              x={gX(sg.start)}
              y={gTop + sg.core * rowH + 2}
              width={Math.max(0.8, gX(sg.end) - gX(sg.start))}
              height={rowH - 4}
              fill={KIND_COLOR[sg.kind]}
            />
          ))}
        <line x1={gLeft} x2={Wd - 16} y1={gTop + cores * rowH + 2} y2={gTop + cores * rowH + 2} stroke="var(--viz-axis)" />

        {/* throughput chart */}
        <text x={12} y={cTop - 10} fontSize={11} fill="var(--viz-ink-2)">
          throughput relative to one core, by core count
        </text>
        {[0, 2, 4, 6, 8].map((v) => (
          <g key={v}>
            <line x1={cLeft} x2={cLeft + cW} y1={cy(v)} y2={cy(v)} stroke="var(--viz-grid)" />
            <text x={cLeft - 6} y={cy(v) + 4} fontSize={10} textAnchor="end" fill="var(--viz-ink-2)">
              {v}×
            </text>
          </g>
        ))}
        {Array.from({ length: 8 }, (_, k) => (
          <text key={k} x={cx(k + 1)} y={cTop + cH + 16} fontSize={10} textAnchor="middle" fill="var(--viz-ink-2)">
            {k + 1}
          </text>
        ))}
        <line x1={cx(cores)} x2={cx(cores)} y1={cTop} y2={cTop + cH} stroke="var(--viz-ink-2)" strokeDasharray="4 3" />
        {(['padded', 'packed', 'shared'] as Layout[]).map((l) => {
          const pts = curves[l].map((v, k) => `${cx(k + 1)},${cy(v)}`).join(' ');
          const endY = cy(curves[l][7]);
          const labelY = l === 'padded' ? endY + 4 : l === 'packed' ? endY - 4 : endY + 12;
          return (
            <g key={l}>
              <polyline points={pts} fill="none" stroke={LAYOUT_COLOR[l]} strokeWidth={l === layout ? 3 : 1.6} strokeDasharray={LAYOUT_DASH[l]} />
              <circle cx={cx(cores)} cy={cy(curves[l][cores - 1])} r={l === layout ? 5 : 3} fill={LAYOUT_COLOR[l]} />
              <text x={cx(8) + 10} y={labelY} fontSize={10} fontWeight={l === layout ? 600 : 400} fill="var(--viz-ink)">
                {l === 'padded' ? '── ' : l === 'packed' ? '- - ' : '··· '}
                {LAYOUT_NAME[l]}
              </text>
            </g>
          );
        })}
      </svg>
    </VizPanel>
  );
}
