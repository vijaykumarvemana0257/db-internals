import { useMemo, useState } from 'react';
import { VizPanel, Segmented, Legend, TooltipHost, useTip, fmtTime, useSize } from './Viz';

type Level = {
  name: string;
  ns: number;
  kind: 'cpu' | 'memory' | 'storage' | 'network';
  note: string;
};

/** Order matters: this is the ladder, fastest first. */
const LEVELS: Level[] = [
  { name: 'L1 cache reference', ns: 1, kind: 'cpu', note: 'A few CPU cycles. The 64-byte cache line is the unit — this is why in-memory layouts fight over cache lines.' },
  { name: 'Branch mispredict', ns: 3, kind: 'cpu', note: 'Pipeline flush. Why branchless code and vectorized execution win in query engines.' },
  { name: 'L2 cache reference', ns: 4, kind: 'cpu', note: 'Still on-core. Sorting and hashing are tuned to stay inside L2/L3.' },
  { name: 'Mutex lock/unlock', ns: 17, kind: 'cpu', note: 'Uncontended. Contended is far worse — the reason engines use latches, not locks, on hot paths.' },
  { name: 'L3 cache reference', ns: 30, kind: 'cpu', note: 'Shared across cores. Crossing it means coherence traffic.' },
  { name: 'Main memory (DRAM)', ns: 100, kind: 'memory', note: 'A random DRAM access. ~100× an L1 hit — the TLB and cache lines are what buy you locality.' },
  { name: 'Read 1MB sequentially from RAM', ns: 3_000, kind: 'memory', note: 'Bandwidth, not latency. Scanning column data lives here.' },
  { name: 'NVMe SSD random 4KB read', ns: 20_000, kind: 'storage', note: 'One page fault to NVMe ≈ 200 DRAM accesses. This is the buffer-pool miss you pay for.' },
  { name: 'Read 1MB sequentially from NVMe', ns: 80_000, kind: 'storage', note: 'Only ~4× one random 4KB read, for 256× the data. This ratio is why engines prefer big sequential I/O.' },
  { name: 'SATA SSD random 4KB read', ns: 150_000, kind: 'storage', note: 'Same NAND as NVMe, but the AHCI/SATA path adds protocol latency and caps queue depth at 32.' },
  { name: 'Same-datacenter round trip', ns: 500_000, kind: 'network', note: 'A network hop costs more than a local NVMe read — replication and 2PC pay this per round.' },
  { name: 'HDD seek', ns: 10_000_000, kind: 'storage', note: 'Mechanical: arm movement plus rotation. ~10⁷ × an L1 hit. B-trees exist because of this number.' },
  { name: 'Cross-continent round trip', ns: 150_000_000, kind: 'network', note: 'Speed of light in fiber. No protocol fixes it — geo-distributed commits pay it every time.' },
];

const KIND_COLOR: Record<Level['kind'], string> = {
  cpu: 'var(--viz-1)',
  memory: 'var(--viz-3)',
  storage: 'var(--viz-2)',
  network: 'var(--viz-7)',
};

/** If an L1 hit took one second, everything else scales by this factor. */
const HUMAN_SCALE = 1e9;

function humanTime(ns: number) {
  const s = (ns * HUMAN_SCALE) / 1e9;
  if (s < 60) return `${s < 10 ? s.toFixed(0) : Math.round(s)} sec`;
  if (s < 3600) return `${(s / 60).toFixed(1)} min`;
  if (s < 86400) return `${(s / 3600).toFixed(1)} hours`;
  if (s < 86400 * 365) return `${(s / 86400).toFixed(1)} days`;
  return `${(s / (86400 * 365)).toFixed(1)} years`;
}

export default function LatencyLadder() {
  const [scale, setScale] = useState<'real' | 'human'>('real');
  const [ref, width] = useSize(760);
  const tip = useTip();

  const labelW = Math.min(260, Math.max(150, width * 0.32));
  const valueW = 92;
  const rowH = 26;
  const plotW = Math.max(80, width - labelW - valueW - 16);
  const height = LEVELS.length * rowH + 26;

  const { lo, hi } = useMemo(
    () => ({ lo: Math.log10(LEVELS[0].ns), hi: Math.log10(LEVELS[LEVELS.length - 1].ns) }),
    [],
  );
  const x = (ns: number) => ((Math.log10(ns) - lo) / (hi - lo)) * plotW;

  // Decade gridlines: 1ns, 10ns, … 100ms
  // Label every other decade when the plot is narrow, so ticks never collide.
  const decades = useMemo(() => {
    const out: { v: number; label: boolean }[] = [];
    const first = Math.ceil(lo);
    const every = plotW / (Math.floor(hi) - first + 1) < 54 ? 2 : 1;
    for (let e = first; e <= Math.floor(hi); e++) {
      out.push({ v: 10 ** e, label: (e - first) % every === 0 });
    }
    return out;
  }, [lo, hi, plotW]);

  return (
    <VizPanel
      title="The latency ladder"
      subtitle="Every storage-engine decision is an answer to these numbers. Bars are log scale — each gridline is 10× the one before."
      controls={
        <Segmented
          label="Show times as"
          value={scale}
          onChange={setScale}
          options={[
            { value: 'real', label: 'Real time' },
            { value: 'human', label: 'If L1 took 1 second' },
          ]}
        />
      }
      legend={
        <Legend
          items={[
            { label: 'CPU', color: KIND_COLOR.cpu },
            { label: 'Memory', color: KIND_COLOR.memory },
            { label: 'Storage', color: KIND_COLOR.storage },
            { label: 'Network', color: KIND_COLOR.network },
          ]}
        />
      }
      note={
        scale === 'human' ? (
          <>
            <strong>Scaled to human time,</strong> an L1 hit is one second and an HDD seek is four
            months. A database is mostly machinery for not doing the slow thing.
          </>
        ) : (
          <>
            <strong>Two ratios do most of the work in this site:</strong> DRAM is ~100× an L1 hit,
            and an HDD seek is ~10⁵× a DRAM access. Pages, buffer pools and B-trees all exist to
            keep you on the fast rungs.
          </>
        )
      }
      table={
        <table className="viz-table">
          <thead>
            <tr>
              <th>Operation</th>
              <th>Latency</th>
              <th>If L1 took 1 second</th>
            </tr>
          </thead>
          <tbody>
            {LEVELS.map((l) => (
              <tr key={l.name}>
                <td>{l.name}</td>
                <td>{fmtTime(l.ns)}</td>
                <td>{humanTime(l.ns)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      }
    >
      <div ref={ref}>
        <TooltipHost>
          <svg width={width} height={height} role="img" aria-label="Latency of operations on a logarithmic scale">
            {decades.map((d) => (
              <g key={d.v}>
                <line
                  className="viz-grid-line"
                  x1={labelW + x(d.v)}
                  x2={labelW + x(d.v)}
                  y1={14}
                  y2={height - 12}
                />
                {d.label ? (
                  <text x={labelW + x(d.v)} y={10} textAnchor="middle" fontSize={10}>
                    {fmtTime(d.v)}
                  </text>
                ) : null}
              </g>
            ))}

            {LEVELS.map((l, i) => {
              const y = 20 + i * rowH;
              const w = Math.max(3, x(l.ns));
              return (
                <g
                  key={l.name}
                  {...tip(
                    <>
                      <strong>{l.name}</strong>
                      <br />
                      {fmtTime(l.ns)} · {humanTime(l.ns)} at human scale
                      <br />
                      <span style={{ color: 'var(--viz-ink-2)' }}>{l.note}</span>
                    </>,
                  )}
                  style={{ cursor: 'help' }}
                >
                  <rect x={0} y={y - 3} width={width} height={rowH - 2} fill="transparent" />
                  <text x={labelW - 8} y={y + 12} textAnchor="end" fill="var(--viz-ink)">
                    {l.name.length > 34 ? l.name.slice(0, 33) + '…' : l.name}
                  </text>
                  <rect
                    x={labelW}
                    y={y + 2}
                    width={w}
                    height={13}
                    rx={4}
                    fill={KIND_COLOR[l.kind]}
                    stroke="var(--viz-surface)"
                    strokeWidth={2}
                  />
                  <text
                    x={labelW + plotW + 8}
                    y={y + 12}
                    fill="var(--viz-ink-2)"
                    style={{ fontVariantNumeric: 'tabular-nums' }}
                  >
                    {scale === 'real' ? fmtTime(l.ns) : humanTime(l.ns)}
                  </text>
                </g>
              );
            })}
          </svg>
        </TooltipHost>
      </div>
    </VizPanel>
  );
}
