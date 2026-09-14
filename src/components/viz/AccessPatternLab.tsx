import { useState } from 'react';
import { VizPanel, Slider, Segmented, Choice, Legend, Stats, fmtBytes, fmtTime, fmtNum, useSize } from './Viz';

/**
 * Why "sequential vs random" is the most important sentence in storage.
 * One simple model: a request costs a fixed access penalty plus transfer time,
 * and queue depth buys parallelism up to what the device can actually overlap.
 */
type Device = {
  id: string;
  label: string;
  accessNs: number; // random-access penalty (seek + rotation, or controller + NAND read)
  bw: number; // bytes/sec streaming bandwidth
  maxParallel: number; // how many requests the device genuinely overlaps
  seqFactor: number; // fraction of the access penalty still paid when sequential
};

const DEVICES: Device[] = [
  { id: 'hdd', label: 'HDD (7200 rpm)', accessNs: 14_000_000, bw: 200e6, maxParallel: 1, seqFactor: 0.002 },
  { id: 'sata', label: 'SATA SSD', accessNs: 150_000, bw: 550e6, maxParallel: 32, seqFactor: 0.05 },
  { id: 'nvme', label: 'NVMe SSD', accessNs: 20_000, bw: 3.5e9, maxParallel: 128, seqFactor: 0.05 },
  { id: 'dram', label: 'DRAM', accessNs: 100, bw: 20e9, maxParallel: 8, seqFactor: 0.1 },
];

function model(d: Device, sizeBytes: number, qd: number, sequential: boolean) {
  const access = d.accessNs * (sequential ? d.seqFactor : 1);
  const transfer = (sizeBytes / d.bw) * 1e9;
  const latencyNs = access + transfer;
  const parallel = Math.min(qd, d.maxParallel);
  const raw = (sizeBytes / (latencyNs / 1e9)) * parallel; // bytes/sec if latency-bound
  const throughput = Math.min(raw, d.bw); // but never above the device's bandwidth
  return { latencyNs, throughput, iops: throughput / sizeBytes };
}

export default function AccessPatternLab() {
  const [deviceId, setDeviceId] = useState('nvme');
  const [sizeExp, setSizeExp] = useState(12); // 2^12 = 4KB
  const [qd, setQd] = useState(1);
  const [pattern, setPattern] = useState<'random' | 'sequential'>('random');
  const [ref, width] = useSize(700);

  const device = DEVICES.find((d) => d.id === deviceId)!;
  const size = 2 ** sizeExp;
  const now = model(device, size, qd, pattern === 'sequential');
  const rnd = model(device, size, qd, false);
  const seq = model(device, size, qd, true);
  const ratio = seq.throughput / rnd.throughput;

  // Bar chart: this device's random vs sequential throughput at the current settings.
  const barW = Math.max(200, Math.min(width - 120, 520));
  const max = Math.max(rnd.throughput, seq.throughput);
  const bar = (v: number) => Math.max(2, (v / max) * barW);

  return (
    <VizPanel
      title="Random vs sequential, on real devices"
      subtitle="Change the request size and queue depth and watch the gap between random and sequential open and close."
      controls={
        <>
          <Choice
            label="Device"
            value={deviceId}
            onChange={setDeviceId}
            options={DEVICES.map((d) => ({ value: d.id, label: d.label }))}
          />
          <Slider
            label="Request size"
            min={9}
            max={20}
            value={sizeExp}
            onChange={setSizeExp}
            format={() => fmtBytes(size)}
          />
          <Slider
            label="Queue depth"
            min={1}
            max={128}
            value={qd}
            onChange={setQd}
            format={(n) => `${n}${n > device.maxParallel ? ` (device overlaps ${device.maxParallel})` : ''}`}
          />
          <Segmented
            label="Access pattern"
            value={pattern}
            onChange={setPattern}
            options={[
              { value: 'random', label: 'Random' },
              { value: 'sequential', label: 'Sequential' },
            ]}
          />
        </>
      }
      legend={
        <Legend
          items={[
            { label: 'Random access', color: 'var(--viz-2)' },
            { label: 'Sequential access', color: 'var(--viz-1)' },
          ]}
        />
      }
      stats={
        <Stats
          items={[
            { label: 'Latency per request', value: fmtTime(now.latencyNs), hint: 'Access penalty + transfer time' },
            { label: 'Throughput', value: `${fmtBytes(now.throughput)}/s` },
            { label: 'IOPS', value: fmtNum(now.iops) },
            { label: 'Sequential advantage', value: `${ratio < 10 ? ratio.toFixed(1) : fmtNum(ratio)}×` },
          ]}
        />
      }
      note={
        <>
          <strong>
            {ratio > 20
              ? 'Sequential wins by more than an order of magnitude here.'
              : ratio > 3
                ? 'Sequential still wins clearly.'
                : 'At this request size the gap has nearly closed.'}
          </strong>{' '}
          The access penalty is paid once per request, so it dominates small requests and disappears
          into transfer time on big ones — which is exactly why engines read {fmtBytes(8192)} pages
          rather than rows, why an LSM tree turns random writes into sequential ones, and why a
          table scan can beat an index that looks cheaper on paper.
        </>
      }
      table={
        <table className="viz-table">
          <thead>
            <tr>
              <th>Device</th>
              <th>Random</th>
              <th>Sequential</th>
              <th>Ratio</th>
            </tr>
          </thead>
          <tbody>
            {DEVICES.map((d) => {
              const r = model(d, size, qd, false);
              const s = model(d, size, qd, true);
              return (
                <tr key={d.id}>
                  <td>{d.label}</td>
                  <td>{fmtBytes(r.throughput)}/s</td>
                  <td>{fmtBytes(s.throughput)}/s</td>
                  <td>{(s.throughput / r.throughput).toFixed(1)}×</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      }
    >
      <div ref={ref}>
        <svg width={Math.max(320, barW + 130)} height={96} role="img" aria-label="Random versus sequential throughput">
          {[
            { label: 'Random', v: rnd.throughput, c: 'var(--viz-2)', y: 14 },
            { label: 'Sequential', v: seq.throughput, c: 'var(--viz-1)', y: 52 },
          ].map((b) => (
            <g key={b.label}>
              <text x={0} y={b.y + 13} fill="var(--viz-ink)">
                {b.label}
              </text>
              <rect x={78} y={b.y} width={bar(b.v)} height={18} rx={4} fill={b.c} />
              <text
                x={78 + bar(b.v) + 8}
                y={b.y + 13}
                fill="var(--viz-ink-2)"
                style={{ fontVariantNumeric: 'tabular-nums' }}
              >
                {fmtBytes(b.v)}/s
              </text>
            </g>
          ))}
        </svg>
      </div>
    </VizPanel>
  );
}
