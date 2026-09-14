import { useMemo, useState } from 'react';
import {
  VizPanel,
  Segmented,
  Slider,
  Choice,
  Legend,
  Stats,
  Note,
  TooltipHost,
  useTip,
  makeRng,
  fmtTime,
  fmtNum,
  useSize,
} from './Viz';

/**
 * Why "a load can fault" is a concurrency problem, not an I/O problem.
 *
 * A discrete-event simulation of N worker threads contending for one latch
 * (think: a B-tree page latch, or a buffer-pool hash-bucket lock).
 *
 *  - mmap:     the miss is discovered by the MMU *inside* the critical section,
 *              so the latch is held for the whole device read and everyone queues.
 *  - explicit: the buffer pool knows residency from its own hash table, so the
 *              read happens before/outside the latch and the critical section
 *              stays a few hundred nanoseconds.
 *
 * Deterministic: the per-op miss draw comes from makeRng and is shared by both
 * modes, so the only thing that changes between them is where the I/O happens.
 */

const BASE_OPS = 48; // work is split evenly over the threads, so every thread finishes its share
const CS_NS = 400; // useful work inside the critical section
const THINK_NS = 2_000; // work outside any latch
const DEVICES = [
  { id: 'nvme', label: 'NVMe SSD (20 µs)', ns: 20_000 },
  { id: 'sata', label: 'SATA SSD (150 µs)', ns: 150_000 },
  { id: 'ebs', label: 'Network disk (600 µs)', ns: 600_000 },
  { id: 'hdd', label: 'HDD (10 ms)', ns: 10_000_000 },
] as const;

type Mode = 'mmap' | 'explicit';
type Kind = 'io' | 'wait' | 'cs' | 'fault' | 'think';
type Seg = { t: number; start: number; end: number; kind: Kind; op: number };

type Result = {
  segs: Seg[];
  end: number;
  ops: number;
  waits: number[];
  latchBusy: number;
  latchStalled: number;
  misses: number;
};

function simulate(threads: number, missPct: number, ioNs: number, mode: Mode, draws: number[]): Result {
  // Every thread runs the same number of operations, so a thread parked on a slow
  // device still has to finish its share — no mode gets to "win" by never completing.
  const perThread = Math.max(3, Math.round(BASE_OPS / threads));
  const ops = perThread * threads;

  const segs: Seg[] = [];
  const waits: number[] = [];
  const ready = Array.from({ length: threads }, (_, i) => i * 50); // when the thread starts its op
  const reqAt = new Array<number>(threads).fill(0); // when it will ask for the latch
  const missOf = new Array<boolean>(threads).fill(false);
  const opOf = new Array<number>(threads).fill(0);
  const doneCount = new Array<number>(threads).fill(0);
  let latchFree = 0;
  let latchBusy = 0;
  let latchStalled = 0;
  let misses = 0;
  let end = 0;

  // Begin one operation on thread i. The miss draw is indexed by (thread, op), so both
  // modes see exactly the same hits and misses and only the placement of I/O differs.
  // In "explicit" mode the buffer pool finds the miss in its own hash table and reads
  // the page BEFORE asking for the latch.
  const begin = (i: number) => {
    if (doneCount[i] >= perThread) {
      reqAt[i] = Infinity;
      return;
    }
    const k = i * perThread + doneCount[i];
    const miss = draws[k % draws.length] < missPct;
    missOf[i] = miss;
    opOf[i] = k;
    if (mode === 'explicit' && miss) {
      segs.push({ t: i, start: ready[i], end: ready[i] + ioNs, kind: 'io', op: k });
      reqAt[i] = ready[i] + ioNs;
    } else {
      reqAt[i] = ready[i];
    }
  };

  for (let i = 0; i < threads; i++) begin(i);

  for (let n = 0; n < ops; n++) {
    // the thread that asks for the latch next (lowest index wins ties: deterministic)
    let t = 0;
    for (let i = 1; i < threads; i++) if (reqAt[i] < reqAt[t]) t = i;
    if (!isFinite(reqAt[t])) break;

    const a = reqAt[t];
    const miss = missOf[t];
    const k = opOf[t];
    if (miss) misses++;

    const start = Math.max(a, latchFree);
    if (start > a) segs.push({ t, start: a, end: start, kind: 'wait', op: k });
    waits.push(start - a);

    let done: number;
    if (mode === 'mmap' && miss) {
      // The MMU finds the miss mid-critical-section: the latch is held across the read.
      const half = CS_NS / 2;
      segs.push({ t, start, end: start + half, kind: 'cs', op: k });
      segs.push({ t, start: start + half, end: start + half + ioNs, kind: 'fault', op: k });
      done = start + CS_NS + ioNs;
      segs.push({ t, start: start + half + ioNs, end: done, kind: 'cs', op: k });
      latchStalled += ioNs;
    } else {
      done = start + CS_NS;
      segs.push({ t, start, end: done, kind: 'cs', op: k });
    }
    latchBusy += done - start;
    latchFree = done;
    segs.push({ t, start: done, end: done + THINK_NS, kind: 'think', op: k });
    end = Math.max(end, done);

    doneCount[t]++;
    ready[t] = done + THINK_NS;
    begin(t);
  }
  return { segs, end, ops, waits, latchBusy, latchStalled, misses };
}

function pct(xs: number[], q: number) {
  if (xs.length === 0) return 0;
  const s = [...xs].sort((a, b) => a - b);
  return s[Math.min(s.length - 1, Math.floor(q * (s.length - 1)))];
}

const KIND_COLOR: Record<Kind, string> = {
  wait: 'var(--viz-stale)',
  cs: 'var(--viz-1)',
  fault: 'var(--viz-critical)',
  io: 'var(--viz-3)',
  think: 'var(--viz-neutral)',
};

const KIND_LABEL: Record<Kind, string> = {
  wait: 'Blocked on the latch',
  cs: 'Holding the latch, working',
  fault: 'Holding the latch, stalled in a page fault',
  io: 'Reading the page outside the latch',
  think: 'Running, holds nothing',
};

export default function MmapFaultUnderLatch() {
  const [threads, setThreads] = useState(8);
  const [missPct, setMissPct] = useState(8);
  const [deviceId, setDeviceId] = useState<string>('nvme');
  const [mode, setMode] = useState<Mode>('mmap');
  const [ref, width] = useSize(760);
  const tip = useTip();

  const device = DEVICES.find((d) => d.id === deviceId) ?? DEVICES[0];

  // One fixed draw sequence, shared by both modes: the same operations miss.
  const draws = useMemo(() => {
    const rng = makeRng(4242);
    return Array.from({ length: 64 }, () => rng() * 100);
  }, []);

  const here = simulate(threads, missPct, device.ns, mode, draws);
  const other = simulate(threads, missPct, device.ns, mode === 'mmap' ? 'explicit' : 'mmap', draws);

  const opsPerSec = (r: Result) => (r.ops / r.end) * 1e9;
  const ratio = opsPerSec(here) / opsPerSec(other);

  const labelW = 74;
  const rowH = Math.max(14, Math.min(22, 260 / threads));
  const plotW = Math.max(240, width - labelW - 16);
  const height = threads * rowH + 34;
  const xOf = (t: number) => labelW + (t / here.end) * plotW;

  return (
    <VizPanel
      title="A page fault inside a critical section"
      subtitle="One latch, N worker threads, the same stream of hits and misses. The only difference between the two modes is where the device read happens."
      controls={
        <>
          <Segmented
            label="Where the I/O happens"
            value={mode}
            onChange={setMode}
            options={[
              { value: 'mmap', label: 'mmap: fault inside the latch', title: 'The MMU discovers the miss mid-critical-section' },
              { value: 'explicit', label: 'pread: I/O outside the latch', title: 'The buffer pool knows residency before latching' },
            ]}
          />
          <Slider label="Worker threads" min={2} max={16} value={threads} onChange={setThreads} />
          <Slider
            label="Buffer miss rate"
            min={0}
            max={40}
            value={missPct}
            onChange={setMissPct}
            format={(n) => `${n}%`}
          />
          <Choice
            label="Device"
            value={deviceId}
            onChange={setDeviceId}
            options={DEVICES.map((d) => ({ value: d.id, label: d.label }))}
          />
        </>
      }
      legend={
        <Legend
          items={(['cs', 'fault', 'io', 'wait', 'think'] as Kind[]).map((k) => ({
            label: KIND_LABEL[k],
            color: KIND_COLOR[k],
          }))}
        />
      }
      stats={
        <Stats
          items={[
            { label: 'Throughput', value: `${fmtNum(opsPerSec(here))} ops/s` },
            {
              label: mode === 'mmap' ? 'pread instead' : 'mmap instead',
              value: `${fmtNum(opsPerSec(other))} ops/s`,
            },
            { label: 'This mode vs the other', value: `${ratio.toFixed(2)}×` },
            {
              label: 'Latch held',
              value: `${((here.latchBusy / here.end) * 100).toFixed(0)}%`,
              hint: 'Fraction of wall time the single latch is owned by somebody',
            },
            {
              label: 'Of that, waiting on the device',
              value: `${here.latchBusy > 0 ? ((here.latchStalled / here.latchBusy) * 100).toFixed(0) : '0'}%`,
            },
            { label: 'Mean latch wait', value: fmtTime(here.waits.reduce((a, b) => a + b, 0) / here.waits.length) },
            { label: 'p99 latch wait', value: fmtTime(pct(here.waits, 0.99)) },
            { label: 'Misses', value: `${here.misses} of ${here.ops}` },
          ]}
        />
      }
      note={
        <Note>
          {mode === 'mmap' ? (
            <>
              <strong>The latch is held for the whole device read.</strong> The MMU cannot tell a
              resident page from a cold one until the load executes, so the miss is discovered after
              the latch is taken — {fmtTime(device.ns)} of device time with the critical section
              owned, and every other worker parked behind it. Raise the miss rate or pick a slower
              device and the queue becomes a convoy: throughput stops tracking core count and starts
              tracking <em>one</em> thread's I/O latency.
            </>
          ) : (
            <>
              <strong>The miss is discovered before the latch is taken.</strong> A buffer pool knows
              residency from its own hash table, so it issues the read (or hands it to an io_uring
              queue), then latches only to install the frame. The critical section stays{' '}
              {fmtTime(CS_NS)} regardless of device latency, the reads overlap across threads, and
              throughput scales with workers instead of with the device.
            </>
          )}
        </Note>
      }
      table={
        <table className="viz-table">
          <thead>
            <tr>
              <th>Device</th>
              <th>Mode</th>
              <th>Throughput (ops/s)</th>
              <th>Latch held</th>
              <th>Latch time in faults</th>
              <th>Mean wait</th>
              <th>p99 wait</th>
            </tr>
          </thead>
          <tbody>
            {DEVICES.flatMap((d) =>
              (['mmap', 'explicit'] as Mode[]).map((m) => {
                const r = simulate(threads, missPct, d.ns, m, draws);
                return (
                  <tr key={`${d.id}-${m}`}>
                    <td>{d.label}</td>
                    <td>{m === 'mmap' ? 'fault inside latch' : 'read outside latch'}</td>
                    <td>{fmtNum((r.ops / r.end) * 1e9)}</td>
                    <td>{((r.latchBusy / r.end) * 100).toFixed(0)}%</td>
                    <td>{r.latchBusy > 0 ? ((r.latchStalled / r.latchBusy) * 100).toFixed(0) : '0'}%</td>
                    <td>{fmtTime(r.waits.reduce((a, b) => a + b, 0) / r.waits.length)}</td>
                    <td>{fmtTime(pct(r.waits, 0.99))}</td>
                  </tr>
                );
              }),
            )}
          </tbody>
        </table>
      }
    >
      <div ref={ref}>
        <TooltipHost>
          <svg
            width={labelW + plotW + 12}
            height={height}
            role="img"
            aria-label="Timeline of each worker thread showing latch waiting, latch holding and page faults"
          >
            {[0, 0.25, 0.5, 0.75, 1].map((f) => (
              <g key={f}>
                <line className="viz-grid-line" x1={xOf(f * here.end)} x2={xOf(f * here.end)} y1={14} y2={height - 18} />
                <text x={xOf(f * here.end)} y={10} textAnchor="middle" fontSize={10}>
                  {fmtTime(f * here.end)}
                </text>
              </g>
            ))}

            {Array.from({ length: threads }, (_, i) => (
              <text key={i} x={0} y={20 + i * rowH + rowH * 0.7} fill="var(--viz-ink-2)" fontSize={11}>
                thread {i}
              </text>
            ))}

            {here.segs.map((sg, i) => {
              const right = labelW + plotW;
              const x0 = Math.min(xOf(sg.start), right);
              const x1 = Math.min(xOf(sg.end), right);
              const w = Math.max(1, x1 - x0);
              if (sg.kind === 'think') {
                return (
                  <rect
                    key={i}
                    x={x0}
                    y={18 + sg.t * rowH}
                    width={w}
                    height={rowH - 3}
                    rx={2}
                    fill={KIND_COLOR[sg.kind]}
                  />
                );
              }
              return (
                <rect
                  key={i}
                  x={x0}
                  y={18 + sg.t * rowH}
                  width={w}
                  height={rowH - 3}
                  rx={2}
                  fill={KIND_COLOR[sg.kind]}
                  {...tip(
                    <>
                      <strong>{KIND_LABEL[sg.kind]}</strong>
                      <br />
                      thread {sg.t} · op {sg.op}
                      <br />
                      {fmtTime(sg.end - sg.start)} ({fmtTime(sg.start)} → {fmtTime(sg.end)})
                    </>,
                  )}
                />
              );
            })}
          </svg>
        </TooltipHost>
      </div>
    </VizPanel>
  );
}
