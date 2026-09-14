import { useEffect, useRef, useState } from 'react';
import {
  VizPanel,
  Slider,
  Segmented,
  Check,
  Button,
  Legend,
  Stats,
  Note,
  makeRng,
  fmtNum,
  useTicker,
  useSize,
} from './Viz';

/**
 * How a request actually gets to the drive: one blocking pread() per thread,
 * versus io_uring's mmap'd submission and completion rings.
 *
 * The simulation is a small multi-server queue with an explicit CPU budget.
 * Every event costs microseconds of CPU — a syscall, a context switch, an SQE
 * store, a CQE load — and the app can only issue as fast as that budget allows.
 * That is the whole point: at high queue depth the sync path runs out of CPU
 * long before the drive runs out of parallelism, while io_uring amortizes one
 * syscall over a whole batch and reaches the device ceiling.
 */

const SERVICE_US = 20; // NVMe 4 KB random read, queue-depth-1 latency
const DEV_SERVERS = 14; // how many of those the drive genuinely overlaps -> 700k IOPS
const RING = 32; // ring entries (io_uring_setup entries=32)

const SYSCALL_US = 1.2; // syscall entry/exit with Meltdown/Spectre mitigations on
const CTX_US = 1.5; // one context switch (sleep or wake)
const SQE_US = 0.25; // fill an SQE / read a CQE out of the mmap'd ring
const SQPOLL_ENTER_US = 0.15; // a store + a barrier, no ring transition

type Req = { id: number; tSub: number };
type InDev = { r: Req; start: number; done: number };

type Sim = {
  t: number;
  nextId: number;
  sq: Req[];
  dev: InDev[];
  cq: Req[];
  servers: number[];
  rng: () => number;
  sqHead: number;
  sqTail: number;
  cqHead: number;
  cqTail: number;
  syscalls: number;
  completed: number;
  latSum: number;
  winDone: number;
  winUs: number;
  iops: number;
  cpuUsed: number;
  cpuTot: number;
};

function newSim(): Sim {
  return {
    t: 0,
    nextId: 1,
    sq: [],
    dev: [],
    cq: [],
    servers: new Array(DEV_SERVERS).fill(0),
    rng: makeRng(20240917),
    sqHead: 0,
    sqTail: 0,
    cqHead: 0,
    cqTail: 0,
    syscalls: 0,
    completed: 0,
    latSum: 0,
    winDone: 0,
    winUs: 0,
    iops: 0,
    cpuUsed: 0,
    cpuTot: 0,
  };
}

type Cfg = {
  sync: boolean;
  qd: number;
  batch: number;
  cores: number;
  sqpoll: boolean;
};

function costs(c: Cfg) {
  const effCores = c.sync ? c.cores : c.sqpoll ? Math.max(0.6, c.cores - 1) : c.cores;
  const effBatch = Math.max(1, Math.min(c.batch, c.qd));
  if (c.sync) {
    return {
      effCores,
      effBatch: 1,
      prep: 0,
      submit: SYSCALL_US + CTX_US, // pread(): enter the kernel, then sleep
      enter: 0,
      reap: CTX_US + 0.3, // wake up, return to userspace
      perReq: SYSCALL_US + 2 * CTX_US + 0.3,
      syscallsPerReq: 1,
    };
  }
  const enter = c.sqpoll ? SQPOLL_ENTER_US : SYSCALL_US;
  return {
    effCores,
    effBatch,
    prep: SQE_US,
    submit: 0,
    enter,
    reap: SQE_US,
    perReq: SQE_US * 2 + enter / effBatch,
    syscallsPerReq: c.sqpoll ? 0 : 1 / effBatch,
  };
}

const STEP_US = 4;
const US_PER_TICK = 640;

function step(s: Sim, c: Cfg, k: ReturnType<typeof costs>, dt: number) {
  const cpuTotal = k.effCores * dt;
  let cpu = cpuTotal;

  // 1. the device posts whatever finished
  for (let i = s.dev.length - 1; i >= 0; i--) {
    if (s.dev[i].done <= s.t) {
      s.cq.push(s.dev[i].r);
      s.cqTail++;
      s.dev.splice(i, 1);
    }
  }

  // 2. the app reaps completions (a CQE load, or a thread waking from pread)
  while (s.cq.length > 0 && cpu >= k.reap) {
    const r = s.cq.shift()!;
    cpu -= k.reap;
    s.cqHead++;
    s.completed++;
    s.winDone++;
    s.latSum += s.t - r.tSub;
  }

  // 3. the app prepares new requests, keeping `qd` outstanding
  let outstanding = s.sq.length + s.dev.length + s.cq.length;
  const sqCap = c.sync ? c.qd : RING; // the sync path has no ring: a free thread IS the slot
  while (outstanding < c.qd && s.sq.length < sqCap && cpu >= k.prep) {
    s.sq.push({ id: s.nextId++, tSub: s.t });
    s.sqTail++;
    cpu -= k.prep;
    outstanding++;
  }

  // 4. submission
  const issue = (r: Req) => {
    let best = 0;
    for (let i = 1; i < s.servers.length; i++) if (s.servers[i] < s.servers[best]) best = i;
    const start = Math.max(s.t, s.servers[best]);
    const done = start + SERVICE_US * (0.8 + s.rng() * 0.4);
    s.servers[best] = done;
    s.dev.push({ r, start, done });
    s.sqHead++;
  };

  if (c.sync) {
    // one syscall per request; the calling thread is parked until it returns
    while (s.sq.length > 0 && cpu >= k.submit) {
      cpu -= k.submit;
      s.syscalls++;
      issue(s.sq.shift()!);
    }
  } else {
    // one io_uring_enter() for a whole batch of SQEs
    while (s.sq.length >= k.effBatch || (s.sq.length > 0 && outstanding >= c.qd)) {
      if (cpu < k.enter) break;
      cpu -= k.enter;
      if (!c.sqpoll) s.syscalls++;
      const n = Math.min(k.effBatch, s.sq.length);
      for (let i = 0; i < n; i++) issue(s.sq.shift()!);
    }
  }

  s.cpuUsed += cpuTotal - cpu;
  s.cpuTot += cpuTotal;
  s.t += dt;
  s.winUs += dt;
  if (s.winUs >= 3000) {
    const inst = (s.winDone / s.winUs) * 1e6;
    s.iops = s.iops === 0 ? inst : s.iops * 0.7 + inst * 0.3;
    s.winDone = 0;
    s.winUs = 0;
  }
}

/** Steady-state ceilings, used for the stat tiles and the comparison table. */
function ceilings(c: Cfg) {
  const k = costs(c);
  const cpuCeil = (k.effCores / k.perReq) * 1e6;
  const devCeil = (Math.min(c.qd, DEV_SERVERS) / SERVICE_US) * 1e6;
  return { k, cpuCeil, devCeil, iops: Math.min(cpuCeil, devCeil) };
}

type Snap = {
  sq: number;
  dev: number;
  queued: number;
  cq: number;
  sqHead: number;
  sqTail: number;
  cqHead: number;
  cqTail: number;
  iops: number;
  lat: number;
  cpu: number;
  syscalls: number;
  completed: number;
};

const EMPTY: Snap = {
  sq: 0,
  dev: 0,
  queued: 0,
  cq: 0,
  sqHead: 0,
  sqTail: 0,
  cqHead: 0,
  cqTail: 0,
  iops: 0,
  lat: 0,
  cpu: 0,
  syscalls: 0,
  completed: 0,
};

export default function SubmissionRingLab() {
  const [path, setPath] = useState<'pread' | 'uring'>('pread');
  const [qd, setQd] = useState(8);
  const [batch, setBatch] = useState(8);
  const [cores, setCores] = useState(1);
  const [sqpoll, setSqpoll] = useState(false);
  const [running, setRunning] = useState(true);
  const [snap, setSnap] = useState<Snap>(EMPTY);
  const [ref, width] = useSize(760);

  const cfg: Cfg = { sync: path === 'pread', qd, batch, cores, sqpoll };
  const { k, cpuCeil, devCeil } = ceilings(cfg);

  const simRef = useRef<Sim>(newSim());
  const cfgRef = useRef(cfg);
  cfgRef.current = cfg;

  // Any parameter change restarts the measurement; steady state is what we report.
  useEffect(() => {
    simRef.current = newSim();
    setSnap(EMPTY);
  }, [path, qd, batch, cores, sqpoll]);

  useTicker(() => {
    const s = simRef.current;
    const c = cfgRef.current;
    const kk = costs(c);
    for (let us = 0; us < US_PER_TICK; us += STEP_US) step(s, c, kk, STEP_US);
    setSnap({
      sq: s.sq.length,
      dev: s.dev.length,
      queued: s.dev.filter((d) => d.start > s.t).length,
      cq: s.cq.length,
      sqHead: s.sqHead,
      sqTail: s.sqTail,
      cqHead: s.cqHead,
      cqTail: s.cqTail,
      iops: s.iops,
      lat: s.completed ? s.latSum / s.completed : 0,
      cpu: s.cpuTot ? s.cpuUsed / s.cpuTot : 0,
      syscalls: s.completed ? s.syscalls / s.completed : 0,
      completed: s.completed,
    });
  }, running);

  /* ------------------------------------------------------------ geometry */
  const svgW = Math.max(640, width);
  const labelW = 168;
  const plotW = svgW - labelW - 120;
  const slotW = Math.min(20, Math.floor(plotW / RING));
  const gap = 3;
  const rowY = [30, 118, 206];
  const height = 268;

  // Report the rate the simulation actually measured; the analytic cost model is
  // only the seed for the first frame, before anything has completed.
  const syscallsPerReq = snap.completed > 0 ? snap.syscalls : k.syscallsPerReq;

  const threads = cfg.sync ? Math.min(qd, snap.sq + snap.dev + snap.cq || qd) : 1;

  function SlotRow({
    y,
    n,
    filled,
    color,
    label,
    sub,
    right,
  }: {
    y: number;
    n: number;
    filled: number;
    color: string;
    label: string;
    sub: string;
    right: string;
  }) {
    const shown = Math.min(n, RING);
    return (
      <g>
        <text x={labelW - 12} y={y + 12} textAnchor="end" fill="var(--viz-ink)">
          {label}
        </text>
        <text x={labelW - 12} y={y + 26} textAnchor="end" fill="var(--viz-ink-muted)" fontSize={10}>
          {sub}
        </text>
        {Array.from({ length: shown }, (_, i) => (
          <rect
            key={i}
            x={labelW + i * (slotW + gap)}
            y={y}
            width={slotW}
            height={22}
            rx={3}
            fill={i < Math.min(filled, shown) ? color : 'var(--viz-neutral)'}
            stroke="var(--viz-axis)"
            strokeWidth={0.75}
          />
        ))}
        {n > RING ? (
          <text x={labelW + shown * (slotW + gap) + 6} y={y + 16} fill="var(--viz-ink-muted)" fontSize={10}>
            +{n - RING} more
          </text>
        ) : null}
        <text
          x={labelW + plotW + 12}
          y={y + 16}
          fill="var(--viz-ink-2)"
          fontSize={10}
          style={{ fontVariantNumeric: 'tabular-nums' }}
        >
          {right}
        </text>
      </g>
    );
  }

  const tableRows = [
    { name: 'sync pread(), 1 thread per I/O', c: { ...cfg, sync: true, sqpoll: false } },
    { name: 'io_uring, submit 1 SQE per enter', c: { ...cfg, sync: false, batch: 1, sqpoll: false } },
    { name: 'io_uring, batch of 8', c: { ...cfg, sync: false, batch: 8, sqpoll: false } },
    { name: 'io_uring, batch of 32', c: { ...cfg, sync: false, batch: 32, sqpoll: false } },
    { name: 'io_uring, batch of 32 + SQPOLL', c: { ...cfg, sync: false, batch: 32, sqpoll: true } },
  ];

  return (
    <VizPanel
      title="Submission path: blocking pread() versus the io_uring rings"
      subtitle="Same drive, same 4 KB random reads. Raise the queue depth and watch where the ceiling comes from — the device, or the CPU you spend getting requests to it."
      controls={
        <>
          <Segmented
            label="Submission path"
            value={path}
            onChange={setPath}
            options={[
              { value: 'pread', label: 'sync pread()', title: 'One blocking syscall per I/O; one parked thread per in-flight request' },
              { value: 'uring', label: 'io_uring', title: 'SQEs written into a shared ring; one io_uring_enter() per batch' },
            ]}
          />
          <Slider label="Queue depth" min={1} max={128} value={qd} onChange={setQd} />
          <Slider
            label="SQEs per io_uring_enter"
            min={1}
            max={32}
            value={batch}
            onChange={setBatch}
            disabled={cfg.sync}
          />
          <Slider label="CPU cores for I/O" min={1} max={8} value={cores} onChange={setCores} />
          <Check label="SQPOLL (kernel poller thread)" checked={sqpoll} onChange={setSqpoll} />
          <Button onClick={() => setRunning((r) => !r)}>{running ? 'Pause' : 'Run'}</Button>
          <Button
            onClick={() => {
              simRef.current = newSim();
              setSnap(EMPTY);
            }}
          >
            Reset
          </Button>
        </>
      }
      legend={
        <Legend
          items={[
            { label: cfg.sync ? 'Thread in pread(), pre-submit' : 'SQE written, not yet submitted', color: 'var(--viz-1)' },
            { label: 'In the device (DMA in flight)', color: 'var(--viz-2)' },
            { label: cfg.sync ? 'Returned, thread waking' : 'CQE posted, not yet reaped', color: 'var(--viz-3)' },
            { label: 'Idle slot', color: 'var(--viz-neutral)' },
          ]}
        />
      }
      stats={
        <Stats
          items={[
            { label: 'Achieved IOPS', value: fmtNum(snap.iops), hint: 'Measured over the running simulation' },
            {
              label: 'Syscalls / request',
              value: syscallsPerReq === 0 ? '0' : syscallsPerReq.toFixed(2),
              hint: 'Measured: sync pread costs exactly one; io_uring amortizes one enter over a batch',
            },
            { label: 'CPU µs / request', value: k.perReq.toFixed(2) },
            { label: 'Mean latency', value: `${snap.lat.toFixed(0)} µs` },
            { label: 'Threads parked', value: cfg.sync ? fmtNum(threads) : '1' },
            { label: 'CPU-bound ceiling', value: fmtNum(cpuCeil) },
          ]}
        />
      }
      note={
        <Note>
          {cfg.sync ? (
            <>
              <strong>One syscall and two context switches per 4 KB read.</strong> That is{' '}
              {k.perReq.toFixed(1)} µs of CPU per request, so {cores} core
              {cores > 1 ? 's' : ''} can issue at most {fmtNum(cpuCeil)} IOPS however fast the drive
              is — and keeping {qd} requests in flight costs {qd} parked threads, {qd * 8} MB of
              stack address space and a run queue full of tasks that do nothing but sleep.
              {cpuCeil < devCeil
                ? ` The drive could serve ${fmtNum(devCeil)} here; you are leaving ${Math.round((1 - cpuCeil / devCeil) * 100)}% of it idle.`
                : ' At this queue depth the drive, not the CPU, is the limit.'}
            </>
          ) : (
            <>
              <strong>
                One io_uring_enter() carries {k.effBatch} request{k.effBatch > 1 ? 's' : ''}
                {sqpoll ? ', and SQPOLL removes even that' : ''}.
              </strong>{' '}
              Requests are stores into an mmap'd SQE array plus a release-store to the tail; the
              kernel picks them up and posts CQEs the app reads with a plain load. Cost per request
              falls to {k.perReq.toFixed(2)} µs, lifting the CPU ceiling to {fmtNum(cpuCeil)} IOPS
              {cpuCeil > devCeil
                ? ` — past the drive's ${fmtNum(devCeil)}, so the device is now the bottleneck, which is where you want it.`
                : `, still under the drive's ${fmtNum(devCeil)}; raise the batch or the core count.`}
              {sqpoll ? ' SQPOLL costs a whole core of busy-polling to do it.' : ''}
            </>
          )}
        </Note>
      }
      table={
        <table className="viz-table">
          <thead>
            <tr>
              <th>Submission path</th>
              <th>CPU µs / request</th>
              <th>Syscalls / request</th>
              <th>CPU ceiling (IOPS)</th>
              <th>Device ceiling (IOPS)</th>
              <th>Achievable</th>
            </tr>
          </thead>
          <tbody>
            {tableRows.map((r) => {
              const m = ceilings(r.c);
              return (
                <tr key={r.name}>
                  <td>{r.name}</td>
                  <td>{m.k.perReq.toFixed(2)}</td>
                  <td>{m.k.syscallsPerReq === 0 ? '0' : m.k.syscallsPerReq.toFixed(2)}</td>
                  <td>{fmtNum(m.cpuCeil)}</td>
                  <td>{fmtNum(m.devCeil)}</td>
                  <td>{fmtNum(m.iops)}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      }
    >
      <div ref={ref}>
        <svg
          width={svgW}
          height={height}
          role="img"
          aria-label="Submission and completion rings filling as queue depth and batch size change"
        >
          <SlotRow
            y={rowY[0]}
            n={cfg.sync ? Math.min(qd, RING) : RING}
            filled={cfg.sync ? snap.sq + snap.dev + snap.cq : snap.sq}
            color="var(--viz-1)"
            label={cfg.sync ? 'Application threads' : 'SQ ring (32 entries)'}
            sub={cfg.sync ? 'one parked per in-flight I/O' : 'SQEs awaiting io_uring_enter()'}
            right={
              cfg.sync
                ? `${fmtNum(Math.min(qd, snap.sq + snap.dev + snap.cq))} blocked`
                : `head ${snap.sqHead} · tail ${snap.sqTail}`
            }
          />
          <SlotRow
            y={rowY[1]}
            n={DEV_SERVERS}
            filled={snap.dev - snap.queued}
            color="var(--viz-2)"
            label="NVMe device"
            sub={`${DEV_SERVERS} overlapping, ${SERVICE_US} µs each`}
            right={`${snap.dev} in flight${snap.queued ? ` · ${snap.queued} queued` : ''}`}
          />
          <SlotRow
            y={rowY[2]}
            n={cfg.sync ? Math.min(qd, RING) : RING}
            filled={snap.cq}
            color="var(--viz-3)"
            label={cfg.sync ? 'Returned pread() calls' : `CQ ring (first ${RING} of ${RING * 2})`}
            sub={cfg.sync ? 'each wakes a thread: one context switch' : 'CQEs awaiting a plain load'}
            right={cfg.sync ? `${snap.cq} waking` : `head ${snap.cqHead} · tail ${snap.cqTail}`}
          />
          <text x={labelW} y={height - 8} fill="var(--viz-ink-muted)" fontSize={10}>
            {fmtNum(snap.completed)} requests completed · CPU utilisation{' '}
            {(snap.cpu * 100).toFixed(0)}% of {k.effCores.toFixed(1)} core
            {k.effCores > 1 ? 's' : ''}
            {cfg.sync ? '' : sqpoll ? ' (one core given to the SQPOLL thread)' : ''}
          </text>
        </svg>
      </div>
    </VizPanel>
  );
}
