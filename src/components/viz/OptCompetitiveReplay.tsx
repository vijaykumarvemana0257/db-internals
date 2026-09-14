import { useMemo, useRef, useState } from 'react';
import {
  VizPanel,
  Choice,
  Slider,
  Button,
  Legend,
  Stats,
  Note,
  TooltipHost,
  useTip,
  useTicker,
  makeRng,
  useSize,
} from './Viz';

/**
 * Belady's MIN against three online policies on the same reference string.
 *
 * OPT is the offline optimum: on a miss it evicts the resident page whose *next*
 * reference is furthest in the future, which requires the whole trace up front and is
 * therefore a yardstick rather than a policy. LRU, CLOCK and ARC see only the prefix.
 * The readout is the competitive ratio misses(ALG)/misses(OPT), which Sleator & Tarjan
 * proved no deterministic online policy can push below k in the worst case.
 */

/* ------------------------------------------------------------------ traces */

type Workload = 'hot' | 'loop' | 'scan' | 'random';

const TRACE_LEN = 150;

const WORKLOADS: { value: Workload; label: string; blurb: string }[] = [
  { value: 'hot', label: 'Hot set (80/20)', blurb: '80% of references land on 6 hot pages, 20% scatter over 32 cold ones.' },
  { value: 'loop', label: 'Looping scan (9 pages)', blurb: 'A cyclic scan over 9 pages — the reference string LRU was designed to lose on.' },
  { value: 'scan', label: 'Hot set + one scan', blurb: 'Four hot pages, interrupted by a single sequential pass over 30 pages that will never be read again.' },
  { value: 'random', label: 'Uniform random (24)', blurb: 'No locality at all: every page equally likely.' },
];

function buildTrace(kind: Workload): number[] {
  const rng = makeRng(1970 + kind.length * 7); // Mattson et al., 1970
  const t: number[] = [];
  if (kind === 'loop') {
    for (let i = 0; i < TRACE_LEN; i++) t.push(i % 9);
    return t;
  }
  if (kind === 'random') {
    for (let i = 0; i < TRACE_LEN; i++) t.push(Math.floor(rng() * 24));
    return t;
  }
  if (kind === 'scan') {
    const scanStart = 40;
    let scanPage = 20;
    for (let i = 0; i < TRACE_LEN; i++) {
      if (i >= scanStart && i < scanStart + 60 && i % 2 === 0) t.push(scanPage++);
      else t.push(Math.floor(rng() * 4));
    }
    return t;
  }
  for (let i = 0; i < TRACE_LEN; i++) {
    t.push(rng() < 0.8 ? Math.floor(rng() * 6) : 8 + Math.floor(rng() * 32));
  }
  return t;
}

/* -------------------------------------------------------------- simulation */

type Slot = { page: number; tag?: string; ghost?: boolean };
type Snap = {
  hit: boolean;
  slots: Slot[];
  ghosts: Slot[];
  hand: number; // CLOCK only; -1 elsewhere
  victim: number | null;
  misses: number;
  why: string;
};

const EMPTY: Snap = { hit: false, slots: [], ghosts: [], hand: -1, victim: null, misses: 0, why: '' };

/** next[i] = index of the next reference to trace[i] after i, or Infinity. */
function nextUseIndex(trace: number[]): number[] {
  const next = new Array<number>(trace.length).fill(Infinity);
  const seen = new Map<number, number>();
  for (let i = trace.length - 1; i >= 0; i--) {
    const p = trace[i];
    next[i] = seen.has(p) ? (seen.get(p) as number) : Infinity;
    seen.set(p, i);
  }
  return next;
}

function simOpt(trace: number[], k: number): Snap[] {
  const next = nextUseIndex(trace);
  const res = new Map<number, number>(); // page -> index of its next reference
  const out: Snap[] = [];
  let misses = 0;
  for (let i = 0; i < trace.length; i++) {
    const p = trace[i];
    const hit = res.has(p);
    let victim: number | null = null;
    let why = '';
    if (hit) {
      why = 'resident — no eviction needed';
    } else {
      misses++;
      if (res.size >= k) {
        let worst = -1;
        let worstAt = -1;
        for (const [q, at] of res) if (at > worstAt) ((worstAt = at), (worst = q));
        res.delete(worst);
        victim = worst;
        why =
          worstAt === Infinity
            ? `evicts page ${worst}: never referenced again`
            : `evicts page ${worst}: next needed at step ${worstAt + 1}, further out than any other resident page`;
      } else {
        why = 'free frame available';
      }
    }
    res.set(p, next[i]);
    const slots = [...res.entries()]
      .sort((a, b) => a[1] - b[1])
      .map(([q, at]) => ({ page: q, tag: at === Infinity ? '∞' : `→${at + 1}` }));
    out.push({ hit, slots, ghosts: [], hand: -1, victim, misses, why });
  }
  return out;
}

function simLru(trace: number[], k: number): Snap[] {
  const order: number[] = []; // index 0 = MRU
  const out: Snap[] = [];
  let misses = 0;
  for (let i = 0; i < trace.length; i++) {
    const p = trace[i];
    const at = order.indexOf(p);
    const hit = at >= 0;
    let victim: number | null = null;
    let why = '';
    if (hit) {
      order.splice(at, 1);
      why = `hit at stack depth ${at + 1}; moves back to MRU`;
    } else {
      misses++;
      if (order.length >= k) {
        victim = order.pop() as number;
        why = `evicts page ${victim}: least recently used`;
      } else why = 'free frame available';
    }
    order.unshift(p);
    out.push({
      hit,
      slots: order.map((q, j) => ({ page: q, tag: j === 0 ? 'MRU' : j === order.length - 1 ? 'LRU' : undefined })),
      ghosts: [],
      hand: -1,
      victim,
      misses,
      why,
    });
  }
  return out;
}

function simClock(trace: number[], k: number): Snap[] {
  const page = new Array<number>(k).fill(-1);
  const ref = new Array<number>(k).fill(0);
  let hand = 0;
  const out: Snap[] = [];
  let misses = 0;
  for (let i = 0; i < trace.length; i++) {
    const p = trace[i];
    const at = page.indexOf(p);
    const hit = at >= 0;
    let victim: number | null = null;
    let why = '';
    if (hit) {
      ref[at] = 1;
      why = `hit in frame ${at}; reference bit set back to 1`;
    } else {
      misses++;
      const free = page.indexOf(-1);
      if (free >= 0) {
        page[free] = p;
        ref[free] = 1;
        why = 'free frame available';
      } else {
        let swept = 0;
        while (ref[hand] === 1) {
          ref[hand] = 0;
          hand = (hand + 1) % k;
          swept++;
        }
        victim = page[hand];
        page[hand] = p;
        ref[hand] = 1;
        hand = (hand + 1) % k;
        why = `sweep cleared ${swept} reference bit${swept === 1 ? '' : 's'}, then evicted page ${victim}`;
      }
    }
    out.push({
      hit,
      slots: page.filter((q) => q >= 0).map((q, j) => ({ page: q, tag: ref[j] ? 'ref=1' : 'ref=0' })),
      ghosts: [],
      hand,
      victim,
      misses,
      why,
    });
  }
  return out;
}

/** ARC as published by Megiddo & Modha (FAST '03): T1/T2 resident, B1/B2 ghosts, adaptive p. */
function simArc(trace: number[], k: number): Snap[] {
  const T1: number[] = []; // last element = MRU
  const T2: number[] = [];
  const B1: number[] = [];
  const B2: number[] = [];
  let p = 0;
  const out: Snap[] = [];
  let misses = 0;
  const drop = (l: number[], x: number) => {
    const i = l.indexOf(x);
    if (i >= 0) l.splice(i, 1);
  };
  const replace = (inB2: boolean): number | null => {
    if (T1.length > 0 && (T1.length > p || (inB2 && T1.length === p))) {
      const y = T1.shift() as number;
      B1.push(y);
      return y;
    }
    if (T2.length > 0) {
      const y = T2.shift() as number;
      B2.push(y);
      return y;
    }
    return null;
  };

  for (let i = 0; i < trace.length; i++) {
    const x = trace[i];
    let hit = false;
    let victim: number | null = null;
    let why = '';
    if (T1.includes(x)) {
      drop(T1, x);
      T2.push(x);
      hit = true;
      why = 'second reference: promoted from T1 (seen once) to T2 (seen twice)';
    } else if (T2.includes(x)) {
      drop(T2, x);
      T2.push(x);
      hit = true;
      why = 'hit in T2, the frequently-used list';
    } else if (B1.includes(x)) {
      misses++;
      const d = B1.length >= B2.length ? 1 : B2.length / B1.length;
      p = Math.min(k, p + d);
      victim = replace(false);
      drop(B1, x);
      T2.push(x);
      why = `ghost hit in B1 — recency is paying off, target |T1| grows to ${p.toFixed(1)}`;
    } else if (B2.includes(x)) {
      misses++;
      const d = B2.length >= B1.length ? 1 : B1.length / B2.length;
      p = Math.max(0, p - d);
      victim = replace(true);
      drop(B2, x);
      T2.push(x);
      why = `ghost hit in B2 — frequency is paying off, target |T1| shrinks to ${p.toFixed(1)}`;
    } else {
      misses++;
      if (T1.length + B1.length === k) {
        if (T1.length < k) {
          B1.shift();
          victim = replace(false);
        } else {
          victim = T1.shift() as number;
        }
      } else if (T1.length + T2.length + B1.length + B2.length >= k) {
        if (T1.length + T2.length + B1.length + B2.length === 2 * k) B2.shift();
        victim = replace(false);
      }
      T1.push(x);
      why = victim === null ? 'free frame available' : `cold miss: enters T1, evicts page ${victim} to the ghost list`;
    }
    out.push({
      hit,
      slots: [
        ...T1.map((q) => ({ page: q, tag: 'T1' })),
        ...T2.map((q) => ({ page: q, tag: 'T2' })),
      ],
      ghosts: [
        ...B1.slice(-4).map((q) => ({ page: q, tag: 'B1', ghost: true })),
        ...B2.slice(-4).map((q) => ({ page: q, tag: 'B2', ghost: true })),
      ],
      hand: -1,
      victim,
      misses,
      why,
    });
  }
  return out;
}

/* ------------------------------------------------------------- the component */

const POLICIES = [
  { id: 'opt', label: 'OPT (Belady MIN)', color: 'var(--viz-1)', sub: 'offline — sees the whole trace' },
  { id: 'lru', label: 'LRU', color: 'var(--viz-2)', sub: 'exact recency order' },
  { id: 'clock', label: 'CLOCK', color: 'var(--viz-3)', sub: 'second chance, one reference bit' },
  { id: 'arc', label: 'ARC', color: 'var(--viz-4)', sub: 'T1/T2 + ghosts, adaptive p' },
] as const;

export default function OptCompetitiveReplay() {
  const [workload, setWorkload] = useState<Workload>('loop');
  const [k, setK] = useState(6);
  const [step, setStep] = useState(0);
  const [running, setRunning] = useState(false);
  const acc = useRef(0);
  const [ref, width] = useSize(760);
  const tip = useTip();

  const trace = useMemo(() => buildTrace(workload), [workload]);
  const sims = useMemo(
    () => ({
      opt: simOpt(trace, k),
      lru: simLru(trace, k),
      clock: simClock(trace, k),
      arc: simArc(trace, k),
    }),
    [trace, k],
  );

  const last = trace.length - 1;
  const i = Math.min(step, last);

  useTicker((dt) => {
    acc.current += dt;
    if (acc.current < 110) return;
    acc.current = 0;
    setStep((s) => {
      if (s >= last) {
        setRunning(false);
        return last;
      }
      return s + 1;
    });
  }, running);

  const snaps = POLICIES.map((pol) => sims[pol.id][i] ?? EMPTY);
  const optMiss = sims.opt[i].misses;
  const ratio = (m: number) => (optMiss === 0 ? 1 : m / optMiss);

  /* geometry */
  const labelW = 150;
  const cell = 22;
  const rowH = 58;
  const chipW = 34;
  const chipGap = 6;
  const maxSlots = Math.max(...snaps.map((s) => s.slots.length + (s.ghosts.length ? s.ghosts.length + 1 : 0)));
  const traceWindow = 34;
  const wStart = Math.max(0, Math.min(i - Math.floor(traceWindow / 2), trace.length - traceWindow));
  const svgW = Math.max(
    width,
    labelW + Math.max(traceWindow * cell, maxSlots * (chipW + chipGap)) + 92,
  );
  const ribbonY = 16;
  const rowsY = ribbonY + 52;
  const height = rowsY + POLICIES.length * rowH + 12;

  const cur = trace[i];

  return (
    <VizPanel
      title="OPT versus the policies that have to guess"
      subtitle="Belady's MIN evicts the page whose next reference is furthest away — it needs the future. Step the same reference string through it, LRU, CLOCK and ARC and watch the competitive ratio."
      controls={
        <>
          <Choice
            label="Reference string"
            value={workload}
            onChange={(v) => {
              setWorkload(v);
              setStep(0);
              setRunning(false);
            }}
            options={WORKLOADS.map((w) => ({ value: w.value, label: w.label }))}
          />
          <Slider
            label="Frames (k)"
            min={3}
            max={10}
            value={k}
            onChange={(n) => {
              setK(n);
              setRunning(false);
            }}
          />
          <Slider label="Step" min={0} max={last} value={i} onChange={(n) => { setStep(n); setRunning(false); }} format={(n) => `${n + 1} / ${trace.length}`} />
          <Button onClick={() => setStep((s) => Math.min(s + 1, last))} disabled={i >= last}>
            Step
          </Button>
          <Button primary onClick={() => setRunning((r) => !r)} disabled={i >= last && !running}>
            {running ? 'Pause' : 'Run'}
          </Button>
          <Button
            onClick={() => {
              setStep(0);
              setRunning(false);
            }}
          >
            Reset
          </Button>
        </>
      }
      legend={
        <Legend
          items={[
            ...POLICIES.map((p) => ({ label: p.label, color: p.color })),
            { label: 'hit this step', color: 'var(--viz-good)' },
            { label: 'miss this step', color: 'var(--viz-critical)' },
            { label: 'ARC ghost (B1/B2, 4 most recent each) — metadata only, holds no frame', color: 'var(--viz-ink-muted)', shape: 'line' as const },
          ]}
        />
      }
      stats={
        <Stats
          items={[
            { label: 'References replayed', value: i + 1 },
            { label: 'OPT misses', value: optMiss, hint: 'The offline optimum — no policy can do better on this string' },
            { label: 'LRU / OPT', value: `${ratio(sims.lru[i].misses).toFixed(2)}×`, hint: `Worst case is bounded by k = ${k}` },
            { label: 'CLOCK / OPT', value: `${ratio(sims.clock[i].misses).toFixed(2)}×` },
            { label: 'ARC / OPT', value: `${ratio(sims.arc[i].misses).toFixed(2)}×` },
            { label: 'Deterministic bound', value: `${k}×`, hint: 'No deterministic online policy is better than k-competitive (Sleator & Tarjan, 1985)' },
          ]}
        />
      }
      note={
        <Note>
          <strong>
            Step {i + 1}: page {cur} — {snaps.map((s, n) => `${POLICIES[n].label.split(' ')[0]} ${s.hit ? 'hit' : 'miss'}`).join(', ')}.
          </strong>{' '}
          OPT {snaps[0].why}. LRU {snaps[1].why}. {WORKLOADS.find((w) => w.value === workload)?.blurb}
        </Note>
      }
      table={
        <table className="viz-table">
          <thead>
            <tr>
              <th>Policy</th>
              <th>Misses</th>
              <th>Hit rate</th>
              <th>Misses / OPT</th>
              <th>Worst-case bound</th>
            </tr>
          </thead>
          <tbody>
            {POLICIES.map((pol) => {
              const s = sims[pol.id][i];
              return (
                <tr key={pol.id}>
                  <td>{pol.label}</td>
                  <td>{s.misses}</td>
                  <td>{(((i + 1 - s.misses) / (i + 1)) * 100).toFixed(1)}%</td>
                  <td>{pol.id === 'opt' ? '1.00× (by definition)' : `${ratio(s.misses).toFixed(2)}×`}</td>
                  <td>{pol.id === 'opt' ? '—' : `${k}× (k-competitive)`}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      }
    >
      <div ref={ref}>
        <TooltipHost>
          <svg width={svgW} height={height} role="img" aria-label="Frame contents of OPT, LRU, CLOCK and ARC replaying one reference string">
            {/* reference string ribbon */}
            <text x={0} y={ribbonY + 14} fill="var(--viz-ink)" fontWeight={600}>
              Reference string
            </text>
            <text x={0} y={ribbonY + 30} fill="var(--viz-ink-muted)">
              past · now · future
            </text>
            {Array.from({ length: traceWindow }, (_, j) => {
              const idx = wStart + j;
              if (idx >= trace.length) return null;
              const isNow = idx === i;
              const isPast = idx < i;
              return (
                <g key={idx} {...tip(<>step {idx + 1}: page {trace[idx]}</>)}>
                  <rect
                    x={labelW + j * cell}
                    y={ribbonY}
                    width={cell - 2}
                    height={22}
                    rx={3}
                    fill={isNow ? 'var(--viz-1)' : 'var(--viz-plane)'}
                    stroke={isNow ? 'var(--viz-1)' : 'var(--viz-border)'}
                    opacity={isPast ? 0.45 : 1}
                  />
                  <text
                    x={labelW + j * cell + (cell - 2) / 2}
                    y={ribbonY + 15}
                    textAnchor="middle"
                    fill={isNow ? 'var(--viz-surface)' : 'var(--viz-ink-2)'}
                    opacity={isPast ? 0.7 : 1}
                  >
                    {trace[idx]}
                  </text>
                </g>
              );
            })}

            {/* policy rows */}
            {POLICIES.map((pol, r) => {
              const s = snaps[r];
              const y = rowsY + r * rowH;
              const x = labelW;
              const ghostX = labelW + s.slots.length * (chipW + chipGap) + 4;
              return (
                <g key={pol.id}>
                  <rect x={labelW - 8} y={y - 6} width={svgW - labelW} height={rowH - 10} rx={8} fill="var(--viz-plane)" stroke="var(--viz-border)" />
                  <rect x={0} y={y - 2} width={4} height={rowH - 20} rx={2} fill={pol.color} />
                  <text x={10} y={y + 12} fill="var(--viz-ink)" fontWeight={600}>
                    {pol.label}
                  </text>
                  <text x={10} y={y + 27} fill="var(--viz-ink-muted)">
                    {pol.sub}
                  </text>
                  <text x={10} y={y + 41} fill="var(--viz-ink-2)">
                    {s.misses} misses
                  </text>
                  {s.slots.map((sl, j) => {
                    const isCur = sl.page === cur;
                    const cx = x + j * (chipW + chipGap);
                    const fill = isCur ? (s.hit ? 'var(--viz-good)' : 'var(--viz-critical)') : 'var(--viz-surface)';
                    return (
                      <g
                        key={`${pol.id}-${j}-${sl.page}`}
                        {...tip(
                          <>
                            <strong>page {sl.page}</strong>
                            <br />
                            {pol.label}
                            {sl.tag ? ` · ${sl.tag}` : ''}
                            {isCur ? (s.hit ? ' · referenced now, hit' : ' · just faulted in') : ''}
                          </>,
                        )}
                        style={{ cursor: 'help' }}
                      >
                        <rect x={cx} y={y} width={chipW} height={22} rx={5} fill={fill} stroke={isCur ? fill : pol.color} strokeWidth={isCur ? 1 : 1.5} />
                        <text x={cx + chipW / 2} y={y + 15} textAnchor="middle" fill={isCur ? 'var(--viz-surface)' : 'var(--viz-ink)'}>
                          {sl.page}
                        </text>
                        {sl.tag ? (
                          <text x={cx + chipW / 2} y={y + 35} textAnchor="middle" fill="var(--viz-ink-muted)">
                            {sl.tag}
                          </text>
                        ) : null}
                        {pol.id === 'clock' && j === s.hand ? (
                          <text x={cx + chipW / 2} y={y + 47} textAnchor="middle" fill="var(--viz-ink-2)">
                            ▲hand
                          </text>
                        ) : null}
                      </g>
                    );
                  })}
                  {s.ghosts.length ? (
                    <>
                      <line
                        x1={ghostX}
                        x2={ghostX}
                        y1={y - 2}
                        y2={y + 26}
                        className="viz-axis-line"
                        strokeDasharray="3 3"
                      />
                      {s.ghosts.map((g, j) => {
                        const cx = ghostX + 10 + j * (chipW + chipGap);
                        return (
                          <g
                            key={`gh-${j}-${g.page}`}
                            {...tip(
                              <>
                                <strong>page {g.page}</strong>
                                <br />
                                ghost in {g.tag}: ARC remembers it was evicted, but holds no frame for it. A reference here is still a miss — it just retunes p.
                              </>,
                            )}
                            style={{ cursor: 'help' }}
                          >
                            <rect x={cx} y={y} width={chipW} height={22} rx={5} fill="none" stroke="var(--viz-ink-muted)" strokeDasharray="3 2" />
                            <text x={cx + chipW / 2} y={y + 15} textAnchor="middle" fill="var(--viz-ink-muted)">
                              {g.page}
                            </text>
                            <text x={cx + chipW / 2} y={y + 35} textAnchor="middle" fill="var(--viz-ink-muted)">
                              {g.tag}
                            </text>
                          </g>
                        );
                      })}
                    </>
                  ) : null}
                  {s.victim !== null ? (
                    <text x={svgW - 10} y={y + 15} textAnchor="end" fill="var(--viz-critical)">
                      evicted {s.victim}
                    </text>
                  ) : null}
                </g>
              );
            })}
          </svg>
        </TooltipHost>
      </div>
    </VizPanel>
  );
}
