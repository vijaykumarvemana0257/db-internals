import { useMemo, useState } from 'react';
import {
  VizPanel,
  Slider,
  Segmented,
  Legend,
  Stats,
  Note,
  TooltipHost,
  useTip,
  makeRng,
  fmtNum,
  fmtTime,
  useSize,
} from './Viz';

/**
 * What a prefetch queue actually buys, and what it costs when the guess is wrong.
 *
 * The model is a single consumer thread (the scan) in front of a device that
 * overlaps LANES reads at a time, plus a small pool of frames the prefetched
 * pages have to live in until someone consumes them. Three things fall out of
 * that and nothing else is hand-waved:
 *
 *  - Little's law: in-flight = latency / per-page CPU. Below that depth the
 *    consumer stalls on the device; above it the extra requests only queue.
 *  - Guessing: kernel readahead extrapolates *physical* blocks from a detected
 *    ascending run, so a strided or random demand sequence makes it read pages
 *    nobody wants (or, once detection fails, stop prefetching entirely). An
 *    engine issuing its own prefetch reads the block list it already has, so it
 *    is never wrong about *which* page — only about *how many* to have in flight.
 *  - Pollution: a prefetched page occupies a frame from arrival until use. Push
 *    the depth past the frames available and LRU evicts the page nearest to being
 *    needed, and it gets read a second time.
 */

type Pattern = 'seq' | 'strided' | 'random';
type Issuer = 'kernel' | 'engine';
type Outcome = 'used' | 'wasted' | 'evicted';

const N = 40; // pages the scan actually consumes
const LANES = 8; // reads this device genuinely overlaps
const SERVICE_US = 90; // one 8 KB read at queue depth 1
const POOL = 24; // frames this scan is allowed to hold

type Read = {
  id: number;
  block: number;
  lane: number;
  start: number;
  done: number;
  kind: 'demand' | 'prefetch';
  outcome: Outcome;
};

type Seg = { start: number; end: number; kind: 'cpu' | 'stall' };

type Cfg = { depth: number; pattern: Pattern; issuer: Issuer; cpuUs: number };

function demandSequence(pattern: Pattern): number[] {
  if (pattern === 'seq') return Array.from({ length: N }, (_, i) => i);
  if (pattern === 'strided') return Array.from({ length: N }, (_, i) => i * 3);
  const rng = makeRng(7717);
  const seen = new Set<number>();
  const out: number[] = [];
  while (out.length < N) {
    const b = Math.floor(rng() * 400);
    if (!seen.has(b)) {
      seen.add(b);
      out.push(b);
    }
  }
  return out;
}

type Result = {
  reads: Read[];
  cpuSegs: Seg[];
  elapsed: number;
  deviceTail: number;
  used: number;
  wasted: number;
  evicted: number;
  stalls: number;
  hits: number;
  stallUs: number;
  maxInflight: number;
};

function simulate({ depth, pattern, issuer, cpuUs }: Cfg): Result {
  const demand = demandSequence(pattern);
  const demandSet = new Set(demand);
  const consumed = new Set<number>();
  const rng = makeRng(20260913);

  const lanes = new Array(LANES).fill(0);
  const reads: Read[] = [];
  const inflight = new Map<number, number>(); // block -> read id
  const pool: { block: number; read: number }[] = []; // LRU order, index 0 is coldest
  const resident = new Map<number, number>(); // block -> read id
  const pending = new Set<number>(); // prefetched, not yet consumed: each one holds a frame
  const cpuSegs: Seg[] = [];

  let t = 0;
  let stallUs = 0;
  let stalls = 0;
  let hits = 0;
  let maxInflight = 0;

  const evictDown = () => {
    while (pool.length > POOL) {
      const e = pool.shift()!;
      resident.delete(e.block);
      pending.delete(e.block);
      const r = reads[e.read];
      // thrown out before anyone used it: a refetch if the scan still wants it,
      // otherwise it was simply a page nobody ever asked for
      if (r && r.outcome === 'wasted' && demandSet.has(e.block) && !consumed.has(e.block)) {
        r.outcome = 'evicted';
      }
    }
  };

  const drain = (upTo: number) => {
    for (const [block, id] of Array.from(inflight)) {
      if (reads[id].done <= upTo) {
        inflight.delete(block);
        if (!resident.has(block)) {
          pool.push({ block, read: id });
          resident.set(block, id);
          evictDown();
        }
      }
    }
  };

  /** A scan keeps the page it just read cold (a ring buffer, usage_count 0): what
   *  competes for frames is the prefetch window ahead of it, not the trail behind. */
  const cool = (block: number) => {
    const idx = pool.findIndex((p) => p.block === block);
    if (idx >= 0) pool.unshift(pool.splice(idx, 1)[0]);
  };

  const issue = (block: number, kind: 'demand' | 'prefetch', now: number) => {
    // fill lanes from the top: the lowest-numbered lane that is free right now,
    // and only if every lane is busy does the request queue behind the earliest
    // one to finish. At depth 0 that means one lane works and the rest are idle.
    let best = 0;
    for (let i = 0; i < LANES; i++) {
      if (lanes[i] <= now) {
        best = i;
        break;
      }
      if (lanes[i] < lanes[best]) best = i;
    }
    const start = Math.max(now, lanes[best]);
    const done = start + SERVICE_US * (0.85 + rng() * 0.3);
    lanes[best] = done;
    const id = reads.length;
    reads.push({ id, block, lane: best, start, done, kind, outcome: 'wasted' });
    inflight.set(block, id);
    if (kind === 'prefetch') pending.add(block);
    if (inflight.size > maxInflight) maxInflight = inflight.size;
    return id;
  };

  const inflightPrefetches = () => {
    let n = 0;
    for (const id of inflight.values()) if (reads[id].kind === 'prefetch') n++;
    return n;
  };

  // kernel readahead detector state
  let raNext = 0;
  let raActive = false;
  let last = -1;

  const issuePrefetch = (i: number, b: number) => {
    if (depth <= 0) return;
    if (issuer === 'engine') {
      // the engine already holds the block list (a bitmap heap scan, a read stream),
      // and bounds the window by frames it is willing to hold, not just by I/O in flight
      for (let j = i + 1; j < demand.length && pending.size < depth; j++) {
        const nb = demand[j];
        if (resident.has(nb) || inflight.has(nb)) continue;
        issue(nb, 'prefetch', t);
      }
      return;
    }
    // the kernel only sees block numbers arriving and has to guess the rest
    if (last >= 0) raActive = b > last && b - last <= depth;
    if (b + 1 > raNext) raNext = b + 1;
    if (!raActive) return;
    while (inflightPrefetches() < depth && raNext <= b + depth) {
      if (!resident.has(raNext) && !inflight.has(raNext)) issue(raNext, 'prefetch', t);
      raNext++;
    }
  };

  for (let i = 0; i < demand.length; i++) {
    const b = demand[i];
    drain(t);
    issuePrefetch(i, b);

    let ready = t;
    if (resident.has(b)) {
      reads[resident.get(b)!].outcome = 'used';
      hits++;
    } else if (inflight.has(b)) {
      const id = inflight.get(b)!;
      ready = Math.max(t, reads[id].done);
      reads[id].outcome = 'used';
      stalls++;
      drain(ready);
    } else {
      const id = issue(b, 'demand', t);
      ready = reads[id].done;
      reads[id].outcome = 'used';
      stalls++;
      drain(ready);
    }
    consumed.add(b);
    pending.delete(b);
    cool(b);

    if (ready > t) {
      cpuSegs.push({ start: t, end: ready, kind: 'stall' });
      stallUs += ready - t;
      t = ready;
    }
    cpuSegs.push({ start: t, end: t + cpuUs, kind: 'cpu' });
    t += cpuUs;
    last = b;
  }

  const deviceTail = reads.reduce((m, r) => Math.max(m, r.done), t);
  const used = reads.filter((r) => r.outcome === 'used').length;
  const wasted = reads.filter((r) => r.outcome === 'wasted').length;
  const evicted = reads.filter((r) => r.outcome === 'evicted').length;

  return { reads, cpuSegs, elapsed: t, deviceTail, used, wasted, evicted, stalls, hits, stallUs, maxInflight };
}

const OUTCOME_FILL: Record<Outcome, string> = {
  used: 'var(--viz-1)',
  wasted: 'var(--viz-stale)',
  evicted: 'var(--viz-8)',
};

const OUTCOME_LABEL: Record<Outcome, string> = {
  used: 'consumed by the scan',
  wasted: 'never demanded',
  evicted: 'evicted before use',
};

const SWEEP = [0, 1, 2, 4, 8, 12, 16, 24, 32];

/** Lives inside <TooltipHost> so useTip() sees the provider. */
function ScanFigure({ r, depth, avail }: { r: Result; depth: number; avail: number }) {
  const tip = useTip();
  const labelW = 74;
  const plotW = Math.max(240, Math.min(avail, 1040) - labelW - 16);
  const rowH = 15;
  const gap = 3;
  const topH = rowH + 10;
  const height = topH + LANES * (rowH + gap) + 26;
  const tmax = Math.max(r.deviceTail, 1);
  const x = (us: number) => labelW + (us / tmax) * plotW;
  const w = (a: number, b: number) => Math.max(1, ((b - a) / tmax) * plotW);

  const raw = tmax / 5;
  const mag = 10 ** Math.floor(Math.log10(raw));
  const step = [1, 2, 5, 10].map((m) => m * mag).find((s) => s >= raw) ?? mag * 10;
  const ticks: number[] = [];
  for (let v = 0; v <= tmax; v += step) ticks.push(v);

  return (
    <svg
      width={Math.max(320, labelW + plotW + 16)}
      height={height}
      role="img"
      aria-label={`Timeline of a ${N} page scan at prefetch depth ${depth}`}
    >
      {ticks.map((v) => (
        <line key={v} x1={x(v)} x2={x(v)} y1={0} y2={height - 20} stroke="var(--viz-grid)" strokeWidth={1} />
      ))}

      <text x={labelW - 8} y={12} textAnchor="end" fill="var(--viz-ink)" fontSize={11}>
        scan CPU
      </text>
      {r.cpuSegs.map((s, i) => (
        <rect
          key={i}
          x={x(s.start)}
          y={0}
          width={w(s.start, s.end)}
          height={rowH}
          rx={2}
          fill={s.kind === 'cpu' ? 'var(--viz-6)' : 'var(--viz-critical)'}
          {...tip(
            s.kind === 'cpu' ? (
              <>processing a page — {Math.round(s.end - s.start)} µs</>
            ) : (
              <>
                <strong>stalled</strong> waiting for a page — {Math.round(s.end - s.start)} µs
              </>
            ),
          )}
        />
      ))}

      {Array.from({ length: LANES }, (_, lane) => {
        const y = topH + lane * (rowH + gap);
        return (
          <g key={lane}>
            <text x={labelW - 8} y={y + 11} textAnchor="end" fill="var(--viz-ink-2)" fontSize={11}>
              lane {lane}
            </text>
            <line
              x1={labelW}
              x2={labelW + plotW}
              y1={y + rowH / 2}
              y2={y + rowH / 2}
              stroke="var(--viz-axis)"
              strokeWidth={0.75}
            />
            {r.reads
              .filter((rd) => rd.lane === lane)
              .map((rd) => (
                <rect
                  key={rd.id}
                  x={x(rd.start)}
                  y={y}
                  width={w(rd.start, rd.done)}
                  height={rowH}
                  rx={2}
                  fill={OUTCOME_FILL[rd.outcome]}
                  stroke={rd.kind === 'demand' ? 'var(--viz-ink)' : 'none'}
                  strokeWidth={rd.kind === 'demand' ? 1.25 : 0}
                  {...tip(
                    <>
                      block {rd.block} · {rd.kind === 'demand' ? 'demand read (scan is waiting)' : 'prefetch'}
                      <br />
                      {Math.round(rd.start)}–{Math.round(rd.done)} µs on lane {rd.lane}
                      <br />
                      {OUTCOME_LABEL[rd.outcome]}
                    </>,
                  )}
                />
              ))}
          </g>
        );
      })}

      <line
        x1={x(r.elapsed)}
        x2={x(r.elapsed)}
        y1={0}
        y2={height - 20}
        stroke="var(--viz-ink)"
        strokeWidth={1.25}
        strokeDasharray="3 3"
      />

      {ticks.map((v) => (
        <text
          key={`t${v}`}
          x={x(v)}
          y={height - 6}
          fill="var(--viz-ink-muted)"
          fontSize={10}
          textAnchor="middle"
          style={{ fontVariantNumeric: 'tabular-nums' }}
        >
          {fmtNum(v)} µs
        </text>
      ))}
    </svg>
  );
}


export default function PrefetchDepthScanTimeline() {
  const [depth, setDepth] = useState(0);
  const [pattern, setPattern] = useState<Pattern>('seq');
  const [issuer, setIssuer] = useState<Issuer>('kernel');
  const [cpuUs, setCpuUs] = useState(25);
  const [ref, width] = useSize(760);

  const cfg: Cfg = { depth, pattern, issuer, cpuUs };
  const r = useMemo(() => simulate(cfg), [depth, pattern, issuer, cpuUs]);
  const base = useMemo(() => simulate({ ...cfg, depth: 0 }), [pattern, issuer, cpuUs]);

  const amp = r.reads.length / N;
  const speedup = base.elapsed / r.elapsed;
  const needed = Math.ceil(SERVICE_US / cpuUs);
  const cpuUtil = (N * cpuUs) / r.elapsed;

  return (
    <VizPanel
      title="A 40-page scan: prefetch depth against the device it is running on"
      subtitle={`Each read costs ${SERVICE_US} µs and the device overlaps ${LANES} of them; the scan holds at most ${POOL} frames. Raise the depth and watch the stalls close up — then watch what the wrong guess costs.`}
      controls={
        <>
          <Slider
            label="Prefetch depth"
            min={0}
            max={32}
            value={depth}
            onChange={setDepth}
            format={(n) => (n === 0 ? 'off (demand only)' : `${n} in flight`)}
          />
          <Segmented
            label="Demand pattern"
            value={pattern}
            onChange={setPattern}
            options={[
              { value: 'seq', label: 'Sequential', title: 'Blocks 0,1,2,… — a heap scan' },
              { value: 'strided', label: 'Strided (1 in 3)', title: 'Blocks 0,3,6,… — a correlated index scan or a sparse bitmap' },
              { value: 'random', label: 'Random', title: '40 scattered blocks — an uncorrelated index scan' },
            ]}
          />
          <Segmented
            label="Who issues it"
            value={issuer}
            onChange={setIssuer}
            options={[
              { value: 'kernel', label: 'OS readahead', title: 'Guesses physical successors once it detects an ascending run' },
              { value: 'engine', label: 'Engine prefetch', title: 'Reads the block list it already computed — never guesses which page' },
            ]}
          />
          <Slider
            label="CPU per page"
            min={5}
            max={80}
            step={5}
            value={cpuUs}
            onChange={setCpuUs}
            format={(n) => `${n} µs`}
          />
        </>
      }
      legend={
        <Legend
          items={[
            { label: 'Read consumed by the scan', color: 'var(--viz-1)' },
            { label: 'Read never demanded (wasted bandwidth)', color: 'var(--viz-stale)' },
            { label: 'Read evicted before use (refetch)', color: 'var(--viz-8)' },
            { label: 'CPU processing a page', color: 'var(--viz-6)' },
            { label: 'CPU stalled on a page read', color: 'var(--viz-critical)' },
          ]}
        />
      }
      stats={
        <Stats
          items={[
            { label: 'Elapsed', value: fmtTime(r.elapsed * 1000), hint: 'Time until the scan consumes its 40th page' },
            { label: 'vs. no prefetch', value: `${speedup.toFixed(2)}×`, hint: `Demand-only baseline: ${Math.round(base.elapsed)} µs` },
            { label: 'CPU utilisation', value: `${Math.round(cpuUtil * 100)}%`, hint: 'The scan is only doing work when this is high' },
            { label: 'Stalls on I/O', value: `${r.stalls} / ${N}`, hint: `${Math.round(r.stallUs)} µs waiting` },
            { label: 'Device reads', value: `${r.reads.length} for ${N} pages`, hint: 'Physical reads issued' },
            { label: 'Read amplification', value: `${amp.toFixed(2)}×` },
            { label: 'Wasted / refetched', value: `${r.wasted} / ${r.evicted}`, hint: 'Never demanded / evicted before use' },
            { label: 'Depth Little’s law wants', value: `${needed}`, hint: `${SERVICE_US} µs latency ÷ ${cpuUs} µs CPU per page` },
          ]}
        />
      }
      note={
        <Note>
          {depth === 0 ? (
            <>
              <strong>Queue depth 1: the scan and the device take turns.</strong> Every page costs{' '}
              {SERVICE_US} µs of latency and then {cpuUs} µs of work, so {LANES - 1} of the device&apos;s{' '}
              {LANES} lanes sit idle for the whole scan and CPU utilisation is{' '}
              {Math.round(cpuUtil * 100)}%. Nothing about the hardware is slow here; nothing is being
              asked of it.
            </>
          ) : r.hits === 0 && r.reads.length === N ? (
            <>
              <strong>The readahead detector never fires.</strong> It only extrapolates after it sees an
              ascending run of block numbers, and this demand order never gives it one, so depth {depth}{' '}
              buys exactly nothing — all {N} pages still stall. This is the case the engine has to cover
              itself: it holds the block list, the kernel only sees the misses.
            </>
          ) : r.evicted > 0 ? (
            <>
              <strong>The window is wider than the frames.</strong> Depth {depth} against {POOL}{' '}
              available frames means {r.evicted} page{r.evicted > 1 ? 's were' : ' was'} read, evicted
              before the scan reached {r.evicted > 1 ? 'them' : 'it'}, and read again — {r.reads.length}{' '}
              device reads to deliver {N} pages
              {r.wasted > 0 ? `, ${r.wasted} of them for blocks nobody ever asks for` : ''}. Prefetching
              past the frames you can hold is worse than not prefetching at all.
            </>
          ) : r.wasted > 0 ? (
            <>
              <strong>Right about direction, wrong about density.</strong> Readahead extrapolates
              physical successors, so it reads {r.reads.length} blocks to deliver {N} — {amp.toFixed(1)}×
              the bytes, {r.wasted} of them pages this scan never asks for. The stalls do close up
              ({speedup.toFixed(2)}× faster than demand-only); you are paying for it in bandwidth and in
              frames. Switch the issuer to engine prefetch and the amplification disappears.
            </>
          ) : depth >= needed * 2 ? (
            <>
              <strong>Past the knee.</strong> Little&apos;s law wanted {needed} request
              {needed > 1 ? 's' : ''} in flight to keep a consumer that eats a page every {cpuUs} µs off
              a {SERVICE_US} µs device; at depth {depth} the extra requests only queue behind each other.
              Elapsed time is {speedup.toFixed(2)}× the demand-only baseline, which is what depth {needed}{' '}
              already gave you.
            </>
          ) : (
            <>
              <strong>Overlap is doing its job.</strong> {r.hits} of {N} pages are already resident when
              the scan reaches them, CPU utilisation is {Math.round(cpuUtil * 100)}%, and the scan runs{' '}
              {speedup.toFixed(2)}× faster than demand-only. Little&apos;s law says {needed} in flight is
              enough here — {SERVICE_US} µs of latency divided by {cpuUs} µs of work per page.
            </>
          )}{' '}
          <span>
            Outlined bars are demand reads, the ones the scan is blocked on; solid bars are prefetches.
            The dashed line is the moment the scan finishes, so device work to its right was issued for
            pages it never reached.
          </span>
        </Note>
      }
      table={
        <table className="viz-table">
          <caption>
            Depth sweep at the current pattern, issuer and CPU cost. Elapsed is the time to consume all{' '}
            {N} pages.
          </caption>
          <thead>
            <tr>
              <th>Depth</th>
              <th>Elapsed (µs)</th>
              <th>Speed-up</th>
              <th>Stalls</th>
              <th>Device reads</th>
              <th>Amplification</th>
              <th>Never demanded</th>
              <th>Evicted before use</th>
              <th>CPU util</th>
            </tr>
          </thead>
          <tbody>
            {SWEEP.map((d) => {
              const s = simulate({ ...cfg, depth: d });
              return (
                <tr key={d}>
                  <td>{d === 0 ? 'off' : d}</td>
                  <td>{fmtNum(s.elapsed)}</td>
                  <td>{(base.elapsed / s.elapsed).toFixed(2)}×</td>
                  <td>{s.stalls}</td>
                  <td>{s.reads.length}</td>
                  <td>{(s.reads.length / N).toFixed(2)}×</td>
                  <td>{s.wasted}</td>
                  <td>{s.evicted}</td>
                  <td>{Math.round(((N * cpuUs) / s.elapsed) * 100)}%</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      }
    >
      <div ref={ref}>
        <TooltipHost>
          <ScanFigure r={r} depth={depth} avail={width} />
        </TooltipHost>
      </div>
    </VizPanel>
  );
}
