import { useMemo, useState } from 'react';
import {
  VizPanel,
  Slider,
  Segmented,
  Choice,
  Legend,
  Stats,
  TooltipHost,
  useTip,
  fmtBytes,
  fmtNum,
  fmtTime,
  useSize,
} from './Viz';

/**
 * Cold-pool warm-up, raced three ways on one bandwidth-limited device.
 *
 * The model is a closed-loop point-read workload over a hot working set of 8 KB pages:
 *
 *   hit          = resident working-set pages / working-set pages
 *   achieved qps = min(offered, device_iops / (1 - hit))        <- the miss rate throttles the app
 *   p99          = device read latency while the miss rate is above 1%, memory latency below it
 *
 * One shared device budget: a random page read costs 1/iops seconds of device time, a
 * sequential page costs page_bytes/bandwidth. Foreground demand misses are served first;
 * whatever device time is left goes to the warm-up strategy:
 *
 *   none  nothing; pages arrive only when a query asks for one.
 *   dump  a background loader walks a dumped list of page identifiers at full queue depth
 *         (innodb_buffer_pool_load_at_startup, pg_prewarm's autoprewarm worker). Every page
 *         it reads is a page the workload wants, but the list only covers dump_pct of the
 *         pre-event resident set.
 *   seq   a streaming read of the whole hot relation (pg_prewarm(..., 'buffer')) at bandwidth
 *         rather than IOPS -- but the relation is RELATION_MULT x the hot set, so most of the
 *         pages it pulls in are ballast that also competes for frames.
 *
 * Deterministic: closed-form integration, no randomness anywhere.
 */

const PAGE = 8192;
const OFFERED = 20_000; // point reads/s the application wants to issue
const HIT_MS = 0.15; // a point read served entirely from the pool
const RELATION_MULT = 2.5; // hot relation size / hot working set
const REPLAY_WARM = 0.35; // fraction of the READ working set a replay-only standby holds
const LIST_BYTES = 10; // bytes per page identifier in the dump file (text "space,page")
const LOADER_SHARE = 0.5; // device time the background warm-up reader takes from the live workload
const BATCH_GAIN = 3; // a sorted, batched list load reads this much faster per page than scattered misses
const TMAX = 21_600; // 6 h simulation ceiling

type Dev = { id: string; label: string; iops: number; bw: number; missMs: number };

const DEVICES: Dev[] = [
  { id: 'gp3', label: 'EBS gp3 — baseline 3,000 IOPS / 125 MB/s', iops: 3_000, bw: 125e6, missMs: 1.4 },
  { id: 'gp3max', label: 'EBS gp3 — maxed 16,000 IOPS / 1,000 MB/s', iops: 16_000, bw: 1000e6, missMs: 0.9 },
  { id: 'nvme', label: 'Local NVMe — 150k IOPS / 3 GB/s', iops: 150_000, bw: 3e9, missMs: 0.25 },
  { id: 'hdd', label: 'HDD RAID-10 — 600 IOPS / 300 MB/s', iops: 600, bw: 300e6, missMs: 12 },
];

type EvId = 'restart' | 'crash' | 'failover' | 'resize';

type Ev = {
  id: EvId;
  label: string;
  parts: { label: string; s: number }[]; // everything before the first query is served
  startWarm: number; // fraction of the working set already resident when serving starts
  listScale: number; // 0 = no usable dump list; 0.5 = the list describes the old, half-size pool
  blurb: string;
};

const EVENTS: Ev[] = [
  {
    id: 'restart',
    label: 'Clean restart',
    parts: [{ label: 'shutdown + startup', s: 25 }],
    startWarm: 0,
    listScale: 1,
    blurb:
      'A clean shutdown is the one case both engines dump for free: innodb_buffer_pool_dump_at_shutdown ' +
      'and the autoprewarm worker both write the page-identifier list on the way out.',
  },
  {
    id: 'crash',
    label: 'Crash / OOM-kill',
    parts: [
      { label: 'process restart', s: 25 },
      { label: 'redo replay', s: 45 },
    ],
    startWarm: 0,
    listScale: 1,
    blurb:
      'Nothing ran on the way down, so the list is whatever the last periodic dump wrote — every ' +
      'pg_prewarm.autoprewarm_interval on Postgres, and on InnoDB nothing at all unless you schedule ' +
      'innodb_buffer_pool_dump_now. Redo replay also has to finish before the first query is served.',
  },
  {
    id: 'failover',
    label: 'Failover to a standby',
    parts: [
      { label: 'failure detection', s: 15 },
      { label: 'promotion', s: 5 },
    ],
    startWarm: REPLAY_WARM,
    listScale: 0,
    blurb:
      'The standby was applying WAL, so its pool is warm for the pages recovery touched — the write ' +
      'set — and cold for everything the read workload is about to ask for. There is no list to load: ' +
      'its pool is already loaded, with the wrong pages.',
  },
  {
    id: 'resize',
    label: 'Pool resize (doubled)',
    parts: [{ label: 'restart for the new size', s: 25 }],
    startWarm: 0,
    listScale: 0.5,
    blurb:
      'Postgres needs a restart to change shared_buffers, and a restart empties the pool. The dump ' +
      'list you load back describes the old, half-size pool, so even a perfect load leaves the new ' +
      'half of the frames empty.',
  },
];

type Strat = 'none' | 'dump' | 'seq';

const STRATS: { id: Strat; label: string; short: string; color: string; dash?: string }[] = [
  { id: 'none', label: 'No prewarm', short: 'none', color: 'var(--viz-1)' },
  { id: 'dump', label: 'Dump / load page-id list', short: 'dump+load', color: 'var(--viz-2)', dash: '7 4' },
  { id: 'seq', label: 'Sequential prewarm of the relation', short: 'sequential', color: 'var(--viz-3)', dash: '2 3' },
];

/* --------------------------------------------------------------- the model */

type Pt = { t: number; hit: number; qps: number; p99: number };
type Sim = { pts: Pt[]; tWarm: number | null; io: number; junk: number; listUsed: number };

type Cfg = {
  dev: Dev;
  poolPages: number;
  wsPages: number;
  listPages: number;
  startWarm: number;
  steady: number;
};

function p99ms(dev: Dev, hit: number, fgIops: number) {
  const miss = 1 - hit;
  if (miss < 0.01) return HIT_MS;
  const queue = Math.min(6, Math.max(1, (OFFERED * miss) / fgIops));
  return dev.missMs * queue;
}

function simulate(strat: Strat, c: Cfg): Sim {
  const { dev, poolPages, wsPages, listPages } = c;
  const relPages = wsPages * RELATION_MULT;
  const target = c.steady * 0.99;
  const batchRate = Math.min(dev.iops * BATCH_GAIN, dev.bw / PAGE); // sorted list reads engage read-ahead
  const seqRate = dev.bw / PAGE;

  let warm = c.startWarm * Math.min(poolPages, wsPages);
  let warmDemand = warm; // pages the live workload faulted in — what a warm-up reader finds already resident
  let junk = 0;
  let listDone = 0;
  let relRead = 0;
  let io = 0;
  let t = 0;
  let tWarm: number | null = null;

  const pts: Pt[] = [];
  const sample = (fgIops: number) => {
    const hit = Math.min(warm / wsPages, 1);
    const qps = hit >= 1 ? OFFERED : Math.min(OFFERED, fgIops / (1 - hit));
    pts.push({ t, hit, qps, p99: p99ms(dev, hit, fgIops) });
  };

  const busy = () =>
    (strat === 'dump' && listDone < listPages - 1) || (strat === 'seq' && relRead < relPages - 1);

  sample(dev.iops * (busy() ? 1 - LOADER_SHARE : 1));
  if (pts[0].hit >= target) tWarm = 0;

  while (t < TMAX) {
    const dt = t < 60 ? 0.1 : t < 900 ? 1 : 5;
    const frac = Math.min(warm / wsPages, 0.999);
    // A warm-up reader walks its list or the file once and never revisits a page, so the
    // pages it finds already resident are the ones DEMAND faulted in — not the ones it
    // warmed itself. Overlap is therefore measured against warmDemand, not warm.
    const dFrac = Math.min(warmDemand / wsPages, 0.999);
    const loading = busy();

    // The warm-up reader and the live workload share one device. The reader takes its
    // share when it wants it, plus anything the foreground does not use.
    const fgIops = dev.iops * (loading ? 1 - LOADER_SHARE : 1);
    const fgServed = Math.min(OFFERED * (1 - frac), fgIops);
    const readerTime = loading ? Math.max(LOADER_SHARE, 1 - fgServed / dev.iops) : 0;

    let add = fgServed * dt;
    io += fgServed * dt;
    warmDemand = Math.min(warmDemand + fgServed * dt, wsPages);

    if (strat === 'dump' && listDone < listPages) {
      const budget = readerTime * batchRate * dt;
      const usefulLeft = (listPages - listDone) * (1 - dFrac);
      const useful = Math.min(budget, usefulLeft);
      listDone = Math.min(listPages, listDone + useful / Math.max(0.001, 1 - dFrac));
      add += useful;
      io += useful;
    }

    if (strat === 'seq' && relRead < relPages) {
      const read = Math.min(readerTime * seqRate * dt, relPages - relRead);
      relRead += read;
      io += read;
      const wsShare = read / RELATION_MULT;
      const junkCap = Math.max(1, relPages - wsPages);
      add += wsShare * (1 - dFrac);
      junk = Math.min(junkCap, junk + (read - wsShare));
    }

    warm = Math.min(warm + add, wsPages);
    if (warm + junk > poolPages) {
      const tot = warm + junk;
      const ex = tot - poolPages;
      warm -= (ex * warm) / tot;
      junk -= (ex * junk) / tot;
    }
    warmDemand = Math.min(warmDemand, warm);

    t += dt;
    sample(fgIops);
    if (tWarm === null && warm / wsPages >= target) tWarm = t;
  }

  return { pts, tWarm, io, junk, listUsed: listDone };
}

function at(pts: Pt[], t: number): Pt {
  let lo = 0;
  let hi = pts.length - 1;
  while (lo < hi) {
    const mid = (lo + hi + 1) >> 1;
    if (pts[mid].t <= t) lo = mid;
    else hi = mid - 1;
  }
  return pts[lo];
}

function fmtDur(s: number | null) {
  if (s === null) return '> 6 h';
  if (s < 1) return `${s.toFixed(1)} s`;
  if (s < 120) return `${Math.round(s)} s`;
  if (s < 7200) return `${(s / 60).toFixed(1)} min`;
  return `${(s / 3600).toFixed(1)} h`;
}

/* -------------------------------------------------------------- the drawing */

export default function ColdPoolWarmupRace() {
  const [devId, setDevId] = useState('gp3');
  const [evId, setEvId] = useState<EvId>('restart');
  const [strat, setStrat] = useState<Strat>('dump');
  const [poolGb, setPoolGb] = useState(32);
  const [wsGb, setWsGb] = useState(48);
  const [dumpPct, setDumpPct] = useState(25);
  const [inspect, setInspect] = useState(35); // percent of the horizon
  const [ref, width] = useSize(760);

  const dev = DEVICES.find((d) => d.id === devId)!;
  const ev = EVENTS.find((e) => e.id === evId)!;
  const delay = ev.parts.reduce((a, p) => a + p.s, 0);

  const poolPages = (poolGb * 2 ** 30) / PAGE;
  const wsPages = (wsGb * 2 ** 30) / PAGE;
  const steady = Math.min(poolPages, wsPages) / wsPages;
  const listPages = Math.min(poolPages * ev.listScale, wsPages) * (dumpPct / 100);

  const cfg: Cfg = { dev, poolPages, wsPages, listPages, startWarm: ev.startWarm, steady };

  const sims = useMemo(
    () => ({
      none: simulate('none', cfg),
      dump: simulate('dump', cfg),
      seq: simulate('seq', cfg),
    }),
    [devId, evId, poolGb, wsGb, dumpPct],
  );

  const sel = sims[strat];
  const fastest = Math.min(...STRATS.map((s) => sims[s.id].tWarm ?? TMAX));
  // Zoom to the strategy being measured, but never so far in that the fastest curve is a wall.
  const horizon = Math.max(30, delay + Math.min(TMAX, Math.max((sel.tWarm ?? TMAX) * 1.3, fastest * 1.6)));
  const tNow = Math.min((inspect / 100) * horizon, horizon);
  const serving = tNow >= delay;
  const now = at(sel.pts, Math.max(0, tNow - delay));

  const W = Math.max(340, Math.min(width, 860));
  const H = 226;
  const M = { l: 44, r: 118, t: 14, b: 26 };
  const pw = W - M.l - M.r;
  const ph = H - M.t - M.b;
  const x = (t: number) => M.l + (t / horizon) * pw;
  const y = (h: number) => M.t + (1 - h) * ph;

  const path = (pts: Pt[]) => {
    const out: string[] = [];
    const stride = Math.max(1, Math.floor(pts.length / 420));
    for (let i = 0; i < pts.length; i += stride) {
      const p = pts[i];
      if (p.t + delay > horizon) break;
      out.push(`${x(p.t + delay).toFixed(1)},${y(p.hit).toFixed(1)}`);
    }
    const last = pts[pts.length - 1];
    if (last.t + delay >= horizon) out.push(`${x(horizon).toFixed(1)},${y(at(pts, horizon - delay).hit).toFixed(1)}`);
    return out.join(' ');
  };

  const bwPages = dev.bw / PAGE;
  const seqEdge = bwPages / dev.iops; // how many random reads one second of streaming is worth
  const evictionRisk = wsPages * RELATION_MULT > poolPages;
  const winner = STRATS.reduce((a, b) => ((sims[a.id].tWarm ?? TMAX) <= (sims[b.id].tWarm ?? TMAX) ? a : b));

  const rtoSegs = [
    ...ev.parts.map((p, i) => ({
      label: p.label,
      s: p.s,
      color: i === 0 ? 'var(--viz-seq-250)' : 'var(--viz-seq-550)',
    })),
    { label: `warm-up (${STRATS.find((s) => s.id === strat)!.short})`, s: sel.tWarm ?? TMAX, color: STRATS.find((s) => s.id === strat)!.color },
  ];
  const rtoTotal = rtoSegs.reduce((a, s) => a + s.s, 0);

  return (
    <VizPanel
      title="Cold pool: how long until the hit rate comes back"
      subtitle="A steady 20,000 point-reads/s over a hot working set of 8 KB pages. Knock the pool over, then race three warm-up strategies on the same device budget."
      controls={
        <>
          <Choice label="Device" value={devId} onChange={setDevId} options={DEVICES.map((d) => ({ value: d.id, label: d.label }))} />
          <Choice label="Event" value={evId} onChange={(v) => setEvId(v as EvId)} options={EVENTS.map((e) => ({ value: e.id, label: e.label }))} />
          <Segmented
            label="Measure"
            value={strat}
            onChange={setStrat}
            options={STRATS.map((s) => ({ value: s.id, label: s.short, title: s.label }))}
          />
          <Slider label="Buffer pool" min={4} max={192} step={4} value={poolGb} onChange={setPoolGb} format={(n) => fmtBytes(n * 2 ** 30)} />
          <Slider label="Hot working set" min={4} max={256} step={4} value={wsGb} onChange={setWsGb} format={(n) => fmtBytes(n * 2 ** 30)} />
          <Slider
            label="innodb_buffer_pool_dump_pct"
            min={5}
            max={100}
            step={5}
            value={dumpPct}
            onChange={setDumpPct}
            format={(n) => `${n}%${ev.listScale === 0 ? ' (no list)' : ''}`}
          />
          <Slider label="Inspect at" min={0} max={100} value={inspect} onChange={setInspect} format={() => fmtDur(tNow)} />
        </>
      }
      legend={
        <Legend
          items={[
            { label: 'No prewarm (solid)', color: 'var(--viz-1)', shape: 'line' },
            { label: 'Dump / load page-id list (dashed)', color: 'var(--viz-2)', shape: 'line' },
            { label: 'Sequential prewarm (dotted)', color: 'var(--viz-3)', shape: 'line' },
            { label: 'Pre-event hit rate', color: 'var(--viz-ink-muted)', shape: 'line' },
            { label: 'Not serving queries yet', color: 'var(--viz-neutral)' },
          ]}
        />
      }
      stats={
        <Stats
          items={[
            { label: 'Pre-event hit rate', value: `${(steady * 100).toFixed(1)}%`, hint: 'min(pool, working set) / working set' },
            { label: 'Warm-up, measured strategy', value: fmtDur(sel.tWarm), hint: 'From first query served to within 1% of the pre-event hit rate' },
            { label: 'Total RTO', value: fmtDur(rtoTotal), hint: 'Detection/restart + replay/promotion + warm-up' },
            { label: `Hit rate at ${fmtDur(tNow)}`, value: serving ? `${(now.hit * 100).toFixed(1)}%` : '—' },
            { label: 'p99 point read', value: serving ? fmtTime(now.p99 * 1e6) : 'not serving' },
            { label: 'Throughput', value: serving ? `${fmtNum(now.qps)} qps` : '0 qps', hint: `Offered ${fmtNum(OFFERED)} qps; the miss rate throttles it to iops / miss-rate` },
            { label: 'Dump file', value: listPages > 0 ? `≈ ${fmtBytes(listPages * LIST_BYTES)}` : 'none', hint: `${fmtNum(listPages)} page identifiers — the list never holds page contents` },
          ]}
        />
      }
      note={
        <>
          <strong>
            {winner.short === 'none'
              ? 'Nothing beats plain demand paging here.'
              : `${winner.label} wins this race.`}
          </strong>{' '}
          Streaming this device is worth {seqEdge.toFixed(1)} random page reads per second of device time, and a
          sweep of the hot relation reads {RELATION_MULT} pages for every one the workload wanted — so the
          sequential route only pays when that ratio comfortably clears {RELATION_MULT}. {ev.blurb}{' '}
          {evictionRisk
            ? `And the relation (${fmtBytes(wsPages * RELATION_MULT * PAGE)}) does not fit in the pool (${fmtBytes(
                poolPages * PAGE,
              )}), so prewarming all of it evicts what it already warmed — the classic way to make warm-up slower by trying harder.`
            : 'The relation fits in the pool, so the sequential sweep costs frames but never evicts what it warmed.'}
        </>
      }
      table={
        <>
          <table className="viz-table">
            <thead>
              <tr>
                <th>Strategy</th>
                <th>Warm-up</th>
                <th>Total RTO</th>
                <th>Read from disk</th>
                <th>Hit rate @ 60 s</th>
                <th>p99 @ 60 s</th>
                <th>qps @ 60 s</th>
              </tr>
            </thead>
            <tbody>
              {STRATS.map((s) => {
                const sim = sims[s.id];
                const p = at(sim.pts, 60);
                return (
                  <tr key={s.id}>
                    <td>{s.label}</td>
                    <td>{fmtDur(sim.tWarm)}</td>
                    <td>{fmtDur(delay + (sim.tWarm ?? TMAX))}</td>
                    <td>{fmtBytes(sim.io * PAGE)}</td>
                    <td>{(p.hit * 100).toFixed(1)}%</td>
                    <td>{fmtTime(p.p99 * 1e6)}</td>
                    <td>{fmtNum(p.qps)}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
          <p>
            Model: {fmtNum(OFFERED)} offered point reads/s over {fmtBytes(wsPages * PAGE)} of hot 8 KB pages, in
            a closed loop — every miss blocks its request, so throughput is device IOPS ÷ miss rate and p99 is the
            device read latency until the miss rate drops under 1%. A background warm-up reader takes{' '}
            {Math.round(LOADER_SHARE * 100)}% of the device from the live workload plus whatever the workload
            leaves; a sorted list load reads {BATCH_GAIN}× faster per page than scattered demand misses (capped
            by bandwidth), and a sequential sweep reads at {fmtBytes(dev.bw)}/s. The hot relation is{' '}
            {RELATION_MULT}× the hot set, and a replay-only standby starts {Math.round(REPLAY_WARM * 100)}% warm
            for the read set.
          </p>
        </>
      }
    >
      <div ref={ref}>
        <TooltipHost>
          <RaceSvg
            W={W}
            H={H}
            M={M}
            pw={pw}
            ph={ph}
            x={x}
            y={y}
            path={path}
            sims={sims}
            strat={strat}
            steady={steady}
            delay={delay}
            horizon={horizon}
            tNow={tNow}
            serving={serving}
            nowHit={now.hit}
            rtoSegs={rtoSegs}
            rtoTotal={rtoTotal}
          />
        </TooltipHost>
      </div>
    </VizPanel>
  );
}

function RaceSvg(props: {
  W: number;
  H: number;
  M: { l: number; r: number; t: number; b: number };
  pw: number;
  ph: number;
  x: (t: number) => number;
  y: (h: number) => number;
  path: (pts: Pt[]) => string;
  sims: Record<Strat, Sim>;
  strat: Strat;
  steady: number;
  delay: number;
  horizon: number;
  tNow: number;
  serving: boolean;
  nowHit: number;
  rtoSegs: { label: string; s: number; color: string }[];
  rtoTotal: number;
}) {
  const { W, H, M, pw, ph, x, y, path, sims, strat, steady, delay, horizon, tNow, serving, nowHit } = props;
  const tip = useTip();
  // Direct end-of-line labels, pushed apart so no series is identified by color alone.
  const labelYs: Record<Strat, number> = (() => {
    const rows = STRATS.map((s) => ({ id: s.id, yy: y(at(sims[s.id].pts, horizon - delay).hit) + 4 })).sort(
      (a, b) => a.yy - b.yy,
    );
    for (let i = 1; i < rows.length; i++) rows[i].yy = Math.max(rows[i].yy, rows[i - 1].yy + 13);
    const out = {} as Record<Strat, number>;
    for (const r of rows) out[r.id] = r.yy;
    return out;
  })();
  const ticks = [0, 0.25, 0.5, 0.75, 1];
  const barY = H + 26;
  const barW = pw;
  const rtoScale = (s: number) => (s / props.rtoTotal) * barW;

  return (
    <svg width={W} height={H + 88} role="img" aria-label="Buffer pool hit rate over time after a restart, three warm-up strategies">
      {/* y grid */}
      {ticks.map((t) => (
        <g key={t}>
          <line x1={M.l} x2={M.l + pw} y1={y(t)} y2={y(t)} stroke="var(--viz-grid)" />
          <text x={M.l - 8} y={y(t) + 4} textAnchor="end" fill="var(--viz-ink-muted)" fontSize={11}>
            {t * 100}%
          </text>
        </g>
      ))}

      {/* not-serving band */}
      <rect x={M.l} y={M.t} width={Math.max(1, x(delay) - M.l)} height={ph} fill="var(--viz-neutral)" />
      <text x={M.l + 4} y={M.t + 13} fill="var(--viz-ink-muted)" fontSize={11}>
        not serving
      </text>

      {/* pre-event hit rate */}
      <line
        x1={M.l}
        x2={M.l + pw}
        y1={y(steady)}
        y2={y(steady)}
        stroke="var(--viz-ink-muted)"
        strokeDasharray="4 4"
      />
      <text x={M.l + pw + 6} y={y(steady) - 6} fill="var(--viz-ink-muted)" fontSize={11}>
        pre-event {(steady * 100).toFixed(0)}%
      </text>

      {/* curves */}
      {STRATS.map((s) => {
        const sim = sims[s.id];
        const endHit = at(sim.pts, horizon - delay).hit;
        const labelY = labelYs[s.id];
        return (
          <g key={s.id}>
            <polyline
              points={path(sim.pts)}
              fill="none"
              stroke={s.color}
              strokeWidth={s.id === strat ? 2.6 : 1.4}
              strokeDasharray={s.dash}
              opacity={s.id === strat ? 1 : 0.55}
            />
            <line
              x1={M.l + pw}
              x2={M.l + pw + 4}
              y1={y(endHit)}
              y2={labelY - 4}
              stroke={s.color}
              strokeWidth={1}
            />
            <text
              x={M.l + pw + 6}
              y={labelY}
              fill={s.color}
              fontSize={11}
              fontWeight={s.id === strat ? 700 : 400}
            >
              {s.short}
            </text>
            {sim.tWarm !== null && sim.tWarm + delay <= horizon ? (
              <g {...tip(
                <>
                  <strong>{s.label}</strong>
                  <br />
                  warm at {fmtDur(sim.tWarm)} after the first query
                </>,
              )}>
                <polygon
                  points={`${x(sim.tWarm + delay)},${M.t + ph} ${x(sim.tWarm + delay) - 5},${M.t + ph + 9} ${
                    x(sim.tWarm + delay) + 5
                  },${M.t + ph + 9}`}
                  fill={s.color}
                />
              </g>
            ) : null}
          </g>
        );
      })}

      {/* inspection marker */}
      <line x1={x(tNow)} x2={x(tNow)} y1={M.t} y2={M.t + ph} stroke="var(--viz-ink)" strokeWidth={1} />
      {serving ? <circle cx={x(tNow)} cy={y(nowHit)} r={4} fill="var(--viz-ink)" /> : null}

      <line x1={M.l} x2={M.l + pw} y1={M.t + ph} y2={M.t + ph} stroke="var(--viz-axis)" />
      <text x={M.l} y={H - 4} fill="var(--viz-ink-muted)" fontSize={11}>
        0
      </text>
      <text x={M.l + pw} y={H - 4} textAnchor="end" fill="var(--viz-ink-muted)" fontSize={11}>
        {fmtDur(horizon)} after the event
      </text>

      {/* RTO budget bar */}
      <text x={M.l} y={barY - 8} fill="var(--viz-ink-2)" fontSize={11}>
        RTO budget — {fmtDur(props.rtoTotal)} total
      </text>
      {props.rtoSegs.map((seg, i) => {
        const off = props.rtoSegs.slice(0, i).reduce((a, s) => a + rtoScale(s.s), 0);
        const w = Math.max(1, rtoScale(seg.s));
        return (
          <g key={seg.label} {...tip(<>
            <strong>{seg.label}</strong>
            <br />
            {fmtDur(seg.s)} — {((seg.s / props.rtoTotal) * 100).toFixed(0)}% of the budget
          </>)}>
            <rect x={M.l + off} y={barY} width={w} height={20} fill={seg.color} />
            {w > 64 ? (
              <text x={M.l + off + 6} y={barY + 14} fill="var(--viz-surface)" fontSize={11}>
                {seg.label} {fmtDur(seg.s)}
              </text>
            ) : null}
          </g>
        );
      })}
      <text x={M.l} y={barY + 38} fill="var(--viz-ink-muted)" fontSize={11}>
        Detection and replay are the terms runbooks budget for; the bar is mostly the term they leave out.
      </text>
    </svg>
  );
}
