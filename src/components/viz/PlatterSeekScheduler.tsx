import { useEffect, useMemo, useRef, useState } from 'react';
import {
  VizPanel,
  Slider,
  Choice,
  Button,
  Legend,
  Stats,
  Note,
  TooltipHost,
  useTip,
  useTicker,
  useSize,
  fmtNum,
} from './Viz';
import { makeRequestStream, WORKLOAD_OPTIONS, LOGICAL_PAGES, type Workload } from './FlashTranslationLab';

/* ------------------------------------------------------------------ model */

const TRACKS = 8;
const SECTORS = LOGICAL_PAGES / TRACKS; // 12 sectors per track, 96 LBAs total
const RPM = 7200;
const ROT_MS = 60_000 / RPM; // 8.33 ms per revolution
const SECTOR_MS = ROT_MS / SECTORS; // time to stream one sector under the head
const SETTLE_MS = 1.0; // track-to-track: accelerate, decelerate, settle
const FULL_STROKE_MS = 15.0; // inner-to-outer-most seek

/** Classic seek model: settle time plus a square-root term in the distance. */
function seekMs(dTracks: number) {
  if (dTracks === 0) return 0;
  return SETTLE_MS + (FULL_STROKE_MS - SETTLE_MS) * Math.sqrt(dTracks / (TRACKS - 1));
}

type Serviced = {
  i: number;
  lba: number;
  track: number;
  sector: number;
  fromTrack: number;
  seek: number;
  rot: number;
  service: number;
  tEnd: number;
};

type DiskState = {
  served: Serviced[];
  batch: { lba: number; track: number; sector: number; order: number }[];
  totalSeekTracks: number;
  totalSeek: number;
  totalRot: number;
  totalXfer: number;
  totalMs: number;
  visited: Uint8Array;
};

/**
 * Service the host's write stream on a platter. `windowSize` is the depth of the
 * reordering window: 1 is FIFO, larger is an elevator (C-SCAN) sweep.
 */
function simulateDisk(workload: Workload, windowSize: number, steps: number): DiskState {
  const raw = makeRequestStream(workload, Math.ceil(steps * 1.4) + 40);
  const reqs = raw.filter((o) => o.kind === 'write').slice(0, steps);

  const served: Serviced[] = [];
  const visited = new Uint8Array(LOGICAL_PAGES);
  let head = 0;
  let t = 0;
  let totalSeekTracks = 0;
  let totalSeek = 0;
  let totalRot = 0;
  let batch: DiskState['batch'] = [];

  for (let start = 0; start < reqs.length; start += windowSize) {
    const chunk = reqs.slice(start, start + windowSize).map((o, k) => ({
      i: start + k,
      lba: o.lba,
      track: Math.floor(o.lba / SECTORS),
      sector: o.lba % SECTORS,
    }));
    // C-SCAN: sweep outward from wherever the arm already is, then snap back.
    const ordered =
      windowSize === 1
        ? chunk
        : (() => {
            const sorted = [...chunk].sort((a, b) => a.track - b.track || a.sector - b.sector);
            const split = sorted.findIndex((r) => r.track >= head);
            return split < 0 ? sorted : [...sorted.slice(split), ...sorted.slice(0, split)];
          })();

    batch = ordered.map((r, k) => ({ lba: r.lba, track: r.track, sector: r.sector, order: k + 1 }));

    for (const r of ordered) {
      const d = Math.abs(r.track - head);
      const seek = seekMs(d);
      const t1 = t + seek;
      const pos = (t1 / ROT_MS) * SECTORS; // rotational position, in sector units
      let turns = (((r.sector - pos) % SECTORS) + SECTORS) % SECTORS;
      if (turns > SECTORS - 1e-6) turns = 0; // the sector is already under the head
      const rot = turns * SECTOR_MS;
      const service = seek + rot + SECTOR_MS;
      t = t1 + rot + SECTOR_MS;
      totalSeekTracks += d;
      totalSeek += seek;
      totalRot += rot;
      visited[r.lba] = 1;
      served.push({ ...r, fromTrack: head, seek, rot, service, tEnd: t });
      head = r.track;
    }
  }

  return {
    served,
    batch,
    totalSeekTracks,
    totalSeek,
    totalRot,
    totalXfer: served.length * SECTOR_MS,
    totalMs: t,
    visited,
  };
}

/* ---------------------------------------------------------------- drawing */

const R_OUT = 110;
const R_IN = 36;
const MAX_STEPS = 400;
const CHART_N = 48;
const CHART_H = 120;

const trackR = (t: number) => R_OUT - (t * (R_OUT - R_IN)) / (TRACKS - 1);
const sectorDeg = (s: number) => (s * 360) / SECTORS - 90;
const rad = (deg: number) => (deg * Math.PI) / 180;

export default function PlatterSeekScheduler() {
  const [workload, setWorkload] = useState<Workload>('hot');
  const [windowSize, setWindowSize] = useState(1);
  const [steps, setSteps] = useState(40);
  const [running, setRunning] = useState(false);
  const [ref, width] = useSize(760);
  const [disp, setDisp] = useState({ r: R_OUT, a: -90 });
  const tip = useTip();
  const acc = useRef(0);

  const sim = useMemo(() => simulateDisk(workload, windowSize, steps), [workload, windowSize, steps]);
  const last = sim.served[sim.served.length - 1];
  const targetR = last ? trackR(last.track) : R_OUT;
  const targetA = last ? sectorDeg(last.sector) : -90;

  useTicker((dt) => {
    acc.current += dt;
    if (acc.current < 220) return;
    acc.current = 0;
    setSteps((s) => (s >= MAX_STEPS ? s : s + 1));
  }, running);

  useEffect(() => {
    if (steps >= MAX_STEPS) setRunning(false);
  }, [steps]);

  const dA = ((targetA - disp.a + 540) % 360) - 180;
  const dR = targetR - disp.r;
  const animating = Math.abs(dR) > 0.4 || Math.abs(dA) > 0.6;
  useTicker((dt) => {
    setDisp((d) => {
      const da = ((targetA - d.a + 540) % 360) - 180;
      const dr = targetR - d.r;
      if (Math.abs(dr) < 0.4 && Math.abs(da) < 0.6) return { r: targetR, a: targetA };
      const k = Math.min(1, dt / 130);
      return { r: d.r + dr * k, a: d.a + da * k };
    });
  }, animating);

  const n = sim.served.length;
  const avgSeek = n ? sim.totalSeek / n : 0;
  const avgRot = n ? sim.totalRot / n : 0;
  const avgService = n ? sim.totalMs / n : 0;
  const iops = avgService > 0 ? 1000 / avgService : 0;
  const vsPerfect = avgService / SECTOR_MS;

  const sideBySide = width >= 640;
  const platterW = 2 * R_OUT + 66;
  const chartX = sideBySide ? platterW + 10 : 34;
  const chartY = sideBySide ? 26 : 2 * R_OUT + 58;
  const chartW = Math.max(180, (sideBySide ? width - platterW - 24 : width - 48));
  const svgW = Math.max(320, Math.min(width, sideBySide ? platterW + chartW + 16 : platterW));
  const svgH = sideBySide ? 2 * R_OUT + 52 : 2 * R_OUT + 58 + CHART_H + 34;

  const cx = R_OUT + 6;
  const cy = R_OUT + 22;
  const pivot = { x: cx + R_OUT + 48, y: cy + R_OUT * 0.62 };
  const headPt = { x: cx + disp.r * Math.cos(rad(disp.a)), y: cy + disp.r * Math.sin(rad(disp.a)) };

  // Rotational-wait arc: how far the platter had to turn after the seek landed.
  const waitDeg = last ? Math.min(359.5, (last.rot / ROT_MS) * 360) : 0;
  const arcR = last ? trackR(last.track) + 9 : R_OUT;
  const a0 = rad(targetA - waitDeg);
  const a1 = rad(targetA);
  const arcPath = last
    ? `M ${cx + arcR * Math.cos(a0)} ${cy + arcR * Math.sin(a0)} A ${arcR} ${arcR} 0 ${waitDeg > 180 ? 1 : 0} 1 ${cx + arcR * Math.cos(a1)} ${cy + arcR * Math.sin(a1)}`
    : '';

  const tail = sim.served.slice(-CHART_N);
  const cxs = (k: number) => chartX + (k * chartW) / Math.max(1, CHART_N - 1);
  const cys = (t: number) => chartY + (t * CHART_H) / (TRACKS - 1);

  return (
    <VizPanel
      title="The same request stream on a platter"
      subtitle="Every read costs an arm movement plus a wait for the sector to rotate under the head. Widen the reordering window and watch the elevator turn chaos into a sweep."
      controls={
        <>
          <Choice label="Workload" value={workload} onChange={setWorkload} options={WORKLOAD_OPTIONS} />
          <Slider
            label="Elevator window"
            min={1}
            max={16}
            value={windowSize}
            onChange={setWindowSize}
            format={(v) => (v === 1 ? 'FIFO (no reordering)' : `${v} requests (C-SCAN)`)}
          />
          <Button onClick={() => setSteps((s) => Math.min(MAX_STEPS, s + 1))}>Step</Button>
          <Button onClick={() => setSteps((s) => Math.min(MAX_STEPS, s + 25))}>+25</Button>
          <Button primary onClick={() => setRunning((r) => !r)}>
            {running ? 'Pause' : 'Run'}
          </Button>
          <Button
            onClick={() => {
              setRunning(false);
              setSteps(0);
            }}
          >
            Reset
          </Button>
        </>
      }
      legend={
        <Legend
          items={[
            { label: 'Actuator arm / head', color: 'var(--viz-1)', shape: 'line' },
            { label: 'Rotational wait', color: 'var(--viz-2)' },
            { label: 'Sector just serviced', color: 'var(--viz-2)', shape: 'dot' },
            { label: 'Sectors already visited', color: 'var(--viz-7)', shape: 'dot' },
            { label: 'Elevator batch, numbered in service order', color: 'var(--viz-6)', shape: 'dot' },
            { label: 'Head track per request', color: 'var(--viz-1)', shape: 'line' },
            { label: 'Request served with no seek (same track)', color: 'var(--viz-8)', shape: 'dot' },
          ]}
        />
      }
      stats={
        <Stats
          items={[
            { label: 'Requests served', value: fmtNum(n) },
            { label: 'Arm travel', value: `${fmtNum(sim.totalSeekTracks)} tracks`, hint: 'Total seek distance' },
            { label: 'Avg seek', value: `${avgSeek.toFixed(2)} ms` },
            { label: 'Avg rotational wait', value: `${avgRot.toFixed(2)} ms`, hint: 'Half a revolution when random: 4.17 ms' },
            { label: 'Avg service time', value: `${avgService.toFixed(2)} ms` },
            { label: 'IOPS', value: fmtNum(iops), hint: 'Single-threaded, queue depth 1' },
            { label: 'vs. perfect sequential', value: `${vsPerfect.toFixed(1)}×` },
          ]}
        />
      }
      note={
        <Note>
          {last ? (
            <>
              <strong>
                lba {last.lba} → track {last.track}, sector {last.sector}
              </strong>{' '}
              — arm moved {Math.abs(last.track - last.fromTrack)} track
              {Math.abs(last.track - last.fromTrack) === 1 ? '' : 's'} ({last.seek.toFixed(2)} ms), then waited{' '}
              {last.rot.toFixed(2)} ms for the sector to come round, then streamed it in{' '}
              {SECTOR_MS.toFixed(2)} ms. Of the {sim.totalMs.toFixed(0)} ms spent so far,{' '}
              <strong>{((sim.totalXfer / sim.totalMs) * 100).toFixed(1)}%</strong> moved data; the rest was
              mechanics.{' '}
              {windowSize > 1
                ? `The elevator is sorting ${windowSize} requests at a time, which is why the staircase climbs instead of thrashing.`
                : 'With FIFO the arm chases the host’s address order — raise the elevator window and watch arm travel fall.'}
            </>
          ) : (
            <>Press <strong>Run</strong>. The arm parks at the outer track and nothing has been serviced yet.</>
          )}
        </Note>
      }
      table={
        <>
          <table className="viz-table">
            <thead>
              <tr>
                <th>#</th>
                <th>LBA</th>
                <th>Track</th>
                <th>Sector</th>
                <th>Seek (ms)</th>
                <th>Rotational wait (ms)</th>
                <th>Service (ms)</th>
              </tr>
            </thead>
            <tbody>
              {sim.served.slice(-14).map((r) => (
                <tr key={r.i}>
                  <td>{r.i + 1}</td>
                  <td>{r.lba}</td>
                  <td>{r.track}</td>
                  <td>{r.sector}</td>
                  <td>{r.seek.toFixed(2)}</td>
                  <td>{r.rot.toFixed(2)}</td>
                  <td>{r.service.toFixed(2)}</td>
                </tr>
              ))}
            </tbody>
          </table>
          <p>
            Model: {RPM} rpm ({ROT_MS.toFixed(2)} ms per revolution), {TRACKS} radial tracks standing in for a
            real drive’s ~100 000 cylinders, {SECTORS} sectors per track, track-to-track settle {SETTLE_MS} ms,
            full-stroke seek {FULL_STROKE_MS} ms, seek time = settle + (stroke − settle)·√(distance/max). Perfect
            sequential service time is one sector time, {SECTOR_MS.toFixed(2)} ms.
          </p>
        </>
      }
    >
      <div ref={ref}>
        <TooltipHost>
          <svg width={svgW} height={svgH} role="img" aria-label="Disk platter with actuator arm, and head track over the last requests">
            {/* ---- platter ---- */}
            <circle cx={cx} cy={cy} r={R_OUT + 6} fill="var(--viz-plane)" stroke="var(--viz-axis)" />
            {Array.from({ length: TRACKS }, (_, t) => (
              <circle key={t} cx={cx} cy={cy} r={trackR(t)} fill="none" className="viz-grid-line" />
            ))}
            <circle cx={cx} cy={cy} r={R_IN - 14} fill="var(--viz-neutral)" stroke="var(--viz-axis)" />

            {Array.from({ length: LOGICAL_PAGES }, (_, lba) => {
              const t = Math.floor(lba / SECTORS);
              const s = lba % SECTORS;
              const r = trackR(t);
              const a = rad(sectorDeg(s));
              const isLast = last && last.lba === lba;
              return (
                <circle
                  key={lba}
                  cx={cx + r * Math.cos(a)}
                  cy={cy + r * Math.sin(a)}
                  r={isLast ? 4.5 : 2.4}
                  fill={isLast ? 'var(--viz-2)' : sim.visited[lba] ? 'var(--viz-7)' : 'var(--viz-neutral)'}
                  stroke={sim.visited[lba] || isLast ? 'var(--viz-surface)' : 'var(--viz-axis)'}
                  strokeWidth={1}
                />
              );
            })}

            {/* ---- rotational wait the last request paid ---- */}
            {last && waitDeg > 1 ? (
              <path d={arcPath} fill="none" stroke="var(--viz-2)" strokeWidth={3} strokeLinecap="round" />
            ) : null}

            {/* ---- the seek the arm just made, as a radial span ---- */}
            {last && last.track !== last.fromTrack ? (
              <line
                x1={cx + trackR(last.fromTrack) * Math.cos(rad(disp.a))}
                y1={cy + trackR(last.fromTrack) * Math.sin(rad(disp.a))}
                x2={headPt.x}
                y2={headPt.y}
                stroke="var(--viz-1)"
                strokeWidth={2}
                strokeDasharray="3 3"
              />
            ) : null}

            {/* ---- queued requests, numbered in service order ---- */}
            {windowSize > 1
              ? sim.batch.slice(0, 10).map((b) => {
                  const r = trackR(b.track);
                  const a = rad(sectorDeg(b.sector));
                  return (
                    <text
                      key={`${b.lba}-${b.order}`}
                      x={cx + (r - 9) * Math.cos(a)}
                      y={cy + (r - 9) * Math.sin(a) + 3}
                      fontSize={9}
                      textAnchor="middle"
                      fill="var(--viz-6)"
                    >
                      {b.order}
                    </text>
                  );
                })
              : null}

            {/* ---- actuator arm ---- */}
            <line x1={pivot.x} y1={pivot.y} x2={headPt.x} y2={headPt.y} stroke="var(--viz-1)" strokeWidth={3} />
            <circle cx={pivot.x} cy={pivot.y} r={6} fill="var(--viz-1)" />
            <circle cx={headPt.x} cy={headPt.y} r={4} fill="var(--viz-1)" stroke="var(--viz-surface)" strokeWidth={1.5} />
            <text x={cx} y={cy - R_OUT - 12} fontSize={10} textAnchor="middle" fill="var(--viz-ink-2)">
              track 0 = outer edge · LBA 0 starts here
            </text>

            {/* ---- head position over the last requests ---- */}
            <text x={chartX} y={chartY - 10} fontSize={11} fill="var(--viz-ink)">
              Head track, last {CHART_N} requests
            </text>
            {Array.from({ length: TRACKS }, (_, t) => (
              <g key={t}>
                <line className="viz-grid-line" x1={chartX} x2={chartX + chartW} y1={cys(t)} y2={cys(t)} />
                <text x={chartX - 6} y={cys(t) + 3} fontSize={9} textAnchor="end" fill="var(--viz-ink-muted)">
                  {t}
                </text>
              </g>
            ))}
            {tail.length > 1 ? (
              <polyline
                points={tail.map((r, k) => `${cxs(k)},${cys(r.track)}`).join(' ')}
                fill="none"
                stroke="var(--viz-1)"
                strokeWidth={1.6}
              />
            ) : null}
            {tail.map((r, k) => (
              <circle
                key={r.i}
                cx={cxs(k)}
                cy={cys(r.track)}
                r={r.seek === 0 ? 2.8 : 2.2}
                fill={r.seek === 0 ? 'var(--viz-8)' : 'var(--viz-1)'}
                {...tip(
                  <>
                    <strong>request {r.i + 1}</strong> · lba {r.lba}
                    <br />
                    track {r.fromTrack} → {r.track}, seek {r.seek.toFixed(2)} ms, rotation {r.rot.toFixed(2)} ms
                  </>,
                )}
              />
            ))}
            <line className="viz-axis-line" x1={chartX} x2={chartX + chartW} y1={cys(TRACKS - 1)} y2={cys(TRACKS - 1)} />
          </svg>
        </TooltipHost>
      </div>
    </VizPanel>
  );
}
