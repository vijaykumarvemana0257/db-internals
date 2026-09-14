import { useEffect, useMemo, useRef, useState } from 'react';
import {
  VizPanel,
  Segmented,
  Slider,
  Button,
  Legend,
  Stats,
  Note,
  TooltipHost,
  useTip,
  makeRng,
  useTicker,
  useSize,
} from './Viz';

/**
 * Sequential flooding and the three real defenses against it.
 *
 * One pool of POOL frames, a HOT-page working set served by point reads, and a
 * table scan that starts at tick WARM and walks a table many times the size of
 * the pool. Three replacement strategies, each modelled the way the engine does it:
 *
 *  - 'lru'      naive: every page read goes to the head of one recency list.
 *  - 'ring'     PostgreSQL BufferAccessStrategy: the scan is given a small ring of
 *               frames taken out of the pool and recycles them round-robin; pages
 *               it reads never enter the main recency list at all.
 *  - 'midpoint' InnoDB: one list split into a young and an old sublist by the
 *               midpoint at innodb_old_blocks_pct. A newly read page is inserted at
 *               the midpoint, and is promoted to the head only if it is touched
 *               again at least innodb_old_blocks_time after its first access.
 *               (A hit inside the young sublist relinks to the head only when the
 *               page is not already near it — InnoDB skips that work for pages it
 *               judges young enough, which is approximated here by position.)
 *
 * Everything is a pure function of (policy, ring frames, old_blocks_time); the only
 * randomness is the point-read sequence, drawn once from makeRng with a fixed seed.
 */

const POOL = 40; // frames in the modelled pool
const HOT = 20; // pages in the hot working set
const WARM = 40; // ticks before the scan starts
const SCAN = 60; // ticks the scan runs for
const TAIL = 40; // ticks after the scan ends
const TOTAL = WARM + SCAN + TAIL;
const POINT_READS = 2; // point reads issued per tick
const SCAN_PAGES = 8; // pages the scan pulls per tick
const SCAN_TOUCHES = 3; // row-level touches of each scan page (many rows per heap page)
const OLD_PCT = 37; // innodb_old_blocks_pct default
const WINDOW = 20; // trailing point reads behind the hit-rate readout
const TABLE_PAGES = SCAN * SCAN_PAGES;

type Policy = 'lru' | 'ring' | 'midpoint';
type Kind = 'hot' | 'scan' | 'cold';
type Frame = { page: string; kind: Kind; ins: number };
type Phase = 'warm' | 'scan' | 'after';

type Snap = {
  list: Frame[];
  ring: (Frame | null)[];
  ringPos: number;
  hitPct: number;
  hotRes: number;
  scanFrames: number;
  misses: number;
  recycles: number;
  phase: Phase;
};

/** The point-read sequence: fixed, so every render of every policy sees the same load. */
const POINT_SEQ: number[] = (() => {
  const rng = makeRng(20250913);
  const out: number[] = [];
  for (let i = 0; i < TOTAL * POINT_READS; i++) out.push(Math.floor(rng() * HOT));
  return out;
})();

const phaseOf = (t: number): Phase => (t < WARM ? 'warm' : t < WARM + SCAN ? 'scan' : 'after');

function simulate(policy: Policy, ringN: number, oldTicks: number) {
  const cap = policy === 'ring' ? POOL - ringN : POOL;
  // The pool starts full, the way a production pool always is: cold pages left over
  // from earlier work, none of which will be read again.
  let list: Frame[] = Array.from({ length: cap }, (_, i) => ({
    page: `c${i}`,
    kind: 'cold' as Kind,
    ins: 0,
  }));
  const ring: (Frame | null)[] = policy === 'ring' ? Array.from({ length: ringN }, () => null) : [];
  let ringPos = 0;
  let misses = 0;
  let recycles = 0;
  const recent: boolean[] = [];
  const snaps: Snap[] = [];

  /** LRU: hit moves to head, miss evicts the tail and inserts at the head. */
  const touchLru = (page: string, kind: Kind, t: number) => {
    const i = list.findIndex((f) => f.page === page);
    if (i >= 0) {
      const [f] = list.splice(i, 1);
      list.unshift(f);
      return true;
    }
    if (list.length >= cap) list.pop();
    list.unshift({ page, kind, ins: t });
    return false;
  };

  /** InnoDB: insert at the midpoint; promote out of the old sublist only after oldTicks. */
  const touchMid = (page: string, kind: Kind, t: number) => {
    const n = list.length;
    const mid = n - Math.round((n * OLD_PCT) / 100); // first index of the old sublist
    const i = list.findIndex((f) => f.page === page);
    if (i >= 0) {
      const f = list[i];
      if (i >= mid) {
        if (t - f.ins >= oldTicks) {
          list.splice(i, 1);
          list.unshift(f);
        }
      } else if (i > Math.floor(mid / 4)) {
        list.splice(i, 1);
        list.unshift(f);
      }
      return true;
    }
    if (list.length >= cap) list.pop();
    const n2 = list.length;
    const mid2 = n2 - Math.round((n2 * OLD_PCT) / 100);
    list.splice(mid2, 0, { page, kind, ins: t });
    return false;
  };

  const touchMain = (page: string, kind: Kind, t: number) =>
    policy === 'midpoint' ? touchMid(page, kind, t) : touchLru(page, kind, t);

  /** PostgreSQL: the scan reuses its ring rather than allocating from the pool. */
  const touchRing = (page: string, t: number) => {
    if (list.some((f) => f.page === page)) return true; // already in shared_buffers: just pin it
    const slot = ring.findIndex((f) => f !== null && f.page === page);
    if (slot >= 0) return true;
    if (ring[ringPos] !== null) recycles++;
    ring[ringPos] = { page, kind: 'scan', ins: t };
    ringPos = (ringPos + 1) % ringN;
    return false;
  };

  for (let t = 0; t < TOTAL; t++) {
    const phase = phaseOf(t);

    for (let k = 0; k < POINT_READS; k++) {
      const hit = touchMain(`h${POINT_SEQ[t * POINT_READS + k]}`, 'hot', t);
      recent.push(hit);
      if (recent.length > WINDOW) recent.shift();
      if (!hit) misses++;
    }

    if (phase === 'scan') {
      const base = (t - WARM) * SCAN_PAGES;
      for (let p = 0; p < SCAN_PAGES; p++) {
        const page = `s${base + p}`;
        for (let r = 0; r < SCAN_TOUCHES; r++) {
          const hit = policy === 'ring' ? touchRing(page, t) : touchMain(page, 'scan', t);
          if (!hit) misses++;
        }
      }
    }

    snaps.push({
      list: list.map((f) => ({ ...f })),
      ring: ring.map((f) => (f ? { ...f } : null)),
      ringPos,
      hitPct: recent.length ? (100 * recent.filter(Boolean).length) / recent.length : 0,
      hotRes: list.filter((f) => f.kind === 'hot').length,
      scanFrames:
        list.filter((f) => f.kind === 'scan').length + ring.filter((f) => f !== null).length,
      misses,
      recycles,
      phase,
    });
  }
  return snaps;
}

function summarize(snaps: Snap[]) {
  const endScan = snaps[WARM + SCAN - 1];
  const during = snaps.slice(WARM + 10, WARM + SCAN);
  const after = snaps.slice(WARM + SCAN + 20, WARM + SCAN + TAIL);
  const avg = (a: Snap[]) => (a.length ? a.reduce((s, x) => s + x.hitPct, 0) / a.length : 0);
  return {
    duringPct: avg(during),
    afterPct: avg(after),
    hotRes: endScan.hotRes,
    scanFrames: endScan.scanFrames,
    misses: endScan.misses - snaps[WARM - 1].misses,
  };
}

const POLICIES: { value: Policy; label: string; title: string }[] = [
  { value: 'lru', label: 'Plain LRU', title: 'Every page read goes to the head of one recency list' },
  { value: 'ring', label: 'Ring buffer (PG)', title: 'PostgreSQL BufferAccessStrategy: a small recycled ring' },
  {
    value: 'midpoint',
    label: 'Midpoint (InnoDB)',
    title: 'Insert at the midpoint; promote only after innodb_old_blocks_time',
  },
];

function Cell({
  x,
  y,
  w,
  h,
  frame,
  outline,
}: {
  x: number;
  y: number;
  w: number;
  h: number;
  frame: Frame | null;
  outline?: boolean;
}) {
  const tip = useTip();
  const fill = !frame
    ? 'var(--viz-neutral)'
    : frame.kind === 'hot'
      ? 'var(--viz-1)'
      : frame.kind === 'scan'
        ? 'var(--viz-2)'
        : 'var(--viz-stale)';
  return (
    <g
      {...tip(
        frame ? (
          <>
            <strong>{frame.page}</strong>
            <br />
            {frame.kind === 'hot'
              ? 'hot working-set page'
              : frame.kind === 'scan'
                ? 'scan page'
                : 'cold page, never read again'}
            <br />
            first read at tick {frame.ins}
          </>
        ) : (
          <>empty frame</>
        ),
      )}
    >
      <rect
        x={x}
        y={y}
        width={Math.max(2, w - 1.5)}
        height={h}
        rx={2}
        fill={fill}
        stroke={outline ? 'var(--viz-ink)' : 'var(--viz-border)'}
        strokeWidth={outline ? 2 : 1}
      />
      {frame && frame.kind === 'hot' ? (
        <circle cx={x + (w - 1.5) / 2} cy={y + h / 2} r={Math.min(2.5, w / 5)} fill="var(--viz-surface)" />
      ) : null}
      {frame && frame.kind === 'scan' ? (
        <line
          x1={x + 2}
          y1={y + h - 3}
          x2={x + w - 4}
          y2={y + 3}
          stroke="var(--viz-surface)"
          strokeWidth={1.5}
        />
      ) : null}
    </g>
  );
}

export default function ScanFloodRingLab() {
  const [policy, setPolicy] = useState<Policy>('lru');
  const [ringN, setRingN] = useState(6);
  const [oldMs, setOldMs] = useState<'0' | '1000'>('1000');
  const [t, setT] = useState(WARM + 25);
  const [playing, setPlaying] = useState(false);
  const [ref, width] = useSize(760);
  const acc = useRef(0);

  // A scan's row-level re-touches of one heap page all land inside a single tick, so
  // any non-zero old_blocks_time is modelled as "must survive into a later tick".
  const oldTicks = oldMs === '0' ? 0 : 1;

  const snaps = useMemo(() => simulate(policy, ringN, oldTicks), [policy, ringN, oldTicks]);
  const all = useMemo(
    () => ({
      lru: summarize(simulate('lru', ringN, oldTicks)),
      ring: summarize(simulate('ring', ringN, oldTicks)),
      midpoint: summarize(simulate('midpoint', ringN, oldTicks)),
    }),
    [ringN, oldTicks],
  );

  useTicker((dt) => {
    acc.current += dt;
    const n = Math.floor(acc.current / 45);
    if (n > 0) {
      acc.current -= n * 45;
      setT((v) => Math.min(TOTAL - 1, v + n));
    }
  }, playing);
  useEffect(() => {
    if (t >= TOTAL - 1 && playing) setPlaying(false);
  }, [t, playing]);

  const s = snaps[Math.min(t, TOTAL - 1)];
  const cap = policy === 'ring' ? POOL - ringN : POOL;
  const mid = s.list.length - Math.round((s.list.length * OLD_PCT) / 100);

  const gutter = 92;
  const avail = Math.max(380, Math.min(width, 880));
  const cw = Math.max(9, Math.min(17, (avail - gutter - 44) / POOL));
  // Never let the strip spill outside the svg box: widen instead, and let .viz-figure scroll.
  const W = Math.max(avail, gutter + POOL * cw + 44);
  const stripY = 30;
  const cellH = 26;
  const ringY = stripY + cellH + 34;
  const chartTop = policy === 'ring' ? ringY + cellH + 34 : stripY + cellH + 40;
  const chartH = 104;
  const chartW = Math.max(180, W - gutter - 28);
  const H = chartTop + chartH + 30;

  const xOfTick = (i: number) => gutter + (i / (TOTAL - 1)) * chartW;
  const yOfPct = (p: number) => chartTop + chartH - (p / 100) * chartH;
  const line = snaps
    .slice(0, t + 1)
    .map((sn, i) => `${xOfTick(i).toFixed(1)},${yOfPct(sn.hitPct).toFixed(1)}`)
    .join(' ');

  const narrate = () => {
    if (s.phase === 'warm')
      return (
        <>
          <strong>Warm-up.</strong> Point reads are pulling the {HOT}-page hot set in and pushing the
          leftover cold pages towards the tail. Nothing is competing for frames yet.
        </>
      );
    if (policy === 'lru')
      return s.phase === 'scan' ? (
        <>
          <strong>Sequential flooding.</strong> Every scan page is the most recently used page in the
          pool the instant it is read, so LRU puts it at the head and evicts a hot page from the tail.
          The scan has no reuse at all, and it is winning every eviction decision.
        </>
      ) : (
        <>
          <strong>Cold recovery.</strong> The scan is over and its pages are still holding frames they
          will never be read from again. The hot set has to be faulted back in one miss at a time.
        </>
      );
    if (policy === 'ring')
      return s.phase === 'scan' ? (
        <>
          <strong>The ring absorbs the scan.</strong> {ringN} frames were taken out of the pool for the
          scan's exclusive use and are being recycled — {s.recycles} reuses so far — so the
          {' '}{TABLE_PAGES}-page table never touches the recency list. The hot set is untouched, and so
          is the hit rate.
        </>
      ) : (
        <>
          <strong>Nothing to recover.</strong> The scan never evicted a hot page, so the hit rate never
          dipped. It also left nothing behind: not one of the {TABLE_PAGES} pages it read is cached.
        </>
      );
    return s.phase === 'scan' ? (
      oldTicks === 0 ? (
        <>
          <strong>Midpoint insertion defeated.</strong> With <code>innodb_old_blocks_time = 0</code> the
          scan's second touch of a page — the next row on the same heap page — promotes it straight to
          the head of the young sublist, and the flood is back.
        </>
      ) : (
        <>
          <strong>The old sublist takes the hit.</strong> Scan pages land at the midpoint and every
          re-touch of one falls inside the tick that read it, so none of them is ever promoted. They
          churn through the {Math.round((POOL * OLD_PCT) / 100)} frames below the midpoint and leave the
          young sublist alone.
        </>
      )
    ) : (
      <>
        <strong>After the scan.</strong> Whatever survived above the midpoint is still resident; pages
        that were still in the old sublist when the scan hit are gone.
      </>
    );
  };

  return (
    <VizPanel
      title="Sequential flooding, and the three defenses"
      subtitle={`A ${POOL}-frame pool, full of cold pages at tick 0, serving a ${HOT}-page hot set at ${POINT_READS} point reads per tick. From tick ${WARM} a scan of a ${TABLE_PAGES}-page table pulls ${SCAN_PAGES} pages per tick and touches each one ${SCAN_TOUCHES} times — the rows on a heap page, read inside one tick. Sizes scaled down from the real thing.`}
      controls={
        <>
          <Segmented label="Strategy" value={policy} onChange={setPolicy} options={POLICIES} />
          <Button
            onClick={() => {
              if (t >= TOTAL - 1) setT(0);
              setPlaying((p) => !p);
            }}
            primary
          >
            {playing ? 'Pause' : t >= TOTAL - 1 ? 'Replay' : 'Play'}
          </Button>
          <Slider
            label="Tick"
            min={0}
            max={TOTAL - 1}
            value={t}
            onChange={(v) => {
              setPlaying(false);
              setT(v);
            }}
            format={(v) => `${v} · ${phaseOf(v) === 'scan' ? 'scan running' : phaseOf(v) === 'warm' ? 'warm-up' : 'after scan'}`}
          />
          <Slider
            label="Ring frames"
            min={2}
            max={12}
            value={ringN}
            onChange={setRingN}
            disabled={policy !== 'ring'}
            format={(v) => `${v} (real BAS_BULKREAD = 32)`}
          />
          <Segmented
            label="innodb_old_blocks_time"
            value={oldMs}
            onChange={setOldMs}
            options={[
              { value: '1000', label: '1000 ms' },
              { value: '0', label: '0' },
            ]}
          />
        </>
      }
      legend={
        <Legend
          items={[
            { label: 'Hot working-set page (dot)', color: 'var(--viz-1)' },
            { label: 'Scan page (slash)', color: 'var(--viz-2)' },
            { label: 'Cold page from earlier work', color: 'var(--viz-stale)' },
            { label: 'Empty ring slot', color: 'var(--viz-neutral)' },
            { label: 'Point-read hit rate', color: 'var(--viz-7)', shape: 'line' },
          ]}
        />
      }
      stats={
        <Stats
          items={[
            { label: 'Point-read hit rate', value: `${s.hitPct.toFixed(0)}%`, hint: `trailing ${WINDOW} point reads` },
            { label: 'Hot pages resident', value: `${s.hotRes} / ${HOT}` },
            { label: 'Frames held by the scan', value: `${s.scanFrames} / ${POOL}` },
            {
              label: policy === 'ring' ? 'Ring reuses' : 'Buffer reads (misses)',
              value: policy === 'ring' ? s.recycles : s.misses,
            },
          ]}
        />
      }
      note={<Note>{narrate()}</Note>}
      table={
        <table className="viz-table">
          <thead>
            <tr>
              <th>Strategy</th>
              <th>Hit rate during scan</th>
              <th>Hit rate after scan</th>
              <th>Hot pages resident at scan end</th>
              <th>Frames held by scan</th>
              <th>Misses during scan</th>
            </tr>
          </thead>
          <tbody>
            {POLICIES.map((p) => {
              const r = all[p.value];
              return (
                <tr key={p.value}>
                  <td>{p.label}</td>
                  <td>{r.duringPct.toFixed(0)}%</td>
                  <td>{r.afterPct.toFixed(0)}%</td>
                  <td>
                    {r.hotRes} / {HOT}
                  </td>
                  <td>{r.scanFrames}</td>
                  <td>{r.misses}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      }
    >
      <TooltipHost>
        <div ref={ref}>
          <svg
            width={W}
            height={H}
            role="img"
            aria-label={`Buffer pool at tick ${t}: ${s.hotRes} of ${HOT} hot pages resident, point-read hit rate ${s.hitPct.toFixed(0)} percent`}
          >
            {/* main recency list */}
            <text x={0} y={stripY + cellH / 2 + 4} fill="var(--viz-ink)">
              Buffer pool
            </text>
            <text x={gutter} y={stripY - 8}>
              MRU head
            </text>
            <text x={gutter + cap * cw} y={stripY - 8} textAnchor="end">
              LRU tail →
            </text>
            {Array.from({ length: cap }, (_, i) => (
              <Cell key={i} x={gutter + i * cw} y={stripY} w={cw} h={cellH} frame={s.list[i] ?? null} />
            ))}
            {policy === 'midpoint' ? (
              <g>
                <line
                  x1={gutter + mid * cw - 1}
                  y1={stripY - 4}
                  x2={gutter + mid * cw - 1}
                  y2={stripY + cellH + 6}
                  stroke="var(--viz-ink)"
                  strokeWidth={2}
                />
                <text x={gutter + 2} y={stripY + cellH + 18} fill="var(--viz-ink)">
                  young sublist (63%)
                </text>
                <text x={gutter + mid * cw + 3} y={stripY + cellH + 18} fill="var(--viz-ink)">
                  old sublist — insert here
                </text>
              </g>
            ) : (
              <text x={gutter} y={stripY + cellH + 18}>
                {policy === 'ring'
                  ? `${cap} frames left for everything else; ${ringN} are on loan to the scan`
                  : 'one list, one insertion point: the head'}
              </text>
            )}

            {/* the scan's private ring */}
            {policy === 'ring' ? (
              <g>
                <text x={0} y={ringY + cellH / 2 + 4} fill="var(--viz-ink)">
                  Scan ring
                </text>
                <text x={gutter} y={ringY - 8}>
                  recycled round-robin · next slot outlined · {s.recycles} reuses
                </text>
                {s.ring.map((f, i) => (
                  <Cell
                    key={i}
                    x={gutter + i * (cw + 6)}
                    y={ringY}
                    w={cw + 6}
                    h={cellH}
                    frame={f}
                    outline={i === s.ringPos}
                  />
                ))}
              </g>
            ) : null}

            {/* hit-rate timeline */}
            <text x={0} y={chartTop + chartH / 2 - 4} fill="var(--viz-ink)">
              Point-read
            </text>
            <text x={0} y={chartTop + chartH / 2 + 10} fill="var(--viz-ink)">
              hit rate
            </text>
            <rect
              x={xOfTick(WARM)}
              y={chartTop}
              width={xOfTick(WARM + SCAN) - xOfTick(WARM)}
              height={chartH}
              fill="var(--viz-neutral)"
            />
            <text x={xOfTick(WARM) + 4} y={chartTop + 12}>
              scan running
            </text>
            {[0, 50, 100].map((p) => (
              <g key={p}>
                <line
                  className="viz-grid-line"
                  x1={gutter}
                  y1={yOfPct(p)}
                  x2={gutter + chartW}
                  y2={yOfPct(p)}
                />
                <text x={gutter - 6} y={yOfPct(p) + 4} textAnchor="end">
                  {p}%
                </text>
              </g>
            ))}
            <polyline points={line} fill="none" stroke="var(--viz-7)" strokeWidth={2} />
            <line
              x1={xOfTick(t)}
              y1={chartTop - 4}
              x2={xOfTick(t)}
              y2={chartTop + chartH + 4}
              stroke="var(--viz-ink-muted)"
              strokeWidth={1}
              strokeDasharray="3 3"
            />
            <circle cx={xOfTick(t)} cy={yOfPct(s.hitPct)} r={3.5} fill="var(--viz-7)" />
            <text x={gutter} y={chartTop + chartH + 18}>
              tick 0
            </text>
            <text x={gutter + chartW} y={chartTop + chartH + 18} textAnchor="end">
              tick {TOTAL - 1}
            </text>
          </svg>
        </div>
      </TooltipHost>
    </VizPanel>
  );
}
