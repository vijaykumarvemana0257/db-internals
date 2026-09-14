import { useMemo, useState } from 'react';
import {
  VizPanel,
  Segmented,
  Slider,
  Legend,
  Stats,
  Note,
  TooltipHost,
  useTip,
  makeRng,
  useSize,
  fmtNum,
  fmtTime,
} from './Viz';

/**
 * Hot-page contention under the buffer pool's synchronization layers.
 *
 * Every hit on a resident page touches four shared things, and the model charges for
 * each one the way the hardware does — an atomic read-modify-write is cheap while the
 * cache line sits in one core's cache and expensive once the line is being passed
 * between cores:
 *
 *   map      the page-table shard lock (PostgreSQL BufMappingLock, one of
 *            NUM_BUFFER_PARTITIONS = 128; InnoDB's per-instance page-hash rw-locks).
 *            Shard is a pure function of hash(tag), so one page = one shard.
 *   hdr      the frame descriptor's packed state word: pin on the way in, unpin on the
 *            way out, both compare-and-swap, both retrying when another core wins.
 *   content  the frame's content LWLock: two atomic RMWs on its state word, plus — for
 *            a writer — a genuinely exclusive critical section of HOLD ns.
 *   lru      the per-instance LRU list mutex, taken only by the share of hits the engine
 *            decides to relink (InnoDB's buf_page_make_young_if_needed).
 *
 * Latency for a thread is WORK plus, for each shared resource it uses, that resource's
 * per-op demand multiplied by the number of threads sharing it — the standard closed-system
 * serialization bound. Throughput is the sum of 1/latency, capped by CORES of real work.
 *
 * Optimistic mode is version-based latching (OLFIT / optimistic lock coupling / LeanStore):
 * a reader reads the version word, reads the page, re-reads the version and restarts if it
 * moved. It never writes a shared line, so readers cost nothing in coherence traffic — but
 * writers still take the latch exclusively, and every write raises the readers' restart rate.
 *
 * Everything is a pure function of the controls; the only randomness is which page each
 * thread draws, taken once from makeRng with a fixed seed.
 */

const CORES = 16; // cores available for real work
const WORK = 220; // ns of actual per-op work: find the tuple, compare the key
const HOLD = 900; // ns a writer genuinely holds the content lock exclusively
const RMW_FREE = 25; // ns: atomic RMW on a line already owned by this core
const RMW_HOT = 80; // ns: same instruction once the line is bouncing between cores
const YOUNG = 0.05; // share of hits that relink the LRU list
const MAX_RETRY = 3; // cap on modelled CAS retries per pin
const VERSION_READ = 8; // ns: two plain loads of a version word, no coherence traffic
const MAX_THREADS = 32;

type Mode = 'pessimistic' | 'optimistic';
type Load = 'insert' | 'mixed' | 'read';

const X_SHARE: Record<Load, number> = { insert: 1, mixed: 0.2, read: 0 };

const hashPage = (p: number) => Math.imul(p + 1, 2654435761) >>> 0;

/** Which page each thread is hitting. Skewed, and stable as threads are added. */
function assign(pages: number, threads: number) {
  const rng = makeRng(90210 + pages * 131);
  const w: number[] = [];
  let tot = 0;
  for (let i = 0; i < pages; i++) {
    const x = 1 / Math.sqrt(i + 1);
    w.push(x);
    tot += x;
  }
  const out: number[] = [];
  for (let t = 0; t < threads; t++) {
    let r = rng() * tot;
    let i = 0;
    while (i < pages - 1 && r > w[i]) {
      r -= w[i];
      i++;
    }
    out.push(i);
  }
  return out;
}

type Row = {
  page: number;
  shard: number;
  inst: number;
  kPage: number;
  kShard: number;
  kInst: number;
  work: number;
  map: number;
  hdr: number;
  content: number;
  lru: number;
  total: number;
  retries: number;
  restart: number;
};

type Parts = { work: number; map: number; hdr: number; content: number; lru: number };

type Result = {
  rows: Row[];
  pageLoad: number[];
  shardOf: (p: number) => number;
  instOf: (p: number) => number;
  ops: number;
  avgLat: number;
  meanTotal: number;
  parts: Parts;
  hottest: number;
  retries: number;
  restart: number;
  capped: boolean;
  dominant: keyof Parts;
};

function simulate(
  threads: number,
  pages: number,
  partitions: number,
  instances: number,
  load: Load,
  mode: Mode,
): Result {
  const x = X_SHARE[load];
  const opt = mode === 'optimistic';
  const pageOf = assign(pages, threads);

  const pageLoad = new Array(pages).fill(0) as number[];
  const shardLoad = new Array(partitions).fill(0) as number[];
  const instLoad = new Array(instances).fill(0) as number[];
  const shardOf = (p: number) => hashPage(p) % partitions;
  const instOf = (p: number) => (hashPage(p) >>> 9) % instances;
  for (const p of pageOf) {
    pageLoad[p]++;
    shardLoad[shardOf(p)]++;
    instLoad[instOf(p)]++;
  }

  const rows: Row[] = pageOf.map((p) => {
    const kPage = pageLoad[p];
    const kShard = shardLoad[shardOf(p)];
    const kInst = instLoad[instOf(p)];

    const dMap = 2 * (kShard > 1 ? RMW_HOT : RMW_FREE);
    const dWord = 2 * (kPage > 1 ? RMW_HOT : RMW_FREE);
    const dLru = 2 * (kInst > 1 ? RMW_HOT : RMW_FREE);
    const retries = Math.min(MAX_RETRY, Math.max(0, (kPage - 1) * 0.25));
    const dHdr = dWord * (1 + retries);

    // An optimistic reader touches neither the pin counter nor the latch word.
    const writeShare = opt ? x : 1;
    const restart = opt ? Math.min(0.85, kPage * x * 0.2) : 0;

    const map = kShard * dMap;
    const hdr = kPage * dHdr * writeShare;
    const content = kPage * (dWord * writeShare + x * HOLD);
    const lru = kInst * dLru * YOUNG;
    const work = WORK + (opt ? VERSION_READ + (1 - x) * restart * (WORK + VERSION_READ) : 0);

    return {
      page: p,
      shard: shardOf(p),
      inst: instOf(p),
      kPage,
      kShard,
      kInst,
      work,
      map,
      hdr,
      content,
      lru,
      total: work + map + hdr + content + lru,
      retries: retries * writeShare,
      restart,
    };
  });

  const raw = rows.reduce((s, r) => s + 1e9 / r.total, 0);
  const cap = (CORES / WORK) * 1e9;
  const ops = Math.min(raw, cap);
  const meanTotal = rows.reduce((s, r) => s + r.total, 0) / threads;
  const avgLat = (threads / ops) * 1e9;
  const mean = (f: (r: Row) => number) => rows.reduce((s, r) => s + f(r), 0) / threads;
  const parts = {
    work: mean((r) => r.work),
    map: mean((r) => r.map),
    hdr: mean((r) => r.hdr),
    content: mean((r) => r.content),
    lru: mean((r) => r.lru),
  };
  // What shows up as a *wait* is the excess over what one uncontended thread would pay:
  // a writer's 900 ns critical section is work, not contention, until somebody queues for it.
  const base: Parts = {
    work: parts.work,
    map: 2 * RMW_FREE,
    hdr: 2 * RMW_FREE * (opt ? x : 1),
    content: 2 * RMW_FREE * (opt ? x : 1) + x * HOLD,
    lru: 2 * RMW_FREE * YOUNG,
  };
  const keys = ['map', 'hdr', 'content', 'lru'] as const;
  let dominant: keyof Parts = 'work';
  let worst = 0;
  for (const k of keys) {
    const excess = parts[k] - base[k];
    if (excess > worst) {
      worst = excess;
      dominant = k;
    }
  }
  if (worst < 0.12 * meanTotal) dominant = 'work';

  return {
    rows,
    pageLoad,
    shardOf,
    instOf,
    ops,
    avgLat,
    meanTotal,
    parts,
    hottest: Math.max(...pageLoad),
    retries: mean((r) => r.retries),
    restart: mean((r) => r.restart),
    capped: raw > cap,
    dominant,
  };
}

const WAIT_EVENT: Record<string, string> = {
  content: 'LWLock:BufferContent',
  hdr: 'none — CAS retries burn CPU',
  map: 'LWLock:BufferMapping',
  lru: 'buf_pool LRU_list_mutex',
  work: 'none — CPU-bound',
};

const COLOR = {
  work: 'var(--viz-1)',
  map: 'var(--viz-2)',
  hdr: 'var(--viz-4)',
  content: 'var(--viz-8)',
  lru: 'var(--viz-3)',
} as const;

const SEGMENTS: { key: keyof typeof COLOR; label: string }[] = [
  { key: 'work', label: 'Useful work' },
  { key: 'map', label: 'Page-table shard lock' },
  { key: 'hdr', label: 'Descriptor state word (pin/unpin CAS)' },
  { key: 'content', label: 'Content latch' },
  { key: 'lru', label: 'LRU list mutex' },
];

function PageBox({
  x,
  y,
  w,
  h,
  page,
  shard,
  inst,
  kShard,
  kInst,
  k,
  threads,
}: {
  x: number;
  y: number;
  w: number;
  h: number;
  page: number;
  shard: number;
  inst: number;
  kShard: number;
  kInst: number;
  k: number;
  threads: number;
}) {
  const tip = useTip();
  const share = threads > 0 ? k / threads : 0;
  const fill =
    k === 0
      ? 'var(--viz-neutral)'
      : share > 0.6
        ? 'var(--viz-seq-700)'
        : share > 0.3
          ? 'var(--viz-seq-550)'
          : share > 0.12
            ? 'var(--viz-seq-400)'
            : share > 0.04
              ? 'var(--viz-seq-250)'
              : 'var(--viz-seq-100)';
  return (
    <g
      {...tip(
        <>
          <strong>page {page}</strong>
          <br />
          {k} of {threads} threads on this frame
          <br />
          page-table shard {shard} · pool instance {inst}
          <br />
          {kShard} threads share that shard, {kInst} share that instance
        </>,
      )}
    >
      <rect
        x={x}
        y={y}
        width={w}
        height={h}
        rx={3}
        fill={fill}
        stroke="var(--viz-border)"
        strokeWidth={1}
      />
      {w > 16 ? (
        <text
          x={x + w / 2}
          y={y + h / 2 + 4}
          textAnchor="middle"
          fill={share > 0.3 ? 'var(--viz-surface)' : 'var(--viz-ink)'}
        >
          {k}
        </text>
      ) : null}
    </g>
  );
}

export default function HotPageLatchContentionLab() {
  const [threads, setThreads] = useState(16);
  const [pagesExp, setPagesExp] = useState(0); // 1 page: the pathology
  const [partExp, setPartExp] = useState(7); // 128 BufMappingLocks
  const [instExp, setInstExp] = useState(3); // 8 buffer pool instances
  const [load, setLoad] = useState<Load>('insert');
  const [mode, setMode] = useState<Mode>('pessimistic');
  const [ref, width] = useSize(760);

  const pages = 1 << pagesExp;
  const partitions = 1 << partExp;
  const instances = 1 << instExp;

  const r = useMemo(
    () => simulate(threads, pages, partitions, instances, load, mode),
    [threads, pages, partitions, instances, load, mode],
  );
  const one = useMemo(
    () => simulate(1, pages, partitions, instances, load, mode),
    [pages, partitions, instances, load, mode],
  );
  const curve = useMemo(
    () =>
      Array.from({ length: MAX_THREADS }, (_, i) =>
        simulate(i + 1, pages, partitions, instances, load, mode),
      ),
    [pages, partitions, instances, load, mode],
  );
  const sweep = useMemo(
    () =>
      [0, 1, 2, 3, 4, 5, 6].map((e) => {
        const p = 1 << e;
        return {
          pages: p,
          hottest: simulate(threads, p, partitions, instances, load, mode).hottest,
          at1: simulate(threads, p, 1, instances, load, mode),
          at16: simulate(threads, p, 16, instances, load, mode),
          at128: simulate(threads, p, 128, instances, load, mode),
        };
      }),
    [threads, partitions, instances, load, mode],
  );

  /* -------------------------------------------------------------- layout */
  const avail = Math.max(380, Math.min(width, 880));
  const gutter = 96;
  const W = Math.max(avail, 560);
  const plotW = W - gutter - 16;

  const shown = Math.min(pages, 24);
  const boxW = Math.max(11, Math.min(52, plotW / shown - 4));
  const dotR = 3;
  const dotPitch = dotR * 2 + 2;
  const dotCols = Math.max(1, Math.floor((boxW - 2) / dotPitch));
  const maxDots = dotCols * 6;
  const dotRows = Math.ceil(Math.min(r.hottest, maxDots) / dotCols);
  const dotsH = Math.max(12, dotRows * dotPitch);
  const stripY = 16 + dotsH;
  const boxH = 26;

  const barY = stripY + boxH + 40;
  const barH = 22;
  const refY = barY + barH + 8;
  const refH = 8;

  const chartTop = refY + refH + 40;
  const chartH = 110;
  const H = chartTop + chartH + 34;

  const queue = Math.max(0, r.avgLat - r.meanTotal);
  const scale = plotW / Math.max(r.avgLat, r.meanTotal, 1);

  const maxOps = Math.max(...curve.map((c) => c.ops)) * 1.08 || 1;
  const xOfN = (n: number) => gutter + ((n - 1) / (MAX_THREADS - 1)) * plotW;
  const yOfOps = (o: number) => chartTop + chartH - (o / maxOps) * chartH;
  const line = curve.map((c, i) => `${xOfN(i + 1).toFixed(1)},${yOfOps(c.ops).toFixed(1)}`).join(' ');
  const ideal = Array.from({ length: MAX_THREADS }, (_, i) =>
    `${xOfN(i + 1).toFixed(1)},${yOfOps(Math.min(one.ops * (i + 1), maxOps)).toFixed(1)}`,
  ).join(' ');

  const segs = SEGMENTS.map((s) => ({ ...s, ns: r.parts[s.key] }));

  const narrate = () => {
    if (pages === 1 && mode === 'pessimistic' && load !== 'read')
      return (
        <>
          <strong>One frame, {threads} threads.</strong> Every op pins the same descriptor, so the
          same 64-byte cache line is taken exclusively {2 * threads} times per round trip and the
          content latch is handed round one writer at a time. The shard and instance of a page are
          functions of its tag — all {threads} threads are on shard {r.rows[0].shard} and instance{' '}
          {r.rows[0].inst} — so raising either knob moves nothing. Throughput is flat in threads:
          the curve below stops rising almost immediately.
        </>
      );
    if (pages === 1 && mode === 'optimistic' && load === 'read')
      return (
        <>
          <strong>The read path stops writing.</strong> Version-based latching lets all {threads}{' '}
          threads hold the frame's cache line in shared state at once: no pin, no latch word, no
          coherence traffic. What is left is the page-table lookup — and{' '}
          {WAIT_EVENT[r.dominant]} is now the thing to chase.
        </>
      );
    if (pages === 1 && mode === 'optimistic')
      return (
        <>
          <strong>Optimism does not fix a write-hot page.</strong> Writers still take the latch
          exclusively and still bump the version, so the {HOLD} ns critical section serializes
          exactly as before — and every bump restarts the readers, {(r.restart * 100).toFixed(0)}%
          of them here. A rightmost leaf under monotonic inserts needs a different key, not a
          different latch.
        </>
      );
    if (r.dominant === 'map')
      return (
        <>
          <strong>Now the knob earns its keep.</strong> {threads} threads over {pages} pages, but
          only {partitions} page-table shard{partitions === 1 ? '' : 's'}, so unrelated pages are
          colliding on one lock word: <code>LWLock:BufferMapping</code>. This is the contention{' '}
          <code>NUM_BUFFER_PARTITIONS</code> exists for — push the shards up and it disappears.
        </>
      );
    return (
      <>
        <strong>Spread out.</strong> The busiest frame now carries {r.hottest} of {threads} threads,
        so no single cache line is serializing the system; {((r.parts.work / r.avgLat) * 100).toFixed(0)}%
        of an operation is real work again. Dominant wait: {WAIT_EVENT[r.dominant]}.
      </>
    );
  };

  return (
    <VizPanel
      title="The hot page no partition count can fix"
      subtitle={`${threads} threads hitting resident pages, charged only for synchronization: an atomic read-modify-write costs ~${RMW_FREE} ns on a line this core already owns and ~${RMW_HOT} ns once the line is bouncing between cores, a writer holds the content latch for ${HOLD} ns, and ${WORK} ns of each op is real work. No WAL, no row locks, no I/O — this is the latch layer alone.`}
      controls={
        <>
          <Slider
            label="Threads"
            min={1}
            max={MAX_THREADS}
            value={threads}
            onChange={setThreads}
            format={(v) => `${v}`}
          />
          <Slider
            label="Pages the workload touches"
            min={0}
            max={6}
            value={pagesExp}
            onChange={setPagesExp}
            format={(v) => (v === 0 ? '1 — one hot page' : `${1 << v}`)}
          />
          <Slider
            label="Page-table shards"
            min={0}
            max={7}
            value={partExp}
            onChange={setPartExp}
            format={(v) => `${1 << v}${v === 7 ? ' (PG default)' : ''}`}
          />
          <Slider
            label="Buffer pool instances"
            min={0}
            max={4}
            value={instExp}
            onChange={setInstExp}
            format={(v) => `${1 << v}${v === 3 ? ' (InnoDB default)' : ''}`}
          />
          <Segmented
            label="Workload"
            value={load}
            onChange={setLoad}
            options={[
              { value: 'insert', label: 'Monotonic insert', title: 'Every op takes the content latch exclusively' },
              { value: 'mixed', label: 'Mixed 20% write', title: 'One op in five is a writer' },
              { value: 'read', label: 'Point read', title: 'Shared mode only' },
            ]}
          />
          <Segmented
            label="Latching"
            value={mode}
            onChange={setMode}
            options={[
              { value: 'pessimistic', label: 'Pin + LWLock', title: 'PostgreSQL / InnoDB: every reader writes shared state' },
              { value: 'optimistic', label: 'Version / optimistic', title: 'OLFIT, optimistic lock coupling, LeanStore: readers validate a version word' },
            ]}
          />
        </>
      }
      legend={
        <Legend
          items={[
            ...SEGMENTS.map((s) => ({ label: s.label, color: COLOR[s.key] })),
            { label: 'Waiting for one of 16 cores', color: 'var(--viz-neutral)' },
            { label: 'Throughput vs threads', color: 'var(--viz-7)', shape: 'line' as const },
            { label: 'Linear scaling from one thread', color: 'var(--viz-ink-muted)', shape: 'line' as const },
          ]}
        />
      }
      stats={
        <Stats
          items={[
            { label: 'Throughput', value: `${fmtNum(r.ops / 1000)}k ops/s` },
            { label: 'Latency per op', value: fmtTime(r.avgLat) },
            { label: 'Threads on hottest frame', value: `${r.hottest} / ${threads}` },
            {
              label: mode === 'optimistic' ? 'Reader restarts' : 'CAS retries per pin',
              value:
                mode === 'optimistic'
                  ? load === 'insert'
                    ? 'no readers'
                    : `${(r.restart * 100).toFixed(0)}%`
                  : r.retries.toFixed(2),
              hint:
                mode === 'optimistic'
                  ? 'Share of optimistic reads that saw the version move and had to redo the read'
                  : 'Failed compare-and-swaps per pin, from other cores winning the line',
            },
            { label: 'Dominant wait', value: WAIT_EVENT[r.dominant] },
          ]}
        />
      }
      note={<Note>{narrate()}</Note>}
      table={
        <table className="viz-table">
          <caption>
            {threads} threads, {load === 'insert' ? 'monotonic insert' : load === 'mixed' ? '20% writes' : 'point reads'},{' '}
            {mode === 'optimistic' ? 'optimistic latching' : 'pin + LWLock'}. Throughput in thousands
            of ops/s.
          </caption>
          <thead>
            <tr>
              <th>Pages touched</th>
              <th>Threads on hottest frame</th>
              <th>1 shard</th>
              <th>16 shards</th>
              <th>128 shards</th>
              <th>Dominant wait at 128 shards</th>
            </tr>
          </thead>
          <tbody>
            {sweep.map((s) => (
              <tr key={s.pages}>
                <td>{s.pages}</td>
                <td>{s.hottest}</td>
                <td>{fmtNum(s.at1.ops / 1000)}</td>
                <td>{fmtNum(s.at16.ops / 1000)}</td>
                <td>{fmtNum(s.at128.ops / 1000)}</td>
                <td>{WAIT_EVENT[s.at128.dominant]}</td>
              </tr>
            ))}
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
            aria-label={`${threads} threads over ${pages} pages: ${r.hottest} threads on the hottest frame, ${fmtNum(r.ops / 1000)} thousand ops per second, dominant wait ${WAIT_EVENT[r.dominant]}`}
          >
            {/* ---- pages and the threads queued on them ---- */}
            <text x={0} y={stripY + boxH / 2 + 4} fill="var(--viz-ink)">
              Frames
            </text>
            <text x={gutter} y={12}>
              one dot = one thread
            </text>
            {Array.from({ length: shown }, (_, i) => {
              const k = r.pageLoad[i] ?? 0;
              const row = r.rows.find((q) => q.page === i);
              const bx = gutter + i * (boxW + 4);
              return (
                <g key={i}>
                  {Array.from({ length: Math.min(k, maxDots) }, (_, d) => (
                    <circle
                      key={d}
                      cx={bx + 2 + (d % dotCols) * dotPitch + dotR}
                      cy={stripY - 6 - Math.floor(d / dotCols) * dotPitch}
                      r={dotR}
                      fill="var(--viz-7)"
                    />
                  ))}
                  <PageBox
                    x={bx}
                    y={stripY}
                    w={boxW}
                    h={boxH}
                    page={i}
                    shard={r.shardOf(i)}
                    inst={r.instOf(i)}
                    kShard={row ? row.kShard : 0}
                    kInst={row ? row.kInst : 0}
                    k={k}
                    threads={threads}
                  />
                </g>
              );
            })}
            {pages > shown ? (
              <text x={gutter + shown * (boxW + 4) + 4} y={stripY + boxH / 2 + 4}>
                +{pages - shown} more
              </text>
            ) : null}
            <text x={gutter} y={stripY + boxH + 16}>
              {pages === 1
                ? `shard ${r.rows[0].shard} of ${partitions} · instance ${r.rows[0].inst} of ${instances} — both are functions of the page id, so every thread lands on the same ones`
                : `${pages} pages over ${partitions} shards and ${instances} instances`}
            </text>

            {/* ---- latency breakdown ---- */}
            <text x={0} y={barY + barH / 2 + 4} fill="var(--viz-ink)">
              One op
            </text>
            {(() => {
              let cx = gutter;
              const out = segs.map((s) => {
                const w = s.ns * scale;
                const el = (
                  <g key={s.key}>
                    <rect
                      x={cx}
                      y={barY}
                      width={Math.max(0, w)}
                      height={barH}
                      fill={COLOR[s.key]}
                      stroke="var(--viz-border)"
                    />
                    {w > 58 ? (
                      <text x={cx + 5} y={barY + barH / 2 + 4} fill="var(--viz-surface)">
                        {fmtTime(s.ns)}
                      </text>
                    ) : null}
                  </g>
                );
                cx += Math.max(0, w);
                return el;
              });
              if (queue > 1)
                out.push(
                  <rect
                    key="q"
                    x={cx}
                    y={barY}
                    width={queue * scale}
                    height={barH}
                    fill="var(--viz-neutral)"
                    stroke="var(--viz-border)"
                  />,
                );
              return out;
            })()}
            <text x={gutter + plotW} y={barY - 6} textAnchor="end" fill="var(--viz-ink)">
              {fmtTime(r.avgLat)} per op
            </text>

            <text x={0} y={refY + refH + 4} fill="var(--viz-ink)">
              1 thread
            </text>
            {(() => {
              let cx = gutter;
              return SEGMENTS.map((s) => {
                const w = one.parts[s.key] * scale;
                const el = (
                  <rect
                    key={s.key}
                    x={cx}
                    y={refY}
                    width={Math.max(0, w)}
                    height={refH}
                    fill={COLOR[s.key]}
                    opacity={0.55}
                  />
                );
                cx += Math.max(0, w);
                return el;
              });
            })()}
            <text x={gutter + one.avgLat * scale + 6} y={refY + refH + 4}>
              {fmtTime(one.avgLat)} uncontended
            </text>

            {/* ---- throughput vs threads ---- */}
            <text x={0} y={chartTop + chartH / 2 - 4} fill="var(--viz-ink)">
              Throughput
            </text>
            <text x={0} y={chartTop + chartH / 2 + 10} fill="var(--viz-ink)">
              vs threads
            </text>
            {[0, 0.5, 1].map((f) => (
              <g key={f}>
                <line
                  className="viz-grid-line"
                  x1={gutter}
                  y1={yOfOps(maxOps * f)}
                  x2={gutter + plotW}
                  y2={yOfOps(maxOps * f)}
                />
                <text x={gutter - 6} y={yOfOps(maxOps * f) + 4} textAnchor="end">
                  {fmtNum((maxOps * f) / 1000)}k
                </text>
              </g>
            ))}
            <polyline
              points={ideal}
              fill="none"
              stroke="var(--viz-ink-muted)"
              strokeWidth={1.5}
              strokeDasharray="4 3"
            />
            <polyline points={line} fill="none" stroke="var(--viz-7)" strokeWidth={2} />
            <circle cx={xOfN(threads)} cy={yOfOps(r.ops)} r={4} fill="var(--viz-7)" />
            <line
              x1={xOfN(threads)}
              y1={chartTop - 4}
              x2={xOfN(threads)}
              y2={chartTop + chartH + 4}
              stroke="var(--viz-ink-muted)"
              strokeWidth={1}
              strokeDasharray="3 3"
            />
            <text x={gutter} y={chartTop + chartH + 18}>
              1 thread
            </text>
            <text x={gutter + plotW} y={chartTop + chartH + 18} textAnchor="end">
              {MAX_THREADS} threads
            </text>
          </svg>
        </div>
      </TooltipHost>
    </VizPanel>
  );
}
