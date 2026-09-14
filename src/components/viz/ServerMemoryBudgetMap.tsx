import { useMemo, useState } from 'react';
import {
  VizPanel,
  Slider,
  Choice,
  Check,
  Legend,
  Stats,
  Note,
  TooltipHost,
  useTip,
  fmtBytes,
  fmtNum,
  useSize,
} from './Viz';

/**
 * One server's RAM, partitioned the way a database process actually partitions it.
 *
 * The point of the model is that the configuration file describes an *upper bound*, not a
 * reservation: the shared segment is allocated once at startup and is real, while the
 * per-connection column is a product — grants x nodes x workers x active connections —
 * that nothing in the engine enforces. Push the product past physical RAM and the bar
 * overflows the dashed line instead of erroring.
 *
 * The page-table column is the other half: PostgreSQL maps the shared segment into every
 * backend process, so its PTEs are paid per backend; mysqld is one process with threads and
 * pays them once. Huge pages divide both by 512.
 */

const KB = 1024;
const MB = 1024 * 1024;
const GB = 1024 * 1024 * 1024;

const RAM_STEPS = [8, 16, 32, 64, 128, 256, 512];
const GRANT_STEPS = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512];
const MAINT_STEPS = [64, 128, 256, 512, 1024, 2048];

/** A modern x86 core's second-level dTLB is in the 1.5k-3k entry range; 2048 is representative. */
const L2_TLB_ENTRIES = 2048;
const BASE_PAGE = 4 * KB;
const HUGE_PAGE = 2 * MB;
const KERNEL_FLOOR = 1 * GB;

type EngineId = 'pg' | 'innodb';

type Cfg = {
  engine: EngineId;
  ramIdx: number;
  poolPct: number;
  conns: number;
  activePct: number;
  grantIdx: number;
  nodes: number;
  workers: number;
  hashMult10: number;
  maintIdx: number;
  huge: boolean;
};

const DEFAULTS: Cfg = {
  engine: 'pg',
  ramIdx: 3, // 64 GB
  poolPct: 25,
  conns: 200,
  activePct: 20,
  grantIdx: 2, // 4 MB
  nodes: 2,
  workers: 2,
  hashMult10: 20,
  maintIdx: 0, // 64 MB
  huge: false,
};

const PRESETS: { value: string; label: string; cfg: Partial<Cfg> }[] = [
  { value: 'stock', label: 'Stock defaults, 64 GB box', cfg: {} },
  {
    value: 'tuned',
    label: 'Tuned OLTP (pooled connections)',
    cfg: { poolPct: 25, conns: 120, activePct: 25, grantIdx: 4, nodes: 2, workers: 2, huge: true },
  },
  {
    value: 'oom',
    label: 'The config that OOMs',
    cfg: { poolPct: 40, conns: 800, activePct: 60, grantIdx: 7, nodes: 4, workers: 4, huge: false },
  },
  {
    value: 'dw',
    label: 'Analytics box: few sessions, huge grants',
    cfg: { ramIdx: 4, poolPct: 20, conns: 40, activePct: 75, grantIdx: 8, nodes: 5, workers: 6, maintIdx: 3, huge: true },
  },
];

/* ------------------------------------------------------------------- model */

type GroupId = 'shared' | 'backend' | 'pt' | 'free' | 'over';

type Slice = {
  key: string;
  group: GroupId;
  label: string;
  knob: string;
  bytes: number;
  color: string;
  dashed?: boolean;
  hint: string;
};

type Layout = {
  ram: number;
  pool: number;
  slices: Slice[];
  sharedTotal: number;
  backendTotal: number;
  ptBytes: number;
  ptPages: number;
  ptCopies: number;
  peakPerBackend: number;
  baselinePerBackend: number;
  grantPerQuery: number;
  active: number;
  free: number;
  over: number;
};

const C_POOL = 'var(--viz-1)';
const C_BOOK = 'var(--viz-2)';
const C_LOG = 'var(--viz-3)';
const C_LOCK = 'var(--viz-4)';
const C_META = 'var(--viz-5)';
const C_BASE = 'var(--viz-6)';
const C_GRANT = 'var(--viz-7)';
const C_PT = 'var(--viz-8)';
const C_FREE = 'var(--viz-neutral)';
const C_OVER = 'var(--viz-critical)';

function clamp(n: number, lo: number, hi: number) {
  return Math.min(hi, Math.max(lo, n));
}

function build(cfg: Cfg): Layout {
  const ram = RAM_STEPS[cfg.ramIdx] * GB;
  const pool = (ram * cfg.poolPct) / 100;
  const grant = GRANT_STEPS[cfg.grantIdx] * MB;
  const maint = MAINT_STEPS[cfg.maintIdx] * MB;
  const active = Math.max(1, Math.round((cfg.conns * cfg.activePct) / 100));
  const pg = cfg.engine === 'pg';

  const slices: Slice[] = [];

  if (pg) {
    const nbuf = Math.floor(pool / (8 * KB));
    const wal = clamp(pool / 32, 64 * KB, 16 * MB);
    const locks = 64 * cfg.conns * 300;
    const predLocks = 64 * cfg.conns * 180;
    const slru = (clamp(Math.floor(nbuf / 512), 16, 1024) * 2 + 96) * 8 * KB;
    const procs = cfg.conns * 8 * KB;

    slices.push(
      {
        key: 'pool',
        group: 'shared',
        label: 'Page pool',
        knob: 'shared_buffers',
        bytes: pool,
        color: C_POOL,
        hint: `${fmtNum(nbuf)} frames of 8 KB. Allocated in full at startup and touched lazily.`,
      },
      {
        key: 'book',
        group: 'shared',
        label: 'Pool bookkeeping',
        knob: 'derived from shared_buffers',
        bytes: nbuf * 132,
        color: C_BOOK,
        hint: 'BufferDesc (64 B, cache-line padded), the buffer-mapping hash entry and the checkpointer’s sort array — on the order of 130 B per frame, paid on top of shared_buffers.',
      },
      {
        key: 'log',
        group: 'shared',
        label: 'WAL buffers',
        knob: 'wal_buffers = -1',
        bytes: wal,
        color: C_LOG,
        hint: 'Auto-sized to shared_buffers/32, clamped to [64 kB, one 16 MB WAL segment]. A ring, not a cache.',
      },
      {
        key: 'lock',
        group: 'shared',
        label: 'Lock + predicate tables',
        knob: 'max_locks_per_transaction, max_pred_locks_per_transaction',
        bytes: locks + predLocks,
        color: C_LOCK,
        hint: 'Sized at startup for max_locks_per_transaction × (max_connections + max_prepared_transactions) lockable objects. Exhaust it and you get "out of shared memory", never a wait.',
      },
      {
        key: 'meta',
        group: 'shared',
        label: 'SLRU + proc array',
        knob: 'transaction_buffers, subtransaction_buffers, …',
        bytes: slru + procs,
        color: C_META,
        hint: 'pg_xact / pg_subtrans / pg_multixact / pg_commit_ts / pg_serial / pg_notify ring buffers, plus PGPROC, fast-path lock slots and the sinval queue for every connection slot.',
      },
    );
  } else {
    const control = pool * 0.1;
    const logBuf = 16 * MB;
    const psMem = 100 * MB + cfg.conns * 192 * KB;

    slices.push(
      {
        key: 'pool',
        group: 'shared',
        label: 'Page pool',
        knob: 'innodb_buffer_pool_size',
        bytes: pool,
        color: C_POOL,
        hint: `${fmtNum(Math.floor(pool / (16 * KB)))} frames of 16 KB, split across innodb_buffer_pool_instances and allocated in innodb_buffer_pool_chunk_size units (128 MB by default).`,
      },
      {
        key: 'book',
        group: 'shared',
        label: 'Control structures',
        knob: '~10% above the configured pool',
        bytes: control,
        color: C_BOOK,
        hint: 'Block descriptors, the page hash, the adaptive hash index and the LRU/flush lists. The manual warns the real allocation runs about 10% over the configured size.',
      },
      {
        key: 'log',
        group: 'shared',
        label: 'Log buffer',
        knob: 'innodb_log_buffer_size = 16M',
        bytes: logBuf,
        color: C_LOG,
        hint: 'Redo records accumulate here before the log writer flushes them. Too small and large transactions force mid-statement flushes.',
      },
      {
        key: 'lock',
        group: 'shared',
        label: 'TempTable pool',
        knob: 'temptable_max_ram',
        bytes: maint,
        color: C_LOCK,
        hint: 'A GLOBAL budget for internal temporary tables — 1 GB by default — unlike per-session tmp_table_size. Past it the TempTable engine falls back to on-disk internal temporary tables in InnoDB.',
      },
      {
        key: 'meta',
        group: 'shared',
        label: 'P_S + dictionary caches',
        knob: 'performance_schema autosized',
        bytes: psMem,
        color: C_META,
        hint: 'Performance Schema tables autosize from max_connections and table_open_cache; sys.memory_global_total reports what was actually taken.',
      },
    );
  }

  const baselineEach = pg ? 8 * MB : 1 * MB;
  const tempEach = pg ? 8 * MB : 0;
  const grantEach = pg ? cfg.nodes * grant * (cfg.hashMult10 / 10) * (1 + cfg.workers) : cfg.nodes * grant;
  const maintTotal = pg ? 3 * maint : 0;

  slices.push({
    key: 'base',
    group: 'backend',
    label: pg ? 'Backend baseline' : 'Thread baseline',
    knob: pg ? 'private stacks, relcache/syscache, temp_buffers' : 'thread_stack, net buffers',
    bytes: cfg.conns * baselineEach + active * tempEach,
    color: C_BASE,
    hint: pg
      ? 'Every backend is a process: its own stack, its own copy of the catalog caches (which grow with relations and partitions and are never trimmed), and up to temp_buffers of private temp-table pages held to session end.'
      : 'Every connection is a thread inside one process: a stack plus its network buffers. The catalog is shared, so the fixed per-connection cost is far lower than a process-per-connection engine’s.',
  });

  slices.push({
    key: 'grant',
    group: 'backend',
    label: pg ? 'Query grants' : 'Per-connection buffers',
    knob: pg ? 'work_mem × hash_mem_multiplier × nodes × (1 + workers)' : 'sort_buffer_size, join_buffer_size, read_rnd_buffer_size',
    bytes: active * grantEach + maintTotal,
    color: C_GRANT,
    hint: pg
      ? `${fmtBytes(grantEach)} per active backend at this plan shape, × ${fmtNum(active)} active backends, plus ${fmtNum(3)} autovacuum workers at maintenance_work_mem. Nothing in the server checks this sum.`
      : `${fmtBytes(grantEach)} per active connection: each buffer is allocated per join or per sort, not per connection, and none of them are counted against a global budget unless global_connection_memory_limit is set.`,
  });

  const sharedTotal = slices.filter((s) => s.group === 'shared').reduce((a, s) => a + s.bytes, 0);
  const backendTotal = slices.filter((s) => s.group === 'backend').reduce((a, s) => a + s.bytes, 0);

  const pageSize = cfg.huge ? HUGE_PAGE : BASE_PAGE;
  const ptPages = Math.ceil(sharedTotal / pageSize);
  const ptCopies = pg ? cfg.conns : 1;
  const ptBytes = ptPages * 8 * ptCopies;

  slices.push({
    key: 'pt',
    group: 'pt',
    label: 'Page tables',
    knob: cfg.huge ? 'huge_pages = on (2 MB)' : '4 kB pages',
    bytes: ptBytes,
    color: C_PT,
    hint: pg
      ? `${fmtNum(ptPages)} PTEs × 8 B to map the shared segment, in each of ${fmtNum(cfg.conns)} backend processes — an upper bound reached when every backend has touched the whole pool.`
      : `${fmtNum(ptPages)} PTEs × 8 B, once: mysqld is a single process, so threads share one page table for the buffer pool.`,
  });

  const used = sharedTotal + backendTotal + ptBytes + KERNEL_FLOOR;
  const free = ram - used;

  slices.push({
    key: 'kernel',
    group: 'free',
    label: 'Kernel + agents',
    knob: 'not yours',
    bytes: KERNEL_FLOOR,
    color: C_FREE,
    dashed: true,
    hint: 'Kernel slab, network buffers, the monitoring agent, the backup process. Budget a floor for it or the OOM killer will find it for you.',
  });

  if (free > 0) {
    slices.push({
      key: 'free',
      group: 'free',
      label: 'OS page cache',
      knob: 'what is left over',
      bytes: free,
      color: C_FREE,
      hint: 'Everything not claimed by the process ends up caching database files. This is the second tier of the cache, and the reason effective_cache_size is set to pool + this, not to shared_buffers.',
    });
  } else if (free < 0) {
    slices.push({
      key: 'over',
      group: 'over',
      label: 'Over physical RAM',
      knob: 'swap, then the OOM killer',
      bytes: -free,
      color: C_OVER,
      hint: 'The worst case does not fit. Nothing fails at configuration time: backends palloc their way past the line and the kernel starts reclaiming, swapping, or killing the process with the largest RSS.',
    });
  }

  return {
    ram,
    pool,
    slices,
    sharedTotal,
    backendTotal,
    ptBytes,
    ptPages,
    ptCopies,
    peakPerBackend: baselineEach + tempEach + grantEach,
    baselinePerBackend: baselineEach,
    grantPerQuery: grantEach,
    active,
    free: Math.max(0, free),
    over: Math.max(0, -free),
  };
}


/* ------------------------------------------------------------------ render */

const COLUMNS: { id: GroupId; title: string }[] = [
  { id: 'shared', title: 'Shared segment' },
  { id: 'backend', title: 'Per-connection' },
  { id: 'pt', title: 'Page tables' },
  { id: 'free', title: 'Unallocated' },
  { id: 'over', title: 'Over budget' },
];

type TlbRow = {
  size: number;
  label: string;
  entries: number;
  reach: number;
  covered: number;
  ptPerProc: number;
  active: boolean;
};

type Col = { id: GroupId; title: string; slices: Slice[]; x: number; w: number; bytes: number };

type FigProps = {
  L: Layout;
  cols: Col[];
  tlbRows: TlbRow[];
  svgW: number;
  height: number;
  treeTop: number;
  treeH: number;
  tlbTop: number;
  barX0: number;
  barW: number;
  logX: (n: number) => number;
  ramX: number | null;
};

/** The figure lives below TooltipHost so that useTip() finds the provider. */
function MemoryFigure({ L, cols, tlbRows, svgW, height, treeTop, treeH, tlbTop, barX0, barW, logX, ramX }: FigProps) {
  const tip = useTip();
  const pct = (b: number) => `${((b / L.ram) * 100).toFixed(1)}%`;

  return (
    <svg
      width={svgW}
      height={height}
      role="img"
      aria-label="Treemap of one database server's RAM split into the shared segment, per-connection allocations, page tables and unallocated memory, with a page-table and TLB reach comparison below"
    >
      {cols.map((c) => {
        let y = treeTop;
        return (
          <g key={c.id}>
            <text x={c.x} y={treeTop - 10} fill="var(--viz-ink-2)" fontSize={12}>
              {c.title} · {fmtBytes(c.bytes)}
            </text>
            {c.slices.map((s) => {
              const h = Math.max(4, (s.bytes / c.bytes) * treeH);
              const yy = y;
              y += h;
              return (
                <g
                  key={s.key}
                  {...tip(
                    <>
                      <strong>
                        {s.label} — {fmtBytes(s.bytes)} ({pct(s.bytes)})
                      </strong>
                      <br />
                      <code>{s.knob}</code>
                      <br />
                      {s.hint}
                    </>,
                  )}
                >
                  <rect
                    x={c.x}
                    y={yy}
                    width={c.w}
                    height={Math.max(2, h - 1)}
                    rx={3}
                    fill={s.color}
                    fillOpacity={s.group === 'free' ? 0.55 : 1}
                    stroke="var(--viz-surface)"
                    strokeWidth={1}
                    strokeDasharray={s.dashed ? '3 3' : undefined}
                  />
                  {h >= 16 && c.w >= 76 ? (
                    <text x={c.x + 6} y={yy + 14} fontSize={11} fill="var(--viz-surface)" style={{ pointerEvents: 'none' }}>
                      {s.label}
                    </text>
                  ) : null}
                  {h >= 32 && c.w >= 76 ? (
                    <text
                      x={c.x + 6}
                      y={yy + 28}
                      fontSize={11}
                      fill="var(--viz-surface)"
                      fillOpacity={0.85}
                      style={{ pointerEvents: 'none' }}
                    >
                      {fmtBytes(s.bytes)}
                    </text>
                  ) : null}
                </g>
              );
            })}
          </g>
        );
      })}

      {ramX !== null ? (
        <>
          <line
            x1={ramX}
            y1={treeTop - 22}
            x2={ramX}
            y2={treeTop + treeH + 8}
            stroke="var(--viz-critical)"
            strokeWidth={2}
            strokeDasharray="5 4"
          />
          <text x={ramX - 6} y={treeTop + treeH + 22} textAnchor="end" fontSize={12} fill="var(--viz-critical)">
            physical RAM ends here
          </text>
        </>
      ) : (
        <text x={1} y={treeTop + treeH + 22} fontSize={12} fill="var(--viz-ink-muted)">
          Full width = {fmtBytes(L.ram)} of physical RAM
        </text>
      )}

      <text x={1} y={tlbTop - 12} fill="var(--viz-ink)" fontSize={12} fontWeight={600}>
        Translating the shared segment: page-table entries needed, log scale
      </text>
      {tlbRows.map((r, i) => {
        const y = tlbTop + i * 40;
        const w = Math.max(2, logX(r.entries) * barW);
        return (
          <g
            key={r.label}
            {...tip(
              <>
                <strong>{r.label}</strong>
                <br />
                {fmtNum(r.entries)} PTEs × 8 B = {fmtBytes(r.ptPerProc)} per process, × {fmtNum(L.ptCopies)} ={' '}
                {fmtBytes(r.ptPerProc * L.ptCopies)} in total.
                <br />A {fmtNum(L2_TLB_ENTRIES)}-entry L2 dTLB reaches {fmtBytes(r.reach)} — {(r.covered * 100).toFixed(2)}% of the
                shared segment.
              </>,
            )}
          >
            <text x={1} y={y + 15} fontSize={12} fill={r.active ? 'var(--viz-ink)' : 'var(--viz-ink-muted)'}>
              {r.active ? '▶ ' : ''}
              {r.label}
            </text>
            <rect
              x={barX0}
              y={y}
              width={w}
              height={20}
              rx={3}
              fill={C_PT}
              fillOpacity={r.active ? 1 : 0.35}
              stroke="var(--viz-surface)"
            />
            <text x={barX0 + w + 8} y={y + 15} fontSize={12} fill="var(--viz-ink-2)">
              {fmtNum(r.entries)} PTEs · {fmtBytes(r.ptPerProc * L.ptCopies)} · covers {(r.covered * 100).toFixed(2)}%
            </text>
          </g>
        );
      })}
      <line
        x1={barX0 + logX(L2_TLB_ENTRIES) * barW}
        y1={tlbTop - 6}
        x2={barX0 + logX(L2_TLB_ENTRIES) * barW}
        y2={tlbTop + 82}
        stroke="var(--viz-ink-2)"
        strokeWidth={1.5}
        strokeDasharray="4 3"
      />
      <text x={barX0 + logX(L2_TLB_ENTRIES) * barW + 5} y={tlbTop + 78} fontSize={11} fill="var(--viz-ink-2)">
        L2 dTLB ≈ {fmtNum(L2_TLB_ENTRIES)} entries
      </text>
    </svg>
  );
}

export default function ServerMemoryBudgetMap() {
  const [cfg, setCfg] = useState<Cfg>(DEFAULTS);
  const [preset, setPreset] = useState('stock');
  const [ref, width] = useSize(880);

  const set = <K extends keyof Cfg>(k: K, v: Cfg[K]) => setCfg((c) => ({ ...c, [k]: v }));
  const applyPreset = (p: string) => {
    setPreset(p);
    const found = PRESETS.find((x) => x.value === p);
    if (found) setCfg((c) => ({ ...DEFAULTS, engine: c.engine, ...found.cfg }));
  };

  const L = useMemo(() => build(cfg), [cfg]);
  const pg = cfg.engine === 'pg';

  const svgW = Math.max(width, 720);
  const treeTop = 28;
  const treeH = 250;
  const tlbTop = treeTop + treeH + 54;
  const height = tlbTop + 92;

  const grouped = COLUMNS.map((c) => ({ ...c, slices: L.slices.filter((s) => s.group === c.id) })).filter(
    (c) => c.slices.length > 0,
  );

  const total = grouped.reduce((a, c) => a + c.slices.reduce((b, s) => b + s.bytes, 0), 0);
  const span = Math.max(total, L.ram);
  const gap = 6;
  const usableW = svgW - gap * (grouped.length - 1) - 2;
  const scale = usableW / span;

  let cursor = 1;
  const cols: Col[] = grouped.map((c) => {
    const bytes = c.slices.reduce((a, s) => a + s.bytes, 0);
    const w = Math.max(7, bytes * scale);
    const col = { ...c, x: cursor, w, bytes };
    cursor += w + gap;
    return col;
  });

  /* The over-budget column starts exactly where physical RAM runs out. */
  const overCol = cols.find((c) => c.id === 'over');
  const ramX = overCol ? overCol.x - gap / 2 : null;

  const tlbRows: TlbRow[] = [
    { size: BASE_PAGE, label: '4 kB pages' },
    { size: HUGE_PAGE, label: '2 MB huge pages' },
  ].map((r) => {
    const entries = Math.ceil(L.sharedTotal / r.size);
    const reach = L2_TLB_ENTRIES * r.size;
    return {
      ...r,
      entries,
      reach,
      covered: Math.min(1, reach / Math.max(1, L.sharedTotal)),
      ptPerProc: entries * 8,
      active: (r.size === HUGE_PAGE) === cfg.huge,
    };
  });
  const maxEntries = Math.max(...tlbRows.map((r) => r.entries), L2_TLB_ENTRIES * 4);
  const logX = (n: number) => Math.log10(1 + n) / Math.log10(1 + maxEntries);
  const barX0 = 140;
  const barW = Math.max(160, svgW - barX0 - 300);

  const pct = (b: number) => `${((b / L.ram) * 100).toFixed(1)}%`;
  const tlbNow = tlbRows.find((r) => r.active)!;

  const narration = (() => {
    if (L.over > 0) {
      return (
        <>
          <strong>The worst case overshoots physical RAM by {fmtBytes(L.over)}.</strong> Nothing
          rejects this configuration. The shared segment fits; the overshoot is a <em>product</em> —{' '}
          {fmtBytes(L.grantPerQuery)} of grants in each of {fmtNum(L.active)} active{' '}
          {pg ? 'backends' : 'connections'} — and it is only reached when enough sessions pick the
          same shape of plan at the same moment. That is why this failure arrives as a 3 a.m.
          incident rather than a startup error.
        </>
      );
    }
    if (L.free < L.ram * 0.1) {
      return (
        <>
          <strong>Only {fmtBytes(L.free)} is left for the OS page cache</strong> ({pct(L.free)} of
          RAM). The process fits, but the second tier of the cache is gone: every miss in the{' '}
          {fmtBytes(L.pool)} pool now goes to the device, and a cold restart has nothing warm to
          read from.
        </>
      );
    }
    return (
      <>
        <strong>
          {fmtBytes(L.sharedTotal)} shared, {fmtBytes(L.backendTotal)} of worst-case private memory,{' '}
          {fmtBytes(L.ptBytes)} of page tables.
        </strong>{' '}
        The pool plus the {fmtBytes(L.free)} of page cache is roughly what effective_cache_size
        should describe — a number that allocates nothing and only moves plan costs.{' '}
        {tlbNow.covered < 0.05
          ? `At ${tlbNow.label} the L2 dTLB covers ${(tlbNow.covered * 100).toFixed(2)}% of the shared segment, so random page access pays a page walk almost every time.`
          : `At ${tlbNow.label} the L2 dTLB covers ${(tlbNow.covered * 100).toFixed(2)}% of the shared segment.`}
      </>
    );
  })();

  return (
    <VizPanel
      title="Where the server's RAM actually goes"
      subtitle="Size the shared segment, then set the per-connection product. Full width is physical RAM — and nothing in the configuration file checks that the parts fit inside it."
      controls={
        <>
          <Choice
            label="Engine"
            value={cfg.engine}
            onChange={(v) => set('engine', v as EngineId)}
            options={[
              { value: 'pg', label: 'PostgreSQL — process per connection' },
              { value: 'innodb', label: 'MySQL/InnoDB — thread per connection' },
            ]}
          />
          <Choice
            label="Preset"
            value={preset}
            onChange={applyPreset}
            options={PRESETS.map((p) => ({ value: p.value, label: p.label }))}
          />
          <Slider
            label="Physical RAM"
            min={0}
            max={RAM_STEPS.length - 1}
            value={cfg.ramIdx}
            onChange={(n) => set('ramIdx', n)}
            format={(n) => `${RAM_STEPS[n]} GB`}
          />
          <Slider
            label={pg ? 'shared_buffers' : 'innodb_buffer_pool_size'}
            min={2}
            max={70}
            value={cfg.poolPct}
            onChange={(n) => set('poolPct', n)}
            format={(n) => `${n}% · ${fmtBytes((RAM_STEPS[cfg.ramIdx] * GB * n) / 100)}`}
          />
          <Slider
            label="max_connections"
            min={10}
            max={1000}
            step={10}
            value={cfg.conns}
            onChange={(n) => set('conns', n)}
            format={(n) => fmtNum(n)}
          />
          <Slider
            label="Actually running a query"
            min={1}
            max={100}
            value={cfg.activePct}
            onChange={(n) => set('activePct', n)}
            format={(n) => `${n}% · ${fmtNum(Math.max(1, Math.round((cfg.conns * n) / 100)))}`}
          />
          <Slider
            label={pg ? 'work_mem' : 'sort_buffer_size'}
            min={0}
            max={GRANT_STEPS.length - 1}
            value={cfg.grantIdx}
            onChange={(n) => set('grantIdx', n)}
            format={(n) => `${GRANT_STEPS[n]} MB`}
          />
          <Slider
            label={pg ? 'Grant-taking nodes per plan' : 'Buffers held per connection'}
            min={1}
            max={6}
            value={cfg.nodes}
            onChange={(n) => set('nodes', n)}
            format={(n) => `${n}×`}
          />
          <Slider
            label="hash_mem_multiplier"
            min={10}
            max={80}
            step={5}
            value={cfg.hashMult10}
            onChange={(n) => set('hashMult10', n)}
            format={(n) => (n / 10).toFixed(1)}
            disabled={!pg}
          />
          <Slider
            label="Parallel workers per gather"
            min={0}
            max={8}
            value={cfg.workers}
            onChange={(n) => set('workers', n)}
            format={(n) => `${n}`}
            disabled={!pg}
          />
          <Slider
            label={pg ? 'maintenance_work_mem' : 'temptable_max_ram'}
            min={0}
            max={MAINT_STEPS.length - 1}
            value={cfg.maintIdx}
            onChange={(n) => set('maintIdx', n)}
            format={(n) => `${MAINT_STEPS[n]} MB`}
          />
          <Check label="Huge pages (2 MB)" checked={cfg.huge} onChange={(b) => set('huge', b)} />
        </>
      }
      legend={
        <Legend
          items={[
            { label: 'Page pool', color: C_POOL },
            { label: 'Pool bookkeeping', color: C_BOOK },
            { label: 'Log buffer', color: C_LOG },
            { label: pg ? 'Lock tables' : 'TempTable pool', color: C_LOCK },
            { label: 'Shared metadata', color: C_META },
            { label: 'Per-connection baseline', color: C_BASE },
            { label: 'Query grants', color: C_GRANT },
            { label: 'Page tables', color: C_PT },
            { label: 'Unallocated / page cache', color: C_FREE },
            { label: 'Over physical RAM', color: C_OVER },
          ]}
        />
      }
      stats={
        <Stats
          items={[
            {
              label: 'Shared segment',
              value: `${fmtBytes(L.sharedTotal)} · ${pct(L.sharedTotal)}`,
              hint: 'Allocated once at startup. This part really is a reservation.',
            },
            {
              label: `Peak per ${pg ? 'backend' : 'connection'}`,
              value: fmtBytes(L.peakPerBackend),
              hint: 'Baseline plus every grant one session can hold at the same time',
            },
            {
              label: 'Worst-case private total',
              value: `${fmtBytes(L.backendTotal)} · ${pct(L.backendTotal)}`,
              hint: 'The product nothing in the server enforces',
            },
            {
              label: 'Page tables',
              value: fmtBytes(L.ptBytes),
              hint: `${fmtNum(L.ptPages)} PTEs × 8 B × ${fmtNum(L.ptCopies)} process${L.ptCopies === 1 ? '' : 'es'}`,
            },
            {
              label: L.over > 0 ? 'Over RAM by' : 'Left for page cache',
              value: L.over > 0 ? fmtBytes(L.over) : fmtBytes(L.free),
              hint: L.over > 0 ? 'Reclaim, then swap, then the OOM killer' : 'The second tier of the cache',
            },
            {
              label: 'Shared segment in the TLB',
              value: `${(tlbNow.covered * 100).toFixed(2)}%`,
              hint: `A ${fmtNum(L2_TLB_ENTRIES)}-entry L2 dTLB reaches ${fmtBytes(tlbNow.reach)} at this page size`,
            },
          ]}
        />
      }
      note={<Note>{narration}</Note>}
      table={
        <>
          <table className="viz-table">
            <thead>
              <tr>
                <th>Allocation</th>
                <th>Knob</th>
                <th>Bytes</th>
                <th>% of RAM</th>
                <th>Reserved at startup?</th>
              </tr>
            </thead>
            <tbody>
              {L.slices.map((s) => (
                <tr key={s.key}>
                  <td>{s.label}</td>
                  <td>
                    <code>{s.knob}</code>
                  </td>
                  <td>{fmtBytes(s.bytes)}</td>
                  <td>{pct(s.bytes)}</td>
                  <td>
                    {s.group === 'shared'
                      ? 'yes — one allocation'
                      : s.group === 'backend'
                        ? 'no — an upper bound'
                        : s.group === 'pt'
                          ? 'no — grows as pages are touched'
                          : '—'}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
          <table className="viz-table">
            <thead>
              <tr>
                <th>Page size</th>
                <th>PTEs for the shared segment</th>
                <th>Page tables per process</th>
                <th>Page tables in total</th>
                <th>L2 dTLB reach</th>
                <th>Shared segment covered</th>
              </tr>
            </thead>
            <tbody>
              {tlbRows.map((r) => (
                <tr key={r.label}>
                  <td>{r.label}</td>
                  <td>{fmtNum(r.entries)}</td>
                  <td>{fmtBytes(r.ptPerProc)}</td>
                  <td>{fmtBytes(r.ptPerProc * L.ptCopies)}</td>
                  <td>{fmtBytes(r.reach)}</td>
                  <td>{(r.covered * 100).toFixed(2)}%</td>
                </tr>
              ))}
            </tbody>
          </table>
        </>
      }
    >
      <div ref={ref}>
        <TooltipHost>
          <MemoryFigure
            L={L}
            cols={cols}
            tlbRows={tlbRows}
            svgW={svgW}
            height={height}
            treeTop={treeTop}
            treeH={treeH}
            tlbTop={tlbTop}
            barX0={barX0}
            barW={barW}
            logX={logX}
            ramX={ramX}
          />
        </TooltipHost>
      </div>
    </VizPanel>
  );
}
