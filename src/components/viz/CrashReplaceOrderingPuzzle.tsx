import { useMemo, useState } from 'react';
import { VizPanel, Choice, Segmented, Slider, Button, Legend, Stats, makeRng, useSize } from './Viz';

/**
 * Crash-consistency ordering puzzle.
 *
 * The learner composes a syscall program that replaces (or creates) a config
 * file, picks a filesystem mode and a crash point, and the simulator enumerates
 * EVERY on-disk end state the filesystem's reordering rules allow.
 *
 * The model: each syscall issues one or more "persist units" (a dirent, a block
 * of file data, an inode's extent map, a rename transaction). At the moment of
 * the crash each issued unit is either durable or still in the page cache /
 * uncommitted journal. An fsync *forces* units durable. The filesystem imposes
 * ordering constraints on which subsets of units may be durable together; what
 * is left after filtering is the exact set of legal post-crash disk images.
 */

/* ------------------------------------------------------------------ units */

type UnitId = 'E' | 'D' | 'I' | 'R' | 'U' | 'T' | 'C0' | 'C1' | 'IC';

const UNIT: Record<UnitId, { label: string; kind: 'meta' | 'data'; desc: string }> = {
  E: { label: 'tmp dirent', kind: 'meta', desc: 'the directory entry naming app.conf.tmp' },
  D: { label: 'tmp data', kind: 'data', desc: 'the 8 KB of new bytes, in the tmp inode’s blocks' },
  I: { label: 'tmp extents', kind: 'meta', desc: 'the tmp inode’s extent map and i_size (delayed allocation materialises here)' },
  R: { label: 'rename', kind: 'meta', desc: 'app.conf’s directory entry now points at the tmp inode' },
  U: { label: 'unlink', kind: 'meta', desc: 'app.conf’s directory entry removed' },
  T: { label: 'truncate', kind: 'meta', desc: 'app.conf truncated to 0, its old blocks freed' },
  C0: { label: 'page 0', kind: 'data', desc: 'first 4 KB of the in-place rewrite' },
  C1: { label: 'page 1', kind: 'data', desc: 'second 4 KB of the in-place rewrite' },
  IC: { label: 'cfg extents', kind: 'meta', desc: 'app.conf’s own extent map and i_size' },
};

/* -------------------------------------------------------------------- ops */

type OpId = 'creat' | 'write' | 'fsyncTmp' | 'rename' | 'fsyncDir' | 'unlink' | 'trunc' | 'writeCfg' | 'fsyncCfg';

type Op = { call: string; short: string; issues: UnitId[]; forces: UnitId[] };

const OPS: Record<OpId, Op> = {
  creat: { call: 'open("app.conf.tmp", O_CREAT|O_WRONLY)', short: 'open tmp', issues: ['E'], forces: [] },
  write: { call: 'write(tmp, buf, 8192)', short: 'write tmp', issues: ['D', 'I'], forces: [] },
  fsyncTmp: { call: 'fsync(tmpfd)', short: 'fsync tmp', issues: [], forces: ['D', 'I'] },
  rename: { call: 'rename("app.conf.tmp", "app.conf")', short: 'rename', issues: ['R'], forces: [] },
  fsyncDir: { call: 'fsync(dirfd)   /* parent directory */', short: 'fsync dir', issues: [], forces: ['E', 'R', 'U'] },
  unlink: { call: 'unlink("app.conf")', short: 'unlink', issues: ['U'], forces: [] },
  trunc: { call: 'open("app.conf", O_WRONLY|O_TRUNC)', short: 'truncate', issues: ['T'], forces: [] },
  writeCfg: { call: 'write(cfgfd, buf, 8192)   /* in place */', short: 'write in place', issues: ['C0', 'C1', 'IC'], forces: [] },
  fsyncCfg: { call: 'fsync(cfgfd)', short: 'fsync file', issues: [], forces: ['T', 'C0', 'C1', 'IC'] },
};

const ADDABLE: OpId[] = ['creat', 'write', 'fsyncTmp', 'rename', 'fsyncDir', 'unlink', 'trunc', 'writeCfg', 'fsyncCfg'];

/* ----------------------------------------------------------- filesystems */

type Order = 'prefixAll' | 'metaPrefix' | 'none';

type Mode = {
  id: string;
  label: string;
  order: Order;
  delalloc: boolean;
  orderedData: boolean; // data=ordered: metadata exposing new blocks waits for those blocks
  fsyncForcesAllMeta: boolean; // one shared ordered journal: any log force flushes earlier metadata
  exposed: string; // what a size-N file with no written blocks reads back as
  blurb: string;
};

const MODES: Mode[] = [
  {
    id: 'ext4-ordered',
    label: 'ext4  data=ordered (default)',
    order: 'metaPrefix',
    delalloc: true,
    orderedData: true,
    fsyncForcesAllMeta: true,
    exposed: 'zeros',
    blurb:
      'One ordered journal for metadata; file data for newly allocated blocks is written out before the transaction that exposes it commits. auto_da_alloc forces the delayed allocation out when you rename over an existing file.',
  },
  {
    id: 'ext4-writeback',
    label: 'ext4  data=writeback',
    order: 'metaPrefix',
    delalloc: true,
    orderedData: false,
    fsyncForcesAllMeta: true,
    exposed: 'zeros',
    blurb:
      'Metadata is still journaled in order, but nothing orders file data against it. The rename can be durable while the bytes it points at are not — this is where zero-length config files come from.',
  },
  {
    id: 'ext4-journal',
    label: 'ext4  data=journal',
    order: 'prefixAll',
    delalloc: false,
    orderedData: false,
    fsyncForcesAllMeta: true,
    exposed: 'zeros',
    blurb:
      'Data goes through the journal too, so data and metadata commit atomically in issue order. Every write is written twice; throughput roughly halves.',
  },
  {
    id: 'xfs',
    label: 'XFS  (metadata-only log)',
    order: 'metaPrefix',
    delalloc: true,
    orderedData: false,
    fsyncForcesAllMeta: true,
    exposed: 'zeros (unwritten extents)',
    blurb:
      'Metadata-only logging with delayed logging (CIL) and aggressive delayed allocation. Same lattice as data=writeback — but XFS marks unallocated ranges as unwritten extents, so you read zeros, never another file’s deleted data.',
  },
  {
    id: 'cow',
    label: 'btrfs / ZFS  (copy-on-write)',
    order: 'prefixAll',
    delalloc: false,
    orderedData: false,
    fsyncForcesAllMeta: true,
    exposed: 'zeros',
    blurb:
      'Nothing is overwritten in place; a transaction group publishes a whole new tree root atomically (btrfs ~30 s, ZFS ~5 s txg). A crash lands you exactly on some past commit — a prefix, never a mixture.',
  },
  {
    id: 'posix',
    label: 'POSIX minimum (what is actually promised)',
    order: 'none',
    delalloc: true,
    orderedData: false,
    fsyncForcesAllMeta: false,
    exposed: 'zeros or stale blocks',
    blurb:
      'The standard promises exactly one thing: what fsync() returned on is durable. Nothing else is ordered, and fsync of a file says nothing about the directory entry naming it. Portable code must survive this column.',
  },
];

/* ------------------------------------------------------------------ plans */

type PlanId = 'inplace' | 'renameOnly' | 'fsyncRename' | 'safe' | 'lateFsync' | 'unlinkFirst' | 'custom';

const PLANS: { id: PlanId; label: string; steps: OpId[] }[] = [
  { id: 'inplace', label: 'Overwrite in place (naive)', steps: ['trunc', 'writeCfg', 'fsyncCfg'] },
  { id: 'renameOnly', label: 'Write temp + rename, no fsync', steps: ['creat', 'write', 'rename'] },
  { id: 'fsyncRename', label: 'fsync file, then rename', steps: ['creat', 'write', 'fsyncTmp', 'rename'] },
  { id: 'safe', label: 'Safe replace (write, fsync, rename, fsync dir)', steps: ['creat', 'write', 'fsyncTmp', 'rename', 'fsyncDir'] },
  { id: 'lateFsync', label: 'Rename first, fsync after', steps: ['creat', 'write', 'rename', 'fsyncTmp', 'fsyncDir'] },
  { id: 'unlinkFirst', label: 'Unlink, then create fresh', steps: ['unlink', 'creat', 'write', 'fsyncTmp', 'rename'] },
];

/* --------------------------------------------------------------- outcomes */

type OutKind = 'new' | 'old' | 'zero' | 'exposed' | 'torn' | 'missing';

function outcomeMeta(kind: OutKind, scenario: 'replace' | 'create', mode: Mode) {
  switch (kind) {
    case 'new':
      return { label: 'New contents', sev: 'good' as const, detail: 'The replacement is on disk and complete.' };
    case 'old':
      return { label: 'Old contents', sev: 'warn' as const, detail: 'Intact, but the update is gone. The app can retry.' };
    case 'missing':
      return scenario === 'create'
        ? { label: 'Still missing', sev: 'warn' as const, detail: 'The file never appeared. Nothing is corrupt; the update is lost.' }
        : { label: 'File gone', sev: 'bad' as const, detail: 'The directory entry is gone and nothing replaced it.' };
    case 'zero':
      return { label: 'Zero-length file', sev: 'bad' as const, detail: 'The name resolves to an inode with i_size = 0. The classic ext4 surprise.' };
    case 'exposed':
      return { label: `Right size, ${mode.exposed}`, sev: 'bad' as const, detail: 'Metadata says 8 KB; the blocks were never written back.' };
    case 'torn':
      return { label: 'Half old, half new', sev: 'bad' as const, detail: 'Page 0 and page 1 disagree — a torn file.' };
  }
}

const SEV_COLOR = { good: 'var(--viz-good)', warn: 'var(--viz-warning)', bad: 'var(--viz-critical)' } as const;

/* ------------------------------------------------------------- simulation */

type Issued = { id: UnitId; at: number; ord: number; forced: boolean };

function issuedUnits(steps: OpId[], crashAfter: number, mode: Mode) {
  const seen = new Set<UnitId>();
  const list: Issued[] = [];
  const forced = new Set<UnitId>();
  for (let i = 0; i < crashAfter && i < steps.length; i++) {
    const op = OPS[steps[i]];
    for (const u of op.issues) {
      if (!seen.has(u)) {
        seen.add(u);
        list.push({ id: u, at: i, ord: list.length, forced: false });
      }
    }
    if (op.forces.length > 0) {
      for (const u of op.forces) if (seen.has(u)) forced.add(u);
      if (mode.fsyncForcesAllMeta) for (const u of seen) if (UNIT[u].kind === 'meta') forced.add(u);
    }
  }
  // Delayed allocation: the extent map does not exist until writeback runs, so
  // it commits *after* everything issued before it — including the rename.
  for (const it of list) {
    it.forced = forced.has(it.id);
    if (mode.delalloc && (it.id === 'I' || it.id === 'IC')) it.ord += 100;
  }
  return list;
}

function dependencies(steps: OpId[], mode: Mode, scenario: 'replace' | 'create'): [UnitId, UnitId[]][] {
  if (!mode.orderedData) return [];
  const deps: [UnitId, UnitId[]][] = [['I', ['D']]];
  // ext4's auto_da_alloc heuristic fires when you rename *over an existing file*.
  if (scenario === 'replace') deps.push(['R', ['D', 'I']]);
  const t = steps.indexOf('trunc');
  const w = steps.indexOf('writeCfg');
  if (t >= 0 && w > t) deps.push(['IC', ['C0', 'C1']]);
  return deps;
}

function isLegal(S: Set<UnitId>, units: Issued[], mode: Mode, deps: [UnitId, UnitId[]][]) {
  for (const [a, bs] of deps) {
    if (!S.has(a)) continue;
    for (const b of bs) if (units.some((u) => u.id === b) && !S.has(b)) return false;
  }
  if (mode.order === 'none') return true;
  const seq = [...units].sort((x, y) => x.ord - y.ord);
  let gap = false;
  for (const u of seq) {
    if (mode.order === 'metaPrefix' && UNIT[u.id].kind !== 'meta') continue;
    if (S.has(u.id)) {
      if (gap) return false;
    } else gap = true;
  }
  return true;
}

function replay(S: Set<UnitId>, units: Issued[], scenario: 'replace' | 'create'): OutKind {
  let cfg: null | 'old' | 'tmp' = scenario === 'replace' ? 'old' : null;
  let cfgSize = scenario === 'replace' ? 2 : 0;
  let cfgTrunc = false;
  const cfgPages = [false, false]; // true = the new bytes landed
  let tmpSize = 0;
  let tmpData = false;

  for (const u of [...units].sort((a, b) => a.ord - b.ord)) {
    if (!S.has(u.id)) continue;
    switch (u.id) {
      case 'E':
        break;
      case 'D':
        tmpData = true;
        break;
      case 'I':
        tmpSize = 2;
        break;
      case 'R':
        cfg = 'tmp';
        break;
      case 'U':
        cfg = null;
        break;
      case 'T':
        cfgSize = 0;
        cfgTrunc = true;
        cfgPages[0] = false;
        cfgPages[1] = false;
        break;
      case 'IC':
        cfgSize = 2;
        break;
      case 'C0':
        cfgPages[0] = true;
        break;
      case 'C1':
        cfgPages[1] = true;
        break;
    }
  }

  if (cfg === null) return 'missing';
  if (cfg === 'tmp') {
    if (tmpSize === 0) return 'zero';
    return tmpData ? 'new' : 'exposed';
  }
  if (cfgSize === 0) return cfgTrunc || scenario === 'create' ? 'zero' : 'old';
  const n = (cfgPages[0] ? 1 : 0) + (cfgPages[1] ? 1 : 0);
  if (n === 2) return 'new';
  if (n === 1) return 'torn';
  return cfgTrunc ? 'exposed' : 'old';
}

type Row = { durable: UnitId[]; kind: OutKind };

function enumerate(steps: OpId[], crashAfter: number, mode: Mode, scenario: 'replace' | 'create') {
  const units = issuedUnits(steps, crashAfter, mode);
  const deps = dependencies(steps, mode, scenario);
  const free = units.filter((u) => !u.forced);
  const rows: Row[] = [];
  const total = 1 << free.length;
  for (let mask = 0; mask < total; mask++) {
    const S = new Set<UnitId>(units.filter((u) => u.forced).map((u) => u.id));
    for (let b = 0; b < free.length; b++) if (mask & (1 << b)) S.add(free[b].id);
    if (!isLegal(S, units, mode, deps)) continue;
    rows.push({ durable: units.filter((u) => S.has(u.id)).map((u) => u.id), kind: replay(S, units, scenario) });
  }
  rows.sort((a, b) => a.durable.length - b.durable.length);
  return { units, rows, deps };
}

/* ------------------------------------------------------------------- view */

export default function CrashReplaceOrderingPuzzle() {
  const [modeId, setModeId] = useState('ext4-ordered');
  const [scenario, setScenario] = useState<'replace' | 'create'>('replace');
  const [planId, setPlanId] = useState<PlanId>('renameOnly');
  const [steps, setSteps] = useState<OpId[]>(PLANS[1].steps);
  const [crashAfter, setCrashAfter] = useState(3);
  const [crashSeed, setCrashSeed] = useState(0);
  const [ref, width] = useSize(760);

  const mode = MODES.find((m) => m.id === modeId)!;
  const clampCrash = Math.min(crashAfter, steps.length);

  const { units, rows } = useMemo(
    () => enumerate(steps, clampCrash, mode, scenario),
    [steps, clampCrash, mode, scenario],
  );

  // Verdict over every crash point, not just the selected one: a protocol is
  // crash-safe only if no crash point admits a corrupt end state.
  const verdict = useMemo(() => {
    let corruptAnywhere = false;
    for (let k = 0; k <= steps.length; k++) {
      for (const r of enumerate(steps, k, mode, scenario).rows) {
        if (outcomeMeta(r.kind, scenario, mode).sev === 'bad') corruptAnywhere = true;
      }
    }
    const final = enumerate(steps, steps.length, mode, scenario).rows;
    const durable = final.every((r) => r.kind === 'new');
    return { corruptAnywhere, durable };
  }, [steps, mode, scenario]);

  const kinds = useMemo(() => {
    const order: OutKind[] = ['new', 'old', 'missing', 'torn', 'exposed', 'zero'];
    const counts = new Map<OutKind, number>();
    for (const r of rows) counts.set(r.kind, (counts.get(r.kind) ?? 0) + 1);
    return order.filter((k) => counts.has(k)).map((k) => ({ kind: k, count: counts.get(k)! }));
  }, [rows]);

  const badCount = kinds.filter((k) => outcomeMeta(k.kind, scenario, mode).sev === 'bad').length;

  // Deterministic "press crash": pick one legal disk image.
  const sampled = crashSeed === 0 || rows.length === 0 ? null : Math.floor(makeRng(crashSeed)() * rows.length) % rows.length;

  const setPlan = (id: PlanId) => {
    const p = PLANS.find((x) => x.id === id);
    if (!p) return;
    setPlanId(id);
    setSteps(p.steps);
    setCrashAfter(p.steps.length);
    setCrashSeed(0);
  };

  const mutate = (next: OpId[]) => {
    setSteps(next);
    setPlanId('custom');
    setCrashAfter(next.length);
    setCrashSeed(0);
  };

  const move = (i: number, d: number) => {
    const next = [...steps];
    const j = i + d;
    if (j < 0 || j >= next.length) return;
    [next[i], next[j]] = [next[j], next[i]];
    mutate(next);
  };

  /* ---- matrix geometry ---- */
  const colW = width < 620 ? 66 : 78;
  const outW = 210;
  const rowH = 24;
  const headH = 34;
  // Show the first 14 images, but never hide the one a "Crash now" landed on:
  // if it falls past the cut, swap it into the last visible slot.
  const shown = useMemo(() => {
    const head = rows.slice(0, 14).map((r, i) => ({ r, idx: i }));
    if (sampled !== null && sampled >= head.length && rows[sampled]) {
      head[head.length - 1] = { r: rows[sampled], idx: sampled };
    }
    return head;
  }, [rows, sampled]);
  const matW = units.length * colW + outW + 8;
  const matH = headH + Math.max(1, shown.length) * rowH + 8;

  return (
    <VizPanel
      title="Ordering puzzle: replacing a config file across a crash"
      subtitle="Arrange the syscalls, choose a filesystem, pick where the power fails. Every row below is a disk image the filesystem's reordering rules actually permit."
      controls={
        <>
          <Choice
            label="Filesystem"
            value={modeId}
            onChange={setModeId}
            options={MODES.map((m) => ({ value: m.id, label: m.label }))}
          />
          <Choice
            label="Plan"
            value={planId}
            onChange={(v) => setPlan(v as PlanId)}
            options={[...PLANS.map((p) => ({ value: p.id as PlanId, label: p.label })), { value: 'custom' as PlanId, label: 'Custom (edited)' }]}
          />
          <Segmented
            label="Target"
            value={scenario}
            onChange={(v) => {
              setScenario(v);
              setCrashSeed(0);
            }}
            options={[
              { value: 'replace', label: 'app.conf exists', title: 'Replacing a file that already has old contents' },
              { value: 'create', label: 'brand-new file', title: 'The name does not exist yet' },
            ]}
          />
          <Slider
            label="Crash after step"
            min={0}
            max={steps.length}
            value={clampCrash}
            onChange={(n) => {
              setCrashAfter(n);
              setCrashSeed(0);
            }}
            format={(n) => (n === 0 ? 'before step 1' : n >= steps.length ? `${n} — program returned` : `${n} (${OPS[steps[n - 1]].short})`)}
          />
          <Button primary onClick={() => setCrashSeed((s) => s + 1)}>
            Crash now
          </Button>
        </>
      }
      legend={
        <Legend
          items={[
            { label: 'Durable at the crash', color: 'var(--viz-clean)' },
            { label: 'Lost — still in page cache / uncommitted', color: 'var(--viz-dirty)' },
            { label: 'Forced by an fsync (✓)', color: 'var(--viz-ink)' },
            { label: 'End state: correct', color: 'var(--viz-good)', shape: 'dot' },
            { label: 'End state: update lost, file intact', color: 'var(--viz-warning)', shape: 'dot' },
            { label: 'End state: corrupt', color: 'var(--viz-critical)', shape: 'dot' },
          ]}
        />
      }
      stats={
        <Stats
          items={[
            { label: 'Legal disk images', value: rows.length, hint: 'Distinct durable-unit subsets the reordering rules allow at this crash point' },
            { label: 'Distinct end states', value: kinds.length },
            { label: 'Corrupting states', value: badCount, hint: 'Zero-length, torn, exposed blocks or a missing file' },
            { label: 'Atomic at every crash point', value: verdict.corruptAnywhere ? 'no' : 'yes' },
            { label: 'Durable once it returns', value: verdict.durable ? 'yes' : 'no', hint: 'After the last syscall returns, is “new contents” the only possible outcome?' },
          ]}
        />
      }
      note={
        <>
          <strong>
            {verdict.corruptAnywhere
              ? 'This program can corrupt the file.'
              : verdict.durable
                ? 'This program is crash-safe and durable on this filesystem.'
                : 'This program never corrupts the file, but the update can still vanish.'}
          </strong>{' '}
          {mode.blurb}
        </>
      }
      table={
        <table className="viz-table">
          <thead>
            <tr>
              <th>#</th>
              {units.map((u) => (
                <th key={u.id}>{UNIT[u.id].label}</th>
              ))}
              <th>app.conf after recovery</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((r, i) => {
              const m = outcomeMeta(r.kind, scenario, mode);
              return (
                <tr key={i}>
                  <td>{i + 1}</td>
                  {units.map((u) => (
                    <td key={u.id}>{r.durable.includes(u.id) ? (u.forced ? 'durable (fsync)' : 'durable') : 'lost'}</td>
                  ))}
                  <td>
                    {m.label} — {m.detail}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      }
    >
      <div ref={ref}>
        {/* ---------------------------------------------------- the program */}
        <div>
          <div style={{ maxWidth: '34rem' }}>
            <ol style={{ listStyle: 'none', margin: 0, padding: 0 }}>
              {steps.map((s, i) => (
                <li
                  key={`${s}-${i}`}
                  style={{
                    display: 'flex',
                    alignItems: 'center',
                    gap: '.4rem',
                    padding: '.2rem 0',
                    opacity: i < clampCrash ? 1 : 0.4,
                  }}
                >
                  <span style={{ width: '1.2rem', color: 'var(--viz-ink-muted)', fontVariantNumeric: 'tabular-nums' }}>{i + 1}</span>
                  <code
                    style={{
                      flex: 1,
                      fontSize: '.78rem',
                      color: OPS[s].forces.length > 0 ? 'var(--viz-clean)' : 'var(--viz-ink)',
                    }}
                  >
                    {OPS[s].call}
                  </code>
                  <Button onClick={() => move(i, -1)} title="Move earlier">
                    ↑
                  </Button>
                  <Button onClick={() => move(i, 1)} title="Move later">
                    ↓
                  </Button>
                  <Button onClick={() => mutate(steps.filter((_, k) => k !== i))} title="Remove this call">
                    ✕
                  </Button>
                </li>
              ))}
              <li
                style={{
                  display: 'flex',
                  alignItems: 'center',
                  gap: '.4rem',
                  padding: '.25rem 0',
                  borderTop: '2px dashed var(--viz-critical)',
                  color: 'var(--viz-critical)',
                  fontWeight: 600,
                }}
              >
                ⚡ power fails here{clampCrash < steps.length ? ` (after step ${clampCrash})` : ' — the program had already returned'}
              </li>
            </ol>
            <div style={{ display: 'flex', flexWrap: 'wrap', gap: '.3rem', marginTop: '.5rem' }}>
              {ADDABLE.map((op) => (
                <Button key={op} onClick={() => mutate([...steps, op])} title={OPS[op].call}>
                  + {OPS[op].short}
                </Button>
              ))}
            </div>
          </div>

          {/* ------------------------------------------ the enumerated images */}
          <p style={{ margin: '.9rem 0 .3rem', color: 'var(--viz-ink-2)', fontSize: '.78rem' }}>
            Every disk image this filesystem may leave behind, one row each — which units made it
            to stable storage, and what <code>app.conf</code> reads back as afterwards.
          </p>
          <svg
            width={matW}
            height={matH}
            role="img"
            aria-label="Every legal post-crash disk image for this program"
          >
            {units.map((u, c) => (
              <g key={u.id}>
                <text x={c * colW + colW / 2} y={13} textAnchor="middle" fill="var(--viz-ink)">
                  {UNIT[u.id].label}
                </text>
                <text x={c * colW + colW / 2} y={25} textAnchor="middle" fill="var(--viz-ink-muted)">
                  {UNIT[u.id].kind}
                </text>
              </g>
            ))}
            <text x={units.length * colW + 6} y={13} fill="var(--viz-ink)">
              app.conf after recovery
            </text>
            <line
              className="viz-axis-line"
              x1={0}
              y1={headH - 6}
              x2={units.length * colW + outW}
              y2={headH - 6}
            />

            {shown.length === 0 ? (
              <text x={0} y={headH + 16} fill="var(--viz-ink-2)">
                Nothing has been issued yet — the disk still holds the old file.
              </text>
            ) : null}

            {shown.map(({ r, idx }, i) => {
              const y = headH + i * rowH;
              const m = outcomeMeta(r.kind, scenario, mode);
              const isSample = sampled === idx;
              return (
                <g key={i}>
                  {isSample ? (
                    <rect x={-4} y={y - 2} width={units.length * colW + outW} height={rowH - 2} rx={5} fill="var(--viz-neutral)" />
                  ) : null}
                  {units.map((u, c) => {
                    const on = r.durable.includes(u.id);
                    return (
                      <g key={u.id}>
                        <rect
                          x={c * colW + 6}
                          y={y}
                          width={colW - 14}
                          height={rowH - 8}
                          rx={4}
                          fill={on ? 'var(--viz-clean)' : 'var(--viz-dirty)'}
                          opacity={on ? 1 : 0.28}
                          stroke={u.forced ? 'var(--viz-ink)' : 'none'}
                          strokeWidth={u.forced ? 1.5 : 0}
                        />
                        <text
                          x={c * colW + colW / 2}
                          y={y + rowH - 13}
                          textAnchor="middle"
                          fill={on ? 'var(--viz-surface)' : 'var(--viz-ink-2)'}
                        >
                          {on ? (u.forced ? '✓ durable' : 'durable') : 'lost'}
                        </text>
                      </g>
                    );
                  })}
                  <circle cx={units.length * colW + 12} cy={y + rowH / 2 - 4} r={5} fill={SEV_COLOR[m.sev]} />
                  <text x={units.length * colW + 24} y={y + rowH - 12} fill="var(--viz-ink)">
                    {m.label}
                    {m.sev === 'bad' ? '  ✗' : m.sev === 'good' ? '  ✓' : '  ~'}
                  </text>
                </g>
              );
            })}
          </svg>
        </div>

        {rows.length > shown.length ? (
          <p style={{ margin: '.5rem 0 0', color: 'var(--viz-ink-muted)', fontSize: '.75rem' }}>
            Showing {shown.length} of {rows.length} legal disk images — “Show the numbers” lists them all.
          </p>
        ) : null}

        {sampled !== null && rows[sampled] ? (
          <p style={{ margin: '.5rem 0 0', fontSize: '.8rem', color: 'var(--viz-ink)' }}>
            ⚡ This crash landed on image {sampled + 1}:{' '}
            <strong>{outcomeMeta(rows[sampled].kind, scenario, mode).label}</strong>{' '}
            — {outcomeMeta(rows[sampled].kind, scenario, mode).detail}
          </p>
        ) : null}

        <p style={{ margin: '.5rem 0 0', fontSize: '.75rem', color: 'var(--viz-ink-2)' }}>
          {units.length === 0
            ? 'Add a syscall to issue something to the filesystem.'
            : units
                .map((u) => `${UNIT[u.id].label}: ${UNIT[u.id].desc}${u.forced ? ' — forced durable by an fsync' : ''}`)
                .join(' · ')}
        </p>
      </div>
    </VizPanel>
  );
}
