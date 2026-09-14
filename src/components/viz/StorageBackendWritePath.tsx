import { useMemo, useState } from 'react';
import {
  VizPanel,
  Choice,
  Slider,
  Legend,
  Stats,
  TooltipHost,
  useTip,
  fmtBytes,
  fmtTime,
  fmtNum,
  makeRng,
  useSize,
} from './Viz';

/**
 * The same engine write path, dropped onto five storage substrates.
 *
 * The model is the point: a POSIX file lets the engine overwrite 4 KB at an offset
 * and then fdatasync it; an object store has no offsets and no fsync, so every step
 * is rewritten into whole-object GET/PUT — which is where the latency, the request
 * bill and the write amplification all come from.
 */

type Mode = 'native' | 'rewritten' | 'absent';
type Group = 'commit' | 'checkpoint' | 'read' | 'recovery';

type FileProfile = {
  buffered: number; // write() into the page cache
  fsync: number; // fdatasync of the WAL
  fsyncBig: number; // fdatasync of a data file with many dirty pages
  dirFsync: number; // rename() + fsync of the parent directory
  read: number; // random 4 KB pread that misses the page cache
  readdir: number; // readdir over ~10 000 segment files
  bwMBs: number;
};

type ObjProfile = {
  putBase: number; // time to first byte of a PUT, small object
  getBase: number;
  listBase: number; // LIST v2, one page of 1000 keys
  bwMBs: number; // per-connection transfer rate
};

type Backend = {
  id: string;
  label: string;
  short: string;
  kind: 'file' | 'object';
  file?: FileProfile;
  obj?: ObjProfile;
  tailMult: number; // how ugly the tail is relative to p50
  gbMonth: number; // $ per GB-month
  putPrice: number; // $ per write-class request
  getPrice: number; // $ per read-class request
  capOps: number; // sustained ops/sec ceiling
  capLabel: string;
  capBwMBs: number; // sustained MB/s ceiling
  api: string;
  survives: Record<'drive' | 'host' | 'az' | 'unflushed' | 'corrupt', boolean>;
};

const MS = 1e6;

const BACKENDS: Backend[] = [
  {
    id: 'nvme',
    label: 'Local NVMe (instance store)',
    short: 'NVMe',
    kind: 'file',
    file: { buffered: 3_000, fsync: 60_000, fsyncBig: 95_000, dirFsync: 80_000, read: 20_000, readdir: 2 * MS, bwMBs: 3000 },
    tailMult: 2,
    gbMonth: 0.08,
    putPrice: 0,
    getPrice: 0,
    capOps: 400_000,
    capLabel: 'device IOPS',
    capBwMBs: 3000,
    api: 'ext4/XFS on /dev/nvme0n1 — full POSIX file API',
    survives: { drive: false, host: false, az: false, unflushed: false, corrupt: false },
  },
  {
    id: 'nvme-raid',
    label: 'Local NVMe ×4, RAID 10 (md)',
    short: 'NVMe RAID10',
    kind: 'file',
    file: { buffered: 3_000, fsync: 70_000, fsyncBig: 110_000, dirFsync: 92_000, read: 18_000, readdir: 2 * MS, bwMBs: 5000 },
    tailMult: 2.2,
    gbMonth: 0.16,
    putPrice: 0,
    getPrice: 0,
    capOps: 700_000,
    capLabel: 'array IOPS',
    capBwMBs: 5000,
    api: 'ext4/XFS on /dev/md0 — full POSIX file API',
    survives: { drive: true, host: false, az: false, unflushed: false, corrupt: false },
  },
  {
    id: 'ebs',
    label: 'EBS gp3 (network block device)',
    short: 'EBS gp3',
    kind: 'file',
    file: { buffered: 3_000, fsync: 700_000, fsyncBig: 1_200_000, dirFsync: 900_000, read: 600_000, readdir: 4 * MS, bwMBs: 125 },
    tailMult: 6,
    gbMonth: 0.08,
    putPrice: 0,
    getPrice: 0,
    capOps: 3_000,
    capLabel: 'gp3 baseline IOPS',
    capBwMBs: 125,
    api: 'ext4/XFS on /dev/nvme1n1 — POSIX file API over a network protocol',
    survives: { drive: true, host: true, az: false, unflushed: false, corrupt: false },
  },
  {
    id: 's3',
    label: 'S3 Standard (object store)',
    short: 'S3 Standard',
    kind: 'object',
    obj: { putBase: 45 * MS, getBase: 25 * MS, listBase: 60 * MS, bwMBs: 60 },
    tailMult: 8,
    gbMonth: 0.023,
    putPrice: 5e-6,
    getPrice: 4e-7,
    capOps: 3_500,
    capLabel: 'PUT/s per prefix',
    capBwMBs: 100_000,
    api: 'PUT / GET / LIST on whole keys — no offsets, no append, no rename',
    survives: { drive: true, host: true, az: true, unflushed: false, corrupt: false },
  },
  {
    id: 's3x',
    label: 'S3 Express One Zone',
    short: 'S3 Express',
    kind: 'object',
    obj: { putBase: 6 * MS, getBase: 4 * MS, listBase: 12 * MS, bwMBs: 90 },
    tailMult: 4,
    gbMonth: 0.11,
    putPrice: 1e-6,
    getPrice: 3e-8,
    capOps: 200_000,
    capLabel: 'PUT/s per directory bucket',
    capBwMBs: 100_000,
    api: 'Same object API, one AZ, session-authenticated',
    survives: { drive: true, host: true, az: false, unflushed: false, corrupt: false },
  },
];

const MODE_COLOR: Record<Mode, string> = {
  native: 'var(--viz-1)',
  rewritten: 'var(--viz-2)',
  absent: 'var(--viz-stale)',
};

const MODE_LABEL: Record<Mode, string> = {
  native: 'native',
  rewritten: 'rewritten',
  absent: 'no equivalent',
};

type Step = {
  id: string;
  want: string;
  call: string;
  mode: Mode;
  group: Group;
  p50: number;
  p99: number;
  put: number;
  get: number;
  bytes: number;
  note: string;
};

/** Deterministic p50/p99 from a base latency and a tail multiplier. */
function tail(base: number, mult: number, seed: number) {
  if (base <= 0) return { p50: 0, p99: 0 };
  const rng = makeRng(seed);
  const xs: number[] = [];
  for (let i = 0; i < 256; i++) {
    const jitter = 0.9 + 0.3 * rng();
    const spike = rng() < 0.03 ? mult : 1;
    xs.push(base * jitter * spike);
  }
  xs.sort((a, b) => a - b);
  return { p50: xs[127], p99: xs[253] };
}

const PAGE = 4096;

function plan(b: Backend, batch: number, segment: number): Step[] {
  const seed = (BACKENDS.findIndex((x) => x.id === b.id) + 1) * 977;
  const mk = (
    i: number,
    s: Omit<Step, 'p50' | 'p99'> & { base: number },
  ): Step => {
    const { base, ...rest } = s;
    const t = tail(base, b.tailMult, seed + i * 31 + 1);
    return { ...rest, p50: t.p50, p99: t.p99 };
  };

  if (b.kind === 'file') {
    const f = b.file!;
    const xfer = (bytes: number) => (bytes / (f.bwMBs * 1e6)) * 1e9;
    return [
      mk(0, {
        id: 'append',
        want: 'Append a commit record to the WAL',
        call: `pwrite(wal_fd, buf, ${fmtBytes(batch)})`,
        mode: 'native',
        group: 'commit',
        base: f.buffered + xfer(batch) * 0.05,
        put: 0,
        get: 0,
        bytes: batch,
        note: 'Lands in the kernel page cache and returns. Identical cost on NVMe and EBS because neither device has been touched yet — this is why a benchmark without fsync tells you nothing about the storage.',
      }),
      mk(1, {
        id: 'flush',
        want: 'Make the commit durable',
        call: 'fdatasync(wal_fd)',
        mode: 'native',
        group: 'commit',
        base: f.fsync + xfer(batch),
        put: 0,
        get: 0,
        bytes: batch,
        note: 'One device write plus a cache-flush command. On EBS this is a network round trip to the storage fleet, which is why the same code is ~10× slower with no code change.',
      }),
      mk(2, {
        id: 'page',
        want: 'Overwrite a 4 KB page in place',
        call: `pwrite(data_fd, page, 4096, offset)`,
        mode: 'native',
        group: 'checkpoint',
        base: f.buffered,
        put: 0,
        get: 0,
        bytes: PAGE,
        note: 'Legal: a block device is a mutable array of 512 B/4 KB sectors, so the engine can update a B-tree page where it already lives. The FTL copies-on-write underneath, but the API hides that.',
      }),
      mk(3, {
        id: 'datasync',
        want: 'Flush the dirty data file',
        call: 'fdatasync(data_fd)',
        mode: 'native',
        group: 'checkpoint',
        base: f.fsyncBig + xfer(segment * 0.02),
        put: 0,
        get: 0,
        bytes: Math.round(segment * 0.02),
        note: 'Checkpoint cost. Postgres spreads this over checkpoint_completion_target precisely because one big fsync stalls every commit behind it.',
      }),
      mk(4, {
        id: 'publish',
        want: 'Atomically publish a new segment',
        call: 'rename(tmp, final) + fsync(dirfd)',
        mode: 'native',
        group: 'checkpoint',
        base: f.dirFsync,
        put: 0,
        get: 0,
        bytes: 0,
        note: 'rename() within a directory is atomic, and the directory fsync is what makes the name survive a crash. This is how an LSM engine installs an SSTable and how a manifest is swapped.',
      }),
      mk(5, {
        id: 'read',
        want: 'Read one 4 KB page at an offset',
        call: 'pread(data_fd, 4096, offset)',
        mode: 'native',
        group: 'read',
        base: f.read,
        put: 0,
        get: 0,
        bytes: PAGE,
        note: 'A buffer-pool miss. Byte-addressable reads are the reason a B-tree descent works at all.',
      }),
      mk(6, {
        id: 'list',
        want: 'Enumerate 10 000 segments at recovery',
        call: 'readdir() / getdents64()',
        mode: 'native',
        group: 'recovery',
        base: f.readdir,
        put: 0,
        get: 0,
        bytes: 0,
        note: 'Directory metadata is local and cached. Startup enumeration is free enough that engines never think about it.',
      }),
    ];
  }

  const o = b.obj!;
  const xfer = (bytes: number) => (bytes / (o.bwMBs * 1e6)) * 1e9;
  const listPages = Math.ceil(10_000 / 1000);
  return [
    mk(0, {
      id: 'append',
      want: 'Append a commit record to the WAL',
      call: 'buffer in RAM — no append API exists',
      mode: 'absent',
      group: 'commit',
      base: 1_000,
      put: 0,
      get: 0,
      bytes: 0,
      note: 'An object is immutable once written. The engine has to accumulate a batch in memory (or on a local disk) and turn it into one whole-object PUT, which is exactly why object-backed engines have a group-commit window measured in tens of milliseconds.',
    }),
    mk(1, {
      id: 'flush',
      want: 'Make the commit durable',
      call: `PUT wal/000123.log (${fmtBytes(batch)})`,
      mode: 'rewritten',
      group: 'commit',
      base: o.putBase + xfer(batch),
      put: 1,
      get: 0,
      bytes: batch,
      note: 'There is no fsync. The HTTP 200 on the PUT is the durability acknowledgement — S3 has committed the object to multiple devices before it answers. You pay for that promise per request, not per byte.',
    }),
    mk(2, {
      id: 'page',
      want: 'Overwrite a 4 KB page in place',
      call: `GET whole object + patch + PUT new object (${fmtBytes(segment)} each way)`,
      mode: 'rewritten',
      group: 'checkpoint',
      base: o.getBase + xfer(segment) + o.putBase + xfer(segment),
      put: 1,
      get: 1,
      bytes: segment * 2,
      note: 'There is no byte-range write. Patching 4 KB inside a segment means moving the whole segment twice — the write amplification that pushes every object-store engine to immutable, log-structured files that are only ever appended to as new keys.',
    }),
    mk(3, {
      id: 'datasync',
      want: 'Flush the dirty data file',
      call: 'nothing to do — the PUT already was the flush',
      mode: 'absent',
      group: 'checkpoint',
      base: 0,
      put: 0,
      get: 0,
      bytes: 0,
      note: 'No page cache under your control, so no flush step, and no torn-page risk from a partial write either: an object either exists in full or does not exist.',
    }),
    mk(4, {
      id: 'publish',
      want: 'Atomically publish a new segment',
      call: 'PUT manifest.json with If-Match: <etag>',
      mode: 'rewritten',
      group: 'checkpoint',
      base: o.putBase + xfer(64 * 1024),
      put: 1,
      get: 0,
      bytes: 64 * 1024,
      note: 'Conditional PUT gives you compare-and-swap on a single key, which is enough for a manifest pointer. There is no rename and no multi-key atomic commit — that is why Iceberg, Delta and every object-store engine funnel commits through one pointer object.',
    }),
    mk(5, {
      id: 'read',
      want: 'Read one 4 KB page at an offset',
      call: 'GET segment Range: bytes=off-off+4095',
      mode: 'native',
      group: 'read',
      base: o.getBase + xfer(PAGE),
      put: 0,
      get: 1,
      bytes: PAGE,
      note: 'Ranged reads do work — the asymmetry is the whole story. You can read any byte range you like; you can never write one. Engines exploit this by keeping index and footer metadata at known offsets in the object.',
    }),
    mk(6, {
      id: 'list',
      want: 'Enumerate 10 000 segments at recovery',
      call: `${listPages} × LIST (1000 keys per page)`,
      mode: 'rewritten',
      group: 'recovery',
      base: o.listBase * listPages,
      put: listPages,
      get: 0,
      bytes: 0,
      note: 'LIST returns 1000 keys per request, is billed at the write-request rate, and is only eventually ordered by key. This is why object-store table formats keep their own file list in a manifest instead of listing the bucket.',
    }),
  ];
}

const FAILURES: { key: keyof Backend['survives']; label: string; detail: string }[] = [
  { key: 'drive', label: 'One drive dies', detail: 'A single NAND device fails or is pulled.' },
  { key: 'host', label: 'Host dies or is stopped', detail: 'Instance terminates, or you stop/start it and it lands on new hardware.' },
  { key: 'az', label: 'Availability zone goes dark', detail: 'Power or network loss for a whole facility.' },
  { key: 'unflushed', label: 'Power loss after write(), before flush', detail: 'The bytes were in a page cache or a client buffer and never acknowledged as durable.' },
  { key: 'corrupt', label: 'Engine writes wrong/corrupt bytes', detail: 'Replication is faithful — it copies the corruption to every replica.' },
];

/* --------------------------------------------------------------- the steps */

function StepRows({
  steps,
  width,
  backend,
}: {
  steps: Step[];
  width: number;
  backend: Backend;
}) {
  const tip = useTip();
  const labelW = Math.min(215, Math.max(132, width * 0.25));
  const callW = Math.min(300, Math.max(165, width * 0.32));
  const valueW = 74;
  const barX = labelW + callW + 10;
  const barW = Math.max(90, width - barX - valueW - 6);
  const rowH = 30;
  const height = steps.length * rowH + 26;

  const lo = Math.log10(1e3); // 1 µs
  const hi = Math.log10(1e9); // 1 s
  const x = (ns: number) => (Math.max(0, Math.log10(Math.max(ns, 1e3)) - lo) / (hi - lo)) * barW;

  const decades = [1e3, 1e4, 1e5, 1e6, 1e7, 1e8, 1e9];

  return (
    <svg
      width={width}
      height={height}
      role="img"
      aria-label={`Engine write path on ${backend.label}, per step legality and latency`}
    >
      {decades.map((d) => (
        <g key={d}>
          <line className="viz-grid-line" x1={barX + x(d)} x2={barX + x(d)} y1={14} y2={height - 12} />
          <text x={barX + x(d)} y={10} textAnchor="middle" fontSize={10}>
            {fmtTime(d)}
          </text>
        </g>
      ))}

      {steps.map((s, i) => {
        const y = 20 + i * rowH;
        const w = s.p50 > 0 ? Math.max(2, x(s.p50)) : 0;
        const tailW = s.p99 > 0 ? Math.max(w + 1, x(s.p99)) : 0;
        return (
          <g
            key={s.id}
            style={{ cursor: 'help' }}
            {...tip(
              <>
                <strong>{s.want}</strong>
                <br />
                {s.call}
                <br />
                {MODE_LABEL[s.mode]} · p50 {fmtTime(s.p50)} · p99 {fmtTime(s.p99)}
                {s.put + s.get > 0 ? ` · ${s.put} PUT-class, ${s.get} GET-class` : ''}
                <br />
                <span style={{ color: 'var(--viz-ink-2)' }}>{s.note}</span>
              </>,
            )}
          >
            <rect x={0} y={y - 4} width={width} height={rowH - 2} fill="transparent" />
            <text x={0} y={y + 12} fill="var(--viz-ink)">
              {s.want.length > 34 ? s.want.slice(0, 33) + '…' : s.want}
            </text>
            <rect
              x={labelW}
              y={y + 1}
              width={6}
              height={15}
              rx={2}
              fill={MODE_COLOR[s.mode]}
            />
            <text x={labelW + 12} y={y + 12}>
              {s.call.length > 44 ? s.call.slice(0, 43) + '…' : s.call}
            </text>
            {tailW > 0 ? (
              <rect
                x={barX}
                y={y + 3}
                width={tailW}
                height={11}
                rx={3}
                fill={MODE_COLOR[s.mode]}
                fillOpacity={0.28}
              />
            ) : null}
            {w > 0 ? (
              <rect
                x={barX}
                y={y + 3}
                width={w}
                height={11}
                rx={3}
                fill={MODE_COLOR[s.mode]}
                stroke="var(--viz-surface)"
                strokeWidth={1}
              />
            ) : (
              <text x={barX} y={y + 12} fill="var(--viz-ink-muted)">
                —
              </text>
            )}
            <text
              x={barX + barW + 6}
              y={y + 12}
              fill="var(--viz-ink-2)"
              style={{ fontVariantNumeric: 'tabular-nums' }}
            >
              {s.p50 > 0 ? fmtTime(s.p50) : 'free'}
            </text>
          </g>
        );
      })}
    </svg>
  );
}

/* ------------------------------------------------------------ the failures */

function FailureMatrix({ width, selected }: { width: number; selected: string }) {
  const labelW = Math.min(250, Math.max(150, width * 0.3));
  const colW = Math.max(84, (width - labelW - 4) / BACKENDS.length);
  const rowH = 28;
  const height = FAILURES.length * rowH + 30;

  return (
    <svg
      width={Math.max(width, labelW + colW * BACKENDS.length)}
      height={height}
      role="img"
      aria-label="Which failures each storage backend survives"
    >
      {BACKENDS.map((b, c) => (
        <text
          key={b.id}
          x={labelW + c * colW + colW / 2}
          y={14}
          textAnchor="middle"
          fill={b.id === selected ? 'var(--viz-ink)' : 'var(--viz-ink-2)'}
          fontWeight={b.id === selected ? 700 : 400}
        >
          {b.short}
        </text>
      ))}
      {FAILURES.map((f, r) => {
        const y = 22 + r * rowH;
        return (
          <g key={f.key}>
            <text x={0} y={y + 17}>
              {f.label}
            </text>
            {BACKENDS.map((b, c) => {
              const ok = b.survives[f.key];
              const color = ok ? 'var(--viz-good)' : 'var(--viz-critical)';
              return (
                <g key={b.id}>
                  <rect
                    x={labelW + c * colW + 3}
                    y={y + 3}
                    width={colW - 6}
                    height={rowH - 8}
                    rx={4}
                    fill={color}
                    fillOpacity={b.id === selected ? 0.24 : 0.1}
                    stroke={color}
                    strokeWidth={b.id === selected ? 1.5 : 0.75}
                  />
                  <text
                    x={labelW + c * colW + colW / 2}
                    y={y + 17}
                    textAnchor="middle"
                    fill="var(--viz-ink)"
                    fontSize={10}
                  >
                    {ok ? 'survives' : 'data lost'}
                  </text>
                </g>
              );
            })}
          </g>
        );
      })}
    </svg>
  );
}

/* ----------------------------------------------------------------- the lab */

const SEC_PER_MONTH = 2_592_000;

export default function StorageBackendWritePath() {
  const [backendId, setBackendId] = useState('nvme');
  const [batchExp, setBatchExp] = useState(14); // 16 KB
  const [segExp, setSegExp] = useState(26); // 64 MB
  const [rateExp, setRateExp] = useState(2.5); // ~316 commits/s
  const [ref, width] = useSize(760);

  const backend = BACKENDS.find((b) => b.id === backendId)!;
  const batch = 2 ** batchExp;
  const segment = 2 ** segExp;
  const rate = Math.round(10 ** rateExp);

  const steps = useMemo(() => plan(backend, batch, segment), [backend, batch, segment]);

  const commit = steps.filter((s) => s.group === 'commit');
  const commitP50 = commit.reduce((a, s) => a + s.p50, 0);
  const commitP99 = commit.reduce((a, s) => a + s.p99, 0);
  const commitPut = commit.reduce((a, s) => a + s.put, 0);
  const commitGet = commit.reduce((a, s) => a + s.get, 0);
  const costPerCommit = commitPut * backend.putPrice + commitGet * backend.getPrice;
  const pageStep = steps.find((s) => s.id === 'page')!;

  const opsPerCommit = Math.max(1, commitPut + commitGet);
  const opsUtil = (rate * opsPerCommit) / backend.capOps;
  const bwUtil = (rate * batch) / (backend.capBwMBs * 1e6);
  const util = Math.max(opsUtil, bwUtil);
  const binding = bwUtil > opsUtil ? 'bandwidth' : backend.capLabel;

  const monthlyRequests = costPerCommit * rate * SEC_PER_MONTH;
  const monthlyStorage1TB = backend.gbMonth * 1024;

  return (
    <VizPanel
      title="The same write path on five substrates"
      subtitle="Pick a backend and watch each step of an engine's commit path get re-annotated: what call is actually legal, what it costs in latency, and what it costs in dollars."
      controls={
        <>
          <Choice
            label="Backend"
            value={backendId}
            onChange={setBackendId}
            options={BACKENDS.map((b) => ({ value: b.id, label: b.label }))}
          />
          <Slider
            label="Commit batch"
            min={12}
            max={22}
            value={batchExp}
            onChange={setBatchExp}
            format={() => fmtBytes(batch)}
          />
          <Slider
            label="Segment / object size"
            min={20}
            max={30}
            value={segExp}
            onChange={setSegExp}
            format={() => fmtBytes(segment)}
          />
          <Slider
            label="Commits/sec"
            min={0}
            max={4.7}
            step={0.1}
            value={rateExp}
            onChange={setRateExp}
            format={() => fmtNum(rate)}
          />
        </>
      }
      legend={
        <Legend
          items={[
            { label: 'Native — the call the engine wanted', color: MODE_COLOR.native },
            { label: 'Rewritten — emulated by other calls', color: MODE_COLOR.rewritten },
            { label: 'No equivalent on this backend', color: MODE_COLOR.absent },
            { label: 'Faded bar = p99 tail', color: 'var(--viz-ink-muted)' },
          ]}
        />
      }
      stats={
        <Stats
          items={[
            { label: 'Durable commit (p50)', value: fmtTime(commitP50), hint: 'Append + make it durable' },
            { label: 'Durable commit (p99)', value: fmtTime(commitP99), hint: 'Deterministic tail model' },
            {
              label: '4 KB in-place update moves',
              value: fmtBytes(pageStep.bytes),
              hint: 'Bytes crossing the bus or the wire to change 4 KB',
            },
            {
              label: '$ / million commits',
              value: costPerCommit === 0 ? 'no per-request charge' : `$${(costPerCommit * 1e6).toFixed(2)}`,
            },
            {
              label: 'Storage $ / GB-month',
              value: `$${backend.gbMonth.toFixed(3)}`,
              hint:
                backend.kind === 'object'
                  ? 'us-east-1 list price'
                  : 'Instance-store capacity is bundled with the instance; shown as its amortized share. RAID 10 doubles it because half the raw capacity is the mirror.',
            },
            {
              label: `Ceiling used (${binding})`,
              value: `${util >= 10 ? fmtNum(util * 100) : (util * 100).toFixed(1)}%`,
              hint: `${backend.capLabel}: ${fmtNum(backend.capOps)} ops/s, ${fmtNum(backend.capBwMBs)} MB/s`,
            },
          ]}
        />
      }
      note={
        <>
          <strong>{backend.label}: {backend.api}.</strong>{' '}
          {backend.kind === 'object' ? (
            <>
              Nothing is durable until a whole object is PUT, so the commit path is one network
              request with a {fmtTime(commitP99)} tail, and patching 4 KB in place costs{' '}
              {fmtBytes(pageStep.bytes)} of transfer plus two billed requests. At {fmtNum(rate)}{' '}
              commits/s the request bill alone is ${fmtNum(monthlyRequests, 0)}/month, against $
              {fmtNum(monthlyStorage1TB, 0)}/month to store 1 TB — requests, not bytes, are the
              budget, and that is what forces big immutable objects and log-structured layouts.
            </>
          ) : (
            <>
              Every call the engine wants is legal, including overwriting a page where it already
              lives. At {fmtNum(rate)} commits/s you are using {(util * 100).toFixed(0)}% of the{' '}
              {binding} ceiling
              {util > 1 ? ' — over it, so commits queue and p99 climbs without bound' : ''}. Storing
              1 TB costs ${fmtNum(monthlyStorage1TB, 0)}/month and no request charge applies.
            </>
          )}
        </>
      }
      table={
        <>
          <table className="viz-table">
            <caption style={{ textAlign: 'left' }}>Write path on {backend.label}</caption>
            <thead>
              <tr>
                <th>Step</th>
                <th>Actual call</th>
                <th>Legality</th>
                <th>p50</th>
                <th>p99</th>
                <th>Bytes moved</th>
                <th>Billed requests</th>
                <th>Cumulative $ / 1M paths</th>
              </tr>
            </thead>
            <tbody>
              {steps.map((s, i) => {
                const cum = steps
                  .slice(0, i + 1)
                  .reduce((a, t) => a + t.put * backend.putPrice + t.get * backend.getPrice, 0);
                return (
                  <tr key={s.id}>
                    <td>{s.want}</td>
                    <td>{s.call}</td>
                    <td>{MODE_LABEL[s.mode]}</td>
                    <td>{s.p50 > 0 ? fmtTime(s.p50) : '—'}</td>
                    <td>{s.p99 > 0 ? fmtTime(s.p99) : '—'}</td>
                    <td>{s.bytes > 0 ? fmtBytes(s.bytes) : '—'}</td>
                    <td>{s.put + s.get > 0 ? `${s.put} PUT, ${s.get} GET` : 'none'}</td>
                    <td>${(cum * 1e6).toFixed(2)}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
          <table className="viz-table">
            <caption style={{ textAlign: 'left' }}>Backends at the current settings</caption>
            <thead>
              <tr>
                <th>Backend</th>
                <th>Commit p50</th>
                <th>Commit p99</th>
                <th>4 KB update moves</th>
                <th>$ / 1M commits</th>
                <th>$ / GB-month</th>
                <th>Sustained ceiling</th>
              </tr>
            </thead>
            <tbody>
              {BACKENDS.map((b) => {
                const st = plan(b, batch, segment);
                const c = st.filter((s) => s.group === 'commit');
                const p50 = c.reduce((a, s) => a + s.p50, 0);
                const p99 = c.reduce((a, s) => a + s.p99, 0);
                const dollars =
                  c.reduce((a, s) => a + s.put * b.putPrice + s.get * b.getPrice, 0) * 1e6;
                return (
                  <tr key={b.id}>
                    <td>{b.label}</td>
                    <td>{fmtTime(p50)}</td>
                    <td>{fmtTime(p99)}</td>
                    <td>{fmtBytes(st.find((s) => s.id === 'page')!.bytes)}</td>
                    <td>${dollars.toFixed(2)}</td>
                    <td>${b.gbMonth.toFixed(3)}</td>
                    <td>
                      {fmtNum(b.capOps)} {b.capLabel} · {fmtNum(b.capBwMBs)} MB/s
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
          <table className="viz-table">
            <caption style={{ textAlign: 'left' }}>What each failure actually does</caption>
            <thead>
              <tr>
                <th>Failure</th>
                {BACKENDS.map((b) => (
                  <th key={b.id}>{b.short}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {FAILURES.map((f) => (
                <tr key={f.key}>
                  <td title={f.detail}>{f.label}</td>
                  {BACKENDS.map((b) => (
                    <td key={b.id}>{b.survives[f.key] ? 'survives' : 'data lost'}</td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </>
      }
    >
      <div ref={ref}>
        <TooltipHost>
          <StepRows steps={steps} width={width} backend={backend} />
        </TooltipHost>
        <p className="viz-sub" style={{ marginTop: '0.75rem' }}>
          What each substrate survives — and the two rows at the bottom that none of them do.
        </p>
        <FailureMatrix width={width} selected={backendId} />
      </div>
    </VizPanel>
  );
}
