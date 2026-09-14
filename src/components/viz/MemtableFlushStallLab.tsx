import { useRef, useState } from 'react';
import {
  VizPanel,
  Slider,
  Choice,
  Button,
  Legend,
  Stats,
  Note,
  TooltipHost,
  useTicker,
  useTip,
  fmtBytes,
  fmtNum,
  fmtTime,
  makeRng,
  useSize,
} from './Viz';

/**
 * The RocksDB write path, running.
 *
 * One client write becomes: a WAL record (durable or not, depending on WriteOptions::sync),
 * a sequence number, and a skiplist insert into the active memtable. At write_buffer_size
 * the memtable is switched — it becomes immutable, a fresh one and a fresh WAL open — and a
 * background flush turns it into one L0 SSTable. Everything downstream of that is a queue,
 * and every queue has a depth at which RocksDB stops accepting writes:
 *
 *   imm.NumNotFlushed() >= max_write_buffer_number   -> stop
 *   L0 files            >= level0_stop_writes_trigger -> stop
 *   pending bytes       >= hard_pending_compaction_bytes_limit -> stop
 *   ... and one trigger below each of those, a multiplicative slowdown pinned to
 *   delayed_write_rate (16 MB/s by default).
 *
 * The simulation advances in fixed 50 ms steps so the state sequence is deterministic
 * regardless of frame timing; key values and skiplist heights come from makeRng tables.
 */

/* ------------------------------------------------------------------ constants */

const ENTRY_USER = 136; // 16 B key + 112 B value + the 8-byte internal-key trailer
const ENTRY_MEM = 176; // the same entry once it carries skiplist pointers in the arena
const ENTRY_WAL = 148; // WAL record: header + kTypeValue + varint-framed key and value
const STEP_MS = 50; // one fixed simulation step
const SIM_SPEED = 0.25; // sim seconds per real second — slow enough to watch a flush
const MB = 1_000_000;
const DELAYED_WRITE_RATE = 16 * MB; // RocksDB's default delayed_write_rate
const SLOWDOWN_STEP = 0.8; // the rate is stepped down as the backlog deepens
const MIN_DELAYED = 0.25 * MB;
const WRITEBACK_LAG = 30; // dirty_expire_centisecs = 3000 -> 30 s
const FSYNC_NS = 250_000; // one barrier on an NVMe SSD with a volatile cache
const MAX_NODES = 16; // skiplist nodes we can draw
const MAX_LEVELS = 4; // drawn levels; RocksDB's InlineSkipList allows 12
const MAX_EVENTS = 36;

/** Deterministic key stream and skiplist coin flips (kBranching = 4, p = 1/4). */
const RNG_KEYS = (() => {
  const rng = makeRng(20130305); // RocksDB's first public release
  return Array.from({ length: 997 }, () => Math.floor(rng() * 1000));
})();
const RNG_HEIGHTS = (() => {
  const rng = makeRng(4242);
  return Array.from({ length: 997 }, () => {
    let h = 1;
    while (rng() < 0.25 && h < MAX_LEVELS) h++;
    return h;
  });
})();

/* ---------------------------------------------------------------------- state */

type Sync = 'buffered' | 'sync' | 'none';

type Imm = { id: number; bytes: number; entries: number; flushed: number };
type L0File = { id: number; bytes: number; entries: number; seqHi: number };
type SkipNode = { key: number; h: number; seq: number };
type WalFile = { id: number; bytes: number; active: boolean };
type Ev = { t: number; kind: string; detail: string };
type Stall = 'ok' | 'slow' | 'stop';

type S = {
  t: number;
  seq: number;
  writes: number;
  nodeN: number;
  acc: number; // fractional entries carried between steps
  memBytes: number;
  memEntries: number;
  memId: number;
  nodes: SkipNode[];
  wal: WalFile[];
  walSynced: number; // bytes of the active WAL known to be on the device
  hist: { t: number; b: number }[]; // (sim time, cumulative WAL bytes) for writeback lag
  walTotal: number;
  imms: Imm[];
  l0: L0File[];
  drainAcc: number;
  nextMemId: number;
  nextFileId: number;
  stall: Stall;
  reason: string;
  okT: number;
  slowT: number;
  stopT: number;
  allowed: number; // bytes/s the WriteController is letting through right now
  evs: Ev[];
};

const INITIAL: S = {
  t: 0,
  seq: 0,
  writes: 0,
  nodeN: 0,
  acc: 0,
  memBytes: 0,
  memEntries: 0,
  memId: 1,
  nodes: [],
  wal: [{ id: 1, bytes: 0, active: true }],
  walSynced: 0,
  hist: [{ t: 0, b: 0 }],
  walTotal: 0,
  imms: [],
  l0: [],
  drainAcc: 0,
  nextMemId: 2,
  nextFileId: 1,
  stall: 'ok',
  reason: 'Writes are flowing at the requested rate.',
  okT: 0,
  slowT: 0,
  stopT: 0,
  allowed: 0,
  evs: [],
};

type Cfg = {
  rate: number; // client write rate, bytes/s
  bufBytes: number; // write_buffer_size
  maxWbn: number; // max_write_buffer_number
  flushBps: number; // aggregate flush throughput
  drainBps: number; // L0 -> base compaction drain
  l0Slow: number; // level0_slowdown_writes_trigger
  l0Stop: number; // level0_stop_writes_trigger
  softPending: number; // soft_pending_compaction_bytes_limit
  sync: Sync;
};

/* ------------------------------------------------------------- the stall check */

type Verdict = { stall: Stall; reason: string; allowed: number };

/** RocksDB's RecalculateWriteStallConditions, in the order the real one tests. */
function verdict(s: S, cfg: Cfg): Verdict {
  const pending = s.l0.reduce((n, f) => n + f.bytes, 0);
  const hard = cfg.softPending * 4; // RocksDB ships 64 GB soft / 256 GB hard
  const imm = s.imms.length;

  if (imm >= cfg.maxWbn)
    return {
      stall: 'stop',
      reason: `Stopped: ${imm} memtable${imm === 1 ? '' : 's'} queued for flush, and max_write_buffer_number is ${cfg.maxWbn}. Nothing can be written until a flush finishes — flush bandwidth is now your write bandwidth.`,
      allowed: 0,
    };
  if (s.l0.length >= cfg.l0Stop)
    return {
      stall: 'stop',
      reason: `Stopped: ${s.l0.length} L0 files >= level0_stop_writes_trigger (${cfg.l0Stop}). Reads would have to open every one of them, so RocksDB refuses new writes until compaction drains L0.`,
      allowed: 0,
    };
  if (pending >= hard)
    return {
      stall: 'stop',
      reason: `Stopped: ${fmtBytes(pending)} of compaction debt >= hard_pending_compaction_bytes_limit (${fmtBytes(hard)}).`,
      allowed: 0,
    };

  let over = 0;
  let why = '';
  if (cfg.maxWbn > 3 && imm >= cfg.maxWbn - 1) {
    over = Math.max(over, 1);
    why = `${imm} memtables are queued and max_write_buffer_number is ${cfg.maxWbn}`;
  }
  if (s.l0.length >= cfg.l0Slow) {
    over = Math.max(over, s.l0.length - cfg.l0Slow + 1);
    why = `${s.l0.length} L0 files >= level0_slowdown_writes_trigger (${cfg.l0Slow})`;
  }
  if (pending >= cfg.softPending) {
    over = Math.max(over, Math.floor(pending / cfg.softPending));
    why = `${fmtBytes(pending)} of compaction debt >= soft_pending_compaction_bytes_limit (${fmtBytes(cfg.softPending)})`;
  }
  if (over > 0) {
    const delayed = Math.max(MIN_DELAYED, DELAYED_WRITE_RATE * Math.pow(SLOWDOWN_STEP, over - 1));
    if (delayed < cfg.rate)
      return {
        stall: 'slow',
        reason: `Slowed down: ${why}. The WriteController is pacing every writer to ${fmtBytes(delayed)}/s — writes now sleep inside Put() instead of returning.`,
        allowed: delayed,
      };
  }
  return { stall: 'ok', reason: 'Writes are flowing at the requested rate.', allowed: cfg.rate };
}

/* --------------------------------------------------------------- the simulator */

function step(s: S, cfg: Cfg): S {
  const dt = (STEP_MS / 1000) * SIM_SPEED;
  const v = verdict(s, cfg);
  const t = s.t + dt;

  const evs: Ev[] = [];
  if (v.stall !== s.stall) {
    evs.push({
      t,
      kind: v.stall === 'ok' ? 'resume' : v.stall === 'slow' ? 'slowdown' : 'stop',
      detail: v.reason.replace(/^(Stopped|Slowed down): /, ''),
    });
  }

  /* --- admitted writes ------------------------------------------------- */
  const acc = s.acc + (v.allowed * dt) / ENTRY_USER;
  const n = Math.floor(acc);
  let memBytes = s.memBytes + n * ENTRY_MEM;
  let memEntries = s.memEntries + n;
  let nodes = s.nodes;
  let nodeN = s.nodeN;
  const seq = s.seq + n;

  if (n > 0) {
    const key = RNG_KEYS[nodeN % RNG_KEYS.length];
    const h = RNG_HEIGHTS[nodeN % RNG_HEIGHTS.length];
    nodeN++;
    nodes = [...nodes.filter((x) => x.key !== key), { key, h, seq }]
      .slice(-MAX_NODES)
      .sort((a, b) => a.key - b.key);
  }

  /* --- the WAL --------------------------------------------------------- */
  let wal = s.wal;
  let walTotal = s.walTotal;
  let walSynced = s.walSynced;
  let hist = s.hist;
  if (cfg.sync !== 'none' && n > 0) {
    const add = n * ENTRY_WAL;
    walTotal += add;
    wal = wal.map((f) => (f.active ? { ...f, bytes: f.bytes + add } : f));
    hist = [...s.hist, { t, b: walTotal }].filter((p) => p.t >= t - WRITEBACK_LAG - 1);
  }
  const active = wal.find((f) => f.active)!;
  if (cfg.sync === 'sync') {
    walSynced = active.bytes;
  } else if (cfg.sync === 'buffered') {
    const old = hist.filter((p) => p.t <= t - WRITEBACK_LAG).pop();
    const backTotal = old ? old.b : 0;
    walSynced = Math.max(0, Math.min(active.bytes, backTotal - (walTotal - active.bytes)));
  } else {
    walSynced = 0;
  }

  /* --- memtable switch ------------------------------------------------- */
  let imms = s.imms;
  let memId = s.memId;
  let nextMemId = s.nextMemId;
  if (memBytes >= cfg.bufBytes && v.stall !== 'stop') {
    imms = [...imms, { id: memId, bytes: memBytes, entries: memEntries, flushed: 0 }];
    evs.push({
      t,
      kind: 'switch',
      detail: `memtable #${memId} hit write_buffer_size (${fmtBytes(memBytes)}, ${fmtNum(memEntries)} entries) — switched to immutable, memtable #${nextMemId} and WAL ${String(nextMemId).padStart(6, '0')}.log opened`,
    });
    memId = nextMemId;
    nextMemId++;
    memBytes = 0;
    memEntries = 0;
    nodes = [];
    wal = [...wal.map((f) => ({ ...f, active: false })), { id: memId, bytes: 0, active: true }];
    walSynced = 0;
  }

  /* --- background flush ------------------------------------------------ */
  let l0 = s.l0;
  let nextFileId = s.nextFileId;
  if (imms.length > 0) {
    const head = imms[0];
    const done = head.flushed + cfg.flushBps * dt;
    if (done >= head.bytes) {
      const bytes = head.entries * ENTRY_USER;
      l0 = [...l0, { id: nextFileId, bytes, entries: head.entries, seqHi: seq }];
      evs.push({
        t,
        kind: 'flush',
        detail: `memtable #${head.id} flushed to L0 file ${String(nextFileId).padStart(6, '0')}.sst (${fmtBytes(bytes)}) — WAL ${String(head.id).padStart(6, '0')}.log is now deletable`,
      });
      nextFileId++;
      imms = imms.slice(1);
      wal = wal.filter((f) => f.active || f.id > head.id);
    } else {
      imms = [{ ...head, flushed: done }, ...imms.slice(1)];
    }
  }

  /* --- L0 drain (compaction: the next module owns the strategy) --------- */
  let drainAcc = s.drainAcc + cfg.drainBps * dt;
  while (l0.length > 0 && drainAcc >= l0[0].bytes) {
    drainAcc -= l0[0].bytes;
    l0 = l0.slice(1);
  }
  if (l0.length === 0) drainAcc = 0;

  return {
    ...s,
    t,
    seq,
    writes: s.writes + n,
    nodeN,
    acc: acc - n,
    memBytes,
    memEntries,
    memId,
    nodes,
    wal,
    walSynced,
    walTotal,
    hist,
    imms,
    l0,
    drainAcc,
    nextMemId,
    nextFileId,
    stall: v.stall,
    reason: v.reason,
    allowed: v.allowed,
    okT: s.okT + (v.stall === 'ok' ? dt : 0),
    slowT: s.slowT + (v.stall === 'slow' ? dt : 0),
    stopT: s.stopT + (v.stall === 'stop' ? dt : 0),
    evs: [...s.evs, ...evs].slice(-MAX_EVENTS),
  };
}

/* ------------------------------------------------------------- the component */

const SYNCS: { value: Sync; label: string }[] = [
  { value: 'buffered', label: 'WAL, buffered (default)' },
  { value: 'sync', label: 'WAL, sync per group' },
  { value: 'none', label: 'disableWAL = true' },
];

const PENDING: { value: string; label: string }[] = [
  { value: String(256 * MB), label: '256 MB' },
  { value: String(4_000 * MB), label: '4 GB' },
  { value: String(64_000 * MB), label: '64 GB (default)' },
];

export default function MemtableFlushStallLab() {
  const [rateMb, setRateMb] = useState(240);
  const [bufMb, setBufMb] = useState(64);
  const [maxWbn, setMaxWbn] = useState(2);
  const [flushMb, setFlushMb] = useState(180);
  const [drainMb, setDrainMb] = useState(110);
  const [l0Slow, setL0Slow] = useState(20);
  const [l0Stop, setL0Stop] = useState(36);
  const [soft, setSoft] = useState(String(64_000 * MB));
  const [sync, setSync] = useState<Sync>('buffered');
  const [running, setRunning] = useState(true);
  const [s, setS] = useState<S>(INITIAL);
  const [ref, width] = useSize(760);
  const tip = useTip();
  const carry = useRef(0);

  const cfg: Cfg = {
    rate: rateMb * MB,
    bufBytes: bufMb * MB,
    maxWbn,
    flushBps: flushMb * MB,
    drainBps: drainMb * MB,
    l0Slow,
    l0Stop: Math.max(l0Stop, l0Slow + 1),
    softPending: Number(soft),
    sync,
  };

  useTicker((dtMs) => {
    carry.current = Math.min(carry.current + dtMs, STEP_MS * 8);
    let k = 0;
    while (carry.current >= STEP_MS && k < 8) {
      carry.current -= STEP_MS;
      k++;
    }
    if (k > 0)
      setS((cur) => {
        let next = cur;
        for (let i = 0; i < k; i++) next = step(next, cfg);
        return next;
      });
  }, running);

  /* ---- derived numbers ---- */
  const writesPerSec = s.allowed / ENTRY_USER;
  const syncsPerSec = sync === 'sync' ? Math.min(writesPerSec, 1e9 / FSYNC_NS) : 0;
  const groupSize = syncsPerSec > 0 ? writesPerSec / syncsPerSec : 0;
  const pending = s.l0.reduce((n, f) => n + f.bytes, 0);
  const memMemory = s.memBytes + s.imms.reduce((n, i) => n + i.bytes, 0);
  const activeWal = s.wal.find((f) => f.active)!;
  const window =
    sync === 'none' ? memMemory : sync === 'sync' ? 0 : Math.max(0, activeWal.bytes - s.walSynced);
  const elapsed = Math.max(s.t, 0.0001);
  const goodput = (s.writes * ENTRY_USER) / elapsed;

  /* ---- geometry ---- */
  const labelW = 152;
  const svgW = Math.max(width, 660);
  const x0 = labelW;
  const cw = svgW - labelW - 10;
  const rows = { wal: 12, mem: 74, imm: 206, l0: 274, gauge: 348 };
  const height = 400;

  const bufScale = cfg.bufBytes * (ENTRY_WAL / ENTRY_MEM);
  const slotCount = Math.max(cfg.l0Stop + 4, 24);
  const slotW = cw / slotCount;
  const stallColor =
    s.stall === 'stop' ? 'var(--viz-critical)' : s.stall === 'slow' ? 'var(--viz-serious)' : 'var(--viz-good)';

  const nodeXs = (i: number) => x0 + 46 + (i + 0.5) * ((cw - 56) / MAX_NODES);

  return (
    <VizPanel
      title="The write path, running: WAL → memtable → flush → L0 → stall"
      subtitle="Client writes stream in on the left and leave as L0 SSTables on the right. Every box between them is a queue with a configured maximum; cross one and RocksDB throttles the writer, cross the next and it stops them. The clock runs at ¼ speed."
      controls={
        <>
          <Slider
            label="Client write rate"
            min={20}
            max={800}
            step={10}
            value={rateMb}
            onChange={setRateMb}
            format={(n) => `${n} MB/s`}
          />
          <Slider
            label="write_buffer_size"
            min={8}
            max={256}
            step={8}
            value={bufMb}
            onChange={setBufMb}
            format={(n) => `${n} MB`}
          />
          <Slider
            label="max_write_buffer_number"
            min={2}
            max={8}
            step={1}
            value={maxWbn}
            onChange={setMaxWbn}
          />
          <Slider
            label="Flush throughput"
            min={20}
            max={600}
            step={10}
            value={flushMb}
            onChange={setFlushMb}
            format={(n) => `${n} MB/s`}
          />
          <Slider
            label="L0→L1 compaction drain"
            min={0}
            max={600}
            step={10}
            value={drainMb}
            onChange={setDrainMb}
            format={(n) => `${n} MB/s`}
          />
          <Slider
            label="level0_slowdown_writes_trigger"
            min={2}
            max={40}
            step={1}
            value={l0Slow}
            onChange={setL0Slow}
          />
          <Slider
            label="level0_stop_writes_trigger"
            min={3}
            max={60}
            step={1}
            value={l0Stop}
            onChange={setL0Stop}
            format={(n) => String(Math.max(n, l0Slow + 1))}
          />
          <Choice
            label="WAL policy"
            value={sync}
            onChange={(v) => setSync(v)}
            options={SYNCS}
          />
          <Choice
            label="soft_pending_compaction_bytes_limit"
            value={soft}
            onChange={setSoft}
            options={PENDING}
          />
          <Button onClick={() => setRunning(!running)} primary>
            {running ? 'Pause' : 'Run'}
          </Button>
          <Button onClick={() => setS((cur) => step(cur, cfg))} disabled={running}>
            Step 50 ms
          </Button>
          <Button
            onClick={() => {
              setS(INITIAL);
              carry.current = 0;
            }}
          >
            Reset
          </Button>
        </>
      }
      legend={
        <Legend
          items={[
            { label: 'WAL bytes still only in DRAM / page cache', color: 'var(--viz-dirty)' },
            { label: 'WAL bytes on the device', color: 'var(--viz-clean)' },
            { label: 'Immutable memtable awaiting flush', color: 'var(--viz-7)' },
            { label: 'Flushed so far', color: 'var(--viz-6)' },
            { label: 'L0 SSTable', color: 'var(--viz-1)' },
            { label: 'slowdown trigger', color: 'var(--viz-serious)', shape: 'line' },
            { label: 'stop trigger', color: 'var(--viz-critical)', shape: 'line' },
          ]}
        />
      }
      stats={
        <Stats
          items={[
            {
              label: 'Sequence number',
              value: fmtNum(s.seq),
              hint: 'One per key written, monotonic across the whole DB; packed with the value type into the 8-byte internal-key trailer',
            },
            {
              label: 'Admitted write rate',
              value: `${fmtBytes(s.allowed)}/s`,
              hint: 'What the WriteController is letting through right now, against the rate the client asked for',
            },
            {
              label: 'Memtable',
              value: `${fmtBytes(s.memBytes)} / ${fmtBytes(cfg.bufBytes)}`,
              hint: `${fmtNum(s.memEntries)} entries in memtable #${s.memId}`,
            },
            {
              label: 'Queued for flush',
              value: `${s.imms.length} / ${maxWbn}`,
              hint: 'Immutable memtables — at max_write_buffer_number, writes stop dead',
            },
            {
              label: 'L0 files',
              value: `${s.l0.length} → ${l0Slow} / ${cfg.l0Stop}`,
              hint: 'Current count against level0_slowdown_writes_trigger and level0_stop_writes_trigger',
            },
            {
              label: 'Compaction debt',
              value: fmtBytes(pending),
              hint: `Bytes sitting in L0 that owe a compaction, against a soft limit of ${fmtBytes(cfg.softPending)}`,
            },
            {
              label: 'Durability window',
              value: sync === 'sync' ? '0 B' : fmtBytes(window),
              hint:
                sync === 'sync'
                  ? 'Every write group is fsynced before Put() returns: a power cut loses nothing'
                  : sync === 'none'
                    ? 'disableWAL: everything in memory dies with the process, not just with the machine'
                    : 'WAL bytes written but not yet pushed to the device — lost on machine crash, survived on process crash',
            },
            {
              label: 'Write group size',
              value: sync === 'sync' ? `${fmtNum(groupSize)} writes/fsync` : '—',
              hint: 'Leader batching: one writer wins the group, writes everyone’s batches with one write() and one barrier',
            },
            {
              label: 'Sustained goodput',
              value: `${fmtBytes(goodput)}/s`,
              hint: 'User bytes actually accepted since reset, stalls included — the number your client sees',
            },
            {
              label: 'Time stalled',
              value: `${((100 * (s.slowT + s.stopT)) / elapsed).toFixed(0)}%`,
              hint: `${fmtTime(s.slowT * 1e9)} slowed, ${fmtTime(s.stopT * 1e9)} stopped`,
            },
          ]}
        />
      }
      note={<Note>{s.reason}</Note>}
      table={
        <>
          <table className="viz-table">
            <thead>
              <tr>
                <th>Sim time</th>
                <th>Event</th>
                <th>Detail</th>
              </tr>
            </thead>
            <tbody>
              {s.evs.length === 0 ? (
                <tr>
                  <td colSpan={3}>No memtable switch, flush or stall yet.</td>
                </tr>
              ) : (
                s.evs
                  .slice()
                  .reverse()
                  .map((e, i) => (
                    <tr key={`${e.t}-${i}`}>
                      <td>{e.t.toFixed(2)} s</td>
                      <td>{e.kind}</td>
                      <td>{e.detail}</td>
                    </tr>
                  ))
              )}
            </tbody>
          </table>
          <table className="viz-table">
            <thead>
              <tr>
                <th>Trigger</th>
                <th>RocksDB default</th>
                <th>Set here</th>
                <th>Now</th>
                <th>Effect when crossed</th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <td>max_write_buffer_number</td>
                <td>2</td>
                <td>{maxWbn}</td>
                <td>{s.imms.length} queued</td>
                <td>stop (slowdown one below it, only if it is &gt; 3)</td>
              </tr>
              <tr>
                <td>level0_slowdown_writes_trigger</td>
                <td>20</td>
                <td>{l0Slow}</td>
                <td>{s.l0.length} files</td>
                <td>pace every writer at delayed_write_rate</td>
              </tr>
              <tr>
                <td>level0_stop_writes_trigger</td>
                <td>36</td>
                <td>{cfg.l0Stop}</td>
                <td>{s.l0.length} files</td>
                <td>stop</td>
              </tr>
              <tr>
                <td>soft_pending_compaction_bytes_limit</td>
                <td>64 GB</td>
                <td>{fmtBytes(cfg.softPending)}</td>
                <td>{fmtBytes(pending)}</td>
                <td>slowdown; 4× that is the hard limit, which stops</td>
              </tr>
              <tr>
                <td>delayed_write_rate</td>
                <td>16 MB/s</td>
                <td>16 MB/s</td>
                <td>{fmtBytes(s.allowed)}/s</td>
                <td>the ceiling a slowed-down writer is paced to</td>
              </tr>
            </tbody>
          </table>
          <table className="viz-table">
            <thead>
              <tr>
                <th>Accounting used by this model</th>
                <th>Bytes</th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <td>User entry (16 B key + 112 B value + 8 B internal-key trailer)</td>
                <td>{ENTRY_USER} B</td>
              </tr>
              <tr>
                <td>Same entry charged against write_buffer_size (arena + skiplist pointers)</td>
                <td>{ENTRY_MEM} B</td>
              </tr>
              <tr>
                <td>Same entry as a WAL record</td>
                <td>{ENTRY_WAL} B</td>
              </tr>
              <tr>
                <td>fsync latency assumed for a write group</td>
                <td>{fmtTime(FSYNC_NS)}</td>
              </tr>
            </tbody>
          </table>
        </>
      }
    >
      <div ref={ref}>
        <TooltipHost>
          <svg
            width={svgW}
            height={height}
            role="img"
            aria-label="Live write path: WAL, memtable skiplist, immutable flush queue, L0 files and the admitted write rate"
          >
            {/* ------------------------------------------------- WAL row */}
            <text x={0} y={rows.wal + 16} fill="var(--viz-ink)" fontWeight={600}>
              Write-ahead log
            </text>
            <text x={0} y={rows.wal + 32} fill="var(--viz-ink-muted)">
              {sync === 'none' ? 'disabled' : `${s.wal.length} file(s) retained`}
            </text>
            <rect x={x0} y={rows.wal} width={cw} height={34} rx={6} fill="var(--viz-plane)" stroke="var(--viz-border)" />
            {(() => {
              if (sync === 'none') return null;
              let cx = x0 + 3;
              return s.wal.map((f) => {
                const w = Math.max(6, Math.min(cw - 6, (f.bytes / bufScale) * (cw - 6)));
                const sw = f.active ? Math.min(w, (s.walSynced / bufScale) * (cw - 6)) : w;
                const el = (
                  <g
                    key={f.id}
                    {...tip(
                      <>
                        <strong>{String(f.id).padStart(6, '0')}.log</strong>
                        <br />
                        {fmtBytes(f.bytes)}
                        {f.active
                          ? ` — ${fmtBytes(s.walSynced)} on the device, ${fmtBytes(Math.max(0, f.bytes - s.walSynced))} still only in DRAM.`
                          : ' — retained because the memtable it covers has not been flushed yet.'}
                      </>,
                    )}
                    style={{ cursor: 'help' }}
                  >
                    <rect x={cx} y={rows.wal + 4} width={w} height={26} rx={4} fill="var(--viz-dirty)" />
                    <rect x={cx} y={rows.wal + 4} width={Math.max(0, sw)} height={26} rx={4} fill="var(--viz-clean)" />
                    <rect
                      x={cx}
                      y={rows.wal + 4}
                      width={w}
                      height={26}
                      rx={4}
                      fill="none"
                      stroke="var(--viz-surface)"
                      strokeWidth={1.5}
                    />
                  </g>
                );
                cx += w + 3;
                return el;
              });
            })()}
            <text x={svgW - 12} y={rows.wal + 22} textAnchor="end" fill="var(--viz-ink-2)">
              {sync === 'none'
                ? 'no WAL record is written at all'
                : sync === 'sync'
                  ? `fsync per write group (${fmtNum(groupSize)} writes/barrier)`
                  : `write() only — ${fmtBytes(window)} not yet on the device`}
            </text>

            {/* ------------------------------------------- memtable row */}
            <text x={0} y={rows.mem + 16} fill="var(--viz-ink)" fontWeight={600}>
              Active memtable
            </text>
            <text x={0} y={rows.mem + 32} fill="var(--viz-ink-muted)">
              #{s.memId} · skiplist
            </text>
            <text x={0} y={rows.mem + 48} fill="var(--viz-ink-muted)">
              {fmtNum(s.memEntries)} entries
            </text>
            <rect
              x={x0}
              y={rows.mem}
              width={cw}
              height={112}
              rx={8}
              fill="var(--viz-plane)"
              stroke="var(--viz-border)"
            />
            {/* fill bar against write_buffer_size */}
            <rect
              x={x0 + 6}
              y={rows.mem + 96}
              width={Math.min(cw - 12, ((cw - 12) * s.memBytes) / cfg.bufBytes)}
              height={8}
              rx={4}
              fill="var(--viz-dirty)"
            />
            <rect
              x={x0 + 6}
              y={rows.mem + 96}
              width={cw - 12}
              height={8}
              rx={4}
              fill="none"
              stroke="var(--viz-axis)"
            />
            <text x={x0 + 8} y={rows.mem + 92} fill="var(--viz-ink-muted)">
              {fmtBytes(s.memBytes)} of write_buffer_size {fmtBytes(cfg.bufBytes)}
            </text>
            {/* skiplist lanes */}
            {Array.from({ length: MAX_LEVELS }).map((_, lv) => {
              const lane = MAX_LEVELS - 1 - lv; // draw level 0 at the bottom
              const y = rows.mem + 14 + lane * 18;
              const idx = s.nodes.map((nd, i) => ({ nd, i })).filter((p) => p.nd.h > lv);
              return (
                <g key={lv}>
                  <text x={x0 + 6} y={y + 4} fill="var(--viz-ink-muted)">
                    L{lv}
                  </text>
                  <line
                    x1={x0 + 30}
                    x2={idx.length ? nodeXs(idx[idx.length - 1].i) : x0 + 34}
                    y1={y}
                    y2={y}
                    stroke="var(--viz-axis)"
                    strokeWidth={1}
                  />
                  <rect x={x0 + 26} y={y - 5} width={10} height={10} rx={2} fill="var(--viz-ink-2)" />
                  {idx.map((p) => (
                    <rect
                      key={p.nd.key}
                      x={nodeXs(p.i) - 5}
                      y={y - 5}
                      width={10}
                      height={10}
                      rx={2}
                      fill={p.nd.seq === s.seq ? 'var(--viz-2)' : 'var(--viz-dirty)'}
                      opacity={p.nd.seq === s.seq ? 1 : 0.75}
                    />
                  ))}
                </g>
              );
            })}
            {s.nodes.map((nd, i) => (
              <g
                key={`k${nd.key}`}
                {...tip(
                  <>
                    <strong>key {String(nd.key).padStart(4, '0')}</strong>
                    <br />
                    height {nd.h}, sequence number {fmtNum(nd.seq)}. The skiplist keeps it in sorted
                    order, which is why the flush can write the SSTable in one linear pass.
                  </>,
                )}
                style={{ cursor: 'help' }}
              >
                <text x={nodeXs(i)} y={rows.mem + 82} textAnchor="middle" fill="var(--viz-ink-2)">
                  {String(nd.key).padStart(4, '0')}
                </text>
              </g>
            ))}
            {s.nodes.length === 0 ? (
              <text x={x0 + 44} y={rows.mem + 60} fill="var(--viz-ink-muted)">
                empty — a fresh memtable and a fresh WAL
              </text>
            ) : null}

            {/* --------------------------------------- immutable queue */}
            <text x={0} y={rows.imm + 16} fill="var(--viz-ink)" fontWeight={600}>
              Queued for flush
            </text>
            <text x={0} y={rows.imm + 32} fill="var(--viz-ink-muted)">
              max {maxWbn}
            </text>
            <rect x={x0} y={rows.imm} width={cw} height={50} rx={8} fill="var(--viz-plane)" stroke="var(--viz-border)" />
            {Array.from({ length: maxWbn }).map((_, i) => {
              const slot = Math.min(150, (cw - 12) / maxWbn - 6);
              const bx = x0 + 6 + i * (slot + 6);
              const im = s.imms[i];
              return (
                <g key={i}>
                  <rect
                    x={bx}
                    y={rows.imm + 8}
                    width={slot}
                    height={34}
                    rx={5}
                    fill="var(--viz-neutral)"
                    stroke="var(--viz-axis)"
                    strokeDasharray="3 3"
                  />
                  {im ? (
                    <g
                      {...tip(
                        <>
                          <strong>immutable memtable #{im.id}</strong>
                          <br />
                          {fmtBytes(im.bytes)}, {fmtNum(im.entries)} entries — {fmtBytes(im.flushed)}{' '}
                          written. Its WAL cannot be deleted until this flush completes.
                        </>,
                      )}
                      style={{ cursor: 'help' }}
                    >
                      <rect x={bx} y={rows.imm + 8} width={slot} height={34} rx={5} fill="var(--viz-7)" />
                      <rect
                        x={bx}
                        y={rows.imm + 8}
                        width={(slot * im.flushed) / im.bytes}
                        height={34}
                        rx={5}
                        fill="var(--viz-6)"
                      />
                      <text x={bx + 8} y={rows.imm + 29} fill="var(--viz-surface)">
                        #{im.id} {i === 0 ? `flushing ${((100 * im.flushed) / im.bytes).toFixed(0)}%` : 'waiting'}
                      </text>
                    </g>
                  ) : null}
                </g>
              );
            })}

            {/* ---------------------------------------------- L0 files */}
            <text x={0} y={rows.l0 + 16} fill="var(--viz-ink)" fontWeight={600}>
              Level 0
            </text>
            <text x={0} y={rows.l0 + 32} fill="var(--viz-ink-muted)">
              {s.l0.length} SSTables
            </text>
            <text x={0} y={rows.l0 + 48} fill="var(--viz-ink-muted)">
              {fmtBytes(pending)}
            </text>
            <rect x={x0} y={rows.l0} width={cw} height={52} rx={8} fill="var(--viz-plane)" stroke="var(--viz-border)" />
            {s.l0.slice(-slotCount).map((f, i) => (
              <g
                key={f.id}
                {...tip(
                  <>
                    <strong>{String(f.id).padStart(6, '0')}.sst</strong>
                    <br />
                    {fmtBytes(f.bytes)}, {fmtNum(f.entries)} entries, newest sequence number{' '}
                    {fmtNum(f.seqHi)}. L0 files overlap in key range, so a read has to look in every
                    one of them.
                  </>,
                )}
                style={{ cursor: 'help' }}
              >
                <rect
                  x={x0 + 3 + i * slotW}
                  y={rows.l0 + 10}
                  width={Math.max(2, slotW - 2)}
                  height={32}
                  rx={2}
                  fill="var(--viz-1)"
                />
              </g>
            ))}
            <line
              x1={x0 + 3 + l0Slow * slotW}
              x2={x0 + 3 + l0Slow * slotW}
              y1={rows.l0 + 2}
              y2={rows.l0 + 50}
              stroke="var(--viz-serious)"
              strokeWidth={2}
              strokeDasharray="4 3"
            />
            <text x={x0 + 7 + l0Slow * slotW} y={rows.l0 + 60} fill="var(--viz-serious)">
              slowdown {l0Slow}
            </text>
            <line
              x1={x0 + 3 + cfg.l0Stop * slotW}
              x2={x0 + 3 + cfg.l0Stop * slotW}
              y1={rows.l0 + 2}
              y2={rows.l0 + 50}
              stroke="var(--viz-critical)"
              strokeWidth={2}
              strokeDasharray="4 3"
            />
            <text x={x0 + 7 + cfg.l0Stop * slotW} y={rows.l0 + 60} fill="var(--viz-critical)">
              stop {cfg.l0Stop}
            </text>

            {/* -------------------------------------------- rate gauge */}
            <text x={0} y={rows.gauge + 16} fill="var(--viz-ink)" fontWeight={600}>
              Admitted rate
            </text>
            <text x={0} y={rows.gauge + 32} fill={stallColor}>
              {s.stall === 'ok' ? 'no stall' : s.stall === 'slow' ? 'slowdown' : 'STOPPED'}
            </text>
            <rect
              x={x0}
              y={rows.gauge}
              width={(cw * rateMb) / 800}
              height={30}
              rx={6}
              fill="none"
              stroke="var(--viz-axis)"
              strokeDasharray="4 3"
            />
            <rect
              x={x0}
              y={rows.gauge}
              width={Math.max(1, (cw * (s.allowed / MB)) / 800)}
              height={30}
              rx={6}
              fill={stallColor}
            />
            <text x={x0 + 8} y={rows.gauge + 20} fill="var(--viz-surface)" fontWeight={600}>
              {fmtBytes(s.allowed)}/s
            </text>
            <text x={x0 + 8 + (cw * rateMb) / 800} y={rows.gauge + 20} fill="var(--viz-ink-muted)">
              requested {rateMb} MB/s
            </text>
          </svg>
        </TooltipHost>
      </div>
    </VizPanel>
  );
}
