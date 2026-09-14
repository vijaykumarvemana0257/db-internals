import { useState } from 'react';
import {
  VizPanel,
  Segmented,
  Choice,
  Check,
  Slider,
  Button,
  Legend,
  Stats,
  Note,
  TooltipHost,
  useTip,
  fmtBytes,
  fmtTime,
  useSize,
  makeRng,
} from './Viz';

/**
 * What one SQL row write actually costs on an LSM.
 *
 * The KV API appends blindly. The *table* above it does not: a unique constraint, or a
 * secondary index whose old entry has to be removed, forces a point lookup through the
 * memtable, the immutable memtables, every overlapping L0 file and one file per level —
 * the full read path, filters and block cache included — before the Put is allowed.
 *
 * Columns are probes, rows are the sources each probe must consult. A probe for a key
 * that is absent (a unique check) can never stop early: it pays for the whole descent.
 * Switching a counter column from read-modify-write to a Merge operand deletes the
 * column entirely — the read moves to read time, where compaction can amortize it.
 */

const MEM_NS = 320; // one skiplist search in a memtable
const FILTER_NS = 260; // one Bloom probe against a filter block held by the table reader
const CACHE_NS = 1_400; // the data block was already uncompressed in the block cache
const IO_NS = 25_000; // 16 KB block off NVMe: the read plus LZ4 decompression
const WRITE_NS = 700; // one KV into the WAL buffer and the skiplist
const MERGE_NS = 90; // applying one merge operand at read time

const ROW_B = 400;
const IDX_B = 48;
const TOMB_B = 24;
const OPERAND_B = 24;

/** Deterministic draws for filter false positives and block-cache hits. */
const RND = (() => {
  const r = makeRng(20131121);
  return Array.from({ length: 1024 }, () => r());
})();
const pick = (a: number, b: number, c: number) => RND[(a * 127 + b * 31 + c * 7 + 11) % 1024];

/* ------------------------------------------------------------------ config */

type Op = 'insert' | 'update' | 'delete' | 'counter';
type Home = 'mem' | 'l0' | 'mid' | 'deep';
type Strategy = 'rmw' | 'merge';

type Cfg = {
  op: Op;
  pkUnique: boolean;
  k: number;
  skUnique: boolean;
  strategy: Strategy;
  l0: number;
  bits: number;
  home: Home;
  hotCache: boolean;
};

type Src = { id: string; label: string; sub: string; kind: 'mem' | 'sst' };

function sources(l0: number): Src[] {
  const out: Src[] = [
    { id: 'mem', label: 'Active memtable', sub: 'skiplist, no filter', kind: 'mem' },
    { id: 'imm', label: 'Immutable memtable', sub: 'flush pending', kind: 'mem' },
  ];
  for (let i = 0; i < l0; i++) {
    out.push({
      id: `l0-${i}`,
      label: `L0 file ${l0 - i}`,
      sub: i === 0 ? 'newest — L0 ranges overlap' : 'overlaps, so it is probed too',
      kind: 'sst',
    });
  }
  out.push({ id: 'l1', label: 'L1', sub: '256 MB — one file by key range', kind: 'sst' });
  out.push({ id: 'l2', label: 'L2', sub: '2.5 GB — one file by key range', kind: 'sst' });
  out.push({ id: 'l3', label: 'L3', sub: '25 GB — one file by key range', kind: 'sst' });
  out.push({ id: 'l4', label: 'L4 — bottommost', sub: '250 GB — one file by key range', kind: 'sst' });
  return out;
}

function homeId(c: Cfg): string {
  if (c.home === 'mem') return 'mem';
  if (c.home === 'l0') return c.l0 > 0 ? 'l0-0' : 'l1';
  if (c.home === 'mid') return 'l2';
  return 'l4';
}

const HOME_LABEL: Record<Home, string> = {
  mem: 'row is still in the memtable',
  l0: 'row is in the newest L0 file',
  mid: 'row is in L2',
  deep: 'row is in the bottommost level',
};

const HOME_SHORT: Record<Home, string> = {
  mem: 'in memtable',
  l0: 'in L0',
  mid: 'in L2',
  deep: 'in L4',
};

/* ------------------------------------------------------------------ probes */

type Probe = { id: string; label: string; sub: string; home: boolean; why: string };

function probeList(c: Cfg): Probe[] {
  const out: Probe[] = [];
  const absent =
    'The key is not there, so no source can answer early: the probe must reach the bottommost level ' +
    'before it can say "unique".';
  if (c.op === 'insert') {
    if (c.pkUnique) {
      out.push({ id: 'pk', label: 'PK unique', sub: 'absent', home: false, why: absent });
    }
    if (c.skUnique) {
      for (let i = 0; i < c.k; i++) {
        out.push({ id: `sk${i}`, label: `unique idx ${i + 1}`, sub: 'absent', home: false, why: absent });
      }
    }
    return out;
  }
  if (c.op === 'counter') {
    if (c.strategy === 'rmw') {
      out.push({
        id: 'row',
        label: 'read counter',
        sub: HOME_SHORT[c.home],
        home: true,
        why: 'An LSM has no in-place update. To add 1 you must fetch the whole row, add 1 in the client, and Put the whole row back.',
      });
    }
    return out;
  }
  out.push({
    id: 'row',
    label: c.op === 'delete' ? 'read old row' : 'read old row',
    sub: HOME_SHORT[c.home],
    home: true,
    why:
      c.op === 'delete'
        ? 'You cannot tombstone index entries you cannot name. The old row supplies the old indexed column values.'
        : 'The new index entries are computable from the statement; the *old* ones are not. They only exist in the stored row.',
  });
  if (c.op === 'update' && c.skUnique) {
    for (let i = 0; i < c.k; i++) {
      out.push({ id: `sk${i}`, label: `unique idx ${i + 1}`, sub: 'absent', home: false, why: absent });
    }
  }
  return out;
}

/* ------------------------------------------------------------------ writes */

type W = { label: string; bytes: number; kind: 'put' | 'del' | 'merge' };

function writeList(c: Cfg): W[] {
  const out: W[] = [];
  if (c.op === 'insert') {
    out.push({ label: 'PK → row', bytes: ROW_B, kind: 'put' });
    for (let i = 0; i < c.k; i++) out.push({ label: `idx ${i + 1} → PK`, bytes: IDX_B, kind: 'put' });
    return out;
  }
  if (c.op === 'delete') {
    out.push({ label: 'PK → Delete', bytes: TOMB_B, kind: 'del' });
    for (let i = 0; i < c.k; i++) out.push({ label: `idx ${i + 1} → SingleDelete`, bytes: TOMB_B, kind: 'del' });
    return out;
  }
  if (c.op === 'counter') {
    if (c.strategy === 'merge') return [{ label: 'counter → Merge(+1)', bytes: OPERAND_B, kind: 'merge' }];
    return [{ label: 'PK → whole row', bytes: ROW_B, kind: 'put' }];
  }
  out.push({ label: 'PK → row', bytes: ROW_B, kind: 'put' });
  for (let i = 0; i < c.k; i++) {
    out.push({ label: `idx ${i + 1} old → SingleDelete`, bytes: TOMB_B, kind: 'del' });
    out.push({ label: `idx ${i + 1} new → PK`, bytes: IDX_B, kind: 'put' });
  }
  return out;
}

/** Bytes the statement actually changes, against which the engine's writes amplify. */
function changedBytes(c: Cfg) {
  if (c.op === 'insert') return ROW_B;
  if (c.op === 'delete') return 0;
  if (c.op === 'counter') return 8;
  return 64;
}

/* -------------------------------------------------------------- simulation */

type CellKind = 'none' | 'rejected' | 'cache' | 'disk' | 'mem';

type Cell = { kind: CellKind; found: boolean; ns: number; hint: string };

type Sim = {
  probes: Probe[];
  writes: W[];
  grid: Cell[][];
  filterChecks: number;
  blockReads: number;
  wasted: number;
  probeNs: number;
  bytes: number;
  fp: number;
};

function simulate(c: Cfg, srcs: Src[], step: number): Sim {
  const probes = probeList(c);
  const writes = writeList(c);
  const fp = c.bits === 0 ? 1 : Math.pow(0.6185, c.bits);
  const home = homeId(c);
  const hitRate = c.hotCache ? 0.9 : 0.15;

  let filterChecks = 0;
  let blockReads = 0;
  let wasted = 0;
  let probeNs = 0;

  const grid = probes.map((p, pi) => {
    let done = false;
    return srcs.map((s, si): Cell => {
      if (done) {
        return { kind: 'none', found: false, ns: 0, hint: 'Not reached — the probe already had its answer.' };
      }
      if (s.kind === 'mem') {
        const found = p.home && home === s.id;
        if (found) done = true;
        probeNs += MEM_NS;
        return {
          kind: 'mem',
          found,
          ns: MEM_NS,
          hint: found
            ? 'Found in memory. A row written seconds ago costs one skiplist search and no I/O at all.'
            : 'A memtable has no Bloom filter for whole-key lookups in the general case: every probe searches the skiplist.',
        };
      }
      const isHome = p.home && home === s.id;
      const r = pick(step, pi, si);
      if (c.bits > 0) filterChecks += 1;
      if (!isHome && c.bits > 0 && r > fp) {
        probeNs += FILTER_NS;
        return {
          kind: 'rejected',
          found: false,
          ns: FILTER_NS,
          hint: `The Bloom filter says "definitely not here" at ${c.bits} bits/key. This is the whole trick: one in-memory hash probe instead of an index lookup and a block read.`,
        };
      }
      const hit = pick(step + 7, pi, si + 3) < hitRate;
      const ns = (c.bits > 0 ? FILTER_NS : 0) + (hit ? CACHE_NS : IO_NS);
      blockReads += 1;
      if (!isHome) wasted += 1;
      probeNs += ns;
      if (isHome) done = true;
      return {
        kind: hit ? 'cache' : 'disk',
        found: isHome,
        ns,
        hint: isHome
          ? `The row is here. Index block binary search, then the data block${hit ? ' out of the block cache' : ' off the device'}, then a restart-point scan to the key.`
          : c.bits === 0
            ? 'No filter configured on this column family, so the only way to know is to open the file: index block, data block, nothing there.'
            : `Bloom false positive (p ≈ ${(fp * 100).toFixed(2)}%). The filter said "maybe", the block${hit ? ' was cached' : ' had to be read'}, and the key was not in it.`,
      };
    });
  });

  const bytes = writes.reduce((a, w) => a + w.bytes, 0);
  return { probes, writes, grid, filterChecks, blockReads, wasted, probeNs, bytes, fp };
}

/* ------------------------------------------------------------------ colors */

const CELL: Record<CellKind, { fill: string; glyph: string; label: string }> = {
  none: { fill: 'none', glyph: '', label: 'not reached' },
  rejected: { fill: 'var(--viz-ink-muted)', glyph: '·', label: 'filter rejected' },
  mem: { fill: 'var(--viz-7)', glyph: 'm', label: 'memtable search' },
  cache: { fill: 'var(--viz-1)', glyph: 'c', label: 'block from cache' },
  disk: { fill: 'var(--viz-2)', glyph: 'io', label: 'block from disk' },
};

const WFILL: Record<W['kind'], string> = {
  put: 'var(--viz-clean)',
  del: 'var(--viz-stale)',
  merge: 'var(--viz-4)',
};

/* --------------------------------------------------------------- component */

type LogRow = {
  n: number;
  op: string;
  probes: number;
  filters: number;
  reads: number;
  wasted: number;
  kv: number;
  bytes: number;
  ns: number;
};

const OP_LABEL: Record<Op, string> = {
  insert: 'INSERT',
  update: 'UPDATE',
  delete: 'DELETE',
  counter: 'UPDATE counter',
};

export default function ReadBeforeWriteCostLab() {
  const [op, setOp] = useState<Op>('insert');
  const [pkUnique, setPkUnique] = useState(true);
  const [k, setK] = useState(2);
  const [skUnique, setSkUnique] = useState(false);
  const [strategy, setStrategy] = useState<Strategy>('rmw');
  const [l0, setL0] = useState(4);
  const [bits, setBits] = useState(10);
  const [home, setHome] = useState<Home>('mid');
  const [hotCache, setHotCache] = useState(false);
  const [step, setStep] = useState(0);
  const [operands, setOperands] = useState(0);
  const [log, setLog] = useState<LogRow[]>([]);
  const [head, setHead] = useState('Nothing run yet.');
  const [body, setBody] = useState(
    'Press "Run statement" to execute one row write. Every column in the grid is a point lookup the ' +
      'engine must finish before it is allowed to append anything, and every filled cell in that column ' +
      'is a source it had to consult.',
  );

  const [ref, width] = useSize(780);
  const tip = useTip();

  const cfg: Cfg = { op, pkUnique, k, skUnique, strategy, l0, bits, home, hotCache };
  const srcs = sources(l0);
  const sim = simulate(cfg, srcs, step);
  const writeNs = sim.writes.length * WRITE_NS;
  const totalNs = sim.probeNs + writeNs;
  const changed = changedBytes(cfg);
  const amp = changed > 0 ? sim.bytes / changed : 0;

  const run = () => {
    const s = simulate(cfg, srcs, step + 1);
    setStep(step + 1);
    setLog((l) => [
      ...l,
      {
        n: l.length + 1,
        op: OP_LABEL[op] + (op === 'counter' ? (strategy === 'merge' ? ' (Merge)' : ' (read-modify-write)') : ''),
        probes: s.probes.length,
        filters: s.filterChecks,
        reads: s.blockReads,
        wasted: s.wasted,
        kv: s.writes.length,
        bytes: s.bytes,
        ns: s.probeNs + s.writes.length * WRITE_NS,
      },
    ]);

    if (op === 'counter' && strategy === 'merge') {
      const next = operands + 1;
      setOperands(next);
      setHead(`Merge operand ${next} appended. No read happened.`);
      setBody(
        'The engine did not look at the current value and does not know it. It appended an operand — ' +
          `"+1" — next to whatever is already there. Cost: one ${OPERAND_B}-byte KV write, zero filter checks, ` +
          'zero block reads. The arithmetic is deferred to whoever reads next, or to the compaction that gets ' +
          'there first.',
      );
      return;
    }

    if (op === 'counter' && strategy === 'rmw') {
      setHead('Read-modify-write: one Get, then a Put of the entire row.');
      setBody(
        `The +1 is 8 bytes of intent and ${ROW_B} bytes of write, because an LSM cannot patch a value in ` +
          'place — the only way to change one column is to write a whole new version of the row. The Get in ' +
          `front of it cost ${fmtTime(sim.probeNs)}.`,
      );
      return;
    }

    if (sim.probes.length === 0) {
      setHead('A genuinely blind append.');
      setBody(
        'No unique constraint, no index entry to retire: this is the write path the LSM marketing material ' +
          'describes. One WAL append, one skiplist insert, no reads, and it is as fast as the hardware allows.',
      );
      return;
    }

    const absentProbes = sim.probes.filter((p) => !p.home).length;
    setHead(
      `${sim.probes.length} read probe${sim.probes.length === 1 ? '' : 's'}, ` +
        `${sim.blockReads} block read${sim.blockReads === 1 ? '' : 's'}, then ${sim.writes.length} KV write` +
        `${sim.writes.length === 1 ? '' : 's'}.`,
    );
    setBody(
      (absentProbes > 0
        ? `${absentProbes} of those probes look for a key that is not there, so they cannot stop early — they ` +
          `descend through every memtable, every L0 file and every level to the bottom. `
        : '') +
        `The reads cost ${fmtTime(sim.probeNs)} and the writes ${fmtTime(writeNs)}: ` +
        `${Math.round((sim.probeNs / Math.max(totalNs, 1)) * 100)}% of this statement is the read path. ` +
        `${sim.wasted} block read${sim.wasted === 1 ? '' : 's'} returned nothing at all.`,
    );
  };

  const readCounter = () => {
    const cost = sim.probeNs + operands * MERGE_NS;
    setHead(`Get collapsed ${operands} operand${operands === 1 ? '' : 's'} at read time.`);
    setBody(
      operands === 0
        ? 'No operands pending — the read found a plain value.'
        : `FullMergeV2 was handed the base value and all ${operands} operands in sequence order and folded them ` +
            `into one answer. That is ${fmtTime(operands * MERGE_NS)} of merging on top of the lookup itself. ` +
            'The operands are still on disk: a Get does not rewrite them. Only compaction does.',
    );
    setLog((l) => [
      ...l,
      {
        n: l.length + 1,
        op: `Get (merge ${operands})`,
        probes: 1,
        filters: sim.filterChecks,
        reads: sim.blockReads,
        wasted: sim.wasted,
        kv: 0,
        bytes: 0,
        ns: cost,
      },
    ]);
  };

  const compact = () => {
    const was = operands;
    setOperands(0);
    setHead(was === 0 ? 'Nothing to collapse.' : `Compaction folded ${was} operands into one value.`);
    setBody(
      was === 0
        ? 'No merge operands were pending.'
        : 'The compaction iterator saw the operands stacked on one key and applied PartialMerge (or FullMergeV2 ' +
            'once it reached the base value) to replace them with a single entry. This is what bounds the read ' +
            'cost of a Merge column: it is only as expensive as the operands compaction has not caught up with.',
    );
  };

  /* -------------------------------------------------------------- geometry */

  const labelW = 178;
  const colW = 96;
  const rowH = 30;
  const headH = 46;
  const cols = Math.max(sim.probes.length, 1);
  const gridW = cols * colW;
  const stripW = Math.max(sim.writes.length * 104, 120);
  const svgW = Math.max(width, labelW + Math.max(gridW, stripW) + 24);
  const stripY = headH + srcs.length * rowH + 26;
  const height = stripY + 62;

  return (
    <VizPanel
      title="What one row write really costs"
      subtitle="Columns are the point lookups the engine must complete before it may append; rows are the sources each lookup consults. Turn the constraint and index knobs and watch the read path reappear inside the write path."
      controls={
        <>
          <Segmented
            label="Statement"
            value={op}
            onChange={setOp}
            options={[
              { value: 'insert', label: 'INSERT' },
              { value: 'update', label: 'UPDATE' },
              { value: 'delete', label: 'DELETE' },
              { value: 'counter', label: 'counter +1' },
            ]}
          />
          <Check label="PK unique check" checked={pkUnique} onChange={setPkUnique} />
          <Slider label="Secondary indexes" min={0} max={4} value={k} onChange={setK} />
          <Check label="Secondary indexes are UNIQUE" checked={skUnique} onChange={setSkUnique} />
          <Segmented
            label="Counter column"
            value={strategy}
            onChange={setStrategy}
            options={[
              { value: 'rmw', label: 'read-modify-write', title: 'Get the row, add 1, Put the row' },
              { value: 'merge', label: 'Merge operand', title: 'Append "+1" and let reads or compaction fold it in' },
            ]}
          />
          <Choice
            label="Row currently lives"
            value={home}
            onChange={setHome}
            options={[
              { value: 'mem', label: 'in the memtable' },
              { value: 'l0', label: 'in the newest L0 file' },
              { value: 'mid', label: 'in L2' },
              { value: 'deep', label: 'in the bottommost level' },
            ]}
          />
          <Slider label="L0 files" min={0} max={6} value={l0} onChange={setL0} />
          <Slider
            label="Bloom"
            min={0}
            max={16}
            value={bits}
            onChange={setBits}
            format={(n) => (n === 0 ? 'no filter' : `${n} bits/key`)}
          />
          <Check label="Hot block cache" checked={hotCache} onChange={setHotCache} />
          <Button onClick={run} primary>
            Run statement
          </Button>
          <Button
            onClick={readCounter}
            disabled={!(op === 'counter' && strategy === 'merge')}
            title="Get the counter — the merge function runs now"
          >
            Read the counter
          </Button>
          <Button onClick={compact} disabled={operands === 0} title="Let a compaction fold the stacked operands">
            Compact operands
          </Button>
          <Button
            onClick={() => {
              setStep(0);
              setOperands(0);
              setLog([]);
              setHead('Reset.');
              setBody('Same configuration, empty log, same deterministic draws for filters and cache hits.');
            }}
          >
            Reset
          </Button>
        </>
      }
      legend={
        <Legend
          items={[
            { label: 'm — memtable skiplist search', color: CELL.mem.fill },
            { label: '· — Bloom filter rejected it (free)', color: CELL.rejected.fill, shape: 'dot' },
            { label: 'c — data block from the block cache', color: CELL.cache.fill },
            { label: 'io — data block read from the device', color: CELL.disk.fill },
            { label: 'ring — where the probe found its answer and stopped', color: 'var(--viz-good)', shape: 'line' },
            { label: 'Put / Merge operand', color: WFILL.put },
            { label: 'tombstone', color: WFILL.del },
          ]}
        />
      }
      stats={
        <Stats
          items={[
            { label: 'Read probes', value: sim.probes.length, hint: 'Full point lookups this one statement must finish first' },
            { label: 'KV writes', value: sim.writes.length, hint: 'Puts, tombstones and merge operands appended for one logical row' },
            { label: 'Block reads', value: sim.blockReads, hint: 'Data blocks opened; each is a cache lookup or a device read' },
            { label: 'Wasted reads', value: sim.wasted, hint: 'Blocks opened that did not contain the key — Bloom false positives, or no filter at all' },
            { label: 'Statement latency', value: fmtTime(totalNs), hint: 'Probes plus the WAL and memtable writes' },
            { label: 'Read share', value: `${Math.round((sim.probeNs / Math.max(totalNs, 1)) * 100)}%`, hint: 'How much of a "write" is actually reading' },
            { label: 'Bytes appended', value: fmtBytes(sim.bytes), hint: 'Before compaction touches any of it' },
            { label: 'Front-door amplification', value: changed > 0 ? `${amp.toFixed(1)}×` : '—', hint: 'Engine bytes written per byte the statement actually changed — index maintenance only, no compaction yet' },
            { label: 'Merge operands pending', value: operands, hint: 'Stacked on the counter key, waiting for a read or a compaction' },
          ]}
        />
      }
      note={
        <Note>
          <strong>{head}</strong> {body}
        </Note>
      }
      table={
        <>
          <table className="viz-table">
            <thead>
              <tr>
                <th>Source</th>
                {sim.probes.map((p) => (
                  <th key={p.id}>{p.label}</th>
                ))}
                {sim.probes.length === 0 ? <th>no probe</th> : null}
              </tr>
            </thead>
            <tbody>
              {srcs.map((s, si) => (
                <tr key={s.id}>
                  <td>{s.label}</td>
                  {sim.probes.map((p, pi) => {
                    const c = sim.grid[pi][si];
                    return (
                      <td key={p.id}>
                        {CELL[c.kind].label}
                        {c.found ? ' — found' : ''}
                        {c.ns > 0 ? ` (${fmtTime(c.ns)})` : ''}
                      </td>
                    );
                  })}
                  {sim.probes.length === 0 ? <td>not consulted</td> : null}
                </tr>
              ))}
            </tbody>
          </table>
          <table className="viz-table">
            <thead>
              <tr>
                <th>#</th>
                <th>Statement</th>
                <th>Probes</th>
                <th>Filter checks</th>
                <th>Block reads</th>
                <th>Wasted</th>
                <th>KV writes</th>
                <th>Bytes</th>
                <th>Latency</th>
              </tr>
            </thead>
            <tbody>
              {log.length === 0 ? (
                <tr>
                  <td colSpan={9}>Nothing run yet.</td>
                </tr>
              ) : (
                log.map((r) => (
                  <tr key={r.n}>
                    <td>{r.n}</td>
                    <td>{r.op}</td>
                    <td>{r.probes}</td>
                    <td>{r.filters}</td>
                    <td>{r.reads}</td>
                    <td>{r.wasted}</td>
                    <td>{r.kv}</td>
                    <td>{r.bytes === 0 ? '—' : fmtBytes(r.bytes)}</td>
                    <td>{fmtTime(r.ns)}</td>
                  </tr>
                ))
              )}
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
            aria-label="A grid of read probes against LSM sources, and the KV writes the statement finally appends"
          >
            {/* probe column headers */}
            {sim.probes.map((p, pi) => (
              <g key={p.id} {...tip(<>{p.why}</>)} style={{ cursor: 'help' }}>
                <rect
                  x={labelW + pi * colW + 3}
                  y={4}
                  width={colW - 6}
                  height={headH - 12}
                  rx={6}
                  fill="var(--viz-plane)"
                  stroke="var(--viz-border)"
                />
                <text x={labelW + pi * colW + colW / 2} y={20} textAnchor="middle" fill="var(--viz-ink)" fontWeight={600}>
                  {p.label}
                </text>
                <text x={labelW + pi * colW + colW / 2} y={33} textAnchor="middle">
                  {p.sub}
                </text>
              </g>
            ))}
            {sim.probes.length === 0 ? (
              <text x={labelW + 8} y={26} fill="var(--viz-good)" fontWeight={600}>
                no read required — this really is a blind append
              </text>
            ) : null}

            {/* source rows */}
            {srcs.map((s, si) => {
              const y = headH + si * rowH;
              return (
                <g key={s.id}>
                  <rect
                    x={labelW}
                    y={y}
                    width={Math.max(gridW, 8)}
                    height={rowH - 4}
                    rx={5}
                    fill={si % 2 === 0 ? 'var(--viz-plane)' : 'none'}
                    stroke="var(--viz-border)"
                  />
                  <text x={0} y={y + 12} fill="var(--viz-ink)" fontWeight={600}>
                    {s.label}
                  </text>
                  <text x={0} y={y + 23} fill="var(--viz-ink-muted)">
                    {s.sub}
                  </text>
                  {sim.probes.map((p, pi) => {
                    const c = sim.grid[pi][si];
                    if (c.kind === 'none') return null;
                    const cx = labelW + pi * colW + colW / 2;
                    const cy = y + (rowH - 4) / 2;
                    const style = CELL[c.kind];
                    return (
                      <g key={p.id} {...tip(<>{c.hint}</>)} style={{ cursor: 'help' }}>
                        {c.kind === 'rejected' ? (
                          <circle cx={cx} cy={cy} r={4} fill={style.fill} />
                        ) : (
                          <rect x={cx - 29} y={cy - 9} width={58} height={18} rx={4} fill={style.fill} />
                        )}
                        {c.kind !== 'rejected' ? (
                          <text x={cx} y={cy + 4} textAnchor="middle" fill="var(--viz-surface)" fontWeight={600}>
                            {style.glyph} {fmtTime(c.ns)}
                          </text>
                        ) : null}
                        {c.found ? (
                          <rect
                            x={cx - 33}
                            y={cy - 13}
                            width={66}
                            height={26}
                            rx={7}
                            fill="none"
                            stroke="var(--viz-good)"
                            strokeWidth={2}
                          />
                        ) : null}
                      </g>
                    );
                  })}
                </g>
              );
            })}

            {/* the writes it finally gets to do */}
            <text x={0} y={stripY + 14} fill="var(--viz-ink)" fontWeight={600}>
              Then the writes
            </text>
            <text x={0} y={stripY + 27} fill="var(--viz-ink-muted)">
              {sim.writes.length} KV entries, {fmtBytes(sim.bytes)}
            </text>
            {sim.writes.map((w, i) => {
              const x = labelW + i * 104;
              return (
                <g
                  key={`${w.label}-${i}`}
                  {...tip(
                    <>
                      <strong>{w.label}</strong>
                      <br />
                      {w.kind === 'merge'
                        ? 'A merge operand: not a value, an instruction to be applied to whatever value turns up underneath it.'
                        : w.kind === 'del'
                          ? 'A tombstone. It is a write like any other, and it lives until a compaction reaches a level where nothing below overlaps this key.'
                          : 'A Put of a complete new version. LSMs have no partial update: one changed column rewrites the row.'}
                      <br />
                      {fmtBytes(w.bytes)} into the WAL and the memtable.
                    </>,
                  )}
                  style={{ cursor: 'help' }}
                >
                  <rect x={x} y={stripY} width={98} height={26} rx={6} fill={WFILL[w.kind]} />
                  <text x={x + 49} y={stripY + 17} textAnchor="middle" fill="var(--viz-surface)" fontWeight={600}>
                    {w.label}
                  </text>
                </g>
              );
            })}
          </svg>
        </TooltipHost>
      </div>
    </VizPanel>
  );
}
