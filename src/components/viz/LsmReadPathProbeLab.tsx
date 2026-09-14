import { useMemo, useState } from 'react';
import {
  VizPanel,
  Slider,
  Segmented,
  Choice,
  Check,
  Button,
  Legend,
  Stats,
  Note,
  TooltipHost,
  useTip,
  fmtBytes,
  fmtNum,
  fmtTime,
  makeRng,
  useSize,
} from './Viz';

/**
 * One point lookup, run against a whole LSM tree.
 *
 * The search order is RocksDB's: the mutable memtable, the immutable list, every L0 file
 * newest-first (they overlap, so none can be ruled out by key range when writes are
 * random), then exactly one file per level below, chosen by binary search over the
 * version's file metadata. Each candidate file costs a filter probe, an index block and a
 * data block, and each of those three is either in the block cache or is an I/O.
 *
 * The block cache is a real LRU here: entries are (file, block) pairs, a hit bumps
 * recency, and index/filter blocks compete with data blocks for capacity only when
 * cache_index_and_filter_blocks is on. Everything is deterministic — one seeded PRNG
 * drives the key sequence, and the whole history is replayed whenever a knob moves.
 */

const KEYS = 100_000;
const BLOCK_BYTES = 4096;
const BLOCKS_PER_FILE = 32;
const LEVEL_FILES = [4, 12, 40, 120, 400, 1200]; // L1..L6, a scale model of a 10x fanout
const SEED = 20130806;

const MEM_NS = 260; // skiplist descent: a handful of dependent pointer chases
const CACHE_NS = 150; // block cache lookup + pin, block already uncompressed
const READER_NS = 60; // filter/index block held open in the table reader
const DISK_NS = 90_000; // 4 KB NVMe read + checksum verify + decompress

function hash32(s: string): number {
  let h = 2166136261 >>> 0;
  for (let i = 0; i < s.length; i++) {
    h ^= s.charCodeAt(i);
    h = Math.imul(h, 16777619) >>> 0;
  }
  return h >>> 0;
}
const unit = (s: string) => hash32(s) / 4294967296;

/* ------------------------------------------------------------------- config */

type Workload = 'uniform' | 'hot' | 'miss';
type Order = 'random' | 'sorted';

type Cfg = {
  l0: number;
  bpk: number; // 0 = filter_policy is nullptr
  cap: number; // block cache capacity, in 4 KB entries
  workload: Workload;
  order: Order;
  cacheMeta: boolean; // cache_index_and_filter_blocks
  gets: number;
};

/** FP rate of a Bloom filter at the optimal k: 0.6185^(bits per key). */
const fpRate = (bpk: number) => (bpk <= 0 ? 1 : Math.pow(0.6185, bpk));

/* -------------------------------------------------------------- where a key lives */

const ROWS = ['mem', 'imm', 'L0', 'L1', 'L2', 'L3', 'L4', 'L5', 'L6'] as const;
type Row = (typeof ROWS)[number];

/** Fraction of the key space resident in each source; the rest was never written. */
const PLACEMENT: [Row | 'none', number][] = [
  ['none', 0.15],
  ['mem', 0.01],
  ['imm', 0.01],
  ['L0', 0.03],
  ['L1', 0.01],
  ['L2', 0.02],
  ['L3', 0.06],
  ['L4', 0.12],
  ['L5', 0.22],
  ['L6', 0.37],
];

function homeOf(keyId: number, cfg: Cfg): { row: Row; file: number } | null {
  if (cfg.workload === 'miss') return null;
  let u = unit(`home:${keyId}`);
  for (const [row, share] of PLACEMENT) {
    if (u < share) {
      if (row === 'none') return null;
      if (row === 'mem' || row === 'imm') return { row, file: 0 };
      if (row === 'L0') {
        if (cfg.l0 === 0) return null; // it has already been compacted out of L0
        return { row, file: l0Candidate(keyId, cfg) ?? hash32(`l0:${keyId}`) % cfg.l0 };
      }
      return { row, file: levelCandidate(keyId, ROWS.indexOf(row) - 2) };
    }
    u -= share;
  }
  return null;
}

/** L0 files overlap under random writes, so nothing can be excluded by range. */
function l0Candidate(keyId: number, cfg: Cfg): number | null {
  if (cfg.l0 === 0) return null;
  if (cfg.order === 'random') return null; // every file is a candidate
  return Math.min(cfg.l0 - 1, Math.floor((keyId / KEYS) * cfg.l0));
}

/** L1+ files are disjoint and sorted: one binary search over the version picks one file. */
const levelCandidate = (keyId: number, lvl: number) =>
  Math.min(LEVEL_FILES[lvl] - 1, Math.floor((keyId / KEYS) * LEVEL_FILES[lvl]));

/* ------------------------------------------------------------------ the cache */

type Kind = 'data' | 'index' | 'filter';
type Cache = { map: Map<string, Kind>; cap: number; hits: number; misses: number };

function touch(c: Cache, key: string, kind: Kind): boolean {
  if (c.map.has(key)) {
    c.map.delete(key);
    c.map.set(key, kind);
    c.hits++;
    return true;
  }
  c.misses++;
  while (c.map.size >= c.cap) {
    const oldest = c.map.keys().next();
    if (oldest.done) break;
    c.map.delete(oldest.value);
  }
  c.map.set(key, kind);
  return false;
}

/* ------------------------------------------------------------------ one Get */

type Status = 'none' | 'range' | 'reject' | 'cached' | 'disk' | 'waste' | 'found';

type Step = {
  row: Row;
  candidates: number;
  files: { idx: number; status: Status }[];
  status: Status;
  detail: string;
  ns: number;
  diskBlocks: number;
};

type Trace = { keyId: number; present: boolean; steps: Step[]; ns: number; result: string };

type Counters = {
  gets: number;
  found: number;
  ns: number;
  filtersChecked: number;
  filtersRejected: number;
  falsePositives: number;
  diskBlocks: number;
  sources: number;
};

function doGet(keyId: number, cfg: Cfg, cache: Cache, rng: () => number, ctr: Counters, readers: Set<string>): Trace {
  const home = homeOf(keyId, cfg);
  const steps: Step[] = [];
  let ns = 0;
  let done = false;
  let result = 'NotFound';

  /** A filter/index block: in the block cache, or pinned in the table reader. */
  const meta = (file: string, kind: Kind) => {
    if (cfg.cacheMeta) return touch(cache, `${file}:${kind}`, kind) ? CACHE_NS : DISK_NS;
    readers.add(file);
    return READER_NS;
  };

  const visitFile = (row: Row, file: string, contains: boolean) => {
    let cost = 0;
    let disk = 0;

    if (cfg.bpk > 0) {
      cost += meta(file, 'filter');
      ctr.filtersChecked++;
      if (!contains && rng() >= fpRate(cfg.bpk)) {
        ctr.filtersRejected++;
        return { status: 'reject' as Status, detail: 'filter: no — file skipped', ns: cost, disk: 0 };
      }
    }

    cost += meta(file, 'index');
    const block = hash32(`b:${keyId}`) % BLOCKS_PER_FILE;
    const hit = touch(cache, `${file}:d${block}`, 'data');
    if (hit) cost += CACHE_NS;
    else {
      cost += DISK_NS;
      disk = 1;
      ctr.diskBlocks++;
    }

    if (contains) return { status: 'found' as Status, detail: `key found in ${file}`, ns: cost, disk };
    if (cfg.bpk > 0) {
      ctr.falsePositives++;
      return {
        status: 'waste' as Status,
        detail: `filter false positive — ${hit ? 'cached' : 'disk'} block read, key absent`,
        ns: cost,
        disk,
      };
    }
    return {
      status: 'waste' as Status,
      detail: `no filter — ${hit ? 'cached' : 'disk'} block read, key absent`,
      ns: cost,
      disk,
    };
  };

  for (const row of ROWS) {
    if (done) {
      steps.push({ row, candidates: 0, files: [], status: 'none', detail: 'not reached', ns: 0, diskBlocks: 0 });
      continue;
    }
    ctr.sources++;

    if (row === 'mem' || row === 'imm') {
      const contains = home?.row === row;
      ns += MEM_NS;
      if (contains) {
        done = true;
        result = 'Found';
      }
      steps.push({
        row,
        candidates: 1,
        files: [{ idx: 0, status: contains ? 'found' : 'cached' }],
        status: contains ? 'found' : 'cached',
        detail: contains
          ? 'key found in the memtable — no filter, no index, no I/O'
          : 'skiplist searched to a miss — memtables carry no whole-key filter by default',
        ns: MEM_NS,
        diskBlocks: 0,
      });
      continue;
    }

    if (row === 'L0') {
      if (cfg.l0 === 0) {
        steps.push({ row, candidates: 0, files: [], status: 'range', detail: 'L0 is empty', ns: 0, diskBlocks: 0 });
        continue;
      }
      const only = l0Candidate(keyId, cfg);
      const files: { idx: number; status: Status }[] = [];
      let rowNs = 0;
      let rowDisk = 0;
      let rowStatus: Status = 'range';
      let detail = '';
      let candidates = 0;
      for (let i = 0; i < cfg.l0; i++) {
        if (done) {
          files.push({ idx: i, status: 'none' });
          continue;
        }
        if (only !== null && i !== only) {
          files.push({ idx: i, status: 'range' });
          continue;
        }
        candidates++;
        const r = visitFile('L0', `L0#${i}`, home?.row === 'L0' && home.file === i);
        files.push({ idx: i, status: r.status });
        rowNs += r.ns;
        rowDisk += r.disk;
        if (r.status === 'found') {
          done = true;
          result = 'Found';
          rowStatus = 'found';
          detail = r.detail;
        } else if (rowStatus !== 'found') {
          rowStatus = r.status === 'reject' && rowStatus === 'waste' ? 'waste' : r.status;
          detail = r.detail;
        }
      }
      ns += rowNs;
      steps.push({
        row,
        candidates,
        files,
        status: rowStatus,
        detail:
          candidates === 0
            ? 'every L0 file excluded by key range'
            : `${candidates} candidate file${candidates === 1 ? '' : 's'} — ${detail}`,
        ns: rowNs,
        diskBlocks: rowDisk,
      });
      continue;
    }

    const lvl = ROWS.indexOf(row) - 2;
    const cand = levelCandidate(keyId, lvl);
    const r = visitFile(row, `${row}#${cand}`, home?.row === row && home.file === cand);
    ns += r.ns;
    if (r.status === 'found') {
      done = true;
      result = 'Found';
    }
    steps.push({
      row,
      candidates: 1,
      files: [{ idx: cand, status: r.status }],
      status: r.status,
      detail: `binary search picked file ${cand + 1} of ${LEVEL_FILES[lvl]} — ${r.detail}`,
      ns: r.ns,
      diskBlocks: r.disk,
    });
  }

  ctr.gets++;
  ctr.ns += ns;
  if (result === 'Found') ctr.found++;
  return { keyId, present: home !== null, steps, ns, result };
}

/* ----------------------------------------------------------------- the run */

function simulate(cfg: Cfg) {
  const rng = makeRng(SEED);
  const cache: Cache = { map: new Map(), cap: Math.max(4, cfg.cap), hits: 0, misses: 0 };
  const readers = new Set<string>();
  const ctr: Counters = {
    gets: 0,
    found: 0,
    ns: 0,
    filtersChecked: 0,
    filtersRejected: 0,
    falsePositives: 0,
    diskBlocks: 0,
    sources: 0,
  };
  let last: Trace | null = null;
  for (let g = 0; g < cfg.gets; g++) {
    const u = rng();
    const keyId = Math.floor(KEYS * (cfg.workload === 'hot' ? Math.pow(u, 4) : u));
    last = doGet(keyId, cfg, cache, rng, ctr, readers);
  }
  let data = 0;
  let index = 0;
  let filter = 0;
  for (const kind of cache.map.values()) {
    if (kind === 'data') data++;
    else if (kind === 'index') index++;
    else filter++;
  }
  return { ctr, last, cache, readers: readers.size, occ: { data, index, filter } };
}

/* --------------------------------------------------------------- appearance */

const STATUS: Record<Status, { fill: string; glyph: string; label: string }> = {
  none: { fill: 'var(--viz-neutral)', glyph: '·', label: 'not reached (the key was already found)' },
  range: { fill: 'var(--viz-neutral)', glyph: '×', label: '× excluded by key range — free' },
  reject: { fill: 'var(--viz-good)', glyph: '∅', label: '∅ Bloom filter said no — file skipped' },
  cached: { fill: 'var(--viz-1)', glyph: 'C', label: 'C searched, blocks came from the cache' },
  disk: { fill: 'var(--viz-warning)', glyph: 'D', label: 'D searched, block read from disk' },
  waste: { fill: 'var(--viz-critical)', glyph: '!', label: '! block read, key not there (false positive)' },
  found: { fill: 'var(--viz-7)', glyph: '★', label: '★ the key is here — search stops' },
};

const ROW_LABEL: Record<Row, string> = {
  mem: 'memtable',
  imm: 'immutable memtable',
  L0: 'L0',
  L1: 'L1',
  L2: 'L2',
  L3: 'L3',
  L4: 'L4',
  L5: 'L5',
  L6: 'L6 (bottommost)',
};

/* ------------------------------------------------------------- the component */

export default function LsmReadPathProbeLab() {
  const [l0, setL0] = useState(4);
  const [bpk, setBpk] = useState(10);
  const [cap, setCap] = useState(64);
  const [workload, setWorkload] = useState<Workload>('uniform');
  const [order, setOrder] = useState<Order>('random');
  const [cacheMeta, setCacheMeta] = useState(false);
  const [gets, setGets] = useState(1);
  const [ref, width] = useSize(820);
  const tip = useTip();

  const cfg: Cfg = { l0, bpk, cap, workload, order, cacheMeta, gets };
  const sim = useMemo(() => simulate(cfg), [l0, bpk, cap, workload, order, cacheMeta, gets]);
  const { ctr, last, occ } = sim;

  const total = sim.cache.hits + sim.cache.misses;
  const hitRate = total ? sim.cache.hits / total : 0;

  /* ------------------------------------------------------------- geometry */
  const labelW = 150;
  const boxW = 30;
  const gap = 5;
  const svgW = Math.max(width, labelW + Math.max(l0, 16) * (boxW + gap) + 250);
  const rowH = 30;
  const rowGap = 7;
  const top = 14;
  const barTop = top + ROWS.length * (rowH + rowGap) + 18;
  const height = barTop + 58;
  const rowY = (i: number) => top + i * (rowH + rowGap);
  const detailX = svgW - 8;

  const stepFor = (row: Row) => last?.steps.find((s) => s.row === row);

  const box = (key: string, x: number, y: number, w: number, status: Status, hint: string) => {
    const st = STATUS[status];
    return (
      <g key={key} {...tip(<>{hint}</>)} style={{ cursor: 'help' }}>
        <rect
          x={x}
          y={y}
          width={w}
          height={rowH - 8}
          rx={4}
          fill={st.fill}
          stroke={status === 'range' || status === 'none' ? 'var(--viz-axis)' : 'var(--viz-surface)'}
          strokeWidth={1.5}
          strokeDasharray={status === 'range' ? '3 2' : undefined}
          opacity={status === 'none' ? 0.45 : 1}
        />
        <text x={x + w / 2} y={y + rowH / 2 + 1} textAnchor="middle" fill="var(--viz-ink)">
          {st.glyph}
        </text>
      </g>
    );
  };

  const capUsed = occ.data + occ.index + occ.filter;
  const barW = svgW - labelW - 16;
  const seg = (n: number) => (cap ? (n / cap) * barW : 0);

  return (
    <VizPanel
      title="A point lookup, source by source"
      subtitle="Get(key) walks the memtable, the immutable list, every L0 file and one file per level below. Watch which files the key range excludes for free, which the Bloom filter rejects, and which cost a block read."
      controls={
        <>
          <Slider label="L0 files" min={0} max={16} value={l0} onChange={setL0} />
          <Choice
            label="filter_policy"
            value={String(bpk)}
            onChange={(v) => setBpk(Number(v))}
            options={[
              { value: '0', label: 'nullptr (no filter)' },
              { value: '4', label: '4 bits/key' },
              { value: '8', label: '8 bits/key' },
              { value: '10', label: '10 bits/key' },
              { value: '16', label: '16 bits/key' },
            ]}
          />
          <Slider
            label="Block cache"
            min={8}
            max={512}
            step={8}
            value={cap}
            onChange={setCap}
            format={(v) => `${v} blocks · ${fmtBytes(v * BLOCK_BYTES)}`}
          />
          <Segmented
            label="Workload"
            value={workload}
            onChange={setWorkload}
            options={[
              { value: 'uniform', label: 'uniform', title: 'Keys drawn uniformly from the key space' },
              { value: 'hot', label: 'hot set', title: 'Skewed: a small set of keys takes most of the traffic' },
              { value: 'miss', label: 'all misses', title: 'Existence checks for keys that were never written' },
            ]}
          />
          <Segmented
            label="L0 key ranges"
            value={order}
            onChange={setOrder}
            options={[
              { value: 'random', label: 'overlapping', title: 'Random-key writes: every L0 file spans the key space' },
              { value: 'sorted', label: 'disjoint', title: 'Monotonic keys: each L0 flush covers its own range' },
            ]}
          />
          <Check label="cache_index_and_filter_blocks" checked={cacheMeta} onChange={setCacheMeta} />
          <Button onClick={() => setGets((g) => g + 1)} primary>
            Get ×1
          </Button>
          <Button onClick={() => setGets((g) => g + 25)}>Get ×25</Button>
          <Button onClick={() => setGets(1)}>Reset</Button>
        </>
      }
      legend={<Legend items={(Object.keys(STATUS) as Status[]).map((s) => ({ label: STATUS[s].label, color: STATUS[s].fill }))} />}
      stats={
        <Stats
          items={[
            { label: 'Last Get', value: last ? fmtTime(last.ns) : '—', hint: last ? `key ${last.keyId}, ${last.result}` : '' },
            { label: 'Mean Get', value: ctr.gets ? fmtTime(ctr.ns / ctr.gets) : '—', hint: `over ${fmtNum(ctr.gets)} lookups` },
            { label: 'Block cache hit rate', value: `${(hitRate * 100).toFixed(1)}%`, hint: `${fmtNum(sim.cache.hits)} hits / ${fmtNum(total)} lookups` },
            {
              label: 'Disk blocks per Get',
              value: ctr.gets ? (ctr.diskBlocks / ctr.gets).toFixed(2) : '—',
              hint: 'Read amplification you actually pay: 4 KB block reads per point lookup',
            },
            {
              label: 'Filters rejecting',
              value: ctr.filtersChecked ? `${((ctr.filtersRejected / ctr.filtersChecked) * 100).toFixed(1)}%` : 'no filter',
              hint: `${fmtNum(ctr.filtersRejected)} of ${fmtNum(ctr.filtersChecked)} filter probes said "no" — each one saved a block read`,
            },
            {
              label: 'Wasted reads / 100 Gets',
              value: ctr.gets ? ((ctr.falsePositives / ctr.gets) * 100).toFixed(1) : '—',
              hint: 'Blocks fetched for a key that was not in the file',
            },
          ]}
        />
      }
      note={
        <Note>
          {last ? (
            <>
              <strong>
                Get({last.keyId}) → {last.result} in {fmtTime(last.ns)}, {last.steps.filter((s) => s.status !== 'none').length}{' '}
                sources consulted.
              </strong>{' '}
              {last.present
                ? 'The search stopped at the first source holding this key, because sources are visited newest-first and the newest version wins.'
                : 'This key does not exist, so nothing could short-circuit the search: every source had to be ruled out, which is exactly the case the Bloom filter exists for.'}{' '}
              {bpk === 0
                ? 'With filter_policy = nullptr every candidate file costs an index and a data block, whether or not the key is anywhere near it.'
                : `At ${bpk} bits per key roughly ${(fpRate(bpk) * 100).toFixed(2)}% of negative filter probes still pass and cost a block read.`}
            </>
          ) : (
            <>Press Get to run a lookup.</>
          )}
        </Note>
      }
      table={
        <>
          <table className="viz-table">
            <thead>
              <tr>
                <th>Source</th>
                <th>Candidate files</th>
                <th>Outcome</th>
                <th>Disk blocks</th>
                <th>Cost</th>
              </tr>
            </thead>
            <tbody>
              {last ? (
                last.steps.map((s) => (
                  <tr key={s.row}>
                    <td>{ROW_LABEL[s.row]}</td>
                    <td>{s.candidates}</td>
                    <td>{s.detail}</td>
                    <td>{s.diskBlocks}</td>
                    <td>{s.ns ? fmtTime(s.ns) : '—'}</td>
                  </tr>
                ))
              ) : (
                <tr>
                  <td colSpan={5}>No lookups yet.</td>
                </tr>
              )}
            </tbody>
          </table>
          <table className="viz-table">
            <thead>
              <tr>
                <th>Counter</th>
                <th>Total</th>
                <th>Per Get</th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <td>Lookups (found / total)</td>
                <td>
                  {fmtNum(ctr.found)} / {fmtNum(ctr.gets)}
                </td>
                <td>—</td>
              </tr>
              <tr>
                <td>Sources consulted</td>
                <td>{fmtNum(ctr.sources)}</td>
                <td>{ctr.gets ? (ctr.sources / ctr.gets).toFixed(1) : '—'}</td>
              </tr>
              <tr>
                <td>Bloom probes (rocksdb.bloom.filter.*)</td>
                <td>{fmtNum(ctr.filtersChecked)}</td>
                <td>{ctr.gets ? (ctr.filtersChecked / ctr.gets).toFixed(1) : '—'}</td>
              </tr>
              <tr>
                <td>… filter useful (said no)</td>
                <td>{fmtNum(ctr.filtersRejected)}</td>
                <td>{ctr.gets ? (ctr.filtersRejected / ctr.gets).toFixed(1) : '—'}</td>
              </tr>
              <tr>
                <td>… false positives</td>
                <td>{fmtNum(ctr.falsePositives)}</td>
                <td>{ctr.gets ? (ctr.falsePositives / ctr.gets).toFixed(2) : '—'}</td>
              </tr>
              <tr>
                <td>Block cache hits / misses</td>
                <td>
                  {fmtNum(sim.cache.hits)} / {fmtNum(sim.cache.misses)}
                </td>
                <td>{(hitRate * 100).toFixed(1)}% hit rate</td>
              </tr>
              <tr>
                <td>Disk block reads</td>
                <td>{fmtNum(ctr.diskBlocks)}</td>
                <td>{ctr.gets ? (ctr.diskBlocks / ctr.gets).toFixed(2) : '—'}</td>
              </tr>
              <tr>
                <td>Bytes read from disk</td>
                <td>{fmtBytes(ctr.diskBlocks * BLOCK_BYTES)}</td>
                <td>{ctr.gets ? fmtBytes((ctr.diskBlocks * BLOCK_BYTES) / ctr.gets) : '—'}</td>
              </tr>
              <tr>
                <td>Block cache occupancy (data / index / filter)</td>
                <td>
                  {occ.data} / {occ.index} / {occ.filter} of {cap}
                </td>
                <td>{fmtBytes(capUsed * BLOCK_BYTES)}</td>
              </tr>
              <tr>
                <td>Table-reader memory outside the cache</td>
                <td>{cacheMeta ? '0 B' : fmtBytes(sim.readers * 2 * BLOCK_BYTES)}</td>
                <td>{cacheMeta ? 'index + filter are cache entries' : `${sim.readers} open files × index + filter`}</td>
              </tr>
            </tbody>
          </table>
        </>
      }
    >
      <div ref={ref}>
        <TooltipHost>
          <svg width={svgW} height={height} role="img" aria-label="Every source an LSM point lookup consults, and what each one cost">
            {ROWS.map((row, i) => {
              const s = stepFor(row);
              const y = rowY(i);
              return (
                <g key={row}>
                  <text x={0} y={y + rowH / 2 + 1} fill="var(--viz-ink)" fontWeight={row === 'L0' ? 600 : 400}>
                    {ROW_LABEL[row]}
                  </text>
                  {row === 'mem' || row === 'imm'
                    ? box(
                        row,
                        labelW,
                        y + 4,
                        boxW * 3,
                        s?.status ?? 'none',
                        s?.detail ?? 'in-memory skiplist',
                      )
                    : row === 'L0'
                      ? Array.from({ length: l0 }, (_, j) => {
                          const st = s?.files.find((ff) => ff.idx === j)?.status ?? 'none';
                          return box(
                            `L0-${j}`,
                            labelW + j * (boxW + gap),
                            y + 4,
                            boxW,
                            st,
                            `L0 file ${j} (${j === 0 ? 'newest' : `${j} flush${j === 1 ? '' : 'es'} older`}) — ${STATUS[st].label}`,
                          );
                        })
                      : (() => {
                          const lvl = ROWS.indexOf(row) - 2;
                          const n = Math.min(LEVEL_FILES[lvl], 18);
                          const cand = s?.files[0]?.idx ?? 0;
                          const candDrawn = Math.min(n - 1, Math.round((cand / LEVEL_FILES[lvl]) * n));
                          const w = (boxW + gap) * 0.7;
                          return Array.from({ length: n }, (_, j) =>
                            j === candDrawn
                              ? box(
                                  `${row}-c`,
                                  labelW + j * w,
                                  y + 4,
                                  w - 3,
                                  s?.status ?? 'none',
                                  `${row}: binary search over the version's file list picked file ${cand + 1} of ${LEVEL_FILES[lvl]} — ${s?.detail ?? ''}`,
                                )
                              : (
                                  <rect
                                    key={`${row}-${j}`}
                                    x={labelW + j * w}
                                    y={y + 10}
                                    width={w - 3}
                                    height={rowH - 20}
                                    rx={2}
                                    fill="var(--viz-neutral)"
                                    stroke="var(--viz-axis)"
                                    strokeWidth={0.5}
                                  />
                                ),
                          );
                        })()}
                  <text x={detailX} y={y + rowH / 2 + 1} textAnchor="end" fill="var(--viz-ink-2)">
                    {row === 'L0'
                      ? `${l0} file${l0 === 1 ? '' : 's'}, ${order === 'random' ? 'overlapping' : 'disjoint ranges'}`
                      : row === 'mem' || row === 'imm'
                        ? 'no filter, no I/O'
                        : `1 of ${fmtNum(LEVEL_FILES[ROWS.indexOf(row) - 2])} files`}
                  </text>
                </g>
              );
            })}

            <text x={0} y={barTop + 14} fill="var(--viz-ink)">
              block cache
            </text>
            <text x={0} y={barTop + 30} fill="var(--viz-ink-muted)">
              {fmtBytes(cap * BLOCK_BYTES)} capacity
            </text>
            <rect x={labelW} y={barTop} width={barW} height={22} rx={4} fill="var(--viz-plane)" stroke="var(--viz-border)" />
            <rect x={labelW} y={barTop} width={seg(occ.data)} height={22} rx={4} fill="var(--viz-1)" />
            <rect x={labelW + seg(occ.data)} y={barTop} width={seg(occ.index)} height={22} fill="var(--viz-3)" />
            <rect x={labelW + seg(occ.data + occ.index)} y={barTop} width={seg(occ.filter)} height={22} fill="var(--viz-4)" />
            <text x={labelW} y={barTop + 40} fill="var(--viz-ink-2)">
              {occ.data} data blocks
              {cacheMeta ? ` · ${occ.index} index · ${occ.filter} filter` : ' · index + filter blocks held in the table reader, outside the cache'}
              {capUsed >= cap ? ' · full, evicting' : ''}
            </text>
          </svg>
        </TooltipHost>
      </div>
    </VizPanel>
  );
}
