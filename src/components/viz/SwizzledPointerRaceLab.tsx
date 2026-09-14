import { useMemo, useState } from 'react';
import {
  VizPanel,
  Slider,
  Segmented,
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
 * One skewed access trace, three ways of finding a page.
 *
 *  - No buffer pool at all (Hekaton / VoltDB / Redis): the index hands you the row's
 *    address. Nothing to look up, nothing to pin — and a hard wall the moment the data
 *    outgrows RAM, unless the engine anti-caches cold tuples and pays aborts instead.
 *  - A classic hash-table pool: every single access hashes the page identity, takes a
 *    partition latch and probes. The mapping is centralised, so eviction is cheap.
 *  - A swizzling pool (LeanStore): a resident child is reached by the raw pointer stored
 *    in its parent, so a hit costs a dereference. The mapping is distributed into every
 *    referencing page, so eviction must walk to the parent and rewrite the reference.
 *
 * Everything is deterministic: the trace and the random eviction sampling both come from
 * makeRng, so the same controls always produce the same run.
 */

const PAGES = 96; // the whole dataset, scaled down so every page gets a cell
const PAGE_BYTES = 8192;
const TRACE_LEN = 600;
const BLOCK_PAGES = 4; // anti-caching evicts and fetches a block, not a tuple
const COOL_SHARE = 0.1; // LeanStore keeps roughly a tenth of the pool in the cooling stage

/** A cost model in nanoseconds. Deliberately coarse; the ratios are the point. */
const COST = {
  deref: 4, // in-memory engine: the index entry already holds the row address
  swizzled: 7, // read the swip, test the tag bit, optimistic version read + validate
  probe: 46, // hash the tag, BufMappingLock shared, walk the bucket, CAS the pin
  coolProbe: 52, // cooling hash table, unlink from the FIFO, re-swizzle into the parent
  unswizzle: 90, // latch the parent, rewrite one swip, insert into the cooling table
  io: 20_000, // random 8 KB read from NVMe
  writeBack: 24_000, // write one dirty page (or one anti-cache block) out
  abort: 3_500, // roll the transaction back, requeue it, execute it a second time
  merge: 1_200, // splice a fetched block's tuples back into the in-memory table
};

type Access = { page: number; write: boolean };

function buildTrace(skew: number): Access[] {
  const rng = makeRng(20180524);
  const exp = 1 + skew / 12; // 1 = uniform; larger = a heavier head
  const out: Access[] = [];
  for (let i = 0; i < TRACE_LEN; i++) {
    const u = rng();
    out.push({
      page: Math.min(PAGES - 1, Math.floor(PAGES * Math.pow(u, exp))),
      write: rng() < 0.2,
    });
  }
  return out;
}

type Counters = {
  hits: number;
  misses: number;
  probes: number;
  derefs: number;
  ios: number;
  writes: number;
  aborts: number;
  unswizzles: number;
  reclaims: number;
  ns: number;
  served: number;
};

const zero = (): Counters => ({
  hits: 0,
  misses: 0,
  probes: 0,
  derefs: 0,
  ios: 0,
  writes: 0,
  aborts: 0,
  unswizzles: 0,
  reclaims: 0,
  ns: 0,
  served: 0,
});

type CellState = 'resident' | 'dirty' | 'cooling' | 'block' | 'disk';

type LaneResult = {
  c: Counters;
  cells: CellState[];
  deadAt: number | null;
  rejected: number;
};

const blank = (): CellState[] => Array.from({ length: PAGES }, () => 'disk' as CellState);

function touch(lru: number[], p: number) {
  const i = lru.indexOf(p);
  if (i >= 0) lru.splice(i, 1);
  lru.push(p);
}

/* ------------------------------------------- 1. no buffer pool, and anti-caching */

function runInMemory(trace: Access[], n: number, frames: number, mode: 'strict' | 'anticache'): LaneResult {
  const c = zero();
  const resident = new Set<number>();
  const lru: number[] = [];
  const evicted = new Map<number, number>(); // page -> block id
  const blocks = new Map<number, number[]>();
  let nextBlock = 1;
  let deadAt: number | null = null;

  const evictBlock = () => {
    const victims = lru.splice(0, Math.min(BLOCK_PAGES, lru.length));
    if (victims.length === 0) return false;
    const id = nextBlock++;
    blocks.set(id, victims);
    for (const v of victims) {
      resident.delete(v);
      evicted.set(v, id);
    }
    c.writes++;
    c.ns += COST.writeBack;
    return true;
  };

  for (let i = 0; i < n; i++) {
    const a = trace[i];
    if (resident.has(a.page)) {
      touch(lru, a.page);
      c.hits++;
      c.derefs++;
      c.ns += COST.deref;
      c.served++;
      continue;
    }
    if (evicted.has(a.page)) {
      // The transaction has already started executing when it reaches an evicted tuple.
      // H-Store cannot block — one thread owns the whole partition — so it aborts.
      const id = evicted.get(a.page)!;
      const members = blocks.get(id)!;
      blocks.delete(id);
      c.aborts++;
      c.ns += COST.abort;
      while (resident.size + members.length > frames && evictBlock()) {
        /* make room for the whole block coming back */
      }
      c.ios++;
      c.ns += COST.io + COST.merge;
      for (const m of members) {
        evicted.delete(m);
        resident.add(m);
        lru.push(m);
      }
      touch(lru, a.page);
      c.misses++;
      c.derefs++;
      c.ns += COST.deref; // the restarted execution finally reads the tuple
      c.served++;
      continue;
    }
    // First touch of this page: the dataset is still growing into memory.
    if (resident.size >= frames) {
      if (mode === 'strict') {
        deadAt = i + 1;
        break;
      }
      if (!evictBlock()) break;
    }
    resident.add(a.page);
    lru.push(a.page);
    c.hits++;
    c.derefs++;
    c.ns += COST.deref;
    c.served++;
  }

  const cells = blank();
  for (const p of resident) cells[p] = 'resident';
  for (const p of evicted.keys()) cells[p] = 'block';
  return { c, cells, deadAt, rejected: deadAt === null ? 0 : n - deadAt + 1 };
}

/* ------------------------------------------------ 2. classic hash-table pool, LRU */

function runHashPool(trace: Access[], n: number, frames: number): LaneResult {
  const c = zero();
  const resident = new Set<number>();
  const dirty = new Set<number>();
  const lru: number[] = [];

  for (let i = 0; i < n; i++) {
    const a = trace[i];
    c.probes++; // every access starts with BufTableLookup, hit or miss
    if (resident.has(a.page)) {
      touch(lru, a.page);
      if (a.write) dirty.add(a.page);
      c.hits++;
      c.ns += COST.probe;
      c.served++;
      continue;
    }
    if (resident.size >= frames) {
      const v = lru.shift()!;
      resident.delete(v);
      if (dirty.has(v)) {
        dirty.delete(v);
        c.writes++;
        c.ns += COST.writeBack;
      }
      c.probes++; // delete the old tag: the partition lock again, exclusively
      c.ns += COST.probe;
    }
    c.probes++; // insert the new tag
    c.misses++;
    c.ios++;
    c.ns += COST.probe + COST.io;
    resident.add(a.page);
    lru.push(a.page);
    if (a.write) dirty.add(a.page);
    c.served++;
  }

  const cells = blank();
  for (const p of resident) cells[p] = dirty.has(p) ? 'dirty' : 'resident';
  return { c, cells, deadAt: null, rejected: 0 };
}

/* -------------------------------------------------- 3. swizzling pool (LeanStore) */

function runSwizzled(trace: Access[], n: number, frames: number): LaneResult {
  const c = zero();
  const rng = makeRng(991);
  const resident = new Map<number, boolean>(); // page -> swizzled?
  const dirty = new Set<number>();
  const cooling: number[] = []; // index 0 is the FIFO tail: the next page to leave
  const coolTarget = Math.max(1, Math.round(frames * COOL_SHARE));
  let pressure = false; // no eviction has been needed yet, so nothing is being cooled

  const coolOne = (skip: number) => {
    const cands: number[] = [];
    for (const [p, sw] of resident) if (sw && p !== skip) cands.push(p);
    if (cands.length === 0) return false;
    // LeanStore does not maintain an LRU list: it samples a candidate at random,
    // unswizzles it from its parent and pushes it into the cooling stage.
    const victim = cands[Math.floor(rng() * cands.length)];
    resident.set(victim, false);
    cooling.push(victim);
    c.unswizzles++;
    c.ns += COST.unswizzle;
    return true;
  };

  const evictOne = (skip: number) => {
    if (cooling.length === 0 && !coolOne(skip)) return false;
    const v = cooling.shift();
    if (v === undefined) return false;
    resident.delete(v);
    if (dirty.has(v)) {
      dirty.delete(v);
      c.writes++;
      c.ns += COST.writeBack;
    }
    return true;
  };

  for (let i = 0; i < n; i++) {
    const a = trace[i];
    const sw = resident.get(a.page);
    if (sw === true) {
      c.hits++;
      c.derefs++;
      c.ns += COST.swizzled;
    } else if (sw === false) {
      // Resident but cooling: the parent's swip is a page id again, so this access
      // does cost a probe — of the cooling hash table — and swizzles the page back.
      cooling.splice(cooling.indexOf(a.page), 1);
      resident.set(a.page, true);
      c.hits++;
      c.probes++;
      c.reclaims++;
      c.ns += COST.coolProbe;
    } else {
      let guard = PAGES * 2;
      if (resident.size >= frames) pressure = true; // the page provider only runs when frames are scarce
      while (resident.size >= frames && guard-- > 0) {
        if (!evictOne(a.page)) break;
      }
      c.misses++;
      c.ios++;
      c.ns += COST.io;
      resident.set(a.page, true);
    }
    if (a.write) dirty.add(a.page);
    c.served++;
    let guard = PAGES * 2;
    while (pressure && resident.size >= frames && cooling.length < coolTarget && guard-- > 0) {
      if (!coolOne(a.page)) break;
    }
  }

  const cells = blank();
  for (const [p, s] of resident) cells[p] = s ? (dirty.has(p) ? 'dirty' : 'resident') : 'cooling';
  return { c, cells, deadAt: null, rejected: 0 };
}

/* ------------------------------------------------------------------- the drawing */

const FILL: Record<CellState, string> = {
  resident: 'var(--viz-clean)',
  dirty: 'var(--viz-dirty)',
  cooling: 'var(--viz-warning)',
  block: 'var(--viz-7)',
  disk: 'var(--viz-neutral)',
};

const CELL_HINT: Record<CellState, string> = {
  resident: 'Resident and reachable — a pointer in the parent, or the index entry itself.',
  dirty: 'Resident and modified: evicting it costs a write.',
  cooling: 'Still in RAM, but unswizzled: the parent holds a page id and the cooling hash table holds the frame.',
  block: 'Evicted into an anti-cache block on disk. A transaction that touches it aborts.',
  disk: 'Not resident. Reaching it costs an I/O.',
};

type LaneSpec = {
  id: 'inmem' | 'hash' | 'swizzle';
  title: string;
  sub: string;
  rows: (r: LaneResult) => { label: string; value: string; hint: string }[];
};

export default function SwizzledPointerRaceLab() {
  const [memPct, setMemPct] = useState(45);
  const [skew, setSkew] = useState(60);
  const [mode, setMode] = useState<'strict' | 'anticache'>('strict');
  const [n, setN] = useState(0);
  const [ref, width] = useSize(760);
  const tip = useTip();

  const frames = Math.max(8, Math.round((PAGES * memPct) / 100));
  const trace = useMemo(() => buildTrace(skew), [skew]);

  const { workingSet, hot10, freq } = useMemo(() => {
    const f = new Array<number>(PAGES).fill(0);
    for (const a of trace) f[a.page]++;
    const sorted = [...f].sort((x, y) => y - x);
    let acc = 0;
    let ws = 0;
    while (ws < PAGES && acc < trace.length * 0.95) acc += sorted[ws++];
    let head = 0;
    for (let p = 0; p < 10; p++) head += f[p];
    return { workingSet: ws, hot10: Math.round((head / trace.length) * 100), freq: f };
  }, [trace]);

  const inmem = useMemo(() => runInMemory(trace, n, frames, mode), [trace, n, frames, mode]);
  const hash = useMemo(() => runHashPool(trace, n, frames), [trace, n, frames]);
  const swz = useMemo(() => runSwizzled(trace, n, frames), [trace, n, frames]);

  const LANES: LaneSpec[] = [
    {
      id: 'inmem',
      title: mode === 'strict' ? 'No buffer pool' : 'Anti-caching',
      sub: mode === 'strict' ? 'Hekaton · VoltDB · Redis' : 'H-Store / VoltDB anti-cache',
      rows: (r) => [
        { label: 'hash probes', value: fmtNum(r.c.probes), hint: 'There is no page table to probe.' },
        { label: 'pointer derefs', value: fmtNum(r.c.derefs), hint: 'The index entry is the row address.' },
        { label: 'page / block I/Os', value: fmtNum(r.c.ios), hint: 'Anti-cache block fetches.' },
        {
          label: mode === 'strict' ? 'rejected' : 'aborts + restarts',
          value: fmtNum(mode === 'strict' ? r.rejected : r.c.aborts),
          hint:
            mode === 'strict'
              ? 'Accesses never served, because the engine ran out of memory.'
              : 'Transactions that touched an evicted tuple, aborted, and ran again after the merge.',
        },
      ],
    },
    {
      id: 'hash',
      title: 'Hash-table pool',
      sub: 'PostgreSQL · InnoDB',
      rows: (r) => [
        { label: 'hash probes', value: fmtNum(r.c.probes), hint: 'One per access, plus one each to insert and remove a tag.' },
        { label: 'pointer derefs', value: '0', hint: 'Every reference is a page id that must be translated.' },
        { label: 'page I/Os', value: fmtNum(r.c.ios), hint: 'Misses that had to read the page.' },
        { label: 'write-backs', value: fmtNum(r.c.writes), hint: 'Dirty victims written before their frame was reused.' },
      ],
    },
    {
      id: 'swizzle',
      title: 'Swizzling pool',
      sub: 'LeanStore · Umbra',
      rows: (r) => [
        { label: 'hash probes', value: fmtNum(r.c.probes), hint: 'Only pages caught in the cooling stage cost a probe.' },
        { label: 'pointer derefs', value: fmtNum(r.c.derefs), hint: 'A swizzled swip is the child frame’s address.' },
        { label: 'page I/Os', value: fmtNum(r.c.ios), hint: 'The unswizzled swip holds the page id — no lookup needed to find it.' },
        { label: 'unswizzles', value: fmtNum(r.c.unswizzles), hint: 'Evictions that had to walk to the parent and rewrite its swip.' },
      ],
    },
  ];

  const results: Record<string, LaneResult> = { inmem, hash, swizzle: swz };

  const colW = 236;
  const gap = 14;
  const svgW = Math.max(width, 3 * colW + 2 * gap + 8);
  const cols = 12;
  const cell = 15;
  const pad = 2;
  const gridTop = 196;
  const gridH = Math.ceil(PAGES / cols) * (cell + pad);
  const barTop = gridTop + gridH + 26;
  const height = barTop + 48;
  const maxNs = Math.max(inmem.c.ns, hash.c.ns, swz.c.ns, 1);

  const perAccess = (r: LaneResult) => (r.c.served > 0 ? r.c.ns / r.c.served : 0);
  const hitRate = (r: LaneResult) => (r.c.hits + r.c.misses > 0 ? (r.c.hits / (r.c.hits + r.c.misses)) * 100 : 0);

  const note = (() => {
    if (n === 0)
      return {
        head: 'Nothing has run yet.',
        body:
          'Set a memory budget and a skew, then step the trace. All three engines see the same ' +
          `${TRACE_LEN} accesses against the same ${PAGES}-page dataset — only the way they turn a page ` +
          'reference into an address differs.',
      };
    if (inmem.deadAt !== null)
      return {
        head: `The in-memory engine stopped at access ${inmem.deadAt}.`,
        body:
          `It had ${frames} frames of RAM and the workload eventually touched more distinct pages than ` +
          'that. There is no eviction path to fall back on: Redis with the default ' +
          "maxmemory-policy = noeviction answers OOM command not allowed when used memory > 'maxmemory'; " +
          'SQL Server refuses the DML when the memory-optimized table’s bound resource pool is full; ' +
          `VoltDB simply has nowhere to put the row. Meanwhile the two buffer-managed engines served all ` +
          `${n} accesses at ${hitRate(hash).toFixed(0)}% and ${hitRate(swz).toFixed(0)}% hit rate. ` +
          'Switch to anti-caching to see the other answer.',
      };
    if (frames >= PAGES)
      return {
        head: 'Everything fits — and the page table is pure overhead.',
        body:
          `The hash pool has paid ${fmtNum(hash.c.probes)} probes to translate ${fmtNum(n)} references it ` +
          `already had in memory; the swizzling pool paid ${fmtNum(swz.c.probes)} and the in-memory engine ` +
          `none at all. That is ${fmtTime(perAccess(hash))} per access against ${fmtTime(perAccess(swz))} ` +
          `and ${fmtTime(perAccess(inmem))}. Shrink memory below the working set and watch which of the ` +
          `three survives.`,
      };
    if (mode === 'anticache')
      return {
        head: `Anti-caching traded ${fmtNum(inmem.c.aborts)} aborts for staying alive.`,
        body:
          `Every one of those transactions had already started executing when it reached an evicted tuple. ` +
          'H-Store runs one thread per partition and cannot block it on an I/O, so the transaction is ' +
          `rolled back, the ${BLOCK_PAGES}-page block is fetched and merged, and the transaction runs again ` +
          `— ${fmtTime(perAccess(inmem))} per access on average against ${fmtTime(perAccess(hash))} for the ` +
          'hash pool. The index never left memory: only tuples are evicted.',
      };
    return {
      head: `${fmtNum(hash.c.probes - swz.c.probes)} hash probes avoided.`,
      body:
        `At ${memPct}% of the dataset resident, the swizzling pool reached ${fmtNum(swz.c.derefs)} of its ` +
        `${fmtNum(n)} accesses through a raw pointer with no lookup, and paid for it with ` +
        `${fmtNum(swz.c.unswizzles)} unswizzles — every eviction had to latch the victim’s parent and ` +
        `rewrite the reference. The hash pool never walks to a parent, and never stops probing.`,
    };
  })();

  return (
    <VizPanel
      title="The same trace, three ways of finding a page"
      subtitle={`${PAGES} pages stand in for the whole dataset. Shrink RAM below the working set and the differences stop being about nanoseconds.`}
      controls={
        <>
          <Slider
            label="Resident memory"
            min={8}
            max={100}
            step={1}
            value={memPct}
            onChange={setMemPct}
            format={(v) => `${v}% · ${Math.max(8, Math.round((PAGES * v) / 100))} frames`}
          />
          <Slider
            label="Access skew"
            min={0}
            max={100}
            step={5}
            value={skew}
            onChange={setSkew}
            format={() => `${hot10}% of accesses on 10 pages`}
          />
          <Segmented
            label="When it does not fit"
            value={mode}
            onChange={setMode}
            options={[
              { value: 'strict', label: 'OOM', title: 'A pure in-memory engine has no spill path' },
              { value: 'anticache', label: 'anti-cache', title: 'Evict cold tuples in blocks; abort and restart on a touch' },
            ]}
          />
          <Button onClick={() => setN((v) => Math.min(TRACE_LEN, v + 1))} disabled={n >= TRACE_LEN} primary>
            Step
          </Button>
          <Button onClick={() => setN((v) => Math.min(TRACE_LEN, v + 50))} disabled={n >= TRACE_LEN}>
            +50 accesses
          </Button>
          <Button onClick={() => setN(TRACE_LEN)} disabled={n >= TRACE_LEN}>
            Run the trace
          </Button>
          <Button onClick={() => setN(0)} disabled={n === 0}>
            Reset
          </Button>
        </>
      }
      legend={
        <Legend
          items={[
            { label: 'resident, pointer-reachable', color: FILL.resident },
            { label: 'resident, dirty', color: FILL.dirty },
            { label: 'cooling — unswizzled, still in RAM (striped)', color: FILL.cooling },
            { label: 'in an anti-cache block (dotted)', color: FILL.block },
            { label: 'not resident', color: FILL.disk },
          ]}
        />
      }
      stats={
        <Stats
          items={[
            {
              label: 'Working set (95%)',
              value: `${workingSet} pages · ${fmtBytes(workingSet * PAGE_BYTES)}`,
              hint: 'Distinct pages that absorb 95% of the trace',
            },
            {
              label: 'Memory budget',
              value: `${frames} frames · ${fmtBytes(frames * PAGE_BYTES)}`,
              hint: `The dataset is ${fmtBytes(PAGES * PAGE_BYTES)}`,
            },
            {
              label: mode === 'strict' ? 'No pool' : 'Anti-caching',
              value: inmem.deadAt !== null ? 'out of memory' : fmtTime(perAccess(inmem)),
              hint: 'Average cost per access',
            },
            { label: 'Hash-table pool', value: fmtTime(perAccess(hash)), hint: 'Average cost per access' },
            { label: 'Swizzling pool', value: fmtTime(perAccess(swz)), hint: 'Average cost per access' },
            {
              label: 'Probes avoided',
              value: fmtNum(Math.max(0, hash.c.probes - swz.c.probes)),
              hint: 'Hash lookups the swizzling pool never had to do',
            },
          ]}
        />
      }
      note={
        <Note>
          <strong>{note.head}</strong> {note.body}
        </Note>
      }
      table={
        <>
          <table className="viz-table">
            <thead>
              <tr>
                <th>Engine</th>
                <th>Served</th>
                <th>Hit rate</th>
                <th>Hash probes</th>
                <th>Pointer derefs</th>
                <th>I/Os</th>
                <th>Write-backs</th>
                <th>Aborts</th>
                <th>Unswizzles</th>
                <th>ns / access</th>
              </tr>
            </thead>
            <tbody>
              {(
                [
                  [mode === 'strict' ? 'No buffer pool' : 'Anti-caching', inmem],
                  ['Hash-table pool', hash],
                  ['Swizzling pool', swz],
                ] as const
              ).map(([name, r]) => (
                <tr key={name}>
                  <td>{name}</td>
                  <td>{fmtNum(r.c.served)}</td>
                  <td>{r.c.served ? `${hitRate(r).toFixed(1)}%` : '—'}</td>
                  <td>{fmtNum(r.c.probes)}</td>
                  <td>{fmtNum(r.c.derefs)}</td>
                  <td>{fmtNum(r.c.ios)}</td>
                  <td>{fmtNum(r.c.writes)}</td>
                  <td>{fmtNum(r.c.aborts)}</td>
                  <td>{fmtNum(r.c.unswizzles)}</td>
                  <td>{r.c.served ? fmtTime(perAccess(r)) : '—'}</td>
                </tr>
              ))}
            </tbody>
          </table>
          <table className="viz-table">
            <thead>
              <tr>
                <th>Modelled operation</th>
                <th>Cost</th>
                <th>Who pays it</th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <td>Row dereference, no page table</td>
                <td>{fmtTime(COST.deref)}</td>
                <td>In-memory engines, per access</td>
              </tr>
              <tr>
                <td>Swizzled swip: tag test + optimistic validate</td>
                <td>{fmtTime(COST.swizzled)}</td>
                <td>LeanStore, per resident access</td>
              </tr>
              <tr>
                <td>Page-table probe under a partition latch</td>
                <td>{fmtTime(COST.probe)}</td>
                <td>Hash pool, per access</td>
              </tr>
              <tr>
                <td>Cooling-stage probe + re-swizzle</td>
                <td>{fmtTime(COST.coolProbe)}</td>
                <td>LeanStore, only for cooling pages</td>
              </tr>
              <tr>
                <td>Unswizzle: latch parent, rewrite the swip</td>
                <td>{fmtTime(COST.unswizzle)}</td>
                <td>LeanStore, per eviction</td>
              </tr>
              <tr>
                <td>Random 8 KB read</td>
                <td>{fmtTime(COST.io)}</td>
                <td>Every miss</td>
              </tr>
              <tr>
                <td>Abort, requeue and re-execute</td>
                <td>{fmtTime(COST.abort)}</td>
                <td>Anti-caching, per evicted-tuple touch</td>
              </tr>
            </tbody>
          </table>
        </>
      }
    >
      <div ref={ref}>
        <TooltipHost>
          <svg width={svgW} height={height} role="img" aria-label="Three buffer-management designs running the same access trace">
            {LANES.map((lane, li) => {
              const r = results[lane.id];
              const x = li * (colW + gap);
              const rows = lane.rows(r);
              return (
                <g key={lane.id}>
                  <rect x={x} y={0} width={colW} height={height - 6} rx={10} fill="var(--viz-plane)" stroke="var(--viz-border)" />
                  <text x={x + 12} y={24} fill="var(--viz-ink)" fontWeight={600}>
                    {lane.title}
                  </text>
                  <text x={x + 12} y={41} fill="var(--viz-ink-muted)">
                    {lane.sub}
                  </text>

                  {rows.map((row, ri) => {
                    const y = 56 + ri * 30;
                    return (
                      <g key={row.label} {...tip(<span>{row.hint}</span>)}>
                        <rect x={x + 12} y={y} width={colW - 24} height={24} rx={6} fill="var(--viz-surface)" stroke="var(--viz-border)" />
                        <text x={x + 20} y={y + 16} fill="var(--viz-ink-2)">
                          {row.label}
                        </text>
                        <text x={x + colW - 20} y={y + 16} textAnchor="end" fill="var(--viz-ink)" fontWeight={600}>
                          {row.value}
                        </text>
                      </g>
                    );
                  })}

                  <text x={x + 12} y={gridTop - 8} fill="var(--viz-ink-muted)">
                    page 0 (hottest) → page {PAGES - 1}
                  </text>
                  {r.cells.map((st, p) => {
                    const cx = x + 12 + (p % cols) * (cell + pad);
                    const cy = gridTop + Math.floor(p / cols) * (cell + pad);
                    return (
                      <g
                        key={p}
                        {...tip(
                          <span>
                            <strong>page {p}</strong> — {freq[p]} accesses in the trace. {CELL_HINT[st]}
                          </span>,
                        )}
                      >
                        <rect
                          x={cx}
                          y={cy}
                          width={cell}
                          height={cell}
                          rx={3}
                          fill={FILL[st]}
                          stroke={st === 'disk' ? 'var(--viz-border)' : 'var(--viz-surface)'}
                        />
                        {st === 'cooling' ? (
                          <line x1={cx + 2} y1={cy + cell - 2} x2={cx + cell - 2} y2={cy + 2} stroke="var(--viz-ink)" strokeWidth={1.4} />
                        ) : null}
                        {st === 'block' ? <circle cx={cx + cell / 2} cy={cy + cell / 2} r={2.2} fill="var(--viz-surface)" /> : null}
                      </g>
                    );
                  })}

                  {lane.id === 'inmem' && r.deadAt !== null ? (
                    <g>
                      <rect x={x + 12} y={gridTop + gridH / 2 - 15} width={colW - 24} height={30} rx={6} fill="var(--viz-critical)" />
                      <text x={x + colW / 2} y={gridTop + gridH / 2 + 5} textAnchor="middle" fill="var(--viz-surface)" fontWeight={600}>
                        out of memory at #{r.deadAt}
                      </text>
                    </g>
                  ) : null}

                  <text x={x + 12} y={barTop - 6} fill="var(--viz-ink-muted)">
                    total time · {r.c.served ? fmtTime(perAccess(r)) : '—'} per access
                  </text>
                  <rect x={x + 12} y={barTop} width={colW - 24} height={14} rx={4} fill="var(--viz-neutral)" />
                  <rect
                    x={x + 12}
                    y={barTop}
                    width={Math.max(0, ((colW - 24) * r.c.ns) / maxNs)}
                    height={14}
                    rx={4}
                    fill="var(--viz-seq-400)"
                  />
                  <text x={x + 12} y={barTop + 32} fill="var(--viz-ink-2)">
                    {fmtTime(r.c.ns)} for {fmtNum(r.c.served)} accesses
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
