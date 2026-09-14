import { useMemo, useState } from 'react';
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
  fmtNum,
  fmtTime,
  makeRng,
  useSize,
} from './Viz';

/**
 * Who owns the copy, and who owns the ordering.
 *
 * The same page trace runs through three engine designs over one fixed RAM budget:
 *
 *  - pool   — an engine cache over a *buffered* file. Every miss lands in the page cache
 *             first and is then memcpy-ed into a frame, so hot pages occupy RAM twice and
 *             every miss costs a copy the device never asked for.
 *  - direct — O_DIRECT. The DMA lands in the frame; the page cache holds nothing of the
 *             data file, so RAM you did not give the pool is RAM the data file cannot use.
 *  - mmap   — no pool at all. A store dirties a page-cache page directly, and the kernel
 *             may write that page to the data file whenever it likes — including before the
 *             WAL record describing the change is durable. That is the write-ahead rule
 *             broken by a layer that has never heard of it.
 */

const RAM = 16; // frames of RAM in this model host
const DATASET = 32; // pages in the data file
const MEMCPY_NS = 700; // one 8 KiB copy, ~12 GB/s
const READ_NS = 90_000; // one 8 KiB random read from an NVMe SSD
const MINOR_NS = 1_200; // a minor fault: install a PTE, no device

/** Skewed page trace — a few hot pages, a long tail. Deterministic across SSR and hydrate. */
const TRACE = (() => {
  const rng = makeRng(20220112); // CIDR 2022, the mmap paper
  return Array.from({ length: 240 }, () => Math.min(DATASET - 1, Math.floor(DATASET * Math.pow(rng(), 2.3))));
})();

type Mode = 'pool' | 'direct' | 'mmap';
type Frame = { page: number; dirty: boolean; lsn: number; t: number };

type Row = {
  n: number;
  op: string;
  page: string;
  distinct: number;
  dup: number;
  memcpy: number;
  reads: number;
  viol: number;
};

type S = {
  pool: Frame[];
  cache: Frame[];
  disk: Record<number, number>;
  diskOrder: number[];
  ahead: number[]; // pages on disk carrying a change the WAL never recorded
  walLsn: number;
  durableLsn: number;
  t: number;
  idx: number;
  reads: number;
  memcpys: number;
  minor: number;
  major: number;
  shoot: number;
  violations: number;
  lost: number[];
  crashed: boolean;
  head: string;
  body: string;
  log: Row[];
};

const INITIAL: S = {
  pool: [],
  cache: [],
  disk: {},
  diskOrder: [],
  ahead: [],
  walLsn: 0,
  durableLsn: 0,
  t: 1,
  idx: 0,
  reads: 0,
  memcpys: 0,
  minor: 0,
  major: 0,
  shoot: 0,
  violations: 0,
  lost: [],
  crashed: false,
  head: 'One RAM budget, three ways to spend it.',
  body:
    'Read a page and watch where the copies land. In pool mode every miss arrives in the page cache and is ' +
    'then copied into a frame, so the hot pages sit in RAM twice. Switch to O_DIRECT and the second copy ' +
    'disappears — along with any use for the RAM you did not give the pool. Switch to mmap and the pool ' +
    'disappears instead, and so does your control over when a dirty page reaches the disk.',
  log: [],
};

type Caps = { poolCap: number; cacheCap: number };

function caps(mode: Mode, poolSlots: number): Caps {
  if (mode === 'mmap') return { poolCap: 0, cacheCap: RAM };
  if (mode === 'direct') return { poolCap: poolSlots, cacheCap: 0 };
  return { poolCap: poolSlots, cacheCap: RAM - poolSlots };
}

type Op = 'read' | 'write' | 'wal' | 'writeback' | 'crash';

const OP_LABEL: Record<Op, string> = {
  read: 'read page',
  write: 'modify page',
  wal: 'fsync(WAL)',
  writeback: 'kernel writeback',
  crash: 'power cut',
};

function lru(list: Frame[]): number {
  let best = 0;
  for (let i = 1; i < list.length; i++) if (list[i].t < list[best].t) best = i;
  return best;
}

function apply(s: S, op: Op, mode: Mode, poolSlots: number): S {
  const { poolCap, cacheCap } = caps(mode, poolSlots);
  const n = { ...s, pool: [...s.pool], cache: [...s.cache], disk: { ...s.disk }, diskOrder: [...s.diskOrder], ahead: [...s.ahead] };
  let head = '';
  let body = '';
  let pageLabel = '—';

  /** Kernel flusher writes one dirty page-cache page. It knows nothing about the WAL. */
  const kernelWrite = (f: Frame) => {
    n.disk[f.page] = f.lsn;
    n.diskOrder = [...n.diskOrder.filter((p) => p !== f.page), f.page];
    if (f.lsn > n.durableLsn) {
      n.violations += 1;
      if (!n.ahead.includes(f.page)) n.ahead = [...n.ahead, f.page];
    }
  };

  /** The engine writes one of its own dirty frames. It flushes the log first — that is the rule. */
  const engineWrite = (f: Frame, viaPageCache: boolean) => {
    if (f.lsn > n.durableLsn) n.durableLsn = f.lsn; // WAL-before-data: flush the log through this page's LSN first
    if (viaPageCache) {
      // write() is a memcpy into a page-cache page that is then dirty on the kernel's books
      n.memcpys += 1;
      if (cacheCap > 0) {
        const at = n.cache.findIndex((c) => c.page === f.page);
        if (at >= 0) n.cache[at] = { ...n.cache[at], dirty: true, lsn: f.lsn, t: n.t };
        else {
          if (n.cache.length >= cacheCap) {
            const v = n.cache[lru(n.cache)];
            if (v.dirty) kernelWrite(v);
            n.cache.splice(lru(n.cache), 1);
          }
          n.cache.push({ page: f.page, dirty: true, lsn: f.lsn, t: n.t });
        }
      } else {
        n.disk[f.page] = f.lsn;
        n.diskOrder = [...n.diskOrder.filter((p) => p !== f.page), f.page];
      }
    } else {
      n.disk[f.page] = f.lsn;
      n.diskOrder = [...n.diskOrder.filter((p) => p !== f.page), f.page];
    }
  };

  /** Make `page` resident wherever this mode keeps pages. Returns a one-line description. */
  const fetch = (page: number): string => {
    n.t += 1;
    if (mode === 'mmap') {
      const at = n.cache.findIndex((c) => c.page === page);
      if (at >= 0) {
        n.cache[at] = { ...n.cache[at], t: n.t };
        n.minor += 1;
        return 'minor fault — the folio was already in the page cache, so the kernel only installed a PTE';
      }
      if (n.cache.length >= cacheCap) {
        const i = lru(n.cache);
        const v = n.cache[i];
        if (v.dirty) kernelWrite(v);
        n.cache.splice(i, 1);
        n.shoot += 1;
      }
      n.cache.push({ page, dirty: false, lsn: n.disk[page] ?? 0, t: n.t });
      n.major += 1;
      n.reads += 1;
      return 'major fault — the load instruction itself blocked on the device';
    }

    const inPool = n.pool.findIndex((f) => f.page === page);
    if (inPool >= 0) {
      n.pool[inPool] = { ...n.pool[inPool], t: n.t };
      return 'pool hit — no syscall, no copy';
    }
    if (n.pool.length >= poolCap) {
      const i = lru(n.pool);
      const v = n.pool[i];
      if (v.dirty) engineWrite(v, mode === 'pool');
      n.pool.splice(i, 1);
    }
    if (mode === 'direct') {
      n.pool.push({ page, dirty: false, lsn: n.disk[page] ?? 0, t: n.t });
      n.reads += 1;
      return 'pool miss — pread(O_DIRECT) DMA-ed the block straight into the frame, no copy';
    }
    const c = n.cache.findIndex((f) => f.page === page);
    if (c >= 0) {
      n.cache[c] = { ...n.cache[c], t: n.t };
      n.memcpys += 1;
      n.pool.push({ page, dirty: false, lsn: n.cache[c].lsn, t: n.t });
      return 'pool miss, page-cache hit — no device I/O, but one 8 KiB memcpy and now two copies in RAM';
    }
    if (n.cache.length >= cacheCap) {
      const i = lru(n.cache);
      const v = n.cache[i];
      if (v.dirty) kernelWrite(v);
      n.cache.splice(i, 1);
    }
    n.cache.push({ page, dirty: false, lsn: n.disk[page] ?? 0, t: n.t });
    n.memcpys += 1;
    n.reads += 1;
    n.pool.push({ page, dirty: false, lsn: n.disk[page] ?? 0, t: n.t });
    return 'pool miss, page-cache miss — one device read into the page cache, then a memcpy into the frame';
  };

  switch (op) {
    case 'read': {
      const page = TRACE[s.idx % TRACE.length];
      n.idx = s.idx + 1;
      pageLabel = `p${page}`;
      const what = fetch(page);
      head = `Read p${page}: ${what}.`;
      body =
        mode === 'pool'
          ? 'Every page the engine caches was in the page cache a moment earlier, and the kernel has no reason ' +
            'to drop it: the duplicated frames are the price of a buffered read path. PostgreSQL accepts this ' +
            'and sizes shared_buffers at roughly a quarter of RAM so the kernel can hold the rest.'
          : mode === 'direct'
            ? 'O_DIRECT removes the page-cache copy and the memcpy with it — the drive DMAs into the frame the ' +
              'engine already owns. The engine now owns readahead, alignment and I/O concurrency too, because ' +
              'the kernel is no longer doing any of it for this file.'
            : 'There is no pool. Residency is a property of the page table, discovered by the MMU in the middle ' +
              'of your load instruction, and eviction is a decision the kernel makes about a page it does not ' +
              'know is a B-tree root.';
      break;
    }

    case 'write': {
      const page = TRACE[s.idx % TRACE.length];
      n.idx = s.idx + 1;
      pageLabel = `p${page}`;
      const what = fetch(page);
      n.walLsn += 1;
      const lsn = n.walLsn;
      if (mode === 'mmap') {
        const at = n.cache.findIndex((c) => c.page === page);
        n.cache[at] = { ...n.cache[at], dirty: true, lsn, t: n.t };
        head = `Modify p${page} at LSN ${lsn} (${what}).`;
        body =
          'The store went into a page-cache folio and set PG_dirty. The WAL record is still in your buffer. ' +
          'From here the kernel flusher, memory reclaim, or your own eviction of a neighbouring page can push ' +
          `p${page} to the data file at any moment — press "kernel writeback" before "fsync(WAL)" and watch it happen.`;
      } else {
        const at = n.pool.findIndex((f) => f.page === page);
        n.pool[at] = { ...n.pool[at], dirty: true, lsn, t: n.t };
        head = `Modify p${page} at LSN ${lsn} (${what}).`;
        body =
          'The dirty page exists only inside the engine’s own frame. Nothing outside the engine can see it, ' +
          'so nothing outside the engine can write it out of order: the page cannot reach the data file until ' +
          'the engine evicts or checkpoints it, and the engine flushes the log first.';
      }
      break;
    }

    case 'wal': {
      const behind = n.walLsn - n.durableLsn;
      n.durableLsn = n.walLsn;
      head = behind > 0 ? `WAL flushed: durable through LSN ${n.walLsn}.` : 'WAL flushed: nothing was pending.';
      body =
        behind > 0
          ? `${behind} log record${behind === 1 ? '' : 's'} reached the media. Every dirty page whose change is ` +
            'now covered by the log may safely be written to the data file, in any order, by anyone. This is the ' +
            'only moment at which that becomes true.'
          : 'The log was already durable up to the last change, so the barrier cost you a flush and bought nothing.';
      break;
    }

    case 'writeback': {
      const dirty = n.cache.filter((c) => c.dirty);
      if (dirty.length === 0) {
        head = mode === 'direct' ? 'Nothing to write back: the data file has no page-cache pages.' : 'No dirty page-cache pages.';
        body =
          mode === 'direct'
            ? 'With O_DIRECT the data file is never in the page cache, so the kernel flusher has no opinion about ' +
              'it at all. Every byte that reaches this file is written by the engine, at a moment the engine chose.'
            : mode === 'pool'
              ? 'In pool mode a data page only becomes dirty in the page cache when the engine calls write() on it — ' +
                'and the engine flushes the log through that page’s LSN first. The kernel flusher can only ever ' +
                'see dirty pages that are already safe to write.'
              : 'No mapped page is dirty yet. Modify one first.';
        break;
      }
      let bad = 0;
      for (let i = 0; i < n.cache.length; i++) {
        if (!n.cache[i].dirty) continue;
        if (n.cache[i].lsn > n.durableLsn) bad += 1;
        kernelWrite(n.cache[i]);
        n.cache[i] = { ...n.cache[i], dirty: false };
      }
      head =
        bad > 0
          ? `Writeback broke the write-ahead rule on ${bad} page${bad === 1 ? '' : 's'}.`
          : `Writeback wrote ${dirty.length} page${dirty.length === 1 ? '' : 's'}, all of them already covered by the log.`;
      body =
        bad > 0
          ? `The flusher wrote a page carrying a change at an LSN above the durable LSN (${n.durableLsn}). There is ` +
            'no msync flag, no ordering hint and no "not yet" — the kernel writes dirty folios on its own schedule ' +
            'and has never heard of write-ahead logging. Pull the plug now and see what recovery can do with it.'
          : 'Every page written had its log record on the media already, so the data file is a legal, recoverable ' +
            'state. That ordering is the entire job the engine’s eviction path exists to do.';
      break;
    }

    case 'crash': {
      const lost = n.diskOrder.filter((p) => (n.disk[p] ?? 0) > n.durableLsn);
      const unflushed = n.walLsn - n.durableLsn;
      n.lost = lost;
      n.crashed = true;
      n.pool = [];
      n.cache = [];
      head =
        lost.length > 0
          ? `Recovery cannot repair ${lost.length} page${lost.length === 1 ? '' : 's'}: ${lost.map((p) => `p${p}`).join(', ')}.`
          : 'Clean recovery: the data file is behind the log, which is exactly where it is allowed to be.';
      body =
        lost.length > 0
          ? `Redo replays the WAL to LSN ${n.durableLsn}. The data file holds a change at a higher LSN that no log ` +
            'record describes, so redo cannot reproduce it and undo has no before-image to restore. The page checksum ' +
            'is perfectly valid — the block is internally consistent and semantically wrong, which is the worst kind of ' +
            'corruption there is.'
          : `Everything in RAM is gone; ${unflushed} uncommitted change${unflushed === 1 ? '' : 's'} went with it, which ` +
            'is correct — those transactions never committed. Redo rolls the data file forward from the last ' +
            'checkpoint using log records that are all on the media.';
      break;
    }
  }

  const dup = n.pool.filter((f) => n.cache.some((c) => c.page === f.page)).length;
  const distinct = new Set([...n.pool.map((f) => f.page), ...n.cache.map((c) => c.page)]).size;

  return {
    ...n,
    head,
    body,
    log: [
      ...s.log,
      {
        n: s.log.length + 1,
        op: OP_LABEL[op],
        page: pageLabel,
        distinct,
        dup,
        memcpy: n.memcpys,
        reads: n.reads,
        viol: n.violations,
      },
    ],
  };
}

/* -------------------------------------------------------------- the drawing */

type ChipKind = 'clean' | 'dirty' | 'dup' | 'ahead' | 'lost';

const CHIP: Record<ChipKind, { fill: string; stroke: string; dash?: string; glyph: string }> = {
  clean: { fill: 'var(--viz-clean)', stroke: 'var(--viz-surface)', glyph: '' },
  dirty: { fill: 'var(--viz-dirty)', stroke: 'var(--viz-surface)', glyph: 'D' },
  dup: { fill: 'var(--viz-plane)', stroke: 'var(--viz-ink-2)', dash: '3 2', glyph: '2×' },
  ahead: { fill: 'var(--viz-critical)', stroke: 'var(--viz-surface)', glyph: '!' },
  lost: { fill: 'var(--viz-stale)', stroke: 'var(--viz-surface)', glyph: '✗' },
};

type Chip = { key: string; kind: ChipKind; label: string; hint: string };

export default function CachePathOwnershipLab() {
  const [mode, setMode] = useState<Mode>('pool');
  const [poolSlots, setPoolSlots] = useState(4);
  const [s, setS] = useState<S>(INITIAL);
  const [ref, width] = useSize(760);
  const tip = useTip();

  const { poolCap, cacheCap } = caps(mode, poolSlots);

  const run = (op: Op) => setS((cur) => apply(cur, op, mode, poolSlots));
  const switchMode = (m: Mode) => {
    setMode(m);
    setS(INITIAL);
  };

  const dupPages = useMemo(
    () => new Set(s.pool.filter((f) => s.cache.some((c) => c.page === f.page)).map((f) => f.page)),
    [s.pool, s.cache],
  );
  const distinct = useMemo(
    () => new Set([...s.pool.map((f) => f.page), ...s.cache.map((c) => c.page)]).size,
    [s.pool, s.cache],
  );

  const lanes: { name: string; sub: string; cap: number | null; chips: Chip[] }[] = [
    {
      name: 'Engine pool',
      sub: mode === 'mmap' ? 'none — mmap deletes this layer' : 'shared_buffers / innodb_buffer_pool_size',
      cap: poolCap,
      chips: s.pool.map((f) => ({
        key: `pool${f.page}`,
        kind: f.dirty ? 'dirty' : dupPages.has(f.page) ? 'dup' : 'clean',
        label: `p${f.page}`,
        hint: f.dirty
          ? `Dirty in a frame the engine owns, change at LSN ${f.lsn}. Nothing outside the engine can write it.`
          : dupPages.has(f.page)
            ? 'This page is in the pool *and* in the page cache. Two frames of RAM, one page of data.'
            : 'Clean frame. The engine can evict it without any I/O.',
      })),
    },
    {
      name: 'OS page cache',
      sub: cacheCap === 0 ? 'holds nothing of this file under O_DIRECT' : 'kernel LRU, kernel writeback, kernel schedule',
      cap: cacheCap,
      chips: s.cache.map((c) => ({
        key: `cache${c.page}`,
        kind: c.dirty ? 'dirty' : dupPages.has(c.page) ? 'dup' : 'clean',
        label: `p${c.page}`,
        hint: c.dirty
          ? `Dirty folio at LSN ${c.lsn}. ${
              c.lsn > s.durableLsn
                ? 'The log record for this change is NOT durable yet, and the flusher does not care.'
                : 'Its log record is already durable, so writing it is safe.'
            }`
          : dupPages.has(c.page)
            ? 'A clean second copy of a page the engine already holds — double buffering, in one picture.'
            : 'Clean page-cache folio.',
      })),
    },
    {
      name: 'Data file on disk',
      sub: 'what recovery will find',
      cap: null,
      chips: s.diskOrder.slice(-12).map((p) => ({
        key: `disk${p}`,
        kind: s.lost.includes(p) ? 'lost' : s.ahead.includes(p) && (s.disk[p] ?? 0) > s.durableLsn ? 'ahead' : 'clean',
        label: `p${p}`,
        hint:
          (s.disk[p] ?? 0) > s.durableLsn
            ? `Written at LSN ${s.disk[p]} while the log is durable only to ${s.durableLsn}. Unrecoverable either way.`
            : `On disk at LSN ${s.disk[p]}, covered by the log.`,
      })),
    },
    {
      name: 'WAL on disk',
      sub: `durable to LSN ${s.durableLsn} of ${s.walLsn}`,
      cap: null,
      chips: Array.from({ length: Math.min(s.walLsn, 12) }, (_, i) => {
        const lsn = s.walLsn - Math.min(s.walLsn, 12) + i + 1;
        return {
          key: `wal${lsn}`,
          kind: (lsn <= s.durableLsn ? 'clean' : 'dirty') as ChipKind,
          label: `#${lsn}`,
          hint: lsn <= s.durableLsn ? 'On the media. Redo can replay this.' : 'Still in the WAL buffer. A crash loses it.',
        };
      }),
    },
  ];

  const labelW = 156;
  const chipW = 38;
  const chipGap = 6;
  const slots = Math.max(RAM, ...lanes.map((l) => l.chips.length));
  const svgW = Math.max(width, labelW + slots * (chipW + chipGap) + 24);
  const barH = 30;
  const laneH = 58;
  const laneGap = 10;
  const top = 8;
  const laneY = (i: number) => top + barH + 22 + i * (laneH + laneGap);
  const height = laneY(lanes.length - 1) + laneH + 8;

  const copyNs = s.memcpys * MEMCPY_NS;
  const ioNs = s.reads * READ_NS + s.minor * MINOR_NS;

  /* RAM bar: every frame of the budget, coloured by who is using it and for what. */
  const barSeg: { w: number; fill: string; label: string; dash?: boolean }[] = [];
  const poolUsed = s.pool.length;
  const dupCount = dupPages.size;
  const cacheDistinct = s.cache.filter((c) => !dupPages.has(c.page)).length;
  if (poolUsed) barSeg.push({ w: poolUsed, fill: 'var(--viz-1)', label: `pool ${poolUsed}` });
  if (poolCap - poolUsed > 0) barSeg.push({ w: poolCap - poolUsed, fill: 'var(--viz-neutral)', label: 'pool free', dash: true });
  if (dupCount) barSeg.push({ w: dupCount, fill: 'var(--viz-2)', label: `duplicated ${dupCount}` });
  if (cacheDistinct) barSeg.push({ w: cacheDistinct, fill: 'var(--viz-3)', label: `cache ${cacheDistinct}` });
  const usedBar = poolCap + dupCount + cacheDistinct;
  if (RAM - usedBar > 0) {
    barSeg.push({
      w: RAM - usedBar,
      fill: 'var(--viz-neutral)',
      label: mode === 'direct' ? 'unused by this file' : 'page cache free',
      dash: true,
    });
  }
  const unitW = (svgW - labelW - 16) / RAM;

  return (
    <VizPanel
      title="Who owns the copy, and who owns the ordering"
      subtitle="One 16-frame RAM budget and one skewed page trace, run three ways. Watch duplicate copies and memcpys appear in pool mode and vanish under O_DIRECT — then, in mmap mode, modify a page, let the kernel write it back before you flush the WAL, and pull the plug."
      controls={
        <>
          <Segmented
            label="I/O path"
            value={mode}
            onChange={switchMode}
            options={[
              { value: 'pool', label: 'pool + buffered', title: 'An engine cache over ordinary pread/pwrite — PostgreSQL' },
              { value: 'direct', label: 'O_DIRECT', title: 'innodb_flush_method=O_DIRECT, RocksDB use_direct_reads, PG debug_io_direct' },
              { value: 'mmap', label: 'mmap', title: 'No pool: the page cache is the cache — LMDB, MMAPv1' },
            ]}
          />
          <Slider
            label="RAM to the engine pool"
            min={1}
            max={RAM - 1}
            value={poolSlots}
            onChange={(v) => {
              setPoolSlots(v);
              setS(INITIAL);
            }}
            disabled={mode === 'mmap'}
            format={(v) => `${v}/${RAM} frames (${Math.round((v / RAM) * 100)}%)`}
          />
          <Button onClick={() => run('read')} disabled={s.crashed} primary>
            Read page
          </Button>
          <Button onClick={() => run('write')} disabled={s.crashed} title="Append a WAL record and dirty the page">
            Modify page
          </Button>
          <Button onClick={() => run('wal')} disabled={s.crashed}>
            fsync(WAL)
          </Button>
          <Button onClick={() => run('writeback')} disabled={s.crashed} title="What the kernel flusher does on its own schedule">
            Kernel writeback
          </Button>
          <Button onClick={() => run('crash')} disabled={s.crashed}>
            Power cut
          </Button>
          <Button onClick={() => setS(INITIAL)}>{s.crashed ? 'Reboot' : 'Reset'}</Button>
        </>
      }
      legend={
        <Legend
          items={[
            { label: 'clean copy', color: CHIP.clean.fill },
            { label: 'D — dirty copy', color: CHIP.dirty.fill },
            { label: '2× — duplicated in both caches (dashed)', color: 'var(--viz-ink-2)', shape: 'line' },
            { label: '! — on disk ahead of the WAL', color: CHIP.ahead.fill },
            { label: '✗ — unrecoverable after the crash', color: CHIP.lost.fill },
          ]}
        />
      }
      stats={
        <Stats
          items={[
            {
              label: 'Distinct pages cached',
              value: `${distinct} / ${RAM} frames`,
              hint: 'How much of the data file your RAM is actually holding',
            },
            {
              label: 'RAM lost to duplicates',
              value: `${dupCount} (${Math.round((dupCount / RAM) * 100)}%)`,
              hint: 'Pages held in the engine pool and in the page cache at the same time',
            },
            { label: 'Device reads', value: fmtNum(s.reads), hint: '8 KiB random reads issued to the drive' },
            {
              label: 'memcpy',
              value: `${fmtNum(s.memcpys)} · ${fmtTime(copyNs)}`,
              hint: 'Copies between the page cache and the engine at ~700 ns per 8 KiB page',
            },
            {
              label: 'Faults (minor / major)',
              value: mode === 'mmap' ? `${fmtNum(s.minor)} / ${fmtNum(s.major)}` : '—',
              hint: 'A major fault stalls the faulting thread inside the load instruction',
            },
            { label: 'TLB shootdowns', value: mode === 'mmap' ? fmtNum(s.shoot) : '—', hint: 'One per eviction of a mapped page' },
            { label: 'I/O time', value: fmtTime(ioNs), hint: '90 µs per device read, 1.2 µs per minor fault' },
            {
              label: 'WAL violations',
              value: fmtNum(s.violations),
              hint: 'Pages the kernel wrote to the data file before their log record was durable',
            },
          ]}
        />
      }
      note={
        <Note>
          <strong>{s.head}</strong> {s.body}
        </Note>
      }
      table={
        <>
          <table className="viz-table">
            <thead>
              <tr>
                <th>#</th>
                <th>Operation</th>
                <th>Page</th>
                <th>Distinct cached</th>
                <th>Duplicated</th>
                <th>memcpys</th>
                <th>Device reads</th>
                <th>WAL violations</th>
              </tr>
            </thead>
            <tbody>
              {s.log.length === 0 ? (
                <tr>
                  <td colSpan={8}>No operations yet.</td>
                </tr>
              ) : (
                s.log.map((r) => (
                  <tr key={r.n}>
                    <td>{r.n}</td>
                    <td>{r.op}</td>
                    <td>{r.page}</td>
                    <td>{r.distinct}</td>
                    <td>{r.dup}</td>
                    <td>{r.memcpy}</td>
                    <td>{r.reads}</td>
                    <td>{r.viol}</td>
                  </tr>
                ))
              )}
            </tbody>
          </table>
          <table className="viz-table">
            <thead>
              <tr>
                <th>Mode</th>
                <th>Pool frames</th>
                <th>Page-cache frames for this file</th>
                <th>Copies per miss</th>
                <th>Who chooses the victim</th>
                <th>Who chooses when a page hits the disk</th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <td>pool + buffered</td>
                <td>{poolSlots}</td>
                <td>{RAM - poolSlots}</td>
                <td>2 (DMA, then memcpy)</td>
                <td>both, independently</td>
                <td>the engine</td>
              </tr>
              <tr>
                <td>O_DIRECT</td>
                <td>{poolSlots}</td>
                <td>0</td>
                <td>1 (DMA into the frame)</td>
                <td>the engine</td>
                <td>the engine</td>
              </tr>
              <tr>
                <td>mmap</td>
                <td>0</td>
                <td>{RAM}</td>
                <td>1 (DMA, then a page fault)</td>
                <td>the kernel</td>
                <td>the kernel</td>
              </tr>
            </tbody>
          </table>
        </>
      }
    >
      <div ref={ref}>
        <TooltipHost>
          <svg width={svgW} height={height} role="img" aria-label="Copies of data pages across the engine pool, the OS page cache, the data file and the WAL">
            {/* RAM budget bar */}
            <text x={0} y={top + 18} fill="var(--viz-ink)" fontWeight={600}>
              RAM budget
            </text>
            {(() => {
              let x = labelW;
              return barSeg.map((seg, i) => {
                const w = seg.w * unitW;
                const el = (
                  <g key={`${seg.label}-${i}`}>
                    <rect
                      x={x}
                      y={top}
                      width={Math.max(0, w - 2)}
                      height={barH}
                      rx={5}
                      fill={seg.fill}
                      stroke={seg.dash ? 'var(--viz-axis)' : 'var(--viz-surface)'}
                      strokeWidth={1.5}
                      strokeDasharray={seg.dash ? '3 3' : undefined}
                    />
                    {w > 58 ? (
                      <text x={x + 8} y={top + 19} fill={seg.dash ? 'var(--viz-ink-2)' : 'var(--viz-ink)'}>
                        {seg.label}
                      </text>
                    ) : null}
                  </g>
                );
                x += w;
                return el;
              });
            })()}

            {lanes.map((ln, i) => (
              <g key={ln.name}>
                <rect
                  x={labelW}
                  y={laneY(i)}
                  width={svgW - labelW - 8}
                  height={laneH}
                  rx={8}
                  fill="var(--viz-plane)"
                  stroke="var(--viz-border)"
                />
                <text x={0} y={laneY(i) + 24} fill="var(--viz-ink)" fontWeight={600}>
                  {ln.name}
                </text>
                <text x={0} y={laneY(i) + 40} fill="var(--viz-ink-muted)">
                  {ln.sub}
                </text>
                {ln.cap !== null ? (
                  <text x={svgW - 16} y={laneY(i) + 18} textAnchor="end" fill="var(--viz-ink-muted)">
                    {ln.chips.length} / {ln.cap} frames
                  </text>
                ) : null}
                {ln.chips.map((c, k) => {
                  const cx = labelW + 12 + k * (chipW + chipGap);
                  const st = CHIP[c.kind];
                  return (
                    <g key={c.key} {...tip(<>
                      <strong>{c.label}</strong>
                      <br />
                      {c.hint}
                    </>)} style={{ cursor: 'help' }}>
                      <rect
                        x={cx}
                        y={laneY(i) + 8}
                        width={chipW}
                        height={20}
                        rx={5}
                        fill={st.fill}
                        stroke={st.stroke}
                        strokeWidth={1.5}
                        strokeDasharray={st.dash}
                      />
                      <text x={cx + chipW / 2} y={laneY(i) + 44} textAnchor="middle" fill="var(--viz-ink)">
                        {c.label}
                        {st.glyph ? ` ${st.glyph}` : ''}
                      </text>
                    </g>
                  );
                })}
              </g>
            ))}
          </svg>
        </TooltipHost>
      </div>
    </VizPanel>
  );
}
