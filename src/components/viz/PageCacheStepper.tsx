import { useState } from 'react';
import {
  VizPanel,
  Segmented,
  Slider,
  Check,
  Button,
  Legend,
  Stats,
  Note,
  TooltipHost,
  useTip,
  makeRng,
  fmtTime,
  fmtNum,
  useSize,
} from './Viz';

/**
 * A step-through of the Linux buffered-I/O path.
 *
 * Models, at page granularity: the page cache (present / dirty / brought-by-readahead),
 * the on-demand readahead state machine (file_ra_state: start, size, async marker),
 * the mmap fault path (minor vs major faults, fault-around, the mmap_miss heuristic),
 * dirty accounting (dirty_background_ratio, dirty_ratio -> balance_dirty_pages),
 * the 5 s flusher waking and the 30 s dirty_expire rule, LRU reclaim, and the two
 * error paths: EIO from read(), SIGBUS from a load.
 *
 * Everything is deterministic: the only randomness is the "random offset" button,
 * driven by makeRng seeded from a counter held in state.
 */

const N_PAGES = 32;
const PAGE_BYTES = 4096;
const FAULT_AROUND_PAGES = 16; // fault_around_bytes = 64 KiB
const DIRTY_EXPIRE_S = 30; // dirty_expire_centisecs = 3000
const FLUSHER_INTERVAL_S = 5; // dirty_writeback_centisecs = 500
const MMAP_LOTSAMISS = 4; // Linux uses 100; scaled down so the give-up is reachable here
const CORES = 16;

/* cost model, nanoseconds */
const SYSCALL_NS = 500;
const COPY_NS = 400; // 4 KiB memcpy kernel <-> user
const LOAD_NS = 1; // resident mmap load, TLB hit
const MINOR_FAULT_NS = 1_200;
const DISK_FIRST_NS = 20_000; // NVMe random 4 KiB read
const DISK_NEXT_NS = 1_500; // each extra contiguous page in the same request
const DISK_WRITE_NS = 20_000;
const FLUSH_FIXED_NS = 300_000; // device cache flush inside fsync()
const SHOOTDOWN_BASE_NS = 3_000;
const SHOOTDOWN_PER_CORE_NS = 900;

type Pg = {
  present: boolean;
  dirty: boolean;
  dirtiedAt: number;
  used: number;
  ra: boolean; // pulled in by readahead and not yet demanded
  marker: boolean; // PG_readahead: touching it fires the next async window
  pte: 0 | 1 | 2; // 0 unmapped, 1 mapped read-only, 2 mapped writable
};

type Sim = {
  pages: Pg[];
  clock: number; // virtual seconds
  tick: number; // logical op counter, drives LRU
  raStart: number;
  raSize: number;
  prev: number; // previously accessed page, for sequential detection
  mmapMiss: number;
  reads: number; // 4 KiB device reads issued
  writes: number; // 4 KiB device writes issued
  raPagesIn: number;
  raUsed: number;
  raWasted: number; // readahead pages evicted without ever being demanded
  minor: number;
  major: number;
  throttles: number;
  shootdowns: number;
  evictMapped: number;
  ioNs: number;
  head: string;
  msg: string;
  dead: boolean;
  rngStep: number;
};

type Cfg = {
  cacheSize: number;
  raPages: number;
  bgPct: number;
  hardPct: number;
  madvRandom: boolean;
  injectEio: boolean;
};

const RA_KB = [0, 16, 32, 64, 128, 256] as const;

function freshPages(): Pg[] {
  return Array.from({ length: N_PAGES }, () => ({
    present: false,
    dirty: false,
    dirtiedAt: 0,
    used: 0,
    ra: false,
    marker: false,
    pte: 0 as const,
  }));
}

function fresh(): Sim {
  return {
    pages: freshPages(),
    clock: 0,
    tick: 0,
    raStart: 0,
    raSize: 0,
    prev: -2,
    mmapMiss: 0,
    reads: 0,
    writes: 0,
    raPagesIn: 0,
    raUsed: 0,
    raWasted: 0,
    minor: 0,
    major: 0,
    throttles: 0,
    shootdowns: 0,
    evictMapped: 0,
    ioNs: 0,
    head: 'Cold cache.',
    msg:
      'Nothing is cached and nothing is mapped. Click a page in the strip to issue the selected operation against that file offset, or use the buttons to walk sequentially.',
    dead: false,
    rngStep: 0,
  };
}

function clone(s: Sim): Sim {
  return { ...s, pages: s.pages.map((p) => ({ ...p })) };
}

const present = (s: Sim) => s.pages.filter((p) => p.present).length;
const dirtyCount = (s: Sim) => s.pages.filter((p) => p.dirty).length;
const mappedCount = (s: Sim) => s.pages.filter((p) => p.pte > 0).length;

const bgLimit = (c: Cfg) => Math.max(1, Math.round((c.cacheSize * c.bgPct) / 100));
const hardLimit = (c: Cfg) => Math.max(2, Math.round((c.cacheSize * c.hardPct) / 100));

/** Reclaim down to the cache size. Clean pages first; a dirty victim costs a write. */
function reclaim(s: Sim, cfg: Cfg, protect: number) {
  let evicted = 0;
  let dirtyEvicted = 0;
  let mappedEvicted = 0;
  while (present(s) > cfg.cacheSize) {
    let victim = -1;
    let best = Infinity;
    for (let i = 0; i < N_PAGES; i++) {
      const p = s.pages[i];
      if (!p.present || i === protect) continue;
      const score = (p.dirty ? 1e9 : 0) + p.used;
      if (score < best) {
        best = score;
        victim = i;
      }
    }
    if (victim < 0) break;
    const v = s.pages[victim];
    if (v.dirty) {
      s.writes++;
      s.ioNs += DISK_WRITE_NS;
      v.dirty = false;
      dirtyEvicted++;
    }
    if (v.ra) s.raWasted++;
    if (v.pte > 0) {
      s.evictMapped++;
      s.shootdowns++;
      mappedEvicted++;
      s.ioNs += SHOOTDOWN_BASE_NS + CORES * SHOOTDOWN_PER_CORE_NS;
    }
    s.pages[victim] = {
      present: false,
      dirty: false,
      dirtiedAt: 0,
      used: 0,
      ra: false,
      marker: false,
      pte: 0,
    };
    evicted++;
  }
  return { evicted, dirtyEvicted, mappedEvicted };
}

/** Populate [from, from+size) that is not already resident. Returns pages actually read. */
function fill(s: Sim, from: number, size: number, target: number, sync: boolean) {
  let fetched = 0;
  if (from >= N_PAGES) return 0; // past EOF: nothing to read, ra state untouched
  const end = Math.min(N_PAGES, from + size);
  for (let i = Math.max(0, from); i < end; i++) {
    const p = s.pages[i];
    if (p.present) continue;
    p.present = true;
    p.dirty = false;
    p.used = s.tick;
    p.ra = i !== target;
    p.marker = false;
    fetched++;
    if (i !== target) s.raPagesIn++;
  }
  if (fetched > 0) {
    s.reads += fetched;
    if (sync) s.ioNs += DISK_FIRST_NS + (fetched - 1) * DISK_NEXT_NS;
  }
  if (size > 1) {
    const async_size = Math.max(1, Math.floor(size / 2));
    const markerIdx = from + size - async_size;
    if (markerIdx >= 0 && markerIdx < N_PAGES && s.pages[markerIdx].present) {
      s.pages[markerIdx].marker = true;
    }
  }
  s.raStart = from;
  s.raSize = size;
  return fetched;
}

/** ondemand_readahead() for the read()/pread() path. */
function syscallReadahead(s: Sim, cfg: Cfg, p: number) {
  const max = cfg.raPages;
  // Linux also treats an access at offset 0 as the start of a sequence.
  const sequential =
    !cfg.madvRandom &&
    max > 0 &&
    (p === s.prev + 1 ||
      (p === 0 && s.prev < 0) ||
      (s.raSize > 0 && p >= s.raStart && p < s.raStart + s.raSize));
  const size = sequential ? Math.min(max, s.raSize === 0 ? 4 : s.raSize * 2) : 1;
  const fetched = fill(s, p, size, p, true);
  return { size, fetched, sequential };
}

/** do_sync_mmap_readahead(): a cold fault reads the whole ra_pages window. */
function mmapReadahead(s: Sim, cfg: Cfg, p: number) {
  const max = cfg.raPages;
  const sequential = p === s.prev + 1 || (p === 0 && s.prev < 0);
  if (sequential && s.mmapMiss > 0) s.mmapMiss--;
  else if (!sequential) s.mmapMiss++;
  const gaveUp = s.mmapMiss > MMAP_LOTSAMISS;
  const size = cfg.madvRandom || gaveUp || max === 0 ? 1 : max;
  const fetched = fill(s, p, size, p, true);
  return { size, fetched, gaveUp };
}

/** Touching the PG_readahead marker fires the next window without stalling. */
function asyncReadahead(s: Sim, cfg: Cfg, p: number) {
  const max = cfg.raPages;
  if (max === 0) return 0;
  s.pages[p].marker = false;
  const size = Math.min(max, Math.max(4, s.raSize * 2));
  const from = s.raStart + s.raSize;
  const fetched = fill(s, from, size, -1, false);
  return fetched;
}

/** fault_around_bytes: map already-cached neighbours, no I/O. */
function faultAround(s: Sim, p: number) {
  const from = Math.floor(p / FAULT_AROUND_PAGES) * FAULT_AROUND_PAGES;
  let mapped = 0;
  for (let i = from; i < Math.min(N_PAGES, from + FAULT_AROUND_PAGES); i++) {
    const q = s.pages[i];
    if (q.present && q.pte === 0 && i !== p) {
      q.pte = 1;
      mapped++;
    }
  }
  return mapped;
}

/** Dirty accounting after a page is dirtied, by write() or by a store through a PTE. */
function balanceDirty(s: Sim, cfg: Cfg, inFault: boolean) {
  const bg = bgLimit(cfg);
  const hard = hardLimit(cfg);
  const d = dirtyCount(s);
  if (d > hard) {
    s.throttles++;
    let flushed = 0;
    while (dirtyCount(s) > bg) {
      let victim = -1;
      let oldest = Infinity;
      for (let i = 0; i < N_PAGES; i++) {
        const p = s.pages[i];
        if (p.dirty && p.dirtiedAt < oldest) {
          oldest = p.dirtiedAt;
          victim = i;
        }
      }
      if (victim < 0) break;
      s.pages[victim].dirty = false;
      s.writes++;
      s.ioNs += DISK_WRITE_NS;
      flushed++;
    }
    return ` Dirty pages crossed vm.dirty_ratio (${hard} of ${cfg.cacheSize}), so balance_dirty_pages() ${
      inFault ? 'throttled the faulting thread' : 'throttled the caller'
    } and made it wait on ${flushed} writeback${flushed === 1 ? '' : 's'}${
      inFault ? ' — inside a page fault, i.e. inside whatever latch you were holding' : ''
    }.`;
  }
  if (d > bg) {
    let flushed = 0;
    while (dirtyCount(s) > bg) {
      let victim = -1;
      let oldest = Infinity;
      for (let i = 0; i < N_PAGES; i++) {
        const p = s.pages[i];
        if (p.dirty && p.dirtiedAt < oldest) {
          oldest = p.dirtiedAt;
          victim = i;
        }
      }
      if (victim < 0) break;
      s.pages[victim].dirty = false;
      s.writes++;
      flushed++;
    }
    return ` Dirty pages crossed vm.dirty_background_ratio (${bg}), so the flusher thread started writing ${flushed} page${
      flushed === 1 ? '' : 's'
    } back in the background — the caller did not wait.`;
  }
  return '';
}

/** A page brought in by readahead has now actually been asked for: it was not wasted. */
function demand(s: Sim, pg: Pg) {
  if (pg.ra) {
    pg.ra = false;
    s.raUsed++;
  }
}

type Op = 'read' | 'write' | 'load' | 'store';

function access(prev: Sim, cfg: Cfg, op: Op, p: number): Sim {
  const s = clone(prev);
  if (s.dead) {
    s.head = 'The process is dead.';
    s.msg = 'It took SIGBUS on an mmap fault. Reset to continue.';
    return s;
  }
  s.tick++;
  const pg = s.pages[p];
  let msg = '';
  let head = '';

  if (op === 'read' || op === 'write') {
    s.ioNs += SYSCALL_NS;
    if (op === 'read') {
      if (pg.present) {
        demand(s, pg);
        pg.used = s.tick;
        s.ioNs += COPY_NS;
        head = `read() hit on page ${p}.`;
        msg = `The page was already in the page cache, so the kernel walked address_space->i_pages, found the folio and memcpy'd ${PAGE_BYTES} bytes into your buffer. No device I/O — but also no way to skip that copy: the bytes now exist twice, once in the page cache and once in your buffer pool.`;
        if (pg.marker) {
          const n = asyncReadahead(s, cfg, p);
          msg += ` This page carried the PG_readahead marker, so the kernel kicked off the next window asynchronously (${n} page${
            n === 1 ? '' : 's'
          }) while your read returned from cache.`;
        }
      } else if (cfg.injectEio) {
        head = `read() on page ${p} returned EIO.`;
        msg =
          'The device failed the request. read() returns -1 with errno EIO and your engine gets to decide: retry, mark the tablespace offline, fail the query. The same media error under mmap is delivered as SIGBUS to whatever thread touched the address.';
        s.ioNs += DISK_FIRST_NS;
        s.prev = p;
        s.head = head;
        s.msg = msg;
        return s;
      } else {
        const r = syscallReadahead(s, cfg, p);
        s.ioNs += COPY_NS;
        head = `read() miss on page ${p} — ${r.fetched} page${r.fetched === 1 ? '' : 's'} from the device.`;
        msg = r.sequential
          ? `The kernel saw page ${p} continue the previous access, so ondemand_readahead() ramped file_ra_state to ${r.size} pages (${
              (r.size * PAGE_BYTES) / 1024
            } KiB) and issued one request for the whole window. The window doubles on every sequential miss, capped at read_ahead_kb.`
          : `This access does not continue the last one, so the readahead state machine classified it as a standalone random read and fetched exactly one page. Random access does not get readahead — the ramp resets to zero.`;
      }
    } else {
      const wasPresent = pg.present;
      if (!pg.present) {
        pg.present = true;
      } else {
        // The folio was sitting in the cache because readahead fetched it; this write
        // is the demand for it, so it must not count as readahead waste later.
        demand(s, pg);
      }
      pg.used = s.tick;
      if (!pg.dirty) {
        pg.dirty = true;
        pg.dirtiedAt = s.clock;
      }
      s.ioNs += COPY_NS;
      head = `write() to page ${p} returned. Nothing is on disk.`;
      msg = `${
        wasPresent ? 'The page was already cached' : 'A full-page write allocates the folio without reading it first'
      }; generic_perform_write() copied your bytes in, set the dirty bit and returned success. The data is in volatile RAM. It becomes durable only when the flusher, reclaim, or your own fsync() writes it out.`;
      msg += balanceDirty(s, cfg, false);
    }
  } else {
    /* mmap */
    if (pg.pte > 0) {
      pg.used = s.tick;
      // fault_around may have mapped this neighbour before anyone asked for it.
      demand(s, pg);
      if (op === 'load') {
        s.ioNs += LOAD_NS;
        head = `Load from page ${p}: no kernel involvement at all.`;
        msg =
          'The PTE is resident and the TLB hit, so this was a single MOV — about a nanosecond, no syscall, no copy. That is the whole appeal of mmap, and the whole problem: the same instruction on the next page may go to the device instead.';
      } else if (pg.pte === 2) {
        s.ioNs += LOAD_NS;
        head = `Store to page ${p}: no trap.`;
        msg =
          'The PTE is already writable and the page is already accounted dirty, so the store is invisible to the kernel. You cannot order it against anything, you cannot tell the kernel "not yet", and msync() is the only flush you get — for the whole range, not for one page.';
      } else {
        s.minor++;
        s.ioNs += MINOR_FAULT_NS;
        pg.pte = 2;
        pg.dirty = true;
        pg.dirtiedAt = s.clock;
        head = `Store to page ${p} took a write-protect fault.`;
        msg =
          'The page was mapped read-only from an earlier load, so the first store traps into the kernel so it can mark the folio dirty for writeback accounting. Even a fully resident mmap costs one minor fault per page per dirtying.';
        msg += balanceDirty(s, cfg, true);
      }
    } else if (pg.present) {
      s.minor++;
      s.ioNs += MINOR_FAULT_NS;
      pg.pte = op === 'store' ? 2 : 1;
      pg.used = s.tick;
      demand(s, pg);
      if (op === 'store' && !pg.dirty) {
        pg.dirty = true;
        pg.dirtiedAt = s.clock;
      }
      const around = faultAround(s, p);
      head = `Minor fault on page ${p} — no device I/O.`;
      msg = `The folio was already in the page cache, so the fault only installed a PTE. fault_around then mapped ${around} already-cached neighbour${
        around === 1 ? '' : 's'
      } in the same 64 KiB block for free, which is why the next few loads will look instant.`;
      if (op === 'store') msg += balanceDirty(s, cfg, true);
    } else if (cfg.injectEio) {
      s.dead = true;
      s.major++;
      s.ioNs += DISK_FIRST_NS;
      head = `SIGBUS on page ${p}.`;
      msg =
        'The device failed the read behind a load instruction. A load has no error return, so the kernel raises SIGBUS. Without a handler the process dies mid-transaction; with one, you are doing longjmp out of a signal handler while holding latches. This is Crotty et al.’s error-handling argument in one click.';
      s.prev = p;
      s.head = head;
      s.msg = msg;
      return s;
    } else {
      s.major++;
      const r = mmapReadahead(s, cfg, p);
      pg.pte = op === 'store' ? 2 : 1;
      pg.used = s.tick;
      pg.ra = false;
      if (op === 'store') {
        pg.dirty = true;
        pg.dirtiedAt = s.clock;
      }
      const around = faultAround(s, p);
      head = `Major fault on page ${p} — the load blocked on the device.`;
      msg = r.gaveUp
        ? `mmap_miss has passed the give-up threshold, so the kernel stopped reading ahead and faulted in a single page. Linux gives up once mmap_miss (up on a wasted fault, down on a useful one) passes MMAP_LOTSAMISS = 100; until then every random fault pulls the whole ${
            (cfg.raPages * PAGE_BYTES) / 1024
          } KiB window.`
        : `A cold mmap fault does not do the careful ramp that read() does: do_sync_mmap_readahead() reads the entire ra_pages window (${
            r.size
          } page${r.size === 1 ? '' : 's'}) on the first miss. ${
            cfg.madvRandom
              ? 'MADV_RANDOM is set, so it read exactly one page.'
              : 'Set MADV_RANDOM to stop that — it is the mmap equivalent of telling the kernel your index is not a scan.'
          }`;
      msg += ` The faulting thread was stopped inside the load instruction for ${fmtTime(
        DISK_FIRST_NS,
      )}; fault_around mapped ${around} neighbour${around === 1 ? '' : 's'} on the way out.`;
      if (op === 'store') msg += balanceDirty(s, cfg, true);
    }
  }

  const ev = reclaim(s, cfg, p);
  if (ev.evicted > 0) {
    msg += ` Reclaim then evicted ${ev.evicted} page${ev.evicted === 1 ? '' : 's'} to stay inside the cache`;
    msg += ev.dirtyEvicted > 0 ? `, ${ev.dirtyEvicted} of them dirty (a synchronous write on the allocation path)` : '';
    msg += ev.mappedEvicted > 0 ? `, ${ev.mappedEvicted} of them mapped (PTEs cleared plus a TLB shootdown)` : '';
    msg += '. The kernel picked the victims by its own LRU — it has no idea which of these is your B-tree root.';
  }
  s.prev = p;
  s.head = head;
  s.msg = msg;
  return s;
}

function advanceClock(prev: Sim): Sim {
  const s = clone(prev);
  s.clock += FLUSHER_INTERVAL_S;
  const expired = s.pages.filter((p) => p.dirty && s.clock - p.dirtiedAt >= DIRTY_EXPIRE_S).length;
  if (expired > 0) {
    for (const p of s.pages) {
      if (p.dirty && s.clock - p.dirtiedAt >= DIRTY_EXPIRE_S) {
        p.dirty = false;
        s.writes++;
      }
    }
    s.head = `t = ${s.clock}s: the flusher wrote back ${expired} expired page${expired === 1 ? '' : 's'}.`;
    s.msg = `The writeback worker wakes every dirty_writeback_centisecs (5 s) and writes anything dirtied longer ago than dirty_expire_centisecs (30 s). Note what this means for durability: a write() that returned 29 seconds ago is still only in RAM.`;
  } else {
    const d = dirtyCount(s);
    s.head = `t = ${s.clock}s: the flusher woke and found nothing expired.`;
    s.msg =
      d === 0
        ? 'No dirty pages. Every cached page matches the copy on the device.'
        : `${d} page${d === 1 ? ' is' : 's are'} dirty but younger than dirty_expire_centisecs (30 s), so they stay in RAM. Lose power now and those bytes are gone, even though every write() returned success.`;
  }
  return s;
}

function fsyncAll(prev: Sim): Sim {
  const s = clone(prev);
  const d = dirtyCount(s);
  for (const p of s.pages) {
    if (p.dirty) {
      p.dirty = false;
      s.writes++;
    }
  }
  s.ioNs += d * DISK_WRITE_NS + (d > 0 ? FLUSH_FIXED_NS : FLUSH_FIXED_NS / 3);
  s.head = `fsync() wrote ${d} dirty page${d === 1 ? '' : 's'} and flushed the device cache.`;
  s.msg = `fsync() walks the inode's dirty list, waits for every one of those writebacks, then issues a cache-flush command to the device. It is the only thing on this page that makes anything durable — and it costs ${fmtTime(
    d * DISK_WRITE_NS + FLUSH_FIXED_NS,
  )} here because it is synchronous by construction.`;
  return s;
}

function munmapAll(prev: Sim): Sim {
  const s = clone(prev);
  const m = mappedCount(s);
  for (const p of s.pages) p.pte = 0;
  if (m > 0) {
    s.shootdowns++;
    s.ioNs += SHOOTDOWN_BASE_NS + CORES * SHOOTDOWN_PER_CORE_NS + m * 120;
  }
  s.head = `munmap() tore down ${m} PTE${m === 1 ? '' : 's'}.`;
  s.msg = `Clearing a PTE is not local: every core that has run this process may hold the translation in its TLB, so the kernel sends an IPI to all ${CORES} of them and spins until each acknowledges. That is ~${fmtTime(
    SHOOTDOWN_BASE_NS + CORES * SHOOTDOWN_PER_CORE_NS,
  )} of all-core stall for a bookkeeping change — and the kernel does the same thing every time it evicts a mapped page under memory pressure, which is why mmap throughput collapses once the working set exceeds RAM.`;
  return s;
}

export default function PageCacheStepper() {
  const [op, setOp] = useState<Op>('read');
  const [cacheIdx, setCacheIdx] = useState(20);
  const [raIdx, setRaIdx] = useState(4); // 128 KiB
  const [bgPct, setBgPct] = useState(10);
  const [hardPct, setHardPct] = useState(20);
  const [madvRandom, setMadvRandom] = useState(false);
  const [injectEio, setInjectEio] = useState(false);
  const [sim, setSim] = useState<Sim>(fresh);
  const [ref, width] = useSize(760);
  const tip = useTip();

  const cfg: Cfg = {
    cacheSize: cacheIdx,
    raPages: RA_KB[raIdx] / 4,
    bgPct,
    hardPct,
    madvRandom,
    injectEio,
  };

  const go = (p: number) => setSim((s) => access(s, cfg, op, p));
  const nextSeq = () => go(sim.prev < 0 ? 0 : (sim.prev + 1) % N_PAGES);
  const randomOffset = () =>
    setSim((s) => {
      const rng = makeRng(9973 + s.rngStep * 31);
      rng();
      const p = Math.floor(rng() * N_PAGES) % N_PAGES;
      const next = access(s, cfg, op, p);
      next.rngStep = s.rngStep + 1;
      return next;
    });

  const labelW = 104;
  const cw = Math.max(14, Math.min(26, (width - labelW - 12) / N_PAGES));
  const svgW = labelW + N_PAGES * cw + 10;
  const boxW = Math.max(8, cw - 3);
  const LANE_PTE = 34;
  const LANE_CACHE = 76;
  const LANE_DISK = 120;
  const boxH = 22;
  const height = 172;

  const cached = present(sim);
  const dirty = dirtyCount(sim);
  const mapped = mappedCount(sim);
  const x = (i: number) => labelW + i * cw;

  const opLabel: Record<Op, string> = {
    read: 'read() / pread()',
    write: 'write() / pwrite()',
    load: 'mmap load',
    store: 'mmap store',
  };

  return (
    <VizPanel
      title="The buffered I/O path, one access at a time"
      subtitle="Click a page in the strip to issue the selected operation at that file offset. Accesses take microseconds; the writeback timers only move when you advance the clock."
      controls={
        <>
          <Segmented
            label="Operation"
            value={op}
            onChange={setOp}
            options={[
              { value: 'read', label: 'read()', title: 'Copies kernel page cache -> your buffer' },
              { value: 'write', label: 'write()', title: 'Copies your buffer -> page cache, sets the dirty bit' },
              { value: 'load', label: 'mmap load', title: 'A plain load that may fault to the device' },
              { value: 'store', label: 'mmap store', title: 'A plain store that dirties a page behind your back' },
            ]}
          />
          <Button onClick={nextSeq}>Next page (sequential)</Button>
          <Button onClick={randomOffset}>Random offset</Button>
          <Button onClick={() => setSim(advanceClock)} title="dirty_writeback_centisecs = 500">
            +5 s clock
          </Button>
          <Button onClick={() => setSim(fsyncAll)} primary>
            fsync()
          </Button>
          <Button onClick={() => setSim(munmapAll)}>munmap()</Button>
          <Button onClick={() => setSim(fresh())}>Reset</Button>
          <Slider
            label="Page cache size"
            min={6}
            max={32}
            value={cacheIdx}
            onChange={setCacheIdx}
            format={(n) => `${n} pages`}
          />
          <Slider
            label="read_ahead_kb"
            min={0}
            max={RA_KB.length - 1}
            value={raIdx}
            onChange={setRaIdx}
            format={(n) => `${RA_KB[n]} KB`}
          />
          <Slider
            label="dirty_background_ratio"
            min={2}
            max={40}
            value={bgPct}
            onChange={setBgPct}
            format={(n) => `${n}% (${bgLimit({ ...cfg, bgPct: n })} pages)`}
          />
          <Slider
            label="dirty_ratio"
            min={5}
            max={80}
            value={hardPct}
            onChange={setHardPct}
            format={(n) => `${n}% (${hardLimit({ ...cfg, hardPct: n })} pages)`}
          />
          <Check label="MADV_RANDOM / FADV_RANDOM" checked={madvRandom} onChange={setMadvRandom} />
          <Check label="Fail every device read (EIO / SIGBUS)" checked={injectEio} onChange={setInjectEio} />
        </>
      }
      legend={
        <Legend
          items={[
            { label: 'Clean in page cache', color: 'var(--viz-clean)' },
            { label: 'Dirty — RAM only, not durable', color: 'var(--viz-dirty)' },
            { label: 'Readahead, not yet demanded ("R")', color: 'var(--viz-3)' },
            { label: 'Mapped PTE (mmap)', color: 'var(--viz-7)' },
            { label: 'Disk copy stale — the RAM version is newer', color: 'var(--viz-stale)' },
          ]}
        />
      }
      stats={
        <Stats
          items={[
            { label: 'Cached', value: `${cached} / ${cfg.cacheSize}`, hint: 'Resident pages vs the cache size you set' },
            { label: 'Dirty', value: `${dirty}`, hint: `background at ${bgLimit(cfg)}, throttle at ${hardLimit(cfg)}` },
            { label: 'Mapped PTEs', value: `${mapped}` },
            { label: 'Device reads', value: fmtNum(sim.reads) },
            { label: 'Device writes', value: fmtNum(sim.writes) },
            {
              label: 'Readahead wasted',
              value: `${sim.raWasted}`,
              hint: 'Pages read ahead and evicted without ever being asked for',
            },
            { label: 'Faults (minor/major)', value: `${sim.minor} / ${sim.major}` },
            { label: 'Time stalled', value: fmtTime(sim.ioNs), hint: 'Cumulative time the thread was not running' },
          ]}
        />
      }
      note={
        <Note>
          <strong>{sim.head}</strong> {sim.msg}
        </Note>
      }
      table={
        <>
          <table className="viz-table">
            <thead>
              <tr>
                <th>Page (4 KiB)</th>
                <th>Offset</th>
                <th>Page cache</th>
                <th>Brought in by</th>
                <th>Dirty for</th>
                <th>PTE</th>
                <th>Disk copy</th>
              </tr>
            </thead>
            <tbody>
              {sim.pages.some((p) => p.present || p.pte > 0) ? (
                sim.pages.map((p, i) =>
                  p.present || p.pte > 0 ? (
                    <tr key={i}>
                      <td>{i}</td>
                      <td>{fmtNum(i * PAGE_BYTES)}</td>
                      <td>{p.present ? (p.dirty ? 'dirty' : 'clean') : 'not resident'}</td>
                      <td>{p.ra ? 'readahead (unused)' : p.present ? 'demand' : '—'}</td>
                      <td>{p.dirty ? `${sim.clock - p.dirtiedAt}s` : '—'}</td>
                      <td>{p.pte === 2 ? 'mapped rw' : p.pte === 1 ? 'mapped ro' : '—'}</td>
                      <td>{p.dirty ? 'stale' : 'current'}</td>
                    </tr>
                  ) : null,
                )
              ) : (
                <tr>
                  <td colSpan={7}>Cache is empty.</td>
                </tr>
              )}
            </tbody>
          </table>
          <table className="viz-table">
            <thead>
              <tr>
                <th>Counter</th>
                <th>Value</th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <td>Virtual clock</td>
                <td>{sim.clock}s</td>
              </tr>
              <tr>
                <td>4 KiB device reads / writes</td>
                <td>
                  {sim.reads} / {sim.writes}
                </td>
              </tr>
              <tr>
                <td>Readahead pages in / used / wasted</td>
                <td>
                  {sim.raPagesIn} / {sim.raUsed} / {sim.raWasted}
                </td>
              </tr>
              <tr>
                <td>Minor / major faults</td>
                <td>
                  {sim.minor} / {sim.major}
                </td>
              </tr>
              <tr>
                <td>balance_dirty_pages() throttles</td>
                <td>{sim.throttles}</td>
              </tr>
              <tr>
                <td>TLB shootdowns (munmap + mapped-page eviction)</td>
                <td>
                  {sim.shootdowns} ({sim.evictMapped} from eviction)
                </td>
              </tr>
              <tr>
                <td>Cumulative stall</td>
                <td>{fmtTime(sim.ioNs)}</td>
              </tr>
              <tr>
                <td>file_ra_state (start, size)</td>
                <td>
                  {sim.raStart}, {sim.raSize}
                </td>
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
            aria-label="Page cache state for a 32-page file: address space, page cache and disk"
          >
            <text x={0} y={LANE_PTE + 15} fill="var(--viz-ink-2)">
              Address space
            </text>
            <text x={0} y={LANE_CACHE + 15} fill="var(--viz-ink-2)">
              Page cache
            </text>
            <text x={0} y={LANE_DISK + 15} fill="var(--viz-ink-2)">
              Disk (file)
            </text>

            {sim.raSize > 0 ? (
              <g>
                <line
                  className="viz-axis-line"
                  x1={x(sim.raStart) + 1}
                  x2={x(Math.min(N_PAGES, sim.raStart + sim.raSize)) - 3}
                  y1={LANE_CACHE - 8}
                  y2={LANE_CACHE - 8}
                />
                <text x={x(sim.raStart) + 1} y={LANE_CACHE - 12} fontSize={10} fill="var(--viz-ink-2)">
                  file_ra_state: {sim.raSize} pages
                </text>
              </g>
            ) : null}

            {sim.pages.map((p, i) => {
              const isLast = i === sim.prev;
              const cacheFill = !p.present
                ? 'var(--viz-plane)'
                : p.dirty
                  ? 'var(--viz-dirty)'
                  : 'var(--viz-clean)';
              return (
                <g
                  key={i}
                  style={{ cursor: 'pointer' }}
                  onClick={() => go(i)}
                  onKeyDown={(e) => {
                    if (e.key === 'Enter' || e.key === ' ') {
                      e.preventDefault();
                      go(i);
                    }
                  }}
                  role="button"
                  aria-label={`Issue ${opLabel[op]} at page ${i}`}
                  {...tip(
                    <>
                      <strong>Page {i}</strong> · offset {fmtNum(i * PAGE_BYTES)}
                      <br />
                      Page cache: {p.present ? (p.dirty ? 'dirty' : 'clean') : 'not resident'}
                      {p.ra ? ' · read ahead, never demanded' : ''}
                      {p.marker ? ' · PG_readahead marker' : ''}
                      <br />
                      PTE: {p.pte === 2 ? 'mapped read-write' : p.pte === 1 ? 'mapped read-only' : 'not mapped'}
                      <br />
                      <span style={{ color: 'var(--viz-ink-2)' }}>Click to {opLabel[op]} here.</span>
                    </>,
                  )}
                >
                  <rect x={x(i)} y={10} width={cw} height={height - 30} fill="transparent" />

                  {/* address space / PTE lane */}
                  <rect
                    x={x(i)}
                    y={LANE_PTE}
                    width={boxW}
                    height={boxH}
                    rx={3}
                    fill={p.pte > 0 ? 'var(--viz-7)' : 'var(--viz-plane)'}
                    stroke={p.pte > 0 ? 'var(--viz-7)' : 'var(--viz-grid)'}
                  />
                  {p.pte === 2 ? (
                    <text
                      x={x(i) + boxW / 2}
                      y={LANE_PTE + 15}
                      textAnchor="middle"
                      fontSize={10}
                      fill="var(--viz-surface)"
                    >
                      W
                    </text>
                  ) : null}

                  {/* page cache lane */}
                  <rect
                    x={x(i)}
                    y={LANE_CACHE}
                    width={boxW}
                    height={boxH}
                    rx={3}
                    fill={cacheFill}
                    stroke={p.ra ? 'var(--viz-3)' : p.present ? cacheFill : 'var(--viz-grid)'}
                    strokeWidth={p.ra ? 2 : 1}
                  />
                  {p.ra ? (
                    <text
                      x={x(i) + boxW / 2}
                      y={LANE_CACHE + 15}
                      textAnchor="middle"
                      fontSize={10}
                      fill="var(--viz-surface)"
                    >
                      R
                    </text>
                  ) : null}
                  {p.marker ? (
                    <path
                      d={`M ${x(i) + boxW - 6} ${LANE_CACHE + 1} l 5 0 l 0 5 z`}
                      fill="var(--viz-ink)"
                      opacity={0.8}
                    />
                  ) : null}

                  {/* disk lane */}
                  <rect
                    x={x(i)}
                    y={LANE_DISK}
                    width={boxW}
                    height={boxH}
                    rx={3}
                    fill={p.dirty ? 'var(--viz-stale)' : 'var(--viz-neutral)'}
                    stroke="var(--viz-grid)"
                  />

                  {isLast ? (
                    <rect
                      x={x(i) - 2}
                      y={LANE_PTE - 4}
                      width={boxW + 4}
                      height={LANE_DISK + boxH - LANE_PTE + 8}
                      rx={5}
                      fill="none"
                      stroke="var(--viz-ink)"
                      strokeWidth={1.5}
                    />
                  ) : null}

                  {i % 4 === 0 ? (
                    <text x={x(i) + boxW / 2} y={height - 4} textAnchor="middle" fontSize={10}>
                      {i}
                    </text>
                  ) : null}
                </g>
              );
            })}
          </svg>
        </TooltipHost>
      </div>
    </VizPanel>
  );
}
