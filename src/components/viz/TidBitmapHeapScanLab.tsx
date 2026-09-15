import { useDeferredValue, useMemo, useState } from 'react';
import { VizPanel, Segmented, Slider, Button, Legend, Stats, Note, useTicker, makeRng, fmtNum, fmtBytes } from './Viz';

/**
 * PostgreSQL's run-time TID bitmap (src/backend/nodes/tidbitmap.c) feeding a Bitmap Heap Scan.
 *
 * What is ported from the source, step for step:
 * - One hash table holds exact page entries (one bit per tuple offset) and lossy chunk entries
 *   (one bit per page, keyed by the chunk's first page, PAGES_PER_CHUNK = BLCKSZ / 32 = 256).
 * - The entry limit is work_mem / (sizeof(PagetableEntry) + 2 pointers) = work_mem / 64 bytes on a
 *   64-bit build (tbm_calculate_entries), with a floor of 16.
 * - tbm_add_tuples: a TID on a page that is already lossy is dropped; otherwise its bit is set, and
 *   as soon as nentries > maxentries, tbm_lossify runs.
 * - tbm_lossify: walk the hash table from where the last pass stopped, turning exact pages into lossy
 *   bits (skipping pages that would be their own chunk header) until nentries <= maxentries / 2; if
 *   that is impossible, raise maxentries to twice nentries.
 * - BitmapAnd: every child builds its own bitmap under its own work_mem, then tbm_intersect folds the
 *   others into the first. Lossy AND exact keeps the lossy page; exact AND lossy keeps the exact bits
 *   and sets recheck. BitmapOr: Bitmap Index Scan children add straight into one shared bitmap.
 * - The heap is read in ascending block order. Exact pages visit only the listed offsets and evaluate
 *   the recheck qual only if the page is flagged; lossy pages visit every tuple and recheck each one.
 *
 * Model assumptions (not PostgreSQL numbers):
 * - A 2,048-page table with exactly 40 visible tuples per page; created_at follows physical order
 *   within +-2 pages, amount is independent of position. Real work_mem limits scale with table size.
 * - The hash table's iteration order is approximated by sorting block numbers on murmurhash32 (the
 *   hash tidbitmap.c uses) masked to 16 bits; simplehash's real bucket order also depends on
 *   collisions and table growth.
 * - The child with fewer TIDs is intersected into (the planner orders AND children by estimated cost).
 */

export const PAGES = 2048;
export const ROWS_PER_PAGE = 40;
export const NROWS = PAGES * ROWS_PER_PAGE;
export const PAGES_PER_CHUNK = 256;
export const ENTRY_BYTES = 64;
export const COLS = 64;

export const maxEntriesFor = (workMemKb: number) => Math.max(16, Math.floor((workMemKb * 1024) / ENTRY_BYTES));

function murmurhash32(x: number) {
  let h = x >>> 0;
  h ^= h >>> 16;
  h = Math.imul(h, 0x85ebca6b) >>> 0;
  h ^= h >>> 13;
  h = Math.imul(h, 0xc2b2ae35) >>> 0;
  h ^= h >>> 16;
  return h >>> 0;
}

let hashOrder: Int32Array | null = null;
/** Block numbers in (approximate) hash-table iteration order. */
export function getHashOrder() {
  if (!hashOrder) {
    const blocks = Array.from({ length: PAGES }, (_, b) => b);
    blocks.sort((x, y) => (murmurhash32(y) & 0xffff) - (murmurhash32(x) & 0xffff) || y - x);
    hashOrder = Int32Array.from(blocks);
  }
  return hashOrder;
}

type Table = { orderA: Int32Array; orderB: Int32Array; rankA: Int32Array; rankB: Int32Array };
let table: Table | null = null;
/** created_at tracks physical position (+-2 pages); amount is independent of it. */
export function getTable(): Table {
  if (table) return table;
  const rng = makeRng(20260914);
  const created = new Float64Array(NROWS);
  const amount = new Float64Array(NROWS);
  for (let r = 0; r < NROWS; r++) {
    const page = Math.floor(r / ROWS_PER_PAGE);
    created[r] = page + (rng() - 0.5) * 4;
    amount[r] = rng();
  }
  const ids = () => Array.from({ length: NROWS }, (_, i) => i);
  const a = ids().sort((x, y) => created[x] - created[y] || x - y);
  const b = ids().sort((x, y) => amount[x] - amount[y] || x - y);
  const rankA = new Int32Array(NROWS);
  const rankB = new Int32Array(NROWS);
  a.forEach((row, i) => (rankA[row] = i));
  b.forEach((row, i) => (rankB[row] = i));
  table = { orderA: Int32Array.from(a), orderB: Int32Array.from(b), rankA, rankB };
  return table;
}

type Entry = { blockno: number; ischunk: boolean; recheck: boolean; bits: Uint8Array };

export class TidBitmap {
  map = new Map<number, Entry>();
  nentries = 0;
  npages = 0;
  nchunks = 0;
  maxentries: number;
  lossifyStart = 0;
  lossifyPasses = 0;
  peakEntries = 0;
  tidsAdded = 0;
  constructor(maxentries: number) {
    this.maxentries = maxentries;
  }

  pageIsLossy(pageno: number) {
    if (this.nchunks === 0) return false;
    const bitno = pageno % PAGES_PER_CHUNK;
    const chunk = this.map.get(pageno - bitno);
    return !!chunk && chunk.ischunk && chunk.bits[bitno] === 1;
  }

  findPageEntry(pageno: number) {
    const e = this.map.get(pageno);
    return e && !e.ischunk ? e : undefined;
  }

  private getPageEntry(pageno: number) {
    let e = this.map.get(pageno);
    if (!e) {
      e = { blockno: pageno, ischunk: false, recheck: false, bits: new Uint8Array(ROWS_PER_PAGE) };
      this.map.set(pageno, e);
      this.nentries++;
      this.npages++;
    }
    return e;
  }

  private markPageLossy(pageno: number) {
    const bitno = pageno % PAGES_PER_CHUNK;
    const chunkno = pageno - bitno;
    if (bitno !== 0 && this.map.delete(pageno)) {
      this.nentries--;
      this.npages--;
    }
    let chunk = this.map.get(chunkno);
    if (!chunk) {
      chunk = { blockno: chunkno, ischunk: true, recheck: false, bits: new Uint8Array(PAGES_PER_CHUNK) };
      this.map.set(chunkno, chunk);
      this.nentries++;
      this.nchunks++;
    } else if (!chunk.ischunk) {
      // the chunk-header page was exact: it becomes the chunk, and is itself lossy from now on
      chunk.ischunk = true;
      chunk.recheck = false;
      chunk.bits = new Uint8Array(PAGES_PER_CHUNK);
      chunk.bits[0] = 1;
      this.nchunks++;
      this.npages--;
    }
    chunk.bits[bitno] = 1;
  }

  private lossify() {
    this.lossifyPasses++;
    const order = getHashOrder();
    const n = order.length;
    for (let step = 0; step < n; step++) {
      const pos = (this.lossifyStart + step) % n;
      const blk = order[pos];
      const e = this.map.get(blk);
      if (!e || e.ischunk) continue;
      if (blk % PAGES_PER_CHUNK === 0) continue;
      this.markPageLossy(blk);
      if (this.nentries <= Math.floor(this.maxentries / 2)) {
        this.lossifyStart = (pos + 1) % n;
        return;
      }
    }
    if (this.nentries > Math.floor(this.maxentries / 2)) this.maxentries = this.nentries * 2;
  }

  /** tbm_add_tuples for TIDs given as row numbers (page * ROWS_PER_PAGE + offset - 1). */
  addRows(rows: ArrayLike<number>, recheck = false) {
    let curblk = -1;
    let page: Entry | null = null;
    for (let i = 0; i < rows.length; i++) {
      const row = rows[i];
      const blk = Math.floor(row / ROWS_PER_PAGE);
      const off = (row % ROWS_PER_PAGE) + 1;
      this.tidsAdded++;
      if (blk !== curblk) {
        page = this.pageIsLossy(blk) ? null : this.getPageEntry(blk);
        curblk = blk;
      }
      if (!page) continue;
      if (page.ischunk) page.bits[0] = 1;
      else page.bits[off - 1] = 1;
      page.recheck = page.recheck || recheck;
      if (this.nentries > this.peakEntries) this.peakEntries = this.nentries;
      if (this.nentries > this.maxentries) {
        this.lossify();
        curblk = -1;
      }
    }
  }

  /** tbm_intersect: this = this AND b. */
  intersect(b: TidBitmap) {
    for (const a of Array.from(this.map.values())) {
      if (a.ischunk) {
        let candelete = true;
        for (let bit = 0; bit < PAGES_PER_CHUNK; bit++) {
          if (!a.bits[bit]) continue;
          const pg = a.blockno + bit;
          if (!b.pageIsLossy(pg) && !b.findPageEntry(pg)) a.bits[bit] = 0;
          else candelete = false;
        }
        if (candelete) {
          this.map.delete(a.blockno);
          this.nchunks--;
          this.nentries--;
        }
      } else if (b.pageIsLossy(a.blockno)) {
        a.recheck = true;
      } else {
        let candelete = true;
        const bp = b.findPageEntry(a.blockno);
        if (bp) {
          for (let i = 0; i < ROWS_PER_PAGE; i++) {
            a.bits[i] &= bp.bits[i];
            if (a.bits[i]) candelete = false;
          }
          a.recheck = a.recheck || bp.recheck;
        }
        if (candelete) {
          this.map.delete(a.blockno);
          this.npages--;
          this.nentries--;
        }
      }
    }
  }

  /** Iteration: exact pages and lossy chunk bits merged into ascending block order. */
  iterate() {
    const out: { blockno: number; lossy: boolean; recheck: boolean; offsets: number[] }[] = [];
    const pages: Entry[] = [];
    const lossy: number[] = [];
    for (const e of this.map.values()) {
      if (e.ischunk) {
        for (let bit = 0; bit < PAGES_PER_CHUNK; bit++) if (e.bits[bit]) lossy.push(e.blockno + bit);
      } else pages.push(e);
    }
    pages.sort((x, y) => x.blockno - y.blockno);
    lossy.sort((x, y) => x - y);
    let i = 0;
    let j = 0;
    while (i < pages.length || j < lossy.length) {
      if (j < lossy.length && (i >= pages.length || lossy[j] < pages[i].blockno)) {
        out.push({ blockno: lossy[j++], lossy: true, recheck: true, offsets: [] });
      } else {
        const p = pages[i++];
        const offsets: number[] = [];
        for (let k = 0; k < ROWS_PER_PAGE; k++) if (p.bits[k]) offsets.push(k + 1);
        out.push({ blockno: p.blockno, lossy: false, recheck: p.recheck, offsets });
      }
    }
    return out;
  }

  /** Per page: 0 = not in the bitmap, 1 = exact, 2 = exact with recheck, 3 = lossy. */
  states() {
    const s = new Uint8Array(PAGES);
    for (const e of this.map.values()) {
      if (e.ischunk) {
        for (let bit = 0; bit < PAGES_PER_CHUNK; bit++) if (e.bits[bit] && e.blockno + bit < PAGES) s[e.blockno + bit] = 3;
      } else if (e.blockno < PAGES) s[e.blockno] = e.recheck ? 2 : 1;
    }
    return s;
  }
}

export type Combine = 'and' | 'or' | 'one';

export type ChildSummary = {
  name: string;
  tids: number;
  states: Uint8Array;
  pagesWithMatches: number;
  entriesNeeded: number;
  peakEntries: number;
  lossifyPasses: number;
  exact: number;
  lossy: number;
  ownBitmap: boolean;
};

export type WalkStep = { blockno: number; lossy: boolean; recheck: boolean; examined: number; removed: number; returned: number };

export function runBitmapScan(selAPct: number, selBPct: number, workMemKb: number, combine: Combine) {
  const T = getTable();
  const kA = Math.round((selAPct / 100) * NROWS);
  const kB = Math.round((selBPct / 100) * NROWS);
  const rowsA = T.orderA.subarray(NROWS - kA); // index order: created_at ascending
  const rowsB = T.orderB.subarray(NROWS - kB); // index order: amount ascending
  const inA = (row: number) => T.rankA[row] >= NROWS - kA;
  const inB = (row: number) => T.rankB[row] >= NROWS - kB;
  const matches = (row: number) => (combine === 'and' ? inA(row) && inB(row) : combine === 'or' ? inA(row) || inB(row) : inA(row));
  const maxentries = maxEntriesFor(workMemKb);

  const pagesOf = (rows: ArrayLike<number>) => {
    const s = new Uint8Array(PAGES);
    for (let i = 0; i < rows.length; i++) s[Math.floor(rows[i] / ROWS_PER_PAGE)] = 1;
    return s;
  };
  const count = (s: Uint8Array, v: number) => s.reduce((acc, x) => acc + (x === v ? 1 : 0), 0);
  const summarize = (name: string, rows: ArrayLike<number>, tbm: TidBitmap | null): ChildSummary => {
    const exactSet = pagesOf(rows);
    const needed = count(exactSet, 1);
    if (!tbm) return { name, tids: rows.length, states: exactSet, pagesWithMatches: needed, entriesNeeded: needed, peakEntries: 0, lossifyPasses: 0, exact: needed, lossy: 0, ownBitmap: false };
    const states = tbm.states();
    return { name, tids: rows.length, states, pagesWithMatches: needed, entriesNeeded: needed, peakEntries: tbm.peakEntries, lossifyPasses: tbm.lossifyPasses, exact: count(states, 1) + count(states, 2), lossy: count(states, 3), ownBitmap: true };
  };

  let result: TidBitmap;
  let childA: ChildSummary;
  let childB: ChildSummary | null = null;
  let firstIsA = true;
  if (combine === 'one') {
    result = new TidBitmap(maxentries);
    result.addRows(rowsA);
    childA = summarize('created_at', rowsA, result);
  } else if (combine === 'and') {
    const a = new TidBitmap(maxentries);
    a.addRows(rowsA);
    const b = new TidBitmap(maxentries);
    b.addRows(rowsB);
    childA = summarize('created_at', rowsA, a);
    childB = summarize('amount', rowsB, b);
    firstIsA = kA <= kB;
    result = firstIsA ? a : b;
    result.intersect(firstIsA ? b : a);
  } else {
    result = new TidBitmap(maxentries);
    result.addRows(rowsA);
    result.addRows(rowsB);
    childA = summarize('created_at', rowsA, null);
    childB = summarize('amount', rowsB, null);
  }

  const walk: WalkStep[] = [];
  for (const p of result.iterate()) {
    let examined = 0;
    let removed = 0;
    let returned = 0;
    if (p.lossy) {
      for (let off = 1; off <= ROWS_PER_PAGE; off++) {
        examined++;
        if (matches(p.blockno * ROWS_PER_PAGE + off - 1)) returned++;
        else removed++;
      }
    } else {
      for (const off of p.offsets) {
        examined++;
        if (!p.recheck || matches(p.blockno * ROWS_PER_PAGE + off - 1)) returned++;
        else removed++;
      }
    }
    walk.push({ blockno: p.blockno, lossy: p.lossy, recheck: p.recheck, examined, removed, returned });
  }

  let trueRows = 0;
  const truePages = new Uint8Array(PAGES);
  for (let r = 0; r < NROWS; r++) {
    if (matches(r)) {
      trueRows++;
      truePages[Math.floor(r / ROWS_PER_PAGE)] = 1;
    }
  }
  const states = result.states();
  const sum = (f: (w: WalkStep) => number) => walk.reduce((s, w) => s + f(w), 0);
  return {
    combine,
    maxentries,
    childA,
    childB,
    firstIsA,
    states,
    walk,
    resultEntries: result.nentries,
    resultLossifyPasses: result.lossifyPasses,
    resultPeak: result.peakEntries,
    peakEntries: Math.max(childA.peakEntries, childB ? childB.peakEntries : 0, result.peakEntries),
    exactPages: walk.filter((w) => !w.lossy).length,
    lossyPages: walk.filter((w) => w.lossy).length,
    recheckFlaggedExact: walk.filter((w) => !w.lossy && w.recheck).length,
    examined: sum((w) => w.examined),
    removed: sum((w) => w.removed),
    returned: sum((w) => w.returned),
    trueRows,
    pagesWithMatch: count(truePages, 1),
  };
}

export type ScanResult = ReturnType<typeof runBitmapScan>;

/* --------------------------------------------------------------------------- drawing */

const STATE_FILL = ['var(--viz-neutral)', 'var(--viz-1)', 'var(--viz-warning)', 'var(--viz-serious)'];

function gridPaths(states: Uint8Array, x0: number, y0: number, cell: number) {
  const s = Math.max(1, cell - (cell >= 8 ? 2 : 1));
  const parts = ['', '', '', ''];
  let hatch = '';
  for (let p = 0; p < PAGES; p++) {
    const x = x0 + (p % COLS) * cell;
    const y = y0 + Math.floor(p / COLS) * cell;
    parts[states[p]] += `M${x} ${y}h${s}v${s}h${-s}z`;
    if (states[p] === 3 && cell >= 8) hatch += `M${x} ${y + s}L${x + s} ${y}`;
  }
  return { parts, hatch, s };
}

function Grid({ states, x0, y0, cell }: { states: Uint8Array; x0: number; y0: number; cell: number }) {
  const g = useMemo(() => gridPaths(states, x0, y0, cell), [states, x0, y0, cell]);
  return (
    <g>
      {g.parts.map((d, i) => (d ? <path key={i} d={d} fill={STATE_FILL[i]} stroke={i === 0 ? 'var(--viz-border)' : 'none'} strokeWidth={i === 0 ? 0.5 : 0} /> : null))}
      {g.hatch ? <path d={g.hatch} stroke="var(--viz-surface)" strokeWidth={1.2} fill="none" /> : null}
    </g>
  );
}

const WORK_MEM_DEFAULT = 256;

export default function TidBitmapHeapScanLab() {
  const [combine, setCombine] = useState<Combine>('and');
  const [selA, setSelA] = useState(40);
  const [selB, setSelB] = useState(5);
  const [workMem, setWorkMem] = useState(WORK_MEM_DEFAULT);
  const [cursor, setCursor] = useState<number | null>(null);
  const [hover, setHover] = useState<number | null>(null);

  const inputsNow = useMemo(() => ({ combine, selA, selB, workMem }), [combine, selA, selB, workMem]);
  const inputs = useDeferredValue(inputsNow);
  const r = useMemo(() => runBitmapScan(inputs.selA, inputs.selB, inputs.workMem, inputs.combine), [inputs]);

  const running = cursor !== null && cursor < r.walk.length;
  // rAF timestamps can precede the ticker's start time, so a frame's dt may be negative: clamp it.
  useTicker((dt) => setCursor((c) => (c === null ? null : c + (Math.max(0, dt) * r.walk.length) / 4500)), running);
  const done = cursor === null || cursor >= r.walk.length;
  const stepsDone = done ? r.walk.length : Math.max(0, Math.min(r.walk.length, Math.floor(cursor ?? 0)));
  const partial = useMemo(() => {
    let exact = 0;
    let lossy = 0;
    let examined = 0;
    let removed = 0;
    let returned = 0;
    for (let i = 0; i < stepsDone; i++) {
      const w = r.walk[i];
      if (w.lossy) lossy++;
      else exact++;
      examined += w.examined;
      removed += w.removed;
      returned += w.returned;
    }
    return { exact, lossy, examined, removed, returned };
  }, [r, stepsDone]);
  const cursorPage = !done && stepsDone < r.walk.length ? r.walk[stepsDone].blockno : null;

  const reset = () => setCursor(null);
  const W = 704;
  const LEFT = 40;
  const SMALL = 5;
  const BIG = 10;
  const smallTop = 22;
  const smallH = (PAGES / COLS) * SMALL;
  const bigTop = smallTop + smallH + 74;
  const bigH = (PAGES / COLS) * BIG;
  const H = bigTop + bigH + 8;

  const opName = combine === 'and' ? 'BitmapAnd' : combine === 'or' ? 'BitmapOr' : '';
  const condA = 'created_at >= $1';
  const condB = 'amount > $2';
  const recheckCond = combine === 'one' ? `(${condA})` : `((${condA}) ${combine === 'and' ? 'AND' : 'OR'} (${condB}))`;

  const childLines = (c: ChildSummary) => [
    `${fmtNum(c.tids)} TIDs on ${fmtNum(c.entriesNeeded)} pages · peak ${fmtNum(c.peakEntries)} entries`,
    !c.ownBitmap
      ? 'added straight into the BitmapOr’s one bitmap'
      : c.lossifyPasses === 0
        ? 'stayed exact'
        : `${c.lossifyPasses} lossify pass${c.lossifyPasses === 1 ? '' : 'es'}: ${fmtNum(c.lossy)} pages now lossy`,
  ];

  const hoverInfo = (() => {
    if (hover === null) return null;
    const st = r.states[hover];
    const step = r.walk.find((w) => w.blockno === hover);
    const T = getTable();
    let a = 0;
    let b = 0;
    const kA = Math.round((inputs.selA / 100) * NROWS);
    const kB = Math.round((inputs.selB / 100) * NROWS);
    for (let off = 0; off < ROWS_PER_PAGE; off++) {
      const row = hover * ROWS_PER_PAGE + off;
      if (T.rankA[row] >= NROWS - kA) a++;
      if (T.rankB[row] >= NROWS - kB) b++;
    }
    const kind = st === 0 ? 'not in the bitmap — never read' : st === 1 ? 'exact entry' : st === 2 ? 'exact entry flagged recheck' : `lossy bit in the chunk starting at page ${hover - (hover % PAGES_PER_CHUNK)}`;
    return `Page ${fmtNum(hover)}: ${kind}. Rows matching created_at: ${a}, amount: ${b}.${step ? ` Heap scan examines ${step.examined}, rechecks ${step.recheck ? step.examined : 0}, returns ${step.returned}.` : ''}`;
  })();

  const lossyNow = r.lossyPages > 0;
  const lossyChildren = [r.childA, r.childB].flatMap((c) => (c && c.ownBitmap && c.lossifyPasses > 0 ? [c.name] : []));
  const explain = [
    `Bitmap Heap Scan on orders`,
    `  Recheck Cond: ${recheckCond}`,
    ...(partial.removed > 0 ? [`  Rows Removed by Index Recheck: ${partial.removed}`] : []),
    ...(partial.exact + partial.lossy > 0 ? [`  Heap Blocks:${partial.exact > 0 ? ` exact=${partial.exact}` : ''}${partial.lossy > 0 ? ` lossy=${partial.lossy}` : ''}`] : []),
    ...(combine === 'one'
      ? [`  ->  Bitmap Index Scan on orders_created_at_idx`, `        Index Cond: (${condA})`]
      : [
          `  ->  ${opName}`,
          ...(combine === 'and' && !r.firstIsA
            ? [`        ->  Bitmap Index Scan on orders_amount_idx`, `              Index Cond: (${condB})`, `        ->  Bitmap Index Scan on orders_created_at_idx`, `              Index Cond: (${condA})`]
            : [`        ->  Bitmap Index Scan on orders_created_at_idx`, `              Index Cond: (${condA})`, `        ->  Bitmap Index Scan on orders_amount_idx`, `              Index Cond: (${condB})`]),
        ]),
  ].join('\n');

  const chunkRows = Array.from({ length: PAGES / PAGES_PER_CHUNK }, (_, c) => {
    const lo = c * PAGES_PER_CHUNK;
    const steps = r.walk.filter((w) => w.blockno >= lo && w.blockno < lo + PAGES_PER_CHUNK);
    return {
      c,
      lo,
      exact: steps.filter((w) => !w.lossy && !w.recheck).length,
      flagged: steps.filter((w) => !w.lossy && w.recheck).length,
      lossy: steps.filter((w) => w.lossy).length,
      examined: steps.reduce((s, w) => s + w.examined, 0),
      removed: steps.reduce((s, w) => s + w.removed, 0),
      returned: steps.reduce((s, w) => s + w.returned, 0),
    };
  });

  return (
    <VizPanel
      title="A TID bitmap, from index scans to heap pages"
      subtitle="Each Bitmap Index Scan sets one bit per matching tuple in a hash table of heap pages. BitmapAnd or BitmapOr combines the bitmaps, and the Bitmap Heap Scan reads the surviving pages in block order. Shrink work_mem and watch pages go lossy: the page is still read, but every tuple on it must be rechecked."
      controls={
        <>
          <Segmented
            label="Plan"
            value={combine}
            onChange={(v) => {
              setCombine(v);
              reset();
            }}
            options={[
              { value: 'and', label: 'BitmapAnd' },
              { value: 'or', label: 'BitmapOr' },
              { value: 'one', label: 'One index' },
            ]}
          />
          <Slider label="created_at >= $1: newest rows" min={1} max={100} value={selA} onChange={(v) => { setSelA(v); reset(); }} format={(v) => `${v}%`} />
          <Slider label="amount > $2: matching rows" min={1} max={50} value={selB} disabled={combine === 'one'} onChange={(v) => { setSelB(v); reset(); }} format={(v) => `${v}%`} />
          <Slider label="work_mem" min={64} max={512} step={8} value={workMem} onChange={(v) => { setWorkMem(v); reset(); }} format={(v) => `${v} kB`} />
        </>
      }
      legend={
        <Legend
          items={[
            { label: 'Exact page entry (tuple bits)', color: 'var(--viz-1)' },
            { label: 'Exact entry flagged recheck', color: 'var(--viz-warning)' },
            { label: 'Lossy page (hatched): one bit for the page', color: 'var(--viz-serious)' },
            { label: 'Not in the bitmap', color: 'var(--viz-neutral)' },
          ]}
        />
      }
      stats={
        <Stats
          items={[
            { label: 'Entries work_mem allows', value: fmtNum(r.maxentries), hint: 'work_mem ÷ 64 bytes per hash-table entry on a 64-bit build (tbm_calculate_entries).' },
            { label: 'Heap pages read', value: `${fmtNum(partial.exact + partial.lossy)}`, hint: 'EXPLAIN ANALYZE reports these as Heap Blocks: exact=… lossy=…' },
            { label: 'Of which lossy', value: fmtNum(partial.lossy) },
            { label: 'Tuples examined', value: fmtNum(partial.examined) },
            { label: 'Rows removed by recheck', value: fmtNum(partial.removed), hint: 'Rows Removed by Index Recheck: tuples the bitmap pointed at that fail the original condition.' },
            { label: 'Rows returned', value: fmtNum(partial.returned) },
          ]}
        />
      }
      note={
        <Note>
          {!done ? (
            <>
              <strong>Reading page {fmtNum(cursorPage ?? 0)}.</strong> The scan only moves forward through the file: {fmtNum(stepsDone)} of {fmtNum(r.walk.length)} pages read so far.
            </>
          ) : r.walk.length === 0 ? (
            <strong>No page survives: the bitmap is empty and the heap is never touched.</strong>
          ) : lossyNow ? (
            <>
              <strong>
                {fmtNum(r.lossyPages)} of {fmtNum(r.walk.length)} pages are lossy.
              </strong>{' '}
              work_mem allows {fmtNum(r.maxentries)} entries, so tbm_lossify traded tuple bits for page bits. The scan still reads {fmtNum(r.walk.length)} pages ({fmtNum(r.pagesWithMatch)} actually hold a match), but it now examines {fmtNum(r.examined)} tuples to return {fmtNum(r.returned)} rows and throws {fmtNum(r.removed)} away on recheck.
            </>
          ) : (
            <>
              <strong>Every page is exact.</strong>{' '}
              {lossyChildren.length > 0
                ? `The ${lossyChildren.join(' and ')} bitmap${lossyChildren.length > 1 ? 's' : ''} went lossy under the ${fmtNum(r.maxentries)}-entry limit, but no lossy page survived the BitmapAnd: `
                : `The largest bitmap peaks at ${fmtNum(r.peakEntries)} entries and work_mem allows ${fmtNum(r.maxentries)} (${fmtBytes(r.maxentries * ENTRY_BYTES)}): `}
              the scan reads {fmtNum(r.walk.length)} pages, examines only the {fmtNum(r.examined)} tuples the bitmap names{r.recheckFlaggedExact > 0 ? `, and evaluates the Recheck Cond only on the ${fmtNum(r.recheckFlaggedExact)} flagged pages, where ${fmtNum(r.removed)} tuples fail it` : ', and never evaluates the Recheck Cond'}.
            </>
          )}
        </Note>
      }
      table={
        <table className="viz-table">
          <thead>
            <tr>
              <th>Chunk (pages)</th>
              <th>Exact</th>
              <th>Exact, recheck</th>
              <th>Lossy</th>
              <th>Tuples examined</th>
              <th>Removed by recheck</th>
              <th>Returned</th>
            </tr>
          </thead>
          <tbody>
            {chunkRows.map((c) => (
              <tr key={c.c}>
                <td>
                  {c.lo}–{c.lo + PAGES_PER_CHUNK - 1}
                </td>
                <td>{fmtNum(c.exact)}</td>
                <td>{fmtNum(c.flagged)}</td>
                <td>{fmtNum(c.lossy)}</td>
                <td>{fmtNum(c.examined)}</td>
                <td>{fmtNum(c.removed)}</td>
                <td>{fmtNum(c.returned)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      }
    >
      <div className="viz-controls">
        <Button primary onClick={() => setCursor(0)}>
          Walk the heap
        </Button>
        <Button onClick={reset} disabled={done}>
          Show totals
        </Button>
      </div>
      <svg viewBox={`0 0 ${W} ${H}`} width={W} height={H} role="img" aria-label={`Heap page grid: ${r.walk.length} pages in the bitmap, ${r.lossyPages} of them lossy`}>
        {[
          { c: r.childA, x: LEFT, label: 'Bitmap Index Scan on orders_created_at_idx' },
          ...(r.childB ? [{ c: r.childB, x: LEFT + COLS * SMALL + 24, label: 'Bitmap Index Scan on orders_amount_idx' }] : []),
        ].map(({ c, x, label }) => (
          <g key={label}>
            <text x={x} y={14} fontSize={12} fill="var(--viz-ink)">
              {label}
            </text>
            <Grid states={c.states} x0={x} y0={smallTop} cell={SMALL} />
            {childLines(c).map((line, li) => (
              <text key={li} x={x} y={smallTop + smallH + 15 + li * 14} fontSize={11} fill="var(--viz-ink-2)">
                {c.ownBitmap || li === 1 ? line : line.replace(/ · peak.*$/, '')}
              </text>
            ))}
          </g>
        ))}
        {combine === 'one' ? (
          <text x={LEFT + COLS * SMALL + 24} y={smallTop + 40} fontSize={11} fill="var(--viz-ink-2)">
            <tspan x={LEFT + COLS * SMALL + 24}>A single index: its bitmap</tspan>
            <tspan x={LEFT + COLS * SMALL + 24} dy={15}>is the one the heap scan reads.</tspan>
          </text>
        ) : null}
        <text x={LEFT} y={bigTop - 12} fontSize={12} fill="var(--viz-ink)">
          {combine === 'and' ? 'BitmapAnd → ' : combine === 'or' ? 'BitmapOr → ' : ''}Bitmap Heap Scan reads these pages in ascending block order ({fmtNum(COLS)} pages per row)
        </text>
        {Array.from({ length: PAGES / PAGES_PER_CHUNK }, (_, c) => {
          const y = bigTop + ((c * PAGES_PER_CHUNK) / COLS) * BIG;
          return (
            <g key={c}>
              <text x={LEFT - 6} y={y + 9} fontSize={10} textAnchor="end" fill="var(--viz-ink-2)">
                {c * PAGES_PER_CHUNK}
              </text>
              {c > 0 ? <line x1={LEFT - 4} x2={LEFT + COLS * BIG} y1={y - 1} y2={y - 1} stroke="var(--viz-ink-muted)" strokeDasharray="3 3" /> : null}
            </g>
          );
        })}
        <Grid states={r.states} x0={LEFT} y0={bigTop} cell={BIG} />
        {cursorPage !== null ? (
          <rect x={LEFT + (cursorPage % COLS) * BIG - 2} y={bigTop + Math.floor(cursorPage / COLS) * BIG - 2} width={BIG + 2} height={BIG + 2} fill="none" stroke="var(--viz-ink)" strokeWidth={2} />
        ) : null}
        {hover !== null ? (
          <rect x={LEFT + (hover % COLS) * BIG - 1} y={bigTop + Math.floor(hover / COLS) * BIG - 1} width={BIG} height={BIG} fill="none" stroke="var(--viz-ink-2)" strokeWidth={1.5} />
        ) : null}
        <rect
          x={LEFT}
          y={bigTop}
          width={COLS * BIG}
          height={bigH}
          fill="transparent"
          onMouseMove={(e) => {
            const box = (e.currentTarget as SVGRectElement).getBoundingClientRect();
            const col = Math.floor(((e.clientX - box.left) / box.width) * COLS);
            const row = Math.floor(((e.clientY - box.top) / box.height) * (PAGES / COLS));
            const p = row * COLS + col;
            setHover(p >= 0 && p < PAGES ? p : null);
          }}
          onMouseLeave={() => setHover(null)}
        />
      </svg>
      <p style={{ margin: '0.25rem 0 0', fontSize: '0.78rem', color: 'var(--viz-ink-2)', minHeight: '1.2em' }}>{hoverInfo ?? 'Hover a page for its entry and what the heap scan does there. Dashed lines mark the 256-page lossy chunks.'}</p>
      <pre style={{ margin: '0.5rem 0 0', fontSize: '0.75rem', lineHeight: 1.45, overflowX: 'auto', color: 'var(--viz-ink)' }} aria-label="EXPLAIN ANALYZE excerpt">
        {explain}
      </pre>
    </VizPanel>
  );
}
