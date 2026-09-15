import { useMemo, useState } from 'react';
import { VizPanel, Segmented, Slider, Check, Button, Legend, Stats, Note, makeRng, fmtNum } from './Viz';

/**
 * A BRIN index over a strip of heap pages, following PostgreSQL's src/backend/access/brin:
 * - The heap is split into block ranges of pages_per_range pages. Each range has at most one summary
 *   tuple, found through the revmap. CREATE INDEX summarizes every range, including the partial one at the end.
 * - Scan (bringetbitmap): walk the ranges in order. A range with no summary is always added to the bitmap;
 *   otherwise the opclass's consistent function decides. Every page of a matching range goes into a lossy
 *   bitmap, and the Bitmap Heap Scan rechecks every row on those pages.
 * - Insert (brininsert): a row landing in a summarized range widens its summary if the value falls outside it;
 *   a row landing in an unsummarized range changes nothing. Summaries never narrow on DELETE or VACUUM.
 * - VACUUM summarizes unsummarized ranges but skips the partial range at the end of the table
 *   (brinsummarize(..., include_partial = false)); brin_summarize_new_values() includes it.
 * - autosummarize: when a row is inserted as the first item of the first page of a new range, the previous range
 *   is queued for summarization by an autovacuum worker. The lab applies the request immediately.
 * - minmax_multi keeps up to values_per_range values (points count 1, intervals count 2); when it runs out it keeps
 *   the widest gaps between values and merges the rest. The lab uses values_per_range = 8 (Postgres's minimum; default 32)
 *   because its ranges hold hundreds of rows, not tens of thousands.
 * - pg_stats.correlation uses analyze.c's formula over every row (ANALYZE uses a sample), and the planner's
 *   expected ranges follow brincostestimate: min(ceil(ranges x selectivity) / |correlation|, ranges).
 * Model assumptions: 12 rows per page (real heap pages hold tens to about 290 rows), values are integer
 * timestamps, appends always go to the end of the heap (in Postgres, free space found through the FSM can put
 * new rows in old pages), and an UPDATE is modeled as a HOT update that keeps the row on its page (PG 16+).
 */

export const PAGE_ROWS = 12;
export const START_PAGES = 100;
export const MAX_PAGES = 136;
export const PPR_OPTIONS = [1, 2, 4, 8, 16, 32] as const;
export const VALUES_PER_RANGE = 8;
export const SCATTER_STEPS = [0, 0.0025, 0.005, 0.01, 0.02, 0.05, 0.1, 0.25, 0.5, 1] as const;
export const JITTER_STEPS = [0, 12, 48, 96, 192, 384, 1200] as const;

export type Pattern = 'scatter' | 'jitter';
export type Opclass = 'minmax' | 'minmax_multi';
export type Row = { v: number; moved: boolean };
export type Interval = [number, number];
/** null = no summary tuple (unsummarized); [] = summarized but empty; otherwise one (minmax) or several intervals. */
export type Summary = Interval[] | null;

export type Lab = {
  pages: Row[][];
  ppr: number;
  opclass: Opclass;
  summaries: Summary[];
  nextTs: number;
  ops: number;
  event: string;
  touched: number[];
};

/* ------------------------------------------------------------------ heap */

/**
 * Mirrors `CREATE TABLE ... AS SELECT ... ORDER BY <key>` with key = random position (scatter) or i + random jitter.
 * Rows given a random position are flagged `moved`, so DELETE out-of-place rows can remove them; jittered rows are not.
 */
export function buildHeap(pattern: Pattern, amount: number, seed = 7): Row[][] {
  const n = START_PAGES * PAGE_ROWS;
  const rng = makeRng(seed);
  const keyed: { i: number; key: number; moved: boolean }[] = [];
  for (let i = 0; i < n; i++) {
    const u = rng();
    const r = rng();
    const moved = pattern === 'scatter' && u < amount;
    const key = pattern === 'scatter' ? (moved ? r * n : i) : i + u * amount;
    keyed.push({ i, key, moved });
  }
  keyed.sort((a, b) => a.key - b.key || a.i - b.i);
  const pages: Row[][] = [];
  for (let p = 0; p < START_PAGES; p++) pages.push(keyed.slice(p * PAGE_ROWS, (p + 1) * PAGE_ROWS).map((k) => ({ v: k.i, moved: k.moved })));
  return pages;
}

export const rangeCount = (pages: Row[][], ppr: number) => Math.max(1, Math.ceil(pages.length / ppr));

/** pg_stats.correlation as analyze.c computes it: sort by value (ties by physical position), correlate rank with position. */
export function correlation(pages: Row[][]): number {
  const vals: { v: number; tupno: number }[] = [];
  for (const pg of pages) for (const r of pg) vals.push({ v: r.v, tupno: vals.length });
  const n = vals.length;
  if (n < 2) return 0;
  vals.sort((a, b) => a.v - b.v || a.tupno - b.tupno);
  let xy = 0;
  for (let i = 0; i < n; i++) xy += i * vals[i].tupno;
  const xsum = ((n - 1) * n) / 2;
  const x2sum = ((n - 1) * n * (2 * n - 1)) / 6;
  return (n * xy - xsum * xsum) / (n * x2sum - xsum * xsum);
}

/* ------------------------------------------------------------- summaries */

const width = (iv: Interval) => (iv[0] === iv[1] ? 1 : 2);
const valuesUsed = (ivs: Interval[]) => ivs.reduce((s, iv) => s + width(iv), 0);

/**
 * minmax_multi's reduce_expanded_ranges: when the summary needs more than values_per_range values, keep only the
 * (values_per_range / 2 − 1) widest gaps between sorted values and merge everything else, leaving at most
 * values_per_range / 2 intervals. Ties go to the leftmost gap; a gap between adjacent integers hides nothing and is never kept.
 */
export function compact(ivs: Interval[], limit = VALUES_PER_RANGE): Interval[] {
  if (ivs.length === 0 || valuesUsed(ivs) <= limit) return ivs.map((iv) => [iv[0], iv[1]] as Interval);
  const keep = limit / 2 - 1;
  const gaps: { i: number; g: number }[] = [];
  for (let i = 0; i + 1 < ivs.length; i++) gaps.push({ i, g: ivs[i + 1][0] - ivs[i][1] });
  gaps.sort((a, b) => b.g - a.g || a.i - b.i);
  const cut = new Set(gaps.slice(0, keep).filter((x) => x.g > 1).map((x) => x.i));
  const out: Interval[] = [];
  let cur: Interval = [ivs[0][0], ivs[0][1]];
  for (let i = 0; i + 1 < ivs.length; i++) {
    if (cut.has(i)) {
      out.push(cur);
      cur = [ivs[i + 1][0], ivs[i + 1][1]];
    } else cur = [cur[0], Math.max(cur[1], ivs[i + 1][1])];
  }
  out.push(cur);
  return out;
}

export function summarizeValues(values: number[], opclass: Opclass): Interval[] {
  if (values.length === 0) return [];
  if (opclass === 'minmax') {
    let lo = Infinity;
    let hi = -Infinity;
    for (const v of values) {
      if (v < lo) lo = v;
      if (v > hi) hi = v;
    }
    return [[lo, hi]];
  }
  const sorted = Array.from(new Set(values)).sort((a, b) => a - b);
  return compact(sorted.map((v) => [v, v] as Interval));
}

const rangePages = (pages: Row[][], ppr: number, r: number) => pages.slice(r * ppr, Math.min(pages.length, (r + 1) * ppr));
const rangeValues = (pages: Row[][], ppr: number, r: number) => rangePages(pages, ppr, r).flatMap((pg) => pg.map((row) => row.v));

export function summarizeRange(pages: Row[][], ppr: number, r: number, opclass: Opclass): Interval[] {
  return summarizeValues(rangeValues(pages, ppr, r), opclass);
}

/** CREATE INDEX: every range is summarized, including the partial range at the end of the heap. */
export function createIndex(pages: Row[][], ppr: number, opclass: Opclass): Summary[] {
  return Array.from({ length: rangeCount(pages, ppr) }, (_, r) => summarizeRange(pages, ppr, r, opclass));
}

/** addValue: returns the widened summary, or the same object when the value is already covered. */
export function addValue(s: Interval[], v: number, opclass: Opclass): Interval[] {
  if (s.some((iv) => iv[0] <= v && v <= iv[1])) return s;
  if (s.length === 0) return [[v, v]];
  if (opclass === 'minmax') return [[Math.min(s[0][0], v), Math.max(s[0][1], v)]];
  const next = [...s.map((iv) => [iv[0], iv[1]] as Interval), [v, v] as Interval].sort((a, b) => a[0] - b[0]);
  return compact(next);
}

export const fmtSummary = (s: Summary) => (s === null ? 'no summary' : s.length === 0 ? 'empty' : s.map((iv) => (iv[0] === iv[1] ? `${iv[0]}` : `${iv[0]}–${iv[1]}`)).join(', '));

export function newLab(pattern: Pattern, amount: number, ppr: number, opclass: Opclass): Lab {
  const pages = buildHeap(pattern, amount);
  return { pages, ppr, opclass, summaries: createIndex(pages, ppr, opclass), nextTs: START_PAGES * PAGE_ROWS, ops: 0, event: '', touched: [] };
}

/** REINDEX with a new pages_per_range or opclass over the current heap. */
export function reindex(lab: Lab, ppr: number, opclass: Opclass): Lab {
  return { ...lab, ppr, opclass, summaries: createIndex(lab.pages, ppr, opclass), event: `Index rebuilt with pages_per_range = ${ppr} and ${opclass}: every range, including the partial one at the end, has a fresh summary.`, touched: [] };
}

/* ------------------------------------------------------------ operations */

export const freeSlots = (lab: Lab) => {
  const last = lab.pages[lab.pages.length - 1];
  return (MAX_PAGES - lab.pages.length) * PAGE_ROWS + (last ? PAGE_ROWS - last.length : 0);
};

type InsertLog = { widened: Map<number, string>; autosummarized: number[]; unsummarizedHits: Set<number>; firstPage: number; lastPage: number };

function insertRows(lab: Lab, rows: Row[], autosummarize: boolean): { lab: Lab; log: InsertLog } {
  const pages = lab.pages.map((pg) => pg.slice());
  const summaries = lab.summaries.slice();
  const log: InsertLog = { widened: new Map(), autosummarized: [], unsummarizedHits: new Set(), firstPage: -1, lastPage: -1 };
  for (const row of rows) {
    let p = pages.length - 1;
    if (pages[p].length >= PAGE_ROWS) {
      if (pages.length >= MAX_PAGES) break;
      pages.push([]);
      p++;
    }
    const offset = pages[p].length; // 0 = FirstOffsetNumber
    pages[p].push(row);
    if (log.firstPage < 0) log.firstPage = p;
    log.lastPage = p;
    const r = Math.floor(p / lab.ppr);
    while (summaries.length <= r) summaries.push(null); // a new range starts with no summary
    const heapBlk = r * lab.ppr;
    // brininsert: first row on the first page of a new, non-first range requests summarization of the previous one.
    if (autosummarize && heapBlk > 0 && heapBlk === p && offset === 0 && summaries[r - 1] === null) {
      summaries[r - 1] = summarizeRange(pages, lab.ppr, r - 1, lab.opclass);
      log.autosummarized.push(r - 1);
    }
    const s = summaries[r];
    if (s === null) {
      log.unsummarizedHits.add(r);
      continue;
    }
    const next = addValue(s, row.v, lab.opclass);
    if (next !== s) {
      if (!log.widened.has(r)) log.widened.set(r, fmtSummary(s));
      summaries[r] = next;
    }
  }
  return { lab: { ...lab, pages, summaries, ops: lab.ops + 1 }, log };
}

const cap = (t: string) => t.charAt(0).toUpperCase() + t.slice(1);
const rangeLabel = (lab: Lab, r: number) => `range ${r} (pages ${r * lab.ppr}–${Math.min(lab.pages.length, (r + 1) * lab.ppr) - 1})`;

function describeInsert(lab: Lab, log: InsertLog, what: string) {
  const parts: string[] = [`${what} landed on pages ${log.firstPage}–${log.lastPage}.`];
  for (const [r, before] of log.widened) parts.push(`The summary of ${rangeLabel(lab, r)} widened from ${before} to ${fmtSummary(lab.summaries[r])}.`);
  if (log.unsummarizedHits.size) parts.push(`${cap([...log.unsummarizedHits].map((r) => rangeLabel(lab, r)).join(', '))} ${log.unsummarizedHits.size === 1 ? 'has' : 'have'} no summary, so the index did not change there — and every scan reads those pages.`);
  if (log.autosummarized.length) parts.push(`autosummarize queued ${log.autosummarized.map((r) => `range ${r}`).join(', ')} for an autovacuum worker, which summarized it.`);
  if (!log.widened.size && !log.unsummarizedHits.size) parts.push('Every value fell inside its range’s existing summary: no index change.');
  return parts.join(' ');
}

export function appendInOrder(lab: Lab, autosummarize: boolean, count = 3 * PAGE_ROWS): Lab {
  const rows = Array.from({ length: count }, (_, k) => ({ v: lab.nextTs + k, moved: false }));
  const { lab: out, log } = insertRows(lab, rows, autosummarize);
  const next = { ...out, nextTs: lab.nextTs + count };
  return { ...next, event: describeInsert(next, log, `${count} new rows in timestamp order`), touched: [...log.widened.keys(), ...log.unsummarizedHits, ...log.autosummarized] };
}

export function appendLate(lab: Lab, autosummarize: boolean, count = PAGE_ROWS): Lab {
  const rng = makeRng(1000 + lab.ops * 7919);
  const rows = Array.from({ length: count }, () => ({ v: Math.floor(rng() * lab.nextTs), moved: true }));
  const { lab: out, log } = insertRows(lab, rows, autosummarize);
  return { ...out, event: describeInsert(out, log, `${count} late rows with old timestamps`), touched: [...log.widened.keys(), ...log.unsummarizedHits, ...log.autosummarized] };
}

/** UPDATE a few old rows to a new timestamp. Modeled as HOT: the new version stays on the same page. */
export function updateToNow(lab: Lab, count = 3): Lab {
  const rng = makeRng(5000 + lab.ops * 104729);
  const pages = lab.pages.map((pg) => pg.slice());
  const summaries = lab.summaries.slice();
  const widened = new Map<number, string>();
  const touchedPages: number[] = [];
  let ts = lab.nextTs;
  const limit = Math.max(1, Math.floor(pages.length * 0.8));
  for (let k = 0, tries = 0; k < count && tries < 200; tries++) {
    const p = Math.floor(rng() * limit);
    if (!pages[p].length) continue;
    const slot = Math.floor(rng() * pages[p].length);
    if (pages[p][slot].moved) continue;
    pages[p][slot] = { v: ts, moved: true };
    const r = Math.floor(p / lab.ppr);
    const s = summaries[r];
    if (s !== null) {
      const next = addValue(s, ts, lab.opclass);
      if (next !== s) {
        if (!widened.has(r)) widened.set(r, fmtSummary(s));
        summaries[r] = next;
      }
    }
    touchedPages.push(p);
    ts++;
    k++;
  }
  const out: Lab = { ...lab, pages, summaries, nextTs: ts, ops: lab.ops + 1, touched: [...widened.keys()] };
  const parts = [`UPDATE set ts = now() on ${touchedPages.length} old rows, on pages ${touchedPages.join(', ')}; each new version stayed on its page as a HOT update.`];
  for (const [r, before] of widened) parts.push(`The summary of ${rangeLabel(out, r)} widened from ${before} to ${fmtSummary(out.summaries[r])}.`);
  return { ...out, event: parts.join(' ') };
}

/** DELETE the rows placed out of order (scattered, late or updated) and let VACUUM reclaim them: summaries do not narrow. */
export function deleteMoved(lab: Lab): Lab {
  let removed = 0;
  const pages = lab.pages.map((pg) => pg.filter((row) => (row.moved ? (removed++, false) : true)));
  return {
    ...lab,
    pages,
    ops: lab.ops + 1,
    touched: [],
    event: removed
      ? `Deleted ${removed} out-of-place row${removed === 1 ? '' : 's'}. The heap no longer holds those values, but no summary narrowed: BRIN never subtracts a value, and VACUUM does not revisit summarized ranges.`
      : 'Nothing to delete: no row was placed at a random page, inserted late or updated.',
  };
}

/** VACUUM: summarize unsummarized ranges, except the partial range at the end of the table. */
export function vacuum(lab: Lab): Lab {
  const summaries = lab.summaries.slice();
  const done: number[] = [];
  let skippedPartial = -1;
  for (let r = 0; r < summaries.length; r++) {
    if (summaries[r] !== null) continue;
    if (r * lab.ppr + lab.ppr > lab.pages.length) {
      skippedPartial = r;
      continue;
    }
    summaries[r] = summarizeRange(lab.pages, lab.ppr, r, lab.opclass);
    done.push(r);
  }
  const out = { ...lab, summaries, ops: lab.ops + 1, touched: done };
  const parts = [done.length ? `VACUUM summarized ${done.map((r) => `range ${r}`).join(', ')}.` : 'VACUUM found no complete unsummarized range.'];
  if (skippedPartial >= 0) parts.push(`It skipped ${rangeLabel(out, skippedPartial)}: that is the partial range at the end of the table, which VACUUM leaves for brin_summarize_new_values().`);
  parts.push('Summaries that already exist are left as they are, however wide.');
  return { ...out, event: parts.join(' ') };
}

/** brin_summarize_new_values(): summarize every unsummarized range, including the partial one at the end. */
export function summarizeNewValues(lab: Lab): Lab {
  const summaries = lab.summaries.slice();
  const done: number[] = [];
  for (let r = 0; r < summaries.length; r++) {
    if (summaries[r] === null) {
      summaries[r] = summarizeRange(lab.pages, lab.ppr, r, lab.opclass);
      done.push(r);
    }
  }
  return { ...lab, summaries, ops: lab.ops + 1, touched: done, event: `brin_summarize_new_values() returned ${done.length}${done.length ? `: ${done.map((r) => `range ${r}`).join(', ')} now ${done.length === 1 ? 'has a summary' : 'have summaries'}` : ' — nothing was unsummarized'}.` };
}

/** brin_desummarize_range() then brin_summarize_range() on every range: summaries shrink back to the rows present now. */
export function resummarizeAll(lab: Lab): Lab {
  const summaries = lab.summaries.map((_, r) => summarizeRange(lab.pages, lab.ppr, r, lab.opclass));
  const narrowed = summaries.map((s, r) => (fmtSummary(s) !== fmtSummary(lab.summaries[r]) ? r : -1)).filter((r) => r >= 0);
  return {
    ...lab,
    summaries,
    ops: lab.ops + 1,
    touched: narrowed,
    event: narrowed.length ? `Desummarized and re-summarized every range: ${narrowed.length} summar${narrowed.length === 1 ? 'y' : 'ies'} changed to match the rows present now (${narrowed.map((r) => `range ${r}`).join(', ')}).` : 'Desummarized and re-summarized every range: every summary already matched the rows present.',
  };
}

/* ------------------------------------------------------------------ scan */

export type RangeVerdict = 'pruned' | 'scan' | 'unsummarized';
export type RangeRow = { r: number; first: number; last: number; summary: Summary; verdict: RangeVerdict; rows: number; matches: number };

/** bringetbitmap + Bitmap Heap Scan for `ts BETWEEN lo AND hi`. */
export function scan(pages: Row[][], ppr: number, summaries: Summary[], lo: number, hi: number) {
  const ranges: RangeRow[] = [];
  let pagesRead = 0;
  let rechecked = 0;
  let matched = 0;
  let missed = 0;
  let totalMatches = 0;
  let totalRows = 0;
  const n = rangeCount(pages, ppr);
  for (let r = 0; r < n; r++) {
    const pgs = rangePages(pages, ppr, r);
    const s = r < summaries.length ? summaries[r] : null;
    const verdict: RangeVerdict = s === null ? 'unsummarized' : s.some((iv) => iv[0] <= hi && iv[1] >= lo) ? 'scan' : 'pruned';
    let rows = 0;
    let matches = 0;
    for (const pg of pgs)
      for (const row of pg) {
        rows++;
        if (row.v >= lo && row.v <= hi) matches++;
      }
    totalRows += rows;
    totalMatches += matches;
    if (verdict === 'pruned') missed += matches;
    else {
      pagesRead += pgs.length;
      rechecked += rows;
      matched += matches;
    }
    ranges.push({ r, first: r * ppr, last: r * ppr + pgs.length - 1, summary: s, verdict, rows, matches });
  }
  const scanned = ranges.filter((x) => x.verdict !== 'pruned').length;
  return {
    ranges,
    pagesRead,
    rechecked,
    matched,
    discarded: rechecked - matched,
    missed,
    totalRows,
    totalMatches,
    rangesScanned: scanned,
    rangesPruned: ranges.length - scanned,
    rangesUnsummarized: ranges.filter((x) => x.verdict === 'unsummarized').length,
    wasted: ranges.filter((x) => x.verdict === 'scan' && x.matches === 0).length,
    bitmapIndexRows: pagesRead * 10, // bringetbitmap reports pages x 10 as its row count
  };
}

/** brincostestimate: ranges the planner expects to visit, from selectivity and |correlation|. */
export function plannerRanges(ranges: number, selectivity: number, corr: number) {
  const c = Math.abs(corr);
  const minimal = Math.ceil(ranges * selectivity);
  return c < 1e-10 ? ranges : Math.min(minimal / c, ranges);
}

/** For the pages_per_range trade-off: a freshly built index at each setting, over the current heap. */
export function sweep(pages: Row[][], opclass: Opclass, lo: number, hi: number) {
  return PPR_OPTIONS.map((ppr) => {
    const res = scan(pages, ppr, createIndex(pages, ppr, opclass), lo, hi);
    return { ppr, entries: rangeCount(pages, ppr), pagesRead: res.pagesRead, rechecked: res.rechecked };
  });
}

/* ----------------------------------------------------------------- view */

export default function BrinZoneMapPruningLab() {
  const [pattern, setPattern] = useState<Pattern>('scatter');
  const [scatterStep, setScatterStep] = useState(0);
  const [jitterStep, setJitterStep] = useState(2);
  const [pprIdx, setPprIdx] = useState(3);
  const [opclass, setOpclass] = useState<Opclass>('minmax');
  const [loRaw, setLo] = useState(600);
  const [span, setSpan] = useState(60);
  const [autosummarize, setAutosummarize] = useState(false);
  const amountFor = (p: Pattern, s: number, j: number) => (p === 'scatter' ? SCATTER_STEPS[s] : JITTER_STEPS[j]);
  const [lab, setLab] = useState<Lab>(() => newLab('scatter', SCATTER_STEPS[0], PPR_OPTIONS[3], 'minmax'));

  const ppr = PPR_OPTIONS[pprIdx];
  const maxTs = Math.max(lab.nextTs, START_PAGES * PAGE_ROWS);
  const lo = Math.min(loRaw, Math.max(0, maxTs - span));
  const hi = lo + span - 1;
  const res = useMemo(() => scan(lab.pages, lab.ppr, lab.summaries, lo, hi), [lab, lo, hi]);
  const corr = useMemo(() => correlation(lab.pages), [lab.pages]);
  const sw = useMemo(() => sweep(lab.pages, lab.opclass, lo, hi), [lab.pages, lab.opclass, lo, hi]);
  const nPages = lab.pages.length;
  const selectivity = res.totalRows ? res.totalMatches / res.totalRows : 0;
  const expected = plannerRanges(res.ranges.length, selectivity, corr);
  const room = freeSlots(lab);

  const rebuildHeap = (p: Pattern, s: number, j: number) => setLab(newLab(p, amountFor(p, s, j), lab.ppr, lab.opclass));

  // ---- main figure geometry
  const W = 720;
  const L = 52;
  const R = 12;
  const TOP = 22;
  const PH = 230;
  const pw = (W - L - R) / MAX_PAGES;
  const X = (p: number) => L + p * pw;
  const Y = (v: number) => TOP + PH - (v / maxTs) * PH;
  const STRIP = TOP + PH + 10;
  const H = STRIP + 44;
  const color: Record<RangeVerdict, string> = { pruned: 'var(--viz-stale)', scan: 'var(--viz-2)', unsummarized: 'var(--viz-warning)' };
  const touched = new Set(lab.touched);

  // ---- sweep chart geometry
  const SW_H = 128;
  const panelW = (W - L - R - 40) / 2;
  const maxEntries = Math.max(...sw.map((s) => s.entries));
  const barW = panelW / sw.length;

  const amountLabel = pattern === 'scatter' ? `${fmtNum(SCATTER_STEPS[scatterStep] * 100, 2)}%` : `±${JITTER_STEPS[jitterStep]} rows`;
  const unsumm = res.ranges.filter((x) => x.verdict === 'unsummarized');
  // A summary is "stretched" when the ts values it covers add up to more than three ranges' worth of ordered rows.
  const anyMoved = useMemo(() => lab.pages.some((pg) => pg.some((row) => row.moved)), [lab.pages]);
  const stretched = res.ranges.filter((x) => x.summary && x.summary.reduce((sum, iv) => sum + iv[1] - iv[0] + 1, 0) > 3 * lab.ppr * PAGE_ROWS).length;
  const expectedText = fmtNum(expected, 1);
  const rangesWord = expectedText === '1' ? 'range' : 'ranges';

  return (
    <VizPanel
      title="Pruning heap pages with block range summaries"
      subtitle="Each dot is a row, placed by the heap page it lives on and its ts value. The index keeps one summary per block range; a query reads every page of every range whose summary might match, then rechecks each row."
      controls={
        <>
          <Segmented
            label="How rows are out of order"
            value={pattern}
            onChange={(p) => {
              setPattern(p);
              rebuildHeap(p, scatterStep, jitterStep);
            }}
            options={[
              { value: 'scatter', label: 'A few rows anywhere' },
              { value: 'jitter', label: 'Every row nearby' },
            ]}
          />
          {pattern === 'scatter' ? (
            <Slider
              label="Rows at random pages"
              min={0}
              max={SCATTER_STEPS.length - 1}
              value={scatterStep}
              onChange={(s) => {
                setScatterStep(s);
                rebuildHeap('scatter', s, jitterStep);
              }}
              format={(s) => `${fmtNum(SCATTER_STEPS[s] * 100, 2)}%`}
            />
          ) : (
            <Slider
              label="Each row displaced by up to"
              min={0}
              max={JITTER_STEPS.length - 1}
              value={jitterStep}
              onChange={(j) => {
                setJitterStep(j);
                rebuildHeap('jitter', scatterStep, j);
              }}
              format={(j) => `${JITTER_STEPS[j]} rows`}
            />
          )}
          <Slider
            label="pages_per_range"
            min={0}
            max={PPR_OPTIONS.length - 1}
            value={pprIdx}
            onChange={(i) => {
              setPprIdx(i);
              setLab((l) => reindex(l, PPR_OPTIONS[i], l.opclass));
            }}
            format={(i) => `${PPR_OPTIONS[i]}`}
          />
          <Segmented
            label="Operator class"
            value={opclass}
            onChange={(o) => {
              setOpclass(o);
              setLab((l) => reindex(l, l.ppr, o));
            }}
            options={[
              { value: 'minmax', label: 'minmax' },
              { value: 'minmax_multi', label: 'minmax_multi' },
            ]}
          />
          <Slider label="WHERE ts from" min={0} max={Math.max(0, maxTs - span)} step={10} value={lo} onChange={setLo} />
          <Slider label="Predicate width" min={10} max={600} step={10} value={span} onChange={setSpan} format={(v) => `${v} ts`} />
        </>
      }
      legend={
        <Legend
          items={[
            { label: 'Row', color: 'var(--viz-1)' },
            { label: 'Predicate and matching rows', color: 'var(--viz-7)' },
            { label: 'Summary rules the range out: pruned', color: 'var(--viz-stale)' },
            { label: 'Summary might match: range read', color: 'var(--viz-2)' },
            { label: 'No summary: range always read', color: 'var(--viz-warning)' },
          ]}
        />
      }
      stats={
        <Stats
          items={[
            { label: 'pg_stats.correlation', value: fmtNum(corr, 4), hint: 'analyze.c’s rank correlation between value order and physical order, computed over every row' },
            { label: 'Ranges pruned', value: `${res.rangesPruned} of ${res.ranges.length}`, hint: 'The index holds one summary per range' },
            { label: 'Heap pages read', value: `${res.pagesRead} of ${nPages}`, hint: 'Every page of every range that was not pruned goes into the lossy bitmap' },
            { label: 'Rows rechecked → kept', value: `${fmtNum(res.rechecked)} → ${fmtNum(res.matched)}`, hint: 'Rows Removed by Index Recheck = the difference' },
            { label: 'Planner expects', value: `≈${expectedText} ${rangesWord}`, hint: 'brincostestimate: min(ceil(ranges × selectivity) / |correlation|, ranges)' },
          ]}
        />
      }
      note={
        <Note>
          <strong>
            ts {lo}–{hi}: {res.rangesPruned} of {res.ranges.length} ranges pruned, {res.pagesRead} of {nPages} pages read, {fmtNum(res.discarded)} rows discarded by the recheck.
          </strong>{' '}
          {res.missed > 0 ? `MODEL ERROR: ${res.missed} matching rows were in pruned ranges. ` : ''}
          {unsumm.length ? `${unsumm.length} range${unsumm.length === 1 ? ' has' : 's have'} no summary and ${unsumm.length === 1 ? 'is' : 'are'} read whatever the predicate. ` : ''}
          {stretched > 0 && Math.abs(corr) > 0.9
            ? `pg_stats.correlation is ${fmtNum(corr, 3)}, yet ${stretched} of ${res.ranges.length} summaries each cover more than three ranges’ worth of ts values: ${anyMoved ? 'one out-of-place row is enough to stretch a min or max' : 'rows displaced across range boundaries widen every min and max'}. `
            : ''}
          {expected < res.rangesScanned / 2 && res.rangesScanned >= 3 ? `The planner, working from correlation alone, expects about ${expectedText} ${rangesWord} and would badly underestimate this scan. ` : ''}
          {lab.event}
        </Note>
      }
      table={
        <table className="viz-table">
          <thead>
            <tr>
              <th>Range</th>
              <th>Pages</th>
              <th>Summary ({lab.opclass})</th>
              <th>Verdict</th>
              <th>Rows rechecked</th>
              <th>Rows matching</th>
            </tr>
          </thead>
          <tbody>
            {res.ranges.map((x) => (
              <tr key={x.r}>
                <td>{x.r}</td>
                <td>
                  {x.first}–{x.last}
                </td>
                <td>{fmtSummary(x.summary)}</td>
                <td>{x.verdict === 'pruned' ? 'pruned' : x.verdict === 'scan' ? (x.matches ? 'read' : 'read, no match') : 'read (no summary)'}</td>
                <td>{x.verdict === 'pruned' ? 0 : x.rows}</td>
                <td>{x.matches}</td>
              </tr>
            ))}
          </tbody>
        </table>
      }
    >
      <div className="viz-controls">
        <Button onClick={() => setLab((l) => appendInOrder(l, autosummarize))} disabled={room < 3 * PAGE_ROWS} title="Append 36 rows with new timestamps">
          INSERT new rows
        </Button>
        <Button onClick={() => setLab((l) => appendLate(l, autosummarize))} disabled={room < PAGE_ROWS} title="Append 12 rows whose timestamps are old">
          INSERT late rows
        </Button>
        <Button onClick={() => setLab((l) => updateToNow(l))} title="Set ts = now() on three old rows (HOT update, same page)">
          UPDATE old rows
        </Button>
        <Button onClick={() => setLab((l) => deleteMoved(l))}>DELETE out-of-place rows</Button>
        <Button onClick={() => setLab((l) => vacuum(l))}>VACUUM</Button>
        <Button primary onClick={() => setLab((l) => summarizeNewValues(l))}>
          brin_summarize_new_values()
        </Button>
        <Button onClick={() => setLab((l) => resummarizeAll(l))} title="brin_desummarize_range + brin_summarize_range on every range">
          Re-summarize all ranges
        </Button>
        <Button onClick={() => setLab(newLab(pattern, amountFor(pattern, scatterStep, jitterStep), ppr, opclass))}>Reset table</Button>
        <Check label="autosummarize" checked={autosummarize} onChange={setAutosummarize} />
      </div>
      <svg viewBox={`0 0 ${W} ${H}`} width={W} height={H} role="img" aria-label={`Heap of ${nPages} pages in ${res.ranges.length} block ranges; ${res.rangesPruned} ranges pruned and ${res.pagesRead} pages read for ts ${lo} to ${hi}`}>
        {/* axes */}
        <line x1={L} x2={L} y1={TOP} y2={TOP + PH} stroke="var(--viz-axis)" />
        <line x1={L} x2={W - R} y1={TOP + PH} y2={TOP + PH} stroke="var(--viz-axis)" />
        {[0, 0.5, 1].map((f) => (
          <text key={f} x={L - 6} y={Y(f * maxTs) + 4} fontSize={10} textAnchor="end" fill="var(--viz-ink-2)">
            {Math.round(f * maxTs)}
          </text>
        ))}
        <text x={12} y={TOP + PH / 2} fontSize={11} fill="var(--viz-ink-2)" transform={`rotate(-90 12 ${TOP + PH / 2})`} textAnchor="middle">
          ts value
        </text>
        {nPages < MAX_PAGES ? (
          <>
            <rect x={X(nPages)} y={TOP} width={X(MAX_PAGES) - X(nPages)} height={PH} fill="none" stroke="var(--viz-grid)" strokeDasharray="3 3" />
            <text x={(X(nPages) + X(MAX_PAGES)) / 2} y={TOP + 14} fontSize={10} textAnchor="middle" fill="var(--viz-ink-2)">
              {X(MAX_PAGES) - X(nPages) > 70 ? 'appends go here' : ''}
            </text>
          </>
        ) : null}

        {/* predicate band */}
        <rect x={L} y={Y(hi)} width={W - L - R} height={Math.max(2, Y(lo) - Y(hi))} fill="var(--viz-7)" fillOpacity={0.12} stroke="var(--viz-7)" strokeWidth={1} strokeDasharray="5 3" />

        {/* range summaries */}
        {res.ranges.map((x) => {
          const x0 = X(x.first) + 0.5;
          const w = Math.max(1.5, X(x.last + 1) - X(x.first) - 1);
          const c = color[x.verdict];
          const bold = touched.has(x.r);
          if (x.summary === null)
            return (
              <g key={x.r}>
                <rect x={x0} y={TOP} width={w} height={PH} fill="var(--viz-warning)" fillOpacity={0.1} stroke={c} strokeWidth={bold ? 2.5 : 1.2} strokeDasharray="4 3" />
                {w > 34 ? (
                  <text x={x0 + w / 2} y={TOP - 6} fontSize={10} textAnchor="middle" fill="var(--viz-ink)">
                    no summary
                  </text>
                ) : null}
              </g>
            );
          return (
            <g key={x.r}>
              {x.summary.map((iv, k) => (
                <rect key={k} x={x0} y={Y(iv[1]) - 1.5} width={w} height={Math.max(3, Y(iv[0]) - Y(iv[1]) + 3)} fill={c} fillOpacity={0.14} stroke={c} strokeWidth={bold ? 2.5 : 1.2} />
              ))}
              {bold && w > 24 ? (
                <text x={x0 + w / 2} y={TOP - 6} fontSize={10} textAnchor="middle" fill="var(--viz-ink)">
                  changed
                </text>
              ) : null}
            </g>
          );
        })}

        {/* rows */}
        {lab.pages.map((pg, p) =>
          pg.map((row, k) => {
            const hit = row.v >= lo && row.v <= hi;
            return <rect key={`${p}-${k}`} x={X(p) + 0.8} y={Y(row.v) - 1} width={Math.max(1.2, pw - 1.6)} height={2} fill={hit ? 'var(--viz-7)' : 'var(--viz-1)'} fillOpacity={hit ? 1 : 0.75} />;
          }),
        )}

        {/* range boundaries */}
        {res.ranges.map((x) => (x.r > 0 && pw * lab.ppr >= 4 ? <line key={x.r} x1={X(x.first)} x2={X(x.first)} y1={TOP + PH} y2={STRIP + 14} stroke="var(--viz-axis)" strokeWidth={0.8} /> : null))}

        {/* lossy page bitmap */}
        <text x={L - 6} y={STRIP + 10} fontSize={10} textAnchor="end" fill="var(--viz-ink-2)">
          bitmap
        </text>
        {res.ranges.map((x) =>
          Array.from({ length: x.last - x.first + 1 }, (_, i) => {
            const p = x.first + i;
            return <rect key={p} x={X(p) + 0.4} y={STRIP} width={Math.max(1, pw - 0.8)} height={12} fill={color[x.verdict]} fillOpacity={x.verdict === 'pruned' ? 0.3 : 0.9} />;
          }),
        )}
        {[0, 32, 64, 96, 128].map((p) => (
          <text key={p} x={X(p)} y={STRIP + 28} fontSize={10} textAnchor="middle" fill="var(--viz-ink-2)">
            {p}
          </text>
        ))}
        <text x={W - R} y={STRIP + 40} fontSize={11} textAnchor="end" fill="var(--viz-ink-2)">
          heap page →
        </text>
      </svg>

      <svg viewBox={`0 0 ${W} ${SW_H}`} width={W} height={SW_H} role="img" aria-label="Index entries and heap pages read at each pages_per_range for this predicate" style={{ marginTop: 6 }}>
        {[
          { title: 'Index entries (one summary per range)', key: 'entries' as const, max: maxEntries, fill: 'var(--viz-5)', x0: L },
          { title: `Heap pages read for ts ${lo}–${hi}`, key: 'pagesRead' as const, max: nPages, fill: 'var(--viz-2)', x0: L + panelW + 40 },
        ].map((panel) => (
          <g key={panel.key}>
            <text x={panel.x0} y={12} fontSize={11} fill="var(--viz-ink)">
              {panel.title}
            </text>
            {sw.map((s, i) => {
              const v = s[panel.key];
              const bh = (v / Math.max(1, panel.max)) * 70;
              const bx = panel.x0 + i * barW + 4;
              const current = s.ppr === lab.ppr;
              return (
                <g key={s.ppr}>
                  <rect x={bx} y={96 - bh} width={barW - 8} height={Math.max(1, bh)} rx={2} fill={panel.fill} fillOpacity={current ? 1 : 0.55} stroke={current ? 'var(--viz-ink)' : 'none'} strokeWidth={1.5} />
                  <text x={bx + (barW - 8) / 2} y={92 - bh} fontSize={10} textAnchor="middle" fill="var(--viz-ink)">
                    {v}
                  </text>
                  <text x={bx + (barW - 8) / 2} y={110} fontSize={10} textAnchor="middle" fill={current ? 'var(--viz-ink)' : 'var(--viz-ink-2)'} fontWeight={current ? 700 : 400}>
                    {s.ppr}
                  </text>
                </g>
              );
            })}
            <text x={panel.x0 + panelW / 2} y={124} fontSize={10} textAnchor="middle" fill="var(--viz-ink-2)">
              pages_per_range (fresh index)
            </text>
          </g>
        ))}
      </svg>
      <p style={{ fontSize: '0.75rem', color: 'var(--viz-ink-2)', margin: '0.3rem 0 0' }}>
        Lab scale: {PAGE_ROWS} rows per page and {START_PAGES} starting pages, so pages_per_range starts at 8 rather than PostgreSQL’s 128; minmax_multi keeps {VALUES_PER_RANGE} values per range. Physical order: {pattern === 'scatter' ? `${amountLabel} of rows placed at random pages` : `each row displaced by up to ${JITTER_STEPS[jitterStep]} rows`}.
      </p>
    </VizPanel>
  );
}
