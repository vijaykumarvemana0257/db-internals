import { useDeferredValue, useMemo, useState, type CSSProperties } from 'react';
import { VizPanel, Segmented, Choice, Check, Slider, Button, Legend, Stats, Note, fmtBytes, fmtNum } from './Viz';

/**
 * One 20-column, 1,000,000-row table stored three ways and scanned by one query:
 *   SELECT <projected columns> FROM t WHERE <predicate on one filter column>
 *
 * Model — every number the lab shows comes from these rules:
 *  - Values are fixed-width and never NULL. No compression, no zone maps, no indexes, so every layout
 *    scans the filter column in full. Encodings and min/max skipping are later pages in the module.
 *  - NSM: a PostgreSQL-shaped 8 KB heap page — 24-byte page header, 4-byte line pointers, a 24-byte tuple
 *    header (23 bytes, MAXALIGNed), each attribute padded to its type's alignment, tuples filled from the
 *    end of the page: 192-byte tuples, 41 per page. The scan deforms every tuple from its first attribute up
 *    to the highest attribute the WHERE clause uses, and — only for rows that pass — on up to the highest
 *    attribute the SELECT list uses (slot_deform_heap_tuple is incremental; ExecQual runs before ExecProject).
 *  - PAX (Ailamaki, DeWitt, Hill, Skounakis, VLDB 2001): the same 41 rows on each 8 KB page, so exactly the
 *    same pages are read. A page holds a header with a minipage directory, a minipage of per-row header
 *    bytes, then one minipage per column; value i of a column sits at minipage start + i × width.
 *  - Column chunks: rows are cut into row groups; each row group stores one chunk per column, in schema order
 *    (Parquet's file layout). Chunks are read in granules of 8,192 rows (ClickHouse's index_granularity);
 *    granules restart at each row group. Optionally a 4-byte explicit row id is stored with every value, as
 *    in the 1985 decomposition storage model.
 *  - Early materialization reads every needed column for every row and stitches rows before filtering.
 *    Late materialization scans the filter column, keeps the matching positions, and reads the other needed
 *    columns only where a match lives: granules with at least one match (as ClickHouse PREWHERE does), or,
 *    inside a PAX page, the cache lines holding matching values.
 *  - Bytes read: whole pages (NSM, PAX) or whole granules (column chunks). Byte ranges: runs of adjacent bytes
 *    a reader must request, before any range coalescing — a seek on a disk, a GET on object storage.
 *  - Cache lines: distinct 64-byte lines holding value bytes the scan reads. Header, line-pointer and row-header
 *    bytes are left out of every layout. Each granule is decoded into its own line-aligned array.
 *  - Stitches: values fetched by row position to attach a column to a row being built. Early: rows × (needed
 *    columns − 1). Late: matching rows × (needed columns other than the filter column). NSM never stitches.
 *  - Matching rows are scattered (a hash of the row number) or clustered (the first rows: a range predicate on
 *    the column the table is sorted by).
 */

/* ------------------------------------------------------------------ schema */

export type Col = { name: string; type: string; width: number; align: number };

export const COLUMNS: readonly Col[] = [
  { name: 'order_id', type: 'int64', width: 8, align: 8 },
  { name: 'customer_id', type: 'int64', width: 8, align: 8 },
  { name: 'product_id', type: 'int32', width: 4, align: 4 },
  { name: 'store_id', type: 'int32', width: 4, align: 4 },
  { name: 'order_date', type: 'date32', width: 4, align: 4 },
  { name: 'ship_date', type: 'date32', width: 4, align: 4 },
  { name: 'quantity', type: 'int32', width: 4, align: 4 },
  { name: 'unit_price', type: 'int64', width: 8, align: 8 },
  { name: 'discount', type: 'int16', width: 2, align: 2 },
  { name: 'tax', type: 'int16', width: 2, align: 2 },
  { name: 'total_cents', type: 'int64', width: 8, align: 8 },
  { name: 'status', type: 'fixed(1)', width: 1, align: 1 },
  { name: 'channel', type: 'fixed(1)', width: 1, align: 1 },
  { name: 'currency', type: 'fixed(3)', width: 3, align: 1 },
  { name: 'country', type: 'fixed(2)', width: 2, align: 1 },
  { name: 'promo_code', type: 'fixed(12)', width: 12, align: 1 },
  { name: 'ship_mode', type: 'fixed(10)', width: 10, align: 1 },
  { name: 'sales_rep', type: 'fixed(16)', width: 16, align: 1 },
  { name: 'comment', type: 'fixed(44)', width: 44, align: 1 },
  { name: 'updated_at', type: 'timestamp', width: 8, align: 8 },
];

export const colIndex = (name: string) => COLUMNS.findIndex((c) => c.name === name);

export const N_ROWS = 1_000_000;
export const PAGE_BYTES = 8192;
export const LINE_BYTES = 64;
export const GRANULE_ROWS = 8192;
export const PAGE_HEADER_BYTES = 24;
export const LINE_POINTER_BYTES = 4;
export const TUPLE_HEADER_BYTES = 24;
export const ROW_ID_BYTES = 4;
export const RG_SIZES = [8192, 16384, 32768, 65536, 122880, 262144, 524288, 1_000_000] as const;
export const SELECTIVITIES = [0.00001, 0.00003, 0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 0.5, 1] as const;

const alignUp = (x: number, a: number) => Math.ceil(x / a) * a;

/** Offset of each attribute inside NSM tuple data, after type-alignment padding. */
export const NSM_OFFSETS: readonly number[] = (() => {
  let o = 0;
  return COLUMNS.map((c) => {
    o = alignUp(o, c.align);
    const at = o;
    o += c.width;
    return at;
  });
})();
const LAST = COLUMNS.length - 1;
export const VALUE_BYTES = COLUMNS.reduce((s, c) => s + c.width, 0);
export const NSM_DATA_BYTES = alignUp(NSM_OFFSETS[LAST] + COLUMNS[LAST].width, 8);
export const NSM_TUPLE_BYTES = TUPLE_HEADER_BYTES + NSM_DATA_BYTES;
export const ROWS_PER_PAGE = Math.floor((PAGE_BYTES - PAGE_HEADER_BYTES) / (NSM_TUPLE_BYTES + LINE_POINTER_BYTES));

export const PAX_DIRECTORY_BYTES = PAGE_HEADER_BYTES + COLUMNS.length * 4;
export const PAX_MINIPAGE_START: readonly number[] = (() => {
  let o = PAX_DIRECTORY_BYTES + ROWS_PER_PAGE * TUPLE_HEADER_BYTES;
  return COLUMNS.map((c) => {
    const at = o;
    o += ROWS_PER_PAGE * c.width;
    return at;
  });
})();

/* --------------------------------------------------------------- matches */

export type Pattern = 'scattered' | 'clustered';

/** Seed chosen so that every selectivity step lands within 1.5 standard deviations of its expected count. */
const MATCH_SEED = Math.imul(21, 0x9e3779b9);

/** lowbias32 integer hash (Wellons): well-mixed bits from consecutive row numbers. */
function mix32(i: number) {
  let x = i ^ MATCH_SEED;
  x ^= x >>> 16;
  x = Math.imul(x, 0x7feb352d);
  x ^= x >>> 15;
  x = Math.imul(x, 0x846ca68b);
  x ^= x >>> 16;
  return x >>> 0;
}

/** prefix[i] = number of matching rows among rows [0, i). */
export function matchPrefix(sel: number, pattern: Pattern, n = N_ROWS) {
  const prefix = new Int32Array(n + 1);
  if (pattern === 'clustered') {
    const m = Math.min(n, Math.round(sel * n));
    for (let i = 0; i < n; i++) prefix[i + 1] = i < m ? i + 1 : m;
  } else {
    const t = sel * 4294967296;
    let c = 0;
    for (let i = 0; i < n; i++) {
      if (mix32(i) < t) c++;
      prefix[i + 1] = c;
    }
  }
  return prefix;
}

/* ----------------------------------------------------------------- model */

export type Query = {
  projected: readonly boolean[];
  filter: number;
  late: boolean;
  rowGroupRows: number;
  explicitIds: boolean;
};

type LineState = { last: number };

/**
 * Distinct cache lines touched by the values at [start + i·w, start + (i+1)·w) for i in [0, n).
 * With all=false only rows that match count. st.last carries the last counted line so that neighbouring
 * segments laid out in address order do not count a shared line twice.
 */
function countLines(start: number, n: number, w: number, rowBase: number, all: boolean, prefix: Int32Array, st: LineState) {
  if (n <= 0) return 0;
  let l = Math.floor(start / LINE_BYTES);
  const lEnd = Math.floor((start + n * w - 1) / LINE_BYTES);
  if (l <= st.last) l = st.last + 1;
  if (all) {
    if (lEnd < l) return 0;
    st.last = lEnd;
    return lEnd - l + 1;
  }
  let count = 0;
  for (; l <= lEnd; l++) {
    const ls = l * LINE_BYTES - start;
    const lo = Math.max(0, Math.floor(ls / w));
    const hi = Math.min(n - 1, Math.ceil((ls + LINE_BYTES) / w) - 1);
    if (hi >= lo && prefix[rowBase + hi + 1] - prefix[rowBase + lo] > 0) {
      count++;
      st.last = l;
    }
  }
  return count;
}

function neededOf(q: Query) {
  const needed = COLUMNS.map((_, c) => c).filter((c) => c === q.filter || q.projected[c]);
  const others = needed.filter((c) => c !== q.filter);
  const endQual = NSM_OFFSETS[q.filter] + COLUMNS[q.filter].width;
  const endAll = others.reduce((m, c) => Math.max(m, NSM_OFFSETS[c] + COLUMNS[c].width), endQual);
  return { needed, others, endQual, endAll };
}

/** Cache lines one NSM page touches: every tuple deformed to the qual's last attribute, passing tuples further. */
export function nsmPageLines(page: number, q: Query, prefix: Int32Array) {
  const n = prefix.length - 1;
  const { endQual, endAll } = neededOf(q);
  const rowBase = page * ROWS_PER_PAGE;
  const rows = Math.min(ROWS_PER_PAGE, n - rowBase);
  const base = page * PAGE_BYTES;
  let count = 0;
  let last = -1;
  // tuples in address order: the last-inserted tuple sits lowest in the page
  for (let t = rows - 1; t >= 0; t--) {
    const hit = prefix[rowBase + t + 1] - prefix[rowBase + t] > 0;
    const a = base + PAGE_BYTES - NSM_TUPLE_BYTES * (t + 1) + TUPLE_HEADER_BYTES;
    const e = a + (hit ? endAll : endQual);
    let l0 = Math.floor(a / LINE_BYTES);
    const l1 = Math.floor((e - 1) / LINE_BYTES);
    if (l0 <= last) l0 = last + 1;
    if (l1 >= l0) {
      count += l1 - l0 + 1;
      last = l1;
    }
  }
  return count;
}

/** Cache lines one PAX page touches: needed minipages whole (early) or filter minipage + lines holding matches (late). */
export function paxPageLines(page: number, q: Query, prefix: Int32Array) {
  const n = prefix.length - 1;
  const { needed } = neededOf(q);
  const rowBase = page * ROWS_PER_PAGE;
  const rows = Math.min(ROWS_PER_PAGE, n - rowBase);
  const base = page * PAGE_BYTES;
  const st = { last: -1 };
  let count = 0;
  for (const c of needed) {
    count += countLines(base + PAX_MINIPAGE_START[c], rows, COLUMNS[c].width, rowBase, !q.late || c === q.filter, prefix, st);
  }
  return count;
}

export type Metrics = { bytesRead: number; totalBytes: number; cacheLines: number; ranges: number; stitches: number };
export type Role = 'filter' | 'projected' | 'unused';
export type ColumnDetail = {
  col: number;
  role: Role;
  granulesRead: number;
  granules: number;
  bytesRead: number;
  cacheLines: number;
  /** merged [rowStart, rowEnd) intervals whose granules were read */
  tracks: [number, number][];
};

export function simulate(q: Query, prefix: Int32Array) {
  const n = prefix.length - 1;
  const matches = prefix[n];
  const { needed, others } = neededOf(q);
  const k = needed.length;
  const stitches = q.late ? matches * others.length : n * (k - 1);

  /* NSM + PAX, page by page. Page bases are multiples of 64, so pages with no match or all matches repeat. */
  const pages = Math.ceil(n / ROWS_PER_PAGE);
  let nsmLines = 0;
  let paxLines = 0;
  const nsmCache = new Map<number, number>();
  const paxCache = new Map<number, number>();
  for (let p = 0; p < pages; p++) {
    const rowBase = p * ROWS_PER_PAGE;
    const rows = Math.min(ROWS_PER_PAGE, n - rowBase);
    const m = prefix[rowBase + rows] - prefix[rowBase];
    const uniform = m === 0 ? 0 : m === rows ? 1 : -1;
    const nKey = uniform < 0 ? -1 : rows * 2 + uniform;
    if (nKey >= 0 && nsmCache.has(nKey)) nsmLines += nsmCache.get(nKey)!;
    else {
      const c = nsmPageLines(p, q, prefix);
      if (nKey >= 0) nsmCache.set(nKey, c);
      nsmLines += c;
    }
    const pKey = !q.late ? rows * 2 : nKey;
    if (pKey >= 0 && paxCache.has(pKey)) paxLines += paxCache.get(pKey)!;
    else {
      const c = paxPageLines(p, q, prefix);
      if (pKey >= 0) paxCache.set(pKey, c);
      paxLines += c;
    }
  }
  const pageBytes = pages * PAGE_BYTES;

  /* Column chunks, in file order: row group → column → granule. */
  const R = q.rowGroupRows;
  const rowGroups = Math.ceil(n / R);
  const idBytes = q.explicitIds ? ROW_ID_BYTES : 0;
  const detail: ColumnDetail[] = COLUMNS.map((_, c) => ({
    col: c,
    role: c === q.filter ? 'filter' : q.projected[c] ? 'projected' : 'unused',
    granulesRead: 0,
    granules: 0,
    bytesRead: 0,
    cacheLines: 0,
    tracks: [],
  }));
  const fileRuns: { start: number; end: number; role: Role }[] = [];
  let colBytes = 0;
  let colLines = 0;
  let ranges = 0;
  let fileOff = 0;
  let prevRead = false;
  for (let g = 0; g < rowGroups; g++) {
    const rgStart = g * R;
    const rgEnd = Math.min(n, rgStart + R);
    for (let c = 0; c < COLUMNS.length; c++) {
      const d = detail[c];
      const W = COLUMNS[c].width + idBytes;
      for (let gs = rgStart; gs < rgEnd; gs += GRANULE_ROWS) {
        const gn = Math.min(GRANULE_ROWS, rgEnd - gs);
        const b = gn * W;
        d.granules++;
        const m = prefix[gs + gn] - prefix[gs];
        const read = d.role === 'filter' || (d.role === 'projected' && (!q.late || m > 0));
        if (read) {
          d.granulesRead++;
          d.bytesRead += b;
          colBytes += b;
          if (!prevRead) ranges++;
          const full = d.role === 'filter' || !q.late || m === gn;
          const lines = full ? Math.ceil(b / LINE_BYTES) : countLines(0, gn, W, gs, false, prefix, { last: -1 });
          d.cacheLines += lines;
          colLines += lines;
          const lt = d.tracks[d.tracks.length - 1];
          if (lt && lt[1] === gs) lt[1] = gs + gn;
          else d.tracks.push([gs, gs + gn]);
          const lf = fileRuns[fileRuns.length - 1];
          if (lf && lf.end === fileOff && lf.role === d.role) lf.end = fileOff + b;
          else fileRuns.push({ start: fileOff, end: fileOff + b, role: d.role });
        }
        prevRead = read;
        fileOff += b;
      }
    }
  }

  const nsm: Metrics = { bytesRead: pageBytes, totalBytes: pageBytes, cacheLines: nsmLines, ranges: 1, stitches: 0 };
  const pax: Metrics = { bytesRead: pageBytes, totalBytes: pageBytes, cacheLines: paxLines, ranges: 1, stitches };
  const col: Metrics = { bytesRead: colBytes, totalBytes: fileOff, cacheLines: colLines, ranges, stitches };
  const otherGranules = others.reduce((s, c) => s + detail[c].granules, 0);
  const otherGranulesRead = others.reduce((s, c) => s + detail[c].granulesRead, 0);
  return { n, matches, needed, others, k, pages, rowGroups, nsm, pax, col, detail, fileRuns, otherGranules, otherGranulesRead };
}

/* ------------------------------------------------------- one page, byte by byte */

/** Byte classes for drawing one page. */
export const CLS = { free: 0, meta: 1, idle: 2, walked: 3, filter: 4, projected: 5 } as const;

export function nsmPageClasses(page: number, q: Query, prefix: Int32Array) {
  const n = prefix.length - 1;
  const { endQual, endAll } = neededOf(q);
  const cls = new Uint8Array(PAGE_BYTES);
  const rowBase = page * ROWS_PER_PAGE;
  const rows = Math.max(0, Math.min(ROWS_PER_PAGE, n - rowBase));
  cls.fill(CLS.meta, 0, PAGE_HEADER_BYTES + rows * LINE_POINTER_BYTES);
  for (let t = 0; t < rows; t++) {
    const T = PAGE_BYTES - NSM_TUPLE_BYTES * (t + 1);
    cls.fill(CLS.meta, T, T + NSM_TUPLE_BYTES);
    const hit = prefix[rowBase + t + 1] - prefix[rowBase + t] > 0;
    const limit = hit ? endAll : endQual;
    COLUMNS.forEach((c, i) => {
      const a = T + TUPLE_HEADER_BYTES + NSM_OFFSETS[i];
      let v: number = CLS.idle;
      if (i === q.filter) v = CLS.filter;
      else if (NSM_OFFSETS[i] + c.width <= limit) v = hit && q.projected[i] ? CLS.projected : CLS.walked;
      cls.fill(v, a, a + c.width);
    });
  }
  return cls;
}

export function paxPageClasses(page: number, q: Query, prefix: Int32Array) {
  const n = prefix.length - 1;
  const cls = new Uint8Array(PAGE_BYTES);
  const rowBase = page * ROWS_PER_PAGE;
  const rows = Math.max(0, Math.min(ROWS_PER_PAGE, n - rowBase));
  cls.fill(CLS.meta, 0, PAX_DIRECTORY_BYTES + rows * TUPLE_HEADER_BYTES);
  COLUMNS.forEach((c, i) => {
    for (let r = 0; r < rows; r++) {
      const a = PAX_MINIPAGE_START[i] + r * c.width;
      const hit = prefix[rowBase + r + 1] - prefix[rowBase + r] > 0;
      let v: number = CLS.idle;
      if (i === q.filter) v = CLS.filter;
      else if (q.projected[i] && (!q.late || hit)) v = CLS.projected;
      cls.fill(v, a, a + c.width);
    }
  });
  return cls;
}

/** Lines (0..127) that hold at least one byte the scan reads. */
export function touchedLines(cls: Uint8Array) {
  const out: number[] = [];
  for (let l = 0; l < PAGE_BYTES / LINE_BYTES; l++) {
    for (let b = l * LINE_BYTES; b < (l + 1) * LINE_BYTES; b++) {
      if (cls[b] >= CLS.walked) {
        out.push(l);
        break;
      }
    }
  }
  return out;
}

/** First page holding a matching row (page 0 if none). */
export function firstMatchPage(prefix: Int32Array) {
  const n = prefix.length - 1;
  if (prefix[n] === 0) return 0;
  let lo = 0;
  let hi = n - 1;
  while (lo < hi) {
    const mid = (lo + hi) >> 1;
    if (prefix[mid + 1] > 0) hi = mid;
    else lo = mid + 1;
  }
  return Math.floor(lo / ROWS_PER_PAGE);
}

function runsOf(cls: Uint8Array) {
  const out: { line: number; x0: number; x1: number; v: number }[] = [];
  for (let l = 0; l < PAGE_BYTES / LINE_BYTES; l++) {
    let s = l * LINE_BYTES;
    const end = s + LINE_BYTES;
    for (let b = s + 1; b <= end; b++) {
      if (b === end || cls[b] !== cls[s]) {
        if (cls[s] !== CLS.free) out.push({ line: l, x0: s - l * LINE_BYTES, x1: b - l * LINE_BYTES, v: cls[s] });
        s = b;
      }
    }
  }
  return out;
}

/* -------------------------------------------------------------------- UI */

const FILL: Record<number, string> = {
  [CLS.meta]: 'var(--viz-axis)',
  [CLS.idle]: 'var(--viz-grid)',
  [CLS.walked]: 'var(--viz-3)',
  [CLS.filter]: 'var(--viz-1)',
  [CLS.projected]: 'var(--viz-2)',
};
const ROLE_FILL: Record<Role, string> = { filter: 'var(--viz-1)', projected: 'var(--viz-2)', unused: 'var(--viz-grid)' };

type PresetKey = 'agg' | 'star' | 'range' | 'wide';
const PRESETS: Record<PresetKey, { label: string; projected: string[]; filter: string; selIdx: number; pattern: Pattern }> = {
  agg: { label: 'Sum 2 columns, 1% of rows match', projected: ['total_cents', 'quantity'], filter: 'order_date', selIdx: 6, pattern: 'scattered' },
  star: { label: 'SELECT * for one customer (10 rows)', projected: COLUMNS.map((c) => c.name), filter: 'customer_id', selIdx: 0, pattern: 'scattered' },
  range: { label: 'Date range on a date-sorted table (3%)', projected: ['total_cents', 'country'], filter: 'order_date', selIdx: 7, pattern: 'clustered' },
  wide: { label: 'Filter on the wide comment column', projected: ['order_id'], filter: 'comment', selIdx: 6, pattern: 'scattered' },
};
const maskOf = (names: string[]) => COLUMNS.map((c) => names.includes(c.name));

const pct = (a: number, b: number) => {
  if (b <= 0) return '—';
  const v = (a / b) * 100;
  return `${fmtNum(v, v < 10 ? 1 : 0)}%`;
};
const fmtSel = (s: number) => `${fmtNum(s * 100, 3)}%`;
const fmtRg = (r: number) => (r >= N_ROWS ? '1,000,000 (one group)' : fmtNum(r));

/** One 8 KB page drawn as 128 vertical strips, one per 64-byte cache line, byte 0 of each line at the top. */
function PageMap({ y, title, cls, lines, sub }: { y: number; title: string; cls: Uint8Array; lines: number[]; sub: string }) {
  const MX = 40;
  const LW = 4.75;
  const BH = 1.4;
  const MY = y + 32;
  const MH = LINE_BYTES * BH;
  const runs = useMemo(() => runsOf(cls), [cls]);
  return (
    <g>
      <text x={0} y={y + 11} fontSize={12} fill="var(--viz-ink)" fontWeight={600}>
        {title}
      </text>
      <text x={0} y={y + 25} fontSize={10.5} fill="var(--viz-ink-2)">
        {sub}
      </text>
      <rect x={MX - 1} y={MY - 1} width={128 * LW + 2} height={MH + 2} fill="var(--viz-plane)" stroke="var(--viz-border)" />
      {runs.map((r, i) => (
        <rect key={i} x={MX + r.line * LW} y={MY + r.x0 * BH} width={LW - 0.6} height={(r.x1 - r.x0) * BH} fill={FILL[r.v]} />
      ))}
      {lines.map((l) => (
        <rect key={`t${l}`} x={MX + l * LW} y={MY + MH + 3} width={LW - 0.6} height={6} fill="var(--viz-4)" />
      ))}
      <text x={MX - 5} y={MY + 8} textAnchor="end" fontSize={9} fill="var(--viz-ink-2)">
        byte 0
      </text>
      <text x={MX - 5} y={MY + MH} textAnchor="end" fontSize={9} fill="var(--viz-ink-2)">
        63
      </text>
      <text x={MX} y={MY + MH + 20} fontSize={9} fill="var(--viz-ink-2)">
        line 0
      </text>
      <text x={MX + 128 * LW} y={MY + MH + 20} textAnchor="end" fontSize={9} fill="var(--viz-ink-2)">
        line 127
      </text>
    </g>
  );
}

export default function ColumnarLayoutScanLab() {
  const [preset, setPreset] = useState<PresetKey | 'custom'>('agg');
  const [projected, setProjected] = useState<boolean[]>(maskOf(PRESETS.agg.projected));
  const [filter, setFilter] = useState(colIndex(PRESETS.agg.filter));
  const [selIdx, setSelIdx] = useState(PRESETS.agg.selIdx);
  const [pattern, setPattern] = useState<Pattern>(PRESETS.agg.pattern);
  const [mat, setMat] = useState<'early' | 'late'>('early');
  const [rgIdx, setRgIdx] = useState(4);
  const [ids, setIds] = useState(false);

  const applyPreset = (key: PresetKey) => {
    const p = PRESETS[key];
    setPreset(key);
    setProjected(maskOf(p.projected));
    setFilter(colIndex(p.filter));
    setSelIdx(p.selIdx);
    setPattern(p.pattern);
  };
  const custom = () => setPreset('custom');

  const inputs = useMemo(
    () => ({ projected, filter, selIdx, pattern, late: mat === 'late', rowGroupRows: RG_SIZES[rgIdx], explicitIds: ids }),
    [projected, filter, selIdx, pattern, mat, rgIdx, ids],
  );
  const d = useDeferredValue(inputs);
  const sel = SELECTIVITIES[d.selIdx];
  const prefix = useMemo(() => matchPrefix(sel, d.pattern), [sel, d.pattern]);
  const q: Query = useMemo(
    () => ({ projected: d.projected, filter: d.filter, late: d.late, rowGroupRows: d.rowGroupRows, explicitIds: d.explicitIds }),
    [d],
  );
  const sim = useMemo(() => simulate(q, prefix), [q, prefix]);
  const page = useMemo(() => firstMatchPage(prefix), [prefix]);
  const nsmCls = useMemo(() => nsmPageClasses(page, q, prefix), [page, q, prefix]);
  const paxCls = useMemo(() => paxPageClasses(page, q, prefix), [page, q, prefix]);
  const nsmTouched = useMemo(() => touchedLines(nsmCls), [nsmCls]);
  const paxTouched = useMemo(() => touchedLines(paxCls), [paxCls]);
  const pageMatches = prefix[Math.min(sim.n, (page + 1) * ROWS_PER_PAGE)] - prefix[page * ROWS_PER_PAGE];

  const fName = COLUMNS[q.filter].name;
  const othersNames = sim.others.map((c) => COLUMNS[c].name);

  /* ------------------------------------------------------------ geometry */
  const W = 680;
  const LX = 104;
  const TW = W - LX - 6;
  const T0 = 346;
  const TP = 11;
  const tracksEnd = T0 + 24 + COLUMNS.length * TP;
  const FY = tracksEnd + 14;
  const H = FY + 34;
  const xRow = (r: number) => LX + (r / sim.n) * TW;
  const xByte = (b: number) => LX + (b / Math.max(1, sim.col.totalBytes)) * TW;

  const chipStyle = (i: number): CSSProperties => {
    const isF = i === filter;
    const isP = projected[i];
    return {
      fontSize: '0.75rem',
      padding: '0.15rem 0.45rem',
      borderWidth: 2,
      borderStyle: 'solid',
      borderColor: isF ? 'var(--viz-1)' : isP ? 'var(--viz-2)' : 'var(--viz-border)',
      boxShadow: isF && isP ? 'inset 0 -3px 0 var(--viz-2)' : undefined,
      fontWeight: isP || isF ? 600 : 400,
    };
  };

  const early = !q.late;
  const noteBytes = (
    <>
      <strong>
        NSM and PAX both read all {fmtNum(sim.pages)} pages ({fmtBytes(sim.nsm.bytesRead)}); the column chunks read {fmtBytes(sim.col.bytesRead)}.
      </strong>{' '}
      A page is read whole, so rearranging bytes inside it (PAX) changes nothing on disk; only storing columns apart lets the scan leave the other{' '}
      {COLUMNS.length - sim.k} columns unread.{' '}
    </>
  );
  const noteCache = (
    <>
      In memory, NSM touches {fmtNum(sim.nsm.cacheLines)} cache lines because every tuple is deformed from its first attribute through <code>{fName}</code>
      {q.filter > 0 ? ` (walking past ${q.filter} other attribute${q.filter === 1 ? '' : 's'})` : ''}; PAX touches {fmtNum(sim.pax.cacheLines)}.{' '}
    </>
  );
  const noteMat = early ? (
    sim.k > 1 ? (
      <>
        Early materialization stitches all {fmtNum(sim.n)} rows from {sim.k} columns — {fmtNum(sim.col.stitches)} positional lookups — before the filter keeps{' '}
        {fmtNum(sim.matches)} of them.
      </>
    ) : (
      <>Only the filter column is needed, so there is nothing to stitch and early and late materialization do the same work.</>
    )
  ) : sim.others.length === 0 ? (
    <>Only the filter column is needed, so there is nothing to stitch and early and late materialization do the same work.</>
  ) : (
    <>
      Late materialization scans <code>{fName}</code> alone, keeps {fmtNum(sim.matches)} positions, and fetches {othersNames.length === 1 ? <code>{othersNames[0]}</code> : `the other ${othersNames.length} columns`}{' '}
      only there: {fmtNum(sim.col.stitches)} lookups.{' '}
      {sim.otherGranulesRead === sim.otherGranules
        ? `But every one of their ${fmtNum(sim.otherGranules)} granules still holds a match, so no bytes are saved — the gain is CPU and cache, not I/O.`
        : `Only ${fmtNum(sim.otherGranulesRead)} of their ${fmtNum(sim.otherGranules)} granules hold a match, so the rest are never read.`}
      {sim.matches >= 0.3 * sim.n
        ? ' Most rows match, so there is little left to save — and the position list and second pass over the columns, which the lab does not count, can make early materialization the faster choice.'
        : ''}
    </>
  );

  return (
    <VizPanel
      title="One table, three layouts, one scan"
      subtitle="1,000,000 rows of a 20-column orders table stored as row-major NSM pages, PAX pages and column chunks in row groups. Pick the columns the query needs, how many rows match, and when rows are stitched back together."
      controls={
        <>
          <Choice
            label="Query"
            value={preset}
            onChange={(v) => (v === 'custom' ? custom() : applyPreset(v as PresetKey))}
            options={[
              ...(Object.keys(PRESETS) as PresetKey[]).map((k) => ({ value: k, label: PRESETS[k].label })),
              { value: 'custom', label: 'Custom' },
            ]}
          />
          <Choice
            label="WHERE column"
            value={String(filter)}
            onChange={(v) => {
              setFilter(Number(v));
              custom();
            }}
            options={COLUMNS.map((c, i) => ({ value: String(i), label: `${c.name} (${c.width} B)` }))}
          />
          <Slider
            label="Rows matching"
            min={0}
            max={SELECTIVITIES.length - 1}
            value={selIdx}
            onChange={(v) => {
              setSelIdx(v);
              custom();
            }}
            format={(v) => fmtSel(SELECTIVITIES[v])}
          />
          <Segmented
            label="Matches are"
            value={pattern}
            onChange={(v) => {
              setPattern(v);
              custom();
            }}
            options={[
              { value: 'scattered', label: 'Scattered' },
              { value: 'clustered', label: 'Clustered' },
            ]}
          />
          <Segmented
            label="Materialization"
            value={mat}
            onChange={setMat}
            options={[
              { value: 'early', label: 'Early' },
              { value: 'late', label: 'Late' },
            ]}
          />
          <Slider label="Row group size (rows)" min={0} max={RG_SIZES.length - 1} value={rgIdx} onChange={setRgIdx} format={(v) => fmtRg(RG_SIZES[v])} />
          <Check label="Store a row id with every value" checked={ids} onChange={setIds} />
        </>
      }
      legend={
        <Legend
          items={[
            { label: 'WHERE column value, read', color: 'var(--viz-1)' },
            { label: 'SELECT column value, read', color: 'var(--viz-2)' },
            { label: 'Read only to reach a later attribute', color: 'var(--viz-3)' },
            { label: 'Cache line touched', color: 'var(--viz-4)' },
            { label: 'Stored, not read', color: 'var(--viz-grid)' },
            { label: 'Headers, line pointers, padding', color: 'var(--viz-axis)' },
          ]}
        />
      }
      stats={
        <Stats
          items={[
            { label: 'Rows matching', value: fmtNum(sim.matches), hint: `${fmtSel(sel)} of ${fmtNum(sim.n)} rows, ${d.pattern}` },
            { label: 'Columns needed', value: `${sim.k} of ${COLUMNS.length}`, hint: 'The SELECT columns plus the WHERE column.' },
            { label: 'Column-chunk bytes vs NSM', value: pct(sim.col.bytesRead, sim.nsm.bytesRead), hint: 'Bytes the column-chunk layout reads, as a share of the pages NSM reads.' },
            { label: 'PAX cache lines vs NSM', value: pct(sim.pax.cacheLines, sim.nsm.cacheLines), hint: 'Same pages, different placement inside them.' },
            {
              label: 'Other columns’ granules read',
              value: sim.others.length === 0 ? '—' : `${fmtNum(sim.otherGranulesRead)} / ${fmtNum(sim.otherGranules)}`,
              hint: 'Granules of the needed columns other than the WHERE column. Early materialization reads them all.',
            },
          ]}
        />
      }
      note={
        <Note>
          {noteBytes}
          {noteCache}
          {noteMat}
          {ids ? ` Row ids add ${fmtBytes(sim.col.totalBytes - sim.n * VALUE_BYTES)} to the column file and tell the reader nothing a position does not.` : ''}
        </Note>
      }
      table={
        <table className="viz-table">
          <thead>
            <tr>
              <th>Column</th>
              <th>Type</th>
              <th>Role</th>
              <th>Granules read</th>
              <th>Bytes read</th>
              <th>Cache lines</th>
            </tr>
          </thead>
          <tbody>
            {sim.detail.map((c) => (
              <tr key={c.col}>
                <td>{COLUMNS[c.col].name}</td>
                <td>
                  {COLUMNS[c.col].type}, {COLUMNS[c.col].width} B
                </td>
                <td>{c.role === 'unused' ? '—' : c.role === 'filter' ? 'WHERE' : 'SELECT'}</td>
                <td>
                  {fmtNum(c.granulesRead)} / {fmtNum(c.granules)}
                </td>
                <td>{fmtBytes(c.bytesRead)}</td>
                <td>{fmtNum(c.cacheLines)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      }
    >
      <div className="viz-controls" style={{ gap: '0.35rem', marginBottom: '0.6rem' }} role="group" aria-label="Columns the query selects">
        <span style={{ fontSize: '0.75rem', color: 'var(--viz-ink-2)', marginRight: '0.25rem' }}>SELECT columns:</span>
        {COLUMNS.map((c, i) => (
          <button
            key={c.name}
            type="button"
            aria-pressed={projected[i]}
            data-col={c.name}
            title={`${c.name}: ${c.type}, ${c.width} bytes${i === filter ? ' — the WHERE column' : ''}`}
            style={chipStyle(i)}
            onClick={() => {
              setProjected((p) => p.map((v, j) => (j === i ? !v : v)));
              custom();
            }}
          >
            {c.name}
          </button>
        ))}
        <Button
          onClick={() => {
            setProjected(COLUMNS.map(() => true));
            custom();
          }}
        >
          All
        </Button>
        <Button
          onClick={() => {
            setProjected(COLUMNS.map(() => false));
            custom();
          }}
        >
          None
        </Button>
      </div>

      <svg
        viewBox={`0 0 ${W} ${H}`}
        width={W}
        height={H}
        style={{ minWidth: 560 }}
        role="img"
        aria-label={`NSM reads ${fmtBytes(sim.nsm.bytesRead)} and touches ${fmtNum(sim.nsm.cacheLines)} cache lines; PAX reads the same pages and touches ${fmtNum(sim.pax.cacheLines)}; column chunks read ${fmtBytes(sim.col.bytesRead)} in ${fmtNum(sim.col.ranges)} byte ranges.`}
      >
        <PageMap
          y={0}
          title="NSM row-major page"
          cls={nsmCls}
          lines={nsmTouched}
          sub={`page ${fmtNum(page + 1)} of ${fmtNum(sim.pages)} · ${pageMatches} matching row${pageMatches === 1 ? '' : 's'} · ${nsmTouched.length} of 128 cache lines touched`}
        />
        <PageMap
          y={150}
          title="PAX page — the same rows, grouped by column"
          cls={paxCls}
          lines={paxTouched}
          sub={`page ${fmtNum(page + 1)} · ${pageMatches} matching row${pageMatches === 1 ? '' : 's'} · ${paxTouched.length} of 128 cache lines touched`}
        />
        <text x={0} y={313} fontSize={10} fill="var(--viz-ink-2)">
          Each strip is one 64-byte cache line; 128 strips make one 8 KB page. Shown: the first page with a matching row.
        </text>

        <text x={0} y={T0 - 8} fontSize={12} fill="var(--viz-ink)" fontWeight={600}>
          Column chunks: {fmtNum(sim.rowGroups)} row group{sim.rowGroups === 1 ? '' : 's'} of {fmtNum(Math.min(q.rowGroupRows, sim.n))} rows, granules of 8,192 rows
        </text>
        <text x={LX} y={T0 + 12} fontSize={10} fill="var(--viz-ink-2)">
          row 0
        </text>
        <text x={W - 6} y={T0 + 12} textAnchor="end" fontSize={10} fill="var(--viz-ink-2)">
          row {fmtNum(sim.n)}
        </text>
        {sim.detail.map((c, i) => {
          const y = T0 + 20 + i * TP;
          return (
            <g key={c.col}>
              <text x={LX - 6} y={y + 8} textAnchor="end" fontSize={9.5} fill={c.role === 'unused' ? 'var(--viz-ink-2)' : 'var(--viz-ink)'} fontWeight={c.role === 'unused' ? 400 : 600}>
                {COLUMNS[c.col].name}
              </text>
              <rect x={LX} y={y} width={TW} height={TP - 3} fill="var(--viz-grid)" />
              {c.tracks.map(([a, b], j) => (
                <rect key={j} x={xRow(a)} y={y} width={Math.max(0.8, xRow(b) - xRow(a))} height={TP - 3} fill={ROLE_FILL[c.role]} />
              ))}
            </g>
          );
        })}
        {Array.from({ length: sim.rowGroups - 1 }, (_, g) => (
          <line
            key={g}
            x1={xRow((g + 1) * q.rowGroupRows)}
            x2={xRow((g + 1) * q.rowGroupRows)}
            y1={T0 + 17}
            y2={tracksEnd - 1}
            stroke="var(--viz-ink-muted)"
            strokeWidth={sim.rowGroups > 40 ? 0.4 : 1}
            strokeDasharray={sim.rowGroups > 40 ? undefined : '3 2'}
          />
        ))}

        <text x={LX - 6} y={FY + 9} textAnchor="end" fontSize={9} fill="var(--viz-ink)">
          file, in order
        </text>
        <rect x={LX} y={FY} width={TW} height={11} fill="var(--viz-grid)" />
        {sim.fileRuns.map((r, i) => (
          <rect key={i} x={xByte(r.start)} y={FY} width={Math.max(0.6, xByte(r.end) - xByte(r.start))} height={11} fill={ROLE_FILL[r.role]} />
        ))}
        <text x={LX} y={FY + 25} fontSize={10} fill="var(--viz-ink-2)">
          {fmtBytes(sim.col.bytesRead)} of {fmtBytes(sim.col.totalBytes)} read in {fmtNum(sim.col.ranges)} separate byte range{sim.col.ranges === 1 ? '' : 's'}
        </text>
      </svg>

      <table className="viz-table" data-testid="layout-metrics">
        <thead>
          <tr>
            <th>Layout</th>
            <th>Bytes read</th>
            <th>Cache lines touched</th>
            <th>Byte ranges (seeks)</th>
            <th>Positional stitches</th>
          </tr>
        </thead>
        <tbody>
          {(
            [
              ['NSM row pages', sim.nsm],
              ['PAX pages', sim.pax],
              ['Column chunks', sim.col],
            ] as const
          ).map(([label, m]) => (
            <tr key={label} data-layout={label}>
              <td>{label}</td>
              <td>
                {fmtBytes(m.bytesRead)} of {fmtBytes(m.totalBytes)}
              </td>
              <td>{fmtNum(m.cacheLines)}</td>
              <td>{fmtNum(m.ranges)}</td>
              <td>{fmtNum(m.stitches)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </VizPanel>
  );
}
