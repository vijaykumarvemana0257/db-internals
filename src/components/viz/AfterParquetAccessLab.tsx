import { useMemo, useState } from 'react';
import { VizPanel, Segmented, Choice, Check, Slider, Legend, Stats, Note, makeRng, fmtBytes, fmtNum } from './Viz';

/**
 * One dataset, three physical layouts, three access patterns.
 *
 * What the model follows (each rule is taken from the format's spec or its reference implementation):
 * - Parquet: row groups hold one column chunk per column; pages close at 1 MB or 20,000 rows (parquet-java,
 *   parquet-rs and Arrow C++ 23.0+ defaults). The Thrift footer carries a ColumnChunk entry for every (row group x column) and must be
 *   decoded from the start. Arrow C++ reads the last 64 KB and issues a second read when the footer is larger.
 *   With the page index (OffsetIndex) a reader fetches only the pages holding the requested rows; without it,
 *   the whole column chunk of every touched row group.
 * - Lance (file format 2.1+): no row groups, ~8 MB pages per column, one protobuf descriptor per column found through
 *   a column-metadata offset table (16 bytes per column). Small values use mini-blocks (at most 4,096 values, under
 *   32 KB, 2 bytes of lookup metadata each, loaded at initialisation). Values over 256 bytes use full-zip: fixed-width
 *   values are located by arithmetic (one read per row); variable-width values through a per-row repetition index
 *   (two reads per row). The dataset reader takes the metadata-index path, decoding only the projected descriptors
 *   with the schema supplied by the manifest, when a query projects fewer than a quarter of the file's columns
 *   (projected x 4 < columns, lance fragment.rs); otherwise, and on every full scan, it reads all descriptors.
 * - Vortex (default WriteStrategyBuilder): rows repartitioned into 8,192-row blocks, coalesced into segments of at least
 *   1 MB uncompressed whose row count is a multiple of min(8,192, ceil(1 MB / value size)) for fixed-width types and
 *   8,192 for strings. A reader never reads less than one segment. The footer's segment map is decoded eagerly; the
 *   FlatBuffer layout tree is read lazily, so only projected columns' layout nodes are charged.
 *
 * Model assumptions (labelled on screen too):
 * - Every format stores the same bytes per value (id 4, text 600, embedding 3,072, feature 4), so differences come
 *   from layout alone. No dictionary pages, no nulls, no filters.
 * - Metadata message sizes are estimates, not measurements: Parquet ~100 B per ColumnChunk (in line with published
 *   parquet-rs measurements), Lance ~60 B per page descriptor, Vortex 16 B per segment-map entry.
 * - Byte ranges are counted before any coalescing. File sizes are known from the table catalog (no HEAD requests).
 *   Table-level metadata (manifests) is excluded for every format.
 * - Decode time = metadata bytes decoded / metadata speed + value bytes produced / data speed. Every fetched unit
 *   (page, mini-block, segment) is decoded whole; encodings with take kernels could do less.
 */

export const N_ROWS = 1_000_000;
export const TAKE_ROWS = 100;
const KIB = 1024;
const MIB = 1024 * 1024;

export type Access = 'scan' | 'project' | 'take';
export type RowGroupPolicy = 'rows1m' | 'bytes128m' | 'rows10k';
export type FormatId = 'parquet' | 'lance' | 'vortex';
export type Kind = 'id' | 'text' | 'embedding' | 'feature';

export const KINDS: Kind[] = ['id', 'text', 'embedding', 'feature'];
export const FORMATS: FormatId[] = ['parquet', 'lance', 'vortex'];
export const FORMAT_NAME: Record<FormatId, string> = { parquet: 'Parquet', lance: 'Lance', vortex: 'Vortex' };

/** raw = decoded bytes per value; stored = bytes on disk per value (identical for every format). */
export const KIND_INFO: Record<Kind, { name: string; raw: number; stored: number; fixedWidth: boolean }> = {
  id: { name: 'id int64', raw: 8, stored: 4, fixedWidth: true },
  text: { name: 'text ~1.2 KB', raw: 1200, stored: 600, fixedWidth: false },
  embedding: { name: 'embedding 768×f32', raw: 3072, stored: 3072, fixedWidth: true },
  feature: { name: 'feature f32', raw: 4, stored: 4, fixedWidth: true },
};

export type Params = {
  access: Access;
  columns: number;
  files: number;
  rowGroup: RowGroupPolicy;
  pageIndex: boolean;
  metaMBps: number;
  dataMBps: number;
};

export const DEFAULTS: Params = { access: 'take', columns: 12, files: 1, rowGroup: 'rows1m', pageIndex: true, metaMBps: 100, dataMBps: 1000 };

export function projection(access: Access, columns: number): { kind: Kind; count: number }[] {
  if (access === 'scan')
    return [
      { kind: 'id', count: 1 },
      { kind: 'text', count: 1 },
      { kind: 'embedding', count: 1 },
      { kind: 'feature', count: columns - 3 },
    ];
  if (access === 'project')
    return [
      { kind: 'id', count: 1 },
      { kind: 'feature', count: 2 },
    ];
  return [
    { kind: 'id', count: 1 },
    { kind: 'text', count: 1 },
    { kind: 'embedding', count: 1 },
  ];
}

const allColumns = (columns: number) => projection('scan', columns);
const ceilDiv = (a: number, b: number) => Math.ceil(a / b);

/** 100 distinct row ids, deterministic. */
export function sampleRows(count = TAKE_ROWS, seed = 7): number[] {
  const rng = makeRng(seed);
  const set = new Set<number>();
  while (set.size < count) set.add(Math.floor(rng() * N_ROWS) % N_ROWS);
  return [...set].sort((a, b) => a - b);
}

/* ---------------------------------------------------------------- Parquet */

export const PQ_PAGE_ROW_LIMIT = 20_000;
export const PQ_FOOTER_TAIL_READ = 64 * KIB;
const PQ_FOOTER_FIXED = 100;
const PQ_SCHEMA_ELEMENT = 20;
const PQ_ROW_GROUP = 30;
export const PQ_COLUMN_CHUNK = 100;
const PQ_PAGE_INDEX_REFS = 12;
const PQ_PAGE_INDEX_ENTRY = 24;

export const rowStoredBytes = (columns: number) => KIND_INFO.id.stored + KIND_INFO.text.stored + KIND_INFO.embedding.stored + KIND_INFO.feature.stored * (columns - 3);

export function parquetRowGroupRows(policy: RowGroupPolicy, columns: number) {
  if (policy === 'rows1m') return 1_048_576;
  if (policy === 'rows10k') return 10_000;
  return Math.max(1, Math.floor((128 * MIB) / rowStoredBytes(columns)));
}
export const parquetPageRows = (k: Kind) => Math.max(1, Math.min(PQ_PAGE_ROW_LIMIT, Math.floor(MIB / KIND_INFO[k].stored)));

function parquetFile(p: Params, n: number) {
  const rgRows = Math.min(parquetRowGroupRows(p.rowGroup, p.columns), n);
  const rgCount = ceilDiv(n, rgRows);
  const rgLen = (rg: number) => Math.min(rgRows, n - rg * rgRows);
  const pagesInFile = (k: Kind) => {
    const pr = parquetPageRows(k);
    const full = Math.floor(n / rgRows);
    const rest = n - full * rgRows;
    return full * ceilDiv(rgRows, pr) + (rest > 0 ? ceilDiv(rest, pr) : 0);
  };
  const footer = PQ_FOOTER_FIXED + p.columns * PQ_SCHEMA_ELEMENT + rgCount * (PQ_ROW_GROUP + p.columns * (PQ_COLUMN_CHUNK + (p.pageIndex ? PQ_PAGE_INDEX_REFS : 0)));
  const pageIndexBytes = p.pageIndex ? allColumns(p.columns).reduce((s, c) => s + c.count * pagesInFile(c.kind) * PQ_PAGE_INDEX_ENTRY, 0) : 0;
  return { rgRows, rgCount, rgLen, footer, pageIndexBytes, pagesInFile };
}

/* ------------------------------------------------------------------ Lance */

export const LANCE_PAGE_BYTES = 8 * MIB;
export const LANCE_MINIBLOCK_MAX_VALUES = 4096;
export const LANCE_FULLZIP_CUTOFF = 256;
export const LANCE_TAIL_READ = 64 * KIB;
const LANCE_FOOTER = 40;
const LANCE_CMO_ENTRY = 16;
const LANCE_DESCRIPTOR_FIXED = 8;
export const LANCE_PAGE_DESCRIPTOR = 60;
const LANCE_REP_INDEX_READ = 8;

export const lancePageRows = (k: Kind) => Math.max(1, Math.floor(LANCE_PAGE_BYTES / KIND_INFO[k].stored));
export const lanceLayout = (k: Kind): 'miniblock' | 'fullzip' => (KIND_INFO[k].raw > LANCE_FULLZIP_CUTOFF ? 'fullzip' : 'miniblock');
/** Power-of-two values, under 32 KB, at most LANCE_MINIBLOCK_MAX_VALUES. */
export function lanceMiniBlockRows(k: Kind) {
  let v = LANCE_MINIBLOCK_MAX_VALUES;
  while (v > 1 && v * KIND_INFO[k].stored >= 32 * KIB) v /= 2;
  return v;
}

function lanceFile(p: Params, n: number) {
  const pages = (k: Kind) => ceilDiv(n, lancePageRows(k));
  const descriptor = (k: Kind) => LANCE_DESCRIPTOR_FIXED + pages(k) * LANCE_PAGE_DESCRIPTOR;
  const descriptors = allColumns(p.columns).reduce((s, c) => s + c.count * descriptor(c.kind), 0);
  const cmo = p.columns * LANCE_CMO_ENTRY;
  const tailMeta = descriptors + cmo + LANCE_CMO_ENTRY + LANCE_FOOTER;
  return { pages, descriptor, descriptors, cmo, tailMeta };
}

/* ----------------------------------------------------------------- Vortex */

export const VX_ROW_BLOCK = 8192;
export const VX_SEGMENT_TARGET = MIB;
export const VX_INITIAL_READ = 65_535;
export const VX_SEGMENT_SPEC = 16;
const VX_LAYOUT_COLUMN = 64;
const VX_LAYOUT_CHUNK = 28;
const VX_DTYPE_FIELD = 24;
const VX_STATS_FIELD = 40;
const VX_POSTSCRIPT = 80;

export function vortexSegmentRows(k: Kind) {
  const { raw, fixedWidth } = KIND_INFO[k];
  const elem = fixedWidth ? raw : raw + 16; // strings are VarBinView: a 16-byte view plus the data
  const len = fixedWidth ? Math.max(1, Math.min(VX_ROW_BLOCK, Math.ceil(VX_SEGMENT_TARGET / raw))) : VX_ROW_BLOCK;
  return len * Math.max(1, Math.ceil(VX_SEGMENT_TARGET / (len * elem)));
}

function vortexFile(p: Params, n: number) {
  const segments = (k: Kind) => ceilDiv(n, vortexSegmentRows(k));
  const cols = allColumns(p.columns);
  const segmentMap = cols.reduce((s, c) => s + c.count * (segments(c.kind) + 1) * VX_SEGMENT_SPEC, 0); // +1: zone-map table
  const layoutFor = (k: Kind) => VX_LAYOUT_COLUMN + (segments(k) + 1) * VX_LAYOUT_CHUNK;
  const layout = cols.reduce((s, c) => s + c.count * layoutFor(c.kind), 0);
  const dtype = 16 + p.columns * VX_DTYPE_FIELD;
  const stats = p.columns * VX_STATS_FIELD;
  // Physical order at the tail: dtype, layout, statistics, footer (segment map), postscript + 8-byte EOF.
  const pieces = [dtype, layout, stats, segmentMap];
  return { segments, segmentMap, layoutFor, layout, dtype, stats, pieces, tailMeta: dtype + layout + stats + segmentMap + VX_POSTSCRIPT + 8 };
}

/* --------------------------------------------------------------- simulate */

export type KindCost = { kind: Kind; count: number; ranges: number; bytes: number; decoded: number; unit: string };
export type Cost = {
  format: FormatId;
  filesOpened: number;
  metaRanges: number;
  metaFetched: number;
  metaDecoded: number;
  dataRanges: number;
  dataBytes: number;
  dataDecoded: number;
  perKind: KindCost[];
  decodeSeconds: number;
  metaSeconds: number;
  dataSeconds: number;
};

type Acc = { metaRanges: number; metaFetched: number; metaDecoded: number; perKind: Map<Kind, KindCost> };

const newAcc = (): Acc => ({ metaRanges: 0, metaFetched: 0, metaDecoded: 0, perKind: new Map() });
function addKind(acc: Acc, kind: Kind, count: number, unit: string, ranges: number, bytes: number, decoded: number) {
  const cur = acc.perKind.get(kind) ?? { kind, count, ranges: 0, bytes: 0, decoded: 0, unit };
  cur.ranges += ranges;
  cur.bytes += bytes;
  cur.decoded += decoded;
  cur.unit = unit;
  acc.perKind.set(kind, cur);
}

function groupByFile(rows: number[], n: number) {
  const m = new Map<number, number[]>();
  for (const r of rows) {
    const f = Math.floor(r / n);
    const list = m.get(f) ?? [];
    list.push(r - f * n);
    m.set(f, list);
  }
  return m;
}

function dataBytesOfFile(p: Params, n: number) {
  return allColumns(p.columns).reduce((s, c) => s + c.count * n * KIND_INFO[c.kind].stored, 0);
}

function parquetCost(p: Params, rows: number[]): Acc & { filesOpened: number } {
  const n = N_ROWS / p.files;
  const f = parquetFile(p, n);
  const fileSize = dataBytesOfFile(p, n) + f.footer + f.pageIndexBytes + 12;
  const acc = newAcc();
  const openFooter = () => {
    const tail = Math.min(fileSize, PQ_FOOTER_TAIL_READ);
    acc.metaRanges += 1;
    acc.metaFetched += tail;
    if (f.footer + 8 > tail) {
      acc.metaRanges += 1;
      acc.metaFetched += f.footer;
    }
    acc.metaDecoded += f.footer;
  };
  const proj = projection(p.access, p.columns);
  if (p.access !== 'take') {
    for (let i = 0; i < p.files; i++) openFooter();
    for (const c of proj) {
      const info = KIND_INFO[c.kind];
      addKind(acc, c.kind, c.count, 'column chunk (one per row group)', p.files * c.count * f.rgCount, p.files * c.count * n * info.stored, p.files * c.count * n * info.raw);
    }
    return { ...acc, filesOpened: p.files };
  }
  const byFile = groupByFile(rows, n);
  for (const local of byFile.values()) {
    openFooter();
    const touchedRgs = new Set(local.map((r) => Math.floor(r / f.rgRows)));
    if (p.pageIndex) {
      acc.metaRanges += 1;
      acc.metaFetched += f.pageIndexBytes;
      for (const c of proj) {
        const pr = parquetPageRows(c.kind);
        for (const rg of touchedRgs) acc.metaDecoded += ceilDiv(f.rgLen(rg), pr) * PQ_PAGE_INDEX_ENTRY;
      }
    }
    for (const c of proj) {
      const info = KIND_INFO[c.kind];
      if (p.pageIndex) {
        const pr = parquetPageRows(c.kind);
        const pages = new Set(local.map((r) => `${Math.floor(r / f.rgRows)}:${Math.floor((r % f.rgRows) / pr)}`));
        let rowsRead = 0;
        for (const key of pages) {
          const [rg, pg] = key.split(':').map(Number);
          rowsRead += Math.min(pr, f.rgLen(rg) - pg * pr);
        }
        addKind(acc, c.kind, 1, `page (≤ ${fmtNum(pr)} rows)`, pages.size, rowsRead * info.stored, rowsRead * info.raw);
      } else {
        let rowsRead = 0;
        for (const rg of touchedRgs) rowsRead += f.rgLen(rg);
        addKind(acc, c.kind, 1, 'whole column chunk', touchedRgs.size, rowsRead * info.stored, rowsRead * info.raw);
      }
    }
  }
  return { ...acc, filesOpened: byFile.size };
}

function lanceCost(p: Params, rows: number[]): Acc & { filesOpened: number } {
  const n = N_ROWS / p.files;
  const f = lanceFile(p, n);
  const fileSize = dataBytesOfFile(p, n) + f.tailMeta;
  const acc = newAcc();
  const proj = projection(p.access, p.columns);
  const tail = Math.min(fileSize, LANCE_TAIL_READ);
  const openAll = () => {
    acc.metaRanges += 1;
    acc.metaFetched += tail;
    if (f.tailMeta > tail) {
      acc.metaRanges += 1;
      acc.metaFetched += f.tailMeta - tail;
    }
    acc.metaDecoded += f.descriptors + f.cmo;
  };
  const openIndex = () => {
    acc.metaRanges += 1;
    acc.metaFetched += tail;
    const cmoAndFooter = f.cmo + LANCE_CMO_ENTRY + LANCE_FOOTER;
    if (cmoAndFooter > tail) {
      acc.metaRanges += 1;
      acc.metaFetched += cmoAndFooter - tail;
    }
    acc.metaDecoded += f.cmo;
    for (const c of proj) {
      const d = f.descriptor(c.kind);
      // Projected columns come first in the file, so their descriptors are the farthest from the tail.
      if (f.tailMeta > tail) {
        acc.metaRanges += c.count;
        acc.metaFetched += c.count * d;
      }
      acc.metaDecoded += c.count * d;
    }
  };
  // Lance's dataset reader opens the lightweight metadata index only when the projection is under a quarter of the
  // file's columns; otherwise it reads every column descriptor.
  const projectedColumns = proj.reduce((s, c) => s + c.count, 0);
  const open = projectedColumns * 4 < p.columns ? openIndex : openAll;
  if (p.access === 'scan' || p.access === 'project') {
    for (let i = 0; i < p.files; i++) open();
    for (const c of proj) {
      const info = KIND_INFO[c.kind];
      addKind(acc, c.kind, c.count, `page (~8 MB, ${fmtNum(Math.min(n, lancePageRows(c.kind)))} rows)`, p.files * c.count * f.pages(c.kind), p.files * c.count * n * info.stored, p.files * c.count * n * info.raw);
    }
    return { ...acc, filesOpened: p.files };
  }
  const byFile = groupByFile(rows, n);
  for (const local of byFile.values()) {
    open();
    for (const c of proj) {
      const info = KIND_INFO[c.kind];
      const pr = lancePageRows(c.kind);
      if (lanceLayout(c.kind) === 'miniblock') {
        const mb = lanceMiniBlockRows(c.kind);
        const pages = new Set(local.map((r) => Math.floor(r / pr)));
        for (const pg of pages) {
          const pageLen = Math.min(pr, n - pg * pr);
          acc.metaRanges += 1; // mini-block lookup buffer, loaded into the search cache
          acc.metaFetched += 2 * ceilDiv(pageLen, mb);
          acc.metaDecoded += 2 * ceilDiv(pageLen, mb);
        }
        const blocks = new Set(local.map((r) => `${Math.floor(r / pr)}:${Math.floor((r % pr) / mb)}`));
        let rowsRead = 0;
        for (const key of blocks) {
          const [pg, b] = key.split(':').map(Number);
          const pageLen = Math.min(pr, n - pg * pr);
          rowsRead += Math.min(mb, pageLen - b * mb);
        }
        addKind(acc, c.kind, 1, `mini-block (≤ ${fmtNum(mb)} values)`, blocks.size, rowsRead * info.stored, rowsRead * info.raw);
      } else if (info.fixedWidth) {
        addKind(acc, c.kind, 1, 'one value at row × 3,072 bytes', local.length, local.length * info.stored, local.length * info.raw);
      } else {
        addKind(acc, c.kind, 1, 'repetition-index entry, then the value', 2 * local.length, local.length * (LANCE_REP_INDEX_READ + info.stored), local.length * info.raw);
      }
    }
  }
  return { ...acc, filesOpened: byFile.size };
}

function vortexCost(p: Params, rows: number[]): Acc & { filesOpened: number } {
  const n = N_ROWS / p.files;
  const f = vortexFile(p, n);
  const fileSize = dataBytesOfFile(p, n) + f.tailMeta;
  const acc = newAcc();
  const proj = projection(p.access, p.columns);
  const open = () => {
    const initial = Math.min(fileSize, VX_INITIAL_READ);
    acc.metaRanges += 1;
    acc.metaFetched += initial;
    // A metadata segment that does not fit inside the initial read is fetched by its own targeted read.
    let end = f.tailMeta - VX_POSTSCRIPT - 8; // bytes before the postscript, counted from the start of the dtype
    for (let i = f.pieces.length - 1; i >= 0; i--) {
      const start = end - f.pieces[i];
      const distanceFromEof = f.tailMeta - start;
      if (distanceFromEof > initial) {
        acc.metaRanges += 1;
        acc.metaFetched += f.pieces[i];
      }
      end = start;
    }
    acc.metaDecoded += f.segmentMap + f.dtype + VX_POSTSCRIPT;
    for (const c of proj) acc.metaDecoded += c.count * f.layoutFor(c.kind);
  };
  if (p.access !== 'take') {
    for (let i = 0; i < p.files; i++) open();
    for (const c of proj) {
      const info = KIND_INFO[c.kind];
      addKind(acc, c.kind, c.count, `segment (${fmtNum(Math.min(n, vortexSegmentRows(c.kind)))} rows)`, p.files * c.count * f.segments(c.kind), p.files * c.count * n * info.stored, p.files * c.count * n * info.raw);
    }
    return { ...acc, filesOpened: p.files };
  }
  const byFile = groupByFile(rows, n);
  for (const local of byFile.values()) {
    open();
    for (const c of proj) {
      const info = KIND_INFO[c.kind];
      const sr = vortexSegmentRows(c.kind);
      const segs = new Set(local.map((r) => Math.floor(r / sr)));
      let rowsRead = 0;
      for (const s of segs) rowsRead += Math.min(sr, n - s * sr);
      addKind(acc, c.kind, 1, `segment (${fmtNum(Math.min(n, sr))} rows)`, segs.size, rowsRead * info.stored, rowsRead * info.raw);
    }
  }
  return { ...acc, filesOpened: byFile.size };
}

export function simulate(p: Params, rows: number[] = sampleRows()): Record<FormatId, Cost> {
  const raw = { parquet: parquetCost(p, rows), lance: lanceCost(p, rows), vortex: vortexCost(p, rows) };
  const out = {} as Record<FormatId, Cost>;
  for (const id of FORMATS) {
    const a = raw[id];
    const perKind = [...a.perKind.values()];
    const dataRanges = perKind.reduce((s, k) => s + k.ranges, 0);
    const dataBytes = perKind.reduce((s, k) => s + k.bytes, 0);
    const dataDecoded = perKind.reduce((s, k) => s + k.decoded, 0);
    const metaSeconds = a.metaDecoded / (p.metaMBps * MIB);
    const dataSeconds = dataDecoded / (p.dataMBps * MIB);
    out[id] = { format: id, filesOpened: a.filesOpened, metaRanges: a.metaRanges, metaFetched: a.metaFetched, metaDecoded: a.metaDecoded, dataRanges, dataBytes, dataDecoded, perKind, metaSeconds, dataSeconds, decodeSeconds: metaSeconds + dataSeconds };
  }
  return out;
}

/* ------------------------------------------------------------ file picture */

export type Band = {
  kind: Kind;
  projected: number; // how many of this band's columns are read
  total: number;
  groupEdges: number[]; // thick boundaries (Parquet row groups)
  unitRows: number; // typical rows per unit, for tick density
  unitCount: number;
  unitLabel: string;
  read: [number, number][]; // local row intervals fetched
  bytes: number; // stored bytes this band's projected columns fetch from this file
  unitOf: (r: number) => [number, number]; // rows fetched to return local row r
  ticksIn: (a: number, b: number) => number[]; // unit boundaries inside [a, b]
};

function mergeIntervals(iv: [number, number][]) {
  const s = iv.slice().sort((a, b) => a[0] - b[0]);
  const out: [number, number][] = [];
  for (const [a, b] of s) {
    const last = out[out.length - 1];
    if (last && a <= last[1]) last[1] = Math.max(last[1], b);
    else out.push([a, b]);
  }
  return out;
}

const stepTicks = (origin: number, step: number, a: number, b: number, cap = 400) => {
  const out: number[] = [];
  for (let t = origin + Math.ceil((a - origin) / step) * step; t <= b && out.length < cap; t += step) out.push(t);
  return out;
};

/** The layout of one file (the first file a take touches, or file 0) and the row ranges each format fetches from it. */
export function filePicture(p: Params, format: FormatId, rows: number[] = sampleRows()) {
  const n = N_ROWS / p.files;
  const byFile = groupByFile(rows, n);
  const fileNo = p.access === 'take' ? Math.min(...byFile.keys()) : 0;
  const local = p.access === 'take' ? byFile.get(fileNo) ?? [] : [];
  const proj = projection(p.access, p.columns);
  const bands: Band[] = KINDS.map((kind) => {
    const info = KIND_INFO[kind];
    const total = kind === 'feature' ? p.columns - 3 : 1;
    const projected = proj.find((c) => c.kind === kind)?.count ?? 0;
    let groupEdges: number[] = [0, n];
    let unitRows = n;
    let unitLabel = '';
    let unitCount = 1;
    let unitOf = (_r: number): [number, number] => [0, n];
    let ticksIn = (_a: number, _b: number): number[] => [];
    if (format === 'parquet') {
      const f = parquetFile(p, n);
      const pr = parquetPageRows(kind);
      groupEdges = Array.from({ length: f.rgCount + 1 }, (_, i) => Math.min(n, i * f.rgRows));
      unitRows = Math.min(pr, f.rgRows);
      unitCount = f.pagesInFile(kind);
      unitLabel = `${fmtNum(f.rgCount)} row group${f.rgCount === 1 ? '' : 's'}; pages of ≤ ${fmtNum(unitRows)} rows`;
      unitOf = p.pageIndex
        ? (r) => {
            const rg = Math.floor(r / f.rgRows);
            const start = rg * f.rgRows + Math.floor((r % f.rgRows) / pr) * pr;
            return [start, Math.min(start + pr, rg * f.rgRows + f.rgLen(rg))];
          }
        : (r) => {
            const rg = Math.floor(r / f.rgRows);
            return [rg * f.rgRows, rg * f.rgRows + f.rgLen(rg)];
          };
      ticksIn = (a, b) => {
        const out: number[] = [];
        for (let rg = Math.floor(a / f.rgRows); rg * f.rgRows <= b && rg < f.rgCount && out.length < 400; rg++) {
          const start = rg * f.rgRows;
          for (const t of stepTicks(start, pr, Math.max(a, start), Math.min(b, start + f.rgLen(rg)))) out.push(t);
        }
        return out;
      };
    } else if (format === 'lance') {
      const pr = lancePageRows(kind);
      const pages = ceilDiv(n, pr);
      if (lanceLayout(kind) === 'miniblock') {
        const mb = lanceMiniBlockRows(kind);
        unitRows = mb;
        unitCount = ceilDiv(n, mb);
        unitLabel = `${fmtNum(pages)} page${pages === 1 ? '' : 's'}; mini-blocks of ${fmtNum(mb)} values`;
        unitOf = (r) => {
          const start = Math.floor(r / pr) * pr + Math.floor((r % pr) / mb) * mb;
          return [start, Math.min(n, start + mb)];
        };
        ticksIn = (a, b) => stepTicks(0, mb, a, Math.min(b, n));
      } else {
        unitRows = 1;
        unitCount = n;
        unitLabel = `${fmtNum(pages)} page${pages === 1 ? '' : 's'}; full-zip, ${info.fixedWidth ? 'offset = row × 3,072 B' : 'per-row offsets'}`;
        unitOf = (r) => [r, r + 1];
        ticksIn = (a, b) => stepTicks(0, pr, a, Math.min(b, n));
      }
    } else {
      const sr = vortexSegmentRows(kind);
      unitRows = Math.min(n, sr);
      unitCount = ceilDiv(n, sr);
      unitLabel = `${fmtNum(unitCount)} segment${unitCount === 1 ? '' : 's'} of ${fmtNum(unitRows)} rows`;
      unitOf = (r) => {
        const start = Math.floor(r / sr) * sr;
        return [start, Math.min(n, start + sr)];
      };
      ticksIn = (a, b) => stepTicks(0, sr, a, Math.min(b, n));
    }
    let read: [number, number][] = [];
    if (projected > 0) read = p.access === 'take' ? mergeIntervals(local.map(unitOf)) : [[0, n]];
    const bytes = read.reduce((s, [a, b]) => s + (b - a), 0) * info.stored * Math.max(projected, 0);
    return { kind, projected, total, groupEdges, unitRows, unitCount, unitLabel, read, bytes, unitOf, ticksIn };
  });
  return { fileNo, rowsInFile: n, localRows: local, bands };
}

/** A window of rows around the first requested row, for the zoom strip. */
export function zoomWindow(rowsInFile: number, localRows: number[], width = 4096): [number, number] {
  if (rowsInFile <= width || localRows.length === 0) return [0, rowsInFile];
  const c = localRows[0];
  const a = Math.max(0, Math.min(rowsInFile - width, c - width / 2));
  return [a, a + width];
}

/* -------------------------------------------------------------------- view */

const W = 700;
const LABEL_X = 8;
const PLOT_X0 = 128;
const PLOT_X1 = 478;
const BYTES_X = 552;
const TAIL_X = 562;
const BAND_H = 14;
const BAND_GAP = 5;
const FORMAT_GAP = 20;

const fmtSec = (s: number) => (s < 1e-3 ? `${(s * 1e6).toFixed(0)} µs` : s < 1 ? `${(s * 1e3).toFixed(s < 0.01 ? 1 : 0)} ms` : `${s.toFixed(2)} s`);
const bandName = (b: Band) => (b.kind !== 'feature' ? KIND_INFO[b.kind].name : b.projected > 0 && b.projected < b.total ? `features: ${fmtNum(b.projected)} of ${fmtNum(b.total)}` : `features ×${fmtNum(b.total)}`);

export default function AfterParquetAccessLab() {
  const [access, setAccess] = useState<Access>(DEFAULTS.access);
  const [columns, setColumns] = useState<number>(DEFAULTS.columns);
  const [files, setFiles] = useState<number>(DEFAULTS.files);
  const [rowGroup, setRowGroup] = useState<RowGroupPolicy>(DEFAULTS.rowGroup);
  const [pageIndex, setPageIndex] = useState(DEFAULTS.pageIndex);
  const [metaMBps, setMetaMBps] = useState(DEFAULTS.metaMBps);
  const [dataMBps, setDataMBps] = useState(DEFAULTS.dataMBps);

  const rows = useMemo(() => sampleRows(), []);
  const params: Params = { access, columns, files, rowGroup, pageIndex, metaMBps, dataMBps };
  const costs = useMemo(() => simulate(params, rows), [access, columns, files, rowGroup, pageIndex, metaMBps, dataMBps, rows]);
  const pictures = useMemo(() => FORMATS.map((f) => filePicture(params, f, rows)), [access, columns, files, rowGroup, pageIndex, rows]);

  const n = N_ROWS / files;
  const X = (r: number) => PLOT_X0 + (r / n) * (PLOT_X1 - PLOT_X0);
  const panelH = 18 + KINDS.length * (BAND_H + BAND_GAP);
  const top = 24;
  const overviewH = top + FORMATS.length * (panelH + FORMAT_GAP);

  const isTake = access === 'take';
  const zoom = isTake ? zoomWindow(n, pictures[0].localRows) : ([0, n] as [number, number]);
  const ZX = (r: number) => PLOT_X0 + ((r - zoom[0]) / Math.max(1, zoom[1] - zoom[0])) * (PLOT_X1 - PLOT_X0);
  const zoomKinds: Kind[] = ['embedding', 'text'];
  const zoomTop = overviewH + 30;
  const zoomRowH = 20;
  const H1 = isTake ? zoomTop + FORMATS.length * zoomKinds.length * zoomRowH + 26 : overviewH + 4;

  const P = costs.parquet;
  const L = costs.lance;
  const V = costs.vortex;
  const pq = parquetFile(params, n);
  const stack = (fn: (c: Cost) => string) => (
    <>
      {FORMATS.map((f, i) => (
        <span key={f}>
          {i > 0 ? <br /> : null}
          {fn(costs[f])}
        </span>
      ))}
    </>
  );

  const metrics: { key: string; label: string; color: string; value: (c: Cost) => number; fmt: (v: number) => string }[] = [
    { key: 'meta', label: 'Metadata decoded', color: 'var(--viz-7)', value: (c) => c.metaDecoded, fmt: fmtBytes },
    { key: 'ranges', label: 'Byte ranges requested', color: 'var(--viz-6)', value: (c) => c.metaRanges + c.dataRanges, fmt: (v) => fmtNum(v) },
    { key: 'data', label: 'Data bytes read', color: 'var(--viz-1)', value: (c) => c.dataBytes, fmt: fmtBytes },
    { key: 'time', label: 'Estimated decode time', color: 'var(--viz-2)', value: (c) => c.decodeSeconds, fmt: fmtSec },
  ];
  const BAR_ROW = 15;
  const GROUP_H = 18 + FORMATS.length * BAR_ROW + 8;
  const H2 = metrics.length * GROUP_H + 4;
  const BAR_X0 = 128;
  const BAR_X1 = 560;

  const accessText = isTake ? `${TAKE_ROWS} random rows of id, text and embedding` : access === 'project' ? 'id, f_1 and f_2 for every row' : `all ${fmtNum(columns)} columns for every row`;

  const note = (() => {
    if (isTake) {
      const pqEmb = P.perKind.find((k) => k.kind === 'embedding');
      const vxText = V.perKind.find((k) => k.kind === 'text');
      return (
        <>
          <strong>
            Point fetch of {TAKE_ROWS} rows from {fmtNum(P.filesOpened)} file{P.filesOpened === 1 ? '' : 's'}.
          </strong>{' '}
          Parquet {pageIndex ? `uses the page index and reads ${fmtNum(P.dataRanges)} whole pages` : `has no page index, so it reads the whole column chunk of every touched row group — ${fmtNum(P.dataRanges)} chunks`}: {fmtBytes(P.dataBytes)}, {fmtBytes(pqEmb?.bytes ?? 0)} of it embeddings. Lance computes each embedding's offset and reads exactly {fmtNum(TAKE_ROWS)} values, and reaches each text value through its repetition index: {fmtBytes(L.dataBytes)} in {fmtNum(L.dataRanges)} small ranges. Vortex reads every segment that holds a requested row: {fmtBytes(V.dataBytes)}, of which {fmtBytes(vxText?.bytes ?? 0)} is text{n >= VX_ROW_BLOCK ? ', because its default writer never cuts a string segment below 8,192 rows' : ''}.
        </>
      );
    }
    const share = (c: Cost) => (c.decodeSeconds > 0 ? Math.round((100 * c.metaSeconds) / c.decodeSeconds) : 0);
    return (
      <>
        <strong>
          {access === 'scan' ? 'Full scan' : 'Narrow projection'} over {fmtNum(files)} file{files === 1 ? '' : 's'}.
        </strong>{' '}
        Every format reads the same {fmtBytes(P.dataBytes)} of column data. Parquet decodes {fmtBytes(P.metaDecoded)} of Thrift footer ({fmtNum(pq.rgCount)} row group{pq.rgCount === 1 ? '' : 's'} × {fmtNum(columns)} columns in each file): {share(P)}% of its estimated decode time. Lance decodes {fmtBytes(L.metaDecoded)} of metadata ({share(L)}%), Vortex {fmtBytes(V.metaDecoded)} ({share(V)}%).
      </>
    );
  })();

  return (
    <VizPanel
      title="One dataset, three layouts"
      subtitle="1,000,000 rows of id, text, a 768-d embedding and float32 features, written as Parquet, Lance and Vortex's default layout. Pick an access pattern and a schema width: the figure shows what each reader fetches from one file, and the bars total the cost over every file it opens."
      controls={
        <>
          <Segmented
            label="Access pattern"
            value={access}
            onChange={setAccess}
            options={[
              { value: 'scan', label: 'Full scan' },
              { value: 'project', label: 'Narrow projection' },
              { value: 'take', label: `Point fetch: ${TAKE_ROWS} random rows` },
            ]}
          />
          <Segmented
            label="Schema width"
            value={String(columns) as '12' | '200' | '2000'}
            onChange={(v) => setColumns(Number(v))}
            options={[
              { value: '12', label: '12 columns' },
              { value: '200', label: '200' },
              { value: '2000', label: '2,000' },
            ]}
          />
          <Segmented
            label="Split into files"
            value={String(files) as '1' | '100' | '1000'}
            onChange={(v) => setFiles(Number(v))}
            options={[
              { value: '1', label: '1 file' },
              { value: '100', label: '100' },
              { value: '1000', label: '1,000' },
            ]}
          />
          <Choice
            label="Parquet row group size"
            value={rowGroup}
            onChange={setRowGroup}
            options={[
              { value: 'rows1m', label: '1,048,576 rows (Arrow C++, parquet-rs)' },
              { value: 'bytes128m', label: '128 MB (parquet-java)' },
              { value: 'rows10k', label: '10,000 rows' },
            ]}
          />
          <Check label="Parquet page index" checked={pageIndex} onChange={setPageIndex} />
          <Slider label="Metadata decode speed" min={50} max={500} step={10} value={metaMBps} onChange={setMetaMBps} format={(v) => `${v} MB/s`} />
          <Slider label="Value decode speed" min={200} max={4000} step={100} value={dataMBps} onChange={setDataMBps} format={(v) => `${fmtNum(v)} MB/s`} />
        </>
      }
      legend={
        <Legend
          items={[
            { label: 'Column data fetched', color: 'var(--viz-1)' },
            { label: 'Metadata decoded', color: 'var(--viz-7)' },
            { label: 'Byte ranges requested', color: 'var(--viz-6)' },
            { label: 'Estimated decode time', color: 'var(--viz-2)' },
            { label: 'Requested row', color: 'var(--viz-ink-2)', shape: 'line' },
            { label: 'Row group boundary', color: 'var(--viz-ink)', shape: 'line' },
          ]}
        />
      }
      stats={
        <Stats
          items={[
            { label: 'Metadata decoded (Parquet · Lance · Vortex)', value: stack((c) => fmtBytes(c.metaDecoded)), hint: 'Parquet: the whole Thrift footer, plus page-index entries on a point fetch. Lance: the column offset table plus the projected column descriptors when a query reads under a quarter of the columns, otherwise every descriptor. Vortex: the segment map, the dtype and the projected columns’ layout nodes.' },
            { label: 'Byte ranges', value: stack((c) => fmtNum(c.metaRanges + c.dataRanges)), hint: 'Metadata plus data ranges, counted before any coalescing of nearby ranges.' },
            { label: 'Data bytes read', value: stack((c) => fmtBytes(c.dataBytes)), hint: 'Stored (encoded, compressed) bytes of column data fetched.' },
            { label: 'Est. decode time', value: stack((c) => fmtSec(c.decodeSeconds)), hint: 'Metadata bytes ÷ metadata speed + decoded value bytes ÷ value speed. Both speeds are editable model constants, not measurements.' },
          ]}
        />
      }
      note={<Note>{note}</Note>}
      table={
        <table className="viz-table">
          <thead>
            <tr>
              <th>Format</th>
              <th>Column</th>
              <th>Unit fetched</th>
              <th>Ranges</th>
              <th>Bytes read</th>
              <th>Bytes decoded</th>
            </tr>
          </thead>
          <tbody>
            {FORMATS.map((f) => {
              const c = costs[f];
              return [
                <tr key={`${f}-meta`}>
                  <td>{FORMAT_NAME[f]}</td>
                  <td>
                    metadata ({fmtNum(c.filesOpened)} file{c.filesOpened === 1 ? '' : 's'})
                  </td>
                  <td>{f === 'parquet' ? `footer tail read${pageIndex && isTake ? ' + page index' : ''}` : f === 'lance' ? `tail read + column descriptors${isTake ? ' + mini-block lookups' : ''}` : 'initial tail read + metadata segments'}</td>
                  <td>{fmtNum(c.metaRanges)}</td>
                  <td>{fmtBytes(c.metaFetched)}</td>
                  <td>{fmtBytes(c.metaDecoded)}</td>
                </tr>,
                ...c.perKind.map((k) => (
                  <tr key={`${f}-${k.kind}`}>
                    <td>{FORMAT_NAME[f]}</td>
                    <td>
                      {KIND_INFO[k.kind].name}
                      {k.count > 1 ? ` ×${fmtNum(k.count)}` : ''}
                    </td>
                    <td>{k.unit}</td>
                    <td>{fmtNum(k.ranges)}</td>
                    <td>{fmtBytes(k.bytes)}</td>
                    <td>{fmtBytes(k.decoded)}</td>
                  </tr>
                )),
                <tr key={`${f}-time`}>
                  <td>{FORMAT_NAME[f]}</td>
                  <td>estimated decode time</td>
                  <td>
                    metadata {fmtSec(c.metaSeconds)} + values {fmtSec(c.dataSeconds)}
                  </td>
                  <td colSpan={3}>{fmtSec(c.decodeSeconds)}</td>
                </tr>,
              ];
            })}
          </tbody>
        </table>
      }
    >
      <svg viewBox={`0 0 ${W} ${H1}`} width={W} height={H1} role="img" aria-label={`Rows each format fetches from one file for ${accessText}`}>
        <text x={LABEL_X} y={14} fontSize={11} fill="var(--viz-ink-2)">
          {isTake ? `File ${fmtNum(pictures[0].fileNo + 1)} of ${fmtNum(files)}: ${fmtNum(pictures[0].localRows.length)} requested rows, ${fmtNum(n)} rows in the file` : `File 1 of ${fmtNum(files)}: ${fmtNum(n)} rows`}
        </text>
        <text x={BYTES_X} y={14} fontSize={10.5} textAnchor="end" fill="var(--viz-ink-2)">
          read here
        </text>
        {isTake
          ? pictures[0].localRows.map((r) => <line key={r} x1={X(r)} x2={X(r)} y1={top - 4} y2={overviewH - FORMAT_GAP + 4} stroke="var(--viz-ink-2)" strokeOpacity={0.16} strokeWidth={1} />)
          : null}
        {FORMATS.map((f, fi) => {
          const pic = pictures[fi];
          const c = costs[f];
          const y0 = top + fi * (panelH + FORMAT_GAP);
          const perFileMeta = c.metaDecoded / Math.max(1, c.filesOpened);
          return (
            <g key={f}>
              <text x={LABEL_X} y={y0 + 11} fontSize={13} fontWeight={600} fill="var(--viz-ink)">
                {FORMAT_NAME[f]}
              </text>
              <text x={PLOT_X0} y={y0 + 11} fontSize={10.5} fill="var(--viz-ink-2)">
                {f === 'parquet' ? `${fmtNum(pq.rgCount)} row group${pq.rgCount === 1 ? '' : 's'}; footer lists ${fmtNum(pq.rgCount * columns)} column chunks` : f === 'lance' ? 'no row groups; each column paged on its own' : 'no row groups; columns cut into ~1 MB segments'}
              </text>
              {pic.bands.map((b, bi) => {
                const y = y0 + 18 + bi * (BAND_H + BAND_GAP);
                const spacing = ((PLOT_X1 - PLOT_X0) * b.unitRows) / n;
                const ticks = spacing >= 5 ? b.ticksIn(1, n - 1) : [];
                return (
                  <g key={b.kind}>
                    <title>{`${FORMAT_NAME[f]} · ${bandName(b)}: ${b.unitLabel}`}</title>
                    <text x={LABEL_X + 6} y={y + 11} fontSize={10.5} fill={b.projected > 0 ? 'var(--viz-ink)' : 'var(--viz-ink-muted)'}>
                      {bandName(b)}
                    </text>
                    <rect x={PLOT_X0} y={y} width={PLOT_X1 - PLOT_X0} height={BAND_H} fill="var(--viz-surface)" stroke="var(--viz-grid)" />
                    {b.read.map(([a, e], i) => (
                      <rect key={i} x={X(a)} y={y + 1} width={Math.max(1.5, X(e) - X(a))} height={BAND_H - 2} fill="var(--viz-1)" fillOpacity={b.kind === 'feature' && b.projected < b.total ? 0.35 : 0.85} />
                    ))}
                    {ticks.map((t) => (
                      <line key={t} x1={X(t)} x2={X(t)} y1={y + 3} y2={y + BAND_H - 3} stroke={b.projected > 0 && !isTake ? 'var(--viz-surface)' : 'var(--viz-ink-muted)'} strokeOpacity={0.8} />
                    ))}
                    {b.groupEdges.length > 2 && (PLOT_X1 - PLOT_X0) / (b.groupEdges.length - 1) >= 8
                      ? b.groupEdges.slice(1, -1).map((g) => <line key={`g${g}`} x1={X(g)} x2={X(g)} y1={y - 2} y2={y + BAND_H + 2} stroke="var(--viz-ink)" strokeWidth={1.5} />)
                      : null}
                    <text x={BYTES_X} y={y + 11} fontSize={10.5} textAnchor="end" fill={b.projected > 0 ? 'var(--viz-ink)' : 'var(--viz-ink-muted)'}>
                      {b.projected > 0 ? fmtBytes(b.bytes) : '—'}
                    </text>
                  </g>
                );
              })}
              <rect x={TAIL_X} y={y0 + 18} width={W - TAIL_X - 4} height={KINDS.length * (BAND_H + BAND_GAP) - BAND_GAP} rx={3} fill="var(--viz-surface)" stroke="var(--viz-7)" strokeWidth={1.5} />
              <text x={TAIL_X + 6} y={y0 + 32} fontSize={10.5} fill="var(--viz-ink)">
                {f === 'parquet' ? (isTake && pageIndex ? 'footer + page index' : 'Thrift footer') : f === 'lance' ? 'column descriptors' : 'segment map + layout'}
              </text>
              <text x={TAIL_X + 6} y={y0 + 46} fontSize={10.5} fill="var(--viz-ink-2)">
                decoded per file
              </text>
              <text x={TAIL_X + 6} y={y0 + 64} fontSize={13} fontWeight={600} fill="var(--viz-ink)">
                {fmtBytes(perFileMeta)}
              </text>
            </g>
          );
        })}
        {isTake ? (
          <g>
            <text x={LABEL_X} y={zoomTop - 12} fontSize={12} fontWeight={600} fill="var(--viz-ink)">
              Zoom: rows {fmtNum(zoom[0])}–{fmtNum(zoom[1] - 1)} around requested row {fmtNum(pictures[0].localRows[0] ?? 0)}
            </text>
            <text x={BYTES_X} y={zoomTop - 12} fontSize={10.5} textAnchor="end" fill="var(--viz-ink-2)">
              rows fetched
            </text>
            {pictures[0].localRows
              .filter((r) => r >= zoom[0] && r < zoom[1])
              .map((r) => (
                <line key={`z${r}`} x1={ZX(r + 0.5)} x2={ZX(r + 0.5)} y1={zoomTop - 6} y2={zoomTop + FORMATS.length * zoomKinds.length * zoomRowH} stroke="var(--viz-ink-2)" strokeWidth={1} />
              ))}
            {FORMATS.flatMap((f, fi) =>
              zoomKinds.map((k, ki) => {
                const b = pictures[fi].bands.find((x) => x.kind === k)!;
                const y = zoomTop + (fi * zoomKinds.length + ki) * zoomRowH;
                const inWin = pictures[0].localRows.filter((r) => r >= zoom[0] && r < zoom[1]);
                const units = mergeIntervals(inWin.map((r) => b.unitOf(r)));
                const fetched = inWin.length ? b.unitOf(inWin[0]) : ([0, 0] as [number, number]);
                return (
                  <g key={`${f}-${k}`}>
                    <text x={LABEL_X + (ki === 0 ? 0 : 0)} y={y + 11} fontSize={10.5} fill="var(--viz-ink)">
                      {ki === 0 ? <tspan fontWeight={600}>{FORMAT_NAME[f]} </tspan> : <tspan fill="var(--viz-ink-muted)">{'  '}</tspan>}
                      <tspan x={LABEL_X + 52}>{k}</tspan>
                    </text>
                    <rect x={PLOT_X0} y={y} width={PLOT_X1 - PLOT_X0} height={BAND_H} fill="var(--viz-surface)" stroke="var(--viz-grid)" />
                    {units.map(([a, e], i) => {
                      const x0 = ZX(Math.max(a, zoom[0]));
                      const x1 = ZX(Math.min(e, zoom[1]));
                      return <rect key={i} x={x0} y={y + 1} width={Math.max(2, x1 - x0)} height={BAND_H - 2} fill="var(--viz-1)" fillOpacity={0.85} />;
                    })}
                    {b.ticksIn(zoom[0] + 1, zoom[1] - 1).map((t) => (
                      <line key={t} x1={ZX(t)} x2={ZX(t)} y1={y + 2} y2={y + BAND_H - 2} stroke="var(--viz-ink-muted)" />
                    ))}
                    {f === 'parquet'
                      ? b.groupEdges.filter((g) => g > zoom[0] && g < zoom[1]).map((g) => <line key={`zg${g}`} x1={ZX(g)} x2={ZX(g)} y1={y - 2} y2={y + BAND_H + 2} stroke="var(--viz-ink)" strokeWidth={1.5} />)
                      : null}
                    <text x={BYTES_X} y={y + 11} fontSize={10.5} textAnchor="end" fill="var(--viz-ink)">
                      {inWin.length ? fmtNum(fetched[1] - fetched[0]) : '—'}
                    </text>
                    <text x={TAIL_X + 4} y={y + 11} fontSize={10} fill="var(--viz-ink-2)">
                      {f === 'parquet' ? (pageIndex ? 'page' : 'column chunk') : f === 'lance' ? (k === 'embedding' ? 'row × 3,072 B' : 'rep. index → value') : 'segment'}
                    </text>
                  </g>
                );
              }),
            )}
            <text x={PLOT_X0} y={H1 - 6} fontSize={10} fill="var(--viz-ink-muted)">
              Ticks: unit boundaries. Filled: rows fetched to return the marked row.
            </text>
          </g>
        ) : null}
      </svg>

      <svg viewBox={`0 0 ${W} ${H2}`} width={W} height={H2} role="img" aria-label="Cost per format, log scale" style={{ marginTop: 8 }}>
        {metrics.map((m, mi) => {
          const vals = FORMATS.map((f) => m.value(costs[f]));
          const max = Math.max(...vals);
          const positive = vals.filter((v) => v > 0);
          const min = Math.max(positive.length ? Math.min(...positive) : 1, max / 1e6);
          const lo = Math.log10(min) - 0.5;
          const hi = Math.log10(Math.max(max, 1e-12)) + 0.05;
          const scale = (v: number) => (v <= 0 ? 0 : Math.max(2, ((Math.log10(v) - lo) / Math.max(0.5, hi - lo)) * (BAR_X1 - BAR_X0)));
          const y0 = mi * GROUP_H;
          return (
            <g key={m.key}>
              <text x={LABEL_X} y={y0 + 13} fontSize={12} fontWeight={600} fill="var(--viz-ink)">
                {m.label}
              </text>
              <text x={W - 8} y={y0 + 13} fontSize={10} textAnchor="end" fill="var(--viz-ink-muted)">
                log scale
              </text>
              {FORMATS.map((f, fi) => {
                const v = vals[fi];
                const y = y0 + 18 + fi * BAR_ROW;
                const w = scale(v);
                return (
                  <g key={f}>
                    <text x={LABEL_X + 8} y={y + 10} fontSize={11} fill="var(--viz-ink-2)">
                      {FORMAT_NAME[f]}
                    </text>
                    <rect x={BAR_X0} y={y + 1} width={w} height={BAR_ROW - 4} rx={2} fill={m.color} />
                    <text x={BAR_X0 + w + 6} y={y + 10} fontSize={11} fill="var(--viz-ink)">
                      {m.fmt(v)}
                    </text>
                  </g>
                );
              })}
            </g>
          );
        })}
      </svg>
      <p className="viz-sub" style={{ marginTop: 4 }}>
        Model constants, not measurements: every format stores the same bytes per value (id 4 B, text 600 B, embedding 3,072 B, feature 4 B); metadata entries are estimated at ~{PQ_COLUMN_CHUNK} B per Parquet ColumnChunk, ~{LANCE_PAGE_DESCRIPTOR} B per Lance page descriptor and {VX_SEGMENT_SPEC} B per Vortex segment; every byte of metadata decoded is charged at the same speed, which overstates FlatBuffers; ranges are counted before coalescing; file sizes come from the catalog.
      </p>
    </VizPanel>
  );
}
