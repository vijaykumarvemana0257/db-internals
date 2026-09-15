import { useDeferredValue, useMemo, useState } from 'react';
import { VizPanel, Segmented, Choice, Slider, Check, Legend, Stats, Note, TooltipHost, useTip, makeRng, fmtBytes, fmtNum } from './Viz';

/**
 * Shredding schema-less JSON into sub-columns, three ways. Each block of documents (a ClickHouse part, a Snowflake
 * micro-partition, a Parquet file written by Spark) is shredded on its own, so the same path can be a typed column in
 * one block and buried in a blob in the next.
 *
 * ClickHouse JSON (docs: data-types/newjson):
 *  - Paths become dynamic sub-columns first-come within a block until max_dynamic_paths (default 1024); every later new
 *    path in that block goes to "shared data" (a Map(String, String) of path -> binary value). JSON null and a missing
 *    key are the same thing. A path whose values change type stays one Dynamic sub-column holding several types.
 *  - A merge keeps the paths with the most non-null values and moves the rarest to shared data.
 *  - Reading a path from shared data: `map` reads the whole Map column, `map_with_buckets` reads one bucket
 *    (8 buckets, the compact-part default), `advanced` reads only that path's data.
 * Snowflake VARIANT (docs: semistructured-considerations; SIGMOD 2016 §4.3):
 *  - Per micro-partition, up to 200 elements are extracted to columns. An element with even one JSON null, or with
 *    more than one type, is never extracted. The rest stays in one parsed semi-structured column.
 *  - A filter on an element that was not extracted scans that whole column; per-file Bloom filters over paths let the
 *    optimizer skip partitions that do not contain the path at all.
 * Spark 4.1 writing Parquet variant (InferVariantShreddingSchema, ParquetOutputWriterWithVariantShredding):
 *  - The writer buffers the first 4096 rows of a file, infers one shredding schema from them, and writes the whole file
 *    with it. Fields seen in fewer than 10% of buffered rows are dropped. Integers and decimals merge; any other type
 *    conflict makes the field an untyped `value`-only column. A budget (maxSchemaWidth, default 300) is spent on one
 *    column per field plus one per typed_value, walking fields in alphabetical order.
 *  - Values that do not fit the file's schema land in the nearest `value` column: a field's own (type mismatch) or
 *    the enclosing object's residual (field not shredded).
 *
 * Model assumptions (labelled in the lab):
 *  - Limits are scaled down so small samples reach them; Spark's 4096-row buffer is scaled to SPARK_BUFFER_ROWS.
 *  - Byte counts are uncompressed, except that a column with no values counts as zero and identical Variant metadata
 *    dictionaries are counted once (Parquet dictionary encoding). Binary blobs are sized with the Parquet Variant
 *    encoding (Snowflake's internal format is not public). Typed values: 8 bytes for numbers, 1 for booleans, strings with a length prefix
 *    (4 bytes in Parquet/Snowflake, 1-byte varint in ClickHouse), plus a 1-bit-per-row presence map per column.
 *  - Snowflake ranks eligible elements by frequency and keeps arrays in the blob (neither is documented).
 *    ClickHouse registers paths in document order. Bucket assignment uses FNV-1a, not ClickHouse's hash.
 *  - Rows/s normalises bytes read at a fixed 1 GB/s, the same for typed columns and blobs; walking a blob costs more
 *    per byte than decoding a typed column, so the real gap is wider.
 */

export type Engine = 'clickhouse' | 'snowflake' | 'spark';
export type Serialization = 'map' | 'map_with_buckets' | 'advanced';
export type Loc = 'sub' | 'multi' | 'blob';

export const SPARK_BUFFER_ROWS = 10;
export const REAL_SPARK_BUFFER_ROWS = 4096;
export const CH_BUCKETS = 8;
export const MAX_DOCS = 240;
export const MAX_BLOCKS = 10;
const READ_RATE = 1e9; // bytes per second, a normalisation, not a measurement

export const ENGINE_INFO: Record<Engine, { block: string; limitName: string; realDefault: number; blob: string; unit: string }> = {
  clickhouse: { block: 'part', limitName: 'max_dynamic_paths', realDefault: 1024, blob: 'shared data', unit: 'dynamic paths' },
  snowflake: { block: 'partition', limitName: 'elements extracted per partition', realDefault: 200, blob: 'rest of the VARIANT', unit: 'extracted' },
  spark: { block: 'file', limitName: 'maxSchemaWidth', realDefault: 300, blob: 'residual value', unit: 'budget' },
};

/* ------------------------------------------------------------------ JSON parsing (keeps number spelling) */

export type PV =
  | { k: 'null' }
  | { k: 'bool'; v: boolean }
  | { k: 'int'; raw: string; digits: number }
  | { k: 'dec'; raw: string; prec: number; scale: number }
  | { k: 'dbl'; raw: string }
  | { k: 'str'; v: string }
  | { k: 'arr'; items: PV[] }
  | { k: 'obj'; keys: string[]; vals: PV[] };

const NUM = /-?(0|[1-9]\d*)(\.\d+)?([eE][+-]?\d+)?/y;

export function parseJson(s: string): PV {
  let i = 0;
  const fail = (m: string): never => {
    throw new Error(`${m} at column ${i + 1}`);
  };
  const ws = () => {
    while (i < s.length && (s[i] === ' ' || s[i] === '\t' || s[i] === '\r' || s[i] === '\n')) i++;
  };
  const str = () => {
    i++;
    let out = '';
    while (i < s.length && s[i] !== '"') {
      if (s[i] === '\\') {
        const e = s[i + 1];
        if (e === 'u') {
          out += String.fromCharCode(parseInt(s.slice(i + 2, i + 6), 16) || 0);
          i += 6;
        } else {
          out += e === 'n' ? '\n' : e === 't' ? '\t' : e === 'r' ? '\r' : e === 'b' ? '\b' : e === 'f' ? '\f' : (e ?? '');
          i += 2;
        }
      } else out += s[i++];
    }
    if (s[i] !== '"') fail('unterminated string');
    i++;
    return out;
  };
  const val = (depth: number): PV => {
    if (depth > 32) fail('nesting deeper than 32');
    ws();
    const c = s[i];
    if (c === '{') {
      i++;
      const keys: string[] = [];
      const vals: PV[] = [];
      ws();
      if (s[i] === '}') {
        i++;
        return { k: 'obj', keys, vals };
      }
      for (;;) {
        ws();
        if (s[i] !== '"') fail('expected a quoted key');
        const key = str();
        if (keys.includes(key)) fail(`duplicate key "${key}"`);
        ws();
        if (s[i] !== ':') fail('expected ":"');
        i++;
        vals.push(val(depth + 1));
        keys.push(key);
        ws();
        if (s[i] === ',') {
          i++;
          continue;
        }
        if (s[i] === '}') {
          i++;
          break;
        }
        fail('expected "," or "}"');
      }
      return { k: 'obj', keys, vals };
    }
    if (c === '[') {
      i++;
      const items: PV[] = [];
      ws();
      if (s[i] === ']') {
        i++;
        return { k: 'arr', items };
      }
      for (;;) {
        items.push(val(depth + 1));
        ws();
        if (s[i] === ',') {
          i++;
          continue;
        }
        if (s[i] === ']') {
          i++;
          break;
        }
        fail('expected "," or "]"');
      }
      return { k: 'arr', items };
    }
    if (c === '"') return { k: 'str', v: str() };
    if (s.startsWith('true', i)) {
      i += 4;
      return { k: 'bool', v: true };
    }
    if (s.startsWith('false', i)) {
      i += 5;
      return { k: 'bool', v: false };
    }
    if (s.startsWith('null', i)) {
      i += 4;
      return { k: 'null' };
    }
    NUM.lastIndex = i;
    const m = NUM.exec(s);
    if (!m || m[0].length === 0) fail('unexpected character');
    const raw = m![0];
    i += raw.length;
    if (m![3]) return { k: 'dbl', raw };
    const unsigned = raw.replace('-', '');
    if (!m![2]) {
      const digits = unsigned.length;
      return digits <= 18 ? { k: 'int', raw, digits } : digits <= 38 ? { k: 'dec', raw, prec: digits, scale: 0 } : { k: 'dbl', raw };
    }
    const scale = m![2].length - 1;
    const sig = unsigned.replace('.', '').replace(/^0+/, '');
    const prec = Math.max(sig.length, scale, 1);
    return prec <= 38 ? { k: 'dec', raw, prec, scale } : { k: 'dbl', raw };
  };
  const v = val(0);
  ws();
  if (i < s.length) fail('unexpected trailing characters');
  return v;
}

export type Doc = { pv: PV; raw: number; leaves: { path: string; v: PV }[] };

export function leavesOf(v: PV, prefix = '', out: { path: string; v: PV }[] = []) {
  if (v.k === 'obj') {
    v.keys.forEach((key, j) => leavesOf(v.vals[j], prefix ? `${prefix}.${key}` : key, out));
  } else if (prefix) out.push({ path: prefix, v });
  return out;
}

export function parseDocs(text: string): { docs: Doc[]; error: string | null; truncated: boolean } {
  const lines = text.split('\n').map((l) => l.trim()).filter((l) => l.length > 0);
  const docs: Doc[] = [];
  for (let n = 0; n < Math.min(lines.length, MAX_DOCS); n++) {
    const line = lines[n].slice(0, 4000);
    try {
      const pv = parseJson(line);
      if (pv.k !== 'obj') return { docs, error: `Line ${n + 1}: each line must be a JSON object`, truncated: false };
      const leaves = leavesOf(pv);
      const seen = new Set<string>();
      for (const l of leaves) {
        if (seen.has(l.path)) return { docs, error: `Line ${n + 1}: two keys flatten to the same path "${l.path}"`, truncated: false };
        seen.add(l.path);
      }
      docs.push({ pv, raw: utf8(line), leaves });
    } catch (e) {
      return { docs, error: `Line ${n + 1}: ${(e as Error).message}`, truncated: false };
    }
  }
  return { docs, error: null, truncated: lines.length > MAX_DOCS };
}

/* ------------------------------------------------------------------ sizes */

export function utf8(s: string) {
  let n = 0;
  for (const ch of s) {
    const c = ch.codePointAt(0) ?? 0;
    n += c < 0x80 ? 1 : c < 0x800 ? 2 : c < 0x10000 ? 3 : 4;
  }
  return n;
}
const width = (n: number) => (n < 256 ? 1 : n < 65536 ? 2 : n < 16777216 ? 3 : 4);

/** Parquet Variant value encoding size (VariantEncoding.md). */
export function variantSize(v: PV): number {
  switch (v.k) {
    case 'null':
    case 'bool':
      return 1;
    case 'int': {
      const a = Math.abs(Number(v.raw));
      return 1 + (a < 128 ? 1 : a < 32768 ? 2 : a < 2 ** 31 ? 4 : 8);
    }
    case 'dec':
      return 2 + (v.prec <= 9 ? 4 : v.prec <= 18 ? 8 : 16);
    case 'dbl':
      return 9;
    case 'str': {
      const n = utf8(v.v);
      return n < 64 ? 1 + n : 5 + n;
    }
    case 'arr': {
      const body = v.items.reduce((s, x) => s + variantSize(x), 0);
      const n = v.items.length;
      return 1 + (n < 256 ? 1 : 4) + (n + 1) * width(body) + body;
    }
    case 'obj': {
      const body = v.vals.reduce((s, x) => s + variantSize(x), 0);
      const n = v.keys.length;
      return 1 + (n < 256 ? 1 : 4) + n + (n + 1) * width(body) + body;
    }
  }
}

/** Variant metadata for one value: header, dictionary size, offsets, key bytes. */
export function metadataSize(v: PV) {
  const keys = new Set<string>();
  const walk = (x: PV) => {
    if (x.k === 'obj') x.keys.forEach((k, j) => (keys.add(k), walk(x.vals[j])));
    if (x.k === 'arr') x.items.forEach(walk);
  };
  walk(v);
  const bytes = [...keys].reduce((s, k) => s + utf8(k), 0);
  const w = width(Math.max(bytes, keys.size));
  return { size: 1 + w + (keys.size + 1) * w + bytes, sig: [...keys].sort().join('\u0000') };
}

/**
 * The metadata column of a block. Rows with the same key set have byte-identical metadata, which Parquet's dictionary
 * encoding stores once; count each distinct dictionary once plus a one-byte index per row.
 */
export function metadataColumn(values: PV[]) {
  const seen = new Map<string, number>();
  for (const v of values) {
    const m = metadataSize(v);
    seen.set(m.sig, m.size);
  }
  return values.length + [...seen.values()].reduce((a, b) => a + b, 0);
}

const presence = (rows: number) => Math.ceil(rows / 8);

/** Typed (shredded) value bytes. strPrefix: 4 for Parquet BYTE_ARRAY / Snowflake, 1 for ClickHouse varint. */
function typedSize(v: PV, strPrefix: number): number {
  switch (v.k) {
    case 'int':
    case 'dbl':
      return 8;
    case 'dec':
      return v.prec <= 18 ? 8 : 16;
    case 'bool':
      return 1;
    case 'str':
      return strPrefix + utf8(v.v);
    case 'null':
      return 0;
    case 'arr':
      return 8 + v.items.reduce((s, x) => s + typedSize(x, strPrefix), 0);
    case 'obj':
      return variantSize(v);
  }
}

function withoutPath(v: PV, segs: string[]): PV {
  if (v.k !== 'obj' || segs.length === 0) return v;
  const j = v.keys.indexOf(segs[0]);
  if (j < 0) return v;
  if (segs.length === 1) return { k: 'obj', keys: v.keys.filter((_, x) => x !== j), vals: v.vals.filter((_, x) => x !== j) };
  const child = withoutPath(v.vals[j], segs.slice(1));
  return { k: 'obj', keys: v.keys, vals: v.vals.map((c, x) => (x === j ? child : c)) };
}

/** Rebuild the object holding only the given leaves (for Snowflake's rest column). */
function objectOf(leaves: { path: string; v: PV }[]): PV {
  const root: PV = { k: 'obj', keys: [], vals: [] };
  for (const l of leaves) {
    const segs = l.path.split('.');
    let node = root as Extract<PV, { k: 'obj' }>;
    segs.forEach((seg, d) => {
      let j = node.keys.indexOf(seg);
      if (d === segs.length - 1) {
        node.keys.push(seg);
        node.vals.push(l.v);
        return;
      }
      if (j < 0) {
        node.keys.push(seg);
        node.vals.push({ k: 'obj', keys: [], vals: [] });
        j = node.keys.length - 1;
      }
      node = node.vals[j] as Extract<PV, { k: 'obj' }>;
    });
  }
  return root;
}

const fnv = (s: string) => {
  let h = 0x811c9dc5;
  for (let i = 0; i < s.length; i++) {
    h ^= s.charCodeAt(i);
    h = Math.imul(h, 0x01000193) >>> 0;
  }
  return h >>> 0;
};

/* ------------------------------------------------------------------ engine results */

export type Cell = { loc: Loc; code: string; detail: string; reason?: string; promoted?: boolean };
export type Read = { bytes: number; loc: Loc | 'skip'; how: string };
export type Block = {
  index: number;
  first: number;
  rows: number;
  used: number;
  limit: number;
  atLimit: boolean;
  cells: Record<string, Cell>;
  blobBytes: number;
  blobPaths: string[];
  totalBytes: number;
  rawBytes: number;
  reads: Record<string, Read>;
};

const CH_TYPE: Record<PV['k'], string> = { int: 'Int64', dec: 'Float64', dbl: 'Float64', str: 'String', bool: 'Bool', arr: 'Array', obj: 'JSON', null: '' };
const CH_CODE: Record<string, string> = { Int64: 'i64', Float64: 'f64', String: 'str', Bool: 'bool', Array: 'arr', JSON: 'obj' };

/** The type a promoted path is declared with: one type for the whole column, from its most common value kind. */
function declaredType(docs: Doc[], path: string) {
  const kinds = new Map<PV['k'], number>();
  for (const d of docs) for (const l of d.leaves) if (l.path === path && l.v.k !== 'null') kinds.set(l.v.k, (kinds.get(l.v.k) ?? 0) + 1);
  const top = [...kinds.entries()].sort((a, b) => b[1] - a[1])[0]?.[0];
  return top ? CH_TYPE[top] : 'String';
}

function promotedColumn(docs: Doc[], path: string, strPrefix: number) {
  let bytes = presence(docs.length);
  let present = false;
  for (const d of docs)
    for (const l of d.leaves)
      if (l.path === path && l.v.k !== 'null') {
        bytes += typedSize(l.v, strPrefix);
        present = true;
      }
  return { bytes: present ? bytes : 0, present };
}

function clickhouseBlock(docs: Doc[], index: number, first: number, limit: number, allPaths: string[], ser: Serialization, promote: string | null, promoteType: string, mergeOrder: string[] | null): Block {
  const rows = docs.length;
  const dynamic: string[] = [];
  const stats = new Map<string, { bytes: number; types: Set<string> }>();
  const shared = new Map<string, number>();
  const cells: Record<string, Cell> = Object.create(null);
  const promoted = promote ? promotedColumn(docs, promote, 1) : null;

  if (mergeOrder) for (const p of mergeOrder.slice(0, limit)) dynamic.push(p);
  for (const d of docs) {
    for (const l of d.leaves) {
      if (l.path === promote) continue;
      if (l.v.k === 'null') continue; // ClickHouse: null and missing are the same
      if (!dynamic.includes(l.path) && !mergeOrder && dynamic.length < limit) dynamic.push(l.path);
      if (dynamic.includes(l.path)) {
        const s = stats.get(l.path) ?? { bytes: 0, types: new Set<string>() };
        s.bytes += typedSize(l.v, 1);
        s.types.add(CH_TYPE[l.v.k]);
        stats.set(l.path, s);
      } else {
        shared.set(l.path, (shared.get(l.path) ?? 0) + 1 + utf8(l.path) + 1 + 1 + typedSize(l.v, 1));
      }
    }
  }
  const liveDynamic = dynamic.filter((p) => stats.has(p));
  let total = 0;
  for (const p of liveDynamic) {
    const s = stats.get(p)!;
    const types = [...s.types];
    const bytes = rows + s.bytes; // one discriminator byte per row + values
    total += bytes;
    cells[p] = {
      loc: types.length > 1 ? 'multi' : 'sub',
      code: types.length > 1 ? `${types.length} types` : CH_CODE[types[0]],
      detail: `Dynamic sub-column${types.length > 1 ? ` holding ${types.join(' and ')}` : ` (${types[0]})`}: ${fmtBytes(bytes)}`,
    };
  }
  const sharedEntries = [...shared.values()].reduce((a, b) => a + b, 0);
  const sharedBytes = sharedEntries ? 8 * rows + sharedEntries : 0; // an empty Map column compresses to nothing
  for (const [p, b] of shared)
    cells[p] = {
      loc: 'blob',
      code: 'shared',
      detail: `In shared data: ${fmtBytes(b)} of path/value entries`,
      reason: mergeOrder ? `the merge kept the ${limit} paths with the most non-null values, and ${p} was not one of them` : `the part already had ${limit} dynamic paths when ${p} first appeared in it`,
    };
  if (promoted && promoted.present) {
    total += promoted.bytes;
    cells[promote!] = { loc: 'sub', code: 'typed', promoted: true, detail: `Type hint ${promote} ${promoteType}: always its own sub-column, outside the dynamic-path limit` };
  }
  total += sharedBytes;

  const bucketBytes = new Array<number>(CH_BUCKETS).fill(0);
  if (ser === 'map_with_buckets') for (const [q, v] of shared) bucketBytes[fnv(q) % CH_BUCKETS] += v;
  const reads: Record<string, Read> = Object.create(null);
  for (const p of allPaths) {
    if (p === promote && promoted) reads[p] = { bytes: promoted.bytes, loc: 'sub', how: 'typed-path sub-column' };
    else if (stats.has(p)) {
      const c = cells[p];
      reads[p] = { bytes: rows + stats.get(p)!.bytes, loc: c.loc, how: 'its Dynamic sub-column' };
    } else if (ser === 'map') reads[p] = { bytes: sharedBytes, loc: 'blob', how: 'the whole shared-data Map column' };
    else if (ser === 'map_with_buckets') {
      const b = fnv(p) % CH_BUCKETS;
      const inBucket = bucketBytes[b];
      reads[p] = { bytes: inBucket ? 8 * rows + inBucket : 0, loc: 'blob', how: `shared-data bucket ${b + 1} of ${CH_BUCKETS}` };
    } else reads[p] = { bytes: shared.has(p) ? presence(rows) + shared.get(p)! : 0, loc: 'blob', how: "only this path's data inside shared data" };
  }
  return {
    index,
    first,
    rows,
    used: liveDynamic.length,
    limit,
    atLimit: shared.size > 0,
    cells,
    blobBytes: sharedBytes,
    blobPaths: [...shared.keys()],
    totalBytes: total,
    rawBytes: docs.reduce((s, d) => s + d.raw, 0),
    reads,
  };
}

function snowflakeBlock(docs: Doc[], index: number, first: number, limit: number, allPaths: string[], promote: string | null): Block {
  const rows = docs.length;
  const kindOf = (v: PV) => (v.k === 'int' || v.k === 'dec' || v.k === 'dbl' ? 'NUMBER' : v.k === 'str' ? 'VARCHAR' : v.k === 'bool' ? 'BOOLEAN' : v.k === 'arr' ? 'ARRAY' : 'null');
  const stats = new Map<string, { count: number; types: Set<string>; hasNull: boolean; order: number; bytes: number }>();
  let order = 0;
  const promoted = promote ? promotedColumn(docs, promote, 4) : null;
  for (const d of docs)
    for (const l of d.leaves) {
      if (l.path === promote) continue;
      const s = stats.get(l.path) ?? { count: 0, types: new Set<string>(), hasNull: false, order: order++, bytes: 0 };
      s.count++;
      if (l.v.k === 'null') s.hasNull = true;
      else s.types.add(kindOf(l.v));
      s.bytes += typedSize(l.v, 4);
      stats.set(l.path, s);
    }
  const why = (s: { types: Set<string>; hasNull: boolean }) =>
    s.hasNull ? 'contains a JSON null' : s.types.size > 1 ? `has ${[...s.types].join(' and ')} values` : s.types.has('ARRAY') ? 'is an array (kept in the blob by this model)' : '';
  const eligible = [...stats.entries()].filter(([, s]) => !why(s)).sort((a, b) => b[1].count - a[1].count || a[1].order - b[1].order);
  const extracted = new Set(eligible.slice(0, limit).map(([p]) => p));
  const cells: Record<string, Cell> = Object.create(null);
  let total = 0;
  for (const p of extracted) {
    const s = stats.get(p)!;
    const bytes = presence(rows) + s.bytes;
    total += bytes;
    const t = [...s.types][0];
    cells[p] = { loc: 'sub', code: t === 'NUMBER' ? 'num' : t === 'VARCHAR' ? 'str' : 'bool', detail: `Extracted ${t} column: ${fmtBytes(bytes)}` };
  }
  let rest = 0;
  const restObjects: PV[] = [];
  for (const d of docs) {
    const left = d.leaves.filter((l) => l.path !== promote && !extracted.has(l.path));
    if (left.length) {
      const o = objectOf(left);
      rest += variantSize(o);
      restObjects.push(o);
    }
  }
  if (restObjects.length) rest += metadataColumn(restObjects);
  const blobPaths = [...stats.keys()].filter((p) => !extracted.has(p));
  for (const p of blobPaths) {
    const s = stats.get(p)!;
    const reason = why(s);
    const because = reason ? `${p} ${reason}` : `the ${limit}-element limit went to more frequent elements`;
    cells[p] = { loc: 'blob', code: 'rest', detail: `Not extracted: ${because}`, reason: because };
  }
  if (promoted && promoted.present) {
    total += promoted.bytes;
    cells[promote!] = { loc: 'sub', code: 'typed', promoted: true, detail: 'Flattened into its own relational column at load time' };
  }
  total += rest;
  const reads: Record<string, Read> = Object.create(null);
  for (const p of allPaths) {
    if (p === promote && promoted) reads[p] = { bytes: promoted.bytes, loc: 'sub', how: 'the relational column' };
    else if (extracted.has(p)) reads[p] = { bytes: presence(rows) + stats.get(p)!.bytes, loc: 'sub', how: 'the extracted column' };
    else if (stats.has(p)) reads[p] = { bytes: rest, loc: 'blob', how: 'the entire rest of the VARIANT, walking every row' };
    else reads[p] = { bytes: 0, loc: 'skip', how: 'skipped: the path Bloom filter shows the path is not in this partition' };
  }
  return { index, first, rows, used: extracted.size, limit, atLimit: eligible.length > limit, cells, blobBytes: rest, blobPaths, totalBytes: total, rawBytes: docs.reduce((s, d) => s + d.raw, 0), reads };
}

/* ---- Spark: port of InferVariantShreddingSchema (branch-4.1) */

export type SType =
  | { t: 'null' | 'boolean' | 'string' | 'double' | 'long' | 'variant' }
  | { t: 'decimal'; p: number; s: number }
  | { t: 'array'; e: SType }
  | { t: 'struct'; f: { name: string; type: SType; count: number }[] };

function schemaOf(v: PV): SType {
  switch (v.k) {
    case 'obj': {
      const f = v.keys.map((name, j) => ({ name, type: schemaOf(v.vals[j]), count: 1 }));
      f.sort((a, b) => (a.name < b.name ? -1 : a.name > b.name ? 1 : 0));
      return { t: 'struct', f };
    }
    case 'arr':
      return { t: 'array', e: v.items.reduce<SType>((acc, x) => mergeSchema(acc, schemaOf(x)), { t: 'null' }) };
    case 'null':
      return { t: 'null' };
    case 'bool':
      return { t: 'boolean' };
    case 'int':
      return { t: 'decimal', p: v.digits, s: 0 };
    case 'dec':
      return { t: 'decimal', p: v.prec, s: v.scale };
    case 'dbl':
      return { t: 'double' };
    case 'str':
      return { t: 'string' };
  }
}

function mergeDecimal(a: { p: number; s: number }, b: { p: number; s: number }): SType {
  const s = Math.max(a.s, b.s);
  const range = Math.max(a.p - a.s, b.p - b.s);
  return range + s > 38 ? { t: 'variant' } : { t: 'decimal', p: range + s, s };
}

export function mergeSchema(a: SType, b: SType): SType {
  if (a.t === 'null') return b;
  if (b.t === 'null') return a;
  if (a.t === 'decimal' && b.t === 'decimal') return mergeDecimal(a, b);
  if (a.t === 'decimal' && b.t === 'long') return a.s === 0 && a.p <= 18 ? { t: 'long' } : mergeDecimal(a, { p: 19, s: 0 });
  if (a.t === 'long' && b.t === 'decimal') return mergeSchema(b, a);
  if (a.t === 'struct' && b.t === 'struct') {
    const out: { name: string; type: SType; count: number }[] = [];
    let i = 0;
    let j = 0;
    while (i < a.f.length && j < b.f.length && out.length < 1000) {
      const x = a.f[i];
      const y = b.f[j];
      if (x.name === y.name) {
        out.push({ name: x.name, type: mergeSchema(x.type, y.type), count: x.count + y.count });
        i++;
        j++;
      } else if (x.name < y.name) {
        out.push(x);
        i++;
      } else {
        out.push(y);
        j++;
      }
    }
    while (i < a.f.length && out.length < 1000) out.push(a.f[i++]);
    while (j < b.f.length && out.length < 1000) out.push(b.f[j++]);
    return { t: 'struct', f: out };
  }
  if (a.t === 'array' && b.t === 'array') return { t: 'array', e: mergeSchema(a.e, b.e) };
  if (a.t === b.t && a.t !== 'variant') return a;
  return { t: 'variant' };
}

export function finalizeSchema(dt: SType, minCard: number, budget: { r: number }): SType {
  budget.r -= 1; // every field uses a value column
  if (budget.r <= 0) return { t: 'variant' };
  switch (dt.t) {
    case 'struct': {
      const f: { name: string; type: SType; count: number }[] = [];
      for (const field of dt.f.filter((x) => x.count >= minCard)) {
        if (budget.r > 0) f.push({ ...field, type: finalizeSchema(field.type, minCard, budget) });
      }
      return f.length ? { t: 'struct', f } : { t: 'variant' };
    }
    case 'array':
      return { t: 'array', e: finalizeSchema(dt.e, minCard, budget) };
    case 'long':
      budget.r -= 1;
      return dt;
    case 'decimal':
      budget.r -= 1;
      return dt.s === 0 && dt.p <= 18 ? { t: 'long' } : { t: 'decimal', p: dt.p <= 18 ? 18 : 38, s: dt.s };
    case 'variant':
    case 'null':
      return { t: 'variant' };
    default:
      budget.r -= 1;
      return dt;
  }
}

export function inferSparkSchema(buffer: PV[], maxWidth: number) {
  const simple = buffer.reduce<SType>((acc, v) => mergeSchema(acc, schemaOf(v)), { t: 'null' });
  const minCard = Math.floor((buffer.length + 9) / 10);
  const budget = { r: maxWidth };
  const schema = finalizeSchema(simple, minCard, budget);
  return { schema, simple, minCard, used: Math.min(maxWidth, maxWidth - budget.r), exhausted: budget.r <= 0 };
}

function fits(v: PV, t: SType): boolean {
  switch (t.t) {
    case 'long':
      return v.k === 'int';
    case 'decimal':
      return (v.k === 'int' && v.digits <= t.p - t.s) || (v.k === 'dec' && v.scale <= t.s && v.prec - v.scale <= t.p - t.s);
    case 'double':
      return v.k === 'dbl';
    case 'string':
      return v.k === 'str';
    case 'boolean':
      return v.k === 'bool';
    case 'array':
      return v.k === 'arr';
    default:
      return false;
  }
}

const sparkCode = (t: SType) => (t.t === 'decimal' ? 'dec' : t.t === 'string' ? 'str' : t.t === 'boolean' ? 'bool' : t.t === 'double' ? 'dbl' : t.t === 'array' ? 'arr' : t.t);

function shredRow(v: PV, t: SType, key: string, cols: Map<string, number>) {
  const add = (c: string, b: number) => cols.set(c, (cols.get(c) ?? 0) + b);
  if (t.t === 'variant') return add(`${key}#value`, variantSize(v));
  if (t.t === 'struct') {
    if (v.k !== 'obj') return add(`${key}#value`, variantSize(v));
    const names = new Set(t.f.map((f) => f.name));
    const rk: string[] = [];
    const rv: PV[] = [];
    v.keys.forEach((k, j) => {
      if (!names.has(k)) {
        rk.push(k);
        rv.push(v.vals[j]);
      }
    });
    if (rk.length) add(`${key}#value`, variantSize({ k: 'obj', keys: rk, vals: rv }));
    for (const f of t.f) {
      const j = v.keys.indexOf(f.name);
      if (j >= 0) shredRow(v.vals[j], f.type, key ? `${key}.${f.name}` : f.name, cols);
    }
    return;
  }
  if (fits(v, t)) add(`${key}#typed`, typedSize(v, 4));
  else add(`${key}#value`, variantSize(v));
}

function schemaColumns(t: SType, key: string, out: string[]) {
  out.push(`${key}#value`);
  if (t.t === 'struct') t.f.forEach((f) => schemaColumns(f.type, key ? `${key}.${f.name}` : f.name, out));
  else if (t.t !== 'variant') out.push(`${key}#typed`);
  return out;
}

export function sparkLocate(schema: SType, path: string) {
  let node = schema;
  let key = '';
  const segs = path.split('.');
  let d = 0;
  while (d < segs.length && node.t === 'struct') {
    const f = node.f.find((x) => x.name === segs[d]);
    if (!f) break;
    node = f.type;
    key = key ? `${key}.${segs[d]}` : segs[d];
    d++;
  }
  const full = d === segs.length;
  if (full && node.t !== 'struct' && node.t !== 'variant') return { loc: 'sub' as Loc, cols: [`${key}#typed`, `${key}#value`], meta: false, code: sparkCode(node), key };
  if (full && node.t === 'variant') return { loc: 'multi' as Loc, cols: [`${key}#value`], meta: true, code: 'value', key };
  return { loc: 'blob' as Loc, cols: [`${key}#value`], meta: true, code: 'value', key: key || '(root)' };
}

function findType(t: SType, segs: string[]): SType | null {
  let node: SType = t;
  for (const seg of segs) {
    if (node.t !== 'struct') return null;
    const f = node.f.find((x) => x.name === seg);
    if (!f) return null;
    node = f.type;
  }
  return node;
}

function countWithKey(buffer: PV[], segs: string[]) {
  let n = 0;
  for (const v of buffer) {
    let node: PV | undefined = v;
    for (const seg of segs) node = node && node.k === 'obj' ? node.vals[node.keys.indexOf(seg)] : undefined;
    if (node) n++;
  }
  return n;
}

/** Why a path did not get its own typed_value in this file. */
function sparkReason(simple: SType, schema: SType, buffer: PV[], minCard: number, path: string) {
  const segs = path.split('.');
  for (let d = 1; d <= segs.length; d++) {
    const prefix = segs.slice(0, d);
    const name = prefix.join('.');
    const fin = findType(schema, prefix);
    if (fin && fin.t !== 'variant' && d < segs.length) continue;
    if (fin && fin.t === 'variant') {
      const s = findType(simple, prefix);
      return s && s.t === 'variant' ? `the first ${buffer.length} rows disagreed on the type of ${name}, so it got only an untyped value column` : `the maxSchemaWidth budget ran out at ${name} (fields are taken alphabetically), so it got only a value column`;
    }
    const seen = countWithKey(buffer, prefix);
    if (seen === 0) return `${name} first appears after the ${buffer.length} rows the writer buffered to choose this file's schema`;
    if (seen < minCard) return `${name} is in only ${seen} of the ${buffer.length} buffered rows, under the 10% threshold`;
    return `the maxSchemaWidth budget ran out before ${name} (fields are taken alphabetically)`;
  }
  return 'its values do not match the shredded type';
}

function sparkBlock(docs: Doc[], index: number, first: number, limit: number, allPaths: string[], promote: string | null, bufferRows: number): Block {
  const rows = docs.length;
  const segs = promote ? promote.split('.') : [];
  const pvs = docs.map((d) => (promote ? withoutPath(d.pv, segs) : d.pv));
  const buffer = pvs.slice(0, bufferRows);
  const { schema, simple, minCard, used, exhausted } = inferSparkSchema(buffer, limit);
  const cols = new Map<string, number>();
  for (const v of pvs) shredRow(v, schema, '', cols);
  const meta = metadataColumn(pvs);
  const allCols = schemaColumns(schema, '', []);
  const colBytes = (c: string) => (cols.get(c) ? cols.get(c)! + presence(rows) : 0); // an all-null column compresses to nothing
  const promoted = promote ? promotedColumn(docs, promote, 4) : null;
  let total = meta + allCols.reduce((s, c) => s + colBytes(c), 0);
  if (promoted && promoted.present) total += promoted.bytes;

  const present = new Map<string, number>();
  for (const d of docs) for (const l of d.leaves) present.set(l.path, (present.get(l.path) ?? 0) + 1);
  const cells: Record<string, Cell> = Object.create(null);
  const blobPaths: string[] = [];
  const reads: Record<string, Read> = Object.create(null);
  for (const p of allPaths) {
    if (p === promote && promoted) {
      if (promoted.present) cells[p] = { loc: 'sub', code: 'typed', promoted: true, detail: 'Extracted into its own top-level typed column' };
      reads[p] = { bytes: promoted.bytes, loc: 'sub', how: 'its own top-level column' };
      continue;
    }
    const at = sparkLocate(schema, p);
    const bytes = at.cols.reduce((s, c) => s + colBytes(c), 0) + (at.meta && colBytes(at.cols[0]) > 0 ? meta : 0);
    const fallback = at.loc === 'sub' ? cols.get(`${at.key}#value`) ?? 0 : 0;
    reads[p] = {
      bytes,
      loc: at.loc,
      how: at.loc === 'sub' ? `typed_value${fallback ? ' plus the field’s value column (type mismatches and JSON nulls)' : ''}` : at.loc === 'multi' ? `the field’s untyped value column` : `the ${at.key === '(root)' ? 'top-level' : `${at.key}`} residual value column`,
    };
    if (present.has(p)) {
      if (at.loc === 'blob') blobPaths.push(p);
      cells[p] = {
        reason: at.loc === 'sub' ? undefined : sparkReason(simple, schema, buffer, minCard, p),
        loc: at.loc,
        code: at.loc === 'sub' ? at.code : at.loc === 'multi' ? 'value' : 'resid',
        detail:
          at.loc === 'sub'
            ? `Shredded as typed_value (${at.code})${fallback ? `; ${fmtBytes(fallback)} of mismatched values or JSON nulls went to its value column` : ''}`
            : at.loc === 'multi'
              ? 'Shredded as an untyped value column: the buffered rows disagreed on its type'
              : `Not in this file’s shredding schema: stored inside the ${at.key === '(root)' ? 'top-level' : at.key} residual value`,
      };
    }
  }
  const blobBytes = colBytes('#value');
  return { index, first, rows, used, limit, atLimit: exhausted, cells, blobBytes, blobPaths, totalBytes: total, rawBytes: docs.reduce((s, d) => s + d.raw, 0), reads };
}

export type LabResult = {
  blocks: Block[];
  paths: string[];
  counts: Record<string, number>;
  rows: number;
  rawBytes: number;
  perBlock: number;
};

export function runLab(docs: Doc[], engine: Engine, limit: number, perBlockWanted: number, opts: { compact?: boolean; serialization?: Serialization; promote?: string | null; bufferRows?: number } = {}): LabResult {
  const perBlock = opts.compact ? Math.max(1, docs.length) : Math.max(perBlockWanted, Math.ceil(docs.length / MAX_BLOCKS), 1);
  const counts: Record<string, number> = Object.create(null);
  const nonNull: Record<string, number> = Object.create(null);
  const paths: string[] = [];
  const firstSeen = new Map<string, number>();
  for (const d of docs)
    for (const l of d.leaves) {
      if (!firstSeen.has(l.path)) {
        counts[l.path] = 0;
        nonNull[l.path] = 0;
        firstSeen.set(l.path, paths.length);
        paths.push(l.path);
      }
      counts[l.path]++;
      if (l.v.k !== 'null') nonNull[l.path]++;
    }
  const promote = opts.promote && firstSeen.has(opts.promote) ? opts.promote : null;
  const promoteType = promote && engine === 'clickhouse' ? declaredType(docs, promote) : '';
  // A merge keeps the paths with the most non-null values as sub-columns.
  const mergeOrder =
    engine === 'clickhouse' && opts.compact
      ? paths.filter((p) => p !== promote && nonNull[p] > 0).sort((x, y) => nonNull[y] - nonNull[x] || firstSeen.get(x)! - firstSeen.get(y)!)
      : null;
  const blocks: Block[] = [];
  for (let b = 0, start = 0; start < docs.length; b++, start += perBlock) {
    const slice = docs.slice(start, start + perBlock);
    if (engine === 'clickhouse') {
      blocks.push(clickhouseBlock(slice, b, start, limit, paths, opts.serialization ?? 'map', promote, promoteType, mergeOrder));
    } else if (engine === 'snowflake') blocks.push(snowflakeBlock(slice, b, start, limit, paths, promote));
    else blocks.push(sparkBlock(slice, b, start, limit, paths, promote, opts.bufferRows ?? SPARK_BUFFER_ROWS));
  }
  return { blocks, paths, counts, rows: docs.length, rawBytes: docs.reduce((s, d) => s + d.raw, 0), perBlock };
}

export function queryCost(r: LabResult, path: string) {
  const byLoc = { sub: 0, multi: 0, blob: 0 };
  let skipped = 0;
  let subBlocks = 0;
  let multiBlocks = 0;
  let blobBlocks = 0;
  for (const b of r.blocks) {
    const rd = b.reads[path];
    if (!rd) continue;
    if (rd.loc === 'skip') skipped++;
    else byLoc[rd.loc] += rd.bytes;
    const c = b.cells[path];
    if (c?.loc === 'sub') subBlocks++;
    if (c?.loc === 'multi') multiBlocks++;
    if (c?.loc === 'blob') blobBlocks++;
  }
  const bytes = byLoc.sub + byLoc.multi + byLoc.blob;
  return { bytes, byLoc, skipped, subBlocks, multiBlocks, blobBlocks, rowsPerSec: bytes > 0 ? (r.rows * READ_RATE) / bytes : Infinity };
}

/* ------------------------------------------------------------------ presets */

type Obj = Record<string, unknown>;
const lines = (docs: Obj[]) => docs.map((d) => JSON.stringify(d)).join('\n');
const cents = (rng: () => number) => {
  let c = 11 + Math.floor(rng() * 88);
  if (c % 10 === 0) c++;
  return c;
};

export const PRESETS: { value: string; label: string; query: string; text: () => string }[] = [
  {
    value: 'drift',
    label: 'Events whose schema drifts',
    query: 'amount',
    text: () => {
      const rng = makeRng(7);
      const events = ['view', 'click', 'add_to_cart', 'purchase'];
      const out: Obj[] = [];
      for (let i = 0; i < 48; i++) {
        const ev = events[Math.floor(rng() * 4)];
        const d: Obj = { ts: 1726000000 + i * 37 };
        if (i < 24) d.user_id = 1000 + Math.floor(rng() * 9000);
        else d.userId = `u-${1000 + Math.floor(rng() * 9000)}`;
        d.event = ev;
        d.page = `/p/${Math.floor(rng() * 500)}`;
        if (i >= 12) d.device = { os: rng() < 0.5 ? 'ios' : 'android', app: `5.${Math.floor(i / 12)}.${Math.floor(rng() * 10)}` };
        if (ev === 'purchase' || ev === 'add_to_cart') {
          const price = `${5 + Math.floor(rng() * 90)}.${cents(rng)}`;
          d.amount = i >= 36 && rng() < 0.5 ? price : Number(price);
        }
        if (i >= 24 && rng() < 0.6) d.experiment = { id: 'checkout-v3', arm: rng() < 0.5 ? 'A' : 'B' };
        if (i >= 36) d.geo = { country: rng() < 0.7 ? 'US' : 'DE', city: rng() < 0.5 ? 'Seattle' : 'Berlin' };
        out.push(d);
      }
      return lines(out);
    },
  },
  {
    value: 'explosion',
    label: 'Path explosion: user-defined keys',
    query: 'user.plan',
    text: () => {
      const rng = makeRng(11);
      const out: Obj[] = [];
      for (let i = 0; i < 48; i++) {
        const props: Obj = {};
        while (Object.keys(props).length < 5) props[`flag_${Math.floor(rng() * 40)}`] = rng() < 0.5;
        const d: Obj = { ts: 1726000000 + i * 11, event: rng() < 0.7 ? 'feature_check' : 'render', props };
        if (rng() < 0.9) d.user = { plan: ['free', 'pro', 'team'][Math.floor(rng() * 3)] };
        out.push(d);
      }
      return lines(out);
    },
  },
  {
    value: 'nulls',
    label: 'JSON nulls and mixed types',
    query: 'zip',
    text: () => {
      const rng = makeRng(5);
      const out: Obj[] = [];
      for (let i = 0; i < 36; i++) {
        const d: Obj = { order_id: 500000 + i, total: Number(`${10 + Math.floor(rng() * 200)}.${cents(rng)}`) };
        const r = rng();
        d.zip = r < 0.65 ? 10000 + Math.floor(rng() * 89999) : `0${1000 + Math.floor(rng() * 8999)}`;
        const c = rng();
        if (c < 0.3) d.coupon = null;
        else if (c < 0.6) d.coupon = 'SAVE10';
        d.gift = rng() < 0.2;
        out.push(d);
      }
      return lines(out);
    },
  },
  {
    value: 'stable',
    label: 'Stable log lines',
    query: 'latency_ms',
    text: () => {
      const rng = makeRng(3);
      const out: Obj[] = [];
      for (let i = 0; i < 36; i++) {
        out.push({
          ts: 1726000000 + i * 3,
          host: `web-${1 + Math.floor(rng() * 4)}`,
          level: rng() < 0.9 ? 'info' : 'warn',
          path: ['/api/cart', '/api/search', '/health'][Math.floor(rng() * 3)],
          latency_ms: 2 + Math.floor(rng() * 400),
        });
      }
      return lines(out);
    },
  },
];

const DEFAULT_LIMIT: Record<Engine, number> = { clickhouse: 6, snowflake: 6, spark: 20 };
const LIMIT_RANGE: Record<Engine, [number, number]> = { clickhouse: [1, 40], snowflake: [1, 40], spark: [2, 80] };

/* ------------------------------------------------------------------ component */

const LOC_COLOR: Record<Loc, string> = { sub: 'var(--viz-1)', blob: 'var(--viz-2)', multi: 'var(--viz-3)' };
const MAX_ROWS = 28;
/** fmtBytes without the space, for narrow block columns. */
const tightBytes = (b: number) => fmtBytes(b).replace(' ', '');
const MAX_PATH_OPTIONS = 200; // thousands of <option> nodes stall the page's mutation observers

function Matrix({ r, engine, query, onPick }: { r: LabResult; engine: Engine; query: string; onPick: (p: string) => void }) {
  const tip = useTip();
  const info = ENGINE_INFO[engine];
  const W = 680;
  const LABEL = 136;
  const BLOCKS_W = 330;
  const BAR_X = LABEL + BLOCKS_W + 14;
  const BAR_W = W - BAR_X - 58;
  const n = r.blocks.length;
  const cw = BLOCKS_W / Math.max(1, n);
  const ROW = 17;
  const TOP = 58;

  let shown = r.paths;
  let hidden = 0;
  if (shown.length > MAX_ROWS) {
    const keep = new Set([...r.paths].sort((a, b) => r.counts[b] - r.counts[a]).slice(0, MAX_ROWS - 1));
    keep.add(query);
    shown = r.paths.filter((p) => keep.has(p));
    hidden = r.paths.length - shown.length;
  }
  const costs = shown.map((p) => queryCost(r, p));
  const domain = Math.max(r.rawBytes, ...costs.map((c) => c.bytes), 1);
  const bw = (b: number) => (b / domain) * BAR_W;
  const H = TOP + shown.length * ROW + (hidden ? ROW : 0) + 40;
  const rawX = BAR_X + bw(r.rawBytes);

  return (
    <svg viewBox={`0 0 ${W} ${H}`} width={W} height={H} style={{ minWidth: 560 }} role="img" aria-label={`Where each JSON path is stored in each ${info.block}, and the bytes a filter on it reads`}>
      <text x={8} y={14} fontSize={11} fill="var(--viz-ink-2)">
        path (click to filter)
      </text>
      <text x={BAR_X} y={14} fontSize={11} fill="var(--viz-ink-2)">
        bytes a filter reads
      </text>
      <line x1={rawX} x2={rawX} y1={TOP - 10} y2={TOP + shown.length * ROW} stroke="var(--viz-ink-2)" strokeDasharray="3 3" />
      <text x={Math.min(rawX, W - 4)} y={TOP - 14} fontSize={10} textAnchor={rawX > W - 60 ? 'end' : 'middle'} fill="var(--viz-ink-2)">
        raw JSON {fmtBytes(r.rawBytes)}
      </text>
      {r.blocks.map((b) => {
        const x = LABEL + b.index * cw;
        const frac = Math.min(1, b.used / Math.max(1, b.limit));
        return (
          <g key={b.index}>
            <text x={x + cw / 2} y={14} fontSize={10} textAnchor="middle" fill="var(--viz-ink)">
              {n > 6 ? `${b.index + 1}` : `${info.block} ${b.index + 1}`}
            </text>
            <text x={x + cw / 2} y={27} fontSize={9} textAnchor="middle" fill="var(--viz-ink-2)">
              {b.used}/{b.limit}
            </text>
            <rect x={x + 4} y={33} width={cw - 8} height={6} rx={2} fill="var(--viz-grid)" />
            <rect x={x + 4} y={33} width={Math.max(0, (cw - 8) * frac)} height={6} rx={2} fill={b.atLimit ? 'var(--viz-critical)' : 'var(--viz-ink-muted)'} />
          </g>
        );
      })}
      {shown.map((p, i) => {
        const y = TOP + i * ROW;
        const c = costs[i];
        const selected = p === query;
        let x = BAR_X;
        return (
          <g
            key={p}
            style={{ cursor: 'pointer' }}
            onClick={() => onPick(p)}
            onKeyDown={(e) => {
              if (e.key === 'Enter' || e.key === ' ') {
                e.preventDefault();
                onPick(p);
              }
            }}
            role="button"
            aria-label={`Filter on ${p}`}
            {...tip(
              <>
                <strong>{p}</strong> — in {fmtNum(r.counts[p])} of {fmtNum(r.rows)} documents. A filter reads {fmtBytes(c.bytes)}
                {c.skipped ? `, skipping ${c.skipped} ${info.block}${c.skipped === 1 ? '' : 's'}` : ''}.
              </>,
            )}
          >
            <rect x={2} y={y - 1} width={W - 4} height={ROW - 1} rx={3} fill={selected ? 'var(--viz-plane)' : 'transparent'} stroke={selected ? 'var(--viz-ink)' : 'none'} strokeDasharray="4 2" />
            <text x={8} y={y + 11} fontSize={11} fill="var(--viz-ink)" fontWeight={selected ? 600 : 400}>
              {p.length > 21 ? `${p.slice(0, 20)}…` : p}
            </text>
            {r.blocks.map((b) => {
              const cell = b.cells[p];
              const bx = LABEL + b.index * cw;
              if (!cell)
                return (
                  <circle key={b.index} cx={bx + cw / 2} cy={y + 7} r={1.5} fill="var(--viz-ink-muted)">
                    <title>{`absent from this ${info.block}`}</title>
                  </circle>
                );
              return (
                <g key={b.index}>
                  <rect
                    x={bx + 2}
                    y={y + 1}
                    width={cw - 4}
                    height={ROW - 4}
                    rx={2}
                    fill="var(--viz-surface)"
                    stroke={LOC_COLOR[cell.loc]}
                    strokeWidth={cell.loc === 'sub' ? 2 : 1.5}
                    strokeDasharray={cell.loc === 'blob' ? '3 2' : undefined}
                  >
                    <title>{cell.detail}</title>
                  </rect>
                  {cw >= 30 ? (
                    <text x={bx + cw / 2} y={y + 11} fontSize={9} textAnchor="middle" fill="var(--viz-ink)">
                      {cw < 40 && cell.code.length > 5 ? cell.code.slice(0, 4) : cell.code}
                    </text>
                  ) : null}
                </g>
              );
            })}
            {(['sub', 'multi', 'blob'] as Loc[]).map((loc) => {
              const w = bw(c.byLoc[loc]);
              if (w <= 0) return null;
              const el = <rect key={loc} x={x} y={y + 2} width={Math.max(1.5, w)} height={ROW - 6} fill={LOC_COLOR[loc]} />;
              x += Math.max(1.5, w);
              return el;
            })}
            <text x={x + 4} y={y + 11} fontSize={10} fill="var(--viz-ink-2)">
              {c.bytes === 0 ? (c.skipped ? 'skipped' : '0 B') : fmtBytes(c.bytes)}
            </text>
          </g>
        );
      })}
      {hidden ? (
        <text x={8} y={TOP + shown.length * ROW + 11} fontSize={10} fill="var(--viz-ink-2)">
          + {hidden} rarer paths (in the table)
        </text>
      ) : null}
      <text x={8} y={H - 20} fontSize={10} fill="var(--viz-ink-2)">
        {info.blob}
      </text>
      <text x={8} y={H - 7} fontSize={10} fill="var(--viz-ink-2)">
        {info.block} total
      </text>
      {r.blocks.map((b) => (
        <g key={b.index}>
          <text x={LABEL + b.index * cw + cw / 2} y={H - 20} fontSize={cw < 44 ? 8 : 9} textAnchor="middle" fill="var(--viz-ink)">
            {cw < 44 ? tightBytes(b.blobBytes) : fmtBytes(b.blobBytes)}
          </text>
          <text x={LABEL + b.index * cw + cw / 2} y={H - 7} fontSize={cw < 44 ? 8 : 9} textAnchor="middle" fill="var(--viz-ink-2)">
            {cw < 44 ? tightBytes(b.totalBytes) : fmtBytes(b.totalBytes)}
          </text>
        </g>
      ))}
    </svg>
  );
}

export default function JsonShreddingDriftLab() {
  const [engine, setEngine] = useState<Engine>('clickhouse');
  const [preset, setPreset] = useState('drift');
  const [text, setText] = useState(() => PRESETS[0].text());
  const [limits, setLimits] = useState<Record<Engine, number>>(DEFAULT_LIMIT);
  const [perBlock, setPerBlock] = useState(12);
  const [query, setQuery] = useState(PRESETS[0].query);
  const [serialization, setSerialization] = useState<Serialization>('map');
  const [compact, setCompact] = useState(false);
  const [promote, setPromote] = useState(false);

  const deferredText = useDeferredValue(text);
  const parsed = useMemo(() => parseDocs(deferredText), [deferredText]);
  const limit = limits[engine];
  const info = ENGINE_INFO[engine];
  const q = useMemo(() => {
    let first = '';
    for (const d of parsed.docs)
      for (const l of d.leaves) {
        if (l.path === query) return query;
        first ||= l.path;
      }
    return first;
  }, [parsed, query]);
  const r = useMemo(
    () => runLab(parsed.docs, engine, limit, perBlock, { compact, serialization, promote: promote ? q : null }),
    [parsed, engine, limit, perBlock, compact, serialization, promote, q],
  );
  const cost = queryCost(r, q);
  const typedEquivalent = useMemo(() => (q ? queryCost(runLab(parsed.docs, engine, limit, perBlock, { compact, serialization, promote: q }), q).bytes : 0), [parsed, engine, limit, perBlock, compact, serialization, q]);
  const pathOptions = useMemo(() => {
    if (r.paths.length <= MAX_PATH_OPTIONS) return r.paths;
    const keep = new Set([...r.paths].sort((a, b) => r.counts[b] - r.counts[a]).slice(0, MAX_PATH_OPTIONS - 1));
    keep.add(q);
    return r.paths.filter((p) => keep.has(p));
  }, [r, q]);
  const penalty = typedEquivalent > 0 ? Math.max(1, cost.bytes / typedEquivalent) : 1;
  const blobTotal = r.blocks.reduce((s, b) => s + b.blobBytes, 0);
  const total = r.blocks.reduce((s, b) => s + b.totalBytes, 0);
  const maxUsed = Math.max(0, ...r.blocks.map((b) => b.used));
  const blocksAtLimit = r.blocks.filter((b) => b.atLimit).length;

  const choosePreset = (v: string) => {
    const p = PRESETS.find((x) => x.value === v) ?? PRESETS[0];
    setPreset(p.value);
    setText(p.text());
    setQuery(p.query);
    setPromote(false);
  };

  const reasons = r.blocks.flatMap((b) => (b.cells[q]?.reason ? [{ block: b.index + 1, reason: b.cells[q].reason! }] : []));

  return (
    <VizPanel
      title="Shredding drifting JSON into sub-columns"
      subtitle={`Each ${info.block} infers its own sub-columns from the documents it holds. Paths that miss the cut spill into the ${info.blob}. Click a path to filter on it and compare the bytes read.`}
      controls={
        <>
          <Segmented
            label="Engine"
            value={engine}
            onChange={setEngine}
            options={[
              { value: 'clickhouse', label: 'ClickHouse JSON' },
              { value: 'snowflake', label: 'Snowflake VARIANT' },
              { value: 'spark', label: 'Spark → Parquet variant' },
            ]}
          />
          <Choice label="Documents" value={preset} onChange={choosePreset} options={PRESETS.map((p) => ({ value: p.value, label: p.label }))} />
          <Slider
            label={`${info.limitName} (real default ${fmtNum(info.realDefault)})`}
            min={LIMIT_RANGE[engine][0]}
            max={LIMIT_RANGE[engine][1]}
            value={limit}
            onChange={(v) => setLimits((l) => ({ ...l, [engine]: v }))}
          />
          <Slider
            label={`Documents per ${info.block}${!compact && r.perBlock !== perBlock ? ` (${r.perBlock} used, at most ${MAX_BLOCKS} ${info.block}s)` : ''}`}
            min={4}
            max={60}
            value={perBlock}
            onChange={setPerBlock}
            disabled={compact}
          />
          <Choice label="Filter on path" value={q} onChange={(v) => setQuery(v)} options={pathOptions.map((p) => ({ value: p, label: p }))} />
          {engine === 'clickhouse' ? (
            <Choice
              label="Shared data serialization"
              value={serialization}
              onChange={setSerialization}
              options={[
                { value: 'map', label: 'map' },
                { value: 'map_with_buckets', label: `map_with_buckets (${CH_BUCKETS})` },
                { value: 'advanced', label: 'advanced' },
              ]}
            />
          ) : null}
          <Check label={engine === 'clickhouse' ? 'Merge all parts into one' : engine === 'snowflake' ? 'Recluster into one partition' : 'Rewrite as one file'} checked={compact} onChange={setCompact} />
          <Check label="Promote the filter path to a typed column" checked={promote} onChange={setPromote} />
        </>
      }
      legend={
        <Legend
          items={[
            { label: 'Typed sub-column (one type)', color: 'var(--viz-1)' },
            { label: engine === 'spark' ? 'Own column, but untyped value' : 'Own sub-column, several types', color: 'var(--viz-3)' },
            { label: `In the ${info.blob} (dashed)`, color: 'var(--viz-2)' },
            { label: 'Raw JSON text of every document', color: 'var(--viz-ink-2)', shape: 'line' },
            { label: 'Limit reached in this block', color: 'var(--viz-critical)' },
          ]}
        />
      }
      stats={
        <Stats
          items={[
            { label: 'Distinct paths', value: fmtNum(r.paths.length) },
            {
              label: `Most ${info.unit} in a ${info.block}`,
              value: `${fmtNum(maxUsed)} / ${fmtNum(limit)}`,
              hint: engine === 'spark' ? 'Budget spent: one column per shredded field plus one per typed_value, including the top-level value column.' : 'Sub-columns in the fullest block, against the limit.',
            },
            { label: `Bytes in ${info.blob}`, value: total ? `${fmtNum((100 * blobTotal) / total)}%` : '—', hint: 'Share of all stored bytes (uncompressed model).' },
            { label: `Filter on ${q || '—'}`, value: fmtBytes(cost.bytes), hint: `Bytes read across all ${info.block}s. Raw JSON text is ${fmtBytes(r.rawBytes)}.` },
            { label: 'vs. its own typed column', value: `${fmtNum(penalty, 1)}×`, hint: 'Bytes read divided by the bytes of a typed column holding just this path.' },
            { label: 'Rows/s at 1 GB/s', value: cost.bytes ? fmtNum(cost.rowsPerSec) : 'all skipped', hint: 'A normalisation, not a benchmark: rows scanned per second if every byte read costs the same. Blobs cost more per byte to walk, so the real gap is wider.' },
          ]}
        />
      }
      note={
        <Note>
          {parsed.error ? (
            <>
              <strong>Could not parse: </strong>
              {parsed.error}. Showing the {parsed.docs.length} documents before it.{' '}
            </>
          ) : null}
          {q ? (
            <>
              <strong>
                {q} is a sub-column in {cost.subBlocks + cost.multiBlocks} of {r.blocks.length} {info.block}
                {r.blocks.length === 1 ? '' : 's'}
                {cost.multiBlocks ? (engine === 'spark' ? ` (an untyped value column in ${cost.multiBlocks})` : ` (holding several types in ${cost.multiBlocks})`) : ''}
                {cost.blobBlocks ? ` and sits in the ${info.blob} in ${cost.blobBlocks}` : ''}.
              </strong>{' '}
              A filter on it reads {fmtBytes(cost.bytes)}
              {penalty >= 1.05 ? ` — ${fmtNum(penalty, 1)}× what a typed column of the same values would cost` : ''}
              {cost.skipped ? `; ${cost.skipped} ${info.block}${cost.skipped === 1 ? ' is' : 's are'} skipped because the path is not in them` : ''}.{' '}
              {reasons.length ? `Why, in ${info.block} ${reasons[0].block}: ${reasons[0].reason}.` : ''}
            </>
          ) : (
            'Paste JSON objects, one per line.'
          )}
        </Note>
      }
      table={
        <table className="viz-table">
          <thead>
            <tr>
              <th>{info.block}</th>
              <th>Rows</th>
              <th>Own columns (path: type)</th>
              <th>In the {info.blob}</th>
              <th>{info.blob} bytes</th>
              <th>Filter on {q}</th>
            </tr>
          </thead>
          <tbody>
            {r.blocks.map((b) => (
              <tr key={b.index}>
                <td>{b.index + 1}</td>
                <td>
                  {b.first + 1}–{b.first + b.rows}
                </td>
                <td>
                  {Object.entries(b.cells)
                    .filter(([, c]) => c.loc !== 'blob')
                    .map(([p, c]) => `${p}: ${c.code}`)
                    .join(', ') || '—'}
                </td>
                <td>{b.blobPaths.join(', ') || '—'}</td>
                <td>{fmtBytes(b.blobBytes)}</td>
                <td>{b.reads[q] ? `${b.reads[q].loc === 'skip' ? 'skipped' : fmtBytes(b.reads[q].bytes)} — ${b.reads[q].how}` : '—'}</td>
              </tr>
            ))}
          </tbody>
        </table>
      }
    >
      <label className="viz-control" style={{ marginBottom: '0.5rem' }}>
        <span>
          Documents, one JSON object per line ({fmtNum(parsed.docs.length)} parsed, {fmtBytes(r.rawBytes)} of text{parsed.truncated ? `, first ${MAX_DOCS} used` : ''})
        </span>
        <textarea
          value={text}
          onChange={(e) => setText(e.currentTarget.value)}
          rows={5}
          spellCheck={false}
          aria-label="JSON documents, one per line"
          style={{ font: 'inherit', fontFamily: 'var(--sl-font-mono, monospace)', fontSize: '0.72rem', background: 'var(--viz-plane)', color: 'var(--viz-ink)', border: '1px solid var(--viz-border)', borderRadius: 6, padding: 6, width: '100%', boxSizing: 'border-box', whiteSpace: 'pre', overflowX: 'auto' }}
        />
      </label>
      <p style={{ margin: '0 0 0.4rem', fontSize: '0.75rem', color: 'var(--viz-ink-2)' }}>
        Lab scale: the limit is shrunk so a few dozen documents reach it
        {engine === 'spark' ? `; Spark infers each file's schema from its first ${SPARK_BUFFER_ROWS} rows (real: ${fmtNum(REAL_SPARK_BUFFER_ROWS)} rows or 64 MB)` : ''}. Byte counts are uncompressed estimates.
      </p>
      <TooltipHost>
        <Matrix r={r} engine={engine} query={q} onPick={setQuery} />
      </TooltipHost>
    </VizPanel>
  );
}
