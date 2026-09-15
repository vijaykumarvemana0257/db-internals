/**
 * Model behind ColumnEncodingChainLab: a column chunk encoded by a stack of lightweight encodings,
 * the exact fields each step writes, and what a predicate has to touch to run on the result.
 *
 * Faithful to a named source:
 *  - `hybrid`   Parquet RLE/bit-packing hybrid, ported from Arrow C++ RleBitPackedEncoder: values are
 *               buffered 8 at a time, a group whose 8 values repeat starts a repeated run
 *               (ULEB128(count << 1), value in ceil(width/8) bytes), anything else joins a literal run
 *               (one indicator byte ULEB128(groups << 1 | 1), at most 63 groups). Width = ceil(log2(max+1)), min 1.
 *  - `pqdelta`  Parquet DELTA_BINARY_PACKED per Encodings.md, with Arrow's INT64 defaults: 256 values per block,
 *               4 miniblocks of 64. Header and min deltas are ULEB128 / zigzag ULEB128.
 *  - `promts`   Prometheus XOR-chunk timestamps (tsdb/chunkenc/xor.go): 16-bit sample count, t0 as zigzag varint,
 *               first delta as uvarint, then delta-of-delta buckets '0', '10'+14, '110'+17, '1110'+20, '1111'+64 bits.
 *  - `gorilla`  Gorilla / Prometheus xorWrite for float64 values: '0' if equal; '10' + meaningful bits inside the
 *               previous window; '11' + 5-bit leading zeros (clamped to 31) + 6-bit length + meaningful bits.
 *  - `alp`      Parquet ALP layout (AlpEncoding.md): 7-byte page header, 4-byte offset per vector, per vector
 *               AlpInfo (exponent, factor, exception count), then FOR reference + bit width from the next steps,
 *               packed values, 16-bit exception positions and raw 64-bit exception values. The (e, f) pair is
 *               searched exhaustively per vector; decode is enc * 10^f * 10^-e and anything that does not
 *               round-trip is an exception.
 *  - `pfor`     Patched FOR from Zukowski et al. (ICDE 2006): b-bit codes (1..24), outliers stored raw, each
 *               exception's code slot holds the gap to the next exception (a linked list), so gaps longer than
 *               2^b force compulsory exceptions. The entry point is modelled as one first-exception offset per block.
 *  - `fsst`     FSST (VLDB 2020): up to 255 symbols of 1-8 bytes, 1-byte codes, code 255 escapes a literal byte,
 *               symbol table built bottom-up in 5 generations by gain = count x length.
 *
 * Lab choices (stated in the UI where a number depends on them):
 *  - `dict`, `rle`, `delta`, `dod`, `zigzag`, `for`, `bitpack` are generic steps in the BtrBlocks sense: the output
 *    of one feeds the next. Dictionary pages are PLAIN (4-byte length + bytes, or 8-byte values); run lengths are
 *    bit-packed at one width; FOR stores the block minimum; `bitpack` stores one 8-bit width per block and pads the
 *    block to a byte.
 *  - A stream no terminal step consumes is stored PLAIN: 64-bit integers, 32-bit dictionary codes, 32-bit length
 *    + bytes for strings.
 *  - FSST learns its table from the first 16 KB of the column and stores a 16-bit compressed length per string.
 *  - Bits are shown most significant bit first within each field. On disk Parquet packs values LSB-first into
 *    little-endian bytes; the bits are the same, only their order inside a byte differs.
 */
import { makeRng } from './Viz';

/* ------------------------------------------------------------------ columns */

export type ColKind = 'int' | 'float' | 'string';
export type Scalar = number | string;
export interface Column {
  kind: ColKind;
  values: Scalar[];
  label: string;
}

export type GenId = 'ids' | 'ints' | 'status' | 'ts' | 'sensor' | 'urls' | 'pasted';
export interface GenParams {
  n: number;
  outlierPct: number;
  sorted: boolean;
  jitterMs: number;
  decimals: number;
}

export const MAX_ROWS = 4096;
const STATUSES = ['delivered', 'shipped', 'pending', 'processing', 'cancelled', 'returned', 'refunded', 'on_hold', 'backordered', 'disputed', 'lost', 'partially_shipped'];
const STATUS_WEIGHTS = [30, 20, 12, 9, 7, 6, 5, 4, 3, 2, 1, 1];
const SECTIONS = ['products', 'search', 'cart', 'account', 'help'];
const WORDS = ['red', 'blue', 'wool', 'linen', 'jacket', 'shirt', 'boots', 'socks', 'lamp', 'desk', 'chair', 'mug', 'kettle', 'tent', 'rope', 'bike', 'helmet', 'watch', 'phone', 'case'];
const SOURCES = ['newsletter', 'google', 'partner', 'direct'];

export function generateColumn(gen: GenId, p: GenParams): Column {
  const n = Math.max(8, Math.min(MAX_ROWS, Math.round(p.n)));
  const rng = makeRng(0x5eed + ['ids', 'ints', 'status', 'ts', 'sensor', 'urls', 'pasted'].indexOf(gen) * 7919);
  const out: Scalar[] = [];
  if (gen === 'ids') {
    let v = 1_000_000;
    for (let i = 0; i < n; i++) {
      out.push(v);
      v += rng() < 0.12 ? 2 + Math.floor(rng() * 8) : 1;
    }
    return { kind: 'int', values: out, label: 'Sorted ids' };
  }
  if (gen === 'ints') {
    for (let i = 0; i < n; i++) {
      out.push(rng() * 100 < p.outlierPct ? 1000 + Math.floor(rng() * 2_000_000_000) : Math.floor(rng() * 1000));
    }
    return { kind: 'int', values: out, label: 'Random ints' };
  }
  if (gen === 'status') {
    const total = STATUS_WEIGHTS.reduce((a, b) => a + b, 0);
    for (let i = 0; i < n; i++) {
      let r = rng() * total;
      let k = 0;
      while (r >= STATUS_WEIGHTS[k]) r -= STATUS_WEIGHTS[k++];
      out.push(STATUSES[k]);
    }
    if (p.sorted) (out as string[]).sort();
    return { kind: 'string', values: out, label: 'Order status' };
  }
  if (gen === 'ts') {
    const t0 = 1_757_808_000_000;
    let slot = 0;
    for (let i = 0; i < n; i++) {
      if (i > 0 && rng() < 0.005) slot++; // a missed scrape
      const jitter = p.jitterMs > 0 ? Math.round((rng() * 2 - 1) * p.jitterMs) : 0;
      out.push(t0 + slot * 10_000 + jitter);
      slot++;
    }
    return { kind: 'int', values: out, label: 'Timestamps (ms)' };
  }
  if (gen === 'sensor') {
    let x = 21.5;
    const d = Math.max(0, Math.min(6, p.decimals));
    for (let i = 0; i < n; i++) {
      if (rng() >= 0.45) x = Number((x + (rng() * 2 - 1) * 5 * 10 ** -d).toFixed(d));
      out.push(Number(x.toFixed(d)));
    }
    return { kind: 'float', values: out, label: 'Sensor readings' };
  }
  // urls
  for (let i = 0; i < n; i++) {
    const sec = SECTIONS[Math.floor(rng() * SECTIONS.length)];
    const w1 = WORDS[Math.floor(rng() * WORDS.length)];
    const w2 = WORDS[Math.floor(rng() * WORDS.length)];
    const id = 10000 + Math.floor(rng() * 90000);
    const src = SOURCES[Math.floor(rng() * SOURCES.length)];
    out.push(`https://shop.example.com/${sec}/${w1}-${w2}-${id}?utm_source=${src}`);
  }
  return { kind: 'string', values: out, label: 'URLs' };
}

/** Pasted values: comma- or newline-separated. All integers → int, all numbers → float, else strings. */
export function parsePasted(text: string): Column | { error: string } {
  const parts = text
    .split(/[\n,]+/)
    .map((s) => s.trim())
    .filter((s) => s.length > 0)
    .slice(0, MAX_ROWS);
  if (parts.length < 2) return { error: 'Paste at least two values, separated by commas or new lines.' };
  if (parts.every((s) => /^-?\d{1,15}$/.test(s))) return { kind: 'int', values: parts.map(Number), label: 'Pasted ints' };
  if (parts.every((s) => /^-?(\d+\.?\d*|\.\d+)(e-?\d+)?$/i.test(s))) return { kind: 'float', values: parts.map(Number), label: 'Pasted floats' };
  return { kind: 'string', values: parts.map((s) => s.slice(0, 256)), label: 'Pasted strings' };
}

export function cmpScalar(a: Scalar, b: Scalar) {
  return a < b ? -1 : a > b ? 1 : 0;
}

/* ------------------------------------------------------------------ fields */

export type Role = 'payload' | 'header' | 'dict' | 'except' | 'control';
export const ROLES: Role[] = ['payload', 'header', 'dict', 'except', 'control'];
export interface Field {
  role: Role;
  w: number;
  v?: bigint;
  bytes?: Uint8Array;
  what: string;
  /** index of the stream item this field encodes (payload fields) */
  vi?: number;
  /** step that wrote it; -1 = PLAIN tail */
  si: number;
  /** block index, for block-structured layouts */
  bi?: number;
}

const enc = new TextEncoder();
export const utf8 = (s: string) => enc.encode(s);
const dv = new DataView(new ArrayBuffer(8));
export function f64bits(x: number): bigint {
  dv.setFloat64(0, x);
  return dv.getBigUint64(0);
}
function uint(w: number, v: number | bigint): bigint {
  if (w <= 0) return 0n;
  return BigInt.asUintN(w, typeof v === 'bigint' ? v : BigInt(Math.trunc(v)));
}
export function bitsNeeded(x: number): number {
  if (!(x > 0)) return 0;
  let b = 0;
  let v = x;
  while (v >= 1) {
    v = Math.floor(v / 2);
    b++;
  }
  return b;
}
export function uleb128(x: number): Uint8Array {
  const out: number[] = [];
  let v = Math.max(0, Math.floor(x));
  do {
    let byte = v % 128;
    v = Math.floor(v / 128);
    if (v > 0) byte |= 0x80;
    out.push(byte);
  } while (v > 0);
  return Uint8Array.from(out);
}
export const zigzag = (v: number) => (v >= 0 ? 2 * v : -2 * v - 1);
export const unzigzag = (u: number) => (u % 2 === 0 ? u / 2 : -(u + 1) / 2);

/** Bit i (0 = most significant) of a field. */
export function fieldBit(f: Field, i: number): number {
  if (f.bytes) return (f.bytes[i >> 3] >> (7 - (i & 7))) & 1;
  if (f.v === undefined) return 0;
  return Number((f.v >> BigInt(f.w - 1 - i)) & 1n);
}

const fmtVal = (v: Scalar) => (typeof v === 'string' ? `'${v.length > 28 ? v.slice(0, 27) + '…' : v}'` : String(v));

/* ------------------------------------------------------------------- steps */

export type StepId = 'dict' | 'dictSorted' | 'rle' | 'delta' | 'dod' | 'zigzag' | 'for' | 'bitpack' | 'pfor' | 'hybrid' | 'pqdelta' | 'promts' | 'gorilla' | 'alp' | 'fsst';

export const STEP_INFO: Record<StepId, { label: string; terminal: boolean; blockwise: boolean; title: string }> = {
  dict: { label: 'Dictionary (first-seen order)', terminal: false, blockwise: false, title: 'Replace each value by its index in a dictionary built in the order values first appear — what Parquet writers do.' },
  dictSorted: { label: 'Dictionary (sorted)', terminal: false, blockwise: false, title: 'Dictionary kept in sorted order, as ORC does for strings: codes compare like the values they stand for.' },
  rle: { label: 'RLE', terminal: false, blockwise: false, title: 'Collapse runs of equal values into (value, length). Later steps encode the run values; lengths are bit-packed.' },
  delta: { label: 'Delta', terminal: false, blockwise: false, title: 'Store the first value, then differences between neighbours.' },
  dod: { label: 'Delta-of-delta', terminal: false, blockwise: false, title: 'Differences of differences: zero for a perfectly regular sequence.' },
  zigzag: { label: 'Zigzag', terminal: false, blockwise: false, title: 'Map signed integers to unsigned: 0, −1, 1, −2 → 0, 1, 2, 3.' },
  for: { label: 'FOR', terminal: false, blockwise: true, title: 'Frame of reference: store each block’s minimum once and every value as an offset from it.' },
  bitpack: { label: 'Bit-pack', terminal: true, blockwise: true, title: 'Store each block’s values in just enough bits for its largest value.' },
  pfor: { label: 'PFOR', terminal: true, blockwise: true, title: 'Patched FOR: pick a small width b, store the values that do not fit as raw exceptions and patch them in after unpacking.' },
  hybrid: { label: 'Parquet RLE/bit-pack hybrid', terminal: true, blockwise: false, title: 'Parquet’s RLE encoding: repeated runs of 8+ values become (count, value); everything else is bit-packed in groups of 8.' },
  pqdelta: { label: 'Parquet DELTA_BINARY_PACKED', terminal: true, blockwise: false, title: 'Deltas, minus each block’s minimum delta, bit-packed per 64-value miniblock.' },
  promts: { label: 'Prometheus delta-of-delta', terminal: true, blockwise: false, title: 'Gorilla-style variable-length timestamps with Prometheus’s millisecond buckets.' },
  gorilla: { label: 'Gorilla XOR', terminal: true, blockwise: false, title: 'XOR each double with the previous one and store only the meaningful bits.' },
  alp: { label: 'ALP', terminal: false, blockwise: true, title: 'Adaptive lossless floating point: turn decimals into integers with a power of ten, exceptions for the rest.' },
  fsst: { label: 'FSST', terminal: true, blockwise: false, title: 'Replace frequent substrings (1–8 bytes) with 1-byte codes from a static symbol table.' },
};

interface SState {
  t: 'str' | 'float' | 'int';
  vals: Scalar[];
  width: 32 | 64;
  fromColumn: boolean;
  after: StepId | null;
  hadRle: boolean;
  /** a dictionary already ran: codes of codes would add nothing */
  hadDict: boolean;
  hadBlock: boolean;
  done: boolean;
}

const minMax = (a: number[]) => {
  let lo = Infinity;
  let hi = -Infinity;
  for (const v of a) {
    if (v < lo) lo = v;
    if (v > hi) hi = v;
  }
  return [lo, hi] as const;
};

/** Steps that may follow, given the stream the chain has produced so far. */
function allowedFor(s: SState): StepId[] {
  if (s.done) return [];
  if (s.after === 'for') return ['bitpack'];
  if (s.after === 'alp') return ['for', 'pfor'];
  if (s.t === 'str') return [...(['dict', 'dictSorted'] as StepId[]), ...(s.hadRle ? [] : (['rle'] as StepId[])), ...(s.fromColumn ? (['fsst'] as StepId[]) : [])];
  if (s.t === 'float') return [...(['dict', 'dictSorted'] as StepId[]), ...(s.hadRle ? [] : (['rle'] as StepId[])), ...(s.fromColumn ? (['gorilla', 'alp'] as StepId[]) : [])];
  const nums = s.vals as number[];
  const [lo, hi] = nums.length ? minMax(nums) : [0, 0];
  const out: StepId[] = [];
  if ((s.fromColumn || s.after === 'rle') && !s.hadDict) out.push('dict', 'dictSorted');
  if (!s.hadRle) out.push('rle');
  if (s.fromColumn || s.after === 'rle') out.push('delta', 'dod');
  if (lo < 0) out.push('zigzag');
  out.push('for', 'pfor');
  if (lo >= 0) out.push('bitpack');
  if (lo >= 0 && hi < 2 ** 32) out.push('hybrid');
  if (s.fromColumn) out.push('pqdelta');
  if (s.fromColumn && nums.every((v, i) => i === 0 || v >= nums[i - 1])) out.push('promts');
  return out;
}

export interface StepTrace {
  id: StepId;
  desc: string;
  data: Record<string, unknown>;
  out: Scalar[];
}
export interface Encoded {
  ok: boolean;
  error?: string;
  fields: Field[];
  offsets: number[];
  bits: Record<Role, number>;
  total: number;
  plain: number;
  steps: StepTrace[];
  plainTail: boolean;
  tailDesc: string;
  blockSize: number;
  blockCount: number;
  next: StepId[];
}

const zeroRoles = (): Record<Role, number> => ({ payload: 0, header: 0, dict: 0, except: 0, control: 0 });

export function plainBits(col: Column) {
  if (col.kind === 'string') return (col.values as string[]).reduce((a, s) => a + 32 + 8 * utf8(s).length, 0);
  return col.values.length * 64;
}

/* ------------------------------------------------------------------ hybrid (Arrow port) */

interface HybridRun {
  kind: 'rle' | 'bp';
  start: number;
  count: number;
}
function hybridEncode(vals: number[], si: number) {
  const bw = Math.max(1, bitsNeeded(vals.length ? Math.max(...vals) : 0));
  const fields: Field[] = [{ role: 'header', w: 8, v: uint(8, bw), what: `bit width = ${bw}`, si }];
  const runs: HybridRun[] = [];
  // Arrow RleBitPackedEncoder state
  let buffered: { v: number; i: number }[] = [];
  let repeat = 0;
  let current = 0;
  let literalCount = 0;
  let indicator: Field | null = null;
  let literalStart = 0;
  const reserveIndicator = () => {
    if (!indicator) {
      indicator = { role: 'control', w: 8, v: 0n, what: 'bit-packed run header', si };
      fields.push(indicator);
      literalStart = buffered.length ? buffered[0].i : literalStart;
    }
    return indicator;
  };
  const writeBuffered = () => {
    for (const b of buffered) fields.push({ role: 'payload', w: bw, v: uint(bw, b.v), what: `value ${b.v} in ${bw} bits`, vi: b.i, si });
    buffered = [];
  };
  const closeIndicator = (end: number) => {
    const ind = reserveIndicator();
    const groups = literalCount / 8;
    ind.v = uint(8, (groups << 1) | 1);
    ind.what = `bit-packed run header: ULEB128(${groups} group${groups === 1 ? '' : 's'} << 1 | 1) = ${(groups << 1) | 1}`;
    runs.push({ kind: 'bp', start: literalStart, count: end - literalStart });
    indicator = null;
    literalCount = 0;
  };
  const flushLiteral = (update: boolean, end: number) => {
    reserveIndicator();
    writeBuffered();
    if (update) closeIndicator(end);
  };
  const flushRepeated = (end: number) => {
    const hdr = uleb128(repeat * 2);
    fields.push({ role: 'control', w: hdr.length * 8, bytes: hdr, what: `RLE run header: ULEB128(${repeat} << 1) = ${repeat * 2}`, si });
    const vb = Math.ceil(bw / 8) * 8;
    fields.push({ role: 'payload', w: vb, v: uint(vb, current), what: `repeated value ${current} (${vb / 8} byte${vb > 8 ? 's' : ''})`, vi: end - repeat, si });
    runs.push({ kind: 'rle', start: end - repeat, count: repeat });
    buffered = [];
    repeat = 0;
  };
  const flushBuffered = (end: number) => {
    if (repeat >= 8) {
      buffered = [];
      if (literalCount !== 0) flushLiteral(true, end - 8);
      return;
    }
    literalCount += buffered.length;
    flushLiteral(literalCount / 8 + 1 >= 64, end);
    repeat = 0;
  };
  for (let i = 0; i < vals.length; i++) {
    const v = vals[i];
    if (v === current) {
      repeat++;
      if (repeat > 8) continue;
    } else {
      if (repeat >= 8) flushRepeated(i);
      repeat = 1;
      current = v;
    }
    buffered.push({ v, i });
    if (buffered.length === 8) flushBuffered(i + 1);
  }
  const n = vals.length;
  if (literalCount > 0 || repeat > 0 || buffered.length > 0) {
    const allRepeat = literalCount === 0 && (repeat === buffered.length || buffered.length === 0);
    if (repeat > 0 && allRepeat) {
      flushRepeated(n);
    } else {
      const real = buffered.length;
      reserveIndicator();
      writeBuffered();
      if (real > 0 && real < 8) fields.push({ role: 'control', w: bw * (8 - real), v: 0n, what: `${8 - real} zero value${8 - real > 1 ? 's' : ''} padding the last group of 8`, si });
      literalCount += real > 0 ? 8 : 0;
      closeIndicator(n);
      repeat = 0;
    }
  }
  runs.sort((a, b) => a.start - b.start);
  return { fields, runs, bw };
}

/* ------------------------------------------------------------------ FSST */

const latin1 = (b: Uint8Array) => {
  let s = '';
  for (let i = 0; i < b.length; i++) s += String.fromCharCode(b[i]);
  return s;
};
export function fsstBuild(sample: string[]): string[] {
  let symbols: string[] = [];
  for (let gen = 0; gen < 5; gen++) {
    const set = new Set(symbols);
    const count1 = new Map<string, number>();
    const count2 = new Map<string, number>();
    for (const s of sample) {
      let pos = 0;
      let prev: string | null = null;
      while (pos < s.length) {
        let sym = s[pos];
        for (let L = Math.min(8, s.length - pos); L >= 1; L--) {
          const cand = s.substr(pos, L);
          if (set.has(cand)) {
            sym = cand;
            break;
          }
        }
        count1.set(sym, (count1.get(sym) ?? 0) + 1);
        if (sym.length > 1) count1.set(s[pos], (count1.get(s[pos]) ?? 0) + 1);
        if (prev !== null) {
          const pair = (prev + sym).slice(0, 8);
          count2.set(pair, (count2.get(pair) ?? 0) + 1);
          if (sym.length > 1) {
            const ext = (prev + s[pos]).slice(0, 8);
            count2.set(ext, (count2.get(ext) ?? 0) + 1);
          }
        }
        prev = sym;
        pos += sym.length;
      }
    }
    const gain = new Map<string, number>();
    for (const [k, c] of count1) gain.set(k, Math.max(gain.get(k) ?? 0, c * k.length));
    for (const [k, c] of count2) gain.set(k, Math.max(gain.get(k) ?? 0, c * k.length));
    symbols = [...gain.entries()]
      .sort((a, b) => b[1] - a[1] || b[0].length - a[0].length || (a[0] < b[0] ? -1 : 1))
      .slice(0, 255)
      .map(([k]) => k);
  }
  return symbols;
}
export function fsstEncodeOne(s: string, index: Map<string, number>): number[] {
  const out: number[] = [];
  let pos = 0;
  while (pos < s.length) {
    let hit = -1;
    let len = 1;
    for (let L = Math.min(8, s.length - pos); L >= 1; L--) {
      const c = index.get(s.substr(pos, L));
      if (c !== undefined) {
        hit = c;
        len = L;
        break;
      }
    }
    if (hit < 0) {
      out.push(255, s.charCodeAt(pos));
      pos += 1;
    } else {
      out.push(hit);
      pos += len;
    }
  }
  return out;
}

/* ------------------------------------------------------------------ ALP */

const P10 = [1e0, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8, 1e9, 1e10, 1e11, 1e12, 1e13, 1e14, 1e15, 1e16, 1e17, 1e18];
const N10 = [1e0, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10, 1e-11, 1e-12, 1e-13, 1e-14, 1e-15, 1e-16, 1e-17, 1e-18];
export function alpTry(v: number, e: number, f: number): number | null {
  if (!Number.isFinite(v) || Object.is(v, -0)) return null;
  const scaled = v * P10[e] * N10[f];
  if (!Number.isFinite(scaled) || Math.abs(scaled) > 2 ** 52) return null;
  const r = Math.round(scaled);
  return r * P10[f] * N10[e] === v ? r : null;
}
function alpVector(vals: number[]) {
  let best = { e: 0, f: 0, bits: Infinity, exc: [] as number[], enc: [] as number[] };
  for (let e = 0; e <= 18; e++) {
    for (let f = 0; f <= e; f++) {
      const encd: number[] = new Array(vals.length);
      const exc: number[] = [];
      let lo = Infinity;
      let hi = -Infinity;
      for (let i = 0; i < vals.length; i++) {
        const r = alpTry(vals[i], e, f);
        if (r === null) {
          exc.push(i);
          encd[i] = NaN;
        } else {
          encd[i] = r;
          if (r < lo) lo = r;
          if (r > hi) hi = r;
        }
      }
      const width = exc.length === vals.length ? 0 : bitsNeeded(hi - lo);
      const bits = width * vals.length + exc.length * 80;
      if (bits < best.bits) best = { e, f, bits, exc, enc: encd };
    }
  }
  const placeholder = best.enc.find((x) => !Number.isNaN(x)) ?? 0;
  best.enc = best.enc.map((x) => (Number.isNaN(x) ? placeholder : x));
  return best;
}

/* ------------------------------------------------------------------ PFOR */

function pforBlock(vals: number[], valueWidth: number) {
  const n = vals.length;
  const sorted = [...vals].sort((a, b) => a - b);
  let best = { b: 1, base: sorted[0] ?? 0, exc: [] as number[], compulsory: 0, bits: Infinity };
  for (let b = 1; b <= 24; b++) {
    const span = 2 ** b - 1;
    let bestCount = -1;
    let base = sorted[0] ?? 0;
    let j = 0;
    for (let i = 0; i < n; i++) {
      while (sorted[i] - sorted[j] > span) j++;
      // window [sorted[j], sorted[i]] fits; count i - j + 1
      if (i - j + 1 > bestCount) {
        bestCount = i - j + 1;
        base = sorted[j];
      }
    }
    const natural: number[] = [];
    for (let i = 0; i < n; i++) if (vals[i] < base || vals[i] - base > span) natural.push(i);
    // compulsory exceptions keep the linked list's gaps within 2^b
    const exc: number[] = [];
    let compulsory = 0;
    for (const p of natural) {
      if (exc.length) {
        let last = exc[exc.length - 1];
        while (p - last > 2 ** b) {
          last += 2 ** b;
          exc.push(last);
          compulsory++;
        }
      }
      exc.push(p);
    }
    const bits = 64 + 8 + 16 + 16 + n * b + exc.length * valueWidth;
    if (bits < best.bits) best = { b, base, exc, compulsory, bits };
    if (natural.length === 0) break;
  }
  return best;
}

/* ------------------------------------------------------------------ encode */

export function encodeChain(col: Column, chain: StepId[], blockSize: number): Encoded {
  const plain = plainBits(col);
  let s: SState = { t: col.kind === 'string' ? 'str' : col.kind, vals: col.values, width: 64, fromColumn: true, after: null, hadRle: false, hadDict: false, hadBlock: false, done: false };
  const pre: Field[] = [];
  const post: Field[] = [];
  const heads: ((b: number, lo: number, hi: number) => Field[])[] = [];
  const tails: ((b: number, lo: number, hi: number) => Field[])[] = [];
  let payload: ((b: number, lo: number, hi: number) => Field[]) | null = null;
  let flat: Field[] | null = null;
  const steps: StepTrace[] = [];
  let blockLen = 0;
  let alpStep = -1;
  const fail = (error: string): Encoded => ({ ok: false, error, fields: [], offsets: [], bits: zeroRoles(), total: 0, plain, steps, plainTail: false, tailDesc: '', blockSize, blockCount: 0, next: [] });

  for (let si = 0; si < chain.length; si++) {
    const id = chain[si];
    if (!allowedFor(s).includes(id)) return fail(`${STEP_INFO[id].label} cannot follow ${si === 0 ? 'the raw column' : STEP_INFO[chain[si - 1]].label} here.`);
    const info = STEP_INFO[id];
    if (info.blockwise && !s.hadBlock) blockLen = s.vals.length;

    if (id === 'dict' || id === 'dictSorted') {
      // doubles are keyed by their bits, so -0.0 and NaN payloads stay distinct entries
      const key = (v: Scalar): Scalar => (s.t === 'float' ? f64bits(v as number).toString() : v);
      const pos = new Map<Scalar, number>();
      const order: Scalar[] = [];
      for (const v of s.vals) {
        if (!pos.has(key(v))) {
          pos.set(key(v), order.length);
          order.push(v);
        }
      }
      const dictVals = id === 'dictSorted' ? [...order].sort(cmpScalar) : order;
      const idx = new Map<Scalar, number>(dictVals.map((v, i) => [key(v), i]));
      const codes = s.vals.map((v) => idx.get(key(v)) as number);
      dictVals.forEach((v, i) => {
        if (typeof v === 'string') {
          const b = utf8(v);
          pre.push({ role: 'dict', w: 32, v: uint(32, b.length), what: `dictionary[${i}] length = ${b.length}`, si });
          if (b.length) pre.push({ role: 'dict', w: b.length * 8, bytes: b, what: `dictionary[${i}] = ${fmtVal(v)}`, si });
        } else if (s.t === 'float') {
          pre.push({ role: 'dict', w: 64, v: f64bits(v), what: `dictionary[${i}] = ${v}`, si });
        } else {
          pre.push({ role: 'dict', w: s.width, v: uint(s.width, v), what: `dictionary[${i}] = ${v}`, si });
        }
      });
      steps.push({ id, desc: `${dictVals.length} distinct → codes 0..${dictVals.length - 1}`, data: { dictVals, codes, sorted: id === 'dictSorted' }, out: codes });
      s = { ...s, t: 'int', vals: codes, width: 32, fromColumn: false, after: id, hadDict: true };
      continue;
    }
    if (id === 'rle') {
      const runVals: Scalar[] = [];
      const runLens: number[] = [];
      const runStarts: number[] = [];
      s.vals.forEach((v, i) => {
        if (runVals.length && Object.is(runVals[runVals.length - 1], v)) runLens[runLens.length - 1]++;
        else {
          runVals.push(v);
          runLens.push(1);
          runStarts.push(i);
        }
      });
      const w = bitsNeeded(Math.max(...runLens));
      post.push({ role: 'control', w: 8, v: uint(8, w), what: `run-length bit width = ${w}`, si });
      runLens.forEach((L, j) => post.push({ role: 'control', w, v: uint(w, L), what: `run ${j} length = ${L}`, vi: j, si }));
      steps.push({ id, desc: `${s.vals.length} → ${runVals.length} runs (avg ${(s.vals.length / runVals.length).toFixed(1)})`, data: { runStarts, runLens }, out: runVals });
      s = { ...s, vals: runVals, fromColumn: false, after: 'rle', hadRle: true };
      continue;
    }
    if (id === 'delta' || id === 'dod') {
      const a = s.vals as number[];
      const out = a.map((v, i) => (i === 0 ? 0 : v - a[i - 1]));
      pre.push({ role: 'header', w: 64, v: uint(64, a[0] ?? 0), what: `first value = ${a[0]}`, si });
      let res = out;
      if (id === 'dod') {
        res = out.map((d, i) => (i < 2 ? 0 : d - out[i - 1]));
        pre.push({ role: 'header', w: 64, v: uint(64, out[1] ?? 0), what: `first delta = ${out[1] ?? 0}`, si });
      }
      const [lo, hi] = minMax(res.slice(id === 'dod' ? 2 : 1));
      steps.push({ id, desc: res.length > 2 ? `values in [${lo}, ${hi}]` : 'too short', data: {}, out: res });
      s = { ...s, vals: res, width: 64, fromColumn: false, after: id };
      continue;
    }
    if (id === 'zigzag') {
      const out = (s.vals as number[]).map(zigzag);
      steps.push({ id, desc: `max ${Math.max(...out)}`, data: {}, out });
      s = { ...s, vals: out, fromColumn: false, after: 'zigzag' };
      continue;
    }
    if (id === 'for') {
      const a = s.vals as number[];
      const refs: number[] = [];
      const out = a.slice();
      for (let b = 0; b * blockSize < a.length; b++) {
        const lo = b * blockSize;
        const hi = Math.min(a.length, lo + blockSize);
        const [mn] = minMax(a.slice(lo, hi));
        refs.push(mn);
        for (let i = lo; i < hi; i++) out[i] = a[i] - mn;
      }
      const width = s.width;
      heads.push((b) => [{ role: 'header', w: width, v: uint(width, refs[b]), what: `block ${b} reference (minimum) = ${refs[b]}`, si, bi: b }]);
      steps.push({ id, desc: `${refs.length} block${refs.length > 1 ? 's' : ''}, offsets up to ${Math.max(...out)}`, data: { refs }, out });
      s = { ...s, vals: out, fromColumn: false, after: 'for', hadBlock: true };
      continue;
    }
    if (id === 'alp') {
      const a = s.vals as number[];
      const blocks: { e: number; f: number; exc: number[]; vals: number[] }[] = [];
      const out: number[] = [];
      for (let b = 0; b * blockSize < a.length; b++) {
        const slice = a.slice(b * blockSize, Math.min(a.length, (b + 1) * blockSize));
        const r = alpVector(slice);
        blocks.push({ e: r.e, f: r.f, exc: r.exc, vals: slice });
        out.push(...r.enc);
      }
      alpStep = si;
      pre.push({ role: 'header', w: 8, v: 0n, what: 'compression_mode = 0 (ALP)', si });
      pre.push({ role: 'header', w: 8, v: 0n, what: 'integer_encoding = 0 (FOR + bit-packing)', si });
      pre.push({ role: 'header', w: 8, v: uint(8, Math.log2(blockSize)), what: `log_vector_size = ${Math.log2(blockSize)}`, si });
      pre.push({ role: 'header', w: 32, v: uint(32, a.length), what: `num_elements = ${a.length}`, si });
      blocks.forEach((_, b) => pre.push({ role: 'header', w: 32, v: 0n, what: `vector ${b} offset`, si }));
      heads.push((b) => [
        { role: 'header', w: 8, v: uint(8, blocks[b].e), what: `vector ${b} exponent e = ${blocks[b].e}`, si, bi: b },
        { role: 'header', w: 8, v: uint(8, blocks[b].f), what: `vector ${b} factor f = ${blocks[b].f}`, si, bi: b },
        { role: 'header', w: 16, v: uint(16, blocks[b].exc.length), what: `vector ${b} exceptions = ${blocks[b].exc.length}`, si, bi: b },
      ]);
      tails.push((b) => [
        ...blocks[b].exc.map((p) => ({ role: 'except' as Role, w: 16, v: uint(16, p), what: `exception position ${p}`, si, bi: b })),
        ...blocks[b].exc.map((p) => ({ role: 'except' as Role, w: 64, v: f64bits(blocks[b].vals[p]), what: `exception value ${blocks[b].vals[p]}`, vi: b * blockSize + p, si, bi: b })),
      ]);
      const ef = new Set(blocks.map((x) => `e=${x.e} f=${x.f}`));
      const excTotal = blocks.reduce((t, x) => t + x.exc.length, 0);
      steps.push({ id, desc: `${[...ef].slice(0, 2).join(', ')}${ef.size > 2 ? '…' : ''}; ${excTotal} exception${excTotal === 1 ? '' : 's'}`, data: { blocks }, out });
      s = { ...s, t: 'int', vals: out, width: 64, fromColumn: false, after: 'alp', hadBlock: true };
      continue;
    }
    if (id === 'bitpack') {
      const a = s.vals as number[];
      const widths: number[] = [];
      for (let b = 0; b * blockSize < a.length; b++) widths.push(bitsNeeded(minMax(a.slice(b * blockSize, Math.min(a.length, (b + 1) * blockSize)))[1]));
      heads.push((b) => [{ role: 'header', w: 8, v: uint(8, widths[b]), what: `block ${b} bit width = ${widths[b]}`, si, bi: b }]);
      payload = (b, lo, hi) => {
        const w = widths[b];
        const f: Field[] = [];
        if (w > 0) for (let i = lo; i < hi; i++) f.push({ role: 'payload', w, v: uint(w, a[i]), what: `item ${i}: ${a[i]} in ${w} bits`, vi: i, si, bi: b });
        const pad = (8 - (((hi - lo) * w) % 8)) % 8;
        if (pad) f.push({ role: 'control', w: pad, v: 0n, what: `${pad} padding bit${pad > 1 ? 's' : ''} to a byte`, si, bi: b });
        return f;
      };
      const [wl, wh] = minMax(widths);
      steps.push({ id, desc: wl === wh ? `${wl} bits per value` : `${wl}–${wh} bits per value`, data: { widths }, out: a });
      s = { ...s, done: true, after: 'bitpack', hadBlock: true };
      continue;
    }
    if (id === 'pfor') {
      const a = s.vals as number[];
      const blocks: ReturnType<typeof pforBlock>[] = [];
      const width = s.width;
      for (let b = 0; b * blockSize < a.length; b++) blocks.push(pforBlock(a.slice(b * blockSize, Math.min(a.length, (b + 1) * blockSize)), width));
      heads.push((b) => {
        const k = blocks[b];
        return [
          { role: 'header', w: 64, v: uint(64, k.base), what: `block ${b} base = ${k.base}`, si, bi: b },
          { role: 'header', w: 8, v: uint(8, k.b), what: `block ${b} code width b = ${k.b}`, si, bi: b },
          { role: 'header', w: 16, v: uint(16, k.exc.length), what: `block ${b} exceptions = ${k.exc.length}`, si, bi: b },
          { role: 'header', w: 16, v: uint(16, k.exc.length ? k.exc[0] : 0xffff), what: k.exc.length ? `entry point: first exception at ${k.exc[0]}` : 'entry point: no exceptions', si, bi: b },
        ];
      });
      payload = (b, lo) => {
        const k = blocks[b];
        const excSet = new Map<number, number>(k.exc.map((p, j) => [p, j]));
        const f: Field[] = [];
        for (let i = 0; i < a.length - lo && i < blockSize; i++) {
          const j = excSet.get(i);
          if (j === undefined) f.push({ role: 'payload', w: k.b, v: uint(k.b, a[lo + i] - k.base), what: `item ${lo + i}: ${a[lo + i]} − base = ${a[lo + i] - k.base}`, vi: lo + i, si, bi: b });
          else {
            const gap = j + 1 < k.exc.length ? k.exc[j + 1] - i - 1 : 0;
            f.push({ role: 'except', w: k.b, v: uint(k.b, gap), what: `exception slot: next exception ${j + 1 < k.exc.length ? `${gap + 1} later` : 'none'}`, vi: lo + i, si, bi: b });
          }
        }
        const pad = (8 - ((f.length * k.b) % 8)) % 8;
        if (pad) f.push({ role: 'control', w: pad, v: 0n, what: `${pad} padding bit${pad > 1 ? 's' : ''} to a byte`, si, bi: b });
        return f;
      };
      tails.push((b, lo) => blocks[b].exc.map((p) => ({ role: 'except' as Role, w: width, v: uint(width, a[lo + p]), what: `exception value ${a[lo + p]}`, vi: lo + p, si, bi: b })));
      const bs = blocks.map((k) => k.b);
      const exc = blocks.reduce((t, k) => t + k.exc.length, 0);
      const comp = blocks.reduce((t, k) => t + k.compulsory, 0);
      steps.push({ id, desc: `b = ${Math.min(...bs)}${Math.max(...bs) !== Math.min(...bs) ? `–${Math.max(...bs)}` : ''}, ${exc} exception${exc === 1 ? '' : 's'}${comp ? ` (${comp} compulsory)` : ''}`, data: { blocks }, out: a });
      s = { ...s, done: true, after: 'pfor', hadBlock: true };
      continue;
    }
    if (id === 'hybrid') {
      const r = hybridEncode(s.vals as number[], si);
      flat = r.fields;
      const nr = r.runs.filter((x) => x.kind === 'rle').length;
      steps.push({ id, desc: `width ${r.bw}: ${nr} RLE run${nr === 1 ? '' : 's'}, ${r.runs.length - nr} bit-packed`, data: { runs: r.runs, bw: r.bw }, out: s.vals });
      s = { ...s, done: true, after: 'hybrid' };
      continue;
    }
    if (id === 'pqdelta') {
      const a = s.vals as number[];
      const f: Field[] = [];
      const hb = (x: Uint8Array, what: string, role: Role = 'header') => f.push({ role, w: x.length * 8, bytes: x, what, si });
      const BLOCK = 256;
      const MINIS = 4;
      const PER = BLOCK / MINIS;
      hb(uleb128(BLOCK), `block size = ${BLOCK} (ULEB128)`);
      hb(uleb128(MINIS), `miniblocks per block = ${MINIS}`);
      hb(uleb128(a.length), `value count = ${a.length}`);
      hb(uleb128(zigzag(a[0] ?? 0)), `first value = ${a[0]} (zigzag ULEB128)`);
      const deltas = a.slice(1).map((v, i) => v - a[i]);
      let widthsAll: number[] = [];
      for (let bstart = 0; bstart < deltas.length; bstart += BLOCK) {
        const blk = deltas.slice(bstart, bstart + BLOCK);
        const [mn] = minMax(blk);
        hb(uleb128(zigzag(mn)), `block min delta = ${mn} (zigzag ULEB128)`);
        const widths: number[] = [];
        for (let m = 0; m < MINIS; m++) {
          const mb = blk.slice(m * PER, (m + 1) * PER);
          widths.push(mb.length ? bitsNeeded(Math.max(...mb.map((d) => d - mn))) : 0);
        }
        widths.forEach((w, m) => f.push({ role: 'header', w: 8, v: uint(8, w), what: `miniblock ${m} width = ${w}`, si }));
        widthsAll = widthsAll.concat(widths.filter((_, m) => m * PER < blk.length));
        for (let m = 0; m < MINIS; m++) {
          const mb = blk.slice(m * PER, (m + 1) * PER);
          if (!mb.length) break;
          const w = widths[m];
          if (w === 0) continue;
          mb.forEach((d, k) => f.push({ role: 'payload', w, v: uint(w, d - mn), what: `delta ${d} − min ${mn} = ${d - mn}`, vi: bstart + m * PER + k + 1, si }));
          if (mb.length < PER) f.push({ role: 'control', w: w * (PER - mb.length), v: 0n, what: `padding the last miniblock to ${PER} values`, si });
        }
      }
      flat = f;
      const [wl, wh] = widthsAll.length ? minMax(widthsAll) : [0, 0];
      steps.push({ id, desc: `miniblock widths ${wl}${wh !== wl ? `–${wh}` : ''} bits`, data: {}, out: a });
      s = { ...s, done: true, after: 'pqdelta' };
      continue;
    }
    if (id === 'promts') {
      const a = s.vals as number[];
      const f: Field[] = [];
      f.push({ role: 'header', w: 16, v: uint(16, Math.min(a.length, 65535)), what: `sample count = ${a.length}`, si });
      const t0 = uleb128(zigzag(a[0] ?? 0));
      f.push({ role: 'header', w: t0.length * 8, bytes: t0, what: `t0 = ${a[0]} (varint)`, vi: 0, si });
      const buckets = [0, 0, 0, 0, 0];
      if (a.length > 1) {
        const d1 = uleb128(a[1] - a[0]);
        f.push({ role: 'header', w: d1.length * 8, bytes: d1, what: `first delta = ${a[1] - a[0]} ms (uvarint)`, vi: 1, si });
      }
      const inRange = (x: number, nb: number) => -(2 ** (nb - 1) - 1) <= x && x <= 2 ** (nb - 1);
      for (let i = 2; i < a.length; i++) {
        const dod = a[i] - a[i - 1] - (a[i - 1] - a[i - 2]);
        if (dod === 0) {
          f.push({ role: 'control', w: 1, v: 0n, what: `'0': delta-of-delta 0`, vi: i, si });
          buckets[0]++;
          continue;
        }
        const [pfx, pw, nb, k] = inRange(dod, 14) ? [0b10, 2, 14, 1] : inRange(dod, 17) ? [0b110, 3, 17, 2] : inRange(dod, 20) ? [0b1110, 4, 20, 3] : [0b1111, 4, 64, 4];
        buckets[k]++;
        f.push({ role: 'control', w: pw, v: uint(pw, pfx), what: `'${pfx.toString(2)}': ${nb}-bit bucket`, vi: i, si });
        f.push({ role: 'payload', w: nb, v: uint(nb, dod), what: `delta-of-delta ${dod} in ${nb} bits`, vi: i, si });
      }
      flat = f;
      const tot = Math.max(1, a.length - 2);
      steps.push({ id, desc: `${Math.round((buckets[0] / tot) * 100)}% one bit; bucket counts 14-bit ${buckets[1]}, 17-bit ${buckets[2]}, 20-bit ${buckets[3]}, 64-bit ${buckets[4]} (each plus a 2–4 bit prefix)`, data: { buckets }, out: a });
      s = { ...s, done: true, after: 'promts' };
      continue;
    }
    if (id === 'gorilla') {
      const a = s.vals as number[];
      const f: Field[] = [];
      const cls = { same: 0, reuse: 0, fresh: 0 };
      if (a.length) f.push({ role: 'header', w: 64, v: f64bits(a[0]), what: `first value ${a[0]}, raw IEEE 754`, vi: 0, si });
      let leading = 0xff;
      let trailing = 0;
      for (let i = 1; i < a.length; i++) {
        const x = f64bits(a[i]) ^ f64bits(a[i - 1]);
        if (x === 0n) {
          f.push({ role: 'control', w: 1, v: 0n, what: `'0': ${a[i]} same as previous`, vi: i, si });
          cls.same++;
          continue;
        }
        const len = x.toString(2).length;
        let nl = 64 - len;
        let nt = 0;
        while (((x >> BigInt(nt)) & 1n) === 0n) nt++;
        if (nl >= 32) nl = 31;
        if (leading !== 0xff && nl >= leading && nt >= trailing) {
          const m = 64 - leading - trailing;
          f.push({ role: 'control', w: 2, v: 0b10n, what: `'10': reuse window (${leading} leading, ${trailing} trailing zeros)`, vi: i, si });
          f.push({ role: 'payload', w: m, v: uint(m, x >> BigInt(trailing)), what: `${m} meaningful XOR bits of ${a[i]}`, vi: i, si });
          cls.reuse++;
          continue;
        }
        leading = nl;
        trailing = nt;
        const sig = 64 - nl - nt;
        f.push({ role: 'control', w: 2, v: 0b11n, what: `'11': new window`, vi: i, si });
        f.push({ role: 'control', w: 5, v: uint(5, nl), what: `leading zeros = ${nl}`, vi: i, si });
        f.push({ role: 'control', w: 6, v: uint(6, sig === 64 ? 0 : sig), what: `meaningful bits = ${sig}`, vi: i, si });
        f.push({ role: 'payload', w: sig, v: uint(sig, x >> BigInt(nt)), what: `${sig} meaningful XOR bits of ${a[i]}`, vi: i, si });
        cls.fresh++;
      }
      flat = f;
      const tot = Math.max(1, a.length - 1);
      steps.push({ id, desc: `${Math.round((cls.same / tot) * 100)}% '0', ${Math.round((cls.reuse / tot) * 100)}% '10', ${Math.round((cls.fresh / tot) * 100)}% '11'`, data: cls, out: a });
      s = { ...s, done: true, after: 'gorilla' };
      continue;
    }
    if (id === 'fsst') {
      const a = s.vals as string[];
      const raw = a.map((x) => latin1(utf8(x)));
      const sample: string[] = [];
      let bytes = 0;
      for (const x of raw) {
        if (bytes >= 16384) break;
        sample.push(x);
        bytes += x.length;
      }
      const symbols = fsstBuild(sample);
      const index = new Map(symbols.map((x, i) => [x, i]));
      const f: Field[] = [];
      f.push({ role: 'dict', w: 8, v: uint(8, symbols.length), what: `symbol count = ${symbols.length}`, si });
      symbols.forEach((sym, i) => {
        f.push({ role: 'dict', w: 8, v: uint(8, sym.length), what: `symbol ${i} length = ${sym.length}`, si });
        f.push({ role: 'dict', w: sym.length * 8, bytes: Uint8Array.from(sym, (c) => c.charCodeAt(0)), what: `symbol ${i} = "${sym}"`, si });
      });
      const compLens: number[] = [];
      const compressed: string[] = [];
      let escapes = 0;
      raw.forEach((x, i) => {
        const codes = fsstEncodeOne(x, index);
        compLens.push(codes.length);
        compressed.push(String.fromCharCode(...codes));
        f.push({ role: 'control', w: 16, v: uint(16, codes.length), what: `string ${i}: ${codes.length} compressed bytes (from ${x.length})`, vi: i, si });
        for (let k = 0; k < codes.length; k++) {
          if (codes[k] === 255) {
            escapes++;
            f.push({ role: 'except', w: 8, v: 255n, what: 'escape code 255', vi: i, si });
            f.push({ role: 'except', w: 8, v: uint(8, codes[k + 1]), what: `literal byte "${String.fromCharCode(codes[k + 1])}"`, vi: i, si });
            k++;
          } else f.push({ role: 'payload', w: 8, v: uint(8, codes[k]), what: `code ${codes[k]} = "${symbols[codes[k]]}"`, vi: i, si });
        }
      });
      flat = f;
      const inBytes = raw.reduce((t, x) => t + x.length, 0);
      const outBytes = compLens.reduce((t, x) => t + x, 0);
      steps.push({ id, desc: `${symbols.length} symbols; ${(inBytes / Math.max(1, outBytes)).toFixed(1)}× on the string bytes, ${escapes} escapes`, data: { symbols, compLens, compressed, index }, out: a });
      s = { ...s, done: true, after: 'fsst' };
      continue;
    }
  }

  // assemble fields in physical order
  const fields: Field[] = [...pre];
  let plainTail = false;
  let tailDesc = '';
  const plainItem = (i: number, bi?: number): Field[] => {
    const v = s.vals[i];
    if (s.t === 'str') {
      const b = utf8(v as string);
      const out: Field[] = [{ role: 'control', w: 32, v: uint(32, b.length), what: `item ${i} length = ${b.length} (PLAIN)`, vi: i, si: -1, bi }];
      if (b.length) out.push({ role: 'payload', w: b.length * 8, bytes: b, what: `item ${i} = ${fmtVal(v)}`, vi: i, si: -1, bi });
      return out;
    }
    if (s.t === 'float') return [{ role: 'payload', w: 64, v: f64bits(v as number), what: `item ${i} = ${v} (PLAIN double)`, vi: i, si: -1, bi }];
    return [{ role: 'payload', w: s.width, v: uint(s.width, v as number), what: `item ${i} = ${v} (PLAIN ${s.width}-bit)`, vi: i, si: -1, bi }];
  };
  if (!s.done && !flat) {
    plainTail = true;
    tailDesc = s.t === 'str' ? 'stored PLAIN: 4-byte length + bytes' : s.t === 'float' ? 'stored PLAIN: 8-byte doubles' : `stored PLAIN: ${s.width}-bit integers`;
  }
  let blockCount = 0;
  if (flat) {
    for (const f of flat) fields.push(f);
  } else if (s.hadBlock) {
    blockCount = Math.ceil(blockLen / blockSize);
    for (let b = 0; b < blockCount; b++) {
      const lo = b * blockSize;
      const hi = Math.min(blockLen, lo + blockSize);
      for (const h of heads) fields.push(...h(b, lo, hi));
      if (payload) fields.push(...payload(b, lo, hi));
      else for (let i = lo; i < hi; i++) fields.push(...plainItem(i, b));
      for (const t of tails) fields.push(...t(b, lo, hi));
    }
  } else {
    for (let i = 0; i < s.vals.length; i++) fields.push(...plainItem(i));
  }
  for (const f of post) fields.push(f);

  if (alpStep >= 0 && blockCount > 0) {
    const sizes = new Array(blockCount).fill(0);
    for (const f of fields) if (f.bi !== undefined) sizes[f.bi] += f.w;
    const offs = fields.filter((f) => f.si === alpStep && f.what.startsWith('vector ') && f.what.endsWith(' offset'));
    let off = blockCount * 4;
    offs.forEach((f, b) => {
      f.v = uint(32, off);
      f.what = `vector ${b} offset = ${off} bytes`;
      off += Math.ceil(sizes[b] / 8);
    });
  }

  const bits = zeroRoles();
  const offsets: number[] = new Array(fields.length);
  let total = 0;
  fields.forEach((f, i) => {
    offsets[i] = total;
    total += f.w;
    bits[f.role] += f.w;
  });
  return { ok: true, fields, offsets, bits, total, plain, steps, plainTail, tailDesc, blockSize, blockCount, next: allowedFor(s) };
}

export function nextSteps(col: Column, chain: StepId[], blockSize: number): StepId[] {
  const e = encodeChain(col, chain, blockSize);
  return e.ok ? e.next : [];
}

/* ------------------------------------------------------------------ predicates */

export type Op = 'eq' | 'lt';
export interface Pred {
  op: Op;
  v: Scalar;
}
export type Output = 'count' | 'values';
/** 0 = settled without touching the value, 1 = tested in its stored form, 2 = decoded before the test */
export type Work = 0 | 1 | 2;

export interface Evaluation {
  matches: number;
  truth: number;
  work: Uint8Array;
  match: Uint8Array;
  testedOn: string;
  dictTested: number;
  encodedTested: number;
  decoded: number;
  blocksSkipped: number;
  blocksAll: number;
  blocksScanned: number;
  exceptionsPatched: number;
  outputDecoded: number;
  story: string[];
}

export function evaluate(col: Column, e: Encoded, pred: Pred, output: Output): Evaluation {
  const n = col.values.length;
  const work = new Uint8Array(n);
  const match = new Uint8Array(n);
  const test = (x: Scalar) => (pred.op === 'eq' ? cmpScalar(x, pred.v) === 0 : cmpScalar(x, pred.v) < 0);
  let truth = 0;
  for (const v of col.values) if (test(v)) truth++;
  const ev: Evaluation = { matches: 0, truth, work, match, testedOn: '', dictTested: 0, encodedTested: 0, decoded: 0, blocksSkipped: 0, blocksAll: 0, blocksScanned: 0, exceptionsPatched: 0, outputDecoded: 0, story: [] };
  if (!e.ok) return ev;

  // current stream item j covers rows [start[j], start[j] + len[j])
  let start: number[] = Array.from({ length: n }, (_, i) => i);
  let len: number[] = new Array(n).fill(1);
  let vals: Scalar[] = col.values;
  let allowed: Uint8Array | null = null; // over dictionary codes
  let refs: number[] | null = null;
  const inDomain = (x: Scalar) => (allowed ? allowed[x as number] === 1 : test(x));
  const settle = (j: number, cls: Work, ok: boolean) => {
    for (let r = start[j]; r < start[j] + len[j]; r++) {
      work[r] = cls;
      match[r] = ok ? 1 : 0;
    }
  };
  const decodeAll = (what: string) => {
    for (let j = 0; j < vals.length; j++) settle(j, 2, inDomain(vals[j]));
    ev.decoded += vals.length;
    ev.testedOn = allowed ? 'decoded codes' : 'decoded values';
    ev.story.push(what);
  };
  let settled = false;

  for (const st of e.steps) {
    if (settled) break;
    const d = st.data as Record<string, unknown>;
    switch (st.id) {
      case 'dict':
      case 'dictSorted': {
        const dictVals = d.dictVals as Scalar[];
        allowed = new Uint8Array(dictVals.length);
        if (st.id === 'dictSorted') {
          const lower = (x: Scalar) => {
            let lo = 0;
            let hi = dictVals.length;
            while (lo < hi) {
              const mid = (lo + hi) >> 1;
              ev.dictTested++;
              if (cmpScalar(dictVals[mid], x) < 0) lo = mid + 1;
              else hi = mid;
            }
            return lo;
          };
          const k = lower(pred.v);
          if (pred.op === 'lt') {
            for (let c = 0; c < k; c++) allowed[c] = 1;
            ev.story.push(`Sorted dictionary: a binary search puts ${fmtVal(pred.v)} at code ${k}, so the predicate becomes code < ${k}.`);
          } else if (k < dictVals.length && cmpScalar(dictVals[k], pred.v) === 0) {
            allowed[k] = 1;
            ev.story.push(`Sorted dictionary: a binary search finds ${fmtVal(pred.v)} at code ${k}; the predicate becomes code = ${k}.`);
          } else ev.story.push(`Sorted dictionary: ${fmtVal(pred.v)} is not in the dictionary, so no row can match.`);
        } else {
          dictVals.forEach((v, c) => {
            ev.dictTested++;
            if (test(v)) allowed![c] = 1;
          });
          const k = allowed.reduce((t, x) => t + x, 0);
          ev.story.push(`First-seen dictionary: the predicate is tested once against each of the ${dictVals.length} entries, leaving a set of ${k} matching code${k === 1 ? '' : 's'}.`);
        }
        vals = d.codes as number[];
        break;
      }
      case 'rle': {
        const rs = d.runStarts as number[];
        const rl = d.runLens as number[];
        const ns: number[] = [];
        const nl: number[] = [];
        rs.forEach((sj, j) => {
          ns.push(start[sj]);
          let t = 0;
          for (let k = sj; k < sj + rl[j]; k++) t += len[k];
          nl.push(t);
        });
        start = ns;
        len = nl;
        vals = st.out;
        ev.story.push(`RLE: the test runs once per run — ${vals.length} runs stand for ${n} rows.`);
        break;
      }
      case 'for':
        refs = d.refs as number[];
        break;
      case 'bitpack': {
        const widths = d.widths as number[];
        const B = e.blockSize;
        let lastAllowedPrefix: Int32Array | null = null;
        if (allowed) {
          lastAllowedPrefix = new Int32Array(allowed.length + 1);
          for (let c = 0; c < allowed.length; c++) lastAllowedPrefix[c + 1] = lastAllowedPrefix[c] + allowed[c];
        }
        for (let b = 0; b * B < vals.length; b++) {
          const lo = refs ? refs[b] : 0;
          const hi = lo + 2 ** widths[b] - 1;
          let cls: 'none' | 'all' | 'scan';
          if (allowed && lastAllowedPrefix) {
            const top = Math.min(hi, allowed.length - 1);
            const cnt = lo > top ? 0 : lastAllowedPrefix[top + 1] - lastAllowedPrefix[lo];
            cls = cnt === 0 ? 'none' : cnt === top - lo + 1 ? 'all' : 'scan';
          } else if (pred.op === 'lt') {
            cls = lo >= (pred.v as number) ? 'none' : hi < (pred.v as number) ? 'all' : 'scan';
          } else {
            const v = pred.v as number;
            cls = v < lo || v > hi ? 'none' : lo === hi && lo === v ? 'all' : 'scan';
          }
          const end = Math.min(vals.length, (b + 1) * B);
          for (let j = b * B; j < end; j++) {
            if (cls === 'scan') settle(j, 1, inDomain(vals[j]));
            else settle(j, 0, cls === 'all');
          }
          if (cls === 'none') ev.blocksSkipped++;
          else if (cls === 'all') ev.blocksAll++;
          else {
            ev.blocksScanned++;
            ev.encodedTested += end - b * B;
          }
        }
        ev.testedOn = allowed ? 'packed dictionary codes' : refs ? 'packed FOR offsets' : 'packed values';
        const bound = `[${refs ? 'ref' : '0'}, ${refs ? 'ref + ' : ''}2^w − 1]`;
        ev.story.push(
          ev.blocksSkipped + ev.blocksAll === 0
            ? `Bit-pack: each block's ${refs ? 'reference and ' : ''}width bound its values to ${bound}, but no bound rules a block in or out, so ${ev.blocksScanned === 1 ? 'the only block is' : `all ${ev.blocksScanned} blocks are`} unpacked and compared.`
            : `Bit-pack: each block's ${refs ? 'reference and ' : ''}width bound its values to ${bound}. From headers alone, ${ev.blocksSkipped} block${ev.blocksSkipped === 1 ? '' : 's'} cannot match and ${ev.blocksAll} match whole; ${ev.blocksScanned} ${ev.blocksScanned === 1 ? 'is' : 'are'} unpacked and compared.`,
        );
        settled = true;
        break;
      }
      case 'pfor': {
        const blocks = d.blocks as { exc: number[] }[];
        for (let j = 0; j < vals.length; j++) settle(j, 1, inDomain(vals[j]));
        ev.encodedTested += vals.length;
        ev.exceptionsPatched = blocks.reduce((t, k) => t + k.exc.length, 0);
        ev.blocksScanned = blocks.length;
        ev.testedOn = allowed ? 'unpacked codes' : 'unpacked PFOR codes';
        ev.story.push(`PFOR: every block is unpacked in one tight loop, then ${ev.exceptionsPatched} exception${ev.exceptionsPatched === 1 ? ' is' : 's are'} patched in before comparing — exceptions make the width useless as a bound.`);
        settled = true;
        break;
      }
      case 'hybrid': {
        const runs = d.runs as HybridRun[];
        let rleRuns = 0;
        for (const r of runs) {
          if (r.kind === 'rle') {
            const ok = inDomain(vals[r.start]);
            for (let j = r.start; j < r.start + r.count; j++) settle(j, 1, ok);
            ev.encodedTested += 1;
            rleRuns++;
          } else {
            for (let j = r.start; j < r.start + r.count && j < vals.length; j++) settle(j, 1, inDomain(vals[j]));
            ev.encodedTested += Math.min(r.count, vals.length - r.start);
          }
        }
        ev.testedOn = allowed ? 'dictionary codes' : 'hybrid-encoded values';
        const bpRuns = runs.length - rleRuns;
        ev.story.push(
          rleRuns === 0
            ? 'Hybrid: every code sits in a bit-packed run, so codes are unpacked 8 at a time and compared as integers.'
            : bpRuns === 0
              ? `Hybrid: ${rleRuns} RLE run${rleRuns === 1 ? ' is' : 's are'} tested once each — no code is unpacked.`
              : `Hybrid: ${rleRuns} RLE run${rleRuns === 1 ? ' is' : 's are'} tested once each; codes in ${bpRuns} bit-packed run${bpRuns === 1 ? '' : 's'} are unpacked 8 at a time and compared.`,
        );
        settled = true;
        break;
      }
      case 'fsst': {
        if (pred.op === 'eq') {
          const index = d.index as Map<string, number>;
          const target = String.fromCharCode(...fsstEncodeOne(latin1(utf8(pred.v as string)), index));
          const compressed = d.compressed as string[];
          for (let j = 0; j < vals.length; j++) settle(j, 1, compressed[j] === target);
          ev.encodedTested += vals.length;
          ev.testedOn = 'FSST-compressed bytes';
          ev.story.push(`FSST equality: the constant is compressed with the same symbol table (${target.length} bytes) and compared byte for byte against each compressed string — nothing is decompressed.`);
        } else {
          decodeAll('FSST range: codes do not preserve order, so every string is decompressed before comparing.');
        }
        settled = true;
        break;
      }
      case 'delta':
        decodeAll('Delta: each value is the previous value plus a delta, so every value must be rebuilt by a running sum before it can be compared.');
        settled = true;
        break;
      case 'dod':
        decodeAll('Delta-of-delta: two running sums rebuild every value before any comparison.');
        settled = true;
        break;
      case 'pqdelta':
        decodeAll('DELTA_BINARY_PACKED: miniblocks unpack fast, but values only exist after the running sum — every value is decoded.');
        settled = true;
        break;
      case 'promts':
        decodeAll('Delta-of-delta bitstream: variable-length codes must be read in order, so every timestamp is decoded.');
        settled = true;
        break;
      case 'gorilla':
        decodeAll('Gorilla XOR: each value is the previous value XOR a variable-length code, so decoding is strictly sequential.');
        settled = true;
        break;
      case 'alp':
        decodeAll('ALP: each vector is unpacked, the reference added back and multiplied by 10^f · 10^−e, exceptions patched — then compared. (DuckDB’s ALP scan has no filter callback; it decodes vectors.)');
        settled = true;
        break;
      case 'zigzag':
        decodeAll('Zigzag: un-zigzagging is cheap but per value.');
        settled = true;
        break;
    }
  }
  if (!settled) {
    if (refs) {
      const B = e.blockSize;
      for (let j = 0; j < vals.length; j++) settle(j, 1, inDomain(vals[j]));
      ev.blocksScanned = Math.ceil(vals.length / B);
      ev.testedOn = 'FOR offsets stored plain';
    } else {
      for (let j = 0; j < vals.length; j++) settle(j, 1, inDomain(vals[j]));
      ev.testedOn = allowed ? 'dictionary codes' : vals === col.values ? 'plain values' : 'run values';
    }
    ev.encodedTested += vals.length;
    ev.story.push(`The remaining stream is stored plain: each of its ${vals.length} items is compared as stored.`);
  }
  let m = 0;
  for (let r = 0; r < n; r++) if (match[r]) m++;
  ev.matches = m;
  if (output === 'values' && e.steps.length > 0) {
    for (let r = 0; r < n; r++) if (match[r] && work[r] < 2) ev.outputDecoded++;
  }
  return ev;
}

/* ------------------------------------------------------------------ presets and auto-pick */

export interface Preset {
  value: string;
  label: string;
  chain: StepId[];
  kinds: ColKind[];
}
export const PRESETS: Preset[] = [
  { value: 'plain', label: 'PLAIN (no encoding)', chain: [], kinds: ['int', 'float', 'string'] },
  { value: 'pq-dict', label: 'Parquet RLE_DICTIONARY', chain: ['dict', 'hybrid'], kinds: ['int', 'float', 'string'] },
  { value: 'sorted-dict', label: 'Sorted dictionary → bit-pack', chain: ['dictSorted', 'bitpack'], kinds: ['int', 'float', 'string'] },
  { value: 'rle-dict', label: 'RLE → dictionary → bit-pack', chain: ['rle', 'dictSorted', 'bitpack'], kinds: ['int', 'float', 'string'] },
  { value: 'bitpack', label: 'Bit-pack', chain: ['bitpack'], kinds: ['int'] },
  { value: 'for', label: 'FOR → bit-pack', chain: ['for', 'bitpack'], kinds: ['int'] },
  { value: 'pfor', label: 'PFOR (patched FOR)', chain: ['pfor'], kinds: ['int'] },
  { value: 'delta-for', label: 'Delta → FOR → bit-pack', chain: ['delta', 'for', 'bitpack'], kinds: ['int'] },
  { value: 'pq-delta', label: 'Parquet DELTA_BINARY_PACKED', chain: ['pqdelta'], kinds: ['int'] },
  { value: 'dod-for', label: 'Delta-of-delta → FOR → bit-pack', chain: ['dod', 'for', 'bitpack'], kinds: ['int'] },
  { value: 'prom', label: 'Prometheus delta-of-delta', chain: ['promts'], kinds: ['int'] },
  { value: 'gorilla', label: 'Gorilla XOR', chain: ['gorilla'], kinds: ['float'] },
  { value: 'alp', label: 'ALP → FOR → bit-pack (Parquet ALP)', chain: ['alp', 'for', 'bitpack'], kinds: ['float'] },
  { value: 'fsst', label: 'FSST', chain: ['fsst'], kinds: ['string'] },
];

export function presetFor(chain: StepId[]): string {
  const p = PRESETS.find((x) => x.chain.length === chain.length && x.chain.every((s, i) => s === chain[i]));
  return p ? p.value : 'custom';
}

/** BtrBlocks-style sample: 10 runs of 64 values from 10 non-overlapping parts of the column. */
export function sampleColumn(col: Column): Column {
  const n = col.values.length;
  if (n < 1280) return col;
  const rng = makeRng(0xb7b);
  const part = Math.floor(n / 10);
  const vals: Scalar[] = [];
  for (let k = 0; k < 10; k++) {
    const s = k * part + Math.floor(rng() * (part - 64));
    vals.push(...col.values.slice(s, s + 64));
  }
  return { ...col, values: vals };
}

export function pickSmallest(col: Column, blockSize: number, maxDepth = 3) {
  const sample = sampleColumn(col);
  const avgRun = (a: Scalar[]) => {
    let r = 1;
    for (let i = 1; i < a.length; i++) if (a[i] !== a[i - 1]) r++;
    return a.length / r;
  };
  const candidates: { chain: StepId[]; ratio: number }[] = [];
  const walk = (chain: StepId[]) => {
    const e = encodeChain(sample, chain, blockSize);
    if (!e.ok) return;
    if (chain.length) candidates.push({ chain, ratio: e.plain / Math.max(1, e.total) });
    if (chain.length >= maxDepth) return;
    const stream = e.steps.length ? e.steps[e.steps.length - 1].out : sample.values;
    const shortRuns = avgRun(stream) < 2;
    for (const nx of e.next) {
      if (nx === 'rle' && shortRuns) continue; // BtrBlocks: RLE is not viable when the average run is < 2
      walk([...chain, nx]);
    }
  };
  walk([]);
  candidates.sort((a, b) => b.ratio - a.ratio || a.chain.length - b.chain.length);
  for (const c of candidates) {
    const full = encodeChain(col, c.chain, blockSize);
    if (full.ok) return { chain: c.chain, sampleRatio: c.ratio, tried: candidates.length, sampled: sample.values.length };
  }
  return { chain: [] as StepId[], sampleRatio: 1, tried: candidates.length, sampled: sample.values.length };
}
