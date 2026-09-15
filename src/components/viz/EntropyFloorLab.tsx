/**
 * EntropyFloorLab — measured bits per value of real encoders against entropy floors.
 *
 * The learner shapes a column (a Zipf distribution over K small-integer codes, or uniformly random
 * 64-bit ids) and its physical order (arrival order, sorted within 1,024- or 8,192-row blocks, fully
 * sorted). Every encoder below is actually run on the 65,536 generated values and its output measured:
 *
 *  - PLAIN: 32 bits (INT32 codes) or 64 bits (INT64 ids) per value.
 *  - Bit-packing at the column's maximum width, plus one width byte.
 *  - Parquet RLE / bit-packing hybrid, a port of Arrow C++'s RleBitPackedEncoder (buffers 8 values,
 *    repeated run once 8 equal values are seen, 1-byte literal indicator, at most 63 groups per literal run),
 *    plus the 1-byte bit width a dictionary-index data page starts with. Bytes are really emitted.
 *  - Parquet DELTA_BINARY_PACKED with 128-value blocks of 4 miniblocks, int64 wrap-around deltas.
 *  - Static Huffman over whole values (order 0).
 *  - Static rANS (byte-wise, 32-bit state, 16-bit frequency scale) over whole values (order 0).
 *    N = 65,536 = 2^16, so the histogram is the exact frequency table: no quantisation loss.
 *  - LZ4 and ZSTD: NOT run in the browser. Their sizes come from EntropyFloorCodecFixture.ts, produced by the
 *    real lz4 1.10.0 and zstd 1.5.6 CLIs over the exact bytes this model emits, hash-checked per column.
 *
 * Model assumptions (also stated on screen):
 *  - Huffman and rANS both pay the same model cost: every distinct value's count at 17 bits, plus its 64-bit
 *    value for the random-id column (the small codes are a dense domain 0..K-1 and need no value list).
 *  - Dictionary pages are not modelled: the codes column is already a dense 0..K-1 domain.
 *
 * Floors:
 *  - H0: entropy of this column's value histogram, sum of -p log2 p over the measured frequencies (64 bits for the
 *    random ids, whose histogram is all distinct values). It is the floor for any coder that encodes each value on
 *    its own from one fixed model, and ordering cannot move it. The source distribution's entropy is shown beside it.
 *  - Hk: empirical order-k conditional entropy of the column as stored (k = 1 or 2), clamped into
 *    [information floor, H0]. The plug-in estimate falls below the truth when contexts are sparse (every context
 *    unique gives 0), and the clamp replaces it by the known bound in that case.
 *  - Information floor: the entropy of the stored column. Values are drawn i.i.d., then each block is sorted.
 *    Sorting a block of W values destroys exactly log2(W! / prod n_v!) bits (the orderings of its multiset), so
 *    H(column) = N*H0 - sum over blocks of E[log2 multinomial], the expectation taken under the column's measured
 *    frequencies. It is an expected value, not a bound on this particular sample.
 */
import { useMemo, useState } from 'react';
import { VizPanel, Segmented, Slider, Legend, Stats, Note, makeRng, fmtNum, siteHref } from './Viz';
import { CODEC_FIXTURE } from './EntropyFloorCodecFixture';

/* ------------------------------------------------------------------ inputs */

export const N = 65536;
export const K_STEPS = [2, 4, 16, 64, 256, 1024, 4096] as const;
export const SKEW_STEPS = [0, 0.5, 1, 1.5, 2, 3] as const;
export type Kind = 'codes' | 'random64';
export type Order = 'random' | 'w1024' | 'w8192' | 'sorted';
export const ORDERS: readonly Order[] = ['random', 'w1024', 'w8192', 'sorted'];
export const LEVELS = [1, 3, 9, 19] as const;
export const windowOf = (o: Order) => (o === 'random' ? 1 : o === 'w1024' ? 1024 : o === 'w8192' ? 8192 : N);

export type Spec = { kind: Kind; k: number; skew: number; order: Order };
export const specKey = (s: Spec) => (s.kind === 'random64' ? `r64|${s.order}` : `c|${s.k}|${s.skew}|${s.order}`);

/** Column as stored: hi/lo 32-bit halves (hi is always 0 for codes). */
export type Column = { spec: Spec; lo: Uint32Array; hi: Uint32Array; width: number; plainBits: number; h0: number; p: Float64Array | null };

export function zipf(k: number, s: number) {
  const p = new Float64Array(k);
  let z = 0;
  for (let r = 1; r <= k; r++) z += 1 / Math.pow(r, s);
  for (let r = 1; r <= k; r++) p[r - 1] = 1 / Math.pow(r, s) / z;
  return p;
}

export function entropyOf(p: ArrayLike<number>) {
  let h = 0;
  for (let i = 0; i < p.length; i++) if (p[i] > 0) h -= p[i] * Math.log2(p[i]);
  return h;
}

export function makeColumn(spec: Spec): Column {
  const W = windowOf(spec.order);
  const lo = new Uint32Array(N);
  const hi = new Uint32Array(N);
  if (spec.kind === 'codes') {
    const K = spec.k;
    const p = zipf(K, spec.skew);
    // A fixed permutation maps frequency rank to code, so the most common value is not simply 0.
    const permRng = makeRng(7919 * K + 13);
    const perm = Array.from({ length: K }, (_, i) => i);
    for (let i = K - 1; i > 0; i--) {
      const j = Math.floor(permRng() * (i + 1));
      [perm[i], perm[j]] = [perm[j], perm[i]];
    }
    const cdf = new Float64Array(K);
    let c = 0;
    for (let i = 0; i < K; i++) cdf[i] = c += p[i];
    const rng = makeRng(1000 + K * 31 + Math.round(spec.skew * 10));
    for (let i = 0; i < N; i++) {
      const r = rng();
      let a = 0;
      let b = K - 1;
      while (a < b) {
        const m = (a + b) >> 1;
        if (cdf[m] > r) b = m;
        else a = m + 1;
      }
      lo[i] = perm[a];
    }
    for (let s = 0; s < N; s += W) lo.subarray(s, Math.min(N, s + W)).sort();
    let max = 0;
    for (let i = 0; i < N; i++) if (lo[i] > max) max = lo[i];
    return { spec, lo, hi, width: Math.max(1, 32 - Math.clz32(max)), plainBits: 32, h0: entropyOf(p), p };
  }
  // Uniform 64-bit ids. makeRng yields 10^6 levels per call; four calls cover ~2^79.7 and are folded mod 2^64.
  const rng = makeRng(424242);
  const big = new BigUint64Array(N);
  const M = 1000000n;
  for (let i = 0; i < N; i++) {
    const a = BigInt(Math.floor(rng() * 1e6));
    const b = BigInt(Math.floor(rng() * 1e6));
    const c = BigInt(Math.floor(rng() * 1e6));
    const d = BigInt(Math.floor(rng() * 1e6));
    big[i] = BigInt.asUintN(64, ((a * M + b) * M + c) * M + d);
  }
  for (let s = 0; s < N; s += W) big.subarray(s, Math.min(N, s + W)).sort();
  for (let i = 0; i < N; i++) {
    lo[i] = Number(big[i] & 0xffffffffn);
    hi[i] = Number(big[i] >> 32n);
  }
  return { spec, lo, hi, width: 64, plainBits: 64, h0: 64, p: null };
}

/* ------------------------------------------------------------ byte writer */

class ByteSink {
  bytes: number[] = [];
  private acc = 0;
  private nbits = 0;
  /** LSB-first packing (Parquet's hybrid order). w <= 16 per call. */
  bits(v: number, w: number) {
    this.acc += v * 2 ** this.nbits;
    this.nbits += w;
    while (this.nbits >= 8) {
      this.bytes.push(this.acc & 0xff);
      this.acc = Math.floor(this.acc / 256);
      this.nbits -= 8;
    }
  }
  /** Write value i in w bits (w may exceed the value's own width), low half first. */
  value(col: Column, i: number, w: number) {
    this.word(col.lo[i], Math.min(32, w));
    if (w > 32) this.word(col.hi[i], w - 32);
  }
  zeros(w: number) {
    for (let rem = w; rem > 0; rem -= 16) this.bits(0, Math.min(16, rem));
  }
  private word(v: number, w: number) {
    this.bits(v & 0xffff, Math.min(16, w));
    if (w > 16) this.bits(v >>> 16, w - 16);
  }
  align() {
    if (this.nbits > 0) this.bits(0, 8 - this.nbits);
  }
  uleb(v: number) {
    this.align();
    do {
      let b = v % 128;
      v = Math.floor(v / 128);
      if (v > 0) b |= 128;
      this.bytes.push(b);
    } while (v > 0);
  }
}

const same = (c: Column, i: number, j: number) => c.lo[i] === c.lo[j] && c.hi[i] === c.hi[j];

/** Port of Arrow C++ RleBitPackedEncoder (parquet RLE/bit-packing hybrid). Returns emitted bytes incl. 1 width byte. */
export function rleHybrid(col: Column) {
  const w = col.width;
  const out = new ByteSink();
  out.bytes.push(w); // dictionary-index data page: bit width as one byte
  let buffered: number[] = [];
  let current = -1;
  let repeat = 0;
  let literal = 0;
  let indicatorAt = -1;
  let rleRuns = 0;
  let literalRuns = 0;

  const flushLiteral = (updateIndicator: boolean) => {
    if (indicatorAt < 0) {
      out.align();
      indicatorAt = out.bytes.length;
      out.bytes.push(0);
    }
    for (const i of buffered) out.value(col, i, w);
    buffered = [];
    if (updateIndicator) {
      out.bytes[indicatorAt] = ((literal / 8) << 1) | 1;
      indicatorAt = -1;
      literal = 0;
      literalRuns++;
    }
  };
  const flushRepeated = () => {
    out.uleb(repeat * 2);
    out.value(col, current, Math.ceil(w / 8) * 8);
    out.align();
    buffered = [];
    repeat = 0;
    rleRuns++;
  };
  const flushBuffered = (done: boolean) => {
    if (repeat >= 8) {
      buffered = [];
      if (literal !== 0) flushLiteral(true);
      return;
    }
    literal += buffered.length;
    const groups = literal / 8;
    if (groups + 1 >= 64) flushLiteral(true);
    else flushLiteral(done);
    repeat = 0;
  };

  for (let i = 0; i < N; i++) {
    if (current >= 0 && same(col, current, i)) {
      repeat++;
      if (repeat > 8) continue;
    } else {
      if (repeat >= 8) flushRepeated();
      repeat = 1;
      current = i;
    }
    buffered.push(i);
    if (buffered.length === 8) flushBuffered(false);
  }
  // Flush()
  if (literal > 0 || repeat > 0 || buffered.length > 0) {
    const allRepeat = literal === 0 && (repeat === buffered.length || buffered.length === 0);
    if (repeat > 0 && allRepeat) flushRepeated();
    else {
      const pad = buffered.length === 0 ? 0 : 8 - buffered.length;
      literal += buffered.length + pad;
      // padding values are zeros
      if (indicatorAt < 0) {
        out.align();
        indicatorAt = out.bytes.length;
        out.bytes.push(0);
      }
      for (const i of buffered) out.value(col, i, w);
      for (let j = 0; j < pad; j++) out.zeros(w);
      buffered = [];
      out.bytes[indicatorAt] = ((literal / 8) << 1) | 1;
      indicatorAt = -1;
      literal = 0;
      literalRuns++;
      repeat = 0;
    }
  }
  out.align();
  return { bytes: Uint8Array.from(out.bytes), rleRuns, literalRuns };
}

/* ------------------------------------------------------ DELTA_BINARY_PACKED */

const bitLenBig = (v: bigint) => (v === 0n ? 0 : v.toString(2).length);
const ulebBytes = (v: bigint) => Math.max(1, Math.ceil(bitLenBig(v) / 7));

export function deltaBinaryPacked(col: Column) {
  const BLOCK = 128;
  const MINI = 4;
  const PER = BLOCK / MINI;
  const is64 = col.spec.kind === 'random64';
  const val = (i: number) => (is64 ? BigInt.asIntN(64, (BigInt(col.hi[i]) << 32n) | BigInt(col.lo[i])) : BigInt(col.lo[i]));
  const zig = (v: bigint) => BigInt.asUintN(64, (v << 1n) ^ (v >> 63n));
  let bits = 0;
  bits += 8 * (ulebBytes(BigInt(BLOCK)) + ulebBytes(BigInt(MINI)) + ulebBytes(BigInt(N)) + ulebBytes(zig(val(0))));
  let prev = val(0);
  let miniBitsTotal = 0;
  for (let b = 1; b < N; b += BLOCK) {
    const end = Math.min(N, b + BLOCK);
    const deltas: bigint[] = [];
    for (let i = b; i < end; i++) {
      const v = val(i);
      deltas.push(BigInt.asIntN(64, v - prev));
      prev = v;
    }
    let min = deltas[0];
    for (const d of deltas) if (d < min) min = d;
    bits += 8 * ulebBytes(zig(min));
    bits += 8 * MINI;
    for (let m = 0; m * PER < deltas.length; m++) {
      let maxRel = 0n;
      for (let j = m * PER; j < Math.min(deltas.length, (m + 1) * PER); j++) {
        const rel = BigInt.asUintN(64, deltas[j] - min);
        if (rel > maxRel) maxRel = rel;
      }
      const w = Math.min(is64 ? 64 : 32, bitLenBig(maxRel));
      const mb = Math.ceil((PER * w) / 8) * 8;
      bits += mb;
      miniBitsTotal += mb;
    }
  }
  return { bits, miniBitsTotal };
}

/* ------------------------------------------------------------- histograms */

/** Distinct values and their counts, plus each position's symbol index. */
export function histogram(col: Column) {
  const order = new Uint32Array(N);
  for (let i = 0; i < N; i++) order[i] = i;
  const idx = Array.from(order).sort((a, b) => (col.hi[a] - col.hi[b]) || (col.lo[a] - col.lo[b]));
  const sym = new Uint32Array(N);
  const counts: number[] = [];
  let d = -1;
  for (let j = 0; j < N; j++) {
    const i = idx[j];
    if (j === 0 || !same(col, idx[j - 1], i)) {
      d++;
      counts.push(0);
    }
    counts[d]++;
    sym[i] = d;
  }
  return { sym, counts, distinct: counts.length };
}

export function empiricalH0(counts: number[]) {
  let h = 0;
  for (const c of counts) h -= (c / N) * Math.log2(c / N);
  return h;
}

/** Plug-in conditional entropy of order k (1 or 2), bits per value. */
export function empiricalHk(sym: Uint32Array, distinct: number, k: 1 | 2) {
  const n = N - k;
  const joint = new Float64Array(n);
  for (let i = k; i < N; i++) {
    const ctx = k === 1 ? sym[i - 1] : sym[i - 2] * distinct + sym[i - 1];
    joint[i - k] = ctx * distinct + sym[i];
  }
  joint.sort();
  let h = 0;
  let j = 0;
  let contexts = 0;
  while (j < n) {
    const ctx = Math.floor(joint[j] / distinct);
    let ctxTotal = 0;
    const runs: number[] = [];
    while (j < n && Math.floor(joint[j] / distinct) === ctx) {
      let r = 0;
      const key = joint[j];
      while (j < n && joint[j] === key) {
        r++;
        j++;
      }
      runs.push(r);
      ctxTotal += r;
    }
    contexts++;
    for (const r of runs) h -= r * Math.log2(r / ctxTotal);
  }
  return { h: h / n, contexts };
}

let LF: Float64Array | null = null;
function log2Fact() {
  if (!LF) {
    LF = new Float64Array(N + 1);
    for (let i = 2; i <= N; i++) LF[i] = LF[i - 1] + Math.log2(i);
  }
  return LF;
}

/**
 * Expected bits a per-block sort destroys for value frequencies p (analyse() passes the measured frequencies):
 * each full block of W i.i.d. values loses log2 W! - sum_v E[log2 n_v!], with n_v ~ Binomial(W, p_v).
 * All block sizes divide N. For random 64-bit ids every n_v is 0 or 1 (a collision among 65,536 draws has
 * probability ~1e-10), so a block loses log2 W! bits.
 */
export function orderingBitsRemoved(spec: Spec, p: Float64Array | null) {
  const W = windowOf(spec.order);
  if (W === 1) return 0;
  const lf = log2Fact();
  let perBlock = lf[W];
  if (p) {
    for (let v = 0; v < p.length; v++) {
      const pv = p[v];
      if (pv <= 0) continue;
      const mean = W * pv;
      const sd = Math.sqrt(W * pv * (1 - pv));
      const lo = Math.max(0, Math.floor(mean - 12 * sd - 6));
      const hi = Math.min(W, Math.ceil(mean + 12 * sd + 6));
      const l2p = Math.log2(pv);
      const l2q = pv < 1 ? Math.log2(1 - pv) : 0;
      let e = 0;
      for (let n = Math.max(2, lo); n <= hi; n++) {
        const logPmf = lf[W] - lf[n] - lf[W - n] + n * l2p + (W - n) * l2q;
        e += Math.pow(2, logPmf) * lf[n];
      }
      perBlock -= e;
    }
  }
  return (N / W) * perBlock;
}

/* ---------------------------------------------------------- entropy coders */

export function huffmanLengths(counts: number[]) {
  const n = counts.length;
  if (n === 1) return [1];
  // two-queue Huffman over counts sorted ascending
  const order = counts.map((c, i) => [c, i] as const).sort((a, b) => a[0] - b[0] || a[1] - b[1]);
  const weight: number[] = order.map((o) => o[0]);
  const parent: number[] = new Array(2 * n - 1).fill(-1);
  let leaf = 0;
  let inner = n;
  let innerEnd = n;
  const pick = () => {
    if (leaf < n && (inner >= innerEnd || weight[leaf] <= weight[inner])) return leaf++;
    return inner++;
  };
  for (let t = 0; t < n - 1; t++) {
    const a = pick();
    const b = pick();
    weight[innerEnd] = weight[a] + weight[b];
    parent[a] = innerEnd;
    parent[b] = innerEnd;
    innerEnd++;
  }
  const depth = new Array(2 * n - 1).fill(0);
  for (let v = 2 * n - 3; v >= 0; v--) depth[v] = depth[parent[v]] + 1;
  const lengths = new Array(n).fill(0);
  for (let j = 0; j < n; j++) lengths[order[j][1]] = depth[j];
  return lengths;
}

export function modelBits(col: Column, distinct: number) {
  return distinct * (17 + (col.spec.kind === 'random64' ? 64 : 0));
}

/** Byte-wise rANS (ryg_rans layout): 32-bit state, L = 2^23, scale 2^16 (== N, so freq = count). */
export function rans(sym: Uint32Array, counts: number[]) {
  const SCALE = 16;
  const L = 1 << 23;
  const cum = new Float64Array(counts.length + 1);
  for (let s = 0; s < counts.length; s++) cum[s + 1] = cum[s] + counts[s];
  const out: number[] = [];
  let x = L;
  for (let i = N - 1; i >= 0; i--) {
    const s = sym[i];
    const f = counts[s];
    const xMax = ((L >>> SCALE) << 8) * f;
    while (x >= xMax) {
      out.push(x & 0xff);
      x = Math.floor(x / 256);
    }
    x = Math.floor(x / f) * (1 << SCALE) + (x % f) + cum[s];
  }
  for (let b = 0; b < 4; b++) {
    out.push(x & 0xff);
    x = Math.floor(x / 256);
  }
  out.reverse();
  return { bytes: out, cum };
}

export function ransDecode(bytes: number[], counts: number[], cum: Float64Array) {
  const L = 1 << 23;
  const lookup = new Uint32Array(1 << 16);
  for (let s = 0; s < counts.length; s++) for (let c = cum[s]; c < cum[s + 1]; c++) lookup[c] = s;
  let p = 0;
  let x = 0;
  for (let b = 0; b < 4; b++) x = x * 256 + bytes[p++];
  const res = new Uint32Array(N);
  for (let i = 0; i < N; i++) {
    const slot = x % 65536;
    const s = lookup[slot];
    res[i] = s;
    x = counts[s] * Math.floor(x / 65536) + slot - cum[s];
    while (x < L) x = x * 256 + bytes[p++];
  }
  return res;
}

/* ------------------------------------------------------------ plain bytes */

export function plainBytes(col: Column) {
  const bpv = col.plainBits / 8;
  const b = new Uint8Array(N * bpv);
  const dv = new DataView(b.buffer);
  for (let i = 0; i < N; i++) {
    dv.setUint32(i * bpv, col.lo[i], true);
    if (bpv === 8) dv.setUint32(i * bpv + 4, col.hi[i], true);
  }
  return b;
}

export function fnv1a(bytes: ArrayLike<number>) {
  let h = 0x811c9dc5;
  for (let i = 0; i < bytes.length; i++) {
    h ^= bytes[i];
    h = Math.imul(h, 0x01000193) >>> 0;
  }
  return h >>> 0;
}

/* ------------------------------------------------------------- the analysis */

export type Analysis = ReturnType<typeof analyse>;

export function analyse(spec: Spec, col: Column = makeColumn(spec)) {
  const { sym, counts, distinct } = histogram(col);
  const h0emp = empiricalH0(counts);
  const h1 = empiricalHk(sym, distinct, 1);
  const h2 = empiricalHk(sym, distinct, 2);
  // H0 is measured from this column's histogram; for random ids (every value distinct) the histogram carries no
  // information about the 2^64-value domain, so the generator's 64 bits are used instead.
  const h0 = col.spec.kind === 'random64' ? 64 : h0emp;
  const pHat = col.spec.kind === 'random64' ? null : Float64Array.from(counts, (c) => c / N);
  const removed = orderingBitsRemoved(spec, pHat);
  const floorBits = Math.max(0, N * h0 - removed);
  const floor = floorBits / N;
  const clampK = (h: number) => Math.min(h0, Math.max(floor, h));
  const rle = rleHybrid(col);
  const delta = deltaBinaryPacked(col);
  const lengths = huffmanLengths(counts);
  let huffPayload = 0;
  for (let s = 0; s < distinct; s++) huffPayload += counts[s] * lengths[s];
  const model = modelBits(col, distinct);
  const r = rans(sym, counts);
  const plain = plainBytes(col);
  return {
    spec,
    width: col.width,
    plainBits: col.plainBits,
    distinct,
    h0,
    h0source: col.h0,
    h0emp,
    h1emp: h1.h,
    h2emp: h2.h,
    h1: clampK(h1.h),
    h2: clampK(h2.h),
    contexts1: h1.contexts,
    contexts2: h2.contexts,
    floor,
    removedPerValue: removed / N,
    bits: {
      plain: col.plainBits,
      bitpack: (N * col.width + 8) / N,
      rle: (rle.bytes.length * 8) / N,
      delta: delta.bits / N,
      huffman: (huffPayload + model) / N,
      rans: (r.bytes.length * 8 + model) / N,
    },
    huffPayload: huffPayload / N,
    ransPayload: (r.bytes.length * 8) / N,
    modelPerValue: model / N,
    rleRuns: rle.rleRuns,
    literalRuns: rle.literalRuns,
    plainHash: fnv1a(plain),
    rleHash: fnv1a(rle.bytes),
    rleBytes: rle.bytes.length,
  };
}

/* ------------------------------------------------------------ view model */

export type BarId = 'plain' | 'bitpack' | 'rle' | 'delta' | 'huffman' | 'rans' | 'lz4' | 'zstd' | 'rlezstd';
export type Bar = { id: BarId; label: string; family: 0 | 1 | 2; bits: number; bytes: number };

const cache = new Map<string, Analysis & { head: number[]; fixtureOk: boolean; fixture: readonly number[] | null; runLength: number }>();

/** analyse() plus the first 1,024 values (shade bins), the fixture row and a hash check. Cached per column. */
export function study(spec: Spec) {
  const key = specKey(spec);
  const hit = cache.get(key);
  if (hit) return hit;
  const col = makeColumn(spec);
  const a = analyse(spec, col);
  const head: number[] = [];
  // Shade by dense rank among the distinct values shown, so order is visible whatever range those values occupy.
  const shown = Array.from({ length: 1024 }, (_, i) => i).sort((x, y) => col.hi[x] - col.hi[y] || col.lo[x] - col.lo[y]);
  const dense = new Array<number>(1024);
  let d = 0;
  shown.forEach((i, j) => {
    if (j > 0 && !same(col, shown[j - 1], i)) d++;
    dense[i] = d;
  });
  for (let i = 0; i < 1024; i++) head.push(d === 0 ? 2 : Math.round((dense[i] / d) * 4));
  let runs = 1;
  for (let i = 1; i < N; i++) if (!same(col, i - 1, i)) runs++;
  const row = CODEC_FIXTURE[key] ?? null;
  const fixtureOk = !!row && row[0] === a.plainHash && row[1] === a.rleHash;
  const out = { ...a, head, fixture: row, fixtureOk, runLength: N / runs };
  cache.set(key, out);
  return out;
}

export function barsFor(st: ReturnType<typeof study>, level: (typeof LEVELS)[number]): Bar[] {
  const b = st.bits;
  const bars: Bar[] = [
    { id: 'plain', label: st.plainBits === 32 ? 'PLAIN (INT32)' : 'PLAIN (INT64)', family: 0, bits: b.plain, bytes: (b.plain * N) / 8 },
    { id: 'bitpack', label: `Bit-packed (${st.width} bit${st.width === 1 ? '' : 's'})`, family: 0, bits: b.bitpack, bytes: (b.bitpack * N) / 8 },
    { id: 'rle', label: 'RLE / bit-pack hybrid', family: 0, bits: b.rle, bytes: st.rleBytes },
    { id: 'delta', label: 'DELTA_BINARY_PACKED', family: 0, bits: b.delta, bytes: (b.delta * N) / 8 },
    { id: 'huffman', label: 'Huffman (order 0)', family: 1, bits: b.huffman, bytes: (b.huffman * N) / 8 },
    { id: 'rans', label: 'rANS (order 0)', family: 1, bits: b.rans, bytes: (b.rans * N) / 8 },
  ];
  if (st.fixtureOk && st.fixture) {
    const li = LEVELS.indexOf(level);
    const f = st.fixture;
    bars.push({ id: 'lz4', label: 'PLAIN → LZ4', family: 2, bits: (f[2] * 8) / N, bytes: f[2] });
    bars.push({ id: 'zstd', label: `PLAIN → ZSTD ${level}`, family: 2, bits: (f[3 + li] * 8) / N, bytes: f[3 + li] });
    bars.push({ id: 'rlezstd', label: `RLE hybrid → ZSTD ${level}`, family: 2, bits: (f[7 + li] * 8) / N, bytes: f[7 + li] });
  }
  return bars;
}

/* ------------------------------------------------------------------- view */

const FAMILY = ['var(--viz-1)', 'var(--viz-2)', 'var(--viz-3)'] as const;
const FAMILY_NAME = ['Lightweight encoding (run here)', 'Order-0 entropy coder (run here)', 'Block compressor (real CLI, precomputed)'] as const;
const C_H0 = 'var(--viz-4)';
const C_H1 = 'var(--viz-5)';
const C_FLOOR = 'var(--viz-ink)';
const SHADE_OPACITY = [0.08, 0.28, 0.48, 0.68, 0.9];

const SVG_W = 600;
const PX0 = 150;
const PW = 330;
const LMIN = -13;
const LMAX = 7;
const xOf = (bits: number) => PX0 + ((Math.min(LMAX, Math.max(LMIN, Math.log2(Math.max(bits, 2 ** LMIN)))) - LMIN) / (LMAX - LMIN)) * PW;

export const fmtBits = (b: number) => (b === 0 ? '0' : b >= 10 ? b.toFixed(1) : b >= 1 ? b.toFixed(2) : b >= 0.01 ? b.toFixed(3) : b >= 0.0001 ? b.toFixed(4) : b.toExponential(1));
const fmtRatio = (r: number) => (r >= 100 ? fmtNum(r) : r >= 10 ? r.toFixed(1) : r.toFixed(2)) + '×';
const ORDER_LABEL: Record<Order, string> = { random: 'arrival order', w1024: 'sorted within 1,024-row blocks', w8192: 'sorted within 8,192-row blocks', sorted: 'fully sorted' };

export default function EntropyFloorLab() {
  const [kind, setKind] = useState<Kind>('codes');
  const [kIdx, setKIdx] = useState(2);
  const [sIdx, setSIdx] = useState(4);
  const [order, setOrder] = useState<Order>('random');
  const [level, setLevel] = useState<'1' | '3' | '9' | '19'>('3');

  const spec: Spec = { kind, k: K_STEPS[kIdx], skew: SKEW_STEPS[sIdx], order };
  const st = useMemo(() => study(spec), [kind, kIdx, sIdx, order]); // eslint-disable-line react-hooks/exhaustive-deps
  const lvl = Number(level) as (typeof LEVELS)[number];
  const bars = barsFor(st, lvl);
  const best = bars.reduce((m, b) => (b.bits < m.bits ? b : m), bars[0]);
  const huffLoss = st.bits.huffman - st.bits.rans;

  const ROW = 22;
  const TOP = 50;
  const rowsY: number[] = [];
  let y = TOP;
  bars.forEach((b, i) => {
    if (i > 0 && b.family !== bars[i - 1].family) y += 8;
    rowsY.push(y);
    y += ROW;
  });
  const plotBottom = y + 2;
  const H = plotBottom + 50;
  const ticks = [-12, -8, -4, 0, 3, 6];
  const tickLabel = (e: number) => (e >= 0 ? String(2 ** e) : `1/${2 ** -e}`);
  const xFloor = xOf(st.floor);

  const lineLabels = [
    { key: 'h0', text: `H₀ ${fmtBits(st.h0)}`, x: xOf(st.h0), color: C_H0, dash: '6 3', y: 12 },
    { key: 'h1', text: `H₁ ${fmtBits(st.h1)}`, x: xOf(st.h1), color: C_H1, dash: '2 2', y: 26 },
    { key: 'fl', text: `information floor ${fmtBits(st.floor)}`, x: xFloor, color: C_FLOOR, dash: '', y: 40 },
  ];

  const runNote =
    kind === 'codes' && st.bits.rle > 0.9 * st.bits.bitpack && order !== 'random'
      ? ` Runs here average ${fmtNum(st.runLength, 1)} values, shorter than the 8 equal values Arrow's Parquet encoder needs before it writes an RLE run, so the hybrid is still bit-packing.`
      : '';

  return (
    <VizPanel
      title="Encoders against the entropy floor"
      subtitle="65,536 values are generated, stored in the order you pick and actually encoded. Bars are measured bits per value on a log scale; the lines are the floors computed from the same column."
      controls={
        <>
          <Segmented
            label="Column"
            value={kind}
            onChange={setKind}
            options={[
              { value: 'codes', label: 'Small-integer codes', title: 'INT32 values from a Zipf distribution over K distinct codes' },
              { value: 'random64', label: 'Random 64-bit ids', title: 'Uniform INT64 values: every value distinct' },
            ]}
          />
          <Slider label="Distinct values K" min={0} max={K_STEPS.length - 1} value={kIdx} onChange={setKIdx} format={(i) => (kind === 'codes' ? fmtNum(K_STEPS[i]) : '2^64')} disabled={kind !== 'codes'} />
          <Slider label="Skew (Zipf s)" min={0} max={SKEW_STEPS.length - 1} value={sIdx} onChange={setSIdx} format={(i) => (kind === 'codes' ? String(SKEW_STEPS[i]) : 'uniform')} disabled={kind !== 'codes'} />
          <Segmented
            label="Physical order"
            value={order}
            onChange={setOrder}
            options={[
              { value: 'random', label: 'Arrival' },
              { value: 'w1024', label: 'Sorted per 1,024' },
              { value: 'w8192', label: 'Sorted per 8,192' },
              { value: 'sorted', label: 'Fully sorted' },
            ]}
          />
          <Segmented
            label="ZSTD level"
            value={level}
            onChange={setLevel}
            options={[
              { value: '1', label: '1' },
              { value: '3', label: '3' },
              { value: '9', label: '9' },
              { value: '19', label: '19' },
            ]}
          />
        </>
      }
      legend={
        <Legend
          items={[
            { label: FAMILY_NAME[0], color: FAMILY[0] },
            { label: FAMILY_NAME[1], color: FAMILY[1] },
            { label: FAMILY_NAME[2], color: FAMILY[2] },
            { label: 'H₀: order-0 entropy', color: C_H0, shape: 'line' },
            { label: 'H₁: order-1 conditional entropy', color: C_H1, shape: 'line' },
            { label: 'Information floor (entropy of the stored column)', color: C_FLOOR, shape: 'line' },
          ]}
        />
      }
      stats={
        <Stats
          items={[
            { label: 'H₀ (order 0)', value: `${fmtBits(st.h0)} bits`, hint: kind === 'random64' ? 'Every value is distinct, so the histogram says nothing; each uniform 64-bit id carries 64 bits.' : 'Entropy of this column’s value histogram. Reordering cannot change it.' },
            { label: 'H₁ (given previous value)', value: `${fmtBits(st.h1)} bits`, hint: `Measured order-1 conditional entropy, kept between the information floor and H₀ (raw plug-in estimate ${fmtBits(st.h1emp)} over ${fmtNum(st.contexts1)} contexts).` },
            { label: 'Information floor', value: `${fmtBits(st.floor)} bits`, hint: 'H₀ minus the ordering information the per-block sort destroyed, computed from the column’s value frequencies.' },
            { label: 'Ratio: best measured / possible', value: `${fmtRatio(st.plainBits / best.bits)} / ${fmtRatio(st.plainBits / Math.max(st.floor, 1e-9))}`, hint: `Best measured: ${best.label} at ${fmtBits(best.bits)} bits. Possible: PLAIN ${st.plainBits} bits ÷ information floor.` },
          ]}
        />
      }
      note={
        <Note>
          {kind === 'random64' ? (
            order === 'random' ? (
              <>
                <strong>Uniform 64-bit ids carry 64 bits each, and every bar stays at or above 64 at every ZSTD level.</strong> Huffman and rANS are worse still, at {fmtBits(st.bits.huffman)} bits: each distinct value has to be listed in the model. Sort the column: only the order can be removed.
              </>
            ) : (
              <>
                <strong>Sorting removed {fmtBits(st.removedPerValue)} bits per value, all of it ordering information; the floor is now {fmtBits(st.floor)} bits.</strong> DELTA_BINARY_PACKED stores the gaps and reaches {fmtBits(st.bits.delta)}; ZSTD cannot subtract, so it stays at {st.fixtureOk ? fmtBits(bars.find((b) => b.id === 'zstd')!.bits) : '64'} bits at level {level}.
              </>
            )
          ) : order === 'random' ? (
            <>
              <strong>In arrival order the previous value says nothing about the next, so H₁ = H₀ = {fmtBits(st.h0)} bits and that is the floor.</strong> rANS lands {fmtBits(Math.max(0, st.bits.rans - st.h0))} bits above it; Huffman spends {fmtBits(Math.max(0, huffLoss))} bits more than rANS because every code length is a whole number of bits; bit-packing spends {st.width} bit{st.width === 1 ? '' : 's'}. Now sort the column and watch which bars follow the floor down.
            </>
          ) : (
            <>
              <strong>{ORDER_LABEL[order][0].toUpperCase() + ORDER_LABEL[order].slice(1)}: H₀ is still {fmtBits(st.h0)} bits, but the information floor fell to {fmtBits(st.floor)}.</strong> Huffman and rANS code each value alone, so they cannot see the order and do not move. RLE hybrid is at {fmtBits(st.bits.rle)}, DELTA_BINARY_PACKED at {fmtBits(st.bits.delta)}
              {st.fixtureOk ? `, PLAIN → ZSTD ${level} at ${fmtBits(bars.find((b) => b.id === 'zstd')!.bits)}` : ''}.{runNote}
            </>
          )}
        </Note>
      }
      table={
        <table className="viz-table">
          <thead>
            <tr>
              <th>Encoding</th>
              <th>Bits per value</th>
              <th>Bytes for 65,536 values</th>
              <th>Ratio vs PLAIN</th>
              <th>Above the floor by</th>
            </tr>
          </thead>
          <tbody>
            {bars.map((b) => (
              <tr key={b.id}>
                <td>{b.label}</td>
                <td>{fmtBits(b.bits)}</td>
                <td>{fmtNum(Math.round(b.bytes))}</td>
                <td>{fmtRatio(st.plainBits / b.bits)}</td>
                <td>{fmtBits(b.bits - st.floor)}</td>
              </tr>
            ))}
            <tr>
              <td>H₀ of this column (of the distribution you shaped)</td>
              <td>
                {fmtBits(st.h0emp)} ({fmtBits(st.h0source)})
              </td>
              <td colSpan={3}>{kind === 'random64' ? 'All values distinct: the histogram gives log2 65,536 = 16, so the lab uses the 64 bits each id carries.' : 'Order-blind: the same for every physical order.'}</td>
            </tr>
            <tr>
              <td>H₁ plug-in estimate</td>
              <td>{fmtBits(st.h1emp)}</td>
              <td colSpan={3}>{fmtNum(st.contexts1)} distinct contexts; shown as {fmtBits(st.h1)} after keeping it between the floor and H₀.</td>
            </tr>
            <tr>
              <td>Model cost inside Huffman and rANS</td>
              <td>{fmtBits(st.modelPerValue)}</td>
              <td colSpan={3}>17 bits per distinct value for its count{kind === 'random64' ? ', plus the 64-bit value itself' : ''}.</td>
            </tr>
          </tbody>
        </table>
      }
    >
      <svg viewBox={`0 0 ${SVG_W} 62`} width={SVG_W} height={62} style={{ minWidth: 520 }} role="img" aria-label={`First 1,024 stored values, ${ORDER_LABEL[order]}`}>
        <text x={0} y={11} fontSize={11} fill="var(--viz-ink-2)">
          First 1,024 values as stored, 256 per row (stronger shade = larger among these)
        </text>
        {st.head.map((v, i) => (
          <rect key={i} x={(i % 256) * (SVG_W / 256)} y={18 + Math.floor(i / 256) * 11} width={SVG_W / 256 + 0.2} height={10} fill="var(--viz-ink)" fillOpacity={SHADE_OPACITY[v]} />
        ))}
      </svg>
      <svg viewBox={`0 0 ${SVG_W} ${H}`} width={SVG_W} height={H} style={{ minWidth: 520 }} role="img" aria-label={`Bits per value: best encoder ${best.label} at ${fmtBits(best.bits)} bits; H0 ${fmtBits(st.h0)}, H1 ${fmtBits(st.h1)}, information floor ${fmtBits(st.floor)}`}>
        <defs>
          <pattern id="entropy-floor-hatch" width={6} height={6} patternUnits="userSpaceOnUse" patternTransform="rotate(45)">
            <line x1={0} y1={0} x2={0} y2={6} stroke="var(--viz-ink-muted)" strokeOpacity={0.35} strokeWidth={2} />
          </pattern>
        </defs>
        <rect x={PX0} y={TOP - 4} width={Math.max(0, xFloor - PX0)} height={plotBottom - TOP + 4} fill="url(#entropy-floor-hatch)" />
        <rect x={PX0} y={plotBottom + 36} width={12} height={10} fill="url(#entropy-floor-hatch)" stroke="var(--viz-ink-muted)" strokeWidth={0.5} />
        <text x={PX0 + 18} y={plotBottom + 45} fontSize={10} fill="var(--viz-ink-2)">
          left of the information floor: no lossless encoding gets there, on average
        </text>
        {ticks.map((e) => (
          <g key={e}>
            <line x1={xOf(2 ** e)} x2={xOf(2 ** e)} y1={TOP - 4} y2={plotBottom} stroke="var(--viz-grid)" />
            <text x={xOf(2 ** e)} y={plotBottom + 14} fontSize={10} textAnchor="middle" fill="var(--viz-ink-2)">
              {tickLabel(e)}
            </text>
          </g>
        ))}
        <text x={PX0 + PW / 2} y={plotBottom + 26} fontSize={10} textAnchor="middle" fill="var(--viz-ink-2)">
          bits per value (log scale)
        </text>
        {bars.map((b, i) => {
          const yy = rowsY[i];
          const xe = xOf(b.bits);
          return (
            <g key={b.id}>
              <text x={PX0 - 8} y={yy + 14} fontSize={11} textAnchor="end" fill="var(--viz-ink)">
                {b.label}
              </text>
              <rect x={PX0} y={yy + 3} width={Math.max(2, xe - PX0)} height={ROW - 7} rx={2} fill={FAMILY[b.family]} />
              <text x={PX0 + PW + 6} y={yy + 14} fontSize={11} fill="var(--viz-ink)">
                {fmtBits(b.bits)} b · {fmtRatio(st.plainBits / b.bits)}
              </text>
            </g>
          );
        })}
        {lineLabels.map((l) => (
          <g key={l.key}>
            <line x1={l.x} x2={l.x} y1={l.y + 3} y2={plotBottom} stroke={l.color} strokeWidth={l.key === 'fl' ? 2.5 : 2} strokeDasharray={l.dash || undefined} />
            <text x={l.x > PX0 + PW * 0.6 ? l.x - 4 : l.x + 4} y={l.y} fontSize={10.5} textAnchor={l.x > PX0 + PW * 0.6 ? 'end' : 'start'} fill="var(--viz-ink)">
              {l.text}
            </text>
          </g>
        ))}
      </svg>
      {!st.fixtureOk ? (
        <p style={{ fontSize: '0.75rem', color: 'var(--viz-ink-2)', margin: '0.3rem 0 0' }}>The precomputed LZ4/ZSTD sizes do not match these bytes, so those bars are hidden.</p>
      ) : (
        <p style={{ fontSize: '0.75rem', color: 'var(--viz-ink-2)', margin: '0.3rem 0 0' }}>
          LZ4 and ZSTD sizes were measured offline with lz4 1.10.0 and zstd 1.5.6 on these exact bytes (checked by hash). Huffman and rANS include a model of 17 bits per distinct value{kind === 'random64' ? ', plus each 64-bit value' : ''}. See how the{' '}
          <a href={siteHref('/p02-storage-engines/column-stores-analytical-storage/02-lightweight-encodings-rle-dictionary-delta-bit-packing-for-a/')}>encodings</a> work.
        </p>
      )}
    </VizPanel>
  );
}
