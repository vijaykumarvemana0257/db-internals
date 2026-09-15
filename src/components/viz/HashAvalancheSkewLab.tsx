import { useMemo, useState } from 'react';
import { VizPanel, Segmented, Choice, Slider, Legend, Stats, Note, fmtNum, makeRng, useSize } from './Viz';

/**
 * Hash functions under real and hostile key sets.
 *
 * Model (all of it runs in the browser on real implementations, checked against the C reference code):
 *  - Every hash takes the key's bytes and returns 32 bits.
 *      java31  : h = 31*h + byte (the String.hashCode polynomial); the "seed" is the initial h.
 *      fnv1a   : FNV-1a 32-bit; the seed is XORed into the offset basis (not part of FNV itself).
 *      pg      : PostgreSQL hash_bytes (Bob Jenkins' lookup3, little-endian path). A non-zero seed is
 *                applied the way hash_bytes_extended does (b += seed; mix) and the low 32 bits (c) kept.
 *      murmur3 : MurmurHash3_x86_32 with its native seed.
 *      xxh32   : XXH32 with its native seed.
 *      sip24   : SipHash-2-4, key bytes 00..0f with the seed XORed into the first four; the low 32 bits
 *                of the 64-bit tag are used (as RocksDB's cache takes Lower32of64 of its 64-bit hash).
 *  - Buckets are chosen with a power-of-two mask on the low bits, as PostgreSQL's hash join,
 *    simplehash and ClickHouse's HashTable do.
 *  - Integer keys are fed as 8 little-endian bytes. PostgreSQL itself hashes int8 with hashint8, which
 *    folds to 32 bits first; the lab hashes the raw bytes for every function so they are comparable.
 *  - "Ideal random hash" curves come from simulating uniformly random bucket choices (makeRng),
 *    averaged over several runs; the chaining curve is also compared with the Poisson distribution.
 *  - The Murmur3 attack set uses a differential construction: two 4-byte blocks whose mixed values differ
 *    in bit 18 and then bit 31 leave the state identical, whatever the seed. The ×31 attack set uses the
 *    equal-hash substrings "Aa" and "BB".
 */

/* ------------------------------------------------------------ hash functions */

export type HashId = 'java31' | 'fnv1a' | 'pg' | 'murmur3' | 'xxh32' | 'sip24';
export type KeySetId = 'seq' | 'orders' | 'hosts' | 'uuid' | 'collide31' | 'collideMurmur';

const rotl32 = (x: number, r: number) => ((x << r) | (x >>> (32 - r))) >>> 0;
const rotr32 = (x: number, r: number) => (r === 0 ? x >>> 0 : ((x >>> r) | (x << (32 - r))) >>> 0);
const u32le = (k: Uint8Array, o: number) => (k[o] | (k[o + 1] << 8) | (k[o + 2] << 16) | (k[o + 3] << 24)) >>> 0;

export function java31(k: Uint8Array, seed = 0) {
  let h = seed >>> 0;
  for (let i = 0; i < k.length; i++) h = (Math.imul(h, 31) + k[i]) >>> 0;
  return h;
}

export function fnv1a(k: Uint8Array, seed = 0) {
  let h = (0x811c9dc5 ^ seed) >>> 0;
  for (let i = 0; i < k.length; i++) {
    h ^= k[i];
    h = Math.imul(h, 0x01000193) >>> 0;
  }
  return h;
}

/** PostgreSQL src/common/hashfn.c hash_bytes (seed 0) / low 32 bits of hash_bytes_extended (32-bit seed). */
export function pgHashBytes(k: Uint8Array, seed = 0) {
  let len = k.length;
  let a = (0x9e3779b9 + len + 3923095) >>> 0;
  let b = a;
  let c = a;
  const mix = () => {
    a = (a - c) >>> 0; a = (a ^ rotl32(c, 4)) >>> 0; c = (c + b) >>> 0;
    b = (b - a) >>> 0; b = (b ^ rotl32(a, 6)) >>> 0; a = (a + c) >>> 0;
    c = (c - b) >>> 0; c = (c ^ rotl32(b, 8)) >>> 0; b = (b + a) >>> 0;
    a = (a - c) >>> 0; a = (a ^ rotl32(c, 16)) >>> 0; c = (c + b) >>> 0;
    b = (b - a) >>> 0; b = (b ^ rotl32(a, 19)) >>> 0; a = (a + c) >>> 0;
    c = (c - b) >>> 0; c = (c ^ rotl32(b, 4)) >>> 0; b = (b + a) >>> 0;
  };
  if (seed !== 0) {
    b = (b + (seed >>> 0)) >>> 0;
    mix();
  }
  let o = 0;
  while (len >= 12) {
    a = (a + u32le(k, o)) >>> 0;
    b = (b + u32le(k, o + 4)) >>> 0;
    c = (c + u32le(k, o + 8)) >>> 0;
    mix();
    o += 12;
    len -= 12;
  }
  /* eslint-disable no-fallthrough */
  switch (len) {
    case 11: c = (c + (k[o + 10] << 24)) >>> 0;
    case 10: c = (c + (k[o + 9] << 16)) >>> 0;
    case 9: c = (c + (k[o + 8] << 8)) >>> 0;
    case 8: b = (b + u32le(k, o + 4)) >>> 0; a = (a + u32le(k, o)) >>> 0; break;
    case 7: b = (b + (k[o + 6] << 16)) >>> 0;
    case 6: b = (b + (k[o + 5] << 8)) >>> 0;
    case 5: b = (b + k[o + 4]) >>> 0;
    case 4: a = (a + u32le(k, o)) >>> 0; break;
    case 3: a = (a + (k[o + 2] << 16)) >>> 0;
    case 2: a = (a + (k[o + 1] << 8)) >>> 0;
    case 1: a = (a + k[o]) >>> 0;
  }
  /* eslint-enable no-fallthrough */
  c ^= b; c = (c - rotl32(b, 14)) >>> 0;
  a ^= c; a = (a - rotl32(c, 11)) >>> 0;
  b ^= a; b = (b - rotl32(a, 25)) >>> 0;
  c ^= b; c = (c - rotl32(b, 16)) >>> 0;
  a ^= c; a = (a - rotl32(c, 4)) >>> 0;
  b ^= a; b = (b - rotl32(a, 14)) >>> 0;
  c ^= b; c = (c - rotl32(b, 24)) >>> 0;
  return c >>> 0;
}

const MC1 = 0xcc9e2d51;
const MC2 = 0x1b873593;
const murmurMixK = (k: number) => Math.imul(rotl32(Math.imul(k, MC1), 15), MC2) >>> 0;

export function murmur3_32(k: Uint8Array, seed = 0) {
  let h = seed >>> 0;
  const len = k.length;
  const nblocks = len >>> 2;
  for (let i = 0; i < nblocks; i++) {
    h ^= murmurMixK(u32le(k, i * 4));
    h = rotl32(h >>> 0, 13);
    h = (Math.imul(h, 5) + 0xe6546b64) >>> 0;
  }
  const t = nblocks * 4;
  let k1 = 0;
  /* eslint-disable no-fallthrough */
  switch (len & 3) {
    case 3: k1 ^= k[t + 2] << 16;
    case 2: k1 ^= k[t + 1] << 8;
    case 1: k1 ^= k[t]; h ^= murmurMixK(k1 >>> 0);
  }
  /* eslint-enable no-fallthrough */
  h ^= len;
  h ^= h >>> 16;
  h = Math.imul(h, 0x85ebca6b);
  h ^= h >>> 13;
  h = Math.imul(h, 0xc2b2ae35);
  h ^= h >>> 16;
  return h >>> 0;
}

const P1 = 0x9e3779b1;
const P2 = 0x85ebca77;
const P3 = 0xc2b2ae3d;
const P4 = 0x27d4eb2f;
const P5 = 0x165667b1;

export function xxh32(k: Uint8Array, seed = 0) {
  const len = k.length;
  let o = 0;
  let h: number;
  const round = (acc: number, input: number) => Math.imul(rotl32((acc + Math.imul(input, P2)) >>> 0, 13), P1) >>> 0;
  if (len >= 16) {
    let v1 = (seed + P1 + P2) >>> 0;
    let v2 = (seed + P2) >>> 0;
    let v3 = seed >>> 0;
    let v4 = (seed - P1) >>> 0;
    while (o + 16 <= len) {
      v1 = round(v1, u32le(k, o));
      v2 = round(v2, u32le(k, o + 4));
      v3 = round(v3, u32le(k, o + 8));
      v4 = round(v4, u32le(k, o + 12));
      o += 16;
    }
    h = (rotl32(v1, 1) + rotl32(v2, 7) + rotl32(v3, 12) + rotl32(v4, 18)) >>> 0;
  } else {
    h = (seed + P5) >>> 0;
  }
  h = (h + len) >>> 0;
  while (o + 4 <= len) {
    h = (h + Math.imul(u32le(k, o), P3)) >>> 0;
    h = Math.imul(rotl32(h, 17), P4) >>> 0;
    o += 4;
  }
  while (o < len) {
    h = (h + Math.imul(k[o], P5)) >>> 0;
    h = Math.imul(rotl32(h, 11), P1) >>> 0;
    o++;
  }
  h ^= h >>> 15;
  h = Math.imul(h, P2);
  h ^= h >>> 13;
  h = Math.imul(h, P3);
  h ^= h >>> 16;
  return h >>> 0;
}

/** SipHash-2-4 on 32-bit halves. Returns [high, low] of the 64-bit tag. */
export function sipHash24Full(k: Uint8Array, seed = 0): [number, number] {
  // key = bytes 00..0f with the seed XORed into bytes 0..3; k0/k1 little-endian 64-bit words
  const k0l = (0x03020100 ^ seed) >>> 0;
  const k0h = 0x07060504;
  const k1l = 0x0b0a0908;
  const k1h = 0x0f0e0d0c;
  const v = new Uint32Array(8); // v0h v0l v1h v1l v2h v2l v3h v3l
  v[0] = 0x736f6d65 ^ k0h; v[1] = 0x70736575 ^ k0l;
  v[2] = 0x646f7261 ^ k1h; v[3] = 0x6e646f6d ^ k1l;
  v[4] = 0x6c796765 ^ k0h; v[5] = 0x6e657261 ^ k0l;
  v[6] = 0x74656462 ^ k1h; v[7] = 0x79746573 ^ k1l;
  const add = (a: number, b: number) => {
    const lo = (v[a + 1] + v[b + 1]) >>> 0;
    const carry = lo < v[a + 1] ? 1 : 0;
    v[a] = (v[a] + v[b] + carry) >>> 0;
    v[a + 1] = lo;
  };
  const rotl = (a: number, r: number) => {
    let hi = v[a];
    let lo = v[a + 1];
    if (r >= 32) {
      const t = hi; hi = lo; lo = t; r -= 32;
    }
    if (r > 0) {
      const nh = ((hi << r) | (lo >>> (32 - r))) >>> 0;
      const nl = ((lo << r) | (hi >>> (32 - r))) >>> 0;
      hi = nh; lo = nl;
    }
    v[a] = hi; v[a + 1] = lo;
  };
  const xor = (a: number, b: number) => {
    v[a] ^= v[b]; v[a + 1] ^= v[b + 1];
  };
  const sipRound = () => {
    add(0, 2); rotl(2, 13); xor(2, 0); rotl(0, 32);
    add(4, 6); rotl(6, 16); xor(6, 4);
    add(0, 6); rotl(6, 21); xor(6, 0);
    add(4, 2); rotl(2, 17); xor(2, 4); rotl(4, 32);
  };
  const len = k.length;
  const end = len - (len % 8);
  for (let o = 0; o < end; o += 8) {
    const ml = u32le(k, o);
    const mh = u32le(k, o + 4);
    v[6] ^= mh; v[7] ^= ml;
    sipRound(); sipRound();
    v[0] ^= mh; v[1] ^= ml;
  }
  let bl = 0;
  let bh = ((len & 0xff) << 24) >>> 0;
  const left = len & 7;
  for (let i = 0; i < left; i++) {
    const byte = k[end + i];
    if (i < 4) bl |= byte << (8 * i);
    else bh |= byte << (8 * (i - 4));
  }
  bl >>>= 0; bh >>>= 0;
  v[6] ^= bh; v[7] ^= bl;
  sipRound(); sipRound();
  v[0] ^= bh; v[1] ^= bl;
  v[5] ^= 0xff;
  sipRound(); sipRound(); sipRound(); sipRound();
  return [(v[0] ^ v[2] ^ v[4] ^ v[6]) >>> 0, (v[1] ^ v[3] ^ v[5] ^ v[7]) >>> 0];
}

export const sipHash24 = (k: Uint8Array, seed = 0) => sipHash24Full(k, seed)[1];

export const HASHES: Record<HashId, { label: string; fn: (k: Uint8Array, seed?: number) => number }> = {
  java31: { label: '×31 polynomial (Java hashCode)', fn: java31 },
  fnv1a: { label: 'FNV-1a 32', fn: fnv1a },
  pg: { label: 'PostgreSQL hash_bytes (lookup3)', fn: pgHashBytes },
  murmur3: { label: 'MurmurHash3 x86_32', fn: murmur3_32 },
  xxh32: { label: 'xxHash32', fn: xxh32 },
  sip24: { label: 'SipHash-2-4 (keyed)', fn: sipHash24 },
};

/* ------------------------------------------------------------------ key sets */

export const MAX_KEYS = 4096;
const enc = (s: string) => {
  const out = new Uint8Array(s.length);
  for (let i = 0; i < s.length; i++) out[i] = s.charCodeAt(i) & 0xff;
  return out;
};

/** Multiplicative inverse of an odd number modulo 2^32 (Newton iteration). */
export function invOdd32(a: number) {
  let x = a >>> 0;
  for (let i = 0; i < 5; i++) x = Math.imul(x, (2 - Math.imul(a, x)) >>> 0) >>> 0;
  return x >>> 0;
}
const MC1_INV = invOdd32(MC1);
const MC2_INV = invOdd32(MC2);
/** The 4-byte block whose Murmur3 mixed value is m. */
export const murmurUnmixK = (m: number) => Math.imul(rotr32(Math.imul(m >>> 0, MC2_INV) >>> 0, 15), MC1_INV) >>> 0;

const putU32 = (buf: Uint8Array, o: number, x: number) => {
  buf[o] = x & 0xff; buf[o + 1] = (x >>> 8) & 0xff; buf[o + 2] = (x >>> 16) & 0xff; buf[o + 3] = (x >>> 24) & 0xff;
};

export function makeKeys(set: KeySetId, n: number): Uint8Array[] {
  const count = Math.max(1, Math.min(MAX_KEYS, Math.floor(n)));
  const keys: Uint8Array[] = [];
  if (set === 'seq') {
    for (let i = 1; i <= count; i++) {
      const b = new Uint8Array(8);
      putU32(b, 0, i);
      keys.push(b);
    }
  } else if (set === 'orders') {
    for (let i = 1; i <= count; i++) keys.push(enc(`ORD-${String(i).padStart(8, '0')}`));
  } else if (set === 'hosts') {
    const roles = ['api', 'web', 'cache', 'db', 'queue', 'auth', 'search', 'batch'];
    const regions = ['us-east-1', 'us-west-2', 'eu-west-1', 'ap-south-1'];
    for (let i = 0; i < count; i++) keys.push(enc(`${roles[i % 8]}-${String(i).padStart(5, '0')}.${regions[(i * 7) % 4]}.internal`));
  } else if (set === 'uuid') {
    const rng = makeRng(2026);
    for (let i = 0; i < count; i++) {
      const b = new Uint8Array(16);
      for (let j = 0; j < 16; j++) b[j] = Math.floor(rng() * 256) & 0xff;
      b[6] = (b[6] & 0x0f) | 0x40;
      b[8] = (b[8] & 0x3f) | 0x80;
      keys.push(b);
    }
  } else if (set === 'collide31') {
    // "Aa" and "BB" have the same ×31 hash, so every string of 12 such blocks has the same hash too.
    for (let i = 0; i < count; i++) {
      let s = '';
      for (let j = 11; j >= 0; j--) s += (i >>> j) & 1 ? 'BB' : 'Aa';
      keys.push(enc(s));
    }
  } else {
    // Murmur3 differential multicollisions: 12 independent 8-byte choices, each with two variants that
    // leave the internal state identical for every seed.
    const rng = makeRng(29);
    const r32 = () => (Math.floor(rng() * 65536) * 65536 + Math.floor(rng() * 65536)) >>> 0;
    const pairs = Array.from({ length: 12 }, () => {
      const a = r32();
      const b = r32();
      const v0 = new Uint8Array(8);
      const v1 = new Uint8Array(8);
      putU32(v0, 0, murmurUnmixK(a));
      putU32(v0, 4, murmurUnmixK(b));
      putU32(v1, 0, murmurUnmixK((a ^ 0x00040000) >>> 0));
      putU32(v1, 4, murmurUnmixK((b ^ 0x80000000) >>> 0));
      return [v0, v1];
    });
    for (let i = 0; i < count; i++) {
      const buf = new Uint8Array(96);
      for (let j = 0; j < 12; j++) buf.set(pairs[j][(i >>> j) & 1], j * 8);
      keys.push(buf);
    }
  }
  return keys;
}

export const KEYSETS: Record<KeySetId, string> = {
  seq: 'Sequential integers (int8)',
  orders: 'Order ids (ORD-00000017)',
  hosts: 'Hostnames',
  uuid: 'Random UUIDs',
  collide31: 'Attack: ×31 equal-hash strings',
  collideMurmur: 'Attack: Murmur3 multicollisions',
};

export function hashAll(keys: Uint8Array[], hash: HashId, seed: number) {
  const fn = HASHES[hash].fn;
  const out = new Uint32Array(keys.length);
  for (let i = 0; i < keys.length; i++) out[i] = fn(keys[i], seed);
  return out;
}

export const distinctHashes = (h: Uint32Array) => new Set(h).size;

/* ---------------------------------------------------------------- avalanche */

export const AV_IN = 64;
export const AV_OUT = 32;

/** p[i*32 + j] = fraction of sample keys where flipping input bit i flips output bit j. */
export function avalanche(keys: Uint8Array[], hash: HashId, seed: number, samples = 256) {
  const fn = HASHES[hash].fn;
  const s = Math.min(samples, keys.length);
  const flips = new Uint32Array(AV_IN * AV_OUT);
  const bases = new Set<number>();
  for (let t = 0; t < s; t++) {
    const key = keys[t];
    const base = fn(key, seed);
    bases.add(base);
    const buf = key.slice();
    for (let i = 0; i < AV_IN; i++) {
      const byte = i >>> 3;
      if (byte >= buf.length) continue;
      const mask = 1 << (i & 7);
      buf[byte] ^= mask;
      let d = (fn(buf, seed) ^ base) >>> 0;
      buf[byte] ^= mask;
      const row = i * AV_OUT;
      for (let j = 0; d !== 0 && j < AV_OUT; j++) {
        if (d & 1) flips[row + j]++;
        d >>>= 1;
      }
    }
  }
  const p = new Float32Array(AV_IN * AV_OUT);
  let biasSum = 0;
  let worst = 0;
  let stuck = 0;
  for (let c = 0; c < p.length; c++) {
    p[c] = flips[c] / s;
    const bias = Math.abs(2 * p[c] - 1);
    biasSum += bias;
    if (bias > worst) worst = bias;
    if (flips[c] === 0 || flips[c] === s) stuck++;
  }
  return { p, samples: s, meanBias: biasSum / p.length, worstBias: worst, stuck, distinctBase: bases.size };
}

/** Mean |2p−1| that an ideal random function shows at this sample size (sampling noise alone). */
export const idealMeanBias = (samples: number) => Math.sqrt(2 / (Math.PI * samples));

export const biasBin = (bias: number) => (bias < 0.1 ? 0 : bias < 0.3 ? 1 : bias < 0.9 ? 2 : 3);
export const BIAS_BINS = ['< 0.1 (ideal)', '0.1–0.3 (sampling noise reaches here)', '0.3–0.9', '≥ 0.9 (bit barely mixed)'];
// Four steps, skipping --viz-seq-550: in the dark theme it is too close to --viz-seq-700 to tell apart.
const BIAS_FILL = ['var(--viz-seq-100)', 'var(--viz-seq-250)', 'var(--viz-seq-400)', 'var(--viz-seq-700)'];

/* -------------------------------------------------------- buckets and probes */

export const CHAIN_BINS = 11; // loads 0..9 and 10+
export const PROBE_BINS = 15; // probes 1..14 and 15+

export function poissonBuckets(m: number, alpha: number) {
  const out: number[] = [];
  let term = Math.exp(-alpha);
  let cum = 0;
  for (let k = 0; k < CHAIN_BINS - 1; k++) {
    out.push(m * term);
    cum += term;
    term = (term * alpha) / (k + 1);
  }
  out.push(m * Math.max(0, 1 - cum));
  return out;
}

export function chaining(hashes: Uint32Array, m: number) {
  const loads = new Uint32Array(m);
  for (const h of hashes) loads[h & (m - 1)]++;
  const hist = new Array(CHAIN_BINS).fill(0);
  let maxLoad = 0;
  let scanned = 0;
  for (const l of loads) {
    hist[Math.min(l, CHAIN_BINS - 1)]++;
    if (l > maxLoad) maxLoad = l;
    scanned += l * l;
  }
  return { loads, hist, maxLoad, empty: hist[0], avgScanned: hashes.length ? scanned / hashes.length : 0 };
}

/** Insert in order with linear probing; probes for a successful search = displacement + 1. */
export function linearProbing(slots: Uint32Array, m: number) {
  const table = new Int32Array(m).fill(-1);
  const hist = new Array(PROBE_BINS).fill(0);
  let total = 0;
  let max = 0;
  const n = Math.min(slots.length, m - 1);
  for (let i = 0; i < n; i++) {
    let pos = slots[i] & (m - 1);
    let probes = 1;
    while (table[pos] !== -1) {
      pos = (pos + 1) & (m - 1);
      probes++;
    }
    table[pos] = i;
    total += probes;
    if (probes > max) max = probes;
    hist[Math.min(probes, PROBE_BINS) - 1]++;
  }
  return { hist, mean: n ? total / n : 0, max };
}

export const knuthSuccessful = (alpha: number) => 0.5 * (1 + 1 / (1 - alpha));
export const knuthUnsuccessful = (alpha: number) => 0.5 * (1 + 1 / ((1 - alpha) * (1 - alpha)));

/** Uniformly random bucket choices, averaged over several runs: the "ideal random hash" reference. */
export function idealReference(m: number, n: number, runs = 8) {
  const chainHist = new Array(CHAIN_BINS).fill(0);
  const probeHist = new Array(PROBE_BINS).fill(0);
  let maxLoad = 0;
  let probeMean = 0;
  let probeMax = 0;
  for (let r = 0; r < runs; r++) {
    const rng = makeRng(7919 * (r + 1));
    const slots = new Uint32Array(n);
    for (let i = 0; i < n; i++) slots[i] = Math.floor(rng() * m);
    const c = chaining(slots, m);
    const p = linearProbing(slots, m);
    c.hist.forEach((v, i) => (chainHist[i] += v / runs));
    p.hist.forEach((v, i) => (probeHist[i] += v / runs));
    maxLoad += c.maxLoad / runs;
    probeMean += p.mean / runs;
    probeMax += p.max / runs;
  }
  return { chainHist, probeHist, maxLoad, probeMean, probeMax };
}

/** What the Buckets view's narration says; thresholds sit well outside single-run noise of a random hash. */
export function bucketVerdict(r: {
  n: number;
  m: number;
  distinct: number;
  chain: { maxLoad: number; avgScanned: number };
  chainIdeal: number;
  probe: { mean: number };
  ideal: { maxLoad: number; probeMean: number };
}) {
  if (r.distinct < r.n / 2) return 'collisions' as const;
  if (r.chain.maxLoad > 2 * r.ideal.maxLoad + 2) return 'chains' as const;
  // A single linear-probing run at high load is heavy-tailed: well-mixed hashes reach about 3× the averaged ideal.
  if (r.probe.mean > 4 * r.ideal.probeMean) return 'clusters' as const;
  // Σ load² / n for a random hash has mean 1 + (n−1)/m and a standard deviation of about √(2/m);
  // seven of those is far outside what any of the well-mixed hashes showed over 10,000 configurations.
  if (r.chain.avgScanned > r.chainIdeal + 7 * Math.sqrt(2 / r.m)) return 'uneven' as const;
  return 'random' as const;
}

export function bucketsAndProbes(keySet: KeySetId, hash: HashId, seed: number, logM: number, alpha: number) {
  const m = 1 << logM;
  const n = Math.max(1, Math.min(m - 1, Math.round(alpha * m)));
  const keys = makeKeys(keySet, n);
  const hashes = hashAll(keys, hash, seed);
  const chain = chaining(hashes, m);
  const probe = linearProbing(hashes, m);
  const ideal = idealReference(m, n);
  return {
    m,
    n,
    alpha: n / m,
    chain,
    probe,
    ideal,
    poisson: poissonBuckets(m, n / m),
    distinct: distinctHashes(hashes),
    knuth: knuthSuccessful(n / m),
    /** Keys compared by a successful chained lookup, Σ load² / n, for a random hash. */
    chainIdeal: 1 + (n - 1) / m,
  };
}

/* ---------------------------------------------------- partition, then join */

export type ReuseMode = 'same' | 'rotate' | 'seed2';
export const REUSE_ROWS = 4096;
/** The join table's second seed: a large odd increment, since nearby seeds need not be independent. */
export const secondSeed = (seed: number) => (seed + 0x9e3779b9) >>> 0;

export function partitionThenJoin(keySet: KeySetId, hash: HashId, seed: number, logP: number, logB: number, mode: ReuseMode) {
  const P = 1 << logP;
  const B = 1 << logB;
  const keys = makeKeys(keySet, REUSE_ROWS);
  const h = hashAll(keys, hash, seed);
  const h2 = mode === 'seed2' ? hashAll(keys, hash, secondSeed(seed)) : h;
  const part = new Uint32Array(keys.length);
  const partRows = new Array(P).fill(0);
  for (let i = 0; i < keys.length; i++) {
    // PostgreSQL: bucketno = hash & (nbuckets-1); batchno = ROR(hash, log2_nbuckets) & (nbatch-1)
    const p = mode === 'rotate' ? rotr32(h[i], logB) & (P - 1) : h[i] & (P - 1);
    part[i] = p;
    partRows[p]++;
  }
  let biggest = 0;
  for (let p = 1; p < P; p++) if (partRows[p] > partRows[biggest]) biggest = p;
  const loads = new Array(B).fill(0);
  for (let i = 0; i < keys.length; i++) if (part[i] === biggest) loads[h2[i] & (B - 1)]++;
  const rows = partRows[biggest];
  let used = 0;
  let longest = 0;
  let sq = 0;
  for (const l of loads) {
    if (l > 0) used++;
    if (l > longest) longest = l;
    sq += l * l;
  }
  const idealUsed = B * (1 - Math.pow(1 - 1 / B, rows));
  return {
    P,
    B,
    partRows,
    biggest,
    rows,
    loads,
    used,
    longest,
    avgScanned: rows ? sq / rows : 0,
    idealUsed,
    idealScanned: rows ? 1 + (rows - 1) / B : 0,
    minPart: Math.min(...partRows),
    distinct: distinctHashes(h),
  };
}

/* ------------------------------------------------------------------ figure */

type View = 'avalanche' | 'buckets' | 'reuse';

function Histogram({
  x0,
  y0,
  w,
  h,
  labels,
  measured,
  ideal,
  title,
  xLabel,
  overflowNote,
}: {
  x0: number;
  y0: number;
  w: number;
  h: number;
  labels: string[];
  measured: number[];
  ideal: number[];
  title: string;
  xLabel: string;
  overflowNote?: string;
}) {
  const top = y0 + 22;
  const plotH = h - 58;
  const left = x0 + 40;
  const plotW = w - 48;
  const max = Math.max(1, ...measured, ...ideal);
  const bw = plotW / labels.length;
  const Y = (v: number) => top + plotH - (v / max) * plotH;
  const ticks = [0, max / 2, max];
  return (
    <g>
      <text x={x0} y={y0 + 12} fontSize={12} fill="var(--viz-ink)">
        {title}
      </text>
      {ticks.map((t, i) => (
        <g key={i}>
          <line className="viz-grid-line" x1={left} x2={left + plotW} y1={Y(t)} y2={Y(t)} />
          <text x={left - 4} y={Y(t) + 4} textAnchor="end" fontSize={10}>
            {fmtNum(t)}
          </text>
        </g>
      ))}
      {measured.map((v, i) => (
        <rect key={i} x={left + i * bw + 2} y={Y(v)} width={Math.max(1, bw - 4)} height={top + plotH - Y(v)} rx={2} fill="var(--viz-1)" />
      ))}
      <polyline
        points={ideal.map((v, i) => `${left + i * bw + bw / 2},${Y(v)}`).join(' ')}
        fill="none"
        stroke="var(--viz-2)"
        strokeWidth={2}
      />
      {ideal.map((v, i) => (
        <circle key={i} cx={left + i * bw + bw / 2} cy={Y(v)} r={2.5} fill="var(--viz-2)" />
      ))}
      <line className="viz-axis-line" x1={left} x2={left + plotW} y1={top + plotH} y2={top + plotH} />
      {labels.map((l, i) => (
        <text key={i} x={left + i * bw + bw / 2} y={top + plotH + 13} textAnchor="middle" fontSize={labels.length > 12 ? 9 : 10}>
          {l}
        </text>
      ))}
      <text x={left + plotW / 2} y={top + plotH + 28} textAnchor="middle" fontSize={10} fill={overflowNote ? 'var(--viz-ink)' : undefined}>
        {overflowNote ? `${xLabel} · ${overflowNote}` : xLabel}
      </text>
    </g>
  );
}

export default function HashAvalancheSkewLab() {
  const [view, setView] = useState<View>('avalanche');
  const [keySet, setKeySet] = useState<KeySetId>('hosts');
  const [hash, setHash] = useState<HashId>('murmur3');
  const [seed, setSeed] = useState(0);
  const [logM, setLogM] = useState(10);
  const [alphaPct, setAlphaPct] = useState(75);
  const [logP, setLogP] = useState(6);
  const [logB, setLogB] = useState(6);
  const [mode, setMode] = useState<ReuseMode>('same');
  const [hover, setHover] = useState<[number, number] | null>(null);
  const [ref, width] = useSize(720);

  const av = useMemo(() => (view === 'avalanche' ? avalanche(makeKeys(keySet, 256), hash, seed) : null), [view, keySet, hash, seed]);
  const bp = useMemo(() => (view === 'buckets' ? bucketsAndProbes(keySet, hash, seed, logM, alphaPct / 100) : null), [view, keySet, hash, seed, logM, alphaPct]);
  const pj = useMemo(() => (view === 'reuse' ? partitionThenJoin(keySet, hash, seed, logP, logB, mode) : null), [view, keySet, hash, seed, logP, logB, mode]);
  const pjAll = useMemo(
    () => (view === 'reuse' ? (['same', 'rotate', 'seed2'] as const).map((md) => ({ md, r: partitionThenJoin(keySet, hash, seed, logP, logB, md) })) : []),
    [view, keySet, hash, seed, logP, logB],
  );

  const W = Math.max(300, Math.min(700, width));
  const hashLabel = HASHES[hash].label;
  const modeLabel: Record<ReuseMode, string> = { same: 'Same hash, same bits', rotate: 'Different bits (PostgreSQL)', seed2: 'Second seed' };

  /* ---- avalanche figure ---- */
  const cell = Math.max(4, Math.min(9, Math.floor((W - 60) / AV_IN)));
  const avLeft = 52;
  const avTop = 22;
  const avW = avLeft + AV_IN * cell + 10;
  const compact = cell < 7;
  const avH = avTop + AV_OUT * cell + (compact ? 54 : 40);

  /* ---- buckets figure ---- */
  const sideBySide = W >= 620;
  const histW = sideBySide ? W / 2 - 6 : W;
  const histH = 220;

  /* ---- reuse figure ---- */
  const rW = W;

  const title =
    view === 'avalanche' ? 'Avalanche: which input bits reach which output bits' : view === 'buckets' ? 'Bucket chains and linear-probe lengths against an ideal random hash' : 'Partition by hash, then build a join table in the biggest partition';

  return (
    <VizPanel
      title={title}
      subtitle={
        view === 'avalanche'
          ? 'Each cell flips one bit in the first 8 bytes of a key and asks how often one output bit flips, over 256 keys. A good hash flips every output bit half the time.'
          : view === 'buckets'
            ? 'Insert n = α·m keys into m buckets chosen by the low bits of the hash: once as chains, once as a linear-probing table. The curves are what a truly random hash would give.'
            : '4,096 rows are routed to P partitions by their hash; the biggest partition then builds a hash table of B buckets. Watch which bits each step reads.'
      }
      controls={
        <>
          <Segmented
            label="View"
            value={view}
            onChange={setView}
            options={[
              { value: 'avalanche', label: 'Avalanche' },
              { value: 'buckets', label: 'Buckets and probes' },
              { value: 'reuse', label: 'Partition, then join' },
            ]}
          />
          <Choice label="Keys" value={keySet} onChange={setKeySet} options={(Object.keys(KEYSETS) as KeySetId[]).map((k) => ({ value: k, label: KEYSETS[k] }))} />
          <Choice label="Hash function" value={hash} onChange={setHash} options={(Object.keys(HASHES) as HashId[]).map((k) => ({ value: k, label: HASHES[k].label }))} />
          <Slider label="Seed" min={0} max={99} value={seed} onChange={setSeed} />
        </>
      }
      legend={
        view === 'avalanche' ? (
          <Legend items={BIAS_BINS.map((b, i) => ({ label: `bias ${b}`, color: BIAS_FILL[i] }))} />
        ) : view === 'buckets' ? (
          <Legend
            items={[
              { label: `Measured: ${hashLabel}`, color: 'var(--viz-1)' },
              { label: 'Ideal random hash', color: 'var(--viz-2)', shape: 'line' },
            ]}
          />
        ) : (
          <Legend
            items={[
              { label: 'Partition: bits it reads, rows per partition', color: 'var(--viz-1)' },
              { label: 'Join table: bits it reads, rows per bucket', color: 'var(--viz-3)' },
              { label: 'Bit read by both', color: 'var(--viz-critical)' },
            ]}
          />
        )
      }
      stats={
        av ? (
          <Stats
            items={[
              { label: 'Mean bias |2p−1|', value: fmtNum(av.meanBias, 3), hint: `An ideal random function shows about ${fmtNum(idealMeanBias(av.samples), 3)} from sampling noise alone` },
              { label: 'Random-function noise', value: fmtNum(idealMeanBias(av.samples), 3) },
              { label: 'Worst cell bias', value: fmtNum(av.worstBias, 2) },
              { label: 'Cells never or always flipping', value: `${fmtNum(av.stuck)} of ${fmtNum(AV_IN * AV_OUT)}` },
            ]}
          />
        ) : bp ? (
          <Stats
            items={[
              { label: 'Keys n / buckets m', value: `${fmtNum(bp.n)} / ${fmtNum(bp.m)}` },
              { label: 'Distinct 32-bit hashes', value: fmtNum(bp.distinct) },
              { label: 'Longest chain (ideal)', value: `${fmtNum(bp.chain.maxLoad)} (${fmtNum(bp.ideal.maxLoad, 1)})` },
              { label: 'Mean probes, linear (Knuth)', value: `${fmtNum(bp.probe.mean, 2)} (${fmtNum(bp.knuth, 2)})`, hint: 'Knuth: ½(1 + 1/(1−α)) for a successful search' },
              { label: 'Worst probe (ideal)', value: `${fmtNum(bp.probe.max)} (${fmtNum(bp.ideal.probeMax, 0)})` },
              { label: 'Chained lookup scans (random)', value: `${fmtNum(bp.chain.avgScanned, 2)} (${fmtNum(bp.chainIdeal, 2)})`, hint: 'Keys compared by a successful lookup in the chained table: Σ load² / n. A random hash gives 1 + (n − 1)/m.' },
            ]}
          />
        ) : pj ? (
          <Stats
            items={[
              { label: 'Rows in biggest partition', value: `${fmtNum(pj.rows)} (even: ${fmtNum(REUSE_ROWS / pj.P)})` },
              { label: 'Buckets used', value: `${fmtNum(pj.used)} of ${fmtNum(pj.B)}`, hint: `A random hash would use about ${fmtNum(pj.idealUsed, 1)}` },
              { label: 'Random hash would use', value: fmtNum(pj.idealUsed, 1) },
              { label: 'Longest chain', value: fmtNum(pj.longest) },
              { label: 'Rows scanned per probe', value: `${fmtNum(pj.avgScanned, 1)} (ideal ${fmtNum(pj.idealScanned, 1)})`, hint: 'Average chain length seen by a probe for one of these rows: Σ load² / rows' },
            ]}
          />
        ) : null
      }
      note={
        <Note>
          {av ? (
            <>
              <strong>
                {hashLabel} on {KEYSETS[keySet].toLowerCase()}: mean bias {fmtNum(av.meanBias, 3)}, {fmtNum(av.stuck)} cells that never or always flip.
              </strong>{' '}
              {av.distinctBase < av.samples / 4
                ? `These ${av.samples} keys share only ${fmtNum(av.distinctBase)} hash value${av.distinctBase === 1 ? '' : 's'}, so they are not independent samples: a flipped bit changes whole groups of them in exactly the same way, and the cells cannot average out. The matrix is measuring the attack, not the function's mixing. Switch to the Buckets view to see what the collisions cost.`
                : av.meanBias < 1.6 * idealMeanBias(av.samples)
                ? 'That is indistinguishable from a random function at this sample size: every input bit reaches every output bit, including the low bits a power-of-two table uses.'
                : hash === 'java31'
                  ? 'A multiply-and-add polynomial only carries upward: an input bit can never change an output bit below it, so the low bits a masked table reads depend on only a few low input bits.'
                  : hash === 'fnv1a'
                    ? 'FNV-1a combines each byte with an XOR and a multiply, and neither carries downward: an input bit can never change an output bit below it, so the low bits a masked table reads depend on only the low bits of each byte.'
                    : 'Some input bits are not reaching some output bits; a table that reads those output bits will see correlated keys land together.'}
            </>
          ) : bp ? (
            <>
              <strong>
                {fmtNum(bp.n)} keys, {fmtNum(bp.m)} buckets (α = {fmtNum(bp.alpha, 2)}): longest chain {fmtNum(bp.chain.maxLoad)} against about {fmtNum(bp.ideal.maxLoad, 1)} for a random hash; linear probing averages {fmtNum(bp.probe.mean, 2)} probes against Knuth’s {fmtNum(bp.knuth, 2)}.
              </strong>{' '}
              {{
                collisions: `Only ${fmtNum(bp.distinct)} distinct hash value${bp.distinct === 1 ? '' : 's'} for ${fmtNum(bp.n)} keys: no table size or load factor helps, because colliding keys share every bit. This is the hash-flooding shape — every insert walks the whole pile.`,
                chains: 'Some buckets hold far more keys than a random hash would put in any bucket: the low bits are not spreading this key set.',
                clusters: `No bucket is overfull, but the hash values land in neighbouring buckets, so linear probing’s occupied runs merge into long clusters. Chaining does not care whether full buckets are adjacent: a chained lookup scans ${fmtNum(bp.chain.avgScanned, 2)} keys against ${fmtNum(bp.chainIdeal, 2)} for a random hash.`,
                uneven: `No bucket is overfull, but ${fmtNum(bp.chain.empty)} of ${fmtNum(bp.m)} buckets are empty against about ${fmtNum(bp.poisson[0], 0)} for a random hash: the low bits leave gaps, so the other buckets hold longer chains and a chained lookup scans ${fmtNum(bp.chain.avgScanned, 2)} keys instead of ${fmtNum(bp.chainIdeal, 2)}.`,
                random: 'Chains and probe lengths are within the range a random hash produces on a single run: for these keys, this function behaves as the theory assumes.',
              }[bucketVerdict(bp)]}
            </>
          ) : pj ? (
            <>
              <strong>
                {modeLabel[mode]}: the biggest partition’s {fmtNum(pj.rows)} rows use {fmtNum(pj.used)} of {fmtNum(pj.B)} buckets (a random hash would use about {fmtNum(pj.idealUsed, 1)}).
              </strong>{' '}
              {pj.rows >= Math.min(REUSE_ROWS, (REUSE_ROWS / pj.P) * 4)
                ? pj.distinct < REUSE_ROWS / 2
                  ? `The partitions themselves are skewed: the ${fmtNum(REUSE_ROWS)} rows have only ${fmtNum(pj.distinct)} distinct hash value${pj.distinct === 1 ? '' : 's'}, so one partition holds ${fmtNum(pj.rows)} rows and no choice of bits or seed can split it.`
                  : `The partitions themselves are skewed: the hash values are distinct, but the bits that pick the partition do not spread these keys, so one partition gets ${fmtNum(pj.rows / (REUSE_ROWS / pj.P), 1)}× its share before any join table is built.`
                : mode === 'same'
                  ? pj.B <= pj.P
                    ? `Every row in a partition already agrees on the low ${logP} bits, and the table reads only the low ${logB}, so the whole partition lands in one bucket.`
                    : `Every row in a partition agrees on the low ${logP} bits, so only 1 in ${pj.P} of the ${pj.B} buckets can ever be used.`
                  : mode === 'rotate'
                    ? pj.used < pj.idealUsed / 2
                      ? `The partition reads bits ${logB}–${logB + logP - 1} and the table reads bits 0–${logB - 1}, but disjoint bits are independent only for a well-mixed hash: under ${hashLabel} they still move together for these keys.`
                      : `The partition reads bits ${logB}–${logB + logP - 1} and the table reads bits 0–${logB - 1}: disjoint bits are independent for a well-mixed hash.`
                    : pj.used < pj.idealUsed / 2
                      ? `A second seed only helps if the function mixes the seed into every bit. ${
                          hash === 'java31'
                            ? 'The ×31 polynomial just adds a constant for each key length, so rows that agreed on the low bits still agree.'
                            : hash === 'fnv1a'
                              ? 'FNV-1a only XORs the seed into its starting value, and its multiplies carry upward only, so the low bits the table reads stay tied to the bits that picked the partition.'
                              : `${hashLabel} does not spread these rows under the second seed.`
                        }`
                      : 'The table hashes with a different seed, so its bucket bits are unrelated to the bits that chose the partition.'}
            </>
          ) : null}
        </Note>
      }
      table={
        av ? (
          <table className="viz-table">
            <thead>
              <tr>
                <th>Flipped input byte</th>
                <th>Mean flip probability</th>
                <th>Mean bias</th>
                <th>Worst output bit bias</th>
              </tr>
            </thead>
            <tbody>
              {Array.from({ length: 8 }, (_, byte) => {
                let sum = 0;
                let bias = 0;
                let worst = 0;
                for (let i = byte * 8; i < byte * 8 + 8; i++)
                  for (let j = 0; j < AV_OUT; j++) {
                    const p = av.p[i * AV_OUT + j];
                    sum += p;
                    bias += Math.abs(2 * p - 1);
                    worst = Math.max(worst, Math.abs(2 * p - 1));
                  }
                return (
                  <tr key={byte}>
                    <td>byte {byte}</td>
                    <td>{fmtNum(sum / (8 * AV_OUT), 3)}</td>
                    <td>{fmtNum(bias / (8 * AV_OUT), 3)}</td>
                    <td>{fmtNum(worst, 2)}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        ) : bp ? (
          <table className="viz-table">
            <thead>
              <tr>
                <th>Chain length</th>
                <th>Buckets (measured)</th>
                <th>Buckets (Poisson)</th>
                <th>Probes</th>
                <th>Keys (measured)</th>
                <th>Keys (ideal random)</th>
              </tr>
            </thead>
            <tbody>
              {Array.from({ length: PROBE_BINS }, (_, i) => (
                <tr key={i}>
                  <td>{i < CHAIN_BINS ? (i === CHAIN_BINS - 1 ? `${i}+` : i) : ''}</td>
                  <td>{i < CHAIN_BINS ? fmtNum(bp.chain.hist[i]) : ''}</td>
                  <td>{i < CHAIN_BINS ? fmtNum(bp.poisson[i], 1) : ''}</td>
                  <td>{i === PROBE_BINS - 1 ? `${i + 1}+` : i + 1}</td>
                  <td>{fmtNum(bp.probe.hist[i])}</td>
                  <td>{fmtNum(bp.ideal.probeHist[i], 1)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        ) : (
          <table className="viz-table">
            <thead>
              <tr>
                <th>Join table hashing</th>
                <th>Rows in biggest partition</th>
                <th>Buckets used</th>
                <th>Longest chain</th>
                <th>Rows scanned per probe</th>
              </tr>
            </thead>
            <tbody>
              {pjAll.map(({ md, r }) => (
                <tr key={md}>
                  <td>{modeLabel[md]}</td>
                  <td>{fmtNum(r.rows)}</td>
                  <td>
                    {fmtNum(r.used)} of {fmtNum(r.B)}
                  </td>
                  <td>{fmtNum(r.longest)}</td>
                  <td>{fmtNum(r.avgScanned, 1)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        )
      }
    >
      <div ref={ref}>
        {view === 'avalanche' && av ? (
          <>
            <svg
              viewBox={`0 0 ${avW} ${avH}`}
              width={avW}
              height={avH}
              role="img"
              aria-label={`Avalanche matrix for ${hashLabel}: mean bias ${fmtNum(av.meanBias, 3)}`}
              onMouseLeave={() => setHover(null)}
              onMouseMove={(e) => {
                const box = (e.currentTarget as SVGSVGElement).getBoundingClientRect();
                const sx = ((e.clientX - box.left) / box.width) * avW;
                const sy = ((e.clientY - box.top) / box.height) * avH;
                const i = Math.floor((sx - avLeft) / cell);
                const j = Math.floor((sy - avTop) / cell);
                setHover(i >= 0 && i < AV_IN && j >= 0 && j < AV_OUT ? [i, j] : null);
              }}
            >
              <text x={avLeft} y={12} fontSize={10}>
                {compact ? 'input bit flipped →' : 'input bit flipped → (bytes 0–7 of the key, low bit first)'}
              </text>
              {Array.from({ length: AV_IN * AV_OUT }, (_, c) => {
                const i = Math.floor(c / AV_OUT);
                const j = c % AV_OUT;
                return <rect key={c} x={avLeft + i * cell} y={avTop + j * cell} width={cell} height={cell} fill={BIAS_FILL[biasBin(Math.abs(2 * av.p[c] - 1))]} />;
              })}
              {Array.from({ length: 9 }, (_, b) => (
                <line key={b} x1={avLeft + b * 8 * cell} x2={avLeft + b * 8 * cell} y1={avTop} y2={avTop + AV_OUT * cell} stroke="var(--viz-surface)" strokeWidth={1.5} />
              ))}
              {Array.from({ length: 8 }, (_, b) => (
                <text key={b} x={avLeft + b * 8 * cell + 4 * cell} y={avTop + AV_OUT * cell + 13} textAnchor="middle" fontSize={10}>
                  {compact ? b : `byte ${b}`}
                </text>
              ))}
              <text x={avLeft - 6} y={avTop + cell} textAnchor="end" fontSize={10}>
                bit 0
              </text>
              <text x={avLeft - 6} y={avTop + AV_OUT * cell} textAnchor="end" fontSize={10}>
                bit 31
              </text>
              <line x1={avLeft - 3} x2={avLeft - 3} y1={avTop} y2={avTop + 10 * cell} stroke="var(--viz-ink)" strokeWidth={2} />
              {compact ? (
                <text x={avLeft - 6} y={avTop + 8 * cell + 4} textAnchor="end" fontSize={9} fill="var(--viz-ink)">
                  mask
                </text>
              ) : (
                <>
                  <text x={avLeft - 6} y={avTop + 6.5 * cell} textAnchor="end" fontSize={9} fill="var(--viz-ink)">
                    1,024-
                  </text>
                  <text x={avLeft - 6} y={avTop + 8 * cell + 2} textAnchor="end" fontSize={9} fill="var(--viz-ink)">
                    bucket
                  </text>
                  <text x={avLeft - 6} y={avTop + 9.5 * cell + 2} textAnchor="end" fontSize={9} fill="var(--viz-ink)">
                    mask
                  </text>
                </>
              )}
              {hover ? (
                <rect x={avLeft + hover[0] * cell} y={avTop + hover[1] * cell} width={cell} height={cell} fill="none" stroke="var(--viz-ink)" strokeWidth={1.5} />
              ) : null}
              {compact ? (
                <>
                  <text x={avLeft} y={avTop + AV_OUT * cell + 27} fontSize={10}>
                    {hover ? `in bit ${hover[0]} (byte ${hover[0] >> 3}) → out bit ${hover[1]}:` : 'input byte (low bit first); rows are'}
                  </text>
                  <text x={avLeft} y={avTop + AV_OUT * cell + 41} fontSize={10} fill="var(--viz-ink)">
                    {hover ? `flips in ${fmtNum(av.p[hover[0] * AV_OUT + hover[1]] * 100, 1)}% of ${av.samples} keys` : 'output bits 0 (top) to 31'}
                  </text>
                </>
              ) : (
                <text x={avLeft} y={avTop + AV_OUT * cell + 30} fontSize={10} fill="var(--viz-ink)">
                  {hover
                    ? `Flip input bit ${hover[0]} (byte ${hover[0] >> 3}, bit ${hover[0] & 7}) → output bit ${hover[1]} flips in ${fmtNum(av.p[hover[0] * AV_OUT + hover[1]] * 100, 1)}% of ${av.samples} keys`
                    : 'Rows are output bits 0 (top) to 31. Hover a cell for its flip rate.'}
                </text>
              )}
            </svg>
          </>
        ) : null}

        {view === 'buckets' && bp ? (
          <>
            <div className="viz-controls">
              <Slider label="Buckets m" min={6} max={12} value={logM} onChange={setLogM} format={(v) => fmtNum(1 << v)} />
              <Slider label="Load factor α" min={25} max={95} step={5} value={alphaPct} onChange={setAlphaPct} format={(v) => (v / 100).toFixed(2)} />
            </div>
            <svg
              viewBox={`0 0 ${W} ${sideBySide ? histH : histH * 2 + 10}`}
              width={W}
              height={sideBySide ? histH : histH * 2 + 10}
              role="img"
              aria-label={`Chaining: longest chain ${bp.chain.maxLoad}. Linear probing: mean ${fmtNum(bp.probe.mean, 2)} probes.`}
            >
              <Histogram
                x0={0}
                y0={0}
                w={histW}
                h={histH}
                title="Chaining: buckets by chain length"
                xLabel="keys in the bucket"
                labels={Array.from({ length: CHAIN_BINS }, (_, i) => (i === CHAIN_BINS - 1 ? `${i}+` : String(i)))}
                measured={bp.chain.hist}
                ideal={bp.poisson}
                overflowNote={bp.chain.maxLoad >= CHAIN_BINS - 1 ? `fullest bucket: ${fmtNum(bp.chain.maxLoad)} keys` : undefined}
              />
              <Histogram
                x0={sideBySide ? histW + 12 : 0}
                y0={sideBySide ? 0 : histH + 10}
                w={histW}
                h={histH}
                title="Linear probing: keys by probes to find them"
                xLabel="probes for a successful search"
                labels={Array.from({ length: PROBE_BINS }, (_, i) => (i === PROBE_BINS - 1 ? `${i + 1}+` : String(i + 1)))}
                measured={bp.probe.hist}
                ideal={bp.ideal.probeHist}
                overflowNote={bp.probe.max >= PROBE_BINS ? `worst: ${fmtNum(bp.probe.max)} probes` : undefined}
              />
            </svg>
          </>
        ) : null}

        {view === 'reuse' && pj ? (
          <>
            <div className="viz-controls">
              <Slider label="Partitions P" min={2} max={8} value={logP} onChange={setLogP} format={(v) => fmtNum(1 << v)} />
              <Slider label="Buckets per join table B" min={2} max={10} value={logB} onChange={setLogB} format={(v) => fmtNum(1 << v)} />
              <Segmented
                label="Join table hashing"
                value={mode}
                onChange={setMode}
                options={[
                  { value: 'same', label: 'Same hash, same bits' },
                  { value: 'rotate', label: 'Different bits (PG)' },
                  { value: 'seed2', label: 'Second seed' },
                ]}
              />
            </div>
            {(() => {
              const bitsTop = 8;
              const strips = mode === 'seed2' ? 2 : 1;
              const bitW = (rW - 120) / 32;
              const stripY = (s: number) => bitsTop + 16 + s * 34;
              const partBits = new Set<number>();
              const bucketBits = new Set<number>();
              for (let b = 0; b < logP; b++) partBits.add(mode === 'rotate' ? (b + logB) % 32 : b);
              for (let b = 0; b < logB; b++) bucketBits.add(b);
              const bitX = (bit: number) => 110 + (31 - bit) * bitW;
              const chartTop = stripY(strips) + 16;
              const partH = 90;
              const bucketTop = chartTop + partH + 46;
              const bucketH = 120;
              const H = bucketTop + bucketH + 44;
              const maxPart = Math.max(...pj.partRows);
              const maxLoad = Math.max(1, ...pj.loads);
              const plotL = 44;
              const plotW = rW - plotL - 8;
              const pbw = plotW / pj.P;
              const bbw = plotW / pj.B;
              const strip = (s: number, label: string, kind: 'part' | 'bucket' | 'both') => (
                <g key={s}>
                  <text x={0} y={stripY(s) + 13} fontSize={10} fill="var(--viz-ink)">
                    {label}
                  </text>
                  {Array.from({ length: 32 }, (_, i) => {
                    const bit = 31 - i;
                    const inP = kind !== 'bucket' && partBits.has(bit);
                    const inB = kind !== 'part' && bucketBits.has(bit);
                    const clash = inP && inB;
                    return (
                      <rect
                        key={bit}
                        x={bitX(bit) + 0.5}
                        y={stripY(s)}
                        width={Math.max(1, bitW - 1)}
                        height={18}
                        rx={1.5}
                        fill={clash ? 'var(--viz-critical)' : 'var(--viz-surface)'}
                        stroke={inP ? 'var(--viz-1)' : inB ? 'var(--viz-3)' : 'var(--viz-border)'}
                        strokeWidth={inP || inB ? 2 : 1}
                      />
                    );
                  })}
                </g>
              );
              return (
                <svg viewBox={`0 0 ${rW} ${H}`} width={rW} height={H} role="img" aria-label={`${modeLabel[mode]}: ${pj.used} of ${pj.B} buckets used in the biggest partition`}>
                  <text x={bitX(31)} y={bitsTop + 4} fontSize={10}>
                    bit 31
                  </text>
                  <text x={bitX(0) + bitW} y={bitsTop + 4} textAnchor="end" fontSize={10}>
                    bit 0
                  </text>
                  {mode === 'seed2' ? (
                    <>
                      {strip(0, `hash, seed ${seed}`, 'part')}
                      {strip(1, 'hash, second seed', 'bucket')}
                    </>
                  ) : (
                    strip(0, 'one hash value', 'both')
                  )}
                  <text x={0} y={chartTop + 10} fontSize={12} fill="var(--viz-ink)">
                    Rows per partition ({fmtNum(pj.P)} partitions)
                  </text>
                  {pj.partRows.map((v, p) => {
                    const hh = (v / maxPart) * (partH - 24);
                    return (
                      <rect
                        key={p}
                        x={plotL + p * pbw + (pbw > 3 ? 0.5 : 0)}
                        y={chartTop + partH - hh}
                        width={Math.max(0.6, pbw - (pbw > 3 ? 1 : 0))}
                        height={hh}
                        fill="var(--viz-1)"
                        stroke={p === pj.biggest ? 'var(--viz-ink)' : 'none'}
                        strokeWidth={p === pj.biggest ? 1.5 : 0}
                      />
                    );
                  })}
                  <text x={plotL - 4} y={chartTop + partH - (partH - 24) + 4} textAnchor="end" fontSize={10}>
                    {fmtNum(maxPart)}
                  </text>
                  <line className="viz-axis-line" x1={plotL} x2={plotL + plotW} y1={chartTop + partH} y2={chartTop + partH} />
                  <text x={plotL + pj.biggest * pbw + pbw / 2} y={chartTop + partH + 13} textAnchor="middle" fontSize={10} fill="var(--viz-ink)">
                    ▲ biggest
                  </text>
                  <text x={0} y={bucketTop - 8} fontSize={12} fill="var(--viz-ink)">
                    {rW < 520 ? `Its join table (${fmtNum(pj.B)} buckets)` : `The biggest partition’s join table: rows per bucket (${fmtNum(pj.B)} buckets)`}
                  </text>
                  {pj.loads.map((v, b) => {
                    const hh = (v / maxLoad) * (bucketH - 10);
                    return v > 0 ? (
                      <rect key={b} x={plotL + b * bbw + (bbw > 3 ? 0.5 : 0)} y={bucketTop + bucketH - hh} width={Math.max(0.8, bbw - (bbw > 3 ? 1 : 0))} height={hh} fill="var(--viz-3)" />
                    ) : null;
                  })}
                  <line
                    x1={plotL}
                    x2={plotL + plotW}
                    y1={bucketTop + bucketH - (pj.rows / pj.B / maxLoad) * (bucketH - 10)}
                    y2={bucketTop + bucketH - (pj.rows / pj.B / maxLoad) * (bucketH - 10)}
                    stroke="var(--viz-ink-muted)"
                    strokeDasharray="4 3"
                  />
                  <text x={plotL + plotW / 2} y={bucketTop + bucketH + 30} textAnchor="middle" fontSize={10}>
                    dashed line: even spread, {fmtNum(pj.rows / pj.B, 1)} {fmtNum(pj.rows / pj.B, 1) === '1' ? 'row' : 'rows'} per bucket
                  </text>
                  <text x={plotL - 4} y={bucketTop + 14} textAnchor="end" fontSize={10}>
                    {fmtNum(maxLoad)}
                  </text>
                  <line className="viz-axis-line" x1={plotL} x2={plotL + plotW} y1={bucketTop + bucketH} y2={bucketTop + bucketH} />
                  <text x={plotL} y={bucketTop + bucketH + 14} fontSize={10}>
                    bucket 0
                  </text>
                  <text x={plotL + plotW} y={bucketTop + bucketH + 14} textAnchor="end" fontSize={10}>
                    bucket {fmtNum(pj.B - 1)}
                  </text>
                </svg>
              );
            })()}
          </>
        ) : null}
      </div>
    </VizPanel>
  );
}
