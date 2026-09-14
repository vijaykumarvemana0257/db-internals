import { useMemo, useState } from 'react';
import {
  VizPanel,
  Slider,
  Segmented,
  Legend,
  Stats,
  Note,
  TooltipHost,
  useTip,
  fmtNum,
  makeRng,
  useSize,
} from './Viz';

/**
 * Approximate membership filters, side by side, running the real algorithms.
 *
 * Two scales run at once:
 *  - the diagrams run a toy instance (8 buckets, 32 slots, 9 keys) so the mechanism is
 *    visible: a cuckoo eviction chain, a quotient filter's runs shifting, an xor filter
 *    being peeled and assigned, a ribbon band being eliminated and back-substituted;
 *  - the chart runs the *same code* over NBIG keys and probes it with NPROBE negatives,
 *    so the false-positive rate, the bits actually spent and the cache lines touched are
 *    measured, not asserted. The log2(1/eps) floor is computed from the measured eps.
 *
 * Everything is seeded (makeRng / a fixed integer hash), so SSR and hydration agree.
 */

/* ------------------------------------------------------------------ basics */

const LINE_BITS = 512; // one 64-byte cache line
const NBIG = 7700; // chosen so 4-slot cuckoo buckets land on a power of two at ~94% load
const NPROBE = 200_000;
const PROBE_BASE = 2_000_000;

function mix32(x: number, seed: number) {
  let h = (x ^ Math.imul(seed, 0x9e3779b1)) >>> 0;
  h = Math.imul(h ^ (h >>> 16), 2246822507) >>> 0;
  h = Math.imul(h ^ (h >>> 13), 3266489909) >>> 0;
  return (h ^ (h >>> 16)) >>> 0;
}

function ctz32(x: number) {
  let n = 0;
  while (((x >>> n) & 1) === 0 && n < 32) n++;
  return n;
}

function lg(x: number) {
  return Math.log(x) / Math.LN2;
}

function popcount(x: number) {
  x = x - ((x >> 1) & 0x55555555);
  x = (x & 0x33333333) + ((x >> 2) & 0x33333333);
  x = (x + (x >> 4)) & 0x0f0f0f0f;
  return (Math.imul(x, 0x01010101) >> 24) & 0xff;
}

const hex = (v: number, w = 2) => v.toString(16).toUpperCase().padStart(w, '0');

/** Parquet / Impala split-block Bloom: eight odd multipliers, one bit set per 32-bit word. */
const SALT = [
  0x47b6137b, 0x44974d91, 0x8824ad5b, 0xa2b7289d, 0x705495c7, 0x2df1424b, 0x9efc4947, 0x5c6bfb31,
];

type DesignId = 'bloom' | 'blocked' | 'cuckoo' | 'quotient' | 'xor' | 'ribbon';

const DESIGNS: { id: DesignId; label: string; color: string }[] = [
  { id: 'bloom', label: 'Bloom (baseline)', color: 'var(--viz-1)' },
  { id: 'blocked', label: 'Split-block Bloom', color: 'var(--viz-2)' },
  { id: 'cuckoo', label: 'Cuckoo', color: 'var(--viz-3)' },
  { id: 'quotient', label: 'Quotient', color: 'var(--viz-4)' },
  { id: 'xor', label: 'Xor', color: 'var(--viz-5)' },
  { id: 'ribbon', label: 'Ribbon', color: 'var(--viz-6)' },
];
const COLOR = Object.fromEntries(DESIGNS.map((d) => [d.id, d.color])) as Record<DesignId, string>;

/* ---------------------------------------------- operational answers per design */

type Verdict = 'yes' | 'no' | 'partial';
type OpId = 'none' | 'delete' | 'merge' | 'incremental';

const OPS: Record<DesignId, Record<Exclude<OpId, 'none'>, { v: Verdict; why: string }>> = {
  bloom: {
    delete: { v: 'no', why: 'Clearing a bit would unset it for every other key that hashed there. A counting Bloom filter replaces each bit with a 4-bit counter and costs about 4x the space.' },
    merge: { v: 'yes', why: 'Two filters with identical m and k union by bitwise OR — but the union keeps keys the compaction just dropped, so the merged filter is less selective than a rebuilt one.' },
    incremental: { v: 'yes', why: 'Bits can be set as keys arrive; RocksDB still buffers hashes and sizes the array at Finish() so it knows the real key count.' },
  },
  blocked: {
    delete: { v: 'no', why: 'Same reason as classic Bloom: bits are shared, and here they are shared inside one 256-bit block, so clearing is even more destructive.' },
    merge: { v: 'yes', why: 'OR works if both filters have the same block count and the same salt constants — Parquet writers that agree on the spec can union row-group filters.' },
    incremental: { v: 'yes', why: 'One block-local read-modify-write per key; the cheapest build of the six.' },
  },
  cuckoo: {
    delete: { v: 'yes', why: 'Remove one matching fingerprint from bucket i1 or i2 — but only for a key you know was inserted. Deleting a key that was never added can silently remove a different key that shares the fingerprint.' },
    merge: { v: 'partial', why: 'Fingerprints can be re-inserted into a table of the same size (the pair {i, i XOR h(f)} survives), but not into a different size: the bucket index carries information the fingerprint does not.' },
    incremental: { v: 'partial', why: 'Insertion is one hash and one store until the table nears its load factor, then eviction chains lengthen and can hit MaxNumKicks (500 in the reference implementation) and fail outright. Size for the final key count or be ready to rebuild.' },
  },
  quotient: {
    delete: { v: 'yes', why: 'Remove the remainder from its run and shift the cluster back, fixing is_continuation / is_shifted as you go. Same caveat as cuckoo: only for keys you actually inserted.' },
    merge: { v: 'yes', why: 'A quotient filter is a sorted list of fingerprints in disguise — the quotient is the slot index, the remainder is the payload — so two filters merge with a linear scan, and resizing is a rebuild from the recovered fingerprints rather than from the keys.' },
    incremental: { v: 'yes', why: 'Insert-as-you-go, at the cost of shifting whole clusters; above about 0.95 load the shifts get long.' },
  },
  xor: {
    delete: { v: 'no', why: 'Every slot is an XOR term shared by other keys. Removing one key would corrupt all of them.' },
    merge: { v: 'no', why: 'The slot assignment is a solution to one specific hypergraph. Two solutions cannot be combined; you rebuild from the union of the key sets.' },
    incremental: { v: 'no', why: 'Peeling needs the whole key set, random access to it, and two passes plus O(n) scratch. A key that arrives after the solve cannot be added at all.' },
  },
  ribbon: {
    delete: { v: 'no', why: 'The stored bits are the solution to a linear system over GF(2); no single key owns a slot.' },
    merge: { v: 'no', why: 'Same as xor — the solution is specific to one system of equations.' },
    incremental: { v: 'partial', why: 'Insertion into the band is a single streaming pass with bounded work per key, but the number of slots m must be fixed before the first row and a back-substitution pass runs at the end. RocksDB buffers a 64-bit hash per key during compaction and solves at Finish().' },
  },
};

const VERDICT_TEXT: Record<Verdict, string> = { yes: 'allowed', no: 'refused', partial: 'conditional' };
const VERDICT_COLOR: Record<Verdict, string> = {
  yes: 'var(--viz-good)',
  no: 'var(--viz-critical)',
  partial: 'var(--viz-warning)',
};

/* ------------------------------------------------------- measured at scale */

type Measured = {
  id: DesignId;
  bitsPerKey: number;
  fpr: number;
  fpCount: number;
  censored: boolean;
  floor: number;
  over: number;
  lines: number;
  work: number;
  ok: boolean;
  detail: string;
};

function summarize(
  id: DesignId,
  bitsPerKey: number,
  fpCount: number,
  lineTotal: number,
  work: number,
  ok: boolean,
  detail: string,
): Measured {
  const censored = fpCount === 0;
  const fpr = censored ? 0.5 / NPROBE : fpCount / NPROBE;
  const floor = lg(1 / fpr);
  return {
    id,
    bitsPerKey,
    fpr,
    fpCount,
    censored,
    floor,
    over: bitsPerKey / floor,
    lines: lineTotal / NPROBE,
    work: work / NBIG,
    ok,
    detail,
  };
}

function measureBloom(bits: number): Measured {
  const m = Math.ceil(NBIG * bits);
  const k = Math.max(1, Math.min(20, Math.round(bits * Math.LN2)));
  const arr = new Uint8Array((m + 7) >> 3);
  let work = 0;
  for (let i = 1; i <= NBIG; i++) {
    let h = mix32(i, 1);
    const d = (mix32(i, 2) | 1) >>> 0;
    for (let j = 0; j < k; j++) {
      const b = h % m;
      arr[b >> 3] |= 1 << (b & 7);
      h = (h + d) >>> 0;
      work++;
    }
  }
  let fpCount = 0;
  let lineTotal = 0;
  const seen = new Int32Array(24);
  for (let p = 0; p < NPROBE; p++) {
    const id = PROBE_BASE + p;
    let h = mix32(id, 1);
    const d = (mix32(id, 2) | 1) >>> 0;
    let hit = true;
    let ns = 0;
    for (let j = 0; j < k; j++) {
      const b = h % m;
      const ln = (b / LINE_BITS) | 0;
      let dup = false;
      for (let t = 0; t < ns; t++) if (seen[t] === ln) dup = true;
      if (!dup) seen[ns++] = ln;
      if (((arr[b >> 3] >> (b & 7)) & 1) === 0) {
        hit = false;
        break;
      }
      h = (h + d) >>> 0;
    }
    if (hit) fpCount++;
    lineTotal += ns;
  }
  return summarize('bloom', m / NBIG, fpCount, lineTotal, work, true, `k = ${k} probes over ${fmtNum(m / 8)} bytes`);
}

function measureBlocked(bits: number): Measured {
  const nB = Math.max(1, Math.ceil((NBIG * bits) / 256));
  const words = new Uint32Array(nB * 8);
  let work = 0;
  for (let i = 1; i <= NBIG; i++) {
    const b = (mix32(i, 3) % nB) * 8;
    const h = mix32(i, 4);
    for (let j = 0; j < 8; j++) {
      words[b + j] |= 1 << (Math.imul(h, SALT[j]) >>> 27);
      work++;
    }
  }
  let fpCount = 0;
  let lineTotal = 0;
  for (let p = 0; p < NPROBE; p++) {
    const id = PROBE_BASE + p;
    const bi = mix32(id, 3) % nB;
    const b = bi * 8;
    const h = mix32(id, 4);
    let hit = true;
    for (let j = 0; j < 8; j++) {
      if ((words[b + j] & (1 << (Math.imul(h, SALT[j]) >>> 27))) === 0) {
        hit = false;
        break;
      }
    }
    if (hit) fpCount++;
    lineTotal += 1; // a 32-byte block never straddles a 64-byte line
  }
  return summarize('blocked', (nB * 256) / NBIG, fpCount, lineTotal, work, true, `${fmtNum(nB)} blocks of 256 bits, 8 bits set per key`);
}

function measureCuckoo(bits: number): Measured {
  const B = 2048; // nextPow2(ceil(7700 / (4 * 0.95)))
  const S = 4;
  const f = Math.max(4, Math.min(16, Math.floor((bits * NBIG) / (B * S))));
  const mask = f >= 32 ? 0xffffffff : (1 << f) - 1;
  const table = new Int32Array(B * S);
  const rng = makeRng(0xc0ffee);
  let work = 0;
  let ok = true;
  let kicks = 0;
  for (let i = 1; i <= NBIG && ok; i++) {
    let fp = mix32(i, 5) & mask;
    if (fp === 0) fp = 1;
    let i1 = mix32(i, 6) & (B - 1);
    let i2 = (i1 ^ (mix32(fp, 7) & (B - 1))) & (B - 1);
    let placed = false;
    for (const bkt of [i1, i2]) {
      for (let s = 0; s < S; s++) {
        work++;
        if (table[bkt * S + s] === 0) {
          table[bkt * S + s] = fp;
          placed = true;
          break;
        }
      }
      if (placed) break;
    }
    if (placed) continue;
    let cur = rng() < 0.5 ? i1 : i2;
    let curFp = fp;
    for (let n = 0; n < 500 && !placed; n++) {
      const slot = Math.floor(rng() * S) % S;
      const victim = table[cur * S + slot];
      table[cur * S + slot] = curFp;
      curFp = victim;
      kicks++;
      work++;
      cur = (cur ^ (mix32(curFp, 7) & (B - 1))) & (B - 1);
      for (let s = 0; s < S; s++) {
        work++;
        if (table[cur * S + s] === 0) {
          table[cur * S + s] = curFp;
          placed = true;
          break;
        }
      }
    }
    if (!placed) ok = false;
    void i1;
    void i2;
  }
  let fpCount = 0;
  let lineTotal = 0;
  const bucketBits = S * f;
  for (let p = 0; p < NPROBE; p++) {
    const id = PROBE_BASE + p;
    let fp = mix32(id, 5) & mask;
    if (fp === 0) fp = 1;
    const i1 = mix32(id, 6) & (B - 1);
    const i2 = (i1 ^ (mix32(fp, 7) & (B - 1))) & (B - 1);
    let hit = false;
    for (let s = 0; s < S && !hit; s++) if (table[i1 * S + s] === fp) hit = true;
    if (!hit) for (let s = 0; s < S && !hit; s++) if (table[i2 * S + s] === fp) hit = true;
    if (hit) fpCount++;
    const l1 = ((i1 * bucketBits) / LINE_BITS) | 0;
    const l2 = ((i2 * bucketBits) / LINE_BITS) | 0;
    lineTotal += l1 === l2 ? 1 : 2;
  }
  return summarize(
    'cuckoo',
    (B * S * f) / NBIG,
    fpCount,
    lineTotal,
    work,
    ok,
    `${f}-bit fingerprints, ${fmtNum(B)} buckets x 4, ${fmtNum(kicks)} evictions during build`,
  );
}

function measureQuotient(bits: number): Measured {
  const q = 13; // 8192 slots for 7700 keys = 0.94 load
  const slots = 1 << q;
  const r = Math.max(2, Math.min(18, Math.round((bits * NBIG) / slots) - 3));
  const p = q + r;
  const shift = 32 - p;
  const fps = new Set<number>();
  const counts = new Int32Array(slots);
  let work = 0;
  for (let i = 1; i <= NBIG; i++) {
    const fp = mix32(i, 8) >>> shift;
    fps.add(fp);
    counts[fp >>> r]++;
    work++;
  }
  const start = new Int32Array(slots);
  const end = new Int32Array(slots);
  let cur = 0;
  for (let i = 0; i < slots; i++) {
    if (cur < i) cur = i;
    start[i] = cur;
    cur += counts[i];
    end[i] = cur;
    work += counts[i]; // shifting cost paid during build
  }
  const slotBits = r + 3;
  let fpCount = 0;
  let lineTotal = 0;
  for (let t = 0; t < NPROBE; t++) {
    const id = PROBE_BASE + t;
    const fp = mix32(id, 8) >>> shift;
    const c = fp >>> r;
    if (fps.has(fp)) fpCount++;
    if (counts[c] === 0) {
      lineTotal += 1; // is_occupied is clear: one slot read and done
    } else {
      const lo = Math.min(c, start[c]);
      const hi = Math.max(c, end[c] - 1);
      const a = ((lo * slotBits) / LINE_BITS) | 0;
      const b = (((hi + 1) * slotBits - 1) / LINE_BITS) | 0;
      lineTotal += b - a + 1;
    }
  }
  return summarize(
    'quotient',
    (slots * (r + 3)) / NBIG,
    fpCount,
    lineTotal,
    work,
    true,
    `${q}-bit quotient + ${r}-bit remainder + 3 metadata bits, ${fmtNum(slots)} slots`,
  );
}

function measureXor(bits: number): Measured {
  const f = Math.max(2, Math.min(24, Math.round(bits / 1.23)));
  const seg = Math.ceil((1.23 * NBIG) / 3);
  const cap = seg * 3;
  const mask = f >= 32 ? 0xffffffff : (1 << f) - 1;
  let work = 0;
  let seed = 0;
  let stackKey = new Int32Array(NBIG);
  let stackSlot = new Int32Array(NBIG);
  let ok = false;
  let attempts = 0;
  const slotsOf = (id: number, sd: number, out: Int32Array) => {
    out[0] = mix32(id, sd * 3 + 11) % seg;
    out[1] = seg + (mix32(id, sd * 3 + 12) % seg);
    out[2] = 2 * seg + (mix32(id, sd * 3 + 13) % seg);
  };
  const tri = new Int32Array(3);
  while (!ok && attempts < 32) {
    attempts++;
    const cnt = new Int32Array(cap);
    const xr = new Int32Array(cap);
    for (let i = 1; i <= NBIG; i++) {
      slotsOf(i, seed, tri);
      for (let j = 0; j < 3; j++) {
        cnt[tri[j]]++;
        xr[tri[j]] ^= i;
        work++;
      }
    }
    const queue: number[] = [];
    for (let s = 0; s < cap; s++) if (cnt[s] === 1) queue.push(s);
    let top = 0;
    while (queue.length > 0) {
      const s = queue.pop() as number;
      if (cnt[s] !== 1) continue;
      const key = xr[s];
      stackKey[top] = key;
      stackSlot[top] = s;
      top++;
      slotsOf(key, seed, tri);
      for (let j = 0; j < 3; j++) {
        const t = tri[j];
        cnt[t]--;
        xr[t] ^= key;
        work++;
        if (cnt[t] === 1) queue.push(t);
      }
    }
    if (top === NBIG) ok = true;
    else seed++;
  }
  const B = new Int32Array(cap);
  if (ok) {
    for (let n = NBIG - 1; n >= 0; n--) {
      const key = stackKey[n];
      const s = stackSlot[n];
      slotsOf(key, seed, tri);
      let v = mix32(key, 99) & mask;
      for (let j = 0; j < 3; j++) if (tri[j] !== s) v ^= B[tri[j]];
      B[s] = v;
      work++;
    }
  }
  let fpCount = 0;
  let lineTotal = 0;
  const sd = seed;
  for (let t = 0; t < NPROBE; t++) {
    const id = PROBE_BASE + t;
    slotsOf(id, sd, tri);
    const v = B[tri[0]] ^ B[tri[1]] ^ B[tri[2]];
    if (v === (mix32(id, 99) & mask)) fpCount++;
    const l0 = ((tri[0] * f) / LINE_BITS) | 0;
    const l1 = ((tri[1] * f) / LINE_BITS) | 0;
    const l2 = ((tri[2] * f) / LINE_BITS) | 0;
    lineTotal += new Set([l0, l1, l2]).size;
  }
  return summarize(
    'xor',
    (cap * f) / NBIG,
    fpCount,
    lineTotal,
    work,
    ok,
    `${f}-bit fingerprints, 3 segments of ${fmtNum(seg)}, ${attempts} peel attempt${attempts === 1 ? '' : 's'}`,
  );
}

type RibbonSolve = { ok: boolean; work: number; Z: Int32Array; m: number };

function ribbonSolve(n: number, m: number, w: number, r: number, keyAt: (i: number) => number): RibbonSolve {
  const wmask = w >= 32 ? 0xffffffff : ((1 << w) - 1) >>> 0;
  const rmask = r >= 32 ? 0xffffffff : ((1 << r) - 1) >>> 0;
  const pivotRow = new Int32Array(m);
  const pivotRes = new Int32Array(m);
  let work = 0;
  let ok = true;
  for (let idx = 0; idx < n && ok; idx++) {
    const id = keyAt(idx);
    let i = mix32(id, 21) % (m - w);
    let c = ((mix32(id, 22) & wmask) | 1) >>> 0;
    let res = mix32(id, 23) & rmask;
    let placed = false;
    while (c !== 0) {
      const t = ctz32(c);
      i += t;
      c = c >>> t;
      if (i >= m) break;
      work++;
      if (pivotRow[i] === 0) {
        pivotRow[i] = c;
        pivotRes[i] = res;
        placed = true;
        break;
      }
      c = (c ^ pivotRow[i]) >>> 0;
      res ^= pivotRes[i];
    }
    if (!placed && (c !== 0 || res !== 0)) ok = false;
  }
  const Z = new Int32Array(m + w + 1);
  if (ok) {
    for (let i = m - 1; i >= 0; i--) {
      if (pivotRow[i] === 0) continue;
      let v = pivotRes[i];
      let c = pivotRow[i] >>> 1;
      let j = 1;
      while (c !== 0) {
        if (c & 1) v ^= Z[i + j];
        c >>>= 1;
        j++;
        work++;
      }
      Z[i] = v;
    }
  }
  return { ok, work, Z, m };
}

const RIBBON_FACTORS = [1.02, 1.03, 1.04, 1.05, 1.07, 1.09, 1.12, 1.16, 1.22, 1.3, 1.45, 1.7, 2.0];

function measureRibbon(bits: number, w: number): Measured {
  let chosen = 0;
  let solved: RibbonSolve | null = null;
  let r = 2;
  let work = 0;
  for (const fct of RIBBON_FACTORS) {
    const m = Math.max(w * 2, Math.ceil(NBIG * fct));
    r = Math.max(2, Math.min(24, Math.round(bits / fct)));
    const s = ribbonSolve(NBIG, m, w, r, (i) => i + 1);
    work += s.work;
    if (s.ok) {
      chosen = fct;
      solved = s;
      break;
    }
  }
  if (!solved) {
    return summarize('ribbon', NBIG * 2, 0, NPROBE, work, false, `no size in the search succeeded at w = ${w}`);
  }
  const m = solved.m;
  const Z = solved.Z;
  const wmask = w >= 32 ? 0xffffffff : ((1 << w) - 1) >>> 0;
  const rmask = ((1 << r) - 1) >>> 0;
  let fpCount = 0;
  let lineTotal = 0;
  for (let t = 0; t < NPROBE; t++) {
    const id = PROBE_BASE + t;
    const s = mix32(id, 21) % (m - w);
    let c = ((mix32(id, 22) & wmask) | 1) >>> 0;
    const want = mix32(id, 23) & rmask;
    let v = 0;
    let j = 0;
    let lo = -1;
    let hi = -1;
    while (c !== 0) {
      if (c & 1) {
        v ^= Z[s + j];
        if (lo < 0) lo = s + j;
        hi = s + j;
      }
      c >>>= 1;
      j++;
    }
    if (v === want) fpCount++;
    const a = ((lo * r) / LINE_BITS) | 0;
    const b = (((hi + 1) * r - 1) / LINE_BITS) | 0;
    lineTotal += b - a + 1;
  }
  return summarize(
    'ribbon',
    (m * r) / NBIG,
    fpCount,
    lineTotal,
    work,
    true,
    `band w = ${w}, ${r}-bit results, smallest table that solved = ${chosen.toFixed(2)}n`,
  );
}
