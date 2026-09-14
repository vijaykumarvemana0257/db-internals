/**
 * Page-checksum model shared by the torn-write visuals.
 *
 * Two real families are modelled:
 *
 *  - PostgreSQL `data_checksums`: 32 parallel FNV-1a lanes over the block,
 *    xor-folded to one 32-bit value, mixed with the block number, then
 *    squeezed into the 16-bit `pd_checksum` field as `(sum % 65535) + 1`
 *    (zero is reserved for "no checksum"). See `pg_checksum_block()` in
 *    src/include/storage/checksum_impl.h.
 *  - InnoDB: CRC-32C (Castagnoli, reflected poly 0x82F63B78) over the page,
 *    stored in the 4-byte header field and repeated in the page trailer.
 *
 * The FNV base offsets below stand in for the fixed 32-constant table
 * PostgreSQL ships; the mixing step, the fold and the modulo are the real ones.
 * Detection probability depends on the width of the stored field, not on the
 * constants, so the simulations these feed are faithful about the thing they
 * teach.
 */

export const N_SUMS = 32;
const FNV_PRIME = 16777619;

const BASE_OFFSETS = Uint32Array.from({ length: N_SUMS }, (_, i) =>
  (Math.imul(i + 1, 0x9e3779b1) ^ 0x5b1f36e1) >>> 0,
);

/** The 32-bit block sum, before it is folded into a stored field. */
export function fnvBlockSum(words: Uint32Array, blkno: number, mixBlkno: boolean): number {
  const sums = new Uint32Array(N_SUMS);
  for (let i = 0; i < N_SUMS; i++) sums[i] = BASE_OFFSETS[i];
  const rows = Math.floor(words.length / N_SUMS);
  for (let r = 0; r < rows; r++) {
    const base = r * N_SUMS;
    for (let j = 0; j < N_SUMS; j++) {
      const tmp = (sums[j] ^ words[base + j]) >>> 0;
      sums[j] = (Math.imul(tmp, FNV_PRIME) ^ (tmp >>> 17)) >>> 0;
    }
  }
  let result = 0;
  for (let i = 0; i < N_SUMS; i++) result = (result ^ sums[i]) >>> 0;
  return mixBlkno ? (result ^ blkno) >>> 0 : result;
}

/** Fold a 32-bit sum into a stored field of `bits` width, PostgreSQL style. */
export function foldToWidth(sum: number, bits: number): number {
  if (bits >= 32) return sum >>> 0;
  const m = 2 ** bits - 1;
  return ((sum >>> 0) % m) + 1;
}

const CRC_TABLE = (() => {
  const t = new Uint32Array(256);
  for (let n = 0; n < 256; n++) {
    let c = n;
    for (let k = 0; k < 8; k++) c = c & 1 ? (0x82f63b78 ^ (c >>> 1)) >>> 0 : c >>> 1;
    t[n] = c >>> 0;
  }
  return t;
})();

/** CRC-32C over the little-endian byte image of the words. */
export function crc32c(words: Uint32Array, blkno: number, mixBlkno: boolean): number {
  let c = 0xffffffff;
  if (mixBlkno) {
    for (let s = 0; s < 32; s += 8) c = (CRC_TABLE[(c ^ ((blkno >>> s) & 0xff)) & 0xff] ^ (c >>> 8)) >>> 0;
  }
  const bytes = new Uint8Array(words.buffer, words.byteOffset, words.byteLength);
  for (let i = 0; i < bytes.length; i++) c = (CRC_TABLE[(c ^ bytes[i]) & 0xff] ^ (c >>> 8)) >>> 0;
  return (c ^ 0xffffffff) >>> 0;
}

export type Scheme = { id: string; label: string; bits: number; engine: string };

export const SCHEMES: Scheme[] = [
  { id: 'fnv8', label: '8-bit fold', bits: 8, engine: 'illustrative only' },
  { id: 'fnv16', label: '16-bit FNV fold', bits: 16, engine: 'PostgreSQL pd_checksum' },
  { id: 'fnv32', label: '32-bit FNV sum', bits: 32, engine: 'no shipping engine' },
  { id: 'crc32c', label: 'CRC-32C', bits: 32, engine: 'InnoDB, ZFS metadata' },
];

/** Compute the stored field for one scheme over a page image. */
export function checksumOf(scheme: string, words: Uint32Array, blkno: number, mixBlkno: boolean): number {
  if (scheme === 'crc32c') return crc32c(words, blkno, mixBlkno);
  const bits = scheme === 'fnv8' ? 8 : scheme === 'fnv16' ? 16 : 32;
  return foldToWidth(fnvBlockSum(words, blkno, mixBlkno), bits);
}

export function hex(v: number, bits: number) {
  return `0x${(v >>> 0).toString(16).toUpperCase().padStart(bits / 4, '0')}`;
}

/**
 * The page is modelled at one 32-bit word per 64 bytes of real page, so a
 * 16 KB page is 256 words. Word 0 is the checksum field (zeroed while the
 * checksum is computed, exactly as PostgreSQL does), word 1 is the pageLSN.
 */
export const BYTES_PER_WORD = 64;

export function makePageWords(pageBytes: number, sectorBytes: number, version: number): Uint32Array {
  const wordsPerSector = sectorBytes / BYTES_PER_WORD;
  const sectors = pageBytes / sectorBytes;
  const out = new Uint32Array(wordsPerSector * sectors);
  for (let s = 0; s < sectors; s++) {
    // Deterministic per (version, sector): no Math.random anywhere.
    let x = (Math.imul(version + 1, 0x85ebca6b) ^ Math.imul(s + 1, 0xc2b2ae35)) >>> 0 || 1;
    for (let i = 0; i < wordsPerSector; i++) {
      x ^= x << 13;
      x >>>= 0;
      x ^= x >>> 17;
      x ^= x << 5;
      x >>>= 0;
      out[s * wordsPerSector + i] = x;
    }
  }
  return out;
}
