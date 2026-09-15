/**
 * Pure model for the block-compression page: sample blocks, an LZ4 block compressor, and the
 * InnoDB page-compression arithmetic. No React here, so node scripts can import it directly.
 *
 * LZ4: a line-by-line port of LZ4_compress_generic() from lz4 v1.10.0 (lib/lz4.c) for the case
 * every 16 KB page hits: a single independent block smaller than LZ4_64Klimit, so the hash table
 * is `byU16` with LZ4_HASHLOG+1 = 13 bits, LZ4_hash4() on a 4-byte little-endian read,
 * acceleration 1 (skip trigger 6), backward "catch up", a table fill at ip-2 and an immediate
 * test of the next position after every match. That is the path LZ4_compress_default() takes,
 * which is what InnoDB's os_file_compress_page() calls, and what `lz4 -1 -BI` emits for a small
 * file. The byte-for-byte equality with lz4 1.10.0 is checked by the build script for the fixture.
 */

/* ------------------------------------------------------------ PRNG (same as Viz.makeRng) */
export function rng(seed = 1) {
  let s = seed >>> 0 || 1;
  return () => {
    s ^= s << 13;
    s ^= s >>> 17;
    s ^= s << 5;
    return ((s >>> 0) % 1e6) / 1e6;
  };
}

/* ------------------------------------------------------------ sample blocks */
export type SampleKind = 'timestamps' | 'json' | 'random';

const TWO32 = 4294967296;
function writeI64(out: Uint8Array, at: number, v: number) {
  // v is an integer with |v| < 2^53; two's complement for negatives.
  let lo: number;
  let hi: number;
  if (v >= 0) {
    lo = v % TWO32;
    hi = Math.floor(v / TWO32);
  } else {
    const a = -v;
    lo = (TWO32 - (a % TWO32)) % TWO32;
    hi = TWO32 - Math.floor(a / TWO32) - (a % TWO32 === 0 ? 0 : 1);
  }
  for (let i = 0; i < 4; i++) out[at + i] = (lo >>> (8 * i)) & 255;
  for (let i = 0; i < 4; i++) out[at + 4 + i] = (hi >>> (8 * i)) & 255;
}
function readI64(b: Uint8Array, at: number) {
  let lo = 0;
  let hi = 0;
  for (let i = 3; i >= 0; i--) lo = lo * 256 + b[at + i];
  for (let i = 3; i >= 0; i--) hi = hi * 256 + b[at + 4 + i];
  if (hi >= 2147483648) return -((TWO32 - hi - (lo === 0 ? 0 : 1)) * TWO32 + (lo === 0 ? 0 : TWO32 - lo));
  return hi * TWO32 + lo;
}

/** Event timestamps in milliseconds, INT64 little-endian, plain encoding: ~1 s cadence with jitter and occasional gaps. */
export function genTimestamps(nBytes: number, seed = 7) {
  const r = rng(seed);
  const n = Math.ceil(nBytes / 8);
  const out = new Uint8Array(n * 8);
  let t = 1_789_372_800_000 + seed * 3_600_000;
  for (let i = 0; i < n; i++) {
    writeI64(out, i * 8, t);
    const u = r();
    t += u < 0.05 ? 1000 + Math.floor(r() * 30) * 1000 : 998 + Math.floor(r() * 5);
  }
  return out.slice(0, nBytes);
}

const EVENTS = ['page_view', 'page_view', 'page_view', 'search', 'add_to_cart', 'checkout', 'login'];
const COUNTRIES = ['US', 'US', 'DE', 'IN', 'BR', 'JP', 'GB', 'FR'];
const pad = (n: number, w: number) => String(n).padStart(w, '0');

/** One JSON row, as an application would log it. */
export function jsonRow(r: () => number, id: number, tsMs: number) {
  const d = new Date(tsMs);
  const ts = `${d.getUTCFullYear()}-${pad(d.getUTCMonth() + 1, 2)}-${pad(d.getUTCDate(), 2)}T${pad(d.getUTCHours(), 2)}:${pad(d.getUTCMinutes(), 2)}:${pad(d.getUTCSeconds(), 2)}.${pad(d.getUTCMilliseconds(), 3)}Z`;
  const ev = EVENTS[Math.floor(r() * EVENTS.length)];
  const user = `u_${pad(Math.floor(r() * 90000) + 10000, 5)}`;
  const sku = `SKU-${pad(Math.floor(r() * 2000) + 10000, 5)}`;
  const country = COUNTRIES[Math.floor(r() * COUNTRIES.length)];
  const ms = 3 + Math.floor(r() * r() * 400);
  return `{"id":${id},"ts":"${ts}","user":"${user}","event":"${ev}","sku":"${sku}","country":"${country}","status":200,"ms":${ms}}\n`;
}

export function genJsonText(nBytes: number, seed = 11) {
  const r = rng(seed);
  let s = '';
  let id = 4_810_000 + seed * 1000;
  let t = 1_789_372_800_000 + seed * 3_600_000;
  while (s.length < nBytes) {
    s += jsonRow(r, id, t);
    id += 1;
    t += 20 + Math.floor(r() * 400);
  }
  return s.slice(0, nBytes);
}

export function genRandom(nBytes: number, seed = 13) {
  const r = rng(seed);
  const out = new Uint8Array(nBytes);
  for (let i = 0; i < nBytes; i++) out[i] = Math.floor(r() * 256);
  return out;
}

export const utf8 = (s: string) => new TextEncoder().encode(s);

export function genSample(kind: SampleKind, nBytes: number, seed?: number): Uint8Array {
  if (kind === 'timestamps') return genTimestamps(nBytes, seed ?? 7);
  if (kind === 'json') return utf8(genJsonText(nBytes, seed ?? 11));
  return genRandom(nBytes, seed ?? 13);
}

/** ClickHouse-style Delta for 8-byte values: first value as is, then v[i] - v[i-1] (two's complement). */
export function delta8(src: Uint8Array) {
  const out = new Uint8Array(src.length);
  const n = Math.floor(src.length / 8);
  let prev = 0;
  for (let i = 0; i < n; i++) {
    const v = readI64(src, i * 8);
    writeI64(out, i * 8, i === 0 ? v : v - prev);
    prev = v;
  }
  out.set(src.subarray(n * 8), n * 8);
  return out;
}

/** Overwrite a fraction of bytes with random bytes. Positions are fixed per seed, so raising the fraction only adds positions. */
export function scramble(src: Uint8Array, fraction: number, seed = 99) {
  if (fraction <= 0) return src;
  const pick = rng(seed);
  const val = rng(seed * 7 + 1);
  const out = src.slice();
  for (let i = 0; i < out.length; i++) {
    const u = pick();
    const b = Math.floor(val() * 256);
    if (u < fraction) out[i] = b;
  }
  return out;
}

/* ------------------------------------------------------------ LZ4 block format */
const MINMATCH = 4;
const LASTLITERALS = 5;
const MFLIMIT = 12;
const LZ4_MIN_LENGTH = MFLIMIT + 1;
const HASHLOG_U16 = 13; // LZ4_MEMORY_USAGE 14 -> LZ4_HASHLOG 12, +1 for byU16
const SKIP_TRIGGER = 6;
export const LZ4_64K_LIMIT = 65536 + MFLIMIT - 1;

export type Lz4Seq = { litStart: number; litLen: number; matchStart: number; matchLen: number; offset: number };
export type Lz4Result = {
  out: Uint8Array;
  seqs: Lz4Seq[]; // the final literal-only sequence has matchLen 0
  literalBytes: number;
  matchedBytes: number;
  tokenBytes: number;
  lengthBytes: number;
  offsetBytes: number;
};

const read32 = (b: Uint8Array, p: number) => b[p] | (b[p + 1] << 8) | (b[p + 2] << 16) | (b[p + 3] << 24);
const hash4 = (b: Uint8Array, p: number) => Math.imul(read32(b, p), 2654435761 | 0) >>> (32 - HASHLOG_U16);

export function lz4Compress(src: Uint8Array): Lz4Result {
  const n = src.length;
  if (n >= LZ4_64K_LIMIT) throw new Error('lz4Compress models the byU16 path only (input < 64 KB)');
  const out = new Uint8Array(n + Math.floor(n / 255) + 16);
  let op = 0;
  const seqs: Lz4Seq[] = [];
  let tokenBytes = 0;
  let lengthBytes = 0;
  let offsetBytes = 0;
  const table = new Uint16Array(1 << HASHLOG_U16);
  const mflimitPlusOne = n - MFLIMIT + 1;
  const matchlimit = n - LASTLITERALS;
  let anchor = 0;
  let ip = 0;

  const writeLen = (len: number) => {
    for (; len >= 255; len -= 255) {
      out[op++] = 255;
      lengthBytes++;
    }
    out[op++] = len;
    lengthBytes++;
  };
  const count = (pIn: number, pMatch: number, limit: number) => {
    const start = pIn;
    while (pIn < limit && src[pIn] === src[pMatch]) {
      pIn++;
      pMatch++;
    }
    return pIn - start;
  };

  if (n >= LZ4_MIN_LENGTH) {
    table[hash4(src, 0)] = 0;
    ip = 1;
    let forwardH = hash4(src, ip);
    main: for (;;) {
      let match = 0;
      // find a match
      {
        let forwardIp = ip;
        let step = 1;
        let searchMatchNb = 1 << SKIP_TRIGGER;
        for (;;) {
          const h = forwardH;
          const current = forwardIp;
          const matchIndex = table[h];
          ip = forwardIp;
          forwardIp += step;
          step = searchMatchNb++ >> SKIP_TRIGGER;
          if (forwardIp > mflimitPlusOne) break main;
          match = matchIndex;
          forwardH = hash4(src, forwardIp);
          table[h] = current;
          if (read32(src, match) === read32(src, ip)) break;
        }
      }
      // catch up
      if (match > 0 && src[ip - 1] === src[match - 1]) {
        do {
          ip--;
          match--;
        } while (ip > anchor && match > 0 && src[ip - 1] === src[match - 1]);
      }
      // literals
      let litLen = ip - anchor;
      let token = op++;
      tokenBytes++;
      if (litLen >= 15) {
        out[token] = 15 << 4;
        writeLen(litLen - 15);
      } else out[token] = litLen << 4;
      out.set(src.subarray(anchor, ip), op);
      op += litLen;
      let litStart = anchor;

      for (;;) {
        // _next_match
        const offset = ip - match;
        out[op++] = offset & 255;
        out[op++] = offset >>> 8;
        offsetBytes += 2;
        const matchStart = ip;
        const matchCode = count(ip + MINMATCH, match + MINMATCH, matchlimit);
        ip += matchCode + MINMATCH;
        if (matchCode >= 15) {
          out[token] += 15;
          writeLen(matchCode - 15);
        } else out[token] += matchCode;
        seqs.push({ litStart, litLen, matchStart, matchLen: matchCode + MINMATCH, offset });
        anchor = ip;
        if (ip >= mflimitPlusOne) break main;
        table[hash4(src, ip - 2)] = ip - 2;
        // test next position
        const h = hash4(src, ip);
        const matchIndex = table[h];
        table[h] = ip;
        if (read32(src, matchIndex) === read32(src, ip)) {
          match = matchIndex;
          token = op++;
          tokenBytes++;
          out[token] = 0;
          litLen = 0;
          litStart = ip;
          continue;
        }
        break;
      }
      forwardH = hash4(src, ++ip);
    }
  }
  // last literals
  const lastRun = n - anchor;
  out[op++] = (lastRun >= 15 ? 15 : lastRun) << 4;
  tokenBytes++;
  if (lastRun >= 15) writeLen(lastRun - 15);
  out.set(src.subarray(anchor, n), op);
  op += lastRun;
  seqs.push({ litStart: anchor, litLen: lastRun, matchStart: n, matchLen: 0, offset: 0 });

  const literalBytes = seqs.reduce((s, q) => s + q.litLen, 0);
  return { out: out.slice(0, op), seqs, literalBytes, matchedBytes: n - literalBytes, tokenBytes, lengthBytes, offsetBytes };
}

/** Reference LZ4 block decoder, used to prove the compressor round-trips. */
export function lz4Decompress(block: Uint8Array, maxOut: number) {
  const out = new Uint8Array(maxOut);
  let ip = 0;
  let op = 0;
  while (ip < block.length) {
    const token = block[ip++];
    let lit = token >>> 4;
    if (lit === 15) {
      let b;
      do {
        b = block[ip++];
        lit += b;
      } while (b === 255);
    }
    out.set(block.subarray(ip, ip + lit), op);
    ip += lit;
    op += lit;
    if (ip >= block.length) break;
    const offset = block[ip] | (block[ip + 1] << 8);
    ip += 2;
    let ml = (token & 15) + MINMATCH;
    if ((token & 15) === 15) {
      let b;
      do {
        b = block[ip++];
        ml += b;
      } while (b === 255);
    }
    for (let i = 0; i < ml; i++) out[op + i] = out[op + i - offset];
    op += ml;
  }
  return out.slice(0, op);
}

/* ------------------------------------------------------------ point-read cost model */
export type CodecId = 'none' | 'lz4' | 'lz4hc' | 'zstd1' | 'zstd3' | 'zstd9' | 'zstd19' | 'zstd3dict' | 'zlib6';

/**
 * Point-read model: one random read of the compressed block, then decompress the whole block.
 * io = latencyUs + compressedBytes / readMBps; cpu = blockBytes / decompressMBps. MB = 1e6 bytes.
 * The block cache is ignored. A compressed block already in the OS page cache is modelled by the caller passing
 * latencyUs = 0 and readMBps = Infinity, which leaves only the decompression.
 */
export function pointReadUs(blockBytes: number, compressedBytes: number, decompressMBps: number | null, latencyUs: number, readMBps: number) {
  const io = latencyUs + (compressedBytes / (readMBps * 1e6)) * 1e6;
  const cpu = decompressMBps == null ? 0 : (blockBytes / (decompressMBps * 1e6)) * 1e6;
  return { io, cpu, total: io + cpu };
}

/* ------------------------------------------------------------ InnoDB: two ways to compress a 16 KB page */
/*
 * Constants from mysql-server 8.4: FIL_PAGE_DATA = 38 (fil0types.h); PAGE_DATA = 38 + 36 + 2 * 10 = 94
 * (page0types.h); on a clustered-index leaf of a ROW_FORMAT=COMPRESSED page every record costs
 * PAGE_ZIP_DIR_SLOT_SIZE (2) + DATA_TRX_ID_LEN (6) + DATA_ROLL_PTR_LEN (7) = 15 uncompressed trailer
 * bytes (page0zip.h, data0type.h); REC_N_NEW_EXTRA_BYTES = 5.
 *
 * Model assumptions (labelled on screen):
 *  - Rows are synthetic JSON-like text; a record is the row text plus a 5-byte header.
 *  - A bulk load fills each uncompressed page to 15/16 of 16 KB. A page whose records outgrow 16 KB
 *    splits; the new right half is allocated at the end of the file.
 *  - The "compressed stream" is the LZ4 size of the page's record bytes. ROW_FORMAT=COMPRESSED really
 *    uses zlib (innodb_compression_level = 6), so real zip pages are smaller than these; page
 *    compression with COMPRESSION='lz4' really calls LZ4_compress_default() on the 16,346 bytes after
 *    the FIL header, which is exactly what punchPage() computes.
 *  - Padding (innodb_compression_pad_pct_max), index-information bytes and the sparse directory are ignored.
 *  - Every update is flushed before the next, so page compression recompresses the page on every update.
 *  - "Same size" updates change one number without changing its width; token updates grow the record once.
 */
export const PAGE_SIZE = 16384;
export const FIL_PAGE_DATA = 38;
export const PAGE_DATA = 94;
export const REC_EXTRA = 5;
export const CLUST_LEAF_SLOT = 15;
const INF_SUP = 26; // infimum + supremum records on a new-style page
const FILL = (PAGE_SIZE * 15) / 16;

export type InnoRow = { id: number; version: number };
export type InnoCfg = { loadTok: number; updateTok: number; zipSize: number; fsBlock: number };

/**
 * Row text for (id, version). Every version is the same row with a new "ms" value of the same width, so an update by
 * itself does not change the record's size. Version 0 carries a loadTok-char session token, later versions a fresh
 * updateTok-char one.
 */
export function rowText(id: number, version: number, cfg: Pick<InnoCfg, 'loadTok' | 'updateTok'>) {
  let s = jsonRow(rng(id * 131 + 17), id, 1_789_372_800_000 + id * 173).trimEnd();
  const r = rng(id * 131 + version * 7919 + 17);
  if (version > 0) s = s.replace(/"ms":(\d+)\}$/, (_m, d: string) => `"ms":${10 ** (d.length - 1) + Math.floor(r() * 9 * 10 ** (d.length - 1))}}`);
  const n = version === 0 ? cfg.loadTok : cfg.updateTok;
  if (n > 0) {
    let tok = '';
    for (let i = 0; i < n; i++) tok += '0123456789abcdef'[Math.floor(r() * 16)];
    s = s.slice(0, -1) + `,"session":"${tok}"}`;
  }
  return s;
}
const recBytes = (row: InnoRow, cfg: InnoCfg) => utf8(rowText(row.id, row.version, cfg));
const recLen = (row: InnoRow, cfg: InnoCfg) => rowText(row.id, row.version, cfg).length;

/** Bytes an uncompressed 16 KB page needs for these records (header, records, a 2-byte slot per 8 records, trailer). */
export const pageBytesUsed = (rows: InnoRow[], cfg: InnoCfg) => PAGE_DATA + INF_SUP + rows.reduce((s, r) => s + recLen(r, cfg) + REC_EXTRA, 0) + Math.ceil((rows.length + 2) / 8) * 2 + 8;

/** The uncompressed 16 KB page image: FIL header, page header, records with 5-byte headers, zeroed free space. */
export function pageImage(rows: InnoRow[], pageNo: number, cfg: InnoCfg) {
  const img = new Uint8Array(PAGE_SIZE);
  img[4] = (pageNo >>> 24) & 255;
  img[5] = (pageNo >>> 16) & 255;
  img[6] = (pageNo >>> 8) & 255;
  img[7] = pageNo & 255;
  img[24] = 0x45; // FIL_PAGE_TYPE = FIL_PAGE_INDEX (17855)
  img[25] = 0xbf;
  img[FIL_PAGE_DATA + 4] = (rows.length + 2) >>> 8;
  img[FIL_PAGE_DATA + 5] = (rows.length + 2) & 255;
  let p = PAGE_DATA + INF_SUP;
  rows.forEach((row, i) => {
    const b = recBytes(row, cfg);
    if (p + REC_EXTRA + b.length > PAGE_SIZE - 8) return;
    img[p + 2] = ((i + 2) << 3) >>> 8;
    img[p + 3] = ((i + 2) << 3) & 255;
    img[p + 4] = b.length & 255;
    p += REC_EXTRA;
    img.set(b, p);
    p += b.length;
  });
  return img;
}

function recordStream(rows: InnoRow[], cfg: InnoCfg) {
  const parts = rows.map((r) => recBytes(r, cfg));
  const out = new Uint8Array(parts.reduce((s, b) => s + b.length, 0));
  let p = 0;
  for (const b of parts) {
    out.set(b, p);
    p += b.length;
  }
  return out;
}

/** Rows per page for a bulk load: records fill an uncompressed page to 15/16. */
export function loadRows(nRows: number, cfg: InnoCfg) {
  const pages: InnoRow[][] = [];
  let cur: InnoRow[] = [];
  let used = PAGE_DATA + INF_SUP;
  for (let id = 1; id <= nRows; id++) {
    const len = rowText(id, 0, cfg).length + REC_EXTRA;
    if (used + len > FILL && cur.length) {
      pages.push(cur);
      cur = [];
      used = PAGE_DATA + INF_SUP;
    }
    cur.push({ id, version: 0 });
    used += len;
  }
  if (cur.length) pages.push(cur);
  return pages;
}

/* ---- ROW_FORMAT=COMPRESSED ---- */
/** dead: records left on the page's free list (a size-changing update deletes the old record and allocates a new one from the heap); they keep their trailer slots until the page is recompressed. */
export type ZipPage = { rows: InnoRow[]; compressed: number; mlog: number; dead: number };
export const zipCompressed = (rows: InnoRow[], cfg: InnoCfg) => lz4Compress(recordStream(rows, cfg)).out.length;
/** page_zip_get_trailer_len(): (n_heap - 2) x 15 bytes on a clustered-index leaf; free-list records count too. */
export const zipTrailer = (p: Pick<ZipPage, 'rows' | 'dead'>) => (p.rows.length + p.dead) * CLUST_LEAF_SLOT;
/** m_end: where the modification log ends. */
export const zipMEnd = (p: ZipPage) => PAGE_DATA + p.compressed + p.mlog;
/**
 * page_zip_available(): (length - (REC_N_NEW_EXTRA_BYTES - 2)) + trailer + m_end < zip size, where a created record
 * (create = true, the delete-and-insert path of a size-changing update) also reserves one more 2-byte directory slot.
 * length is the physical record, which includes DB_TRX_ID and DB_ROLL_PTR (13 bytes); the log entry does not carry
 * them (they live in the trailer), so the check covers a new record's 15-byte trailer slot.
 */
const SYS_COLS = 13;
export const zipAvailable = (p: ZipPage, rowLen: number, zipSize: number, create: boolean) =>
  rowLen + REC_EXTRA + SYS_COLS - (REC_EXTRA - 2) + zipTrailer(p) + (create ? 2 : 0) + zipMEnd(p) < zipSize;
export const zipFits = (p: ZipPage, cfg: InnoCfg) => PAGE_DATA + p.compressed + zipTrailer(p) < cfg.zipSize && pageBytesUsed(p.rows, cfg) <= PAGE_SIZE;

/** Compress; while a page does not fit its zip size (a compression failure) or its 16 KB image, split it in half. */
export function zipBuild(rows: InnoRow[], cfg: InnoCfg): ZipPage[] {
  const p: ZipPage = { rows, compressed: zipCompressed(rows, cfg), mlog: 0, dead: 0 };
  if (zipFits(p, cfg) || rows.length <= 1) return [p];
  const mid = Math.ceil(rows.length / 2);
  return [...zipBuild(rows.slice(0, mid), cfg), ...zipBuild(rows.slice(mid), cfg)];
}

export function zipLoad(nRows: number, cfg: InnoCfg) {
  return loadRows(nRows, cfg).flatMap((rows) => zipBuild(rows, cfg));
}

/** mlog: appended to the log. reorganize: recompressed, then the record fits the empty log. repack: recompressed with the record inside the stream. split: compression failure. */
export type ZipOutcome = 'mlog' | 'reorganize' | 'repack' | 'split';

/**
 * Update one row on a compressed page (mysql-server 8.4).
 *  1. btr_cur_update_alloc_zip() (btr0cur.cc): if page_zip_available() says the new record fits between m_end and the
 *     trailer, it is appended to the modification log. Otherwise, unless the page is freshly compressed (empty log,
 *     no garbage), btr_page_reorganize() rebuilds the page from its uncompressed frame and recompresses it with an
 *     empty log, and page_zip_available() is asked again.
 *  2. If the record still does not fit, the update takes the pessimistic path: page_cur_insert_rec_zip() (page0cur.cc)
 *     inserts the record into the uncompressed page and compresses the whole page with it (page_zip_reorganize()).
 *  3. Only if that compression fails — a compression failure — does btr_cur_pessimistic_insert() split the page.
 *     Split-off pages are allocated at the end of the file.
 * A page whose uncompressed records outgrow 16 KB splits like any B-tree page.
 */
export function zipUpdate(pages: ZipPage[], pageIdx: number, rowIdx: number, cfg: InnoCfg) {
  const page = pages[pageIdx];
  const row = page.rows[rowIdx];
  const newRow = { id: row.id, version: row.version + 1 };
  const len = recLen(newRow, cfg);
  const create = len !== recLen(row, cfg); // same size: btr_cur_update_in_place (create = false); size change: delete + insert (create = true)
  const entry = len + 2; // heap_no (1-2 bytes) + the record, uncompressed
  const rows = page.rows.map((r, i) => (i === rowIdx ? newRow : r));
  const pageFull = pageBytesUsed(rows, cfg) > PAGE_SIZE;
  const done = (outcome: ZipOutcome, p: ZipPage, recompressions: number) => {
    const next = pages.slice();
    next[pageIdx] = p;
    return { pages: next, outcome, entry, touched: [pageIdx], recompressions, reason: null };
  };
  let recompressions = 0;
  if (!pageFull) {
    if (zipAvailable(page, len, cfg.zipSize, create)) return done('mlog', { ...page, rows, mlog: page.mlog + entry, dead: page.dead + (create ? 1 : 0) }, 0);
    if (page.mlog > 0 || page.dead > 0) {
      const reorganized: ZipPage = { rows: page.rows, compressed: zipCompressed(page.rows, cfg), mlog: 0, dead: 0 };
      recompressions++;
      if (zipAvailable(reorganized, len, cfg.zipSize, create)) return done('reorganize', { ...reorganized, rows, mlog: entry, dead: create ? 1 : 0 }, recompressions);
    }
    const repacked: ZipPage = { rows, compressed: zipCompressed(rows, cfg), mlog: 0, dead: 0 };
    recompressions++;
    if (zipFits(repacked, cfg)) return done('repack', repacked, recompressions);
  }
  const reason: 'page-full' | 'compression-failure' = pageFull ? 'page-full' : 'compression-failure';
  const mid = Math.ceil(rows.length / 2);
  const left = zipBuild(rows.slice(0, mid), cfg);
  const right = zipBuild(rows.slice(mid), cfg);
  const next = pages.slice();
  next.splice(pageIdx, 1, left[0]);
  const appended = [...left.slice(1), ...right];
  next.push(...appended);
  const touched = [pageIdx, ...appended.map((_, i) => pages.length + i)];
  return { pages: next, outcome: 'split' as ZipOutcome, entry, touched, recompressions: recompressions + left.length + right.length, reason };
}

/* ---- COMPRESSION='lz4' + hole punching ---- */
export type PunchPage = { rows: InnoRow[]; lz4: number; stored: number; blocks: number; raw: boolean };

/** os_file_compress_page(): compress the bytes after the 38-byte FIL header; keep it only if it saves at least one FS block. */
export function punchPage(rows: InnoRow[], pageNo: number, cfg: InnoCfg): PunchPage {
  const img = pageImage(rows, pageNo, cfg);
  const lz4 = lz4Compress(img.subarray(FIL_PAGE_DATA)).out.length;
  const outLen = PAGE_SIZE - (FIL_PAGE_DATA + cfg.fsBlock);
  if (PAGE_SIZE < cfg.fsBlock * 2 || lz4 >= outLen) return { rows, lz4, stored: PAGE_SIZE, blocks: PAGE_SIZE / cfg.fsBlock, raw: true };
  const stored = Math.ceil((lz4 + FIL_PAGE_DATA) / cfg.fsBlock) * cfg.fsBlock;
  return { rows, lz4, stored, blocks: stored / cfg.fsBlock, raw: false };
}

export function punchBuild(rows: InnoRow[], pageNo: number, cfg: InnoCfg): InnoRow[][] {
  if (pageBytesUsed(rows, cfg) <= PAGE_SIZE || rows.length <= 1) return [rows];
  const mid = Math.ceil(rows.length / 2);
  return [...punchBuild(rows.slice(0, mid), pageNo, cfg), ...punchBuild(rows.slice(mid), pageNo, cfg)];
}

export function punchLoad(nRows: number, cfg: InnoCfg) {
  return loadRows(nRows, cfg).map((rows, i) => punchPage(rows, i, cfg));
}

/** Update a row, flush: the whole page is recompressed and rewritten; a full page splits, its right half appended to the file. */
export function punchUpdate(pages: PunchPage[], pageIdx: number, rowIdx: number, cfg: InnoCfg) {
  const page = pages[pageIdx];
  const rows = page.rows.map((r, i) => (i === rowIdx ? { id: r.id, version: r.version + 1 } : r));
  const parts = punchBuild(rows, pageIdx, cfg);
  const next = pages.slice();
  next[pageIdx] = punchPage(parts[0], pageIdx, cfg);
  const appended = parts.slice(1).map((rs, i) => punchPage(rs, pages.length + i, cfg));
  next.push(...appended);
  return { pages: next, before: page.blocks, after: next[pageIdx].blocks, split: appended.length > 0, touched: [pageIdx, ...appended.map((_, i) => pages.length + i)] };
}
