import { useMemo, useState } from 'react';
import {
  VizPanel,
  Slider,
  Choice,
  Check,
  Button,
  Legend,
  Stats,
  Note,
  TooltipHost,
  useTip,
  makeRng,
  fmtNum,
} from './Viz';
import { fnvBlockSum, foldToWidth } from './pageChecksumModel';

/**
 * A page inspector: real header layouts, a real verification path.
 *
 * Everything structural here is the shipping format:
 *
 *  - PostgreSQL `PageHeaderData`: pd_lsn(8) pd_checksum(2) pd_flags(2) pd_lower(2)
 *    pd_upper(2) pd_special(2) pd_pagesize_version(2) pd_prune_xid(4), then the
 *    ItemIdData line pointers. pd_checksum is zeroed before hashing and the block
 *    number is mixed in, so the page only verifies at the block it was written for.
 *    Layout version 3 (8.0) has no pd_prune_xid; before 9.3 the two bytes at offset 8
 *    were pd_tli, and checksums were bolted on by *reusing* that field rather than
 *    bumping PG_PAGE_LAYOUT_VERSION — which is why data_checksums is a control-file
 *    flag, not a per-page one.
 *  - InnoDB: 38-byte FIL header + 8-byte trailer. The checksum deliberately covers
 *    only bytes 4..25 and 38..(size-8): the checksum field itself, FIL_PAGE_FILE_FLUSH_LSN
 *    and the trailer are excluded. The trailer repeats the low 32 bits of FIL_PAGE_LSN,
 *    so a head/tail mismatch is a half-written page. MariaDB's full_crc32 replaces all
 *    of that with one CRC-32C over everything but the last four bytes.
 *  - SQLite: no page checksums at all. The 100-byte file header carries the page size,
 *    the read/write format numbers and the per-page reserved-bytes count that the
 *    optional checksum VFS uses for its 8-byte trailer.
 *
 * The checksum *constants* are stand-ins (see pageChecksumModel.ts) — the widths, the
 * covered byte ranges, the zeroing and the block-number mixing are the real ones, and
 * they are what the page teaches.
 */

/* ------------------------------------------------------------------- bytes */

const hx = (n: number) => n.toString(16).toUpperCase().padStart(2, '0');
const hex32 = (n: number) => `0x${(n >>> 0).toString(16).toUpperCase().padStart(8, '0')}`;
const hex16 = (n: number) => `0x${(n & 0xffff).toString(16).toUpperCase().padStart(4, '0')}`;

const u16be = (b: Uint8Array, o: number) => (b[o] << 8) | b[o + 1];
const u32be = (b: Uint8Array, o: number) => ((b[o] << 24) | (b[o + 1] << 16) | (b[o + 2] << 8) | b[o + 3]) >>> 0;
const u16le = (b: Uint8Array, o: number) => b[o] | (b[o + 1] << 8);
const u32le = (b: Uint8Array, o: number) => (b[o] | (b[o + 1] << 8) | (b[o + 2] << 16) | (b[o + 3] << 24)) >>> 0;

function w16be(b: Uint8Array, o: number, v: number) {
  b[o] = (v >>> 8) & 0xff;
  b[o + 1] = v & 0xff;
}
function w32be(b: Uint8Array, o: number, v: number) {
  b[o] = (v >>> 24) & 0xff;
  b[o + 1] = (v >>> 16) & 0xff;
  b[o + 2] = (v >>> 8) & 0xff;
  b[o + 3] = v & 0xff;
}
function w16le(b: Uint8Array, o: number, v: number) {
  b[o] = v & 0xff;
  b[o + 1] = (v >>> 8) & 0xff;
}
function w32le(b: Uint8Array, o: number, v: number) {
  b[o] = v & 0xff;
  b[o + 1] = (v >>> 8) & 0xff;
  b[o + 2] = (v >>> 16) & 0xff;
  b[o + 3] = (v >>> 24) & 0xff;
}

function fill(b: Uint8Array, from: number, to: number, seed: number) {
  const rng = makeRng(seed);
  for (let i = from; i < to; i++) b[i] = Math.floor(rng() * 256);
}

/* -------------------------------------------------------------- checksums */

function toWords(b: Uint8Array): Uint32Array {
  const n = b.length >>> 2;
  const w = new Uint32Array(n);
  for (let i = 0; i < n; i++) w[i] = u32le(b, i * 4);
  return w;
}

/** PostgreSQL: pd_checksum zeroed, 32 FNV lanes over the whole block, block number mixed, folded to 16 bits. */
function pgChecksum(page: Uint8Array, blkno: number): number {
  const copy = page.slice();
  copy[8] = 0;
  copy[9] = 0;
  return foldToWidth(fnvBlockSum(toWords(copy), blkno, true), 16);
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

/** CRC-32C over an explicit list of [start, end) byte ranges — InnoDB skips three of them. */
function crc32cRanges(b: Uint8Array, ranges: [number, number][]): number {
  let c = 0xffffffff;
  for (const [s, e] of ranges) for (let i = s; i < e; i++) c = (CRC_TABLE[(c ^ b[i]) & 0xff] ^ (c >>> 8)) >>> 0;
  return (c ^ 0xffffffff) >>> 0;
}

/** Stand-in for InnoDB's pre-5.6 ut_fold_binary sum: a non-CRC additive fold, same coverage. */
function innodbFold(b: Uint8Array, ranges: [number, number][]): number {
  let f = 0x1b873593;
  for (const [s, e] of ranges)
    for (let i = s; i < e; i++) f = (Math.imul(f ^ b[i], 0x85ebca6b) + ((f >>> 13) ^ i)) >>> 0;
  return f >>> 0;
}

/* ---------------------------------------------------------------- the page */

type Role = 'ident' | 'lsn' | 'checksum' | 'bounds' | 'slot' | 'tuple' | 'flags' | 'free';

const ROLE_COLOR: Record<Role, string> = {
  ident: 'var(--viz-1)',
  lsn: 'var(--viz-2)',
  checksum: 'var(--viz-3)',
  bounds: 'var(--viz-4)',
  slot: 'var(--viz-5)',
  tuple: 'var(--viz-6)',
  flags: 'var(--viz-7)',
  free: 'var(--viz-stale)',
};

const ROLE_LABEL: Record<Role, string> = {
  ident: 'identity — who this page is',
  lsn: 'LSN — which version this page is',
  checksum: 'checksum / integrity field',
  bounds: 'space bounds — where data starts and ends',
  slot: 'slot array (line pointers / cell pointers)',
  tuple: 'tuple or record bytes',
  flags: 'flags and counters',
  free: 'free space / payload not decoded here',
};

type Field = {
  kind: 'field';
  off: number;
  len: number;
  name: string;
  role: Role;
  covered: boolean;
  dec: (b: Uint8Array) => string;
  why: string;
};

type Gap = {
  kind: 'gap';
  from: number;
  to: number;
  name: string;
  role: Role;
  covered: boolean;
  flipOff: number;
  why: string;
};

type Item = Field | Gap;
type Region = { label: string; sub: string; items: Item[] };

type Page = {
  bytes: Uint8Array;
  regions: Region[];
  size: number;
  /** bytes the primary checksum actually hashes */
  covered: number;
  lsnText: string;
  blockText: string;
};

type EngineId = 'pg' | 'innodb' | 'sqlite';

const VERSIONS: Record<EngineId, { label: string; short: string }[]> = {
  pg: [
    { label: 'layout 3 — PostgreSQL 8.0–8.2', short: 'v3' },
    { label: 'layout 4 — 8.3+, HOT adds pd_prune_xid', short: 'v4' },
    { label: 'layout 4 + data_checksums — 9.3+', short: 'v4+ck' },
  ],
  innodb: [
    { label: 'innodb — the legacy fold, default through 5.6', short: 'innodb' },
    { label: 'crc32 — default since MySQL 5.7', short: 'crc32' },
    { label: 'full_crc32 — MariaDB 10.4.3+', short: 'full_crc32' },
  ],
  sqlite: [
    { label: 'file format 1 — rollback journal', short: 'fmt 1' },
    { label: 'file format 2 — WAL, 3.7.0+', short: 'fmt 2' },
    { label: 'file format 2 + checksum VFS', short: 'cksumvfs' },
  ],
};

/* -------------------------------------------------------- PostgreSQL 8 KB */

const PG_SIZE = 8192;
const LP_FLAG = ['LP_UNUSED', 'LP_NORMAL', 'LP_REDIRECT', 'LP_DEAD'];

function buildPg(version: number, checksums: boolean, gen: 'old' | 'new', writtenFor: number): Page {
  const b = new Uint8Array(PG_SIZE);
  const hdrLen = version >= 1 ? 24 : 20;
  const nlp = 5;
  const lower = hdrLen + 4 * nlp;
  const tupLen = 40;
  const upper = PG_SIZE - nlp * tupLen;

  // pd_lsn: PostgreSQL stores it as two 32-bit halves, low word first on x86.
  const lsnHi = 1;
  const lsnLo = gen === 'new' ? 0x6b3a70c8 : 0x6b3a6f30;
  w32le(b, 0, lsnLo);
  w32le(b, 4, lsnHi);

  if (version >= 2) {
    w16le(b, 8, 0); // pd_checksum, filled in below
  } else {
    w16le(b, 8, 1); // pd_tli: the timeline this page was last written on
  }
  w16le(b, 10, 0x0004); // pd_flags: PD_ALL_VISIBLE
  w16le(b, 12, lower);
  w16le(b, 14, upper);
  w16le(b, 16, PG_SIZE); // pd_special == BLCKSZ: a heap page has no special space
  w16le(b, 18, PG_SIZE | (version >= 1 ? 4 : 3));
  if (version >= 1) w32le(b, 20, gen === 'new' ? 0x0004e21f : 0);

  // Line pointers: lp_off:15, lp_flags:2, lp_len:15, packed into one 32-bit word.
  const lps = [
    { off: PG_SIZE - 40, flags: 1, len: 40 },
    { off: PG_SIZE - 80, flags: 1, len: 40 },
    { off: 5, flags: 2, len: 0 }, // LP_REDIRECT → line pointer 5
    { off: 0, flags: 3, len: 0 }, // LP_DEAD
    { off: PG_SIZE - 120, flags: 1, len: 40 },
  ];
  lps.forEach((lp, i) => w32le(b, hdrLen + 4 * i, (lp.off | (lp.flags << 15) | (lp.len << 17)) >>> 0));

  // Tuples grow up from the end of the page; free space between pd_lower and pd_upper is zeroed.
  for (let i = 0; i < nlp; i++) {
    const o = PG_SIZE - (i + 1) * tupLen;
    w32le(b, o, 0x0004e100 + i); // t_xmin
    w32le(b, o + 4, i === 0 && gen === 'new' ? 0x0004e21f : 0); // t_xmax
    w32le(b, o + 8, 0); // t_cid
    w32le(b, o + 12, writtenFor); // t_ctid.blkno
    w16le(b, o + 16, i + 1); // t_ctid.offnum
    w16le(b, o + 18, 0x8003); // t_infomask2: 3 attrs | HEAP_ONLY_TUPLE
    w16le(b, o + 20, 0x0901); // t_infomask: HASNULL | XMIN_COMMITTED
    b[o + 22] = 24; // t_hoff
    b[o + 23] = 0b00000111; // null bitmap
    fill(b, o + 24, o + tupLen, 7000 + i * 31 + (gen === 'new' ? 1 : 0));
  }

  if (version >= 2 && checksums) w16le(b, 8, pgChecksum(b, writtenFor));

  const f = (o: number, len: number, name: string, role: Role, dec: (x: Uint8Array) => string, why: string): Field => ({
    kind: 'field',
    off: o,
    len,
    name,
    role,
    covered: true,
    dec,
    why,
  });

  const header: Item[] = [
    f(0, 8, 'pd_lsn', 'lsn', (x) => `${u32le(x, 4).toString(16).toUpperCase()}/${u32le(x, 0).toString(16).toUpperCase().padStart(8, '0')}`,
      'The LSN of the last WAL record that changed this page. Recovery replays a record only if its LSN is greater than this; the buffer manager refuses to write the page until the WAL is flushed past it.'),
    version >= 2
      ? f(8, 2, 'pd_checksum', 'checksum', (x) => `${u16le(x, 8)} (${hex16(u16le(x, 8))})`,
          'Sixteen bits, zeroed before hashing so the field never feeds its own sum. It is only compared when data_checksums is on. Added in 9.3 by repurposing pd_tli rather than bumping the layout version.')
      : f(8, 2, 'pd_tli', 'ident', (x) => `timeline ${u16le(x, 8)}`,
          'The timeline the page was last written on. Vestigial by 9.3 — which is exactly why it was available to become pd_checksum without changing the page layout version.'),
    f(10, 2, 'pd_flags', 'flags', (x) => {
      const v = u16le(x, 10);
      const on = [
        v & 1 ? 'PD_HAS_FREE_LINES' : '',
        v & 2 ? 'PD_PAGE_FULL' : '',
        v & 4 ? 'PD_ALL_VISIBLE' : '',
      ].filter(Boolean);
      return on.length ? on.join(' | ') : `${hex16(v)} — none`;
    }, 'Only three bits are defined (PD_VALID_FLAG_BITS = 0x0007); PageIsVerified rejects a page with any other bit set, so this field is checked even when checksums are off.'),
    f(12, 2, 'pd_lower', 'bounds', (x) => `${u16le(x, 12)}`,
      'End of the line pointer array. PageIsVerified requires pd_lower ≤ pd_upper on every read — a bound loose enough that a plausible-looking wrong value passes.'),
    f(14, 2, 'pd_upper', 'bounds', (x) => `${u16le(x, 14)}`,
      'Start of the tuple area. pd_upper − pd_lower is the contiguous free space the page still has.'),
    f(16, 2, 'pd_special', 'bounds', (x) => `${u16le(x, 16)}`,
      'Start of the access-method-specific area. Equal to BLCKSZ on a heap page; a B-tree page puts its opaque data there instead.'),
    f(18, 2, 'pd_pagesize_version', 'ident', (x) => {
      const v = u16le(x, 18);
      return `${v & 0xff00 ? v & 0xff00 : 0} B | layout ${v & 0xff}`;
    }, 'Page size in the high byte, PG_PAGE_LAYOUT_VERSION in the low. Stamped into every page, but not compared on the read path — a server built for a different BLCKSZ is stopped earlier, by the block size recorded in pg_control.'),
  ];
  if (version >= 1)
    header.push(
      f(20, 4, 'pd_prune_xid', 'flags', (x) => (u32le(x, 20) === 0 ? '0 — nothing to prune' : `xid ${u32le(x, 20)}`),
        'Oldest xid that might make pruning worthwhile. Added in 8.3 together with HOT — the change that made this layout version 4.'),
    );

  const slots: Item[] = lps.map((lp, i) =>
    f(hdrLen + 4 * i, 4, `lp[${i + 1}]`, 'slot', (x) => {
      const w = u32le(x, hdrLen + 4 * i);
      const off = w & 0x7fff;
      const fl = (w >>> 15) & 3;
      const len = (w >>> 17) & 0x7fff;
      return `${LP_FLAG[fl]} off=${off} len=${len}`;
    }, i === 2
      ? 'LP_REDIRECT: the slot holds no tuple, only the offset of the line pointer the HOT chain continues at. Index entries still point here.'
      : i === 3
        ? 'LP_DEAD: pruned. The line pointer must stay — an index may still point at it — but its 40 bytes of tuple have been reclaimed.'
        : 'A live slot: 15 bits of offset, 2 flag bits, 15 bits of length, packed into one 32-bit word so the array costs 4 bytes per tuple.'),
  );

  const regions: Region[] = [
    { label: `PageHeaderData — ${hdrLen} bytes`, sub: 'little-endian on x86; the layout is not portable across byte orders and PostgreSQL does not pretend it is', items: header },
    { label: 'Line pointer array (ItemIdData)', sub: `grows down from byte ${hdrLen} to pd_lower = ${lower}`, items: slots },
    {
      label: 'Free space',
      sub: `pd_lower ${lower} → pd_upper ${upper}`,
      items: [
        {
          kind: 'gap',
          from: lower,
          to: upper,
          name: `${fmtNum(upper - lower)} zero bytes`,
          role: 'free',
          covered: true,
          flipOff: lower + 64,
          why: 'Nothing reads these bytes — but the checksum hashes all 8192, so a flip in the middle of free space still fails verification. Click to flip a bit here.',
        },
      ],
    },
    {
      label: 'Tuple 1 — HeapTupleHeaderData',
      sub: `at pd_upper ${PG_SIZE - 40}, 23-byte header MAXALIGNed to t_hoff = 24`,
      items: [
        f(PG_SIZE - 40, 4, 't_xmin', 'tuple', (x) => `${u32le(x, PG_SIZE - 40)}`, 'Inserting transaction id.'),
        f(PG_SIZE - 36, 4, 't_xmax', 'tuple', (x) => (u32le(x, PG_SIZE - 36) === 0 ? '0 — live' : `${u32le(x, PG_SIZE - 36)}`), 'Deleting or updating transaction id; zero while the tuple is live.'),
        f(PG_SIZE - 32, 4, 't_cid', 'tuple', (x) => `${u32le(x, PG_SIZE - 32)}`, 'Command id within the transaction (union with t_xvac).'),
        f(PG_SIZE - 28, 6, 't_ctid', 'tuple', (x) => `(${u32le(x, PG_SIZE - 28)},${u16le(x, PG_SIZE - 24)})`, 'Self-pointer, or the next version for an updated row. Six bytes: 4-byte block number, 2-byte offset number.'),
        f(PG_SIZE - 22, 2, 't_infomask2', 'flags', (x) => `${u16le(x, PG_SIZE - 22) & 0x7ff} attrs${u16le(x, PG_SIZE - 22) & 0x8000 ? ' | HEAP_ONLY_TUPLE' : ''}`, 'Attribute count in the low 11 bits; HOT bits in the high ones.'),
        f(PG_SIZE - 20, 2, 't_infomask', 'flags', (x) => hex16(u16le(x, PG_SIZE - 20)), 'Hint bits: HEAP_HASNULL, HEAP_XMIN_COMMITTED and friends. Set lazily, which is why a page can become dirty on a pure SELECT.'),
        f(PG_SIZE - 18, 1, 't_hoff', 'bounds', (x) => `${x[PG_SIZE - 18]}`, 'Offset from the start of the tuple to the user data, after the optional null bitmap and MAXALIGN padding.'),
        f(PG_SIZE - 17, 1, 'null bitmap', 'flags', (x) => `0b${x[PG_SIZE - 17].toString(2).padStart(8, '0')}`, 'One bit per attribute, present only when HEAP_HASNULL is set.'),
        {
          kind: 'gap',
          from: PG_SIZE - 16,
          to: PG_SIZE,
          name: '16 bytes of user data',
          role: 'tuple',
          covered: true,
          flipOff: PG_SIZE - 8,
          why: 'The actual column values. Click to corrupt one — this is the flip a checksum exists to catch, and the flip nothing else in the engine would ever notice.',
        },
      ],
    },
  ];

  return {
    bytes: b,
    regions,
    size: PG_SIZE,
    covered: PG_SIZE,
    lsnText: `${lsnHi.toString(16).toUpperCase()}/${lsnLo.toString(16).toUpperCase()}`,
    blockText: `block ${writtenFor} of base/16384/16427`,
  };
}

/* ------------------------------------------------------------ InnoDB 16 KB */

const IN_SIZE = 16384;

function innodbRanges(version: number): [number, number][] {
  // full_crc32: everything except the last four bytes. Otherwise the two classic ranges.
  return version >= 2
    ? [[0, IN_SIZE - 4]]
    : [
        [4, 26],
        [38, IN_SIZE - 8],
      ];
}

function buildInnodb(version: number, checksums: boolean, gen: 'old' | 'new', writtenFor: number): Page {
  const b = new Uint8Array(IN_SIZE);
  const lsnHi = 0;
  const lsnLo = gen === 'new' ? 0x2f1a44c0 : 0x2f1a3d18;

  w32be(b, 4, writtenFor); // FIL_PAGE_OFFSET
  w32be(b, 8, 0xffffffff); // FIL_PAGE_PREV — FIL_NULL
  w32be(b, 12, writtenFor + 1); // FIL_PAGE_NEXT
  w32be(b, 16, lsnHi);
  w32be(b, 20, lsnLo);
  w16be(b, 24, 17855); // FIL_PAGE_INDEX = 0x45BF
  w32be(b, 26, 0);
  w32be(b, 30, 0); // FIL_PAGE_FILE_FLUSH_LSN — only meaningful on page 0 of the system tablespace
  w32be(b, 34, 4); // FIL_PAGE_SPACE_ID

  // PAGE_HEADER, from byte 38.
  w16be(b, 38, 3); // PAGE_N_DIR_SLOTS
  w16be(b, 40, 1200); // PAGE_HEAP_TOP
  w16be(b, 42, 0x8000 | 27); // PAGE_N_HEAP — high bit marks a COMPACT-family row format
  w16be(b, 54, 25); // PAGE_N_RECS
  w16be(b, 64, 0); // PAGE_LEVEL — 0 is a leaf
  w32be(b, 66, 0);
  w32be(b, 70, 62); // PAGE_INDEX_ID

  fill(b, 74, IN_SIZE - 8, gen === 'new' ? 424242 : 424241);

  const ranges = innodbRanges(version);
  if (version >= 2) {
    w32be(b, 0, 0); // unused under full_crc32: the field is ordinary payload
    if (checksums) w32be(b, IN_SIZE - 4, crc32cRanges(b, ranges));
  } else {
    const head = checksums
      ? version === 1
        ? crc32cRanges(b, ranges)
        : innodbFold(b, ranges)
      : 0xdeadbeef; // BUF_NO_CHECKSUM_MAGIC is what "none" writes
    w32be(b, 0, head);
    // Trailer: 4 bytes of checksum, then the low 32 bits of FIL_PAGE_LSN repeated.
    const tail = checksums ? (version === 1 ? head : innodbFold(b, [[0, 26]])) : 0xdeadbeef;
    w32be(b, IN_SIZE - 8, tail);
    w32be(b, IN_SIZE - 4, lsnLo);
  }

  const f = (o: number, len: number, name: string, role: Role, covered: boolean, dec: (x: Uint8Array) => string, why: string): Field => ({
    kind: 'field',
    off: o,
    len,
    name,
    role,
    covered,
    dec,
    why,
  });

  const filHeader: Item[] = [
    f(0, 4, 'FIL_PAGE_SPACE_OR_CHKSUM', 'checksum', false, (x) => hex32(u32be(x, 0)),
      version >= 2
        ? 'Under full_crc32 this is no longer a checksum — it is ordinary page content, covered by the single CRC in the trailer.'
        : 'The page checksum. It is never covered by itself: the hash starts at byte 4.'),
    f(4, 4, 'FIL_PAGE_OFFSET', 'ident', true, (x) => `page ${u32be(x, 4)}`,
      'The page number this page believes it is. Compared against the page actually requested, which is how InnoDB catches a misdirected write that content-only checksums would pass.'),
    f(8, 4, 'FIL_PAGE_PREV', 'ident', true, (x) => (u32be(x, 8) === 0xffffffff ? 'FIL_NULL' : `page ${u32be(x, 8)}`),
      'Left sibling at this B-tree level — the doubly linked list that makes a leaf scan sequential.'),
    f(12, 4, 'FIL_PAGE_NEXT', 'ident', true, (x) => (u32be(x, 12) === 0xffffffff ? 'FIL_NULL' : `page ${u32be(x, 12)}`), 'Right sibling at this level.'),
    f(16, 8, 'FIL_PAGE_LSN', 'lsn', true, (x) => `${u32be(x, 16)}:${u32be(x, 20)}`,
      'LSN of the last redo record applied to this page. Its low 32 bits are repeated in the trailer — head and tail land in different sectors, so a tear splits them.'),
    f(24, 2, 'FIL_PAGE_TYPE', 'ident', true, (x) => (u16be(x, 24) === 17855 ? 'FIL_PAGE_INDEX (0x45BF)' : hex16(u16be(x, 24))),
      'Page type. FIL_PAGE_INDEX, FIL_PAGE_UNDO_LOG, FIL_PAGE_TYPE_BLOB and so on — checked on read, so a garbage value is caught even before the structure is walked.'),
    f(26, 8, 'FIL_PAGE_FILE_FLUSH_LSN', 'lsn', false, (x) => `${u32be(x, 26)}:${u32be(x, 30)}`,
      'Meaningful only on page 0 of the system tablespace. Excluded from the checksum — which means a bit flip in these 8 bytes verifies clean on every other page in the file.'),
    f(34, 4, 'FIL_PAGE_SPACE_ID', 'ident', false, (x) => `space ${u32be(x, 34)}`,
      'The tablespace this page belongs to. Also outside the checksummed ranges, so InnoDB checks it explicitly against the space it was read from.'),
  ];

  const pageHeader: Item[] = [
    f(38, 2, 'PAGE_N_DIR_SLOTS', 'slot', true, (x) => `${u16be(x, 38)}`, 'Entries in the page directory at the end of the page — the sparse index that makes an intra-page search binary rather than linear.'),
    f(40, 2, 'PAGE_HEAP_TOP', 'bounds', true, (x) => `${u16be(x, 40)}`, 'End of the used record heap: the InnoDB equivalent of pd_lower/pd_upper.'),
    f(42, 2, 'PAGE_N_HEAP', 'flags', true, (x) => `${u16be(x, 42) & 0x7fff} records${u16be(x, 42) & 0x8000 ? ' | COMPACT format' : ' | REDUNDANT format'}`,
      'Record count in the low 15 bits; the top bit records whether rows on this page use the COMPACT-family layout. Row format is stamped into the page, not just the dictionary.'),
    { kind: 'gap', from: 44, to: 54, name: 'PAGE_FREE, PAGE_GARBAGE, PAGE_LAST_INSERT, PAGE_DIRECTION…', role: 'flags', covered: true, flipOff: 46, why: 'Free-list head, reclaimable byte count and the insert-direction heuristic that decides whether a split goes 50/50 or 100/0.' },
    f(54, 2, 'PAGE_N_RECS', 'flags', true, (x) => `${u16be(x, 54)}`, 'User records on the page, excluding the infimum and supremum sentinels.'),
    { kind: 'gap', from: 56, to: 64, name: 'PAGE_MAX_TRX_ID', role: 'flags', covered: true, flipOff: 58, why: 'The largest transaction id that has modified this page — used by secondary-index MVCC to decide whether the index alone can answer a read.' },
    f(64, 2, 'PAGE_LEVEL', 'ident', true, (x) => (u16be(x, 64) === 0 ? '0 — leaf' : `${u16be(x, 64)}`), 'Height above the leaves. Zero means this page holds rows.'),
  ];

  const trailer: Item[] =
    version >= 2
      ? [
          f(IN_SIZE - 4, 4, 'CRC-32C (full page)', 'checksum', false, (x) => hex32(u32be(x, IN_SIZE - 4)),
            'full_crc32 stores one CRC-32C over every preceding byte of the page. No excluded ranges, no second copy, no LSN duplication — the trailer shrinks from 8 bytes to 4.'),
        ]
      : [
          f(IN_SIZE - 8, 4, 'trailer checksum', 'checksum', false, (x) => hex32(u32be(x, IN_SIZE - 8)),
            version === 1
              ? 'Under crc32 this repeats the header value, so the two ends of the page carry the same CRC.'
              : 'Under the legacy algorithm this covers only the first 26 bytes of the page — almost nothing. Its real job was always the four bytes beside it.'),
          f(IN_SIZE - 4, 4, 'FIL_PAGE_LSN (low 32 bits)', 'lsn', false, (x) => `${u32be(x, IN_SIZE - 4)}`,
            'The low half of the LSN from byte 16, repeated at the far end of the page. If the head and the tail disagree, the two halves came from different writes: that is a torn page, detected without any checksum at all.'),
        ];

  return {
    bytes: b,
    regions: [
      { label: 'FIL header — 38 bytes', sub: 'big-endian, identical on every page type in every tablespace', items: filHeader },
      { label: 'Index page header (PAGE_HEADER, from byte 38)', sub: 'the slotted-page bookkeeping for a B-tree leaf', items: pageHeader },
      {
        label: 'Records, free space and the page directory',
        sub: `bytes 74 → ${IN_SIZE - (version >= 2 ? 4 : 8)}`,
        items: [
          {
            kind: 'gap',
            from: 74,
            to: IN_SIZE - (version >= 2 ? 4 : 8),
            name: `${fmtNum(IN_SIZE - 74 - (version >= 2 ? 4 : 8))} bytes of records`,
            role: 'free',
            covered: true,
            flipOff: 4096,
            why: 'Rows, free space and the page directory growing back from the trailer. Click to flip a bit inside a record.',
          },
        ],
      },
      { label: version >= 2 ? 'Page trailer — 4 bytes' : 'Page trailer — 8 bytes', sub: 'the last bytes of the page, and therefore the last sector written', items: trailer },
    ],
    size: IN_SIZE,
    covered: version >= 2 ? IN_SIZE - 4 : 22 + (IN_SIZE - 46),
    lsnText: `${lsnHi}:${lsnLo}`,
    blockText: `space 4, page ${writtenFor}`,
  };
}

/* ----------------------------------------------------------- SQLite 4 KB */

const SQ_SIZE = 4096;
const SQ_BTREE = 100; // page 1 puts its b-tree header after the 100-byte file header

function buildSqlite(version: number, checksums: boolean, gen: 'old' | 'new'): Page {
  const b = new Uint8Array(SQ_SIZE);
  const reserved = version >= 2 && checksums ? 8 : 0;
  const usable = SQ_SIZE - reserved;

  const magic = 'SQLite format 3\0';
  for (let i = 0; i < 16; i++) b[i] = magic.charCodeAt(i);
  w16be(b, 16, SQ_SIZE);
  b[18] = version >= 1 ? 2 : 1; // write format
  b[19] = version >= 1 ? 2 : 1; // read format
  b[20] = reserved;
  b[21] = 64;
  b[22] = 32;
  b[23] = 32;
  w32be(b, 24, gen === 'new' ? 19 : 18); // file change counter
  w32be(b, 28, 64); // database size in pages
  w32be(b, 40, 7); // schema cookie
  w32be(b, 44, 4); // schema format
  w32be(b, 56, 1); // text encoding — UTF-8
  w32be(b, 92, gen === 'new' ? 19 : 18); // version-valid-for
  w32be(b, 96, 3046000);

  const cells = [usable - 56, usable - 116, usable - 190];
  b[SQ_BTREE] = 0x0d; // leaf table b-tree
  w16be(b, SQ_BTREE + 1, 0); // first freeblock
  w16be(b, SQ_BTREE + 3, cells.length);
  w16be(b, SQ_BTREE + 5, cells[cells.length - 1]); // cell content start
  b[SQ_BTREE + 7] = 0; // fragmented free bytes
  cells.forEach((c, i) => w16be(b, SQ_BTREE + 8 + 2 * i, c));

  fill(b, cells[cells.length - 1], usable, gen === 'new' ? 90210 : 90211);

  if (reserved) {
    // The checksum VFS puts an 8-byte checksum in the reserved bytes at the end of every page.
    const c1 = crc32cRanges(b, [[0, usable]]);
    w32be(b, usable, c1);
    w32be(b, usable + 4, (c1 ^ 0x5bd1e995) >>> 0);
  }

  const f = (o: number, len: number, name: string, role: Role, dec: (x: Uint8Array) => string, why: string): Field => ({
    kind: 'field',
    off: o,
    len,
    name,
    role,
    covered: reserved > 0,
    dec,
    why,
  });

  return {
    bytes: b,
    regions: [
      {
        label: 'Database file header — 100 bytes, page 1 only',
        sub: 'big-endian by specification; unchanged in every release since SQLite 3.0.0 in 2004',
        items: [
          f(0, 16, 'header string', 'ident', () => '"SQLite format 3\\0"', 'The magic. It has not changed in twenty years, and it is the reason a file written today opens in a 2005 binary.'),
          f(16, 2, 'page size', 'ident', (x) => `${u16be(x, 16) === 1 ? 65536 : u16be(x, 16)} bytes`, 'A power of two from 512 up. The value 1 means 65536 — a hack added in 3.7.1 precisely because the field is two bytes and the format could not grow one.'),
          f(18, 1, 'write format', 'ident', (x) => (x[18] === 2 ? '2 — WAL' : `${x[18]} — rollback journal`), 'Bumped to 2 when WAL mode arrived in 3.7.0. An older binary sees the 2, does not recognise it, and refuses to write rather than corrupting the file.'),
          f(19, 1, 'read format', 'ident', (x) => (x[19] === 2 ? '2 — WAL' : `${x[19]} — rollback journal`), 'Same idea for readers. Two bytes are the entire forward-compatibility mechanism of the format.'),
          f(20, 1, 'reserved bytes/page', 'checksum', (x) => `${x[20]}`, 'Bytes at the end of every page reserved for extensions. The checksum VFS sets this to 8 and stores its checksum there; a build without the VFS still reads the file, it just ignores those bytes.'),
          { kind: 'gap', from: 21, to: 24, name: 'payload fractions', role: 'flags', covered: reserved > 0, flipOff: 21, why: 'Must be 64/32/32. Any other value makes the file unreadable — three constants that were never allowed to vary.' },
          f(24, 4, 'change counter', 'lsn', (x) => `${u32be(x, 24)}`, 'Incremented on every write transaction. This, not a checksum, is how SQLite decides whether a cached page is stale.'),
          f(28, 4, 'database size (pages)', 'bounds', (x) => `${u32be(x, 28)}`, 'Valid only when it matches the version-valid-for number at byte 92 — an in-band way to tell whether an older writer touched the file.'),
          { kind: 'gap', from: 32, to: 92, name: 'freelist, schema cookie, text encoding, user_version…', role: 'flags', covered: reserved > 0, flipOff: 56, why: 'The rest of the header. New fields were only ever taken from bytes that were previously required to be zero.' },
          f(92, 4, 'version-valid-for', 'lsn', (x) => `${u32be(x, 92)}`, 'The change-counter value the size field above was written at.'),
          f(96, 4, 'SQLITE_VERSION_NUMBER', 'ident', (x) => `${u32be(x, 96)}`, 'The library version that last wrote the file. Informational: nothing refuses a file because of it.'),
        ],
      },
      {
        label: 'B-tree page header + cell pointer array',
        sub: 'the slotted page, at byte 100 on page 1 and at byte 0 on every other page',
        items: [
          f(SQ_BTREE, 1, 'page type', 'ident', (x) => (x[SQ_BTREE] === 0x0d ? '0x0D — leaf table' : hex16(x[SQ_BTREE])), 'One of 0x02, 0x05, 0x0A, 0x0D. Anything else is "database disk image is malformed" the moment the page is walked.'),
          f(SQ_BTREE + 1, 2, 'first freeblock', 'bounds', (x) => `${u16be(x, SQ_BTREE + 1)}`, 'Head of the intra-page free list; zero when the free space is all contiguous.'),
          f(SQ_BTREE + 3, 2, 'cell count', 'slot', (x) => `${u16be(x, SQ_BTREE + 3)}`, 'Number of cell pointers that follow. Corrupt this and the b-tree layer walks off the end of the array — one of the few things SQLite does check.'),
          f(SQ_BTREE + 5, 2, 'cell content start', 'bounds', (x) => `${u16be(x, SQ_BTREE + 5)}`, 'Low-water mark of the cell content area. Must lie between the end of the pointer array and the usable page size.'),
          f(SQ_BTREE + 7, 1, 'fragmented free bytes', 'flags', (x) => `${x[SQ_BTREE + 7]}`, 'Bytes lost to fragments too small for the free list; 60 or more triggers a defragment on the next insert.'),
          ...cells.map((_, i) =>
            f(SQ_BTREE + 8 + 2 * i, 2, `cell[${i}]`, 'slot', (x) => `offset ${u16be(x, SQ_BTREE + 8 + 2 * i)}`,
              'A 2-byte offset to the cell. Sorted by key, while the cells themselves sit wherever they fit — the slotted page in its smallest possible form.'),
          ),
        ],
      },
      {
        label: 'Cell content',
        sub: `${cells[cells.length - 1]} → ${usable}`,
        items: [
          {
            kind: 'gap',
            from: cells[cells.length - 1],
            to: usable,
            name: `${fmtNum(usable - cells[cells.length - 1])} bytes of records`,
            role: 'tuple',
            covered: reserved > 0,
            flipOff: usable - 40,
            why: 'Varint header plus column bodies, grown down from the end of the page. Click to corrupt a record — with no checksum, nothing here is ever verified.',
          },
        ],
      },
      ...(reserved
        ? [
            {
              label: 'Reserved bytes — checksum VFS',
              sub: 'the 8 bytes the header at byte 20 reserved on every page',
              items: [
                f(usable, 8, 'page checksum', 'checksum', (x) => `${hex32(u32be(x, usable))} ${hex32(u32be(x, usable + 4))}`,
                  'Written by cksumvfs over the first pagesize−8 bytes on the way out, verified on the way in. A mismatch fails the read with SQLITE_IOERR_DATA. Not part of the file format: it is a VFS shim living in space the format set aside.'),
              ],
            } as Region,
          ]
        : []),
    ],
    size: SQ_SIZE,
    covered: reserved ? usable : 0,
    lsnText: `change counter ${gen === 'new' ? 19 : 18}`,
    blockText: 'page 1',
  };
}

/* -------------------------------------------------------------- damage */

type Damage = 'none' | 'torn' | 'misdirect' | 'lost';

const DAMAGE_LABEL: Record<Damage, string> = {
  none: 'intact',
  torn: 'torn write',
  misdirect: 'misdirected write',
  lost: 'lost write',
};

const READ_BLOCK = { pg: 40, innodb: 40 } as const;

function build(engine: EngineId, version: number, checksums: boolean, damage: Damage): Page {
  const mk = (gen: 'old' | 'new', writtenFor: number) =>
    engine === 'pg'
      ? buildPg(version, checksums, gen, writtenFor)
      : engine === 'innodb'
        ? buildInnodb(version, checksums, gen, writtenFor)
        : buildSqlite(version, checksums, gen);

  const home = engine === 'sqlite' ? 1 : READ_BLOCK[engine];

  if (damage === 'lost') return mk('old', home);
  if (damage === 'misdirect') {
    // The page was correctly written *for* another block and landed here.
    const stray = mk('new', engine === 'sqlite' ? 1 : 12);
    return stray;
  }
  const fresh = mk('new', home);
  if (damage === 'torn') {
    const stale = mk('old', home);
    // The first half made it to media; the second half is the new image. Sectors, not pages.
    const half = fresh.size / 2;
    const torn = fresh.bytes.slice();
    torn.set(stale.bytes.subarray(0, half), 0);
    return { ...fresh, bytes: torn, lsnText: `${stale.lsnText} / ${fresh.lsnText}` };
  }
  return fresh;
}

/* ---------------------------------------------------------- verification */

type CheckRow = { name: string; pass: boolean | null; detail: string };
type ReadResult = { ok: boolean; checks: CheckRow[]; head: string; log: string[] };

function verifyPg(b: Uint8Array, version: number, checksums: boolean, blkno: number): ReadResult {
  const checks: CheckRow[] = [];
  const allZero = b.every((v) => v === 0);
  if (allZero)
    return {
      ok: true,
      checks: [{ name: 'PageIsNew', pass: true, detail: 'The page is all zeros, which is a legal empty page. A lost write that leaves a hole reads as "new", not as an error.' }],
      head: 'Read succeeded — the page is empty.',
      log: ['(no message: an all-zero page is valid)'],
    };

  const stored = u16le(b, 8);
  const calc = pgChecksum(b, blkno);
  if (version >= 2 && checksums)
    checks.push({
      name: 'pd_checksum',
      pass: stored === calc,
      detail: `stored ${stored}, recomputed ${calc} over all 8192 bytes with pd_checksum zeroed and the block number mixed in`,
    });
  else
    checks.push({
      name: 'pd_checksum',
      pass: null,
      detail: version >= 2 ? 'data_checksums = off — the field is not written and not compared' : 'this layout predates page checksums entirely',
    });

  const lower = u16le(b, 12);
  const upper = u16le(b, 14);
  const special = u16le(b, 16);
  const psv = u16le(b, 18);
  const flags = u16le(b, 10);
  const hdrLen = version >= 1 ? 24 : 20;
  // PageIsVerifiedExtended's header_sane test, and only that: no page-size or
  // layout-version comparison happens here, and pd_lower is not compared to the
  // header size either.
  const sane =
    (flags & ~0x0007) === 0 && lower <= upper && upper <= special && special <= PG_SIZE && special % 8 === 0;
  checks.push({
    name: 'PageIsVerified header checks',
    pass: sane,
    detail: `pd_flags ${hex16(flags)} has only defined bits, pd_lower ${lower} ≤ pd_upper ${upper} ≤ pd_special ${special} ≤ 8192, pd_special is MAXALIGNed (pd_pagesize_version ${hex16(psv)} is not part of this test; header is ${hdrLen} bytes)`,
  });

  const ckFail = checks[0].pass === false;
  const hdrFail = !sane;
  if (!ckFail && !hdrFail)
    return { ok: true, checks, head: 'Read succeeded.', log: ['(no message: the page verified and went into shared_buffers)'] };

  const log: string[] = [];
  if (ckFail) log.push(`WARNING:  page verification failed, calculated checksum ${calc} but expected ${stored}`);
  log.push('ERROR:  invalid page in block 40 of relation base/16384/16427');
  return {
    ok: false,
    checks,
    head: ckFail ? 'Checksum mismatch — the read is aborted.' : 'The header is self-inconsistent — the read is aborted.',
    log,
  };
}

function verifyInnodb(b: Uint8Array, version: number, checksums: boolean, blkno: number): ReadResult {
  const checks: CheckRow[] = [];
  const ranges = innodbRanges(version);

  if (!checksums) {
    checks.push({ name: 'page checksum', pass: null, detail: 'checksum verification disabled — InnoDB writes BUF_NO_CHECKSUM_MAGIC and compares nothing' });
  } else if (version >= 2) {
    const stored = u32be(b, IN_SIZE - 4);
    const calc = crc32cRanges(b, ranges);
    checks.push({ name: 'CRC-32C (full page)', pass: stored === calc, detail: `stored ${hex32(stored)}, recomputed ${hex32(calc)} over bytes 0…${IN_SIZE - 5}` });
  } else {
    const stored = u32be(b, 0);
    const calc = version === 1 ? crc32cRanges(b, ranges) : innodbFold(b, ranges);
    checks.push({
      name: version === 1 ? 'FIL header CRC-32C' : 'FIL header checksum (legacy fold)',
      pass: stored === calc,
      detail: `stored ${hex32(stored)}, recomputed ${hex32(calc)} over bytes 4…25 and 38…${IN_SIZE - 9} — 24 bytes of the page are outside both ranges`,
    });
    const tStored = u32be(b, IN_SIZE - 8);
    const tCalc = version === 1 ? calc : innodbFold(b, [[0, 26]]);
    checks.push({
      name: 'trailer checksum',
      pass: tStored === tCalc,
      detail: version === 1 ? `stored ${hex32(tStored)}, expected the same value as the header` : `stored ${hex32(tStored)}, recomputed ${hex32(tCalc)} over the first 26 bytes only`,
    });
  }

  if (version < 2) {
    const head = u32be(b, 20);
    const tail = u32be(b, IN_SIZE - 4);
    checks.push({
      name: 'head/tail LSN',
      pass: head === tail,
      detail: `FIL_PAGE_LSN low half ${head} at byte 20, ${tail} at byte ${IN_SIZE - 4}`,
    });
  } else {
    checks.push({ name: 'head/tail LSN', pass: null, detail: 'full_crc32 drops the duplicated LSN: one CRC over the whole page already catches a tear' });
  }

  const off = u32be(b, 4);
  checks.push({ name: 'FIL_PAGE_OFFSET', pass: off === blkno, detail: `page says ${off}, the buffer pool asked for ${blkno}` });
  const type = u16be(b, 24);
  checks.push({ name: 'FIL_PAGE_TYPE', pass: type === 17855, detail: `${hex16(type)}${type === 17855 ? ' — FIL_PAGE_INDEX' : ' — not a recognised page type'}` });

  const bad = checks.find((c) => c.pass === false);
  if (!bad) return { ok: true, checks, head: 'Read succeeded.', log: ['(no message: the page went into the buffer pool)'] };
  return {
    ok: false,
    checks,
    head: `${bad.name} failed — InnoDB refuses the page.`,
    log: [
      `InnoDB: Database page corruption on disk or a failed file read of page [page id: space=4, page number=${blkno}].`,
      'InnoDB: You may have to recover from a backup.',
      'InnoDB: Ending processing because of a corrupt database page.',
    ],
  };
}

function verifySqlite(b: Uint8Array, version: number, checksums: boolean): ReadResult {
  const checks: CheckRow[] = [];
  const reserved = b[20];
  const usable = SQ_SIZE - reserved;

  if (version >= 2 && checksums && reserved === 8) {
    const stored = u32be(b, usable);
    const calc = crc32cRanges(b, [[0, usable]]);
    checks.push({ name: 'checksum VFS', pass: stored === calc, detail: `stored ${hex32(stored)}, recomputed ${hex32(calc)} over the first ${usable} bytes` });
  } else {
    checks.push({ name: 'page checksum', pass: null, detail: 'SQLite stores no page checksum. Nothing in the file format verifies these bytes, ever.' });
  }

  const type = b[SQ_BTREE];
  const n = u16be(b, SQ_BTREE + 3);
  const start = u16be(b, SQ_BTREE + 5);
  const arrayEnd = SQ_BTREE + 8 + 2 * n;
  let structOk = (type === 0x02 || type === 0x05 || type === 0x0a || type === 0x0d) && start >= arrayEnd && start <= usable;
  for (let i = 0; i < n && structOk; i++) {
    const p = u16be(b, SQ_BTREE + 8 + 2 * i);
    if (p < arrayEnd || p >= usable) structOk = false;
  }
  checks.push({
    name: 'b-tree structural checks',
    pass: structOk,
    detail: `type ${hex16(type)}, ${n} cells, content starts at ${start}, pointer array ends at ${arrayEnd}, usable size ${usable}`,
  });

  const ckFail = checks[0].pass === false;
  if (!ckFail && structOk)
    return {
      ok: true,
      checks,
      head: 'Read succeeded.',
      log: [reserved === 8 ? '(no message: the checksum matched)' : '(no message: nothing was checked)'],
    };
  if (ckFail) return { ok: false, checks, head: 'The checksum VFS rejected the page.', log: ['SQLITE_IOERR_DATA — sqlite3_step() returns "disk I/O error"'] };
  return {
    ok: false,
    checks,
    head: 'The b-tree layer walked into nonsense.',
    log: ['Error: database disk image is malformed (11)', 'PRAGMA integrity_check names the page; nothing named the byte.'],
  };
}

/* ------------------------------------------------------------- component */

const ENGINES: { value: EngineId; label: string }[] = [
  { value: 'pg', label: 'PostgreSQL — 8 KB heap page' },
  { value: 'innodb', label: 'InnoDB — 16 KB index page' },
  { value: 'sqlite', label: 'SQLite — 4 KB b-tree page' },
];

const CK_LABEL: Record<EngineId, string> = {
  pg: 'data_checksums',
  innodb: 'verify checksum on read',
  sqlite: 'checksum VFS',
};

export default function PageIntegrityInspector() {
  const [engine, setEngine] = useState<EngineId>('pg');
  const [version, setVersion] = useState(2);
  const [checksums, setChecksums] = useState(true);
  const [bit, setBit] = useState(0);
  const [damage, setDamage] = useState<Damage>('none');
  const [flips, setFlips] = useState<Record<number, number>>({});
  const [result, setResult] = useState<ReadResult | null>(null);
  const [msg, setMsg] = useState<{ head: string; body: string }>({
    head: 'Nothing has read this page.',
    body:
      'Hover any byte to decode the field it belongs to. Click one to flip a bit — the page on disk is now wrong and no software anywhere knows it. Then press “Read page” and watch which checks, if any, notice.',
  });
  const tip = useTip();

  const ckSupported = engine === 'sqlite' ? version >= 2 : engine === 'pg' ? version >= 2 : true;
  const ckOn = checksums && ckSupported;

  const page = useMemo(() => {
    const p = build(engine, version, ckOn, damage);
    const bytes = p.bytes.slice();
    for (const [k, v] of Object.entries(flips)) bytes[Number(k)] ^= v;
    return { ...p, bytes };
  }, [engine, version, ckOn, damage, flips]);

  const blkno = engine === 'sqlite' ? 1 : READ_BLOCK[engine];

  const reset = (next?: Partial<{ engine: EngineId; version: number }>) => {
    setFlips({});
    setDamage('none');
    setResult(null);
    if (next?.engine) setEngine(next.engine);
    if (next?.version !== undefined) setVersion(next.version);
  };

  const findItem = (off: number): { region: Region; item: Item } | null => {
    for (const r of page.regions)
      for (const it of r.items) {
        if (it.kind === 'field' && off >= it.off && off < it.off + it.len) return { region: r, item: it };
        if (it.kind === 'gap' && off >= it.from && off < it.to) return { region: r, item: it };
      }
    return null;
  };

  const flip = (off: number) => {
    const mask = 1 << bit;
    setFlips((cur) => {
      const nextMask = (cur[off] ?? 0) ^ mask;
      const out = { ...cur };
      if (nextMask === 0) delete out[off];
      else out[off] = nextMask;
      return out;
    });
    setResult(null);
    const hit = findItem(off);
    const name = hit ? (hit.item.kind === 'field' ? hit.item.name : hit.item.name) : 'the page';
    const covered = hit ? hit.item.covered : true;
    setMsg({
      head: `Bit ${bit} of byte ${off} flipped — inside ${name}.`,
      body: covered
        ? 'These bytes are inside the checksummed region. Nothing has happened yet: the corruption sits on disk, invisible, until something reads this page. Press “Read page”.'
        : 'These bytes are outside every checksummed range, which means the page will still verify perfectly. Press “Read page” and see what does — and does not — catch it.',
    });
  };

  const doDamage = (d: Damage) => {
    setDamage(d);
    setFlips({});
    setResult(null);
    const bodies: Record<Damage, string> = {
      none: 'Back to a clean page.',
      torn: `The first half of the page is the previous image and the second half is the new one — the device wrote some 4 KB sectors and not others. The header, including whatever integrity fields live there, is old; the payload is new.`,
      misdirect: `The engine wrote a perfectly formed page and the firmware put it at the wrong address. Every byte inside the page is internally consistent; it is simply not the page that belongs here.`,
      lost: `The most recent write never reached the media — a thin-provisioned LUN, a lying cache, a reverted snapshot. What is on disk is the previous version of this page: complete, self-consistent and stale.`,
    };
    setMsg({ head: `Applied: ${DAMAGE_LABEL[d]}.`, body: bodies[d] });
  };

  const read = () => {
    const r =
      engine === 'pg'
        ? verifyPg(page.bytes, version, ckOn, blkno)
        : engine === 'innodb'
          ? verifyInnodb(page.bytes, version, ckOn, blkno)
          : verifySqlite(page.bytes, version, ckOn);
    setResult(r);
    const stale = damage === 'lost' && r.ok;
    setMsg({
      head: stale ? 'Read succeeded — and the page is the wrong version.' : r.head,
      body: stale
        ? 'Every check passed, because every check asks "are these bytes the bytes that were hashed?" and they are. A checksum is a statement about internal consistency, never about recency. Catching this needs an external record of what the page should be: the LSN the log says this page reached, Oracle’s DB_LOST_WRITE_PROTECT comparing read SCNs on a standby, or ZFS keeping the checksum and birth transaction in the parent block pointer.'
        : r.ok
          ? 'Nothing was rejected. Note what that does and does not prove: the bytes on disk are the bytes that were hashed, at the address they were hashed for — nothing about whether they are the newest version.'
          : 'The read is refused. Detection is where a checksum stops: it names the page, not the byte, and it cannot rebuild anything. Recovery comes from full-page writes, the doublewrite buffer or a backup.',
    });
  };

  const vlist = VERSIONS[engine];
  const flipped = Object.keys(flips).length;

  const legendItems = (Object.keys(ROLE_COLOR) as Role[]).map((r) => ({ label: ROLE_LABEL[r], color: ROLE_COLOR[r] }));

  const allFields: { region: string; item: Item }[] = page.regions.flatMap((r) => r.items.map((item) => ({ region: r.label, item })));

  return (
    <VizPanel
      title="Page inspector: what a page says about itself, and who checks it"
      subtitle="Real header layouts, decoded byte by byte. Flip a bit, tear the write, misdirect it or lose it, then read the page and watch which verification step notices — and which engine, at which format version, has one at all."
      controls={
        <>
          <Choice
            label="Engine"
            value={engine}
            onChange={(v) => {
              reset({ engine: v, version: v === 'sqlite' ? 1 : 2 });
            }}
            options={ENGINES}
          />
          <Slider
            label="On-disk format"
            min={0}
            max={2}
            value={version}
            onChange={(n) => reset({ version: n })}
            format={(n) => vlist[n].label}
          />
          <Check
            label={ckSupported ? CK_LABEL[engine] : `${CK_LABEL[engine]} — needs ${vlist[2].short}`}
            checked={ckOn}
            onChange={(v) => {
              setChecksums(v);
              setResult(null);
              if (v && !ckSupported) reset({ version: 2 });
            }}
          />
          <Slider label="Bit to flip" min={0} max={7} value={bit} onChange={setBit} format={(n) => `bit ${n}`} />
          <Button onClick={() => doDamage('torn')} title="First half old, second half new — sectors, not pages">
            Tear the write
          </Button>
          <Button onClick={() => doDamage('misdirect')} title="A valid page written to the wrong address">
            Misdirect the write
          </Button>
          <Button onClick={() => doDamage('lost')} title="The newest write never reached the media">
            Lose the write
          </Button>
          <Button onClick={read} primary>
            Read page
          </Button>
          <Button onClick={() => { reset(); setMsg({ head: 'Clean page.', body: 'A freshly written page, checksummed on the way out.' }); }}>Reset</Button>
        </>
      }
      legend={<Legend items={legendItems} />}
      stats={
        <Stats
          items={[
            { label: 'Page', value: `${fmtNum(page.size)} B`, hint: page.blockText },
            {
              label: 'Bytes the checksum covers',
              value: ckOn ? `${fmtNum(page.covered)} / ${fmtNum(page.size)}` : `0 / ${fmtNum(page.size)}`,
              hint: ckOn ? 'Everything outside this is invisible to verification' : 'Nothing is hashed at this setting',
            },
            { label: 'Page version', value: page.lsnText, hint: 'The LSN or counter that says which write this page is from' },
            { label: 'State on disk', value: DAMAGE_LABEL[damage] + (flipped ? ` + ${flipped} flipped byte${flipped === 1 ? '' : 's'}` : ''), hint: 'What is physically in the file right now' },
            {
              label: 'Verdict',
              value: result ? (result.ok ? 'read OK' : 'read refused') : 'not read',
              hint: result ? undefined : 'Corruption on disk is silent until something reads the page — which is why scrubs exist',
            },
          ]}
        />
      }
      note={
        <Note>
          <strong>{msg.head}</strong> {msg.body}
          {result ? (
            <>
              <br />
              <span style={{ fontVariantNumeric: 'tabular-nums', color: result.ok ? 'var(--viz-ink-2)' : 'var(--viz-critical)' }}>
                {result.log.join(' ')}
              </span>
            </>
          ) : null}
        </Note>
      }
      table={
        <>
          <table className="viz-table">
            <thead>
              <tr>
                <th>Region</th>
                <th>Offset</th>
                <th>Bytes</th>
                <th>Field</th>
                <th>Value</th>
                <th>Hashed?</th>
              </tr>
            </thead>
            <tbody>
              {allFields.map(({ region, item }) => (
                <tr key={`${region}-${item.kind === 'field' ? item.off : item.from}-${item.name}`}>
                  <td>{region}</td>
                  <td>{item.kind === 'field' ? item.off : `${item.from}…${item.to - 1}`}</td>
                  <td>{item.kind === 'field' ? item.len : item.to - item.from}</td>
                  <td>{item.name}</td>
                  <td>{item.kind === 'field' ? item.dec(page.bytes) : '—'}</td>
                  <td>{ckOn && item.covered ? 'yes' : 'no'}</td>
                </tr>
              ))}
            </tbody>
          </table>
          <table className="viz-table">
            <thead>
              <tr>
                <th>Check performed on read</th>
                <th>Result</th>
                <th>Detail</th>
              </tr>
            </thead>
            <tbody>
              {result === null ? (
                <tr>
                  <td colSpan={3}>The page has not been read.</td>
                </tr>
              ) : (
                result.checks.map((c) => (
                  <tr key={c.name}>
                    <td>{c.name}</td>
                    <td>{c.pass === null ? 'not performed' : c.pass ? 'pass' : 'FAIL'}</td>
                    <td>{c.detail}</td>
                  </tr>
                ))
              )}
            </tbody>
          </table>
        </>
      }
    >
      <TooltipHost>
        <div style={{ display: 'flex', flexDirection: 'column', gap: '.75rem', minWidth: '320px' }}>
          {page.regions.map((r) => (
            <div key={r.label}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'baseline', gap: '.75rem' }}>
                <strong style={{ color: 'var(--viz-ink)', fontSize: '.8125rem' }}>{r.label}</strong>
                <span style={{ color: 'var(--viz-ink-2)', fontSize: '.6875rem', textAlign: 'right' }}>{r.sub}</span>
              </div>
              <div style={{ display: 'flex', flexWrap: 'wrap', gap: '.45rem .6rem', marginTop: '.3rem' }}>
                {r.items.map((item) =>
                  item.kind === 'field' ? (
                    <div key={`${item.off}-${item.name}`}>
                      <div style={{ fontSize: '.625rem', color: 'var(--viz-ink-2)', marginBottom: '2px', whiteSpace: 'nowrap' }}>
                        {item.name}
                      </div>
                      <div style={{ display: 'flex', gap: '2px', flexWrap: 'wrap' }}>
                        {Array.from({ length: item.len }, (_, k) => {
                          const off = item.off + k;
                          const mutated = flips[off] !== undefined;
                          return (
                            <span
                              key={off}
                              {...tip(
                                <>
                                  <strong>
                                    {item.name} — byte {off}, 0x{hx(page.bytes[off])}
                                  </strong>
                                  <br />
                                  {item.dec(page.bytes)}
                                  <br />
                                  <span style={{ color: 'var(--viz-ink-2)' }}>{item.why}</span>
                                  <br />
                                  <span style={{ color: 'var(--viz-ink-2)' }}>
                                    {ckOn && item.covered ? 'inside the checksummed range' : 'not covered by any checksum'} · click to flip bit {bit}
                                  </span>
                                </>,
                              )}
                              role="button"
                              aria-label={`byte ${off}, ${item.name}, flip bit ${bit}`}
                              onClick={() => flip(off)}
                              onKeyDown={(e) => {
                                if (e.key === 'Enter' || e.key === ' ') {
                                  e.preventDefault();
                                  flip(off);
                                }
                              }}
                              style={{
                                display: 'inline-block',
                                minWidth: '1.85rem',
                                textAlign: 'center',
                                padding: '.12rem .15rem',
                                fontSize: '.6875rem',
                                lineHeight: 1.4,
                                color: 'var(--viz-ink)',
                                fontVariantNumeric: 'tabular-nums',
                                background: mutated
                                  ? 'color-mix(in srgb, var(--viz-critical) 32%, var(--viz-plane))'
                                  : `color-mix(in srgb, ${ROLE_COLOR[item.role]} 16%, var(--viz-plane))`,
                                border: '1px solid var(--viz-border)',
                                borderBottom: `3px solid ${mutated ? 'var(--viz-critical)' : ROLE_COLOR[item.role]}`,
                                borderStyle: ckOn && item.covered ? 'solid' : 'dashed',
                                borderRadius: '4px',
                                cursor: 'pointer',
                              }}
                            >
                              {hx(page.bytes[off])}
                            </span>
                          );
                        })}
                      </div>
                    </div>
                  ) : (
                    <div key={`gap-${item.from}-${item.name}`} style={{ flex: '1 1 14rem', minWidth: '10rem' }}>
                      <div style={{ fontSize: '.625rem', color: 'var(--viz-ink-2)', marginBottom: '2px' }}>
                        bytes {item.from}…{item.to - 1}
                      </div>
                      <span
                        {...tip(
                          <>
                            <strong>{item.name}</strong>
                            <br />
                            <span style={{ color: 'var(--viz-ink-2)' }}>{item.why}</span>
                            <br />
                            <span style={{ color: 'var(--viz-ink-2)' }}>
                              {ckOn && item.covered ? 'inside the checksummed range' : 'not covered by any checksum'} · click to flip bit {bit} of byte{' '}
                              {item.flipOff}
                            </span>
                          </>,
                        )}
                        role="button"
                        aria-label={`${item.name}, flip bit ${bit} of byte ${item.flipOff}`}
                        onClick={() => flip(item.flipOff)}
                        onKeyDown={(e) => {
                          if (e.key === 'Enter' || e.key === ' ') {
                            e.preventDefault();
                            flip(item.flipOff);
                          }
                        }}
                        style={{
                          display: 'block',
                          padding: '.3rem .5rem',
                          fontSize: '.6875rem',
                          color: 'var(--viz-ink)',
                          background: `repeating-linear-gradient(135deg, color-mix(in srgb, ${ROLE_COLOR[item.role]} 14%, var(--viz-plane)) 0 6px, var(--viz-plane) 6px 12px)`,
                          border: '1px solid var(--viz-border)',
                          borderBottom: `3px solid ${ROLE_COLOR[item.role]}`,
                          borderStyle: ckOn && item.covered ? 'solid' : 'dashed',
                          borderRadius: '4px',
                          cursor: 'pointer',
                        }}
                      >
                        {item.name}
                        {Object.keys(flips).some((k) => Number(k) >= item.from && Number(k) < item.to) ? ' · corrupted' : ''}
                      </span>
                    </div>
                  ),
                )}
              </div>
            </div>
          ))}
        </div>
      </TooltipHost>
    </VizPanel>
  );
}
