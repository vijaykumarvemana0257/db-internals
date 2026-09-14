import { useMemo, useState } from 'react';
import {
  VizPanel,
  Slider,
  Segmented,
  Choice,
  Legend,
  Stats,
  TooltipHost,
  useTip,
  useSize,
} from './Viz';

/**
 * Type / encoding / collation switchboard.
 *
 * Everything modelled here is the real behaviour:
 *  - PostgreSQL text, varchar(n) and char(n) are one varlena storage path: a
 *    1-byte header for values under 127 bytes, a 4-byte header above that.
 *    char(n) blank-pads to n *characters*. uuid is 16 fixed bytes.
 *  - InnoDB puts a 1-byte length prefix on a variable-length column whose
 *    maximum byte length is <= 255, 2 bytes above that; CHAR(n) in a
 *    variable-length charset is stored with trailing spaces stripped but never
 *    below n bytes; BINARY pads with 0x00, not spaces.
 *  - A btree index tuple in Postgres is an 8-byte IndexTupleData header plus a
 *    MAXALIGNed key, and costs a 4-byte line pointer on the page. An InnoDB
 *    secondary index entry carries a 5-byte record header, the key, and the
 *    primary key appended as the row locator.
 *  - Collation is compared as multi-level weight keys: primary (base letter),
 *    secondary (accent), tertiary (case), then — for a *deterministic*
 *    Postgres collation only — a memcmp tiebreak. That last step is why NFC
 *    and NFD are never equal in Postgres unless the collation is declared
 *    deterministic = false.
 */

/* ------------------------------------------------------------------ model */

type Engine = 'pg' | 'mysql';

type Fixture = {
  id: string;
  label: string;
  short: string;
  cps: number[];
  graphemes: number;
  why: string;
};

const UUID_TEXT = '018f3a9c-6e2b-7c41-9d5e-2a7b4c8f1e03';

const FIXTURES: Fixture[] = [
  {
    id: 'Zoe',
    label: 'Zoe',
    short: 'Zoe',
    cps: [0x5a, 0x6f, 0x65],
    graphemes: 3,
    why: 'Plain ASCII with a capital. Differs from "zoe" only at the tertiary (case) level.',
  },
  {
    id: 'zoe',
    label: 'zoe',
    short: 'zoe',
    cps: [0x7a, 0x6f, 0x65],
    graphemes: 3,
    why: 'The probe value. Three bytes in every Unicode encoding on this page.',
  },
  {
    id: 'zoe_sp',
    label: 'zoe␣ (trailing space)',
    short: 'zoe␣',
    cps: [0x7a, 0x6f, 0x65, 0x20],
    graphemes: 4,
    why: 'A trailing space: significant in text/varchar, erased by char(n) blank padding, and ignored outright by a PAD SPACE collation.',
  },
  {
    id: 'zo_e',
    label: 'zo e (internal space)',
    short: 'zo e',
    cps: [0x7a, 0x6f, 0x20, 0x65],
    graphemes: 4,
    why: 'An internal space. glibc locales treat it as variable (ignored until a later level); ICU/CLDR root gives it a primary weight, so the two providers disagree about the order.',
  },
  {
    id: 'nfc',
    label: 'zoë (NFC, U+00EB)',
    short: 'zoë NFC',
    cps: [0x7a, 0x6f, 0xeb],
    graphemes: 3,
    why: 'Precomposed e-diaeresis: one code point, two UTF-8 bytes. UCA expands it to the same weights as e + U+0308.',
  },
  {
    id: 'nfd',
    label: 'zoë (NFD, e + U+0308)',
    short: 'zoë NFD',
    cps: [0x7a, 0x6f, 0x65, 0x308],
    graphemes: 3,
    why: 'Decomposed: the same grapheme, one more code point, one more byte, a different byte string. What macOS filenames and some iOS keyboards hand you.',
  },
  {
    id: 'burger',
    label: '🍔 (U+1F354)',
    short: '🍔',
    cps: [0x1f354],
    graphemes: 1,
    why: 'Outside the BMP: four UTF-8 bytes, one code point, and unstorable in MySQL utf8mb3.',
  },
  {
    id: 'family',
    label: '👩‍👧 (ZWJ sequence)',
    short: '👩‍👧',
    cps: [0x1f469, 0x200d, 0x1f467],
    graphemes: 1,
    why: 'One grapheme cluster, three code points, eleven bytes. The ZWJ is completely ignorable in UCA, so it contributes no weight at any level.',
  },
];

const UUID_FIXTURE: Fixture = {
  id: 'uuid',
  label: `${UUID_TEXT.slice(0, 13)}… (UUID)`,
  short: 'uuid',
  cps: Array.from(UUID_TEXT).map((c) => c.codePointAt(0)!),
  graphemes: 36,
  why: 'A canonical UUID string: 36 characters of text, or 16 bytes if you let the type system hold it.',
};

const RULER_VALUES = [...FIXTURES, UUID_FIXTURE];

/* ------------------------------------------------------------- encodings */

type Charset = {
  id: string;
  label: string;
  engine: Engine;
  maxLen: number;
  kind: 'utf8' | 'utf8bmp' | 'latin1' | 'rawbytes';
  note: string;
};

const CHARSETS: Charset[] = [
  {
    id: 'UTF8',
    label: 'UTF8 (server encoding)',
    engine: 'pg',
    maxLen: 4,
    kind: 'utf8',
    note: 'Validated on input; length() counts code points.',
  },
  {
    id: 'LATIN1',
    label: 'LATIN1',
    engine: 'pg',
    maxLen: 1,
    kind: 'latin1',
    note: 'One byte per character; anything above U+00FF is rejected at the conversion, not stored lossily.',
  },
  {
    id: 'SQL_ASCII',
    label: 'SQL_ASCII (no encoding)',
    engine: 'pg',
    maxLen: 1,
    kind: 'rawbytes',
    note: 'No validation and no conversion: bytes go in and come out unexamined, so length() degenerates to a byte count.',
  },
  {
    id: 'utf8mb4',
    label: 'utf8mb4',
    engine: 'mysql',
    maxLen: 4,
    kind: 'utf8',
    note: 'Real UTF-8. The MySQL 8.0 default, and the only one that holds an emoji.',
  },
  {
    id: 'utf8mb3',
    label: 'utf8mb3 (the old "utf8")',
    engine: 'mysql',
    maxLen: 3,
    kind: 'utf8bmp',
    note: 'Three bytes maximum: the BMP only. Everything above U+FFFF is rejected or truncated.',
  },
  {
    id: 'latin1',
    label: 'latin1',
    engine: 'mysql',
    maxLen: 1,
    kind: 'latin1',
    note: 'The pre-8.0 default. One byte per character, no Unicode above U+00FF.',
  },
];

function encode(cps: number[], cs: Charset): { bytes: number[]; error?: string } {
  const out: number[] = [];
  for (const cp of cps) {
    if (cs.kind === 'latin1') {
      if (cp > 0xff) {
        return {
          bytes: [],
          error:
            cs.engine === 'pg'
              ? `ERROR: character with byte sequence ${utf8Of([cp])
                  .map((b) => '0x' + b.toString(16))
                  .join(' ')} in encoding "UTF8" has no equivalent in encoding "LATIN1"`
              : `ERROR 1366 (HY000): Incorrect string value: '\\x${utf8Of([cp])
                  .map((b) => b.toString(16).toUpperCase())
                  .join('\\x')}' for column 'name' at row 1`,
        };
      }
      out.push(cp);
      continue;
    }
    if (cs.kind === 'utf8bmp' && cp > 0xffff) {
      return {
        bytes: [],
        error: `ERROR 1366 (HY000): Incorrect string value: '\\x${utf8Of([cp])
          .map((b) => b.toString(16).toUpperCase())
          .join('\\x')}' for column 'name' at row 1`,
      };
    }
    out.push(...utf8Of([cp]));
  }
  return { bytes: out };
}

const TE = new TextEncoder();
function utf8Of(cps: number[]): number[] {
  return Array.from(TE.encode(cps.map((c) => String.fromCodePoint(c)).join('')));
}

const hex = (n: number) => n.toString(16).toUpperCase().padStart(2, '0');

/* ------------------------------------------------------------ collations */

type Coll = {
  id: string;
  label: string;
  engine: Engine;
  raw?: boolean;
  strength: 1 | 2 | 3;
  variable: 'ignore' | 'nonignorable';
  pad: 'nopad' | 'padspace';
  deterministic: boolean;
  combiningIsPrimary?: boolean;
  note: string;
};

const BYTES_COLL: Coll = {
  id: 'memcmp',
  label: 'none (binary type)',
  engine: 'pg',
  raw: true,
  strength: 3,
  variable: 'nonignorable',
  pad: 'nopad',
  deterministic: true,
  note: 'A binary type has no encoding and no collation. Every comparison is memcmp on the raw bytes, ORDER BY is byte order, and the collation control above is inert.',
};

const COLLS: Coll[] = [
  {
    id: 'C',
    label: 'C / POSIX',
    engine: 'pg',
    raw: true,
    strength: 3,
    variable: 'nonignorable',
    pad: 'nopad',
    deterministic: true,
    note: 'memcmp on the stored bytes. No locale, no table, no version — and therefore nothing an OS upgrade can change.',
  },
  {
    id: 'en_US.utf8',
    label: 'en_US.utf8 (libc)',
    engine: 'pg',
    strength: 3,
    variable: 'ignore',
    pad: 'nopad',
    deterministic: true,
    note: 'glibc strcoll(). Spaces and punctuation are variable — ignored until a later level — which is why "zo e" lands next to "zoe" instead of before it.',
  },
  {
    id: 'und-x-icu',
    label: 'und-x-icu (ICU root)',
    engine: 'pg',
    strength: 3,
    variable: 'nonignorable',
    pad: 'nopad',
    deterministic: true,
    note: 'ICU with the CLDR root locale. CLDR sets alternate = non-ignorable, so a space gets a primary weight and sorts before every letter.',
  },
  {
    id: 'und-u-ks-level2',
    label: 'und-u-ks-level2-x-icu (deterministic = false)',
    engine: 'pg',
    strength: 2,
    variable: 'nonignorable',
    pad: 'nopad',
    deterministic: false,
    note: 'Case-insensitive, accent-sensitive, and nondeterministic: equal weights mean equal values, with no byte tiebreak. LIKE and pattern matching are not supported on this column.',
  },
  {
    id: 'utf8mb4_bin',
    label: 'utf8mb4_bin',
    engine: 'mysql',
    raw: true,
    strength: 3,
    variable: 'nonignorable',
    pad: 'nopad',
    deterministic: true,
    note: 'Code-point order, i.e. memcmp on the stored UTF-8 bytes. The one utf8mb4 collation with no weight table behind it.',
  },
  {
    id: 'utf8mb4_0900_as_cs',
    label: 'utf8mb4_0900_as_cs',
    engine: 'mysql',
    strength: 3,
    variable: 'nonignorable',
    pad: 'nopad',
    deterministic: false,
    note: 'UCA 9.0.0, accent- and case-sensitive, NO PAD. Trailing spaces are significant; canonically equivalent sequences share weights and compare equal.',
  },
  {
    id: 'utf8mb4_0900_ai_ci',
    label: 'utf8mb4_0900_ai_ci (8.0 default)',
    engine: 'mysql',
    strength: 1,
    variable: 'nonignorable',
    pad: 'nopad',
    deterministic: false,
    note: 'Primary level only: accents and case are both invisible to = and to a UNIQUE index. The MySQL 8.0 default collation.',
  },
  {
    id: 'utf8mb4_general_ci',
    label: 'utf8mb4_general_ci (legacy, PAD SPACE)',
    engine: 'mysql',
    strength: 1,
    variable: 'nonignorable',
    pad: 'padspace',
    deterministic: false,
    combiningIsPrimary: true,
    note: 'Not a UCA collation: one weight per character, no expansions, and PAD SPACE — trailing spaces are stripped before comparison.',
  },
];

/* Collation elements: primary = base letter, secondary = accent, tertiary = case. */
type CE = { p: number; s: number; t: number; variable: boolean };

function elements(cp: number, combiningIsPrimary: boolean): CE[] {
  if (cp === 0x200d) return []; // ZWJ — completely ignorable at every level
  if (cp === 0x0308)
    return combiningIsPrimary
      ? [{ p: 0x90, s: 1, t: 1, variable: false }]
      : [{ p: 0, s: 5, t: 1, variable: false }];
  if (cp === 0x00eb)
    return combiningIsPrimary
      ? [{ p: 0x104, s: 1, t: 1, variable: false }] // general_ci has one weight per character: e-diaeresis folds to e
      : [...elements(0x65, false), ...elements(0x0308, false)]; // UCA expands the precomposed form
  if (cp === 0x20) return [{ p: 0x10, s: 1, t: 1, variable: true }];
  if (cp === 0x2d) return [{ p: 0x12, s: 1, t: 1, variable: true }];
  if (cp >= 0x30 && cp <= 0x39) return [{ p: 0x40 + (cp - 0x30), s: 1, t: 1, variable: false }];
  if (cp >= 0x61 && cp <= 0x7a) return [{ p: 0x100 + (cp - 0x61), s: 1, t: 1, variable: false }];
  if (cp >= 0x41 && cp <= 0x5a) return [{ p: 0x100 + (cp - 0x41), s: 1, t: 2, variable: false }];
  return [{ p: 0x10000 + cp, s: 1, t: 1, variable: false }]; // implicit weights
}

type SortKey = { l1: number[]; l2: number[]; l3: number[]; l4: number[] };

function sortKey(cps: number[], c: Coll): SortKey {
  const ces = cps.flatMap((cp) => elements(cp, !!c.combiningIsPrimary));
  const l1: number[] = [];
  const l2: number[] = [];
  const l3: number[] = [];
  const l4: number[] = [];
  for (const e of ces) {
    if (e.variable && c.variable === 'ignore') {
      l4.push(e.p);
      continue;
    }
    if (e.p > 0) l1.push(e.p);
    l2.push(e.s);
    l3.push(e.t);
  }
  return { l1, l2, l3, l4 };
}

function cmpArr(a: number[], b: number[]): number {
  const n = Math.min(a.length, b.length);
  for (let i = 0; i < n; i++) if (a[i] !== b[i]) return a[i] < b[i] ? -1 : 1;
  return a.length === b.length ? 0 : a.length < b.length ? -1 : 1;
}

type Decision = 'L1' | 'L2' | 'L3' | 'L4' | 'bytes' | 'equal';

function collCompare(
  a: { cps: number[]; bytes: number[] },
  b: { cps: number[]; bytes: number[] },
  c: Coll,
): { cmp: number; at: Decision } {
  if (c.raw) {
    const r = cmpArr(a.bytes, b.bytes);
    return { cmp: r, at: r === 0 ? 'equal' : 'bytes' };
  }
  const ka = sortKey(a.cps, c);
  const kb = sortKey(b.cps, c);
  let r = cmpArr(ka.l1, kb.l1);
  if (r !== 0) return { cmp: r, at: 'L1' };
  if (c.strength >= 2) {
    r = cmpArr(ka.l2, kb.l2);
    if (r !== 0) return { cmp: r, at: 'L2' };
  }
  if (c.strength >= 3) {
    r = cmpArr(ka.l3, kb.l3);
    if (r !== 0) return { cmp: r, at: 'L3' };
  }
  r = cmpArr(ka.l4, kb.l4);
  if (r !== 0) return { cmp: r, at: 'L4' };
  if (c.deterministic) {
    r = cmpArr(a.bytes, b.bytes);
    if (r !== 0) return { cmp: r, at: 'bytes' };
  }
  return { cmp: 0, at: 'equal' };
}

/* ----------------------------------------------------------------- types */

type TypeId = 'text' | 'varchar' | 'char' | 'uuid' | 'bytea' | 'binary16';

type TypeDef = { id: TypeId; label: string; engine: Engine };

const TYPES: TypeDef[] = [
  { id: 'text', label: 'text', engine: 'pg' },
  { id: 'varchar', label: 'varchar(n)', engine: 'pg' },
  { id: 'char', label: 'char(n)', engine: 'pg' },
  { id: 'uuid', label: 'uuid', engine: 'pg' },
  { id: 'bytea', label: 'bytea', engine: 'pg' },
  { id: 'varchar', label: 'VARCHAR(n)', engine: 'mysql' },
  { id: 'char', label: 'CHAR(n)', engine: 'mysql' },
  { id: 'text', label: 'TEXT', engine: 'mysql' },
  { id: 'binary16', label: 'BINARY(16)', engine: 'mysql' },
];

const NS = [8, 16, 20, 36, 63, 64, 128, 191, 192, 255, 256, 768, 769, 1024];

/* Pad / trim semantics applied before any comparison happens. */
function effectiveCps(cps: number[], engine: Engine, type: TypeId, n: number, coll: Coll): number[] {
  let out = cps.slice();
  if (type === 'char') {
    if (engine === 'pg') {
      while (out.length < n) out.push(0x20); // blank-padded to n characters
    } else {
      while (out.length && out[out.length - 1] === 0x20) out.pop(); // InnoDB strips trailing spaces
    }
  }
  if (coll.pad === 'padspace') {
    while (out.length && out[out.length - 1] === 0x20) out.pop();
  }
  return out;
}

/* ------------------------------------------------------------- storage */

type Role = 'len' | 'payload' | 'pad' | 'ihdr' | 'ptr';

const ROLE_COLOR: Record<Role, string> = {
  len: 'var(--viz-4)',
  payload: 'var(--viz-1)',
  pad: 'var(--viz-stale)',
  ihdr: 'var(--viz-2)',
  ptr: 'var(--viz-7)',
};

const ROLE_LABEL: Record<Role, string> = {
  len: 'length prefix / varlena header',
  payload: 'value payload',
  pad: 'padding',
  ihdr: 'index tuple header',
  ptr: 'row locator (heap TID / primary key)',
};

type Cell = { v: number | null; role: Role; field: string; why: string };

const MAXALIGN = (n: number) => Math.ceil(n / 8) * 8;

type Layout = {
  row: Cell[];
  idx: Cell[];
  rowBytes: number;
  idxBytes: number;
  error?: string;
  noteLines: string[];
};

function layout(f: Fixture, engine: Engine, type: TypeId, n: number, cs: Charset): Layout {
  const notes: string[] = [];
  const row: Cell[] = [];
  const idx: Cell[] = [];

  const pushBytes = (bytes: number[], role: Role, field: string, why: (i: number) => string, into: Cell[]) => {
    bytes.forEach((b, i) => into.push({ v: b, role, field, why: why(i) }));
  };

  /* uuid / binary(16) / a bytea holding decoded bytes: the value is parsed, not stored as text */
  const text = f.cps.map((c) => String.fromCodePoint(c)).join('');
  const isUuid = /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/.test(text);
  if (type === 'binary16') {
    /* BINARY(16) is a fixed-width byte string, not a UUID type: it takes any value of
       16 bytes or fewer and zero-pads it. A canonical UUID only fits once the
       application has parsed it — UNHEX(REPLACE(u,'-','')) — because the text form is
       36 bytes. */
    const bin = isUuid
      ? text.replace(/-/g, '').match(/../g)!.map((h) => parseInt(h, 16))
      : utf8Of(f.cps);
    if (bin.length > 16) {
      return {
        row: [],
        idx: [],
        rowBytes: 0,
        idxBytes: 0,
        error: `ERROR 1406 (22001): Data too long for column 'id' at row 1`,
        noteLines: [
          `BINARY(16) holds exactly 16 bytes, and this value is ${bin.length}. Strict SQL mode makes that an error; with strict mode off it is a warning and a truncation at a byte boundary.`,
        ],
      };
    }
    pushBytes(
      bin,
      'payload',
      isUuid ? 'BINARY(16) — parsed UUID' : 'BINARY(16) — value bytes',
      (i) =>
        isUuid
          ? `byte ${i + 1}/16 of the parsed UUID. The 36-character text form costs 37 bytes with its length prefix; the binary form costs 16 and compares as one memcmp.`
          : `byte ${i + 1}/${bin.length} of the value. BINARY has no charset and no collation: comparison is memcmp on these bytes.`,
      row,
    );
    for (let i = bin.length; i < 16; i++) {
      row.push({
        v: 0x00,
        role: 'pad',
        field: 'zero padding to 16 B',
        why: 'BINARY(n) right-pads with 0x00, not with spaces, and the pad bytes are significant in comparison — so a shorter value is not equal to itself stored in a wider BINARY column.',
      });
    }
    pushBytes([0, 0, 0, 0, 0], 'ihdr', 'record header (5 B)', () =>
      'InnoDB COMPACT/DYNAMIC record header: info bits, n_owned, heap number, record type and next-record offset.',
      idx,
    );
    pushBytes(bin.concat(new Array(16 - bin.length).fill(0)), 'payload', 'key', (i) => `key byte ${i + 1}/16`, idx);
    pushBytes([0, 0, 0, 0, 0, 0, 0, 0], 'ptr', 'PK (BIGINT, 8 B)', () =>
      'Every InnoDB secondary index entry carries the primary key as its row locator. A wide primary key is paid for once per secondary index, per row.',
      idx,
    );
    notes.push(
      'BINARY(16) is fixed width: no length prefix, zero-padded if short, no charset and no collation — every comparison is memcmp.',
    );
    return { row, idx, rowBytes: row.length, idxBytes: idx.length, noteLines: notes };
  }

  if (type === 'uuid' || (type === 'bytea' && isUuid)) {
    if (!isUuid) {
      return {
        row: [],
        idx: [],
        rowBytes: 0,
        idxBytes: 0,
        error: `ERROR: invalid input syntax for type uuid: "${f.short}"  (SQLSTATE 22P02)`,
        noteLines: ['The type refuses the value. That refusal is the feature: a text column would have taken it.'],
      };
    }
    const raw = text.replace(/-/g, '').match(/../g)!.map((h) => parseInt(h, 16));
    if (type === 'bytea') {
      pushBytes([(17 << 1) | 1], 'len', 'varlena header (1 B)', () =>
        'bytea is a varlena like text, so it still pays the 1-byte header — but it has no encoding and no collation, and every comparison is memcmp.',
        row,
      );
    }
    pushBytes(
      raw,
      'payload',
      type === 'uuid' ? 'uuid (16 B, no header)' : 'decoded bytes',
      (i) =>
        `byte ${i + 1}/16 of the parsed UUID. The 36-character text form costs 37 bytes once you add its varlena header; the binary form costs ${
          type === 'bytea' ? 17 : 16
        } and compares as one memcmp.`,
      row,
    );
    if (engine === 'pg') {
      const datum = type === 'bytea' ? 17 : 16;
      pushBytes([0, 0, 0, 0, 0, 0, 0, 0], 'ihdr', 'IndexTupleData (8 B)', () =>
        'Every Postgres btree index tuple starts with an 8-byte IndexTupleData: a 6-byte heap TID plus t_info holding the tuple size and flags.',
        idx,
      );
      if (type === 'bytea') pushBytes([(17 << 1) | 1], 'len', 'varlena header', () => 'The index stores the datum exactly as the heap does, header and all.', idx);
      pushBytes(raw, 'payload', 'key', (i) => `key byte ${i + 1}/16`, idx);
      for (let i = datum; i < MAXALIGN(datum); i++) {
        idx.push({
          v: null,
          role: 'pad',
          field: 'MAXALIGN pad',
          why: 'Index tuples are MAXALIGNed to 8 bytes, so 17 bytes of key costs 24 bytes of page space.',
        });
      }
    } else {
      pushBytes([0, 0, 0, 0, 0], 'ihdr', 'record header (5 B)', () =>
        'InnoDB COMPACT/DYNAMIC record header: 5 bytes of info bits, n_owned, heap number, record type and next-record offset.',
        idx,
      );
      pushBytes(raw, 'payload', 'key', (i) => `key byte ${i + 1}/16`, idx);
      pushBytes([0, 0, 0, 0, 0, 0, 0, 0], 'ptr', 'PK (BIGINT, 8 B)', () =>
        'Every InnoDB secondary index entry carries the primary key as its row locator. A wide primary key is paid for once per secondary index, per row.',
        idx,
      );
    }
    notes.push(
      engine === 'pg'
        ? 'uuid is typlen 16, typalign c: sixteen bytes, no header, no alignment hole, and no collation.'
        : 'BINARY(16) is fixed width: no length prefix, padded with 0x00 (not spaces) if short, and the pad bytes are significant in comparison.',
    );
    return { row, idx, rowBytes: row.length, idxBytes: idx.length, noteLines: notes };
  }

  /* text path */
  const padded = type === 'char' && engine === 'pg' ? effectiveCps(f.cps, engine, 'char', n, COLLS[0]) : f.cps;
  const stored =
    type === 'char' && engine === 'mysql'
      ? effectiveCps(f.cps, engine, 'char', n, COLLS[4])
      : padded;

  const enc = type === 'bytea' ? { bytes: utf8Of(stored), error: undefined } : encode(stored, cs);
  if (enc.error) {
    return { row: [], idx: [], rowBytes: 0, idxBytes: 0, error: enc.error, noteLines: [cs.note] };
  }
  if ((type === 'varchar' || type === 'char') && f.cps.length > n) {
    return {
      row: [],
      idx: [],
      rowBytes: 0,
      idxBytes: 0,
      error:
        engine === 'pg'
          ? `ERROR: value too long for type character${type === 'char' ? '' : ' varying'}(${n})  (SQLSTATE 22001)`
          : `ERROR 1406 (22001): Data too long for column 'name' at row 1`,
      noteLines: [
        engine === 'pg'
          ? 'n in Postgres counts characters, not bytes, and the check happens on assignment.'
          : 'With strict SQL mode off this is a warning and a silent truncation instead — at a character boundary if you are lucky.',
      ],
    };
  }

  const payload = enc.bytes;
  const padCount =
    type === 'char' && engine === 'pg' ? Math.max(0, n - f.cps.length) : 0;
  const realPayload = payload.length - padCount; // padding is one space byte each

  if (engine === 'pg') {
    const headerLen = payload.length + 1 <= 127 ? 1 : 4; // VARATT_SHORT_MAX is 0x7F, total size included
    pushBytes(
      headerLen === 1 ? [((payload.length + 1) << 1) | 1] : [0, 0, 0, 0],
      'len',
      headerLen === 1 ? 'varlena header (1 B)' : 'varlena header (4 B)',
      () =>
        headerLen === 1
          ? `A value whose total size fits in 127 bytes gets the short varlena header: one byte holding the length, low bit set. text, varchar(n) and char(n) all share this representation — the type only changes what is checked on the way in.`
          : `Above 126 payload bytes the header is 4 bytes, and the datum is 4-byte aligned, so a wide text column can also cost alignment padding. Past roughly 2 KB the whole value is compressed and moved to the TOAST table.`,
      row,
    );
    payload.forEach((b, i) => {
      const isPad = padCount > 0 && i >= realPayload;
      row.push({
        v: b,
        role: isPad ? 'pad' : 'payload',
        field: isPad ? `blank padding to ${n} chars` : 'payload',
        why: isPad
          ? `char(${n}) blank-pads to n CHARACTERS. These spaces are stored, cached, indexed, replicated and shipped to the client — and then ignored by every comparison, because bpchar comparison strips them again.`
          : byteWhy(b, i, payload.length, cs),
      });
    });
    const keyHeader = headerLen === 1 ? [((payload.length + 1) << 1) | 1] : [0, 0, 0, 0];
    const datum = keyHeader.length + payload.length;
    const keyLen = MAXALIGN(datum);
    pushBytes([0, 0, 0, 0, 0, 0, 0, 0], 'ihdr', 'IndexTupleData (8 B)', () =>
      'An 8-byte IndexTupleData: 6-byte heap TID (block number + offset) plus 2 bytes of size and flags. That is the floor on an index entry no matter how narrow the key.',
      idx,
    );
    pushBytes(keyHeader, 'len', 'key varlena header', () =>
      'The index stores the datum exactly as the heap does, header and all.',
      idx,
    );
    pushBytes(payload, 'payload', 'key datum', (i) =>
      `key byte ${i + 1}/${payload.length}. The index stores the value, not a precomputed sort key: order is produced by calling the collation on every comparison during a descent.`,
      idx,
    );
    for (let i = datum; i < keyLen; i++) {
      idx.push({
        v: null,
        role: 'pad',
        field: 'MAXALIGN pad',
        why: 'Index tuples are MAXALIGNed to 8 bytes, so a 4-byte key and a 1-byte key can cost exactly the same page space.',
      });
    }
    if (datum > 2704) {
      return {
        row,
        idx,
        rowBytes: row.length,
        idxBytes: idx.length,
        error: `ERROR: index row size ${MAXALIGN(datum) + 8} exceeds btree version 4 maximum 2704 for index "name_idx"`,
        noteLines: ['A btree tuple must fit three to a page, so the key limit is roughly 8 KB / 3.'],
      };
    }
  } else {
    const maxBytes = (type === 'text' ? 65535 : n) * cs.maxLen;
    const prefixLen = type === 'text' ? 2 : maxBytes > 255 ? 2 : 1;
    const fixedMin = type === 'char' ? n : 0;
    pushBytes(
      new Array(prefixLen).fill(payload.length & 0xff),
      'len',
      `length prefix (${prefixLen} B)`,
      () =>
        type === 'text'
          ? 'A TEXT column carries a 2-byte in-record length; if the row outgrows half a page the value moves to an overflow page and the record keeps a 20-byte pointer instead.'
          : `InnoDB uses a 1-byte length prefix when the column's MAXIMUM byte length is 255 or less, and 2 bytes above that. ${n} x ${cs.maxLen} = ${maxBytes}, so this column costs ${prefixLen}. Widening the charset from utf8mb3 to utf8mb4 can flip that byte on every row.`,
      row,
    );
    payload.forEach((b, i) => row.push({ v: b, role: 'payload', field: 'payload', why: byteWhy(b, i, payload.length, cs) }));
    for (let i = payload.length; i < fixedMin; i++) {
      row.push({
        v: 0x20,
        role: 'pad',
        field: `CHAR(${n}) floor`,
        why: `InnoDB strips trailing spaces from a CHAR(n) in a variable-length charset, but never lets the column occupy fewer than n bytes — so you pay for n and store fewer. On retrieval MySQL also strips trailing spaces, unless sql_mode includes PAD_CHAR_TO_FULL_LENGTH.`,
      });
    }
    pushBytes([0, 0, 0, 0, 0], 'ihdr', 'record header (5 B)', () =>
      'InnoDB COMPACT/DYNAMIC record header: info bits, n_owned, heap number, record type, next-record offset — 5 bytes before any data.',
      idx,
    );
    if (prefixLen) pushBytes(new Array(prefixLen).fill(payload.length & 0xff), 'len', 'key length', () => 'The variable-length field length, repeated in the index record.', idx);
    payload.forEach((b, i) => idx.push({ v: b, role: 'payload', field: 'key', why: byteWhy(b, i, payload.length, cs) }));
    pushBytes([0, 0, 0, 0, 0, 0, 0, 0], 'ptr', 'PK (BIGINT, 8 B)', () =>
      'The clustered-index key is appended to every secondary index entry as the row locator. This is why a UUID text primary key is so expensive: it is copied into every secondary index of the table.',
      idx,
    );
    if (maxBytes > 3072 && type !== 'text') {
      return {
        row,
        idx,
        rowBytes: row.length,
        idxBytes: idx.length,
        error: `ERROR 1071 (42000): Specified key was too long; max key length is 3072 bytes`,
        noteLines: [
          `${n} x ${cs.maxLen} bytes = ${maxBytes} > 3072. On the older COMPACT/REDUNDANT row formats the limit is 767 bytes, which is where varchar(191) in half the world's schemas comes from: 191 x 4 = 764.`,
        ],
      };
    }
  }

  return { row, idx, rowBytes: row.length, idxBytes: idx.length, noteLines: notes };
}

function byteWhy(b: number, i: number, n: number, cs: Charset) {
  if (cs.kind === 'latin1') return `single-byte ${cs.label} character, byte ${i + 1}/${n}`;
  if (b < 0x80) return `ASCII '${String.fromCharCode(b)}' — UTF-8 byte ${i + 1}/${n}`;
  if (b >= 0xf0) return `lead byte of a 4-byte UTF-8 sequence (U+10000 and above), byte ${i + 1}/${n}`;
  if (b >= 0xe0) return `lead byte of a 3-byte UTF-8 sequence, byte ${i + 1}/${n}`;
  if (b >= 0xc0) return `lead byte of a 2-byte UTF-8 sequence, byte ${i + 1}/${n}`;
  return `continuation byte (10xxxxxx), byte ${i + 1}/${n} — it is not a character, and splitting here produces invalid UTF-8`;
}

/* ---------------------------------------------------------------- render */

function ByteStrip({ cells, cap = 40 }: { cells: Cell[]; cap?: number }) {
  const tip = useTip();
  const runs: { field: string; role: Role; items: { c: Cell; i: number }[] }[] = [];
  cells.slice(0, cap).forEach((c, i) => {
    const last = runs[runs.length - 1];
    if (last && last.field === c.field) last.items.push({ c, i });
    else runs.push({ field: c.field, role: c.role, items: [{ c, i }] });
  });
  const hidden = Math.max(0, cells.length - cap);

  return (
    <div style={{ display: 'flex', flexWrap: 'wrap', gap: '.4rem .5rem', alignItems: 'flex-end' }}>
      {runs.map((run, k) => (
        <div key={`${run.field}-${k}`} style={{ maxWidth: '100%' }}>
          <div
            style={{
              fontSize: '.625rem',
              color: 'var(--viz-ink-2)',
              marginBottom: '2px',
              whiteSpace: 'nowrap',
              overflow: 'hidden',
              textOverflow: 'ellipsis',
            }}
          >
            {run.field} &middot; {run.items.length} B
          </div>
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: '2px' }}>
            {run.items.map(({ c, i }, j) => (
              <span
                key={i}
                {...tip(
                  <>
                    <strong>
                      {c.v === null ? 'pad' : `0x${hex(c.v)}`} at offset {i}
                    </strong>
                    <br />
                    {ROLE_LABEL[c.role]} &middot; {c.field}
                    <br />
                    <span style={{ color: 'var(--viz-ink-2)' }}>{c.why}</span>
                  </>,
                )}
                tabIndex={j === 0 ? 0 : -1}
                style={{
                  display: 'inline-block',
                  minWidth: '1.9rem',
                  textAlign: 'center',
                  padding: '.12rem .15rem',
                  fontSize: '.6875rem',
                  lineHeight: 1.4,
                  color: 'var(--viz-ink)',
                  fontVariantNumeric: 'tabular-nums',
                  background: `color-mix(in srgb, ${ROLE_COLOR[c.role]} 16%, var(--viz-plane))`,
                  border: '1px solid var(--viz-border)',
                  borderBottom: `3px solid ${ROLE_COLOR[c.role]}`,
                  borderRadius: '4px',
                  cursor: 'help',
                }}
              >
                {c.v === null ? '··' : hex(c.v)}
              </span>
            ))}
          </div>
        </div>
      ))}
      {hidden > 0 ? (
        <span style={{ fontSize: '.6875rem', color: 'var(--viz-ink-2)', paddingBottom: '.2rem' }}>
          + {hidden} more bytes
        </span>
      ) : null}
    </div>
  );
}

const DECIDED: Record<Decision, string> = {
  L1: 'decided at L1 — base letters differ',
  L2: 'decided at L2 — accents differ',
  L3: 'decided at L3 — case differs',
  L4: 'decided at L4 — only the variable characters (space) differ',
  bytes: 'weights tie: decided by the deterministic memcmp tiebreak',
  equal: 'EQUAL under this collation',
};

export default function TextTypeCollationSwitchboard() {
  const [engine, setEngine] = useState<Engine>('pg');
  const [typeId, setTypeId] = useState<TypeId>('text');
  const [nIdx, setNIdx] = useState(2);
  const [csId, setCsId] = useState('UTF8');
  const [collId, setCollId] = useState('C');
  const [valueId, setValueId] = useState('nfd');
  const [ref, width] = useSize(760);

  const n = NS[nIdx];
  const cs = CHARSETS.find((c) => c.id === csId && c.engine === engine) ?? CHARSETS.find((c) => c.engine === engine)!;
  const collSel = COLLS.find((c) => c.id === collId && c.engine === engine) ?? COLLS.find((c) => c.engine === engine)!;
  const types = TYPES.filter((t) => t.engine === engine);
  const type = types.find((t) => t.id === typeId) ?? types[0];
  const value = RULER_VALUES.find((v) => v.id === valueId) ?? RULER_VALUES[0];
  const binaryType = typeId === 'bytea' || typeId === 'binary16' || typeId === 'uuid';
  const coll = binaryType ? BYTES_COLL : collSel;

  const switchEngine = (e: Engine) => {
    setEngine(e);
    setTypeId(e === 'pg' ? 'text' : 'varchar');
    setCsId(e === 'pg' ? 'UTF8' : 'utf8mb4');
    setCollId(e === 'pg' ? 'C' : 'utf8mb4_0900_ai_ci');
  };

  const lay = useMemo(
    () => layout(value, engine, type.id, n, cs),
    [value.id, engine, type.id, n, cs.id],
  );

  /* every fixture under the current declaration */
  const rows = useMemo(() => {
    return FIXTURES.map((f) => {
      const l = layout(f, engine, type.id, n, cs);
      const eff = effectiveCps(f.cps, engine, type.id, n, coll);
      const bytes = encode(eff, cs).bytes;
      return { f, l, eff, bytes, ok: !l.error };
    });
  }, [engine, type.id, n, cs.id, coll.id]);

  const accepted = rows.filter((r) => r.ok);
  const sorted = useMemo(() => {
    const arr = accepted.slice();
    arr.sort((a, b) => {
      const r = collCompare({ cps: a.eff, bytes: a.bytes }, { cps: b.eff, bytes: b.bytes }, coll).cmp;
      return r !== 0 ? r : a.f.id < b.f.id ? -1 : 1;
    });
    return arr;
  }, [accepted, coll.id]);

  /* equality classes: each group is one value as far as = and UNIQUE are concerned */
  const groups: (typeof sorted)[] = [];
  sorted.forEach((r) => {
    const g = groups[groups.length - 1];
    if (
      g &&
      collCompare({ cps: g[0].eff, bytes: g[0].bytes }, { cps: r.eff, bytes: r.bytes }, coll).cmp === 0
    ) {
      g.push(r);
    } else groups.push([r]);
  });

  const dupes = groups.filter((g) => g.length > 1);
  const rejected = rows.filter((r) => !r.ok);

  const pgPageUsable = 8152;
  const myPageUsable = 16256;
  const perEntry = engine === 'pg' ? lay.idxBytes + 4 : lay.idxBytes + 2;
  const fanout = lay.idxBytes ? Math.floor((engine === 'pg' ? pgPageUsable : myPageUsable) / perEntry) : 0;

  const lenFnName = engine === 'pg' ? 'length()' : 'CHAR_LENGTH()';
  const charLen = (r: { f: Fixture; eff: number[]; bytes: number[] }) => {
    if (cs.kind === 'rawbytes') return r.bytes.length; // SQL_ASCII: no encoding, so length() counts bytes
    const e = r.eff.slice();
    // bpchar's length() strips trailing blanks, even though octet_length() still sees the padding
    if (engine === 'pg' && type.id === 'char') while (e.length && e[e.length - 1] === 0x20) e.pop();
    return e.length;
  };

  const notes: string[] = [];
  if (lay.error) notes.push(lay.error);
  if (dupes.length)
    notes.push(
      `${coll.label} makes ${dupes.map((g) => g.map((r) => r.f.short).join(' = ')).join('; ')} the same value: a UNIQUE index accepts the first row of each group and rejects the rest with ${
        engine === 'pg'
          ? 'ERROR: duplicate key value violates unique constraint (23505)'
          : "ERROR 1062 (23000): Duplicate entry"
      }.`,
    );
  if (rejected.length)
    notes.push(`${rejected.length} value${rejected.length > 1 ? 's' : ''} cannot be stored at all: ${rejected[0].l.error}`);
  if (!notes.length) notes.push(coll.note);

  const cols = width >= 700 ? 2 : 1;

  return (
    <VizPanel
      title="Type, encoding and collation switchboard"
      subtitle="Change the column definition on the left and watch the bytes, the index entry and the ordering answers change. The eight fixture values are re-sorted, re-compared and re-counted under whatever you just declared."
      controls={
        <>
          <Segmented
            label="Engine"
            value={engine}
            onChange={switchEngine}
            options={[
              { value: 'pg', label: 'PostgreSQL' },
              { value: 'mysql', label: 'MySQL / InnoDB' },
            ]}
          />
          <Choice
            label="Column type"
            value={type.id}
            onChange={(v) => setTypeId(v as TypeId)}
            options={types.map((t) => ({ value: t.id, label: t.label }))}
          />
          <Slider
            label="n"
            min={0}
            max={NS.length - 1}
            value={nIdx}
            onChange={setNIdx}
            format={() => String(n)}
            disabled={type.id !== 'varchar' && type.id !== 'char'}
          />
          <Choice
            label="Encoding"
            value={cs.id}
            onChange={setCsId}
            options={CHARSETS.filter((c) => c.engine === engine).map((c) => ({ value: c.id, label: c.label }))}
          />
          <Choice
            label="Collation"
            value={collSel.id}
            onChange={setCollId}
            options={COLLS.filter((c) => c.engine === engine).map((c) => ({ value: c.id, label: c.label }))}
          />
          <Choice
            label="Value in the ruler"
            value={value.id}
            onChange={setValueId}
            options={RULER_VALUES.map((v) => ({ value: v.id, label: v.label }))}
          />
        </>
      }
      legend={
        <Legend
          items={[
            { label: ROLE_LABEL.len, color: ROLE_COLOR.len },
            { label: ROLE_LABEL.payload, color: ROLE_COLOR.payload },
            { label: ROLE_LABEL.pad, color: ROLE_COLOR.pad },
            { label: ROLE_LABEL.ihdr, color: ROLE_COLOR.ihdr },
            { label: ROLE_LABEL.ptr, color: ROLE_COLOR.ptr },
            { label: 'compares equal — UNIQUE rejects', color: 'var(--viz-warning)' },
            { label: 'rejected by the encoding or the type', color: 'var(--viz-critical)' },
          ]}
        />
      }
      stats={
        <Stats
          items={[
            { label: 'Row bytes', value: lay.error && !lay.rowBytes ? 'rejected' : `${lay.rowBytes} B`, hint: 'Stored bytes for the selected value, header and padding included' },
            { label: 'Index entry', value: lay.idxBytes ? `${lay.idxBytes} B` : 'n/a', hint: engine === 'pg' ? '8-byte IndexTupleData + MAXALIGNed key' : '5-byte record header + key + primary key' },
            { label: 'Entries / leaf page', value: fanout ? String(fanout) : 'n/a', hint: engine === 'pg' ? 'Approximate: 8152 usable bytes per 8 KB page, 4 bytes of line pointer per entry' : 'Approximate: ~16256 usable bytes per 16 KB page' },
            { label: 'Distinct values', value: `${groups.length} of ${accepted.length}`, hint: 'Equality classes under this collation — what a UNIQUE index would keep' },
            { label: 'Unstorable', value: String(rejected.length), hint: 'Fixtures the encoding or the type refuses' },
          ]}
        />
      }
      note={<strong>{notes[0]}</strong>}
      table={
        <table className="viz-table">
          <thead>
            <tr>
              <th>Value</th>
              <th>Stored bytes</th>
              <th>octet_length</th>
              <th>{lenFnName}</th>
              <th>Graphemes</th>
              <th>Rank</th>
              <th>Equality class</th>
              <th>Row bytes</th>
              <th>Index entry</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((r) => {
              const rank = sorted.findIndex((s) => s.f.id === r.f.id);
              const gi = groups.findIndex((g) => g.some((s) => s.f.id === r.f.id));
              return (
                <tr key={r.f.id}>
                  <td>{r.f.label}</td>
                  <td style={{ wordBreak: 'break-word' }}>
                    {r.ok ? r.bytes.slice(0, 16).map(hex).join(' ') + (r.bytes.length > 16 ? ' …' : '') : '—'}
                  </td>
                  <td>{r.ok ? r.bytes.length : '—'}</td>
                  <td>{r.ok ? charLen(r) : '—'}</td>
                  <td>{r.f.graphemes}</td>
                  <td>{rank >= 0 ? rank + 1 : '—'}</td>
                  <td>{gi >= 0 ? `#${gi + 1}${groups[gi].length > 1 ? ` (${groups[gi].length} rows collide)` : ''}` : 'rejected'}</td>
                  <td>{r.ok ? r.l.rowBytes : r.l.error?.slice(0, 28)}</td>
                  <td>{r.ok ? r.l.idxBytes : '—'}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      }
    >
      <div ref={ref}>
        <TooltipHost>
          <div style={{ display: 'grid', gridTemplateColumns: cols === 2 ? '1fr 1fr' : '1fr', gap: '1rem 1.25rem', minWidth: '280px' }}>
            <section>
              <div style={{ fontSize: '.8125rem', color: 'var(--viz-ink)' }}>
                <strong>
                  {value.short} stored as {type.label.replace('(n)', `(${n})`)}{' '}
                  {binaryType ? '' : cs.label.split(' ')[0]}
                </strong>
              </div>
              <div style={{ fontSize: '.6875rem', color: 'var(--viz-ink-2)', margin: '.15rem 0 .5rem' }}>{value.why}</div>
              {lay.error && lay.rowBytes === 0 ? (
                <div
                  style={{
                    fontSize: '.75rem',
                    color: 'var(--viz-ink)',
                    padding: '.5rem .6rem',
                    borderLeft: '3px solid var(--viz-critical)',
                    background: 'color-mix(in srgb, var(--viz-critical) 10%, var(--viz-plane))',
                    borderRadius: '4px',
                  }}
                >
                  {lay.error}
                  <div style={{ color: 'var(--viz-ink-2)', marginTop: '.3rem' }}>{lay.noteLines[0]}</div>
                </div>
              ) : (
                <>
                  <div style={{ fontSize: '.6875rem', color: 'var(--viz-ink-2)', marginBottom: '.25rem' }}>
                    Row storage &mdash; {lay.rowBytes} bytes
                  </div>
                  <ByteStrip cells={lay.row} />
                  <div style={{ fontSize: '.6875rem', color: 'var(--viz-ink-2)', margin: '.6rem 0 .25rem' }}>
                    One btree entry &mdash; {lay.idxBytes} bytes, about {fanout} per leaf page
                  </div>
                  <ByteStrip cells={lay.idx} />
                  {lay.error ? (
                    <div style={{ fontSize: '.6875rem', color: 'var(--viz-critical)', marginTop: '.4rem' }}>{lay.error}</div>
                  ) : null}
                </>
              )}
            </section>

            <section>
              <div style={{ fontSize: '.8125rem', color: 'var(--viz-ink)' }}>
                <strong>
                  {binaryType
                    ? 'ORDER BY id  — byte order, no collation'
                    : `ORDER BY name COLLATE "${coll.label.split(' ')[0]}"`}
                </strong>
              </div>
              <div style={{ fontSize: '.6875rem', color: 'var(--viz-ink-2)', margin: '.15rem 0 .5rem' }}>{coll.note}</div>
              <div style={{ display: 'flex', flexDirection: 'column' }}>
                {groups.map((g, gi) => {
                  const dup = g.length > 1;
                  const prevG = gi > 0 ? groups[gi - 1] : null;
                  const boundary = prevG
                    ? collCompare(
                        { cps: prevG[prevG.length - 1].eff, bytes: prevG[prevG.length - 1].bytes },
                        { cps: g[0].eff, bytes: g[0].bytes },
                        coll,
                      ).at
                    : null;
                  return (
                    <div key={g[0].f.id}>
                      {boundary ? (
                        <div style={{ fontSize: '.625rem', color: 'var(--viz-ink-2)', padding: '.2rem 0 .2rem .55rem' }}>
                          &darr; {DECIDED[boundary]}
                        </div>
                      ) : null}
                      <div
                        style={{
                          borderLeft: `3px solid ${dup ? 'var(--viz-warning)' : 'var(--viz-border)'}`,
                          background: dup
                            ? 'color-mix(in srgb, var(--viz-warning) 12%, var(--viz-plane))'
                            : 'var(--viz-plane)',
                          borderRadius: '4px',
                          padding: '.25rem .45rem',
                        }}
                      >
                        {g.map((r) => (
                          <div
                            key={r.f.id}
                            style={{ display: 'flex', gap: '.5rem', alignItems: 'baseline', fontSize: '.75rem' }}
                          >
                            <span
                              style={{
                                color: 'var(--viz-ink-2)',
                                fontVariantNumeric: 'tabular-nums',
                                minWidth: '1.5rem',
                              }}
                            >
                              {sorted.findIndex((s2) => s2.f.id === r.f.id) + 1}.
                            </span>
                            <span style={{ color: 'var(--viz-ink)' }}>{r.f.short}</span>
                            <span
                              style={{
                                color: 'var(--viz-ink-2)',
                                fontSize: '.625rem',
                                marginLeft: 'auto',
                                whiteSpace: 'nowrap',
                              }}
                            >
                              {r.bytes.length} B &middot; {charLen(r)} cp &middot; {r.f.graphemes} gr
                            </span>
                          </div>
                        ))}
                        {dup ? (
                          <div style={{ fontSize: '.625rem', color: 'var(--viz-ink-2)', marginTop: '.15rem' }}>
                            = EQUAL: one value to <code>=</code>, to GROUP BY, to DISTINCT and to UNIQUE.{' '}
                            {engine === 'pg'
                              ? 'ERROR: duplicate key value violates unique constraint (23505)'
                              : 'ERROR 1062 (23000): Duplicate entry'}
                          </div>
                        ) : null}
                      </div>
                    </div>
                  );
                })}
                {rejected.map((r) => (
                  <div
                    key={r.f.id}
                    style={{
                      borderLeft: '3px solid var(--viz-critical)',
                      background: 'color-mix(in srgb, var(--viz-critical) 10%, var(--viz-plane))',
                      borderRadius: '4px',
                      padding: '.25rem .45rem',
                      marginTop: '.35rem',
                      fontSize: '.75rem',
                      color: 'var(--viz-ink)',
                    }}
                  >
                    <span style={{ textDecoration: 'line-through' }}>{r.f.short}</span>
                    <div style={{ fontSize: '.625rem', color: 'var(--viz-ink-2)' }}>{r.l.error}</div>
                  </div>
                ))}
              </div>
            </section>
          </div>
        </TooltipHost>
      </div>
    </VizPanel>
  );
}
