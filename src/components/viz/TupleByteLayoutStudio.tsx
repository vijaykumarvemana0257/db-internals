import { useMemo, useState } from 'react';
import {
  VizPanel,
  Segmented,
  Choice,
  Slider,
  Button,
  Legend,
  Stats,
  Note,
  TooltipHost,
  useTip,
  useSize,
  fmtNum,
} from './Viz';

/**
 * One row, three on-disk record formats, byte for byte.
 *
 * Everything here follows the real rules:
 *  - PostgreSQL: 23-byte HeapTupleHeaderData (t_xmin, t_xmax, t_cid, t_ctid,
 *    t_infomask2, t_infomask, t_hoff), an optional ceil(natts/8) null bitmap,
 *    then MAXALIGN(8) to t_hoff, then attributes padded to their typalign.
 *    A varlena whose value fits a 1-byte short header (<= 126 bytes) is stored
 *    unaligned; anything longer takes a 4-byte header and 4-byte alignment.
 *    The whole tuple is MAXALIGNed into its page slot and costs 4 more bytes
 *    for its ItemIdData line pointer.
 *  - InnoDB COMPACT/DYNAMIC: variable-length lengths (1 or 2 bytes each, in
 *    reverse column order), the null bit vector sized by the number of nullable
 *    columns, a 5-byte record header, then the key, DB_TRX_ID (6), DB_ROLL_PTR
 *    (7) and the remaining columns. No alignment padding anywhere.
 *  - SQLite: a varint cell length, a varint rowid, then a record whose header is
 *    a varint header size plus one varint serial type per column, then the
 *    bodies. Serial types 8 and 9 encode the integers 0 and 1 in the type itself;
 *    an INTEGER PRIMARY KEY is serial type 0 because the value lives in the rowid.
 */

/* ------------------------------------------------------------------ model */

type TypeKey = 'bool' | 'int4' | 'int8' | 'text' | 'numeric' | 'timestamptz';
type NullState = 'notnull' | 'nullable' | 'null';
type Engine = 'pg' | 'innodb' | 'sqlite';
type Role = 'val' | 'hdr' | 'nullmap' | 'len' | 'sys' | 'pad';

const ROLE_COLOR: Record<Role, string> = {
  val: 'var(--viz-1)',
  hdr: 'var(--viz-2)',
  nullmap: 'var(--viz-3)',
  len: 'var(--viz-4)',
  sys: 'var(--viz-5)',
  pad: 'var(--viz-stale)',
};

const ROLE_LABEL: Record<Role, string> = {
  val: 'attribute value',
  hdr: 'tuple / record header',
  nullmap: 'null bitmap',
  len: 'length or varlena header',
  sys: 'hidden system column',
  pad: 'alignment padding',
};

type TypeInfo = {
  pg: string;
  pgLen: number; // > 0 fixed width, -1 varlena
  pgAlign: number; // typalign: c=1, s=2, i=4, d=8
  my: string;
  myLen: number; // -1 = variable length
  myMax: number; // maximum byte length of the column, for the 1-vs-2-byte length rule
  sq: string;
};

const T: Record<TypeKey, TypeInfo> = {
  bool: { pg: 'boolean', pgLen: 1, pgAlign: 1, my: 'TINYINT(1)', myLen: 1, myMax: 1, sq: 'INTEGER' },
  int4: { pg: 'integer', pgLen: 4, pgAlign: 4, my: 'INT', myLen: 4, myMax: 4, sq: 'INTEGER' },
  int8: { pg: 'bigint', pgLen: 8, pgAlign: 8, my: 'BIGINT', myLen: 8, myMax: 8, sq: 'INTEGER' },
  timestamptz: { pg: 'timestamptz', pgLen: 8, pgAlign: 8, my: 'TIMESTAMP', myLen: 4, myMax: 4, sq: 'TEXT' },
  numeric: { pg: 'numeric(10,2)', pgLen: -1, pgAlign: 4, my: 'DECIMAL(10,2)', myLen: 5, myMax: 5, sq: 'REAL' },
  text: { pg: 'text', pgLen: -1, pgAlign: 4, my: 'VARCHAR(1000)', myLen: -1, myMax: 4000, sq: 'TEXT' },
};

type Col = { key: string; name: string; type: TypeKey; ns: NullState; pk?: boolean };

const BASE: Col[] = [
  { key: 'active', name: 'active', type: 'bool', ns: 'nullable' },
  { key: 'id', name: 'id', type: 'int8', ns: 'notnull', pk: true },
  { key: 'views', name: 'views', type: 'int4', ns: 'nullable' },
  { key: 'created_at', name: 'created_at', type: 'timestamptz', ns: 'notnull' },
  { key: 'price', name: 'price', type: 'numeric', ns: 'nullable' },
  { key: 'title', name: 'title', type: 'text', ns: 'nullable' },
];

/** numeric 1234.56 on disk: uint16 n_header + two base-10000 NumericDigits. */
const NUMERIC_PAYLOAD = 6;
/** The ISO-8601 string a SQLite schema stores a timestamp as: '2026-09-13 18:22:41'. */
const SQLITE_TS_BYTES = 19;

const TEXT_SIZES = [8, 40, 200, 3000] as const;

type Run = { key: string; role: Role; label: string; bytes: number; why: string };

const MAXALIGN = (n: number) => Math.ceil(n / 8) * 8;
const alignTo = (off: number, a: number) => Math.ceil(off / a) * a;

/* ------------------------------------------------------------- PostgreSQL */

const PG_PAGE_FREE = 8192 - 24; // BLCKSZ minus PageHeaderData; heap pages have no special area
const TOAST_TUPLE_THRESHOLD = 2048; // MAXALIGN(BLCKSZ / 4)

type Layout = {
  runs: Run[];
  total: number; // bytes of the record itself
  pad: number;
  header: number; // bytes spent before the first attribute value
  perRow: number; // bytes the row costs on a page, slot/pointer included
  rows: number; // rows per page
  toasted: boolean;
};

function pgLayout(cols: Col[], textBytes: number): Layout {
  const natts = cols.length;
  const anyNull = cols.some((c) => c.ns === 'null');
  const bitmap = anyNull ? Math.ceil(natts / 8) : 0;
  const hoff = MAXALIGN(23 + bitmap);

  const build = (toastText: boolean) => {
    const runs: Run[] = [];
    const h = (key: string, label: string, bytes: number, why: string) =>
      runs.push({ key, role: 'hdr', label, bytes, why });

    h('xmin', 't_xmin', 4, 'The xid that inserted this version. Visibility is decided from these four bytes plus the snapshot — there is no separate version table.');
    h('xmax', 't_xmax', 4, 'The xid that deleted or row-locked this version; 0 while the row is live. An UPDATE stamps it here and writes a whole new tuple.');
    h('cid', 't_cid', 4, 'Command id inside the inserting transaction, so a statement cannot see rows it wrote itself. It shares its 4 bytes with t_xvac in a union — the header has no room for both.');
    h('ctid', 't_ctid', 6, 'ItemPointerData: BlockIdData (4 bytes) + OffsetNumber (2). Normally this tuple’s own address; after an UPDATE it points at the next version, which is how HOT chains are walked.');
    h('im2', 't_infomask2', 2, `Low 11 bits (HEAP_NATTS_MASK) hold natts = ${natts} — the number of attributes actually stored, which is how a tuple written before ADD COLUMN still deforms. HEAP_HOT_UPDATED and HEAP_ONLY_TUPLE live in the high bits.`);
    h('im', 't_infomask', 2, `Hint bits: HEAP_HASNULL ${anyNull ? 'is set, so a null bitmap follows' : 'is clear, so there is no null bitmap at all'}; HEAP_HASVARWIDTH; HEAP_XMIN_COMMITTED / HEAP_XMAX_INVALID are cached commit answers so a reader does not have to consult the commit log twice.`);
    h('hoff', 't_hoff', 1, `Offset from the start of the tuple to the first attribute: ${hoff} here. Everything before it is bookkeeping you pay on every single row.`);

    if (bitmap > 0) {
      runs.push({
        key: 'tbits',
        role: 'nullmap',
        label: 't_bits',
        bytes: bitmap,
        why: `One bit per attribute — ceil(${natts}/8) = ${bitmap} byte${bitmap === 1 ? '' : 's'}, bit set means NOT NULL. It exists only because some column in this row is null; a row with no nulls carries no bitmap. NULLs cost bits, not bytes.`,
      });
    }
    const headPad = hoff - (23 + bitmap);
    if (headPad > 0) {
      runs.push({
        key: 'hpad',
        role: 'pad',
        label: 'MAXALIGN',
        bytes: headPad,
        why: `t_hoff is MAXALIGN(23 + ${bitmap}) so the first attribute starts 8-byte aligned. With 8 or fewer columns the null bitmap fits inside this slack and is genuinely free; at 9 columns t_hoff jumps from 24 to 32.`,
      });
    }

    let off = 0;
    for (const c of cols) {
      const info = T[c.type];
      if (c.ns === 'null') continue; // a null attribute occupies no bytes at all

      if (info.pgLen > 0) {
        const p = alignTo(off, info.pgAlign) - off;
        if (p > 0) {
          runs.push({
            key: `${c.key}-pad`,
            role: 'pad',
            label: `pad ${p}`,
            bytes: p,
            why: `${c.name} is ${info.pg}: typalign ${info.pgAlign === 8 ? "'d'" : info.pgAlign === 4 ? "'i'" : "'c'"} means it must start at a multiple of ${info.pgAlign}. The cursor was at ${off}, so ${p} dead byte${p === 1 ? '' : 's'} go in first. Move a wider column ahead of the narrow ones and this disappears.`,
          });
          off += p;
        }
        runs.push({
          key: c.key,
          role: 'val',
          label: c.name,
          bytes: info.pgLen,
          why: `${c.name} ${info.pg}: fixed ${info.pgLen} bytes at offset ${off}. Fixed width and no preceding null or varlena means its offset is constant for every row, so pg_attribute.attcacheoff caches it and heap_deform_tuple jumps straight to it.`,
        });
        off += info.pgLen;
        continue;
      }

      // varlena
      const payload = c.type === 'text' ? textBytes : NUMERIC_PAYLOAD;
      if (c.type === 'text' && toastText) {
        runs.push({
          key: c.key,
          role: 'len',
          label: `${c.name} → TOAST ptr`,
          bytes: 18,
          why: `The tuple crossed TOAST_TUPLE_THRESHOLD (${TOAST_TUPLE_THRESHOLD} bytes), so ${c.name} was compressed and then pushed out of line. What is left in the heap tuple is an 18-byte varatt_external: a 1-byte header and tag, va_rawsize, va_extsize, va_valueid and va_toastrelid. It has a short header, so it needs no alignment either.`,
        });
        off += 18;
        continue;
      }
      if (payload <= 126) {
        runs.push({
          key: `${c.key}-h`,
          role: 'len',
          label: '1B hdr',
          bytes: 1,
          why: `Short varlena header: one byte holding (length << 1) | 0x01, covering values up to 126 bytes. Because the header is short the value is allowed to start unaligned, so ${c.name} pays no padding at all — the single biggest space win of the 8.3 varlena rework.`,
        });
        runs.push({
          key: c.key,
          role: 'val',
          label: c.name,
          bytes: payload,
          why:
            c.type === 'numeric'
              ? 'numeric is a varlena: uint16 of sign, display scale and weight, then base-10000 digits packed two decimal digits per byte. 1234.56 is two NumericDigits, so 6 payload bytes — and numeric(10,2) and numeric(1000,2) cost the same until the value gets bigger.'
              : `${payload} bytes of UTF-8, stored inline with no terminator. Postgres never trims a text value to a declared length; varchar(n) and text have identical on-disk representations.`,
        });
        off += 1 + payload;
      } else {
        const p = alignTo(off, 4) - off;
        if (p > 0) {
          runs.push({
            key: `${c.key}-pad`,
            role: 'pad',
            label: `pad ${p}`,
            bytes: p,
            why: `A 4-byte varlena header is an int32, so it has to be 4-byte aligned (typalign 'i'). Under 127 bytes the same value would have used the short header and skipped this padding entirely.`,
          });
          off += p;
        }
        runs.push({
          key: `${c.key}-h`,
          role: 'len',
          label: '4B hdr',
          bytes: 4,
          why: `At 127 bytes and up the short header cannot hold the length, so va_header becomes a 4-byte int32 (length including the header, with two of its bits reserved for the compressed and external flags — which is where the 1 GB limit on a single field comes from). Crossing 126 bytes costs 3 header bytes plus up to 3 bytes of new alignment padding.`,
        });
        runs.push({ key: c.key, role: 'val', label: c.name, bytes: payload, why: `${payload} bytes of UTF-8, stored inline.` });
        off += 4 + payload;
      }
    }
    return { runs, dataLen: off };
  };

  let built = build(false);
  let toasted = false;
  if (hoff + built.dataLen > TOAST_TUPLE_THRESHOLD && cols.some((c) => c.type === 'text' && c.ns !== 'null')) {
    built = build(true);
    toasted = true;
  }

  const tupleLen = hoff + built.dataLen;
  const slot = MAXALIGN(tupleLen);
  const runs = [...built.runs];
  if (slot > tupleLen) {
    runs.push({
      key: 'tailpad',
      role: 'pad',
      label: 'MAXALIGN',
      bytes: slot - tupleLen,
      why: `The tuple is stored MAXALIGNed in the page, so pd_upper drops by MAXALIGN(${tupleLen}) = ${slot}. These bytes are never read, and you still write them, cache them and log them.`,
    });
  }
  const perRow = slot + 4;
  return {
    runs,
    total: slot, // what the tuple actually occupies in the page, MAXALIGN included
    pad: runs.filter((r) => r.role === 'pad').reduce((a, r) => a + r.bytes, 0),
    header: hoff,
    perRow,
    rows: Math.floor(PG_PAGE_FREE / perRow),
    toasted,
  };
}

/* ------------------------------------------------------------------ InnoDB */

const INNO_PAGE_FREE = 16384 - 38 - 56 - 26 - 8; // FIL header, page header, infimum+supremum, FIL trailer

function myValueBytes(c: Col, textBytes: number) {
  const info = T[c.type];
  return info.myLen > 0 ? info.myLen : textBytes;
}

function innoLayout(cols: Col[], textBytes: number): Layout {
  const runs: Run[] = [];
  const nullable = cols.filter((c) => c.ns !== 'notnull').length;
  const nullVec = nullable > 0 ? Math.ceil(nullable / 8) : 0;
  const varCols = cols.filter((c) => T[c.type].myLen < 0 && c.ns !== 'null');

  for (const c of [...varCols].reverse()) {
    const len = myValueBytes(c, textBytes);
    const n = len < 128 || T[c.type].myMax < 256 ? 1 : 2;
    runs.push({
      key: `len-${c.key}`,
      role: 'len',
      label: `len(${c.name})`,
      bytes: n,
      why: `Lengths of the variable-length columns are stored in reverse column order, ahead of the record origin. One byte while the value is under 128 bytes; two bytes above that (${len} bytes here), and the spare bits in the second byte are what flag a column stored off-page. Fixed-width columns get no length entry at all, and a NULL column is dropped from this list entirely.`,
    });
  }
  if (nullVec > 0) {
    runs.push({
      key: 'nullvec',
      role: 'nullmap',
      label: 'null vector',
      bytes: nullVec,
      why: `ceil(${nullable}/8) = ${nullVec} byte${nullVec === 1 ? '' : 's'}, one bit per column *declared* nullable — not per column. Declaring a column NOT NULL genuinely shrinks the record, and a table with no nullable columns has no vector at all. InnoDB also allocates this from the declared schema, so unlike Postgres the size does not change row by row.`,
    });
  }
  runs.push({
    key: 'rechdr',
    role: 'hdr',
    label: 'record header',
    bytes: 5,
    why: 'Five bytes in COMPACT and DYNAMIC: delete-mark and min-rec info bits, n_owned for the page directory, a 13-bit heap_no, a 3-bit record type, and a 16-bit signed offset to the next record in key order. That last field is why records on an InnoDB page form a singly linked list rather than needing a sorted slot array.',
  });

  const pk = cols.find((c) => c.pk);
  const emit = (c: Col) => {
    if (c.ns === 'null') return;
    runs.push({
      key: c.key,
      role: 'val',
      label: c.name,
      bytes: myValueBytes(c, textBytes),
      why: `${c.name} ${T[c.type].my}: ${T[c.type].myLen > 0 ? `fixed ${T[c.type].myLen} bytes` : `${textBytes} bytes, length carried in the list above`}. InnoDB inserts no alignment padding anywhere in a record, so reordering the columns of this table changes its size by exactly zero bytes.`,
    });
  };
  if (pk) {
    runs.push({
      key: pk.key,
      role: 'val',
      label: `${pk.name} (key)`,
      bytes: myValueBytes(pk, textBytes),
      why: `The clustered index stores the key columns first, whatever order the table declares them in. This is a clustered-index leaf record, so it *is* the row: there is no separate heap.`,
    });
  }
  runs.push({
    key: 'trxid',
    role: 'sys',
    label: 'DB_TRX_ID',
    bytes: 6,
    why: 'The transaction id that last modified this row — 6 bytes on every clustered-index record, the InnoDB equivalent of t_xmin. Secondary index records do not carry it.',
  });
  runs.push({
    key: 'rollptr',
    role: 'sys',
    label: 'DB_ROLL_PTR',
    bytes: 7,
    why: 'A pointer into the undo log at the previous version of this row. Old versions live in the undo tablespace, not in the table, which is why InnoDB updates in place and Postgres cannot. If the table had no PRIMARY KEY or unique NOT NULL key, a hidden 6-byte DB_ROW_ID would sit here too.',
  });
  for (const c of cols) if (!c.pk) emit(c);

  const total = runs.reduce((a, r) => a + r.bytes, 0);
  // One 2-byte page-directory slot owns 4 to 8 records; 6 is the usual steady state.
  let rows = 0;
  let used = 0;
  while (used + total + (rows % 6 === 0 ? 2 : 0) <= INNO_PAGE_FREE) {
    used += total + (rows % 6 === 0 ? 2 : 0);
    rows++;
  }
  return {
    runs,
    total,
    pad: 0,
    header: runs.filter((r) => r.role !== 'val' && r.role !== 'sys').reduce((a, r) => a + r.bytes, 0),
    perRow: total + 2 / 6,
    rows,
    toasted: false,
  };
}

/* ------------------------------------------------------------------ SQLite */

const SQLITE_PAGE = 4096;
const SQLITE_USABLE = SQLITE_PAGE; // no reserved bytes by default
// Table b-tree leaf: X = U - 35. (Index pages use the ((U-12)*64/255)-23 form.)
const MAX_LOCAL = SQLITE_USABLE - 35;
const MIN_LOCAL = Math.floor(((SQLITE_USABLE - 12) * 32) / 255) - 23;

const varintLen = (n: number) => (n < 128 ? 1 : n < 16384 ? 2 : n < 2097152 ? 3 : 4);

function sqSerial(c: Col, textBytes: number): { serial: number; body: number; why: string } {
  if (c.ns === 'null')
    return { serial: 0, body: 0, why: 'Serial type 0 is NULL: one byte in the header, nothing in the body. SQLite has no null bitmap because the type code already carries it.' };
  if (c.pk)
    return {
      serial: 0,
      body: 0,
      why: 'An INTEGER PRIMARY KEY is an alias for the rowid, so the column is written as serial type 0 (NULL) and the real value lives in the cell’s rowid varint. The column costs one header byte and no body at all.',
    };
  switch (c.type) {
    case 'bool':
      return { serial: 9, body: 0, why: 'Serial types 8 and 9 mean the integers 0 and 1 and carry no body: a true boolean is literally one header byte. SQLite has no BOOLEAN type; the column has INTEGER affinity.' };
    case 'int4':
      return { serial: 3, body: 3, why: 'Integers are narrowed per row to the smallest of 1, 2, 3, 4, 6 or 8 bytes that holds the value (serial types 1–6). 41231 fits in three bytes, so this row stores three — a different row in the same column may store one.' };
    case 'int8':
      return { serial: 1, body: 1, why: 'A bigint column is not a bigint on disk: SQLite is dynamically typed per value, so a small value in a BIGINT column takes one byte.' };
    case 'timestamptz':
      return {
        serial: 13 + 2 * SQLITE_TS_BYTES,
        body: SQLITE_TS_BYTES,
        why: `SQLite has no date or time type. The conventional storage is an ISO-8601 string — '2026-09-13 18:22:41', ${SQLITE_TS_BYTES} bytes — encoded as serial type 13 + 2 × length. Storing it as a Julian-day REAL or a Unix-epoch integer instead would cost 8 bytes or fewer, and lose the offset.`,
      };
    case 'numeric':
      return { serial: 7, body: 8, why: 'NUMERIC affinity with a fractional value stores an IEEE-754 double (serial type 7, 8 bytes). SQLite has no exact decimal type, so numeric(10,2) silently becomes binary floating point.' };
    case 'text':
      return { serial: 13 + 2 * textBytes, body: textBytes, why: `Serial type 13 + 2 × ${textBytes}: odd codes are text, even codes from 12 up are blobs, and the length is baked into the type code rather than stored separately.` };
  }
}

function sqliteLayout(cols: Col[], textBytes: number): Layout {
  const items = cols.map((c) => ({ c, ...sqSerial(c, textBytes) }));
  const inner = items.reduce((a, i) => a + varintLen(i.serial), 0);
  const hdrSize = 1 + inner < 128 ? 1 + inner : 2 + inner;
  const body = items.reduce((a, i) => a + i.body, 0);
  const payload = hdrSize + body;

  let local = payload;
  if (payload > MAX_LOCAL) {
    const surplus = MIN_LOCAL + ((payload - MIN_LOCAL) % (SQLITE_USABLE - 4));
    local = surplus <= MAX_LOCAL ? surplus : MIN_LOCAL;
  }
  const spilled = payload - local;

  const runs: Run[] = [
    {
      key: 'celllen',
      role: 'hdr',
      label: 'payload len',
      bytes: varintLen(payload),
      why: `A big-endian base-128 varint holding the record length (${payload}). Cells grow forward from the cell content area; the 2-byte cell-pointer array at the front of the page is what makes a row addressable.`,
    },
    {
      key: 'rowid',
      role: 'hdr',
      label: 'rowid',
      bytes: 2,
      why: 'The 64-bit rowid as a varint — 2 bytes for this row. The rowid is the b-tree key, so an INTEGER PRIMARY KEY is stored exactly once, here, and never in the record body.',
    },
    {
      key: 'hdrsize',
      role: 'len',
      label: 'hdr size',
      bytes: hdrSize - inner,
      why: `A varint giving the size of the record header including itself (${hdrSize}). A reader parses this, then knows exactly where the bodies start without scanning them.`,
    },
  ];
  for (const i of items) {
    runs.push({
      key: `st-${i.c.key}`,
      role: 'len',
      label: `type(${i.c.name})`,
      bytes: varintLen(i.serial),
      why: `Serial type ${i.serial}. ${i.why}`,
    });
  }
  let budget = local - hdrSize;
  for (const i of items) {
    if (i.body === 0) continue;
    const take = Math.max(0, Math.min(i.body, budget));
    budget -= take;
    if (take === 0) continue;
    runs.push({
      key: `b-${i.c.key}`,
      role: 'val',
      label: take < i.body ? `${i.c.name} (local)` : i.c.name,
      bytes: take,
      why:
        take < i.body
          ? `Only ${take} of ${i.body} bytes fit in the cell. A table b-tree leaf keeps a payload of up to X = usable − 35 = ${MAX_LOCAL} bytes entirely local; past that it keeps M + ((P−M) mod (U−4)) bytes here and spills the rest into a linked list of overflow pages, each spending 4 bytes on the pointer to the next.`
          : i.why,
    });
  }
  if (spilled > 0) {
    runs.push({
      key: 'ovfl',
      role: 'sys',
      label: 'overflow ptr',
      bytes: 4,
      why: `A 4-byte page number for the first overflow page. ${spilled} bytes live out there in a singly linked chain, and reading the column means reading every page in the chain.`,
    });
  }

  const cell = runs.reduce((a, r) => a + r.bytes, 0);
  const perRow = cell + 2; // plus its entry in the cell pointer array
  return {
    runs,
    total: cell,
    pad: 0,
    header: runs.filter((r) => r.role !== 'val').reduce((a, r) => a + r.bytes, 0),
    perRow,
    rows: Math.floor((SQLITE_PAGE - 8) / perRow),
    toasted: spilled > 0,
  };
}

/* ------------------------------------------------------------------ render */

function Cells({ run }: { run: Run }) {
  const tip = useTip();
  const shown = run.bytes <= 14 ? run.bytes : 10;
  const cells = Array.from({ length: shown }, (_, i) => i);
  return (
    <div style={{ marginRight: '.45rem', marginBottom: '.45rem' }}>
      <div
        style={{
          fontSize: '.625rem',
          color: 'var(--viz-ink-2)',
          marginBottom: '2px',
          whiteSpace: 'nowrap',
          overflow: 'hidden',
          textOverflow: 'ellipsis',
          maxWidth: '11rem',
        }}
      >
        {run.label}
      </div>
      <div
        {...tip(
          <>
            <strong>
              {run.label} — {run.bytes} byte{run.bytes === 1 ? '' : 's'}
            </strong>
            <br />
            {ROLE_LABEL[run.role]}
            <br />
            <span style={{ color: 'var(--viz-ink-2)' }}>{run.why}</span>
          </>,
        )}
        style={{ display: 'flex', gap: '2px', alignItems: 'flex-end', cursor: 'help' }}
      >
        {cells.map((i) => (
          <span
            key={i}
            style={{
              display: 'inline-block',
              width: run.role === 'pad' ? '.85rem' : '.9rem',
              height: '1.15rem',
              borderRadius: '3px',
              border: run.role === 'pad' ? '1px dashed var(--viz-stale)' : '1px solid var(--viz-border)',
              borderBottom: `3px solid ${ROLE_COLOR[run.role]}`,
              background:
                run.role === 'pad'
                  ? 'var(--viz-plane)'
                  : `color-mix(in srgb, ${ROLE_COLOR[run.role]} 22%, var(--viz-plane))`,
            }}
          />
        ))}
        {run.bytes > shown ? (
          <span
            style={{
              fontSize: '.625rem',
              color: 'var(--viz-ink-2)',
              fontVariantNumeric: 'tabular-nums',
              paddingLeft: '.2rem',
            }}
          >
            +{run.bytes - shown}
          </span>
        ) : null}
      </div>
    </div>
  );
}

const NEXT: Record<NullState, NullState> = { notnull: 'nullable', nullable: 'null', null: 'notnull' };
const NS_LABEL: Record<NullState, string> = { notnull: 'NOT NULL', nullable: 'null-able', null: 'IS NULL' };

export default function TupleByteLayoutStudio() {
  const [engine, setEngine] = useState<Engine>('pg');
  const [cols, setCols] = useState<Col[]>(BASE);
  const [spare, setSpare] = useState(0);
  const [textIdx, setTextIdx] = useState(0);
  const [ref, width] = useSize(760);

  const textBytes = TEXT_SIZES[textIdx];

  const all = useMemo<Col[]>(
    () => [
      ...cols,
      ...Array.from({ length: spare }, (_, i) => ({
        key: `spare${i}`,
        name: `tag_${i + 1}`,
        type: 'int4' as TypeKey,
        ns: 'notnull' as NullState,
      })),
    ],
    [cols, spare],
  );

  const pg = useMemo(() => pgLayout(all, textBytes), [all, textBytes]);
  const inno = useMemo(() => innoLayout(all, textBytes), [all, textBytes]);
  const sq = useMemo(() => sqliteLayout(all, textBytes), [all, textBytes]);
  const lay = engine === 'pg' ? pg : engine === 'innodb' ? inno : sq;

  const move = (i: number, d: -1 | 1) =>
    setCols((cur) => {
      const j = i + d;
      if (j < 0 || j >= cur.length) return cur;
      const next = [...cur];
      [next[i], next[j]] = [next[j], next[i]];
      return next;
    });

  const cycle = (i: number) =>
    setCols((cur) => cur.map((c, k) => (k === i && !c.pk ? { ...c, ns: NEXT[c.ns] } : c)));

  const packed = useMemo(() => {
    const rank = (c: Col) => (T[c.type].pgLen > 0 ? 100 - T[c.type].pgAlign * 10 - T[c.type].pgLen : 900);
    return [...BASE].sort((a, b) => rank(a) - rank(b));
  }, []);
  const isPacked = cols.map((c) => c.key).join() === packed.map((c) => c.key).join();

  const bestPg = useMemo(() => pgLayout([...packed, ...all.slice(cols.length)], textBytes), [packed, all, cols.length, textBytes]);

  const note = (() => {
    if (engine === 'pg' && pg.toasted)
      return `title is ${fmtNum(textBytes)} bytes, so the tuple passed TOAST_TUPLE_THRESHOLD (2048) and the attribute left the heap: what stays behind is an 18-byte varatt_external pointer, and every read of title becomes a second lookup in the TOAST relation.`;
    if (engine === 'pg' && textBytes > 126)
      return `At ${textBytes} bytes, title no longer fits a 1-byte short varlena header: it now carries a 4-byte int32 header and must start 4-byte aligned, so crossing 126 bytes cost 3 header bytes plus whatever padding the alignment demanded.`;
    if (engine === 'pg' && pg.pad > 0)
      return `${pg.pad} of ${pg.total} bytes in this tuple are alignment padding that no query will ever read. Widest-first ordering gets the row to ${bestPg.total} bytes and ${fmtNum(bestPg.rows)} rows per page — the same columns, the same data, ${fmtNum(bestPg.rows - pg.rows)} more rows on every 8 KB page.`;
    if (engine === 'pg')
      return `Zero padding: every attribute already starts on its typalign boundary. The 8-byte MAXALIGN of the whole tuple is the only rounding left, and the 24-byte header plus the 4-byte line pointer is the floor Postgres charges for any row at all.`;
    if (engine === 'innodb')
      return `InnoDB inserts no padding, so column order changes this record by exactly zero bytes — but it spends ${inno.header} bytes on the length list, null vector and record header and another 13 on DB_TRX_ID and DB_ROLL_PTR, because the clustered-index record carries its own MVCC pointer into the undo log.`;
    return sq.toasted
      ? `The record is ${sq.total} bytes in the cell plus an overflow chain: a table leaf holds a payload of up to usable − 35 = ${MAX_LOCAL} bytes locally, and spills the remainder into 4-byte-linked overflow pages.`
      : `SQLite spends nothing on transaction ids and nothing on padding: the whole record header is ${sq.header} bytes of varints, and a NULL, a 0, a 1 and an INTEGER PRIMARY KEY each cost one header byte and no body.`;
  })();

  const cellCols = width >= 700 ? 2 : 1;

  return (
    <VizPanel
      title="The same row, byte for byte, in three engines"
      subtitle="Reorder the columns, toggle a value to NULL, grow the text and watch the row size, the padding and the rows-per-page move. Hover any run of bytes to see what it encodes."
      controls={
        <>
          <Segmented
            label="Engine"
            value={engine}
            onChange={setEngine}
            options={[
              { value: 'pg', label: 'PostgreSQL', title: 'Heap tuple, 8 KB page' },
              { value: 'innodb', label: 'InnoDB', title: 'DYNAMIC clustered-index record, 16 KB page' },
              { value: 'sqlite', label: 'SQLite', title: 'Table b-tree leaf cell, 4 KB page' },
            ]}
          />
          <Choice
            label="title value"
            value={String(textIdx)}
            onChange={(v) => setTextIdx(Number(v))}
            options={TEXT_SIZES.map((n, i) => ({ value: String(i), label: `${fmtNum(n)} bytes` }))}
          />
          <Slider label="Spare int columns" min={0} max={6} value={spare} onChange={setSpare} format={(n) => `+${n}`} />
          <Button onClick={() => setCols(packed)} disabled={isPacked} primary>
            Pack widest-first
          </Button>
          <Button onClick={() => setCols(BASE)} disabled={cols === BASE}>
            Reset order
          </Button>
        </>
      }
      legend={
        <Legend
          items={(['val', 'hdr', 'nullmap', 'len', 'sys', 'pad'] as Role[]).map((r) => ({
            label: ROLE_LABEL[r],
            color: ROLE_COLOR[r],
          }))}
        />
      }
      stats={
        <Stats
          items={[
            {
              label: 'Bytes per row',
              value: `${fmtNum(lay.total)} B`,
              hint:
                engine === 'pg'
                  ? 'The tuple as it sits in the page: header, null bitmap, attributes, padding and the trailing MAXALIGN'
                  : 'The record itself, headers included',
            },
            { label: 'Overhead before values', value: `${fmtNum(lay.header)} B`, hint: 'Header, null map and length bytes you pay on every row' },
            { label: 'Alignment padding', value: `${lay.pad} B`, hint: 'Bytes stored, cached and logged that no query can read' },
            {
              label: 'Rows per page',
              value: fmtNum(lay.rows),
              hint:
                engine === 'pg'
                  ? '8192 − 24 byte page header, each row costing MAXALIGN(tuple) + a 4-byte line pointer, fillfactor 100'
                  : engine === 'innodb'
                    ? '16 KB page less FIL header, page header, infimum/supremum and trailer; one 2-byte directory slot per ~6 records'
                    : '4096 − 8 byte leaf header, each cell costing its bytes plus a 2-byte cell pointer',
            },
            {
              label: 'Best Postgres order',
              value: `${fmtNum(bestPg.total)} B / ${fmtNum(bestPg.rows)} rows`,
              hint: 'What widest-alignment-first ordering would give you for the same columns',
            },
          ]}
        />
      }
      note={<Note>{note}</Note>}
      table={
        <>
          <table className="viz-table">
            <thead>
              <tr>
                <th>#</th>
                <th>Column</th>
                <th>Postgres type</th>
                <th>typalign</th>
                <th>State</th>
                <th>PG bytes</th>
                <th>PG padding</th>
                <th>InnoDB bytes</th>
                <th>SQLite bytes</th>
              </tr>
            </thead>
            <tbody>
              {all.map((c, i) => {
                const pgRuns = pg.runs.filter((r) => r.key === c.key || r.key === `${c.key}-h`);
                const padRun = pg.runs.find((r) => r.key === `${c.key}-pad`);
                const s = sqSerial(c, textBytes);
                const iv = c.ns === 'null' ? 0 : myValueBytes(c, textBytes);
                return (
                  <tr key={c.key}>
                    <td>{i + 1}</td>
                    <td>{c.name}</td>
                    <td>{T[c.type].pg}</td>
                    <td>{T[c.type].pgLen > 0 ? T[c.type].pgAlign : `${T[c.type].pgAlign} (varlena)`}</td>
                    <td>{c.pk ? 'PRIMARY KEY' : NS_LABEL[c.ns]}</td>
                    <td>{pgRuns.reduce((a, r) => a + r.bytes, 0)}</td>
                    <td>{padRun ? padRun.bytes : 0}</td>
                    <td>{iv}</td>
                    <td>{varintLen(s.serial) + s.body}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
          <table className="viz-table">
            <thead>
              <tr>
                <th>Engine</th>
                <th>Page</th>
                <th>Record bytes</th>
                <th>Fixed overhead</th>
                <th>Padding</th>
                <th>Per-row page cost</th>
                <th>Rows per page</th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <td>PostgreSQL heap tuple</td>
                <td>8 KB</td>
                <td>{pg.total}</td>
                <td>{pg.header}</td>
                <td>{pg.pad}</td>
                <td>{pg.perRow}</td>
                <td>{fmtNum(pg.rows)}</td>
              </tr>
              <tr>
                <td>InnoDB DYNAMIC record</td>
                <td>16 KB</td>
                <td>{inno.total}</td>
                <td>{inno.header + 13}</td>
                <td>0</td>
                <td>{inno.perRow.toFixed(1)}</td>
                <td>{fmtNum(inno.rows)}</td>
              </tr>
              <tr>
                <td>SQLite leaf cell</td>
                <td>4 KB</td>
                <td>{sq.total}</td>
                <td>{sq.header}</td>
                <td>0</td>
                <td>{sq.perRow}</td>
                <td>{fmtNum(sq.rows)}</td>
              </tr>
            </tbody>
          </table>
        </>
      }
    >
      <div ref={ref}>
        <TooltipHost>
          <div style={{ minWidth: '280px' }}>
            <div
              style={{
                display: 'grid',
                gridTemplateColumns: cellCols === 2 ? '1fr 1fr' : '1fr',
                gap: '.15rem 1.25rem',
                marginBottom: '.7rem',
              }}
            >
              {cols.map((c, i) => (
                <div
                  key={c.key}
                  style={{
                    display: 'flex',
                    alignItems: 'center',
                    gap: '.35rem',
                    fontSize: '.75rem',
                    padding: '.1rem 0',
                    borderBottom: '1px solid var(--viz-border)',
                  }}
                >
                  <button type="button" onClick={() => move(i, -1)} disabled={i === 0} title="Move earlier">
                    ↑
                  </button>
                  <button type="button" onClick={() => move(i, 1)} disabled={i === cols.length - 1} title="Move later">
                    ↓
                  </button>
                  <span style={{ color: 'var(--viz-ink)', fontWeight: 600, minWidth: '5.5rem' }}>{c.name}</span>
                  <span style={{ color: 'var(--viz-ink-2)', minWidth: '7rem' }}>
                    {engine === 'pg' ? T[c.type].pg : engine === 'innodb' ? T[c.type].my : T[c.type].sq}
                  </span>
                  <button
                    type="button"
                    onClick={() => cycle(i)}
                    disabled={!!c.pk}
                    title={
                      c.pk
                        ? 'The primary key cannot be null'
                        : 'Cycle NOT NULL → null-able → IS NULL: declared nullability sizes InnoDB’s vector, an actual NULL creates Postgres’s bitmap'
                    }
                  >
                    {c.pk ? 'PRIMARY KEY' : NS_LABEL[c.ns]}
                  </button>
                </div>
              ))}
              {spare > 0 ? (
                <div style={{ fontSize: '.75rem', color: 'var(--viz-ink-2)', padding: '.25rem 0' }}>
                  + {spare} spare integer column{spare === 1 ? '' : 's'} (NOT NULL), appended
                </div>
              ) : null}
            </div>

            <div style={{ display: 'flex', flexWrap: 'wrap', alignItems: 'flex-end' }}>
              {lay.runs.map((r) => (
                <Cells key={r.key} run={r} />
              ))}
            </div>
          </div>
        </TooltipHost>
      </div>
    </VizPanel>
  );
}
