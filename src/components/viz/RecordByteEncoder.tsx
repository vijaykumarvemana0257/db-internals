import { useMemo, useState } from 'react';
import {
  VizPanel,
  Slider,
  Segmented,
  Choice,
  Check,
  Legend,
  Stats,
  TooltipHost,
  useTip,
  useSize,
} from './Viz';

/**
 * One record, four encodings, byte for byte.
 *
 * Everything here is the real format:
 *  - SQLite record format: big-endian base-128 varint header (size + one serial
 *    type per column), then the bodies. Serial types 8/9 store the integers 0
 *    and 1 in the type code itself; text is 13 + 2*len with no terminator.
 *  - Protobuf (proto3): key = (field_number << 3) | wire_type, LEB128 varints,
 *    fixed64 little-endian, scalar fields at their default are omitted entirely.
 *  - JSON: UTF-8 text, field names in every record, no NaN and no -0.
 *  - Native struct: a null bitmap, natural alignment padding, and whichever byte
 *    order the machine that wrote it happened to use.
 */

/* ------------------------------------------------------------------ model */

type Role = 'header' | 'tag' | 'payload' | 'len' | 'delim' | 'pad';

const ROLE_COLOR: Record<Role, string> = {
  header: 'var(--viz-1)',
  tag: 'var(--viz-2)',
  payload: 'var(--viz-3)',
  len: 'var(--viz-4)',
  delim: 'var(--viz-5)',
  pad: 'var(--viz-stale)',
};

const ROLE_LABEL: Record<Role, string> = {
  header: 'header / type code',
  tag: 'field tag or key name',
  payload: 'value payload',
  len: 'length prefix',
  delim: 'delimiter / structure',
  pad: 'alignment padding',
};

type Byte = { v: number; role: Role; field: string; why: string };

type Rec = {
  id: number;
  name: string;
  score: number;
  scoreLabel: string;
  flag: 'true' | 'false' | 'null';
};

/* -------------------------------------------------------------- encoders */

const TE = new TextEncoder();
const utf8 = (s: string) => Array.from(TE.encode(s));
const hex = (n: number) => n.toString(16).toUpperCase().padStart(2, '0');

/** Protobuf / LEB128: base-128, little-endian groups, 64-bit two's complement. */
function protoVarint(x: bigint): number[] {
  let v = BigInt.asUintN(64, x);
  const out: number[] = [];
  do {
    const b = Number(v & 0x7fn);
    v >>= 7n;
    out.push(b | (v > 0n ? 0x80 : 0));
  } while (v > 0n);
  return out;
}

/** SQLite varint: base-128 but BIG-endian groups, 1-9 bytes. */
function sqliteVarint(n: number): number[] {
  if (n < 0x80) return [n];
  const groups: number[] = [];
  let v = n;
  while (v > 0) {
    groups.unshift(v % 128);
    v = Math.floor(v / 128);
  }
  return groups.map((g, i) => (i < groups.length - 1 ? g | 0x80 : g));
}

function intBytes(v: number, width: number, little: boolean): number[] {
  let x = BigInt.asUintN(width * 8, BigInt(v));
  const be: number[] = [];
  for (let i = 0; i < width; i++) {
    be.unshift(Number(x & 0xffn));
    x >>= 8n;
  }
  return little ? be.slice().reverse() : be;
}

function f64Bytes(v: number, little: boolean): number[] {
  const buf = new ArrayBuffer(8);
  new DataView(buf).setFloat64(0, v, little);
  return Array.from(new Uint8Array(buf));
}

/** SQLite serial type for an integer, narrowest that fits. */
function intSerial(v: number): { t: number; width: number } {
  if (v === 0) return { t: 8, width: 0 };
  if (v === 1) return { t: 9, width: 0 };
  if (v >= -128 && v <= 127) return { t: 1, width: 1 };
  if (v >= -32768 && v <= 32767) return { t: 2, width: 2 };
  if (v >= -8388608 && v <= 8388607) return { t: 3, width: 3 };
  if (v >= -2147483648 && v <= 2147483647) return { t: 4, width: 4 };
  if (v >= -140737488355328 && v <= 140737488355327) return { t: 5, width: 6 };
  return { t: 6, width: 8 };
}

function varintWhy(b: number, i: number, n: number, label: string) {
  const more = (b & 0x80) !== 0;
  return `${label}: varint byte ${i + 1}/${n} — continuation bit ${more ? '1 (more bytes follow)' : '0 (last byte)'}, 7 payload bits ${(b & 0x7f).toString(2).padStart(7, '0')}`;
}

function utf8Why(b: number, i: number, n: number, s: string) {
  const what =
    b < 0x80
      ? `ASCII '${String.fromCharCode(b)}'`
      : b >= 0xc0
        ? 'lead byte of a multi-byte UTF-8 code point'
        : 'continuation byte (10xxxxxx) of a multi-byte code point';
  return `"${s}": UTF-8 byte ${i + 1}/${n} — ${what}. Length lives in a prefix, not in a terminator, so a NUL byte is legal data here.`;
}

function encSqlite(r: Rec): Byte[] {
  type Col = { key: string; t: number; role: Role; typeWhy: string; body: Byte[] };
  const cols: Col[] = [];

  const si = intSerial(r.id);
  cols.push({
    key: 'id',
    t: si.t,
    role: 'header',
    typeWhy:
      si.width === 0
        ? `serial type ${si.t}: the integer ${r.id} IS the type code — zero payload bytes in the body`
        : `serial type ${si.t}: ${si.width}-byte big-endian two's-complement integer, the narrowest width that holds ${r.id}`,
    body: intBytes(r.id, si.width, false).map((b, i) => ({
      v: b,
      role: 'payload' as Role,
      field: 'id',
      why: `id = ${r.id}: big-endian byte ${i + 1}/${si.width}. SQLite stores the value in the narrowest of 1, 2, 3, 4, 6 or 8 bytes.`,
    })),
  });

  const nb = utf8(r.name);
  cols.push({
    key: 'name',
    t: 13 + 2 * nb.length,
    role: 'len',
    typeWhy: `serial type ${13 + 2 * nb.length} = 13 + 2x${nb.length}: TEXT of ${nb.length} bytes. Odd type codes >= 13 are text; the length rides inside the type code, so there is no separate length field and no terminator.`,
    body: nb.map((b, i) => ({
      v: b,
      role: 'payload' as Role,
      field: 'name',
      why: utf8Why(b, i, nb.length, r.name),
    })),
  });

  if (Number.isNaN(r.score)) {
    cols.push({
      key: 'score',
      t: 0,
      role: 'header',
      typeWhy:
        'serial type 0 = NULL. SQLite has no way to store NaN in a REAL column, so it silently converts NaN to NULL on the way in — the value you read back is not the value you wrote.',
      body: [],
    });
  } else {
    cols.push({
      key: 'score',
      t: 7,
      role: 'header',
      typeWhy: 'serial type 7: IEEE-754 binary64, 8 bytes, big-endian — always, on every platform',
      body: f64Bytes(r.score, false).map((b, i) => ({
        v: b,
        role: 'payload' as Role,
        field: 'score',
        why: `score = ${r.scoreLabel}: IEEE-754 binary64, big-endian byte ${i + 1}/8${i === 0 ? ` — the top bit of this byte is the sign bit (${b & 0x80 ? '1, negative' : '0, positive'})` : ''}`,
      })),
    });
  }

  if (r.flag === 'null') {
    cols.push({ key: 'flag', t: 0, role: 'header', typeWhy: 'serial type 0 = NULL, zero payload bytes', body: [] });
  } else if (r.flag === 'true') {
    cols.push({
      key: 'flag',
      t: 9,
      role: 'header',
      typeWhy: 'serial type 9: the integer 1, stored with zero payload bytes. Booleans cost one header byte and nothing else.',
      body: [],
    });
  } else {
    cols.push({
      key: 'flag',
      t: 8,
      role: 'header',
      typeWhy: 'serial type 8: the integer 0, stored with zero payload bytes',
      body: [],
    });
  }

  const typeVarints = cols.map((c) => sqliteVarint(c.t));
  const serialLen = typeVarints.reduce((a, b) => a + b.length, 0);
  let n = 1;
  let sizeV = sqliteVarint(serialLen + 1);
  while (sizeV.length !== n) {
    n = sizeV.length;
    sizeV = sqliteVarint(serialLen + n);
  }

  const out: Byte[] = sizeV.map((b, i) => ({
    v: b,
    role: 'header' as Role,
    field: 'hdr size',
    why: `header size varint = ${serialLen + sizeV.length} bytes, and the count includes this varint itself. Byte ${i + 1}/${sizeV.length}. A reader uses it to find where the body starts without parsing the serial types.`,
  }));

  cols.forEach((c, ci) => {
    typeVarints[ci].forEach((b) =>
      out.push({ v: b, role: c.role, field: `${c.key} type`, why: c.typeWhy }),
    );
  });
  cols.forEach((c) => out.push(...c.body));
  return out;
}

function encProto(r: Rec, fixedInts: boolean): Byte[] {
  const out: Byte[] = [];

  if (r.id !== 0) {
    const wt = fixedInts ? 1 : 0;
    const key = (1 << 3) | wt;
    out.push({
      v: key,
      role: 'tag',
      field: 'id tag',
      why: `key byte 0x${hex(key)} = (field number 1 << 3) | wire type ${wt} (${wt === 0 ? 'VARINT' : 'I64'}). Field numbers 1-15 fit in this single byte; 16-2047 cost two — which is why hot fields get low numbers.`,
    });
    if (fixedInts) {
      intBytes(r.id, 8, true).forEach((b, i) =>
        out.push({
          v: b,
          role: 'payload',
          field: 'id',
          why: `id = ${r.id} as fixed64: always 8 bytes, always little-endian, regardless of the machine. Byte ${i + 1}/8.`,
        }),
      );
    } else {
      const vb = protoVarint(BigInt(r.id));
      vb.forEach((b, i) =>
        out.push({
          v: b,
          role: 'payload',
          field: 'id',
          why:
            varintWhy(b, i, vb.length, `id = ${r.id}`) +
            (r.id < 0
              ? '. A negative int64 is sign-extended to 64 bits before encoding, so it always costs 10 bytes — sint64 (zigzag) would cost 1.'
              : ''),
        }),
      );
    }
  }

  const nb = utf8(r.name);
  if (nb.length > 0) {
    out.push({
      v: 0x12,
      role: 'tag',
      field: 'name tag',
      why: 'key byte 0x12 = (field number 2 << 3) | wire type 2 (LEN). The field name "name" never appears on the wire — renaming the field in the .proto costs nothing, renumbering it is a data-corrupting change.',
    });
    const lv = protoVarint(BigInt(nb.length));
    lv.forEach((b, i) =>
      out.push({
        v: b,
        role: 'len',
        field: 'name len',
        why: varintWhy(b, i, lv.length, `length = ${nb.length} bytes`) + '. Strings under 128 bytes get a 1-byte length prefix.',
      }),
    );
    nb.forEach((b, i) =>
      out.push({ v: b, role: 'payload', field: 'name', why: utf8Why(b, i, nb.length, r.name) }),
    );
  }

  // Proto3 omits a scalar at its default. For a double the implementations
  // (C++ doubleToRawLongBits-style memcmp, Java doubleToRawLongBits, Go's
  // "== 0 && !Signbit") compare the RAW BITS, not the value — so -0.0 and NaN
  // are written, and only +0.0 is treated as absent.
  if (!Object.is(r.score, 0)) {
    out.push({
      v: 0x19,
      role: 'tag',
      field: 'score tag',
      why: `key byte 0x19 = (field number 3 << 3) | wire type 1 (I64). A double is never varint-encoded — its high bits are the exponent, so varints would make it bigger, not smaller.${Object.is(r.score, -0) ? ' The field survives the proto3 default check because implementations compare the raw 64 bits, not the value: -0.0 has a bit set, so it is not "zero".' : ''}`,
    });
    f64Bytes(r.score, true).forEach((b, i) =>
      out.push({
        v: b,
        role: 'payload',
        field: 'score',
        why: `score = ${r.scoreLabel}: IEEE-754 binary64, little-endian byte ${i + 1}/8 (the spec pins little-endian, so the bytes are identical on ARM and on s390x)`,
      }),
    );
  }

  if (r.flag === 'true') {
    out.push({
      v: 0x20,
      role: 'tag',
      field: 'flag tag',
      why: 'key byte 0x20 = (field number 4 << 3) | wire type 0 (VARINT). A bool is carried as a varint: the encoder writes 1, and a decoder reads any non-zero varint as true.',
    });
    out.push({ v: 1, role: 'payload', field: 'flag', why: 'bool true = varint 1' });
  }

  return out;
}

function encJson(r: Rec): Byte[] {
  type Piece = { text: string; role: Role; field: string; why: string };
  const pieces: Piece[] = [];
  const num = (v: number) => (Number.isFinite(v) ? String(v) : 'null');

  pieces.push({
    text: '{',
    role: 'delim',
    field: '{',
    why: 'structural byte. JSON is delimited, not length-prefixed: a reader cannot skip a value without scanning it, and cannot know the record length without reaching the end.',
  });

  const fields: [string, string, string][] = [
    [
      'id',
      String(r.id),
      `the integer is written as decimal ASCII, one byte per digit. JSON has a single "number" type with no integer/float distinction, so anything above 2^53 loses precision in most parsers — the reason APIs ship an id and an id_str.`,
    ],
    ['name', JSON.stringify(r.name), 'string value, quoted and escaped; non-ASCII is emitted as raw UTF-8'],
    [
      'score',
      num(r.score),
      Number.isFinite(r.score)
        ? `the double is printed as the shortest decimal that round-trips${Object.is(r.score, -0) ? ' — and the sign of negative zero is lost: JSON.stringify(-0) is "0"' : ''}`
        : 'JSON has no NaN and no Infinity. Every conforming writer turns them into null, so the value silently changes type on the way out.',
    ],
    [
      'flag',
      r.flag,
      'null is a first-class JSON value, so JSON can say "present but null" — a distinction a proto3 scalar cannot express',
    ],
  ];

  fields.forEach(([k, v, why], i) => {
    pieces.push({
      text: `"${k}"`,
      role: 'tag',
      field: `${k} key`,
      why: `the field name is stored in full, in every single record: ${k.length + 2} bytes of schema repeated on every row. This is what makes JSON self-describing and what makes it expensive.`,
    });
    pieces.push({ text: ':', role: 'delim', field: ':', why: 'structural byte' });
    pieces.push({ text: v, role: 'payload', field: k, why });
    if (i < fields.length - 1) pieces.push({ text: ',', role: 'delim', field: ',', why: 'structural byte' });
  });

  pieces.push({ text: '}', role: 'delim', field: '}', why: 'structural byte — the only thing marking the end of the record' });

  const out: Byte[] = [];
  pieces.forEach((p) => {
    const bs = utf8(p.text);
    bs.forEach((b, i) =>
      out.push({
        v: b,
        role: p.role,
        field: p.field,
        why: `${p.text.length > 24 ? p.text.slice(0, 24) + '...' : p.text}${bs.length > 1 ? ` (byte ${i + 1}/${bs.length})` : ''} - ${p.why}`,
      }),
    );
  });
  return out;
}

function encStruct(r: Rec, little: boolean, aligned: boolean): Byte[] {
  const out: Byte[] = [];
  const put = (bytes: number[], role: Role, field: string, why: (i: number) => string) =>
    bytes.forEach((b, i) => out.push({ v: b, role, field, why: why(i) }));
  const align = (a: number) => {
    if (!aligned) return;
    while (out.length % a !== 0) {
      out.push({
        v: 0,
        role: 'pad',
        field: 'padding',
        why: `alignment padding. The next field must start at an offset divisible by ${a} or the CPU cannot load it with a single aligned access, so the compiler inserts a dead byte at offset ${out.length}. You store it, you cache it, you write it to disk, and it means nothing.`,
      });
    }
  };

  put([r.flag === 'null' ? 1 << 3 : 0], 'header', 'null map', () =>
    `null bitmap, one bit per column (Postgres calls this t_bits and only writes it when some column is null). Bit 3 = flag is ${r.flag === 'null' ? 'NULL' : 'not null'}. Without it, a fixed-width row has no way to say "absent".`,
  );
  put([r.flag === 'true' ? 1 : 0], 'payload', 'flag', () =>
    'a bool costs a whole byte, and when the column is NULL the byte is still written — it just holds garbage that the null bitmap tells you to ignore',
  );
  align(8);
  put(intBytes(r.id, 8, little), 'payload', 'id', (i) =>
    `id = ${r.id} as int64: ${little ? 'little' : 'big'}-endian byte ${i + 1}/8. ${little ? 'Little-endian puts the least significant byte first — x86 and ARM.' : 'Big-endian puts the most significant byte first — network byte order, SPARC, s390x.'} Fixed width means ${r.id} and 9223372036854775807 cost exactly the same.`,
  );
  put(f64Bytes(r.score, little), 'payload', 'score', (i) =>
    `score = ${r.scoreLabel}: IEEE-754 binary64, ${little ? 'little' : 'big'}-endian byte ${i + 1}/8`,
  );
  const nb = utf8(r.name);
  align(2);
  put(intBytes(nb.length, 2, little), 'len', 'name_len', (i) =>
    `uint16 length prefix = ${nb.length}, ${little ? 'little' : 'big'}-endian byte ${i + 1}/2. Length-prefixed, so the string may contain any byte including 0x00.`,
  );
  put(nb, 'payload', 'name', (i) => utf8Why(nb[i], i, nb.length, r.name));
  if (aligned) {
    while (out.length % 8 !== 0) {
      out.push({
        v: 0,
        role: 'pad',
        field: 'tail pad',
        why: 'trailing padding, so that an array of these structs keeps every element 8-byte aligned. sizeof() is always a multiple of the widest member.',
      });
    }
  }
  return out;
}

/* ------------------------------------------------------------- inventory */

const IDS = [
  0, 1, 42, 127, 128, 255, 300, 16383, 16384, 65535, 1000000, 2147483647, -1, -42, -129,
];

const NAMES = [
  { value: 'ada', label: '"ada" (3 ASCII bytes)' },
  { value: '', label: '"" (empty string)' },
  { value: 'Ada Lovelace', label: '"Ada Lovelace" (12 bytes)' },
  { value: 'Ada ☕', label: '"Ada ☕" (7 bytes, multi-byte UTF-8)' },
];

const SCORES = [
  { value: '3.5', n: 3.5 },
  { value: '0.1', n: 0.1 },
  { value: '0.0', n: 0 },
  { value: '-0.0', n: -0 },
  { value: 'NaN', n: NaN },
  { value: 'Infinity', n: Infinity },
];

type LaneKey = 'sqlite' | 'proto' | 'json' | 'struct';

/* ---------------------------------------------------------------- render */

function splitRuns(bytes: Byte[]) {
  const out: { field: string; start: number; items: { b: Byte; i: number }[] }[] = [];
  bytes.forEach((b, i) => {
    const last = out[out.length - 1];
    if (last && last.field === b.field) last.items.push({ b, i });
    else out.push({ field: b.field, start: i, items: [{ b, i }] });
  });
  return out;
}

function Lane({
  title,
  sub,
  bytes,
  empty,
}: {
  title: string;
  sub: string;
  bytes: Byte[];
  empty: string;
}) {
  const tip = useTip();
  const runs = useMemo(() => splitRuns(bytes), [bytes]);

  return (
    <div>
      <div
        style={{
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'baseline',
          gap: '.5rem',
          fontSize: '.8125rem',
        }}
      >
        <strong style={{ color: 'var(--viz-ink)' }}>{title}</strong>
        <span style={{ color: 'var(--viz-ink-2)', fontVariantNumeric: 'tabular-nums', whiteSpace: 'nowrap' }}>
          {bytes.length} bytes
        </span>
      </div>
      <div style={{ fontSize: '.6875rem', color: 'var(--viz-ink-2)', margin: '.15rem 0 .4rem' }}>{sub}</div>

      {bytes.length === 0 ? (
        <div style={{ fontSize: '.75rem', color: 'var(--viz-ink-2)', padding: '.5rem 0' }}>{empty}</div>
      ) : (
        <div style={{ display: 'flex', flexWrap: 'wrap', gap: '.4rem .55rem' }}>
          {runs.map((run) => (
            <div key={run.start} style={{ maxWidth: '100%' }}>
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
                {run.field}
              </div>
              <div style={{ display: 'flex', flexWrap: 'wrap', gap: '2px' }}>
                {run.items.map(({ b, i }, k) => (
                  <span
                    key={i}
                    {...tip(
                      <>
                        <strong>
                          0x{hex(b.v)} at offset {i}
                        </strong>
                        <br />
                        {ROLE_LABEL[b.role]} &middot; {b.field}
                        <br />
                        <span style={{ color: 'var(--viz-ink-2)' }}>{b.why}</span>
                      </>,
                    )}
                    tabIndex={k === 0 ? 0 : -1}
                    style={{
                      display: 'inline-block',
                      minWidth: '1.9rem',
                      textAlign: 'center',
                      padding: '.12rem .15rem',
                      fontSize: '.6875rem',
                      lineHeight: 1.4,
                      color: 'var(--viz-ink)',
                      fontVariantNumeric: 'tabular-nums',
                      background: `color-mix(in srgb, ${ROLE_COLOR[b.role]} 16%, var(--viz-plane))`,
                      border: '1px solid var(--viz-border)',
                      borderBottom: `3px solid ${ROLE_COLOR[b.role]}`,
                      borderRadius: '4px',
                      cursor: 'help',
                    }}
                  >
                    {hex(b.v)}
                  </span>
                ))}
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

export default function RecordByteEncoder() {
  const [idIdx, setIdIdx] = useState(2);
  const [name, setName] = useState(NAMES[0].value);
  const [scoreKey, setScoreKey] = useState(SCORES[0].value);
  const [flag, setFlag] = useState<'true' | 'false' | 'null'>('true');
  const [ints, setInts] = useState<'varint' | 'fixed'>('varint');
  const [endian, setEndian] = useState<'le' | 'be'>('le');
  const [aligned, setAligned] = useState(true);
  const [ref, width] = useSize(720);

  const score = SCORES.find((s) => s.value === scoreKey)!;
  const rec: Rec = { id: IDS[idIdx], name, score: score.n, scoreLabel: score.value, flag };

  const lanes = useMemo(() => {
    const little = endian === 'le';
    return {
      sqlite: encSqlite(rec),
      proto: encProto(rec, ints === 'fixed'),
      json: encJson(rec),
      struct: encStruct(rec, little, aligned),
    } as Record<LaneKey, Byte[]>;
  }, [rec.id, rec.name, rec.scoreLabel, rec.flag, ints, endian, aligned]);

  const count = (k: LaneKey, role: Role) => lanes[k].filter((b) => b.role === role).length;
  const padBytes = count('struct', 'pad');

  const META: { key: LaneKey; title: string; sub: string; empty: string }[] = [
    {
      key: 'sqlite',
      title: 'SQLite record format',
      sub: 'Big-endian by specification. The byte-order toggle does nothing here — that is the point of an on-disk format.',
      empty: 'empty',
    },
    {
      key: 'proto',
      title: 'Protobuf (proto3)',
      sub: 'Tag + wire type per field. Varints are base-128 little-endian; fixed64 is little-endian. Byte order is pinned by the spec.',
      empty: 'Zero bytes: every field is at its proto3 default, so nothing is written at all. A reader cannot tell "absent" from "zero".',
    },
    {
      key: 'json',
      title: 'JSON text (UTF-8)',
      sub: 'Self-describing: the schema is repeated in every record. Text, so endianness never applies.',
      empty: 'empty',
    },
    {
      key: 'struct',
      title: 'Native fixed-width row',
      sub: `${endian === 'le' ? 'Little-endian (x86, ARM)' : 'Big-endian (s390x, network byte order)'} - ${aligned ? 'natural alignment' : 'packed, no padding'}. Nothing in the bytes records either choice.`,
      empty: 'empty',
    },
  ];

  const cols = width >= 660 ? 2 : 1;

  const notes: string[] = [];
  if (rec.id === 128)
    notes.push('id crossed 127: a base-128 varint needs a second byte from 128 upward, while the fixed-width row did not move at all.');
  if (rec.id < 0 && ints === 'varint')
    notes.push(`id is negative, so protobuf sign-extends it to 64 bits before encoding: 10 varint bytes for ${rec.id}. sint64 zigzag would cost 1. SQLite picks a 1-byte signed serial type instead.`);
  if (Number.isNaN(rec.score))
    notes.push('NaN: SQLite stores it as NULL, JSON writes null, only the raw IEEE-754 bytes (7F F8 00...) survive a round trip.');
  if (Object.is(rec.score, -0))
    notes.push('Negative zero: the IEEE bytes differ from +0 in exactly one bit (80 00 ...). JSON drops the sign entirely — JSON.stringify(-0) is "0". Protobuf keeps it, because the proto3 default check compares the raw 64 bits, so -0.0 is not "zero".');
  if (Object.is(rec.score, 0) || rec.name === '' || rec.flag !== 'true')
    notes.push('A proto3 scalar at its default value is not written at all, so "false", "" and 0 are indistinguishable from "absent" on the wire.');
  if (aligned && padBytes > 0)
    notes.push(`${padBytes} of ${lanes.struct.length} bytes in the native row are alignment padding — ${Math.round((padBytes / lanes.struct.length) * 100)}% of the row, paid on every row, in the cache and on the disk.`);

  return (
    <VizPanel
      title="One record, four encodings, byte for byte"
      subtitle="Edit the record and the toggles; every byte is colored by the job it does. Hover any byte to see what it encodes."
      controls={
        <>
          <Slider
            label="id (INTEGER)"
            min={0}
            max={IDS.length - 1}
            value={idIdx}
            onChange={setIdIdx}
            format={() => String(IDS[idIdx])}
          />
          <Choice label="name (TEXT)" value={name} onChange={setName} options={NAMES} />
          <Choice
            label="score (REAL)"
            value={scoreKey}
            onChange={setScoreKey}
            options={SCORES.map((s) => ({ value: s.value, label: s.value }))}
          />
          <Choice
            label="flag"
            value={flag}
            onChange={(v) => setFlag(v as 'true' | 'false' | 'null')}
            options={[
              { value: 'true', label: 'true' },
              { value: 'false', label: 'false' },
              { value: 'null', label: 'NULL' },
            ]}
          />
          <Segmented
            label="Integers"
            value={ints}
            onChange={setInts}
            options={[
              { value: 'varint', label: 'Varint', title: 'Protobuf int64 / LEB128' },
              { value: 'fixed', label: 'fixed64', title: 'Protobuf fixed64, always 8 bytes' },
            ]}
          />
          <Segmented
            label="Native byte order"
            value={endian}
            onChange={setEndian}
            options={[
              { value: 'le', label: 'Little' },
              { value: 'be', label: 'Big' },
            ]}
          />
          <Check label="Natural alignment" checked={aligned} onChange={setAligned} />
        </>
      }
      legend={
        <Legend
          items={(['header', 'tag', 'payload', 'len', 'delim', 'pad'] as Role[]).map((r) => ({
            label: ROLE_LABEL[r],
            color: ROLE_COLOR[r],
          }))}
        />
      }
      stats={
        <Stats
          items={[
            { label: 'SQLite record', value: `${lanes.sqlite.length} B`, hint: 'Varint header + narrowest-width bodies' },
            { label: 'Protobuf', value: `${lanes.proto.length} B`, hint: 'Tags + varints, defaults omitted' },
            { label: 'JSON', value: `${lanes.json.length} B`, hint: 'Field names and punctuation in every record' },
            { label: 'Native row', value: `${lanes.struct.length} B`, hint: `${padBytes} bytes of that is padding` },
            {
              label: 'JSON / Protobuf',
              value: lanes.proto.length ? `${(lanes.json.length / lanes.proto.length).toFixed(1)}x` : 'n/a',
              hint: 'How much the self-describing format costs you',
            },
          ]}
        />
      }
      note={
        <>
          <strong>{notes[0] ?? 'Every format spends its bytes differently: SQLite on a varint type header, protobuf on one tag byte per present field, JSON on field names it repeats forever, the native row on padding it can never use.'}</strong>
          {notes[1] ? <> {notes[1]}</> : null}
        </>
      }
      table={
        <table className="viz-table">
          <thead>
            <tr>
              <th>Format</th>
              <th>Total</th>
              <th>Header/tag</th>
              <th>Length</th>
              <th>Payload</th>
              <th>Delim</th>
              <th>Pad</th>
              <th>Hex</th>
            </tr>
          </thead>
          <tbody>
            {META.map((m) => (
              <tr key={m.key}>
                <td>{m.title}</td>
                <td>{lanes[m.key].length}</td>
                <td>{count(m.key, 'header') + count(m.key, 'tag')}</td>
                <td>{count(m.key, 'len')}</td>
                <td>{count(m.key, 'payload')}</td>
                <td>{count(m.key, 'delim')}</td>
                <td>{count(m.key, 'pad')}</td>
                <td style={{ wordBreak: 'break-word' }}>{lanes[m.key].map((b) => hex(b.v)).join(' ') || '(empty)'}</td>
              </tr>
            ))}
          </tbody>
        </table>
      }
    >
      <div ref={ref}>
        <TooltipHost>
          <div
            style={{
              display: 'grid',
              gridTemplateColumns: cols === 2 ? '1fr 1fr' : '1fr',
              gap: '1rem 1.25rem',
              minWidth: '280px',
            }}
          >
            {META.map((m) => (
              <Lane key={m.key} title={m.title} sub={m.sub} bytes={lanes[m.key]} empty={m.empty} />
            ))}
          </div>
        </TooltipHost>
      </div>
    </VizPanel>
  );
}
