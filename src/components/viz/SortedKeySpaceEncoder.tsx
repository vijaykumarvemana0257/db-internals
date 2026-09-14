import { useMemo, useState, type ReactNode } from 'react';
import {
  VizPanel,
  Segmented,
  Choice,
  Slider,
  Legend,
  Stats,
  TooltipHost,
  useTip,
  makeRng,
} from './Viz';

/**
 * One SQL row, four key-value physical models.
 *
 * The shared trick: turn the row's key columns into a byte string whose memcmp
 * order IS the logical order, prefix it with something that names the table and
 * the index, and put everything else in an opaque value. Range scans then become
 * "iterate from this prefix", and a secondary index is just another prefix in the
 * same sorted map.
 *
 * Byte-level detail:
 *  - CockroachDB: util/encoding. Uvarints <= 109 are a single tag byte
 *    (intZero 0x88 + v); larger ones are tagged with their byte length
 *    (0x88 + 109 + len) followed by big-endian magnitude. Strings carry the
 *    bytes marker 0x12, escape an embedded 0x00 as 0x00 0xff, and terminate
 *    with 0x00 0x01. Key = /tenant/table/index/key-cols/family.
 *  - TiDB on TiKV: 't' + EncodeInt(tableID) + '_r' + handle for row data,
 *    '_i' + EncodeInt(indexID) + index values for index data. EncodeInt flips
 *    the sign bit and writes 8 big-endian bytes; EncodeBytes writes 8-byte
 *    groups, zero-padded, each followed by a marker byte 0xFF - padCount.
 *  - MongoDB on WiredTiger: the collection is a record store keyed by RecordId
 *    (int64) holding BSON; every index is a separate WiredTiger table keyed by
 *    KeyString — an order-preserving encoding of the indexed values — with the
 *    RecordId appended.
 *  - Cassandra: partition key -> murmur3 token (global order), clustering
 *    columns sort rows inside the partition.
 */

/* ------------------------------------------------------------------ roles */

type Role = 'space' | 'index' | 'k1' | 'k2' | 'suffix' | 'value';

const ROLE_COLOR: Record<Role, string> = {
  space: 'var(--viz-1)',
  index: 'var(--viz-2)',
  k1: 'var(--viz-3)',
  k2: 'var(--viz-4)',
  suffix: 'var(--viz-5)',
  value: 'var(--viz-6)',
};

const ROLE_LABEL: Record<Role, string> = {
  space: 'table / key-space prefix',
  index: 'index or record-type prefix',
  k1: 'first key column',
  k2: 'second key column',
  suffix: 'key suffix (family, handle, RecordId)',
  value: 'value bytes',
};

/* --------------------------------------------------------------- encoders */

const TE = new TextEncoder();
const hex = (v: number) => v.toString(16).toUpperCase().padStart(2, '0');

/** CockroachDB EncodeUvarintAscending. */
function crdbUvarint(v: number): number[] {
  const intZero = 0x88;
  const intSmall = 109;
  if (v <= intSmall) return [intZero + v];
  const b: number[] = [];
  let x = v;
  while (x > 0) {
    b.unshift(x % 256);
    x = Math.floor(x / 256);
  }
  return [intZero + intSmall + b.length, ...b];
}

/** CockroachDB EncodeStringAscending: 0x12 marker, escaped 0x00, 0x00 0x01 terminator. */
function crdbString(s: string): number[] {
  const out: number[] = [0x12];
  for (const b of TE.encode(s)) {
    if (b === 0x00) out.push(0x00, 0xff);
    else out.push(b);
  }
  out.push(0x00, 0x01);
  return out;
}

function be64(v: number): number[] {
  const out: number[] = [];
  let x = Math.abs(Math.trunc(v));
  for (let i = 0; i < 8; i++) {
    out.unshift(x % 256);
    x = Math.floor(x / 256);
  }
  if (v < 0) {
    // two's complement
    let carry = 1;
    for (let i = 7; i >= 0; i--) {
      const n = (~out[i] & 0xff) + carry;
      out[i] = n & 0xff;
      carry = n > 0xff ? 1 : 0;
    }
  }
  return out;
}

/** TiDB codec.EncodeInt: flip the sign bit so memcmp order == numeric order. */
function tidbInt(v: number): number[] {
  const b = be64(v);
  b[0] ^= 0x80;
  return b;
}

/** TiDB codec.EncodeBytes: 8-byte groups, zero padded, marker 0xFF - padCount. */
function tidbBytes(s: string): number[] {
  const src = Array.from(TE.encode(s));
  const out: number[] = [];
  let i = 0;
  for (;;) {
    const g = src.slice(i, i + 8);
    const pad = 8 - g.length;
    out.push(...g, ...new Array(pad).fill(0), 0xff - pad);
    i += 8;
    if (pad > 0) break;
  }
  return out;
}

function cmpBytes(a: number[], b: number[]) {
  const n = Math.min(a.length, b.length);
  for (let i = 0; i < n; i++) if (a[i] !== b[i]) return a[i] - b[i];
  return a.length - b.length;
}

/**
 * Deterministic stand-in for Murmur3Partitioner's token. Cassandra hashes the
 * partition key with MurmurHash3 x64_128 and keeps the first 64 bits; reproducing
 * that bit-for-bit is not the point here, and the panel says so — what matters is
 * that a hash scatters partitions into an order unrelated to the key's value.
 */
function tokenOf(s: string) {
  let h = 2166136261 >>> 0;
  for (let i = 0; i < s.length; i++) {
    h ^= s.charCodeAt(i);
    h = Math.imul(h, 16777619) >>> 0;
  }
  const rng = makeRng(h || 1);
  return Math.round((rng() - 0.5) * 2 * 9.2e18);
}

/** BSON size of one orders document, computed the way BSON actually counts. */
function bsonSize(o: Order) {
  const field = (name: string, payload: number) => 1 + name.length + 1 + payload;
  return (
    4 + // int32 document length
    field('_id', 12) + // ObjectId
    field('region', 4 + o.region.length + 1) + // string: int32 len + bytes + NUL
    field('order_id', 8) + // int64
    field('status', 4 + o.status.length + 1) +
    field('total_cents', 8) +
    field('customer_id', 8) +
    1 // terminating 0x00
  );
}

/* ------------------------------------------------------------------- data */

type Order = {
  orderId: number;
  region: string;
  status: string;
  totalCents: number;
  customerId: number;
};

const BASE: Order[] = [
  { orderId: 7, region: 'us-east', status: 'paid', totalCents: 4295, customerId: 3311 },
  { orderId: 512, region: 'us-west', status: 'shipped', totalCents: 12900, customerId: 981 },
  { orderId: 0, region: '', status: 'pending', totalCents: 8600, customerId: 4102 }, // the edited row
  { orderId: 90, region: 'eu-west', status: 'refunded', totalCents: 2450, customerId: 77 },
  { orderId: 4096, region: 'us-east', status: 'shipped', totalCents: 31000, customerId: 1204 },
  { orderId: 1200, region: 'eu-west', status: 'paid', totalCents: 990, customerId: 550 },
];
const EDITED = 2;

const REGIONS = ['us-east', 'us-west', 'eu-west', 'ap-south'] as const;

type EngineId = 'crdb' | 'tikv' | 'mongo' | 'cassandra';

const ENGINES: { value: EngineId; label: string; title: string }[] = [
  { value: 'crdb', label: 'CockroachDB', title: 'SQL rows in one sorted KV map on Pebble' },
  { value: 'tikv', label: 'TiDB / TiKV', title: 'SQL rows as t{table}_r{handle} keys on RocksDB' },
  { value: 'mongo', label: 'MongoDB', title: 'RecordId -> BSON on WiredTiger, KeyString indexes' },
  { value: 'cassandra', label: 'Cassandra', title: 'partition key token + clustering columns' },
];

/* ------------------------------------------------------------------ build */

type Field = { label: string; role: Role; bytes: number[] | null; size: number; why: string };
type Entry = { pretty: string; sort: number[]; lane: number; rowIdx: number | null; size: number };
type Lane = { title: string; sub: string };

type Built = {
  keyFields: Field[];
  valueFields: Field[];
  pretty: string;
  lanes: Lane[];
  entries: Entry[];
  ddl: string;
  note: ReactNode;
};

function build(engine: EngineId, rows: Order[], idFirst: boolean): Built {
  const e = rows[EDITED];

  if (engine === 'crdb') {
    const kcol1: Field = idFirst
      ? {
          label: `order_id ${e.orderId}`,
          role: 'k1',
          bytes: crdbUvarint(e.orderId),
          size: crdbUvarint(e.orderId).length,
          why:
            e.orderId <= 109
              ? `uvarint: ${e.orderId} <= 109, so it fits in the single tag byte 0x88 + ${e.orderId}`
              : `uvarint: too big for the tag byte, so the tag says "big-endian, ${crdbUvarint(e.orderId).length - 1} bytes follow"`,
        }
      : {
          label: `region "${e.region}"`,
          role: 'k1',
          bytes: crdbString(e.region),
          size: crdbString(e.region).length,
          why: '0x12 marks a byte string; 0x00 0x01 terminates it so a shorter string sorts before a longer one with the same prefix',
        };
    const kcol2: Field = idFirst
      ? {
          label: `region "${e.region}"`,
          role: 'k2',
          bytes: crdbString(e.region),
          size: crdbString(e.region).length,
          why: 'terminated byte string: embedded 0x00 is escaped as 0x00 0xff so it can never be confused with the terminator',
        }
      : {
          label: `order_id ${e.orderId}`,
          role: 'k2',
          bytes: crdbUvarint(e.orderId),
          size: crdbUvarint(e.orderId).length,
          why:
            e.orderId <= 109
              ? `uvarint: ${e.orderId} <= 109, so it fits in the single tag byte 0x88 + ${e.orderId}`
              : `uvarint: the tag byte encodes the length, so 1-byte values sort before 2-byte values and the order stays numeric`,
        };

    const keyFields: Field[] = [
      {
        label: 'table 106',
        role: 'space',
        bytes: crdbUvarint(106),
        size: 1,
        why: 'every key in the cluster starts with the table (and, in a multi-tenant cluster, the tenant) — that prefix is the only thing separating this table from every other one',
      },
      {
        label: 'index 1 (primary)',
        role: 'index',
        bytes: crdbUvarint(1),
        size: 1,
        why: 'index 1 is the primary index: its rows ARE the table',
      },
      kcol1,
      kcol2,
      {
        label: 'column family 0',
        role: 'suffix',
        bytes: crdbUvarint(0),
        size: 1,
        why: 'columns are grouped into families; each family is a separate KV pair under the same row prefix',
      },
    ];

    const valueFields: Field[] = [
      { label: 'CRC32 checksum', role: 'value', bytes: null, size: 4, why: 'covers key + value, checked on read' },
      { label: 'value type tag', role: 'value', bytes: null, size: 1, why: 'TUPLE, for a multi-column family' },
      { label: 'status', role: 'value', bytes: null, size: 2 + e.status.length, why: 'column-id delta + type, then the datum' },
      { label: 'total_cents', role: 'value', bytes: null, size: 3, why: 'column-id delta + type, then a varint' },
      { label: 'customer_id', role: 'value', bytes: null, size: 3, why: 'column-id delta + type, then a varint' },
    ];

    const rowKey = (o: Order) => [
      ...crdbUvarint(106),
      ...crdbUvarint(1),
      ...(idFirst
        ? [...crdbUvarint(o.orderId), ...crdbString(o.region)]
        : [...crdbString(o.region), ...crdbUvarint(o.orderId)]),
      ...crdbUvarint(0),
    ];
    const idxKey = (o: Order) => [
      ...crdbUvarint(106),
      ...crdbUvarint(2),
      ...crdbString(o.status),
      ...(idFirst
        ? [...crdbUvarint(o.orderId), ...crdbString(o.region)]
        : [...crdbString(o.region), ...crdbUvarint(o.orderId)]),
    ];

    const entries: Entry[] = [
      ...rows.map((o, i) => ({
        pretty: idFirst
          ? `/Table/106/1/${o.orderId}/"${o.region}"/0`
          : `/Table/106/1/"${o.region}"/${o.orderId}/0`,
        sort: rowKey(o),
        lane: 0,
        rowIdx: i,
        size: rowKey(o).length,
      })),
      ...rows.map((o, i) => ({
        pretty: idFirst
          ? `/Table/106/2/"${o.status}"/${o.orderId}/"${o.region}"`
          : `/Table/106/2/"${o.status}"/"${o.region}"/${o.orderId}`,
        sort: idxKey(o),
        lane: 1,
        rowIdx: i,
        size: idxKey(o).length,
      })),
    ];

    return {
      keyFields,
      valueFields,
      pretty: idFirst
        ? `/Table/106/1/${e.orderId}/"${e.region}"/0`
        : `/Table/106/1/"${e.region}"/${e.orderId}/0`,
      lanes: [
        { title: 'index 1 — the table itself', sub: 'primary index: key columns in the key, everything else in the value' },
        { title: 'index 2 — CREATE INDEX ON orders (status)', sub: 'same map, next prefix; the value is empty because the key already carries the PK' },
      ],
      entries,
      ddl: idFirst
        ? 'CREATE TABLE orders (order_id INT, region STRING, …, PRIMARY KEY (order_id, region))'
        : 'CREATE TABLE orders (region STRING, order_id INT, …, PRIMARY KEY (region, order_id))',
      note: (
        <>
          Both the table and its secondary index live in <strong>one sorted map</strong>, separated only by
          the index byte. Everything for table 106 is a contiguous range, and so is everything for one
          region — which is exactly what a range scan, a range split and a rebalance operate on.
        </>
      ),
    };
  }

  if (engine === 'tikv') {
    const keyFields: Field[] = [
      { label: "'t'", role: 'space', bytes: [0x74], size: 1, why: 'the table-data prefix; TiDB also uses m… for metadata' },
      {
        label: 'table 106',
        role: 'space',
        bytes: tidbInt(106),
        size: 8,
        why: 'EncodeInt: sign bit flipped, 8 big-endian bytes — fixed width, so the prefix length is always the same',
      },
      { label: "'_r' row data", role: 'index', bytes: [0x5f, 0x72], size: 2, why: "'_r' is row data, '_i' is index data — the two never interleave" },
      ...(idFirst
        ? [
            { label: `order_id ${e.orderId}`, role: 'k1' as Role, bytes: tidbInt(e.orderId), size: 8, why: 'memcomparable int: flip the sign bit and the byte order matches the numeric order, negatives included' },
            { label: `region "${e.region}"`, role: 'k2' as Role, bytes: tidbBytes(e.region), size: tidbBytes(e.region).length, why: 'memcomparable bytes: 8-byte groups, zero padded, each closed by a marker of 0xFF minus the pad count' },
          ]
        : [
            { label: `region "${e.region}"`, role: 'k1' as Role, bytes: tidbBytes(e.region), size: tidbBytes(e.region).length, why: 'memcomparable bytes: 8-byte groups, zero padded, each closed by a marker of 0xFF minus the pad count' },
            { label: `order_id ${e.orderId}`, role: 'k2' as Role, bytes: tidbInt(e.orderId), size: 8, why: 'memcomparable int: flip the sign bit and the byte order matches the numeric order, negatives included' },
          ]),
    ];

    const valueFields: Field[] = [
      { label: 'row format v2 marker', role: 'value', bytes: [0x80], size: 1, why: 'tidb_row_format_version = 2' },
      { label: 'flag + column counts', role: 'value', bytes: null, size: 5, why: 'large-row flag, count of non-null and null columns' },
      { label: 'column ids', role: 'value', bytes: null, size: 3, why: 'sorted, so a projection can binary-search the id it wants' },
      { label: 'value offsets', role: 'value', bytes: null, size: 6, why: 'offsets let the decoder jump straight to one column without walking the row' },
      { label: 'column data', role: 'value', bytes: null, size: e.status.length + 8 + 4, why: 'the non-key columns, back to back' },
    ];

    const rowKey = (o: Order) => [
      0x74,
      ...tidbInt(106),
      0x5f,
      0x72,
      ...(idFirst ? [...tidbInt(o.orderId), ...tidbBytes(o.region)] : [...tidbBytes(o.region), ...tidbInt(o.orderId)]),
    ];
    const idxKey = (o: Order) => [
      0x74,
      ...tidbInt(106),
      0x5f,
      0x69,
      ...tidbInt(1),
      ...tidbBytes(o.status),
      ...(idFirst ? [...tidbInt(o.orderId), ...tidbBytes(o.region)] : [...tidbBytes(o.region), ...tidbInt(o.orderId)]),
    ];

    const pretty = (o: Order) =>
      idFirst ? `t{106}_r{${o.orderId}, "${o.region}"}` : `t{106}_r{"${o.region}", ${o.orderId}}`;

    return {
      keyFields,
      valueFields,
      pretty: pretty(e),
      lanes: [
        { title: "t{106}_r… — row data", sub: 'clustered primary key: the handle IS the encoded PK columns' },
        { title: "t{106}_i{1}… — index data", sub: 'index on (status); the handle is appended so the index entry can find its row' },
      ],
      entries: [
        ...rows.map((o, i) => ({ pretty: pretty(o), sort: rowKey(o), lane: 0, rowIdx: i, size: rowKey(o).length })),
        ...rows.map((o, i) => ({
          pretty: idFirst
            ? `t{106}_i{1}{"${o.status}", ${o.orderId}, "${o.region}"}`
            : `t{106}_i{1}{"${o.status}", "${o.region}", ${o.orderId}}`,
          sort: idxKey(o),
          lane: 1,
          rowIdx: i,
          size: idxKey(o).length,
        })),
      ],
      ddl: idFirst
        ? 'CREATE TABLE orders (order_id BIGINT, region VARCHAR(32), …, PRIMARY KEY (order_id, region) CLUSTERED)'
        : 'CREATE TABLE orders (region VARCHAR(32), order_id BIGINT, …, PRIMARY KEY (region, order_id) CLUSTERED)',
      note: (
        <>
          Fixed-width prefixes mean TiKV can split a range anywhere without parsing SQL: every key for
          table 106 shares the same 11-byte head, and the region encoding is padded to 8-byte groups so
          that <em>byte</em> comparison and <em>string</em> comparison can never disagree.
        </>
      ),
    };
  }

  if (engine === 'mongo') {
    const recordIds = rows.map((_, i) => (i + 1) * 4 + 1); // WiredTiger record numbers, insertion order
    const keyFields: Field[] = [
      {
        label: `RecordId ${recordIds[EDITED]}`,
        role: 'suffix',
        bytes: null,
        size: 8,
        why: 'the collection is a WiredTiger record store keyed by an int64 RecordId handed out at insert time — nothing in your document decides where the document lands',
      },
    ];
    const valueFields: Field[] = [
      { label: 'BSON: length', role: 'value', bytes: null, size: 4, why: 'int32 total size, so the reader can skip the document' },
      { label: '_id (ObjectId)', role: 'value', bytes: null, size: 17, why: 'type byte + "_id\\0" + 12 bytes: 4-byte timestamp, 5-byte random, 3-byte counter' },
      { label: `region "${e.region}"`, role: 'value', bytes: null, size: 1 + 7 + 4 + e.region.length + 1, why: 'BSON stores the field NAME in every document — that is the cost of a schemaless value' },
      { label: 'order_id', role: 'value', bytes: null, size: 1 + 9 + 8, why: 'type byte + name + int64' },
      { label: 'other fields + terminator', role: 'value', bytes: null, size: bsonSize(e) - (4 + 17 + (1 + 7 + 4 + e.region.length + 1) + 18), why: 'status, total_cents, customer_id, then the trailing 0x00' },
    ];

    const idxSort = (o: Order) =>
      idFirst ? [...crdbUvarint(o.orderId), ...crdbString(o.region)] : [...crdbString(o.region), ...crdbUvarint(o.orderId)];

    return {
      keyFields,
      valueFields,
      pretty: `RecordId(${recordIds[EDITED]}) → BSON(${bsonSize(e)} bytes)`,
      lanes: [
        { title: 'collection orders — record store', sub: 'ordered by RecordId, i.e. roughly by insertion; a query on order_id has nothing to seek to' },
        {
          title: idFirst
            ? 'index {order_id: 1, region: 1} — its own WiredTiger table'
            : 'index {region: 1, order_id: 1} — its own WiredTiger table',
          sub: 'KeyString: an order-preserving encoding of the indexed values, with the RecordId appended',
        },
      ],
      entries: [
        ...rows.map((o, i) => ({
          pretty: `RecordId(${recordIds[i]}) → {_id:…, region:"${o.region}", order_id:${o.orderId}, …}`,
          sort: be64(recordIds[i]),
          lane: 0,
          rowIdx: i,
          size: 8 + bsonSize(o),
        })),
        ...rows.map((o, i) => ({
          pretty: idFirst
            ? `KeyString(${o.orderId}, "${o.region}") + RecordId(${recordIds[i]})`
            : `KeyString("${o.region}", ${o.orderId}) + RecordId(${recordIds[i]})`,
          sort: idxSort(o),
          lane: 1,
          rowIdx: i,
          size: idxSort(o).length + 2,
        })),
      ],
      ddl: idFirst
        ? 'db.orders.createIndex({ order_id: 1, region: 1 })'
        : 'db.orders.createIndex({ region: 1, order_id: 1 })',
      note: (
        <>
          The document store splits the two jobs: the collection is keyed by an opaque RecordId, so
          <em> nothing about your data</em> decides its position, and every ordering you want has to be
          bought as a separate index table whose KeyString does the same order-preserving trick as the SQL
          engines above.
        </>
      ),
    };
  }

  // Cassandra
  const partKey = (o: Order) => (idFirst ? String(o.orderId) : o.region);
  const clustering = (o: Order) => (idFirst ? o.region : String(o.orderId));
  const keyFields: Field[] = [
    {
      label: `token("${partKey(e)}")`,
      role: 'space',
      bytes: null,
      size: 8,
      why: 'Murmur3Partitioner hashes the partition key to a signed 64-bit token: it picks the replica set AND fixes the global order, which is why you cannot range-scan the partition key. The token values shown here come from a stand-in hash — only the ordering they produce is the point',
    },
    { label: `partition key "${partKey(e)}"`, role: 'k1', bytes: null, size: partKey(e).length, why: 'stored with the partition; everything under it is contiguous in every SSTable' },
    {
      label: `clustering ${idFirst ? `"${clustering(e)}"` : clustering(e)}`,
      role: 'k2',
      bytes: null,
      size: idFirst ? clustering(e).length : 8,
      why: 'clustering columns sort the rows inside the partition — this is the only ordering Cassandra gives you for free',
    },
  ];
  const valueFields: Field[] = [
    { label: 'row liveness timestamp', role: 'value', bytes: null, size: 8, why: 'microsecond write time; last-write-wins resolution happens per cell' },
    { label: 'status cell', role: 'value', bytes: null, size: 1 + e.status.length, why: 'the column is identified by position in the SSTable serialization header, not by name' },
    { label: 'total_cents cell', role: 'value', bytes: null, size: 9, why: 'a cell may carry its own timestamp and TTL when it differs from the row' },
    { label: 'customer_id cell', role: 'value', bytes: null, size: 9, why: 'unset columns are simply absent — a sparse row costs nothing' },
  ];

  const sortKey = (o: Order) => [
    ...tidbInt(Math.trunc(tokenOf(partKey(o)) / 1e9)),
    ...(idFirst ? crdbString(o.region) : tidbInt(o.orderId)),
  ];

  return {
    keyFields,
    valueFields,
    pretty: `token("${partKey(e)}") / "${partKey(e)}" / ${clustering(e)}`,
    lanes: [
      {
        title: 'orders — one table, partitions in token order',
        sub: 'rows sorted by clustering column inside each partition; token values are illustrative — only their order is meaningful',
      },
    ],
    entries: rows.map((o, i) => ({
      pretty: `token ${(tokenOf(partKey(o)) / 1e18).toFixed(2)}e18 · partition "${partKey(o)}" · row ${clustering(o)}`,
      sort: sortKey(o),
      lane: 0,
      rowIdx: i,
      size: 8 + partKey(o).length + clustering(o).length,
    })),
    ddl: idFirst
      ? 'CREATE TABLE orders (…, PRIMARY KEY ((order_id), region))'
      : 'CREATE TABLE orders (…, PRIMARY KEY ((region), order_id))',
    note: (
      <>
        The token is the whole story: partitions land in hash order, so{' '}
        <em>{idFirst ? 'every order is its own partition' : 'all of one region is one partition'}</em> and
        the only cheap range query is over the clustering column inside it. Change the partition key and
        you have not tuned a query — you have changed which queries exist.
      </>
    ),
  };
}

/* ---------------------------------------------------------------- display */

function FieldRun({ f, first }: { f: Field; first: boolean }) {
  const tip = useTip();
  return (
    <div style={{ maxWidth: '100%' }}>
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
        {f.label}
      </div>
      <div style={{ display: 'flex', flexWrap: 'wrap', gap: '2px' }}>
        {f.bytes ? (
          f.bytes.map((b, i) => (
            <span
              key={i}
              {...tip(
                <>
                  <strong>0x{hex(b)}</strong> · {f.label}
                  <br />
                  {ROLE_LABEL[f.role]}
                  <br />
                  <span style={{ color: 'var(--viz-ink-2)' }}>{f.why}</span>
                </>,
              )}
              tabIndex={first && i === 0 ? 0 : -1}
              style={{
                display: 'inline-block',
                minWidth: '1.9rem',
                textAlign: 'center',
                padding: '.12rem .15rem',
                fontSize: '.6875rem',
                lineHeight: 1.4,
                color: 'var(--viz-ink)',
                fontVariantNumeric: 'tabular-nums',
                background: `color-mix(in srgb, ${ROLE_COLOR[f.role]} 16%, var(--viz-plane))`,
                border: '1px solid var(--viz-border)',
                borderBottom: `3px solid ${ROLE_COLOR[f.role]}`,
                borderRadius: '4px',
                cursor: 'help',
              }}
            >
              {hex(b)}
            </span>
          ))
        ) : (
          <span
            {...tip(
              <>
                <strong>{f.label}</strong> · {f.size} bytes
                <br />
                {ROLE_LABEL[f.role]}
                <br />
                <span style={{ color: 'var(--viz-ink-2)' }}>{f.why}</span>
              </>,
            )}
            tabIndex={first ? 0 : -1}
            style={{
              display: 'inline-block',
              minWidth: `${Math.min(9, 1.6 + f.size * 0.22)}rem`,
              textAlign: 'center',
              padding: '.12rem .35rem',
              fontSize: '.6875rem',
              lineHeight: 1.4,
              color: 'var(--viz-ink-2)',
              background: `color-mix(in srgb, ${ROLE_COLOR[f.role]} 12%, var(--viz-plane))`,
              border: '1px dashed var(--viz-border)',
              borderBottom: `3px solid ${ROLE_COLOR[f.role]}`,
              borderRadius: '4px',
              cursor: 'help',
            }}
          >
            {f.size} B
          </span>
        )}
      </div>
    </div>
  );
}

/* ------------------------------------------------------------------ panel */

export default function SortedKeySpaceEncoder() {
  const [engine, setEngine] = useState<EngineId>('crdb');
  const [region, setRegion] = useState<string>('us-east');
  const [orderId, setOrderId] = useState(90);
  const [idFirst, setIdFirst] = useState(false);

  const rows = useMemo(() => {
    const r = BASE.map((o) => ({ ...o }));
    r[EDITED] = { ...r[EDITED], region, orderId };
    return r;
  }, [region, orderId]);

  const b = useMemo(() => build(engine, rows, idFirst), [engine, rows, idFirst]);

  const sorted = useMemo(() => {
    const byLane = b.lanes.map((_, li) =>
      b.entries.filter((x) => x.lane === li).sort((p, q) => cmpBytes(p.sort, q.sort)),
    );
    return byLane;
  }, [b]);

  const keyBytes = b.keyFields.reduce((a, f) => a + (f.bytes ? f.bytes.length : f.size), 0);
  const valueBytes = b.valueFields.reduce((a, f) => a + (f.bytes ? f.bytes.length : f.size), 0);
  const rank = sorted[0].findIndex((x) => x.rowIdx === EDITED) + 1;

  return (
    <VizPanel
      title="One row, four key spaces"
      subtitle="Edit row 3 of the orders table and watch its key bytes rebuild, then watch where those bytes land in the sorted map."
      controls={
        <>
          <Segmented label="Engine" value={engine} onChange={setEngine} options={ENGINES} />
          <Choice
            label="region"
            value={region}
            onChange={setRegion}
            options={REGIONS.map((r) => ({ value: r as string, label: r }))}
          />
          <Slider label="order_id" min={1} max={4096} value={orderId} onChange={setOrderId} />
          <Segmented
            label="Key column order"
            value={idFirst ? 'id' : 'region'}
            onChange={(v) => setIdFirst(v === 'id')}
            options={[
              { value: 'region', label: '(region, order_id)' },
              { value: 'id', label: '(order_id, region)' },
            ]}
          />
        </>
      }
      legend={
        <Legend
          items={[
            { label: 'table / key-space prefix', color: ROLE_COLOR.space },
            { label: 'index prefix', color: ROLE_COLOR.index },
            { label: 'first key column', color: ROLE_COLOR.k1 },
            { label: 'second key column', color: ROLE_COLOR.k2 },
            { label: 'key suffix', color: ROLE_COLOR.suffix },
            { label: 'value', color: ROLE_COLOR.value },
          ]}
        />
      }
      stats={
        <Stats
          items={[
            { label: 'Key bytes', value: keyBytes },
            { label: 'Value bytes', value: valueBytes },
            {
              label: 'Row 3 sorts at',
              value: `#${rank} of ${sorted[0].length}`,
              hint: 'position of the edited row inside the first lane, by raw byte comparison',
            },
            { label: 'KV pairs for 6 rows', value: b.entries.length },
          ]}
        />
      }
      note={b.note}
      table={
        <table className="viz-table">
          <thead>
            <tr>
              <th>Lane</th>
              <th>#</th>
              <th>Key</th>
              <th>Bytes</th>
              <th>Row</th>
            </tr>
          </thead>
          <tbody>
            {sorted.flatMap((lane, li) =>
              lane.map((x, i) => (
                <tr key={`${li}-${i}`}>
                  <td>{b.lanes[li].title}</td>
                  <td>{i + 1}</td>
                  <td style={{ fontVariantNumeric: 'tabular-nums' }}>{x.pretty}</td>
                  <td>{x.size}</td>
                  <td>{x.rowIdx === EDITED ? 'row 3 (edited)' : `row ${(x.rowIdx ?? 0) + 1}`}</td>
                </tr>
              )),
            )}
          </tbody>
        </table>
      }
    >
      <TooltipHost>
        <div style={{ minWidth: 320 }}>
          <div style={{ fontSize: '.6875rem', color: 'var(--viz-ink-2)', marginBottom: '.45rem' }}>{b.ddl}</div>

          <div style={{ fontSize: '.75rem', fontWeight: 600, color: 'var(--viz-ink)', marginBottom: '.3rem' }}>
            Key
          </div>
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: '.4rem .55rem' }}>
            {b.keyFields.map((f, i) => (
              <FieldRun key={`${f.label}-${i}`} f={f} first={i === 0} />
            ))}
          </div>
          <div
            style={{
              marginTop: '.4rem',
              fontSize: '.75rem',
              color: 'var(--viz-ink)',
              fontVariantNumeric: 'tabular-nums',
            }}
          >
            {b.pretty}
          </div>

          <div
            style={{ fontSize: '.75rem', fontWeight: 600, color: 'var(--viz-ink)', margin: '.75rem 0 .3rem' }}
          >
            Value
          </div>
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: '.4rem .55rem' }}>
            {b.valueFields.map((f, i) => (
              <FieldRun key={`${f.label}-${i}`} f={f} first={i === 0} />
            ))}
          </div>

          <div
            style={{ fontSize: '.75rem', fontWeight: 600, color: 'var(--viz-ink)', margin: '.9rem 0 .3rem' }}
          >
            The sorted key space
          </div>
          {sorted.map((lane, li) => (
            <div key={li} style={{ marginBottom: '.6rem' }}>
              <div style={{ fontSize: '.6875rem', color: 'var(--viz-ink)' }}>{b.lanes[li].title}</div>
              <div style={{ fontSize: '.625rem', color: 'var(--viz-ink-2)', marginBottom: '.2rem' }}>
                {b.lanes[li].sub}
              </div>
              {lane.map((x, i) => {
                const isEdited = x.rowIdx === EDITED;
                return (
                  <div
                    key={i}
                    style={{
                      display: 'flex',
                      gap: '.5rem',
                      alignItems: 'baseline',
                      padding: '.12rem .35rem',
                      fontSize: '.6875rem',
                      lineHeight: 1.55,
                      color: isEdited ? 'var(--viz-ink)' : 'var(--viz-ink-2)',
                      fontVariantNumeric: 'tabular-nums',
                      whiteSpace: 'nowrap',
                      background: isEdited
                        ? `color-mix(in srgb, ${ROLE_COLOR.space} 18%, var(--viz-plane))`
                        : 'transparent',
                      borderLeft: `3px solid ${isEdited ? ROLE_COLOR.space : 'var(--viz-border)'}`,
                      borderRadius: '0 3px 3px 0',
                      marginBottom: '1px',
                    }}
                  >
                    <span style={{ color: 'var(--viz-ink-muted)', minWidth: '1.2rem' }}>{i + 1}</span>
                    <span style={{ flex: 1 }}>{x.pretty}</span>
                    <span style={{ color: 'var(--viz-ink-muted)' }}>{x.size} B</span>
                  </div>
                );
              })}
            </div>
          ))}
        </div>
      </TooltipHost>
    </VizPanel>
  );
}
