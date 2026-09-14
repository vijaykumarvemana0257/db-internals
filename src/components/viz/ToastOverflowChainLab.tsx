import { useState } from 'react';
import {
  VizPanel,
  Slider,
  Segmented,
  Choice,
  Button,
  Legend,
  Stats,
  Note,
  TooltipHost,
  useTip,
  fmtBytes,
  fmtNum,
  makeRng,
  useSize,
} from './Viz';

/**
 * Where a wide value actually lives.
 *
 * A tuple cannot span pages, so every engine needs an escape hatch. PostgreSQL's is TOAST:
 * once the whole tuple crosses TOAST_TUPLE_TARGET (2032 B by default) the largest varlena
 * attributes are compressed and then sliced into 1996-byte chunks in a pg_toast relation,
 * leaving an 18-byte varatt_external pointer in the row. InnoDB and SQLite solve the same
 * problem with a linked chain of overflow pages instead of an indexed chunk table — which
 * is why a mid-value substr() is one index probe in Postgres and a chain walk everywhere else.
 *
 * Everything below is derived from the real constants, not fitted to a picture.
 */

/* ------------------------------------------------------------------ constants */

const PG_PAGE = 8192; // BLCKSZ
const PG_TARGET = 2032; // TOAST_TUPLE_THRESHOLD / default TOAST_TUPLE_TARGET
const PG_CHUNK = 1996; // TOAST_MAX_CHUNK_SIZE
const PG_PTR = 18; // sizeof(varatt_external) + the 2-byte varlena-1B-E header
const PG_HDR = 24; // HeapTupleHeader (23) MAXALIGNed, no null bitmap
const PG_MAX_TUPLE = 8160; // what "row is too big: size %d, maximum size 8160" refers to
const PG_IDX_FANOUT = 400; // (chunk_id, chunk_seq) index tuples per 8 KB leaf, ~20 B each

const INNO_PAGE = 16384; // innodb_page_size
const INNO_MAX_ROW = 8126; // "Row size too large (> 8126)"
const INNO_PTR = 20; // BTR_EXTERN_FIELD_REF_SIZE
const INNO_PREFIX = 768; // DICT_MAX_FIXED_COL_LEN, the COMPACT/REDUNDANT local prefix
const INNO_MIN_EXT = 40; // 2 * BTR_EXTERN_FIELD_REF_SIZE: shorter fields never go off-page
const INNO_BLOB = 16330; // 16384 - 38 FIL header - 8 BTR_BLOB_HDR - 8 FIL trailer
const INNO_HDR = 18; // record header + DB_TRX_ID(6) + DB_ROLL_PTR(7)

const SQLITE_PAGE = 4096; // PRAGMA page_size default since 3.12
const SQLITE_MAXLOCAL = 4061; // U - 35
const SQLITE_MINLOCAL = 489; // ((U-12)*32/255) - 23
const SQLITE_OVF = 4092; // U - 4, the rest of an overflow page is the next-page number
const SQLITE_HDR = 8; // record header + rowid varints, approximated

/** Deterministic "this text does not compress uniformly" grain. */
const GRAIN = (() => {
  const rng = makeRng(19960424);
  return Array.from({ length: 32 }, () => 0.88 + rng() * 0.24);
})();
const grain = (n: number) => GRAIN[Math.abs(Math.round(Math.log2(Math.max(2, n))) * 7) % GRAIN.length];

/* ---------------------------------------------------------------------- model */

type EngineId = 'pg' | 'dynamic' | 'compact' | 'sqlite';
type Strategy = 'extended' | 'external' | 'main' | 'plain';
type Algo = 'pglz' | 'lz4' | 'none';

const ENGINES: { value: EngineId; label: string }[] = [
  { value: 'pg', label: 'PostgreSQL 8 KB — TOAST' },
  { value: 'dynamic', label: 'InnoDB 16 KB — DYNAMIC' },
  { value: 'compact', label: 'InnoDB 16 KB — COMPACT' },
  { value: 'sqlite', label: 'SQLite 4 KB — overflow' },
];

type Layout = {
  stored: number; // value bytes actually stored, after compression
  compressed: boolean;
  compWhy: string;
  inline: number; // value bytes kept in the row (a prefix, or the whole thing)
  otherInline: number; // bytes of the *other* columns that stay in the record
  ptr: number; // pointer bytes in the row, 0 when fully inline
  off: number; // value bytes stored out of line
  row: number; // total record bytes in the main page
  units: number; // chunks (Postgres) or overflow pages (others)
  unitLabel: string;
  storePages: number; // pages of out-of-line storage
  idxFull: number; // toast-index pages touched by a full fetch
  idxDescent: number; // toast-index pages touched by one probe
  pageBytes: number;
  threshold: number;
  thresholdLabel: string;
  error: string | null;
  why: string;
};

function compress(size: number, ability: number, algo: Algo): { out: number; ok: boolean; why: string } {
  if (algo === 'none') return { out: size, ok: false, why: 'Compression off — the bytes are stored raw.' };
  if (size < 32)
    return { out: size, ok: false, why: 'Below pglz min_input_size = 32 bytes: not even attempted.' };
  const ratio = Math.min(0.95, ability * grain(size) * (algo === 'lz4' ? 0.9 : 1));
  const out = Math.max(24, Math.round(size * (1 - ratio)));
  if (algo === 'pglz') {
    if (ability < 0.12)
      return {
        out: size,
        ok: false,
        why: 'pglz found no match in the first 1024 bytes (first_success_by) and gave up — stored raw.',
      };
    if (out > size * 0.75)
      return {
        out: size,
        ok: false,
        why: 'pglz requires min_comp_rate 25%: the result was not 25% smaller, so the raw bytes are kept.',
      };
    return { out, ok: true, why: `pglz: ${fmtBytes(size)} → ${fmtBytes(out)}.` };
  }
  if (out >= size) return { out: size, ok: false, why: 'LZ4 output was no smaller; the raw bytes are kept.' };
  return { out, ok: true, why: `LZ4: ${fmtBytes(size)} → ${fmtBytes(out)}, several times faster to decode than pglz.` };
}

function layout(
  engine: EngineId,
  size: number,
  other: number,
  strat: Strategy,
  algo: Algo,
  ability: number,
): Layout {
  if (engine === 'pg') return pgLayout(size, other, strat, algo, ability);
  if (engine === 'sqlite') return sqliteLayout(size, other);
  return innoLayout(engine, size, other);
}

function pgLayout(size: number, other: number, strat: Strategy, algo: Algo, ability: number): Layout {
  const canCompress = strat === 'extended' || strat === 'main';
  const c = canCompress
    ? compress(size, ability, algo)
    : {
        out: size,
        ok: false,
        why:
          strat === 'external'
            ? 'EXTERNAL never compresses — that is the whole point of it.'
            : 'PLAIN never compresses.',
      };

  const stored = c.out;
  const inlineTuple = PG_HDR + other + 4 + stored; // 4-byte varlena header on a value this size
  const canExternalize = strat !== 'plain';
  const mustMove = inlineTuple > PG_TARGET;

  let inline = stored;
  let ptr = 0;
  let off = 0;
  let why: string;

  if (canExternalize && mustMove) {
    inline = 0;
    ptr = PG_PTR;
    off = stored;
    why =
      strat === 'main'
        ? 'MAIN is evicted last (pass 4 of the loop), but with nothing else to move it still goes out of line.'
        : strat === 'external'
          ? 'EXTERNAL: straight out of line, uncompressed, so slices can address a chunk directly.'
          : 'EXTENDED: compressed first, and still over target, so the value moved to the toast relation.';
  } else if (mustMove) {
    why = 'PLAIN forbids both compression and out-of-line storage: the value has to fit on the page as it is.';
  } else {
    why = `The whole tuple is ${fmtBytes(inlineTuple)}, under TOAST_TUPLE_TARGET — nothing is toasted at all.`;
  }

  const row = PG_HDR + other + (ptr > 0 ? ptr : 4 + inline);
  const units = off > 0 ? Math.ceil(off / PG_CHUNK) : 0;
  const leaves = units > 0 ? Math.max(1, Math.ceil(units / PG_IDX_FANOUT)) : 0;

  return {
    stored,
    compressed: c.ok,
    compWhy: c.why,
    inline,
    otherInline: other,
    ptr,
    off,
    row,
    units,
    unitLabel: 'chunk',
    storePages: Math.ceil(units / 4),
    idxFull: leaves > 1 ? leaves + 1 : leaves,
    idxDescent: leaves > 1 ? 2 : leaves,
    pageBytes: PG_PAGE,
    threshold: PG_TARGET,
    thresholdLabel: 'TOAST_TUPLE_TARGET 2032 B',
    error:
      row > PG_MAX_TUPLE
        ? `ERROR: row is too big: size ${fmtNum(row)}, maximum size ${PG_MAX_TUPLE}`
        : null,
    why,
  };
}

function innoLayout(engine: EngineId, size: number, other: number): Layout {
  const rowIfInline = INNO_HDR + other + size;
  const eligible = size > INNO_MIN_EXT;
  const goesOff = rowIfInline > INNO_MAX_ROW && eligible;

  let inline = size;
  let ptr = 0;
  let off = 0;
  let why: string;

  if (goesOff) {
    inline = engine === 'compact' ? Math.min(INNO_PREFIX, size) : 0;
    ptr = INNO_PTR;
    off = size - inline;
    why =
      engine === 'compact'
        ? 'COMPACT keeps a 768-byte prefix of every off-page column in the clustered record, plus the 20-byte pointer.'
        : 'DYNAMIC stores nothing locally: the whole column goes off-page behind a 20-byte pointer.';
  } else {
    why = rowIfInline > INNO_MAX_ROW
      ? `Fields of ${INNO_MIN_EXT} bytes or less are never stored off-page, so this row cannot be shrunk.`
      : `The clustered-index record is ${fmtBytes(rowIfInline)}, inside the ~${INNO_MAX_ROW}-byte row limit — nothing moves.`;
  }

  const row = INNO_HDR + other + (ptr > 0 ? inline + ptr : inline);
  const units = off > 0 ? Math.ceil(off / INNO_BLOB) : 0;

  return {
    stored: size,
    compressed: false,
    compWhy: 'InnoDB does not compress individual columns: that needs ROW_FORMAT=COMPRESSED or page compression.',
    inline,
    otherInline: other,
    ptr,
    off,
    row,
    units,
    unitLabel: 'BLOB page',
    storePages: units,
    idxFull: 0,
    idxDescent: 0,
    pageBytes: INNO_PAGE,
    threshold: INNO_MAX_ROW,
    thresholdLabel: 'row limit ~8126 B (half a page)',
    error: row > INNO_MAX_ROW ? `ERROR 1118 (42000): Row size too large (> ${INNO_MAX_ROW})` : null,
    why,
  };
}

function sqliteLayout(size: number, other: number): Layout {
  const P = SQLITE_HDR + other + size;
  let local = P;
  if (P > SQLITE_MAXLOCAL) {
    const K = SQLITE_MINLOCAL + ((P - SQLITE_MINLOCAL) % (SQLITE_PAGE - 4));
    local = K <= SQLITE_MAXLOCAL ? K : SQLITE_MINLOCAL;
  }
  const off = P - local;
  const otherInline = Math.min(other, Math.max(0, local - SQLITE_HDR));
  const inline = Math.max(0, local - SQLITE_HDR - otherInline);

  const units = off > 0 ? Math.ceil(off / SQLITE_OVF) : 0;
  return {
    stored: size,
    compressed: false,
    compWhy: 'SQLite stores blobs verbatim; compression is an application or extension concern.',
    inline,
    otherInline,
    ptr: off > 0 ? 4 : 0,
    off,
    row: local + (off > 0 ? 4 : 0),
    units,
    unitLabel: 'overflow page',
    storePages: units,
    idxFull: 0,
    idxDescent: 0,
    pageBytes: SQLITE_PAGE,
    threshold: SQLITE_MAXLOCAL,
    thresholdLabel: 'maxLocal = U − 35 = 4061 B',
    error: null,
    why:
      off > 0
        ? `The cell keeps ${fmtBytes(inline)} of payload plus a 4-byte page number; the rest spills into a singly-linked chain.`
        : 'The whole record fits in the leaf cell — no overflow page is allocated.',
  };
}

/* ------------------------------------------------------------------- I/O trace */

type Action = 'idle' | 'read' | 'prefix' | 'middle' | 'updOther' | 'updValue';

type Step = { what: string; pages: number; detail: string };

function trace(action: Action, engine: EngineId, L: Layout): { steps: Step[]; reads: number[]; writes: number } {
  const steps: Step[] = [];
  const reads: number[] = []; // indices of out-of-line units touched
  let writes = 0;
  const isPg = engine === 'pg';

  const all = () => {
    for (let i = 0; i < L.units; i++) reads.push(i);
  };

  switch (action) {
    case 'read': {
      steps.push({ what: 'heap / clustered page', pages: 1, detail: 'The row itself, found by the index or the scan.' });
      if (L.off === 0) break;
      if (isPg) {
        steps.push({
          what: 'toast index descent + leaf scan',
          pages: L.idxFull,
          detail: `pg_toast_<oid>_index, scan key chunk_id = va_valueid, walking ${fmtNum(L.units)} entries in chunk_seq order.`,
        });
        steps.push({
          what: 'toast chunk pages',
          pages: L.storePages,
          detail: `${fmtNum(L.units)} chunks of ${PG_CHUNK} bytes, four to an 8 KB page.`,
        });
      } else {
        steps.push({
          what: `${L.unitLabel} chain`,
          pages: L.storePages,
          detail: 'Each page carries the number of the next one; there is no index to descend.',
        });
      }
      if (L.compressed) steps.push({ what: 'decompress', pages: 0, detail: 'CPU, every single time this column is read.' });
      all();
      break;
    }
    case 'prefix': {
      steps.push({ what: 'heap / clustered page', pages: 1, detail: 'The row, and with it the pointer.' });
      if (L.off === 0) break;
      if (isPg) {
        steps.push({
          what: 'toast index probe',
          pages: L.idxDescent,
          detail: 'chunk_id = va_valueid AND chunk_seq >= 0 — toast_fetch_datum_slice adds the chunk_seq key itself.',
        });
        steps.push({ what: 'first chunk', pages: 1, detail: 'A prefix needs chunk 0 only, compressed or not: pglz and LZ4 both decode a prefix without the tail.' });
        reads.push(0);
      } else {
        steps.push({ what: `first ${L.unitLabel}`, pages: 1, detail: 'The head of the chain is one pointer away.' });
        reads.push(0);
      }
      break;
    }
    case 'middle': {
      steps.push({ what: 'heap / clustered page', pages: 1, detail: 'The row, and with it the pointer.' });
      if (L.off === 0) break;
      const mid = Math.floor(L.units / 2);
      if (isPg && !L.compressed) {
        steps.push({
          what: 'toast index probe',
          pages: L.idxDescent,
          detail: `chunk_seq >= ${fmtNum(mid)}: the offset divides by ${PG_CHUNK}, so the chunk holding it is computed, not searched.`,
        });
        steps.push({ what: 'one chunk', pages: 1, detail: 'This is exactly what SET STORAGE EXTERNAL buys you.' });
        reads.push(mid);
      } else if (isPg) {
        steps.push({
          what: 'toast index probe',
          pages: L.idxDescent,
          detail: 'Compressed: the offset in the plaintext says nothing about which chunk holds it.',
        });
        steps.push({
          what: 'chunks 0 … mid',
          pages: Math.max(1, Math.ceil((mid + 1) / 4)),
          detail: 'Decompression is sequential, so every chunk up to the slice has to be fetched and decoded.',
        });
        for (let i = 0; i <= mid; i++) reads.push(i);
      } else {
        steps.push({
          what: `${L.unitLabel}s 0 … mid`,
          pages: Math.max(1, mid + 1),
          detail: 'A singly-linked chain has no random access: the only way to page N is through pages 0..N−1.',
        });
        for (let i = 0; i <= mid; i++) reads.push(i);
      }
      break;
    }
    case 'updOther': {
      if (isPg) {
        steps.push({ what: 'new heap tuple', pages: 1, detail: 'The 18-byte varatt_external pointer is copied verbatim into the new version.' });
        steps.push({ what: 'toast writes', pages: 0, detail: 'None. Both row versions reference the same chunk_id until VACUUM removes the dead one.' });
        writes = 1;
      } else if (engine === 'sqlite') {
        steps.push({ what: 'leaf cell', pages: 1, detail: 'SQLite rewrites the whole record.' });
        steps.push({ what: 'overflow chain', pages: L.storePages, detail: 'The payload is one byte string: touching any column rewrites all of it.' });
        writes = 1 + L.storePages;
      } else {
        steps.push({ what: 'clustered record', pages: 1, detail: 'Updated in place; the old version lives in the undo log.' });
        steps.push({ what: 'BLOB pages', pages: 0, detail: 'The 20-byte pointer is unchanged, so the chain is not touched.' });
        writes = 1;
      }
      break;
    }
    case 'updValue': {
      steps.push({ what: 'row', pages: 1, detail: 'A new row version with a new pointer.' });
      if (isPg) {
        steps.push({
          what: 'new chunks + index entries',
          pages: L.storePages + L.idxFull,
          detail: `A fresh chunk_id: all ${fmtNum(L.units)} chunks are written again, and the old ones become dead rows in the toast table.`,
        });
      } else {
        steps.push({
          what: `new ${L.unitLabel} chain`,
          pages: L.storePages,
          detail: 'The whole value is written again; the old chain is freed by purge (InnoDB) or the freelist (SQLite).',
        });
      }
      writes = 1 + L.storePages + (isPg ? L.idxFull : 0);
      all();
      break;
    }
    default:
      break;
  }
  return { steps, reads, writes };
}

const ACTION_HEAD: Record<Action, string> = {
  idle: 'Drag the value size and watch the row cross the threshold.',
  read: 'SELECT big_col FROM t WHERE id = 1',
  prefix: 'SELECT substr(big_col, 1, 100) …',
  middle: 'SELECT substr(big_col, <halfway>, 100) …',
  updOther: 'UPDATE t SET other_col = other_col + 1 WHERE id = 1',
  updValue: 'UPDATE t SET big_col = … WHERE id = 1',
};

/* ------------------------------------------------------------------ the figure */

const C_HDR = 'var(--viz-1)';
const C_OTHER = 'var(--viz-2)';
const C_VALUE = 'var(--viz-3)';
const C_PTR = 'var(--viz-6)';
const C_UNIT = 'var(--viz-7)';
const C_IDX = 'var(--viz-4)';

const MAX_CELLS = 60;

export default function ToastOverflowChainLab() {
  const [engine, setEngine] = useState<EngineId>('pg');
  const [sizeExp, setSizeExp] = useState(52);
  const [other, setOther] = useState(120);
  const [strat, setStrat] = useState<Strategy>('extended');
  const [algo, setAlgo] = useState<Algo>('pglz');
  const [ability, setAbility] = useState(60);
  const [action, setAction] = useState<Action>('idle');
  const [ref, width] = useSize(820);
  const tip = useTip();

  // 100 B … 10 MB on a log slider.
  const size = Math.round(100 * Math.pow((10 * 1024 * 1024) / 100, sizeExp / 100));
  const strategy: Strategy = engine === 'pg' ? strat : 'extended';
  const algorithm: Algo = engine === 'pg' ? algo : 'none';

  const L = layout(engine, size, other, strategy, algorithm, ability / 100);
  const T = trace(action, engine, L);
  const readPages = T.steps.reduce((a, s) => a + s.pages, 0);
  const readSet = new Set(T.reads);
  const isRead = action === 'read' || action === 'prefix' || action === 'middle';

  /* ----- geometry */
  const svgW = Math.max(width, 780);
  const leftW = Math.min(330, Math.max(260, svgW * 0.38));
  const rightX = leftW + 28;
  const rightW = svgW - rightX - 8;

  const barX = 8;
  const barY = 46;
  const barW = leftW - 16;
  const barH = 34;
  const scale = barW / L.pageBytes;

  const hdrBytes = engine === 'pg' ? PG_HDR : engine === 'sqlite' ? SQLITE_HDR : INNO_HDR;
  const valBytes = L.inline > 0 ? L.inline + (engine === 'pg' && L.ptr === 0 ? 4 : 0) : 0;

  /** The record, drawn to page scale, clipped at the page edge (PLAIN storage can exceed it). */
  const bars = (() => {
    const want: { key: string; bytes: number; fill: string; tip: string }[] = [
      {
        key: 'hdr',
        bytes: hdrBytes,
        fill: C_HDR,
        tip:
          engine === 'pg'
            ? 'HeapTupleHeader: 23 bytes, MAXALIGNed to 24 with no null bitmap.'
            : engine === 'sqlite'
              ? 'The record header: serial types and the rowid varints.'
              : 'Record header plus DB_TRX_ID (6 B) and DB_ROLL_PTR (7 B).',
      },
      { key: 'other', bytes: L.otherInline, fill: C_OTHER, tip: 'The other columns in this row. The threshold is about the whole record, not one value.' },
      { key: 'val', bytes: valBytes, fill: C_VALUE, tip: `${fmtBytes(L.inline)} of the value kept in the record${L.compressed ? ', compressed' : ''}.` },
      {
        key: 'ptr',
        bytes: L.ptr,
        fill: C_PTR,
        tip:
          engine === 'pg'
            ? '18-byte varatt_external: va_rawsize, va_extsize, va_valueid, va_toastrelid.'
            : engine === 'sqlite'
              ? 'The 4-byte page number of the first overflow page.'
              : '20-byte BTR_EXTERN_FIELD_REF: space id, page number, offset, length.',
      },
    ];
    let x = barX;
    let clipped = false;
    const out = want
      .filter((b) => b.bytes > 0)
      .map((b) => {
        const room = barX + barW - x;
        const w = Math.min(Math.max(2, b.bytes * scale), Math.max(0, room));
        if (b.bytes * scale > room) clipped = true;
        const r = { ...b, x, w };
        x += w;
        return r;
      });
    return { segs: out, clipped };
  })();

  const thresholdX = barX + L.threshold * scale;

  const cells = Math.min(L.units, MAX_CELLS);
  const cellW = 30;
  const cellH = 22;
  const gap = 7;
  const perRow = Math.max(4, Math.floor((rightW + gap) / (cellW + gap)));
  const gridY = barY + (engine === 'pg' ? 74 : 40);
  const rows = Math.ceil(cells / perRow);
  const height = Math.max(210, gridY + rows * (cellH + gap) + 44);

  const chunkFill = (i: number) => {
    if (isRead && readSet.has(i)) return C_UNIT;
    if (action === 'updValue' && readSet.has(i)) return 'var(--viz-stale)';
    return 'var(--viz-neutral)';
  };

  const run = (a: Action) => setAction((cur) => (cur === a ? 'idle' : a));

  /** What the row would weigh with nothing moved out of line. */
  const inlineWould = L.row - L.ptr - L.inline + L.stored + (engine === 'pg' ? 4 : 0);

  const ladder = [1024, 2048, 8192, 65536, 1048576, 10 * 1024 * 1024];

  return (
    <VizPanel
      title="Where a wide value actually lives"
      subtitle="Grow one text column and watch the row cross the out-of-line threshold: the value leaves the page, a pointer takes its place, and every read has to go and get it back."
      controls={
        <>
          <Choice label="Engine" value={engine} onChange={setEngine} options={ENGINES} />
          <Slider
            label="Value size"
            min={0}
            max={100}
            value={sizeExp}
            onChange={setSizeExp}
            format={() => fmtBytes(size)}
          />
          <Slider label="Other columns" min={0} max={1400} step={20} value={other} onChange={setOther} format={fmtBytes} />
          {engine === 'pg' ? (
            <>
              <Segmented
                label="SET STORAGE"
                value={strat}
                onChange={setStrat}
                options={[
                  { value: 'extended', label: 'EXTENDED', title: 'Compress, then move out of line. The default for text/jsonb/bytea.' },
                  { value: 'external', label: 'EXTERNAL', title: 'Never compress; move out of line. Makes substr() cheap.' },
                  { value: 'main', label: 'MAIN', title: 'Compress; moved out of line only as a last resort.' },
                  { value: 'plain', label: 'PLAIN', title: 'Neither. The only legal setting for fixed-length types.' },
                ]}
              />
              <Segmented
                label="Compression"
                value={algo}
                onChange={setAlgo}
                options={[
                  { value: 'pglz', label: 'pglz' },
                  { value: 'lz4', label: 'lz4', title: 'default_toast_compression = lz4, PostgreSQL 14+' },
                  { value: 'none', label: 'off' },
                ]}
              />
              <Slider label="Compressibility" min={0} max={95} value={ability} onChange={setAbility} format={(n) => `${n}%`} />
            </>
          ) : null}
          <Button onClick={() => run('read')} primary>
            Read the column
          </Button>
          <Button onClick={() => run('prefix')} title="A prefix slice">
            substr(1, 100)
          </Button>
          <Button onClick={() => run('middle')} title="A slice from the middle of the value">
            substr(mid, 100)
          </Button>
          <Button onClick={() => run('updOther')}>UPDATE another column</Button>
          <Button onClick={() => run('updValue')}>UPDATE this column</Button>
        </>
      }
      legend={
        <Legend
          items={[
            { label: 'tuple header', color: C_HDR },
            { label: 'other columns', color: C_OTHER },
            { label: 'the wide value, in the row', color: C_VALUE },
            { label: 'out-of-line pointer', color: C_PTR },
            { label: `${L.unitLabel}s touched`, color: C_UNIT },
            ...(engine === 'pg' ? [{ label: 'toast index pages', color: C_IDX }] : []),
          ]}
        />
      }
      stats={
        <Stats
          items={[
            { label: 'Stored size', value: fmtBytes(L.stored), hint: L.compressed ? 'After compression' : 'Uncompressed' },
            { label: 'In the row', value: L.ptr > 0 ? `${fmtBytes(L.inline)} + ${L.ptr} B ptr` : fmtBytes(L.inline) },
            { label: 'Row bytes', value: `${fmtBytes(L.row)} / ${fmtBytes(L.threshold)}`, hint: L.thresholdLabel },
            { label: `Out of line`, value: L.off > 0 ? `${fmtNum(L.units)} ${L.unitLabel}s` : '—' },
            {
              label: isRead ? 'Pages read' : 'Pages read, full fetch',
              value: fmtNum(isRead ? readPages : 1 + L.idxFull + L.storePages),
              hint: 'Including the page holding the row itself',
            },
            {
              label: 'Pages written, last UPDATE',
              value: action === 'updOther' || action === 'updValue' ? fmtNum(T.writes) : '—',
            },
          ]}
        />
      }
      note={
        <Note>
          <strong>{ACTION_HEAD[action]}</strong> {L.error ? `${L.error}. ` : ''}
          {L.why} {engine === 'pg' && L.compWhy ? L.compWhy : ''}{' '}
          {action !== 'idle' && T.steps.length > 0
            ? T.steps.map((s) => `${s.what}: ${s.pages} page${s.pages === 1 ? '' : 's'}`).join(' · ') + '.'
            : ''}
        </Note>
      }
      table={
        <>
          <table className="viz-table">
            <thead>
              <tr>
                <th>Step</th>
                <th>What is read or written</th>
                <th>Pages</th>
                <th>Why</th>
              </tr>
            </thead>
            <tbody>
              {T.steps.length === 0 ? (
                <tr>
                  <td colSpan={4}>Pick an action above.</td>
                </tr>
              ) : (
                T.steps.map((s, i) => (
                  <tr key={s.what}>
                    <td>{i + 1}</td>
                    <td>{s.what}</td>
                    <td>{fmtNum(s.pages)}</td>
                    <td>{s.detail}</td>
                  </tr>
                ))
              )}
            </tbody>
          </table>
          <table className="viz-table">
            <thead>
              <tr>
                <th>Value size</th>
                <th>Stored</th>
                <th>In the row</th>
                <th>Out of line</th>
                <th>{L.unitLabel}s</th>
                <th>Pages, full read</th>
                <th>Pages, mid slice</th>
              </tr>
            </thead>
            <tbody>
              {ladder.map((b) => {
                const l = layout(engine, b, other, strategy, algorithm, ability / 100);
                const full = trace('read', engine, l);
                const mid = trace('middle', engine, l);
                return (
                  <tr key={b}>
                    <td>{fmtBytes(b)}</td>
                    <td>{fmtBytes(l.stored)}</td>
                    <td>{l.ptr > 0 ? `${fmtBytes(l.inline)} + ${l.ptr} B ptr` : fmtBytes(l.inline)}</td>
                    <td>{l.off > 0 ? fmtBytes(l.off) : '—'}</td>
                    <td>{fmtNum(l.units)}</td>
                    <td>{fmtNum(full.steps.reduce((a, s) => a + s.pages, 0))}</td>
                    <td>{fmtNum(mid.steps.reduce((a, s) => a + s.pages, 0))}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </>
      }
    >
      <div ref={ref}>
        <TooltipHost>
          <svg width={svgW} height={height} role="img" aria-label="One row and its out-of-line storage, with the pages each read touches">
            {/* ---------------------------------------------------------- the row */}
            <text x={barX} y={18} fill="var(--viz-ink)" fontWeight={600}>
              {engine === 'pg' ? 'Heap page — 8 KB' : engine === 'sqlite' ? 'Table b-tree leaf — 4 KB' : 'Clustered index leaf — 16 KB'}
            </text>
            <text x={barX} y={34} fill="var(--viz-ink-muted)">
              one row, drawn to page scale
            </text>

            <rect x={barX} y={barY} width={barW} height={barH} rx={6} fill="var(--viz-plane)" stroke="var(--viz-border)" />

            {bars.segs.map((b) => (
              <g key={b.key} {...tip(<>{b.tip}</>)}>
                <rect x={b.x} y={barY} width={b.w} height={barH} fill={b.fill} />
              </g>
            ))}
            {bars.clipped ? (
              <text x={barX + barW + 4} y={barY + 22} fill="var(--viz-critical)">
                ▸
              </text>
            ) : null}

            <line x1={thresholdX} x2={thresholdX} y1={barY - 8} y2={barY + barH + 8} stroke="var(--viz-critical)" strokeWidth={1.5} strokeDasharray="5 4" />
            <text x={Math.min(thresholdX + 6, barX + barW - 4)} y={barY + barH + 22} fill="var(--viz-critical)" textAnchor={thresholdX > barX + barW * 0.6 ? 'end' : 'start'}>
              {L.thresholdLabel}
            </text>

            <text x={barX} y={barY + barH + 40} fill="var(--viz-ink-2)">
              row = {fmtBytes(L.row)}
              {L.ptr > 0 ? `  —  ${fmtBytes(inlineWould)} if it had stayed in the row` : ''}
            </text>
            {L.error ? (
              <text x={barX} y={barY + barH + 58} fill="var(--viz-critical)">
                {L.error}
              </text>
            ) : null}

            {/* -------------------------------------------------- out-of-line store */}
            <text x={rightX} y={18} fill="var(--viz-ink)" fontWeight={600}>
              {engine === 'pg'
                ? 'pg_toast.pg_toast_<oid>'
                : engine === 'sqlite'
                  ? 'Overflow page chain'
                  : 'Off-page BLOB chain'}
            </text>
            <text x={rightX} y={34} fill="var(--viz-ink-muted)">
              {L.off > 0
                ? engine === 'pg'
                  ? `${fmtNum(L.units)} chunks of ${PG_CHUNK} B · ${fmtNum(L.storePages)} pages · keyed (chunk_id, chunk_seq)`
                  : `${fmtNum(L.units)} pages, each pointing at the next`
                : 'empty — nothing has left the row'}
            </text>

            {engine === 'pg' && L.units > 0 ? (
              <g>
                <rect x={rightX} y={barY} width={54} height={cellH} rx={4} fill={isRead ? C_IDX : 'var(--viz-neutral)'} stroke="var(--viz-border)" />
                <text x={rightX + 62} y={barY + 15} fill="var(--viz-ink-2)">
                  toast index: {fmtNum(L.idxFull)} page{L.idxFull === 1 ? '' : 's'} for a full fetch, {fmtNum(L.idxDescent)} for one probe
                </text>
                <text x={rightX + 6} y={barY + 15} fill="var(--viz-ink)">
                  idx
                </text>
              </g>
            ) : null}

            {Array.from({ length: cells }, (_, i) => {
              const col = i % perRow;
              const row = Math.floor(i / perRow);
              const x = rightX + col * (cellW + gap);
              const y = gridY + row * (cellH + gap);
              const touched = readSet.has(i);
              return (
                <g
                  key={i}
                  {...tip(
                    <>
                      <strong>
                        {L.unitLabel} {i}
                      </strong>
                      <br />
                      {engine === 'pg'
                        ? `chunk_seq = ${i}, up to ${PG_CHUNK} bytes of chunk_data`
                        : `${fmtBytes(engine === 'sqlite' ? SQLITE_OVF : INNO_BLOB)} of payload, plus the next page number`}
                      {touched ? ' — read by this query' : ''}
                    </>,
                  )}
                >
                  <rect
                    x={x}
                    y={y}
                    width={cellW}
                    height={cellH}
                    rx={3}
                    fill={chunkFill(i)}
                    stroke={touched ? 'var(--viz-good)' : 'var(--viz-border)'}
                    strokeWidth={touched ? 2 : 1}
                  />
                  <text x={x + cellW / 2} y={y + 15} textAnchor="middle" fill="var(--viz-ink-2)">
                    {touched ? '•' : i < 3 || i === cells - 1 ? i : ''}
                  </text>
                  {engine !== 'pg' && col < perRow - 1 && i < cells - 1 ? (
                    <line x1={x + cellW} x2={x + cellW + gap} y1={y + cellH / 2} y2={y + cellH / 2} stroke="var(--viz-axis)" />
                  ) : null}
                </g>
              );
            })}

            {L.units > cells ? (
              <text x={rightX} y={gridY + rows * (cellH + gap) + 16} fill="var(--viz-ink-muted)">
                + {fmtNum(L.units - cells)} more {L.unitLabel}s not drawn ({fmtNum(L.storePages)} pages,{' '}
                {fmtBytes(L.storePages * L.pageBytes)} of storage)
              </text>
            ) : null}
          </svg>
        </TooltipHost>
      </div>
    </VizPanel>
  );
}
