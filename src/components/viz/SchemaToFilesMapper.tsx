import { useMemo, useState, type ReactNode } from 'react';
import {
  VizPanel,
  Segmented,
  Slider,
  Check,
  Legend,
  Stats,
  TooltipHost,
  useTip,
  fmtBytes,
  fmtNum,
  makeRng,
  useSize,
} from './Viz';

/**
 * One DDL script, three engines, and the files it actually produces.
 *
 * The model is the point: a table is a *catalog row* that names a physical object,
 * and an ALTER is classified by whether it can be satisfied by editing that row
 * (PostgreSQL attmissingval, InnoDB instant ADD, SQLite's sqlite_schema.sql) or
 * whether the rows themselves have to be re-materialised into new storage.
 *
 * Every size here is computed from the documented record layout of each engine —
 * page header sizes, line pointers, varlena headers, DB_TRX_ID/DB_ROLL_PTR,
 * SQLite serial types — not measured on a real instance.
 */

type EngineId = 'pg' | 'innodb' | 'sqlite';

type Verdict = 'create' | 'load' | 'build' | 'metadata' | 'rewrite' | 'rejected' | 'unsupported';

const VERDICT_LABEL: Record<Verdict, string> = {
  create: 'files created',
  load: 'rows written',
  build: 'new tree, table untouched',
  metadata: 'catalog row only',
  rewrite: 'full rewrite',
  rejected: 'refused',
  unsupported: 'no such statement',
};

const VERDICT_COLOR: Record<Verdict, string> = {
  create: 'var(--viz-ink-2)',
  load: 'var(--viz-ink-2)',
  build: 'var(--viz-good)',
  metadata: 'var(--viz-good)',
  rewrite: 'var(--viz-critical)',
  rejected: 'var(--viz-warning)',
  unsupported: 'var(--viz-warning)',
};

type ObjKind = 'data' | 'index' | 'toast' | 'map' | 'shared' | 'orphan';

const KIND_COLOR: Record<ObjKind, string> = {
  data: 'var(--viz-1)',
  index: 'var(--viz-2)',
  toast: 'var(--viz-3)',
  map: 'var(--viz-4)',
  shared: 'var(--viz-5)',
  orphan: 'var(--viz-stale)',
};

const KIND_LABEL: Record<ObjKind, string> = {
  data: 'table data',
  index: 'index',
  toast: 'out-of-line values',
  map: 'space bookkeeping',
  shared: 'shared / catalog',
  orphan: 'old or temporary copy',
};

type Obj = {
  key: string;
  name: string;
  kind: ObjKind;
  bytes: number | null; // null = real object, size not modelled here
  isNew: boolean;
  detail: string;
};

type FileGroup = { key: string; path: string; objs: Obj[] };

type EngineState = {
  engine: EngineId;
  label: string;
  root: string;
  files: FileGroup[];
  total: number;
  verdict: Verdict;
  note: string;
  lock: string;
  written: number; // bytes this statement writes
  peak: number; // bytes on disk at the worst moment of this statement
  logicalWidth: number; // row width implied by the catalog
  physicalWidth: number; // row width actually on disk
  cols: number;
};

/* ------------------------------------------------------------- the script */

type Stmt = {
  sql: string[];
  alt?: string; // how the same intent is spelled on MySQL / SQLite
};

const SCRIPT: Stmt[] = [
  { sql: ['-- empty database'] },
  {
    sql: [
      'CREATE TABLE orders (id bigint PRIMARY KEY, customer_id int NOT NULL,',
      '  code varchar(32) NOT NULL, total numeric(10,2), notes text);',
    ],
  },
  { sql: ['-- bulk load', 'COPY orders FROM …   /   LOAD DATA INFILE   /   INSERT … in one txn'] },
  { sql: ['CREATE INDEX orders_customer_idx ON orders (customer_id);'] },
  { sql: ['ALTER TABLE orders ADD COLUMN status text;'] },
  {
    sql: ['ALTER TABLE orders ADD COLUMN created_at timestamptz NOT NULL DEFAULT now();'],
    alt: 'MySQL: ADD COLUMN created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP',
  },
  {
    sql: ['ALTER TABLE orders ALTER COLUMN code TYPE varchar(64);'],
    alt: 'MySQL: MODIFY COLUMN code VARCHAR(64) NOT NULL',
  },
  {
    sql: ['ALTER TABLE orders ALTER COLUMN customer_id TYPE bigint;'],
    alt: 'MySQL: MODIFY COLUMN customer_id BIGINT NOT NULL',
  },
  {
    sql: ['ALTER TABLE orders ADD COLUMN token uuid NOT NULL DEFAULT gen_random_uuid();'],
    alt: 'MySQL: ADD COLUMN token CHAR(36) NOT NULL DEFAULT (UUID())',
  },
  {
    sql: ['CLUSTER orders USING orders_customer_idx;'],
    alt: 'MySQL: OPTIMIZE TABLE orders   ·   SQLite: VACUUM',
  },
];

const LAST = SCRIPT.length - 1;

/* ------------------------------------------------------- schema bookkeeping */

type Schema = {
  status: boolean;
  created: boolean;
  code64: boolean;
  custBig: boolean;
  token: boolean;
};

function schemaAt(engine: EngineId, s: number): Schema {
  if (engine === 'sqlite') {
    // created_at is refused (non-constant default); the two ALTER COLUMNs do not exist.
    return { status: s >= 4, created: false, code64: false, custBig: false, token: false };
  }
  return { status: s >= 4, created: s >= 5, code64: s >= 6, custBig: s >= 7, token: s >= 8 };
}

/** Steps at which each engine re-materialises every row into new storage. */
const MATERIALISE: Record<EngineId, number[]> = {
  pg: [2, 7, 8, 9],
  innodb: [2, 6, 7, 8, 9],
  sqlite: [2, 9],
};

function physStep(engine: EngineId, step: number) {
  const ms = MATERIALISE[engine].filter((m) => m <= step);
  return ms.length ? ms[ms.length - 1] : 0;
}

const align = (n: number, a: number) => Math.ceil(n / a) * a;

/** Deterministic sample of the loaded data — average field lengths. */
const SAMPLE = (() => {
  const rng = makeRng(20260913);
  let code = 0;
  let notes = 0;
  for (let i = 0; i < 512; i++) {
    code += 6 + Math.floor(rng() * 12); // 6–17 chars
    notes += 24 + Math.floor(rng() * 80); // 24–103 chars
  }
  return { code: code / 512, notes: notes / 512 };
})();

const WIDE_NOTES = 8192; // bytes per notes value when "notes are documents" is on

/* ------------------------------------------------------------- PostgreSQL */

const PG_BLCKSZ = 8192;

function pgWidth(s: Schema, wide: boolean) {
  let w = 24; // t_hoff: 23-byte HeapTupleHeader + null bitmap, MAXALIGNed
  w += 8; // id bigint
  w = s.custBig ? align(w, 8) + 8 : align(w, 4) + 4; // customer_id
  w += 1 + Math.round(SAMPLE.code); // code: short varlena header + chars
  w = align(w, 4) + 10; // numeric(10,2)
  w += wide ? 18 : 1 + Math.round(SAMPLE.notes); // notes: TOAST pointer, or inline
  // status is NULL for every existing row: it costs a bit in the null bitmap, nothing else
  if (s.created) w = align(w, 8) + 8; // timestamptz
  if (s.token) w += 16; // uuid is char-aligned
  return align(w, 8);
}

const PG_IDX_PER_PAGE = Math.floor(((PG_BLCKSZ - 24 - 16) * 0.9) / 20); // 16-byte IndexTuple + 4-byte line pointer, fillfactor 90
const PG_TOAST_CHUNK = 1996; // TOAST_MAX_CHUNK_SIZE for BLCKSZ 8192
const PG_TOAST_PER_PAGE = Math.floor((PG_BLCKSZ - 24) / (align(24 + 8 + 4 + PG_TOAST_CHUNK, 8) + 4));

function pgBtreeBytes(entries: number) {
  if (entries <= 0) return PG_BLCKSZ; // metapage only
  const leaf = Math.ceil(entries / PG_IDX_PER_PAGE);
  const internal = Math.max(1, Math.ceil(leaf / PG_IDX_PER_PAGE));
  return (leaf + internal + 1) * PG_BLCKSZ;
}

function pgState(step: number, rows: number, wide: boolean, tblspc: boolean): EngineState {
  const logical = schemaAt('pg', step);
  const physical = schemaAt('pg', physStep('pg', step));
  const logicalWidth = pgWidth(logical, wide);
  const physicalWidth = pgWidth(physical, wide);
  const live = step >= 2 ? rows : 0;

  const perPage = Math.max(1, Math.floor((PG_BLCKSZ - 24) / (physicalWidth + 4)));
  const heapPages = Math.ceil(live / perPage);
  const heap = heapPages * PG_BLCKSZ;
  const fsm = heapPages ? (Math.ceil(heapPages / 4000) + 2) * PG_BLCKSZ : 0;
  const vm = heapPages ? Math.max(1, Math.ceil(heapPages / 32672)) * PG_BLCKSZ : 0;
  const pkey = step >= 1 ? pgBtreeBytes(live) : 0;
  const cidx = step >= 3 ? pgBtreeBytes(live) : 0;
  const chunks = wide ? live * Math.ceil(WIDE_NOTES / PG_TOAST_CHUNK) : 0;
  const toast = Math.ceil(chunks / PG_TOAST_PER_PAGE) * PG_BLCKSZ;
  const toastIdx = pgBtreeBytes(chunks);

  // relfilenodes: equal to the OID at creation, reallocated by every rewrite
  let next = 16500;
  const rf = { heap: 16391, pkey: 16396, toast: 16392, toastIdx: 16394, cidx: 16400 };
  let orphan: number | null = null;
  for (const m of [7, 8, 9]) {
    if (step >= m) {
      orphan = rf.heap;
      rf.heap = next++;
      rf.pkey = next++;
      rf.cidx = next++;
      rf.toast = next++;
      rf.toastIdx = next++;
    }
  }
  const rewriteNow = [7, 8, 9].includes(step);

  const dir = tblspc ? 'pg_tblspc/16450/PG_17_<catver>/16384' : 'base/16384';
  const f: FileGroup[] = [];
  const mk = (
    key: string,
    path: string,
    name: string,
    kind: ObjKind,
    bytes: number | null,
    isNew: boolean,
    detail: string,
  ) => f.push({ key, path, objs: [{ key, name, kind, bytes, isNew, detail }] });

  if (step >= 1) {
    f.push({
      key: 'heap',
      path: `${dir}/${rf.heap}`,
      objs: [
        {
          key: 'main',
          name: 'orders — main fork',
          kind: 'data',
          bytes: heap,
          isNew: step === 1 || rewriteNow,
          detail: `The heap itself: ${fmtNum(perPage)} tuples per 8 KB page at a physical row width of ${physicalWidth} B. Segmented into 1 GB files (${rf.heap}, ${rf.heap}.1, …) once it grows past 1 GB.`,
        },
        ...(heapPages
          ? [
              {
                key: 'fsm',
                name: `${rf.heap}_fsm — free space map`,
                kind: 'map' as ObjKind,
                bytes: fsm,
                isNew: step === 2,
                detail:
                  'A three-level tree of one byte per heap page, recording free space in 1/256ths. Created lazily — it does not exist until something needs to find room.',
              },
              {
                key: 'vm',
                name: `${rf.heap}_vm — visibility map`,
                kind: 'map' as ObjKind,
                bytes: vm,
                isNew: step === 2,
                detail:
                  'Two bits per heap page (all-visible, all-frozen). Written by VACUUM; it is what makes index-only scans and freeze skipping possible.',
              },
            ]
          : []),
      ],
    });
    mk(
      'pkey',
      `${dir}/${rf.pkey}`,
      'orders_pkey — B-tree',
      'index',
      pkey,
      step === 1 || rewriteNow,
      'A separate relation with its own pg_class row and its own file. The heap does not know it exists; only pg_index links them.',
    );
    if (step >= 3)
      mk(
        'cidx',
        `${dir}/${rf.cidx}`,
        'orders_customer_idx — B-tree',
        'index',
        cidx,
        step === 3 || rewriteNow,
        'Leaf entries are (customer_id, ctid): the physical location of the heap tuple, which is why every rewrite must rebuild every index.',
      );
    f.push({
      key: 'toast',
      path: `${dir}/${rf.toast}`,
      objs: [
        {
          key: 'toastrel',
          name: `pg_toast_16391 — TOAST table`,
          kind: 'toast',
          bytes: toast,
          isNew: step === 1 || rewriteNow,
          detail: wide
            ? `Values over ~2032 B are compressed, and if still too big, sliced into ${PG_TOAST_CHUNK}-byte chunks stored here; the heap keeps an 18-byte pointer.`
            : 'Created with the table because it has a varlena column, and 0 bytes until some value exceeds the ~2032-byte TOAST threshold.',
        },
      ],
    });
    mk(
      'toastidx',
      `${dir}/${rf.toastIdx}`,
      'pg_toast_16391_index',
      'toast',
      toastIdx,
      step === 1 || rewriteNow,
      'The (chunk_id, chunk_seq) index that reassembles a toasted value.',
    );
  }
  if (orphan !== null && rewriteNow) {
    mk(
      'orphan',
      `${dir}/${orphan}`,
      `${orphan} — old relfilenode`,
      'orphan',
      heap,
      false,
      'The old file still exists while the rewrite runs and is unlinked at commit. Peak disk usage is old + new, which is how a 2 AM ALTER fills a volume.',
    );
  }
  f.push({
    key: 'cat',
    path: `${dir}/1259, /1249, /2610 …`,
    objs: [
      {
        key: 'catalog',
        name: 'pg_class, pg_attribute, pg_index',
        kind: 'shared',
        bytes: null,
        isNew: false,
        detail:
          'The catalogs are ordinary heap relations with their own files. The bootstrap problem — you cannot read pg_class to find pg_class — is solved by nailed relations whose filenode comes from base/16384/pg_filenode.map.',
      },
    ],
  });

  const dataTotal = heap + fsm + vm + pkey + cidx + toast + toastIdx;
  const verdictByStep: Record<number, { v: Verdict; note: string; lock: string }> = {
    0: { v: 'create', note: 'Nothing created yet.', lock: 'none' },
    1: {
      v: 'create',
      note: 'Four pg_class rows — heap, PK index, TOAST table, TOAST index — four files, and rows in pg_type and pg_attribute besides. The heap file is 0 bytes; the FSM and VM forks do not exist yet.',
      lock: 'none (new relation)',
    },
    2: {
      v: 'load',
      note: `${fmtNum(live)} tuples at ${physicalWidth} B each, ${fmtNum(perPage)} per 8 KB page. The FSM and VM forks appear here.`,
      lock: 'RowExclusiveLock',
    },
    3: {
      v: 'build',
      note: 'A new pg_class row and a new file. The heap is read but never written; CREATE INDEX takes a ShareLock (blocks writers); CREATE INDEX CONCURRENTLY trades a second heap scan, plus a wait for older transactions, for not blocking them.',
      lock: 'ShareLock (or SHARE UPDATE EXCLUSIVE with CONCURRENTLY)',
    },
    4: {
      v: 'metadata',
      note: 'One pg_attribute row with attnum 6. The default is NULL, so nothing needs to be filled in: a row with five attributes is simply read as having a NULL sixth. No file is touched.',
      lock: 'AccessExclusiveLock, held for microseconds',
    },
    5: {
      v: 'metadata',
      note: 'now() is STABLE, not VOLATILE, so PostgreSQL 11+ evaluates it once, stores the result in pg_attribute.attmissingval, sets atthasmissing, and returns it for every row too short to contain the column.',
      lock: 'AccessExclusiveLock, held for microseconds',
    },
    6: {
      v: 'metadata',
      note: 'Raising a varchar length limit is a no-op coercion: the on-disk representation of varchar(32) and varchar(64) is identical, so atttypmod changes and no rewrite or reindex happens.',
      lock: 'AccessExclusiveLock, held for microseconds',
    },
    7: {
      v: 'rewrite',
      note: 'int → bigint changes the bytes, so every tuple is re-formed into a brand-new relfilenode and every index is rebuilt against the new ctids. AccessExclusiveLock is held for the whole scan.',
      lock: 'AccessExclusiveLock, held for the whole rewrite',
    },
    8: {
      v: 'rewrite',
      note: 'gen_random_uuid() is VOLATILE, so there is no single missing value to store — each row needs its own. That is the entire difference between this ALTER and the one two steps ago, and it costs a full rewrite.',
      lock: 'AccessExclusiveLock, held for the whole rewrite',
    },
    9: {
      v: 'rewrite',
      note: 'CLUSTER rewrites the heap in index order into a new relfilenode and rebuilds every index. It is a one-shot physical reordering: nothing maintains it, so the ordering decays with every UPDATE.',
      lock: 'AccessExclusiveLock, held for the whole rewrite',
    },
  };
  const v = verdictByStep[step];
  const written =
    step === 2 ? dataTotal : v.v === 'rewrite' ? dataTotal : v.v === 'build' ? cidx : 0;

  return {
    engine: 'pg',
    label: 'PostgreSQL 17',
    root: `$PGDATA/${dir}/`,
    files: f,
    total: dataTotal,
    verdict: v.v,
    note: v.note,
    lock: v.lock,
    written,
    peak: v.v === 'rewrite' ? dataTotal * 2 : dataTotal,
    logicalWidth,
    physicalWidth,
    cols: 5 + (logical.status ? 1 : 0) + (logical.created ? 1 : 0) + (logical.token ? 1 : 0),
  };
}

/* ----------------------------------------------------------------- InnoDB */

const IB_PAGE = 16384;
const IB_USABLE = IB_PAGE - 130; // FIL header/trailer, page header, infimum/supremum
const IB_FILL = 15 / 16; // pages built by a sorted insert are filled 15/16

function ibWidth(s: Schema, wide: boolean) {
  let w = 5 + 1; // record header + nullable-column bitmap
  w += 6 + 7; // DB_TRX_ID + DB_ROLL_PTR (clustered index only)
  w += 8; // id bigint (the key, stored in the record)
  w += s.custBig ? 8 : 4; // customer_id
  w += (s.code64 ? 2 : 1) + Math.round(SAMPLE.code); // VARCHAR length prefix + bytes
  w += 5; // DECIMAL(10,2)
  w += 2 + (wide ? 20 : Math.round(SAMPLE.notes)); // TEXT: length bytes, then inline or 20-byte LOB pointer
  if (s.created) w += 4; // TIMESTAMP
  if (s.token) w += 36; // CHAR(36)
  return w;
}

function ibState(step: number, rows: number, wide: boolean, tblspc: boolean): EngineState {
  const logical = schemaAt('innodb', step);
  const physical = schemaAt('innodb', physStep('innodb', step));
  const logicalWidth = ibWidth(logical, wide);
  const physicalWidth = ibWidth(physical, wide);
  const live = step >= 2 ? rows : 0;

  const perPage = Math.max(1, Math.floor((IB_USABLE * IB_FILL) / (physicalWidth + 2)));
  const clusLeaf = Math.ceil(live / perPage);
  const clus = (clusLeaf + Math.max(1, Math.ceil(clusLeaf / 500)) + 1) * IB_PAGE;
  const secEntry = 5 + (physical.custBig ? 8 : 4) + 8 + 2; // key + PK + header + slot
  const secPerPage = Math.floor((IB_USABLE * IB_FILL) / secEntry);
  const secLeaf = step >= 3 ? Math.ceil(live / secPerPage) : 0;
  const sec = step >= 3 ? (secLeaf + Math.max(1, Math.ceil(secLeaf / 500)) + 1) * IB_PAGE : 0;
  const lob = wide ? live * IB_PAGE : 0; // one 16 KB LOB page per 8 KB value
  const meta = step >= 1 ? 6 * IB_PAGE : 0; // FSP header, ibuf bitmap, inode, segment headers
  const raw = clus + sec + lob + meta;
  // the file is extended one 1 MB extent at a time
  const ibd = raw ? Math.ceil(raw / (1024 * 1024)) * 1024 * 1024 : 0;

  const spaceId = 42;
  const path = tblspc ? '/mnt/nvme/fast_ssd.ibd' : '/var/lib/mysql/shop/orders.ibd';
  const rewriteNow = [6, 7, 8, 9].includes(step);

  const f: FileGroup[] = [];
  if (step >= 1) {
    f.push({
      key: 'ibd',
      path,
      objs: [
        {
          key: 'clus',
          name: 'PRIMARY — clustered B+tree',
          kind: 'data',
          bytes: clus,
          isNew: step === 1 || rewriteNow,
          detail: `The table *is* this tree: leaf pages hold the full row keyed by id, ${fmtNum(perPage)} per 16 KB page at ${physicalWidth} B each. There is no separate heap to point at.`,
        },
        ...(step >= 3
          ? [
              {
                key: 'sec',
                name: 'orders_customer_idx — B+tree',
                kind: 'index' as ObjKind,
                bytes: sec,
                isNew: step === 3 || rewriteNow,
                detail:
                  'Leaf entries are (customer_id, id) — the primary key, not a physical address. Lookups cost a second descent of the clustered tree, and a table rebuild does not invalidate them.',
              },
            ]
          : []),
        ...(wide
          ? [
              {
                key: 'lob',
                name: 'off-page LOB pages',
                kind: 'toast' as ObjKind,
                bytes: lob,
                isNew: step === 2 || rewriteNow,
                detail:
                  'DYNAMIC row format stores a 20-byte pointer in the record and the whole value in chained 16 KB pages inside the same tablespace file.',
              },
            ]
          : []),
        {
          key: 'meta',
          name: 'FSP header, INODE, ibuf bitmap',
          kind: 'map',
          bytes: meta,
          isNew: step === 1,
          detail:
            'Page 0 is the space header, page 1 the change-buffer bitmap, page 2 the INODE page listing the file segments each index owns. A segment\'s first 32 pages are allocated individually — a fresh .ibd is 112 KB — and past that the file grows in whole 1 MB extents.',
        },
      ],
    });
  }
  if (rewriteNow) {
    f.push({
      key: 'tmp',
      path: '/var/lib/mysql/shop/#sql-ib1089-*.ibd',
      objs: [
        {
          key: 'tmpibd',
          name: 'rebuild target, renamed at commit',
          kind: 'orphan',
          bytes: clus + sec + lob,
          isNew: true,
          detail:
            'InnoDB builds a second complete copy of the table in a temporary tablespace, applies the online DDL row log on top, then swaps the files. Peak disk is old + new, plus innodb_online_alter_log_max_size for the row log.',
        },
      ],
    });
  }
  f.push({
    key: 'shared',
    path: '/var/lib/mysql/',
    objs: [
      {
        key: 'dd',
        name: 'mysql.ibd — data dictionary',
        kind: 'shared',
        bytes: null,
        isNew: false,
        detail:
          'Since MySQL 8.0 the dictionary lives in InnoDB tables inside mysql.ibd — which is what made DDL crash-safe and atomic, and what killed the .frm file.',
      },
      {
        key: 'undo',
        name: 'undo_001, undo_002, #innodb_redo/',
        kind: 'shared',
        bytes: null,
        isNew: false,
        detail:
          'Old row versions live in the undo tablespaces, not in the table — the opposite of PostgreSQL, where dead tuples stay in the heap until VACUUM.',
      },
    ],
  });

  const byStep: Record<number, { v: Verdict; note: string; lock: string }> = {
    0: { v: 'create', note: 'Nothing created yet.', lock: 'none' },
    1: {
      v: 'create',
      note: 'One file. The table, its primary key and its row format are all the same object: a clustered B+tree whose leaves are the rows. Rows in mysql.ibd describe it.',
      lock: 'none (new table)',
    },
    2: {
      v: 'load',
      note: `${fmtNum(live)} rows at ${physicalWidth} B, ${fmtNum(perPage)} per 16 KB page. Inserting in primary-key order fills pages to 15/16; random-order inserts leave them around half full and the file is much bigger.`,
      lock: 'row locks',
    },
    3: {
      v: 'build',
      note: 'A second B+tree inside the same .ibd, built by sorting. ALGORITHM=INPLACE: the table is not rebuilt and concurrent DML is allowed, replayed afterwards from the online DDL row log.',
      lock: 'metadata lock; brief exclusive at start and end',
    },
    4: {
      v: 'metadata',
      note: 'ALGORITHM=INSTANT: the column is appended to the data dictionary and INSTANT_COLS records how many columns the old rows have. Rows are not touched; short records are extended on read.',
      lock: 'metadata lock, held for milliseconds',
    },
    5: {
      v: 'metadata',
      note: 'Also instant. The default is stored once in the dictionary and materialised for every pre-existing row on read.',
      lock: 'metadata lock, held for milliseconds',
    },
    6: {
      v: 'rewrite',
      note: 'The trap: in utf8mb4, VARCHAR(32) is 128 bytes and needs a 1-byte length prefix, VARCHAR(64) is 256 bytes and needs 2. Crossing the 255-byte line changes the record format, so the "harmless" widening is ALGORITHM=COPY — a full rebuild.',
      lock: 'metadata lock; COPY blocks writes',
    },
    7: {
      v: 'rewrite',
      note: 'Changing a column type is ALGORITHM=COPY. Every row is re-encoded into a new tablespace and every index rebuilt.',
      lock: 'metadata lock; COPY blocks writes',
    },
    8: {
      v: 'rewrite',
      note: 'Instant ADD stores exactly one default value in the dictionary, so a per-row value cannot come from it, however you spell it. The rows have to be touched — which is what gh-ost and pt-online-schema-change exist to schedule.',
      lock: 'metadata lock; rebuild',
    },
    9: {
      v: 'rewrite',
      note: 'OPTIMIZE TABLE maps to ALTER TABLE … FORCE: rebuild into a fresh tablespace and rename. Note what you do not get — the clustered index is already in primary-key order, so this compacts, it does not reorder.',
      lock: 'metadata lock; rebuild',
    },
  };
  const v = byStep[step];
  const written = step === 2 ? raw : v.v === 'rewrite' ? clus + sec + lob : v.v === 'build' ? sec : 0;

  return {
    engine: 'innodb',
    label: 'MySQL 8.0 / InnoDB',
    root: path,
    files: f,
    total: ibd,
    verdict: v.v,
    note: v.note,
    lock: v.lock,
    written,
    peak: v.v === 'rewrite' ? ibd * 2 : ibd,
    logicalWidth,
    physicalWidth,
    cols: 5 + (logical.status ? 1 : 0) + (logical.created ? 1 : 0) + (logical.token ? 1 : 0),
  };
}

/* ----------------------------------------------------------------- SQLite */

const LITE_PAGE = 4096;
const LITE_MAX_LOCAL = Math.floor(((LITE_PAGE - 12) * 64) / 255) - 23; // 1002 bytes

function liteCell(s: Schema, wide: boolean, withoutRowid: boolean) {
  const payload =
    1 + // record header length varint
    6 + // one serial-type byte per column (id, customer_id, code, total, notes, status)
    4 + // id
    4 + // customer_id
    Math.round(SAMPLE.code) +
    8 + // total as REAL
    (wide ? Math.min(WIDE_NOTES, LITE_MAX_LOCAL) : Math.round(SAMPLE.notes));
  // status is NULL: serial type 0, zero payload bytes
  const key = withoutRowid ? 0 : 4; // rowid varint, absent in a WITHOUT ROWID table
  return 2 + key + payload + 2; // payload-length varint + key + payload + cell pointer
}

function liteState(step: number, rows: number, wide: boolean, withoutRowid: boolean): EngineState {
  const logical = schemaAt('sqlite', step);
  const physical = schemaAt('sqlite', physStep('sqlite', step));
  const logicalWidth = liteCell(logical, wide, withoutRowid);
  const physicalWidth = liteCell(physical, wide, withoutRowid);
  const live = step >= 2 ? rows : 0;

  const perPage = Math.max(1, Math.floor(((LITE_PAGE - 8) * 0.95) / physicalWidth));
  const leaf = Math.ceil(live / perPage);
  const tablePages = leaf ? leaf + Math.max(1, Math.ceil(leaf / 200)) : 0;
  const idxCell = 2 + 1 + 2 + 4 + 4 + 2;
  const idxPerPage = Math.floor(((LITE_PAGE - 8) * 0.95) / idxCell);
  const idxPages = (n: number) => (n ? Math.ceil(n / idxPerPage) + 1 : 1);
  const autoPages = withoutRowid ? 0 : step >= 1 ? idxPages(live) : 0;
  const cidxPages = step >= 3 ? idxPages(live) : 0;
  const overflowPerRow = wide
    ? Math.ceil((WIDE_NOTES - LITE_MAX_LOCAL) / (LITE_PAGE - 4))
    : 0;
  const overflow = overflowPerRow * live;

  const pages = (step >= 1 ? 1 : 0) + tablePages + autoPages + cidxPages + overflow;
  const dbBytes = pages * LITE_PAGE;

  const f: FileGroup[] = [];
  f.push({
    key: 'db',
    path: '/srv/shop.db',
    objs: [
      {
        key: 'schema',
        name: 'page 1 — sqlite_schema',
        kind: 'shared',
        bytes: step >= 1 ? LITE_PAGE : 0,
        isNew: false,
        detail:
          'The catalog is a b-tree rooted at page 1 of the same file, and its own row is not in it — the parser hard-codes sqlite_schema. Every DDL statement rewrites a row here and bumps the schema cookie in the 100-byte file header.',
      },
      {
        key: 'tab',
        name: withoutRowid
          ? 'orders — WITHOUT ROWID b-tree (keyed by id)'
          : 'orders — table b-tree (keyed by rowid)',
        kind: 'data',
        bytes: tablePages * LITE_PAGE,
        isNew: step === 1 || step === 9,
        detail: withoutRowid
          ? 'WITHOUT ROWID makes the table a b-tree keyed by the declared primary key — clustered, like InnoDB. Secondary indexes then store the primary key, and there is no autoindex.'
          : `A rowid table: the b-tree key is the implicit 64-bit rowid, ${fmtNum(perPage)} records per 4 KB page. Note that id is declared bigint, not INTEGER, so it is not a rowid alias.`,
      },
      ...(autoPages
        ? [
            {
              key: 'auto',
              name: 'sqlite_autoindex_orders_1',
              kind: 'index' as ObjKind,
              bytes: autoPages * LITE_PAGE,
              isNew: step === 1 || step === 9,
              detail:
                'Only a column declared exactly INTEGER PRIMARY KEY becomes an alias for the rowid. "bigint" has INTEGER affinity but is not that spelling, so SQLite enforces the primary key with a separate unique index — a whole extra b-tree bought with one word.',
            },
          ]
        : []),
      ...(cidxPages
        ? [
            {
              key: 'cidx',
              name: 'orders_customer_idx',
              kind: 'index' as ObjKind,
              bytes: cidxPages * LITE_PAGE,
              isNew: step === 3 || step === 9,
              detail: withoutRowid
                ? 'Entries are (customer_id, id): in a WITHOUT ROWID table the primary key is the row locator.'
                : 'Entries are (customer_id, rowid). A new root page inside the same file, recorded in sqlite_schema.rootpage.',
            },
          ]
        : []),
      ...(overflow
        ? [
            {
              key: 'ovf',
              name: 'overflow page chains',
              kind: 'toast' as ObjKind,
              bytes: overflow * LITE_PAGE,
              isNew: step === 2 || step === 9,
              detail: `A cell keeps the first ${LITE_MAX_LOCAL} bytes of its payload in the b-tree page; the rest spills into a singly linked chain of 4 KB pages, 4 bytes of each being the next-page pointer.`,
            },
          ]
        : []),
    ],
  });
  f.push({
    key: 'wal',
    path: '/srv/shop.db-wal, /srv/shop.db-shm',
    objs: [
      {
        key: 'walf',
        name: 'write-ahead log + shared-memory index',
        kind: 'shared',
        bytes: step >= 2 ? 1000 * LITE_PAGE : 0,
        isNew: false,
        detail:
          'In WAL mode the log hovers around wal_autocheckpoint pages (1000 by default) before a checkpoint folds it back into the main file. The -shm file is the reader/writer index and is rebuilt from scratch after a crash.',
      },
    ],
  });

  const byStep: Record<number, { v: Verdict; note: string; lock: string }> = {
    0: { v: 'create', note: 'Nothing created yet.', lock: 'none' },
    1: {
      v: 'create',
      note: 'No new files: one row in sqlite_schema and a new b-tree root page inside the existing database file. The "path" of a table is a page number.',
      lock: 'write lock on the database',
    },
    2: {
      v: 'load',
      note: `${fmtNum(live)} records at ~${physicalWidth} B, ${fmtNum(perPage)} per 4 KB page. Pages are appended to the one file; nothing distinguishes table pages from index pages except what points at them.`,
      lock: 'write lock on the database',
    },
    3: {
      v: 'build',
      note: 'Another root page in the same file and another sqlite_schema row. There is one writer for the whole database, so building this index blocks every other write, not just writes to orders.',
      lock: 'write lock on the database',
    },
    4: {
      v: 'metadata',
      note: 'SQLite rewrites the CREATE TABLE text in sqlite_schema and increments the schema cookie. Existing records are simply short; missing trailing columns are read as their default. O(1) regardless of table size.',
      lock: 'write lock, held for microseconds',
    },
    5: {
      v: 'rejected',
      note: 'Error: "Cannot add a column with non-constant default". ADD COLUMN is metadata-only by construction, so SQLite refuses the cases that would need per-row work: CURRENT_TIMESTAMP, a parenthesised expression, or NOT NULL without a constant default.',
      lock: 'none — the statement fails',
    },
    6: {
      v: 'unsupported',
      note: 'There is no ALTER COLUMN in SQLite, and it would change nothing if there were: varchar(64) and varchar(32) are the same TEXT affinity and the length is never enforced.',
      lock: 'none — syntax error',
    },
    7: {
      v: 'unsupported',
      note: 'Same answer. The documented workaround is the 12-step procedure: create a new table, INSERT … SELECT, drop, rename — an explicit rewrite you write by hand.',
      lock: 'none — syntax error',
    },
    8: {
      v: 'rejected',
      note: 'No uuid type, and the non-constant default is refused for the same reason as created_at.',
      lock: 'none — the statement fails',
    },
    9: {
      v: 'rewrite',
      note: 'VACUUM rebuilds the entire database into a temporary file and copies it back: every table and every index, not just orders, re-laid out from page 1 upward and any freelist pages dropped. It needs roughly twice the database size in free space.',
      lock: 'exclusive lock on the database',
    },
  };
  const v = byStep[step];
  const written = step === 2 ? dbBytes : v.v === 'rewrite' ? dbBytes : v.v === 'build' ? cidxPages * LITE_PAGE : 0;

  return {
    engine: 'sqlite',
    label: 'SQLite 3',
    root: '/srv/shop.db',
    files: f,
    total: dbBytes,
    verdict: v.v,
    note: v.note,
    lock: v.lock,
    written,
    peak: v.v === 'rewrite' ? dbBytes * 2 : dbBytes,
    logicalWidth,
    physicalWidth,
    cols: 5 + (logical.status ? 1 : 0),
  };
}

/* ------------------------------------------------------------------ figure */

const COL_W = 306;
const GAP = 14;
const HEAD = 128;
const ROW_H = 22;
const GROUP_GAP = 12;

function Columns({
  states,
  width,
  focus,
  step,
}: {
  states: EngineState[];
  width: number;
  focus: EngineId;
  step: number;
}) {
  const tip = useTip();
  const all = states.flatMap((s) => s.files.flatMap((g) => g.objs.map((o) => o.bytes ?? 0)));
  const max = Math.max(1, ...all);
  const bar = (b: number) => (b <= 0 ? 0 : Math.max(2, (Math.log10(b + 1) / Math.log10(max + 1)) * 78));

  const heights = states.map(
    (s) => s.files.reduce((a, g) => a + 15 + g.objs.length * ROW_H + GROUP_GAP, 0) + 6,
  );
  const height = HEAD + Math.max(...heights);
  const svgW = Math.max(width, COL_W * 3 + GAP * 2);

  return (
    <svg
      width={svgW}
      height={height}
      role="img"
      aria-label={`On-disk objects after step ${step} of the DDL script, on three engines`}
    >
      {SCRIPT.map((_, i) => (
        <g key={i}>
          <rect
            x={i * 26}
            y={4}
            width={22}
            height={5}
            rx={2}
            fill={i === step ? 'var(--viz-1)' : i < step ? 'var(--viz-ink-muted)' : 'var(--viz-grid)'}
          />
          <text x={i * 26} y={22} fontSize={9} fill={i === step ? 'var(--viz-ink)' : 'var(--viz-ink-muted)'}>
            S{i}
          </text>
        </g>
      ))}
      {SCRIPT[step].sql.map((line, i) => (
        <text key={i} x={0} y={40 + i * 15} fontSize={12} fill="var(--viz-ink)">
          {line}
        </text>
      ))}
      {states.map((s, ci) => {
        const x0 = ci * (COL_W + GAP);
        let y = HEAD;
        const rows: ReactNode[] = [];
        for (const g of s.files) {
          rows.push(
            <text key={`${g.key}-p`} x={x0 + 2} y={y + 10} fontSize={10} fill="var(--viz-ink-muted)">
              {g.path.length > 48 ? g.path.slice(0, 47) + '…' : g.path}
            </text>,
          );
          y += 15;
          for (const o of g.objs) {
            const oy = y;
            const w = bar(o.bytes ?? 0);
            rows.push(
              <g
                key={`${g.key}-${o.key}`}
                style={{ cursor: 'help' }}
                {...tip(
                  <>
                    <strong>{o.name}</strong>
                    <br />
                    {g.path}
                    <br />
                    {o.bytes === null ? 'size not modelled' : fmtBytes(o.bytes)}
                    <br />
                    <span style={{ color: 'var(--viz-ink-2)' }}>{o.detail}</span>
                  </>,
                )}
              >
                <rect x={x0} y={oy} width={COL_W - 6} height={ROW_H - 3} rx={4} fill="var(--viz-plane)" />
                <rect x={x0} y={oy} width={5} height={ROW_H - 3} rx={2} fill={KIND_COLOR[o.kind]} />
                <text x={x0 + 11} y={oy + 13} fontSize={11} fill="var(--viz-ink)">
                  {o.name.length > 30 ? o.name.slice(0, 29) + '…' : o.name}
                </text>
                {w > 0 ? (
                  <rect
                    x={x0 + COL_W - 158}
                    y={oy + 5}
                    width={w}
                    height={ROW_H - 13}
                    rx={2}
                    fill={KIND_COLOR[o.kind]}
                    fillOpacity={0.5}
                  />
                ) : null}
                <text
                  x={x0 + COL_W - 10}
                  y={oy + 13}
                  fontSize={10}
                  textAnchor="end"
                  fill="var(--viz-ink-2)"
                  style={{ fontVariantNumeric: 'tabular-nums' }}
                >
                  {o.bytes === null ? '—' : o.bytes === 0 ? '0 B' : fmtBytes(o.bytes)}
                </text>
                {o.isNew ? (
                  <rect
                    x={x0 - 1}
                    y={oy - 1}
                    width={COL_W - 4}
                    height={ROW_H - 1}
                    rx={5}
                    fill="none"
                    stroke="var(--viz-good)"
                    strokeWidth={1.5}
                  />
                ) : null}
              </g>,
            );
            y += ROW_H;
          }
          y += GROUP_GAP;
        }
        const vc = VERDICT_COLOR[s.verdict];
        return (
          <g key={s.engine}>
            <text
              x={x0}
              y={74}
              fontSize={12}
              fontWeight={s.engine === focus ? 700 : 500}
              fill={s.engine === focus ? 'var(--viz-ink)' : 'var(--viz-ink-2)'}
            >
              {s.label}
            </text>
            <rect x={x0} y={82} width={COL_W - 6} height={20} rx={5} fill={vc} fillOpacity={0.16} stroke={vc} />
            <text x={x0 + 8} y={96} fontSize={11} fill="var(--viz-ink)">
              {VERDICT_LABEL[s.verdict]}
            </text>
            <text
              x={x0 + COL_W - 14}
              y={96}
              fontSize={10}
              textAnchor="end"
              fill="var(--viz-ink-2)"
              style={{ fontVariantNumeric: 'tabular-nums' }}
            >
              {s.total ? fmtBytes(s.total) : '—'}
            </text>
            <text x={x0} y={116} fontSize={10} fill="var(--viz-ink-muted)">
              lock: {s.lock.length > 44 ? s.lock.slice(0, 43) + '…' : s.lock}
            </text>
            {rows}
          </g>
        );
      })}
    </svg>
  );
}

/* -------------------------------------------------------------- catalogues */

function PgCatalog({ step, tblspc }: { step: number; tblspc: boolean }) {
  const s = schemaAt('pg', step);
  let next = 16500;
  const rf = { heap: 16391, pkey: 16396, toast: 16392, toastIdx: 16394, cidx: 16400 };
  for (const m of [7, 8, 9]) {
    if (step >= m) {
      rf.heap = next++;
      rf.pkey = next++;
      rf.cidx = next++;
      rf.toast = next++;
      rf.toastIdx = next++;
    }
  }
  const spc = tblspc ? '16450' : '0';
  const rel = [
    { name: 'orders', kind: 'r', oid: 16391, fn: rf.heap },
    { name: 'orders_pkey', kind: 'i', oid: 16396, fn: rf.pkey },
    ...(step >= 3 ? [{ name: 'orders_customer_idx', kind: 'i', oid: 16400, fn: rf.cidx }] : []),
    { name: 'pg_toast_16391', kind: 't', oid: 16392, fn: rf.toast },
    { name: 'pg_toast_16391_index', kind: 'i', oid: 16394, fn: rf.toastIdx },
  ];
  const atts = [
    { n: 1, name: 'id', type: 'int8', nn: 't', miss: '' },
    { n: 2, name: 'customer_id', type: s.custBig ? 'int8' : 'int4', nn: 't', miss: '' },
    { n: 3, name: 'code', type: s.code64 ? 'varchar(64)' : 'varchar(32)', nn: 't', miss: '' },
    { n: 4, name: 'total', type: 'numeric(10,2)', nn: 'f', miss: '' },
    { n: 5, name: 'notes', type: 'text', nn: 'f', miss: '' },
    ...(s.status ? [{ n: 6, name: 'status', type: 'text', nn: 'f', miss: '' }] : []),
    ...(s.created
      ? [
          {
            n: 7,
            name: 'created_at',
            type: 'timestamptz',
            nn: 't',
            miss: step >= 7 ? '' : '2026-09-13 10:14:22+00',
          },
        ]
      : []),
    ...(s.token ? [{ n: 8, name: 'token', type: 'uuid', nn: 't', miss: '' }] : []),
  ];
  return (
    <>
      <table className="viz-table">
        <caption style={{ textAlign: 'left' }}>pg_class — one row per relation</caption>
        <thead>
          <tr>
            <th>relname</th>
            <th>relkind</th>
            <th>oid</th>
            <th>relfilenode</th>
            <th>reltablespace</th>
          </tr>
        </thead>
        <tbody>
          {step >= 1 ? (
            rel.map((r) => (
              <tr key={r.name}>
                <td>{r.name}</td>
                <td>{r.kind}</td>
                <td>{r.oid}</td>
                <td>{r.fn}</td>
                <td>{spc}</td>
              </tr>
            ))
          ) : (
            <tr>
              <td colSpan={5}>no user relations yet</td>
            </tr>
          )}
        </tbody>
      </table>
      <table className="viz-table">
        <caption style={{ textAlign: 'left' }}>
          pg_attribute — orders. attmissingval is what lets ADD COLUMN skip the rewrite.
        </caption>
        <thead>
          <tr>
            <th>attnum</th>
            <th>attname</th>
            <th>atttypid</th>
            <th>attnotnull</th>
            <th>atthasmissing</th>
            <th>attmissingval</th>
          </tr>
        </thead>
        <tbody>
          {step >= 1 ? (
            atts.map((a) => (
              <tr key={a.n}>
                <td>{a.n}</td>
                <td>{a.name}</td>
                <td>{a.type}</td>
                <td>{a.nn}</td>
                <td>{a.miss ? 't' : 'f'}</td>
                <td>{a.miss || '—'}</td>
              </tr>
            ))
          ) : (
            <tr>
              <td colSpan={6}>no user relations yet</td>
            </tr>
          )}
        </tbody>
      </table>
    </>
  );
}

function InnodbCatalog({ step }: { step: number }) {
  const s = schemaAt('innodb', step);
  const user = 5 + (s.status ? 1 : 0) + (s.created ? 1 : 0) + (s.token ? 1 : 0);
  const rebuilt = step >= 6;
  const instant = !rebuilt && (s.status || s.created) ? 5 : 0;
  return (
    <>
      <table className="viz-table">
        <caption style={{ textAlign: 'left' }}>
          information_schema.INNODB_TABLES — N_COLS counts the hidden DB_TRX_ID and DB_ROLL_PTR
        </caption>
        <thead>
          <tr>
            <th>NAME</th>
            <th>TABLE_ID</th>
            <th>N_COLS</th>
            <th>SPACE</th>
            <th>ROW_FORMAT</th>
            <th>INSTANT_COLS</th>
          </tr>
        </thead>
        <tbody>
          {step >= 1 ? (
            <tr>
              <td>shop/orders</td>
              <td>1089</td>
              <td>{user + 2}</td>
              <td>42</td>
              <td>Dynamic</td>
              <td>{instant}</td>
            </tr>
          ) : (
            <tr>
              <td colSpan={6}>no user tables yet</td>
            </tr>
          )}
        </tbody>
      </table>
      <table className="viz-table">
        <caption style={{ textAlign: 'left' }}>
          information_schema.INNODB_INDEXES — PAGE_NO is the root page inside the .ibd
        </caption>
        <thead>
          <tr>
            <th>NAME</th>
            <th>INDEX_ID</th>
            <th>TYPE</th>
            <th>N_FIELDS</th>
            <th>PAGE_NO</th>
            <th>SPACE</th>
          </tr>
        </thead>
        <tbody>
          {step >= 1 ? (
            <>
              <tr>
                <td>PRIMARY</td>
                <td>{rebuilt ? 204 : 187}</td>
                <td>3 (clustered)</td>
                <td>1</td>
                <td>4</td>
                <td>42</td>
              </tr>
              {step >= 3 ? (
                <tr>
                  <td>orders_customer_idx</td>
                  <td>{rebuilt ? 205 : 188}</td>
                  <td>0 (secondary)</td>
                  <td>1</td>
                  <td>5</td>
                  <td>42</td>
                </tr>
              ) : null}
            </>
          ) : (
            <tr>
              <td colSpan={6}>no user tables yet</td>
            </tr>
          )}
        </tbody>
      </table>
    </>
  );
}

function SqliteCatalog({ step, withoutRowid }: { step: number; withoutRowid: boolean }) {
  const s = schemaAt('sqlite', step);
  const cols = `id bigint PRIMARY KEY, customer_id int NOT NULL, code varchar(32) NOT NULL, total numeric(10,2), notes text${
    s.status ? ', status text' : ''
  }`;
  const rows = [
    {
      type: 'table',
      name: 'orders',
      tbl: 'orders',
      root: 2,
      sql: `CREATE TABLE orders(${cols})${withoutRowid ? ' WITHOUT ROWID' : ''}`,
    },
    ...(withoutRowid
      ? []
      : [
          {
            type: 'index',
            name: 'sqlite_autoindex_orders_1',
            tbl: 'orders',
            root: 3,
            sql: '(null — implied by PRIMARY KEY)',
          },
        ]),
    ...(step >= 3
      ? [
          {
            type: 'index',
            name: 'orders_customer_idx',
            tbl: 'orders',
            root: withoutRowid ? 3 : 4,
            sql: 'CREATE INDEX orders_customer_idx ON orders (customer_id)',
          },
        ]
      : []),
  ];
  const cookie = [1, 3, 4, 9].filter((d) => step >= d).length;
  return (
    <table className="viz-table">
      <caption style={{ textAlign: 'left' }}>
        sqlite_schema — the whole catalog is five columns. PRAGMA schema_version = {step >= 1 ? cookie : 0}
      </caption>
      <thead>
        <tr>
          <th>type</th>
          <th>name</th>
          <th>tbl_name</th>
          <th>rootpage</th>
          <th>sql</th>
        </tr>
      </thead>
      <tbody>
        {step >= 1 ? (
          rows.map((r) => (
            <tr key={r.name}>
              <td>{r.type}</td>
              <td>{r.name}</td>
              <td>{r.tbl}</td>
              <td>{r.root}</td>
              <td>{r.sql}</td>
            </tr>
          ))
        ) : (
          <tr>
            <td colSpan={5}>empty database</td>
          </tr>
        )}
      </tbody>
    </table>
  );
}

/* -------------------------------------------------------------------- lab */

export default function SchemaToFilesMapper() {
  const [step, setStep] = useState(3);
  const [rowsExp, setRowsExp] = useState(6.3);
  const [focus, setFocus] = useState<EngineId>('pg');
  const [wide, setWide] = useState(false);
  const [withoutRowid, setWithoutRowid] = useState(false);
  const [tblspc, setTblspc] = useState(false);
  const [ref, width] = useSize(940);

  const rows = Math.round(10 ** rowsExp);
  const states = useMemo(
    () => [
      pgState(step, rows, wide, tblspc),
      ibState(step, rows, wide, tblspc),
      liteState(step, rows, wide, withoutRowid),
    ],
    [step, rows, wide, tblspc, withoutRowid],
  );
  const me = states.find((s) => s.engine === focus)!;
  const stmt = SCRIPT[step];

  return (
    <VizPanel
      title="One DDL script, three engines, and the files it actually makes"
      subtitle="Scrub through the script and watch each statement get classified: a catalog row edited in place, or every row re-materialised into new storage. Sizes are computed from each engine's documented record layout."
      controls={
        <>
          <Slider
            label="DDL step"
            min={0}
            max={LAST}
            value={step}
            onChange={setStep}
            format={(n) => `${n} / ${LAST}`}
          />
          <Slider
            label="Rows loaded"
            min={4}
            max={7.3}
            step={0.1}
            value={rowsExp}
            onChange={setRowsExp}
            format={() => fmtNum(rows)}
          />
          <Segmented
            label="Catalog"
            value={focus}
            options={[
              { value: 'pg', label: 'PostgreSQL' },
              { value: 'innodb', label: 'InnoDB' },
              { value: 'sqlite', label: 'SQLite' },
            ]}
            onChange={setFocus}
          />
          <Check label="notes are 8 KB documents" checked={wide} onChange={setWide} />
          <Check label="SQLite: WITHOUT ROWID" checked={withoutRowid} onChange={setWithoutRowid} />
          <Check label="PG/MySQL: own tablespace" checked={tblspc} onChange={setTblspc} />
        </>
      }
      legend={
        <Legend
          items={[
            { label: 'Table data (heap / clustered tree / table b-tree)', color: KIND_COLOR.data },
            { label: 'Index', color: KIND_COLOR.index },
            { label: 'Out-of-line values (TOAST / LOB / overflow)', color: KIND_COLOR.toast },
            { label: 'Space bookkeeping (FSM, VM, FSP, INODE)', color: KIND_COLOR.map },
            { label: 'Shared: catalog, undo, log', color: KIND_COLOR.shared },
            { label: 'Old/temporary copy, dropped at commit', color: KIND_COLOR.orphan },
          ]}
        />
      }
      stats={
        <Stats
          items={[
            { label: 'PostgreSQL on disk', value: states[0].total ? fmtBytes(states[0].total) : '—' },
            { label: 'InnoDB .ibd', value: states[1].total ? fmtBytes(states[1].total) : '—' },
            { label: 'SQLite .db', value: states[2].total ? fmtBytes(states[2].total) : '—' },
            {
              label: `Bytes written by S${step} (${focus})`,
              value: me.written ? fmtBytes(me.written) : 'none',
              hint: 'A metadata-only DDL writes one catalog row and a WAL record; a rewrite writes the whole relation again.',
            },
            {
              label: `Peak disk during S${step} (${focus})`,
              value: me.peak ? fmtBytes(me.peak) : '—',
              hint: 'During a rewrite the old and new copies both exist.',
            },
            {
              label: 'Row width: catalog vs disk',
              value: `${me.logicalWidth} B / ${me.physicalWidth} B`,
              hint: 'The gap is exactly what attmissingval, INSTANT_COLS and short SQLite records paper over: the catalog says more columns than the stored record contains.',
            },
          ]}
        />
      }
      note={
        <>
          <strong>
            S{step}: {stmt.sql.join(' ')}
          </strong>{' '}
          {stmt.alt ? <em>({stmt.alt})</em> : null} — <strong>{me.label}:</strong> {me.note}
        </>
      }
      table={
        <>
          <table className="viz-table">
            <caption style={{ textAlign: 'left' }}>
              Every object at step {step}, {fmtNum(rows)} rows loaded
            </caption>
            <thead>
              <tr>
                <th>Engine</th>
                <th>Path</th>
                <th>Object</th>
                <th>Role</th>
                <th>Size</th>
              </tr>
            </thead>
            <tbody>
              {states.flatMap((s) =>
                s.files.flatMap((g) =>
                  g.objs.map((o) => (
                    <tr key={`${s.engine}-${g.key}-${o.key}`}>
                      <td>{s.label}</td>
                      <td>{g.path}</td>
                      <td>{o.name}</td>
                      <td>{KIND_LABEL[o.kind]}</td>
                      <td>{o.bytes === null ? 'not modelled' : fmtBytes(o.bytes)}</td>
                    </tr>
                  )),
                ),
              )}
            </tbody>
          </table>
          <table className="viz-table">
            <caption style={{ textAlign: 'left' }}>
              How each statement is classified, engine by engine
            </caption>
            <thead>
              <tr>
                <th>#</th>
                <th>Statement</th>
                <th>PostgreSQL</th>
                <th>InnoDB</th>
                <th>SQLite</th>
              </tr>
            </thead>
            <tbody>
              {SCRIPT.map((st, i) => {
                if (i === 0) return null;
                const row = [
                  pgState(i, rows, wide, tblspc),
                  ibState(i, rows, wide, tblspc),
                  liteState(i, rows, wide, withoutRowid),
                ];
                return (
                  <tr key={i}>
                    <td>S{i}</td>
                    <td>{st.sql.join(' ')}</td>
                    {row.map((r) => (
                      <td key={r.engine}>{VERDICT_LABEL[r.verdict]}</td>
                    ))}
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
          <Columns states={states} width={width} focus={focus} step={step} />
        </TooltipHost>
        <p className="viz-sub" style={{ marginTop: '0.75rem' }}>
          {me.label} catalog after S{step} — the rows an engine consults to find the bytes.
        </p>
        {focus === 'pg' ? <PgCatalog step={step} tblspc={tblspc} /> : null}
        {focus === 'innodb' ? <InnodbCatalog step={step} /> : null}
        {focus === 'sqlite' ? <SqliteCatalog step={step} withoutRowid={withoutRowid} /> : null}
      </div>
    </VizPanel>
  );
}
