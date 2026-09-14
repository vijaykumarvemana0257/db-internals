import { useState } from 'react';
import {
  VizPanel,
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
  useSize,
} from './Viz';

/**
 * The catalog as ordinary tables, the bootstrap that finds them, and the invalidation
 * queue that keeps two backends honest about them.
 *
 * Modelled on PostgreSQL's actual machinery: pg_class / pg_attribute / pg_index /
 * pg_statistic heap tuples with MVCC row versions, pg_filenode.map for mapped relations,
 * pg_internal.init for the relcache init file, and the shared invalidation (sinval) ring
 * that a DDL commit writes into and that every other backend drains at its next lock
 * acquisition.
 */

/* ------------------------------------------------------------------- schema */

type Col = { attnum: number; name: string; typ: string; dropped: boolean };
type Idx = { oid: number; name: string; ready: boolean; valid: boolean };

type Schema = {
  natts: number; // pg_class.relnatts — never decremented; attnums are never reused
  cols: Col[];
  idx: Idx[];
  relpages: number;
  reltuples: number; // -1 since PostgreSQL 14 means "never analyzed"
  hasStats: boolean;
};

const TAB_OID = 16401;
const DB_OID = 16384;
const IDX_OID = 16408;
const RING = 4096; // sinvaladt.c: MAXNUMMESSAGES

const SCHEMA0: Schema = {
  natts: 4,
  cols: [
    { attnum: 1, name: 'id', typ: 'int8', dropped: false },
    { attnum: 2, name: 'cust_id', typ: 'int4', dropped: false },
    { attnum: 3, name: 'total', typ: 'numeric', dropped: false },
    { attnum: 4, name: 'note', typ: 'text', dropped: false },
  ],
  idx: [],
  relpages: 0,
  reltuples: -1,
  hasStats: false,
};

const ADD_NAMES = ['ship_by', 'priority', 'region_id', 'promo_code'];

/** Deterministic relfilenodes for the rewrites VACUUM FULL performs. */
const NEW_FILENODES = (() => {
  const rng = makeRng(1259); // pg_class's OID
  return Array.from({ length: 8 }, () => 16500 + Math.floor(rng() * 900));
})();

/* -------------------------------------------------------------- catalog rows */

type Cat = 'pg_class' | 'pg_attribute' | 'pg_index' | 'pg_statistic';

type NewRow = { key: string; cat: Cat; cells: string; detail: string; inplace?: boolean };

type Row = NewRow & { id: number; xmin: number; committed: boolean; dead: boolean };

/* ------------------------------------------------------------ sinval messages */

type MsgKind = 'relcache' | 'catcache' | 'relmap' | 'smgr';
type MsgBody = { kind: MsgKind; target: string; hint: string };
type Msg = MsgBody & { seq: number };

/* ---------------------------------------------------------- the relation map */

type MapEntry = { rel: string; oid: number; filenode: number; scope: 'shared' | 'db' };

const MAP0: MapEntry[] = [
  { rel: 'pg_database', oid: 1262, filenode: 1262, scope: 'shared' },
  { rel: 'pg_authid', oid: 1260, filenode: 1260, scope: 'shared' },
  { rel: 'pg_class', oid: 1259, filenode: 1259, scope: 'db' },
  { rel: 'pg_attribute', oid: 1249, filenode: 1249, scope: 'db' },
  { rel: 'pg_proc', oid: 1255, filenode: 1255, scope: 'db' },
  { rel: 'pg_type', oid: 1247, filenode: 1247, scope: 'db' },
];

/* -------------------------------------------------------------------- state */

type LogRow = {
  n: number;
  actor: string;
  action: string;
  wrote: number;
  queued: number;
  unread: number;
  bcols: string;
};

type S = {
  rows: Row[];
  nextRowId: number;
  map: MapEntry[];
  initFile: boolean; // base/<db>/pg_internal.init exists and is valid
  committed: Schema; // what a fresh catalog read returns
  working: Schema; // backend A's uncommitted view
  xid: number;
  aOpen: boolean;
  pending: MsgBody[]; // A's private invalidation queue, broadcast at commit
  queue: Msg[];
  seq: number;
  bPtr: number; // how far B has drained the ring
  bEntry: Schema | null; // B's relcache entry for orders
  bValid: boolean;
  bReset: boolean;
  bMapStale: boolean;
  booting: boolean;
  boot: number; // bootstrap phases completed
  bootReads: number;
  stepIx: number;
  vacCount: number;
  head: string;
  body: string;
  log: LogRow[];
};

const S0: S = {
  rows: [],
  nextRowId: 1,
  map: MAP0,
  initFile: true,
  committed: SCHEMA0,
  working: SCHEMA0,
  xid: 4711,
  aOpen: false,
  pending: [],
  queue: [],
  seq: 0,
  bPtr: 0,
  bEntry: SCHEMA0,
  bValid: true,
  bReset: false,
  bMapStale: false,
  booting: false,
  boot: 0,
  bootReads: 0,
  stepIx: 0,
  vacCount: 0,
  head: 'Two backends, one set of catalog tables.',
  body:
    'Backend B already holds a relcache entry for orders, built the ordinary way: index scans on ' +
    'pg_class and pg_attribute, cached in its own private memory. Run DDL in backend A and watch ' +
    'which catalog row versions appear, when B stops being right, and what makes it notice.',
  log: [],
};

/* ----------------------------------------------------------------- the script */

type DdlId = 'addcol' | 'dropcol' | 'createidx' | 'cic' | 'analyze' | 'vacfull';

type StepOut = {
  w?: Schema;
  rows?: NewRow[];
  kill?: (r: Row) => boolean;
  hard?: boolean; // an in-place update replaces the row rather than tombstoning it
  msgs?: MsgBody[];
  map?: { oid: number; filenode: number };
  dropDead?: boolean;
  head: string;
  body: string;
};

type Step = {
  label: string;
  sql: string;
  commit: boolean;
  waitsForTxns?: boolean; // CIC's commits wait for older snapshots rather than taking a blocking lock
  run: (s: S) => StepOut;
};

/**
 * The lock each scripted statement actually takes, and whether it conflicts with the
 * AccessShareLock backend B holds on orders for the length of its transaction.
 * CREATE INDEX takes ShareLock and CIC/ANALYZE take ShareUpdateExclusiveLock — none of
 * those conflict with AccessShareLock, so a reader does not block them.
 */
const DDL_LOCK: Record<DdlId, { mode: string; target: string; conflicts: boolean }> = {
  addcol: { mode: 'AccessExclusiveLock', target: 'orders', conflicts: true },
  dropcol: { mode: 'AccessExclusiveLock', target: 'orders', conflicts: true },
  createidx: { mode: 'ShareLock', target: 'orders', conflicts: false },
  cic: { mode: 'ShareUpdateExclusiveLock', target: 'orders', conflicts: false },
  analyze: { mode: 'ShareUpdateExclusiveLock', target: 'orders', conflicts: false },
  vacfull: { mode: 'AccessExclusiveLock', target: 'pg_class', conflicts: false },
};

const relMsg = (hint: string): MsgBody => ({ kind: 'relcache', target: `orders (${TAB_OID})`, hint });

const DDL: Record<DdlId, { label: string; steps: Step[] }> = {
  addcol: {
    label: 'ALTER TABLE … ADD COLUMN',
    steps: [
      {
        label: 'execute',
        sql: 'ALTER TABLE orders ADD COLUMN … date;',
        commit: false,
        run: (s) => {
          const added = s.working.cols.filter((c) => c.attnum > 4).length;
          const name = ADD_NAMES[added % ADD_NAMES.length];
          const attnum = s.working.natts + 1;
          return {
            w: {
              ...s.working,
              natts: attnum,
              cols: [...s.working.cols, { attnum, name, typ: 'date', dropped: false }],
            },
            rows: [
              {
                key: `attr:${attnum}`,
                cat: 'pg_attribute',
                cells: `attrelid=${TAB_OID} attname=${name} attnum=${attnum} atttypid=date`,
                detail:
                  'attlen=4, attalign=i, attnotnull=f, atthasmissing=f. A plain ADD COLUMN with no ' +
                  'default writes one pg_attribute tuple and touches no heap page of orders at all.',
              },
              {
                key: 'class:orders',
                cat: 'pg_class',
                cells: `oid=${TAB_OID} relname=orders relnatts=${attnum}`,
                detail:
                  'A new version of the pg_class row. The old version stays on the page as a dead ' +
                  'tuple until vacuum reclaims it — the catalogs bloat exactly like user tables.',
              },
            ],
            kill: (r) => r.key === 'class:orders',
            msgs: [
              relMsg('RelationCacheInvalidate: every backend must rebuild the TupleDesc for orders.'),
              {
                kind: 'catcache',
                target: 'ATTNUM, ATTNAME, RELOID',
                hint: 'One catcache message per modified catalog tuple, carrying its hash value and TID.',
              },
            ],
            head: `The column exists — in one uncommitted pg_attribute tuple.`,
            body:
              'ADD COLUMN is two catalog writes and nothing else: a pg_attribute row for the new ' +
              'attribute and a new version of the pg_class row with relnatts bumped. Both carry ' +
              "backend A's xid in xmin, so no other snapshot can see them. A itself can, because " +
              'CommandCounterIncrement advances its command counter between subcommands. The ' +
              'invalidation messages are sitting in A’s private queue, not yet in shared memory.',
          };
        },
      },
      {
        label: 'COMMIT',
        sql: 'COMMIT;',
        commit: true,
        run: () => ({
          head: 'Commit: the rows become visible and the messages go into shared memory.',
          body:
            'At commit A appends its queued invalidation messages to the shared sinval ring and ' +
            `unlinks base/${DB_OID}/pg_internal.init so no new backend loads a stale relcache init ` +
            'file. It does not wait for anybody to read those messages. Backend B is now holding a ' +
            'relcache entry that describes a table which no longer has that shape.',
        }),
      },
    ],
  },

  dropcol: {
    label: 'ALTER TABLE … DROP COLUMN',
    steps: [
      {
        label: 'execute',
        sql: 'ALTER TABLE orders DROP COLUMN …;',
        commit: false,
        run: (s) => {
          const target = s.working.cols.find((c) => !c.dropped && c.attnum > 1);
          if (!target) {
            return { head: 'Nothing left to drop.', body: 'Reset to start again with four columns.' };
          }
          const n = target.attnum;
          return {
            w: {
              ...s.working,
              cols: s.working.cols.map((c) => (c.attnum === n ? { ...c, dropped: true } : c)),
            },
            rows: [
              {
                key: `attr:${n}`,
                cat: 'pg_attribute',
                cells: `attnum=${n} attisdropped=t attname=........pg.dropped.${n}........`,
                detail:
                  'attlen and attalign are deliberately preserved: every existing heap tuple still ' +
                  'contains those bytes, and the deformer has to skip exactly the right width to find ' +
                  'the attributes after it. The name is rewritten only so a later column can reuse it.',
              },
            ],
            kill: (r) => r.key === `attr:${n}`,
            msgs: [
              relMsg('The TupleDesc changes shape: one fewer user-visible attribute.'),
              {
                kind: 'catcache',
                target: 'ATTNUM, ATTNAME',
                hint: `The ATTNAME entry for (${TAB_OID}, "${target.name}") must die, or the parser would still resolve the name.`,
              },
            ],
            head: 'DROP COLUMN deletes no data and removes no catalog row.',
            body:
              `The pg_attribute tuple for ${target.name} is updated — attisdropped = true, name rewritten ` +
              `to ........pg.dropped.${n}........ — and relnatts on pg_class is not decremented, because ` +
              'attnums are never reused. The column’s bytes stay in every heap tuple until the next ' +
              'rewrite. The catalog is the only thing that changed, which is why the statement returns ' +
              'instantly on a terabyte table.',
          };
        },
      },
      {
        label: 'COMMIT',
        sql: 'COMMIT;',
        commit: true,
        run: () => ({
          head: 'Committed. Every other backend is now one message behind the truth.',
          body:
            'Until B processes that message it will happily plan a query against a cached TupleDesc ' +
            'that still contains the dropped column. What stops that being a correctness bug is lock ' +
            'ordering, not timing — tick the checkbox and try to commit again.',
        }),
      },
    ],
  },

  createidx: {
    label: 'CREATE INDEX',
    steps: [
      {
        label: 'execute',
        sql: 'CREATE INDEX orders_cust_idx ON orders (cust_id);',
        commit: false,
        run: (s) => {
          const n = s.working.idx.length;
          const oid = IDX_OID + n;
          const name = n === 0 ? 'orders_cust_idx' : `orders_idx_${n + 1}`;
          return {
            w: { ...s.working, idx: [...s.working.idx, { oid, name, ready: true, valid: true }] },
            rows: [
              {
                key: `class:i${oid}`,
                cat: 'pg_class',
                cells: `oid=${oid} relname=${name} relkind=i relam=btree relfilenode=${oid}`,
                detail:
                  'An index is a relation: its own pg_class row, its own file on disk, its own ' +
                  'relpages and reltuples.',
              },
              {
                key: `attr:i${oid}`,
                cat: 'pg_attribute',
                cells: `attrelid=${oid} attname=cust_id attnum=1 atttypid=int4`,
                detail: 'The index has its own attributes, describing the tuples in its leaf pages.',
              },
              {
                key: `index:${oid}`,
                cat: 'pg_index',
                cells: `indexrelid=${oid} indrelid=${TAB_OID} indkey=2 indisready=t indisvalid=t`,
                detail:
                  'indisready = writers maintain it; indisvalid = the planner may use it. A plain ' +
                  'CREATE INDEX sets both in one transaction, because it holds a SHARE lock and no ' +
                  'concurrent writer can exist.',
              },
              {
                key: 'class:orders',
                cat: 'pg_class',
                cells: `oid=${TAB_OID} relname=orders relhasindex=t`,
                detail:
                  'A hint, not the truth: relhasindex only tells the planner it is worth scanning ' +
                  'pg_index for this relation.',
              },
            ],
            kill: (r) => r.key === 'class:orders',
            msgs: [
              relMsg('The relcache entry caches the list of index OIDs; that list just changed.'),
              { kind: 'catcache', target: 'INDEXRELID, RELOID', hint: 'Planner lookups of the new pg_index row.' },
              { kind: 'smgr', target: `relfilenode ${oid}`, hint: 'Backends must forget cached file descriptors for this relfilenode.' },
            ],
            head: 'Three catalogs, four rows, one new file.',
            body:
              'The index exists as a pg_class row with relkind = i, its own pg_attribute rows, and a ' +
              'pg_index row holding the column list and the two boolean flags the planner cares ' +
              'about. Nothing about the heap changed except a relhasindex hint.',
          };
        },
      },
      {
        label: 'COMMIT',
        sql: 'COMMIT;',
        commit: true,
        run: () => ({
          head: 'Committed — and B will still plan a sequential scan.',
          body:
            'B’s relcache entry carries a cached list of index OIDs. Until it processes the ' +
            'invalidation it does not know the index exists, and the planner cannot consider what ' +
            'the relcache does not report.',
        }),
      },
    ],
  },

  cic: {
    label: 'CREATE INDEX CONCURRENTLY (3 transactions)',
    steps: [
      {
        label: 'txn 1: publish an empty index',
        sql: 'CREATE INDEX CONCURRENTLY orders_cust_idx ON orders (cust_id);',
        commit: false,
        run: (s) => {
          const n = s.working.idx.length;
          const oid = IDX_OID + n;
          const name = n === 0 ? 'orders_cust_idx' : `orders_idx_${n + 1}`;
          return {
            w: { ...s.working, idx: [...s.working.idx, { oid, name, ready: false, valid: false }] },
            rows: [
              {
                key: `class:i${oid}`,
                cat: 'pg_class',
                cells: `oid=${oid} relname=${name} relkind=i relfilenode=${oid}`,
                detail: 'Created empty, so the next transaction has something to start maintaining.',
              },
              {
                key: `index:${oid}`,
                cat: 'pg_index',
                cells: `indexrelid=${oid} indrelid=${TAB_OID} indisready=f indisvalid=f`,
                detail:
                  'The index exists and is empty. Writers ignore it (not ready) and the planner ' +
                  'ignores it (not valid). This is the state a cancelled CIC leaves behind forever, ' +
                  'and what psql’s \\d prints as INVALID.',
              },
            ],
            msgs: [relMsg('Everyone must learn the index exists before anyone starts filling it.')],
            head: 'CIC transaction 1: announce the index before building it.',
            body:
              'The point of CONCURRENTLY is never holding a lock that blocks writers, which forces ' +
              'the work into separate transactions with separate commits. First it publishes an ' +
              'empty pg_index row with both flags false.',
          };
        },
      },
      {
        label: 'COMMIT (then wait for old txns)',
        sql: 'COMMIT;',
        commit: true,
        waitsForTxns: true,
        run: () => ({
          head: 'Commit 1, then wait.',
          body:
            'CIC now waits for every transaction that started before this commit to finish: those ' +
            'transactions have not processed the invalidation and could still be inserting rows the ' +
            'index would never see. The queue is the reason that wait is bounded rather than ' +
            'indefinite.',
        }),
      },
      {
        label: 'txn 2: indisready = true',
        sql: 'UPDATE pg_index SET indisready = true …',
        commit: false,
        run: (s) => {
          const last = s.working.idx[s.working.idx.length - 1];
          if (!last) return { head: 'Nothing to flip.', body: 'Run transaction 1 first.' };
          return {
            w: {
              ...s.working,
              idx: s.working.idx.map((i) => (i.oid === last.oid ? { ...i, ready: true } : i)),
            },
            rows: [
              {
                key: `index:${last.oid}`,
                cat: 'pg_index',
                cells: `indexrelid=${last.oid} indisready=t indisvalid=f`,
                detail:
                  'Writers now insert into the index; readers still ignore it. Only after this commit ' +
                  'can the build scan take a snapshot that is guaranteed to miss nothing.',
              },
            ],
            kill: (r) => r.key === `index:${last.oid}`,
            msgs: [relMsg('Writers must start maintaining the new index from this point on.')],
            head: 'CIC transaction 2: writers start maintaining it.',
            body:
              'Flipping indisready is a one-column update of a pg_index row — a new heap tuple in a ' +
              'catalog table, nothing more. The heap scan that actually builds the index runs after ' +
              'this commit, under its own snapshot.',
          };
        },
      },
      {
        label: 'COMMIT (build + validate pass)',
        sql: 'COMMIT;',
        commit: true,
        run: () => ({
          head: 'Commit 2. Two heap passes and another wait.',
          body:
            'The first pass builds the index from one snapshot; the second pass adds everything ' +
            'inserted since. Only then can the last transaction mark it usable.',
        }),
      },
      {
        label: 'txn 3: indisvalid = true',
        sql: 'UPDATE pg_index SET indisvalid = true …',
        commit: false,
        run: (s) => {
          const last = s.working.idx[s.working.idx.length - 1];
          if (!last) return { head: 'Nothing to flip.', body: 'Run transaction 1 first.' };
          return {
            w: {
              ...s.working,
              idx: s.working.idx.map((i) => (i.oid === last.oid ? { ...i, valid: true } : i)),
            },
            rows: [
              {
                key: `index:${last.oid}`,
                cat: 'pg_index',
                cells: `indexrelid=${last.oid} indisready=t indisvalid=t`,
                detail: 'Only now will the planner consider this index for a scan.',
              },
            ],
            kill: (r) => r.key === `index:${last.oid}`,
            msgs: [relMsg('The planner may now use the index — cached plans must be discarded.')],
            head: 'CIC transaction 3: the planner may use it.',
            body:
              'Three commits, three broadcasts, three windows in which another backend is briefly out ' +
              'of date — and every window is safe because a backend that has not seen the message has ' +
              'also not seen the index, and correctness only requires that writers learn about it ' +
              'before readers do.',
          };
        },
      },
      {
        label: 'COMMIT',
        sql: 'COMMIT;',
        commit: true,
        run: () => ({
          head: 'Done. Compare this row list with the plain CREATE INDEX script.',
          body:
            'The same final rows, three times the commits, and a pg_index row that passed through two ' +
            'intermediate states any other backend could have observed.',
        }),
      },
    ],
  },

  analyze: {
    label: 'ANALYZE (writes pg_statistic)',
    steps: [
      {
        label: 'execute',
        sql: 'ANALYZE orders;',
        commit: false,
        run: (s) => ({
          w: { ...s.working, relpages: 812, reltuples: 120000, hasStats: true },
          rows: [
            {
              key: 'class:orders',
              cat: 'pg_class',
              cells: `oid=${TAB_OID} relpages=812 reltuples=120000 [inplace]`,
              detail:
                'vac_update_relstats overwrites the existing tuple in place instead of creating a ' +
                'version. It is therefore not transactional: it survives a ROLLBACK and creates no ' +
                'dead tuple — what you want for a statistic nobody reads transactionally.',
              inplace: true,
            },
            {
              key: 'stat:2',
              cat: 'pg_statistic',
              cells: `starelid=${TAB_OID} staattnum=2 stakind1=1 (MCV) stanumbers1={…}`,
              detail:
                'The most-common-values slot for cust_id. pg_statistic gives every column five ' +
                'generic slots; the planner reads them through the STATRELATTINH syscache.',
            },
            {
              key: 'stat:3',
              cat: 'pg_statistic',
              cells: `starelid=${TAB_OID} staattnum=3 stakind2=2 (histogram) stavalues2={…}`,
              detail:
                'Histogram bounds for total, from default_statistics_target = 100 buckets. Readable ' +
                'only through the pg_stats view unless you have SELECT on the column: the raw values ' +
                'are sampled user data.',
            },
          ],
          kill: (r) => r.key === 'class:orders',
          hard: true,
          msgs: [relMsg('The cached relpages/reltuples estimate on the relcache entry is stale.')],
          head: 'ANALYZE is a catalog write, and one of its writes skips MVCC.',
          body:
            'The sampled statistics go into pg_statistic as ordinary MVCC tuples. relpages and ' +
            'reltuples go into the existing pg_class tuple in place, bypassing MVCC entirely — no new ' +
            'version, no bloat, no rollback. This row is where cost-based optimization begins: every ' +
            'selectivity estimate the planner makes is a read of these bytes through the syscache.',
        }),
      },
      {
        label: 'COMMIT',
        sql: 'COMMIT;',
        commit: true,
        run: () => ({
          head: 'Committed. B still costs the query with reltuples = -1 until it catches up.',
          body:
            'A never-analyzed table reports reltuples = -1, the "unknown" marker since PostgreSQL 14, ' +
            'and the planner falls back to a guess derived from the physical file size. Read in B ' +
            'before and after the invalidation and watch the estimated row count move.',
        }),
      },
    ],
  },

  vacfull: {
    label: 'VACUUM FULL pg_class (rewrites a mapped catalog)',
    steps: [
      {
        label: 'execute',
        sql: 'VACUUM FULL pg_class;',
        commit: false,
        run: (s) => {
          const fn = NEW_FILENODES[s.vacCount % NEW_FILENODES.length];
          return {
            rows: [
              {
                key: 'class:pg_class',
                cat: 'pg_class',
                cells: `oid=1259 pg_class rewritten into relfilenode ${fn} (relfilenode column = 0)`,
                detail:
                  'The new filenode cannot be recorded in pg_class.relfilenode: the row describing ' +
                  'pg_class lives in the file that is moving, and a backend must find that file ' +
                  'before it can read any row at all. So the column is 0 and the real number lives ' +
                  'in pg_filenode.map.',
              },
            ],
            map: { oid: 1259, filenode: fn },
            dropDead: true,
            msgs: [
              {
                kind: 'relmap',
                target: `base/${DB_OID}/pg_filenode.map`,
                hint: 'Every backend must re-read the map file; its cached filenode for pg_class is wrong.',
              },
              { kind: 'smgr', target: `relfilenode ${fn}`, hint: 'Cached file descriptors for the old filenode must be closed.' },
            ],
            head: `pg_class now lives in file ${fn}, and no catalog row says so.`,
            body:
              'This is the case that forces pg_filenode.map to exist. A rewrite of a mapped relation ' +
              'updates a 512-byte CRC-protected file, WAL-logs it as XLOG_RELMAP_UPDATE, and ' +
              'broadcasts a relmap invalidation. The nailed and shared catalogs are exactly the ' +
              'relations whose location cannot be stored in a catalog.',
          };
        },
      },
      {
        label: 'COMMIT',
        sql: 'COMMIT;',
        commit: true,
        run: () => ({
          head: 'Committed. The dead versions are gone and the map entry moved.',
          body:
            'VACUUM FULL rewrote the table into a new file and discarded every dead version — the ' +
            'standard cure for a pg_attribute that has grown to gigabytes under temp-table churn, ' +
            'and the reason it takes an AccessExclusiveLock on the catalog itself and blocks every ' +
            'other backend that needs to look anything up.',
        }),
      },
    ],
  },
};

/* ----------------------------------------------------------------- bootstrap */

const BOOT = [
  {
    name: 'RelationMapInitialize',
    file: 'global/pg_filenode.map',
    reads: 1,
    body:
      'Before any catalog can be read, the backend reads a 512-byte flat file with a CRC. It is not ' +
      'a table, has no MVCC and needs no relcache entry — that is the entire point. It yields the ' +
      'relfilenode of the shared catalogs, pg_database (OID 1262) among them.',
  },
  {
    name: 'formrdesc(): nail four catalogs',
    file: '(compiled into the binary)',
    reads: 0,
    body:
      'relcache.c builds relcache entries for pg_class, pg_attribute, pg_proc and pg_type from ' +
      'hard-coded schemas generated at build time from the catalog headers. These four are "nailed": ' +
      'their descriptors exist before any tuple has been read, and they are never evicted.',
  },
  {
    name: 'Find the database',
    file: 'global/1262 — pg_database',
    reads: 2,
    body:
      'Now a real catalog scan is possible. Looking up "shop" in pg_database yields the database OID ' +
      `${DB_OID} and its tablespace, which is what names the directory base/${DB_OID}/ that ` +
      'everything else lives in.',
  },
  {
    name: 'Per-database relation map',
    file: `base/${DB_OID}/pg_filenode.map`,
    reads: 1,
    body:
      'The second map file gives the relfilenodes of the four nailed catalogs and their indexes in ' +
      'this database — the numbers a VACUUM FULL or REINDEX on a catalog would have changed.',
  },
  {
    name: 'Load the relcache init file',
    file: `base/${DB_OID}/pg_internal.init`,
    reads: 1,
    body:
      'A serialized dump of the relcache entries for the system catalogs and their indexes, read ' +
      'sequentially in one go instead of running dozens of index scans. Any DDL commit unlinks it; ' +
      'the first backend to find it missing rebuilds it and writes a fresh copy.',
  },
  {
    name: 'Open orders',
    file: `base/${DB_OID}/${TAB_OID}`,
    reads: 3,
    body:
      'Only now is an ordinary relcache lookup possible: SearchSysCache1(RELOID) for the pg_class ' +
      'row, a scan of pg_attribute for every attnum from 1 to relnatts, a scan of pg_index for the ' +
      'index list. The result is one TupleDesc in this backend’s CacheMemoryContext, and it stays ' +
      'there until something invalidates it.',
  },
];

/* ----------------------------------------------------------------- behaviour */

function liveCols(sch: Schema) {
  return sch.cols.filter((c) => !c.dropped);
}

function planOf(sch: Schema) {
  const idx = sch.idx.find((i) => i.valid);
  const rows = sch.reltuples < 0 ? 'guessed from file size' : `${fmtNum(sch.reltuples)} est.`;
  return `${idx ? `Index Scan using ${idx.name}` : 'Seq Scan on orders'} (rows=${rows})`;
}

function fingerprint(sch: Schema) {
  return JSON.stringify([
    liveCols(sch).map((c) => c.name),
    sch.idx.filter((i) => i.valid).map((i) => i.name),
    sch.reltuples,
  ]);
}

function tally(s: S) {
  return {
    live: s.rows.filter((r) => !r.dead).length,
    dead: s.rows.filter((r) => r.dead).length,
    unread: s.queue.length - s.bPtr,
  };
}

type Op = 'step' | 'readB' | 'acceptB' | 'flood' | 'cold' | 'bootStep';

function apply(s: S, op: Op, ddl: DdlId, bLocked: boolean): S {
  const next: S = { ...s };
  let head = s.head;
  let body = s.body;
  let actor = 'backend B';
  let action = '';
  let wrote = 0;
  let queued = 0;

  switch (op) {
    case 'step': {
      const script = DDL[ddl].steps;
      const step = script[s.stepIx % script.length];
      actor = 'backend A';
      action = step.sql;

      const lock = DDL_LOCK[ddl];
      const stalled =
        bLocked && lock.conflicts && s.stepIx === 0
          ? 'lock'
          : bLocked && step.waitsForTxns
            ? 'wait'
            : null;

      if (stalled) {
        return {
          ...next,
          head:
            stalled === 'lock'
              ? `A is blocked: it wants ${lock.mode} on ${lock.target} and B holds AccessShareLock.`
              : 'A cannot finish: CREATE INDEX CONCURRENTLY is waiting for B’s transaction.',
          body:
            stalled === 'lock'
              ? `${DDL[ddl].label} takes ${lock.mode} on ${lock.target} at the start of the statement, ` +
                'and that conflicts with the AccessShareLock B took when it first read the table and ' +
                'holds until its transaction ends. This is the argument that makes a stale relcache ' +
                'entry safe rather than merely unlikely: B cannot be using an out-of-date definition ' +
                'at a moment when the new one is committed, because the DDL cannot even acquire its ' +
                'lock while B holds the table. Untick the box to end B’s transaction and let A through.'
              : 'CREATE INDEX CONCURRENTLY deliberately takes only ShareUpdateExclusiveLock, so it ' +
                'never blocks a reader or a writer — instead it waits for every transaction older ' +
                'than this commit to finish, because those transactions have not processed the ' +
                'invalidation and could still be writing rows the index would miss. B’s open ' +
                'transaction is exactly such a transaction. Untick the box to let the build proceed.',
          log: [
            ...s.log,
            {
              n: s.log.length + 1,
              actor,
              action: stalled === 'lock' ? `${step.sql} — waiting on lock` : 'COMMIT — waiting for old transactions',
              wrote: 0,
              queued: 0,
              unread: tally(s).unread,
              bcols: s.bEntry ? String(liveCols(s.bEntry).length) : '—',
            },
          ],
        };
      }

      const out = step.run(s);
      head = out.head;
      body = out.body;

      let rows = s.rows;
      if (out.dropDead) rows = rows.filter((r) => !r.dead);
      if (out.kill) {
        const hit = (r: Row) => out.kill!(r) && !r.dead;
        rows = out.hard ? rows.filter((r) => !hit(r)) : rows.map((r) => (hit(r) ? { ...r, dead: true } : r));
      }
      if (out.rows) {
        let id = s.nextRowId;
        const added: Row[] = out.rows.map((r) => ({ ...r, id: id++, xmin: s.xid, committed: false, dead: false }));
        rows = [...rows, ...added];
        next.nextRowId = id;
        wrote = added.length;
      }
      next.rows = rows;
      if (out.w) next.working = out.w;
      if (out.map) {
        next.map = s.map.map((m) => (m.oid === out.map!.oid ? { ...m, filenode: out.map!.filenode } : m));
        next.vacCount = s.vacCount + 1;
      }

      const pend = [...s.pending, ...(out.msgs ?? [])];
      if (step.commit) {
        const broadcast: Msg[] = pend.map((m, i) => ({ ...m, seq: s.seq + i + 1 }));
        next.queue = [...s.queue, ...broadcast];
        next.seq = s.seq + broadcast.length;
        next.pending = [];
        queued = broadcast.length;
        if (broadcast.some((m) => m.kind === 'relmap')) next.bMapStale = true;
        next.committed = next.working;
        next.rows = next.rows.map((r) => (!r.committed && r.xmin === s.xid ? { ...r, committed: true } : r));
        next.xid = s.xid + 1;
        next.aOpen = false;
        next.initFile = false;
      } else {
        next.pending = pend;
        next.aOpen = true;
      }

      next.stepIx = (s.stepIx + 1) % script.length;
      break;
    }

    case 'acceptB': {
      action = 'AcceptInvalidationMessages()';
      const unread = s.queue.length - s.bPtr;
      if (s.bReset) {
        next.bEntry = null;
        next.bValid = false;
        next.bReset = false;
        next.bPtr = s.queue.length;
        next.bMapStale = false;
        head = 'Cache reset: B threw away every catalog cache entry it had.';
        body =
          'B had fallen so far behind the ring that unread messages were about to be overwritten, so ' +
          'the sinval machinery set its reset flag. There is no way to know which entries the lost ' +
          'messages referred to, so InvalidateSystemCaches() discards all of them — the relcache, ' +
          'every catcache, the catalog snapshot. Correct, and expensive: the next few hundred queries ' +
          'in this backend all pay full catalog lookups.';
        break;
      }
      if (unread === 0) {
        head = 'Nothing to accept.';
        body =
          'B drained the ring and found no new messages. This call happens constantly — at the start ' +
          'of every command, and inside every LockRelationOid — and when nothing is queued it is an ' +
          'atomic read of a shared counter.';
        break;
      }
      const msgs = s.queue.slice(s.bPtr);
      next.bPtr = s.queue.length;
      if (msgs.some((m) => m.kind === 'relcache' || m.kind === 'catcache')) next.bValid = false;
      if (msgs.some((m) => m.kind === 'relmap')) next.bMapStale = false;
      head = `B consumed ${fmtNum(unread)} message${unread === 1 ? '' : 's'} and flagged its entry invalid.`;
      body =
        'This is what LockRelationOid does the moment it holds the lock — and it repeats the drain if ' +
        'the lock had to wait, because the definition can have changed while it slept. Note what it ' +
        'did not do: it did not rebuild anything. The entry is flagged and rebuilt lazily at the next ' +
        'RelationIdGetRelation, so a backend that never touches orders again never pays for the DDL.';
      break;
    }

    case 'readB': {
      action = 'SELECT * FROM orders;';
      if (s.booting) {
        head = 'B is still starting up.';
        body = 'Finish the bootstrap sequence first — there is no relcache to read from yet.';
        break;
      }
      if (s.aOpen && DDL_LOCK[ddl].conflicts) {
        head = `B is blocked: A holds ${DDL_LOCK[ddl].mode} on orders.`;
        body =
          'This is the other half of the argument. Between A taking its lock and A committing, no ' +
          'reader can touch the table at all — and when B finally is granted its AccessShareLock, ' +
          'LockRelationOid drains the invalidation queue before returning, precisely because the ' +
          'definition may have changed while it slept. Commit in A, then read.';
        break;
      }
      if (s.bReset) {
        head = 'B has a pending cache reset it has not processed.';
        body = 'Press AcceptInvalidationMessages to see what a reset costs.';
        break;
      }
      if (s.bMapStale) {
        head = 'B is about to open the wrong file for pg_class.';
        body =
          'Its cached relfilenode points at a file that VACUUM FULL replaced. A real backend cannot ' +
          'get here, because the relmap invalidation is processed before the file is opened — which ' +
          'is exactly why the mapping has to live in shared state and a flat file, and not only in a ' +
          'per-backend cache.';
        break;
      }
      if (!s.bEntry || !s.bValid) {
        const had = s.bEntry !== null;
        next.bEntry = s.committed;
        next.bValid = true;
        head = had ? 'Relcache miss: B rebuilt the entry.' : 'Cold relcache lookup for orders.';
        body =
          'RelationIdGetRelation found the entry missing or flagged invalid and rebuilt it: ' +
          'SearchSysCache1(RELOID) for the pg_class row, a systable scan of pg_attribute on ' +
          'pg_attribute_relid_attnum_index for every attnum up to relnatts, a scan of pg_index for ' +
          `the index list. B now sees ${liveCols(s.committed).length} live columns and plans ` +
          `${planOf(s.committed)}.`;
        break;
      }
      const stale = fingerprint(s.bEntry) !== fingerprint(s.committed);
      head = stale
        ? 'B answered from a stale relcache entry.'
        : 'B answered from cache — it touched no catalog page at all.';
      body = stale
        ? `B reports [${liveCols(s.bEntry).map((c) => c.name).join(', ')}] and plans ${planOf(s.bEntry)}. ` +
          `The committed catalogs say [${liveCols(s.committed).map((c) => c.name).join(', ')}] and ` +
          `${planOf(s.committed)}. Nothing in B consulted a catalog to produce that answer: a relcache ` +
          'hit is a hash lookup and a pointer dereference, which is precisely why it can be wrong. ' +
          'What bounds the error is that B must take a lock on orders before touching its data, and ' +
          'taking that lock drains the queue.'
        : `${liveCols(s.bEntry).map((c) => c.name).join(', ')} — and ${planOf(s.bEntry)}. A relcache hit ` +
          'costs a hash lookup in this backend’s own memory. That is the whole reason the cache ' +
          'exists: a trivial query would otherwise need a dozen index scans across pg_class, ' +
          'pg_attribute, pg_index, pg_type and pg_proc before it could even be parsed.';
      break;
    }

    case 'flood': {
      actor = 'other backends';
      action = `${fmtNum(RING + 200)} DDL statements`;
      const filler: Msg[] = Array.from({ length: RING + 200 }, (_, i) => ({
        seq: s.seq + i + 1,
        kind: 'catcache' as MsgKind,
        target: 'temp-table churn',
        hint: 'One of thousands of messages from CREATE TEMP TABLE / DROP TABLE in other sessions.',
      }));
      next.queue = [...s.queue, ...filler];
      next.seq = s.seq + filler.length;
      next.bReset = true;
      queued = filler.length;
      head = 'B fell off the back of the ring.';
      body =
        `The shared invalidation buffer holds MAXNUMMESSAGES = ${fmtNum(RING)} entries. A backend ` +
        'that lags gets a catchup signal telling it to drain; a backend that lags far enough for its ' +
        'unread messages to be overwritten gets its reset flag set instead. The ring never blocks ' +
        'the writer — the reader pays. Thousands of temporary tables a second behind a connection ' +
        'pool is the ordinary way to get here.';
      break;
    }

    case 'cold': {
      next.booting = true;
      next.boot = 0;
      next.bootReads = 0;
      next.bEntry = null;
      next.bValid = false;
      next.bPtr = s.queue.length;
      next.bReset = false;
      next.bMapStale = false;
      action = 'fork a new backend';
      head = 'A brand-new backend. It cannot read pg_class yet.';
      body =
        'It has no relcache, so it has no descriptor for pg_class; without a descriptor it cannot ' +
        'deform a pg_class tuple; and the row describing pg_class lives in pg_class. Step through the ' +
        'bootstrap and watch the circle get broken from outside the catalogs.';
      break;
    }

    case 'bootStep': {
      const i = s.boot;
      const p = BOOT[i];
      const slow = i === 4 && !s.initFile;
      next.boot = i + 1;
      next.bootReads = s.bootReads + (slow ? 9 : p.reads);
      action = p.name;
      head = `${i + 1}. ${p.name}`;
      body = slow
        ? 'pg_internal.init is gone — a DDL commit unlinked it. The backend falls back to building ' +
          'every system-catalog relcache entry the slow way, with index scans on pg_class and ' +
          'pg_attribute for each one, then writes a fresh init file on its way out. This is why the ' +
          'first connection after a schema migration is measurably slower than the next thousand.'
        : p.body;
      if (next.boot >= BOOT.length) {
        next.booting = false;
        next.bEntry = s.committed;
        next.bValid = true;
      }
      break;
    }
  }

  const t = tally(next);
  return {
    ...next,
    head,
    body,
    log: [
      ...s.log,
      {
        n: s.log.length + 1,
        actor,
        action: action || '—',
        wrote,
        queued,
        unread: t.unread,
        bcols: next.bEntry ? String(liveCols(next.bEntry).length) : '—',
      },
    ],
  };
}

/* --------------------------------------------------------------- the drawing */

const CAT_COLOR: Record<Cat, string> = {
  pg_class: 'var(--viz-1)',
  pg_attribute: 'var(--viz-2)',
  pg_index: 'var(--viz-7)',
  pg_statistic: 'var(--viz-6)',
};

const KIND_TAG: Record<MsgKind, string> = {
  relcache: 'rel',
  catcache: 'cat',
  relmap: 'map',
  smgr: 'smgr',
};

export default function CatalogBootstrapInvalidationLab() {
  const [ddl, setDdl] = useState<DdlId>('addcol');
  const [bLocked, setBLocked] = useState(false);
  const [s, setS] = useState<S>(S0);
  const [ref, width] = useSize(900);
  const tip = useTip();

  const run = (op: Op) => setS((cur) => apply(cur, op, ddl, bLocked));
  const t = tally(s);
  const stale = s.bEntry ? fingerprint(s.bEntry) !== fingerprint(s.committed) : false;
  const script = DDL[ddl].steps;
  const nextStep = script[s.stepIx % script.length];

  const svgW = Math.max(width, 880);
  const LW = Math.round(svgW * 0.48);
  const RX = LW + 24;
  const RW = svgW - RX - 4;

  const shown = s.rows.slice(-13);
  const hidden = s.rows.length - shown.length;
  const rowTop = 136;
  const leftH = rowTop + Math.max(2, shown.length) * 19 + 20;
  const height = Math.max(leftH, 466);

  const bCols = s.bEntry ? liveCols(s.bEntry) : [];
  const bIdx = s.bEntry ? s.bEntry.idx.filter((i) => i.valid) : [];
  const mapPgClass = s.map.find((m) => m.oid === 1259)!;

  const FILES = [
    { name: 'global/pg_filenode.map', note: '512 B + CRC — no catalog needed', phase: 1, bad: false },
    { name: `base/${DB_OID}/pg_filenode.map`, note: `pg_class → ${mapPgClass.filenode}`, phase: 4, bad: false },
    {
      name: `base/${DB_OID}/pg_internal.init`,
      note: s.initFile ? 'present — relcache preload' : 'unlinked by a DDL commit',
      phase: 5,
      bad: !s.initFile,
    },
  ];

  return (
    <VizPanel
      title="The catalog, the bootstrap and the invalidation queue"
      subtitle="Run DDL in backend A, watch the pg_class / pg_attribute / pg_index rows it writes, then read in backend B before and after the invalidation message reaches it."
      controls={
        <>
          <Choice
            label="Statement in backend A"
            value={ddl}
            onChange={(v) => {
              setDdl(v);
              setS((cur) => ({ ...cur, stepIx: 0 }));
            }}
            options={(Object.keys(DDL) as DdlId[]).map((k) => ({ value: k, label: DDL[k].label }))}
          />
          <Button onClick={() => run('step')} primary disabled={s.booting}>
            A: {nextStep.label}
          </Button>
          <Check label="B holds an open transaction on orders" checked={bLocked} onChange={setBLocked} />
          <Button onClick={() => run('readB')} disabled={s.booting}>
            B: SELECT * FROM orders
          </Button>
          <Button
            onClick={() => run('acceptB')}
            disabled={s.booting}
            title="What LockRelationOid() calls the moment it holds the lock"
          >
            B: AcceptInvalidationMessages()
          </Button>
          <Button onClick={() => run('flood')} disabled={s.booting} title="Overrun the shared invalidation ring">
            Flood the sinval ring
          </Button>
          <Button onClick={() => run(s.booting ? 'bootStep' : 'cold')}>
            {s.booting ? `Bootstrap step ${s.boot + 1} of ${BOOT.length}` : 'Cold start backend B'}
          </Button>
          <Button onClick={() => setS(S0)}>Reset</Button>
        </>
      }
      legend={
        <Legend
          items={[
            { label: 'pg_class row', color: CAT_COLOR.pg_class },
            { label: 'pg_attribute row', color: CAT_COLOR.pg_attribute },
            { label: 'pg_index row', color: CAT_COLOR.pg_index },
            { label: 'pg_statistic row', color: CAT_COLOR.pg_statistic },
            { label: 'uncommitted (dashed outline)', color: 'var(--viz-dirty)' },
            { label: 'dead version', color: 'var(--viz-stale)' },
            { label: 'sinval message unread by B', color: 'var(--viz-warning)' },
            { label: 'sinval message already consumed by B', color: 'var(--viz-neutral)' },
          ]}
        />
      }
      stats={
        <Stats
          items={[
            { label: 'Live catalog rows', value: fmtNum(t.live), hint: 'Row versions written in this session that are still live' },
            { label: 'Dead versions', value: fmtNum(t.dead), hint: 'Catalog bloat — vacuumed like any other heap table' },
            {
              label: 'Unread by B',
              value: fmtNum(t.unread),
              hint: `The ring holds ${fmtNum(RING)} messages before a lagging backend is forced to reset`,
            },
            {
              label: "B's relcache entry",
              value: s.bReset ? 'reset pending' : !s.bEntry ? 'absent' : !s.bValid ? 'invalid' : stale ? 'STALE' : 'valid',
              hint: 'Invalid = flagged, rebuilt on next use. Stale = still flagged valid, and already wrong.',
            },
            { label: 'B sees columns', value: s.bEntry ? fmtNum(bCols.length) : '—', hint: 'Live attributes in the cached TupleDesc' },
            {
              label: 'Cold-start reads',
              value: s.booting || s.bootReads ? fmtNum(s.bootReads) : '—',
              hint: 'File reads plus catalog index scans performed during bootstrap',
            },
          ]}
        />
      }
      note={
        <Note>
          <strong>{s.head}</strong> {s.body}
        </Note>
      }
      table={
        <>
          <table className="viz-table">
            <thead>
              <tr>
                <th>#</th>
                <th>Actor</th>
                <th>Action</th>
                <th>Catalog rows written</th>
                <th>Messages broadcast</th>
                <th>Unread by B</th>
                <th>B sees columns</th>
              </tr>
            </thead>
            <tbody>
              {s.log.length === 0 ? (
                <tr>
                  <td colSpan={7}>Nothing run yet.</td>
                </tr>
              ) : (
                s.log.map((r) => (
                  <tr key={r.n}>
                    <td>{r.n}</td>
                    <td>{r.actor}</td>
                    <td>{r.action}</td>
                    <td>{r.wrote || '—'}</td>
                    <td>{r.queued ? fmtNum(r.queued) : '—'}</td>
                    <td>{fmtNum(r.unread)}</td>
                    <td>{r.bcols}</td>
                  </tr>
                ))
              )}
            </tbody>
          </table>
          <table className="viz-table">
            <thead>
              <tr>
                <th>Mapped relation</th>
                <th>OID</th>
                <th>pg_class.relfilenode</th>
                <th>pg_filenode.map says</th>
                <th>File</th>
              </tr>
            </thead>
            <tbody>
              {s.map.map((m) => (
                <tr key={m.oid}>
                  <td>{m.rel}</td>
                  <td>{m.oid}</td>
                  <td>0</td>
                  <td>{m.filenode}</td>
                  <td>{m.scope === 'shared' ? `global/${m.filenode}` : `base/${DB_OID}/${m.filenode}`}</td>
                </tr>
              ))}
            </tbody>
          </table>
          <table className="viz-table">
            <thead>
              <tr>
                <th>Bootstrap phase</th>
                <th>Reads from</th>
                <th>Reads</th>
                <th>Needs a working catalog?</th>
              </tr>
            </thead>
            <tbody>
              {BOOT.map((p, i) => (
                <tr key={p.name}>
                  <td>
                    {i + 1}. {p.name}
                  </td>
                  <td>{p.file}</td>
                  <td>{p.reads}</td>
                  <td>{i < 2 ? 'no — this is what breaks the circle' : 'yes'}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </>
      }
    >
      <div ref={ref}>
        <TooltipHost>
          <svg
            width={svgW}
            height={height}
            role="img"
            aria-label="Catalog row versions on the left; backend A, the shared invalidation ring and backend B on the right"
          >
            {/* ------------------------------------------------- left: disk + rows */}
            <text x={0} y={12} fill="var(--viz-ink)" fontWeight={600}>
              Shared, on disk
            </text>

            {FILES.map((f, i) => {
              const y = 22 + i * 30;
              const hot = s.booting && s.boot + 1 === f.phase;
              return (
                <g
                  key={f.name}
                  {...tip(
                    <>
                      <strong>{f.name}</strong>
                      <br />
                      {f.note}
                    </>,
                  )}
                  style={{ cursor: 'help' }}
                >
                  <rect
                    x={0}
                    y={y}
                    width={LW - 8}
                    height={24}
                    rx={6}
                    fill="var(--viz-plane)"
                    stroke={hot ? 'var(--viz-good)' : f.bad ? 'var(--viz-critical)' : 'var(--viz-border)'}
                    strokeWidth={hot ? 2 : 1}
                  />
                  <text x={8} y={y + 16} fill="var(--viz-ink)">
                    {f.name}
                  </text>
                  <text
                    x={LW - 16}
                    y={y + 16}
                    textAnchor="end"
                    fill={f.bad ? 'var(--viz-critical)' : 'var(--viz-ink-muted)'}
                  >
                    {f.note}
                  </text>
                </g>
              );
            })}

            <text x={0} y={rowTop - 14} fill="var(--viz-ink)" fontWeight={600}>
              Catalog row versions written this session
            </text>
            {hidden > 0 ? (
              <text x={LW - 8} y={rowTop - 14} textAnchor="end" fill="var(--viz-ink-muted)">
                {hidden} earlier version{hidden === 1 ? '' : 's'} not shown
              </text>
            ) : null}

            {shown.length === 0 ? (
              <text x={0} y={rowTop + 12} fill="var(--viz-ink-muted)">
                None yet — press the “A:” button to run the first step.
              </text>
            ) : null}

            {shown.map((r, i) => {
              const y = rowTop + i * 19;
              const color = r.dead ? 'var(--viz-stale)' : CAT_COLOR[r.cat];
              const solid = r.committed || r.inplace;
              const text = r.cells.length > 52 ? `${r.cells.slice(0, 51)}…` : r.cells;
              return (
                <g
                  key={r.id}
                  {...tip(
                    <>
                      <strong>{r.cat}</strong> — xmin {r.xmin}
                      {r.dead ? ', superseded' : r.committed ? ', committed' : ', uncommitted'}
                      <br />
                      {r.cells}
                      <br />
                      {r.detail}
                    </>,
                  )}
                  style={{ cursor: 'help' }}
                  opacity={r.dead ? 0.5 : 1}
                >
                  <rect
                    x={0}
                    y={y}
                    width={LW - 8}
                    height={16}
                    rx={4}
                    fill="var(--viz-plane)"
                    stroke={solid ? color : 'var(--viz-dirty)'}
                    strokeWidth={solid ? 1 : 1.5}
                    strokeDasharray={solid ? undefined : '3 2'}
                  />
                  <rect x={0} y={y} width={4} height={16} rx={2} fill={color} />
                  <text x={10} y={y + 12} fill={r.dead ? 'var(--viz-ink-muted)' : 'var(--viz-ink)'}>
                    {r.cat}
                  </text>
                  <text x={94} y={y + 12} fill="var(--viz-ink-2)">
                    {text}
                  </text>
                </g>
              );
            })}

            {/* ------------------------------------------- right: A, the ring, B */}
            <text x={RX} y={12} fill="var(--viz-ink)" fontWeight={600}>
              Backend A — xid {s.xid}
            </text>
            <rect
              x={RX}
              y={22}
              width={RW}
              height={66}
              rx={8}
              fill="var(--viz-plane)"
              stroke={s.aOpen ? 'var(--viz-dirty)' : 'var(--viz-border)'}
              strokeWidth={s.aOpen ? 2 : 1}
            />
            <text x={RX + 10} y={42} fill="var(--viz-ink)">
              {s.aOpen ? 'transaction OPEN — rows written, invisible to everyone else' : 'idle — last transaction committed'}
            </text>
            <text x={RX + 10} y={60} fill="var(--viz-ink-2)">
              next: {nextStep.sql.length > 56 ? `${nextStep.sql.slice(0, 55)}…` : nextStep.sql}
            </text>
            <text x={RX + 10} y={78} fill={s.pending.length ? 'var(--viz-dirty)' : 'var(--viz-ink-muted)'}>
              private invalidation queue: {s.pending.length} message{s.pending.length === 1 ? '' : 's'} — broadcast at COMMIT
            </text>

            <text x={RX} y={116} fill="var(--viz-ink)" fontWeight={600}>
              Shared invalidation ring (sinval)
            </text>
            <text x={RX + RW} y={116} textAnchor="end" fill="var(--viz-ink-muted)">
              {fmtNum(s.queue.length)} written · {fmtNum(t.unread)} unread by B
            </text>
            {(() => {
              const slots = 16;
              const w = Math.floor((RW - 4) / slots);
              const view = s.queue.slice(-slots);
              const base = s.queue.length - view.length;
              const pad = slots - view.length;
              return Array.from({ length: slots }, (_, i) => {
                const m = i >= pad ? view[i - pad] : null;
                const unread = m ? base + (i - pad) >= s.bPtr : false;
                return (
                  <g
                    key={i}
                    {...(m
                      ? tip(
                          <>
                            <strong>
                              #{fmtNum(m.seq)} {m.kind}
                            </strong>{' '}
                            — {m.target}
                            <br />
                            {m.hint}
                          </>,
                        )
                      : {})}
                    style={{ cursor: m ? 'help' : 'default' }}
                  >
                    <rect
                      x={RX + i * w}
                      y={124}
                      width={w - 3}
                      height={22}
                      rx={4}
                      fill={m ? (unread ? 'var(--viz-warning)' : 'var(--viz-neutral)') : 'var(--viz-plane)'}
                      stroke={m ? 'var(--viz-border)' : 'var(--viz-grid)'}
                    />
                    {m ? (
                      <text x={RX + i * w + (w - 3) / 2} y={139} textAnchor="middle" fill="var(--viz-ink)">
                        {KIND_TAG[m.kind]}
                      </text>
                    ) : null}
                  </g>
                );
              });
            })()}
            <text x={RX} y={162} fill={s.bReset ? 'var(--viz-critical)' : 'var(--viz-ink-muted)'}>
              B’s read pointer: {fmtNum(s.bPtr)} of {fmtNum(s.queue.length)}
              {s.bReset ? ' — RESET FLAG SET, discard everything' : ''}
            </text>

            {s.booting ? (
              <>
                <text x={RX} y={194} fill="var(--viz-ink)" fontWeight={600}>
                  Backend B — bootstrapping
                </text>
                {BOOT.map((p, i) => {
                  const y = 202 + i * 42;
                  const done = s.boot > i;
                  const cur = s.boot === i;
                  return (
                    <g key={p.name}>
                      <rect
                        x={RX}
                        y={y}
                        width={RW}
                        height={36}
                        rx={6}
                        fill="var(--viz-plane)"
                        stroke={cur ? 'var(--viz-good)' : done ? 'var(--viz-1)' : 'var(--viz-border)'}
                        strokeWidth={cur ? 2 : 1}
                        opacity={done || cur ? 1 : 0.5}
                      />
                      <text x={RX + 10} y={y + 15} fill="var(--viz-ink)">
                        {done ? '✓' : `${i + 1}.`} {p.name}
                      </text>
                      <text x={RX + 10} y={y + 29} fill="var(--viz-ink-muted)">
                        {p.file}
                      </text>
                    </g>
                  );
                })}
              </>
            ) : (
              <>
                <text x={RX} y={194} fill="var(--viz-ink)" fontWeight={600}>
                  Backend B — private caches
                </text>
                <rect
                  x={RX}
                  y={202}
                  width={RW}
                  height={126}
                  rx={8}
                  fill="var(--viz-plane)"
                  stroke={
                    !s.bEntry
                      ? 'var(--viz-border)'
                      : stale
                        ? 'var(--viz-critical)'
                        : !s.bValid
                          ? 'var(--viz-warning)'
                          : 'var(--viz-1)'
                  }
                  strokeWidth={stale ? 2 : 1}
                />
                <text x={RX + 10} y={222} fill="var(--viz-ink)">
                  relcache: orders ({TAB_OID}){' '}
                  {!s.bEntry
                    ? '— absent'
                    : !s.bValid
                      ? '— flagged INVALID'
                      : stale
                        ? '— flagged valid, and WRONG'
                        : '— valid'}
                </text>
                <text x={RX + 10} y={240} fill="var(--viz-ink-2)">
                  TupleDesc: {s.bEntry ? bCols.map((c) => `${c.name}:${c.typ}`).join('  ') || '(none)' : '—'}
                </text>
                <text x={RX + 10} y={258} fill="var(--viz-ink-2)">
                  relnatts {s.bEntry ? s.bEntry.natts : '—'} · indexes{' '}
                  {s.bEntry ? bIdx.map((i) => i.name).join(', ') || 'none' : '—'}
                </text>
                <text x={RX + 10} y={276} fill="var(--viz-ink-2)">
                  reltuples{' '}
                  {s.bEntry ? (s.bEntry.reltuples < 0 ? '-1 (never analyzed)' : fmtNum(s.bEntry.reltuples)) : '—'} ·
                  relpages {s.bEntry ? s.bEntry.relpages : '—'}
                </text>
                <text x={RX + 10} y={298} fill={stale ? 'var(--viz-critical)' : 'var(--viz-ink)'}>
                  plan: {s.bEntry ? planOf(s.bEntry) : 'nothing cached — a full catalog lookup first'}
                </text>
                <text x={RX + 10} y={318} fill="var(--viz-ink-muted)">
                  syscache: RELOID, ATTNAME, ATTNUM, INDEXRELID, TYPEOID
                  {s.committed.hasStats ? ', STATRELATTINH' : ''}
                </text>

                <rect x={RX} y={340} width={RW} height={48} rx={8} fill="var(--viz-plane)" stroke="var(--viz-border)" />
                <text x={RX + 10} y={358} fill="var(--viz-ink)">
                  What a fresh catalog read would say
                </text>
                <text x={RX + 10} y={376} fill="var(--viz-ink-2)">
                  {liveCols(s.committed).map((c) => c.name).join(', ')} · {planOf(s.committed)}
                </text>

                {bLocked ? (
                  <text x={RX} y={406} fill={DDL_LOCK[ddl].conflicts ? 'var(--viz-warning)' : 'var(--viz-ink-muted)'}>
                    B holds AccessShareLock on orders · {DDL[ddl].label} wants {DDL_LOCK[ddl].mode} on{' '}
                    {DDL_LOCK[ddl].target} — {DDL_LOCK[ddl].conflicts ? 'conflict, A waits' : 'no conflict'}
                  </text>
                ) : null}
                {s.bMapStale ? (
                  <text x={RX} y={424} fill="var(--viz-critical)">
                    B’s cached relfilenode for pg_class is stale — it must re-read pg_filenode.map.
                  </text>
                ) : null}
              </>
            )}
          </svg>
        </TooltipHost>
      </div>
    </VizPanel>
  );
}
