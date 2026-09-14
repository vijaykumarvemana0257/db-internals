import { useMemo, useState } from 'react';
import {
  VizPanel,
  Segmented,
  Choice,
  Check,
  Legend,
  Stats,
  Note,
  TooltipHost,
  useTip,
  useSize,
  fmtBytes,
  fmtNum,
} from './Viz';

/**
 * A clickable index of an engine's layers.
 *
 * The stack is the same one Hellerstein, Stonebraker and Hamilton describe in
 * "Architecture of a Database System": a request falls through client communications,
 * the relational query processor, and the transactional storage manager, while the
 * transaction, lock, log and catalog managers cut across all of it.
 *
 * Switching engines adds, merges or removes layers *in place* — RocksDB has no parser
 * or planner, SQLite has no network layer and compiles straight to VDBE bytecode,
 * MySQL exposes a handler-API seam PostgreSQL only partly has.
 *
 * Picking a statement annotates every layer with what passes through it. Those counts
 * are computed from one stated model (2,000,000-row `orders` table, 120-byte rows,
 * 7 days out of 365 matched, 4,200 distinct customers) — arithmetic, not a benchmark.
 */

/* ------------------------------------------------------------------ engines */

type EngineId = 'pg' | 'innodb' | 'sqlite' | 'rocksdb';

type Engine = {
  id: EngineId;
  label: string;
  short: string;
  page: number;
  clustered: boolean; // the index leaf holds the row, so no separate heap fetch
  inProcess: boolean; // linked into the caller: no socket, no wire protocol
  pageWord: string;
};

const ENGINES: Engine[] = [
  { id: 'pg', label: 'PostgreSQL', short: 'PostgreSQL', page: 8192, clustered: false, inProcess: false, pageWord: 'page' },
  { id: 'innodb', label: 'MySQL + InnoDB', short: 'InnoDB', page: 16384, clustered: true, inProcess: false, pageWord: 'page' },
  { id: 'sqlite', label: 'SQLite', short: 'SQLite', page: 4096, clustered: true, inProcess: true, pageWord: 'page' },
  { id: 'rocksdb', label: 'RocksDB', short: 'RocksDB', page: 4096, clustered: true, inProcess: true, pageWord: 'block' },
];

/* --------------------------------------------------------------- managers */

type MgrId = 'txn' | 'lock' | 'log' | 'catalog';

const MGRS: { id: MgrId; glyph: string; label: string; color: string; detail: string; module: string; href: string }[] = [
  {
    id: 'txn',
    glyph: 'T',
    label: 'Transaction manager',
    color: 'var(--viz-1)',
    detail:
      'Hands out transaction ids and snapshots, tracks which transactions are in flight, and decides at commit what the rest of the engine is allowed to see. In PostgreSQL that is the proc array plus clog; in InnoDB it is trx_sys plus the read view. Nothing below it can answer "is this row visible to me" without asking.',
    module: 'Transactions, Serializability and Isolation Levels',
    href: '/p04-transactions-concurrency-control-and-recovery/transactions-and-isolation/01-acid-precisely/',
  },
  {
    id: 'lock',
    glyph: 'K',
    label: 'Lock manager',
    color: 'var(--viz-2)',
    detail:
      'A hash table of logical locks — table, row, key range — with a wait-for graph for deadlock detection. Held for the length of a transaction. It is not the same thing as a latch: latches protect an in-memory structure for the duration of a few instructions and are never deadlock-detected.',
    module: 'Lock-Based Concurrency Control',
    href: '/p04-transactions-concurrency-control-and-recovery/lock-based-concurrency-control/01-two-phase-locking-and-strict-2pl/',
  },
  {
    id: 'log',
    glyph: 'L',
    label: 'Log manager',
    color: 'var(--viz-3)',
    detail:
      'Owns the write-ahead log: assigns LSNs, buffers records, and enforces the rule that a modified page may not reach storage before the log record describing it is durable. It is why the buffer pool cannot evict a dirty page on its own schedule, and why commit costs a flush.',
    module: 'Write-Ahead Logging, Checkpoints and the Commit Path',
    href: '/p04-transactions-concurrency-control-and-recovery/write-ahead-logging-and-recovery/01-log-records-lsns-and-the-two-rules-of-write-ahead-logging/',
  },
  {
    id: 'catalog',
    glyph: 'C',
    label: 'Catalog / metadata',
    color: 'var(--viz-4)',
    detail:
      'The engine’s own tables describing your tables: pg_class, pg_attribute, pg_statistic in PostgreSQL; the InnoDB data dictionary in mysql.* since 8.0; sqlite_schema in SQLite. Every layer above the access methods reads it, which is why a catalog lock stalls parsing, planning and execution at once.',
    module: 'Logical schema vs physical storage',
    href: '/p01-foundations/anatomy-of-a-database-engine/02-logical-schema-vs-physical-storage/',
  },
];

/* ----------------------------------------------------------------- layers */

type LayerId =
  | 'wire'
  | 'parser'
  | 'binder'
  | 'rewriter'
  | 'optimizer'
  | 'executor'
  | 'seam'
  | 'access'
  | 'pool'
  | 'storage';

type Variant = {
  state: 'present' | 'merged' | 'absent';
  name: string;
  sub: string;
  knobs: string;
  detail: string;
};

type Layer = {
  id: LayerId;
  generic: string;
  mgrs: MgrId[];
  module: string;
  href: string;
  byEngine: Record<EngineId, Variant>;
};

const LAYERS: Layer[] = [
  {
    id: 'wire',
    generic: 'Wire protocol / client comms',
    mgrs: [],
    module: 'Connections and Pooling',
    href: '/p05-using-the-database-well/connections-pooling-and-process-models/01-connection-lifecycle-and-cost-tcp-tls-scram-and-process-per-/',
    byEngine: {
      pg: {
        state: 'present',
        name: 'Wire protocol (libpq v3)',
        sub: 'postmaster forks a backend per connection · port 5432',
        knobs: 'max_connections, listen_addresses, ssl, tcp_keepalives_idle',
        detail:
          'Message-framed protocol 3.0: Query for the simple path, Parse/Bind/Describe/Execute for the extended one. A connection is an OS process, so it costs memory and a fork, which is the entire reason PgBouncer exists.',
      },
      innodb: {
        state: 'present',
        name: 'Classic MySQL protocol',
        sub: 'one thread per connection · port 3306',
        knobs: 'max_connections, thread_cache_size, max_allowed_packet',
        detail:
          'Length-prefixed packets with a 3-byte length and a sequence id, capped by max_allowed_packet. A connection is a thread rather than a process, so it is cheaper than a PostgreSQL backend but still not free.',
      },
      sqlite: {
        state: 'absent',
        name: 'No network layer',
        sub: 'SQLite is a library: sqlite3_step() runs in your thread',
        knobs: '—',
        detail:
          'There is no server, no socket and no protocol: the query runs on the calling thread, inside the calling process, against a file. Concurrency is therefore a file-locking problem, not a connection-management one — which is what WAL mode and busy_timeout are about.',
      },
      rocksdb: {
        state: 'absent',
        name: 'No network layer',
        sub: 'an embedded C++ library: DB::Get / DB::Put',
        knobs: '—',
        detail:
          'RocksDB is linked into the process that uses it. When you talk to CockroachDB or TiKV over a wire, the wire protocol belongs to the SQL layer above; RocksDB (or Pebble) never sees a socket.',
      },
    },
  },
  {
    id: 'parser',
    generic: 'Parser',
    mgrs: [],
    module: 'From SQL Text to Logical Plan',
    href: '/p03-query-processing/sql-to-logical-plan/01-parsing-and-the-ast/',
    byEngine: {
      pg: {
        state: 'present',
        name: 'Parser (flex + bison)',
        sub: 'scan.l, gram.y → raw parse tree',
        knobs: 'standard_conforming_strings, backslash_quote',
        detail:
          'Pure syntax. It builds a raw parse tree of SelectStmt/ColumnRef nodes without touching the catalog, so a typo in a table name is *not* a parse error — it surfaces one layer down.',
      },
      innodb: {
        state: 'present',
        name: 'Parser (sql_yacc.yy)',
        sub: 'SQL text → parse tree of Item / SELECT_LEX',
        knobs: 'sql_mode, character_set_client',
        detail:
          'Server-layer, above the storage engine entirely. sql_mode changes what parses at all, which is why the same statement is accepted on one MySQL and rejected on another.',
      },
      sqlite: {
        state: 'present',
        name: 'Parser (lemon)',
        sub: 'parse.y → AST, then straight into code generation',
        knobs: 'SQLITE_MAX_SQL_LENGTH (compile time)',
        detail:
          'Lemon is SQLite’s own LALR(1) generator, chosen for reentrancy and no global state. The parse tree is transient: SQLite does not keep a query tree around, it emits bytecode as it parses.',
      },
      rocksdb: {
        state: 'absent',
        name: 'No parser',
        sub: 'the API is the query language',
        knobs: '—',
        detail:
          'Get(key), Put(key, value), Delete(key), NewIterator(). There is nothing to parse, which is exactly what a key-value store trades away in exchange for a predictable cost model.',
      },
    },
  },
  {
    id: 'binder',
    generic: 'Binder / analyzer',
    mgrs: ['catalog'],
    module: 'Binding, name resolution and semantic analysis',
    href: '/p03-query-processing/sql-to-logical-plan/02-binding-name-resolution-and-semantic-analysis/',
    byEngine: {
      pg: {
        state: 'present',
        name: 'Analyzer (parse_analyze)',
        sub: 'names → OIDs · builds a Query with range-table entries',
        knobs: 'search_path',
        detail:
          'The first layer that reads the catalog. It resolves every identifier against pg_class/pg_attribute through search_path, type-checks expressions, resolves operators, and locks the relations it touches. "relation does not exist" is raised here, not by the parser.',
      },
      innodb: {
        state: 'present',
        name: 'Name resolution / prepare',
        sub: 'Item::fix_fields() binds columns to tables',
        knobs: 'table_definition_cache, table_open_cache',
        detail:
          'Resolution opens table definitions through the data dictionary and the table-definition cache. A cold cache turns name resolution into disk I/O, which is why table_open_cache shows up in slow-start incidents.',
      },
      sqlite: {
        state: 'merged',
        name: 'Merged into code generation',
        sub: 'sqlite3Prepare resolves and emits in one pass',
        knobs: '—',
        detail:
          'SQLite resolves names against sqlite_schema during code generation: there is no separate bound-query representation. The compiled form is the VDBE program, and it is cached per prepared statement.',
      },
      rocksdb: {
        state: 'absent',
        name: 'No binder',
        sub: 'keys are opaque byte strings',
        knobs: '—',
        detail:
          'A key has no name, type or schema. Whatever structure your keys have (a table prefix, an encoded tuple) is imposed by the layer above — which is precisely how CockroachDB and TiKV build SQL on top.',
      },
    },
  },
  {
    id: 'rewriter',
    generic: 'Rewriter',
    mgrs: ['catalog'],
    module: 'Algebraic equivalences and rewrite rules',
    href: '/p03-query-processing/sql-to-logical-plan/05-algebraic-equivalences-and-rewrite-rules/',
    byEngine: {
      pg: {
        state: 'present',
        name: 'Rewriter (rewriteHandler.c)',
        sub: 'views, rules and row-level security expand here',
        knobs: 'row_security',
        detail:
          'A view is a rule, and the rewriter substitutes its definition into the query tree before the planner ever sees it — which is why a view costs nothing by itself and why RLS policies appear as extra quals in EXPLAIN.',
      },
      innodb: {
        state: 'present',
        name: 'Transformations',
        sub: 'view merging, IN → semijoin, derived-table merging',
        knobs: 'optimizer_switch (derived_merge, semijoin, …)',
        detail:
          'MySQL calls these transformations and does them between resolution and optimization. optimizer_switch turns individual rewrites off, which is the blunt instrument for a rewrite that makes a specific query worse.',
      },
      sqlite: {
        state: 'merged',
        name: 'Merged into the planner',
        sub: 'flattening, pushdown and co-routines in where.c',
        knobs: '—',
        detail:
          'SQLite applies subquery flattening and predicate pushdown as part of planning rather than as a separate pass over a query tree.',
      },
      rocksdb: {
        state: 'absent',
        name: 'No rewriter',
        sub: 'nothing to rewrite',
        knobs: '—',
        detail: 'There is no declarative form, so there is no equivalence-preserving transformation to apply.',
      },
    },
  },
  {
    id: 'optimizer',
    generic: 'Optimizer / planner',
    mgrs: ['catalog'],
    module: 'Cost-Based Optimization',
    href: '/p03-query-processing/cost-based-optimization/01-cost-models-i-o-cpu-and-their-calibration/',
    byEngine: {
      pg: {
        state: 'present',
        name: 'Planner (path enumeration)',
        sub: 'costs every access path and join order, picks one plan',
        knobs: 'random_page_cost, effective_cache_size, geqo_threshold, jit',
        detail:
          'Bottom-up dynamic programming over join orders, switching to the genetic optimizer past geqo_threshold (12 relations). Costs come from pg_statistic; when they are wrong, everything below this layer does the wrong amount of work at full speed.',
      },
      innodb: {
        state: 'present',
        name: 'Optimizer',
        sub: 'greedy join-order search with a cost model',
        knobs: 'optimizer_search_depth, optimizer_switch, eq_range_index_dive_limit',
        detail:
          'MySQL searches join orders greedily to a bounded depth and asks the storage engine for row estimates through handler methods such as records_in_range() — so the numbers the optimizer reasons about come from InnoDB, not from the server.',
      },
      sqlite: {
        state: 'present',
        name: 'Query planner (NGQP)',
        sub: 'index selection driven by sqlite_stat1',
        knobs: 'ANALYZE, PRAGMA optimize, LIKELY()/UNLIKELY()',
        detail:
          'Small and deliberately stable — SQLite treats a plan change as a compatibility risk. Without ANALYZE it plans on heuristics, and the first ANALYZE on an old database can change every plan at once.',
      },
      rocksdb: {
        state: 'absent',
        name: 'No planner',
        sub: 'the caller chose the access path when it chose the API call',
        knobs: '—',
        detail:
          'Get is a point lookup and NewIterator is a range scan; there is no cost model and no alternative to consider. Every plan decision in a RocksDB-backed SQL database happens in the layer above.',
      },
    },
  },
  {
    id: 'executor',
    generic: 'Executor',
    mgrs: ['txn', 'lock', 'catalog'],
    module: 'Execution Engines, Vectorization and Parallelism',
    href: '/p03-query-processing/execution-engines-and-resource-control/01-volcano-iterator-model/',
    byEngine: {
      pg: {
        state: 'present',
        name: 'Executor (Volcano iterator)',
        sub: 'ExecProcNode pulls one tuple at a time up the plan tree',
        knobs: 'work_mem, hash_mem_multiplier, max_parallel_workers_per_gather',
        detail:
          'Demand-driven: each node pulls from its children. work_mem is a limit per node per worker, not per query, so one plan with three hash joins and four workers can reserve twelve times work_mem.',
      },
      innodb: {
        state: 'present',
        name: 'Iterator executor',
        sub: 'MySQL 8.0 iterators; hash join for equi-joins',
        knobs: 'join_buffer_size, sort_buffer_size, tmp_table_size',
        detail:
          'MySQL 8.0 replaced the old row-at-a-time executor with an iterator model that EXPLAIN ANALYZE can report timings for. Spill buffers are per-session knobs, so raising them raises worst-case memory by the number of connections.',
      },
      sqlite: {
        state: 'present',
        name: 'VDBE bytecode machine',
        sub: 'a register machine, not an operator tree',
        knobs: 'PRAGMA temp_store, SQLITE_MAX_MEMORY',
        detail:
          'SQLite compiles a statement to opcodes (OpenRead, SeekGE, Column, ResultRow) and steps them. EXPLAIN prints the program itself — the closest thing in any mainstream engine to reading the executor’s mind.',
      },
      rocksdb: {
        state: 'merged',
        name: 'Merged into the API call',
        sub: 'the iterator is the executor',
        knobs: 'ReadOptions (snapshot, iterate_upper_bound, fill_cache)',
        detail:
          'An Iterator merges the memtables and the SST levels into one sorted stream; the caller drives it. That merge is the only execution RocksDB does.',
      },
    },
  },
  {
    id: 'seam',
    generic: 'Storage-engine seam',
    mgrs: ['txn'],
    module: 'Row, column and key-value physical models',
    href: '/p01-foundations/anatomy-of-a-database-engine/03-row-column-and-key-value-physical-models/',
    byEngine: {
      pg: {
        state: 'present',
        name: 'Table access method (pg_am)',
        sub: 'heap is one AM among possible others (since PG 12)',
        knobs: 'default_table_access_method',
        detail:
          'PostgreSQL 12 turned the table layer into an interface: slot-based tuple fetch, scan, insert/update/delete callbacks. The index AM interface is far older. The seam is real but narrow — the WAL, the buffer pool and vacuum are still shared, so a table AM cannot bring its own log.',
      },
      innodb: {
        state: 'present',
        name: 'Handler API (class handler)',
        sub: 'ha_innobase, ha_rocksdb, ha_myisam, ha_memory',
        knobs: 'default_storage_engine, SHOW ENGINES',
        detail:
          'The widest seam in any mainstream database: rnd_next(), index_read(), write_row(), plus transaction callbacks. Everything above it is the MySQL server; everything below is a replaceable engine with its own pages, its own log and its own locking — which is how MyRocks (an LSM) and InnoDB (a B+tree) sit under the same parser.',
      },
      sqlite: {
        state: 'absent',
        name: 'No storage-engine seam',
        sub: 'btree.c is the only backend; the VFS seam sits lower',
        knobs: '—',
        detail:
          'SQLite’s pluggable interface is the VFS, one layer further down — it replaces the OS, not the storage engine. Virtual tables let you expose foreign data as a table, but they do not replace the B-tree for real tables.',
      },
      rocksdb: {
        state: 'absent',
        name: 'RocksDB *is* the pluggable engine',
        sub: 'it sits below somebody else’s seam',
        knobs: '—',
        detail:
          'MyRocks plugs RocksDB into MySQL’s handler API; TiKV and (historically) CockroachDB embed it directly. From RocksDB’s side there is no seam, only an API.',
      },
    },
  },
  {
    id: 'access',
    generic: 'Access methods',
    mgrs: ['txn', 'lock', 'log', 'catalog'],
    module: 'B-Trees: Structure, Maintenance and Key Encoding',
    href: '/p02-storage-engines/b-trees-and-variants/01-b-tree-structure-invariants-and-search/',
    byEngine: {
      pg: {
        state: 'present',
        name: 'Heap + index AMs',
        sub: 'btree, hash, gist, gin, spgist, brin · heap tuples with xmin/xmax',
        knobs: 'fillfactor, autovacuum_*, index-specific storage parameters',
        detail:
          'The heap is unordered and indexes point at it by (block, offset). An index scan therefore costs a descent *plus* a heap fetch unless the visibility map allows an index-only scan.',
      },
      innodb: {
        state: 'present',
        name: 'Clustered B+tree',
        sub: 'the PK index leaf *is* the row; secondary indexes store the PK',
        knobs: 'innodb_fill_factor, innodb_page_size, adaptive hash index',
        detail:
          'A primary-key lookup ends at the leaf with the row in hand. A secondary-index lookup pays a second descent through the clustered index unless the index covers the query — the reason a wide PK makes every secondary index bigger.',
      },
      sqlite: {
        state: 'present',
        name: 'B-tree (btree.c)',
        sub: 'rowid tables and WITHOUT ROWID tables',
        knobs: 'PRAGMA page_size, auto_vacuum',
        detail:
          'One file holds every table and index as a separate B-tree, rooted at a page number recorded in sqlite_schema. INTEGER PRIMARY KEY is the rowid, so it is a clustered lookup.',
      },
      rocksdb: {
        state: 'present',
        name: 'LSM: memtable + SSTables',
        sub: 'skiplist memtable, levelled SSTs, bloom filters per file',
        knobs: 'write_buffer_size, max_bytes_for_level_base, filter_policy (bloom bits/key)',
        detail:
          'A read probes the memtables, then one file per level, skipping files whose bloom filter says no. A write never reads: it appends to the WAL and inserts into the memtable, and compaction does the sorting later.',
      },
    },
  },
  {
    id: 'pool',
    generic: 'Buffer pool / page cache',
    mgrs: ['log', 'lock'],
    module: 'The Buffer Pool',
    href: '/p02-storage-engines/buffer-pool/01-buffer-pool-structure-frames-page-table-and-pins/',
    byEngine: {
      pg: {
        state: 'present',
        name: 'Shared buffers (bufmgr.c)',
        sub: 'clock sweep · pinned frames · also relies on the OS page cache',
        knobs: 'shared_buffers (128 MB default), bgwriter_*, effective_cache_size',
        detail:
          'PostgreSQL deliberately keeps a modest pool and lets the kernel cache the rest, so a page can be cached twice. effective_cache_size tells the planner how much total caching to assume; it allocates nothing.',
      },
      innodb: {
        state: 'present',
        name: 'InnoDB buffer pool',
        sub: 'LRU with midpoint insertion · young/old sublists',
        knobs: 'innodb_buffer_pool_size (128 MB default), innodb_buffer_pool_instances, innodb_old_blocks_pct',
        detail:
          'Sized to most of the machine because InnoDB is normally run with innodb_flush_method=O_DIRECT and does not want the kernel caching the same pages again. New pages enter at the midpoint of the LRU list so a full scan cannot evict the hot set in one pass.',
      },
      sqlite: {
        state: 'present',
        name: 'Page cache (pcache1)',
        sub: 'per-connection by default, or one shared cache',
        knobs: 'PRAGMA cache_size (negative = KiB), PRAGMA mmap_size',
        detail:
          'Small by design — a couple of megabytes unless you raise it — because SQLite expects the operating system’s page cache to be doing the real caching underneath it.',
      },
      rocksdb: {
        state: 'present',
        name: 'Block cache + memtables',
        sub: 'LRU/clock cache of *uncompressed* SST blocks',
        knobs: 'block_cache size, cache_index_and_filter_blocks, pin_l0_filter_and_index_blocks_in_cache',
        detail:
          'Two very different memory pools: memtables hold unflushed writes, the block cache holds read-side SST blocks. Leaving index and filter blocks out of the cache accounting is the classic way a RocksDB process exceeds the memory you budgeted for it.',
      },
    },
  },
  {
    id: 'storage',
    generic: 'Storage manager / files',
    mgrs: ['log'],
    module: 'Pages, Tuples and Heap Files',
    href: '/p02-storage-engines/pages-tuples-heap-files/01-page-anatomy-and-the-slotted-page/',
    byEngine: {
      pg: {
        state: 'present',
        name: 'smgr / md.c',
        sub: 'base/<db>/<relfilenode>, 1 GB segments, WAL in pg_wal',
        knobs: 'wal_buffers, wal_segment_size (16 MB), max_wal_size, full_page_writes',
        detail:
          'Translates a (relation, block) into a file descriptor and an offset. Files grow in 1 GB segments; the WAL is a separate stream of pre-allocated, recycled 16 MB segments so a commit flush changes no file metadata.',
      },
      innodb: {
        state: 'present',
        name: 'Tablespaces + redo log',
        sub: 'per-table .ibd files, doublewrite buffer, ib_logfile / #innodb_redo',
        knobs: 'innodb_flush_method, innodb_log_buffer_size (16 MB), innodb_flush_log_at_trx_commit, innodb_doublewrite',
        detail:
          'InnoDB writes every dirty page twice — once into the doublewrite buffer, once home — so that a torn 16 KB page can be repaired at recovery, because a partial page write is not something the redo log alone can fix.',
      },
      sqlite: {
        state: 'present',
        name: 'Pager + VFS',
        sub: 'one file, plus -wal and -shm, through a swappable VFS',
        knobs: 'PRAGMA journal_mode, PRAGMA synchronous, PRAGMA wal_autocheckpoint',
        detail:
          'The pager owns transactions, locking and recovery over a single file; the VFS is the seam where the OS goes, which is how SQLite runs on phones, in browsers and over network filesystems it does not trust.',
      },
      rocksdb: {
        state: 'present',
        name: 'Env / FileSystem',
        sub: 'immutable .sst files, WAL, MANIFEST, CURRENT',
        knobs: 'max_background_jobs, target_file_size_base, WAL_ttl_seconds',
        detail:
          'SST files are written once and never modified; the MANIFEST is the log of which files make up each level. Recovery means replaying the WAL into a fresh memtable and reading the MANIFEST — never repairing a page in place.',
      },
    },
  },
];

/* ------------------------------------------------------------- the model */

type StmtId = 'point' | 'range' | 'update';

const STMTS: { id: StmtId; label: string; sql: string; kv: string }[] = [
  {
    id: 'point',
    label: 'Point lookup',
    sql: 'SELECT * FROM orders WHERE id = 42;',
    kv: 'db->Get(ReadOptions(), "orders/42", &value)',
  },
  {
    id: 'range',
    label: 'Range + GROUP BY',
    sql: "SELECT customer_id, count(*) FROM orders WHERE created_at >= now() - interval '7 days' GROUP BY customer_id;",
    kv: 'it->Seek("orders/ts/…"); while (it->Valid()) …',
  },
  {
    id: 'update',
    label: 'Single-row UPDATE',
    sql: "UPDATE orders SET status = 'shipped' WHERE id = 42;",
    kv: 'db->Put(WriteOptions(), "orders/42", value)',
  },
];

const ROWS = 2_000_000;
const ROW_BYTES = 120;
const SLOT = 4;
const KEY_BYTES = 20; // key + downlink in an internal B-tree node
const MATCHED = Math.round((ROWS * 7) / 365); // one week out of a year
const GROUPS = 4_200; // distinct customers in that week
const HIT_RATE_HOT = 0.9; // assumed buffer-pool hit rate for a hot single-row access
const HIT_RATE_SCAN = 0.3; // assumed hit rate while scanning a week of rows

/**
 * Only PostgreSQL changes its log volume on the first write to a page after a checkpoint.
 * The other three answer the torn-page problem somewhere other than the log record.
 */
const LOG_HINT: Record<EngineId, (fpi: boolean) => string> = {
  pg: (fpi) =>
    fpi
      ? 'full_page_writes puts a whole 8 KB page image in the WAL the first time a page is touched after a checkpoint'
      : 'Just the change description',
  innodb: () => 'A redo delta; torn pages are handled by the doublewrite buffer, not by a bigger record',
  sqlite: () => 'A WAL frame is a whole page by construction, checkpoint or not',
  rocksdb: () => 'The WAL record is the key and value themselves; there are no pages to tear',
};

type Ann = { a: string; b: string };

type Model = {
  ann: Record<LayerId, Ann>;
  requests: number;
  hits: number;
  misses: number;
  rowsOut: number;
  wireBytes: number;
  logBytes: number;
  dirtied: number;
  height: number;
  rowsPerPage: number;
};

function model(e: Engine, stmt: StmtId, fpi: boolean): Model {
  const rowsPerPage = Math.floor((e.page - 128) / (ROW_BYTES + SLOT));
  const fanout = Math.floor((e.page - 128) / KEY_BYTES);
  const height = Math.max(2, Math.ceil(Math.log(ROWS) / Math.log(fanout)));
  const lsm = e.id === 'rocksdb';
  const levels = 5; // L0 plus four levelled tiers holding a 240 MB dataset

  let idxPages = 0;
  let dataPages = 0;
  let rowsScanned = 0;
  let rowsOut = 0;
  let wireBytes = 0;
  let logBytes = 0;
  let dirtied = 0;
  let paths = 0;

  if (stmt === 'point') {
    // In an LSM a bloom filter answers "not here" for most levels without touching a block,
    // so a point read costs a filter/index block plus the one data block that holds the key.
    idxPages = lsm ? 2 : height;
    dataPages = lsm ? 1 : e.clustered ? 0 : 1;
    rowsScanned = 1;
    rowsOut = 1;
    wireBytes = e.inProcess ? 0 : ROW_BYTES + 60;
    paths = 2;
  } else if (stmt === 'range') {
    const leaves = Math.ceil(MATCHED / fanout);
    idxPages = lsm ? levels : leaves + height - 1;
    dataPages = lsm
      ? Math.ceil((MATCHED * (ROW_BYTES + 16)) / e.page)
      : Math.ceil(MATCHED / rowsPerPage);
    rowsScanned = MATCHED;
    rowsOut = GROUPS;
    wireBytes = e.inProcess ? 0 : GROUPS * 24 + 120;
    paths = 4;
  } else {
    idxPages = lsm ? 0 : height;
    dataPages = lsm ? 0 : e.clustered ? 0 : 1;
    rowsScanned = 1;
    rowsOut = 0;
    wireBytes = e.inProcess ? 0 : 45;
    dirtied = lsm ? 0 : 1;
    paths = 2;
    if (e.id === 'pg') logBytes = fpi ? e.page + 96 : 190;
    else if (e.id === 'innodb') logBytes = 90; // redo is a delta; torn pages are handled by doublewrite, not by a bigger record
    else if (e.id === 'sqlite') logBytes = e.page + 24;
    else logBytes = 16 + ROW_BYTES + 12;
  }

  const requests = idxPages + dataPages;
  // A stated assumption, not a measurement: upper index levels of a hot key stay resident,
  // while a week-wide scan mostly walks pages nobody has touched.
  const hitRate = stmt === 'range' ? HIT_RATE_SCAN : HIT_RATE_HOT;
  const hits = Math.round(requests * hitRate);
  const misses = requests - hits;

  const sqlBytes = STMTS.find((x) => x.id === stmt)!.sql.length;
  const dash: Ann = { a: '—', b: '—' };
  const idents = stmt === 'range' ? 4 : stmt === 'update' ? 3 : 2;

  const ann: Record<LayerId, Ann> = {
    wire: e.inProcess
      ? { a: 'in-process call', b: 'rows by pointer' }
      : { a: `${sqlBytes} B of SQL in`, b: `${fmtBytes(wireBytes)} out` },
    parser: lsm ? dash : { a: `${sqlBytes} B of SQL`, b: '1 parse tree' },
    binder: lsm ? dash : { a: '1 parse tree', b: `${idents} names → catalog ids` },
    rewriter: lsm ? dash : { a: '1 query tree', b: '1 query tree' },
    optimizer: lsm ? dash : { a: '1 query tree', b: `1 plan of ${paths} costed` },
    executor: lsm
      ? { a: `${fmtNum(rowsScanned)} keys merged`, b: `${fmtNum(rowsOut)} to the caller` }
      : { a: `${fmtNum(rowsScanned)} rows in`, b: `${fmtNum(rowsOut)} rows out` },
    seam: e.id === 'pg' || e.id === 'innodb' ? { a: `${fmtNum(rowsScanned)} AM calls`, b: 'tuples by slot' } : dash,
    access: { a: `${fmtNum(idxPages)} index ${e.pageWord}s`, b: `${fmtNum(dataPages)} data ${e.pageWord}s` },
    pool: { a: `${fmtNum(requests)} requests`, b: `${fmtNum(hits)} hit · ${fmtNum(misses)} miss` },
    storage: {
      a: `${fmtNum(misses)} reads · ${fmtBytes(misses * e.page)}`,
      b: logBytes ? `${fmtNum(dirtied)} dirty · ${fmtBytes(logBytes)} log` : `${fmtNum(dirtied)} dirty · no log`,
    },
  };

  return { ann, requests, hits, misses, rowsOut, wireBytes, logBytes, dirtied, height, rowsPerPage };
}

/* --------------------------------------------------------------- drawing */

const ROW_H = 56;
const ROW_GAP = 7;
const LW = 470; // layer box width
const COL_W = 62; // manager column width
const HEAD_H = 26;

type Sel = { kind: 'layer'; id: LayerId } | { kind: 'mgr'; id: MgrId };

export default function EngineComponentIndex() {
  const [engineId, setEngineId] = useState<EngineId>('pg');
  const [stmtId, setStmtId] = useState<StmtId>('point');
  const [fpi, setFpi] = useState(false);
  const [sel, setSel] = useState<Sel>({ kind: 'layer', id: 'optimizer' });
  const [ref, width] = useSize(820);
  const tip = useTip();

  const engine = ENGINES.find((e) => e.id === engineId)!;
  const stmt = STMTS.find((s) => s.id === stmtId)!;
  const m = useMemo(() => model(engine, stmtId, fpi), [engine, stmtId, fpi]);

  const present = LAYERS.filter((l) => l.byEngine[engineId].state === 'present').length;
  const merged = LAYERS.filter((l) => l.byEngine[engineId].state === 'merged').length;

  const svgW = Math.max(width, LW + 12 + MGRS.length * COL_W + 8);
  const mx = LW + 12;
  const height = HEAD_H + LAYERS.length * (ROW_H + ROW_GAP);
  const rowY = (i: number) => HEAD_H + i * (ROW_H + ROW_GAP);

  const selLayer = sel.kind === 'layer' ? LAYERS.find((l) => l.id === sel.id)! : null;
  const selMgr = sel.kind === 'mgr' ? MGRS.find((g) => g.id === sel.id)! : null;

  const detail = selLayer ? (
    (() => {
      const v = selLayer.byEngine[engineId];
      const touch = selLayer.mgrs.map((id) => MGRS.find((g) => g.id === id)!.label);
      return (
        <>
          <strong>
            {v.name} — {engine.label}
            {v.state === 'absent' ? ' (absent)' : v.state === 'merged' ? ' (merged)' : ''}
          </strong>{' '}
          {v.detail}{' '}
          {v.state === 'present' ? (
            <>
              <strong>Knobs:</strong> <code>{v.knobs}</code>.{' '}
            </>
          ) : null}
          {touch.length ? (
            <>
              <strong>Cut across by:</strong> {touch.join(', ')}.{' '}
            </>
          ) : (
            <>
              <strong>No cross-cutting manager touches it</strong> — nothing here is transactional.{' '}
            </>
          )}
          <strong>Owned by:</strong> <a href={selLayer.href}>{selLayer.module}</a>.
        </>
      );
    })()
  ) : (
    <>
      <strong>{selMgr!.label}</strong> {selMgr!.detail} <strong>Owned by:</strong>{' '}
      <a href={selMgr!.href}>{selMgr!.module}</a>.
    </>
  );

  return (
    <VizPanel
      title="The engine, layer by layer"
      subtitle={
        <>
          Click any layer, or a manager column, to open it. Counts come from one stated model — a
          2,000,000-row <code>orders</code> table, 120-byte rows, a week out of a year matched, 4,200 distinct
          customers — not from a benchmark.
        </>
      }
      controls={
        <>
          <Segmented
            label="Engine"
            value={engineId}
            onChange={(v) => setEngineId(v)}
            options={ENGINES.map((e) => ({ value: e.id, label: e.short }))}
          />
          <Choice
            label="Statement"
            value={stmtId}
            onChange={(v) => setStmtId(v)}
            options={STMTS.map((s) => ({ value: s.id, label: s.label }))}
          />
          <Check
            label="First write to the page since the checkpoint"
            checked={fpi}
            onChange={setFpi}
          />
        </>
      }
      legend={
        <Legend
          items={[
            { label: 'solid — layer present', color: 'var(--viz-ink-2)', shape: 'line' },
            { label: 'tinted — merged into another layer', color: 'var(--viz-neutral)' },
            { label: 'dashed — absent in this engine', color: 'var(--viz-stale)', shape: 'line' },
            ...MGRS.map((g) => ({ label: `${g.glyph} — ${g.label}`, color: g.color })),
          ]}
        />
      }
      stats={
        <Stats
          items={[
            { label: 'Layers present', value: `${present} of ${LAYERS.length}`, hint: `${merged} merged into another layer` },
            { label: `${engine.pageWord}s requested`, value: fmtNum(m.requests), hint: 'Buffer-pool / block-cache lookups' },
            {
              label: 'Read from storage',
              value: fmtBytes(m.misses * engine.page),
              hint: `${fmtNum(m.misses)} misses × ${fmtBytes(engine.page)} · assumes a ${Math.round(
                (stmtId === 'range' ? HIT_RATE_SCAN : HIT_RATE_HOT) * 100,
              )}% hit rate`,
            },
            { label: 'Rows to caller', value: fmtNum(m.rowsOut) },
            {
              label: 'Bytes on the wire',
              value: engine.inProcess ? 'in-process' : fmtBytes(m.wireBytes),
              hint: engine.inProcess ? 'No socket: results are returned by pointer' : 'Result set plus protocol framing',
            },
            {
              label: 'Log written',
              value: m.logBytes ? fmtBytes(m.logBytes) : '0 B',
              hint: LOG_HINT[engineId](fpi),
            },
          ]}
        />
      }
      note={<Note>{detail}</Note>}
      table={
        <>
          <table className="viz-table">
            <caption>{engine.label} · {stmt.label}: {engineId === 'rocksdb' ? stmt.kv : stmt.sql}</caption>
            <thead>
              <tr>
                <th>Layer</th>
                <th>In this engine</th>
                <th>State</th>
                <th>Knobs</th>
                <th>In</th>
                <th>Out</th>
                <th>Managers</th>
              </tr>
            </thead>
            <tbody>
              {LAYERS.map((l) => {
                const v = l.byEngine[engineId];
                return (
                  <tr key={l.id}>
                    <td>{l.generic}</td>
                    <td>{v.name}</td>
                    <td>{v.state}</td>
                    <td>{v.knobs}</td>
                    <td>{m.ann[l.id].a}</td>
                    <td>{m.ann[l.id].b}</td>
                    <td>{l.mgrs.map((g) => MGRS.find((x) => x.id === g)!.glyph).join(' ') || '—'}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
          <table className="viz-table">
            <caption>The same statement across all four engines</caption>
            <thead>
              <tr>
                <th>Engine</th>
                <th>Page / block</th>
                <th>Layers present</th>
                <th>Pages requested</th>
                <th>Misses</th>
                <th>Bytes read</th>
                <th>Log written</th>
              </tr>
            </thead>
            <tbody>
              {ENGINES.map((e) => {
                const mm = model(e, stmtId, fpi);
                const p = LAYERS.filter((l) => l.byEngine[e.id].state === 'present').length;
                return (
                  <tr key={e.id}>
                    <td>{e.label}</td>
                    <td>{fmtBytes(e.page)}</td>
                    <td>{p}</td>
                    <td>{fmtNum(mm.requests)}</td>
                    <td>{fmtNum(mm.misses)}</td>
                    <td>{fmtBytes(mm.misses * e.page)}</td>
                    <td>{mm.logBytes ? fmtBytes(mm.logBytes) : '0 B'}</td>
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
          <svg
            width={svgW}
            height={height}
            role="img"
            aria-label={`Layers of ${engine.label} and the managers that cut across them`}
          >
            {/* manager column headers */}
            {MGRS.map((g, k) => {
              const x = mx + k * COL_W;
              const on = sel.kind === 'mgr' && sel.id === g.id;
              return (
                <g
                  key={g.id}
                  onClick={() => setSel({ kind: 'mgr', id: g.id })}
                  onKeyDown={(ev) => {
                    if (ev.key === 'Enter' || ev.key === ' ') setSel({ kind: 'mgr', id: g.id });
                  }}
                  style={{ cursor: 'pointer' }}
                  {...tip(
                    <>
                      <strong>{g.label}</strong>
                      <br />
                      Click to open. Marked rows are the layers it touches.
                    </>,
                  )}
                >
                  <rect x={x} y={0} width={COL_W - 6} height={HEAD_H - 6} rx={5} fill={g.color} opacity={on ? 1 : 0.22} />
                  <text
                    x={x + (COL_W - 6) / 2}
                    y={HEAD_H - 12}
                    textAnchor="middle"
                    fill="var(--viz-ink)"
                    fontWeight={600}
                  >
                    {g.glyph} {g.label.split(' ')[0].slice(0, 3)}
                  </text>
                </g>
              );
            })}

            {LAYERS.map((l, i) => {
              const v = l.byEngine[engineId];
              const y = rowY(i);
              const on = sel.kind === 'layer' && sel.id === l.id;
              const absent = v.state === 'absent';
              const mergedRow = v.state === 'merged';
              const a = m.ann[l.id];
              return (
                <g key={l.id}>
                  <g
                    onClick={() => setSel({ kind: 'layer', id: l.id })}
                    onKeyDown={(ev) => {
                      if (ev.key === 'Enter' || ev.key === ' ') setSel({ kind: 'layer', id: l.id });
                    }}
                    style={{ cursor: 'pointer' }}
                    {...tip(
                      <>
                        <strong>{v.name}</strong>
                        <br />
                        {l.generic} — {v.sub}
                      </>,
                    )}
                  >
                    <rect
                      x={0}
                      y={y}
                      width={LW}
                      height={ROW_H}
                      rx={8}
                      fill={absent ? 'none' : mergedRow ? 'var(--viz-neutral)' : 'var(--viz-plane)'}
                      stroke={on ? 'var(--viz-1)' : absent ? 'var(--viz-axis)' : 'var(--viz-border)'}
                      strokeWidth={on ? 2 : 1}
                      strokeDasharray={absent || mergedRow ? '5 4' : undefined}
                    />
                    <text
                      x={12}
                      y={y + 22}
                      fill={absent ? 'var(--viz-ink-muted)' : 'var(--viz-ink)'}
                      fontWeight={600}
                    >
                      {v.name}
                    </text>
                    <text x={12} y={y + 40} fill="var(--viz-ink-muted)">
                      {v.sub}
                    </text>
                    <text x={LW - 12} y={y + 22} textAnchor="end" fill="var(--viz-ink-2)">
                      {a.a}
                    </text>
                    <text x={LW - 12} y={y + 40} textAnchor="end" fill="var(--viz-ink-2)">
                      {a.b}
                    </text>
                  </g>

                  {MGRS.map((g, k) => {
                    const touches = l.mgrs.includes(g.id) && !absent;
                    const x = mx + k * COL_W;
                    const dim = sel.kind === 'mgr' && sel.id !== g.id;
                    return (
                      <g key={g.id}>
                        <rect
                          x={x}
                          y={y}
                          width={COL_W - 6}
                          height={ROW_H}
                          rx={6}
                          fill={touches ? g.color : 'none'}
                          opacity={touches ? (dim ? 0.14 : 0.5) : 1}
                          stroke={touches ? 'none' : 'var(--viz-grid)'}
                          strokeDasharray={touches ? undefined : '3 4'}
                        />
                        {touches ? (
                          <text
                            x={x + (COL_W - 6) / 2}
                            y={y + ROW_H / 2 + 5}
                            textAnchor="middle"
                            fill="var(--viz-ink)"
                            fontWeight={600}
                          >
                            {g.glyph}
                          </text>
                        ) : null}
                      </g>
                    );
                  })}
                </g>
              );
            })}
          </svg>
        </TooltipHost>
      </div>
    </VizPanel>
  );
}
