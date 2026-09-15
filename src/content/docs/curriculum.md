---
title: The full curriculum
description: Every part, module and page on this site — 12 parts, 104 modules, 716 pages.
tableOfContents: false
---

12 parts · 104 modules · 716 pages.
**89** are written so far; the rest are specified and queued.

## Part 1 — Foundations: Machines, Data and the Shape of an Engine

The physics of storage and durability, the data models and SQL a beginner needs, and a map of a database engine including a WAL primer so every later module has its anchors.

### 1.1 Start Here: Routes, Depth and How to Read a Mechanism

*foundations* — Ninety-odd modules and several hundred subtopics presented as one front-to-back sequence serve nobody by default: an application engineer who needs index selection, EXPLAIN and isolation should not be told to read the FTL and epoch reclamation first, an SRE who arrives mid-incident needs a symptom-shaped entry point rather than chapter one, and the distributed track needs exactly one module from Part 5 rather than all thirteen. A short opener that states the routes, the depth legend and the predict-then-run method keeps the linear order as the default rather than the only option — and every route, count and time estimate on this page is generated from the curriculum's own module and prerequisite data, so no route it prints can contradict the dependency graph it draws.

- [Three routes through the same modules](../p01-foundations/start-here-routes-and-method/01-three-routes-through-the-same-modules/)
- [Routes for operators, data engineers and platform engineers](../p01-foundations/start-here-routes-and-method/02-routes-for-operators-data-engineers-and-platform-engineers/)
- [Depth levels, prerequisites and stopping points](../p01-foundations/start-here-routes-and-method/03-depth-levels-prerequisites-and-stopping-points/)
- [How to use the simulations](../p01-foundations/start-here-routes-and-method/04-how-to-use-the-simulations/)
- [Five questions to ask of any data system](../p01-foundations/start-here-routes-and-method/05-five-questions-to-ask-of-any-data-system/)

### 1.2 Data Models and a Hands-On SQL Primer

*foundations* — A true beginner needs the relational model, a way to turn a domain into tables, working SQL and a map of the non-relational models before relational algebra, query plans, isolation anomalies or per-store modeling can mean anything; this module gives every later module its vocabulary and a live sandbox to test claims against.

- [Why a DBMS instead of files](../p01-foundations/data-models-and-sql-primer/01-why-a-dbms-instead-of-files/)
- [The relational model precisely: relations, keys, constraints and NULLs](../p01-foundations/data-models-and-sql-primer/02-the-relational-model-precisely-relations-keys-constraints-an/)
- [Entities, relationships and cardinality](../p01-foundations/data-models-and-sql-primer/03-entities-relationships-and-cardinality/)
- [Mapping an ER model to tables](../p01-foundations/data-models-and-sql-primer/04-mapping-an-er-model-to-tables/)
- [SQL primer: DDL, DML and single-table queries](../p01-foundations/data-models-and-sql-primer/05-sql-primer-ddl-dml-and-single-table-queries/)
- [SQL primer: joins, GROUP BY, subqueries and set operations](../p01-foundations/data-models-and-sql-primer/06-sql-primer-joins-group-by-subqueries-and-set-operations/)
- [Document, key-value, wide-column and graph models](../p01-foundations/data-models-and-sql-primer/07-document-key-value-wide-column-and-graph-models/)
- [OLTP vs OLAP vs HTAP and the systems landscape](../p01-foundations/data-models-and-sql-primer/08-oltp-vs-olap-vs-htap-and-the-systems-landscape/)

### 1.3 SQL Semantics, Types and Schema Design

*foundations* — Once a beginner can write SQL and draw a schema, the next failures are silent rather than loud — NULL and fan-out traps that return plausible wrong numbers, type choices that lose money or shift timestamps across a DST boundary, and schemas whose redundancy makes a single update rewrite a cross product — and every later module on partitioning by time, ledgers, indexes and query plans assumes these are already right.

- [SQL semantics engineers get wrong](../p01-foundations/sql-semantics-types-and-schema-design/01-sql-semantics-engineers-get-wrong/)
- [Temporal types: instants, civil time and DST](../p01-foundations/sql-semantics-types-and-schema-design/02-temporal-types-instants-civil-time-and-dst/)
- [Numeric types: exact money vs float drift](../p01-foundations/sql-semantics-types-and-schema-design/03-numeric-types-exact-money-vs-float-drift/)
- [Text, encoding and collation](../p01-foundations/sql-semantics-types-and-schema-design/04-text-encoding-and-collation/)
- [Normalization, functional dependencies and schema design](../p01-foundations/sql-semantics-types-and-schema-design/05-normalization-functional-dependencies-and-schema-design/)
- [Optional deep dive: multivalued dependencies, 4NF and the chase](../p01-foundations/sql-semantics-types-and-schema-design/06-optional-deep-dive-multivalued-dependencies-4nf-and-the-chas/)

### 1.4 Hardware, the OS and What Durable Means

*foundations* — Now that tables, rows and queries are on the table, this module supplies the machine-level facts the rest of the site keeps referring back to: every storage-engine decision is a response to latency ladders, NAND erase blocks, page-cache behavior and the exact promise fsync does or does not make.

- [Memory hierarchy and latency numbers](../p01-foundations/hardware-os-durability/01-memory-hierarchy-and-latency-numbers/)
- [HDD and SSD internals: seeks, NAND and the FTL](../p01-foundations/hardware-os-durability/02-hdd-and-ssd-internals-seeks-nand-and-the-ftl/)
- [Storage abstractions: files, block devices and object stores](../p01-foundations/hardware-os-durability/03-storage-abstractions-files-block-devices-and-object-stores/)
- [Encoding data: bytes, endianness and serialization](../p01-foundations/hardware-os-durability/04-encoding-data-bytes-endianness-and-serialization/)
- [The OS I/O path: syscalls, page cache and mmap](../p01-foundations/hardware-os-durability/05-the-os-i-o-path-syscalls-page-cache-and-mmap/)
- [fsync, write barriers and the durability contract](../p01-foundations/hardware-os-durability/06-fsync-write-barriers-and-the-durability-contract/)
- [Atomicity of writes and torn pages](../p01-foundations/hardware-os-durability/07-atomicity-of-writes-and-torn-pages/)
- [Filesystem crash consistency: journaling, rename and the safe-replace protocol](../p01-foundations/hardware-os-durability/08-filesystem-crash-consistency-journaling-rename-and-the-safe-/)
- [Direct I/O and io_uring: alignment, submission and completion](../p01-foundations/hardware-os-durability/09-direct-i-o-and-io-uring-alignment-submission-and-completion/)

### 1.5 Anatomy of a Database Engine

*foundations* — A component map, a schema-to-files mapping, one traced read, a first look at transactions and versions, and a WAL primer give every later module a place to plug in; the module deliberately stops at 'enough to continue' after the WAL primer — the page-level write path belongs to Pages, Tuples and Heap Files and the concurrency primitives to The Buffer Pool, where the structures they touch are defined.

- [Component map: parser, planner, executor, storage manager, log](../p01-foundations/anatomy-of-a-database-engine/01-component-map-parser-planner-executor-storage-manager-log/)
- [Logical schema vs physical storage](../p01-foundations/anatomy-of-a-database-engine/02-logical-schema-vs-physical-storage/)
- [Row, column and key-value physical models](../p01-foundations/anatomy-of-a-database-engine/03-row-column-and-key-value-physical-models/)
- [Life of a query: from SQL text to rows](../p01-foundations/anatomy-of-a-database-engine/04-life-of-a-query-from-sql-text-to-rows/)
- [Transactions at 10,000 feet](../p01-foundations/anatomy-of-a-database-engine/05-transactions-at-10-000-feet/)
- [Versions, snapshots and the garbage horizon](../p01-foundations/anatomy-of-a-database-engine/06-versions-snapshots-and-the-garbage-horizon/)
- [WAL primer: log records, LSNs and log-before-page](../p01-foundations/anatomy-of-a-database-engine/07-wal-primer-log-records-lsns-and-log-before-page/)

## Part 2 — Storage Engines: Pages, Trees, Logs and Columns

The single-node storage substrate in depth: page layout, buffer management, B-trees and LSM trees, the full family of access methods, practical index selection, and columnar analytical storage.

### 2.1 Pages, Tuples and Heap Files

*core* — The slotted page and heap file are the unit of everything above them; the byte layout of a tuple decides what every page, index and log record above it has to move around.

- [Page anatomy and the slotted page](../p02-storage-engines/pages-tuples-heap-files/01-page-anatomy-and-the-slotted-page/)
- [Tuple layout: headers, null bitmaps, alignment and varlena](../p02-storage-engines/pages-tuples-heap-files/02-tuple-layout-headers-null-bitmaps-alignment-and-varlena/)
- [Heap file organization and free space management](../p02-storage-engines/pages-tuples-heap-files/03-heap-file-organization-and-free-space-management/)
- [HOT updates, redirect line pointers and opportunistic page pruning](../p02-storage-engines/pages-tuples-heap-files/04-hot-updates-redirect-line-pointers-and-opportunistic-page-pr/)
- [Large values: TOAST and overflow pages](../p02-storage-engines/pages-tuples-heap-files/05-large-values-toast-and-overflow-pages/)
- [Checksums, corruption detection and on-disk format evolution](../p02-storage-engines/pages-tuples-heap-files/06-checksums-corruption-detection-and-on-disk-format-evolution/)
- [Catalog storage, the bootstrap problem and cache invalidation](../p02-storage-engines/pages-tuples-heap-files/07-catalog-storage-the-bootstrap-problem-and-cache-invalidation/)

### 2.2 The Buffer Pool

*core* — The buffer pool decides which pages hit disk, when dirty pages may leave memory, and how long a restarted or failed-over system runs cold.

- [Buffer pool structure: frames, page table and pins](../p02-storage-engines/buffer-pool/01-buffer-pool-structure-frames-page-table-and-pins/)
- [Cache warm-up: cold pools, dump/load and prewarming](../p02-storage-engines/buffer-pool/02-cache-warm-up-cold-pools-dump-load-and-prewarming/)
- [Replacement policies: LRU, CLOCK, LRU-K, 2Q and ARC](../p02-storage-engines/buffer-pool/03-replacement-policies-lru-clock-lru-k-2q-and-arc/)
- [How good can a cache policy be? OPT, competitive ratios and the stack property](../p02-storage-engines/buffer-pool/04-how-good-can-a-cache-policy-be-opt-competitive-ratios-and-th/)
- [Scan resistance: sequential flooding, ring buffers and midpoint insertion](../p02-storage-engines/buffer-pool/05-scan-resistance-sequential-flooding-ring-buffers-and-midpoin/)
- [Dirty pages, write-back and the WAL-before-data rule](../p02-storage-engines/buffer-pool/06-dirty-pages-write-back-and-the-wal-before-data-rule/)
- [Prefetching and asynchronous reads](../p02-storage-engines/buffer-pool/07-prefetching-and-asynchronous-reads/)

### 2.3 Buffer Pool Memory, mmap and Contention

*advanced* — Past the caching policy, the pool becomes a memory-budgeting, kernel-interface and scalability problem: where the rest of the server's RAM goes, why mmap cannot replace a buffer pool, what engines that delete the page table gain, and the hot page no amount of partitioning fixes.

- [Memory beyond the pool: shared, per-backend and context allocation](../p02-storage-engines/buffer-pool-memory-mmap-and-contention/01-memory-beyond-the-pool-shared-per-backend-and-context-alloca/)
- [Buffer pool vs OS page cache vs mmap](../p02-storage-engines/buffer-pool-memory-mmap-and-contention/02-buffer-pool-vs-os-page-cache-vs-mmap/)
- [Beyond the page table: pointer swizzling, in-memory engines and anti-caching](../p02-storage-engines/buffer-pool-memory-mmap-and-contention/03-beyond-the-page-table-pointer-swizzling-in-memory-engines-an/)
- [Latching the buffer pool and hot-page contention](../p02-storage-engines/buffer-pool-memory-mmap-and-contention/04-latching-the-buffer-pool-and-hot-page-contention/)

### 2.4 B-Trees: Structure, Maintenance and Key Encoding

*core* — The B+tree is the default index of every OLTP engine, and its shape, split behaviour, key encoding and the external-memory bounds behind them decide real lookup and range-scan cost.

- [B+tree structure, invariants and search](../p02-storage-engines/b-trees-and-variants/01-b-tree-structure-invariants-and-search/)
- [Insert, split, delete and merge](../p02-storage-engines/b-trees-and-variants/02-insert-split-delete-and-merge/)
- [Crash safety of structure modifications: atomic splits and incomplete-split repair](../p02-storage-engines/b-trees-and-variants/03-crash-safety-of-structure-modifications-atomic-splits-and-in/)
- [Fill factor, fragmentation and splits under load](../p02-storage-engines/b-trees-and-variants/04-fill-factor-fragmentation-and-splits-under-load/)
- [Bulk loading and index construction](../p02-storage-engines/b-trees-and-variants/05-bulk-loading-and-index-construction/)
- [Key layout, prefix compression and suffix truncation](../p02-storage-engines/b-trees-and-variants/06-key-layout-prefix-compression-and-suffix-truncation/)
- [Collation, Unicode and byte-comparable key encoding](../p02-storage-engines/b-trees-and-variants/07-collation-unicode-and-byte-comparable-key-encoding/)
- [Why fanout B: the external-memory model, optimality bounds and cache-oblivious layouts](../p02-storage-engines/b-trees-and-variants/08-why-fanout-b-the-external-memory-model-optimality-bounds-and/)

### 2.5 Log-Structured Merge Trees

*core* — LSM trees power RocksDB, Cassandra, and most distributed stores; the write path, the sorted-run file format and the filtered read path explain both their speed and where that speed quietly disappears.

- [Append-only logs and Bitcask hash indexes](../p02-storage-engines/lsm-trees/01-append-only-logs-and-bitcask-hash-indexes/)
- [Write path: WAL, memtable, flush and write stalls](../p02-storage-engines/lsm-trees/02-write-path-wal-memtable-flush-and-write-stalls/)
- [Skip lists: probabilistic balance without rebalancing](../p02-storage-engines/lsm-trees/03-skip-lists-probabilistic-balance-without-rebalancing/)
- [Writes that must read first: constraints, secondary indexes and Merge](../p02-storage-engines/lsm-trees/04-writes-that-must-read-first-constraints-secondary-indexes-an/)
- [SSTable format: blocks, restart points, index, filters, footer](../p02-storage-engines/lsm-trees/05-sstable-format-blocks-restart-points-index-filters-footer/)
- [Read path, Bloom filters and the block cache](../p02-storage-engines/lsm-trees/06-read-path-bloom-filters-and-the-block-cache/)
- [Deletes, tombstones, TTL and snapshots](../p02-storage-engines/lsm-trees/07-deletes-tombstones-ttl-and-snapshots/)
- [Beyond Bloom: cuckoo, quotient, xor and ribbon filters](../p02-storage-engines/lsm-trees/08-beyond-bloom-cuckoo-quotient-xor-and-ribbon-filters/)

### 2.6 LSM Compaction, Amplification and Crash Safety

*core* — Everything an LSM costs in practice — write amplification, space bloat, stalls, recovery time and bulk-load behavior — is decided by the compaction policy and the version/manifest machinery that installs its output.

- [Why compaction exists: the three amplifications and the RUM trade-off](../p02-storage-engines/lsm-compaction-and-maintenance/01-why-compaction-exists-the-three-amplifications-and-the-rum-t/)
- [Compaction strategies: leveled, tiered, TWCS and lazy leveling](../p02-storage-engines/lsm-compaction-and-maintenance/02-compaction-strategies-leveled-tiered-twcs-and-lazy-leveling/)
- [Key-value separation and blob storage](../p02-storage-engines/lsm-compaction-and-maintenance/03-key-value-separation-and-blob-storage/)
- [Manifest, versions and crash safety](../p02-storage-engines/lsm-compaction-and-maintenance/04-manifest-versions-and-crash-safety/)
- [Bulk loading and external SSTable ingestion](../p02-storage-engines/lsm-compaction-and-maintenance/05-bulk-loading-and-external-sstable-ingestion/)

### 2.7 B-Tree Concurrency and Write-Optimized Variants

*advanced* — Once many cores hit the same index, correctness and throughput come from latch protocols, the memory model beneath them and safe reclamation — and the same pressure produced the latch-free, write-optimized and copy-on-write descendants of the B+tree, which only make sense once you have seen both a B+tree and an LSM and can price one against the other; this module is the first 'enough for most engineers, stop here' boundary in Part 2, so a reader who stops after it still has the complete core storage spine, while everything here is what you need before reading engine source or designing an index yourself.

- [B-tree concurrency: latch crabbing and Lehman-Yao](../p02-storage-engines/b-tree-concurrency-and-write-optimized-variants/01-b-tree-concurrency-latch-crabbing-and-lehman-yao/)
- [Memory models, atomics and what makes optimistic latching correct](../p02-storage-engines/b-tree-concurrency-and-write-optimized-variants/02-memory-models-atomics-and-what-makes-optimistic-latching-cor/)
- [Bw-tree: delta chains and the mapping table](../p02-storage-engines/b-tree-concurrency-and-write-optimized-variants/03-bw-tree-delta-chains-and-the-mapping-table/)
- [Memory reclamation for latch-free structures: epochs, hazard pointers, RCU](../p02-storage-engines/b-tree-concurrency-and-write-optimized-variants/04-memory-reclamation-for-latch-free-structures-epochs-hazard-p/)
- [Bε-trees, fractal trees and copy-on-write B-trees](../p02-storage-engines/b-tree-concurrency-and-write-optimized-variants/05-b-trees-fractal-trees-and-copy-on-write-b-trees/)
- [Persistence by path copying: fat nodes, versions and why snapshots are cheap](../p02-storage-engines/b-tree-concurrency-and-write-optimized-variants/06-persistence-by-path-copying-fat-nodes-versions-and-why-snaps/)

### 2.8 Indexes Beyond the B-Tree

*core* — Hash, bitmap, spatial, inverted and range-summary access methods each serve predicates a B-tree cannot, the hash function underneath them decides whether buckets stay short or collapse into skew, and building any of them online is its own mechanism.

- [Hash indexes: extendible and linear hashing](../p02-storage-engines/indexes-beyond-the-b-tree/01-hash-indexes-extendible-and-linear-hashing/)
- [Hash functions inside the engine: universality, avalanche and adversarial keys](../p02-storage-engines/indexes-beyond-the-b-tree/02-hash-functions-inside-the-engine-universality-avalanche-and-/)
- [Bitmap indexes and bitmap heap scans](../p02-storage-engines/indexes-beyond-the-b-tree/03-bitmap-indexes-and-bitmap-heap-scans/)
- [BRIN, zone maps and min/max data skipping](../p02-storage-engines/indexes-beyond-the-b-tree/04-brin-zone-maps-and-min-max-data-skipping/)
- [GiST and R-trees for spatial and range data](../p02-storage-engines/indexes-beyond-the-b-tree/05-gist-and-r-trees-for-spatial-and-range-data/)
- [GIN inverted indexes and posting lists](../p02-storage-engines/indexes-beyond-the-b-tree/06-gin-inverted-indexes-and-posting-lists/)
- [Full-text indexes: term dictionaries, postings and skip lists](../p02-storage-engines/indexes-beyond-the-b-tree/07-full-text-indexes-term-dictionaries-postings-and-skip-lists/)
- [Top-k without scanning everything: the threshold algorithm, WAND and block-max](../p02-storage-engines/indexes-beyond-the-b-tree/08-top-k-without-scanning-everything-the-threshold-algorithm-wa/)
- [Building indexes online: CREATE INDEX CONCURRENTLY internals](../p02-storage-engines/indexes-beyond-the-b-tree/09-building-indexes-online-create-index-concurrently-internals/)

### 2.9 Tries, Learned and Vector Indexes

*advanced* — Trie- and radix-shaped structures, succinct encodings, learned models and approximate-nearest-neighbour graphs are where access-method design has moved, and each one's real cost shows up after the first build — in updates, drift and rebuilds.

- SP-GiST, tries and adaptive radix trees <span class="pending">soon</span>
- Succinct and learned indexes: rank/select, Elias-Fano and RMI <span class="pending">soon</span>
- Why high dimensions break indexes: distance concentration and intrinsic dimension <span class="pending">soon</span>
- Vector indexes: IVF, HNSW and quantization <span class="pending">soon</span>
- How churn degrades a vector index: tombstones, merges and centroid drift <span class="pending">soon</span>

### 2.10 Index Selection and Query-Aware Indexing

*advanced* — Choosing composite, covering and partial indexes against real predicates, sort orders and write costs is the most common daily database decision, and getting it wrong is the leading cause of both slow queries and write-amplified, bloated tables; it sits at the end of Part 3, after the planner modules, because every judgement call here is read straight off the cost model and cardinality estimates taught in cost-based-optimization and off the real plans, buffer counts and workload inventories the learner has just learned to collect and read in explain-and-query-tuning.

- Selectivity, correlation and the decision to build an index <span class="pending">soon</span>
- Composite indexes and the leftmost-prefix rule <span class="pending">soon</span>
- Covering indexes and index-only scans <span class="pending">soon</span>
- Partial and expression indexes <span class="pending">soon</span>
- Pagination done right: keyset vs OFFSET <span class="pending">soon</span>
- Write cost, HOT and the indexes-per-update meter <span class="pending">soon</span>
- Choosing the index for JSONB, arrays and text search <span class="pending">soon</span>
- Indexing for hybrid search: which indexes the pipeline needs <span class="pending">soon</span>
- The list endpoint: optional filters, sort options and one index set <span class="pending">soon</span>
- Designing the index set for a workload <span class="pending">soon</span>

### 2.11 Column Stores, Compression and Columnar Formats

*advanced* — Analytical storage wins not by clever execution but by what it refuses to read: columnar layout touches only projected columns, lightweight encodings shrink them enough to scan in cache and evaluate predicates without decoding, entropy sets the floor those encoders are chasing and explains why ordering beats codec tuning, block compressors and open formats like Parquet/ORC/Arrow — plus the post-2022 successors aimed at random access, wide schemas and fast decode — decide what a reader must fetch over a network, semi-structured JSON either shreds into typed sub-columns or costs 10-100x to scan, and sort keys plus min/max metadata skip most blocks outright — these are the primitives every analytical engine and table format in the next module is built out of.

- Columnar layouts: NSM, DSM, PAX and row groups <span class="pending">soon</span>
- Lightweight encodings: RLE, dictionary, delta, bit-packing, FOR and friends <span class="pending">soon</span>
- Entropy and the limits of compression: what ratio should you expect? <span class="pending">soon</span>
- Block compression: LZ4, ZSTD, trained dictionaries and hole punching <span class="pending">soon</span>
- Open formats: Parquet, ORC, Arrow and nested data (Dremel) <span class="pending">soon</span>
- After Parquet: Lance, Vortex and Nimble <span class="pending">soon</span>
- Shredding JSON into columns: VARIANT, dynamic columns and variant Parquet <span class="pending">soon</span>
- Sort keys, clustering and multi-dimensional data skipping <span class="pending">soon</span>

### 2.12 Analytical Storage Engines

*advanced* — Columnar layout and encodings are inert until an engine organizes them into a live table: MergeTree parts and granules, real-time segments handed to deep storage, delta-plus-main stores that keep a columnar table fresh under OLTP writes, the newer move of putting a second execution engine and columnar format inside the OLTP server itself, and the mirror-image move the cloud vendors shipped after 2022 — a transactional row store stitched in under a columnar platform — the concrete answers to "how do rows arrive, become queryable and get reorganized" that separate ClickHouse from Pinot from an HTAP engine from a Postgres with an embedded analytical engine bolted on from a warehouse selling itself as your operational database (the lakehouse file-and-catalog answer comes later, in Lakehouse Table Formats, Catalogs and Change Feeds, once object-store semantics and optimistic concurrency are established).

- ClickHouse MergeTree internals: parts, granules, marks and merges <span class="pending">soon</span>
- Segment-based real-time OLAP: Druid, Pinot and upsert tables <span class="pending">soon</span>
- Hybrid row/column and HTAP storage: delta store plus columnar main <span class="pending">soon</span>
- Analytics inside Postgres: embedded engines, columnar mirrors and Iceberg-backed tables <span class="pending">soon</span>
- OLTP inside the warehouse: hybrid tables and lakebases <span class="pending">soon</span>

### 2.13 Lakehouse Table Formats, Catalogs and Change Feeds

*expert* — A pile of immutable Parquet files in an object store becomes a table only when a metadata tree, a conditional pointer write and a catalog that arbitrates writers turn it into something with snapshots, deletes, schema evolution and concurrent commits — so this module belongs in Part 8 immediately after cloud-native-disaggregated-storage, whose object-store semantics (put-if-absent, compare-and-set, listing costs) it builds on directly, and after serializability-without-blocking, whose read-validate-write discipline is literally the commit protocol, and it runs before distributed-and-analytical-query so that distributed execution can assume table formats exist; from there it answers the five questions a lakehouse actually raises: how a commit is made atomic, what changes when the table is continuously upserted rather than appended to, who performs and guards the commit, whether the metadata belongs in files or in a database, and how a downstream consumer reads the table as a stream of changes instead of a snapshot.

- Lakehouse table formats: Iceberg, Delta and Hudi <span class="pending">soon</span>
- Mutable tables on object storage: LSM formats and primary-key tables <span class="pending">soon</span>
- Catalogs: the Iceberg REST protocol, Polaris, Unity, Glue and Nessie <span class="pending">soon</span>
- Branches, tags and write-audit-publish on the lake <span class="pending">soon</span>
- Where table metadata lives: manifest trees versus a catalog database <span class="pending">soon</span>
- Reading changes out of a lakehouse table: CDF, incremental scans and row lineage <span class="pending">soon</span>

## Part 3 — Query Processing: From SQL Text to Executed Plan

Parsing, rewriting, physical operators, cost-based optimization, execution engines with resource control, and the EXPLAIN literacy that ties them together.

### 3.1 From SQL Text to Logical Plan

*core* — Every rewrite that matters happens before a single cost is computed; a senior engineer must know what the front end resolves (and silently casts), which transformations the engine applies blindly, which genuinely need statistics, and which fences (views, CTEs, outer joins, volatile functions) stop them cold.

- Parsing and the AST <span class="pending">soon</span>
- Binding, name resolution and semantic analysis <span class="pending">soon</span>
- Relational algebra operators and the logical evaluation order <span class="pending">soon</span>
- Relational calculus, Codd's theorem and the limits of first-order queries <span class="pending">soon</span>
- Algebraic equivalences and rewrite rules <span class="pending">soon</span>
- Views, CTEs and materialization fences <span class="pending">soon</span>
- Rule pipelines: where heuristics end and cost begins <span class="pending">soon</span>

### 3.2 Physical Operators: Scans, Joins, Sorting and Aggregation

*core* — Every plan node is one of a dozen algorithms whose I/O and memory behavior you must be able to predict.

- Access paths: sequential, index, index-only and bitmap scans <span class="pending">soon</span>
- Nested loop and index nested loop join <span class="pending">soon</span>
- External sort and sort-merge join <span class="pending">soon</span>
- Hash join: build, probe, partitioning and spilling <span class="pending">soon</span>
- Aggregation, DISTINCT and grouping strategies <span class="pending">soon</span>
- Window function execution <span class="pending">soon</span>
- Aggregates as algebra: monoids, inverses and the cost of a window <span class="pending">soon</span>
- Semi, anti joins and set operations <span class="pending">soon</span>
- Write operators: ModifyTable, junk TIDs and Halloween protection <span class="pending">soon</span>
- Choosing among operators: cost intuition <span class="pending">soon</span>

### 3.3 Join Algorithms at the Limit: Worst-Case-Optimal, Cache-Aware and Distributed

*advanced* — Binary hash and merge joins run into three separate ceilings — a provable complexity bound on output size, the memory hierarchy, and the network between shards — and the algorithms that break each one are where modern join execution is headed.

- Join complexity: the AGM bound and worst-case-optimal joins <span class="pending">soon</span>
- Acyclic queries: GYO, semi-join reduction and Yannakakis's algorithm <span class="pending">soon</span>
- In-memory parallel joins: radix partitioning and cache-aware hashing <span class="pending">soon</span>
- Distributed join strategies: co-located, broadcast, shuffle and semi-join reduction <span class="pending">soon</span>

### 3.4 Cost-Based Optimization

*advanced* — Cardinality estimation and join-order search are where plans go wrong, and knowing why is the difference between guessing and fixing.

- Cost models: I/O, CPU and their calibration <span class="pending">soon</span>
- Statistics: histograms, MCVs, ndistinct and sketches <span class="pending">soon</span>
- Single-column cardinality estimation <span class="pending">soon</span>
- Plan stability, hints and regressions <span class="pending">soon</span>
- Correlation, multi-column statistics and join estimation <span class="pending">soon</span>
- Bounding instead of guessing: q-error and pessimistic cardinality estimation <span class="pending">soon</span>
- Selinger bottom-up dynamic programming <span class="pending">soon</span>
- Cascades and Volcano top-down optimization <span class="pending">soon</span>
- Join ordering search space and heuristics <span class="pending">soon</span>
- Adaptive and learned optimization <span class="pending">soon</span>

### 3.5 Rewrite Legality, Decorrelation and Recursive Evaluation

*advanced* — Whether a correlated subquery becomes one free semi join or a per-row re-execution, and whether an aggressive rewrite still returns the same answer, is decided in this layer; a senior engineer needs the actual decision procedure for legality (containment, the chase, and the bag-, NULL- and incomplete-information traps that void textbook equivalences), the unnesting machinery that removes dependent joins, and the fixpoint theory behind recursion, magic sets and incremental maintenance. It is positioned fifth in Part 3, after physical-operators, join-algorithms-at-the-limit and cost-based-optimization, because the rewrites here are only meaningful once hash/semi/anti joins, Bloom-filter semi-join reduction and the planner's search space are already familiar — and it opens with the data-vs-combined-complexity frame that explains every blow-up in the module, so the application-engineer route in start-here-routes-and-method can skip it entirely and lose nothing it needs downstream.

- Data complexity vs combined complexity: cheap in the rows, expensive in the query <span class="pending">soon</span>
- Query equivalence, containment and the chase: when is a rewrite legal? <span class="pending">soon</span>
- Incomplete information: certain answers and what a NULL actually means <span class="pending">soon</span>
- Subquery unnesting and decorrelation <span class="pending">soon</span>
- Advanced rewrites: join elimination, aggregate pushdown, CSE and magic sets <span class="pending">soon</span>
- Row-level provenance: why is this row in my answer? <span class="pending">soon</span>
- Datalog and fixpoint evaluation: recursion the engine can reason about <span class="pending">soon</span>

### 3.6 Execution Engines, Vectorization and Parallelism

*advanced* — Operator models, per-tuple CPU cost, vectorization, parallelism, pruning, spill behaviour and plan caching decide throughput and tail latency, and they are the layer where a plan that looked cheap turns into hours of CPU and temp files.

- Volcano iterator model <span class="pending">soon</span>
- Expression evaluation, tuple deforming and the per-tuple CPU budget <span class="pending">soon</span>
- Vectorized execution <span class="pending">soon</span>
- Parallel query execution <span class="pending">soon</span>
- Partition pruning and partition-wise execution <span class="pending">soon</span>
- Spill shapes: what each operator writes to disk <span class="pending">soon</span>
- Prepared statements and plan caching <span class="pending">soon</span>

### 3.7 Morsel Parallelism, JIT Compilation and User Code

*advanced* — Push-based morsel scheduling and data-centric code generation are how modern engines spend CPU on your data instead of on dispatch, and user code — inlined, interpreted, sandboxed or across a network hop — is the one thing the optimizer cannot see inside, so it silently sets the plan, the parallelism and the per-row cost.

- Push-based and morsel-driven parallelism <span class="pending">soon</span>
- JIT compilation of queries <span class="pending">soon</span>
- UDFs, stored procedures and server-side logic <span class="pending">soon</span>
- Running user code in a sandbox: WASM, Arrow-batched and remote UDFs <span class="pending">soon</span>

### 3.8 Query Scheduling, Admission Control and Cancellation

*advanced* — Once queries outnumber cores and memory grants, the queue discipline and the abort path decide whose latency suffers and whether one runaway query takes down the server.

- Scheduling policy: FIFO, shortest-job-first, fair sharing and head-of-line blocking <span class="pending">soon</span>
- Query scheduling and admission control <span class="pending">soon</span>
- Feedback control in databases: setpoints, gain and oscillation <span class="pending">soon</span>
- Query cancellation and timeouts inside the engine <span class="pending">soon</span>

### 3.9 EXPLAIN Literacy and Query Tuning

*advanced* — Plan reading turns every earlier mechanism — statistics, operators, access paths and memory budgets — into a diagnosable symptom in the plan output, and a repeatable tuning workflow turns those symptoms into fixes that survive the next stats refresh or upgrade.

- Reading the plan tree: costs, rows, loops, buffers and time <span class="pending">soon</span>
- Scan-level diagnostics: Index Cond vs Filter, Heap Fetches and lossy bitmaps <span class="pending">soon</span>
- Spotting misestimates and their causes <span class="pending">soon</span>
- Spills, batches and memory symptoms in plans <span class="pending">soon</span>
- Join order and join method problems <span class="pending">soon</span>
- Common anti-patterns and rewrites <span class="pending">soon</span>
- Plans in other engines: MySQL, SQLite, DuckDB, ClickHouse <span class="pending">soon</span>
- The tuning workflow: find, reproduce, hypothesize, verify, guard <span class="pending">soon</span>

## Part 4 — Transactions, Concurrency Control and Recovery

Serializability and isolation levels precisely, locking, MVCC from semantics through vacuum, non-blocking serializability, and write-ahead logging with ARIES recovery.

### 4.1 Transactions, Serializability and Isolation Levels

*core* — The anomaly catalog and the engine isolation matrix are the vocabulary for every concurrency and distributed-consistency discussion that follows.

- ACID precisely <span class="pending">soon</span>
- Schedules, conflicts and serializability <span class="pending">soon</span>
- The anomaly catalog <span class="pending">soon</span>
- Isolation levels: ANSI, the Berenson critique and Adya's graphs <span class="pending">soon</span>
- Snapshot isolation semantics <span class="pending">soon</span>
- Read Committed up close: statement snapshots and re-checks <span class="pending">soon</span>
- Transaction boundaries in practice <span class="pending">soon</span>
- Multiversion serializability and one-copy serializability <span class="pending">soon</span>
- What real engines actually give <span class="pending">soon</span>

### 4.2 Lock-Based Concurrency Control

*advanced* — Strict 2PL, intention locks, gap locks and deadlock detection are what InnoDB, SQL Server and (for writes) PostgreSQL actually run; a senior engineer must be able to predict which rows and ranges a statement locks, why a queued ALTER TABLE freezes an entire service, and how to read a blocking chain at 3 a.m., because lock pile-ups are the single most common database production incident — note that PostgreSQL's habit of recording row locks inside the tuple header is only previewed here and is fully defined, together with version chains and MultiXacts, in mvcc-storage-vacuum-gc.

- Two-phase locking and strict 2PL <span class="pending">soon</span>
- Lock modes and compatibility matrices <span class="pending">soon</span>
- Granularity and intention locks <span class="pending">soon</span>
- Row-level locking on real engines: FOR UPDATE, SKIP LOCKED, NOWAIT <span class="pending">soon</span>
- Deadlocks: detection, prevention and victims <span class="pending">soon</span>
- Lock queues, DDL locks and blocking chains in production <span class="pending">soon</span>
- Phantoms, predicate locks, gap and next-key locks <span class="pending">soon</span>
- Commutativity as a lock mode: abstract locks, open nesting and logical undo <span class="pending">soon</span>
- Lock manager internals <span class="pending">soon</span>

### 4.3 MVCC: Semantics, Version Storage and Snapshots

*advanced* — Multiversioning is how modern engines read without blocking, and the version-storage design an engine picks decides read cost, write amplification, index bloat and whether it needs a vacuum at all.

- Multiversion fundamentals: version chains, visibility and snapshots <span class="pending">soon</span>
- Version storage: append-only heap, undo deltas, version stores and timestamped keys <span class="pending">soon</span>
- Swapping Postgres's heap: table access methods, zheap and OrioleDB <span class="pending">soon</span>
- PostgreSQL MVCC: xmin/xmax, snapshots, virtual xids and pg_xact <span class="pending">soon</span>
- Hint bits and reads that write <span class="pending">soon</span>
- InnoDB MVCC: undo logs, read views and purge <span class="pending">soon</span>
- Snapshot acquisition scalability: ProcArray, CSN and epochs <span class="pending">soon</span>

### 4.4 Garbage Collection: Vacuum, Freezing and Horizons

*advanced* — Every multiversion engine must reclaim old versions, and the horizon that pins them, the freezing race against a 32-bit counter and the tiny SLRUs behind both cause the bloat, stalls and write-refusal outages seniors must recognize on sight.

- Vacuum, dead tuples and bloat <span class="pending">soon</span>
- Freezing and transaction-id wraparound <span class="pending">soon</span>
- Postgres SLRUs: subtransactions, multixacts and their wraparound <span class="pending">soon</span>
- MVCC and indexes: dead entries, bottom-up deletion and page recycling <span class="pending">soon</span>
- Garbage collection in LSM and in-memory engines <span class="pending">soon</span>

### 4.5 Serializability Without Blocking: TO, OCC and SSI

*advanced* — Timestamp ordering, optimistic validation and serializable snapshot isolation are how PostgreSQL SERIALIZABLE, in-memory engines like Hekaton and Silo, FoundationDB and application code with version columns get serializable results without holding 2PL locks; the price is aborts instead of waits, so the SQLSTATE 40001 retry loop, its false positives and the contention regime where optimism thrashes follow directly from these protocols.

- Optimistic locking in application code: version columns and compare-and-set <span class="pending">soon</span>
- Timestamp ordering: read/write timestamps, the Thomas write rule and MVTO <span class="pending">soon</span>
- Optimistic concurrency control: read, validate, write <span class="pending">soon</span>
- Serializable snapshot isolation: rw-antidependencies and dangerous structures <span class="pending">soon</span>
- SSI in PostgreSQL: SIREAD locks, promotion and false positives <span class="pending">soon</span>
- MVOCC in Hekaton, Silo and in-memory OLTP <span class="pending">soon</span>
- Retry loops and serialization failures (SQLSTATE 40001) <span class="pending">soon</span>
- Transaction chopping: splitting long transactions without losing serializability <span class="pending">soon</span>
- Robustness: proving a workload is safe at a weaker isolation level <span class="pending">soon</span>
- Choosing a concurrency control protocol <span class="pending">soon</span>

### 4.6 Write-Ahead Logging, Checkpoints and the Commit Path

*advanced* — The log is the contract behind every commit acknowledgement: what a COMMIT returns means exactly "these bytes were durable", and the record format, the two WAL rules, the commit pipeline, the checkpoint cadence and the torn-page defenses together decide which bytes those are and how much I/O and latency the guarantee costs.

- Log records, LSNs and the two rules of write-ahead logging <span class="pending">soon</span>
- Steal/no-steal and force/no-force: why ARIES needs both undo and redo <span class="pending">soon</span>
- Group commit and the log I/O pipeline <span class="pending">soon</span>
- Checkpoints: fuzzy checkpoints and the recovery-time vs I/O trade-off <span class="pending">soon</span>
- Defenses against torn pages: full-page writes, doublewrite and checksums <span class="pending">soon</span>
- Log retention, archiving and logical decoding internals <span class="pending">soon</span>

### 4.7 Crash Recovery: ARIES, Recovery Time and What Survives

*advanced* — Recovery is two questions at once: which consistent state can be rebuilt from the bytes that were durable at the instant of the crash, and how many minutes or hours that rebuild takes while the service is down or half-open — ARIES answers the first, redo rate and the undo tail answer the second, and every durability knob is a bet on the loss window between them.

- Shadow paging and copy-on-write recovery: SQLite journals and LMDB <span class="pending">soon</span>
- ARIES recovery: analysis, redo and undo with compensation log records <span class="pending">soon</span>
- How long recovery takes: redo rate, prefetch, parallel apply and the undo tail <span class="pending">soon</span>
- Engine durability designs compared and what each knob actually loses <span class="pending">soon</span>
- Testing crash consistency: ALICE, CrashMonkey and fault injection <span class="pending">soon</span>

## Part 5 — Using the Database Well

Advanced SQL, application access paths, modeling and per-store usage, application data patterns, and zero-downtime schema evolution on a single database.

### 5.1 Advanced SQL for Engineers

*core* — Window functions, grouping sets, LATERAL, time-series operators, event-analytics patterns and recursive CTEs turn application loops and N+1 round trips into single statements the planner can optimize, and each has precise semantics (frames and peers, bucket and counter-reset rules, session-gap and conversion-window definitions, working-table iteration) that a senior engineer must know to get correct results, not just fast ones.

- Set-based thinking: replacing loops and N+1 with joins, arrays and VALUES <span class="pending">soon</span>
- Window functions: frames, peers, ranking and running aggregates <span class="pending">soon</span>
- GROUPING SETS, ROLLUP, CUBE and FILTER <span class="pending">soon</span>
- LATERAL joins and top-N-per-group <span class="pending">soon</span>
- Time-series SQL: buckets, gap filling, rates and ASOF joins <span class="pending">soon</span>
- Funnels, retention and sessionization over event tables <span class="pending">soon</span>
- Recursive CTEs: hierarchies, graphs and series <span class="pending">soon</span>

### 5.2 Advanced SQL II: Documents, Text and the Write Side

*core* — This module sits immediately after Advanced SQL because the rest of the part is written on top of it: data modeling's JSONB attribute bags, document stores' embed-versus-reference decision, search's relevance tuning over tsvector, write-side patterns' idempotent upserts and schema evolution's trigger-maintained backfills all assume this vocabulary already exists. JSONB, full-text search and pattern matching put semi-structured and human text inside the same statement as relational columns, while upserts, updatable views, data-modifying CTEs and triggers decide what a single write actually does to the database — and each has exact semantics (containment, lax versus strict jsonpath, lexeme pipelines, automaton evaluation and sargability, arbiter indexes and speculative insertion, view-translation rules, trigger and constraint timing) that separate a statement that is merely fast from one that is correct under concurrency and adversarial input.

- JSONB: operators, jsonpath and modeling in SQL <span class="pending">soon</span>
- Full-text search in SQL: tsvector, tsquery, ranking and trigrams <span class="pending">soon</span>
- Pattern matching in the database: automata, backtracking and index acceleration <span class="pending">soon</span>
- The view update problem: which views you can write through <span class="pending">soon</span>
- Upserts, MERGE, RETURNING and data-modifying CTEs <span class="pending">soon</span>
- Triggers, constraints and server-side execution order <span class="pending">soon</span>

### 5.3 Data Modeling in Practice

*core* — Most production data bugs are modeling bugs that only surface at scale or a year later — a schema nobody documented whose real join edges are folklore, a duplicated copy nobody reconciles, an orphaned child row whose parent was deleted out from under it, a customer-defined field that can never be constrained or migrated, a translation that silently falls back to English on a list page, a drag-to-reorder list that rewrites ten thousand rows per move — so a senior engineer needs an explicit discipline for reading an inherited schema and for derived data, type hierarchies, tenant-defined attributes, locales, trees and manual order.

- Inheriting an undocumented schema: archaeology and profiling <span class="pending">soon</span>
- Denormalization and read models <span class="pending">soon</span>
- Schema for embeddings: chunks, model versions and re-embedding <span class="pending">soon</span>
- Polymorphic associations and type hierarchies <span class="pending">soon</span>
- Custom fields: EAV, JSONB bags, sparse columns and DDL per tenant <span class="pending">soon</span>
- Translatable content: per-locale rows, jsonb maps and fallback chains <span class="pending">soon</span>
- Hierarchies and graphs in relational databases <span class="pending">soon</span>
- User-defined ordering: fractional indexing and reorderable lists <span class="pending">soon</span>

### 5.4 Using Document, Graph and Key-Value Stores

*core* — Document, graph and key-value stores are not schemaless — each has a modeling discipline and a query language whose execution model decides whether a request is one pointer chase, one index probe or a full scan, and the wrong document boundary, traversal language, Redis structure or multi-model claim becomes the write-amplification, latency, runaway-bill or silently-lost-message bug that only shows up in production.

- Document and JSON modeling: embed vs reference <span class="pending">soon</span>
- MongoDB usage: documents, aggregation pipeline and indexes <span class="pending">soon</span>
- Firestore and mobile-backend document stores <span class="pending">soon</span>
- Neo4j and Cypher: property graphs and traversal <span class="pending">soon</span>
- Gremlin, SPARQL and SQL/PGQ: languages and their execution <span class="pending">soon</span>
- Redis data structures and usage patterns <span class="pending">soon</span>
- Multi-model engines: what is shared and what is bolted on <span class="pending">soon</span>

### 5.5 Search in Production: Relevance, Analysis and Clusters

*advanced* — The curriculum builds inverted indexes, BM25 and analyzers but never how a search system is shaped, tuned, measured, paginated, kept fresh or defended — and those are exactly the questions a senior who owns a search box is asked: why is this result third, did the change help, why did two conditions on different children produce a false hit, why did facet counts melt the cluster, why does Japanese return nothing, and why did one client's regexp take the tier down. This module is the declared owner of document modeling for an index, relevance evaluation, analysis chains, reverse search over stored queries, search topology and search-tier admission control; the vector, hybrid and embedding half of the old module now lives in its sibling vector-and-hybrid-retrieval, Part 2's indexes-beyond-the-b-tree owns the static inverted-index structures, index-selection-strategy only points forward, and query-scheduling-and-admission-control owns the SQL-engine version of load shedding rather than the search tier's.

- Modeling documents for a search index: flattening, nested and parent-child <span class="pending">soon</span>
- Relevance tuning and evaluation <span class="pending">soon</span>
- Stored queries and reverse search: percolate and alerting rules <span class="pending">soon</span>
- Multilingual and CJK analysis: segmentation, folding and per-language fields <span class="pending">soon</span>
- Facets, aggregations and deep pagination <span class="pending">soon</span>
- Autocomplete, fuzzy and did-you-mean <span class="pending">soon</span>
- Search index topology and zero-downtime reindex <span class="pending">soon</span>
- Search under load: thread pools, circuit breakers and expensive queries <span class="pending">soon</span>

### 5.6 Geospatial Data: Types, Queries and Cell Indexing

*advanced* — Geospatial appears in the curriculum only as R-tree and GiST mechanics, yet distances are silently wrong under the wrong projection, invalid geometries make predicates lie, spatial plans are the one place an index returns candidates that must be rechecked, most production 'find nearby' runs on a sorted-key store with no R-tree at all, 'how do I get there and how long will it take' is answered by a precomputed graph index no bounding box can substitute for, and the dense raster and array side of the domain fits neither row nor column storage.

- Spatial types, projections and predicates <span class="pending">soon</span>
- Filter-and-refine, spatial joins and nearest neighbour <span class="pending">soon</span>
- Cell-based spatial indexing on sorted keys <span class="pending">soon</span>
- Moving objects: geofences, proximity and trajectories <span class="pending">soon</span>
- Routing on a road network: shortest paths, precomputation and isochrones <span class="pending">soon</span>
- Serving maps: tiles, generalization and precomputation <span class="pending">soon</span>
- Chunked arrays and rasters: tiles, chunk shape and pyramids <span class="pending">soon</span>

### 5.7 Time-Series and Observability Stores

*advanced* — Every senior operates a metrics, logs and traces stack, eventually causes or debugs a cardinality explosion, and is asked to cut its bill; purpose-built time-series engines differ from both OLTP and general analytics in their partitioning, compression, label indexing and downsampling. The module is self-contained: it introduces the two ideas it needs from general partitioning — time as the partition key, and retention implemented by dropping a whole partition instead of deleting rows — in its first subtopic rather than assuming them, so it can be read before or after the application-side partitioning material. It stops at the single-node mechanism: the scale-out metrics stack (remote write, compactor downsampling, cross-replica dedup) is the Part 9 case study.

- Time-series engine designs: chunks, hypertables and TSM <span class="pending">soon</span>
- Prometheus TSDB internals: head block, chunks and postings <span class="pending">soon</span>
- Cardinality: label design, churn and enforcement <span class="pending">soon</span>
- Rollups, downsampling, retention and continuous aggregates <span class="pending">soon</span>
- Sampling and aliasing: what a monitoring graph cannot show <span class="pending">soon</span>
- Logs, traces and wide events at scale <span class="pending">soon</span>

### 5.8 Application Access: ORMs, Generated APIs, Protocols and Drivers

*advanced* — Most OLTP latency is spent outside the executor: in the ORM session or generated API layer that actually emits the SQL, in the driver that encodes and decodes every value, and in round trips and handshakes that are invisible unless you can read the wire protocol, so this module makes the client-to-backend path as legible as a query plan — and makes each statement attributable back to the endpoint, tenant and deploy that emitted it.

- The round-trip tax: ORMs, lazy loading and the N+1 problem <span class="pending">soon</span>
- ORM internals: identity map, unit of work and flush ordering <span class="pending">soon</span>
- Compiling REST and GraphQL to SQL: PostgREST, Hasura and the arbitrary-query problem <span class="pending">soon</span>
- Wire protocols: PostgreSQL simple vs extended, MySQL and pipelining <span class="pending">soon</span>
- Inside the driver: type mapping, statement caches and the errors they produce <span class="pending">soon</span>
- Seeing the database from the application: query tags, spans and per-endpoint budgets <span class="pending">soon</span>

### 5.9 Moving Data and Routing Requests: Bulk Load, Streaming, Replicas and Retries

*advanced* — Once the connection path is understood, the remaining client-side work is moving bytes in bulk without holding locks or snapshots too long, streaming results that do not fit in memory, sending each query to the right role, wiring change notifications through the same pooling and replica plumbing, and surviving the timeouts and in-doubt commits that every one of those paths eventually produces.

- Batching, pipelining and bulk loading: multi-row INSERT and COPY <span class="pending">soon</span>
- Export, Arrow Flight and cross-engine data sharing <span class="pending">soon</span>
- Read/write splitting in the application framework <span class="pending">soon</span>
- Wiring LISTEN/NOTIFY through the client stack: pooling, payload limits and the re-read <span class="pending">soon</span>
- Server-side cursors, portals and result streaming <span class="pending">soon</span>
- Timeouts, retries and error handling from the client <span class="pending">soon</span>

### 5.10 Caching Tiers, Stampedes and Client-Side Data

*advanced* — Every read the application does not send to the database lives in a cache tier, a change notification or a client-side replica, and all three are correctness systems disguised as performance systems: invalidation races, stampedes, slab fragmentation, dropped notifications and offline mutation rebase all decide what a user actually sees, so this module makes the staleness window, the delivery guarantee and the memory behaviour of that layer explicit instead of accidental.

- Caching patterns: cache-aside, read-through, write-through and invalidation races <span class="pending">soon</span>
- Stampedes, thundering herds and request coalescing <span class="pending">soon</span>
- Inside the cache tier: slabs, admission and topology <span class="pending">soon</span>
- LISTEN/NOTIFY, change streams and realtime subscriptions <span class="pending">soon</span>
- Databases in the browser: WASM engines, OPFS and hybrid execution <span class="pending">soon</span>
- Sync engines: shapes, buckets, optimistic mutations and rebase <span class="pending">soon</span>

### 5.11 Application Data Design Patterns

*advanced* — Deletion policy, ID scheme, event-table layout, where ephemeral state lives, how fleet-wide configuration is distributed and the tenancy model are chosen once at design time and paid for forever — each has a default that looks fine at small scale and a failure mode (re-registration blocked by a unique index, random keys splitting every leaf, retention that can only be done by DELETE, a session table that becomes the WAL and vacuum bottleneck, one bad config row reaching every node in seconds, one tenant's migration fanning out across thousands of schemas) that appears only later, so the invariant behind the choice matters more than the pattern's name.

- Soft deletes, archival and retention <span class="pending">soon</span>
- ID generation: sequences, UUIDs, UUIDv7 and Snowflake IDs <span class="pending">soon</span>
- Time-series, events and append-only tables <span class="pending">soon</span>
- High-churn ephemeral state: sessions, presence, carts and view counts <span class="pending">soon</span>
- Reference and configuration data: tiny tables, enormous fan-out <span class="pending">soon</span>
- Multi-tenancy layouts: shared schema, schema-per-tenant, database-per-tenant <span class="pending">soon</span>

### 5.12 Write-Side Patterns: Side Effects, Idempotency and Background Work

*advanced* — Retries, concurrent writers and crashes in the gap between a commit and an external side effect are where correct-looking application code loses money — double charges, orphaned objects, duplicated jobs, an order advanced twice from a stale read, a customer's CSV imported twice, a chunk swept out from under a live manifest, a recurring billing run that fires twice or silently skips a day — and each hazard has a specific mechanism (claim-before-effect, conditional transitions, leases and fencing, content-addressed immutability with safe reclamation, chunked resumable jobs, due-time indexes with catch-up policy) that a senior engineer is expected to reach for before inventing a worse one.

- Transactions in application code: boundaries and side effects <span class="pending">soon</span>
- Files and blobs: the database, the object store and the gap between them <span class="pending">soon</span>
- Content-addressed blobs: chunking, dedup and reference-counted GC <span class="pending">soon</span>
- Idempotency keys and deduplication tables <span class="pending">soon</span>
- Status columns, legal transitions and concurrent advancement <span class="pending">soon</span>
- Advisory locks and database-native mutual exclusion <span class="pending">soon</span>
- Queues and job tables in the database: SKIP LOCKED <span class="pending">soon</span>
- Customer-facing bulk import and export <span class="pending">soon</span>
- Timers and schedules: due indexes, recurrence and catch-up <span class="pending">soon</span>

### 5.13 Hot Rows: Counters, Quotas, Escrow and Ledgers

*advanced* — A single row that every writer touches — a view counter, a rate-limit bucket, a stock level, an account balance — caps throughput at one update per lock hold time no matter how many cores or nodes you add, and the escapes (sharding, batched rollups, approximate sketches, budget leases across a fleet, escrow intervals, append-only balanced postings) each trade exactness, read cost or invariant strength in a way a senior engineer is expected to choose deliberately rather than discover under load.

- Counters, rate limits and aggregates <span class="pending">soon</span>
- Fleet-wide rate limits and quotas: local buckets versus global truth <span class="pending">soon</span>
- Escrow and commutative updates for hot rows with invariants <span class="pending">soon</span>
- Double-entry ledgers and transfer lifecycles <span class="pending">soon</span>

### 5.14 Schema Evolution and Zero-Downtime Migrations

*advanced* — Every ALTER is a lock plus (sometimes) a full table rewrite, and rolling deploys mean two versions of the application share one schema at every moment; the expand/contract discipline, two-step online constraints, throttled backfills, idempotent data and seed migrations and generated-then-reviewed migration files are the mechanisms that turn a 30-minute outage into a boring change.

- What DDL locks and rewrites <span class="pending">soon</span>
- The expand/contract pattern <span class="pending">soon</span>
- Adding constraints and indexes online: NOT VALID, CONCURRENTLY, lock_timeout <span class="pending">soon</span>
- Backfills at scale <span class="pending">soon</span>
- Data migrations and reference data as code <span class="pending">soon</span>
- Migration tooling, linting and governance <span class="pending">soon</span>
- Autogenerated migrations and what the differ cannot see <span class="pending">soon</span>
- Branching databases: copy-on-write forks, deploy requests and ephemeral environments <span class="pending">soon</span>

### 5.15 Developing and Testing Against Databases

*core* — Every backend engineer maintains a test suite that talks to a database and a pipeline that ships schema changes, and the recurring failures are mechanism questions the site otherwise teaches nowhere: tests passing on SQLite while production is Postgres, per-test isolation that silently breaks on code that commits, a 50-row fixture set that guarantees plan regressions reach production, two branches whose migrations apply in either order, and a change approved because 'staging looked fine'.

- Ephemeral test databases: containers, templates and per-test isolation <span class="pending">soon</span>
- Testing logic that lives in the database <span class="pending">soon</span>
- Realistic test data: factories, subsetting, anonymization and branching <span class="pending">soon</span>
- Migrations in the pipeline: ordering, branching, preview databases and drift <span class="pending">soon</span>
- Capture and replay: validating a change against the real workload <span class="pending">soon</span>
- Regression gates in CI: query counts, plan snapshots and lock budgets <span class="pending">soon</span>

### 5.16 Databases and AI Workloads

*advanced* — LLM features are now shipped over production databases, and their failure mode is not a crash but a confidently wrong number, a cache that answers a paraphrase of a different question, a stale or over-permissive retrieval set, or an inference call inside the executor stalling ingest — all of which are database-side controls a senior is asked to design, while the privilege, row-level-security and tenant-isolation half of the problem (including agents holding SQL access) belongs to Part 10's security module.

- Text-to-SQL: schema linking, grounding and execution-guided validation <span class="pending">soon</span>
- RAG data plumbing: chunking, freshness and permission-aware retrieval <span class="pending">soon</span>
- Semantic caching: approximate keys, thresholds and invalidation <span class="pending">soon</span>
- Generating model output where the data lives <span class="pending">soon</span>

### 5.17 Modeling Space, Time and Schema Change

*core* — The modeling decisions that bite hardest a year out are the ones about where a thing is, when it was true, when we recorded it, and what the schema meant at the time — a location column that cannot serve the radius query the product later needs, a meeting that moves an hour when tzdata ships new DST rules, a correction that silently rewrites last quarter's history, a renamed field that a stale consumer reads as a default — so a senior engineer needs an explicit discipline for spatial representation, recorded time, bitemporal history and compatible schema change.

- Modeling location: which spatial representation a schema should store <span class="pending">soon</span>
- Modeling time: recurrence, per-user days and the recording clock <span class="pending">soon</span>
- Temporal modeling: history tables and bitemporal data <span class="pending">soon</span>
- Schema evolution principles: Avro, Protobuf and compatibility <span class="pending">soon</span>

### 5.18 Vector and Hybrid Retrieval in Production

*advanced* — The curriculum builds ANN structures and BM25 separately but never how a retrieval system fuses them, decides whether it needs an index at all, measures recall as an SLO, re-embeds when the model changes or lays a vector collection across shards — and those are exactly the questions a senior who owns a RAG pipeline is asked: why did adding a filter destroy recall, why is brute force faster than our index, why did recall drop after the rebuild, and why does the coordinator lose true neighbours. This module is the declared owner of hybrid retrieval, the embedding lifecycle and vector topology: Part 2's tries-learned-and-vector-indexes owns the static ANN structures and the churn mechanism, search-and-vector-retrieval owns the lexical half these stages fuse with, index-selection-strategy only points forward, data-modeling keeps the chunk/version table shape and links here for re-embedding, and databases-and-ai-workloads and retrieval-toy-engines-and-frontier apply these rather than re-teach them.

- When a full scan beats an ANN index <span class="pending">soon</span>
- Hybrid retrieval and reranking <span class="pending">soon</span>
- Embedding lifecycle and vector index operations <span class="pending">soon</span>
- Vector search topology: shards, replicas and per-shard top-k <span class="pending">soon</span>

### 5.19 Connections and Pooling: Process Models, Poolers and Serverless Access

*advanced* — A connection is the most expensive object an application creates and the one it manages worst: a PostgreSQL backend is a process with private memory and a handshake, the worker model (fork, threads or an event loop) silently decides how many of them a deploy demands, and the pooler that multiplexes them trades away exactly the session state your framework assumes — so connection budget, pooling mode and process model, not the executor, decide whether the database falls over under a traffic burst.

- Connection lifecycle and cost: TCP, TLS, SCRAM and process-per-connection <span class="pending">soon</span>
- Connections across process models: forks, threads and event loops <span class="pending">soon</span>
- Connection pooling: PgBouncer modes, session state and pool sizing <span class="pending">soon</span>
- Serverless and edge access: connection management without a long-lived process <span class="pending">soon</span>
- Driver and client-library upgrades: auth plugins, TLS defaults and protocol negotiation <span class="pending">soon</span>

### 5.20 Migrating Large Live Tables and Non-Relational Stores

*advanced* — Past a few hundred gigabytes the ordinary online-DDL tricks run out and the change has to be staged through a shadow or ghost table, a partition attach or a shadow column with its own trigger, backfill and cut-over — and outside the relational engines the schema lives in documents, gossiped table definitions and retained event streams where old versions never stop arriving; these are the mechanisms behind the largest, scariest and most common migrations a senior engineer will own.

- Schema evolution in NoSQL and event streams <span class="pending">soon</span>
- Online schema change tools: gh-ost, pt-osc and InnoDB online DDL <span class="pending">soon</span>
- Partitioning a large table that is already in production <span class="pending">soon</span>
- Key and type migrations: int to bigint and primary key changes <span class="pending">soon</span>

## Part 6 — Distributed Foundations: Failure, Consistency, Time and Truth

Why distribute at all, the consistency model hierarchy, clocks and ordering, and failure detection with leases and fencing — the rules before the protocols.

### 6.1 Why Distribute: Goals, Failure Models and the Rules of the Game

*core* — Partial failure, tail latency and metastable overload are the new physics once data leaves one machine; this module fixes the goals, the timing and failure assumptions, and the vocabulary that every later protocol (consistency models, clocks, failure detection, replication, consensus) is judged against.

- Scale up vs scale out; shared-nothing vs shared-disk <span class="pending">soon</span>
- The three goals: scalability, availability and latency <span class="pending">soon</span>
- Fallacies of distributed computing and partial failure <span class="pending">soon</span>
- Anatomy of a distributed request <span class="pending">soon</span>
- System models: synchrony, crash-stop and Byzantine <span class="pending">soon</span>
- Tail latency, fan-out and hedged requests <span class="pending">soon</span>
- Metastable failures and cascading overload <span class="pending">soon</span>

### 6.2 Networks, RPC and the Shape of Partial Failure

*core* — Every distributed mechanism later in the course — replication streams, heartbeats, quorum messages, commit protocols — rides on an unreliable network that can drop, duplicate, reorder, delay and asymmetrically cut messages, and an engineer who has not internalized exactly which of those the wire permits will design protocols whose safety argument quietly assumes a network that does not exist.

- What the wire actually does: TCP, TLS, buffers and RTT <span class="pending">soon</span>
- Message delivery: loss, duplication, reordering and head-of-line blocking <span class="pending">soon</span>
- RPC semantics: at-most-once, at-least-once and idempotent servers <span class="pending">soon</span>
- Retries, jitter and retry budgets: when retrying makes the outage <span class="pending">soon</span>
- Flow control and backpressure: bounded queues and where bytes pile up <span class="pending">soon</span>
- Partitions in the wild: asymmetric loss, gray failure and the one-way link <span class="pending">soon</span>

### 6.3 Consistency Models and Impossibility Results

*advanced* — A consistency model is a precise statement of which histories a replicated system may produce; linearizability and its weaker siblings are the only honest vocabulary for what a system promises, and CAP, PACELC, harvest/yield and the highly-available-transactions results only make sense once that vocabulary is in place.

- Histories, registers and linearizability <span class="pending">soon</span>
- The hierarchy: sequential, causal, PRAM and eventual <span class="pending">soon</span>
- Session guarantees: read-your-writes, monotonic reads and causal sessions <span class="pending">soon</span>
- CAP and PACELC done right <span class="pending">soon</span>
- Harvest, yield and degraded modes <span class="pending">soon</span>
- The availability boundary: which models survive a partition <span class="pending">soon</span>
- Coordination avoidance: CALM, I-confluence and which invariants really need it <span class="pending">soon</span>
- Isolation meets consistency: strict serializability and friends <span class="pending">soon</span>
- The latency price of consistency: Attiya-Welch and Lipton-Sandberg bounds <span class="pending">soon</span>
- Where you actually need linearizability <span class="pending">soon</span>

### 6.4 Time, Clocks and Ordering

*advanced* — Ordering events without a trustworthy clock is the core trick behind every replication, conflict-resolution and distributed-transaction protocol.

- Physical clocks, NTP and their failures <span class="pending">soon</span>
- Happens-before and why wall clocks cannot order events <span class="pending">soon</span>
- Operating on assumed clock bounds <span class="pending">soon</span>
- Lamport clocks and total order <span class="pending">soon</span>
- Consistent global snapshots: cuts and the Chandy-Lamport algorithm <span class="pending">soon</span>
- Vector clocks and version vectors <span class="pending">soon</span>
- Ordering without clocks: sequencers and timestamp oracles <span class="pending">soon</span>
- TrueTime: the interval API <span class="pending">soon</span>
- Hybrid logical clocks <span class="pending">soon</span>

### 6.5 Failure Detection, Membership, Leases and Fencing

*advanced* — Deciding who is alive and who holds authority is the mechanism behind failover, and getting it wrong is the origin of split brain.

- Failure detectors and FLP: why slow is indistinguishable from dead <span class="pending">soon</span>
- Heartbeats, timeouts and phi accrual detection <span class="pending">soon</span>
- Gossip protocols: epidemic dissemination and anti-entropy of membership <span class="pending">soon</span>
- Classic leader election and why it split-brains <span class="pending">soon</span>
- Leases: time-bounded authority and the pause that breaks it <span class="pending">soon</span>
- Fencing tokens: making the storage layer the last line of defense <span class="pending">soon</span>
- SWIM and Lifeguard: scalable membership in memberlist, Serf and Consul <span class="pending">soon</span>
- Membership changes and quorum safety <span class="pending">soon</span>

## Part 7 — Replication, Partitioning and Consensus

How data is copied, split and agreed upon: leader-based replication, sharding with modeling for partitioned stores, Dynamo-style quorums and CRDTs, then Paxos, Raft and coordination services.

### 7.1 Replication

*advanced* — Replication is the first distributed mechanism most engineers operate, and lag, failover data loss and replica reads are its daily consequences; understanding where the ack sits, what the follower actually applies and what happens when the leader dies is the difference between a read replica and a silent data-loss incident.

- Why replicate and the three topologies: single-leader, multi-leader, leaderless <span class="pending">soon</span>
- Single-leader replication and log formats: physical, logical, statement <span class="pending">soon</span>
- Synchronous, asynchronous and semi-synchronous replication <span class="pending">soon</span>
- Replication lag anomalies and replica reads <span class="pending">soon</span>
- Failover, split brain and data loss on async failover <span class="pending">soon</span>
- Multi-leader replication and conflict resolution <span class="pending">soon</span>
- Chain replication and CRAQ <span class="pending">soon</span>

### 7.2 Operating Replication: Slots, GTIDs and Apply Conflicts

*advanced* — Understanding replication mechanisms is one thing and running them is another: the pages that actually wake engineers are a stuck apply worker filling the publisher's disk, a slot lost at promotion forcing a full re-snapshot, an errant GTID that quietly makes a replica unpromotable, and a standby cancelling queries because vacuum caught up with its snapshot — each a knob-level fault you can only fix if you know which artifact to read and which recovery action silently drops data.

- Replication in practice: Postgres slots and timelines, MySQL GTIDs, MongoDB replica sets <span class="pending">soon</span>
- Adding and removing replicas without hurting the primary <span class="pending">soon</span>
- Logical replication apply conflicts and the stuck subscriber <span class="pending">soon</span>
- Slot failover, pg_createsubscriber and logical replication after promotion <span class="pending">soon</span>
- MySQL replication operations: errant GTIDs, skips and filters <span class="pending">soon</span>
- Long transactions vs replicas: hot_standby_feedback and query conflicts <span class="pending">soon</span>

### 7.3 Partitioning and Sharding: Key Design, Placement and Rebalancing

*advanced* — The partition key fixes where every byte lives, so it decides which queries stay single-partition, how much load skews even under a perfect hash, what a membership change costs to move, how requests find the right shard, and whether a rebalance or a celebrity key can be absorbed without taking foreground traffic down.

- Range vs hash partitioning and key design <span class="pending">soon</span>
- Consistent hashing, virtual nodes and alternatives <span class="pending">soon</span>
- Balls into bins: why uniform hashing still gives you a hot shard <span class="pending">soon</span>
- Placement as constrained optimization: balance, constraints and move cost <span class="pending">soon</span>
- Request routing and cluster metadata <span class="pending">soon</span>
- Rebalancing and resharding <span class="pending">soon</span>
- Hot partitions, skew and adaptive capacity <span class="pending">soon</span>

### 7.4 Querying and Modeling for Partitioned Stores

*advanced* — Once the data is split, ordinary query shapes stop being free — joins, aggregates, pagination and lookups by a non-key attribute all become fan-out or remote-index problems — and DynamoDB and Cassandra modeling is exactly that discipline applied up front, designing the keys so the access patterns that matter stay single-partition.

- Cross-shard queries: scatter-gather, joins and aggregation basics <span class="pending">soon</span>
- Pagination and cursors across shards <span class="pending">soon</span>
- Secondary indexes across partitions: local vs global <span class="pending">soon</span>
- Global uniqueness across shards: claim tables and reserve-then-create <span class="pending">soon</span>
- Modeling for DynamoDB: access-pattern-first and single-table design <span class="pending">soon</span>
- Modeling for Cassandra: partitions, clustering and time-ordered data <span class="pending">soon</span>

### 7.5 Leaderless Replication, Quorums, Anti-Entropy and CRDTs

*advanced* — Dynamo-style systems trade coordination for availability, and quorum math, repair and CRDTs are how they stay correct enough.

- The coordinator path and quorum arithmetic: N, W, R and consistency levels <span class="pending">soon</span>
- Quorum systems: weighted voting, grids, load and availability <span class="pending">soon</span>
- Why strict quorums are still not linearizable <span class="pending">soon</span>
- Sloppy quorums and hinted handoff <span class="pending">soon</span>
- Read repair, digest reads and distributed tombstones <span class="pending">soon</span>
- Anti-entropy with Merkle trees and repair scheduling <span class="pending">soon</span>
- Conflict detection: last-write-wins, siblings and dotted version vectors <span class="pending">soon</span>
- CRDTs I: convergence by construction with counters, registers and sets <span class="pending">soon</span>
- CRDTs II: sequences, collaborative text and garbage <span class="pending">soon</span>
- The Dynamo lineage compared: Dynamo, Riak, Cassandra, Voldemort, ScyllaDB and DynamoDB <span class="pending">soon</span>

### 7.6 Consensus Foundations and Raft

*expert* — Consensus is the primitive that turns a set of machines into one linearizable log, and Raft is the implementation a senior engineer actually meets — etcd, Consul, TiKV, CockroachDB and Kafka's KRaft all run it — so this module establishes what consensus guarantees, why no deterministic asynchronous protocol can guarantee termination, and then walks Raft's safety argument, reconfiguration, snapshotting, flow control and read paths in enough detail to debug a real cluster.

- The consensus problem and total order broadcast <span class="pending">soon</span>
- Impossibility results and how systems dodge them: FLP, consensus numbers and randomization <span class="pending">soon</span>
- Raft: election and log replication <span class="pending">soon</span>
- Raft: membership changes, snapshots, log retention and disruptive rejoiners <span class="pending">soon</span>
- Raft: linearizable reads — log-write, ReadIndex and leader leases <span class="pending">soon</span>

### 7.7 Paxos, Leaderless Consensus and Byzantine Faults

*expert* — Paxos is the language the literature and the big systems are written in — Chubby, Spanner's Paxos groups, Cassandra LWT — and its variants map the design space Raft deliberately narrows: where the message-delay floor really is, whether a distinguished leader is needed at all, and what changes when a node may lie; a senior engineer needs this to read the papers and view-change protocols, to answer 'why does this commit cost two inter-region round trips and can any design make it one?', and to explain why a database inside one trust domain still budgets for crashes and corrupt disks rather than tripling replicas.

- Paxos: single-decree <span class="pending">soon</span>
- Multi-Paxos and log replication <span class="pending">soon</span>
- How many message delays a commit must take: consensus latency lower bounds <span class="pending">soon</span>
- Zab, Viewstamped Replication and Paxos Made Live lessons <span class="pending">soon</span>
- EPaxos and leaderless consensus <span class="pending">soon</span>
- Byzantine faults: PBFT, 3f+1 and why databases assume crashes <span class="pending">soon</span>

### 7.8 Coordination Services and Recipes

*expert* — Almost nobody implements a consensus protocol, but nearly every distributed system leans on one through ZooKeeper, etcd or a conditional write, and that is where the failures actually happen — a paused lock holder, a watch that fired while nobody was listening, a log entry that never reached the disk — so a senior engineer needs the primitives, the correct recipes and their pitfalls in one frame.

- ZooKeeper: znodes, sessions and watches <span class="pending">soon</span>
- etcd v3: revisions, leases, txn and watch streams <span class="pending">soon</span>
- Chubby and what a coordination service is for <span class="pending">soon</span>
- Recipes: leader election, herd-free locks, barriers and membership <span class="pending">soon</span>
- Coordination pitfalls: paused holders, watch gaps and misuse <span class="pending">soon</span>
- Consensus under storage faults: fsync vs replication durability <span class="pending">soon</span>
- Recovering from permanent quorum loss <span class="pending">soon</span>
- Lightweight transactions: Paxos inside Cassandra <span class="pending">soon</span>

## Part 8 — Distributed Transactions, Architectures, Streams and Verification

Atomic commit and distributed SQL protocols, cloud-native and disaggregated architectures, distributed and analytical query execution, logs and derived data, and how distributed claims are verified.

### 8.1 Distributed Transactions: Atomic Commitment Across Partitions

*expert* — Every multi-partition write has to answer the same question — who holds the commit decision, and what happens to the other partitions when that holder dies mid-flight — and the answers run from classic 2PC and its blocking window, through replicated coordinators and Percolator's primary-key trick, to protocols that give up ordering transactions against each other in exchange for never blocking or aborting at all.

- Two-phase commit <span class="pending">soon</span>
- Write paths in Cassandra and DynamoDB <span class="pending">soon</span>
- Read-atomic isolation and RAMP: atomic visibility without two-phase commit <span class="pending">soon</span>
- 3PC, Paxos Commit and replicated coordinators <span class="pending">soon</span>
- Percolator and TiDB <span class="pending">soon</span>

### 8.2 Timestamp-Ordered and Deterministic Distributed Transactions

*expert* — Once the commit decision is safe, the remaining problem is order: Spanner, CockroachDB and Calvin are three different answers to how machines with unsynchronized clocks agree on one transaction order — pay for it with commit-wait, with uncertainty restarts, or with a replicated input log — and each answer sets what anomalies survive, how contention is arbitrated and when old versions can finally be reclaimed.

- Spanner: TrueTime commit-wait and external consistency <span class="pending">soon</span>
- CockroachDB transactions: intents, HLC and parallel commits <span class="pending">soon</span>
- Calvin and deterministic databases <span class="pending">soon</span>
- Distributed deadlock detection and priorities <span class="pending">soon</span>
- Clock-skew anomalies: causal reverse and stale reads <span class="pending">soon</span>
- Isolation across shards and cluster-wide MVCC garbage collection <span class="pending">soon</span>

### 8.3 Distributed SQL Architectures: Sharded Raft, Shared-Disk and Compatibility

*expert* — Range-sharded Raft clusters, sharding middleware and shared-disk clusters are the families a senior engineer picks among when one machine stops being enough, and each one moves the durability point, the failover unit and the coordination tax somewhere different; on top of that almost every new entrant sells 'Postgres compatibility', a layered claim you have to be able to take apart before you bet an application on it.

- Architecture families: one write, four systems <span class="pending">soon</span>
- 'Postgres-compatible': which layer is actually compatible <span class="pending">soon</span>
- Range-sharded KV with Raft: CockroachDB, TiKV, YugabyteDB <span class="pending">soon</span>
- Geo-distribution, follower reads and placement <span class="pending">soon</span>
- Shared-disk clustering: Oracle RAC cache fusion <span class="pending">soon</span>
- Online schema change in distributed SQL: the F1 state machine <span class="pending">soon</span>

### 8.4 Cloud-Native and Disaggregated Architectures: Aurora, Object Stores and Serverless

*expert* — Once storage is a service rather than a disk, the durability point moves off the compute node entirely — into a redo quorum, a journal, or an immutable object — and everything an engineer cares about (commit latency, recovery time, branch cost, cold starts, the monthly bill) is decided by how many requests, bytes and inter-AZ hops sit between the client and that point.

- Aurora: the log is the database <span class="pending">soon</span>
- Object stores as substrate: S3 semantics and erasure coding vs replication <span class="pending">soon</span>
- Durability math: AFR, repair windows and what eleven nines means <span class="pending">soon</span>
- The object-storage tier ladder and what each tier makes possible <span class="pending">soon</span>
- Disaggregated storage engines: Neon, SlateDB, WarpStream, Snowflake <span class="pending">soon</span>
- The shared-storage OLTP family: Socrates, PolarDB, AlloyDB and the swing back to local NVMe <span class="pending">soon</span>
- Zero-disk analytics and search: shared storage under ClickHouse, Elasticsearch and StarRocks <span class="pending">soon</span>
- Aurora DSQL: adjudicators, journals and OCC at region scale <span class="pending">soon</span>
- Serverless and elastic scaling internals <span class="pending">soon</span>

### 8.5 Distributed Query Execution: Joins, Exchange and Shuffles

*expert* — MPP join strategy, exchange placement, skew handling and shuffle fault tolerance decide analytical cost and latency by orders of magnitude.

- Distributed join strategies: co-located, broadcast, shuffle and semi-join reduction <span class="pending">soon</span>
- Skew, salting and adaptive query execution <span class="pending">soon</span>
- Exchange operators, plan fragments and distribution properties <span class="pending">soon</span>
- Distributed aggregation, DISTINCT, top-N and mergeable sketches <span class="pending">soon</span>
- Scan splits, columnar batches and late materialization across the wire <span class="pending">soon</span>
- Graph analytics: vertex-centric, linear algebra and worst-case optimal joins <span class="pending">soon</span>
- Cancellation, stragglers and fault tolerance across fragments <span class="pending">soon</span>

### 8.6 Analytical Query Platforms: Warehouses, Federation and Approximation

*expert* — Above the execution layer, what actually decides analytical cost is how much data is pruned, where the query runs, which plan fragments are pushed into another engine, and whether an approximate or precomputed answer is good enough.

- Cloud warehouse query path: pruning, caches, virtual warehouses and slots <span class="pending">soon</span>
- Federated queries and connector pushdown <span class="pending">soon</span>
- Composable engines: Arrow, ADBC/Flight SQL, Substrait and reusable executors <span class="pending">soon</span>
- Approximate answers: sketches, samples and error bars <span class="pending">soon</span>
- Segments and set membership: bitmaps, intersections and incremental updates <span class="pending">soon</span>
- Materialized views: query rewrite, incremental maintenance and HTAP replicas <span class="pending">soon</span>
- Single-node vectorized vs cluster: when DuckDB beats MPP <span class="pending">soon</span>

### 8.7 Logs, Streams and Derived Data

*expert* — The log unifies replication, CDC, messaging and derived views; knowing what a broker acknowledgement, an offset commit and a rebalance actually promise is what keeps data moving between systems without loss or duplication.

- The log as the universal primitive <span class="pending">soon</span>
- Change data capture and the outbox pattern <span class="pending">soon</span>
- Queues that are not logs: SQS, RabbitMQ, Pulsar, NATS and Redis compared <span class="pending">soon</span>
- Kafka consumer groups, rebalancing and offset semantics <span class="pending">soon</span>
- Batch processing: MapReduce, shuffles and reprocessing <span class="pending">soon</span>
- Kafka replication internals: ISR, leader epochs, acks and fsync <span class="pending">soon</span>
- Exactly-once semantics: idempotent producers and transactions <span class="pending">soon</span>
- Shared-log abstractions: Corfu, Delos, BookKeeper/Pulsar vs Kafka <span class="pending">soon</span>
- One copy of the bytes: Kafka topics as Iceberg tables <span class="pending">soon</span>

### 8.8 Data Engineering: Warehouse Modeling, Ingestion and Pipelines

*advanced* — The site teaches columnar storage, lakehouse formats, CDC and MPP execution but never how analytical tables are shaped, extracted, loaded or scheduled — and the classic failures (a wrong grain that double-counts, a fact joined to the current rather than the then-current dimension, a rerun that duplicates rows, an API extract whose watermark silently skips rows, a partner file delivered twice) are modeling and pipeline problems that no amount of engine knowledge fixes; this module opens the site's data-platform track, which sits after the streaming and verification thread rather than inside it, and it is the single owner of dimensional modeling and slowly changing dimensions, because the grain, the SCD type and the load that maintains them are one decision and Part 5's data-modeling-and-per-store-usage only raises the grain question and points here.

- Dimensional modeling: star schemas, grain and fact tables <span class="pending">soon</span>
- Slowly changing dimensions and incremental loads <span class="pending">soon</span>
- Ingesting from APIs and SaaS sources: incremental extraction without a log <span class="pending">soon</span>
- Landing zones: ingesting partner files from SFTP and object storage <span class="pending">soon</span>
- One-time imports: onboarding a legacy or customer dataset <span class="pending">soon</span>
- Zero-ETL and managed replication: what the provider actually runs <span class="pending">soon</span>
- Loading the warehouse: staged files, streaming ingest and clustering upkeep <span class="pending">soon</span>
- Orchestration and idempotent pipelines: schedules, backfills and watermarks <span class="pending">soon</span>
- The transformation layer: models, materializations and incrementality <span class="pending">soon</span>
- CI for the transformation layer: slim runs, PR environments and data diffs <span class="pending">soon</span>

### 8.9 Analytics Delivery: Quality, Semantics, BI and Activation

*advanced* — Once tables are modeled and loaded, the remaining half of a data platform is making them trustworthy and consumable, and this is where the failures a senior is actually paged for live — bad data travelling six hops before anyone notices, a customer table where the same person exists four times, three dashboards disagreeing about 'revenue' because each re-implemented the filter, a concurrency queue backing up behind scheduled deliveries with a surprise bill attached, and a warehouse-derived field quietly becoming the value an operational transaction depends on.

- Data quality, contracts and lineage <span class="pending">soon</span>
- Deduplication and record linkage: blocking, scoring and stable clusters <span class="pending">soon</span>
- Semantic and metrics layers: definitions, join paths and pre-aggregation routing <span class="pending">soon</span>
- Serving BI: dashboards, extracts, concurrency and the query bill <span class="pending">soon</span>
- Reverse ETL: pushing warehouse results back into operational systems <span class="pending">soon</span>

### 8.10 Stream Processing, Derived State and Durable Execution

*expert* — Derived state is only as correct as the semantics that maintain it: windows and watermarks, incremental views, sagas and durable workflows, tamper-evident history and end-to-end checks are what keep an unbundled system honest.

- Sagas and compensations <span class="pending">soon</span>
- Auditing derived state against its source log <span class="pending">soon</span>
- Stream processing semantics: time, windows, watermarks and state <span class="pending">soon</span>
- Enriching a stream: async lookups, caches and pushing back on the source <span class="pending">soon</span>
- What a stream cannot compute: space lower bounds and predicting state size <span class="pending">soon</span>
- Dataflow, incremental computation and event sourcing <span class="pending">soon</span>
- Choosing a streaming database: Materialize, RisingWave, Flink SQL, ksqlDB <span class="pending">soon</span>
- Durable execution: workflow histories, deterministic replay and versioning <span class="pending">soon</span>
- Unbundling the database and end-to-end correctness <span class="pending">soon</span>

### 8.11 Verifying Data Systems: Jepsen, Simulation and Formal Methods

*expert* — Vendor claims are checked by generating histories under faults, by fuzzing the engine for wrong answers and by exhaustively exploring protocol state spaces; seniors must read Jepsen reports critically, verify their own replicas and clusters, and know which method (black-box history checking, query oracles and differential testing, model-based property testing against a reference model, deterministic simulation, model checking) finds which class of bug.

- Jepsen: history generation and fault injection <span class="pending">soon</span>
- Chaos engineering and game days <span class="pending">soon</span>
- Linearizability checking: Knossos and Porcupine <span class="pending">soon</span>
- Elle and Adya cycles: checking isolation from histories <span class="pending">soon</span>
- Replica consistency verification: checksums, pt-table-checksum and repair <span class="pending">soon</span>
- Finding wrong answers: sqlsmith, SQLancer oracles and differential testing <span class="pending">soon</span>
- Model-based property testing: a reference model, generated sequences and shrinking <span class="pending">soon</span>
- Deterministic simulation testing: FoundationDB, TigerBeetle, Antithesis <span class="pending">soon</span>
- TLA+ and model checking protocols <span class="pending">soon</span>
- Famous Jepsen findings and the bug classes behind them <span class="pending">soon</span>

## Part 9 — Real-System Case Studies

One page per engine, tying every mechanism module to how a shipping system actually assembles storage, query, transactions and replication.

### 9.1 Case Studies: Relational, Embedded and Disaggregated Engines

*expert* — PostgreSQL, InnoDB, SQLite, DuckDB, RocksDB and Aurora are the engines seniors actually run, and each is a distinct design point; every page here assembles mechanisms already taught in Parts 2-8 into a whole engine and never re-derives them, so a case study only adds the engine's own configuration, its seams and the operational consequences of its choices.

- PostgreSQL end to end <span class="pending">soon</span>
- Extending PostgreSQL: extensions, hooks and access methods <span class="pending">soon</span>
- MySQL and InnoDB end to end <span class="pending">soon</span>
- SQLite <span class="pending">soon</span>
- SQLite everywhere: WAL shipping, embedded replicas and local-first sync <span class="pending">soon</span>
- DuckDB <span class="pending">soon</span>
- RocksDB <span class="pending">soon</span>
- LMDB, WiredTiger, Badger and Redwood: same trace, different engines <span class="pending">soon</span>
- Aurora and Neon as an operator <span class="pending">soon</span>
- Vitess: sharding MySQL <span class="pending">soon</span>

### 9.2 Case Studies: Distributed OLTP, Wide-Column and NewSQL Stores

*expert* — Bigtable, Cassandra, DynamoDB, Cosmos DB, MongoDB, graph engines, CockroachDB/TiDB, Spanner and FoundationDB each pick a different point in the operational design space — master vs masterless, quorum vs consensus, single-row atomicity vs strict serializability, pointer chase vs shard hop, and in the cloud offerings a metered currency that prices every one of those choices — and seeing how each assembles the storage, replication and transaction mechanisms of earlier modules into a shipping system is what lets a senior engineer predict its failure modes, read its Jepsen report and choose it for the right job.

- Bigtable and HBase <span class="pending">soon</span>
- Cassandra and ScyllaDB <span class="pending">soon</span>
- DynamoDB <span class="pending">soon</span>
- Azure Cosmos DB: request units, partitions and five consistency levels <span class="pending">soon</span>
- MongoDB and WiredTiger <span class="pending">soon</span>
- Neo4j, JanusGraph and distributed traversal <span class="pending">soon</span>
- Serving a social graph: objects, associations and the cache tier <span class="pending">soon</span>
- CockroachDB and TiDB <span class="pending">soon</span>
- Spanner <span class="pending">soon</span>
- FoundationDB <span class="pending">soon</span>

### 9.3 Case Studies: Streaming, Search, Caching and Real-Time Analytics

*expert* — Redis, Kafka, Elasticsearch, ClickHouse, Druid/Pinot, the time-series engines and the Spark job that feeds half of them are what engineers reach for when the system of record is not enough — a cache, a log, a derived search index, a warehouse, a sub-second serving tier, a metrics database, a batch compute engine — and each one buys its speed by giving something up, so knowing exactly which mechanism pays for that speed is what keeps a derived store from being trusted as a system of record and a tuning knob from being turned by superstition.

- Redis <span class="pending">soon</span>
- Kafka storage: segments, indexes and tiered storage <span class="pending">soon</span>
- Elasticsearch and Lucene <span class="pending">soon</span>
- ClickHouse <span class="pending">soon</span>
- Apache Druid and Pinot: real-time OLAP serving <span class="pending">soon</span>
- Prometheus TSDB and TimescaleDB: time-series engines <span class="pending">soon</span>
- Apache Spark as an engine: executors, memory regions and shuffle files <span class="pending">soon</span>

## Part 10 — Production Practice

Securing, operating, tuning, sizing, backing up and recovering databases, plus the incidents that recur and what postmortems teach.

### 10.1 Provisioning, Platform and Fleet Lifecycle

*advanced* — The curriculum jumps from design straight to operating a database that already exists: nothing covers how it came to be, what the managed service takes away, what the network path between client and engine costs, how the bill decomposes, which account quota or missing AZ capacity will block the replacement instance at 3 a.m., why extensions block a major upgrade, how other teams get a database at all without each inventing its own topology, how a change reaches a thousand tenant databases, or how anything is ever retired — yet these are where unplanned restarts, accidental instance replacement, unowned unbacked-up instances and irreversible deletions actually come from.

- Provisioning as code: IaC, parameter groups, secrets and drift <span class="pending">soon</span>
- Managed databases: what RDS, Aurora and Cloud SQL take away <span class="pending">soon</span>
- Provider quotas, capacity and the limits you meet at 3 a.m. <span class="pending">soon</span>
- The network path to the database <span class="pending">soon</span>
- Cloud database cost models and the levers that move them <span class="pending">soon</span>
- Extensions and plugins: install, upgrade and the version trap <span class="pending">soon</span>
- Running a fleet: staged rollouts, resumable fan-out and drift detection <span class="pending">soon</span>
- Databases as a paved road: self-service provisioning, guardrails and ownership <span class="pending">soon</span>
- Decommissioning: proving it is unused, then taking it apart safely <span class="pending">soon</span>

### 10.2 Security and Access Control: Identity, Privileges and Untrusted Callers

*advanced* — Who can reach the listener, who the server believes they are, what privileges that identity carries, and what an attacker or a careless human can do once a statement reaches the parser are each a concrete mechanism inside the engine (a host-rule matcher, a SCRAM exchange, a privilege bitmask on a catalog row, a parse tree built from concatenated text, a search_path lookup in a definer function), and a senior engineer must know their exact semantics and bypass paths because exposed listeners, injected statements, standing human access and privilege-escalation chains are how real database breaches actually happen.

- Authentication: hba rules, password schemes, certificates and short-lived credentials <span class="pending">soon</span>
- Authorization: roles, ownership, grants, default privileges and least privilege <span class="pending">soon</span>
- Exposure and hardening: the database's attack surface <span class="pending">soon</span>
- Rotating credentials and certificates without an outage <span class="pending">soon</span>
- Untrusted input in SQL: parameterization, dynamic SQL and defense in depth <span class="pending">soon</span>
- Human access to production: break-glass, four-eyes DML and session recording <span class="pending">soon</span>
- Privilege escalation paths inside the database <span class="pending">soon</span>

### 10.3 Data Protection: Row-Level Access, Encrypted Data and Tenant Isolation

*advanced* — Once identities and privileges are settled, the remaining question is what each caller may see of the data itself — which rows a predicate lets through, what a ciphertext still reveals to the server holding it, what an agent can pull out of a sandbox, and which tenant's workload can starve another's — and every one of these is a mechanism with an exact failure mode (a predicate the rewriter injects, a userset traversal served from a stale snapshot, a frequency histogram over deterministic ciphertexts, a shared buffer pool no quota can partition) that produces the cross-tenant leaks and data-exposure incidents senior engineers are expected to design against.

- Row-level security and data masking <span class="pending">soon</span>
- Encryption in transit and at rest: TLS, TDE and key management <span class="pending">soon</span>
- Searching encrypted columns: blind indexes, structured encryption and what leaks <span class="pending">soon</span>
- Zanzibar-style authorization stores: tuples, checks and the new-enemy problem <span class="pending">soon</span>
- Agents with SQL access: least privilege, budgets and injection through data <span class="pending">soon</span>
- Multi-tenant isolation and noisy neighbors <span class="pending">soon</span>

### 10.4 Compliance, Auditing and Data Governance

*advanced* — Retention, residency, anonymization and right-to-erasure obligations are not policy documents but concrete data-plane work — you cannot erase what you cannot find, a dataset with the names stripped out is not anonymous, a residency promise holds only if every replica, backup and support session obeys it, an audit trail that a privileged operator can silently edit proves nothing in the dispute it exists for, and a recorded access nobody alerts on is how a valid credential walks off with the customer table — so a senior engineer has to know the discovery, lineage, placement, retention-vs-hold, privacy, misuse-detection and tamper-evidence mechanisms, the honest limits of each, and how to turn them into evidence a third party will accept, before promising a regulator or a customer anything.

- Finding the PII: classification, catalogs and column lineage <span class="pending">soon</span>
- Anonymization that holds: k-anonymity, l-diversity and differential privacy <span class="pending">soon</span>
- Auditing, PII lifecycle and right-to-erasure <span class="pending">soon</span>
- Detecting misuse: bulk reads, anomalous access and alerting on query telemetry <span class="pending">soon</span>
- Retention, legal hold and subject access requests in conflict <span class="pending">soon</span>
- Data residency and sovereignty: pinning rows, replicas, backups and reader location <span class="pending">soon</span>
- Tamper-evident audit: hash chains, Merkle proofs and ledger tables <span class="pending">soon</span>
- Audit readiness: mapping controls to database evidence <span class="pending">soon</span>

### 10.5 Operating Databases: Observability and Tuning

*expert* — Every dashboard number, wait event and configuration knob is a window onto a mechanism you have already studied; this module maps the signals back to those mechanisms so you can diagnose a live system instead of pattern-matching, turns the signals into SLOs, pages and runbooks worth waking a human for, covers the scheduled maintenance clockwork whose silent failure is its own outage, and then works down the knob stack from engine parameters to the kernel and filesystem settings underneath them — ending with the automatic tuners that now turn those knobs for you and must be supervised.

- Metrics that matter: signals mapped to mechanisms <span class="pending">soon</span>
- SLOs, burn-rate alerting and the runbook <span class="pending">soon</span>
- Query-level telemetry and wait events <span class="pending">soon</span>
- The database log as an operational artifact <span class="pending">soon</span>
- Monitoring vacuum, bloat, wraparound and history lists <span class="pending">soon</span>
- Scheduled maintenance: partition creation, retention and reindex windows <span class="pending">soon</span>
- Host, kernel and filesystem tuning for database servers <span class="pending">soon</span>
- Tuning knobs mapped to mechanisms <span class="pending">soon</span>
- Self-tuning claims: automatic indexing, knob tuners and autonomous databases <span class="pending">soon</span>

### 10.6 Running Databases: Memory, Capacity, Storage and Platforms

*expert* — Once you can read a database's signals and turn its knobs, the next layer of operational risk is resources and substrate: how RAM is really partitioned and who the OOM killer picks, which resource runs out first and what headroom costs, what an SRE actually does to the volume when the disk fills, where cold data goes when deleting it is not an option, and what a managed provider or a Kubernetes scheduler quietly took away from you.

- Memory sizing and the OOM killer <span class="pending">soon</span>
- Capacity planning and cost modeling <span class="pending">soon</span>
- Growing, moving and reclaiming storage without downtime <span class="pending">soon</span>
- Archival and tiering: keeping cold data cheap and still reachable <span class="pending">soon</span>
- Running on managed services: what the provider controls and what it hides <span class="pending">soon</span>
- Databases on Kubernetes: operators, volumes and eviction <span class="pending">soon</span>

### 10.7 Failover Stacks, the Pooler Tier and Upgrades

*expert* — The last operational risks sit between the client and the engine and at the moments the engine changes version: the pooler tier every connection passes through and whose failure takes down every database behind it, the automation that actually performs your failovers and the routing layer that decides who the primary is, the security advisory that turns an upgrade into a deadline you did not choose, and the one-way doors of major-version upgrades, OS collation changes that silently corrupt indexes, and mixed-version clusters that pass the point of no return.

- Operating the pooler tier: PgBouncer and ProxySQL in production <span class="pending">soon</span>
- HA stacks in practice: Patroni, Orchestrator and the routing layer <span class="pending">soon</span>
- Security patches: CVE triage and the emergency upgrade path <span class="pending">soon</span>
- Major-version upgrades: pg_upgrade, logical cutover and blue/green <span class="pending">soon</span>
- The collation trap: silent index corruption after an OS upgrade <span class="pending">soon</span>
- Rolling upgrades and compatibility in clusters <span class="pending">soon</span>

### 10.8 Backups, Restore and Disaster Recovery

*expert* — A backup is only a hypothesis until a restore proves it, and the kind of restore you can actually perform — full, partial, cross-shard, cross-region — decides how much data and how many hours an outage costs.

- Logical backups and their limits <span class="pending">soon</span>
- Physical backups and point-in-time recovery <span class="pending">soon</span>
- Restore testing and backup validation <span class="pending">soon</span>
- Partial recovery: restoring one table or one tenant without rolling back the cluster <span class="pending">soon</span>
- Disaster recovery: RPO/RTO and multi-region failover drills <span class="pending">soon</span>
- Degrading gracefully: read-only mode when the primary is gone <span class="pending">soon</span>
- Consistent backups across shards and replicas <span class="pending">soon</span>

### 10.9 Incident Response: Classic Failures, Data Repair and Corruption

*expert* — Database incidents are the ones where the cure can destroy more than the failure did, so the senior skill is triage under pressure plus a protocol that keeps the recovery path and the evidence alive.

- Classic incidents: locks, connections and queues <span class="pending">soon</span>
- Classic incidents: capacity, disk full and replication lag <span class="pending">soon</span>
- Ad-hoc production data fixes and undoing a bad one <span class="pending">soon</span>
- Running a data incident: roles, severity and the irreversible-action gate <span class="pending">soon</span>
- Retry storms, load shedding and backpressure in practice <span class="pending">soon</span>
- Corruption triage: detect, contain, salvage and forced recovery <span class="pending">soon</span>
- Learning from postmortems: GitHub 2018, GitLab 2017, Cloudflare <span class="pending">soon</span>

## Part 11 — Senior Engineer Judgement

Performance literacy, tradeoff reasoning and misconceptions, then capstone designs, toy-engine builds and the hardware frontier.

### 11.1 Performance Literacy: Measurement, Benchmarks and Profiling

*expert* — Seniors reason from percentiles, queues and measured evidence rather than folklore, and can tell a real speedup from noise and a vendor's headline number from the workload it quietly excludes.

- Latency, throughput and percentiles; coordinated omission <span class="pending">soon</span>
- Queueing theory, Little's law and the Universal Scalability Law <span class="pending">soon</span>
- Amortized cost versus tail latency: the potential method and deamortization <span class="pending">soon</span>
- Benchmarks and what they measure: TPC-C, YCSB, TPC-H <span class="pending">soon</span>
- Benchmarks for search, vector, time-series and streaming <span class="pending">soon</span>
- Is that speedup real? Variance, confidence intervals and A/B on database changes <span class="pending">soon</span>
- Misconceptions catalog <span class="pending">soon</span>
- Profiling a database: perf, flame graphs and wait analysis <span class="pending">soon</span>

### 11.2 Tradeoff Reasoning: Bottlenecks, Boundaries and Design Review

*expert* — Seniors can name the next bottleneck before it appears, say where a schema may be cut and what that cut costs, and read a design or a paper for the invariant that breaks rather than the features it lists.

- Service boundaries and data ownership: shared database versus database per service <span class="pending">soon</span>
- Scaling reads and writes: the movable bottleneck <span class="pending">soon</span>
- Choosing and integrating systems <span class="pending">soon</span>
- Design reviews and reading papers: the invariant hunt <span class="pending">soon</span>
- System design interview playbook <span class="pending">soon</span>

### 11.3 Capstones: Worked System Designs

*expert* — Applying every mechanism to whole product systems — the order, booking, money, metering, in-product analytics, tenancy and authorization designs a senior engineer is actually asked to build, defend and inherit — turns knowledge into judgement, and running them all as presets over one shared simulator makes each design's invariants, bottlenecks and failure modes visible instead of asserted.

- The capstone preset library and assertion format <span class="pending">soon</span>
- E-commerce OLTP at scale <span class="pending">soon</span>
- Booking and availability: holds, overlaps and capacity <span class="pending">soon</span>
- Double-entry ledgers and money movement <span class="pending">soon</span>
- Usage metering and billing: exact counts, late events and frozen invoices <span class="pending">soon</span>
- In-product analytics: per-tenant dashboards on operational data <span class="pending">soon</span>
- Global multi-tenant SaaS <span class="pending">soon</span>
- Authorization data: ACLs, roles and relationship checks <span class="pending">soon</span>

### 11.4 Capstones: Feeds, Messaging and Delivery

*expert* — The fan-out family — feeds, recommendations, chat, notifications and outbound webhooks — shares one storage problem the OLTP capstones never hit: per-user or per-destination state that is written far more often than it is read, skewed by celebrities, mega-groups and dead endpoints, and correct only if ordering, dedupe and delivery state are designed deliberately, so each design runs as another preset on the same simulator with its own invariants and its own signature hot spot.

- Social feed and hot keys <span class="pending">soon</span>
- Recommendation serving: candidates, already-seen filtering and feedback <span class="pending">soon</span>
- Chat and messaging storage <span class="pending">soon</span>
- Notifications: fan-out, dedupe windows and delivery state <span class="pending">soon</span>
- Outbound webhooks: per-destination ordering, fairness and poison endpoints <span class="pending">soon</span>

### 11.5 Capstones: Data Platforms, Boundaries and Migrations

*expert* — The second half of the capstone set moves off the request path and onto the data platform and its lifecycle — ingestion, lakehouse, feature serving, and the two moves a senior is most often handed, splitting a shared database along service boundaries and migrating between engines — where correctness is judged by what survives a backfill, a cutover and a late event rather than by a single request's latency.

- IoT and time-series ingestion <span class="pending">soon</span>
- Analytics platform and lakehouse <span class="pending">soon</span>
- Feature store: online/offline parity and point-in-time joins <span class="pending">soon</span>
- Splitting a shared database: ownership boundaries, broken joins and staged cutover <span class="pending">soon</span>
- Migrating between systems <span class="pending">soon</span>

### 11.6 Capstones II: Retrieval, Toy Engines and the Frontier

*expert* — Assembling the search and retrieval stacks nobody hands you a recipe for, then building the storage and consensus engines by hand and looking at where access methods and hardware go next, converts understanding of mechanisms into the ability to build one and to judge what is new.

- Autocomplete and typeahead <span class="pending">soon</span>
- Nearby search: geospatial queries at scale <span class="pending">soon</span>
- Retrieval service: keeping indexes true to the source <span class="pending">soon</span>
- Build a toy KV store, B-tree and LSM <span class="pending">soon</span>
- Build a toy Raft <span class="pending">soon</span>
- Learned indexes and adaptive indexing <span class="pending">soon</span>
- New hardware and future engines: ZNS, CXL, GPU and DPU <span class="pending">soon</span>

## Part 12 — Reference, Playbooks and Decision Guides

The lookup half of the site: a glossary that disambiguates the field's overloaded words, notation and anchor-number cards, theorem-to-consequence rules, engine and managed-platform comparison matrices, symptom-first on-call playbooks, and an annotated reading map. Nothing here introduces a mechanism — every entry states a fact, a rule or a diagnostic and links to the module that derives it, so the site can be entered sideways from a search engine or mid-incident rather than only read front to back.

### 12.1 Reference Cards: Glossary, Notation, Numbers and Laws

*core* — A learner who lands on the SSI, LSM or ISR page from a search engine needs to decode pageLSN, xmin, q-error, ef or fencing token without reading four prerequisite modules, and a senior in a design review needs the impossibility results as production rules rather than proofs; this module is the site's single authoritative vocabulary, notation, anchor-number and theorem reference.

- Glossary and the ambiguous-term decoder <span class="pending">soon</span>
- Notation index and cross-engine synonyms <span class="pending">soon</span>
- Probability toolkit: skew, collisions and tail bounds on one card <span class="pending">soon</span>
- Theorem-to-consequence cards: what each result forbids <span class="pending">soon</span>
- Napkin estimation: rows, bytes, QPS and IOPS <span class="pending">soon</span>
- Anchor numbers by workload: bytes per document, vector, sample and message <span class="pending">soon</span>
- Cost and complexity cheat sheet: every operator in the I/O model <span class="pending">soon</span>
- What breaks at each order of magnitude <span class="pending">soon</span>

### 12.2 Decision Guides: Choosing Stores, Freshness and Placement

*advanced* — The question a senior is actually asked is 'what do I use for this, and can the database we already run do it?', and today that answer is spread across forty modules; these guides answer it in one page with concrete thresholds, hard limits and the specific reason each runner-up was ruled out, then link back to the mechanism that explains the limit.

- Which store for which workload <span class="pending">soon</span>
- Choosing freshness: batch, micro-batch, incremental or streaming <span class="pending">soon</span>
- From question to mechanism: the design-question router <span class="pending">soon</span>
- In the database or in the application? <span class="pending">soon</span>

### 12.3 Comparison Matrices: Engines, Platforms, Formats and Claims

*advanced* — Once the workload has picked a shape, the remaining questions are all lookups a senior must answer precisely — does this engine support what we need, will this managed service take it away, will the version we run allow it, may we legally ship it, and does the vendor's word mean what it sounds like — so all of them share one MatrixExplorer component and one interaction (requirement filter, grey-out with the failing limit named, cell detail, jump link to the mechanism module), and the capability page ends by driving the learner's own numbers into a limit until it breaks.

- Capability and limits matrix by data type <span class="pending">soon</span>
- Engine comparison tables: defaults, limits and dialect differences <span class="pending">soon</span>
- Table format and catalog capability matrix: can your engine read what you just wrote <span class="pending">soon</span>
- Version-delta cards: which release unlocked which design <span class="pending">soon</span>
- Managed platform matrix: what each service takes away <span class="pending">soon</span>
- Licenses, forks and the managed-service clause <span class="pending">soon</span>
- The vendor-claim decoder <span class="pending">soon</span>

### 12.4 Playbooks: Symptom-First Triage

*advanced* — At 3 a.m. nobody navigates a curriculum by mechanism — they navigate by symptom, and a senior needs the ordered diagnostics, the discriminating signal, the mitigation and its side effects in one place, with every branch linking back to the module that owns the mechanism so the playbook teaches rather than merely instructs.

- The on-call router: symptom to mechanism index <span class="pending">soon</span>
- Playbook: you just inherited this database <span class="pending">soon</span>
- Playbook: the database is at 100% CPU <span class="pending">soon</span>
- Playbook: will it finish, and what does cancelling cost? <span class="pending">soon</span>
- Playbook: the database will not start <span class="pending">soon</span>
- Playbook: someone deleted the data <span class="pending">soon</span>

### 12.5 Playbooks: Wrong Data, Broken Pipelines and After-the-Fact Forensics

*advanced* — The second half of on-call is not a database that is slow or down but one that is up and answering wrongly — an error code the client must decide how to retry, an anomaly with a name, rows that vanished somewhere in a five-hop pipeline, a bill and a tail latency governed by object-store request classes — plus the incident that is already over and owes an RCA tomorrow, where the only evidence is whatever telemetry someone configured in advance.

- Error-code triage: SQLSTATE, MySQL and driver errors by response <span class="pending">soon</span>
- Anomaly triage: from symptom to named anomaly <span class="pending">soon</span>
- Playbooks for specialized workloads <span class="pending">soon</span>
- Playbook: the incident is over — reconstructing what happened <span class="pending">soon</span>
- Playbook: the rows are missing downstream <span class="pending">soon</span>
- Playbook: latency and cost in disaggregated and serverless systems <span class="pending">soon</span>

### 12.6 Lifecycle Checklists and Review Gates

*advanced* — Part 12 tells you how to choose a system and what to do when one breaks, but nothing covers the gates in between: the evidence a database must produce before it carries real traffic, the recurring work that keeps that evidence true, and the classification that decides how much ceremony a proposed change deserves. Seniors are asked for exactly these artifacts at launch reviews, change-approval meetings and audits, and their absence is the shared root cause of the classic incidents — a backup configured but never restored, a failover never drilled, a certificate nobody re-checked, a routine ALTER that turned out to rewrite a terabyte at peak. Every line here is stated as required evidence or as a predicted failure, and links back to the module that derives it.

- Production readiness review: the go-live checklist <span class="pending">soon</span>
- Evidence, not assertion: auditing a readiness claim <span class="pending">soon</span>
- The operational cadence: what must be re-proved and how often <span class="pending">soon</span>
- Change-risk classification: blast radius, reversibility and one-way doors <span class="pending">soon</span>
- Safeguards and timing: lock budgets, throttles, canaries and freeze windows <span class="pending">soon</span>
- Running the review: the questions that catch the bad changes <span class="pending">soon</span>
- From incident to gate: turning postmortems into checklist lines <span class="pending">soon</span>

### 12.7 The Reading Map: Canon, Modern Sources and What Changed

*advanced* — The site is drawn from a specific literature and a senior's growth path is reading it — but only in the right order, for the right takeaway, and knowing which claims have since been superseded; the value a curriculum adds over a bibliography is exactly that annotation, so the site should own it rather than sending people to a stale blog post.

- The annotated canon: papers and practitioner books in reading order <span class="pending">soon</span>
- The modern reading list, 2020–2026 <span class="pending">soon</span>
- What changed since DDIA: a 2022–2026 orientation map <span class="pending">soon</span>
- Reading a systems paper critically, and where to read source <span class="pending">soon</span>

### 12.8 The Visual System: Shared Primitives, Presets and Build Order

*foundations* — Dozens of interactive visuals across the curriculum are specified in terms of shared components that no module owns — PageView, the TimelineEditor and its precedence/Adya/version-chain graph builders, the space-time message-lane diagram that roughly forty-five distributed subtopics draw, the ring/topology view, the thread-contention animator, ScanLab, the site-wide engine selector that must persist across every comparison page, the toy SQL parser plus instrumented pull executor and planner, and the capstone simulator runtime — while at least seven pages render numbers from data generated offline and committed, and the misconceptions deck already depends on a site-wide rule (a serialized preset passed as a single state prop and reflected in the URL) that is named nowhere. If these are not specified as one contract before module 1 is built, each module reinvents its own lane renderer and its own state shape, every declared reuse silently becomes a rebuild, presets cannot be exchanged between pages, and committed fixtures rot with nobody noticing. This module is the site's engineering appendix and its build order: it reads last and is built first.

- Shared conventions: glyphs, status encoding and the honesty rule <span class="pending">soon</span>
- The preset and fixture contracts: serialized state, versioned data and provenance <span class="pending">soon</span>
- PageView: slot map, byte grid, hex decode and fill heat <span class="pending">soon</span>
- TimelineEditor and the graph builders <span class="pending">soon</span>
- MessageTimeline: node lanes, delivery faults and annotation overlays <span class="pending">soon</span>
- RingView and ContentionLab: placement and thread contention <span class="pending">soon</span>
- ScanLab and the site-wide engine selector <span class="pending">soon</span>
- The toy engine kit: parser, instrumented executor and planner <span class="pending">soon</span>
- The capstone simulator runtime <span class="pending">soon</span>
- Build order, ownership and the reuse ledger <span class="pending">soon</span>
