import { useMemo, useState } from 'react';
import {
  VizPanel,
  Slider,
  Segmented,
  Legend,
  Stats,
  Note,
  TooltipHost,
  useTip,
  fmtBytes,
  fmtTime,
  fmtNum,
  makeRng,
  useSize,
} from './Viz';

/**
 * Why one workload gets PostgreSQL and the next gets ClickHouse.
 *
 * Every card is one storage family with one small closed-form cost model per access
 * shape. The models share three constants — a cold random 8 KB read, a streaming scan
 * rate, and a per-value CPU cost — so the cards are comparable, and every term is
 * printed next to the bar. The point of the thing is the crossovers: push selectivity
 * or columns-touched past a threshold and the ranking inverts, because a row store
 * reads whole rows (bytes ∝ rows × row width) while a column store reads referenced
 * columns (bytes ∝ rows × columns touched × column width ÷ compression).
 *
 * This is a teaching model, not a benchmark: the constants are order-of-magnitude
 * honest, the arithmetic is exact and deterministic, and there is no I/O concurrency,
 * no buffer-pool warmth and no parallelism in it.
 */

/* ------------------------------------------------------------- constants */

const PAGE = 8192; // Postgres BLCKSZ
const RAND_NS = 25_000; // one cold random 8 KB read on NVMe
const BW = 2e9; // bytes/s of streaming scan off the same drive
const GRANULE = 8192; // ClickHouse index_granularity
const TS_CHUNK = 120; // samples per Prometheus head chunk
const ROW_CPU = 60; // ns to deform one row-store tuple and evaluate a predicate
const VEC_CPU = 2; // ns per value in a vectorized column engine
const POST_CPU = 20; // ns per posting-list entry
const DOCVAL_CPU = 10; // ns per doc-values value
const DIST_CPU = 500; // ns for one 768-dimension float distance
const BTREE_ENTRY = 8_000; // ns to place one entry in a B+tree leaf (random read-modify-write)
const LSM_ENTRY = 1_200; // ns per LSM / segment entry, amortized compaction included
const COMMIT = 40_000; // ns of WAL flush per commit, already group-committed
const RPC = 120_000; // ns for one in-datacenter round trip
const COL_CF = 4; // columnar compression ratio on general data
const TS_CF = 10; // delta + XOR compression on numeric time series
const VEC_BYTES = 768 * 4; // one 768-d float32 embedding
const HNSW_M = 16; // pgvector default m
const HNSW_EF = 40; // pgvector default hnsw.ef_search

/* ---------------------------------------------------------------- shapes */

type Shape = 'point' | 'range' | 'aggregate' | 'fulltext' | 'similarity' | 'append';

const SHAPES: readonly { value: Shape; label: string; title: string }[] = [
  { value: 'point', label: 'Point', title: 'SELECT * FROM t WHERE pk = ?' },
  { value: 'range', label: 'Range', title: 'SELECT a, b, c FROM t WHERE k BETWEEN ? AND ?' },
  { value: 'aggregate', label: 'Aggregate', title: 'SELECT k, sum(x) FROM t WHERE … GROUP BY k' },
  { value: 'fulltext', label: 'Full text', title: "WHERE to_tsvector(body) @@ to_tsquery('…')" },
  { value: 'similarity', label: 'Similarity', title: 'ORDER BY embedding <-> ? LIMIT 10' },
  { value: 'append', label: 'Append stream', title: 'INSERT in arrival order; read the tail' },
];

/* -------------------------------------------------------------- families */

type FamId = 'rowstore' | 'column' | 'htap' | 'search' | 'vector' | 'tsdb' | 'cache' | 'log';

type Fam = {
  id: FamId;
  name: string;
  products: string;
  color: string;
  setupNs: number; // fixed per-query cost: parse, plan, worker start, segment open
  slot: 'sor' | 'log' | 'analytics' | 'search' | 'vector' | 'cache';
};

const FAMS: readonly Fam[] = [
  {
    id: 'rowstore',
    name: 'Row store + B+tree',
    products: 'PostgreSQL, InnoDB, SQLite, Oracle',
    color: 'var(--viz-1)',
    setupNs: 30_000,
    slot: 'sor',
  },
  {
    id: 'column',
    name: 'Column store',
    products: 'ClickHouse, DuckDB, Snowflake, BigQuery',
    color: 'var(--viz-2)',
    setupNs: 5_000_000,
    slot: 'analytics',
  },
  {
    id: 'htap',
    name: 'HTAP dual format',
    products: 'TiDB + TiFlash, SingleStore, Oracle In-Memory',
    color: 'var(--viz-3)',
    setupNs: 1_000_000,
    slot: 'sor',
  },
  {
    id: 'search',
    name: 'Inverted index',
    products: 'Elasticsearch, OpenSearch, Postgres GIN',
    color: 'var(--viz-4)',
    setupNs: 2_000_000,
    slot: 'search',
  },
  {
    id: 'vector',
    name: 'Vector ANN index',
    products: 'pgvector HNSW, Milvus, Qdrant',
    color: 'var(--viz-5)',
    setupNs: 500_000,
    slot: 'vector',
  },
  {
    id: 'tsdb',
    name: 'Time-series store',
    products: 'TimescaleDB, InfluxDB, Prometheus',
    color: 'var(--viz-6)',
    setupNs: 2_000_000,
    slot: 'analytics',
  },
  {
    id: 'cache',
    name: 'Key-value cache',
    products: 'Redis, Memcached',
    color: 'var(--viz-7)',
    setupNs: 0,
    slot: 'cache',
  },
  {
    id: 'log',
    name: 'Append-only log',
    products: 'Kafka, Redpanda, Pulsar',
    color: 'var(--viz-8)',
    setupNs: 300_000,
    slot: 'log',
  },
];

const FAM_BY_ID = new Map(FAMS.map((f) => [f.id, f] as const));

/* ----------------------------------------------------------- the model */

type In = {
  shape: Shape;
  rows: number;
  cols: number;
  colBytes: number;
  touch: number;
  sel: number;
  wshare: number; // 0..1
  sidx: number; // secondary indexes on the table
};

type Cost = {
  ok: boolean;
  why: string;
  plan: string;
  rowsScanned: number;
  bytes: number;
  ios: number;
  wpr: number; // index / chunk writes per row written
  readNs: number;
  writeNs: number;
  totalNs: number;
  readTerms: string;
  writeTerms: string;
};

type ReadSpec = {
  plan: string;
  ios: number;
  bytes: number;
  units?: number;
  unitNs?: number;
  unitName?: string;
  extraNs?: number;
  extraLabel?: string;
};

function mkRead(f: Fam, r: ReadSpec) {
  const units = r.units ?? 0;
  const unitNs = r.unitNs ?? 0;
  const cpu = units * unitNs;
  const scan = (r.bytes / BW) * 1e9;
  const extra = r.extraNs ?? 0;
  const ns = f.setupNs + r.ios * RAND_NS + scan + cpu + extra;
  const parts: string[] = [];
  if (f.setupNs > 0) parts.push(`${fmtTime(f.setupNs)} setup`);
  if (extra > 0) parts.push(`${fmtTime(extra)} ${r.extraLabel ?? ''}`.trim());
  if (r.ios > 0) parts.push(`${fmtNum(r.ios)} rnd IO × ${fmtTime(RAND_NS)}`);
  if (r.bytes > 0) parts.push(`${fmtBytes(r.bytes)} ÷ 2 GB/s`);
  if (cpu > 0) parts.push(`${fmtNum(units)} ${r.unitName ?? 'rows'} × ${fmtTime(unitNs)}`);
  return { ns, terms: `${r.plan} — ${parts.join(' + ')} = ${fmtTime(ns)}` };
}

type WriteSpec = { wpr: number; entryNs: number; entryName: string; fixedNs?: number; fixedLabel?: string };

function mkWrite(w: WriteSpec) {
  const fixed = w.fixedNs ?? 0;
  const ns = w.wpr * w.entryNs + fixed;
  const parts: string[] = [];
  if (w.wpr > 0 && w.entryNs > 0) parts.push(`${fmtNum(w.wpr)} ${w.entryName} × ${fmtTime(w.entryNs)}`);
  if (fixed > 0) parts.push(`${fmtTime(fixed)} ${w.fixedLabel ?? ''}`.trim());
  return { ns, terms: `${parts.join(' + ')} = ${fmtTime(ns)} per row written` };
}

const NOPE = (why: string): Cost => ({
  ok: false,
  why,
  plan: '—',
  rowsScanned: 0,
  bytes: 0,
  ios: 0,
  wpr: 0,
  readNs: Infinity,
  writeNs: Infinity,
  totalNs: Infinity,
  readTerms: why,
  writeTerms: '—',
});

function clamp(n: number, lo: number, hi: number) {
  return Math.max(lo, Math.min(hi, n));
}

/** The ANN search cost, shared by the vector card and by its own insert path. */
function annSearch(f: Fam, rows: number) {
  const layers = Math.max(1, Math.ceil(Math.log(rows) / Math.log(HNSW_M)));
  const visited = HNSW_EF + HNSW_M * layers;
  return {
    visited,
    ...mkRead(f, {
      plan: `HNSW traversal, ef_search=${HNSW_EF}, m=${HNSW_M}`,
      ios: visited,
      bytes: visited * VEC_BYTES,
      units: visited,
      unitNs: DIST_CPU,
      unitName: 'distances',
    }),
  };
}

function cost(f: Fam, i: In): Cost {
  const W = i.cols * i.colBytes;
  const T = Math.min(i.touch, i.cols);
  const M = i.shape === 'point' ? 1 : clamp(Math.round(i.rows * i.sel), 1, i.rows);
  const ordered = i.shape === 'append'; // the filter follows insertion order
  const mutate = i.shape !== 'append'; // writes are updates by key, not appends
  const tablePages = Math.ceil((i.rows * W) / PAGE);

  const fin = (r: { ns: number; terms: string }, w: { ns: number; terms: string }, m: Partial<Cost>): Cost => ({
    ok: true,
    why: '',
    plan: m.plan ?? '',
    rowsScanned: m.rowsScanned ?? 0,
    bytes: m.bytes ?? 0,
    ios: m.ios ?? 0,
    wpr: m.wpr ?? 0,
    readNs: r.ns,
    writeNs: w.ns,
    totalNs: (1 - i.wshare) * r.ns + i.wshare * w.ns,
    readTerms: r.terms,
    writeTerms: w.terms,
  });

  switch (f.id) {
    /* ------------------------------------------------------------ row store */
    case 'rowstore': {
      const fanout = clamp(Math.floor(PAGE / (i.colBytes + 8)), 2, 400);
      const h = Math.max(1, Math.ceil(Math.log(i.rows) / Math.log(fanout)));
      const w = mkWrite({
        wpr: 1 + i.sidx,
        entryNs: BTREE_ENTRY,
        entryName: 'index entries/row',
        fixedNs: COMMIT,
        fixedLabel: 'WAL commit',
      });
      const seq: ReadSpec = {
        plan: 'sequential scan',
        ios: 1,
        bytes: tablePages * PAGE,
        units: i.rows,
        unitNs: ROW_CPU,
        unitName: 'rows',
      };
      if (i.shape === 'point') {
        const r = mkRead(f, {
          plan: `index point lookup, height ${h}`,
          ios: h + 1,
          bytes: (h + 1) * PAGE,
          units: 1,
          unitNs: ROW_CPU,
          unitName: 'row',
        });
        return fin(r, w, { plan: 'index point lookup', rowsScanned: 1, bytes: (h + 1) * PAGE, ios: h + 1, wpr: 1 + i.sidx });
      }
      if (i.shape === 'fulltext') {
        const r = mkRead(f, { ...seq, plan: "sequential scan — no index serves LIKE '%…%'", unitNs: ROW_CPU * 3 });
        return fin(r, w, { plan: 'sequential scan', rowsScanned: i.rows, bytes: tablePages * PAGE, ios: 1, wpr: 1 + i.sidx });
      }
      if (i.shape === 'similarity') {
        const bytes = Math.ceil((i.rows * (W + VEC_BYTES)) / PAGE) * PAGE;
        const r = mkRead(f, {
          plan: 'sequential scan + exact distance',
          ios: 1,
          bytes,
          units: i.rows,
          unitNs: DIST_CPU,
          unitName: 'distances',
        });
        return fin(r, w, { plan: 'exact kNN scan', rowsScanned: i.rows, bytes, ios: 1, wpr: 1 + i.sidx });
      }
      // range / aggregate / append tail: the planner picks index vs sequential.
      const idxIos = ordered ? h + 1 : h + Math.min(M, tablePages);
      const idxBytes = ordered ? h * PAGE + Math.ceil((M * W) / PAGE) * PAGE : idxIos * PAGE;
      const idx: ReadSpec = {
        plan: ordered ? `index scan, clustered — height ${h}` : `index scan, ${fmtNum(M)} heap fetches`,
        ios: idxIos,
        bytes: idxBytes,
        units: M,
        unitNs: ROW_CPU,
        unitName: 'rows',
      };
      const a = mkRead(f, idx);
      const b = mkRead(f, seq);
      const useIdx = a.ns <= b.ns;
      const pick = useIdx ? a : b;
      return fin(pick, w, {
        plan: useIdx ? 'index scan' : 'sequential scan',
        rowsScanned: useIdx ? M : i.rows,
        bytes: useIdx ? idxBytes : tablePages * PAGE,
        ios: useIdx ? idxIos : 1,
        wpr: 1 + i.sidx,
      });
    }

    /* --------------------------------------------------------- column store */
    case 'column': {
      const w = mkWrite({
        wpr: i.cols,
        entryNs: LSM_ENTRY,
        entryName: 'column chunks/row',
        fixedNs: mutate ? 250_000 : 0,
        fixedLabel: 'part rewrite (no in-place update)',
      });
      if (i.shape === 'fulltext') {
        const bytes = (i.rows * i.colBytes) / COL_CF;
        const r = mkRead(f, {
          plan: 'full scan of the text column',
          ios: 1,
          bytes,
          units: i.rows,
          unitNs: ROW_CPU * 3,
          unitName: 'rows',
        });
        return fin(r, w, { plan: 'full column scan', rowsScanned: i.rows, bytes, ios: 1, wpr: i.cols });
      }
      if (i.shape === 'similarity') {
        const bytes = i.rows * VEC_BYTES; // embeddings do not compress
        const r = mkRead(f, {
          plan: 'brute-force distance over the vector column',
          ios: 1,
          bytes,
          units: i.rows,
          unitNs: DIST_CPU,
          unitName: 'distances',
        });
        return fin(r, w, { plan: 'brute-force kNN', rowsScanned: i.rows, bytes, ios: 1, wpr: i.cols });
      }
      const touched = i.shape === 'point' ? i.cols : T;
      const scanned = Math.max(GRANULE, Math.ceil(M / GRANULE) * GRANULE);
      const bytes = (scanned * touched * i.colBytes) / COL_CF;
      const r = mkRead(f, {
        plan:
          i.shape === 'point'
            ? `granule scan — index_granularity ${fmtNum(GRANULE)}, SELECT *`
            : `granule scan, ${fmtNum(touched)} of ${fmtNum(i.cols)} columns`,
        ios: touched,
        bytes,
        units: scanned * touched,
        unitNs: VEC_CPU,
        unitName: 'values',
      });
      return fin(r, w, { plan: 'granule scan', rowsScanned: scanned, bytes, ios: touched, wpr: i.cols });
    }

    /* ----------------------------------------------------------------- HTAP */
    case 'htap': {
      const row = cost(FAM_BY_ID.get('rowstore')!, i);
      const col = cost(FAM_BY_ID.get('column')!, i);
      const best = row.readNs <= col.readNs ? row : col;
      const which = row.readNs <= col.readNs ? 'row replica' : 'columnar replica';
      const ns = f.setupNs + best.readNs * 1.25;
      const r = {
        ns,
        terms: `optimizer routes to the ${which} — ${fmtTime(best.readNs)} × 1.25 remote read + ${fmtTime(
          f.setupNs,
        )} coordination = ${fmtTime(ns)}`,
      };
      const wns = row.writeNs * 1.6 + LSM_ENTRY;
      const w = {
        ns: wns,
        terms: `${fmtNum(1 + i.sidx)} index entries/row × ${fmtTime(BTREE_ENTRY)} + ${fmtTime(
          COMMIT,
        )} commit, × 1.6 replication + 1 columnar replica = ${fmtTime(wns)} per row written`,
      };
      return fin(r, w, {
        plan: `routed to the ${which}`,
        rowsScanned: best.rowsScanned,
        bytes: best.bytes,
        ios: best.ios,
        wpr: 1 + i.sidx + 1,
      });
    }

    /* ------------------------------------------------------- inverted index */
    case 'search': {
      const tokens = Math.max(1, Math.round(W / 6));
      const w = mkWrite({
        wpr: tokens + 1,
        entryNs: LSM_ENTRY,
        entryName: 'postings/row',
        fixedNs: mutate ? 200_000 : 0,
        fixedLabel: 'delete + full reindex of the document',
      });
      if (i.shape === 'similarity') {
        const ann = annSearch(f, i.rows);
        const ns = ann.ns * 1.4;
        return fin(
          { ns, terms: `${ann.terms} × 1.4 shard fan-out` },
          w,
          { plan: 'dense-vector kNN', rowsScanned: ann.visited, bytes: ann.visited * VEC_BYTES, ios: ann.visited, wpr: tokens + 1 },
        );
      }
      if (i.shape === 'point') {
        const r = mkRead(f, { plan: 'term lookup on _id + stored fields', ios: 2, bytes: W, units: 1, unitNs: ROW_CPU, unitName: 'doc' });
        return fin(r, w, { plan: 'term lookup', rowsScanned: 1, bytes: W, ios: 2, wpr: tokens + 1 });
      }
      if (i.shape === 'fulltext') {
        const bytes = M * 6 + 10 * W;
        const r = mkRead(f, {
          plan: 'posting list scan + top-10 stored-field fetch',
          ios: 3,
          bytes,
          units: M,
          unitNs: POST_CPU,
          unitName: 'postings',
        });
        return fin(r, w, { plan: 'posting list scan', rowsScanned: M, bytes, ios: 3, wpr: tokens + 1 });
      }
      if (i.shape === 'aggregate') {
        const bytes = M * T * i.colBytes; // doc values are columnar but barely compressed
        const r = mkRead(f, {
          plan: 'doc-values scan (columnar side of Lucene)',
          ios: T + 1,
          bytes,
          units: M * T,
          unitNs: DOCVAL_CPU,
          unitName: 'values',
        });
        return fin(r, w, { plan: 'doc-values aggregation', rowsScanned: M, bytes, ios: T + 1, wpr: tokens + 1 });
      }
      const bytes = M * 8 + M * W; // BKD postings, then _source for every hit
      const r = mkRead(f, {
        plan: 'BKD range + _source fetch per hit',
        ios: 3,
        bytes,
        units: M,
        unitNs: POST_CPU,
        unitName: 'hits',
      });
      return fin(r, w, { plan: 'BKD range scan', rowsScanned: M, bytes, ios: 3, wpr: tokens + 1 });
    }

    /* ----------------------------------------------------------- vector ANN */
    case 'vector': {
      const ann = annSearch(f, i.rows);
      const w = {
        ns: ann.ns,
        terms: `1 vector + ${HNSW_M} neighbour link updates/row; an insert runs a search = ${fmtTime(ann.ns)} per row written`,
      };
      if (i.shape === 'similarity') {
        return fin(ann, w, { plan: 'HNSW traversal', rowsScanned: ann.visited, bytes: ann.visited * VEC_BYTES, ios: ann.visited, wpr: 1 + HNSW_M });
      }
      if (i.shape === 'point') {
        const r = mkRead(f, { plan: 'primary-key fetch', ios: 1, bytes: W + VEC_BYTES, units: 1, unitNs: ROW_CPU, unitName: 'row' });
        return fin(r, w, { plan: 'point fetch', rowsScanned: 1, bytes: W + VEC_BYTES, ios: 1, wpr: 1 + HNSW_M });
      }
      if (i.shape === 'range' || i.shape === 'append') {
        const bytes = i.rows * W;
        const r = mkRead(f, {
          plan: 'payload filter — no ordered secondary index',
          ios: 1,
          bytes,
          units: i.rows,
          unitNs: ROW_CPU,
          unitName: 'rows',
        });
        return fin(r, w, { plan: 'payload scan', rowsScanned: i.rows, bytes, ios: 1, wpr: 1 + HNSW_M });
      }
      if (i.shape === 'aggregate') return NOPE('no aggregation engine — an ANN index returns ids and distances');
      return NOPE('no inverted index — full text is a different structure');
    }

    /* ------------------------------------------------------------ time series */
    case 'tsdb': {
      const w = mkWrite({
        wpr: 1,
        entryNs: LSM_ENTRY,
        entryName: 'chunk append/row',
        fixedNs: mutate ? 400_000 : 0,
        fixedLabel: 'decompress + rewrite the chunk',
      });
      if (i.shape === 'fulltext') return NOPE('no inverted index — labels are indexed, bodies are not');
      if (i.shape === 'similarity') return NOPE('no vector index');
      if (i.shape === 'point') {
        const bytes = (TS_CHUNK * i.colBytes) / TS_CF;
        const r = mkRead(f, {
          plan: `series index + one ${TS_CHUNK}-sample chunk`,
          ios: 3,
          bytes,
          units: TS_CHUNK,
          unitNs: VEC_CPU,
          unitName: 'samples',
        });
        return fin(r, w, { plan: 'series + chunk fetch', rowsScanned: TS_CHUNK, bytes, ios: 3, wpr: 1 });
      }
      const bytes = (M * T * i.colBytes) / TS_CF;
      const chunks = 2 + Math.ceil(M / TS_CHUNK); // one chunk read per 120 samples of a series
      const r = mkRead(f, {
        plan: `chunk pruning by time, ${fmtNum(T)} of ${fmtNum(i.cols)} columns`,
        ios: chunks,
        bytes,
        units: M * T,
        unitNs: VEC_CPU,
        unitName: 'samples',
      });
      return fin(r, w, { plan: 'time-pruned chunk scan', rowsScanned: M, bytes, ios: chunks, wpr: 1 });
    }

    /* ------------------------------------------------------------------ cache */
    case 'cache': {
      const w = mkWrite({ wpr: 1, entryNs: 2_000, entryName: 'hash slot/row', fixedNs: RPC, fixedLabel: 'RTT, no fsync' });
      if (i.shape !== 'point') return NOPE('no secondary index — SCAN walks the whole keyspace');
      const r = mkRead(f, {
        plan: 'GET on a hash slot',
        ios: 0,
        bytes: W,
        units: 1,
        unitNs: ROW_CPU,
        unitName: 'value',
        extraNs: RPC,
        extraLabel: 'RTT',
      });
      return fin(r, w, { plan: 'hash GET', rowsScanned: 1, bytes: W, ios: 0, wpr: 1 });
    }

    /* -------------------------------------------------------------------- log */
    case 'log': {
      const w = mkWrite({ wpr: 1, entryNs: 1_000, entryName: 'append/row (batched, acks=all)' });
      if (i.shape === 'point') return NOPE('addressable only by partition + offset');
      if (i.shape === 'aggregate') return NOPE('no query engine — this needs a stream processor');
      if (i.shape === 'fulltext' || i.shape === 'similarity') return NOPE('no index of any kind — the log is ordered bytes');
      if (i.shape === 'append') {
        const bytes = M * W;
        const r = mkRead(f, { plan: 'sequential replay from an offset (zero-copy)', ios: 1, bytes });
        return fin(r, w, { plan: 'offset replay', rowsScanned: M, bytes, ios: 1, wpr: 1 });
      }
      const bytes = i.rows * W;
      const r = mkRead(f, {
        plan: 'full replay — the filter runs in the consumer',
        ios: 1,
        bytes,
        units: i.rows,
        unitNs: ROW_CPU,
        unitName: 'records',
      });
      return fin(r, w, { plan: 'full replay', rowsScanned: i.rows, bytes, ios: 1, wpr: 1 });
    }
  }
}

/* ---------------------------------------------- ranking + crossover search */

type Ranked = { fam: Fam; c: Cost };

function rank(i: In): Ranked[] {
  const all = FAMS.map((fam) => ({ fam, c: cost(fam, i) }));
  return all.sort((a, b) => {
    if (a.c.ok !== b.c.ok) return a.c.ok ? -1 : 1;
    return a.c.totalNs - b.c.totalNs;
  });
}

function leaderAt(i: In): FamId | null {
  let best: Ranked | null = null;
  for (const fam of FAMS) {
    const c = cost(fam, i);
    if (!c.ok) continue;
    if (!best || c.totalNs < best.c.totalNs) best = { fam, c };
  }
  return best ? best.fam.id : null;
}

/** Sweep selectivity and report the nearest place the ranking inverts. */
/** A true one-line reason for a given hand-over, rather than one story asserted for all. */
function flipWhy(from: FamId, to: FamId): string {
  const pair = [from, to];
  if (pair.includes('rowstore'))
    return 'a B+tree plan pays one random fetch per matching row, so its cost tracks rows matched; a scanning engine pays for the columns it reads whatever the match count.';
  if (pair.includes('cache') || pair.includes('log'))
    return 'one side answers by address and the other has to look, so the crossover is really about whether the query has a key at all.';
  return 'both read only what matches, but the bytes and the random IOs they pay per match differ — compare the terms printed on each bar.';
}

function findFlip(i: In): { sel: number; from: FamId; to: FamId } | null {
  if (i.shape === 'point') return null;
  const steps: { e: number; id: FamId | null }[] = [];
  for (let e = -9; e <= 0.0001; e += 0.1) {
    steps.push({ e, id: leaderAt({ ...i, sel: Math.pow(10, e) }) });
  }
  const here = Math.log10(i.sel);
  let bestIdx = -1;
  let bestDist = Infinity;
  for (let k = 1; k < steps.length; k++) {
    const a = steps[k - 1];
    const b = steps[k];
    if (!a.id || !b.id || a.id === b.id) continue;
    const d = Math.abs(b.e - here);
    if (d < bestDist) {
      bestDist = d;
      bestIdx = k;
    }
  }
  if (bestIdx < 0) return null;
  return {
    sel: Math.pow(10, steps[bestIdx].e),
    from: steps[bestIdx - 1].id as FamId,
    to: steps[bestIdx].id as FamId,
  };
}

/* ------------------------------------------------------------- the strip */

type Slot = Fam['slot'];

const DERIVED: readonly { slot: Slot; label: string; sub: string }[] = [
  { slot: 'analytics', label: 'Columnar / time-series store', sub: 'ClickHouse, Snowflake, Timescale' },
  { slot: 'search', label: 'Search index', sub: 'Elasticsearch, OpenSearch' },
  { slot: 'vector', label: 'Vector index', sub: 'pgvector, Milvus, Qdrant' },
  { slot: 'cache', label: 'Read cache', sub: 'Redis, Memcached' },
];

/** Deterministic dot positions along the CDC arrow — a stream, not a ruler. */
const STREAM = (() => {
  const rng = makeRng(20231107);
  return Array.from({ length: 16 }, () => rng());
})();

/* ----------------------------------------------------------- the component */

const pct = (s: number) => (s >= 0.01 ? `${(s * 100).toFixed(s >= 0.1 ? 0 : 1)}%` : `${s.toExponential(0)}`);

export default function WorkloadShapeRanker() {
  const [shape, setShape] = useState<Shape>('range');
  const [rowsHalf, setRowsHalf] = useState(18); // rows = 10^(n/2)
  const [cols, setCols] = useState(40);
  const [colBytes, setColBytes] = useState(24);
  const [touch, setTouch] = useState(3);
  const [selTenth, setSelTenth] = useState(-40); // sel = 10^(n/10)
  const [wshare, setWshare] = useState(10);
  const [sidx, setSidx] = useState(2);
  const [ref, width] = useSize(880);

  const input: In = useMemo(
    () => ({
      shape,
      rows: Math.round(Math.pow(10, rowsHalf / 2)),
      cols,
      colBytes,
      touch: Math.min(touch, cols),
      sel: Math.pow(10, selTenth / 10),
      wshare: wshare / 100,
      sidx,
    }),
    [shape, rowsHalf, cols, colBytes, touch, selTenth, wshare, sidx],
  );

  const ranked = useMemo(() => rank(input), [input]);
  const flip = useMemo(() => findFlip(input), [input]);
  const tip = useTip();

  const supported = ranked.filter((r) => r.c.ok);
  const winner = supported[0];
  const second = supported[1];
  const margin = winner && second ? second.c.totalNs / winner.c.totalNs : 0;

  /* layout */
  const svgW = Math.max(900, Math.min(width, 1180));
  const labelW = 188;
  const barW = Math.max(200, svgW - labelW - 470);
  const cardH = 54;
  const top = 30;
  const cardsH = FAMS.length * cardH;
  const stripY = top + cardsH + 26;
  const stripH = 168;
  const height = stripY + stripH;

  const lo = supported.length ? Math.log10(supported[supported.length - 1].c.totalNs) : 3;
  const hi = supported.length ? Math.log10(supported[0].c.totalNs) : 6;
  const axLo = Math.min(hi, lo) - 0.2;
  const axHi = Math.max(hi, lo) + 0.15;
  const span = Math.max(0.5, axHi - axLo);
  const xOf = (ns: number) => labelW + clamp((Math.log10(ns) - axLo) / span, 0, 1) * barW;
  const decades: number[] = [];
  for (let d = Math.ceil(axLo); d <= Math.floor(axHi); d++) decades.push(d);

  const winnerRow: Slot | null = winner ? winner.fam.slot : null;

  return (
    <VizPanel
      title="Which engine family wins this workload"
      subtitle="One cost model per access shape, every term printed. Nudge selectivity or columns-touched past a crossover and the ranking inverts. Cold cache, no parallelism — a teaching model, not a benchmark."
      controls={
        <>
          <Segmented label="Query shape" value={shape} onChange={setShape} options={SHAPES} />
          <Slider
            label="Rows in the table"
            min={8}
            max={22}
            value={rowsHalf}
            onChange={setRowsHalf}
            format={() => fmtNum(input.rows)}
          />
          <Slider label="Columns in the row" min={4} max={300} step={2} value={cols} onChange={setCols} />
          <Slider
            label="Avg column width"
            min={8}
            max={128}
            step={4}
            value={colBytes}
            onChange={setColBytes}
            format={(n) => `${n} B → ${fmtBytes(n * cols)} row`}
          />
          <Slider
            label="Columns touched"
            min={1}
            max={300}
            value={touch}
            onChange={setTouch}
            format={(n) => `${Math.min(n, cols)} of ${cols}`}
          />
          <Slider
            label="Selectivity"
            min={-90}
            max={0}
            value={selTenth}
            onChange={setSelTenth}
            format={() => `${pct(input.sel)} → ${fmtNum(Math.max(1, Math.round(input.rows * input.sel)))} rows`}
          />
          <Slider label="Writes in the mix" min={0} max={100} step={5} value={wshare} onChange={setWshare} format={(n) => `${n}%`} />
          <Slider label="Secondary indexes" min={0} max={8} value={sidx} onChange={setSidx} />
        </>
      }
      legend={<Legend items={FAMS.map((f) => ({ label: `${f.name} — ${f.products}`, color: f.color }))} />}
      stats={
        <Stats
          items={[
            { label: 'Winner', value: winner ? winner.fam.name : '—', hint: winner?.c.plan },
            { label: 'Time per operation', value: winner ? fmtTime(winner.c.totalNs) : '—', hint: 'Read and write blended at the current mix' },
            {
              label: 'Margin over 2nd',
              value: margin ? `${margin < 10 ? margin.toFixed(1) : fmtNum(margin)}× vs ${second.fam.name}` : '—',
            },
            { label: 'Rows scanned', value: winner ? fmtNum(winner.c.rowsScanned) : '—', hint: 'By the winner, for one read' },
            { label: 'Bytes read', value: winner ? fmtBytes(winner.c.bytes) : '—' },
            { label: 'Writes per row', value: winner ? fmtNum(winner.c.wpr) : '—', hint: 'Index entries / column chunks / postings touched per row written' },
          ]}
        />
      }
      note={
        <Note>
          <strong>
            {winner
              ? `${winner.fam.name} wins this workload by ${margin < 10 ? margin.toFixed(1) : fmtNum(margin)}× over ${second?.fam.name ?? 'the field'}.`
              : 'No family in this list can serve that shape.'}
          </strong>{' '}
          {winner ? winner.c.readTerms : ''}{' '}
          {flip ? (
            <>
              Hold everything else and drag <em>selectivity</em>: at about {pct(flip.sel)} the lead passes from{' '}
              {FAM_BY_ID.get(flip.from)!.name} to {FAM_BY_ID.get(flip.to)!.name} — {flipWhy(flip.from, flip.to)}
            </>
          ) : (
            <>Selectivity does not move the ranking for this shape; columns touched and row width do.</>
          )}
        </Note>
      }
      table={
        <table className="viz-table">
          <thead>
            <tr>
              <th>#</th>
              <th>Family</th>
              <th>Plan</th>
              <th>Rows scanned</th>
              <th>Bytes read</th>
              <th>Random IOs</th>
              <th>Writes/row</th>
              <th>Read</th>
              <th>Write</th>
              <th>Blended</th>
            </tr>
          </thead>
          <tbody>
            {ranked.map((r, k) => (
              <tr key={r.fam.id}>
                <td>{r.c.ok ? k + 1 : '—'}</td>
                <td>{r.fam.name}</td>
                <td>{r.c.ok ? r.c.plan : r.c.why}</td>
                <td>{r.c.ok ? fmtNum(r.c.rowsScanned) : '—'}</td>
                <td>{r.c.ok ? fmtBytes(r.c.bytes) : '—'}</td>
                <td>{r.c.ok ? fmtNum(r.c.ios) : '—'}</td>
                <td>{r.c.ok ? fmtNum(r.c.wpr) : '—'}</td>
                <td>{r.c.ok ? fmtTime(r.c.readNs) : '—'}</td>
                <td>{r.c.ok ? fmtTime(r.c.writeNs) : '—'}</td>
                <td>{r.c.ok ? fmtTime(r.c.totalNs) : '—'}</td>
              </tr>
            ))}
          </tbody>
        </table>
      }
    >
      <div ref={ref}>
        <TooltipHost>
          <svg width={svgW} height={height} role="img" aria-label="Engine families ranked by modelled cost for the chosen workload">
            {/* log-scale decade grid */}
            {decades.map((d) => (
              <g key={d}>
                <line
                  className="viz-grid-line"
                  x1={xOf(Math.pow(10, d))}
                  x2={xOf(Math.pow(10, d))}
                  y1={top - 12}
                  y2={top + cardsH}
                />
                <text x={xOf(Math.pow(10, d))} y={top - 16} textAnchor="middle" fill="var(--viz-ink-muted)">
                  {fmtTime(Math.pow(10, d))}
                </text>
              </g>
            ))}
            <text x={0} y={top - 16} fill="var(--viz-ink-muted)">
              time per operation, log scale →
            </text>

            {ranked.map((r, k) => {
              const y = top + k * cardH;
              const c = r.c;
              return (
                <g key={r.fam.id}>
                  {k === 0 && c.ok ? (
                    <rect x={0} y={y - 2} width={svgW} height={cardH - 4} rx={6} fill="var(--viz-plane)" />
                  ) : null}
                  <text x={0} y={y + 14} fill="var(--viz-ink-muted)">
                    {c.ok ? `${k + 1}.` : '✕'}
                  </text>
                  <text x={18} y={y + 14} fill="var(--viz-ink)" fontWeight={600}>
                    {r.fam.name}
                  </text>
                  {c.ok ? (
                    <g
                      {...tip(
                        <>
                          <strong>{r.fam.name}</strong>
                          <br />
                          {r.fam.products}
                          <br />
                          read {fmtTime(c.readNs)} · write {fmtTime(c.writeNs)} per row
                          <br />
                          {fmtNum(c.rowsScanned)} rows scanned, {fmtBytes(c.bytes)} read, {fmtNum(c.ios)} random IOs
                        </>,
                      )}
                      style={{ cursor: 'help' }}
                    >
                      <rect
                        x={labelW}
                        y={y + 3}
                        width={Math.max(3, xOf(c.totalNs) - labelW)}
                        height={14}
                        rx={3}
                        fill={r.fam.color}
                      />
                      <text
                        x={xOf(c.totalNs) + 8}
                        y={y + 14}
                        fill="var(--viz-ink)"
                        style={{ fontVariantNumeric: 'tabular-nums' }}
                      >
                        {fmtTime(c.totalNs)}
                      </text>
                    </g>
                  ) : (
                    <text x={labelW} y={y + 14} fill="var(--viz-ink-muted)">
                      cannot serve this shape — {c.why}
                    </text>
                  )}
                  <text x={18} y={y + 30} fill="var(--viz-ink-2)">
                    {c.ok ? c.readTerms : r.fam.products}
                  </text>
                  {c.ok ? (
                    <text x={18} y={y + 44} fill="var(--viz-ink-muted)">
                      writes: {c.writeTerms}
                    </text>
                  ) : null}
                </g>
              );
            })}

            {/* ---------------- composed stack: one system of record + derived stores */}
            <line className="viz-axis-line" x1={0} x2={svgW} y1={stripY - 14} y2={stripY - 14} />
            <text x={0} y={stripY - 2} fill="var(--viz-ink)" fontWeight={600}>
              The usual deployment: one system of record, everything else derived from its log
            </text>

            {(() => {
              const boxY = stripY + 10;
              const sorW = 200;
              const logX = sorW + 130;
              const logW = 150;
              const derX = logX + logW + 90;
              const derW = Math.max(200, svgW - derX - 8);
              const derH = 28;
              const gap = 8;
              const highlight = (slot: Slot) => winnerRow === slot;
              return (
                <g>
                  {/* system of record */}
                  <rect
                    x={0}
                    y={boxY + 36}
                    width={sorW}
                    height={44}
                    rx={6}
                    fill="var(--viz-plane)"
                    stroke={highlight('sor') ? 'var(--viz-1)' : 'var(--viz-border)'}
                    strokeWidth={highlight('sor') ? 2 : 1}
                  />
                  <text x={10} y={boxY + 54} fill="var(--viz-ink)" fontWeight={600}>
                    System of record
                  </text>
                  <text x={10} y={boxY + 70} fill="var(--viz-ink-muted)">
                    row store, the only writable copy
                  </text>

                  {/* CDC arrow */}
                  <line x1={sorW + 6} x2={logX - 8} y1={boxY + 58} y2={boxY + 58} stroke="var(--viz-axis)" strokeWidth={1.5} />
                  {STREAM.map((u, n) => (
                    <circle
                      key={n}
                      cx={sorW + 10 + u * (logX - sorW - 22)}
                      cy={boxY + 58}
                      r={2.2}
                      fill="var(--viz-8)"
                      opacity={0.85}
                    />
                  ))}
                  <text x={sorW + 8} y={boxY + 48} fill="var(--viz-ink-muted)">
                    WAL → logical decoding / binlog
                  </text>

                  {/* the log */}
                  <rect
                    x={logX}
                    y={boxY + 36}
                    width={logW}
                    height={44}
                    rx={6}
                    fill="var(--viz-plane)"
                    stroke={highlight('log') ? 'var(--viz-8)' : 'var(--viz-border)'}
                    strokeWidth={highlight('log') ? 2 : 1}
                  />
                  <text x={logX + 10} y={boxY + 54} fill="var(--viz-ink)" fontWeight={600}>
                    Change log
                  </text>
                  <text x={logX + 10} y={boxY + 70} fill="var(--viz-ink-muted)">
                    Kafka / Debezium
                  </text>

                  {/* derived stores */}
                  {DERIVED.map((d, n) => {
                    const yy = boxY + n * (derH + gap);
                    const on = highlight(d.slot);
                    const col =
                      d.slot === 'analytics'
                        ? 'var(--viz-2)'
                        : d.slot === 'search'
                          ? 'var(--viz-4)'
                          : d.slot === 'vector'
                            ? 'var(--viz-5)'
                            : 'var(--viz-7)';
                    return (
                      <g key={d.slot}>
                        <path
                          d={`M ${logX + logW + 6} ${boxY + 58} C ${derX - 40} ${boxY + 58}, ${derX - 40} ${yy + derH / 2}, ${derX - 6} ${yy + derH / 2}`}
                          fill="none"
                          stroke="var(--viz-axis)"
                          strokeWidth={1}
                        />
                        <rect
                          x={derX}
                          y={yy}
                          width={derW}
                          height={derH}
                          rx={5}
                          fill="var(--viz-plane)"
                          stroke={on ? col : 'var(--viz-border)'}
                          strokeWidth={on ? 2 : 1}
                        />
                        <text x={derX + 8} y={yy + 12} fill="var(--viz-ink)">
                          {d.label}
                        </text>
                        <text x={derX + 8} y={yy + 24} fill="var(--viz-ink-muted)">
                          {d.sub} — rebuildable, never authoritative
                        </text>
                      </g>
                    );
                  })}

                  <text x={0} y={boxY + 100} fill="var(--viz-ink-2)">
                    {winner
                      ? winner.fam.id === 'htap'
                        ? 'HTAP collapses both halves into one system: you stop running a pipeline and start paying ~1.6× on every write plus the isolation problem.'
                        : winnerRow === 'sor'
                          ? 'This workload belongs on the system of record itself — no derived store needed.'
                          : winnerRow === 'log'
                            ? 'This workload belongs on the log: append, replay by offset, and let consumers build their own views.'
                            : `This workload belongs in a derived store fed by the log — highlighted above — not on the primary.`
                      : ''}
                  </text>
                </g>
              );
            })()}
          </svg>
        </TooltipHost>
      </div>
    </VizPanel>
  );
}
