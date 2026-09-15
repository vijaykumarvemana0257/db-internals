import { useEffect, useMemo, useState } from 'react';
import { VizPanel, Slider, Choice, Segmented, Check, Legend, Stats, Note, makeRng, fmtBytes, fmtNum } from './Viz';

/**
 * Hybrid search as an indexing problem: one table, three query parts — a tsvector text match, an ORDER BY
 * embedding distance LIMIT k branch, and a tenant_id filter — and a candidate index set that decides whether each
 * part seeks or scans, and where the filter can run.
 *
 * What is real (followed from source):
 * - HNSW build follows pgvector: level = floor(-ln(U)/ln(m)), greedy descent with ef = 1, ef_construction search per
 *   layer, Algorithm-4 neighbor heuristic that refills with pruned candidates, 2m neighbors on layer 0, and on every
 *   insert each chosen neighbor's list is updated (append if not full, else heuristic replace) — each update rewrites
 *   that neighbor's neighbor tuple (hnswinsert.c HnswUpdateNeighborsOnDisk).
 * - HNSW scan follows hnswscan.c: a layer-0 search with hnsw.ef_search; with iterative scans on, candidates that were
 *   discarded are kept and later scans resume from the nearest ef_search of them; after hnsw.max_scan_tuples the scan
 *   only drains what it already has; strict_order drops tuples that come back out of order.
 * - IVFFlat follows ivfscan.c: order lists by center distance, read every tuple of the nearest `probes` lists, sort,
 *   return in order; iterative scans read the next `probes` lists.
 * - "Filter inside the walk" follows Lucene's AbstractKnnVectorQuery: evaluate the filter first, exact search if it
 *   matches no more than the candidate count, otherwise walk the graph collecting only accepted docs with a visit
 *   limit of (matches + 1), falling back to exact search when the limit is hit. The candidate count
 *   (num_candidates in Elasticsearch) is the ef_search slider, relabelled in that mode.
 * Model assumptions (labelled in the UI):
 * - 6,000 rows, 16-dimensional normalized vectors in 24 clusters (the search runs on these); tenant is independent of
 *   the embedding; 10 queries, k = 10.
 * - Byte sizes are projected to 1,000,000 rows at the chosen dimension from the tuple layouts in pgvector's hnsw.h
 *   and IVFFlat index tuples on 8 KB pages; the B-tree is taken as deduplicated (6-byte TIDs); GIN bytes per posting
 *   come from varbyte-encoding this corpus's posting lists with 40 heap rows per page.
 * - "Planner's choice" is a simplification: pgvector's startup-cost formula plus the planner's LIMIT fraction, in
 *   tuples instead of cost units. It sees cost, never recall.
 */

/* ------------------------------------------------------------------ corpus */

export const N = 6000;
export const DIM = 16;
export const K = 10;
export const QUERIES = 10;
export const CLUSTERS = 24;
export const HNSW_M = 16;
export const HNSW_EFC = 64;
export const MAX_SCAN_TUPLES = 20000;
export const IVF_LISTS = N / 1000; // pgvector's starting point: rows / 1000
export const TENANT_ROWS = [2400, 1500, 900, 600, 300, 180, 90, 30];
const TOPIC_LEXEMES = 12;
const GLOBAL_LEXEMES = 400;
const TOKENS_PER_DOC = 24;
export const ROWS_PER_HEAP_PAGE = 40;

export type Corpus = {
  vec: Float32Array;
  tenant: Uint8Array;
  cluster: Uint8Array;
  lexemes: Int32Array[];
  postings: Map<number, number[]>;
  queries: { vec: Float32Array; terms: [number, number] }[];
  tenantRows: number[][];
};

function gauss(rng: () => number) {
  const u = Math.max(1e-9, rng());
  const v = rng();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

function normalize(a: Float32Array, off: number, dim: number) {
  let s = 0;
  for (let i = 0; i < dim; i++) s += a[off + i] * a[off + i];
  const r = Math.sqrt(s) || 1;
  for (let i = 0; i < dim; i++) a[off + i] /= r;
}

export function makeCorpus(seed = 7): Corpus {
  const rng = makeRng(seed);
  const centers = new Float32Array(CLUSTERS * DIM);
  for (let i = 0; i < centers.length; i++) centers[i] = gauss(rng);
  for (let c = 0; c < CLUSTERS; c++) normalize(centers, c * DIM, DIM);
  const vec = new Float32Array(N * DIM);
  const cluster = new Uint8Array(N);
  const lexemes: Int32Array[] = [];
  const postings = new Map<number, number[]>();
  // Zipf over the global vocabulary
  const zipfCdf: number[] = [];
  let z = 0;
  for (let r = 1; r <= GLOBAL_LEXEMES; r++) {
    z += 1 / r;
    zipfCdf.push(z);
  }
  for (let r = 0; r < zipfCdf.length; r++) zipfCdf[r] /= z;
  for (let row = 0; row < N; row++) {
    const c = Math.floor(rng() * CLUSTERS);
    cluster[row] = c;
    for (let i = 0; i < DIM; i++) vec[row * DIM + i] = centers[c * DIM + i] + 0.45 * gauss(rng) / Math.sqrt(DIM) * 2.2;
    normalize(vec, row * DIM, DIM);
    const set = new Set<number>();
    for (let t = 0; t < TOKENS_PER_DOC; t++) {
      if (rng() < 0.5) set.add(c * TOPIC_LEXEMES + Math.floor(rng() * TOPIC_LEXEMES));
      else {
        const u = rng();
        let lo = 0;
        let hi = zipfCdf.length - 1;
        while (lo < hi) {
          const mid = (lo + hi) >> 1;
          if (zipfCdf[mid] < u) lo = mid + 1;
          else hi = mid;
        }
        set.add(CLUSTERS * TOPIC_LEXEMES + lo);
      }
    }
    const arr = Int32Array.from([...set].sort((a, b) => a - b));
    lexemes.push(arr);
    for (const l of arr) {
      let p = postings.get(l);
      if (!p) postings.set(l, (p = []));
      p.push(row);
    }
  }
  // tenants: exact row counts, assigned independently of cluster
  const order = Array.from({ length: N }, (_, i) => i);
  for (let i = N - 1; i > 0; i--) {
    const j = Math.floor(rng() * (i + 1));
    [order[i], order[j]] = [order[j], order[i]];
  }
  const tenant = new Uint8Array(N);
  const tenantRows: number[][] = TENANT_ROWS.map(() => []);
  let at = 0;
  TENANT_ROWS.forEach((cnt, t) => {
    for (let i = 0; i < cnt; i++) tenant[order[at++]] = t;
  });
  for (let row = 0; row < N; row++) tenantRows[tenant[row]].push(row);
  const queries: Corpus['queries'] = [];
  for (let q = 0; q < QUERIES; q++) {
    const c = Math.floor((q * CLUSTERS) / QUERIES);
    const v = new Float32Array(DIM);
    for (let i = 0; i < DIM; i++) v[i] = centers[c * DIM + i] + 0.35 * gauss(rng) / Math.sqrt(DIM) * 2.2;
    normalize(v, 0, DIM);
    const a = c * TOPIC_LEXEMES + Math.floor(rng() * TOPIC_LEXEMES);
    let b = c * TOPIC_LEXEMES + Math.floor(rng() * TOPIC_LEXEMES);
    if (b === a) b = c * TOPIC_LEXEMES + ((a - c * TOPIC_LEXEMES + 1) % TOPIC_LEXEMES);
    queries.push({ vec: v, terms: [a, b] });
  }
  return { vec, tenant, cluster, lexemes, postings, queries, tenantRows };
}

/* ------------------------------------------------------------------- heaps */

/** Binary min-heap on key. */
class Heap {
  ids: number[] = [];
  keys: number[] = [];
  get size() {
    return this.ids.length;
  }
  push(id: number, key: number) {
    const ids = this.ids;
    const keys = this.keys;
    let i = ids.length;
    ids.push(id);
    keys.push(key);
    while (i > 0) {
      const p = (i - 1) >> 1;
      if (keys[p] <= key) break;
      ids[i] = ids[p];
      keys[i] = keys[p];
      i = p;
    }
    ids[i] = id;
    keys[i] = key;
  }
  pop(): [number, number] {
    const ids = this.ids;
    const keys = this.keys;
    const topId = ids[0];
    const topKey = keys[0];
    const lastId = ids.pop()!;
    const lastKey = keys.pop()!;
    const n = ids.length;
    if (n > 0) {
      let i = 0;
      for (;;) {
        const l = 2 * i + 1;
        if (l >= n) break;
        const r = l + 1;
        const c = r < n && keys[r] < keys[l] ? r : l;
        if (keys[c] >= lastKey) break;
        ids[i] = ids[c];
        keys[i] = keys[c];
        i = c;
      }
      ids[i] = lastId;
      keys[i] = lastKey;
    }
    return [topId, topKey];
  }
}

type Cand = { id: number; d: number };
export type Counter = { dist: number; tuples: number };

function l2(a: Float32Array, ao: number, b: Float32Array, bo: number) {
  let s = 0;
  for (let i = 0; i < DIM; i++) {
    const x = a[ao + i] - b[bo + i];
    s += x * x;
  }
  return s;
}

/* -------------------------------------------------------------------- HNSW */

export type Hnsw = {
  rows: number[];
  level: Int8Array;
  nb: number[][][];
  entry: number;
  entryLevel: number;
  m: number;
  buildDist: number;
  /** Neighbor tuples rewritten by each insert (hnswinsert.c HnswUpdateNeighborsOnDisk). */
  updates: number[];
};

const layerM = (m: number, lc: number) => (lc === 0 ? 2 * m : m);

/** pgvector HnswSearchLayer (Algorithm 2), with the discarded heap used by iterative scans. */
function searchLayer(
  g: Hnsw,
  dq: (node: number) => number,
  ep: Cand[],
  ef: number,
  lc: number,
  visited: Uint8Array,
  discarded: Heap | null,
  initVisited: boolean,
  counter: Counter | null,
): Cand[] {
  const C = new Heap();
  const W = new Heap(); // max-heap via negated key
  let wlen = 0;
  for (const e of ep) {
    if (initVisited) {
      visited[e.id] = 1;
      if (counter) counter.tuples++;
    }
    C.push(e.id, e.d);
    W.push(e.id, -e.d);
    wlen++;
  }
  while (C.size > 0) {
    const [cid, cd] = C.pop();
    if (cd > -W.keys[0]) break;
    const list = g.nb[cid][lc] ?? [];
    const unvisited: number[] = [];
    for (const nbr of list) {
      if (!visited[nbr]) {
        visited[nbr] = 1;
        unvisited.push(nbr);
      }
    }
    if (counter) counter.tuples += unvisited.length;
    for (const e of unvisited) {
      const fD = -W.keys[0];
      const alwaysAdd = wlen < ef;
      const ed = dq(e);
      if (!(ed < fD || alwaysAdd)) {
        if (discarded) discarded.push(e, ed);
        continue;
      }
      if (g.level[e] < lc) continue;
      C.push(e, ed);
      W.push(e, -ed);
      wlen++;
      if (wlen > ef) {
        const [did, dk] = W.pop();
        if (discarded) discarded.push(did, -dk);
      }
    }
  }
  const out: Cand[] = [];
  while (W.size > 0) {
    const [id, k] = W.pop();
    out.push({ id, d: -k });
  }
  return out.reverse();
}

type SelCand = { id: number; d: number; closer: boolean };

/**
 * pgvector SelectNeighbors (Algorithm 4, refilled with pruned candidates), including its "closer" cache: once a
 * neighbor list has been selected in sorted order, a later update only compares the new candidate with the kept
 * neighbors and re-checks the rest against what was added.
 */
function selectNeighbors(dn: (a: number, b: number) => number, c: SelCand[], lm: number, closerSet: boolean, newCand: SelCand | null) {
  if (c.length <= lm) return { r: c, pruned: null as SelCand | null };
  const closerThanAll = (e: SelCand, r: SelCand[]) => {
    for (const ri of r) if (dn(e.id, ri.id) <= e.d) return false;
    return true;
  };
  const mustCalculate = !closerSet;
  const r: SelCand[] = [];
  const wd: SelCand[] = [];
  const added: SelCand[] = [];
  let removedAny = false;
  let idx = 0;
  while (idx < c.length && r.length < lm) {
    const e = c[idx++];
    if (mustCalculate) e.closer = closerThanAll(e, r);
    else if (added.length > 0) {
      if (e.closer) {
        e.closer = closerThanAll(e, added);
        if (!e.closer) removedAny = true;
      } else if (removedAny) {
        e.closer = closerThanAll(e, r);
        if (e.closer) added.push(e);
      }
    } else if (e === newCand) {
      e.closer = closerThanAll(e, r);
      if (e.closer) added.push(e);
    }
    (e.closer ? r : wd).push(e);
  }
  let wdoff = 0;
  while (wdoff < wd.length && r.length < lm) r.push(wd[wdoff++]);
  const pruned = wdoff < wd.length ? wd[wdoff] : c[c.length - 1];
  return { r, pruned };
}

/** Incremental pgvector-style HNSW build, so the page can build 6,000 rows across several short tasks. */
export function hnswBuilder(corpus: Corpus, rows: number[], m = HNSW_M, efc = HNSW_EFC, seed = 11) {
  const rng = makeRng(seed);
  const n = rows.length;
  const ml = 1 / Math.log(m);
  const vec = corpus.vec;
  const g: Hnsw = { rows, level: new Int8Array(n), nb: [], entry: -1, entryLevel: -1, m, buildDist: 0, updates: [] };
  const lists: SelCand[][][] = [];
  const closerSet: boolean[][] = [];
  const dn = (a: number, b: number) => {
    g.buildDist++;
    return l2(vec, rows[a] * DIM, vec, rows[b] * DIM);
  };
  const visited = new Uint8Array(n);
  const touched: number[] = [];
  const sortCands = (a: SelCand, b: SelCand) => a.d - b.d || a.id - b.id;
  let i = 0;
  const insertOne = () => {
    const u = rng();
    const lvl = Math.min(8, u === 0 ? 8 : Math.floor(-Math.log(u) * ml));
    g.level[i] = lvl;
    g.nb.push(Array.from({ length: lvl + 1 }, () => []));
    lists.push(Array.from({ length: lvl + 1 }, () => []));
    closerSet.push(Array.from({ length: lvl + 1 }, () => false));
    if (g.entry < 0) {
      g.entry = i;
      g.entryLevel = lvl;
      g.updates.push(0);
      return;
    }
    const dq = (node: number) => dn(i, node);
    const run = (eps: Cand[], ef: number, lc: number) => {
      for (const t of touched) visited[t] = 0;
      touched.length = 0;
      for (const e of eps) touched.push(e.id);
      return searchLayer(g, (node) => {
        touched.push(node);
        return dq(node);
      }, eps, ef, lc, visited, null, true, null);
    };
    let ep: Cand[] = [{ id: g.entry, d: dq(g.entry) }];
    for (let lc = g.entryLevel; lc > lvl; lc--) ep = run(ep, 1, lc);
    const top = Math.min(lvl, g.entryLevel);
    for (let lc = top; lc >= 0; lc--) {
      const w = run(ep, efc, lc);
      const { r } = selectNeighbors(dn, w.map((x) => ({ id: x.id, d: x.d, closer: false })), layerM(m, lc), false, null);
      lists[i][lc] = r;
      g.nb[i][lc] = r.map((x) => x.id);
      ep = w;
    }
    let upd = 0;
    for (let lc = top; lc >= 0; lc--) {
      const lm = layerM(m, lc);
      for (const mine of lists[i][lc]) {
        const j = mine.id;
        const list = lists[j][lc];
        const newCand: SelCand = { id: i, d: mine.d, closer: false };
        if (list.length < lm) {
          list.push(newCand);
          g.nb[j][lc].push(i);
          upd++;
          continue;
        }
        const cands = [...list, newCand].sort(sortCands);
        const { pruned } = selectNeighbors(dn, cands, lm, closerSet[j][lc], newCand);
        closerSet[j][lc] = true;
        if (pruned && pruned !== newCand) {
          const at = list.indexOf(pruned);
          if (at >= 0) {
            list[at] = newCand;
            g.nb[j][lc][at] = i;
            upd++;
          }
        }
      }
    }
    g.updates.push(upd);
    if (lvl > g.entryLevel) {
      g.entry = i;
      g.entryLevel = lvl;
    }
  };
  return {
    g,
    get done() {
      return i >= n;
    },
    /** Insert up to `budget` more rows; returns true when the graph holds every row. */
    step(budget: number) {
      const stop = Math.min(n, i + budget);
      for (; i < stop; i++) insertOne();
      return i >= n;
    },
  };
}

export function buildHnsw(corpus: Corpus, rows: number[], m = HNSW_M, efc = HNSW_EFC, seed = 11): Hnsw {
  const b = hnswBuilder(corpus, rows, m, efc, seed);
  b.step(rows.length);
  return b.g;
}

export type Iterative = 'off' | 'relaxed_order' | 'strict_order';
export type ScanResult = {
  rows: number[];
  /** Tuples the index handed the executor (each costs a heap fetch to check the filter). */
  emitted: number;
  /** pgvector's tuple counter for the graph walk, or rows read for IVFFlat / exact paths. */
  visited: number;
  dist: number;
  fallback?: boolean;
  /** An iterative scan reached hnsw.max_scan_tuples and stopped exploring the graph. */
  capped?: boolean;
};

/** pgvector hnswgettuple loop with a Filter + Limit k above it. */
export function hnswScan(corpus: Corpus, g: Hnsw, qv: Float32Array, efSearch: number, iterative: Iterative, accept: (row: number) => boolean, k = K, maxScanTuples = MAX_SCAN_TUPLES): ScanResult {
  const counter: Counter = { dist: 0, tuples: 0 };
  const n = g.rows.length;
  const res: ScanResult = { rows: [], emitted: 0, visited: 0, dist: 0 };
  if (g.entry < 0) return res;
  const dq = (node: number) => {
    counter.dist++;
    return l2(corpus.vec, g.rows[node] * DIM, qv, 0);
  };
  let ep: Cand[] = [{ id: g.entry, d: dq(g.entry) }];
  for (let lc = g.entryLevel; lc >= 1; lc--) ep = searchLayer(g, dq, ep, 1, lc, new Uint8Array(n), null, true, null);
  const visited = new Uint8Array(n);
  const discarded = iterative === 'off' ? null : new Heap();
  let w = searchLayer(g, dq, ep, efSearch, 0, visited, discarded, true, counter);
  let wi = 0;
  let prev = -Infinity;
  const guard = n * 4 + 1000;
  let steps = 0;
  while (steps++ < guard) {
    if (wi >= w.length) {
      if (!discarded || discarded.size === 0) break;
      if (counter.tuples >= maxScanTuples) {
        res.capped = true;
        const [id, d] = discarded.pop();
        w = [{ id, d }];
      } else {
        const batch: Cand[] = [];
        for (let i = 0; i < efSearch && discarded.size > 0; i++) {
          const [id, d] = discarded.pop();
          batch.push({ id, d });
        }
        w = searchLayer(g, dq, batch, efSearch, 0, visited, discarded, false, counter);
      }
      wi = 0;
      if (w.length === 0) break;
    }
    const c = w[wi++];
    if (iterative === 'strict_order') {
      if (c.d < prev) continue;
      prev = c.d;
    }
    res.emitted++;
    const row = g.rows[c.id];
    if (accept(row)) {
      res.rows.push(row);
      if (res.rows.length >= k) break;
    }
  }
  res.visited = counter.tuples;
  res.dist = counter.dist;
  return res;
}

/** Exact top-k over a set of rows: what a B-tree pre-filter or a sequential scan feeds into a top-N sort. */
export function exactTopK(corpus: Corpus, rowsIn: Iterable<number>, qv: Float32Array, k = K): ScanResult {
  const all: Cand[] = [];
  for (const row of rowsIn) all.push({ id: row, d: l2(corpus.vec, row * DIM, qv, 0) });
  all.sort((a, b) => a.d - b.d || a.id - b.id);
  return { rows: all.slice(0, k).map((c) => c.id), emitted: all.length, visited: all.length, dist: all.length };
}

/* ------------------------------------------------------------------ IVFFlat */

export type Ivf = { centers: Float32Array; lists: number[][]; buildDist: number };

export function buildIvf(corpus: Corpus, lists = IVF_LISTS, seed = 5): Ivf {
  const rng = makeRng(seed);
  const vec = corpus.vec;
  let dist = 0;
  const centers = new Float32Array(lists * DIM);
  // k-means++ seeding over all rows (pgvector samples max(50 * lists, 10000) rows, which is every row here)
  const first = Math.floor(rng() * N);
  centers.set(vec.subarray(first * DIM, first * DIM + DIM), 0);
  const best = new Float64Array(N).fill(Infinity);
  for (let c = 1; c < lists; c++) {
    let sum = 0;
    for (let row = 0; row < N; row++) {
      const d = l2(vec, row * DIM, centers, (c - 1) * DIM);
      dist++;
      if (d < best[row]) best[row] = d;
      sum += best[row];
    }
    let u = rng() * sum;
    let pick = N - 1;
    for (let row = 0; row < N; row++) {
      u -= best[row];
      if (u <= 0) {
        pick = row;
        break;
      }
    }
    centers.set(vec.subarray(pick * DIM, pick * DIM + DIM), c * DIM);
  }
  const assign = new Int32Array(N).fill(-1);
  for (let iter = 0; iter < 25; iter++) {
    let changed = 0;
    for (let row = 0; row < N; row++) {
      let bc = 0;
      let bd = Infinity;
      for (let c = 0; c < lists; c++) {
        const d = l2(vec, row * DIM, centers, c * DIM);
        dist++;
        if (d < bd) {
          bd = d;
          bc = c;
        }
      }
      if (assign[row] !== bc) changed++;
      assign[row] = bc;
    }
    const sums = new Float64Array(lists * DIM);
    const counts = new Int32Array(lists);
    for (let row = 0; row < N; row++) {
      counts[assign[row]]++;
      for (let i = 0; i < DIM; i++) sums[assign[row] * DIM + i] += vec[row * DIM + i];
    }
    for (let c = 0; c < lists; c++) if (counts[c] > 0) for (let i = 0; i < DIM; i++) centers[c * DIM + i] = sums[c * DIM + i] / counts[c];
    if (changed === 0) break;
  }
  const out: number[][] = Array.from({ length: lists }, () => []);
  for (let row = 0; row < N; row++) out[assign[row]].push(row);
  return { centers, lists: out, buildDist: dist };
}

/** pgvector ivfflatgettuple with a Filter + Limit k above it. */
export function ivfScan(corpus: Corpus, ivf: Ivf, qv: Float32Array, probes: number, iterative: boolean, accept: (row: number) => boolean, k = K): ScanResult {
  const L = ivf.lists.length;
  const order = Array.from({ length: L }, (_, c) => ({ c, d: l2(ivf.centers, c * DIM, qv, 0) })).sort((a, b) => a.d - b.d);
  const p = Math.min(probes, L);
  const maxProbes = iterative ? L : p;
  const res: ScanResult = { rows: [], emitted: 0, visited: 0, dist: L };
  let li = 0;
  while (li < maxProbes && res.rows.length < k) {
    const batch: Cand[] = [];
    for (let b = 0; b < p && li < maxProbes; b++, li++) {
      for (const row of ivf.lists[order[li].c]) batch.push({ id: row, d: l2(corpus.vec, row * DIM, qv, 0) });
    }
    res.visited += batch.length;
    res.dist += batch.length;
    batch.sort((a, b) => a.d - b.d);
    for (const c of batch) {
      res.emitted++;
      if (accept(c.id)) {
        res.rows.push(c.id);
        if (res.rows.length >= k) break;
      }
    }
  }
  return res;
}

/* --------------------------------------------- filter inside the graph walk */

/** Lucene AbstractKnnVectorQuery.getLeafResults + HnswGraphSearcher.searchLevel with acceptOrds. */
export function filteredWalk(corpus: Corpus, g: Hnsw, qv: Float32Array, numCandidates: number, acceptRows: number[], k = K): ScanResult {
  const n = g.rows.length;
  const accept = new Uint8Array(N);
  for (const r of acceptRows) accept[r] = 1;
  const cost = acceptRows.length;
  const exact = () => {
    const e = exactTopK(corpus, acceptRows, qv, k);
    return e;
  };
  if (cost <= numCandidates) return { ...exact(), fallback: true };
  let dist = 0;
  const dq = (node: number) => {
    dist++;
    return l2(corpus.vec, g.rows[node] * DIM, qv, 0);
  };
  let ep: Cand[] = [{ id: g.entry, d: dq(g.entry) }];
  for (let lc = g.entryLevel; lc >= 1; lc--) ep = searchLayer(g, dq, ep, 1, lc, new Uint8Array(n), null, true, null);
  const visitLimit = cost + 1;
  let visitedCount = 0;
  const visited = new Uint8Array(n);
  const cand = new Heap();
  const results = new Heap(); // max-heap via negated key, accepted docs only
  const e0 = ep[0];
  visited[e0.id] = 1;
  visitedCount++;
  cand.push(e0.id, e0.d);
  if (accept[g.rows[e0.id]]) results.push(e0.id, -e0.d);
  const worst = () => (results.size >= numCandidates ? -results.keys[0] : Infinity);
  let early = false;
  while (cand.size > 0 && !early) {
    if (cand.keys[0] > worst()) break;
    const [cid] = cand.pop();
    for (const nbr of g.nb[cid][0] ?? []) {
      if (visited[nbr]) continue;
      if (visitedCount >= visitLimit) {
        early = true;
        break;
      }
      visited[nbr] = 1;
      visitedCount++;
      const d = dq(nbr);
      if (d <= worst()) {
        cand.push(nbr, d);
        if (accept[g.rows[nbr]]) {
          results.push(nbr, -d);
          if (results.size > numCandidates) results.pop();
        }
      }
    }
  }
  if (early || results.size < numCandidates) {
    const e = exact();
    return { rows: e.rows, emitted: e.emitted, visited: visitedCount + e.visited, dist: dist + e.dist, fallback: true };
  }
  const out: Cand[] = [];
  while (results.size > 0) {
    const [id, key] = results.pop();
    out.push({ id, d: -key });
  }
  out.reverse();
  const rows = out.slice(0, k).map((c) => g.rows[c.id]);
  return { rows, emitted: rows.length, visited: visitedCount, dist, fallback: false };
}

/* ------------------------------------------------------------ index set */

export type IndexKey = 'gin' | 'hnsw' | 'ivf' | 'btree' | 'partial';
export type IndexSet = Record<IndexKey, boolean>;
export type Placement = 'after' | 'iterative' | 'inside';
export type PlanKey = 'auto' | 'seq' | 'btree' | 'hnsw' | 'ivf' | 'partial';
export type Dims = 384 | 768 | 1536;

export type Built = { corpus: Corpus; hnsw: Hnsw; ivf: Ivf; partial: Hnsw[] };

export function buildAll(seed = 7): Built {
  const corpus = makeCorpus(seed);
  const hnsw = buildHnsw(corpus, Array.from({ length: N }, (_, i) => i));
  const ivf = buildIvf(corpus);
  const partial = corpus.tenantRows.map((rows, t) => buildHnsw(corpus, rows, HNSW_M, HNSW_EFC, 100 + t));
  return { corpus, hnsw, ivf, partial };
}

const align8 = (x: number) => Math.ceil(x / 8) * 8;
const PAGE = 8192;
const USABLE = 8192 - 24 - 8; // page header + 8-byte opaque area (pgvector HNSW and IVFFlat pages)

/** Index bytes per row for vector(D): element tuple + layer-0 neighbor tuple + two line pointers, packed on 8 KB pages. */
export function hnswBytesPerRow(dims: number, m = HNSW_M) {
  const etup = align8(72 + 8 + 4 * dims);
  const ntup = align8(4 + 6 * 2 * m);
  const perPage = Math.max(1, Math.floor(USABLE / (etup + ntup + 8)));
  return { etup, ntup, perPage, bytes: PAGE / perPage };
}
/** IVFFlat list tuple: IndexTupleData + vector varlena + line pointer. */
export function ivfBytesPerRow(dims: number) {
  const tup = align8(8 + 8 + 4 * dims) + 4;
  const perPage = Math.max(1, Math.floor(USABLE / tup));
  return { tup, perPage, bytes: PAGE / perPage };
}
const varbyteLen = (x: number) => (x < 1 << 7 ? 1 : x < 1 << 14 ? 2 : x < 1 << 21 ? 3 : x < 2 ** 28 ? 4 : 5);

/** ginpostinglist.c: item pointer as (block << 11 | offset), first TID stored whole, then varbyte deltas. */
export function ginPostingStats(corpus: Corpus) {
  let bytes = 0;
  let postings = 0;
  const val = (row: number) => Math.floor(row / ROWS_PER_HEAP_PAGE) * 2048 + ((row % ROWS_PER_HEAP_PAGE) + 1);
  for (const list of corpus.postings.values()) {
    bytes += 6;
    postings += list.length;
    for (let i = 1; i < list.length; i++) bytes += varbyteLen(val(list[i]) - val(list[i - 1]));
  }
  return { bytesPerPosting: bytes / postings, postingsPerRow: postings / N, lexemes: corpus.postings.size, postings };
}

const avgTail = (xs: number[]) => {
  const tail = xs.slice(Math.floor(xs.length / 2));
  return tail.length ? tail.reduce((a, b) => a + b, 0) / tail.length : 0;
};

export type IndexFacts = { key: IndexKey | 'none'; bytes: number; entriesPerInsert: number; buildWork: number; buildUnit: string };

export function indexFacts(b: Built, dims: Dims): Record<IndexKey | 'none', IndexFacts> {
  const rows1m = 1_000_000;
  const gin = ginPostingStats(b.corpus);
  const partialUpd = b.partial.reduce((s, g) => s + avgTail(g.updates) * g.rows.length, 0) / N;
  return {
    none: { key: 'none', bytes: 0, entriesPerInsert: 0, buildWork: 0, buildUnit: '' },
    gin: { key: 'gin', bytes: rows1m * gin.postingsPerRow * gin.bytesPerPosting, entriesPerInsert: gin.postingsPerRow, buildWork: gin.postings, buildUnit: 'postings sorted' },
    hnsw: { key: 'hnsw', bytes: rows1m * hnswBytesPerRow(dims).bytes, entriesPerInsert: 2 + avgTail(b.hnsw.updates), buildWork: b.hnsw.buildDist, buildUnit: 'distance computations' },
    ivf: { key: 'ivf', bytes: rows1m * ivfBytesPerRow(dims).bytes, entriesPerInsert: 1, buildWork: b.ivf.buildDist, buildUnit: 'distance computations' },
    btree: { key: 'btree', bytes: rows1m * 6, entriesPerInsert: 1, buildWork: N, buildUnit: 'keys sorted' },
    partial: { key: 'partial', bytes: rows1m * hnswBytesPerRow(dims).bytes, entriesPerInsert: 2 + partialUpd, buildWork: b.partial.reduce((s, g) => s + g.buildDist, 0), buildUnit: 'distance computations' },
  };
}

/* ---------------------------------------------------------------- planner */

/** pgvector hnswcostestimate's startup tuples: entryLevel * m + 2m * ef_search * layer0Selectivity. */
export function hnswStartupTuples(rows: number, efSearch: number, m = HNSW_M) {
  const ml = 1 / Math.log(m);
  const entryLevel = Math.floor(Math.log(rows) * ml);
  const sel0 = (0.55 * Math.log(rows)) / (Math.log(m) * (1 + Math.log(efSearch)));
  return Math.min(rows, entryLevel * m + 2 * m * efSearch * sel0);
}

export function planEstimates(set: IndexSet, tenant: number, efSearch: number, probes: number) {
  const nt = TENANT_ROWS[tenant];
  const frac = Math.min(1, K / nt);
  const est: Partial<Record<Exclude<PlanKey, 'auto'>, number>> = { seq: N };
  if (set.btree) est.btree = nt;
  if (set.hnsw) {
    const s = hnswStartupTuples(N, efSearch);
    est.hnsw = s + (N - s) * frac;
  }
  if (set.ivf) {
    const s = N * Math.min(1, probes / IVF_LISTS);
    est.ivf = s + (N - s) * frac;
  }
  if (set.partial) {
    const s = hnswStartupTuples(nt, efSearch);
    est.partial = s + (nt - s) * Math.min(1, K / nt);
  }
  return est;
}

export function choosePlan(set: IndexSet, tenant: number, placement: Placement, efSearch: number, probes: number, plan: PlanKey): Exclude<PlanKey, 'auto'> {
  const avail = availablePlans(set);
  if (plan !== 'auto') return avail.includes(plan) ? plan : 'seq';
  if (placement === 'inside' && set.hnsw) return 'hnsw';
  const est = planEstimates(set, tenant, efSearch, probes);
  let best: Exclude<PlanKey, 'auto'> = 'seq';
  for (const k of ['partial', 'btree', 'hnsw', 'ivf', 'seq'] as const) {
    if (est[k] !== undefined && est[k]! < est[best]!) best = k;
  }
  return best;
}

export function availablePlans(set: IndexSet): Exclude<PlanKey, 'auto'>[] {
  const out: Exclude<PlanKey, 'auto'>[] = ['seq'];
  if (set.btree) out.push('btree');
  if (set.hnsw) out.push('hnsw');
  if (set.ivf) out.push('ivf');
  if (set.partial) out.push('partial');
  return out;
}

/* ----------------------------------------------------------------- evaluate */

export type Config = { set: IndexSet; tenant: number; placement: Placement; plan: PlanKey; efSearch: number; probes: number; maxScanTuples?: number };

export type QueryOutcome = { truth: number[]; returned: number[]; missed: number; extra: number };

export function evaluate(b: Built, cfg: Config) {
  const { corpus } = b;
  const { set, tenant, placement, efSearch, probes } = cfg;
  const mst = cfg.maxScanTuples ?? MAX_SCAN_TUPLES;
  const tRows = corpus.tenantRows[tenant];
  const nt = tRows.length;
  const accept = (row: number) => corpus.tenant[row] === tenant;
  const planUsed = choosePlan(set, tenant, placement, efSearch, probes, cfg.plan);
  const estimates = planEstimates(set, tenant, efSearch, probes);
  const outcomes: QueryOutcome[] = [];
  let emitted = 0;
  let visited = 0;
  let dist = 0;
  let fallbacks = 0;
  let capped = 0;
  let filterWhere = '';
  for (const q of corpus.queries) {
    const truth = exactTopK(corpus, tRows, q.vec).rows;
    let r: ScanResult;
    switch (planUsed) {
      case 'seq':
        r = { ...exactTopK(corpus, tRows, q.vec), visited: N };
        filterWhere = 'A sequential scan filters every row, then a top-N sort ranks the survivors';
        break;
      case 'btree':
        r = exactTopK(corpus, tRows, q.vec);
        filterWhere = `The filter runs first, as a B-tree seek on tenant_id; each match gets an exact distance and a top-N sort keeps ${K}`;
        break;
      case 'hnsw':
        if (placement === 'inside') {
          r = filteredWalk(corpus, b.hnsw, q.vec, efSearch, tRows);
          if (!set.btree) r = { ...r, visited: r.visited + N };
          filterWhere = set.btree ? 'The filter runs inside the walk: an accept set from the tenant_id index is consulted at every node' : 'The filter runs inside the walk, but with no index on tenant_id its accept set costs a scan of every row';
        } else {
          r = hnswScan(corpus, b.hnsw, q.vec, efSearch, placement === 'iterative' ? 'relaxed_order' : 'off', accept, K, mst);
          filterWhere = placement === 'iterative' ? `The filter runs after the index, repeatedly: the iterative scan resumes until ${K} rows pass` : `The filter runs after the index: HNSW hands up at most ef_search = ${efSearch} tuples and a Filter node discards other tenants`;
        }
        break;
      case 'ivf':
        r = ivfScan(corpus, b.ivf, q.vec, probes, placement === 'iterative', accept);
        filterWhere = placement === 'iterative' ? `The filter runs after the index, repeatedly: IVFFlat probes further lists until ${K} rows pass` : `The filter runs after the index: IVFFlat reads ${Math.min(probes, IVF_LISTS)} of ${IVF_LISTS} lists and a Filter node discards other tenants`;
        break;
      case 'partial':
        r = hnswScan(corpus, b.partial[tenant], q.vec, efSearch, placement === 'iterative' ? 'relaxed_order' : 'off', () => true, K, mst);
        filterWhere = 'No filter step: the partial index contains only this tenant';
        break;
    }
    const got = new Set(r.rows);
    const missed = truth.filter((x) => !got.has(x)).length;
    const tset = new Set(truth);
    const extra = r.rows.filter((x) => !tset.has(x)).length;
    outcomes.push({ truth, returned: r.rows, missed, extra });
    emitted += r.emitted;
    visited += r.visited;
    dist += r.dist;
    if (r.fallback) fallbacks++;
    if (r.capped) capped++;
  }
  const missed = outcomes.reduce((s, o) => s + o.missed, 0);
  const returned = outcomes.reduce((s, o) => s + o.returned.length, 0);

  // text branch
  const tSet = new Uint8Array(N);
  for (const r of tRows) tSet[r] = 1;
  let textHeap = 0;
  let textMatches = 0;
  let textRanked = 0;
  const textPlans: { key: string; heap: number }[] = [];
  let textPlan = 'seq';
  for (const q of corpus.queries) {
    const a = corpus.postings.get(q.terms[0]) ?? [];
    const bb = new Set(corpus.postings.get(q.terms[1]) ?? []);
    const matches = a.filter((x) => bb.has(x));
    const both = matches.filter((x) => tSet[x]).length;
    const options: { key: string; heap: number }[] = [{ key: 'seq', heap: N }];
    if (set.gin) options.push({ key: 'gin', heap: matches.length });
    if (set.btree) options.push({ key: 'btree', heap: nt });
    if (set.gin && set.btree) options.push({ key: 'and', heap: both });
    options.sort((x, y) => x.heap - y.heap);
    textPlan = options[0].key;
    textHeap += options[0].heap;
    textMatches += matches.length;
    textRanked += both;
    if (textPlans.length === 0) textPlans.push(...options);
  }
  const textFilterWhere: Record<string, string> = {
    seq: 'a sequential scan checks @@ and tenant_id on every row',
    gin: 'a GIN bitmap scan finds every @@ match, then a Filter drops other tenants',
    btree: 'a B-tree seek on tenant_id, then @@ is checked on each row',
    and: 'a BitmapAnd of the GIN and B-tree bitmaps, so only rows passing both are read',
  };
  return {
    planUsed,
    estimates,
    outcomes,
    missed,
    returned,
    recall: 1 - missed / (QUERIES * K),
    qualifying: nt,
    avgEmitted: emitted / QUERIES,
    avgVisited: visited / QUERIES,
    avgDist: dist / QUERIES,
    fallbacks,
    capped,
    filterWhere,
    text: { plan: textPlan, avgHeap: textHeap / QUERIES, avgMatches: textMatches / QUERIES, avgRanked: textRanked / QUERIES, filterWhere: textFilterWhere[textPlan] },
  };
}

/* ======================================================================= UI */

const pct = (t: number) => `${fmtNum((TENANT_ROWS[t] / N) * 100, 1)}%`;
const per = (x: number) => fmtNum(x, x < 10 && x % 1 !== 0 ? 1 : 0);

const PLAN_LABEL: Record<Exclude<PlanKey, 'auto'>, string> = {
  seq: 'Seq scan + top-N sort',
  btree: 'B-tree on tenant_id, exact distances',
  hnsw: 'HNSW index scan',
  ivf: 'IVFFlat index scan',
  partial: 'Partial HNSW for this tenant',
};

const PLAN_SHORT: Record<Exclude<PlanKey, 'auto'>, string> = { seq: 'Seq scan', btree: 'B-tree pre-filter', hnsw: 'HNSW scan', ivf: 'IVFFlat scan', partial: 'Partial HNSW' };

const PLACEMENT_OPTIONS: { value: Placement; label: string }[] = [
  { value: 'after', label: 'After the scan (pgvector default)' },
  { value: 'iterative', label: 'After, with iterative_scan = relaxed_order' },
  { value: 'inside', label: 'Inside the graph walk (Lucene / Elasticsearch)' },
];

const ROWS_META: { key: IndexKey | 'none'; name: string; ddl: string }[] = [
  { key: 'gin', name: 'GIN (tsv)', ddl: 'CREATE INDEX ON chunks USING gin (tsv)' },
  { key: 'hnsw', name: 'HNSW (embedding)', ddl: 'CREATE INDEX ON chunks USING hnsw (embedding vector_cosine_ops)' },
  { key: 'ivf', name: `IVFFlat (embedding)`, ddl: `CREATE INDEX ON chunks USING ivfflat (embedding vector_cosine_ops) WITH (lists = ${IVF_LISTS})` },
  { key: 'btree', name: 'B-tree (tenant_id)', ddl: 'CREATE INDEX ON chunks (tenant_id)' },
  { key: 'partial', name: `Partial HNSW per tenant`, ddl: `CREATE INDEX ON chunks USING hnsw (embedding vector_cosine_ops) WHERE tenant_id = t, for each of ${TENANT_ROWS.length} tenants` },
  { key: 'none', name: 'None', ddl: 'no index: every branch is a sequential scan' },
];

/** Whether an index serves each part of the query, with the reason, under the chosen filter placement. */
function servable(key: IndexKey | 'none', placement: Placement): Record<'text' | 'vector' | 'filter', [string, string]> {
  switch (key) {
    case 'gin':
      return { text: ['yes', 'posting lists'], vector: ['no', 'no distance order'], filter: ['no', 'column not in it'] };
    case 'hnsw':
      return {
        text: ['no', ''],
        vector: ['yes', 'graph walk'],
        filter: placement === 'inside' ? ['inside', 'accept set in walk'] : placement === 'iterative' ? ['after', 'rescans until k'] : ['after', 'ef_search cap'],
      };
    case 'ivf':
      return { text: ['no', ''], vector: ['yes', 'probed lists'], filter: ['after', placement === 'iterative' ? 'more lists until k' : 'probed lists only'] };
    case 'btree':
      return { text: ['helps', 'BitmapAnd'], vector: ['helps', 'pre-filter, exact'], filter: ['yes', 'sargable ='] };
    case 'partial':
      return { text: ['no', ''], vector: ['yes', 'tenant graph'], filter: ['yes', 'literal tenant only'] };
    default:
      return { text: ['scan', ''], vector: ['scan', ''], filter: ['scan', ''] };
  }
}

const fmtCompact = (n: number) => (n >= 1e6 ? `${fmtNum(n / 1e6, 1)}M` : n >= 1e4 ? `${fmtNum(n / 1e3, 0)}k` : fmtNum(n));

type Stage = { title: string; detail: string; kind: 'seek' | 'scan' | 'filter' | 'result' | 'step'; filter?: boolean };

function vectorStages(r: ReturnType<typeof evaluate>, placement: Placement, set: IndexSet, efSearch: number, probes: number, tenant: number): Stage[] {
  const nt = TENANT_ROWS[tenant];
  const out = `${per(r.returned / QUERIES)} rows`;
  switch (r.planUsed) {
    case 'seq':
      return [
        { title: 'Seq Scan', detail: `${fmtNum(N)} rows read`, kind: 'scan' },
        { title: 'Filter tenant_id', detail: `${fmtNum(nt)} pass`, kind: 'filter', filter: true },
        { title: 'Top-N sort', detail: `${fmtNum(nt)} distances`, kind: 'step' },
        { title: `Limit ${K}`, detail: out, kind: 'result' },
      ];
    case 'btree':
      return [
        { title: 'Index Scan (B-tree)', detail: `tenant_id: ${fmtNum(nt)} rows`, kind: 'seek', filter: true },
        { title: 'Distance per row', detail: `${fmtNum(nt)} exact`, kind: 'step' },
        { title: 'Top-N sort', detail: `keep ${K}`, kind: 'step' },
        { title: `Limit ${K}`, detail: out, kind: 'result' },
      ];
    case 'hnsw':
      if (placement === 'inside')
        return [
          set.btree ? { title: 'Accept set (B-tree)', detail: `${fmtNum(nt)} rows`, kind: 'seek', filter: true } : { title: 'Accept set (scan)', detail: `${fmtNum(N)} rows read`, kind: 'scan', filter: true },
          { title: 'HNSW walk + filter', detail: r.fallbacks === QUERIES ? 'exact fallback' : `${per(r.avgDist)} distances`, kind: 'seek', filter: true },
          { title: `Top ${K}`, detail: `${r.fallbacks} of ${QUERIES} went exact`, kind: 'step' },
          { title: `Limit ${K}`, detail: out, kind: 'result' },
        ];
      return [
        { title: placement === 'iterative' ? 'HNSW iterative scan' : `HNSW scan, ef ${efSearch}`, detail: `${per(r.avgVisited)} tuples visited`, kind: 'seek' },
        { title: 'Heap fetch', detail: `${per(r.avgEmitted)} tuples`, kind: 'step' },
        { title: 'Filter tenant_id', detail: `${per(r.returned / QUERIES)} pass`, kind: 'filter', filter: true },
        { title: `Limit ${K}`, detail: out, kind: 'result' },
      ];
    case 'ivf':
      return [
        { title: `IVFFlat, probes ${probes}`, detail: `${per(r.avgVisited)} rows in lists`, kind: 'seek' },
        { title: 'Sort by distance', detail: `${per(r.avgEmitted)} handed up`, kind: 'step' },
        { title: 'Filter tenant_id', detail: `${per(r.returned / QUERIES)} pass`, kind: 'filter', filter: true },
        { title: `Limit ${K}`, detail: out, kind: 'result' },
      ];
    case 'partial':
      return [
        { title: 'Partial HNSW', detail: `only tenant ${tenant + 1}: ${fmtNum(nt)} rows`, kind: 'seek', filter: true },
        { title: `Scan, ef ${efSearch}`, detail: `${per(r.avgVisited)} tuples visited`, kind: 'step' },
        { title: `Limit ${K}`, detail: out, kind: 'result' },
      ];
  }
}

function textStages(r: ReturnType<typeof evaluate>, tenant: number): Stage[] {
  const nt = TENANT_ROWS[tenant];
  const t = r.text;
  const ranked = `${per(t.avgRanked)} rows ranked`;
  switch (t.plan) {
    case 'gin':
      return [
        { title: 'Bitmap Index Scan', detail: `GIN: ${per(t.avgMatches)} TIDs`, kind: 'seek' },
        { title: 'Bitmap Heap Scan', detail: `${per(t.avgHeap)} heap rows`, kind: 'step' },
        { title: 'Filter tenant_id', detail: `${per(t.avgRanked)} pass`, kind: 'filter', filter: true },
        { title: 'ts_rank + sort', detail: ranked, kind: 'result' },
      ];
    case 'btree':
      return [
        { title: 'Index Scan (B-tree)', detail: `tenant_id: ${fmtNum(nt)} rows`, kind: 'seek', filter: true },
        { title: 'Filter tsv @@ query', detail: `${fmtNum(nt)} rows checked`, kind: 'step' },
        { title: 'ts_rank + sort', detail: ranked, kind: 'result' },
      ];
    case 'and':
      return [
        { title: 'BitmapAnd', detail: `GIN ${per(t.avgMatches)} ∧ B-tree ${fmtNum(nt)}`, kind: 'seek', filter: true },
        { title: 'Bitmap Heap Scan', detail: `${per(t.avgHeap)} heap rows`, kind: 'step' },
        { title: 'ts_rank + sort', detail: ranked, kind: 'result' },
      ];
    default:
      return [
        { title: 'Seq Scan', detail: `${fmtNum(N)} rows read`, kind: 'scan' },
        { title: 'Filter @@, tenant_id', detail: `${per(t.avgRanked)} pass`, kind: 'filter', filter: true },
        { title: 'ts_rank + sort', detail: ranked, kind: 'result' },
      ];
  }
}

const STROKE: Record<Stage['kind'], string> = {
  seek: 'var(--viz-1)',
  scan: 'var(--viz-ink-muted)',
  filter: 'var(--viz-2)',
  step: 'var(--viz-grid)',
  result: 'var(--viz-7)',
};

function Lane({ y, label, sub, stages, W }: { y: number; label: string; sub: string; stages: Stage[]; W: number }) {
  const left = 96;
  const gap = 18;
  const bw = (W - left - gap * (stages.length - 1)) / stages.length;
  return (
    <g>
      <text x={0} y={y + 22} fontSize={12} fill="var(--viz-ink)">
        {label}
      </text>
      <text x={0} y={y + 38} fontSize={10} fill="var(--viz-ink-2)">
        {sub}
      </text>
      {stages.map((s, i) => {
        const x = left + i * (bw + gap);
        return (
          <g key={i}>
            {i > 0 ? (
              <path d={`M ${x - gap + 2} ${y + 26} L ${x - 3} ${y + 26} M ${x - 8} ${y + 21} L ${x - 3} ${y + 26} L ${x - 8} ${y + 31}`} stroke="var(--viz-ink-muted)" fill="none" />
            ) : null}
            <rect x={x} y={y + 4} width={bw} height={44} rx={5} fill="var(--viz-surface)" stroke={s.filter ? 'var(--viz-2)' : STROKE[s.kind]} strokeWidth={s.filter || s.kind === 'seek' || s.kind === 'result' ? 2 : 1.2} strokeDasharray={s.kind === 'scan' ? '5 3' : undefined} />
            {s.filter && s.kind === 'seek' ? <rect x={x + 3} y={y + 7} width={bw - 6} height={38} rx={3} fill="none" stroke="var(--viz-1)" strokeWidth={1.5} /> : null}
            <text x={x + 8} y={y + 22} fontSize={11} fill="var(--viz-ink)">
              {s.title}
            </text>
            <text x={x + 8} y={y + 38} fontSize={10.5} fill="var(--viz-ink-2)">
              {s.detail}
            </text>
          </g>
        );
      })}
    </g>
  );
}

export default function HybridIndexSetLab() {
  const [built, setBuilt] = useState<Built | null>(null);
  const [progress, setProgress] = useState(0);
  const [set, setSet] = useState<IndexSet>({ gin: true, hnsw: true, ivf: false, btree: false, partial: false });
  const [tenant, setTenant] = useState(6);
  const [placement, setPlacement] = useState<Placement>('after');
  const [plan, setPlan] = useState<PlanKey>('auto');
  const [efSearch, setEfSearch] = useState(40);
  const [probes, setProbes] = useState(1);
  const [mst, setMst] = useState(MAX_SCAN_TUPLES);
  const [dims, setDims] = useState<Dims>(768);

  useEffect(() => {
    let cancelled = false;
    let timer: ReturnType<typeof setTimeout> | undefined;
    const corpus = makeCorpus();
    const main = hnswBuilder(corpus, Array.from({ length: N }, (_, i) => i));
    const parts = corpus.tenantRows.map((rows, t) => hnswBuilder(corpus, rows, HNSW_M, HNSW_EFC, 100 + t));
    const queue = [main, ...parts];
    let qi = 0;
    let done = 0;
    const tick = () => {
      if (cancelled) return;
      const start = Date.now();
      while (qi < queue.length && Date.now() - start < 40) {
        const b = queue[qi];
        const before = b.g.updates.length;
        if (b.step(60)) qi++;
        done += b.g.updates.length - before;
      }
      setProgress(done / (2 * N));
      if (qi < queue.length) timer = setTimeout(tick, 0);
      else setBuilt({ corpus, hnsw: main.g, ivf: buildIvf(corpus), partial: parts.map((p) => p.g) });
    };
    timer = setTimeout(tick, 0);
    return () => {
      cancelled = true;
      if (timer) clearTimeout(timer);
    };
  }, []);

  const avail = availablePlans(set);
  const effPlan: PlanKey = plan === 'auto' || avail.includes(plan) ? plan : 'auto';
  const cfg: Config = { set, tenant, placement, plan: effPlan, efSearch, probes, maxScanTuples: mst };
  const r = useMemo(() => (built ? evaluate(built, cfg) : null), [built, set, tenant, placement, effPlan, efSearch, probes, mst]);
  const allPlans = useMemo(
    () => (built ? avail.map((p) => ({ plan: p, r: evaluate(built, { ...cfg, plan: p }) })) : []),
    [built, set, tenant, placement, efSearch, probes, mst],
  );
  const factsAll = useMemo(() => (built ? { 384: indexFacts(built, 384), 768: indexFacts(built, 768), 1536: indexFacts(built, 1536) } : null), [built]);
  const facts = factsAll ? factsAll[dims] : null;

  const checkedKeys = (Object.keys(set) as IndexKey[]).filter((k) => set[k]);
  const totalBytes = facts ? checkedKeys.reduce((s, k) => s + facts[k].bytes, 0) : 0;
  const totalWrites = facts ? 1 + checkedKeys.reduce((s, k) => s + facts[k].entriesPerInsert, 0) : 0;
  const nt = TENANT_ROWS[tenant];

  const W = 680;
  const gridX = 96;
  const cell = 15;
  const gridY = 214;
  const plansX = 400;
  const H = Math.max(gridY + 10 * (cell + 3) + 34, gridY + 4 * 42 + 44);
  const maxWork = Math.max(1, ...allPlans.map((p) => (p.plan === 'seq' ? N : p.r.avgVisited)));

  const toggle = (k: IndexKey) => setSet((s) => ({ ...s, [k]: !s[k] }));

  return (
    <VizPanel
      title="One table, three predicates, one index set"
      subtitle={`A hybrid query over ${fmtNum(N)} chunks: a tsvector match, ORDER BY embedding distance LIMIT ${K}, and tenant_id = t. Tick candidate indexes, tighten the tenant filter, and watch each branch seek or scan — and how many of the true ${K} nearest rows for that tenant the vector branch fails to return.`}
      controls={
        <>
          <Slider label="Filter: tenant_id matches" min={0} max={TENANT_ROWS.length - 1} step={1} value={tenant} onChange={setTenant} format={(t) => `${pct(t)} (${fmtNum(TENANT_ROWS[t])} rows)`} />
          <Choice label="Where the ANN filter runs" value={placement} onChange={setPlacement} options={PLACEMENT_OPTIONS} />
          <Choice
            label="Vector branch plan"
            value={effPlan}
            onChange={setPlan}
            options={[{ value: 'auto' as PlanKey, label: "Planner's choice (lab cost model)" }, ...avail.map((p) => ({ value: p as PlanKey, label: PLAN_LABEL[p] }))]}
          />
          <Slider label={placement === 'inside' ? 'num_candidates' : 'hnsw.ef_search'} min={10} max={200} step={10} value={efSearch} onChange={setEfSearch} />
          <Slider label="ivfflat.probes" min={1} max={IVF_LISTS} step={1} value={probes} onChange={setProbes} disabled={!set.ivf} />
          <Slider label="hnsw.max_scan_tuples" min={500} max={20000} step={500} value={mst} onChange={setMst} format={(v) => fmtNum(v)} disabled={placement !== 'iterative'} />
          <Segmented
            label="Embedding dimensions (sizes)"
            value={String(dims) as '384' | '768' | '1536'}
            onChange={(v) => setDims(Number(v) as Dims)}
            options={[
              { value: '384', label: '384' },
              { value: '768', label: '768' },
              { value: '1536', label: '1536' },
            ]}
          />
        </>
      }
      legend={
        <Legend
          items={[
            { label: 'Index access', color: 'var(--viz-1)' },
            { label: 'Full scan (dashed)', color: 'var(--viz-ink-muted)' },
            { label: 'Where tenant_id is applied', color: 'var(--viz-2)' },
            { label: 'Branch output', color: 'var(--viz-7)' },
            { label: 'True top-10 row returned', color: 'var(--viz-good)' },
            { label: 'True top-10 row missed', color: 'var(--viz-critical)' },
          ]}
        />
      }
      stats={
        r && facts ? (
          <Stats
            items={[
              { label: 'Qualifying rows missed', value: `${r.missed} of ${QUERIES * K}`, hint: `Across ${QUERIES} queries, rows in the exact top ${K} for this tenant that the vector branch did not return.` },
              { label: 'Vector plan', value: r.planUsed === 'hnsw' && placement === 'inside' ? 'HNSW filtered walk' : r.planUsed === 'hnsw' && placement === 'iterative' ? 'HNSW iterative scan' : r.planUsed === 'ivf' && placement === 'iterative' ? 'IVFFlat iterative scan' : PLAN_SHORT[r.planUsed], hint: effPlan === 'auto' ? 'Chosen by the lab planner: pgvector startup estimate plus LIMIT fraction, in tuples. It never sees recall.' : 'Forced.' },
              { label: 'Vector tuples touched / query', value: per(r.planUsed === 'seq' ? N : r.avgVisited) },
              { label: 'Text heap rows / query', value: per(r.text.avgHeap) },
              { label: `Index bytes at 1M rows, ${dims}-d`, value: fmtBytes(totalBytes), hint: 'Projected from tuple layouts on 8 KB pages; see the table.' },
              { label: 'Tuples written per INSERT', value: per(totalWrites), hint: 'One heap tuple plus every index entry an insert writes (GIN entries go to the pending list while fastupdate is on).' },
            ]}
          />
        ) : null
      }
      note={
        <Note>
          {!r || !built ? (
            <>
              <strong>Building the indexes over {fmtNum(N)} rows… {Math.round(progress * 100)}%</strong> — one HNSW graph with m = {HNSW_M}, ef_construction = {HNSW_EFC}, then one per tenant, the way pgvector inserts them.
            </>
          ) : (
            <>
              <strong>
                {r.missed === 0 ? `The vector branch returns all ${QUERIES * K} qualifying rows. ` : `The vector branch misses ${r.missed} of ${QUERIES * K} qualifying rows. `}
              </strong>
              The filter keeps {pct(tenant)} of rows ({fmtNum(nt)}). {r.filterWhere}.{' '}
              {r.planUsed === 'hnsw' && placement === 'after' && r.missed > 0
                ? `About ${fmtNum(efSearch * (nt / N), 1)} of ${efSearch} candidates are expected to belong to this tenant, so most queries run out before ${K} rows pass.`
                : r.planUsed === 'hnsw' && placement === 'iterative'
                  ? `The scan kept going: ${per(r.avgVisited)} tuples visited per query looking for ${K} that pass.${r.capped > 0 ? ` In ${r.capped} of ${QUERIES} queries it reached hnsw.max_scan_tuples, stopped exploring and handed up only candidates it already held.` : r.missed > 0 ? ' The few misses are rows the approximate walk never reached before enough rows passed.' : ''}`
                  : r.planUsed === 'btree'
                    ? `Exact: every qualifying row gets a distance, which costs ${fmtNum(nt)} rows per query — cheap only because the filter is selective.`
                    : r.planUsed === 'seq'
                      ? 'Exact but every query reads the whole table.'
                      : r.planUsed === 'partial'
                        ? 'Exact filtering at approximate-index cost, for as many indexes as there are tenants.'
                        : r.planUsed === 'ivf'
                          ? `IVFFlat hands up every row of the probed list${probes > 1 ? 's' : ''}, so post-filtering starves later than HNSW, but rows in unprobed lists are never seen.`
                          : r.fallbacks > 0
                            ? `${r.fallbacks} of ${QUERIES} queries hit the visit limit (or matched too few rows) and fell back to an exact search over the accept set.`
                            : 'The walk collected enough accepted rows before its visit limit.'}{' '}
              Text branch: {r.text.filterWhere}.
            </>
          )}
        </Note>
      }
      table={
        r && facts ? (
          <>
            <table className="viz-table">
              <thead>
                <tr>
                  <th>Vector plan (available)</th>
                  <th>Planner estimate (tuples)</th>
                  <th>Tuples touched / query</th>
                  <th>Rows returned (of {QUERIES * K})</th>
                  <th>Qualifying rows missed</th>
                </tr>
              </thead>
              <tbody>
                {allPlans.map((p) => (
                  <tr key={p.plan}>
                    <td>
                      {PLAN_LABEL[p.plan]}
                      {p.plan === r.planUsed ? ' (used)' : ''}
                    </td>
                    <td>{fmtNum(r.estimates[p.plan] ?? 0)}</td>
                    <td>{per(p.plan === 'seq' ? N : p.r.avgVisited)}</td>
                    <td>{p.r.returned}</td>
                    <td>{p.r.missed}</td>
                  </tr>
                ))}
              </tbody>
            </table>
            <table className="viz-table">
              <thead>
                <tr>
                  <th>Index</th>
                  <th>Bytes at 1M rows (384 / 768 / 1536-d)</th>
                  <th>Tuples written per insert</th>
                  <th>Build work on this corpus</th>
                </tr>
              </thead>
              <tbody>
                {(['gin', 'hnsw', 'ivf', 'btree', 'partial'] as IndexKey[]).map((k) => {
                  const f = facts[k];
                  const sizes = ([384, 768, 1536] as Dims[]).map((d) => fmtBytes(factsAll![d][k].bytes)).join(' / ');
                  return (
                    <tr key={k}>
                      <td>{ROWS_META.find((m) => m.key === k)!.ddl}</td>
                      <td>{sizes}</td>
                      <td>{per(f.entriesPerInsert)}</td>
                      <td>
                        {fmtNum(f.buildWork)} {f.buildUnit}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </>
        ) : null
      }
    >
      <div style={{ overflowX: 'auto' }}>
        <table className="viz-table" style={{ fontSize: '0.74rem', minWidth: 560 }}>
          <thead>
            <tr>
              <th>Candidate index</th>
              <th>Text @@</th>
              <th>Vector ORDER BY</th>
              <th>tenant_id filter</th>
              <th>Size, 1M rows</th>
              <th>Tuples per insert</th>
              <th>Build work</th>
            </tr>
          </thead>
          <tbody>
            {ROWS_META.map((m) => {
              const sv = servable(m.key, placement);
              const f = facts ? facts[m.key] : null;
              const on = m.key === 'none' ? checkedKeys.length === 0 : set[m.key as IndexKey];
              return (
                <tr key={m.key} style={{ opacity: on ? 1 : 0.6 }} title={m.ddl}>
                  <td style={{ whiteSpace: 'nowrap' }}>
                    {m.key === 'none' ? (
                      <span style={{ color: 'var(--viz-ink-2)', paddingLeft: '1.3rem' }}>None{on ? ' (all scans)' : ''}</span>
                    ) : (
                      <Check label={m.name} checked={set[m.key as IndexKey]} onChange={() => toggle(m.key as IndexKey)} />
                    )}
                  </td>
                  {(['text', 'vector', 'filter'] as const).map((col) => (
                    <td key={col}>
                      <span style={{ color: 'var(--viz-ink)', fontWeight: 600 }}>{sv[col][0]}</span>
                      {sv[col][1] ? (
                        <>
                          <br />
                          <span style={{ color: 'var(--viz-ink-2)' }}>{sv[col][1]}</span>
                        </>
                      ) : null}
                    </td>
                  ))}
                  <td>{f ? (m.key === 'none' ? '0' : fmtBytes(f.bytes)) : '…'}</td>
                  <td>{f ? (m.key === 'none' ? '0' : per(f.entriesPerInsert)) : '…'}</td>
                  <td>{f ? (m.key === 'none' ? '—' : `${fmtCompact(f.buildWork)} ${f.buildUnit === 'distance computations' ? 'distances' : f.buildUnit.split(' ')[0]}`) : '…'}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>

      <svg viewBox={`0 0 ${W} ${H}`} width={W} height={H} role="img" aria-label={r ? `Text branch uses ${r.text.plan}; vector branch uses ${PLAN_LABEL[r.planUsed]} and misses ${r.missed} of ${QUERIES * K} qualifying rows` : 'Indexes are building'} style={{ marginTop: 12, minWidth: 560 }}>
        <text x={0} y={14} fontSize={12} fill="var(--viz-ink)">
          WHERE tenant_id = {tenant + 1} — {fmtNum(nt)} of {fmtNum(N)} rows ({pct(tenant)}); numbers are per query, averaged over {QUERIES} queries
        </text>
        {r ? (
          <>
            <Lane y={24} W={W} label="Text branch" sub="tsv @@ query" stages={textStages(r, tenant)} />
            <Lane y={94} W={W} label="Vector branch" sub={`<=> LIMIT ${K}`} stages={vectorStages(r, placement, set, efSearch, probes, tenant)} />

            <text x={0} y={gridY - 26} fontSize={12} fill="var(--viz-ink)">
              Ground truth: this tenant's exact top {K}, one row per query
            </text>
            {r.outcomes.map((o, qi) => {
              const got = new Set(o.returned);
              return (
                <g key={qi}>
                  <text x={0} y={gridY + qi * (cell + 3) + 11} fontSize={10} fill="var(--viz-ink-2)">
                    query {qi + 1}
                  </text>
                  {o.truth.map((row, ci) => {
                    const hit = got.has(row);
                    const x = gridX + ci * (cell + 3);
                    const y = gridY + qi * (cell + 3);
                    return hit ? (
                      <rect key={ci} x={x} y={y} width={cell} height={cell} rx={2} fill="var(--viz-good)" />
                    ) : (
                      <g key={ci}>
                        <rect x={x + 0.75} y={y + 0.75} width={cell - 1.5} height={cell - 1.5} rx={2} fill="var(--viz-surface)" stroke="var(--viz-critical)" strokeWidth={1.5} />
                        <path d={`M ${x + 4} ${y + 4} L ${x + cell - 4} ${y + cell - 4} M ${x + cell - 4} ${y + 4} L ${x + 4} ${y + cell - 4}`} stroke="var(--viz-critical)" strokeWidth={1.3} />
                      </g>
                    );
                  })}
                  <text x={gridX + K * (cell + 3) + 4} y={gridY + qi * (cell + 3) + 11} fontSize={10} fill="var(--viz-ink-2)">
                    {K - o.missed}/{K}
                  </text>
                </g>
              );
            })}

            <text x={plansX} y={gridY - 26} fontSize={12} fill="var(--viz-ink)">
              Every available vector plan
            </text>
            {allPlans.map((p, i) => {
              const y = gridY + i * 42;
              const work = p.plan === 'seq' ? N : p.r.avgVisited;
              const bw = Math.max(2, (work / maxWork) * (W - plansX - 8));
              const used = p.plan === r.planUsed;
              return (
                <g key={p.plan}>
                  <text x={plansX} y={y + 10} fontSize={11} fill="var(--viz-ink)">
                    {PLAN_LABEL[p.plan]}
                    {used ? ' — used' : ''}
                  </text>
                  <rect x={plansX} y={y + 15} width={bw} height={9} rx={2} fill={p.plan === 'seq' ? 'var(--viz-ink-muted)' : 'var(--viz-1)'} fillOpacity={used ? 1 : 0.45} />
                  <text x={plansX} y={y + 35} fontSize={10} fill="var(--viz-ink-2)">
                    {per(work)} tuples · {p.r.missed} missed
                  </text>
                </g>
              );
            })}
          </>
        ) : (
          <text x={0} y={60} fontSize={12} fill="var(--viz-ink-2)">
            Building HNSW graphs… {Math.round(progress * 100)}%
          </text>
        )}
      </svg>
    </VizPanel>
  );
}
