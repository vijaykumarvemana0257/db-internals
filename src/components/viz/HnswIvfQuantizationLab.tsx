import { useDeferredValue, useMemo, useState, type ReactNode } from 'react';
import { VizPanel, Segmented, Slider, Check, Choice, Button, Legend, Stats, Note, makeRng, fmtNum, fmtBytes, useSize } from './Viz';

/**
 * Vector index model (pure functions, exported for node tests).
 *
 * HNSW follows Malkov & Yashunin (arXiv 1603.09320) as implemented in hnswlib, pgvector and Lucene:
 *  - level = floor(-ln(U) * mL) with mL = 1/ln(M); at most M links per node on upper layers, 2M on layer 0.
 *  - insertion: greedy (ef = 1) descent to the node's top layer, then SEARCH-LAYER with efConstruction on
 *    each layer from there down to 0. The new node keeps up to Mmax(layer) neighbours picked by the paper's
 *    Algorithm 4 heuristic (the pgvector and Lucene convention; hnswlib keeps M); back-links are added and an
 *    overfull neighbour list is re-pruned with the same rule. "Nearest M" replaces the heuristic with Algorithm 3.
 *  - query: greedy on layers L..1 (scan all neighbours, move to any closer one, repeat), then SEARCH-LAYER
 *    on layer 0 with ef = max(efSearch, k): a min-heap of candidates C and a max-heap W bounded at ef, stopping
 *    when the nearest unexpanded candidate is farther than W's farthest member.
 * Distance computations count distances actually computed (the entry point plus every newly visited node, plus
 * the exact re-rank when enabled). Heuristic-pruning distances count toward build cost only.
 * Product quantization splits a vector into m subspaces of `sub` dimensions, each with its own 2^bits-centroid
 * k-means codebook; search ranks by asymmetric distance (exact query vs reconstructed vector). IVF encodes the
 * residual from the cell centroid (IVFADC).
 * Filtered search: post-filter and iterative scan follow pgvector (0.8 iterative scans resume layer-0 search
 * from the discarded-candidate heap, ef_search at a time, relaxed order, no max_scan_tuples cap at this scale);
 * in-graph filtering follows hnswlib's isIdAllowed (non-matching nodes are traversed but never enter W).
 * Model assumptions: synthetic Gaussian-mixture data in a unit cube, squared Euclidean distance, no deletes.
 */

/* ------------------------------------------------------------------ data */

export type Cloud = { n: number; d: number; v: Float64Array; u: Float64Array };

const MIX2 = [
  { c: [0.2, 0.24], s: 0.05, w: 0.16 },
  { c: [0.72, 0.2], s: 0.07, w: 0.14 },
  { c: [0.46, 0.52], s: 0.04, w: 0.1 },
  { c: [0.18, 0.72], s: 0.075, w: 0.14 },
  { c: [0.8, 0.66], s: 0.05, w: 0.14 },
  { c: [0.55, 0.86], s: 0.035, w: 0.08 },
  { c: [0.9, 0.4], s: 0.03, w: 0.06 },
];
const BACKGROUND = 0.18;

function gauss(rng: () => number) {
  const u = Math.max(1e-6, rng());
  const v = rng();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

type Mixture = { c: number[]; s: number; w: number }[];

function mixtureFor(d: number, spread = 1): Mixture {
  if (d === 2) return MIX2;
  const rng = makeRng(1000 + d);
  const out: Mixture = [];
  const k = 10;
  for (let j = 0; j < k; j++) {
    out.push({ c: Array.from({ length: d }, () => 0.2 + 0.6 * rng()), s: (0.05 + 0.07 * rng()) * spread, w: (1 - BACKGROUND) / k });
  }
  return out;
}

function sampleInto(rng: () => number, mix: Mixture, d: number, out: Float64Array, off: number) {
  const r = rng();
  let acc = 0;
  for (const m of mix) {
    acc += m.w;
    if (r < acc) {
      for (let j = 0; j < d; j++) out[off + j] = Math.min(0.99, Math.max(0.01, m.c[j] + m.s * gauss(rng)));
      return;
    }
  }
  for (let j = 0; j < d; j++) out[off + j] = 0.02 + 0.96 * rng();
}

export function makeCloud(n: number, d = 2, seed = 7, spread = 1): Cloud {
  const rng = makeRng(seed + d * 101);
  const tagRng = makeRng(seed * 31 + 5);
  const mix = mixtureFor(d, spread);
  const v = new Float64Array(n * d);
  const u = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    sampleInto(rng, mix, d, v, i * d);
    u[i] = tagRng();
  }
  return { n, d, v, u };
}

/** Held-out queries drawn from the same mixture as the data. */
export function makeQueries(q: number, d = 2, seed = 991, spread = 1): Float64Array[] {
  const rng = makeRng(seed + d);
  const mix = mixtureFor(d, spread);
  const out: Float64Array[] = [];
  for (let i = 0; i < q; i++) {
    const a = new Float64Array(d);
    sampleInto(rng, mix, d, a, 0);
    out.push(a);
  }
  return out;
}

/** Squared distance from query q to row i of a flat n*d array. */
export function distTo(v: Float64Array, d: number, i: number, q: ArrayLike<number>) {
  let s = 0;
  const o = i * d;
  for (let j = 0; j < d; j++) {
    const t = v[o + j] - q[j];
    s += t * t;
  }
  return s;
}

function distRows(v: Float64Array, d: number, a: number, b: number) {
  let s = 0;
  const oa = a * d;
  const ob = b * d;
  for (let j = 0; j < d; j++) {
    const t = v[oa + j] - v[ob + j];
    s += t * t;
  }
  return s;
}

/* ------------------------------------------------------------------ heap */

export type Cand = { id: number; d: number };

class Heap {
  ids: number[] = [];
  ds: number[] = [];
  constructor(private readonly maxHeap: boolean) {}
  get size() {
    return this.ids.length;
  }
  topD() {
    return this.ds[0];
  }
  private better(a: number, b: number) {
    return this.maxHeap ? a > b : a < b;
  }
  push(id: number, d: number) {
    const { ids, ds } = this;
    let i = ids.length;
    ids.push(id);
    ds.push(d);
    while (i > 0) {
      const p = (i - 1) >> 1;
      if (!this.better(ds[i], ds[p])) break;
      [ids[i], ids[p]] = [ids[p], ids[i]];
      [ds[i], ds[p]] = [ds[p], ds[i]];
      i = p;
    }
  }
  pop(): [number, number] {
    const { ids, ds } = this;
    const top: [number, number] = [ids[0], ds[0]];
    const lastId = ids.pop()!;
    const lastD = ds.pop()!;
    if (ids.length > 0) {
      ids[0] = lastId;
      ds[0] = lastD;
      let i = 0;
      for (;;) {
        const l = 2 * i + 1;
        const r = l + 1;
        let m = i;
        if (l < ids.length && this.better(ds[l], ds[m])) m = l;
        if (r < ids.length && this.better(ds[r], ds[m])) m = r;
        if (m === i) break;
        [ids[i], ids[m]] = [ids[m], ids[i]];
        [ds[i], ds[m]] = [ds[m], ds[i]];
        i = m;
      }
    }
    return top;
  }
  sorted(): Cand[] {
    return this.ids.map((id, i) => ({ id, d: this.ds[i] })).sort((a, b) => a.d - b.d || a.id - b.id);
  }
}

type Counter = { comps: number };

/* ------------------------------------------------------------------ HNSW */

export type Hnsw = {
  M: number;
  efC: number;
  heuristic: boolean;
  levels: Int8Array;
  /** links[node][layer] = neighbour ids */
  links: number[][][];
  entry: number;
  maxLevel: number;
  layerCounts: number[];
  edgeCount: number[];
  buildComps: number;
  visited: Int32Array;
  stamp: number;
};

function selectNeighbours(cloud: Cloud, cands: Cand[], max: number, heuristic: boolean, cnt: Counter): number[] {
  if (cands.length <= max) return cands.map((c) => c.id);
  if (!heuristic) return cands.slice(0, max).map((c) => c.id);
  const out: number[] = [];
  for (const c of cands) {
    if (out.length >= max) break;
    let good = true;
    for (const r of out) {
      cnt.comps++;
      if (distRows(cloud.v, cloud.d, r, c.id) < c.d) {
        good = false;
        break;
      }
    }
    if (good) out.push(c.id);
  }
  return out;
}

export type Expansion = { node: number; w: Cand[]; cSize: number; comps: number; visited: number };

type LayerOpts = {
  accept?: (id: number) => boolean;
  discarded?: Heap;
  trace?: Expansion[];
  visitedList?: number[];
};

function searchLayer(g: Hnsw, eps: Cand[], ef: number, lc: number, dist: (id: number) => number, cnt: Counter, opts: LayerOpts = {}): Cand[] {
  const { accept, discarded, trace, visitedList } = opts;
  const visited = g.visited;
  const stamp = g.stamp;
  const C = new Heap(false);
  const W = new Heap(true);
  for (const ep of eps) {
    if (visited[ep.id] !== stamp) {
      visited[ep.id] = stamp;
      visitedList?.push(ep.id);
    }
    C.push(ep.id, ep.d);
    if (!accept || accept(ep.id)) W.push(ep.id, ep.d);
  }
  while (W.size > ef) {
    const [id, d] = W.pop();
    discarded?.push(id, d);
  }
  let lower = W.size ? W.topD() : Infinity;
  while (C.size) {
    if (C.topD() > lower && (!accept || W.size >= ef)) break;
    const [cur] = C.pop();
    for (const nb of g.links[cur][lc] ?? []) {
      if (visited[nb] === stamp) continue;
      visited[nb] = stamp;
      visitedList?.push(nb);
      const d = dist(nb);
      cnt.comps++;
      if (W.size < ef || d < lower) {
        C.push(nb, d);
        if (!accept || accept(nb)) W.push(nb, d);
        while (W.size > ef) {
          const [id, dd] = W.pop();
          discarded?.push(id, dd);
        }
        if (W.size) lower = W.topD();
      } else {
        discarded?.push(nb, d);
      }
    }
    trace?.push({ node: cur, w: W.sorted(), cSize: C.size, comps: cnt.comps, visited: visitedList ? visitedList.length : 0 });
  }
  return W.sorted();
}

function greedy(g: Hnsw, ep: Cand, lc: number, dist: (id: number) => number, cnt: Counter, path?: number[]): Cand {
  let cur = ep.id;
  let curD = ep.d;
  let changed = true;
  path?.push(cur);
  while (changed) {
    changed = false;
    for (const nb of g.links[cur][lc] ?? []) {
      const d = dist(nb);
      cnt.comps++;
      if (d < curD) {
        curD = d;
        cur = nb;
        changed = true;
      }
    }
    if (changed) path?.push(cur);
  }
  return { id: cur, d: curD };
}

export function buildHnsw(cloud: Cloud, M: number, efC: number, heuristic = true, seed = 3): Hnsw {
  const { n, v, d } = cloud;
  const rng = makeRng(seed);
  const mL = 1 / Math.log(Math.max(2, M));
  const levels = new Int8Array(n);
  const links: number[][][] = new Array(n);
  const g: Hnsw = { M, efC, heuristic, levels, links, entry: -1, maxLevel: -1, layerCounts: [], edgeCount: [], buildComps: 0, visited: new Int32Array(n), stamp: 0 };
  const cnt: Counter = { comps: 0 };
  for (let i = 0; i < n; i++) {
    const l = Math.min(12, Math.floor(-Math.log(1 - rng()) * mL));
    levels[i] = l;
    links[i] = Array.from({ length: l + 1 }, () => []);
    if (g.entry < 0) {
      g.entry = i;
      g.maxLevel = l;
      continue;
    }
    const dist = (id: number) => distRows(v, d, i, id);
    let ep: Cand = { id: g.entry, d: dist(g.entry) };
    cnt.comps++;
    for (let lc = g.maxLevel; lc > l; lc--) ep = greedy(g, ep, lc, dist, cnt);
    let eps: Cand[] = [ep];
    for (let lc = Math.min(l, g.maxLevel); lc >= 0; lc--) {
      g.stamp++;
      const W = searchLayer(g, eps, efC, lc, dist, cnt);
      const max = lc === 0 ? 2 * M : M;
      const chosen = selectNeighbours(cloud, W, max, heuristic, cnt);
      links[i][lc] = chosen;
      for (const e of chosen) {
        const list = links[e][lc];
        list.push(i);
        if (list.length > max) {
          const cands = list.map((id) => ({ id, d: distRows(v, d, e, id) })).sort((a, b) => a.d - b.d || a.id - b.id);
          cnt.comps += list.length;
          links[e][lc] = selectNeighbours(cloud, cands, max, heuristic, cnt);
        }
      }
      eps = W;
    }
    if (l > g.maxLevel) {
      g.maxLevel = l;
      g.entry = i;
    }
  }
  g.buildComps = cnt.comps;
  g.layerCounts = Array.from({ length: g.maxLevel + 1 }, (_, lc) => {
    let c = 0;
    for (let i = 0; i < n; i++) if (levels[i] >= lc) c++;
    return c;
  });
  g.edgeCount = Array.from({ length: g.maxLevel + 1 }, (_, lc) => {
    let c = 0;
    for (let i = 0; i < n; i++) if (levels[i] >= lc) c += links[i][lc].length;
    return c;
  });
  return g;
}

export type HnswTrace = {
  res: Cand[];
  comps: number;
  visited: number[];
  upper: { layer: number; path: number[]; comps: number }[];
  layer0Entry: number;
  expansions: Expansion[];
  candidates: Cand[];
};

/** One k-NN query. `dist` ranks nodes (exact or PQ); `rerank` rescores the ef candidates with `exact`. */
export function searchHnsw(
  g: Hnsw,
  dist: (id: number) => number,
  k: number,
  efSearch: number,
  opts: { exact?: (id: number) => number; rerank?: boolean; trace?: boolean; accept?: (id: number) => boolean } = {},
): HnswTrace {
  const cnt: Counter = { comps: 0 };
  const trace = opts.trace;
  const upper: HnswTrace['upper'] = [];
  let ep: Cand = { id: g.entry, d: dist(g.entry) };
  cnt.comps++;
  for (let lc = g.maxLevel; lc > 0; lc--) {
    const before = cnt.comps;
    const path: number[] = [];
    ep = greedy(g, ep, lc, dist, cnt, trace ? path : undefined);
    if (trace) upper.push({ layer: lc, path, comps: cnt.comps - before });
  }
  g.stamp++;
  const visited: number[] = [];
  const expansions: Expansion[] = [];
  const ef = Math.max(efSearch, k);
  const W = searchLayer(g, [ep], ef, 0, dist, cnt, { accept: opts.accept, trace: trace ? expansions : undefined, visitedList: visited });
  let res = W;
  if (opts.rerank && opts.exact) {
    const ex = opts.exact;
    res = W.map((c) => ({ id: c.id, d: ex(c.id) })).sort((a, b) => a.d - b.d || a.id - b.id);
    cnt.comps += W.length;
  }
  return { res: res.slice(0, k), comps: cnt.comps, visited, upper, layer0Entry: ep.id, expansions, candidates: W };
}

/* ---------------------------------------------------------- filtered search */

export type FilterStrategy = 'post' | 'iterative' | 'ingraph' | 'exact';

export function filteredSearch(g: Hnsw, cloud: Cloud, q: ArrayLike<number>, k: number, efSearch: number, sel: number, strategy: FilterStrategy) {
  const { v, d, u, n } = cloud;
  const pass = (id: number) => u[id] < sel;
  const dist = (id: number) => distTo(v, d, id, q);
  let matching = 0;
  for (let i = 0; i < n; i++) if (pass(i)) matching++;
  const ef = Math.max(efSearch, k);
  if (strategy === 'exact') {
    const res: Cand[] = [];
    for (let i = 0; i < n; i++) if (pass(i)) res.push({ id: i, d: dist(i) });
    res.sort((a, b) => a.d - b.d || a.id - b.id);
    return { res: res.slice(0, k), comps: matching, visited: [] as number[], batches: 0, matching, emitted: matching };
  }
  if (strategy === 'ingraph') {
    const t = searchHnsw(g, dist, k, efSearch, { accept: pass });
    return { res: t.res, comps: t.comps, visited: t.visited, batches: 1, matching, emitted: t.candidates.length };
  }
  // Post-filter and the pgvector iterative scan share one unfiltered graph search; the WHERE clause runs on its output.
  const cnt: Counter = { comps: 0 };
  let ep: Cand = { id: g.entry, d: dist(g.entry) };
  cnt.comps++;
  for (let lc = g.maxLevel; lc > 0; lc--) ep = greedy(g, ep, lc, dist, cnt);
  g.stamp++;
  const visited: number[] = [];
  const discarded = strategy === 'iterative' ? new Heap(false) : undefined;
  let batch = searchLayer(g, [ep], ef, 0, dist, cnt, { discarded, visitedList: visited });
  const out: Cand[] = [];
  let emitted = 0;
  let batches = 1;
  for (;;) {
    for (const c of batch) {
      emitted++;
      if (pass(c.id)) out.push(c);
      if (out.length >= k) break;
    }
    if (out.length >= k || !discarded || discarded.size === 0) break;
    const eps: Cand[] = [];
    while (eps.length < ef && discarded.size) {
      const [id, dd] = discarded.pop();
      eps.push({ id, d: dd });
    }
    batches++;
    batch = searchLayer(g, eps, ef, 0, dist, cnt, { discarded, visitedList: visited });
  }
  return { res: out.slice(0, k), comps: cnt.comps, visited, batches, matching, emitted };
}

/* ------------------------------------------------------------ k-means + PQ */

/** Lloyd's k-means over `n` rows of `d` dims, initialised from distinct sampled rows (as FAISS does). */
export function kmeans(v: Float64Array, n: number, d: number, K: number, seed: number, iters = 25) {
  const k = Math.max(1, Math.min(K, n));
  const rng = makeRng(seed);
  const c = new Float64Array(k * d);
  const taken = new Set<number>();
  for (let j = 0; j < k; j++) {
    let p = Math.floor(rng() * n);
    while (taken.has(p)) p = (p + 1) % n;
    taken.add(p);
    for (let t = 0; t < d; t++) c[j * d + t] = v[p * d + t];
  }
  const assign = new Int32Array(n).fill(-1);
  const sum = new Float64Array(k * d);
  const cnt = new Int32Array(k);
  for (let it = 0; it < iters; it++) {
    sum.fill(0);
    cnt.fill(0);
    let changed = 0;
    for (let i = 0; i < n; i++) {
      let best = 0;
      let bd = Infinity;
      for (let j = 0; j < k; j++) {
        let s = 0;
        for (let t = 0; t < d; t++) {
          const x = v[i * d + t] - c[j * d + t];
          s += x * x;
        }
        if (s < bd) {
          bd = s;
          best = j;
        }
      }
      if (assign[i] !== best) changed++;
      assign[i] = best;
      cnt[best]++;
      for (let t = 0; t < d; t++) sum[best * d + t] += v[i * d + t];
    }
    for (let j = 0; j < k; j++) {
      if (cnt[j] > 0) {
        for (let t = 0; t < d; t++) c[j * d + t] = sum[j * d + t] / cnt[j];
      } else {
        // empty cell: re-seed it at the row farthest from its centroid
        let far = 0;
        let fd = -1;
        for (let i = 0; i < n; i++) {
          let s = 0;
          for (let t = 0; t < d; t++) s += (v[i * d + t] - c[assign[i] * d + t]) ** 2;
          if (s > fd) {
            fd = s;
            far = i;
          }
        }
        for (let t = 0; t < d; t++) c[j * d + t] = v[far * d + t];
      }
    }
    if (changed === 0) break;
  }
  return { k, c, assign };
}

export type Pq = { bits: number; m: number; sub: number; recon: Float64Array; mse: number; codeBits: number };

/**
 * Product quantizer: m = d / sub subspaces, 2^bits centroids each, trained on the rows (or on their residuals
 * from `base`, for IVFADC). `recon` holds each row's reconstruction; distances to it are the ADC estimates.
 */
export function trainPq(cloud: Cloud, bits: number, sub: number, base?: Float64Array, seed = 17): Pq {
  const { n, d, v } = cloud;
  const m = Math.floor(d / sub);
  const K = 1 << bits;
  const recon = new Float64Array(n * d);
  const part = new Float64Array(n * sub);
  for (let s = 0; s < m; s++) {
    for (let i = 0; i < n; i++) for (let t = 0; t < sub; t++) part[i * sub + t] = v[i * d + s * sub + t] - (base ? base[i * d + s * sub + t] : 0);
    const km = kmeans(part, n, sub, K, seed + s * 7, 20);
    for (let i = 0; i < n; i++) {
      for (let t = 0; t < sub; t++) recon[i * d + s * sub + t] = km.c[km.assign[i] * sub + t] + (base ? base[i * d + s * sub + t] : 0);
    }
  }
  let se = 0;
  for (let i = 0; i < n * d; i++) se += (recon[i] - v[i]) ** 2;
  return { bits, m, sub, recon, mse: se / n, codeBits: m * bits };
}

/* -------------------------------------------------------------------- IVF */

export type Ivf = { nlist: number; cents: Float64Array; assign: Int32Array; lists: number[][]; base: Float64Array };

export function buildIvf(cloud: Cloud, nlist: number, seed = 11): Ivf {
  const { n, d, v } = cloud;
  const km = kmeans(v, n, d, nlist, seed, 25);
  const lists: number[][] = Array.from({ length: km.k }, () => []);
  const base = new Float64Array(n * d);
  for (let i = 0; i < n; i++) {
    lists[km.assign[i]].push(i);
    for (let t = 0; t < d; t++) base[i * d + t] = km.c[km.assign[i] * d + t];
  }
  return { nlist: km.k, cents: km.c, assign: km.assign, lists, base };
}

/** IVF query: rank centroids, scan the nprobe nearest lists (exact rows or PQ reconstructions), keep top kk. */
export function searchIvf(ivf: Ivf, cloud: Cloud, vecs: Float64Array, q: ArrayLike<number>, k: number, nprobe: number, rerank = false, rerankFactor = 4) {
  const d = cloud.d;
  const order = Array.from({ length: ivf.nlist }, (_, j) => ({ j, d: distTo(ivf.cents, d, j, q) })).sort((a, b) => a.d - b.d || a.j - b.j);
  const probed = order.slice(0, Math.max(1, Math.min(nprobe, ivf.nlist))).map((o) => o.j);
  const kk = rerank ? k * rerankFactor : k;
  const W = new Heap(true);
  let scanned = 0;
  for (const j of probed) {
    for (const id of ivf.lists[j]) {
      const dd = distTo(vecs, d, id, q);
      scanned++;
      if (W.size < kk) W.push(id, dd);
      else if (dd < W.topD()) {
        W.pop();
        W.push(id, dd);
      }
    }
  }
  let res = W.sorted();
  let comps = ivf.nlist + scanned;
  if (rerank) {
    res = res.map((c) => ({ id: c.id, d: distTo(cloud.v, d, c.id, q) })).sort((a, b) => a.d - b.d || a.id - b.id);
    comps += W.size;
  }
  return { res: res.slice(0, k), probed, scanned, comps };
}

/** Voronoi cell of each 2-D centroid inside the unit square, by half-plane clipping. */
export function voronoi(cents: Float64Array): [number, number][][] {
  const k = cents.length / 2;
  const cells: [number, number][][] = [];
  for (let i = 0; i < k; i++) {
    let poly: [number, number][] = [
      [0, 0],
      [1, 0],
      [1, 1],
      [0, 1],
    ];
    for (let j = 0; j < k && poly.length; j++) {
      if (j === i) continue;
      const ax = cents[2 * j] - cents[2 * i];
      const ay = cents[2 * j + 1] - cents[2 * i + 1];
      const b = (cents[2 * j] ** 2 + cents[2 * j + 1] ** 2 - cents[2 * i] ** 2 - cents[2 * i + 1] ** 2) / 2;
      const f = (p: [number, number]) => ax * p[0] + ay * p[1] - b;
      const next: [number, number][] = [];
      for (let t = 0; t < poly.length; t++) {
        const P = poly[t];
        const Q = poly[(t + 1) % poly.length];
        const fp = f(P);
        const fq = f(Q);
        if (fp <= 0) next.push(P);
        if (fp <= 0 !== fq <= 0) {
          const s = fp / (fp - fq);
          next.push([P[0] + s * (Q[0] - P[0]), P[1] + s * (Q[1] - P[1])]);
        }
      }
      poly = next;
    }
    cells.push(poly);
  }
  return cells;
}

/* ------------------------------------------------------------ evaluation */

export function exactKnn(cloud: Cloud, q: ArrayLike<number>, k: number, accept?: (id: number) => boolean): Cand[] {
  const res: Cand[] = [];
  for (let i = 0; i < cloud.n; i++) {
    if (accept && !accept(i)) continue;
    res.push({ id: i, d: distTo(cloud.v, cloud.d, i, q) });
  }
  res.sort((a, b) => a.d - b.d || a.id - b.id);
  return res.slice(0, k);
}

export function recallOf(res: Cand[], truth: Cand[]) {
  if (truth.length === 0) return 1;
  const t = new Set(truth.map((c) => c.id));
  let hit = 0;
  for (const c of res) if (t.has(c.id)) hit++;
  return hit / truth.length;
}

export type SweepPoint = { param: number; recall: number; comps: number; returned?: number };

export const EF_SWEEP = [10, 12, 16, 20, 30, 40, 60, 80, 120, 160, 240];

export function hnswSweep(g: Hnsw, cloud: Cloud, queries: Float64Array[], k: number, efs: number[], pq?: Pq, rerank = false): SweepPoint[] {
  const truths = queries.map((q) => exactKnn(cloud, q, k));
  const vecs = pq ? pq.recon : cloud.v;
  return efs.map((ef) => {
    let rec = 0;
    let comps = 0;
    queries.forEach((q, qi) => {
      const dist = (id: number) => distTo(vecs, cloud.d, id, q);
      const exact = (id: number) => distTo(cloud.v, cloud.d, id, q);
      const t = searchHnsw(g, dist, k, ef, { exact, rerank: !!pq && rerank });
      rec += recallOf(t.res, truths[qi]);
      comps += t.comps;
    });
    return { param: ef, recall: rec / queries.length, comps: comps / queries.length };
  });
}

export function nprobeValues(nlist: number) {
  const out = new Set<number>();
  for (let p = 1; p <= nlist; p = Math.max(p + 1, Math.round(p * 1.5))) out.add(p);
  out.add(nlist);
  return [...out].sort((a, b) => a - b);
}

export function ivfSweep(ivf: Ivf, cloud: Cloud, queries: Float64Array[], k: number, nprobes: number[], pq?: Pq, rerank = false): SweepPoint[] {
  const truths = queries.map((q) => exactKnn(cloud, q, k));
  const vecs = pq ? pq.recon : cloud.v;
  return nprobes.map((np) => {
    let rec = 0;
    let comps = 0;
    queries.forEach((q, qi) => {
      const r = searchIvf(ivf, cloud, vecs, q, k, np, !!pq && rerank);
      rec += recallOf(r.res, truths[qi]);
      comps += r.comps;
    });
    return { param: np, recall: rec / queries.length, comps: comps / queries.length };
  });
}

export function filteredSweep(g: Hnsw, cloud: Cloud, queries: Float64Array[], k: number, efs: number[], sel: number, strategy: FilterStrategy): SweepPoint[] {
  const accept = (id: number) => cloud.u[id] < sel;
  const truths = queries.map((q) => exactKnn(cloud, q, k, accept));
  const list = strategy === 'exact' ? [efs[0]] : efs;
  return list.map((ef) => {
    let rec = 0;
    let comps = 0;
    let returned = 0;
    queries.forEach((q, qi) => {
      const r = filteredSearch(g, cloud, q, k, ef, sel, strategy);
      rec += recallOf(r.res, truths[qi]);
      comps += r.comps;
      returned += r.res.length;
    });
    return { param: ef, recall: rec / queries.length, comps: comps / queries.length, returned: returned / queries.length };
  });
}

/* ===================================================================== UI */

const K = 10;
const MAP_N = 500;
const CHART_DATA = {
  d16: { n: 2000, d: 16, spread: 3, sub: 2, label: '16-D clustered vectors, n = 2,000' },
  d2: { n: MAP_N, d: 2, spread: 1, sub: 1, label: 'the 2-D map, n = 500' },
} as const;
type ChartKey = keyof typeof CHART_DATA;
const CHART_QUERIES = 60;
const FILTER_QUERIES = 30;
const FILTER_EFS = [10, 20, 40, 80, 160, 240];

const PRESETS: { value: string; label: string; q?: [number, number] }[] = [
  { value: 'between', label: 'Between two clusters', q: [0.47, 0.37] },
  { value: 'dense', label: 'Inside a dense cluster', q: [0.2, 0.25] },
  { value: 'corner', label: 'Empty corner', q: [0.96, 0.95] },
  { value: 'custom', label: 'Clicked on the map' },
];

const SIDE = 440;
const PAD = 12;
const MAPW = SIDE + 2 * PAD;
const px = (x: number) => PAD + x * SIDE;
const py = (y: number) => PAD + (1 - y) * SIDE;
const eu = (sq: number) => Math.sqrt(sq);

function dots(v: Float64Array, ids: Iterable<number>, r: number) {
  let s = '';
  for (const i of ids) {
    const x = px(v[2 * i]);
    const y = py(v[2 * i + 1]);
    s += `M${(x - r).toFixed(1)} ${y.toFixed(1)}a${r} ${r} 0 1 0 ${2 * r} 0a${r} ${r} 0 1 0 ${-2 * r} 0`;
  }
  return s;
}

function edgesPath(g: Hnsw, v: Float64Array, layer: number) {
  const seen = new Set<number>();
  let s = '';
  const n = g.levels.length;
  for (let i = 0; i < n; i++) {
    if (g.levels[i] < layer) continue;
    for (const j of g.links[i][layer]) {
      const key = i < j ? i * n + j : j * n + i;
      if (seen.has(key)) continue;
      seen.add(key);
      s += `M${px(v[2 * i]).toFixed(1)} ${py(v[2 * i + 1]).toFixed(1)}L${px(v[2 * j]).toFixed(1)} ${py(v[2 * j + 1]).toFixed(1)}`;
    }
  }
  return s;
}

type Series = { key: string; label: string; color: string; points: SweepPoint[]; current?: SweepPoint; currentLabel?: string; dashed?: boolean; tag?: string; tagDy?: number };

function RecallSweepChart({ series, n, title, points }: { series: Series[]; n: number; title: string; points?: { label: string; p: SweepPoint }[] }) {
  const [wrapRef, wrapWidth] = useSize(700);
  const W = Math.round(Math.max(380, Math.min(700, wrapWidth)));
  const H = 256;
  const L = 50;
  const R = 18;
  const T = 28;
  const B = 42;
  const all = [...series.flatMap((s) => s.points.map((p) => p.comps)), n, ...(points ?? []).map((p) => p.p.comps)];
  const recalls = [...series.flatMap((s) => s.points.map((p) => p.recall)), ...(points ?? []).map((p) => p.p.recall)];
  const maxX = Math.max(...all) * 1.12;
  const minX = [500, 200, 100, 50, 20, 10].find((t) => t <= Math.min(...all) * 0.8) ?? 10;
  const minR = Math.min(1, ...recalls);
  const lowR = minR >= 0.85 ? 0.8 : minR >= 0.45 ? 0.4 : 0;
  const lx = (c: number) => Math.log10(Math.max(minX, c));
  const X = (c: number) => L + ((lx(c) - lx(minX)) / (lx(maxX) - lx(minX))) * (W - L - R);
  const Y = (r: number) => T + ((1 - r) / (1 - lowR)) * (H - T - B);
  const xticks = [10, 20, 50, 100, 200, 500, 1000, 2000, 5000].filter((t) => t >= minX && t <= maxX);
  const yticks = [0, 1, 2, 3, 4].map((i) => lowR + (i * (1 - lowR)) / 4);
  // Direct labels: series tags first, then each current-point label takes the first offset that overlaps nothing placed so far.
  type Box = { x0: number; x1: number; y0: number; y1: number };
  const boxOf = (x: number, y: number, text: string, anchor: 'start' | 'end'): Box => {
    const w = text.length * 6.2;
    const x0 = anchor === 'end' ? x - w : x;
    return { x0, x1: x0 + w, y0: y - 10, y1: y + 3 };
  };
  const exactText = `exact scan: ${fmtNum(n)}`;
  const marks = [...series.flatMap((s) => [...s.points, ...(s.current ? [s.current] : [])]), ...(points ?? []).map((p) => p.p)].map((p) => [X(p.comps), Y(p.recall)]);
  const clear = (b: Box) => !marks.some(([mx, my]) => mx > b.x0 - 8 && mx < b.x1 + 8 && my > b.y0 - 8 && my < b.y1 + 8);
  const exactY = [H - B - 6, T + 12, Math.round((T + H - B) / 2)].find((y) => clear(boxOf(X(n) - 5, y, exactText, 'end'))) ?? H - B - 6;
  const placed: Box[] = [boxOf(X(n) - 5, exactY, exactText, 'end')];
  for (const s of series) if (s.tag && s.points.length) placed.push(boxOf(X(s.points[0].comps) - 6, Y(s.points[0].recall) + (s.tagDy ?? -8), s.tag, 'end'));
  const hitsPlaced = (b: Box) => placed.some((o) => b.x0 < o.x1 && b.x1 > o.x0 && b.y0 < o.y1 && b.y1 > o.y0);
  const inPlot = (b: Box) => b.y0 >= 4 && b.y1 <= H - B + 12 && b.x0 >= L - 40 && b.x1 <= W;
  const pointPos = (points ?? []).map(({ label, p }) => {
    const px0 = X(p.comps);
    const py0 = Y(p.recall);
    const near: 'start' | 'end' = px0 < L + 130 ? 'start' : 'end';
    const far: 'start' | 'end' = near === 'start' ? 'end' : 'start';
    const tries: ['start' | 'end', number][] = [[near, 20], [near, -12], [far, 20], [far, -12], [near, 34], [near, 48]];
    const [anchor, dy] = tries.find(([a, d]) => {
      const b = boxOf(px0 + (a === 'start' ? 10 : -10), py0 + d, label, a);
      return inPlot(b) && !hitsPlaced(b);
    }) ?? tries[0];
    const x = px0 + (anchor === 'start' ? 10 : -10);
    placed.push(boxOf(x, py0 + dy, label, anchor));
    return { x, y: py0 + dy, anchor };
  });
  const currentPos = new Map<string, { x: number; y: number; anchor: 'start' | 'end' }>();
  for (const s of series) {
    if (!s.current) continue;
    const cx = X(s.current.comps);
    const cy = Y(s.current.recall);
    const pref: 'start' | 'end' = cx > W - R - 90 ? 'end' : 'start';
    const other: 'start' | 'end' = pref === 'end' ? 'start' : 'end';
    const label = s.currentLabel ?? '';
    const tries: ['start' | 'end', number][] = [[pref, 14], [pref, -10], [pref, 28], [pref, -24], [other, 14], [other, -10], [pref, 42], [pref, 56], [pref, 70], [pref, -38]];
    const [anchor, dy] = tries.find(([an, d]) => {
      const b = boxOf(cx + (an === 'end' ? -10 : 10), cy + d, label, an);
      return inPlot(b) && !hitsPlaced(b);
    }) ?? tries[0];
    const x = cx + (anchor === 'end' ? -10 : 10);
    placed.push(boxOf(x, cy + dy, label, anchor));
    currentPos.set(s.key, { x, y: cy + dy, anchor });
  }
  return (
    <div ref={wrapRef} style={{ marginTop: 12 }}>
    <p style={{ margin: '0 0 2px', fontSize: '0.78rem', color: 'var(--viz-ink-2)' }}>{title}</p>
    <svg viewBox={`0 0 ${W} ${H}`} width={W} height={H} role="img" aria-label={`${title}: recall at 10 against distance computations per query`}>
      {yticks.map((t) => (
        <g key={t}>
          <line className="viz-grid-line" x1={L} x2={W - R} y1={Y(t)} y2={Y(t)} />
          <text x={L - 8} y={Y(t) + 4} textAnchor="end">
            {t.toFixed(2)}
          </text>
        </g>
      ))}
      {xticks.map((t) => (
        <g key={t}>
          <line className="viz-grid-line" x1={X(t)} x2={X(t)} y1={T} y2={H - B} />
          <text x={X(t)} y={H - B + 15} textAnchor="middle">
            {fmtNum(t)}
          </text>
        </g>
      ))}
      <line className="viz-axis-line" x1={L} x2={W - R} y1={H - B} y2={H - B} />
      <line className="viz-axis-line" x1={L} x2={L} y1={T} y2={H - B} />
      <text x={(L + W - R) / 2} y={H - 6} textAnchor="middle">
        distance computations per query (log scale)
      </text>
      <text x={14} y={(T + H - B) / 2} textAnchor="middle" transform={`rotate(-90 14 ${(T + H - B) / 2})`}>
        recall@10
      </text>
      <line x1={X(n)} x2={X(n)} y1={T} y2={H - B} stroke="var(--viz-ink-2)" strokeDasharray="5 4" />
      <text x={X(n) - 5} y={exactY} textAnchor="end" fill="var(--viz-ink-2)">
        {exactText}
      </text>
      {series.map((s) => (
        <g key={s.key}>
          <polyline points={s.points.map((p) => `${X(p.comps).toFixed(1)},${Y(p.recall).toFixed(1)}`).join(' ')} fill="none" stroke={s.color} strokeWidth={2} strokeDasharray={s.dashed ? '6 4' : undefined} />
          {s.points.map((p) => (
            <circle key={p.param} cx={X(p.comps)} cy={Y(p.recall)} r={2.6} fill={s.dashed ? 'var(--viz-surface)' : s.color} stroke={s.color} strokeWidth={1.2} />
          ))}
          {s.tag && s.points.length ? (
            <text x={X(s.points[0].comps) - 6} y={Y(s.points[0].recall) + (s.tagDy ?? -8)} textAnchor="end" fill="var(--viz-ink)" stroke="var(--viz-surface)" strokeWidth={3} paintOrder="stroke">
              {s.tag}
            </text>
          ) : null}
          {s.current ? (
            <g>
              <circle cx={X(s.current.comps)} cy={Y(s.current.recall)} r={7} fill="none" stroke="var(--viz-ink)" strokeWidth={1.8} />
              <text x={currentPos.get(s.key)?.x} y={currentPos.get(s.key)?.y} textAnchor={currentPos.get(s.key)?.anchor} fill="var(--viz-ink)" stroke="var(--viz-surface)" strokeWidth={3} paintOrder="stroke">
                {s.currentLabel}
              </text>
            </g>
          ) : null}
        </g>
      ))}
      {(points ?? []).map(({ label, p }, i) => (
        <g key={label}>
          <path d={`M${X(p.comps)} ${Y(p.recall) - 7}l7 7l-7 7l-7 -7z`} fill="var(--viz-surface)" stroke="var(--viz-ink-2)" strokeWidth={1.8} />
          <text x={pointPos[i].x} y={pointPos[i].y} textAnchor={pointPos[i].anchor} fill="var(--viz-ink-2)" stroke="var(--viz-surface)" strokeWidth={3} paintOrder="stroke">
            {label}
          </text>
        </g>
      ))}
    </svg>
    </div>
  );
}

type Tab = 'hnsw' | 'ivf' | 'filter';
type Step = { kind: 'upper'; idx: number } | { kind: 'layer0'; upto: number } | { kind: 'done' };

const STRATEGY_LABEL: Record<FilterStrategy, string> = {
  post: 'Post-filter',
  iterative: 'Iterative scan',
  ingraph: 'In-graph filter',
  exact: 'Exact over matching rows',
};

export default function HnswIvfQuantizationLab() {
  const [tab, setTab] = useState<Tab>('hnsw');
  const [chartKey, setChartKey] = useState<ChartKey>('d16');
  const [M, setM] = useState(16);
  const [efC, setEfC] = useState(64);
  const [efSearch, setEfSearch] = useState(40);
  const [heuristic, setHeuristic] = useState(true);
  const [pqOn, setPqOn] = useState(false);
  const [bits, setBits] = useState(4);
  const [rerank, setRerank] = useState(false);
  const [preset, setPreset] = useState('between');
  const [q, setQ] = useState<[number, number]>([0.47, 0.37]);
  const [step, setStep] = useState(99);
  const [layerView, setLayerView] = useState(0);
  const [nlist, setNlist] = useState(16);
  const [nprobe, setNprobe] = useState(2);
  const [selPct, setSelPct] = useState(10);
  const [strategy, setStrategy] = useState<FilterStrategy>('post');

  const dM = useDeferredValue(M);
  const dEfC = useDeferredValue(efC);
  const dHeur = useDeferredValue(heuristic);
  const dBits = useDeferredValue(bits);
  const dNlist = useDeferredValue(nlist);
  const dSel = useDeferredValue(selPct / 100);
  const dChart = useDeferredValue(chartKey);
  const dEf = useDeferredValue(efSearch);
  const dNprobe = useDeferredValue(nprobe);
  const cfg = CHART_DATA[dChart];
  const sel = selPct / 100;

  /* ---- map (2-D, drawn) ---- */
  const mapCloud = useMemo(() => makeCloud(MAP_N, 2), []);
  const mapGraph = useMemo(() => buildHnsw(mapCloud, dM, dEfC, dHeur), [mapCloud, dM, dEfC, dHeur]);
  const mapPq = useMemo(() => (pqOn ? trainPq(mapCloud, dBits, 1) : undefined), [mapCloud, pqOn, dBits]);
  const mapIvf = useMemo(() => buildIvf(mapCloud, dNlist), [mapCloud, dNlist]);
  const cells = useMemo(() => voronoi(mapIvf.cents), [mapIvf]);
  const mapIvfPq = useMemo(() => (pqOn && tab === 'ivf' ? trainPq(mapCloud, dBits, 1, mapIvf.base) : undefined), [mapCloud, pqOn, tab, dBits, mapIvf]);
  const qv = useMemo(() => Float64Array.from(q), [q]);
  const nprobeEff = Math.min(nprobe, mapIvf.nlist);

  /* ---- chart data (same code, averaged over held-out queries) ---- */
  const chartCloud = useMemo(() => (dChart === 'd2' ? mapCloud : makeCloud(cfg.n, cfg.d, 7, cfg.spread)), [dChart, mapCloud, cfg]);
  const chartQueries = useMemo(() => makeQueries(CHART_QUERIES, cfg.d, 991, cfg.spread), [cfg]);
  const chartGraph = useMemo(() => (dChart === 'd2' ? mapGraph : buildHnsw(chartCloud, dM, dEfC, dHeur)), [dChart, mapGraph, chartCloud, dM, dEfC, dHeur]);
  const chartPq = useMemo(() => (pqOn && tab !== 'filter' ? (dChart === 'd2' ? mapPq : trainPq(chartCloud, dBits, cfg.sub)) : undefined), [pqOn, tab, dChart, mapPq, chartCloud, dBits, cfg]);

  const hnswFull = useMemo(() => (tab === 'filter' ? [] : hnswSweep(chartGraph, chartCloud, chartQueries, K, EF_SWEEP)), [tab, chartGraph, chartCloud, chartQueries]);
  const hnswPq = useMemo(() => (chartPq ? hnswSweep(chartGraph, chartCloud, chartQueries, K, EF_SWEEP, chartPq, rerank) : []), [chartPq, chartGraph, chartCloud, chartQueries, rerank]);
  const hnswNow = useMemo(() => (tab === 'filter' ? undefined : hnswSweep(chartGraph, chartCloud, chartQueries, K, [dEf], chartPq, rerank)[0]), [tab, chartGraph, chartCloud, chartQueries, dEf, chartPq, rerank]);

  const chartIvf = useMemo(() => (tab !== 'ivf' ? undefined : dChart === 'd2' ? mapIvf : buildIvf(chartCloud, dNlist)), [tab, dChart, mapIvf, chartCloud, dNlist]);
  const chartIvfPq = useMemo(() => (chartIvf && pqOn ? (dChart === 'd2' ? mapIvfPq ?? trainPq(chartCloud, dBits, cfg.sub, chartIvf.base) : trainPq(chartCloud, dBits, cfg.sub, chartIvf.base)) : undefined), [chartIvf, pqOn, dChart, mapIvfPq, chartCloud, dBits, cfg]);
  const ivfSeries = useMemo(() => (chartIvf ? ivfSweep(chartIvf, chartCloud, chartQueries, K, nprobeValues(chartIvf.nlist), chartIvfPq, rerank) : []), [chartIvf, chartCloud, chartQueries, chartIvfPq, rerank]);
  const ivfNow = useMemo(() => (chartIvf ? ivfSweep(chartIvf, chartCloud, chartQueries, K, [Math.min(dNprobe, chartIvf.nlist)], chartIvfPq, rerank)[0] : undefined), [chartIvf, chartCloud, chartQueries, dNprobe, chartIvfPq, rerank]);

  const filterSweeps = useMemo(() => {
    if (tab !== 'filter') return undefined;
    const qs = chartQueries.slice(0, FILTER_QUERIES);
    const run = (s: FilterStrategy, efs: number[]) => filteredSweep(chartGraph, chartCloud, qs, K, efs, dSel, s);
    return {
      post: run('post', FILTER_EFS),
      iterative: run('iterative', FILTER_EFS),
      ingraph: run('ingraph', FILTER_EFS),
      exact: run('exact', [K])[0],
      now: {
        post: run('post', [dEf])[0],
        iterative: run('iterative', [dEf])[0],
        ingraph: run('ingraph', [dEf])[0],
      },
    };
  }, [tab, chartGraph, chartCloud, chartQueries, dSel, dEf]);

  /* ---- the one query drawn on the map ---- */
  const mapVecs = pqOn && mapPq ? mapPq.recon : mapCloud.v;
  const truth = useMemo(() => exactKnn(mapCloud, qv, K), [mapCloud, qv]);
  const tr = useMemo(
    () =>
      searchHnsw(mapGraph, (id) => distTo(mapVecs, 2, id, qv), K, efSearch, {
        exact: (id) => distTo(mapCloud.v, 2, id, qv),
        rerank: pqOn && rerank,
        trace: true,
      }),
    [mapGraph, mapVecs, mapCloud, qv, efSearch, pqOn, rerank],
  );
  const steps = useMemo<Step[]>(() => {
    const out: Step[] = tr.upper.map((_, idx) => ({ kind: 'upper' as const, idx }));
    const E = tr.expansions.length;
    const chunks = Math.min(6, E);
    for (let j = 0; j < chunks; j++) out.push({ kind: 'layer0', upto: Math.ceil(((j + 1) * E) / chunks) - 1 });
    out.push({ kind: 'done' });
    return out;
  }, [tr]);
  const cur: Step = steps[Math.min(step, steps.length - 1)];
  const done = cur.kind === 'done';
  const shownLayer = cur.kind === 'upper' ? tr.upper[cur.idx].layer : cur.kind === 'layer0' ? 0 : Math.min(layerView, mapGraph.maxLevel);

  const ivfVecs = pqOn && mapIvfPq ? mapIvfPq.recon : mapCloud.v;
  const ivfRes = useMemo(() => searchIvf(mapIvf, mapCloud, ivfVecs, qv, K, nprobeEff, pqOn && rerank), [mapIvf, mapCloud, ivfVecs, qv, nprobeEff, pqOn, rerank]);

  const accept = (id: number) => mapCloud.u[id] < sel;
  const fTruth = useMemo(() => exactKnn(mapCloud, qv, K, (id) => mapCloud.u[id] < sel), [mapCloud, qv, sel]);
  const fRes = useMemo(() => filteredSearch(mapGraph, mapCloud, qv, K, efSearch, sel, strategy), [mapGraph, mapCloud, qv, efSearch, sel, strategy]);

  const edges = useMemo(() => (tab === 'hnsw' ? edgesPath(mapGraph, mapCloud.v, shownLayer) : ''), [tab, mapGraph, mapCloud, shownLayer]);

  const choosePreset = (v: string) => {
    setPreset(v);
    const p = PRESETS.find((x) => x.value === v);
    if (p?.q) setQ(p.q);
    setStep(99);
  };
  const onMapClick = (e: React.MouseEvent<SVGSVGElement>) => {
    const r = e.currentTarget.getBoundingClientRect();
    if (r.width === 0) return;
    const sx = ((e.clientX - r.left) / r.width) * MAPW;
    const sy = ((e.clientY - r.top) / r.height) * MAPW;
    const x = Math.min(1, Math.max(0, (sx - PAD) / SIDE));
    const y = Math.min(1, Math.max(0, 1 - (sy - PAD) / SIDE));
    setQ([Math.round(x * 1000) / 1000, Math.round(y * 1000) / 1000]);
    setPreset('custom');
    setStep(99);
  };

  /* ---- derived drawing sets ---- */
  const v = mapCloud.v;
  const truthIds = new Set((tab === 'filter' ? fTruth : truth).map((c) => c.id));
  const resultList = tab === 'hnsw' ? tr.res : tab === 'ivf' ? ivfRes.res : fRes.res;
  const resultIds = new Set(resultList.map((c) => c.id));
  const found = [...truthIds].filter((id) => resultIds.has(id));
  const missed = [...truthIds].filter((id) => !resultIds.has(id));
  const wrong = [...resultIds].filter((id) => !truthIds.has(id));
  const recallMap = truthIds.size ? found.length / truthIds.size : 1;

  let visitedIds: number[] = [];
  let wNow: Cand[] = [];
  if (tab === 'hnsw') {
    if (cur.kind === 'layer0') {
      const ex = tr.expansions[cur.upto];
      visitedIds = tr.visited.slice(0, ex.visited);
      wNow = ex.w;
    } else if (done) {
      visitedIds = tr.visited;
      wNow = tr.candidates;
    }
  } else if (tab === 'ivf') {
    visitedIds = ivfRes.probed.flatMap((j) => mapIvf.lists[j]);
  } else {
    visitedIds = strategy === 'exact' ? [] : fRes.visited;
  }
  const upperShown = cur.kind === 'upper' ? tr.upper.slice(0, cur.idx + 1) : tr.upper;
  const pqRecon = tab === 'hnsw' ? mapPq?.recon : tab === 'ivf' ? mapIvfPq?.recon : undefined;

  const layerNodes = useMemo(() => {
    const ids: number[] = [];
    for (let i = 0; i < mapCloud.n; i++) if (mapGraph.levels[i] >= shownLayer) ids.push(i);
    return ids;
  }, [mapGraph, mapCloud, shownLayer]);
  const otherNodes = useMemo(() => {
    const ids: number[] = [];
    for (let i = 0; i < mapCloud.n; i++) if (mapGraph.levels[i] < shownLayer) ids.push(i);
    return ids;
  }, [mapGraph, mapCloud, shownLayer]);

  const bytesFloat = cfg.d * 4;
  const pqBytes = Math.ceil((Math.floor(cfg.d / cfg.sub) * dBits) / 8);
  const linkBytes = (2 * dM + dM / Math.log(Math.max(2, dM))) * 4;

  /* ---- per-tab text ---- */
  let note: ReactNode = null;
  let title = '';
  let subtitle = '';
  if (tab === 'hnsw') {
    title = 'HNSW: descend the layers, then search layer 0 with a bounded candidate list';
    subtitle = 'Build the graph with M and efConstruction, place a query (click the map) and step through the search. The chart runs the same code over 60 held-out queries and plots recall@10 against distance computations as efSearch varies.';
    if (cur.kind === 'upper') {
      const u = tr.upper[cur.idx];
      const from = u.path[0];
      const to = u.path[u.path.length - 1];
      note = (
        <>
          <strong>
            Layer {u.layer} ({fmtNum(mapGraph.layerCounts[u.layer])} nodes): greedy search with ef = 1.
          </strong>{' '}
          {u.path.length > 1
            ? `From node #${from} it hops ${u.path.length - 1} time${u.path.length - 1 === 1 ? '' : 's'} to #${to}, where no neighbour on this layer is closer to the query`
            : `No neighbour of node #${from} on this layer is closer to the query, so the search stays put`}
          , computing {u.comps} distance{u.comps === 1 ? '' : 's'}. #{to} becomes the entry point one layer down.
        </>
      );
    } else if (cur.kind === 'layer0') {
      const ex = tr.expansions[cur.upto];
      const far = ex.w.length ? eu(ex.w[ex.w.length - 1].d) : 0;
      note = (
        <>
          <strong>
            Layer 0, candidate {cur.upto + 1} of {tr.expansions.length} expanded: W holds {ex.w.length} of ef = {Math.max(efSearch, K)}.
          </strong>{' '}
          {ex.visited} nodes have had their distance computed so far. Each expansion pops the nearest unexpanded candidate and scores its unvisited neighbours; a neighbour joins W only if W has room or it beats W’s farthest member (now {far.toFixed(3)} away). The search ends when the nearest remaining candidate is farther than that.
        </>
      );
    } else {
      note = (
        <>
          <strong>
            Returned the {K} nearest of {tr.candidates.length} candidates after {fmtNum(tr.comps)} distance computations — an exact scan needs {fmtNum(MAP_N)}. Recall@10 for this query: {fmtNum(recallMap * 100)}%.
          </strong>{' '}
          {pqOn
            ? `The graph ranked nodes by distance to their PQ reconstructions (${2 * bits} bits per point)${rerank ? ', then re-ranked the candidate list with the full vectors' : ', so near-ties inside one code cell are decided by the code, not the vector'}.`
            : !heuristic
              ? 'Nearest-M neighbour selection: with small M, links stay inside clusters and the greedy search can get stuck on the wrong side of a gap.'
              : efC < 2 * M
                ? `pgvector would reject this build: ef_construction must be at least 2 × m (${2 * M}).`
                : `On the chart, efSearch ${efSearch} averages ${hnswNow ? fmtNum(hnswNow.recall * 100, 1) : '…'}% recall for ${hnswNow ? fmtNum(hnswNow.comps) : '…'} distance computations on ${cfg.label}.`}
        </>
      );
    }
  } else if (tab === 'ivf') {
    title = 'IVF: k-means cells, and only nprobe lists are scanned';
    subtitle = 'Training fits nlist centroids; every vector is stored in the list of its nearest centroid. A query ranks the centroids and scans the nprobe nearest lists. True neighbours that live in an unprobed cell are simply missed.';
    const sizes = mapIvf.lists.map((l) => l.length);
    note = (
      <>
        <strong>
          nprobe = {nprobeEff} of nlist = {mapIvf.nlist}: scanned {fmtNum(ivfRes.scanned)} vectors ({fmtNum((ivfRes.scanned / MAP_N) * 100)}% of the data) plus {mapIvf.nlist} centroid distances. Recall@10: {fmtNum(recallMap * 100)}%.
        </strong>{' '}
        {missed.length
          ? `${missed.length} true neighbour${missed.length === 1 ? ' sits' : 's sit'} in a cell that was not probed. `
          : 'Every true neighbour sits in a probed cell. '}
        Lists range from {Math.min(...sizes)} to {Math.max(...sizes)} vectors, so the scanned fraction is not simply nprobe/nlist.
        {pqOn ? ` Each vector is stored as a code of ${2 * bits} bits for its residual from the centroid (IVFADC)${rerank ? `; the top ${K * 4} by code distance are re-ranked with full vectors` : ''}.` : ''}
      </>
    );
  } else {
    title = 'Filtered search: where the WHERE clause runs changes what comes back';
    subtitle = 'A fraction of rows match the filter (violet). Post-filtering drops non-matching rows after a normal graph search, an iterative scan keeps resuming the search, in-graph filtering lets only matching nodes into the candidate list, and a B-tree prefilter scores every matching row exactly.';
    const lucene = fRes.matching <= K ? 'exact' : strategy === 'ingraph' && fRes.visited.length > fRes.matching ? 'fallback' : 'graph';
    note = (
      <>
        <strong>
          {STRATEGY_LABEL[strategy]}: {fRes.res.length} of {K} rows returned, recall@10 {fmtNum(recallMap * 100)}%, {fmtNum(fRes.comps)} distance computations ({fmtNum(fRes.matching)} of {MAP_N} rows match).
        </strong>{' '}
        {strategy === 'post'
          ? `The graph search found ${Math.max(efSearch, K)} candidates without looking at the filter; only the matching ones survive, so about ef × selectivity rows come back.`
          : strategy === 'iterative'
            ? `pgvector-style iterative scan: ${fRes.batches} batch${fRes.batches === 1 ? '' : 'es'} of ef = ${Math.max(efSearch, K)}, each resuming from the candidates the previous batch discarded.`
            : strategy === 'ingraph'
              ? `Non-matching nodes are still traversed as routing hops but never enter W. It visited ${fRes.visited.length} nodes${lucene === 'fallback' ? `, more than the ${fRes.matching} matching rows — Lucene would abandon the graph here and score the matches exactly` : lucene === 'exact' ? ` — Lucene would not have entered the graph, since no more than k = ${K} rows match` : ''}.`
              : 'Every matching row is scored: exact, and cheap only when few rows match.'}
      </>
    );
  }

  /* ---- chart ---- */
  let chart: ReactNode = null;
  if (tab === 'hnsw') {
    const series: Series[] = [
      { key: 'full', label: 'HNSW', color: 'var(--viz-1)', points: hnswFull, dashed: !!chartPq, current: chartPq ? undefined : hnswNow, currentLabel: `efSearch ${efSearch}` },
    ];
    if (chartPq) series.push({ key: 'pq', label: 'HNSW + PQ', color: 'var(--viz-1)', points: hnswPq, current: hnswNow, currentLabel: `efSearch ${efSearch}` });
    chart = <RecallSweepChart series={series} n={cfg.n} title={`Recall@10 vs work, efSearch 10→240, on ${cfg.label}`} />;
  } else if (tab === 'ivf') {
    chart = (
      <RecallSweepChart
        series={[
          { key: 'hnsw', label: 'HNSW', color: 'var(--viz-1)', points: hnswPq.length ? hnswPq : hnswFull },
          { key: 'ivf', label: 'IVF', color: 'var(--viz-2)', points: ivfSeries, current: ivfNow, currentLabel: `nprobe ${Math.min(nprobe, chartIvf?.nlist ?? nprobe)}` },
        ]}
        n={cfg.n}
        title={`IVF nprobe 1→nlist vs HNSW efSearch 10→240, on ${cfg.label}`}
      />
    );
  } else if (filterSweeps) {
    const fs = filterSweeps;
    chart = (
      <RecallSweepChart
        series={[
          { key: 'post', label: 'Post-filter', color: 'var(--viz-5)', tag: 'post-filter', tagDy: 4, points: fs.post, current: strategy === 'post' ? fs.now.post : undefined, currentLabel: `efSearch ${efSearch}` },
          { key: 'iter', label: 'Iterative scan', color: 'var(--viz-2)', tag: 'iterative', tagDy: -8, points: fs.iterative, current: strategy === 'iterative' ? fs.now.iterative : undefined, currentLabel: `efSearch ${efSearch}` },
          { key: 'graph', label: 'In-graph filter', color: 'var(--viz-1)', tag: 'in-graph', tagDy: 16, points: fs.ingraph, current: strategy === 'ingraph' ? fs.now.ingraph : undefined, currentLabel: `efSearch ${efSearch}` },
        ]}
        points={[{ label: 'exact over matches', p: fs.exact }]}
        n={cfg.n}
        title={`${selPct}% of rows match — efSearch 10→240, on ${cfg.label}`}
      />
    );
  }

  const legend =
    tab === 'hnsw' ? (
      <Legend
        items={[
          { label: `Node on layer ${shownLayer}`, color: 'var(--viz-1)', shape: 'dot' },
          { label: 'Entry point (diamond)', color: 'var(--viz-ink)' },
          { label: 'Greedy hops (upper layers)', color: 'var(--viz-ink)', shape: 'line' },
          { label: 'Distance computed (layer 0)', color: 'var(--viz-4)', shape: 'dot' },
          { label: 'Candidate list W', color: 'var(--viz-7)', shape: 'dot' },
          { label: 'True neighbour returned', color: 'var(--viz-good)', shape: 'dot' },
          { label: 'True neighbour missed', color: 'var(--viz-critical)', shape: 'dot' },
          ...(pqOn ? [{ label: 'PQ reconstruction', color: 'var(--viz-5)' }] : []),
          { label: chartPq ? 'Chart: HNSW, full vectors (dashed) / PQ codes (solid)' : 'Chart: HNSW', color: 'var(--viz-1)', shape: 'line' as const },
        ]}
      />
    ) : tab === 'ivf' ? (
      <Legend
        items={[
          { label: 'Centroid and cell (probed cells shaded)', color: 'var(--viz-2)' },
          { label: 'Scanned vector (distance computed)', color: 'var(--viz-4)', shape: 'dot' },
          { label: 'True neighbour returned', color: 'var(--viz-good)', shape: 'dot' },
          { label: 'True neighbour missed', color: 'var(--viz-critical)', shape: 'dot' },
          ...(pqOn ? [{ label: 'PQ reconstruction (centroid + coded residual)', color: 'var(--viz-5)' }] : []),
          { label: 'Chart: IVF', color: 'var(--viz-2)', shape: 'line' },
          { label: 'Chart: HNSW for comparison', color: 'var(--viz-1)', shape: 'line' },
        ]}
      />
    ) : (
      <Legend
        items={[
          { label: 'Row matches the filter', color: 'var(--viz-7)', shape: 'dot' },
          { label: 'Distance computed', color: 'var(--viz-4)', shape: 'dot' },
          { label: 'True filtered neighbour returned', color: 'var(--viz-good)', shape: 'dot' },
          { label: 'True filtered neighbour missed', color: 'var(--viz-critical)', shape: 'dot' },
          { label: 'Chart: post-filter', color: 'var(--viz-5)', shape: 'line' },
          { label: 'Chart: iterative scan', color: 'var(--viz-2)', shape: 'line' },
          { label: 'Chart: in-graph filter', color: 'var(--viz-1)', shape: 'line' },
        ]}
      />
    );

  const stats =
    tab === 'hnsw' ? (
      <Stats
        items={[
          { label: 'This query: distance computations', value: fmtNum(tr.comps), hint: 'Entry point, every greedy hop on upper layers, every newly visited node on layer 0, plus the re-rank if enabled.' },
          { label: 'This query: recall@10', value: `${fmtNum(recallMap * 100)}%` },
          { label: 'Graph layers (nodes)', value: mapGraph.layerCounts.slice().reverse().join(' / '), hint: 'Top layer first. Each node’s top layer is floor(−ln U × mL), mL = 1/ln M.' },
          { label: 'Build: distance computations', value: fmtNum(mapGraph.buildComps), hint: 'Includes the heuristic’s pairwise checks. Grows with efConstruction.' },
          { label: `Chart @ efSearch ${efSearch}`, value: hnswNow ? `${fmtNum(hnswNow.recall * 100, 1)}% / ${fmtNum(hnswNow.comps)}` : '…', hint: `Mean recall@10 / distance computations over ${CHART_QUERIES} queries on ${cfg.label}.` },
          { label: `Bytes per vector (d = ${cfg.d})`, value: pqOn ? `${pqBytes} code vs ${bytesFloat} float` : `${bytesFloat} + ~${fmtNum(linkBytes)} links`, hint: 'float32 is 4 bytes per dimension; links follow the HNSW paper’s (Mmax0 + mL·Mmax) × 4 bytes.' },
        ]}
      />
    ) : tab === 'ivf' ? (
      <Stats
        items={[
          { label: 'This query: distance computations', value: fmtNum(ivfRes.comps), hint: 'nlist centroid distances plus every vector in the probed lists.' },
          { label: 'This query: recall@10', value: `${fmtNum(recallMap * 100)}%` },
          { label: 'Vectors scanned', value: `${fmtNum(ivfRes.scanned)} of ${MAP_N}` },
          { label: `Chart @ nprobe ${ivfNow ? Math.min(nprobe, chartIvf?.nlist ?? nprobe) : ''}`, value: ivfNow ? `${fmtNum(ivfNow.recall * 100, 1)}% / ${fmtNum(ivfNow.comps)}` : '…', hint: `Mean recall@10 / distance computations over ${CHART_QUERIES} queries on ${cfg.label}.` },
          { label: `Bytes per vector (d = ${cfg.d})`, value: pqOn ? `${pqBytes} + 8 id` : `${bytesFloat} + 8 id`, hint: 'FAISS stores an 8-byte id per entry in an inverted list.' },
        ]}
      />
    ) : (
      <Stats
        items={[
          { label: 'Rows matching', value: `${fmtNum(fRes.matching)} of ${MAP_N}` },
          { label: 'Rows returned', value: `${fRes.res.length} of ${K}` },
          { label: 'Recall@10 (filtered truth)', value: `${fmtNum(recallMap * 100)}%` },
          { label: 'Distance computations', value: fmtNum(fRes.comps) },
          { label: 'Batches', value: strategy === 'iterative' ? fmtNum(fRes.batches) : '—', hint: 'Iterative scans resume the layer-0 search ef_search candidates at a time.' },
        ]}
      />
    );

  const table =
    tab === 'hnsw' ? (
      <table className="viz-table">
        <thead>
          <tr>
            <th>efSearch</th>
            <th>Recall@10, full vectors</th>
            <th>Distance computations</th>
            {chartPq ? <th>Recall@10, PQ{rerank ? ' + re-rank' : ''}</th> : null}
            {chartPq ? <th>Distance computations</th> : null}
          </tr>
        </thead>
        <tbody>
          {hnswFull.map((p, i) => (
            <tr key={p.param}>
              <td>{p.param}</td>
              <td>{fmtNum(p.recall * 100, 1)}%</td>
              <td>{fmtNum(p.comps)}</td>
              {chartPq ? <td>{fmtNum((hnswPq[i]?.recall ?? 0) * 100, 1)}%</td> : null}
              {chartPq ? <td>{fmtNum(hnswPq[i]?.comps ?? 0)}</td> : null}
            </tr>
          ))}
        </tbody>
        <caption style={{ captionSide: 'bottom', textAlign: 'left', paddingTop: 6 }}>
          At d = 768 a float32 vector is {fmtBytes(3072)}, an int8 scalar-quantized one {fmtBytes(768)}, a 1-bit binary one {fmtBytes(96)}, and PQ with 96 subspaces × 8 bits {fmtBytes(96)}; HNSW links at M = {M} add about {fmtNum((2 * M + M / Math.log(Math.max(2, M))) * 4)} bytes per vector.
        </caption>
      </table>
    ) : tab === 'ivf' ? (
      <table className="viz-table">
        <thead>
          <tr>
            <th>nprobe</th>
            <th>Recall@10</th>
            <th>Distance computations</th>
          </tr>
        </thead>
        <tbody>
          {ivfSeries.map((p) => (
            <tr key={p.param}>
              <td>{p.param}</td>
              <td>{fmtNum(p.recall * 100, 1)}%</td>
              <td>{fmtNum(p.comps)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    ) : (
      <table className="viz-table">
        <thead>
          <tr>
            <th>Strategy @ efSearch {efSearch}</th>
            <th>Recall@10</th>
            <th>Rows returned</th>
            <th>Distance computations</th>
          </tr>
        </thead>
        <tbody>
          {filterSweeps
            ? (['post', 'iterative', 'ingraph'] as const).map((s) => (
                <tr key={s}>
                  <td>{STRATEGY_LABEL[s]}</td>
                  <td>{fmtNum(filterSweeps.now[s].recall * 100, 1)}%</td>
                  <td>{fmtNum(filterSweeps.now[s].returned ?? 0, 1)}</td>
                  <td>{fmtNum(filterSweeps.now[s].comps)}</td>
                </tr>
              ))
            : null}
          {filterSweeps ? (
            <tr>
              <td>{STRATEGY_LABEL.exact}</td>
              <td>{fmtNum(filterSweeps.exact.recall * 100, 1)}%</td>
              <td>{fmtNum(filterSweeps.exact.returned ?? 0, 1)}</td>
              <td>{fmtNum(filterSweeps.exact.comps)}</td>
            </tr>
          ) : null}
        </tbody>
      </table>
    );

  /* ---- map svg ---- */
  const qx = px(q[0]);
  const qy = py(q[1]);
  const map = (
    <svg
      viewBox={`0 0 ${MAPW} ${MAPW}`}
      width={MAPW}
      height={MAPW}
      role="img"
      aria-label={
        tab === 'hnsw'
          ? `HNSW graph, layer ${shownLayer}, query search ${done ? 'complete' : 'in progress'}`
          : tab === 'ivf'
            ? `IVF with ${mapIvf.nlist} cells, ${nprobeEff} probed`
            : `Filtered search, ${selPct}% of rows match`
      }
      onClick={onMapClick}
      style={{ cursor: 'crosshair', flex: '0 1 auto' }}
    >
      <rect x={PAD} y={PAD} width={SIDE} height={SIDE} fill="var(--viz-plane)" stroke="var(--viz-border)" />
      {tab === 'ivf'
        ? cells.map((poly, j) => {
            const probedRank = ivfRes.probed.indexOf(j);
            if (poly.length < 3) return null;
            return (
              <path
                key={j}
                d={`M${poly.map(([x, y]) => `${px(x).toFixed(1)} ${py(y).toFixed(1)}`).join('L')}Z`}
                fill={probedRank >= 0 ? 'var(--viz-2)' : 'none'}
                fillOpacity={probedRank >= 0 ? 0.13 : 0}
                stroke="var(--viz-2)"
                strokeOpacity={probedRank >= 0 ? 0.9 : 0.45}
                strokeWidth={probedRank >= 0 ? 1.6 : 0.8}
              />
            );
          })
        : null}
      {tab === 'hnsw' ? <path d={edges} stroke="var(--viz-1)" strokeOpacity={shownLayer === 0 ? 0.22 : 0.5} strokeWidth={shownLayer === 0 ? 0.7 : 1.2} fill="none" /> : null}

      {tab === 'hnsw' ? (
        <>
          <path d={dots(v, otherNodes, 1.6)} fill="var(--viz-axis)" />
          <path d={dots(v, layerNodes, shownLayer === 0 ? 1.9 : 3.4)} fill="var(--viz-1)" fillOpacity={shownLayer === 0 ? 0.55 : 1} />
        </>
      ) : tab === 'ivf' ? (
        <path d={dots(v, Array.from({ length: mapCloud.n }, (_, i) => i), 1.8)} fill="var(--viz-axis)" />
      ) : (
        <>
          <path d={dots(v, Array.from({ length: mapCloud.n }, (_, i) => i).filter((i) => !accept(i)), 1.6)} fill="var(--viz-axis)" />
        </>
      )}

      {pqRecon ? (
        <>
          <path
            d={(() => {
              const seen = new Set<string>();
              let s = '';
              for (let i = 0; i < mapCloud.n; i++) {
                const key = `${pqRecon[2 * i].toFixed(4)},${pqRecon[2 * i + 1].toFixed(4)}`;
                if (seen.has(key)) continue;
                seen.add(key);
                s += `M${(px(pqRecon[2 * i]) - 2).toFixed(1)} ${(py(pqRecon[2 * i + 1]) - 2).toFixed(1)}h4v4h-4z`;
              }
              return s;
            })()}
            fill="var(--viz-5)"
            fillOpacity={0.55}
          />
          <path
            d={resultList.map((c) => `M${px(v[2 * c.id]).toFixed(1)} ${py(v[2 * c.id + 1]).toFixed(1)}L${px(pqRecon[2 * c.id]).toFixed(1)} ${py(pqRecon[2 * c.id + 1]).toFixed(1)}`).join('')}
            stroke="var(--viz-5)"
            strokeWidth={1.6}
          />
        </>
      ) : null}

      {visitedIds.length ? <path d={dots(v, visitedIds, 2.6)} fill="var(--viz-4)" /> : null}
      {tab === 'filter' ? <path d={dots(v, Array.from({ length: mapCloud.n }, (_, i) => i).filter(accept), 3.2)} fill="var(--viz-7)" stroke="var(--viz-surface)" strokeWidth={0.8} /> : null}

      {tab === 'hnsw'
        ? upperShown.map((u, i) => (
            <g key={u.layer}>
              <polyline
                points={u.path.map((id) => `${px(v[2 * id]).toFixed(1)},${py(v[2 * id + 1]).toFixed(1)}`).join(' ')}
                fill="none"
                stroke="var(--viz-ink)"
                strokeWidth={cur.kind === 'upper' && i === upperShown.length - 1 ? 2.4 : 1.4}
                strokeOpacity={cur.kind === 'upper' && i === upperShown.length - 1 ? 1 : 0.6}
              />
              {u.path.map((id, h) => (
                <circle key={h} cx={px(v[2 * id])} cy={py(v[2 * id + 1])} r={3} fill="var(--viz-surface)" stroke="var(--viz-ink)" strokeWidth={1.2} />
              ))}
            </g>
          ))
        : null}
      {tab === 'hnsw' && mapGraph.entry >= 0 ? (
        <path d={`M${px(v[2 * mapGraph.entry])} ${py(v[2 * mapGraph.entry + 1]) - 7}l7 7l-7 7l-7 -7z`} fill="none" stroke="var(--viz-ink)" strokeWidth={1.6} />
      ) : null}
      {tab === 'hnsw' && wNow.length ? <path d={dots(v, wNow.map((c) => c.id), 4.6)} fill="none" stroke="var(--viz-7)" strokeWidth={1.6} /> : null}

      {tab === 'ivf'
        ? Array.from({ length: mapIvf.nlist }, (_, j) => {
            const x = px(mapIvf.cents[2 * j]);
            const y = py(mapIvf.cents[2 * j + 1]);
            const rank = ivfRes.probed.indexOf(j);
            return (
              <g key={j}>
                <path d={`M${x - 5} ${y}h10M${x} ${y - 5}v10`} stroke="var(--viz-2)" strokeWidth={2.2} />
                {rank >= 0 ? (
                  <text x={x + 6} y={y - 5} fontSize={11} fill="var(--viz-ink)">
                    {rank + 1}
                  </text>
                ) : null}
              </g>
            );
          })
        : null}

      {done || tab !== 'hnsw' ? (
        <>
          <path d={dots(v, found, 7)} fill="none" stroke="var(--viz-good)" strokeWidth={2} />
          <path d={dots(v, missed, 7)} fill="none" stroke="var(--viz-critical)" strokeWidth={2} strokeDasharray="3 2" />
          <path d={dots(v, wrong, 5.5)} fill="none" stroke="var(--viz-ink-2)" strokeWidth={1.2} />
        </>
      ) : null}

      <circle cx={qx} cy={qy} r={9} fill="none" stroke="var(--viz-ink)" strokeWidth={1.4} />
      <path d={`M${qx - 6} ${qy - 6}l12 12M${qx - 6} ${qy + 6}l12 -12`} stroke="var(--viz-ink)" strokeWidth={2} />
      <text x={Math.min(qx + 12, MAPW - 40)} y={Math.max(qy - 10, 24)} fill="var(--viz-ink)" fontSize={12}>
        query
      </text>
    </svg>
  );

  /* ---- side panel ---- */
  const side =
    tab === 'hnsw' ? (
      <div style={{ flex: '1 1 210px', minWidth: 200, fontSize: '0.78rem', color: 'var(--viz-ink)', display: 'flex', flexWrap: 'wrap', gap: '0 1.2rem', alignItems: 'flex-start' }}>
        <div>
        <table className="viz-table">
          <thead>
            <tr>
              <th>Layer</th>
              <th>Nodes</th>
              <th>Max links</th>
            </tr>
          </thead>
          <tbody>
            {mapGraph.layerCounts
              .map((c, l) => ({ c, l }))
              .reverse()
              .map(({ c, l }) => (
                <tr key={l} aria-current={l === shownLayer ? 'true' : undefined}>
                  <td>{l === shownLayer ? <strong>▶ L{l}</strong> : `L${l}`}</td>
                  <td>{fmtNum(c)}</td>
                  <td>{l === 0 ? 2 * dM : dM}</td>
                </tr>
              ))}
          </tbody>
        </table>
        </div>
        <div>
        <p style={{ margin: '0.6rem 0 0.25rem', color: 'var(--viz-ink-2)' }}>
          {cur.kind === 'upper' ? 'Greedy path on this layer (ef = 1)' : `Candidate list W (ef = ${Math.max(efSearch, K)})${pqOn ? ', PQ distances' : ''}`}
        </p>
        {cur.kind === 'upper' ? (
          <ol style={{ margin: 0, paddingLeft: '1.2rem' }}>
            {tr.upper[cur.idx].path.map((id, i) => (
              <li key={i}>
                #{id} — {eu(distTo(mapVecs, 2, id, qv)).toFixed(3)}
              </li>
            ))}
          </ol>
        ) : (
          <table className="viz-table">
            <thead>
              <tr>
                <th>#</th>
                <th>Node</th>
                <th>Distance</th>
                <th>Top-10?</th>
              </tr>
            </thead>
            <tbody>
              {wNow.slice(0, 10).map((c, i) => (
                <tr key={c.id}>
                  <td>{i + 1}</td>
                  <td>#{c.id}</td>
                  <td>{eu(c.d).toFixed(3)}</td>
                  <td>{truthIds.has(c.id) ? 'yes' : 'no'}</td>
                </tr>
              ))}
            </tbody>
            {wNow.length > 10 ? (
              <caption style={{ captionSide: 'bottom', textAlign: 'left', paddingTop: 4 }}>… and {wNow.length - 10} more, out to {eu(wNow[wNow.length - 1].d).toFixed(3)}</caption>
            ) : null}
          </table>
        )}
        </div>
      </div>
    ) : tab === 'ivf' ? (
      <div style={{ flex: '1 1 210px', minWidth: 200, fontSize: '0.78rem', color: 'var(--viz-ink)' }}>
        <table className="viz-table">
          <thead>
            <tr>
              <th>Probe</th>
              <th>List</th>
              <th>Vectors</th>
              <th>Top-10 inside</th>
            </tr>
          </thead>
          <tbody>
            {ivfRes.probed.map((j, r) => (
              <tr key={j}>
                <td>{r + 1}</td>
                <td>#{j}</td>
                <td>{mapIvf.lists[j].length}</td>
                <td>{mapIvf.lists[j].filter((id) => truthIds.has(id)).length}</td>
              </tr>
            ))}
          </tbody>
          <caption style={{ captionSide: 'bottom', textAlign: 'left', paddingTop: 4 }}>
            {missed.length} of the true top-10 {missed.length === 1 ? 'lives' : 'live'} in unprobed lists.
          </caption>
        </table>
      </div>
    ) : (
      <div style={{ flex: '1 1 210px', minWidth: 200, fontSize: '0.78rem', color: 'var(--viz-ink)' }}>
        <table className="viz-table">
          <thead>
            <tr>
              <th>Strategy</th>
              <th>Rows</th>
              <th>Recall</th>
              <th>Work</th>
            </tr>
          </thead>
          <tbody>
            {(['post', 'iterative', 'ingraph', 'exact'] as const).map((s) => {
              const r = s === strategy ? fRes : filteredSearch(mapGraph, mapCloud, qv, K, efSearch, sel, s);
              return (
                <tr key={s} aria-current={s === strategy ? 'true' : undefined}>
                  <td>{s === strategy ? <strong>{STRATEGY_LABEL[s]}</strong> : STRATEGY_LABEL[s]}</td>
                  <td>{r.res.length}</td>
                  <td>{fmtNum(recallOf(r.res, fTruth) * 100)}%</td>
                  <td>{fmtNum(r.comps)}</td>
                </tr>
              );
            })}
          </tbody>
          <caption style={{ captionSide: 'bottom', textAlign: 'left', paddingTop: 4 }}>This query on the map; work = distance computations.</caption>
        </table>
      </div>
    );

  return (
    <VizPanel
      title={title}
      subtitle={subtitle}
      controls={
        <>
          <Segmented
            label="Index"
            value={tab}
            onChange={(t) => {
              setTab(t);
              setStep(99);
            }}
            options={[
              { value: 'hnsw', label: 'HNSW graph' },
              { value: 'ivf', label: 'IVF cells' },
              { value: 'filter', label: 'Filtered search' },
            ]}
          />
          <Segmented
            label="Chart data"
            value={chartKey}
            onChange={setChartKey}
            options={[
              { value: 'd16', label: '16-D vectors', title: 'Same code, 2,000 clustered 16-dimensional vectors' },
              { value: 'd2', label: '2-D map', title: 'The 500 points drawn on the map' },
            ]}
          />
        </>
      }
      legend={legend}
      stats={stats}
      note={<Note>{note}</Note>}
      table={table}
    >
      <div className="viz-controls">
        {tab !== 'ivf' ? (
          <>
            <Slider label="M" min={2} max={32} value={M} onChange={(x) => { setM(x); setStep(99); }} />
            {tab === 'hnsw' ? <Slider label="efConstruction" min={8} max={200} step={4} value={efC} onChange={(x) => { setEfC(x); setStep(99); }} /> : null}
            <Slider label="efSearch" min={10} max={240} step={2} value={efSearch} onChange={(x) => { setEfSearch(x); setStep(99); }} />
          </>
        ) : (
          <>
            <Slider label="nlist" min={4} max={64} value={nlist} onChange={(x) => { setNlist(x); if (nprobe > x) setNprobe(x); }} />
            <Slider label="nprobe" min={1} max={nlist} value={Math.min(nprobe, nlist)} onChange={setNprobe} />
          </>
        )}
        {tab === 'hnsw' ? <Check label="Neighbour-selection heuristic" checked={heuristic} onChange={(b) => { setHeuristic(b); setStep(99); }} /> : null}
        {tab === 'filter' ? (
          <>
            <Slider label="Rows matching the filter" min={1} max={100} value={selPct} onChange={setSelPct} format={(x) => `${x}%`} />
            <Segmented
              label="Strategy"
              value={strategy}
              onChange={setStrategy}
              options={[
                { value: 'post', label: 'Post-filter' },
                { value: 'iterative', label: 'Iterative scan' },
                { value: 'ingraph', label: 'In-graph' },
                { value: 'exact', label: 'Exact' },
              ]}
            />
          </>
        ) : (
          <>
            <Check label="PQ codes" checked={pqOn} onChange={(b) => { setPqOn(b); setStep(99); }} />
            {pqOn ? <Slider label="Bits per subspace" min={2} max={8} value={bits} onChange={(x) => { setBits(x); setStep(99); }} /> : null}
            {pqOn ? <Check label="Re-rank with full vectors" checked={rerank} onChange={(b) => { setRerank(b); setStep(99); }} /> : null}
          </>
        )}
        <Choice label="Query" value={preset} onChange={choosePreset} options={PRESETS.map((p) => ({ value: p.value, label: p.label }))} />
      </div>
      {tab === 'hnsw' ? (
        <div className="viz-controls">
          <Button onClick={() => setStep(0)}>Search step by step</Button>
          <Button primary onClick={() => setStep((s) => Math.min(steps.length - 1, s + 1))} disabled={done}>
            Next step
          </Button>
          <Button onClick={() => setStep(99)} disabled={done}>
            Show result
          </Button>
          {done ? (
            <Segmented
              label="Show layer"
              value={String(Math.min(layerView, mapGraph.maxLevel))}
              onChange={(x) => setLayerView(Number(x))}
              options={mapGraph.layerCounts.map((_, l) => ({ value: String(l), label: `L${l}` })).reverse()}
            />
          ) : null}
        </div>
      ) : null}
      <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.9rem', alignItems: 'flex-start' }}>
        {map}
        {side}
      </div>
      {chart}
    </VizPanel>
  );
}
