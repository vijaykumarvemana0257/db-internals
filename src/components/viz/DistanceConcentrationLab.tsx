/**
 * Distance concentration, intrinsic dimension and what they do to indexes.
 *
 * Model (everything here is computed, nothing is a canned number):
 *  - Data. `uniform`: n points and Q queries i.i.d. uniform in [0,1]^D, so intrinsic dimension = D.
 *    `manifold`: a latent z uniform in [0,1]^m is drawn as a curved sheet — each latent coordinate becomes
 *    (cos πz, sin πz), 2m coordinates in all — and those 2m coordinates are rotated into D dimensions by a fixed
 *    random matrix with orthonormal columns. Rotation preserves distances, so the manifold's distance distribution
 *    does not depend on D (each D draws a fresh sample of the same sheet). It needs D >= 2m.
 *  - Ground truth: brute force, squared Euclidean distance.
 *  - kd-tree: split the coordinate with the largest spread at the median, leaves of <= LEAF points. Search is
 *    best-bin-first with the exact box lower bound (Arya & Mount incremental distance), so with an unlimited budget
 *    it is an exact 10-NN search and "leaves visited" is the real pruning result.
 *  - Hyperplane tree: Annoy's split (two sampled points refined by a short weighted 2-means, hyperplane equidistant
 *    between the two centres). Lower bound for a far branch = the largest hyperplane margin crossed on the way.
 *  - LSH: the E2LSH p-stable family h(v) = floor((a·v + b) / w), a ~ N(0, I), b ~ U[0, w]; k hashes concatenated per
 *    table, L tables. w = 4 × (mean distance to the 10th neighbour), the E2LSH manual's w = 4 at R = 1.
 *    Under a budget, candidates are checked in order of how many tables they collided in.
 *  - HNSW: Malkov & Yashunin with simple (closest-M) neighbour selection, M = 8, Mmax0 = 16, efConstruction = 32,
 *    mL = 1/ln M. Under a budget the layer-0 beam search stops when the budget is spent.
 *  - Budget: every method may compute at most B% of n distances to data points per query; recall@10 = true top-10
 *    found / 10. LSH's k × L projections and the hyperplane tree's margin tests are not charged, which flatters both.
 *    A random B% sample finds B% of them on average — the "no index" line.
 *  - LID: the MLE used by Aumüller & Ceccarello, -((1/k) Σ ln(r_i / r_k))^-1 over the 20 nearest neighbours,
 *    median over queries.
 * Sizes are small (n = 1,500) so a browser can build every index at every dimension; real crossover dimensions
 * grow with log n.
 */
import { useDeferredValue, useEffect, useMemo, useRef, useState } from 'react';
import { VizPanel, Segmented, Slider, Legend, Stats, Note, makeRng, fmtNum, useTip, useSize, TooltipHost } from './Viz';

/* ================================================================= model */

export const N_POINTS = 1500;
export const N_QUERIES = 40;
export const K = 10;
export const LEAF = 12;
export const DIMS = [2, 4, 8, 16, 32, 64, 128, 256] as const;
export type DataKind = 'uniform' | 'manifold';

type Rng = () => number;

function gauss(rng: Rng) {
  let u = 0;
  while (u <= 1e-9) u = rng();
  const v = rng();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

export type Dataset = {
  kind: DataKind;
  D: number;
  m: number;
  n: number;
  q: number;
  X: Float64Array; // n × D
  Qs: Float64Array; // q × D
};

/** Random D × f matrix with orthonormal columns (Gram–Schmidt on Gaussian columns). */
function orthonormalColumns(D: number, f: number, rng: Rng) {
  const cols: Float64Array[] = [];
  for (let c = 0; c < f; c++) {
    const v = new Float64Array(D);
    for (let i = 0; i < D; i++) v[i] = gauss(rng);
    for (const u of cols) {
      let dot = 0;
      for (let i = 0; i < D; i++) dot += v[i] * u[i];
      for (let i = 0; i < D; i++) v[i] -= dot * u[i];
    }
    let norm = 0;
    for (let i = 0; i < D; i++) norm += v[i] * v[i];
    norm = Math.sqrt(norm) || 1;
    for (let i = 0; i < D; i++) v[i] /= norm;
    cols.push(v);
  }
  return cols;
}

export function makeDataset(kind: DataKind, D: number, m: number, seed = 7, n = N_POINTS, q = N_QUERIES): Dataset {
  const rng = makeRng(seed * 7919 + D * 131 + (kind === 'manifold' ? m * 17 + 1 : 0));
  const X = new Float64Array(n * D);
  const Qs = new Float64Array(q * D);
  if (kind === 'uniform') {
    for (let i = 0; i < X.length; i++) X[i] = rng();
    for (let i = 0; i < Qs.length; i++) Qs[i] = rng();
  } else {
    const f = 2 * m;
    const R = orthonormalColumns(D, f, rng);
    const feat = new Float64Array(f);
    const fill = (out: Float64Array, row: number) => {
      for (let j = 0; j < m; j++) {
        const z = rng();
        feat[2 * j] = Math.cos(Math.PI * z) / Math.PI;
        feat[2 * j + 1] = Math.sin(Math.PI * z) / Math.PI;
      }
      const base = row * D;
      for (let i = 0; i < D; i++) out[base + i] = 0;
      for (let c = 0; c < f; c++) {
        const col = R[c];
        const s = feat[c];
        for (let i = 0; i < D; i++) out[base + i] += s * col[i];
      }
    };
    for (let r = 0; r < n; r++) fill(X, r);
    for (let r = 0; r < q; r++) fill(Qs, r);
  }
  return { kind, D, m, n, q, X, Qs };
}

function sqDist(A: Float64Array, ai: number, B: Float64Array, bi: number, D: number) {
  let s = 0;
  const a = ai * D;
  const b = bi * D;
  for (let i = 0; i < D; i++) {
    const t = A[a + i] - B[b + i];
    s += t * t;
  }
  return s;
}

/* --------------------------------------------------------- ground truth */

export type Truth = {
  /** per query: indices of the 20 nearest points, nearest first */
  nn: Int32Array[];
  /** per query: distances (not squared) of those 20 */
  nnDist: Float64Array[];
  /** per query: mean distance to all points, max distance */
  mean: Float64Array;
  max: Float64Array;
  /** pooled histogram of distance / (that query's mean distance), bins over [0, HIST_MAX) */
  hist: Float64Array;
};

export const HIST_BINS = 48;
export const HIST_MAX = 2;

export function groundTruth(ds: Dataset): Truth {
  const { n, q, D, X, Qs } = ds;
  const nn: Int32Array[] = [];
  const nnDist: Float64Array[] = [];
  const mean = new Float64Array(q);
  const max = new Float64Array(q);
  const hist = new Float64Array(HIST_BINS);
  const d = new Float64Array(n);
  const idx = new Int32Array(n);
  for (let j = 0; j < q; j++) {
    let sum = 0;
    let mx = 0;
    for (let i = 0; i < n; i++) {
      const v = Math.sqrt(sqDist(Qs, j, X, i, D));
      d[i] = v;
      idx[i] = i;
      sum += v;
      if (v > mx) mx = v;
    }
    mean[j] = sum / n;
    max[j] = mx;
    for (let i = 0; i < n; i++) {
      const b = Math.floor((d[i] / mean[j] / HIST_MAX) * HIST_BINS);
      if (b >= 0 && b < HIST_BINS) hist[b] += 1;
    }
    const sorted = Array.from(idx).sort((a, b) => d[a] - d[b]);
    const top = new Int32Array(20);
    const topD = new Float64Array(20);
    for (let t = 0; t < 20; t++) {
      top[t] = sorted[t];
      topD[t] = d[sorted[t]];
    }
    nn.push(top);
    nnDist.push(topD);
  }
  return { nn, nnDist, mean, max, hist };
}

function median(a: number[]) {
  const s = a.slice().sort((x, y) => x - y);
  const h = s.length >> 1;
  return s.length % 2 ? s[h] : (s[h - 1] + s[h]) / 2;
}

export type Concentration = {
  /** mean over queries of Dmax / Dmin */
  maxOverMin: number;
  /** He, Kumar & Chang relative contrast: E[Dmean] / E[Dmin] */
  relativeContrast: number;
  /** mean 10th-NN distance / mean distance */
  tenthOverMean: number;
  nnOverMean: number;
  /** median over queries of the MLE local intrinsic dimension at k = 20 */
  lid: number;
  /** coefficient of variation of the pooled normalized distances */
  cv: number;
};

export function concentration(ds: Dataset, t: Truth): Concentration {
  let mom = 0;
  let sMean = 0;
  let sMin = 0;
  let s10 = 0;
  const lids: number[] = [];
  for (let j = 0; j < ds.q; j++) {
    const dmin = Math.max(t.nnDist[j][0], 1e-12);
    mom += t.max[j] / dmin;
    sMean += t.mean[j];
    sMin += dmin;
    s10 += t.nnDist[j][K - 1];
    const k = 20;
    const rk = t.nnDist[j][k - 1];
    let acc = 0;
    for (let i = 0; i < k; i++) acc += Math.log(Math.max(t.nnDist[j][i], 1e-12) / rk);
    lids.push(acc < 0 ? -k / acc : 0);
  }
  // CV from the histogram (bin centres)
  let tot = 0;
  let m1 = 0;
  let m2 = 0;
  for (let b = 0; b < HIST_BINS; b++) {
    const x = ((b + 0.5) / HIST_BINS) * HIST_MAX;
    tot += t.hist[b];
    m1 += t.hist[b] * x;
    m2 += t.hist[b] * x * x;
  }
  m1 /= tot;
  m2 /= tot;
  return {
    maxOverMin: mom / ds.q,
    relativeContrast: sMean / sMin,
    tenthOverMean: s10 / sMean,
    nnOverMean: sMin / sMean,
    lid: median(lids),
    cv: Math.sqrt(Math.max(0, m2 - m1 * m1)) / m1,
  };
}

/* ------------------------------------------------------------- top-k heap */

class TopK {
  idx: Int32Array;
  d: Float64Array;
  size = 0;
  constructor(public k: number) {
    this.idx = new Int32Array(k);
    this.d = new Float64Array(k);
  }
  /** largest kept squared distance, or Infinity while not full */
  worst() {
    return this.size < this.k ? Infinity : this.d[0];
  }
  push(i: number, dist: number) {
    if (this.size < this.k) {
      let c = this.size++;
      while (c > 0) {
        const p = (c - 1) >> 1;
        if (this.d[p] >= dist) break;
        this.d[c] = this.d[p];
        this.idx[c] = this.idx[p];
        c = p;
      }
      this.d[c] = dist;
      this.idx[c] = i;
    } else if (dist < this.d[0]) {
      let c = 0;
      for (;;) {
        const l = 2 * c + 1;
        if (l >= this.k) break;
        const r = l + 1;
        const big = r < this.k && this.d[r] > this.d[l] ? r : l;
        if (this.d[big] <= dist) break;
        this.d[c] = this.d[big];
        this.idx[c] = this.idx[big];
        c = big;
      }
      this.d[c] = dist;
      this.idx[c] = i;
    }
  }
}

/** Min-heap of (key, node, payload) for best-bin-first. */
class MinQueue {
  key: number[] = [];
  node: number[] = [];
  push(k: number, v: number) {
    const key = this.key;
    const node = this.node;
    let c = key.length;
    key.push(k);
    node.push(v);
    while (c > 0) {
      const p = (c - 1) >> 1;
      if (key[p] <= k) break;
      key[c] = key[p];
      node[c] = node[p];
      c = p;
    }
    key[c] = k;
    node[c] = v;
  }
  get length() {
    return this.key.length;
  }
  topKey() {
    return this.key[0];
  }
  pop() {
    const key = this.key;
    const node = this.node;
    const top = node[0];
    const lastK = key.pop()!;
    const lastV = node.pop()!;
    const n = key.length;
    if (n > 0) {
      let c = 0;
      for (;;) {
        const l = 2 * c + 1;
        if (l >= n) break;
        const r = l + 1;
        const small = r < n && key[r] < key[l] ? r : l;
        if (key[small] >= lastK) break;
        key[c] = key[small];
        node[c] = node[small];
        c = small;
      }
      key[c] = lastK;
      node[c] = lastV;
    }
    return top;
  }
}

export type SearchResult = {
  found: Int32Array;
  distanceComputations: number;
  leavesVisited: number;
  visitedLeafIds?: number[];
};

export function recallAt10(found: ArrayLike<number>, truth: Int32Array) {
  const s = new Set<number>();
  for (let i = 0; i < found.length; i++) s.add(found[i]);
  let hit = 0;
  for (let i = 0; i < K; i++) if (s.has(truth[i])) hit++;
  return hit / K;
}

/* --------------------------------------------------------------- kd-tree */

export type KdTree = {
  kind: 'kd';
  perm: Int32Array;
  // per node
  dim: Int32Array;
  val: Float64Array;
  left: Int32Array;
  right: Int32Array;
  start: Int32Array;
  end: Int32Array;
  leafId: Int32Array; // -1 for internal nodes
  leaves: number;
  nodes: number;
  /** distinct coordinates used by at least one split */
  dimsSplit: number;
  depth: number;
};

export function buildKd(ds: Dataset): KdTree {
  const { n, D, X } = ds;
  const cap = 4 * Math.ceil(n / LEAF) + 8;
  const dim = new Int32Array(cap).fill(-1);
  const val = new Float64Array(cap);
  const left = new Int32Array(cap).fill(-1);
  const right = new Int32Array(cap).fill(-1);
  const start = new Int32Array(cap);
  const end = new Int32Array(cap);
  const leafId = new Int32Array(cap).fill(-1);
  const perm = new Int32Array(n);
  for (let i = 0; i < n; i++) perm[i] = i;
  let nodes = 0;
  let leaves = 0;
  let depthMax = 0;
  const used = new Uint8Array(D);
  const rec = (s: number, e: number, depth: number): number => {
    const id = nodes++;
    start[id] = s;
    end[id] = e;
    if (depth > depthMax) depthMax = depth;
    if (e - s <= LEAF) {
      leafId[id] = leaves++;
      return id;
    }
    let best = 0;
    let bestSpread = -1;
    for (let c = 0; c < D; c++) {
      let lo = Infinity;
      let hi = -Infinity;
      for (let i = s; i < e; i++) {
        const v = X[perm[i] * D + c];
        if (v < lo) lo = v;
        if (v > hi) hi = v;
      }
      if (hi - lo > bestSpread) {
        bestSpread = hi - lo;
        best = c;
      }
    }
    const slice = Array.from(perm.subarray(s, e)).sort((a, b) => X[a * D + best] - X[b * D + best]);
    for (let i = 0; i < slice.length; i++) perm[s + i] = slice[i];
    const mid = s + ((e - s) >> 1);
    dim[id] = best;
    val[id] = (X[perm[mid - 1] * D + best] + X[perm[mid] * D + best]) / 2;
    used[best] = 1;
    left[id] = rec(s, mid, depth + 1);
    right[id] = rec(mid, e, depth + 1);
    return id;
  };
  rec(0, n, 0);
  let dimsSplit = 0;
  for (let c = 0; c < D; c++) dimsSplit += used[c];
  return { kind: 'kd', perm, dim, val, left, right, start, end, leafId, leaves, nodes, dimsSplit, depth: depthMax };
}

/**
 * Best-bin-first search over the kd-tree with the exact box bound. Unexplored branches wait in a queue keyed by
 * the squared distance from the query to their cell; a branch whose bound is >= the current 10th-best distance
 * is pruned. `budget` caps distance computations (Infinity = exact search).
 */
export function searchKd(ds: Dataset, t: KdTree, qi: number, budget: number, trackLeaves = false): SearchResult {
  const { D, X, Qs } = ds;
  const top = new TopK(K);
  const off = new Float64Array(D);
  // each queued entry needs its own offset vector for the incremental bound; store them in a side table
  const offs: Float64Array[] = [];
  const queue = new MinQueue();
  let comps = 0;
  let leavesVisited = 0;
  const visited: number[] = [];
  offs.push(off);
  queue.push(0, 0); // entry 0: node 0 with offset table 0
  const entryNode: number[] = [0];
  while (queue.length) {
    if (queue.topKey() >= top.worst()) break;
    if (comps >= budget) break;
    const e = queue.pop();
    let node = entryNode[e];
    const o = offs[e];
    let rd = 0;
    for (let c = 0; c < D; c++) rd += o[c] * o[c];
    while (t.leafId[node] < 0) {
      const c = t.dim[node];
      const diff = Qs[qi * D + c] - t.val[node];
      const near = diff < 0 ? t.left[node] : t.right[node];
      const far = diff < 0 ? t.right[node] : t.left[node];
      const oldOff = o[c];
      const farRd = rd - oldOff * oldOff + diff * diff;
      if (farRd < top.worst()) {
        const no = o.slice();
        no[c] = diff;
        offs.push(no);
        entryNode.push(far);
        queue.push(farRd, offs.length - 1);
      }
      node = near;
    }
    // leaf
    leavesVisited++;
    if (trackLeaves) visited.push(t.leafId[node]);
    for (let i = t.start[node]; i < t.end[node] && comps < budget; i++) {
      const p = t.perm[i];
      top.push(p, sqDist(Qs, qi, X, p, D));
      comps++;
    }
  }
  return { found: top.idx.slice(0, top.size), distanceComputations: comps, leavesVisited, visitedLeafIds: trackLeaves ? visited : undefined };
}

/* ------------------------------------------------ hyperplane (Annoy) tree */

export type HpTree = {
  kind: 'hp';
  perm: Int32Array;
  normal: Float64Array[]; // per internal node, unit vector (empty for leaves)
  offset: Float64Array;
  left: Int32Array;
  right: Int32Array;
  start: Int32Array;
  end: Int32Array;
  leafId: Int32Array;
  leaves: number;
  nodes: number;
};

export function buildHp(ds: Dataset, seed = 3): HpTree {
  const { n, D, X } = ds;
  const rng = makeRng(seed * 101 + D);
  const cap = 4 * Math.ceil(n / LEAF) + 8;
  const normal: Float64Array[] = new Array(cap);
  const offset = new Float64Array(cap);
  const left = new Int32Array(cap).fill(-1);
  const right = new Int32Array(cap).fill(-1);
  const start = new Int32Array(cap);
  const end = new Int32Array(cap);
  const leafId = new Int32Array(cap).fill(-1);
  const perm = new Int32Array(n);
  for (let i = 0; i < n; i++) perm[i] = i;
  let nodes = 0;
  let leaves = 0;
  const p = new Float64Array(D);
  const qv = new Float64Array(D);
  const rec = (s: number, e: number): number => {
    const id = nodes++;
    start[id] = s;
    end[id] = e;
    if (e - s <= LEAF) {
      leafId[id] = leaves++;
      normal[id] = new Float64Array(0);
      return id;
    }
    const count = e - s;
    const i0 = Math.floor(rng() * count);
    let j0 = Math.floor(rng() * (count - 1));
    if (j0 >= i0) j0++;
    for (let c = 0; c < D; c++) {
      p[c] = X[perm[s + i0] * D + c];
      qv[c] = X[perm[s + j0] * D + c];
    }
    // Annoy's two_means: weighted online 2-means from the two seeds
    let ic = 1;
    let jc = 1;
    const steps = 48;
    for (let l = 0; l < steps; l++) {
      const k = perm[s + Math.floor(rng() * count)];
      let di = 0;
      let dj = 0;
      for (let c = 0; c < D; c++) {
        const x = X[k * D + c];
        di += (p[c] - x) * (p[c] - x);
        dj += (qv[c] - x) * (qv[c] - x);
      }
      di *= ic;
      dj *= jc;
      if (di < dj) {
        for (let c = 0; c < D; c++) p[c] = (p[c] * ic + X[k * D + c]) / (ic + 1);
        ic++;
      } else if (dj < di) {
        for (let c = 0; c < D; c++) qv[c] = (qv[c] * jc + X[k * D + c]) / (jc + 1);
        jc++;
      }
    }
    const w = new Float64Array(D);
    let norm = 0;
    for (let c = 0; c < D; c++) {
      w[c] = p[c] - qv[c];
      norm += w[c] * w[c];
    }
    norm = Math.sqrt(norm);
    let a = 0;
    if (norm > 1e-12) {
      for (let c = 0; c < D; c++) {
        w[c] /= norm;
        a -= (w[c] * (p[c] + qv[c])) / 2;
      }
    }
    // partition by side; fall back to a median split on the margin if one side is empty
    const margins = new Float64Array(count);
    for (let i = 0; i < count; i++) {
      let s2 = a;
      const r = perm[s + i];
      for (let c = 0; c < D; c++) s2 += w[c] * X[r * D + c];
      margins[i] = s2;
    }
    const order = Array.from({ length: count }, (_, i) => i).sort((x, y) => margins[y] - margins[x]);
    let pos = 0;
    while (pos < count && margins[order[pos]] > 0) pos++;
    if (pos === 0 || pos === count) {
      pos = count >> 1;
      a -= margins[order[pos]];
    }
    const saved = order.map((i) => perm[s + i]);
    for (let i = 0; i < count; i++) perm[s + i] = saved[i];
    normal[id] = w;
    offset[id] = a;
    left[id] = rec(s, s + pos); // positive side
    right[id] = rec(s + pos, e); // non-positive side
    return id;
  };
  rec(0, n);
  return { kind: 'hp', perm, normal, offset, left, right, start, end, leafId, leaves, nodes };
}

export function searchHp(ds: Dataset, t: HpTree, qi: number, budget: number, trackLeaves = false): SearchResult {
  const { D, X, Qs } = ds;
  const top = new TopK(K);
  const queue = new MinQueue();
  let comps = 0;
  let leavesVisited = 0;
  const visited: number[] = [];
  const entryNode: number[] = [0];
  const entryBound: number[] = [0];
  queue.push(0, 0);
  while (queue.length) {
    if (queue.topKey() >= top.worst()) break;
    if (comps >= budget) break;
    const e = queue.pop();
    let node = entryNode[e];
    const bound = entryBound[e];
    while (t.leafId[node] < 0) {
      const w = t.normal[node];
      let m = t.offset[node];
      for (let c = 0; c < D; c++) m += w[c] * Qs[qi * D + c];
      const near = m > 0 ? t.left[node] : t.right[node];
      const far = m > 0 ? t.right[node] : t.left[node];
      const farBound = Math.max(bound, m * m);
      if (farBound < top.worst()) {
        entryNode.push(far);
        entryBound.push(farBound);
        queue.push(farBound, entryNode.length - 1);
      }
      node = near;
    }
    leavesVisited++;
    if (trackLeaves) visited.push(t.leafId[node]);
    for (let i = t.start[node]; i < t.end[node] && comps < budget; i++) {
      const p = t.perm[i];
      top.push(p, sqDist(Qs, qi, X, p, D));
      comps++;
    }
  }
  return { found: top.idx.slice(0, top.size), distanceComputations: comps, leavesVisited, visitedLeafIds: trackLeaves ? visited : undefined };
}

/* ------------------------------------------------------------------- LSH */

/** Collision probability of the 2-stable hash for two points at distance c with bucket width w (E2LSH). */
export function pStableCollision(c: number, w: number) {
  if (c <= 1e-12) return 1;
  const t = w / c;
  // 1 - 2Φ(-t) - 2/(sqrt(2π) t) (1 - e^{-t²/2})
  const phiNeg = 0.5 * erfc(t / Math.SQRT2);
  return 1 - 2 * phiNeg - (2 / (Math.sqrt(2 * Math.PI) * t)) * (1 - Math.exp((-t * t) / 2));
}

function erfc(x: number) {
  // Numerical Recipes erfc approximation (fractional error < 1.2e-7)
  const z = Math.abs(x);
  const t = 1 / (1 + 0.5 * z);
  const r =
    t *
    Math.exp(
      -z * z -
        1.26551223 +
        t * (1.00002368 + t * (0.37409196 + t * (0.09678418 + t * (-0.18628806 + t * (0.27886807 + t * (-1.13520398 + t * (1.48851587 + t * (-0.82215223 + t * 0.17087277)))))))),
    );
  return x >= 0 ? r : 2 - r;
}

/** Probability a point at distance c becomes a candidate: at least one of L tables agrees on all k hashes. */
export function candidateProbability(c: number, w: number, k: number, L: number) {
  const p = pStableCollision(c, w);
  return 1 - Math.pow(1 - Math.pow(p, k), L);
}

export type Lsh = {
  k: number;
  L: number;
  w: number;
  A: Float64Array[]; // L*k projection vectors
  B: Float64Array; // L*k offsets
  tables: Map<string, number[]>[];
};

export function lshWidth(t: Truth) {
  let s = 0;
  for (const d of t.nnDist) s += d[K - 1];
  return 4 * (s / t.nnDist.length);
}

/**
 * Builds the L hash tables one at a time, so the browser can spread the k × L × n projections over small time slices.
 * All projection vectors and offsets are drawn up front, so the tables match a one-shot build exactly.
 */
export class LshBuilder {
  lsh: Lsh;
  constructor(public ds: Dataset, w: number, k: number, L: number, seed = 11) {
    const { D } = ds;
    const rng = makeRng(seed * 977 + D * 13 + k * 7 + L);
    const A: Float64Array[] = [];
    const B = new Float64Array(k * L);
    for (let h = 0; h < k * L; h++) {
      const a = new Float64Array(D);
      for (let c = 0; c < D; c++) a[c] = gauss(rng);
      A.push(a);
      B[h] = rng() * w;
    }
    this.lsh = { k, L, w, A, B, tables: [] };
  }
  get done() {
    return this.lsh.tables.length >= this.lsh.L;
  }
  /** Build up to `count` more tables. */
  step(count: number) {
    const { n, D, X } = this.ds;
    const { k, w, A, B, tables } = this.lsh;
    const codes = new Int32Array(k);
    const stop = Math.min(this.lsh.L, tables.length + count);
    for (let l = tables.length; l < stop; l++) {
      const table = new Map<string, number[]>();
      for (let i = 0; i < n; i++) {
        for (let h = 0; h < k; h++) {
          const a = A[l * k + h];
          let dot = B[l * k + h];
          for (let c = 0; c < D; c++) dot += a[c] * X[i * D + c];
          codes[h] = Math.floor(dot / w);
        }
        const key = codes.join(',');
        const bucket = table.get(key);
        if (bucket) bucket.push(i);
        else table.set(key, [i]);
      }
      tables.push(table);
    }
    return this.done;
  }
}

export function buildLsh(ds: Dataset, w: number, k: number, L: number, seed = 11): Lsh {
  const b = new LshBuilder(ds, w, k, L, seed);
  b.step(L);
  return b.lsh;
}

export function searchLsh(ds: Dataset, lsh: Lsh, qi: number, budget: number): SearchResult & { candidates: number } {
  const { D, X, Qs, n } = ds;
  const votes = new Uint16Array(n);
  const cand: number[] = [];
  const codes = new Int32Array(lsh.k);
  for (let l = 0; l < lsh.L; l++) {
    for (let h = 0; h < lsh.k; h++) {
      const a = lsh.A[l * lsh.k + h];
      let dot = lsh.B[l * lsh.k + h];
      for (let c = 0; c < D; c++) dot += a[c] * Qs[qi * D + c];
      codes[h] = Math.floor(dot / lsh.w);
    }
    const bucket = lsh.tables[l].get(codes.join(','));
    if (!bucket) continue;
    for (const p of bucket) {
      if (votes[p] === 0) cand.push(p);
      votes[p]++;
    }
  }
  cand.sort((a, b) => votes[b] - votes[a] || a - b);
  const top = new TopK(K);
  let comps = 0;
  for (const p of cand) {
    if (comps >= budget) break;
    top.push(p, sqDist(Qs, qi, X, p, D));
    comps++;
  }
  return { found: top.idx.slice(0, top.size), distanceComputations: comps, leavesVisited: 0, candidates: cand.length };
}

/* ------------------------------------------------------------------ HNSW */

export const HNSW_M = 8;
export const HNSW_M0 = 16;
export const HNSW_EFC = 32;

export type Hnsw = {
  levels: Int32Array;
  /** links[layer][node] = neighbour ids */
  links: Int32Array[][];
  entry: number;
  maxLevel: number;
  buildComputations: number;
};

type HnswCtx = { ds: Dataset; stamp: Int32Array; epoch: number; comps: number };

/** Beam search in one layer (Algorithm 2). Returns ids sorted nearest first, with their squared distances. */
function searchLayer(ctx: HnswCtx, h: Hnsw, vec: Float64Array, vi: number, eps: number[], ef: number, layer: number, budget: number) {
  const { ds, stamp } = ctx;
  const D = ds.D;
  const X = ds.X;
  ctx.epoch++;
  const ep = ctx.epoch;
  const cand = new MinQueue();
  // W as a max-structure: keep arrays and track worst linearly (ef is small)
  const W: number[] = [];
  const Wd: number[] = [];
  let worstI = -1;
  const recomputeWorst = () => {
    worstI = 0;
    for (let i = 1; i < Wd.length; i++) if (Wd[i] > Wd[worstI]) worstI = i;
  };
  for (const e of eps) {
    if (stamp[e] === ep) continue;
    stamp[e] = ep;
    const d = sqDist(vec, vi, X, e, D);
    ctx.comps++;
    cand.push(d, e);
    W.push(e);
    Wd.push(d);
  }
  recomputeWorst();
  while (cand.length) {
    const cd = cand.topKey();
    if (Wd.length >= ef && cd > Wd[worstI]) break;
    const c = cand.pop();
    const nb = h.links[layer][c];
    if (!nb) continue;
    for (let t = 0; t < nb.length; t++) {
      const e = nb[t];
      if (stamp[e] === ep) continue;
      if (ctx.comps >= budget) break;
      stamp[e] = ep;
      const d = sqDist(vec, vi, X, e, D);
      ctx.comps++;
      if (Wd.length < ef || d < Wd[worstI]) {
        cand.push(d, e);
        if (Wd.length < ef) {
          W.push(e);
          Wd.push(d);
          if (Wd.length === 1 || d > Wd[worstI]) worstI = Wd.length - 1;
        } else {
          W[worstI] = e;
          Wd[worstI] = d;
          recomputeWorst();
        }
      }
    }
    if (ctx.comps >= budget) break;
  }
  const order = W.map((_, i) => i).sort((a, b) => Wd[a] - Wd[b]);
  return { ids: order.map((i) => W[i]), d: order.map((i) => Wd[i]) };
}

/** Incremental HNSW construction (Algorithm 1), so the browser can build it in small time slices. */
export class HnswBuilder {
  h: Hnsw;
  next = 1;
  private ctx: HnswCtx;
  constructor(public ds: Dataset, seed = 5) {
    const { n, D } = ds;
    const rng = makeRng(seed * 31 + D * 7 + 3);
    const mL = 1 / Math.log(HNSW_M);
    const levels = new Int32Array(n);
    let maxLevelAll = 0;
    for (let i = 0; i < n; i++) {
      let u = rng();
      if (u <= 1e-9) u = 1e-9;
      levels[i] = Math.min(8, Math.floor(-Math.log(u) * mL));
      if (levels[i] > maxLevelAll) maxLevelAll = levels[i];
    }
    const links: Int32Array[][] = [];
    for (let l = 0; l <= maxLevelAll; l++) links.push(new Array(n));
    this.h = { levels, links, entry: 0, maxLevel: levels[0], buildComputations: 0 };
    this.ctx = { ds, stamp: new Int32Array(n), epoch: 0, comps: 0 };
    for (let l = 0; l <= levels[0]; l++) links[l][0] = new Int32Array(0);
  }
  get done() {
    return this.next >= this.ds.n;
  }
  /** Insert up to `count` more points. */
  step(count: number) {
    const { ds, h, ctx } = this;
    const { D, X } = ds;
    const { levels, links } = h;
    const stop = Math.min(ds.n, this.next + count);
    for (let i = this.next; i < stop; i++) {
      const li = levels[i];
      let eps = [h.entry];
      for (let l = h.maxLevel; l > li; l--) {
        const r = searchLayer(ctx, h, X, i, eps, 1, l, Infinity);
        eps = [r.ids[0]];
      }
      for (let l = Math.min(li, h.maxLevel); l >= 0; l--) {
        const r = searchLayer(ctx, h, X, i, eps, HNSW_EFC, l, Infinity);
        const mmax = l === 0 ? HNSW_M0 : HNSW_M;
        const chosen = r.ids.slice(0, HNSW_M);
        links[l][i] = Int32Array.from(chosen);
        for (const nb of chosen) {
          const cur = links[l][nb];
          if (cur.length < mmax) {
            const grown = new Int32Array(cur.length + 1);
            grown.set(cur);
            grown[cur.length] = i;
            links[l][nb] = grown;
          } else {
            // shrink back to the mmax closest (simple selection)
            const all = Array.from(cur);
            all.push(i);
            const dd = all.map((e) => {
              ctx.comps++;
              return sqDist(X, nb, X, e, D);
            });
            const ord = all.map((_, t) => t).sort((a, b) => dd[a] - dd[b]);
            links[l][nb] = Int32Array.from(ord.slice(0, mmax).map((t) => all[t]));
          }
        }
        eps = r.ids;
      }
      for (let l = h.maxLevel + 1; l <= li; l++) links[l][i] = new Int32Array(0);
      if (li > h.maxLevel) {
        h.maxLevel = li;
        h.entry = i;
      }
    }
    this.next = stop;
    h.buildComputations = ctx.comps;
    return this.done;
  }
}

export function buildHnsw(ds: Dataset, seed = 5): Hnsw {
  const b = new HnswBuilder(ds, seed);
  b.step(ds.n);
  return b.h;
}

export function searchHnsw(ds: Dataset, h: Hnsw, qi: number, budget: number, ef = 64): SearchResult {
  const ctx: HnswCtx = { ds, stamp: new Int32Array(ds.n), epoch: 0, comps: 0 };
  let eps = [h.entry];
  for (let l = h.maxLevel; l > 0; l--) {
    const r = searchLayer(ctx, h, ds.Qs, qi, eps, 1, l, budget);
    eps = [r.ids[0]];
  }
  const r = searchLayer(ctx, h, ds.Qs, qi, eps, Math.max(ef, K), 0, budget);
  return { found: Int32Array.from(r.ids.slice(0, K)), distanceComputations: ctx.comps, leavesVisited: 0 };
}

/* -------------------------------------------------------- per-dimension run */

export type IndexKind = 'kd' | 'hp';

export type DimResult = {
  D: number;
  conc: Concentration;
  hist: number[];
  kdLeaves: number;
  kdLeavesVisitedExact: number;
  kdFracScannedExact: number;
  kdDimsSplit: number;
  kdDepth: number;
  hpLeaves: number;
  hpLeavesVisitedExact: number;
  hpFracScannedExact: number;
  recallKd: number;
  recallHp: number;
  recallLsh: number;
  recallHnsw: number;
  recallRandom: number;
  lshCandidates: number;
  hnswComps: number;
  w: number;
  /** leaf ids visited by the exact search for query 0, for each tree */
  kdVisited0: number[];
  hpVisited0: number[];
  /** normalized distance of the 1st and 10th neighbour for the S-curve overlay */
  nnNorm: number;
  tenthNorm: number;
  meanDist: number;
};

export type Built = { ds: Dataset; truth: Truth; kd: KdTree; hp: HpTree; conc: Concentration };

export function buildStatic(kind: DataKind, D: number, m: number): Built {
  const ds = makeDataset(kind, D, m);
  const truth = groundTruth(ds);
  return { ds, truth, kd: buildKd(ds), hp: buildHp(ds), conc: concentration(ds, truth) };
}

/**
 * Exact tree searches, concentration stats and histogram geometry for one built dataset. With a `budgetFrac`, also the
 * budgeted recall of both trees, plus LSH and HNSW when they are given; without one those fields are NaN, which is how
 * the top panel stays cheap.
 */
export function evaluate(b: Built, lsh: Lsh | null, hnsw: Hnsw | null, budgetFrac: number | null): DimResult {
  const { ds, truth, kd, hp, conc } = b;
  const budgeted = budgetFrac !== null;
  const budget = budgeted ? Math.max(K, Math.round(budgetFrac * ds.n)) : 0;
  let kdVis = 0;
  let kdComps = 0;
  let hpVis = 0;
  let hpComps = 0;
  let rKd = 0;
  let rHp = 0;
  let rL = 0;
  let rH = 0;
  let lshC = 0;
  let hC = 0;
  let kdVisited0: number[] = [];
  let hpVisited0: number[] = [];
  for (let j = 0; j < ds.q; j++) {
    const ek = searchKd(ds, kd, j, Infinity, j === 0);
    kdVis += ek.leavesVisited;
    kdComps += ek.distanceComputations;
    if (j === 0) kdVisited0 = ek.visitedLeafIds ?? [];
    const eh = searchHp(ds, hp, j, Infinity, j === 0);
    hpVis += eh.leavesVisited;
    hpComps += eh.distanceComputations;
    if (j === 0) hpVisited0 = eh.visitedLeafIds ?? [];
    if (!budgeted) continue;
    rKd += recallAt10(searchKd(ds, kd, j, budget).found, truth.nn[j]);
    rHp += recallAt10(searchHp(ds, hp, j, budget).found, truth.nn[j]);
    if (lsh) {
      const sl = searchLsh(ds, lsh, j, budget);
      rL += recallAt10(sl.found, truth.nn[j]);
      lshC += sl.candidates;
    }
    if (hnsw) {
      const sh = searchHnsw(ds, hnsw, j, budget, budget);
      rH += recallAt10(sh.found, truth.nn[j]);
      hC += sh.distanceComputations;
    }
  }
  const q = ds.q;
  let meanDist = 0;
  let nn = 0;
  let tenth = 0;
  for (let j = 0; j < q; j++) {
    meanDist += truth.mean[j];
    nn += truth.nnDist[j][0] / truth.mean[j];
    tenth += truth.nnDist[j][K - 1] / truth.mean[j];
  }
  let hmax = 0;
  for (const v of truth.hist) hmax = Math.max(hmax, v);
  return {
    D: ds.D,
    conc,
    hist: Array.from(truth.hist, (v) => v / (hmax || 1)),
    kdLeaves: kd.leaves,
    kdLeavesVisitedExact: kdVis / q,
    kdFracScannedExact: kdComps / q / ds.n,
    kdDimsSplit: kd.dimsSplit,
    kdDepth: kd.depth,
    hpLeaves: hp.leaves,
    hpLeavesVisitedExact: hpVis / q,
    hpFracScannedExact: hpComps / q / ds.n,
    recallKd: budgeted ? rKd / q : NaN,
    recallHp: budgeted ? rHp / q : NaN,
    recallLsh: budgeted && lsh ? rL / q : NaN,
    recallHnsw: budgeted && hnsw ? rH / q : NaN,
    recallRandom: budgeted ? Math.min(1, budget / ds.n) : NaN,
    lshCandidates: budgeted && lsh ? lshC / q : NaN,
    hnswComps: budgeted && hnsw ? hC / q : NaN,
    w: lsh ? lsh.w : lshWidth(truth),
    kdVisited0,
    hpVisited0,
    nnNorm: nn / q,
    tenthNorm: tenth / q,
    meanDist: meanDist / q,
  };
}

/* ==================================================================== UI */

export const validDims = (kind: DataKind, m: number) => DIMS.filter((D) => kind === 'uniform' || D >= 2 * m);

/** LSH exponent ρ = ln(1/p1) / ln(1/p2) with p1 at the 10th-neighbour distance and p2 at the mean distance. */
export function lshRho(conc: Concentration) {
  const w = 4; // distances measured in units of the 10th-neighbour distance, so w = 4 as in lshWidth
  const p1 = pStableCollision(1, w);
  const p2 = pStableCollision(1 / Math.max(conc.tenthOverMean, 1e-9), w);
  return Math.log(1 / p1) / Math.log(1 / Math.max(p2, 1e-12));
}

type Caches = {
  built: Map<string, Built>;
  lsh: Map<string, Lsh>;
  hnsw: Map<string, Hnsw>;
};

const cfgKey = (kind: DataKind, m: number, D: number) => `${kind}|${kind === 'manifold' ? m : 0}|${D}`;

function getBuilt(c: Caches, kind: DataKind, m: number, D: number) {
  const key = cfgKey(kind, m, D);
  let b = c.built.get(key);
  if (!b) {
    if (c.built.size > 12) {
      c.built.clear();
      c.lsh.clear();
    }
    b = buildStatic(kind, D, m);
    c.built.set(key, b);
  }
  return b;
}

const SERIES_KD = 'var(--viz-3)';
const SERIES_HP = 'var(--viz-2)';
const SERIES_LSH = 'var(--viz-4)';
const SERIES_HNSW = 'var(--viz-1)';
const RANDOM_INK = 'var(--viz-ink-2)';
const HIST_FILL = 'var(--viz-ink-muted)';

const f2 = (v: number) => v.toFixed(2);
const f1 = (v: number) => v.toFixed(1);
const pct = (v: number) => `${fmtNum(v * 100, v < 0.1 ? 1 : 0)}%`;

type Sweep = { key: string; rows: Record<number, DimResult> };

function Histogram({ r, k, L, width }: { r: DimResult; k: number; L: number; width: number }) {
  const H = 180;
  const ml = 36;
  const mr = 28;
  const mt = 8;
  const mb = 34;
  const pw = width - ml - mr;
  const ph = H - mt - mb;
  const x = (v: number) => ml + (v / HIST_MAX) * pw;
  const y = (v: number) => mt + ph * (1 - v);
  const bw = pw / HIST_BINS;
  const curve: string[] = [];
  for (let i = 0; i <= 80; i++) {
    const v = (i / 80) * HIST_MAX;
    const p = candidateProbability(v * r.meanDist, r.w, k, L);
    curve.push(`${i === 0 ? 'M' : 'L'}${x(v).toFixed(1)},${y(p).toFixed(1)}`);
  }
  const nnX = x(r.nnNorm);
  const tenthX = x(r.tenthNorm);
  return (
    <svg viewBox={`0 0 ${width} ${H}`} width={width} height={H} role="img" aria-label={`Histogram of query-to-point distances divided by the mean distance at D = ${r.D}; the nearest neighbour sits at ${f2(r.nnNorm)} of the mean.`}>
      {[0, 0.5, 1].map((g) => (
        <line key={g} className="viz-grid-line" x1={ml} x2={ml + pw} y1={y(g)} y2={y(g)} />
      ))}
      {r.hist.map((v, i) => (v > 0 ? <rect key={i} x={ml + i * bw + 0.5} y={y(v)} width={Math.max(0.5, bw - 1)} height={ph * v} fill={HIST_FILL} fillOpacity={0.5} /> : null))}
      <path d={curve.join('')} fill="none" stroke={SERIES_LSH} strokeWidth={2} />
      <line x1={nnX} x2={nnX} y1={mt} y2={mt + ph} stroke="var(--viz-ink)" strokeWidth={1} strokeDasharray="3 2" />
      <line x1={tenthX} x2={tenthX} y1={mt} y2={mt + ph} stroke="var(--viz-ink)" strokeWidth={1} />
      <text x={nnX - 3} y={mt + 10} fontSize={10} textAnchor="end">
        1st
      </text>
      <text x={tenthX + 3} y={mt + 22} fontSize={10} textAnchor="start">
        10th
      </text>
      <line className="viz-axis-line" x1={ml} x2={ml + pw} y1={mt + ph} y2={mt + ph} />
      {[0, 0.5, 1, 1.5, 2].map((t) => (
        <text key={t} x={x(t)} y={mt + ph + 13} fontSize={10} textAnchor="middle">
          {t}
        </text>
      ))}
      <text x={ml + pw / 2} y={H - 4} fontSize={10} textAnchor="middle">
        distance ÷ mean distance
      </text>
      <text x={ml - 6} y={mt + 4} fontSize={10} textAnchor="end">
        max
      </text>
      <text x={ml - 6} y={mt + ph} fontSize={10} textAnchor="end">
        0
      </text>
      {[0, 0.5, 1].map((t) => (
        <text key={t} x={ml + pw + 5} y={y(t) + 3} fontSize={10}>
          {t}
        </text>
      ))}
    </svg>
  );
}

function LeafGrid({ leaves, visited, color, width, label }: { leaves: number; visited: number[]; color: string; width: number; label: string }) {
  const cols = width >= 520 ? 32 : 16;
  const cell = Math.max(10, Math.min(18, Math.floor((width - 4) / cols)));
  const rows = Math.ceil(leaves / cols);
  const H = 22 + rows * cell + 4;
  const seen = new Set(visited);
  const first = visited.length ? visited[0] : -1;
  const W = cols * cell + 4;
  return (
    <svg viewBox={`0 0 ${Math.max(W, width)} ${H}`} width={Math.max(W, width)} height={H} role="img" aria-label={`${label}: ${visited.length} of ${leaves} leaf cells opened by an exact 10-nearest-neighbour search`}>
      <text x={2} y={12} fontSize={11} fill="var(--viz-ink)">
        {label}: {visited.length} of {leaves} leaves opened (query 1)
      </text>
      {Array.from({ length: leaves }, (_, i) => {
        const on = seen.has(i);
        return (
          <rect
            key={i}
            x={2 + (i % cols) * cell}
            y={20 + Math.floor(i / cols) * cell}
            width={cell - 2}
            height={cell - 2}
            rx={2}
            fill={on ? color : 'var(--viz-plane)'}
            fillOpacity={on ? 0.85 : 1}
            stroke={i === first ? 'var(--viz-ink)' : 'var(--viz-border)'}
            strokeWidth={i === first ? 2 : 0.75}
          />
        );
      })}
    </svg>
  );
}

function SweepChart({ rows, dims, currentD, budgetFrac, width, pending }: { rows: Record<number, DimResult>; dims: number[]; currentD: number; budgetFrac: number; width: number; pending: number | null }) {
  const tip = useTip();
  const H = 232;
  const narrow = width < 460;
  const ml = narrow ? 32 : 40;
  const mr = narrow ? 76 : 104;
  const mt = 8;
  const mb = 36;
  const pw = width - ml - mr;
  const ph = H - mt - mb;
  const x = (D: number) => ml + ((Math.log2(D) - 1) / 7) * pw;
  const y = (v: number) => mt + ph * (1 - v);
  const series: { key: keyof DimResult; label: string; color: string }[] = [
    { key: 'recallKd', label: 'kd-tree', color: SERIES_KD },
    { key: 'recallHp', label: narrow ? 'hyperplane' : 'hyperplane tree', color: SERIES_HP },
    { key: 'recallLsh', label: 'LSH', color: SERIES_LSH },
    { key: 'recallHnsw', label: 'HNSW', color: SERIES_HNSW },
  ];
  const have = dims.filter((D) => rows[D]);
  const lastD = have.length ? have[have.length - 1] : null;
  // direct labels at the right edge, pushed apart so they never overlap
  const labels = lastD
    ? series
        .map((s) => ({ label: s.label, color: s.color, y: y(rows[lastD][s.key] as number) }))
        .concat([{ label: `random ${pct(budgetFrac)}`, color: RANDOM_INK, y: y(Math.min(1, budgetFrac)) }])
        .sort((a, b) => a.y - b.y)
    : [];
  for (let i = 1; i < labels.length; i++) if (labels[i].y - labels[i - 1].y < 12) labels[i].y = labels[i - 1].y + 12;
  const overflow = labels.length ? labels[labels.length - 1].y - (mt + ph - 3) : 0; // keep the lowest label clear of the x-axis tick text
  if (overflow > 0) for (const l of labels) l.y -= overflow;
  return (
    <svg viewBox={`0 0 ${width} ${H}`} width={width} height={H} role="img" aria-label={`Recall at 10 against ambient dimension with ${pct(budgetFrac)} of the distance computations`}>
      {[0, 0.25, 0.5, 0.75, 1].map((g) => (
        <g key={g}>
          <line className="viz-grid-line" x1={ml} x2={ml + pw} y1={y(g)} y2={y(g)} />
          <text x={ml - 6} y={y(g) + 3} fontSize={10} textAnchor="end">
            {g}
          </text>
        </g>
      ))}
      <rect x={Math.max(ml, x(currentD) - 9)} y={mt} width={Math.min(ml + pw, x(currentD) + 9) - Math.max(ml, x(currentD) - 9)} height={ph} fill="var(--viz-neutral)" fillOpacity={0.7} stroke="var(--viz-border)" />
      {dims[0] > DIMS[0] && x(dims[0]) - ml > 70 ? (
        <text x={(ml + x(dims[0])) / 2} y={mt + ph / 2} fontSize={10} textAnchor="middle">
          <tspan x={(ml + x(dims[0])) / 2}>sheet needs</tspan>
          <tspan x={(ml + x(dims[0])) / 2} dy={12}>
            D ≥ {dims[0]}
          </tspan>
        </text>
      ) : null}
      <line x1={ml} x2={ml + pw} y1={y(Math.min(1, budgetFrac))} y2={y(Math.min(1, budgetFrac))} stroke={RANDOM_INK} strokeWidth={1.5} strokeDasharray="5 4" />
      {series.map((s) => {
        const pts = have.map((D) => [x(D), y(rows[D][s.key] as number)] as const);
        return (
          <g key={s.label}>
            {pts.length > 1 ? <path d={pts.map((p, i) => `${i ? 'L' : 'M'}${p[0].toFixed(1)},${p[1].toFixed(1)}`).join('')} fill="none" stroke={s.color} strokeWidth={2} /> : null}
            {have.map((D) => (
              <circle
                key={D}
                cx={x(D)}
                cy={y(rows[D][s.key] as number)}
                r={3.5}
                fill={s.color}
                stroke="var(--viz-surface)"
                strokeWidth={1}
                {...tip(
                  <>
                    <strong>{s.label}</strong>, D = {D}: recall@10 {f2(rows[D][s.key] as number)}
                  </>,
                )}
              />
            ))}
          </g>
        );
      })}
      {labels.map((l) => (
        <g key={l.label}>
          <line x1={ml + pw + 4} x2={ml + pw + 14} y1={l.y} y2={l.y} stroke={l.color} strokeWidth={2} strokeDasharray={l.color === RANDOM_INK ? '3 2' : undefined} />
          <text x={ml + pw + 17} y={l.y + 3} fontSize={10}>
            {l.label}
          </text>
        </g>
      ))}
      <line className="viz-axis-line" x1={ml} x2={ml + pw} y1={mt + ph} y2={mt + ph} />
      {DIMS.map((D) => (
        <text key={D} x={x(D)} y={mt + ph + 14} fontSize={10} textAnchor="middle" fill={D === currentD ? 'var(--viz-ink)' : undefined}>
          {D}
        </text>
      ))}
      <text x={ml + pw / 2} y={H - 4} fontSize={10} textAnchor="middle">
        ambient dimension D (log scale)
      </text>
      {pending !== null ? (
        <text x={ml + pw - 4} y={mt + 12} fontSize={10} textAnchor="end">
          building indexes at D = {pending}…
        </text>
      ) : null}
    </svg>
  );
}

export default function DistanceConcentrationLab() {
  const [kind, setKind] = useState<DataKind>('uniform');
  const [dimIdx, setDimIdx] = useState(1); // D = 4
  const [m, setM] = useState(4);
  const [tree, setTree] = useState<IndexKind>('kd');
  const [budgetPct, setBudgetPct] = useState(5);
  const [lshK, setLshK] = useState(6);
  const [lshL, setLshL] = useState(10);
  const [ref, width] = useSize(720);

  const caches = useRef<Caches>({ built: new Map(), lsh: new Map(), hnsw: new Map() });
  const dims = useMemo(() => validDims(kind, m), [kind, m]);
  const wanted = DIMS[dimIdx];
  const D = dims.find((d) => d >= wanted) ?? dims[dims.length - 1];
  const clamped = D !== wanted;
  const budgetFrac = budgetPct / 100;

  const dKind = useDeferredValue(kind);
  const dD = useDeferredValue(D);
  const dM = useDeferredValue(m);
  const dK = useDeferredValue(lshK);
  const dL = useDeferredValue(lshL);
  const dBudget = useDeferredValue(budgetFrac);

  // The top panel needs no hash tables or graph: LSH tables at D = 256 with k × L = 640 take ~250 ms to build, so
  // they are built table by table in the background job below, and the candidates tile reads the job's row.
  const detail = useMemo(() => {
    const safeD = dKind === 'manifold' && dD < 2 * dM ? validDims(dKind, dM)[0] : dD;
    const b = getBuilt(caches.current, dKind, dM, safeD);
    return evaluate(b, null, null, null);
  }, [dKind, dD, dM]);

  const sweepKey = `${kind}|${kind === 'manifold' ? m : 0}|${lshK}|${lshL}|${budgetPct}`;
  const [sweep, setSweep] = useState<Sweep>({ key: '', rows: {} });
  const [pending, setPending] = useState<number | null>(null);
  const job = useRef(0);
  const firstD = useRef(D);
  firstD.current = D;

  useEffect(() => {
    const token = ++job.current;
    const c = caches.current;
    const order = [...dims].sort((a, b) => (a === firstD.current ? -1 : b === firstD.current ? 1 : a - b));
    let i = 0;
    let builder: HnswBuilder | null = null;
    let builderKey = '';
    let lshBuilder: LshBuilder | null = null;
    let lshBuilderKey = '';
    let timer: ReturnType<typeof setTimeout> | undefined;
    setSweep((s) => (s.key === sweepKey ? s : { key: sweepKey, rows: {} }));
    const tick = () => {
      if (job.current !== token) return;
      const t0 = performance.now();
      while (i < order.length && performance.now() - t0 < 12) {
        const dim = order[i];
        const key = cfgKey(kind, m, dim);
        setPending(dim);
        if (!c.built.has(key)) {
          getBuilt(c, kind, m, dim);
          continue; // re-check the time slice before the next phase
        }
        const b = getBuilt(c, kind, m, dim);
        let h = c.hnsw.get(key);
        if (!h) {
          if (!builder || builderKey !== key) {
            builder = new HnswBuilder(b.ds);
            builderKey = key;
          }
          if (!builder.step(60)) continue;
          h = builder.h;
          if (c.hnsw.size > 24) c.hnsw.clear();
          c.hnsw.set(key, h);
          builder = null;
        }
        const lshKey = `${key}|${lshK}|${lshL}`;
        let l = c.lsh.get(lshKey);
        if (!l) {
          if (!lshBuilder || lshBuilderKey !== lshKey) {
            lshBuilder = new LshBuilder(b.ds, lshWidth(b.truth), lshK, lshL);
            lshBuilderKey = lshKey;
          }
          if (!lshBuilder.step(1)) continue;
          l = lshBuilder.lsh;
          if (c.lsh.size > 24) c.lsh.clear();
          c.lsh.set(lshKey, l);
          lshBuilder = null;
        }
        const row = evaluate(b, l, h, budgetFrac);
        setSweep((s) => ({ key: sweepKey, rows: { ...(s.key === sweepKey ? s.rows : {}), [dim]: row } }));
        i++;
      }
      if (i < order.length) timer = setTimeout(tick, 0);
      else setPending(null);
    };
    timer = setTimeout(tick, 0);
    return () => {
      if (timer !== undefined) clearTimeout(timer);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [sweepKey]);

  const rows = sweep.key === sweepKey ? sweep.rows : {};
  const here = rows[detail.D];
  const c = detail.conc;
  const trueDim = kind === 'uniform' ? detail.D : m;
  const rho = lshRho(c);
  const narrow = width < 620;
  const leftW = narrow ? width : Math.floor(width * 0.56) - 8;
  const rightW = narrow ? width : width - leftW - 16;
  // every query opened every leaf (at uniform D = 16 one query in 40 still skips one, so test the count, not a percentage)
  const kdAll = detail.kdLeavesVisitedExact >= detail.kdLeaves;

  const note = (
    <Note>
      {clamped ? (
        <>
          <strong>A curved sheet of intrinsic dimension {m} needs at least {2 * m} coordinates, so D = {D} is shown.</strong>{' '}
        </>
      ) : null}
      {kind === 'uniform' ? (
        <>
          <strong>
            D = {detail.D}: the farthest point is {f2(c.maxOverMin)}× as far as the nearest, and the 10th neighbour sits at {f2(detail.tenthNorm)} of the mean distance.
          </strong>{' '}
          {kdAll
            ? `An exact 10-NN search opened all ${detail.kdLeaves} kd-tree leaves for every query — no bound ever pruned a branch, so the index is a full scan plus overhead.`
            : `An exact 10-NN search opened ${fmtNum(detail.kdLeavesVisitedExact, detail.kdLeavesVisitedExact > detail.kdLeaves - 1 ? 2 : 1)} of ${detail.kdLeaves} kd-tree leaves on average (${detail.kdFracScannedExact > 0.99 ? `${fmtNum(detail.kdFracScannedExact * 100, 2)}%` : pct(detail.kdFracScannedExact)} of the points).`}{' '}
          {detail.kdDepth < detail.D ? `Every leaf cell is cut on at most ${detail.kdDepth} of the ${detail.D} coordinates and spans the full range of the rest. ` : ''}
        </>
      ) : (
        <>
          <strong>
            Points on a curved sheet of intrinsic dimension {m} inside D = {detail.D}: estimated intrinsic dimension {f1(c.lid)}, relative contrast {f2(c.relativeContrast)} — nearly the same at every D.
          </strong>{' '}
          The hyperplane tree, whose splits follow the data, opens {fmtNum(detail.hpLeavesVisitedExact, 1)} of {detail.hpLeaves} leaves; the kd-tree, which can only cut one coordinate at a time, opens{' '}
          {fmtNum(detail.kdLeavesVisitedExact, 1)} of {detail.kdLeaves}.{' '}
        </>
      )}
      {here ? `With ${budgetPct}% of the distances, HNSW finds ${f1(here.recallHnsw * 10)} of the true 10 neighbours and LSH ${f1(here.recallLsh * 10)}.` : ''}
    </Note>
  );

  return (
    <VizPanel
      title="Distance concentration versus intrinsic dimension"
      subtitle="1,500 points and 40 queries. Raise the ambient dimension and watch the distance histogram, the leaves an exact tree search must open, and the recall every index gets for a fixed number of distance computations."
      controls={
        <>
          <Segmented
            label="Data"
            value={kind}
            onChange={setKind}
            options={[
              { value: 'uniform', label: 'Uniform cube', title: 'Intrinsic dimension = D' },
              { value: 'manifold', label: 'Curved m-dim sheet', title: 'Intrinsic dimension = m, rotated into D' },
            ]}
          />
          <Slider label="Ambient dimension D" min={0} max={DIMS.length - 1} value={dimIdx} onChange={setDimIdx} format={(i) => `${DIMS[i]}`} />
          {kind === 'manifold' ? <Slider label="Intrinsic dimension m" min={1} max={16} value={m} onChange={setM} /> : null}
        </>
      }
      legend={
        <Legend
          items={[
            { label: 'kd-tree', color: SERIES_KD },
            { label: 'Hyperplane tree (Annoy-style)', color: SERIES_HP },
            { label: 'LSH (histogram curve: candidate probability)', color: SERIES_LSH, shape: 'line' },
            { label: 'HNSW', color: SERIES_HNSW, shape: 'line' },
            { label: 'Random sample of the same size', color: RANDOM_INK, shape: 'line' },
            { label: 'Distance histogram', color: HIST_FILL },
          ]}
        />
      }
      stats={
        <Stats
          items={[
            { label: 'Farthest ÷ nearest', value: f2(c.maxOverMin), hint: 'Mean over 40 queries of Dmax / Dmin — Beyer et al. show it tends to 1 as dimension grows' },
            { label: 'Relative contrast', value: f2(c.relativeContrast), hint: 'E[mean distance] / E[nearest distance] (He, Kumar & Chang)' },
            { label: 'Intrinsic dim. estimate', value: `${f1(c.lid)} (true ${trueDim})`, hint: 'MLE over each query’s 20 nearest neighbours, median over queries; it underestimates at high dimension' },
            { label: 'kd-tree exact search', value: `${fmtNum(detail.kdLeavesVisitedExact, 0)} / ${detail.kdLeaves} leaves`, hint: `${pct(detail.kdFracScannedExact)} of points distance-computed` },
            { label: 'Hyperplane tree exact', value: `${fmtNum(detail.hpLeavesVisitedExact, 0)} / ${detail.hpLeaves} leaves`, hint: `${pct(detail.hpFracScannedExact)} of points distance-computed` },
            { label: 'LSH candidates', value: here ? `${fmtNum(here.lshCandidates, 0)} of 1,500` : '…', hint: `Points sharing a bucket with the query in at least one of ${lshL} tables of ${lshK} hashes, averaged over the 40 queries` },
            { label: 'LSH exponent ρ', value: f2(rho), hint: 'ln(1/p(10th-NN distance)) / ln(1/p(mean distance)) for w = 4 × the 10th-NN distance; query cost grows like n^ρ' },
          ]}
        />
      }
      note={note}
      table={
        <table className="viz-table">
          <thead>
            <tr>
              <th>D</th>
              <th>Far ÷ near</th>
              <th>Rel. contrast</th>
              <th>ID estimate</th>
              <th>kd exact: points scanned</th>
              <th>Hyperplane exact: points scanned</th>
              <th>Recall@10 kd</th>
              <th>Hyperplane</th>
              <th>LSH</th>
              <th>HNSW</th>
              <th>Random</th>
            </tr>
          </thead>
          <tbody>
            {dims.map((d) => {
              const r = rows[d];
              return (
                <tr key={d}>
                  <td>{d}</td>
                  {r ? (
                    <>
                      <td>{f2(r.conc.maxOverMin)}</td>
                      <td>{f2(r.conc.relativeContrast)}</td>
                      <td>{f1(r.conc.lid)}</td>
                      <td>{pct(r.kdFracScannedExact)}</td>
                      <td>{pct(r.hpFracScannedExact)}</td>
                      <td>{f2(r.recallKd)}</td>
                      <td>{f2(r.recallHp)}</td>
                      <td>{f2(r.recallLsh)}</td>
                      <td>{f2(r.recallHnsw)}</td>
                      <td>{f2(r.recallRandom)}</td>
                    </>
                  ) : (
                    <td colSpan={10}>computing…</td>
                  )}
                </tr>
              );
            })}
          </tbody>
        </table>
      }
    >
      <TooltipHost>
        <div ref={ref}>
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.5rem 1rem', alignItems: 'flex-start' }}>
            <div style={{ width: leftW, maxWidth: '100%' }}>
              <div className="viz-controls">
                <Slider label="LSH hashes per table k" min={1} max={16} value={lshK} onChange={setLshK} />
                <Slider label="LSH tables L" min={1} max={40} value={lshL} onChange={setLshL} />
              </div>
              <p className="viz-sub">Bars: distance from each query to every point ÷ that query’s mean distance. Curve: probability that LSH makes a point at that distance a candidate.</p>
              <Histogram r={detail} k={dK} L={dL} width={leftW} />
            </div>
            <div style={{ width: rightW, maxWidth: '100%' }}>
              <div className="viz-controls">
                <Segmented
                  label="Tree"
                  value={tree}
                  onChange={setTree}
                  options={[
                    { value: 'kd', label: 'kd-tree' },
                    { value: 'hp', label: 'Hyperplane tree' },
                  ]}
                />
              </div>
              <p className="viz-sub">Cells: the tree’s leaves. Filled: opened by an exact 10-NN search for the first query. Outlined: the leaf that query falls in, opened first.</p>
              <LeafGrid
                leaves={tree === 'kd' ? detail.kdLeaves : detail.hpLeaves}
                visited={tree === 'kd' ? detail.kdVisited0 : detail.hpVisited0}
                color={tree === 'kd' ? SERIES_KD : SERIES_HP}
                width={rightW}
                label={tree === 'kd' ? 'kd-tree' : 'Hyperplane tree'}
              />
            </div>
          </div>
          <div className="viz-controls" style={{ marginTop: '0.75rem' }}>
            <Slider label="Distance budget" min={1} max={30} value={budgetPct} onChange={setBudgetPct} format={(v) => `${v}% of n`} />
          </div>
          <p className="viz-sub">
            Recall@10 when every index may compute only {pct(dBudget)} of the distances to data points, at each ambient dimension. LSH’s k × L projections and the hyperplane tree’s split
            tests are not charged.
          </p>
          <SweepChart rows={rows} dims={dims} currentD={detail.D} budgetFrac={dBudget} width={width} pending={pending} />
        </div>
      </TooltipHost>
    </VizPanel>
  );
}
