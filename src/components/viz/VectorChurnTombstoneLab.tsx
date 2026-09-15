import { useDeferredValue, useMemo, useState } from 'react';
import { VizPanel, Segmented, Slider, Choice, Check, Legend, Stats, Note, makeRng, fmtNum } from './Viz';

/**
 * How churn degrades a vector index — a small but real HNSW and IVF, run in 2D.
 *
 * HNSW (hnswlib's algorithm: level = floor(-ln U · 1/ln M), greedy descent on upper layers, a bounded
 * candidate heap on layer 0, the neighbour-selection heuristic, Mmax0 = 2M) with 300 clustered points,
 * M = 6, efConstruction = 40, k = 10, 40 fixed queries. Each churn step deletes 5% of the original points.
 * What each engine does with a deleted vector, as read in its source:
 *  - hnswlib markDelete: sets a bit, "does NOT really change the current graph". Search (searchBaseLayerST)
 *    still expands dead nodes but keeps going until it holds ef LIVE results; construction never picks a
 *    dead node as a neighbour. No reclamation (replace_deleted slot reuse is not modelled).
 *  - pgvector before VACUUM: the index does not know a row is dead, so dead entries count toward
 *    hnsw.ef_search and new inserts may link to them. VACUUM (hnswbulkdelete) re-runs a full neighbour
 *    search with efConstruction for every live element that points at a dead one (RepairGraph), writes
 *    back-links, then frees the dead elements' space for reuse.
 *  - DiskANN lazy_delete: tag erased, node stays; search_with_tags drops dead nodes from the L list after the
 *    search, so they occupy slots. consolidate_deletes (process_delete) replaces each dead out-neighbour with
 *    that neighbour's live out-neighbours and RobustPrunes (alpha 1.2) back to the degree bound. DiskANN's graph
 *    is single-layer: the lab searches only layer 0 from a fixed start point that is never deleted (DiskANN
 *    refuses to consolidate a deleted start node), and splices only layer 0.
 *  - Lucene: a delete clears a live-docs bit; search collects only accepted docs but traverses all. An update
 *    is delete + add, and the add lands in a NEW segment with its own graph, searched separately.
 *    Merge, 10.3: only a segment graph with no deletions may seed the merged graph; everything else is
 *    re-inserted. Merge, 10.5.1 (the Lucene PR 15003 algorithm of 10.4.0, whose reuse check never matched the default
 *    per-field codec until Lucene PR 16400): the largest graph with <= 40% deleted may seed it; nodes that kept < 85%
 *    of their neighbours are repaired by a search from the neighbours they kept, and diverse results are appended
 *    to the neighbours they kept (addDiverseNeighbors). The out-degree floor added later (Lucene PR 16552) is not
 *    modelled. Lucene's join-set shortcut for no-delete graphs and its level rebalancing are not modelled, so
 *    insert work at merge is overstated.
 *  - Every engine starts from the same hnswlib-style graph. pgvector's SelectNeighbors also keeps pruned
 *    connections to fill each list, so its real graph is denser and random deletes touch more neighbour lists.
 * Maintenance runs when the deleted fraction reaches the threshold (Qdrant's deleted_threshold, Lucene's
 * deletesPctAllowed are both 20% by default). CPU is counted as distance
 * computations, the unit every one of these algorithms spends. Index size counts stored nodes (live + dead).
 *
 * IVF: 400 build-time points in 4 clusters, nlist = 16 k-means centroids (k-means++ init, 25 Lloyd iterations),
 * then up to 600 inserted points assigned to the nearest FIXED centroid, as pgvector's ivfinsert does.
 * "Outside its fitted cell" = farther from its centroid than every build-time member of that list was.
 */

/* =================================================================== shared */

export type P = { x: number; y: number };
const dist = (a: P, b: P) => {
  const dx = a.x - b.x;
  const dy = a.y - b.y;
  return Math.sqrt(dx * dx + dy * dy);
};
const clamp01 = (v: number) => Math.min(0.98, Math.max(0.02, v));

function mix32(a: number) {
  a |= 0;
  a = Math.imul(a ^ (a >>> 16), 0x85ebca6b);
  a = Math.imul(a ^ (a >>> 13), 0xc2b2ae35);
  return (a ^ (a >>> 16)) >>> 0;
}
/** Deterministic uniform in (0,1) from two integers. */
export const hashU = (a: number, b: number) => (mix32(Math.imul(a, 73856093) ^ mix32(b + 40503)) + 0.5) / 4294967296;

function gauss(rng: () => number) {
  const u = Math.max(1e-6, rng());
  const v = rng();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

/** Min-heap of (key, value). */
class Heap {
  k: number[] = [];
  v: number[] = [];
  get size() {
    return this.k.length;
  }
  push(key: number, val: number) {
    const k = this.k;
    const v = this.v;
    let i = k.length;
    k.push(key);
    v.push(val);
    while (i > 0) {
      const p = (i - 1) >> 1;
      if (k[p] <= key) break;
      k[i] = k[p];
      v[i] = v[p];
      i = p;
    }
    k[i] = key;
    v[i] = val;
  }
  pop(): [number, number] {
    const k = this.k;
    const v = this.v;
    const topK = k[0];
    const topV = v[0];
    const lastK = k.pop() as number;
    const lastV = v.pop() as number;
    const n = k.length;
    if (n > 0) {
      let i = 0;
      for (;;) {
        const l = 2 * i + 1;
        if (l >= n) break;
        const r = l + 1;
        const c = r < n && k[r] < k[l] ? r : l;
        if (k[c] >= lastK) break;
        k[i] = k[c];
        v[i] = v[c];
        i = c;
      }
      k[i] = lastK;
      v[i] = lastV;
    }
    return [topK, topV];
  }
  peekKey() {
    return this.k[0];
  }
}

/* ===================================================================== HNSW */

export const HNSW = { N0: 300, M: 6, MMAX0: 12, EFC: 40, K: 10, STEPS: 12, PER_STEP: 15, QUERIES: 40 } as const;
const ML = 1 / Math.log(HNSW.M);
const mmaxFor = (layer: number) => (layer === 0 ? HNSW.MMAX0 : HNSW.M);

export type HNode = { id: number; p: P; level: number; dead: boolean; links: number[][] };
export type HGraph = { nodes: Map<number, HNode>; entry: number; maxLevel: number; name: string };
export type Cost = { d: number };

export type Engine = 'hnswlib' | 'pgvector' | 'diskann' | 'lucene103' | 'lucene104';
export type Workload = 'random' | 'region' | 'update';

export const ENGINES: Record<Engine, { label: string; searchSkipsDead: boolean; buildSkipsDead: boolean; newSegment: boolean; maint: string; maintVerb: string }> = {
  hnswlib: { label: 'hnswlib markDelete', searchSkipsDead: true, buildSkipsDead: true, newSegment: false, maint: 'none', maintVerb: 'nothing — hnswlib never reclaims a tombstone' },
  pgvector: { label: 'pgvector HNSW + VACUUM', searchSkipsDead: false, buildSkipsDead: false, newSegment: false, maint: 'VACUUM', maintVerb: 'VACUUM re-searches neighbours for every element that pointed at a dead one' },
  diskann: { label: 'DiskANN consolidate_deletes', searchSkipsDead: false, buildSkipsDead: false, newSegment: false, maint: 'consolidate', maintVerb: 'consolidate_deletes splices each dead node’s neighbours into the nodes that pointed at it' },
  lucene103: { label: 'Lucene 10.3 segment merge', searchSkipsDead: true, buildSkipsDead: true, newSegment: true, maint: 'merge', maintVerb: 'the merge rebuilds the graph (a segment with deletions cannot seed it)' },
  lucene104: { label: 'Lucene 10.5.1 segment merge', searchSkipsDead: true, buildSkipsDead: true, newSegment: true, maint: 'merge', maintVerb: 'the merge copies the surviving graph and repairs nodes that lost neighbours' },
};

const levelFor = (id: number) => Math.min(4, Math.floor(-Math.log(hashU(id, 1)) * ML));

const HNSW_CENTERS: P[] = [
  { x: 0.22, y: 0.27 },
  { x: 0.72, y: 0.22 },
  { x: 0.5, y: 0.56 },
  { x: 0.2, y: 0.76 },
  { x: 0.8, y: 0.74 },
];
export const REGION_CENTER: P = { x: 0.2, y: 0.76 };

export function hnswPoints(n: number, seed: number): P[] {
  const rng = makeRng(seed);
  const out: P[] = [];
  for (let i = 0; i < n; i++) {
    if (rng() < 0.2) out.push({ x: clamp01(rng()), y: clamp01(rng()) });
    else {
      const c = HNSW_CENTERS[Math.floor(rng() * HNSW_CENTERS.length)];
      out.push({ x: clamp01(c.x + gauss(rng) * 0.085), y: clamp01(c.y + gauss(rng) * 0.085) });
    }
  }
  return out;
}

const emptyGraph = (name: string): HGraph => ({ nodes: new Map(), entry: -1, maxLevel: -1, name });

type SearchOut = { res: { id: number; d: number }[]; visited: Set<number>; expanded: number[]; peak: number; deadExpanded: number };

/** hnswlib searchBaseLayer / searchBaseLayerST. skipDead=false is the bare-bone search where dead nodes fill the list. */
function searchLayer(g: HGraph, q: P, eps: number[], ef: number, layer: number, skipDead: boolean, cost: Cost, exclude = -1, trace = false, bare = false): SearchOut {
  const visited = new Set<number>();
  const expanded: number[] = [];
  const cand = new Heap();
  const top = new Heap(); // max-heap via negated keys
  let lower = Infinity;
  const counts = (n: HNode) => n.id !== exclude && !(skipDead && n.dead);
  for (const e of eps) {
    if (visited.has(e)) continue;
    const n = g.nodes.get(e);
    if (!n) continue;
    visited.add(e);
    const dd = dist(q, n.p);
    cost.d++;
    cand.push(dd, e);
    if (counts(n)) {
      top.push(-dd, e);
      if (top.size > ef) top.pop();
      lower = -top.peekKey();
    }
  }
  let peak = cand.size;
  let deadExpanded = 0;
  while (cand.size) {
    const cd = cand.peekKey();
    // Construction and hnswlib's filtered search stop only once ef results are held; the bare-bone search stops at the first worse candidate.
    if (cd > lower && (bare || top.size >= ef)) break;
    const [, cid] = cand.pop();
    const cn = g.nodes.get(cid) as HNode;
    if (cn.dead) deadExpanded++;
    if (trace) expanded.push(cid);
    const lst = cn.links[layer];
    if (!lst) continue;
    for (const nb of lst) {
      if (visited.has(nb)) continue;
      const nn = g.nodes.get(nb);
      if (!nn) continue;
      visited.add(nb);
      const dd = dist(q, nn.p);
      cost.d++;
      if (top.size < ef || dd < lower) {
        cand.push(dd, nb);
        if (cand.size > peak) peak = cand.size;
        if (counts(nn)) {
          top.push(-dd, nb);
          if (top.size > ef) top.pop();
        }
        if (top.size) lower = -top.peekKey();
      }
    }
  }
  const res: { id: number; d: number }[] = [];
  while (top.size) {
    const [k, v] = top.pop();
    res.push({ id: v, d: -k });
  }
  res.reverse();
  return { res, visited, expanded, peak, deadExpanded };
}

/** Greedy walk on layers top..bottom+1 (hnswlib does not check deletion marks here). */
function greedy(g: HGraph, q: P, from: number, top: number, bottomExclusive: number, cost: Cost) {
  let cur = from;
  let curD = dist(q, (g.nodes.get(cur) as HNode).p);
  cost.d++;
  for (let l = top; l > bottomExclusive; l--) {
    let changed = true;
    while (changed) {
      changed = false;
      const lst = (g.nodes.get(cur) as HNode).links[l] ?? [];
      for (const nb of lst) {
        const nn = g.nodes.get(nb);
        if (!nn) continue;
        const dd = dist(q, nn.p);
        cost.d++;
        if (dd < curD) {
          curD = dd;
          cur = nb;
          changed = true;
        }
      }
    }
  }
  return cur;
}

/** getNeighborsByHeuristic2: keep a candidate only if it is closer to the base than to every neighbour kept so far. */
function selectHeuristic(g: HGraph, cands: { id: number; d: number }[], m: number, cost: Cost) {
  if (cands.length <= m) return cands.map((c) => c.id);
  const out: HNode[] = [];
  for (const c of cands) {
    if (out.length >= m) break;
    const cn = g.nodes.get(c.id) as HNode;
    let ok = true;
    for (const s of out) {
      cost.d++;
      if (dist(cn.p, s.p) < c.d) {
        ok = false;
        break;
      }
    }
    if (ok) out.push(cn);
  }
  return out.map((n) => n.id);
}

function connect(g: HGraph, nbId: number, newId: number, layer: number, cost: Cost) {
  const nb = g.nodes.get(nbId);
  if (!nb || !nb.links[layer]) return;
  const lst = nb.links[layer];
  if (lst.includes(newId)) return;
  const mmax = mmaxFor(layer);
  if (lst.length < mmax) {
    lst.push(newId);
    return;
  }
  const cands = [...lst, newId]
    .filter((x) => g.nodes.has(x))
    .map((x) => {
      cost.d++;
      return { id: x, d: dist(nb.p, (g.nodes.get(x) as HNode).p) };
    })
    .sort((a, b) => a.d - b.d);
  nb.links[layer] = selectHeuristic(g, cands, mmax, cost);
}

export function insertNode(g: HGraph, id: number, p: P, skipDead: boolean, cost: Cost) {
  const level = levelFor(id);
  const node: HNode = { id, p, level, dead: false, links: Array.from({ length: level + 1 }, () => []) };
  g.nodes.set(id, node);
  if (g.entry < 0) {
    g.entry = id;
    g.maxLevel = level;
    return;
  }
  let cur = greedy(g, p, g.entry, g.maxLevel, level, cost);
  for (let l = Math.min(level, g.maxLevel); l >= 0; l--) {
    const { res } = searchLayer(g, p, [cur], HNSW.EFC, l, skipDead, cost, id);
    const chosen = selectHeuristic(g, res, HNSW.M, cost);
    node.links[l] = chosen;
    for (const nb of chosen) connect(g, nb, id, l, cost);
    if (res.length) cur = res[0].id;
  }
  if (level > g.maxLevel) {
    g.entry = id;
    g.maxLevel = level;
  }
}

function highestLive(g: HGraph) {
  let best = -1;
  let lvl = -1;
  for (const n of g.nodes.values()) if (!n.dead && (n.level > lvl || (n.level === lvl && n.id < best))) {
    best = n.id;
    lvl = n.level;
  }
  return [best, lvl] as const;
}

function dropDead(g: HGraph) {
  let removed = 0;
  for (const [id, n] of [...g.nodes]) if (n.dead) {
    g.nodes.delete(id);
    removed++;
  }
  for (const n of g.nodes.values()) for (let l = 0; l < n.links.length; l++) n.links[l] = n.links[l].filter((x) => g.nodes.has(x));
  if (!g.nodes.has(g.entry)) {
    const [e, lvl] = highestLive(g);
    g.entry = e;
    g.maxLevel = lvl;
  }
  return removed;
}

/** DiskANN process_delete on every layer, then release the dead slots. */
export function consolidate(g: HGraph, cost: Cost, flat = true) {
  const alpha = 1.2;
  for (const p of g.nodes.values()) {
    if (p.dead) continue;
    for (let l = 0; l <= (flat ? 0 : p.level); l++) {
      const adj = p.links[l];
      if (!adj.some((x) => (g.nodes.get(x) as HNode | undefined)?.dead)) continue;
      const set = new Set<number>();
      for (const n of adj) {
        const nn = g.nodes.get(n);
        if (!nn) continue;
        if (!nn.dead) set.add(n);
        else for (const j of nn.links[l] ?? []) if (j !== p.id && g.nodes.has(j) && !(g.nodes.get(j) as HNode).dead) set.add(j);
      }
      const range = mmaxFor(l);
      if (set.size <= range) {
        p.links[l] = [...set];
        continue;
      }
      let pool = [...set]
        .map((id) => {
          cost.d++;
          return { id, d: dist(p.p, (g.nodes.get(id) as HNode).p) };
        })
        .sort((a, b) => a.d - b.d);
      const out: number[] = [];
      while (pool.length && out.length < range) {
        const star = pool.shift() as { id: number; d: number };
        out.push(star.id);
        const sp = (g.nodes.get(star.id) as HNode).p;
        pool = pool.filter((c) => {
          cost.d++;
          return alpha * dist(sp, (g.nodes.get(c.id) as HNode).p) > c.d;
        });
      }
      p.links[l] = out;
    }
  }
  return dropDead(g);
}

/** pgvector hnswbulkdelete: RemoveHeapTids, RepairGraph (full neighbour search per affected element), MarkDeleted. */
export function vacuumRepair(g: HGraph, cost: Cost) {
  const dead = new Set<number>();
  for (const n of g.nodes.values()) if (n.dead) dead.add(n.id);
  if ((g.nodes.get(g.entry) as HNode).dead) {
    const [e, lvl] = highestLive(g);
    g.entry = e;
    g.maxLevel = lvl;
  }
  const affected = [...g.nodes.values()].filter((n) => !n.dead && n.id !== g.entry && n.links.some((lst) => lst.some((x) => dead.has(x))));
  for (const e of affected) {
    for (let l = 0; l <= e.level; l++) e.links[l] = [];
    let cur = greedy(g, e.p, g.entry, g.maxLevel, e.level, cost);
    for (let l = Math.min(e.level, g.maxLevel); l >= 0; l--) {
      // Dead elements help the search but do not count toward ef, and are removed before selection.
      const { res } = searchLayer(g, e.p, [cur], HNSW.EFC, l, true, cost, e.id);
      const chosen = selectHeuristic(g, res, mmaxFor(l), cost);
      e.links[l] = chosen;
      for (const nb of chosen) connect(g, nb, e.id, l, cost);
      if (res.length) cur = res[0].id;
    }
  }
  return { removed: dropDead(g), repaired: affected.length };
}

function cloneGraph(g: HGraph, name = g.name): HGraph {
  const nodes = new Map<number, HNode>();
  for (const [id, n] of g.nodes) nodes.set(id, { id, p: n.p, level: n.level, dead: n.dead, links: n.links.map((l) => l.slice()) });
  return { nodes, entry: g.entry, maxLevel: g.maxLevel, name };
}

/** Lucene merge of every segment graph into one. */
export function luceneMerge(segs: HGraph[], version: '10.3' | '10.4', cost: Cost) {
  const sized = segs
    .map((s) => {
      let dead = 0;
      for (const n of s.nodes.values()) if (n.dead) dead++;
      return { s, size: s.nodes.size, dead, live: s.nodes.size - dead };
    })
    .filter((x) => x.size > 0);
  const eligible = sized.filter((x) => (version === '10.3' ? x.dead === 0 : Math.floor((x.dead * 100) / x.size) <= 40)).sort((a, b) => b.live - a.live);
  const base = eligible[0];
  const out = emptyGraph('merged');
  let repaired = 0;
  let copied = 0;
  if (base) {
    // copyGraphStructure: surviving nodes keep surviving neighbours; flag nodes that kept < 85%.
    const disconnected: number[][] = [];
    for (const n of base.s.nodes.values()) {
      if (n.dead) continue;
      const links = n.links.map((lst, l) => {
        const kept = lst.filter((x) => base.s.nodes.has(x) && !(base.s.nodes.get(x) as HNode).dead);
        if (kept.length < lst.length * 0.85) (disconnected[l] ??= []).push(n.id);
        return kept;
      });
      out.nodes.set(n.id, { id: n.id, p: n.p, level: n.level, dead: false, links });
      copied++;
    }
    const [e, lvl] = highestLive(out);
    out.entry = e;
    out.maxLevel = lvl;
    for (let l = disconnected.length - 1; l >= 0; l--) {
      for (const id of disconnected[l] ?? []) {
        const n = out.nodes.get(id) as HNode;
        repaired++;
        const eps = n.links[l].length ? n.links[l].slice() : [greedy(out, n.p, out.entry, out.maxLevel, l, cost)];
        const { res } = searchLayer(out, n.p, eps, HNSW.EFC, l, true, cost, id);
        // addDiverseNeighbors with isLinkRepair: the kept neighbours stay; a candidate is appended, best first,
        // only if it is closer to this node than to every neighbour already in the list, up to maxConn.
        const lst = n.links[l];
        const added: number[] = [];
        for (const r of res) {
          if (lst.length >= mmaxFor(l)) break;
          if (r.id === id || lst.includes(r.id)) continue;
          const cp = (out.nodes.get(r.id) as HNode).p;
          let diverse = true;
          for (const s of lst) {
            cost.d++;
            if (dist(cp, (out.nodes.get(s) as HNode).p) <= r.d) {
              diverse = false;
              break;
            }
          }
          if (diverse) {
            lst.push(r.id);
            added.push(r.id);
          }
        }
        for (const x of added) connect(out, x, id, l, cost);
      }
    }
  }
  let inserted = 0;
  for (const x of sized) {
    if (base && x.s === base.s) continue;
    const ids = [...x.s.nodes.values()].filter((n) => !n.dead).sort((a, b) => a.id - b.id);
    for (const n of ids) {
      insertNode(out, n.id, n.p, true, cost);
      inserted++;
    }
  }
  return { graph: out, copied, repaired, inserted, seeded: !!base };
}

export type Snapshot = { step: number; segs: HGraph[]; event: null | { kind: string; cost: number; removed: number; detail: string } };
export type ChurnRun = { snaps: Snapshot[]; buildCost: number; ingestCost: number[]; maintCost: number[] };

let baseCache: { g: HGraph; cost: number; start: number } | null = null;
export function baseGraph() {
  if (!baseCache) {
    const pts = hnswPoints(HNSW.N0, 4242);
    const g = emptyGraph('main');
    const cost = { d: 0 };
    pts.forEach((p, i) => insertNode(g, i, p, true, cost));
    // DiskANN's fixed start point: the vector nearest the dataset mean (a medoid stand-in).
    const mean = { x: pts.reduce((a, q) => a + q.x, 0) / pts.length, y: pts.reduce((a, q) => a + q.y, 0) / pts.length };
    let start = 0;
    pts.forEach((q, i) => {
      if (dist(q, mean) < dist(pts[start], mean)) start = i;
    });
    baseCache = { g, cost: cost.d, start };
  }
  return baseCache;
}

export const hnswQueries = () => hnswPoints(HNSW.QUERIES, 9001);

export function deletionOrder(workload: Workload) {
  const { g, start } = baseGraph();
  const ids = [...g.nodes.keys()].filter((id) => id !== start);
  if (workload === 'region') return ids.sort((a, b) => dist((g.nodes.get(a) as HNode).p, REGION_CENTER) - dist((g.nodes.get(b) as HNode).p, REGION_CENTER) || a - b);
  return ids.sort((a, b) => hashU(a, 77) - hashU(b, 77));
}

export function deadFraction(segs: HGraph[]) {
  let dead = 0;
  let all = 0;
  for (const s of segs)
    for (const n of s.nodes.values()) {
      all++;
      if (n.dead) dead++;
    }
  return all ? dead / all : 0;
}

/** Replay the churn stream step by step for one engine. maintenance=false keeps every tombstone. */
export function runChurn(workload: Workload, engine: Engine, threshold: number, maintenance: boolean): ChurnRun {
  const cfg = ENGINES[engine];
  const base = baseGraph();
  let segs: HGraph[] = [cloneGraph(base.g)];
  const order = deletionOrder(workload);
  let nextId = HNSW.N0;
  const snaps: Snapshot[] = [{ step: 0, segs: segs.map((s) => cloneGraph(s)), event: null }];
  const ingestCost: number[] = [0];
  const maintCost: number[] = [0];
  for (let s = 1; s <= HNSW.STEPS; s++) {
    const ing = { d: 0 };
    for (let j = 0; j < HNSW.PER_STEP; j++) {
      const victim = order[(s - 1) * HNSW.PER_STEP + j];
      let old: HNode | undefined;
      for (const sg of segs) {
        const n = sg.nodes.get(victim);
        if (n) old = n;
      }
      if (!old) continue;
      old.dead = true;
      if (workload === 'update') {
        const np = { x: clamp01(old.p.x + (hashU(victim, 5) - 0.5) * 0.06), y: clamp01(old.p.y + (hashU(victim, 6) - 0.5) * 0.06) };
        let target = segs[0];
        if (cfg.newSegment) {
          if (segs.length < 2) segs.push(emptyGraph('new segment'));
          target = segs[1];
        }
        insertNode(target, nextId++, np, cfg.buildSkipsDead, ing);
      }
    }
    let event: Snapshot['event'] = null;
    const mc = { d: 0 };
    const mainDead = cfg.newSegment ? deadFraction([segs[0]]) : deadFraction(segs);
    if (maintenance && cfg.maint !== 'none' && mainDead >= threshold) {
      if (engine === 'diskann') {
        const removed = consolidate(segs[0], mc, true);
        event = { kind: 'consolidate', cost: mc.d, removed, detail: `${removed} dead nodes spliced out` };
      } else if (engine === 'pgvector') {
        const r = vacuumRepair(segs[0], mc);
        event = { kind: 'VACUUM', cost: mc.d, removed: r.removed, detail: `${r.repaired} elements re-searched, ${r.removed} dead elements freed` };
      } else {
        const before = segs.reduce((a, sg) => a + sg.nodes.size, 0);
        const r = luceneMerge(segs, engine === 'lucene103' ? '10.3' : '10.4', mc);
        segs = [r.graph];
        event = {
          kind: 'merge',
          cost: mc.d,
          removed: before - r.graph.nodes.size,
          detail: r.seeded ? `${r.copied} nodes copied from a seed graph, ${r.repaired} repaired, ${r.inserted} re-inserted` : `no graph could seed it: all ${r.inserted} live vectors re-inserted`,
        };
      }
    }
    ingestCost.push(ing.d);
    maintCost.push(mc.d);
    snaps.push({ step: s, segs: segs.map((sg) => cloneGraph(sg)), event });
  }
  return { snaps, buildCost: base.cost, ingestCost, maintCost };
}

export type QueryTrace = { found: number[]; truth: number[]; visited: Set<number>; expanded: Set<number>; deadExpanded: number; dist: number; peak: number };

export function searchIndex(segs: HGraph[], engine: Engine, ef: number, q: P, trace = false): QueryTrace {
  const cfg = ENGINES[engine];
  const k = HNSW.K;
  const cost = { d: 0 };
  let all: { id: number; d: number }[] = [];
  const visited = new Set<number>();
  const expanded = new Set<number>();
  let peak = 0;
  let deadExpanded = 0;
  const start = baseGraph().start;
  for (const g of segs) {
    if (g.entry < 0 || !g.nodes.size) continue;
    // DiskANN: one layer, always entered at the fixed start point. HNSW: greedy descent through the upper layers.
    const ep = engine === 'diskann' && g.nodes.has(start) ? start : greedy(g, q, g.entry, g.maxLevel, 0, cost);
    const r = searchLayer(g, q, [ep], Math.max(ef, k), 0, cfg.searchSkipsDead, cost, -1, trace, !cfg.searchSkipsDead);
    peak = Math.max(peak, r.peak);
    deadExpanded += r.deadExpanded;
    if (trace) {
      r.visited.forEach((v) => visited.add(v));
      r.expanded.forEach((v) => expanded.add(v));
    }
    // Results the engine can return: dead entries are dropped after the search (pgvector's heap check, DiskANN's tag check).
    all = all.concat(r.res.filter((x) => !(g.nodes.get(x.id) as HNode).dead).slice(0, k));
  }
  all.sort((a, b) => a.d - b.d);
  const found = all.slice(0, k).map((x) => x.id);
  const live: { id: number; d: number }[] = [];
  for (const g of segs) for (const n of g.nodes.values()) if (!n.dead) live.push({ id: n.id, d: dist(q, n.p) });
  live.sort((a, b) => a.d - b.d || a.id - b.id);
  const truth = live.slice(0, k).map((x) => x.id);
  return { found, truth, visited, expanded, deadExpanded, dist: cost.d, peak };
}

/** Live nodes a search could no longer reach if every dead node were removed without repair. */
export function strandedIfDropped(segs: HGraph[], engine: Engine = 'hnswlib') {
  const stranded = new Set<number>();
  const flat = engine === 'diskann';
  for (const g of segs) {
    let start = flat && g.nodes.has(baseGraph().start) ? baseGraph().start : g.entry;
    if (start < 0) continue;
    if ((g.nodes.get(start) as HNode).dead) start = highestLive(g)[0];
    if (start < 0) continue;
    const seen = new Set<number>([start]);
    const stack = [start];
    while (stack.length) {
      const n = g.nodes.get(stack.pop() as number) as HNode;
      for (const lst of flat ? n.links.slice(0, 1) : n.links)
        for (const x of lst) {
          const xn = g.nodes.get(x);
          if (!xn || xn.dead || seen.has(x)) continue;
          seen.add(x);
          stack.push(x);
        }
    }
    for (const n of g.nodes.values()) if (!n.dead && !seen.has(n.id)) stranded.add(n.id);
  }
  return stranded;
}

export type StepMetrics = { step: number; deletedPct: number; recall: number; dist: number; peak: number; deadHops: number; dead: number; stored: number; stranded: number; segments: number };

export function measureSnap(snap: Snapshot, engine: Engine, ef: number): StepMetrics {
  const qs = hnswQueries();
  let recall = 0;
  let d = 0;
  let peak = 0;
  let deadHops = 0;
  for (const q of qs) {
    const t = searchIndex(snap.segs, engine, ef, q);
    const truth = new Set(t.truth);
    recall += t.found.filter((x) => truth.has(x)).length / Math.max(1, t.truth.length);
    d += t.dist;
    peak += t.peak;
    deadHops += t.deadExpanded;
  }
  let dead = 0;
  let stored = 0;
  for (const g of snap.segs)
    for (const n of g.nodes.values()) {
      stored++;
      if (n.dead) dead++;
    }
  return {
    step: snap.step,
    deletedPct: (snap.step * HNSW.PER_STEP * 100) / HNSW.N0,
    recall: recall / qs.length,
    dist: d / qs.length,
    peak: peak / qs.length,
    deadHops: deadHops / qs.length,
    dead,
    stored,
    stranded: strandedIfDropped(snap.segs, engine).size,
    segments: snap.segs.filter((g) => g.nodes.size).length,
  };
}

/* ====================================================================== IVF */

export const IVF = { BUILD: 400, MAX_INSERT: 600, NLIST: 16, K: 10, QUERIES: 40 } as const;
export type Shift = 'none' | 'move' | 'new';
const IVF_CENTERS: P[] = [
  { x: 0.25, y: 0.28 },
  { x: 0.74, y: 0.27 },
  { x: 0.27, y: 0.73 },
  { x: 0.73, y: 0.72 },
];

export type IvfOpts = { center: P; sigma: number; frac: number; move: P; queriesFromNew: boolean };
export const IVF_DEFAULT: IvfOpts = { center: { x: 0.5, y: 0.92 }, sigma: 0.06, frac: 0.65, move: { x: 0.22, y: 0.2 }, queriesFromNew: true };

export function ivfPoints(shift: Shift, inserted: number, o: IvfOpts = IVF_DEFAULT) {
  const rng = makeRng(31);
  const build: P[] = [];
  for (let i = 0; i < IVF.BUILD; i++) {
    const c = IVF_CENTERS[Math.floor(rng() * 4)];
    build.push({ x: clamp01(c.x + gauss(rng) * 0.075), y: clamp01(c.y + gauss(rng) * 0.075) });
  }
  const rng2 = makeRng(shift === 'none' ? 57 : shift === 'move' ? 58 : 59);
  const added: P[] = [];
  for (let i = 0; i < IVF.MAX_INSERT; i++) {
    const t = (i + 1) / IVF.MAX_INSERT;
    const pick = rng2();
    const c = IVF_CENTERS[Math.floor(rng2() * 4)];
    const g1 = gauss(rng2);
    const g2 = gauss(rng2);
    let p: P;
    if (shift === 'new' && pick < o.frac) p = { x: o.center.x + g1 * o.sigma, y: o.center.y + g2 * o.sigma };
    else if (shift === 'move') p = { x: c.x + o.move.x * t + g1 * 0.075, y: c.y + o.move.y * t + g2 * 0.075 };
    else p = { x: c.x + g1 * 0.075, y: c.y + g2 * 0.075 };
    added.push({ x: clamp01(p.x), y: clamp01(p.y) });
  }
  return { build, added: added.slice(0, inserted) };
}

function nearest(cents: P[], p: P) {
  let best = 0;
  let bd = Infinity;
  for (let c = 0; c < cents.length; c++) {
    const d = dist(p, cents[c]);
    if (d < bd) {
      bd = d;
      best = c;
    }
  }
  return best;
}

function kmeansOnce(pts: P[], k: number, seed: number, iters: number) {
  const rng = makeRng(seed);
  const cents: P[] = [pts[Math.floor(rng() * pts.length)]];
  let cost = 0;
  const d2 = pts.map((p) => dist(p, cents[0]) ** 2);
  cost += pts.length;
  while (cents.length < k) {
    const total = d2.reduce((a, b) => a + b, 0);
    let r = rng() * total;
    let idx = 0;
    while (idx < pts.length - 1 && r > d2[idx]) r -= d2[idx++];
    const c = pts[idx];
    cents.push(c);
    for (let i = 0; i < pts.length; i++) d2[i] = Math.min(d2[i], dist(pts[i], c) ** 2);
    cost += pts.length;
  }
  const assign = new Int32Array(pts.length);
  for (let it = 0; it < iters; it++) {
    for (let i = 0; i < pts.length; i++) assign[i] = nearest(cents, pts[i]);
    cost += pts.length * k;
    const sx = new Float64Array(k);
    const sy = new Float64Array(k);
    const cnt = new Int32Array(k);
    for (let i = 0; i < pts.length; i++) {
      sx[assign[i]] += pts[i].x;
      sy[assign[i]] += pts[i].y;
      cnt[assign[i]]++;
    }
    for (let c = 0; c < k; c++) if (cnt[c]) cents[c] = { x: sx[c] / cnt[c], y: sy[c] / cnt[c] };
  }
  let inertia = 0;
  for (let i = 0; i < pts.length; i++) inertia += dist(pts[i], cents[nearest(cents, pts[i])]) ** 2;
  return { cents, cost, inertia };
}

/** k-means++ then Lloyd iterations; best of three seeds by inertia. Cost = distance computations. */
export function kmeans(pts: P[], k: number, seed: number, iters = 25) {
  let best = kmeansOnce(pts, k, seed, iters);
  let cost = best.cost;
  for (let r = 1; r < 3; r++) {
    const t = kmeansOnce(pts, k, seed + r * 101, iters);
    cost += t.cost;
    if (t.inertia < best.inertia) best = t;
  }
  return { cents: best.cents, cost };
}

export function ivfQueries(pts: P[]) {
  const out: P[] = [];
  for (let i = 0; i < IVF.QUERIES; i++) {
    const src = pts[Math.floor(hashU(i, pts.length) * pts.length)];
    out.push({ x: src.x + (hashU(i, 3) - 0.5) * 0.03, y: src.y + (hashU(i, 4) - 0.5) * 0.03 });
  }
  return out;
}

export function ivfTruth(pts: P[], q: P) {
  return pts
    .map((p, id) => ({ id, d: dist(q, p) }))
    .sort((a, b) => a.d - b.d || a.id - b.id)
    .slice(0, IVF.K)
    .map((x) => x.id);
}

export function ivfSearch(cents: P[], lists: number[][], pts: P[], q: P, nprobe: number, truth: number[]) {
  const order = cents.map((c, i) => ({ i, d: dist(q, c) })).sort((a, b) => a.d - b.d);
  const probed = order.slice(0, nprobe).map((o) => o.i);
  const cand: { id: number; d: number }[] = [];
  for (const l of probed) for (const id of lists[l]) cand.push({ id, d: dist(q, pts[id]) });
  cand.sort((a, b) => a.d - b.d || a.id - b.id);
  const found = cand.slice(0, IVF.K).map((x) => x.id);
  const t = new Set(truth);
  return { probed, found, truth, scanned: cand.length, recall: found.filter((x) => t.has(x)).length / IVF.K };
}

export function ivfState(shift: Shift, inserted: number, recluster: boolean, o: IvfOpts = IVF_DEFAULT) {
  const { build, added } = ivfPoints(shift, inserted, o);
  const pts = [...build, ...added];
  const fit = kmeans(build, IVF.NLIST, 7);
  const refit = kmeans(pts, IVF.NLIST, 7);
  const cents = recluster ? refit.cents : fit.cents;
  const lists: number[][] = Array.from({ length: IVF.NLIST }, () => []);
  pts.forEach((p, i) => lists[nearest(cents, p)].push(i));
  // Fitted radius: the farthest member each list had when its centroids were trained.
  const trainSet = recluster ? pts.length : build.length;
  const radius = new Array(IVF.NLIST).fill(0);
  for (let i = 0; i < trainSet; i++) {
    const l = nearest(cents, pts[i]);
    radius[l] = Math.max(radius[l], dist(pts[i], cents[l]));
  }
  const outOfFit = new Set<number>();
  for (let i = trainSet; i < pts.length; i++) {
    const l = nearest(cents, pts[i]);
    if (dist(pts[i], cents[l]) > radius[l]) outOfFit.add(i);
  }
  const queries = ivfQueries(o.queriesFromNew && added.length ? added : pts);
  const truths = queries.map((q) => ivfTruth(pts, q));
  const curve = (cs: P[]) => {
    const ls: number[][] = Array.from({ length: IVF.NLIST }, () => []);
    pts.forEach((p, i) => ls[nearest(cs, p)].push(i));
    return Array.from({ length: IVF.NLIST }, (_, j) => {
      let rec = 0;
      let scanned = 0;
      queries.forEach((q, qi) => {
        const r = ivfSearch(cs, ls, pts, q, j + 1, truths[qi]);
        rec += r.recall;
        scanned += r.scanned;
      });
      return { nprobe: j + 1, recall: rec / queries.length, scanned: scanned / queries.length };
    });
  };
  const fittedCurve = curve(fit.cents);
  const refitCurve = curve(refit.cents);
  const sizes = lists.map((l) => l.length);
  return { build, added, pts, cents, lists, radius, outOfFit, queries, fittedCurve, refitCurve, sizes, reclusterCost: refit.cost + pts.length * IVF.NLIST };
}

export const nprobeFor = (curve: { nprobe: number; recall: number }[], target: number) => curve.find((c) => c.recall >= target)?.nprobe ?? null;
export const scanFor = (curve: { recall: number; scanned: number }[], target: number) => curve.find((c) => c.recall >= target)?.scanned ?? null;

/* ======================================================================= UI */

type Tab = 'hnsw' | 'ivf';
const TABS = [
  { value: 'hnsw' as const, label: 'HNSW under deletes' },
  { value: 'ivf' as const, label: 'IVF under drift' },
];

const PAD = 12;
const SIDE = 400;
const sx = (v: number) => PAD + v * (SIDE - 2 * PAD);

function toData(e: React.MouseEvent<SVGSVGElement>): P {
  const r = e.currentTarget.getBoundingClientRect();
  const vx = ((e.clientX - r.left) / r.width) * SIDE;
  const vy = ((e.clientY - r.top) / r.height) * SIDE;
  return { x: Math.min(0.99, Math.max(0.01, (vx - PAD) / (SIDE - 2 * PAD))), y: Math.min(0.99, Math.max(0.01, (vy - PAD) / (SIDE - 2 * PAD))) };
}

function Star({ p, label }: { p: P; label: string }) {
  const x = sx(p.x);
  const y = sx(p.y);
  return (
    <g pointerEvents="none">
      <path d={`M${x - 7},${y} L${x + 7},${y} M${x},${y - 7} L${x},${y + 7} M${x - 5},${y - 5} L${x + 5},${y + 5} M${x - 5},${y + 5} L${x + 5},${y - 5}`} stroke="var(--viz-ink)" strokeWidth={2} />
      <text x={Math.min(SIDE - 40, x + 9)} y={Math.max(12, y - 8)} fontSize={11} fill="var(--viz-ink)" stroke="var(--viz-surface)" strokeWidth={3} paintOrder="stroke">
        {label}
      </text>
    </g>
  );
}

type Series = { key: string; color: string; dashed?: boolean; label: string; values: number[] };

/** Small line chart with a shared x axis (step index 0..STEPS or arbitrary xs). */
function MiniChart({
  title,
  xs,
  series,
  xLabel,
  xFmt,
  yMin,
  yMax,
  yFmt,
  cursor,
  markers,
  ariaLabel,
  pointLabels,
}: {
  title: string;
  xs: number[][];
  series: Series[];
  xLabel: string;
  xFmt: (v: number) => string;
  yMin: number;
  yMax: number;
  yFmt: (v: number) => string;
  cursor?: { x: number; y?: number; seriesIndex?: number };
  markers?: { x: number; label: string }[];
  ariaLabel: string;
  pointLabels?: string[][];
}) {
  const W = 290;
  const H = 180;
  const L = 40;
  const R = 34;
  const T = 26;
  const B = 34;
  const allX = xs.flat();
  const x0 = Math.min(...allX);
  const x1 = Math.max(...allX, x0 + 1e-9);
  const X = (v: number) => L + ((v - x0) / (x1 - x0)) * (W - L - R);
  const Y = (v: number) => T + (1 - (v - yMin) / (yMax - yMin || 1)) * (H - T - B);
  const ticks = [yMin, (yMin + yMax) / 2, yMax];
  // Direct end labels, pushed apart when the series end at nearly the same value.
  const endLabelY = series.map((s, si) => Y(Math.min(yMax, Math.max(yMin, s.values[s.values.length - 1]))) + 3.5 + 0 * si);
  if (endLabelY.length === 2 && Math.abs(endLabelY[0] - endLabelY[1]) < 12) {
    const mid = (endLabelY[0] + endLabelY[1]) / 2;
    const up = series[0].values[series[0].values.length - 1] >= series[1].values[series[1].values.length - 1] ? 0 : 1;
    endLabelY[up] = mid - 6;
    endLabelY[1 - up] = mid + 6;
  }
  const xt = [x0, (x0 + x1) / 2, x1];
  return (
    <svg viewBox={`0 0 ${W} ${H}`} width={W} height={H} role="img" aria-label={ariaLabel}>
      <text x={L} y={14} fontSize={11.5} fill="var(--viz-ink)">
        {title}
      </text>
      {ticks.map((t) => (
        <g key={t}>
          <line x1={L} x2={W - R} y1={Y(t)} y2={Y(t)} stroke="var(--viz-grid)" />
          <text x={L - 5} y={Y(t) + 3.5} fontSize={10} textAnchor="end" fill="var(--viz-ink-2)">
            {yFmt(t)}
          </text>
        </g>
      ))}
      <line x1={L} x2={W - R} y1={H - B} y2={H - B} stroke="var(--viz-axis)" />
      {xt.map((t) => (
        <text key={t} x={X(t)} y={H - B + 13} fontSize={10} textAnchor="middle" fill="var(--viz-ink-2)">
          {xFmt(t)}
        </text>
      ))}
      <text x={(L + W - R) / 2} y={H - 4} fontSize={10} textAnchor="middle" fill="var(--viz-ink-2)">
        {xLabel}
      </text>
      {markers?.map((m, i) => (
        <g key={i}>
          <title>{m.label}</title>
          <line x1={X(m.x)} x2={X(m.x)} y1={T} y2={H - B} stroke="var(--viz-ink-2)" strokeDasharray="2 3" strokeOpacity={0.7} />
          <path d={`M${X(m.x) - 4},${T - 7} L${X(m.x) + 4},${T - 7} L${X(m.x)},${T - 1} Z`} fill="var(--viz-ink-2)" />
        </g>
      ))}
      {cursor ? <line x1={X(cursor.x)} x2={X(cursor.x)} y1={T} y2={H - B} stroke="var(--viz-ink-2)" strokeWidth={1.2} /> : null}
      {series.map((s, si) => (
        <g key={s.key}>
          <polyline
            points={s.values.map((v, i) => `${X(xs[si][i])},${Y(Math.min(yMax, Math.max(yMin, v)))}`).join(' ')}
            fill="none"
            stroke={s.color}
            strokeWidth={2}
            strokeDasharray={s.dashed ? '5 3' : undefined}
          />
          {s.values.map((v, i) => (
            <circle key={i} cx={X(xs[si][i])} cy={Y(Math.min(yMax, Math.max(yMin, v)))} r={cursor && cursor.seriesIndex === si && Math.abs(xs[si][i] - cursor.x) < 1e-9 ? 4.5 : 2.2} fill={s.color} />
          ))}
          {s.label ? (
            <text x={X(xs[si][s.values.length - 1]) + 4} y={endLabelY[si]} fontSize={9.5} fill="var(--viz-ink-2)">
              {s.label}
            </text>
          ) : null}
          {pointLabels?.[si]?.map((lab, i) =>
            lab ? (
              <text key={i} x={X(xs[si][i]) + 4} y={Y(Math.min(yMax, Math.max(yMin, s.values[i]))) + 13} fontSize={9} fill="var(--viz-ink-2)">
                {lab}
              </text>
            ) : null,
          )}
        </g>
      ))}
    </svg>
  );
}

const pct = (v: number, d = 0) => `${fmtNum(v * 100, d)}%`;

/** Small bounded memo so switching tabs or revisiting a setting does not recompute a whole churn run. */
const memo = new Map<string, unknown>();
function cached<T>(key: string, fn: () => T): T {
  if (memo.has(key)) return memo.get(key) as T;
  const v = fn();
  memo.set(key, v);
  if (memo.size > 64) memo.delete(memo.keys().next().value as string);
  return v;
}

export function HnswChurnPanel({ tabControl }: { tabControl: React.ReactNode }) {
  const [engine, setEngine] = useState<Engine>('hnswlib');
  const [workload, setWorkload] = useState<Workload>('region');
  const [deletedPct, setDeletedPct] = useState(40);
  const [ef, setEf] = useState(20);
  const [threshold, setThreshold] = useState(20);
  const [showStranded, setShowStranded] = useState(false);
  const [query, setQuery] = useState<P>({ x: 0.27, y: 0.7 });

  const cfg = ENGINES[engine];
  const hasMaint = cfg.maint !== 'none';
  const efD = useDeferredValue(ef);
  const thrD = useDeferredValue(threshold);
  const step = Math.round((deletedPct / 100) * HNSW.N0) / HNSW.PER_STEP;

  const keyA = `${workload}|${engine}|${thrD}`;
  const keyB = `${workload}|${engine}|never`;
  const runA = useMemo(() => cached(`run|${keyA}`, () => runChurn(workload, engine, thrD / 100, true)), [keyA, workload, engine, thrD]);
  const runB = useMemo(() => (hasMaint ? cached(`run|${keyB}`, () => runChurn(workload, engine, 1, false)) : runA), [keyB, workload, engine, hasMaint, runA]);
  const seriesA = useMemo(() => cached(`m|${keyA}|${efD}`, () => runA.snaps.map((sn) => measureSnap(sn, engine, efD))), [keyA, runA, engine, efD]);
  const seriesB = useMemo(() => (hasMaint ? cached(`m|${keyB}|${efD}`, () => runB.snaps.map((sn) => measureSnap(sn, engine, efD))) : seriesA), [keyB, runB, engine, efD, hasMaint, seriesA]);

  const snap = runA.snaps[step];
  const cur = seriesA[step];
  const trace = useMemo(() => searchIndex(snap.segs, engine, efD, query, true), [snap, engine, efD, query]);
  const stranded = useMemo(() => (showStranded ? strandedIfDropped(snap.segs, engine) : new Set<number>()), [snap, engine, showStranded]);

  const maintSoFar = runA.maintCost.slice(1, step + 1).reduce((a, b) => a + b, 0);
  const ingestSoFar = runA.ingestCost.slice(1, step + 1).reduce((a, b) => a + b, 0);
  const events = runA.snaps.filter((sn) => sn.event && sn.step <= HNSW.STEPS).map((sn) => ({ x: (sn.step * HNSW.PER_STEP * 100) / HNSW.N0, label: sn.event?.kind === 'consolidate' ? 'splice' : (sn.event?.kind ?? '') }));
  const lastEvent = [...runA.snaps.slice(0, step + 1)].reverse().find((sn) => sn.event);
  const start = baseGraph().start;
  const flat = engine === 'diskann';

  const found = new Set(trace.found);
  const truth = new Set(trace.truth);

  const xsSteps = seriesA.map((r) => r.deletedPct);
  const minRecall = Math.min(...seriesA.map((r) => r.recall), ...seriesB.map((r) => r.recall));
  const yMinRecall = Math.min(0.8, Math.floor(minRecall * 10) / 10);
  const maxDist = Math.max(...seriesA.map((r) => r.dist), ...seriesB.map((r) => r.dist));

  // Edges, drawn once per pair, layer 0 only.
  const edges: { a: P; b: P; kind: 'live' | 'dead' | 'seg' }[] = [];
  snap.segs.forEach((g, si) => {
    const seen = new Set<string>();
    for (const n of g.nodes.values()) {
      for (const j of n.links[0]) {
        const m = g.nodes.get(j);
        if (!m) continue;
        const key = n.id < j ? `${n.id}-${j}` : `${j}-${n.id}`;
        if (seen.has(key)) continue;
        seen.add(key);
        edges.push({ a: n.p, b: m.p, kind: n.dead || m.dead ? 'dead' : si > 0 ? 'seg' : 'live' });
      }
    }
  });

  const tombCount = cur.dead;
  const baseDist = seriesA[0].dist;

  const noteText = (() => {
    const at = `At ${deletedPct}% of the original vectors deleted`;
    if (engine === 'hnswlib' || ((engine === 'lucene103' || engine === 'lucene104') && !snap.event)) {
      const segTxt = cur.segments > 1 ? ` The updated vectors went into a second segment with its own graph, so every query searches ${cur.segments} graphs.` : '';
      return (
        <>
          <strong>
            {at}, {fmtNum(tombCount)} tombstones still sit in the graph as routing hops.
          </strong>{' '}
          Queries now compute {fmtNum(cur.dist, 0)} distances instead of {fmtNum(baseDist, 0)} and expand {fmtNum(cur.deadHops, 1)} dead nodes each, because the search keeps going until it holds ef live results — recall@10 is {cur.recall.toFixed(2)}.{segTxt}{' '}
          {cur.stranded > 0 ? `Removing the tombstones without repair would strand ${cur.stranded} live nodes.` : ''}
          {!hasMaint ? ' hnswlib never reclaims them: only a rebuild (or reusing slots with replace_deleted) does.' : ''}
        </>
      );
    }
    if (snap.event) {
      return (
        <>
          <strong>
            {at}, the dead fraction crossed {threshold}% and {cfg.maint} ran: {snap.event.detail}.
          </strong>{' '}
          It cost {fmtNum(snap.event.cost)} distance computations ({fmtNum(snap.event.cost / runA.buildCost, 2)}× building the original graph). Recall@10 is now {cur.recall.toFixed(2)}, versus {seriesB[step].recall.toFixed(2)} if nothing had run.
        </>
      );
    }
    return (
      <>
        <strong>
          {at}, {fmtNum(tombCount)} dead entries are still in the graph and count toward the {efD}-entry result list.
        </strong>{' '}
        The search stops at its usual size, then the dead entries are thrown away, so recall@10 is {cur.recall.toFixed(2)}
        {lastEvent ? ` (the last ${cfg.maint} ran at ${(lastEvent.step * HNSW.PER_STEP * 100) / HNSW.N0}%)` : ''}. Distance computations stay at {fmtNum(cur.dist, 0)} per query: this engine does not pay for tombstones in CPU, it pays in results.
      </>
    );
  })();

  return (
    <VizPanel
      title="Deleting from a live HNSW graph"
      subtitle="Each step deletes 5% of the original 300 vectors. Tombstones stay linked until the engine’s maintenance runs. Click the graph to move the query."
      controls={
        <>
          {tabControl}
          <Choice label="Engine" value={engine} onChange={setEngine} options={(Object.keys(ENGINES) as Engine[]).map((k) => ({ value: k, label: ENGINES[k].label }))} />
          <Choice
            label="Workload"
            value={workload}
            onChange={setWorkload}
            options={[
              { value: 'region', label: 'Delete one region of the space' },
              { value: 'random', label: 'Random deletes' },
              { value: 'update', label: 'Updates: delete + insert nearby' },
            ]}
          />
          <Slider label="Deleted so far" min={0} max={60} step={5} value={deletedPct} onChange={setDeletedPct} format={(v) => `${v}%`} />
          <Slider label="efSearch" min={10} max={60} step={5} value={ef} onChange={setEf} />
          <Slider label="Maintenance at dead fraction" min={10} max={50} step={5} value={threshold} onChange={setThreshold} format={(v) => (hasMaint ? `${v}%` : 'n/a')} disabled={!hasMaint} />
          <Check label="Mark nodes stranded if tombstones were dropped" checked={showStranded} onChange={setShowStranded} />
        </>
      }
      legend={
        <Legend
          items={[
            { label: 'Live vector', color: 'var(--viz-1)', shape: 'dot' },
            ...(cfg.newSegment && workload === 'update' ? [{ label: 'Vector in the new segment', color: 'var(--viz-2)' }] : []),
            { label: 'Tombstone (and its links)', color: 'var(--viz-stale)', shape: 'dot' as const },
            { label: 'Distance computed by this query', color: 'var(--viz-4)', shape: 'dot' as const },
            { label: 'True neighbour found', color: 'var(--viz-good)', shape: 'dot' as const },
            { label: 'True neighbour missed', color: 'var(--viz-critical)', shape: 'dot' as const },
            ...(showStranded ? [{ label: 'Stranded without tombstones', color: 'var(--viz-5)', shape: 'dot' as const }] : []),
            { label: hasMaint ? `Chart “runs”: ${cfg.maint} at ${threshold}% dead` : 'Chart: this engine', color: 'var(--viz-7)', shape: 'line' as const },
            ...(hasMaint
              ? [
                  { label: `Chart “never”: ${cfg.maint} never runs`, color: 'var(--viz-3)', shape: 'line' as const },
                  { label: `Chart ▼ dotted: ${cfg.maint} ran`, color: 'var(--viz-ink-2)', shape: 'line' as const },
                ]
              : []),
          ]}
        />
      }
      stats={
        <Stats
          items={[
            { label: 'Tombstones in the index', value: `${fmtNum(cur.dead)} of ${fmtNum(cur.stored)}`, hint: 'Stored nodes, live and dead. A tombstone keeps its vector and its links.' },
            { label: 'recall@10', value: cur.recall.toFixed(2), hint: `Mean over 40 fixed queries at efSearch ${efD}, against exact search over the live vectors.` },
            { label: 'Distance computations / query', value: `${fmtNum(cur.dist, 0)}`, hint: `${fmtNum(baseDist, 0)} before any delete.` },
            { label: 'Candidate heap peak', value: fmtNum(cur.peak, 1), hint: 'Largest size the candidate queue reached on layer 0, averaged over the queries.' },
            { label: 'Dead nodes expanded / query', value: fmtNum(cur.deadHops, 1), hint: 'Tombstones whose neighbour lists the search read in order to route.' },
            { label: 'Stranded if tombstones were dropped', value: fmtNum(cur.stranded), hint: 'Live nodes no longer reachable from the entry point if every dead node were removed without repairing links. With no dead nodes left, these are live nodes the graph already cannot reach.' },
            {
              label: 'Maintenance CPU so far',
              value: hasMaint ? `${fmtNum(maintSoFar)} (${fmtNum(maintSoFar / runA.buildCost, 2)}× build)` : 'none',
              hint: `Distance computations spent by ${cfg.maint === 'none' ? 'maintenance' : cfg.maint}. Building the original 300-node graph took ${fmtNum(runA.buildCost)}. Inserts in this run so far took ${fmtNum(ingestSoFar)}.`,
            },
          ]}
        />
      }
      note={<Note>{noteText}</Note>}
      table={
        <table className="viz-table">
          <thead>
            <tr>
              <th>Deleted</th>
              <th>Tombstones / stored</th>
              <th>recall@10{hasMaint ? ` (with ${cfg.maint})` : ''}</th>
              {hasMaint ? <th>recall@10 (never runs)</th> : null}
              <th>Distances / query</th>
              {hasMaint ? <th>Distances / query (never runs)</th> : null}
              <th>Maintenance at this step</th>
            </tr>
          </thead>
          <tbody>
            {seriesA.map((r, i) => (
              <tr key={r.step}>
                <td>{fmtNum(r.deletedPct)}%</td>
                <td>
                  {fmtNum(r.dead)} / {fmtNum(r.stored)}
                </td>
                <td>{r.recall.toFixed(2)}</td>
                {hasMaint ? <td>{seriesB[i].recall.toFixed(2)}</td> : null}
                <td>{fmtNum(r.dist, 0)}</td>
                {hasMaint ? <td>{fmtNum(seriesB[i].dist, 0)}</td> : null}
                <td>{runA.snaps[i].event ? `${runA.snaps[i].event?.detail}; ${fmtNum(runA.snaps[i].event?.cost ?? 0)} distance computations` : '—'}</td>
              </tr>
            ))}
          </tbody>
        </table>
      }
    >
      <div>
        <svg
          viewBox={`0 0 ${SIDE} ${SIDE}`}
          width={SIDE}
          height={SIDE}
          role="img"
          aria-label={`HNSW graph at ${deletedPct}% deleted: ${cur.dead} tombstones among ${cur.stored} stored nodes; the query computed ${trace.dist} distances`}
          onClick={(e) => setQuery(toData(e))}
          style={{ cursor: 'crosshair', width: '100%', maxWidth: 520, margin: '0 auto' }}
        >
          <rect x={0.5} y={0.5} width={SIDE - 1} height={SIDE - 1} fill="var(--viz-surface)" stroke="var(--viz-border)" />
          {edges.map((e, i) => (
            <line
              key={i}
              x1={sx(e.a.x)}
              y1={sx(e.a.y)}
              x2={sx(e.b.x)}
              y2={sx(e.b.y)}
              stroke={e.kind === 'dead' ? 'var(--viz-stale)' : e.kind === 'seg' ? 'var(--viz-2)' : 'var(--viz-ink-muted)'}
              strokeOpacity={e.kind === 'dead' ? 0.75 : e.kind === 'seg' ? 0.6 : 0.3}
              strokeDasharray={e.kind === 'dead' ? '3 2' : undefined}
              strokeWidth={e.kind === 'live' ? 0.8 : 1}
            />
          ))}
          {snap.segs.map((g, si) =>
            [...g.nodes.values()].map((n) => {
              const x = sx(n.p.x);
              const y = sx(n.p.y);
              return (
                <g key={`${si}-${n.id}`} pointerEvents="none">
                  {trace.visited.has(n.id) ? <circle cx={x} cy={y} r={6} fill="var(--viz-4)" fillOpacity={0.55} /> : null}
                  {n.dead ? (
                    <>
                      <circle cx={x} cy={y} r={3.2} fill="var(--viz-surface)" stroke="var(--viz-stale)" strokeWidth={1.3} />
                      <path d={`M${x - 2},${y - 2} L${x + 2},${y + 2} M${x - 2},${y + 2} L${x + 2},${y - 2}`} stroke="var(--viz-stale)" strokeWidth={1} />
                    </>
                  ) : si > 0 ? (
                    <rect x={x - 3} y={y - 3} width={6} height={6} fill="var(--viz-2)" />
                  ) : (
                    <circle cx={x} cy={y} r={3} fill="var(--viz-1)" />
                  )}
                  {truth.has(n.id) ? <circle cx={x} cy={y} r={8} fill="none" stroke={found.has(n.id) ? 'var(--viz-good)' : 'var(--viz-critical)'} strokeWidth={2} strokeDasharray={found.has(n.id) ? undefined : '3 2'} /> : null}
                  {stranded.has(n.id) ? <circle cx={x} cy={y} r={6.5} fill="none" stroke="var(--viz-5)" strokeWidth={2.2} /> : null}
                </g>
              );
            }),
          )}
          {(() => {
            const g = snap.segs[0];
            const id = flat && g.nodes.has(start) ? start : g.entry;
            const n = g.nodes.get(id);
            if (!n) return null;
            return (
              <text x={Math.min(SIDE - 90, sx(n.p.x) + 8)} y={sx(n.p.y) + 16} fontSize={10.5} fill="var(--viz-ink)" stroke="var(--viz-surface)" strokeWidth={3} paintOrder="stroke" pointerEvents="none">
                {flat ? 'start point' : `entry (layer ${n.level})`}
              </text>
            );
          })()}
          <Star p={query} label="query" />
        </svg>
        <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.5rem', justifyContent: 'center', marginTop: '0.5rem' }}>
          <MiniChart
            title="recall@10 vs deletions applied"
            xs={hasMaint ? [xsSteps, xsSteps] : [xsSteps]}
            series={[
              { key: 'a', color: 'var(--viz-7)', label: hasMaint ? 'runs' : '', values: seriesA.map((r) => r.recall) },
              ...(hasMaint ? [{ key: 'b', color: 'var(--viz-3)', dashed: true, label: 'never', values: seriesB.map((r) => r.recall) }] : []),
            ]}
            xLabel="% of original vectors deleted"
            xFmt={(v) => `${fmtNum(v)}%`}
            yMin={yMinRecall}
            yMax={1}
            yFmt={(v) => v.toFixed(2)}
            cursor={{ x: deletedPct, seriesIndex: 0 }}
            markers={events}
            ariaLabel={`recall at 10 against deletions: ${cur.recall.toFixed(2)} at ${deletedPct}%`}
          />
          <MiniChart
            title="distance computations per query"
            xs={hasMaint ? [xsSteps, xsSteps] : [xsSteps]}
            series={[
              { key: 'a', color: 'var(--viz-7)', label: hasMaint ? 'runs' : '', values: seriesA.map((r) => r.dist) },
              ...(hasMaint ? [{ key: 'b', color: 'var(--viz-3)', dashed: true, label: 'never', values: seriesB.map((r) => r.dist) }] : []),
            ]}
            xLabel="% of original vectors deleted"
            xFmt={(v) => `${fmtNum(v)}%`}
            yMin={0}
            yMax={Math.ceil(maxDist / 20) * 20}
            yFmt={(v) => fmtNum(v)}
            cursor={{ x: deletedPct, seriesIndex: 0 }}
            markers={events}
            ariaLabel={`distance computations per query: ${fmtNum(cur.dist, 0)} at ${deletedPct}%`}
          />
        </div>
      </div>
    </VizPanel>
  );
}

/* ------------------------------------------------------------------ Voronoi */

function clipHalfPlane(poly: P[], a: P, b: P) {
  // keep points closer to a than to b
  const mx = (a.x + b.x) / 2;
  const my = (a.y + b.y) / 2;
  const nx = b.x - a.x;
  const ny = b.y - a.y;
  const side = (p: P) => (p.x - mx) * nx + (p.y - my) * ny;
  const out: P[] = [];
  for (let i = 0; i < poly.length; i++) {
    const p = poly[i];
    const q = poly[(i + 1) % poly.length];
    const sp = side(p);
    const sq = side(q);
    if (sp <= 0) out.push(p);
    if ((sp < 0 && sq > 0) || (sp > 0 && sq < 0)) {
      const t = sp / (sp - sq);
      out.push({ x: p.x + t * (q.x - p.x), y: p.y + t * (q.y - p.y) });
    }
  }
  return out;
}

export function voronoiCells(cents: P[]) {
  return cents.map((c, i) => {
    let poly: P[] = [
      { x: 0, y: 0 },
      { x: 1, y: 0 },
      { x: 1, y: 1 },
      { x: 0, y: 1 },
    ];
    cents.forEach((o, j) => {
      if (j !== i && poly.length) poly = clipHalfPlane(poly, c, o);
    });
    return poly;
  });
}

export function IvfDriftPanel({ tabControl }: { tabControl: React.ReactNode }) {
  const [shift, setShift] = useState<Shift>('new');
  const [inserted, setInserted] = useState(600);
  const [nprobe, setNprobe] = useState(1);
  const [recluster, setRecluster] = useState(false);
  const [query, setQuery] = useState<P>({ x: 0.53, y: 0.9 });

  const insD = useDeferredValue(inserted);
  const st = useMemo(() => cached(`ivf|${shift}|${insD}|${recluster}`, () => ivfState(shift, insD, recluster)), [shift, insD, recluster]);
  const cells = useMemo(() => voronoiCells(st.cents), [st]);
  const truth = useMemo(() => ivfTruth(st.pts, query), [st, query]);
  const tr = ivfSearch(st.cents, st.lists, st.pts, query, nprobe, truth);
  const probed = new Set(tr.probed);
  const found = new Set(tr.found);

  const curve = recluster ? st.refitCurve : st.fittedCurve;
  const cur = curve[nprobe - 1];
  const mean = st.pts.length / IVF.NLIST;
  const maxList = Math.max(...st.sizes);
  const scan95Fit = scanFor(st.fittedCurve, 0.95);
  const scan95Refit = scanFor(st.refitCurve, 0.95);
  const trainN = recluster ? st.pts.length : st.build.length;
  const minRec = Math.min(...st.fittedCurve.slice(0, 8).map((c) => c.recall), ...st.refitCurve.slice(0, 8).map((c) => c.recall));
  const barW = 290;

  return (
    <VizPanel
      title="Inserting into IVF lists whose centroids were fitted to older data"
      subtitle="k-means fitted 16 centroids to the first 400 vectors. Every later insert goes to its nearest existing centroid; nothing moves the centroids until you recluster. Click to move the query."
      controls={
        <>
          {tabControl}
          <Choice
            label="New data"
            value={shift}
            onChange={setShift}
            options={[
              { value: 'new', label: 'A new cluster appears' },
              { value: 'move', label: 'The clusters move' },
              { value: 'none', label: 'Same distribution as the build' },
            ]}
          />
          <Slider label="Rows inserted after build" min={0} max={IVF.MAX_INSERT} step={100} value={inserted} onChange={setInserted} format={fmtNum} />
          <Slider label="nprobe" min={1} max={8} value={nprobe} onChange={setNprobe} />
          <Check label="Recluster (refit centroids on all rows)" checked={recluster} onChange={setRecluster} />
        </>
      }
      legend={
        <Legend
          items={[
            { label: `Row the centroids were ${recluster ? 'refitted on' : 'fitted on'}`, color: 'var(--viz-1)', shape: 'dot' },
            ...(recluster ? [] : [{ label: 'Inserted row inside its list’s fitted radius', color: 'var(--viz-2)' }, { label: 'Inserted row outside it', color: 'var(--viz-5)' }]),
            { label: 'List probed for this query', color: 'var(--viz-4)' },
            { label: 'True neighbour found', color: 'var(--viz-good)', shape: 'dot' },
            { label: 'True neighbour missed', color: 'var(--viz-critical)', shape: 'dot' },
            { label: 'Chart: centroids fitted at build', color: 'var(--viz-7)', shape: 'line' },
            { label: 'Chart: centroids refitted on all rows', color: 'var(--viz-3)', shape: 'line' },
          ]}
        />
      }
      stats={
        <Stats
          items={[
            { label: `recall@10 at nprobe ${nprobe}`, value: cur.recall.toFixed(2), hint: '40 queries drawn from the most recently inserted rows (or from the build rows when nothing was inserted).' },
            { label: 'Rows scanned / query', value: fmtNum(cur.scanned, 0), hint: 'Every vector in the probed lists is compared with the query.' },
            { label: 'Largest list vs average', value: `${fmtNum(maxList)} vs ${fmtNum(mean, 1)}`, hint: 'Rows per list. A list that absorbs the new data is scanned in full whenever it is probed.' },
            { label: 'Inserted rows outside fitted radius', value: recluster ? '0 (refitted)' : `${fmtNum(st.outOfFit.size)} of ${fmtNum(st.added.length)}`, hint: 'Farther from their centroid than any row that centroid was trained on.' },
            { label: 'Rows scanned for 95% recall', value: `${scan95Fit === null ? '—' : fmtNum(scan95Fit, 0)} fitted · ${scan95Refit === null ? '—' : fmtNum(scan95Refit, 0)} refit`, hint: 'The smallest nprobe that reaches recall 0.95, in rows scanned per query.' },
            { label: 'Reclustering cost', value: `${fmtNum(st.reclusterCost)} distances`, hint: `k-means++ and 25 Lloyd iterations, best of three, over ${fmtNum(st.pts.length)} rows, then every row reassigned (rewriting every list).` },
          ]}
        />
      }
      note={
        <Note>
          {st.added.length === 0 ? (
            <>
              <strong>Nothing has been inserted yet.</strong> The 16 lists hold the 400 rows they were fitted on. Add rows to see where they land.
            </>
          ) : recluster ? (
            <>
              <strong>Reclustered on all {fmtNum(st.pts.length)} rows.</strong> The largest list now holds {fmtNum(maxList)} rows against an average of {fmtNum(mean, 1)}, and nprobe {nprobe} scans {fmtNum(cur.scanned, 0)} rows for recall {cur.recall.toFixed(2)}. The refit cost {fmtNum(st.reclusterCost)} distance computations and moved rows between lists — the rebuild an IVF index needs, not an update.
            </>
          ) : (
            <>
              <strong>
                {fmtNum(st.outOfFit.size)} of {fmtNum(st.added.length)} inserted rows ({pct(st.outOfFit.size / Math.max(1, st.added.length))}) landed outside the region their list was fitted to.
              </strong>{' '}
              {shift === 'none' ? 'They go to the nearest centroid, as the build rows did. The largest list holds' : 'They still go to the nearest centroid, so the lists nearest the new data swell. The largest holds'} {fmtNum(maxList)} rows against an average of {fmtNum(mean, 1)}, and nprobe {nprobe} now scans {fmtNum(cur.scanned, 0)} rows for recall {cur.recall.toFixed(2)}. Reaching 95% recall takes {scan95Fit === null ? 'more than 16 probes' : `${fmtNum(scan95Fit, 0)} rows`} here and {scan95Refit === null ? 'more than 16 probes' : `${fmtNum(scan95Refit, 0)} rows`} after reclustering.
            </>
          )}
        </Note>
      }
      table={
        <table className="viz-table">
          <thead>
            <tr>
              <th>nprobe</th>
              <th>recall@10, fitted at build</th>
              <th>rows scanned, fitted</th>
              <th>recall@10, refitted</th>
              <th>rows scanned, refitted</th>
            </tr>
          </thead>
          <tbody>
            {st.fittedCurve.map((c, i) => (
              <tr key={c.nprobe}>
                <td>{c.nprobe}</td>
                <td>{c.recall.toFixed(2)}</td>
                <td>{fmtNum(c.scanned, 0)}</td>
                <td>{st.refitCurve[i].recall.toFixed(2)}</td>
                <td>{fmtNum(st.refitCurve[i].scanned, 0)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      }
    >
      <div>
        <svg
          viewBox={`0 0 ${SIDE} ${SIDE}`}
          width={SIDE}
          height={SIDE}
          role="img"
          aria-label={`IVF lists: ${st.added.length} rows inserted, ${st.outOfFit.size} outside their fitted cell, nprobe ${nprobe} scans ${tr.scanned} rows`}
          onClick={(e) => setQuery(toData(e))}
          style={{ cursor: 'crosshair', width: '100%', maxWidth: 520, margin: '0 auto' }}
        >
          <rect x={0.5} y={0.5} width={SIDE - 1} height={SIDE - 1} fill="var(--viz-surface)" stroke="var(--viz-border)" />
          {cells.map((poly, i) =>
            poly.length ? (
              <polygon
                key={i}
                points={poly.map((p) => `${sx(p.x)},${sx(p.y)}`).join(' ')}
                fill={probed.has(i) ? 'var(--viz-4)' : 'none'}
                fillOpacity={probed.has(i) ? 0.22 : 0}
                stroke={probed.has(i) ? 'var(--viz-ink-2)' : 'var(--viz-axis)'}
                strokeWidth={probed.has(i) ? 1.6 : 0.8}
                pointerEvents="none"
              />
            ) : null,
          )}
          {st.cents.map((c, i) => (
            <circle key={`r${i}`} cx={sx(c.x)} cy={sx(c.y)} r={st.radius[i] * (SIDE - 2 * PAD)} fill="none" stroke="var(--viz-ink-muted)" strokeDasharray="2 3" strokeOpacity={0.55} pointerEvents="none" />
          ))}
          {st.pts.map((p, i) => {
            const x = sx(p.x);
            const y = sx(p.y);
            const isNew = i >= trainN;
            const out = st.outOfFit.has(i);
            return (
              <g key={i} pointerEvents="none">
                {isNew ? <path d={`M${x},${y - 3.2} L${x + 3.2},${y} L${x},${y + 3.2} L${x - 3.2},${y} Z`} fill={out ? 'var(--viz-5)' : 'var(--viz-2)'} /> : <circle cx={x} cy={y} r={2.2} fill="var(--viz-1)" />}
                {truth.includes(i) ? <circle cx={x} cy={y} r={6.5} fill="none" stroke={found.has(i) ? 'var(--viz-good)' : 'var(--viz-critical)'} strokeWidth={1.8} strokeDasharray={found.has(i) ? undefined : '3 2'} /> : null}
              </g>
            );
          })}
          {st.cents.map((c, i) => (
            <g key={`c${i}`} pointerEvents="none">
              <path d={`M${sx(c.x) - 6},${sx(c.y)} L${sx(c.x) + 6},${sx(c.y)} M${sx(c.x)},${sx(c.y) - 6} L${sx(c.x)},${sx(c.y) + 6}`} stroke="var(--viz-ink)" strokeWidth={2.2} />
            </g>
          ))}
          <Star p={query} label="query" />
        </svg>
        <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.5rem', justifyContent: 'center', marginTop: '0.5rem' }}>
          <MiniChart
            title="recall@10 vs rows scanned (nprobe 1–8)"
            xs={[st.fittedCurve.slice(0, 8).map((c) => c.scanned), st.refitCurve.slice(0, 8).map((c) => c.scanned)]}
            series={[
              { key: 'fit', color: 'var(--viz-7)', label: 'fitted', values: st.fittedCurve.slice(0, 8).map((c) => c.recall) },
              { key: 'refit', color: 'var(--viz-3)', dashed: true, label: 'refit', values: st.refitCurve.slice(0, 8).map((c) => c.recall) },
            ]}
            xLabel="rows scanned per query"
            xFmt={(v) => fmtNum(v)}
            yMin={Math.min(0.7, Math.floor(minRec * 10) / 10)}
            yMax={1}
            yFmt={(v) => v.toFixed(2)}
            cursor={{ x: cur.scanned, seriesIndex: recluster ? 1 : 0 }}
            ariaLabel={`recall against rows scanned; current nprobe ${nprobe} scans ${fmtNum(cur.scanned, 0)} rows at recall ${cur.recall.toFixed(2)}`}
            pointLabels={[st.fittedCurve.slice(0, 8).map((c) => (c.nprobe === 1 ? 'nprobe 1' : '')), st.refitCurve.slice(0, 8).map((c) => (c.nprobe === 1 ? 'nprobe 1' : ''))]}
          />
          <svg viewBox={`0 0 ${barW} 150`} width={barW} height={150} role="img" aria-label={`Rows per list: largest ${maxList}, average ${fmtNum(mean, 1)}`}>
            <text x={44} y={14} fontSize={11.5} fill="var(--viz-ink)">
              rows per list, {recluster ? 'refitted' : 'as fitted at build'}
            </text>
            {(() => {
              const top = Math.max(maxList, 100);
              const bw = (barW - 56) / IVF.NLIST;
              const Y = (v: number) => 128 - (v / top) * 100;
              return (
                <>
                  <line x1={44} x2={barW - 12} y1={128} y2={128} stroke="var(--viz-axis)" />
                  <text x={40} y={Y(top) + 4} fontSize={10} textAnchor="end" fill="var(--viz-ink-2)">
                    {fmtNum(top)}
                  </text>
                  <text x={40} y={131} fontSize={10} textAnchor="end" fill="var(--viz-ink-2)">
                    0
                  </text>
                  {st.sizes.map((v, i) => (
                    <rect key={i} x={46 + i * bw} y={Y(v)} width={bw - 3} height={128 - Y(v)} fill={probed.has(i) ? 'var(--viz-seq-550)' : 'var(--viz-seq-250)'} stroke={probed.has(i) ? 'var(--viz-ink)' : 'none'} strokeWidth={1.5} />
                  ))}
                  <line x1={44} x2={barW - 12} y1={Y(mean)} y2={Y(mean)} stroke="var(--viz-ink-2)" strokeDasharray="4 3" />
                  <text x={(44 + barW - 12) / 2} y={146} fontSize={10} textAnchor="middle" fill="var(--viz-ink-2)">
                    16 lists · dark = probed · dashed = average
                  </text>
                </>
              );
            })()}
          </svg>
        </div>
      </div>
    </VizPanel>
  );
}

export default function VectorChurnTombstoneLab() {
  const [tab, setTab] = useState<Tab>('hnsw');
  const tabControl = <Segmented label="Index" value={tab} options={TABS} onChange={setTab} />;
  return tab === 'hnsw' ? <HnswChurnPanel tabControl={tabControl} /> : <IvfDriftPanel tabControl={tabControl} />;
}
