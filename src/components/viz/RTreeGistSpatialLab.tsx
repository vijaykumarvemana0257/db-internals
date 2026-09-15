import { useDeferredValue, useId, useMemo, useRef, useState } from 'react';
import { VizPanel, Segmented, Choice, Check, Slider, Button, Legend, Stats, Note, TooltipHost, useTip, makeRng, fmtNum } from './Viz';
/**
 * An in-memory R-tree over a 100 x 100 plane, rebuilt from the insertion sequence on every change.
 *
 * Algorithms (sources: Guttman, SIGMOD 1984; Beckmann, Kriegel, Schneider & Seeger, SIGMOD 1990):
 * - linear / quadratic: ChooseLeaf by least area enlargement (ties: smaller area). Quadratic PickSeeds maximizes
 *   area(J) - area(E1) - area(E2); PickNext maximizes |d1 - d2|; LinearPickSeeds uses the greatest normalized
 *   separation and PickNext takes entries in order. A group that needs every remaining entry to reach m gets them.
 * - rstar: ChooseSubtree by least overlap enlargement when the children are leaves, else least area enlargement;
 *   split axis by minimum summed margin over both sorts, cut by minimum overlap then area; forced reinsert of
 *   round(0.3 M) entries farthest from the node centre on the first overflow per level (root excepted), close order.
 *
 * Model assumptions, labelled where the reader sees numbers:
 * - Pages hold M entries regardless of key size; the minimum fill is m = max(2, round(0.4 M)) for every algorithm.
 * - Areas are reported as a percentage of the plane; "sibling-box overlap" sums pairwise intersections of the
 *   entries of every internal page, so it can exceed 100%.
 * - A window query reads the root plus every page whose box intersects the window. KNN is best-first over one
 *   queue keyed by MINDIST, with objects ahead of pages on equal distance; distances to rectangles are exact, so
 *   there is no recheck step (PostGIS rechecks leaf distances).
 * - Workload averages use a fixed 10 x 10 grid of windows and KNN probe points.
 * - Page boxes are drawn slightly expanded per level so that coincident edges stay visible.
 */

/* ================================================================== model */

export type Rect = { x1: number; y1: number; x2: number; y2: number };
export type Algo = 'linear' | 'quadratic' | 'rstar';
/** A node entry. Leaf entries point at an object (obj >= 0); internal entries point at a child node. */
export type Entry = { rect: Rect; child: number; obj: number };
export type TNode = { id: number; level: number; entries: Entry[] };
export type BuildEvent = { kind: 'choose' | 'split' | 'reinsert' | 'root'; level: number; node: number; text: string };
export type Tree = {
  nodes: TNode[];
  root: number;
  M: number;
  m: number;
  algo: Algo;
  reinsert: boolean;
  splits: number;
  reinserted: number;
  /** Events of the most recent top-level insertion. */
  last: BuildEvent[];
};

export const PLANE = 100;
const EPS = 1e-9;

export const area = (r: Rect) => Math.max(0, r.x2 - r.x1) * Math.max(0, r.y2 - r.y1);
/** R*-tree "margin": the sum of the edge lengths of the rectangle. */
export const margin = (r: Rect) => 2 * (r.x2 - r.x1 + (r.y2 - r.y1));
export const union = (a: Rect, b: Rect): Rect => ({ x1: Math.min(a.x1, b.x1), y1: Math.min(a.y1, b.y1), x2: Math.max(a.x2, b.x2), y2: Math.max(a.y2, b.y2) });
export const intersects = (a: Rect, b: Rect) => a.x1 <= b.x2 && b.x1 <= a.x2 && a.y1 <= b.y2 && b.y1 <= a.y2;
export const overlapArea = (a: Rect, b: Rect) => Math.max(0, Math.min(a.x2, b.x2) - Math.max(a.x1, b.x1)) * Math.max(0, Math.min(a.y2, b.y2) - Math.max(a.y1, b.y1));
export const enlargement = (r: Rect, add: Rect) => area(union(r, add)) - area(r);
export const mbrOf = (es: { rect: Rect }[]): Rect => es.reduce((acc, e) => union(acc, e.rect), es[0].rect);
/** Smallest distance from a point to a rectangle (0 inside): the MINDIST lower bound used by KNN. */
export const minDist = (px: number, py: number, r: Rect) => {
  const dx = Math.max(r.x1 - px, 0, px - r.x2);
  const dy = Math.max(r.y1 - py, 0, py - r.y2);
  return Math.sqrt(dx * dx + dy * dy);
};

export const minFill = (M: number) => Math.max(2, Math.round(0.4 * M));
export const reinsertCount = (M: number) => Math.max(1, Math.round(0.3 * M));

const f1 = (n: number) => (Math.abs(n) >= 100 ? Math.round(n).toString() : n.toFixed(1));
/** An area in plane units (the plane is 100 x 100) as a percentage of the plane. */
const pct = (n: number) => `${(n / 100).toFixed(2)}% of the plane`;

function refresh(t: Tree, id = t.root): Rect | null {
  const n = t.nodes[id];
  if (!n.entries.length) return null;
  if (n.level > 0) for (const e of n.entries) e.rect = refresh(t, e.child) ?? e.rect;
  return mbrOf(n.entries);
}

/* ------------------------------------------------------- ChooseSubtree */

function chooseSubtree(t: Tree, rect: Rect, level: number, ev: BuildEvent[] | null): number[] {
  const path = [t.root];
  let n = t.nodes[t.root];
  while (n.level > level) {
    let best = 0;
    let k1 = Infinity;
    let k2 = Infinity;
    let k3 = Infinity;
    const useOverlap = t.algo === 'rstar' && n.level === 1;
    for (let i = 0; i < n.entries.length; i++) {
      const e = n.entries[i];
      const enl = enlargement(e.rect, rect);
      const a = area(e.rect);
      let ov = enl;
      if (useOverlap) {
        const grown = union(e.rect, rect);
        ov = 0;
        for (let j = 0; j < n.entries.length; j++) {
          if (j === i) continue;
          ov += overlapArea(grown, n.entries[j].rect) - overlapArea(e.rect, n.entries[j].rect);
        }
      }
      const better = useOverlap
        ? ov < k1 - EPS || (Math.abs(ov - k1) <= EPS && (enl < k2 - EPS || (Math.abs(enl - k2) <= EPS && a < k3 - EPS)))
        : enl < k1 - EPS || (Math.abs(enl - k1) <= EPS && a < k3 - EPS);
      if (i === 0 || better) {
        best = i;
        k1 = useOverlap ? ov : enl;
        k2 = enl;
        k3 = a;
      }
    }
    if (ev)
      ev.push({
        kind: 'choose',
        level: n.level,
        node: n.id,
        text: useOverlap
          ? `at level ${n.level}, pick the child whose box gains the least overlap with its siblings (+${pct(k1)}), then the least area (+${pct(k2)})`
          : `at level ${n.level}, pick the child whose box needs the least area enlargement (+${pct(k2)})`,
      });
    n = t.nodes[n.entries[best].child];
    path.push(n.id);
  }
  return path;
}

/* ---------------------------------------------------------------- splits */

type Split = { a: Entry[]; b: Entry[]; how: string };

/** Guttman's DistributeEntry tie rules: least enlargement, then smaller area, then fewer entries. */
function assign(e: Entry, a: Entry[], b: Entry[], ra: Rect, rb: Rect) {
  const d1 = enlargement(ra, e.rect);
  const d2 = enlargement(rb, e.rect);
  if (d1 < d2 - EPS) return 0;
  if (d2 < d1 - EPS) return 1;
  if (area(ra) < area(rb) - EPS) return 0;
  if (area(rb) < area(ra) - EPS) return 1;
  return a.length <= b.length ? 0 : 1;
}

function distribute(entries: Entry[], s1: number, s2: number, m: number, pickNext: (rest: Entry[], ra: Rect, rb: Rect) => number): { a: Entry[]; b: Entry[] } {
  const a = [entries[s1]];
  const b = [entries[s2]];
  let ra = entries[s1].rect;
  let rb = entries[s2].rect;
  const rest = entries.filter((_, i) => i !== s1 && i !== s2);
  while (rest.length) {
    if (a.length + rest.length <= m) {
      a.push(...rest.splice(0));
      break;
    }
    if (b.length + rest.length <= m) {
      b.push(...rest.splice(0));
      break;
    }
    const i = pickNext(rest, ra, rb);
    const [e] = rest.splice(i, 1);
    if (assign(e, a, b, ra, rb) === 0) {
      a.push(e);
      ra = union(ra, e.rect);
    } else {
      b.push(e);
      rb = union(rb, e.rect);
    }
  }
  return { a, b };
}

export function quadraticSplit(entries: Entry[], m: number): Split {
  let s1 = 0;
  let s2 = 1;
  let worst = -Infinity;
  for (let i = 0; i < entries.length; i++)
    for (let j = i + 1; j < entries.length; j++) {
      const d = area(union(entries[i].rect, entries[j].rect)) - area(entries[i].rect) - area(entries[j].rect);
      if (d > worst + EPS) {
        worst = d;
        s1 = i;
        s2 = j;
      }
    }
  const { a, b } = distribute(entries, s1, s2, m, (rest, ra, rb) => {
    let best = 0;
    let diff = -Infinity;
    rest.forEach((e, i) => {
      const d = Math.abs(enlargement(ra, e.rect) - enlargement(rb, e.rect));
      if (d > diff + EPS) {
        diff = d;
        best = i;
      }
    });
    return best;
  });
  return { a, b, how: `quadratic split: the two seeds would waste the most area in one box (${pct(worst)}); each next entry is the one with the strongest preference for one group` };
}

export function linearSplit(entries: Entry[], m: number): Split {
  let bestSep = -Infinity;
  let s1 = 0;
  let s2 = 1;
  let axisName = 'x';
  for (const axis of ['x', 'y'] as const) {
    const lo = (e: Entry) => (axis === 'x' ? e.rect.x1 : e.rect.y1);
    const hi = (e: Entry) => (axis === 'x' ? e.rect.x2 : e.rect.y2);
    let highLow = 0;
    let lowHigh = 0;
    for (let i = 1; i < entries.length; i++) {
      if (lo(entries[i]) > lo(entries[highLow])) highLow = i;
      if (hi(entries[i]) < hi(entries[lowHigh])) lowHigh = i;
    }
    if (highLow === lowHigh) {
      // The same entry is extreme on both sides: pair it with the next-lowest high side.
      let alt = highLow === 0 ? 1 : 0;
      for (let i = 0; i < entries.length; i++) if (i !== highLow && hi(entries[i]) < hi(entries[alt])) alt = i;
      lowHigh = alt;
    }
    const width = Math.max(...entries.map(hi)) - Math.min(...entries.map(lo));
    const sep = width > EPS ? (lo(entries[highLow]) - hi(entries[lowHigh])) / width : 0;
    if (sep > bestSep + EPS) {
      bestSep = sep;
      s1 = lowHigh;
      s2 = highLow;
      axisName = axis;
    }
  }
  const { a, b } = distribute(entries, s1, s2, m, () => 0);
  return { a, b, how: `linear split: seeds are the most separated pair along ${axisName} (normalized ${bestSep.toFixed(2)}); the rest go in arbitrary order` };
}

export function rstarSplit(entries: Entry[], m: number): Split {
  const n = entries.length;
  const sorts = (axis: 'x' | 'y') => {
    const lo = (e: Entry) => (axis === 'x' ? e.rect.x1 : e.rect.y1);
    const hi = (e: Entry) => (axis === 'x' ? e.rect.x2 : e.rect.y2);
    return [entries.slice().sort((p, q) => lo(p) - lo(q) || hi(p) - hi(q)), entries.slice().sort((p, q) => hi(p) - hi(q) || lo(p) - lo(q))];
  };
  const dists = (sorted: Entry[]) => {
    const out: { a: Entry[]; b: Entry[]; ra: Rect; rb: Rect }[] = [];
    for (let size = m; size <= n - m; size++) {
      const a = sorted.slice(0, size);
      const b = sorted.slice(size);
      out.push({ a, b, ra: mbrOf(a), rb: mbrOf(b) });
    }
    return out;
  };
  let axis: 'x' | 'y' = 'x';
  let bestS = Infinity;
  const S: Record<string, number> = {};
  for (const ax of ['x', 'y'] as const) {
    let s = 0;
    for (const sorted of sorts(ax)) for (const d of dists(sorted)) s += margin(d.ra) + margin(d.rb);
    S[ax] = s;
    if (s < bestS - EPS) {
      bestS = s;
      axis = ax;
    }
  }
  let pick: { a: Entry[]; b: Entry[] } | null = null;
  let bestOv = Infinity;
  let bestArea = Infinity;
  for (const sorted of sorts(axis))
    for (const d of dists(sorted)) {
      const ov = overlapArea(d.ra, d.rb);
      const ar = area(d.ra) + area(d.rb);
      if (!pick || ov < bestOv - EPS || (Math.abs(ov - bestOv) <= EPS && ar < bestArea - EPS)) {
        pick = d;
        bestOv = ov;
        bestArea = ar;
      }
    }
  return {
    a: pick!.a,
    b: pick!.b,
    how: `R*-tree split: sorting along ${axis} gives the smaller summed margin over every allowed cut (x ${f1(S.x)}, y ${f1(S.y)}); on that axis the cut with the least overlap (${pct(bestOv)}) wins`,
  };
}

export function splitEntries(algo: Algo, entries: Entry[], m: number): Split {
  return algo === 'linear' ? linearSplit(entries, m) : algo === 'quadratic' ? quadraticSplit(entries, m) : rstarSplit(entries, m);
}

/* ---------------------------------------------------------------- insert */

function insertAt(t: Tree, entry: Entry, level: number, overflowed: Set<number>, ev: BuildEvent[], depth: number) {
  const path = chooseSubtree(t, entry.rect, level, depth === 0 ? ev : null);
  t.nodes[path[path.length - 1]].entries.push(entry);
  refresh(t);
  for (let i = path.length - 1; i >= 0; i--) {
    const node = t.nodes[path[i]];
    if (node.entries.length <= t.M) break;
    if (t.algo === 'rstar' && t.reinsert && node.id !== t.root && !overflowed.has(node.level) && depth < 12) {
      overflowed.add(node.level);
      const box = mbrOf(node.entries);
      const cx = (box.x1 + box.x2) / 2;
      const cy = (box.y1 + box.y2) / 2;
      const d = (e: Entry) => Math.hypot((e.rect.x1 + e.rect.x2) / 2 - cx, (e.rect.y1 + e.rect.y2) / 2 - cy);
      const sorted = node.entries.slice().sort((p, q) => d(q) - d(p));
      const p = reinsertCount(t.M);
      const removed = sorted.slice(0, p);
      node.entries = sorted.slice(p);
      refresh(t);
      t.reinserted += removed.length;
      ev.push({ kind: 'reinsert', level: node.level, node: node.id, text: `level ${node.level} overflowed for the first time: forced reinsert of the ${p} entr${p === 1 ? 'y' : 'ies'} farthest from the node's centre` });
      // Close reinsert: the removed entry nearest the centre goes back in first.
      for (let k = removed.length - 1; k >= 0; k--) insertAt(t, removed[k], node.level, overflowed, ev, depth + 1);
      return;
    }
    const s = splitEntries(t.algo, node.entries, t.m);
    node.entries = s.a;
    const sib: TNode = { id: t.nodes.length, level: node.level, entries: s.b };
    t.nodes.push(sib);
    t.splits++;
    ev.push({ kind: 'split', level: node.level, node: node.id, text: `level ${node.level} node overflowed (${s.a.length + s.b.length} > M=${t.M}): ${s.how} → ${s.a.length} + ${s.b.length}` });
    if (node.id === t.root) {
      const root: TNode = { id: t.nodes.length, level: node.level + 1, entries: [{ rect: mbrOf(s.a), child: node.id, obj: -1 }, { rect: mbrOf(s.b), child: sib.id, obj: -1 }] };
      t.nodes.push(root);
      t.root = root.id;
      ev.push({ kind: 'root', level: root.level, node: root.id, text: `the root split, so a new root is added and the tree grows to height ${root.level + 1}` });
      refresh(t);
      break;
    }
    t.nodes[path[i - 1]].entries.push({ rect: mbrOf(s.b), child: sib.id, obj: -1 });
    refresh(t);
  }
}

export function buildTree(rects: Rect[], M: number, algo: Algo, reinsert = true): Tree {
  const t: Tree = { nodes: [{ id: 0, level: 0, entries: [] }], root: 0, M, m: minFill(M), algo, reinsert, splits: 0, reinserted: 0, last: [] };
  rects.forEach((r, i) => {
    const ev: BuildEvent[] = [];
    insertAt(t, { rect: r, child: -1, obj: i }, 0, new Set(), ev, 0);
    t.last = ev;
  });
  return t;
}

/* --------------------------------------------------------------- metrics */

/** Live nodes reachable from the root, parents before children, left to right. */
export function levelOrder(t: Tree): TNode[] {
  const out: TNode[] = [];
  let frontier = [t.root];
  while (frontier.length) {
    const next: number[] = [];
    for (const id of frontier) {
      const n = t.nodes[id];
      out.push(n);
      if (n.level > 0) for (const e of n.entries) next.push(e.child);
    }
    frontier = next;
  }
  return out;
}

export function treeMetrics(t: Tree) {
  const nodes = levelOrder(t);
  let overlap = 0;
  let dirArea = 0;
  let dirMargin = 0;
  let leaves = 0;
  let entries = 0;
  for (const n of nodes) {
    if (n.level === 0) leaves++;
    entries += n.entries.length;
    for (let i = 0; i < n.entries.length; i++) for (let j = i + 1; j < n.entries.length; j++) if (n.level > 0) overlap += overlapArea(n.entries[i].rect, n.entries[j].rect);
    if (n.id !== t.root && n.entries.length) {
      const b = mbrOf(n.entries);
      dirArea += area(b);
      dirMargin += margin(b);
    }
  }
  return { nodes: nodes.length, leaves, height: t.nodes[t.root].level + 1, overlap, dirArea, dirMargin, fill: entries / (nodes.length * t.M), splits: t.splits, reinserted: t.reinserted };
}

/* ---------------------------------------------------------------- search */

/** Window query (&&): every entry whose box intersects the window is followed — possibly down several subtrees. */
export function windowQuery(t: Tree, q: Rect) {
  const visited: number[] = [];
  const fruitless: number[] = [];
  const hits: number[] = [];
  /** Internal pages where more than one child box intersected the window: the search forks there. */
  let forks = 0;
  const visit = (id: number): number => {
    visited.push(id);
    const n = t.nodes[id];
    let c = 0;
    let followed = 0;
    for (const e of n.entries) {
      if (!intersects(e.rect, q)) continue;
      if (n.level === 0) {
        hits.push(e.obj);
        c++;
      } else {
        followed++;
        c += visit(e.child);
      }
    }
    if (followed > 1) forks++;
    if (c === 0) fruitless.push(id);
    return c;
  };
  if (t.nodes[t.root].entries.length) visit(t.root);
  return { visited, fruitless, hits, forks };
}

export type KnnItem = { kind: 'node' | 'obj'; id: number; dist: number; seq: number };
export type KnnStep = { pop: KnnItem; rank: number; pushed: number; queue: KnnItem[] };

/** Best-first k-nearest-neighbour search: one priority queue of nodes and objects ordered by MINDIST. */
export function knnSearch(t: Tree, px: number, py: number, k: number) {
  const queue: KnnItem[] = [];
  let seq = 0;
  const cmp = (a: KnnItem, b: KnnItem) => a.dist - b.dist || (a.kind === b.kind ? 0 : a.kind === 'obj' ? -1 : 1) || a.seq - b.seq;
  const push = (it: KnnItem) => {
    let i = queue.length;
    while (i > 0 && cmp(queue[i - 1], it) > 0) i--;
    queue.splice(i, 0, it);
  };
  const steps: KnnStep[] = [];
  const results: number[] = [];
  let maxQueue = 0;
  let expanded = 0;
  const root = t.nodes[t.root];
  if (root.entries.length) push({ kind: 'node', id: t.root, dist: minDist(px, py, mbrOf(root.entries)), seq: seq++ });
  while (queue.length && results.length < k && steps.length < 2000) {
    const pop = queue.shift()!;
    let pushed = 0;
    if (pop.kind === 'obj') results.push(pop.id);
    else {
      expanded++;
      const n = t.nodes[pop.id];
      for (const e of n.entries) {
        push({ kind: n.level === 0 ? 'obj' : 'node', id: n.level === 0 ? e.obj : e.child, dist: minDist(px, py, e.rect), seq: seq++ });
        pushed++;
      }
    }
    maxQueue = Math.max(maxQueue, queue.length);
    steps.push({ pop, rank: pop.kind === 'obj' ? results.length : 0, pushed, queue: queue.slice(0, 6) });
  }
  return { steps, results, expanded, maxQueue };
}

/** Average cost of a workload: a 10x10 grid of windows of one size, and of KNN probes at the grid centres. */
export function workloadCost(t: Tree, winSize: number, k: number) {
  let pages = 0;
  let fruitless = 0;
  let knnPages = 0;
  const G = 10;
  for (let i = 0; i < G; i++)
    for (let j = 0; j < G; j++) {
      const cx = ((i + 0.5) / G) * PLANE;
      const cy = ((j + 0.5) / G) * PLANE;
      const q = windowQuery(t, { x1: cx - winSize / 2, y1: cy - winSize / 2, x2: cx + winSize / 2, y2: cy + winSize / 2 });
      pages += q.visited.length;
      fruitless += q.fruitless.length;
      knnPages += knnSearch(t, cx, cy, k).expanded;
    }
  return { pages: pages / (G * G), fruitless: fruitless / (G * G), knnPages: knnPages / (G * G) };
}

/* --------------------------------------------------------------- presets */

export type PresetName = 'uniform' | 'clustered' | 'strips' | 'sorted';

export function presetRects(name: PresetName, count = 120, seed = 7): Rect[] {
  const rnd = makeRng(seed * 7919 + name.length * 131);
  const clamp = (r: Rect): Rect => {
    const w = r.x2 - r.x1;
    const h = r.y2 - r.y1;
    const x1 = Math.min(Math.max(0, r.x1), PLANE - w);
    const y1 = Math.min(Math.max(0, r.y1), PLANE - h);
    return { x1: +x1.toFixed(1), y1: +y1.toFixed(1), x2: +(x1 + w).toFixed(1), y2: +(y1 + h).toFixed(1) };
  };
  const box = (cx: number, cy: number, w: number, h: number) => clamp({ x1: cx - w / 2, y1: cy - h / 2, x2: cx + w / 2, y2: cy + h / 2 });
  const out: Rect[] = [];
  if (name === 'uniform' || name === 'sorted') {
    for (let i = 0; i < count; i++) out.push(box(3 + rnd() * 94, 3 + rnd() * 94, 1 + rnd() * 4, 1 + rnd() * 4));
    if (name === 'sorted') out.sort((a, b) => a.x1 - b.x1);
  } else if (name === 'clustered') {
    const centres = [
      [22, 26],
      [70, 22],
      [30, 74],
      [76, 70],
    ];
    for (let i = 0; i < count; i++) {
      const [cx, cy] = centres[i % 4];
      out.push(box(cx + (rnd() - 0.5) * 30, cy + (rnd() - 0.5) * 30, 1 + rnd() * 3, 1 + rnd() * 3));
    }
  } else {
    const strips = Math.round(count * 0.6);
    for (let i = 0; i < strips; i++) {
      const horizontal = i % 2 === 0;
      const len = 12 + rnd() * 30;
      const thick = 0.6 + rnd() * 0.9;
      out.push(horizontal ? box(8 + rnd() * 84, 5 + rnd() * 90, len, thick) : box(5 + rnd() * 90, 8 + rnd() * 84, thick, len));
    }
    for (let i = strips; i < count; i++) out.push(box(5 + rnd() * 90, 5 + rnd() * 90, 1 + rnd() * 3, 1 + rnd() * 3));
  }
  return out;
}

/* ===================================================================== UI */

type Mode = 'draw' | 'query' | 'knn';
type Drag = { x0: number; y0: number; x: number; y: number };

const LEVEL_COLORS = ['var(--viz-2)', 'var(--viz-3)', 'var(--viz-4)', 'var(--viz-5)', 'var(--viz-6)', 'var(--viz-7)'];
const levelColor = (l: number) => LEVEL_COLORS[l] ?? 'var(--viz-ink-muted)';
const ALGO_LABEL: Record<Algo, string> = { linear: 'Guttman linear', quadratic: 'Guttman quadratic', rstar: 'R*-tree' };
const PRESETS: { value: PresetName; label: string }[] = [
  { value: 'uniform', label: '200 small boxes, random order' },
  { value: 'sorted', label: '200 small boxes, inserted left to right' },
  { value: 'clustered', label: '200 boxes in four clusters' },
  { value: 'strips', label: 'Long thin strips (roads) + boxes' },
];

const MAX_RECTS = 300;
const PX = 4.2;
const PAD = 10;
const SIDE = PAD * 2 + PLANE * PX;
const X = (v: number) => PAD + v * PX;
const clamp = (v: number, lo: number, hi: number) => Math.min(hi, Math.max(lo, v));
const norm = (d: Drag): Rect => ({ x1: Math.min(d.x0, d.x), y1: Math.min(d.y0, d.y), x2: Math.max(d.x0, d.x), y2: Math.max(d.y0, d.y) });
const around = (x: number, y: number, w: number, h: number): Rect => {
  const x1 = clamp(x - w / 2, 0, PLANE - w);
  const y1 = clamp(y - h / 2, 0, PLANE - h);
  return { x1, y1, x2: x1 + w, y2: y1 + h };
};
const d1 = (n: number) => (Number.isFinite(n) ? n.toFixed(1) : '—');

function TreeDiagram({
  t,
  visited,
  fruitless,
  expandedOrder,
  mode,
  newestLeaf,
}: {
  t: Tree;
  visited: Set<number>;
  fruitless: Set<number>;
  expandedOrder: Map<number, number>;
  mode: Mode;
  newestLeaf: number;
}) {
  const tip = useTip();
  const order = levelOrder(t);
  const height = t.nodes[t.root].level + 1;
  const leaves = order.filter((n) => n.level === 0);
  const W = 680;
  const LEFT = 66;
  const slot = (W - LEFT - 8) / Math.max(1, leaves.length);
  const nodeW = clamp(slot - 3, 4, 34);
  const ROW = 46;
  const H = height * ROW + 8;
  const pos = new Map<number, { x: number; y: number }>();
  leaves.forEach((n, i) => pos.set(n.id, { x: LEFT + (i + 0.5) * slot, y: 0 }));
  for (let lvl = 1; lvl < height; lvl++)
    for (const n of order.filter((q) => q.level === lvl)) {
      const xs = n.entries.map((e) => pos.get(e.child)?.x ?? 0);
      pos.set(n.id, { x: xs.reduce((a, b) => a + b, 0) / Math.max(1, xs.length), y: 0 });
    }
  const rowY = (lvl: number) => 10 + (height - 1 - lvl) * ROW;
  const highlight = (id: number) => (mode === 'query' ? visited.has(id) : mode === 'knn' ? expandedOrder.has(id) : id === newestLeaf);
  return (
    <svg viewBox={`0 0 ${W} ${H}`} width={W} height={H} style={{ minWidth: 520 }} role="img" aria-label={`Tree of ${order.length} pages in ${height} levels${mode === 'query' ? `; the window query read ${visited.size} of them` : ''}`}>
      {Array.from({ length: height }, (_, i) => height - 1 - i).map((lvl) => (
        <text key={lvl} x={4} y={rowY(lvl) + 12} fontSize={11} fill="var(--viz-ink-2)">
          {lvl === height - 1 ? 'root' : lvl === 0 ? 'leaves' : `level ${lvl}`}
        </text>
      ))}
      {order.map((n) =>
        n.level === 0
          ? null
          : n.entries.map((e) => {
              const a = pos.get(n.id)!;
              const b = pos.get(e.child)!;
              const on = highlight(n.id) && highlight(e.child);
              return <line key={`${n.id}-${e.child}`} x1={a.x} y1={rowY(n.level) + 16} x2={b.x} y2={rowY(n.level - 1)} stroke={on ? 'var(--viz-ink-2)' : 'var(--viz-grid)'} strokeWidth={on ? 1.6 : 1} />;
            }),
      )}
      {order.map((n) => {
        const p = pos.get(n.id)!;
        const w = n.level === 0 ? nodeW : Math.max(nodeW, 16);
        const on = highlight(n.id);
        const dashed = mode === 'query' && fruitless.has(n.id);
        const rank = expandedOrder.get(n.id);
        return (
          <g key={n.id} {...tip(<>{n.level === 0 ? 'Leaf page' : `Level ${n.level} page`} · {n.entries.length} of M={t.M} entries{mode === 'query' ? (visited.has(n.id) ? (fruitless.has(n.id) ? ' · read, nothing matched' : ' · read') : ' · skipped') : ''}{rank ? ` · expanded #${rank}` : ''}</>)}>
            <rect x={p.x - w / 2} y={rowY(n.level)} width={w} height={16} rx={3} fill={on ? levelColor(n.level) : 'var(--viz-surface)'} fillOpacity={on ? 0.3 : 1} stroke={levelColor(n.level)} strokeWidth={on ? 2 : 1.2} strokeDasharray={dashed ? '3 2' : undefined} strokeOpacity={mode !== 'draw' && !on ? 0.45 : 1} />
            {w >= 15 ? (
              <text x={p.x} y={rowY(n.level) + 12} textAnchor="middle" fontSize={10} fill="var(--viz-ink)">
                {n.entries.length}
              </text>
            ) : null}
            {rank ? (
              <text x={p.x} y={rowY(n.level) - 3} textAnchor="middle" fontSize={10} fontWeight={600} fill="var(--viz-ink)">
                {rank}
              </text>
            ) : null}
          </g>
        );
      })}
    </svg>
  );
}

export default function RTreeGistSpatialLab() {
  const [preset, setPreset] = useState<PresetName>('uniform');
  const [rects, setRects] = useState<Rect[]>(() => presetRects('uniform', 200));
  const [upto, setUpto] = useState(200);
  const [algo, setAlgo] = useState<Algo>('rstar');
  const [reinsert, setReinsert] = useState(true);
  const [M, setM] = useState(10);
  const [mode, setMode] = useState<Mode>('query');
  const [winSize, setWinSize] = useState(24);
  const [win, setWin] = useState<Rect>(() => around(50, 42, 24, 24));
  const [pt, setPt] = useState({ x: 50, y: 80 });
  const [k, setK] = useState(5);
  const [knnShown, setKnnShown] = useState(9999);
  const [showOverlap, setShowOverlap] = useState(true);
  const [drag, setDrag] = useState<Drag | null>(null);
  const svgRef = useRef<SVGSVGElement>(null);
  const hatchId = `rtree-hatch-${useId().replace(/[^a-zA-Z0-9_-]/g, '')}`;

  const data = useMemo(() => rects.slice(0, upto), [rects, upto]);
  const t = useMemo(() => buildTree(data, M, algo, reinsert), [data, M, algo, reinsert]);
  const mt = useMemo(() => treeMetrics(t), [t]);
  const order = useMemo(() => levelOrder(t), [t]);

  const dragRect = drag ? norm(drag) : null;
  const liveWin = mode === 'query' && dragRect && dragRect.x2 - dragRect.x1 > 1 && dragRect.y2 - dragRect.y1 > 1 ? dragRect : win;
  const q = useMemo(() => windowQuery(t, liveWin), [t, liveWin.x1, liveWin.y1, liveWin.x2, liveWin.y2]);
  const knn = useMemo(() => knnSearch(t, pt.x, pt.y, k), [t, pt.x, pt.y, k]);
  const shown = knn.steps.slice(0, Math.min(knnShown, knn.steps.length));
  const knnDone = shown.length === knn.steps.length;
  const expandedOrder = new Map<number, number>();
  if (mode === 'knn')
    shown.forEach((s) => {
      if (s.pop.kind === 'node') expandedOrder.set(s.pop.id, expandedOrder.size + 1);
    });
  const returned = shown.filter((s) => s.pop.kind === 'obj').map((s) => s.pop.id);
  const lastStep = shown[shown.length - 1];
  const kthDist = returned.length ? minDist(pt.x, pt.y, data[returned[returned.length - 1]]) : NaN;

  const visited = new Set(q.visited);
  const fruitless = new Set(q.fruitless);
  const hitSet = new Set(mode === 'query' ? q.hits : mode === 'knn' ? returned : []);

  // Leaf page holding the newest rectangle.
  const newestObj = data.length - 1;
  const newestLeaf = order.find((n) => n.level === 0 && n.entries.some((e) => e.obj === newestObj))?.id ?? -1;

  // Comparison table: the same rectangles under each split algorithm.
  const deferredData = useDeferredValue(data);
  const deferredM = useDeferredValue(M);
  const deferredWin = useDeferredValue(winSize);
  const deferredK = useDeferredValue(k);
  const compare = useMemo(
    () =>
      (['linear', 'quadratic', 'rstar'] as Algo[]).map((a) => {
        const tree = buildTree(deferredData, deferredM, a, reinsert);
        return { algo: a, m: treeMetrics(tree), w: workloadCost(tree, deferredWin, deferredK) };
      }),
    [deferredData, deferredM, deferredWin, deferredK, reinsert],
  );

  const overlaps = useMemo(() => {
    const out: { r: Rect; key: string }[] = [];
    for (const n of order) {
      if (n.level === 0) continue;
      for (let i = 0; i < n.entries.length; i++)
        for (let j = i + 1; j < n.entries.length; j++) {
          const a = n.entries[i].rect;
          const b = n.entries[j].rect;
          if (overlapArea(a, b) > 0) out.push({ key: `${n.id}-${i}-${j}`, r: { x1: Math.max(a.x1, b.x1), y1: Math.max(a.y1, b.y1), x2: Math.min(a.x2, b.x2), y2: Math.min(a.y2, b.y2) } });
        }
    }
    return out;
  }, [order]);

  const toPlane = (e: React.PointerEvent) => {
    const svg = svgRef.current;
    const ctm = svg?.getScreenCTM();
    if (!svg || !ctm) return null;
    const p = new DOMPoint(e.clientX, e.clientY).matrixTransform(ctm.inverse());
    return { x: clamp((p.x - PAD) / PX, 0, PLANE), y: clamp((p.y - PAD) / PX, 0, PLANE) };
  };
  const onDown = (e: React.PointerEvent<SVGSVGElement>) => {
    const p = toPlane(e);
    if (!p) return;
    e.currentTarget.setPointerCapture?.(e.pointerId);
    setDrag({ x0: p.x, y0: p.y, x: p.x, y: p.y });
  };
  const onMove = (e: React.PointerEvent<SVGSVGElement>) => {
    if (!drag) return;
    const p = toPlane(e);
    if (p) setDrag({ ...drag, x: p.x, y: p.y });
  };
  const onUp = () => {
    if (!drag) return;
    const r = norm(drag);
    const tiny = r.x2 - r.x1 < 1 && r.y2 - r.y1 < 1;
    const round = (v: number) => Math.round(v * 10) / 10;
    if (mode === 'draw' && data.length < MAX_RECTS) {
      const nr = tiny ? around(drag.x, drag.y, 3, 3) : { x1: r.x1, y1: r.y1, x2: Math.max(r.x2, r.x1 + 0.5), y2: Math.max(r.y2, r.y1 + 0.5) };
      setRects([...data, { x1: round(nr.x1), y1: round(nr.y1), x2: round(nr.x2), y2: round(nr.y2) }]);
      setUpto(data.length + 1);
    } else if (mode === 'query') {
      if (tiny || r.x2 - r.x1 <= 1 || r.y2 - r.y1 <= 1) setWin(around(drag.x, drag.y, winSize, winSize));
      else {
        setWin(r);
        setWinSize(clamp(Math.round(Math.max(r.x2 - r.x1, r.y2 - r.y1)), 4, 60));
      }
    } else if (mode === 'knn') {
      setPt({ x: round(drag.x), y: round(drag.y) });
      setKnnShown(9999);
    }
    setDrag(null);
  };

  /** Keyboard alternative to dragging: arrow keys move the query window or the KNN point (Shift moves 10 units). */
  const onKey = (e: React.KeyboardEvent<SVGSVGElement>) => {
    const step = e.shiftKey ? 10 : 2;
    const moves: Record<string, [number, number]> = { ArrowLeft: [-step, 0], ArrowRight: [step, 0], ArrowUp: [0, -step], ArrowDown: [0, step] };
    const d = moves[e.key];
    if (!d || mode === 'draw') return;
    e.preventDefault();
    if (mode === 'query') {
      const w = win.x2 - win.x1;
      const h = win.y2 - win.y1;
      setWin(around((win.x1 + win.x2) / 2 + d[0], (win.y1 + win.y2) / 2 + d[1], w, h));
    } else {
      setPt({ x: clamp(pt.x + d[0], 0, PLANE), y: clamp(pt.y + d[1], 0, PLANE) });
      setKnnShown(9999);
    }
  };

  const choosePreset = (p: PresetName) => {
    const r = presetRects(p, 200);
    setPreset(p);
    setRects(r);
    setUpto(r.length);
    setKnnShown(9999);
  };
  const addRandom = () => {
    const rnd = makeRng(data.length * 7919 + 17);
    const extra: Rect[] = [];
    for (let i = 0; i < 20 && data.length + extra.length < MAX_RECTS; i++) {
      const w = 1 + rnd() * 4;
      const h = 1 + rnd() * 4;
      const b = around(3 + rnd() * 94, 3 + rnd() * 94, w, h);
      extra.push({ x1: +b.x1.toFixed(1), y1: +b.y1.toFixed(1), x2: +b.x2.toFixed(1), y2: +b.y2.toFixed(1) });
    }
    setRects([...data, ...extra]);
    setUpto(data.length + extra.length);
  };

  const top = t.nodes[t.root].level;
  const internalVisited = q.visited.filter((id) => t.nodes[id].level > 0).length;
  const pad = (lvl: number) => lvl * 0.35;

  const note = (() => {
    if (!data.length) return <>The tree is empty. Choose a data set, press <strong>Add 20 random</strong>, or switch the pointer to <strong>Draw</strong> and drag on the plane.</>;
    if (mode === 'draw')
      return (
        <>
          {t.last.length ? (
            <>
              <strong>Rectangle #{data.length}</strong> went in like this: {t.last.map((e) => e.text).join('; ')}.{' '}
              {t.last.some((e) => e.kind !== 'choose') ? '' : 'The leaf had room, so its box and its ancestors’ boxes just grew to cover it (union). '}
            </>
          ) : (
            <>
              <strong>Rectangle #{data.length}</strong> went straight into the root, which is still a single leaf page with room for M={t.M} entries.{' '}
            </>
          )}
          Drag on the plane to insert another, or move <strong>Rectangles inserted</strong> to replay the build.
        </>
      );
    if (mode === 'query')
      return (
        <>
          The window intersects <strong>{q.hits.length}</strong> rectangle{q.hits.length === 1 ? '' : 's'}. The search read <strong>{q.visited.length} of {mt.nodes}</strong> pages: at {q.forks} of the {internalVisited} internal pages it read, more than one child box intersected the window, so it went down several subtrees.{' '}
          {q.fruitless.length ? `${q.fruitless.length} page${q.fruitless.length === 1 ? '' : 's'} (dashed) matched nothing — a bounding box covers empty space and its siblings’ territory, so "intersects the box" does not mean "holds a match".` : 'Every page it read contributed a match.'}
        </>
      );
    if (!lastStep) return <>Press <strong>Step</strong> to pop the root from the priority queue.</>;
    const s = lastStep;
    return (
      <>
        Step {shown.length}:{' '}
        {s.pop.kind === 'node' ? (
          <>
            popped a {t.nodes[s.pop.id].level === 0 ? 'leaf' : `level ${t.nodes[s.pop.id].level}`} page at distance ≥ {d1(s.pop.dist)} and pushed its {s.pushed} entries, each keyed by the smallest distance it could possibly have.{' '}
          </>
        ) : (
          <>
            popped rectangle #{s.pop.id + 1} at distance {d1(s.pop.dist)}. Nothing still in the queue can be closer, so it is result {s.rank} of {k}.{' '}
          </>
        )}
        {knnDone ? `Done: ${returned.length} nearest found after expanding ${expandedOrder.size} of ${mt.nodes} pages; the queue peaked at ${knn.maxQueue} entries.` : `Queue head: ${s.queue.length ? `${s.queue[0].kind === 'node' ? 'a page' : `rectangle #${s.queue[0].id + 1}`} at ${d1(s.queue[0].dist)}` : 'empty'}.`}
      </>
    );
  })();

  return (
    <VizPanel
      title="An R-tree you can build, query and search for nearest neighbours"
      subtitle="Every page stores bounding boxes. Choose a split algorithm and a page size, then drag on the plane: draw rectangles, sweep a query window, or drop a point for a k-nearest-neighbour search."
      controls={
        <>
          <Segmented
            label="Split algorithm"
            value={algo}
            onChange={(v) => {
              setAlgo(v);
              setKnnShown(9999);
            }}
            options={[
              { value: 'linear', label: 'Linear' },
              { value: 'quadratic', label: 'Quadratic' },
              { value: 'rstar', label: 'R*-tree' },
            ]}
          />
          {algo === 'rstar' ? <Check label="Forced reinsert" checked={reinsert} onChange={setReinsert} /> : null}
          <Slider label="Max entries per page (M)" min={4} max={16} value={M} onChange={(v) => { setM(v); setKnnShown(9999); }} format={(v) => `${v}, min ${minFill(v)}`} />
          <Choice label="Data" value={preset} onChange={choosePreset} options={PRESETS} />
          {rects.length ? <Slider label="Rectangles inserted" min={1} max={rects.length} value={Math.max(1, upto)} onChange={(v) => { setUpto(v); setKnnShown(9999); }} /> : null}
          <Segmented
            label="Pointer"
            value={mode}
            onChange={(v) => {
              setMode(v);
              setKnnShown(9999);
            }}
            options={[
              { value: 'draw', label: 'Draw' },
              { value: 'query', label: 'Query window' },
              { value: 'knn', label: 'KNN point' },
            ]}
          />
          <Slider label="Window size" min={4} max={60} value={winSize} onChange={(v) => { setWinSize(v); setWin(around((win.x1 + win.x2) / 2, (win.y1 + win.y2) / 2, v, v)); }} />
          <Slider label="k nearest" min={1} max={10} value={k} onChange={(v) => { setK(v); setKnnShown(9999); }} />
          <Check label="Shade overlap" checked={showOverlap} onChange={setShowOverlap} />
        </>
      }
      legend={
        <Legend
          items={[
            { label: 'Rectangle (solid: returned by the search)', color: 'var(--viz-1)' },
            ...Array.from({ length: top + 1 }, (_, l) => ({ label: `${l === 0 ? 'Leaf page' : l === top ? 'Root page' : `Level ${l} page`} box`, color: levelColor(l), shape: 'line' as const })),
            { label: 'Overlap between sibling boxes (hatched)', color: 'var(--viz-critical)' },
            ...(mode === 'draw' ? [] : [{ label: mode === 'knn' ? 'Query point, distance to k-th result' : 'Query window', color: 'var(--viz-ink)', shape: 'line' as const }]),
          ]}
        />
      }
      stats={
        <Stats
          items={[
            { label: 'Pages (height)', value: `${mt.nodes} (${mt.height})`, hint: `${mt.splits} splits so far${algo === 'rstar' && reinsert ? `, ${mt.reinserted} entries force-reinserted` : ''}. Average fill ${Math.round(mt.fill * 100)}%.` },
            { label: 'Sibling-box overlap', value: `${d1(mt.overlap / 100)}%`, hint: 'Sum over every internal page of the pairwise intersection areas of its entries’ boxes, as a percentage of the plane. Can exceed 100%.' },
            { label: 'Window: pages read', value: `${q.visited.length} of ${mt.nodes}`, hint: 'Pages whose box intersects the window, plus the root.' },
            { label: 'Read, no match', value: fmtNum(q.fruitless.length) },
            { label: `KNN (k=${k}): pages expanded`, value: `${mode === 'knn' ? expandedOrder.size : knn.expanded}`, hint: 'Pages popped from the priority queue before the k-th result.' },
          ]}
        />
      }
      note={<Note>{note}</Note>}
      table={
        <table className="viz-table">
          <thead>
            <tr>
              <th>Split algorithm (same {deferredData.length} rectangles, M={deferredM})</th>
              <th>Pages</th>
              <th>Height</th>
              <th>Sibling-box overlap</th>
              <th>Avg pages per {deferredWin}×{deferredWin} window (100 windows)</th>
              <th>Avg pages read with no match</th>
              <th>Avg pages expanded, KNN k={deferredK} (100 points)</th>
            </tr>
          </thead>
          <tbody>
            {compare.map((c) => (
              <tr key={c.algo}>
                <td>{ALGO_LABEL[c.algo]}{c.algo === 'rstar' ? (reinsert ? ' + forced reinsert' : ', no reinsert') : ''}</td>
                <td>{c.m.nodes}</td>
                <td>{c.m.height}</td>
                <td>{d1(c.m.overlap / 100)}%</td>
                <td>{c.w.pages.toFixed(2)}</td>
                <td>{c.w.fruitless.toFixed(2)}</td>
                <td>{c.w.knnPages.toFixed(2)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      }
    >
      <div className="viz-controls">
        <Button onClick={addRandom} disabled={data.length >= MAX_RECTS}>
          Add 20 random
        </Button>
        <Button onClick={() => { setRects(data.slice(0, -1)); setUpto(Math.max(0, data.length - 1)); }} disabled={!data.length}>
          Undo last
        </Button>
        <Button onClick={() => { setRects([]); setUpto(0); }} disabled={!data.length}>
          Clear
        </Button>
        {mode === 'knn' ? (
          <>
            <Button onClick={() => setKnnShown(0)}>Restart search</Button>
            <Button primary onClick={() => setKnnShown(Math.min(knn.steps.length, shown.length + 1))} disabled={knnDone}>
              Step
            </Button>
            <Button onClick={() => setKnnShown(9999)} disabled={knnDone}>
              Run to end
            </Button>
          </>
        ) : null}
      </div>
      <TooltipHost>
        <svg
          ref={svgRef}
          viewBox={`0 0 ${SIDE} ${SIDE}`}
          width={SIDE}
          height={SIDE}
          role="img"
          aria-label={`Plane with ${data.length} rectangles and the bounding boxes of ${mt.nodes} R-tree pages${mode === 'query' ? '. Arrow keys move the query window' : mode === 'knn' ? '. Arrow keys move the query point' : ''}`}
          tabIndex={mode === 'draw' ? undefined : 0}
          onKeyDown={onKey}
          style={{ touchAction: 'none', cursor: mode === 'knn' ? 'crosshair' : 'cell' }}
          onPointerDown={onDown}
          onPointerMove={onMove}
          onPointerUp={onUp}
          onPointerCancel={() => setDrag(null)}
        >
          <defs>
            <pattern id={hatchId} width={5} height={5} patternUnits="userSpaceOnUse" patternTransform="rotate(45)">
              <line x1={0} y1={0} x2={0} y2={5} stroke="var(--viz-critical)" strokeWidth={1.5} />
            </pattern>
          </defs>
          <rect x={PAD} y={PAD} width={PLANE * PX} height={PLANE * PX} fill="var(--viz-plane)" stroke="var(--viz-axis)" />
          {[25, 50, 75].map((g) => (
            <g key={g}>
              <line x1={X(g)} x2={X(g)} y1={PAD} y2={PAD + PLANE * PX} className="viz-grid-line" />
              <line y1={X(g)} y2={X(g)} x1={PAD} x2={PAD + PLANE * PX} className="viz-grid-line" />
            </g>
          ))}
          {showOverlap
            ? overlaps.map((o) => <rect key={o.key} x={X(o.r.x1)} y={X(o.r.y1)} width={Math.max(0.5, (o.r.x2 - o.r.x1) * PX)} height={Math.max(0.5, (o.r.y2 - o.r.y1) * PX)} fill={`url(#${hatchId})`} fillOpacity={0.55} />)
            : null}
          {data.map((r, i) => {
            const hit = hitSet.has(i);
            return (
              <rect
                key={i}
                x={X(r.x1)}
                y={X(r.y1)}
                width={Math.max(1, (r.x2 - r.x1) * PX)}
                height={Math.max(1, (r.y2 - r.y1) * PX)}
                fill="var(--viz-1)"
                fillOpacity={hit ? 0.85 : 0.22}
                stroke={mode === 'draw' && i === data.length - 1 ? 'var(--viz-ink)' : 'var(--viz-1)'}
                strokeWidth={mode === 'draw' && i === data.length - 1 ? 2 : 0.8}
              />
            );
          })}
          {order
            .slice()
            .sort((a, b) => a.level - b.level)
            .map((n) => {
              if (!n.entries.length) return null;
              const b = mbrOf(n.entries);
              const d = pad(n.level) + 0.3;
              const on = mode === 'query' ? visited.has(n.id) : mode === 'knn' ? expandedOrder.has(n.id) : true;
              return (
                <rect
                  key={n.id}
                  x={X(b.x1 - d)}
                  y={X(b.y1 - d)}
                  width={(b.x2 - b.x1 + 2 * d) * PX}
                  height={(b.y2 - b.y1 + 2 * d) * PX}
                  fill="none"
                  stroke={levelColor(n.level)}
                  strokeWidth={on && mode !== 'draw' ? 2.2 : 1.3}
                  strokeOpacity={on ? 1 : 0.25}
                  strokeDasharray={mode === 'query' && fruitless.has(n.id) ? '5 3' : undefined}
                />
              );
            })}
          {mode === 'knn'
            ? order.map((n) => {
                const rank = expandedOrder.get(n.id);
                if (!rank || !n.entries.length || n.id === t.root) return null;
                const b = mbrOf(n.entries);
                const d = pad(n.level) + 0.3;
                const x = clamp(X(b.x1 - d), PAD, SIDE - PAD - 18);
                const y = clamp(X(b.y1 - d), PAD, SIDE - PAD - 14);
                return (
                  <g key={`r${n.id}`}>
                    <rect x={x} y={y} width={rank >= 10 ? 18 : 12} height={13} rx={2} fill="var(--viz-surface)" stroke={levelColor(n.level)} />
                    <text x={x + (rank >= 10 ? 9 : 6)} y={y + 10} textAnchor="middle" fontSize={10} fontWeight={600} fill="var(--viz-ink)">
                      {rank}
                    </text>
                  </g>
                );
              })
            : null}
          {mode === 'query' ? <rect x={X(liveWin.x1)} y={X(liveWin.y1)} width={(liveWin.x2 - liveWin.x1) * PX} height={(liveWin.y2 - liveWin.y1) * PX} fill="none" stroke="var(--viz-ink)" strokeWidth={2} strokeDasharray="6 4" /> : null}
          {mode === 'knn' ? (
            <g>
              {knnDone && Number.isFinite(kthDist) ? <circle cx={X(pt.x)} cy={X(pt.y)} r={Math.max(2, kthDist * PX)} fill="none" stroke="var(--viz-ink-2)" strokeDasharray="4 3" /> : null}
              <line x1={X(pt.x) - 7} x2={X(pt.x) + 7} y1={X(pt.y)} y2={X(pt.y)} stroke="var(--viz-ink)" strokeWidth={2} />
              <line y1={X(pt.y) - 7} y2={X(pt.y) + 7} x1={X(pt.x)} x2={X(pt.x)} stroke="var(--viz-ink)" strokeWidth={2} />
            </g>
          ) : null}
          {mode === 'draw' && dragRect ? <rect x={X(dragRect.x1)} y={X(dragRect.y1)} width={(dragRect.x2 - dragRect.x1) * PX} height={(dragRect.y2 - dragRect.y1) * PX} fill="var(--viz-1)" fillOpacity={0.15} stroke="var(--viz-ink)" strokeDasharray="3 2" /> : null}
        </svg>
        <p style={{ margin: '0.6rem 0 0.2rem', fontSize: '0.8rem', color: 'var(--viz-ink-2)' }}>
          The same tree as pages{mode === 'query' ? ' — filled pages were read, dashed ones matched nothing' : mode === 'knn' ? ' — numbers give the order pages were expanded' : ' — the filled leaf holds the newest rectangle'}. Numbers inside pages are entry counts.
        </p>
        <TreeDiagram t={t} visited={visited} fruitless={fruitless} expandedOrder={expandedOrder} mode={mode} newestLeaf={newestLeaf} />
      </TooltipHost>
      {mode === 'knn' ? (
        <div style={{ fontSize: '0.8rem', marginTop: '0.4rem', color: 'var(--viz-ink)' }}>
          <strong>Priority queue</strong> (smallest possible distance first){knnDone ? ' when the search stopped' : ''}:{' '}
          {lastStep && lastStep.queue.length
            ? lastStep.queue.map((it, i) => (
                <span key={i} style={{ marginRight: '0.8rem', whiteSpace: 'nowrap', display: 'inline-block' }}>
                  {it.kind === 'node' ? `${t.nodes[it.id].level === 0 ? 'leaf' : `L${t.nodes[it.id].level}`} page` : `rect #${it.id + 1}`} @ {d1(it.dist)}
                </span>
              ))
            : shown.length === 0
              ? `root page @ ${d1(knn.steps[0]?.pop.dist ?? 0)}`
              : 'empty'}
        </div>
      ) : null}
    </VizPanel>
  );
}
