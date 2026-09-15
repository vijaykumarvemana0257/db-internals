import { useMemo, useState, type ReactNode } from 'react';
import { VizPanel, Segmented, Slider, Button, Legend, Stats, Note, fmtBytes, fmtNum } from './Viz';

/**
 * Persistence by path copying, fat nodes and node copying (Driscoll, Sarnak, Sleator and Tarjan, JCSS 1989).
 *
 * One history of leaf updates is replayed in three representations of the same complete tree:
 * - Path copying: nodes are immutable pages. A commit copies the root-to-leaf path (one page per level), the copies keep
 *   every other child pointer, and the new root identifies the new version. Pages 0-1 stand for LMDB/bbolt's two meta
 *   pages; a copy takes the lowest free page, else extends the file. Branches (fully persistent) are modelled here only.
 * - Fat nodes: one node per position; an update appends a (version stamp, value) entry to the leaf. Internal pointers
 *   never change. Reading version v finds the largest stamp <= v: binary search (DSST) or a newest-first walk (the way
 *   an undo chain is read).
 * - Node copying: each node has `slots` extra fields. An update goes into a free slot; a full node is copied with its
 *   newest values and the pointer to the copy is stored in its parent the same way, which can cascade to the root.
 *   This is the variant that lets extra slots hold values as well as pointers.
 * Reclamation runs after every event. Retained versions: main head, branch head, an open reader, and the `keep` most
 * recent older main versions. "reach" frees whatever no retained root reaches (bbolt's releaseRange, Btrfs/ZFS/Git
 * semantics); "watermark" frees only what was superseded at or before the oldest retained version (LMDB, original Bolt,
 * the MVCC horizon). LMDB's writer is one commit more conservative than this model (mdb_find_oldest starts at txnid - 1).
 * Every node is one 4 KiB page; probes count entries examined, not I/Os.
 */
/* ------------------------------------------------------------------ model */

export type Rep = 'path' | 'fat' | 'copy';
export type Rule = 'reach' | 'watermark';
export type Line = 'main' | 'branch';
export type Search = 'binary' | 'walk';
export type Status = 'main' | 'branch' | 'pinned' | 'held';
export type Config = { rep: Rep; fanout: number; levels: number; keep: number; rule: Rule; slots: number; search: Search };
export type Ev = { t: 'commit'; leaf: number; on: Line } | { t: 'open' } | { t: 'close' } | { t: 'branch'; from: string } | { t: 'drop' };

/** Model assumption: every tree node is one fixed-size page, as in LMDB and bbolt (both use the OS page size). */
export const PAGE_BYTES = 4096;
/** Pages 0 and 1 hold the two meta pages in LMDB and bbolt; tree pages start at 2. */
export const META_PAGES = 2;
export const MAX_COMMITS = 60;

export const leavesOf = (fanout: number, levels: number) => fanout ** (levels - 1);
export const maxLevelsFor = (fanout: number) => (fanout === 2 ? 5 : 3);

/** Slot index at every level on the way from the root to `leaf`. */
export function pathSlots(fanout: number, levels: number, leaf: number) {
  const idx: number[] = new Array(levels);
  let j = leaf;
  for (let l = levels - 1; l >= 0; l--) {
    idx[l] = j;
    j = Math.floor(j / fanout);
  }
  return idx;
}

const why = (m: Map<string, string[]>, key: string | null, reason: string) => {
  if (!key) return;
  const r = m.get(key);
  if (r) {
    if (!r.includes(reason)) r.push(reason);
  } else m.set(key, [reason]);
};

/* ------------------------------------------------------- path copying */

export type PPage = { pg: number; level: number; idx: number; kids: number[]; seq: number; born: string; died: number | null };
export type PVersion = { key: string; line: Line; n: number; parent: string | null; root: number; leaf: number | null; written: number[] };

export type PathState = {
  kind: 'path';
  pages: Map<number, PPage>;
  free: number[];
  hwm: number;
  versions: PVersion[];
  mainHead: string;
  mainN: number;
  branchHead: string | null;
  branchFrom: string | null;
  branchN: number;
  reader: string | null;
  retained: Map<string, string[]>;
  status: Map<number, Status>;
  written: number;
  reused: number;
  lastWritten: number[];
  lastFreed: number[];
  lastEvent: string;
  ignored: number;
};

function reachPages(pages: Map<number, PPage>, root: number, out: Set<number>) {
  const stack = [root];
  while (stack.length) {
    const pg = stack.pop()!;
    if (out.has(pg)) continue;
    const p = pages.get(pg);
    if (!p) continue;
    out.add(pg);
    for (const k of p.kids) stack.push(k);
  }
  return out;
}

export function versionOf(s: { versions: PVersion[] }, key: string | null) {
  return key ? s.versions.find((v) => v.key === key) ?? null : null;
}

function pathRetained(s: PathState, keep: number) {
  const m = new Map<string, string[]>();
  why(m, s.mainHead, 'main head');
  why(m, s.branchHead, 'branch head');
  why(m, s.reader, 'reader');
  for (let n = s.mainN - 1; n >= Math.max(0, s.mainN - keep); n--) why(m, `v${n}`, 'kept');
  return m;
}

function pathGc(s: PathState, cfg: Config) {
  s.retained = pathRetained(s, cfg.keep);
  const mainSet = reachPages(s.pages, versionOf(s, s.mainHead)!.root, new Set());
  const branchSet = s.branchHead ? reachPages(s.pages, versionOf(s, s.branchHead)!.root, new Set()) : new Set<number>();
  const oldSet = new Set<number>();
  let oldest = s.mainN;
  for (const key of s.retained.keys()) {
    if (key === s.mainHead || key === s.branchHead) continue;
    const v = versionOf(s, key)!;
    reachPages(s.pages, v.root, oldSet);
    if (v.line === 'main') oldest = Math.min(oldest, v.n);
  }
  s.status = new Map();
  const freed: number[] = [];
  for (const p of s.pages.values()) {
    if (mainSet.has(p.pg)) s.status.set(p.pg, 'main');
    else if (branchSet.has(p.pg)) s.status.set(p.pg, 'branch');
    else if (oldSet.has(p.pg)) s.status.set(p.pg, 'pinned');
    else if (cfg.rule === 'reach' || (p.died !== null && p.died <= oldest)) freed.push(p.pg);
    else s.status.set(p.pg, 'held');
  }
  for (const pg of freed) s.pages.delete(pg);
  s.free = [...s.free, ...freed].sort((a, b) => a - b);
  return freed;
}

export function simulatePath(cfg: Config, events: Ev[]): PathState {
  const { fanout, levels } = cfg;
  const leaves = leavesOf(fanout, levels);
  let seq = 0;
  const s: PathState = {
    kind: 'path', pages: new Map(), free: [], hwm: META_PAGES, versions: [], mainHead: 'v0', mainN: 0, branchHead: null, branchFrom: null, branchN: 0,
    reader: null, retained: new Map(), status: new Map(), written: 0, reused: 0, lastWritten: [], lastFreed: [], lastEvent: 'v0: the initial tree', ignored: 0,
  };
  const alloc = () => {
    if (s.free.length) {
      s.reused++;
      return s.free.shift()!;
    }
    return s.hwm++;
  };
  // Initial build, level by level from the root.
  const byLevel: number[][] = [];
  for (let l = 0; l < levels; l++) {
    byLevel[l] = [];
    for (let j = 0; j < fanout ** l; j++) byLevel[l].push(alloc());
  }
  for (let l = 0; l < levels; l++) {
    byLevel[l].forEach((pg, j) => {
      const kids = l === levels - 1 ? [] : Array.from({ length: fanout }, (_, c) => byLevel[l + 1][j * fanout + c]);
      s.pages.set(pg, { pg, level: l, idx: j, kids, seq: seq++, born: 'v0', died: null });
    });
  }
  s.versions.push({ key: 'v0', line: 'main', n: 0, parent: null, root: byLevel[0][0], leaf: null, written: byLevel.flat() });
  pathGc(s, cfg);
  s.lastWritten = [];

  for (const ev of events) {
    if (ev.t === 'open') {
      s.reader = s.mainHead;
      s.lastWritten = [];
      s.lastFreed = pathGc(s, cfg);
      s.lastEvent = `Reader opened on ${s.mainHead}`;
    } else if (ev.t === 'close') {
      const was = s.reader;
      s.reader = null;
      s.lastWritten = [];
      s.lastFreed = pathGc(s, cfg);
      s.lastEvent = was ? `Reader on ${was} closed` : 'No reader to close';
    } else if (ev.t === 'branch') {
      if (cfg.rule !== 'reach' || s.branchHead || !s.retained.has(ev.from)) {
        s.ignored++;
        continue;
      }
      s.branchHead = ev.from;
      s.branchFrom = ev.from;
      s.lastWritten = [];
      s.lastFreed = pathGc(s, cfg);
      s.lastEvent = `Branch created from ${ev.from}`;
    } else if (ev.t === 'drop') {
      if (!s.branchHead) continue;
      s.branchHead = null;
      s.branchFrom = null;
      s.lastWritten = [];
      s.lastFreed = pathGc(s, cfg);
      s.lastEvent = 'Branch dropped';
    } else {
      const onBranch = ev.on === 'branch';
      if (onBranch && !s.branchHead) {
        s.ignored++;
        continue;
      }
      const leaf = ((ev.leaf % leaves) + leaves) % leaves;
      const head = versionOf(s, onBranch ? s.branchHead : s.mainHead)!;
      const slots = pathSlots(fanout, levels, leaf);
      const old: PPage[] = [];
      let cur = s.pages.get(head.root)!;
      old.push(cur);
      for (let l = 1; l < levels; l++) {
        cur = s.pages.get(cur.kids[slots[l] % fanout])!;
        old.push(cur);
      }
      const n = onBranch ? ++s.branchN : ++s.mainN;
      const key = onBranch ? `b${n}` : `v${n}`;
      const fresh = old.map(() => alloc()); // copies are allocated root first, on the way down
      old.forEach((o, l) => {
        const kids = o.kids.slice();
        if (l < levels - 1) kids[slots[l + 1] % fanout] = fresh[l + 1];
        s.pages.set(fresh[l], { pg: fresh[l], level: l, idx: o.idx, kids, seq: seq++, born: key, died: null });
        if (!onBranch && o.died === null) o.died = n;
      });
      s.versions.push({ key, line: ev.on, n, parent: head.key, root: fresh[0], leaf, written: fresh });
      if (onBranch) s.branchHead = key;
      else s.mainHead = key;
      s.written += fresh.length;
      s.lastWritten = fresh;
      s.lastFreed = pathGc(s, cfg);
      s.lastEvent = `${key} updated leaf ${leaf}`;
    }
  }
  return s;
}

/** Pages reachable from a version's root, and how many of them no other retained root reaches (ZFS `used`, Btrfs "exclusive"). */
export function referencedAndExclusive(s: PathState, key: string) {
  const v = versionOf(s, key);
  if (!v || !s.pages.has(v.root)) return { referenced: 0, exclusive: 0 };
  const mine = reachPages(s.pages, v.root, new Set());
  const others = new Set<number>();
  for (const k of s.retained.keys()) if (k !== key) reachPages(s.pages, versionOf(s, k)!.root, others);
  let exclusive = 0;
  for (const pg of mine) if (!others.has(pg)) exclusive++;
  return { referenced: mine.size, exclusive };
}

/** Walk from a version's root to a leaf: returns the pages visited (one probe each). */
export function pathLookup(s: PathState, key: string, leaf: number, fanout: number, levels: number) {
  const v = versionOf(s, key);
  if (!v) return null;
  const slots = pathSlots(fanout, levels, leaf);
  const visited: number[] = [];
  let p = s.pages.get(v.root);
  for (let l = 0; p && l < levels; l++) {
    visited.push(p.pg);
    if (l < levels - 1) p = s.pages.get(p.kids[slots[l + 1] % fanout]);
  }
  return { visited, probes: visited.length };
}

/* ------------------------------------------------------------ fat nodes */

export type FatState = {
  kind: 'fat';
  stamps: number[][]; // per leaf: version numbers of the value stamps still stored, ascending
  stampStatus: Map<string, Status>; // `${leaf}:${n}`
  mainN: number;
  reader: number | null;
  retained: Map<number, string[]>;
  written: number;
  pruned: number;
  lastLeaf: number | null;
  lastEvent: string;
  ignored: number;
  commitLeaf: (number | null)[]; // per version: which leaf it stamped
};

function fatGc(s: FatState, cfg: Config) {
  const m = new Map<number, string[]>();
  const add = (n: number | null, r: string) => {
    if (n === null) return;
    const x = m.get(n);
    if (x) x.push(r);
    else m.set(n, [r]);
  };
  add(s.mainN, 'main head');
  add(s.reader, 'reader');
  for (let n = s.mainN - 1; n >= Math.max(0, s.mainN - cfg.keep); n--) add(n, 'kept');
  s.retained = m;
  const ret = [...m.keys()];
  const old = ret.filter((n) => n !== s.mainN);
  const oldest = Math.min(...ret);
  s.stampStatus = new Map();
  s.stamps = s.stamps.map((list, leaf) => {
    const keepList: number[] = [];
    list.forEach((st, i) => {
      const next = i + 1 < list.length ? list[i + 1] : Infinity;
      if (next === Infinity) {
        keepList.push(st);
        s.stampStatus.set(`${leaf}:${st}`, 'main');
        return;
      }
      const needed = old.some((v) => v >= st && v < next);
      if (needed) {
        keepList.push(st);
        s.stampStatus.set(`${leaf}:${st}`, 'pinned');
      } else if (cfg.rule === 'watermark' && next > oldest) {
        keepList.push(st);
        s.stampStatus.set(`${leaf}:${st}`, 'held');
      } else s.pruned++;
    });
    return keepList;
  });
}

export function simulateFat(cfg: Config, events: Ev[]): FatState {
  const leaves = leavesOf(cfg.fanout, cfg.levels);
  const s: FatState = {
    kind: 'fat', stamps: Array.from({ length: leaves }, () => [0]), stampStatus: new Map(), mainN: 0, reader: null, retained: new Map(),
    written: 0, pruned: 0, lastLeaf: null, lastEvent: 'v0: the initial tree', ignored: 0, commitLeaf: [null],
  };
  fatGc(s, cfg);
  for (const ev of events) {
    if (ev.t === 'open') {
      s.reader = s.mainN;
      s.lastLeaf = null;
      s.lastEvent = `Reader opened on v${s.mainN}`;
    } else if (ev.t === 'close') {
      s.lastEvent = s.reader === null ? 'No reader to close' : `Reader on v${s.reader} closed`;
      s.reader = null;
      s.lastLeaf = null;
    } else if (ev.t === 'branch' || ev.t === 'drop' || ev.on === 'branch') {
      s.ignored++;
      continue;
    } else {
      const leaf = ((ev.leaf % leaves) + leaves) % leaves;
      s.mainN++;
      s.stamps[leaf].push(s.mainN);
      s.commitLeaf.push(leaf);
      s.written++;
      s.lastLeaf = leaf;
      s.lastEvent = `v${s.mainN} updated leaf ${leaf}`;
    }
    fatGc(s, cfg);
  }
  return s;
}

/** Probes to read `leaf` as of version v: one per internal fat node (their pointers never change), then a stamp search in the leaf. */
export function fatLookup(s: FatState, v: number, leaf: number, levels: number, search: Search) {
  const list = s.stamps[leaf];
  let found = list[0];
  for (const st of list) if (st <= v) found = st;
  const newer = list.filter((st) => st > v).length;
  const leafProbes = search === 'binary' ? Math.ceil(Math.log2(list.length + 1)) : newer + 1;
  return { found, leafProbes, probes: levels - 1 + leafProbes, stored: list.length };
}

/* ---------------------------------------------------------- node copying */

export type CMod = { field: number; n: number; target: number };
export type CNode = { id: number; level: number; idx: number; born: number; seq: number; kids: number[]; val: number; mods: CMod[]; copiedAt: number | null };

export type CopyState = {
  kind: 'copy';
  nodes: Map<number, CNode>;
  live: Map<string, number>;
  rootAt: number[];
  mainN: number;
  reader: number | null;
  retained: Map<number, string[]>;
  status: Map<number, Status>;
  touchedTotal: number;
  copiedTotal: number;
  modsTotal: number;
  lastTouched: number[];
  lastCopied: number[];
  lastEvent: string;
  ignored: number;
  history: { n: number; leaf: number; touched: number; copied: number; rootCopied: boolean }[];
};

const newestKid = (x: CNode, c: number) => {
  let t = x.kids[c];
  for (const m of x.mods) if (m.field === c) t = m.target;
  return t;
};
const newestVal = (x: CNode) => {
  let t = x.val;
  for (const m of x.mods) if (m.field === -1) t = m.target;
  return t;
};

/** Pick the field value valid at version v: the entry with the largest stamp <= v. Probes = every entry scanned (original + extra slots). */
function fieldAt(x: CNode, field: number, v: number) {
  let t = field === -1 ? x.val : x.kids[field];
  for (const m of x.mods) if (m.field === field && m.n <= v) t = m.target;
  return { value: t, probes: 1 + x.mods.length };
}

export function copyLookup(s: CopyState, v: number, leaf: number, fanout: number, levels: number) {
  const slots = pathSlots(fanout, levels, leaf);
  let node = s.nodes.get(s.rootAt[v]);
  const visited: number[] = [];
  let probes = 0;
  for (let l = 0; node && l < levels; l++) {
    visited.push(node.id);
    if (l < levels - 1) {
      const r = fieldAt(node, slots[l + 1] % fanout, v);
      probes += r.probes;
      node = s.nodes.get(r.value);
    } else {
      const r = fieldAt(node, -1, v);
      probes += r.probes;
      return { visited, probes, value: r.value };
    }
  }
  return { visited, probes, value: -1 };
}

function copyReach(s: CopyState, v: number, fanout: number, levels: number, out: Set<number>) {
  const walk = (id: number, l: number) => {
    const x = s.nodes.get(id);
    if (!x) return;
    out.add(id);
    if (l === levels - 1) return;
    for (let c = 0; c < fanout; c++) walk(fieldAt(x, c, v).value, l + 1);
  };
  walk(s.rootAt[v], 0);
  return out;
}

function copyGc(s: CopyState, cfg: Config) {
  const m = new Map<number, string[]>();
  const add = (n: number | null, r: string) => {
    if (n === null) return;
    const x = m.get(n);
    if (x) x.push(r);
    else m.set(n, [r]);
  };
  add(s.mainN, 'main head');
  add(s.reader, 'reader');
  for (let n = s.mainN - 1; n >= Math.max(0, s.mainN - cfg.keep); n--) add(n, 'kept');
  s.retained = m;
  const headSet = copyReach(s, s.mainN, cfg.fanout, cfg.levels, new Set());
  const oldSet = new Set<number>();
  let oldest = s.mainN;
  for (const n of m.keys()) {
    if (n === s.mainN) continue;
    copyReach(s, n, cfg.fanout, cfg.levels, oldSet);
    oldest = Math.min(oldest, n);
  }
  s.status = new Map();
  for (const x of [...s.nodes.values()]) {
    if (headSet.has(x.id)) s.status.set(x.id, 'main');
    else if (oldSet.has(x.id)) s.status.set(x.id, 'pinned');
    else if (cfg.rule === 'reach' || (x.copiedAt !== null && x.copiedAt <= oldest)) s.nodes.delete(x.id);
    else s.status.set(x.id, 'held');
  }
}

export function simulateCopy(cfg: Config, events: Ev[]): CopyState {
  const { fanout, levels, slots: cap } = cfg;
  const leaves = leavesOf(fanout, levels);
  let nextId = 0;
  let seq = 0;
  const s: CopyState = {
    kind: 'copy', nodes: new Map(), live: new Map(), rootAt: [], mainN: 0, reader: null, retained: new Map(), status: new Map(),
    touchedTotal: 0, copiedTotal: 0, modsTotal: 0, lastTouched: [], lastCopied: [], lastEvent: 'v0: the initial tree', ignored: 0, history: [],
  };
  const ids: number[][] = [];
  for (let l = 0; l < levels; l++) ids[l] = Array.from({ length: fanout ** l }, () => nextId++);
  for (let l = 0; l < levels; l++) {
    ids[l].forEach((id, j) => {
      const kids = l === levels - 1 ? [] : Array.from({ length: fanout }, (_, c) => ids[l + 1][j * fanout + c]);
      s.nodes.set(id, { id, level: l, idx: j, born: 0, seq: seq++, kids, val: 0, mods: [], copiedAt: null });
      s.live.set(`${l}:${j}`, id);
    });
  }
  s.rootAt[0] = ids[0][0];
  copyGc(s, cfg);

  for (const ev of events) {
    if (ev.t === 'open') {
      s.reader = s.mainN;
      s.lastTouched = [];
      s.lastCopied = [];
      s.lastEvent = `Reader opened on v${s.mainN}`;
    } else if (ev.t === 'close') {
      s.lastEvent = s.reader === null ? 'No reader to close' : `Reader on v${s.reader} closed`;
      s.reader = null;
      s.lastTouched = [];
      s.lastCopied = [];
    } else if (ev.t === 'branch' || ev.t === 'drop' || ev.on === 'branch') {
      s.ignored++;
      continue;
    } else {
      const leaf = ((ev.leaf % leaves) + leaves) % leaves;
      const n = ++s.mainN;
      const pos = pathSlots(fanout, levels, leaf);
      const touched: number[] = [];
      const copied: number[] = [];
      let rootCopied = false;
      // Store `target` in `field` of the live node at (level, idx); copy the node if its extra slots are full.
      const setField = (level: number, field: number, target: number) => {
        const x = s.nodes.get(s.live.get(`${level}:${pos[level]}`)!)!;
        if (x.mods.length < cap) {
          x.mods.push({ field, n, target });
          s.modsTotal++;
          if (!touched.includes(x.id)) touched.push(x.id);
          return;
        }
        const kids = x.kids.map((_, c) => newestKid(x, c));
        let val = newestVal(x);
        if (field === -1) val = target;
        else kids[field] = target;
        const y: CNode = { id: nextId++, level, idx: x.idx, born: n, seq: seq++, kids, val, mods: [], copiedAt: null };
        x.copiedAt = n;
        s.nodes.set(y.id, y);
        s.live.set(`${level}:${pos[level]}`, y.id);
        touched.push(y.id);
        copied.push(y.id);
        if (level === 0) rootCopied = true;
        else setField(level - 1, pos[level] % fanout, y.id);
      };
      setField(levels - 1, -1, n);
      s.rootAt[n] = s.live.get(`0:0`)!;
      s.touchedTotal += touched.length;
      s.copiedTotal += copied.length;
      s.lastTouched = touched;
      s.lastCopied = copied;
      s.history.push({ n, leaf, touched: touched.length, copied: copied.length, rootCopied });
      s.lastEvent = `v${n} updated leaf ${leaf}`;
    }
    copyGc(s, cfg);
  }
  return s;
}

export function simulate(cfg: Config, events: Ev[]) {
  return cfg.rep === 'path' ? simulatePath(cfg, events) : cfg.rep === 'fat' ? simulateFat(cfg, events) : simulateCopy(cfg, events);
}

/* --------------------------------------------------------------------- UI */

const W = 700;
const PAD = 10;
const GAP = 3;
const LEVEL_GAP = 34;
const MAX_ROWS = 6;
const STAMP_ROWS = 9;
const OVERFLOW_H = 10;

const STATUS_COLOR: Record<Status, string> = {
  main: 'var(--viz-clean)',
  branch: 'var(--viz-7)',
  pinned: 'var(--viz-warning)',
  held: 'var(--viz-5)',
};
/** Second channel besides color: watermark-held garbage is drawn with a dashed outline. */
const HELD_DASH = '5 2';
const STATUS_TEXT: Record<Status, string> = {
  main: 'reachable from the main head',
  branch: 'reachable from the branch head but not from main',
  pinned: 'reachable only from an older retained version',
  held: 'unreachable, but not yet past the watermark',
};

export const INITIAL_EVENTS: Ev[] = [{ t: 'commit', leaf: 5, on: 'main' }];

type Item = { id: number; level: number; idx: number; status: Status; label: string; inView: boolean; thick: boolean; tip: string; priority: number; seq: number; used?: number; cap?: number };
type Placed = Item & { x: number; y: number; w: number; h: number };

function layoutGrid(fanout: number, levels: number, items: Item[], boxH: number) {
  const treeW = W - 2 * PAD;
  const bySlot = new Map<string, Item[]>();
  for (const it of items) {
    const k = `${it.level}:${it.idx}`;
    const a = bySlot.get(k);
    if (a) a.push(it);
    else bySlot.set(k, [it]);
  }
  const placed = new Map<number, Placed>();
  const overflow: { x: number; y: number; text: string }[] = [];
  const levelY: number[] = [];
  const levelH: number[] = [];
  let y = 12;
  for (let l = 0; l < levels; l++) {
    const slotW = treeW / fanout ** l;
    const cols = Math.max(1, Math.floor((slotW - 4) / 34));
    let rows = 1;
    for (let j = 0; j < fanout ** l; j++) rows = Math.max(rows, Math.ceil(Math.min((bySlot.get(`${l}:${j}`) ?? []).length, cols * MAX_ROWS) / cols));
    levelY[l] = y;
    levelH[l] = rows * (boxH + GAP) - GAP;
    for (let j = 0; j < fanout ** l; j++) {
      const all = bySlot.get(`${l}:${j}`) ?? [];
      const visible = all
        .slice()
        .sort((a, b) => a.priority - b.priority || b.seq - a.seq)
        .slice(0, cols * MAX_ROWS)
        .sort((a, b) => a.seq - b.seq);
      const used = Math.max(1, Math.min(cols, visible.length));
      const bw = Math.min(30, (slotW - 4) / used - GAP);
      const x0 = PAD + j * slotW + (slotW - (used * (bw + GAP) - GAP)) / 2;
      visible.forEach((it, i) => {
        const r = Math.floor(i / cols);
        const c = i % cols;
        placed.set(it.id, { ...it, x: x0 + c * (bw + GAP), y: y + r * (boxH + GAP), w: bw, h: boxH });
      });
      if (all.length > visible.length) overflow.push({ x: PAD + j * slotW + slotW / 2, y: y + levelH[l] + 11, text: `+${all.length - visible.length}` });
    }
    y += levelH[l] + LEVEL_GAP;
  }
  // A "+N" count under the leaf row sits inside the selected leaf's outline, so both grow by OVERFLOW_H.
  const leafOverflow = overflow.some((o) => o.y > levelY[levels - 1] + levelH[levels - 1]) ? OVERFLOW_H : 0;
  return { placed, overflow, levelY, levelH, leafOverflow, height: y - LEVEL_GAP + 30 + leafOverflow };
}

function Chip({ label, tags, pressed, disabled, onClick }: { label: string; tags: string[]; pressed: boolean; disabled: boolean; onClick: () => void }) {
  return (
    <button
      type="button"
      onClick={onClick}
      disabled={disabled}
      aria-pressed={pressed}
      title={disabled ? 'Expired: no retained root, so this version can no longer be read' : 'Show this version’s tree'}
      style={{
        fontSize: '0.75rem',
        padding: '2px 8px',
        borderRadius: 999,
        border: `1.5px ${disabled ? 'dashed' : 'solid'} ${pressed ? 'var(--viz-ink)' : 'var(--viz-border)'}`,
        background: 'var(--viz-surface)',
        color: disabled ? 'var(--viz-ink-muted)' : 'var(--viz-ink)',
        fontWeight: pressed ? 600 : 400,
      }}
    >
      {label}
      {tags.length ? <span style={{ color: 'var(--viz-ink-2)', fontWeight: 400 }}> · {tags.join(', ')}</span> : null}
    </button>
  );
}

export default function PathCopyPersistenceLab() {
  const [rep, setRep] = useState<Rep>('path');
  const [fanout, setFanout] = useState(2);
  const [levelsRaw, setLevels] = useState(4);
  const [keep, setKeep] = useState(1);
  const [rule, setRule] = useState<Rule>('reach');
  const [slots, setSlots] = useState(1);
  const [search, setSearch] = useState<Search>('binary');
  const [leafRaw, setLeaf] = useState(5);
  const [events, setEvents] = useState<Ev[]>(INITIAL_EVENTS);
  const [on, setOn] = useState<Line>('main');
  const [view, setView] = useState<string | null>(null);

  const levels = Math.min(levelsRaw, maxLevelsFor(fanout));
  const leaves = leavesOf(fanout, levels);
  const leaf = leafRaw % leaves;
  const cfg: Config = { rep, fanout, levels, keep, rule, slots, search };
  const sim = useMemo(() => simulate(cfg, events), [rep, fanout, levels, keep, rule, slots, search, events]);
  const commits = events.filter((e) => e.t === 'commit').length;
  const push = (...evs: Ev[]) => setEvents((es) => [...es, ...evs]);
  const commit = (count: number) => {
    const room = Math.max(0, MAX_COMMITS - commits);
    const target: Line = rep === 'path' && sim.kind === 'path' && sim.branchHead ? on : 'main';
    push(...Array.from({ length: Math.min(count, room) }, () => ({ t: 'commit' as const, leaf, on: target })));
  };

  const readerOpen = sim.reader !== null;
  const branchExists = sim.kind === 'path' && sim.branchHead !== null;

  /* ---- version list, retained set, viewed version ---- */
  type Row = { key: string; tags: string[]; retained: boolean; label: string };
  let rows: Row[] = [];
  let headKey = 'v0';
  if (sim.kind === 'path') {
    headKey = sim.mainHead;
    rows = sim.versions.map((v) => {
      const r = sim.retained.get(v.key) ?? [];
      const tags = [...r];
      if (v.key === sim.branchFrom && sim.branchHead) tags.push('branch origin');
      return { key: v.key, tags, retained: r.length > 0, label: `${v.key} → p${v.root}` };
    });
  } else {
    headKey = `v${sim.mainN}`;
    rows = Array.from({ length: sim.mainN + 1 }, (_, n) => {
      const r = sim.retained.get(n) ?? [];
      return { key: `v${n}`, tags: [...r], retained: r.length > 0, label: sim.kind === 'copy' ? `v${n} → n${sim.rootAt[n]}` : `v${n}` };
    });
  }
  const retainedKeys = rows.filter((r) => r.retained).map((r) => r.key);
  const viewKey = view && retainedKeys.includes(view) ? view : headKey;
  const recentExpired = new Set(rows.filter((r) => !r.retained).slice(-3).map((r) => r.key));
  const shownRows = rows.filter((r) => r.retained || recentExpired.has(r.key));
  const hiddenRows = rows.length - shownRows.length;
  const oldKey = sim.reader !== null ? (sim.kind === 'path' ? sim.reader : `v${sim.reader}`) : retainedKeys.filter((k) => k !== headKey && k.startsWith('v')).sort((a, b) => Number(a.slice(1)) - Number(b.slice(1)))[0] ?? null;

  /* ---- figure ---- */
  let figure: ReactNode = null;
  let strip: ReactNode = null;
  let statsItems: { label: string; value: ReactNode; hint?: string }[] = [];
  let note: ReactNode = null;
  let table: ReactNode = null;

  const leafTargets = (levelY: number, levelH: number) =>
    Array.from({ length: leaves }, (_, j) => {
      const slotW = (W - 2 * PAD) / leaves;
      return (
        <rect
          key={`t${j}`}
          x={PAD + j * slotW + 1}
          y={levelY - 5}
          width={slotW - 2}
          height={levelH + 10}
          rx={4}
          fill="transparent"
          stroke={j === leaf ? 'var(--viz-ink-2)' : 'none'}
          strokeDasharray="4 3"
          style={{ cursor: 'pointer' }}
          onClick={() => setLeaf(j)}
        >
          <title>{`Leaf ${j}${j === leaf ? ' (selected)' : ': click to select'}`}</title>
        </rect>
      );
    });

  if (sim.kind === 'path') {
    const s = sim;
    const viewV = versionOf(s, viewKey)!;
    const inView = new Set<number>();
    const edges: [number, number][] = [];
    {
      const stack = [viewV.root];
      while (stack.length) {
        const pg = stack.pop()!;
        if (inView.has(pg)) continue;
        const p = s.pages.get(pg);
        if (!p) continue;
        inView.add(pg);
        for (const k of p.kids) {
          edges.push([pg, k]);
          stack.push(k);
        }
      }
    }
    const otherEdges: [number, number][] = [];
    {
      const seen = new Set<string>(edges.map(([a, b]) => `${a}>${b}`));
      for (const key of s.retained.keys()) {
        if (key === viewKey) continue;
        const stack = [versionOf(s, key)!.root];
        const done = new Set<number>();
        while (stack.length) {
          const pg = stack.pop()!;
          if (done.has(pg) || inView.has(pg)) continue;
          done.add(pg);
          const p = s.pages.get(pg);
          if (!p) continue;
          for (const k of p.kids) {
            const e = `${pg}>${k}`;
            if (!seen.has(e)) {
              seen.add(e);
              otherEdges.push([pg, k]);
            }
            stack.push(k);
          }
        }
      }
    }
    const lastW = new Set(s.lastWritten);
    const pr: Record<Status, number> = { main: 1, branch: 2, pinned: 3, held: 4 };
    const items: Item[] = [...s.pages.values()].map((p) => {
      const st = s.status.get(p.pg)!;
      return {
        id: p.pg, level: p.level, idx: p.idx, status: st, label: `p${p.pg}`, inView: inView.has(p.pg), thick: lastW.has(p.pg),
        tip: `Page ${p.pg}: written by ${p.born}; ${STATUS_TEXT[st]}${p.died !== null ? `; superseded on main by v${p.died}` : ''}`,
        priority: inView.has(p.pg) ? 0 : pr[st], seq: p.seq,
      };
    });
    const lay = layoutGrid(fanout, levels, items, 20);
    const H = lay.height;
    figure = (
      <svg viewBox={`0 0 ${W} ${H}`} width={W} height={H} role="img" aria-label={`Path-copied tree: version ${viewKey} has root page ${viewV.root}; ${s.pages.size} pages allocated, ${s.lastWritten.length} written by the last commit`}>
        {leafTargets(lay.levelY[levels - 1], lay.levelH[levels - 1] + lay.leafOverflow)}
        {otherEdges.map(([a, b], i) => {
          const pa = lay.placed.get(a);
          const pb = lay.placed.get(b);
          if (!pa || !pb) return null;
          return <line key={`x${i}`} x1={pa.x + pa.w / 2} y1={pa.y + pa.h} x2={pb.x + pb.w / 2} y2={pb.y} stroke="var(--viz-ink-muted)" strokeWidth={1} strokeDasharray="3 3" />;
        })}
        {edges.map(([a, b], i) => {
          const pa = lay.placed.get(a);
          const pb = lay.placed.get(b);
          if (!pa || !pb) return null;
          return <line key={i} x1={pa.x + pa.w / 2} y1={pa.y + pa.h} x2={pb.x + pb.w / 2} y2={pb.y} stroke="var(--viz-ink-2)" strokeWidth={1.2} />;
        })}
        {[...lay.placed.values()].map((b) => (
          <g key={b.id} opacity={b.inView ? 1 : 0.8} onClick={b.level === levels - 1 ? () => setLeaf(b.idx) : undefined} style={b.level === levels - 1 ? { cursor: 'pointer' } : undefined}>
            <title>{b.tip}</title>
            <rect x={b.x} y={b.y} width={b.w} height={b.h} rx={3} fill="var(--viz-surface)" stroke={STATUS_COLOR[b.status]} strokeWidth={b.thick ? 3 : 1.6} strokeDasharray={b.status === 'held' ? HELD_DASH : undefined} />
            <text x={b.x + b.w / 2} y={b.y + b.h / 2 + 3.5} textAnchor="middle" fontSize={9.5} fill="var(--viz-ink)" fontWeight={b.thick ? 700 : 400}>
              {b.label}
            </text>
          </g>
        ))}
        {lay.overflow.map((o, i) => (
          <text key={`o${i}`} x={o.x} y={o.y} textAnchor="middle" fontSize={10} fill="var(--viz-ink-2)">
            {o.text}
          </text>
        ))}
        <text x={PAD} y={H - 6} fontSize={11} fill="var(--viz-ink-2)">
          Solid edges: {viewKey}’s tree. Dashed: other retained versions. Thick outline: written by the last commit. Click a leaf to pick it.
        </text>
      </svg>
    );

    const cell = 14;
    const perRow = Math.floor((W - 2 * PAD) / (cell + 2));
    const total = s.hwm;
    const stripRows = Math.ceil(total / perRow);
    const SH = stripRows * (cell + 2) + 22;
    const freeSet = new Set(s.free);
    strip = (
      <svg viewBox={`0 0 ${W} ${SH}`} width={W} height={SH} role="img" aria-label={`Database file: ${total} pages, ${s.free.length} free`} style={{ marginTop: 6 }}>
        <text x={PAD} y={12} fontSize={11} fill="var(--viz-ink)">
          The file, page by page: {total} pages = {fmtBytes(total * PAGE_BYTES)}, {s.free.length} free (pages 0–1 are the meta pages)
        </text>
        {Array.from({ length: total }, (_, pg) => {
          const x = PAD + (pg % perRow) * (cell + 2);
          const y = 18 + Math.floor(pg / perRow) * (cell + 2);
          const st = s.status.get(pg);
          const meta = pg < META_PAGES;
          const isFree = freeSet.has(pg);
          return (
            <rect key={pg} x={x} y={y} width={cell} height={cell} rx={2} fill={meta ? 'var(--viz-neutral)' : 'var(--viz-surface)'} stroke={meta ? 'var(--viz-ink-2)' : isFree ? 'var(--viz-stale)' : st ? STATUS_COLOR[st] : 'var(--viz-stale)'} strokeWidth={isFree ? 1 : 1.8} strokeDasharray={isFree ? '2 2' : st === 'held' ? HELD_DASH : undefined}>
              <title>{meta ? `Page ${pg}: meta page` : isFree ? `Page ${pg}: free, reusable by the next commit` : `Page ${pg}: ${st ? STATUS_TEXT[st] : ''}`}</title>
            </rect>
          );
        })}
      </svg>
    );

    const count = (st: Status) => [...s.status.values()].filter((x) => x === st).length;
    const pinned = count('pinned');
    const held = count('held');
    const branchOnly = count('branch');
    const branchOwn = [...s.pages.values()].filter((p) => s.status.get(p.pg) === 'branch' && p.born.startsWith('b')).length;
    const liveHead = count('main');
    const readerOnHead = s.reader !== null && (s.reader === s.mainHead || s.reader === s.branchHead);
    const readerOnly = s.reader && !readerOnHead ? referencedAndExclusive(s, s.reader).exclusive : 0;
    const oldestKept = s.versions.filter((v) => v.line === 'main' && s.retained.has(v.key))[0]?.key ?? s.mainHead;
    const rHead = pathLookup(s, s.mainHead, leaf, fanout, levels)!;
    const rOld = oldKey ? pathLookup(s, oldKey, leaf, fanout, levels) : null;
    statsItems = [
      { label: 'Last commit wrote', value: s.lastWritten.length ? `${s.lastWritten.length} pages` : '—', hint: `One page per level (${fmtBytes(levels * PAGE_BYTES)} at 4 KiB pages): proportional to depth, not to tree size.` },
      { label: 'Bytes written by commits', value: fmtBytes(s.written * PAGE_BYTES), hint: `${s.written} pages since the initial load` },
      { label: 'Live in main head', value: `${liveHead} pages` },
      rule === 'watermark'
        ? { label: 'Pinned · held back', value: `${pinned} · ${held}`, hint: 'Pinned: only an older retained root reaches them. Held back: unreachable, but superseded after the oldest retained version, so the watermark rule keeps them.' }
        : { label: branchExists ? 'Pinned · branch, not main' : 'Pinned by old versions', value: branchExists ? `${pinned} · ${branchOnly}` : `${pinned} pages`, hint: branchExists ? 'Pinned: only an older retained root reaches them. Branch, not main: the branch head reaches them and the main head does not.' : 'Pages that only an older retained root (a reader or a kept version) still reaches.' },
      { label: 'File size', value: fmtBytes(s.hwm * PAGE_BYTES), hint: `${s.hwm} pages (the high-water mark), ${s.free.length} of them free. Freed pages are reused, but the file does not shrink.` },
      { label: `Read leaf ${leaf}: head${rOld ? ` / ${oldKey}` : ''}`, value: `${rHead.probes}${rOld ? ` / ${rOld.probes}` : ''} probes`, hint: 'One node per level in every version: reading an old version costs the same, it just starts from a different root.' },
    ];

    const last = s.versions[s.versions.length - 1];
    const lastIsCommit = s.lastWritten.length > 0 && last.key === s.lastEvent.split(' ')[0];
    const sharedInLast = lastIsCommit ? (referencedAndExclusive(s, last.key).referenced || 0) - s.lastWritten.length : 0;
    note = (
      <Note>
        {lastIsCommit ? (
          <>
            <strong>
              {last.key} updated leaf {last.leaf}: copied {s.lastWritten.length} pages, root to leaf (p{s.lastWritten.join(', p')}).
            </strong>{' '}
            Every other pointer in those copies still names {last.parent}’s pages, so {sharedInLast} of {last.key}’s {sharedInLast + s.lastWritten.length} pages are shared. {last.key} exists because its root, p{last.root}, is recorded as the newest root.{' '}
          </>
        ) : (
          <>
            <strong>{s.lastEvent}.</strong>{' '}
          </>
        )}
        {s.lastFreed.length ? `Reclaimed ${s.lastFreed.length} page${s.lastFreed.length === 1 ? '' : 's'} that no retained root reaches any more. ` : ''}
        {readerOpen ? `${readerOnHead ? `The reader holds ${s.reader}, which is still a head, so it pins nothing of its own yet` : `The reader holds ${s.reader}, whose root is the only retained root that reaches ${readerOnly} page${readerOnly === 1 ? '' : 's'}`}${rule === 'watermark' ? `; the watermark also holds back ${held} unreachable page${held === 1 ? '' : 's'} superseded after the oldest retained version, ${oldestKept}` : ''}. ` : ''}
        {branchExists ? `${branchOnly} page${branchOnly === 1 ? ' is' : 's are'} reachable from the branch but not from main: ${branchOwn} it wrote itself where it diverged, and ${branchOnly - branchOwn} it still shares with its origin ${s.branchFrom} that main has since replaced. ` : ''}
        {s.ignored ? `(${s.ignored} branch action${s.ignored === 1 ? '' : 's'} ignored: branches need reachability reclamation.)` : ''}
      </Note>
    );

    table = (
      <table className="viz-table">
        <thead>
          <tr>
            <th>Version</th>
            <th>Root page</th>
            <th>Parent</th>
            <th>Leaf updated</th>
            <th>Pages written</th>
            <th>Retained because</th>
            <th>Referenced</th>
            <th>Exclusive</th>
          </tr>
        </thead>
        <tbody>
          {s.versions.slice(-14).map((v) => {
            const r = s.retained.get(v.key);
            const re = r ? referencedAndExclusive(s, v.key) : null;
            return (
              <tr key={v.key}>
                <td>{v.key}</td>
                <td>p{v.root}</td>
                <td>{v.parent ?? '—'}</td>
                <td>{v.leaf ?? 'initial load'}</td>
                <td>{v.written.length}</td>
                <td>{r ? r.join(', ') : 'expired'}</td>
                <td>{re ? re.referenced : '—'}</td>
                <td>{re ? re.exclusive : '—'}</td>
              </tr>
            );
          })}
        </tbody>
      </table>
    );
  } else if (sim.kind === 'fat') {
    const s = sim;
    const viewN = Number(viewKey.slice(1));
    const treeW = W - 2 * PAD;
    const boxH = 20;
    const rowY = (l: number) => 12 + l * (boxH + LEVEL_GAP);
    const leafY = rowY(levels - 1);
    const stampY0 = leafY + boxH + 8;
    const maxStored = Math.max(...s.stamps.map((x) => x.length));
    const shownStamps = Math.min(STAMP_ROWS, maxStored);
    const H = stampY0 + shownStamps * 17 + (maxStored > STAMP_ROWS ? 14 : 0) + 26;
    const slotsOnPath = pathSlots(fanout, levels, leaf);
    const oldN = oldKey ? Number(oldKey.slice(1)) : null;
    const nodes: ReactNode[] = [];
    const lines: ReactNode[] = [];
    for (let l = 0; l < levels; l++) {
      const slotW = treeW / fanout ** l;
      for (let j = 0; j < fanout ** l; j++) {
        const cx = PAD + j * slotW + slotW / 2;
        const bw = Math.min(30, slotW - 6);
        const onPath = slotsOnPath[l] === j;
        if (l < levels - 1) {
          for (let c = 0; c < fanout; c++) {
            const childW = treeW / fanout ** (l + 1);
            const ccx = PAD + (j * fanout + c) * childW + childW / 2;
            const hot = onPath && slotsOnPath[l + 1] === j * fanout + c;
            lines.push(<line key={`e${l}-${j}-${c}`} x1={cx} y1={rowY(l) + boxH} x2={ccx} y2={rowY(l + 1)} stroke={hot ? 'var(--viz-ink)' : 'var(--viz-ink-muted)'} strokeOpacity={hot ? 1 : 0.5} strokeWidth={hot ? 2 : 1} />);
          }
        }
        const isLeaf = l === levels - 1;
        const list = isLeaf ? s.stamps[j] : [];
        nodes.push(
          <g key={`n${l}-${j}`} onClick={isLeaf ? () => setLeaf(j) : undefined} style={isLeaf ? { cursor: 'pointer' } : undefined}>
            <title>{isLeaf ? `Leaf ${j}: ${list.length} value stamp${list.length === 1 ? '' : 's'} stored (${list.map((x) => `v${x}`).join(', ')})` : 'Internal fat node: its child pointers never change, so it carries no extra stamps'}</title>
            <rect x={cx - bw / 2} y={rowY(l)} width={bw} height={boxH} rx={3} fill="var(--viz-surface)" stroke="var(--viz-clean)" strokeWidth={isLeaf && s.lastLeaf === j ? 3 : 1.6} />
            <text x={cx} y={rowY(l) + boxH / 2 + 3.5} textAnchor="middle" fontSize={9.5} fill="var(--viz-ink)">
              {isLeaf ? `L${j}` : l === 0 ? 'root' : ''}
            </text>
          </g>,
        );
        if (isLeaf) {
          // Keep every stamp a retained version needs visible; fill the remaining rows with the newest held-back ones.
          const needed = list.filter((st) => s.stampStatus.get(`${j}:${st}`) !== 'held');
          const heldShown = list.filter((st) => s.stampStatus.get(`${j}:${st}`) === 'held').slice(-Math.max(0, STAMP_ROWS - needed.length));
          const showList = [...needed, ...heldShown].sort((a, b) => a - b).slice(-STAMP_ROWS);
          const hidden = list.length - showList.length;
          const resolved = fatLookup(s, viewN, j, levels, search).found;
          showList.forEach((st, i) => {
            const status = s.stampStatus.get(`${j}:${st}`) ?? 'main';
            const y = stampY0 + i * 17;
            const isResolved = j === leaf && st === resolved;
            nodes.push(
              <g key={`s${j}-${st}`}>
                <title>{`Leaf ${j}, value written by v${st}: ${status === 'main' ? 'the current value' : STATUS_TEXT[status]}`}</title>
                <rect x={cx - bw / 2} y={y} width={bw} height={14} rx={2} fill="var(--viz-surface)" stroke={STATUS_COLOR[status]} strokeWidth={isResolved ? 2.6 : 1.3} strokeDasharray={status === 'held' ? HELD_DASH : undefined} />
                <text x={cx} y={y + 10.5} textAnchor="middle" fontSize={8.5} fill="var(--viz-ink)" fontWeight={isResolved ? 700 : 400}>
                  v{st}
                </text>
              </g>,
            );
          });
          if (hidden > 0)
            nodes.push(
              <text key={`h${j}`} x={cx} y={stampY0 + showList.length * 17 + 9} textAnchor="middle" fontSize={9.5} fill="var(--viz-ink-2)">
                +{hidden} held
              </text>,
            );
        }
      }
    }
    figure = (
      <svg viewBox={`0 0 ${W} ${H}`} width={W} height={H} role="img" aria-label={`Fat-node tree: leaf ${leaf} stores ${s.stamps[leaf].length} version stamps`}>
        {leafTargets(leafY, stampY0 - leafY + shownStamps * 17 + (maxStored > STAMP_ROWS ? OVERFLOW_H : 0))}
        {lines}
        {nodes}
        <text x={PAD} y={H - 6} fontSize={11} fill="var(--viz-ink-2)">
          One node per position. Stamps under each leaf: the version that wrote each stored value. Bold stamp: what {viewKey} reads for leaf {leaf}.
        </text>
      </svg>
    );
    const head = fatLookup(s, s.mainN, leaf, levels, search);
    const old = oldN !== null ? fatLookup(s, oldN, leaf, levels, search) : null;
    const stored = s.stamps.reduce((a, x) => a + x.length, 0);
    const pinned = [...s.stampStatus.values()].filter((x) => x === 'pinned').length;
    const held = [...s.stampStatus.values()].filter((x) => x === 'held').length;
    statsItems = [
      { label: 'Last commit wrote', value: s.lastLeaf !== null ? '1 stamp in 1 node' : '—', hint: 'A fat node records the change where it happened; no ancestor is touched.' },
      { label: 'Stamps stored', value: fmtNum(stored), hint: `One current value per leaf (${leaves}), plus the older stamps a retained version${rule === 'watermark' ? ' or the watermark' : ''} still holds.` },
      { label: rule === 'watermark' ? 'Pinned · held back' : 'Pinned by old versions', value: rule === 'watermark' ? `${pinned} · ${held}` : fmtNum(pinned) },
      { label: `Stamps on leaf ${leaf}`, value: fmtNum(s.stamps[leaf].length) },
      { label: `Read leaf ${leaf} at head`, value: `${head.probes} probes`, hint: `${levels - 1} internal nodes + ${head.leafProbes} stamp comparison${head.leafProbes === 1 ? '' : 's'}` },
      { label: old ? `Read leaf ${leaf} at ${oldKey}` : 'Read at an old version', value: old ? `${old.probes} probes` : '—', hint: 'Open a reader or keep old versions to compare.' },
    ];
    note = (
      <Note>
        <strong>{s.lastLeaf !== null ? `${s.lastEvent}: one stamp (v${s.mainN}) appended to leaf ${s.lastLeaf}, and nothing else written.` : `${s.lastEvent}.`}</strong>{' '}
        Reading leaf {leaf} costs {levels - 1} pointer hops plus {head.leafProbes} stamp {search === 'binary' ? 'comparison' : 'check'}{head.leafProbes === 1 ? '' : 's'} at head
        {old ? ` and ${old.leafProbes} at ${oldKey}` : ''}
        {search === 'binary' ? ' — binary search over the stamps, so the cost grows with the log of how many are stored.' : ' — walking newest-first, so an old snapshot pays one check for every newer value, like a long undo chain.'}{' '}
        {s.ignored ? `(${s.ignored} branch action${s.ignored === 1 ? '' : 's'} ignored: this view replays main’s linear history.)` : ''}
      </Note>
    );
    table = (
      <table className="viz-table">
        <thead>
          <tr>
            <th>Leaf</th>
            <th>Stamps stored</th>
            <th>Read at head (binary / walk)</th>
            <th>{old ? `Read at ${oldKey} (binary / walk)` : 'Read at an old version'}</th>
          </tr>
        </thead>
        <tbody>
          {s.stamps.map((list, j) => (
            <tr key={j}>
              <td>{j}</td>
              <td>{list.map((x) => `v${x}`).join(' ')}</td>
              <td>
                {fatLookup(s, s.mainN, j, levels, 'binary').probes} / {fatLookup(s, s.mainN, j, levels, 'walk').probes}
              </td>
              <td>{oldN !== null ? `${fatLookup(s, oldN, j, levels, 'binary').probes} / ${fatLookup(s, oldN, j, levels, 'walk').probes}` : '—'}</td>
            </tr>
          ))}
        </tbody>
      </table>
    );
  } else {
    const s = sim;
    const viewN = Number(viewKey.slice(1));
    const inView = new Set<number>();
    const edges: [number, number][] = [];
    const walk = (id: number, l: number) => {
      const x = s.nodes.get(id);
      if (!x || inView.has(id)) return;
      inView.add(id);
      if (l === levels - 1) return;
      for (let c = 0; c < fanout; c++) {
        let t = x.kids[c];
        for (const m of x.mods) if (m.field === c && m.n <= viewN) t = m.target;
        edges.push([id, t]);
        walk(t, l + 1);
      }
    };
    walk(s.rootAt[viewN], 0);
    const otherEdges: [number, number][] = [];
    {
      const seen = new Set<string>(edges.map(([a, b]) => `${a}>${b}`));
      for (const n of s.retained.keys()) {
        if (n === viewN) continue;
        const go = (id: number, l: number) => {
          const x = s.nodes.get(id);
          if (!x || l === levels - 1) return;
          for (let c = 0; c < fanout; c++) {
            let t = x.kids[c];
            for (const m of x.mods) if (m.field === c && m.n <= n) t = m.target;
            const e = `${id}>${t}`;
            if (!seen.has(e)) {
              seen.add(e);
              otherEdges.push([id, t]);
            }
            go(t, l + 1);
          }
        };
        go(s.rootAt[n], 0);
      }
    }
    const lastT = new Set(s.lastTouched);
    const pr: Record<Status, number> = { main: 1, branch: 2, pinned: 3, held: 4 };
    const items: Item[] = [...s.nodes.values()].map((x) => {
      const st = s.status.get(x.id)!;
      return {
        id: x.id, level: x.level, idx: x.idx, status: st, label: `n${x.id}`, inView: inView.has(x.id), thick: lastT.has(x.id),
        tip: `Node ${x.id}: created by v${x.born}${x.copiedAt !== null ? `, copied at v${x.copiedAt}` : ''}; extra slots: ${x.mods.length ? x.mods.map((m) => `${m.field === -1 ? 'value' : `child ${m.field} → n${m.target}`} @v${m.n}`).join('; ') : 'empty'}; ${STATUS_TEXT[st]}`,
        priority: inView.has(x.id) ? 0 : pr[st], seq: x.seq, used: x.mods.length, cap: slots,
      };
    });
    const lay = layoutGrid(fanout, levels, items, 28);
    const H = lay.height;
    figure = (
      <svg viewBox={`0 0 ${W} ${H}`} width={W} height={H} role="img" aria-label={`Node-copying tree with ${slots} extra slot${slots === 1 ? '' : 's'} per node: ${s.nodes.size} persistent nodes`}>
        {leafTargets(lay.levelY[levels - 1], lay.levelH[levels - 1] + lay.leafOverflow)}
        {otherEdges.map(([a, b], i) => {
          const pa = lay.placed.get(a);
          const pb = lay.placed.get(b);
          if (!pa || !pb) return null;
          return <line key={`x${i}`} x1={pa.x + pa.w / 2} y1={pa.y + pa.h} x2={pb.x + pb.w / 2} y2={pb.y} stroke="var(--viz-ink-muted)" strokeWidth={1} strokeDasharray="3 3" />;
        })}
        {edges.map(([a, b], i) => {
          const pa = lay.placed.get(a);
          const pb = lay.placed.get(b);
          if (!pa || !pb) return null;
          return <line key={i} x1={pa.x + pa.w / 2} y1={pa.y + pa.h} x2={pb.x + pb.w / 2} y2={pb.y} stroke="var(--viz-ink-2)" strokeWidth={1.2} />;
        })}
        {[...lay.placed.values()].map((b) => (
          <g key={b.id} opacity={b.inView ? 1 : 0.8} onClick={b.level === levels - 1 ? () => setLeaf(b.idx) : undefined} style={b.level === levels - 1 ? { cursor: 'pointer' } : undefined}>
            <title>{b.tip}</title>
            <rect x={b.x} y={b.y} width={b.w} height={b.h} rx={3} fill="var(--viz-surface)" stroke={STATUS_COLOR[b.status]} strokeWidth={b.thick ? 3 : 1.6} strokeDasharray={b.status === 'held' ? HELD_DASH : undefined} />
            <text x={b.x + b.w / 2} y={b.y + 11} textAnchor="middle" fontSize={9.5} fill="var(--viz-ink)" fontWeight={b.thick ? 700 : 400}>
              {b.label}
            </text>
            {Array.from({ length: b.cap ?? 0 }, (_, i) => {
              const cap = b.cap ?? 1;
              const dx = b.x + b.w / 2 + (i - (cap - 1) / 2) * 8;
              return <circle key={i} cx={dx} cy={b.y + 21} r={2.6} fill={i < (b.used ?? 0) ? 'var(--viz-ink-2)' : 'var(--viz-surface)'} stroke="var(--viz-ink-2)" strokeWidth={1} />;
            })}
          </g>
        ))}
        {lay.overflow.map((o, i) => (
          <text key={`o${i}`} x={o.x} y={o.y} textAnchor="middle" fontSize={10} fill="var(--viz-ink-2)">
            {o.text}
          </text>
        ))}
        <text x={PAD} y={H - 6} fontSize={11} fill="var(--viz-ink-2)">
          Dots: extra slots in use. Solid edges: {viewKey}’s tree; dashed: other retained versions. Thick: written by the last commit.
        </text>
      </svg>
    );
    const head = copyLookup(s, s.mainN, leaf, fanout, levels);
    const oldN = oldKey ? Number(oldKey.slice(1)) : null;
    const old = oldN !== null ? copyLookup(s, oldN, leaf, fanout, levels) : null;
    const pinned = [...s.status.values()].filter((x) => x === 'pinned').length;
    const held = [...s.status.values()].filter((x) => x === 'held').length;
    const lastH = s.history[s.history.length - 1];
    const avg = s.history.length ? s.touchedTotal / s.history.length : 0;
    statsItems = [
      { label: 'Last commit: written · new', value: s.lastTouched.length ? `${s.lastTouched.length} · ${s.lastCopied.length}` : '—', hint: 'Nodes the last commit wrote, and how many of them are new copies.' },
      { label: 'Nodes written per commit', value: s.history.length ? fmtNum(avg, 2) : '—', hint: 'Average over all commits: amortized O(1) in DSST, against one page per level for path copying.' },
      { label: 'Persistent nodes', value: fmtNum(s.nodes.size) },
      { label: rule === 'watermark' ? 'Pinned · held back' : 'Pinned by old versions', value: rule === 'watermark' ? `${pinned} · ${held}` : fmtNum(pinned) },
      { label: `Read leaf ${leaf}: head${old ? ` / ${oldKey}` : ''}`, value: `${head.probes}${old ? ` / ${old.probes}` : ''} probes`, hint: `At most ${slots + 1} entries per node, whatever the history length.` },
    ];
    note = (
      <Note>
        {lastH && s.lastTouched.length ? (
          <strong>
            v{lastH.n} updated leaf {lastH.leaf}:{' '}
            {lastH.copied === 0 ? 'the leaf had a free slot, so the new value went there — one node written, no copy.' : `${lastH.copied} full node${lastH.copied === 1 ? ' was' : 's were'} copied with newest values${lastH.rootCopied ? ', all the way to a new root' : ', and the pointer to the last copy fit in a free slot of its parent'}.`}
          </strong>
        ) : (
          <strong>{s.lastEvent}.</strong>
        )}{' '}
        Every read scans at most {slots + 1} entries per node, however long the history. {s.ignored ? `(${s.ignored} branch action${s.ignored === 1 ? '' : 's'} ignored: node copying here is partially persistent.)` : ''}
      </Note>
    );
    table = (
      <table className="viz-table">
        <thead>
          <tr>
            <th>Version</th>
            <th>Root node</th>
            <th>Leaf updated</th>
            <th>Nodes written</th>
            <th>Nodes copied</th>
            <th>Root copied</th>
          </tr>
        </thead>
        <tbody>
          {s.history.slice(-14).map((h) => (
            <tr key={h.n}>
              <td>v{h.n}</td>
              <td>n{s.rootAt[h.n]}</td>
              <td>{h.leaf}</td>
              <td>{h.touched}</td>
              <td>{h.copied}</td>
              <td>{h.rootCopied ? 'yes' : 'no'}</td>
            </tr>
          ))}
        </tbody>
      </table>
    );
  }

  const legend =
    rep === 'fat' ? (
      <Legend
        items={[
          { label: 'Current value / fat node', color: STATUS_COLOR.main },
          { label: 'Stamp pinned by an older retained version', color: STATUS_COLOR.pinned },
          ...(rule === 'watermark' ? [{ label: 'Unneeded, held by the watermark (dashed)', color: STATUS_COLOR.held }] : []),
        ]}
      />
    ) : (
      <Legend
        items={[
          { label: 'Reachable from main head', color: STATUS_COLOR.main },
          ...(branchExists ? [{ label: 'Branch reaches it, main does not', color: STATUS_COLOR.branch }] : []),
          { label: 'Only an older retained version reaches it', color: STATUS_COLOR.pinned },
          ...(rule === 'watermark' ? [{ label: 'Unreachable, held by the watermark (dashed)', color: STATUS_COLOR.held }] : []),
          ...(rep === 'path' ? [{ label: 'Free page (thin, dotted)', color: 'var(--viz-stale)' }] : []),
        ]}
      />
    );

  const reset = () => {
    setEvents([]);
    setView(null);
    setOn('main');
  };

  return (
    <VizPanel
      title="One tree, many versions"
      subtitle="Commit updates to leaves and every version stays readable. Path copying shares untouched subtrees and names each version by its root; fat nodes keep the history inside the nodes. Hold a reader or keep old versions and watch what cannot be reclaimed."
      controls={
        <>
          <Segmented
            label="Representation"
            value={rep}
            onChange={(v) => {
              setRep(v);
              setView(null);
            }}
            options={[
              { value: 'path', label: 'Path copying' },
              { value: 'fat', label: 'Fat nodes' },
              { value: 'copy', label: 'Node copying' },
            ]}
          />
          <Segmented label="Fanout" value={String(fanout) as '2' | '3' | '4'} onChange={(v) => setFanout(Number(v))} options={[{ value: '2', label: '2' }, { value: '3', label: '3' }, { value: '4', label: '4' }]} />
          <Slider label="Levels" min={2} max={maxLevelsFor(fanout)} value={levels} onChange={setLevels} />
          <Slider label="Keep old versions" min={0} max={4} value={keep} onChange={setKeep} />
          <Segmented
            label="Reclaim by"
            value={rule}
            onChange={setRule}
            options={[
              { value: 'reach', label: 'Reachability', title: 'Free what no retained root reaches (bbolt, Btrfs, ZFS, Git, Iceberg)' },
              { value: 'watermark', label: 'Oldest-version watermark', title: 'Free only what was superseded at or before the oldest retained version (LMDB, MVCC horizons)' },
            ]}
          />
          {rep === 'copy' ? <Slider label="Extra slots per node" min={1} max={3} value={slots} onChange={setSlots} /> : null}
          {rep === 'fat' ? (
            <Segmented
              label="Find the stamp"
              value={search}
              onChange={setSearch}
              options={[
                { value: 'binary', label: 'Binary search' },
                { value: 'walk', label: 'Walk newest-first' },
              ]}
            />
          ) : null}
        </>
      }
      legend={legend}
      stats={<Stats items={statsItems} />}
      note={note}
      table={table}
    >
      <div className="viz-controls">
        <Slider label="Leaf to update" min={0} max={leaves - 1} value={leaf} onChange={setLeaf} />
        <Button primary onClick={() => commit(1)} disabled={commits >= MAX_COMMITS}>
          Commit: update leaf {leaf}
        </Button>
        <Button onClick={() => commit(5)} disabled={commits >= MAX_COMMITS}>
          Commit ×5
        </Button>
        {readerOpen ? <Button onClick={() => push({ t: 'close' })}>Close reader</Button> : <Button onClick={() => push({ t: 'open' })}>Open reader on head</Button>}
        {rep === 'path' ? (
          branchExists ? (
            <>
              <Segmented label="Commit to" value={on} onChange={setOn} options={[{ value: 'main', label: 'main' }, { value: 'branch', label: 'branch' }]} />
              <Button
                onClick={() => {
                  push({ t: 'drop' });
                  setOn('main');
                }}
              >
                Drop branch
              </Button>
            </>
          ) : (
            <Button
              onClick={() => {
                push({ t: 'branch', from: viewKey });
                setOn('branch');
              }}
              disabled={rule !== 'reach'}
              title={rule !== 'reach' ? 'A single watermark cannot describe a branching history: switch to reachability' : `Create a writable branch whose root is ${viewKey}’s root`}
            >
              Branch from {viewKey}
            </Button>
          )
        ) : null}
        <Button onClick={reset}>Reset</Button>
      </div>
      <div style={{ display: 'flex', flexWrap: 'wrap', gap: 4, margin: '0 0 6px', alignItems: 'center' }}>
        <span style={{ fontSize: '0.75rem', color: 'var(--viz-ink-2)', marginRight: 4 }}>Versions{hiddenRows ? ` (${hiddenRows} older expired)` : ''}:</span>
        {shownRows.map((r) => (
          <Chip key={r.key} label={r.label} tags={r.retained ? r.tags : ['expired']} pressed={r.key === viewKey} disabled={!r.retained} onClick={() => setView(r.key)} />
        ))}
      </div>
      {figure}
      {strip}
    </VizPanel>
  );
}
