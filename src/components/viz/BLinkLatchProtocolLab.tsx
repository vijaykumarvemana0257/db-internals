import { useMemo, useState } from 'react';
import { VizPanel, Segmented, Choice, Check, Button, Legend, Stats, Note } from './Viz';

/**
 * Two threads — an inserter and a reader — descend one small B+tree under four latch protocols.
 *
 * Model assumptions (labelled where the reader sees them):
 * - A fixed three-level tree, at most 3 keys per page. Separators are inclusive upper bounds, the Lehman–Yao and
 *   PostgreSQL nbtree convention: a child holds keys <= its separator, and every page's high key equals the separator
 *   its parent keeps for it. Every page keeps a right link. Leaf splits put the lower two of four keys on the left page.
 * - Latches are S / X, plus SX for InnoDB (S–SX compatible, SX–SX not, X conflicts with everything, as in
 *   InnoDB's sync0rw.cc matrix). No writer priority: a waiting X request does not stop new S requests.
 *   A blocked request changes nothing; stepping that thread again retries it.
 * - One step = one latch acquisition, one page visit or one structural change.
 * - Crabbing: "safe" for an insert means the page has room for one more key. Deletes are not modelled.
 *   Optimistic descent: read latches to the leaf, write latch on the leaf, restart pessimistically if the leaf is full.
 * - InnoDB (MySQL 5.7+ btr_cur_search_to_nth_level): searches and BTR_MODIFY_LEAF take index->lock in S and S-latch the
 *   path, holding it until the leaf is latched. BTR_MODIFY_TREE takes index->lock in SX, buffer-fixes the path without
 *   latches, then latches the root (SX, or X if it will change) and the parent (X) top-down, then the leaf's left
 *   sibling, the leaf and its right sibling (X) left to right. MySQL 5.6: index->lock in X, non-leaf pages only
 *   buffer-fixed by every descent; the split X-latches the root when it allocates the new page (btr_page_alloc reads
 *   the segment header there), then the parent when the node pointer is inserted.
 * - OLC (Leis et al. reference B-tree): one version counter per page, incremented when a write lock is released;
 *   the lock bit is drawn separately here (real code packs both into one 64-bit word). Readers never write shared
 *   memory. Writers lock at most a page and its parent: a full inner page is split eagerly on the way down, and a full
 *   leaf is split on its own (the new key is not added), and in both cases the insert unlocks and restarts.
 * - B-link (PostgreSQL nbtree): read locks on one page at a time, released before the next page is locked; the leaf
 *   is write-locked; a split holds the left page, the new right page and the old right sibling; the parent is
 *   write-locked while the left page is still held, and the right page is released once the parent is locked.
 */

export type Mode = 'crab' | 'innodb' | 'olc' | 'blink';
export type Tid = 'I' | 'R';
export type LatchMode = 'S' | 'SX' | 'X';

export type Config = {
  mode: Mode;
  insertKey: number;
  parentFull: boolean;
  optimisticDescent: boolean;
  mysql56: boolean;
  highKeyCheck: boolean;
};

export const READ_KEY = 30;
export const CAP = 3;

export type PageNode = {
  id: string;
  label: string;
  leaf: boolean;
  level: number;
  keys: number[];
  children: string[];
  high: number | null;
  right: string | null;
  version: number;
  incompleteSplit: boolean;
  isNew: boolean;
  s: Tid[];
  sx: Tid | null;
  x: Tid | null;
  fix: Tid[];
};

export type ThreadState = {
  id: Tid;
  key: number;
  pc: string;
  at: string | null;
  next: string | null;
  parent: string | null;
  vAt: number;
  vParent: number;
  stack: string[];
  child: string | null;
  plan: string[];
  spec: { found: boolean; node: string } | null;
  blocked: { node: string; want: LatchMode; by: Tid } | null;
  done: boolean;
  result: { found: boolean; node: string } | null;
  steps: number;
  waits: number;
  restarts: number;
  hops: number;
  maxHeld: number;
};

export type LogKind = 'latch' | 'release' | 'block' | 'restart' | 'split' | 'hop' | 'read' | 'done' | 'wrong';
export type LogEntry = { n: number; who: Tid; kind: LogKind; text: string };
type Lockable = { s: Tid[]; sx: Tid | null; x: Tid | null };

export type World = {
  cfg: Config;
  nodes: Record<string, PageNode>;
  root: string;
  index: Lockable;
  th: Record<Tid, ThreadState>;
  log: LogEntry[];
  n: number;
};

export const DEFAULT_CONFIG: Config = {
  mode: 'blink',
  insertKey: 27,
  parentFull: false,
  optimisticDescent: false,
  mysql56: false,
  highKeyCheck: true,
};

/* ------------------------------------------------------------------ tree */

function page(id: string, leaf: boolean, level: number, keys: number[], children: string[], high: number | null, right: string | null): PageNode {
  return { id, label: id, leaf, level, keys, children, high, right, version: 1, incompleteSplit: false, isNew: false, s: [], sx: null, x: null, fix: [] };
}

function thread(id: Tid, key: number): ThreadState {
  return {
    id, key, pc: 'start', at: null, next: null, parent: null, vAt: 0, vParent: 0, stack: [], child: null, plan: [], spec: null,
    blocked: null, done: false, result: null, steps: 0, waits: 0, restarts: 0, hops: 0, maxHeld: 0,
  };
}

export function initialWorld(cfg: Config): World {
  const list: PageNode[] = [page('R', false, 2, [45], ['P', 'Q'], null, null)];
  if (cfg.parentFull) {
    list.push(
      page('P', false, 1, [15, 30, 38], ['L1', 'L2', 'L3', 'L4'], 45, 'Q'),
      page('Q', false, 1, [70], ['L5', 'L6'], null, null),
      page('L1', true, 0, [5, 10], [], 15, 'L2'),
      page('L2', true, 0, [20, 25, 30], [], 30, 'L3'),
      page('L3', true, 0, [33, 36], [], 38, 'L4'),
      page('L4', true, 0, [40, 44], [], 45, 'L5'),
      page('L5', true, 0, [50, 60], [], 70, 'L6'),
      page('L6', true, 0, [75, 90], [], null, null),
    );
  } else {
    list.push(
      page('P', false, 1, [15, 30], ['L1', 'L2', 'L3'], 45, 'Q'),
      page('Q', false, 1, [70], ['L4', 'L5'], null, null),
      page('L1', true, 0, [5, 10], [], 15, 'L2'),
      page('L2', true, 0, [20, 25, 30], [], 30, 'L3'),
      page('L3', true, 0, [35, 40], [], 45, 'L4'),
      page('L4', true, 0, [50, 60], [], 70, 'L5'),
      page('L5', true, 0, [75, 90], [], null, null),
    );
  }
  const nodes: Record<string, PageNode> = {};
  for (const p of list) nodes[p.id] = p;
  return { cfg, nodes, root: 'R', index: { s: [], sx: null, x: null }, th: { I: thread('I', cfg.insertKey), R: thread('R', READ_KEY) }, log: [], n: 0 };
}

const clone = <T,>(v: T): T => JSON.parse(JSON.stringify(v)) as T;
const other = (t: Tid): Tid => (t === 'I' ? 'R' : 'I');
export const threadName = (t: Tid) => (t === 'I' ? 'inserter' : 'reader');
const L = (w: World, id: string | null) => (id ? w.nodes[id]?.label ?? id : '—');

const routeIdx = (nd: PageNode, key: number) => {
  let i = 0;
  while (i < nd.keys.length && key > nd.keys[i]) i++;
  return i;
};
const childFor = (nd: PageNode, key: number) => nd.children[routeIdx(nd, key)];
const hasRoom = (nd: PageNode) => nd.keys.length < CAP;
export const parentOf = (w: World, id: string) => Object.values(w.nodes).find((n) => n.children.includes(id))?.id ?? null;

/** Split a leaf. With a key, the key goes in as part of the split; with null (OLC) the full page splits on its own. */
function splitLeafInsert(w: World, id: string, key: number | null) {
  const nd = w.nodes[id];
  const all = (key === null ? [...nd.keys] : [...nd.keys, key]).sort((a, b) => a - b);
  const newId = `${id}n`;
  const nn = page(newId, true, nd.level, all.slice(2), [], nd.high, nd.right);
  nn.label = `${nd.label}′`;
  nn.isNew = true;
  nd.keys = all.slice(0, 2);
  nd.high = nd.keys[nd.keys.length - 1];
  nd.right = newId;
  w.nodes[newId] = nn;
  return { sep: nd.high, newId, moved: nn.keys };
}

/**
 * Split an inner page. An overflowing page (4 keys) keeps 2 keys and 3 children and moves keys[2] up; a full page
 * split eagerly (3 keys, OLC) keeps 1 key and 2 children and moves keys[1] up.
 */
function splitInner(w: World, id: string) {
  const nd = w.nodes[id];
  const newId = `${id}n`;
  const keep = Math.floor(nd.keys.length / 2);
  const nn = page(newId, false, nd.level, nd.keys.slice(keep + 1), nd.children.slice(keep + 1), nd.high, nd.right);
  nn.label = `${nd.label}′`;
  nn.isNew = true;
  const sep = nd.keys[keep];
  nd.keys = nd.keys.slice(0, keep);
  nd.children = nd.children.slice(0, keep + 1);
  nd.high = sep;
  nd.right = newId;
  w.nodes[newId] = nn;
  return { sep, newId };
}

function insertSeparator(w: World, parentId: string, leftChild: string, sep: number, newChild: string) {
  const p = w.nodes[parentId];
  const i = p.children.indexOf(leftChild);
  p.keys.splice(i, 0, sep);
  p.children.splice(i + 1, 0, newChild);
}

/* ---------------------------------------------------------------- latches */

function conflict(nd: Lockable, t: Tid, want: LatchMode): Tid | null {
  if (nd.x && nd.x !== t) return nd.x;
  if (want === 'S') return null;
  if (nd.sx && nd.sx !== t) return nd.sx;
  if (want === 'X') return nd.s.find((h) => h !== t) ?? null;
  return null;
}

function unhold(nd: Lockable, t: Tid) {
  nd.s = nd.s.filter((h) => h !== t);
  if (nd.sx === t) nd.sx = null;
  if (nd.x === t) nd.x = null;
}

function say(w: World, t: ThreadState, kind: LogKind, text: string) {
  w.log.push({ n: w.n, who: t.id, kind, text });
}

function block(w: World, t: ThreadState, node: string, want: LatchMode, by: Tid) {
  t.blocked = { node, want, by };
  t.waits++;
  const what = node === 'index' ? 'index->lock' : L(w, node);
  const last = [...w.log].reverse().find((e) => e.who === t.id);
  const text = `wait: ${want} on ${what} is blocked by the ${threadName(by)}`;
  if (!last || last.text !== text) say(w, t, 'block', text);
}

function lockable(w: World, id: string): Lockable {
  return id === 'index' ? w.index : w.nodes[id];
}

function grab(w: World, t: ThreadState, id: string, want: LatchMode): boolean {
  const nd = lockable(w, id);
  const by = conflict(nd, t.id, want);
  if (by) {
    block(w, t, id, want, by);
    return false;
  }
  unhold(nd, t.id);
  if (want === 'S') nd.s.push(t.id);
  else if (want === 'SX') nd.sx = t.id;
  else nd.x = t.id;
  t.maxHeld = Math.max(t.maxHeld, heldPages(w, t.id).length);
  return true;
}

/** A page the thread just allocated starts out write-latched by it. */
function holdNew(w: World, t: ThreadState, id: string) {
  w.nodes[id].x = t.id;
  t.maxHeld = Math.max(t.maxHeld, heldPages(w, t.id).length);
}

function release(w: World, t: Tid, id: string | null) {
  if (!id) return;
  unhold(lockable(w, id), t);
}

export function heldPages(w: World, t: Tid) {
  return Object.values(w.nodes)
    .filter((n) => n.s.includes(t) || n.sx === t || n.x === t)
    .map((n) => n.id);
}

function releaseExcept(w: World, t: Tid, keep: string[]) {
  const gone = heldPages(w, t).filter((id) => !keep.includes(id));
  for (const id of gone) release(w, t, id);
  return gone.map((id) => L(w, id));
}

function releaseAll(w: World, t: Tid) {
  for (const n of Object.values(w.nodes)) {
    unhold(n, t);
    n.fix = n.fix.filter((h) => h !== t);
  }
  unhold(w.index, t);
}

const list = (xs: string[]) => (xs.length === 0 ? 'nothing' : xs.join(' and '));
const keysOf = (nd: PageNode) => `[${nd.keys.join(', ')}]`;

function finishRead(w: World, t: ThreadState, nd: PageNode, extra = '') {
  const found = nd.keys.includes(t.key);
  t.result = { found, node: nd.id };
  t.done = true;
  releaseAll(w, t.id);
  say(
    w,
    t,
    found ? 'done' : 'wrong',
    found
      ? `search ${nd.label} ${keysOf(nd)}: found ${t.key}; release and return`
      : `search ${nd.label} ${keysOf(nd)}: ${t.key} is not here — return "not found". Wrong: ${t.key} is in the tree${extra}`,
  );
}

function insertIntoLeaf(nd: PageNode, key: number) {
  nd.keys = [...nd.keys, key].sort((a, b) => a - b);
}

/* ------------------------------------------------------ latch crabbing */

function crab(w: World, t: ThreadState) {
  const isI = t.id === 'I';
  const writeAll = isI && (t.plan.includes('pessimistic') || !w.cfg.optimisticDescent);
  switch (t.pc) {
    case 'start': {
      const want: LatchMode = writeAll ? 'X' : 'S';
      if (!grab(w, t, w.root, want)) return;
      t.at = w.root;
      t.pc = 'descend';
      say(w, t, 'latch', `${want}-latch the root R`);
      return;
    }
    case 'descend': {
      const nd = w.nodes[t.at!];
      const c = childFor(nd, t.key);
      const cn = w.nodes[c];
      const want: LatchMode = writeAll || (isI && cn.leaf) ? 'X' : 'S';
      if (!grab(w, t, c, want)) return;
      if (writeAll) {
        if (hasRoom(cn)) {
          const gone = releaseExcept(w, t.id, [c]);
          say(w, t, 'latch', `X-latch ${cn.label} ${keysOf(cn)}: it has room, so no split can reach above it — release ${list(gone)}`);
        } else {
          say(w, t, 'latch', `X-latch ${cn.label} ${keysOf(cn)}: it is full, so a split could climb — keep ${list(heldPages(w, t.id).filter((id) => id !== c).map((id) => L(w, id)))}`);
        }
      } else {
        release(w, t.id, nd.id);
        say(w, t, 'latch', `${want}-latch ${cn.label}, then release ${nd.label}`);
      }
      t.at = c;
      if (cn.leaf) t.pc = isI ? 'modify' : 'read';
      return;
    }
    case 'read':
      finishRead(w, t, w.nodes[t.at!]);
      return;
    case 'modify': {
      const nd = w.nodes[t.at!];
      if (hasRoom(nd)) {
        insertIntoLeaf(nd, t.key);
        releaseAll(w, t.id);
        t.done = true;
        say(w, t, 'done', `insert ${t.key} into ${nd.label} → ${keysOf(nd)}; release`);
        return;
      }
      if (!writeAll) {
        releaseAll(w, t.id);
        t.restarts++;
        t.plan = ['pessimistic'];
        t.pc = 'start';
        say(w, t, 'restart', `${nd.label} is full and the read latches above it are gone — release it and restart from the root with X latches`);
        return;
      }
      const pid = parentOf(w, nd.id)!;
      const sp = splitLeafInsert(w, nd.id, t.key);
      holdNew(w, t, sp.newId);
      insertSeparator(w, pid, nd.id, sp.sep, sp.newId);
      let text = `split ${nd.label}: [${sp.moved.join(', ')}] move to new page ${L(w, sp.newId)}; post separator ${sp.sep} into ${L(w, pid)}`;
      if (w.nodes[pid].keys.length > CAP) {
        const r = splitInner(w, pid);
        holdNew(w, t, r.newId);
        const gp = parentOf(w, pid)!;
        insertSeparator(w, gp, pid, r.sep, r.newId);
        text += `, which overflows: split ${L(w, pid)} and post ${r.sep} into ${L(w, gp)}`;
      }
      t.pc = 'release';
      say(w, t, 'split', text);
      return;
    }
    case 'release':
      releaseAll(w, t.id);
      t.done = true;
      say(w, t, 'done', 'release every latch');
      return;
  }
}

/* ---------------------------------------------------------------- InnoDB */

function innodb(w: World, t: ThreadState) {
  const v56 = w.cfg.mysql56;
  const isI = t.id === 'I';
  const leafWant: LatchMode = isI ? 'X' : 'S';
  switch (t.pc) {
    case 'start': {
      if (!grab(w, t, 'index', 'S')) return;
      t.pc = v56 ? 'fix' : 'root';
      say(w, t, 'latch', `S-lock index->lock (${isI ? 'BTR_MODIFY_LEAF' : 'BTR_SEARCH_LEAF'})`);
      return;
    }
    case 'root': {
      if (!grab(w, t, w.root, 'S')) return;
      t.at = w.root;
      t.pc = 'descend';
      say(w, t, 'latch', 'S-latch the root R');
      return;
    }
    case 'descend': {
      const nd = w.nodes[t.at!];
      const c = childFor(nd, t.key);
      const cn = w.nodes[c];
      if (!cn.leaf) {
        if (!grab(w, t, c, 'S')) return;
        t.at = c;
        say(w, t, 'latch', `S-latch ${cn.label} and keep ${nd.label}: the path stays latched until the leaf is`);
        return;
      }
      if (!grab(w, t, c, leafWant)) return;
      const gone = releaseExcept(w, t.id, [c]);
      release(w, t.id, 'index');
      t.at = c;
      t.pc = isI ? 'modify' : 'read';
      say(w, t, 'latch', `${leafWant}-latch leaf ${cn.label}; release index->lock and ${list(gone)}`);
      return;
    }
    case 'fix': {
      let id = w.root;
      const path: string[] = [];
      while (!w.nodes[childFor(w.nodes[id], t.key)].leaf) {
        path.push(id);
        id = childFor(w.nodes[id], t.key);
      }
      path.push(id);
      for (const p of path) w.nodes[p].fix.push(t.id);
      t.at = id;
      t.pc = 'leaf';
      say(w, t, 'read', `descend ${path.map((p) => L(w, p)).join(' → ')} with buffer fixes only, no page latches: index->lock keeps tree changes out`);
      return;
    }
    case 'leaf': {
      const c = childFor(w.nodes[t.at!], t.key);
      if (!grab(w, t, c, leafWant)) return;
      for (const n of Object.values(w.nodes)) n.fix = n.fix.filter((h) => h !== t.id);
      release(w, t.id, 'index');
      t.at = c;
      t.pc = isI ? 'modify' : 'read';
      say(w, t, 'latch', `${leafWant}-latch leaf ${L(w, c)}; drop the buffer fixes and release index->lock`);
      return;
    }
    case 'read':
      finishRead(w, t, w.nodes[t.at!]);
      return;
    case 'modify': {
      const nd = w.nodes[t.at!];
      if (hasRoom(nd)) {
        insertIntoLeaf(nd, t.key);
        releaseAll(w, t.id);
        t.done = true;
        say(w, t, 'done', `insert ${t.key} into ${nd.label} → ${keysOf(nd)}; commit the mini-transaction`);
        return;
      }
      releaseAll(w, t.id);
      t.restarts++;
      t.pc = 'tree';
      say(w, t, 'restart', `${nd.label} is full: the optimistic insert returns DB_FAIL — release it and retry with BTR_MODIFY_TREE`);
      return;
    }
    case 'tree': {
      const want: LatchMode = v56 ? 'X' : 'SX';
      if (!grab(w, t, 'index', want)) return;
      t.pc = 'fixPath';
      say(w, t, 'latch', `${want}-lock index->lock (BTR_MODIFY_TREE)${v56 ? ': no other descent can enter the index' : ': other tree changes wait, searches still enter'}`);
      return;
    }
    case 'fixPath': {
      const rootN = w.nodes[w.root];
      const pid = childFor(rootN, t.key);
      const pn = w.nodes[pid];
      const leaf = childFor(pn, t.key);
      rootN.fix.push(t.id);
      pn.fix.push(t.id);
      t.child = leaf;
      t.parent = pid;
      const left = Object.values(w.nodes).find((n) => n.right === leaf)?.id ?? null;
      const right = w.nodes[leaf].right;
      const leaves = [left ? `${left}:X` : '', `${leaf}:X`, right ? `${right}:X` : ''].filter(Boolean);
      const parentWillSplit = !hasRoom(pn);
      t.plan = v56 ? leaves : [`${w.root}:${parentWillSplit ? 'X' : 'SX'}`, `${pid}:X`, ...leaves];
      t.pc = 'latchPlan';
      say(
        w,
        t,
        'read',
        `descend R → ${pn.label} with buffer fixes only; ${pn.label} ${parentWillSplit ? 'is full, so R will change too' : 'has room for the new node pointer'}`,
      );
      return;
    }
    case 'latchPlan': {
      const [id, mode] = t.plan[0].split(':') as [string, LatchMode];
      if (!grab(w, t, id, mode)) return;
      w.nodes[id].fix = w.nodes[id].fix.filter((h) => h !== t.id);
      t.plan = t.plan.slice(1);
      const why =
        id === w.root
          ? mode === 'SX'
            ? ' (the root page holds the segment headers the new page is allocated from)'
            : ' (it will receive a node pointer)'
          : id === t.parent
            ? ' (it will receive the node pointer)'
            : id === t.child
              ? ''
              : w.nodes[id].right === t.child
                ? ' — left sibling first: leaves are latched left to right'
                : ' — right sibling last: its prev link will change';
      say(w, t, 'latch', `${mode}-latch ${L(w, id)}${why}`);
      if (t.plan.length === 0) t.pc = 'split';
      return;
    }
    case 'split': {
      const pid = t.parent!;
      const pn = w.nodes[pid];
      // 5.6 btr_page_split_and_insert: btr_page_alloc X-latches the root (segment headers), then the node pointer
      // insert X-latches the parent.
      if (v56 && (!grab(w, t, w.root, 'X') || !grab(w, t, pid, 'X'))) return;
      const sp = splitLeafInsert(w, t.child!, t.key);
      holdNew(w, t, sp.newId);
      insertSeparator(w, pid, t.child!, sp.sep, sp.newId);
      let text = `${v56 ? `X-latch R to allocate the new page, then ${pn.label}; ` : ''}split ${L(w, t.child)}: [${sp.moved.join(', ')}] move to new page ${L(w, sp.newId)}; insert node pointer ${sp.sep} into ${pn.label}`;
      if (pn.keys.length > CAP) {
        const r = splitInner(w, pid);
        holdNew(w, t, r.newId);
        insertSeparator(w, w.root, pid, r.sep, r.newId);
        text += `, which overflows: split ${pn.label} and insert ${r.sep} into R`;
      }
      t.pc = 'commit';
      say(w, t, 'split', text);
      return;
    }
    case 'commit':
      releaseAll(w, t.id);
      t.done = true;
      say(w, t, 'done', 'commit the mini-transaction: release every page latch and index->lock');
      return;
  }
}

/* --------------------------------------------- optimistic lock coupling */

function olc(w: World, t: ThreadState) {
  const isI = t.id === 'I';
  const restart = (why: string) => {
    releaseAll(w, t.id);
    t.restarts++;
    t.pc = 'start';
    t.at = null;
    t.parent = null;
    t.spec = null;
    say(w, t, 'restart', `${why} — restart from the root`);
  };
  const lockedByOther = (nd: PageNode) => nd.x !== null && nd.x !== t.id;
  const changed = (id: string | null, v: number) => (id ? lockedByOther(w.nodes[id]) || w.nodes[id].version !== v : false);
  const unlock = (nd: PageNode) => {
    nd.x = null;
    nd.version++;
  };
  switch (t.pc) {
    case 'start': {
      const r = w.nodes[w.root];
      if (lockedByOther(r)) return restart('R is write-locked');
      t.at = w.root;
      t.vAt = r.version;
      t.parent = null;
      t.pc = 'inner';
      say(w, t, 'read', `read R's version v${r.version} — no latch, nothing written`);
      return;
    }
    case 'inner': {
      const nd = w.nodes[t.at!];
      if (isI && !hasRoom(nd)) {
        const p = t.parent ? w.nodes[t.parent] : null;
        if (p && changed(p.id, t.vParent)) return restart(`upgrade of ${p.label} failed: its version moved`);
        if (changed(nd.id, t.vAt)) return restart(`upgrade of ${nd.label} failed: its version moved`);
        if (p) p.x = t.id;
        nd.x = t.id;
        t.pc = 'eagerSplit';
        say(w, t, 'latch', `${nd.label} is full: write-lock ${p ? `${p.label} (v${t.vParent}) and ` : ''}${nd.label} (v${t.vAt}) by compare-and-swap, to split it before going lower`);
        return;
      }
      if (t.parent && changed(t.parent, t.vParent)) return restart(`${L(w, t.parent)} changed (was v${t.vParent}, now v${w.nodes[t.parent].version})`);
      const c = childFor(nd, t.key);
      if (changed(nd.id, t.vAt)) return restart(`${nd.label} changed while its downlink was read (v${t.vAt} → v${nd.version})`);
      const cn = w.nodes[c];
      if (lockedByOther(cn)) return restart(`${cn.label} is write-locked`);
      say(w, t, 'read', `read downlink → ${cn.label}; ${nd.label} is still v${t.vAt} ✓; read ${cn.label}'s version v${cn.version}`);
      t.parent = nd.id;
      t.vParent = t.vAt;
      t.at = c;
      t.vAt = cn.version;
      t.pc = cn.leaf ? (isI ? 'leaf' : 'readLeaf') : 'inner';
      return;
    }
    case 'eagerSplit': {
      const nd = w.nodes[t.at!];
      const r = splitInner(w, nd.id);
      const pid = t.parent ?? w.root;
      insertSeparator(w, pid, nd.id, r.sep, r.newId);
      unlock(nd);
      unlock(w.nodes[pid]);
      t.restarts++;
      t.pc = 'start';
      t.at = null;
      t.parent = null;
      say(w, t, 'restart', `split ${nd.label} → ${nd.label}, ${L(w, r.newId)}; post ${r.sep} into ${L(w, pid)}; unlock both (versions +1) and restart the insert`);
      return;
    }
    case 'readLeaf': {
      const nd = w.nodes[t.at!];
      const found = nd.keys.includes(t.key);
      t.spec = { found, node: nd.id };
      t.pc = 'validateLeaf';
      say(w, t, 'read', `search ${nd.label} ${keysOf(nd)} speculatively: ${found ? 'found' : 'not found'} — not trusted until validated`);
      return;
    }
    case 'validateLeaf': {
      const nd = w.nodes[t.at!];
      const moved = [t.parent && changed(t.parent, t.vParent) ? t.parent : null, changed(nd.id, t.vAt) ? nd.id : null]
        .filter((x): x is string => x !== null)
        .map((id) => {
          const m = w.nodes[id];
          const v = id === nd.id ? t.vAt : t.vParent;
          return lockedByOther(m) ? `${m.label} is write-locked` : `${m.label} is v${m.version}, not v${v}`;
        });
      if (moved.length) return restart(`validate: ${moved.join(' and ')} — discard the speculative "${t.spec?.found ? 'found' : 'not found'}"`);
      t.result = t.spec;
      t.done = true;
      const ok = t.spec?.found;
      say(w, t, ok ? 'done' : 'wrong', ok ? `${nd.label} still v${t.vAt} ✓ — the answer "found" stands` : `validated "not found" — wrong`);
      return;
    }
    case 'leaf': {
      const nd = w.nodes[t.at!];
      const p = t.parent ? w.nodes[t.parent] : null;
      if (hasRoom(nd)) {
        if (changed(nd.id, t.vAt)) return restart(`upgrade of ${nd.label} failed: its version moved`);
        if (p && changed(p.id, t.vParent)) return restart(`${p.label} changed`);
        nd.x = t.id;
        t.pc = 'insertLeaf';
        say(w, t, 'latch', `write-lock ${nd.label} by compare-and-swap on v${t.vAt}; ${p ? `${p.label} still v${t.vParent} ✓` : ''}`);
        return;
      }
      if (p && changed(p.id, t.vParent)) return restart(`upgrade of ${p.label} failed: its version moved`);
      if (changed(nd.id, t.vAt)) return restart(`upgrade of ${nd.label} failed: its version moved`);
      if (p) p.x = t.id;
      nd.x = t.id;
      t.pc = 'splitLeaf';
      say(w, t, 'latch', `${nd.label} is full: write-lock ${p ? `${p.label} (v${t.vParent}) and ` : ''}${nd.label} (v${t.vAt}) by compare-and-swap`);
      return;
    }
    case 'insertLeaf': {
      const nd = w.nodes[t.at!];
      insertIntoLeaf(nd, t.key);
      unlock(nd);
      t.done = true;
      say(w, t, 'done', `insert ${t.key} → ${keysOf(nd)}; unlock, version becomes v${nd.version}`);
      return;
    }
    case 'splitLeaf': {
      const nd = w.nodes[t.at!];
      const pid = t.parent!;
      const sp = splitLeafInsert(w, nd.id, null);
      insertSeparator(w, pid, nd.id, sp.sep, sp.newId);
      unlock(nd);
      unlock(w.nodes[pid]);
      t.restarts++;
      t.pc = 'start';
      t.at = null;
      t.parent = null;
      say(w, t, 'restart', `split ${nd.label}: [${sp.moved.join(', ')}] move to new page ${L(w, sp.newId)}; post ${sp.sep} into ${L(w, pid)}; unlock both — ${nd.label} v${nd.version}, ${L(w, pid)} v${w.nodes[pid].version} — and restart the insert`);
      return;
    }
  }
}

/* ------------------------------------------------ Lehman–Yao B-link tree */

function blink(w: World, t: ThreadState) {
  const isI = t.id === 'I';
  switch (t.pc) {
    case 'start': {
      if (!grab(w, t, w.root, 'S')) return;
      t.at = w.root;
      t.stack = [];
      t.pc = 'atNode';
      say(w, t, 'latch', 'read-lock the root R');
      return;
    }
    case 'enter': {
      const target = w.nodes[t.next!];
      const want: LatchMode = isI && target.leaf ? 'X' : 'S';
      if (!grab(w, t, target.id, want)) return;
      t.at = target.id;
      t.next = null;
      t.pc = 'atNode';
      say(w, t, 'latch', `${want === 'S' ? 'read' : 'write'}-lock ${target.label}`);
      return;
    }
    case 'atNode': {
      const nd = w.nodes[t.at!];
      const checks = isI || w.cfg.highKeyCheck;
      if (checks && nd.high !== null && t.key > nd.high && nd.right) {
        release(w, t.id, nd.id);
        t.next = nd.right;
        t.hops++;
        t.pc = 'enter';
        say(w, t, 'hop', `${t.key} > ${nd.label}'s high key ${nd.high}: it split after its downlink was read — release it and follow the right link to ${L(w, nd.right)}`);
        return;
      }
      if (!nd.leaf) {
        const c = childFor(nd, t.key);
        if (isI) t.stack.push(nd.id);
        release(w, t.id, nd.id);
        t.next = c;
        t.pc = 'enter';
        say(w, t, 'release', `read downlink → ${L(w, c)}; release ${nd.label} before locking ${L(w, c)} — no latch held`);
        return;
      }
      if (!isI) {
        finishRead(w, t, nd, checks ? '' : ` — it moved right to ${L(w, nd.right)}, and without the high-key check the reader never finds out`);
        return;
      }
      if (hasRoom(nd)) {
        insertIntoLeaf(nd, t.key);
        releaseAll(w, t.id);
        t.done = true;
        say(w, t, 'done', `insert ${t.key} into ${nd.label} → ${keysOf(nd)}; release`);
        return;
      }
      const oldRight = nd.right;
      if (oldRight && !grab(w, t, oldRight, 'X')) return;
      const sp = splitLeafInsert(w, nd.id, t.key);
      holdNew(w, t, sp.newId);
      nd.incompleteSplit = true;
      release(w, t.id, oldRight);
      t.child = nd.id;
      t.pc = 'latchParent';
      say(
        w,
        t,
        'split',
        `split ${nd.label}: [${sp.moved.join(', ')}] move to new page ${L(w, sp.newId)}; ${nd.label} gets high key ${sp.sep} and a right link to ${L(w, sp.newId)}, flagged INCOMPLETE_SPLIT${oldRight ? ` (${L(w, oldRight)} was write-locked briefly to fix its left link)` : ''}`,
      );
      return;
    }
    case 'latchParent': {
      let p = t.stack[t.stack.length - 1];
      while (!w.nodes[p].children.includes(t.child!) && w.nodes[p].right) p = w.nodes[p].right!;
      if (!grab(w, t, p, 'X')) return;
      t.stack = t.stack.slice(0, -1);
      t.parent = p;
      const newRight = w.nodes[t.child!].right;
      release(w, t.id, newRight);
      t.pc = 'post';
      say(w, t, 'latch', `write-lock parent ${L(w, p)} while still holding ${L(w, t.child)}; release the new right page ${L(w, newRight)}`);
      return;
    }
    case 'post': {
      const p = w.nodes[t.parent!];
      const left = w.nodes[t.child!];
      const sep = left.high!;
      const newId = left.right!;
      const overflow = !hasRoom(p);
      if (overflow && p.right && !grab(w, t, p.right, 'X')) return;
      insertSeparator(w, p.id, left.id, sep, newId);
      left.incompleteSplit = false;
      if (!overflow) {
        release(w, t.id, left.id);
        release(w, t.id, p.id);
        t.done = true;
        say(w, t, 'done', `insert downlink ${sep} → ${L(w, newId)} into ${p.label} and clear ${left.label}'s INCOMPLETE_SPLIT in one step; release both`);
        return;
      }
      const oldRight = p.right;
      const r = splitInner(w, p.id);
      holdNew(w, t, r.newId);
      p.incompleteSplit = true;
      release(w, t.id, oldRight);
      release(w, t.id, left.id);
      t.child = p.id;
      t.pc = 'latchParent';
      say(w, t, 'split', `insert downlink ${sep} → ${L(w, newId)} into ${p.label}, clear ${left.label}'s flag, release ${left.label} — ${p.label} overflows: split it (high key ${r.sep}, right link to ${L(w, r.newId)})`);
      return;
    }
  }
}

const PROGRAMS: Record<Mode, (w: World, t: ThreadState) => void> = { crab, innodb, olc, blink };

/* ---------------------------------------------------------------- driver */

export function step(w0: World, tid: Tid): World {
  if (w0.th[tid].done) return w0;
  const w = clone(w0);
  const t = w.th[tid];
  t.blocked = null;
  w.n++;
  PROGRAMS[w.cfg.mode](w, t);
  if (!t.blocked) t.steps++;
  t.maxHeld = Math.max(t.maxHeld, heldPages(w, tid).length);
  return w;
}

/** Would stepping this thread right now only produce a wait? */
export function wouldBlock(w: World, tid: Tid) {
  if (w.th[tid].done) return false;
  return step(w, tid).th[tid].blocked !== null;
}

/** The text the next step of a thread would log. */
export function preview(w: World, tid: Tid) {
  if (w.th[tid].done) return 'finished';
  const n = step(w, tid);
  const e = n.log.slice(w.log.length).filter((x) => x.who === tid);
  if (n.th[tid].blocked) {
    const b = n.th[tid].blocked!;
    return `waits: ${b.want} on ${b.node === 'index' ? 'index->lock' : L(n, b.node)} is held by the ${threadName(b.by)}`;
  }
  return e.length ? e[e.length - 1].text : '…';
}

/** The reader's next action would enter a leaf page. */
export function readerAtDoor(w: World) {
  const t = w.th.R;
  if (t.done) return false;
  switch (w.cfg.mode) {
    case 'crab':
      return t.pc === 'descend' && w.nodes[childFor(w.nodes[t.at!], t.key)].leaf;
    case 'innodb':
      return w.cfg.mysql56 ? t.pc === 'leaf' : t.pc === 'descend' && w.nodes[childFor(w.nodes[t.at!], t.key)].leaf;
    case 'olc':
      return t.pc === 'readLeaf';
    case 'blink':
      return t.pc === 'enter' && w.nodes[t.next!].leaf;
  }
}

/**
 * The race schedule: the reader descends until it is about to enter the leaf, then the inserter runs while it can,
 * then the reader, and so on. Returns null when both are done (or, if it ever happened, both blocked).
 */
export function racePick(w: World): Tid | null {
  const { I, R } = w.th;
  if (!R.done && I.steps === 0 && I.waits === 0 && !readerAtDoor(w) && !wouldBlock(w, 'R')) return 'R';
  // A thread that would block still gets one attempt, so the wait is recorded and drawn; then the other runs.
  const turn = (t: ThreadState) => !t.done && (!wouldBlock(w, t.id) || t.blocked === null);
  if (turn(I)) return 'I';
  if (turn(R)) return 'R';
  return null;
}

export const MAX_ACTIONS = 160;

export function replay(cfg: Config, actions: Tid[]) {
  let w = initialWorld(cfg);
  for (const a of actions.slice(0, MAX_ACTIONS)) w = step(w, a);
  return w;
}

export function raceActions(cfg: Config, from: Tid[] = []) {
  const actions = [...from];
  let w = replay(cfg, actions);
  while (actions.length < MAX_ACTIONS) {
    const p = racePick(w);
    if (!p) break;
    actions.push(p);
    w = step(w, p);
  }
  return actions;
}

export const deadlocked = (w: World) => !w.th.I.done && !w.th.R.done && wouldBlock(w, 'I') && wouldBlock(w, 'R');

/** Structural checks used by the tests: key order along every level, high keys, and B-link reachability. */
export function checkTree(w: World) {
  const problems: string[] = [];
  let lm = w.root;
  const levels: string[][] = [];
  for (;;) {
    const chain: string[] = [];
    let n: string | null = lm;
    while (n) {
      chain.push(n);
      n = w.nodes[n].right;
    }
    levels.push(chain);
    if (w.nodes[lm].leaf) break;
    lm = w.nodes[lm].children[0];
  }
  const leaves = levels[levels.length - 1];
  let prevHigh = -Infinity;
  const allKeys: number[] = [];
  for (const id of leaves) {
    const nd = w.nodes[id];
    for (const k of nd.keys) {
      if (k <= prevHigh) problems.push(`${id}: key ${k} <= left high ${prevHigh}`);
      if (nd.high !== null && k > nd.high) problems.push(`${id}: key ${k} > high ${nd.high}`);
      allKeys.push(k);
    }
    if (nd.high !== null) prevHigh = nd.high;
  }
  for (let i = 1; i < allKeys.length; i++) if (allKeys[i] <= allKeys[i - 1]) problems.push('leaf chain not sorted');
  for (const nd of Object.values(w.nodes)) {
    if (nd.leaf) continue;
    if (nd.children.length !== nd.keys.length + 1) problems.push(`${nd.id}: ${nd.keys.length} keys but ${nd.children.length} children`);
    if (nd.keys.length > CAP) problems.push(`${nd.id}: over capacity`);
    nd.children.forEach((c, i) => {
      const bound = i < nd.keys.length ? nd.keys[i] : nd.high;
      // A page flagged INCOMPLETE_SPLIT has handed the top of its range to right siblings the parent does not know yet.
      let cur: string | null = c;
      for (let hop = 0; cur && w.nodes[cur].high !== bound && w.nodes[cur].incompleteSplit && hop < 3; hop++) cur = w.nodes[cur].right;
      if (!cur || w.nodes[cur].high !== bound) problems.push(`${c}: high ${w.nodes[c].high} but parent ${nd.id} bound ${bound}`);
    });
  }
  const findBlink = (key: number, moveRight: boolean) => {
    let id = w.root;
    for (let guard = 0; guard < 20; guard++) {
      const nd = w.nodes[id];
      if (moveRight && nd.high !== null && key > nd.high && nd.right) {
        id = nd.right;
        continue;
      }
      if (nd.leaf) return nd.keys.includes(key);
      id = childFor(nd, key);
    }
    return false;
  };
  for (const k of allKeys) if (!findBlink(k, true)) problems.push(`key ${k} unreachable with move-right`);
  const quiescent = w.th.I.done;
  if (quiescent) {
    for (const k of allKeys) if (!findBlink(k, false)) problems.push(`key ${k} unreachable by plain descent`);
    if (Object.values(w.nodes).some((n) => n.incompleteSplit)) problems.push('incomplete split left behind');
  }
  return { problems, keys: allKeys, levels };
}

/* -------------------------------------------------------------------- UI */

const MODE_OPTIONS = [
  { value: 'crab', label: 'Latch crabbing', title: 'Bayer–Schkolnick latch coupling' },
  { value: 'innodb', label: 'InnoDB index lock', title: 'index->lock S/SX/X plus page latches' },
  { value: 'olc', label: 'Optimistic lock coupling', title: 'Per-page version counters, validate and restart' },
  { value: 'blink', label: 'B-link (PostgreSQL)', title: 'Lehman–Yao right links and high keys' },
] as const;

const TC: Record<Tid, string> = { I: 'var(--viz-1)', R: 'var(--viz-2)' };
const KW = 22;
const LEVEL_Y = [250, 150, 50];
const BASE_W = 680;
/** Page box height: B-link pages carry a third row for the INCOMPLETE_SPLIT flag. */
const nodeH = (mode: Mode) => (mode === 'blink' ? 52 : 40);

type Box = { id: string; x: number; y: number; w: number };

export function layoutTree(w: World) {
  const { levels } = checkTree(w);
  const boxes: Record<string, Box> = {};
  const width = (id: string) => Math.max(12 + KW * Math.max(2, w.nodes[id].keys.length), w.cfg.mode === 'blink' ? 86 : 0);
  const LEFT = 14;
  const leaves = levels[levels.length - 1];
  const sum = leaves.reduce((acc, id) => acc + width(id), 0);
  const gap = Math.min(64, Math.max(20, (BASE_W - 2 * LEFT - sum) / Math.max(1, leaves.length - 1)));
  let x = Math.max(LEFT, (BASE_W - (sum + gap * (leaves.length - 1))) / 2);
  for (const id of leaves) {
    const bw = width(id);
    boxes[id] = { id, x, y: LEVEL_Y[0], w: bw };
    x += bw + gap;
  }
  let right = x - gap;
  for (let li = levels.length - 2; li >= 0; li--) {
    let minX = LEFT;
    for (const id of levels[li]) {
      const nd = w.nodes[id];
      const bw = width(id);
      const kids = nd.children.map((c) => boxes[c]).filter(Boolean);
      const centre = kids.length ? kids.reduce((acc, b) => acc + b.x + b.w / 2, 0) / kids.length : minX + bw / 2;
      const bx = Math.max(minX, centre - bw / 2);
      boxes[id] = { id, x: bx, y: LEVEL_Y[nd.level], w: bw };
      minX = bx + bw + 20;
      right = Math.max(right, bx + bw);
    }
  }
  return { boxes, width: Math.max(BASE_W, right + LEFT) };
}

function latchBadges(w: World, id: string): { who: Tid; text: string; dashed?: boolean }[] {
  const nd = w.nodes[id];
  const out: { who: Tid; text: string; dashed?: boolean }[] = [];
  for (const who of ['I', 'R'] as Tid[]) {
    if (nd.x === who) out.push({ who, text: `${who}·X` });
    else if (nd.sx === who) out.push({ who, text: `${who}·SX` });
    else if (nd.s.includes(who)) out.push({ who, text: `${who}·S` });
    else if (nd.fix.includes(who)) out.push({ who, text: `${who}·fix`, dashed: true });
  }
  return out;
}

export default function BLinkLatchProtocolLab() {
  const [cfg, setCfg] = useState<Config>(DEFAULT_CONFIG);
  const [actions, setActions] = useState<Tid[]>([]);
  const w = useMemo(() => replay(cfg, actions), [cfg, actions]);
  const setC = (patch: Partial<Config>) => {
    setCfg((c) => ({ ...c, ...patch }));
    setActions([]);
  };
  const push = (t: Tid) => setActions((a) => (a.length >= MAX_ACTIONS ? a : [...a, t]));
  const nextRace = racePick(w);
  const { I, R } = w.th;
  const waitI = wouldBlock(w, 'I');
  const waitR = wouldBlock(w, 'R');
  const stuck = deadlocked(w);
  const { boxes, width: W } = useMemo(() => layoutTree(w), [w]);
  const NH = nodeH(cfg.mode);
  const H = LEVEL_Y[0] + NH + 30;
  const recent = w.log.slice(-6);
  const last = w.log[w.log.length - 1];
  const mode = cfg.mode;
  const readerStatus = R.done
    ? R.result?.found
      ? `found in ${L(w, R.result.node)}`
      : 'not found — wrong'
    : waitR
      ? `waiting on ${R.pc === 'start' && mode === 'innodb' ? 'index->lock' : L(w, step(w, 'R').th.R.blocked?.node ?? null)}`
      : R.steps === 0
        ? 'not started'
        : 'descending';
  const inserterBlockedOn = waitI ? step(w, 'I').th.I.blocked?.node ?? null : null;
  const inserterStatus = I.done ? `${cfg.insertKey} inserted` : waitI ? `waiting on ${inserterBlockedOn === 'index' ? 'index->lock' : L(w, inserterBlockedOn)}` : I.steps === 0 ? 'not started' : 'working';

  const blockedNode = (tid: Tid) => (tid === 'I' ? inserterBlockedOn : waitR ? step(w, 'R').th.R.blocked?.node ?? null : null);
  /** Where a thread's tag is drawn: the page it waits for, else the page its next step works on. */
  const cursorAt = (t: ThreadState): string | null => {
    if (t.done) return null;
    const b = blockedNode(t.id);
    if (b) return b === 'index' ? null : b;
    if (t.pc === 'start' || t.pc === 'tree' || t.pc === 'fixPath') return null;
    if (t.pc === 'enter' && t.next) return t.next;
    if (t.pc === 'latchPlan' && t.plan.length) return t.plan[0].split(':')[0];
    if (t.pc === 'latchParent') return t.child;
    if (t.pc === 'post') return t.parent;
    return t.at;
  };

  const modeCheck =
    mode === 'crab' ? (
      <Check label="Optimistic descent first" checked={cfg.optimisticDescent} onChange={(v) => setC({ optimisticDescent: v })} />
    ) : mode === 'innodb' ? (
      <Check label="MySQL 5.6 (X index lock)" checked={cfg.mysql56} onChange={(v) => setC({ mysql56: v })} />
    ) : mode === 'blink' ? (
      <Check label="Reader checks the high key" checked={cfg.highKeyCheck} onChange={(v) => setC({ highKeyCheck: v })} />
    ) : null;

  const noteText = (() => {
    if (stuck) return <><strong>Deadlock:</strong> both threads wait on each other.</>;
    if (w.log.length === 0)
      return (
        <>
          The inserter will add <strong>{cfg.insertKey}</strong>
          {cfg.insertKey === 27 ? ' to the full leaf L2, which must split' : ' to L1, which has room'}; the reader looks up <strong>{READ_KEY}</strong>, which lives in L2. Step either thread, or run the race: the reader goes first and stops just before it enters the leaf, then the inserter runs.
        </>
      );
    const summary = R.done && I.done
      ? R.result?.found
        ? ` Both finished: the reader found ${READ_KEY} in ${L(w, R.result.node)} after ${R.restarts} restart${R.restarts === 1 ? '' : 's'}${R.hops ? ` and ${R.hops} right-link hop${R.hops === 1 ? '' : 's'}` : ''}; it waited ${R.waits} time${R.waits === 1 ? '' : 's'}, the inserter ${I.waits}.`
        : ` Both finished, and the reader returned a wrong answer.`
      : '';
    return (
      <>
        <strong>{last.who === 'I' ? 'Inserter' : 'Reader'}:</strong> {last.text}.{summary}
      </>
    );
  })();

  return (
    <VizPanel
      title="Two threads, one B+tree: which latches does each protocol take?"
      subtitle="An inserter and a reader descend the same three-level tree (at most 3 keys per page). Pick a protocol, then step each thread — latch badges sit on the pages, waits are outlined, and version checks, restarts and right-link hops appear in the log."
      controls={
        <>
          <Segmented label="Protocol" value={mode} onChange={(m) => setC({ mode: m })} options={MODE_OPTIONS} />
          <Choice
            label="Inserter adds"
            value={String(cfg.insertKey)}
            onChange={(v) => setC({ insertKey: Number(v) })}
            options={[
              { value: '27', label: '27 — into full leaf L2 (splits)' },
              { value: '12', label: '12 — into L1 (has room)' },
            ]}
          />
          <Check label="Parent P is full too" checked={cfg.parentFull} onChange={(v) => setC({ parentFull: v })} />
          {modeCheck}
        </>
      }
      legend={
        <Legend
          items={[
            { label: 'Inserter (I) latch or position', color: TC.I },
            { label: 'Reader (R) latch or position', color: TC.R },
            { label: 'Waiting for a latch', color: 'var(--viz-warning)', shape: 'line' },
            { label: 'Page created by the split', color: 'var(--viz-3)', shape: 'line' },
          ]}
        />
      }
      stats={
        <Stats
          items={[
            { label: 'Reader looks up 30', value: readerStatus, hint: 'Key 30 is always in the tree, so "not found" is a wrong answer.' },
            { label: 'Inserter', value: inserterStatus },
            { label: 'Restarts I / R', value: `${I.restarts} / ${R.restarts}`, hint: 'Optimistic descents and failed version checks restart from the root.' },
            { label: 'Waits I / R', value: `${I.waits} / ${R.waits}`, hint: 'Steps that found the wanted latch held by the other thread.' },
            { label: 'Most pages the inserter held at once', value: I.maxHeld, hint: 'Page latches only (S, SX or X); index->lock and buffer fixes not counted.' },
            { label: 'Right-link hops (reader)', value: R.hops },
          ]}
        />
      }
      note={<Note>{noteText}</Note>}
      table={
        <table className="viz-table">
          <thead>
            <tr>
              <th>#</th>
              <th>Thread</th>
              <th>Step</th>
            </tr>
          </thead>
          <tbody>
            {w.log.map((e, i) => (
              <tr key={i}>
                <td>{e.n}</td>
                <td>{threadName(e.who)}</td>
                <td>{e.text}</td>
              </tr>
            ))}
          </tbody>
        </table>
      }
    >
      <div className="viz-controls">
        <Button onClick={() => push('I')} disabled={I.done || actions.length >= MAX_ACTIONS} title={preview(w, 'I')}>
          Step inserter
        </Button>
        <Button onClick={() => push('R')} disabled={R.done || actions.length >= MAX_ACTIONS} title={preview(w, 'R')}>
          Step reader
        </Button>
        <Button primary onClick={() => nextRace && push(nextRace)} disabled={!nextRace}>
          Race: next step
        </Button>
        <Button onClick={() => setActions(raceActions(cfg, actions))} disabled={!nextRace}>
          Race to the end
        </Button>
        <Button onClick={() => setActions([])} disabled={actions.length === 0}>
          Reset
        </Button>
      </div>
      <div style={{ fontSize: '0.78rem', color: 'var(--viz-ink-2)', display: 'grid', gap: 2, margin: '0 0 0.4rem' }}>
        <span>
          <strong style={{ color: 'var(--viz-ink)' }}>Inserter next:</strong> {preview(w, 'I')}
        </span>
        <span>
          <strong style={{ color: 'var(--viz-ink)' }}>Reader next:</strong> {preview(w, 'R')}
        </span>
      </div>
      <svg
        viewBox={`0 0 ${W} ${H}`}
        width={W}
        height={H}
        style={{ minWidth: 560 }}
        role="img"
        aria-label={`${MODE_OPTIONS.find((o) => o.value === mode)?.label}: reader ${readerStatus}; inserter ${inserterStatus}; inserter holds ${heldPages(w, 'I').map((id) => L(w, id)).join(', ') || 'no pages'}.`}
      >
        <defs>
          <marker id="blink-arrow" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
            <path d="M0,0 L8,4 L0,8 z" fill="var(--viz-ink-muted)" />
          </marker>
        </defs>

        {mode === 'innodb' ? (
          <g>
            <rect x={8} y={6} width={150} height={26} rx={5} fill="var(--viz-surface)" stroke={waitI && inserterBlockedOn === 'index' ? 'var(--viz-warning)' : 'var(--viz-ink-muted)'} strokeWidth={waitI && inserterBlockedOn === 'index' ? 2.5 : 1} strokeDasharray={waitI && inserterBlockedOn === 'index' ? '5 3' : undefined} />
            <text x={16} y={23} fontSize={11} fill="var(--viz-ink)">
              index-&gt;lock
            </text>
            {(['I', 'R'] as Tid[]).map((who, i) => {
              const m = w.index.x === who ? 'X' : w.index.sx === who ? 'SX' : w.index.s.includes(who) ? 'S' : null;
              if (!m) return null;
              return (
                <g key={who}>
                  <rect x={84 + i * 36} y={11} width={32} height={16} rx={8} fill="var(--viz-surface)" stroke={TC[who]} strokeWidth={2} />
                  <text x={100 + i * 36} y={23} fontSize={10} textAnchor="middle" fill="var(--viz-ink)">
                    {who}·{m}
                  </text>
                </g>
              );
            })}
          </g>
        ) : null}

        {/* downlinks */}
        {Object.values(w.nodes)
          .filter((n) => !n.leaf && boxes[n.id])
          .flatMap((n) =>
            n.children.map((c, i) => {
              const a = boxes[n.id];
              const b = boxes[c];
              if (!b) return null;
              const sx = a.x + 6 + (n.children.length === 1 ? (a.w - 12) / 2 : (i * (a.w - 12)) / (n.children.length - 1));
              return <line key={`${n.id}-${c}`} x1={sx} y1={a.y + NH} x2={b.x + b.w / 2} y2={b.y - 18} stroke="var(--viz-ink-muted)" strokeOpacity={0.55} />;
            }),
          )}

        {/* right links */}
        {Object.values(w.nodes)
          .filter((n) => n.right && boxes[n.id] && boxes[n.right] && (n.leaf || mode === 'blink'))
          .map((n) => {
            const a = boxes[n.id];
            const b = boxes[n.right!];
            return <line key={`r-${n.id}`} x1={a.x + a.w} y1={a.y + 12} x2={b.x - 1} y2={b.y + 12} stroke="var(--viz-ink-muted)" markerEnd="url(#blink-arrow)" />;
          })}

        {Object.values(w.nodes).map((nd) => {
          const b = boxes[nd.id];
          if (!b) return null;
          const badges = latchBadges(w, nd.id);
          const waiting = (['I', 'R'] as Tid[]).filter((tid) => blockedNode(tid) === nd.id);
          const meta =
            mode === 'blink' ? `${nd.label} · high ${nd.high === null ? '+∞' : nd.high}` : mode === 'olc' ? `${nd.label} · v${nd.version}` : nd.label;
          return (
            <g key={nd.id}>
              {waiting.length ? (
                <rect x={b.x - 4} y={b.y - 4} width={b.w + 8} height={NH + 8} rx={7} fill="none" stroke="var(--viz-warning)" strokeWidth={2.5} strokeDasharray="5 3" />
              ) : null}
              <rect x={b.x} y={b.y} width={b.w} height={NH} rx={5} fill="var(--viz-surface)" stroke={nd.isNew ? 'var(--viz-3)' : 'var(--viz-ink-muted)'} strokeWidth={nd.isNew ? 2.2 : 1} />
              {nd.keys.map((k, i) => {
                const kx = b.x + 6 + i * KW;
                return (
                  <g key={i}>
                    {i > 0 ? <line x1={kx} x2={kx} y1={b.y + 4} y2={b.y + 21} stroke="var(--viz-grid)" /> : null}
                    <text x={kx + KW / 2} y={b.y + 17} fontSize={11.5} textAnchor="middle" fill="var(--viz-ink)" fontWeight={nd.leaf && (k === READ_KEY || k === cfg.insertKey) ? 700 : 400}>
                      {k}
                    </text>
                    {nd.leaf && k === READ_KEY ? <rect x={kx + 4} y={b.y + 20} width={KW - 8} height={2.5} fill={TC.R} /> : null}
                    {nd.leaf && k === cfg.insertKey ? <rect x={kx + 4} y={b.y + 20} width={KW - 8} height={2.5} fill={TC.I} /> : null}
                  </g>
                );
              })}
              <text x={b.x + 6} y={b.y + 34} fontSize={9.5} fill="var(--viz-ink-2)">
                {meta}
              </text>
              {mode === 'blink' && nd.incompleteSplit ? (
                <text x={b.x + 6} y={b.y + 47} fontSize={8.5} fontWeight={700} fill="var(--viz-ink)">
                  INCOMPLETE
                </text>
              ) : null}
              {badges.map((bd, i) => (
                <g key={bd.who}>
                  <rect x={b.x + i * 36} y={b.y - 19} width={33} height={15} rx={7.5} fill="var(--viz-surface)" stroke={TC[bd.who]} strokeWidth={2} strokeDasharray={bd.dashed ? '3 2' : undefined} />
                  <text x={b.x + i * 36 + 16.5} y={b.y - 8} fontSize={9.5} textAnchor="middle" fill="var(--viz-ink)">
                    {bd.text}
                  </text>
                </g>
              ))}
              {(['I', 'R'] as Tid[])
                .filter((tid) => cursorAt(w.th[tid]) === nd.id)
                .map((tid, i) => {
                  const t = w.th[tid];
                  const pending = t.pc === 'enter';
                  const label = `${tid}${waiting.includes(tid) ? ' waits' : pending ? ' next' : mode === 'olc' ? ' reads' : ' here'}`;
                  return (
                    <g key={tid}>
                      <rect x={b.x + i * 54} y={b.y + NH + 5} width={50} height={15} rx={3} fill="var(--viz-surface)" stroke={TC[tid]} strokeWidth={1.8} strokeDasharray={pending ? '3 2' : undefined} />
                      <text x={b.x + i * 54 + 25} y={b.y + NH + 16} fontSize={9.5} textAnchor="middle" fill="var(--viz-ink)">
                        ▲ {label}
                      </text>
                    </g>
                  );
                })}
            </g>
          );
        })}
      </svg>

      <ol style={{ margin: '0.4rem 0 0', paddingLeft: '2.2rem', fontSize: '0.8rem', display: 'grid', gap: 3 }} start={Math.max(1, w.log.length - recent.length + 1)}>
        {recent.map((e, i) => (
          <li
            key={w.log.length - recent.length + i}
            style={{
              color: 'var(--viz-ink)',
              paddingLeft: 6,
              borderLeft: `3px solid ${e.kind === 'wrong' ? 'var(--viz-critical)' : e.kind === 'block' ? 'var(--viz-warning)' : e.kind === 'done' ? 'var(--viz-good)' : TC[e.who]}`,
            }}
          >
            <strong>{e.who === 'I' ? 'Inserter' : 'Reader'}</strong>
            {e.kind === 'restart' ? ' ↻' : e.kind === 'hop' ? ' →' : ''}: {e.text}
          </li>
        ))}
      </ol>
    </VizPanel>
  );
}
