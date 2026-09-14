import { useState } from 'react';
import {
  VizPanel,
  Slider,
  Segmented,
  Button,
  Legend,
  Stats,
  Note,
  TooltipHost,
  useTip,
  makeRng,
  useSize,
  fmtNum,
} from './Viz';

/**
 * Insert / split / delete / merge, run on a real B+tree.
 *
 * The model is a genuine B+tree with a tunable page capacity: records live only in
 * leaves, leaves are doubly linked (btpo_prev / btpo_next), a leaf split copies the
 * first key of the new right page up as a separator, an internal split pushes its
 * middle key up and out, and a split that reaches the root grows the tree by one level.
 *
 * Deletion is where engines diverge, so it is a mode:
 *   textbook — underflow below ceil(cap/2) borrows from a sibling, else merges and
 *              pulls the separator down; an emptied root collapses.
 *   postgres — nbtree never merges or redistributes. A page that becomes completely
 *              empty waits for VACUUM, which removes its downlink (half-dead), then
 *              unlinks it (deleted), then recycles it into the FSM on a later pass.
 *   innodb   — merge is attempted only when fill drops under MERGE_THRESHOLD, and the
 *              attempt fails if the pages do not fit together.
 *
 * Every operation reports the set of pages whose bytes changed, because that count —
 * not the key comparison count — is what a split or a merge actually costs.
 */

/* ------------------------------------------------------------------- model */

type Mode = 'textbook' | 'postgres' | 'innodb';
type PState = 'live' | 'empty' | 'halfdead' | 'deleted' | 'free';

type Page = {
  id: number;
  leaf: boolean;
  keys: number[];
  kids: number[];
  next: number | null;
  prev: number | null;
  state: PState;
};

type Tree = { pages: Record<number, Page>; root: number; nextId: number };

type LogRow = {
  n: number;
  op: string;
  key: string;
  outcome: string;
  pages: number;
  height: number;
};

type Res = {
  tree: Tree;
  written: number[];
  created: number[];
  head: string;
  body: string;
  splits: number;
  redist: number;
  mergeTries: number;
  mergeOk: number;
  op: string;
  keyLabel: string;
  outcome: string;
};

const minLeafOf = (cap: number) => Math.ceil(cap / 2);
const minInnerOf = (cap: number) => Math.ceil((cap + 1) / 2) - 1;

/** A copy-on-write editor over the page map, so "which pages were written" is exact. */
class Draft {
  pages: Record<number, Page>;
  written = new Set<number>();
  created = new Set<number>();
  cloned = new Set<number>();
  nextId: number;
  root: number;

  constructor(t: Tree) {
    this.pages = { ...t.pages };
    this.nextId = t.nextId;
    this.root = t.root;
  }

  edit(id: number): Page {
    if (!this.cloned.has(id)) {
      const p = this.pages[id];
      this.pages[id] = { ...p, keys: [...p.keys], kids: [...p.kids] };
      this.cloned.add(id);
    }
    this.written.add(id);
    return this.pages[id];
  }

  /** Prefer a recycled page over extending the file — this is what the FSM buys. */
  alloc(leaf: boolean): { page: Page; recycled: boolean } {
    const freeId = Object.values(this.pages)
      .filter((p) => p.state === 'free')
      .map((p) => p.id)
      .sort((a, b) => a - b)[0];
    if (freeId !== undefined) {
      const p = this.edit(freeId);
      p.leaf = leaf;
      p.keys = [];
      p.kids = [];
      p.next = null;
      p.prev = null;
      p.state = 'live';
      return { page: p, recycled: true };
    }
    const p: Page = { id: this.nextId++, leaf, keys: [], kids: [], next: null, prev: null, state: 'live' };
    this.pages[p.id] = p;
    this.cloned.add(p.id);
    this.created.add(p.id);
    this.written.add(p.id);
    return { page: p, recycled: false };
  }

  tree(): Tree {
    return { pages: this.pages, root: this.root, nextId: this.nextId };
  }
}

function pathTo(pages: Record<number, Page>, root: number, key: number): number[] {
  const path = [root];
  let cur = pages[root];
  while (!cur.leaf && cur.kids.length > 0) {
    let i = 0;
    while (i < cur.keys.length && key >= cur.keys[i]) i++;
    const c = cur.kids[Math.min(i, cur.kids.length - 1)];
    path.push(c);
    cur = pages[c];
  }
  return path;
}

export function levelsOf(t: Tree): number[][] {
  const out: number[][] = [];
  let cur = [t.root];
  while (cur.length > 0) {
    out.push(cur);
    const nxt: number[] = [];
    for (const id of cur) {
      const p = t.pages[id];
      if (!p.leaf) nxt.push(...p.kids);
    }
    cur = nxt;
  }
  return out;
}

function finish(
  d: Draft,
  r: {
    head: string;
    body: string;
    op: string;
    keyLabel: string;
    outcome: string;
    splits?: number;
    redist?: number;
    mergeTries?: number;
    mergeOk?: number;
  },
): Res {
  return {
    tree: d.tree(),
    written: [...d.written],
    created: [...d.created],
    head: r.head,
    body: r.body,
    op: r.op,
    keyLabel: r.keyLabel,
    outcome: r.outcome,
    splits: r.splits ?? 0,
    redist: r.redist ?? 0,
    mergeTries: r.mergeTries ?? 0,
    mergeOk: r.mergeOk ?? 0,
  };
}

/* ------------------------------------------------------------------ insert */

function insertKey(t: Tree, key: number, cap: number): Res {
  const d = new Draft(t);
  const path = pathTo(d.pages, d.root, key);
  const leafId = path[path.length - 1];

  if (d.pages[leafId].keys.includes(key)) {
    return finish(d, {
      head: `Key ${key} already lives in page ${leafId}.`,
      body:
        'Nothing is written. A unique index would raise a duplicate-key error here; a non-unique index ' +
        'would append a second entry with the same key and a different row pointer.',
      op: 'insert',
      keyLabel: String(key),
      outcome: 'duplicate — no write',
    });
  }

  const leaf = d.edit(leafId);
  let at = 0;
  while (at < leaf.keys.length && leaf.keys[at] < key) at++;
  leaf.keys.splice(at, 0, key);
  if (leaf.state === 'empty') leaf.state = 'live';

  const steps: string[] = [];
  let splits = 0;
  let grew = false;
  let recycledNote = '';
  let idx = path.length - 1;

  while (d.pages[path[idx]].keys.length > cap) {
    const node = d.edit(path[idx]);
    let sep: number;
    const made = d.alloc(node.leaf);
    const right = made.page;
    if (made.recycled) recycledNote = ` Page ${right.id} was reused from the free space map rather than extending the file.`;

    if (node.leaf) {
      const mid = Math.ceil(node.keys.length / 2);
      right.keys = node.keys.splice(mid);
      right.next = node.next;
      right.prev = node.id;
      node.next = right.id;
      if (right.next !== null) d.edit(right.next).prev = right.id;
      sep = right.keys[0];
      steps.push(
        `leaf ${node.id} overflowed at ${cap + 1} keys, split into ${node.id} (${node.keys.length}) + ${right.id} (${right.keys.length}), ` +
          `separator ${sep} copied up`,
      );
    } else {
      const mid = Math.floor(node.keys.length / 2);
      sep = node.keys[mid];
      right.keys = node.keys.slice(mid + 1);
      right.kids = node.kids.slice(mid + 1);
      node.keys = node.keys.slice(0, mid);
      node.kids = node.kids.slice(0, mid + 1);
      steps.push(
        `internal ${node.id} overflowed, split into ${node.id} + ${right.id}, separator ${sep} pushed up and out of both`,
      );
    }
    splits++;

    if (idx === 0) {
      const nr = d.alloc(false).page;
      nr.keys = [sep];
      nr.kids = [node.id, right.id];
      d.root = nr.id;
      grew = true;
      steps.push(`the split reached the root, so page ${nr.id} became the new root and the tree got one level taller`);
      break;
    }

    const parent = d.edit(path[idx - 1]);
    let p = 0;
    while (p < parent.keys.length && parent.keys[p] < sep) p++;
    parent.keys.splice(p, 0, sep);
    parent.kids.splice(p + 1, 0, right.id);
    idx--;
  }

  const pagesWritten = d.written.size;
  if (splits === 0) {
    return finish(d, {
      head: `Inserted ${key} into leaf ${leafId}: one page written.`,
      body:
        `The descent found the leaf by separator comparison, the entry went into the slot array in key order, ` +
        `and the page still had room (${d.pages[leafId].keys.length}/${cap}). This is the cheap case, and it is ` +
        `the overwhelming majority of inserts.`,
      op: 'insert',
      keyLabel: String(key),
      outcome: 'in place',
      splits,
    });
  }

  return finish(d, {
    head: `Inserted ${key}: ${splits} split${splits === 1 ? '' : 's'}, ${pagesWritten} pages written.`,
    body:
      steps.join('; ') +
      '. ' +
      (grew
        ? 'A B+tree only ever grows at the root, which is exactly why every leaf stays at the same depth. '
        : '') +
      `Every page in that list has to be written, and the split itself has to reach the log before the new downlink means anything.` +
      recycledNote,
    op: 'insert',
    keyLabel: String(key),
    outcome: `${splits} split${splits === 1 ? '' : 's'}${grew ? ' + new root' : ''}`,
    splits,
  });
}

/* ------------------------------------------------------------------ delete */

function deleteKey(t: Tree, key: number, cap: number, mode: Mode, threshold: number): Res {
  const d = new Draft(t);
  const path = pathTo(d.pages, d.root, key);
  const leafId = path[path.length - 1];

  if (!d.pages[leafId].keys.includes(key)) {
    return finish(d, {
      head: `Key ${key} is not in the tree.`,
      body: `The descent ended at leaf ${leafId} and the key is not there, so nothing is written.`,
      op: 'delete',
      keyLabel: String(key),
      outcome: 'not found',
    });
  }

  const leaf = d.edit(leafId);
  leaf.keys.splice(leaf.keys.indexOf(key), 1);

  const minLeaf = minLeafOf(cap);
  const minInner = minInnerOf(cap);
  const steps: string[] = [];
  let redist = 0;
  let mergeTries = 0;
  let mergeOk = 0;
  let collapsed = false;

  const under = (p: Page): boolean => {
    if (mode === 'postgres') return false;
    if (mode === 'innodb') return (p.keys.length / cap) * 100 < threshold;
    return p.keys.length < (p.leaf ? minLeaf : minInner);
  };

  if (mode === 'postgres') {
    if (leaf.keys.length === 0 && leaf.id !== d.root) {
      leaf.state = 'empty';
      steps.push(`leaf ${leaf.id} is now completely empty, so it becomes a candidate for deletion by the next VACUUM`);
    }
  } else {
    let idx = path.length - 1;
    while (idx > 0) {
      const node = d.pages[path[idx]];
      if (!under(node)) break;
      const parent = d.edit(path[idx - 1]);
      const ci = parent.kids.indexOf(node.id);
      if (ci < 0) break;
      const leftId = ci > 0 ? parent.kids[ci - 1] : null;
      const rightId = ci < parent.kids.length - 1 ? parent.kids[ci + 1] : null;
      const min = node.leaf ? minLeaf : minInner;
      let handled = false;

      if (mode === 'textbook') {
        if (leftId !== null && d.pages[leftId].keys.length > min) {
          const L = d.edit(leftId);
          const N = d.edit(node.id);
          if (N.leaf) {
            N.keys.unshift(L.keys.pop() as number);
            parent.keys[ci - 1] = N.keys[0];
          } else {
            N.keys.unshift(parent.keys[ci - 1]);
            parent.keys[ci - 1] = L.keys.pop() as number;
            N.kids.unshift(L.kids.pop() as number);
          }
          redist++;
          handled = true;
          steps.push(`page ${N.id} underflowed and borrowed one entry from its left sibling ${L.id}; the separator in ${parent.id} moved with it`);
        } else if (rightId !== null && d.pages[rightId].keys.length > min) {
          const R = d.edit(rightId);
          const N = d.edit(node.id);
          if (N.leaf) {
            N.keys.push(R.keys.shift() as number);
            parent.keys[ci] = R.keys[0];
          } else {
            N.keys.push(parent.keys[ci]);
            parent.keys[ci] = R.keys.shift() as number;
            N.kids.push(R.kids.shift() as number);
          }
          redist++;
          handled = true;
          steps.push(`page ${N.id} underflowed and borrowed one entry from its right sibling ${R.id}; the separator in ${parent.id} moved with it`);
        }
      }

      if (!handled) {
        const sibId = leftId !== null ? leftId : rightId;
        if (sibId === null) break;
        mergeTries++;
        const sib = d.pages[sibId];
        const extra = node.leaf ? 0 : 1;
        if (sib.keys.length + node.keys.length + extra > cap) {
          steps.push(
            `page ${node.id} is under the threshold but its sibling ${sibId} has ${sib.keys.length} keys, so the merge would not fit — the attempt fails and the page stays under-full`,
          );
          break;
        }
        if (leftId !== null) {
          const L = d.edit(leftId);
          const N = d.edit(node.id);
          if (N.leaf) {
            L.keys.push(...N.keys);
            L.next = N.next;
            if (N.next !== null) d.edit(N.next).prev = L.id;
            steps.push(`page ${N.id} merged into its left sibling ${L.id}; the separator in ${parent.id} was simply dropped`);
          } else {
            L.keys.push(parent.keys[ci - 1], ...N.keys);
            L.kids.push(...N.kids);
            steps.push(`internal page ${N.id} merged into ${L.id}, pulling separator ${parent.keys[ci - 1]} down out of ${parent.id}`);
          }
          parent.keys.splice(ci - 1, 1);
          parent.kids.splice(ci, 1);
          N.state = 'free';
          N.keys = [];
          N.kids = [];
          N.next = null;
          N.prev = null;
        } else {
          const R = d.edit(rightId as number);
          const N = d.edit(node.id);
          if (N.leaf) {
            N.keys.push(...R.keys);
            N.next = R.next;
            if (R.next !== null) d.edit(R.next).prev = N.id;
            steps.push(`right sibling ${R.id} merged into page ${N.id}; the separator in ${parent.id} was simply dropped`);
          } else {
            N.keys.push(parent.keys[ci], ...R.keys);
            N.kids.push(...R.kids);
            steps.push(`right sibling ${R.id} merged into ${N.id}, pulling separator ${parent.keys[ci]} down out of ${parent.id}`);
          }
          parent.keys.splice(ci, 1);
          parent.kids.splice(ci + 1, 1);
          R.state = 'free';
          R.keys = [];
          R.kids = [];
          R.next = null;
          R.prev = null;
        }
        mergeOk++;
      }
      idx--;
    }

    const rootPage = d.pages[d.root];
    if (!rootPage.leaf && rootPage.kids.length === 1) {
      const childId = rootPage.kids[0];
      const old = d.edit(d.root);
      old.state = 'free';
      old.keys = [];
      old.kids = [];
      d.root = childId;
      collapsed = true;
      steps.push(`the root was left with a single child, so it was freed and page ${childId} became the root — the tree lost a level`);
    }
  }

  const empties = Object.values(d.pages).filter((p) => p.state === 'empty').length;
  const head =
    steps.length === 0
      ? `Deleted ${key} from leaf ${leafId}: one page written.`
      : `Deleted ${key}: ${d.written.size} pages written.`;
  let body =
    steps.length > 0
      ? steps.join('; ') + '.'
      : mode === 'postgres'
        ? `Leaf ${leafId} is now ${Math.round((d.pages[leafId].keys.length / cap) * 100)}% full and nbtree leaves it exactly like that: there is no merge and no redistribution, only the entry removal.`
        : `Leaf ${leafId} is still at or above its occupancy floor (${d.pages[leafId].keys.length}/${cap}), so no rebalancing is needed.`;
  if (collapsed) body += ' Collapsing the root is the only way a B+tree ever gets shorter.';
  if (mode === 'postgres' && empties > 0) body += ` ${empties} empty page${empties === 1 ? '' : 's'} now sit in the tree doing nothing until VACUUM runs.`;

  return finish(d, {
    head,
    body,
    op: 'delete',
    keyLabel: String(key),
    outcome:
      mergeOk > 0
        ? `${mergeOk} merge${mergeOk === 1 ? '' : 's'}${collapsed ? ' + root collapse' : ''}`
        : redist > 0
          ? `${redist} redistribution${redist === 1 ? '' : 's'}`
          : mergeTries > 0
            ? 'merge attempt failed'
            : mode === 'postgres' && empties > 0
              ? 'page left empty'
              : 'in place',
    redist,
    mergeTries,
    mergeOk,
  });
}

/* ------------------------------------------------------------------ vacuum */

function runVacuum(t: Tree): Res {
  const d = new Draft(t);
  const levels = levelsOf(t);
  const rightmost = new Set(levels.map((lv) => lv[lv.length - 1]));
  const steps: string[] = [];

  // Stage 3 first, so each page advances exactly one stage per pass.
  for (const p of Object.values(d.pages)) {
    if (p.state === 'deleted') {
      const q = d.edit(p.id);
      q.state = 'free';
      steps.push(`page ${q.id} is old enough that no scan can still reach it, so it goes on the free space map and can be reused`);
    }
  }
  for (const p of Object.values(t.pages)) {
    if (p.state === 'halfdead') {
      const q = d.edit(p.id);
      if (q.prev !== null) d.edit(q.prev).next = q.next;
      if (q.next !== null) d.edit(q.next).prev = q.prev;
      q.prev = null;
      q.next = null;
      q.state = 'deleted';
      steps.push(`half-dead page ${q.id} was unlinked from its siblings and marked deleted with a safe transaction id`);
    }
  }
  for (const p of Object.values(t.pages)) {
    if (p.state !== 'empty') continue;
    if (rightmost.has(p.id)) {
      steps.push(`page ${p.id} is empty but it is the rightmost page on its level, and nbtree never deletes those — it stays`);
      continue;
    }
    // Remove the downlink from the parent, which is what makes the page half-dead.
    let parentId: number | null = null;
    for (const q of Object.values(d.pages)) {
      if (!q.leaf && q.kids.includes(p.id) && q.state === 'live') parentId = q.id;
    }
    if (parentId === null) continue;
    const parent = d.edit(parentId);
    const ci = parent.kids.indexOf(p.id);
    parent.kids.splice(ci, 1);
    parent.keys.splice(Math.max(ci - 1, 0), 1);
    const q = d.edit(p.id);
    q.state = 'halfdead';
    steps.push(`the downlink to page ${q.id} was removed from parent ${parent.id}; page ${q.id} is now half-dead — no longer reachable from above, still in the sibling chain`);

    // A parent left with no children goes in the same pass: _bt_pagedel deletes the whole
    // chain of now-empty ancestors at once, and leaving a childless page reachable from
    // above would make the keyspace it covers unroutable.
    let child = parent;
    while (child.kids.length === 0 && child.id !== d.root && !rightmost.has(child.id)) {
      let gpId: number | null = null;
      for (const q2 of Object.values(d.pages)) {
        if (!q2.leaf && q2.kids.includes(child.id) && q2.state === 'live') gpId = q2.id;
      }
      if (gpId === null) break;
      const gp = d.edit(gpId);
      const gi = gp.kids.indexOf(child.id);
      gp.kids.splice(gi, 1);
      gp.keys.splice(Math.max(gi - 1, 0), 1);
      child.state = 'halfdead';
      steps.push(`internal page ${child.id} lost its last child, so the whole branch goes in the same pass — its downlink was removed from ${gp.id} and it is half-dead too`);
      child = gp;
    }
  }

  if (steps.length === 0) {
    return finish(d, {
      head: 'VACUUM found nothing to do.',
      body: 'No page in this index is completely empty, and nbtree will not touch a page that still holds even one entry.',
      op: 'VACUUM',
      keyLabel: '—',
      outcome: 'no work',
    });
  }
  return finish(d, {
    head: `VACUUM: ${d.written.size} pages written.`,
    body:
      steps.join('; ') +
      '. Page deletion is two separate log records — mark half-dead, then unlink — which is why a crash can leave a half-dead page behind for a later VACUUM to finish. Each press here advances every page one stage so you can see the intermediate states.',
    op: 'VACUUM',
    keyLabel: '—',
    outcome: `${steps.length} action${steps.length === 1 ? '' : 's'}`,
  });
}

/* ------------------------------------------------------------- seed + setup */

const SEED_KEYS = (() => {
  const rng = makeRng(0xb7ee);
  const out: number[] = [];
  while (out.length < 14) {
    const k = 4 + Math.floor(rng() * 92);
    if (!out.includes(k)) out.push(k);
  }
  return out;
})();

const STREAM = (() => {
  const rng = makeRng(97531);
  return Array.from({ length: 120 }, () => 2 + Math.floor(rng() * 96));
})();

function emptyTree(): Tree {
  return {
    pages: { 1: { id: 1, leaf: true, keys: [], kids: [], next: null, prev: null, state: 'live' } },
    root: 1,
    nextId: 2,
  };
}

function buildInitial(cap: number): Tree {
  let t = emptyTree();
  for (const k of SEED_KEYS) t = insertKey(t, k, cap).tree;
  return t;
}

/* -------------------------------------------------------------- the drawing */

const SLOT = 24;
const NODE_H = 36;
const LEVEL_GAP = 54;
const NODE_GAP = 22;
const TOP = 16;

type Placed = { id: number; x: number; y: number; level: number };

function place(t: Tree, cap: number): { placed: Placed[]; xs: Record<number, number>; w: number; levels: number[][] } {
  const levels = levelsOf(t);
  const W = cap * SLOT + 16;
  const xs: Record<number, number> = {};
  const leaves = levels[levels.length - 1];
  leaves.forEach((id, i) => {
    xs[id] = i * (W + NODE_GAP);
  });
  for (let l = levels.length - 2; l >= 0; l--) {
    let after = 0;
    for (const id of levels[l]) {
      const kids = t.pages[id].kids;
      // A page whose last downlink has been removed has no children to centre over.
      xs[id] =
        kids.length === 0
          ? after
          : kids.reduce((a, k) => a + (xs[k] ?? 0) + W / 2, 0) / kids.length - W / 2;
      after = xs[id] + W + NODE_GAP;
    }
  }
  const placed: Placed[] = [];
  levels.forEach((lv, l) => {
    for (const id of lv) placed.push({ id, x: xs[id], y: TOP + l * (NODE_H + LEVEL_GAP), level: l });
  });
  return { placed, xs, w: W, levels };
}

const STATE_LABEL: Record<PState, string> = {
  live: 'live',
  empty: 'empty — awaiting VACUUM',
  halfdead: 'half-dead — downlink removed',
  deleted: 'deleted — awaiting recycle',
  free: 'free — on the FSM',
};

/* ------------------------------------------------------------ the component */

export default function BtreeSplitMergeConsole() {
  const [cap, setCap] = useState(4);
  const [mode, setMode] = useState<Mode>('textbook');
  const [threshold, setThreshold] = useState(50);
  const [key, setKey] = useState(50);
  const [seq, setSeq] = useState(0);
  const [tree, setTree] = useState<Tree>(() => buildInitial(4));
  const [written, setWritten] = useState<number[]>([]);
  const [created, setCreated] = useState<number[]>([]);
  const [head, setHead] = useState('A B+tree of 14 keys, four keys to a page.');
  const [body, setBody] = useState(
    'Insert keys until a leaf overflows and watch the split promote a separator; delete keys and watch the mode decide whether the tree repairs itself, merges lazily, or just gets emptier.',
  );
  const [log, setLog] = useState<LogRow[]>([]);
  const [tot, setTot] = useState({ pages: 0, splits: 0, redist: 0, mergeOk: 0, mergeTries: 0 });
  const [ref, width] = useSize(820);
  const tip = useTip();

  const apply = (r: Res) => {
    setTree(r.tree);
    setWritten(r.written);
    setCreated(r.created);
    setHead(r.head);
    setBody(r.body);
    setTot((p) => ({
      pages: p.pages + r.written.length,
      splits: p.splits + r.splits,
      redist: p.redist + r.redist,
      mergeOk: p.mergeOk + r.mergeOk,
      mergeTries: p.mergeTries + r.mergeTries,
    }));
    setLog((l) =>
      [
        ...l,
        {
          n: l.length + 1,
          op: r.op,
          key: r.keyLabel,
          outcome: r.outcome,
          pages: r.written.length,
          height: levelsOf(r.tree).length,
        },
      ].slice(-30),
    );
  };

  const reset = (nextCap: number) => {
    setCap(nextCap);
    setTree(buildInitial(nextCap));
    setWritten([]);
    setCreated([]);
    setLog([]);
    setSeq(0);
    setTot({ pages: 0, splits: 0, redist: 0, mergeOk: 0, mergeTries: 0 });
    setHead(`Rebuilt with ${nextCap} keys to a page.`);
    setBody(
      `Occupancy floor for the textbook rules is ceil(${nextCap}/2) = ${minLeafOf(nextCap)} keys in a leaf and ` +
        `${minInnerOf(nextCap)} separators in an internal page. Real fanout is in the hundreds; this is the same algebra at a readable size.`,
    );
  };

  const insertStream = () => {
    let t = tree;
    const w = new Set<number>();
    const c = new Set<number>();
    let splits = 0;
    let n = 0;
    for (let i = 0; i < 5; i++) {
      const k = STREAM[(seq + i) % STREAM.length];
      const r = insertKey(t, k, cap);
      t = r.tree;
      r.written.forEach((x) => w.add(x));
      r.created.forEach((x) => c.add(x));
      splits += r.splits;
      n += r.written.length;
    }
    setSeq(seq + 5);
    setTree(t);
    setWritten([...w]);
    setCreated([...c]);
    setHead(`Five keys inserted: ${splits} split${splits === 1 ? '' : 's'}, ${n} page writes.`);
    setBody(
      'Splits are rare per insert and expensive when they happen — that is the whole shape of B+tree write cost. Amortized over a run of inserts, each key costs a little more than one page write.',
    );
    setTot((p) => ({ ...p, pages: p.pages + n, splits: p.splits + splits }));
    setLog((l) =>
      [...l, { n: l.length + 1, op: 'insert ×5', key: '—', outcome: `${splits} split${splits === 1 ? '' : 's'}`, pages: n, height: levelsOf(t).length }].slice(-30),
    );
  };

  const { placed, w: W, levels } = place(tree, cap);
  const dead = Object.values(tree.pages)
    .filter((p) => p.state === 'halfdead' || p.state === 'deleted' || p.state === 'free')
    .sort((a, b) => a.id - b.id);
  const leafIds = levels[levels.length - 1];
  const liveLeaves = leafIds.map((id) => tree.pages[id]);
  const keyCount = liveLeaves.reduce((a, p) => a + p.keys.length, 0);
  const livePages = Object.values(tree.pages).filter((p) => p.state === 'live' || p.state === 'empty').length;
  const avgFill = liveLeaves.length ? (keyCount / (liveLeaves.length * cap)) * 100 : 0;
  const minLeaf = minLeafOf(cap);

  const maxX = placed.reduce((a, p) => Math.max(a, p.x + W), 0);
  const deadLaneY = TOP + levels.length * (NODE_H + LEVEL_GAP) + 6;
  const svgW = Math.max(width, maxX + 10, dead.length * 96 + 120);
  const svgH = deadLaneY + (dead.length > 0 ? 46 : 0) + 8;
  const lastPages = log.length ? log[log.length - 1].pages : 0;

  const strokeFor = (p: Page) => {
    if (created.includes(p.id)) return { s: 'var(--viz-6)', w: 2.5 };
    if (written.includes(p.id)) return { s: 'var(--viz-2)', w: 2.5 };
    if (p.state !== 'live') return { s: 'var(--viz-stale)', w: 1.5 };
    return { s: 'var(--viz-border)', w: 1.5 };
  };

  return (
    <VizPanel
      title="Insert, split, delete, merge — and what each one writes"
      subtitle="A real B+tree at a readable fanout. Insert until a page overflows and the split promotes a separator; delete and watch the chosen engine's rules decide between redistribution, a merge, or an empty page nobody reclaims. Click any key in a leaf to delete it."
      controls={
        <>
          <Segmented
            label="Deletion rules"
            value={mode}
            onChange={setMode}
            options={[
              { value: 'textbook', label: 'textbook', title: 'Underflow below ceil(cap/2): borrow from a sibling, else merge' },
              { value: 'postgres', label: 'PostgreSQL nbtree', title: 'Never merges; only completely empty pages are deleted, by VACUUM' },
              { value: 'innodb', label: 'InnoDB', title: 'Merge attempted only under MERGE_THRESHOLD' },
            ]}
          />
          <Slider label="Keys per page" min={3} max={6} value={cap} onChange={reset} format={(n) => `${n}`} />
          <Slider
            label="MERGE_THRESHOLD"
            min={10}
            max={90}
            step={5}
            value={threshold}
            disabled={mode !== 'innodb'}
            onChange={setThreshold}
            format={(n) => `${n}%`}
          />
          <Slider label="Key" min={1} max={99} value={key} onChange={setKey} />
          <Button primary onClick={() => apply(insertKey(tree, key, cap))}>
            Insert
          </Button>
          <Button onClick={() => apply(deleteKey(tree, key, cap, mode, threshold))}>Delete</Button>
          <Button onClick={insertStream} title="Five keys from a fixed deterministic stream">
            Insert ×5
          </Button>
          <Button
            onClick={() => apply(runVacuum(tree))}
            disabled={mode !== 'postgres'}
            title={mode === 'postgres' ? 'Advance every dead page one stage' : 'Only PostgreSQL defers page deletion to VACUUM'}
          >
            VACUUM
          </Button>
          <Button onClick={() => reset(cap)}>Reset</Button>
        </>
      }
      legend={
        <Legend
          items={[
            { label: 'rewritten by the last operation', color: 'var(--viz-2)', shape: 'line' },
            { label: 'allocated by the last operation', color: 'var(--viz-6)', shape: 'line' },
            { label: `fill bar — at or above ${minLeaf}/${cap}`, color: 'var(--viz-1)' },
            { label: 'fill bar — below the half-full floor', color: 'var(--viz-warning)' },
            { label: 'empty / half-dead / deleted / free', color: 'var(--viz-stale)' },
          ]}
        />
      }
      stats={
        <Stats
          items={[
            { label: 'Height', value: levels.length, hint: 'Levels from root to leaf — page reads for a point lookup' },
            { label: 'Live pages', value: livePages },
            { label: 'Keys', value: keyCount },
            { label: 'Avg leaf fill', value: `${avgFill.toFixed(0)}%`, hint: 'Keys held over keys the leaves could hold' },
            { label: 'Pages written, last op', value: lastPages },
            { label: 'Pages written, total', value: fmtNum(tot.pages) },
            { label: 'Splits', value: tot.splits },
            {
              label: 'Merges',
              value: `${tot.mergeOk} / ${tot.mergeTries}`,
              hint: 'Successful merges over attempted merges — InnoDB exposes exactly this pair as index_page_merge_successful and index_page_merge_attempts',
            },
            { label: 'Redistributions', value: tot.redist, hint: 'Borrowing from a sibling instead of merging — textbook rules only' },
            { label: 'Dead pages', value: dead.length, hint: 'Empty, half-dead, deleted or free — space the index still occupies' },
          ]}
        />
      }
      note={
        <Note>
          <strong>{head}</strong> {body}
        </Note>
      }
      table={
        <>
          <table className="viz-table">
            <thead>
              <tr>
                <th>Page</th>
                <th>Level</th>
                <th>Kind</th>
                <th>Entries</th>
                <th>Fill</th>
                <th>Contents</th>
                <th>Left / right sibling</th>
                <th>State</th>
              </tr>
            </thead>
            <tbody>
              {placed.map((pl) => {
                const p = tree.pages[pl.id];
                return (
                  <tr key={p.id}>
                    <td>{p.id}</td>
                    <td>{pl.level}</td>
                    <td>{p.leaf ? 'leaf' : pl.level === 0 ? 'root' : 'internal'}</td>
                    <td>{p.keys.length}</td>
                    <td>{Math.round((p.keys.length / cap) * 100)}%</td>
                    <td>{p.leaf ? p.keys.join(' ') || '—' : p.keys.join(' | ') || '—'}</td>
                    <td>{p.leaf ? `${p.prev ?? '—'} / ${p.next ?? '—'}` : '—'}</td>
                    <td>{STATE_LABEL[p.state]}</td>
                  </tr>
                );
              })}
              {dead.map((p) => (
                <tr key={p.id}>
                  <td>{p.id}</td>
                  <td>—</td>
                  <td>{p.leaf ? 'leaf' : 'internal'}</td>
                  <td>{p.keys.length}</td>
                  <td>—</td>
                  <td>—</td>
                  <td>{p.leaf ? `${p.prev ?? '—'} / ${p.next ?? '—'}` : '—'}</td>
                  <td>{STATE_LABEL[p.state]}</td>
                </tr>
              ))}
            </tbody>
          </table>
          <table className="viz-table">
            <thead>
              <tr>
                <th>#</th>
                <th>Operation</th>
                <th>Key</th>
                <th>Outcome</th>
                <th>Pages written</th>
                <th>Height after</th>
              </tr>
            </thead>
            <tbody>
              {log.length === 0 ? (
                <tr>
                  <td colSpan={6}>No operations yet.</td>
                </tr>
              ) : (
                log.map((r) => (
                  <tr key={r.n}>
                    <td>{r.n}</td>
                    <td>{r.op}</td>
                    <td>{r.key}</td>
                    <td>{r.outcome}</td>
                    <td>{r.pages}</td>
                    <td>{r.height}</td>
                  </tr>
                ))
              )}
            </tbody>
          </table>
        </>
      }
    >
      <div ref={ref}>
        <TooltipHost>
          <svg width={svgW} height={svgH} role="img" aria-label="A B+tree with its pages, separators, leaf chain and dead pages">
            {/* downlinks */}
            {placed.map((pl) => {
              const p = tree.pages[pl.id];
              if (p.leaf) return null;
              const k = p.kids.length;
              return p.kids.map((cid, i) => {
                const cpl = placed.find((q) => q.id === cid);
                if (!cpl) return null;
                const x1 = pl.x + ((i + 0.5) * W) / k;
                const y1 = pl.y + NODE_H;
                const x2 = cpl.x + W / 2;
                const y2 = cpl.y;
                return (
                  <path
                    key={`${pl.id}-${cid}`}
                    d={`M${x1} ${y1} C ${x1} ${y1 + 20}, ${x2} ${y2 - 20}, ${x2} ${y2}`}
                    fill="none"
                    stroke="var(--viz-axis)"
                    strokeWidth={1.2}
                  />
                );
              });
            })}

            {/* leaf sibling chain */}
            {leafIds.map((id) => {
              const p = tree.pages[id];
              if (p.next === null) return null;
              const a = placed.find((q) => q.id === id);
              const b = placed.find((q) => q.id === p.next);
              if (!a || !b) return null;
              const y = a.y + NODE_H / 2;
              return (
                <g key={`chain-${id}`}>
                  <line x1={a.x + W} y1={y} x2={b.x} y2={y} stroke="var(--viz-ink-muted)" strokeWidth={1} strokeDasharray="3 3" />
                  <path d={`M${b.x} ${y} l -5 -3 l 0 6 z`} fill="var(--viz-ink-muted)" />
                  <path d={`M${a.x + W} ${y} l 5 -3 l 0 6 z`} fill="var(--viz-ink-muted)" />
                </g>
              );
            })}

            {/* pages */}
            {placed.map((pl) => {
              const p = tree.pages[pl.id];
              const st = strokeFor(p);
              const fillFrac = p.keys.length / cap;
              const below = p.leaf && p.keys.length < minLeaf;
              return (
                <g key={pl.id}>
                  <g
                    {...tip(
                      <>
                        <strong>
                          page {p.id} · {p.leaf ? (levels.length === 1 ? 'root leaf' : 'leaf') : pl.level === 0 ? 'root' : 'internal'}
                        </strong>
                        <br />
                        {p.keys.length}/{cap} entries · {Math.round(fillFrac * 100)}% full · {STATE_LABEL[p.state]}
                        <br />
                        {p.leaf
                          ? `keys ${p.keys.join(', ') || '(none)'} · prev ${p.prev ?? '—'} · next ${p.next ?? '—'}`
                          : `separators ${p.keys.join(', ') || '(none)'} · children ${p.kids.join(', ')}`}
                      </>,
                    )}
                  >
                    <rect
                      x={pl.x}
                      y={pl.y}
                      width={W}
                      height={NODE_H}
                      rx={7}
                      fill="var(--viz-plane)"
                      stroke={st.s}
                      strokeWidth={st.w}
                      strokeDasharray={p.state === 'empty' ? '5 3' : undefined}
                    />
                    <text x={pl.x + 3} y={pl.y - 4} fill="var(--viz-ink-muted)">
                      {p.id}
                      {p.state === 'empty' ? ' · empty' : ''}
                    </text>
                  </g>

                  {p.leaf ? (
                    <>
                      {Array.from({ length: cap }, (_, i) => {
                        const k = p.keys[i];
                        const sx = pl.x + 8 + i * SLOT;
                        if (k === undefined) {
                          return (
                            <rect
                              key={i}
                              x={sx}
                              y={pl.y + 6}
                              width={SLOT - 4}
                              height={17}
                              rx={3}
                              fill="none"
                              stroke="var(--viz-grid)"
                              strokeWidth={1}
                              strokeDasharray="2 2"
                            />
                          );
                        }
                        return (
                          <g
                            key={i}
                            style={{ cursor: 'pointer' }}
                            onClick={() => apply(deleteKey(tree, k, cap, mode, threshold))}
                            {...tip(
                              <>
                                <strong>key {k}</strong>
                                <br />
                                in leaf {p.id}, slot {i} — click to delete it
                              </>,
                            )}
                          >
                            <rect x={sx} y={pl.y + 6} width={SLOT - 4} height={17} rx={3} fill="var(--viz-neutral)" stroke="var(--viz-border)" />
                            <text x={sx + (SLOT - 4) / 2} y={pl.y + 18} textAnchor="middle" fill="var(--viz-ink)">
                              {k}
                            </text>
                          </g>
                        );
                      })}
                      <rect x={pl.x + 8} y={pl.y + NODE_H - 7} width={cap * SLOT - 4} height={3} rx={1.5} fill="var(--viz-grid)" />
                      <rect
                        x={pl.x + 8}
                        y={pl.y + NODE_H - 7}
                        width={Math.max(fillFrac * (cap * SLOT - 4), 0)}
                        height={3}
                        rx={1.5}
                        fill={below ? 'var(--viz-warning)' : 'var(--viz-1)'}
                      />
                    </>
                  ) : (
                    <>
                      {p.kids.map((cid, i) => (
                        <circle key={`d${cid}`} cx={pl.x + ((i + 0.5) * W) / p.kids.length} cy={pl.y + NODE_H - 6} r={2.5} fill="var(--viz-axis)" />
                      ))}
                      {p.keys.map((sk, i) => (
                        <text
                          key={`s${i}`}
                          x={pl.x + ((i + 1) * W) / p.kids.length}
                          y={pl.y + 20}
                          textAnchor="middle"
                          fill="var(--viz-ink)"
                          fontWeight={600}
                        >
                          {sk}
                        </text>
                      ))}
                    </>
                  )}
                </g>
              );
            })}

            {/* dead lane */}
            {dead.length > 0 ? (
              <g>
                <text x={0} y={deadLaneY + 12} fill="var(--viz-ink-muted)">
                  not in the tree:
                </text>
                {dead.map((p, i) => (
                  <g
                    key={p.id}
                    {...tip(
                      <>
                        <strong>page {p.id}</strong>
                        <br />
                        {STATE_LABEL[p.state]}
                      </>,
                    )}
                  >
                    <rect
                      x={110 + i * 96}
                      y={deadLaneY}
                      width={88}
                      height={22}
                      rx={5}
                      fill="var(--viz-plane)"
                      stroke={written.includes(p.id) ? 'var(--viz-2)' : 'var(--viz-stale)'}
                      strokeWidth={written.includes(p.id) ? 2.5 : 1.5}
                      strokeDasharray="4 3"
                    />
                    <text x={110 + i * 96 + 44} y={deadLaneY + 15} textAnchor="middle" fill="var(--viz-ink-2)">
                      {p.id} · {p.state === 'halfdead' ? 'half-dead' : p.state}
                    </text>
                  </g>
                ))}
              </g>
            ) : null}
          </svg>
        </TooltipHost>
      </div>
    </VizPanel>
  );
}
