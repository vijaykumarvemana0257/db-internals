/**
 * Two space-partitioning index families, run for real on small inputs.
 *
 * ART tab — the Adaptive Radix Tree of Leis, Kemper and Neumann (ICDE 2013):
 *  - Span of 8 bits: each inner node is indexed by one key byte. Inner node types Node4, Node16, Node48 and
 *    Node256 hold 2-4, 5-16, 17-48 and 49-256 children; a full node grows to the next type on insert.
 *  - Shrinking uses DuckDB's thresholds (base_node.cpp, node48.cpp, node256.cpp): Node16 becomes Node4 below 4
 *    children, Node48 becomes Node16 below 12, Node256 becomes Node48 at 36 or fewer. A Node4 left with one
 *    child is removed and its byte is folded into the child's prefix. The paper only says "underfull".
 *  - Lazy expansion: a leaf hangs as high as it can while still being distinguished from every other key.
 *    Path compression: one-child inner nodes are removed and their bytes stored as the child's prefix.
 *  - Prefix storage for lookups: pessimistic compares every stored prefix byte; optimistic stores only the
 *    length and skips the bytes, so the leaf must compare the whole key; hybrid compares up to 8 bytes and
 *    skips the rest (the paper's choice).
 *  - Memory, from the paper's Table I (16-byte header, 8-byte pointers): Node4 52, Node16 160, Node48 656,
 *    Node256 2064 bytes. The header holds up to 8 prefix bytes; pessimistic storage adds a byte per prefix
 *    byte beyond 8. Leaves are either a tuple id in a tagged child pointer (0 extra bytes, key re-read from
 *    the table as HyPer does) or a leaf object holding an 8-byte value plus a copy of the key.
 *  - Keys are binary-comparable byte strings: text gets a 0x00 terminator so no key is a prefix of another;
 *    integers are 4-byte big-endian, with the sign bit optionally flipped.
 *  - The comparison B+tree is a MODEL: in-memory nodes of 16 keys, 16-byte header, 8-byte key slots and
 *    pointers (leaf 272 bytes, inner 280), keys longer than 8 bytes stored out of line, built by inserting
 *    the live keys in insertion order with half splits.
 *
 * SP-GiST tab — PostgreSQL's quad_point_ops and kd_point_ops (spgquadtreeproc.c, spgkdtreeproc.c):
 *  - quad: picksplit takes the MEAN of the points as the centroid prefix and makes 4 unlabeled nodes;
 *    choose sends a point to getQuadrant(centroid, point); inner_consistent for "point <@ box" follows all
 *    four nodes when the centroid is inside the box, otherwise only the quadrants holding the box's corners.
 *  - kd: picksplit sorts by y at even levels and x at odd levels and splits at the median into 2 nodes.
 *  - A leaf list splits when it exceeds "leaf capacity" — a stand-in for PostgreSQL's real rule, which is that
 *    the chain of leaf tuples under one node no longer fits on its 8 kB index page (a short chain is moved to
 *    another page instead). If picksplit puts every tuple in one node, the core builds an allTheSame inner tuple
 *    with 8 equivalent nodes; the lab deals tuples to them in turn and inserts into the emptiest, where the core
 *    picks at random.
 */
import { useEffect, useMemo, useState } from 'react';
import { VizPanel, Segmented, Slider, Choice, Check, Button, Legend, Stats, Note, makeRng, fmtNum, fmtBytes, useSize } from './Viz';

/* =========================================================================================== ART */

export type Bytes = number[];
export type NodeType = 4 | 16 | 48 | 256;
export type Collapse = 'none' | 'lazy' | 'full';
export type PrefixMode = 'pessimistic' | 'optimistic' | 'hybrid';
export type LeafMode = 'tid' | 'key';

export type ArtLeaf = { kind: 'leaf'; id: number; key: Bytes };
export type ArtInner = { kind: 'inner'; id: number; type: NodeType; prefix: Bytes; keys: number[]; children: ArtNode[] };
export type ArtNode = ArtLeaf | ArtInner;

export const NODE_BYTES: Record<NodeType, number> = { 4: 52, 16: 160, 48: 656, 256: 2064 };
export const HEADER_PREFIX_BYTES = 8;
const NEXT: Record<NodeType, NodeType> = { 4: 16, 16: 48, 48: 256, 256: 256 };

export function equalBytes(a: Bytes, b: Bytes) {
  if (a.length !== b.length) return false;
  for (let i = 0; i < a.length; i++) if (a[i] !== b[i]) return false;
  return true;
}

export function makeArt(collapse: Collapse) {
  let nextId = 1;
  let root: ArtNode | null = null;
  const leaf = (key: Bytes): ArtLeaf => ({ kind: 'leaf', id: nextId++, key });
  const inner = (prefix: Bytes): ArtInner => ({ kind: 'inner', id: nextId++, type: 4, prefix, keys: [], children: [] });

  const addChild = (n: ArtInner, byte: number, child: ArtNode) => {
    if (n.keys.length >= n.type) n.type = NEXT[n.type];
    let i = 0;
    while (i < n.keys.length && n.keys[i] < byte) i++;
    n.keys.splice(i, 0, byte);
    n.children.splice(i, 0, child);
  };
  /** A new key's subtree below depth d: one leaf, or (no lazy expansion) a one-way node per remaining byte. */
  const tail = (key: Bytes, d: number): ArtNode => {
    if (collapse !== 'none' || d >= key.length) return leaf(key);
    const n = inner([]);
    addChild(n, key[d], tail(key, d + 1));
    return n;
  };

  const ins = (node: ArtNode | null, key: Bytes, depth: number): ArtNode => {
    if (!node) return tail(key, depth);
    if (node.kind === 'leaf') {
      if (equalBytes(node.key, key) || depth >= key.length) return node;
      let p = 0;
      while (depth + p < key.length && key[depth + p] === node.key[depth + p]) p++;
      if (depth + p >= key.length || depth + p >= node.key.length) return node; // not prefix-free; refuse
      if (collapse === 'full') {
        const n = inner(key.slice(depth, depth + p));
        addChild(n, key[depth + p], leaf(key));
        addChild(n, node.key[depth + p], node);
        return n;
      }
      let bottom = inner([]);
      addChild(bottom, key[depth + p], leaf(key));
      addChild(bottom, node.key[depth + p], node);
      for (let i = p - 1; i >= 0; i--) {
        const up = inner([]);
        addChild(up, key[depth + i], bottom);
        bottom = up;
      }
      return bottom;
    }
    if (collapse === 'full' && node.prefix.length) {
      let p = 0;
      while (p < node.prefix.length && node.prefix[p] === key[depth + p]) p++;
      if (p !== node.prefix.length) {
        if (depth + p >= key.length) return node;
        const n = inner(node.prefix.slice(0, p));
        const oldByte = node.prefix[p];
        node.prefix = node.prefix.slice(p + 1);
        addChild(n, key[depth + p], tail(key, depth + p + 1));
        addChild(n, oldByte, node);
        return n;
      }
      depth += node.prefix.length;
    }
    if (depth >= key.length) return node;
    const i = node.keys.indexOf(key[depth]);
    if (i >= 0) node.children[i] = ins(node.children[i], key, depth + 1);
    else addChild(node, key[depth], tail(key, depth + 1));
    return node;
  };

  const del = (node: ArtNode | null, key: Bytes, depth: number): ArtNode | null => {
    if (!node) return null;
    if (node.kind === 'leaf') return equalBytes(node.key, key) ? null : node;
    if (collapse === 'full' && node.prefix.length) {
      for (let p = 0; p < node.prefix.length; p++) if (node.prefix[p] !== key[depth + p]) return node;
      depth += node.prefix.length;
    }
    const i = node.keys.indexOf(key[depth]);
    if (i < 0) return node;
    const child = node.children[i];
    const next = del(child, key, depth + 1);
    if (next) {
      node.children[i] = next;
    } else {
      node.keys.splice(i, 1);
      node.children.splice(i, 1);
      const c = node.keys.length;
      // DuckDB's shrink thresholds.
      if (node.type === 16 && c < 4) node.type = 4;
      else if (node.type === 48 && c < 12) node.type = 16;
      else if (node.type === 256 && c <= 36) node.type = 48;
    }
    if (node.keys.length === 0) return null;
    if (node.keys.length === 1) {
      const only = node.children[0];
      if (collapse !== 'none' && only.kind === 'leaf') return only;
      if (collapse === 'full' && only.kind === 'inner') {
        only.prefix = [...node.prefix, node.keys[0], ...only.prefix];
        return only;
      }
    }
    return node;
  };

  return {
    insert(key: Bytes) {
      root = ins(root, key, 0);
    },
    remove(key: Bytes) {
      root = del(root, key, 0);
    },
    get root() {
      return root;
    },
  };
}

export type ArtCounts = { 4: number; 16: number; 48: number; 256: number; leaves: number; prefixBytes: number; extraPrefixBytes: number; height: number; oneWay: number };

export function countArt(root: ArtNode | null): ArtCounts {
  const c: ArtCounts = { 4: 0, 16: 0, 48: 0, 256: 0, leaves: 0, prefixBytes: 0, extraPrefixBytes: 0, height: 0, oneWay: 0 };
  const walk = (n: ArtNode, d: number) => {
    if (n.kind === 'leaf') {
      c.leaves++;
      c.height = Math.max(c.height, d);
      return;
    }
    c[n.type]++;
    if (n.keys.length === 1) c.oneWay++;
    c.prefixBytes += n.prefix.length;
    c.extraPrefixBytes += Math.max(0, n.prefix.length - HEADER_PREFIX_BYTES);
    for (const ch of n.children) walk(ch, d + 1);
  };
  if (root) walk(root, 0);
  return c;
}

export function artMemory(root: ArtNode | null, prefixMode: PrefixMode, leafMode: LeafMode) {
  const c = countArt(root);
  let leafBytes = 0;
  const walk = (n: ArtNode) => {
    if (n.kind === 'leaf') leafBytes += leafMode === 'key' ? 8 + n.key.length : 0;
    else n.children.forEach(walk);
  };
  if (root) walk(root);
  const nodes = { 4: c[4] * NODE_BYTES[4], 16: c[16] * NODE_BYTES[16], 48: c[48] * NODE_BYTES[48], 256: c[256] * NODE_BYTES[256] };
  const prefix = prefixMode === 'pessimistic' ? c.extraPrefixBytes : 0;
  const total = nodes[4] + nodes[16] + nodes[48] + nodes[256] + prefix + leafBytes;
  return { counts: c, nodes, prefix, leafBytes, total };
}

export type LookupStep = { node: number; kind: 'inner' | 'leaf'; what: string; examined: number; oneWayByte?: number };

export function artLookup(root: ArtNode | null, key: Bytes, collapse: Collapse, prefixMode: PrefixMode, fmt: (b: Bytes) => string) {
  const steps: LookupStep[] = [];
  const path: number[] = [];
  let node = root;
  let depth = 0;
  let skipped = false;
  let examined = 0;
  const done = (found: boolean, where: 'prefix' | 'child' | 'leaf' | 'empty') => ({ found, where, steps, path, examined, skipped });
  if (!node) return done(false, 'empty');
  while (node) {
    path.push(node.id);
    if (node.kind === 'leaf') {
      if (collapse === 'none') {
        steps.push({ node: node.id, kind: 'leaf', what: 'leaf reached: every key byte was consumed on the way down, so no comparison is needed', examined: 0 });
        return done(true, 'leaf');
      }
      const from = skipped ? 0 : depth;
      let n = 0;
      let match = key.length === node.key.length;
      for (let i = from; i < Math.max(key.length, node.key.length); i++) {
        n++;
        if (key[i] !== node.key[i]) {
          match = false;
          break;
        }
      }
      examined += n;
      steps.push({
        node: node.id,
        kind: 'leaf',
        what: `${skipped ? 'compare the WHOLE key with leaf' : 'compare the remaining bytes with leaf'} ${fmt(node.key)}: ${match ? 'match' : 'mismatch'} after ${n} byte${n === 1 ? '' : 's'}`,
        examined: n,
      });
      return done(match, 'leaf');
    }
    let ex = 0;
    const notes: string[] = [];
    if (collapse === 'full' && node.prefix.length) {
      const len = node.prefix.length;
      const cmp = prefixMode === 'pessimistic' ? len : prefixMode === 'optimistic' ? 0 : Math.min(len, HEADER_PREFIX_BYTES);
      for (let i = 0; i < cmp; i++) {
        ex++;
        if (node.prefix[i] !== key[depth + i]) {
          examined += ex;
          steps.push({ node: node.id, kind: 'inner', what: `N${node.type}: prefix byte ${i + 1} of ${len} differs — not found`, examined: ex });
          return done(false, 'prefix');
        }
      }
      if (cmp < len) {
        skipped = true;
        notes.push(cmp === 0 ? `skip ${len} prefix byte${len === 1 ? '' : 's'} unchecked` : `compare ${cmp} prefix bytes, skip ${len - cmp} unchecked`);
      } else notes.push(`compare ${len} prefix byte${len === 1 ? '' : 's'}: match`);
      depth += len;
    }
    if (depth >= key.length) {
      examined += ex;
      steps.push({ node: node.id, kind: 'inner', what: `N${node.type}: key ends inside the tree — not found`, examined: ex });
      return done(false, 'child');
    }
    const b = key[depth];
    const idx = node.keys.indexOf(b);
    ex++;
    const find =
      node.type === 4
        ? `loop over ${node.keys.length} sorted key${node.keys.length === 1 ? '' : 's'}`
        : node.type === 16
          ? 'one SIMD compare of all 16 keys'
          : node.type === 48
            ? `child_index[${b}] then children[slot]`
            : `children[${b}] directly`;
    notes.push(`${find} for byte ${fmt([b])}: ${idx >= 0 ? 'child found' : 'no child — not found'}`);
    examined += ex;
    const oneWay = collapse !== 'full' && node.keys.length === 1 && idx === 0;
    steps.push({ node: node.id, kind: 'inner', what: `N${node.type}: ${notes.join('; ')}`, examined: ex, ...(oneWay ? { oneWayByte: b } : {}) });
    if (idx < 0) return done(false, 'child');
    node = node.children[idx];
    depth++;
  }
  return done(false, 'child');
}

/** In-order traversal: leaves in byte order. */
export function artLeaves(root: ArtNode | null) {
  const out: Bytes[] = [];
  const walk = (n: ArtNode) => {
    if (n.kind === 'leaf') out.push(n.key);
    else n.children.forEach(walk);
  };
  if (root) walk(root);
  return out;
}

/* --------------------------------------------------------------------------- B+tree (model) */

export const BT_CAP = 16;
export const BT_LEAF_BYTES = 16 + BT_CAP * 16;
export const BT_INNER_BYTES = 16 + BT_CAP * 8 + (BT_CAP + 1) * 8;

type BNode = { leaf: boolean; keys: Bytes[]; kids: BNode[] };

export function cmpBytes(a: Bytes, b: Bytes) {
  const n = Math.min(a.length, b.length);
  for (let i = 0; i < n; i++) if (a[i] !== b[i]) return { c: a[i] - b[i], examined: i + 1 };
  return { c: a.length - b.length, examined: n + (a.length === b.length ? 0 : 1) };
}

export function buildBtree(keys: Bytes[]) {
  let root: BNode = { leaf: true, keys: [], kids: [] };
  const ins = (n: BNode, k: Bytes): { sep: Bytes; right: BNode } | null => {
    if (n.leaf) {
      let i = 0;
      while (i < n.keys.length && cmpBytes(n.keys[i], k).c < 0) i++;
      if (i < n.keys.length && cmpBytes(n.keys[i], k).c === 0) return null;
      n.keys.splice(i, 0, k);
      if (n.keys.length <= BT_CAP) return null;
      const mid = n.keys.length >> 1;
      const right: BNode = { leaf: true, keys: n.keys.slice(mid), kids: [] };
      n.keys = n.keys.slice(0, mid);
      return { sep: right.keys[0], right };
    }
    let i = 0;
    while (i < n.keys.length && cmpBytes(n.keys[i], k).c <= 0) i++;
    const s = ins(n.kids[i], k);
    if (!s) return null;
    n.keys.splice(i, 0, s.sep);
    n.kids.splice(i + 1, 0, s.right);
    if (n.keys.length <= BT_CAP) return null;
    const mid = n.keys.length >> 1;
    const sep = n.keys[mid];
    const right: BNode = { leaf: false, keys: n.keys.slice(mid + 1), kids: n.kids.slice(mid + 1) };
    n.keys = n.keys.slice(0, mid);
    n.kids = n.kids.slice(0, mid + 1);
    return { sep, right };
  };
  for (const k of keys) {
    const s = ins(root, k);
    if (s) root = { leaf: false, keys: [s.sep], kids: [root, s.right] };
  }
  let leaves = 0;
  let inners = 0;
  let height = 0;
  const walk = (n: BNode, d: number) => {
    height = Math.max(height, d + 1);
    if (n.leaf) leaves++;
    else {
      inners++;
      n.kids.forEach((k) => walk(k, d + 1));
    }
  };
  walk(root, 0);
  const live = new Map<string, Bytes>();
  for (const k of keys) live.set(k.join(','), k);
  const keyBytes = [...live.values()].reduce((s, k) => s + (k.length > 8 ? k.length : 0), 0);
  const total = leaves * BT_LEAF_BYTES + inners * BT_INNER_BYTES + keyBytes;

  const lookup = (k: Bytes) => {
    let n = root;
    let examined = 0;
    let comparisons = 0;
    const search = (arr: Bytes[], upper: boolean) => {
      let lo = 0;
      let hi = arr.length;
      let eq = false;
      while (lo < hi) {
        const mid = (lo + hi) >> 1;
        const r = cmpBytes(arr[mid], k);
        comparisons++;
        examined += r.examined;
        if (r.c === 0) eq = true;
        if (upper ? r.c <= 0 : r.c < 0) lo = mid + 1;
        else hi = mid;
      }
      return { pos: lo, eq };
    };
    for (;;) {
      if (n.leaf) return { found: search(n.keys, false).eq, examined, comparisons, nodes: height };
      n = n.kids[search(n.keys, true).pos];
    }
  };
  return { leaves, inners, height, keyBytes, total, lookup };
}

/* ----------------------------------------------------------------------------- key sets */

export type PresetId = 'words' | 'urls' | 'siblings' | 'dense' | 'signed';
export const SIBLING_CHARS = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwx';

export const PRESET_VALUES: Record<PresetId, (string | number)[]> = {
  words: ['romane', 'romanus', 'romulus', 'rubens', 'ruber', 'rubicon', 'rubicundus', 'rom', 'rune', 'rust'],
  urls: [
    'shop.example.com/cart',
    'shop.example.com/catalog/shirts',
    'shop.example.com/catalog/shoes',
    'shop.example.com/checkout',
    'shop.example.com/customer/7',
    'shop.example.com/customer/42',
    'shop.example.com/help',
    'shop.example.com/orders/1001',
    'shop.example.com/orders/1002',
    'shop.example.com/search',
  ],
  siblings: SIBLING_CHARS.split('').map((ch) => `id-${ch}`),
  dense: Array.from({ length: 256 }, (_, i) => i),
  signed: [3, -1, 7, -8, 0, 5, -4, 1, -6, 8, -2, 4, -7, 2, 6, -3, -5],
};

export const isIntPreset = (p: PresetId) => p === 'dense' || p === 'signed';

export function encodeText(s: string): Bytes {
  const out: Bytes = [];
  for (let i = 0; i < s.length; i++) out.push(s.charCodeAt(i) & 0xff);
  out.push(0);
  return out;
}

export function encodeInt(v: number, flipSign: boolean): Bytes {
  const u = v >>> 0;
  const b = [(u >>> 24) & 0xff, (u >>> 16) & 0xff, (u >>> 8) & 0xff, u & 0xff];
  if (flipSign) b[0] ^= 0x80;
  return b;
}

export function decodeInt(b: Bytes, flipSign: boolean) {
  const b0 = flipSign ? b[0] ^ 0x80 : b[0];
  return ((b0 << 24) | (b[1] << 16) | (b[2] << 8) | b[3]) | 0;
}

export function encodeValue(v: string | number, preset: PresetId, flipSign: boolean): Bytes {
  if (typeof v === 'number') return encodeInt(v, preset === 'signed' ? flipSign : false);
  return encodeText(v);
}

/** Bytes as the learner reads them: text for printable strings (terminator hidden), hex for integers. */
export function fmtKey(b: Bytes, ints: boolean) {
  if (ints) return b.map((x) => x.toString(16).padStart(2, '0')).join(' ');
  let s = '';
  for (const x of b) s += x === 0 ? '∅' : x >= 0x20 && x < 0x7f ? String.fromCharCode(x) : `\\x${x.toString(16).padStart(2, '0')}`;
  return s;
}
export function fmtKeyShort(b: Bytes, ints: boolean) {
  if (ints) return fmtKey(b, true);
  const t = b[b.length - 1] === 0 ? b.slice(0, -1) : b;
  return fmtKey(t, false);
}

export type ArtConfig = { preset: PresetId; custom: (string | number)[]; inserted: number; deleted: number; collapse: Collapse; flipSign: boolean };

/** Insert the first `inserted` keys of preset ++ custom, then delete the newest `deleted` of them, newest first. */
export function runArt(cfg: ArtConfig) {
  const values = [...PRESET_VALUES[cfg.preset], ...cfg.custom];
  const n = Math.max(0, Math.min(cfg.inserted, values.length));
  const m = Math.max(0, Math.min(cfg.deleted, n));
  const keys = values.slice(0, n).map((v) => encodeValue(v, cfg.preset, cfg.flipSign));
  const art = makeArt(cfg.collapse);
  for (const k of keys) art.insert(k);
  for (let i = 0; i < m; i++) art.remove(keys[n - 1 - i]);
  const live = keys.slice(0, n - m);
  return { root: art.root, live, values: values.slice(0, n - m), total: values.length };
}

export type ProbeKind = 'hit' | 'prefix' | 'child';

/** Probe keys come from the fully collapsed tree of the live keys, so they stay fixed while modes change. */
export function makeProbes(live: Bytes[], ints: boolean) {
  const out: Record<ProbeKind, Bytes | null> = { hit: null, prefix: null, child: null };
  if (!live.length) return out;
  out.hit = live[Math.floor((live.length - 1) / 2)];
  const art = makeArt('full');
  for (const k of live) art.insert(k);
  const root = art.root;
  if (!root || root.kind === 'leaf') return out;
  const firstLeaf = (n: ArtNode): ArtLeaf => (n.kind === 'leaf' ? n : firstLeaf(n.children[0]));
  let best: { node: ArtInner; depth: number } | null = null;
  const walk = (n: ArtNode, depth: number) => {
    if (n.kind === 'leaf') return;
    if (n.prefix.length && (!best || n.prefix.length >= best.node.prefix.length)) best = { node: n, depth };
    n.children.forEach((c) => walk(c, depth + n.prefix.length + 1));
  };
  walk(root, 0);
  const lo = ints ? 0 : 0x21;
  const hi = ints ? 255 : 0x7e;
  const liveSet = new Set(live.map((k) => k.join(',')));
  if (best) {
    const { node, depth } = best as { node: ArtInner; depth: number };
    const k = firstLeaf(node).key.slice();
    const pos = depth + node.prefix.length - 1;
    const orig = k[pos];
    for (let d = 1; d < 256; d++) {
      const cand = orig + (d % 2 ? (d + 1) / 2 : -d / 2);
      if (cand >= lo && cand <= hi && cand !== orig) {
        k[pos] = cand;
        if (!liveSet.has(k.join(','))) break;
      }
    }
    out.prefix = k;
  }
  // child miss at the root node's branching byte
  const r = root as ArtInner;
  const leafKey = firstLeaf(r.children[0]).key.slice();
  const pos = r.prefix.length;
  const used = new Set(r.keys);
  for (let d = 1; d < 256; d++) {
    const cand = leafKey[pos] + (d % 2 ? (d + 1) / 2 : -d / 2);
    if (cand >= lo && cand <= hi && !used.has(cand)) {
      leafKey[pos] = cand;
      out.child = liveSet.has(leafKey.join(',')) ? null : leafKey;
      break;
    }
  }
  return out;
}

/* ====================================================================================== SP-GiST */

export type Pt = { x: number; y: number };
export type Box = { lx: number; ly: number; hx: number; hy: number };
export type Opclass = 'quad' | 'kd';
export type QLeaf = { kind: 'leaf'; pts: number[]; level: number; region: Box };
export type QInner = { kind: 'inner'; level: number; region: Box; centroid?: Pt; coord?: number; axis?: 'x' | 'y'; allTheSame?: boolean; children: (QNode | null)[] };
export type QNode = QLeaf | QInner;

/** PostgreSQL's getQuadrant(), including its tie-breaking on the centroid's lines. */
export function getQuadrant(c: Pt, p: Pt) {
  if (p.y >= c.y && p.x >= c.x) return 1;
  if (p.y < c.y && p.x >= c.x) return 2;
  if (p.y <= c.y && p.x < c.x) return 3;
  return 4; // p.y > c.y && p.x < c.x
}

function quadrantBox(b: Box, c: Pt, q: number): Box {
  if (q === 1) return { lx: c.x, ly: c.y, hx: b.hx, hy: b.hy };
  if (q === 2) return { lx: c.x, ly: b.ly, hx: b.hx, hy: c.y };
  if (q === 3) return { lx: b.lx, ly: b.ly, hx: c.x, hy: c.y };
  return { lx: b.lx, ly: c.y, hx: c.x, hy: b.hy };
}

export const PLANE: Box = { lx: 0, ly: 0, hx: 100, hy: 100 };
/** spgdoinsert.c checkAllTheSame(): "out->nNodes = 8; arbitrary number of child nodes", tuples assigned i % 8. */
export const ALL_THE_SAME_NODES = 8;

export type TraceLine = string;

export function buildSpgist(pts: Pt[], opclass: Opclass, capacity: number) {
  let root: QNode = { kind: 'leaf', pts: [], level: 0, region: PLANE };
  let trace: TraceLine[] = [];
  let splits = 0;
  const f = (v: number) => v.toFixed(1);

  const picksplit = (leaf: QLeaf): QInner => {
    splits++;
    const ps = leaf.pts.map((i) => pts[i]);
    if (opclass === 'quad') {
      const c = { x: ps.reduce((s, p) => s + p.x, 0) / ps.length, y: ps.reduce((s, p) => s + p.y, 0) / ps.length };
      const children: (QNode | null)[] = [null, null, null, null];
      for (const i of leaf.pts) {
        const q = getQuadrant(c, pts[i]) - 1;
        if (!children[q]) children[q] = { kind: 'leaf', pts: [], level: leaf.level + 1, region: quadrantBox(leaf.region, c, q + 1) };
        (children[q] as QLeaf).pts.push(i);
      }
      if (children.filter(Boolean).length === 1) {
        // Every tuple landed in one quadrant (identical points): the core overrides picksplit with an
        // allTheSame tuple whose equivalent nodes share the tuples.
        const all = leaf.pts.slice();
        const same: (QNode | null)[] = Array.from({ length: ALL_THE_SAME_NODES }, (_, q) => ({ kind: 'leaf', pts: all.filter((_, j) => j % ALL_THE_SAME_NODES === q), level: leaf.level + 1, region: leaf.region }));
        trace.push(`picksplit @ level ${leaf.level}: every tuple fell in one quadrant, so the core builds an allTheSame inner tuple and deals them over ${ALL_THE_SAME_NODES} equivalent nodes`);
        return { kind: 'inner', level: leaf.level, region: leaf.region, centroid: c, allTheSame: true, children: same };
      }
      trace.push(
        `picksplit @ level ${leaf.level}: ${ps.length} leaf tuples > capacity ${capacity}; centroid = mean (${f(c.x)}, ${f(c.y)}); 4 nodes get ${children.map((ch) => (ch ? (ch as QLeaf).pts.length : 0)).join(' / ')}`,
      );
      return { kind: 'inner', level: leaf.level, region: leaf.region, centroid: c, children };
    }
    const axis: 'x' | 'y' = leaf.level % 2 ? 'x' : 'y';
    const sorted = leaf.pts.slice().sort((a, b) => pts[a][axis] - pts[b][axis]);
    const middle = sorted.length >> 1;
    const coord = pts[sorted[middle]][axis];
    const lowRegion = axis === 'x' ? { ...leaf.region, hx: coord } : { ...leaf.region, hy: coord };
    const highRegion = axis === 'x' ? { ...leaf.region, lx: coord } : { ...leaf.region, ly: coord };
    const children: (QNode | null)[] = [
      { kind: 'leaf', pts: sorted.slice(0, middle), level: leaf.level + 1, region: lowRegion },
      { kind: 'leaf', pts: sorted.slice(middle), level: leaf.level + 1, region: highRegion },
    ];
    trace.push(`picksplit @ level ${leaf.level}: ${ps.length} leaf tuples > capacity ${capacity}; sort by ${axis}, median ${axis} = ${f(coord)}; 2 nodes get ${middle} / ${sorted.length - middle}`);
    return { kind: 'inner', level: leaf.level, region: leaf.region, coord, axis, children };
  };

  const ins = (node: QNode, i: number): QNode => {
    const p = pts[i];
    if (node.kind === 'leaf') {
      node.pts.push(i);
      if (node.pts.length <= capacity) {
        trace.push(`leaf page has room: append leaf tuple (${f(p.x)}, ${f(p.y)}) to the list (${node.pts.length} of ${capacity})`);
        return node;
      }
      return picksplit(node);
    }
    let nodeN: number;
    if (node.allTheSame) {
      const sizes = node.children.map((c) => (c && c.kind === 'leaf' ? c.pts.length : 1e9));
      nodeN = sizes.indexOf(Math.min(...sizes));
      trace.push(`choose @ level ${node.level}: allTheSame tuple → spgMatchNode into any equivalent node (node ${nodeN})`);
    } else if (opclass === 'quad') {
      nodeN = getQuadrant(node.centroid as Pt, p) - 1;
      trace.push(`choose @ level ${node.level}: centroid (${f(node.centroid!.x)}, ${f(node.centroid!.y)}) → spgMatchNode, quadrant ${nodeN + 1}`);
    } else {
      nodeN = p[node.axis!] < node.coord! ? 0 : 1;
      trace.push(`choose @ level ${node.level}: split ${node.axis} = ${f(node.coord!)} → spgMatchNode, node ${nodeN} (${node.axis} ${nodeN ? '≥' : '<'} ${f(node.coord!)}), levelAdd 1`);
    }
    const child = node.children[nodeN];
    if (!child) {
      const region = opclass === 'quad' ? quadrantBox(node.region, node.centroid!, nodeN + 1) : node.axis === 'x' ? (nodeN ? { ...node.region, lx: node.coord! } : { ...node.region, hx: node.coord! }) : nodeN ? { ...node.region, ly: node.coord! } : { ...node.region, hy: node.coord! };
      node.children[nodeN] = { kind: 'leaf', pts: [i], level: node.level + 1, region };
      trace.push(`downlink was empty: start a new leaf list`);
    } else node.children[nodeN] = ins(child, i);
    return node;
  };

  for (let i = 0; i < pts.length; i++) {
    trace = [];
    root = ins(root, i);
  }
  return { root, lastTrace: trace, splits };
}

export function inBox(p: Pt, b: Box) {
  return p.x >= b.lx && p.x <= b.hx && p.y >= b.ly && p.y <= b.hy;
}

export function spgistQuery(root: QNode, pts: Pt[], box: Box, opclass: Opclass) {
  const visited = new Set<QNode>();
  const trace: string[] = [];
  let innerVisited = 0;
  let leafLists = 0;
  let tested = 0;
  const matches: number[] = [];
  const testedIdx: number[] = [];
  const f = (v: number) => v.toFixed(1);
  const walk = (n: QNode) => {
    visited.add(n);
    if (n.kind === 'leaf') {
      leafLists++;
      for (const i of n.pts) {
        tested++;
        testedIdx.push(i);
        if (inBox(pts[i], box)) matches.push(i);
      }
      return;
    }
    innerVisited++;
    let which: number[];
    if (n.allTheSame) {
      which = n.children.map((_, i) => i);
      trace.push(`inner_consistent @ level ${n.level}: allTheSame tuple → all nodes or none; all ${n.children.length} here`);
    } else if (opclass === 'quad') {
      const c = n.centroid!;
      if (inBox(c, box)) {
        which = [1, 2, 3, 4];
        trace.push(`inner_consistent @ level ${n.level}: centroid (${f(c.x)}, ${f(c.y)}) is inside the box → all 4 nodes`);
      } else {
        const r = new Set([getQuadrant(c, { x: box.lx, y: box.ly }), getQuadrant(c, { x: box.lx, y: box.hy }), getQuadrant(c, { x: box.hx, y: box.hy }), getQuadrant(c, { x: box.hx, y: box.ly })]);
        which = [...r].sort();
        trace.push(`inner_consistent @ level ${n.level}: centroid (${f(c.x)}, ${f(c.y)}) outside the box → quadrants of its corners {${which.join(', ')}}`);
      }
      which = which.map((q) => q - 1);
    } else {
      const lo = n.axis === 'x' ? box.lx : box.ly;
      const hi = n.axis === 'x' ? box.hx : box.hy;
      which = hi < n.coord! ? [0] : lo > n.coord! ? [1] : [0, 1];
      trace.push(`inner_consistent @ level ${n.level}: split ${n.axis} = ${f(n.coord!)}, box ${n.axis} ∈ [${f(lo)}, ${f(hi)}] → node${which.length > 1 ? 's' : ''} {${which.join(', ')}}`);
    }
    for (const w of which) {
      const ch = n.children[w];
      if (ch) walk(ch);
    }
  };
  walk(root);
  return { visited, trace, innerVisited, leafLists, tested, matches, testedIdx };
}

export function spgistShape(root: QNode) {
  let inner = 0;
  let leaves = 0;
  let minD = Infinity;
  let maxD = 0;
  const walk = (n: QNode, d: number) => {
    if (n.kind === 'leaf') {
      if (!n.pts.length) return;
      leaves++;
      minD = Math.min(minD, d);
      maxD = Math.max(maxD, d);
      return;
    }
    inner++;
    n.children.forEach((c) => c && walk(c, d + 1));
  };
  walk(root, 0);
  return { inner, leaves, minDepth: minD === Infinity ? 0 : minD, maxDepth: maxD };
}

export type Distribution = 'uniform' | 'clustered' | 'diagonal';

export function makePoints(dist: Distribution, n: number, seed = 7): Pt[] {
  const rng = makeRng(seed);
  const gauss = () => {
    const u = Math.max(1e-6, rng());
    const v = rng();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
  };
  const clamp = (v: number) => Math.max(0.5, Math.min(99.5, v));
  const r1 = (v: number) => Math.round(v * 10) / 10;
  const out: Pt[] = [];
  for (let i = 0; i < n; i++) {
    let x: number;
    let y: number;
    if (dist === 'uniform') {
      x = rng() * 100;
      y = rng() * 100;
    } else if (dist === 'clustered') {
      if (rng() < 0.7) {
        x = 26 + gauss() * 5;
        y = 72 + gauss() * 5;
      } else {
        x = 72 + gauss() * 9;
        y = 28 + gauss() * 9;
      }
    } else {
      const t = rng();
      x = t * 100;
      y = t * 100 + gauss() * 2.5;
    }
    out.push({ x: r1(clamp(x)), y: r1(clamp(y)) });
  }
  return out;
}

/* ============================================================================================ UI */

const TYPE_COLOR: Record<NodeType, string> = { 4: 'var(--viz-1)', 16: 'var(--viz-2)', 48: 'var(--viz-3)', 256: 'var(--viz-4)' };
const LEAF_COLOR = 'var(--viz-5)';
// Neutral ink, not --viz-8: that red would read as the 'not found' state colour in the same legend.
const EXTRA_PREFIX_COLOR = 'var(--viz-ink-2)';
const BT_NODE_COLOR = 'var(--viz-7)';
// A lighter shade of the B+tree colour, not --viz-6: that green would read as the 'found' state colour.
const BT_KEY_COLOR = 'color-mix(in srgb, var(--viz-7) 45%, var(--viz-surface))';

const PRESET_OPTIONS: { value: PresetId; label: string }[] = [
  { value: 'words', label: 'Words with shared prefixes' },
  { value: 'urls', label: 'URLs with a long common prefix' },
  { value: 'siblings', label: 'Many siblings: id-0 … id-x' },
  { value: 'dense', label: 'Dense integers 0 … 255' },
  { value: 'signed', label: 'Signed integers −8 … 8' },
];

const textW = (s: string, px = 6.6) => s.length * px;
const clip = (s: string, max: number) => (s.length > max ? `…${s.slice(s.length - max + 1)}` : s);

type LItem =
  | { t: 'inner'; id: number; x: number; y: number; w: number; h: number; node: ArtInner; l1: string; l2: string }
  | { t: 'chain'; id: number; x: number; y: number; w: number; h: number; len: number; l2: string; ids: number[] }
  | { t: 'leaf'; id: number; x: number; y: number; w: number; h: number; label: string; full: string }
  | { t: 'grid'; id: number; x: number; y: number; w: number; h: number; occupied: number[]; label: string; type: NodeType };
type LEdge = { x1: number; y1: number; x2: number; y2: number; label: string; to: number };

const ROW = 70;
const chainLabel = (len: number) => `${len} × N4, one child each`;
const chainW = (len: number, l2: string) => Math.max(textW(chainLabel(len), 7.2), textW(l2)) + 18;
const innerW = (n: ArtInner, l2: string) => Math.max(textW(`N${n.type} · ${n.keys.length}/${n.type}`, 7.2), textW(l2), 44) + 18;

function layoutArt(root: ArtNode | null, ints: boolean, collapse: Collapse, prefixMode: PrefixMode) {
  const items: LItem[] = [];
  const edges: LEdge[] = [];
  if (!root) return { items, edges, width: 320, height: 60 };
  const byteLabel = (b: number) => (ints ? b.toString(16).padStart(2, '0') : b === 0 ? '∅' : fmtKey([b], false));
  const prefixText = (n: ArtInner) => {
    if (collapse !== 'full' || !n.prefix.length) return '';
    const len = n.prefix.length;
    const show = (b: Bytes) => (ints ? fmtKey(b, true) : `"${fmtKey(b, false)}"`);
    if (prefixMode === 'optimistic') return `prefix: skip ${len} byte${len === 1 ? '' : 's'}`;
    if (prefixMode === 'hybrid' && len > HEADER_PREFIX_BYTES) return `${show(n.prefix.slice(0, HEADER_PREFIX_BYTES))} +${len - HEADER_PREFIX_BYTES} skipped`;
    return `prefix ${show(n.prefix)}`;
  };
  const chainText = (bytes: Bytes) => (ints ? fmtKey(bytes, true) : `"${clip(fmtKey(bytes, false), 22)}"`);
  // one-way chain (only drawn as a chain when path compression is off)
  const chainOf = (n: ArtNode) => {
    const nodes: ArtInner[] = [];
    let cur: ArtNode = n;
    while (collapse !== 'full' && cur.kind === 'inner' && cur.keys.length === 1) {
      nodes.push(cur);
      cur = cur.children[0];
    }
    return { nodes, end: cur };
  };
  const memo = new Map<number, number>();
  const measure = (n: ArtNode): number => {
    const hit = memo.get(n.id);
    if (hit !== undefined) return hit;
    let w: number;
    if (n.kind === 'leaf') w = Math.max(34, textW(clip(fmtKeyShort(n.key, ints), 16)) + 12);
    else {
      const ch = chainOf(n);
      if (ch.nodes.length >= 2) {
        const bytes = ch.nodes.map((c) => c.keys[0]);
        w = Math.max(chainW(ch.nodes.length, chainText(bytes)), measure(ch.end));
      } else {
        const self = innerW(n, prefixText(n));
        const kids = n.keys.length > 16 ? 150 : n.children.reduce((s, c) => s + measure(c), 0) + (n.children.length - 1) * 10;
        w = Math.max(self, kids);
      }
    }
    memo.set(n.id, w);
    return w;
  };
  let maxY = 0;
  const place = (n: ArtNode, x0: number, y: number, parent: { x: number; y: number; label: string } | null) => {
    const W = measure(n);
    const cx = x0 + W / 2;
    const edge = (h: number) => {
      if (parent) edges.push({ x1: parent.x, y1: parent.y, x2: cx, y2: y, label: parent.label, to: n.id });
      maxY = Math.max(maxY, y + h);
    };
    if (n.kind === 'leaf') {
      const w = Math.max(34, textW(clip(fmtKeyShort(n.key, ints), 16)) + 12);
      items.push({ t: 'leaf', id: n.id, x: cx - w / 2, y, w, h: 22, label: clip(fmtKeyShort(n.key, ints), 16), full: fmtKeyShort(n.key, ints) });
      edge(22);
      return;
    }
    const ch = chainOf(n);
    if (ch.nodes.length >= 2) {
      const bytes = ch.nodes.map((c) => c.keys[0]);
      const l2c = chainText(bytes);
      const w = chainW(ch.nodes.length, l2c);
      items.push({ t: 'chain', id: n.id, x: cx - w / 2, y, w, h: 36, len: ch.nodes.length, l2: l2c, ids: ch.nodes.map((c) => c.id) });
      edge(36);
      const last = ch.nodes[ch.nodes.length - 1];
      place(ch.end, cx - measure(ch.end) / 2, y + ROW, { x: cx, y: y + 36, label: byteLabel(last.keys[0]) });
      return;
    }
    const l2 = prefixText(n);
    const w = innerW(n, l2);
    items.push({ t: 'inner', id: n.id, x: cx - w / 2, y, w, h: 36, node: n, l1: `N${n.type} · ${n.keys.length}/${n.type}`, l2 });
    edge(36);
    if (n.keys.length > 16) {
      const leaves = n.children.filter((c) => c.kind === 'leaf').length;
      const label = leaves === n.children.length ? `${n.children.length} leaves` : `${n.children.length} children (${n.children.length - leaves} inner)`;
      items.push({ t: 'grid', id: -n.id, x: cx - 72, y: y + ROW - 10, w: 144, h: 58, occupied: n.keys.slice(), label, type: n.type });
      edges.push({ x1: cx, y1: y + 36, x2: cx, y2: y + ROW - 10, label: '', to: -n.id });
      maxY = Math.max(maxY, y + ROW + 48);
      return;
    }
    const total = n.children.reduce((s, c) => s + measure(c), 0) + (n.children.length - 1) * 10;
    let x = cx - total / 2;
    n.children.forEach((c, i) => {
      place(c, x, y + ROW, { x: cx, y: y + 36, label: byteLabel(n.keys[i]) });
      x += measure(c) + 10;
    });
  };
  const width = measure(root) + 20;
  place(root, 10, 8, null);
  return { items, edges, width: Math.max(320, width), height: maxY + 12 };
}

function ArtPanel() {
  const [preset, setPreset] = useState<PresetId>('words');
  const [custom, setCustom] = useState<(string | number)[]>([]);
  const [inserted, setInserted] = useState(PRESET_VALUES.words.length);
  const [deleted, setDeleted] = useState(0);
  const [collapse, setCollapse] = useState<Collapse>('full');
  const [prefixMode, setPrefixMode] = useState<PrefixMode>('pessimistic');
  const [leafMode, setLeafMode] = useState<LeafMode>('key');
  const [probeKind, setProbeKind] = useState<ProbeKind>('hit');
  const [flipSign, setFlipSign] = useState(true);
  const [draft, setDraft] = useState('');
  const [message, setMessage] = useState('');

  const ints = isIntPreset(preset);
  const total = PRESET_VALUES[preset].length + custom.length;
  const run = useMemo(() => runArt({ preset, custom, inserted, deleted, collapse, flipSign }), [preset, custom, inserted, deleted, collapse, flipSign]);
  const mem = useMemo(() => artMemory(run.root, prefixMode, leafMode), [run, prefixMode, leafMode]);
  const bt = useMemo(() => buildBtree(run.live), [run]);
  const probes = useMemo(() => makeProbes(run.live, ints), [run, ints]);
  const probe = probes[probeKind];
  const fmt = (b: Bytes) => fmtKeyShort(b, ints);
  const look = useMemo(() => (probe ? artLookup(run.root, probe, collapse, prefixMode, (b) => fmtKeyShort(b, ints)) : null), [run, probe, collapse, prefixMode, ints]);
  const btLook = probe ? bt.lookup(probe) : null;
  const lay = useMemo(() => layoutArt(run.root, ints, collapse, prefixMode), [run, ints, collapse, prefixMode]);
  const onPath = new Set(look ? look.path : []);
  const mergedSteps = useMemo(() => {
    const out: { what: string; examined: number }[] = [];
    if (!look) return out;
    let run: number[] = [];
    const flush = () => {
      if (!run.length) return;
      out.push(
        run.length === 1
          ? { what: `N4: loop over 1 sorted key for byte ${fmtKeyShort([run[0]], ints) || '∅'}: child found`, examined: 1 }
          : { what: `${run.length} one-way N4 nodes, one byte each: ${ints ? fmtKey(run, true) : `"${fmtKey(run, false)}"`}`, examined: run.length },
      );
      run = [];
    };
    for (const st of look.steps) {
      if (st.oneWayByte !== undefined) run.push(st.oneWayByte);
      else {
        flush();
        out.push(st);
      }
    }
    flush();
    return out;
  }, [look, ints]);
  const [treeBox, boxW] = useSize(600);
  const rootItem = lay.items[0];
  const rootCenter = rootItem ? rootItem.x + rootItem.w / 2 : 0;
  const lastId = look && look.path.length ? look.path[look.path.length - 1] : -1;
  const endItem = lay.items.find((it) => (it.t === 'chain' ? it.ids.includes(lastId) : it.id === lastId));
  const endCenter = endItem ? endItem.x + endItem.w / 2 : rootCenter;
  useEffect(() => {
    const el = treeBox.current;
    if (!el || lay.width <= boxW * 1.2) return;
    // Show the root and the node where the lookup ended when both fit; otherwise follow the lookup to its end.
    const target = Math.abs(endCenter - rootCenter) < el.clientWidth - 120 ? (endCenter + rootCenter) / 2 : endCenter;
    el.scrollLeft = Math.max(0, target - el.clientWidth / 2);
  }, [lay, rootCenter, endCenter, boxW]);
  const liveCount = run.live.length;
  const c = mem.counts;
  const perKey = (b: number) => (liveCount ? fmtNum(b / liveCount, 1) : '—');
  const order = useMemo(() => artLeaves(run.root).map((k) => (ints ? String(decodeInt(k, preset === 'signed' ? flipSign : false)) : fmtKeyShort(k, false))), [run, ints, preset, flipSign]);

  const choosePreset = (p: PresetId) => {
    setPreset(p);
    setCustom([]);
    setInserted(PRESET_VALUES[p].length);
    setDeleted(0);
    setMessage('');
    setDraft('');
  };
  const addCustom = () => {
    const raw = draft.trim();
    if (!raw) return;
    let v: string | number = raw;
    if (ints) {
      const num = Number(raw);
      const ok = Number.isInteger(num) && (preset === 'signed' ? num >= -2147483648 && num <= 2147483647 : num >= 0 && num <= 4294967295);
      if (!ok) {
        setMessage(preset === 'signed' ? 'Enter a 32-bit signed integer.' : 'Enter an integer from 0 to 4294967295.');
        return;
      }
      v = num;
    } else if (raw.length > 32 || [...raw].some((ch) => ch.charCodeAt(0) < 0x20 || ch.charCodeAt(0) > 0x7e)) {
      setMessage('Use up to 32 printable ASCII characters.');
      return;
    }
    const all = [...PRESET_VALUES[preset], ...custom];
    if (all.some((x) => x === v)) {
      setMessage(`${raw} is already in the key set.`);
      return;
    }
    if (custom.length >= 12) {
      setMessage('Twelve custom keys is the limit here.');
      return;
    }
    setCustom([...custom, v]);
    setInserted(all.length + 1);
    setDeleted(0);
    setDraft('');
    setMessage(`Inserted ${raw}.`);
  };

  // the busiest inner node, for the growth narration
  let busiest: ArtInner | null = null;
  const walk = (n: ArtNode | null) => {
    if (!n || n.kind === 'leaf') return;
    if (!busiest || n.keys.length > busiest.keys.length) busiest = n;
    n.children.forEach(walk);
  };
  walk(run.root);
  const b = busiest as ArtInner | null;
  const growthNote = (() => {
    if (!b) return '';
    const k = b.keys.length;
    const natural: NodeType = k <= 4 ? 4 : k <= 16 ? 16 : k <= 48 ? 48 : 256;
    if (b.type !== natural)
      return `The widest node is still a Node${b.type} with only ${k} children: deletes shrink a node lazily (Node256 waits for 36 or fewer, Node48 for fewer than 12, Node16 for fewer than 4), so it does not flap between types.`;
    if (b.type === 256) return `The widest node is a Node256 with ${k} children: a key byte indexes its pointer array directly.`;
    const next = b.type === 4 ? 16 : b.type === 16 ? 48 : 256;
    return `The widest node is a Node${b.type} with ${k} of ${b.type} slots used; child ${b.type + 1} turns it into a Node${next}.`;
  })();

  const probeLabel = probe ? fmt(probe) : '';
  const where = look ? (look.found ? 'found' : look.where === 'prefix' ? 'rejected by a prefix comparison' : look.where === 'leaf' ? 'rejected only at the leaf' : 'rejected: no child for the byte') : '';

  const maxBar = Math.max(mem.total, bt.total, 1);
  const barW = 216;
  const sc = (v: number) => (v / maxBar) * barW;
  const artParts = [
    { v: mem.nodes[4], color: TYPE_COLOR[4], label: `Node4 × ${c[4]}` },
    { v: mem.nodes[16], color: TYPE_COLOR[16], label: `Node16 × ${c[16]}` },
    { v: mem.nodes[48], color: TYPE_COLOR[48], label: `Node48 × ${c[48]}` },
    { v: mem.nodes[256], color: TYPE_COLOR[256], label: `Node256 × ${c[256]}` },
    { v: mem.prefix, color: EXTRA_PREFIX_COLOR, label: 'prefix bytes past 8' },
    { v: mem.leafBytes, color: LEAF_COLOR, label: leafMode === 'key' ? `leaves × ${c.leaves}` : 'leaves (tagged tids)' },
  ];
  const btParts = [
    { v: bt.leaves * BT_LEAF_BYTES + bt.inners * BT_INNER_BYTES, color: BT_NODE_COLOR, label: `nodes × ${bt.leaves + bt.inners}` },
    { v: bt.keyBytes, color: BT_KEY_COLOR, label: 'out-of-line key bytes' },
  ];
  const memRows = [...artParts.map((p) => ({ ...p, who: 'ART' })), ...btParts.map((p) => ({ ...p, who: 'B+tree' }))].filter((p) => p.v > 0);
  const memH = 96 + memRows.length * 17;

  return (
    <>
      <div className="viz-controls">
        <Choice label="Key set" value={preset} onChange={choosePreset} options={PRESET_OPTIONS} />
        <Slider label="Keys inserted" min={0} max={total} value={Math.min(inserted, total)} onChange={(v) => { setInserted(v); setDeleted((d) => Math.min(d, v)); }} />
        <Slider label="Then delete newest" min={0} max={Math.min(inserted, total)} value={Math.min(deleted, inserted)} onChange={setDeleted} />
        <label className="viz-control">
          <span>{ints ? 'Your integer' : 'Your key'}</span>
          <input
            type="text"
            value={draft}
            aria-label={ints ? 'Integer key to insert' : 'String key to insert'}
            onChange={(e) => setDraft(e.currentTarget.value)}
            onKeyDown={(e) => { if (e.key === 'Enter') addCustom(); }}
            style={{ width: '9rem', padding: '0.3rem 0.5rem', border: '1px solid var(--viz-border)', borderRadius: 6, background: 'var(--viz-plane)', color: 'var(--viz-ink)' }}
          />
        </label>
        <Button onClick={addCustom}>Insert</Button>
        {preset === 'signed' ? <Check label="Flip the sign bit" checked={flipSign} onChange={setFlipSign} /> : null}
      </div>
      <div className="viz-controls">
        <Segmented
          label="Collapsing"
          value={collapse}
          onChange={setCollapse}
          options={[
            { value: 'none', label: 'None', title: 'One inner node per key byte' },
            { value: 'lazy', label: 'Lazy expansion' },
            { value: 'full', label: '+ Path compression' },
          ]}
        />
        {collapse === 'full' ? (
          <Segmented
            label="Prefix check"
            value={prefixMode}
            onChange={setPrefixMode}
            options={[
              { value: 'pessimistic', label: 'Pessimistic' },
              { value: 'optimistic', label: 'Optimistic' },
              { value: 'hybrid', label: 'Hybrid (8 bytes)' },
            ]}
          />
        ) : null}
        <Segmented
          label="Leaves"
          value={leafMode}
          onChange={setLeafMode}
          options={[
            { value: 'key', label: 'Leaf stores the key' },
            { value: 'tid', label: 'Tuple id in the pointer' },
          ]}
        />
        <Choice
          label="Look up"
          value={probeKind}
          onChange={setProbeKind}
          options={[
            { value: 'hit', label: 'A stored key' },
            { value: 'prefix', label: 'Missing: differs inside a compressed prefix' },
            { value: 'child', label: 'Missing: no child for a byte' },
          ]}
        />
      </div>
      {message ? <p style={{ fontSize: '0.75rem', color: 'var(--viz-ink-2)', margin: '0 0 0.5rem' }}>{message}</p> : null}
      <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.75rem', alignItems: 'flex-start' }}>
        <div ref={treeBox} style={{ flex: '1 1 380px', minWidth: 0, overflowX: 'auto' }} data-testid="art-tree-box">
          <svg
            viewBox={`0 0 ${lay.width} ${lay.height}`}
            width={lay.width}
            height={lay.height}
            style={{ maxWidth: lay.width > boxW * 1.2 ? 'none' : '100%' }}
            role="img"
            aria-label={`Adaptive radix tree with ${liveCount} keys: ${c[4]} Node4, ${c[16]} Node16, ${c[48]} Node48 and ${c[256]} Node256 inner nodes`}
            data-testid="art-tree"
          >
            {lay.edges.map((e, i) => {
              const hot = onPath.has(e.to) || (e.to > 0 && lay.items.some((it) => it.t === 'chain' && it.id === e.to && onPath.has(it.id)));
              return (
                <g key={`e${i}`}>
                  <line x1={e.x1} y1={e.y1} x2={e.x2} y2={e.y2} stroke={hot ? 'var(--viz-ink)' : 'var(--viz-axis)'} strokeWidth={hot ? 2.2 : 1} />
                  {e.label ? (
                    <text x={(e.x1 + e.x2) / 2 + 4} y={(e.y1 + e.y2) / 2 + 3} fontSize={11} fill={hot ? 'var(--viz-ink)' : 'var(--viz-ink-2)'} stroke="var(--viz-surface)" strokeWidth={3} paintOrder="stroke" style={{ fontWeight: hot ? 600 : 400 }}>
                      {e.label}
                    </text>
                  ) : null}
                </g>
              );
            })}
            {lay.items.map((it) => {
              if (it.t === 'grid') {
                return (
                  <g key={`g${it.id}`}>
                    <rect x={it.x} y={it.y} width={it.w} height={it.h} rx={5} fill="var(--viz-plane)" stroke="var(--viz-border)" />
                    {Array.from({ length: 256 }, (_, i) => (
                      <rect key={i} x={it.x + 8 + (i % 32) * 4} y={it.y + 6 + Math.floor(i / 32) * 4} width={3.4} height={3.4} fill={it.occupied.includes(i) ? TYPE_COLOR[it.type] : 'var(--viz-grid)'} />
                    ))}
                    <text x={it.x + it.w / 2} y={it.y + 51} textAnchor="middle" fontSize={10.5} fill="var(--viz-ink-2)">
                      {it.label}: filled = byte used
                    </text>
                  </g>
                );
              }
              if (it.t === 'leaf') {
                const hot = onPath.has(it.id);
                const last = it.id === lastId;
                return (
                  <g key={`l${it.id}`}>
                    <title>{it.full}</title>
                    <rect x={it.x} y={it.y} width={it.w} height={it.h} rx={4} fill="var(--viz-surface)" stroke={last ? (look?.found ? 'var(--viz-good)' : 'var(--viz-critical)') : LEAF_COLOR} strokeWidth={hot ? 3 : 1.5} />
                    <text x={it.x + it.w / 2} y={it.y + 15} textAnchor="middle" fontSize={11} fill="var(--viz-ink)">
                      {it.label}
                    </text>
                  </g>
                );
              }
              if (it.t === 'chain') {
                const hot = it.ids.some((id) => onPath.has(id));
                const last = it.ids.includes(lastId);
                return (
                  <g key={`c${it.id}`}>
                    <rect x={it.x} y={it.y} width={it.w} height={it.h} rx={5} fill="var(--viz-surface)" stroke={last ? 'var(--viz-critical)' : TYPE_COLOR[4]} strokeWidth={hot ? 3 : 1.5} strokeDasharray="5 3" />
                    <text x={it.x + it.w / 2} y={it.y + 15} textAnchor="middle" fontSize={11} fill="var(--viz-ink)" style={{ fontWeight: 600 }}>
                      {chainLabel(it.len)}
                    </text>
                    <text x={it.x + it.w / 2} y={it.y + 29} textAnchor="middle" fontSize={10.5} fill="var(--viz-ink-2)">
                      {it.l2}
                    </text>
                  </g>
                );
              }
              const hot = onPath.has(it.id);
              const last = it.id === lastId && look && !look.found;
              return (
                <g key={`n${it.id}`}>
                  <rect x={it.x} y={it.y} width={it.w} height={it.h} rx={5} fill="var(--viz-surface)" stroke={last ? 'var(--viz-critical)' : TYPE_COLOR[it.node.type]} strokeWidth={hot ? 3 : 1.8} />
                  <text x={it.x + it.w / 2} y={it.y + (it.l2 ? 15 : 22)} textAnchor="middle" fontSize={11} fill="var(--viz-ink)" style={{ fontWeight: 600 }}>
                    {it.l1}
                  </text>
                  {it.l2 ? (
                    <text x={it.x + it.w / 2} y={it.y + 29} textAnchor="middle" fontSize={10.5} fill="var(--viz-ink-2)">
                      {it.l2}
                    </text>
                  ) : null}
                </g>
              );
            })}
            {!run.root ? (
              <text x={16} y={32} fontSize={12} fill="var(--viz-ink-2)">
                Empty tree: drag “Keys inserted” to the right.
              </text>
            ) : null}
          </svg>
        </div>
        <div style={{ flex: '0 1 240px', minWidth: 0 }}>
          <svg viewBox={`0 0 240 ${memH}`} width={240} height={memH} role="img" aria-label={`Memory: adaptive radix tree ${fmtBytes(mem.total)}, B+tree ${fmtBytes(bt.total)}`} data-testid="art-memory">
            <text x={0} y={12} fontSize={11.5} fill="var(--viz-ink)" style={{ fontWeight: 600 }}>
              ART · {fmtNum(mem.total)} B · {perKey(mem.total)} B/key
            </text>
            {(() => {
              let x = 0;
              return artParts.map((p, i) => {
                if (p.v <= 0) return null;
                const w = Math.max(1.5, sc(p.v));
                const el = <rect key={i} x={x} y={18} width={w} height={16} fill={p.color} />;
                x += w;
                return el;
              });
            })()}
            <text x={0} y={56} fontSize={11.5} fill="var(--viz-ink)" style={{ fontWeight: 600 }}>
              B+tree · {fmtNum(bt.total)} B · {perKey(bt.total)} B/key
            </text>
            {(() => {
              let x = 0;
              return btParts.map((p, i) => {
                if (p.v <= 0) return null;
                const w = Math.max(1.5, sc(p.v));
                const el = <rect key={i} x={x} y={62} width={w} height={16} fill={p.color} />;
                x += w;
                return el;
              });
            })()}
            {memRows.map((p, i) => (
              <g key={`r${i}`}>
                <rect x={0} y={92 + i * 17} width={10} height={10} rx={2} fill={p.color} />
                <text x={16} y={101 + i * 17} fontSize={10.5} fill="var(--viz-ink-2)">
                  {p.who} {p.label}: {fmtNum(p.v)} B
                </text>
              </g>
            ))}
          </svg>
        </div>
      </div>
      {look ? (
        <ol style={{ margin: '0.5rem 0 0', paddingLeft: '1.2rem', fontSize: '0.8rem', display: 'grid', gap: 3 }} data-testid="art-steps">
          {mergedSteps.map((s, i) => (
            <li key={i} style={{ color: 'var(--viz-ink)' }}>
              {s.what} <span style={{ color: 'var(--viz-ink-2)' }}>({s.examined} key byte{s.examined === 1 ? '' : 's'})</span>
            </li>
          ))}
        </ol>
      ) : null}
      <p style={{ fontSize: '0.75rem', color: 'var(--viz-ink-2)', margin: '0.5rem 0 0' }} data-testid="art-order">
        In-order scan of the leaves: {order.slice(0, 20).join(' · ')}
        {order.length > 20 ? ` · … (${order.length})` : ''}
      </p>
      <Legend
        items={[
          { label: 'Node4', color: TYPE_COLOR[4] },
          { label: 'Node16', color: TYPE_COLOR[16] },
          { label: 'Node48', color: TYPE_COLOR[48] },
          { label: 'Node256', color: TYPE_COLOR[256] },
          { label: 'Leaf', color: LEAF_COLOR },
          { label: 'Prefix bytes past the 8-byte header (pessimistic)', color: EXTRA_PREFIX_COLOR },
          { label: 'B+tree nodes (16 keys each)', color: BT_NODE_COLOR },
          { label: 'B+tree key bytes stored out of line', color: BT_KEY_COLOR },
          { label: 'Lookup path', color: 'var(--viz-ink)', shape: 'line' },
          { label: 'Found', color: 'var(--viz-good)' },
          { label: 'Not found', color: 'var(--viz-critical)' },
        ]}
      />
      <Stats
        items={[
          { label: 'Inner nodes', value: `${c[4]} · ${c[16]} · ${c[48]} · ${c[256]}`, hint: 'Node4 · Node16 · Node48 · Node256' },
          { label: 'ART bytes / key', value: perKey(mem.total), hint: 'Node sizes from the ART paper’s Table I: 52, 160, 656 and 2064 bytes' },
          { label: 'B+tree bytes / key', value: perKey(bt.total), hint: 'Model: 16 keys per node, 8-byte key slots and pointers, long keys out of line' },
          { label: 'Height: ART / B+tree', value: `${c.height} / ${liveCount ? bt.height : 0}`, hint: 'ART: node hops from root to the deepest leaf. B+tree: levels.' },
          { label: 'Key bytes examined: ART / B+tree', value: look && btLook ? `${look.examined} / ${btLook.examined}` : '—', hint: 'Bytes of the search key read by the lookup below. B+tree: every comparison of its binary searches, each reading up to the first differing byte.' },
        ]}
      />
      <Note>
        {!run.root ? (
          <>
            <strong>The tree is empty.</strong> Drag <em>Keys inserted</em> to the right to add keys one at a time.
          </>
        ) : (
          <>
            <strong>
              {liveCount} key{liveCount === 1 ? '' : 's'}, {c[4] + c[16] + c[48] + c[256]} inner node{c[4] + c[16] + c[48] + c[256] === 1 ? '' : 's'}
              {collapse !== 'full' && c.oneWay ? `, ${c.oneWay} of them one-way` : ''}: {perKey(mem.total)} bytes per key against {perKey(bt.total)} for the B+tree.
            </strong>{' '}
            {growthNote}{' '}
            {probe && look ? (
              <>
                Looking up <code>{probeLabel}</code>: {where} after {look.examined} key byte{look.examined === 1 ? '' : 's'} and {look.path.length} node{look.path.length === 1 ? '' : 's'}
                {btLook ? `; the B+tree reads ${btLook.examined} bytes in ${btLook.comparisons} comparisons` : ''}.
                {look.skipped && !look.found && look.where === 'leaf' ? ' Skipped prefix bytes are only verified at the leaf, so the wrong turn costs a whole descent.' : ''}
              </>
            ) : probeKind === 'child' ? (
              'Every byte value already has a child here, so there is no byte to miss on.'
            ) : probeKind === 'prefix' ? (
              'This tree has no compressed prefix to miss inside.'
            ) : null}
          </>
        )}
      </Note>
      <details className="viz-data">
        <summary>Show the numbers</summary>
        <table className="viz-table">
          <thead>
            <tr>
              <th>Structure</th>
              <th>Part</th>
              <th>Count</th>
              <th>Bytes</th>
            </tr>
          </thead>
          <tbody>
            {([4, 16, 48, 256] as NodeType[]).map((t) => (
              <tr key={t}>
                <td>ART</td>
                <td>Node{t} ({NODE_BYTES[t]} B each)</td>
                <td>{c[t]}</td>
                <td>{fmtNum(mem.nodes[t])}</td>
              </tr>
            ))}
            <tr>
              <td>ART</td>
              <td>Prefix bytes stored ({prefixMode}); past the 8-byte header</td>
              <td>{c.prefixBytes}</td>
              <td>{fmtNum(mem.prefix)}</td>
            </tr>
            <tr>
              <td>ART</td>
              <td>Leaves ({leafMode === 'key' ? '8-byte value + key copy' : 'tuple id tagged into the child pointer'})</td>
              <td>{c.leaves}</td>
              <td>{fmtNum(mem.leafBytes)}</td>
            </tr>
            <tr>
              <td>B+tree</td>
              <td>Leaf nodes ({BT_LEAF_BYTES} B each)</td>
              <td>{bt.leaves}</td>
              <td>{fmtNum(bt.leaves * BT_LEAF_BYTES)}</td>
            </tr>
            <tr>
              <td>B+tree</td>
              <td>Inner nodes ({BT_INNER_BYTES} B each)</td>
              <td>{bt.inners}</td>
              <td>{fmtNum(bt.inners * BT_INNER_BYTES)}</td>
            </tr>
            <tr>
              <td>B+tree</td>
              <td>Key bytes for keys longer than 8 bytes</td>
              <td>—</td>
              <td>{fmtNum(bt.keyBytes)}</td>
            </tr>
          </tbody>
        </table>
      </details>
    </>
  );
}

/* --------------------------------------------------------------------------- SP-GiST panel */

const PLANE_PX = 360;

function SpgistPanel() {
  const [opclass, setOpclass] = useState<Opclass>('quad');
  const [dist, setDist] = useState<Distribution>('clustered');
  const [count, setCount] = useState(120);
  const [capacity, setCapacity] = useState(6);
  const [bx, setBx] = useState(70);
  const [by, setBy] = useState(25);
  const [bs, setBs] = useState(20);
  const [clicked, setClicked] = useState<Pt[]>([]);

  const pts = useMemo(() => [...makePoints(dist, count), ...clicked], [dist, count, clicked]);
  const tree = useMemo(() => buildSpgist(pts, opclass, capacity), [pts, opclass, capacity]);
  const box: Box = { lx: bx, ly: by, hx: Math.min(100, bx + bs), hy: Math.min(100, by + bs) };
  const q = useMemo(() => spgistQuery(tree.root, pts, box, opclass), [tree, pts, box.lx, box.ly, box.hx, box.hy, opclass]);
  const shape = useMemo(() => spgistShape(tree.root), [tree]);
  const testedSet = new Set(q.testedIdx);
  const matchSet = new Set(q.matches);
  const X = (v: number) => (v / 100) * PLANE_PX;
  const Y = (v: number) => PLANE_PX - (v / 100) * PLANE_PX;

  const lines: { x1: number; y1: number; x2: number; y2: number; level: number }[] = [];
  const cells: { b: Box; visited: boolean }[] = [];
  const centroids: Pt[] = [];
  const collect = (n: QNode | null) => {
    if (!n) return;
    if (n.kind === 'leaf') {
      if (n.pts.length) cells.push({ b: n.region, visited: q.visited.has(n) });
      return;
    }
    const r = n.region;
    if (opclass === 'quad' && n.centroid && !n.allTheSame) {
      lines.push({ x1: n.centroid.x, y1: r.ly, x2: n.centroid.x, y2: r.hy, level: n.level });
      lines.push({ x1: r.lx, y1: n.centroid.y, x2: r.hx, y2: n.centroid.y, level: n.level });
      centroids.push(n.centroid);
    } else if (opclass === 'kd' && n.coord !== undefined) {
      if (n.axis === 'x') lines.push({ x1: n.coord, y1: r.ly, x2: n.coord, y2: r.hy, level: n.level });
      else lines.push({ x1: r.lx, y1: n.coord, x2: r.hx, y2: n.coord, level: n.level });
    }
    n.children.forEach(collect);
  };
  collect(tree.root);

  // icicle: width proportional to points under a node
  const ICW = 360;
  const IROW = 22;
  const ice: { x: number; w: number; d: number; kind: 'inner' | 'leaf'; visited: boolean; n: number }[] = [];
  const size = (n: QNode | null): number => (!n ? 0 : n.kind === 'leaf' ? n.pts.length : n.children.reduce((s, c) => s + size(c), 0));
  const totalPts = Math.max(1, size(tree.root));
  const layIce = (n: QNode | null, x: number, d: number) => {
    if (!n) return;
    const s = size(n);
    if (!s) return;
    const w = (s / totalPts) * ICW;
    ice.push({ x, w, d, kind: n.kind, visited: q.visited.has(n), n: s });
    if (n.kind === 'inner') {
      let cx = x;
      n.children.forEach((c) => {
        layIce(c, cx, d + 1);
        cx += (size(c) / totalPts) * ICW;
      });
    }
  };
  layIce(tree.root, 0, 0);
  const maxDepth = ice.reduce((m, e) => Math.max(m, e.d), 0);
  const iceH = (maxDepth + 1) * IROW + 26;

  const byDepth = Array.from({ length: maxDepth + 1 }, (_, d) => {
    const row = ice.filter((e) => e.d === d);
    return {
      d,
      inner: row.filter((e) => e.kind === 'inner').length,
      leaf: row.filter((e) => e.kind === 'leaf').length,
      pts: row.filter((e) => e.kind === 'leaf').reduce((s, e) => s + e.n, 0),
      innerVisited: row.filter((e) => e.kind === 'inner' && e.visited).length,
      leafVisited: row.filter((e) => e.kind === 'leaf' && e.visited).length,
    };
  });

  const onPlaneClick = (e: React.MouseEvent<SVGSVGElement>) => {
    const r = e.currentTarget.getBoundingClientRect();
    // viewBox is (-2, -2, PLANE_PX + 4, PLANE_PX + 4): convert to viewBox units, then to plane coordinates.
    const vx = ((e.clientX - r.left) / r.width) * (PLANE_PX + 4) - 2;
    const vy = ((e.clientY - r.top) / r.height) * (PLANE_PX + 4) - 2;
    const x = Math.round((vx / PLANE_PX) * 1000) / 10;
    const y = Math.round((100 - (vy / PLANE_PX) * 100) * 10) / 10;
    if (x < 0 || x > 100 || y < 0 || y > 100 || clicked.length >= 80) return;
    setClicked([...clicked, { x, y }]);
  };

  return (
    <>
      <div className="viz-controls">
        <Segmented
          label="Operator class"
          value={opclass}
          onChange={setOpclass}
          options={[
            { value: 'quad', label: 'quad_point_ops' },
            { value: 'kd', label: 'kd_point_ops' },
          ]}
        />
        <Choice
          label="Points"
          value={dist}
          onChange={(v) => setDist(v)}
          options={[
            { value: 'clustered', label: 'Two clusters' },
            { value: 'uniform', label: 'Uniform' },
            { value: 'diagonal', label: 'Along a diagonal' },
          ]}
        />
        <Slider label="Points inserted" min={0} max={200} value={count} onChange={setCount} />
        <Slider label="Leaf capacity" min={2} max={16} value={capacity} onChange={setCapacity} />
        <Button onClick={() => setClicked([])} disabled={!clicked.length}>
          Remove my {clicked.length || ''} points
        </Button>
      </div>
      <div className="viz-controls">
        <Slider label="Box left x" min={0} max={95} value={bx} onChange={setBx} />
        <Slider label="Box bottom y" min={0} max={95} value={by} onChange={setBy} />
        <Slider label="Box size" min={2} max={100} value={bs} onChange={setBs} />
      </div>
      <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.75rem', alignItems: 'flex-start' }}>
        <svg
          viewBox={`-2 -2 ${PLANE_PX + 4} ${PLANE_PX + 4}`}
          width={PLANE_PX + 4}
          height={PLANE_PX + 4}
          role="img"
          aria-label={`${opclass === 'quad' ? 'Quadtree' : 'k-d tree'} partitions over ${pts.length} points; the box query visits ${q.innerVisited} inner tuples and tests ${q.tested} points`}
          onClick={onPlaneClick}
          style={{ cursor: 'crosshair', flex: '0 1 364px' }}
          data-testid="spgist-plane"
        >
          <rect x={0} y={0} width={PLANE_PX} height={PLANE_PX} fill="var(--viz-plane)" stroke="var(--viz-border)" />
          {cells.map((cl, i) =>
            cl.visited ? <rect key={`v${i}`} x={X(cl.b.lx)} y={Y(cl.b.hy)} width={X(cl.b.hx) - X(cl.b.lx)} height={Y(cl.b.ly) - Y(cl.b.hy)} fill="var(--viz-3)" fillOpacity={0.22} /> : null,
          )}
          {lines.map((l, i) => (
            <line key={`s${i}`} x1={X(l.x1)} y1={Y(l.y1)} x2={X(l.x2)} y2={Y(l.y2)} stroke="var(--viz-ink-2)" strokeOpacity={Math.max(0.35, 0.9 - l.level * 0.12)} strokeWidth={Math.max(0.6, 1.6 - l.level * 0.2)} />
          ))}
          {centroids.map((p, i) => (
            <path key={`c${i}`} d={`M${X(p.x) - 3} ${Y(p.y)} L${X(p.x)} ${Y(p.y) - 3} L${X(p.x) + 3} ${Y(p.y)} L${X(p.x)} ${Y(p.y) + 3} Z`} fill="var(--viz-7)" />
          ))}
          {pts.map((p, i) =>
            matchSet.has(i) ? (
              <circle key={i} cx={X(p.x)} cy={Y(p.y)} r={3} fill="var(--viz-good)" />
            ) : testedSet.has(i) ? (
              <circle key={i} cx={X(p.x)} cy={Y(p.y)} r={2.6} fill="var(--viz-1)" />
            ) : (
              <circle key={i} cx={X(p.x)} cy={Y(p.y)} r={2.2} fill="var(--viz-ink-muted)" />
            ),
          )}
          <rect x={X(box.lx)} y={Y(box.hy)} width={X(box.hx) - X(box.lx)} height={Y(box.ly) - Y(box.hy)} fill="none" stroke="var(--viz-2)" strokeWidth={2} strokeDasharray="6 3" />
        </svg>
        <svg viewBox={`0 0 ${ICW + 30} ${iceH}`} width={ICW + 30} height={iceH} role="img" aria-label={`Tree shape: leaf depths from ${shape.minDepth} to ${shape.maxDepth}`} style={{ flex: '0 1 390px' }} data-testid="spgist-icicle">
          <text x={0} y={12} fontSize={11} fill="var(--viz-ink)" style={{ fontWeight: 600 }}>
            Tree by depth (width ∝ points underneath)
          </text>
          {ice.map((e, i) => (
            <rect
              key={i}
              x={30 + e.x + 0.5}
              y={20 + e.d * IROW}
              width={Math.max(0.8, e.w - 1)}
              height={IROW - 4}
              rx={2}
              fill={e.visited ? 'var(--viz-3)' : e.kind === 'leaf' ? 'var(--viz-plane)' : 'var(--viz-surface)'}
              fillOpacity={e.visited ? 0.45 : 1}
              stroke={e.kind === 'leaf' ? 'var(--viz-ink-muted)' : 'var(--viz-ink-2)'}
              strokeWidth={0.6}
            />
          ))}
          {byDepth.map((r) => (
            <text key={r.d} x={0} y={20 + r.d * IROW + 13} fontSize={10} fill="var(--viz-ink-2)">
              {r.d}
            </text>
          ))}
        </svg>
      </div>
      <ol style={{ margin: '0.5rem 0 0', paddingLeft: '1.2rem', fontSize: '0.78rem', display: 'grid', gap: 2 }} data-testid="spgist-trace">
        {tree.lastTrace.map((t, i) => (
          <li key={`t${i}`} style={{ color: 'var(--viz-ink)' }}>
            {i === 0 ? <span style={{ color: 'var(--viz-ink-2)' }}>last insert · </span> : null}
            {t}
          </li>
        ))}
        {q.trace.slice(0, 5).map((t, i) => (
          <li key={`q${i}`} style={{ color: 'var(--viz-ink)' }}>
            {i === 0 ? <span style={{ color: 'var(--viz-ink-2)' }}>box query · </span> : null}
            {t}
          </li>
        ))}
        {q.trace.length > 5 ? <li style={{ color: 'var(--viz-ink-2)' }}>… {q.trace.length - 5} more inner_consistent calls</li> : null}
      </ol>
      <Legend
        items={[
          { label: 'Point (leaf tuple)', color: 'var(--viz-ink-muted)', shape: 'dot' },
          { label: 'Tested by leaf_consistent', color: 'var(--viz-1)', shape: 'dot' },
          { label: 'Inside the box (match)', color: 'var(--viz-good)', shape: 'dot' },
          { label: 'Query box', color: 'var(--viz-2)', shape: 'line' },
          { label: 'Cell / node the query visited', color: 'var(--viz-3)' },
          ...(opclass === 'quad' ? [{ label: 'Centroid (inner tuple prefix)', color: 'var(--viz-7)' }] : []),
        ]}
      />
      <Stats
        items={[
          { label: 'Inner tuples', value: fmtNum(shape.inner) },
          { label: 'Leaf depth, min–max', value: `${shape.minDepth}–${shape.maxDepth}`, hint: 'SP-GiST never rebalances: depth follows the data and the insertion order' },
          { label: 'Inner tuples visited', value: `${q.innerVisited} of ${shape.inner}` },
          { label: 'Points tested', value: `${q.tested} of ${pts.length}`, hint: 'Leaf tuples passed to leaf_consistent' },
          { label: 'Matches', value: fmtNum(q.matches.length) },
        ]}
      />
      <Note>
        {pts.length === 0 ? (
          <>
            <strong>No points yet.</strong> Drag <em>Points inserted</em> or click the plane to add points.
          </>
        ) : shape.inner === 0 ? (
          <>
            <strong>All {pts.length} points still fit in the root’s leaf list</strong> (capacity {capacity}), so there is nothing to prune: the query tests every point. Add points until picksplit runs.
          </>
        ) : (
          <>
            <strong>
              The box query visited {q.innerVisited} of {shape.inner} inner tuples and tested {q.tested} of {pts.length} points to find {q.matches.length}.
            </strong>{' '}
            {opclass === 'quad'
              ? 'At each inner tuple, inner_consistent follows every quadrant when the centroid lies inside the box and otherwise only the quadrants holding the box’s corners; the other quadrants’ subtrees are never read.'
              : 'At each inner tuple, inner_consistent follows one side of the median split unless the box straddles it.'}{' '}
            Leaves sit between depth {shape.minDepth} and {shape.maxDepth}: {shape.maxDepth - shape.minDepth >= 3 ? 'dense regions split again and again while sparse ones stay shallow — nothing rebalances the tree.' : 'the tree is fairly even for this data.'}
          </>
        )}
      </Note>
      <details className="viz-data">
        <summary>Show the numbers</summary>
        <table className="viz-table">
          <thead>
            <tr>
              <th>Depth</th>
              <th>Inner tuples</th>
              <th>Leaf lists</th>
              <th>Points in those lists</th>
              <th>Inner tuples visited</th>
              <th>Leaf lists scanned</th>
            </tr>
          </thead>
          <tbody>
            {byDepth.map((r) => (
              <tr key={r.d}>
                <td>{r.d}</td>
                <td>{r.inner}</td>
                <td>{r.leaf}</td>
                <td>{r.pts}</td>
                <td>{r.innerVisited}</td>
                <td>{r.leafVisited}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </details>
    </>
  );
}

export default function AdaptiveRadixSpGistLab() {
  const [tab, setTab] = useState<'art' | 'spgist'>('art');
  return (
    <VizPanel
      title={tab === 'art' ? 'An adaptive radix tree, byte by byte' : 'SP-GiST: partitioning space, not keys'}
      subtitle={
        tab === 'art'
          ? 'Insert keys and watch inner nodes grow from Node4 to Node16, Node48 and Node256, one-way chains collapse into prefixes, and memory compare with a B+tree over the same keys. Pick a lookup to trace its path.'
          : 'PostgreSQL’s quad_point_ops and kd_point_ops, run for real: an overflowing leaf list calls picksplit, inserts call choose, and a box query calls inner_consistent to skip every cell that cannot touch the box. Click the plane to add points.'
      }
      controls={
        <Segmented
          label="Structure"
          value={tab}
          onChange={setTab}
          options={[
            { value: 'art', label: 'Adaptive radix tree' },
            { value: 'spgist', label: 'SP-GiST quadtree / k-d tree' },
          ]}
        />
      }
    >
      {tab === 'art' ? <ArtPanel /> : <SpgistPanel />}
    </VizPanel>
  );
}
