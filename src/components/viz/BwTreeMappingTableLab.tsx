import { useMemo, useState } from 'react';
import { VizPanel, Slider, Choice, Check, Button, Legend, Stats, Note } from './Viz';

/**
 * A Bw-tree in memory, following Levandoski, Lomet and Sengupta (ICDE 2013) and the CMU OpenBw-Tree:
 * - A mapping table maps each logical page ID (PID) to the physical address of the page's newest state.
 *   Every link between pages (downlinks, side links) is a PID; only mapping slots and `next` pointers are
 *   physical addresses.
 * - Pages are never modified. A write builds a delta record whose `next` is the address it read from the
 *   slot, then compare-and-swaps the slot from that address to the delta. A thread whose CAS fails
 *   discards its delta and restarts from the root (OpenBw-Tree's retry rule).
 * - A thread whose write leaves a chain at or past the threshold consolidates after that write (in the lab reads
 *   never consolidate; the ICDE design lets a reader do it too, which the Longest-chain hint says): it builds a new
 *   base page privately and CASes it in. If that CAS fails it gives up and does not retry. The old chain is
 *   retired, not freed: epochs (a later page) decide when.
 * - Splits are B-link half splits in two CAS phases: build the new right page Q and store it into its
 *   fresh mapping slot without a CAS, CAS a split delta onto P, then CAS an index-entry delta onto the
 *   parent. A thread that has to follow the split delta's side link, or that starts its own split or
 *   consolidation of either page, first completes the unfinished split (help-along).
 *
 * Model simplifications, labelled in the UI: one root with up to six leaves (the root never splits), one
 * threshold for leaf and root chains, integer keys 1–99, a separator is a child's inclusive upper bound,
 * split key = median live key, merges are not simulated. "Pointer chases" counts mapping-slot loads plus
 * nodes dereferenced on a read; it is a count of memory loads, not a measurement of cache misses.
 */

export type Pid = number;
export type Addr = number;
export const ROOT: Pid = 1;
export const KEY_MAX = 99;
export const MAX_LEAVES = 6;

export type LeafBase = { kind: 'leaf'; entries: [number, number][]; high: number | null; right: Pid | null };
export type InnerBase = { kind: 'inner'; seps: { high: number | null; pid: Pid }[] };
export type RecDelta = { kind: 'insert' | 'modify' | 'delete'; key: number; val: number; next: Addr };
export type SplitDelta = { kind: 'split'; sep: number; right: Pid; next: Addr };
export type IndexDelta = { kind: 'index'; lo: number; hi: number | null; pid: Pid; next: Addr };
export type MemNode = LeafBase | InnerBase | RecDelta | SplitDelta | IndexDelta;

export type BwState = {
  mem: Record<number, MemNode>;
  map: Record<number, Addr | null>;
  nextAddr: number;
  nextPid: number;
  retired: Addr[];
  casOk: number;
  casFail: number;
  consolidations: number;
  failedConsolidations: number;
  splits: number;
  failedSplits: number;
  wastedHops: number;
};

export type Race = 'none' | 'second' | 'first';
export type WriteKind = 'insert' | 'update' | 'delete';
export type Actor = 'A' | 'B';
export type ThreadView = { actor: Actor; pid: Pid; text: string; status: 'work' | 'ok' | 'fail' };
export type Frame = {
  state: BwState;
  text: string;
  actor: Actor | null;
  tone: 'plain' | 'ok' | 'fail';
  threads: ThreadView[];
  cas?: { pid: Pid; ok: boolean };
};

const isDelta = (n: MemNode): n is RecDelta | SplitDelta | IndexDelta => 'next' in n;

export function initialState(): BwState {
  const mem: Record<number, MemNode> = {
    1: { kind: 'inner', seps: [{ high: 49, pid: 2 }, { high: null, pid: 3 }] },
    2: { kind: 'leaf', entries: [[10, 1], [20, 1], [30, 1], [40, 1]], high: 49, right: 3 },
    3: { kind: 'leaf', entries: [[50, 1], [60, 1], [70, 1], [80, 1], [90, 1]], high: null, right: null },
    4: { kind: 'insert', key: 25, val: 1, next: 2 },
    5: { kind: 'modify', key: 10, val: 2, next: 4 },
    6: { kind: 'delete', key: 30, val: 0, next: 5 },
  };
  return {
    mem,
    map: { 1: 1, 2: 6, 3: 3 },
    nextAddr: 7,
    nextPid: 4,
    retired: [],
    casOk: 0,
    casFail: 0,
    consolidations: 0,
    failedConsolidations: 0,
    splits: 0,
    failedSplits: 0,
    wastedHops: 0,
  };
}

const clone = (s: BwState): BwState => ({ ...s, mem: { ...s.mem }, map: { ...s.map }, retired: [...s.retired] });

/** Physical addresses from the mapping slot to the base page, newest first. */
export function chainAddrs(s: BwState, pid: Pid): Addr[] {
  const out: Addr[] = [];
  let a = s.map[pid];
  while (a != null && out.length < 64) {
    out.push(a);
    const n = s.mem[a];
    a = isDelta(n) ? n.next : null;
  }
  return out;
}

export const chainLength = (s: BwState, pid: Pid) => Math.max(0, chainAddrs(s, pid).length - 1);

export const livePids = (s: BwState) =>
  Object.keys(s.map)
    .map(Number)
    .filter((p) => s.map[p] != null)
    .sort((a, b) => a - b);

/** The logical contents of a leaf: base page with its deltas applied, oldest first. */
export function leafView(s: BwState, pid: Pid) {
  const addrs = chainAddrs(s, pid);
  const base = s.mem[addrs[addrs.length - 1]] as LeafBase;
  const entries = new Map<number, number>(base.entries);
  let high = base.high;
  let right = base.right;
  let split: { sep: number; right: Pid; addr: Addr } | null = null;
  for (let i = addrs.length - 2; i >= 0; i--) {
    const n = s.mem[addrs[i]];
    if (n.kind === 'insert' || n.kind === 'modify') entries.set(n.key, n.val);
    else if (n.kind === 'delete') entries.delete(n.key);
    else if (n.kind === 'split') {
      for (const k of [...entries.keys()]) if (k > n.sep) entries.delete(k);
      high = n.sep;
      right = n.right;
      split = { sep: n.sep, right: n.right, addr: addrs[i] };
    }
  }
  const sorted = [...entries.entries()].sort((a, b) => a[0] - b[0]);
  return { entries: sorted, high, right, split, chain: addrs.length - 1 };
}

/** The root's routing after its index-entry deltas are applied, oldest first. */
export function innerView(s: BwState) {
  const addrs = chainAddrs(s, ROOT);
  const base = s.mem[addrs[addrs.length - 1]] as InnerBase;
  const seps = base.seps.map((e) => ({ ...e }));
  for (let i = addrs.length - 2; i >= 0; i--) {
    const n = s.mem[addrs[i]];
    if (n.kind !== 'index') continue;
    const at = seps.findIndex((e) => e.high === n.hi);
    if (at >= 0) seps.splice(at, 1, { high: n.lo, pid: seps[at].pid }, { high: n.hi, pid: n.pid });
  }
  return { seps, chain: addrs.length - 1 };
}

export type Hop = { t: 'map'; pid: Pid } | { t: 'node'; pid: Pid; addr: Addr } | { t: 'side'; from: Pid; to: Pid };

/** A point read, exactly as a thread walks it: slot, chain newest-first, base page, side links. */
export function lookup(s: BwState, key: number) {
  const hops: Hop[] = [];
  let pid: Pid = ROOT;
  let guard = 0;
  outer: while (guard++ < 40) {
    hops.push({ t: 'map', pid });
    let addr = s.map[pid];
    while (addr != null && guard++ < 400) {
      const n = s.mem[addr];
      hops.push({ t: 'node', pid, addr });
      switch (n.kind) {
        case 'index':
          if (key > n.lo && (n.hi === null || key <= n.hi)) {
            pid = n.pid;
            continue outer;
          }
          addr = n.next;
          break;
        case 'inner': {
          const e = n.seps.find((x) => x.high === null || key <= x.high)!;
          pid = e.pid;
          continue outer;
        }
        case 'split':
          if (key > n.sep) {
            hops.push({ t: 'side', from: pid, to: n.right });
            pid = n.right;
            continue outer;
          }
          addr = n.next;
          break;
        case 'insert':
        case 'modify':
          if (n.key === key) return result(hops, pid, true, n.val);
          addr = n.next;
          break;
        case 'delete':
          if (n.key === key) return result(hops, pid, false, 0);
          addr = n.next;
          break;
        case 'leaf': {
          if (n.high !== null && key > n.high && n.right != null) {
            hops.push({ t: 'side', from: pid, to: n.right });
            pid = n.right;
            continue outer;
          }
          const e = n.entries.find((x) => x[0] === key);
          return result(hops, pid, !!e, e ? e[1] : 0);
        }
      }
    }
    break;
  }
  return result(hops, pid, false, 0);
}

function result(hops: Hop[], leaf: Pid, found: boolean, val: number) {
  const nodes = hops.filter((h) => h.t === 'node').length;
  const maps = hops.filter((h) => h.t === 'map').length;
  const sides = hops.filter((h) => h.t === 'side').length;
  return { hops, leaf, found, val, chases: nodes + maps, maps, nodes, sides };
}

export function deltasOnPath(s: BwState, key: number) {
  return lookup(s, key).hops.filter((h) => h.t === 'node' && isDelta(s.mem[h.addr])).length;
}

/** Leaves whose split delta has no entry in the root yet. */
export function pendingSplits(s: BwState) {
  const routed = new Set(innerView(s).seps.map((e) => e.pid));
  const out: { pid: Pid; sep: number; right: Pid }[] = [];
  for (const pid of livePids(s)) {
    if (pid === ROOT) continue;
    const v = leafView(s, pid);
    if (v.split && !routed.has(v.split.right)) out.push({ pid, sep: v.split.sep, right: v.split.right });
  }
  return out;
}

export const leafCount = (s: BwState) => livePids(s).length - 1;

/** A leaf's key range (lo, hi], as the tree currently routes it. */
export function leafRange(s: BwState, pid: Pid): [number, number] {
  const seps = innerView(s).seps;
  const i = seps.findIndex((e) => e.pid === pid);
  let lo = 0;
  let hi = KEY_MAX;
  if (i >= 0) {
    lo = i > 0 ? (seps[i - 1].high as number) : 0;
    hi = seps[i].high ?? KEY_MAX;
  } else {
    const p = pendingSplits(s).find((x) => x.right === pid);
    if (p) {
      lo = p.sep;
      hi = seps.find((e) => e.pid === p.pid)?.high ?? KEY_MAX;
    }
  }
  const v = leafView(s, pid);
  if (v.split) hi = Math.min(hi, v.split.sep);
  return [lo, hi];
}

/* ------------------------------------------------------------ actions */

class Run {
  w: BwState;
  frames: Frame[] = [];
  threads: Partial<Record<Actor, ThreadView>> = {};
  constructor(s: BwState) {
    this.w = clone(s);
  }
  push(text: string, actor: Actor | null, tone: Frame['tone'] = 'plain', cas?: Frame['cas']) {
    const threads = (['A', 'B'] as Actor[]).map((a) => this.threads[a]).filter(Boolean) as ThreadView[];
    this.frames.push({ state: clone(this.w), text, actor, tone, threads: threads.map((t) => ({ ...t })), cas });
  }
  alloc(n: MemNode) {
    const a = this.w.nextAddr++;
    this.w.mem[a] = n;
    return a;
  }
  free(a: Addr) {
    delete this.w.mem[a];
  }
  cas(pid: Pid, expect: Addr, next: Addr) {
    if (this.w.map[pid] === expect) {
      this.w.map[pid] = next;
      this.w.casOk++;
      return true;
    }
    this.w.casFail++;
    return false;
  }
}

const at = (a: Addr | null | undefined) => (a == null ? '∅' : `@${a}`);

export function deltaLabel(n: MemNode) {
  switch (n.kind) {
    case 'insert':
      return `ins ${n.key}`;
    case 'modify':
      return `mod ${n.key}`;
    case 'delete':
      return `del ${n.key}`;
    case 'split':
      return `split >${n.sep} → P${n.right}`;
    case 'index':
      return `index (${n.lo},${n.hi ?? '∞'}] → P${n.pid}`;
    default:
      return n.kind === 'leaf' ? 'base page' : 'root base';
  }
}

type BWrite = { kind: 'insert' | 'modify'; key: number } | null;

/** The racing writer's operation: a free key in (lo, hi] nearest `prefer`, else an update of an existing key. */
function pickB(s: BwState, pid: Pid, lo: number, hi: number, prefer: number, avoid: number): BWrite {
  const present = new Set(leafView(s, pid).entries.map((e) => e[0]));
  for (let d = 0; d <= KEY_MAX; d++) {
    for (const k of [prefer + d, prefer - d]) {
      if (k > lo && k <= hi && k >= 1 && k <= KEY_MAX && k !== avoid && !present.has(k)) return { kind: 'insert', key: k };
    }
  }
  const existing = [...present].filter((k) => k !== avoid && k > lo && k <= hi);
  return existing.length ? { kind: 'modify', key: existing[0] } : null;
}

function buildRecord(w: BwState, kind: 'insert' | 'modify' | 'delete', key: number, next: Addr): RecDelta {
  const cur = lookup(w, key);
  return { kind, key, val: kind === 'insert' ? 1 : kind === 'modify' ? cur.val + 1 : 0, next };
}

/** Post the root's index entry for the unfinished split of `pid`. Returns whether the root now routes to Q. */
function postIndexSteps(r: Run, pid: Pid, actor: Actor, race: Race, helping: false | 'side' | 'page', T: number) {
  const w = r.w;
  const leaf = leafView(w, pid);
  if (!leaf.split) return false;
  const q = leaf.split.right;
  const sep = leaf.split.sep;
  const hi = innerView(w).seps.find((e) => e.pid === pid)?.high ?? null;
  const other: Actor = actor === 'A' ? 'B' : 'A';
  const label = `index (${sep},${hi ?? '∞'}] → P${q}`;
  const expect = w.map[ROOT] as Addr;
  const mine = r.alloc({ kind: 'index', lo: sep, hi, pid: q, next: expect });
  r.threads[actor] = { actor, pid: ROOT, status: 'work', text: `${actor} expects ${at(expect)}, holds private ${label}` };
  r.push(
    helping === 'side'
      ? `${actor} must cross P${pid}'s side link, but the root has no entry for P${q}: the split is unfinished. Before its own work, ${actor} completes it — loads P1's slot (${at(expect)}) and builds ${label}.`
      : helping === 'page'
      ? `P${pid}'s split to P${q} is unfinished: the root has no entry for P${q}. Before its own work on the page, ${actor} completes it — loads P1's slot (${at(expect)}) and builds ${label}.`
      : `Phase 2. ${actor} loads the root's slot (${at(expect)}) and builds an index-entry delta, ${label}: keys above ${sep} can go straight to P${q}.`,
    actor,
  );
  let bKey: BWrite = null;
  if (race !== 'none') {
    const qhi = hi ?? KEY_MAX;
    bKey = pickB(w, q, sep, qhi, Math.min(qhi, sep + Math.ceil((qhi - sep) / 2)), -1);
    const theirs = r.alloc({ kind: 'index', lo: sep, hi, pid: q, next: expect });
    r.threads[other] = { actor: other, pid: ROOT, status: 'work', text: `${other} expects ${at(expect)}, holds the same index entry` };
    r.push(
      `${other}, on its way to write key ${bKey?.key ?? '—'} in P${q}'s range, also hit the side link and is helping finish the same split: it loaded ${at(expect)} and built its own copy of the entry (@${theirs}).`,
      other,
    );
    const [first, firstAddr, second, secondAddr] = race === 'second' ? [actor, mine, other, theirs] : [other, theirs, actor, mine];
    r.cas(ROOT, expect, firstAddr);
    r.threads[first] = { actor: first, pid: ROOT, status: 'ok', text: `${first}: CAS P1 ${at(expect)} → @${firstAddr} ✓` };
    r.push(`${first}'s CAS on P1 succeeds: the slot still held ${at(expect)}, so @${firstAddr} is now the root's newest state.`, first, 'ok', { pid: ROOT, ok: true });
    r.cas(ROOT, expect, secondAddr);
    r.free(secondAddr);
    r.threads[second] = { actor: second, pid: ROOT, status: 'fail', text: `${second}: CAS P1 expected ${at(expect)}, found @${firstAddr} ✗` };
    r.push(
      `${second}'s CAS fails — the slot holds @${firstAddr}, not ${at(expect)}. ${second} discards its copy, re-reads the root, and finds P${q} already routed: nothing left to do for the split.`,
      second,
      'fail',
      { pid: ROOT, ok: false },
    );
    delete r.threads[second];
    delete r.threads[first];
    if (bKey) writeAt(r, 'B', bKey.kind, bKey.key, 'none', T, false);
  } else {
    r.cas(ROOT, expect, mine);
    r.threads[actor] = { actor, pid: ROOT, status: 'ok', text: `${actor}: CAS P1 ${at(expect)} → @${mine} ✓` };
    r.push(`${actor}'s CAS on the root succeeds. The split is complete: a search for a key above ${sep} now goes from P1 directly to P${q}.`, actor, 'ok', { pid: ROOT, ok: true });
    delete r.threads[actor];
  }
  maybeConsolidate(r, ROOT, T, actor);
  return true;
}

/** Complete any unfinished split that `key`'s route crosses, or that involves `pid`. */
function helpAlong(r: Run, key: number | null, pid: Pid | null, actor: Actor, T: number) {
  for (let guard = 0; guard < 4; guard++) {
    const pend = pendingSplits(r.w);
    let target: Pid | null = null;
    let why: 'side' | 'page' = 'side';
    if (key != null) {
      for (const h of lookup(r.w, key).hops) if (h.t === 'side' && pend.some((p) => p.pid === h.from)) target = h.from;
    }
    if (target == null && pid != null) {
      const p = pend.find((x) => x.pid === pid || x.right === pid);
      if (p) {
        target = p.pid;
        why = 'page';
      }
    }
    if (target == null) return;
    postIndexSteps(r, target, actor, 'none', why, T);
  }
}

function consolidateSteps(r: Run, pid: Pid, actor: Actor, race: Race, T: number, reason: string) {
  const w = r.w;
  if (pid !== ROOT) helpAlong(r, null, pid, actor, T);
  const len = chainLength(w, pid);
  if (len === 0) {
    r.push(`P${pid} is already a bare base page: there is nothing to consolidate.`, actor);
    return;
  }
  const expect = w.map[pid] as Addr;
  const old = chainAddrs(w, pid);
  let base: MemNode;
  let summary: string;
  if (pid === ROOT) {
    const v = innerView(w);
    base = { kind: 'inner', seps: v.seps };
    summary = `${v.seps.length} separators`;
  } else {
    const v = leafView(w, pid);
    base = { kind: 'leaf', entries: v.entries, high: v.high, right: v.right };
    summary = `${v.entries.length} live keys, high key ${v.high ?? '∞'}`;
  }
  const mine = r.alloc(base);
  const other: Actor = actor === 'A' ? 'B' : 'A';
  r.threads[actor] = { actor, pid, status: 'work', text: `${actor} expects ${at(expect)}, holds private base @${mine}` };
  r.push(`${reason}${actor} loads P${pid}'s slot (${at(expect)}), replays its ${len} delta${len === 1 ? '' : 's'} and writes a new base page @${mine} privately (${summary}).`, actor);

  let b: BWrite = null;
  let bAddr: Addr | null = null;
  if (race !== 'none' && pid !== ROOT) {
    const [lo, hi] = leafRange(w, pid);
    b = pickB(w, pid, lo, hi, Math.floor((lo + hi) / 2) + 1, -1);
    if (b) {
      bAddr = r.alloc(buildRecord(w, b.kind, b.key, expect));
      r.threads[other] = { actor: other, pid, status: 'work', text: `${other} expects ${at(expect)}, holds private ${b.kind === 'insert' ? 'ins' : 'mod'} ${b.key}` };
      r.push(`Meanwhile ${other} loads the same slot (${at(expect)}) to write key ${b.key}.`, other);
    }
  }
  if (race === 'first' && b && bAddr != null) {
    r.cas(pid, expect, bAddr);
    r.threads[other] = { actor: other, pid, status: 'ok', text: `${other}: CAS P${pid} ${at(expect)} → @${bAddr} ✓` };
    r.push(`${other}'s small delta wins the race: CAS ${at(expect)} → @${bAddr} succeeds.`, other, 'ok', { pid, ok: true });
    r.cas(pid, expect, mine);
    r.free(mine);
    w.failedConsolidations++;
    r.threads[actor] = { actor, pid, status: 'fail', text: `${actor}: CAS P${pid} expected ${at(expect)}, found @${bAddr} ✗` };
    r.push(
      `${actor}'s consolidation CAS fails. It frees @${mine} and does not retry — the chain is one delta longer, and a later thread that finds it over the threshold will consolidate.`,
      actor,
      'fail',
      { pid, ok: false },
    );
    delete r.threads[actor];
    delete r.threads[other];
    return;
  }
  r.cas(pid, expect, mine);
  w.consolidations++;
  w.retired.push(...old);
  r.threads[actor] = { actor, pid, status: 'ok', text: `${actor}: CAS P${pid} ${at(expect)} → @${mine} ✓` };
  r.push(
    `${actor}'s CAS installs @${mine}: same page ID, new address. The old chain (${old.map((a) => `@${a}`).join(', ')}) is retired, not freed — another thread may still be walking it.`,
    actor,
    'ok',
    { pid, ok: true },
  );
  delete r.threads[actor];
  if (b && bAddr != null) {
    r.cas(pid, expect, bAddr);
    r.free(bAddr);
    w.wastedHops += lookup(w, b.key).chases;
    r.threads[other] = { actor: other, pid, status: 'fail', text: `${other}: CAS P${pid} expected ${at(expect)}, found @${mine} ✗` };
    r.push(`${other}'s CAS fails: the slot now holds the consolidated page @${mine}. ${other} discards its delta and restarts from the root.`, other, 'fail', { pid, ok: false });
    delete r.threads[other];
    writeAt(r, other, b.kind, b.key, 'none', T, true);
  }
}

/** One record write by `actor`, optionally racing the other thread, then auto-consolidation. */
function writeAt(r: Run, actor: Actor, kind: 'insert' | 'modify' | 'delete', key: number, race: Race, T: number, retry: boolean) {
  const w = r.w;
  helpAlong(r, key, null, actor, T);
  const route = lookup(w, key);
  const pid = route.leaf;
  const expect = w.map[pid] as Addr;
  const mine = r.alloc(buildRecord(w, kind, key, expect));
  const other: Actor = actor === 'A' ? 'B' : 'A';
  const label = deltaLabel(w.mem[mine]);
  r.threads[actor] = { actor, pid, status: 'work', text: `${actor} expects ${at(expect)}, holds private ${label}` };
  r.push(
    `${actor} ${retry ? 'restarts from the root and ' : ''}descends to P${pid} (${route.chases} pointer chases), loads its slot (${at(expect)}) and builds ${label} with next = ${at(expect)}. No other thread can see it yet.`,
    actor,
  );
  if (race === 'none') {
    r.cas(pid, expect, mine);
    r.threads[actor] = { actor, pid, status: 'ok', text: `${actor}: CAS P${pid} ${at(expect)} → @${mine} ✓` };
    r.push(`${actor}'s CAS on P${pid}'s slot finds ${at(expect)} still there and stores @${mine}. One atomic store published the ${kind === 'modify' ? 'update' : kind}.`, actor, 'ok', { pid, ok: true });
    delete r.threads[actor];
  } else {
    const [lo, hi] = leafRange(w, pid);
    const b = pickB(w, pid, lo, hi, Math.min(hi, key + 1), key);
    if (!b) {
      r.push(`${other} has nothing to write on P${pid}.`, other);
      return writeAt(r, actor, kind, key, 'none', T, retry);
    }
    const theirs = r.alloc(buildRecord(w, b.kind, b.key, expect));
    r.threads[other] = { actor: other, pid, status: 'work', text: `${other} expects ${at(expect)}, holds private ${deltaLabel(w.mem[theirs])}` };
    r.push(`${other} reaches P${pid} at the same moment, loads the same slot (${at(expect)}) and builds ${deltaLabel(w.mem[theirs])}, also pointing at ${at(expect)}.`, other);
    const [first, firstAddr, second, secondAddr, secondOp] =
      race === 'second'
        ? ([actor, mine, other, theirs, { kind: b.kind, key: b.key }] as const)
        : ([other, theirs, actor, mine, { kind, key }] as const);
    r.cas(pid, expect, firstAddr);
    r.threads[first] = { actor: first, pid, status: 'ok', text: `${first}: CAS P${pid} ${at(expect)} → @${firstAddr} ✓` };
    r.push(`${first}'s CAS lands first: the slot held ${at(expect)}, so it now holds @${firstAddr}.`, first, 'ok', { pid, ok: true });
    r.cas(pid, expect, secondAddr);
    r.free(secondAddr);
    w.wastedHops += route.chases;
    r.threads[second] = { actor: second, pid, status: 'fail', text: `${second}: CAS P${pid} expected ${at(expect)}, found @${firstAddr} ✗` };
    r.push(
      `${second}'s CAS fails: it expected ${at(expect)} but the slot holds @${firstAddr}. Installing its delta now would silently drop ${first}'s write, so ${second} discards @${secondAddr} and retries.`,
      second,
      'fail',
      { pid, ok: false },
    );
    delete r.threads[first];
    delete r.threads[second];
    maybeConsolidate(r, pid, T, first);
    writeAt(r, second, secondOp.kind, secondOp.key, 'none', T, true);
    return;
  }
  maybeConsolidate(r, pid, T, actor);
}

function maybeConsolidate(r: Run, pid: Pid, T: number, actor: Actor) {
  const len = chainLength(r.w, pid);
  if (len >= T) consolidateSteps(r, pid, actor, 'none', T, `P${pid}'s chain is now ${len} deltas, at or past the threshold of ${T}. After its write, `);
}

export function actWrite(s: BwState, kind: WriteKind, key: number, race: Race, T: number): Frame[] {
  const r = new Run(s);
  const pre = lookup(r.w, key);
  if (kind === 'insert' && pre.found) {
    r.push(`Key ${key} already exists in P${pre.leaf}. A unique insert returns false and writes nothing.`, 'A');
    return r.frames;
  }
  if (kind !== 'insert' && !pre.found) {
    r.push(`Key ${key} is not in the tree, so the ${kind} writes nothing.`, 'A');
    return r.frames;
  }
  writeAt(r, 'A', kind === 'update' ? 'modify' : kind, key, race, T, false);
  return r.frames;
}

export function actConsolidate(s: BwState, key: number, race: Race, T: number): Frame[] {
  const r = new Run(s);
  helpAlong(r, key, null, 'A', T);
  const pid = lookup(r.w, key).leaf;
  consolidateSteps(r, pid, 'A', race, T, '');
  return r.frames;
}

export function actSplit(s: BwState, key: number, race: Race, T: number): Frame[] {
  const r = new Run(s);
  const w = r.w;
  helpAlong(r, key, lookup(w, key).leaf, 'A', T);
  const pid = lookup(w, key).leaf;
  const v = leafView(w, pid);
  if (v.entries.length < 2) {
    r.push(`P${pid} holds ${v.entries.length} key${v.entries.length === 1 ? '' : 's'}; a split needs at least two.`, 'A');
    return r.frames;
  }
  if (leafCount(w) >= MAX_LEAVES) {
    r.push(`The lab's root holds at most ${MAX_LEAVES} leaves, so it does not split further.`, 'A');
    return r.frames;
  }
  const keys = v.entries.map((e) => e[0]);
  const sep = keys[Math.floor((keys.length - 1) / 2)];
  const q = w.nextPid++;
  const upper = v.entries.filter((e) => e[0] > sep);
  const qAddr = r.alloc({ kind: 'leaf', entries: upper, high: v.high, right: v.right });
  w.map[q] = qAddr;
  const expect = w.map[pid] as Addr;
  r.threads.A = { actor: 'A', pid, status: 'work', text: `A expects ${at(expect)} on P${pid}; P${q} built` };
  r.push(
    `Phase 1. A splits P${pid} at key ${sep}: allocates page ID P${q}, builds its base page @${qAddr} with the keys above ${sep} (${upper.map((e) => e[0]).join(', ')}), high key ${v.high ?? '∞'} and side link ${v.right ? `P${v.right}` : '∅'}, and stores @${qAddr} in P${q}'s slot with a plain write — no CAS, because no other thread knows P${q} exists.`,
    'A',
  );
  let b: BWrite = null;
  let bAddr: Addr | null = null;
  if (race !== 'none') {
    const [lo] = leafRange(w, pid);
    b = pickB(w, pid, lo, sep, Math.floor((lo + sep) / 2) + 1, -1);
    if (b) {
      bAddr = r.alloc(buildRecord(w, b.kind, b.key, expect));
      r.threads.B = { actor: 'B', pid, status: 'work', text: `B expects ${at(expect)}, holds private ${deltaLabel(w.mem[bAddr])}` };
      r.push(`B loads P${pid}'s slot (${at(expect)}) to write key ${b.key}, below the split key.`, 'B');
    }
  }
  const sd = r.alloc({ kind: 'split', sep, right: q, next: expect });
  if (race === 'first' && b && bAddr != null) {
    r.cas(pid, expect, bAddr);
    r.threads.B = { actor: 'B', pid, status: 'ok', text: `B: CAS P${pid} ${at(expect)} → @${bAddr} ✓` };
    r.push(`B's CAS lands first: P${pid}'s slot now holds @${bAddr}.`, 'B', 'ok', { pid, ok: true });
    r.cas(pid, expect, sd);
    r.free(sd);
    r.free(qAddr);
    w.map[q] = null;
    delete w.map[q];
    w.nextPid--;
    w.failedSplits++;
    r.threads.A = { actor: 'A', pid, status: 'fail', text: `A: CAS P${pid} expected ${at(expect)}, found @${bAddr} ✗` };
    r.push(
      `A's split-delta CAS fails: P${pid} changed after A copied it, and a split delta pointing at ${at(expect)} would bypass B's ${deltaLabel(w.mem[bAddr])}. This is a failed split. A frees P${q}'s page and page ID — no other thread ever saw them — and the split can be tried again.`,
      'A',
      'fail',
      { pid, ok: false },
    );
    delete r.threads.A;
    delete r.threads.B;
    return r.frames;
  }
  r.cas(pid, expect, sd);
  w.splits++;
  r.threads.A = { actor: 'A', pid, status: 'ok', text: `A: CAS P${pid} ${at(expect)} → @${sd} ✓` };
  r.push(
    `A's CAS installs the split delta @${sd} on P${pid}: a half split. The root still sends every key up to ${leafRange(w, q)[1] === KEY_MAX ? '∞' : leafRange(w, q)[1]} to P${pid}; a search for a key above ${sep} reads the split delta and follows the side link to P${q}.`,
    'A',
    'ok',
    { pid, ok: true },
  );
  delete r.threads.A;
  if (b && bAddr != null) {
    r.cas(pid, expect, bAddr);
    r.free(bAddr);
    w.wastedHops += lookup(w, b.key).chases;
    r.threads.B = { actor: 'B', pid, status: 'fail', text: `B: CAS P${pid} expected ${at(expect)}, found @${sd} ✗` };
    r.push(`B's CAS fails against the split delta. B discards its delta and restarts from the root.`, 'B', 'fail', { pid, ok: false });
    delete r.threads.B;
    const route = lookup(w, b.key);
    const e2 = w.map[route.leaf] as Addr;
    const again = r.alloc(buildRecord(w, b.kind, b.key, e2));
    r.cas(route.leaf, e2, again);
    r.threads.B = { actor: 'B', pid: route.leaf, status: 'ok', text: `B: CAS P${route.leaf} ${at(e2)} → @${again} ✓` };
    r.push(`B's key ${b.key} is not above ${sep}, so B stays on P${pid}: it builds its delta on top of the split delta and the CAS succeeds.`, 'B', 'ok', { pid: route.leaf, ok: true });
    delete r.threads.B;
    maybeConsolidate(r, route.leaf, T, 'B');
  }
  return r.frames;
}

export function actPostIndex(s: BwState, race: Race, T: number): Frame[] {
  const r = new Run(s);
  const p = pendingSplits(r.w)[0];
  if (!p) {
    r.push('No split is waiting for its index entry.', 'A');
    return r.frames;
  }
  postIndexSteps(r, p.pid, 'A', race, false, T);
  return r.frames;
}

/* ------------------------------------------------------------------ UI */

const DELTA_H = 26;
const BASE_H = 36;

function nodeWidth(n: MemNode) {
  switch (n.kind) {
    case 'insert':
    case 'modify':
    case 'delete':
      return 54;
    case 'split':
      return 100;
    case 'index':
      return 126;
    default: {
      const [l1, l2] = baseLines(n);
      return Math.ceil(Math.max(90, 18 + l1.length * 6.3, 18 + (l2.length + 5) * 5.2));
    }
  }
}

function baseLines(n: MemNode): [string, string] {
  if (n.kind === 'inner') return [n.seps.map((e) => `${e.high === null ? '∞' : `≤${e.high}`}:P${e.pid}`).join(' '), 'root base page'];
  if (n.kind === 'leaf') {
    const ks = n.entries.map((e) => e[0]);
    const shown = ks.length > 9 ? `${ks.slice(0, 8).join(' ')} +${ks.length - 8}` : ks.length ? ks.join(' ') : '(empty)';
    return [shown, `high ${n.high ?? '∞'} · right ${n.right ? `P${n.right}` : '∅'}`];
  }
  return ['', ''];
}

export default function BwTreeMappingTableLab() {
  const [base, setBase] = useState<BwState>(() => initialState());
  const [frames, setFrames] = useState<Frame[]>([]);
  const [shown, setShown] = useState(0);
  const [key, setKey] = useState(40);
  const [T, setT] = useState(5);
  const [race, setRace] = useState<Race>('none');
  const [stepping, setStepping] = useState(true);

  const committed = frames.length ? frames[frames.length - 1].state : base;
  const frame = frames.length && shown > 0 ? frames[shown - 1] : null;
  const view = frame ? frame.state : base;
  const done = shown >= frames.length;

  const run = (fn: (s: BwState) => Frame[]) => {
    const start = committed;
    const fr = fn(start);
    setBase(start);
    setFrames(fr);
    setShown(stepping ? 1 : fr.length);
  };

  const read = useMemo(() => lookup(view, key), [view, key]);
  const pathDeltas = read.hops.filter((h) => h.t === 'node' && isDelta(view.mem[h.addr])).length;
  const pageOfKey = lookup(committed, key).leaf;
  const pending = pendingSplits(committed);
  const pids = livePids(view);
  const leafOrder = pids
    .filter((p) => p !== ROOT)
    .sort((a, b) => leafRange(view, a)[0] - leafRange(view, b)[0] || a - b);
  const rows = [ROOT, ...leafOrder];
  const maxChain = Math.max(...pids.map((p) => chainLength(view, p)));
  const routed = new Set(innerView(view).seps.map((e) => e.pid));
  const pendView = pendingSplits(view);

  const hopNodes = new Set(read.hops.filter((h) => h.t === 'node').map((h) => (h as { addr: Addr }).addr));
  const hopMaps = new Set(read.hops.filter((h) => h.t === 'map').map((h) => (h as { pid: Pid }).pid));

  // Layout
  const W = 640;
  const SLOT_X = 80;
  const SLOT_W = 58;
  const CHAIN_X = SLOT_X + SLOT_W + 22;
  const threadRows = (pid: Pid) => (frame ? frame.threads.filter((t) => t.pid === pid).length : 0);
  let y = 24;
  const layout = rows.map((pid) => {
    const n = threadRows(pid);
    const h = 50 + n * 20 + (n ? 8 : 0);
    const r = { pid, y, h };
    y += h;
    return r;
  });
  const retiredY = y + 8;
  const H = retiredY + 40;

  const pathText = read.hops
    .map((h) => (h.t === 'map' ? `P${h.pid} slot` : h.t === 'side' ? 'side link' : isDelta(view.mem[h.addr]) ? deltaLabel(view.mem[h.addr]) : `base @${h.addr}`))
    .join(' › ');

  return (
    <VizPanel
      title="A Bw-tree through its mapping table"
      subtitle="Every page is a logical ID in the mapping table. Writes prepend immutable delta records and publish them with one compare-and-swap on the page's slot; consolidation and splits use the same CAS. Pick a key, write it, split its page, and turn on a racing writer to watch a CAS fail."
      controls={
        <>
          <Slider label="Key" min={1} max={KEY_MAX} value={key} onChange={setKey} />
          <Slider label="Consolidate at chain length" min={2} max={16} value={T} onChange={setT} />
          <Choice
            label="Racing writer B"
            value={race}
            onChange={setRace}
            options={[
              { value: 'none', label: 'None' },
              { value: 'second', label: 'B’s CAS lands second' },
              { value: 'first', label: 'B’s CAS lands first' },
            ]}
          />
          <Check label="Step through each CAS" checked={stepping} onChange={setStepping} />
        </>
      }
      legend={
        <Legend
          items={[
            { label: 'Base page (immutable)', color: 'var(--viz-1)' },
            { label: 'Record delta: ins / mod / del', color: 'var(--viz-2)' },
            { label: 'Structure delta: split / index entry', color: 'var(--viz-3)' },
            { label: 'Retired, awaiting epoch', color: 'var(--viz-stale)' },
            { label: 'CAS succeeded', color: 'var(--viz-good)' },
            { label: 'CAS failed', color: 'var(--viz-warning)' },
            { label: 'Path of the read (dashed)', color: 'var(--viz-ink)', shape: 'line' },
          ]}
        />
      }
      stats={
        <Stats
          items={[
            { label: `Read key ${key}`, value: read.found ? `found (v${read.val})` : 'not found', hint: 'Evaluated on the state shown in the figure.' },
            {
              label: 'Pointer chases (B+tree: 2)',
              value: String(read.chases),
              hint: `Mapping-slot loads (${read.maps}) plus nodes dereferenced (${read.nodes}) on this read${read.sides ? `, including ${read.sides} side-link hop` : ''}. An in-place B+tree of the same two levels dereferences 2 nodes. A count of memory loads, not measured cache misses.`,
            },
            { label: 'Deltas walked', value: String(pathDeltas) },
            { label: 'Longest chain', value: String(maxChain), hint: `Consolidation runs when a write leaves a chain at ${T} or more. In the lab only writes trigger it; in the ICDE design a reader that notices a long chain consolidates too.` },
            { label: 'CAS ok / failed', value: `${view.casOk} / ${view.casFail}`, hint: `Failed CASes cost ${view.wastedHops} pointer chases of work that had to be redone.` },
            { label: 'Retired objects', value: String(view.retired.length), hint: 'Old chains and base pages unlinked by consolidation. They are freed only when no thread can still hold a pointer to them.' },
          ]}
        />
      }
      note={
        <Note>
          {frame ? (
            <>
              <strong>
                Step {shown} of {frames.length}.
              </strong>{' '}
              {frame.text}{' '}
            </>
          ) : (
            <>
              <strong>P2 is a base page plus three deltas.</strong> Its slot points at the newest delta, which points at the next, down to the base page.{' '}
            </>
          )}
          Reading key {key}: {pathText} — {read.chases} pointer chases{pendView.length ? `; ${pendView.map((p) => `P${p.pid}'s split is waiting for its index entry`).join('; ')}` : ''}.
        </Note>
      }
      table={
        <table className="viz-table">
          <thead>
            <tr>
              <th>Page ID</th>
              <th>Slot</th>
              <th>Chain, newest first</th>
              <th>Live keys</th>
              <th>High key / side link</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((pid) => {
              const addrs = chainAddrs(view, pid);
              const lv = pid === ROOT ? null : leafView(view, pid);
              return (
                <tr key={pid}>
                  <td>
                    P{pid}
                    {pid === ROOT ? ' (root)' : ''}
                  </td>
                  <td>{at(view.map[pid])}</td>
                  <td>{addrs.map((a) => `${deltaLabel(view.mem[a])} @${a}`).join(' → ')}</td>
                  <td>{lv ? lv.entries.map((e) => e[0]).join(', ') || '—' : innerView(view).seps.map((e) => `${e.high === null ? '∞' : `≤${e.high}`}→P${e.pid}`).join(', ')}</td>
                  <td>{lv ? `${lv.high ?? '∞'} / ${lv.right ? `P${lv.right}` : '∅'}` : '—'}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      }
    >
      <div className="viz-controls">
        <Button onClick={() => run((s) => actWrite(s, 'insert', key, race, T))}>Insert {key}</Button>
        <Button onClick={() => run((s) => actWrite(s, 'update', key, race, T))}>Update {key}</Button>
        <Button onClick={() => run((s) => actWrite(s, 'delete', key, race, T))}>Delete {key}</Button>
        <Button onClick={() => run((s) => actConsolidate(s, key, race, T))}>Consolidate P{pageOfKey}</Button>
        <Button onClick={() => run((s) => actSplit(s, key, race, T))}>Split P{pageOfKey}</Button>
        <Button onClick={() => run((s) => actPostIndex(s, race, T))} disabled={pending.length === 0}>
          Post index entry{pending.length ? ` for P${pending[0].right}` : ''}
        </Button>
        <Button primary onClick={() => setShown((n) => Math.min(frames.length, n + 1))} disabled={done}>
          Next step
        </Button>
        <Button onClick={() => setShown(frames.length)} disabled={done}>
          Finish
        </Button>
        <Button
          onClick={() => {
            setBase(initialState());
            setFrames([]);
            setShown(0);
          }}
        >
          Reset
        </Button>
      </div>
      <svg viewBox={`0 0 ${W} ${H}`} width={W} height={H} style={{ minWidth: 560 }} role="img" aria-label={`Bw-tree mapping table with ${rows.length} pages; reading key ${key} takes ${read.chases} pointer chases`}>
        <text x={8} y={14} fontSize={11} fill="var(--viz-ink-2)">
          page
        </text>
        <text x={SLOT_X} y={14} fontSize={11} fill="var(--viz-ink-2)">
          mapping slot
        </text>
        <text x={CHAIN_X} y={14} fontSize={11} fill="var(--viz-ink-2)">
          physical chain: newest delta → … → base page
        </text>
        {layout.map(({ pid, y: ry, h }) => {
          const addrs = chainAddrs(view, pid);
          const nodes = addrs.map((a) => ({ a, n: view.mem[a] }));
          const avail = W - CHAIN_X - 8;
          const gap = 12;
          let widths = nodes.map((x) => nodeWidth(x.n));
          let items: ({ a: Addr; n: MemNode; w: number } | { more: number; w: number; hidden: Addr[] })[] = nodes.map((x, i) => ({ ...x, w: widths[i] }));
          const total = () => items.reduce((s2, it) => s2 + it.w + gap, 0);
          if (total() > avail && nodes.length > 3) {
            let keepHead = nodes.length - 2;
            while (keepHead >= 1) {
              const hidden = nodes.slice(keepHead, nodes.length - 1);
              items = [...nodes.slice(0, keepHead).map((x) => ({ ...x, w: nodeWidth(x.n) })), { more: hidden.length, w: 46, hidden: hidden.map((x) => x.a) }, { ...nodes[nodes.length - 1], w: nodeWidth(nodes[nodes.length - 1].n) }];
              if (total() <= avail) break;
              keepHead--;
            }
          }
          widths = items.map((it) => it.w);
          const isQUnrouted = pid !== ROOT && !routed.has(pid);
          const reachable = !isQUnrouted || pendView.some((p) => p.right === pid);
          const cy = ry + 22;
          const slotStroke = frame?.cas?.pid === pid ? (frame.cas.ok ? 'var(--viz-good)' : 'var(--viz-warning)') : 'var(--viz-ink-muted)';
          let x = CHAIN_X;
          const leafRng = pid === ROOT ? null : leafRange(view, pid);
          return (
            <g key={pid}>
              <line x1={4} x2={W - 4} y1={ry + h - 2} y2={ry + h - 2} stroke="var(--viz-grid)" />
              <text x={8} y={cy - 2} fontSize={13} fontWeight={600} fill="var(--viz-ink)">
                P{pid}
              </text>
              <text x={8} y={cy + 12} fontSize={9} fill="var(--viz-ink-2)">
                {pid === ROOT ? 'root' : isQUnrouted ? (reachable ? 'not in root,' : 'unreachable') : leafRng ? `keys ${leafRng[0] + 1}–${leafRng[1]}` : ''}
              </text>
              {isQUnrouted && reachable ? (
                <text x={8} y={cy + 22} fontSize={9} fill="var(--viz-ink-2)">
                  via P{pendView.find((pp) => pp.right === pid)?.pid} link
                </text>
              ) : null}
              <rect x={SLOT_X} y={cy - 12} width={SLOT_W} height={24} rx={4} fill="var(--viz-surface)" stroke={slotStroke} strokeWidth={frame?.cas?.pid === pid ? 2.5 : 1} />
              <text x={SLOT_X + SLOT_W / 2} y={cy + 4} textAnchor="middle" fontSize={12} fill="var(--viz-ink)">
                {at(view.map[pid])}
              </text>
              {hopMaps.has(pid) ? <rect x={SLOT_X - 3} y={cy - 15} width={SLOT_W + 6} height={30} rx={5} fill="none" stroke="var(--viz-ink)" strokeWidth={1.5} strokeDasharray="4 3" /> : null}
              {items.map((it, i) => {
                const x0 = x;
                x += it.w + gap;
                const arrow = (
                  <g>
                    <line x1={i === 0 ? SLOT_X + SLOT_W : x0 - gap} x2={x0 - 3} y1={cy} y2={cy} stroke="var(--viz-ink-2)" strokeWidth={1.2} />
                    <path d={`M${x0 - 1},${cy} l-6,-3.5 v7 z`} fill="var(--viz-ink-2)" />
                  </g>
                );
                if ('more' in it) {
                  const onPath = it.hidden.some((a) => hopNodes.has(a));
                  return (
                    <g key={`more-${i}`}>
                      {arrow}
                      <rect x={x0} y={cy - DELTA_H / 2} width={it.w} height={DELTA_H} rx={4} fill="var(--viz-surface)" stroke="var(--viz-ink-2)" />
                      <text x={x0 + it.w / 2} y={cy + 4} textAnchor="middle" fontSize={11} fill="var(--viz-ink)">
                        +{it.more}
                      </text>
                      {onPath ? <rect x={x0 - 3} y={cy - DELTA_H / 2 - 3} width={it.w + 6} height={DELTA_H + 6} rx={5} fill="none" stroke="var(--viz-ink)" strokeWidth={1.5} strokeDasharray="4 3" /> : null}
                    </g>
                  );
                }
                const n = it.n;
                const delta = isDelta(n);
                const stroke = !delta ? 'var(--viz-1)' : n.kind === 'split' || n.kind === 'index' ? 'var(--viz-3)' : 'var(--viz-2)';
                const hgt = delta ? DELTA_H : BASE_H;
                const [l1, l2] = delta ? [deltaLabel(n), `@${it.a}`] : baseLines(n);
                return (
                  <g key={it.a}>
                    {arrow}
                    <rect x={x0} y={cy - hgt / 2} width={it.w} height={hgt} rx={4} fill="var(--viz-surface)" stroke={stroke} strokeWidth={2} />
                    {delta ? (
                      <>
                        <text x={x0 + it.w / 2} y={cy + 1} textAnchor="middle" fontSize={11} fill="var(--viz-ink)">
                          {l1}
                        </text>
                        <text x={x0 + it.w / 2} y={cy + 11} textAnchor="middle" fontSize={9} fill="var(--viz-ink-2)">
                          {l2}
                        </text>
                      </>
                    ) : (
                      <>
                        <text x={x0 + 8} y={cy - 3} fontSize={11} fill="var(--viz-ink)">
                          {l1}
                        </text>
                        <text x={x0 + 8} y={cy + 12} fontSize={9} fill="var(--viz-ink-2)">
                          @{it.a} · {l2}
                        </text>
                      </>
                    )}
                    {hopNodes.has(it.a) ? <rect x={x0 - 3} y={cy - hgt / 2 - 3} width={it.w + 6} height={hgt + 6} rx={5} fill="none" stroke="var(--viz-ink)" strokeWidth={1.5} strokeDasharray="4 3" /> : null}
                  </g>
                );
              })}
              {frame
                ? frame.threads
                    .filter((t) => t.pid === pid)
                    .map((t, i) => {
                      const ty = ry + 56 + i * 20;
                      const col = t.status === 'ok' ? 'var(--viz-good)' : t.status === 'fail' ? 'var(--viz-warning)' : 'var(--viz-ink-2)';
                      return (
                        <g key={t.actor}>
                          <rect x={SLOT_X} y={ty - 12} width={22} height={17} rx={8} fill="var(--viz-surface)" stroke={col} strokeWidth={2} />
                          <text x={SLOT_X + 11} y={ty + 1} textAnchor="middle" fontSize={11} fontWeight={600} fill="var(--viz-ink)">
                            {t.actor}
                          </text>
                          <text x={SLOT_X + 30} y={ty + 1} fontSize={11} fill="var(--viz-ink)">
                            {t.text}
                          </text>
                        </g>
                      );
                    })
                : null}
            </g>
          );
        })}
        <text x={8} y={retiredY + 18} fontSize={11} fill="var(--viz-ink-2)">
          retired
        </text>
        {view.retired.length === 0 ? (
          <text x={SLOT_X} y={retiredY + 18} fontSize={11} fill="var(--viz-ink-2)">
            nothing yet — consolidation retires the old chain here until no reader can still hold it
          </text>
        ) : (
          (() => {
            const cap = 10;
            const list = view.retired.slice(-cap);
            return (
              <>
                {list.map((a, i) => (
                  <g key={a}>
                    <rect x={SLOT_X + i * 50} y={retiredY + 4} width={44} height={20} rx={3} fill="var(--viz-surface)" stroke="var(--viz-stale)" strokeDasharray="3 2" />
                    <text x={SLOT_X + i * 50 + 22} y={retiredY + 18} textAnchor="middle" fontSize={10} fill="var(--viz-ink-2)">
                      @{a}
                    </text>
                  </g>
                ))}
                {view.retired.length > cap ? (
                  <text x={SLOT_X + cap * 50} y={retiredY + 18} fontSize={10} fill="var(--viz-ink-2)">
                    +{view.retired.length - cap} older
                  </text>
                ) : null}
              </>
            );
          })()
        )}
      </svg>
      {frames.length ? (
        <ol style={{ margin: '0.5rem 0 0', paddingLeft: '1.2rem', fontSize: '0.8rem', display: 'grid', gap: 4 }}>
          {frames.slice(0, shown).map((f, i) => (
            <li
              key={i}
              style={{
                color: 'var(--viz-ink)',
                borderLeft: `3px solid ${f.tone === 'fail' ? 'var(--viz-warning)' : f.tone === 'ok' ? 'var(--viz-good)' : 'transparent'}`,
                paddingLeft: 6,
              }}
            >
              {f.text}
            </li>
          ))}
        </ol>
      ) : null}
    </VizPanel>
  );
}
