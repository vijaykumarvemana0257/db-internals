import { useEffect, useId, useMemo, useRef, useState } from 'react';
import { VizPanel, Segmented, Choice, Check, Slider, Button, Legend, Stats, Note, TooltipHost, useTip, useTicker, makeRng, fmtNum } from './Viz';

/**
 * Safe memory reclamation for a latch-free linked list: three readers walk the list while a writer
 * unlinks nodes, and the chosen scheme decides when an unlinked node's memory may be freed.
 *
 * Model (discrete ticks, deterministic):
 * - The list always holds 10 live nodes. Every `interval` ticks the writer inserts a node after a random
 *   position, then unlinks a random node. Addresses come from a LIFO free list, so freed memory is reused
 *   by the next insert — which is how a stale pointer turns into an ABA read.
 * - R1 and R2 run point lookups (walk to a target depth, 1 tick per node). R3 runs a range scan over the
 *   whole list at `scanTicksPerNode` ticks per node. Any reader can be paused (a descheduled or stalled
 *   thread); preset stalls resolve at the first tick the reader is mid-scan or between operations.
 * - Epochs ('ebr') follow crossbeam-epoch's rules: an operation pins the current global epoch; the global
 *   epoch advances only when every pinned reader is pinned in it; a node retired in epoch e is freed once
 *   the global epoch reaches e + 2. crossbeam tries to advance every 128 pins and seals garbage in bags of
 *   64; the model tries every tick and tags each node individually. The writer is not a participant (a real
 *   writer would pin too while it finds its victim).
 * - Hazard pointers ('hp') follow Michael (2004): each reader publishes the node it stands on. To advance it
 *   publishes the successor and validates that the node it came from is still linked; if not, it restarts
 *   from the head. The writer scans once its retire list reaches R and frees every node no hazard pointer
 *   names. One published pointer per reader is a simplification (Michael's list-based set uses two, *hp0 and *hp1).
 * - RCU ('rcu') runs one grace period at a time. Nodes retired while a grace period is running wait for the
 *   next one. A grace period waits for the readers inside a read-side critical section when it starts
 *   (urcu-memb style) or, with `qsbr`, for every registered reader to report a quiescent state, which the
 *   model does at the end of each operation.
 * - 'none' frees a node the moment it is unlinked.
 * - "Reader bookkeeping writes" is a model count of the shared-memory stores readers make: pin + unpin per
 *   operation (epochs), one publish per node visited (hazard pointers), lock + unlock per operation (RCU),
 *   one quiescent-state report per operation (QSBR).
 */

export type Scheme = 'ebr' | 'hp' | 'rcu' | 'none';
export type PauseWhen = 'now' | 'midscan' | 'idle';
export type PauseEvent = { t: number; reader: number; paused: boolean; when?: PauseWhen };
export type Config = { interval: number; scanTicksPerNode: number; hpThreshold: number; qsbr: boolean };

export const LIST_LEN = 10;
export const MAX_T = 600;
export const DEFAULT_CONFIG: Config = { interval: 3, scanTicksPerNode: 3, hpThreshold: 6, qsbr: false };
const LOOKUP_DEPTHS = [
  [3, 8, 5, 10, 2, 7, 4, 9, 6],
  [6, 2, 9, 4, 7, 10, 3, 5, 8],
];

type SlotState = 'live' | 'retired' | 'free';
export type Slot = { key: number; next: number | null; gen: number; state: SlotState; retiredAt: number; tag: number };
export type ReaderState = {
  idx: number;
  name: string;
  kind: 'lookup' | 'scan';
  tpn: number;
  idle: number;
  phase: 'idle' | 'walk';
  idleLeft: number;
  pos: number | null;
  posGen: number;
  wait: number;
  visited: number;
  depth: number;
  ops: number;
  paused: boolean;
  pending: PauseEvent | null;
  pinned: number | null;
  inCS: boolean;
  hp: number | null;
  writes: number;
  restarts: number;
  unsafe: number;
};
export type Sim = {
  t: number;
  scheme: Scheme;
  cfg: Config;
  slots: Slot[];
  head: number | null;
  freeList: number[];
  nextKey: number;
  rng: () => number;
  readers: ReaderState[];
  epoch: number;
  rlist: number[];
  hpScans: number;
  gp: { id: number; batch: number[]; wait: Set<number> } | null;
  nextBatch: number[];
  gpId: number;
  gpDone: number;
  freed: number;
  peak: number;
  violations: number;
  unsafeReads: number;
  /** 'none' only: unsafe reads split into use-after-free and reuse (ABA). */
  uafReads: number;
  abaReads: number;
  log: string[];
  /** Per-tick series, index = tick. */
  hist: { un: number[]; freed: number[]; restarts: number[]; writes: number[]; ops: number[]; unsafe: number[] };
};

export const READER_NAMES = ['R1', 'R2', 'R3'];

function newSim(scheme: Scheme, cfg: Config): Sim {
  const slots: Slot[] = [];
  for (let i = 0; i < LIST_LEN; i++) slots.push({ key: i + 1, next: i + 1 < LIST_LEN ? i + 1 : null, gen: i + 1, state: 'live', retiredAt: -1, tag: -1 });
  const mk = (idx: number, kind: 'lookup' | 'scan', tpn: number, idle: number): ReaderState => ({
    idx, name: READER_NAMES[idx], kind, tpn, idle, phase: 'idle', idleLeft: idx, pos: null, posGen: 0, wait: 0, visited: 0, depth: 0, ops: 0,
    paused: false, pending: null, pinned: null, inCS: false, hp: null, writes: 0, restarts: 0, unsafe: 0,
  });
  return {
    t: 0, scheme, cfg, slots, head: 0, freeList: [], nextKey: LIST_LEN + 1, rng: makeRng(7),
    readers: [mk(0, 'lookup', 1, 1), mk(1, 'lookup', 1, 2), mk(2, 'scan', cfg.scanTicksPerNode, 3)],
    epoch: 1, rlist: [], hpScans: 0, gp: null, nextBatch: [], gpId: 0, gpDone: 0, freed: 0, peak: 0, violations: 0, unsafeReads: 0, uafReads: 0, abaReads: 0, log: [],
    hist: { un: [0], freed: [0], restarts: [0], writes: [0], ops: [0], unsafe: [0] },
  };
}

export function liveOrder(sim: Sim) {
  const out: number[] = [];
  let p = sim.head;
  while (p !== null && out.length < 1000) {
    out.push(p);
    p = sim.slots[p].next;
  }
  return out;
}

export const nodeLabel = (sim: Sim, s: number) => `#${sim.slots[s].key}`;

function freeSlot(sim: Sim, s: number) {
  sim.slots[s].state = 'free';
  sim.freeList.push(s);
  sim.freed++;
}

function alloc(sim: Sim, next: number | null) {
  const key = sim.nextKey++;
  const reuse = sim.freeList.pop();
  // A node's key doubles as its allocation generation: a reused address gets a new key.
  const slot: Slot = { key, next, gen: key, state: 'live', retiredAt: -1, tag: -1 };
  if (reuse !== undefined) {
    sim.slots[reuse] = slot;
    return reuse;
  }
  sim.slots.push(slot);
  return sim.slots.length - 1;
}

function setNext(sim: Sim, pred: number | null, next: number | null) {
  if (pred === null) sim.head = next;
  else sim.slots[pred].next = next;
}

function hpScan(sim: Sim) {
  const held = new Set(sim.readers.map((r) => r.hp).filter((x): x is number => x !== null));
  const keep: number[] = [];
  let n = 0;
  for (const s of sim.rlist) {
    if (held.has(s)) keep.push(s);
    else {
      freeSlot(sim, s);
      n++;
    }
  }
  sim.hpScans++;
  sim.log.push(
    `Retire list reached R = ${sim.cfg.hpThreshold}: scan the ${sim.readers.length} hazard pointers, free ${n} node${n === 1 ? '' : 's'}` +
      (keep.length ? `, keep ${keep.map((s) => nodeLabel(sim, s)).join(', ')} (still published).` : '.'),
  );
  sim.rlist = keep;
}

function writerStep(sim: Sim) {
  // Insert after a random position (0 = after head), reusing the most recently freed address.
  const live = liveOrder(sim);
  const at = Math.floor(sim.rng() * (live.length + 1));
  const pred = at === 0 ? null : live[at - 1];
  const succ = pred === null ? sim.head : sim.slots[pred].next;
  const reusing = sim.freeList.length > 0 ? sim.freeList[sim.freeList.length - 1] : null;
  const n = alloc(sim, succ);
  setNext(sim, pred, n);
  // Unlink a random live node (never the one just inserted).
  const live2 = liveOrder(sim).filter((s) => s !== n);
  const vi = Math.floor(sim.rng() * live2.length);
  const victim = live2[vi];
  const order = liveOrder(sim);
  const pi = order.indexOf(victim);
  const vpred = pi === 0 ? null : order[pi - 1];
  setNext(sim, vpred, sim.slots[victim].next);
  const v = sim.slots[victim];
  v.state = 'retired';
  v.retiredAt = sim.t;
  const vl = nodeLabel(sim, victim);
  const reuseNote = reusing !== null ? ` at address @${reusing} (memory freed earlier)` : '';
  switch (sim.scheme) {
    case 'ebr':
      v.tag = sim.epoch;
      sim.log.push(`Writer inserts #${sim.slots[n].key}, unlinks ${vl} and retires it in epoch ${sim.epoch}.`);
      break;
    case 'hp':
      sim.rlist.push(victim);
      sim.log.push(`Writer inserts #${sim.slots[n].key}, unlinks ${vl} and adds it to its retire list (${sim.rlist.length}/${sim.cfg.hpThreshold}).`);
      if (sim.rlist.length >= sim.cfg.hpThreshold) hpScan(sim);
      break;
    case 'rcu':
      sim.nextBatch.push(victim);
      sim.log.push(`Writer inserts #${sim.slots[n].key}, unlinks ${vl} and queues it for a grace period.`);
      break;
    case 'none':
      freeSlot(sim, victim);
      sim.log.push(`Writer inserts #${sim.slots[n].key}${reuseNote}, unlinks ${vl} and frees it at once (address @${victim}).`);
      break;
  }
}

function endOp(sim: Sim, r: ReaderState) {
  if (sim.scheme === 'ebr') {
    r.pinned = null;
    r.writes++;
  } else if (sim.scheme === 'rcu') {
    r.writes++;
    r.inCS = false;
    if (sim.gp) sim.gp.wait.delete(r.idx);
  } else if (sim.scheme === 'hp') {
    r.hp = null;
  }
  r.ops++;
  r.phase = 'idle';
  r.idleLeft = r.idle;
  r.pos = null;
}

function beginOp(sim: Sim, r: ReaderState) {
  if (sim.scheme === 'ebr') {
    r.pinned = sim.epoch;
    r.writes++;
  } else if (sim.scheme === 'rcu' && !sim.cfg.qsbr) {
    r.inCS = true;
    r.writes++;
  }
  r.depth = r.kind === 'scan' ? Infinity : LOOKUP_DEPTHS[r.idx][r.ops % LOOKUP_DEPTHS[r.idx].length];
  const first = sim.head;
  if (first === null) return endOp(sim, r);
  r.phase = 'walk';
  r.pos = first;
  r.posGen = sim.slots[first].gen;
  r.visited = 1;
  r.wait = r.tpn;
  if (sim.scheme === 'hp') {
    r.hp = first;
    r.writes++;
  }
}

function readerStep(sim: Sim, r: ReaderState) {
  if (r.paused) return;
  if (r.phase === 'idle') {
    if (r.idleLeft > 0) {
      r.idleLeft--;
      return;
    }
    beginOp(sim, r);
    return;
  }
  r.wait--;
  if (r.wait > 0) return;
  const pos = r.pos as number;
  const s = sim.slots[pos];
  if ((s.state === 'free' && r.posGen !== -1) || (s.state !== 'free' && s.gen !== r.posGen)) {
    if (sim.scheme === 'none') {
      r.unsafe++;
      sim.unsafeReads++;
      if (s.state === 'free') {
        sim.uafReads++;
        // Freed memory usually still holds the old bytes, so the read "works" — undefined behaviour that goes unnoticed.
        sim.log.push(`${r.name} reads the next pointer of address @${pos}, which was already freed: use-after-free. The old bytes are still there, so nothing visibly fails.`);
        r.posGen = -1;
      } else {
        sim.abaReads++;
        sim.log.push(`${r.name} reads address @${pos}, which has been reused for #${s.key}: it follows that node's next pointer into the wrong part of the list (ABA).`);
        r.posGen = s.gen;
      }
    } else {
      sim.violations++;
    }
  }
  if (sim.scheme === 'hp' && s.state === 'retired') {
    r.restarts++;
    sim.log.push(`${r.name} publishes the successor of ${nodeLabel(sim, pos)}, re-validates and finds ${nodeLabel(sim, pos)} unlinked: restart from the head.`);
    const first = sim.head as number;
    r.pos = first;
    r.posGen = sim.slots[first].gen;
    r.hp = first;
    r.writes++;
    r.visited = 1;
    r.wait = r.tpn;
    return;
  }
  if (r.visited >= r.depth || s.next === null) return endOp(sim, r);
  const nx = s.next;
  r.pos = nx;
  r.posGen = sim.slots[nx].gen;
  r.visited++;
  r.wait = r.tpn;
  if (sim.scheme === 'hp') {
    r.hp = nx;
    r.writes++;
  }
}

function applyPauses(sim: Sim, events: PauseEvent[]) {
  for (const ev of events) if (ev.t === sim.t) sim.readers[ev.reader].pending = ev;
  for (const r of sim.readers) {
    const ev = r.pending;
    if (!ev) continue;
    const when = ev.when ?? 'now';
    const ready = when === 'now' || (when === 'midscan' && r.phase === 'walk' && r.visited >= 3) || (when === 'idle' && r.phase === 'idle');
    if (ready) {
      if (r.paused !== ev.paused) sim.log.push(ev.paused ? `${r.name} stalls ${r.phase === 'walk' ? `mid-operation, standing on ${nodeLabel(sim, r.pos as number)}` : 'between operations'}.` : `${r.name} resumes.`);
      r.paused = ev.paused;
      r.pending = null;
    }
  }
}

function reclaim(sim: Sim) {
  if (sim.scheme === 'ebr') {
    if (sim.readers.every((r) => r.pinned === null || r.pinned === sim.epoch)) sim.epoch++;
    let n = 0;
    for (let i = 0; i < sim.slots.length; i++) {
      const s = sim.slots[i];
      if (s.state === 'retired' && s.tag <= sim.epoch - 2) {
        freeSlot(sim, i);
        n++;
      }
    }
    if (n) sim.log.push(`Global epoch is ${sim.epoch}: free ${n} node${n === 1 ? '' : 's'} retired in epoch ${sim.epoch - 2} or earlier.`);
  } else if (sim.scheme === 'rcu') {
    if (sim.gp && sim.gp.wait.size === 0) {
      for (const s of sim.gp.batch) freeSlot(sim, s);
      sim.gpDone++;
      sim.log.push(`Grace period ${sim.gp.id} ends: every reader it waited for has ${sim.cfg.qsbr ? 'reported a quiescent state' : 'left its read-side critical section'}. Free ${sim.gp.batch.length} node${sim.gp.batch.length === 1 ? '' : 's'}.`);
      sim.gp = null;
    }
    if (!sim.gp && sim.nextBatch.length > 0) {
      const wait = new Set(sim.readers.filter((r) => (sim.cfg.qsbr ? true : r.inCS)).map((r) => r.idx));
      sim.gp = { id: ++sim.gpId, batch: sim.nextBatch, wait };
      sim.nextBatch = [];
      if (wait.size === 0) {
        for (const s of sim.gp.batch) freeSlot(sim, s);
        sim.gpDone++;
        sim.gp = null;
      }
    }
  }
}

/** Run one scheme from tick 0 to `until`. */
export function simulate(scheme: Scheme, cfg: Config, events: PauseEvent[], until: number) {
  const sim = newSim(scheme, cfg);
  const T = Math.max(0, Math.min(MAX_T, Math.floor(until)));
  for (let t = 1; t <= T; t++) {
    sim.t = t;
    sim.log = [];
    applyPauses(sim, events);
    if (t % cfg.interval === 0) writerStep(sim);
    for (const r of sim.readers) readerStep(sim, r);
    reclaim(sim);
    const un = unreclaimed(sim);
    sim.peak = Math.max(sim.peak, un);
    let restarts = 0, writes = 0, ops = 0;
    for (const r of sim.readers) {
      restarts += r.restarts;
      writes += r.writes;
      ops += r.ops;
    }
    sim.hist.un.push(un);
    sim.hist.freed.push(sim.freed);
    sim.hist.restarts.push(restarts);
    sim.hist.writes.push(writes);
    sim.hist.ops.push(ops);
    sim.hist.unsafe.push(sim.unsafeReads);
  }
  return sim;
}

export function unreclaimed(sim: Sim) {
  let n = 0;
  for (const s of sim.slots) if (s.state === 'retired') n++;
  return n;
}

/* ------------------------------------------------------------------ view helpers */

export const SCENARIOS: Record<string, { label: string; events: PauseEvent[]; showAt: number }> = {
  mid: {
    label: 'R3 stalls mid-scan, resumes at t = 300',
    events: [
      { t: 20, reader: 2, paused: true, when: 'midscan' },
      { t: 300, reader: 2, paused: false, when: 'now' },
    ],
    showAt: 200,
  },
  idle: {
    label: 'R3 stalls between operations, resumes at t = 300',
    events: [
      { t: 20, reader: 2, paused: true, when: 'idle' },
      { t: 300, reader: 2, paused: false, when: 'now' },
    ],
    showAt: 200,
  },
  steady: { label: 'No stalls', events: [], showAt: 150 },
};

export type Group = { label: string; nodes: number[]; tone: 'stale' | 'critical' };

const names = (sim: Sim, idxs: Iterable<number>) => [...idxs].map((i) => sim.readers[i].name).join(', ');

/** Readers currently preventing reclamation (or, without a scheme, reading memory they do not own). */
export function blockers(sim: Sim) {
  const out = new Set<number>();
  for (const r of sim.readers) {
    if (sim.scheme === 'ebr' && r.pinned !== null && r.pinned < sim.epoch) out.add(r.idx);
    if (sim.scheme === 'hp' && r.hp !== null && sim.slots[r.hp].state === 'retired') out.add(r.idx);
    if (sim.scheme === 'rcu' && sim.gp && sim.gp.wait.has(r.idx)) out.add(r.idx);
    if (sim.scheme === 'none' && r.phase === 'walk' && r.pos !== null && (sim.slots[r.pos].state === 'free' || sim.slots[r.pos].gen !== r.posGen)) out.add(r.idx);
  }
  return out;
}

export function retiredGroups(sim: Sim): Group[] {
  const standing = new Set(sim.readers.filter((r) => r.phase === 'walk' && r.pos !== null).map((r) => r.pos as number));
  const order = (ns: number[]) => [...ns].sort((a, b) => Number(standing.has(b)) - Number(standing.has(a)) || sim.slots[a].retiredAt - sim.slots[b].retiredAt);
  const retired = sim.slots.map((s, i) => (s.state === 'retired' ? i : -1)).filter((i) => i >= 0);
  if (sim.scheme === 'ebr') {
    const tags = [...new Set(retired.map((i) => sim.slots[i].tag))].sort((a, b) => a - b);
    return tags.map((tag) => ({ label: `retired in epoch ${tag}, freed at ${tag + 2}`, nodes: order(retired.filter((i) => sim.slots[i].tag === tag)), tone: 'stale' as const }));
  }
  if (sim.scheme === 'hp') return sim.rlist.length ? [{ label: `writer's retire list, ${sim.rlist.length} of R = ${sim.cfg.hpThreshold}`, nodes: order(sim.rlist), tone: 'stale' }] : [];
  if (sim.scheme === 'rcu') {
    const g: Group[] = [];
    if (sim.gp) g.push({ label: `grace period ${sim.gp.id}, waiting for ${sim.gp.wait.size ? names(sim, sim.gp.wait) : 'nobody'}`, nodes: order(sim.gp.batch), tone: 'stale' });
    if (sim.nextBatch.length) g.push({ label: 'queued for the next grace period', nodes: order(sim.nextBatch), tone: 'stale' });
    return g;
  }
  const bad = sim.readers.filter((r) => r.phase === 'walk' && r.pos !== null && sim.slots[r.pos].state === 'free').map((r) => r.pos as number);
  return bad.length ? [{ label: 'freed memory a reader still points at', nodes: [...new Set(bad)], tone: 'critical' }] : [];
}

export function statusLine(sim: Sim) {
  const R = sim.readers;
  switch (sim.scheme) {
    case 'ebr': {
      const pinned = R.filter((r) => r.pinned !== null);
      return `Global epoch ${sim.epoch} · pinned: ${pinned.length ? pinned.map((r) => `${r.name} in ${r.pinned}`).join(', ') : 'nobody'} · it advances once every pinned reader is in ${sim.epoch}`;
    }
    case 'hp':
      return `Hazard pointers: ${R.map((r) => `${r.name} → ${r.hp === null ? '—' : nodeLabel(sim, r.hp)}`).join(', ')} · retire list ${sim.rlist.length}, scan at R = ${sim.cfg.hpThreshold}`;
    case 'rcu':
      return sim.gp
        ? `Grace period ${sim.gp.id} waits for ${sim.gp.wait.size ? names(sim, sim.gp.wait) : 'nobody'} to ${sim.cfg.qsbr ? 'report a quiescent state' : 'leave a read-side critical section'}. ${sim.gpDone} grace period${sim.gpDone === 1 ? '' : 's'} done.`
        : `No grace period running. ${sim.gpDone} grace period${sim.gpDone === 1 ? '' : 's'} done.`;
    default:
      return `Free list: ${sim.freeList.length ? sim.freeList.map((s) => `@${s}`).join(', ') : 'empty'}. The next insert reuses the last address freed.`;
  }
}

/* ------------------------------------------------------------------ component */

const W = 700;
const NODE_W = 46;
const NODE_H = 28;
const STEP_X = 64;
const LIST_X = 58;
const MAX_GROUP_BOXES = 6;
const MAX_GROUPS = 6;

function ReaderBadges({ sim, readers, cx, y, block }: { sim: Sim; readers: ReaderState[]; cx: number; y: number; block: Set<number> }) {
  const tip = useTip();
  const bw = 20;
  const x0 = cx - (readers.length * (bw + 1)) / 2;
  return (
    <>
      {readers.map((r, i) => {
        const danger = sim.scheme === 'none' && block.has(r.idx);
        return (
          <g key={r.idx} {...tip(<>{r.name}{r.paused ? ' (stalled)' : ''}{block.has(r.idx) ? (danger ? ': reading memory it does not own' : ': holding back reclamation') : ''}</>)}>
            <rect
              x={x0 + i * (bw + 1)}
              y={y}
              width={bw}
              height={16}
              rx={3}
              fill="var(--viz-surface)"
              stroke={block.has(r.idx) ? (danger ? 'var(--viz-critical)' : 'var(--viz-warning)') : 'var(--viz-ink-2)'}
              strokeWidth={block.has(r.idx) ? 2.5 : 1}
              strokeDasharray={r.paused ? '3 2' : undefined}
            />
            <text x={x0 + i * (bw + 1) + bw / 2} y={y + 12} textAnchor="middle" fontSize={10} fill="var(--viz-ink)">
              {r.name}
            </text>
          </g>
        );
      })}
    </>
  );
}

function NodeBox({ sim, slot, x, y, tone, w = NODE_W, h = NODE_H }: { sim: Sim; slot: number; x: number; y: number; tone: 'live' | 'stale' | 'critical'; w?: number; h?: number }) {
  const tip = useTip();
  const s = sim.slots[slot];
  const stroke = tone === 'live' ? 'var(--viz-ink-2)' : tone === 'stale' ? 'var(--viz-stale)' : 'var(--viz-critical)';
  const heldBy = sim.scheme === 'hp' ? sim.readers.filter((r) => r.hp === slot).map((r) => r.name) : [];
  return (
    <g
      {...tip(
        <>
          {s.state === 'free' ? `address @${slot}, freed` : `#${s.key} at address @${slot}`}
          {s.state === 'retired' ? `, unlinked at t = ${s.retiredAt}` : ''}
          {heldBy.length ? `, named by ${heldBy.join(' and ')}'s hazard pointer` : ''}
        </>,
      )}
    >
      <rect x={x} y={y} width={w} height={h} rx={4} fill={tone === 'live' ? 'var(--viz-surface)' : 'var(--viz-neutral)'} stroke={stroke} strokeWidth={tone === 'live' ? 1.5 : 2} strokeDasharray={tone === 'live' ? undefined : '4 2'} />
      <text x={x + w / 2} y={y + h / 2 + 4} textAnchor="middle" fontSize={11} fill="var(--viz-ink)">
        {s.state === 'free' ? `@${slot}` : `#${s.key}`}
      </text>
    </g>
  );
}

export default function ReclamationSchemesRaceLab() {
  const [scheme, setScheme] = useState<Scheme>('ebr');
  const [scenario, setScenario] = useState('mid');
  const [events, setEvents] = useState<PauseEvent[]>(SCENARIOS.mid.events);
  const [t, setT] = useState(SCENARIOS.mid.showAt);
  const [playing, setPlaying] = useState(false);
  const [interval, setWriteInterval] = useState(DEFAULT_CONFIG.interval);
  const [scanTicks, setScanTicks] = useState(DEFAULT_CONFIG.scanTicksPerNode);
  const [hpThreshold, setHpThreshold] = useState(DEFAULT_CONFIG.hpThreshold);
  const [qsbr, setQsbr] = useState(false);
  const acc = useRef(0);
  const markerId = `rcl-arrow-${useId().replace(/[^a-zA-Z0-9]/g, '')}`;

  const cfg: Config = useMemo(() => ({ interval, scanTicksPerNode: scanTicks, hpThreshold, qsbr }), [interval, scanTicks, hpThreshold, qsbr]);
  const full = useMemo(
    () => ({
      ebr: simulate('ebr', cfg, events, MAX_T),
      hp: simulate('hp', cfg, events, MAX_T),
      rcu: simulate('rcu', cfg, events, MAX_T),
      none: simulate('none', cfg, events, MAX_T),
    }),
    [cfg, events],
  );
  const sim = useMemo(() => simulate(scheme, cfg, events, t), [scheme, cfg, events, t]);

  useEffect(() => {
    if (!playing) return;
    if (t >= MAX_T) setPlaying(false);
  }, [playing, t]);
  useTicker((dt) => {
    acc.current += dt;
    if (acc.current >= 110) {
      acc.current = 0;
      setT((x) => Math.min(MAX_T, x + 1));
    }
  }, playing);

  const choose = (v: string) => {
    const p = SCENARIOS[v] ?? SCENARIOS.mid;
    setScenario(v);
    setEvents(p.events);
    setT(p.showAt);
    setPlaying(false);
  };
  const togglePause = (idx: number) => {
    const at = Math.max(1, t);
    // Tick 0 is the initial state; a stall clicked there takes effect at tick 1, so show tick 1.
    if (t === 0) setT(1);
    const want = !sim.readers[idx].paused;
    setEvents((ev) => [...ev.filter((e) => !(e.t === at && e.reader === idx)), { t: at, reader: idx, paused: want, when: 'now' }]);
  };

  const block = blockers(sim);
  const groups = retiredGroups(sim);
  const shownGroups = groups.slice(0, MAX_GROUPS);
  const hiddenGroups = groups.slice(MAX_GROUPS);
  const live = liveOrder(sim);
  const un = unreclaimed(sim);
  const walkers = (slot: number) => sim.readers.filter((r) => r.phase === 'walk' && r.pos === slot);

  // Layout of retired groups: flow into rows of up to two groups.
  const groupW = (g: Group) => Math.max((g.label.length + 6) * 6.2, Math.min(g.nodes.length, MAX_GROUP_BOXES) * 44 + (g.nodes.length > MAX_GROUP_BOXES ? 44 : 0));
  const rowsY0 = 150;
  const placed: { g: Group; x: number; y: number }[] = [];
  let gx = 12;
  let gy = rowsY0 + 24;
  for (const g of shownGroups) {
    const w = groupW(g);
    if (gx + w > W - 8 && gx > 12) {
      gx = 12;
      gy += 74;
    }
    placed.push({ g, x: gx, y: gy });
    gx += w + 24;
  }
  const H = (placed.length ? gy + 74 : rowsY0 + 44) + (hiddenGroups.length ? 16 : 0);

  const ops = sim.readers.reduce((a, r) => a + r.ops, 0);
  const writes = sim.readers.reduce((a, r) => a + r.writes, 0);

  const note = (() => {
    const stalled = sim.readers.filter((r) => r.paused);
    switch (sim.scheme) {
      case 'ebr': {
        const pinnedStalled = stalled.filter((r) => r.pinned !== null).sort((a, b) => (a.pinned as number) - (b.pinned as number));
        const holder = pinnedStalled.find((r) => block.has(r.idx));
        const unpinned = stalled.filter((r) => r.pinned === null);
        return holder ? (
          <>
            <strong>Memory climbs while {holder.name} stays pinned.</strong> {holder.name} pinned epoch {holder.pinned} and stalled, so the global epoch cannot pass {(holder.pinned ?? 0) + 1}; nothing retired in epoch {holder.pinned} or later can be freed. {un} unlinked nodes are waiting, one more every {cfg.interval} tick{cfg.interval === 1 ? '' : 's'}. The scheme cannot tell which of them {holder.name} could still reach, so it keeps them all.
          </>
        ) : pinnedStalled.length ? (
          <>
            <strong>{names(sim, pinnedStalled.map((r) => r.idx))} stalled while pinned in epoch {pinnedStalled[0].pinned}, the current one.</strong> The global epoch can still advance once; after that the stall holds it back and memory starts to climb. {un} nodes wait.
          </>
        ) : unpinned.length ? (
          <>
            <strong>{names(sim, unpinned.map((r) => r.idx))} stalled while unpinned</strong>, so {unpinned.length === 1 ? 'it does' : 'they do'} not hold the epoch back: {un} node{un === 1 ? '' : 's'} wait{un === 1 ? 's' : ''}, and each is freed two epoch advances after it was retired.
          </>
        ) : (
          <>
            <strong>{un} unlinked nodes wait.</strong> A node retired in epoch e is freed at e + 2, and the epoch advances only once every pinned reader has caught up — so each node effectively waits for the slowest operation in flight, R3's scan at {cfg.scanTicksPerNode} ticks per node.
          </>
        );
      }
      case 'hp': {
        const held = sim.readers.filter((r) => r.hp !== null && sim.slots[r.hp].state === 'retired');
        return held.length && held.some((r) => r.paused) ? (
          <>
            <strong>Only the published node survives the scans.</strong> {held.map((r) => `${r.name}'s hazard pointer names ${nodeLabel(sim, r.hp as number)}, which was unlinked at t = ${sim.slots[r.hp as number].retiredAt}`).join('; ')}. Each scan frees the rest of the retire list, so {un} node{un === 1 ? '' : 's'} wait{un === 1 ? 's' : ''} — never more than R − 1 plus one per hazard pointer.
          </>
        ) : (
          <>
            <strong>{un} node{un === 1 ? '' : 's'} on the retire list.</strong> Memory is bounded by R and the number of hazard pointers, whatever the readers do. The price is on the read side: {fmtNum(writes / Math.max(1, ops), 1)} hazard-pointer publishes per operation, and {sim.readers.map((r) => r.restarts).reduce((a, b) => a + b, 0)} restarts so far when validation found the previous node unlinked.
          </>
        );
      }
      case 'rcu': {
        const waitingStalled = sim.gp ? stalled.filter((r) => sim.gp?.wait.has(r.idx)) : [];
        const insideStalled = stalled.filter((r) => r.inCS && !waitingStalled.includes(r));
        const are = (rs: ReaderState[]) => (rs.length === 1 ? 'is' : 'are');
        return sim.gp && waitingStalled.length ? (
          <>
            <strong>The grace period cannot end.</strong> Grace period {sim.gp.id} waits for {names(sim, sim.gp.wait)}, and {names(sim, waitingStalled.map((r) => r.idx))} {are(waitingStalled)} stalled {sim.cfg.qsbr ? (waitingStalled.every((r) => r.phase === 'idle') ? 'between operations without reporting a quiescent state' : 'without reporting a quiescent state') : 'inside a read-side critical section'}. {un} nodes wait behind it.
          </>
        ) : insideStalled.length && !sim.cfg.qsbr ? (
          <>
            <strong>{names(sim, insideStalled.map((r) => r.idx))} {are(insideStalled)} stalled inside a read-side critical section.</strong> The next grace period to start must wait for {insideStalled.length === 1 ? 'it' : 'them'}; {un} nodes wait.
          </>
        ) : stalled.length && !sim.cfg.qsbr ? (
          <>
            <strong>A reader stalled outside a read-side critical section blocks nothing.</strong> Grace periods keep completing; {un} nodes wait. Tick QSBR to see the same stall pin memory.
          </>
        ) : (
          <>
            <strong>{un} nodes wait for a grace period.</strong> {sim.cfg.qsbr ? 'With QSBR every registered reader must report a quiescent state, so each grace period lasts at least one full R3 scan.' : 'A grace period waits only for readers already inside a read-side critical section when it starts; readers that enter later are not waited for.'}
          </>
        );
      }
      default:
        return (
          <>
            <strong>{sim.unsafeReads} unsafe reads so far: {sim.uafReads} use-after-free, {sim.abaReads} reuse (ABA).</strong> With no reclamation scheme a reader can follow a pointer into memory that was freed after it read the pointer, or that was already reused for a different node. Nothing crashes in the lab; in a real process this is silent corruption or a segfault.
          </>
        );
    }
  })();

  // chart
  const CH = 150;
  const cx0 = 44;
  const cw = W - cx0 - 96;
  const series = [
    { key: 'ebr' as const, label: 'Epochs', color: 'var(--viz-1)', dash: undefined },
    { key: 'hp' as const, label: 'Hazard pointers', color: 'var(--viz-2)', dash: undefined },
    { key: 'rcu' as const, label: qsbr ? 'RCU (QSBR)' : 'RCU', color: 'var(--viz-3)', dash: '6 3' },
  ];
  const yMax = Math.max(10, Math.ceil(Math.max(...series.map((s) => Math.max(...full[s.key].hist.un))) / 10) * 10);
  const X = (tt: number) => cx0 + (tt / MAX_T) * cw;
  const Y = (v: number) => 12 + (1 - v / yMax) * (CH - 40);
  const pathFor = (arr: number[]) => {
    let d = '';
    const step = 1;
    for (let i = 0; i <= t; i += step) d += `${i === 0 ? 'M' : 'L'}${X(i).toFixed(1)},${Y(arr[i]).toFixed(1)}`;
    return d;
  };
  const endLabels = series
    .map((s) => ({ ...s, v: full[s.key].hist.un[t], y: Y(full[s.key].hist.un[t]) }))
    .sort((a, b) => a.y - b.y);
  for (let i = 1; i < endLabels.length; i++) if (endLabels[i].y - endLabels[i - 1].y < 12) endLabels[i].y = endLabels[i - 1].y + 12;
  // Keep the direct labels inside the plot: if the lowest one would sit on the tick labels, shift the stack up.
  const overflow = endLabels.length ? endLabels[endLabels.length - 1].y - Y(0) : 0;
  if (overflow > 0) for (const l of endLabels) l.y -= overflow;

  const perOp = (k: Scheme) => {
    const h = full[k].hist;
    return h.ops[t] ? h.writes[t] / h.ops[t] : 0;
  };

  return (
    <VizPanel
      title="Who may free an unlinked node?"
      subtitle="A writer keeps unlinking nodes from a latch-free list while three readers walk it. Pick a reclamation scheme, stall a reader, and watch which unlinked nodes can be freed and which must wait."
      controls={
        <>
          <Segmented label="Reclamation scheme" value={scheme} onChange={(v) => setScheme(v)} options={[
            { value: 'ebr', label: 'Epochs' },
            { value: 'hp', label: 'Hazard pointers' },
            { value: 'rcu', label: 'RCU' },
            { value: 'none', label: 'Free at once (unsafe)' },
          ]} />
          <Choice label="Scenario" value={scenario} onChange={choose} options={Object.entries(SCENARIOS).map(([value, s]) => ({ value, label: s.label }))} />
          <Slider label="Writer unlinks a node every" min={1} max={8} value={interval} onChange={setWriteInterval} format={(n) => `${n} tick${n === 1 ? '' : 's'}`} />
          <Slider label="R3 scan speed" min={1} max={8} value={scanTicks} onChange={setScanTicks} format={(n) => `${n} tick${n === 1 ? '' : 's'} per node`} />
          {scheme === 'hp' ? <Slider label="Scan threshold R" min={1} max={12} value={hpThreshold} onChange={setHpThreshold} /> : null}
          {scheme === 'rcu' ? <Check label="QSBR flavor" checked={qsbr} onChange={setQsbr} /> : null}
        </>
      }
      legend={<Legend items={series.map((s) => ({ label: `${s.label} (chart)`, color: s.color, shape: 'line' as const }))} />}
      stats={
        <Stats
          items={[
            { label: 'Tick', value: fmtNum(t) },
            { label: 'Unlinked, not freed', value: fmtNum(un), hint: 'Nodes the writer has unlinked whose memory the scheme may not free yet.' },
            { label: 'Peak so far', value: fmtNum(sim.peak) },
            { label: 'Freed', value: fmtNum(sim.freed) },
            sim.scheme === 'ebr'
              ? { label: 'Global epoch', value: fmtNum(sim.epoch) }
              : sim.scheme === 'hp'
                ? { label: 'Reader restarts', value: fmtNum(sim.readers.reduce((a, r) => a + r.restarts, 0)), hint: 'Validation found the node a reader came from unlinked, so it restarted from the head.' }
                : sim.scheme === 'rcu'
                  ? { label: 'Grace periods done', value: fmtNum(sim.gpDone) }
                  : { label: 'Unsafe reads', value: fmtNum(sim.unsafeReads) },
            {
              label: 'Reader bookkeeping writes per op',
              value: ops ? (writes / ops).toFixed(1) : '—',
              hint: 'Model count of shared-memory stores by readers: pin + unpin (epochs), one publish per node visited (hazard pointers), lock + unlock (RCU) or one quiescent-state report (QSBR).',
            },
          ]}
        />
      }
      note={<Note>{note}</Note>}
      table={
        <table className="viz-table">
          <thead>
            <tr>
              <th>Scheme, at tick {t}</th>
              <th>Unlinked, not freed</th>
              <th>Peak</th>
              <th>Freed</th>
              <th>Reader restarts</th>
              <th>Bookkeeping writes per op</th>
              <th>Unsafe reads</th>
            </tr>
          </thead>
          <tbody>
            {(['ebr', 'hp', 'rcu', 'none'] as const).map((k) => (
              <tr key={k}>
                <td>{k === 'ebr' ? 'Epochs' : k === 'hp' ? `Hazard pointers (R = ${hpThreshold})` : k === 'rcu' ? (qsbr ? 'RCU, QSBR' : 'RCU') : 'Free at once'}</td>
                <td>{fmtNum(full[k].hist.un[t])}</td>
                <td>{fmtNum(Math.max(...full[k].hist.un.slice(0, t + 1)))}</td>
                <td>{fmtNum(full[k].hist.freed[t])}</td>
                <td>{fmtNum(full[k].hist.restarts[t])}</td>
                <td>{perOp(k).toFixed(1)}</td>
                <td>{fmtNum(full[k].hist.unsafe[t])}</td>
              </tr>
            ))}
          </tbody>
        </table>
      }
    >
      <div className="viz-controls">
        <Button primary onClick={() => setPlaying((p) => !p)} disabled={t >= MAX_T && !playing}>
          {playing ? 'Pause' : 'Play'}
        </Button>
        <Button onClick={() => setT((x) => Math.min(MAX_T, x + 1))} disabled={t >= MAX_T}>
          Step
        </Button>
        <Button onClick={() => setT((x) => Math.min(MAX_T, x + 10))} disabled={t >= MAX_T}>
          +10 ticks
        </Button>
        <Button
          onClick={() => {
            setPlaying(false);
            setT(0);
            setEvents(SCENARIOS[scenario]?.events ?? []);
          }}
        >
          Reset
        </Button>
        <Slider label="Time" min={0} max={MAX_T} value={t} onChange={(v) => { setPlaying(false); setT(v); }} format={(n) => `t = ${n}`} />
      </div>
      <TooltipHost>
        <svg viewBox={`0 0 ${W} ${H}`} width={W} height={H} style={{ minWidth: 560 }} role="img" aria-label={`${scheme === 'ebr' ? 'Epoch-based reclamation' : scheme === 'hp' ? 'Hazard pointers' : scheme === 'rcu' ? 'RCU' : 'No reclamation scheme'} at tick ${t}: ${un} unlinked nodes not yet freed`}>
          <defs>
            <marker id={markerId} viewBox="0 0 8 8" refX={7} refY={4} markerWidth={6} markerHeight={6} orient="auto">
              <path d="M0,0 L8,4 L0,8 z" fill="var(--viz-ink-muted)" />
            </marker>
          </defs>
          <text x={8} y={16} fontSize={11} fill="var(--viz-ink-2)">
            {statusLine(sim)}
          </text>
          <rect x={6} y={36} width={40} height={NODE_H} rx={4} fill="none" stroke="var(--viz-ink-muted)" />
          <text x={26} y={54} textAnchor="middle" fontSize={11} fill="var(--viz-ink-2)">
            head
          </text>
          {live.map((slot, i) => {
            const x = LIST_X + i * STEP_X;
            return (
              <g key={`${slot}-${sim.slots[slot].key}`}>
                <line x1={i === 0 ? 46 : x - STEP_X + NODE_W} y1={50} x2={x - 1} y2={50} stroke="var(--viz-ink-muted)" markerEnd={`url(#${markerId})`} />
                <NodeBox sim={sim} slot={slot} x={x} y={36} tone="live" />
                <ReaderBadges sim={sim} readers={walkers(slot)} cx={x + NODE_W / 2} y={70} block={block} />
              </g>
            );
          })}
          <text x={8} y={112} fontSize={11} fill="var(--viz-ink-2)">
            Between operations: {sim.readers.filter((r) => r.phase === 'idle').map((r) => `${r.name}${r.paused ? ' (stalled)' : ''}`).join(', ') || 'nobody'}
          </text>
          <line x1={6} x2={W - 6} y1={rowsY0 - 18} y2={rowsY0 - 18} stroke="var(--viz-grid)" />
          <text x={8} y={rowsY0} fontSize={12} fill="var(--viz-ink)">
            {sim.scheme === 'none' ? `Unlinked nodes are freed immediately (${sim.freed} so far)` : `Unlinked, waiting to be freed: ${un}`}
          </text>
          {placed.map(({ g, x, y }) => {
            const boxes = g.nodes.slice(0, MAX_GROUP_BOXES);
            return (
              <g key={g.label}>
                <text x={x} y={y + 4} fontSize={11} fill="var(--viz-ink-2)">
                  {g.label} ({g.nodes.length})
                </text>
                {boxes.map((slot, i) => (
                  <g key={slot}>
                    <NodeBox sim={sim} slot={slot} x={x + i * 44} y={y + 12} w={40} h={24} tone={g.tone} />
                    <ReaderBadges sim={sim} readers={walkers(slot)} cx={x + i * 44 + 20} y={y + 40} block={block} />
                  </g>
                ))}
                {g.nodes.length > MAX_GROUP_BOXES ? (
                  <text x={x + boxes.length * 44 + 2} y={y + 29} fontSize={11} fill="var(--viz-ink-2)">
                    +{g.nodes.length - MAX_GROUP_BOXES}
                  </text>
                ) : null}
              </g>
            );
          })}
          {hiddenGroups.length ? (
            <text x={12} y={H - 6} fontSize={11} fill="var(--viz-ink-2)">
              +{hiddenGroups.reduce((a, g) => a + g.nodes.length, 0)} more nodes in {hiddenGroups.length} later group{hiddenGroups.length === 1 ? '' : 's'}
            </text>
          ) : null}
        </svg>
      </TooltipHost>
      <Legend
        items={[
          { label: 'Live node (solid outline)', color: 'var(--viz-ink-2)' },
          { label: 'Unlinked, not yet freed (dashed)', color: 'var(--viz-stale)' },
          { label: 'Reader holding back reclamation', color: 'var(--viz-warning)' },
          { label: 'Read of freed or reused memory', color: 'var(--viz-critical)' },
        ]}
      />

      <div style={{ overflowX: 'auto' }}>
        <table className="viz-table" style={{ marginTop: 4 }}>
          <thead>
            <tr>
              <th>Reader</th>
              <th>Where</th>
              <th>{scheme === 'ebr' ? 'Epoch pin' : scheme === 'hp' ? 'Hazard pointer' : scheme === 'rcu' ? (qsbr ? 'Quiescent state' : 'Read-side section') : 'Protection'}</th>
              <th>Ops done</th>
              <th>Stall</th>
            </tr>
          </thead>
          <tbody>
            {sim.readers.map((r) => (
              <tr key={r.idx}>
                <td>
                  {r.name}, {r.kind === 'scan' ? 'range scan' : 'point lookups'}
                </td>
                <td>
                  {r.phase === 'idle' ? 'between operations' : `at ${nodeLabel(sim, r.pos as number)}${sim.slots[r.pos as number].state === 'retired' ? ' (unlinked)' : sim.slots[r.pos as number].state === 'free' ? ' (freed!)' : sim.slots[r.pos as number].gen !== r.posGen ? ' (address reused!)' : ''}`}
                  {r.paused ? ', stalled' : ''}
                </td>
                <td>
                  {scheme === 'ebr'
                    ? r.pinned === null
                      ? 'not pinned'
                      : `pinned in ${r.pinned}${block.has(r.idx) ? ' — holds the epoch back' : ''}`
                    : scheme === 'hp'
                      ? `${r.hp === null ? '—' : `${nodeLabel(sim, r.hp)}${block.has(r.idx) ? ' — keeps an unlinked node alive' : ''}`} · ${r.restarts} restart${r.restarts === 1 ? '' : 's'}`
                      : scheme === 'rcu'
                        ? block.has(r.idx)
                          ? qsbr
                            ? 'owes a quiescent state'
                            : 'inside, grace period waits'
                          : qsbr
                            ? 'reported'
                            : r.inCS
                              ? 'inside'
                              : 'outside'
                        : 'none'}
                </td>
                <td>{fmtNum(r.ops)}</td>
                <td>
                  <Check label={`Stall ${r.name}`} checked={r.paused} onChange={() => togglePause(r.idx)} />
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {sim.log.length ? (
        <ol style={{ margin: '0.5rem 0 0', paddingLeft: '1.2rem', fontSize: '0.8rem', display: 'grid', gap: 4 }} aria-label={`What happened at tick ${t}`}>
          {sim.log.map((l, i) => (
            <li key={i} style={{ color: 'var(--viz-ink)' }}>
              {l}
            </li>
          ))}
        </ol>
      ) : null}

      <svg viewBox={`0 0 ${W} ${CH}`} width={W} height={CH} role="img" aria-label={`Unlinked but unfreed nodes over time for epochs, hazard pointers and RCU, up to tick ${t}`} style={{ marginTop: 12, minWidth: 560 }}>
        {[0, 0.5, 1].map((f) => (
          <g key={f}>
            <line x1={cx0} x2={cx0 + cw} y1={Y(yMax * f)} y2={Y(yMax * f)} stroke="var(--viz-grid)" />
            <text x={cx0 - 6} y={Y(yMax * f) + 4} textAnchor="end" fontSize={10} fill="var(--viz-ink-muted)">
              {fmtNum(yMax * f)}
            </text>
          </g>
        ))}
        {[0, 100, 200, 300, 400, 500, 600].map((tt) => (
          <text key={tt} x={X(tt)} y={CH - 12} textAnchor="middle" fontSize={10} fill="var(--viz-ink-muted)">
            {tt}
          </text>
        ))}
        <text x={cx0} y={CH - 1} fontSize={10} fill="var(--viz-ink-2)">
          tick — unlinked nodes not yet freed, same writer and readers under each scheme
        </text>
        <line x1={X(t)} x2={X(t)} y1={8} y2={CH - 26} stroke="var(--viz-ink-muted)" strokeDasharray="3 3" />
        {series.map((s) => (
          <path key={s.key} d={pathFor(full[s.key].hist.un)} fill="none" stroke={s.color} strokeWidth={s.key === scheme ? 2.5 : 1.6} strokeDasharray={s.dash} />
        ))}
        {endLabels.map((s) => (
          <text key={s.key} x={X(t) + 6} y={s.y + 4} fontSize={10} fill="var(--viz-ink)">
            {s.label} {s.v}
          </text>
        ))}
      </svg>
    </VizPanel>
  );
}
