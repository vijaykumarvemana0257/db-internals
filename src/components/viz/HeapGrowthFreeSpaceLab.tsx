import { useState } from 'react';
import {
  VizPanel,
  Segmented,
  Choice,
  Slider,
  Button,
  Legend,
  Stats,
  Note,
  TooltipHost,
  useTip,
  makeRng,
  fmtBytes,
  fmtNum,
  useSize,
} from './Viz';

/**
 * Heap growth and free space management, run as a simulation.
 *
 * PostgreSQL lane models, at page granularity:
 *   - the heap fork: live bytes / dead bytes / free bytes per 8 KB page
 *   - the FSM fork: one byte per page holding free space quantised to BLCKSZ/256
 *     (32-byte steps), with upper-level nodes holding the max of their children
 *   - the FSM descent that picks the target page, including rd_targblock (the
 *     per-backend cached target that skips the lookup entirely) and fp_next_slot
 *     (the round-robin start that spreads concurrent inserters)
 *   - relation extension at the tail, in batches scaled by the number of waiters
 *   - the VM fork: all-visible / all-frozen bits, set by VACUUM, cleared by any write
 *   - VACUUM: reclaims dead space, repairs FSM bytes, and truncates ONLY the
 *     trailing run of empty pages (REL_TRUNCATE_FRACTION), so interior holes stay
 *
 * InnoDB lane models the same workload against tablespace geometry: fragment pages
 * first, then whole extents, XDES states (FREE / FREE_FRAG / FULL_FRAG / owned by a
 * file segment), page splits and merges at MERGE_THRESHOLD, freed pages returning to
 * the segment rather than to the filesystem, and OPTIMIZE TABLE's rebuild.
 *
 * Two numbers are deliberately scaled down so a legible strip of pages still shows
 * the mechanism; both are called out in the figure's note and the table.
 */

/* ------------------------------------------------------------------ shared */

const OPS_PER_BATCH = 250;
const MAX_PAGES = 64;

/* ---------------------------------------------------------------- postgres */

const BLCKSZ = 8192;
const PG_USABLE = BLCKSZ - 24; // page header; a heap page has no special space
const LP_BYTES = 4; // ItemIdData
const FSM_STEP = BLCKSZ / 256; // 32 bytes: one FSM category step
const FSM_GROUP = 8; // leaves under one internal node in this scaled tree
const TRUNC_FRACTION = 16; // REL_TRUNCATE_FRACTION

type HeapPage = {
  live: number; // live tuples
  dead: number; // dead tuples, space not reusable until VACUUM
  allVisible: boolean;
  allFrozen: boolean;
};

type PgSim = {
  pages: HeapPage[];
  unit: number; // bytes one row costs on a page, including its line pointer
  target: number; // rd_targblock, -1 when invalid
  nextSlot: number; // fp_next_slot
  targetHits: number;
  fsmLookups: number;
  fsmMisses: number; // searches that found nothing and forced an extension
  extensions: number;
  extendedPages: number;
  waiterStalls: number;
  vmClears: number;
  inserts: number;
  deletes: number;
  vacuums: number;
  truncatedPages: number;
  reclaimed: number; // bytes freed by VACUUM
  batch: number;
  path: { group: number; leaf: number; start: number } | null;
  msg: string;
};

function rowUnit(rowBytes: number) {
  // 23-byte HeapTupleHeader + data, MAXALIGNed to 8, plus the line pointer.
  const t = 23 + rowBytes;
  return Math.ceil(t / 8) * 8 + LP_BYTES;
}

const freeBytes = (p: HeapPage, unit: number) => PG_USABLE - (p.live + p.dead) * unit;
const fsmCat = (free: number) => Math.max(0, Math.min(255, Math.floor(free / FSM_STEP)));
const needCat = (unit: number) => Math.min(255, Math.ceil(unit / FSM_STEP));

function freshPg(rowBytes: number): PgSim {
  return {
    pages: [],
    unit: rowUnit(rowBytes),
    target: -1,
    nextSlot: 0,
    targetHits: 0,
    fsmLookups: 0,
    fsmMisses: 0,
    extensions: 0,
    extendedPages: 0,
    waiterStalls: 0,
    vmClears: 0,
    inserts: 0,
    deletes: 0,
    vacuums: 0,
    truncatedPages: 0,
    reclaimed: 0,
    batch: 0,
    path: null,
    msg: 'Empty relation: no heap pages, no FSM fork, no VM fork. Run a batch.',
  };
}

function fsmLeaves(s: PgSim) {
  return s.pages.map((p) => fsmCat(freeBytes(p, s.unit)));
}

function fsmGroups(leaves: number[]) {
  const g: number[] = [];
  for (let i = 0; i < leaves.length; i += FSM_GROUP) {
    g.push(Math.max(0, ...leaves.slice(i, i + FSM_GROUP)));
  }
  return g;
}

/** Descend the FSM from the root, starting the scan at fp_next_slot. */
function fsmSearch(s: PgSim): { group: number; leaf: number; start: number } | null {
  const need = needCat(s.unit);
  const leaves = fsmLeaves(s);
  if (leaves.length === 0) return null;
  const groups = fsmGroups(leaves);
  if (Math.max(0, ...groups) < need) return null; // the root says no page qualifies
  const start = s.nextSlot % leaves.length;
  const startGroup = Math.floor(start / FSM_GROUP);
  for (let gi = 0; gi < groups.length; gi++) {
    const g = (startGroup + gi) % groups.length;
    if (groups[g] < need) continue;
    const base = g * FSM_GROUP;
    const n = Math.min(FSM_GROUP, leaves.length - base);
    const from = gi === 0 ? start - base : 0;
    for (let li = 0; li < n; li++) {
      const leaf = base + ((Math.max(0, from) + li) % n);
      if (leaves[leaf] >= need) return { group: g, leaf, start };
    }
  }
  return null;
}

function placePg(s: PgSim, i: number) {
  const p = s.pages[i];
  p.live += 1;
  if (p.allVisible) s.vmClears += 1;
  p.allVisible = false;
  p.allFrozen = false;
  s.inserts += 1;
  s.target = i;
}

function insertPg(s: PgSim, extendBy: number, waiters: number): boolean {
  // 1. rd_targblock: the page this backend last inserted into, no FSM lookup at all.
  if (s.target >= 0 && s.target < s.pages.length && freeBytes(s.pages[s.target], s.unit) >= s.unit) {
    s.targetHits += 1;
    placePg(s, s.target);
    s.path = null;
    return true;
  }
  // 2. Ask the FSM.
  s.fsmLookups += 1;
  const hit = fsmSearch(s);
  if (hit) {
    s.path = hit;
    s.nextSlot = (hit.leaf + 1) % Math.max(1, s.pages.length);
    placePg(s, hit.leaf);
    return true;
  }
  // 3. Nothing fits: extend the relation under the extension lock.
  s.fsmMisses += 1;
  s.path = null;
  if (s.pages.length >= MAX_PAGES) {
    s.msg = `Relation capped at ${MAX_PAGES} pages for this simulation — VACUUM or reset.`;
    return false;
  }
  const add = Math.min(extendBy, MAX_PAGES - s.pages.length);
  const first = s.pages.length;
  for (let k = 0; k < add; k++) {
    s.pages.push({ live: 0, dead: 0, allVisible: false, allFrozen: false });
  }
  s.extensions += 1;
  s.extendedPages += add;
  s.waiterStalls += waiters - 1;
  placePg(s, first);
  return true;
}

function deletePg(s: PgSim, i: number) {
  const p = s.pages[i];
  if (p.live <= 0) return;
  p.live -= 1;
  p.dead += 1;
  if (p.allVisible) s.vmClears += 1;
  p.allVisible = false;
  p.allFrozen = false;
  s.deletes += 1;
}

type Workload = 'insert' | 'update' | 'delprefix' | 'delsuffix' | 'delrandom';

function pickLive(s: PgSim, rng: () => number, mode: 'low' | 'high' | 'rand'): number {
  const live = s.pages.map((p, i) => [i, p.live] as const).filter(([, n]) => n > 0);
  if (live.length === 0) return -1;
  if (mode === 'low') return live[0][0];
  if (mode === 'high') return live[live.length - 1][0];
  const total = live.reduce((a, [, n]) => a + n, 0);
  let r = rng() * total;
  for (const [i, n] of live) {
    r -= n;
    if (r <= 0) return i;
  }
  return live[live.length - 1][0];
}

function runPgBatch(prev: PgSim, workload: Workload, extendBy: number, waiters: number): PgSim {
  const s: PgSim = {
    ...prev,
    pages: prev.pages.map((p) => ({ ...p })),
    batch: prev.batch + 1,
  };
  s.msg = '';
  const rng = makeRng(7919 + s.batch * 131 + s.pages.length);
  let done = 0;
  for (let k = 0; k < OPS_PER_BATCH; k++) {
    if (workload === 'insert') {
      if (!insertPg(s, extendBy, waiters)) break;
    } else if (workload === 'update') {
      const i = pickLive(s, rng, 'rand');
      if (i < 0) {
        if (!insertPg(s, extendBy, waiters)) break;
      } else {
        deletePg(s, i);
        if (!insertPg(s, extendBy, waiters)) break;
      }
    } else {
      const mode = workload === 'delprefix' ? 'low' : workload === 'delsuffix' ? 'high' : 'rand';
      const i = pickLive(s, rng, mode);
      if (i < 0) {
        s.msg = 'No live rows left to delete.';
        break;
      }
      deletePg(s, i);
    }
    done++;
  }
  const label: Record<Workload, string> = {
    insert: 'inserts',
    update: 'updates (delete old version + insert new one)',
    delprefix: 'deletes from the lowest-numbered pages',
    delsuffix: 'deletes from the highest-numbered pages',
    delrandom: 'scattered deletes',
  };
  if (s.msg === '') {
    s.msg =
      `Ran ${fmtNum(done)} ${label[workload]}. ` +
      (workload === 'insert' || workload === 'update'
        ? `${fmtNum(s.targetHits)} landed on the cached rd_targblock with no FSM lookup; ` +
          `${fmtNum(s.fsmLookups)} needed a descent from the FSM root.`
        : 'Dead tuples still occupy their bytes — nothing is reusable until VACUUM.');
  }
  return s;
}

function vacuumPg(prev: PgSim): PgSim {
  const s: PgSim = { ...prev, pages: prev.pages.map((p) => ({ ...p })), vacuums: prev.vacuums + 1 };
  let reclaimed = 0;
  for (const p of s.pages) {
    reclaimed += p.dead * s.unit;
    p.dead = 0;
    // A page nobody touched since the last VACUUM is old enough to freeze here;
    // real freezing is driven by xid age against vacuum_freeze_min_age.
    p.allFrozen = p.allVisible;
    p.allVisible = true;
  }
  s.reclaimed += reclaimed;

  let trailing = 0;
  for (let i = s.pages.length - 1; i >= 0; i--) {
    if (s.pages[i].live === 0 && s.pages[i].dead === 0) trailing++;
    else break;
  }
  const threshold = Math.max(1, Math.floor(s.pages.length / TRUNC_FRACTION));
  let truncated = 0;
  if (trailing >= threshold) {
    truncated = trailing;
    s.pages.length -= trailing;
    s.truncatedPages += truncated;
  }
  if (s.target >= s.pages.length) s.target = -1;
  s.nextSlot = s.pages.length ? s.nextSlot % s.pages.length : 0;
  s.path = null;

  const interior = s.pages.filter((p) => p.live === 0).length;
  s.msg =
    `VACUUM: reclaimed ${fmtBytes(reclaimed)} of dead tuples and rewrote every FSM byte. ` +
    (truncated
      ? `Truncated ${truncated} trailing empty page${truncated === 1 ? '' : 's'} — the file actually shrank.`
      : trailing
        ? `${trailing} trailing empty page${trailing === 1 ? '' : 's'} is below the 1/16-of-relation truncation threshold (${threshold} here), so the file kept its size.`
        : 'No trailing empty pages, so nothing could be returned to the filesystem.') +
    (interior ? ` ${interior} empty page${interior === 1 ? '' : 's'} remain inside the file as reusable bloat.` : '');
  return s;
}

/* ------------------------------------------------------------------ innodb */

const INNO_PAGE = 16384;
const INNO_USABLE = INNO_PAGE - 128; // page header + trailer + directory, approximated
const EXTENT_PAGES = 8; // scaled: a real extent is 64 pages / 1 MB at 16 KB
const FRAG_SLOTS = 4; // scaled: FSEG_FRAG_ARR_N_SLOTS is 32
const MERGE_THRESHOLD = 0.5;
const INNO_MAX_PAGES = 64;

type InnoKind = 'meta' | 'root' | 'leaf' | 'free';
type InnoPage = { kind: InnoKind; rows: number; frag: boolean };

type InnoSim = {
  pages: InnoPage[]; // length is always a multiple of EXTENT_PAGES: the file
  extentOwner: ('none' | 'frag' | 'seg')[];
  unit: number;
  fragUsed: number;
  splits: number;
  merges: number;
  extentsAdded: number;
  fragAllocs: number;
  extentAllocs: number;
  freedPages: number;
  rebuilds: number;
  inserts: number;
  deletes: number;
  batch: number;
  lastPage: number;
  msg: string;
};

const innoUnit = (rowBytes: number) => rowBytes + 5 + 2; // record header + directory slot share
const innoCap = (unit: number) => Math.max(2, Math.floor(INNO_USABLE / unit));

function freshInno(rowBytes: number): InnoSim {
  const s: InnoSim = {
    pages: [],
    extentOwner: [],
    unit: innoUnit(rowBytes),
    fragUsed: 0,
    splits: 0,
    merges: 0,
    extentsAdded: 0,
    fragAllocs: 0,
    extentAllocs: 0,
    freedPages: 0,
    rebuilds: 0,
    inserts: 0,
    deletes: 0,
    batch: 0,
    lastPage: -1,
    msg: 'Fresh .ibd: one extent, FSP_HDR + ibuf bitmap + inode pages, and an empty clustered index root.',
  };
  addExtent(s, 'frag');
  s.pages[0] = { kind: 'meta', rows: 0, frag: true };
  s.pages[1] = { kind: 'meta', rows: 0, frag: true };
  s.pages[2] = { kind: 'meta', rows: 0, frag: true };
  s.pages[3] = { kind: 'root', rows: 0, frag: true }; // fragment page of the internal segment
  return s;
}

function addExtent(s: InnoSim, owner: 'none' | 'frag' | 'seg'): number {
  const base = s.pages.length;
  for (let i = 0; i < EXTENT_PAGES; i++) s.pages.push({ kind: 'free', rows: 0, frag: false });
  s.extentOwner.push(owner);
  s.extentsAdded += 1;
  return base;
}

/** Allocate one leaf page: fragment pages first, whole extents afterwards. */
function allocLeaf(s: InnoSim): number {
  if (s.fragUsed < FRAG_SLOTS) {
    for (let i = 0; i < s.pages.length; i++) {
      if (s.pages[i].kind === 'free' && s.extentOwner[Math.floor(i / EXTENT_PAGES)] === 'frag') {
        s.pages[i] = { kind: 'leaf', rows: 0, frag: true };
        s.fragUsed += 1;
        s.fragAllocs += 1;
        return i;
      }
    }
    if (s.pages.length < INNO_MAX_PAGES) {
      const base = addExtent(s, 'frag');
      s.pages[base] = { kind: 'leaf', rows: 0, frag: true };
      s.fragUsed += 1;
      s.fragAllocs += 1;
      return base;
    }
  }
  // A free page already owned by the leaf segment.
  for (let i = 0; i < s.pages.length; i++) {
    if (s.pages[i].kind === 'free' && s.extentOwner[Math.floor(i / EXTENT_PAGES)] === 'seg') {
      s.pages[i] = { kind: 'leaf', rows: 0, frag: false };
      return i;
    }
  }
  // Otherwise take a whole extent for the segment, extending the file if needed.
  for (let e = 0; e < s.extentOwner.length; e++) {
    if (s.extentOwner[e] === 'none') {
      s.extentOwner[e] = 'seg';
      s.extentAllocs += 1;
      const base = e * EXTENT_PAGES;
      s.pages[base] = { kind: 'leaf', rows: 0, frag: false };
      return base;
    }
  }
  if (s.pages.length >= INNO_MAX_PAGES) return -1;
  const base = addExtent(s, 'seg');
  s.extentAllocs += 1;
  s.pages[base] = { kind: 'leaf', rows: 0, frag: false };
  return base;
}

const leafIdx = (s: InnoSim) => s.pages.map((p, i) => (p.kind === 'leaf' ? i : -1)).filter((i) => i >= 0);

function insertInno(s: InnoSim, rng: () => number, sequential: boolean): boolean {
  const cap = innoCap(s.unit);
  const leaves = leafIdx(s);
  if (leaves.length === 0) {
    const i = allocLeaf(s);
    if (i < 0) return false;
    s.pages[i].rows = 1;
    s.inserts += 1;
    s.lastPage = i;
    return true;
  }
  const target = sequential ? leaves[leaves.length - 1] : leaves[Math.floor(rng() * leaves.length) % leaves.length];
  if (s.pages[target].rows < cap) {
    s.pages[target].rows += 1;
    s.inserts += 1;
    s.lastPage = target;
    return true;
  }
  const fresh = allocLeaf(s);
  if (fresh < 0) {
    s.msg = `Tablespace capped at ${INNO_MAX_PAGES} pages for this simulation — rebuild or reset.`;
    return false;
  }
  if (sequential) {
    // Right-edge insert: InnoDB leaves the full page alone and starts a new one.
    s.pages[fresh].rows = 1;
  } else {
    const move = Math.floor(s.pages[target].rows / 2);
    s.pages[target].rows -= move;
    s.pages[fresh].rows = move + 1;
    s.splits += 1;
  }
  s.inserts += 1;
  s.lastPage = fresh;
  return true;
}

function deleteInno(s: InnoSim, i: number) {
  if (s.pages[i].rows <= 0) return;
  s.pages[i].rows -= 1; // purge reclaims the record's bytes inside the page
  s.deletes += 1;
  const cap = innoCap(s.unit);
  if (s.pages[i].rows / cap >= MERGE_THRESHOLD) return;
  const leaves = leafIdx(s);
  const at = leaves.indexOf(i);
  const sib = at >= 0 && at + 1 < leaves.length ? leaves[at + 1] : at > 0 ? leaves[at - 1] : -1;
  if (sib < 0) return;
  if (s.pages[i].rows + s.pages[sib].rows > cap) return;
  const keep = Math.min(i, sib);
  const drop = Math.max(i, sib);
  s.pages[keep].rows = s.pages[i].rows + s.pages[sib].rows;
  const wasFrag = s.pages[drop].frag;
  s.pages[drop] = { kind: 'free', rows: 0, frag: wasFrag };
  if (wasFrag) s.fragUsed = Math.max(0, s.fragUsed - 1);
  s.freedPages += 1;
  s.merges += 1;
}

function runInnoBatch(prev: InnoSim, workload: Workload): InnoSim {
  const s: InnoSim = {
    ...prev,
    pages: prev.pages.map((p) => ({ ...p })),
    extentOwner: [...prev.extentOwner],
    batch: prev.batch + 1,
  };
  s.msg = '';
  const rng = makeRng(104729 + s.batch * 977 + s.pages.length);
  let done = 0;
  for (let k = 0; k < OPS_PER_BATCH; k++) {
    if (workload === 'insert') {
      if (!insertInno(s, rng, true)) break;
    } else if (workload === 'update') {
      if (!insertInno(s, rng, false)) break;
      const l = leafIdx(s).filter((i) => s.pages[i].rows > 0);
      if (l.length) deleteInno(s, l[Math.floor(rng() * l.length) % l.length]);
    } else {
      const l = leafIdx(s).filter((i) => s.pages[i].rows > 0);
      if (l.length === 0) {
        s.msg = 'No rows left to delete.';
        break;
      }
      const i =
        workload === 'delprefix' ? l[0] : workload === 'delsuffix' ? l[l.length - 1] : l[Math.floor(rng() * l.length) % l.length];
      deleteInno(s, i);
    }
    done++;
  }
  if (s.msg === '') {
    s.msg =
      `Ran ${fmtNum(done)} operations against the clustered index. ` +
      `${s.fragAllocs} page${s.fragAllocs === 1 ? '' : 's'} came individually from fragment extents, ` +
      `${s.extentAllocs} whole extent${s.extentAllocs === 1 ? '' : 's'} were handed to the leaf segment. ` +
      (s.freedPages
        ? `${s.freedPages} page${s.freedPages === 1 ? '' : 's'} freed by merges went back to the segment, not to the filesystem.`
        : 'The file only ever grows.');
  }
  return s;
}

function rebuildInno(prev: InnoSim): InnoSim {
  const rows = prev.pages.reduce((a, p) => a + (p.kind === 'leaf' ? p.rows : 0), 0);
  const before = prev.pages.length;
  const s = freshInno(0);
  s.unit = prev.unit;
  s.rebuilds = prev.rebuilds + 1;
  s.inserts = prev.inserts;
  s.deletes = prev.deletes;
  s.batch = prev.batch;
  const cap = innoCap(s.unit);
  let left = rows;
  while (left > 0) {
    const i = allocLeaf(s);
    if (i < 0) break;
    const put = Math.min(cap, left);
    s.pages[i].rows = put;
    left -= put;
  }
  s.msg =
    `OPTIMIZE TABLE rebuilt the table into a new .ibd and renamed it: ${fmtNum(rows)} rows repacked into ` +
    `${s.pages.filter((p) => p.kind === 'leaf').length} full leaf pages, file ${before} → ${s.pages.length} pages. ` +
    'This is the only way an InnoDB tablespace gives space back — and it needs room for a second full copy.';
  return s;
}


/* ------------------------------------------------------------------- view */

const WORKLOADS: { value: Workload; label: string }[] = [
  { value: 'insert', label: 'Sequential insert' },
  { value: 'update', label: 'Random update' },
  { value: 'delprefix', label: 'Delete a prefix (oldest pages)' },
  { value: 'delsuffix', label: 'Delete a suffix (newest pages)' },
  { value: 'delrandom', label: 'Delete at random' },
];

const HEAT = [
  'var(--viz-seq-100)',
  'var(--viz-seq-250)',
  'var(--viz-seq-400)',
  'var(--viz-seq-550)',
  'var(--viz-seq-700)',
];
const catColor = (c: number) => HEAT[Math.min(4, Math.floor((c / 256) * 5))];

const CW = 16;
const GAP = 2;
const LEFT = 92;
const xAt = (i: number) => LEFT + i * (CW + GAP);
const widthFor = (n: number, avail: number) => Math.max(avail, LEFT + Math.max(n, 8) * (CW + GAP) + 12);

/** Lives inside <TooltipHost> so useTip() sees the provider. */
function HeapFigure({ sim, avail }: { sim: PgSim; avail: number }) {
  const tip = useTip();
  const ROOT_Y = 6;
  const GRP_Y = 30;
  const LEAF_Y = 54;
  const TILE_Y = 86;
  const TILE_H = 64;
  const VM_Y = TILE_Y + TILE_H + 8;
  const height = VM_Y + 34;
  const svgW = widthFor(sim.pages.length, avail);

  const leaves = fsmLeaves(sim);
  const groups = fsmGroups(leaves);
  const root = Math.max(0, ...groups);
  const need = needCat(sim.unit);
  const cap = Math.max(1, Math.floor(PG_USABLE / sim.unit));

  return (
    <svg
      width={svgW}
      height={height}
      role="img"
      aria-label="PostgreSQL heap pages with their free space map bytes and visibility map bits"
    >
      <text x={0} y={ROOT_Y + 13} fontSize={11} fill="var(--viz-ink-2)">
        FSM root
      </text>
      <text x={0} y={GRP_Y + 13} fontSize={11} fill="var(--viz-ink-2)">
        FSM upper
      </text>
      <text x={0} y={LEAF_Y + 12} fontSize={11} fill="var(--viz-ink-2)">
        FSM byte
      </text>
      <text x={0} y={TILE_Y + 16} fontSize={11} fill="var(--viz-ink-2)">
        Heap page
      </text>
      <text x={0} y={VM_Y + 10} fontSize={11} fill="var(--viz-ink-2)">
        VM bits
      </text>

      <g
        {...tip(
          <>
            <strong>FSM root page</strong>
            <br />
            Holds the maximum category of everything below it: {root} (≥ {root * FSM_STEP} B).
            <br />
            This insert needs category {need} ({sim.unit} B including the line pointer).
            <br />
            {root >= need
              ? 'The root says some page qualifies, so the descent continues.'
              : 'The root says no page qualifies — the backend extends the relation.'}
          </>,
        )}
      >
        <rect
          x={LEFT}
          y={ROOT_Y}
          width={Math.max(90, Math.min(170, Math.max(sim.pages.length, 8) * (CW + GAP) - 4))}
          height={18}
          rx={4}
          fill={sim.pages.length ? catColor(root) : 'var(--viz-neutral)'}
          stroke={sim.path ? 'var(--viz-ink)' : 'var(--viz-grid)'}
          strokeWidth={sim.path ? 2 : 1}
        />
        <text x={LEFT + 6} y={ROOT_Y + 13} fontSize={10} fill="var(--viz-ink)">
          max cat {root} · need {need}
        </text>
      </g>

      {groups.map((g, gi) => {
        const base = gi * FSM_GROUP;
        const n = Math.min(FSM_GROUP, sim.pages.length - base);
        const w = n * (CW + GAP) - GAP;
        const chosen = sim.path?.group === gi;
        return (
          <g
            key={gi}
            {...tip(
              <>
                <strong>FSM upper node {gi}</strong>
                <br />
                max of pages {base}–{base + n - 1}: category {g}
                <br />
                In real PostgreSQL this is a node of the binary tree inside an FSM page; one FSM page covers
                roughly 4 000 heap pages and its own maximum is stored one level up.
              </>,
            )}
          >
            <rect
              x={xAt(base)}
              y={GRP_Y}
              width={w}
              height={18}
              rx={3}
              fill={catColor(g)}
              stroke={chosen ? 'var(--viz-ink)' : 'var(--viz-grid)'}
              strokeWidth={chosen ? 2 : 1}
            />
            <text x={xAt(base) + w / 2} y={GRP_Y + 13} fontSize={10} textAnchor="middle" fill="var(--viz-ink)">
              {g}
            </text>
          </g>
        );
      })}

      {sim.pages.map((p, i) => {
        const f = freeBytes(p, sim.unit);
        const c = fsmCat(f);
        const liveH = Math.round((p.live / cap) * TILE_H);
        const deadH = Math.round((p.dead / cap) * TILE_H);
        const chosen = sim.path?.leaf === i;
        const isTarget = sim.target === i;
        return (
          <g
            key={i}
            {...tip(
              <>
                <strong>Page {i}</strong> · {p.live} live, {p.dead} dead of {cap} row slots
                <br />
                free {fmtNum(f)} B → FSM byte {c} (one category is {FSM_STEP} B)
                <br />
                {c >= need ? 'Qualifies' : 'Does not qualify'} for a {sim.unit} B row
                <br />
                VM: {p.allFrozen ? 'all-visible + all-frozen' : p.allVisible ? 'all-visible' : 'neither bit set'}
                {isTarget ? <> · this backend&apos;s rd_targblock</> : null}
                {p.live === 0 && p.dead === 0 ? (
                  <> · empty: reusable through the FSM, but only truncatable if it is at the tail</>
                ) : null}
              </>,
            )}
          >
            <rect
              x={xAt(i)}
              y={LEAF_Y}
              width={CW}
              height={16}
              rx={2}
              fill={catColor(c)}
              stroke={chosen ? 'var(--viz-ink)' : 'var(--viz-grid)'}
              strokeWidth={chosen ? 2 : 1}
            />
            <text
              x={xAt(i) + CW / 2}
              y={LEAF_Y + 12}
              fontSize={8}
              textAnchor="middle"
              fill={c > 150 ? 'var(--viz-surface)' : 'var(--viz-ink)'}
            >
              {c}
            </text>

            <rect
              x={xAt(i)}
              y={TILE_Y}
              width={CW}
              height={TILE_H}
              rx={2}
              fill="var(--viz-neutral)"
              stroke={chosen || isTarget ? 'var(--viz-ink)' : 'var(--viz-grid)'}
              strokeWidth={chosen || isTarget ? 2 : 1}
            />
            <rect x={xAt(i)} y={TILE_Y + TILE_H - liveH} width={CW} height={liveH} fill="var(--viz-clean)" />
            <rect
              x={xAt(i)}
              y={TILE_Y + TILE_H - liveH - deadH}
              width={CW}
              height={deadH}
              fill="var(--viz-stale)"
            />

            <rect
              x={xAt(i)}
              y={VM_Y}
              width={CW / 2 - 1}
              height={12}
              rx={2}
              fill={p.allVisible ? 'var(--viz-6)' : 'var(--viz-plane)'}
              stroke="var(--viz-grid)"
            />
            <rect
              x={xAt(i) + CW / 2 + 1}
              y={VM_Y}
              width={CW / 2 - 1}
              height={12}
              rx={2}
              fill={p.allFrozen ? 'var(--viz-7)' : 'var(--viz-plane)'}
              stroke="var(--viz-grid)"
            />

            {i % 4 === 0 ? (
              <text x={xAt(i) + CW / 2} y={height - 4} fontSize={9} textAnchor="middle" fill="var(--viz-ink-2)">
                {i}
              </text>
            ) : null}
          </g>
        );
      })}

      {sim.pages.length ? (
        <path
          d={`M ${xAt(sim.nextSlot % sim.pages.length) + CW / 2} ${LEAF_Y - 3} l -4 -6 l 8 0 z`}
          fill="var(--viz-ink)"
        />
      ) : null}

      {sim.pages.length === 0 ? (
        <text x={LEFT} y={TILE_Y + 32} fontSize={12} fill="var(--viz-ink-2)">
          No pages yet — and a relation under four pages never gets an FSM fork at all.
        </text>
      ) : null}
    </svg>
  );
}

/** Lives inside <TooltipHost> so useTip() sees the provider. */
function TablespaceFigure({ sim, avail }: { sim: InnoSim; avail: number }) {
  const tip = useTip();
  const height = 190;
  const svgW = widthFor(sim.pages.length, avail);
  const cap = innoCap(sim.unit);
  const states = extentStates(sim);

  return (
    <svg
      width={svgW}
      height={height}
      role="img"
      aria-label="InnoDB tablespace pages grouped into extents with their XDES states"
    >
      <text x={0} y={24} fontSize={11} fill="var(--viz-ink-2)">
        Extents
      </text>
      <text x={0} y={78} fontSize={11} fill="var(--viz-ink-2)">
        Pages
      </text>

      {sim.extentOwner.map((o, e) => {
        const base = e * EXTENT_PAGES;
        const w = EXTENT_PAGES * (CW + GAP) - GAP;
        const st = states[e];
        const fill =
          st === 'FSEG'
            ? 'var(--viz-1)'
            : st === 'FSP_FULL_FRAG'
              ? 'var(--viz-2)'
              : st === 'FSP_FREE_FRAG'
                ? 'var(--viz-4)'
                : 'var(--viz-neutral)';
        return (
          <g
            key={e}
            {...tip(
              <>
                <strong>Extent {e}</strong> · pages {base}–{base + EXTENT_PAGES - 1}
                <br />
                XDES state: {st}
                <br />
                {o === 'seg'
                  ? 'Owned outright by the clustered index leaf segment; its pages hang off FSEG_FREE / FSEG_NOT_FULL / FSEG_FULL.'
                  : o === 'frag'
                    ? 'A fragment extent: single pages are handed out to different segments, which is why a small table does not cost a whole extent.'
                    : 'Unallocated: on the tablespace FSP_FREE list, available to any segment.'}
              </>,
            )}
          >
            <rect x={xAt(base)} y={10} width={w} height={20} rx={3} fill={fill} stroke="var(--viz-grid)" />
            <text x={xAt(base) + w / 2} y={24} fontSize={9} textAnchor="middle" fill="var(--viz-ink)">
              {st}
            </text>
          </g>
        );
      })}

      {sim.pages.map((p, i) => {
        const h = 56;
        const fillH = p.kind === 'leaf' ? Math.round((p.rows / cap) * h) : 0;
        const mark = p.kind === 'meta' ? (i === 0 ? 'H' : i === 1 ? 'I' : 'N') : p.kind === 'root' ? 'R' : '';
        return (
          <g
            key={i}
            {...tip(
              <>
                <strong>Page {i}</strong> ·{' '}
                {p.kind === 'meta'
                  ? i === 0
                    ? 'FSP_HDR: the file space header plus the XDES entries for the first extents'
                    : i === 1
                      ? 'change buffer bitmap'
                      : 'inode page: the FSEG entries describing every file segment'
                  : p.kind === 'root'
                    ? 'clustered index root — a fragment page of the internal file segment'
                    : p.kind === 'leaf'
                      ? `clustered index leaf: ${p.rows} of ${cap} rows, ${Math.round((p.rows / cap) * 100)}% full`
                      : 'free page, still inside the file'}
                {p.frag && p.kind === 'leaf' ? <> · taken from the segment&apos;s 32-slot fragment array</> : null}
              </>,
            )}
          >
            <rect
              x={xAt(i)}
              y={40}
              width={CW}
              height={h}
              rx={2}
              fill={p.kind === 'free' ? 'var(--viz-plane)' : 'var(--viz-neutral)'}
              stroke={i === sim.lastPage ? 'var(--viz-ink)' : 'var(--viz-grid)'}
              strokeWidth={i === sim.lastPage ? 2 : 1}
            />
            {fillH > 0 ? <rect x={xAt(i)} y={40 + h - fillH} width={CW} height={fillH} fill="var(--viz-clean)" /> : null}
            {mark ? (
              <text x={xAt(i) + CW / 2} y={62} fontSize={9} textAnchor="middle" fill="var(--viz-ink-2)">
                {mark}
              </text>
            ) : null}
            {i % 4 === 0 ? (
              <text x={xAt(i) + CW / 2} y={height - 22} fontSize={9} textAnchor="middle" fill="var(--viz-ink-2)">
                {i}
              </text>
            ) : null}
          </g>
        );
      })}

      <text x={LEFT} y={height - 4} fontSize={10} fill="var(--viz-ink-2)">
        H = FSP_HDR · I = change buffer bitmap · N = inode page · R = index root · hollow = free page, still inside the
        file
      </text>
    </svg>
  );
}

function extentStates(s: InnoSim) {
  return s.extentOwner.map((o, e) => {
    const base = e * EXTENT_PAGES;
    const used = s.pages.slice(base, base + EXTENT_PAGES).filter((p) => p.kind !== 'free').length;
    if (o === 'seg') return 'FSEG';
    if (used === 0) return 'FSP_FREE';
    if (used === EXTENT_PAGES) return 'FSP_FULL_FRAG';
    return 'FSP_FREE_FRAG';
  });
}

export default function HeapGrowthFreeSpaceLab() {
  const [engine, setEngine] = useState<'pg' | 'innodb'>('pg');
  const [workload, setWorkload] = useState<Workload>('insert');
  const [rowBytes, setRowBytes] = useState(180);
  const [waiters, setWaiters] = useState(1);
  const [pg, setPg] = useState(() => freshPg(180));
  const [inno, setInno] = useState(() => freshInno(180));
  const [ref, width] = useSize(760);

  const extendBy = waiters === 1 ? 1 : Math.min(16, waiters * 2);

  const reset = (bytes = rowBytes) => {
    setPg(freshPg(bytes));
    setInno(freshInno(bytes));
  };

  const onRow = (n: number) => {
    setRowBytes(n);
    reset(n);
  };

  const run = () => {
    if (engine === 'pg') setPg((s) => runPgBatch(s, workload, extendBy, waiters));
    else setInno((s) => runInnoBatch(s, workload));
  };

  const maintain = () => {
    if (engine === 'pg') setPg(vacuumPg);
    else setInno(rebuildInno);
  };

  /* --------------------------------------------------------------- stats */
  const pgLive = pg.pages.reduce((a, p) => a + p.live, 0);
  const pgDead = pg.pages.reduce((a, p) => a + p.dead, 0);
  const pgFile = pg.pages.length * BLCKSZ;
  const pgLiveBytes = pgLive * pg.unit;
  const pgVisible = pg.pages.filter((p) => p.allVisible).length;
  const pgFrozen = pg.pages.filter((p) => p.allFrozen).length;
  const pgEmpty = pg.pages.filter((p) => p.live === 0 && p.dead === 0).length;
  let pgTrailing = 0;
  for (let i = pg.pages.length - 1; i >= 0 && pg.pages[i].live === 0 && pg.pages[i].dead === 0; i--) pgTrailing++;
  const need = needCat(pg.unit);

  const innoRows = inno.pages.reduce((a, p) => a + (p.kind === 'leaf' ? p.rows : 0), 0);
  const innoLeafPages = inno.pages.filter((p) => p.kind === 'leaf').length;
  const innoFreePages = inno.pages.filter((p) => p.kind === 'free').length;
  const innoCapacity = innoCap(inno.unit);
  const innoFill = innoLeafPages ? innoRows / (innoLeafPages * innoCapacity) : 0;
  const states = extentStates(inno);

  const stats =
    engine === 'pg'
      ? [
          { label: 'Heap fork', value: `${pg.pages.length} pages · ${fmtBytes(pgFile)}` },
          { label: 'Live / dead rows', value: `${fmtNum(pgLive)} / ${fmtNum(pgDead)}` },
          {
            label: 'Not live tuples',
            value: pgFile ? `${Math.round(((pgFile - pgLiveBytes) / pgFile) * 100)}%` : '—',
            hint: 'Dead tuples plus free space, as a share of the file — the bloat number',
          },
          {
            label: 'FSM descents / skipped',
            value: `${fmtNum(pg.fsmLookups)} / ${fmtNum(pg.targetHits)}`,
            hint: 'Skipped = inserts served by the cached rd_targblock with no lookup at all',
          },
          {
            label: 'Extensions',
            value: `${fmtNum(pg.extensions)} → ${fmtNum(pg.extendedPages)} pages`,
            hint: 'Each one takes the relation extension lock',
          },
          { label: 'all-visible / all-frozen', value: `${pgVisible} / ${pgFrozen} pages` },
          { label: 'Truncatable tail', value: `${pgTrailing} page${pgTrailing === 1 ? '' : 's'}` },
          {
            label: 'Empty interior pages',
            value: `${Math.max(0, pgEmpty - pgTrailing)}`,
            hint: 'Reusable by the FSM, never returned to the filesystem',
          },
        ]
      : [
          { label: 'Tablespace', value: `${inno.pages.length} pages · ${fmtBytes(inno.pages.length * INNO_PAGE)}` },
          {
            label: 'Extents',
            value: `${inno.extentOwner.length} (${states.filter((e) => e === 'FSEG').length} owned by a segment)`,
          },
          { label: 'Rows', value: fmtNum(innoRows) },
          { label: 'Leaf pages', value: `${innoLeafPages} @ ${Math.round(innoFill * 100)}% full` },
          { label: 'Fragment pages used', value: `${inno.fragUsed} / ${FRAG_SLOTS}` },
          { label: 'Splits / merges', value: `${fmtNum(inno.splits)} / ${fmtNum(inno.merges)}` },
          { label: 'Free pages inside the file', value: `${innoFreePages}` },
          { label: 'Rebuilds', value: `${inno.rebuilds}` },
        ];

  return (
    <VizPanel
      title="A heap file growing, shrinking, and failing to shrink"
      subtitle={
        engine === 'pg'
          ? 'One 8 KB heap page per column. Run a workload and watch the FSM byte change, the FSM descent pick the page the next row lands on, the relation extend at the tail, and VACUUM give back only what is at the end of the file.'
          : 'The same workloads against an InnoDB tablespace. Two numbers are scaled so the strip stays legible: an extent is 8 pages here (really 64 pages / 1 MB at the default 16 KB page) and the segment fragment array has 4 slots (really 32).'
      }
      controls={
        <>
          <Segmented
            label="Engine"
            value={engine}
            onChange={setEngine}
            options={[
              { value: 'pg', label: 'PostgreSQL heap' },
              { value: 'innodb', label: 'InnoDB tablespace' },
            ]}
          />
          <Choice label="Workload" value={workload} onChange={setWorkload} options={WORKLOADS} />
          <Slider
            label="Row width"
            min={40}
            max={700}
            step={20}
            value={rowBytes}
            onChange={onRow}
            format={(n) => `${n} B (resets)`}
          />
          <Slider
            label="Concurrent inserters"
            min={1}
            max={8}
            value={waiters}
            onChange={setWaiters}
            disabled={engine !== 'pg'}
            format={(n) => (n === 1 ? '1 · one page per extension' : `${n} · +${extendBy} pages per extension`)}
          />
          <Button onClick={run} primary>
            Run {OPS_PER_BATCH} ops
          </Button>
          <Button onClick={maintain}>{engine === 'pg' ? 'VACUUM' : 'OPTIMIZE TABLE'}</Button>
          <Button onClick={() => reset()}>Reset</Button>
        </>
      }
      legend={
        engine === 'pg' ? (
          <Legend
            items={[
              { label: 'Live tuples', color: 'var(--viz-clean)' },
              { label: 'Dead tuples — bytes held until VACUUM', color: 'var(--viz-stale)' },
              { label: 'Free space', color: 'var(--viz-neutral)' },
              { label: 'FSM category (darker = more free space)', color: 'var(--viz-seq-550)' },
              { label: 'all-visible bit', color: 'var(--viz-6)' },
              { label: 'all-frozen bit', color: 'var(--viz-7)' },
            ]}
          />
        ) : (
          <Legend
            items={[
              { label: 'Rows in a clustered-index leaf', color: 'var(--viz-clean)' },
              { label: 'Free space in a page', color: 'var(--viz-neutral)' },
              { label: 'Extent on FSP_FREE_FRAG', color: 'var(--viz-4)' },
              { label: 'Extent on FSP_FULL_FRAG', color: 'var(--viz-2)' },
              { label: 'Extent owned by the leaf file segment', color: 'var(--viz-1)' },
            ]}
          />
        )
      }
      stats={<Stats items={stats} />}
      note={<Note>{engine === 'pg' ? pg.msg : inno.msg}</Note>}
      table={
        engine === 'pg' ? (
          <table className="viz-table">
            <thead>
              <tr>
                <th>Page</th>
                <th>Live</th>
                <th>Dead</th>
                <th>Free bytes</th>
                <th>FSM byte</th>
                <th>Fits a {pg.unit} B row?</th>
                <th>VM bits</th>
              </tr>
            </thead>
            <tbody>
              {pg.pages.map((p, i) => {
                const f = freeBytes(p, pg.unit);
                const c = fsmCat(f);
                return (
                  <tr key={i}>
                    <td>{i}</td>
                    <td>{p.live}</td>
                    <td>{p.dead}</td>
                    <td>{fmtNum(f)}</td>
                    <td>
                      {c} (≥ {c * FSM_STEP} B)
                    </td>
                    <td>{c >= need ? 'yes' : 'no'}</td>
                    <td>{p.allFrozen ? 'all-visible + all-frozen' : p.allVisible ? 'all-visible' : '—'}</td>
                  </tr>
                );
              })}
              {pg.pages.length === 0 ? (
                <tr>
                  <td colSpan={7}>No pages yet.</td>
                </tr>
              ) : null}
            </tbody>
          </table>
        ) : (
          <table className="viz-table">
            <thead>
              <tr>
                <th>Extent</th>
                <th>Pages</th>
                <th>XDES state / list</th>
                <th>Used pages</th>
                <th>Rows</th>
              </tr>
            </thead>
            <tbody>
              {inno.extentOwner.map((_, e) => {
                const base = e * EXTENT_PAGES;
                const slice = inno.pages.slice(base, base + EXTENT_PAGES);
                return (
                  <tr key={e}>
                    <td>{e}</td>
                    <td>
                      {base}–{base + EXTENT_PAGES - 1}
                    </td>
                    <td>{states[e]}</td>
                    <td>{slice.filter((p) => p.kind !== 'free').length}</td>
                    <td>{fmtNum(slice.reduce((a, p) => a + p.rows, 0))}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        )
      }
    >
      <div ref={ref}>
        <TooltipHost>
          {engine === 'pg' ? <HeapFigure sim={pg} avail={width} /> : <TablespaceFigure sim={inno} avail={width} />}
        </TooltipHost>
      </div>
    </VizPanel>
  );
}
