import { useState } from 'react';
import {
  VizPanel,
  Slider,
  Segmented,
  Check,
  Button,
  Legend,
  Stats,
  Note,
  TooltipHost,
  useTip,
  fmtBytes,
  fmtNum,
  makeRng,
  useSize,
} from './Viz';

/**
 * The update path, one UPDATE at a time.
 *
 * PostgreSQL never updates a heap tuple in place: every UPDATE writes a new version. The
 * only question is whether that version can stay on the same page with no index entry
 * (HOT) or has to be inserted somewhere else and announced to every index. The answer is
 * exactly two predicates — "no indexed column changed" and "the new version fits on this
 * page" — and fillfactor exists only to keep the second one true.
 *
 * The page here is a real 8 KB heap page: 24-byte PageHeaderData, a 4-byte ItemIdData per
 * line pointer growing up from pd_lower, tuples growing down from pd_upper. Pruning frees
 * tuple bytes and the line pointers of heap-only tuples; only VACUUM's second pass may turn
 * an LP_DEAD root into LP_UNUSED, because only VACUUM has removed the index entries that
 * point at it.
 *
 * InnoDB mode replays the same stream against the other design: update the clustered-index
 * record in place, push the prior column values into an undo record reached by DB_ROLL_PTR,
 * and touch only the secondary indexes whose own key changed.
 */

/* ------------------------------------------------------------------ model */

const BLCKSZ = 8192;
const PAGE_HDR = 24; // PageHeaderData, no special space on a heap page
const LP_BYTES = 4; // ItemIdData
const TUP_HDR = 24; // MAXALIGN(23-byte HeapTupleHeaderData + null bitmap)
const MAX_LPS = 20; // drawing cap; the real limit is MaxHeapTuplesPerPage = 291

const align8 = (n: number) => Math.ceil(n / 8) * 8;
const tupLen = (rowBytes: number) => align8(TUP_HDR + rowBytes);

type LpState = 'unused' | 'normal' | 'redirect' | 'dead';

type Ver = {
  row: number; // logical row (primary key)
  ver: number; // which version of that row
  xmin: number;
  xmax: number; // 0 = still live
  ctid: [number, number]; // page, line pointer (1-based), self when live
  hot: boolean; // HEAP_ONLY_TUPLE — no index entry points at it
  hotUpdated: boolean; // HEAP_HOT_UPDATED — t_ctid is a chain link, not self
};

type Lp = { state: LpState; redirect: number; ver: Ver | null };

type IdxEntry = { row: number; page: number; lp: number; killed: boolean };

type Undo = { id: number; row: number; trx: number; purged: boolean };

type LogRow = {
  n: number;
  op: string;
  detail: string;
  idx: number;
  free: number;
  lps: number;
};

type S = {
  lps: Lp[];
  idx: IdxEntry[];
  moved: number[]; // rows whose live version has left this page
  xid: number;
  prunePending: boolean; // pd_prune_xid is set: an update left a chain candidate here
  vacPass1: boolean;
  deadTids: number[]; // collected by VACUUM's first pass, awaiting the index vacuum
  // InnoDB side
  recs: { row: number; trx: number; roll: number }[];
  undo: Undo[];
  delMarked: number;
  // counters
  nUpd: number;
  nHot: number;
  nNewpage: number;
  idxWrites: number;
  scanPath: number[];
  scanRow: number;
  scanHops: number;
  head: string;
  body: string;
  log: LogRow[];
};

type Cfg = {
  engine: 'pg' | 'innodb';
  ff: number;
  rowBytes: number;
  nIdx: number;
  keyed: boolean; // the UPDATE changes an indexed column
  snapXid: number | null; // an open repeatable-read snapshot pinning the horizon
};

/** Which rows an UPDATE picks, fixed for every render and every reload. */
const PICK = (() => {
  const rng = makeRng(19960101);
  return Array.from({ length: 96 }, () => rng());
})();

function initial(cfg: Cfg): S {
  const len = tupLen(cfg.rowBytes);
  const usable = BLCKSZ - PAGE_HDR;
  const rows = Math.max(2, Math.floor((usable * cfg.ff) / 100 / (len + LP_BYTES)));
  const lps: Lp[] = [];
  const idx: IdxEntry[] = [];
  const recs: { row: number; trx: number; roll: number }[] = [];
  for (let i = 0; i < rows; i++) {
    lps.push({
      state: 'normal',
      redirect: 0,
      ver: { row: i + 1, ver: 1, xmin: 100, xmax: 0, ctid: [0, i + 1], hot: false, hotUpdated: false },
    });
    idx.push({ row: i + 1, page: 0, lp: i + 1, killed: false });
    recs.push({ row: i + 1, trx: 100, roll: 0 });
  }
  return {
    lps,
    idx,
    moved: [],
    xid: 101,
    prunePending: false,
    vacPass1: false,
    deadTids: [],
    recs,
    undo: [],
    delMarked: 0,
    nUpd: 0,
    nHot: 0,
    nNewpage: 0,
    idxWrites: 0,
    scanPath: [],
    scanRow: 0,
    scanHops: 0,
    head:
      cfg.engine === 'pg'
        ? `${rows} rows loaded, ${100 - cfg.ff}% of the page held back by fillfactor.`
        : `${rows} clustered-index records loaded, no undo history yet.`,
    body:
      cfg.engine === 'pg'
        ? 'Every line pointer is LP_NORMAL and every index entry points straight at a live tuple. Run an ' +
          'UPDATE and watch which of the two HOT predicates decides where the new version goes.'
        : 'Each record carries DB_TRX_ID and DB_ROLL_PTR. An UPDATE rewrites the record where it sits and ' +
          'pushes the old column values into an undo record; the page itself never grows a second version.',
    log: [],
  };
}

function pageUsed(s: S, cfg: Cfg) {
  let b = PAGE_HDR + LP_BYTES * s.lps.length;
  for (const lp of s.lps) if (lp.ver) b += tupLen(cfg.rowBytes);
  return b;
}

const pageFree = (s: S, cfg: Cfg) => BLCKSZ - pageUsed(s, cfg);

/** oldestXmin: with an open snapshot, nothing that snapshot might still need can be removed. */
const horizon = (s: S, cfg: Cfg) => (cfg.snapXid === null ? s.xid : cfg.snapXid);

const isDead = (v: Ver, s: S, cfg: Cfg) => v.xmax !== 0 && v.xmax < horizon(s, cfg);

function liveRowsOnPage(s: S) {
  const out: number[] = [];
  s.lps.forEach((lp) => {
    if (lp.state === 'normal' && lp.ver && lp.ver.xmax === 0) out.push(lp.ver.row);
  });
  return out;
}

function tally(s: S, cfg: Cfg) {
  let live = 0;
  let dead = 0;
  let deadLp = 0;
  let redirects = 0;
  let unused = 0;
  for (const lp of s.lps) {
    if (lp.state === 'dead') deadLp++;
    if (lp.state === 'redirect') redirects++;
    if (lp.state === 'unused') unused++;
    if (lp.ver) (isDead(lp.ver, s, cfg) ? dead++ : live++);
  }
  return { live, dead, deadLp, redirects, unused };
}

/* ------------------------------------------------------------ operations */

type Op = 'update' | 'read' | 'vac1' | 'vac2' | 'scan' | 'purge';

const OP_LABEL: Record<Op, string> = {
  update: 'UPDATE',
  read: 'SELECT (page read)',
  vac1: 'VACUUM pass 1',
  vac2: 'index vacuum + pass 2',
  scan: 'index scan',
  purge: 'purge',
};

/** Free an in-page version: the bytes go back to pd_upper on the next defragmentation. */
const clearVer = (state: LpState): Lp => ({ state, redirect: 0, ver: null });

function pgUpdate(s: S, cfg: Cfg): S {
  const candidates = liveRowsOnPage(s);
  if (candidates.length === 0) return s;
  const row = candidates[Math.floor(PICK[s.nUpd % PICK.length] * candidates.length)];
  const fromLp = s.lps.findIndex((lp) => lp.ver && lp.ver.row === row && lp.ver.xmax === 0);
  const old = s.lps[fromLp].ver!;
  const xid = s.xid;
  const len = tupLen(cfg.rowBytes);
  const free = pageFree(s, cfg);
  const lps = s.lps.map((lp) => ({ ...lp }));
  const idx = s.idx.map((e) => ({ ...e }));

  // A free slot is reusable only if the line pointer is LP_UNUSED; LP_DEAD stubs are not.
  let slot = lps.findIndex((lp) => lp.state === 'unused');
  const needsNewLp = slot === -1;
  const fits = free >= len + (needsNewLp ? LP_BYTES : 0) && (!needsNewLp || lps.length < MAX_LPS);
  const hot = !cfg.keyed && fits;

  let head: string;
  let body: string;
  let detail: string;
  let idxWrites = 0;

  if (!fits) {
    // No room: the new version is inserted on another page and every index gets an entry.
    const offPage = idx.filter((e) => e.page === 1).length + 1;
    lps[fromLp] = { ...lps[fromLp], ver: { ...old, xmax: xid, ctid: [1, offPage], hotUpdated: false } };
    idxWrites = cfg.nIdx;
    idx.push({ row, page: 1, lp: offPage, killed: false });
    head = `Row ${row}: no room on the page — the new version went to another block.`;
    body =
      `The page has ${fmtBytes(free)} between pd_lower and pd_upper, less than the ${fmtBytes(len + LP_BYTES)} a ` +
      `new version plus its line pointer needs, so PageGetHeapFreeSpace came up short, heap_update called ` +
      `RelationGetBufferForTuple and landed on a different ` +
      `page. That makes it a non-HOT update by definition: ${cfg.nIdx} index entr${cfg.nIdx === 1 ? 'y' : 'ies'} ` +
      `written, the old tuple's t_ctid now points off-page, and n_tup_newpage_upd ticks. Raising the free space ` +
      `here — a lower fillfactor, or a prune — is the entire fix.`;
    detail = 'spilled to another page';
  } else if (hot) {
    const target = needsNewLp ? lps.length + 1 : slot + 1;
    const nv: Ver = { row, ver: old.ver + 1, xmin: xid, xmax: 0, ctid: [0, target], hot: true, hotUpdated: false };
    if (needsNewLp) lps.push({ state: 'normal', redirect: 0, ver: nv });
    else lps[slot] = { state: 'normal', redirect: 0, ver: nv };
    lps[fromLp] = { ...lps[fromLp], ver: { ...old, xmax: xid, ctid: [0, target], hotUpdated: true } };
    head = `Row ${row}: HOT update — no index entry written.`;
    body =
      `No indexed column changed and the new version fit, so heap_update took the HOT path: the new tuple is ` +
      `flagged HEAP_ONLY_TUPLE at lp ${target}, the old one gets HEAP_HOT_UPDATED and its t_ctid becomes the ` +
      `chain link (0,${target}). Every index still points at lp ${fromLp + 1} and is still correct, because a ` +
      `scan that lands there walks the chain. n_tup_hot_upd + 1, index writes + 0.`;
    detail = `HOT chain link lp${fromLp + 1} → lp${target}`;
  } else {
    const target = needsNewLp ? lps.length + 1 : slot + 1;
    const nv: Ver = { row, ver: old.ver + 1, xmin: xid, xmax: 0, ctid: [0, target], hot: false, hotUpdated: false };
    if (needsNewLp) lps.push({ state: 'normal', redirect: 0, ver: nv });
    else lps[slot] = { state: 'normal', redirect: 0, ver: nv };
    lps[fromLp] = { ...lps[fromLp], ver: { ...old, xmax: xid, ctid: [0, target], hotUpdated: false } };
    idxWrites = cfg.nIdx;
    idx.push({ row, page: 0, lp: target, killed: false });
    head = `Row ${row}: an indexed column changed — no HOT, even on the same page.`;
    body =
      `The new version fits at lp ${target}, but it carries a different value for an indexed column, so an index ` +
      `entry has to exist for it. heap_update writes into all ${cfg.nIdx} index` +
      `${cfg.nIdx === 1 ? '' : 'es'} — not just the one whose key changed — because Postgres has no way to ` +
      `reach the new tuple from the old index entry. The old tuple is left with t_ctid = (0,${target}) and no ` +
      `HEAP_HOT_UPDATED flag, so the chain is not HOT and pruning can never collapse it into a redirect.`;
    detail = `non-HOT, ${cfg.nIdx} index entr${cfg.nIdx === 1 ? 'y' : 'ies'}`;
  }

  const moved = !fits ? [...s.moved, row] : s.moved;
  const next: S = {
    ...s,
    lps,
    idx,
    moved,
    xid: xid + 1,
    prunePending: true,
    nUpd: s.nUpd + 1,
    nHot: s.nHot + (hot ? 1 : 0),
    nNewpage: s.nNewpage + (fits ? 0 : 1),
    idxWrites: s.idxWrites + idxWrites,
    scanPath: [],
    scanHops: 0,
    head,
    body,
    log: s.log,
  };
  return log(next, cfg, 'update', detail, idxWrites);
}

/** heap_page_prune: collapse dead chain members, free their bytes, leave the roots alone. */
function prune(s: S, cfg: Cfg): { s: S; freed: number; redirected: number; killedLps: number } {
  const lps = s.lps.map((lp) => ({ ...lp }));
  let freed = 0;
  let redirected = 0;
  let killedLps = 0;

  const rootLps = lps
    .map((lp, i) => ({ lp, i }))
    .filter(({ lp }) => (lp.state === 'normal' && lp.ver && !lp.ver.hot) || lp.state === 'redirect');

  for (const { i } of rootLps) {
    // Walk the chain from the root, collecting members.
    const chain: number[] = [];
    let cur = lps[i].state === 'redirect' ? lps[i].redirect - 1 : i;
    const seen = new Set<number>();
    while (cur >= 0 && cur < lps.length && !seen.has(cur) && lps[cur].ver) {
      seen.add(cur);
      chain.push(cur);
      const v = lps[cur].ver!;
      if (v.hotUpdated && v.ctid[0] === 0) cur = v.ctid[1] - 1;
      else break;
    }
    if (chain.length === 0) continue;

    // Dead members at the head of the chain go away; the first survivor becomes the target.
    let first = 0;
    while (first < chain.length && isDead(lps[chain[first]].ver!, s, cfg)) first++;

    if (first === 0) continue; // nothing removable in this chain

    for (let k = 0; k < first; k++) {
      const at = chain[k];
      const v = lps[at].ver!;
      freed += tupLen(cfg.rowBytes);
      if (v.hot) {
        // A heap-only tuple is referenced by nothing but its predecessor: the slot is reusable now.
        lps[at] = clearVer('unused');
        killedLps++;
      } else {
        // An index may still point here, so the stub has to stay until VACUUM removes those entries.
        lps[at] = clearVer('dead');
      }
    }

    if (first < chain.length) {
      // The root item survives as a pointer to whatever is still live in its chain.
      const target = chain[first] + 1;
      const wasRedirect = lps[i].state === 'redirect';
      lps[i] = { state: 'redirect', redirect: target, ver: null };
      if (!wasRedirect) redirected++;
    } else if (lps[i].state === 'redirect') {
      // Nothing left alive: the redirect becomes a dead stub, not a free slot — indexes still name it.
      lps[i] = clearVer('dead');
    }
  }

  return {
    s: { ...s, lps, prunePending: freed > 0 ? false : s.prunePending },
    freed,
    redirected,
    killedLps,
  };
}

function pgRead(s: S, cfg: Cfg): S {
  const free = pageFree(s, cfg);
  // heap_page_prune_opt: Max(the relation's fillfactor-derived target free space, BLCKSZ/10)
  const minfree = Math.max((BLCKSZ * (100 - cfg.ff)) / 100, BLCKSZ / 10);
  if (!s.prunePending) {
    return log(
      {
        ...s,
        scanPath: [],
        head: 'The page read did nothing: pd_prune_xid is not set.',
        body:
          'heap_page_prune_opt checks the page header first. No update on this page has left a chain candidate ' +
          'behind, so the read skips straight to returning tuples — no cleanup lock is even attempted.',
      },
      cfg,
      'read',
      'no candidate',
      0,
    );
  }
  if (free >= minfree) {
    return log(
      {
        ...s,
        scanPath: [],
        head: `The page read declined to prune: ${fmtBytes(free)} free is above the threshold.`,
        body:
          `heap_page_prune_opt only bothers when the page is marked full or has less free space than ` +
          `Max(fillfactor target, BLCKSZ/10) = ${fmtBytes(minfree)}. Pruning takes a cleanup lock — an exclusive ` +
          `lock plus the requirement that ` +
          `nobody else holds a pin — and it dirties the page and writes WAL, so a page with room to spare is left ` +
          `alone. Keep updating until it fills.`,
      },
      cfg,
      'read',
      'above threshold',
      0,
    );
  }

  const r = prune(s, cfg);
  const held = cfg.snapXid !== null;
  if (r.freed === 0) {
    return log(
      {
        ...r.s,
        scanPath: [],
        head: held
          ? 'Pruning ran and removed nothing: the open snapshot still needs every version.'
          : 'Pruning ran and found nothing removable.',
        body: held
          ? `A transaction holding a snapshot at xid ${cfg.snapXid} pins oldestXmin there, and no version deleted ` +
            `at or after that xid may be removed. The page keeps filling, HOT updates start failing the ` +
            `"fits on the page" test, and the table bloats. This is exactly what a forgotten "idle in transaction" ` +
            `session does to an update-heavy table.`
          : 'Every version in every chain is still visible to somebody, so there is nothing to collapse.',
      },
      cfg,
      'read',
      held ? 'blocked by snapshot' : 'nothing dead',
      0,
    );
  }
  return log(
    {
      ...r.s,
      scanPath: [],
      head: `Opportunistic prune: ${fmtBytes(r.freed)} reclaimed by a plain SELECT.`,
      body:
        `heap_page_prune_opt took the cleanup lock, removed ${r.freed / tupLen(cfg.rowBytes)} dead version` +
        `${r.freed / tupLen(cfg.rowBytes) === 1 ? '' : 's'}, and called PageRepairFragmentation to slide the ` +
        `survivors together so the free space is one contiguous run between pd_lower and pd_upper. ` +
        `${r.redirected > 0 ? `${r.redirected} root line pointer${r.redirected === 1 ? '' : 's'} became LP_REDIRECT, ` : ''}` +
        `${r.killedLps > 0 ? `${r.killedLps} heap-only slot${r.killedLps === 1 ? '' : 's'} went back to LP_UNUSED, ` : ''}` +
        `and the roots that indexes point at stayed put. No VACUUM was involved and the operation is WAL-logged, ` +
        `which is why a read-only query can dirty a page.`,
    },
    cfg,
    'read',
    `pruned ${fmtBytes(r.freed)}`,
    0,
  );
}

function pgVacuum1(s: S, cfg: Cfg): S {
  const r = prune(s, cfg);
  const dead = r.s.lps.map((lp, i) => ({ lp, i })).filter(({ lp }) => lp.state === 'dead').map(({ i }) => i + 1);
  return log(
    {
      ...r.s,
      deadTids: dead,
      vacPass1: true,
      scanPath: [],
      head: `VACUUM pass 1: pruned the page and collected ${dead.length} dead TID${dead.length === 1 ? '' : 's'}.`,
      body:
        `The first heap pass does the same pruning a SELECT can do — unconditionally this time, no free-space ` +
        `threshold — and then records every LP_DEAD item's TID in the dead-TID store (a TidStore since ` +
        `PostgreSQL 17, a flat array bounded by maintenance_work_mem before that). It cannot free those line ` +
        `pointers yet: an index entry may still point at each one, and setting the slot LP_UNUSED would let a ` +
        `future insert reuse it under an index entry that has not been removed. That is why the heap is visited ` +
        `twice.`,
    },
    cfg,
    'vac1',
    `${dead.length} dead TIDs`,
    0,
  );
}

function pgVacuum2(s: S, cfg: Cfg): S {
  const dead = new Set(s.deadTids);
  const removed = s.idx.filter((e) => e.page === 0 && dead.has(e.lp)).length * cfg.nIdx;
  const idx = s.idx.filter((e) => !(e.page === 0 && dead.has(e.lp)));
  const lps = s.lps.map((lp, i) => (dead.has(i + 1) ? { state: 'unused' as LpState, redirect: 0, ver: null } : lp));
  // Trailing LP_UNUSED slots are truncated off pd_lower.
  while (lps.length > 0 && lps[lps.length - 1].state === 'unused') lps.pop();
  return log(
    {
      ...s,
      lps,
      idx,
      deadTids: [],
      vacPass1: false,
      scanPath: [],
      head: `Index vacuum removed ${removed} entr${removed === 1 ? 'y' : 'ies'}; pass 2 released ${dead.size} line pointer${dead.size === 1 ? '' : 's'}.`,
      body:
        `Between the two heap passes, VACUUM scans every index in full and deletes the entries whose TID is in ` +
        `the dead list — that scan is proportional to the size of the indexes, not to the number of dead rows, ` +
        `which is why a table with eight indexes is eight times the vacuum. Only now is it safe for the second ` +
        `heap pass to set those items LP_UNUSED, and only now can the freed slots be reused. Trailing LP_UNUSED ` +
        `slots are given back by lowering pd_lower.`,
    },
    cfg,
    'vac2',
    `${removed} index entries removed`,
    0,
  );
}

function pgScan(s: S, cfg: Cfg): S {
  const onPage = s.idx.filter((e) => !e.killed && e.page === 0);
  /** How many heap hops this entry costs before it reaches a live version; -1 if it reaches none. */
  const hopsTo = (e: IdxEntry) => {
    let at = e.lp - 1;
    let hops = 0;
    if (s.lps[at].state === 'dead' || s.lps[at].state === 'unused') return -1;
    if (s.lps[at].state === 'redirect') {
      at = s.lps[at].redirect - 1;
      hops++;
    }
    const seen = new Set<number>();
    while (at >= 0 && at < s.lps.length && s.lps[at].ver && !seen.has(at)) {
      seen.add(at);
      const v = s.lps[at].ver!;
      if (v.xmax === 0) return hops;
      if (v.hotUpdated && v.ctid[0] === 0) {
        at = v.ctid[1] - 1;
        hops++;
      } else return -1;
    }
    return -1;
  };
  // Prefer an entry that actually has a chain to walk — that is the interesting one to watch.
  const chased = onPage.filter((e) => hopsTo(e) > 0).map((e) => e.row);
  const flat = onPage.filter((e) => hopsTo(e) === 0).map((e) => e.row);
  const rows = chased.length > 0 ? chased : flat.length > 0 ? flat : onPage.map((e) => e.row);
  if (rows.length === 0) {
    return log({ ...s, scanPath: [], head: 'No index entry points into this page any more.', body: 'Every entry was either killed or vacuumed away.' }, cfg, 'scan', 'no entries', 0);
  }
  const row = rows[Math.floor(PICK[(s.nUpd + s.log.length) % PICK.length] * rows.length)];
  const entries = s.idx.filter((e) => e.row === row && e.page === 0 && !e.killed);
  const idx = s.idx.map((e) => ({ ...e }));
  const path: number[] = [];
  let hops = 0;
  let found: number | null = null;
  let killed = 0;
  let viaRedirect = false;

  for (const e of entries) {
    let at = e.lp - 1;
    if (s.lps[at].state === 'dead' || s.lps[at].state === 'unused') {
      const k = idx.find((x) => x.row === e.row && x.lp === e.lp && x.page === 0);
      if (k) k.killed = true;
      killed++;
      path.push(at);
      continue;
    }
    path.push(at);
    if (s.lps[at].state === 'redirect') {
      viaRedirect = true;
      at = s.lps[at].redirect - 1;
      hops++;
      path.push(at);
    }
    const seen = new Set<number>();
    while (at >= 0 && at < s.lps.length && s.lps[at].ver && !seen.has(at)) {
      seen.add(at);
      const v = s.lps[at].ver!;
      if (v.xmax === 0) {
        found = at;
        break;
      }
      if (v.hotUpdated && v.ctid[0] === 0) {
        at = v.ctid[1] - 1;
        hops++;
        path.push(at);
      } else break;
    }
    if (found !== null) break;
  }

  return log(
    {
      ...s,
      idx,
      scanPath: path,
      scanRow: row,
      scanHops: hops,
      head:
        found !== null
          ? `Index scan for row ${row}: ${hops} heap hop${hops === 1 ? '' : 's'} after the index entry.`
          : `Index scan for row ${row}: the entry pointed at a dead stub.`,
      body:
        found !== null
          ? `The index entry still names the root TID (0,${entries[0]?.lp}) it was given when the row was first ` +
            `inserted, and it was never rewritten. The scan pins the page once and follows it: ` +
            `${viaRedirect ? 'the root item is LP_REDIRECT, so it jumps straight to the chain head' : 'the root item is LP_NORMAL'}` +
            `${hops > (viaRedirect ? 1 : 0) ? ', then walks t_ctid while HEAP_HOT_UPDATED is set, checking each version against the snapshot' : ''}` +
            `, and stops at lp ${found + 1} where xmax = 0. Every hop is a pointer chase inside one ` +
            `already-pinned buffer — cheap, but not free: a long chain is real CPU on a hot row, which is why ` +
            `pruning collapses chains rather than letting them grow.`
          : `The root item is LP_DEAD, so nothing in this chain is visible to anyone. The scan sets the index ` +
            `tuple's LP_DEAD hint — the kill_prior_tuple mechanism — so later scans skip the entry without ` +
            `visiting the heap at all, and a B-tree page split can drop it to make room. That is the cheap ` +
            `counterpart to VACUUM: it never frees a heap line pointer, only index work.`,
    },
    cfg,
    'scan',
    found !== null ? `${hops} hops` : `${killed} entry killed`,
    0,
  );
}

/* ------------------------------------------------------------ InnoDB side */

function ibUpdate(s: S, cfg: Cfg): S {
  const row = s.recs[Math.floor(PICK[s.nUpd % PICK.length] * s.recs.length)].row;
  const recs = s.recs.map((r) => ({ ...r }));
  const at = recs.findIndex((r) => r.row === row);
  const trx = s.xid;
  const undoId = s.undo.length + 1;
  const undo = [...s.undo, { id: undoId, row, trx: recs[at].trx, purged: false }];
  const secOps = cfg.keyed ? 2 : 0;
  recs[at] = { row, trx, roll: undoId };
  return log(
    {
      ...s,
      recs,
      undo,
      xid: trx + 1,
      delMarked: s.delMarked + (cfg.keyed ? 1 : 0),
      nUpd: s.nUpd + 1,
      idxWrites: s.idxWrites + secOps,
      scanPath: [],
      head: cfg.keyed
        ? `Row ${row}: updated in place, one secondary index maintained.`
        : `Row ${row}: updated in place, no index touched at all.`,
      body:
        `InnoDB rewrote the clustered-index record where it sits, set DB_TRX_ID = ${trx} and pointed DB_ROLL_PTR ` +
        `at undo record ${undoId}, which holds the prior column values. The page did not grow a second version, ` +
        `so there is no fillfactor question and no chain to prune. ` +
        (cfg.keyed
          ? `Because an indexed column changed, the old secondary entry is delete-marked and a new one inserted — ` +
            `two operations in that one index. The other ${Math.max(0, cfg.nIdx - 1)} index` +
            `${cfg.nIdx - 1 === 1 ? '' : 'es'} are untouched, because their keys did not change: InnoDB maintains ` +
            `only the indexes whose own key moved, which is the structural advantage of updating in place.`
          : `No secondary index was touched, and unlike a Postgres HOT update that is not conditional on free ` +
            `space — it holds however full the page is.`),
    },
    cfg,
    'update',
    cfg.keyed ? '1 index: delete-mark + insert' : 'in place, 0 index ops',
    secOps,
  );
}

function ibPurge(s: S, cfg: Cfg): S {
  const h = horizon(s, cfg);
  const undo = s.undo.map((u) => (u.trx < h ? { ...u, purged: true } : u));
  const gone = undo.filter((u) => u.purged).length - s.undo.filter((u) => u.purged).length;
  const held = cfg.snapXid !== null;
  return log(
    {
      ...s,
      undo,
      delMarked: held ? s.delMarked : 0,
      scanPath: [],
      head: held
        ? `Purge stalled: an open read view pins the history at xid ${cfg.snapXid}.`
        : `Purge released ${gone} undo record${gone === 1 ? '' : 's'}.`,
      body: held
        ? `The purge threads may only remove undo records and delete-marked entries older than the oldest read ` +
          `view. One long-running transaction therefore grows the history list without bound — visible as ` +
          `History list length in SHOW ENGINE INNODB STATUS — and every consistent read of a hot row has to walk ` +
          `further back down DB_ROLL_PTR to build the version it is allowed to see. Postgres bloats the heap; ` +
          `InnoDB bloats the undo and slows its own reads.`
        : `The purge threads dropped the undo records no read view can still need and removed the delete-marked ` +
          `secondary entries left by keyed updates. This is InnoDB's equivalent of VACUUM: same job, different ` +
          `place, and it never has to rewrite the record itself.`,
    },
    cfg,
    'purge',
    held ? 'blocked by read view' : `${gone} undo records freed`,
    0,
  );
}

/* ---------------------------------------------------------------- logging */

function log(s: S, cfg: Cfg, op: Op, detail: string, idxWrites: number): S {
  return {
    ...s,
    log: [
      ...s.log,
      {
        n: s.log.length + 1,
        op: OP_LABEL[op],
        detail,
        idx: idxWrites,
        free: cfg.engine === 'pg' ? pageFree(s, cfg) : BLCKSZ - pageUsed(s, cfg),
        lps: cfg.engine === 'pg' ? s.lps.length : s.undo.filter((u) => !u.purged).length,
      },
    ],
  };
}

/* ------------------------------------------------------------ the drawing */

const LP_STYLE: Record<LpState, { fill: string; text: string }> = {
  normal: { fill: 'var(--viz-neutral)', text: 'LP_NORMAL' },
  redirect: { fill: 'var(--viz-4)', text: 'LP_REDIRECT' },
  dead: { fill: 'var(--viz-critical)', text: 'LP_DEAD' },
  unused: { fill: 'var(--viz-plane)', text: 'LP_UNUSED' },
};

export default function HotChainPruneLab() {
  const [engine, setEngine] = useState<'pg' | 'innodb'>('pg');
  const [ff, setFf] = useState(100);
  const [rowBytes, setRowBytes] = useState(900);
  const [nIdx, setNIdx] = useState(2);
  const [keyed, setKeyed] = useState(false);
  const [snapXid, setSnapXid] = useState<number | null>(null);
  const cfg: Cfg = { engine, ff, rowBytes, nIdx, keyed, snapXid };
  const [s, setS] = useState<S>(() => initial({ engine: 'pg', ff: 100, rowBytes: 900, nIdx: 2, keyed: false, snapXid: null }));
  const [ref, width] = useSize(900);
  const tip = useTip();

  const reset = (over: Partial<Cfg> = {}) => {
    const c = { ...cfg, ...over, snapXid: null };
    setSnapXid(null);
    setS(initial(c));
  };

  const run = (f: (s: S, cfg: Cfg) => S) => setS((cur) => f(cur, cfg));
  const t = tally(s, cfg);
  const free = pageFree(s, cfg);
  const canUpdate = engine === 'innodb' || liveRowsOnPage(s).length > 0;

  /* geometry */
  const rows = engine === 'pg' ? s.lps.length : s.recs.length;
  const rowH = 28;
  const top = 54;
  const idxW = 118;
  const pageX = idxW + 22;
  const lpX = pageX + 10;
  const lpW = 96;
  const tupX = lpX + lpW + 10;
  const tupW = 312;
  const gutterX = tupX + tupW + 6;
  const pageW = lpW + tupW + 20 + 34;
  const rightX = pageX + pageW + 22;
  const svgW = Math.max(width, rightX + 230);
  const height = top + Math.max(rows, 3) * rowH + 96;
  const rowY = (i: number) => top + i * rowH;

  const barY = top + Math.max(rows, 3) * rowH + 16;
  const barW = pageW - 20;
  const liveBytes = t.live * tupLen(rowBytes);
  const deadBytes = t.dead * tupLen(rowBytes);
  const lpBytes = PAGE_HDR + LP_BYTES * s.lps.length;

  const spilled = s.idx.filter((e) => e.page === 1);
  const killedEntries = s.idx.filter((e) => e.killed).length;
  const hotPct = s.nUpd > 0 ? (100 * s.nHot) / s.nUpd : 0;

  return (
    <VizPanel
      title="The update path: HOT chains, redirects, pruning and the two-pass VACUUM"
      subtitle="One 8 KB heap page. Set fillfactor, decide whether the UPDATE touches an indexed column, then stream updates and watch where each new version lands and what it costs the indexes."
      controls={
        <>
          <Segmented
            label="Engine"
            value={engine}
            onChange={(v) => {
              setEngine(v);
              setSnapXid(null);
              setS(initial({ ...cfg, engine: v, snapXid: null }));
            }}
            options={[
              { value: 'pg', label: 'PostgreSQL', title: 'New version in the heap, HOT if it can be' },
              { value: 'innodb', label: 'InnoDB', title: 'Update in place, old values into an undo record' },
            ]}
          />
          <Slider
            label="fillfactor"
            min={60}
            max={100}
            step={5}
            value={ff}
            onChange={(n) => {
              setFf(n);
              reset({ ff: n });
            }}
            format={(n) => `${n}%`}
            disabled={engine === 'innodb'}
          />
          <Slider
            label="row width"
            min={800}
            max={1600}
            step={100}
            value={rowBytes}
            onChange={(n) => {
              setRowBytes(n);
              reset({ rowBytes: n });
            }}
            format={(n) => fmtBytes(n)}
          />
          <Slider label="indexes" min={1} max={5} value={nIdx} onChange={setNIdx} format={(n) => `${n}`} />
          <Check label="UPDATE changes an indexed column" checked={keyed} onChange={setKeyed} />
          <Check
            label="long-running snapshot open"
            checked={snapXid !== null}
            onChange={(b) => setSnapXid(b ? s.xid : null)}
          />
          <Button onClick={() => run(engine === 'pg' ? pgUpdate : ibUpdate)} disabled={!canUpdate} primary>
            UPDATE
          </Button>
          <Button
            onClick={() =>
              setS((cur) => {
                let n = cur;
                for (let i = 0; i < 5; i++) n = engine === 'pg' ? pgUpdate(n, cfg) : ibUpdate(n, cfg);
                return n;
              })
            }
            disabled={!canUpdate}
          >
            UPDATE ×5
          </Button>
          {engine === 'pg' ? (
            <>
              <Button onClick={() => run(pgRead)} title="heap_page_prune_opt runs on ordinary page access">
                SELECT this page
              </Button>
              <Button onClick={() => run(pgScan)}>Index scan</Button>
              <Button onClick={() => run(pgVacuum1)}>VACUUM pass 1</Button>
              <Button onClick={() => run(pgVacuum2)} disabled={!s.vacPass1}>
                index vacuum + pass 2
              </Button>
            </>
          ) : (
            <Button onClick={() => run(ibPurge)}>Purge</Button>
          )}
          <Button onClick={() => reset()}>Reset</Button>
        </>
      }
      legend={
        <Legend
          items={
            engine === 'pg'
              ? [
                  { label: 'live version (xmax = 0)', color: 'var(--viz-clean)' },
                  { label: 'dead version', color: 'var(--viz-stale)' },
                  { label: 'LP_REDIRECT root', color: 'var(--viz-4)' },
                  { label: 'LP_DEAD stub', color: 'var(--viz-critical)' },
                  { label: 'index entry / scan path', color: 'var(--viz-7)' },
                  { label: 'HOT — dashed border, HEAP_ONLY_TUPLE', color: 'var(--viz-ink-2)', shape: 'line' },
                ]
              : [
                  { label: 'clustered-index record (updated in place)', color: 'var(--viz-clean)' },
                  { label: 'undo record', color: 'var(--viz-3)' },
                  { label: 'purged undo', color: 'var(--viz-stale)' },
                  { label: 'secondary index entry', color: 'var(--viz-7)' },
                ]
          }
        />
      }
      stats={
        <Stats
          items={
            engine === 'pg'
              ? [
                  { label: 'n_tup_upd', value: fmtNum(s.nUpd) },
                  { label: 'n_tup_hot_upd', value: fmtNum(s.nHot), hint: 'Updates that wrote no index entry' },
                  { label: 'n_tup_newpage_upd', value: fmtNum(s.nNewpage), hint: 'New version had to go to another block' },
                  { label: 'HOT ratio', value: `${hotPct.toFixed(0)}%`, hint: 'n_tup_hot_upd / n_tup_upd' },
                  { label: 'Index entries written', value: fmtNum(s.idxWrites), hint: `${nIdx} per non-HOT update` },
                  {
                    label: 'Index entries',
                    value: `${fmtNum(s.idx.length * nIdx)}${killedEntries > 0 ? ` (${killedEntries * nIdx} killed)` : ''}`,
                    hint: 'Across all indexes; killed ones are skipped, but only the index vacuum removes any',
                  },
                  { label: 'Free on page', value: fmtBytes(free), hint: 'pd_upper − pd_lower' },
                  { label: 'Line pointers', value: `${s.lps.length} (${t.deadLp} dead, ${t.redirects} redirect)` },
                ]
              : [
                  { label: 'Rows updated', value: fmtNum(s.nUpd) },
                  { label: 'Secondary index ops', value: fmtNum(s.idxWrites), hint: 'Only indexes whose key changed' },
                  { label: 'Undo records', value: fmtNum(s.undo.length) },
                  {
                    label: 'History list length',
                    value: fmtNum(s.undo.filter((u) => !u.purged).length),
                    hint: 'Undo not yet purgeable — what a long read view pins',
                  },
                  { label: 'Delete-marked entries', value: fmtNum(s.delMarked) },
                  { label: 'Page space used', value: fmtBytes(pageUsed(s, cfg)), hint: 'In-place updates do not grow it' },
                ]
          }
        />
      }
      note={
        <Note>
          <strong>{s.head}</strong> {s.body}
        </Note>
      }
      table={
        <>
          <table className="viz-table">
            <thead>
              <tr>
                {engine === 'pg' ? (
                  <>
                    <th>Item</th>
                    <th>lp_flags</th>
                    <th>Row</th>
                    <th>xmin</th>
                    <th>xmax</th>
                    <th>t_ctid</th>
                    <th>Flags</th>
                  </>
                ) : (
                  <>
                    <th>Record</th>
                    <th>DB_TRX_ID</th>
                    <th>DB_ROLL_PTR</th>
                    <th>Undo depth</th>
                  </>
                )}
              </tr>
            </thead>
            <tbody>
              {engine === 'pg'
                ? s.lps.map((lp, i) => (
                    <tr key={i}>
                      <td>(0,{i + 1})</td>
                      <td>
                        {LP_STYLE[lp.state].text}
                        {lp.state === 'redirect' ? ` → ${lp.redirect}` : ''}
                      </td>
                      <td>{lp.ver ? lp.ver.row : '—'}</td>
                      <td>{lp.ver ? lp.ver.xmin : '—'}</td>
                      <td>{lp.ver ? (lp.ver.xmax === 0 ? '0' : lp.ver.xmax) : '—'}</td>
                      <td>{lp.ver ? `(${lp.ver.ctid[0]},${lp.ver.ctid[1]})` : '—'}</td>
                      <td>
                        {lp.ver
                          ? [lp.ver.hot ? 'HEAP_ONLY_TUPLE' : '', lp.ver.hotUpdated ? 'HEAP_HOT_UPDATED' : '']
                              .filter(Boolean)
                              .join(' + ') || '—'
                          : '—'}
                      </td>
                    </tr>
                  ))
                : s.recs.map((r) => (
                    <tr key={r.row}>
                      <td>{r.row}</td>
                      <td>{r.trx}</td>
                      <td>{r.roll === 0 ? 'NULL' : `undo ${r.roll}`}</td>
                      <td>{s.undo.filter((u) => u.row === r.row && !u.purged).length}</td>
                    </tr>
                  ))}
            </tbody>
          </table>
          <table className="viz-table">
            <thead>
              <tr>
                <th>#</th>
                <th>Operation</th>
                <th>What happened</th>
                <th>Index writes</th>
                <th>Free on page</th>
                <th>{engine === 'pg' ? 'Line pointers' : 'History length'}</th>
              </tr>
            </thead>
            <tbody>
              {s.log.length === 0 ? (
                <tr>
                  <td colSpan={6}>Nothing run yet.</td>
                </tr>
              ) : (
                s.log.map((r) => (
                  <tr key={r.n}>
                    <td>{r.n}</td>
                    <td>{r.op}</td>
                    <td>{r.detail}</td>
                    <td>{r.idx}</td>
                    <td>{fmtBytes(r.free)}</td>
                    <td>{r.lps}</td>
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
          <svg
            width={svgW}
            height={height}
            role="img"
            aria-label={
              engine === 'pg'
                ? 'A heap page showing line pointers, tuple versions, HOT chain links and index entries'
                : 'A clustered-index page whose records are updated in place with undo records behind them'
            }
          >
            <defs>
              <marker id="hcp-arrow" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="6" markerHeight="6" orient="auto">
                <path d="M0,0 L8,4 L0,8 z" fill="var(--viz-ink-2)" />
              </marker>
              <marker id="hcp-arrow-hot" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="6" markerHeight="6" orient="auto">
                <path d="M0,0 L8,4 L0,8 z" fill="var(--viz-7)" />
              </marker>
            </defs>

            {/* the page itself */}
            <rect
              x={pageX}
              y={top - 32}
              width={pageW}
              height={Math.max(rows, 3) * rowH + 88}
              rx={8}
              fill="var(--viz-plane)"
              stroke="var(--viz-border)"
            />
            <text x={pageX + 10} y={top - 14} fill="var(--viz-ink)" fontWeight={600}>
              {engine === 'pg' ? 'heap page 0 — 8 KB' : 'clustered index leaf page'}
            </text>
            <text x={pageX + pageW - 10} y={top - 14} textAnchor="end">
              {engine === 'pg' ? `pd_lower ${lpBytes} · free ${fmtBytes(free)}` : `records updated in place`}
            </text>
            <text x={8} y={top - 14} fill="var(--viz-ink)" fontWeight={600}>
              {engine === 'pg' ? `${nIdx} index${nIdx === 1 ? '' : 'es'}` : 'secondary index'}
            </text>
            <text x={rightX} y={top - 14} fill="var(--viz-ink)" fontWeight={600}>
              {engine === 'pg' ? 'elsewhere' : 'undo log'}
            </text>

            {engine === 'pg'
              ? s.lps.map((lp, i) => {
                  const y = rowY(i);
                  const v = lp.ver;
                  const dead = v ? isDead(v, s, cfg) : false;
                  const onPath = s.scanPath.includes(i);
                  const entries = s.idx.filter((e) => e.page === 0 && e.lp === i + 1);
                  const st = LP_STYLE[lp.state];
                  return (
                    <g key={i}>
                      {/* index entries that name this item */}
                      {entries.map((e, k) => (
                        <g key={k} {...tip(<>Index entry for row {e.row} → TID (0,{e.lp}). {e.killed ? 'Marked LP_DEAD by kill_prior_tuple: later scans skip it without touching the heap.' : 'Points at the root item, never at a heap-only tuple.'}</>)} style={{ cursor: 'help' }}>
                          <rect
                            x={8}
                            y={y + 3}
                            width={idxW - 8}
                            height={18}
                            rx={4}
                            fill={e.killed ? 'var(--viz-stale)' : 'var(--viz-7)'}
                            opacity={e.killed ? 0.45 : 0.9}
                          />
                          <text x={14} y={y + 16} fill="var(--viz-surface)">
                            {e.killed ? 'killed' : 'key'} {e.row} → (0,{e.lp})
                          </text>
                          <line
                            x1={idxW}
                            x2={lpX}
                            y1={y + 12}
                            y2={y + 12}
                            stroke={e.killed ? 'var(--viz-stale)' : 'var(--viz-7)'}
                            strokeWidth={onPath ? 2 : 1}
                            markerEnd="url(#hcp-arrow-hot)"
                          />
                        </g>
                      ))}

                      {/* the line pointer */}
                      <g {...tip(<><strong>ItemIdData {i + 1}</strong> — 4 bytes: 15-bit offset, 2-bit flags, 15-bit length. State {st.text}{lp.state === 'redirect' ? `, pointing at item ${lp.redirect}` : ''}.</>)} style={{ cursor: 'help' }}>
                        <rect
                          x={lpX}
                          y={y + 2}
                          width={lpW}
                          height={20}
                          rx={4}
                          fill={st.fill}
                          stroke={onPath ? 'var(--viz-7)' : 'var(--viz-border)'}
                          strokeWidth={onPath ? 2 : 1}
                          strokeDasharray={lp.state === 'unused' ? '3 2' : undefined}
                        />
                        <text
                          x={lpX + 6}
                          y={y + 16}
                          fill={lp.state === 'dead' ? 'var(--viz-surface)' : 'var(--viz-ink)'}
                        >
                          {i + 1}: {st.text.replace('LP_', '')}
                          {lp.state === 'redirect' ? `→${lp.redirect}` : ''}
                        </text>
                      </g>

                      {/* the tuple */}
                      {v ? (
                        <g {...tip(<><strong>Row {v.row}, version {v.ver}</strong> — {fmtBytes(tupLen(rowBytes))} including a {TUP_HDR}-byte header. xmin {v.xmin}, xmax {v.xmax === 0 ? '0 (live)' : v.xmax}, t_ctid ({v.ctid[0]},{v.ctid[1]}). {v.hot ? 'HEAP_ONLY_TUPLE: no index entry names it. ' : ''}{v.hotUpdated ? 'HEAP_HOT_UPDATED: t_ctid is a chain link. ' : ''}{dead ? 'Dead to every snapshot — prunable.' : ''}</>)} style={{ cursor: 'help' }}>
                          <rect
                            x={tupX}
                            y={y + 2}
                            width={tupW}
                            height={20}
                            rx={4}
                            fill={dead ? 'var(--viz-stale)' : 'var(--viz-clean)'}
                            opacity={dead ? 0.55 : 0.92}
                            stroke={v.hot ? 'var(--viz-ink)' : 'transparent'}
                            strokeWidth={v.hot ? 1.5 : 0}
                            strokeDasharray={v.hot ? '4 3' : undefined}
                          />
                          <text x={tupX + 8} y={y + 16} fill="var(--viz-surface)">
                            row {v.row} v{v.ver} · xmin {v.xmin} · xmax {v.xmax === 0 ? '0' : v.xmax} · t_ctid (
                            {v.ctid[0]},{v.ctid[1]}){v.hot ? ' · HOT' : ''}
                            {v.hotUpdated ? ' · HHU' : ''}
                          </text>
                        </g>
                      ) : (
                        <text x={tupX + 8} y={y + 16}>
                          {lp.state === 'dead'
                            ? 'tuple bytes reclaimed — stub kept for the index entries'
                            : lp.state === 'redirect'
                              ? 'no tuple: the item is a pointer to the chain head'
                              : 'free slot'}
                        </text>
                      )}

                      {/* chain link drawn in the gutter */}
                      {v && v.hotUpdated && v.ctid[0] === 0 && v.ctid[1] - 1 !== i ? (
                        <path
                          d={`M${gutterX},${y + 12} C${gutterX + 26},${y + 12} ${gutterX + 26},${rowY(v.ctid[1] - 1) + 12} ${gutterX + 2},${rowY(v.ctid[1] - 1) + 12}`}
                          fill="none"
                          stroke={onPath ? 'var(--viz-7)' : 'var(--viz-ink-2)'}
                          strokeWidth={onPath ? 2 : 1}
                          markerEnd={onPath ? 'url(#hcp-arrow-hot)' : 'url(#hcp-arrow)'}
                        />
                      ) : null}
                      {lp.state === 'redirect' ? (
                        <path
                          d={`M${gutterX},${y + 12} C${gutterX + 26},${y + 12} ${gutterX + 26},${rowY(lp.redirect - 1) + 12} ${gutterX + 2},${rowY(lp.redirect - 1) + 12}`}
                          fill="none"
                          stroke={onPath ? 'var(--viz-7)' : 'var(--viz-4)'}
                          strokeWidth={2}
                          markerEnd={onPath ? 'url(#hcp-arrow-hot)' : 'url(#hcp-arrow)'}
                        />
                      ) : null}
                      {v && v.xmax !== 0 && v.ctid[0] === 1 ? (
                        <line
                          x1={gutterX}
                          x2={rightX - 4}
                          y1={y + 12}
                          y2={y + 12}
                          stroke="var(--viz-ink-2)"
                          strokeWidth={1}
                          strokeDasharray="4 3"
                          markerEnd="url(#hcp-arrow)"
                        />
                      ) : null}
                    </g>
                  );
                })
              : s.recs.map((r, i) => {
                  const y = rowY(i);
                  const chain = s.undo.filter((u) => u.row === r.row);
                  return (
                    <g key={r.row}>
                      <rect x={8} y={y + 3} width={idxW - 8} height={18} rx={4} fill="var(--viz-7)" opacity={keyed ? 0.9 : 0.35} />
                      <text x={14} y={y + 16} fill="var(--viz-surface)">
                        key {r.row} → PK {r.row}
                      </text>
                      <g {...tip(<><strong>Record {r.row}</strong> — updated in place. DB_TRX_ID {r.trx}, DB_ROLL_PTR {r.roll === 0 ? 'NULL' : `undo ${r.roll}`}. A read older than {r.trx} rebuilds its version by following the roll pointer.</>)} style={{ cursor: 'help' }}>
                        <rect x={lpX} y={y + 2} width={lpW + tupW + 10} height={20} rx={4} fill="var(--viz-clean)" opacity={0.92} />
                        <text x={lpX + 8} y={y + 16} fill="var(--viz-surface)">
                          PK {r.row} · DB_TRX_ID {r.trx} · DB_ROLL_PTR {r.roll === 0 ? 'NULL' : `→ undo ${r.roll}`} ·{' '}
                          {fmtBytes(tupLen(rowBytes))}
                        </text>
                      </g>
                      {chain.length > 0 ? (
                        <line x1={gutterX + 12} x2={rightX - 4} y1={y + 12} y2={y + 12} stroke="var(--viz-ink-2)" strokeWidth={1} markerEnd="url(#hcp-arrow)" />
                      ) : null}
                      {chain.slice(-6).map((u, k) => (
                        <g key={u.id} {...tip(<>Undo record {u.id} for row {u.row}: the column values as of DB_TRX_ID {u.trx}. {u.purged ? 'Purged.' : 'Still needed by some read view.'}</>)} style={{ cursor: 'help' }}>
                          <rect
                            x={rightX + k * 34}
                            y={y + 4}
                            width={30}
                            height={16}
                            rx={3}
                            fill={u.purged ? 'var(--viz-stale)' : 'var(--viz-3)'}
                            opacity={u.purged ? 0.4 : 0.9}
                          />
                          <text x={rightX + k * 34 + 4} y={y + 16} fill="var(--viz-surface)">
                            u{u.id}
                          </text>
                        </g>
                      ))}
                    </g>
                  );
                })}

            {/* off-page versions */}
            {engine === 'pg' && spilled.length > 0 ? (
              <g>
                <rect x={rightX} y={top - 4} width={200} height={Math.min(spilled.length, 6) * 22 + 12} rx={6} fill="var(--viz-plane)" stroke="var(--viz-border)" />
                {spilled.slice(0, 6).map((e, k) => (
                  <g key={k}>
                    <rect x={rightX + 8} y={top + 4 + k * 22} width={184} height={18} rx={4} fill="var(--viz-clean)" opacity={0.92} />
                    <text x={rightX + 14} y={top + 17 + k * 22} fill="var(--viz-surface)">
                      row {e.row} on another page · {nIdx} index entr{nIdx === 1 ? 'y' : 'ies'}
                    </text>
                  </g>
                ))}
              </g>
            ) : null}

            {/* free-space bar */}
            {engine === 'pg' ? (
              <g>
                <rect x={pageX + 10} y={barY} width={barW} height={16} rx={3} fill="var(--viz-neutral)" />
                <rect x={pageX + 10} y={barY} width={(barW * lpBytes) / BLCKSZ} height={16} rx={3} fill="var(--viz-ink-2)" />
                <rect
                  x={pageX + 10 + (barW * lpBytes) / BLCKSZ}
                  y={barY}
                  width={(barW * liveBytes) / BLCKSZ}
                  height={16}
                  fill="var(--viz-clean)"
                  opacity={0.9}
                />
                <rect
                  x={pageX + 10 + (barW * (lpBytes + liveBytes)) / BLCKSZ}
                  y={barY}
                  width={(barW * deadBytes) / BLCKSZ}
                  height={16}
                  fill="var(--viz-stale)"
                  opacity={0.6}
                />
                <line
                  x1={pageX + 10 + (barW * ff) / 100}
                  x2={pageX + 10 + (barW * ff) / 100}
                  y1={barY - 5}
                  y2={barY + 21}
                  stroke="var(--viz-critical)"
                  strokeWidth={1.5}
                  strokeDasharray="4 3"
                />
                <text x={pageX + 10 + (barW * ff) / 100 + 5} y={barY + 30} fill="var(--viz-critical)">
                  fillfactor {ff}% — the insert limit, not the update limit
                </text>
                <text x={pageX + 10} y={barY + 30}>
                  header + line pointers · live · dead · free
                </text>
              </g>
            ) : null}
          </svg>
        </TooltipHost>
      </div>
    </VizPanel>
  );
}
