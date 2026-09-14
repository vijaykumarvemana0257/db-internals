import { useMemo, useState } from 'react';
import {
  VizPanel,
  Segmented,
  Slider,
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
 * One page, three engines, the same rows.
 *
 * The learner edits a single heap/leaf page and watches the two structures that make a
 * slotted page work move against each other: a pointer array growing from the low end and
 * record bytes growing from the high end, with free space defined as the gap between them.
 *
 * The whole history is kept as a list of logical operations and *replayed* for whichever
 * engine is selected, so switching PostgreSQL -> InnoDB -> SQLite re-lays-out the same rows
 * under a different set of rules:
 *
 *   PostgreSQL  24 B PageHeaderData, 4 B ItemIdData line pointers in insertion order,
 *               MAXALIGN(8) tuples growing down from pd_special. Holes left by pruning are
 *               NOT reusable until PageRepairFragmentation runs. Slot numbers are the
 *               second half of a TID and therefore never renumbered.
 *   InnoDB      16 KB page, records growing UP from the page header into the heap and a
 *               2 B-per-slot page directory growing DOWN from the FIL trailer; records are
 *               linked in key order between the infimum and supremum sentinels; freed
 *               records go on PAGE_FREE and are reusable at once.
 *   SQLite      4 KB page, 8 B b-tree header, 2 B cell pointers kept sorted by key, cells
 *               growing down from the end, a freeblock chain plus a 1-byte fragmented-bytes
 *               counter for gaps too small to chain.
 */

/* ------------------------------------------------------------------- engines */

type Engine = 'pg' | 'innodb' | 'sqlite';

type Spec = {
  id: Engine;
  label: string;
  page: number;
  frontFixed: number;
  frontLabel: string;
  backFixed: number;
  backLabel: string;
  ptrBytes: number;
  ptrLabel: string;
  ptrRegion: string;
  recOverhead: number;
  align: number;
  dataAtEnd: boolean; // records grow down from the high end of the page
  reuseHoles: boolean; // can an insert take a hole without a defragmentation pass first
  reclaim: string; // second-phase button label ('' = no such phase)
  compact: string;
  lowName: string;
  highName: string;
  recName: string;
};

const SPECS: Record<Engine, Spec> = {
  pg: {
    id: 'pg',
    label: 'PostgreSQL heap',
    page: 8192,
    frontFixed: 24,
    frontLabel: 'PageHeaderData, 24 B',
    backFixed: 0,
    backLabel: '',
    ptrBytes: 4,
    ptrLabel: 'ItemIdData',
    ptrRegion: 'line pointer array',
    recOverhead: 24, // MAXALIGN'd HeapTupleHeaderData: 23 B rounded to t_hoff = 24
    align: 8,
    dataAtEnd: true,
    reuseHoles: false,
    reclaim: 'Prune',
    compact: 'Compact',
    lowName: 'pd_lower',
    highName: 'pd_upper',
    recName: 'tuple',
  },
  innodb: {
    id: 'innodb',
    label: 'InnoDB index page',
    page: 16384,
    frontFixed: 120, // 38 FIL + 36 index header + 20 FSEG header + 26 infimum/supremum
    frontLabel: 'FIL + index + FSEG headers, infimum/supremum',
    backFixed: 8,
    backLabel: 'FIL trailer, 8 B',
    ptrBytes: 2,
    ptrLabel: 'directory slot',
    ptrRegion: 'page directory',
    recOverhead: 18, // 5 B COMPACT record header + DB_TRX_ID 6 + DB_ROLL_PTR 7
    align: 1,
    dataAtEnd: false,
    reuseHoles: true,
    reclaim: 'Purge',
    compact: 'Reorganize',
    lowName: 'PAGE_HEAP_TOP',
    highName: 'directory start',
    recName: 'record',
  },
  sqlite: {
    id: 'sqlite',
    label: 'SQLite table leaf',
    page: 4096,
    frontFixed: 8,
    frontLabel: 'b-tree page header, 8 B',
    backFixed: 0,
    backLabel: '',
    ptrBytes: 2,
    ptrLabel: 'cell pointer',
    ptrRegion: 'cell pointer array',
    recOverhead: 5, // payload-size varint + rowid varint + record header bytes
    align: 1,
    dataAtEnd: true,
    reuseHoles: true,
    reclaim: '',
    compact: 'Defragment',
    lowName: 'cell pointer end',
    highName: 'cell content start',
    recName: 'cell',
  },
};

/* ----------------------------------------------------- deterministic inputs */

/** Per-row size jitter, so the page holds genuinely variable-length records. */
const JITTER = (() => {
  const rng = makeRng(8192);
  return Array.from({ length: 64 }, () => 0.72 + rng() * 0.56);
})();

/** Primary keys in a deliberately non-monotonic order: heap order != key order. */
const KEYS = (() => {
  const rng = makeRng(1729);
  const a = Array.from({ length: 64 }, (_, i) => 101 + i);
  for (let i = a.length - 1; i > 0; i--) {
    const j = Math.floor(rng() * (i + 1));
    [a[i], a[j]] = [a[j], a[i]];
  }
  return a;
})();

/* --------------------------------------------------------------- operations */

type Op =
  | { kind: 'insert'; size: number }
  | { kind: 'update'; id: number; size: number; hot: boolean }
  | { kind: 'delete'; id: number }
  | { kind: 'reclaim' }
  | { kind: 'compact' }
  | { kind: 'vacuum' };

type Rec = {
  id: number; // logical row identity, stable across engines
  key: number;
  ver: number;
  off: number;
  len: number;
  dead: boolean; // deleted / delete-marked; the bytes are still on the page
  slot: number; // PostgreSQL line-pointer number; heap sequence elsewhere
  succ: number | null; // HOT successor's slot number
};

type Slot = {
  n: number;
  flag: 'normal' | 'dead' | 'unused' | 'redirect';
  rec: number | null;
  target: number | null;
};

type Extent = { off: number; len: number };

type Ptr = {
  key: string;
  label: string;
  glyph: string;
  kind: 'normal' | 'dead' | 'unused' | 'redirect' | 'dir' | 'sentinel';
  recId: number | null;
  target: number | null;
  hint: string;
};

type Page = {
  spec: Spec;
  recs: Rec[];
  slots: Slot[];
  ptrs: Ptr[];
  holes: Extent[];
  low: number; // end of the low-growing region
  high: number; // start of the high-growing region
  frag: number; // SQLite fragmented free bytes
  garbage: number; // InnoDB PAGE_GARBAGE
  refused: number;
  head: string;
  body: string;
  order: number[]; // record ids in key order (the InnoDB record list)
};

const align = (n: number, a: number) => (a <= 1 ? n : Math.ceil(n / a) * a);

function build(engine: Engine, ops: Op[], seedCount: number): Page {
  const spec = SPECS[engine];
  const recs: Rec[] = [];
  let slots: Slot[] = [];
  let holes: Extent[] = [];
  let contentStart = spec.page - spec.backFixed; // pd_upper / cell content area start
  let heapTop = spec.frontFixed; // InnoDB PAGE_HEAP_TOP
  let frag = 0;
  let garbage = 0;
  let refused = 0;
  let nextId = 1;
  let head = 'Six rows inserted.';
  let body =
    `Every record sits in a page of ${fmtBytes(spec.page)}. Insert, update and delete rows and watch the ` +
    'pointer array and the record bytes grow towards each other; the gap between them is the free space.';

  const live = () => recs.filter((r) => !r.dead);
  const dirSlots = () => 2 + Math.ceil(live().length / 8);
  const ptrCount = () => (engine === 'pg' ? slots.length : engine === 'sqlite' ? live().length : dirSlots());
  const lowBound = (extraPtrs = 0) =>
    engine === 'innodb' ? heapTop : spec.frontFixed + spec.ptrBytes * (ptrCount() + extraPtrs);
  const highBound = (extraPtrs = 0) =>
    engine === 'innodb' ? spec.page - spec.backFixed - spec.ptrBytes * (ptrCount() + extraPtrs) : contentStart;

  /** Free contiguous bytes between the two growing regions. */
  const gap = (extraPtrs = 0) => highBound(extraPtrs) - lowBound(extraPtrs);

  const defragment = () => {
    const sorted = [...recs].sort((a, b) =>
      engine === 'innodb' ? a.key - b.key : engine === 'sqlite' ? a.key - b.key : b.off - a.off,
    );
    if (spec.dataAtEnd) {
      let cursor = spec.page - spec.backFixed;
      for (const r of sorted) {
        cursor -= r.len;
        r.off = cursor;
      }
      contentStart = cursor;
    } else {
      let cursor = spec.frontFixed;
      for (const r of sorted) {
        r.off = cursor;
        cursor += r.len;
      }
      heapTop = cursor;
    }
    holes = [];
    frag = 0;
    garbage = 0;
  };

  /** Place len bytes, reusing a hole where the engine allows it. Returns the offset. */
  const place = (len: number, extraPtrs: number): number | null => {
    if (spec.reuseHoles) {
      const i = holes.findIndex((h) => h.len >= len);
      if (i >= 0) {
        const h = holes[i];
        const off = h.off;
        if (engine === 'innodb') {
          // InnoDB takes the freed record's space; anything left over stays on the free list.
          garbage = Math.max(0, garbage - len);
          if (h.len - len >= 8) {
            h.off += len;
            h.len -= len;
          } else {
            garbage = Math.max(0, garbage - (h.len - len));
            holes.splice(i, 1);
          }
        } else {
          h.off += len;
          h.len -= len;
          if (h.len < 4) {
            frag += h.len; // gaps under 4 bytes cannot be chained as freeblocks
            holes.splice(i, 1);
          }
        }
        return off;
      }
    }
    if (gap(extraPtrs) >= len) {
      if (spec.dataAtEnd) {
        contentStart -= len;
        return contentStart;
      }
      const off = heapTop;
      heapTop += len;
      return off;
    }
    if (engine === 'sqlite' && (holes.length > 0 || frag > 0)) {
      // SQLite defragments the page rather than failing, then retries the allocation.
      defragment();
      if (gap(extraPtrs) >= len) {
        contentStart -= len;
        return contentStart;
      }
    }
    return null;
  };

  const freeBytes = (r: Rec) => {
    if (engine === 'sqlite' && r.len < 4) {
      frag += r.len;
      return;
    }
    holes.push({ off: r.off, len: r.len });
    holes.sort((a, b) => a.off - b.off);
    // coalesce neighbours
    for (let i = holes.length - 1; i > 0; i--) {
      if (holes[i - 1].off + holes[i - 1].len === holes[i].off) {
        holes[i - 1].len += holes[i].len;
        holes.splice(i, 1);
      }
    }
    if (engine === 'innodb') garbage += r.len;
  };

  const drop = (r: Rec) => {
    const i = recs.indexOf(r);
    if (i >= 0) recs.splice(i, 1);
  };

  for (let o = 0; o < ops.length; o++) {
    const op = ops[o];
    const seeding = o < seedCount;
    switch (op.kind) {
      case 'insert': {
        const id = nextId++;
        const payload = Math.max(8, Math.round(op.size * JITTER[id % JITTER.length]));
        const len = align(spec.recOverhead + payload, spec.align);
        const reuse = engine === 'pg' ? slots.findIndex((s) => s.flag === 'unused') : -1;
        const extraPtrs = engine === 'pg' ? (reuse >= 0 ? 0 : 1) : engine === 'sqlite' ? 1 : 0;
        const off = place(len, extraPtrs);
        if (off === null) {
          refused++;
          nextId--;
          if (!seeding) {
            head = `INSERT refused: ${fmtBytes(len)} does not fit.`;
            body =
              `Only ${gap(extraPtrs)} B of contiguous free space is left` +
              (holes.length ? `, plus ${fmtBytes(holes.reduce((a, h) => a + h.len, 0))} stranded in holes` : '') +
              `. ${
                engine === 'pg'
                  ? 'PageAddItem returns InvalidOffsetNumber, the executor consults the free space map and puts the tuple on some other page.'
                  : engine === 'innodb'
                    ? 'InnoDB reorganizes the page, and if it still does not fit, splits it.'
                    : 'SQLite balances the b-tree and moves cells to a sibling page.'
              }`;
          }
          break;
        }
        let slotNo: number;
        if (engine === 'pg') {
          if (reuse >= 0) {
            slotNo = slots[reuse].n;
            slots[reuse] = { n: slotNo, flag: 'normal', rec: id, target: null };
          } else {
            slotNo = slots.length + 1;
            slots.push({ n: slotNo, flag: 'normal', rec: id, target: null });
          }
        } else {
          slotNo = recs.length + 1;
        }
        recs.push({ id, key: KEYS[id % KEYS.length], ver: 1, off, len, dead: false, slot: slotNo, succ: null });
        if (!seeding) {
          head = `INSERT → row ${id}${engine === 'pg' ? ` at TID (0,${slotNo})` : ''}`;
          body =
            engine === 'pg'
              ? `${len} B tuple = 24 B header + ${payload} B of data, MAXALIGN'd to 8. ${
                  reuse >= 0
                    ? `Line pointer ${slotNo} was LP_UNUSED and got reused, so pd_lower did not move.`
                    : `A new 4 B ItemIdData pushed pd_lower to ${lowBound()}.`
                } pd_upper fell to ${contentStart}; free space is pd_upper − pd_lower = ${gap()} B.`
              : engine === 'innodb'
                ? `${len} B record = 5 B COMPACT header + 6 B DB_TRX_ID + 7 B DB_ROLL_PTR + ${payload} B of columns, ` +
                  `allocated at ${off}${off < heapTop - len ? ' out of the PAGE_FREE list' : ' at PAGE_HEAP_TOP'} and linked ` +
                  `into the record list in key order (key ${KEYS[id % KEYS.length]}). The directory now holds ${dirSlots()} slots.`
                : `${len} B cell = ${payload} B payload + varint length + rowid. The 2 B cell pointer is inserted in key ` +
                  `order, so every pointer after it shifted by two bytes — the cells themselves did not move.`;
        }
        break;
      }
      case 'update': {
        const r = recs.find((x) => x.id === op.id && !x.dead);
        if (!r) break;
        const payload = Math.max(8, Math.round(op.size * JITTER[(r.id + 7) % JITTER.length]));
        const len = align(spec.recOverhead + payload, spec.align);
        if (engine !== 'pg' && len <= r.len) {
          // A same-or-smaller record is rewritten where it stands.
          const slack = r.len - len;
          if (slack > 0) {
            if (engine === 'sqlite') frag += slack;
            else garbage += slack;
          }
          r.len = len;
          r.ver++;
          head = `UPDATE row ${r.id} in place`;
          body =
            engine === 'innodb'
              ? `The new version is no larger, so InnoDB overwrites the record where it sits and pushes the old column ` +
                `values into the undo log; the record's position, its list links and the directory are untouched.`
              : `The new cell fits inside the old one, so SQLite overwrites it; the leftover ${slack} B are counted ` +
                `as fragmented free bytes.`;
          break;
        }
        const extraPtrs = engine === 'pg' ? 1 : engine === 'sqlite' ? 0 : 0;
        const off = place(len, extraPtrs);
        if (off === null) {
          refused++;
          head = `UPDATE row ${r.id}: no room on this page`;
          body =
            engine === 'pg'
              ? `A Postgres UPDATE is an insert plus a delete, and the new version needs ${len} B the page does not have. ` +
                `The new tuple lands on another page, which means a new index entry for every index on the table — and ` +
                `no HOT chain, because HOT requires both versions on the same page.`
              : 'The larger version does not fit, so the page must be reorganized or split before the row can grow.';
          break;
        }
        if (engine === 'pg') {
          const newSlot = (() => {
            const reuse = slots.findIndex((s) => s.flag === 'unused');
            if (reuse >= 0) {
              slots[reuse] = { n: slots[reuse].n, flag: 'normal', rec: -1, target: null };
              return slots[reuse].n;
            }
            const n = slots.length + 1;
            slots.push({ n, flag: 'normal', rec: -1, target: null });
            return n;
          })();
          const id = nextId++;
          recs.push({ id, key: r.key, ver: r.ver + 1, off, len, dead: false, slot: newSlot, succ: null });
          const s = slots.find((x) => x.n === newSlot)!;
          s.rec = id;
          r.dead = true;
          r.succ = op.hot ? newSlot : null;
          head = `UPDATE row ${r.id} → new version at (0,${newSlot})`;
          body = op.hot
            ? `Heap-only tuple: no indexed column changed, so the new version gets a line pointer but no index entry, ` +
              `and the old tuple's t_ctid points at slot ${newSlot}. Prune the page and slot ${r.slot} becomes ` +
              `LP_REDIRECT → ${newSlot}, keeping the old TID valid for every index that still points at it.`
            : `An indexed column changed, so this is a plain update: a second tuple, a second line pointer, and a new ` +
              `entry in every index. The old tuple stays until it is pruned.`;
        } else {
          freeBytes(r);
          r.off = off;
          r.len = len;
          r.ver++;
          head = `UPDATE row ${r.id} relocated`;
          body = `The new version is larger than the old one, so the record is freed and re-allocated at ${off}. ${
            engine === 'innodb'
              ? 'Its key never changed, so the list links and every secondary index entry — which store the primary key, not a position — are still correct.'
              : 'The cell pointer for this row is rewritten with the new offset; every other pointer is untouched.'
          }`;
        }
        break;
      }
      case 'delete': {
        const r = recs.find((x) => x.id === op.id && !x.dead);
        if (!r) break;
        if (engine === 'sqlite') {
          freeBytes(r);
          drop(r);
          head = `DELETE row ${r.id}`;
          body =
            `SQLite frees the cell immediately: ${r.len} B ${
              r.len >= 4 ? 'go on the freeblock chain in offset order' : 'are too small to chain and are added to the fragmented-byte counter'
            }, and the cell pointer array closes up behind it. There is no second phase and no dead row.`;
          break;
        }
        r.dead = true;
        head = `DELETE row ${r.id}`;
        body =
          engine === 'pg'
            ? `Nothing is freed. The tuple's xmax is stamped with the deleting transaction and the line pointer stays ` +
              `LP_NORMAL; the bytes are still readable by any snapshot older than the delete. Pruning is what reclaims them.`
            : `The record is delete-marked in its header and stays linked into the record list. Purge removes it once no ` +
              `read view can still need the old version.`;
        break;
      }
      case 'reclaim': {
        if (engine === 'sqlite') {
          head = 'SQLite has no second phase';
          body = 'A deleted cell is freed at delete time; there is no dead-row state to reclaim later.';
          break;
        }
        const dead = recs.filter((r) => r.dead);
        if (dead.length === 0) {
          head = engine === 'pg' ? 'Nothing to prune' : 'Nothing to purge';
          body = 'No dead versions on this page.';
          break;
        }
        let redirects = 0;
        for (const r of dead) {
          freeBytes(r);
          if (engine === 'pg') {
            const s = slots.find((x) => x.n === r.slot);
            if (s) {
              if (r.succ !== null) {
                s.flag = 'redirect';
                s.target = r.succ;
                s.rec = null;
                redirects++;
              } else {
                s.flag = 'dead';
                s.rec = null;
                s.target = null;
              }
            }
          }
          drop(r);
        }
        head = engine === 'pg' ? `Pruned ${dead.length} dead tuple${dead.length === 1 ? '' : 's'}` : `Purged ${dead.length} record${dead.length === 1 ? '' : 's'}`;
        body =
          engine === 'pg'
            ? `${fmtBytes(dead.reduce((a, r) => a + r.len, 0))} of tuple storage is now free — and completely unusable, ` +
              `because PageAddItem only allocates between pd_lower and pd_upper. ${redirects > 0 ? `${redirects} line pointer${redirects === 1 ? '' : 's'} became LP_REDIRECT; ` : ''}` +
              `the rest became LP_DEAD, which must stay until VACUUM has removed the index entries pointing at them. ` +
              `Real pruning ends by calling PageRepairFragmentation — press Compact to see that half.`
            : `${fmtBytes(dead.reduce((a, r) => a + r.len, 0))} went onto the PAGE_FREE list and into PAGE_GARBAGE. ` +
              `Unlike Postgres, InnoDB can hand those bytes straight to the next record that fits.`;
        break;
      }
      case 'compact': {
        const before = holes.reduce((a, h) => a + h.len, 0) + frag;
        defragment();
        head =
          engine === 'pg'
            ? 'PageRepairFragmentation'
            : engine === 'innodb'
              ? 'btr_page_reorganize'
              : 'defragmentPage';
        body =
          engine === 'pg'
            ? `The page is rebuilt in place under an exclusive buffer lock: the live tuples are sorted by offset, memmoved ` +
              `towards pd_special, and pd_upper is recomputed to ${contentStart}. ${fmtBytes(before)} of holes became ` +
              `contiguous free space. Every lp_off was rewritten — and not one slot number changed, so every TID in every ` +
              `index still resolves.`
            : engine === 'innodb'
              ? `The page is rebuilt from a copy: records are written out in key order, PAGE_GARBAGE returns to free space ` +
                `and the directory is regenerated. Heap positions change freely, because nothing outside the page addresses ` +
                `a record by position — secondary indexes store the primary key.`
              : `Cells are repacked against the end of the page in cell-pointer order, the freeblock chain is emptied and ` +
                `the fragmented-byte counter returns to zero. ${fmtBytes(before)} of scattered free space became one run.`;
        break;
      }
      case 'vacuum': {
        if (engine !== 'pg') {
          head = 'No equivalent here';
          body = 'LP_DEAD line pointers are a PostgreSQL construct: they exist because index entries address tuples by TID.';
          break;
        }
        const deadSlots = slots.filter((s) => s.flag === 'dead').length;
        slots = slots.map((s) => (s.flag === 'dead' ? { ...s, flag: 'unused', rec: null, target: null } : s));
        let trimmed = 0;
        while (slots.length > 0 && slots[slots.length - 1].flag === 'unused') {
          slots.pop();
          trimmed++;
        }
        head = `VACUUM: ${deadSlots} LP_DEAD → LP_UNUSED`;
        body =
          `Vacuum's second heap pass runs only after the index entries pointing at those TIDs are gone, which is the whole ` +
          `reason LP_DEAD exists. ${
            trimmed > 0
              ? `${trimmed} trailing line pointer${trimmed === 1 ? '' : 's'} were truncated off the end of the array, pulling pd_lower back to ${lowBound()}.`
              : 'Unused pointers in the middle of the array cannot be removed — only trailing ones — so pd_lower stays where it is.'
          }`;
        break;
      }
    }
  }

  // ---- the pointer lane, per engine
  const liveRecs = recs.filter((r) => !r.dead);
  const byKey = [...liveRecs].sort((a, b) => a.key - b.key);
  const ptrs: Ptr[] = [];
  if (engine === 'pg') {
    for (const s of slots) {
      const r = recs.find((x) => x.id === s.rec);
      ptrs.push({
        key: `s${s.n}`,
        label: String(s.n),
        glyph: s.flag === 'normal' ? 'N' : s.flag === 'dead' ? 'D' : s.flag === 'redirect' ? 'R' : 'U',
        kind: s.flag,
        recId: r ? r.id : null,
        target: s.target,
        hint:
          s.flag === 'normal'
            ? `LP_NORMAL — lp_off ${r ? r.off : '?'}, lp_len ${r ? r.len : '?'}. TID (0,${s.n}) is what every index entry stores.`
            : s.flag === 'redirect'
              ? `LP_REDIRECT — lp_off holds slot ${s.target}, lp_len 0. Old TIDs still resolve; the chase costs one extra hop.`
              : s.flag === 'dead'
                ? 'LP_DEAD — the tuple is gone but the pointer must stay until VACUUM has removed the index entries that address it.'
                : 'LP_UNUSED — 4 bytes of reusable pointer. The next insert on this page takes it.',
      });
    }
  } else if (engine === 'sqlite') {
    byKey.forEach((r, i) => {
      ptrs.push({
        key: `c${r.id}`,
        label: String(i),
        glyph: `k${r.key}`,
        kind: 'normal',
        recId: r.id,
        target: null,
        hint: `Cell pointer ${i}: a 2-byte offset (${r.off}) to the cell for key ${r.key}. The array is kept in key order, so a binary search over it finds a row without touching the cells.`,
      });
    });
  } else {
    ptrs.push({
      key: 'inf',
      label: 'inf',
      glyph: '⊣',
      kind: 'sentinel',
      recId: null,
      target: null,
      hint: 'The infimum record: a fixed sentinel that owns the first directory slot and is the head of the record list. It sorts below every real key.',
    });
    for (let g = 0; g < Math.ceil(byKey.length / 8); g++) {
      const group = byKey.slice(g * 8, g * 8 + 8);
      const owner = group[group.length - 1];
      ptrs.push({
        key: `d${g}`,
        label: `d${g + 1}`,
        glyph: `${group.length}`,
        kind: 'dir',
        recId: owner ? owner.id : null,
        target: null,
        hint: `Directory slot ${g + 1}: a 2-byte offset to the record with key ${owner ? owner.key : '?'}, which owns ${group.length} record${group.length === 1 ? '' : 's'} (n_owned). A search binary-searches the directory, then walks at most 8 links.`,
      });
    }
    ptrs.push({
      key: 'sup',
      label: 'sup',
      glyph: '⊢',
      kind: 'sentinel',
      recId: null,
      target: null,
      hint: 'The supremum record: the tail sentinel. It sorts above every real key, so a scan always terminates on it instead of on a length check.',
    });
  }

  return {
    spec,
    recs,
    slots,
    ptrs,
    holes,
    low: lowBound(),
    high: highBound(),
    frag,
    garbage,
    refused,
    head,
    body,
    order: byKey.map((r) => r.id),
  };
}

/* ------------------------------------------------------------------ drawing */

const FILL = {
  fixed: 'var(--viz-7)',
  ptr: 'var(--viz-2)',
  live: 'var(--viz-1)',
  dead: 'var(--viz-stale)',
  hole: 'var(--viz-4)',
  free: 'var(--viz-neutral)',
};

const SEED_SIZE = 520;
const SEED: Op[] = Array.from({ length: 6 }, () => ({ kind: 'insert', size: SEED_SIZE }) as Op);

export default function SlottedPageEditor() {
  const [engine, setEngine] = useState<Engine>('pg');
  const [size, setSize] = useState(SEED_SIZE);
  const [hot, setHot] = useState(false);
  const [ops, setOps] = useState<Op[]>(SEED);
  const [sel, setSel] = useState<number | null>(null);
  const [ref, width] = useSize(760);
  const tip = useTip();

  const page = useMemo(() => build(engine, ops, SEED.length), [engine, ops]);
  const spec = page.spec;
  const liveIds = page.recs.filter((r) => !r.dead).map((r) => r.id);
  const target = sel !== null && liveIds.includes(sel) ? sel : (liveIds[0] ?? null);
  const push = (op: Op) => setOps((cur) => [...cur, op]);

  const holeBytes = page.holes.reduce((a, h) => a + h.len, 0);
  const freeGap = Math.max(0, page.high - page.low);
  const overhead = spec.frontFixed + spec.backFixed + spec.ptrBytes * page.ptrs.length;
  const liveBytes = page.recs.filter((r) => !r.dead).reduce((a, r) => a + r.len, 0);
  const deadBytes = page.recs.filter((r) => r.dead).reduce((a, r) => a + r.len, 0);
  const biggest =
    engine === 'pg'
      ? Math.max(0, freeGap - spec.ptrBytes - spec.recOverhead)
      : Math.max(0, Math.max(freeGap - spec.ptrBytes, ...page.holes.map((h) => h.len)) - spec.recOverhead);

  /* geometry */
  const PAD = 8;
  const svgW = Math.max(width, 560);
  const inner = svgW - PAD * 2;
  const nptr = Math.max(page.ptrs.length, 1);
  const boxW = Math.max(16, Math.min(54, inner / nptr - 4));
  const ptrY = 22;
  const ptrH = 26;
  const ribY = 108;
  const ribH = 48;
  const listY = ribY + ribH + 26;
  const hasList = engine === 'innodb';
  const axisY = (hasList ? listY + 22 : ribY + ribH + 16) + 10;
  const height = axisY + 22;
  const x = (b: number) => PAD + (b / spec.page) * inner;
  const w = (b: number) => Math.max(1.5, (b / spec.page) * inner);
  const boxX = (i: number) => PAD + i * (inner / nptr) + (inner / nptr - boxW) / 2;
  const recX = (r: Rec) => x(r.off) + w(r.len) / 2;

  const ptrFill = (k: Ptr['kind']) =>
    k === 'normal' || k === 'dir' ? FILL.ptr : k === 'redirect' ? 'var(--viz-3)' : k === 'dead' ? FILL.dead : k === 'sentinel' ? 'var(--viz-7)' : 'var(--viz-plane)';

  const ticks = [0, 0.25, 0.5, 0.75, 1].map((f) => f * spec.page);

  return (
    <VizPanel
      title="One page, edited"
      subtitle={`${spec.label}, ${fmtBytes(spec.page)}. A pointer array grows from one end, record bytes from the other, and free space is whatever is left between them. Switch engines to re-lay-out the same rows under different rules.`}
      controls={
        <>
          <Segmented
            label="Engine"
            value={engine}
            onChange={setEngine}
            options={[
              { value: 'pg', label: 'PostgreSQL', title: '8 KB heap page, 4-byte line pointers' },
              { value: 'innodb', label: 'InnoDB', title: '16 KB index page, 2-byte directory slots' },
              { value: 'sqlite', label: 'SQLite', title: '4 KB table leaf, 2-byte cell pointers' },
            ]}
          />
          <Slider label="Row size" min={40} max={1400} step={20} value={size} onChange={setSize} format={(n) => `${n} B`} />
          {engine === 'pg' ? (
            <Check label="HOT update (no indexed column changed)" checked={hot} onChange={setHot} />
          ) : null}
          <Button primary onClick={() => push({ kind: 'insert', size })}>
            Insert row
          </Button>
          <Button
            onClick={() => target !== null && push({ kind: 'update', id: target, size, hot })}
            disabled={target === null}
            title="Rewrites the selected row at the current row size"
          >
            Update {target !== null ? `row ${target}` : ''}
          </Button>
          <Button onClick={() => target !== null && push({ kind: 'delete', id: target })} disabled={target === null}>
            Delete {target !== null ? `row ${target}` : ''}
          </Button>
          {spec.reclaim ? <Button onClick={() => push({ kind: 'reclaim' })}>{spec.reclaim}</Button> : null}
          <Button onClick={() => push({ kind: 'compact' })}>{spec.compact}</Button>
          {engine === 'pg' ? <Button onClick={() => push({ kind: 'vacuum' })}>VACUUM</Button> : null}
          <Button
            onClick={() => {
              setOps(SEED);
              setSel(null);
            }}
          >
            Reset
          </Button>
        </>
      }
      legend={
        <Legend
          items={[
            { label: 'fixed header / trailer', color: FILL.fixed },
            { label: `${spec.ptrRegion} (${spec.ptrBytes} B each)`, color: FILL.ptr },
            { label: `live ${spec.recName}`, color: FILL.live },
            { label: engine === 'pg' ? 'dead tuple (deleted, not yet pruned)' : 'delete-marked record', color: FILL.dead },
            { label: engine === 'pg' ? 'hole — free but unusable until compaction' : 'hole — free and reusable', color: FILL.hole },
            { label: 'contiguous free space', color: FILL.free },
          ]}
        />
      }
      stats={
        <Stats
          items={[
            { label: spec.lowName, value: fmtNum(page.low), hint: 'End of the region growing from the low end of the page' },
            { label: spec.highName, value: fmtNum(page.high), hint: 'Start of the region growing from the high end' },
            { label: 'Contiguous free', value: fmtBytes(freeGap), hint: `${spec.highName} − ${spec.lowName}` },
            {
              label: engine === 'sqlite' ? 'Freeblocks + fragmented' : 'In holes',
              value: fmtBytes(holeBytes + page.frag),
              hint:
                engine === 'pg'
                  ? 'Reclaimed tuple space that PageAddItem cannot touch until PageRepairFragmentation runs'
                  : engine === 'innodb'
                    ? 'PAGE_GARBAGE — on the PAGE_FREE list and immediately reusable'
                    : 'The freeblock chain plus the 1-byte fragmented-free-bytes counter',
            },
            { label: `Live ${spec.recName}s`, value: `${liveIds.length}`, hint: fmtBytes(liveBytes) },
            {
              label: engine === 'pg' ? 'Dead tuple bytes' : engine === 'innodb' ? 'PAGE_GARBAGE' : 'Fragmented bytes',
              value: fmtBytes(engine === 'pg' ? deadBytes : engine === 'innodb' ? page.garbage : page.frag),
            },
            { label: 'Per-page overhead', value: fmtBytes(overhead), hint: 'Header, pointer array and trailer — bytes that hold no row data' },
            { label: 'Largest row that fits', value: fmtBytes(biggest), hint: 'Payload bytes, after per-record overhead' },
          ]}
        />
      }
      note={
        <Note>
          <strong>{page.head}</strong> {page.body}
          {page.refused > 0 ? ` (${page.refused} operation${page.refused === 1 ? '' : 's'} in this history did not fit on this engine's page.)` : ''}
        </Note>
      }
      table={
        <>
          <table className="viz-table">
            <thead>
              <tr>
                <th>{engine === 'pg' ? 'Slot' : engine === 'innodb' ? 'Directory' : 'Pointer'}</th>
                <th>State</th>
                <th>Row</th>
                <th>Key</th>
                <th>Offset</th>
                <th>Length</th>
              </tr>
            </thead>
            <tbody>
              {page.ptrs.length === 0 ? (
                <tr>
                  <td colSpan={6}>Page is empty.</td>
                </tr>
              ) : (
                page.ptrs.map((p) => {
                  const r = page.recs.find((x) => x.id === p.recId);
                  return (
                    <tr key={p.key}>
                      <td>{p.label}</td>
                      <td>
                        {engine === 'pg'
                          ? p.kind === 'normal'
                            ? 'LP_NORMAL'
                            : p.kind === 'dead'
                              ? 'LP_DEAD'
                              : p.kind === 'redirect'
                                ? `LP_REDIRECT → ${p.target}`
                                : 'LP_UNUSED'
                          : p.kind === 'sentinel'
                            ? 'sentinel'
                            : p.kind === 'dir'
                              ? `owns ${p.glyph}`
                              : 'cell'}
                      </td>
                      <td>{r ? `row ${r.id}${r.ver > 1 ? ` v${r.ver}` : ''}` : '—'}</td>
                      <td>{r ? r.key : '—'}</td>
                      <td>{r ? fmtNum(r.off) : '—'}</td>
                      <td>{r ? `${r.len} B` : '—'}</td>
                    </tr>
                  );
                })
              )}
            </tbody>
          </table>
          <table className="viz-table">
            <thead>
              <tr>
                <th></th>
                <th>PostgreSQL heap</th>
                <th>InnoDB index page</th>
                <th>SQLite table leaf</th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <td>Page size</td>
                <td>8 KB (BLCKSZ)</td>
                <td>16 KB (innodb_page_size)</td>
                <td>4 KB (PRAGMA page_size)</td>
              </tr>
              <tr>
                <td>Fixed header</td>
                <td>24 B PageHeaderData</td>
                <td>38 B FIL + 36 B index + 20 B FSEG + 26 B sentinels</td>
                <td>8 B (12 B on interior pages)</td>
              </tr>
              <tr>
                <td>Pointer</td>
                <td>4 B ItemIdData, insertion order</td>
                <td>2 B directory slot per 4–8 records</td>
                <td>2 B cell pointer, key order</td>
              </tr>
              <tr>
                <td>Per-record overhead</td>
                <td>23 B header, MAXALIGN'd to 24</td>
                <td>5 B header + 6 B trx id + 7 B roll pointer</td>
                <td>payload varint + rowid varint</td>
              </tr>
              <tr>
                <td>Records grow</td>
                <td>down from pd_special</td>
                <td>up from the page header</td>
                <td>down from the end of the page</td>
              </tr>
              <tr>
                <td>Reuse of holes</td>
                <td>only after PageRepairFragmentation</td>
                <td>immediately, from PAGE_FREE</td>
                <td>immediately, from the freeblock chain</td>
              </tr>
              <tr>
                <td>Row address</td>
                <td>TID (block, slot) — physical, stable</td>
                <td>primary key — logical</td>
                <td>rowid — logical</td>
              </tr>
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
            aria-label={`A ${fmtBytes(spec.page)} ${spec.label} showing its pointer array, record bytes, holes and free space`}
          >
            <defs>
              <marker id="spe-arrow" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="6" markerHeight="6" orient="auto">
                <path d="M0,0 L8,4 L0,8 z" fill="var(--viz-ink-2)" />
              </marker>
            </defs>

            <text x={PAD} y={12} fill="var(--viz-ink-2)">
              {spec.ptrRegion} — drawn oversized; {page.ptrs.length} × {spec.ptrBytes} B ={' '}
              {page.ptrs.length * spec.ptrBytes} B of the page
            </text>

            {/* pointer lane */}
            {page.ptrs.map((p, i) => {
              const r = page.recs.find((x) => x.id === p.recId);
              const bx = boxX(i);
              return (
                <g key={p.key} {...tip(<><strong>{spec.ptrLabel} {p.label}</strong><br />{p.hint}</>)} style={{ cursor: 'pointer' }} onClick={() => r && setSel(r.id)}>
                  <rect
                    x={bx}
                    y={ptrY}
                    width={boxW}
                    height={ptrH}
                    rx={4}
                    fill={ptrFill(p.kind)}
                    stroke={r && r.id === target ? 'var(--viz-ink)' : 'var(--viz-border)'}
                    strokeWidth={r && r.id === target ? 2 : 1}
                    strokeDasharray={p.kind === 'unused' ? '3 2' : undefined}
                  />
                  <text x={bx + boxW / 2} y={ptrY + 17} textAnchor="middle" fill="var(--viz-surface)" fontWeight={600}>
                    {boxW >= 30 ? `${p.label} ${p.glyph}` : p.label}
                  </text>
                </g>
              );
            })}

            {/* pointer → record connectors */}
            {page.ptrs.map((p, i) => {
              const r = page.recs.find((x) => x.id === p.recId);
              if (!r) {
                if (p.kind !== 'redirect' || p.target === null) return null;
                const j = page.ptrs.findIndex((q) => q.label === String(p.target));
                if (j < 0) return null;
                const from = boxX(i) + boxW / 2;
                const to = boxX(j) + boxW / 2;
                return (
                  <path
                    key={`red${p.key}`}
                    d={`M ${from} ${ptrY} C ${from} ${ptrY - 16}, ${to} ${ptrY - 16}, ${to} ${ptrY}`}
                    fill="none"
                    stroke="var(--viz-3)"
                    strokeWidth={1.5}
                    markerEnd="url(#spe-arrow)"
                  />
                );
              }
              const from = boxX(i) + boxW / 2;
              const to = recX(r);
              return (
                <path
                  key={`c${p.key}`}
                  d={`M ${from} ${ptrY + ptrH} C ${from} ${ptrY + ptrH + 30}, ${to} ${ribY - 30}, ${to} ${ribY}`}
                  fill="none"
                  stroke={r.id === target ? 'var(--viz-ink)' : 'var(--viz-axis)'}
                  strokeWidth={r.id === target ? 1.8 : 1}
                />
              );
            })}

            {/* the page, to scale */}
            <rect x={PAD} y={ribY} width={inner} height={ribH} rx={4} fill={FILL.free} stroke="var(--viz-border)" />
            <g {...tip(<><strong>{spec.frontLabel}</strong><br />Fixed bytes at the front of every page of this kind: {spec.frontFixed} B.</>)}>
              <rect x={PAD} y={ribY} width={w(spec.frontFixed)} height={ribH} fill={FILL.fixed} />
            </g>
            {spec.backFixed > 0 ? (
              <g {...tip(<><strong>{spec.backLabel}</strong><br />A copy of the LSN and the page checksum, at the very end of the page, so a torn write is detectable.</>)}>
                <rect x={x(spec.page - spec.backFixed)} y={ribY} width={w(spec.backFixed)} height={ribH} fill={FILL.fixed} />
              </g>
            ) : null}
            <g {...tip(<><strong>{spec.ptrRegion}</strong><br />{page.ptrs.length} × {spec.ptrBytes} B = {page.ptrs.length * spec.ptrBytes} B.</>)}>
              <rect
                x={engine === 'innodb' ? x(page.high) : x(spec.frontFixed)}
                y={ribY}
                width={w(page.ptrs.length * spec.ptrBytes)}
                height={ribH}
                fill={FILL.ptr}
              />
            </g>

            {page.holes.map((h) => (
              <g key={`h${h.off}`} {...tip(<><strong>Hole: {h.len} B at offset {h.off}</strong><br />{engine === 'pg' ? 'Free, and unreachable: PageAddItem only allocates between pd_lower and pd_upper. Compaction is what turns it back into usable space.' : 'On the free list and available to the next record that fits.'}</>)}>
                <rect x={x(h.off)} y={ribY} width={w(h.len)} height={ribH} fill={FILL.hole} />
                {w(h.len) > 34 ? (
                  <text x={x(h.off) + w(h.len) / 2} y={ribY + ribH + 12} textAnchor="middle" fill="var(--viz-ink-2)">
                    hole
                  </text>
                ) : null}
              </g>
            ))}

            {page.recs.map((r) => (
              <g
                key={`r${r.id}`}
                {...tip(
                  <>
                    <strong>
                      row {r.id}
                      {r.ver > 1 ? ` (version ${r.ver})` : ''}, key {r.key}
                    </strong>
                    <br />
                    {r.len} B at offset {r.off} — {spec.recOverhead} B of record header plus {r.len - spec.recOverhead} B of column data.
                    <br />
                    {r.dead
                      ? engine === 'pg'
                        ? 'Deleted: xmax is set, but the bytes are still here and still visible to older snapshots.'
                        : 'Delete-marked and still linked into the record list until purge runs.'
                      : engine === 'pg'
                        ? `Addressed as TID (0,${r.slot}).`
                        : `Addressed by its key, not by its position.`}
                  </>,
                )}
                style={{ cursor: 'pointer' }}
                onClick={() => !r.dead && setSel(r.id)}
              >
                <rect
                  x={x(r.off)}
                  y={ribY}
                  width={w(r.len)}
                  height={ribH}
                  fill={r.dead ? FILL.dead : FILL.live}
                  stroke={r.id === target ? 'var(--viz-ink)' : 'var(--viz-surface)'}
                  strokeWidth={r.id === target ? 2 : 1}
                />
                {w(r.len) > 26 ? (
                  <text x={recX(r)} y={ribY + ribH / 2 + 4} textAnchor="middle" fill="var(--viz-surface)" fontWeight={600}>
                    {r.dead ? '×' : ''}
                    {r.id}
                  </text>
                ) : null}
              </g>
            ))}

            {/* boundary markers */}
            {[
              { at: page.low, name: spec.lowName },
              { at: page.high, name: spec.highName },
            ].map((m) => (
              <g key={m.name}>
                <line x1={x(m.at)} x2={x(m.at)} y1={ribY - 6} y2={ribY + ribH + 6} stroke="var(--viz-ink)" strokeWidth={1.2} strokeDasharray="4 3" />
                <text x={x(m.at)} y={ribY - 10} textAnchor="middle" fill="var(--viz-ink)">
                  {m.name} = {m.at}
                </text>
              </g>
            ))}

            {/* InnoDB record list */}
            {hasList ? (
              <g>
                <text x={PAD} y={listY - 12} fill="var(--viz-ink-2)">
                  record list — infimum → keys ascending → supremum (heap order is insertion order; the links are the sort order)
                </text>
                {page.order.map((id, i) => {
                  const r = page.recs.find((q) => q.id === id)!;
                  const next = i + 1 < page.order.length ? page.recs.find((q) => q.id === page.order[i + 1])! : null;
                  const from = recX(r);
                  const to = next ? recX(next) : x(spec.frontFixed) + 10;
                  const mid = (from + to) / 2;
                  return (
                    <g key={`l${id}`}>
                      <circle cx={from} cy={listY} r={4} fill={FILL.live} />
                      <path
                        d={`M ${from} ${listY} Q ${mid} ${listY + (to > from ? 16 : -16)}, ${to} ${listY}`}
                        fill="none"
                        stroke="var(--viz-ink-2)"
                        strokeWidth={1}
                        markerEnd="url(#spe-arrow)"
                      />
                    </g>
                  );
                })}
                <text x={x(spec.frontFixed) + 2} y={listY + 16} fill="var(--viz-ink-2)">
                  inf / sup
                </text>
              </g>
            ) : null}

            {/* byte axis */}
            <line className="viz-axis-line" x1={PAD} x2={PAD + inner} y1={axisY} y2={axisY} />
            {ticks.map((t) => (
              <g key={t}>
                <line className="viz-axis-line" x1={x(t)} x2={x(t)} y1={axisY} y2={axisY + 4} />
                <text x={x(t)} y={axisY + 16} textAnchor={t === 0 ? 'start' : t === spec.page ? 'end' : 'middle'} fill="var(--viz-ink-muted)">
                  {t === 0 ? '0' : fmtBytes(t)}
                </text>
              </g>
            ))}
          </svg>
        </TooltipHost>
      </div>
    </VizPanel>
  );
}
