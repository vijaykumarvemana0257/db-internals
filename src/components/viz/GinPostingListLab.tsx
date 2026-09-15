import { useMemo, useState, type ReactNode } from 'react';
import { VizPanel, Segmented, Check, Slider, Button, Legend, Stats, Note, TooltipHost, useTip, fmtNum, makeRng } from './Viz';

/**
 * A GIN index being built and maintained, following PostgreSQL's src/backend/access/gin and jsonb_gin.c.
 *
 * Real mechanism kept:
 * - extractValue turns one row into many keys; ginExtractEntries sorts and de-duplicates them per row.
 *   text[] (array_ops): one key per distinct element.
 *   jsonb_ops: one key per object key and one per scalar value, each prefixed with a flag byte
 *   (JGINFLAG_KEY, _STR, _NUM, _BOOL, _NULL); string ARRAY ELEMENTS are flagged as keys.
 *   jsonb_path_ops: one uint32 per scalar value, folding hash = rotl1(hash) ^ hash(x) over the object keys on
 *   the path to it and then the value (JsonbHashScalarValue); arrays add nothing to the path.
 * - The entry tree is a B-tree over keys; each leaf entry holds a posting list of heap TIDs, delta + varbyte
 *   coded (TID -> block << 11 | offset, first TID stored raw in an 8-byte header). A list too big to fit in the
 *   entry tuple becomes a posting tree, and never converts back.
 * - fastupdate: a row's keys are appended to the tail pending-list page (a new page if they do not fit); after
 *   the append, if pending pages exceed gin_pending_list_limit the INSERTING backend runs ginInsertCleanup and
 *   merges the whole list into the entry tree, one merge per distinct key. VACUUM does the same in the background.
 *   With fastupdate off, every key is inserted into the entry tree immediately.
 *
 * Lab scaling (labelled in hints): heap pages hold 4 tuples; a posting list becomes a posting tree above
 * 48 bytes (real limit: GinMaxItemSize, 2,712 bytes at 8 KB pages); posting-tree leaves hold 16 TIDs;
 * a pending-list page holds 12 entries and entry-tree leaves hold 7 keys. hash_any is stood in for by FNV-1a.
 * Rows are appended, so TIDs arrive in increasing order. Tag and document frequencies are invented; the 'rust'
 * tag only appears from row 260 on, to model a key whose rows cluster at the end of the heap.
 */

/* ------------------------------------------------------------------ data */

export type Json = null | boolean | number | string | Json[] | { [k: string]: Json };
export type Kind = 'tags' | 'jsonb';
export type Opclass = 'array_ops' | 'jsonb_ops' | 'jsonb_path_ops';

export const HEAP_TUPLES_PER_PAGE = 4;
export const INLINE_LIMIT_BYTES = 48;
export const REAL_GIN_MAX_ITEM_SIZE = 2712;
export const POSTING_LEAF_TIDS = 16;
export const POSTING_SEGMENT_TIDS = 4;
export const POSTING_FANOUT = 8;
export const PENDING_PAGE_ENTRIES = 12;
export const ENTRY_LEAF_CAP = 7;
export const INITIAL_ROWS = 24;
export const MAX_ROWS = 400;

export function fnv32(s: string) {
  let h = 0x811c9dc5;
  for (let i = 0; i < s.length; i++) {
    h ^= s.charCodeAt(i);
    h = Math.imul(h, 0x01000193) >>> 0;
  }
  return h >>> 0;
}
const rotl1 = (h: number) => ((h << 1) | (h >>> 31)) >>> 0;

export const tidOf = (row: number) => ({ blk: Math.floor(row / HEAP_TUPLES_PER_PAGE), off: (row % HEAP_TUPLES_PER_PAGE) + 1 });
/** itemptr_to_uint64: offset number in the low 11 bits, block number above it. */
export const tidCode = (row: number) => {
  const t = tidOf(row);
  return t.blk * 2048 + t.off;
};
export const tidLabel = (row: number) => {
  const t = tidOf(row);
  return `(${t.blk},${t.off})`;
};
export function varbyteLen(n: number) {
  let len = 1;
  while (n > 0x7f) {
    n = Math.floor(n / 128);
    len++;
  }
  return len;
}
/** Bytes of a GinPostingList: 6-byte first TID + uint16 length, then varbyte deltas, short-aligned. */
export function postingListBytes(rows: readonly number[]) {
  if (rows.length === 0) return 0;
  let s = 0;
  for (let i = 1; i < rows.length; i++) s += varbyteLen(tidCode(rows[i]) - tidCode(rows[i - 1]));
  return 8 + s + (s % 2);
}
export const uncompressedBytes = (n: number) => n * 6;

const TAG_FREQ: [string, number][] = [
  ['postgres', 0.6],
  ['linux', 0.45],
  ['sql', 0.4],
  ['jsonb', 0.3],
  ['vacuum', 0.25],
  ['replication', 0.2],
  ['wal', 0.15],
  ['btree', 0.12],
  ['gin', 0.08],
];
/** A tag that only started appearing recently: its TIDs cluster at the end of the heap. */
export const RECENT_TAG = { tag: 'rust', fromRow: 260, p: 0.3 };

export function rowTags(row: number): string[] {
  const r = makeRng(fnv32(`tags:${row}`));
  const out = TAG_FREQ.filter(([, p]) => r() < p).map(([t]) => t);
  if (r() < (row >= RECENT_TAG.fromRow ? RECENT_TAG.p : 0)) out.push(RECENT_TAG.tag);
  return out.length ? out : [TAG_FREQ[Math.floor(r() * 3)][0]];
}

export function rowDoc(row: number): { [k: string]: Json } {
  const r = makeRng(fnv32(`doc:${row}`));
  const pick = (opts: [string, number][]) => {
    const x = r();
    let acc = 0;
    for (const [v, p] of opts) {
      acc += p;
      if (x < acc) return v;
    }
    return opts[opts.length - 1][0];
  };
  const doc: { [k: string]: Json } = {};
  doc.status = pick([
    ['active', 0.6],
    ['archived', 0.3],
    ['draft', 0.1],
  ]);
  doc.owner = { team: pick([
    ['web', 0.5],
    ['infra', 0.35],
    ['db', 0.15],
  ]) };
  const tags: [string, number][] = [
    ['api', 0.4],
    ['billing', 0.3],
    ['db', 0.25],
    ['urgent', 0.2],
  ];
  doc.tags = tags.filter(([, p]) => r() < p).map(([t]) => t);
  doc.priority = 1 + Math.floor(r() * 3);
  if (r() < 0.35) doc.reviewer = { team: pick([
    ['db', 0.4],
    ['web', 0.6],
  ]) };
  return doc;
}

export const rowValue = (kind: Kind, row: number): string[] | Json => (kind === 'tags' ? rowTags(row) : rowDoc(row));

export function showValue(v: string[] | Json): string {
  if (Array.isArray(v) && v.every((x) => typeof x === 'string')) return `{${(v as string[]).join(',')}}`;
  return JSON.stringify(v).replace(/,"/g, ', "').replace(/":/g, '": ');
}

/* ------------------------------------------------------- key extraction */

/** One GIN key. `id` sorts the way the opclass's compare function orders keys. */
export type GinKey = { id: string; label: string; detail: string };

const FLAG = { key: 1, null: 2, bool: 3, num: 4, str: 5 } as const;

function jsonbOpsKeys(value: Json, out: GinKey[]) {
  const scalar = (v: Json, asKey: boolean) => {
    if (v === null) out.push({ id: `${FLAG.null}`, label: 'null', detail: 'null value (JGINFLAG_NULL)' });
    else if (typeof v === 'boolean') out.push({ id: `${FLAG.bool}${v ? 't' : 'f'}`, label: `bool:${v ? 't' : 'f'}`, detail: 'boolean value (JGINFLAG_BOOL)' });
    else if (typeof v === 'number') out.push({ id: `${FLAG.num}${v}`, label: `num:${v}`, detail: 'numeric value, normalized text (JGINFLAG_NUM)' });
    else if (typeof v === 'string')
      out.push(
        asKey
          ? { id: `${FLAG.key}${v}`, label: `key:${v}`, detail: 'object key or string array element (JGINFLAG_KEY)' }
          : { id: `${FLAG.str}${v}`, label: `str:${v}`, detail: 'string value (JGINFLAG_STR)' },
      );
  };
  if (Array.isArray(value)) {
    for (const el of value) {
      if (el !== null && typeof el === 'object') jsonbOpsKeys(el, out);
      else scalar(el, typeof el === 'string');
    }
  } else if (value !== null && typeof value === 'object') {
    for (const k of Object.keys(value)) {
      scalar(k, true);
      const v = value[k];
      if (v !== null && typeof v === 'object') jsonbOpsKeys(v, out);
      else scalar(v, false);
    }
  } else scalar(value, false);
}

const scalarHash = (v: Json) => (v === null ? 0x01 : typeof v === 'boolean' ? (v ? 0x02 : 0x04) : fnv32(String(v)));

function jsonbPathKeys(value: Json, hash: number, path: string, out: GinKey[]) {
  const emit = (v: Json) => {
    const h = (rotl1(hash) ^ scalarHash(v)) >>> 0;
    const signed = (h ^ 0x80000000) >>> 0; // int4 order of the stored uint32
    out.push({ id: signed.toString(16).padStart(8, '0'), label: `0x${h.toString(16).padStart(8, '0')}`, detail: `${path || '(top)'} = ${JSON.stringify(v)}` });
  };
  if (Array.isArray(value)) {
    for (const el of value) {
      if (el !== null && typeof el === 'object') jsonbPathKeys(el, hash, `${path}[]`, out);
      else emit(el);
    }
  } else if (value !== null && typeof value === 'object') {
    for (const k of Object.keys(value)) {
      const hk = (rotl1(hash) ^ fnv32(k)) >>> 0;
      const v = value[k];
      const p = path ? `${path}.${k}` : k;
      if (v !== null && typeof v === 'object') jsonbPathKeys(v, hk, p, out);
      else {
        const h = (rotl1(hk) ^ scalarHash(v)) >>> 0;
        const signed = (h ^ 0x80000000) >>> 0;
        out.push({ id: signed.toString(16).padStart(8, '0'), label: `0x${h.toString(16).padStart(8, '0')}`, detail: `${p} = ${JSON.stringify(v)}` });
      }
    }
  } else emit(value);
}

/** extractValue + ginExtractEntries: the sorted, de-duplicated keys of one item. */
export function extractKeys(opclass: Opclass, value: string[] | Json): GinKey[] {
  const out: GinKey[] = [];
  if (opclass === 'array_ops') for (const t of value as string[]) out.push({ id: t, label: t, detail: 'array element' });
  else if (opclass === 'jsonb_ops') jsonbOpsKeys(value as Json, out);
  else jsonbPathKeys(value as Json, 0, '', out);
  out.sort((a, b) => (a.id < b.id ? -1 : a.id > b.id ? 1 : 0));
  return out.filter((k, i) => i === 0 || k.id !== out[i - 1].id);
}

export const rowKeys = (kind: Kind, opclass: Opclass, row: number) => extractKeys(opclass, rowValue(kind, row));

/* ------------------------------------------------------------ the index */

export type InsertEvent = {
  row: number;
  keys: number;
  pendingAppend: boolean;
  entryInserts: number;
  cleanup: null | { rows: number; entries: number; keys: number };
};

export type IndexState = {
  kind: Kind;
  opclass: Opclass;
  rows: number;
  main: Record<string, number[]>;
  info: Record<string, GinKey>;
  pending: { row: number; keys: string[] }[];
  pendingPages: number[];
  history: InsertEvent[];
  touched: string[];
  vacuums: number;
  last: { kind: 'build' | 'insert' | 'vacuum' | 'reset'; rows?: number; event?: InsertEvent; merged?: { rows: number; entries: number; keys: number } };
};

export function buildIndex(kind: Kind, opclass: Opclass, rows: number): IndexState {
  const main: Record<string, number[]> = {};
  const info: Record<string, GinKey> = {};
  for (let r = 0; r < rows; r++) {
    for (const k of rowKeys(kind, opclass, r)) {
      info[k.id] = k;
      (main[k.id] ??= []).push(r);
    }
  }
  return { kind, opclass, rows, main, info, pending: [], pendingPages: [], history: [], touched: [], vacuums: 0, last: { kind: 'build', rows } };
}

function appendPending(pages: number[], n: number) {
  const out = pages.slice();
  if (out.length && out[out.length - 1] + n <= PENDING_PAGE_ENTRIES) out[out.length - 1] += n;
  else
    for (let left = n; left > 0; left -= PENDING_PAGE_ENTRIES) out.push(Math.min(PENDING_PAGE_ENTRIES, left));
  return out;
}

/** ginInsertCleanup: move every pending entry into the entry tree, one merge per distinct key. */
function mergePending(s: IndexState) {
  const main = { ...s.main };
  const byKey = new Map<string, number[]>();
  let entries = 0;
  for (const p of s.pending)
    for (const k of p.keys) {
      entries++;
      if (!byKey.has(k)) byKey.set(k, []);
      byKey.get(k)!.push(p.row);
    }
  for (const [k, rows] of byKey) main[k] = [...(main[k] ?? []), ...rows];
  return { main, merged: { rows: s.pending.length, entries, keys: byKey.size }, touched: [...byKey.keys()] };
}

export function insertRow(s: IndexState, fastupdate: boolean, limitPages: number): IndexState {
  if (s.rows >= MAX_ROWS) return s;
  const row = s.rows;
  const keys = rowKeys(s.kind, s.opclass, row);
  const info = { ...s.info };
  for (const k of keys) info[k.id] = k;
  if (!fastupdate) {
    const main = { ...s.main };
    for (const k of keys) main[k.id] = [...(main[k.id] ?? []), row];
    const event: InsertEvent = { row, keys: keys.length, pendingAppend: false, entryInserts: keys.length, cleanup: null };
    return { ...s, rows: row + 1, main, info, history: [...s.history, event], touched: keys.map((k) => k.id), last: { kind: 'insert', event } };
  }
  const pending = [...s.pending, { row, keys: keys.map((k) => k.id) }];
  const pendingPages = appendPending(s.pendingPages, keys.length);
  const withRow: IndexState = { ...s, rows: row + 1, info, pending, pendingPages };
  if (pendingPages.length > limitPages) {
    const m = mergePending(withRow);
    const event: InsertEvent = { row, keys: keys.length, pendingAppend: true, entryInserts: m.merged.keys, cleanup: m.merged };
    return { ...withRow, main: m.main, pending: [], pendingPages: [], history: [...s.history, event], touched: m.touched, last: { kind: 'insert', event } };
  }
  const event: InsertEvent = { row, keys: keys.length, pendingAppend: true, entryInserts: 0, cleanup: null };
  return { ...withRow, history: [...s.history, event], touched: [], last: { kind: 'insert', event } };
}

export function vacuumIndex(s: IndexState): IndexState {
  if (s.pending.length === 0) return { ...s, touched: [], vacuums: s.vacuums + 1, last: { kind: 'vacuum', merged: { rows: 0, entries: 0, keys: 0 } } };
  const m = mergePending(s);
  return { ...s, main: m.main, pending: [], pendingPages: [], touched: m.touched, vacuums: s.vacuums + 1, last: { kind: 'vacuum', merged: m.merged } };
}

export function postingTreeShape(n: number) {
  const leaves = Math.max(1, Math.ceil(n / POSTING_LEAF_TIDS));
  const height = leaves <= 1 ? 1 : 1 + Math.ceil(Math.log(leaves) / Math.log(POSTING_FANOUT));
  return { leaves, height };
}

export type EntryView = { id: string; key: GinKey; rows: number[]; bytes: number; tree: boolean; leaves: number };

export function entries(s: IndexState): EntryView[] {
  return Object.keys(s.main)
    .sort()
    .map((id) => {
      const rows = s.main[id];
      const bytes = postingListBytes(rows);
      const tree = bytes > INLINE_LIMIT_BYTES;
      return { id, key: s.info[id], rows, bytes, tree, leaves: tree ? postingTreeShape(rows.length).leaves : 0 };
    });
}

export function summarize(s: IndexState) {
  const es = entries(s);
  const tids = es.reduce((a, e) => a + e.rows.length, 0);
  const pendingEntries = s.pending.reduce((a, p) => a + p.keys.length, 0);
  const inlineBytes = es.filter((e) => !e.tree).reduce((a, e) => a + e.bytes, 0);
  const inlineTids = es.filter((e) => !e.tree).reduce((a, e) => a + e.rows.length, 0);
  const history = s.history;
  return {
    entries: es,
    distinctKeys: es.length,
    tids,
    pendingEntries,
    pendingPages: s.pendingPages.length,
    postingTrees: es.filter((e) => e.tree).length,
    inlineBytes,
    inlineRawBytes: uncompressedBytes(inlineTids),
    keysPerRow: s.rows ? (tids + pendingEntries) / s.rows : 0,
    insertCount: history.length,
    entryInsertsTotal: history.reduce((a, h) => a + h.entryInserts, 0),
    keysInsertedTotal: history.reduce((a, h) => a + h.keys, 0),
    cleanups: history.filter((h) => h.cleanup).length,
  };
}

/** Average keys per row for each opclass over the same rows. */
export function keysPerRow(kind: Kind, rows: number): Partial<Record<Opclass, number>> {
  const avg = (oc: Opclass) => {
    let t = 0;
    for (let r = 0; r < rows; r++) t += rowKeys(kind, oc, r).length;
    return rows ? t / rows : 0;
  };
  return kind === 'tags' ? { array_ops: avg('array_ops') } : { jsonb_ops: avg('jsonb_ops'), jsonb_path_ops: avg('jsonb_path_ops') };
}

/* -------------------------------------------------------------------- UI */

const HISTORY_BARS = 36;
const plural = (n: number, word: string) => `${fmtNum(n)} ${word}${n === 1 ? '' : 's'}`;
const compactPath = (detail: string) => detail.replace(' = ', '=').replace(/"/g, '');

function EntryRow({ e, x, y, w, maxRows, touched, pathOps }: { e: EntryView; x: number; y: number; w: number; maxRows: number; touched: boolean; pathOps: boolean }) {
  const tip = useTip();
  const labelW = pathOps ? 116 : 94;
  const barX = x + labelW;
  const barMax = w - labelW - 56;
  const bw = Math.max(2, (e.rows.length / Math.max(1, maxRows)) * barMax);
  const text = pathOps ? compactPath(e.key.detail) : e.key.label;
  const shown = text.length > (pathOps ? 17 : 14) ? `${text.slice(0, pathOps ? 16 : 13)}…` : text;
  return (
    <g
      {...tip(
        <>
          <strong>{e.key.label}</strong>
          <br />
          {e.key.detail}
          <br />
          {fmtNum(e.rows.length)} TIDs:{' '}
          {e.tree
            ? `posting tree with ${e.leaves} leaf page${e.leaves === 1 ? '' : 's'} (the list would need ${e.bytes} B, over the lab's ${INLINE_LIMIT_BYTES} B inline limit)`
            : `inline posting list, ${e.bytes} B compressed vs ${uncompressedBytes(e.rows.length)} B as raw 6-byte TIDs`}
          <br />
          {e.rows.slice(0, 8).map(tidLabel).join(' ')}
          {e.rows.length > 8 ? ' …' : ''}
        </>,
      )}
    >
      <rect x={x + 2} y={y} width={w - 4} height={17} fill="transparent" />
      <text x={x + 6} y={y + 12} fontSize={11.5} fill="var(--viz-ink)">
        {shown}
      </text>
      <rect x={barX} y={y + 3} width={bw} height={12} rx={2} fill={e.tree ? 'var(--viz-7)' : 'var(--viz-1)'} stroke={touched ? 'var(--viz-ink)' : 'none'} strokeWidth={1.5} />
      <text x={barX + bw + 4} y={y + 13} fontSize={11} fill="var(--viz-ink-2)">
        {e.rows.length}
        {e.tree ? ' tree' : ''}
      </text>
    </g>
  );
}

export default function GinPostingListLab() {
  const [kind, setKind] = useState<Kind>('tags');
  const [jsonOpclass, setJsonOpclass] = useState<Opclass>('jsonb_ops');
  const [fastupdate, setFastupdate] = useState(true);
  const [limit, setLimit] = useState(2);
  const [state, setState] = useState<IndexState>(() => buildIndex('tags', 'array_ops', INITIAL_ROWS));

  const opclass: Opclass = kind === 'tags' ? 'array_ops' : jsonOpclass;
  const sum = useMemo(() => summarize(state), [state]);
  const kpr = useMemo(() => keysPerRow(state.kind, state.rows), [state.kind, state.rows]);

  const rebuild = (k: Kind, oc: Opclass, rows: number) => setState(buildIndex(k, oc, rows));
  const insertN = (count: number) =>
    setState((s0) => {
      let s = s0;
      for (let i = 0; i < count && s.rows < MAX_ROWS; i++) s = insertRow(s, fastupdate, limit);
      return s;
    });

  const W = 680;
  const newest = state.rows - 1;
  const newestKeys = rowKeys(state.kind, state.opclass, newest);
  const newestPending = state.pending.some((p) => p.row === newest);
  const pathOps = state.opclass === 'jsonb_path_ops';

  // chips for the newest row
  const chips: { k: GinKey; x: number; y: number; w: number }[] = [];
  {
    let cx = 8;
    let cy = 40;
    for (const k of newestKeys) {
      const label = pathOps ? compactPath(k.detail) : k.label;
      const w = Math.min(220, label.length * 6.6 + 14);
      if (cx + w > W - 8) {
        cx = 8;
        cy += 23;
      }
      chips.push({ k, x: cx, y: cy, w });
      cx += w + 5;
    }
  }
  const chipsBottom = (chips.length ? chips[chips.length - 1].y : 40) + 18;

  // entry tree
  const es = sum.entries;
  const leaves: EntryView[][] = [];
  for (let i = 0; i < es.length; i += ENTRY_LEAF_CAP) leaves.push(es.slice(i, i + ENTRY_LEAF_CAP));
  const cols = Math.max(1, Math.min(3, leaves.length));
  const gap = 10;
  const leafW = (W - 16 - gap * (cols - 1)) / cols;
  const rootY = chipsBottom + 26;
  const leafY = rootY + 48;
  const rowH = 18;
  const leafH = 20 + ENTRY_LEAF_CAP * rowH;
  const leafRows = Math.ceil(leaves.length / cols);
  const treeBottom = leafY + leafRows * (leafH + gap);
  const touched = new Set(state.touched);
  const separators = leaves.slice(1).map((l) => l[0].key.label);

  // pending list
  const pendY = treeBottom + 22;
  const cell = 8;
  const pageW = PENDING_PAGE_ENTRIES * cell + 6;
  const maxPagesShown = 7;
  const pagesShown = state.pendingPages.slice(0, maxPagesShown);
  const pendBoxY = pendY + 10;
  const limitX = 8 + limit * (pageW + 8) - 4;

  // history
  const histY = pendBoxY + 50;
  const hist = state.history.slice(-HISTORY_BARS);
  const barMaxH = 56;
  const histTop = Math.max(12, ...hist.map((h) => h.entryInserts));
  const histMax = Math.ceil(histTop * 1.25);
  const bw = (W - 70) / HISTORY_BARS;
  const H = histY + barMaxH + 44;

  const last = state.last;
  let note: ReactNode;
  if (last.kind === 'build') {
    note = (
      <>
        <strong>
          CREATE INDEX built the entry tree in bulk over {fmtNum(state.rows)} rows: {fmtNum(sum.tids)} TIDs under {fmtNum(sum.distinctKeys)} keys.
        </strong>{' '}
        Each row produced {fmtNum(sum.keysPerRow, 1)} keys on average{state.kind === 'jsonb' ? ` with ${state.opclass}` : ''}. Insert rows to watch posting lists grow{fastupdate ? ' and the pending list fill' : ''}.
      </>
    );
  } else if (last.kind === 'vacuum') {
    note = last.merged && last.merged.entries ? (
      <>
        <strong>
          VACUUM merged {fmtNum(last.merged.entries)} pending entries from {plural(last.merged.rows, 'row')} into {plural(last.merged.keys, 'posting list')}.
        </strong>{' '}
        Same bulk merge an over-limit INSERT does, but run by VACUUM (or autovacuum) instead of inside some writer’s INSERT.
      </>
    ) : (
      <>
        <strong>VACUUM found the pending list empty.</strong> Nothing to merge.
      </>
    );
  } else if (last.event) {
    const ev = last.event;
    const t = tidLabel(ev.row);
    const trees = es.filter((e) => e.tree && touched.has(e.id)).map((e) => e.key.label);
    note = ev.cleanup ? (
      <>
        <strong>
          The INSERT of row {t} pushed the pending list past {limit} page{limit === 1 ? '' : 's'}, so that INSERT ran the cleanup itself: {fmtNum(ev.cleanup.entries)} entries from {fmtNum(ev.cleanup.rows)} rows merged into {fmtNum(ev.cleanup.keys)} posting lists.
        </strong>{' '}
        {fmtNum(ev.cleanup.keys)} entry-tree insertions instead of 0 — the tall bar is the latency spike one unlucky writer pays.{trees.length ? ` Posting trees now: ${trees.join(', ')}.` : ''}
      </>
    ) : ev.pendingAppend ? (
      <>
        <strong>
          Row {t} produced {plural(ev.keys, 'key')}, appended to the pending list ({state.pendingPages.length} of {limit} page{limit === 1 ? '' : 's'} allowed).
        </strong>{' '}
        No entry-tree work at all, so this INSERT was cheap. Scans now have {fmtNum(sum.pendingEntries)} unsorted entries to read on top of the entry tree.
      </>
    ) : (
      <>
        <strong>
          Row {t} produced {plural(ev.keys, 'key')}; with fastupdate off each one descended the entry tree and rewrote its posting list: {plural(ev.keys, 'entry-tree insertion')}.
        </strong>{' '}
        {trees.length ? `${trees.join(', ')} ${trees.length === 1 ? 'is' : 'are'} past the inline limit and stored as posting tree${trees.length === 1 ? '' : 's'}.` : 'Every INSERT pays this, so latency is even but higher.'}
        {state.pending.length ? ` The ${fmtNum(sum.pendingEntries)} older pending entries stay put until VACUUM: turning fastupdate off does not flush them.` : ''}
      </>
    );
  }

  return (
    <VizPanel
      title="A GIN index taking inserts"
      subtitle="Every row is split into keys. Each key owns a posting list of heap TIDs in the entry tree. With fastupdate on, new keys wait in the pending list until it outgrows gin_pending_list_limit."
      controls={
        <>
          <Segmented
            label="Column"
            value={kind}
            onChange={(k) => {
              setKind(k);
              rebuild(k, k === 'tags' ? 'array_ops' : jsonOpclass, INITIAL_ROWS);
            }}
            options={[
              { value: 'tags', label: 'tags text[]' },
              { value: 'jsonb', label: 'doc jsonb' },
            ]}
          />
          {kind === 'jsonb' ? (
            <Segmented
              label="Operator class (rebuilds over the same rows)"
              value={jsonOpclass}
              onChange={(oc) => {
                setJsonOpclass(oc);
                rebuild('jsonb', oc, state.rows);
              }}
              options={[
                { value: 'jsonb_ops', label: 'jsonb_ops' },
                { value: 'jsonb_path_ops', label: 'jsonb_path_ops' },
              ]}
            />
          ) : null}
          <Check label="fastupdate" checked={fastupdate} onChange={setFastupdate} />
          <Slider label="gin_pending_list_limit (lab pages)" min={1} max={6} value={limit} onChange={setLimit} />
        </>
      }
      legend={
        <Legend
          items={[
            { label: 'Inline posting list', color: 'var(--viz-1)' },
            { label: 'Posting tree', color: 'var(--viz-7)' },
            { label: 'Pending-list entry', color: 'var(--viz-2)' },
            { label: 'Entry-tree insertions in one INSERT', color: 'var(--viz-3)' },
            { label: 'Changed by the last action', color: 'var(--viz-ink)', shape: 'line' },
          ]}
        />
      }
      stats={
        <Stats
          items={[
            { label: 'Rows', value: fmtNum(state.rows) },
            {
              label: 'Keys per row',
              value: state.kind === 'jsonb' ? `${fmtNum(kpr.jsonb_ops ?? 0, 1)} vs ${fmtNum(kpr.jsonb_path_ops ?? 0, 1)}` : fmtNum(kpr.array_ops ?? 0, 1),
              hint: state.kind === 'jsonb' ? 'jsonb_ops vs jsonb_path_ops over the same rows' : 'distinct array elements per row',
            },
            { label: 'Keys in the entry tree', value: fmtNum(sum.distinctKeys) },
            { label: 'TIDs in posting lists', value: fmtNum(sum.tids), hint: 'One index entry per (key, row) pair: this, not the row count, is what a write maintains.' },
            { label: 'Posting trees', value: fmtNum(sum.postingTrees), hint: `Lab lists move to a posting tree above ${INLINE_LIMIT_BYTES} B; the real limit is GinMaxItemSize, ${fmtNum(REAL_GIN_MAX_ITEM_SIZE)} B at 8 KB pages.` },
            { label: 'Pending entries (pages)', value: `${fmtNum(sum.pendingEntries)} (${sum.pendingPages})`, hint: `Lab pages hold ${PENDING_PAGE_ENTRIES} entries. At the real 4 MB default, cleanup fires once more than 514 pending pages exist (515 × 8,160 B > 4,096 kB).` },
            { label: 'Entry-tree insertions / keys inserted', value: `${fmtNum(sum.entryInsertsTotal)} / ${fmtNum(sum.keysInsertedTotal)}`, hint: 'Since the last rebuild. A cleanup merges each distinct key once, however many pending rows carry it.' },
          ]}
        />
      }
      note={<Note>{note}</Note>}
      table={
        <table className="viz-table">
          <thead>
            <tr>
              <th>Key</th>
              <th>Meaning</th>
              <th>TIDs</th>
              <th>Storage</th>
            </tr>
          </thead>
          <tbody>
            {es.map((e) => (
              <tr key={e.id}>
                <td>{e.key.label}</td>
                <td>{e.key.detail}</td>
                <td>{fmtNum(e.rows.length)}</td>
                <td>{e.tree ? `posting tree, ${e.leaves} leaf pages` : `inline, ${e.bytes} B (raw TIDs ${uncompressedBytes(e.rows.length)} B)`}</td>
              </tr>
            ))}
            <tr>
              <td>pending list</td>
              <td>unsorted, appended per row</td>
              <td>{fmtNum(sum.pendingEntries)}</td>
              <td>{sum.pendingPages} pages</td>
            </tr>
          </tbody>
        </table>
      }
    >
      <div className="viz-controls">
        <Button primary onClick={() => insertN(1)} disabled={state.rows >= MAX_ROWS}>
          INSERT 1 row
        </Button>
        <Button onClick={() => insertN(10)} disabled={state.rows >= MAX_ROWS}>
          INSERT 10 rows
        </Button>
        <Button onClick={() => setState((s) => vacuumIndex(s))}>VACUUM</Button>
        <Button onClick={() => rebuild(kind, opclass, INITIAL_ROWS)}>Start over</Button>
      </div>
      <TooltipHost>
        <svg viewBox={`0 0 ${W} ${H}`} width={W} height={H} role="img" aria-label={`GIN index: ${sum.distinctKeys} keys, ${sum.tids} TIDs, ${sum.pendingEntries} pending entries`}>
          <text x={8} y={14} fontSize={13} fill="var(--viz-ink)">
            {`Newest row ${tidLabel(newest)}: `}
            <tspan fill="var(--viz-ink-2)">{(() => {
              const v = showValue(rowValue(state.kind, newest));
              return v.length > 92 ? `${v.slice(0, 91)}…` : v;
            })()}</tspan>
          </text>
          <text x={8} y={30} fontSize={11.5} fill="var(--viz-ink-2)">
            extractValue → {plural(newestKeys.length, 'key')}{pathOps ? ' (uint32 hashes, shown as path=value)' : ', sorted and de-duplicated'}
            {newestPending ? ' — waiting in the pending list' : state.history.length ? ' — in the entry tree' : ''}
          </text>
          {chips.map((c) => (
            <g key={c.k.id}>
              <rect x={c.x} y={c.y} width={c.w} height={18} rx={9} fill="var(--viz-surface)" stroke={newestPending ? 'var(--viz-2)' : 'var(--viz-axis)'} strokeWidth={1.5} />
              <text x={c.x + c.w / 2} y={c.y + 13} fontSize={11} textAnchor="middle" fill="var(--viz-ink)">
                {pathOps ? compactPath(c.k.detail) : c.k.label}
              </text>
            </g>
          ))}

          <text x={8} y={rootY - 8} fontSize={13} fill="var(--viz-ink)">
            Entry tree — a B-tree over keys{pathOps ? ', ordered by hash' : ''}
          </text>
          <rect x={W / 2 - 170} y={rootY} width={340} height={24} rx={4} fill="var(--viz-surface)" stroke="var(--viz-axis)" />
          <text x={W / 2} y={rootY + 16} fontSize={10.5} textAnchor="middle" fill="var(--viz-ink-2)">
            {leaves.length > 1 ? `root: ${separators.join('  |  ')}` : 'root = the only leaf'}
          </text>
          {leaves.map((leaf, li) => {
            const cx = 8 + (li % cols) * (leafW + gap);
            const cy = leafY + Math.floor(li / cols) * (leafH + gap);
            return (
              <g key={li}>
                <line x1={W / 2} y1={rootY + 24} x2={cx + leafW / 2} y2={cy} stroke="var(--viz-axis)" />
                <rect x={cx} y={cy} width={leafW} height={leafH} rx={4} fill="var(--viz-surface)" stroke="var(--viz-axis)" />
                <text x={cx + 6} y={cy + 13} fontSize={10.5} fill="var(--viz-ink-2)">
                  leaf page {li + 1}
                </text>
                {leaf.map((e, ei) => (
                  <EntryRow key={e.id} e={e} x={cx} y={cy + 18 + ei * rowH} w={leafW} maxRows={state.rows} touched={touched.has(e.id)} pathOps={pathOps} />
                ))}
              </g>
            );
          })}

          <text x={8} y={pendY} fontSize={13} fill="var(--viz-ink)">
            Pending list {fastupdate ? '' : '(fastupdate off: new rows bypass it)'}
            <tspan fontSize={11.5} fill="var(--viz-ink-2)">
              {' '}
              — {fmtNum(sum.pendingEntries)} entries on {sum.pendingPages} page{sum.pendingPages === 1 ? '' : 's'}; cleanup when pages exceed the limit
            </tspan>
          </text>
          {Array.from({ length: Math.max(limit, pagesShown.length) }, (_, pi) => {
            const px = 8 + pi * (pageW + 8);
            const used = pagesShown[pi] ?? 0;
            return (
              <g key={pi}>
                <rect x={px} y={pendBoxY} width={pageW} height={cell * 2 + 8} rx={3} fill="none" stroke="var(--viz-axis)" strokeDasharray={pi < pagesShown.length ? undefined : '3 2'} />
                {Array.from({ length: PENDING_PAGE_ENTRIES }, (_, ci) => (
                  <rect key={ci} x={px + 3 + ((ci % 6) * (pageW - 6)) / 6} y={pendBoxY + 4 + Math.floor(ci / 6) * (cell + 1)} width={(pageW - 6) / 6 - 1.5} height={cell} rx={1.5} fill={ci < used ? 'var(--viz-2)' : 'var(--viz-plane)'} />
                ))}
              </g>
            );
          })}
          {state.pendingPages.length > maxPagesShown ? (
            <text x={8 + maxPagesShown * (pageW + 8)} y={pendBoxY + 16} fontSize={10} fill="var(--viz-ink-2)">
              +{state.pendingPages.length - maxPagesShown} more
            </text>
          ) : null}
          <line x1={limitX} x2={limitX} y1={pendBoxY - 6} y2={pendBoxY + cell * 2 + 14} stroke="var(--viz-ink)" strokeDasharray="4 3" />
          <text x={limitX + 4} y={pendBoxY + cell * 2 + 16} fontSize={10.5} fill="var(--viz-ink-2)">
            limit
          </text>

          <text x={8} y={histY - 4} fontSize={13} fill="var(--viz-ink)">
            Work inside each INSERT
            <tspan fontSize={11.5} fill="var(--viz-ink-2)">
              {' '}
              — bars: entry-tree insertions · dots: pending-list appends
            </tspan>
          </text>
          <line x1={40} x2={W - 12} y1={histY + 8 + barMaxH} y2={histY + 8 + barMaxH} stroke="var(--viz-axis)" />
          <line x1={40} x2={W - 12} y1={histY + 8 + barMaxH - (histTop / histMax) * barMaxH} y2={histY + 8 + barMaxH - (histTop / histMax) * barMaxH} stroke="var(--viz-grid)" strokeDasharray="2 3" />
          <text x={34} y={histY + 8 + barMaxH - (histTop / histMax) * barMaxH + 4} fontSize={9.5} textAnchor="end" fill="var(--viz-ink-2)">
            {histTop}
          </text>
          <text x={34} y={histY + 8 + barMaxH} fontSize={9.5} textAnchor="end" fill="var(--viz-ink-2)">
            0
          </text>
          {hist.length === 0 ? (
            <text x={W / 2} y={histY + 8 + barMaxH / 2} fontSize={11} textAnchor="middle" fill="var(--viz-ink-2)">
              no INSERTs since the index was built
            </text>
          ) : null}
          {hist.map((h, i) => {
            const bx = 56 + i * bw;
            const bh = (h.entryInserts / histMax) * barMaxH;
            return (
              <g key={h.row}>
                {bh > 0 ? <rect x={bx} y={histY + 8 + barMaxH - bh} width={Math.max(2, bw - 3)} height={bh} rx={1.5} fill="var(--viz-3)" /> : null}
                {h.pendingAppend ? <circle cx={bx + (bw - 3) / 2} cy={histY + 8 + barMaxH + 7} r={2.6} fill="var(--viz-2)" /> : null}
                {h.cleanup && !hist.slice(Math.max(0, i - 3), i).some((p) => p.cleanup) ? (
                  <text x={bx + (bw - 3) / 2} y={histY + 4 + barMaxH - bh} fontSize={9} textAnchor="middle" fill="var(--viz-ink)">
                    cleanup
                  </text>
                ) : null}
              </g>
            );
          })}
          <text x={56} y={H - 6} fontSize={10.5} fill="var(--viz-ink-2)">
            older
          </text>
          <text x={W - 12} y={H - 6} fontSize={9.5} textAnchor="end" fill="var(--viz-ink-2)">
            newest INSERT
          </text>
        </svg>
      </TooltipHost>
    </VizPanel>
  );
}
