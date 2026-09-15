import { useMemo, useState, type CSSProperties, type DragEvent } from 'react';
import { VizPanel, Segmented, Choice, Slider, Button, Legend, Stats, Note, makeRng, fmtNum } from './Viz';

/**
 * A composite B-tree index over a 96-row `orders` table, and what one query does with it.
 *
 * Mechanism modelled (sources in the page prose):
 *  - Seek keys, PostgreSQL 17 and MySQL 8.0: equality on leading index columns, plus a range on the first
 *    column without an equality, bound the scanned leaf range. Conditions on later index columns are checked
 *    per entry inside the index (PostgreSQL: non-required Index Cond keys; MySQL/InnoDB: Index Condition
 *    Pushdown). Conditions on columns not in the index are checked after the heap fetch (Filter / Using where).
 *  - PostgreSQL 18 skip scan: every index column before the last column that has a condition, and that lacks
 *    an `=`, gets a skip array (a range skip array when it has a range), following _bt_num_array_keys in
 *    nbtpreprocesskeys.c. Every index condition then bounds the scan. MODEL SIMPLIFICATION of the run-time
 *    scheduling in nbtutils.c: the next group is reached by reading through the leaf when it is on the same leaf;
 *    by stepping onto the sibling leaf when the group ran to the leaf's edge or the primitive scan has already
 *    left its first leaf (and the sibling holds the next match); and by a new index search otherwise. The real
 *    heuristics have more cases, so treat the lab's "Index searches" for PostgreSQL 18 as illustrative.
 *  - MySQL's skip scan needs a query that reads only index columns; this lab's query is SELECT *, so it never
 *    applies. MySQL runs a full index scan (type=index) without Index Condition Pushdown.
 *  - Ordering (PostgreSQL build_index_pathkeys + pathkey_is_redundant): columns bound by `=` are dropped; the
 *    index's order stops at the first remaining column that the ORDER BY does not name. A backward scan flips
 *    direction AND null placement of every column. If the ORDER BY is a prefix of that order: no sort. If only a
 *    leading part matches: PostgreSQL uses Incremental Sort (13+), MySQL a filesort. Otherwise a full sort.
 *  - NULLs: PostgreSQL ASC ⇒ NULLS LAST, DESC ⇒ NULLS FIRST unless declared; MySQL has no NULLS clause and sorts
 *    NULL lowest (first in ASC, last in DESC), in indexes and ORDER BY alike.
 *  - LIMIT: with no sort the scan stops at the LIMIT-th returned row; with Incremental Sort (LIMIT ≤ 30 < 32,
 *    nodeIncrementalSort.c) it reads through the end of the presorted group holding that row plus one lookahead
 *    row; a full sort reads everything.
 *  - Plans: the lab always shows the path through the index being built, not the cost-based planner's choice.
 *    When that path neither seeks, filters inside the index, nor supplies a usable order, the note says a real
 *    planner would scan the table instead. The MySQL type/Extra lines are indicative flags, not full EXPLAIN output.
 *  - Counting: "entries read" counts entries inside the scanned range; a real scan also examines the entry past
 *    the end that stops it. Leaves hold 8 entries; the tree is root → 3 internal pages → 12 leaves.
 */

/* ------------------------------------------------------------------ data */

export const COLUMNS = ['tenant_id', 'status', 'created_at', 'shipped_at', 'total'] as const;
export type Col = (typeof COLUMNS)[number];
export type Val = number | string | null;
export type Row = { id: number } & Record<Col, Val>;
export type Engine = 'pg17' | 'pg18' | 'mysql';

export const LEAF_CAP = 8;
export const ROWS = 96;

export function makeOrders(seed = 7): Row[] {
  const rng = makeRng(seed);
  const rows: Row[] = [];
  for (let id = 1; id <= ROWS; id++) {
    const tenant = 1 + Math.floor(rng() * 4);
    const s = rng();
    const status = s < 0.25 ? 'new' : s < 0.85 ? 'paid' : 'void';
    const created = 1 + Math.floor(rng() * 60);
    const shipped = status === 'paid' && rng() < 0.8 ? Math.min(60, created + 1 + Math.floor(rng() * 5)) : null;
    const total = 10 + Math.floor(rng() * 990);
    rows.push({ id, tenant_id: tenant, status, created_at: created, shipped_at: shipped, total });
  }
  return rows;
}

/* ---------------------------------------------------------------- parsing */

export type Op = '=' | '<' | '<=' | '>' | '>=';
export type Pred = { col: Col; op: Op; value: number | string; text: string };
export type OrderItem = { col: Col; desc: boolean; nullsFirst: boolean; explicitNulls: boolean };
export type IndexCol = { col: Col; desc: boolean; nullsFirst: boolean };

type Tok = { t: 'id' | 'num' | 'str' | 'op' | 'comma'; v: string };

function tokenize(s: string): Tok[] | string {
  const out: Tok[] = [];
  let i = 0;
  while (i < s.length) {
    const ch = s[i];
    if (/\s/.test(ch)) { i++; continue; }
    if (ch === ',') { out.push({ t: 'comma', v: ',' }); i++; continue; }
    if (ch === "'") {
      const j = s.indexOf("'", i + 1);
      if (j < 0) return 'Unclosed string: add the closing quote.';
      out.push({ t: 'str', v: s.slice(i + 1, j) });
      i = j + 1;
      continue;
    }
    const two = s.slice(i, i + 2);
    if (two === '<=' || two === '>=' || two === '<>' || two === '!=') { out.push({ t: 'op', v: two }); i += 2; continue; }
    if (ch === '=' || ch === '<' || ch === '>') { out.push({ t: 'op', v: ch }); i++; continue; }
    const num = /^-?\d+/.exec(s.slice(i));
    if (num) { out.push({ t: 'num', v: num[0] }); i += num[0].length; continue; }
    const id = /^[A-Za-z_][A-Za-z_0-9]*/.exec(s.slice(i));
    if (id) { out.push({ t: 'id', v: id[0] }); i += id[0].length; continue; }
    return `Unexpected character “${ch}”.`;
  }
  return out;
}

const isCol = (v: string): v is Col => (COLUMNS as readonly string[]).includes(v.toLowerCase());

function valueFor(col: Col, tok: Tok | undefined): number | string | { error: string } {
  if (!tok) return { error: `Missing a value after ${col}.` };
  if (col === 'status') {
    if (tok.t !== 'str') return { error: `status is text: quote the value, e.g. status = 'paid'.` };
    return tok.v;
  }
  if (tok.t !== 'num') return { error: `${col} is an integer: compare it with a number.` };
  return Number(tok.v);
}

export function parseWhere(input: string): { preds: Pred[] } | { error: string } {
  const text = input.trim().replace(/^where\s+/i, '');
  if (!text) return { preds: [] };
  const toks = tokenize(text);
  if (typeof toks === 'string') return { error: toks };
  const preds: Pred[] = [];
  let i = 0;
  const kw = (k: string) => toks[i]?.t === 'id' && toks[i].v.toUpperCase() === k;
  while (i < toks.length) {
    const c = toks[i];
    if (!c || c.t !== 'id' || !isCol(c.v)) return { error: `Expected a column (${COLUMNS.join(', ')}), got “${c?.v ?? 'end'}”.` };
    const col = c.v.toLowerCase() as Col;
    i++;
    if (kw('BETWEEN')) {
      i++;
      const lo = valueFor(col, toks[i]);
      if (typeof lo === 'object') return lo;
      i++;
      if (!kw('AND')) return { error: 'BETWEEN needs “low AND high”.' };
      i++;
      const hi = valueFor(col, toks[i]);
      if (typeof hi === 'object') return hi;
      i++;
      const q = (v: number | string) => (typeof v === 'string' ? `'${v}'` : String(v));
      preds.push({ col, op: '>=', value: lo, text: `${col} >= ${q(lo)}` });
      preds.push({ col, op: '<=', value: hi, text: `${col} <= ${q(hi)}` });
    } else if (kw('IS') || kw('IN') || kw('LIKE')) {
      return { error: `${toks[i].v.toUpperCase()} is not modelled here: use =, <, <=, >, >= or BETWEEN.` };
    } else {
      const o = toks[i];
      if (!o || o.t !== 'op') return { error: `Expected an operator after ${col}.` };
      if (o.v === '<>' || o.v === '!=') return { error: `${o.v} is not modelled here: use =, <, <=, >, >= or BETWEEN.` };
      i++;
      const v = valueFor(col, toks[i]);
      if (typeof v === 'object') return v;
      i++;
      preds.push({ col, op: o.v as Op, value: v, text: `${col} ${o.v} ${typeof v === 'string' ? `'${v}'` : v}` });
    }
    if (i >= toks.length) break;
    if (kw('OR')) return { error: 'OR is not modelled: one index scan uses only AND-ed conditions (OR needs a BitmapOr or a UNION).' };
    if (!kw('AND')) return { error: `Expected AND, got “${toks[i].v}”.` };
    i++;
    if (i >= toks.length) return { error: 'Dangling AND at the end.' };
  }
  return { preds };
}

export function parseOrderBy(input: string, engine: Engine): { items: OrderItem[] } | { error: string } {
  const text = input.trim().replace(/^order\s+by\s+/i, '');
  if (!text) return { items: [] };
  const toks = tokenize(text);
  if (typeof toks === 'string') return { error: toks };
  const items: OrderItem[] = [];
  let i = 0;
  const kw = (k: string) => toks[i]?.t === 'id' && toks[i].v.toUpperCase() === k;
  while (i < toks.length) {
    const c = toks[i];
    if (!c || c.t !== 'id' || !isCol(c.v)) return { error: `Expected a column to sort by, got “${c?.v ?? 'end'}”.` };
    const col = c.v.toLowerCase() as Col;
    i++;
    let desc = false;
    if (kw('ASC')) i++;
    else if (kw('DESC')) { desc = true; i++; }
    let nullsFirst = engine === 'mysql' ? !desc : desc;
    let explicitNulls = false;
    if (kw('NULLS')) {
      if (engine === 'mysql') return { error: 'MySQL has no NULLS FIRST / NULLS LAST: NULLs sort first in ASC and last in DESC.' };
      i++;
      if (kw('FIRST')) nullsFirst = true;
      else if (kw('LAST')) nullsFirst = false;
      else return { error: 'NULLS must be followed by FIRST or LAST.' };
      explicitNulls = true;
      i++;
    }
    items.push({ col, desc, nullsFirst, explicitNulls });
    if (i >= toks.length) break;
    if (toks[i].t !== 'comma') return { error: `Expected a comma, ASC, DESC${engine === 'mysql' ? '' : ' or NULLS'}, got “${toks[i].v}”.` };
    i++;
  }
  return { items };
}

/* ------------------------------------------------------------------ model */

function cmpVal(a: Val, b: Val, desc: boolean, nullsFirst: boolean) {
  if (a === null && b === null) return 0;
  if (a === null) return nullsFirst ? -1 : 1;
  if (b === null) return nullsFirst ? 1 : -1;
  const c = typeof a === 'number' && typeof b === 'number' ? a - b : String(a) < String(b) ? -1 : String(a) > String(b) ? 1 : 0;
  return desc ? -c : c;
}

function holds(p: Pred, v: Val) {
  if (v === null) return false; // SQL: a comparison with NULL is never true
  const c = typeof v === 'number' ? v - (p.value as number) : String(v) < String(p.value) ? -1 : String(v) > String(p.value) ? 1 : 0;
  switch (p.op) {
    case '=': return c === 0;
    case '<': return c < 0;
    case '<=': return c <= 0;
    case '>': return c > 0;
    case '>=': return c >= 0;
  }
}

/** Index entries in key order (heap TID — here the row id — breaks ties, as in nbtree). */
export function buildIndex(rows: Row[], cols: IndexCol[]) {
  const entries = rows.slice().sort((r1, r2) => {
    for (const c of cols) {
      const d = cmpVal(r1[c.col], r2[c.col], c.desc, c.nullsFirst);
      if (d !== 0) return d;
    }
    return r1.id - r2.id;
  });
  const leaves: Row[][] = [];
  for (let i = 0; i < entries.length; i += LEAF_CAP) leaves.push(entries.slice(i, i + LEAF_CAP));
  return { entries, leaves };
}

export type EntryState = 'unread' | 'rejected-index' | 'rejected-heap' | 'fetched';
export type SortKind = 'none' | 'incremental' | 'full';

export type ScanResult = {
  engine: Engine;
  seekPreds: Pred[];
  indexFilterPreds: Pred[];
  heapPreds: Pred[];
  skipCols: { col: Col; ranged: boolean; distinct: number }[];
  fullIndexScan: boolean;
  backward: boolean;
  sort: SortKind;
  presorted: number;
  queryKeys: OrderItem[];
  searches: number;
  searchStarts: number[]; // index positions where a descent lands
  readOrder: number[]; // positions, in the order read
  state: EntryState[];
  outputRank: (number | null)[];
  entriesRead: number;
  rejectedInIndex: number;
  heapFetched: number;
  rejectedByFilter: number;
  returned: number;
  leavesRead: number;
  bitmap: { applicable: boolean; why: string; entries: number; heapRows: number; indexes: Col[] };
  planLines: string[];
  lookahead: number; // position of the row Incremental Sort read past its group, or -1
};

export function scan(rows: Row[], colsIn: IndexCol[], preds: Pred[], orderIn: OrderItem[], limit: number, engine: Engine): ScanResult {
  // MySQL has no NULLS clause: NULL is the lowest value, in every index column and every ORDER BY item.
  const cols = engine === 'mysql' ? colsIn.map((c) => ({ ...c, nullsFirst: !c.desc })) : colsIn;
  const order = engine === 'mysql' ? orderIn.map((o) => ({ ...o, nullsFirst: !o.desc })) : orderIn;
  const { entries } = buildIndex(rows, cols);
  const n = entries.length;
  const idxCols = cols.map((c) => c.col);
  const posOf = (col: Col) => idxCols.indexOf(col);
  const predsOn = (col: Col) => preds.filter((p) => p.col === col);
  const hasEq = (col: Col) => predsOn(col).some((p) => p.op === '=');

  // --- which conditions bound the scan
  const boundaryCols: Col[] = [];
  for (const c of idxCols) {
    const ps = predsOn(c);
    if (ps.some((p) => p.op === '=')) { boundaryCols.push(c); continue; }
    if (ps.length) boundaryCols.push(c);
    break;
  }
  let seekPreds = preds.filter((p) => boundaryCols.includes(p.col));
  let indexFilterPreds = preds.filter((p) => posOf(p.col) >= 0 && !boundaryCols.includes(p.col));
  let heapPreds = preds.filter((p) => posOf(p.col) < 0);
  const fullIndexScan = seekPreds.length === 0;

  const skipCols: ScanResult['skipCols'] = [];
  const lastPredPos = Math.max(-1, ...preds.map((p) => posOf(p.col)));
  if (engine === 'pg18' && indexFilterPreds.length) {
    for (let i = 0; i < lastPredPos; i++) {
      const c = idxCols[i];
      if (!hasEq(c)) {
        const ps = predsOn(c);
        const vals = new Set(entries.filter((r) => ps.every((p) => holds(p, r[c]))).map((r) => r[c]));
        skipCols.push({ col: c, ranged: ps.length > 0, distinct: vals.size });
      }
    }
    seekPreds = preds.filter((p) => posOf(p.col) >= 0);
    indexFilterPreds = [];
  }
  if (engine === 'mysql' && fullIndexScan) {
    // type=index: Index Condition Pushdown only applies to range, ref, eq_ref and ref_or_null access.
    heapPreds = preds.slice();
    indexFilterPreds = [];
  }

  // --- ordering
  const eqCols = new Set(preds.filter((p) => p.op === '=').map((p) => p.col));
  const queryKeys: OrderItem[] = [];
  for (const it of order) if (!eqCols.has(it.col) && !queryKeys.some((q) => q.col === it.col)) queryKeys.push(it);
  const orderCols = new Set(queryKeys.map((q) => q.col));
  const indexKeys = (backward: boolean) => {
    const keys: { col: Col; desc: boolean; nullsFirst: boolean }[] = [];
    for (const c of cols) {
      if (eqCols.has(c.col)) continue; // redundant: bound to a constant
      if (!orderCols.has(c.col)) break; // not an interesting order: later columns cannot help
      keys.push({ col: c.col, desc: backward ? !c.desc : c.desc, nullsFirst: backward ? !c.nullsFirst : c.nullsFirst });
    }
    return keys;
  };
  const matchCount = (backward: boolean) => {
    const ik = indexKeys(backward);
    let k = 0;
    while (k < queryKeys.length && k < ik.length && ik[k].col === queryKeys[k].col && ik[k].desc === queryKeys[k].desc && ik[k].nullsFirst === queryKeys[k].nullsFirst) k++;
    return k;
  };
  const fwd = matchCount(false);
  const bwd = matchCount(true);
  let sort: SortKind = 'none';
  let backward = false;
  let presorted = 0;
  if (queryKeys.length) {
    if (fwd === queryKeys.length) sort = 'none';
    else if (bwd === queryKeys.length) { sort = 'none'; backward = true; }
    else {
      presorted = Math.max(fwd, bwd);
      backward = bwd > fwd;
      sort = presorted > 0 && engine !== 'mysql' ? 'incremental' : 'full';
      if (sort === 'full') { presorted = 0; backward = false; }
    }
  }

  // --- which entries the scan reads, in order
  const seq = Array.from({ length: n }, (_, i) => (backward ? n - 1 - i : i));
  const leafOf = (p: number) => Math.floor(p / LEAF_CAP);
  const allIndexPreds = preds.filter((p) => posOf(p.col) >= 0);
  const passIndex = (r: Row) => indexFilterPreds.every((p) => holds(p, r[p.col])) && seekPreds.every((p) => holds(p, r[p.col]));
  const readOrder: number[] = [];
  const searchStarts: number[] = [];
  if (skipCols.length) {
    // Simplified PostgreSQL 18 primitive-scan scheduling (nbtutils.c _bt_advance_array_keys). A group is a run
    // of equal skip-column values. Each group's search lands on its first matching entry (or, when the group
    // has none, on its first entry, which is read and rejected). The next group is reached by reading through
    // the leaf when it starts on the same leaf; by stepping onto the sibling when the previous group ran to the
    // leaf's edge, or the primitive scan has already left its first leaf, and the landing is on that sibling;
    // otherwise by a new index search.
    const skipSet = new Set(skipCols.map((k) => k.col));
    const firstSkip = idxCols.indexOf(skipCols[0].col);
    const inRegion = (r: Row) =>
      preds.every((q) => {
        const qp = posOf(q.col);
        return qp < 0 || !(qp < firstSkip || skipSet.has(q.col)) || holds(q, r[q.col]);
      });
    const isMatch = (r: Row) => allIndexPreds.every((q) => holds(q, r[q.col]));
    const region = seq.filter((p) => inRegion(entries[p]));
    const groups: number[][] = [];
    const gkey = (r: Row) => skipCols.map((k) => fmtVal(r[k.col])).join('\u0000');
    for (const p of region) {
      const last = groups[groups.length - 1];
      if (last && gkey(entries[last[last.length - 1]]) === gkey(entries[p]) && Math.abs(last[last.length - 1] - p) === 1) last.push(p);
      else groups.push([p]);
    }
    const step = backward ? -1 : 1;
    const leafEdge = (p: number) => (backward ? leafOf(p) * LEAF_CAP : Math.min(n - 1, (leafOf(p) + 1) * LEAF_CAP - 1));
    const siblingStart = (p: number) => (backward ? leafOf(p) * LEAF_CAP - 1 : (leafOf(p) + 1) * LEAF_CAP);
    let prev = -1;
    let firstLeaf = true;
    const goTo = (q: number) => {
      if (prev < 0) { searchStarts.push(q); return; }
      if (leafOf(q) === leafOf(prev)) {
        for (let x = prev + step; x !== q; x += step) readOrder.push(x);
        return;
      }
      const sib = siblingStart(prev);
      const onSibling = sib >= 0 && sib < n && leafOf(q) === leafOf(sib);
      if ((prev === leafEdge(prev) || !firstLeaf) && onSibling) {
        for (let x = sib; x !== q; x += step) readOrder.push(x);
        firstLeaf = false;
      } else {
        searchStarts.push(q);
        firstLeaf = true;
      }
    };
    for (const g of groups) {
      const ms = g.filter((p) => isMatch(entries[p]));
      const visit = ms.length ? ms : [g[0]];
      for (const p of visit) {
        goTo(p);
        readOrder.push(p);
        prev = p;
      }
    }
    if (!groups.length) searchStarts.push(backward ? n - 1 : 0);
  } else {
    const inRange = seq.filter((p) => seekPreds.every((q) => holds(q, entries[p][q.col])));
    readOrder.push(...inRange);
    searchStarts.push(inRange.length ? inRange[0] : backward ? n - 1 : 0);
  }

  // --- execute with LIMIT
  const state: EntryState[] = Array(n).fill('unread');
  const outputs: number[] = [];
  let entriesRead = 0;
  let pivot: Row | null = null;
  let lookahead = -1;
  const samePrefix = (a: Row, b: Row) => queryKeys.slice(0, presorted).every((k) => cmpVal(a[k.col], b[k.col], false, false) === 0);
  for (const p of readOrder) {
    const r = entries[p];
    entriesRead++;
    if (!passIndex(r)) { state[p] = 'rejected-index'; continue; }
    if (!heapPreds.every((q) => holds(q, r[q.col]))) { state[p] = 'rejected-heap'; continue; }
    state[p] = 'fetched';
    // Incremental Sort with a LIMIT below 32 compares prefix keys once it holds LIMIT rows: the first row of the
    // next group is read (as a lookahead) and ends the batch.
    if (limit > 0 && sort === 'incremental' && pivot && !samePrefix(pivot, r)) { lookahead = p; break; }
    outputs.push(p);
    if (limit > 0 && sort === 'none' && outputs.length >= limit) break;
    if (limit > 0 && sort === 'incremental' && outputs.length === limit) pivot = r;
  }
  let emitted = outputs.slice();
  if (sort !== 'none') {
    const readPos = new Map(readOrder.map((p, i) => [p, i]));
    emitted.sort((a, b) => {
      for (const k of order) {
        const d = cmpVal(entries[a][k.col], entries[b][k.col], k.desc, k.nullsFirst);
        if (d !== 0) return d;
      }
      return (readPos.get(a) ?? 0) - (readPos.get(b) ?? 0);
    });
  }
  if (limit > 0) emitted = emitted.slice(0, limit);
  const outputRank: (number | null)[] = Array(n).fill(null);
  emitted.forEach((p, i) => (outputRank[p] = i + 1));

  const readSet = readOrder.slice(0, entriesRead);
  const leavesRead = new Set(readSet.map(leafOf)).size;
  const rejectedInIndex = state.filter((s) => s === 'rejected-index').length;
  const rejectedByFilter = state.filter((s) => s === 'rejected-heap').length;
  const heapFetched = rejectedByFilter + state.filter((s) => s === 'fetched').length;
  const searches = searchStarts.filter((s) => readSet.length === 0 || readSet.includes(s)).length || 1;

  // --- the alternative: one single-column index per condition column, combined
  const predCols = Array.from(new Set(preds.map((p) => p.col)));
  let bitmap: ScanResult['bitmap'];
  if (engine === 'mysql') {
    const eq = Array.from(new Set(preds.filter((p) => p.op === '=').map((p) => p.col)));
    if (eq.length < 2) bitmap = { applicable: false, why: 'Index Merge intersection needs equality on two or more single-column indexes', entries: 0, heapRows: 0, indexes: eq };
    else {
      const ent = eq.reduce((a, c) => a + rows.filter((r) => predsOn(c).filter((p) => p.op === '=').every((p) => holds(p, r[c]))).length, 0);
      const heapRows = rows.filter((r) => eq.every((c) => predsOn(c).filter((p) => p.op === '=').every((p) => holds(p, r[c])))).length;
      bitmap = { applicable: true, why: 'Using intersect(…); rows come back unordered', entries: ent, heapRows, indexes: eq };
    }
  } else if (predCols.length < 2) {
    bitmap = { applicable: false, why: 'BitmapAnd needs conditions on two or more columns', entries: 0, heapRows: 0, indexes: predCols };
  } else {
    const ent = predCols.reduce((a, c) => a + rows.filter((r) => predsOn(c).every((p) => holds(p, r[c]))).length, 0);
    const heapRows = rows.filter((r) => preds.every((p) => holds(p, r[p.col]))).length;
    bitmap = { applicable: true, why: 'Bitmap Heap Scan visits rows in physical order, so any ORDER BY needs a Sort', entries: ent, heapRows, indexes: predCols };
  }

  // --- plan shape
  const planLines: string[] = [];
  const keyText = (k: OrderItem) => `${k.col}${k.desc ? ' DESC' : ''}${k.nullsFirst !== k.desc ? (k.nullsFirst ? ' NULLS FIRST' : ' NULLS LAST') : ''}`;
  const conj = (ps: Pred[]) => (ps.length === 1 ? `(${ps[0].text})` : `(${ps.map((p) => `(${p.text})`).join(' AND ')})`);
  if (engine === 'mysql') {
    const type = fullIndexScan ? 'index' : seekPreds.every((p) => p.op === '=') ? 'ref' : 'range';
    const extra = [
      indexFilterPreds.length ? 'Using index condition' : '',
      heapPreds.length ? 'Using where' : '',
      backward ? 'Backward index scan' : '',
      sort !== 'none' ? 'Using filesort' : '',
    ].filter(Boolean);
    planLines.push(`type: ${type}   key: orders_idx`);
    planLines.push(`Extra: ${extra.length ? extra.join('; ') : '(none)'}`);
  } else {
    let pad = '';
    if (limit > 0) { planLines.push(`Limit (rows=${limit})`); pad = '  ->  '; }
    if (sort !== 'none') {
      planLines.push(`${pad}${sort === 'incremental' ? 'Incremental Sort' : 'Sort'}`);
      planLines.push(`${' '.repeat(pad.length + 6)}Sort Key: ${queryKeys.map(keyText).join(', ')}`);
      // explain.c prints Presorted Key as bare expressions, without DESC / NULLS options.
      if (sort === 'incremental') planLines.push(`${' '.repeat(pad.length + 6)}Presorted Key: ${queryKeys.slice(0, presorted).map((k) => k.col).join(', ')}`);
      pad = ' '.repeat(pad.length) + '  ->  ';
    }
    const ind = ' '.repeat(pad.length + 6);
    planLines.push(`${pad}Index Scan${backward ? ' Backward' : ''} using orders_idx on orders`);
    const ic = preds.filter((p) => posOf(p.col) >= 0);
    if (ic.length) planLines.push(`${ind}Index Cond: ${conj(ic)}`);
    const hf = preds.filter((p) => posOf(p.col) < 0);
    if (hf.length) planLines.push(`${ind}Filter: ${conj(hf)}`);
    if (engine === 'pg18') planLines.push(`${ind}Index Searches: ${searches}`);
  }

  return {
    engine, seekPreds, indexFilterPreds, heapPreds, skipCols, fullIndexScan, backward, sort, presorted, queryKeys,
    searches, searchStarts, readOrder: readSet, state, outputRank, entriesRead, rejectedInIndex, heapFetched,
    rejectedByFilter, returned: emitted.length, leavesRead, bitmap, planLines, lookahead,
  };
}

/* --------------------------------------------------------------- presets */

type Preset = { label: string; index: [Col, boolean?, boolean?][]; where: string; order: string; limit: number; engine?: Engine };

export const PRESETS: Record<string, Preset> = {
  prefix: { label: 'Equality on a leftmost prefix', index: [['tenant_id'], ['status'], ['created_at']], where: "tenant_id = 3 AND status = 'paid'", order: 'created_at', limit: 5 },
  gap: { label: 'A gap: middle column skipped', index: [['tenant_id'], ['status'], ['created_at']], where: 'tenant_id = 3 AND created_at >= 40', order: '', limit: 0 },
  ers: { label: 'Range before sort (E, R, S)', index: [['tenant_id'], ['total'], ['created_at']], where: 'tenant_id = 2 AND total > 100', order: 'created_at', limit: 5 },
  esr: { label: 'Sort before range (E, S, R)', index: [['tenant_id'], ['created_at'], ['total']], where: 'tenant_id = 2 AND total > 100', order: 'created_at', limit: 5 },
  noLeading: { label: 'No condition on the leading column', index: [['status'], ['created_at']], where: 'created_at BETWEEN 20 AND 24', order: '', limit: 0 },
  leadingRange: { label: 'Range on the leading column', index: [['created_at'], ['status']], where: "created_at > 10 AND status = 'void'", order: '', limit: 0 },
  mixed: { label: 'Mixed directions in ORDER BY', index: [['tenant_id'], ['created_at']], where: '', order: 'tenant_id DESC, created_at ASC', limit: 5 },
  nulls: { label: 'DESC NULLS LAST on a default index', index: [['shipped_at']], where: '', order: 'shipped_at DESC NULLS LAST', limit: 5 },
};

const presetIndex = (p: Preset): IndexCol[] => p.index.map(([col, desc = false, nf]) => ({ col, desc, nullsFirst: nf ?? desc }));

/* -------------------------------------------------------------------- UI */

const fmtVal = (v: Val) => (v === null ? '∅' : String(v));
const chip: CSSProperties = {
  display: 'inline-flex', alignItems: 'center', gap: '0.3rem', padding: '0.25rem 0.4rem', borderRadius: 6,
  border: '1px solid var(--viz-border)', background: 'var(--viz-plane)', color: 'var(--viz-ink)', fontSize: '0.8rem',
};
const textInput: CSSProperties = {
  font: 'inherit', color: 'var(--viz-ink)', background: 'var(--viz-plane)', border: '1px solid var(--viz-border)',
  borderRadius: 6, padding: '0.3rem 0.5rem', width: 'min(22rem, 78vw)',
};
const small: CSSProperties = { padding: '0.05rem 0.35rem', fontSize: '0.72rem' };

const STATE_COLOR: Record<EntryState, string> = {
  unread: 'none',
  fetched: 'var(--viz-1)',
  'rejected-index': 'var(--viz-2)',
  'rejected-heap': 'var(--viz-3)',
};

export default function CompositeIndexLeftmostPrefixLab() {
  const rows = useMemo(() => makeOrders(), []);
  const [engine, setEngineRaw] = useState<Engine>('pg17');
  const [preset, setPreset] = useState<string>('prefix');
  const [cols, setCols] = useState<IndexCol[]>(presetIndex(PRESETS.prefix));
  const [where, setWhere] = useState(PRESETS.prefix.where);
  const [orderText, setOrderText] = useState(PRESETS.prefix.order);
  const [limit, setLimit] = useState(PRESETS.prefix.limit);
  const [drag, setDrag] = useState<{ from: 'index' | 'pool'; col: Col } | null>(null);

  const applyPreset = (key: string) => {
    const p = PRESETS[key];
    setPreset(key);
    setCols(presetIndex(p).map((c) => (engine === 'mysql' ? { ...c, nullsFirst: !c.desc } : c)));
    setWhere(p.where);
    setOrderText(engine === 'mysql' ? p.order.replace(/\s+NULLS\s+(FIRST|LAST)/gi, '') : p.order);
    setLimit(p.limit);
  };
  const setEngine = (e: Engine) => {
    setEngineRaw(e);
    // MySQL has no NULLS clause: NULL is the lowest value in every index column.
    setCols((cs) => cs.map((c) => (e === 'mysql' ? { ...c, nullsFirst: !c.desc } : { ...c, nullsFirst: c.desc })));
    if (e === 'mysql') setOrderText((t) => t.replace(/\s+NULLS\s+(FIRST|LAST)/gi, ''));
  };

  const parsedWhere = useMemo(() => parseWhere(where), [where]);
  const parsedOrder = useMemo(() => parseOrderBy(orderText, engine), [orderText, engine]);
  const error = 'error' in parsedWhere ? `WHERE: ${parsedWhere.error}` : 'error' in parsedOrder ? `ORDER BY: ${parsedOrder.error}` : null;
  const preds = 'preds' in parsedWhere ? parsedWhere.preds : [];
  const items = 'items' in parsedOrder ? parsedOrder.items : [];

  const idx = useMemo(() => buildIndex(rows, cols), [rows, cols]);
  const res = useMemo(() => (cols.length ? scan(rows, cols, error ? [] : preds, error ? [] : items, limit, engine) : null), [rows, cols, preds, items, limit, engine, error]);

  const pool = COLUMNS.filter((c) => !cols.some((ic) => ic.col === c));
  const move = (i: number, d: number) => setCols((cs) => {
    const j = i + d;
    if (j < 0 || j >= cs.length) return cs;
    const next = cs.slice();
    [next[i], next[j]] = [next[j], next[i]];
    return next;
  });
  const toggleDir = (i: number) => setCols((cs) => cs.map((c, k) => (k === i ? { ...c, desc: !c.desc, nullsFirst: engine === 'mysql' ? c.desc : !c.desc } : c)));
  const toggleNulls = (i: number) => setCols((cs) => cs.map((c, k) => (k === i ? { ...c, nullsFirst: !c.nullsFirst } : c)));
  const remove = (i: number) => setCols((cs) => cs.filter((_, k) => k !== i));
  const add = (col: Col, at?: number) => setCols((cs) => {
    if (cs.length >= 4 || cs.some((c) => c.col === col)) return cs;
    const next = cs.slice();
    next.splice(at ?? next.length, 0, { col, desc: false, nullsFirst: engine === 'mysql' });
    return next;
  });
  const onDropAt = (at: number) => (e: DragEvent) => {
    e.preventDefault();
    if (!drag) return;
    if (drag.from === 'pool') add(drag.col, at);
    else setCols((cs) => {
      const from = cs.findIndex((c) => c.col === drag.col);
      if (from < 0) return cs;
      const next = cs.slice();
      const [it] = next.splice(from, 1);
      next.splice(at > from ? at - 1 : at, 0, it);
      return next;
    });
    setDrag(null);
  };

  const createSql = `CREATE INDEX orders_idx ON orders (${cols.map((c) => `${c.col}${c.desc ? ' DESC' : ''}${engine !== 'mysql' && c.nullsFirst !== c.desc ? (c.nullsFirst ? ' NULLS FIRST' : ' NULLS LAST') : ''}`).join(', ')})`;

  // ---- geometry: root and the three internal pages on the left, each internal page's four leaves in a row
  const perInner = 4;
  const nLeaves = idx.leaves.length;
  const nInner = Math.ceil(nLeaves / perInner);
  const innerW = 96;
  const leafGap = 8;
  const leafW = 116;
  const leafX0 = innerW + 22;
  const rowH = 13;
  const leafH = LEAF_CAP * rowH + 8;
  const rootH = 26;
  const rowTop0 = rootH + 22;
  const rowPitch = leafH + 36;
  const width = leafX0 + perInner * leafW + (perInner - 1) * leafGap + 2;
  const height = rowTop0 + nInner * rowPitch - 8;
  const leafX = (i: number) => leafX0 + (i % perInner) * (leafW + leafGap);
  const rowTop = (j: number) => rowTop0 + j * rowPitch;
  const leafY = (i: number) => rowTop(Math.floor(i / perInner)) + 14;
  const keyLabel = (r: Row) => cols.map((c) => fmtVal(r[c.col])).join('·');
  const searchLeaves = res ? res.searchStarts.filter((s) => res.readOrder.length === 0 || res.readOrder.includes(s)).map((s) => Math.floor(s / LEAF_CAP)) : [];

  const sortBadge = !res
    ? null
    : res.sort === 'none'
      ? { color: 'var(--viz-good)', text: res.queryKeys.length ? `No sort: index order serves ORDER BY (${res.backward ? 'backward' : 'forward'} scan)` : items.length ? 'No sort: every ORDER BY column is fixed by an equality' : 'No ORDER BY: nothing to sort' }
      : res.sort === 'incremental'
        ? { color: 'var(--viz-warning)', text: `Sort needed — Incremental Sort, ${res.presorted} of ${res.queryKeys.length} key${res.queryKeys.length > 1 ? 's' : ''} presorted` }
        : { color: 'var(--viz-critical)', text: `Sort needed — ${engine === 'mysql' ? 'Using filesort' : 'full Sort'} of every row the scan returns` };

  const engineName = engine === 'pg17' ? 'PostgreSQL 17' : engine === 'pg18' ? 'PostgreSQL 18' : 'MySQL 8.0 (InnoDB)';

  return (
    <VizPanel
      title="Which part of a composite index does this query actually seek?"
      subtitle="Build the index by dragging columns into order and flipping ASC/DESC and NULL placement, then edit the query. The tree shows the entries the scan reads, which ones the index conditions reject, and whether the rows still need a sort."
      controls={
        <>
          <Segmented label="Engine" value={engine} onChange={setEngine} options={[{ value: 'pg17', label: 'PostgreSQL 17' }, { value: 'pg18', label: 'PostgreSQL 18' }, { value: 'mysql', label: 'MySQL 8.0' }]} />
          <Choice label="Scenario" value={preset} onChange={applyPreset} options={Object.entries(PRESETS).map(([value, p]) => ({ value, label: p.label }))} />
          <Button onClick={() => applyPreset(preset)} title="Restore this scenario's index and query">Reset scenario</Button>
          <Slider label="LIMIT" min={0} max={30} value={limit} onChange={setLimit} format={(v) => (v === 0 ? 'none' : String(v))} />
        </>
      }
      legend={
        <Legend
          items={[
            { label: 'Read, passed the index conditions → heap row fetched', color: STATE_COLOR.fetched },
            { label: 'Read, rejected inside the index (no heap visit)', color: STATE_COLOR['rejected-index'] },
            { label: 'Heap row fetched, then rejected by the Filter', color: STATE_COLOR['rejected-heap'] },
            { label: 'Index search (descent) and scanned range', color: 'var(--viz-ink-2)', shape: 'line' },
          ]}
        />
      }
      stats={
        res ? (
          <Stats
            items={[
              { label: 'Index searches', value: fmtNum(res.searches), hint: 'Root-to-leaf descents. PostgreSQL 18 reports this as “Index Searches” in EXPLAIN ANALYZE.' },
              { label: 'Leaf entries read', value: `${fmtNum(res.entriesRead)} of ${ROWS}`, hint: 'Entries inside the scanned ranges, until the scan stops' },
              { label: 'Leaf pages read', value: `${res.leavesRead} of ${nLeaves}` },
              { label: 'Rejected inside index', value: fmtNum(res.rejectedInIndex) },
              { label: 'Heap rows fetched', value: fmtNum(res.heapFetched) },
              { label: 'Rows returned', value: fmtNum(res.returned) },
              {
                label: engine === 'mysql' ? 'Index Merge alternative' : 'BitmapAnd alternative',
                value: res.bitmap.applicable ? `${res.bitmap.entries} entries, ${res.bitmap.heapRows} rows` : 'n/a',
                hint: res.bitmap.applicable ? `Separate single-column indexes on ${res.bitmap.indexes.join(', ')}: every matching entry of each is read, then intersected. ${res.bitmap.why}.` : res.bitmap.why,
              },
            ]}
          />
        ) : null
      }
      note={
        <Note>
          {error ? (
            <><strong>{error}</strong> The tree shows the index with no query applied until the clause parses.</>
          ) : !res ? (
            <><strong>The index has no columns.</strong> Drag a column into the index, or use its “+” button.</>
          ) : (
            <NoteText res={res} engine={engineName} limit={limit} cols={cols} />
          )}
        </Note>
      }
      table={
        res ? (
          <table className="viz-table">
            <thead>
              <tr><th>Read #</th><th>Leaf</th><th>Key ({cols.map((c) => c.col).join(', ')})</th><th>Outcome</th><th>Output position</th></tr>
            </thead>
            <tbody>
              {res.readOrder.map((p, i) => (
                <tr key={p}>
                  <td>{i + 1}</td>
                  <td>{Math.floor(p / LEAF_CAP) + 1}</td>
                  <td>{keyLabel(idx.entries[p])}</td>
                  <td>{res.state[p] === 'fetched' ? 'heap row fetched' : res.state[p] === 'rejected-index' ? 'rejected inside the index' : 'fetched, rejected by Filter'}</td>
                  <td>{res.outputRank[p] ?? '—'}</td>
                </tr>
              ))}
              {res.readOrder.length === 0 ? <tr><td colSpan={5}>No entries read.</td></tr> : null}
            </tbody>
          </table>
        ) : null
      }
    >
      <div className="viz-controls" style={{ alignItems: 'flex-start' }}>
        <div className="viz-control" style={{ minWidth: 0, maxWidth: '100%' }}>
          <span>Index columns, in key order (drag to reorder; at most 4)</span>
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.35rem', alignItems: 'center' }} onDragOver={(e) => e.preventDefault()} onDrop={onDropAt(cols.length)}>
            {cols.map((c, i) => (
              <span key={c.col} draggable style={{ ...chip, cursor: 'grab', borderColor: 'var(--viz-ink-2)' }} onDragStart={() => setDrag({ from: 'index', col: c.col })} onDragOver={(e) => e.preventDefault()} onDrop={(e) => { e.stopPropagation(); onDropAt(i)(e); }}>
                <strong>{i + 1}. {c.col}</strong>
                <button type="button" style={small} onClick={() => toggleDir(i)} aria-label={`${c.col} direction: ${c.desc ? 'DESC' : 'ASC'}`}>{c.desc ? 'DESC' : 'ASC'}</button>
                {engine !== 'mysql' ? (
                  <button type="button" style={small} onClick={() => toggleNulls(i)} aria-label={`${c.col} null placement: ${c.nullsFirst ? 'NULLS FIRST' : 'NULLS LAST'}`}>{c.nullsFirst ? 'NULLS FIRST' : 'NULLS LAST'}</button>
                ) : null}
                <button type="button" style={small} onClick={() => move(i, -1)} disabled={i === 0} aria-label={`Move ${c.col} left`}>←</button>
                <button type="button" style={small} onClick={() => move(i, 1)} disabled={i === cols.length - 1} aria-label={`Move ${c.col} right`}>→</button>
                <button type="button" style={small} onClick={() => remove(i)} aria-label={`Remove ${c.col}`}>×</button>
              </span>
            ))}
            {cols.length === 0 ? <span style={{ ...chip, color: 'var(--viz-ink-muted)' }}>drop a column here</span> : null}
          </div>
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.35rem', alignItems: 'center', marginTop: '0.3rem' }}>
            <span style={{ fontSize: '0.75rem', color: 'var(--viz-ink-2)' }}>Not indexed:</span>
            {pool.map((c) => (
              <span key={c} draggable style={{ ...chip, cursor: 'grab' }} onDragStart={() => setDrag({ from: 'pool', col: c })}>
                {c}
                <button type="button" style={small} onClick={() => add(c)} disabled={cols.length >= 4} aria-label={`Add ${c} to the index`}>+</button>
              </span>
            ))}
          </div>
        </div>
      </div>
      <div className="viz-controls">
        <label className="viz-control">
          <span>WHERE</span>
          <input type="text" value={where} onChange={(e) => setWhere(e.currentTarget.value)} style={textInput} spellCheck={false} aria-label="WHERE clause" />
        </label>
        <label className="viz-control">
          <span>ORDER BY</span>
          <input type="text" value={orderText} onChange={(e) => setOrderText(e.currentTarget.value)} style={textInput} spellCheck={false} aria-label="ORDER BY clause" />
        </label>
      </div>
      <p style={{ margin: '0 0 0.3rem', fontSize: '0.78rem', color: 'var(--viz-ink-2)' }}>
        <code>{createSql}</code>
        <br />
        <code>SELECT * FROM orders{preds.length && !error ? ` WHERE ${where.trim().replace(/^where\s+/i, '')}` : ''}{items.length && !error ? ` ORDER BY ${orderText.trim().replace(/^order\s+by\s+/i, '')}` : ''}{limit ? ` LIMIT ${limit}` : ''}</code>
      </p>
      {sortBadge ? (
        <div data-testid="sort-badge" data-sort={res?.sort} style={{ display: 'inline-block', margin: '0.2rem 0 0.4rem', padding: '0.2rem 0.6rem', borderRadius: 999, border: `2px solid ${sortBadge.color}`, background: 'var(--viz-surface)', color: 'var(--viz-ink)', fontSize: '0.8rem', fontWeight: 600 }}>
          {sortBadge.text}
        </div>
      ) : null}
      {cols.length ? (
        <svg width={width} height={height} viewBox={`0 0 ${width} ${height}`} style={{ minWidth: Math.min(width, 520) }} role="img" aria-label={res ? `B+tree of orders_idx: ${res.entriesRead} of ${ROWS} leaf entries read over ${res.searches} index search${res.searches === 1 ? '' : 'es'}; sort: ${res.sort}` : 'B+tree of orders_idx'}>
          <defs>
            <marker id="cilp-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
              <path d="M0,0 L10,5 L0,10 z" fill="var(--viz-ink-2)" />
            </marker>
          </defs>
          {/* root page */}
          <rect x={1} y={1} width={width - 2} height={rootH} rx={5} fill="var(--viz-surface)" stroke="var(--viz-axis)" />
          <text x={10} y={18} fontSize={10.5} fill="var(--viz-ink-2)">
            <tspan fontWeight={700} fill="var(--viz-ink)">root</tspan>
            {'   separators: '}
            {Array.from({ length: nInner - 1 }, (_, j) => keyLabel(idx.leaves[(j + 1) * perInner][0])).join('  │  ')}
          </text>
          {Array.from({ length: nInner }, (_, j) => {
            const kids = Math.min(perInner, nLeaves - j * perInner);
            const hot = searchLeaves.some((li) => Math.floor(li / perInner) === j);
            const y0 = rowTop(j);
            const busY = y0 + 6;
            return (
              <g key={`in${j}`}>
                <polyline points={`6,${rootH + 1} 6,${y0 + 20} 12,${y0 + 20}`} fill="none" stroke={hot ? 'var(--viz-ink-2)' : 'var(--viz-grid)'} strokeWidth={hot ? 1.6 : 1} />
                <rect x={12} y={y0} width={innerW - 11} height={leafH + 14} rx={5} fill="var(--viz-surface)" stroke="var(--viz-axis)" />
                <text x={18} y={y0 + 15} fontSize={10} fontWeight={700} fill="var(--viz-ink)">internal {j + 1}</text>
                {Array.from({ length: kids - 1 }, (_, k) => (
                  <text key={k} x={18} y={y0 + 33 + k * 16} fontSize={9.5} fill="var(--viz-ink-2)">
                    │ {keyLabel(idx.leaves[j * perInner + k + 1][0])}
                  </text>
                ))}
                <line x1={innerW + 1} y1={busY} x2={leafX(j * perInner + kids - 1) + leafW / 2} y2={busY} stroke="var(--viz-grid)" />
                {Array.from({ length: kids }, (_, k) => {
                  const li = j * perInner + k;
                  return <line key={li} x1={leafX(li) + leafW / 2} y1={busY} x2={leafX(li) + leafW / 2} y2={leafY(li)} stroke="var(--viz-grid)" />;
                })}
              </g>
            );
          })}
          {/* index searches: root -> internal page -> leaf */}
          {searchLeaves.map((li, s) => {
            const j = Math.floor(li / perInner);
            const busY = rowTop(j) + 6 + ((s % 3) - 1) * 2;
            const x = leafX(li) + leafW / 2 + ((s % 3) - 1) * 4;
            return (
              <g key={`d${s}`}>
                <polyline points={`${innerW + 1},${busY} ${x},${busY} ${x},${leafY(li) - 1}`} fill="none" stroke="var(--viz-ink-2)" strokeWidth={1.6} markerEnd="url(#cilp-arrow)" />
              </g>
            );
          })}
          {/* leaves */}
          {idx.leaves.map((leaf, li) => (
            <g key={`leaf${li}`}>
              <rect x={leafX(li)} y={leafY(li)} width={leafW} height={leafH} rx={4} fill="var(--viz-surface)" stroke="var(--viz-axis)" />
              {leaf.map((r, k) => {
                const p = li * LEAF_CAP + k;
                const st = res ? res.state[p] : 'unread';
                const y = leafY(li) + 4 + k * rowH;
                const read = st !== 'unread';
                const rank = res?.outputRank[p];
                return (
                  <g key={r.id}>
                    {read ? <rect x={leafX(li) + 2.5} y={y + 0.5} width={leafW - 5} height={rowH - 1} rx={2} fill="none" stroke="var(--viz-ink-2)" strokeWidth={0.6} strokeDasharray="2 2" /> : null}
                    {read ? <rect x={leafX(li) + 4} y={y + 2} width={5} height={rowH - 4} rx={1} fill={STATE_COLOR[st]} /> : null}
                    <text x={leafX(li) + 13} y={y + 10} fontSize={10} fill={read ? 'var(--viz-ink)' : 'var(--viz-ink-muted)'} textDecoration={st === 'rejected-index' || st === 'rejected-heap' ? 'line-through' : undefined}>
                      {keyLabel(r)}
                    </text>
                    {rank ? <text x={leafX(li) + leafW - 5} y={y + 10} fontSize={9} textAnchor="end" fill="var(--viz-ink)" fontWeight={700}>#{rank}</text> : null}
                  </g>
                );
              })}
              <text x={leafX(li) + leafW / 2} y={leafY(li) + leafH + 11} fontSize={9} textAnchor="middle" fill="var(--viz-ink-muted)">leaf {li + 1}</text>
            </g>
          ))}
        </svg>
      ) : null}
      {res ? (
        <div style={{ marginTop: '0.4rem', fontSize: '0.75rem', color: 'var(--viz-ink-2)', overflowX: 'auto' }}>
          <span>{engine === 'mysql' ? 'EXPLAIN fields to look for if this index is used' : 'Plan shape if this index is used (EXPLAIN ANALYZE, abridged)'}:</span>
          <pre style={{ margin: '0.2rem 0 0', padding: '0.4rem 0.6rem', background: 'var(--viz-plane)', color: 'var(--viz-ink)', border: '1px solid var(--viz-border)', borderRadius: 6, fontSize: '0.75rem', whiteSpace: 'pre' }} data-testid="plan">
            {res.planLines.join('\n')}
          </pre>
        </div>
      ) : null}
    </VizPanel>
  );
}

function NoteText({ res, engine, limit, cols }: { res: ScanResult; engine: string; limit: number; cols: IndexCol[] }) {
  const list = (ps: Pred[]) => ps.map((p) => p.text).join(' AND ');
  const seek = res.fullIndexScan && !res.skipCols.length
    ? res.seekPreds.length + res.indexFilterPreds.length + res.heapPreds.length === 0
      ? `No WHERE clause: ${engine} reads the index from one end.`
      : `No condition on the leading column ${cols[0].col}, so ${engine} cannot seek: it reads the index from one end.`
    : res.skipCols.length
      ? `Skip scan: ${res.skipCols.map((s) => `${s.ranged ? 'a range skip array' : 'a skip array'} on ${s.col} (${s.distinct} value${s.distinct === 1 ? '' : 's'})`).join(' and ')} supplies an equality for each group, so ${list(res.seekPreds)} can bound the scan.`
      : `Seek on ${list(res.seekPreds)}.`;
  const filt = res.indexFilterPreds.length
    ? ` ${list(res.indexFilterPreds)} cannot narrow the range and is checked entry by entry${engine.startsWith('MySQL') ? ' (Index Condition Pushdown)' : ' inside the index'}.`
    : '';
  const heap = res.heapPreds.length ? ` ${list(res.heapPreds)} ${res.fullIndexScan && engine.startsWith('MySQL') ? 'is' : 'is not in the index, so it is'} checked only after each heap fetch.` : '';
  const outcome = ` ${res.entriesRead} leaf entr${res.entriesRead === 1 ? 'y' : 'ies'} read in ${res.searches} index search${res.searches === 1 ? '' : 'es'}, ${res.heapFetched} heap row${res.heapFetched === 1 ? '' : 's'} fetched, ${res.returned} returned.`;
  const sort =
    res.sort === 'none'
      ? limit && res.queryKeys.length ? ' Rows arrive in ORDER BY order, so the scan stops at the LIMIT.' : ''
      : res.sort === 'incremental'
        ? ` Only the first ${res.presorted === 1 ? 'ORDER BY key matches' : `${res.presorted} ORDER BY keys match`} the index order, so rows are sorted group by group${limit ? ' and the scan stops after the group that completes the LIMIT' : ''}.`
        : ` The index order does not match the ORDER BY${limit ? ', so every qualifying row is read and sorted before the LIMIT applies' : ', so every row is sorted'}.`;
  const forced = res.fullIndexScan && !res.skipCols.length && res.rejectedInIndex === 0 && (res.sort === 'full' || (!res.queryKeys.length && !limit))
    ? ` The index adds nothing here: no seek, no in-index filtering, no usable order. A real planner would ${engine.startsWith('MySQL') ? 'scan the table (type: ALL)' : 'use a sequential scan'} instead.`
    : '';
  const bad = res.skipCols.length && res.entriesRead >= ROWS * 0.6 ? ' The groups are so small that the scan reads most of the index anyway: skipping over a high-cardinality column is a full index scan in disguise, and the planner would prefer a sequential scan.' : '';
  return (
    <>
      <strong>{seek}</strong>
      {filt}
      {heap}
      {outcome}
      {sort}
      {forced}
      {bad}
    </>
  );
}
