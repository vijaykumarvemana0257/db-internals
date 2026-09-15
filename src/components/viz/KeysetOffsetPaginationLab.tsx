import { useEffect, useMemo, useRef, useState } from 'react';
import { VizPanel, Segmented, Choice, Check, Slider, Button, Legend, Stats, Note, makeRng, fmtNum, fmtTime, useTicker, useSize } from './Viz';

/**
 * OFFSET versus keyset pagination.
 *
 * Panel "Cost per request" models PostgreSQL 17 serving `SELECT * FROM events ORDER BY created_at, id LIMIT m`
 * from a B-tree on (created_at, id) over a 1,000,000-row table. The shape is taken from a real table measured with
 * pgstatindex and EXPLAIN (ANALYZE, BUFFERS): 2 internal levels, 261 entries per leaf page (leaf fill 90%), 30 rows per
 * 8 KB heap page, 4 rows per created_at value. A request that walks index positions [lo, hi) touches
 *   2 internal pages + the leaf pages spanning [lo, hi) + heap pages
 * where heap pages are 0 for an index-only scan (plus one visibility-map page), the pages spanning [lo, hi) when the
 * heap is stored in index order (correlation 1), and one page per row when it is stored in random order.
 * OFFSET walks [0, start + m) and discards `start` rows in the Limit node (nodeLimit.c). Keyset forms:
 *   row  — (created_at, id) > ($1, $2): an index condition, the walk is [start, start + m).
 *   or   — created_at > $1 OR (created_at = $1 AND id > $2): a Filter over a scan from the first entry. Once at most
 *          OR_BITMAP_MAX_ROWS rows lie past the cursor the planner switched plans: BitmapOr of the two index conditions
 *          (two descents), a Bitmap Heap Scan of every matching heap page (even when the index covers the SELECT list),
 *          then a Sort. Measured on the last page: 7, 8 and 10 buffers for 10, 20 and 100 rows; 26 for 20 in random order.
 *   or-bound — created_at >= $1 AND (…): index condition on created_at, filter removes the cursor's timestamp peers.
 * These formulas reproduce the measured buffer counts (OFFSET 10,000 → 375; 100,000 → 3,720; 500,000 → 18,586;
 * random heap OFFSET 10,000 → 10,061; keyset → 5, or 23 with a random heap). Cold latency assumes every buffer is a
 * 20 µs NVMe random read — an assumption, not a measurement; warm caches shrink the constant, not the slope.
 *
 * Panel "Paging under concurrent writes" replays a feed ordered by created_at DESC, id DESC, starting from the top.
 * Before each request after the first, the chosen writes happen: new rows are inserted at the head (sharing one new
 * timestamp), rows above OFFSET's position that a reader has already been shown are deleted, and one of the next 2m
 * rows at or past OFFSET's position that neither reader has been shown has its sort key bumped to now, per move (as
 * when sorting by updated_at). Victims are drawn with makeRng, so every replay is identical. Both readers see the same list. Initial rows share a timestamp
 * three at a time. "Skipped" counts pre-existing rows that are now behind the reader but were never returned.
 */

/* ------------------------------------------------------------ cost model */

export const TABLE = {
  rows: 1_000_000,
  entriesPerLeaf: 261,
  rowsPerHeapPage: 30,
  internalLevels: 2,
  rowsPerTimestamp: 4,
  leafPages: 3_832,
  heapPages: 33_334,
  coldReadNs: 20_000,
} as const;

export type RowAccess = 'index-only' | 'heap-ordered' | 'heap-random';
export type KeysetForm = 'row' | 'or' | 'or-bound';
export type Strategy = 'offset' | 'keyset';
export type Plan = 'ordered' | 'bitmap';

/**
 * Rows past the cursor at or below which the OR form is planned as BitmapOr + Sort. On the measured table PostgreSQL 17
 * chose that plan with 19 and with 100 rows left for every page size (10, 20, 100) and row access tried; the crossover
 * itself fell between 100 and 4,000 rows depending on both. Within the page slider's reach only the last page is this
 * close to the end (one slider step earlier, over 9,000 rows remain).
 */
export const OR_BITMAP_MAX_ROWS = 100;

const spanned = (lo: number, hi: number, per: number) => (hi <= lo ? 0 : Math.floor((hi - 1) / per) - Math.floor(lo / per) + 1);

export const maxPage = (m: number) => Math.floor(TABLE.rows / m);

export function rangeCost(lo: number, hi: number, access: RowAccess, plan: Plan = 'ordered') {
  const leaves = spanned(lo, hi, TABLE.entriesPerLeaf);
  if (plan === 'bitmap') {
    // Two Bitmap Index Scans, each its own descent, meeting in one leaf; the Bitmap Heap Scan reads every heap page
    // holding a match, whether or not the index covers the SELECT list.
    const heapPages = access === 'heap-random' ? hi - lo : spanned(lo, hi, TABLE.rowsPerHeapPage);
    return { leaves, heap: heapPages, vm: 0, buffers: 2 * TABLE.internalLevels + leaves + 1 + heapPages };
  }
  const heap = access === 'index-only' ? 0 : access === 'heap-ordered' ? spanned(lo, hi, TABLE.rowsPerHeapPage) : hi - lo;
  const vm = access === 'index-only' ? 1 : 0;
  return { leaves, heap, vm, buffers: TABLE.internalLevels + leaves + heap + vm };
}

export function requestCost(strategy: Strategy, page: number, m: number, access: RowAccess, form: KeysetForm) {
  const p = Math.min(Math.max(1, Math.round(page)), maxPage(m));
  const start = (p - 1) * m;
  let hi = start + m;
  let lo = start;
  let removed = 0;
  let plan: Plan = 'ordered';
  if (strategy === 'offset') {
    lo = 0;
    removed = start;
  } else if (start > 0 && form === 'or' && TABLE.rows - start <= OR_BITMAP_MAX_ROWS) {
    // Sorts every row past the cursor; the Limit keeps the first m.
    plan = 'bitmap';
    hi = TABLE.rows;
    removed = hi - start - m;
  } else if (start > 0 && form === 'or') {
    lo = 0;
    removed = start;
  } else if (start > 0 && form === 'or-bound') {
    // The cursor is the row at position start - 1, id = start; its timestamp group begins at id 4g, position 4g - 1.
    const g = Math.floor(start / TABLE.rowsPerTimestamp);
    const first = Math.max(0, TABLE.rowsPerTimestamp * g - 1);
    lo = first;
    removed = start - first;
  }
  const c = rangeCost(lo, hi, access, plan);
  return { page: p, start, end: start + m, lo, hi, removed, plan, entriesRead: hi - lo, ...c, coldNs: c.buffers * TABLE.coldReadNs };
}

/** Log-spaced sample of pages for the cost chart. */
export function costCurve(m: number, access: RowAccess, form: KeysetForm, points = 64) {
  const top = maxPage(m);
  const out: { page: number; offset: number; keyset: number }[] = [];
  let last = 0;
  for (let i = 0; i < points; i++) {
    const page = Math.max(1, Math.round(Math.pow(top, i / (points - 1))));
    if (page === last) continue;
    last = page;
    out.push({ page, offset: requestCost('offset', page, m, access, form).buffers, keyset: requestCost('keyset', page, m, access, form).buffers });
  }
  return out;
}

/* ------------------------------------------------------------ feed model */

export type FeedRow = { id: number; ts: number };
export type FeedParams = { m: number; inserts: number; deletes: number; moves: number; tiebreak: boolean; seed?: number };
export const TOP_ID = 1_000_000;
export const MAX_REQUESTS = 40;
const INITIAL_ROWS = 1_200;

/** a sorts before b in ORDER BY ts DESC, id DESC */
const precedes = (a: FeedRow, b: FeedRow) => a.ts > b.ts || (a.ts === b.ts && a.id > b.id);

type ServedPage = { rows: FeedRow[]; dupIds: number[] };
export type FeedRequest = {
  k: number;
  writes: { inserted: number[]; deleted: number[]; moved: number[] };
  offset: ServedPage & { offsetValue: number; discarded: number };
  keyset: ServedPage & { cursor: FeedRow | null };
};

export function runFeed(p: FeedParams, requests: number) {
  const rng = makeRng(p.seed ?? 7);
  const m = p.m;
  let list: FeedRow[] = Array.from({ length: INITIAL_ROWS }, (_, i) => {
    const id = TOP_ID - i;
    return { id, ts: Math.floor((id - 1) / 3) };
  });
  let nextId = TOP_ID + 1;
  let nowTs = list[0].ts + 1;
  const offRet = new Set<number>();
  const keyRet = new Set<number>();
  let cursor: FeedRow | null = null;
  const log: FeedRequest[] = [];
  const n = Math.min(Math.max(0, requests), MAX_REQUESTS);

  const after = (c: FeedRow | null, r: FeedRow) => (c === null ? true : p.tiebreak ? precedes(c, r) : r.ts < c.ts);
  const returned = (r: FeedRow) => offRet.has(r.id) || keyRet.has(r.id);

  for (let k = 1; k <= n; k++) {
    const writes = { inserted: [] as number[], deleted: [] as number[], moved: [] as number[] };
    if (k > 1) {
      const ts = nowTs++;
      const reader = (k - 1) * m;
      const head: FeedRow[] = [];
      for (let j = 0; j < p.moves; j++) {
        const pool: number[] = [];
        for (let i = reader; i < list.length && pool.length < 2 * m; i++) if (!returned(list[i])) pool.push(i);
        if (!pool.length) break;
        const idx = pool[Math.floor(rng() * pool.length)];
        const [row] = list.splice(idx, 1);
        head.push({ id: row.id, ts });
        writes.moved.push(row.id);
      }
      for (let j = 0; j < p.deletes; j++) {
        const pool: number[] = [];
        for (let i = 0; i < Math.min(list.length, reader - j); i++) if (returned(list[i])) pool.push(i);
        if (!pool.length) break;
        const idx = pool[Math.floor(rng() * pool.length)];
        const [row] = list.splice(idx, 1);
        writes.deleted.push(row.id);
      }
      for (let j = 0; j < p.inserts; j++) {
        head.push({ id: nextId, ts });
        writes.inserted.push(nextId);
        nextId++;
      }
      head.sort((a, b) => b.id - a.id);
      list = head.concat(list);
    }

    const offsetValue = (k - 1) * m;
    const offRows = list.slice(offsetValue, offsetValue + m);
    const offDup = offRows.filter((r) => offRet.has(r.id)).map((r) => r.id);
    offRows.forEach((r) => offRet.add(r.id));

    const firstAfter = list.findIndex((r) => after(cursor, r));
    const keyRows = firstAfter < 0 ? [] : list.slice(firstAfter, firstAfter + m);
    const keyDup = keyRows.filter((r) => keyRet.has(r.id)).map((r) => r.id);
    keyRows.forEach((r) => keyRet.add(r.id));
    const usedCursor = cursor;
    if (keyRows.length) cursor = { ...keyRows[keyRows.length - 1] };

    log.push({
      k,
      writes,
      offset: { rows: offRows, dupIds: offDup, offsetValue, discarded: offsetValue },
      keyset: { rows: keyRows, dupIds: keyDup, cursor: usedCursor },
    });
  }

  // Rows that existed before the session, still exist, are behind the reader, and were never returned.
  const nextOffset = n * m;
  const offBehind = list.slice(0, Math.min(nextOffset, list.length));
  const offsetSkipped = offBehind.filter((r) => r.id <= TOP_ID && !offRet.has(r.id));
  const keyBehind = n === 0 ? [] : list.filter((r) => !after(cursor, r));
  const keysetSkipped = keyBehind.filter((r) => r.id <= TOP_ID && !keyRet.has(r.id));
  const newAbove = keyBehind.filter((r) => r.id > TOP_ID && !keyRet.has(r.id)).length;
  const sum = (f: (x: FeedRequest) => number) => log.reduce((s, x) => s + f(x), 0);

  return {
    log,
    list,
    cursor,
    offsetSkipped,
    keysetSkipped,
    newAbove,
    offsetDups: sum((x) => x.offset.dupIds.length),
    keysetDups: sum((x) => x.keyset.dupIds.length),
    offsetEntriesRead: sum((x) => x.offset.offsetValue + x.offset.rows.length),
    keysetEntriesRead: sum((x) => x.keyset.rows.length),
  };
}

/* ------------------------------------------------------------------ view */

const MAX_W = 680;
const MIN_W = 300;
const LABEL_W = 84;

const ACCESS_LABEL: Record<RowAccess, string> = {
  'index-only': 'Index-only scan (all-visible)',
  'heap-ordered': 'Heap fetch, table stored in index order',
  'heap-random': 'Heap fetch, table stored in random order',
};

const FORM_SQL: Record<KeysetForm, string> = {
  row: 'WHERE (created_at, id) > ($1, $2)',
  or: 'WHERE created_at > $1 OR (created_at = $1 AND id > $2)',
  'or-bound': 'WHERE created_at >= $1 AND (created_at > $1 OR (created_at = $1 AND id > $2))',
};

/** Feed timestamps are whole seconds; show them as a time of day. */
export const clock = (ts: number) => {
  const t = ((ts % 86_400) + 86_400) % 86_400;
  const two = (n: number) => String(n).padStart(2, '0');
  return `${two(Math.floor(t / 3600))}:${two(Math.floor((t % 3600) / 60))}:${two(t % 60)}`;
};

/** Deterministic scatter for heap pages hit in random order. */
function heapTicks(lo: number, hi: number) {
  const count = Math.min(hi - lo, 360);
  const rng = makeRng((lo * 31 + hi) % 2147483647 || 1);
  return Array.from({ length: count }, () => rng());
}

function CostLane({
  label,
  sql,
  cost,
  color,
  progress,
  access,
  strategy,
  width,
}: {
  label: string;
  sql: string;
  cost: ReturnType<typeof requestCost>;
  color: string;
  progress: number;
  access: RowAccess;
  strategy: Strategy;
  width: number;
}) {
  const W = width;
  const stripW = W - LABEL_W - 6;
  const X = (pos: number) => LABEL_W + (pos / TABLE.rows) * stripW;
  const scanTo = cost.lo + (cost.hi - cost.lo) * progress;
  const discardedEnd = Math.min(scanTo, cost.lo + cost.removed);
  const windowShown = progress >= 1 || scanTo > cost.start;
  const winX = X(cost.start);
  const winW = Math.max(3, X(cost.end) - winX);
  // A bitmap heap scan reads the heap even when the index covers the SELECT list.
  const heapAccess: RowAccess = cost.plan === 'bitmap' && access === 'index-only' ? 'heap-ordered' : access;
  const ticks = heapAccess === 'heap-random' ? heapTicks(cost.lo, cost.hi) : [];
  const heapFrac = (scanTo - cost.lo) / Math.max(1, cost.hi - cost.lo);
  const seekNote =
    cost.plan === 'bitmap'
      ? 'two bitmap index scans from the cursor, then a sort'
      : strategy === 'offset' || cost.lo === 0
      ? cost.start === 0
        ? 'descend to the first leaf'
        : 'descend to the first leaf, then follow right-links'
      : 'descend straight to the cursor';
  const noteW = seekNote.length * 5.4;
  const noteEnd = X(cost.lo) + 8 + noteW > W - 2;
  const noteX = noteEnd ? Math.max(noteW + 2, X(cost.lo) + 4) : Math.max(LABEL_W, X(cost.lo) - 4);
  const vmNote = W < 460 ? 'not visited (all-visible)' : 'not visited: visibility map says all-visible (1 VM page)';
  const H = 84;
  return (
    <div style={{ margin: '0.5rem 0 0' }}>
      <div style={{ margin: 0, display: 'flex', flexWrap: 'wrap', alignItems: 'baseline', gap: '0.15rem 0.5rem', fontSize: '0.78rem', color: 'var(--viz-ink)' }}>
        <span aria-hidden="true" style={{ width: 10, height: 10, borderRadius: 2, background: color, alignSelf: 'center' }} />
        <strong>{label}</strong>
        <code style={{ fontSize: '0.7rem', color: 'var(--viz-ink-2)', whiteSpace: 'normal', overflowWrap: 'anywhere' }}>{sql}</code>
      </div>
      <svg viewBox={`0 0 ${W} ${H}`} width={W} height={H} role="img" aria-label={`${label}: reads index positions ${fmtNum(cost.lo)} to ${fmtNum(cost.hi)}, discards ${fmtNum(cost.removed)}, ${fmtNum(cost.buffers)} buffers`} style={{ margin: '4px 0 0' }}>
        <text x={0} y={27} fontSize={11} fill="var(--viz-ink)">
          index leaves
        </text>
        <rect x={LABEL_W} y={14} width={stripW} height={18} rx={3} fill="var(--viz-plane)" stroke="var(--viz-border)" />
        {cost.removed > 0 ? (
          <rect x={X(cost.lo)} y={14} width={Math.max(discardedEnd > cost.lo ? 1.5 : 0, X(discardedEnd) - X(cost.lo))} height={18} fill="var(--viz-stale)" opacity={0.85} />
        ) : null}
        {windowShown ? <rect x={winX} y={11} width={winW} height={24} rx={2} fill={color} /> : null}
        <path d={`M ${X(cost.lo)} 8 l -4 -6 h 8 z`} fill="var(--viz-ink-muted)" />
        {progress < 1 ? <line x1={X(scanTo)} x2={X(scanTo)} y1={8} y2={38} stroke="var(--viz-ink)" strokeWidth={1.5} /> : null}
        <text x={noteX} y={47} fontSize={10} fill="var(--viz-ink-2)" textAnchor={noteEnd ? 'end' : 'start'}>
          {seekNote}
        </text>

        <text x={0} y={69} fontSize={11} fill="var(--viz-ink)">
          heap pages
        </text>
        <rect x={LABEL_W} y={56} width={stripW} height={16} rx={3} fill="var(--viz-plane)" stroke="var(--viz-border)" />
        {heapAccess === 'index-only' ? (
          <text x={LABEL_W + 8} y={68} fontSize={10} fill="var(--viz-ink-2)">
            {vmNote}
          </text>
        ) : heapAccess === 'heap-ordered' ? (
          <rect x={X(cost.lo)} y={58} width={Math.max(2, (X(cost.hi) - X(cost.lo)) * heapFrac)} height={12} fill="var(--viz-3)" />
        ) : (
          ticks.slice(0, Math.ceil(ticks.length * heapFrac)).map((t, i) => <rect key={i} x={LABEL_W + t * (stripW - 2)} y={58} width={1.5} height={12} fill="var(--viz-3)" />)
        )}
      </svg>
      <div style={{ margin: '2px 0 0', fontSize: '0.75rem', color: 'var(--viz-ink)', textAlign: 'right', fontVariantNumeric: 'tabular-nums' }}>
        {fmtNum(cost.entriesRead)} entries read · {fmtNum(cost.removed)} discarded · {fmtNum(cost.buffers)} buffers
      </div>
    </div>
  );
}

const compact = (n: number) => (n < 1_000 ? fmtNum(n) : n < 1_000_000 ? `${fmtNum(n / 1_000)}k` : `${fmtNum(n / 1_000_000)}M`);

function CostChart({ m, access, form, page, width }: { m: number; access: RowAccess; form: KeysetForm; page: number; width: number }) {
  const curve = useMemo(() => costCurve(m, access, form), [m, access, form]);
  const W = width;
  const H = 230;
  const L = 50;
  const R = W - 64;
  const T = 14;
  const B = H - 34;
  const top = maxPage(m);
  const peak = curve.reduce((mx, c) => Math.max(mx, c.offset, c.keyset), 1);
  const maxY = Math.max(4, Math.ceil(Math.log10(peak)));
  const X = (pg: number) => L + (Math.log10(pg) / Math.log10(top)) * (R - L);
  const Y = (b: number) => B - (Math.log10(Math.max(1, b)) / maxY) * (B - T);
  const path = (key: 'offset' | 'keyset') => curve.map((c, i) => `${i ? 'L' : 'M'} ${X(c.page).toFixed(1)} ${Y(c[key]).toFixed(1)}`).join(' ');
  const cur = { offset: requestCost('offset', page, m, access, form).buffers, keyset: requestCost('keyset', page, m, access, form).buffers };
  const xTicks = [1, 10, 100, 1_000, 10_000, 100_000].filter((t) => t <= top);
  return (
    <svg viewBox={`0 0 ${W} ${H}`} width={W} height={H} role="img" aria-label={`Buffers per request against page number: at page ${fmtNum(page)} OFFSET reads ${fmtNum(cur.offset)} buffers and keyset ${fmtNum(cur.keyset)}`} style={{ marginTop: 10 }}>
      {Array.from({ length: maxY + 1 }, (_, e) => (
        <g key={e}>
          <line x1={L} x2={R} y1={Y(10 ** e)} y2={Y(10 ** e)} className="viz-grid-line" />
          <text x={L - 6} y={Y(10 ** e) + 4} textAnchor="end" fontSize={10}>
            {compact(10 ** e)}
          </text>
          <text x={R + 5} y={Y(10 ** e) + 4} fontSize={10}>
            {fmtTime(10 ** e * TABLE.coldReadNs)}
          </text>
        </g>
      ))}
      {xTicks.map((t) => (
        <g key={t}>
          <line x1={X(t)} x2={X(t)} y1={T} y2={B} className="viz-grid-line" />
          <text x={X(t)} y={B + 14} textAnchor="middle" fontSize={10}>
            {compact(t)}
          </text>
        </g>
      ))}
      <line x1={L} x2={R} y1={B} y2={B} className="viz-axis-line" />
      <line x1={L} x2={L} y1={T} y2={B} className="viz-axis-line" />
      <text x={(L + R) / 2} y={H - 4} textAnchor="middle" fontSize={11} fill="var(--viz-ink)">
        page number (log scale)
      </text>
      <text x={10} y={(T + B) / 2} fontSize={11} fill="var(--viz-ink)" transform={`rotate(-90 10 ${(T + B) / 2})`} textAnchor="middle">
        buffers per request
      </text>
      <text x={W - 6} y={(T + B) / 2} fontSize={10} fill="var(--viz-ink-2)" transform={`rotate(90 ${W - 6} ${(T + B) / 2})`} textAnchor="middle">
        cold latency, 20 µs/page
      </text>
      <path d={path('offset')} fill="none" stroke="var(--viz-2)" strokeWidth={2.5} />
      <path d={path('keyset')} fill="none" stroke="var(--viz-1)" strokeWidth={2.5} strokeDasharray="7 4" />
      <line x1={X(page)} x2={X(page)} y1={T} y2={B} stroke="var(--viz-ink-muted)" strokeDasharray="3 3" />
      <circle cx={X(page)} cy={Y(cur.offset)} r={4.5} fill="var(--viz-2)" stroke="var(--viz-surface)" strokeWidth={1.5} />
      <rect x={X(page) - 4} y={Y(cur.keyset) - 4} width={8} height={8} fill="var(--viz-1)" stroke="var(--viz-surface)" strokeWidth={1.5} />
    </svg>
  );
}

function Chip({ row, dup, skipped }: { row: FeedRow; dup?: boolean; skipped?: boolean }) {
  return (
    <span
      style={{
        display: 'inline-flex',
        alignItems: 'center',
        gap: 3,
        padding: '1px 6px',
        borderRadius: 4,
        fontSize: '0.72rem',
        fontVariantNumeric: 'tabular-nums',
        color: 'var(--viz-ink)',
        background: 'var(--viz-surface)',
        border: dup ? '2px solid var(--viz-critical)' : skipped ? '2px dashed var(--viz-serious)' : '1px solid var(--viz-border)',
      }}
    >
      {row.id}
      {row.id > TOP_ID ? <small style={{ color: 'var(--viz-ink-2)' }}>new</small> : null}
      {dup ? <strong style={{ fontSize: '0.65rem' }}>dup</strong> : null}
    </span>
  );
}

function PageRow({ label, page }: { label: string; page: ServedPage | undefined }) {
  return (
    <div style={{ display: 'grid', gridTemplateColumns: '4.2rem 1fr', gap: 6, alignItems: 'start', marginTop: 6 }}>
      <span style={{ fontSize: '0.72rem', color: 'var(--viz-ink-2)', paddingTop: 2 }}>{label}</span>
      <div style={{ display: 'flex', flexWrap: 'wrap', gap: 4 }}>
        {page && page.rows.length ? page.rows.map((r) => <Chip key={r.id} row={r} dup={page.dupIds.includes(r.id)} />) : <span style={{ fontSize: '0.72rem', color: 'var(--viz-ink-muted)' }}>—</span>}
      </div>
    </div>
  );
}

function SkippedRow({ rows }: { rows: FeedRow[] }) {
  const shown = rows.slice(0, 12);
  return (
    <div style={{ display: 'grid', gridTemplateColumns: '4.2rem 1fr', gap: 6, alignItems: 'start', marginTop: 8 }}>
      <span style={{ fontSize: '0.72rem', color: 'var(--viz-ink-2)', paddingTop: 2 }}>skipped</span>
      <div style={{ display: 'flex', flexWrap: 'wrap', gap: 4, alignItems: 'center' }}>
        {shown.length ? shown.map((r) => <Chip key={r.id} row={r} skipped />) : <span style={{ fontSize: '0.72rem', color: 'var(--viz-ink-muted)' }}>none</span>}
        {rows.length > shown.length ? <span style={{ fontSize: '0.72rem', color: 'var(--viz-ink-2)' }}>+{rows.length - shown.length} more</span> : null}
      </div>
    </div>
  );
}

export default function KeysetOffsetPaginationLab() {
  const [panel, setPanel] = useState<'cost' | 'writes'>('cost');

  // cost panel
  const [m, setM] = useState(20);
  const [pageExp, setPageExp] = useState(750);
  const [access, setAccess] = useState<RowAccess>('heap-ordered');
  const [form, setForm] = useState<KeysetForm>('row');
  const top = maxPage(m);
  const page = Math.max(1, Math.round(Math.pow(top, pageExp / 1000)));
  const off = requestCost('offset', page, m, access, form);
  const key = requestCost('keyset', page, m, access, form);
  const [sizeRef, measured] = useSize(MAX_W);
  const figW = Math.round(Math.min(MAX_W, Math.max(MIN_W, measured)));

  // scan animation: starts at 1 so server and first client render agree
  const [progress, setProgress] = useState(1);
  const animKey = `${m}|${page}|${access}|${form}`;
  const firstKey = useRef(animKey);
  useEffect(() => {
    if (firstKey.current === animKey) return;
    firstKey.current = animKey;
    setProgress(0);
  }, [animKey]);
  useTicker((dt) => setProgress((p) => Math.min(1, p + dt / 900)), progress < 1);

  // writes panel
  const [fm, setFm] = useState(10);
  const [inserts, setInserts] = useState(3);
  const [deletes, setDeletes] = useState(0);
  const [moves, setMoves] = useState(0);
  const [tiebreak, setTiebreak] = useState(true);
  const [requests, setRequests] = useState(2);
  const feed = useMemo(() => runFeed({ m: fm, inserts, deletes, moves, tiebreak }, requests), [fm, inserts, deletes, moves, tiebreak, requests]);
  const cur = feed.log[feed.log.length - 1];
  const prev = feed.log[feed.log.length - 2];

  const costTable = (
    <table className="viz-table">
      <thead>
        <tr>
          <th>Page</th>
          <th>OFFSET</th>
          <th>OFFSET entries read</th>
          <th>OFFSET buffers</th>
          <th>Keyset entries read</th>
          <th>Keyset buffers</th>
          <th>Cold latency, OFFSET vs keyset</th>
        </tr>
      </thead>
      <tbody>
        {[1, 10, 100, 1_000, 5_000, 25_000, top]
          .filter((p, i, a) => p <= top && a.indexOf(p) === i)
          .map((p) => {
            const o = requestCost('offset', p, m, access, form);
            const k = requestCost('keyset', p, m, access, form);
            return (
              <tr key={p}>
                <td>{fmtNum(p)}</td>
                <td>{fmtNum(o.start)}</td>
                <td>{fmtNum(o.entriesRead)}</td>
                <td>{fmtNum(o.buffers)}</td>
                <td>{fmtNum(k.entriesRead)}</td>
                <td>{fmtNum(k.buffers)}</td>
                <td>
                  {fmtTime(o.coldNs)} vs {fmtTime(k.coldNs)}
                </td>
              </tr>
            );
          })}
      </tbody>
    </table>
  );

  const feedTable = (
    <table className="viz-table">
      <thead>
        <tr>
          <th>Request</th>
          <th>Writes before it</th>
          <th>OFFSET page</th>
          <th>OFFSET dups</th>
          <th>Keyset page</th>
          <th>Keyset dups</th>
        </tr>
      </thead>
      <tbody>
        {feed.log.map((x) => (
          <tr key={x.k}>
            <td>{x.k}</td>
            <td>
              +{x.writes.inserted.length} / −{x.writes.deleted.length} / moved {x.writes.moved.length}
            </td>
            <td>{x.offset.rows.map((r) => r.id).join(', ')}</td>
            <td>{x.offset.dupIds.length}</td>
            <td>{x.keyset.rows.map((r) => r.id).join(', ')}</td>
            <td>{x.keyset.dupIds.length}</td>
          </tr>
        ))}
      </tbody>
    </table>
  );

  const ratio = off.buffers / Math.max(1, key.buffers);

  return (
    <VizPanel
      title={panel === 'cost' ? 'What one page request costs, OFFSET versus keyset' : 'Paging a live feed while rows change'}
      subtitle={
        panel === 'cost'
          ? 'SELECT * FROM events ORDER BY created_at, id LIMIT m over a one-million-row PostgreSQL table with an index on (created_at, id). OFFSET walks every earlier entry and throws it away; keyset descends straight to the cursor.'
          : 'Both readers page a feed ordered by created_at DESC, id DESC from the top, one request at a time, while writes land between requests. OFFSET counts positions; keyset remembers the last key it returned.'
      }
      controls={
        <Segmented
          label="Panel"
          value={panel}
          onChange={setPanel}
          options={[
            { value: 'cost', label: 'Cost per request' },
            { value: 'writes', label: 'Paging under concurrent writes' },
          ]}
        />
      }
      legend={
        panel === 'cost' ? (
          <Legend
            items={[
              { label: 'OFFSET (page returned; solid line)', color: 'var(--viz-2)' },
              { label: 'Keyset (page returned; dashed line)', color: 'var(--viz-1)' },
              { label: 'Entries read, then discarded', color: 'var(--viz-stale)' },
              { label: 'Heap pages fetched', color: 'var(--viz-3)' },
            ]}
          />
        ) : (
          <Legend
            items={[
              { label: 'Row returned', color: 'var(--viz-border)' },
              { label: 'Duplicate: returned on an earlier page too', color: 'var(--viz-critical)' },
              { label: 'Skipped: existed all along, now behind the reader, never returned', color: 'var(--viz-serious)' },
            ]}
          />
        )
      }
      stats={
        panel === 'cost' ? (
          <Stats
            items={[
              { label: 'OFFSET buffers', value: fmtNum(off.buffers), hint: 'Internal pages + leaf pages + heap pages touched, as EXPLAIN (ANALYZE, BUFFERS) counts them.' },
              { label: 'Keyset buffers', value: fmtNum(key.buffers) },
              { label: 'Rows discarded', value: `${fmtNum(off.removed)} vs ${fmtNum(key.removed)}`, hint: 'OFFSET discards in the Limit node; a keyset Filter shows them as Rows Removed by Filter.' },
              { label: 'Work ratio', value: `${fmtNum(ratio, ratio < 10 ? 1 : 0)}×` },
              { label: 'Cold latency', value: `${fmtTime(off.coldNs)} vs ${fmtTime(key.coldNs)}`, hint: 'Assumes every buffer is an uncached NVMe random read. Warm caches shrink both numbers, not their ratio of growth.' },
            ]}
          />
        ) : (
          <Stats
            items={[
              { label: 'OFFSET duplicates', value: fmtNum(feed.offsetDups) },
              { label: 'OFFSET skipped', value: fmtNum(feed.offsetSkipped.length) },
              { label: 'Keyset duplicates', value: fmtNum(feed.keysetDups) },
              { label: 'Keyset skipped', value: fmtNum(feed.keysetSkipped.length) },
              { label: 'Entries read, session', value: `${fmtNum(feed.offsetEntriesRead)} vs ${fmtNum(feed.keysetEntriesRead)}`, hint: 'Summed over every request in the session.' },
            ]}
          />
        )
      }
      note={
        <Note>
          {panel === 'cost' ? (
            <>
              <strong>
                Page {fmtNum(page)}: OFFSET reads {fmtNum(off.entriesRead)} index entries to return {m}; keyset reads {fmtNum(key.entriesRead)}.
              </strong>{' '}
              {page === 1
                ? 'On the first page the two queries are the same query. Drag the page slider right and watch OFFSET’s discarded span grow while keyset’s cost stays flat.'
                : form === 'or' && key.plan === 'bitmap'
                  ? `Only ${fmtNum(key.entriesRead)} rows lie past the cursor, few enough that PostgreSQL switches plans: a BitmapOr of created_at > $1 and (created_at = $1 AND id > $2), a heap read even for an index-only query, then a sort. One slider step to the left, the walk from the first entry is back.`
                  : form === 'or'
                  ? 'This keyset form has no index condition: PostgreSQL walks the index from the first entry and discards every earlier row in a Filter, so it costs exactly what OFFSET costs. Add the leading created_at >= $1 bound to turn it back into a seek.'
                  : form === 'or-bound'
                    ? `The created_at >= $1 bound becomes the index condition; the OR only filters out ${fmtNum(key.removed)} row${key.removed === 1 ? '' : 's'}: the cursor’s timestamp group up to and including the cursor.`
                    : 'The row comparison is an index condition, so the scan starts at the cursor and stops after one page.'}{' '}
              {access === 'heap-random' && page > 1
                ? 'With the heap in random order every discarded row is its own heap page, so OFFSET’s buffers track rows one for one.'
                : access === 'index-only' && page > 1
                  ? 'An index-only scan never visits the heap, which shrinks the constant — but the discarded entries are still read.'
                  : ''}
            </>
          ) : requests >= 1 && cur ? (
            <>
              <strong>
                Request {cur.k}: OFFSET {fmtNum(cur.offset.offsetValue)} returned {cur.offset.dupIds.length} duplicate{cur.offset.dupIds.length === 1 ? '' : 's'}; keyset returned {cur.keyset.dupIds.length}.
              </strong>{' '}
              {cur.k === 1
                ? 'Page 1 is identical for both. Press Next page: the writes happen first, then both readers fetch page 2.'
                : `Before this request ${cur.writes.inserted.length} row${cur.writes.inserted.length === 1 ? ' was' : 's were'} inserted at the top, ${cur.writes.deleted.length} already-seen row${cur.writes.deleted.length === 1 ? ' was' : 's were'} deleted and ${cur.writes.moved.length} unseen row${cur.writes.moved.length === 1 ? ' was' : 's were'} bumped to the top. `}
              {cur.k > 1 && cur.writes.inserted.length > 0 ? 'Every insert above OFFSET’s position pushes rows it already returned back into the next window. ' : ''}
              {cur.k > 1 && cur.writes.deleted.length > 0 ? 'Every delete above OFFSET’s position pulls an unread row back behind it, where it is never returned. ' : ''}
              {cur.k > 1 && cur.writes.moved.length > 0 ? 'A row whose sort key moves behind the cursor is lost to both readers: keyset is exact only for keys that never change. ' : ''}
              {!tiebreak ? 'The keyset cursor compares created_at alone, so rows that share the last row’s timestamp but were not on the page fall behind it and are skipped. ' : ''}
              {feed.newAbove > 0 ? `${feed.newAbove} new row${feed.newAbove === 1 ? ' is' : 's are'} above the keyset cursor — correct: they appear when the user reloads from the top.` : ''}
            </>
          ) : null}
        </Note>
      }
      table={panel === 'cost' ? costTable : feedTable}
    >
      <div ref={sizeRef}>
      {panel === 'cost' ? (
        <>
          <div className="viz-controls">
            <Segmented
              label="Rows per page"
              value={String(m)}
              onChange={(v) => setM(Number(v))}
              options={[10, 20, 50, 100].map((v) => ({ value: String(v), label: String(v) }))}
            />
            <Slider label="Page" min={0} max={1000} value={pageExp} onChange={setPageExp} format={() => `${fmtNum(page)} of ${fmtNum(top)}`} />
            <Choice label="Row access" value={access} onChange={setAccess} options={(Object.keys(ACCESS_LABEL) as RowAccess[]).map((a) => ({ value: a, label: ACCESS_LABEL[a] }))} />
            <Choice
              label="Keyset predicate"
              value={form}
              onChange={setForm}
              options={[
                { value: 'row', label: '(created_at, id) > ($1, $2)' },
                { value: 'or', label: 'created_at > $1 OR (… AND id > $2)' },
                { value: 'or-bound', label: 'created_at >= $1 AND (… OR …)' },
              ]}
            />
          </div>
          <div>
            <CostLane label="OFFSET" sql={`ORDER BY created_at, id LIMIT ${m} OFFSET ${fmtNum(off.start)}`} cost={off} color="var(--viz-2)" progress={progress} access={access} strategy="offset" width={figW} />
            <CostLane label="Keyset" sql={page === 1 ? `ORDER BY created_at, id LIMIT ${m}` : `${FORM_SQL[form]} ORDER BY created_at, id LIMIT ${m}`} cost={key} color="var(--viz-1)" progress={progress} access={access} strategy="keyset" width={figW} />
            <CostChart m={m} access={access} form={form} page={page} width={figW} />
          </div>
        </>
      ) : (
        <>
          <div className="viz-controls">
            <Segmented label="Rows per page" value={String(fm)} onChange={(v) => setFm(Number(v))} options={[5, 10, 20].map((v) => ({ value: String(v), label: String(v) }))} />
            <Slider label="Inserts at the top, per request" min={0} max={10} value={inserts} onChange={setInserts} />
            <Slider label="Deletes of rows already seen" min={0} max={5} value={deletes} onChange={setDeletes} />
            <Slider label="Unseen rows bumped to the top" min={0} max={3} value={moves} onChange={setMoves} />
            <Check label="Cursor includes the id tiebreaker" checked={tiebreak} onChange={setTiebreak} />
          </div>
          <div className="viz-controls">
            <Button primary onClick={() => setRequests((r) => Math.min(MAX_REQUESTS, r + 1))} disabled={requests >= MAX_REQUESTS}>
              Next page
            </Button>
            <Button onClick={() => setRequests(1)}>Back to page 1</Button>
            <span style={{ fontSize: '0.75rem', color: 'var(--viz-ink-2)' }}>
              request {requests} of at most {MAX_REQUESTS}
            </span>
          </div>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(17rem, 1fr))', gap: '0.75rem 1.25rem' }}>
            {(['offset', 'keyset'] as const).map((s) => (
              <div key={s} style={{ margin: 0, border: '1px solid var(--viz-border)', borderRadius: 8, padding: '0.5rem 0.625rem', background: 'var(--viz-surface)' }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: 6, fontSize: '0.8rem', color: 'var(--viz-ink)', fontWeight: 600 }}>
                  <span aria-hidden="true" style={{ width: 10, height: 10, borderRadius: 2, background: s === 'offset' ? 'var(--viz-2)' : 'var(--viz-1)' }} />
                  {s === 'offset' ? 'OFFSET' : 'Keyset'}
                </div>
                <code style={{ display: 'block', fontSize: '0.7rem', color: 'var(--viz-ink-2)', marginTop: 2, whiteSpace: 'normal', overflowWrap: 'anywhere' }}>
                  {s === 'offset'
                    ? `ORDER BY created_at DESC, id DESC LIMIT ${fm} OFFSET ${cur ? cur.offset.offsetValue : 0}`
                    : cur && cur.keyset.cursor
                      ? `WHERE ${tiebreak ? `(created_at, id) < ('${clock(cur.keyset.cursor.ts)}', ${cur.keyset.cursor.id})` : `created_at < '${clock(cur.keyset.cursor.ts)}'`} ORDER BY created_at DESC, id DESC LIMIT ${fm}`
                      : `ORDER BY created_at DESC, id DESC LIMIT ${fm}`}
                </code>
                <PageRow label={prev ? `page ${prev.k}` : 'previous'} page={prev ? prev[s] : undefined} />
                <PageRow label={cur ? `page ${cur.k}` : 'this page'} page={cur ? cur[s] : undefined} />
                <SkippedRow rows={s === 'offset' ? feed.offsetSkipped : feed.keysetSkipped} />
              </div>
            ))}
          </div>
        </>
      )}
      </div>
    </VizPanel>
  );
}
