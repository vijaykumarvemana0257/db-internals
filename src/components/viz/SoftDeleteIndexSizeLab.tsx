import { useMemo, useState } from 'react';
import { VizPanel, Segmented, Slider, Legend, Stats, Note, makeRng, fmtBytes, fmtNum, useSize } from './Viz';

/**
 * Full versus partial B-tree on a soft-deleted orders table, in PostgreSQL page arithmetic.
 *
 * Both indexes are on (customer_id bigint, created_at timestamptz); the partial one adds WHERE deleted_at IS NULL.
 * The query is: WHERE customer_id = $1 AND deleted_at IS NULL ORDER BY created_at DESC LIMIT 20.
 *
 * Size: an index tuple is an 8-byte IndexTupleData header plus two 8-byte keys (24 bytes, already MAXALIGNed) and a
 * 4-byte line pointer: 28 bytes. CREATE INDEX fills leaves to fillfactor 90 and internal pages to 70
 * (BTREE_DEFAULT_FILLFACTOR / BTREE_NONLEAF_FILLFACTOR), leaving (8192 - 24 page header - 16 btree opaque - reserve)
 * for tuples: 261 per leaf, 203 per internal page, plus one metapage. On PostgreSQL 17 a 1,000,000-row build of this
 * exact index came to 3,853 pages (3,832 leaves, 20 internal, meta) and the 5%-live partial index to 193 pages —
 * the same counts this arithmetic gives.
 *
 * Lookup: the full index returns the customer's entries newest first, and every one needs a heap fetch to test
 * deleted_at, until 20 live rows are found (Rows Removed by Filter counts the rest). The partial index only holds
 * live rows, so it reads 20 entries and fetches 20 rows. Pages = internal levels + leaves read + heap pages fetched.
 *
 * Assumptions: keys are unique (so nbtree deduplication does not apply); the table has been vacuumed, so the partial
 * index holds no entries for rows deleted since; each customer has 2,000 orders scattered through the heap, so each
 * fetched row is a different heap page; whether a row is deleted is independent of its age; the metapage is cached.
 * Page counts ignore the extra page or two a real scan reads at leaf boundaries.
 */

export const PAGE_BYTES = 8192;
export const ENTRY_BYTES = 28;
const USABLE = PAGE_BYTES - 24 - 16;
export const LEAF_ENTRIES = Math.floor((USABLE - Math.floor((PAGE_BYTES * 10) / 100)) / ENTRY_BYTES);
export const INTERNAL_ENTRIES = Math.floor((USABLE - Math.floor((PAGE_BYTES * 30) / 100)) / ENTRY_BYTES);
export const ROWS_PER_CUSTOMER = 2000;
export const LIMIT = 20;
export const TABLE_ROWS = [100_000, 1_000_000, 10_000_000, 100_000_000, 1_000_000_000] as const;
export const LIVE_STEPS = [0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 30, 50, 70, 90, 100] as const;

export type Shape = { entries: number; leaves: number; internal: number; levels: number; pages: number; bytes: number };

export function btreeShape(entries: number): Shape {
  const leaves = Math.max(1, Math.ceil(entries / LEAF_ENTRIES));
  let levels = 1;
  let internal = 0;
  let n = leaves;
  while (n > 1) {
    n = Math.ceil(n / INTERNAL_ENTRIES);
    internal += n;
    levels++;
  }
  const pages = leaves + internal + 1;
  return { entries, leaves, internal, levels, pages, bytes: pages * PAGE_BYTES };
}

/** One fixed uniform per order row, newest first; a row is live when its draw is below the live fraction. */
const DRAWS = (() => {
  const rng = makeRng(20260914);
  return Array.from({ length: ROWS_PER_CUSTOMER }, () => rng());
})();

export type Lookup = { entriesRead: number; heapPages: number; leafPages: number; internalPages: number; pages: number; returned: number; removed: number };

export function customerLiveFlags(livePct: number) {
  return DRAWS.map((u) => u < livePct / 100);
}

export function simulate(tableRows: number, livePct: number) {
  const f = livePct / 100;
  const full = btreeShape(tableRows);
  const partial = btreeShape(Math.round(tableRows * f));
  const live = customerLiveFlags(livePct);

  let read = 0;
  let found = 0;
  while (read < live.length && found < LIMIT) {
    if (live[read]) found++;
    read++;
  }
  const leafPages = (n: number) => Math.max(1, Math.ceil(n / LEAF_ENTRIES));
  const fullLookup: Lookup = {
    entriesRead: read,
    heapPages: read,
    leafPages: leafPages(read),
    internalPages: full.levels - 1,
    pages: full.levels - 1 + leafPages(read) + read,
    returned: found,
    removed: read - found,
  };
  const liveCount = live.filter(Boolean).length;
  const pRead = Math.min(LIMIT, liveCount);
  const partialLookup: Lookup = {
    entriesRead: pRead,
    heapPages: pRead,
    leafPages: leafPages(pRead),
    internalPages: partial.levels - 1,
    pages: partial.levels - 1 + leafPages(pRead) + pRead,
    returned: pRead,
    removed: 0,
  };
  return { full, partial, fullLookup, partialLookup, live, liveCount };
}

const plural = (n: number, one: string, many: string) => `${fmtNum(n)} ${n === 1 ? one : many}`;

const ROWS = 7;
const CELL = 11;
const GAP = 2;

export default function SoftDeleteIndexSizeLab() {
  const [rowsIdx, setRowsIdx] = useState(1);
  const [liveIdx, setLiveIdx] = useState(5);
  const [sizeRef, width] = useSize(680);
  const tableRows = TABLE_ROWS[rowsIdx];
  const livePct = LIVE_STEPS[liveIdx];
  const s = useMemo(() => simulate(tableRows, livePct), [tableRows, livePct]);

  // Lay out to the container width; on a phone, labels move above the marks instead of beside them.
  const W = Math.max(280, Math.min(680, Math.floor(width)));
  const narrow = W < 540;
  const LEFT = narrow ? 8 : 112;
  const cols = Math.max(12, Math.min(42, Math.floor((W - LEFT - 8 + GAP) / (CELL + GAP))));
  const cells = cols * ROWS;
  const barMax = narrow ? W - 16 : W - LEFT - 215;
  const sizeBar = (b: number) => Math.max(2, (b / s.full.bytes) * barMax);
  const pageMax = Math.max(s.fullLookup.pages, s.partialLookup.pages, 1);
  const pageBar = (p: number) => (p / pageMax) * barMax;

  const rowH = narrow ? 58 : 36;
  const barH = narrow ? 18 : 24;
  const gridH = ROWS * (CELL + GAP);
  const ySize = 28;
  const yGridTitle = ySize + 2 * rowH + 22;
  const yGrid1 = yGridTitle + 14 + (narrow ? 14 : 0);
  const yGrid2 = yGrid1 + gridH + 40 + (narrow ? 14 : 0);
  const yPagesTitle = yGrid2 + gridH + 44;
  const yPages = yPagesTitle + 12;
  const H = yPages + 2 * rowH + 8;

  const grid = (y: number, kind: 'full' | 'partial') => {
    const readN = kind === 'full' ? s.fullLookup.entriesRead : s.partialLookup.entriesRead;
    const total = kind === 'full' ? ROWS_PER_CUSTOMER : s.liveCount;
    const out = [];
    for (let c = 0; c < Math.min(cells, total); c++) {
      const isLive = kind === 'partial' ? true : s.live[c];
      const x = LEFT + (c % cols) * (CELL + GAP);
      const yy = y + Math.floor(c / cols) * (CELL + GAP);
      const read = c < readN;
      out.push(
        <rect
          key={c}
          x={x}
          y={yy}
          width={CELL}
          height={CELL}
          rx={2}
          fill={isLive ? 'var(--viz-3)' : 'var(--viz-axis)'}
          stroke={read ? 'var(--viz-2)' : 'none'}
          strokeWidth={1.6}
        />,
      );
    }
    const beyond = readN - Math.min(readN, cells);
    return (
      <g>
        {out}
        {total === 0 ? (
          <text x={LEFT} y={y + 12} fontSize={11} fill="var(--viz-ink-2)">
            this customer has no live orders
          </text>
        ) : null}
        {beyond > 0 ? (
          <text x={LEFT} y={y + gridH + 12} fontSize={10} fill="var(--viz-ink-2)">
            … and {plural(beyond, 'more entry', 'more entries')} read past the {fmtNum(cells)} shown
          </text>
        ) : null}
      </g>
    );
  };

  const pct = (v: number) => `${v < 1 ? v.toFixed(1) : fmtNum(v)}%`;
  const fl = s.fullLookup;
  const pl = s.partialLookup;

  return (
    <VizPanel
      title="Full versus partial index on a soft-deleted table"
      subtitle="orders(customer_id, created_at) with or without WHERE deleted_at IS NULL, serving: WHERE customer_id = $1 AND deleted_at IS NULL ORDER BY created_at DESC LIMIT 20. Change how much of the table is still live."
      controls={
        <>
          <Segmented
            label="Rows in orders"
            value={String(rowsIdx)}
            onChange={(v) => setRowsIdx(Number(v))}
            options={TABLE_ROWS.map((r, i) => ({ value: String(i), label: r >= 1e9 ? '1B' : r >= 1e6 ? `${r / 1e6}M` : `${r / 1e3}K` }))}
          />
          <Slider label="Live rows (deleted_at IS NULL)" min={0} max={LIVE_STEPS.length - 1} value={liveIdx} onChange={setLiveIdx} format={(i) => pct(LIVE_STEPS[i])} />
        </>
      }
      legend={
        <Legend
          items={[
            { label: 'Full index', color: 'var(--viz-1)' },
            { label: 'Partial index', color: 'var(--viz-7)' },
            { label: 'Entry for a live order', color: 'var(--viz-3)' },
            { label: 'Entry for a soft-deleted order', color: 'var(--viz-axis)' },
            { label: 'Entries this lookup reads', color: 'var(--viz-2)', shape: 'line' },
          ]}
        />
      }
      stats={
        <Stats
          items={[
            { label: 'Full index', value: `${fmtBytes(s.full.bytes)} · ${s.full.levels} lvl`, hint: `${fmtNum(s.full.pages)} pages` },
            { label: 'Partial index', value: `${fmtBytes(s.partial.bytes)} · ${s.partial.levels} lvl`, hint: `${fmtNum(s.partial.pages)} pages` },
            { label: 'Pages per lookup', value: `${fmtNum(fl.pages)} → ${fmtNum(pl.pages)}`, hint: 'Internal pages + leaf pages + heap pages, full index then partial index (estimate).' },
            { label: 'Rows Removed by Filter', value: fmtNum(fl.removed), hint: 'Full index only: entries whose heap row turned out to be soft-deleted.' },
            { label: 'Rows returned', value: fmtNum(pl.returned) },
          ]}
        />
      }
      note={
        <Note>
          At {pct(livePct)} live, the partial index holds {fmtNum(s.partial.entries)} of {fmtNum(tableRows)} entries: {s.partial.pages === s.full.pages ? `${fmtNum(s.full.pages)} pages, the same as the full index` : `${fmtNum(s.partial.pages)} pages instead of ${fmtNum(s.full.pages)}`}
          {s.full.levels > s.partial.levels ? `, ${s.full.levels - s.partial.levels} level${s.full.levels - s.partial.levels === 1 ? '' : 's'} shorter` : s.partial.pages === s.full.pages ? '' : ', the same height'}. For one customer, the full index
          reads {plural(fl.entriesRead, 'entry', 'entries')} newest-first and fetches {plural(fl.heapPages, 'heap row', 'heap rows')} to test deleted_at, throwing {fmtNum(fl.removed)} away
          {fl.returned < LIMIT ? ` and still finding only ${plural(fl.returned, 'live order', 'live orders')}` : ''}; the partial index reads {plural(pl.entriesRead, 'entry', 'entries')}, {pl.entriesRead === 1 ? 'which is wanted' : 'all of them wanted'}, and needs no Filter.
        </Note>
      }
      table={
        <table className="viz-table">
          <thead>
            <tr>
              <th>Index</th>
              <th>Entries</th>
              <th>Leaf pages</th>
              <th>Internal pages</th>
              <th>Levels</th>
              <th>Size</th>
              <th>Entries read</th>
              <th>Heap pages</th>
              <th>Pages per lookup</th>
            </tr>
          </thead>
          <tbody>
            {[
              { name: 'Full (customer_id, created_at)', sh: s.full, lk: fl },
              { name: 'Partial … WHERE deleted_at IS NULL', sh: s.partial, lk: pl },
            ].map((r) => (
              <tr key={r.name}>
                <td>{r.name}</td>
                <td>{fmtNum(r.sh.entries)}</td>
                <td>{fmtNum(r.sh.leaves)}</td>
                <td>{fmtNum(r.sh.internal)}</td>
                <td>{r.sh.levels}</td>
                <td>{fmtBytes(r.sh.bytes)}</td>
                <td>{fmtNum(r.lk.entriesRead)}</td>
                <td>{fmtNum(r.lk.heapPages)}</td>
                <td>{fmtNum(r.lk.pages)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      }
    >
      <div ref={sizeRef}>
        <svg viewBox={`0 0 ${W} ${H}`} width={W} height={H} role="img" aria-label={`Full index ${fmtBytes(s.full.bytes)}, partial index ${fmtBytes(s.partial.bytes)}; one lookup touches ${fl.pages} pages with the full index and ${pl.pages} with the partial index.`}>
          <text x={8} y={16} fontSize={12} fill="var(--viz-ink)">
            Index size
          </text>
          {[
            { label: 'Full index', sh: s.full, color: 'var(--viz-1)', y: ySize },
            { label: 'Partial index', sh: s.partial, color: 'var(--viz-7)', y: ySize + rowH },
          ].map((r) => {
            const by = narrow ? r.y + 16 : r.y;
            const text = `${fmtBytes(r.sh.bytes)} · ${fmtNum(r.sh.pages)} pages · ${r.sh.levels} level${r.sh.levels === 1 ? '' : 's'}`;
            return (
              <g key={r.label}>
                <text x={8} y={narrow ? r.y + 11 : r.y + 17} fontSize={11} fill="var(--viz-ink-2)">
                  {r.label}
                </text>
                <rect x={LEFT} y={by} width={sizeBar(r.sh.bytes)} height={barH} rx={3} fill={r.color} />
                <text x={narrow ? LEFT : LEFT + sizeBar(r.sh.bytes) + 8} y={narrow ? by + barH + 14 : by + 16} fontSize={11} fill="var(--viz-ink)">
                  {text}
                </text>
              </g>
            );
          })}

          <text x={8} y={yGridTitle} fontSize={12} fill="var(--viz-ink)">
            {narrow ? "One customer's newest entries, newest first" : "One customer's newest index entries (one square each, newest first)"}
          </text>
          {[
            { label: 'Full index', count: ROWS_PER_CUSTOMER, y: yGrid1, kind: 'full' as const },
            { label: 'Partial index', count: s.liveCount, y: yGrid2, kind: 'partial' as const },
          ].map((g) => (
            <g key={g.label}>
              {narrow ? (
                <text x={8} y={g.y - 5} fontSize={11} fill="var(--viz-ink-2)">
                  {g.label} · {plural(g.count, 'entry', 'entries')}
                </text>
              ) : (
                <>
                  <text x={8} y={g.y + 12} fontSize={11} fill="var(--viz-ink-2)">
                    {g.label}
                  </text>
                  <text x={8} y={g.y + 27} fontSize={10} fill="var(--viz-ink-muted)">
                    {plural(g.count, 'entry', 'entries')}
                  </text>
                </>
              )}
              {grid(g.y, g.kind)}
            </g>
          ))}

          <text x={8} y={yPagesTitle} fontSize={12} fill="var(--viz-ink)">
            Pages this lookup touches
          </text>
          {[
            { label: 'Full index', lk: fl, color: 'var(--viz-1)', y: yPages },
            { label: 'Partial index', lk: pl, color: 'var(--viz-7)', y: yPages + rowH },
          ].map((r) => {
            const idxPages = r.lk.internalPages + r.lk.leafPages;
            const wIdx = Math.max(2, pageBar(idxPages));
            const wHeap = pageBar(r.lk.heapPages);
            const by = narrow ? r.y + 16 : r.y;
            return (
              <g key={r.label}>
                <text x={8} y={narrow ? r.y + 11 : r.y + 17} fontSize={11} fill="var(--viz-ink-2)">
                  {r.label}
                </text>
                <rect x={LEFT} y={by} width={wIdx} height={barH} rx={2} fill={r.color} />
                {wHeap > 0 ? <rect x={LEFT + wIdx + 1} y={by} width={Math.max(1, Math.min(wHeap, barMax - wIdx - 1))} height={barH} rx={2} fill={r.color} fillOpacity={0.4} stroke={r.color} /> : null}
                <text x={narrow ? LEFT : LEFT + wIdx + wHeap + 8} y={narrow ? by + barH + 14 : by + 16} fontSize={11} fill="var(--viz-ink)">
                  {fmtNum(idxPages)} index + {fmtNum(r.lk.heapPages)} heap = {fmtNum(r.lk.pages)}
                </text>
              </g>
            );
          })}
        </svg>
      </div>
    </VizPanel>
  );
}
