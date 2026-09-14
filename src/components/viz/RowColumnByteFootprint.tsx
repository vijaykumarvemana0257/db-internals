import { useMemo, useState } from 'react';
import {
  VizPanel,
  Slider,
  Segmented,
  Legend,
  Stats,
  TooltipHost,
  useTip,
  useSize,
  fmtBytes,
  fmtNum,
} from './Viz';

/**
 * The same 'orders' bytes, written in two orders.
 *
 * NSM (row-major): every column of a row is contiguous, so the read granule —
 * an 8 KB heap page, a 16 KB InnoDB clustered leaf — always carries whole rows.
 * DSM (column-major): every value of a column is contiguous, so the read granule
 * is a column chunk (a ClickHouse mark covering index_granularity = 8192 rows,
 * a Parquet page inside a column chunk) and carries one column of many rows.
 *
 * The query's needed bytes are the same set of cells in both pictures. What
 * differs is how much unwanted data comes along with them.
 */

/* ------------------------------------------------------------------ model */

type Col = { name: string; type: string; bytes: number };

const COLS: Col[] = [
  { name: 'order_id', type: 'bigint', bytes: 8 },
  { name: 'customer_id', type: 'bigint', bytes: 8 },
  { name: 'status', type: 'text', bytes: 10 },
  { name: 'region', type: 'text', bytes: 8 },
  { name: 'channel', type: 'text', bytes: 6 },
  { name: 'placed_at', type: 'timestamptz', bytes: 8 },
  { name: 'shipped_at', type: 'timestamptz', bytes: 8 },
  { name: 'item_count', type: 'int', bytes: 4 },
  { name: 'subtotal_cents', type: 'bigint', bytes: 8 },
  { name: 'tax_cents', type: 'bigint', bytes: 8 },
  { name: 'shipping_cents', type: 'bigint', bytes: 8 },
  { name: 'total_cents', type: 'bigint', bytes: 8 },
];

const EXTRA_WIDTH = 8; // each additional column is a bigint
const PAGE = 8192; // PostgreSQL BLCKSZ
const GRANULE_ROWS = 8192; // ClickHouse index_granularity: one mark per 8192 rows

type QueryId = 'point' | 'sum' | 'group';

const QUERIES: {
  value: QueryId;
  label: string;
  sql: string;
  needs: string[];
  scan: boolean;
}[] = [
  {
    value: 'point',
    label: 'Point lookup',
    sql: 'SELECT * FROM orders WHERE order_id = 90210',
    needs: COLS.map((c) => c.name),
    scan: false,
  },
  {
    value: 'sum',
    label: 'Sum one column',
    sql: 'SELECT sum(total_cents) FROM orders',
    needs: ['total_cents'],
    scan: true,
  },
  {
    value: 'group',
    label: 'Filtered group-by',
    sql: "SELECT region, sum(total_cents) FROM orders\n  WHERE placed_at >= now() - interval '7 days' GROUP BY region",
    needs: ['region', 'placed_at', 'total_cents'],
    scan: true,
  },
];

const SAMPLE_ROWS = 6;
const HIT_ROW = 3; // the row the point lookup wants

type CellState = 'needed' | 'wasted' | 'cold';

const STATE_FILL: Record<CellState, string> = {
  needed: 'color-mix(in srgb, var(--viz-1) 72%, var(--viz-plane))',
  wasted: 'color-mix(in srgb, var(--viz-stale) 42%, var(--viz-plane))',
  cold: 'var(--viz-plane)',
};

/* ----------------------------------------------------------------- render */

function Tape({
  title,
  sub,
  lines,
  width,
  gutter,
  pxPerByte,
  lineH,
}: {
  title: string;
  sub: string;
  lines: {
    label: string;
    cells: { w: number; state: CellState; tip: string; num?: number }[];
  }[];
  width: number;
  gutter: number;
  pxPerByte: number;
  lineH: number;
}) {
  const tip = useTip();
  const gap = 3;
  const top = 16;
  const h = top + lines.length * (lineH + gap) + 6;
  return (
    <div style={{ flex: '1 1 320px', minWidth: 300 }}>
      <div style={{ fontSize: '.8125rem', fontWeight: 600, color: 'var(--viz-ink)' }}>{title}</div>
      <div style={{ fontSize: '.6875rem', color: 'var(--viz-ink-2)', margin: '.15rem 0 .35rem' }}>{sub}</div>
      <svg viewBox={`0 0 ${width} ${h}`} width={width} height={h} role="img" aria-label={`${title}. ${sub}`}>
        {lines.map((ln, i) => {
          const y = top + i * (lineH + gap);
          let x = gutter;
          return (
            <g key={ln.label}>
              <text x={gutter - 6} y={y + lineH / 2 + 3.5} textAnchor="end" style={{ fontSize: 10 }}>
                {ln.label}
              </text>
              {ln.cells.map((c, j) => {
                const w = Math.max(1.5, c.w * pxPerByte);
                const cx = x;
                x += w + 1;
                return (
                  <g key={j} {...tip(<span>{c.tip}</span>)} tabIndex={j === 0 ? 0 : -1}>
                    <rect
                      x={cx}
                      y={y}
                      width={w}
                      height={lineH}
                      rx={1.5}
                      fill={STATE_FILL[c.state]}
                      stroke="var(--viz-border)"
                      strokeWidth={1}
                      strokeDasharray={c.state === 'cold' ? '2 2' : undefined}
                    />
                    {c.num !== undefined && w >= 15 ? (
                      <text
                        x={cx + w / 2}
                        y={y + lineH / 2 + 3.5}
                        textAnchor="middle"
                        style={{ fontSize: 9 }}
                        fill={c.state === 'needed' ? 'var(--viz-surface)' : 'var(--viz-ink-2)'}
                      >
                        {c.num}
                      </text>
                    ) : null}
                  </g>
                );
              })}
            </g>
          );
        })}
      </svg>
    </div>
  );
}

/* ------------------------------------------------------------------ panel */

export default function RowColumnByteFootprint() {
  const [queryId, setQueryId] = useState<QueryId>('sum');
  const [extra, setExtra] = useState(0);
  const [rowsExp, setRowsExp] = useState(7);
  const [ref, width] = useSize(720);

  const q = QUERIES.find((x) => x.value === queryId)!;
  const rows = 10 ** rowsExp;

  const namedBytes = COLS.reduce((a, c) => a + c.bytes, 0);
  const extraBytes = extra * EXTRA_WIDTH;
  const rowBytes = namedBytes + extraBytes;
  const totalCols = COLS.length + extra;

  const needsSet = useMemo(() => new Set(q.needs), [q]);
  const neededPerRow = COLS.filter((c) => needsSet.has(c.name)).reduce((a, c) => a + c.bytes, 0);

  // Bytes actually pulled off storage.
  const rowStore = q.scan ? rows * rowBytes : PAGE;
  const colStore = q.scan ? rows * neededPerRow : GRANULE_ROWS * rowBytes;
  const needed = q.scan ? rows * neededPerRow : rowBytes;
  const ratio = colStore / rowStore;

  // geometry
  const twoUp = width >= 700;
  const panelW = twoUp ? Math.floor((width - 18) / 2) : Math.max(300, width);
  const nsmGutter = 44;
  const dsmGutter = 96;
  const widestCol = Math.max(...COLS.map((c) => c.bytes), extraBytes);
  const pxPerByteN = (panelW - nsmGutter - COLS.length - 2) / rowBytes;
  const pxPerByteD = (panelW - dsmGutter - SAMPLE_ROWS - 2) / (SAMPLE_ROWS * widestCol);

  const cellState = (rowIdx: number, colName: string | null): CellState => {
    if (q.scan) {
      return colName !== null && needsSet.has(colName) ? 'needed' : 'wasted';
    }
    return rowIdx === HIT_ROW ? 'needed' : 'wasted';
  };

  // NSM: one line per row, cells are the columns in declared order.
  const nsmLines = Array.from({ length: SAMPLE_ROWS }, (_, r) => ({
    label: `row ${r + 1}`,
    cells: [
      ...COLS.map((c, i) => ({
        w: c.bytes,
        state: cellState(r, c.name),
        num: i + 1,
        tip: `row ${r + 1} · ${i + 1} ${c.name} (${c.type}, ${c.bytes} B) — ${
          cellState(r, c.name) === 'needed' ? 'the query needs this' : 'read anyway: it shares the page with a needed row'
        }`,
      })),
      ...(extra > 0
        ? [
            {
              w: extraBytes,
              state: cellState(r, null),
              tip: `row ${r + 1} · ${extra} further bigint columns (${extraBytes} B) — never referenced by this query, still on the page`,
            },
          ]
        : []),
    ],
  }));

  // DSM: one line per column, cells are the rows.
  const dsmLines = [
    ...COLS.map((c, i) => ({
      label: `${i + 1} ${c.name}`,
      cells: Array.from({ length: SAMPLE_ROWS }, (_, r) => {
        const st: CellState = q.scan
          ? needsSet.has(c.name)
            ? 'needed'
            : 'cold'
          : r === HIT_ROW
            ? 'needed'
            : 'wasted';
        return {
          w: c.bytes,
          state: st,
          tip: `${c.name}[row ${r + 1}] (${c.bytes} B) — ${
            st === 'needed'
              ? 'the query needs this'
              : st === 'cold'
                ? 'this column chunk is never opened'
                : 'read anyway: it is inside the granule that holds the needed value'
          }`,
        };
      }),
    })),
    ...(extra > 0
      ? [
          {
            label: `+${extra} more`,
            cells: Array.from({ length: SAMPLE_ROWS }, (_, r) => ({
              w: extraBytes,
              state: (q.scan ? 'cold' : r === HIT_ROW ? 'needed' : 'wasted') as CellState,
              tip: `${extra} further bigint columns for row ${r + 1} (${extraBytes} B)`,
            })),
          },
        ]
      : []),
  ];

  return (
    <VizPanel
      title="One orders table, two byte orders"
      subtitle="The same cells in both pictures. Pick a query and watch which bytes the storage layer is forced to drag along with the ones you asked for."
      controls={
        <>
          <Segmented
            label="Query"
            value={queryId}
            onChange={setQueryId}
            options={QUERIES.map((x) => ({ value: x.value, label: x.label, title: x.sql }))}
          />
          <Slider
            label="Columns in table"
            min={0}
            max={88}
            step={4}
            value={extra}
            onChange={setExtra}
            format={() => `${totalCols}`}
          />
          <Slider
            label="Rows scanned"
            min={4}
            max={9}
            value={rowsExp}
            onChange={setRowsExp}
            format={() => fmtNum(rows)}
          />
        </>
      }
      legend={
        <Legend
          items={[
            { label: 'Bytes the query needs', color: 'var(--viz-1)' },
            { label: 'Read anyway (same granule)', color: 'var(--viz-stale)' },
            { label: 'Never touched (dashed)', color: 'var(--viz-neutral)' },
          ]}
        />
      }
      stats={
        <Stats
          items={[
            { label: 'Bytes the answer needs', value: fmtBytes(needed) },
            {
              label: 'Row store reads',
              value: fmtBytes(rowStore),
              hint: q.scan ? 'every column of every row' : 'one 8 KB heap page',
            },
            {
              label: 'Column store reads',
              value: fmtBytes(colStore),
              hint: q.scan
                ? 'only the referenced column chunks'
                : 'one granule per column: 8192 rows × every column',
            },
            {
              label: 'Columnar / row-wise',
              value: ratio >= 1 ? `${ratio.toFixed(1)}× worse` : `${(ratio * 100).toFixed(ratio < 0.1 ? 1 : 0)}%`,
            },
          ]}
        />
      }
      note={
        queryId === 'point' ? (
          <>
            <strong>The row store wins by ~{Math.round(ratio)}×.</strong> The lookup wants every column of
            one row; row-major put them in one page, so one {fmtBytes(PAGE)} read answers it. Column-major
            has to open {totalCols} separate chunks and each one hands back a whole granule —{' '}
            {fmtNum(GRANULE_ROWS)} rows of a column you needed one value from.
          </>
        ) : (
          <>
            <strong>
              The column store reads {(ratio * 100).toFixed(ratio < 0.1 ? 1 : 0)}% of what the row store
              reads.
            </strong>{' '}
            The query touches {q.needs.length} of {totalCols} columns. Row-major cannot read a column
            without reading the rows it lives in, so it pays for all {totalCols}; column-major never opens
            the other {totalCols - q.needs.length} files at all. Drag the column slider to 100 and the
            scan collapses to about one percent.
          </>
        )
      }
    table={
      <table className="viz-table">
        <thead>
          <tr>
            <th>Query</th>
              <th>Columns in table</th>
              <th>Needed bytes</th>
              <th>Row store</th>
              <th>Column store</th>
              <th>Ratio</th>
            </tr>
          </thead>
          <tbody>
            {QUERIES.flatMap((qq) =>
              [12, 36, 100].map((nc) => {
                const rb = namedBytes + (nc - COLS.length) * EXTRA_WIDTH;
                const nb = COLS.filter((c) => qq.needs.includes(c.name)).reduce((a, c) => a + c.bytes, 0);
                const rs = qq.scan ? rows * rb : PAGE;
                const cs = qq.scan ? rows * nb : GRANULE_ROWS * rb;
                return (
                  <tr key={`${qq.value}-${nc}`}>
                    <td>{qq.label}</td>
                    <td>{nc}</td>
                    <td>{fmtBytes(qq.scan ? rows * nb : rb)}</td>
                    <td>{fmtBytes(rs)}</td>
                    <td>{fmtBytes(cs)}</td>
                    <td>{cs >= rs ? `${(cs / rs).toFixed(1)}× worse` : `${((cs / rs) * 100).toFixed(1)}%`}</td>
                  </tr>
                );
              }),
            )}
          </tbody>
      </table>
    }
    >
      <div ref={ref}>
        <TooltipHost>
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: '.75rem 1rem' }}>
            <Tape
              title="NSM — row-major"
              sub="PostgreSQL heap page, InnoDB clustered-index leaf. One line = one row; the numbers are column positions."
              lines={nsmLines}
              width={panelW}
              gutter={nsmGutter}
              pxPerByte={pxPerByteN}
              lineH={18}
            />
            <Tape
              title="DSM — column-major"
              sub={`ClickHouse MergeTree part (one .bin per column), Parquet column chunk. One line = one column across ${SAMPLE_ROWS} rows; each panel fills its own width, so compare within a panel and read the totals below.`}
              lines={dsmLines}
              width={panelW}
              gutter={dsmGutter}
              pxPerByte={pxPerByteD}
              lineH={12}
            />
          </div>
        </TooltipHost>
      </div>
    </VizPanel>
  );
}
