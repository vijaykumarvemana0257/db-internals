import { useState } from 'react';
import {
  VizPanel,
  Choice,
  Segmented,
  Button,
  Legend,
  Stats,
  Note,
  TooltipHost,
  useTip,
  useSize,
} from './Viz';

/**
 * The logical evaluation pipeline for a two-table query, run one clause at a time.
 *
 * The semantics being modelled (not the physical plan — the optimizer is free to
 * reorder anything it can prove equivalent):
 *
 *   FROM      every candidate pair of rows exists: |customers| x |orders|
 *   JOIN/ON   the ON predicate keeps or drops each pair
 *   (pad)     outer joins re-add rows the ON predicate dropped entirely, padding
 *             the missing side with NULLs
 *   WHERE     filters the joined rows — after padding, which is why the same
 *             predicate behaves differently in ON and in WHERE
 *   GROUP BY  surviving rows fall into buckets; NULL keys form one bucket
 *   HAVING    whole buckets are dropped, on aggregate values
 *   SELECT    each bucket collapses to one row
 */

/* ------------------------------------------------------------------- data */

type Cust = { id: number; name: string; region: string };
type Order = { id: number; cust: number | null; amount: number; status: string };

const CUSTOMERS: Cust[] = [
  { id: 1, name: 'Ada', region: 'west' },
  { id: 2, name: 'Borg', region: 'west' },
  { id: 3, name: 'Cyd', region: 'east' },
  { id: 4, name: 'Dov', region: 'east' },
  { id: 5, name: 'Eze', region: 'north' },
];

const ORDERS: Order[] = [
  { id: 101, cust: 1, amount: 120, status: 'paid' },
  { id: 102, cust: 1, amount: 40, status: 'paid' },
  { id: 103, cust: 2, amount: 250, status: 'open' },
  { id: 104, cust: 3, amount: 90, status: 'paid' },
  { id: 105, cust: 3, amount: 60, status: 'open' },
  { id: 106, cust: null, amount: 75, status: 'paid' },
  { id: 107, cust: 5, amount: 300, status: 'open' },
];

/* -------------------------------------------------------------- predicates */

type JoinId = 'inner' | 'left' | 'right' | 'full' | 'cross';

const JOINS: { value: JoinId; label: string; sql: string }[] = [
  { value: 'inner', label: 'INNER', sql: 'JOIN' },
  { value: 'left', label: 'LEFT', sql: 'LEFT JOIN' },
  { value: 'right', label: 'RIGHT', sql: 'RIGHT JOIN' },
  { value: 'full', label: 'FULL', sql: 'FULL JOIN' },
  { value: 'cross', label: 'CROSS', sql: 'CROSS JOIN' },
];

type OnId = 'eq' | 'eqpaid' | 'theta' | 'true';

const ON_PREDS: { value: OnId; label: string; fn: (c: Cust, o: Order) => boolean }[] = [
  { value: 'eq', label: 'o.customer_id = c.id', fn: (c, o) => o.cust !== null && o.cust === c.id },
  {
    value: 'eqpaid',
    label: "o.customer_id = c.id AND o.status = 'paid'",
    fn: (c, o) => o.cust !== null && o.cust === c.id && o.status === 'paid',
  },
  { value: 'theta', label: 'o.amount >= 100   (non-equi)', fn: (_c, o) => o.amount >= 100 },
  { value: 'true', label: 'TRUE   (full cartesian product)', fn: () => true },
];

type JRow = { c: Cust | null; o: Order | null; pad: 'none' | 'left' | 'right' };

type WhereId = 'none' | 'paid' | 'west' | 'anti';

const WHERES: { value: WhereId; label: string; fn: (r: JRow) => boolean }[] = [
  { value: 'none', label: '(no WHERE)', fn: () => true },
  { value: 'paid', label: "o.status = 'paid'", fn: (r) => r.o !== null && r.o.status === 'paid' },
  { value: 'west', label: "c.region = 'west'", fn: (r) => r.c !== null && r.c.region === 'west' },
  { value: 'anti', label: 'o.id IS NULL   (anti-join)', fn: (r) => r.o === null },
];

type GroupId = 'none' | 'region' | 'cust' | 'status';

const GROUPS: { value: GroupId; label: string; key: (r: JRow) => string | null }[] = [
  { value: 'none', label: '(no GROUP BY)', key: () => '*' },
  { value: 'region', label: 'c.region', key: (r) => (r.c ? r.c.region : null) },
  { value: 'cust', label: 'c.id, c.name', key: (r) => (r.c ? `${r.c.id} ${r.c.name}` : null) },
  { value: 'status', label: 'o.status', key: (r) => (r.o ? r.o.status : null) },
];

type HavingId = 'none' | 'cnt2' | 'sum200' | 'noorders';

const HAVINGS: { value: HavingId; label: string; fn: (b: Bucket) => boolean }[] = [
  { value: 'none', label: '(no HAVING)', fn: () => true },
  { value: 'cnt2', label: 'count(*) >= 2', fn: (b) => b.countStar >= 2 },
  { value: 'sum200', label: 'sum(o.amount) > 200', fn: (b) => b.sum !== null && b.sum > 200 },
  { value: 'noorders', label: 'count(o.id) = 0', fn: (b) => b.countCol === 0 },
];

/* ----------------------------------------------------------------- the run */

type Cell = { c: Cust; o: Order; kept: boolean };

type Bucket = {
  key: string;
  label: string;
  isNull: boolean;
  rows: JRow[];
  countStar: number;
  countCol: number;
  sum: number | null;
  passed: boolean;
};

type Cfg = { join: JoinId; on: OnId; where: WhereId; group: GroupId; having: HavingId };

type Run = {
  cells: Cell[];
  rows: JRow[];
  afterOn: number;
  padded: number;
  afterWhere: number;
  buckets: Bucket[];
  kept: number;
};

function analyze(cfg: Cfg): Run {
  const pred = cfg.join === 'cross' ? () => true : ON_PREDS.find((p) => p.value === cfg.on)!.fn;

  const cells: Cell[] = [];
  for (const c of CUSTOMERS) for (const o of ORDERS) cells.push({ c, o, kept: pred(c, o) });

  const matchedO = new Set(cells.filter((x) => x.kept).map((x) => x.o.id));

  const rows: JRow[] = [];
  for (const c of CUSTOMERS) {
    const mine = cells.filter((x) => x.c.id === c.id && x.kept);
    for (const m of mine) rows.push({ c, o: m.o, pad: 'none' });
    if (mine.length === 0 && (cfg.join === 'left' || cfg.join === 'full')) {
      rows.push({ c, o: null, pad: 'left' });
    }
  }
  if (cfg.join === 'right' || cfg.join === 'full') {
    for (const o of ORDERS) if (!matchedO.has(o.id)) rows.push({ c: null, o, pad: 'right' });
  }

  const wf = WHERES.find((w) => w.value === cfg.where)!.fn;
  const survivors = rows.filter(wf);

  const gdef = GROUPS.find((g) => g.value === cfg.group)!;
  const hf = HAVINGS.find((h) => h.value === cfg.having)!.fn;

  const order: string[] = [];
  const bag = new Map<string, JRow[]>();
  for (const r of survivors) {
    const k = gdef.key(r);
    const id = k === null ? ' NULL' : k;
    if (!bag.has(id)) {
      bag.set(id, []);
      order.push(id);
    }
    bag.get(id)!.push(r);
  }
  // Aggregates with no GROUP BY form one group over the whole result, and that group
  // exists even when nothing survived: SELECT count(*) always returns exactly one row.
  if (cfg.group === 'none' && order.length === 0) {
    order.push('*');
    bag.set('*', []);
  }

  const buckets: Bucket[] = order.map((id) => {
    const rs = bag.get(id)!;
    const withOrder = rs.filter((r) => r.o !== null);
    const sum = withOrder.length === 0 ? null : withOrder.reduce((a, r) => a + r.o!.amount, 0);
    const b: Bucket = {
      key: id,
      label: id === '*' ? 'whole result' : id === ' NULL' ? 'NULL' : id,
      isNull: id === ' NULL',
      rows: rs,
      countStar: rs.length,
      countCol: withOrder.length,
      sum,
      passed: true,
    };
    b.passed = hf(b);
    return b;
  });

  return {
    cells,
    rows,
    afterOn: cells.filter((x) => x.kept).length,
    padded: rows.filter((r) => r.pad !== 'none').length,
    afterWhere: survivors.length,
    buckets,
    kept: buckets.filter((b) => b.passed).length,
  };
}

/* ------------------------------------------------------------------ stages */

const STAGES = [
  { value: '0', label: 'FROM' },
  { value: '1', label: 'JOIN / ON' },
  { value: '2', label: 'NULL pad' },
  { value: '3', label: 'WHERE' },
  { value: '4', label: 'GROUP BY' },
  { value: '5', label: 'HAVING' },
  { value: '6', label: 'SELECT' },
] as const;

function narrate(stage: number, cfg: Cfg, r: Run): { head: string; body: string } {
  const onLabel = cfg.join === 'cross' ? 'TRUE' : ON_PREDS.find((p) => p.value === cfg.on)!.label;
  const nCells = CUSTOMERS.length * ORDERS.length;
  switch (stage) {
    case 0:
      return {
        head: `FROM: ${CUSTOMERS.length} customers by ${ORDERS.length} orders = ${nCells} candidate pairs.`,
        body:
          'The relational definition of a join starts here: every row of the left table paired with every ' +
          'row of the right one. Nothing has been filtered and no NULL has been invented yet. A real engine ' +
          'never materializes this grid — it builds a hash table or merges two sorted inputs — but the answer ' +
          'it must produce is defined as if it did.',
      };
    case 1:
      return {
        head: `ON ${onLabel} keeps ${r.afterOn} of ${nCells} pairs.`,
        body:
          cfg.join === 'cross'
            ? 'CROSS JOIN takes no ON clause: every pair survives. This is what an accidentally omitted join ' +
              'predicate produces, and why a two-table query over a few thousand rows can return millions.'
            : 'The ON clause is a filter over pairs and nothing more. A pair survives only when the predicate ' +
              'evaluates to TRUE — FALSE and UNKNOWN are both dropped, which is why order 106, whose ' +
              'customer_id is NULL, matches no customer at all under an equality predicate.',
      };
    case 2: {
      if (cfg.join === 'inner' || cfg.join === 'cross') {
        return {
          head: 'No padding: an inner join emits only matched pairs.',
          body:
            'Switch to LEFT, RIGHT or FULL and the rows the ON clause eliminated entirely come back, with the ' +
            'other side filled in as NULL. That is the whole difference between an inner and an outer join.',
        };
      }
      return {
        head: `${r.padded} row${r.padded === 1 ? '' : 's'} re-added with NULL padding.`,
        body:
          'An outer join preserves rows that found no partner at all. The preserved row keeps its own columns ' +
          'and gets NULLs for every column of the other side — those NULLs are manufactured by the join, not ' +
          'stored anywhere, and they are why count(*) and count(o.amount) start to disagree.',
      };
    }
    case 3: {
      const w = WHERES.find((x) => x.value === cfg.where)!;
      if (cfg.where === 'none') {
        return {
          head: `WHERE: nothing filtered, ${r.afterWhere} rows carry on.`,
          body:
            'Pick a WHERE predicate and watch where it cuts. With an outer join the placement matters: the same ' +
            'predicate in ON filters pairs before padding, in WHERE it filters after — and a NULL-padded row ' +
            'fails almost every predicate written against the padded side.',
        };
      }
      return {
        head: `WHERE ${w.label} leaves ${r.afterWhere} row${r.afterWhere === 1 ? '' : 's'}.`,
        body:
          cfg.where === 'anti'
            ? 'This is the anti-join idiom: take an outer join, then keep only the rows where the padded side is ' +
              'NULL. Under an INNER join it can never match anything, because an inner join produces no padding at all.'
            : 'WHERE runs on joined rows, after padding. A NULL-padded row has NULL in every column of the other ' +
              'side, so a predicate on that side evaluates to UNKNOWN and the row disappears — silently turning ' +
              'your outer join back into an inner one.',
      };
    }
    case 4: {
      const g = GROUPS.find((x) => x.value === cfg.group)!;
      if (cfg.group === 'none') {
        return {
          head: 'No GROUP BY: the aggregates see one implicit group.',
          body:
            'A SELECT list containing only aggregates and no GROUP BY is one group over the entire result — and ' +
            'it returns exactly one row even when zero rows reached it. Add a GROUP BY and zero input rows produce ' +
            'zero output rows instead. That asymmetry is what breaks dashboards expecting a 0.',
        };
      }
      return {
        head: `GROUP BY ${g.label} makes ${r.buckets.length} bucket${r.buckets.length === 1 ? '' : 's'}.`,
        body:
          'Each surviving row is routed to exactly one bucket by its grouping key. NULL keys all land in a single ' +
          'NULL bucket: GROUP BY compares keys with the null-safe "is not distinct from", not with =, which makes ' +
          'it one of the few places in SQL where two NULLs count as the same value.',
      };
    }
    case 5: {
      const h = HAVINGS.find((x) => x.value === cfg.having)!;
      if (cfg.having === 'none') {
        return {
          head: 'HAVING: no buckets dropped.',
          body:
            'HAVING is WHERE for buckets: it runs after aggregation and can therefore see count(*) and sum(). A ' +
            'filter that needs no aggregate belongs in WHERE, where it removes rows before they are ever grouped.',
        };
      }
      return {
        head: `HAVING ${h.label} keeps ${r.kept} of ${r.buckets.length} buckets.`,
        body:
          cfg.having === 'sum200'
            ? 'Watch a bucket whose only rows are NULL-padded: sum() over no non-NULL values is NULL, NULL > 200 is ' +
              'UNKNOWN, and the bucket is discarded exactly as if the comparison had been false.'
            : 'Whole buckets vanish, not individual rows. The aggregate values were computed over every row that ' +
              'reached the bucket, including the ones this predicate now throws away.',
      };
    }
    default:
      return {
        head: `SELECT emits ${r.kept} row${r.kept === 1 ? '' : 's'} — one per surviving bucket.`,
        body:
          'Only now is the SELECT list evaluated, which is why you cannot reference a SELECT alias in WHERE or in ' +
          'ON: when those clauses run, the output columns do not exist yet. count(*) counts rows in the bucket; ' +
          'count(o.amount) counts non-NULL values, and the gap between them is the number of NULL-padded rows this ' +
          'join manufactured.',
      };
  }
}

/* ---------------------------------------------------------------- the draw */

type CellState = 'candidate' | 'kept' | 'dropped' | 'wherecut';

const STATE_FILL: Record<CellState, string> = {
  candidate: 'var(--viz-neutral)',
  kept: 'var(--viz-1)',
  dropped: 'var(--viz-plane)',
  wherecut: 'var(--viz-stale)',
};

export default function JoinGroupPipeline() {
  const [join, setJoin] = useState<JoinId>('left');
  const [on, setOn] = useState<OnId>('eq');
  const [where, setWhere] = useState<WhereId>('none');
  const [group, setGroup] = useState<GroupId>('region');
  const [having, setHaving] = useState<HavingId>('none');
  const [stage, setStage] = useState(1);
  const [ref, width] = useSize(720);
  const tip = useTip();

  const cfg: Cfg = { join, on, where, group, having };
  const run = analyze(cfg);
  const msg = narrate(stage, cfg, run);
  const wf = WHERES.find((w) => w.value === where)!.fn;
  const joinDef = JOINS.find((j) => j.value === join)!;
  const onLabel = join === 'cross' ? '' : ON_PREDS.find((p) => p.value === on)!.label;

  const showPadCol = stage >= 2 && (join === 'left' || join === 'full');
  const showPadRow = stage >= 2 && (join === 'right' || join === 'full');

  /* geometry */
  const labelW = 116;
  const cellW = 58;
  const rowH = 40;
  const cols = ORDERS.length + 1; // + the "no match" gutter column
  const sqlH = 8 * 16 + 10;
  const matTop = sqlH + 6;
  const headH = 34;
  const gridTop = matTop + headH;
  const matH = headH + CUSTOMERS.length * rowH + rowH;
  const bucketsTop = matTop + matH + 22;
  const bucketW = 112;
  const bucketH = 92;
  const showBuckets = stage >= 4;
  const baseW = labelW + cols * cellW + 18;
  const svgW = Math.max(width, baseW, showBuckets ? labelW + run.buckets.length * bucketW + 18 : 0);
  const svgH = bucketsTop + (showBuckets ? bucketH + 12 : 0);

  const colX = (i: number) => labelW + i * cellW;
  const rowY = (i: number) => gridTop + i * rowH;
  const padRowY = rowY(CUSTOMERS.length);

  /* the query text, with the clause for the current stage highlighted */
  const sql: { stages: number[]; text: string }[] = [
    {
      stages: [6],
      text: `SELECT   ${group === 'none' ? "'all'" : GROUPS.find((g) => g.value === group)!.label},`,
    },
    { stages: [6], text: '         count(*), count(o.amount), sum(o.amount)' },
    { stages: [0], text: 'FROM     customers c' },
    { stages: [1, 2], text: `${joinDef.sql.padEnd(9)}orders o` },
    { stages: [1, 2], text: join === 'cross' ? '' : `  ON     ${onLabel}` },
    {
      stages: [3],
      text: where === 'none' ? '' : `WHERE    ${WHERES.find((w) => w.value === where)!.label}`,
    },
    {
      stages: [4],
      text: group === 'none' ? '' : `GROUP BY ${GROUPS.find((g) => g.value === group)!.label}`,
    },
    {
      stages: [5],
      text: having === 'none' ? '' : `HAVING   ${HAVINGS.find((h) => h.value === having)!.label}`,
    },
  ];

  const cellState = (c: Cell): CellState => {
    if (stage === 0) return 'candidate';
    if (!c.kept) return 'dropped';
    if (stage >= 3 && !wf({ c: c.c, o: c.o, pad: 'none' })) return 'wherecut';
    return 'kept';
  };

  const padState = (r: JRow): CellState => (stage >= 3 && !wf(r) ? 'wherecut' : 'kept');
  const leftPads = run.rows.filter((r) => r.pad === 'left');
  const rightPads = run.rows.filter((r) => r.pad === 'right');

  return (
    <VizPanel
      title="One query, one clause at a time"
      subtitle="Every candidate row pair is drawn. Step the pipeline and watch ON keep or drop pairs, the outer join pad survivors back in with NULLs, WHERE cut what the padding created, and GROUP BY collapse the rest into buckets."
      controls={
        <>
          <Segmented label="Join" value={join} onChange={setJoin} options={JOINS} />
          <Choice
            label="ON"
            value={join === 'cross' ? 'true' : on}
            onChange={setOn}
            options={
              join === 'cross'
                ? [{ value: 'true' as OnId, label: 'CROSS JOIN takes no ON clause' }]
                : ON_PREDS.map((p) => ({ value: p.value, label: p.label }))
            }
          />
          <Choice
            label="WHERE"
            value={where}
            onChange={setWhere}
            options={WHERES.map((w) => ({ value: w.value, label: w.label }))}
          />
          <Choice
            label="GROUP BY"
            value={group}
            onChange={setGroup}
            options={GROUPS.map((g) => ({ value: g.value, label: g.label }))}
          />
          <Choice
            label="HAVING"
            value={having}
            onChange={setHaving}
            options={HAVINGS.map((h) => ({ value: h.value, label: h.label }))}
          />
          <Segmented label="Stage" value={String(stage)} onChange={(v) => setStage(Number(v))} options={STAGES} />
          <Button onClick={() => setStage((s) => (s + 1) % STAGES.length)} primary>
            Step
          </Button>
        </>
      }
      legend={
        <Legend
          items={[
            { label: 'pair kept by ON (shows o.amount)', color: 'var(--viz-1)' },
            { label: 'NULL-padded row (outer join)', color: 'var(--viz-2)' },
            { label: 'removed by WHERE, marked x', color: 'var(--viz-stale)' },
            { label: 'pair dropped by ON (outlined)', color: 'var(--viz-ink-muted)', shape: 'line' },
            { label: 'GROUP BY bucket', color: 'var(--viz-7)' },
          ]}
        />
      }
      stats={
        <Stats
          items={[
            {
              label: 'Candidate pairs',
              value: CUSTOMERS.length * ORDERS.length,
              hint: 'customers x orders — the cartesian product a join is defined over',
            },
            { label: 'Kept by ON', value: run.afterOn },
            {
              label: 'NULL-padded',
              value: run.padded,
              hint: 'Rows the outer join re-added with NULLs on the unmatched side',
            },
            { label: 'Rows after WHERE', value: run.afterWhere },
            { label: 'Buckets', value: run.buckets.length },
            { label: 'Result rows', value: run.kept, hint: 'One row per bucket that survived HAVING' },
          ]}
        />
      }
      note={
        <Note>
          <strong>{msg.head}</strong> {msg.body}
        </Note>
      }
      table={
        <>
          <table className="viz-table">
            <thead>
              <tr>
                <th colSpan={3}>customers c</th>
              </tr>
              <tr>
                <th>id</th>
                <th>name</th>
                <th>region</th>
              </tr>
            </thead>
            <tbody>
              {CUSTOMERS.map((c) => (
                <tr key={c.id}>
                  <td>{c.id}</td>
                  <td>{c.name}</td>
                  <td>{c.region}</td>
                </tr>
              ))}
            </tbody>
          </table>
          <table className="viz-table">
            <thead>
              <tr>
                <th colSpan={4}>orders o</th>
              </tr>
              <tr>
                <th>id</th>
                <th>customer_id</th>
                <th>amount</th>
                <th>status</th>
              </tr>
            </thead>
            <tbody>
              {ORDERS.map((o) => (
                <tr key={o.id}>
                  <td>{o.id}</td>
                  <td>{o.cust === null ? 'NULL' : o.cust}</td>
                  <td>{o.amount}</td>
                  <td>{o.status}</td>
                </tr>
              ))}
            </tbody>
          </table>
          <table className="viz-table">
            <thead>
              <tr>
                <th colSpan={6}>result set</th>
              </tr>
              <tr>
                <th>group</th>
                <th>count(*)</th>
                <th>count(o.amount)</th>
                <th>sum(o.amount)</th>
                <th>member rows</th>
                <th>HAVING</th>
              </tr>
            </thead>
            <tbody>
              {run.buckets.length === 0 ? (
                <tr>
                  <td colSpan={6}>no rows</td>
                </tr>
              ) : (
                run.buckets.map((b) => (
                  <tr key={b.key}>
                    <td>{b.label}</td>
                    <td>{b.countStar}</td>
                    <td>{b.countCol}</td>
                    <td>{b.sum === null ? 'NULL' : b.sum}</td>
                    <td>{b.rows.map((r) => (r.o ? `o${r.o.id}` : `c${r.c!.id}+NULL`)).join(' ') || 'none'}</td>
                    <td>{b.passed ? 'kept' : 'dropped'}</td>
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
            height={svgH}
            role="img"
            aria-label="Candidate row pairs, join filtering, NULL padding and group buckets"
          >
            {/* the query, with the clause evaluated at this stage highlighted */}
            {sql.map((ln, i) =>
              ln.text === '' ? null : (
                <g key={i}>
                  {ln.stages.includes(stage) ? (
                    <rect x={0} y={i * 16 + 2} width={Math.max(260, svgW - 4)} height={16} rx={4} fill="var(--viz-neutral)" />
                  ) : null}
                  <text
                    x={6}
                    y={i * 16 + 14}
                    fill={ln.stages.includes(stage) ? 'var(--viz-ink)' : 'var(--viz-ink-muted)'}
                    fontWeight={ln.stages.includes(stage) ? 600 : 400}
                    xmlSpace="preserve"
                  >
                    {ln.text}
                  </text>
                </g>
              ),
            )}

            {/* column headers: one per orders row */}
            {ORDERS.map((o, i) => (
              <g key={o.id}>
                <text x={colX(i) + 25} y={matTop + 13} textAnchor="middle" fill="var(--viz-ink)" fontWeight={600}>
                  o{o.id}
                </text>
                <text x={colX(i) + 25} y={matTop + 26} textAnchor="middle" fill="var(--viz-ink-muted)">
                  {o.cust === null ? 'NULL' : `c${o.cust}`} / {o.amount}
                </text>
              </g>
            ))}
            {showPadCol ? (
              <text
                x={colX(ORDERS.length) + 25}
                y={matTop + 13}
                textAnchor="middle"
                fill="var(--viz-2)"
                fontWeight={600}
              >
                no match
              </text>
            ) : null}

            {/* row labels + the grid of candidate pairs */}
            {CUSTOMERS.map((c, ri) => {
              const pr = leftPads.find((r) => r.c!.id === c.id);
              return (
                <g key={c.id}>
                  <text x={0} y={rowY(ri) + 18} fill="var(--viz-ink)" fontWeight={600}>
                    c{c.id} {c.name}
                  </text>
                  <text x={0} y={rowY(ri) + 31} fill="var(--viz-ink-muted)">
                    {c.region}
                  </text>
                  {ORDERS.map((o, ci) => {
                    const cell = run.cells.find((x) => x.c.id === c.id && x.o.id === o.id)!;
                    const st = cellState(cell);
                    const why =
                      st === 'kept'
                        ? 'ON is TRUE for this pair, so it becomes a joined row.'
                        : st === 'wherecut'
                          ? 'Joined by ON, then removed by the WHERE predicate.'
                          : o.cust === null && (on === 'eq' || on === 'eqpaid')
                            ? 'o.customer_id is NULL, so NULL = c.id is UNKNOWN. Not TRUE, so the pair is dropped.'
                            : 'ON is FALSE for this pair, so it is dropped.';
                    return (
                      <g
                        key={o.id}
                        {...tip(
                          <>
                            <strong>
                              c{c.id} {c.name} x o{o.id}
                            </strong>
                            <br />
                            region {c.region}, {o.cust === null ? 'customer_id NULL' : `customer_id ${o.cust}`},
                            amount {o.amount}, {o.status}
                            <br />
                            {why}
                          </>,
                        )}
                      >
                        <rect
                          x={colX(ci) + 3}
                          y={rowY(ri) + 4}
                          width={cellW - 8}
                          height={rowH - 10}
                          rx={5}
                          fill={STATE_FILL[st]}
                          stroke={st === 'dropped' ? 'var(--viz-border)' : 'var(--viz-surface)'}
                          strokeWidth={st === 'dropped' ? 1 : 1.5}
                        />
                        <text
                          x={colX(ci) + 25}
                          y={rowY(ri) + 22}
                          textAnchor="middle"
                          fill={st === 'kept' || st === 'wherecut' ? 'var(--viz-surface)' : 'var(--viz-ink-muted)'}
                          fontWeight={st === 'kept' ? 600 : 400}
                        >
                          {st === 'kept' ? o.amount : st === 'wherecut' ? 'x' : '.'}
                        </text>
                      </g>
                    );
                  })}
                  {showPadCol && pr ? (
                    <g
                      {...tip(
                        <>
                          <strong>NULL-padded row</strong>
                          <br />c{c.id} {c.name} matched no order under this ON clause, so the outer join emits the
                          customer with every orders column set to NULL.
                          {padState(pr) === 'wherecut' ? ' The WHERE predicate then dropped it.' : ''}
                        </>,
                      )}
                    >
                      <rect
                        x={colX(ORDERS.length) + 3}
                        y={rowY(ri) + 4}
                        width={cellW - 8}
                        height={rowH - 10}
                        rx={5}
                        fill={padState(pr) === 'wherecut' ? 'var(--viz-stale)' : 'var(--viz-2)'}
                        stroke="var(--viz-surface)"
                        strokeWidth={1.5}
                      />
                      <text
                        x={colX(ORDERS.length) + 25}
                        y={rowY(ri) + 22}
                        textAnchor="middle"
                        fill="var(--viz-surface)"
                        fontWeight={600}
                      >
                        {padState(pr) === 'wherecut' ? 'x' : 'NULL'}
                      </text>
                    </g>
                  ) : null}
                </g>
              );
            })}

            {/* gutter row: orders preserved by a RIGHT or FULL join */}
            {showPadRow ? (
              <g>
                <text x={0} y={padRowY + 18} fill="var(--viz-2)" fontWeight={600}>
                  no match
                </text>
                <text x={0} y={padRowY + 31} fill="var(--viz-ink-muted)">
                  NULL customer
                </text>
                {ORDERS.map((o, ci) => {
                  const pr = rightPads.find((r) => r.o!.id === o.id);
                  if (!pr) return null;
                  const st = padState(pr);
                  return (
                    <g
                      key={o.id}
                      {...tip(
                        <>
                          <strong>NULL-padded row</strong>
                          <br />o{o.id} matched no customer, so a {joinDef.label} join emits it with every
                          customers column NULL.
                        </>,
                      )}
                    >
                      <rect
                        x={colX(ci) + 3}
                        y={padRowY + 4}
                        width={cellW - 8}
                        height={rowH - 10}
                        rx={5}
                        fill={st === 'wherecut' ? 'var(--viz-stale)' : 'var(--viz-2)'}
                        stroke="var(--viz-surface)"
                        strokeWidth={1.5}
                      />
                      <text
                        x={colX(ci) + 25}
                        y={padRowY + 22}
                        textAnchor="middle"
                        fill="var(--viz-surface)"
                        fontWeight={600}
                      >
                        {st === 'wherecut' ? 'x' : 'NULL'}
                      </text>
                    </g>
                  );
                })}
              </g>
            ) : null}

            {/* the buckets */}
            {showBuckets ? (
              <g>
                <text x={0} y={bucketsTop - 6} fill="var(--viz-ink-2)">
                  {run.buckets.length} bucket{run.buckets.length === 1 ? '' : 's'}, {run.kept} result row
                  {run.kept === 1 ? '' : 's'}
                </text>
                {run.buckets.map((b, i) => {
                  const dropped = stage >= 5 && !b.passed;
                  return (
                    <g
                      key={b.key}
                      {...tip(
                        <>
                          <strong>bucket {b.label}</strong>
                          <br />
                          {b.countStar} row{b.countStar === 1 ? '' : 's'}, {b.countCol} with a non-NULL amount.
                          <br />
                          sum(o.amount) = {b.sum === null ? 'NULL (no non-NULL values, not 0)' : b.sum}
                          {dropped ? (
                            <>
                              <br />
                              Dropped by HAVING.
                            </>
                          ) : null}
                        </>,
                      )}
                    >
                      <rect
                        x={labelW + i * bucketW}
                        y={bucketsTop}
                        width={bucketW - 8}
                        height={bucketH}
                        rx={7}
                        fill="var(--viz-plane)"
                        stroke={dropped ? 'var(--viz-stale)' : 'var(--viz-7)'}
                        strokeWidth={dropped ? 1 : 2}
                        strokeDasharray={dropped ? '4 3' : undefined}
                      />
                      <text
                        x={labelW + i * bucketW + 8}
                        y={bucketsTop + 17}
                        fill={dropped ? 'var(--viz-stale)' : 'var(--viz-ink)'}
                        fontWeight={600}
                      >
                        {b.label}
                      </text>
                      <text x={labelW + i * bucketW + 8} y={bucketsTop + 34} fill="var(--viz-ink-2)">
                        count(*) = {b.countStar}
                      </text>
                      <text x={labelW + i * bucketW + 8} y={bucketsTop + 48} fill="var(--viz-ink-2)">
                        count(amt) = {b.countCol}
                      </text>
                      <text
                        x={labelW + i * bucketW + 8}
                        y={bucketsTop + 62}
                        fill={b.sum === null ? 'var(--viz-2)' : 'var(--viz-ink-2)'}
                        fontWeight={b.sum === null ? 600 : 400}
                      >
                        sum = {b.sum === null ? 'NULL' : b.sum}
                      </text>
                      <text x={labelW + i * bucketW + 8} y={bucketsTop + 79} fill="var(--viz-ink-muted)">
                        {dropped
                          ? 'x HAVING'
                          : b.rows.map((r) => (r.o ? String(r.o.id) : 'pad')).join(' ') || 'empty'}
                      </text>
                    </g>
                  );
                })}
              </g>
            ) : null}
          </svg>
        </TooltipHost>
      </div>
    </VizPanel>
  );
}
