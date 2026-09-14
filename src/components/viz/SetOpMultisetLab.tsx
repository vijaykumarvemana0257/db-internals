import { useState } from 'react';
import { VizPanel, Segmented, Check, Legend, Stats, Note, TooltipHost, useTip, useSize } from './Viz';

/**
 * UNION / INTERSECT / EXCEPT as multiset arithmetic.
 *
 * Every set operator has two forms. The bag form (ALL) does arithmetic on
 * multiplicities: a+b for UNION ALL, min(a,b) for INTERSECT ALL, max(a-b,0) for
 * EXCEPT ALL. The set form clamps every non-zero multiplicity to 1, which costs a
 * sort or a hash over the whole result. Both forms compare rows with the null-safe
 * "is not distinct from", so two NULLs are duplicates of each other here even
 * though NULL = NULL is UNKNOWN everywhere else.
 */

type OpId = 'union' | 'intersect' | 'except';

const OPS: { value: OpId; label: string }[] = [
  { value: 'union', label: 'UNION' },
  { value: 'intersect', label: 'INTERSECT' },
  { value: 'except', label: 'EXCEPT' },
];

const LEFT_BASE = ['west', 'west', 'east', 'east', 'north'];
const RIGHT_BASE = ['east', 'east', 'east', 'south'];
const NULLV = '(NULL)';

type Fate = 'out' | 'dup' | 'cut' | 'match' | 'cancel';

const TAG: Record<Fate, string> = {
  out: 'keep',
  dup: 'dup',
  cut: 'drop',
  match: 'match',
  cancel: 'cancels',
};

const HINT: Record<Fate, string> = {
  out: 'Emitted into the result.',
  dup: 'Folded into an identical row already emitted. The set form keeps one copy of each distinct value.',
  cut: 'Eliminated by the operator: this value does not qualify.',
  match: 'Its twin on the left is what gets emitted; this row only licenses that one.',
  cancel: 'Cancels one copy of the same value on the left.',
};

const PLAN: Record<string, string> = {
  'union-all': 'Append',
  'union-set': 'HashAggregate over Append',
  'intersect-all': 'HashSetOp Intersect All',
  'intersect-set': 'HashSetOp Intersect',
  'except-all': 'HashSetOp Except All',
  'except-set': 'HashSetOp Except',
};

function multiplicity(op: OpId, all: boolean, a: number, b: number): number {
  if (op === 'union') return all ? a + b : a + b > 0 ? 1 : 0;
  if (op === 'intersect') return all ? Math.min(a, b) : a > 0 && b > 0 ? 1 : 0;
  return all ? Math.max(a - b, 0) : a > 0 && b === 0 ? 1 : 0;
}

function rule(op: OpId, all: boolean): string {
  if (op === 'union') return all ? 'a + b' : 'a + b > 0 ? 1 : 0';
  if (op === 'intersect') return all ? 'min(a, b)' : 'a>0 and b>0 ? 1 : 0';
  return all ? 'max(a - b, 0)' : 'a>0 and b=0 ? 1 : 0';
}

function leftFate(op: OpId, all: boolean, a: number, b: number, occ: number): Fate {
  if (op === 'union') return all || occ === 0 ? 'out' : 'dup';
  if (op === 'intersect') {
    if (all) return occ < Math.min(a, b) ? 'out' : 'cut';
    return b > 0 ? (occ === 0 ? 'out' : 'dup') : 'cut';
  }
  if (all) return occ < Math.max(a - b, 0) ? 'out' : 'cut';
  return b === 0 ? (occ === 0 ? 'out' : 'dup') : 'cut';
}

function rightFate(op: OpId, all: boolean, a: number, b: number, occ: number): Fate {
  if (op === 'union') {
    if (all) return 'out';
    return a === 0 && occ === 0 ? 'out' : 'dup';
  }
  if (op === 'intersect') {
    if (all) return occ < Math.min(a, b) ? 'match' : 'cut';
    return a > 0 ? (occ === 0 ? 'match' : 'dup') : 'cut';
  }
  if (all) return occ < Math.min(a, b) ? 'cancel' : 'cut';
  return a > 0 ? (occ === 0 ? 'cancel' : 'dup') : 'cut';
}

export default function SetOpMultisetLab() {
  const [op, setOp] = useState<OpId>('union');
  const [all, setAll] = useState(false);
  const [nulls, setNulls] = useState(true);
  const [ref, width] = useSize(680);
  const tip = useTip();

  const left = nulls ? [...LEFT_BASE, NULLV] : LEFT_BASE;
  const right = nulls ? [...RIGHT_BASE, NULLV] : RIGHT_BASE;

  const countA = new Map<string, number>();
  const countB = new Map<string, number>();
  for (const v of left) countA.set(v, (countA.get(v) ?? 0) + 1);
  for (const v of right) countB.set(v, (countB.get(v) ?? 0) + 1);

  const values: string[] = [];
  for (const v of [...left, ...right]) if (!values.includes(v)) values.push(v);

  const rows = values.map((v) => {
    const a = countA.get(v) ?? 0;
    const b = countB.get(v) ?? 0;
    return { v, a, b, out: multiplicity(op, all, a, b) };
  });

  // Which branch each emitted copy came from: the left branch supplies its `a` copies
  // first, and only UNION ALL ever runs past them into the right branch's rows.
  const result: { v: string; origin: 'left' | 'right' }[] = [];
  for (const r of rows) {
    for (let i = 0; i < r.out; i++) result.push({ v: r.v, origin: i < r.a ? 'left' : 'right' });
  }

  const seenL = new Map<string, number>();
  const leftChips = left.map((v) => {
    const occ = seenL.get(v) ?? 0;
    seenL.set(v, occ + 1);
    return { v, fate: leftFate(op, all, countA.get(v) ?? 0, countB.get(v) ?? 0, occ) };
  });
  const seenR = new Map<string, number>();
  const rightChips = right.map((v) => {
    const occ = seenR.get(v) ?? 0;
    seenR.set(v, occ + 1);
    return { v, fate: rightFate(op, all, countA.get(v) ?? 0, countB.get(v) ?? 0, occ) };
  });

  const plan = PLAN[`${op}-${all ? 'all' : 'set'}`];
  const eliminated = left.length + right.length - result.length;

  /* geometry */
  const laneW = 168;
  const chipH = 22;
  const chipGap = 6;
  const laneTop = 34;
  const maxRows = Math.max(left.length, right.length, result.length, 1);
  const laneH = maxRows * (chipH + chipGap);
  const arithTop = laneTop + laneH + 26;
  const svgW = Math.max(width, laneW * 3 + 24);
  const svgH = arithTop + (rows.length + 1) * 16 + 10;
  const laneX = (i: number) => i * (laneW + 8);

  const chipFill = (origin: 'left' | 'right', fate: Fate) =>
    fate === 'out'
      ? origin === 'left'
        ? 'var(--viz-1)'
        : 'var(--viz-2)'
      : fate === 'cut'
        ? 'var(--viz-plane)'
        : 'var(--viz-neutral)';

  const chipInk = (fate: Fate) => (fate === 'out' ? 'var(--viz-surface)' : 'var(--viz-ink-2)');

  const sql = `SELECT region FROM a\n${op.toUpperCase()}${all ? ' ALL' : ''}\nSELECT region FROM b`;

  return (
    <VizPanel
      title="Set operators are arithmetic on multiplicities"
      subtitle="Two query results, each with duplicates and a NULL. Switch between the set form and the ALL form and watch which input rows are emitted, folded into an earlier copy, or eliminated outright."
      controls={
        <>
          <Segmented label="Operator" value={op} onChange={setOp} options={OPS} />
          <Segmented
            label="Duplicates"
            value={all ? 'all' : 'set'}
            onChange={(v) => setAll(v === 'all')}
            options={[
              { value: 'set', label: 'set form', title: 'UNION / INTERSECT / EXCEPT — every multiplicity clamped to 1' },
              { value: 'all', label: 'ALL', title: 'bag form — multiplicities are arithmetic' },
            ]}
          />
          <Check label="Include a NULL row on each side" checked={nulls} onChange={setNulls} />
        </>
      }
      legend={
        <Legend
          items={[
            { label: 'row from the left query, emitted', color: 'var(--viz-1)' },
            { label: 'row from the right query, emitted', color: 'var(--viz-2)' },
            { label: 'consumed, not emitted — dup / match / cancels', color: 'var(--viz-neutral)' },
            { label: 'drop — eliminated by the operator', color: 'var(--viz-ink-muted)', shape: 'line' },
          ]}
        />
      }
      stats={
        <Stats
          items={[
            { label: 'Left rows', value: left.length },
            { label: 'Right rows', value: right.length },
            { label: 'Result rows', value: result.length },
            { label: 'Rows eliminated', value: eliminated, hint: 'Folded as duplicates or removed by the operator' },
            { label: 'Postgres plan node', value: plan, hint: 'What EXPLAIN prints above the two branches' },
          ]}
        />
      }
      note={
        <Note>
          <strong>
            {op.toUpperCase()}
            {all ? ' ALL' : ''}: {rule(op, all)} copies of each distinct value.
          </strong>{' '}
          {all
            ? 'The ALL form is pure bag arithmetic: nothing is deduplicated, so the engine can stream both branches — UNION ALL is just an Append with no memory footprint at all.'
            : 'The set form has to prove each output row is unique, which means hashing or sorting the whole result. That is the real cost difference between UNION and UNION ALL, and it is why UNION on a large result set shows up as a HashAggregate that spills to disk.'}{' '}
          {nulls
            ? all
              ? 'The two NULL rows are ordinary values to a set operator — it compares with "is not distinct from", so they count as copies of one another rather than as UNKNOWN. The ALL form keeps both copies; switch to the set form to watch them fold into one.'
              : 'The two NULL rows fold into a single output row: set operators compare with "is not distinct from", unlike = anywhere else in the language.'
            : 'Turn the NULL rows back on: they behave as ordinary equal values here, which is the one place SQL lets two NULLs match.'}
        </Note>
      }
      table={
        <table className="viz-table">
          <thead>
            <tr>
              <th>value</th>
              <th>count in left (a)</th>
              <th>count in right (b)</th>
              <th>rule</th>
              <th>count in result</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((r) => (
              <tr key={r.v}>
                <td>{r.v}</td>
                <td>{r.a}</td>
                <td>{r.b}</td>
                <td>{rule(op, all)}</td>
                <td>{r.out}</td>
              </tr>
            ))}
            <tr>
              <td>total</td>
              <td>{left.length}</td>
              <td>{right.length}</td>
              <td>{plan}</td>
              <td>{result.length}</td>
            </tr>
          </tbody>
        </table>
      }
    >
      <div ref={ref}>
        <TooltipHost>
          <svg width={svgW} height={svgH} role="img" aria-label={sql}>
            {['SELECT region FROM a', `${op.toUpperCase()}${all ? ' ALL' : ''}  SELECT region FROM b`, 'result'].map(
              (t, i) => (
                <text key={t} x={laneX(i)} y={14} fill="var(--viz-ink)" fontWeight={600}>
                  {t}
                </text>
              ),
            )}
            <text x={laneX(0)} y={28} fill="var(--viz-ink-muted)">
              {left.length} rows
            </text>
            <text x={laneX(1)} y={28} fill="var(--viz-ink-muted)">
              {right.length} rows
            </text>
            <text x={laneX(2)} y={28} fill="var(--viz-ink-muted)">
              {result.length} rows
            </text>

            {[
              { chips: leftChips, origin: 'left' as const, i: 0 },
              { chips: rightChips, origin: 'right' as const, i: 1 },
            ].map((lane) =>
              lane.chips.map((ch, r) => (
                <g
                  key={`${lane.i}-${r}`}
                  {...tip(
                    <>
                      <strong>{ch.v}</strong> — {TAG[ch.fate]}
                      <br />
                      {HINT[ch.fate]}
                    </>,
                  )}
                >
                  <rect
                    x={laneX(lane.i)}
                    y={laneTop + r * (chipH + chipGap)}
                    width={laneW - 16}
                    height={chipH}
                    rx={5}
                    fill={chipFill(lane.origin, ch.fate)}
                    stroke={ch.fate === 'cut' ? 'var(--viz-border)' : 'var(--viz-surface)'}
                    strokeWidth={1.5}
                    strokeDasharray={ch.fate === 'cut' ? '3 3' : undefined}
                  />
                  <text x={laneX(lane.i) + 8} y={laneTop + r * (chipH + chipGap) + 15} fill={chipInk(ch.fate)}>
                    {ch.v}
                  </text>
                  <text
                    x={laneX(lane.i) + laneW - 24}
                    y={laneTop + r * (chipH + chipGap) + 15}
                    textAnchor="end"
                    fill={ch.fate === 'out' ? 'var(--viz-surface)' : 'var(--viz-ink-muted)'}
                  >
                    {TAG[ch.fate]}
                  </text>
                </g>
              )),
            )}

            {result.length === 0 ? (
              <text x={laneX(2)} y={laneTop + 15} fill="var(--viz-ink-muted)">
                no rows
              </text>
            ) : (
              result.map((r, i) => (
                <g
                  key={`res-${i}`}
                  {...tip(
                    <>
                      <strong>{r.v}</strong>
                      <br />
                      Emitted from the {r.origin} branch. Multiplicity in the result:{' '}
                      {rows.find((x) => x.v === r.v)!.out}.
                    </>,
                  )}
                >
                  <rect
                    x={laneX(2)}
                    y={laneTop + i * (chipH + chipGap)}
                    width={laneW - 16}
                    height={chipH}
                    rx={5}
                    fill={r.origin === 'left' ? 'var(--viz-1)' : 'var(--viz-2)'}
                    stroke="var(--viz-surface)"
                    strokeWidth={1.5}
                  />
                  <text x={laneX(2) + 8} y={laneTop + i * (chipH + chipGap) + 15} fill="var(--viz-surface)">
                    {r.v}
                  </text>
                </g>
              ))
            )}

            {/* the multiset arithmetic, value by value */}
            <text x={0} y={arithTop} fill="var(--viz-ink)" fontWeight={600}>
              value
            </text>
            <text x={110} y={arithTop} fill="var(--viz-ink)" fontWeight={600}>
              a
            </text>
            <text x={140} y={arithTop} fill="var(--viz-ink)" fontWeight={600}>
              b
            </text>
            <text x={180} y={arithTop} fill="var(--viz-ink)" fontWeight={600}>
              {rule(op, all)}
            </text>
            {rows.map((r, i) => (
              <g key={`ar-${r.v}`}>
                <text x={0} y={arithTop + (i + 1) * 16} fill="var(--viz-ink-2)">
                  {r.v}
                </text>
                <text x={110} y={arithTop + (i + 1) * 16} fill="var(--viz-ink-2)">
                  {r.a}
                </text>
                <text x={140} y={arithTop + (i + 1) * 16} fill="var(--viz-ink-2)">
                  {r.b}
                </text>
                <text
                  x={180}
                  y={arithTop + (i + 1) * 16}
                  fill={r.out === 0 ? 'var(--viz-ink-muted)' : 'var(--viz-ink)'}
                  fontWeight={r.out === 0 ? 400 : 600}
                >
                  {r.out} {r.out === 1 ? 'copy' : 'copies'} in the result
                </text>
              </g>
            ))}
          </svg>
        </TooltipHost>
      </div>
    </VizPanel>
  );
}
