import { useMemo, useState } from 'react';
import {
  VizPanel,
  Choice,
  Button,
  Legend,
  Stats,
  Note,
  TooltipHost,
  useTip,
  fmtNum,
  useSize,
} from './Viz';

/**
 * The chase, run one rule application at a time.
 *
 * The lossless-join test for a decomposition {R1..Rn} of R under a set of dependencies is a
 * tableau: one row per fragment, a distinguished symbol a_k in the columns that fragment
 * keeps and a placeholder b_ik everywhere else. Then apply the dependencies as rewrite
 * rules — an FD equates two symbols, an MVD or a JD adds a row — until nothing changes.
 * The decomposition is lossless exactly when some row becomes all-a.
 *
 * When it gets stuck instead, the final tableau IS the counterexample: read it as a relation
 * over the symbols and it satisfies every dependency while its projections rejoin to more
 * rows than it contains.
 */

type Dep =
  | { kind: 'fd'; lhs: number[]; rhs: number[]; text: string }
  | { kind: 'mvd'; lhs: number[]; rhs: number[]; text: string }
  | { kind: 'jd'; comps: number[][]; text: string };

type Scenario = {
  id: string;
  label: string;
  attrs: string[];
  frags: { name: string; cols: number[] }[];
  deps: Dep[];
  blurb: string;
  moral: string;
};

const SCENARIOS: Scenario[] = [
  {
    id: 'fd-ok',
    label: 'FD: order → customer (lossless)',
    attrs: ['order_id', 'item', 'customer'],
    frags: [
      { name: 'order_item', cols: [0, 1] },
      { name: 'order_customer', cols: [0, 2] },
    ],
    deps: [{ kind: 'fd', lhs: [0], rhs: [2], text: 'order_id → customer' }],
    blurb:
      'The two fragments share order_id, and order_id functionally determines customer — the classic binary test: R1 ∩ R2 → R1 − R2.',
    moral:
      'The FD rule fired once: two rows agreed on order_id, so their customer symbols had to be equal, and a distinguished a beats a placeholder b. One row went all-a — the join is lossless.',
  },
  {
    id: 'fd-bad',
    label: 'FD: item → customer (lossy)',
    attrs: ['order_id', 'item', 'customer'],
    frags: [
      { name: 'order_item', cols: [0, 1] },
      { name: 'order_customer', cols: [0, 2] },
    ],
    deps: [{ kind: 'fd', lhs: [1], rhs: [2], text: 'item → customer' }],
    blurb:
      'Same decomposition, a different dependency. item → customer is a perfectly real constraint and it is useless here: the shared column is order_id.',
    moral:
      'No two rows ever agree on item, so the FD rule never fires and the tableau is already at its fixpoint. Read the stuck tableau as data: one order with two items and two customers rejoins to four rows. Losslessness is a property of the intersection, not of the columns you like.',
  },
  {
    id: 'mvd',
    label: 'MVD: emp ↠ skill (lossless, 4NF)',
    attrs: ['emp', 'skill', 'phone'],
    frags: [
      { name: 'employee_skill', cols: [0, 1] },
      { name: 'employee_phone', cols: [0, 2] },
    ],
    deps: [{ kind: 'mvd', lhs: [0], rhs: [1], text: 'emp ↠ skill' }],
    blurb:
      'No FD holds here at all — the only key is the whole heading — yet the split is lossless. Fagin (1977): a binary decomposition is lossless iff R1 ∩ R2 ↠ R1 − R2.',
    moral:
      'The MVD rule swaps the tails of two rows that agree on emp, and the swapped row is all-a. That is exactly the "completion" tuple the MVD promises exists, which is why the projections can be rejoined without inventing anything.',
  },
  {
    id: 'jd-3',
    label: 'JD: supplier/part/project, three-way (lossless, 5NF)',
    attrs: ['supplier', 'part', 'project'],
    frags: [
      { name: 'supplies', cols: [0, 1] },
      { name: 'part_used_by', cols: [1, 2] },
      { name: 'works_on', cols: [2, 0] },
    ],
    deps: [
      {
        kind: 'jd',
        comps: [
          [0, 1],
          [1, 2],
          [2, 0],
        ],
        text: '⋈[ (supplier,part), (part,project), (project,supplier) ]',
      },
    ],
    blurb:
      'The cyclic rule: if s supplies p, p is used by j, and s works on j, then s supplies p to j. That is a join dependency no pair of columns can express.',
    moral:
      'The JD rule takes one row per component, checks they agree on every shared column, and glues their projections into a new row — here an all-a row. A three-way split is lossless; every binary split of the same relation is not.',
  },
  {
    id: 'jd-2',
    label: 'JD: same relation, split in two (lossy)',
    attrs: ['supplier', 'part', 'project'],
    frags: [
      { name: 'supplies', cols: [0, 1] },
      { name: 'part_used_by', cols: [1, 2] },
    ],
    deps: [
      {
        kind: 'jd',
        comps: [
          [0, 1],
          [1, 2],
          [2, 0],
        ],
        text: '⋈[ (supplier,part), (part,project), (project,supplier) ]',
      },
    ],
    blurb:
      'Drop the third fragment and keep the same join dependency. The JD rule needs a row for every component, and the (project, supplier) component has none.',
    moral:
      'Stuck at two rows. This is what "5NF is not reachable by repeated binary decomposition" means concretely: a relation can be in 4NF, carry a genuine join dependency, and only a three-way projection reconstructs it.',
  },
];

/* ---------------------------------------------------------------- the chase */

type Cell = string; // 'a2' or 'b13'
type Row = Cell[];

type Change = { rows: Row[]; cells: string[]; newRow: number | null; rule: string; detail: string };

const isA = (c: Cell) => c.startsWith('a');

function initialTableau(sc: Scenario): Row[] {
  return sc.frags.map((f, i) => sc.attrs.map((_, k) => (f.cols.includes(k) ? `a${k + 1}` : `b${i + 1}${k + 1}`)));
}

const rowKey = (r: Row) => r.join('');

/** One rule application, chosen deterministically: FDs first, then MVDs, then JDs. */
function chaseStep(sc: Scenario, rows: Row[]): Change | null {
  // --- FD: two rows agreeing on X must agree on Y. Equate the symbols everywhere.
  for (const d of sc.deps) {
    if (d.kind !== 'fd') continue;
    for (let i = 0; i < rows.length; i++) {
      for (let j = i + 1; j < rows.length; j++) {
        if (!d.lhs.every((k) => rows[i][k] === rows[j][k])) continue;
        for (const k of d.rhs) {
          const s1 = rows[i][k];
          const s2 = rows[j][k];
          if (s1 === s2) continue;
          const winner = isA(s1) ? s1 : isA(s2) ? s2 : s1 < s2 ? s1 : s2;
          const loser = winner === s1 ? s2 : s1;
          const cells: string[] = [];
          const next = rows.map((r, ri) =>
            r.map((c, ci) => {
              if (c !== loser) return c;
              cells.push(`${ri},${ci}`);
              return winner;
            }),
          );
          return {
            rows: next,
            cells,
            newRow: null,
            rule: `FD ${d.text}`,
            detail: `Rows t${i + 1} and t${j + 1} agree on ${d.lhs
              .map((x) => sc.attrs[x])
              .join(', ')}, so the dependency forces their ${sc.attrs[k]} values to be the same symbol. ${loser} is replaced by ${winner} everywhere — a distinguished a always wins, because a placeholder is only a name for "some value".`,
          };
        }
      }
    }
  }

  // --- MVD: X ->> Y. Two rows agreeing on X imply the row that takes Y from one and the rest from the other.
  for (const d of sc.deps) {
    if (d.kind !== 'mvd') continue;
    const rest = sc.attrs.map((_, k) => k).filter((k) => !d.lhs.includes(k) && !d.rhs.includes(k));
    const seen = new Set(rows.map(rowKey));
    for (let i = 0; i < rows.length; i++) {
      for (let j = 0; j < rows.length; j++) {
        if (i === j) continue;
        if (!d.lhs.every((k) => rows[i][k] === rows[j][k])) continue;
        const w = sc.attrs.map((_, k) => (rest.includes(k) ? rows[j][k] : rows[i][k]));
        if (seen.has(rowKey(w))) continue;
        return {
          rows: [...rows, w],
          cells: [],
          newRow: rows.length,
          rule: `MVD ${d.text}`,
          detail: `t${i + 1} and t${j + 1} agree on ${d.lhs
            .map((x) => sc.attrs[x])
            .join(', ')}. The MVD says the tails are independent, so the relation must also contain the row that takes ${d.rhs
            .map((x) => sc.attrs[x])
            .join(', ')} from t${i + 1} and ${rest.map((x) => sc.attrs[x]).join(', ')} from t${j + 1}. Added as t${rows.length + 1}.`,
        };
      }
    }
  }

  // --- JD: one row per component, agreeing on every shared column, glue the projections.
  for (const d of sc.deps) {
    if (d.kind !== 'jd') continue;
    const n = d.comps.length;
    const seen = new Set(rows.map(rowKey));
    const pick: number[] = [];
    const walk = (): Change | null => {
      if (pick.length === n) {
        const w = sc.attrs.map((_, k) => {
          const ci = d.comps.findIndex((c) => c.includes(k));
          return ci >= 0 ? rows[pick[ci]][k] : rows[pick[0]][k];
        });
        for (let ci = 0; ci < n; ci++) {
          for (const k of d.comps[ci]) if (rows[pick[ci]][k] !== w[k]) return null;
        }
        if (seen.has(rowKey(w))) return null;
        return {
          rows: [...rows, w],
          cells: [],
          newRow: rows.length,
          rule: `JD ${d.text}`,
          detail: `Take ${pick
            .map((p, ci) => `(${d.comps[ci].map((x) => sc.attrs[x]).join(', ')}) from t${p + 1}`)
            .join(', ')}. They agree on every shared column, so the join dependency says the relation must also contain the row those projections combine into. Added as t${rows.length + 1}.`,
        };
      }
      for (let r = 0; r < rows.length; r++) {
        pick.push(r);
        const got = walk();
        pick.pop();
        if (got) return got;
      }
      return null;
    };
    const got = walk();
    if (got) return got;
  }

  return null;
}

const allA = (rows: Row[]) => rows.findIndex((r) => r.every(isA));

/* ------------------------------------------------------------- the component */

type LogRow = { n: number; rule: string; rows: number; detail: string };

export default function ChaseTableauStepper() {
  const [scId, setScId] = useState(SCENARIOS[2].id);
  const sc = SCENARIOS.find((s) => s.id === scId)!;

  const [rows, setRows] = useState<Row[]>(() => initialTableau(sc));
  const [log, setLog] = useState<LogRow[]>([]);
  const [hi, setHi] = useState<{ cells: string[]; newRow: number | null }>({ cells: [], newRow: null });
  const [stuck, setStuck] = useState(false);
  const [ref, width] = useSize(720);
  const tip = useTip();

  const winner = allA(rows);
  const finished = winner >= 0 || stuck;

  const reset = (next: Scenario) => {
    setRows(initialTableau(next));
    setLog([]);
    setHi({ cells: [], newRow: null });
    setStuck(false);
  };

  const step = () => {
    if (finished) return;
    const ch = chaseStep(sc, rows);
    if (!ch) {
      setStuck(true);
      setHi({ cells: [], newRow: null });
      setLog((l) => [
        ...l,
        { n: l.length + 1, rule: 'fixpoint', rows: rows.length, detail: 'No dependency applies to any pair of rows. The chase has terminated without an all-a row.' },
      ]);
      return;
    }
    setRows(ch.rows);
    setHi({ cells: ch.cells, newRow: ch.newRow });
    setLog((l) => [...l, { n: l.length + 1, rule: ch.rule, rows: ch.rows.length, detail: ch.detail }]);
  };

  const runAll = () => {
    let cur = rows;
    const entries: LogRow[] = [...log];
    let last: Change | null = null;
    for (let guard = 0; guard < 24; guard++) {
      if (allA(cur) >= 0) break;
      const ch = chaseStep(sc, cur);
      if (!ch) {
        entries.push({ n: entries.length + 1, rule: 'fixpoint', rows: cur.length, detail: 'No dependency applies. Terminated with placeholders still in every row.' });
        setStuck(true);
        break;
      }
      cur = ch.rows;
      last = ch;
      entries.push({ n: entries.length + 1, rule: ch.rule, rows: ch.rows.length, detail: ch.detail });
    }
    setRows(cur);
    setLog(entries);
    setHi(last ? { cells: last.cells, newRow: last.newRow } : { cells: [], newRow: null });
  };

  const placeholders = useMemo(() => new Set(rows.flat().filter((c) => !isA(c))).size, [rows]);

  /* geometry */
  const gutter = 136;
  const cellW = Math.max(88, Math.min(140, Math.floor((width - gutter - 40) / sc.attrs.length)));
  const rowH = 26;
  const headH = 26;
  const svgW = Math.max(width, gutter + sc.attrs.length * cellW + 30);
  const height = headH + rows.length * rowH + 14;

  const last = log.length ? log[log.length - 1] : null;

  return (
    <VizPanel
      title="The chase, one rule application at a time"
      subtitle="One tableau row per fragment: a distinguished aₖ where the fragment keeps the column, a placeholder bᵢₖ where it does not. Apply the dependencies until nothing changes. An all-a row means the join is lossless."
      controls={
        <>
          <Choice
            label="Decomposition"
            value={scId}
            onChange={(v) => {
              setScId(v);
              reset(SCENARIOS.find((s) => s.id === v)!);
            }}
            options={SCENARIOS.map((s) => ({ value: s.id, label: s.label }))}
          />
          <Button onClick={step} disabled={finished} primary>
            Chase step
          </Button>
          <Button onClick={runAll} disabled={finished}>
            Run to fixpoint
          </Button>
          <Button onClick={() => reset(sc)}>Reset</Button>
        </>
      }
      legend={
        <Legend
          items={[
            { label: 'aₖ — distinguished symbol', color: 'var(--viz-1)' },
            { label: 'bᵢₖ — placeholder ("some value")', color: 'var(--viz-ink-muted)' },
            { label: 'changed or added by the last step', color: 'var(--viz-2)' },
            { label: '✓ all-a row — decomposition is lossless', color: 'var(--viz-good)' },
          ]}
        />
      }
      stats={
        <Stats
          items={[
            { label: 'Fragments', value: fmtNum(sc.frags.length), hint: 'One tableau row per projection in the decomposition' },
            { label: 'Tableau rows', value: fmtNum(rows.length), hint: 'MVD and JD rules add rows; FD rules only equate symbols' },
            { label: 'Steps applied', value: fmtNum(log.filter((l) => l.rule !== 'fixpoint').length) },
            { label: 'Distinct placeholders left', value: fmtNum(placeholders), hint: 'Every b the chase kills is one column the join can pin down' },
            {
              label: 'Verdict',
              value: winner >= 0 ? 'lossless' : stuck ? 'lossy' : 'running',
              hint: winner >= 0 ? `t${winner + 1} is all-a` : stuck ? 'fixpoint reached with no all-a row' : 'keep chasing',
            },
          ]}
        />
      }
      note={
        <Note>
          {winner >= 0 ? (
            <>
              <strong>All-a row at t{winner + 1}: the decomposition is lossless.</strong> {sc.moral}
            </>
          ) : stuck ? (
            <>
              <strong>Fixpoint with no all-a row: the decomposition is lossy.</strong> {sc.moral} The tableau you are
              looking at is the counterexample — treat each symbol as a value and it satisfies every dependency listed
              below while its projections rejoin to strictly more rows.
            </>
          ) : last ? (
            <>
              <strong>{last.rule}.</strong> {last.detail}
            </>
          ) : (
            <>
              <strong>{sc.blurb}</strong> Press “Chase step” to apply the first dependency that changes anything.
            </>
          )}
        </Note>
      }
      table={
        <>
          <table className="viz-table">
            <thead>
              <tr>
                <th>row</th>
                <th>from</th>
                {sc.attrs.map((a) => (
                  <th key={a}>{a}</th>
                ))}
                <th>all-a?</th>
              </tr>
            </thead>
            <tbody>
              {rows.map((r, i) => (
                <tr key={`t${i}-${rowKey(r)}`}>
                  <td>t{i + 1}</td>
                  <td>{sc.frags[i]?.name ?? 'derived by the chase'}</td>
                  {r.map((c, k) => (
                    <td key={k}>{c}</td>
                  ))}
                  <td>{r.every(isA) ? 'yes' : 'no'}</td>
                </tr>
              ))}
            </tbody>
          </table>
          <table className="viz-table">
            <thead>
              <tr>
                <th>#</th>
                <th>Rule applied</th>
                <th>Rows after</th>
                <th>What it did</th>
              </tr>
            </thead>
            <tbody>
              {log.length === 0 ? (
                <tr>
                  <td colSpan={4}>No step taken yet.</td>
                </tr>
              ) : (
                log.map((l) => (
                  <tr key={l.n}>
                    <td>{l.n}</td>
                    <td>{l.rule}</td>
                    <td>{l.rows}</td>
                    <td>{l.detail}</td>
                  </tr>
                ))
              )}
            </tbody>
          </table>
          <table className="viz-table">
            <thead>
              <tr>
                <th>Dependency set being chased</th>
              </tr>
            </thead>
            <tbody>
              {sc.deps.length === 0 ? (
                <tr>
                  <td>none</td>
                </tr>
              ) : (
                sc.deps.map((d) => (
                  <tr key={d.text}>
                    <td>
                      {d.kind.toUpperCase()} — {d.text}
                    </td>
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
          <svg width={svgW} height={height} role="img" aria-label="Chase tableau with one row per fragment of the decomposition">
            {sc.attrs.map((a, k) => (
              <text key={a} x={gutter + k * cellW + cellW / 2} y={14} textAnchor="middle" fill="var(--viz-ink-2)" fontWeight={600}>
                {a}
              </text>
            ))}
            {rows.map((r, i) => {
              const y = headH + i * rowH;
              const isWin = r.every(isA);
              const isNew = hi.newRow === i;
              return (
                <g key={`row${i}`}>
                  <rect
                    x={gutter - 6}
                    y={y}
                    width={sc.attrs.length * cellW + 12}
                    height={rowH - 4}
                    rx={6}
                    fill="var(--viz-plane)"
                    stroke={isWin ? 'var(--viz-good)' : isNew ? 'var(--viz-2)' : 'var(--viz-border)'}
                    strokeWidth={isWin || isNew ? 2 : 1}
                    strokeDasharray={isNew && !isWin ? '5 3' : undefined}
                  />
                  <text x={0} y={y + 16} fill={isWin ? 'var(--viz-good)' : 'var(--viz-ink-2)'} fontWeight={isWin ? 700 : 400}>
                    {isWin ? '✓ ' : ''}t{i + 1}
                    {sc.frags[i] ? ` · ${sc.frags[i].name}` : ' · derived'}
                  </text>
                  {r.map((c, k) => {
                    const changed = hi.cells.includes(`${i},${k}`);
                    return (
                      <g
                        key={k}
                        {...tip(
                          <>
                            <strong>
                              t{i + 1}.{sc.attrs[k]} = {c}
                            </strong>
                            <br />
                            {isA(c)
                              ? 'A distinguished symbol: this fragment keeps the column, or the chase proved the value is pinned down.'
                              : 'A placeholder: the fragment dropped this column, so the join can only say "some value here".'}
                          </>,
                        )}
                        style={{ cursor: 'help' }}
                      >
                        <rect
                          x={gutter + k * cellW + 4}
                          y={y + 3}
                          width={cellW - 8}
                          height={rowH - 10}
                          rx={4}
                          fill="var(--viz-surface)"
                          stroke={changed ? 'var(--viz-2)' : 'transparent'}
                          strokeWidth={2}
                        />
                        <text
                          x={gutter + k * cellW + cellW / 2}
                          y={y + 16}
                          textAnchor="middle"
                          fill={changed ? 'var(--viz-2)' : isA(c) ? 'var(--viz-1)' : 'var(--viz-ink-muted)'}
                          fontWeight={isA(c) ? 700 : 400}
                        >
                          {c}
                        </text>
                      </g>
                    );
                  })}
                </g>
              );
            })}
          </svg>
        </TooltipHost>
      </div>
    </VizPanel>
  );
}
