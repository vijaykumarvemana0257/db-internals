import { useMemo, useState } from 'react';
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
  fmtNum,
  useSize,
} from './Viz';

/**
 * Multivalued dependencies as row explosion, and 4NF as the fix.
 *
 * One employee has two INDEPENDENT one-to-many facts — skills and phone numbers. Forcing
 * both into a single relation means the relation can only satisfy emp ->> skill (and its
 * complement emp ->> phone) by storing the full cross product: every skill paired with
 * every phone. BCNF cannot see this, because no FD is violated: the only key is the whole
 * heading.
 *
 * Turn the MVD off and the flat table holds only the facts somebody actually typed; the
 * projections onto (emp, skill) and (emp, phone) then rejoin to MORE rows than you stored.
 * That gap is Fagin's theorem made visible: the binary decomposition is lossless exactly
 * when the MVD holds.
 */

const SKILLS = ['SQL tuning', 'Kafka', 'Terraform', 'Go', 'Incident cmd'];
const PHONE_A = ['+1 206 555 0118', '+1 206 555 0142', '+1 425 555 0193', '+1 360 555 0107'];
const PHONE_ALT = '+1 564 555 0771'; // what "change my number" writes over slot 0

const E1 = 'a.okafor';
const E2 = 'r.mehta';
const E2_SKILLS = ['SQL tuning', 'Go'];
const E2_PHONES = ['+1 206 555 0155'];

type Op = 'rekey' | 'addskill' | 'droppost';

const OPS: readonly { value: Op; label: string; title: string }[] = [
  { value: 'rekey', label: 'change a phone number', title: 'UPDATE the first phone number of a.okafor' },
  { value: 'addskill', label: 'add a skill', title: 'INSERT one new skill for a.okafor' },
  { value: 'droppost', label: 'drop a phone', title: 'DELETE the last phone number of a.okafor' },
];

type Row = { emp: string; skill: string; phone: string };

type LogRow = { n: number; op: string; flat: number; nf: number; note: string };

/* ----------------------------------------------------------------- the data */

function phonesOf(m: number, swapped: boolean) {
  const p = PHONE_A.slice(0, m);
  return swapped ? [PHONE_ALT, ...p.slice(1)] : p;
}

/** The rows a flat EMP table actually holds. */
function flatRows(skills: string[], phones: string[], mvd: boolean): Row[] {
  const out: Row[] = [];
  if (mvd) {
    // The MVD forces the completion: every skill with every phone.
    for (const s of skills) for (const p of phones) out.push({ emp: E1, skill: s, phone: p });
  } else {
    // Only what a person typed: each fact recorded once, paired arbitrarily.
    const n = Math.max(skills.length, phones.length);
    for (let k = 0; k < n; k++) out.push({ emp: E1, skill: skills[k % skills.length], phone: phones[k % phones.length] });
  }
  for (const s of E2_SKILLS) for (const p of E2_PHONES) out.push({ emp: E2, skill: s, phone: p });
  return out;
}

/** π(emp, skill) ⋈ π(emp, phone) — what the decomposition can reconstruct. */
function joinRows(skills: string[], phones: string[]): Row[] {
  const out: Row[] = [];
  for (const s of skills) for (const p of phones) out.push({ emp: E1, skill: s, phone: p });
  for (const s of E2_SKILLS) for (const p of E2_PHONES) out.push({ emp: E2, skill: s, phone: p });
  return out;
}

const key = (r: Row) => `${r.emp}|${r.skill}|${r.phone}`;

/** Rows each schema must write for one logical change. */
function cost(op: Op, skills: string[], phones: string[], mvd: boolean, flat: Row[]) {
  const m = phones.length;
  switch (op) {
    case 'rekey': {
      const target = phones[0];
      const touched = flat.filter((r) => r.emp === E1 && r.phone === target).length;
      return { flat: touched, nf: 1, verb: 'UPDATE' };
    }
    case 'addskill':
      return { flat: mvd ? m : 1, nf: 1, verb: 'INSERT' };
    case 'droppost': {
      const target = phones[m - 1];
      const touched = flat.filter((r) => r.emp === E1 && r.phone === target).length;
      return { flat: m > 1 ? touched : 0, nf: m > 1 ? 1 : 0, verb: 'DELETE' };
    }
    default:
      return { flat: 0, nf: 0, verb: '' };
  }
}

/* -------------------------------------------------------------- the drawing */

type Panel = {
  name: string;
  sub: string;
  cols: string[];
  rows: { cells: string[]; tone: 'fact' | 'forced' | 'spurious'; hint: string }[];
  stored: boolean;
};

const TONE: Record<'fact' | 'forced' | 'spurious', { fill: string; glyph: string }> = {
  fact: { fill: 'var(--viz-1)', glyph: '' },
  forced: { fill: 'var(--viz-2)', glyph: '×' },
  spurious: { fill: 'var(--viz-critical)', glyph: '!' },
};

export default function MvdCrossProductLab() {
  const [nSkills, setNSkills] = useState(3);
  const [nPhones, setNPhones] = useState(3);
  const [mvd, setMvd] = useState(true);
  const [schema, setSchema] = useState<'flat' | 'fourNF'>('flat');
  const [op, setOp] = useState<Op>('rekey');
  const [swapped, setSwapped] = useState(false);
  const [log, setLog] = useState<LogRow[]>([]);
  const [ref, width] = useSize(880);
  const tip = useTip();

  const skills = SKILLS.slice(0, nSkills);
  const phones = phonesOf(nPhones, swapped);

  const flat = useMemo(() => flatRows(skills, phones, mvd), [nSkills, nPhones, mvd, swapped]);
  const join = useMemo(() => joinRows(skills, phones), [nSkills, nPhones, swapped]);
  const flatKeys = useMemo(() => new Set(flat.map(key)), [flat]);

  // The "fact-bearing" rows: one per distinct skill and one per distinct phone is all the
  // information there is. Everything past that is combinatorial padding.
  const factRows = Math.max(nSkills, nPhones) + E2_SKILLS.length * E2_PHONES.length;
  const spurious = join.filter((r) => !flatKeys.has(key(r))).length;
  const esRows = nSkills + E2_SKILLS.length;
  const epRows = nPhones + E2_PHONES.length;
  const nfRows = esRows + epRows;

  const c = cost(op, skills, phones, mvd, flat);
  const totals = log.reduce((a, r) => ({ flat: a.flat + r.flat, nf: a.nf + r.nf }), { flat: 0, nf: 0 });

  const apply = () => {
    const opLabel = OPS.find((o) => o.value === op)!.label;
    let note = '';
    if (op === 'rekey') {
      note = mvd
        ? `Every one of a.okafor's ${nSkills} skill rows repeats that phone number, so all ${c.flat} must change together.`
        : `Only ${c.flat} row(s) carry that number here — because this table is already missing combinations.`;
      setSwapped((s) => !s);
    } else if (op === 'addskill') {
      note = mvd
        ? `One new skill, ${c.flat} inserted rows — one per phone number, or the MVD breaks.`
        : 'One row inserted. The table now says less than the projections do.';
      setNSkills((n) => Math.min(SKILLS.length, n + 1));
    } else {
      note = `Dropping one phone number deletes ${c.flat} row(s) from the flat table.`;
      setNPhones((n) => Math.max(1, n - 1));
    }
    setLog((l) => [...l, { n: l.length + 1, op: `${c.verb} — ${opLabel}`, flat: c.flat, nf: c.nf, note }]);
  };

  const panels: Panel[] = [
    {
      name: 'employee',
      sub: mvd ? 'flat, MVD satisfied' : 'flat, MVD violated',
      cols: ['emp', 'skill', 'phone'],
      rows: flat.map((r) => {
        const forced = mvd && !isFactRow(r, skills, phones);
        return {
          cells: [r.emp, r.skill, r.phone],
          tone: forced ? ('forced' as const) : ('fact' as const),
          hint: forced
            ? 'Stored only because the MVD demands it: this pairing asserts nothing that the skill row and the phone row did not already say.'
            : 'Carries a fact somebody actually entered.',
        };
      }),
      stored: schema === 'flat',
    },
    {
      name: 'employee_skill + employee_phone',
      sub: '4NF projections',
      cols: ['emp', 'value', 'table'],
      rows: [
        ...skills.map((s) => ({ cells: [E1, s, 'skill'], tone: 'fact' as const, hint: 'Row of π(emp, skill) — one per skill, whatever the phone list does.' })),
        ...E2_SKILLS.map((s) => ({ cells: [E2, s, 'skill'], tone: 'fact' as const, hint: 'Row of π(emp, skill).' })),
        ...phones.map((p) => ({ cells: [E1, p, 'phone'], tone: 'fact' as const, hint: 'Row of π(emp, phone) — one per number, whatever the skill list does.' })),
        ...E2_PHONES.map((p) => ({ cells: [E2, p, 'phone'], tone: 'fact' as const, hint: 'Row of π(emp, phone).' })),
      ],
      stored: schema === 'fourNF',
    },
    {
      name: 'natural join of the two projections',
      sub: spurious === 0 ? 'reconstructs the original exactly' : `${spurious} spurious tuple(s)`,
      cols: ['emp', 'skill', 'phone'],
      rows: join.map((r) => {
        const bad = !flatKeys.has(key(r));
        return {
          cells: [r.emp, r.skill, r.phone],
          tone: bad ? ('spurious' as const) : ('fact' as const),
          hint: bad
            ? 'Not in the original table. The decomposition invented it — the join is lossy because the MVD does not hold.'
            : 'Present in the original table too.',
        };
      }),
      stored: false,
    },
  ];

  /* geometry */
  const colW = [92, 110, 126];
  const panelW = (p: Panel) => (p.cols.length === 3 ? colW[0] + colW[1] + colW[2] : colW[0] + colW[2]) + 16;
  const gap = 18;
  const rowH = 17;
  const headH = 40;
  const maxRows = Math.max(...panels.map((p) => p.rows.length));
  const height = headH + maxRows * rowH + 34;
  const svgW = Math.max(width, panels.reduce((a, p) => a + panelW(p), 0) + gap * 2 + 8);

  let x = 0;
  const placed = panels.map((p) => {
    const at = x;
    x += panelW(p) + gap;
    return { p, x: at };
  });

  const redundancy = nfRows > 0 ? flat.length / nfRows : 0;

  return (
    <VizPanel
      title="One employee, two independent lists"
      subtitle="skill and phone are independent multivalued facts about emp. Keep them in one table and the MVD forces the cross product; split them and the join has to give the original back exactly."
      controls={
        <>
          <Slider label="Skills" min={1} max={SKILLS.length} value={nSkills} onChange={setNSkills} />
          <Slider label="Phone numbers" min={1} max={PHONE_A.length} value={nPhones} onChange={setNPhones} />
          <Check
            label="MVD emp ↠ skill holds"
            checked={mvd}
            onChange={(b) => setMvd(b)}
          />
          <Segmented
            label="Stored schema"
            value={schema}
            onChange={setSchema}
            options={[
              { value: 'flat', label: 'one table', title: 'employee(emp, skill, phone) — BCNF, not 4NF' },
              { value: 'fourNF', label: '4NF pair', title: 'employee_skill + employee_phone' },
            ]}
          />
          <Segmented label="Operation" value={op} onChange={setOp} options={OPS} />
          <Button
            onClick={apply}
            primary
            disabled={(op === 'droppost' && nPhones === 1) || (op === 'addskill' && nSkills === SKILLS.length)}
          >
            Apply
          </Button>
          <Button
            onClick={() => {
              setNSkills(3);
              setNPhones(3);
              setMvd(true);
              setSwapped(false);
              setLog([]);
            }}
          >
            Reset
          </Button>
        </>
      }
      legend={
        <Legend
          items={[
            { label: 'row carrying a fact', color: TONE.fact.fill },
            { label: '× forced by the MVD (pure redundancy)', color: TONE.forced.fill },
            { label: '! spurious tuple the join invented', color: TONE.spurious.fill },
            { label: 'stored schema (solid frame)', color: 'var(--viz-ink-2)', shape: 'line' },
          ]}
        />
      }
      stats={
        <Stats
          items={[
            { label: 'Rows, one table', value: fmtNum(flat.length), hint: 'skills × phones per employee when the MVD holds' },
            { label: 'Rows, 4NF pair', value: fmtNum(nfRows), hint: 'skills + phones per employee' },
            { label: 'Redundancy', value: `${redundancy.toFixed(2)}×`, hint: 'stored rows in the flat table per stored row in the 4NF pair' },
            {
              label: 'Spurious on rejoin',
              value: fmtNum(spurious),
              hint: 'Tuples in π(emp,skill) ⋈ π(emp,phone) that were never in the table. Zero exactly when the MVD holds — Fagin, 1977.',
            },
            { label: `Last ${c.verb || 'op'} writes`, value: `${fmtNum(c.flat)} vs ${fmtNum(c.nf)}`, hint: 'rows the flat table must rewrite vs rows the 4NF pair must rewrite' },
            { label: 'Rows written so far', value: `${fmtNum(totals.flat)} vs ${fmtNum(totals.nf)}`, hint: 'cumulative across the operations you applied' },
          ]}
        />
      }
      note={
        <Note>
          {mvd ? (
            <>
              <strong>
                The MVD holds, so the flat table stores {fmtNum(flat.length)} rows to say what{' '}
                {fmtNum(factRows)} rows' worth of facts.
              </strong>{' '}
              Nothing is wrong by BCNF's standards — the only key of employee(emp, skill, phone) is all three
              columns, so no FD is violated. What is wrong is that {c.verb === 'UPDATE' ? 'changing' : 'touching'} one
              phone number now means rewriting {fmtNum(cost('rekey', skills, phones, mvd, flat).flat)} rows atomically;
              miss one and the MVD breaks. The join panel shows the 4NF pair reconstructing the original exactly.
            </>
          ) : (
            <>
              <strong>The MVD does not hold — and now the decomposition is lossy.</strong> The flat table records{' '}
              {fmtNum(flat.length)} rows, but π(emp, skill) ⋈ π(emp, phone) returns {fmtNum(join.length)}:{' '}
              {fmtNum(spurious)} tuple(s) that were never stored, each one a plausible-looking claim that this person
              has that skill and that number. Losslessness of a binary split is not a property of the columns you
              chose; it is the MVD on the intersection.
            </>
          )}
        </Note>
      }
      table={
        <>
          <table className="viz-table">
            <thead>
              <tr>
                <th>emp</th>
                <th>skill</th>
                <th>phone</th>
                <th>in the flat table</th>
                <th>in the rejoin</th>
                <th>classification</th>
              </tr>
            </thead>
            <tbody>
              {join.map((r) => {
                const inFlat = flatKeys.has(key(r));
                const fact = isFactRow(r, skills, phones) || r.emp === E2;
                return (
                  <tr key={key(r)}>
                    <td>{r.emp}</td>
                    <td>{r.skill}</td>
                    <td>{r.phone}</td>
                    <td>{inFlat ? 'yes' : 'no'}</td>
                    <td>yes</td>
                    <td>{!inFlat ? 'spurious' : fact ? 'fact' : 'forced by the MVD'}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
          <table className="viz-table">
            <thead>
              <tr>
                <th>#</th>
                <th>Operation applied</th>
                <th>Rows written, one table</th>
                <th>Rows written, 4NF pair</th>
                <th>What happened</th>
              </tr>
            </thead>
            <tbody>
              {log.length === 0 ? (
                <tr>
                  <td colSpan={5}>No operation applied yet.</td>
                </tr>
              ) : (
                log.map((r) => (
                  <tr key={r.n}>
                    <td>{r.n}</td>
                    <td>{r.op}</td>
                    <td>{r.flat}</td>
                    <td>{r.nf}</td>
                    <td>{r.note}</td>
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
          <svg width={svgW} height={height} role="img" aria-label="A flat employee table, its 4NF projections, and the natural join of those projections">
            {placed.map(({ p, x: px }) => {
              const w = panelW(p);
              const widths = p.cols.length === 3 ? colW : [colW[0], colW[2]];
              return (
                <g key={p.name}>
                  <rect
                    x={px}
                    y={headH - 8}
                    width={w}
                    height={maxRows * rowH + 18}
                    rx={8}
                    fill="var(--viz-plane)"
                    stroke={p.stored ? 'var(--viz-ink-2)' : 'var(--viz-border)'}
                    strokeWidth={p.stored ? 2 : 1}
                    strokeDasharray={p.stored ? undefined : '4 3'}
                  />
                  <text x={px + 8} y={12} fill="var(--viz-ink)" fontWeight={600}>
                    {p.name}
                  </text>
                  <text x={px + 8} y={26} fill="var(--viz-ink-muted)">
                    {p.sub}
                  </text>
                  {p.cols.map((cname, ci) => (
                    <text
                      key={cname}
                      x={px + 8 + widths.slice(0, ci).reduce((a, b) => a + b, 0)}
                      y={headH + 4}
                      fill="var(--viz-ink-2)"
                      fontWeight={600}
                    >
                      {cname}
                    </text>
                  ))}
                  {p.rows.map((r, ri) => {
                    const y = headH + 12 + ri * rowH;
                    const tone = TONE[r.tone];
                    return (
                      <g key={`${p.name}-${ri}-${r.cells.join('|')}`} {...tip(<>{r.hint}</>)} style={{ cursor: 'help' }}>
                        <rect x={px + 4} y={y} width={w - 8} height={rowH - 2} rx={3} fill="var(--viz-surface)" opacity={0.6} />
                        <rect x={px + 4} y={y} width={3} height={rowH - 2} fill={tone.fill} />
                        {r.cells.map((cell, ci) => (
                          <text
                            key={ci}
                            x={px + 8 + widths.slice(0, ci).reduce((a, b) => a + b, 0) + (ci === 0 ? 4 : 0)}
                            y={y + 11}
                            fill={r.tone === 'fact' ? 'var(--viz-ink)' : tone.fill}
                          >
                            {cell}
                          </text>
                        ))}
                        {tone.glyph ? (
                          <text x={px + w - 10} y={y + 11} textAnchor="end" fill={tone.fill} fontWeight={700}>
                            {tone.glyph}
                          </text>
                        ) : null}
                      </g>
                    );
                  })}
                  <text x={px + 8} y={headH + 12 + maxRows * rowH + 16} fill="var(--viz-ink-muted)">
                    {p.rows.length} row{p.rows.length === 1 ? '' : 's'}
                    {p.stored ? ' — stored' : ''}
                  </text>
                </g>
              );
            })}
          </svg>
        </TooltipHost>
      </div>
    </VizPanel>
  );
}

/** True when this pairing is one of the rows a human would have typed (the zip), not padding. */
function isFactRow(r: Row, skills: string[], phones: string[]) {
  if (r.emp !== E1) return true;
  const n = Math.max(skills.length, phones.length);
  for (let k = 0; k < n; k++) {
    if (skills[k % skills.length] === r.skill && phones[k % phones.length] === r.phone) return true;
  }
  return false;
}
