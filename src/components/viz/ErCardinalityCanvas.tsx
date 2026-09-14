import { useMemo, useState } from 'react';
import {
  VizPanel,
  Choice,
  Segmented,
  Check,
  Legend,
  Stats,
  Note,
  TooltipHost,
  useTip,
  makeRng,
  useSize,
} from './Viz';

/**
 * Entities, relationships, cardinality and participation — as a constraint, not a picture.
 *
 * The model is four numbers: how many B may exist per A (min, max) and how many A per B.
 * Everything else on screen is a *rendering* of those four numbers — the Chen diagram, the
 * crow's-foot diagram, the cardinality x participation matrix — plus one consequence: a fixed
 * candidate population of links is filtered through the constraint, so every edit visibly
 * permits some instances and forbids others.
 *
 * The pedagogically important asymmetry is preserved exactly: the maximums sit beside the same
 * entity in both notations, but participation does not. "Every Customer has at least one Order"
 * is a double line on the *Customer* end in Chen and a mandatory bar on the *Order* end in
 * crow's foot.
 */

/* ------------------------------------------------------------------- model */

type Bound = { min: 0 | 1; max: 1 | 'N' };
type EndKey = '0..1' | '1..1' | '0..N' | '1..N';

const END_ORDER: EndKey[] = ['0..1', '1..1', '0..N', '1..N'];

function parseEnd(k: EndKey): Bound {
  return { min: k[0] === '1' ? 1 : 0, max: k[3] === 'N' ? 'N' : 1 };
}
function endKey(b: Bound): EndKey {
  return `${b.min}..${b.max}` as EndKey;
}
function cap(b: Bound) {
  return b.max === 'N' ? Infinity : 1;
}

type Scenario = {
  id: string;
  label: string;
  a: string; // entity A
  b: string; // entity B
  verb: string; // relationship name, read A -> B
  passive: string; // read B -> A
  aRows: string[];
  bRows: string[];
  /** how many B per one A */
  perA: EndKey;
  /** how many A per one B */
  perB: EndKey;
  weak: boolean; // B is a weak entity of A, via an identifying relationship
  attrs: { key: string; multi: string; derived: string };
  seed: number;
};

const SCENARIOS: Scenario[] = [
  {
    id: 'order',
    label: 'Customer — places — Order',
    a: 'Customer',
    b: 'Order',
    verb: 'places',
    passive: 'is placed by',
    aRows: ['Ada', 'Bo', 'Chen', 'Dia'],
    bRows: ['#1001', '#1002', '#1003', '#1004'],
    perA: '0..N',
    perB: '1..1',
    weak: false,
    attrs: { key: 'cust_id', multi: 'phone', derived: 'total_spent' },
    seed: 5,
  },
  {
    id: 'line',
    label: 'Order — contains — OrderLine (weak)',
    a: 'Order',
    b: 'OrderLine',
    verb: 'contains',
    passive: 'belongs to',
    aRows: ['#1001', '#1002', '#1003', '#1004'],
    bRows: ['line 1', 'line 2', 'line 3', 'line 4'],
    perA: '1..N',
    perB: '1..1',
    weak: true,
    attrs: { key: 'order_id', multi: 'note', derived: 'order_total' },
    seed: 13,
  },
  {
    id: 'enrol',
    label: 'Student — enrols in — Course',
    a: 'Student',
    b: 'Course',
    verb: 'enrols in',
    passive: 'enrols',
    aRows: ['Ada', 'Bo', 'Chen', 'Dia'],
    bRows: ['CS101', 'CS240', 'MA201', 'PH110'],
    perA: '0..N',
    perB: '0..N',
    weak: false,
    attrs: { key: 'student_id', multi: 'email', derived: 'credits' },
    seed: 24,
  },
  {
    id: 'passport',
    label: 'Person — holds — Passport',
    a: 'Person',
    b: 'Passport',
    verb: 'holds',
    passive: 'is held by',
    aRows: ['Ada', 'Bo', 'Chen', 'Dia'],
    bRows: ['P-771', 'P-772', 'P-773', 'P-774'],
    perA: '0..1',
    perB: '1..1',
    weak: false,
    attrs: { key: 'person_id', multi: 'phone', derived: 'age' },
    seed: 34,
  },
  {
    id: 'manage',
    label: 'Employee — manages — Department',
    a: 'Employee',
    b: 'Department',
    verb: 'manages',
    passive: 'is managed by',
    aRows: ['Ada', 'Bo', 'Chen', 'Dia'],
    bRows: ['Sales', 'Eng', 'Legal', 'Ops'],
    perA: '0..1',
    perB: '0..1',
    weak: false,
    attrs: { key: 'emp_id', multi: 'skill', derived: 'tenure' },
    seed: 62,
  },
];

/** A fixed candidate population: six A–B pairs someone tried to record, in a fixed order. */
function candidatePool(seed: number): [number, number][] {
  const rng = makeRng(seed);
  const all: { a: number; b: number; k: number }[] = [];
  for (let a = 0; a < 4; a++) for (let b = 0; b < 4; b++) all.push({ a, b, k: rng() });
  all.sort((x, y) => x.k - y.k);
  return all.slice(0, 6).map(({ a, b }) => [a, b] as [number, number]);
}

type LinkVerdict = { a: number; b: number; kept: boolean; why: string };

/** Filter the candidate population through the two maximums, first-come first-served. */
function applyConstraint(pool: [number, number][], perA: Bound, perB: Bound, sc: Scenario) {
  const degA = [0, 0, 0, 0];
  const degB = [0, 0, 0, 0];
  const links: LinkVerdict[] = [];
  for (const [a, b] of pool) {
    if (degA[a] >= cap(perA)) {
      links.push({ a, b, kept: false, why: `${sc.a} ${sc.aRows[a]} already has its one ${sc.b}` });
    } else if (degB[b] >= cap(perB)) {
      links.push({ a, b, kept: false, why: `${sc.b} ${sc.bRows[b]} already has its one ${sc.a}` });
    } else {
      degA[a]++;
      degB[b]++;
      links.push({ a, b, kept: true, why: 'within both maximums' });
    }
  }
  const orphanA = degA.map((d, i) => (d === 0 && perA.min === 1 ? i : -1)).filter((i) => i >= 0);
  const orphanB = degB.map((d, i) => (d === 0 && perB.min === 1 ? i : -1)).filter((i) => i >= 0);
  return { links, degA, degB, orphanA, orphanB };
}

function cardClass(perA: Bound, perB: Bound): '1:1' | '1:N' | 'M:N' {
  if (perA.max === 'N' && perB.max === 'N') return 'M:N';
  if (perA.max === 'N' || perB.max === 'N') return '1:N';
  return '1:1';
}

/** What this shape becomes in DDL — the bridge to the mapping page. */
function mapsTo(sc: Scenario, perA: Bound, perB: Bound) {
  const k = cardClass(perA, perB);
  if (k === 'M:N') return `junction table ${sc.a.toLowerCase()}_${sc.b.toLowerCase()}, PK on both FKs`;
  if (k === '1:N') {
    const many = perA.max === 'N' ? sc.b : sc.a; // the side that holds the FK
    const notNull = (perA.max === 'N' ? perB.min : perA.min) === 1;
    return `FK on ${many.toLowerCase()}, ${notNull ? 'NOT NULL' : 'nullable'}`;
  }
  const host = perB.min === 1 ? sc.b : sc.a;
  const nn = (host === sc.b ? perB.min : perA.min) === 1;
  return `FK on ${host.toLowerCase()} with UNIQUE, ${nn ? 'NOT NULL' : 'nullable'}`;
}

const WORDS: Record<EndKey, string> = {
  '0..1': 'zero or one',
  '1..1': 'exactly one',
  '0..N': 'zero or more',
  '1..N': 'one or more',
};

/* ----------------------------------------------------------------- drawing */

const INK = 'var(--viz-ink)';
const INK2 = 'var(--viz-ink-2)';
const LINE = 'var(--viz-axis)';
const OK = 'var(--viz-1)';
const BAD = 'var(--viz-stale)';
const VIOL = 'var(--viz-critical)';

function EntityBox({
  x,
  y,
  w,
  h,
  name,
  weak,
  round,
}: {
  x: number;
  y: number;
  w: number;
  h: number;
  name: string;
  weak?: boolean;
  round?: boolean;
}) {
  return (
    <g>
      <rect x={x} y={y} width={w} height={h} rx={round ? 10 : 2} fill="var(--viz-plane)" stroke={INK2} strokeWidth={1.4} />
      {weak ? (
        <rect
          x={x + 4}
          y={y + 4}
          width={w - 8}
          height={h - 8}
          rx={round ? 7 : 2}
          fill="none"
          stroke={INK2}
          strokeWidth={1.1}
        />
      ) : null}
      <text x={x + w / 2} y={y + h / 2 + 4} textAnchor="middle" fill={INK} style={{ fontWeight: 600 }}>
        {name}
      </text>
    </g>
  );
}

/** One crow's-foot end: outer symbol = maximum (touches the entity), inner symbol = minimum. */
function CrowEnd({ x, y, dir, bound }: { x: number; y: number; dir: 1 | -1; bound: Bound }) {
  const o = x + dir * 12; // apex of the crow / position of the "one" bar
  const i = x + dir * 26; // the optionality symbol
  return (
    <g stroke={OK} strokeWidth={1.6} fill="none">
      {bound.max === 'N' ? (
        <>
          <path d={`M ${o} ${y} L ${x} ${y - 9}`} />
          <path d={`M ${o} ${y} L ${x} ${y + 9}`} />
          <path d={`M ${o} ${y} L ${x} ${y}`} />
        </>
      ) : (
        <path d={`M ${o} ${y - 9} L ${o} ${y + 9}`} />
      )}
      {bound.min === 1 ? (
        <path d={`M ${i} ${y - 9} L ${i} ${y + 9}`} />
      ) : (
        <circle cx={i} cy={y} r={5} fill="var(--viz-surface)" />
      )}
    </g>
  );
}

/* --------------------------------------------------------------- component */

export default function ErCardinalityCanvas() {
  const [scenarioId, setScenarioId] = useState('order');
  const sc = SCENARIOS.find((s) => s.id === scenarioId)!;

  const [perAKey, setPerAKey] = useState<EndKey>(sc.perA);
  const [perBKey, setPerBKey] = useState<EndKey>(sc.perB);
  const [identifying, setIdentifying] = useState(sc.weak);
  const [showAttrs, setShowAttrs] = useState(false);
  const [ref, width] = useSize(760);
  const tip = useTip();

  const pickScenario = (id: string) => {
    const next = SCENARIOS.find((s) => s.id === id)!;
    setScenarioId(id);
    setPerAKey(next.perA);
    setPerBKey(next.perB);
    setIdentifying(next.weak);
  };

  // An identifying relationship is not a style choice: a weak entity has no key of its own, so it
  // must have exactly one owner, always. Turning it on pins the B end to 1..1.
  const perA = parseEnd(perAKey);
  const perB = identifying ? parseEnd('1..1') : parseEnd(perBKey);

  const pool = useMemo(() => candidatePool(sc.seed), [sc.seed]);
  const { links, degA, degB, orphanA, orphanB } = applyConstraint(pool, perA, perB, sc);
  const kind = cardClass(perA, perB);
  const kept = links.filter((l) => l.kept).length;
  const rejected = links.length - kept;

  const cycle = (which: 'a' | 'b') => {
    if (which === 'b' && identifying) return;
    const cur = which === 'a' ? perAKey : perBKey;
    const nextKey = END_ORDER[(END_ORDER.indexOf(cur) + 1) % END_ORDER.length];
    (which === 'a' ? setPerAKey : setPerBKey)(nextKey);
  };

  /* ---- geometry ---- */
  const W = Math.max(width, 680);
  const half = W / 2;
  const attrH = showAttrs ? 76 : 0;
  const rowY = 74 + attrH; // top of the entity boxes in both notation panels
  const h1 = rowY + 96;

  const aBoxW = 96;
  const cy = rowY + 19;

  // Chen panel: [0, half-14]
  const chenW = half - 14;
  const dx = chenW / 2;

  // Crow's-foot panel: [half+14, W]
  const cf0 = half + 14;
  const cfW = W - cf0;
  const aRight = cf0 + 6 + aBoxW;
  const bLeft = cf0 + cfW - 6 - aBoxW;

  const h2 = 244;
  const W2 = Math.max(width, 700);
  const colAx = 10;
  const colBx = 210;
  const chipW = 108;
  const chipH = 28;
  const instY = (i: number) => 52 + i * 46;
  const mx = 396; // matrix origin
  const cellW = 104;
  const cellH = 44;

  const rows: ('1:1' | '1:N' | 'M:N')[] = ['1:1', '1:N', 'M:N'];
  const badgeCol = (b: Bound) => (b.min === 1 ? 1 : 0);

  /* ---- narration: one instance this shape permits, one it forbids ---- */
  const firstReject = links.find((l) => !l.kept);
  const multiA = degA.findIndex((d) => d >= 2);
  const multiB = degB.findIndex((d) => d >= 2);
  const freeA = degA.findIndex((d) => d === 0);
  let permits: string;
  if (multiA >= 0) permits = `${sc.aRows[multiA]} ${sc.verb} ${degA[multiA]} different ${sc.b}s.`;
  else if (multiB >= 0) permits = `${sc.bRows[multiB]} ${sc.passive} ${degB[multiB]} different ${sc.a}s.`;
  else if (freeA >= 0 && perA.min === 0) permits = `${sc.aRows[freeA]} exists with no ${sc.b} at all.`;
  else permits = `every ${sc.a} and every ${sc.b} is paired off one-to-one.`;
  let forbids: string;
  if (orphanB.length) forbids = `${sc.bRows[orphanB[0]]} with no ${sc.a} — total participation is violated.`;
  else if (orphanA.length) forbids = `${sc.aRows[orphanA[0]]} with no ${sc.b} — total participation is violated.`;
  else if (firstReject)
    forbids = `${sc.aRows[firstReject.a]} — ${sc.bRows[firstReject.b]}: ${firstReject.why}.`;
  else forbids = 'nothing in this population — every candidate row fits.';

  const noun = (name: string, b: Bound) => (b.max === 'N' ? `${name}s` : name);
  const sentenceA = `Each ${sc.a} ${sc.verb} ${WORDS[endKey(perA)]} ${noun(sc.b, perA)}.`;
  const sentenceB = `Each ${sc.b} ${sc.passive} ${WORDS[endKey(perB)]} ${noun(sc.a, perB)}.`;

  return (
    <VizPanel
      title="An ER edge is four numbers"
      subtitle="Set how many B may exist per A and how many A per B — by control or by clicking the handles on the crow's-foot ends — and watch the same constraint render in both notations and filter a real population of rows."
      controls={
        <>
          <Choice
            label="Relationship"
            value={scenarioId}
            onChange={pickScenario}
            options={SCENARIOS.map((s) => ({ value: s.id, label: s.label }))}
          />
          <Segmented
            label={`${sc.b} per ${sc.a}`}
            value={perAKey}
            onChange={(v) => setPerAKey(v)}
            options={END_ORDER.map((k) => ({ value: k, label: k, title: WORDS[k] }))}
          />
          <Segmented
            label={`${sc.a} per ${sc.b}`}
            value={endKey(perB)}
            onChange={(v) => {
              if (!identifying) setPerBKey(v);
            }}
            options={END_ORDER.map((k) => ({
              value: k,
              label: k,
              title: identifying ? 'pinned to 1..1 by the identifying relationship' : WORDS[k],
            }))}
          />
          <Check
            label={`Identifying (${sc.b} is weak)`}
            checked={identifying}
            onChange={(v) => {
              setIdentifying(v);
              if (v) setPerBKey('1..1');
            }}
          />
          <Check label="Show attributes" checked={showAttrs} onChange={setShowAttrs} />
        </>
      }
      legend={
        <Legend
          items={[
            { label: 'Link the constraint allows', color: OK, shape: 'line' },
            { label: 'Rejected: a maximum is already used up (dashed, ✕)', color: BAD, shape: 'line' },
            { label: 'Violates total participation (row with 0 links, ! badge)', color: VIOL },
          ]}
        />
      }
      stats={
        <Stats
          items={[
            { label: 'Cardinality', value: kind },
            {
              label: 'Participation',
              value: `${perA.min === 1 ? sc.a : '—'} / ${perB.min === 1 ? sc.b : '—'}`,
              hint: 'Which side must participate in at least one relationship instance',
            },
            { label: 'Rows accepted', value: `${kept} of ${links.length}` },
            { label: 'Orphan violations', value: orphanA.length + orphanB.length },
            { label: 'Maps to', value: <span style={{ fontSize: '0.78rem' }}>{mapsTo(sc, perA, perB)}</span> },
          ]}
        />
      }
      note={
        <Note>
          <strong>{sentenceA}</strong> {sentenceB} Now permitted: {permits} Now forbidden: {forbids}
          {identifying ? (
            <>
              {' '}
              The identifying relationship pins the {sc.a} end to <code>1..1</code>: a weak entity has no
              key of its own, so it cannot exist without exactly one owner.
            </>
          ) : null}
        </Note>
      }
      table={
        <table className="viz-table">
          <thead>
            <tr>
              <th>#</th>
              <th>{sc.a}</th>
              <th>{sc.b}</th>
              <th>Verdict</th>
              <th>Reason</th>
            </tr>
          </thead>
          <tbody>
            {links.map((l, i) => (
              <tr key={`${l.a}-${l.b}-${i}`}>
                <td>{i + 1}</td>
                <td>{sc.aRows[l.a]}</td>
                <td>{sc.bRows[l.b]}</td>
                <td>{l.kept ? 'accepted' : 'rejected'}</td>
                <td>{l.why}</td>
              </tr>
            ))}
            {[0, 1, 2, 3].map((i) => (
              <tr key={`da${i}`}>
                <td>—</td>
                <td>{sc.aRows[i]}</td>
                <td>—</td>
                <td>
                  {degA[i]} link{degA[i] === 1 ? '' : 's'}
                </td>
                <td>{orphanA.includes(i) ? `violates total participation of ${sc.a}` : 'within bounds'}</td>
              </tr>
            ))}
            {[0, 1, 2, 3].map((i) => (
              <tr key={`db${i}`}>
                <td>—</td>
                <td>—</td>
                <td>{sc.bRows[i]}</td>
                <td>
                  {degB[i]} link{degB[i] === 1 ? '' : 's'}
                </td>
                <td>{orphanB.includes(i) ? `violates total participation of ${sc.b}` : 'within bounds'}</td>
              </tr>
            ))}
          </tbody>
        </table>
      }
    >
      <TooltipHost>
        <div ref={ref}>
          {/* -------------------------------------------------- notations */}
          <svg width={W} height={h1} role="img" aria-label="The same relationship in Chen and crow's-foot notation">
            <text x={4} y={14} fill={INK2} style={{ fontWeight: 600 }}>
              Chen — the double line is participation
            </text>
            <line x1={half} y1={4} x2={half} y2={h1 - 4} className="viz-grid-line" />
            <text x={cf0} y={14} fill={INK2} style={{ fontWeight: 600 }}>
              Crow&rsquo;s foot (IE) — click an end handle to cycle it
            </text>

            {/* ---- Chen ---- */}
            {showAttrs ? (
              <g>
                {[
                  { cx: 44, label: sc.attrs.key, kind: 'key' },
                  { cx: 122, label: sc.attrs.multi, kind: 'multi' },
                  { cx: 200, label: sc.attrs.derived, kind: 'derived' },
                ].map((at) => (
                  <g key={at.label}>
                    <line x1={at.cx} y1={62} x2={54} y2={rowY} stroke={LINE} strokeWidth={1} />
                    <ellipse
                      cx={at.cx}
                      cy={46}
                      rx={37}
                      ry={16}
                      fill="var(--viz-plane)"
                      stroke={INK2}
                      strokeWidth={1.2}
                      strokeDasharray={at.kind === 'derived' ? '4 3' : undefined}
                    />
                    {at.kind === 'multi' ? (
                      <ellipse cx={at.cx} cy={46} rx={32} ry={12} fill="none" stroke={INK2} strokeWidth={1} />
                    ) : null}
                    <text x={at.cx} y={49} textAnchor="middle" fill={INK} style={{ fontSize: 9.5 }}>
                      {at.label}
                    </text>
                    {at.kind === 'key' ? (
                      <line
                        x1={at.cx - at.label.length * 2.6}
                        y1={52}
                        x2={at.cx + at.label.length * 2.6}
                        y2={52}
                        stroke={INK}
                        strokeWidth={1}
                      />
                    ) : null}
                    <text x={at.cx} y={74} textAnchor="middle" fill={INK2} style={{ fontSize: 9 }}>
                      {at.kind === 'key' ? 'key' : at.kind === 'multi' ? 'multi-valued' : 'derived'}
                    </text>
                  </g>
                ))}
              </g>
            ) : null}

            {/* A — diamond */}
            {perA.min === 1 ? (
              <>
                <line x1={6 + aBoxW} y1={cy - 3} x2={dx - 46} y2={cy - 3} stroke={LINE} strokeWidth={1.4} />
                <line x1={6 + aBoxW} y1={cy + 3} x2={dx - 46} y2={cy + 3} stroke={LINE} strokeWidth={1.4} />
              </>
            ) : (
              <line x1={6 + aBoxW} y1={cy} x2={dx - 46} y2={cy} stroke={LINE} strokeWidth={1.4} />
            )}
            {perB.min === 1 ? (
              <>
                <line x1={dx + 46} y1={cy - 3} x2={chenW - 6 - aBoxW} y2={cy - 3} stroke={LINE} strokeWidth={1.4} />
                <line x1={dx + 46} y1={cy + 3} x2={chenW - 6 - aBoxW} y2={cy + 3} stroke={LINE} strokeWidth={1.4} />
              </>
            ) : (
              <line x1={dx + 46} y1={cy} x2={chenW - 6 - aBoxW} y2={cy} stroke={LINE} strokeWidth={1.4} />
            )}

            <EntityBox x={6} y={rowY} w={aBoxW} h={38} name={sc.a} />
            <EntityBox x={chenW - 6 - aBoxW} y={rowY} w={aBoxW} h={38} name={sc.b} weak={identifying} />

            <polygon
              points={`${dx - 46},${cy} ${dx},${cy - 22} ${dx + 46},${cy} ${dx},${cy + 22}`}
              fill="var(--viz-plane)"
              stroke={INK2}
              strokeWidth={1.4}
            />
            {identifying ? (
              <polygon
                points={`${dx - 39},${cy} ${dx},${cy - 18.5} ${dx + 39},${cy} ${dx},${cy + 18.5}`}
                fill="none"
                stroke={INK2}
                strokeWidth={1.1}
              />
            ) : null}
            <text x={dx} y={cy + 4} textAnchor="middle" fill={INK} style={{ fontSize: 10 }}>
              {sc.verb}
            </text>

            {/* Chen's ratio labels: the number beside an entity counts THAT entity, per one of the other. */}
            <text x={6 + aBoxW + 10} y={cy - 9} fill={OK} style={{ fontWeight: 700 }}>
              {perB.max === 'N' ? (perA.max === 'N' ? 'M' : 'N') : '1'}
            </text>
            <text x={chenW - 6 - aBoxW - 10} y={cy - 9} textAnchor="end" fill={OK} style={{ fontWeight: 700 }}>
              {perA.max === 'N' ? 'N' : '1'}
            </text>
            <text x={6} y={rowY + 56} fill={INK2} style={{ fontSize: 10 }}>
              double line = total participation ({perA.min === 1 ? sc.a : '—'}
              {perB.min === 1 ? `, ${sc.b}` : ''})
            </text>

            {/* ---- crow's foot ---- */}
            <line
              x1={aRight}
              y1={cy}
              x2={bLeft}
              y2={cy}
              stroke={OK}
              strokeWidth={1.6}
              strokeDasharray={identifying ? undefined : '5 4'}
            />
            <CrowEnd x={aRight} y={cy} dir={1} bound={perB} />
            <CrowEnd x={bLeft} y={cy} dir={-1} bound={perA} />
            <EntityBox x={cf0 + 6} y={rowY} w={aBoxW} h={38} name={sc.a} />
            <EntityBox x={bLeft} y={rowY} w={aBoxW} h={38} name={sc.b} round={identifying} />

            {/* click targets: the handle at each end */}
            {(
              [
                { which: 'b' as const, x: aRight, label: `${sc.a} per ${sc.b}: ${endKey(perB)}` },
                { which: 'a' as const, x: bLeft - 44, label: `${sc.b} per ${sc.a}: ${endKey(perA)}` },
              ]
            ).map((hnd) => (
              <g
                key={hnd.which}
                onClick={() => cycle(hnd.which)}
                onKeyDown={(e) => {
                  if (e.key === 'Enter' || e.key === ' ') {
                    e.preventDefault();
                    cycle(hnd.which);
                  }
                }}
                role="button"
                aria-label={`Cycle ${hnd.label}`}
                style={{ cursor: hnd.which === 'b' && identifying ? 'not-allowed' : 'pointer' }}
                {...tip(
                  <>
                    <strong>{hnd.label}</strong>
                    <br />
                    {hnd.which === 'b' && identifying
                      ? 'Pinned by the identifying relationship.'
                      : 'Click to cycle 0..1 → 1..1 → 0..N → 1..N.'}
                  </>,
                )}
              >
                <rect x={hnd.x} y={cy - 16} width={44} height={32} fill="transparent" />
              </g>
            ))}

            <text x={cf0 + 6} y={rowY + 56} fill={INK2} style={{ fontSize: 10 }}>
              {sentenceA}
            </text>
            <text x={cf0 + 6} y={rowY + 70} fill={INK2} style={{ fontSize: 10 }}>
              {sentenceB}
            </text>
            <text x={cf0 + 6} y={rowY + 86} fill={INK2} style={{ fontSize: 9.5 }}>
              {identifying ? 'solid line = identifying (IDEF1X)' : 'dashed line = non-identifying (IDEF1X)'}
            </text>
          </svg>

          {/* ------------------------------------ population + the matrix */}
          <svg
            width={W2}
            height={h2}
            role="img"
            aria-label="A candidate population filtered through the constraint, and the cardinality by participation matrix"
          >
            <text x={colAx} y={16} fill={INK2} style={{ fontWeight: 600 }}>
              Six candidate rows, applied in order
            </text>
            <text x={colAx} y={32} fill={INK}>
              {sc.a}
            </text>
            <text x={colBx} y={32} fill={INK}>
              {sc.b}
            </text>

            {links.map((l, i) => {
              const y1 = instY(l.a) + chipH / 2;
              const y2 = instY(l.b) + chipH / 2;
              const x1 = colAx + chipW;
              const x2 = colBx;
              const mxp = (x1 + x2) / 2;
              const myp = (y1 + y2) / 2;
              return (
                <g key={`l${i}`} {...tip(
                  <>
                    <strong>
                      {sc.aRows[l.a]} — {sc.bRows[l.b]}
                    </strong>
                    <br />
                    {l.kept ? 'accepted' : 'rejected'}: {l.why}
                  </>,
                )}>
                  <line
                    x1={x1}
                    y1={y1}
                    x2={x2}
                    y2={y2}
                    stroke={l.kept ? OK : BAD}
                    strokeWidth={l.kept ? 1.8 : 1.2}
                    strokeDasharray={l.kept ? undefined : '4 3'}
                  />
                  {l.kept ? null : (
                    <text x={mxp} y={myp + 4} textAnchor="middle" fill={BAD} style={{ fontWeight: 700 }}>
                      ✕
                    </text>
                  )}
                </g>
              );
            })}

            {[0, 1, 2, 3].map((i) => {
              const bad = orphanA.includes(i);
              return (
                <g key={`a${i}`} {...tip(
                  <>
                    <strong>{sc.aRows[i]}</strong>
                    <br />
                    {degA[i]} accepted link{degA[i] === 1 ? '' : 's'}
                    {bad ? ` — but every ${sc.a} must have at least one ${sc.b}` : ''}
                  </>,
                )}>
                  <rect
                    x={colAx}
                    y={instY(i)}
                    width={chipW}
                    height={chipH}
                    rx={6}
                    fill="var(--viz-plane)"
                    stroke={bad ? VIOL : INK2}
                    strokeWidth={bad ? 2 : 1.2}
                  />
                  <text x={colAx + 10} y={instY(i) + 18} fill={INK}>
                    {sc.aRows[i]}
                  </text>
                  <text x={colAx + chipW - 8} y={instY(i) + 18} textAnchor="end" fill={bad ? VIOL : INK2}>
                    {bad ? '! 0' : degA[i]}
                  </text>
                </g>
              );
            })}

            {[0, 1, 2, 3].map((i) => {
              const bad = orphanB.includes(i);
              return (
                <g key={`b${i}`} {...tip(
                  <>
                    <strong>{sc.bRows[i]}</strong>
                    <br />
                    {degB[i]} accepted link{degB[i] === 1 ? '' : 's'}
                    {bad ? ` — but every ${sc.b} must have at least one ${sc.a}` : ''}
                  </>,
                )}>
                  <rect
                    x={colBx}
                    y={instY(i)}
                    width={chipW}
                    height={chipH}
                    rx={6}
                    fill="var(--viz-plane)"
                    stroke={bad ? VIOL : INK2}
                    strokeWidth={bad ? 2 : 1.2}
                  />
                  <text x={colBx + 10} y={instY(i) + 18} fill={INK}>
                    {sc.bRows[i]}
                  </text>
                  <text x={colBx + chipW - 8} y={instY(i) + 18} textAnchor="end" fill={bad ? VIOL : INK2}>
                    {bad ? '! 0' : degB[i]}
                  </text>
                </g>
              );
            })}

            {/* ---- the six shapes ---- */}
            <text x={mx} y={16} fill={INK2} style={{ fontWeight: 600 }}>
              Cardinality × participation — the six named shapes
            </text>
            {['partial (min 0)', 'total (min 1)'].map((c, j) => (
              <text key={c} x={mx + 66 + j * cellW + cellW / 2} y={40} textAnchor="middle" fill={INK2}>
                {c}
              </text>
            ))}
            {rows.map((r, i) => (
              <g key={r}>
                <text x={mx} y={52 + i * cellH + cellH / 2 + 4} fill={r === kind ? INK : INK2} style={{ fontWeight: r === kind ? 700 : 400 }}>
                  {r}
                </text>
                {[0, 1].map((j) => {
                  const here = r === kind;
                  const badges = here
                    ? [
                        ...(badgeCol(perA) === j ? [sc.a] : []),
                        ...(badgeCol(perB) === j ? [sc.b] : []),
                      ]
                    : [];
                  return (
                    <g key={j}>
                      <rect
                        x={mx + 66 + j * cellW}
                        y={52 + i * cellH}
                        width={cellW - 6}
                        height={cellH - 6}
                        rx={6}
                        fill={here ? 'var(--viz-neutral)' : 'var(--viz-plane)'}
                        stroke={here ? OK : 'var(--viz-grid)'}
                        strokeWidth={here ? 1.6 : 1}
                      />
                      {badges.map((bname, k) => (
                        <text
                          key={bname}
                          x={mx + 66 + j * cellW + (cellW - 6) / 2}
                          y={52 + i * cellH + (badges.length === 1 ? 24 : 16 + k * 14)}
                          textAnchor="middle"
                          fill={INK}
                          style={{ fontWeight: 600 }}
                        >
                          {bname}
                        </text>
                      ))}
                    </g>
                  );
                })}
              </g>
            ))}
            <text x={mx} y={52 + 3 * cellH + 22} fill={INK2} style={{ fontSize: 10 }}>
              A badge shows which entity&rsquo;s participation each column describes.
            </text>
            <text x={mx} y={52 + 3 * cellH + 36} fill={INK2} style={{ fontSize: 10 }}>
              Both ends move independently: 3 × 2 × 2 = 12 distinct edges in all.
            </text>
          </svg>
        </div>
      </TooltipHost>
    </VizPanel>
  );
}
