import { useMemo, useState, type ReactNode } from 'react';
import {
  VizPanel,
  Choice,
  Segmented,
  Check,
  Button,
  Slider,
  Legend,
  Stats,
  Note,
  TooltipHost,
  useTip,
  makeRng,
  fmtNum,
  useSize,
} from './Viz';

/**
 * A functional-dependency decomposer that actually runs the algorithms.
 *
 * The learner switches declared FDs on and off; everything below recomputes from scratch:
 * attribute closure X+, the candidate keys, which subsets violate 2NF/3NF/BCNF, the
 * decomposition itself (Bernstein's 3NF synthesis from a minimal cover, or the recursive
 * BCNF split), the dependency-preservation test on the projected dependencies, and an
 * empirical lossless-join check that literally natural-joins the fragment instances back
 * together and looks for spurious tuples.
 *
 * Then a single UPDATE is staged against both the flat relation and the decomposition, and
 * the rows each one has to rewrite are highlighted. That difference is the whole argument.
 *
 * Every number on screen is computed from the instance in PRESETS — nothing is asserted.
 */

/* -------------------------------------------------------------- attribute sets */

type Mask = number;

const bit = (i: number) => 1 << i;
const hasAttr = (m: Mask, i: number) => (m & bit(i)) !== 0;

function ids(m: Mask, n: number): number[] {
  const out: number[] = [];
  for (let i = 0; i < n; i++) if (hasAttr(m, i)) out.push(i);
  return out;
}

function popcount(m: Mask) {
  let c = 0;
  let x = m;
  while (x) {
    x &= x - 1;
    c++;
  }
  return c;
}

/** Every subset of m, largest first (the standard submask enumeration). */
function subsets(m: Mask): Mask[] {
  const out: Mask[] = [];
  for (let s = m; ; s = (s - 1) & m) {
    out.push(s);
    if (s === 0) break;
  }
  return out;
}

type FD = { l: Mask; r: Mask };

/** X+ under F: the only algorithm in this page you ever run by hand. */
function closure(seed: Mask, fds: FD[]): Mask {
  let x = seed;
  for (let guard = 0; guard < 64; guard++) {
    let grew = false;
    for (const f of fds) {
      if ((x & f.l) === f.l && (x | f.r) !== x) {
        x |= f.r;
        grew = true;
      }
    }
    if (!grew) break;
  }
  return x;
}

/** Minimal superkeys of `rel` under F. Deterministic order: fewest attributes first. */
function candidateKeys(rel: Mask, fds: FD[]): Mask[] {
  const supers = subsets(rel).filter((s) => s !== 0 && (closure(s, fds) & rel) === rel);
  const keys = supers.filter((s) => !supers.some((t) => t !== s && (t & s) === t));
  return keys.sort((a, b) => popcount(a) - popcount(b) || a - b);
}

type Violation = { x: Mask; extra: Mask };

/**
 * The smallest X ⊆ rel whose non-trivial closure inside rel does not cover rel.
 * For '3nf' an FD is forgiven when every dependent attribute is prime.
 */
function minViolation(rel: Mask, fds: FD[], prime: Mask, mode: 'bcnf' | '3nf'): Violation | null {
  let best: Violation | null = null;
  for (const x of subsets(rel)) {
    if (x === 0 || x === rel) continue;
    const cl = closure(x, fds) & rel;
    if (cl === x) continue; // trivial
    if (cl === rel) continue; // X is a superkey — allowed in every normal form
    const extra = cl & ~x;
    if (mode === '3nf' && (extra & ~prime) === 0) continue;
    if (!best || popcount(x) < popcount(best.x) || (popcount(x) === popcount(best.x) && x < best.x)) {
      best = { x, extra };
    }
  }
  return best;
}

/** 2NF fails only when a *proper subset of a candidate key* determines a non-prime attribute. */
function partialViolation(rel: Mask, fds: FD[], keys: Mask[], prime: Mask): Violation | null {
  let best: Violation | null = null;
  for (const k of keys) {
    for (const x of subsets(k)) {
      if (x === 0 || x === k) continue;
      const extra = (closure(x, fds) & rel) & ~x;
      if ((extra & ~prime) === 0) continue;
      if (!best || popcount(x) < popcount(best.x) || (popcount(x) === popcount(best.x) && x < best.x)) {
        best = { x, extra };
      }
    }
  }
  return best;
}

type NF = '1NF' | '2NF' | '3NF' | 'BCNF';

function normalForm(rel: Mask, fds: FD[]): NF {
  const keys = candidateKeys(rel, fds);
  const prime = keys.reduce((a, b) => a | b, 0);
  if (!minViolation(rel, fds, prime, 'bcnf')) return 'BCNF';
  if (!minViolation(rel, fds, prime, '3nf')) return '3NF';
  if (!partialViolation(rel, fds, keys, prime)) return '2NF';
  return '1NF';
}

/* ------------------------------------------------------------- decompositions */

type Step = { text: string };

/** Recursive BCNF split. Always lossless: the shared attributes X are a key of X+. */
function bcnfDecompose(rel: Mask, fds: FD[], steps: Step[], names: string[]): Mask[] {
  let frags = [rel];
  for (let guard = 0; guard < 16; guard++) {
    let at = -1;
    let v: Violation | null = null;
    for (let i = 0; i < frags.length; i++) {
      const keys = candidateKeys(frags[i], fds);
      const prime = keys.reduce((a, b) => a | b, 0);
      const found = minViolation(frags[i], fds, prime, 'bcnf');
      if (found) {
        at = i;
        v = found;
        break;
      }
    }
    if (at < 0 || !v) break;
    const cl = closure(v.x, fds);
    const a1 = cl & frags[at];
    const a2 = v.x | (frags[at] & ~cl);
    steps.push({
      text:
        `${show(v.x, names)} → ${show(v.extra, names)} holds inside ${show(frags[at], names)} but ` +
        `${show(v.x, names)} is not a superkey of it — split off ${show(a1, names)}, leaving ${show(a2, names)}. ` +
        `The two share exactly ${show(v.x, names)}, which is a key of the first, so the join is lossless.`,
    });
    frags = [...frags.slice(0, at), a1, a2, ...frags.slice(at + 1)];
  }
  return frags;
}

/** Minimal (canonical) cover: singleton right sides, no extraneous left attributes, no redundant FDs. */
function minimalCover(fds: FD[]): FD[] {
  let set: FD[] = [];
  for (const f of fds) {
    for (let i = 0; i < 24; i++) {
      if (hasAttr(f.r & ~f.l, i)) set.push({ l: f.l, r: bit(i) });
    }
  }
  // strip extraneous left-hand attributes
  set = set.map((f) => {
    let l = f.l;
    for (const i of ids(f.l, 24)) {
      const trial = l & ~bit(i);
      if (trial !== 0 && (closure(trial, set) & f.r) === f.r) l = trial;
    }
    return { l, r: f.r };
  });
  // drop redundant dependencies, in declaration order
  const kept: FD[] = [];
  for (let i = 0; i < set.length; i++) {
    const rest = [...kept, ...set.slice(i + 1)];
    if ((closure(set[i].l, rest) & set[i].r) !== set[i].r) kept.push(set[i]);
  }
  return kept;
}

/** Bernstein synthesis: one relation per determinant group, plus a candidate key if none is covered. */
function synth3NF(rel: Mask, fds: FD[], steps: Step[], names: string[]): Mask[] {
  const cover = minimalCover(fds);
  const groups = new Map<Mask, Mask>();
  for (const f of cover) groups.set(f.l, (groups.get(f.l) ?? 0) | f.l | f.r);
  let frags = [...groups.values()];
  frags = frags.filter((f, i) => !frags.some((g, j) => j !== i && (g & f) === f && (g !== f || j < i)));
  const keys = candidateKeys(rel, fds);
  if (keys.length && !frags.some((f) => (f & keys[0]) === keys[0])) {
    frags.push(keys[0]);
    steps.push({
      text: `No synthesized relation contains a candidate key, so ${show(keys[0], names)} is added as its own relation to keep the join lossless.`,
    });
  }
  steps.unshift({
    text:
      `Minimal cover: ${cover.map((f) => `${show(f.l, names)}→${show(f.r, names)}`).join(', ')}. ` +
      `One relation per distinct determinant, then absorb any relation contained in another.`,
  });
  return frags.length ? frags : [rel];
}

/** 2NF only: lift every partial dependency out, leave the key plus what genuinely depends on all of it. */
function decompose2NF(rel: Mask, fds: FD[], steps: Step[], names: string[]): Mask[] {
  const keys = candidateKeys(rel, fds);
  const prime = keys.reduce((a, b) => a | b, 0);
  const key = keys[0] ?? rel;
  const parts: Mask[] = [];
  let lifted = 0;
  for (const x of subsets(key).sort((a, b) => popcount(b) - popcount(a))) {
    if (x === 0 || x === key) continue;
    const dep = (closure(x, fds) & rel) & ~x & ~prime & ~lifted;
    if (dep === 0) continue;
    parts.push(x | dep);
    lifted |= dep;
    steps.push({
      text: `${show(x, names)} is a proper subset of the key ${show(key, names)} and already determines ${show(dep, names)} — a partial dependency. Lift it into ${show(x | dep, names)}.`,
    });
  }
  const remainder = (rel & ~lifted) | key;
  parts.push(remainder);
  return parts;
}

function naiveSplit(order: number[], p: number): Mask[] {
  const a = order.slice(0, p).reduce((m, i) => m | bit(i), 0);
  const b = order.slice(p - 1).reduce((m, i) => m | bit(i), 0);
  return [a, b];
}

/* ------------------------------------------------------- dependency preservation */

/**
 * Is X→Y implied by the union of the projections πRi(F)? The textbook fixpoint:
 * grow Z by (Z∩Ri)+ ∩ Ri until nothing changes.
 */
function isPreserved(fd: FD, frags: Mask[], fds: FD[]): boolean {
  let z = fd.l;
  for (let guard = 0; guard < 32; guard++) {
    let grew = false;
    for (const f of frags) {
      const add = closure(z & f, fds) & f;
      if ((z | add) !== z) {
        z |= add;
        grew = true;
      }
    }
    if (!grew) break;
  }
  return (z & fd.r) === fd.r;
}

/* --------------------------------------------------------------------- presets */

type Row = Record<string, string>;
type FDDef = { id: string; lhs: string[]; rhs: string[]; note: string };
type UpdDef = { id: string; label: string; setAttr: string; whereAttr: string; whereVal: string; newVal: string };

type Preset = {
  id: string;
  label: string;
  rel: string;
  attrs: string[];
  fds: FDDef[];
  rows: Row[];
  updates: UpdDef[];
};

/** Deterministic quantities — the same table on every render and every SSR pass. */
const QTY = (() => {
  const rng = makeRng(1974); // Armstrong's axioms
  return Array.from({ length: 12 }, () => String(1 + Math.floor(rng() * 6)));
})();

const ORDER_ROWS: Row[] = (
  [
    ['o-1001', 'A-100', 'c1', 'kay@vine.io', 'gold', 'w1', 'us-west'],
    ['o-1001', 'B-220', 'c1', 'kay@vine.io', 'gold', 'w1', 'us-west'],
    ['o-1002', 'A-100', 'c1', 'kay@vine.io', 'gold', 'w1', 'us-west'],
    ['o-1003', 'C-310', 'c2', 'lin@bay.dev', 'silver', 'w2', 'us-east'],
    ['o-1003', 'A-100', 'c2', 'lin@bay.dev', 'silver', 'w2', 'us-east'],
    ['o-1004', 'B-220', 'c3', 'rav@dock.co', 'gold', 'w1', 'us-west'],
    ['o-1004', 'D-450', 'c3', 'rav@dock.co', 'gold', 'w1', 'us-west'],
    ['o-1004', 'C-310', 'c3', 'rav@dock.co', 'gold', 'w1', 'us-west'],
    ['o-1005', 'A-100', 'c1', 'kay@vine.io', 'gold', 'w2', 'us-east'],
  ] as const
).map((r, i) => ({
  order_id: r[0],
  sku: r[1],
  qty: QTY[i],
  cust_id: r[2],
  email: r[3],
  tier: r[4],
  whse: r[5],
  region: r[6],
}));

const PRESETS: Preset[] = [
  {
    id: 'orders',
    label: 'order_line — the full ladder',
    rel: 'order_line',
    attrs: ['order_id', 'sku', 'qty', 'cust_id', 'email', 'tier', 'whse', 'region'],
    fds: [
      { id: 'f1', lhs: ['order_id', 'sku'], rhs: ['qty'], note: 'One quantity per line: this is the key dependency.' },
      { id: 'f2', lhs: ['order_id'], rhs: ['cust_id'], note: 'An order belongs to one customer — a partial dependency on half the key.' },
      { id: 'f3', lhs: ['order_id'], rhs: ['whse'], note: 'An order ships from one warehouse.' },
      { id: 'f4', lhs: ['cust_id'], rhs: ['email'], note: 'Transitive through order_id: the classic 3NF violation.' },
      { id: 'f5', lhs: ['cust_id'], rhs: ['tier'], note: 'Same shape as the email dependency.' },
      { id: 'f6', lhs: ['email'], rhs: ['cust_id'], note: 'Email is unique, so it is a second determinant of the customer.' },
      { id: 'f7', lhs: ['whse'], rhs: ['region'], note: 'Warehouse geography, stapled onto every order line.' },
    ],
    rows: ORDER_ROWS,
    updates: [
      { id: 'u1', label: "customer c1's email changes", setAttr: 'email', whereAttr: 'cust_id', whereVal: 'c1', newVal: 'kay@vine.dev' },
      { id: 'u2', label: 'warehouse w1 is re-homed to us-east', setAttr: 'region', whereAttr: 'whse', whereVal: 'w1', newVal: 'us-east' },
      { id: 'u3', label: 'customer c2 upgrades to gold', setAttr: 'tier', whereAttr: 'cust_id', whereVal: 'c2', newVal: 'gold' },
    ],
  },
  {
    id: 'address',
    label: 'address — 3NF but not BCNF',
    rel: 'address',
    attrs: ['street', 'city', 'state', 'zip'],
    fds: [
      { id: 'g1', lhs: ['street', 'city', 'state'], rhs: ['zip'], note: 'A street address sits in exactly one ZIP.' },
      { id: 'g2', lhs: ['zip'], rhs: ['city'], note: 'A ZIP is inside one city — and city is prime, which is why 3NF forgives this.' },
      { id: 'g3', lhs: ['zip'], rhs: ['state'], note: 'A ZIP is inside one state.' },
    ],
    rows: [
      { street: '12 Pike', city: 'Seattle', state: 'WA', zip: '98101' },
      { street: '44 Pine', city: 'Seattle', state: 'WA', zip: '98101' },
      { street: '9 Alder', city: 'Seattle', state: 'WA', zip: '98104' },
      { street: '70 Elm', city: 'Portland', state: 'OR', zip: '97205' },
      { street: '8 Cedar', city: 'Portland', state: 'OR', zip: '97205' },
      { street: '3 Bay', city: 'Tacoma', state: 'WA', zip: '98402' },
    ],
    updates: [
      { id: 'v1', label: '98101 is renamed to Belltown', setAttr: 'city', whereAttr: 'zip', whereVal: '98101', newVal: 'Belltown' },
      { id: 'v2', label: '97205 moves to state WA', setAttr: 'state', whereAttr: 'zip', whereVal: '97205', newVal: 'WA' },
    ],
  },
  {
    id: 'section',
    label: 'section — BCNF costs a dependency',
    rel: 'section',
    attrs: ['student', 'course', 'instructor'],
    fds: [
      { id: 'h1', lhs: ['student', 'course'], rhs: ['instructor'], note: 'A student takes a course from exactly one instructor.' },
      { id: 'h2', lhs: ['instructor'], rhs: ['course'], note: 'Each instructor teaches exactly one course. Not a superkey — so BCNF says split.' },
    ],
    rows: [
      { student: 'ada', course: 'db', instructor: 'ravi' },
      { student: 'ada', course: 'os', instructor: 'nia' },
      { student: 'kai', course: 'db', instructor: 'ravi' },
      { student: 'kai', course: 'ml', instructor: 'sol' },
      { student: 'mira', course: 'db', instructor: 'tess' },
      { student: 'mira', course: 'os', instructor: 'nia' },
    ],
    updates: [
      { id: 'w1', label: "nia's course is renamed to systems", setAttr: 'course', whereAttr: 'instructor', whereVal: 'nia', newVal: 'systems' },
      { id: 'w2', label: 'ravi hands the db section to tess', setAttr: 'instructor', whereAttr: 'instructor', whereVal: 'ravi', newVal: 'tess' },
    ],
  },
];

/* ------------------------------------------------------------------ instances */

function show(m: Mask, names: string[]) {
  const l = ids(m, names.length).map((i) => names[i]);
  return l.length ? `{${l.join(', ')}}` : '{}';
}

function project(rows: Row[], frag: Mask, names: string[]): Row[] {
  const cols = ids(frag, names.length).map((i) => names[i]);
  const seen = new Set<string>();
  const out: Row[] = [];
  for (const r of rows) {
    const o: Row = {};
    for (const c of cols) o[c] = r[c];
    const k = cols.map((c) => o[c]).join('');
    if (!seen.has(k)) {
      seen.add(k);
      out.push(o);
    }
  }
  return out;
}

function naturalJoin(a: Row[], b: Row[]): Row[] {
  if (!a.length || !b.length) return [];
  const shared = Object.keys(a[0]).filter((k) => k in b[0]);
  const out: Row[] = [];
  for (const x of a) {
    for (const y of b) {
      if (shared.every((k) => x[k] === y[k])) {
        out.push({ ...x, ...y });
        if (out.length >= 400) return out;
      }
    }
  }
  return out;
}

const rowKey = (r: Row, cols: string[]) => cols.map((c) => r[c]).join('');

/* -------------------------------------------------------------------- analysis */

type Frag = { mask: Mask; cols: string[]; rows: Row[]; keys: Mask[]; nf: NF };

type Target = 'flat' | '2nf' | '3nf' | 'bcnf' | 'naive';

function analyze(p: Preset, off: Set<string>, target: Target, naiveP: number) {
  const names = p.attrs;
  const idx = new Map(names.map((a, i) => [a, i] as const));
  const toMask = (l: string[]) => l.reduce((m, a) => m | bit(idx.get(a)!), 0);
  const rel = names.reduce((m, _, i) => m | bit(i), 0);

  const active = p.fds.filter((f) => !off.has(f.id));
  const fds: FD[] = active.map((f) => ({ l: toMask(f.lhs), r: toMask(f.rhs) }));

  const keys = candidateKeys(rel, fds);
  const prime = keys.reduce((a, b) => a | b, 0);
  const relNF = normalForm(rel, fds);

  const steps: Step[] = [];
  let masks: Mask[];
  if (target === 'flat') masks = [rel];
  else if (target === 'naive') masks = naiveSplit(names.map((_, i) => i), naiveP);
  else if (target === '2nf') masks = decompose2NF(rel, fds, steps, names);
  else if (target === '3nf') masks = synth3NF(rel, fds, steps, names);
  else masks = bcnfDecompose(rel, fds, steps, names);

  masks = masks.filter((m, i) => masks.indexOf(m) === i);

  const frags: Frag[] = masks.map((m) => ({
    mask: m,
    cols: ids(m, names.length).map((i) => names[i]),
    rows: project(p.rows, m, names),
    keys: candidateKeys(m, fds),
    nf: normalForm(m, fds),
  }));

  // lossless-join check, run for real against the instance
  let joined: Row[] = frags[0]?.rows ?? [];
  for (let i = 1; i < frags.length; i++) joined = naturalJoin(joined, frags[i].rows);
  const origKeys = new Set(p.rows.map((r) => rowKey(r, names)));
  const joinedFull = joined.filter((r) => names.every((c) => c in r));
  const spurious = joinedFull.filter((r) => !origKeys.has(rowKey(r, names)));
  const lossless = frags.length === 1 || (spurious.length === 0 && joinedFull.length >= p.rows.length);

  // dependency preservation
  const dep = active.map((f) => {
    const fd = { l: toMask(f.lhs), r: toMask(f.rhs) };
    const local = frags.some((g) => ((fd.l | fd.r) & g.mask) === (fd.l | fd.r));
    const kept = isPreserved(fd, masks, fds);
    return { id: f.id, local, kept };
  });
  const lost = dep.filter((d) => !d.kept).length;

  // which declared FDs still violate BCNF inside the relation that holds them
  const violating = new Set<string>();
  for (const f of active) {
    const fd = { l: toMask(f.lhs), r: toMask(f.rhs) };
    for (const g of frags) {
      if (((fd.l | fd.r) & g.mask) !== (fd.l | fd.r)) continue;
      if ((closure(fd.l, fds) & g.mask) !== g.mask) violating.add(f.id);
    }
  }

  // redundancy: copies of the same X→Y fact stored more than once in the flat relation
  let redundant = 0;
  for (const f of active) {
    const fd = { l: toMask(f.lhs), r: toMask(f.rhs) };
    if ((closure(fd.l, fds) & rel) === rel) continue; // determinant is a superkey: no repetition
    const seen = new Set(p.rows.map((r) => f.lhs.map((a) => r[a]).join('')));
    redundant += p.rows.length - seen.size;
  }

  const cellsFlat = p.rows.length * names.length;
  const cellsFrag = frags.reduce((n, g) => n + g.rows.length * g.cols.length, 0);

  return { names, rel, fds, active, keys, prime, relNF, frags, steps, lossless, spurious, dep, lost, violating, redundant, cellsFlat, cellsFrag };
}

/* ---------------------------------------------------------------------- drawing */

const trunc = (s: string, n: number) => (s.length > n ? `${s.slice(0, n - 1)}…` : s);
const colW = (a: string) => Math.max(52, Math.min(96, a.length * 6.6 + 16));

const NF_COLOR: Record<NF, string> = {
  '1NF': 'var(--viz-critical)',
  '2NF': 'var(--viz-serious)',
  '3NF': 'var(--viz-warning)',
  BCNF: 'var(--viz-good)',
};

export default function NormalFormDecomposer() {
  const [presetId, setPresetId] = useState('orders');
  const [target, setTarget] = useState<Target>('flat');
  const [off, setOff] = useState<Set<string>>(new Set());
  const [probe, setProbe] = useState<Mask>(0);
  const [updId, setUpdId] = useState('u1');
  const [naiveP, setNaiveP] = useState(2);
  const [shadeDup, setShadeDup] = useState(true);
  const [showJoin, setShowJoin] = useState(false);
  const [ref, width] = useSize(760);
  const tip = useTip();

  const preset = PRESETS.find((p) => p.id === presetId)!;
  const a = useMemo(() => analyze(preset, off, target, naiveP), [preset, off, target, naiveP]);
  const upd = preset.updates.find((u) => u.id === updId) ?? preset.updates[0];

  const pickPreset = (id: string) => {
    const p = PRESETS.find((q) => q.id === id)!;
    setPresetId(id);
    setOff(new Set());
    setProbe(0);
    setUpdId(p.updates[0].id);
    setNaiveP(2);
    setShowJoin(false);
  };

  const toggleFd = (id: string) =>
    setOff((cur) => {
      const next = new Set(cur);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });

  const toggleAttr = (i: number) => setProbe((m) => m ^ bit(i));

  /* ------------------------------------------------------------ the staged UPDATE */

  const matched = preset.rows.filter((r) => r[upd.whereAttr] === upd.whereVal);
  const matchedSet = new Set(matched.map((r) => rowKey(r, a.names)));
  const fragTouched = a.frags.map((g) => {
    if (!g.cols.includes(upd.setAttr)) return [] as number[];
    const hit: number[] = [];
    g.rows.forEach((fr, i) => {
      const fromMatched = matched.some((r) => g.cols.every((c) => r[c] === fr[c]));
      if (fromMatched) hit.push(i);
    });
    return hit;
  });
  const normRows = fragTouched.reduce((n, h) => n + h.length, 0);
  const flatRows = matched.length;

  /* -------------------------------------------------------------- anomaly probes */

  const nonKeyFd = a.active.find((f) => {
    const idx = new Map(a.names.map((x, i) => [x, i] as const));
    const l = f.lhs.reduce((m, x) => m | bit(idx.get(x)!), 0);
    return (closure(l, a.fds) & a.rel) !== a.rel;
  });
  const delAnomaly = (() => {
    if (!nonKeyFd) return null;
    const counts = new Map<string, number>();
    for (const r of preset.rows) {
      const k = nonKeyFd.lhs.map((x) => r[x]).join(', ');
      counts.set(k, (counts.get(k) ?? 0) + 1);
    }
    for (const [k, n] of counts) if (n === 1) return { k, fd: nonKeyFd };
    return null;
  })();

  /* ------------------------------------------------------------------- geometry */

  const fdH = 22;
  const leftW = 306;
  const topW = Math.max(width, 720);
  const topH = 46 + Math.max(a.active.length, 1) * fdH + 34;

  const cw = Object.fromEntries(a.names.map((n) => [n, colW(n)] as const));
  const rowH = 18;
  const flatW = 34 + a.names.reduce((s, n) => s + cw[n], 0);
  const extraRows = showJoin ? a.spurious.length : 0;
  const flatH = 52 + (preset.rows.length + extraRows) * rowH;

  // lay the fragment grids out left to right, wrapping
  const gap = 18;
  const boxes = a.frags.map((g) => ({ g, w: 10 + g.cols.reduce((s, c) => s + cw[c], 0), h: 44 + g.rows.length * rowH }));
  const lines: { items: typeof boxes; y: number; h: number }[] = [];
  {
    let cur: typeof boxes = [];
    let used = 0;
    const limit = Math.max(topW, 720);
    for (const b of boxes) {
      if (cur.length && used + b.w + gap > limit) {
        lines.push({ items: cur, y: 0, h: 0 });
        cur = [];
        used = 0;
      }
      cur.push(b);
      used += b.w + gap;
    }
    if (cur.length) lines.push({ items: cur, y: 0, h: 0 });
  }
  let cursor = flatH + 26;
  for (const ln of lines) {
    ln.y = cursor;
    ln.h = Math.max(...ln.items.map((b) => b.h));
    cursor += ln.h + 24;
  }
  const botW = Math.max(topW, flatW + 12, ...lines.map((l) => l.items.reduce((s, b) => s + b.w + gap, 0)));
  const botH = cursor + 4;

  const dupCells = useMemo(() => {
    // second and later copies of a non-key determinant's dependent values
    const marks = new Set<string>();
    if (!shadeDup) return marks;
    const idx = new Map(a.names.map((x, i) => [x, i] as const));
    for (const f of a.active) {
      const l = f.lhs.reduce((m, x) => m | bit(idx.get(x)!), 0);
      if ((closure(l, a.fds) & a.rel) === a.rel) continue;
      const seen = new Set<string>();
      preset.rows.forEach((r, ri) => {
        const k = f.lhs.map((x) => r[x]).join('');
        if (seen.has(k)) for (const c of f.rhs) marks.add(`${ri}:${c}`);
        else seen.add(k);
      });
    }
    return marks;
  }, [a, preset, shadeDup]);

  const note = (() => {
    if (target === 'flat')
      return a.relNF === 'BCNF'
        ? `${preset.rel} is already in BCNF under the dependencies you left switched on: every determinant is a superkey, so there is nothing to split.`
        : `${preset.rel} is in ${a.relNF}. ${a.redundant} stored fact${a.redundant === 1 ? ' is a redundant copy' : 's are redundant copies'} — the highlighted cells repeat something the schema already told you. Pick a decomposition target to remove them.`;
    if (target === 'naive') {
      const shared = show((a.frags[0]?.mask ?? 0) & (a.frags[1]?.mask ?? a.frags[0]?.mask ?? 0), a.names);
      return a.lossless
        ? `This split happens to be lossless: the shared attribute ${shared} is a key of one side, so the join reproduces the original rows. Slide the split point and watch that stop being true.`
        : `Lossy. The two fragments share ${shared}, which is a key of neither, so joining them back invents ${a.spurious.length} row${a.spurious.length === 1 ? '' : 's'} that were never in the table. Tick "join the fragments back" to see them marked ✕. This is the failure the lossless-join test exists to prevent.`;
    }
    const lostNames = a.dep.filter((d) => !d.kept).map((d) => d.id);
    const lostText = lostNames.length
      ? ` But ${lostNames
          .map((id) => {
            const f = a.active.find((x) => x.id === id)!;
            return `${show(f.lhs.reduce((m, x) => m | bit(a.names.indexOf(x)), 0), a.names)}→${show(f.rhs.reduce((m, x) => m | bit(a.names.indexOf(x)), 0), a.names)}`;
          })
          .join(', ')} is no longer checkable inside any one relation: enforcing it now needs a join, a trigger, or a materialized helper table.`
      : a.dep.every((d) => d.local)
        ? ' Every declared dependency still fits inside a single relation, so each one is enforceable as a key or a unique constraint.'
        : ' No dependency was lost: the ones that no longer fit in a single relation are still implied by the projections taken together.';
    return (
      `${a.steps.length} step${a.steps.length === 1 ? '' : 's'}: ${a.frags.length} relation${a.frags.length === 1 ? '' : 's'}, ` +
      `the weakest of them in ${a.frags.map((g) => g.nf).sort()[0]}. ` +
      `${a.lossless ? 'Joining the fragments back reproduces the original rows exactly — the decomposition is lossless.' : `The join produces ${a.spurious.length} spurious rows.`}` +
      lostText
    );
  })();

  return (
    <VizPanel
      title="Functional dependencies, decomposed"
      subtitle="Click a dependency to declare or retract it; click an attribute to build a probe set X and read its closure. Everything below — candidate keys, the normal form, the split, the lossless-join check, the dependency-preservation test — is recomputed from the dependencies that are on."
      controls={
        <>
          <Choice
            label="Schema"
            value={presetId}
            onChange={pickPreset}
            options={PRESETS.map((p) => ({ value: p.id, label: p.label }))}
          />
          <Segmented
            label="Decompose to"
            value={target}
            onChange={(v) => setTarget(v)}
            options={[
              { value: 'flat', label: 'flat', title: 'One wide relation — where every schema starts' },
              { value: '2nf', label: '2NF', title: 'Lift out dependencies on part of the key' },
              { value: '3nf', label: '3NF', title: "Bernstein synthesis from a minimal cover — always dependency-preserving" },
              { value: 'bcnf', label: 'BCNF', title: 'Recursive split until every determinant is a superkey' },
              { value: 'naive', label: 'naive', title: 'Split anywhere — and watch the join go wrong' },
            ]}
          />
          <Slider
            label="Naive split after column"
            min={2}
            max={Math.max(2, preset.attrs.length - 1)}
            value={Math.min(naiveP, preset.attrs.length - 1)}
            onChange={setNaiveP}
            disabled={target !== 'naive'}
            format={(n) => preset.attrs[n - 1]}
          />
          <Choice
            label="Stage an UPDATE"
            value={upd.id}
            onChange={setUpdId}
            options={preset.updates.map((u) => ({ value: u.id, label: u.label }))}
          />
          <Check label="Shade redundant copies" checked={shadeDup} onChange={setShadeDup} />
          <Check label="Join the fragments back" checked={showJoin} onChange={setShowJoin} />
          <Button onClick={() => setProbe(0)} disabled={probe === 0}>
            Clear X
          </Button>
          <Button
            onClick={() => {
              setOff(new Set());
              setTarget('flat');
              setProbe(0);
              setShowJoin(false);
            }}
          >
            Reset
          </Button>
        </>
      }
      legend={
        <Legend
          items={[
            { label: 'X — probe set for the closure', color: 'var(--viz-1)' },
            { label: 'rewritten by the staged UPDATE', color: 'var(--viz-2)' },
            { label: 'redundant copy of a stored fact', color: 'var(--viz-neutral)' },
            { label: '✕ spurious row invented by the join', color: 'var(--viz-critical)' },
            { label: '✓ relation in BCNF', color: 'var(--viz-good)' },
            { label: '! dependency lost by the split', color: 'var(--viz-warning)' },
          ]}
        />
      }
      stats={
        <Stats
          items={[
            { label: 'Candidate keys', value: a.keys.length ? a.keys.map((k) => show(k, a.names)).join('  ') : 'none', hint: 'Minimal superkeys of the whole relation under the dependencies you left on' },
            {
              label: probe ? `${show(probe, a.names)}⁺` : 'X⁺',
              value: probe ? show(closure(probe, a.fds) & a.rel, a.names) : 'pick attributes',
              hint: 'Attribute closure: everything this set functionally determines',
            },
            {
              label: 'X is a…',
              value: !probe
                ? '—'
                : (closure(probe, a.fds) & a.rel) !== a.rel
                  ? 'not a superkey'
                  : a.keys.some((k) => k === probe)
                    ? 'candidate key'
                    : 'superkey, not minimal',
            },
            { label: 'Relations / normal form', value: `${a.frags.length} · ${a.frags.every((g) => g.nf === 'BCNF') ? 'BCNF' : a.frags.map((g) => g.nf).sort()[0]}` },
            {
              label: 'Lossless join',
              value: a.lossless ? 'yes' : `no · +${a.spurious.length} spurious`,
              hint: 'Computed by actually joining the fragment instances back together and diffing against the original rows',
            },
            { label: 'Dependencies preserved', value: `${a.dep.length - a.lost} of ${a.dep.length}`, hint: 'Implied by the union of the projections — the rest can only be checked with a join' },
            { label: 'Rows rewritten by the UPDATE', value: `${flatRows} → ${normRows}`, hint: 'Flat relation vs the current decomposition' },
            { label: 'Cells stored', value: `${fmtNum(a.cellsFlat)} → ${fmtNum(a.cellsFrag)}` },
          ]}
        />
      }
      note={
        <Note>
          {note}
          {delAnomaly && nonKeyFd ? (
            <>
              {' '}
              <strong>Deletion anomaly:</strong> {nonKeyFd.lhs.join(', ')} = {delAnomaly.k} appears in exactly one flat row, so
              deleting that row also deletes the only record of {nonKeyFd.rhs.join(', ')} for it — and the mirror image, the
              insertion anomaly, means you cannot record that fact until something supplies a value for every attribute of{' '}
              {a.keys[0] ? show(a.keys[0], a.names) : 'the key'}.
            </>
          ) : null}
        </Note>
      }
      table={
        <>
          <table className="viz-table">
            <thead>
              <tr>
                <th>Dependency</th>
                <th>Declared</th>
                <th>LHS closure</th>
                <th>Determinant is a superkey?</th>
                <th>In one relation?</th>
                <th>Preserved?</th>
              </tr>
            </thead>
            <tbody>
              {preset.fds.map((f) => {
                const l = f.lhs.reduce((m, x) => m | bit(a.names.indexOf(x)), 0);
                const d = a.dep.find((x) => x.id === f.id);
                const on = !off.has(f.id);
                return (
                  <tr key={f.id}>
                    <td>
                      {f.lhs.join(', ')} → {f.rhs.join(', ')}
                    </td>
                    <td>{on ? 'yes' : 'retracted'}</td>
                    <td>{on ? show(closure(l, a.fds) & a.rel, a.names) : '—'}</td>
                    <td>{on ? ((closure(l, a.fds) & a.rel) === a.rel ? 'yes' : 'no') : '—'}</td>
                    <td>{d ? (d.local ? 'yes' : 'no') : '—'}</td>
                    <td>{d ? (d.kept ? 'yes' : 'LOST') : '—'}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
          <table className="viz-table">
            <thead>
              <tr>
                <th>Relation</th>
                <th>Attributes</th>
                <th>Candidate keys</th>
                <th>Normal form</th>
                <th>Rows</th>
                <th>Rows the UPDATE rewrites</th>
              </tr>
            </thead>
            <tbody>
              {a.frags.map((g, i) => (
                <tr key={g.mask}>
                  <td>R{i + 1}</td>
                  <td>{g.cols.join(', ')}</td>
                  <td>{g.keys.map((k) => show(k, a.names)).join('  ')}</td>
                  <td>{g.nf}</td>
                  <td>{g.rows.length}</td>
                  <td>{fragTouched[i].length}</td>
                </tr>
              ))}
              <tr>
                <td>flat {preset.rel}</td>
                <td>{a.names.join(', ')}</td>
                <td>{a.keys.map((k) => show(k, a.names)).join('  ')}</td>
                <td>{a.relNF}</td>
                <td>{preset.rows.length}</td>
                <td>{flatRows}</td>
              </tr>
            </tbody>
          </table>
        </>
      }
    >
      <div ref={ref}>
        <TooltipHost>
          <svg width={topW} height={topH} role="img" aria-label="Declared functional dependencies and the attribute closure probe">
            <text x={0} y={12} fill="var(--viz-ink)" fontWeight={600}>
              Declared dependencies — click to retract
            </text>
            {a.active.length === 0 ? (
              <text x={0} y={40} fill="var(--viz-ink-muted)">
                Every dependency is retracted: the only key is the whole row.
              </text>
            ) : null}
            {preset.fds.map((f, i) => {
              const on = !off.has(f.id);
              const d = a.dep.find((x) => x.id === f.id);
              const bad = a.violating.has(f.id);
              const lost = d && !d.kept;
              const color = !on
                ? 'var(--viz-stale)'
                : lost
                  ? 'var(--viz-warning)'
                  : bad
                    ? 'var(--viz-critical)'
                    : 'var(--viz-good)';
              const glyph = !on ? '○' : lost ? '!' : bad ? '✕' : '✓';
              const y = 24 + i * fdH;
              return (
                <g
                  key={f.id}
                  role="button"
                  aria-pressed={on}
                  onClick={() => toggleFd(f.id)}
                  onKeyDown={(e) => {
                    if (e.key === 'Enter' || e.key === ' ') {
                      e.preventDefault();
                      toggleFd(f.id);
                    }
                  }}
                  style={{ cursor: 'pointer' }}
                  {...tip(
                    <>
                      <strong>
                        {f.lhs.join(', ')} → {f.rhs.join(', ')}
                      </strong>
                      <br />
                      {f.note}
                      <br />
                      {on
                        ? bad
                          ? 'Its determinant is not a superkey of the relation holding it — this is the BCNF violation.'
                          : lost
                            ? 'Not implied by the projections: only a join can check it now.'
                            : 'Enforceable inside one relation as a key or unique constraint.'
                        : 'Retracted — click to declare it again.'}
                    </>,
                  )}
                >
                  <rect
                    x={0}
                    y={y}
                    width={leftW}
                    height={fdH - 4}
                    rx={5}
                    fill="var(--viz-plane)"
                    stroke={color}
                    strokeDasharray={on ? undefined : '3 3'}
                  />
                  <text x={8} y={y + 13} fill={color} fontWeight={700}>
                    {glyph}
                  </text>
                  <text x={24} y={y + 13} fill={on ? 'var(--viz-ink)' : 'var(--viz-ink-muted)'}>
                    {f.lhs.join(', ')}
                  </text>
                  <text x={24 + f.lhs.join(', ').length * 6.4 + 6} y={y + 13} fill="var(--viz-ink-muted)">
                    →
                  </text>
                  <text x={24 + f.lhs.join(', ').length * 6.4 + 20} y={y + 13} fill={on ? 'var(--viz-ink-2)' : 'var(--viz-ink-muted)'}>
                    {f.rhs.join(', ')}
                  </text>
                </g>
              );
            })}

            <text x={leftW + 24} y={12} fill="var(--viz-ink)" fontWeight={600}>
              {preset.rel} — click an attribute to put it in X
            </text>
            {(() => {
              let x = leftW + 24;
              let row = 0;
              return a.names.map((n, i) => {
                const w = n.length * 6.6 + 18;
                if (x + w > topW - 4) {
                  x = leftW + 24;
                  row++;
                }
                const cx = x;
                x += w + 6;
                const inX = hasAttr(probe, i);
                const isPrime = hasAttr(a.prime, i);
                return (
                  <g
                    key={n}
                    role="button"
                    aria-pressed={inX}
                    onClick={() => toggleAttr(i)}
                    onKeyDown={(e) => {
                      if (e.key === 'Enter' || e.key === ' ') {
                        e.preventDefault();
                        toggleAttr(i);
                      }
                    }}
                    style={{ cursor: 'pointer' }}
                    {...tip(
                      <>
                        <strong>{n}</strong>
                        <br />
                        {isPrime ? 'Prime: it belongs to at least one candidate key.' : 'Non-prime: it is in no candidate key.'}
                      </>,
                    )}
                  >
                    <rect
                      x={cx}
                      y={24 + row * 26}
                      width={w}
                      height={20}
                      rx={10}
                      fill={inX ? 'var(--viz-1)' : 'var(--viz-plane)'}
                      stroke={isPrime ? 'var(--viz-1)' : 'var(--viz-border)'}
                    />
                    <text
                      x={cx + w / 2}
                      y={24 + row * 26 + 14}
                      textAnchor="middle"
                      fill={inX ? 'var(--viz-surface)' : 'var(--viz-ink)'}
                      fontWeight={isPrime ? 600 : 400}
                    >
                      {n}
                    </text>
                  </g>
                );
              });
            })()}
            <text x={leftW + 24} y={topH - 42} fill="var(--viz-ink-2)">
              {probe
                ? `${show(probe, a.names)}⁺ = ${show(closure(probe, a.fds) & a.rel, a.names)}`
                : 'X⁺ = — (pick one or more attributes above)'}
            </text>
            <text x={leftW + 24} y={topH - 26} fill="var(--viz-ink-2)">
              candidate keys: {a.keys.map((k) => show(k, a.names)).join('  ')} · bold attributes are prime
            </text>
            <text x={leftW + 24} y={topH - 10} fill={NF_COLOR[a.relNF]} fontWeight={600}>
              {preset.rel} as one relation is in {a.relNF}
            </text>
          </svg>

          <svg
            width={botW}
            height={botH}
            role="img"
            aria-label="The flat instance and the decomposed fragments, with the rows a single UPDATE has to rewrite"
          >
            <text x={0} y={12} fill="var(--viz-ink)" fontWeight={600}>
              flat {preset.rel} — {preset.rows.length} rows × {a.names.length} columns
            </text>
            <text x={0} y={28} fill="var(--viz-ink-muted)">
              UPDATE {preset.rel} SET {upd.setAttr} = '{upd.newVal}' WHERE {upd.whereAttr} = '{upd.whereVal}' → {flatRows} row
              {flatRows === 1 ? '' : 's'}
            </text>
            {(() => {
              const out: ReactNode[] = [];
              let x = 34;
              for (const n of a.names) {
                const keyAttr = hasAttr(a.keys[0] ?? 0, a.names.indexOf(n));
                out.push(
                  <g key={`h-${n}`}>
                    <text x={x + 4} y={46} fill="var(--viz-ink)" fontWeight={keyAttr ? 700 : 500}>
                      {trunc(n, 11)}
                    </text>
                    {keyAttr ? (
                      <line x1={x + 4} y1={49} x2={x + Math.min(cw[n] - 8, n.length * 6.4)} y2={49} stroke="var(--viz-1)" strokeWidth={1.5} />
                    ) : null}
                  </g>,
                );
                x += cw[n];
              }
              return out;
            })()}
            {preset.rows.map((r, ri) => {
              const hit = matchedSet.has(rowKey(r, a.names));
              const y = 54 + ri * rowH;
              let x = 34;
              return (
                <g key={`r-${ri}`}>
                  <text x={0} y={y + 13} fill="var(--viz-ink-muted)">
                    {ri + 1}
                  </text>
                  {a.names.map((n) => {
                    const cx = x;
                    x += cw[n];
                    const dup = dupCells.has(`${ri}:${n}`);
                    const isTargetCell = hit && n === upd.setAttr;
                    return (
                      <g key={n}>
                        <rect
                          x={cx}
                          y={y}
                          width={cw[n] - 3}
                          height={rowH - 3}
                          rx={3}
                          fill={isTargetCell ? 'var(--viz-2)' : dup ? 'var(--viz-neutral)' : 'var(--viz-plane)'}
                          stroke={hit ? 'var(--viz-2)' : 'var(--viz-border)'}
                          strokeWidth={hit ? 1.2 : 1}
                        />
                        <text x={cx + 4} y={y + 11} fill={isTargetCell ? 'var(--viz-surface)' : 'var(--viz-ink)'}>
                          {trunc(String(r[n]), Math.floor((cw[n] - 8) / 6.2))}
                        </text>
                      </g>
                    );
                  })}
                </g>
              );
            })}
            {showJoin
              ? a.spurious.map((r, si) => {
                  const y = 54 + (preset.rows.length + si) * rowH;
                  let x = 34;
                  return (
                    <g key={`s-${si}`}>
                      <text x={0} y={y + 13} fill="var(--viz-critical)" fontWeight={700}>
                        ✕
                      </text>
                      {a.names.map((n) => {
                        const cx = x;
                        x += cw[n];
                        return (
                          <g key={n}>
                            <rect x={cx} y={y} width={cw[n] - 3} height={rowH - 3} rx={3} fill="var(--viz-plane)" stroke="var(--viz-critical)" strokeDasharray="3 2" />
                            <text x={cx + 4} y={y + 11} fill="var(--viz-critical)">
                              {trunc(String(r[n]), Math.floor((cw[n] - 8) / 6.2))}
                            </text>
                          </g>
                        );
                      })}
                    </g>
                  );
                })
              : null}

            {lines.map((ln, li) => {
              let bx = 0;
              return (
                <g key={li}>
                  {ln.items.map(({ g, w, h }) => {
                    const i = a.frags.indexOf(g);
                    const x0 = bx;
                    bx += w + gap;
                    const touched = new Set(fragTouched[i]);
                    let cx = x0 + 5;
                    return (
                      <g key={g.mask}>
                        <rect x={x0} y={ln.y} width={w} height={h} rx={8} fill="var(--viz-plane)" stroke="var(--viz-border)" />
                        <g
                          {...tip(
                            <>
                              <strong>R{i + 1}({g.cols.join(', ')})</strong>
                              <br />
                              candidate keys: {g.keys.map((k) => show(k, a.names)).join(' ')}
                              <br />
                              {g.nf === 'BCNF'
                                ? 'Every determinant here is a superkey — nothing left to split.'
                                : `Still only in ${g.nf}: some determinant is neither a superkey nor made of prime attributes.`}
                            </>,
                          )}
                        >
                          <text x={x0 + 6} y={ln.y + 14} fill="var(--viz-ink)" fontWeight={600}>
                            R{i + 1}
                          </text>
                          <text x={x0 + 26} y={ln.y + 14} fill={NF_COLOR[g.nf]} fontWeight={600}>
                            {g.nf === 'BCNF' ? '✓ BCNF' : g.nf}
                          </text>
                          <text x={x0 + w - 6} y={ln.y + 14} textAnchor="end" fill="var(--viz-ink-muted)">
                            {g.rows.length} rows
                          </text>
                        </g>
                        {g.cols.map((c) => {
                          const kx = cx;
                          cx += cw[c];
                          const keyAttr = hasAttr(g.keys[0] ?? 0, a.names.indexOf(c));
                          return (
                            <g key={c}>
                              <text x={kx + 4} y={ln.y + 30} fill="var(--viz-ink)" fontWeight={keyAttr ? 700 : 500}>
                                {trunc(c, 11)}
                              </text>
                              {keyAttr ? (
                                <line x1={kx + 4} y1={ln.y + 33} x2={kx + Math.min(cw[c] - 8, c.length * 6.4)} y2={ln.y + 33} stroke="var(--viz-1)" strokeWidth={1.5} />
                              ) : null}
                            </g>
                          );
                        })}
                        {g.rows.map((fr, fi) => {
                          let vx = x0 + 5;
                          const hit = touched.has(fi);
                          const y = ln.y + 38 + fi * rowH;
                          return (
                            <g key={fi}>
                              {g.cols.map((c) => {
                                const kx = vx;
                                vx += cw[c];
                                const isTarget = hit && c === upd.setAttr;
                                return (
                                  <g key={c}>
                                    <rect
                                      x={kx}
                                      y={y}
                                      width={cw[c] - 3}
                                      height={rowH - 3}
                                      rx={3}
                                      fill={isTarget ? 'var(--viz-2)' : 'var(--viz-surface)'}
                                      stroke={hit ? 'var(--viz-2)' : 'var(--viz-border)'}
                                    />
                                    <text x={kx + 4} y={y + 11} fill={isTarget ? 'var(--viz-surface)' : 'var(--viz-ink)'}>
                                      {trunc(String(fr[c]), Math.floor((cw[c] - 8) / 6.2))}
                                    </text>
                                  </g>
                                );
                              })}
                            </g>
                          );
                        })}
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
