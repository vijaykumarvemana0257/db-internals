import { useMemo, useState } from 'react';
import {
  VizPanel,
  Slider,
  Segmented,
  Choice,
  Check,
  Button,
  Legend,
  Stats,
  Note,
  TooltipHost,
  useTip,
  fmtNum,
  makeRng,
  useSize,
} from './Viz';

/**
 * One SELECT, all five stages, on a toy two-table catalog.
 *
 * Everything here is a real mechanism rather than a picture of one:
 *  - a hand-written lexer + recursive-descent parser produces a raw parse tree
 *    (SelectStmt / RangeVar / ColumnRef / A_Expr), with PostgreSQL-shaped syntax errors;
 *  - analysis resolves names against a toy pg_class/pg_attribute and turns ColumnRefs
 *    into Vars and A_Exprs into OpExprs, with PostgreSQL-shaped semantic errors;
 *  - the rewriter expands the view big_orders the way ON SELECT DO INSTEAD rules do;
 *  - the planner estimates selectivity from a toy pg_statistic (MCVs + a histogram),
 *    costs seq / index / bitmap paths and hash / nested-loop joins with PostgreSQL's
 *    own cost formulas (including the Mackert-Lohman page estimate and the
 *    correlation interpolation in cost_index), and keeps every rejected path;
 *  - the executor is an actual Volcano iterator tree over 120,000 deterministic rows,
 *    where every tuple access goes through an LRU buffer pool that counts hits,
 *    misses and evictions per operator.
 *
 * So flipping an index or dragging random_page_cost changes the chosen plan, and the
 * execution counters move with it, because the same numbers drive both.
 */

/* ==================================================================== catalog */

type RelName = 'customers' | 'orders';
type Ty = 'integer' | 'text' | 'numeric';

type Col = {
  name: string;
  type: Ty;
  attnum: number;
  width: number;
  ndistinct: number;
  mcv?: { v: string; f: number }[];
  range?: [number, number];
};

type Rel = {
  oid: number;
  name: string;
  kind: 'r' | 'v';
  reltuples: number;
  relpages: number;
  perPage: number;
  cols: Col[];
  baseRel?: RelName;
  viewQual?: { col: string; op: string; val: number };
};

const CITIES = [
  'Seattle',
  'Portland',
  'Denver',
  'Boise',
  'Austin',
  'Tacoma',
  'Spokane',
  'Salem',
  'Eugene',
  'Bend',
  'Yakima',
  'Olympia',
  'Kirkland',
];
const CITY_F = [0.22, 0.16, 0.12, 0.1, 0.08, 0.07, 0.06, 0.05, 0.045, 0.04, 0.03, 0.017, 0.008];
const STATUSES = ['shipped', 'pending', 'cancelled'];
const STATUS_F = [0.62, 0.26, 0.12];

const CUSTOMERS: Rel = {
  oid: 16385,
  name: 'customers',
  kind: 'r',
  reltuples: 20_000,
  relpages: 223,
  perPage: 90,
  cols: [
    { name: 'id', type: 'integer', attnum: 1, width: 4, ndistinct: 20_000 },
    { name: 'name', type: 'text', attnum: 2, width: 21, ndistinct: 20_000 },
    { name: 'email', type: 'text', attnum: 3, width: 27, ndistinct: 20_000 },
    {
      name: 'city',
      type: 'text',
      attnum: 4,
      width: 9,
      ndistinct: CITIES.length,
      mcv: CITIES.map((v, i) => ({ v, f: CITY_F[i] })),
    },
  ],
};

const ORDERS: Rel = {
  oid: 16390,
  name: 'orders',
  kind: 'r',
  reltuples: 100_000,
  relpages: 770,
  perPage: 130,
  cols: [
    { name: 'id', type: 'integer', attnum: 1, width: 4, ndistinct: 100_000 },
    { name: 'customer_id', type: 'integer', attnum: 2, width: 4, ndistinct: 20_000 },
    { name: 'amount', type: 'numeric', attnum: 3, width: 8, ndistinct: 2001, range: [0, 2000] },
    {
      name: 'status',
      type: 'text',
      attnum: 4,
      width: 10,
      ndistinct: 3,
      mcv: STATUSES.map((v, i) => ({ v, f: STATUS_F[i] })),
    },
  ],
};

const BIG_ORDERS: Rel = {
  oid: 16401,
  name: 'big_orders',
  kind: 'v',
  reltuples: 25_000,
  relpages: 0,
  perPage: 0,
  cols: ORDERS.cols,
  baseRel: 'orders',
  viewQual: { col: 'amount', op: '>', val: 1500 },
};

const CATALOG: Record<string, Rel> = {
  customers: CUSTOMERS,
  orders: ORDERS,
  big_orders: BIG_ORDERS,
};

type Idx = {
  name: string;
  rel: RelName;
  col: string;
  pages: number;
  perLeaf: number;
  corr: number;
  unique: boolean;
  toggle?: 'city' | 'cust' | 'amt';
};

const INDEXES: Idx[] = [
  { name: 'customers_pkey', rel: 'customers', col: 'id', pages: 45, perLeaf: 450, corr: 1, unique: true },
  { name: 'customers_city_idx', rel: 'customers', col: 'city', pages: 90, perLeaf: 230, corr: 0.05, unique: false, toggle: 'city' },
  { name: 'orders_pkey', rel: 'orders', col: 'id', pages: 220, perLeaf: 450, corr: 1, unique: true },
  { name: 'orders_customer_id_idx', rel: 'orders', col: 'customer_id', pages: 280, perLeaf: 360, corr: 0.02, unique: false, toggle: 'cust' },
  { name: 'orders_amount_idx', rel: 'orders', col: 'amount', pages: 280, perLeaf: 360, corr: 0.03, unique: false, toggle: 'amt' },
];

const OPNAME: Record<string, Record<Ty, string>> = {
  '=': { integer: 'int4eq', text: 'texteq', numeric: 'numeric_eq' },
  '<>': { integer: 'int4ne', text: 'textne', numeric: 'numeric_ne' },
  '<': { integer: 'int4lt', text: 'text_lt', numeric: 'numeric_lt' },
  '<=': { integer: 'int4le', text: 'text_le', numeric: 'numeric_le' },
  '>': { integer: 'int4gt', text: 'text_gt', numeric: 'numeric_gt' },
  '>=': { integer: 'int4ge', text: 'text_ge', numeric: 'numeric_ge' },
};

/* ======================================================================= data */

type Data = {
  custCity: Uint8Array;
  cityOff: Int32Array; // prefix offsets into custByCity
  custByCity: Int32Array;
  ordCust: Int32Array;
  ordAmt: Int16Array;
  ordStatus: Uint8Array;
  custOff: Int32Array; // prefix offsets into ordByCust, one per customer id
  ordByCust: Int32Array;
  ordByAmt: Int32Array; // order rows sorted by (amount, ctid) — btree leaf order
  amtStart: Int32Array; // first position in ordByAmt with amount >= v
};

let DATA: Data | null = null;

function pick(r: number, freqs: number[]) {
  let acc = 0;
  for (let i = 0; i < freqs.length; i++) {
    acc += freqs[i];
    if (r < acc) return i;
  }
  return freqs.length - 1;
}

function buildData(): Data {
  const rng = makeRng(19960708); // any fixed seed — SSR and hydration must agree
  const nc = CUSTOMERS.reltuples;
  const no = ORDERS.reltuples;

  const custCity = new Uint8Array(nc);
  for (let i = 0; i < nc; i++) custCity[i] = pick(rng(), CITY_F);

  const cityCount = new Int32Array(CITIES.length);
  for (let i = 0; i < nc; i++) cityCount[custCity[i]]++;
  const cityOff = new Int32Array(CITIES.length + 1);
  for (let k = 0; k < CITIES.length; k++) cityOff[k + 1] = cityOff[k] + cityCount[k];
  const custByCity = new Int32Array(nc);
  const fill = Int32Array.from(cityOff.subarray(0, CITIES.length));
  for (let i = 0; i < nc; i++) custByCity[fill[custCity[i]]++] = i;

  const ordCust = new Int32Array(no);
  const ordAmt = new Int16Array(no);
  const ordStatus = new Uint8Array(no);
  for (let i = 0; i < no; i++) {
    ordCust[i] = 1 + Math.floor(rng() * nc);
    ordAmt[i] = Math.floor(rng() * 2001);
    ordStatus[i] = pick(rng(), STATUS_F);
  }

  const custCnt = new Int32Array(nc);
  for (let i = 0; i < no; i++) custCnt[ordCust[i] - 1]++;
  const custOff = new Int32Array(nc + 1);
  for (let k = 0; k < nc; k++) custOff[k + 1] = custOff[k] + custCnt[k];
  const ordByCust = new Int32Array(no);
  const fill2 = Int32Array.from(custOff.subarray(0, nc));
  for (let i = 0; i < no; i++) ordByCust[fill2[ordCust[i] - 1]++] = i;

  const amtCnt = new Int32Array(2001);
  for (let i = 0; i < no; i++) amtCnt[ordAmt[i]]++;
  const amtStart = new Int32Array(2002);
  for (let v = 0; v < 2001; v++) amtStart[v + 1] = amtStart[v] + amtCnt[v];
  const ordByAmt = new Int32Array(no);
  const fill3 = Int32Array.from(amtStart.subarray(0, 2001));
  for (let i = 0; i < no; i++) ordByAmt[fill3[ordAmt[i]]++] = i;

  return { custCity, cityOff, custByCity, ordCust, ordAmt, ordStatus, custOff, ordByCust, ordByAmt, amtStart };
}

function data(): Data {
  if (!DATA) DATA = buildData();
  return DATA;
}

function valOf(rel: RelName, col: string, row: number, D: Data): number | string {
  if (rel === 'customers') {
    if (col === 'id') return row + 1;
    if (col === 'name') return `Customer ${row + 1}`;
    if (col === 'email') return `c${row + 1}@example.net`;
    return CITIES[D.custCity[row]];
  }
  if (col === 'id') return row + 1;
  if (col === 'customer_id') return D.ordCust[row];
  if (col === 'amount') return D.ordAmt[row];
  return STATUSES[D.ordStatus[row]];
}

/* ====================================================================== lexer */

type Tok = { t: 'word' | 'num' | 'str' | 'punct' | 'op'; v: string; pos: number };

class SqlError extends Error {
  pos: number;
  constructor(msg: string, pos: number) {
    super(msg);
    this.pos = pos;
  }
}

const KEYWORDS = new Set([
  'select',
  'from',
  'where',
  'and',
  'or',
  'join',
  'inner',
  'on',
  'limit',
  'as',
  'order',
  'group',
  'having',
  'by',
  'union',
  'left',
  'right',
  'outer',
  'distinct',
]);

function lex(s: string): Tok[] {
  const out: Tok[] = [];
  let i = 0;
  while (i < s.length) {
    const c = s[i];
    if (/\s/.test(c)) {
      i++;
      continue;
    }
    if (c === "'") {
      let j = i + 1;
      let v = '';
      for (;;) {
        if (j >= s.length) throw new SqlError('unterminated quoted string', i);
        if (s[j] === "'") {
          if (s[j + 1] === "'") {
            v += "'";
            j += 2;
            continue;
          }
          j++;
          break;
        }
        v += s[j++];
      }
      out.push({ t: 'str', v, pos: i });
      i = j;
      continue;
    }
    if (/[0-9]/.test(c)) {
      let j = i;
      while (j < s.length && /[0-9.]/.test(s[j])) j++;
      out.push({ t: 'num', v: s.slice(i, j), pos: i });
      i = j;
      continue;
    }
    if (/[A-Za-z_]/.test(c)) {
      let j = i;
      while (j < s.length && /[A-Za-z0-9_$]/.test(s[j])) j++;
      out.push({ t: 'word', v: s.slice(i, j), pos: i });
      i = j;
      continue;
    }
    const two = s.slice(i, i + 2);
    if (two === '<=' || two === '>=' || two === '<>' || two === '!=') {
      out.push({ t: 'op', v: two === '!=' ? '<>' : two, pos: i });
      i += 2;
      continue;
    }
    if (c === '=' || c === '<' || c === '>') {
      out.push({ t: 'op', v: c, pos: i });
      i++;
      continue;
    }
    if (',.();*'.includes(c)) {
      out.push({ t: 'punct', v: c, pos: i });
      i++;
      continue;
    }
    throw new SqlError(`syntax error at or near "${c}"`, i);
  }
  return out;
}

/* ===================================================================== parser */

type ColRef = { qual?: string; name: string; pos: number };
type Konst = { kind: 'num' | 'str'; num?: number; str?: string; text: string; pos: number };
type AExpr = { op: string; left: ColRef; right: ColRef | Konst; pos: number };
type FromItem = { rel: string; alias?: string; pos: number };

type Ast = {
  targets: { star: boolean; ref?: ColRef }[];
  from: FromItem[];
  explicitJoin: boolean;
  joinOn?: AExpr;
  where: AExpr[];
  limit?: number;
};

function isConst(x: ColRef | Konst): x is Konst {
  return 'kind' in x;
}

function parse(sql: string): Ast {
  const toks = lex(sql);
  let p = 0;
  const at = () => toks[p];
  const eof = () => p >= toks.length;
  const near = () => (eof() ? `syntax error at end of input` : `syntax error at or near "${toks[p].v}"`);
  const pos = () => (eof() ? sql.length : toks[p].pos);
  const fail = (): never => {
    throw new SqlError(near(), pos());
  };
  const isWord = (w: string) => !eof() && at().t === 'word' && at().v.toLowerCase() === w;
  const takeWord = (w: string) => {
    if (!isWord(w)) fail();
    p++;
  };
  const takePunct = (c: string) => {
    if (eof() || at().t !== 'punct' || at().v !== c) fail();
    p++;
  };

  const colRef = (): ColRef => {
    if (eof() || at().t !== 'word') fail();
    const first = toks[p];
    p++;
    if (!eof() && at().t === 'punct' && at().v === '.') {
      p++;
      if (eof() || at().t !== 'word') fail();
      const second = toks[p];
      p++;
      return { qual: first.v, name: second.v, pos: first.pos };
    }
    return { name: first.v, pos: first.pos };
  };

  const operand = (): ColRef | Konst => {
    if (eof()) fail();
    if (at().t === 'str') {
      const t = toks[p++];
      return { kind: 'str', str: t.v, text: `'${t.v}'`, pos: t.pos };
    }
    if (at().t === 'num') {
      const t = toks[p++];
      return { kind: 'num', num: Number(t.v), text: t.v, pos: t.pos };
    }
    return colRef();
  };

  const aExpr = (): AExpr => {
    const left = colRef();
    if (eof() || at().t !== 'op') fail();
    const op = toks[p].v;
    const opPos = toks[p].pos;
    p++;
    const right = operand();
    return { op, left, right, pos: opPos };
  };

  const alias = (): string | undefined => {
    if (isWord('as')) {
      p++;
      if (eof() || at().t !== 'word') fail();
      return toks[p++].v;
    }
    if (!eof() && at().t === 'word' && !KEYWORDS.has(at().v.toLowerCase())) return toks[p++].v;
    return undefined;
  };

  const fromItem = (): FromItem => {
    if (eof() || at().t !== 'word') fail();
    const t = toks[p++];
    return { rel: t.v, alias: alias(), pos: t.pos };
  };

  takeWord('select');
  const targets: { star: boolean; ref?: ColRef }[] = [];
  if (!eof() && at().t === 'punct' && at().v === '*') {
    p++;
    targets.push({ star: true });
  } else {
    for (;;) {
      targets.push({ star: false, ref: colRef() });
      if (!eof() && at().t === 'punct' && at().v === ',') {
        p++;
        continue;
      }
      break;
    }
  }

  takeWord('from');
  const from: FromItem[] = [fromItem()];
  let explicitJoin = false;
  let joinOn: AExpr | undefined;
  if (isWord('inner')) p++;
  if (isWord('join')) {
    p++;
    explicitJoin = true;
    from.push(fromItem());
    takeWord('on');
    joinOn = aExpr();
  } else if (!eof() && at().t === 'punct' && at().v === ',') {
    p++;
    from.push(fromItem());
  }

  const where: AExpr[] = [];
  if (isWord('where')) {
    p++;
    for (;;) {
      where.push(aExpr());
      if (isWord('and')) {
        p++;
        continue;
      }
      break;
    }
  }

  let limit: number | undefined;
  if (isWord('limit')) {
    p++;
    if (eof() || at().t !== 'num') fail();
    limit = Math.max(1, Math.floor(Number(toks[p++].v)));
  }

  if (!eof() && at().t === 'punct' && at().v === ';') p++;
  if (!eof()) {
    const w = at().t === 'word' ? at().v.toLowerCase() : '';
    if (['or', 'order', 'group', 'having', 'union', 'left', 'right', 'distinct'].includes(w)) {
      throw new SqlError(
        `this sandbox parses only SELECT … FROM … [JOIN … ON …] [WHERE … AND …] [LIMIT n] — "${at().v}" is out of scope`,
        pos(),
      );
    }
    fail();
  }
  return { targets, from, explicitJoin, joinOn, where, limit };
}

/* =================================================================== analysis */

type Rte = { idx: number; alias: string; rel: Rel; rewrittenFrom?: string };
type Var = { rte: Rte; col: Col };
type RQual = { rte: Rte; col: Col; op: string; val: number | string; fromView?: boolean };
type JoinQual = { a: Var; b: Var };

type Analyzed = {
  rtes: Rte[];
  targets: { v: Var; label: string }[];
  restricts: RQual[];
  join?: JoinQual;
  limit?: number;
  rewritten: boolean;
  width: number;
};

function resolveVar(ref: ColRef, rtes: Rte[]): Var {
  if (ref.qual) {
    const rte = rtes.find((r) => r.alias === ref.qual);
    if (!rte) throw new SqlError(`missing FROM-clause entry for table "${ref.qual}"`, ref.pos);
    const col = rte.rel.cols.find((c) => c.name === ref.name);
    if (!col) throw new SqlError(`column ${ref.qual}.${ref.name} does not exist`, ref.pos);
    return { rte, col };
  }
  const hits = rtes
    .map((rte) => ({ rte, col: rte.rel.cols.find((c) => c.name === ref.name) }))
    .filter((h) => h.col);
  if (hits.length === 0) throw new SqlError(`column "${ref.name}" does not exist`, ref.pos);
  if (hits.length > 1) throw new SqlError(`column reference "${ref.name}" is ambiguous`, ref.pos);
  return { rte: hits[0].rte, col: hits[0].col as Col };
}

function constFor(col: Col, k: Konst): number | string {
  if (col.type === 'text') {
    if (k.kind !== 'str') throw new SqlError(`operator does not exist: text = integer`, k.pos);
    return k.str as string;
  }
  if (k.kind !== 'num') throw new SqlError(`invalid input syntax for type ${col.type}: "${k.str}"`, k.pos);
  return k.num as number;
}

function analyze(a: Ast): Analyzed {
  const rtes: Rte[] = [];
  let rewritten = false;
  a.from.forEach((f, i) => {
    const rel = CATALOG[f.rel.toLowerCase()];
    if (!rel) throw new SqlError(`relation "${f.rel}" does not exist`, f.pos);
    rtes.push({ idx: i + 1, alias: f.alias ?? rel.name, rel });
  });
  if (new Set(rtes.map((r) => r.alias)).size !== rtes.length)
    throw new SqlError(`table name "${rtes[0].alias}" specified more than once`, a.from[1].pos);

  const restricts: RQual[] = [];
  let join: JoinQual | undefined;

  const addExpr = (e: AExpr) => {
    const lv = resolveVar(e.left, rtes);
    if (isConst(e.right)) {
      restricts.push({ rte: lv.rte, col: lv.col, op: e.op, val: constFor(lv.col, e.right) });
      return;
    }
    const rv = resolveVar(e.right, rtes);
    if (rv.rte === lv.rte)
      throw new SqlError(`this sandbox does not plan same-relation column comparisons`, e.pos);
    if (e.op !== '=') throw new SqlError(`this sandbox plans equijoins only`, e.pos);
    if (join) throw new SqlError(`this sandbox supports one join clause`, e.pos);
    join = { a: lv, b: rv };
  };

  if (a.joinOn) addExpr(a.joinOn);
  a.where.forEach(addExpr);

  // --- rewrite: a view RTE is replaced by its base relation and its qual is pulled up
  for (const rte of rtes) {
    if (rte.rel.kind === 'v' && rte.rel.baseRel && rte.rel.viewQual) {
      const base = CATALOG[rte.rel.baseRel];
      const q = rte.rel.viewQual;
      const col = base.cols.find((c) => c.name === q.col) as Col;
      rte.rewrittenFrom = rte.rel.name;
      rte.rel = base;
      restricts.push({ rte, col, op: q.op, val: q.val, fromView: true });
      rewritten = true;
    }
  }

  const targets: { v: Var; label: string }[] = [];
  for (const t of a.targets) {
    if (t.star) {
      for (const rte of rtes) for (const c of rte.rel.cols) targets.push({ v: { rte, col: c }, label: `${rte.alias}.${c.name}` });
    } else {
      const v = resolveVar(t.ref as ColRef, rtes);
      targets.push({ v, label: `${v.rte.alias}.${v.col.name}` });
    }
  }

  if (rtes.length === 2 && !join)
    throw new SqlError(`no join clause between ${rtes[0].alias} and ${rtes[1].alias}: that is a cartesian product`, 0);

  const width = targets.reduce((s, t) => s + t.v.col.width, 0);
  return { rtes, targets, restricts, join, limit: a.limit, rewritten, width };
}

/* ================================================================ selectivity */

function selectivity(q: RQual): number {
  const c = q.col;
  if (q.op === '=') {
    if (c.mcv) {
      const hit = c.mcv.find((m) => m.v === q.val);
      if (hit) return hit.f;
      const covered = c.mcv.reduce((s, m) => s + m.f, 0);
      return Math.max(1e-5, (1 - covered) / Math.max(1, c.ndistinct - c.mcv.length));
    }
    return 1 / c.ndistinct;
  }
  if (q.op === '<>') return 1 - selectivity({ ...q, op: '=' });
  if (c.range) {
    const [lo, hi] = c.range;
    const v = Math.min(hi, Math.max(lo, q.val as number));
    const above = (hi - v) / (hi - lo);
    const below = (v - lo) / (hi - lo);
    const s = q.op === '>' || q.op === '>=' ? above : below;
    return Math.min(1, Math.max(1e-5, s));
  }
  return 0.3333;
}

/* ======================================================================= cost */

const SEQ_PAGE_COST = 1.0;
const CPU_TUPLE_COST = 0.01;
const CPU_INDEX_TUPLE_COST = 0.005;
const CPU_OPERATOR_COST = 0.0025;
const WORK_MEM = 4 * 1024 * 1024;

type Cfg = { rpc: number; idxCity: boolean; idxCust: boolean; idxAmt: boolean; pool: number };

function indexEnabled(i: Idx, cfg: Cfg) {
  if (!i.toggle) return true;
  return i.toggle === 'city' ? cfg.idxCity : i.toggle === 'cust' ? cfg.idxCust : cfg.idxAmt;
}

/** Mackert & Lohman: heap pages touched when fetching `t` tuples at random from `T` pages. */
function pagesFetched(t: number, T: number) {
  if (t <= 0) return 0;
  return Math.min(T, (2 * T * t) / (2 * T + t));
}

type PathKind = 'Seq Scan' | 'Index Scan' | 'Bitmap Heap Scan';

type Path = {
  kind: PathKind;
  rte: Rte;
  index?: Idx;
  cond?: RQual;
  filters: RQual[];
  sel: number;
  rows: number;
  startup: number;
  total: number;
  pages: number;
  why: string;
};

function seqPath(rte: Rte, quals: RQual[], sel: number): Path {
  const rel = rte.rel;
  const rows = Math.max(1, Math.round(rel.reltuples * sel));
  const total =
    SEQ_PAGE_COST * rel.relpages + (CPU_TUPLE_COST + CPU_OPERATOR_COST * quals.length) * rel.reltuples;
  return {
    kind: 'Seq Scan',
    rte,
    filters: quals,
    sel,
    rows,
    startup: 0,
    total,
    pages: rel.relpages,
    why: `${rel.relpages} pages read sequentially, ${fmtNum(rel.reltuples)} tuples through the filter`,
  };
}

function idxCosts(rte: Rte, idx: Idx, cond: RQual, filters: RQual[], cfg: Cfg) {
  const rel = rte.rel;
  const condSel = selectivity(cond);
  const filtSel = filters.reduce((s, q) => s * selectivity(q), 1);
  const tuples = Math.max(1, rel.reltuples * condSel);
  const leaf = Math.max(1, Math.ceil(condSel * idx.pages)); // leaf pages scanned
  const descent = Math.ceil(Math.log2(Math.max(2, rel.reltuples))) * CPU_OPERATOR_COST;
  const indexTotal = cfg.rpc * leaf + descent + (CPU_INDEX_TUPLE_COST + CPU_OPERATOR_COST) * tuples;
  const rows = Math.max(1, Math.round(tuples * filtSel));

  // --- plain index scan: heap pages in index order, interpolated by correlation
  const pagesU = pagesFetched(tuples, rel.relpages);
  const maxIO = cfg.rpc * pagesU;
  const minIO = cfg.rpc + Math.max(0, Math.ceil(condSel * rel.relpages) - 1) * SEQ_PAGE_COST;
  const csq = idx.corr * idx.corr;
  const io = maxIO + csq * (minIO - maxIO);
  const idxScan: Path = {
    kind: 'Index Scan',
    rte,
    index: idx,
    cond,
    filters,
    sel: condSel * filtSel,
    rows,
    startup: cfg.rpc,
    total: indexTotal + io + (CPU_TUPLE_COST + CPU_OPERATOR_COST * filters.length) * tuples,
    pages: pagesU,
    why: `${fmtNum(Math.round(tuples))} index entries, ~${Math.round(pagesU)} heap pages at correlation ${idx.corr}`,
  };

  // --- bitmap heap scan: the same index, then the heap in physical order
  const pagesB = pagesFetched(tuples, rel.relpages);
  const perPage =
    pagesB >= 2 ? cfg.rpc - (cfg.rpc - SEQ_PAGE_COST) * Math.sqrt(pagesB / rel.relpages) : cfg.rpc;
  const bstart = indexTotal + 0.1 * CPU_OPERATOR_COST * tuples;
  const bitmap: Path = {
    kind: 'Bitmap Heap Scan',
    rte,
    index: idx,
    cond,
    filters,
    sel: condSel * filtSel,
    rows,
    startup: bstart,
    total:
      bstart +
      pagesB * perPage +
      (CPU_TUPLE_COST + CPU_OPERATOR_COST * (1 + filters.length)) * tuples,
    pages: pagesB,
    why: `bitmap of ${fmtNum(Math.round(tuples))} tuples → ${Math.round(pagesB)} heap pages in ctid order at ${perPage.toFixed(2)}/page`,
  };
  return { idxScan, bitmap };
}

function indexableOps(col: Col) {
  return col.type === 'text' ? ['='] : ['=', '<', '<=', '>', '>='];
}

function scanPaths(rte: Rte, quals: RQual[], cfg: Cfg): Path[] {
  const sel = quals.reduce((s, q) => s * selectivity(q), 1);
  const out: Path[] = [seqPath(rte, quals, sel)];
  for (const idx of INDEXES) {
    if (idx.rel !== rte.rel.name || !indexEnabled(idx, cfg)) continue;
    const cond = quals.find((q) => q.col.name === idx.col && indexableOps(q.col).includes(q.op));
    if (!cond) continue;
    const filters = quals.filter((q) => q !== cond);
    const { idxScan, bitmap } = idxCosts(rte, idx, cond, filters, cfg);
    out.push(idxScan, bitmap);
  }
  return out;
}

/** add_path()'s STD_FUZZ_FACTOR: a new path must be more than 1% cheaper to displace the incumbent. */
const FUZZ = 1.01;

function cheapest(paths: Path[], frac: number) {
  const eff = (p: Path) => p.startup + (p.total - p.startup) * frac;
  return paths.reduce((a, b) => (eff(b) * FUZZ < eff(a) ? b : a));
}

/* =================================================================== the plan */

type ExecSpec =
  | { kind: 'seq'; rel: RelName; filters: RQual[] }
  | { kind: 'index'; rel: RelName; idx: Idx; cond: RQual; filters: RQual[]; bind?: { rel: RelName; col: string } }
  | { kind: 'bitmap'; rel: RelName; idx: Idx; cond: RQual; filters: RQual[]; idxNode: number }
  | { kind: 'bitmapidx' }
  | { kind: 'hash' }
  | { kind: 'hashjoin'; buildRel: RelName; buildCol: string; probeRel: RelName; probeCol: string }
  | { kind: 'nl' }
  | { kind: 'limit'; n: number };

type PNode = {
  id: number;
  op: string;
  title: string;
  detail: string[];
  startup: number;
  total: number;
  rows: number;
  width: number;
  children: PNode[];
  exec: ExecSpec;
  hint: string;
};

type Alt = { what: string; kind: string; startup: number; total: number; rows: number; chosen: boolean; why: string };

type Plan = { root: PNode; nodes: PNode[]; alts: Alt[]; join: 'Hash Join' | 'Nested Loop' | '—' };

function qualText(q: RQual) {
  const v = typeof q.val === 'string' ? `'${q.val}'` : String(q.val);
  return `${q.rte.alias}.${q.col.name} ${q.op} ${v}`;
}

function scanNode(p: Path, width: number, nextId: () => number): PNode {
  const rel = p.rte.rel.name as RelName;
  const alias = p.rte.alias === rel ? '' : ` ${p.rte.alias}`;
  if (p.kind === 'Seq Scan') {
    return {
      id: nextId(),
      op: 'Seq Scan',
      title: `Seq Scan on ${rel}${alias}`,
      detail: p.filters.length ? [`Filter: ${p.filters.map(qualText).join(' AND ')}`] : [],
      startup: p.startup,
      total: p.total,
      rows: p.rows,
      width,
      children: [],
      exec: { kind: 'seq', rel, filters: p.filters },
      hint: p.why,
    };
  }
  if (p.kind === 'Index Scan') {
    return {
      id: nextId(),
      op: 'Index Scan',
      title: `Index Scan using ${(p.index as Idx).name}`,
      detail: [
        `Index Cond: ${qualText(p.cond as RQual)}`,
        ...(p.filters.length ? [`Filter: ${p.filters.map(qualText).join(' AND ')}`] : []),
      ],
      startup: p.startup,
      total: p.total,
      rows: p.rows,
      width,
      children: [],
      exec: { kind: 'index', rel, idx: p.index as Idx, cond: p.cond as RQual, filters: p.filters },
      hint: p.why,
    };
  }
  const idxId = nextId();
  const heapId = nextId();
  const child: PNode = {
    id: idxId,
    op: 'Bitmap Index Scan',
    title: `Bitmap Index Scan on ${(p.index as Idx).name}`,
    detail: [`Index Cond: ${qualText(p.cond as RQual)}`],
    startup: 0,
    total: p.startup,
    rows: Math.round(p.rte.rel.reltuples * selectivity(p.cond as RQual)),
    width: 0,
    children: [],
    exec: { kind: 'bitmapidx' },
    hint: 'Walks the index leaves and sets a bit per matching ctid. No heap access at all.',
  };
  return {
    id: heapId,
    op: 'Bitmap Heap Scan',
    title: `Bitmap Heap Scan on ${rel}${alias}`,
    detail: [
      `Recheck Cond: ${qualText(p.cond as RQual)}`,
      ...(p.filters.length ? [`Filter: ${p.filters.map(qualText).join(' AND ')}`] : []),
    ],
    startup: p.startup,
    total: p.total,
    rows: p.rows,
    width,
    children: [child],
    exec: { kind: 'bitmap', rel, idx: p.index as Idx, cond: p.cond as RQual, filters: p.filters, idxNode: idxId },
    hint: p.why,
  };
}

function planQuery(an: Analyzed, cfg: Cfg): Plan {
  let id = 0;
  const nextId = () => id++;
  const alts: Alt[] = [];

  const qualsFor = (rte: Rte) => an.restricts.filter((q) => q.rte === rte);

  if (an.rtes.length === 1) {
    const rte = an.rtes[0];
    const paths = scanPaths(rte, qualsFor(rte), cfg);
    const frac = an.limit ? Math.min(1, an.limit / Math.max(1, paths[0].rows)) : 1;
    const best = cheapest(paths, frac);
    paths.forEach((p) =>
      alts.push({
        what: rte.alias,
        kind: p.kind,
        startup: p.startup,
        total: p.total,
        rows: p.rows,
        chosen: p === best,
        why: p.why,
      }),
    );
    let root = scanNode(best, an.width, nextId);
    if (an.limit) root = limitNode(root, an.limit, nextId);
    return { root, nodes: flatten(root), alts, join: '—' };
  }

  // two relations, one equijoin clause
  const jq = an.join as JoinQual;
  const sideA = jq.a.rte;
  const sideB = jq.b.rte;
  const colA = jq.a.col;
  const colB = jq.b.col;
  const jsel = 1 / Math.max(colA.ndistinct, colB.ndistinct);

  const pathsA = scanPaths(sideA, qualsFor(sideA), cfg);
  const pathsB = scanPaths(sideB, qualsFor(sideB), cfg);
  const bestA = cheapest(pathsA, 1);
  const bestB = cheapest(pathsB, 1);
  [
    [sideA, pathsA, bestA] as const,
    [sideB, pathsB, bestB] as const,
  ].forEach(([rte, paths, best]) =>
    paths.forEach((p) =>
      alts.push({
        what: rte.alias,
        kind: p.kind,
        startup: p.startup,
        total: p.total,
        rows: p.rows,
        chosen: p === best,
        why: p.why,
      }),
    ),
  );

  const joinRows = Math.max(1, Math.round(bestA.rows * bestB.rows * jsel));

  type Cand = {
    method: 'Hash Join' | 'Nested Loop';
    outer: Path;
    inner: Path;
    innerIdx?: Idx;
    innerCond?: RQual;
    innerFilters?: RQual[];
    innerPer?: number;
    startup: number;
    total: number;
    why: string;
  };
  const cands: Cand[] = [];

  const hashCand = (build: Path, probe: Path, buildVar: Var, probeVar: Var): Cand => {
    const bytes = build.rows * (an.width + 24);
    const batches = Math.max(1, Math.ceil(bytes / WORK_MEM));
    const spill =
      batches > 1
        ? 2 * SEQ_PAGE_COST * ((build.rows * (an.width + 24) + probe.rows * (an.width + 24)) / 8192)
        : 0;
    const startup = build.total + (CPU_OPERATOR_COST + CPU_TUPLE_COST) * build.rows + probe.startup;
    const total =
      startup +
      (probe.total - probe.startup) +
      CPU_OPERATOR_COST * probe.rows +
      CPU_TUPLE_COST * joinRows +
      spill;
    return {
      method: 'Hash Join',
      outer: probe,
      inner: build,
      startup,
      total,
      why:
        `build a hash table on ${buildVar.rte.alias}.${buildVar.col.name} from ${fmtNum(build.rows)} rows` +
        (batches > 1 ? `, ${batches} batches — the hash exceeds work_mem, so both sides spill to temp files` : '') +
        `, then probe once per ${probeVar.rte.alias} row`,
    };
  };

  const nlCand = (outer: Path, outerVar: Var, innerRte: Rte, innerVar: Var): Cand | null => {
    const idx = INDEXES.find(
      (i) => i.rel === innerRte.rel.name && i.col === innerVar.col.name && indexEnabled(i, cfg),
    );
    if (!idx) return null;
    const bound: RQual = { rte: innerRte, col: innerVar.col, op: '=', val: 0 };
    const filters = qualsFor(innerRte);
    const { idxScan } = idxCosts(innerRte, idx, bound, filters, cfg);
    const per = idxScan.total;
    const total = outer.total + outer.rows * per + CPU_TUPLE_COST * joinRows;
    return {
      method: 'Nested Loop',
      outer,
      inner: idxScan,
      innerIdx: idx,
      innerCond: bound,
      innerFilters: filters,
      innerPer: per,
      startup: outer.startup + idxScan.startup,
      total,
      why: `${fmtNum(outer.rows)} outer rows × ${per.toFixed(2)} per index rescan on ${idx.name}`,
    };
  };

  cands.push(hashCand(bestA, bestB, jq.a, jq.b));
  cands.push(hashCand(bestB, bestA, jq.b, jq.a));
  const nl1 = nlCand(bestA, jq.a, sideB, jq.b);
  const nl2 = nlCand(bestB, jq.b, sideA, jq.a);
  if (nl1) cands.push(nl1);
  if (nl2) cands.push(nl2);

  const fracJ = an.limit ? Math.min(1, an.limit / joinRows) : 1;
  const effC = (c: Cand) => c.startup + (c.total - c.startup) * fracJ;
  const bestJ = cands.reduce((a, b) => (effC(b) * FUZZ < effC(a) ? b : a));
  cands.forEach((c) =>
    alts.push({
      what: 'join',
      kind: `${c.method} (${c.inner.rte.alias} inner)`,
      startup: c.startup,
      total: c.total,
      rows: joinRows,
      chosen: c === bestJ,
      why: c.why,
    }),
  );
  if (!nl1 && !nl2)
    alts.push({
      what: 'join',
      kind: 'Nested Loop',
      startup: 0,
      total: NaN,
      rows: joinRows,
      chosen: false,
      why: 'no inner index path on the join key — a rescan would re-read the whole relation per outer row',
    });

  let root: PNode;
  if (bestJ.method === 'Hash Join') {
    const outerNode = scanNode(bestJ.outer, an.width, nextId);
    const innerScan = scanNode(bestJ.inner, an.width, nextId);
    const hashId = nextId();
    const hashNode: PNode = {
      id: hashId,
      op: 'Hash',
      title: 'Hash',
      detail: [`Buckets: ${1 << Math.ceil(Math.log2(Math.max(2, bestJ.inner.rows)))}`],
      startup: bestJ.inner.total,
      total: bestJ.inner.total,
      rows: bestJ.inner.rows,
      width: an.width,
      children: [innerScan],
      exec: { kind: 'hash' },
      hint: 'Drains its child completely on the first next() from above, then holds the build side in memory.',
    };
    const buildRte = bestJ.inner.rte;
    const probeRte = bestJ.outer.rte;
    const buildCol = buildRte === jq.a.rte ? jq.a.col.name : jq.b.col.name;
    const probeCol = probeRte === jq.a.rte ? jq.a.col.name : jq.b.col.name;
    root = {
      id: nextId(),
      op: 'Hash Join',
      title: 'Hash Join',
      detail: [`Hash Cond: ${probeRte.alias}.${probeCol} = ${buildRte.alias}.${buildCol}`],
      startup: bestJ.startup,
      total: bestJ.total,
      rows: joinRows,
      width: an.width,
      children: [outerNode, hashNode],
      exec: {
        kind: 'hashjoin',
        buildRel: buildRte.rel.name as RelName,
        buildCol,
        probeRel: probeRte.rel.name as RelName,
        probeCol,
      },
      hint: bestJ.why,
    };
  } else {
    const outerNode = scanNode(bestJ.outer, an.width, nextId);
    const innerRte = bestJ.inner.rte;
    const outerRte = bestJ.outer.rte;
    const innerCol = innerRte === jq.a.rte ? jq.a.col.name : jq.b.col.name;
    const outerCol = outerRte === jq.a.rte ? jq.a.col.name : jq.b.col.name;
    const innerNode: PNode = {
      id: nextId(),
      op: 'Index Scan',
      title: `Index Scan using ${(bestJ.innerIdx as Idx).name}`,
      detail: [
        `Index Cond: ${innerRte.alias}.${innerCol} = ${outerRte.alias}.${outerCol}`,
        ...((bestJ.innerFilters as RQual[]).length
          ? [`Filter: ${(bestJ.innerFilters as RQual[]).map(qualText).join(' AND ')}`]
          : []),
      ],
      startup: bestJ.inner.startup,
      total: bestJ.innerPer as number,
      rows: Math.max(1, bestJ.inner.rows),
      width: an.width,
      children: [],
      exec: {
        kind: 'index',
        rel: innerRte.rel.name as RelName,
        idx: bestJ.innerIdx as Idx,
        cond: bestJ.innerCond as RQual,
        filters: bestJ.innerFilters as RQual[],
        bind: { rel: outerRte.rel.name as RelName, col: outerCol },
      },
      hint: 'Re-descends the index once per outer row. Cost is charged per rescan; the buffer pool usually turns those into hits.',
    };
    root = {
      id: nextId(),
      op: 'Nested Loop',
      title: 'Nested Loop',
      startup: bestJ.startup,
      detail: [],
      total: bestJ.total,
      rows: joinRows,
      width: an.width,
      children: [outerNode, innerNode],
      exec: { kind: 'nl' },
      hint: bestJ.why,
    };
  }

  if (an.limit) root = limitNode(root, an.limit, nextId);
  return { root, nodes: flatten(root), alts, join: bestJ.method };
}

function limitNode(child: PNode, n: number, nextId: () => number): PNode {
  const rows = Math.min(n, child.rows);
  const frac = rows / Math.max(1, child.rows);
  return {
    id: nextId(),
    op: 'Limit',
    title: 'Limit',
    detail: [],
    startup: child.startup,
    total: child.startup + (child.total - child.startup) * frac,
    rows,
    width: child.width,
    children: [child],
    exec: { kind: 'limit', n },
    hint: 'A LIMIT does not just truncate output — the planner multiplies run cost by the fraction of rows it expects to need, which favours low-startup plans.',
  };
}

function flatten(n: PNode): PNode[] {
  return [n, ...n.children.flatMap(flatten)];
}

/* =================================================================== executor */

const C_CALLS = 0;
const C_ROWS = 1;
const C_HIT = 2;
const C_READ = 3;
const C_LOOPS = 4;
const C_SLOTS = 5;

class Pool {
  cap: number;
  m = new Map<string, number>();
  hits = 0;
  misses = 0;
  evictions = 0;
  constructor(cap: number) {
    this.cap = cap;
  }
  read(key: string, C: Float64Array, id: number) {
    if (this.m.has(key)) {
      this.m.delete(key);
      this.m.set(key, 1);
      this.hits++;
      C[id * C_SLOTS + C_HIT]++;
      return;
    }
    if (this.m.size >= this.cap) {
      const victim = this.m.keys().next().value as string;
      this.m.delete(victim);
      this.evictions++;
    }
    this.m.set(key, 1);
    this.misses++;
    C[id * C_SLOTS + C_READ]++;
  }
}

type Tup = { c: number; o: number };
type Iter = { next: () => Tup | null };
type Ctx = { pool: Pool; C: Float64Array; D: Data };

function cmp(a: number | string, op: string, b: number | string) {
  switch (op) {
    case '=':
      return a === b;
    case '<>':
      return a !== b;
    case '<':
      return a < b;
    case '<=':
      return a <= b;
    case '>':
      return a > b;
    default:
      return a >= b;
  }
}

function passes(filters: RQual[], rel: RelName, row: number, D: Data) {
  for (const f of filters) if (!cmp(valOf(rel, f.col.name, row, D), f.op, f.val)) return false;
  return true;
}

function heapKey(rel: RelName, blk: number) {
  return `${rel === 'customers' ? 'C' : 'O'}h${blk}`;
}
function idxKey(idx: Idx, blk: number) {
  return `${idx.rel === 'customers' ? 'C' : 'O'}i${idx.name}:${blk}`;
}

/** A contiguous run of btree entries: entry ordinal `base + j` points at heap row `rowAt(j)`. */
type Run = { base: number; count: number; rowAt: (j: number) => number };

function lookup(idx: Idx, op: string, val: number | string, D: Data): Run {
  if (idx.name === 'customers_pkey' || idx.name === 'orders_pkey') {
    const n = idx.rel === 'customers' ? CUSTOMERS.reltuples : ORDERS.reltuples;
    const v = val as number;
    if (op !== '=' || v < 1 || v > n) return { base: 0, count: 0, rowAt: () => 0 };
    return { base: v - 1, count: 1, rowAt: () => v - 1 };
  }
  if (idx.name === 'customers_city_idx') {
    const k = CITIES.indexOf(val as string);
    if (k < 0) return { base: 0, count: 0, rowAt: () => 0 };
    const from = D.cityOff[k];
    const to = D.cityOff[k + 1];
    return { base: from, count: to - from, rowAt: (j) => D.custByCity[from + j] };
  }
  if (idx.name === 'orders_customer_id_idx') {
    const v = val as number;
    if (v < 1 || v > CUSTOMERS.reltuples) return { base: 0, count: 0, rowAt: () => 0 };
    const from = D.custOff[v - 1];
    const to = D.custOff[v];
    return { base: from, count: to - from, rowAt: (j) => D.ordByCust[from + j] };
  }
  // orders_amount_idx — a range over the (amount, ctid)-ordered leaves
  const v = Math.max(0, Math.min(2000, Math.floor(val as number)));
  let from = 0;
  let to = ORDERS.reltuples;
  if (op === '=') {
    from = D.amtStart[v];
    to = D.amtStart[v + 1];
  } else if (op === '>') {
    from = D.amtStart[Math.min(2000, v + 1)];
  } else if (op === '>=') {
    from = D.amtStart[v];
  } else if (op === '<') {
    to = D.amtStart[v];
  } else if (op === '<=') {
    to = D.amtStart[Math.min(2000, v + 1)];
  }
  return { base: from, count: Math.max(0, to - from), rowAt: (j) => D.ordByAmt[from + j] };
}

function mkTup(rel: RelName, row: number): Tup {
  return rel === 'customers' ? { c: row, o: -1 } : { c: -1, o: row };
}

function mkIter(node: PNode, ctx: Ctx, bound?: number | string): Iter {
  const { pool, C, D } = ctx;
  const id = node.id;
  const bump = (slot: number, n = 1) => {
    C[id * C_SLOTS + slot] += n;
  };

  switch (node.exec.kind) {
    case 'seq': {
      const spec = node.exec;
      const rel = CATALOG[spec.rel];
      let i = 0;
      let blk = -1;
      return {
        next() {
          bump(C_CALLS);
          while (i < rel.reltuples) {
            const r = i++;
            const b = Math.floor(r / rel.perPage);
            if (b !== blk) {
              pool.read(heapKey(spec.rel, b), C, id);
              blk = b;
            }
            if (passes(spec.filters, spec.rel, r, D)) {
              bump(C_ROWS);
              return mkTup(spec.rel, r);
            }
          }
          return null;
        },
      };
    }
    case 'index': {
      const spec = node.exec;
      const rel = CATALOG[spec.rel];
      const key = spec.bind ? (bound as number | string) : spec.cond.val;
      const run = lookup(spec.idx, spec.cond.op, key, D);
      let j = 0;
      let opened = false;
      return {
        next() {
          bump(C_CALLS);
          if (!opened) {
            opened = true;
            pool.read(idxKey(spec.idx, -1), C, id); // the btree root / internal descent
          }
          while (j < run.count) {
            const e = run.base + j;
            const r = run.rowAt(j);
            j++;
            pool.read(idxKey(spec.idx, Math.floor(e / spec.idx.perLeaf)), C, id);
            pool.read(heapKey(spec.rel, Math.floor(r / rel.perPage)), C, id);
            if (passes(spec.filters, spec.rel, r, D)) {
              bump(C_ROWS);
              return mkTup(spec.rel, r);
            }
          }
          return null;
        },
      };
    }
    case 'bitmap': {
      const spec = node.exec;
      const rel = CATALOG[spec.rel];
      const idxId = spec.idxNode;
      let built = false;
      let rows: number[] = [];
      let k = 0;
      let curPage = -1;
      return {
        next() {
          bump(C_CALLS);
          if (!built) {
            built = true;
            const run = lookup(spec.idx, spec.cond.op, spec.cond.val, D);
            pool.read(idxKey(spec.idx, -1), C, idxId);
            let lastLeaf = -2;
            const acc: number[] = [];
            for (let j = 0; j < run.count; j++) {
              const leaf = Math.floor((run.base + j) / spec.idx.perLeaf);
              if (leaf !== lastLeaf) {
                pool.read(idxKey(spec.idx, leaf), C, idxId);
                lastLeaf = leaf;
              }
              C[idxId * C_SLOTS + C_ROWS]++;
              acc.push(run.rowAt(j));
            }
            C[idxId * C_SLOTS + C_CALLS]++;
            acc.sort((a, b) => a - b); // the bitmap is in ctid order by construction
            rows = acc;
          }
          while (k < rows.length) {
            const r = rows[k++];
            const blk = Math.floor(r / rel.perPage);
            if (blk !== curPage) {
              pool.read(heapKey(spec.rel, blk), C, id);
              curPage = blk;
            }
            if (passes(spec.filters, spec.rel, r, D)) {
              bump(C_ROWS);
              return mkTup(spec.rel, r);
            }
          }
          return null;
        },
      };
    }
    case 'hash': {
      const child = mkIter(node.children[0], ctx);
      return {
        next() {
          bump(C_CALLS);
          const t = child.next();
          if (t) bump(C_ROWS);
          return t;
        },
      };
    }
    case 'hashjoin': {
      const spec = node.exec;
      const probe = mkIter(node.children[0], ctx);
      const hash = mkIter(node.children[1], ctx);
      const table = new Map<number | string, Tup[]>();
      let built = false;
      let bucket: Tup[] | undefined;
      let bi = 0;
      let cur: Tup | null = null;
      return {
        next() {
          bump(C_CALLS);
          if (!built) {
            built = true;
            for (;;) {
              const t = hash.next();
              if (!t) break;
              const row = spec.buildRel === 'customers' ? t.c : t.o;
              const k = valOf(spec.buildRel, spec.buildCol, row, D);
              const list = table.get(k);
              if (list) list.push(t);
              else table.set(k, [t]);
            }
          }
          for (;;) {
            if (bucket && bi < bucket.length && cur) {
              const m = bucket[bi++];
              bump(C_ROWS);
              return { c: cur.c >= 0 ? cur.c : m.c, o: cur.o >= 0 ? cur.o : m.o };
            }
            const p = probe.next();
            if (!p) return null;
            cur = p;
            const row = spec.probeRel === 'customers' ? p.c : p.o;
            bucket = table.get(valOf(spec.probeRel, spec.probeCol, row, D));
            bi = 0;
          }
        },
      };
    }
    case 'nl': {
      const outer = mkIter(node.children[0], ctx);
      const innerNode = node.children[1];
      const spec = innerNode.exec as Extract<ExecSpec, { kind: 'index' }>;
      let inner: Iter | null = null;
      let cur: Tup | null = null;
      return {
        next() {
          bump(C_CALLS);
          for (;;) {
            if (!inner) {
              const o = outer.next();
              if (!o) return null;
              cur = o;
              const bindRow = spec.bind && spec.bind.rel === 'customers' ? o.c : o.o;
              const key = valOf(spec.bind!.rel, spec.bind!.col, bindRow, D);
              inner = mkIter(innerNode, ctx, key);
              C[innerNode.id * C_SLOTS + C_LOOPS]++;
            }
            const i = inner.next();
            if (!i) {
              inner = null;
              continue;
            }
            bump(C_ROWS);
            return { c: cur!.c >= 0 ? cur!.c : i.c, o: cur!.o >= 0 ? cur!.o : i.o };
          }
        },
      };
    }
    default: {
      const spec = node.exec as Extract<ExecSpec, { kind: 'limit' }>;
      const child = mkIter(node.children[0], ctx);
      let n = 0;
      return {
        next() {
          bump(C_CALLS);
          if (n >= spec.n) return null;
          const t = child.next();
          if (!t) return null;
          n++;
          bump(C_ROWS);
          return t;
        },
      };
    }
  }
}

type Frame = { key: string; rel: 'C' | 'O'; index: boolean; label: string };

type RunResult = {
  C: Float64Array;
  hits: number;
  misses: number;
  evictions: number;
  steps: number;
  done: boolean;
  rows: (string | number)[][];
  produced: number;
  frames: Frame[];
  delta: Float64Array;
  touched: number[];
};

function execute(plan: Plan, an: Analyzed, poolCap: number, maxSteps: number): RunResult {
  const D = data();
  const pool = new Pool(poolCap);
  const n = plan.nodes.length;
  const C = new Float64Array(n * C_SLOTS);
  const ctx: Ctx = { pool, C, D };
  const it = mkIter(plan.root, ctx);
  const rows: (string | number)[][] = [];
  let steps = 0;
  let produced = 0;
  let done = false;
  const before = C.slice();
  const HARD = 400_000;
  while (steps < maxSteps && steps < HARD) {
    before.set(C);
    const t = it.next();
    steps++;
    if (!t) {
      done = true;
      break;
    }
    produced++;
    if (rows.length < 10)
      rows.push(
        an.targets.map((tg) =>
          valOf(tg.v.rte.rel.name as RelName, tg.v.col.name, tg.v.rte.rel.name === 'customers' ? t.c : t.o, D),
        ),
      );
  }
  const delta = C.slice();
  for (let i = 0; i < delta.length; i++) delta[i] -= before[i];
  const touched: number[] = [];
  for (let id = 0; id < n; id++) {
    let any = false;
    for (let s = 0; s < C_SLOTS; s++) if (delta[id * C_SLOTS + s] > 0) any = true;
    if (any) touched.push(id);
  }
  const frames: Frame[] = Array.from(pool.m.keys()).map((key) => ({
    key,
    rel: key[0] === 'C' ? 'C' : 'O',
    index: key[1] === 'i',
    label: key[1] === 'i' ? 'i' : key[0],
  }));
  return {
    C,
    hits: pool.hits,
    misses: pool.misses,
    evictions: pool.evictions,
    steps,
    done,
    rows,
    produced,
    frames,
    delta,
    touched,
  };
}

/* ================================================================ parse trees */

type TNode = { label: string; text?: string; note?: string; added?: boolean; kids: TNode[] };

function astTree(a: Ast, an: Analyzed | null, stage: 'parse' | 'analyze' | 'rewrite'): TNode[] {
  const ann = stage !== 'parse' && an;
  const rteOf = (nameOrAlias: string) =>
    an?.rtes.find((r) => r.alias === nameOrAlias || r.rel.name === nameOrAlias);

  const varNote = (ref: ColRef): string | undefined => {
    if (!ann || !an) return undefined;
    try {
      const v = resolveVar(ref, an.rtes);
      return `Var  varno=${v.rte.idx} varattno=${v.col.attnum}  ${v.col.type}`;
    } catch {
      return undefined;
    }
  };

  const colNode = (ref: ColRef): TNode => ({
    label: 'ColumnRef',
    text: ref.qual ? `${ref.qual}.${ref.name}` : ref.name,
    note: varNote(ref),
    kids: [],
  });

  const constNode = (k: Konst): TNode => ({
    label: 'A_Const',
    text: k.text,
    note: ann ? `Const  ${k.kind === 'str' ? 'text' : 'numeric'}` : undefined,
    kids: [],
  });

  const exprNode = (e: AExpr): TNode => {
    let note: string | undefined;
    if (ann && an) {
      try {
        const lv = resolveVar(e.left, an.rtes);
        const ty = lv.col.type;
        note = `OpExpr  ${OPNAME[e.op]?.[ty] ?? e.op}`;
      } catch {
        note = undefined;
      }
    }
    return {
      label: 'A_Expr',
      text: `"${e.op}"`,
      note,
      kids: [colNode(e.left), isConst(e.right) ? constNode(e.right) : colNode(e.right)],
    };
  };

  const rangeVar = (f: FromItem): TNode => {
    const rte = rteOf(f.alias ?? f.rel);
    let note: string | undefined;
    if (ann && rte) {
      if (stage === 'rewrite' && rte.rewrittenFrom)
        note = `view ${rte.rewrittenFrom} expanded → RTE ${rte.idx}: ${rte.rel.name} (oid ${rte.rel.oid})`;
      else
        note = `RTE ${rte.idx}  relid=${rte.rel.oid}  ${rte.rel.kind === 'v' ? 'view' : `heap, ${rte.rel.relpages} pages, ${fmtNum(rte.rel.reltuples)} tuples`}`;
    }
    return {
      label: 'RangeVar',
      text: `"${f.rel}"${f.alias ? ` AS ${f.alias}` : ''}`,
      note,
      kids: [],
    };
  };

  const targets: TNode = {
    label: 'targetList',
    kids: a.targets.map((t) =>
      t.star ? { label: 'ResTarget', text: '*', note: ann ? 'expanded to every column of every RTE' : undefined, kids: [] } : { label: 'ResTarget', kids: [colNode(t.ref as ColRef)] },
    ),
  };

  const fromKids: TNode[] = a.from.map(rangeVar);
  if (a.joinOn) fromKids.push({ label: 'JoinExpr', text: 'JOIN_INNER', kids: [exprNode(a.joinOn)] });
  const from: TNode = { label: 'fromClause', kids: fromKids };

  const whereKids: TNode[] = a.where.map(exprNode);
  if (stage === 'rewrite' && an?.rewritten) {
    const vq = an.restricts.find((q) => q.fromView);
    if (vq)
      whereKids.push({
        label: 'OpExpr',
        text: `${vq.rte.alias}.${vq.col.name} ${vq.op} ${vq.val}`,
        note: 'added by QueryRewrite from the view definition',
        added: true,
        kids: [],
      });
  }
  const out: TNode[] = [targets, from];
  if (whereKids.length)
    out.push({ label: 'whereClause', kids: whereKids.length > 1 ? [{ label: 'BoolExpr', text: 'AND_EXPR', kids: whereKids }] : whereKids });
  if (a.limit !== undefined)
    out.push({ label: 'limitCount', kids: [{ label: 'A_Const', text: String(a.limit), kids: [] }] });
  return [{ label: stage === 'parse' ? 'SelectStmt' : 'Query', text: stage === 'parse' ? '(raw parse tree)' : '(analysed tree)', kids: out }];
}

function flattenTree(nodes: TNode[], depth = 0): { n: TNode; d: number }[] {
  return nodes.flatMap((n) => [{ n, d: depth }, ...flattenTree(n.kids, depth + 1)]);
}

/* ================================================================== examples */

const EXAMPLES = [
  {
    value: 'join',
    label: 'Join, filter on each side',
    sql: "SELECT c.city, o.id, o.amount\nFROM customers c JOIN orders o ON o.customer_id = c.id\nWHERE c.city = 'Kirkland' AND o.amount > 1500",
  },
  {
    value: 'one',
    label: 'One customer’s orders',
    sql: 'SELECT c.name, o.id, o.amount, o.status\nFROM customers c JOIN orders o ON o.customer_id = c.id\nWHERE c.id = 4242',
  },
  {
    value: 'view',
    label: 'Through a view (rewrite)',
    sql: "SELECT c.city, b.id, b.amount\nFROM customers c JOIN big_orders b ON b.customer_id = c.id\nWHERE c.city = 'Yakima'\nLIMIT 25",
  },
  {
    value: 'one_table',
    label: 'Single table',
    sql: "SELECT id, city FROM customers WHERE city = 'Olympia'",
  },
] as const;

const STAGES = [
  { value: 'parse' as const, label: 'Parse', title: 'raw_parser(): text → SelectStmt, no catalog involved' },
  { value: 'analyze' as const, label: 'Analyze', title: 'parse_analyze(): names → OIDs, Vars and OpExprs' },
  { value: 'rewrite' as const, label: 'Rewrite', title: 'QueryRewrite(): views and RLS quals expanded' },
  { value: 'plan' as const, label: 'Plan', title: 'planner(): paths costed, cheapest becomes the plan tree' },
  { value: 'execute' as const, label: 'Execute', title: 'ExecutorRun(): next() through the operator tree' },
];

/* ================================================================= component */

export default function QueryLifecycleStepper() {
  const [example, setExample] = useState<string>('join');
  const [sql, setSql] = useState<string>(EXAMPLES[0].sql);
  const [stage, setStage] = useState<'parse' | 'analyze' | 'rewrite' | 'plan' | 'execute'>('plan');
  const [idxCity, setIdxCity] = useState(false);
  const [idxCust, setIdxCust] = useState(true);
  const [idxAmt, setIdxAmt] = useState(false);
  const [rpc, setRpc] = useState(4);
  const [poolCap, setPoolCap] = useState(64);
  const [steps, setSteps] = useState(0);
  const [ref, width] = useSize(760);
  const tip = useTip();

  const cfg: Cfg = { rpc, idxCity, idxCust, idxAmt, pool: poolCap };

  const compiled = useMemo(() => {
    try {
      const a = parse(sql);
      const an = analyze(a);
      const plan = planQuery(an, cfg);
      return { a, an, plan, err: null as SqlError | null, stageOfErr: null as string | null };
    } catch (e) {
      if (e instanceof SqlError) {
        try {
          const a = parse(sql);
          return { a, an: null, plan: null, err: e, stageOfErr: 'analyze' };
        } catch {
          return { a: null, an: null, plan: null, err: e, stageOfErr: 'parse' };
        }
      }
      throw e;
    }
  }, [sql, rpc, idxCity, idxCust, idxAmt]);

  const run = useMemo(() => {
    if (!compiled.plan || !compiled.an) return null;
    return execute(compiled.plan, compiled.an, poolCap, steps);
  }, [compiled, poolCap, steps]);

  const err = compiled.err;
  const plan = compiled.plan;
  const an = compiled.an;

  /* ------------------------------------------------------------- geometry */
  const svgW = Math.max(width, 720);
  let figure: React.ReactNode = null;
  let height = 260;

  if (err && (compiled.stageOfErr === 'parse' || stage !== 'parse')) {
    const line = sql.slice(0, err.pos).split('\n').length;
    const lineStart = sql.lastIndexOf('\n', Math.max(0, err.pos - 1)) + 1;
    const col = err.pos - lineStart;
    const lineText = sql.slice(lineStart, sql.indexOf('\n', lineStart) === -1 ? undefined : sql.indexOf('\n', lineStart));
    height = 96;
    figure = (
      <svg width={svgW} height={height} role="img" aria-label="Query compilation error">
        <text x={0} y={22} fill="var(--viz-critical)" fontWeight={600}>
          ERROR: {err.message}
        </text>
        <text x={0} y={46} fill="var(--viz-ink-2)">
          LINE {line}: {lineText}
        </text>
        <text x={0} y={62} fill="var(--viz-2)">
          ↑ at character {col + 1} of that line
        </text>
        <text x={0} y={86} fill="var(--viz-ink-muted)">
          {compiled.stageOfErr === 'parse'
            ? 'Raised by the raw parser, before any catalog lookup: no relation has been opened and no lock has been taken.'
            : 'Raised by parse analysis: the grammar accepted this, the catalog did not.'}
        </text>
      </svg>
    );
  } else if (stage === 'parse' || stage === 'analyze' || stage === 'rewrite') {
    const rows = flattenTree(astTree(compiled.a as Ast, an, stage));
    const rowH = 21;
    height = rows.length * rowH + 16;
    const noteX = Math.min(330, svgW * 0.44);
    figure = (
      <svg width={svgW} height={height} role="img" aria-label={`${stage} tree for the query`}>
        {rows.map((r, i) => {
          const y = 16 + i * rowH;
          const x = r.d * 16;
          return (
            <g key={i}>
              {r.d > 0 ? (
                <path
                  d={`M ${x - 10} ${y - rowH + 4} L ${x - 10} ${y - 3} L ${x - 3} ${y - 3}`}
                  fill="none"
                  stroke="var(--viz-grid)"
                />
              ) : null}
              <text x={x} y={y} fill={r.n.added ? 'var(--viz-2)' : 'var(--viz-ink)'} fontWeight={600}>
                {r.n.label}
              </text>
              <text x={x + r.n.label.length * 6.6 + 8} y={y} fill="var(--viz-ink-2)">
                {r.n.text ?? ''}
              </text>
              {r.n.note ? (
                <text x={noteX} y={y} fill={r.n.added ? 'var(--viz-2)' : 'var(--viz-1)'}>
                  → {r.n.note}
                </text>
              ) : null}
            </g>
          );
        })}
      </svg>
    );
  } else if (plan) {
    /* ---------------------------------------------------- plan / execute */
    const exec = stage === 'execute';
    const BW = 248;
    const BH = exec ? 82 : 62;
    const HGAP = 16;
    const VGAP = 30;
    const pos = new Map<number, { x: number; y: number }>();
    let slot = 0;
    const place = (n: PNode, d: number): number => {
      if (n.children.length === 0) {
        const x = slot++ * (BW + HGAP);
        pos.set(n.id, { x, y: d * (BH + VGAP) });
        return x;
      }
      const xs = n.children.map((c) => place(c, d + 1));
      const x = (Math.min(...xs) + Math.max(...xs)) / 2;
      pos.set(n.id, { x, y: d * (BH + VGAP) });
      return x;
    };
    place(plan.root, 0);
    const maxDepth = Math.max(...plan.nodes.map((n) => (pos.get(n.id) as { y: number }).y));
    const treeW = slot * (BW + HGAP);
    const poolRows = Math.ceil(poolCap / 32);
    const poolH = exec ? poolRows * 15 + 30 : 0;
    height = maxDepth + BH + 20 + poolH;
    const w = Math.max(svgW, treeW + 8);

    figure = (
      <svg width={w} height={height} role="img" aria-label="Plan tree and buffer pool">
        {plan.nodes.map((n) =>
          n.children.map((c) => {
            const a = pos.get(n.id) as { x: number; y: number };
            const b = pos.get(c.id) as { x: number; y: number };
            return (
              <path
                key={`${n.id}-${c.id}`}
                d={`M ${a.x + BW / 2} ${a.y + BH} C ${a.x + BW / 2} ${a.y + BH + 16}, ${b.x + BW / 2} ${b.y - 16}, ${b.x + BW / 2} ${b.y}`}
                fill="none"
                stroke="var(--viz-axis)"
              />
            );
          }),
        )}
        {plan.nodes.map((n) => {
          const p = pos.get(n.id) as { x: number; y: number };
          const hot = exec && run ? run.touched.includes(n.id) : false;
          const i = n.id;
          const cnt = run ? Array.from({ length: C_SLOTS }, (_, s) => run.C[i * C_SLOTS + s]) : null;
          const rowsOut = run ? run.C[i * C_SLOTS + C_ROWS] : 0;
          const loops = run ? Math.max(1, run.C[i * C_SLOTS + C_LOOPS]) : 1;
          const hit = run ? run.C[i * C_SLOTS + C_HIT] : 0;
          const read = run ? run.C[i * C_SLOTS + C_READ] : 0;
          return (
            <g
              key={n.id}
              {...tip(
                <>
                  <strong>{n.title}</strong>
                  <br />
                  {n.hint}
                  {exec && cnt ? (
                    <>
                      <br />
                      next() calls: {fmtNum(cnt[C_CALLS])} · rows: {fmtNum(cnt[C_ROWS])} · buffers hit{' '}
                      {fmtNum(cnt[C_HIT])} / read {fmtNum(cnt[C_READ])}
                    </>
                  ) : null}
                </>,
              )}
            >
              <rect
                x={p.x}
                y={p.y}
                width={BW}
                height={BH}
                rx={8}
                fill="var(--viz-plane)"
                stroke={hot ? 'var(--viz-2)' : 'var(--viz-border)'}
                strokeWidth={hot ? 2 : 1}
              />
              <text x={p.x + 10} y={p.y + 17} fill="var(--viz-ink)" fontWeight={600}>
                {n.title.length > 36 ? `${n.title.slice(0, 35)}…` : n.title}
              </text>
              <text x={p.x + 10} y={p.y + 32} fill="var(--viz-ink-2)" style={{ fontVariantNumeric: 'tabular-nums' }}>
                (cost={n.startup.toFixed(2)}..{n.total.toFixed(2)} rows={fmtNum(n.rows)} width={n.width})
              </text>
              {n.detail[0] ? (
                <text x={p.x + 10} y={p.y + 46} fill="var(--viz-ink-muted)">
                  {n.detail[0].length > 41 ? `${n.detail[0].slice(0, 40)}…` : n.detail[0]}
                </text>
              ) : null}
              {n.detail[1] ? (
                <text x={p.x + 10} y={p.y + 58} fill="var(--viz-ink-muted)">
                  {n.detail[1].length > 41 ? `${n.detail[1].slice(0, 40)}…` : n.detail[1]}
                </text>
              ) : null}
              {exec ? (
                <text x={p.x + 10} y={p.y + BH - 20} fill="var(--viz-1)" style={{ fontVariantNumeric: 'tabular-nums' }}>
                  actual rows={fmtNum(rowsOut)} loops={fmtNum(loops)}
                </text>
              ) : null}
              {exec ? (
                <text x={p.x + 10} y={p.y + BH - 7} fill="var(--viz-ink-2)" style={{ fontVariantNumeric: 'tabular-nums' }}>
                  buffers: hit={fmtNum(hit)} read={fmtNum(read)}
                </text>
              ) : null}
            </g>
          );
        })}
        {exec && run ? (
          <g transform={`translate(0 ${maxDepth + BH + 20})`}>
            <text x={0} y={10} fill="var(--viz-ink-2)">
              buffer pool — {poolCap} frames × 8 KB, least-recently-used on the left
            </text>
            {Array.from({ length: poolCap }, (_, i) => {
              const f = run.frames[i];
              const col = i % 32;
              const rowI = Math.floor(i / 32);
              const x = col * 15;
              const y = 18 + rowI * 15;
              return (
                <g key={i} {...(f ? tip(<>{f.key.replace(/^([CO])([hi])/, '$1 · $2 · ')}</>) : {})}>
                  <rect
                    x={x}
                    y={y}
                    width={12}
                    height={12}
                    rx={2}
                    fill={
                      !f
                        ? 'var(--viz-neutral)'
                        : f.index
                          ? 'var(--viz-7)'
                          : f.rel === 'C'
                            ? 'var(--viz-1)'
                            : 'var(--viz-2)'
                    }
                    stroke="var(--viz-surface)"
                  />
                  {f ? (
                    <text x={x + 6} y={y + 9} textAnchor="middle" fill="var(--viz-surface)" fontWeight={600}>
                      {f.label}
                    </text>
                  ) : null}
                </g>
              );
            })}
          </g>
        ) : null}
      </svg>
    );
  }

  /* --------------------------------------------------------------- copy */
  let head = '';
  let body = '';
  if (err) {
    head = `${compiled.stageOfErr === 'parse' ? 'The parser' : 'Parse analysis'} rejected this statement.`;
    body =
      compiled.stageOfErr === 'parse'
        ? 'Nothing has touched the catalog yet: the raw parser only knows the grammar, which is why a misspelt keyword and a misspelt table name fail at different stages and with different messages.'
        : 'The grammar accepted the text; resolving it against pg_class and pg_attribute did not. This stage is also where the query takes an AccessShareLock on every relation it names.';
  } else if (stage === 'parse') {
    head = 'raw_parser(): text in, SelectStmt out.';
    body =
      'The lexer and grammar build a tree of RangeVars, ColumnRefs and A_Exprs that still holds nothing but names and literals. No relation has been opened, no type is known, and "SELECT * FROM no_such_table" parses perfectly.';
  } else if (stage === 'analyze') {
    head = 'parse_analyze(): names become OIDs, Vars and OpExprs.';
    body =
      'Each RangeVar becomes a range-table entry with a relid; each ColumnRef becomes a Var carrying varno/varattno and a type; each operator is resolved against pg_operator, which is where an implicit cast or a "operator does not exist" error appears.';
  } else if (stage === 'rewrite') {
    head = an?.rewritten
      ? 'QueryRewrite(): the view was expanded into its base relation.'
      : 'QueryRewrite(): nothing to do for this query.';
    body = an?.rewritten
      ? 'A view is stored as an ON SELECT DO INSTEAD rule; the rewriter substitutes its definition for the reference and pulls the resulting qual up into the parent query, so the planner never sees a view at all. Row-level-security policies are injected here too.'
      : 'With no views, rules or RLS policies in play the rewriter returns the Query unchanged. Load the view example to watch big_orders disappear and its amount > 1500 qual appear in the WHERE clause.';
  } else if (stage === 'plan' && plan) {
    const chosen = plan.alts.filter((x) => x.chosen);
    head = `Cheapest total plan: ${plan.root.total.toFixed(2)} cost units${plan.join !== '—' ? `, joined with a ${plan.join}` : ''}.`;
    body = `${chosen.map((c) => `${c.what}: ${c.kind}`).join('; ')}. Costs are in units of one sequential page read; open "Show the numbers" for every path that was built and rejected.`;
  } else if (plan && run) {
    head = run.steps === 0 ? 'Nothing has executed yet.' : run.done ? `Finished: ${fmtNum(run.produced)} rows returned.` : `After ${fmtNum(run.steps)} next() call${run.steps === 1 ? '' : 's'} at the root.`;
    const hot = plan.nodes.filter((n) => run.touched.includes(n.id)).map((n) => n.op);
    body =
      run.steps === 0
        ? 'Press Step. One next() at the root ripples all the way down: a blocking node such as Hash drains its entire child before the first row can come back, while a Nested Loop returns as soon as its inner side yields.'
        : `The last next() entered ${hot.length ? hot.join(' → ') : 'no operator'}${
            run.done ? ' and returned nothing, which is how the executor learns the scan is exhausted' : ''
          }. Every tuple access went through the buffer pool: ${fmtNum(run.hits)} hits, ${fmtNum(run.misses)} misses, ${fmtNum(run.evictions)} evictions so far.`;
  }

  const estRows = plan ? plan.root.rows : 0;
  const actRows = run ? run.produced : 0;

  const legend =
    stage === 'execute' ? (
      <Legend
        items={[
          { label: 'C — customers heap page', color: 'var(--viz-1)' },
          { label: 'O — orders heap page', color: 'var(--viz-2)' },
          { label: 'i — index page', color: 'var(--viz-7)' },
          { label: 'free frame', color: 'var(--viz-neutral)' },
          { label: 'operator touched by the last next()', color: 'var(--viz-2)', shape: 'line' },
        ]}
      />
    ) : stage === 'plan' ? (
      <Legend
        items={[
          { label: 'cost = startup..total, in sequential-page-read units', color: 'var(--viz-ink-2)' },
          { label: 'rows / width = the planner’s estimate', color: 'var(--viz-1)' },
        ]}
      />
    ) : (
      <Legend
        items={[
          { label: 'raw parse node', color: 'var(--viz-ink-2)' },
          { label: 'resolved by analysis', color: 'var(--viz-1)' },
          { label: 'added by the rewriter', color: 'var(--viz-2)' },
        ]}
      />
    );

  return (
    <VizPanel
      title="One SELECT, five stages"
      subtitle="Edit the SQL, walk the stages, then step the executor. Toggling an index or dragging random_page_cost re-plans the query, and the buffer counters move with the new plan."
      controls={
        <>
          <Choice
            label="Example"
            value={example}
            onChange={(v) => {
              setExample(v);
              const ex = EXAMPLES.find((e) => e.value === v);
              if (ex) setSql(ex.sql);
              setSteps(0);
            }}
            options={EXAMPLES.map((e) => ({ value: e.value, label: e.label }))}
          />
          <Segmented label="Stage" value={stage} onChange={setStage} options={STAGES} />
          <Check
            label="index on orders(customer_id)"
            checked={idxCust}
            onChange={(b) => {
              setIdxCust(b);
              setSteps(0);
            }}
          />
          <Check
            label="index on customers(city)"
            checked={idxCity}
            onChange={(b) => {
              setIdxCity(b);
              setSteps(0);
            }}
          />
          <Check
            label="index on orders(amount)"
            checked={idxAmt}
            onChange={(b) => {
              setIdxAmt(b);
              setSteps(0);
            }}
          />
          <Slider
            label="random_page_cost"
            min={1}
            max={10}
            step={0.1}
            value={rpc}
            onChange={(n) => {
              setRpc(n);
              setSteps(0);
            }}
            format={(n) => n.toFixed(1)}
          />
          <Slider
            label="buffer pool"
            min={16}
            max={256}
            step={16}
            value={poolCap}
            onChange={(n) => {
              setPoolCap(n);
              setSteps(0);
            }}
            format={(n) => `${n} frames`}
          />
          <Button
            primary
            onClick={() => {
              setStage('execute');
              setSteps((s) => s + 1);
            }}
            disabled={!plan || (run?.done ?? false)}
          >
            Step
          </Button>
          <Button
            onClick={() => {
              setStage('execute');
              setSteps(400_000);
            }}
            disabled={!plan}
            title="Run the plan to completion and total the counters"
          >
            Run
          </Button>
          <Button onClick={() => setSteps(0)} disabled={steps === 0}>
            Reset
          </Button>
        </>
      }
      legend={legend}
      stats={
        <Stats
          items={
            stage === 'execute'
              ? [
                  { label: 'Rows returned', value: fmtNum(actRows), hint: 'One DataRow message each' },
                  { label: 'Estimated rows', value: fmtNum(estRows), hint: 'What the planner costed the plan on' },
                  { label: 'Buffer hits', value: fmtNum(run?.hits ?? 0) },
                  { label: 'Disk reads', value: fmtNum(run?.misses ?? 0), hint: 'Buffer misses — the only requests that reach the OS' },
                  {
                    label: 'Hit rate',
                    value: run && run.hits + run.misses > 0 ? `${((100 * run.hits) / (run.hits + run.misses)).toFixed(1)}%` : '—',
                  },
                  { label: 'Evictions', value: fmtNum(run?.evictions ?? 0) },
                ]
              : [
                  { label: 'Estimated cost', value: plan ? plan.root.total.toFixed(2) : '—', hint: 'Units of one sequential 8 KB page read' },
                  { label: 'Estimated rows', value: plan ? fmtNum(plan.root.rows) : '—' },
                  { label: 'Join method', value: plan ? plan.join : '—' },
                  { label: 'Paths considered', value: plan ? fmtNum(plan.alts.length) : '—' },
                  { label: 'seq_page_cost', value: SEQ_PAGE_COST.toFixed(2) },
                  { label: 'random_page_cost', value: rpc.toFixed(1) },
                ]
          }
        />
      }
      note={
        <Note>
          <strong>{head}</strong> {body}
        </Note>
      }
      table={
        plan ? (
          <>
            <table className="viz-table">
              <thead>
                <tr>
                  <th>For</th>
                  <th>Path</th>
                  <th>Startup</th>
                  <th>Total</th>
                  <th>Est. rows</th>
                  <th>Chosen</th>
                  <th>Why that number</th>
                </tr>
              </thead>
              <tbody>
                {plan.alts.map((a2, i) => (
                  <tr key={i}>
                    <td>{a2.what}</td>
                    <td>{a2.kind}</td>
                    <td>{Number.isNaN(a2.total) ? '—' : a2.startup.toFixed(2)}</td>
                    <td>{Number.isNaN(a2.total) ? 'not built' : a2.total.toFixed(2)}</td>
                    <td>{fmtNum(a2.rows)}</td>
                    <td>{a2.chosen ? 'yes' : ''}</td>
                    <td>{a2.why}</td>
                  </tr>
                ))}
              </tbody>
            </table>
            {run && run.steps > 0 ? (
              <table className="viz-table">
                <thead>
                  <tr>
                    <th>Operator</th>
                    <th>next() calls</th>
                    <th>Rows out</th>
                    <th>Loops</th>
                    <th>Buffer hits</th>
                    <th>Disk reads</th>
                  </tr>
                </thead>
                <tbody>
                  {plan.nodes.map((n) => (
                    <tr key={n.id}>
                      <td>{n.title}</td>
                      <td>{fmtNum(run.C[n.id * C_SLOTS + C_CALLS])}</td>
                      <td>{fmtNum(run.C[n.id * C_SLOTS + C_ROWS])}</td>
                      <td>{fmtNum(Math.max(1, run.C[n.id * C_SLOTS + C_LOOPS]))}</td>
                      <td>{fmtNum(run.C[n.id * C_SLOTS + C_HIT])}</td>
                      <td>{fmtNum(run.C[n.id * C_SLOTS + C_READ])}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            ) : null}
            {run && run.rows.length > 0 && an ? (
              <table className="viz-table">
                <thead>
                  <tr>
                    {an.targets.map((t) => (
                      <th key={t.label}>{t.label}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {run.rows.map((r, i) => (
                    <tr key={i}>
                      {r.map((v, j) => (
                        <td key={j}>{String(v)}</td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            ) : null}
          </>
        ) : null
      }
    >
      <div ref={ref}>
        <textarea
          value={sql}
          spellCheck={false}
          aria-label="SELECT statement"
          rows={Math.min(8, Math.max(3, sql.split('\n').length))}
          onChange={(e) => {
            setSql(e.currentTarget.value);
            setSteps(0);
          }}
          style={{
            width: '100%',
            boxSizing: 'border-box',
            marginBottom: '.6rem',
            fontFamily: 'ui-monospace, SFMono-Regular, Menlo, monospace',
            fontSize: '.8125rem',
            lineHeight: 1.5,
            color: 'var(--viz-ink)',
            background: 'var(--viz-plane)',
            border: `1px solid ${err ? 'var(--viz-critical)' : 'var(--viz-border)'}`,
            borderRadius: '6px',
            padding: '.5rem .6rem',
            resize: 'vertical',
          }}
        />
        <TooltipHost>{figure}</TooltipHost>
      </div>
    </VizPanel>
  );
}
