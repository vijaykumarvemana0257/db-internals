import { useMemo, useState } from 'react';
import { VizPanel, Choice, Check, Button, Legend, Stats, Note, useSize } from './Viz';

/**
 * Partial and expression index usability, modelled on PostgreSQL 17's planner.
 *
 * - CREATE INDEX: every function, operator and cast in the key expressions and the WHERE predicate must be
 *   IMMUTABLE (indexcmds.c CheckPredicate / ComputeIndexAttrs; the predicate is checked first). Volatilities come
 *   from the PostgreSQL 17 pg_proc/pg_operator/pg_cast catalogs (unaccent from contrib/unaccent 1.1).
 * - Expressions are normalized the way eval_const_expressions() matters here: immutable calls on constants are
 *   folded (lower('A') -> 'a'), x = true -> x, x = false -> NOT x, NOT is pushed through negator operators,
 *   De Morgan and IS [NOT] NULL, AND/OR are flattened, BETWEEN becomes two comparisons, IN (list) becomes
 *   = ANY (a one-element IN becomes =). Stable functions such as now() are not folded.
 * - The implication proof follows predtest.c: predicate_implied_by_recurse()'s AND/OR/atom rules, IN lists of up
 *   to MAX_SAOP_ARRAY_SIZE (100) constants treated as OR/AND, equal() matches, "foo IS NOT NULL" implied by a
 *   clause strict in foo, and operator_predicate_proof(): same collation, one operand equal(), the other a Const on
 *   both sides (a Param or now() fails), then the btree test "pred_const test_op clause_const" from BT_implic_table,
 *   refused when the test operator is only STABLE (timestamptz vs date).
 * - check_index_predicates(): clauses implied by the predicate are dropped from the plan unless the table is the
 *   target of UPDATE/DELETE/MERGE/FOR UPDATE.
 * - Key matching follows match_index_to_operand(): the operand must be equal() to the key expression, the other side
 *   must not reference the table or be volatile, and the comparison's collation must equal the key's.
 * Lab assumptions: string constants compare by code point (the C collation), timestamps parse as UTC, only a small
 * catalog of functions is known, and LIKE is never turned into an index condition.
 */
/* ================================================================ types */

export type Ty = 'int4' | 'int8' | 'numeric' | 'text' | 'bool' | 'timestamptz' | 'timestamp' | 'date' | 'interval' | 'jsonb' | 'text[]' | 'unknown';
export type Vol = 'i' | 's' | 'v';

/** The table every expression in the lab is resolved against. */
export const SCHEMA: Record<string, Ty> = {
  id: 'int8',
  customer_id: 'int8',
  status: 'text',
  amount: 'int4',
  email: 'text',
  created_at: 'timestamptz',
  deleted_at: 'timestamptz',
  data: 'jsonb',
  archived: 'bool',
};

export const TYPE_NAME: Record<Ty, string> = {
  int4: 'integer',
  int8: 'bigint',
  numeric: 'numeric',
  text: 'text',
  bool: 'boolean',
  timestamptz: 'timestamp with time zone',
  timestamp: 'timestamp without time zone',
  date: 'date',
  interval: 'interval',
  jsonb: 'jsonb',
  'text[]': 'text[]',
  unknown: 'unknown',
};

export type Node =
  | { k: 'col'; name: string; ty: Ty; ecoll?: string }
  | { k: 'const'; ty: Ty; v: number | string | boolean | null; raw: string; ecoll?: string }
  | { k: 'param'; n: number; ty: Ty; ecoll?: string }
  | { k: 'func'; name: string; args: Node[]; ty: Ty; vol: Vol; strict: boolean; sig: string; ecoll?: string }
  | { k: 'op'; op: string; l: Node; r: Node; ty: Ty; vol: Vol; strict: boolean; sig: string; coll: string | null; ecoll?: string }
  | { k: 'cast'; arg: Node; ty: Ty; vol: Vol; sig: string; ecoll?: string }
  | { k: 'and'; args: Node[]; ty: Ty; ecoll?: string }
  | { k: 'or'; args: Node[]; ty: Ty; ecoll?: string }
  | { k: 'not'; arg: Node; ty: Ty; ecoll?: string }
  | { k: 'nulltest'; arg: Node; isNull: boolean; ty: Ty; ecoll?: string }
  | { k: 'booltest'; arg: Node; test: 'IS TRUE' | 'IS NOT TRUE' | 'IS FALSE' | 'IS NOT FALSE'; ty: Ty; ecoll?: string }
  | { k: 'saop'; op: string; useOr: boolean; l: Node; elems: Node[]; ty: Ty; coll: string | null; vol: Vol; ecoll?: string };

export class SqlError extends Error {}

const NUMERIC: Ty[] = ['int4', 'int8', 'numeric'];
const isNum = (t: Ty) => NUMERIC.includes(t);
const isTextish = (t: Ty) => t === 'text';
const BTREE_OPS = ['<', '<=', '=', '>=', '>', '<>'];

/* ============================================================ tokenizer */

type Tok = { t: 'num' | 'str' | 'id' | 'qid' | 'param' | 'op'; v: string; pos: number };

export function tokenize(src: string): Tok[] {
  const out: Tok[] = [];
  let i = 0;
  const ops = ['::', '->>', '->', '#>>', '||', '<=', '>=', '<>', '!=', '@>', '=', '<', '>', '+', '-', '*', '/', '(', ')', ','];
  while (i < src.length) {
    const c = src[i];
    if (/\s/.test(c)) {
      i++;
      continue;
    }
    if (/[0-9]/.test(c) || (c === '.' && /[0-9]/.test(src[i + 1] ?? ''))) {
      let j = i;
      while (j < src.length && /[0-9.]/.test(src[j])) j++;
      out.push({ t: 'num', v: src.slice(i, j), pos: i });
      i = j;
      continue;
    }
    if (c === "'") {
      let j = i + 1;
      let s = '';
      for (;;) {
        if (j >= src.length) throw new SqlError('unterminated quoted string');
        if (src[j] === "'") {
          if (src[j + 1] === "'") {
            s += "'";
            j += 2;
            continue;
          }
          break;
        }
        s += src[j++];
      }
      out.push({ t: 'str', v: s, pos: i });
      i = j + 1;
      continue;
    }
    if (c === '"') {
      const j = src.indexOf('"', i + 1);
      if (j < 0) throw new SqlError('unterminated quoted identifier');
      out.push({ t: 'qid', v: src.slice(i + 1, j), pos: i });
      i = j + 1;
      continue;
    }
    if (c === '$' && /[0-9]/.test(src[i + 1] ?? '')) {
      let j = i + 1;
      while (j < src.length && /[0-9]/.test(src[j])) j++;
      out.push({ t: 'param', v: src.slice(i + 1, j), pos: i });
      i = j;
      continue;
    }
    if (/[A-Za-z_]/.test(c)) {
      let j = i;
      while (j < src.length && /[A-Za-z0-9_]/.test(src[j])) j++;
      out.push({ t: 'id', v: src.slice(i, j).toLowerCase(), pos: i });
      i = j;
      continue;
    }
    const op = ops.find((o) => src.startsWith(o, i));
    if (!op) throw new SqlError(`syntax error at or near "${c}"`);
    out.push({ t: 'op', v: op, pos: i });
    i += op.length;
  }
  return out;
}

/* ============================================================ typing helpers */

const MONTHS = /^(\d{4})-(\d{2})-(\d{2})(?:[ T](\d{2}):(\d{2})(?::(\d{2}))?)?$/;

/** Parses a literal into a comparable value. Timestamps are read as UTC (the lab's session TimeZone). */
export function coerceLiteral(raw: string, ty: Ty): Node {
  const s = raw.trim();
  const bad = () => new SqlError(`invalid input syntax for type ${TYPE_NAME[ty]}: "${raw}"`);
  switch (ty) {
    case 'int4':
    case 'int8':
      if (!/^-?\d+$/.test(s)) throw bad();
      return { k: 'const', ty, v: Number(s), raw: s };
    case 'numeric':
      if (!/^-?(\d+\.?\d*|\.\d+)$/.test(s)) throw bad();
      return { k: 'const', ty, v: Number(s), raw: s };
    case 'timestamptz':
    case 'timestamp':
    case 'date': {
      const m = MONTHS.exec(s);
      if (!m) throw bad();
      const ms = Date.UTC(+m[1], +m[2] - 1, +m[3], ty === 'date' ? 0 : +(m[4] ?? 0), ty === 'date' ? 0 : +(m[5] ?? 0), ty === 'date' ? 0 : +(m[6] ?? 0));
      if (Number.isNaN(ms)) throw bad();
      return { k: 'const', ty, v: ms, raw: s };
    }
    case 'interval': {
      const m = /^(\d+)\s*(second|minute|hour|day|week|month|year)s?$/i.exec(s);
      if (!m) throw bad();
      const unit: Record<string, number> = { second: 1e3, minute: 6e4, hour: 36e5, day: 864e5, week: 6048e5, month: 2592e6, year: 31536e6 };
      return { k: 'const', ty, v: +m[1] * unit[m[2].toLowerCase()], raw: s };
    }
    case 'bool':
      if (/^(t|true|y|yes|on|1)$/i.test(s)) return { k: 'const', ty, v: true, raw: 'true' };
      if (/^(f|false|n|no|off|0)$/i.test(s)) return { k: 'const', ty, v: false, raw: 'false' };
      throw bad();
    default:
      return { k: 'const', ty, v: raw, raw };
  }
}

const collOf = (n: Node): string | null => (n.ecoll ? n.ecoll : isTextish(n.ty) ? 'default' : null);

function resolveCompareColl(l: Node, r: Node): string | null {
  if (!isTextish(l.ty) && !isTextish(r.ty)) return null;
  if (l.ecoll && r.ecoll && l.ecoll !== r.ecoll) throw new SqlError(`collation mismatch between explicit collations "${l.ecoll}" and "${r.ecoll}"`);
  return l.ecoll ?? r.ecoll ?? 'default';
}

/** Castable types, with the volatility PostgreSQL 17 records for the conversion. */
function castVol(from: Ty, to: Ty): { vol: Vol; sig: string } | null {
  if (from === to) return { vol: 'i', sig: `${TYPE_NAME[to]}` };
  const sig = `${TYPE_NAME[to]}(${TYPE_NAME[from]})`;
  if (isNum(from) && isNum(to)) return { vol: 'i', sig };
  if (to === 'text') {
    // Casts to text go through the type's output function: date/time output depends on DateStyle.
    const stable = ['timestamptz', 'timestamp', 'date', 'interval'].includes(from);
    return { vol: stable ? 's' : 'i', sig: `${TYPE_NAME[from]} output function (${stable ? 'depends on DateStyle' : 'immutable'})` };
  }
  const table: Record<string, Vol> = {
    'timestamptz>date': 's',
    'timestamp>date': 'i',
    'timestamptz>timestamp': 's',
    'timestamp>timestamptz': 's',
    'date>timestamptz': 's',
    'date>timestamp': 'i',
    'text>jsonb': 'i',
  };
  const v = table[`${from}>${to}`];
  return v ? { vol: v, sig } : null;
}

function typeFromName(name: string): Ty {
  const m: Record<string, Ty> = {
    int: 'int4', integer: 'int4', int4: 'int4', bigint: 'int8', int8: 'int8', numeric: 'numeric', decimal: 'numeric',
    text: 'text', varchar: 'text', date: 'date', timestamptz: 'timestamptz', timestamp: 'timestamp', interval: 'interval',
    jsonb: 'jsonb', bool: 'bool', boolean: 'bool',
  };
  const t = m[name];
  if (!t) throw new SqlError(`type "${name}" is not in this lab's catalog`);
  return t;
}

export function makeCast(arg: Node, to: Ty): Node {
  if (arg.k === 'const' && arg.v === null) return { ...arg, ty: to };
  if (arg.k === 'const' && arg.ty === 'unknown') return coerceLiteral(String(arg.v), to);
  if (arg.ty === to) return arg;
  const cv = castVol(arg.ty, to);
  if (!cv) throw new SqlError(`cannot cast type ${TYPE_NAME[arg.ty]} to ${TYPE_NAME[to]}`);
  return { k: 'cast', arg, ty: to, vol: cv.vol, sig: cv.sig };
}

/** Unifies an operand pair for an operator: unknown literals take the other side's type. */
function unify(l: Node, r: Node): [Node, Node] {
  if (l.k === 'param' && l.ty === 'unknown') l = { ...l, ty: r.ty === 'unknown' ? 'text' : r.ty };
  if (r.k === 'param' && r.ty === 'unknown') r = { ...r, ty: l.ty === 'unknown' ? 'text' : l.ty };
  if (l.k === 'const' && l.v === null && l.ty === 'unknown') l = { ...l, ty: r.ty === 'unknown' ? 'text' : r.ty };
  if (r.k === 'const' && r.v === null && r.ty === 'unknown') r = { ...r, ty: l.ty === 'unknown' ? 'text' : l.ty };
  if (l.k === 'const' && l.ty === 'unknown' && r.ty !== 'unknown') l = coerceLiteral(String(l.v), r.ty === 'jsonb' ? 'text' : r.ty);
  else if (r.k === 'const' && r.ty === 'unknown' && l.ty !== 'unknown') r = coerceLiteral(String(r.v), l.ty === 'jsonb' ? 'text' : l.ty);
  else if (l.k === 'const' && l.ty === 'unknown' && r.k === 'const' && r.ty === 'unknown') {
    l = { ...l, ty: 'text' };
    r = { ...r, ty: 'text' };
  }
  return [l, r];
}

export function makeCompare(op: string, l0: Node, r0: Node): Node {
  if (op === '!=') op = '<>';
  let [l, r] = unify(l0, r0);
  let vol: Vol = 'i';
  let sig = `${TYPE_NAME[l.ty]} ${op} ${TYPE_NAME[r.ty]}`;
  if (isNum(l.ty) && isNum(r.ty)) {
    if ((l.ty === 'numeric') !== (r.ty === 'numeric')) {
      // integer vs numeric: the integer side is cast to numeric, so the operand is no longer the bare column.
      if (l.ty !== 'numeric') l = makeCast(l, 'numeric');
      else r = makeCast(r, 'numeric');
      sig = `numeric ${op} numeric`;
    }
  } else if (l.ty !== r.ty) {
    const dt = ['timestamptz', 'timestamp', 'date'];
    if (dt.includes(l.ty) && dt.includes(r.ty)) {
      // Cross-type datetime comparisons that involve timestamptz depend on TimeZone.
      vol = l.ty === 'timestamptz' || r.ty === 'timestamptz' ? 's' : 'i';
    } else {
      throw new SqlError(`operator does not exist: ${TYPE_NAME[l.ty]} ${op} ${TYPE_NAME[r.ty]}`);
    }
  } else if (!['int4', 'int8', 'numeric', 'text', 'bool', 'timestamptz', 'timestamp', 'date', 'interval'].includes(l.ty)) {
    throw new SqlError(`operator does not exist: ${TYPE_NAME[l.ty]} ${op} ${TYPE_NAME[r.ty]}`);
  }
  if (l.ty === 'bool' && !['=', '<>'].includes(op)) throw new SqlError(`operator ${op} on boolean is not modelled`);
  return { k: 'op', op, l, r, ty: 'bool', vol, strict: true, sig, coll: resolveCompareColl(l, r) };
}

function makeBinary(op: string, l0: Node, r0: Node): Node {
  if (BTREE_OPS.includes(op) || op === '!=') return makeCompare(op, l0, r0);
  let l = l0;
  let r = r0;
  const opNode = (ty: Ty, vol: Vol, sig: string, strict = true): Node => ({ k: 'op', op, l, r, ty, vol, strict, sig, coll: isTextish(ty) ? collOf(l) : null });
  if (op === '->>' || op === '->') {
    if (l.ty !== 'jsonb') throw new SqlError(`operator does not exist: ${TYPE_NAME[l.ty]} ${op} ${TYPE_NAME[r.ty]}`);
    if (r.ty === 'unknown') r = { ...(r as Node), ty: 'text' } as Node;
    if (r.ty !== 'text' && r.ty !== 'int4') throw new SqlError(`operator does not exist: jsonb ${op} ${TYPE_NAME[r.ty]}`);
    return opNode(op === '->>' ? 'text' : 'jsonb', 'i', op === '->>' ? 'jsonb_object_field_text' : 'jsonb_object_field');
  }
  if (op === '#>>') {
    if (l.ty !== 'jsonb') throw new SqlError(`operator does not exist: ${TYPE_NAME[l.ty]} #>> ${TYPE_NAME[r.ty]}`);
    if (r.k === 'const' && r.ty === 'unknown') r = { ...r, ty: 'text[]' };
    return opNode('text', 'i', 'jsonb_extract_path_text');
  }
  if (op === '@>') {
    [l, r] = unify(l, r);
    if (l.ty !== 'jsonb' || r.ty !== 'text' && r.ty !== 'jsonb') throw new SqlError(`operator does not exist: ${TYPE_NAME[l.ty]} @> ${TYPE_NAME[r.ty]}`);
    if (r.k === 'const') r = { ...r, ty: 'jsonb' };
    return opNode('bool', 'i', 'jsonb_contains');
  }
  if (op === '||') {
    if (l.ty === 'unknown') l = { ...(l as Node), ty: 'text' } as Node;
    if (r.ty === 'unknown') r = { ...(r as Node), ty: 'text' } as Node;
    if (l.ty !== 'text' && r.ty !== 'text') throw new SqlError(`operator does not exist: ${TYPE_NAME[l.ty]} || ${TYPE_NAME[r.ty]}`);
    // text || non-text is an inlined SQL function: $1 || $2::text. Its volatility is that of the output function.
    if (l.ty !== 'text') l = makeCast(l, 'text');
    if (r.ty !== 'text') r = makeCast(r, 'text');
    return opNode('text', 'i', 'textcat');
  }
  if (op === '~~' || op === '~~*' || op === '!~~' || op === '!~~*') {
    [l, r] = unify(l, r);
    if (l.ty !== 'text' || r.ty !== 'text') throw new SqlError(`operator does not exist: ${TYPE_NAME[l.ty]} ${op} ${TYPE_NAME[r.ty]}`);
    return { k: 'op', op, l, r, ty: 'bool', vol: 'i', strict: true, sig: op.includes('*') ? 'texticlike' : 'textlike', coll: resolveCompareColl(l, r) };
  }
  if (['+', '-', '*', '/'].includes(op)) {
    if (l.k === 'const' && l.ty === 'unknown' && r.ty === 'timestamptz') l = coerceLiteral(String(l.v), 'interval');
    if (r.k === 'const' && r.ty === 'unknown' && ['timestamptz', 'timestamp', 'date'].includes(l.ty)) {
      r = /^\s*\d+\s*[a-z]/i.test(String(r.v)) ? coerceLiteral(String(r.v), 'interval') : coerceLiteral(String(r.v), l.ty);
    }
    [l, r] = unify(l, r);
    if (isNum(l.ty) && isNum(r.ty)) {
      const ty: Ty = l.ty === 'numeric' || r.ty === 'numeric' ? 'numeric' : l.ty === 'int8' || r.ty === 'int8' ? 'int8' : 'int4';
      return opNode(ty, 'i', `${TYPE_NAME[l.ty]} ${op} ${TYPE_NAME[r.ty]}`);
    }
    if ((op === '+' || op === '-') && l.ty === 'timestamptz' && r.ty === 'interval') return opNode('timestamptz', 's', `timestamptz_${op === '+' ? 'pl' : 'mi'}_interval`);
    if ((op === '+' || op === '-') && l.ty === 'timestamp' && r.ty === 'interval') return opNode('timestamp', 'i', `timestamp_${op === '+' ? 'pl' : 'mi'}_interval`);
    if (op === '-' && l.ty === 'timestamptz' && r.ty === 'timestamptz') return opNode('interval', 'i', 'timestamptz_mi');
    throw new SqlError(`operator does not exist: ${TYPE_NAME[l.ty]} ${op} ${TYPE_NAME[r.ty]}`);
  }
  throw new SqlError(`operator ${op} is not modelled`);
}

type FnDef = { args: (Ty | 'any')[]; variadic?: boolean; ret: Ty; vol: Vol; strict: boolean };
/** Volatility and strictness exactly as pg_proc records them in PostgreSQL 17 (unaccent from contrib/unaccent 1.1). */
const FUNCS: Record<string, FnDef[]> = {
  lower: [{ args: ['text'], ret: 'text', vol: 'i', strict: true }],
  upper: [{ args: ['text'], ret: 'text', vol: 'i', strict: true }],
  btrim: [{ args: ['text'], ret: 'text', vol: 'i', strict: true }],
  length: [{ args: ['text'], ret: 'int4', vol: 'i', strict: true }],
  md5: [{ args: ['text'], ret: 'text', vol: 'i', strict: true }],
  unaccent: [{ args: ['text'], ret: 'text', vol: 's', strict: true }],
  concat: [{ args: ['any'], variadic: true, ret: 'text', vol: 's', strict: false }],
  concat_ws: [{ args: ['text', 'any'], variadic: true, ret: 'text', vol: 's', strict: false }],
  now: [{ args: [], ret: 'timestamptz', vol: 's', strict: true }],
  clock_timestamp: [{ args: [], ret: 'timestamptz', vol: 'v', strict: true }],
  random: [{ args: [], ret: 'numeric', vol: 'v', strict: true }],
  date_trunc: [
    { args: ['text', 'timestamptz'], ret: 'timestamptz', vol: 's', strict: true },
    { args: ['text', 'timestamptz', 'text'], ret: 'timestamptz', vol: 'i', strict: true },
    { args: ['text', 'timestamp'], ret: 'timestamp', vol: 'i', strict: true },
  ],
  date_part: [
    { args: ['text', 'timestamptz'], ret: 'numeric', vol: 's', strict: true },
    { args: ['text', 'timestamp'], ret: 'numeric', vol: 'i', strict: true },
    { args: ['text', 'date'], ret: 'numeric', vol: 'i', strict: true },
  ],
  to_char: [
    { args: ['timestamptz', 'text'], ret: 'text', vol: 's', strict: true },
    { args: ['timestamp', 'text'], ret: 'text', vol: 's', strict: true },
    { args: ['int8', 'text'], ret: 'text', vol: 's', strict: true },
  ],
  jsonb_extract_path_text: [{ args: ['jsonb', 'text'], variadic: true, ret: 'text', vol: 'i', strict: true }],
  date: [
    { args: ['timestamptz'], ret: 'date', vol: 's', strict: true },
    { args: ['timestamp'], ret: 'date', vol: 'i', strict: true },
  ],
  timezone: [
    { args: ['text', 'timestamptz'], ret: 'timestamp', vol: 'i', strict: true },
    { args: ['text', 'timestamp'], ret: 'timestamptz', vol: 'i', strict: true },
  ],
};

export function makeFunc(name: string, args0: Node[]): Node {
  if (name === 'coalesce') {
    if (args0.length < 1) throw new SqlError('COALESCE needs arguments');
    const target = args0.find((a) => a.ty !== 'unknown')?.ty ?? 'text';
    const args = args0.map((a) => (a.ty === target ? a : makeCast(a, target)));
    const vol: Vol = args.some((a) => volatilityOf(a) === 'v') ? 'v' : args.some((a) => volatilityOf(a) === 's') ? 's' : 'i';
    return { k: 'func', name, args, ty: target, vol, strict: false, sig: 'COALESCE' };
  }
  const defs = FUNCS[name];
  const argSig = args0.map((a) => TYPE_NAME[a.ty]).join(', ');
  if (!defs) throw new SqlError(`function ${name}(${argSig}) is not in this lab's catalog`);
  for (const d of defs) {
    if (!d.variadic && d.args.length !== args0.length) continue;
    if (d.variadic && args0.length < d.args.length) continue;
    const args: Node[] = [];
    let ok = true;
    for (let i = 0; i < args0.length; i++) {
      const want = d.args[Math.min(i, d.args.length - 1)];
      const a = args0[i];
      if (want === 'any') {
        args.push(a.ty === 'unknown' ? { ...(a as Node), ty: 'text' } as Node : a);
      } else if (a.ty === want) args.push(a);
      else if (a.k === 'const' && a.ty === 'unknown') {
        try {
          args.push(coerceLiteral(String(a.v), want));
        } catch {
          ok = false;
        }
      } else ok = false;
      if (!ok) break;
    }
    if (!ok) continue;
    const sig = `${name}(${d.args.map((t) => (t === 'any' ? '"any"' : TYPE_NAME[t])).join(', ')}${d.variadic ? ', ...' : ''})`;
    return { k: 'func', name, args, ty: d.ret, vol: d.vol, strict: d.strict, sig };
  }
  throw new SqlError(`function ${name}(${argSig}) does not exist`);
}

export function volatilityOf(n: Node): Vol {
  let worst: Vol = 'i';
  walk(n, (m) => {
    const v = 'vol' in m ? m.vol : 'i';
    if (v === 'v') worst = 'v';
    else if (v === 's' && worst === 'i') worst = 's';
  });
  return worst;
}

export function walk(n: Node, fn: (m: Node) => void) {
  fn(n);
  switch (n.k) {
    case 'func':
    case 'and':
    case 'or':
      n.args.forEach((a) => walk(a, fn));
      break;
    case 'op':
      walk(n.l, fn);
      walk(n.r, fn);
      break;
    case 'cast':
    case 'not':
    case 'nulltest':
    case 'booltest':
      walk(n.arg, fn);
      break;
    case 'saop':
      walk(n.l, fn);
      n.elems.forEach((a) => walk(a, fn));
      break;
    default:
      break;
  }
}
/* ================================================================ parser */

export function parse(src: string, allowParams: boolean): Node {
  if (src.length > 400) throw new SqlError('expression too long for the lab (400 characters)');
  const toks = tokenize(src);
  let p = 0;
  let depth = 0;
  const peek = (o = 0) => toks[p + o];
  const isId = (v: string, o = 0) => peek(o)?.t === 'id' && peek(o)?.v === v;
  const isOp = (v: string, o = 0) => peek(o)?.t === 'op' && peek(o)?.v === v;
  const expectOp = (v: string) => {
    if (!isOp(v)) throw new SqlError(`syntax error: expected "${v}"${peek() ? ` at "${peek().v}"` : ' at end of input'}`);
    p++;
  };
  const guard = () => {
    if (++depth > 60) throw new SqlError('expression nests too deeply for the lab');
  };

  function orExpr(): Node {
    guard();
    let n = andExpr();
    while (isId('or')) {
      p++;
      n = { k: 'or', args: [boolArg(n), boolArg(andExpr())], ty: 'bool' };
    }
    depth--;
    return n;
  }
  function andExpr(): Node {
    let n = notExpr();
    while (isId('and')) {
      p++;
      n = { k: 'and', args: [boolArg(n), boolArg(notExpr())], ty: 'bool' };
    }
    return n;
  }
  function notExpr(): Node {
    if (isId('not')) {
      p++;
      return { k: 'not', arg: boolArg(notExpr()), ty: 'bool' };
    }
    return isExpr();
  }
  function isExpr(): Node {
    let n = cmpExpr();
    while (isId('is')) {
      p++;
      let neg = false;
      if (isId('not')) {
        neg = true;
        p++;
      }
      if (isId('null')) {
        p++;
        n = { k: 'nulltest', arg: n, isNull: !neg, ty: 'bool' };
      } else if (isId('true') || isId('false')) {
        const v = peek().v.toUpperCase();
        p++;
        n = { k: 'booltest', arg: boolArg(n), test: `IS ${neg ? 'NOT ' : ''}${v}` as 'IS TRUE', ty: 'bool' };
      } else throw new SqlError('syntax error after IS (the lab supports IS [NOT] NULL / TRUE / FALSE)');
    }
    return n;
  }
  function cmpExpr(): Node {
    const l = predExpr();
    const t = peek();
    if (t && t.t === 'op' && ['=', '<', '>', '<=', '>=', '<>', '!='].includes(t.v)) {
      p++;
      return makeCompare(t.v, l, predExpr());
    }
    return l;
  }
  function predExpr(): Node {
    const l = otherOpExpr();
    let neg = false;
    if (isId('not') && (isId('in', 1) || isId('between', 1) || isId('like', 1) || isId('ilike', 1))) {
      neg = true;
      p++;
    }
    if (isId('between')) {
      p++;
      const lo = otherOpExpr();
      if (!isId('and')) throw new SqlError('syntax error: BETWEEN needs AND');
      p++;
      const hi = otherOpExpr();
      // The parser rewrites BETWEEN into two comparisons joined by AND (or NOT BETWEEN into OR).
      return neg
        ? { k: 'or', args: [makeCompare('<', l, lo), makeCompare('>', l, hi)], ty: 'bool' }
        : { k: 'and', args: [makeCompare('>=', l, lo), makeCompare('<=', l, hi)], ty: 'bool' };
    }
    if (isId('in')) {
      p++;
      expectOp('(');
      if (isId('select')) throw new SqlError('subqueries are not modelled (and are not allowed in an index predicate)');
      const elems: Node[] = [otherOpExpr()];
      while (isOp(',')) {
        p++;
        elems.push(otherOpExpr());
      }
      expectOp(')');
      if (elems.length === 1) return makeCompare(neg ? '<>' : '=', l, elems[0]);
      const typed = elems.map((e) => (makeCompare('=', l, e) as Extract<Node, { k: 'op' }>).r);
      const probe = makeCompare('=', l, elems[0]) as Extract<Node, { k: 'op' }>;
      return { k: 'saop', op: neg ? '<>' : '=', useOr: !neg, l: probe.l, elems: typed, ty: 'bool', coll: probe.coll, vol: probe.vol };
    }
    if (isId('like') || isId('ilike')) {
      const ci = peek().v === 'ilike';
      p++;
      return makeBinary(`${neg ? '!' : ''}~~${ci ? '*' : ''}`, l, otherOpExpr());
    }
    if (neg) throw new SqlError('syntax error after NOT');
    return l;
  }
  function otherOpExpr(): Node {
    let n = addExpr();
    while (peek()?.t === 'op' && ['||', '->>', '->', '#>>', '@>'].includes(peek().v)) {
      const op = peek().v;
      p++;
      n = makeBinary(op, n, addExpr());
    }
    return n;
  }
  function addExpr(): Node {
    let n = mulExpr();
    while (isOp('+') || isOp('-')) {
      const op = peek().v;
      p++;
      n = makeBinary(op, n, mulExpr());
    }
    return n;
  }
  function mulExpr(): Node {
    let n = atExpr();
    while (isOp('*') || isOp('/')) {
      const op = peek().v;
      p++;
      n = makeBinary(op, n, atExpr());
    }
    return n;
  }
  function atExpr(): Node {
    let n = collateExpr();
    while (isId('at') && isId('time', 1) && isId('zone', 2)) {
      p += 3;
      const zone = collateExpr();
      n = makeFunc('timezone', [zone, n]);
    }
    return n;
  }
  function collateExpr(): Node {
    let n = unary();
    while (isId('collate')) {
      p++;
      const t = peek();
      if (!t || (t.t !== 'qid' && t.t !== 'id')) throw new SqlError('syntax error: COLLATE needs a collation name');
      p++;
      if (!isTextish(n.ty) && n.ty !== 'unknown') throw new SqlError(`collations are not supported by type ${TYPE_NAME[n.ty]}`);
      n = { ...n, ty: n.ty === 'unknown' ? 'text' : n.ty, ecoll: t.v } as Node;
    }
    return n;
  }
  function unary(): Node {
    if (isOp('-')) {
      p++;
      const n = unary();
      if (n.k === 'const' && typeof n.v === 'number') return { ...n, v: -n.v, raw: `-${n.raw}` };
      return makeBinary('-', { k: 'const', ty: 'int4', v: 0, raw: '0' }, n);
    }
    return postfix();
  }
  function postfix(): Node {
    let n = primary();
    while (isOp('::')) {
      p++;
      const t = peek();
      if (!t || t.t !== 'id') throw new SqlError('syntax error: expected a type name after ::');
      p++;
      n = makeCast(n, typeFromName(t.v));
    }
    return n;
  }
  function primary(): Node {
    const t = peek();
    if (!t) throw new SqlError('syntax error at end of input');
    if (t.t === 'num') {
      p++;
      if (t.v.includes('.')) return { k: 'const', ty: 'numeric', v: Number(t.v), raw: t.v };
      const v = Number(t.v);
      return { k: 'const', ty: Math.abs(v) > 2147483647 ? 'int8' : 'int4', v, raw: t.v };
    }
    if (t.t === 'str') {
      p++;
      return { k: 'const', ty: 'unknown', v: t.v, raw: t.v };
    }
    if (t.t === 'param') {
      p++;
      if (!allowParams) throw new SqlError(`there is no parameter $${t.v}`);
      return { k: 'param', n: Number(t.v), ty: 'unknown' };
    }
    if (t.t === 'op' && t.v === '(') {
      p++;
      const n = orExpr();
      expectOp(')');
      return n;
    }
    if (t.t === 'id') {
      if (t.v === 'select') throw new SqlError('subqueries are not modelled (and are not allowed in an index predicate)');
      if (t.v === 'true' || t.v === 'false') {
        p++;
        return { k: 'const', ty: 'bool', v: t.v === 'true', raw: t.v };
      }
      if (t.v === 'null') {
        p++;
        return { k: 'const', ty: 'unknown', v: null, raw: 'NULL' };
      }
      if (['interval', 'date', 'timestamptz', 'timestamp'].includes(t.v) && peek(1)?.t === 'str') {
        p += 2;
        return coerceLiteral(peek(-1).v, typeFromName(t.v));
      }
      if (t.v === 'current_timestamp') {
        p++;
        return makeFunc('now', []);
      }
      if (t.v === 'cast' && isOp('(', 1)) {
        p += 2;
        const inner = orExpr();
        if (!isId('as')) throw new SqlError('syntax error: CAST needs AS');
        p++;
        const ty = typeFromName(peek().v);
        p++;
        expectOp(')');
        return makeCast(inner, ty);
      }
      if (t.v === 'trim' && isOp('(', 1)) {
        p += 2;
        if (isId('both')) {
          p++;
          if (isId('from')) p++;
        }
        const inner = orExpr();
        expectOp(')');
        return makeFunc('btrim', [inner]);
      }
      if (isOp('(', 1)) {
        p += 2;
        const args: Node[] = [];
        if (!isOp(')')) {
          args.push(orExpr());
          while (isOp(',')) {
            p++;
            args.push(orExpr());
          }
        }
        expectOp(')');
        return makeFunc(t.v, args);
      }
      const ty = SCHEMA[t.v];
      if (!ty) throw new SqlError(`column "${t.v}" does not exist`);
      p++;
      return { k: 'col', name: t.v, ty };
    }
    throw new SqlError(`syntax error at or near "${t.v}"`);
  }
  function boolArg(n: Node): Node {
    if (n.ty !== 'bool') throw new SqlError(`argument of AND/OR/NOT must be type boolean, not type ${TYPE_NAME[n.ty]}`);
    return n;
  }

  const n = orExpr();
  if (p < toks.length) throw new SqlError(`syntax error at or near "${peek().v}"`);
  return n;
}

/** Splits "a, lower(b)" at top-level commas. */
export function splitTopLevel(src: string): string[] {
  const out: string[] = [];
  let d = 0;
  let q = false;
  let cur = '';
  for (const c of src) {
    if (c === "'") q = !q;
    if (!q && c === '(') d++;
    if (!q && c === ')') d--;
    if (!q && d === 0 && c === ',') {
      out.push(cur);
      cur = '';
    } else cur += c;
  }
  if (cur.trim()) out.push(cur);
  return out.map((s) => s.trim()).filter(Boolean);
}

/* ============================================================ deparse / equal */

const fmtConst = (n: Extract<Node, { k: 'const' }>) => {
  if (n.v === null) return 'NULL';
  if (n.ty === 'bool') return String(n.v);
  if (isNum(n.ty)) return n.raw;
  if (n.ty === 'interval') return `'${n.raw}'::interval`;
  if (n.ty === 'date') return `'${n.raw}'::date`;
  return `'${String(n.raw).replace(/'/g, "''")}'`;
};

export function deparse(n: Node, top = true): string {
  const wrap = (s: string) => (top ? s : `(${s})`);
  const coll = n.ecoll ? ` COLLATE "${n.ecoll}"` : '';
  switch (n.k) {
    case 'col':
      return n.name + coll;
    case 'const':
      return fmtConst(n) + coll;
    case 'param':
      return `$${n.n}`;
    case 'func':
      if (n.name === 'timezone') return `(${deparse(n.args[1], false)} AT TIME ZONE ${deparse(n.args[0], false)})${coll}`;
      if (n.name === 'coalesce') return `COALESCE(${n.args.map((a) => deparse(a)).join(', ')})`;
      return `${n.name}(${n.args.map((a) => deparse(a)).join(', ')})${coll}`;
    case 'op': {
      const sym = n.op === '~~' ? 'LIKE' : n.op === '~~*' ? 'ILIKE' : n.op === '!~~' ? 'NOT LIKE' : n.op === '!~~*' ? 'NOT ILIKE' : n.op;
      return wrap(`${deparse(n.l, false)} ${sym} ${deparse(n.r, false)}`) + coll;
    }
    case 'cast': {
      const name = TYPE_NAME[n.ty] === 'timestamp with time zone' ? 'timestamptz' : TYPE_NAME[n.ty] === 'timestamp without time zone' ? 'timestamp' : TYPE_NAME[n.ty];
      const inner = deparse(n.arg, false);
      return `${inner}::${name}`;
    }
    case 'and':
      return wrap(n.args.map((a) => deparse(a, false)).join(' AND '));
    case 'or':
      return wrap(n.args.map((a) => deparse(a, false)).join(' OR '));
    case 'not':
      return wrap(`NOT ${deparse(n.arg, false)}`);
    case 'nulltest':
      return wrap(`${deparse(n.arg, false)} IS ${n.isNull ? '' : 'NOT '}NULL`);
    case 'booltest':
      return wrap(`${deparse(n.arg, false)} ${n.test}`);
    case 'saop':
      return wrap(`${deparse(n.l, false)} ${n.op} ${n.useOr ? 'ANY' : 'ALL'} ('{${n.elems.map((e) => (e.k === 'const' ? String(e.raw) : deparse(e))).join(',')}}')`);
  }
}

/** Structural equality, the analogue of PostgreSQL's equal() on two expression trees. */
export function equal(a: Node, b: Node): boolean {
  if (a.k !== b.k || a.ty !== b.ty || (a.ecoll ?? '') !== (b.ecoll ?? '')) return false;
  switch (a.k) {
    case 'col':
      return a.name === (b as typeof a).name;
    case 'const':
      return a.v === (b as typeof a).v;
    case 'param':
      return a.n === (b as typeof a).n;
    case 'func': {
      const bb = b as typeof a;
      return a.name === bb.name && a.args.length === bb.args.length && a.args.every((x, i) => equal(x, bb.args[i]));
    }
    case 'op': {
      const bb = b as typeof a;
      return a.op === bb.op && a.coll === bb.coll && equal(a.l, bb.l) && equal(a.r, bb.r);
    }
    case 'cast':
      return equal(a.arg, (b as typeof a).arg);
    case 'and':
    case 'or': {
      const bb = b as typeof a;
      return a.args.length === bb.args.length && a.args.every((x, i) => equal(x, bb.args[i]));
    }
    case 'not':
      return equal(a.arg, (b as typeof a).arg);
    case 'nulltest':
      return a.isNull === (b as typeof a).isNull && equal(a.arg, (b as typeof a).arg);
    case 'booltest':
      return a.test === (b as typeof a).test && equal(a.arg, (b as typeof a).arg);
    case 'saop': {
      const bb = b as typeof a;
      return a.op === bb.op && a.useOr === bb.useOr && equal(a.l, bb.l) && a.elems.length === bb.elems.length && a.elems.every((x, i) => equal(x, bb.elems[i]));
    }
  }
}

/** Why two expressions are not equal(): the first difference, in words. */
export function diffReason(q: Node, key: Node): string {
  if (q.k !== key.k) {
    const what = (n: Node) =>
      n.k === 'col' ? `the bare column ${n.name}` : n.k === 'func' && n.name === 'timezone' ? 'an AT TIME ZONE conversion' : n.k === 'func' ? `a call to ${n.name}()` : n.k === 'cast' ? `a cast to ${TYPE_NAME[n.ty]}` : n.k === 'op' ? `an ${n.op} expression` : n.k;
    return `the query has ${what(q)} where the key has ${what(key)}`;
  }
  if ((q.ecoll ?? '') !== (key.ecoll ?? '')) return `collations differ`;
  switch (q.k) {
    case 'col':
      return q.name === (key as typeof q).name ? 'same column' : `column ${q.name} is not ${(key as typeof q).name}`;
    case 'const':
      return `constant ${deparse(q)} is not ${deparse(key)}`;
    case 'func': {
      const kk = key as typeof q;
      if (q.name !== kk.name) return `${q.name}() is not ${kk.name}()`;
      for (let i = 0; i < Math.min(q.args.length, kk.args.length); i++) if (!equal(q.args[i], kk.args[i])) return diffReason(q.args[i], kk.args[i]);
      return 'different number of arguments';
    }
    case 'op': {
      const kk = key as typeof q;
      if (q.op !== kk.op) return `operator ${q.op} is not ${kk.op}`;
      if (equal(q.l, kk.r) && equal(q.r, kk.l)) return 'the operands are in the other order, and equal() does not commute them';
      if (!equal(q.l, kk.l)) return diffReason(q.l, kk.l);
      return diffReason(q.r, kk.r);
    }
    case 'cast':
      return q.ty !== key.ty ? `cast to ${TYPE_NAME[q.ty]} is not cast to ${TYPE_NAME[key.ty]}` : diffReason(q.arg, (key as typeof q).arg);
    default:
      return 'the expressions differ';
  }
}
/* ============================================================ normalize */

const NEGATOR: Record<string, string> = { '=': '<>', '<>': '=', '<': '>=', '>=': '<', '>': '<=', '<=': '>', '~~': '!~~', '!~~': '~~', '~~*': '!~~*', '!~~*': '~~*' };
const COMMUTATOR: Record<string, string> = { '=': '=', '<>': '<>', '<': '>', '>': '<', '<=': '>=', '>=': '<=' };

function negate(n: Node): Node {
  switch (n.k) {
    case 'const':
      return typeof n.v === 'boolean' ? { ...n, v: !n.v, raw: String(!n.v) } : { k: 'not', arg: n, ty: 'bool' };
    case 'op':
      return NEGATOR[n.op] ? { ...n, op: NEGATOR[n.op] } : { k: 'not', arg: n, ty: 'bool' };
    case 'saop':
      return { ...n, op: NEGATOR[n.op] ?? n.op, useOr: !n.useOr };
    case 'and':
      return { k: 'or', args: n.args.map(negate), ty: 'bool' };
    case 'or':
      return { k: 'and', args: n.args.map(negate), ty: 'bool' };
    case 'not':
      return n.arg;
    case 'nulltest':
      return { ...n, isNull: !n.isNull };
    case 'booltest': {
      const flip: Record<string, 'IS TRUE' | 'IS NOT TRUE' | 'IS FALSE' | 'IS NOT FALSE'> = { 'IS TRUE': 'IS NOT TRUE', 'IS NOT TRUE': 'IS TRUE', 'IS FALSE': 'IS NOT FALSE', 'IS NOT FALSE': 'IS FALSE' };
      return { ...n, test: flip[n.test] };
    }
    default:
      return { k: 'not', arg: n, ty: 'bool' };
  }
}

function evalConst(n: Node): Node | null {
  const cs = (xs: Node[]) => xs.every((x) => x.k === 'const' && x.v !== null);
  if (n.k === 'func' && n.vol === 'i' && cs(n.args)) {
    const a = n.args.map((x) => (x as Extract<Node, { k: 'const' }>).v);
    const s = String(a[0]);
    const text = (v: string): Node => ({ k: 'const', ty: 'text', v, raw: v });
    if (n.name === 'lower') return text(s.toLowerCase());
    if (n.name === 'upper') return text(s.toUpperCase());
    if (n.name === 'btrim') return text(s.trim());
    if (n.name === 'length') return { k: 'const', ty: 'int4', v: s.length, raw: String(s.length) };
  }
  if (n.k === 'op' && n.vol === 'i' && cs([n.l, n.r])) {
    const l = (n.l as Extract<Node, { k: 'const' }>).v;
    const r = (n.r as Extract<Node, { k: 'const' }>).v;
    if (typeof l === 'number' && typeof r === 'number' && ['+', '-', '*', '/'].includes(n.op) && isNum(n.ty)) {
      let v = n.op === '+' ? l + r : n.op === '-' ? l - r : n.op === '*' ? l * r : l / r;
      if (n.ty !== 'numeric') v = Math.trunc(v);
      return { k: 'const', ty: n.ty, v, raw: String(v) };
    }
    if (n.op === '||' && typeof l === 'string' && typeof r === 'string') return { k: 'const', ty: 'text', v: l + r, raw: l + r };
  }
  if (n.k === 'cast' && n.vol === 'i' && n.arg.k === 'const' && n.arg.v !== null) {
    if (isNum(n.ty) && typeof n.arg.v === 'number') {
      const v = n.ty === 'numeric' ? n.arg.v : Math.round(n.arg.v);
      return { k: 'const', ty: n.ty, v, raw: String(v) };
    }
  }
  return null;
}

/**
 * The part of eval_const_expressions() that matters for proofs: fold immutable calls on constants, simplify
 * boolean equality (x = true -> x, x = false -> NOT x), push NOT down through negator operators, De Morgan and
 * IS [NOT] NULL, and flatten nested AND/OR. Stable functions such as now() are NOT folded.
 */
export function normalize(n: Node): Node {
  switch (n.k) {
    case 'func': {
      const m = { ...n, args: n.args.map(normalize) };
      return evalConst(m) ?? m;
    }
    case 'op': {
      const m = { ...n, l: normalize(n.l), r: normalize(n.r) };
      if ((m.op === '=' || m.op === '<>') && m.l.ty === 'bool' && m.r.k === 'const' && typeof m.r.v === 'boolean') {
        const positive = (m.op === '=') === m.r.v;
        return positive ? m.l : normalize({ k: 'not', arg: m.l, ty: 'bool' });
      }
      return evalConst(m) ?? m;
    }
    case 'cast': {
      const m = { ...n, arg: normalize(n.arg) };
      return evalConst(m) ?? m;
    }
    case 'saop':
      return { ...n, l: normalize(n.l), elems: n.elems.map(normalize) };
    case 'not': {
      const inner = normalize(n.arg);
      const neg = negate(inner);
      return neg.k === 'not' ? neg : normalize(neg);
    }
    case 'nulltest':
    case 'booltest':
      return { ...n, arg: normalize(n.arg) };
    case 'and':
    case 'or': {
      const flat: Node[] = [];
      for (const a of n.args.map(normalize)) {
        if (a.k === n.k) flat.push(...a.args);
        else flat.push(a);
      }
      return flat.length === 1 ? flat[0] : { ...n, args: flat };
    }
    default:
      return n;
  }
}

/* ============================================================ predtest */

export type Step = { depth: number; text: string; ok: boolean | null; rule?: string };

const STRAT: Record<string, number> = { '<': 1, '<=': 2, '=': 3, '>=': 4, '>': 5, '<>': 6 };
const STRAT_OP = ['', '<', '<=', '=', '>=', '>', '<>'];
/** predtest.c BT_implic_table[clause_op-1][pred_op-1]: the test "pred_const test_op clause_const". 0 = cannot decide. */
export const BT_IMPLIC: number[][] = [
  [4, 4, 0, 0, 0, 4],
  [5, 4, 0, 0, 0, 5],
  [5, 4, 3, 2, 1, 6],
  [0, 0, 0, 2, 1, 1],
  [0, 0, 0, 2, 2, 2],
  [0, 0, 0, 0, 0, 3],
];
/** predtest.c BT_implies_table: same two subexpressions on both sides. */
const BT_IMPLIES: boolean[][] = [
  [true, true, false, false, false, true],
  [false, true, false, false, false, false],
  [false, true, true, true, false, false],
  [false, false, false, true, false, false],
  [false, false, false, true, true, true],
  [false, false, false, false, false, true],
];
export const MAX_SAOP_ARRAY_SIZE = 100;

type PredClass = { cls: 'ATOM' } | { cls: 'AND' | 'OR'; items: Node[]; how: string };

function classify(n: Node): PredClass {
  if (n.k === 'and') return { cls: 'AND', items: n.args, how: 'AND' };
  if (n.k === 'or') return { cls: 'OR', items: n.args, how: 'OR' };
  if (n.k === 'saop' && n.elems.every((e) => e.k === 'const') && n.elems.length <= MAX_SAOP_ARRAY_SIZE) {
    const items = n.elems.map((e) => ({ k: 'op', op: n.op, l: n.l, r: e, ty: 'bool', vol: n.vol, strict: true, sig: '', coll: n.coll }) as Node);
    return { cls: n.useOr ? 'OR' : 'AND', items, how: `${n.op} ${n.useOr ? 'ANY' : 'ALL'} over ${n.elems.length} constants` };
  }
  return { cls: 'ATOM' };
}

const cmpValues = (a: Extract<Node, { k: 'const' }>, b: Extract<Node, { k: 'const' }>) => {
  const x = a.v as number | string | boolean;
  const y = b.v as number | string | boolean;
  return x < y ? -1 : x > y ? 1 : 0;
};

function testHolds(strategy: number, predConst: Extract<Node, { k: 'const' }>, clauseConst: Extract<Node, { k: 'const' }>) {
  const c = cmpValues(predConst, clauseConst);
  switch (strategy) {
    case 1:
      return c < 0;
    case 2:
      return c <= 0;
    case 3:
      return c === 0;
    case 4:
      return c >= 0;
    case 5:
      return c > 0;
    default:
      return c !== 0;
  }
}

const q = (n: Node) => deparse(n);

/** clause_is_strict_for(): would the clause be NULL (or, at top level, false) whenever subexpr is NULL? */
export function strictFor(clause: Node, sub: Node, allowFalse: boolean): boolean {
  if (equal(clause, sub)) return true;
  if (clause.k === 'op') return clause.strict && [clause.l, clause.r].some((a) => strictFor(a, sub, false));
  if (clause.k === 'func') return clause.strict && clause.args.some((a) => strictFor(a, sub, false));
  if (clause.k === 'cast') return strictFor(clause.arg, sub, false);
  if (clause.k === 'saop') return strictFor(clause.l, sub, false) && ((allowFalse && clause.useOr) || clause.elems.length > 0);
  return false;
}

export class Prover {
  steps: Step[] = [];
  calls = 0;
  truncated = false;
  exhausted = false;
  constructor(private record: boolean) {}

  private log(depth: number, text: string, ok: boolean | null, rule?: string) {
    if (!this.record) return -1;
    if (this.steps.length >= 160) {
      this.truncated = true;
      return -1;
    }
    this.steps.push({ depth, text, ok, rule });
    return this.steps.length - 1;
  }
  private settle(i: number, ok: boolean) {
    if (i >= 0) this.steps[i].ok = ok;
  }

  /** predicate_implied_by_recurse() */
  implies(clause: Node, pred: Node, depth = 0): boolean {
    if (++this.calls > 25000) {
      // The lab's safety limit, not PostgreSQL's: predtest.c has no step cap (only the 100-element IN list limit).
      this.exhausted = true;
      return false;
    }
    const c = classify(clause);
    const p = classify(pred);
    const label = (n: Node, k: PredClass) => (k.cls === 'ATOM' ? `\`${q(n)}\`` : k.cls === 'AND' ? `AND of ${k.items.length}` : `OR of ${k.items.length}`);
    const head = `${label(clause, c)} ⇒ ${label(pred, p)}?`;
    const any = (xs: Node[], f: (x: Node) => boolean) => xs.some(f);
    const all = (xs: Node[], f: (x: Node) => boolean) => xs.every(f);

    if (c.cls === 'ATOM' && p.cls === 'ATOM') return this.simple(clause, pred, depth);

    let rule = '';
    let res = false;
    let i = -1;
    if (p.cls === 'AND' && c.cls !== 'OR') {
      rule = `${c.cls === 'AND' ? 'AND' : 'atom'} ⇒ AND: must prove every predicate item${p.how !== 'AND' ? ` (${p.how})` : ''}`;
      i = this.log(depth, head, null, rule);
      res = all(p.items, (x) => this.implies(clause, x, depth + 1));
    } else if (c.cls === 'AND' && p.cls === 'OR') {
      rule = `AND ⇒ OR: the whole clause proves any predicate item, or any clause item proves the whole OR${p.how !== 'OR' ? ` (${p.how})` : ''}`;
      i = this.log(depth, head, null, rule);
      res = any(p.items, (x) => this.implies(clause, x, depth + 1)) || any(c.items, (x) => this.implies(x, pred, depth + 1));
    } else if (c.cls === 'AND' && p.cls === 'ATOM') {
      rule = 'AND ⇒ atom: any one query clause may prove it';
      i = this.log(depth, head, null, rule);
      res = any(c.items, (x) => this.implies(x, pred, depth + 1));
    } else if (c.cls === 'OR' && p.cls === 'OR') {
      rule = `OR ⇒ OR: every clause arm must prove some predicate arm${c.how !== 'OR' ? ` (clause is ${c.how})` : ''}`;
      i = this.log(depth, head, null, rule);
      res = all(c.items, (ci) => any(p.items, (pi) => this.implies(ci, pi, depth + 1)));
    } else if (c.cls === 'OR') {
      rule = `OR ⇒ ${p.cls === 'AND' ? 'AND' : 'atom'}: every arm of the OR must prove it${c.how !== 'OR' ? ` (${c.how})` : ''}`;
      i = this.log(depth, head, null, rule);
      res = all(c.items, (x) => this.implies(x, pred, depth + 1));
    } else if (p.cls === 'OR') {
      // atom => OR
      rule = `atom ⇒ OR: proving any one arm is enough${p.how !== 'OR' ? ` (${p.how})` : ''}`;
      i = this.log(depth, head, null, rule);
      res = any(p.items, (x) => this.implies(clause, x, depth + 1));
    }
    this.settle(i, res);
    return res;
  }

  /** predicate_implied_by_simple_clause() */
  private simple(clause: Node, pred: Node, depth: number): boolean {
    const head = `\`${q(clause)}\` ⇒ \`${q(pred)}\`?`;
    if (equal(pred, clause)) {
      this.log(depth, head, true, 'equal(): the clause is the predicate');
      return true;
    }
    if (pred.k === 'nulltest' && !pred.isNull) {
      const what = pred.arg.k === 'col' ? pred.arg.name : q(pred.arg);
      if (strictFor(clause, pred.arg, true)) {
        this.log(depth, head, true, `${what} IS NOT NULL: the clause is strict, so it cannot be true when that is NULL`);
        return true;
      }
      this.log(depth, head, false, `not equal(), and the clause is not strict in ${what}, the only other way to prove IS NOT NULL`);
      return false;
    }
    const r = this.operatorProof(clause, pred);
    this.log(depth, head, r.ok, r.why);
    return r.ok;
  }

  /** operator_predicate_proof() */
  private operatorProof(clause: Node, pred: Node): { ok: boolean; why: string } {
    const fail = (why: string) => ({ ok: false, why });
    if (pred.k !== 'op' || clause.k !== 'op') {
      const kind = (n: Node) => (n.k === 'nulltest' ? `an IS ${n.isNull ? '' : 'NOT '}NULL test` : n.k === 'booltest' ? `an ${n.test} test` : n.k === 'not' ? 'a NOT' : n.k === 'col' ? 'a bare boolean column' : n.k === 'saop' ? 'an array comparison too large to expand' : 'not an operator');
      if (pred.k !== 'op') return fail(`not equal(), and the predicate is ${kind(pred)} — nothing else can prove it`);
      return fail(`not equal(), and the clause is ${kind(clause)}, not a binary operator`);
    }
    if (STRAT[pred.op] === undefined || STRAT[clause.op] === undefined) return fail(`${pred.op === clause.op ? pred.op : `${clause.op} / ${pred.op}`} is not a btree comparison operator, and the expressions are not equal()`);
    if ((pred.coll ?? '') !== (clause.coll ?? '')) {
      const shared = equal(pred.l, clause.l) || equal(pred.r, clause.r) || equal(pred.l, clause.r) || equal(pred.r, clause.l);
      return fail(shared ? `the two comparisons use different collations (${clause.coll ?? 'none'} vs ${pred.coll ?? 'none'}): no proof` : `no operand of the clause is equal() to an operand of the predicate (${q(clause.l)} vs ${q(pred.l)})`);
    }
    let predOp = pred.op;
    let clauseOp = clause.op;
    let pc: Node;
    let cc: Node;
    if (equal(pred.l, clause.l)) {
      if (equal(pred.r, clause.r)) {
        const ok = BT_IMPLIES[STRAT[clauseOp] - 1][STRAT[predOp] - 1] && clause.vol === 'i';
        return { ok, why: ok ? `same operands: ${clauseOp} implies ${predOp}` : `same operands, but ${clauseOp} does not imply ${predOp}` };
      }
      pc = pred.r;
      cc = clause.r;
    } else if (equal(pred.r, clause.r)) {
      pc = pred.l;
      cc = clause.l;
      predOp = COMMUTATOR[predOp];
      clauseOp = COMMUTATOR[clauseOp];
    } else if (equal(pred.l, clause.r)) {
      if (equal(pred.r, clause.l)) {
        const ok = BT_IMPLIES[STRAT[clauseOp] - 1][STRAT[COMMUTATOR[predOp]] - 1] && clause.vol === 'i';
        return { ok, why: ok ? `same operands in the other order: ${clauseOp} implies ${predOp}` : `same operands, but ${clauseOp} does not imply ${predOp}` };
      }
      pc = pred.r;
      cc = clause.l;
      clauseOp = COMMUTATOR[clauseOp];
    } else if (equal(pred.r, clause.l)) {
      pc = pred.l;
      cc = clause.r;
      predOp = COMMUTATOR[predOp];
    } else {
      return fail(`no operand of the clause is equal() to an operand of the predicate (${q(clause.l)} vs ${q(pred.l)})`);
    }
    if (cc.k !== 'const') {
      const what = cc.k === 'param' ? `$${cc.n} is a Param, not a Const (a generic plan has no value to compare)` : `${q(cc)} is not a Const (${volatilityOf(cc) === 'i' ? 'not folded' : 'contains a non-immutable function, so it is evaluated at run time'})`;
      return fail(what);
    }
    if (pc.k !== 'const') return fail(`${q(pc)} in the predicate is not a Const`);
    if (cc.v === null) return { ok: true, why: 'the clause compares with NULL, so it is never true: implication is vacuous' };
    const strategy = BT_IMPLIC[STRAT[clauseOp] - 1][STRAT[predOp] - 1];
    if (!strategy) return fail(`BT_implic_table has no test for clause ${clauseOp} against predicate ${predOp}`);
    const dt = ['timestamptz', 'timestamp', 'date'];
    if (pc.ty !== cc.ty && dt.includes(pc.ty) && dt.includes(cc.ty) && (pc.ty === 'timestamptz' || cc.ty === 'timestamptz')) {
      return fail(`the test operator ${TYPE_NAME[pc.ty]} ${STRAT_OP[strategy]} ${TYPE_NAME[cc.ty]} is only STABLE: no deduction`);
    }
    const ok = testHolds(strategy, pc, cc);
    return { ok, why: `btree test ${fmtConst(pc)} ${STRAT_OP[strategy]} ${fmtConst(cc)} is ${ok ? 'true' : 'false'}` };
  }
}

export const conjuncts = (n: Node | null): Node[] => (!n ? [] : n.k === 'and' ? n.args : [n]);
const listNode = (xs: Node[]): Node => (xs.length === 1 ? xs[0] : { k: 'and', args: xs, ty: 'bool' });

export function containsMutable(n: Node) {
  return volatilityOf(n) !== 'i';
}
/* ============================================================ index analysis */

export type ClauseRole = 'cond' | 'filter' | 'implied' | 'unused';
export type ClauseInfo = { node: Node; text: string; role: ClauseRole; key: number; why: string; cond?: string };
export type PredAtom = { node: Node; text: string; proved: boolean; by: number[] };
export type Stage = 'index-error' | 'query-error' | 'create-error' | 'not-implied' | 'usable' | 'full-scan' | 'no-key-match';
export type Analysis = {
  stage: Stage;
  error: string;
  problems: { text: string; sig: string; vol: Vol; where: 'expression' | 'predicate'; key: number }[];
  keys: Node[];
  keyTexts: string[];
  pred: Node | null;
  clauses: ClauseInfo[];
  predAtoms: PredAtom[];
  predOK: boolean | null;
  steps: Step[];
  truncated: boolean;
  exhausted: boolean;
  explain: string[];
  statement: string;
};

const VOL_WORD: Record<Vol, string> = { i: 'IMMUTABLE', s: 'STABLE', v: 'VOLATILE' };
export const volWord = (v: Vol) => VOL_WORD[v];

function nonImmutableNodes(n: Node, where: 'expression' | 'predicate', key: number) {
  const out: Analysis['problems'] = [];
  walk(n, (m) => {
    if ((m.k === 'func' || m.k === 'op' || m.k === 'cast') && m.vol !== 'i') {
      out.push({ text: deparse(m), sig: m.sig, vol: m.vol, where, key });
    }
  });
  // Report the innermost offender first: now() rather than the now() - interval around it.
  return out.reverse();
}

const hasColumns = (n: Node) => {
  let found = false;
  walk(n, (m) => {
    if (m.k === 'col') found = true;
  });
  return found;
};
const columnsOf = (n: Node) => {
  const s = new Set<string>();
  walk(n, (m) => {
    if (m.k === 'col') s.add(m.name);
  });
  return s;
};
const stripColl = (n: Node): Node => (n.ecoll ? ({ ...n, ecoll: undefined } as Node) : n);

function keyDisplay(n: Node) {
  const t = deparse(n);
  return n.k === 'col' || (n.k === 'func' && n.name !== 'timezone' && n.name !== 'coalesce') ? t : `(${t})`;
}

/** match_clause_to_indexcol(), reduced to btree: which key, if any, can this clause search on? */
export type KeyMatch = { ok: true; key: number; cond: string } | { ok: false; why: string };
export function matchClause(clause: Node, keys: Node[]): KeyMatch {
  const pseudoConst = (n: Node) => !hasColumns(n) && volatilityOf(n) !== 'v';
  let collationMiss = '';
  for (let i = 0; i < keys.length; i++) {
    const key = keys[i];
    const keyColl = collOf(key);
    if (clause.k === 'op' && STRAT[clause.op] !== undefined && clause.op !== '<>') {
      const sides: [Node, Node, string][] = [
        [clause.l, clause.r, clause.op],
        [clause.r, clause.l, COMMUTATOR[clause.op]],
      ];
      for (const [operand, other, op] of sides) {
        if (!equal(stripColl(operand), key) || !pseudoConst(other)) continue;
        if (keyColl !== null && clause.coll !== keyColl) {
          collationMiss = `the comparison uses collation "${clause.coll}" but the index key has collation "${keyColl}"`;
          continue;
        }
        return { ok: true, key: i, cond: `${deparse(stripColl(operand), false)} ${op} ${deparse(other, false)}` };
      }
    }
    if (clause.k === 'saop' && clause.useOr && clause.op === '=' && equal(stripColl(clause.l), key)) {
      if (keyColl !== null && clause.coll !== keyColl) {
        collationMiss = `the comparison uses collation "${clause.coll}" but the index key has collation "${keyColl}"`;
        continue;
      }
      return { ok: true, key: i, cond: deparse(clause) };
    }
    if (clause.k === 'nulltest' && equal(clause.arg, key)) return { ok: true, key: i, cond: deparse(clause) };
    if (key.ty === 'bool' && equal(clause, key)) return { ok: true, key: i, cond: `${deparse(key, false)} = true` };
    if (key.ty === 'bool' && clause.k === 'not' && equal(clause.arg, key)) return { ok: true, key: i, cond: `${deparse(key, false)} = false` };
  }
  if (collationMiss) return { ok: false, why: collationMiss };
  if (clause.k === 'op' && clause.op === '<>') return { ok: false, why: '<> is not a btree search operator' };
  if (clause.k === 'op' && clause.op.includes('~~')) return { ok: false, why: 'the lab does not model LIKE as an index search condition' };
  if (clause.k === 'op' && STRAT[clause.op] === undefined) return { ok: false, why: `${clause.op} is not a btree comparison` };
  // Explain the nearest miss: a key that shares a column with the clause's non-constant operand.
  const operand = clause.k === 'op' ? (hasColumns(clause.l) ? clause.l : clause.r) : clause.k === 'saop' ? clause.l : clause.k === 'nulltest' || clause.k === 'not' || clause.k === 'booltest' ? clause.arg : clause;
  if (clause.k === 'op' && hasColumns(clause.l) && hasColumns(clause.r)) return { ok: false, why: 'both sides reference the table, so neither is a constant to search for' };
  if (clause.k === 'op' && !pseudoConst(clause.k === 'op' && hasColumns(clause.l) ? clause.r : clause.l)) return { ok: false, why: 'the comparison value is volatile' };
  const cols = columnsOf(operand);
  for (let i = 0; i < keys.length; i++) {
    if ([...columnsOf(keys[i])].some((c) => cols.has(c))) {
      if (clause.k === 'booltest') return { ok: false, why: `${clause.test} is not an indexable form` };
      return { ok: false, why: `${deparse(stripColl(operand))} is not equal() to key ${keyDisplay(keys[i])}: ${diffReason(stripColl(operand), keys[i])}` };
    }
  }
  return { ok: false, why: `no index key uses ${[...cols].join(', ') || 'this expression'}` };
}

const wrapQuals = (xs: string[]) => (xs.length === 1 ? `(${xs[0]})` : `(${xs.map((x) => `(${x})`).join(' AND ')})`);

export function analyze(keysSrc: string, whereSrc: string, querySrc: string, isTarget: boolean): Analysis {
  const base: Analysis = { stage: 'index-error', error: '', problems: [], keys: [], keyTexts: [], pred: null, clauses: [], predAtoms: [], predOK: null, steps: [], truncated: false, exhausted: false, explain: [], statement: '' };
  let rawKeys: Node[] = [];
  let rawPred: Node | null = null;
  try {
    const parts = splitTopLevel(keysSrc);
    if (!parts.length) throw new SqlError('an index needs at least one key');
    if (parts.length > 4) throw new SqlError('the lab allows up to four index keys');
    rawKeys = parts.map((s) => parse(s, false));
    if (whereSrc.trim()) {
      rawPred = parse(whereSrc, false);
      if (rawPred.ty !== 'bool') throw new SqlError(`argument of WHERE must be type boolean, not type ${TYPE_NAME[rawPred.ty]}`);
    }
  } catch (e) {
    return { ...base, error: e instanceof Error ? e.message : String(e) };
  }
  const keys = rawKeys.map(normalize);
  const pred = rawPred ? normalize(rawPred) : null;
  const keyTexts = keys.map(keyDisplay);
  const statement = `CREATE INDEX idx ON orders (${rawKeys.map(keyDisplay).join(', ')})${rawPred ? ` WHERE ${deparse(rawPred)}` : ''};`;
  const withIdx = { ...base, keys, keyTexts, pred, statement };

  // DefineIndex checks the predicate first, then the key expressions.
  const predProblems = rawPred ? nonImmutableNodes(rawPred, 'predicate', -1) : [];
  const keyProblems = rawKeys.flatMap((k, i) => nonImmutableNodes(k, 'expression', i));
  if (predProblems.length || keyProblems.length) {
    const where = predProblems.length ? 'predicate' : 'expression';
    return { ...withIdx, stage: 'create-error', problems: [...predProblems, ...keyProblems], error: `functions in index ${where} must be marked IMMUTABLE`, explain: [`ERROR:  functions in index ${where} must be marked IMMUTABLE`] };
  }

  let query: Node;
  try {
    const raw = parse(querySrc, true);
    if (raw.ty !== 'bool') throw new SqlError(`argument of WHERE must be type boolean, not type ${TYPE_NAME[raw.ty]}`);
    query = normalize(raw);
  } catch (e) {
    return { ...withIdx, stage: 'query-error', error: e instanceof Error ? e.message : String(e) };
  }

  const qs = conjuncts(query);
  const ps = conjuncts(pred);
  const clauses: ClauseInfo[] = qs.map((n) => ({ node: n, text: deparse(n), role: 'filter', key: -1, why: '' }));
  let predOK: boolean | null = null;
  let steps: Step[] = [];
  let truncated = false;
  let exhausted = false;
  let predAtoms: PredAtom[] = [];
  if (pred) {
    const prover = new Prover(true);
    predOK = prover.implies(listNode(qs), listNode(ps));
    steps = prover.steps;
    truncated = prover.truncated;
    exhausted = prover.exhausted;
    predAtoms = ps.map((atom) => ({
      node: atom,
      text: deparse(atom),
      proved: new Prover(false).implies(listNode(qs), atom),
      by: qs.map((c, i) => (new Prover(false).implies(c, atom) ? i : -1)).filter((i) => i >= 0),
    }));
  }

  if (pred && !predOK) {
    clauses.forEach((c) => {
      c.role = 'unused';
      c.why = 'the index is not considered, so every clause is checked some other way';
    });
    return { ...withIdx, stage: 'not-implied', clauses, predAtoms, predOK, steps, truncated, exhausted, explain: ['Seq Scan on orders', `  Filter: ${wrapQuals(qs.map((n) => deparse(n)))}`] };
  }

  for (const c of clauses) {
    if (pred && !isTarget && !containsMutable(c.node) && new Prover(false).implies(listNode(ps), c.node)) {
      c.role = 'implied';
      c.why = 'implied by the index predicate: every row in the index already satisfies it';
      continue;
    }
    const m = matchClause(c.node, keys);
    if (m.ok) {
      c.role = 'cond';
      c.key = m.key;
      c.cond = m.cond;
      c.why = `searches key ${m.key + 1}, ${keyTexts[m.key]}`;
    } else {
      c.role = 'filter';
      c.why = m.why;
      if (pred && isTarget && new Prover(false).implies(listNode(ps), c.node)) c.why = 'implied by the predicate, but kept: UPDATE/DELETE and FOR UPDATE rechecks need it in the plan';
    }
  }
  const conds = clauses.filter((c) => c.role === 'cond');
  const filters = clauses.filter((c) => c.role === 'filter');
  const explain = [`Index Scan using idx on orders`];
  if (conds.length) explain.push(`  Index Cond: ${wrapQuals(conds.map((c) => c.cond ?? c.text))}`);
  if (filters.length) explain.push(`  Filter: ${wrapQuals(filters.map((c) => c.text))}`);
  const stage: Stage = conds.length ? 'usable' : pred ? 'full-scan' : 'no-key-match';
  if (stage === 'no-key-match') explain.splice(0, explain.length, 'Seq Scan on orders', `  Filter: ${wrapQuals(qs.map((n) => deparse(n)))}`);
  return { ...withIdx, stage, clauses, predAtoms, predOK, steps, truncated, exhausted, explain };
}
/* ============================================================ presets + UI */

export type Preset = { value: string; label: string; keys: string; where: string; query: string; target?: boolean };

export const PRESETS: Preset[] = [
  { value: 'soft', label: 'Soft delete: predicate repeated in the query', keys: 'customer_id', where: 'deleted_at IS NULL', query: 'customer_id = 42 AND deleted_at IS NULL' },
  { value: 'forgot', label: 'Soft delete: query forgets the predicate', keys: 'customer_id', where: 'deleted_at IS NULL', query: 'customer_id = 42' },
  { value: 'tighter', label: 'Range: amount > 600 proves amount > 500', keys: 'customer_id', where: 'amount > 500', query: 'customer_id = 42 AND amount > 600' },
  { value: 'ge', label: 'Range: amount >= 500 does not', keys: 'customer_id', where: 'amount > 500', query: 'customer_id = 42 AND amount >= 500' },
  { value: 'inlist', label: 'IN list expands to an OR', keys: 'customer_id', where: 'amount > 500', query: 'customer_id = 42 AND amount IN (600, 700)' },
  { value: 'param', label: 'Bind parameter in a generic plan', keys: 'customer_id', where: 'amount > 500', query: 'customer_id = 42 AND amount > $1' },
  { value: 'not', label: 'NOT pushed down before the proof', keys: 'customer_id', where: 'amount > 500', query: 'customer_id = 42 AND NOT (amount <= 500)' },
  { value: 'ne', label: "status = 'paid' proves status <> 'cancelled'", keys: 'customer_id', where: "status <> 'cancelled'", query: "customer_id = 42 AND status = 'paid'" },
  { value: 'or', label: 'OR in the query, IN list in the predicate', keys: 'customer_id', where: "status IN ('pending', 'paid')", query: "customer_id = 42 AND (status = 'paid' OR status = 'pending')" },
  { value: 'strict', label: 'IS NOT NULL proved by a strict operator', keys: 'customer_id', where: 'email IS NOT NULL', query: "customer_id = 42 AND email = 'ann@example.com'" },
  { value: 'now', label: 'now() in the query is not a constant', keys: 'customer_id', where: "created_at > '2026-09-01'", query: "customer_id = 42 AND created_at > now() - interval '7 days'" },
  { value: 'update', label: 'UPDATE keeps the implied clause', keys: 'customer_id', where: 'deleted_at IS NULL', query: 'customer_id = 42 AND deleted_at IS NULL', target: true },
  { value: 'lower', label: 'Expression: lower(email), constant folded', keys: 'lower(email)', where: '', query: "lower(email) = lower('Ann@Example.com')" },
  { value: 'upper', label: 'Expression: upper() does not match lower()', keys: 'lower(email)', where: '', query: "upper(email) = 'ANN@EXAMPLE.COM'" },
  { value: 'collate', label: 'Expression: COLLATE "C" on the comparison', keys: 'lower(email)', where: '', query: 'lower(email) = \'ann@example.com\' COLLATE "C"' },
  { value: 'json', label: "Expression: #>> is not ->>", keys: "data->>'sku'", where: '', query: "data #>> '{sku}' = 'SKU-5'" },
  { value: 'tz', label: 'Expression: created_at::date is STABLE', keys: 'created_at::date', where: '', query: "created_at::date = '2026-09-01'" },
  { value: 'tzfix', label: "Expression: AT TIME ZONE 'UTC' fixes it", keys: "(created_at AT TIME ZONE 'UTC')::date", where: '', query: "(created_at AT TIME ZONE 'UTC')::date = '2026-09-01'" },
  { value: 'concat', label: 'Expression: concat() is STABLE, || is not', keys: "concat(status, ':', email)", where: '', query: "concat(status, ':', email) = 'paid:ann@example.com'" },
  { value: 'nowpred', label: 'Predicate: WHERE created_at > now() - 7 days', keys: 'customer_id', where: "created_at > now() - interval '7 days'", query: 'customer_id = 42' },
];

const ROLE_COLOR: Record<ClauseRole, string> = { cond: 'var(--viz-1)', filter: 'var(--viz-2)', implied: 'var(--viz-7)', unused: 'var(--viz-ink-muted)' };
const ROLE_LABEL: Record<ClauseRole, string> = { cond: 'Index Cond', filter: 'Filter', implied: 'dropped: implied', unused: 'index not usable' };

type Box = { x: number; y: number; w: number; text: string; full: string; stroke: string; dash?: string; tag: string };

function flow(items: { text: string; stroke: string; dash?: string; tag: string }[], y0: number, W: number, left: number) {
  const boxes: Box[] = [];
  let x = left;
  let y = y0;
  for (const it of items) {
    const shown = it.text.length > 44 ? `${it.text.slice(0, 43)}…` : it.text;
    const w = Math.max(64, Math.min(W - left - 8, shown.length * 6.3 + 18, Math.max(shown.length, it.tag.length) * 6.3 + 18));
    if (x + w > W - 8 && x > left) {
      x = left;
      y += 44;
    }
    boxes.push({ x, y, w, text: shown, full: it.text, stroke: it.stroke, dash: it.dash, tag: it.tag });
    x += w + 10;
  }
  return { boxes, bottom: y + 34 };
}

const inputStyle = {
  font: 'inherit',
  fontSize: '0.8rem',
  background: 'var(--viz-plane)',
  color: 'var(--viz-ink)',
  border: '1px solid var(--viz-border)',
  borderRadius: 6,
  padding: '0.3rem 0.45rem',
  width: '100%',
  boxSizing: 'border-box' as const,
};

export default function PartialIndexPredtestLab() {
  const [preset, setPreset] = useState('soft');
  const [keys, setKeys] = useState(PRESETS[0].keys);
  const [where, setWhere] = useState(PRESETS[0].where);
  const [query, setQuery] = useState(PRESETS[0].query);
  const [target, setTarget] = useState(false);
  const [shownSteps, setShownSteps] = useState(999);
  const [sizeRef, width] = useSize(680);

  const a = useMemo(() => analyze(keys, where, query, target), [keys, where, query, target]);

  const choose = (v: string) => {
    const p = PRESETS.find((x) => x.value === v) ?? PRESETS[0];
    setPreset(p.value);
    setKeys(p.keys);
    setWhere(p.where);
    setQuery(p.query);
    setTarget(!!p.target);
    setShownSteps(999);
  };
  const edit = (f: (s: string) => void) => (e: { currentTarget: { value: string } }) => {
    f(e.currentTarget.value);
    setPreset('custom');
    setShownSteps(999);
  };

  // Lay out to the container: on a phone the row labels move above the boxes instead of beside them.
  const W = Math.max(280, Math.min(680, Math.floor(width)));
  const narrow = W < 540;
  const LEFT = narrow ? 8 : 110;
  const LABEL_H = narrow ? 18 : 0;
  const keyRow = flow(
    a.keyTexts.length ? a.keyTexts.map((t, i) => ({ text: t, stroke: a.problems.some((p) => p.key === i) ? 'var(--viz-critical)' : 'var(--viz-ink-2)', tag: a.problems.some((p) => p.key === i) ? `key ${i + 1}: not immutable` : `key ${i + 1}` })) : [{ text: '—', stroke: 'var(--viz-ink-muted)', tag: 'key' }],
    8 + LABEL_H,
    W,
    LEFT,
  );
  const clauseRow = flow(
    a.clauses.length ? a.clauses.map((c) => ({ text: c.text, stroke: ROLE_COLOR[c.role], dash: c.role === 'implied' ? '5 3' : undefined, tag: ROLE_LABEL[c.role] })) : [{ text: a.stage === 'create-error' ? 'not planned: CREATE INDEX failed' : a.stage === 'query-error' ? 'query does not parse' : '—', stroke: 'var(--viz-ink-muted)', tag: '' }],
    keyRow.bottom + 40 + LABEL_H,
    W,
    LEFT,
  );
  const predItems = a.pred
    ? a.stage === 'create-error'
      ? [{ text: deparse(a.pred), stroke: a.problems.some((p) => p.where === 'predicate') ? 'var(--viz-critical)' : 'var(--viz-ink-2)', tag: a.problems.some((p) => p.where === 'predicate') ? 'not immutable' : 'predicate' }]
      : a.predAtoms.length
        ? a.predAtoms.map((p) => ({ text: p.text, stroke: p.proved ? 'var(--viz-good)' : 'var(--viz-critical)', tag: p.proved ? 'proved' : 'not proved' }))
        : [{ text: deparse(a.pred), stroke: 'var(--viz-ink-muted)', tag: 'predicate' }]
    : [{ text: 'none — the index is not partial', stroke: 'var(--viz-ink-muted)', dash: '3 3', tag: '' }];
  const predRow = flow(predItems, clauseRow.bottom + 40 + LABEL_H, W, LEFT);
  const H = predRow.bottom + 12;

  const links: { x1: number; y1: number; x2: number; y2: number; color: string }[] = [];
  a.clauses.forEach((c, i) => {
    const cb = clauseRow.boxes[i];
    if (!cb) return;
    if (c.role === 'cond' && keyRow.boxes[c.key]) {
      const kb = keyRow.boxes[c.key];
      links.push({ x1: cb.x + cb.w / 2, y1: cb.y, x2: kb.x + kb.w / 2, y2: kb.y + 34, color: 'var(--viz-1)' });
    }
  });
  a.predAtoms.forEach((p, j) => {
    const pb = predRow.boxes[j];
    if (!pb) return;
    for (const i of p.by) {
      const cb = clauseRow.boxes[i];
      if (cb) links.push({ x1: cb.x + cb.w / 2, y1: cb.y + 34, x2: pb.x + pb.w / 2, y2: pb.y, color: 'var(--viz-good)' });
    }
  });

  const steps = a.steps.slice(0, Math.min(shownSteps, a.steps.length));
  const allShown = steps.length === a.steps.length;
  const counts = (r: ClauseRole) => a.clauses.filter((c) => c.role === r).length;

  const verdict =
    a.stage === 'index-error'
      ? 'index definition error'
      : a.stage === 'create-error'
        ? 'CREATE INDEX fails'
        : a.stage === 'query-error'
          ? 'query error'
          : a.stage === 'not-implied'
            ? 'index not usable'
            : a.stage === 'no-key-match'
              ? 'no clause matches a key'
              : a.stage === 'full-scan'
                ? 'usable, whole index'
                : 'usable';

  const rowLabel = (y: number, text: string) => (
    <text x={8} y={narrow ? y - 6 : y + 21} fontSize={12} fill="var(--viz-ink)" stroke="var(--viz-surface)" strokeWidth={4} paintOrder="stroke">
      {text}
    </text>
  );

  return (
    <VizPanel
      title="Will the planner use this index?"
      subtitle="Edit the index keys, the index WHERE clause and the query. The lab parses all three against an orders table, checks that CREATE INDEX would accept them, runs PostgreSQL's implication proof step by step, and matches query clauses to index keys the way the planner does."
      controls={
        <>
          <Choice label="Scenario" value={preset} onChange={choose} options={[...PRESETS.map((p) => ({ value: p.value, label: p.label })), ...(preset === 'custom' ? [{ value: 'custom', label: 'Your own edits' }] : [])]} />
          <Check label="Query is UPDATE / DELETE / FOR UPDATE" checked={target} onChange={(v) => { setTarget(v); setShownSteps(999); }} />
        </>
      }
      legend={
        <Legend
          items={[
            { label: 'Index Cond (searches a key)', color: 'var(--viz-1)' },
            { label: 'Filter (checked per row)', color: 'var(--viz-2)' },
            { label: 'Dropped: implied by the predicate', color: 'var(--viz-7)' },
            { label: 'Predicate item proved', color: 'var(--viz-good)' },
            { label: 'Not proved / not immutable', color: 'var(--viz-critical)' },
          ]}
        />
      }
      stats={
        <Stats
          items={[
            { label: 'Verdict', value: verdict },
            { label: 'CREATE INDEX', value: a.stage === 'index-error' ? '—' : a.stage === 'create-error' ? 'error' : 'accepted', hint: 'Every function, operator and cast in the keys and the predicate must be IMMUTABLE.' },
            { label: 'Predicate implied', value: a.pred === null || a.stage === 'create-error' || a.stage === 'index-error' || a.stage === 'query-error' ? '—' : a.predOK ? 'yes' : 'no' },
            { label: 'Index Cond / Filter', value: `${counts('cond')} / ${counts('filter')}` },
            { label: 'Clauses dropped', value: String(counts('implied')), hint: 'Query clauses implied by the index predicate need not be rechecked, so they leave the plan.' },
          ]}
        />
      }
      note={<Note>{narrate(a, target)}</Note>}
      table={
        <table className="viz-table">
          <thead>
            <tr>
              <th>Item</th>
              <th>Role</th>
              <th>Why</th>
            </tr>
          </thead>
          <tbody>
            {a.clauses.map((c, i) => (
              <tr key={`c${i}`}>
                <td>
                  <code>{c.text}</code>
                </td>
                <td>{ROLE_LABEL[c.role]}</td>
                <td>{c.why}</td>
              </tr>
            ))}
            {a.predAtoms.map((p, i) => (
              <tr key={`p${i}`}>
                <td>
                  <code>{p.text}</code>
                </td>
                <td>{p.proved ? 'predicate item proved' : 'predicate item not proved'}</td>
                <td>{p.by.length ? `by ${p.by.map((j) => a.clauses[j]?.text).join('; ')}` : p.proved ? 'by the WHERE clause as a whole' : 'no clause implies it'}</td>
              </tr>
            ))}
            {a.problems.map((p, i) => (
              <tr key={`x${i}`}>
                <td>
                  <code>{p.text}</code>
                </td>
                <td>{volWord(p.vol)}</td>
                <td>{p.sig}</td>
              </tr>
            ))}
          </tbody>
        </table>
      }
    >
      <div style={{ display: 'grid', gap: '0.45rem', marginBottom: '0.75rem', minWidth: 0 }}>
        <label style={{ display: 'grid', gap: 2, fontSize: '0.75rem', color: 'var(--viz-ink-2)' }}>
          CREATE INDEX idx ON orders ( keys, comma-separated )
          <input type="text" spellCheck={false} value={keys} onChange={edit(setKeys)} style={inputStyle} aria-label="Index keys" />
        </label>
        <label style={{ display: 'grid', gap: 2, fontSize: '0.75rem', color: 'var(--viz-ink-2)' }}>
          WHERE ( index predicate — leave empty for a full index )
          <input type="text" spellCheck={false} value={where} onChange={edit(setWhere)} style={inputStyle} aria-label="Index predicate" />
        </label>
        <label style={{ display: 'grid', gap: 2, fontSize: '0.75rem', color: 'var(--viz-ink-2)' }}>
          {target ? 'UPDATE orders SET … WHERE' : 'SELECT * FROM orders WHERE'}
          <input type="text" spellCheck={false} value={query} onChange={edit(setQuery)} style={inputStyle} aria-label="Query WHERE clause" />
        </label>
        <p style={{ margin: 0, fontSize: '0.72rem', color: 'var(--viz-ink-muted)' }}>
          orders(id bigint, customer_id bigint, status text, amount integer, email text, created_at timestamptz, deleted_at timestamptz, data jsonb, archived boolean)
        </p>
      </div>

      {a.error && (a.stage === 'index-error' || a.stage === 'query-error') ? (
        <p style={{ margin: '0 0 0.6rem', fontSize: '0.8rem', color: 'var(--viz-ink)', borderLeft: '3px solid var(--viz-critical)', paddingLeft: 8 }}>
          ERROR: {a.error}
        </p>
      ) : null}

      <div ref={sizeRef}>
      <svg viewBox={`0 0 ${W} ${H}`} width={W} height={H} role="img" aria-label={`Index keys, query clauses and index predicate. Verdict: ${verdict}.`}>
        {links.map((l, i) => (
          <line key={i} x1={l.x1} y1={l.y1} x2={l.x2} y2={l.y2} stroke={l.color} strokeWidth={2} />
        ))}
        {[...keyRow.boxes, ...clauseRow.boxes, ...predRow.boxes].map((b, i) => (
          <g key={i}>
            <title>{b.full}</title>
            <rect x={b.x} y={b.y} width={b.w} height={34} rx={5} fill="var(--viz-surface)" stroke={b.stroke} strokeWidth={2} strokeDasharray={b.dash} />
            <text x={b.x + 8} y={b.y + 14} fontSize={11} fill="var(--viz-ink)">
              {b.text}
            </text>
            {b.tag ? (
              <text x={b.x + 8} y={b.y + 28} fontSize={10} fill="var(--viz-ink-2)">
                {b.tag}
              </text>
            ) : null}
          </g>
        ))}
        {rowLabel(keyRow.boxes[0]?.y ?? 0, 'Index keys')}
        {rowLabel(clauseRow.boxes[0]?.y ?? 0, 'Query clauses')}
        {rowLabel(predRow.boxes[0]?.y ?? 0, 'Index predicate')}
      </svg>
      </div>

      {a.pred && a.stage !== 'create-error' && a.stage !== 'index-error' && a.stage !== 'query-error' ? (
        <div style={{ marginTop: '0.6rem' }}>
          <div className="viz-controls" style={{ marginBottom: '0.4rem' }}>
            <Button onClick={() => setShownSteps(1)}>Step through the proof</Button>
            <Button primary onClick={() => setShownSteps((s) => Math.min(a.steps.length, s + 1))} disabled={allShown}>
              Next step
            </Button>
            <Button onClick={() => setShownSteps(999)} disabled={allShown}>
              Show all
            </Button>
          </div>
          <ol aria-label="Implication proof" style={{ margin: 0, paddingLeft: '1.2rem', fontSize: '0.78rem', display: 'grid', gap: 3 }}>
            {steps.map((s, i) => (
              <li key={i} style={{ color: 'var(--viz-ink)', marginLeft: s.depth * 16, borderLeft: `3px solid ${s.ok === null ? 'var(--viz-ink-muted)' : s.ok ? 'var(--viz-good)' : 'var(--viz-critical)'}`, paddingLeft: 6 }}>
                <span>{withCode(s.text)}</span> <strong>{s.ok === null ? '' : s.ok ? 'proved' : 'not proved'}</strong>
                {s.rule ? <span style={{ display: 'block', color: 'var(--viz-ink-2)' }}>{s.rule}</span> : null}
              </li>
            ))}
          </ol>
          {a.truncated || a.exhausted ? (
            <p style={{ fontSize: '0.75rem', color: 'var(--viz-ink-2)', margin: '0.3rem 0 0' }}>
              {a.exhausted ? 'The lab stopped this proof search at its own step limit and reports it as not proved; PostgreSQL would keep going.' : 'Proof trace shortened for display; the verdict uses the full proof.'}
            </p>
          ) : null}
        </div>
      ) : null}

      {a.problems.length ? (
        <ul aria-label="Non-immutable parts" style={{ margin: '0.6rem 0 0', paddingLeft: '1.2rem', fontSize: '0.78rem' }}>
          {a.problems.map((p, i) => (
            <li key={i} style={{ color: 'var(--viz-ink)' }}>
              <code>{p.text}</code> in the index {p.where} calls {p.sig}, which is {volWord(p.vol)}.
            </li>
          ))}
        </ul>
      ) : null}

      {a.explain.length ? (
        <pre aria-label="Plan shape" style={{ margin: '0.6rem 0 0', padding: '0.5rem 0.65rem', background: 'var(--viz-plane)', border: '1px solid var(--viz-border)', borderRadius: 6, fontSize: '0.75rem', color: 'var(--viz-ink)', overflowX: 'auto', whiteSpace: 'pre' }}>
          {a.explain.join('\n')}
        </pre>
      ) : null}
    </VizPanel>
  );
}

/** Renders `backticked` spans of a trace line as code. */
function withCode(text: string) {
  return text.split('`').map((part, i) => (i % 2 ? <code key={i}>{part}</code> : <span key={i}>{part}</span>));
}

export function narrate(a: Analysis, target: boolean): string {
  switch (a.stage) {
    case 'index-error':
      return `The index definition does not parse: ${a.error}.`;
    case 'query-error':
      return `The query does not parse: ${a.error}.`;
    case 'create-error': {
      const p = a.problems[0];
      const what = p.sig.startsWith(`${p.text.replace(/\(.*$/, '')}(`) ? `${p.sig} is` : `${p.text} uses ${p.sig}, which is`;
      return `CREATE INDEX stops with "${a.error}": ${what} ${volWord(p.vol)}. An index stores each computed value once, so anything whose result can change while the row stays the same is refused.`;
    }
    case 'not-implied': {
      const miss = a.predAtoms.find((p) => !p.proved);
      return `The proof fails${miss ? ` for ${miss.text}` : ''}: the planner cannot show that every row this query wants is in the index, so it builds no index path from it at all.`;
    }
    case 'no-key-match':
      return `The index is a full index, and no query clause matches one of its keys, so the planner has nothing to search it with: ${a.clauses.find((c) => c.role === 'filter')?.why ?? 'no clause refers to a key'}.`;
    case 'full-scan':
      return `The predicate is proved, but no clause searches a key, so the only option is to read the whole partial index — still possibly cheap if the index is small.`;
    default: {
      const dropped = a.clauses.filter((c) => c.role === 'implied').length;
      const kept = a.clauses.filter((c) => c.role === 'filter');
      let s = a.pred ? 'The WHERE clause implies the index predicate, so the index is a candidate. ' : 'A query clause is equal() to an index key, so the index is a candidate. ';
      if (dropped) s += `${dropped === 1 ? 'One clause is' : `${dropped} clauses are`} implied by the predicate itself and ${dropped === 1 ? 'leaves' : 'leave'} the plan. `;
      if (kept.length) s += `${kept.length === 1 ? 'One clause stays' : `${kept.length} clauses stay`} as a Filter: ${kept[0].why}.`;
      return s.trim();
    }
  }
}
