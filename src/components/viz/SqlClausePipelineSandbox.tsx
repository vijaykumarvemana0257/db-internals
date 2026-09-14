import { useState } from 'react';
import {
  VizPanel,
  Segmented,
  Choice,
  Button,
  Legend,
  Stats,
  Note,
  TooltipHost,
  useTip,
  useSize,
  makeRng,
  fmtNum,
} from './Viz';

/**
 * A tiny, deterministic SQL interpreter over a fixed customers/orders/items dataset.
 *
 * The point is the pipeline, not the parser: a SELECT is written
 * SELECT → FROM → WHERE → ORDER BY → LIMIT but evaluated
 * FROM → WHERE → SELECT → DISTINCT → ORDER BY → LIMIT, and every confusing thing about
 * aliases, NULLs and LIMIT falls out of that reordering. Each stage keeps its own
 * intermediate row set so the learner can step through the interpreter's actual state.
 *
 * The engine switch is not cosmetic: PostgreSQL, MySQL and SQLite genuinely disagree
 * about ||, integer division, text→number coercion, declared-type enforcement,
 * VARCHAR length, NULL ordering and whether RETURNING exists at all.
 */

/* ====================================================================== values */

type Eng = 'pg' | 'mysql' | 'sqlite';
type Ty = 'int' | 'num' | 'text' | 'bool' | 'date' | 'unknown' | 'null';
type Val = number | string | boolean | null;
type Cell = { v: Val; t: Ty; aff?: 'num' | 'text' };

class SqlErr extends Error {}
function fail(m: string): never {
  throw new SqlErr(m);
}

const PGNAME: Record<Ty, string> = {
  int: 'integer',
  num: 'numeric',
  text: 'text',
  bool: 'boolean',
  date: 'date',
  unknown: 'unknown',
  null: 'unknown',
};

const ENGINES = [
  { value: 'pg' as const, label: 'PostgreSQL', title: 'Strict types, || is concat, / on two integers truncates' },
  { value: 'mysql' as const, label: 'MySQL 8', title: 'Default sql_mode: || is logical OR, no RETURNING, silent numeric coercion' },
  { value: 'sqlite' as const, label: 'SQLite', title: 'Type affinity rather than type enforcement; VARCHAR(n) length ignored' },
];

/* ====================================================================== schema */

type Base = 'INTEGER' | 'NUMERIC' | 'TEXT' | 'VARCHAR' | 'DATE' | 'BOOLEAN';
type ColType = { base: Base; len?: number };
type ColDef = {
  name: string;
  type: ColType;
  notNull: boolean;
  def?: Expr;
  pk: boolean;
  unique: boolean;
  check?: Expr;
};
type Row = Record<string, Val>;
type TableDef = { name: string; cols: ColDef[]; rows: Row[] };
type DB = Record<string, TableDef>;

function typeName(t: ColType) {
  return t.len ? `${t.base}(${t.len})` : t.base;
}

/** Declared type → the type an expression referencing this column carries. */
function declaredTy(t: ColType): Ty {
  switch (t.base) {
    case 'INTEGER':
      return 'int';
    case 'NUMERIC':
      return 'num';
    case 'DATE':
      return 'date';
    case 'BOOLEAN':
      return 'bool';
    default:
      return 'text';
  }
}

/** SQLite has storage classes, not column types: the value decides. */
function storageTy(v: Val): Ty {
  if (v === null) return 'null';
  if (typeof v === 'boolean') return 'int';
  if (typeof v === 'number') return Number.isInteger(v) ? 'int' : 'num';
  return 'text';
}

/* ====================================================================== the AST */

type Expr =
  | { k: 'num'; v: number }
  | { k: 'str'; v: string }
  | { k: 'null' }
  | { k: 'bool'; v: boolean }
  | { k: 'typed'; ty: Base; v: string }
  | { k: 'star' }
  | { k: 'col'; name: string }
  | { k: 'bin'; op: string; l: Expr; r: Expr }
  | { k: 'neg'; e: Expr }
  | { k: 'not'; e: Expr }
  | { k: 'isnull'; e: Expr; neg: boolean }
  | { k: 'in'; e: Expr; list: Expr[]; neg: boolean }
  | { k: 'between'; e: Expr; lo: Expr; hi: Expr; neg: boolean }
  | { k: 'like'; e: Expr; pat: Expr; neg: boolean }
  | { k: 'cast'; e: Expr; to: ColType; pgOp: boolean }
  | { k: 'call'; fn: string; args: Expr[] };

type Item = { e: Expr; alias?: string };
type OrderKey = { e: Expr; desc: boolean; nulls?: 'first' | 'last' };

type Stmt =
  | {
      k: 'select';
      distinct: boolean;
      items: Item[];
      from: string;
      where?: Expr;
      order?: OrderKey[];
      limit?: number;
      offset?: number;
    }
  | { k: 'insert'; table: string; cols?: string[]; tuples: Expr[][]; returning?: Item[] }
  | { k: 'update'; table: string; sets: { col: string; e: Expr }[]; where?: Expr; returning?: Item[] }
  | { k: 'delete'; table: string; where?: Expr; returning?: Item[] }
  | { k: 'create'; table: string; cols: ColDef[] }
  | { k: 'drop'; table: string };

function unparse(e: Expr): string {
  switch (e.k) {
    case 'num':
      return String(e.v);
    case 'str':
      return `'${e.v}'`;
    case 'null':
      return 'NULL';
    case 'bool':
      return e.v ? 'TRUE' : 'FALSE';
    case 'typed':
      return `${e.ty} '${e.v}'`;
    case 'star':
      return '*';
    case 'col':
      return e.name;
    case 'bin':
      return `${unparse(e.l)} ${e.op} ${unparse(e.r)}`;
    case 'neg':
      return `-${unparse(e.e)}`;
    case 'not':
      return `NOT ${unparse(e.e)}`;
    case 'isnull':
      return `${unparse(e.e)} IS ${e.neg ? 'NOT ' : ''}NULL`;
    case 'in':
      return `${unparse(e.e)} ${e.neg ? 'NOT ' : ''}IN (${e.list.map(unparse).join(', ')})`;
    case 'between':
      return `${unparse(e.e)} ${e.neg ? 'NOT ' : ''}BETWEEN ${unparse(e.lo)} AND ${unparse(e.hi)}`;
    case 'like':
      return `${unparse(e.e)} ${e.neg ? 'NOT ' : ''}LIKE ${unparse(e.pat)}`;
    case 'cast':
      return e.pgOp ? `${unparse(e.e)}::${typeName(e.to)}` : `CAST(${unparse(e.e)} AS ${typeName(e.to)})`;
    case 'call':
      return `${e.fn}(${e.args.map(unparse).join(', ')})`;
  }
}

/* ==================================================================== the lexer */

type Tok = { k: 'id' | 'num' | 'str' | 'op' | 'eof'; s: string; u: string };

const OPS2 = ['<=', '>=', '<>', '!=', '||', '::'];
const OPS1 = ['(', ')', ',', ';', '*', '/', '%', '+', '-', '=', '<', '>', '.'];

function lex(src: string): Tok[] {
  const out: Tok[] = [];
  let i = 0;
  while (i < src.length) {
    const c = src[i];
    if (c === ' ' || c === '\n' || c === '\t' || c === '\r') {
      i++;
      continue;
    }
    if (c === '-' && src[i + 1] === '-') {
      while (i < src.length && src[i] !== '\n') i++;
      continue;
    }
    if (c === "'") {
      let j = i + 1;
      let s = '';
      let closed = false;
      while (j < src.length) {
        if (src[j] === "'" && src[j + 1] === "'") {
          s += "'";
          j += 2;
          continue;
        }
        if (src[j] === "'") {
          closed = true;
          break;
        }
        s += src[j];
        j++;
      }
      if (!closed) fail('unterminated quoted string');
      out.push({ k: 'str', s, u: s });
      i = j + 1;
      continue;
    }
    if (c === '"') {
      const j = src.indexOf('"', i + 1);
      if (j < 0) fail('unterminated quoted identifier');
      const s = src.slice(i + 1, j);
      out.push({ k: 'id', s, u: s.toUpperCase() });
      i = j + 1;
      continue;
    }
    if ((c >= '0' && c <= '9') || (c === '.' && src[i + 1] >= '0' && src[i + 1] <= '9')) {
      let j = i;
      while (j < src.length && ((src[j] >= '0' && src[j] <= '9') || src[j] === '.')) j++;
      const s = src.slice(i, j);
      out.push({ k: 'num', s, u: s });
      i = j;
      continue;
    }
    if (/[A-Za-z_]/.test(c)) {
      let j = i;
      while (j < src.length && /[A-Za-z0-9_]/.test(src[j])) j++;
      const s = src.slice(i, j);
      out.push({ k: 'id', s, u: s.toUpperCase() });
      i = j;
      continue;
    }
    const two = src.slice(i, i + 2);
    if (OPS2.includes(two)) {
      out.push({ k: 'op', s: two, u: two });
      i += 2;
      continue;
    }
    if (OPS1.includes(c)) {
      out.push({ k: 'op', s: c, u: c });
      i++;
      continue;
    }
    fail(`syntax error at or near "${c}"`);
  }
  out.push({ k: 'eof', s: '', u: '' });
  return out;
}

/* =================================================================== the parser */

const TYPE_WORDS: Record<string, ColType> = {
  INT: { base: 'INTEGER' },
  INTEGER: { base: 'INTEGER' },
  BIGINT: { base: 'INTEGER' },
  SIGNED: { base: 'INTEGER' },
  UNSIGNED: { base: 'INTEGER' },
  NUMERIC: { base: 'NUMERIC' },
  DECIMAL: { base: 'NUMERIC' },
  REAL: { base: 'NUMERIC' },
  DOUBLE: { base: 'NUMERIC' },
  TEXT: { base: 'TEXT' },
  CHAR: { base: 'TEXT' },
  VARCHAR: { base: 'VARCHAR' },
  DATE: { base: 'DATE' },
  BOOLEAN: { base: 'BOOLEAN' },
  BOOL: { base: 'BOOLEAN' },
};

function parse(src: string, eng: Eng): Stmt[] {
  const t = lex(src);
  let p = 0;
  const peek = () => t[p];
  const at = (u: string) => t[p].u === u && t[p].k !== 'str';
  const take = (u: string) => {
    if (!at(u)) return false;
    p++;
    return true;
  };
  const want = (u: string) => {
    if (!take(u)) fail(`syntax error at or near "${t[p].s || 'end of statement'}" — expected ${u}`);
  };
  const ident = () => {
    if (t[p].k !== 'id') fail(`syntax error at or near "${t[p].s || 'end of statement'}"`);
    return t[p++].s;
  };

  /* ------------------------------------------------------------ expressions */

  function primary(): Expr {
    const tk = peek();
    if (tk.k === 'num') {
      p++;
      return { k: 'num', v: Number(tk.s) };
    }
    if (tk.k === 'str') {
      p++;
      return { k: 'str', v: tk.s };
    }
    if (tk.k === 'op' && tk.u === '(') {
      p++;
      const e = expr();
      want(')');
      return e;
    }
    if (tk.k === 'op' && tk.u === '-') {
      p++;
      return { k: 'neg', e: unary() };
    }
    if (tk.k === 'op' && tk.u === '+') {
      p++;
      return unary();
    }
    if (tk.k === 'op' && tk.u === '*') {
      p++;
      return { k: 'star' };
    }
    if (tk.k === 'id') {
      if (tk.u === 'NULL') {
        p++;
        return { k: 'null' };
      }
      if (tk.u === 'TRUE' || tk.u === 'FALSE') {
        p++;
        return { k: 'bool', v: tk.u === 'TRUE' };
      }
      if (tk.u === 'NOT') {
        p++;
        return { k: 'not', e: unary() };
      }
      if (tk.u === 'CAST') {
        p++;
        want('(');
        const e = expr();
        want('AS');
        const to = colType(true);
        want(')');
        return { k: 'cast', e, to, pgOp: false };
      }
      if ((tk.u === 'DATE' || tk.u === 'NUMERIC' || tk.u === 'INTEGER') && t[p + 1].k === 'str') {
        p++;
        const lit = t[p++].s;
        return { k: 'typed', ty: TYPE_WORDS[tk.u].base, v: lit };
      }
      const name = ident();
      if (at('(')) {
        p++;
        const args: Expr[] = [];
        if (!at(')')) {
          do {
            args.push(expr());
          } while (take(','));
        }
        want(')');
        return { k: 'call', fn: name.toUpperCase(), args };
      }
      // qualified name customers.city — the single-table pages only need the tail
      if (at('.')) {
        p++;
        return { k: 'col', name: ident() };
      }
      return { k: 'col', name };
    }
    return fail(`syntax error at or near "${tk.s || 'end of statement'}"`);
  }

  // MySQL's CAST target list is its own: SIGNED/UNSIGNED, DECIMAL, CHAR, DATE, DOUBLE,
  // FLOAT, REAL, BINARY, JSON, TIME, YEAR — INTEGER, TEXT and BOOLEAN are not in it.
  const MYSQL_CAST_TARGETS = new Set(['SIGNED', 'UNSIGNED', 'DECIMAL', 'CHAR', 'DATE', 'DOUBLE', 'REAL']);

  function colType(inCast = false): ColType {
    const w = peek();
    if (w.k !== 'id' || !TYPE_WORDS[w.u]) fail(`type "${w.s}" does not exist`);
    // SQLite accepts any type name in a CAST and applies affinity rules, so SIGNED is legal
    // there; Postgres has no such type.
    if (eng === 'pg' && (w.u === 'SIGNED' || w.u === 'UNSIGNED')) fail(`type "${w.s.toLowerCase()}" does not exist`);
    if (inCast && eng === 'mysql' && !MYSQL_CAST_TARGETS.has(w.u))
      fail(
        `ERROR 1064 (42000): You have an error in your SQL syntax near '${w.s}' — MySQL casts to SIGNED/UNSIGNED, DECIMAL or CHAR, not ${w.s.toUpperCase()}`,
      );
    p++;
    const ct: ColType = { ...TYPE_WORDS[w.u] };
    if (at('(')) {
      p++;
      const n = Number(peek().s);
      p++;
      if (at(',')) {
        p++;
        p++;
      }
      want(')');
      if (ct.base === 'VARCHAR' || ct.base === 'TEXT') ct.len = n;
    }
    return ct;
  }

  function postfix(): Expr {
    let e = primary();
    while (at('::')) {
      if (eng !== 'pg') fail(`syntax error at or near "::" — the :: cast operator is PostgreSQL-only`);
      p++;
      e = { k: 'cast', e, to: colType(true), pgOp: true };
    }
    return e;
  }

  function unary(): Expr {
    return postfix();
  }

  function mul(): Expr {
    let e = unary();
    while (at('*') || at('/') || at('%')) {
      const op = t[p++].u;
      e = { k: 'bin', op, l: e, r: unary() };
    }
    return e;
  }

  function add(): Expr {
    let e = mul();
    for (;;) {
      if (at('+') || at('-')) {
        const op = t[p++].u;
        e = { k: 'bin', op, l: e, r: mul() };
      } else if (at('||') && eng !== 'mysql') {
        p++;
        e = { k: 'bin', op: '||', l: e, r: mul() };
      } else break;
    }
    return e;
  }

  function compare(): Expr {
    let e = add();
    for (;;) {
      if (at('=') || at('<') || at('>') || at('<=') || at('>=') || at('<>') || at('!=')) {
        const op = t[p++].u;
        e = { k: 'bin', op: op === '!=' ? '<>' : op, l: e, r: add() };
        continue;
      }
      if (at('IS')) {
        p++;
        const neg = take('NOT');
        want('NULL');
        e = { k: 'isnull', e, neg };
        continue;
      }
      let neg = false;
      if (at('NOT') && (t[p + 1].u === 'IN' || t[p + 1].u === 'BETWEEN' || t[p + 1].u === 'LIKE')) {
        p++;
        neg = true;
      }
      if (at('IN')) {
        p++;
        want('(');
        const list: Expr[] = [];
        do {
          list.push(expr());
        } while (take(','));
        want(')');
        e = { k: 'in', e, list, neg };
        continue;
      }
      if (at('BETWEEN')) {
        p++;
        const lo = add();
        want('AND');
        const hi = add();
        e = { k: 'between', e, lo, hi, neg };
        continue;
      }
      if (at('LIKE')) {
        p++;
        e = { k: 'like', e, pat: add(), neg };
        continue;
      }
      if (neg) fail('syntax error after NOT');
      return e;
    }
  }

  function andE(): Expr {
    let e = compare();
    while (at('AND')) {
      p++;
      e = { k: 'bin', op: 'AND', l: e, r: compare() };
    }
    return e;
  }

  function expr(): Expr {
    let e = andE();
    for (;;) {
      if (at('OR')) {
        p++;
        e = { k: 'bin', op: 'OR', l: e, r: andE() };
      } else if (at('||') && eng === 'mysql') {
        // Default sql_mode has no PIPES_AS_CONCAT: || really is logical OR here.
        p++;
        e = { k: 'bin', op: 'OR', l: e, r: andE() };
      } else return e;
    }
  }

  /* ------------------------------------------------------------- statements */

  function items(): Item[] {
    const out: Item[] = [];
    do {
      const e = expr();
      let alias: string | undefined;
      if (take('AS')) alias = ident();
      else if (peek().k === 'id' && !RESERVED.has(peek().u)) alias = ident();
      out.push({ e, alias });
    } while (take(','));
    return out;
  }

  function returning(): Item[] | undefined {
    if (!at('RETURNING')) return undefined;
    if (eng === 'mysql')
      fail("ERROR 1064 (42000): You have an error in your SQL syntax near 'RETURNING' — MySQL has no RETURNING clause");
    p++;
    return items();
  }

  function selectStmt(): Stmt {
    want('SELECT');
    const distinct = take('DISTINCT');
    if (at('ALL')) p++;
    const its = items();
    want('FROM');
    const from = ident();
    let where: Expr | undefined;
    if (take('WHERE')) where = expr();
    let order: OrderKey[] | undefined;
    if (at('ORDER')) {
      p++;
      want('BY');
      order = [];
      do {
        const e = expr();
        const desc = take('DESC') ? true : (take('ASC'), false);
        let nulls: 'first' | 'last' | undefined;
        if (at('NULLS')) {
          if (eng === 'mysql')
            fail("ERROR 1064 (42000): You have an error in your SQL syntax near 'NULLS' — MySQL has no NULLS FIRST/LAST");
          p++;
          nulls = take('FIRST') ? 'first' : (want('LAST'), 'last');
        }
        order.push({ e, desc, nulls });
      } while (take(','));
    }
    let limit: number | undefined;
    let offset: number | undefined;
    if (take('LIMIT')) limit = Number(t[p++].s);
    if (take('OFFSET')) offset = Number(t[p++].s);
    return { k: 'select', distinct, items: its, from, where, order, limit, offset };
  }

  function createStmt(): Stmt {
    want('CREATE');
    want('TABLE');
    if (at('IF')) {
      p++;
      want('NOT');
      want('EXISTS');
    }
    const table = ident();
    want('(');
    const cols: ColDef[] = [];
    do {
      const name = ident();
      const type = colType();
      const col: ColDef = { name, type, notNull: false, pk: false, unique: false };
      for (;;) {
        if (at('NOT')) {
          p++;
          want('NULL');
          col.notNull = true;
        } else if (at('NULL')) {
          p++;
        } else if (at('PRIMARY')) {
          p++;
          want('KEY');
          col.pk = true;
          col.notNull = true;
        } else if (at('UNIQUE')) {
          p++;
          col.unique = true;
        } else if (at('DEFAULT')) {
          p++;
          col.def = add();
        } else if (at('CHECK')) {
          p++;
          want('(');
          col.check = expr();
          want(')');
        } else break;
      }
      cols.push(col);
    } while (take(','));
    want(')');
    return { k: 'create', table, cols };
  }

  function stmt(): Stmt {
    if (at('SELECT')) return selectStmt();
    if (at('CREATE')) return createStmt();
    if (at('DROP')) {
      p++;
      want('TABLE');
      if (at('IF')) {
        p++;
        want('EXISTS');
      }
      return { k: 'drop', table: ident() };
    }
    if (at('INSERT')) {
      p++;
      want('INTO');
      const table = ident();
      let cols: string[] | undefined;
      if (at('(')) {
        p++;
        cols = [];
        do {
          cols.push(ident());
        } while (take(','));
        want(')');
      }
      want('VALUES');
      const tuples: Expr[][] = [];
      do {
        want('(');
        const row: Expr[] = [];
        do {
          row.push(expr());
        } while (take(','));
        want(')');
        tuples.push(row);
      } while (take(','));
      return { k: 'insert', table, cols, tuples, returning: returning() };
    }
    if (at('UPDATE')) {
      p++;
      const table = ident();
      want('SET');
      const sets: { col: string; e: Expr }[] = [];
      do {
        const col = ident();
        want('=');
        sets.push({ col, e: expr() });
      } while (take(','));
      let where: Expr | undefined;
      if (take('WHERE')) where = expr();
      return { k: 'update', table, sets, where, returning: returning() };
    }
    if (at('DELETE')) {
      p++;
      want('FROM');
      const table = ident();
      let where: Expr | undefined;
      if (take('WHERE')) where = expr();
      return { k: 'delete', table, where, returning: returning() };
    }
    return fail(`syntax error at or near "${peek().s || 'end of input'}"`);
  }

  const out: Stmt[] = [];
  while (peek().k !== 'eof') {
    if (take(';')) continue;
    out.push(stmt());
    if (peek().k !== 'eof') want(';');
  }
  return out;
}

const RESERVED = new Set([
  'FROM', 'WHERE', 'ORDER', 'BY', 'LIMIT', 'OFFSET', 'GROUP', 'HAVING', 'AS', 'AND', 'OR', 'NOT',
  'IS', 'NULL', 'IN', 'BETWEEN', 'LIKE', 'SELECT', 'DISTINCT', 'INSERT', 'UPDATE', 'DELETE', 'SET',
  'VALUES', 'INTO', 'RETURNING', 'ASC', 'DESC', 'NULLS', 'CREATE', 'TABLE', 'DROP', 'ON', 'JOIN',
  'DEFAULT', 'CHECK', 'PRIMARY', 'KEY', 'UNIQUE', 'CAST',
]);

/* ================================================================ evaluation */

/** MySQL and SQLite both read the leading numeric prefix of a string and call it a number. */
function prefixNum(s: string): number {
  const m = /^\s*[-+]?(\d+\.?\d*|\.\d+)/.exec(s);
  return m ? Number(m[0]) : 0;
}
function fullNum(s: string): number | null {
  const trimmed = s.trim();
  if (trimmed === '' || !/^[-+]?(\d+\.?\d*|\.\d+)$/.test(trimmed)) return null;
  return Number(trimmed);
}

const NULLC: Cell = { v: null, t: 'null' };

function numeric(c: Cell, eng: Eng, op: string, other: Cell): number {
  if (c.v === null) return 0;
  if (typeof c.v === 'number') return c.v;
  if (typeof c.v === 'boolean') {
    if (eng === 'pg') fail(`operator does not exist: ${PGNAME[c.t]} ${op} ${PGNAME[other.t]}`);
    return c.v ? 1 : 0;
  }
  if (eng === 'pg') {
    if (c.t === 'unknown') {
      const n = fullNum(c.v);
      if (n === null) fail(`invalid input syntax for type ${PGNAME[other.t] === 'unknown' ? 'integer' : PGNAME[other.t]}: "${c.v}"`);
      return n;
    }
    fail(`operator does not exist: ${PGNAME[c.t]} ${op} ${PGNAME[other.t]}`);
  }
  return prefixNum(c.v);
}

function arith(op: string, a: Cell, b: Cell, eng: Eng): Cell {
  if (a.v === null || b.v === null) return NULLC;
  if (eng === 'pg' && a.t === 'unknown' && b.t === 'unknown')
    fail(`operator is not unique: unknown ${op} unknown`);
  // date + integer / date - date, in the engines that have a date type
  if (eng !== 'sqlite' && (a.t === 'date' || b.t === 'date')) {
    const day = 86400000;
    if (a.t === 'date' && b.t === 'date' && op === '-')
      return { v: Math.round((Date.parse(String(a.v)) - Date.parse(String(b.v))) / day), t: 'int' };
    if (a.t === 'date' && (op === '+' || op === '-')) {
      const n = numeric(b, eng, op, a);
      const d = new Date(Date.parse(String(a.v)) + (op === '+' ? n : -n) * day);
      return { v: d.toISOString().slice(0, 10), t: 'date' };
    }
    if (eng === 'pg') fail(`operator does not exist: ${PGNAME[a.t]} ${op} ${PGNAME[b.t]}`);
  }
  const x = numeric(a, eng, op, b);
  const y = numeric(b, eng, op, a);
  const bothInt =
    (a.t === 'int' || (a.t === 'unknown' && Number.isInteger(x))) &&
    (b.t === 'int' || (b.t === 'unknown' && Number.isInteger(y)));
  // NUMERIC is exact decimal in every engine here, so round away IEEE-754 residue.
  const exact = (n: number) => Math.round(n * 1e6) / 1e6;
  switch (op) {
    case '+':
      return { v: exact(x + y), t: bothInt ? 'int' : 'num' };
    case '-':
      return { v: exact(x - y), t: bothInt ? 'int' : 'num' };
    case '*':
      return { v: exact(x * y), t: bothInt ? 'int' : 'num' };
    case '%':
      if (y === 0) return eng === 'pg' ? fail('division by zero') : NULLC;
      return { v: x % y, t: 'int' };
    case '/': {
      if (y === 0) {
        // Postgres raises; MySQL and SQLite both return NULL for x/0.
        if (eng === 'pg') fail('division by zero');
        return NULLC;
      }
      // Integer division truncates in Postgres and SQLite. MySQL's / is always decimal.
      if (bothInt && eng !== 'mysql') return { v: Math.trunc(x / y), t: 'int' };
      return { v: Math.round((x / y) * 1e6) / 1e6, t: 'num' };
    }
  }
  return fail(`unknown operator ${op}`);
}

type Tri = true | false | null;

function compareCells(a: Cell, b: Cell, eng: Eng): number | null {
  if (a.v === null || b.v === null) return null;
  if (eng === 'sqlite') {
    // Storage-class ordering, softened by column affinity: a numeric-affinity column
    // compared against a text literal converts the literal first.
    let av = a.v;
    let bv = b.v;
    if (a.aff === 'num' && typeof bv === 'string') {
      const n = fullNum(bv);
      if (n !== null) bv = n;
    }
    if (b.aff === 'num' && typeof av === 'string') {
      const n = fullNum(av);
      if (n !== null) av = n;
    }
    const an = typeof av === 'number' || typeof av === 'boolean';
    const bn = typeof bv === 'number' || typeof bv === 'boolean';
    if (an !== bn) return an ? -1 : 1; // INTEGER/REAL sort before TEXT, always
    if (an) {
      const x = Number(av);
      const y = Number(bv);
      return x < y ? -1 : x > y ? 1 : 0;
    }
    return String(av) < String(bv) ? -1 : String(av) > String(bv) ? 1 : 0;
  }
  if (typeof a.v === 'string' && typeof b.v === 'string') {
    // MySQL's default collation (utf8mb4_0900_ai_ci) is case- and accent-insensitive;
    // Postgres compares text byte-wise under the C/ICU collation you chose.
    const x = eng === 'mysql' ? a.v.toLowerCase() : a.v;
    const y = eng === 'mysql' ? b.v.toLowerCase() : b.v;
    return x < y ? -1 : x > y ? 1 : 0;
  }
  if (typeof a.v === 'boolean' || typeof b.v === 'boolean') {
    if (eng === 'pg' && typeof a.v !== typeof b.v)
      fail(`operator does not exist: ${PGNAME[a.t]} = ${PGNAME[b.t]}`);
    const x = a.v ? 1 : 0;
    const y = b.v ? 1 : 0;
    return x < y ? -1 : x > y ? 1 : 0;
  }
  if (typeof a.v === 'string' && typeof b.v === 'number') {
    if (eng === 'pg' && a.t !== 'unknown') fail(`operator does not exist: ${PGNAME[a.t]} = ${PGNAME[b.t]}`);
    const x = eng === 'pg' ? fullNum(a.v) : prefixNum(a.v);
    if (x === null) fail(`invalid input syntax for type ${PGNAME[b.t] === 'unknown' ? 'integer' : PGNAME[b.t]}: "${a.v}"`);
    return x < b.v ? -1 : x > b.v ? 1 : 0;
  }
  if (typeof a.v === 'number' && typeof b.v === 'string') {
    const r = compareCells(b, a, eng);
    return r === null ? null : -r;
  }
  const x = Number(a.v);
  const y = Number(b.v);
  return x < y ? -1 : x > y ? 1 : 0;
}

function likeRe(pat: string, eng: Eng) {
  let re = '';
  for (const ch of pat) {
    if (ch === '%') re += '[\\s\\S]*';
    else if (ch === '_') re += '[\\s\\S]';
    else re += ch.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
  }
  // Postgres LIKE is case-sensitive; MySQL's default collation and SQLite's LIKE are not.
  return new RegExp(`^${re}$`, eng === 'pg' ? '' : 'i');
}

type Ctx = { row: Row; cols: ColDef[]; eng: Eng; extra?: Record<string, Cell> };

function colCell(name: string, ctx: Ctx): Cell {
  const def = ctx.cols.find((c) => c.name.toLowerCase() === name.toLowerCase());
  if (!def) {
    if (ctx.extra && name.toLowerCase() in ctx.extra) return ctx.extra[name.toLowerCase()];
    if (ctx.eng === 'mysql') fail(`ERROR 1054 (42S22): Unknown column '${name}' in 'where clause'`);
    fail(`column "${name}" does not exist`);
  }
  const v = ctx.row[def.name] ?? null;
  const t = ctx.eng === 'sqlite' ? storageTy(v) : v === null ? 'null' : declaredTy(def.type);
  const aff: 'num' | 'text' =
    def.type.base === 'INTEGER' || def.type.base === 'NUMERIC' ? 'num' : 'text';
  return { v, t, aff };
}

function evalE(e: Expr, ctx: Ctx): Cell {
  const eng = ctx.eng;
  switch (e.k) {
    case 'num':
      return { v: e.v, t: Number.isInteger(e.v) ? 'int' : 'num' };
    case 'str':
      return { v: e.v, t: eng === 'pg' ? 'unknown' : 'text' };
    case 'null':
      return NULLC;
    case 'bool':
      return { v: e.v, t: 'bool' };
    case 'typed':
      return { v: e.v, t: e.ty === 'DATE' ? 'date' : e.ty === 'NUMERIC' ? 'num' : 'int' };
    case 'star':
      return fail('* is only valid in the select list');
    case 'col':
      return colCell(e.name, ctx);
    case 'neg': {
      const c = evalE(e.e, ctx);
      return c.v === null ? NULLC : { v: -numeric(c, eng, '-', { v: 0, t: 'int' }), t: c.t === 'int' ? 'int' : 'num' };
    }
    case 'not': {
      const c = evalE(e.e, ctx);
      const tv = truth(c, eng);
      return tv === null ? NULLC : { v: !tv, t: 'bool' };
    }
    case 'isnull': {
      const c = evalE(e.e, ctx);
      return { v: e.neg ? c.v !== null : c.v === null, t: 'bool' };
    }
    case 'in': {
      const c = evalE(e.e, ctx);
      if (c.v === null) return NULLC;
      let sawNull = false;
      for (const item of e.list) {
        const o = evalE(item, ctx);
        if (o.v === null) {
          sawNull = true;
          continue;
        }
        if (compareCells(c, o, eng) === 0) return { v: !e.neg, t: 'bool' };
      }
      // IN is OR of equalities, so an unmatched list containing NULL is UNKNOWN, not FALSE.
      return sawNull ? NULLC : { v: e.neg, t: 'bool' };
    }
    case 'between': {
      const c = evalE(e.e, ctx);
      const lo = compareCells(c, evalE(e.lo, ctx), eng);
      const hi = compareCells(c, evalE(e.hi, ctx), eng);
      if (lo === null || hi === null) return NULLC;
      const inside = lo >= 0 && hi <= 0;
      return { v: e.neg ? !inside : inside, t: 'bool' };
    }
    case 'like': {
      const c = evalE(e.e, ctx);
      const pat = evalE(e.pat, ctx);
      if (c.v === null || pat.v === null) return NULLC;
      const m = likeRe(String(pat.v), eng).test(String(c.v));
      return { v: e.neg ? !m : m, t: 'bool' };
    }
    case 'cast':
      return castCell(evalE(e.e, ctx), e.to, eng);
    case 'call':
      return callFn(e, ctx);
    case 'bin': {
      if (e.op === 'AND' || e.op === 'OR') {
        const a = truth(evalE(e.l, ctx), eng);
        const b = truth(evalE(e.r, ctx), eng);
        const r = e.op === 'AND' ? and3(a, b) : or3(a, b);
        return r === null ? NULLC : { v: r, t: 'bool' };
      }
      const l = evalE(e.l, ctx);
      const r = evalE(e.r, ctx);
      if (e.op === '||') {
        if (l.v === null || r.v === null) return NULLC;
        return { v: String(fmtVal(l.v, eng)) + String(fmtVal(r.v, eng)), t: 'text' };
      }
      if ('+-*/%'.includes(e.op) && e.op.length === 1) return arith(e.op, l, r, eng);
      const c = compareCells(l, r, eng);
      if (c === null) return NULLC;
      switch (e.op) {
        case '=':
          return { v: c === 0, t: 'bool' };
        case '<>':
          return { v: c !== 0, t: 'bool' };
        case '<':
          return { v: c < 0, t: 'bool' };
        case '<=':
          return { v: c <= 0, t: 'bool' };
        case '>':
          return { v: c > 0, t: 'bool' };
        case '>=':
          return { v: c >= 0, t: 'bool' };
      }
      return fail(`unknown operator ${e.op}`);
    }
  }
}

function truth(c: Cell, eng: Eng): Tri {
  if (c.v === null) return null;
  if (typeof c.v === 'boolean') return c.v;
  if (eng === 'pg' && c.t !== 'unknown') fail(`argument of AND must be type boolean, not type ${PGNAME[c.t]}`);
  if (typeof c.v === 'number') return c.v !== 0;
  const s = String(c.v).toLowerCase();
  if (eng === 'pg') {
    if (s === 'true' || s === 't') return true;
    if (s === 'false' || s === 'f') return false;
    fail(`invalid input syntax for type boolean: "${c.v}"`);
  }
  return prefixNum(String(c.v)) !== 0;
}
const and3 = (a: Tri, b: Tri): Tri => (a === false || b === false ? false : a === null || b === null ? null : true);
const or3 = (a: Tri, b: Tri): Tri => (a === true || b === true ? true : a === null || b === null ? null : false);

function castCell(c: Cell, to: ColType, eng: Eng): Cell {
  if (c.v === null) return NULLC;
  switch (to.base) {
    case 'INTEGER': {
      if (typeof c.v === 'number') return { v: Math.round(c.v), t: 'int' };
      if (typeof c.v === 'boolean') return { v: c.v ? 1 : 0, t: 'int' };
      const n = eng === 'pg' ? fullNum(c.v) : prefixNum(c.v);
      if (n === null) fail(`invalid input syntax for type integer: "${c.v}"`);
      return { v: Math.round(n), t: 'int' };
    }
    case 'NUMERIC': {
      if (typeof c.v === 'number') return { v: c.v, t: 'num' };
      const n = eng === 'pg' ? fullNum(String(c.v)) : prefixNum(String(c.v));
      if (n === null) fail(`invalid input syntax for type numeric: "${c.v}"`);
      return { v: n, t: 'num' };
    }
    case 'BOOLEAN':
      return { v: truth(c, eng) === true, t: 'bool' };
    case 'DATE': {
      const s = String(c.v);
      if (eng !== 'sqlite' && !/^\d{4}-\d{2}-\d{2}$/.test(s)) fail(`invalid input syntax for type date: "${s}"`);
      return { v: s, t: 'date' };
    }
    default: {
      let s = fmtVal(c.v, eng);
      if (to.len && s.length > to.len) s = s.slice(0, to.len); // CAST truncates; assignment does not
      return { v: s, t: 'text' };
    }
  }
}

function callFn(e: Extract<Expr, { k: 'call' }>, ctx: Ctx): Cell {
  const eng = ctx.eng;
  const a = e.args.map((x) => evalE(x, ctx));
  switch (e.fn) {
    case 'COALESCE': {
      for (const c of a) if (c.v !== null) return c;
      return NULLC;
    }
    case 'NULLIF': {
      if (a.length !== 2) fail('NULLIF takes exactly two arguments');
      const c = compareCells(a[0], a[1], eng);
      return c === 0 ? NULLC : a[0];
    }
    case 'UPPER':
      return a[0].v === null ? NULLC : { v: fmtVal(a[0].v, eng).toUpperCase(), t: 'text' };
    case 'LOWER':
      return a[0].v === null ? NULLC : { v: fmtVal(a[0].v, eng).toLowerCase(), t: 'text' };
    case 'LENGTH':
      return a[0].v === null ? NULLC : { v: fmtVal(a[0].v, eng).length, t: 'int' };
    case 'ABS':
      return a[0].v === null ? NULLC : { v: Math.abs(numeric(a[0], eng, '-', a[0])), t: a[0].t === 'int' ? 'int' : 'num' };
    case 'ROUND': {
      if (a[0].v === null) return NULLC;
      const d = a.length > 1 ? Number(a[1].v) : 0;
      const f = 10 ** d;
      return { v: Math.round(numeric(a[0], eng, '+', a[0]) * f) / f, t: d > 0 ? 'num' : 'int' };
    }
    default:
      if (eng === 'mysql') fail(`ERROR 1305 (42000): FUNCTION ${e.fn.toLowerCase()} does not exist`);
      return fail(`function ${e.fn.toLowerCase()}() does not exist`);
  }
}

function fmtVal(v: Val, eng: Eng): string {
  if (v === null) return '';
  if (typeof v === 'boolean') return eng === 'pg' ? (v ? 'true' : 'false') : v ? '1' : '0';
  return String(v);
}

/* ================================================ storing a value in a column */

function coerceForColumn(c: Cell, col: ColDef, eng: Eng, table: string): Val {
  if (c.v === null) return null;
  const b = col.type.base;
  if (eng === 'sqlite') {
    // Affinity, not enforcement: convert when the text is losslessly numeric, otherwise
    // store the value exactly as given, whatever the column says.
    if (b === 'INTEGER' || b === 'NUMERIC') {
      if (typeof c.v === 'string') {
        const n = fullNum(c.v);
        return n === null ? c.v : b === 'INTEGER' && Number.isInteger(n) ? n : n;
      }
      if (typeof c.v === 'boolean') return c.v ? 1 : 0;
      return c.v;
    }
    if (b === 'BOOLEAN') return typeof c.v === 'boolean' ? (c.v ? 1 : 0) : c.v;
    if (typeof c.v === 'boolean') return c.v ? 1 : 0;
    return c.v; // VARCHAR(n) length is parsed and then ignored
  }
  if (b === 'INTEGER' || b === 'NUMERIC') {
    if (typeof c.v === 'boolean') {
      if (eng === 'pg') fail(`column "${col.name}" is of type ${PGNAME[declaredTy(col.type)]} but expression is of type boolean`);
      return c.v ? 1 : 0;
    }
    let n: number;
    if (typeof c.v === 'number') n = c.v;
    else if (eng === 'pg') {
      if (c.t !== 'unknown') fail(`column "${col.name}" is of type ${b.toLowerCase()} but expression is of type ${PGNAME[c.t]}`);
      const f = fullNum(c.v);
      if (f === null) fail(`invalid input syntax for type ${b.toLowerCase()}: "${c.v}"`);
      n = f;
    } else {
      const f = fullNum(c.v);
      if (f === null) fail(`ERROR 1366 (HY000): Incorrect ${b === 'INTEGER' ? 'integer' : 'decimal'} value: '${c.v}' for column '${col.name}' at row 1`);
      n = f;
    }
    return b === 'INTEGER' ? Math.round(n) : n; // assignment to an integer column rounds
  }
  if (b === 'BOOLEAN') {
    if (typeof c.v === 'boolean') return eng === 'pg' ? c.v : c.v ? 1 : 0;
    if (eng === 'pg') fail(`column "${col.name}" is of type boolean but expression is of type ${PGNAME[c.t]}`);
    return prefixNum(String(c.v)) !== 0 ? 1 : 0;
  }
  if (b === 'DATE') {
    const s = fmtVal(c.v, eng);
    if (!/^\d{4}-\d{2}-\d{2}$/.test(s)) {
      if (eng === 'pg') fail(`invalid input syntax for type date: "${s}"`);
      fail(`ERROR 1292 (22007): Incorrect date value: '${s}' for column '${col.name}' at row 1`);
    }
    return s;
  }
  const s = fmtVal(c.v, eng);
  if (col.type.len && s.length > col.type.len) {
    if (eng === 'pg') fail(`value too long for type character varying(${col.type.len})`);
    fail(`ERROR 1406 (22001): Data too long for column '${col.name}' at row 1`);
  }
  void table;
  return s;
}

function enforce(tbl: TableDef, row: Row, eng: Eng, selfIndex: number) {
  for (const col of tbl.cols) {
    const v = row[col.name] ?? null;
    if (col.notNull && v === null) {
      if (eng === 'pg') fail(`null value in column "${col.name}" of relation "${tbl.name}" violates not-null constraint`);
      if (eng === 'mysql') fail(`ERROR 1048 (23000): Column '${col.name}' cannot be null`);
      fail(`NOT NULL constraint failed: ${tbl.name}.${col.name}`);
    }
    if ((col.pk || col.unique) && v !== null) {
      const dup = tbl.rows.some((r, i) => i !== selfIndex && r[col.name] === v);
      if (dup) {
        if (eng === 'pg')
          fail(`duplicate key value violates unique constraint "${tbl.name}_${col.pk ? 'pkey' : `${col.name}_key`}"`);
        if (eng === 'mysql')
          fail(`ERROR 1062 (23000): Duplicate entry '${v}' for key '${tbl.name}.${col.pk ? 'PRIMARY' : col.name}'`);
        fail(`UNIQUE constraint failed: ${tbl.name}.${col.name}`);
      }
    }
    if (col.check) {
      // A CHECK passes on UNKNOWN — only an outright FALSE rejects the row.
      const t = truth(evalE(col.check, { row, cols: tbl.cols, eng }), eng);
      if (t === false) {
        if (eng === 'pg') fail(`new row for relation "${tbl.name}" violates check constraint "${tbl.name}_${col.name}_check"`);
        if (eng === 'mysql') fail(`ERROR 3819 (HY000): Check constraint '${tbl.name}_chk_1' is violated.`);
        fail(`CHECK constraint failed: ${tbl.name}`);
      }
    }
  }
}

/* ================================================================= the dataset */

function col(
  name: string,
  type: ColType,
  o: Partial<Omit<ColDef, 'name' | 'type'>> = {},
): ColDef {
  return { name, type, notNull: false, pk: false, unique: false, ...o };
}

const CUSTOMER_ROWS: Row[] = [
  { id: 1, name: 'Ines Okafor', city: 'Seattle', credit: 5000, signed_on: '2021-03-04', active: true },
  { id: 2, name: 'Bo Lindqvist', city: 'Tacoma', credit: null, signed_on: '2022-07-19', active: true },
  { id: 3, name: 'Priya Raman', city: null, credit: 12000, signed_on: '2020-11-30', active: false },
  { id: 4, name: 'Diego Salas', city: 'Seattle', credit: 0, signed_on: '2023-01-08', active: true },
  { id: 5, name: 'Nadia Haddad', city: 'Everett', credit: 2500, signed_on: '2023-09-22', active: false },
  { id: 6, name: 'Tom Becker', city: null, credit: null, signed_on: '2024-02-14', active: true },
  { id: 7, name: 'Ayla Demir', city: 'Tacoma', credit: 7500, signed_on: '2022-05-01', active: true },
  { id: 8, name: 'Kofi Mensah', city: 'Olympia', credit: 300, signed_on: '2024-04-27', active: true },
];

const STATUSES = ['pending', 'paid', 'shipped', 'cancelled'];
const SKUS = ['DSK-100', 'DSK-220', 'KBD-01', 'MON-27', 'CBL-USB', 'HUB-4P'];

function seedDb(): DB {
  const rng = makeRng(19740601); // Codd's relational model paper, near enough
  // A balanced status column, shuffled deterministically: a random draw per row clumps
  // badly at this sample size and leaves some example predicates matching nothing.
  const statusBag = Array.from({ length: 18 }, (_, i) => STATUSES[i % 4]);
  for (let i = statusBag.length - 1; i > 0; i--) {
    const j = Math.floor(rng() * (i + 1));
    [statusBag[i], statusBag[j]] = [statusBag[j], statusBag[i]];
  }
  const orders: Row[] = [];
  for (let i = 0; i < 18; i++) {
    const cust = 1 + Math.floor(rng() * 8);
    const day = 1 + Math.floor(rng() * 28);
    const mon = 1 + Math.floor(rng() * 5);
    const status = statusBag[i];
    const total = Math.round((18 + rng() * 900) * 100) / 100;
    const placed = `2024-${String(mon).padStart(2, '0')}-${String(day).padStart(2, '0')}`;
    orders.push({
      id: 100 + i,
      customer_id: cust,
      placed_on: placed,
      status,
      total,
      shipped_on: status === 'shipped' ? `2024-${String(mon).padStart(2, '0')}-${String(Math.min(28, day + 3)).padStart(2, '0')}` : null,
    });
  }
  const items: Row[] = [];
  for (let i = 0; i < 16; i++) {
    items.push({
      id: 500 + i,
      order_id: 100 + Math.floor(rng() * 18),
      sku: SKUS[Math.floor(rng() * SKUS.length)],
      qty: 1 + Math.floor(rng() * 4),
      price: Math.round((9 + rng() * 380) * 100) / 100,
    });
  }
  return {
    customers: {
      name: 'customers',
      cols: [
        col('id', { base: 'INTEGER' }, { pk: true, notNull: true }),
        col('name', { base: 'TEXT' }, { notNull: true }),
        col('city', { base: 'VARCHAR', len: 12 }),
        col('credit', { base: 'NUMERIC' }),
        col('signed_on', { base: 'DATE' }, { notNull: true }),
        col('active', { base: 'BOOLEAN' }, { notNull: true, def: { k: 'bool', v: true } }),
      ],
      rows: CUSTOMER_ROWS.map((r) => ({ ...r })),
    },
    orders: {
      name: 'orders',
      cols: [
        col('id', { base: 'INTEGER' }, { pk: true, notNull: true }),
        col('customer_id', { base: 'INTEGER' }, { notNull: true }),
        col('placed_on', { base: 'DATE' }, { notNull: true }),
        col(
          'status',
          { base: 'TEXT' },
          {
            notNull: true,
            def: { k: 'str', v: 'pending' },
            check: {
              k: 'in',
              e: { k: 'col', name: 'status' },
              list: STATUSES.map((s) => ({ k: 'str', v: s }) as Expr),
              neg: false,
            },
          },
        ),
        col('total', { base: 'NUMERIC' }, { notNull: true }),
        col('shipped_on', { base: 'DATE' }),
      ],
      rows: orders,
    },
    items: {
      name: 'items',
      cols: [
        col('id', { base: 'INTEGER' }, { pk: true, notNull: true }),
        col('order_id', { base: 'INTEGER' }, { notNull: true }),
        col('sku', { base: 'VARCHAR', len: 8 }, { notNull: true }),
        col('qty', { base: 'INTEGER' }, { notNull: true, def: { k: 'num', v: 1 } }),
        col('price', { base: 'NUMERIC' }, { notNull: true }),
      ],
      rows: items,
    },
  };
}

function cloneDb(db: DB): DB {
  const out: DB = {};
  for (const [k, t] of Object.entries(db)) out[k] = { ...t, rows: t.rows.map((r) => ({ ...r })) };
  return out;
}

/* ================================================================ the pipeline */

type StageRow = { cells: Cell[]; kept: boolean; mark?: string };
type Stage = {
  name: string;
  written: string;
  present: boolean;
  note: string;
  cols: string[];
  rows: StageRow[];
  inN: number;
  outN: number;
};

type RunOut = {
  db: DB;
  stages: Stage[];
  tag: string;
  error: string | null;
  head: string;
  body: string;
  kind: string;
  unknowns: number;
  scanned: number;
  returned: number;
  log: { sql: string; tag: string }[];
};

function outName(it: Item, eng: Eng): string {
  if (it.alias) return it.alias;
  if (it.e.k === 'col') return it.e.name;
  if (it.e.k === 'cast') return it.e.e.k === 'col' ? it.e.e.name : eng === 'pg' ? typeName(it.e.to).toLowerCase() : unparse(it.e);
  if (eng === 'pg') return it.e.k === 'call' ? it.e.fn.toLowerCase() : '?column?';
  return unparse(it.e);
}

function expandItems(items: Item[], tbl: TableDef): Item[] {
  const out: Item[] = [];
  for (const it of items) {
    if (it.e.k === 'star') for (const c of tbl.cols) out.push({ e: { k: 'col', name: c.name } });
    else out.push(it);
  }
  return out;
}

function baseCells(r: Row, tbl: TableDef, eng: Eng): Cell[] {
  return tbl.cols.map((c) => colCell(c.name, { row: r, cols: tbl.cols, eng }));
}

function runSelect(st: Extract<Stmt, { k: 'select' }>, db: DB, eng: Eng): RunOut {
  const tbl = db[st.from.toLowerCase()];
  if (!tbl) fail(`relation "${st.from}" does not exist`);
  const cols = tbl.cols.map((c) => c.name);
  const stages: Stage[] = [];

  /* 1 — FROM */
  const from: Row[] = tbl.rows;
  stages.push({
    name: 'FROM',
    written: 'written 2nd',
    present: true,
    note: `Names the relation. ${from.length} rows enter the pipeline; nothing has been filtered or projected yet.`,
    cols,
    rows: from.map((r) => ({ cells: baseCells(r, tbl, eng), kept: true })),
    inN: from.length,
    outN: from.length,
  });

  const items = expandItems(st.items, tbl);
  const names = items.map((it) => outName(it, eng));

  /* 2 — WHERE */
  let unknowns = 0;
  let falses = 0;
  const kept: Row[] = [];
  const whereRows: StageRow[] = from.map((r) => {
    if (!st.where) {
      kept.push(r);
      return { cells: baseCells(r, tbl, eng), kept: true, mark: '—' };
    }
    // SQLite documents a deliberate deviation: result-set aliases are visible in WHERE.
    let extra: Record<string, Cell> | undefined;
    if (eng === 'sqlite') {
      extra = {};
      items.forEach((it, i) => {
        try {
          extra![names[i].toLowerCase()] = evalE(it.e, { row: r, cols: tbl.cols, eng });
        } catch {
          /* an alias that cannot be evaluated simply stays invisible */
        }
      });
    }
    const t = truth(evalE(st.where, { row: r, cols: tbl.cols, eng, extra }), eng);
    if (t === true) kept.push(r);
    else if (t === null) unknowns++;
    else falses++;
    return { cells: baseCells(r, tbl, eng), kept: t === true, mark: t === null ? 'UNKNOWN' : t ? 'TRUE' : 'FALSE' };
  });
  stages.push({
    name: 'WHERE',
    written: 'written 3rd',
    present: !!st.where,
    note: st.where
      ? `Each row is tested independently against ${unparse(st.where)}. Only rows whose predicate is strictly TRUE survive — ${falses} were FALSE and ${unknowns} were UNKNOWN, and both are discarded alike. Output-list aliases do not exist yet at this point.`
      : 'No WHERE clause: every row passes through untouched.',
    cols: [...cols, 'predicate'],
    rows: whereRows,
    inN: from.length,
    outN: kept.length,
  });

  /* 3 — SELECT (projection) */
  type Proj = { cells: Cell[]; src: Row };
  const projected: Proj[] = kept.map((r) => ({
    cells: items.map((it) => evalE(it.e, { row: r, cols: tbl.cols, eng })),
    src: r,
  }));
  stages.push({
    name: 'SELECT',
    written: 'written 1st',
    present: true,
    note: `The output list is evaluated now, per surviving row. This is where aliases (${names.join(', ')}) come into existence — which is exactly why WHERE could not use them and ORDER BY can.`,
    cols: names,
    rows: projected.map((p) => ({ cells: p.cells, kept: true })),
    inN: kept.length,
    outN: projected.length,
  });

  /* 4 — DISTINCT */
  const seen = new Set<string>();
  const distinctRows: StageRow[] = [];
  const afterDistinct: Proj[] = [];
  for (const pr of projected) {
    if (!st.distinct) {
      distinctRows.push({ cells: pr.cells, kept: true });
      afterDistinct.push(pr);
      continue;
    }
    // DISTINCT groups by "is not distinct from", so two NULLs collapse into one output
    // row here even though NULL = NULL is UNKNOWN two stages earlier.
    const key = pr.cells.map((c) => (c.v === null ? ' NULL' : `${typeof c.v}:${String(c.v)}`)).join('');
    const dup = seen.has(key);
    seen.add(key);
    distinctRows.push({ cells: pr.cells, kept: !dup, mark: dup ? 'dup' : undefined });
    if (!dup) afterDistinct.push(pr);
  }
  stages.push({
    name: 'DISTINCT',
    written: 'written 1st',
    present: st.distinct,
    note: st.distinct
      ? `Duplicate output rows are folded. DISTINCT compares with "is not distinct from", so two NULLs count as the same value here even though NULL = NULL is UNKNOWN in WHERE.`
      : 'No DISTINCT: duplicate output rows are kept.',
    cols: names,
    rows: distinctRows,
    inN: projected.length,
    outN: afterDistinct.length,
  });

  /* 5 — ORDER BY */
  let ordered: Proj[] = afterDistinct;
  if (st.order) {
    // ORDER BY runs after projection, so a key may be an output alias or a 1-based
    // ordinal; anything else is resolved against the base row.
    const idxOf = (o: OrderKey): number | null => {
      if (o.e.k === 'col') {
        const want = o.e.name.toLowerCase();
        const i = names.findIndex((n) => n.toLowerCase() === want);
        if (i >= 0) return i;
      }
      if (o.e.k === 'num') {
        const i = Math.round(o.e.v) - 1;
        if (i >= 0 && i < names.length) return i;
      }
      return null;
    };
    const keyed = st.order.map((o) => ({ o, i: idxOf(o) }));
    ordered = afterDistinct
      .map((pr, n) => ({ pr, n }))
      .sort((A, B) => {
        for (const { o, i } of keyed) {
          const a = i !== null ? A.pr.cells[i] : evalE(o.e, { row: A.pr.src, cols: tbl.cols, eng });
          const b = i !== null ? B.pr.cells[i] : evalE(o.e, { row: B.pr.src, cols: tbl.cols, eng });
          if (a.v === null && b.v === null) continue;
          if (a.v === null || b.v === null) {
            if (o.nulls) {
              // An explicit NULLS FIRST/LAST is absolute: ASC/DESC does not flip it.
              const aFirst = o.nulls === 'first';
              return a.v === null ? (aFirst ? -1 : 1) : aFirst ? 1 : -1;
            }
            // Postgres treats NULL as larger than every value; MySQL and SQLite as smaller.
            const nullIsBig = eng === 'pg';
            const c = a.v === null ? (nullIsBig ? 1 : -1) : nullIsBig ? -1 : 1;
            return o.desc ? -c : c;
          }
          const c = compareCells(a, b, eng) ?? 0;
          if (c !== 0) return o.desc ? -c : c;
        }
        return A.n - B.n; // a stable sort: equal keys keep the order they arrived in
      })
      .map((x) => x.pr);
  }
  stages.push({
    name: 'ORDER BY',
    written: 'written 4th',
    present: !!st.order,
    note: st.order
      ? `Sorting happens after projection, so an ORDER BY key may be an output alias or a 1-based ordinal. Default NULL placement is engine-specific: ${eng === 'pg' ? 'Postgres treats NULL as larger than every value, so ASC puts NULLs last and DESC puts them first' : 'MySQL and SQLite treat NULL as smaller than every value, so ASC puts NULLs first'}.`
      : 'No ORDER BY: the row order is whatever the scan produced, and the engine owes you nothing — adding an index or a parallel worker can silently change it.',
    cols: names,
    rows: ordered.map((pr) => ({ cells: pr.cells, kept: true })),
    inN: afterDistinct.length,
    outN: ordered.length,
  });

  /* 6 — LIMIT / OFFSET */
  const off = st.offset ?? 0;
  const lim = st.limit ?? Infinity;
  const limitRows: StageRow[] = ordered.map((pr, i) => ({
    cells: pr.cells,
    kept: i >= off && i < off + lim,
    mark: i < off ? 'skipped' : i >= off + lim ? 'cut' : undefined,
  }));
  const final = limitRows.filter((r) => r.kept);
  stages.push({
    name: 'LIMIT',
    written: 'written 5th',
    present: st.limit !== undefined || st.offset !== undefined,
    note:
      st.limit !== undefined || st.offset !== undefined
        ? `The last stage, and the only one that can be cheap: the engine may stop as soon as ${off + (st.limit ?? 0)} rows have been produced — but only if the sort above it can stream, which a full sort cannot.`
        : 'No LIMIT: every row of the sorted result is returned.',
    cols: names,
    rows: limitRows,
    inN: ordered.length,
    outN: final.length,
  });

  return {
    db,
    stages,
    tag: eng === 'pg' ? `SELECT ${final.length}` : eng === 'mysql' ? `${final.length} row${final.length === 1 ? '' : 's'} in set` : `${final.length} rows`,
    error: null,
    head: `${final.length} row${final.length === 1 ? '' : 's'} returned from ${from.length} scanned.`,
    body: stages[0].note,
    kind: 'select',
    unknowns,
    scanned: from.length,
    returned: final.length,
    log: [],
  };
}

function runDml(st: Exclude<Stmt, { k: 'select' }>, db: DB, eng: Eng): RunOut {
  const stages: Stage[] = [];
  const mk = (name: string, note: string, cols: string[], rows: StageRow[], inN: number, outN: number, present = true): Stage => ({
    name,
    written: '',
    present,
    note,
    cols,
    rows,
    inN,
    outN,
  });

  if (st.k === 'create') {
    if (db[st.table.toLowerCase()]) fail(`relation "${st.table}" already exists`);
    db[st.table.toLowerCase()] = { name: st.table, cols: st.cols, rows: [] };
    stages.push(
      mk(
        'CREATE',
        `The catalog now holds ${st.cols.length} column definitions for ${st.table}. Nothing is stored yet; what a CREATE TABLE buys you is every later write being checked against these declarations.`,
        ['column', 'type', 'nullable', 'default', 'constraints'],
        st.cols.map((c) => ({
          kept: true,
          cells: [
            { v: c.name, t: 'text' },
            { v: typeName(c.type), t: 'text' },
            { v: c.notNull ? 'NOT NULL' : 'nullable', t: 'text' },
            { v: c.def ? unparse(c.def) : null, t: c.def ? 'text' : 'null' },
            { v: [c.pk ? 'PRIMARY KEY' : '', c.unique ? 'UNIQUE' : '', c.check ? `CHECK (${unparse(c.check)})` : ''].filter(Boolean).join(', ') || null, t: 'text' },
          ] as Cell[],
        })),
        0,
        0,
      ),
    );
    return done(db, stages, 'CREATE TABLE', `Table ${st.table} created with ${st.cols.length} columns.`, 'ddl');
  }

  if (st.k === 'drop') {
    if (!db[st.table.toLowerCase()]) fail(`table "${st.table}" does not exist`);
    delete db[st.table.toLowerCase()];
    stages.push(mk('DROP', `${st.table} and every row in it are gone. DDL in Postgres is transactional, so this would roll back; in MySQL 8 a DDL statement commits the open transaction implicitly.`, ['result'], [], 0, 0));
    return done(db, stages, 'DROP TABLE', `Table ${st.table} dropped.`, 'ddl');
  }

  const tbl = db[(st as { table: string }).table.toLowerCase()];
  if (!tbl) fail(`relation "${(st as { table: string }).table}" does not exist`);

  if (st.k === 'insert') {
    const targets = st.cols ?? tbl.cols.map((c) => c.name);
    for (const t of targets)
      if (!tbl.cols.some((c) => c.name.toLowerCase() === t.toLowerCase())) fail(`column "${t}" of relation "${tbl.name}" does not exist`);
    const supplied: StageRow[] = [];
    const defaulted: StageRow[] = [];
    const added: Row[] = [];
    for (const tuple of st.tuples) {
      if (tuple.length !== targets.length) fail(`INSERT has more expressions than target columns`);
      const row: Row = {};
      supplied.push({ kept: true, cells: tuple.map((e) => evalE(e, { row: {}, cols: [], eng })) });
      for (const c of tbl.cols) {
        const i = targets.findIndex((t) => t.toLowerCase() === c.name.toLowerCase());
        if (i >= 0) row[c.name] = coerceForColumn(evalE(tuple[i], { row: {}, cols: [], eng }), c, eng, tbl.name);
        else if (c.def) row[c.name] = coerceForColumn(evalE(c.def, { row: {}, cols: [], eng }), c, eng, tbl.name);
        else row[c.name] = null;
      }
      defaulted.push({ kept: true, cells: baseCells(row, tbl, eng) });
      tbl.rows.push(row);
      enforce(tbl, row, eng, tbl.rows.length - 1);
      added.push(row);
    }
    stages.push(
      mk('VALUES', `The row constructor is evaluated first, with no table context at all — nothing here can reference a column.`, targets, supplied, st.tuples.length, st.tuples.length),
      mk(
        'DEFAULTS + COERCE',
        `Every column the INSERT did not name is filled from its DEFAULT (or NULL), then each value is coerced to the declared type. ${eng === 'sqlite' ? 'SQLite applies affinity here rather than enforcement: a value that will not convert is simply stored as-is.' : 'A value that does not convert is an error, not a silent cast.'}`,
        tbl.cols.map((c) => c.name),
        defaulted,
        st.tuples.length,
        st.tuples.length,
      ),
      mk('CONSTRAINTS', `NOT NULL, then PRIMARY KEY/UNIQUE, then CHECK. A CHECK that evaluates to UNKNOWN passes — only FALSE rejects the row.`, tbl.cols.map((c) => c.name), defaulted, st.tuples.length, st.tuples.length),
    );
    pushReturning(stages, st.returning, added, tbl, eng);
    return done(
      db,
      stages,
      eng === 'pg' ? `INSERT 0 ${added.length}` : eng === 'mysql' ? `Query OK, ${added.length} row${added.length === 1 ? '' : 's'} affected` : `${added.length} rows`,
      `${added.length} row${added.length === 1 ? '' : 's'} inserted into ${tbl.name}.`,
      'dml',
    );
  }

  if (st.k === 'update') {
    const before: StageRow[] = [];
    const touched: Row[] = [];
    tbl.rows.forEach((r, i) => {
      const t = st.where ? truth(evalE(st.where, { row: r, cols: tbl.cols, eng }), eng) : true;
      before.push({ cells: baseCells(r, tbl, eng), kept: t === true, mark: t === null ? 'UNKNOWN' : t ? 'TRUE' : 'FALSE' });
      if (t !== true) return;
      // All SET expressions read the OLD row: an UPDATE is not a sequence of assignments.
      const next: Row = { ...r };
      for (const s of st.sets) {
        const c = tbl.cols.find((x) => x.name.toLowerCase() === s.col.toLowerCase());
        if (!c) fail(`column "${s.col}" of relation "${tbl.name}" does not exist`);
        next[c.name] = coerceForColumn(evalE(s.e, { row: r, cols: tbl.cols, eng }), c, eng, tbl.name);
      }
      tbl.rows[i] = next;
      enforce(tbl, next, eng, i);
      touched.push(next);
    });
    stages.push(
      mk('FROM (target)', `${tbl.rows.length} rows are candidates. An UPDATE with no WHERE updates all of them, which is the single most expensive typo in this language.`, tbl.cols.map((c) => c.name), before.map((r) => ({ cells: r.cells, kept: true })), tbl.rows.length, tbl.rows.length),
      mk('WHERE', st.where ? `Rows whose predicate is TRUE are updated; FALSE and UNKNOWN are both left alone.` : 'No WHERE: every row is updated.', [...tbl.cols.map((c) => c.name), 'predicate'], before, tbl.rows.length, touched.length, !!st.where),
      mk('SET', eng === 'mysql' ? `MySQL evaluates single-table UPDATE assignments left to right, so a later SET sees what an earlier one wrote: SET a = b, b = a copies rather than swaps. (This sandbox models the standard behaviour below.)` : `Every assignment reads the row as it was before the statement, so SET a = b, b = a swaps rather than copies.`, tbl.cols.map((c) => c.name), touched.map((r) => ({ cells: baseCells(r, tbl, eng), kept: true })), touched.length, touched.length),
    );
    pushReturning(stages, st.returning, touched, tbl, eng);
    return done(
      db,
      stages,
      eng === 'pg' ? `UPDATE ${touched.length}` : eng === 'mysql' ? `Query OK, ${touched.length} row${touched.length === 1 ? '' : 's'} affected` : `${touched.length} rows`,
      `${touched.length} row${touched.length === 1 ? '' : 's'} updated in ${tbl.name}.`,
      'dml',
    );
  }

  // DELETE
  const before: StageRow[] = [];
  const removed: Row[] = [];
  const survivors: Row[] = [];
  for (const r of tbl.rows) {
    const t = st.where ? truth(evalE(st.where, { row: r, cols: tbl.cols, eng }), eng) : true;
    before.push({ cells: baseCells(r, tbl, eng), kept: t === true, mark: t === null ? 'UNKNOWN' : t ? 'TRUE' : 'FALSE' });
    if (t === true) removed.push(r);
    else survivors.push(r);
  }
  tbl.rows = survivors;
  stages.push(
    mk('FROM (target)', `${before.length} candidate rows.`, tbl.cols.map((c) => c.name), before.map((r) => ({ cells: r.cells, kept: true })), before.length, before.length),
    mk('WHERE', st.where ? `Only strictly-TRUE rows are deleted. A predicate that goes UNKNOWN silently spares the row.` : 'No WHERE: the whole table is emptied, row by row, generating WAL for each.', [...tbl.cols.map((c) => c.name), 'predicate'], before, before.length, removed.length, !!st.where),
    mk('DELETE', `${removed.length} rows removed. In Postgres these are only marked dead (xmax set) and VACUUM reclaims them later; InnoDB marks them and the purge thread follows.`, tbl.cols.map((c) => c.name), removed.map((r) => ({ cells: baseCells(r, tbl, eng), kept: true })), removed.length, removed.length),
  );
  pushReturning(stages, st.returning, removed, tbl, eng);
  return done(
    db,
    stages,
    eng === 'pg' ? `DELETE ${removed.length}` : eng === 'mysql' ? `Query OK, ${removed.length} row${removed.length === 1 ? '' : 's'} affected` : `${removed.length} rows`,
    `${removed.length} row${removed.length === 1 ? '' : 's'} deleted from ${tbl.name}.`,
    'dml',
  );
}

function pushReturning(stages: Stage[], ret: Item[] | undefined, rows: Row[], tbl: TableDef, eng: Eng) {
  const items = ret ? expandItems(ret, tbl) : [];
  stages.push({
    name: 'RETURNING',
    written: '',
    present: !!ret,
    note: ret
      ? `RETURNING hands back the rows as they are after the write — defaults filled in, coercions applied, generated keys populated — without a second round trip. Postgres has had it since 8.2 and SQLite since 3.35; MySQL has none.`
      : 'No RETURNING clause: the statement reports a row count and nothing else.',
    cols: ret ? items.map((it) => outName(it, eng)) : ['(none)'],
    rows: ret ? rows.map((r) => ({ kept: true, cells: items.map((it) => evalE(it.e, { row: r, cols: tbl.cols, eng })) })) : [],
    inN: rows.length,
    outN: ret ? rows.length : 0,
  });
}

function done(db: DB, stages: Stage[], tag: string, head: string, kind: string): RunOut {
  return { db, stages, tag, error: null, head, body: stages[0]?.note ?? '', kind, unknowns: 0, scanned: 0, returned: 0, log: [] };
}

function execute(dbIn: DB, sql: string, eng: Eng): RunOut {
  const db = cloneDb(dbIn);
  const log: { sql: string; tag: string }[] = [];
  let last: RunOut | null = null;
  try {
    const stmts = parse(sql, eng);
    if (stmts.length === 0) fail('nothing to run');
    for (const st of stmts) {
      const r = st.k === 'select' ? runSelect(st, db, eng) : runDml(st, db, eng);
      log.push({ sql: st.k.toUpperCase(), tag: r.tag });
      last = r;
    }
  } catch (err) {
    const msg = err instanceof SqlErr ? err.message : String(err);
    const prefix = eng === 'pg' ? 'ERROR:  ' : eng === 'sqlite' ? 'Error: ' : '';
    return {
      db: dbIn, // the failed batch leaves the stored rows exactly as they were
      stages: last?.stages ?? [],
      tag: 'ERROR',
      error: msg.startsWith('ERROR ') ? msg : prefix + msg,
      head: 'The statement was rejected.',
      body: 'Nothing was written. Read the message as the engine would print it, then fix one thing and run again.',
      kind: 'error',
      unknowns: 0,
      scanned: 0,
      returned: 0,
      log,
    };
  }
  return { ...last!, db, log };
}

/* ==================================================================== examples */

const EXAMPLES: { value: string; label: string; sql: string }[] = [
  {
    value: 'scan',
    label: '1 — the whole table',
    sql: 'SELECT * FROM customers;',
  },
  {
    value: 'unknown',
    label: '2 — WHERE drops UNKNOWN rows',
    sql: "SELECT id, name, city\nFROM customers\nWHERE city <> 'Tacoma';",
  },
  {
    value: 'nulls',
    label: '3 — IS NULL, COALESCE, NULLIF',
    sql:
      "SELECT name,\n       city,\n       COALESCE(city, '(none on file)') AS where_from,\n       NULLIF(credit, 0) AS real_credit\nFROM customers\nWHERE city IS NULL OR credit IS NULL;",
  },
  {
    value: 'order',
    label: '4 — ORDER BY and NULL placement',
    sql: 'SELECT name, credit\nFROM customers\nORDER BY credit DESC;',
  },
  {
    value: 'limit',
    label: '5 — LIMIT is the last stage',
    sql: 'SELECT id, total\nFROM orders\nORDER BY total DESC\nLIMIT 3 OFFSET 2;',
  },
  {
    value: 'distinct',
    label: '6 — DISTINCT folds NULLs together',
    sql: 'SELECT DISTINCT city FROM customers;',
  },
  {
    value: 'alias',
    label: '7 — alias in WHERE (fails)',
    sql: 'SELECT id, ROUND(total * 0.0925, 2) AS tax\nFROM orders\nWHERE tax > 40;',
  },
  {
    value: 'aliasok',
    label: '8 — the same alias in ORDER BY (works)',
    sql: 'SELECT id, ROUND(total * 0.0925, 2) AS tax\nFROM orders\nORDER BY tax DESC\nLIMIT 5;',
  },
  {
    value: 'casts',
    label: '9 — implicit coercion, three answers',
    sql: "SELECT '7' + 1 AS implicit,\n       7 / 2 AS division,\n       'a' || 'b' AS concat,\n       'Seattle' = 'seattle' AS same_city\nFROM customers\nLIMIT 1;",
  },
  {
    value: 'explicitcast',
    label: '10 — an explicit cast is spelled differently',
    sql: "SELECT CAST('7' AS INTEGER) + 1 AS explicit\nFROM customers\nLIMIT 1;",
  },
  {
    value: 'insert',
    label: '11 — INSERT … RETURNING',
    sql:
      "INSERT INTO customers (id, name, city, signed_on)\nVALUES (9, 'Rowan Idris', NULL, '2024-06-02')\nRETURNING id, name, city, active;",
  },
  {
    value: 'update',
    label: '12 — UPDATE … RETURNING',
    sql:
      "UPDATE orders\nSET status = 'shipped', shipped_on = '2024-06-15'\nWHERE status = 'paid'\nRETURNING id, status, shipped_on;",
  },
  {
    value: 'delete',
    label: '13 — DELETE … RETURNING',
    sql: 'DELETE FROM items\nWHERE qty = 4\nRETURNING id, sku, qty;',
  },
  {
    value: 'check',
    label: '14 — CHECK constraint violation',
    sql:
      "INSERT INTO orders (id, customer_id, placed_on, status, total)\nVALUES (200, 1, '2024-06-10', 'refunded', 42.00);",
  },
  {
    value: 'toolong',
    label: '15 — VARCHAR(12) overflow',
    sql:
      "INSERT INTO customers (id, name, city, signed_on)\nVALUES (10, 'Ada Okonkwo', 'San Francisco', '2024-06-03');",
  },
  {
    value: 'affinity',
    label: '16 — text into an INTEGER column',
    sql: "INSERT INTO items (id, order_id, sku, qty, price)\nVALUES (600, 100, 'KBD-01', 'three', 49.00);",
  },
  {
    value: 'ddl',
    label: '17 — CREATE TABLE with constraints',
    sql:
      "CREATE TABLE refunds (\n  id        INTEGER PRIMARY KEY,\n  order_id  INTEGER NOT NULL,\n  reason    VARCHAR(20) NOT NULL DEFAULT 'unspecified',\n  amount    NUMERIC NOT NULL CHECK (amount > 0),\n  issued_on DATE NOT NULL\n);",
  },
];

/* =================================================================== rendering */

function ValCell({ v, eng }: { v: Val; eng: Eng }) {
  if (v === null)
    return (
      <em style={{ color: 'var(--viz-ink-muted)', fontStyle: 'italic' }} title="the SQL null marker — not a value">
        NULL
      </em>
    );
  if (typeof v === 'boolean') return <>{eng === 'pg' ? (v ? 't' : 'f') : v ? '1' : '0'}</>;
  return <>{String(v)}</>;
}

const MARK_COLOR: Record<string, string> = {
  TRUE: 'var(--viz-good)',
  FALSE: 'var(--viz-ink-muted)',
  UNKNOWN: 'var(--viz-warning)',
  dup: 'var(--viz-ink-muted)',
  skipped: 'var(--viz-ink-muted)',
  cut: 'var(--viz-ink-muted)',
};

function Grid({ stage, eng, max = 10 }: { stage: Stage; eng: Eng; max?: number }) {
  const shown = stage.rows.slice(0, max);
  const hidden = stage.rows.length - shown.length;
  const marks = stage.rows.some((r) => r.mark);
  return (
    <>
      <table className="viz-table">
        <thead>
          <tr>
            <th style={{ width: '1.4rem' }} />
            {stage.cols.map((c, i) =>
              c === 'predicate' ? null : (
                <th key={`${c}-${i}`}>{c}</th>
              ),
            )}
            {marks ? <th>verdict</th> : null}
          </tr>
        </thead>
        <tbody>
          {shown.length === 0 ? (
            <tr>
              <td colSpan={stage.cols.length + 2} style={{ color: 'var(--viz-ink-muted)' }}>
                no rows
              </td>
            </tr>
          ) : (
            shown.map((r, i) => (
              <tr key={i} style={{ opacity: r.kept ? 1 : 0.4 }}>
                <td style={{ color: r.kept ? 'var(--viz-1)' : 'var(--viz-ink-muted)' }}>{r.kept ? '✓' : '✕'}</td>
                {r.cells.map((c, j) => (
                  <td key={j} style={{ textDecoration: r.kept ? undefined : 'line-through' }}>
                    <ValCell v={c.v} eng={eng} />
                  </td>
                ))}
                {marks ? (
                  <td style={{ color: MARK_COLOR[r.mark ?? ''] ?? 'var(--viz-ink-muted)', fontWeight: r.mark === 'UNKNOWN' ? 600 : 400 }}>
                    {r.mark ?? '—'}
                  </td>
                ) : null}
              </tr>
            ))
          )}
        </tbody>
      </table>
      {hidden > 0 ? (
        <p style={{ margin: '.25rem 0 0', color: 'var(--viz-ink-muted)', fontSize: '.75rem' }}>
          … {hidden} more row{hidden === 1 ? '' : 's'} at this stage (all of them are in “Show the numbers”).
        </p>
      ) : null}
    </>
  );
}

/* =================================================================== component */

const SEED = seedDb();
const FIRST = execute(SEED, EXAMPLES[1].sql, 'pg');

export default function SqlClausePipelineSandbox() {
  const [eng, setEng] = useState<Eng>('pg');
  const [sql, setSql] = useState(EXAMPLES[1].sql);
  const [example, setExample] = useState('unknown');
  const [state, setState] = useState<{ before: DB; db: DB; run: RunOut; step: number }>({
    before: SEED,
    db: FIRST.db,
    run: FIRST,
    step: 1,
  });
  const [ref, width] = useSize(760);
  const tip = useTip();

  const doRun = (text = sql, engine = eng, from = state.db) => {
    const r = execute(from, text, engine);
    setState({ before: from, db: r.db, run: r, step: r.kind === 'select' ? 1 : 0 });
  };
  const changeEngine = (e: Eng) => {
    setEng(e);
    const r = execute(state.before, sql, e);
    setState({ before: state.before, db: r.db, run: r, step: r.kind === 'select' ? 1 : 0 });
  };
  const loadExample = (v: string) => {
    setExample(v);
    const ex = EXAMPLES.find((x) => x.value === v)!;
    setSql(ex.sql);
    doRun(ex.sql, eng, state.db);
  };
  const reset = () => {
    const fresh = seedDb();
    const r = execute(fresh, sql, eng);
    setState({ before: fresh, db: r.db, run: r, step: r.kind === 'select' ? 1 : 0 });
  };

  const run = state.run;
  const stages = run.stages;
  const step = Math.min(state.step, Math.max(0, stages.length - 1));
  const cur = stages[step];

  /* the clause strip */
  const boxW = 118;
  const gap = 26;
  const svgW = Math.max(width, stages.length * boxW + (stages.length - 1) * gap + 16);
  const svgH = 104;

  const tableRows = Object.values(state.db).map((t) => ({ name: t.name, cols: t.cols.length, rows: t.rows.length }));

  return (
    <VizPanel
      title="A SELECT is not evaluated in the order you wrote it"
      subtitle="A deterministic SQL interpreter over 42 rows of customers, orders and items. Run a statement, then step through the interpreter's own intermediate row set after each clause."
      controls={
        <>
          <Segmented label="Engine" value={eng} options={ENGINES} onChange={changeEngine} />
          <Choice label="Example" value={example} options={EXAMPLES.map((e) => ({ value: e.value, label: e.label }))} onChange={loadExample} />
          <Button primary onClick={() => doRun()} title="Ctrl/⌘ + Enter">
            Run
          </Button>
          <Button onClick={() => setState((s) => ({ ...s, step: Math.max(0, s.step - 1) }))} disabled={step === 0}>
            ◀ stage
          </Button>
          <Button onClick={() => setState((s) => ({ ...s, step: Math.min(stages.length - 1, s.step + 1) }))} disabled={step >= stages.length - 1}>
            stage ▶
          </Button>
          <Button onClick={reset} title="Restore the 42 seed rows">
            Reset data
          </Button>
        </>
      }
      legend={
        <Legend
          items={[
            { label: 'clause present — this stage does work', color: 'var(--viz-1)' },
            { label: 'clause absent — rows pass straight through (dashed)', color: 'var(--viz-ink-muted)', shape: 'line' },
            { label: 'the stage you are standing on', color: 'var(--viz-2)' },
            { label: '✓ row survives this stage / ✕ row is dropped here', color: 'var(--viz-good)', shape: 'dot' },
          ]}
        />
      }
      stats={
        <Stats
          items={[
            { label: 'Result', value: run.error ? 'error' : run.tag, hint: 'What the client reports for this statement' },
            { label: 'Rows scanned', value: fmtNum(stages[0]?.inN ?? 0), hint: 'What FROM handed to the pipeline' },
            { label: 'Rows returned', value: fmtNum(stages[stages.length - 1]?.outN ?? 0) },
            { label: 'Dropped as UNKNOWN', value: fmtNum(run.unknowns), hint: 'Rows WHERE discarded because the predicate was neither TRUE nor FALSE' },
            { label: 'Stage', value: cur ? `${step + 1}/${stages.length} · ${cur.name}` : '—' },
            { label: 'Tables', value: `${tableRows.length} · ${fmtNum(tableRows.reduce((a, t) => a + t.rows, 0))} rows` },
          ]}
        />
      }
      note={
        <Note>
          <strong>{run.error ? run.error : run.head}</strong>{' '}
          {run.error ? run.body : cur ? cur.note : ''}
        </Note>
      }
      table={
        <>
          <table className="viz-table">
            <thead>
              <tr>
                <th>Stage</th>
                <th>Clause present</th>
                <th>Rows in</th>
                <th>Rows out</th>
                <th>Written</th>
              </tr>
            </thead>
            <tbody>
              {stages.map((s, i) => (
                <tr key={s.name}>
                  <td>
                    {i + 1}. {s.name}
                  </td>
                  <td>{s.present ? 'yes' : 'no — pass-through'}</td>
                  <td>{fmtNum(s.inN)}</td>
                  <td>{fmtNum(s.outN)}</td>
                  <td>{s.written || '—'}</td>
                </tr>
              ))}
            </tbody>
          </table>
          <table className="viz-table">
            <thead>
              <tr>
                <th>Table</th>
                <th>Columns</th>
                <th>Rows now</th>
              </tr>
            </thead>
            <tbody>
              {tableRows.map((t) => (
                <tr key={t.name}>
                  <td>{t.name}</td>
                  <td>{t.cols}</td>
                  <td>{fmtNum(t.rows)}</td>
                </tr>
              ))}
            </tbody>
          </table>
          {cur ? (
            <table className="viz-table">
              <thead>
                <tr>
                  <th>kept</th>
                  {cur.cols.filter((c) => c !== 'predicate').map((c, i) => (
                    <th key={`${c}-${i}`}>{c}</th>
                  ))}
                  <th>verdict</th>
                </tr>
              </thead>
              <tbody>
                {cur.rows.map((r, i) => (
                  <tr key={i}>
                    <td>{r.kept ? 'yes' : 'no'}</td>
                    {r.cells.map((c, j) => (
                      <td key={j}>{c.v === null ? 'NULL' : String(c.v)}</td>
                    ))}
                    <td>{r.mark ?? '—'}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          ) : null}
        </>
      }
    >
      <div ref={ref}>
        <textarea
          value={sql}
          spellCheck={false}
          onChange={(e) => setSql(e.currentTarget.value)}
          onKeyDown={(e) => {
            if (e.key === 'Enter' && (e.metaKey || e.ctrlKey)) {
              e.preventDefault();
              doRun();
            }
          }}
          rows={Math.min(10, Math.max(4, sql.split('\n').length + 1))}
          aria-label="SQL statement"
          style={{
            width: '100%',
            boxSizing: 'border-box',
            fontFamily: 'ui-monospace, SFMono-Regular, Menlo, monospace',
            fontSize: '.8125rem',
            lineHeight: 1.5,
            color: 'var(--viz-ink)',
            background: 'var(--viz-plane)',
            border: `1px solid ${run.error ? 'var(--viz-critical)' : 'var(--viz-border)'}`,
            borderRadius: '6px',
            padding: '.5rem .6rem',
            resize: 'vertical',
          }}
        />

        <TooltipHost>
          <svg width={svgW} height={svgH} role="img" aria-label="The clause pipeline, in logical evaluation order">
            {stages.map((s, i) => {
              const x = 8 + i * (boxW + gap);
              const active = i === step;
              const stroke = active ? 'var(--viz-2)' : s.present ? 'var(--viz-1)' : 'var(--viz-axis)';
              return (
                <g
                  key={s.name}
                  onClick={() => setState((cs) => ({ ...cs, step: i }))}
                  style={{ cursor: 'pointer' }}
                  {...tip(
                    <>
                      <strong>
                        {i + 1}. {s.name}
                      </strong>
                      <br />
                      {s.note}
                    </>,
                  )}
                >
                  <rect
                    x={x}
                    y={22}
                    width={boxW}
                    height={54}
                    rx={8}
                    fill={active ? 'var(--viz-neutral)' : 'var(--viz-plane)'}
                    stroke={stroke}
                    strokeWidth={active ? 2.5 : 1.5}
                    strokeDasharray={s.present ? undefined : '5 4'}
                    opacity={s.present ? 1 : 0.75}
                  />
                  <text x={x + boxW / 2} y={42} textAnchor="middle" fill="var(--viz-ink)" fontWeight={600}>
                    {s.name}
                  </text>
                  <text x={x + boxW / 2} y={58} textAnchor="middle" fill="var(--viz-ink-2)">
                    {fmtNum(s.inN)} → {fmtNum(s.outN)} rows
                  </text>
                  <text x={x + boxW / 2} y={70} textAnchor="middle" fill="var(--viz-ink-muted)">
                    {s.present ? s.written || 'applies' : 'pass-through'}
                  </text>
                  <text x={x + boxW / 2} y={16} textAnchor="middle" fill={active ? 'var(--viz-2)' : 'var(--viz-ink-muted)'} fontWeight={active ? 600 : 400}>
                    evaluated {i + 1}
                    {i === 0 ? 'st' : i === 1 ? 'nd' : i === 2 ? 'rd' : 'th'}
                  </text>
                  {i < stages.length - 1 ? (
                    <path
                      d={`M ${x + boxW + 4} 49 L ${x + boxW + gap - 6} 49 M ${x + boxW + gap - 12} 45 L ${x + boxW + gap - 6} 49 L ${x + boxW + gap - 12} 53`}
                      stroke="var(--viz-axis)"
                      strokeWidth={1.5}
                      fill="none"
                    />
                  ) : null}
                  {active ? (
                    <rect x={x} y={82} width={boxW} height={4} rx={2} fill="var(--viz-2)" />
                  ) : null}
                </g>
              );
            })}
            {stages.length === 0 ? (
              <text x={12} y={50} fill="var(--viz-ink-muted)">
                The statement did not run — see the message below.
              </text>
            ) : null}
          </svg>
        </TooltipHost>

        {cur ? (
          <div style={{ marginTop: '.5rem' }}>
            <p style={{ margin: '0 0 .15rem', color: 'var(--viz-ink-2)', fontSize: '.78rem' }}>
              Intermediate row set after <strong style={{ color: 'var(--viz-ink)' }}>{cur.name}</strong>
              {cur.present ? '' : ' (clause absent — unchanged from the stage before it)'}
            </p>
            <Grid stage={cur} eng={eng} />
          </div>
        ) : null}
      </div>
    </VizPanel>
  );
}
