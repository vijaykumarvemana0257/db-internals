import { useEffect, useMemo, useState } from 'react';
import { VizPanel, Choice, Slider, Check, Button, Legend, Stats, Note, fmtNum } from './Viz';

/**
 * Record shredding and assembly (Dremel, VLDB 2010) plus a minimal, real Parquet writer.
 *
 * Model, and what it follows:
 * - Schemas use Dremel's required / optional / repeated modifiers. Repetition level = which repeated field in the
 *   path repeated most recently (0 = new record); definition level = how many optional or repeated fields in the
 *   path are present. Checked against Figure 3 of the paper for the Document records r1 and r2.
 * - The assembly FSM has one state per selected leaf, in schema order. From field f with next repetition level l:
 *   if l <= the common repetition level of f and the next field (the "barrier"), go to the barrier; otherwise go
 *   to the first selected field inside f's ancestor that repeats at level l. Reproduces the paper's Figures 4 and 5.
 * - Assembly keeps a stack of open nested records. Before appending a value it closes records not on the field's
 *   path and opens the missing ones down to the value's definition level; a back-transition at level l closes the
 *   ancestor repeating at l, so the next value starts a new instance of it.
 * - The writer emits a valid uncompressed Parquet file: "PAR1", per row group one column chunk per leaf made of an
 *   optional dictionary page (PLAIN) and v1 data pages (RLE/bit-packed levels, PLAIN or RLE_DICTIONARY values),
 *   then (optionally) every ColumnIndex followed by every OffsetIndex, then the Thrift-compact FileMetaData, its
 *   4-byte little-endian length and "PAR1" — the order pyarrow 25 uses. Pages are cut at row boundaries after a
 *   fixed number of rows (real writers cut by bytes: 1 MB and 20,000 rows by default in parquet-java and Arrow).
 */

export type Rep = 'required' | 'optional' | 'repeated';
export type SNode = { name: string; rep: Rep; type?: 'int64' | 'string'; children?: SNode[]; list?: boolean };
export type Json = null | boolean | number | string | Json[] | { [k: string]: Json };
type Obj = { [k: string]: unknown };

export const DOCUMENT: SNode = {
  name: 'Document',
  rep: 'required',
  children: [
    { name: 'DocId', rep: 'required', type: 'int64' },
    {
      name: 'Links',
      rep: 'optional',
      children: [
        { name: 'Backward', rep: 'repeated', type: 'int64' },
        { name: 'Forward', rep: 'repeated', type: 'int64' },
      ],
    },
    {
      name: 'Name',
      rep: 'repeated',
      children: [
        {
          name: 'Language',
          rep: 'repeated',
          children: [
            { name: 'Code', rep: 'required', type: 'string' },
            { name: 'Country', rep: 'optional', type: 'string' },
          ],
        },
        { name: 'Url', rep: 'optional', type: 'string' },
      ],
    },
  ],
};

export const TAGGED: SNode = {
  name: 'Event',
  rep: 'required',
  children: [
    { name: 'id', rep: 'required', type: 'int64' },
    {
      name: 'tags',
      rep: 'optional',
      list: true,
      children: [{ name: 'list', rep: 'repeated', children: [{ name: 'element', rep: 'optional', type: 'string' }] }],
    },
  ],
};

export type Leaf = { index: number; id: string; name: string; node: SNode; path: SNode[]; maxRep: number; maxDef: number; type: 'int64' | 'string' };

export function leavesOf(schema: SNode): Leaf[] {
  const out: Leaf[] = [];
  const walk = (node: SNode, path: SNode[]) => {
    for (const c of node.children ?? []) {
      const p = [...path, c];
      if (c.children) walk(c, p);
      else
        out.push({
          index: out.length,
          id: p.map((n) => n.name).join('.'),
          name: c.name,
          node: c,
          path: p,
          maxRep: p.filter((n) => n.rep === 'repeated').length,
          maxDef: p.filter((n) => n.rep !== 'required').length,
          type: c.type ?? 'string',
        });
    }
  };
  walk(schema, []);
  return out;
}

/** Leaves under a node, by leaf index. */
function leafRange(schema: SNode, leaves: Leaf[], node: SNode) {
  if (node === schema) return leaves.map((l) => l.index);
  return leaves.filter((l) => l.path.includes(node)).map((l) => l.index);
}

/* ------------------------------------------------------------------ parsing JSON into schema-shaped records */

export const MAX_RECORDS = 12;
export const MAX_ENTRIES = 400;

function fromJson(node: SNode, v: unknown, where: string, errors: string[]): unknown {
  // A LIST-annotated group is written in JSON as a plain array: [x, y] -> { list: [{ element: x }, { element: y }] }
  if (node.list) {
    if (v === null || v === undefined) return undefined;
    if (!Array.isArray(v)) {
      errors.push(`${where} must be an array or null`);
      return undefined;
    }
    const listNode = node.children![0];
    const elNode = listNode.children![0];
    return { list: v.map((e, i) => ({ element: leafValue(elNode, e, `${where}[${i}]`, errors) })) };
  }
  if (!node.children) return leafValue(node, v, where, errors);
  if (typeof v !== 'object' || v === null || Array.isArray(v)) {
    errors.push(`${where} must be an object`);
    return {};
  }
  const o = v as Obj;
  const out: Obj = {};
  for (const k of Object.keys(o)) if (!node.children.some((c) => c.name === k)) errors.push(`${where}.${k} is not in the schema`);
  for (const c of node.children) {
    const cv = o[c.name];
    const cw = `${where}.${c.name}`;
    if (c.rep === 'repeated') {
      if (cv === undefined || cv === null) continue;
      if (!Array.isArray(cv)) {
        errors.push(`${cw} is repeated, so it must be an array`);
        continue;
      }
      out[c.name] = cv.map((item, i) => (c.children ? fromJson(c, item, `${cw}[${i}]`, errors) : leafValue(c, item, `${cw}[${i}]`, errors)));
    } else if (c.rep === 'optional') {
      if (cv === undefined || cv === null) continue;
      out[c.name] = fromJson(c, cv, cw, errors);
    } else {
      if (cv === undefined || cv === null) {
        errors.push(`${cw} is required`);
        continue;
      }
      out[c.name] = fromJson(c, cv, cw, errors);
    }
  }
  return out;
}

function leafValue(node: SNode, v: unknown, where: string, errors: string[]) {
  if (v === null || v === undefined) {
    if (node.rep === 'optional') return null;
    errors.push(`${where} cannot be null`);
    return node.type === 'int64' ? 0 : '';
  }
  if (node.type === 'int64') {
    if (typeof v !== 'number' || !Number.isSafeInteger(v)) {
      errors.push(`${where} must be an integer`);
      return 0;
    }
    return v;
  }
  if (typeof v !== 'string') {
    errors.push(`${where} must be a string`);
    return '';
  }
  if (v.length > 40) errors.push(`${where} is longer than 40 characters`);
  return v;
}

export type Parsed = { ok: true; records: Obj[] } | { ok: false; error: string };

export function parseRecords(schema: SNode, text: string): Parsed {
  let raw: unknown;
  try {
    raw = JSON.parse(text);
  } catch (e) {
    return { ok: false, error: `Not valid JSON: ${(e as Error).message}` };
  }
  if (!Array.isArray(raw)) return { ok: false, error: 'Write an array of records: [ {...}, {...} ]' };
  if (raw.length === 0) return { ok: false, error: 'Add at least one record.' };
  if (raw.length > MAX_RECORDS) return { ok: false, error: `The lab takes at most ${MAX_RECORDS} records.` };
  const errors: string[] = [];
  const records = raw.map((r, i) => fromJson(schema, r, `record ${i}`, errors) as Obj);
  if (errors.length) return { ok: false, error: errors.slice(0, 3).join('; ') };
  const cols = shred(schema, records);
  const n = cols.reduce((s, c) => s + c.length, 0);
  if (n > MAX_ENTRIES) return { ok: false, error: `Too many values (${n}); the lab takes at most ${MAX_ENTRIES}.` };
  return { ok: true, records };
}

/** Back from schema-shaped records to the JSON the learner writes. */
export function toJson(schema: SNode, rec: unknown): Json {
  const conv = (node: SNode, v: unknown): Json => {
    if (!node.children) return v as Json;
    if (node.list) {
      const o = (v ?? {}) as Obj;
      const items = (o.list as Obj[] | undefined) ?? [];
      return items.map((it) => (it.element === undefined ? null : (it.element as Json)));
    }
    const o = v as Obj;
    const out: { [k: string]: Json } = {};
    for (const c of node.children) {
      if (!(c.name in o)) continue;
      const cv = o[c.name];
      out[c.name] = c.rep === 'repeated' ? (cv as unknown[]).map((x) => conv(c, x)) : conv(c, cv);
    }
    return out;
  };
  return conv(schema, rec);
}

/* ------------------------------------------------------------------ shredding */

export type Entry = { v: number | string | null; r: number; d: number; row: number };

export function shred(schema: SNode, records: Obj[]): Entry[][] {
  const leaves = leavesOf(schema);
  const cols: Entry[][] = leaves.map(() => []);
  const leafIndex = new Map(leaves.map((l) => [l.node, l.index] as const));
  const repOf = new Map<SNode, number>();
  const annotate = (node: SNode, rep: number) => {
    for (const c of node.children ?? []) {
      const r = rep + (c.rep === 'repeated' ? 1 : 0);
      repOf.set(c, r);
      annotate(c, r);
    }
  };
  annotate(schema, 0);

  const emitNull = (node: SNode, r: number, d: number, row: number) => {
    for (const i of leafRange(schema, leaves, node)) cols[i].push({ v: null, r, d, row });
  };
  const field = (node: SNode, value: unknown, r: number, d: number, row: number) => {
    if (node.rep === 'repeated') {
      const arr = (value as unknown[] | undefined) ?? [];
      if (arr.length === 0) return emitNull(node, r, d, row);
      arr.forEach((item, i) => instance(node, item, i === 0 ? r : repOf.get(node)!, d + 1, row));
    } else if (node.rep === 'optional') {
      if (value === undefined || value === null) return emitNull(node, r, d, row);
      instance(node, value, r, d + 1, row);
    } else instance(node, value, r, d, row);
  };
  const instance = (node: SNode, v: unknown, r: number, d: number, row: number) => {
    if (!node.children) cols[leafIndex.get(node)!].push({ v: v as number | string, r, d, row });
    else for (const c of node.children) field(c, (v as Obj)[c.name], r, d, row);
  };
  records.forEach((rec, row) => instance(schema, rec, 0, 0, row));
  return cols;
}

/* ------------------------------------------------------------------ the assembly FSM */

export function commonRep(a: Leaf, b: Leaf) {
  let r = 0;
  for (let k = 0; k < a.path.length && k < b.path.length && a.path[k] === b.path[k]; k++) if (a.path[k].rep === 'repeated') r++;
  return r;
}

export type Fsm = { fields: Leaf[]; trans: number[][]; barrierLevel: number[] };

/** trans[i][l] = index of the next field, or fields.length for the end state. */
export function buildFsm(fields: Leaf[]): Fsm {
  const trans: number[][] = [];
  const barrierLevel: number[] = [];
  fields.forEach((f, i) => {
    const barrier = i + 1 < fields.length ? fields[i + 1] : null;
    const bl = barrier ? commonRep(f, barrier) : 0;
    barrierLevel.push(bl);
    const row: number[] = [];
    for (let l = 0; l <= f.maxRep; l++) {
      if (l <= bl) row.push(i + 1);
      else row.push(fields.findIndex((g, j) => j <= i && commonRep(g, f) >= l));
    }
    trans.push(row);
  });
  return { fields, trans, barrierLevel };
}

/** Depth (1-based position in the path) of the ancestor that repeats at level l. */
function repAncestorDepth(f: Leaf, l: number) {
  let seen = 0;
  for (let k = 0; k < f.path.length; k++) {
    if (f.path[k].rep === 'repeated') seen++;
    if (seen === l) return k + 1;
  }
  return f.path.length;
}

/** Deepest path position whose definition level is <= d (0 = nothing below the record is defined). */
function definedDepth(f: Leaf, d: number) {
  let def = 0;
  let depth = 0;
  for (let k = 0; k < f.path.length; k++) {
    if (f.path[k].rep !== 'required') def++;
    if (def > d) break;
    depth = k + 1;
  }
  return depth;
}

export type AsmStep = {
  field: number;
  entry: number;
  e: Entry;
  opened: string[];
  appended: boolean;
  nextLevel: number;
  exhausted: boolean;
  to: number;
  back: boolean;
  closed: string | null;
  record: number;
};

export function assemble(schema: SNode, cols: Entry[][], selected: number[], maxSteps = Infinity) {
  const leaves = leavesOf(schema);
  const fields = [...new Set(selected)].sort((a, b) => a - b).map((i) => leaves[i]);
  const fsm = buildFsm(fields);
  const records: Obj[] = [];
  const steps: AsmStep[] = [];
  const total = fields.reduce((s, f) => s + cols[f.index].length, 0);
  if (!fields.length) return { fsm, records, steps, total, openPath: [] as string[], current: 0, done: true };
  const cursors = fields.map(() => 0);
  type Frame = { node: SNode; obj: Obj; label: string };
  let stack: Frame[] | null = null;
  let cur = 0;
  while (steps.length < maxSteps && steps.length < total) {
    const f = fields[cur];
    const col = cols[f.index];
    if (cursors[cur] >= col.length) break;
    if (stack === null) {
      const rec: Obj = {};
      records.push(rec);
      stack = [{ node: schema, obj: rec, label: `record ${records.length - 1}` }];
    }
    const entryIndex = cursors[cur]++;
    const e = col[entryIndex];
    // MoveToLevel: keep only the open records that are on this field's path ...
    let k = 1;
    while (k < stack.length && k <= f.path.length && stack[k].node === f.path[k - 1]) k++;
    stack.length = k;
    // ... then open the missing ones down to the value's definition level.
    const groupDepth = e.v !== null ? f.path.length - 1 : Math.min(definedDepth(f, e.d), f.path.length - 1);
    const opened: string[] = [];
    for (let depth = stack.length; depth <= groupDepth; depth++) {
      const node = f.path[depth - 1];
      const parent = stack[depth - 1].obj;
      let inst: Obj;
      let label = node.name;
      if (node.rep === 'repeated') {
        const arr = (parent[node.name] ??= []) as Obj[];
        inst = {};
        arr.push(inst);
        label = `${node.name}[${arr.length - 1}]`;
      } else inst = (parent[node.name] ??= {}) as Obj;
      stack.push({ node, obj: inst, label });
      opened.push(label);
    }
    if (e.v !== null) {
      const parent = stack[f.path.length - 1].obj;
      if (f.node.rep === 'repeated') ((parent[f.node.name] ??= []) as unknown[]).push(e.v);
      else parent[f.node.name] = e.v;
    }
    const exhausted = cursors[cur] >= col.length;
    const nextLevel = exhausted ? 0 : col[cursors[cur]].r;
    const to = fsm.trans[cur][nextLevel];
    const back = nextLevel > fsm.barrierLevel[cur];
    let closed: string | null = null;
    if (back) {
      const dA = repAncestorDepth(f, nextLevel);
      if (dA < stack.length) closed = stack[dA].label;
      stack.length = Math.min(stack.length, dA);
    }
    steps.push({ field: cur, entry: entryIndex, e, opened, appended: e.v !== null, nextLevel, exhausted, to, back, closed, record: records.length - 1 });
    if (to === fields.length) {
      stack = null;
      cur = 0;
    } else cur = to;
  }
  const openPath = stack ? stack.map((s) => s.label) : [];
  return { fsm, records, steps, total, openPath, current: cur, done: steps.length >= total };
}

/** What assembly from a subset must produce: the records with every unselected leaf (and group left empty of selected leaves) stripped. */
export function project(schema: SNode, rec: Obj, selected: number[]): Obj {
  const leaves = leavesOf(schema);
  const sel = new Set(selected);
  const keep = (node: SNode) => leaves.some((l) => sel.has(l.index) && (l.node === node || l.path.includes(node)));
  const inst = (node: SNode, v: unknown): unknown => {
    if (!node.children) return v;
    const o = v as Obj;
    const out: Obj = {};
    for (const c of node.children) {
      if (!keep(c)) continue;
      const cv = o[c.name];
      if (c.rep === 'repeated') {
        const arr = (cv as unknown[] | undefined) ?? [];
        if (arr.length) out[c.name] = arr.map((x) => inst(c, x));
      } else if (cv !== undefined && cv !== null) out[c.name] = inst(c, cv);
    }
    return out;
  };
  return inst(schema, rec) as Obj;
}

/** Rebuild a record with keys in schema order, dropping empty repeated fields and nulls (they shred identically to missing ones). */
export function canonical(schema: SNode, rec: Obj): string {
  const inst = (node: SNode, v: unknown): unknown => {
    if (!node.children) return v;
    const o = v as Obj;
    const out: Obj = {};
    for (const c of node.children) {
      const cv = o[c.name];
      if (cv === undefined || cv === null || (c.rep === 'repeated' && (cv as unknown[]).length === 0)) continue;
      out[c.name] = c.rep === 'repeated' ? (cv as unknown[]).map((x) => inst(c, x)) : inst(c, cv);
    }
    return out;
  };
  return JSON.stringify(inst(schema, rec));
}

/* ------------------------------------------------------------------ Thrift compact protocol */

type TVal =
  | { t: 'i16' | 'i32' | 'i64'; v: number }
  | { t: 'bool'; v: boolean }
  | { t: 'bin'; v: Uint8Array | string }
  | { t: 'struct'; v: TField[] }
  | { t: 'list'; et: 'i32' | 'i64' | 'bin' | 'struct' | 'bool'; v: TVal[] };
type TField = [number, TVal];

const TYPE_ID = { bool: 1, i16: 4, i32: 5, i64: 6, bin: 8, list: 9, struct: 12 } as const;
const enc = new TextEncoder();

function varint(out: number[], n: number) {
  while (n >= 0x80) {
    out.push((n % 0x80) | 0x80);
    n = Math.floor(n / 0x80);
  }
  out.push(n);
}
const zigzag = (n: number) => (n >= 0 ? n * 2 : -n * 2 - 1);

function writeVal(out: number[], x: TVal) {
  switch (x.t) {
    case 'i16':
    case 'i32':
    case 'i64':
      varint(out, zigzag(x.v));
      break;
    case 'bool':
      out.push(x.v ? 1 : 2);
      break;
    case 'bin': {
      const b = typeof x.v === 'string' ? enc.encode(x.v) : x.v;
      varint(out, b.length);
      for (const c of b) out.push(c);
      break;
    }
    case 'list':
      if (x.v.length < 15) out.push((x.v.length << 4) | TYPE_ID[x.et]);
      else {
        out.push(0xf0 | TYPE_ID[x.et]);
        varint(out, x.v.length);
      }
      for (const e of x.v) writeVal(out, e);
      break;
    case 'struct':
      writeStruct(out, x.v);
      break;
  }
}

function writeStruct(out: number[], fields: TField[]) {
  let last = 0;
  for (const [id, val] of fields) {
    const type = val.t === 'bool' ? (val.v ? 1 : 2) : TYPE_ID[val.t];
    const delta = id - last;
    if (delta > 0 && delta <= 15) out.push((delta << 4) | type);
    else {
      out.push(type);
      varint(out, zigzag(id));
    }
    last = id;
    if (val.t !== 'bool') writeVal(out, val);
  }
  out.push(0);
}

const i32 = (v: number): TVal => ({ t: 'i32', v });
const i64 = (v: number): TVal => ({ t: 'i64', v });
const bin = (v: Uint8Array | string): TVal => ({ t: 'bin', v });
const st = (...v: TField[]): TVal => ({ t: 'struct', v });

/* ------------------------------------------------------------------ Parquet encodings */

/** RLE / bit-packing hybrid: runs of 8+ equal values become RLE runs, the rest is bit-packed LSB-first in groups of 8. */
export function hybrid(values: number[], width: number): number[] {
  const out: number[] = [];
  const n = values.length;
  const byteW = Math.ceil(width / 8);
  const runAt = (i: number) => {
    let r = 1;
    while (i + r < n && values[i + r] === values[i]) r++;
    return r;
  };
  let i = 0;
  while (i < n) {
    const run = runAt(i);
    if (run >= 8) {
      varint(out, run * 2);
      for (let b = 0; b < byteW; b++) out.push(Math.floor(values[i] / 2 ** (8 * b)) & 255);
      i += run;
      continue;
    }
    let j = i;
    do j += 8;
    while (j < n && runAt(j) < 8);
    const groups = Math.ceil((Math.min(j, n) - i) / 8);
    varint(out, groups * 2 + 1);
    const bits: number[] = [];
    for (let k = 0; k < groups * 8; k++) {
      const v = i + k < n ? values[i + k] : 0;
      for (let b = 0; b < width; b++) bits.push((v >> b) & 1);
    }
    for (let k = 0; k < bits.length; k += 8) {
      let byte = 0;
      for (let b = 0; b < 8 && k + b < bits.length; b++) byte |= bits[k + b] << b;
      out.push(byte);
    }
    i = Math.min(j, n);
  }
  return out;
}

const bitWidth = (maxValue: number) => (maxValue <= 0 ? 0 : Math.floor(Math.log2(maxValue)) + 1);

function plain(v: number | string, type: 'int64' | 'string'): number[] {
  if (type === 'int64') {
    let x = BigInt.asUintN(64, BigInt(v as number));
    const out: number[] = [];
    for (let i = 0; i < 8; i++) {
      out.push(Number(x & BigInt(255)));
      x >>= BigInt(8);
    }
    return out;
  }
  const b = enc.encode(v as string);
  return [b.length & 255, (b.length >> 8) & 255, (b.length >> 16) & 255, (b.length >>> 24) & 255, ...b];
}
/** Statistics bounds are PLAIN without the length prefix for byte arrays. */
const statBytes = (v: number | string, type: 'int64' | 'string') => new Uint8Array(type === 'int64' ? plain(v, type) : enc.encode(v as string));

function compare(a: number | string, b: number | string, type: 'int64' | 'string') {
  if (type === 'int64') return (a as number) - (b as number);
  const x = enc.encode(a as string);
  const y = enc.encode(b as string);
  for (let i = 0; i < Math.min(x.length, y.length); i++) if (x[i] !== y[i]) return x[i] - y[i];
  return x.length - y.length;
}

function lenPrefixed(body: number[]) {
  const n = body.length;
  return [n & 255, (n >> 8) & 255, (n >> 16) & 255, (n >>> 24) & 255, ...body];
}

/* ------------------------------------------------------------------ the writer */

export type WriteOpts = { rowsPerGroup: number; rowsPerPage: number; dictionary: boolean; pageIndex: boolean };

export type PageInfo = {
  rg: number;
  col: number;
  kind: 'dict' | 'data';
  offset: number;
  headerLen: number;
  bodyLen: number;
  numValues: number;
  nulls: number;
  firstRow: number;
  rows: number;
  min: number | string | null;
  max: number | string | null;
  repBytes: number;
  defBytes: number;
  valueBytes: number;
  encoding: 'PLAIN' | 'RLE_DICTIONARY';
};

export type ChunkInfo = {
  rg: number;
  col: number;
  start: number;
  len: number;
  dictPageOffset: number | null;
  dataPageOffset: number;
  numValues: number;
  nullCount: number;
  min: number | string | null;
  max: number | string | null;
  ci: [number, number] | null;
  oi: [number, number] | null;
};

export type Region = { kind: 'magic' | 'dict' | 'data' | 'colindex' | 'offindex' | 'footer' | 'footerlen'; start: number; len: number; rg?: number; col?: number };

export type ParquetFile = {
  bytes: Uint8Array;
  regions: Region[];
  pages: PageInfo[];
  chunks: ChunkInfo[];
  rowGroups: { firstRow: number; rows: number; start: number; len: number }[];
  footerStart: number;
  footerLen: number;
};

export function writeParquet(schema: SNode, cols: Entry[][], nRows: number, opts: WriteOpts): ParquetFile {
  const leaves = leavesOf(schema);
  const out: number[] = [];
  const regions: Region[] = [];
  const pages: PageInfo[] = [];
  const chunks: ChunkInfo[] = [];
  const rowGroups: ParquetFile['rowGroups'] = [];
  const push = (bytes: number[]) => {
    for (const b of bytes) out.push(b);
  };
  push([0x50, 0x41, 0x52, 0x31]);
  regions.push({ kind: 'magic', start: 0, len: 4 });

  const rpg = Math.max(1, Math.floor(opts.rowsPerGroup));
  const rpp = Math.max(1, Math.floor(opts.rowsPerPage));
  for (let rg = 0, firstRow = 0; firstRow < nRows; rg++, firstRow += rpg) {
    const rows = Math.min(rpg, nRows - firstRow);
    const rgStart = out.length;
    for (const leaf of leaves) {
      const entries = cols[leaf.index].filter((e) => e.row >= firstRow && e.row < firstRow + rows);
      const chunkStart = out.length;
      const nonNull = entries.filter((e) => e.v !== null).map((e) => e.v as number | string);
      let dict: (number | string)[] = [];
      const dictIndex = new Map<number | string, number>();
      let dictPageOffset: number | null = null;
      if (opts.dictionary) {
        for (const v of nonNull)
          if (!dictIndex.has(v)) {
            dictIndex.set(v, dict.length);
            dict.push(v);
          }
        const body = dict.flatMap((v) => plain(v, leaf.type));
        const header: number[] = [];
        writeStruct(header, [
          [1, i32(2)],
          [2, i32(body.length)],
          [3, i32(body.length)],
          [7, st([1, i32(dict.length)], [2, i32(0)])],
        ]);
        dictPageOffset = out.length;
        pages.push({ rg, col: leaf.index, kind: 'dict', offset: out.length, headerLen: header.length, bodyLen: body.length, numValues: dict.length, nulls: 0, firstRow: 0, rows: 0, min: null, max: null, repBytes: 0, defBytes: 0, valueBytes: body.length, encoding: 'PLAIN' });
        regions.push({ kind: 'dict', start: out.length, len: header.length + body.length, rg, col: leaf.index });
        push(header);
        push(body);
      }
      const dataPageOffset = out.length;
      const colPages: PageInfo[] = [];
      for (let p = 0; p * rpp < rows; p++) {
        const lo = firstRow + p * rpp;
        const pe = entries.filter((e) => e.row >= lo && e.row < lo + rpp);
        const vals = pe.filter((e) => e.v !== null).map((e) => e.v as number | string);
        const rep = leaf.maxRep > 0 ? lenPrefixed(hybrid(pe.map((e) => e.r), bitWidth(leaf.maxRep))) : [];
        const def = leaf.maxDef > 0 ? lenPrefixed(hybrid(pe.map((e) => e.d), bitWidth(leaf.maxDef))) : [];
        let values: number[];
        if (opts.dictionary) {
          const w = Math.max(1, bitWidth(dict.length - 1));
          values = [w, ...hybrid(vals.map((v) => dictIndex.get(v)!), w)];
        } else values = vals.flatMap((v) => plain(v, leaf.type));
        const body = [...rep, ...def, ...values];
        let min: number | string | null = null;
        let max: number | string | null = null;
        for (const v of vals) {
          if (min === null || compare(v, min, leaf.type) < 0) min = v;
          if (max === null || compare(v, max, leaf.type) > 0) max = v;
        }
        const dph: TField[] = [
          [1, i32(pe.length)],
          [2, i32(opts.dictionary ? 8 : 0)],
          [3, i32(3)],
          [4, i32(3)],
        ];
        if (!opts.pageIndex) {
          const s: TField[] = [[3, i64(pe.length - vals.length)]];
          if (max !== null && min !== null) s.push([5, bin(statBytes(max, leaf.type))], [6, bin(statBytes(min, leaf.type))]);
          dph.push([5, st(...s)]);
        }
        const header: number[] = [];
        writeStruct(header, [
          [1, i32(0)],
          [2, i32(body.length)],
          [3, i32(body.length)],
          [5, st(...dph)],
        ]);
        const info: PageInfo = { rg, col: leaf.index, kind: 'data', offset: out.length, headerLen: header.length, bodyLen: body.length, numValues: pe.length, nulls: pe.length - vals.length, firstRow: p * rpp, rows: Math.min(rpp, rows - p * rpp), min, max, repBytes: rep.length, defBytes: def.length, valueBytes: values.length, encoding: opts.dictionary ? 'RLE_DICTIONARY' : 'PLAIN' };
        pages.push(info);
        colPages.push(info);
        regions.push({ kind: 'data', start: out.length, len: header.length + body.length, rg, col: leaf.index });
        push(header);
        push(body);
      }
      let min: number | string | null = null;
      let max: number | string | null = null;
      for (const pg of colPages) {
        if (pg.min !== null && (min === null || compare(pg.min, min, leaf.type) < 0)) min = pg.min;
        if (pg.max !== null && (max === null || compare(pg.max, max, leaf.type) > 0)) max = pg.max;
      }
      chunks.push({ rg, col: leaf.index, start: chunkStart, len: out.length - chunkStart, dictPageOffset, dataPageOffset, numValues: entries.length, nullCount: entries.length - nonNull.length, min, max, ci: null, oi: null });
    }
    rowGroups.push({ firstRow, rows, start: rgStart, len: out.length - rgStart });
  }

  if (opts.pageIndex) {
    for (const ch of chunks) {
      const ps = pages.filter((p) => p.rg === ch.rg && p.col === ch.col && p.kind === 'data');
      const type = leaves[ch.col].type;
      const nonNullPages = ps.filter((p) => p.min !== null);
      let asc = true;
      let desc = true;
      for (let i = 1; i < nonNullPages.length; i++) {
        const a = nonNullPages[i - 1];
        const b = nonNullPages[i];
        if (compare(b.min!, a.min!, type) < 0 || compare(b.max!, a.max!, type) < 0) asc = false;
        if (compare(b.min!, a.min!, type) > 0 || compare(b.max!, a.max!, type) > 0) desc = false;
      }
      const buf: number[] = [];
      writeStruct(buf, [
        [1, { t: 'list', et: 'bool', v: ps.map((p) => ({ t: 'bool', v: p.min === null }) as TVal) }],
        [2, { t: 'list', et: 'bin', v: ps.map((p) => bin(p.min === null ? new Uint8Array(0) : statBytes(p.min, type))) }],
        [3, { t: 'list', et: 'bin', v: ps.map((p) => bin(p.max === null ? new Uint8Array(0) : statBytes(p.max, type))) }],
        [4, i32(asc ? 1 : desc ? 2 : 0)],
        [5, { t: 'list', et: 'i64', v: ps.map((p) => i64(p.nulls)) }],
      ]);
      ch.ci = [out.length, buf.length];
      regions.push({ kind: 'colindex', start: out.length, len: buf.length, rg: ch.rg, col: ch.col });
      push(buf);
    }
    for (const ch of chunks) {
      const ps = pages.filter((p) => p.rg === ch.rg && p.col === ch.col && p.kind === 'data');
      const buf: number[] = [];
      writeStruct(buf, [[1, { t: 'list', et: 'struct', v: ps.map((p) => st([1, i64(p.offset)], [2, i32(p.headerLen + p.bodyLen)], [3, i64(p.firstRow)])) }]]);
      ch.oi = [out.length, buf.length];
      regions.push({ kind: 'offindex', start: out.length, len: buf.length, rg: ch.rg, col: ch.col });
      push(buf);
    }
  }

  const schemaElems: TVal[] = [];
  const walk = (node: SNode, root: boolean) => {
    const f: TField[] = [];
    if (!node.children) f.push([1, i32(node.type === 'int64' ? 2 : 6)]);
    if (!root) f.push([3, i32(node.rep === 'required' ? 0 : node.rep === 'optional' ? 1 : 2)]);
    f.push([4, bin(root ? 'schema' : node.name)]);
    if (node.children) f.push([5, i32(node.children.length)]);
    if (!node.children && node.type === 'string') f.push([6, i32(0)], [10, st([1, st()])]);
    if (node.list) f.push([6, i32(3)], [10, st([3, st()])]);
    schemaElems.push(st(...f));
    for (const c of node.children ?? []) walk(c, false);
  };
  walk(schema, true);

  const footerFields: TField[] = [
    [1, i32(1)],
    [2, { t: 'list', et: 'struct', v: schemaElems }],
    [3, i64(nRows)],
    [
      4,
      {
        t: 'list',
        et: 'struct',
        v: rowGroups.map((g, rg) => {
          const cs = chunks.filter((c) => c.rg === rg);
          return st(
            [
              1,
              {
                t: 'list',
                et: 'struct',
                v: cs.map((c) => {
                  const leaf = leaves[c.col];
                  const stats: TField[] = [[3, i64(c.nullCount)]];
                  if (c.max !== null && c.min !== null) stats.push([5, bin(statBytes(c.max, leaf.type))], [6, bin(statBytes(c.min, leaf.type))]);
                  const md: TField[] = [
                    [1, i32(leaf.type === 'int64' ? 2 : 6)],
                    [2, { t: 'list', et: 'i32', v: (opts.dictionary ? [0, 3, 8] : [0, 3]).map(i32) }],
                    [3, { t: 'list', et: 'bin', v: leaf.path.map((n) => bin(n.name)) }],
                    [4, i32(0)],
                    [5, i64(c.numValues)],
                    [6, i64(c.len)],
                    [7, i64(c.len)],
                    [9, i64(c.dataPageOffset)],
                  ];
                  if (c.dictPageOffset !== null) md.push([11, i64(c.dictPageOffset)]);
                  md.push([12, st(...stats)]);
                  const cc: TField[] = [
                    [2, i64(0)],
                    [3, st(...md)],
                  ];
                  if (c.oi) cc.push([4, i64(c.oi[0])], [5, i32(c.oi[1])]);
                  if (c.ci) cc.push([6, i64(c.ci[0])], [7, i32(c.ci[1])]);
                  return st(...cc);
                }),
              },
            ],
            [2, i64(g.len)],
            [3, i64(g.rows)],
            [5, i64(g.start)],
            [6, i64(g.len)],
            [7, { t: 'i16', v: rg }],
          );
        }),
      },
    ],
    [6, bin('db-internals shredder lab')],
    [7, { t: 'list', et: 'struct', v: leaves.map(() => st([1, st()])) }],
  ];
  const footer: number[] = [];
  writeStruct(footer, footerFields);
  const footerStart = out.length;
  regions.push({ kind: 'footer', start: out.length, len: footer.length });
  push(footer);
  const n = footer.length;
  regions.push({ kind: 'footerlen', start: out.length, len: 8 });
  push([n & 255, (n >> 8) & 255, (n >> 16) & 255, (n >>> 24) & 255, 0x50, 0x41, 0x52, 0x31]);
  return { bytes: new Uint8Array(out), regions, pages, chunks, rowGroups, footerStart, footerLen: footer.length };
}

export type Fetch = { what: string; start: number; len: number };

/** The ranged reads a footer-first reader issues to scan the selected leaf columns in full. */
export function readPlan(file: ParquetFile, selected: number[], leaves: Leaf[]): Fetch[] {
  const size = file.bytes.length;
  const plan: Fetch[] = [
    { what: 'last 8 bytes: footer length + "PAR1"', start: size - 8, len: 8 },
    { what: `footer: FileMetaData (${file.footerLen} bytes, Thrift compact)`, start: file.footerStart, len: file.footerLen },
  ];
  const sel = new Set(selected);
  for (const c of file.chunks) if (sel.has(c.col)) plan.push({ what: `row group ${c.rg}, ${leaves[c.col].id}`, start: c.start, len: c.len });
  return plan;
}

/* ================================================================== UI */

export function definedNames(f: Leaf, d: number) {
  return f.path.slice(0, definedDepth(f, d)).map((n) => n.name);
}

export function pretty(v: Json, ind = ''): string {
  if (Array.isArray(v)) {
    if (!v.length) return '[]';
    if (v.every((x) => x === null || typeof x !== 'object')) return `[${v.map((x) => JSON.stringify(x)).join(', ')}]`;
    return `[\n${v.map((x) => `${ind}  ${pretty(x, `${ind}  `)}`).join(',\n')}\n${ind}]`;
  }
  if (v !== null && typeof v === 'object') {
    const ks = Object.keys(v);
    if (!ks.length) return '{}';
    const inner = ks.map((k) => `"${k}": ${pretty(v[k], `${ind}  `)}`);
    const one = `{ ${inner.join(', ')} }`;
    if (one.length <= 44 && !one.includes('\n')) return one;
    return `{\n${inner.map((s) => `${ind}  ${s}`).join(',\n')}\n${ind}}`;
  }
  return JSON.stringify(v);
}

export function schemaText(schema: SNode) {
  const lines: string[] = [];
  const walk = (n: SNode, ind: string) => {
    for (const c of n.children ?? []) {
      if (c.children) {
        lines.push(`${ind}${c.rep} group ${c.name}${c.list ? ' (LIST)' : ''} {`);
        walk(c, `${ind}  `);
        lines.push(`${ind}}`);
      } else lines.push(`${ind}${c.rep} ${c.type === 'int64' ? 'int64' : 'binary'} ${c.name}${c.type === 'string' ? ' (STRING)' : ''};`);
    }
  };
  lines.push(`message ${schema.name} {`);
  walk(schema, '  ');
  lines.push('}');
  return lines.join('\n');
}

export type DatasetKey = 'paper' | 'sparse' | 'list';
export const DATASETS: Record<DatasetKey, { label: string; schema: SNode; text: string }> = {
  paper: {
    label: 'Dremel paper: records r1 and r2',
    schema: DOCUMENT,
    text: `[
  {
    "DocId": 10,
    "Links": { "Forward": [20, 40, 60] },
    "Name": [
      { "Language": [{ "Code": "en-us", "Country": "us" }, { "Code": "en" }],
        "Url": "http://A" },
      { "Url": "http://B" },
      { "Language": [{ "Code": "en-gb", "Country": "gb" }] }
    ]
  },
  {
    "DocId": 20,
    "Links": { "Backward": [10, 30], "Forward": [80] },
    "Name": [{ "Url": "http://C" }]
  }
]`,
  },
  sparse: {
    label: 'Document: missing and empty groups',
    schema: DOCUMENT,
    text: `[
  { "DocId": 30 },
  { "DocId": 40, "Links": {}, "Name": [{ "Language": [] }, {}] },
  { "DocId": 50, "Links": { "Backward": [7] },
    "Name": [{ "Language": [{ "Code": "fr" }, { "Code": "de", "Country": "de" }] }] }
]`,
  },
  list: {
    label: 'Three-level LIST<string>, as pyarrow writes it',
    schema: TAGGED,
    text: `[
  { "id": 1, "tags": ["a", "b"] },
  { "id": 2, "tags": [] },
  { "id": 3, "tags": null },
  { "id": 4, "tags": ["c", null] }
]`,
  },
};

const KIND: Record<Region['kind'], { label: string; color: string }> = {
  magic: { label: '"PAR1" magic', color: 'var(--viz-axis)' },
  footerlen: { label: 'footer length + "PAR1"', color: 'var(--viz-axis)' },
  dict: { label: 'Dictionary page', color: 'var(--viz-3)' },
  data: { label: 'Data page', color: 'var(--viz-1)' },
  colindex: { label: 'ColumnIndex', color: 'var(--viz-4)' },
  offindex: { label: 'OffsetIndex', color: 'var(--viz-5)' },
  footer: { label: 'Footer: FileMetaData', color: 'var(--viz-7)' },
};

const NOW = 'var(--viz-2)';
const mono = { fontFamily: 'var(--sl-font-mono, monospace)' } as const;
const fmtVal = (v: number | string | null) => (v === null ? 'NULL' : typeof v === 'string' ? `'${v}'` : String(v));

function describe(s: AsmStep, fsm: Fsm) {
  const f = fsm.fields[s.field];
  const head = `${f.id} value ${s.entry} = ${fmtVal(s.e.v)} (r=${s.e.r}, d=${s.e.d}).`;
  let body: string;
  if (s.e.v !== null) body = s.opened.length ? ` Open ${s.opened.join(' › ')} and append the value.` : ' Every record on its path is already open: append the value.';
  else if (s.e.d === 0) body = ` d=0: nothing on its path exists in this record, so nothing is added.`;
  else {
    const names = definedNames(f, s.e.d);
    body = ` d=${s.e.d} of ${f.maxDef}: ${names.join(' and ')} ${names.length === 1 ? 'exists' : 'exist'} but the value does not${s.opened.length ? `, so open ${s.opened.join(' › ')} and append nothing` : ', and nothing new needs opening'}.`;
  }
  const to = s.to === fsm.fields.length ? 'end of record' : fsm.fields[s.to].id;
  const next = s.exhausted ? ' The column has no more values, so the next level counts as 0' : ` The next value in this column has r=${s.nextLevel}`;
  const where = s.to === fsm.fields.length ? ': the record is complete.' : s.back ? `: a back-transition to ${to}${s.closed ? `, closing ${s.closed} so the next value starts a new one` : ''}.` : ` → ${to}.`;
  return head + body + next + where;
}

export default function DremelRecordShredderLab() {
  const [dataset, setDataset] = useState<DatasetKey>('paper');
  const [text, setText] = useState(DATASETS.paper.text);
  const schema = DATASETS[dataset].schema;
  const leaves = useMemo(() => leavesOf(schema), [schema]);
  const [selected, setSelected] = useState<number[]>(() => leavesOf(DOCUMENT).map((l) => l.index));
  const [step, setStep] = useState(9999);
  const [rowsPerGroup, setRowsPerGroup] = useState(12);
  const [rowsPerPage, setRowsPerPage] = useState(12);
  const [dictionary, setDictionary] = useState(true);
  const [pageIndex, setPageIndex] = useState(false);
  const [inspect, setInspect] = useState(4);
  const [inspectRg, setInspectRg] = useState(0);

  const parsed = useMemo(() => parseRecords(schema, text), [schema, text]);
  const [good, setGood] = useState<{ schema: SNode; records: Obj[] }>(() => {
    const p = parseRecords(DOCUMENT, DATASETS.paper.text);
    return { schema: DOCUMENT, records: p.ok ? p.records : [] };
  });
  useEffect(() => {
    if (parsed.ok) setGood({ schema, records: parsed.records });
  }, [parsed, schema]);
  const records = parsed.ok ? parsed.records : good.schema === schema ? good.records : [];
  const cols = useMemo(() => shred(schema, records), [schema, records]);
  const asm = useMemo(() => assemble(schema, cols, selected, step), [schema, cols, selected, step]);
  const fullSteps = useMemo(() => assemble(schema, cols, selected).total, [schema, cols, selected]);
  const nRows = records.length;
  const rpg = Math.min(rowsPerGroup, Math.max(1, nRows));
  const rpp = Math.min(rowsPerPage, rpg);
  const file = useMemo(() => writeParquet(schema, cols, nRows, { rowsPerGroup: rpg, rowsPerPage: rpp, dictionary, pageIndex }), [schema, cols, nRows, rpg, rpp, dictionary, pageIndex]);
  const plan = readPlan(file, selected, leaves);
  const planBytes = plan.reduce((s, p) => s + p.len, 0);
  const inspectLeaf = leaves[Math.min(inspect, leaves.length - 1)];
  const rgCount = file.rowGroups.length;
  const rgSel = Math.min(inspectRg, rgCount - 1);
  const chunk = file.chunks.find((c) => c.rg === rgSel && c.col === inspectLeaf.index)!;
  const chunkPages = file.pages.filter((p) => p.rg === rgSel && p.col === inspectLeaf.index);

  const stepping = step < fullSteps;
  const shownSteps = asm.steps;
  const last = shownSteps[shownSteps.length - 1];
  const fsm = asm.fsm;
  const cursorField = stepping ? fsm.fields[asm.current] : undefined;
  const consumed = new Map<number, number>();
  for (const s of shownSteps) {
    const li = fsm.fields[s.field].index;
    consumed.set(li, Math.max(consumed.get(li) ?? 0, s.entry + 1));
  }

  const chooseDataset = (k: DatasetKey) => {
    setDataset(k);
    setText(DATASETS[k].text);
    setSelected(leavesOf(DATASETS[k].schema).map((l) => l.index));
    setStep(9999);
    setRowsPerGroup(12);
    setRowsPerPage(12);
    setInspect(k === 'list' ? 1 : 4);
    setInspectRg(0);
  };
  const toggle = (i: number, on: boolean) => {
    setSelected((s) => (on ? [...new Set([...s, i])].sort((a, b) => a - b) : s.filter((x) => x !== i)));
    setStep(9999);
  };

  /* ---------------- FSM geometry */
  const NODE_W = 150;
  const NODE_H = 28;
  const GAP = 48;
  const n = fsm.fields.length;
  const yOf = (i: number) => 10 + i * GAP;
  const backEdges: { from: number; to: number; levels: number[] }[] = [];
  const fwdLevels: number[][] = fsm.fields.map(() => []);
  fsm.trans.forEach((row, i) =>
    row.forEach((t, l) => {
      if (t === i + 1) fwdLevels[i].push(l);
      else {
        const e = backEdges.find((b) => b.from === i && b.to === t);
        if (e) e.levels.push(l);
        else backEdges.push({ from: i, to: t, levels: [l] });
      }
    }),
  );
  backEdges.sort((a, b) => a.from - a.to - (b.from - b.to));
  const lanes = new Map<string, number>();
  let lane = 0;
  for (const e of backEdges) if (e.from !== e.to) lanes.set(`${e.from}-${e.to}`, lane++);
  const FSM_W = 10 + NODE_W + 56 + 24 * lane + 18;
  const FSM_H = yOf(n) + NODE_H + 8;
  const taken = last && stepping ? { from: last.field, to: last.to } : null;

  /* ---------------- byte map geometry */
  const MAP_W = 680;
  const X0 = 8;
  const size = file.bytes.length;
  const bx = (b: number) => X0 + (b / size) * (MAP_W - 2 * X0);
  const planRanges = plan.map((p, i) => ({ ...p, i }));

  const summaryNote = !parsed.ok ? (
    <>
      <strong>Can’t read that JSON:</strong> {parsed.error} The lab keeps showing the last valid records.
    </>
  ) : !selected.length ? (
    <>Tick at least one column to assemble records from it.</>
  ) : stepping && last ? (
    <>
      <strong>Read {shownSteps.length} of {fullSteps} values.</strong> {asm.records.length} record{asm.records.length === 1 ? '' : 's'} started; open now: {asm.openPath.length ? asm.openPath.join(' › ') : 'none (between records)'}. The next read is from {stepping && asm.current < fsm.fields.length ? fsm.fields[asm.current].id : '—'}.
    </>
  ) : stepping ? (
    <>
      <strong>Ready.</strong> The FSM starts at {fsm.fields[0].id}, the first selected field. Press Next read.
    </>
  ) : (
    <>
      <strong>
        Assembled {asm.records.length} record{asm.records.length === 1 ? '' : 's'} from {selected.length} of {leaves.length} columns by reading {fullSteps} values.
      </strong>{' '}
      A reader scanning those columns fetches {plan.length} byte ranges — the 8-byte tail, the {file.footerLen}-byte footer and {plan.length - 2} column chunk{plan.length - 2 === 1 ? '' : 's'} — {fmtNum(planBytes)} of the file’s {fmtNum(size)} bytes.
    </>
  );

  return (
    <VizPanel
      title="Record shredder: nested records to Parquet columns and back"
      subtitle="Edit the records. Each leaf field becomes a column of values with repetition (r) and definition (d) levels; tick columns to build the assembly automaton and step it; the file below is a real, uncompressed Parquet file built from the same columns."
      controls={
        <>
          <Choice label="Records" value={dataset} onChange={chooseDataset} options={(Object.keys(DATASETS) as DatasetKey[]).map((k) => ({ value: k, label: DATASETS[k].label }))} />
          <Slider label="Rows per row group" min={1} max={Math.max(1, nRows)} value={rpg} onChange={(v) => { setRowsPerGroup(v); setInspectRg(0); }} />
          <Slider label="Rows per data page" min={1} max={Math.max(1, rpg)} value={rpp} onChange={setRowsPerPage} />
          <Check label="Dictionary encoding" checked={dictionary} onChange={setDictionary} />
          <Check label="Write page index" checked={pageIndex} onChange={setPageIndex} />
        </>
      }
      legend={
        <Legend
          items={[
            { label: 'Value being read / transition taken', color: NOW },
            ...(['dict', 'data', 'colindex', 'offindex', 'footer'] as const).map((k) => ({ label: KIND[k].label, color: KIND[k].color })),
            { label: 'Magic bytes and footer length', color: KIND.magic.color },
          ]}
        />
      }
      stats={
        <Stats
          items={[
            { label: 'Records', value: fmtNum(nRows) },
            { label: 'Values read by the FSM', value: `${fmtNum(shownSteps.length)} / ${fmtNum(fullSteps)}`, hint: 'One read per entry in the selected columns, NULL placeholders included.' },
            { label: 'File size', value: `${fmtNum(size)} B`, hint: 'Uncompressed pages. Real files are usually compressed per page.' },
            { label: 'Footer', value: `${fmtNum(file.footerLen)} B`, hint: 'Thrift-compact FileMetaData: schema, row groups, column chunk offsets and statistics.' },
            { label: 'Fetched to scan selected columns', value: `${fmtNum(planBytes)} B · ${plan.length} reads`, hint: 'Tail read of 8 bytes, the footer, then one range per selected column chunk. Arrow C++ instead reads the last 64 KB in one request, which for a file this small is the whole file.' },
          ]}
        />
      }
      note={<Note>{summaryNote}</Note>}
      table={
        <>
        <table className="viz-table">
          <thead>
            <tr>
              <th>Column</th>
              <th>Value</th>
              <th>r</th>
              <th>d</th>
              <th>Row</th>
            </tr>
          </thead>
          <tbody>
            {leaves.flatMap((l) =>
              cols[l.index].map((e, i) => (
                <tr key={`${l.index}-${i}`}>
                  <td>{i === 0 ? l.id : ''}</td>
                  <td>{fmtVal(e.v)}</td>
                  <td>{e.r}</td>
                  <td>{e.d}</td>
                  <td>{e.row}</td>
                </tr>
              )),
            )}
          </tbody>
        </table>
        <table className="viz-table">
          <thead>
            <tr>
              <th>File region</th>
              <th>Row group</th>
              <th>Column</th>
              <th>Bytes</th>
              <th>Length</th>
            </tr>
          </thead>
          <tbody>
            {file.regions.map((g, i) => (
              <tr key={i}>
                <td>{KIND[g.kind].label}</td>
                <td>{g.rg ?? '—'}</td>
                <td>{g.col !== undefined ? leaves[g.col].id : '—'}</td>
                <td>
                  {g.start}–{g.start + g.len - 1}
                </td>
                <td>{g.len}</td>
              </tr>
            ))}
          </tbody>
        </table>
        </>
      }
    >
      <div style={{ display: 'flex', flexWrap: 'wrap', gap: 12, alignItems: 'stretch' }}>
        <label className="viz-control" style={{ flex: '1 1 320px', minWidth: 0 }}>
          <span>Records as JSON (edit freely)</span>
          <textarea
            value={text}
            onChange={(e) => {
              setText(e.currentTarget.value);
              setStep(9999);
            }}
            rows={13}
            spellCheck={false}
            aria-label="Records as JSON"
            style={{ ...mono, font: 'inherit', fontSize: '0.75rem', background: 'var(--viz-plane)', color: 'var(--viz-ink)', border: `1px solid ${parsed.ok ? 'var(--viz-border)' : 'var(--viz-critical)'}`, borderRadius: 6, padding: 6, width: '100%', boxSizing: 'border-box', resize: 'vertical' }}
          />
        </label>
        <div className="viz-control" style={{ flex: '1 1 240px', minWidth: 0 }}>
          <span>Schema (fixed for this data set)</span>
          <pre style={{ ...mono, margin: 0, fontSize: '0.72rem', background: 'var(--viz-plane)', color: 'var(--viz-ink)', border: '1px solid var(--viz-border)', borderRadius: 6, padding: 6, overflowX: 'auto' }}>{schemaText(schema)}</pre>
        </div>
      </div>

      <p className="viz-sub" style={{ margin: '0.9rem 0 0.35rem' }}>
        Column stripes. Tick the columns the query reads.
      </p>
      <div style={{ display: 'flex', flexWrap: 'wrap', gap: 8 }}>
        {leaves.map((l) => {
          const on = selected.includes(l.index);
          const used = consumed.get(l.index) ?? 0;
          const nextHere = cursorField?.index === l.index;
          return (
            <div key={l.index} style={{ border: '1px solid var(--viz-border)', borderRadius: 8, padding: '4px 6px', background: 'var(--viz-plane)', opacity: on ? 1 : 0.55, flex: '0 1 auto', minWidth: 132 }}>
              <Check label={l.id} checked={on} onChange={(v) => toggle(l.index, v)} />
              <div style={{ fontSize: '0.68rem', color: 'var(--viz-ink-2)', margin: '1px 0 2px' }}>
                max r={l.maxRep}, max d={l.maxDef}
              </div>
              <table className="viz-table" style={{ marginTop: 0, width: 'auto' }}>
                <thead>
                  <tr>
                    <th>value</th>
                    <th>r</th>
                    <th>d</th>
                  </tr>
                </thead>
                <tbody>
                  {cols[l.index].map((e, i) => {
                    const justRead = stepping && last && fsm.fields[last.field].index === l.index && last.entry === i;
                    const upNext = nextHere && i === used;
                    return (
                      <tr key={i} style={{ boxShadow: justRead ? `inset 3px 0 0 ${NOW}` : upNext ? 'inset 3px 0 0 var(--viz-ink-muted)' : undefined, fontWeight: justRead ? 700 : 400, color: stepping && on && i >= used ? 'var(--viz-ink-2)' : 'var(--viz-ink)' }}>
                        <td style={mono}>{fmtVal(e.v)}</td>
                        <td>{e.r}</td>
                        <td>{e.d}</td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          );
        })}
      </div>

      <div className="viz-controls" style={{ marginTop: '0.9rem' }}>
        <Button onClick={() => setStep(0)} disabled={!selected.length}>
          Assemble step by step
        </Button>
        <Button primary onClick={() => setStep((s) => Math.min(fullSteps, (s >= fullSteps ? 0 : s) + 1))} disabled={!stepping}>
          Next read
        </Button>
        <Button onClick={() => setStep(9999)} disabled={!stepping}>
          Finish
        </Button>
      </div>

      <p role="status" style={{ margin: '0.2rem 0 0.5rem', fontSize: '0.8rem', color: 'var(--viz-ink)', minHeight: '2.4em', borderLeft: `3px solid ${stepping ? NOW : 'var(--viz-border)'}`, paddingLeft: 8 }}>
        {!selected.length
          ? 'Tick at least one column.'
          : stepping && last
            ? `Read ${shownSteps.length} of ${fullSteps}: ${describe(last, fsm)}`
            : stepping
              ? `The automaton starts at ${fsm.fields[0].id}, the first ticked field. Press Next read.`
              : `One state per ticked column, in schema order. Solid arrows go on to the next field; dashed arrows jump back when the next repetition level says an enclosing repeated field repeats. Press Assemble step by step.`}
      </p>
      <div style={{ display: 'flex', flexWrap: 'wrap', gap: 12, alignItems: 'flex-start' }}>
        <svg viewBox={`0 0 ${FSM_W} ${FSM_H}`} width={FSM_W} height={FSM_H} role="img" aria-label={`Assembly automaton with ${n} field states: ${fsm.fields.map((f, i) => `${f.id}: ${fsm.trans[i].map((t, l) => `r${l} to ${t === n ? 'end' : fsm.fields[t].id}`).join(', ')}`).join('; ')}`} style={{ flex: '0 0 auto' }}>
          <defs>
            <marker id="dremel-arrow" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="8" markerHeight="8" markerUnits="userSpaceOnUse" orient="auto-start-reverse">
              <path d="M0,0 L8,4 L0,8 z" fill="var(--viz-ink-2)" />
            </marker>
            <marker id="dremel-arrow-now" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="9" markerHeight="9" markerUnits="userSpaceOnUse" orient="auto-start-reverse">
              <path d="M0,0 L8,4 L0,8 z" fill={NOW} />
            </marker>
          </defs>
          {fsm.fields.map((f, i) => {
            const y = yOf(i);
            const isTaken = taken && taken.from === i && taken.to === i + 1;
            return (
              <g key={f.index}>
                <line x1={10 + NODE_W / 2} x2={10 + NODE_W / 2} y1={y + NODE_H} y2={yOf(i + 1) - 2} stroke={isTaken ? NOW : 'var(--viz-ink-2)'} strokeWidth={isTaken ? 2.5 : 1.2} markerEnd={`url(#${isTaken ? 'dremel-arrow-now' : 'dremel-arrow'})`} />
                <text x={16 + NODE_W / 2} y={y + NODE_H + 14} fontSize={11} fill="var(--viz-ink)">
                  r = {fwdLevels[i].join(',')}
                </text>
              </g>
            );
          })}
          {backEdges.map((e) => {
            const isTaken = taken && taken.from === e.from && taken.to === e.to;
            const stroke = isTaken ? NOW : 'var(--viz-ink-2)';
            const x1 = 10 + NODE_W;
            if (e.from === e.to) {
              const y = yOf(e.from);
              return (
                <g key={`${e.from}-${e.to}`}>
                  <path d={`M${x1},${y + 8} C${x1 + 30},${y - 6} ${x1 + 30},${y + 34} ${x1 + 2},${y + 20}`} fill="none" stroke={stroke} strokeWidth={isTaken ? 2.5 : 1.2} strokeDasharray="4 3" markerEnd={`url(#${isTaken ? 'dremel-arrow-now' : 'dremel-arrow'})`} />
                  <text x={x1 + 30} y={y + 18} fontSize={11} fill="var(--viz-ink)">
                    r = {e.levels.join(',')}
                  </text>
                </g>
              );
            }
            const xl = x1 + 56 + 24 * (lanes.get(`${e.from}-${e.to}`) ?? 0);
            const ya = yOf(e.from) + 18;
            const yb = yOf(e.to) + 10;
            return (
              <g key={`${e.from}-${e.to}`}>
                <path d={`M${x1},${ya} H${xl} V${yb} H${x1 + 2}`} fill="none" stroke={stroke} strokeWidth={isTaken ? 2.5 : 1.2} strokeDasharray="4 3" markerEnd={`url(#${isTaken ? 'dremel-arrow-now' : 'dremel-arrow'})`} />
                <text x={xl + 4} y={(ya + yb) / 2 + 4} fontSize={11} fill="var(--viz-ink)">
                  {e.levels.join(',')}
                </text>
              </g>
            );
          })}
          {fsm.fields.map((f, i) => {
            const y = yOf(i);
            const now = stepping && asm.current === i;
            return (
              <g key={`n${f.index}`}>
                <rect x={10} y={y} width={NODE_W} height={NODE_H} rx={6} fill="var(--viz-surface)" stroke={now ? NOW : 'var(--viz-ink-2)'} strokeWidth={now ? 2.5 : 1} />
                <text x={10 + NODE_W / 2} y={y + 18} textAnchor="middle" fontSize={11} fill="var(--viz-ink)">
                  {f.id}
                </text>
              </g>
            );
          })}
          <rect x={10} y={yOf(n)} width={NODE_W} height={NODE_H} rx={14} fill="var(--viz-surface)" stroke="var(--viz-ink-muted)" strokeDasharray="3 2" />
          <text x={10 + NODE_W / 2} y={yOf(n) + 18} textAnchor="middle" fontSize={11} fill="var(--viz-ink-2)">
            end of record
          </text>
        </svg>
        <div style={{ flex: '1 1 240px', minWidth: 0 }}>
          <div style={{ fontSize: '0.75rem', color: 'var(--viz-ink-2)' }}>
            {stepping ? (
              <>
                Open records: <span style={{ ...mono, color: 'var(--viz-ink)' }}>{asm.openPath.length ? asm.openPath.join(' › ') : '— (between records)'}</span>
              </>
            ) : (
              <>Assembled records ({selected.length === leaves.length ? 'all columns' : 'selected columns only'})</>
            )}
          </div>
          <pre aria-label="Assembled records" style={{ ...mono, margin: '4px 0 0', fontSize: '0.72rem', background: 'var(--viz-plane)', color: 'var(--viz-ink)', border: '1px solid var(--viz-border)', borderRadius: 6, padding: 6, overflowX: 'auto', maxHeight: 340, overflowY: 'auto' }}>
            {pretty(asm.records.map((r) => toJson(schema, r)))}
          </pre>
        </div>
      </div>

      <p className="viz-sub" style={{ margin: '1rem 0 0.35rem' }}>
        The Parquet file: {fmtNum(size)} bytes, {rgCount} row group{rgCount === 1 ? '' : 's'}. Numbered brackets are the reads a footer-first reader issues to scan the ticked columns. Pages here are cut after a fixed number of rows; real writers cut at about 1 MB or 20,000 rows.
      </p>
      <svg viewBox={`0 0 ${MAP_W} 96`} width={MAP_W} height={96} role="img" aria-label={`Byte map of a ${size}-byte Parquet file: ${file.regions.map((r) => `${KIND[r.kind].label} at ${r.start}, ${r.len} bytes`).join('; ')}`}>
        {file.rowGroups.map((g, i) => (
          <g key={i}>
            <line x1={bx(g.start) + 1} x2={bx(g.start + g.len) - 1} y1={12} y2={12} stroke="var(--viz-ink-muted)" />
            <line x1={bx(g.start) + 1} x2={bx(g.start) + 1} y1={12} y2={18} stroke="var(--viz-ink-muted)" />
            <line x1={bx(g.start + g.len) - 1} x2={bx(g.start + g.len) - 1} y1={12} y2={18} stroke="var(--viz-ink-muted)" />
            {bx(g.start + g.len) - bx(g.start) > 70 ? (
              <text x={(bx(g.start) + bx(g.start + g.len)) / 2} y={9} textAnchor="middle" fontSize={10} fill="var(--viz-ink-2)">
                row group {i}
              </text>
            ) : null}
          </g>
        ))}
        {file.regions.map((r, i) => (
          <rect key={i} x={bx(r.start)} y={22} width={Math.max(1, bx(r.start + r.len) - bx(r.start) - 0.6)} height={26} fill={KIND[r.kind].color}>
            <title>{`${KIND[r.kind].label}${r.col !== undefined ? ` · ${leaves[r.col].id}` : ''}${r.rg !== undefined ? ` · row group ${r.rg}` : ''} · bytes ${r.start}–${r.start + r.len - 1}`}</title>
          </rect>
        ))}
        {chunk ? <rect x={bx(chunk.start) - 1} y={19} width={bx(chunk.start + chunk.len) - bx(chunk.start) + 1.4} height={32} fill="none" stroke="var(--viz-ink)" strokeWidth={1.6} strokeDasharray="3 2" /> : null}
        {planRanges.map((p) => {
          const xa = bx(p.start);
          const xb = Math.max(xa + 3, bx(p.start + p.len));
          const y = 58 + (p.i < 2 ? 0 : 0);
          return (
            <g key={p.i}>
              <path d={`M${xa + 0.5},${y} V${y + 6} H${xb - 0.5} V${y}`} fill="none" stroke="var(--viz-ink-2)" strokeWidth={1.2} />
              <text x={p.i === 0 ? xb - 2 : (xa + xb) / 2} y={y + 18} textAnchor={p.i === 0 ? 'end' : 'middle'} fontSize={10.5} fill="var(--viz-ink)">
                {p.i + 1}
              </text>
            </g>
          );
        })}
        <text x={X0} y={92} fontSize={10} fill="var(--viz-ink-muted)">
          byte 0
        </text>
        <text x={MAP_W - X0} y={92} textAnchor="end" fontSize={10} fill="var(--viz-ink-muted)">
          byte {fmtNum(size)}
        </text>
      </svg>

      <div style={{ marginTop: 6 }}>
        <div>
          <div className="viz-controls" style={{ marginBottom: 4 }}>
            <Choice label="Inspect column chunk" value={String(inspectLeaf.index)} onChange={(v) => setInspect(Number(v))} options={leaves.map((l) => ({ value: String(l.index), label: l.id }))} />
            {rgCount > 1 ? <Slider label="Row group" min={0} max={rgCount - 1} value={rgSel} onChange={setInspectRg} /> : null}
          </div>
          <div style={{ overflowX: 'auto' }}>
          <table className="viz-table" style={{ marginTop: 0 }}>
            <thead>
              <tr>
                <th>Page</th>
                <th>Offset</th>
                <th>Header + body</th>
                <th>num_values</th>
                <th>Rows</th>
                <th>Page stats</th>
              </tr>
            </thead>
            <tbody>
              {chunkPages.map((p, i) => (
                <tr key={i}>
                  <td>{p.kind === 'dict' ? 'DICTIONARY_PAGE' : `DATA_PAGE (${p.encoding})`}</td>
                  <td>{p.offset}</td>
                  <td title={p.kind === 'data' ? `repetition levels ${p.repBytes} B, definition levels ${p.defBytes} B, values ${p.valueBytes} B` : 'PLAIN-encoded dictionary entries'}>
                    {p.headerLen} + {p.bodyLen}
                    {p.kind === 'data' ? ` (r ${p.repBytes}, d ${p.defBytes}, v ${p.valueBytes})` : ''}
                  </td>
                  <td>{p.numValues}{p.kind === 'data' ? `, ${p.nulls} null` : ' entries'}</td>
                  <td>{p.kind === 'data' ? `first_row_index ${p.firstRow}, ${p.rows} row${p.rows === 1 ? '' : 's'}` : '—'}</td>
                  <td>{p.kind === 'dict' ? '—' : pageIndex ? 'in ColumnIndex' : p.min === null ? 'all null' : `${fmtVal(p.min)} … ${fmtVal(p.max)}`}</td>
                </tr>
              ))}
            </tbody>
          </table>
          </div>
        </div>
        <div style={{ marginTop: 8, overflowX: 'auto' }}>
          <div style={{ fontSize: '0.75rem', color: 'var(--viz-ink-2)', marginBottom: 2 }}>Footer entry for this chunk (ColumnChunk → ColumnMetaData)</div>
          <table className="viz-table" style={{ marginTop: 0 }}>
            <tbody>
              {(
                [
                  ['path_in_schema', inspectLeaf.path.map((p) => p.name).join(', ')],
                  ['type / codec', `${inspectLeaf.type === 'int64' ? 'INT64' : 'BYTE_ARRAY'} / UNCOMPRESSED`],
                  ['num_values', `${chunk.numValues} (statistics.null_count ${chunk.nullCount})`],
                  ['min_value / max_value', chunk.min === null ? 'absent: no non-null values' : `${fmtVal(chunk.min)} / ${fmtVal(chunk.max)}`],
                  ['dictionary_page_offset', chunk.dictPageOffset ?? 'absent'],
                  ['data_page_offset', chunk.dataPageOffset],
                  ['total_compressed_size', chunk.len],
                  ['column_index_offset / length', chunk.ci ? `${chunk.ci[0]} / ${chunk.ci[1]}` : 'absent'],
                  ['offset_index_offset / length', chunk.oi ? `${chunk.oi[0]} / ${chunk.oi[1]}` : 'absent'],
                ] as [string, string | number][]
              ).map(([k, v]) => (
                <tr key={k}>
                  <td style={mono}>{k}</td>
                  <td>{v}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
      <ol style={{ margin: '0.6rem 0 0', paddingLeft: '1.3rem', fontSize: '0.78rem', color: 'var(--viz-ink)' }} aria-label="Read plan">
        {plan.map((p, i) => (
          <li key={i}>
            <span style={mono}>
              bytes {p.start}–{p.start + p.len - 1}
            </span>{' '}
            ({p.len} B): {p.what}
          </li>
        ))}
      </ol>
    </VizPanel>
  );
}
