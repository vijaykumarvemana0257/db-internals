import { useMemo, useState } from 'react';
import { VizPanel, Choice, Slider, Check, Segmented, Legend, Stats, Note, fmtNum } from './Viz';
import { DATASETS, parseRecords, leavesOf, shred, type DatasetKey, type SNode, type Leaf } from './DremelRecordShredderLab';

/**
 * Arrow's in-memory layout for the same records the shredder lab uses, following the Arrow columnar spec:
 * - every array has its own validity bitmap (1 = valid, LSB bit order); like Arrow C++ and pyarrow, the model leaves
 *   it unallocated when the array has no nulls;
 * - List and Utf8 arrays have an offsets buffer of length + 1 int32 values; a List's offsets index its child array,
 *   a Utf8 array's offsets index its data bytes;
 * - a Struct has a validity bitmap and one child array per field, each as long as the struct;
 * - a dictionary array stores int32 indices plus a separate dictionary array.
 * Schema mapping (an assumption of this lab, the usual one): a Dremel `repeated` field becomes a non-nullable
 * List (missing and empty are the same thing in Dremel), an `optional` field or group becomes nullable, a LIST-annotated
 * group becomes a nullable List of nullable elements. Child slots under a null struct slot hold a null (nullable child)
 * or an empty/zero value (non-nullable child); the spec leaves those slots undefined.
 */

export type ArrowArr = {
  path: string;
  name: string;
  kind: 'int64' | 'utf8' | 'list' | 'struct' | 'dict';
  type: string;
  nullable: boolean;
  length: number;
  nullCount: number;
  validity: number[] | null;
  offsets?: number[];
  values?: number[];
  data?: number[];
  indices?: number[];
  dictionary?: ArrowArr;
  children: ArrowArr[];
};

const utf8 = new TextEncoder();
type O = { [k: string]: unknown };

function finish(a: Omit<ArrowArr, 'nullCount' | 'validity'> & { valid: boolean[] }): ArrowArr {
  const nullCount = a.valid.filter((v) => !v).length;
  const { valid, ...rest } = a;
  return { ...rest, nullCount, validity: nullCount ? valid.map((v) => (v ? 1 : 0)) : null };
}

function typeName(node: SNode, mode: 'field' | 'item'): string {
  if (mode === 'field' && node.rep === 'repeated') return `list<${typeName(node, 'item')}>`;
  if (node.list) return `list<${typeName(node.children![0].children![0], 'field')}>`;
  if (node.children) return `struct<${node.children.map((c) => c.name).join(', ')}>`;
  return node.type === 'int64' ? 'int64' : 'utf8';
}

function build(node: SNode, vals: unknown[], mode: 'field' | 'item', path: string, dict: boolean): ArrowArr {
  const name = mode === 'item' ? 'item' : node.name;
  if (mode === 'field' && node.rep === 'repeated') {
    const offsets = [0];
    const items: unknown[] = [];
    for (const v of vals) {
      for (const it of (v as unknown[] | undefined) ?? []) items.push(it);
      offsets.push(items.length);
    }
    return finish({ path, name, kind: 'list', type: typeName(node, 'field'), nullable: false, length: vals.length, valid: vals.map(() => true), offsets, children: [build(node, items, 'item', `${path}.item`, dict)] });
  }
  if (node.list) {
    const el = node.children![0].children![0];
    const offsets = [0];
    const items: unknown[] = [];
    const valid = vals.map((v) => v !== undefined && v !== null);
    for (const v of vals) {
      for (const it of ((v as O | undefined)?.list as O[] | undefined) ?? []) items.push(it.element === null ? undefined : it.element);
      offsets.push(items.length);
    }
    return finish({ path, name, kind: 'list', type: typeName(node, mode), nullable: true, length: vals.length, valid, offsets, children: [build(el, items, 'field', `${path}.element`, dict)] });
  }
  const nullable = mode === 'field' && node.rep === 'optional';
  const valid = vals.map((v) => !nullable || (v !== undefined && v !== null));
  if (node.children) {
    return finish({
      path,
      name,
      kind: 'struct',
      type: typeName(node, mode),
      nullable,
      length: vals.length,
      valid,
      children: node.children.map((c) => build(c, vals.map((v) => (v === undefined || v === null ? undefined : (v as O)[c.name])), 'field', `${path}.${c.name}`, dict)),
    });
  }
  if (node.type === 'int64') return finish({ path, name, kind: 'int64', type: 'int64', nullable, length: vals.length, valid, values: vals.map((v) => (typeof v === 'number' ? v : 0)), children: [] });
  const strings = vals.map((v) => (typeof v === 'string' ? v : ''));
  if (dict) {
    const dictVals: string[] = [];
    const idx = new Map<string, number>();
    const indices = vals.map((v, i) => {
      if (!valid[i] || typeof v !== 'string') return 0;
      if (!idx.has(v)) {
        idx.set(v, dictVals.length);
        dictVals.push(v);
      }
      return idx.get(v)!;
    });
    const dictionary = build({ name: 'dictionary', rep: 'required', type: 'string' }, dictVals, 'field', `${path}.dictionary`, false);
    return finish({ path, name, kind: 'dict', type: 'dictionary<int32, utf8>', nullable, length: vals.length, valid, indices, dictionary, children: [] });
  }
  const offsets = [0];
  const data: number[] = [];
  strings.forEach((s, i) => {
    if (valid[i]) for (const b of utf8.encode(s)) data.push(b);
    offsets.push(data.length);
  });
  return finish({ path, name, kind: 'utf8', type: 'utf8', nullable, length: vals.length, valid, offsets, data, children: [] });
}

/** The record batch's top-level columns. */
export function toArrow(schema: SNode, records: O[], dictionary: boolean): ArrowArr[] {
  return (schema.children ?? []).map((c) => build(c, records.map((r) => r[c.name]), 'field', c.name, dictionary));
}

export type Buf = { array: ArrowArr; kind: 'validity' | 'offsets' | 'values' | 'data' | 'indices'; bytes: number };

export function buffersOf(cols: ArrowArr[]): Buf[] {
  const out: Buf[] = [];
  const walk = (a: ArrowArr) => {
    if (a.validity) out.push({ array: a, kind: 'validity', bytes: Math.ceil(a.length / 8) });
    if (a.kind === 'list' || a.kind === 'utf8') out.push({ array: a, kind: 'offsets', bytes: 4 * (a.length + 1) });
    if (a.kind === 'int64') out.push({ array: a, kind: 'values', bytes: 8 * a.length });
    if (a.kind === 'utf8') out.push({ array: a, kind: 'data', bytes: a.data!.length });
    if (a.kind === 'dict') {
      out.push({ array: a, kind: 'indices', bytes: 4 * a.length });
      walk(a.dictionary!);
    }
    a.children.forEach(walk);
  };
  cols.forEach(walk);
  return out;
}

export const padTo = (n: number, align: number) => Math.ceil(n / align) * align;

export type Hop = { array: ArrowArr; buffer: 'offsets' | 'values' | 'data' | 'indices' | 'dictionary'; from: number; to: number; result: [number, number]; text: string };

/** Follow one record's value(s) of a leaf column down through the offsets: constant work per nesting level, whatever the row number. */
export function traceRow(cols: ArrowArr[], leaf: Leaf, row: number): { hops: Hop[]; values: (string | number | null)[] } {
  const hops: Hop[] = [];
  let arr = cols.find((c) => c.name === leaf.path[0].name)!;
  let lo = row;
  let hi = row + 1;
  const nullAt = (a: ArrowArr, i: number) => a.validity !== null && a.validity[i] === 0;
  for (let k = 0; ; ) {
    const node = leaf.path[k];
    if (arr.kind === 'list') {
      const a = arr;
      const nlo = a.offsets![lo];
      const nhi = a.offsets![hi];
      hops.push({ array: a, buffer: 'offsets', from: lo, to: hi, result: [nlo, nhi], text: `${a.path}: offsets[${lo}] = ${nlo}, offsets[${hi}] = ${nhi} → ${nhi > nlo ? `child slots ${nlo}..${nhi - 1}` : 'no child slots'}` });
      lo = nlo;
      hi = nhi;
      arr = a.children[0];
      if (node.list) {
        k++; // the LIST's middle "list" group has no array of its own
        k++;
        continue;
      }
      if (arr.kind === 'struct') {
        k++;
        if (k >= leaf.path.length) break;
        arr = arr.children.find((c) => c.name === leaf.path[k].name)!;
        continue;
      }
      break;
    }
    if (arr.kind === 'struct') {
      k++;
      arr = arr.children.find((c) => c.name === leaf.path[k].name)!;
      continue;
    }
    break;
  }
  const values: (string | number | null)[] = [];
  const dec = new TextDecoder();
  if (arr.kind === 'int64') {
    for (let i = lo; i < hi; i++) values.push(nullAt(arr, i) ? null : arr.values![i]);
    hops.push({ array: arr, buffer: 'values', from: lo, to: hi, result: [lo * 8, hi * 8], text: hi > lo ? `${arr.path}: slots ${lo}..${hi - 1} are bytes ${lo * 8}..${hi * 8 - 1} of the values buffer (fixed 8 bytes per slot, no offsets)` : `${arr.path}: no slots` });
  } else if (arr.kind === 'utf8') {
    for (let i = lo; i < hi; i++) values.push(nullAt(arr, i) ? null : dec.decode(new Uint8Array(arr.data!.slice(arr.offsets![i], arr.offsets![i + 1]))));
    hops.push({ array: arr, buffer: 'offsets', from: lo, to: hi, result: [arr.offsets![lo], arr.offsets![hi]], text: `${arr.path}: offsets[${lo}] = ${arr.offsets![lo]}, offsets[${hi}] = ${arr.offsets![hi]} → ${arr.offsets![hi] > arr.offsets![lo] ? `data bytes ${arr.offsets![lo]}..${arr.offsets![hi] - 1}` : 'no bytes'}` });
  } else if (arr.kind === 'dict') {
    const d = arr.dictionary!;
    for (let i = lo; i < hi; i++) values.push(nullAt(arr, i) ? null : dec.decode(new Uint8Array(d.data!.slice(d.offsets![arr.indices![i]], d.offsets![arr.indices![i] + 1]))));
    hops.push({ array: arr, buffer: 'indices', from: lo, to: hi, result: [lo, hi], text: hi > lo ? `${arr.path}: indices[${lo}..${hi - 1}] = [${arr.indices!.slice(lo, hi).join(', ')}] → dictionary lookups` : `${arr.path}: no slots` });
    // Each non-null index is one more offset read, in the dictionary's own offsets buffer.
    for (let i = lo; i < hi; i++) {
      if (nullAt(arr, i)) continue;
      const ix = arr.indices![i];
      const a = d.offsets![ix];
      const b = d.offsets![ix + 1];
      hops.push({ array: d, buffer: 'offsets', from: ix, to: ix + 1, result: [a, b], text: `${d.path}: offsets[${ix}] = ${a}, offsets[${ix + 1}] = ${b} → data bytes ${a}..${b - 1}` });
    }
  }
  return { hops, values };
}

/* ================================================================== UI */

const NOW = 'var(--viz-2)';
const mono = { fontFamily: 'var(--sl-font-mono, monospace)' } as const;

function Cells({ values, hot, nulls, label }: { values: (string | number)[]; hot: Set<number>; nulls?: Set<number>; label: string }) {
  return (
    <div style={{ overflowX: 'auto', whiteSpace: 'nowrap', paddingBottom: 2 }} aria-label={label}>
      {values.length === 0 ? (
        <span style={{ fontSize: '0.72rem', color: 'var(--viz-ink-muted)' }}>(empty)</span>
      ) : (
        values.map((v, i) => (
          <span
            key={i}
            title={`${label} [${i}]`}
            style={{
              ...mono,
              display: 'inline-block',
              minWidth: '1.7em',
              padding: '0 3px',
              marginRight: 2,
              textAlign: 'center',
              fontSize: '0.72rem',
              lineHeight: '1.5',
              borderRadius: 3,
              border: hot.has(i) ? `2px solid ${NOW}` : nulls?.has(i) ? '1px dashed var(--viz-stale)' : '1px solid var(--viz-border)',
              fontWeight: hot.has(i) ? 700 : 400,
              color: nulls?.has(i) ? 'var(--viz-ink-muted)' : 'var(--viz-ink)',
              textDecoration: nulls?.has(i) ? 'line-through' : undefined,
              background: 'var(--viz-plane)',
            }}
          >
            {v}
          </span>
        ))
      )}
    </div>
  );
}

const byteChar = (b: number) => (b >= 32 && b < 127 ? String.fromCharCode(b) : '·');

export default function ArrowNestedBuffersLab() {
  const [dataset, setDataset] = useState<DatasetKey>('paper');
  const schema = DATASETS[dataset].schema;
  const records = useMemo(() => {
    const p = parseRecords(schema, DATASETS[dataset].text);
    return p.ok ? p.records : [];
  }, [dataset, schema]);
  const leaves = useMemo(() => leavesOf(schema), [schema]);
  const [row, setRow] = useState(1);
  const [leafIdx, setLeafIdx] = useState(5);
  const [dict, setDict] = useState(false);
  const [align, setAlign] = useState<'8' | '64'>('8');

  const cols = useMemo(() => toArrow(schema, records, dict), [schema, records, dict]);
  const shredded = useMemo(() => shred(schema, records), [schema, records]);
  const leaf = leaves[Math.min(leafIdx, leaves.length - 1)];
  const r = Math.min(row, records.length - 1);
  const trace = traceRow(cols, leaf, r);
  const bufs = buffersOf(cols);
  const A = Number(align);
  const logical = bufs.reduce((s, b) => s + b.bytes, 0);
  const padded = bufs.reduce((s, b) => s + padTo(b.bytes, A), 0);
  const offsetReads = trace.hops.filter((h) => h.buffer === 'offsets').length;
  const levelsBefore = shredded[leaf.index].filter((e) => e.row < r).length;

  // Which cells the trace touches, per array path and buffer.
  const hot = new Map<string, Set<number>>();
  const mark = (path: string, buf: string, from: number, to: number) => {
    const k = `${path}|${buf}`;
    const s = hot.get(k) ?? new Set<number>();
    for (let i = from; i < to; i++) s.add(i);
    hot.set(k, s);
  };
  for (const h of trace.hops) {
    const a = h.array;
    if (a.validity) mark(a.path, 'validity', h.from, h.to);
    if (h.buffer === 'offsets') {
      mark(a.path, 'offsets', h.from, h.from + 1);
      mark(a.path, 'offsets', h.to, h.to + 1);
      if (a.kind === 'utf8') mark(a.path, 'data', h.result[0], h.result[1]);
    } else if (h.buffer === 'values') mark(a.path, 'values', h.from, h.to);
    else if (h.buffer === 'indices') mark(a.path, 'indices', h.from, h.to);
  }
  const get = (path: string, buf: string) => hot.get(`${path}|${buf}`) ?? new Set<number>();

  const rows: { a: ArrowArr; depth: number }[] = [];
  const walk = (a: ArrowArr, depth: number) => {
    rows.push({ a, depth });
    if (a.dictionary) walk(a.dictionary, depth + 1);
    a.children.forEach((c) => walk(c, depth + 1));
  };
  cols.forEach((c) => walk(c, 0));

  const valuesText = trace.values.length ? trace.values.map((v) => (v === null ? 'null' : typeof v === 'string' ? `'${v}'` : v)).join(', ') : 'no values';

  return (
    <VizPanel
      title="The same records in Arrow: validity bitmaps and offsets"
      subtitle="Arrow keeps nesting as offsets into child arrays instead of levels. Pick a record and a leaf column to follow the offsets down to its values."
      controls={
        <>
          <Choice
            label="Records"
            value={dataset}
            onChange={(k) => {
              setDataset(k);
              setLeafIdx(k === 'list' ? 1 : 5);
              setRow(1);
            }}
            options={(Object.keys(DATASETS) as DatasetKey[]).map((k) => ({ value: k, label: DATASETS[k].label }))}
          />
          <Slider label="Record" min={0} max={Math.max(0, records.length - 1)} value={r} onChange={setRow} />
          <Choice label="Leaf column" value={String(leaf.index)} onChange={(v) => setLeafIdx(Number(v))} options={leaves.map((l) => ({ value: String(l.index), label: l.id }))} />
          <Check label="Dictionary-encode strings" checked={dict} onChange={setDict} />
          <Segmented label="Pad each buffer to" value={align} onChange={setAlign} options={[{ value: '8', label: '8 bytes' }, { value: '64', label: '64 bytes' }]} />
        </>
      }
      legend={
        <Legend
          items={[
            { label: 'Read to fetch the selected record', color: NOW },
            { label: 'Null slot (its value bytes are ignored)', color: 'var(--viz-stale)' },
          ]}
        />
      }
      stats={
        <Stats
          items={[
            { label: 'Buffers', value: fmtNum(bufs.length), hint: 'A validity bitmap is only allocated when an array has nulls.' },
            {
              label: `Bytes, padded to ${A}`,
              value: `${fmtNum(logical)} → ${fmtNum(padded)}`,
              hint: dict ? 'Dictionaries travel in separate DictionaryBatch messages in Arrow IPC.' : A === 8 ? 'With 8-byte padding this equals the body of the IPC RecordBatch message pyarrow 25 writes for these records.' : '64-byte padding lets a 512-bit SIMD loop run past the last value without a bounds check.',
            },
            { label: 'Offset reads for this fetch', value: fmtNum(offsetReads), hint: dict ? 'One per list level, plus one dictionary offsets read per non-null string value: independent of the record number.' : 'One per list level plus one for the string offsets: independent of the record number.' },
            { label: 'Parquet levels decoded first', value: fmtNum(levelsBefore), hint: 'Within a data page there is no per-row index: to reach this record a reader decodes the repetition and definition levels of every earlier entry in the page. Assumes one page.' },
          ]}
        />
      }
      note={
        <Note>
          <strong>
            Record {r}, {leaf.id}: {valuesText}.
          </strong>{' '}
          Arrow resolved that with {offsetReads} offset read{offsetReads === 1 ? '' : 's'}, {dict ? 'one per list level plus one dictionary lookup per non-null string value' : 'one per list or string level on the path'}, and needs the same number for any record{dict ? ' holding as many values' : ''}. {levelsBefore === 0 ? (
            <>A Parquet reader at the start of the data page finds this record at once, but for any later record it must first decode the levels of every earlier entry in the page.</>
          ) : (
            <>
              A Parquet reader at the start of the data page must first decode the levels of {levelsBefore} earlier entr{levelsBefore === 1 ? 'y' : 'ies'} to find where record {r} begins, and skip their values.
            </>
          )}
        </Note>
      }
      table={
        <table className="viz-table">
          <thead>
            <tr>
              <th>Array</th>
              <th>Buffer</th>
              <th>Bytes</th>
              <th>Padded to {A}</th>
            </tr>
          </thead>
          <tbody>
            {bufs.map((b, i) => (
              <tr key={i}>
                <td>{b.array.path}</td>
                <td>{b.kind}</td>
                <td>{b.bytes}</td>
                <td>{padTo(b.bytes, A)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      }
    >
      <div style={{ display: 'grid', gap: 6 }}>
        {rows.map(({ a, depth }) => {
          const nulls = new Set<number>();
          a.validity?.forEach((v, i) => {
            if (!v) nulls.add(i);
          });
          const onPath = trace.hops.some((h) => h.array === a);
          return (
            <div key={a.path} style={{ marginLeft: `${depth * 0.9}rem`, borderLeft: `3px solid ${onPath ? NOW : 'var(--viz-border)'}`, paddingLeft: 8 }}>
              <div style={{ fontSize: '0.76rem', color: 'var(--viz-ink)' }}>
                <strong style={mono}>{a.name}</strong> <span style={{ color: 'var(--viz-ink-2)' }}>{a.type}{a.nullable ? '' : ' not null'} · length {a.length} · null_count {a.nullCount}</span>
              </div>
              <div style={{ display: 'grid', gridTemplateColumns: 'minmax(4.5rem, max-content) minmax(0, 1fr)', columnGap: 8, rowGap: 2, alignItems: 'center' }}>
                <span style={{ fontSize: '0.7rem', color: 'var(--viz-ink-2)' }}>validity</span>
                {a.validity ? <Cells label={`${a.path} validity`} values={a.validity} hot={get(a.path, 'validity')} /> : <span style={{ fontSize: '0.7rem', color: 'var(--viz-ink-muted)' }}>not allocated (no nulls)</span>}
                {a.offsets ? (
                  <>
                    <span style={{ fontSize: '0.7rem', color: 'var(--viz-ink-2)' }}>offsets</span>
                    <Cells label={`${a.path} offsets`} values={a.offsets} hot={get(a.path, 'offsets')} />
                  </>
                ) : null}
                {a.values ? (
                  <>
                    <span style={{ fontSize: '0.7rem', color: 'var(--viz-ink-2)' }}>values</span>
                    <Cells label={`${a.path} values`} values={a.values} hot={get(a.path, 'values')} nulls={nulls} />
                  </>
                ) : null}
                {a.indices ? (
                  <>
                    <span style={{ fontSize: '0.7rem', color: 'var(--viz-ink-2)' }}>indices</span>
                    <Cells label={`${a.path} indices`} values={a.indices} hot={get(a.path, 'indices')} nulls={nulls} />
                  </>
                ) : null}
                {a.data ? (
                  <>
                    <span style={{ fontSize: '0.7rem', color: 'var(--viz-ink-2)' }}>data</span>
                    <Cells label={`${a.path} data`} values={a.data.map(byteChar)} hot={get(a.path, 'data')} />
                  </>
                ) : null}
              </div>
            </div>
          );
        })}
      </div>
      <ol style={{ margin: '0.7rem 0 0', paddingLeft: '1.3rem', fontSize: '0.78rem', color: 'var(--viz-ink)' }} aria-label="Offset trace">
        {trace.hops.map((h, i) => (
          <li key={i} style={mono}>
            {h.text}
          </li>
        ))}
      </ol>
    </VizPanel>
  );
}
