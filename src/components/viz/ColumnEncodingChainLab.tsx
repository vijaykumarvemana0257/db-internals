import { useDeferredValue, useMemo, useState } from 'react';
import { VizPanel, Segmented, Slider, Check, Choice, Button, Legend, Stats, Note, TooltipHost, useTip, fmtBytes, fmtNum } from './Viz';
import {
  MAX_ROWS,
  PRESETS,
  ROLES,
  STEP_INFO,
  cmpScalar,
  encodeChain,
  evaluate,
  fieldBit,
  generateColumn,
  parsePasted,
  pickSmallest,
  presetFor,
  type Column,
  type GenId,
  type Op,
  type Output,
  type Role,
  type Scalar,
  type StepId,
} from './columnEncodingChainModel';

/**
 * Encoding lab: generate or paste a column, stack lightweight encodings, read the exact bits each step writes, then
 * run a predicate and see whether it runs on codes, runs, packed offsets or compressed bytes — or has to decode.
 * The encoders and their sources are documented in columnEncodingChainModel.ts.
 */

const ROLE_COLOR: Record<Role, string> = {
  payload: 'var(--viz-1)',
  header: 'var(--viz-2)',
  dict: 'var(--viz-3)',
  except: 'var(--viz-4)',
  control: 'var(--viz-5)',
};
const ROLE_LABEL: Record<Role, string> = {
  payload: 'Values / codes',
  header: 'Headers & block metadata',
  dict: 'Dictionary / symbol table',
  except: 'Exceptions & escapes',
  control: 'Lengths, run headers, control bits',
};
/** More work, darker violet: a light tint of one hue for 'tested in stored form', full strength for 'decoded first'. */
const WORK_COLOR = ['var(--viz-neutral)', 'color-mix(in srgb, var(--viz-7) 45%, var(--viz-surface))', 'var(--viz-7)'];
const MATCH_COLOR = 'var(--viz-6)';

const GENS: { value: GenId; label: string }[] = [
  { value: 'status', label: 'Low-cardinality strings (order status)' },
  { value: 'ids', label: 'Sorted ids' },
  { value: 'ints', label: 'Random ints with outliers' },
  { value: 'ts', label: 'Monotonic timestamps (ms)' },
  { value: 'sensor', label: 'Sensor floats' },
  { value: 'urls', label: 'URLs (high-cardinality strings)' },
  { value: 'pasted', label: 'Paste your own values' },
];

const DEFAULT_CHAIN: Record<GenId, StepId[]> = {
  status: ['dict', 'hybrid'],
  ids: ['delta', 'for', 'bitpack'],
  ints: ['for', 'bitpack'],
  ts: ['promts'],
  sensor: ['gorilla'],
  urls: ['fsst'],
  pasted: [],
};

const BITS_PER_ROW = 48;
const WINDOW_ROWS = 8;
const CELL = 12;
const MAX_STEPS = 4;

const short = (v: Scalar, n = 22) => {
  const s = typeof v === 'string' ? `'${v}'` : String(v);
  return s.length > n ? s.slice(0, n - 1) + '…' : s;
};

function sortedDistinct(col: Column): Scalar[] {
  return [...new Set(col.values)].sort(cmpScalar);
}

function defaultPredicate(col: Column, distinct: Scalar[]): { op: Op; idx: number } {
  if (col.kind !== 'string') return { op: 'lt', idx: Math.floor(distinct.length * 0.3) };
  const counts = new Map<Scalar, number>();
  for (const v of col.values) counts.set(v, (counts.get(v) ?? 0) + 1);
  let best = col.values[0];
  for (const [v, c] of counts) if (c > (counts.get(best) ?? 0)) best = v;
  // the most frequent value is usually the one people filter on; for all-distinct columns take row 0
  const pick = counts.get(best) === 1 ? col.values[0] : best;
  return { op: 'eq', idx: Math.max(0, distinct.indexOf(pick)) };
}

export default function ColumnEncodingChainLab() {
  const [gen, setGen] = useState<GenId>('status');
  const [n, setN] = useState(MAX_ROWS);
  const [outlierPct, setOutlierPct] = useState(2);
  const [sorted, setSorted] = useState(false);
  const [jitterMs, setJitterMs] = useState(0);
  const [decimals, setDecimals] = useState(2);
  const [pasteText, setPasteText] = useState('3, 3, 3, 3, 3, 3, 3, 3, 3, 7, 7, 8, 9, 1000000, 9, 9, 10, 11');
  const [pasted, setPasted] = useState<Column | null>(null);
  const [pasteError, setPasteError] = useState('');
  const [blockSize, setBlockSize] = useState(256);
  const [chain, setChain] = useState<StepId[]>(DEFAULT_CHAIN.status);
  const [op, setOp] = useState<Op>('eq');
  const [constIdx, setConstIdx] = useState(-1);
  const [output, setOutput] = useState<Output>('values');
  const [view, setView] = useState<'start' | Role>('payload');
  const [autoNote, setAutoNote] = useState('');

  const dn = useDeferredValue(n);
  const params = { n: dn, outlierPct, sorted, jitterMs, decimals };
  const col = useMemo<Column>(
    () => (gen === 'pasted' ? (pasted ?? parsePastedOr(pasteText)) : generateColumn(gen, params)),
    [gen, dn, outlierPct, sorted, jitterMs, decimals, pasted],
  );
  const distinct = useMemo(() => sortedDistinct(col), [col]);
  const pdef = useMemo(() => defaultPredicate(col, distinct), [col, distinct]);
  const idx = constIdx < 0 ? pdef.idx : Math.min(constIdx, distinct.length - 1);
  const pred = { op, v: distinct[idx] };

  const enc = useMemo(() => encodeChain(col, chain, blockSize), [col, chain, blockSize]);
  const ev = useMemo(() => evaluate(col, enc, pred, output), [col, enc, op, idx, output]);
  const flags = useMemo(() => {
    if (col.kind !== 'int') return { nonneg: false, monotonic: false };
    const a = col.values as number[];
    return { nonneg: a.every((v) => v >= 0), monotonic: a.every((v, i) => i === 0 || v >= a[i - 1]) };
  }, [col]);

  const chooseGen = (g: GenId) => {
    setGen(g);
    setAutoNote('');
    setConstIdx(-1);
    const c = g === 'pasted' ? (pasted ?? parsePastedOr(pasteText)) : generateColumn(g, params);
    setOp(c.kind === 'string' ? 'eq' : 'lt');
    setChain(g === 'pasted' ? (c.kind === 'string' ? ['dict', 'hybrid'] : c.kind === 'float' ? ['alp', 'for', 'bitpack'] : []) : DEFAULT_CHAIN[g]);
    setView('payload');
  };
  const usePaste = () => {
    const r = parsePasted(pasteText);
    if ('error' in r) {
      setPasteError(r.error);
      return;
    }
    setPasteError('');
    setPasted(r);
    setConstIdx(-1);
    setOp(r.kind === 'string' ? 'eq' : 'lt');
    setChain(r.kind === 'string' ? ['dict', 'hybrid'] : r.kind === 'float' ? ['alp', 'for', 'bitpack'] : []);
  };

  const presetOptions = PRESETS.filter((p) => p.kinds.includes(col.kind))
    .filter((p) => !(p.value === 'prom' && !flags.monotonic) && !(p.value === 'bitpack' && !flags.nonneg))
    .map((p) => ({ value: p.value, label: p.label }));
  const presetValue = presetFor(chain);
  if (presetValue === 'custom') presetOptions.push({ value: 'custom', label: 'Custom chain' });

  // ---- bit window
  const present = ROLES.filter((r) => enc.bits[r] > 0);
  const windowStart = useMemo(() => {
    if (!enc.ok || view === 'start') return 0;
    let k = enc.fields.findIndex((f) => f.role === view);
    if (k < 0) return 0;
    // for data, back up over the run header or width byte that introduces it
    if (view === 'payload') while (k > 0 && enc.fields[k - 1].si === enc.fields[k].si && enc.fields[k - 1].role !== 'payload' && enc.fields[k - 1].w <= 32 && enc.offsets[k] - enc.offsets[k - 1] <= 32) k--;
    return enc.offsets[k];
  }, [enc, view]);
  const windowBits = BITS_PER_ROW * WINDOW_ROWS;
  const visible = useMemo(() => {
    if (!enc.ok) return [];
    const out: { fi: number; start: number; end: number }[] = [];
    let lo = 0;
    let hi = enc.fields.length;
    while (lo < hi) {
      const mid = (lo + hi) >> 1;
      if (enc.offsets[mid] + enc.fields[mid].w <= windowStart) lo = mid + 1;
      else hi = mid;
    }
    for (let i = lo; i < enc.fields.length && enc.offsets[i] < windowStart + windowBits; i++) {
      if (enc.fields[i].w === 0) continue;
      out.push({ fi: i, start: enc.offsets[i], end: enc.offsets[i] + enc.fields[i].w });
    }
    return out;
  }, [enc, windowStart]);

  // ---- work strip
  const cells = Math.min(128, col.values.length);
  const strip = useMemo(() => {
    const per = col.values.length / cells;
    return Array.from({ length: cells }, (_, c) => {
      const a = Math.floor(c * per);
      const b = Math.max(a + 1, Math.floor((c + 1) * per));
      let w = 0;
      let m = 0;
      for (let r = a; r < b; r++) {
        if (ev.work[r] > w) w = ev.work[r];
        m += ev.match[r];
      }
      return { a, b, w, frac: m / (b - a) };
    });
  }, [ev, cells, col]);

  const tip = useTip();
  const nRows = col.values.length;
  const bpv = enc.ok ? enc.total / nRows : 0;
  const ratio = enc.ok && enc.total > 0 ? enc.plain / enc.total : 0;
  const blockwise = enc.ok && enc.blockCount > 0;
  const rleStarts = (enc.steps.find((s) => s.id === 'rle')?.data.runStarts as number[] | undefined) ?? null;
  const chainName = presetValue === 'custom' ? chain.map((s) => STEP_INFO[s].label).join(' → ') : (PRESETS.find((p) => p.value === presetValue)?.label ?? 'PLAIN');
  const W = 640;
  const barMax = Math.max(enc.plain, enc.total, 1);
  const barW = (bits: number) => (bits / barMax) * 430;

  return (
    <VizPanel
      title="Stack encodings on a column, then query it"
      subtitle="Pick a column, build an encoding chain step by step, and read the exact bits every step writes. Then run a predicate: does it run on codes, runs, packed offsets or compressed bytes, or does every value have to be decoded first?"
      controls={
        <>
          <Choice label="Column" value={gen} onChange={chooseGen} options={GENS} />
          {gen !== 'pasted' ? <Slider label="Rows" min={64} max={MAX_ROWS} step={64} value={n} onChange={setN} format={fmtNum} /> : null}
          {gen === 'ints' ? <Slider label="Outliers" min={0} max={10} step={0.5} value={outlierPct} onChange={setOutlierPct} format={(v) => `${v}%`} /> : null}
          {gen === 'status' ? <Check label="Sorted (clustered)" checked={sorted} onChange={setSorted} /> : null}
          {gen === 'ts' ? <Slider label="Scrape jitter" min={0} max={50} value={jitterMs} onChange={setJitterMs} format={(v) => `±${v} ms`} /> : null}
          {gen === 'sensor' ? <Slider label="Decimal places" min={1} max={4} value={decimals} onChange={setDecimals} /> : null}
          <Segmented
            label="Block size (FOR, bit-pack, PFOR, ALP)"
            value={String(blockSize)}
            onChange={(v) => setBlockSize(Number(v))}
            options={['128', '256', '1024', '2048'].map((v) => ({ value: v, label: v }))}
          />
        </>
      }
      legend={
        <Legend
          items={[
            { label: 'Settled from block headers alone', color: WORK_COLOR[0] },
            { label: 'Tested in stored form', color: WORK_COLOR[1] },
            { label: 'Decoded before testing', color: WORK_COLOR[2] },
            { label: 'Share of rows matching', color: MATCH_COLOR },
          ]}
        />
      }
      stats={
        <Stats
          items={[
            { label: 'Encoded size', value: enc.ok ? fmtBytes(enc.total / 8) : '—', hint: `PLAIN would be ${fmtBytes(enc.plain / 8)}` },
            { label: 'Bits per value', value: enc.ok ? bpv.toFixed(2) : '—', hint: 'Every bit the chain writes, including dictionaries and headers, divided by the row count.' },
            { label: 'vs PLAIN', value: enc.ok ? (ratio >= 1 ? `${fmtNum(ratio, 1)}× smaller` : `${fmtNum(1 / ratio, 2)}× larger`) : '—' },
            { label: 'Predicate runs on', value: enc.ok ? ev.testedOn : '—' },
            { label: 'Stored items tested', value: enc.ok ? fmtNum(ev.encodedTested) : '—', hint: 'Codes, packed offsets or FSST strings compared without decoding; an RLE run counts once.' },
            { label: 'Values decoded first', value: enc.ok ? fmtNum(ev.decoded) : '—', hint: 'Values that had to be rebuilt (running sums, XOR chains, ALP multiplication, FSST decompression) before the test.' },
            { label: 'Blocks skipped / accepted whole', value: blockwise ? `${ev.blocksSkipped} / ${ev.blocksAll} of ${enc.blockCount}` : '—', hint: 'Decided from each block’s reference and bit width alone.' },
            { label: output === 'values' ? 'Survivors decoded for output' : 'Matching rows', value: enc.ok ? fmtNum(output === 'values' ? ev.outputDecoded : ev.matches) : '—', hint: output === 'values' ? `${fmtNum(ev.matches)} rows match. Only those are unpacked or looked up to be returned.` : 'COUNT(*) never materialises a value.' },
          ]}
        />
      }
      note={
        <Note>
          {!enc.ok ? (
            <>
              <strong>This chain no longer fits the data.</strong> {enc.error} Remove the last step or pick a preset.
            </>
          ) : (
            <>
              <strong>
                {chainName} on {fmtNum(nRows)} rows of {col.label}: {bpv.toFixed(2)} bits per value, {ratio >= 1 ? `${fmtNum(ratio, 1)}× smaller` : `${fmtNum(1 / ratio, 2)}× larger`} than PLAIN.
              </strong>{' '}
              {pred.op === 'eq' ? 'x = ' : 'x < '}
              {short(pred.v, 40)} matches {fmtNum(ev.matches)} rows. {ev.story.join(' ')}{' '}
              {output === 'values' && enc.steps.length > 0
                ? ev.decoded > 0
                  ? 'The survivors were already decoded, at the price of decoding everything.'
                  : `Only the ${fmtNum(ev.outputDecoded)} survivors are unpacked or looked up to return them.`
                : ''}
            </>
          )}
        </Note>
      }
      table={
        enc.ok ? (
          <>
            <table className="viz-table">
              <thead>
                <tr>
                  <th>Step</th>
                  <th>Result</th>
                  {ROLES.map((r) => (
                    <th key={r}>{ROLE_LABEL[r]} (bits)</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {[...enc.steps.map((s, i) => ({ name: STEP_INFO[s.id].label, desc: s.desc, si: i })), ...(enc.plainTail ? [{ name: 'PLAIN tail', desc: enc.tailDesc, si: -1 }] : [])].map((row) => (
                  <tr key={row.si + row.name}>
                    <td>{row.name}</td>
                    <td>{row.desc}</td>
                    {ROLES.map((r) => (
                      <td key={r}>{fmtNum(enc.fields.reduce((t, f) => t + (f.si === row.si && f.role === r ? f.w : 0), 0))}</td>
                    ))}
                  </tr>
                ))}
                <tr>
                  <td>Total</td>
                  <td>
                    {fmtNum(enc.total)} bits vs PLAIN {fmtNum(enc.plain)}
                  </td>
                  {ROLES.map((r) => (
                    <td key={r}>{fmtNum(enc.bits[r])}</td>
                  ))}
                </tr>
              </tbody>
            </table>
            <table className="viz-table">
              <tbody>
                <tr><td>Predicate</td><td>{pred.op === 'eq' ? 'x = ' : 'x < '}{short(pred.v, 60)}</td></tr>
                <tr><td>Matching rows</td><td>{fmtNum(ev.matches)}</td></tr>
                <tr><td>Dictionary entries tested</td><td>{fmtNum(ev.dictTested)}</td></tr>
                <tr><td>Stored items tested (codes, runs, offsets, compressed strings)</td><td>{fmtNum(ev.encodedTested)}</td></tr>
                <tr><td>Values decoded before testing</td><td>{fmtNum(ev.decoded)}</td></tr>
                <tr><td>Exceptions patched</td><td>{fmtNum(ev.exceptionsPatched)}</td></tr>
                <tr><td>Blocks skipped / accepted whole / unpacked</td><td>{ev.blocksSkipped} / {ev.blocksAll} / {ev.blocksScanned}</td></tr>
                <tr><td>Survivors decoded for output</td><td>{fmtNum(ev.outputDecoded)}</td></tr>
              </tbody>
            </table>
          </>
        ) : null
      }
    >
      {gen === 'pasted' ? (
        <div style={{ marginBottom: '0.6rem' }}>
          <label style={{ display: 'block', fontSize: '0.8rem', color: 'var(--viz-ink-2)' }}>
            Values, separated by commas or new lines (up to {fmtNum(MAX_ROWS)}). All integers → integer column; all numbers → doubles; anything else → strings.
            <textarea value={pasteText} onChange={(e) => setPasteText(e.currentTarget.value)} rows={3} style={{ display: 'block', width: '100%', boxSizing: 'border-box', marginTop: 4, fontSize: '0.8rem', color: 'var(--viz-ink)', background: 'var(--viz-surface)', border: '1px solid var(--viz-border)' }} />
          </label>
          <div className="viz-controls">
            <Button primary onClick={usePaste}>
              Use these values
            </Button>
            {pasteError ? <span style={{ color: 'var(--viz-ink)', fontSize: '0.8rem' }}>{pasteError}</span> : <span style={{ color: 'var(--viz-ink-2)', fontSize: '0.8rem' }}>Using {fmtNum(col.values.length)} {col.kind === 'string' ? 'strings' : col.kind === 'float' ? 'doubles' : 'integers'}.</span>}
          </div>
        </div>
      ) : null}

      <div className="viz-controls">
        <Choice
          label="Preset chain"
          value={presetValue}
          onChange={(v) => {
            const p = PRESETS.find((x) => x.value === v);
            if (p) {
              setChain(p.chain);
              setAutoNote('');
              setView('payload');
            }
          }}
          options={presetOptions}
        />
        {enc.ok && enc.next.length > 0 && chain.length < MAX_STEPS ? (
          <Choice
            label="Add a step"
            value={'' as StepId | ''}
            onChange={(v) => {
              if (v) {
                setChain([...chain, v as StepId]);
                setAutoNote('');
              }
            }}
            options={[{ value: '' as StepId | '', label: 'choose…' }, ...enc.next.map((s) => ({ value: s as StepId | '', label: STEP_INFO[s].label }))]}
          />
        ) : null}
        <Button onClick={() => setChain(chain.slice(0, -1))} disabled={chain.length === 0}>
          Remove last step
        </Button>
        <Button
          onClick={() => {
            const r = pickSmallest(col, blockSize);
            setChain(r.chain);
            setView('payload');
            setAutoNote(`Picked by sampling ${fmtNum(r.sampled)} values (10 runs of 64 when the column is large enough, as BtrBlocks does) and trying ${r.tried} chains of up to 3 steps; the winner is then encoded on the whole column.`);
          }}
        >
          Auto-pick from a sample
        </Button>
      </div>

      <div style={{ display: 'flex', flexWrap: 'wrap', alignItems: 'stretch', gap: 6, margin: '0.2rem 0 0.6rem' }} aria-label="Encoding chain">
        <ChainBox title={`${col.label}`} body={`${fmtNum(nRows)} ${col.kind === 'string' ? 'strings' : col.kind === 'float' ? 'doubles' : 'int64'} · PLAIN ${fmtBytes(enc.plain / 8)}`} />
        {enc.steps.map((s, i) => (
          <ChainBox key={i} arrow title={STEP_INFO[s.id].label} body={s.desc} hint={STEP_INFO[s.id].title} />
        ))}
        {enc.ok && enc.plainTail ? <ChainBox arrow title="PLAIN" body={enc.tailDesc} dashed /> : null}
      </div>
      {autoNote ? <p style={{ fontSize: '0.78rem', color: 'var(--viz-ink-2)', margin: '0 0 0.5rem' }}>{autoNote}</p> : null}

      <svg viewBox={`0 0 ${W} 64`} width={W} height={64} role="img" aria-label={`Encoded ${fmtNum(enc.total)} bits against PLAIN ${fmtNum(enc.plain)} bits`}>
        <text x={0} y={17} fontSize={12} fill="var(--viz-ink)">
          PLAIN
        </text>
        <rect x={80} y={4} width={Math.max(2, barW(enc.plain))} height={18} rx={3} fill="var(--viz-plane)" stroke="var(--viz-ink-muted)" />
        <text x={86 + barW(enc.plain)} y={17} fontSize={11} fill="var(--viz-ink-2)">
          {fmtNum(enc.plain / nRows, 1)} bits/value
        </text>
        <text x={0} y={49} fontSize={12} fill="var(--viz-ink)">
          Encoded
        </text>
        {(() => {
          let x = 80;
          return ROLES.map((r) => {
            const w = barW(enc.bits[r]);
            if (enc.bits[r] <= 0) return null;
            const el = <rect key={r} x={x} y={36} width={Math.max(1, w)} height={18} fill={ROLE_COLOR[r]} {...tip(<>{ROLE_LABEL[r]}: {fmtNum(enc.bits[r])} bits ({fmtNum((enc.bits[r] / Math.max(1, enc.total)) * 100, 1)}%)</>)} />;
            x += Math.max(1, w);
            return el;
          });
        })()}
        <text x={86 + Math.max(barW(enc.total), 2)} y={49} fontSize={11} fill="var(--viz-ink-2)">
          {enc.ok ? `${bpv.toFixed(2)} bits/value` : ''}
        </text>
      </svg>

      <Legend items={ROLES.map((r) => ({ label: ROLE_LABEL[r], color: ROLE_COLOR[r] }))} />
      <div className="viz-controls" style={{ marginTop: 6 }}>
        <Choice
          label="Show bits from"
          value={present.includes(view as Role) || view === 'start' ? view : 'start'}
          onChange={setView}
          options={[{ value: 'start' as 'start' | Role, label: 'Start' }, ...present.map((r) => ({ value: r as 'start' | Role, label: r === 'payload' ? 'First data' : r === 'dict' ? 'Dictionary' : r === 'header' ? 'Headers' : r === 'except' ? 'First exception' : 'Control bits' }))]}
        />
      </div>
      <TooltipHost>
        <svg viewBox={`0 0 ${W} ${WINDOW_ROWS * (CELL + 4) + 6}`} width={W} height={WINDOW_ROWS * (CELL + 4) + 6} role="img" aria-label={`Bits ${windowStart} to ${windowStart + windowBits - 1} of the encoded column`}>
          {Array.from({ length: WINDOW_ROWS }, (_, r) => (
            <text key={r} x={0} y={r * (CELL + 4) + CELL} fontSize={10} fill="var(--viz-ink-2)">
              {fmtNum(windowStart + r * BITS_PER_ROW)}
            </text>
          ))}
          {visible.map(({ fi, start, end }) => {
            const f = enc.fields[fi];
            const color = ROLE_COLOR[f.role];
            const from = Math.max(start, windowStart);
            const to = Math.min(end, windowStart + windowBits);
            const cellsOut = [];
            for (let b = from; b < to; b++) {
              const rel = b - windowStart;
              const x = 52 + (rel % BITS_PER_ROW) * CELL;
              const y = Math.floor(rel / BITS_PER_ROW) * (CELL + 4);
              const bit = fieldBit(f, b - start);
              cellsOut.push(<rect key={b} x={x + 0.5} y={y + 0.5} width={CELL - 2} height={CELL - 2} rx={1.5} fill={bit ? color : 'var(--viz-surface)'} stroke={color} strokeWidth={1} />);
            }
            const relStart = start - windowStart;
            return (
              <g key={fi} {...tip(<><strong>{ROLE_LABEL[f.role]}</strong><br />{f.what}<br />bits {fmtNum(start)}–{fmtNum(end - 1)} ({f.w} bit{f.w === 1 ? '' : 's'})</>)}>
                {cellsOut}
                {relStart >= 0 ? <line x1={52 + (relStart % BITS_PER_ROW) * CELL - 0.5} x2={52 + (relStart % BITS_PER_ROW) * CELL - 0.5} y1={Math.floor(relStart / BITS_PER_ROW) * (CELL + 4) - 1} y2={Math.floor(relStart / BITS_PER_ROW) * (CELL + 4) + CELL + 1} stroke="var(--viz-ink)" strokeWidth={1.5} /> : null}
              </g>
            );
          })}
        </svg>
      </TooltipHost>
      <ol style={{ margin: '0.3rem 0 0.6rem', paddingLeft: '1.2rem', fontSize: '0.76rem', display: 'grid', gap: 2 }} aria-label="Fields in the bit window">
        {visible.slice(0, 12).map(({ fi, start, end }) => {
          const f = enc.fields[fi];
          return (
            <li key={fi} style={{ color: 'var(--viz-ink)', borderLeft: `4px solid ${ROLE_COLOR[f.role]}`, paddingLeft: 6 }}>
              <span style={{ color: 'var(--viz-ink-2)' }}>
                bits {fmtNum(start)}–{fmtNum(end - 1)} ·
              </span>{' '}
              {f.what}
            </li>
          );
        })}
        {visible.length > 12 ? <li style={{ color: 'var(--viz-ink-2)', listStyle: 'none' }}>… {visible.length - 12} more fields in this window (hover the bits)</li> : null}
      </ol>

      <div className="viz-controls">
        <Segmented
          label="Predicate"
          value={op}
          onChange={setOp}
          options={[
            { value: 'eq', label: 'x = c' },
            { value: 'lt', label: 'x < c' },
          ]}
        />
        <Slider label="Constant c" min={0} max={Math.max(0, distinct.length - 1)} value={idx} onChange={setConstIdx} format={(i) => short(distinct[i] ?? '', 26)} />
        <Segmented
          label="Query"
          value={output}
          onChange={setOutput}
          options={[
            { value: 'count', label: 'COUNT(*)' },
            { value: 'values', label: 'Return matching values' },
          ]}
        />
      </div>
      <TooltipHost>
        <svg viewBox={`0 0 ${W} 74`} width={W} height={74} role="img" aria-label={`Predicate work across ${nRows} rows: ${ev.decoded} decoded, ${ev.encodedTested} stored items tested, ${ev.blocksSkipped} blocks skipped`}>
          <text x={0} y={16} fontSize={11} fill="var(--viz-ink)">
            Work
          </text>
          <text x={0} y={40} fontSize={11} fill="var(--viz-ink)">
            Matches
          </text>
          {strip.map((c, i) => {
            const cw = (W - 64) / cells;
            const x = 60 + i * cw;
            return (
              <g key={i} {...tip(<>rows {fmtNum(c.a)}–{fmtNum(c.b - 1)}<br />{['settled from block headers alone', 'tested in stored form', 'decoded before testing'][c.w]}<br />{fmtNum(c.frac * 100, 0)}% match</>)}>
                <rect x={x} y={4} width={Math.max(1, cw - 0.6)} height={16} fill={WORK_COLOR[c.w]} stroke="var(--viz-border)" strokeWidth={0.5} />
                <rect x={x} y={28} width={Math.max(1, cw - 0.6)} height={16} fill="var(--viz-surface)" stroke="var(--viz-border)" strokeWidth={0.5} />
                {c.frac > 0 ? <rect x={x} y={28 + 16 * (1 - c.frac)} width={Math.max(1, cw - 0.6)} height={16 * c.frac} fill={MATCH_COLOR} /> : null}
              </g>
            );
          })}
          {blockwise
            ? Array.from({ length: enc.blockCount + 1 }, (_, b) => {
                // blocks count stream items; after RLE an item is a run, so place the line at the run's first row
                const item = b * blockSize;
                const row = rleStarts ? (item < rleStarts.length ? rleStarts[item] : nRows) : Math.min(item, nRows);
                const x = 60 + (row / nRows) * (W - 64);
                return <line key={b} x1={x} x2={x} y1={1} y2={47} stroke="var(--viz-ink)" strokeWidth={0.8} strokeOpacity={0.55} />;
              })
            : null}
          <text x={60} y={62} fontSize={10} fill="var(--viz-ink-2)">
            row 0
          </text>
          <text x={W - 4} y={62} fontSize={10} fill="var(--viz-ink-2)" textAnchor="end">
            row {fmtNum(nRows - 1)}
          </text>
          {blockwise ? (
            <text x={W / 2} y={62} fontSize={10} fill="var(--viz-ink-2)" textAnchor="middle">
              lines = block boundaries ({blockSize} {enc.steps.some((s) => s.id === 'rle') ? 'runs' : 'rows'} each)
            </text>
          ) : null}
        </svg>
      </TooltipHost>
    </VizPanel>
  );
}

function parsePastedOr(text: string): Column {
  const r = parsePasted(text);
  return 'error' in r ? { kind: 'int', values: [1, 2, 3], label: 'Pasted ints' } : r;
}

function ChainBox({ title, body, arrow, dashed, hint }: { title: string; body: string; arrow?: boolean; dashed?: boolean; hint?: string }) {
  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: 6 }} title={hint}>
      {arrow ? (
        <span aria-hidden="true" style={{ color: 'var(--viz-ink-muted)' }}>
          →
        </span>
      ) : null}
      <div style={{ border: `1px ${dashed ? 'dashed' : 'solid'} var(--viz-border)`, borderRadius: 6, padding: '4px 8px', background: 'var(--viz-surface)', maxWidth: 230 }}>
        <div style={{ fontSize: '0.8rem', color: 'var(--viz-ink)', fontWeight: 600 }}>{title}</div>
        <div style={{ fontSize: '0.74rem', color: 'var(--viz-ink-2)' }}>{body}</div>
      </div>
    </div>
  );
}
