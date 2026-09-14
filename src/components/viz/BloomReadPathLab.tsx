import { useMemo, useState } from 'react';
import { VizPanel, Segmented, Slider, Check, Choice, Legend, Stats, Note, fmtNum } from './Viz';

/* ------------------------------------------------------------------ Bloom */

/** 32-bit FNV-1a: stands in for LevelDB's BloomHash. */
function hash32(s: string) {
  let h = 0x811c9dc5;
  for (let i = 0; i < s.length; i++) {
    h ^= s.charCodeAt(i);
    h = Math.imul(h, 0x01000193) >>> 0;
  }
  return h >>> 0;
}

/** LevelDB's probing: one hash, then add delta = rotate-right-17 of it for each further probe. */
function probes(key: string, k: number, m: number) {
  let h = hash32(key);
  const delta = ((h >>> 17) | (h << 15)) >>> 0;
  const out: number[] = [];
  for (let j = 0; j < k; j++) {
    out.push(h % m);
    h = (h + delta) >>> 0;
  }
  return out;
}

const formulaFp = (k: number, bitsPerKey: number) => Math.pow(1 - Math.exp(-k / bitsPerKey), k);

export function bloom(n: number, bitsPerKey: number, k: number) {
  const m = Math.max(64, n * bitsPerKey);
  const bits = new Uint8Array(m);
  for (let i = 0; i < n; i++) for (const b of probes(`key:${i}`, k, m)) bits[b] = 1;
  const trials = 20000;
  let fp = 0;
  for (let i = 0; i < trials; i++) {
    if (probes(`absent:${i}`, k, m).every((b) => bits[b])) fp++;
  }
  const set = bits.reduce((a, b) => a + b, 0);
  return { m, bits, fillRatio: set / m, empirical: fp / trials };
}

/* ------------------------------------------------------------- read path */

type Lookup = 'present-deep' | 'present-l0' | 'absent' | 'scan';

const FPR_BY_BITS = (bits: number) => formulaFp(Math.max(1, Math.round(bits * Math.LN2)), bits);

export function readPath(lookup: Lookup, l0: number, levels: number, bitsPerKey: number, filtersOn: boolean, cacheHit: number, prefixBloom: boolean) {
  const fpr = filtersOn ? FPR_BY_BITS(bitsPerKey) : 1;
  const memtables = 2; // active + one immutable
  const rows: { source: string; filter: string; blockReads: number; note: string }[] = [];
  rows.push({ source: 'active memtable', filter: '—', blockReads: 0, note: 'in-memory search' });
  rows.push({ source: 'immutable memtable', filter: '—', blockReads: 0, note: 'in-memory search' });

  if (lookup === 'scan') {
    const sources = memtables + l0 + levels;
    for (let i = 0; i < l0; i++) rows.push({ source: `L0 file ${i + 1}`, filter: prefixBloom ? 'prefix filter' : 'not usable', blockReads: prefixBloom ? fpr : 1, note: 'merged into the iterator' });
    for (let i = 1; i <= levels; i++) rows.push({ source: `L${i}`, filter: prefixBloom ? 'prefix filter' : 'not usable', blockReads: prefixBloom ? (i === levels ? 1 : fpr) : 1, note: 'one level iterator' });
    const blocks = rows.reduce((a, r) => a + r.blockReads, 0);
    return { rows, sources, filtersChecked: prefixBloom ? l0 + levels : 0, blocks, disk: blocks * (1 - cacheHit), fpr, memtables };
  }

  const deepest = levels;
  for (let i = 0; i < l0; i++) {
    // L0 is searched newest first, and file 1 is the newest: a key in the newest file is found immediately.
    const holds = lookup === 'present-l0' && i === 0;
    rows.push({
      source: `L0 file ${i + 1}`,
      filter: filtersOn ? (holds ? 'maybe (true)' : 'checked') : 'none',
      blockReads: holds ? 1 : filtersOn ? fpr : 1,
      note: holds ? 'key found — stop' : 'L0 files overlap: each one is checked',
    });
    if (holds) break;
  }
  if (lookup !== 'present-l0' || l0 === 0) {
    for (let i = 1; i <= levels; i++) {
      const holds = lookup === 'present-deep' && i === deepest;
      rows.push({
        source: `L${i}`,
        filter: filtersOn ? (holds ? 'maybe (true)' : 'checked') : 'none',
        blockReads: holds ? 1 : filtersOn ? fpr : 1,
        note: holds ? 'key found — stop' : 'binary search picks one file, then its filter',
      });
    }
  }
  const filtersChecked = filtersOn ? rows.filter((r) => r.filter !== '—' && r.filter !== 'none').length : 0;
  const blocks = rows.reduce((a, r) => a + r.blockReads, 0);
  return { rows, sources: rows.length, filtersChecked, blocks, disk: blocks * (1 - cacheHit), fpr, memtables };
}

export default function BloomReadPathLab() {
  const [panel, setPanel] = useState<'bloom' | 'path'>('bloom');

  // bloom
  const [bitsPerKey, setBitsPerKey] = useState(10);
  const [k, setK] = useState(7);
  const [n, setN] = useState(60);
  const b = useMemo(() => bloom(n, bitsPerKey, k), [n, bitsPerKey, k]);
  const optimalK = Math.max(1, Math.round(bitsPerKey * Math.LN2));

  // read path
  const [lookup, setLookup] = useState<Lookup>('absent');
  const [l0, setL0] = useState(4);
  const [levels, setLevels] = useState(5);
  const [filtersOn, setFiltersOn] = useState(true);
  const [cachePct, setCachePct] = useState(80);
  const [prefixBloom, setPrefixBloom] = useState(false);
  const rp = useMemo(() => readPath(lookup, l0, levels, bitsPerKey, filtersOn, cachePct / 100, prefixBloom), [lookup, l0, levels, bitsPerKey, filtersOn, cachePct, prefixBloom]);

  const cols = 40;
  const shown = Math.min(b.m, 800);
  const cell = 14;

  return (
    <VizPanel
      title={panel === 'bloom' ? 'A Bloom filter, bit by bit' : 'Every place a read has to look'}
      subtitle={
        panel === 'bloom'
          ? 'Insert n keys: each sets k bits chosen by hashing. A lookup for a key that was never inserted is a false positive only if all k of its bits happen to be set already.'
          : 'A point read checks the memtables, every L0 file, and one file per deeper level, asking each file’s filter before reading a block. A range scan has to merge every source and mostly cannot use filters.'
      }
      controls={
        <Segmented
          label="Panel"
          value={panel}
          onChange={setPanel}
          options={[
            { value: 'bloom', label: 'Bloom filter' },
            { value: 'path', label: 'Read path' },
          ]}
        />
      }
      legend={
        panel === 'bloom' ? (
          <Legend
            items={[
              { label: 'Bit set', color: 'var(--viz-1)' },
              { label: 'Bit clear', color: 'var(--viz-plane)' },
            ]}
          />
        ) : (
          <Legend
            items={[
              { label: 'In-memory work', color: 'var(--viz-3)' },
              { label: 'Expected data-block reads', color: 'var(--viz-2)' },
              { label: 'Key found', color: 'var(--viz-good)' },
            ]}
          />
        )
      }
      stats={
        panel === 'bloom' ? (
          <Stats
            items={[
              { label: 'Bits (m)', value: fmtNum(b.m) },
              { label: 'Bits set', value: `${fmtNum(b.fillRatio * 100, 1)}%`, hint: 'The optimum is about 50%' },
              { label: 'False positives, measured', value: `${fmtNum(b.empirical * 100, 2)}%`, hint: '20,000 lookups of keys never inserted' },
              { label: 'Formula (1−e^(−k/b))^k', value: `${fmtNum(formulaFp(k, bitsPerKey) * 100, 2)}%` },
              { label: 'Optimal k for these bits', value: `${optimalK}` },
            ]}
          />
        ) : (
          <Stats
            items={[
              { label: 'Sources consulted', value: fmtNum(rp.sources) },
              { label: 'Filters checked', value: fmtNum(rp.filtersChecked) },
              { label: 'Expected block reads', value: fmtNum(rp.blocks, 2) },
              { label: `Disk reads at ${cachePct}% cache hits`, value: fmtNum(rp.disk, 2) },
            ]}
          />
        )
      }
      note={
        <Note>
          {panel === 'bloom' ? (
            <>
              <strong>
                {bitsPerKey} bits per key, k = {k}: {fmtNum(b.empirical * 100, 2)}% false positives measured, {fmtNum(formulaFp(k, bitsPerKey) * 100, 2)}% predicted.
              </strong>{' '}
              {k < optimalK ? `Raise k toward ${optimalK}: too few hash functions leave matching bits easy to find.` : k > optimalK ? `Lower k toward ${optimalK}: too many hash functions fill the array, so every lookup finds its bits set.` : 'k is at its optimum for this many bits per key, where roughly half the bits end up set.'}
            </>
          ) : lookup === 'present-l0' && l0 === 0 ? (
            <>
              <strong>There are no L0 files, so no key can be in one.</strong> Raise <em>L0 files</em> above zero — the read then stops at the newest L0 file that holds the key.
            </>
          ) : lookup === 'scan' ? (
            <>
              <strong>A range scan merges {rp.sources} sources.</strong>{' '}
              {prefixBloom
                ? 'With a prefix extractor and a scan bounded to one prefix, each file’s prefix filter can rule it out — the one way filters help a scan.'
                : 'A filter answers “is this exact key present?”, which is useless for “give me every key between a and b”, so every L0 file and every level contributes an iterator and at least one block read.'}{' '}
              Raise the L0 file count and watch the sources climb one for one.
            </>
          ) : (
            <>
              <strong>
                {lookup === 'absent' ? 'A key that is not in the tree' : lookup === 'present-l0' ? 'A key in the newest L0 file' : `A key in the bottom level, L${levels}`}: {fmtNum(rp.blocks, 2)} expected block reads.
              </strong>{' '}
              {filtersOn
                ? `Each filter check is in memory; with ${bitsPerKey} bits per key a file that does not hold the key is read only ${fmtNum(rp.fpr * 100, 2)}% of the time.`
                : 'Filters are off: every file that could hold the key by range must have a block read.'}{' '}
              {lookup === 'present-l0' ? 'The search stops at the first, newest match — deeper levels are never consulted.' : ''}
            </>
          )}
        </Note>
      }
      table={
        panel === 'path' ? (
          <table className="viz-table">
            <thead>
              <tr>
                <th>Source</th>
                <th>Filter</th>
                <th>Expected block reads</th>
                <th>Note</th>
              </tr>
            </thead>
            <tbody>
              {rp.rows.map((r) => (
                <tr key={r.source}>
                  <td>{r.source}</td>
                  <td>{r.filter}</td>
                  <td>{fmtNum(r.blockReads, 3)}</td>
                  <td>{r.note}</td>
                </tr>
              ))}
            </tbody>
          </table>
        ) : (
          <table className="viz-table">
            <thead>
              <tr>
                <th>Bits per key</th>
                <th>Optimal k</th>
                <th>Predicted false-positive rate</th>
              </tr>
            </thead>
            <tbody>
              {[4, 6, 8, 10, 12, 16, 20].map((bpk) => {
                const ok = Math.max(1, Math.round(bpk * Math.LN2));
                return (
                  <tr key={bpk}>
                    <td>{bpk}</td>
                    <td>{ok}</td>
                    <td>{fmtNum(formulaFp(ok, bpk) * 100, 3)}%</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        )
      }
    >
      {panel === 'bloom' ? (
        <>
          <div className="viz-controls">
            <Slider label="Bits per key" min={2} max={20} value={bitsPerKey} onChange={setBitsPerKey} />
            <Slider label="Hash functions (k)" min={1} max={14} value={k} onChange={setK} />
            <Slider label="Keys inserted (n)" min={10} max={80} value={n} onChange={setN} />
          </div>
          <svg width={cols * cell + 2} height={Math.ceil(shown / cols) * cell + 2} role="img" aria-label={`Bloom filter bit array: ${fmtNum(b.fillRatio * 100, 1)}% set`}>
            {Array.from({ length: shown }, (_, i) => (
              <rect key={i} x={1 + (i % cols) * cell} y={1 + Math.floor(i / cols) * cell} width={cell - 2} height={cell - 2} rx={2} fill={b.bits[i] ? 'var(--viz-1)' : 'var(--viz-plane)'} stroke="var(--viz-border)" strokeWidth={0.5} />
            ))}
          </svg>
          {b.m > shown ? <p style={{ fontSize: '0.75rem', color: 'var(--viz-ink-2)', margin: '0.3rem 0 0' }}>Showing the first {shown} of {fmtNum(b.m)} bits.</p> : null}
        </>
      ) : (
        <>
          <div className="viz-controls">
            <Choice
              label="Operation"
              value={lookup}
              onChange={setLookup}
              options={[
                { value: 'absent', label: 'Get a key that does not exist' },
                { value: 'present-deep', label: 'Get a key in the bottom level' },
                { value: 'present-l0', label: 'Get a key in the newest L0 file' },
                { value: 'scan', label: 'Range scan' },
              ]}
            />
            <Slider label="L0 files" min={0} max={20} value={l0} onChange={setL0} />
            <Slider label="Levels below L0" min={1} max={7} value={levels} onChange={setLevels} />
            <Check label={`Bloom filters (${bitsPerKey} bits/key)`} checked={filtersOn} onChange={setFiltersOn} />
            <Slider label="Block cache hit rate" min={0} max={99} value={cachePct} onChange={setCachePct} format={(v) => `${v}%`} />
            {lookup === 'scan' ? <Check label="Prefix Bloom (scan within one prefix)" checked={prefixBloom} onChange={setPrefixBloom} /> : null}
          </div>
          <svg width={660} height={24 + rp.rows.length * 20} role="img" aria-label="Sources a read consults">
            {rp.rows.map((r, i) => {
              const y = 6 + i * 20;
              const found = r.note.startsWith('key found');
              return (
                <g key={r.source}>
                  <text x={0} y={y + 12} fontSize={11} fill={found ? 'var(--viz-ink)' : undefined}>
                    {r.source}
                  </text>
                  <rect x={140} y={y + 2} width={r.filter === '—' ? 70 : 34} height={13} rx={3} fill="var(--viz-3)" />
                  <rect x={250} y={y + 2} width={Math.max(r.blockReads > 0 ? 2 : 0, 300 * r.blockReads)} height={13} rx={3} fill={found ? 'var(--viz-good)' : 'var(--viz-2)'} />
                  <text x={560} y={y + 12} fontSize={10}>
                    {fmtNum(r.blockReads, 2)} reads
                  </text>
                </g>
              );
            })}
          </svg>
        </>
      )}
    </VizPanel>
  );
}
