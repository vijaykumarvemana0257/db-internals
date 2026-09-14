import { useDeferredValue, useMemo, useState } from 'react';
import { VizPanel, Segmented, Slider, Check, Legend, Stats, Note, fmtNum, makeRng } from './Viz';

type Policy = 'leveled' | 'tiered';

/**
 * A size-level simulation of compaction. Batches of uniformly random keys are flushed as runs.
 * - Leveled: L0 runs merge into L1 at 4 files; any level over its target compacts a file-sized
 *   chunk into the next level, rewriting only the overlapping part of that level (as RocksDB does).
 * - Tiered: a level holding `ratio` runs merges them into one run in the next level.
 * Overlap between runs uses the expected size of a union of uniformly random key sets, so
 * duplicates (overwrites) are removed exactly as often as random keys collide.
 */
export function simulate(policy: Policy, ratio: number, keySpace: number, batches: number, batchSize: number, seed = 42) {
  const rng = makeRng(seed);
  const K = keySpace;
  const union = (a: number, b: number, space: number) => space * (1 - (1 - Math.min(a, space) / space) * (1 - Math.min(b, space) / space));
  const L0_TRIGGER = 4;
  const FILE = batchSize;
  const liveKeys = new Set<number>();
  let ingested = 0;
  let written = 0;
  let maxRuns = 0;
  let peakSpace = 0;

  // leveled state
  let l0: number[] = [];
  const lv: number[] = [0]; // lv[i] = entries in level i (i >= 1); lv[0] unused
  // tiered state
  const tiers: number[][] = [[]];

  for (let b = 0; b < batches; b++) {
    let fresh = 0;
    for (let j = 0; j < batchSize; j++) {
      // Two draws: makeRng alone has only 10^6 distinct outputs, which would fake collisions in a large key space.
      const k = (Math.floor(rng() * 1e6) * 1e6 + Math.floor(rng() * 1e6)) % K;
      if (!liveKeys.has(k)) { liveKeys.add(k); fresh++; }
    }
    const runSize = K * (1 - Math.pow(1 - 1 / K, batchSize)); // expected distinct keys in the batch
    ingested += batchSize;
    written += runSize;

    if (policy === 'leveled') {
      l0.push(runSize);
      if (l0.length >= L0_TRIGGER) {
        let merged = lv[1] ?? 0;
        for (const r of l0) merged = union(merged, r, K);
        written += merged;
        lv[1] = merged;
        l0 = [];
      }
      for (let i = 1; i < lv.length; i++) {
        const target = FILE * L0_TRIGGER * Math.pow(ratio, i - 1);
        let guard = 0;
        while ((lv[i] ?? 0) > target && guard++ < 1000) {
          const c = Math.min(lv[i], FILE);
          const phi = c / lv[i]; // share of this level's key range the chunk covers
          const next = lv[i + 1] ?? 0;
          const overlap = next * phi;
          const merged = union(c, overlap, K * phi);
          written += merged;
          lv[i] -= c;
          lv[i + 1] = next - overlap + merged;
        }
      }
      const runs = l0.length + lv.slice(1).filter((x) => x > 0.5).length;
      const space = (l0.reduce((a, x) => a + x, 0) + lv.slice(1).reduce((a, x) => a + x, 0)) / Math.max(1, liveKeys.size);
      maxRuns = Math.max(maxRuns, runs);
      peakSpace = Math.max(peakSpace, space);
    } else {
      tiers[0].push(runSize);
      for (let i = 0; i < tiers.length; i++) {
        if (tiers[i].length >= ratio) {
          let merged = 0;
          for (const r of tiers[i]) merged = union(merged, r, K);
          written += merged;
          tiers[i] = [];
          (tiers[i + 1] ??= []).push(merged);
        }
      }
      const runs = tiers.reduce((a, t) => a + t.length, 0);
      const space = tiers.reduce((a, t) => a + t.reduce((c, x) => c + x, 0), 0) / Math.max(1, liveKeys.size);
      maxRuns = Math.max(maxRuns, runs);
      peakSpace = Math.max(peakSpace, space);
    }
    void fresh;
  }
  const sizes = policy === 'leveled' ? [...l0, ...lv.slice(1)] : tiers.flat();
  const total = sizes.reduce((a, x) => a + x, 0);
  const runs = policy === 'leveled' ? l0.length + lv.slice(1).filter((x) => x > 0.5).length : tiers.reduce((a, t) => a + t.length, 0);
  return { writeAmp: written / Math.max(1, ingested), spaceAmp: total / Math.max(1, liveKeys.size), peakSpace, runs, maxRuns };
}

/** The key-space size at which `n` uniformly random keys contain the requested share of overwrites. */
export function keySpaceFor(overwritePct: number, n: number) {
  const target = 1 - overwritePct / 100; // distinct keys / keys written
  if (target >= 0.999) return 1e12;
  let lo = 1;
  let hi = 1e12;
  for (let i = 0; i < 100; i++) {
    const mid = Math.sqrt(lo * hi);
    const distinct = (mid / n) * (1 - Math.exp(-n / mid));
    if (distinct < target) lo = mid;
    else hi = mid;
  }
  return Math.round(Math.sqrt(lo * hi));
}

/** Expected data-block reads for a lookup of an absent key: one false-positive chance per run. */
const lookupCost = (runs: number, bitsPerKey: number) => runs * Math.pow(0.6185, bitsPerKey);

/* ----------------------------------------------------------- cache panel */
export function cacheTimeline(pinIndexFilter: boolean, steps = 120, compactAt = 40, seed = 7) {
  const rng = makeRng(seed);
  const FILES = 8;
  const BLOCKS_PER_FILE = 24;
  const CAP = 90;
  const cache = new Map<string, number>(); // key -> last used
  let clock = 0;
  const gen = Array.from({ length: FILES }, () => 0); // file generation: bumps when rewritten
  const out: { step: number; hitRate: number }[] = [];
  const touch = (key: string) => {
    clock++;
    if (cache.has(key)) {
      cache.set(key, clock);
      return true;
    }
    cache.set(key, clock);
    if (cache.size > CAP) {
      let oldest = '';
      let t = Infinity;
      for (const [k, v] of cache) if (v < t) { t = v; oldest = k; }
      cache.delete(oldest);
    }
    return false;
  };
  for (let s = 0; s < steps; s++) {
    if (s === compactAt) {
      // Compaction rewrites files 0..3 into new files: every cached block for them is now useless.
      for (let f = 0; f < 4; f++) gen[f]++;
    }
    let hits = 0;
    let lookups = 0;
    for (let q = 0; q < 40; q++) {
      const f = Math.min(FILES - 1, Math.floor(Math.pow(rng(), 1.6) * FILES)); // skewed toward low file numbers
      const blk = Math.floor(Math.pow(rng(), 2) * BLOCKS_PER_FILE);
      const id = `${f}.${gen[f]}`;
      // index + filter blocks for the file, then the data block
      for (const part of ['index', 'filter']) {
        lookups++;
        if (pinIndexFilter) hits++; // pinned: loaded when the new file is opened, never evicted
        else if (touch(`${id}:${part}`)) hits++;
      }
      lookups++;
      if (touch(`${id}:d${blk}`)) hits++;
    }
    out.push({ step: s, hitRate: hits / lookups });
  }
  return out;
}

export default function AmplificationTriangleLab() {
  const [panel, setPanel] = useState<'amp' | 'cache'>('amp');
  const [policy, setPolicy] = useState<Policy>('leveled');
  const [ratio, setRatio] = useState(10);
  const [overwrite, setOverwrite] = useState(50);
  const [bits, setBits] = useState(10);
  const [pin, setPin] = useState(false);

  const BATCHES = 1200;
  const BATCH = 250;
  // Deferred so the sliders stay responsive while the simulations (~30 ms each) catch up.
  const dRatio = useDeferredValue(ratio);
  const dOverwrite = useDeferredValue(overwrite);
  const keySpace = useMemo(() => keySpaceFor(dOverwrite, BATCHES * BATCH), [dOverwrite]);
  const sim = useMemo(() => simulate(policy, dRatio, keySpace, BATCHES, BATCH), [policy, dRatio, keySpace]);
  const other = useMemo(() => simulate(policy === 'leveled' ? 'tiered' : 'leveled', dRatio, keySpace, BATCHES, BATCH), [policy, dRatio, keySpace]);
  const btree = { writeAmp: 8192 / 100, spaceAmp: 1.5, runs: 1 };

  const cacheOff = useMemo(() => cacheTimeline(false), []);
  const cacheOn = useMemo(() => cacheTimeline(true), []);
  const series = pin ? cacheOn : cacheOff;
  const before = series.slice(30, 40).reduce((a, x) => a + x.hitRate, 0) / 10;
  const dip = Math.min(...series.slice(40, 55).map((x) => x.hitRate));

  // Ternary placement: normalize each design's costs against the three it is compared with.
  const pts = [
    { name: 'Leveled LSM', w: policy === 'leveled' ? sim.writeAmp : other.writeAmp, s: policy === 'leveled' ? sim.spaceAmp : other.spaceAmp, r: policy === 'leveled' ? sim.maxRuns : other.maxRuns, color: 'var(--viz-1)' },
    { name: 'Tiered LSM', w: policy === 'tiered' ? sim.writeAmp : other.writeAmp, s: policy === 'tiered' ? sim.spaceAmp : other.spaceAmp, r: policy === 'tiered' ? sim.maxRuns : other.maxRuns, color: 'var(--viz-2)' },
    { name: 'B+tree', w: btree.writeAmp, s: btree.spaceAmp, r: btree.runs, color: 'var(--viz-3)' },
  ];
  const maxW = Math.max(...pts.map((p) => p.w));
  const maxS = Math.max(...pts.map((p) => p.s));
  const maxR = Math.max(...pts.map((p) => p.r));
  const T = { a: [300, 20], b: [60, 250], c: [540, 250] }; // read, write, space corners
  const place = (p: (typeof pts)[number]) => {
    const r = p.r / maxR;
    const w = p.w / maxW;
    const s = p.s / maxS;
    const sum = r + w + s || 1;
    const [x, y] = [
      (r * T.a[0] + w * T.b[0] + s * T.c[0]) / sum,
      (r * T.a[1] + w * T.b[1] + s * T.c[1]) / sum,
    ];
    return { x, y };
  };

  return (
    <VizPanel
      title={panel === 'amp' ? 'Read, write and space amplification' : 'The cost the triangle hides: a cold cache after compaction'}
      subtitle={
        panel === 'amp'
          ? 'The same stream of key batches is flushed and compacted under each policy, and every entry written, kept or probed is counted. A point closer to a corner means that design pays more of that cost.'
          : 'Reads fill a block cache with data, index and filter blocks. At step 40 a compaction rewrites half the files: their blocks now belong to files that no longer exist.'
      }
      controls={
        <Segmented
          label="Panel"
          value={panel}
          onChange={setPanel}
          options={[
            { value: 'amp', label: 'Amplification' },
            { value: 'cache', label: 'Cache after compaction' },
          ]}
        />
      }
      legend={
        panel === 'amp' ? (
          <Legend items={pts.map((p) => ({ label: p.name, color: p.color, shape: 'dot' as const }))} />
        ) : (
          <Legend
            items={[
              { label: 'Block-cache hit rate', color: 'var(--viz-1)', shape: 'line' },
              { label: 'Compaction commits', color: 'var(--viz-ink)', shape: 'line' },
            ]}
          />
        )
      }
      stats={
        panel === 'amp' ? (
          <Stats
            items={[
              { label: `${policy} write amp (measured)`, value: `${fmtNum(sim.writeAmp, 1)}×` },
              { label: 'Space amp at end', value: `${fmtNum(sim.spaceAmp, 2)}×`, hint: `peak during the run ${fmtNum(sim.peakSpace, 2)}×` },
              { label: 'Most sorted runs a read had to probe', value: fmtNum(sim.maxRuns), hint: `${sim.runs} at the end of the run` },
              { label: `Worst absent-key block reads at ${bits} bits/key`, value: fmtNum(lookupCost(sim.maxRuns, bits), 3) },
            ]}
          />
        ) : (
          <Stats
            items={[
              { label: 'Hit rate before compaction', value: `${fmtNum(before * 100, 0)}%` },
              { label: 'Lowest hit rate after', value: `${fmtNum(dip * 100, 0)}%` },
              { label: 'Index/filter blocks', value: pin ? 'pinned' : 'in the LRU' },
            ]}
          />
        )
      }
      note={
        <Note>
          {panel === 'amp' ? (
            <>
              <strong>
                {policy === 'leveled' ? 'Leveled' : 'Tiered'} with size ratio {dRatio}: {fmtNum(sim.writeAmp, 1)}× write amplification, {fmtNum(sim.spaceAmp, 2)}× space, up to {sim.maxRuns} runs to probe.
              </strong>{' '}
              The {policy === 'leveled' ? 'tiered' : 'leveled'} policy on the same stream measures {fmtNum(other.writeAmp, 1)}× write, {fmtNum(other.spaceAmp, 2)}× space and up to {other.maxRuns} runs. A B+tree is placed from reference values: rewriting an 8 KB page for a 100-byte update is about {fmtNum(btree.writeAmp)}× write amplification, and pages around two-thirds full are about 1.5× space, with a single place to look.
            </>
          ) : (
            <>
              <strong>
                The hit rate falls from {fmtNum(before * 100, 0)}% to {fmtNum(dip * 100, 0)}% the moment the compaction commits, then recovers as reads re-warm the cache.
              </strong>{' '}
              {pin
                ? 'With index and filter blocks pinned, only data blocks go cold, so the dip is shallower.'
                : 'Every rewritten file also needs its index and filter blocks read again, so the dip is deeper. Check “pin index and filter blocks” to compare.'}{' '}
              The compaction’s own I/O is not what causes this — it is the cache losing everything it knew about that range.
            </>
          )}
        </Note>
      }
      table={
        panel === 'amp' ? (
          <table className="viz-table">
            <thead>
              <tr>
                <th>Design</th>
                <th>Write amp</th>
                <th>Space amp</th>
                <th>Most runs a read probes</th>
              </tr>
            </thead>
            <tbody>
              {pts.map((p) => (
                <tr key={p.name}>
                  <td>{p.name}</td>
                  <td>{fmtNum(p.w, 1)}×</td>
                  <td>{fmtNum(p.s, 2)}×</td>
                  <td>{p.r}</td>
                </tr>
              ))}
            </tbody>
          </table>
        ) : undefined
      }
    >
      {panel === 'amp' ? (
        <>
          <div className="viz-controls">
            <Segmented label="Policy" value={policy} onChange={setPolicy} options={[{ value: 'leveled', label: 'Leveled' }, { value: 'tiered', label: 'Tiered' }]} />
            <Slider label="Size ratio" min={2} max={20} value={ratio} onChange={setRatio} />
            <Slider label="Overwrites" min={0} max={90} step={10} value={overwrite} onChange={setOverwrite} format={(v) => `${v}%`} />
            <Slider label="Filter bits per key" min={2} max={16} value={bits} onChange={setBits} />
          </div>
          <svg width={600} height={290} role="img" aria-label="Designs placed between read, write and space amplification">
            <polygon points={`${T.a.join(',')} ${T.b.join(',')} ${T.c.join(',')}`} fill="var(--viz-plane)" stroke="var(--viz-axis)" />
            <text x={T.a[0]} y={T.a[1] - 6} textAnchor="middle" fill="var(--viz-ink)" fontSize={12}>
              read amplification
            </text>
            <text x={T.b[0] - 6} y={T.b[1] + 18} textAnchor="start" fill="var(--viz-ink)" fontSize={12}>
              write amplification
            </text>
            <text x={T.c[0] + 6} y={T.c[1] + 18} textAnchor="end" fill="var(--viz-ink)" fontSize={12}>
              space amplification
            </text>
            {pts.map((p) => {
              const { x, y } = place(p);
              return (
                <g key={p.name}>
                  <circle cx={x} cy={y} r={9} fill={p.color} stroke="var(--viz-surface)" strokeWidth={2} />
                  <text x={x + 13} y={y + 4} fontSize={11} fill="var(--viz-ink)">
                    {p.name}
                  </text>
                </g>
              );
            })}
          </svg>
        </>
      ) : (
        <>
          <div className="viz-controls">
            <Check label="Pin index and filter blocks" checked={pin} onChange={setPin} />
          </div>
          <svg width={600} height={170} role="img" aria-label="Block cache hit rate over time">
            {[0, 0.5, 1].map((v) => (
              <g key={v}>
                <line className="viz-grid-line" x1={40} x2={590} y1={140 - v * 120} y2={140 - v * 120} />
                <text x={34} y={144 - v * 120} textAnchor="end" fontSize={10}>
                  {v * 100}%
                </text>
              </g>
            ))}
            <line x1={40 + (40 / 120) * 550} x2={40 + (40 / 120) * 550} y1={16} y2={140} stroke="var(--viz-ink)" strokeDasharray="4 3" />
            <text x={44 + (40 / 120) * 550} y={14} fontSize={10}>
              compaction commits
            </text>
            <polyline
              fill="none"
              stroke="var(--viz-1)"
              strokeWidth={2}
              points={series.map((x) => `${40 + (x.step / 120) * 550},${140 - x.hitRate * 120}`).join(' ')}
            />
            <text x={315} y={164} textAnchor="middle" fontSize={10}>
              time (batches of reads)
            </text>
          </svg>
        </>
      )}
    </VizPanel>
  );
}
