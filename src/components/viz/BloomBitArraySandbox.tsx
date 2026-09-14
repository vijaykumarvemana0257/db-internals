import { useMemo, useState } from 'react';
import {
  VizPanel,
  Slider,
  Segmented,
  Check,
  Button,
  Legend,
  Stats,
  Note,
  TooltipHost,
  useTip,
  fmtBytes,
  fmtNum,
  useSize,
} from './Viz';

/**
 * A Bloom filter you can watch fill up.
 *
 * The bit array is real: keys are hashed with the double-hashing scheme LevelDB uses in
 * util/bloom.cc — one 32-bit hash, then a rotated delta added k times — so the k probe
 * positions for a key are genuinely derived from one hash, exactly as the engine does it.
 * The measured false-positive rate comes from probing 4 000 keys that were never inserted,
 * which is what makes the convergence onto (1 - e^(-kn/m))^k visible rather than asserted.
 */

const TRIALS = 8000;
const COLS = 64;
const K_MAX = 14;

/* ------------------------------------------------------------------ hashing */

/** FNV-1a, 32-bit. Stands in for LevelDB's BloomHash (a MurmurHash variant). */
function hash32(s: string): number {
  let h = 2166136261 >>> 0;
  for (let i = 0; i < s.length; i++) {
    h ^= s.charCodeAt(i);
    h = Math.imul(h, 16777619) >>> 0;
  }
  return h >>> 0;
}

/**
 * LevelDB's k probes from a single hash:
 *   const uint32_t delta = (h >> 17) | (h << 15);   // rotate right 17 bits
 *   for (j = 0; j < k; j++) { bitpos = h % bits; ...; h += delta; }
 * Two hashes' worth of spread out of one hash computation — cheap, and very slightly
 * worse than k independent hashes, which is why the measured rate sits a little above
 * the textbook one.
 */
function probesOf(h0: number, k: number, bits: number): number[] {
  let h = h0;
  const delta = ((h >>> 17) | (h << 15)) >>> 0;
  const out: number[] = [];
  for (let i = 0; i < k; i++) {
    out.push(h % bits);
    h = (h + delta) >>> 0;
  }
  return out;
}

const probes = (key: string, k: number, bits: number) => probesOf(hash32(key), k, bits);

const storedKey = (i: number) => `user:${100000 + i * 37}`;
const absentKey = (j: number) => `miss:${j}`;

/** Hashed once at module load: the measurement loop is then pure arithmetic. */
const ABSENT_H = Array.from({ length: TRIALS }, (_, j) => hash32(absentKey(j)));

/* ---------------------------------------------------------------- the filter */

type Filter = { bits: number; arr: Uint8Array; set: number };

function build(n: number, bpk: number, k: number): Filter {
  const bits = n * bpk;
  const arr = new Uint8Array(bits);
  for (let i = 0; i < n; i++) for (const p of probes(storedKey(i), k, bits)) arr[p] = 1;
  let set = 0;
  for (let i = 0; i < bits; i++) set += arr[i];
  return { bits, arr, set };
}

function mayMatch(f: Filter, key: string, k: number): boolean {
  for (const p of probes(key, k, f.bits)) if (!f.arr[p]) return false;
  return true;
}

function measureFp(f: Filter, k: number): number {
  let fp = 0;
  for (let j = 0; j < TRIALS; j++) {
    let h = ABSENT_H[j];
    const delta = ((h >>> 17) | (h << 15)) >>> 0;
    let hit = true;
    for (let i = 0; i < k; i++) {
      if (!f.arr[h % f.bits]) {
        hit = false;
        break;
      }
      h = (h + delta) >>> 0;
    }
    if (hit) fp++;
  }
  return fp / TRIALS;
}

/** (1 - e^(-kn/m))^k — the textbook rate, assuming independent probe positions. */
const predictFp = (n: number, bpk: number, k: number) => Math.pow(1 - Math.exp(-(k * n) / (n * bpk)), k);
const predictFill = (n: number, bpk: number, k: number) => 1 - Math.exp(-(k * n) / (n * bpk));
const optimalK = (bpk: number) => Math.max(1, Math.round(Math.LN2 * bpk));

/* -------------------------------------------------------------- the component */

export default function BloomBitArraySandbox() {
  const [n, setN] = useState(32);
  const [bpk, setBpk] = useState(10);
  const [k, setK] = useState(7);
  const [autoK, setAutoK] = useState(true);
  const [probe, setProbe] = useState<{ kind: 'stored' | 'absent'; idx: number } | null>(null);
  const [ref, width] = useSize(760);
  const tip = useTip();

  const kEff = autoK ? optimalK(bpk) : k;
  const f = useMemo(() => build(n, bpk, kEff), [n, bpk, kEff]);
  const measured = useMemo(() => measureFp(f, kEff), [f, kEff]);

  /** The whole FP-vs-k curve at this bits/key, measured as well as predicted. */
  const curve = useMemo(
    () =>
      Array.from({ length: K_MAX }, (_, i) => {
        const kk = i + 1;
        const ff = build(n, bpk, kk);
        return { k: kk, pred: predictFp(n, bpk, kk), meas: measureFp(ff, kk), fill: ff.set / ff.bits };
      }),
    [n, bpk],
  );

  const probeKey = probe ? (probe.kind === 'stored' ? storedKey(probe.idx) : absentKey(probe.idx)) : null;
  const probePos = probeKey ? probes(probeKey, kEff, f.bits) : [];
  const probeHit = probeKey ? mayMatch(f, probeKey, kEff) : false;
  const firstZero = probeKey ? probePos.findIndex((p) => !f.arr[p]) : -1;

  const findFp = () => {
    for (let j = 0; j < TRIALS * 4; j++) {
      if (mayMatch(f, absentKey(j), kEff)) {
        setProbe({ kind: 'absent', idx: j });
        return;
      }
    }
    setProbe(null);
  };

  /* ---------------------------------------------------------------- drawing */

  const svgW = Math.max(width, 620);
  const cell = Math.min(11, Math.floor((svgW - 130) / COLS));
  const rows = Math.ceil(f.bits / COLS);
  const gridX = 8;
  const gridY = 18;
  const gridH = rows * cell;

  const cTop = gridY + gridH + 46;
  const cH = 132;
  const cL = 46;
  const cR = svgW - 12;
  const cW = cR - cL;
  const xk = (kk: number) => cL + ((kk - 1) / (K_MAX - 1)) * cW;
  const yfp = (p: number) => {
    const e = Math.max(-6, Math.log10(Math.max(p, 1e-7)));
    return cTop + (-e / 6) * cH;
  };

  const line = (sel: (d: (typeof curve)[number]) => number) =>
    curve.map((d) => `${xk(d.k).toFixed(1)},${yfp(sel(d)).toFixed(1)}`).join(' ');

  const bytesPerMillion = (bpk * 1e6) / 8;

  return (
    <VizPanel
      title="Bloom filter sandbox: bits, probes and the false-positive rate"
      subtitle="One SSTable's filter, built with LevelDB's double-hashing scheme. Set bits per key and k, then probe keys that were never inserted and watch the measured rate land on (1 − e^(−kn/m))^k."
      controls={
        <>
          <Slider label="Keys in this file (n)" min={8} max={64} step={8} value={n} onChange={setN} />
          <Slider
            label="Bits per key (m/n)"
            min={2}
            max={20}
            value={bpk}
            onChange={setBpk}
            format={(v) => `${v} bits`}
          />
          <Slider
            label="Probes per key (k)"
            min={1}
            max={K_MAX}
            value={kEff}
            onChange={setK}
            disabled={autoK}
            format={(v) => `k = ${v}`}
          />
          <Check label="k = ln2 × bits/key (optimal)" checked={autoK} onChange={setAutoK} />
          <Segmented
            label="Probe a key"
            value={probe?.kind ?? 'stored'}
            onChange={(v) => setProbe({ kind: v, idx: probe ? probe.idx : 0 })}
            options={[
              { value: 'stored', label: 'stored', title: 'A key that was inserted — must return "maybe"' },
              { value: 'absent', label: 'absent', title: 'A key that was never inserted — usually returns "no"' },
            ]}
          />
          <Button
            onClick={() => setProbe({ kind: probe?.kind ?? 'absent', idx: (probe?.idx ?? -1) + 1 })}
            primary
          >
            Next probe
          </Button>
          <Button onClick={findFp} title="Scan absent keys until one collides on all k bits">
            Find a false positive
          </Button>
          <Button onClick={() => setProbe(null)}>Clear</Button>
        </>
      }
      legend={
        <Legend
          items={[
            { label: 'bit set by an inserted key', color: 'var(--viz-1)' },
            { label: 'bit still zero', color: 'var(--viz-neutral)' },
            { label: 'this probe’s k positions (ringed)', color: 'var(--viz-7)' },
            { label: 'predicted FP — (1−e^(−kn/m))^k', color: 'var(--viz-2)', shape: 'line' },
            { label: `measured FP over ${fmtNum(TRIALS)} absent probes`, color: 'var(--viz-8)', shape: 'dot' },
          ]}
        />
      }
      stats={
        <Stats
          items={[
            { label: 'Filter size', value: `${fmtNum(f.bits)} bits`, hint: 'm = n × bits per key' },
            {
              label: 'Bits set',
              value: `${((f.set / f.bits) * 100).toFixed(1)}%`,
              hint: `predicted 1 − e^(−kn/m) = ${(predictFill(n, bpk, kEff) * 100).toFixed(1)}%`,
            },
            { label: 'Predicted FP', value: `${(predictFp(n, bpk, kEff) * 100).toFixed(2)}%` },
            { label: 'Measured FP', value: `${(measured * 100).toFixed(2)}%`, hint: `${Math.round(measured * TRIALS)} of ${fmtNum(TRIALS)} absent keys said "maybe"` },
            { label: 'Optimal k here', value: optimalK(bpk), hint: 'ln2 × bits per key' },
            {
              label: 'Filter RAM per 1M keys',
              value: fmtBytes(bytesPerMillion),
              hint: 'What this bits/key setting costs in memory for every million keys in the table',
            },
          ]}
        />
      }
      note={
        <Note>
          {probeKey ? (
            probe?.kind === 'stored' ? (
              <>
                <strong>{probeKey} → “maybe”, and it must.</strong> All {kEff} bits this key hashes to were set
                when it was inserted, so a Bloom filter can never produce a false <em>negative</em>. That one-sided
                error is the whole contract: “no” is a proof, “maybe” is a hint that costs you a block read.
              </>
            ) : probeHit ? (
              <>
                <strong>{probeKey} → “maybe”, and it is lying.</strong> This key was never inserted, but all {kEff}{' '}
                of its bit positions happen to have been set by other keys. RocksDB will now fetch the index block
                and a data block for this file, search it, find nothing, and move to the next level — one wasted
                I/O, counted as <code>rocksdb.bloom.filter.full.positive</code> without a matching
                <code> .full.true.positive</code>.
              </>
            ) : (
              <>
                <strong>{probeKey} → “no”, after {firstZero + 1} probe{firstZero === 0 ? '' : 's'}.</strong> Bit{' '}
                {probePos[firstZero]} is still zero, so this key cannot be in the file and the lookup short-circuits
                immediately — the remaining {kEff - firstZero - 1} probes are never made. The whole SSTable is
                skipped without touching its index or any data block.
              </>
            )
          ) : (
            <>
              <strong>m = {fmtNum(f.bits)} bits holding {n} keys, k = {kEff} probes each.</strong> Each insert sets
              k bits; each lookup tests the same k. Raise bits per key and the array stays sparse and the
              false-positive rate falls; raise k too far at a fixed size and you saturate the array instead. The
              curve below shows both effects at once — and the measured points sit a little above the predicted
              line, because the k positions come from one hash plus a rotated delta rather than k independent
              hashes.
            </>
          )}
        </Note>
      }
      table={
        <>
          <table className="viz-table">
            <thead>
              <tr>
                <th>k</th>
                <th>Bits set</th>
                <th>Predicted FP</th>
                <th>Measured FP</th>
              </tr>
            </thead>
            <tbody>
              {curve.map((d) => (
                <tr key={d.k}>
                  <td>
                    {d.k}
                    {d.k === optimalK(bpk) ? ' (optimal)' : ''}
                  </td>
                  <td>{(d.fill * 100).toFixed(1)}%</td>
                  <td>{(d.pred * 100).toFixed(3)}%</td>
                  <td>{(d.meas * 100).toFixed(3)}%</td>
                </tr>
              ))}
            </tbody>
          </table>
          <table className="viz-table">
            <thead>
              <tr>
                <th>Bits per key</th>
                <th>Optimal k</th>
                <th>FP at optimal k</th>
                <th>RAM per 1M keys</th>
              </tr>
            </thead>
            <tbody>
              {[4, 6, 8, 10, 12, 16, 20].map((b) => (
                <tr key={b}>
                  <td>{b}</td>
                  <td>{optimalK(b)}</td>
                  <td>{(predictFp(n, b, optimalK(b)) * 100).toFixed(3)}%</td>
                  <td>{fmtBytes((b * 1e6) / 8)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </>
      }
    >
      <div ref={ref}>
        <TooltipHost>
          <svg
            width={svgW}
            height={cTop + cH + 34}
            role="img"
            aria-label="A Bloom filter bit array and its false-positive rate as a function of k"
          >
            <text x={gridX} y={12} fill="var(--viz-ink-2)">
              bit array — {fmtNum(f.bits)} bits, {fmtNum(f.set)} set
            </text>

            {Array.from({ length: f.bits }, (_, i) => {
              const r = Math.floor(i / COLS);
              const c = i % COLS;
              const ringIdx = probePos.indexOf(i);
              const ringed = ringIdx >= 0;
              return (
                <rect
                  key={i}
                  x={gridX + c * cell}
                  y={gridY + r * cell}
                  width={cell - 1}
                  height={cell - 1}
                  rx={1.5}
                  fill={f.arr[i] ? 'var(--viz-1)' : 'var(--viz-neutral)'}
                  stroke={ringed ? 'var(--viz-7)' : 'var(--viz-border)'}
                  strokeWidth={ringed ? 2 : 0.5}
                />
              );
            })}

            {probeKey ? (
              <text x={gridX} y={gridY + gridH + 18} fill="var(--viz-ink-2)">
                {probeKey} → bits {probePos.join(', ')} → {probeHit ? 'maybe (read the file)' : 'no (skip the file)'}
              </text>
            ) : null}

            {/* ---- FP against k ---- */}
            <text x={8} y={cTop - 12} fill="var(--viz-ink-2)">
              false-positive rate against k, at {bpk} bits per key (log scale)
            </text>
            {[0, -1, -2, -3, -4, -5, -6].map((e) => (
              <g key={e}>
                <line x1={cL} x2={cR} y1={yfp(Math.pow(10, e))} y2={yfp(Math.pow(10, e))} className="viz-grid" stroke="var(--viz-grid)" />
                <text x={cL - 6} y={yfp(Math.pow(10, e)) + 4} textAnchor="end" fill="var(--viz-ink-muted)">
                  {e === 0 ? '100%' : e === -1 ? '10%' : e === -2 ? '1%' : `1e${e + 2}%`}
                </text>
              </g>
            ))}
            <line
              x1={xk(optimalK(bpk))}
              x2={xk(optimalK(bpk))}
              y1={cTop}
              y2={cTop + cH}
              stroke="var(--viz-ink-muted)"
              strokeDasharray="4 3"
            />
            <text x={xk(optimalK(bpk)) + 5} y={cTop + 12} fill="var(--viz-ink-muted)">
              k = ln2 × m/n
            </text>
            <polyline points={line((d) => d.pred)} fill="none" stroke="var(--viz-2)" strokeWidth={2} />
            {curve.map((d) => (
              <g key={d.k} {...tip(
                <>
                  <strong>k = {d.k}</strong>
                  <br />
                  predicted {(d.pred * 100).toFixed(3)}% · measured {(d.meas * 100).toFixed(3)}%
                  <br />
                  {(d.fill * 100).toFixed(1)}% of bits set
                </>,
              )} style={{ cursor: 'help' }}>
                <circle cx={xk(d.k)} cy={yfp(d.meas)} r={d.k === kEff ? 5 : 3.5} fill="var(--viz-8)" />
                <rect x={xk(d.k) - 9} y={cTop} width={18} height={cH} fill="transparent" />
                <text x={xk(d.k)} y={cTop + cH + 16} textAnchor="middle" fill={d.k === kEff ? 'var(--viz-ink)' : 'var(--viz-ink-muted)'}>
                  {d.k}
                </text>
              </g>
            ))}
            <text x={cL} y={cTop + cH + 30} fill="var(--viz-ink-muted)">
              probes per key (k)
            </text>
          </svg>
        </TooltipHost>
      </div>
    </VizPanel>
  );
}
