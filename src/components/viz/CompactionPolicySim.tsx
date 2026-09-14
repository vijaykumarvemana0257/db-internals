import { useMemo, useState } from 'react';
import { VizPanel, Segmented, Slider, Choice, Button, Legend, Stats, Note, fmtNum } from './Viz';

/**
 * File-level compaction simulator. Files carry real key ranges, so overlap, trivial moves and
 * whole-window expiry fall out of the mechanics rather than being scripted.
 * Units: a flush writes one file of size 1 (think one memtable). Merges of overlapping random
 * keys shrink by DEDUP to model overwritten versions being removed.
 */
type Policy = 'leveled' | 'tiered' | 'twcs' | 'lazy';
type Workload = 'random' | 'monotonic' | 'timeseries';
type F = { id: number; lo: number; hi: number; size: number; level: number; born: number; corrupt?: boolean };
type Event = { t: number; kind: 'compact' | 'trivial' | 'expire' | 'fail'; level: number; inputs: number; outputs: number; bytes: number };

const KEYSPACE = 1000;
const DEDUP = 0.85;
const L0_TRIGGER_DEFAULT = 4;
const TTL = 60; // flushes, for the time-series workload
const WINDOW = 10; // flushes per TWCS window

const overlaps = (a: F, lo: number, hi: number) => a.lo <= hi && a.hi >= lo;

export function run(policy: Policy, workload: Workload, ratio: number, l0Trigger: number, flushes: number, corruptAt: number | null, quarantineAt: number | null) {
  let files: F[] = [];
  let nextId = 1;
  const events: Event[] = [];
  const timeline: { t: number; debt: number; l0: number; p99: number; state: 'ok' | 'slow' | 'stop' }[] = [];
  let written = 0;
  let ingested = 0;
  let cold = 0;
  let failures = 0;

  const target = (level: number) => l0Trigger * Math.pow(ratio, level - 1);
  const levelSize = (level: number) => files.filter((f) => f.level === level).reduce((a, f) => a + f.size, 0);

  const mergeInto = (inputs: F[], outLevel: number, t: number, kind: Event['kind'] = 'compact', onePiece = false) => {
    if (inputs.some((f) => f.corrupt)) {
      failures++;
      events.push({ t, kind: 'fail', level: outLevel, inputs: inputs.length, outputs: 0, bytes: 0 });
      return false;
    }
    const lo = Math.min(...inputs.map((f) => f.lo));
    const hi = Math.max(...inputs.map((f) => f.hi));
    const anyOverlap = inputs.length > 1 && inputs.some((a, i) => inputs.some((b, j) => i < j && overlaps(a, b.lo, b.hi)));
    const total = inputs.reduce((a, f) => a + f.size, 0) * (workload === 'random' && anyOverlap ? DEDUP : 1);
    const pieces = onePiece ? 1 : Math.max(1, Math.round(total));
    const span = (hi - lo) / pieces;
    files = files.filter((f) => !inputs.includes(f));
    for (let p = 0; p < pieces; p++) {
      files.push({ id: nextId++, lo: lo + p * span, hi: lo + (p + 1) * span - (p === pieces - 1 ? 0 : 0.001), size: total / pieces, level: outLevel, born: Math.max(...inputs.map((f) => f.born)) });
    }
    written += total;
    cold += total;
    events.push({ t, kind, level: outLevel, inputs: inputs.length, outputs: pieces, bytes: total });
    return true;
  };

  for (let t = 1; t <= flushes; t++) {
    if (corruptAt !== null && t === corruptAt) {
      // A bad block in L1 blocks every L0 → L1 merge when keys are random, so it is the one that surfaces as L0 buildup.
      const victim = files.filter((f) => f.level === 1).sort((a, b) => b.size - a.size)[0] ?? files.filter((f) => f.level >= 1)[0];
      if (victim) victim.corrupt = true;
    }
    if (quarantineAt !== null && t === quarantineAt) files = files.filter((f) => !f.corrupt);

    // flush — unless L0 has reached the stop trigger, in which case foreground writes are blocked
    const stopped = files.filter((f) => f.level === 0).length >= 36;
    const lo = workload === 'random' ? 0 : ((t - 1) * 7) % 100000;
    const hi = workload === 'random' ? KEYSPACE : lo + 6.99;
    if (!stopped) {
      files.push({ id: nextId++, lo, hi, size: 1, level: 0, born: t });
      written += 1;
      ingested += 1;
    }

    // TTL drop (FIFO-style for time series): whole files past TTL
    if (workload === 'timeseries' && (policy === 'twcs' || policy === 'tiered' || policy === 'leveled' || policy === 'lazy')) {
      const expired = files.filter((f) => t - f.born >= TTL && (policy === 'twcs' ? true : false));
      if (expired.length) {
        files = files.filter((f) => !expired.includes(f));
        events.push({ t, kind: 'expire', level: -1, inputs: expired.length, outputs: 0, bytes: 0 });
      }
    }

    let guard = 0;
    let progressed = true;
    while (progressed && guard++ < 50) {
      progressed = false;
      if (policy === 'leveled' || policy === 'lazy') {
        const l0 = files.filter((f) => f.level === 0);
        if (l0.length >= l0Trigger) {
          const lo0 = Math.min(...l0.map((f) => f.lo));
          const hi0 = Math.max(...l0.map((f) => f.hi));
          const l1 = files.filter((f) => f.level === 1 && overlaps(f, lo0, hi0));
          if (policy === 'lazy') {
            // tier the upper levels: L0 files become one new run in L1 without merging existing L1 runs
            if (mergeInto(l0, 1, t)) progressed = true;
          } else {
            const l0Overlap = l0.some((a, i) => l0.some((b, j) => i < j && overlaps(a, b.lo, b.hi)));
            if (!l1.length && !l0Overlap) {
              l0.forEach((f) => (f.level = 1));
              events.push({ t, kind: 'trivial', level: 1, inputs: l0.length, outputs: l0.length, bytes: 0 });
              progressed = true;
            } else if (mergeInto([...l0, ...l1], 1, t)) progressed = true;
          }
        }
        const maxLevel = Math.max(1, ...files.map((f) => f.level));
        for (let lvl = 1; lvl <= maxLevel && !progressed; lvl++) {
          if (policy === 'lazy' && lvl < maxLevel) {
            // lazy leveling: tier every level but the last; merge a level's runs when it exceeds target
            if (levelSize(lvl) > target(lvl)) {
              const inputs = files.filter((f) => f.level === lvl);
              const deeper = files.filter((f) => f.level === lvl + 1);
              const isLast = lvl + 1 >= maxLevel;
              if (mergeInto(isLast ? [...inputs, ...deeper] : inputs, lvl + 1, t)) progressed = true;
            }
            continue;
          }
          if (levelSize(lvl) > target(lvl)) {
            // pick the file with the smallest overlap with the next level (kMinOverlappingRatio-style)
            const cand = files
              .filter((f) => f.level === lvl && !f.corrupt)
              .concat(files.filter((f) => f.level === lvl && f.corrupt))
              .map((f) => ({ f, ov: files.filter((g) => g.level === lvl + 1 && overlaps(g, f.lo, f.hi)) }))
              .sort((a, b) => a.ov.reduce((s, g) => s + g.size, 0) / a.f.size - b.ov.reduce((s, g) => s + g.size, 0) / b.f.size)[0];
            if (!cand) break;
            if (!cand.ov.length && !cand.f.corrupt) {
              cand.f.level = lvl + 1;
              events.push({ t, kind: 'trivial', level: lvl + 1, inputs: 1, outputs: 1, bytes: 0 });
              progressed = true;
            } else if (mergeInto([cand.f, ...cand.ov], lvl + 1, t)) progressed = true;
          }
        }
      } else if (policy === 'tiered') {
        const maxLevel = Math.max(0, ...files.map((f) => f.level));
        for (let lvl = 0; lvl <= maxLevel && !progressed; lvl++) {
          const tier = files.filter((f) => f.level === lvl);
          const runs = lvl === 0 ? tier.length : new Set(tier.map((f) => f.born)).size;
          if (runs >= ratio) {
            if (mergeInto(tier, lvl + 1, t)) progressed = true;
          }
        }
      } else {
        // TWCS: within each closed time window, merge its files into one; never merge across windows
        const windows = new Map<number, F[]>();
        for (const f of files) {
          const w = Math.floor((f.born - 1) / WINDOW);
          (windows.get(w) ?? windows.set(w, []).get(w)!).push(f);
        }
        const current = Math.floor((t - 1) / WINDOW);
        for (const [w, fs] of windows) {
          if (w < current && fs.length > 1) {
            // TWCS writes one file per closed window, so a merged window is never revisited
            if (mergeInto(fs, 1, t, 'compact', true)) progressed = true;
            break;
          }
        }
      }
    }

    const l0 = files.filter((f) => f.level === 0).length;
    let debt = 0;
    if (policy === 'leveled') {
      debt = Math.max(0, l0 - l0Trigger);
      const maxLevel = Math.max(1, ...files.map((f) => f.level));
      for (let lvl = 1; lvl <= maxLevel; lvl++) debt += Math.max(0, levelSize(lvl) - target(lvl));
    } else {
      debt = files.filter((f) => f.corrupt).length ? failures : Math.max(0, l0 - l0Trigger);
    }
    cold *= 0.8;
    timeline.push({ t, debt, l0, p99: 1 + cold / 4, state: l0 >= 36 ? 'stop' : l0 >= 20 ? 'slow' : 'ok' });
  }
  return { files, events, timeline, writeAmp: written / Math.max(1, ingested), failures, trivial: events.filter((e) => e.kind === 'trivial').length, expired: events.filter((e) => e.kind === 'expire').reduce((a, e) => a + e.inputs, 0) };
}

const LEVEL_COLOR = ['var(--viz-dirty)', 'var(--viz-1)', 'var(--viz-3)', 'var(--viz-4)', 'var(--viz-5)', 'var(--viz-6)', 'var(--viz-7)'];

export default function CompactionPolicySim() {
  const [policy, setPolicy] = useState<Policy>('leveled');
  const [workload, setWorkload] = useState<Workload>('random');
  const [ratio, setRatio] = useState(4);
  const [l0Trigger, setL0Trigger] = useState(L0_TRIGGER_DEFAULT);
  const [flushes, setFlushes] = useState(120);
  const [corrupt, setCorrupt] = useState(false);
  const [quarantine, setQuarantine] = useState(false);

  const corruptAt = corrupt ? 40 : null;
  const quarantineAt = corrupt && quarantine ? 90 : null;
  const sim = useMemo(() => run(policy, workload, ratio, l0Trigger, flushes, corruptAt, quarantineAt), [policy, workload, ratio, l0Trigger, flushes, corruptAt, quarantineAt]);

  const cur = sim.timeline[sim.timeline.length - 1];
  const maxLevel = Math.max(0, ...sim.files.map((f) => f.level));
  const last = [...sim.events].reverse().find((e) => e.kind !== 'expire');
  const peakL0 = Math.max(...sim.timeline.map((x) => x.l0));
  const maxDebt = Math.max(1, ...sim.timeline.map((x) => x.debt));
  const maxP99 = Math.max(...sim.timeline.map((x) => x.p99));
  const keyMax = workload === 'random' ? KEYSPACE : Math.max(...sim.files.map((f) => f.hi), 10);
  const keyMin = workload === 'random' ? 0 : Math.min(...sim.files.map((f) => f.lo), 0);
  const W = 660;

  return (
    <VizPanel
      title="Compaction policies, file by file"
      subtitle="Each flush adds one file. Pick a policy and a workload, and watch how files spread across levels, which merges happen, and what they cost. Every file is drawn at its real key range."
      controls={
        <>
          <Choice
            label="Policy"
            value={policy}
            onChange={setPolicy}
            options={[
              { value: 'leveled', label: 'Leveled (RocksDB default, Cassandra LCS)' },
              { value: 'tiered', label: 'Tiered (universal, Cassandra STCS)' },
              { value: 'twcs', label: 'Time-window (TWCS)' },
              { value: 'lazy', label: 'Lazy leveling' },
            ]}
          />
          <Choice
            label="Workload"
            value={workload}
            onChange={setWorkload}
            options={[
              { value: 'random', label: 'Random keys' },
              { value: 'monotonic', label: 'Increasing keys' },
              { value: 'timeseries', label: 'Time series with TTL' },
            ]}
          />
          <Slider label="Size ratio" min={2} max={10} value={ratio} onChange={setRatio} />
          <Slider label="L0 trigger" min={2} max={8} value={l0Trigger} onChange={setL0Trigger} />
          <Slider label="Flushes" min={20} max={200} step={10} value={flushes} onChange={setFlushes} />
        </>
      }
      legend={
        <Legend
          items={[
            { label: 'L0 (overlapping)', color: LEVEL_COLOR[0] },
            { label: 'L1', color: LEVEL_COLOR[1] },
            { label: 'L2', color: LEVEL_COLOR[2] },
            { label: 'L3+', color: LEVEL_COLOR[3] },
            { label: '✕ corrupt file', color: 'var(--viz-critical)' },
          ]}
        />
      }
      stats={
        <Stats
          items={[
            { label: 'Write amplification', value: `${fmtNum(sim.writeAmp, 1)}×` },
            { label: 'Files / levels', value: `${sim.files.length} / ${maxLevel + 1}` },
            { label: 'Trivial moves', value: fmtNum(sim.trivial), hint: 'Files moved down a level without rewriting a byte' },
            { label: 'Files expired whole', value: fmtNum(sim.expired) },
            { label: 'Peak L0 files', value: fmtNum(peakL0) },
            { label: 'Failed compactions', value: fmtNum(sim.failures) },
          ]}
        />
      }
      note={
        <Note>
          {corrupt && sim.failures > 0 ? (
            <>
              <strong>A corrupt file made {fmtNum(sim.failures)} compaction attempts fail.</strong> Nothing reports a read error — the file is only unreadable when compaction tries to merge it — so the symptom is L0 files climbing to {peakL0}
              {peakL0 >= 20 ? `, past the slowdown trigger${peakL0 >= 36 ? ' and the stop trigger' : ''}` : ''}, and compaction debt that never shrinks.{' '}
              {quarantine ? 'At flush 90 the file is removed and rebuilt from a replica, and compaction immediately catches up.' : 'Check “quarantine the file at flush 90” to see compaction recover.'}
            </>
          ) : (
            <>
              <strong>
                {policy === 'leveled' ? 'Leveled' : policy === 'tiered' ? 'Tiered' : policy === 'twcs' ? 'Time-window' : 'Lazy leveling'} on {workload === 'random' ? 'random keys' : workload === 'monotonic' ? 'increasing keys' : 'a TTL time series'}: {fmtNum(sim.writeAmp, 1)}× write amplification.
              </strong>{' '}
              {last ? `Last merge at flush ${last.t}: ${last.kind === 'trivial' ? `a trivial move of ${last.inputs} file${last.inputs === 1 ? '' : 's'} into L${last.level}, rewriting nothing` : `${last.inputs} input files into ${last.outputs} output file${last.outputs === 1 ? '' : 's'} in L${last.level}, writing ${fmtNum(last.bytes, 1)} units`}.` : 'No merges yet.'}{' '}
              {workload === 'monotonic' && sim.trivial ? `${sim.trivial} trivial moves: with increasing keys, new files rarely overlap older ones, so many are moved down without being rewritten.` : ''}
              {policy === 'twcs' && sim.expired ? `${sim.expired} files expired as whole windows aged past the TTL — dropped without being read or rewritten.` : ''}
            </>
          )}
        </Note>
      }
      table={
        <table className="viz-table">
          <thead>
            <tr>
              <th>Flush</th>
              <th>Event</th>
              <th>Into</th>
              <th>Inputs → outputs</th>
              <th>Units written</th>
            </tr>
          </thead>
          <tbody>
            {sim.events.slice(-14).map((e, i) => (
              <tr key={i}>
                <td>{e.t}</td>
                <td>{e.kind}</td>
                <td>{e.level < 0 ? '—' : `L${e.level}`}</td>
                <td>
                  {e.inputs} → {e.outputs}
                </td>
                <td>{fmtNum(e.bytes, 1)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      }
    >
      <div className="viz-controls">
        <Button onClick={() => { setCorrupt((c) => !c); setQuarantine(false); }}>{corrupt ? 'Remove the corruption' : 'Corrupt a block at flush 40'}</Button>
        {corrupt ? (
          <label className="viz-control" style={{ flexDirection: 'row', alignItems: 'center', gap: '.4rem' }}>
            <input type="checkbox" checked={quarantine} onChange={(e) => setQuarantine(e.currentTarget.checked)} />
            <span>Quarantine the file at flush 90</span>
          </label>
        ) : null}
      </div>

      <svg width={W} height={(maxLevel + 1) * 30 + 150} role="img" aria-label="Files by level and key range, with debt and latency over time">
        {Array.from({ length: maxLevel + 1 }, (_, lvl) => {
          const y = 8 + lvl * 30;
          const lvlFiles = sim.files.filter((f) => f.level === lvl);
          return (
            <g key={lvl}>
              <text x={0} y={y + 16} fontSize={11} fill="var(--viz-ink)">
                L{lvl}
              </text>
              <rect x={30} y={y} width={W - 40} height={24} rx={3} fill="var(--viz-plane)" stroke="var(--viz-border)" />
              {lvlFiles.map((f, i) => {
                const x = 30 + ((f.lo - keyMin) / Math.max(1, keyMax - keyMin)) * (W - 40);
                const w = Math.max(2, ((f.hi - f.lo) / Math.max(1, keyMax - keyMin)) * (W - 40));
                const stack = lvl === 0 ? (i % 4) * 5 : 0;
                return (
                  <rect key={f.id} x={x} y={y + 2 + stack} width={w} height={lvl === 0 ? 5 : 20} rx={2} fill={f.corrupt ? 'var(--viz-critical)' : LEVEL_COLOR[Math.min(lvl, LEVEL_COLOR.length - 1)]} stroke="var(--viz-surface)" strokeWidth={1} opacity={0.9} />
                );
              })}
              <text x={W - 12} y={y + 16} fontSize={10} textAnchor="end">
                {lvlFiles.length} files
              </text>
            </g>
          );
        })}
        {(() => {
          const top = (maxLevel + 1) * 30 + 20;
          const x = (t: number) => 30 + ((t - 1) / Math.max(1, flushes - 1)) * (W - 40);
          return (
            <g>
              <text x={0} y={top + 10} fontSize={11} fill="var(--viz-ink)">
                debt
              </text>
              <polyline fill="none" stroke="var(--viz-2)" strokeWidth={2} points={sim.timeline.map((p) => `${x(p.t)},${top + 44 - (p.debt / maxDebt) * 40}`).join(' ')} />
              <text x={0} y={top + 72} fontSize={11} fill="var(--viz-ink)">
                p99
              </text>
              <polyline fill="none" stroke="var(--viz-1)" strokeWidth={2} points={sim.timeline.map((p) => `${x(p.t)},${top + 106 - ((p.p99 - 1) / Math.max(0.01, maxP99 - 1)) * 40}`).join(' ')} />
              {sim.timeline.map((p) => (p.state !== 'ok' ? <rect key={p.t} x={x(p.t) - 1.5} y={top + 112} width={3} height={8} fill={p.state === 'stop' ? 'var(--viz-critical)' : 'var(--viz-warning)'} /> : null))}
              <text x={W - 10} y={top + 128} fontSize={10} textAnchor="end">
                ▲ slowdown / ■ stop (L0 ≥ 20 / 36)
              </text>
            </g>
          );
        })()}
      </svg>
    </VizPanel>
  );
}
