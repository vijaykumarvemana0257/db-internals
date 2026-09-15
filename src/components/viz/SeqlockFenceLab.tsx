import { useMemo, useState } from 'react';
import { VizPanel, Choice, Check, Slider, Button, Legend, Stats, Note, makeRng, fmtNum } from './Viz';

/**
 * A seqlock-style optimistic read racing one writer, replayed on two memory models.
 *
 * The code being modelled is Linux's seqcount (include/linux/seqlock.h):
 *
 *   writer                               reader
 *   w1  seq = 1          (odd)           r1  v0 = smp_load_acquire(seq); if (v0 & 1) spin
 *       smp_wmb()   <- fence A           r2  x  = a
 *   w2  a = 2                            r3  y  = b
 *   w3  b = 2                                smp_rmb()   <- fence R
 *       smp_wmb()   <- fence B           r4  v1 = READ_ONCE(seq); if (v1 != v0) retry
 *   w4  seq = 2          (even)
 *
 * Every store has an execution time and a propagation delay: the time it would take to become visible to
 * another core if nothing constrained it. Every load binds the value that is visible when it is satisfied.
 *
 *  x86-TSO (Owens, Sarkar, Sewell 2009): stores sit in a per-core FIFO buffer and reach memory in program
 *    order, so a store is visible at max(own time, previous store's visible time). Loads are satisfied in
 *    program order. smp_wmb()/smp_rmb() compile to barrier() on x86, so the fences change nothing here.
 *  Weak ordering (after the Power and ARM architecture models in Alglave, Maranget, Tautschnig 2014):
 *    stores to different locations may become visible in any order; stores to the same location stay in
 *    program order. A write fence makes every store before it visible before any store after it. Without
 *    the read fence, the re-check load r4 may be satisfied early — before the payload loads — at the
 *    schedule's `early` time. The begin load r1 is always satisfied first: seqcount reads it with smp_load_acquire(),
 *    which carries its own read ordering (READ_ONCE() then lwsync on 64-bit Power), so no lab fence removes it.
 *  Compiler barrier: when the re-check is a plain load of a variable the reader never writes, the compiler
 *    may merge it with the first load (memory-barriers.txt, "COMPILER BARRIER"). Modelled as v1 = v0, on
 *    both machines, because it is a compiler transformation, not a CPU one.
 *
 * Times are abstract units; the batch run draws 2,000 schedules from makeRng with a fixed seed.
 */

export type Machine = 'weak' | 'tso';
export type Loc = 'seq' | 'a' | 'b';
export type Fences = { wmbBegin: boolean; wmbEnd: boolean; rmb: boolean; realRecheck: boolean };
export type Schedule = { w: [number, number][]; r: [number, number, number, number]; early: number };
export type Outcome = 'odd' | 'recheck' | 'consistent' | 'torn';

export const STORES: { op: string; loc: Loc; val: number; label: string }[] = [
  { op: 'w1', loc: 'seq', val: 1, label: 'seq=1' },
  { op: 'w2', loc: 'a', val: 2, label: 'a=2' },
  { op: 'w3', loc: 'b', val: 2, label: 'b=2' },
  { op: 'w4', loc: 'seq', val: 2, label: 'seq=2' },
];
export const LOADS: { op: string; loc: Loc }[] = [
  { op: 'r1', loc: 'seq' },
  { op: 'r2', loc: 'a' },
  { op: 'r3', loc: 'b' },
  { op: 'r4', loc: 'seq' },
];
const INITIAL: Record<Loc, number> = { seq: 0, a: 1, b: 1 };
export const ALL_FENCES: Fences = { wmbBegin: true, wmbEnd: true, rmb: true, realRecheck: true };

export const PRESETS: { value: string; label: string; s: Schedule }[] = [
  {
    value: 'wmbBegin',
    label: 'The payload store outruns seq = 1',
    s: { w: [[10, 26], [13, 2], [16, 22], [19, 2]], r: [20, 24, 28, 32], early: 22 },
  },
  {
    value: 'wmbEnd',
    label: 'seq = 2 outruns the payload store',
    s: { w: [[10, 1], [13, 1], [16, 24], [19, 1]], r: [24, 28, 32, 36], early: 26 },
  },
  {
    value: 'rmb',
    label: 'The re-check load is satisfied early',
    s: { w: [[12, 1], [14, 1], [22, 1], [24, 1]], r: [10, 17, 20, 28], early: 11 },
  },
  {
    value: 'apart',
    label: 'The reader finishes before the writer starts',
    s: { w: [[30, 2], [33, 2], [36, 2], [39, 2]], r: [5, 9, 13, 17], early: 7 },
  },
];

/** When each store becomes visible to the reader's core. */
export function visibleTimes(s: Schedule, m: Machine, f: Fences): number[] {
  const own = s.w.map(([t, d]) => t + d);
  if (m === 'tso') {
    const out: number[] = [];
    own.forEach((v, i) => out.push(i === 0 ? v : Math.max(v, out[i - 1])));
    return out;
  }
  const v = own.slice();
  v[3] = Math.max(v[3], v[0]); // two stores to seq keep their program order
  if (f.wmbBegin) {
    v[1] = Math.max(v[1], v[0]);
    v[2] = Math.max(v[2], v[0]);
    v[3] = Math.max(v[3], v[0]);
  }
  if (f.wmbEnd) v[3] = Math.max(v[3], v[0], v[1], v[2]);
  return v;
}

/** When each load is satisfied; the re-check is NaN when the compiler folded it into the first load. */
export function loadTimes(s: Schedule, m: Machine, f: Fences): number[] {
  const r4 = !f.realRecheck ? NaN : m === 'weak' && !f.rmb ? s.early : s.r[3];
  return [s.r[0], s.r[1], s.r[2], r4];
}

function valueAt(loc: Loc, t: number, vis: number[]) {
  let v = INITIAL[loc];
  STORES.forEach((st, i) => {
    if (st.loc === loc && vis[i] <= t) v = st.val;
  });
  return v;
}

export type Eval = {
  vis: number[];
  sat: number[];
  v0: number;
  x: number | null;
  y: number | null;
  v1: number | null;
  outcome: Outcome;
  early: boolean;
  folded: boolean;
};

export function evaluate(s: Schedule, m: Machine, f: Fences): Eval {
  const vis = visibleTimes(s, m, f);
  const sat = loadTimes(s, m, f);
  const v0 = valueAt('seq', sat[0], vis);
  const folded = !f.realRecheck;
  const early = m === 'weak' && f.realRecheck && !f.rmb && s.early < s.r[1];
  if (v0 & 1) return { vis, sat, v0, x: null, y: null, v1: null, outcome: 'odd', early, folded };
  const x = valueAt('a', sat[1], vis);
  const y = valueAt('b', sat[2], vis);
  const v1 = folded ? v0 : valueAt('seq', sat[3], vis);
  const outcome: Outcome = v1 !== v0 ? 'recheck' : x === y ? 'consistent' : 'torn';
  return { vis, sat, v0, x, y, v1, outcome, early, folded };
}

export function randomSchedule(rng: () => number, skew: number): Schedule {
  const w1 = 5 + 20 * rng();
  const w2 = w1 + 1 + 5 * rng();
  const w3 = w2 + 1 + 5 * rng();
  const w4 = w3 + 1 + 5 * rng();
  const d = () => skew * rng() * rng();
  const r1 = 40 * rng();
  const r2 = r1 + 1 + 4 * rng();
  const r3 = r2 + 1 + 4 * rng();
  const r4 = r3 + 1 + 4 * rng();
  return { w: [[w1, d()], [w2, d()], [w3, d()], [w4, d()]], r: [r1, r2, r3, r4], early: r1 + (r4 - r1) * rng() };
}

export type Tally = Record<Outcome, number> & { runs: number };

export function batch(f: Fences, skew: number, runs = 2000, seed = 7): Record<Machine, Tally> {
  const rng = makeRng(seed);
  const blank = (): Tally => ({ runs, odd: 0, recheck: 0, consistent: 0, torn: 0 });
  const out: Record<Machine, Tally> = { weak: blank(), tso: blank() };
  for (let i = 0; i < runs; i++) {
    const s = randomSchedule(rng, skew);
    out.weak[evaluate(s, 'weak', f).outcome]++;
    out.tso[evaluate(s, 'tso', f).outcome]++;
  }
  return out;
}

export function describe(e: Eval) {
  switch (e.outcome) {
    case 'odd':
      return `begin loads seq = ${e.v0} (odd): spin, then retry`;
    case 'recheck':
      return `re-check loads seq = ${e.v1} ≠ ${e.v0}: retry`;
    case 'consistent':
      return `accepts a=${e.x}, b=${e.y}: consistent`;
    default:
      return `accepts a=${e.x}, b=${e.y}: TORN`;
  }
}

const LOC_COLOR: Record<Loc, string> = { seq: 'var(--viz-1)', a: 'var(--viz-2)', b: 'var(--viz-3)' };
const OUTCOME_COLOR: Record<Outcome, string> = {
  odd: 'var(--viz-warning)',
  recheck: 'var(--viz-warning)',
  consistent: 'var(--viz-ink-2)',
  torn: 'var(--viz-critical)',
};

export default function SeqlockFenceLab() {
  const [preset, setPreset] = useState('wmbBegin');
  const [seed, setSeed] = useState(3);
  const [skew, setSkew] = useState(30);
  const [f, setF] = useState<Fences>(ALL_FENCES);
  const [step, setStep] = useState(999);

  const schedule = useMemo<Schedule>(() => {
    if (preset === 'random') return randomSchedule(makeRng(1000 + seed), skew);
    return (PRESETS.find((p) => p.value === preset) ?? PRESETS[0]).s;
  }, [preset, seed, skew]);

  const weak = evaluate(schedule, 'weak', f);
  const tso = evaluate(schedule, 'tso', f);
  const tally = useMemo(() => batch(f, skew), [f, skew]);

  // replay cursor over every distinct event time on either machine
  const times = useMemo(() => {
    const all = [...schedule.w.map(([t]) => t), ...weak.vis, ...tso.vis, ...weak.sat, ...tso.sat].filter((t) => Number.isFinite(t));
    return Array.from(new Set(all.map((t) => Math.round(t * 100) / 100))).sort((a, b) => a - b);
  }, [schedule, weak, tso]);
  const idx = Math.min(step, times.length - 1);
  const cursor = times[idx];
  const done = idx === times.length - 1;

  const set = (k: keyof Fences) => (v: boolean) => {
    setF((o) => ({ ...o, [k]: v }));
    setStep(999);
  };

  const W = 680;
  const LEFT = 118;
  const lo = Math.min(...times) - 2;
  const hi = Math.max(...times) + 3;
  const X = (t: number) => LEFT + ((t - lo) / (hi - lo)) * (W - LEFT - 16);
  const PANEL = 200;
  const H = PANEL * 2 + 8;
  // cursor values are rounded to 0.01, so allow the same tolerance the narration uses
  const shown = (t: number) => t <= cursor + 0.006;

  const panel = (m: Machine, e: Eval, y0: number) => {
    const readNotReached = e.outcome === 'odd';
    const wy = y0 + 46;
    const vy = y0 + 84;
    const ry = y0 + 150;
    // stack labels whose markers would overlap: level k sits k rows further from its lane
    const levels = (xs: number[], gap = 36) => {
      const order = xs.map((x, i) => [x, i] as const).filter(([x]) => Number.isFinite(x)).sort((a, b) => a[0] - b[0]);
      const out = new Array<number>(xs.length).fill(0);
      const placed: { x: number; lvl: number }[] = [];
      for (const [x, i] of order) {
        let lvl = 0;
        while (placed.some((q) => q.lvl === lvl && Math.abs(q.x - x) < gap)) lvl++;
        out[i] = lvl;
        placed.push({ x, lvl });
      }
      return out;
    };
    const visLvl = levels(e.vis.map((v) => X(v)));
    const readX = LOADS.map((_, i) => (readNotReached && i > 0 ? NaN : i === 3 && e.folded ? NaN : X(e.sat[i])));
    const readLvl = levels(readX, 40);
    const faint = m === 'tso';
    const finished = done;
    const title = m === 'weak' ? 'Weak ordering (Power/ARM-style model)' : 'x86-TSO (FIFO store buffer), same schedule';
    return (
      <g key={m}>
        <rect x={2} y={y0 + 2} width={W - 4} height={PANEL - 6} rx={6} fill="none" stroke="var(--viz-border)" />
        <text x={12} y={y0 + 20} fontSize={12} fontWeight={600} fill="var(--viz-ink)">
          {title}
        </text>
        {finished ? (
          <g>
            <rect x={W - 262} y={y0 + 8} width={252} height={18} rx={4} fill="var(--viz-surface)" stroke={OUTCOME_COLOR[e.outcome]} strokeWidth={2} />
            <text x={W - 136} y={y0 + 21} fontSize={11} textAnchor="middle" fill="var(--viz-ink)">
              {describe(e)}
            </text>
          </g>
        ) : null}
        {[
          ['writer executes', wy],
          ['visible to reader', vy],
          ['reader loads', ry],
        ].map(([label, y]) => (
          <g key={label as string}>
            <text x={12} y={(y as number) + 4} fontSize={11} fill="var(--viz-ink-2)">
              {label}
            </text>
            <line x1={LEFT - 6} x2={W - 12} y1={y as number} y2={y as number} stroke="var(--viz-grid)" />
          </g>
        ))}

        {/* write fences */}
        {[
          { on: f.wmbBegin, t: (schedule.w[0][0] + schedule.w[1][0]) / 2, name: 'A' },
          { on: f.wmbEnd, t: (schedule.w[2][0] + schedule.w[3][0]) / 2, name: 'B' },
        ].map((fe) =>
          fe.on ? (
            <g key={fe.name} opacity={faint ? 0.35 : 1}>
              <line x1={X(fe.t)} x2={X(fe.t)} y1={wy - 5} y2={wy + 14} stroke="var(--viz-ink-2)" strokeWidth={3} strokeDasharray="3 2" />
            </g>
          ) : null,
        )}
        {f.rmb && f.realRecheck ? (
          <line
            x1={X((schedule.r[2] + schedule.r[3]) / 2)}
            x2={X((schedule.r[2] + schedule.r[3]) / 2)}
            y1={ry - 5}
            y2={ry + 8}
            stroke="var(--viz-ink-2)"
            strokeWidth={3}
            strokeDasharray="3 2"
            opacity={faint ? 0.35 : 1}
          />
        ) : null}

        {/* stores: execution -> visibility */}
        {STORES.map((st, i) => {
          const t = schedule.w[i][0];
          const v = e.vis[i];
          const c = LOC_COLOR[st.loc];
          return (
            <g key={st.op}>
              <g opacity={shown(t) ? 1 : 0.2}>
                <circle cx={X(t)} cy={wy} r={5} fill="var(--viz-surface)" stroke={c} strokeWidth={2} />
                <text x={X(t)} y={wy - 10} fontSize={10} textAnchor="middle" fill="var(--viz-ink)">
                  {st.op}
                </text>
              </g>
              <g opacity={shown(v) ? 1 : 0.2}>
                <line x1={X(t)} y1={wy + 5} x2={X(v)} y2={vy - 5} stroke={c} strokeWidth={1.6} />
                <circle cx={X(v)} cy={vy} r={5} fill={c} />
                <text x={X(v)} y={vy + 17 + 11 * Math.min(3, visLvl[i])} fontSize={10} textAnchor="middle" fill="var(--viz-ink)">
                  {st.label}
                </text>
              </g>
            </g>
          );
        })}

        {/* loads */}
        {LOADS.map((ld, i) => {
          const c = LOC_COLOR[ld.loc];
          const planned = schedule.r[i];
          const t = e.sat[i];
          if (readNotReached && i > 0) {
            return (
              <text key={ld.op} x={X(planned)} y={ry + 4} fontSize={10} textAnchor="middle" fill="var(--viz-ink-muted)" opacity={shown(planned) ? 1 : 0.2}>
                ·
              </text>
            );
          }
          if (i === 3 && e.folded) {
            return (
              <g key={ld.op} opacity={shown(planned) ? 1 : 0.2}>
                <rect x={X(planned) - 5} y={ry - 5} width={10} height={10} fill="none" stroke="var(--viz-ink-muted)" strokeDasharray="2 2" />
                <text x={X(planned)} y={ry + 36} fontSize={10} textAnchor="middle" fill="var(--viz-ink-2)">
                  folded into r1
                </text>
              </g>
            );
          }
          const val = i === 0 ? e.v0 : i === 1 ? e.x : i === 2 ? e.y : e.v1;
          const moved = Math.abs(t - planned) > 1e-9;
          return (
            <g key={ld.op} opacity={shown(t) ? 1 : 0.2}>
              {moved ? (
                <>
                  <rect x={X(planned) - 5} y={ry - 5} width={10} height={10} fill="none" stroke={c} strokeDasharray="2 2" />
                  <path d={`M ${X(planned)} ${ry + 8} Q ${(X(planned) + X(t)) / 2} ${ry + 34} ${X(t)} ${ry + 8}`} fill="none" stroke="var(--viz-ink-2)" strokeDasharray="3 2" />
                  <text x={(X(planned) + X(t)) / 2} y={ry + 36} fontSize={10} textAnchor="middle" fill="var(--viz-ink-2)">
                    r4 satisfied early
                  </text>
                </>
              ) : null}
              <rect x={X(t) - 5} y={ry - 5} width={10} height={10} fill={c} />
              <text x={X(t)} y={readLvl[i] % 2 === 0 ? ry - 9 - 11 * Math.floor(readLvl[i] / 2) : ry + 19 + 11 * Math.floor(readLvl[i] / 2)} fontSize={10} textAnchor="middle" fill="var(--viz-ink)">
                {ld.loc}→{val}
              </text>
            </g>
          );
        })}

        <text x={12} y={y0 + PANEL - 14} fontSize={10} fill="var(--viz-ink-muted)">
          time →
        </text>
        {!done ? <line x1={X(cursor)} x2={X(cursor)} y1={y0 + 30} y2={y0 + PANEL - 10} stroke="var(--viz-ink-2)" strokeDasharray="4 3" /> : null}
      </g>
    );
  };

  const removed = [!f.wmbBegin && 'smp_wmb() A', !f.wmbEnd && 'smp_wmb() B', !f.rmb && 'smp_rmb()', !f.realRecheck && 'the compiler barrier'].filter(Boolean) as string[];
  const narrate = () => {
    if (!done) {
      const ev: string[] = [];
      const near = (t: number) => Math.abs(t - cursor) < 0.006;
      STORES.forEach((st, i) => {
        if (near(schedule.w[i][0])) ev.push(`writer executes ${st.op} (${st.label})`);
        if (near(weak.vis[i])) ev.push(`${st.label} becomes visible on the weak machine`);
        if (near(tso.vis[i]) && Math.abs(tso.vis[i] - weak.vis[i]) > 0.006) ev.push(`${st.label} reaches memory on TSO`);
      });
      LOADS.forEach((ld, i) => {
        if (near(weak.sat[i]) && !(weak.outcome === 'odd' && i > 0)) ev.push(`weak reader's ${ld.op} loads ${ld.loc}`);
        if (near(tso.sat[i]) && Math.abs(tso.sat[i] - weak.sat[i]) > 0.006 && !(tso.outcome === 'odd' && i > 0)) ev.push(`TSO reader's ${ld.op} loads ${ld.loc}`);
      });
      return (
        <>
          <strong>t = {fmtNum(cursor, 1)}:</strong> {ev.join('; ') || 'nothing new'}.
        </>
      );
    }
    const hidden = weak.outcome === 'torn' && tso.outcome !== 'torn';
    return (
      <>
        <strong>
          Weak: {describe(weak)}. TSO: {describe(tso)}.
        </strong>{' '}
        {hidden
          ? `The same timing that tears the record on the weak machine is harmless on TSO, because ${
              evaluate(schedule, 'weak', { ...f, rmb: true }).outcome !== 'torn'
                ? 'TSO satisfies one core\'s loads in program order, so the re-check cannot run before the copy.'
                : 'a FIFO store buffer never lets a later store become visible before an earlier one.'
            }`
          : weak.outcome === 'torn' && tso.outcome === 'torn'
            ? 'Both machines accept a torn record: with the re-check folded away by the compiler, no memory model can save the reader.'
            : removed.length
              ? tally.weak.torn + tally.tso.torn > 0
                ? `Removed: ${removed.join(', ')}. This schedule does not expose it, but ${fmtNum(tally.weak.torn)} of the 2,000 random schedules tear the record on the weak machine and ${fmtNum(tally.tso.torn)} on TSO.`
                : `Removed: ${removed.join(', ')}. Neither this schedule nor the 2,000 random ones expose it at this propagation delay — raise the delay${f.rmb && f.realRecheck ? ': the hazard is timing skew between stores' : ''}.`
              : 'With every fence in place, the reader either sees a single version or retries — on both machines, in all 2,000 random schedules.'}
      </>
    );
  };

  const pct = (n: number, d: number) => `${fmtNum((100 * n) / d, 1)}%`;

  return (
    <VizPanel
      title="A seqlock read on two memory models"
      subtitle="One writer bumps seq to odd, rewrites a two-field record and bumps seq to even; one reader loads seq, copies the record and re-checks seq. Delete a fence and replay the same schedule on a weakly ordered machine and on x86-TSO."
      controls={
        <>
          <Choice
            label="Schedule"
            value={preset}
            onChange={(v) => {
              setPreset(v);
              setStep(999);
            }}
            options={[...PRESETS.map((p) => ({ value: p.value, label: p.label })), { value: 'random', label: 'Random schedule' }]}
          />
          {preset === 'random' ? <Slider label="Seed" min={1} max={60} value={seed} onChange={(v) => { setSeed(v); setStep(999); }} /> : null}
          <Slider label="Max propagation delay (random runs)" min={0} max={40} value={skew} onChange={(v) => { setSkew(v); setStep(999); }} />
          <Check label="smp_wmb() after seq = 1 (A)" checked={f.wmbBegin} onChange={set('wmbBegin')} />
          <Check label="smp_wmb() before seq = 2 (B)" checked={f.wmbEnd} onChange={set('wmbEnd')} />
          <Check label="smp_rmb() before re-check" checked={f.rmb} onChange={set('rmb')} />
          <Check label="Re-check is READ_ONCE (compiler barrier)" checked={f.realRecheck} onChange={set('realRecheck')} />
        </>
      }
      legend={
        <Legend
          items={[
            { label: 'seq', color: LOC_COLOR.seq },
            { label: 'record field a', color: LOC_COLOR.a },
            { label: 'record field b', color: LOC_COLOR.b },
            { label: 'fence (dashed bar)', color: 'var(--viz-ink-2)', shape: 'line' },
            { label: 'torn record accepted', color: 'var(--viz-critical)' },
            { label: 'retry', color: 'var(--viz-warning)' },
          ]}
        />
      }
      stats={
        <Stats
          items={[
            { label: 'Weak: this schedule', value: weak.outcome === 'torn' ? 'torn' : weak.outcome === 'consistent' ? 'consistent' : 'retry' },
            { label: 'TSO: this schedule', value: tso.outcome === 'torn' ? 'torn' : tso.outcome === 'consistent' ? 'consistent' : 'retry' },
            {
              label: 'Weak: torn / retries (2,000 runs)',
              value: `${fmtNum(tally.weak.torn)} / ${fmtNum(tally.weak.odd + tally.weak.recheck)}`,
              hint: `${fmtNum(tally.weak.odd)} saw odd seq at begin, ${fmtNum(tally.weak.recheck)} failed the re-check`,
            },
            {
              label: 'TSO: torn / retries (2,000 runs)',
              value: `${fmtNum(tally.tso.torn)} / ${fmtNum(tally.tso.odd + tally.tso.recheck)}`,
              hint: `${fmtNum(tally.tso.odd)} saw odd seq at begin, ${fmtNum(tally.tso.recheck)} failed the re-check`,
            },
          ]}
        />
      }
      note={<Note>{narrate()}</Note>}
      table={
        <table className="viz-table">
          <thead>
            <tr>
              <th>Op</th>
              <th>Program time</th>
              <th>Weak: visible / satisfied</th>
              <th>TSO: visible / satisfied</th>
            </tr>
          </thead>
          <tbody>
            {STORES.map((st, i) => (
              <tr key={st.op}>
                <td>
                  {st.op} {st.label}
                </td>
                <td>{fmtNum(schedule.w[i][0], 1)}</td>
                <td>{fmtNum(weak.vis[i], 1)}</td>
                <td>{fmtNum(tso.vis[i], 1)}</td>
              </tr>
            ))}
            {LOADS.map((ld, i) => (
              <tr key={ld.op}>
                <td>
                  {ld.op} load {ld.loc}
                </td>
                <td>{fmtNum(schedule.r[i], 1)}</td>
                <td>{Number.isFinite(weak.sat[i]) ? fmtNum(weak.sat[i], 1) : 'folded'}</td>
                <td>{Number.isFinite(tso.sat[i]) ? fmtNum(tso.sat[i], 1) : 'folded'}</td>
              </tr>
            ))}
            {(['weak', 'tso'] as Machine[]).map((m) => (
              <tr key={m}>
                <td>{m === 'weak' ? 'Weak' : 'TSO'}: 2,000 random schedules</td>
                <td colSpan={3}>
                  consistent {pct(tally[m].consistent, tally[m].runs)}, odd at begin {pct(tally[m].odd, tally[m].runs)}, failed re-check{' '}
                  {pct(tally[m].recheck, tally[m].runs)}, torn accepted {fmtNum(tally[m].torn)}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      }
    >
      <div className="viz-controls">
        <Button onClick={() => setStep(0)}>Replay from start</Button>
        <Button primary onClick={() => setStep((s) => Math.min(times.length - 1, Math.min(s, times.length - 1) + 1))} disabled={done}>
          Next event
        </Button>
        <Button onClick={() => setStep(999)} disabled={done}>
          Show all
        </Button>
      </div>
      <svg
        viewBox={`0 0 ${W} ${H}`}
        width={W}
        height={H}
        role="img"
        aria-label={`Seqlock read. Weak machine: ${describe(weak)}. TSO: ${describe(tso)}.`}
      >
        {panel('weak', weak, 0)}
        {panel('tso', tso, PANEL + 8)}
      </svg>
    </VizPanel>
  );
}
