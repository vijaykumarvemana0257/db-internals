import { useState } from 'react';
import { VizPanel, Button, Legend, Stats, Note } from './Viz';

/**
 * Two threads insert into the same gap of a lock-free skip list at once.
 * Each links its node bottom-up with one compare-and-swap per level; a CAS that loses
 * recomputes that level's predecessor and retries — the scheme RocksDB's
 * InlineSkipList::InsertConcurrently uses. Keys 25 (height 2) and 27 (height 1) both
 * start between 20 and 30.
 */
type Ptrs = Record<string, Record<number, string>>; // node -> level -> successor
type Step = {
  actor: 'A' | 'B' | '—';
  title: string;
  body: string;
  ptrs: Ptrs; // committed pointers after this step
  pending?: { from: string; level: number; to: string; ok: boolean; expect: string };
  linked: { A: number; B: number }; // levels each new node is linked at
  cas: number;
  fails: number;
};

const BASE: Ptrs = {
  head: { 1: '10', 2: '10' },
  '10': { 1: '20', 2: '20' },
  '20': { 1: '30', 2: '30' },
  '30': { 1: 'nil', 2: 'nil' },
};
const clone = (p: Ptrs): Ptrs => JSON.parse(JSON.stringify(p));

function script(): Step[] {
  const steps: Step[] = [];
  let p = clone(BASE);
  const push = (s: Omit<Step, 'ptrs'>) => steps.push({ ...s, ptrs: clone(p) });

  push({ actor: '—', title: 'Start', body: 'List 10 → 20 → 30 on both levels. Thread A will insert 25 with height 2; thread B will insert 27 with height 1. Both keys belong between 20 and 30.', linked: { A: 0, B: 0 }, cas: 0, fails: 0 });
  push({ actor: 'A', title: 'A searches', body: "A walks the list and records its splice: on level 1 and level 2, predecessor 20, successor 30. It takes no lock; it only remembers what it saw.", linked: { A: 0, B: 0 }, cas: 0, fails: 0 });
  push({ actor: 'B', title: 'B searches', body: 'B, running at the same moment, sees the same list: on level 1, predecessor 20, successor 30.', linked: { A: 0, B: 0 }, cas: 0, fails: 0 });
  push({ actor: 'A', title: 'A links level 1', body: "A sets its new node's level-1 pointer to 30, then CAS(20.next[1], expected 30, new 25). The pointer still holds 30, so the swap succeeds. 25 is now visible to every reader walking level 1.", pending: { from: '20', level: 1, to: '25', ok: true, expect: '30' }, linked: { A: 1, B: 0 }, cas: 1, fails: 0 });
  p['20'][1] = '25';
  p['25'] = { 1: '30' };
  steps[steps.length - 1].ptrs = clone(p);
  push({ actor: 'B', title: 'B links level 1 — and loses', body: "B sets its node's level-1 pointer to 30 and tries CAS(20.next[1], expected 30, new 27). But A already changed that pointer to 25. The compare fails, nothing is written, and B's node is still unlinked. No reader ever saw a half-inserted list.", pending: { from: '20', level: 1, to: '27', ok: false, expect: '30' }, linked: { A: 1, B: 0 }, cas: 2, fails: 1 });
  push({ actor: 'B', title: 'B retries from its predecessor', body: 'B does not restart from the head. It resumes the search at level 1 from 20, now finds 25 < 27, moves to 25, and records the new splice: predecessor 25, successor 30.', linked: { A: 1, B: 0 }, cas: 2, fails: 1 });
  push({ actor: 'B', title: 'B links level 1 again', body: 'B points 27 at 30 and tries CAS(25.next[1], expected 30, new 27). Nobody changed it, so it succeeds. Level 1 is now 20 → 25 → 27 → 30.', pending: { from: '25', level: 1, to: '27', ok: true, expect: '30' }, linked: { A: 1, B: 1 }, cas: 3, fails: 1 });
  p['25'][1] = '27';
  p['27'] = { 1: '30' };
  steps[steps.length - 1].ptrs = clone(p);
  push({ actor: 'A', title: 'A links level 2', body: "A finishes its tower: its level-2 pointer is set to 30 and CAS(20.next[2], expected 30, new 25) succeeds — B's height-1 node never touched level 2, so there was no conflict there.", pending: { from: '20', level: 2, to: '25', ok: true, expect: '30' }, linked: { A: 2, B: 1 }, cas: 4, fails: 1 });
  p['20'][2] = '25';
  p['25'][2] = '30';
  steps[steps.length - 1].ptrs = clone(p);
  push({ actor: '—', title: 'Done', body: 'Both inserts committed with four compare-and-swaps, one of which failed and was retried. No lock was held at any point, and every intermediate state was a valid skip list: a node linked on level 1 but not yet level 2 is simply a shorter tower for a moment.', linked: { A: 2, B: 1 }, cas: 4, fails: 1 });
  return steps;
}

const STEPS = script();
const ORDER = ['head', '10', '20', '25', '27', '30'];
const NEW_NODE: Record<string, 'A' | 'B'> = { '25': 'A', '27': 'B' };
const HEIGHT: Record<string, number> = { head: 2, '10': 2, '20': 2, '25': 2, '27': 1, '30': 2 };

export default function SkipListCasRaceLab() {
  const [i, setI] = useState(0);
  const s = STEPS[i];

  const colW = 92;
  const xOf = (n: string) => 30 + ORDER.indexOf(n) * colW;
  const yOf = (level: number) => 150 - level * 46;

  // A node is drawn linked once its level-1 CAS succeeded; a failed CAS leaves it unlinked.
  const present = (n: string) => !NEW_NODE[n] || s.linked[NEW_NODE[n]] > 0;

  return (
    <VizPanel
      title="Two inserts, no lock: per-level compare-and-swap"
      subtitle="Step through two threads inserting into the same gap. Each link is a single atomic compare-and-swap on one forward pointer. When the pointer changed underneath a thread, its swap fails and it retries from where it stood."
      controls={
        <>
          <Button onClick={() => setI((n) => Math.max(0, n - 1))} disabled={i === 0}>
            Back
          </Button>
          <Button primary onClick={() => setI((n) => Math.min(STEPS.length - 1, n + 1))} disabled={i === STEPS.length - 1}>
            Next step
          </Button>
          <Button onClick={() => setI(0)}>Reset</Button>
        </>
      }
      legend={
        <Legend
          items={[
            { label: 'Committed forward pointer', color: 'var(--viz-ink-2)', shape: 'line' },
            { label: 'Thread A (inserting 25)', color: 'var(--viz-1)' },
            { label: 'Thread B (inserting 27)', color: 'var(--viz-2)' },
            { label: '✓ CAS succeeded', color: 'var(--viz-good)' },
            { label: '✕ CAS failed — retry', color: 'var(--viz-critical)' },
          ]}
        />
      }
      stats={
        <Stats
          items={[
            { label: 'Step', value: `${i + 1} / ${STEPS.length}` },
            { label: 'Compare-and-swaps', value: s.cas },
            { label: 'Failed and retried', value: s.fails },
            { label: 'Locks taken', value: 0 },
          ]}
        />
      }
      note={
        <Note>
          <strong>{s.title}.</strong> {s.body}
        </Note>
      }
      table={
        <table className="viz-table">
          <thead>
            <tr>
              <th>#</th>
              <th>Thread</th>
              <th>Action</th>
              <th>CAS</th>
            </tr>
          </thead>
          <tbody>
            {STEPS.map((x, n) => (
              <tr key={n}>
                <td>{n + 1}</td>
                <td>{x.actor}</td>
                <td>{x.title}</td>
                <td>{x.pending ? `${x.pending.from}.next[${x.pending.level}]: ${x.pending.expect} → ${x.pending.to} ${x.pending.ok ? 'succeeded' : 'failed'}` : '—'}</td>
              </tr>
            ))}
          </tbody>
        </table>
      }
    >
      <svg width={30 + ORDER.length * colW} height={190} role="img" aria-label={`Skip list state: ${s.title}`}>
        {[1, 2].map((level) => (
          <text key={level} x={0} y={yOf(level) + 4} fontSize={10}>
            L{level}
          </text>
        ))}

        {/* committed pointers */}
        {ORDER.map((n) =>
          [1, 2].map((level) => {
            const to = s.ptrs[n]?.[level];
            if (!to || to === 'nil') return null;
            return (
              <line key={`${n}-${level}`} x1={xOf(n) + 14} x2={xOf(to) - 16} y1={yOf(level)} y2={yOf(level)} stroke="var(--viz-ink-2)" strokeWidth={1.5} markerEnd="url(#arrow-cas)" />
            );
          }),
        )}

        {/* the CAS being attempted this step */}
        {s.pending ? (
          <g>
            <path
              d={`M ${xOf(s.pending.from) + 14} ${yOf(s.pending.level) - 8} Q ${(xOf(s.pending.from) + xOf(s.pending.to)) / 2} ${yOf(s.pending.level) - 34} ${xOf(s.pending.to) - 6} ${yOf(s.pending.level) - 14}`}
              fill="none"
              stroke={s.pending.ok ? 'var(--viz-good)' : 'var(--viz-critical)'}
              strokeWidth={2.5}
              strokeDasharray={s.pending.ok ? undefined : '5 3'}
            />
            <text x={(xOf(s.pending.from) + xOf(s.pending.to)) / 2} y={yOf(s.pending.level) - 36} textAnchor="middle" fontSize={11} fill="var(--viz-ink)">
              {s.pending.ok ? '✓ CAS' : '✕ CAS failed'}
            </text>
          </g>
        ) : null}

        {/* towers */}
        {ORDER.map((n) => {
          const owner = NEW_NODE[n];
          const linkedLevels = owner ? s.linked[owner] : HEIGHT[n];
          if (owner && !present(n)) {
            return (
              <g key={n} opacity={0.5}>
                <rect x={xOf(n) - 14} y={yOf(HEIGHT[n]) - 14} width={28} height={HEIGHT[n] * 46 - 18} rx={5} fill="none" stroke={owner === 'A' ? 'var(--viz-1)' : 'var(--viz-2)'} strokeDasharray="4 3" />
                <text x={xOf(n)} y={172} textAnchor="middle" fontSize={11}>
                  {n} (unlinked)
                </text>
              </g>
            );
          }
          return (
            <g key={n}>
              {Array.from({ length: HEIGHT[n] }, (_, k) => {
                const level = k + 1;
                const isLinked = !owner || level <= linkedLevels;
                return (
                  <rect
                    key={level}
                    x={xOf(n) - 14}
                    y={yOf(level) - 14}
                    width={28}
                    height={28}
                    rx={5}
                    fill={owner ? (owner === 'A' ? 'var(--viz-1)' : 'var(--viz-2)') : 'var(--viz-plane)'}
                    fillOpacity={isLinked ? 1 : 0.25}
                    stroke="var(--viz-border)"
                  />
                );
              })}
              <text x={xOf(n)} y={172} textAnchor="middle" fontSize={11} fill="var(--viz-ink)">
                {n === 'head' ? 'head' : n}
              </text>
            </g>
          );
        })}
        <defs>
          <marker id="arrow-cas" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
            <path d="M 0 0 L 10 5 L 0 10 z" fill="var(--viz-ink-2)" />
          </marker>
        </defs>
      </svg>
    </VizPanel>
  );
}
