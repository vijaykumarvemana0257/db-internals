import { useMemo, useState } from 'react';
import { VizPanel, Segmented, Slider, Button, Legend, Stats, Note } from './Viz';

type Policy = 'lru' | 'fifo';
type Step = {
  page: string;
  before: string[]; // frames before the access, in the policy's victim order (victim last)
  after: string[];
  hit: boolean;
  evicted: string | null;
};

const FRAMES = 3;
/** Chosen so LRU and FIFO diverge early: they evict different pages at access 5. */
const SEQUENCE = ['A', 'B', 'C', 'A', 'D', 'B', 'A', 'E', 'B', 'C'];

/**
 * Frames are kept in victim order: index 0 is the most protected, the last element is
 * the next victim. LRU moves a hit to the front; FIFO leaves a hit where it is.
 */
function simulate(policy: Policy): Step[] {
  let frames: string[] = [];
  return SEQUENCE.map((page) => {
    const before = [...frames];
    const hit = frames.includes(page);
    let evicted: string | null = null;
    if (hit) {
      if (policy === 'lru') frames = [page, ...frames.filter((p) => p !== page)];
    } else {
      if (frames.length === FRAMES) evicted = frames[frames.length - 1];
      frames = [page, ...frames.filter((p) => p !== evicted)];
    }
    return { page, before, after: [...frames], hit, evicted };
  });
}

type Answer = 'hit' | 'free' | `evict:${string}`;
const answerFor = (s: Step): Answer => (s.hit ? 'hit' : s.evicted ? `evict:${s.evicted}` : 'free');
const describe = (a: Answer) => (a === 'hit' ? 'a hit — no disk read' : a === 'free' ? 'a miss into a free frame' : `a miss that evicts ${a.slice(6)}`);

export default function PredictThenRunTour() {
  const [policy, setPolicy] = useState<Policy>('lru');
  const steps = useMemo(() => simulate(policy), [policy]);
  const [i, setI] = useState(0); // index of the access being predicted
  const [guess, setGuess] = useState<Answer | null>(null);
  const [revealed, setRevealed] = useState(false);
  const [score, setScore] = useState({ right: 0, asked: 0 });
  const [rateGuess, setRateGuess] = useState(50);
  const [rateRevealed, setRateRevealed] = useState(false);

  const done = i >= steps.length;
  const cur = done ? null : steps[i];
  const shown = cur ? (revealed ? cur.after : cur.before) : steps[steps.length - 1].after;
  const hits = steps.filter((s) => s.hit).length;
  const actualRate = Math.round((hits / steps.length) * 100);

  const reset = (p: Policy = policy) => {
    setPolicy(p);
    setI(0);
    setGuess(null);
    setRevealed(false);
    setScore({ right: 0, asked: 0 });
    setRateRevealed(false);
  };

  const options: { value: Answer; label: string }[] = cur
    ? [
        { value: 'hit', label: 'Hit' },
        ...(cur.before.length < FRAMES ? [{ value: 'free' as Answer, label: 'Miss → free frame' }] : []),
        ...cur.before.map((p) => ({ value: `evict:${p}` as Answer, label: `Miss → evict ${p}` })),
      ]
    : [];

  const reveal = () => {
    if (!cur || !guess) return;
    setRevealed(true);
    setScore((s) => ({ right: s.right + (guess === answerFor(cur) ? 1 : 0), asked: s.asked + 1 }));
  };
  const next = () => {
    setI((n) => n + 1);
    setGuess(null);
    setRevealed(false);
  };

  const frameColor = (p: string | undefined) => {
    // Plain categorical slots: --viz-clean/--viz-dirty mean committed/uncommitted site-wide,
    // and a page read from disk is neither.
    if (!p || !cur) return 'var(--viz-1)';
    if (revealed && p === cur.page) return cur.hit ? 'var(--viz-3)' : 'var(--viz-2)';
    return 'var(--viz-1)';
  };

  return (
    <VizPanel
      title="Predict, then run"
      subtitle={`A buffer pool with ${FRAMES} frames and a fixed sequence of page accesses. Before each access, predict what happens. Then compare. Switch the policy and do it again.`}
      controls={
        <>
          <Segmented
            label="Replacement policy"
            value={policy}
            onChange={(p) => reset(p)}
            options={[
              { value: 'lru', label: 'LRU' },
              { value: 'fifo', label: 'FIFO' },
            ]}
          />
          <Button onClick={() => reset()}>Start over</Button>
        </>
      }
      legend={
        <Legend
          items={[
            { label: 'Page in a frame', color: 'var(--viz-1)' },
            { label: 'Just read from disk (miss)', color: 'var(--viz-2)' },
            { label: 'Just served from memory (hit)', color: 'var(--viz-3)' },
          ]}
        />
      }
      stats={
        <Stats
          items={[
            { label: 'Access', value: done ? `${steps.length} / ${steps.length}` : `${i + 1} / ${steps.length}` },
            { label: 'Your predictions', value: `${score.right} of ${score.asked} right` },
            { label: 'Hits so far', value: `${steps.slice(0, revealed ? i + 1 : i).filter((s) => s.hit).length}` },
            { label: 'Policy', value: policy.toUpperCase() },
          ]}
        />
      }
      note={
        <Note>
          {done ? (
            rateRevealed ? (
              <>
                <strong>
                  {policy.toUpperCase()} served {hits} of {steps.length} accesses from memory ({actualRate}%).
                </strong>{' '}
                You guessed {rateGuess}%. Now switch the policy: the sequence, the pool size and the pages are identical, and only the eviction rule changed.
              </>
            ) : (
              <>
                <strong>Last prediction:</strong> what fraction of the {steps.length} accesses were hits? Set the slider, then reveal.
              </>
            )
          ) : revealed && cur ? (
            <>
              <strong>{guess === answerFor(cur) ? 'Right.' : 'Not quite.'}</strong> Access to {cur.page} was {describe(answerFor(cur))}.
              {!cur.hit && cur.evicted
                ? policy === 'lru'
                  ? ` LRU evicts the page used least recently — ${cur.evicted} had gone longest without an access.`
                  : ` FIFO evicts the page loaded earliest — ${cur.evicted}, regardless of how recently it was used.`
                : ''}
              {cur.hit && policy === 'fifo' ? ' Note that under FIFO a hit does not protect the page: its place in line is unchanged.' : ''}
            </>
          ) : cur ? (
            <>
              <strong>Next access: page {cur.page}.</strong> Frames are shown in eviction order — the rightmost frame is the next victim. Pick your prediction.
            </>
          ) : null}
        </Note>
      }
      table={
        <table className="viz-table">
          <thead>
            <tr>
              <th>#</th>
              <th>Access</th>
              <th>Frames before</th>
              <th>Outcome</th>
              <th>Frames after</th>
            </tr>
          </thead>
          <tbody>
            {steps.map((s, n) => (
              <tr key={n}>
                <td>{n + 1}</td>
                <td>{s.page}</td>
                <td>{s.before.join(' ') || '—'}</td>
                <td>{describe(answerFor(s))}</td>
                <td>{s.after.join(' ')}</td>
              </tr>
            ))}
          </tbody>
        </table>
      }
    >
      <div style={{ minWidth: 320 }}>
        <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap', marginBottom: 12 }} aria-label="Access sequence">
          {SEQUENCE.map((p, n) => (
            <span
              key={n}
              style={{
                width: 28,
                textAlign: 'center',
                padding: '2px 0',
                borderRadius: 5,
                fontSize: '0.8rem',
                fontWeight: n === i ? 700 : 400,
                color: n < i ? 'var(--viz-ink-muted)' : 'var(--viz-ink)',
                border: `1px solid ${n === i ? 'var(--viz-ink)' : 'var(--viz-border)'}`,
                background: 'var(--viz-plane)',
              }}
            >
              {p}
            </span>
          ))}
        </div>

        <svg width={FRAMES * 92 + 40} height={96} role="img" aria-label={`Buffer pool frames: ${shown.join(', ') || 'empty'}`}>
          {Array.from({ length: FRAMES }, (_, f) => {
            const p = shown[f];
            const x = 10 + f * 92;
            return (
              <g key={f}>
                <rect x={x} y={18} width={80} height={56} rx={8} fill="var(--viz-plane)" stroke="var(--viz-border)" />
                {p ? <rect x={x + 6} y={24} width={68} height={44} rx={6} fill="var(--viz-surface)" stroke={frameColor(p)} strokeWidth={4} /> : null}
                <text x={x + 40} y={52} textAnchor="middle" fontSize={18} fontWeight={700} fill={p ? 'var(--viz-ink)' : 'var(--viz-ink-muted)'}>
                  {p ?? 'free'}
                </text>
                <text x={x + 40} y={12} textAnchor="middle" fontSize={10}>
                  {f === 0 ? 'safest' : f === FRAMES - 1 ? 'next victim' : ''}
                </text>
                <text x={x + 40} y={90} textAnchor="middle" fontSize={10}>
                  frame {f + 1}
                </text>
              </g>
            );
          })}
        </svg>

        {!done ? (
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6, marginTop: 10 }}>
            {options.map((o) => (
              <button
                key={o.value}
                type="button"
                aria-pressed={guess === o.value}
                disabled={revealed}
                onClick={() => setGuess(o.value)}
                data-variant={guess === o.value ? 'primary' : undefined}
              >
                {o.label}
              </button>
            ))}
            <Button primary onClick={revealed ? next : reveal} disabled={!revealed && !guess}>
              {revealed ? (i === steps.length - 1 ? 'Finish' : 'Next access') : 'Reveal'}
            </Button>
          </div>
        ) : (
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: 12, alignItems: 'end', marginTop: 10 }}>
            <Slider label="Your hit-rate prediction" min={0} max={100} step={10} value={rateGuess} onChange={setRateGuess} format={(n) => `${n}%`} disabled={rateRevealed} />
            <Button primary onClick={() => setRateRevealed(true)} disabled={rateRevealed}>
              Reveal hit rate
            </Button>
          </div>
        )}
      </div>
    </VizPanel>
  );
}
