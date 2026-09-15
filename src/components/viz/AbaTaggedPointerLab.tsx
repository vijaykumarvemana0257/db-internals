import { useMemo, useState } from 'react';
import { VizPanel, Segmented, Check, Button, Legend, Stats, Note } from './Viz';

/**
 * The ABA problem on the classic lock-free stack, replayed step by step.
 *
 * pop():  loop { t = top; n = t->next; if CAS(&top, t, n) return t->value; }
 * push(): loop { node->next = top; if CAS(&top, node->next, node) return; }
 *
 * T1 loads top = A and A->next = B, then stalls before its CAS. T2 pops A, pops B, and pushes 40 into a node the
 * allocator hands back at A's address. T1's CAS then compares top against A:
 *  - plain pointer: the address matches, the CAS succeeds and installs B — a node already on the free list;
 *  - tagged pointer: the top-of-stack word carries a modification counter incremented by every successful CAS
 *    (Michael & Scott 1996's <ptr, count>, which needs a double-width CAS or an index packed with the counter);
 *    the counter no longer matches, the CAS fails and T1 retries against the real stack.
 * With "allocator reuses A" off, T2's push gets a fresh node D and even the plain CAS fails.
 * The replay is deterministic: no timing, only this interleaving.
 */

export type PtrMode = 'plain' | 'tagged';
type Cell = { value: number; next: string | null; freed: boolean };
export type AbaState = {
  top: string | null;
  tag: number;
  nodes: Record<string, Cell>;
  free: string[];
  t1: { ptr: string; tag: number; next: string | null } | null;
  popped: number[];
  cas: { who: 'T1' | 'T2'; expected: string; actual: string; ok: boolean } | null;
  text: string;
};

const clone = (s: AbaState): AbaState => ({
  ...s,
  nodes: Object.fromEntries(Object.entries(s.nodes).map(([k, v]) => [k, { ...v }])),
  free: s.free.slice(),
  popped: s.popped.slice(),
  t1: s.t1 ? { ...s.t1 } : null,
  cas: null,
});

const word = (ptr: string | null, tag: number, mode: PtrMode) => (mode === 'tagged' ? `<${ptr ?? 'null'}, ${tag}>` : `${ptr ?? 'null'}`);

export function chain(s: AbaState) {
  const out: string[] = [];
  let p = s.top;
  const seen = new Set<string>();
  while (p && !seen.has(p) && out.length < 8) {
    seen.add(p);
    out.push(p);
    p = s.nodes[p].next;
  }
  return out;
}

export function abaReplay(mode: PtrMode, reuse: boolean): AbaState[] {
  const s0: AbaState = {
    top: 'A',
    tag: 0,
    nodes: { A: { value: 10, next: 'B', freed: false }, B: { value: 20, next: 'C', freed: false }, C: { value: 30, next: null, freed: false } },
    free: [],
    t1: null,
    popped: [],
    cas: null,
    text: `Stack is A(10) → B(20) → C(30); top = ${word('A', 0, mode)}.`,
  };
  const states = [s0];
  let s = clone(s0);

  // 1. T1 loads top and next, then stalls.
  s.t1 = { ptr: 'A', tag: s.tag, next: 'B' };
  s.text = `T1 starts pop(): loads top = ${word('A', s.tag, mode)} and A.next = B, then is descheduled before its CAS.`;
  states.push(s);

  // 2. T2 pops A.
  s = clone(s);
  s.cas = { who: 'T2', expected: word('A', s.tag, mode), actual: word('A', s.tag, mode), ok: true };
  s.top = 'B';
  s.tag += 1;
  s.popped.push(10);
  s.nodes.A.freed = true;
  s.free.push('A');
  s.text = `T2 pop(): CAS(top, ${s.cas.expected} → ${word('B', s.tag, mode)}) succeeds and returns 10. Node A goes to the free list.`;
  states.push(s);

  // 3. T2 pops B.
  s = clone(s);
  s.cas = { who: 'T2', expected: word('B', s.tag, mode), actual: word('B', s.tag, mode), ok: true };
  s.top = 'C';
  s.tag += 1;
  s.popped.push(20);
  s.nodes.B.freed = true;
  s.free.push('B');
  s.text = `T2 pop(): CAS(top, ${s.cas.expected} → ${word('C', s.tag, mode)}) succeeds and returns 20. Node B goes to the free list.`;
  states.push(s);

  // 4. T2 pushes 40, into A's recycled address or a fresh node.
  s = clone(s);
  const id = reuse ? 'A' : 'D';
  if (reuse) {
    s.free = s.free.filter((x) => x !== 'A');
    s.nodes.A = { value: 40, next: 'C', freed: false };
  } else {
    s.nodes.D = { value: 40, next: 'C', freed: false };
  }
  s.cas = { who: 'T2', expected: word('C', s.tag, mode), actual: word('C', s.tag, mode), ok: true };
  s.top = id;
  s.tag += 1;
  s.text = reuse
    ? `T2 push(40): the allocator hands back A's address. A.value = 40, A.next = C, CAS(top, ${s.cas.expected} → ${word('A', s.tag, mode)}) succeeds. top is A again.`
    : `T2 push(40) gets a fresh node D: D.next = C, CAS(top, ${s.cas.expected} → ${word('D', s.tag, mode)}) succeeds.`;
  states.push(s);

  // 5. T1 resumes with its stale snapshot.
  s = clone(s);
  const t1 = s.t1!;
  const expected = word(t1.ptr, t1.tag, mode);
  const actual = word(s.top, s.tag, mode);
  const ok = mode === 'plain' ? s.top === t1.ptr : s.top === t1.ptr && s.tag === t1.tag;
  s.cas = { who: 'T1', expected, actual, ok };
  if (ok) {
    const val = s.nodes[t1.ptr].value;
    s.top = t1.next;
    s.tag += 1;
    s.popped.push(val);
    s.t1 = null;
    s.text = `T1 resumes: CAS(top, ${expected} → ${word(t1.next, s.tag, mode)}) compares only the address, finds A, and succeeds. top now points at B — a node that was popped and freed — and T1 returns 40, the value T2 pushed.`;
  } else {
    s.text =
      mode === 'tagged' && s.top === t1.ptr
        ? `T1 resumes: CAS(top, ${expected} → …) fails — the address is A again, but the counter is ${s.tag}, not ${t1.tag}. Three successful CASes happened in between.`
        : `T1 resumes: CAS(top, ${expected} → …) fails because top is ${actual}: D is a different address, so the CAS sees the change.`;
  }
  states.push(s);

  // 6. Aftermath.
  s = clone(s);
  if (ok) {
    const b = s.top ? s.nodes[s.top] : null;
    s.text = `Damage: the stack is ${chain(s).join(' → ')}. Its top node is on the free list, so the next pop returns ${b?.value} — a value T2 already popped — and the allocator may hand B's memory to someone else at any moment.`;
  } else {
    const top = s.top!;
    const next = s.nodes[top].next;
    s.cas = { who: 'T1', expected: word(top, s.tag, mode), actual: word(top, s.tag, mode), ok: true };
    const val = s.nodes[top].value;
    s.top = next;
    s.tag += 1;
    s.popped.push(val);
    s.t1 = null;
    s.text = `T1 retries: loads top = ${s.cas.expected} and ${top}.next = C, CAS succeeds, returns ${val}. The stack is C(30), with every value popped exactly once.`;
  }
  states.push(s);
  return states;
}

export function verdict(s: AbaState) {
  const reach = chain(s);
  const freedReachable = reach.filter((n) => s.nodes[n].freed);
  const values = reach.map((n) => s.nodes[n].value);
  const dup = values.filter((v) => s.popped.includes(v));
  return { freedReachable, dup, ok: freedReachable.length === 0 && dup.length === 0 };
}

export default function AbaTaggedPointerLab() {
  const [mode, setMode] = useState<PtrMode>('plain');
  const [reuse, setReuse] = useState(true);
  const [step, setStep] = useState(99);
  const states = useMemo(() => abaReplay(mode, reuse), [mode, reuse]);
  const i = Math.min(step, states.length - 1);
  const s = states[i];
  const done = i === states.length - 1;
  const v = verdict(s);
  const t1Cas = states[5].cas;

  const W = 680;
  const H = 250;
  const reach = chain(s);
  // the real free list: a freed node that is still reachable from top appears in both rows, which is the corruption
  const others = s.free;
  const box = (id: string, x: number, y: number, freedReach: boolean) => {
    const c = s.nodes[id];
    return (
      <g key={`${id}-${x}-${y}`}>
        <rect
          x={x}
          y={y}
          width={86}
          height={40}
          rx={5}
          fill="var(--viz-surface)"
          stroke={freedReach ? 'var(--viz-critical)' : c.freed ? 'var(--viz-stale)' : 'var(--viz-1)'}
          strokeWidth={freedReach ? 3 : 2}
          strokeDasharray={c.freed && !freedReach ? '4 3' : undefined}
        />
        <text x={x + 10} y={y + 25} fontSize={14} fontWeight={600} fill="var(--viz-ink)">
          {id}
        </text>
        <text x={x + 34} y={y + 25} fontSize={12} fill="var(--viz-ink)">
          {c.value}
        </text>
        <text x={x + 60} y={y + 25} fontSize={10} fill="var(--viz-ink-2)">
          →{c.next ?? '∅'}
        </text>
        {c.freed ? (
          <text x={x + 43} y={y + 54} fontSize={10} textAnchor="middle" fill="var(--viz-ink-2)">
            freed
          </text>
        ) : null}
      </g>
    );
  };

  return (
    <VizPanel
      title="ABA on a lock-free stack"
      subtitle="T1 reads the top of the stack and stalls. T2 pops two nodes and pushes one that reuses the first node's address. Step through T1's compare-and-swap with a plain pointer, then with a tagged one."
      controls={
        <>
          <Segmented
            label="Top-of-stack word"
            value={mode}
            onChange={(m) => {
              setMode(m);
              setStep(99);
            }}
            options={[
              { value: 'plain', label: 'Plain pointer' },
              { value: 'tagged', label: 'Tagged pointer <ptr, counter>' },
            ]}
          />
          <Check
            label="Allocator reuses A's address"
            checked={reuse}
            onChange={(b) => {
              setReuse(b);
              setStep(99);
            }}
          />
        </>
      }
      legend={
        <Legend
          items={[
            { label: 'Node on the stack', color: 'var(--viz-1)' },
            { label: 'Freed node', color: 'var(--viz-stale)' },
            { label: 'Corruption: freed node reachable, and the CAS that did it', color: 'var(--viz-critical)' },
            { label: "T1's stale snapshot", color: 'var(--viz-2)' },
            { label: 'CAS that swaps', color: 'var(--viz-good)' },
            { label: 'CAS that fails', color: 'var(--viz-warning)' },
          ]}
        />
      }
      stats={
        <Stats
          items={[
            { label: 'top', value: word(s.top, s.tag, mode) },
            { label: "T1's CAS", value: i < 5 ? 'not yet' : t1Cas?.ok ? 'succeeded' : 'failed → retry' },
            { label: 'Values popped', value: s.popped.length ? s.popped.join(', ') : '—' },
            { label: 'Stack', value: v.ok ? 'intact' : 'corrupt', hint: v.ok ? 'Every reachable node is live and no popped value is still on the stack.' : 'A freed node is reachable from top.' },
          ]}
        />
      }
      note={
        <Note>
          <strong>
            Step {i} of {states.length - 1}.
          </strong>{' '}
          {s.text}
        </Note>
      }
      table={
        <table className="viz-table">
          <thead>
            <tr>
              <th>Step</th>
              <th>CAS by</th>
              <th>Expected</th>
              <th>Found</th>
              <th>Result</th>
            </tr>
          </thead>
          <tbody>
            {states.map((st, k) =>
              st.cas ? (
                <tr key={k}>
                  <td>{k}</td>
                  <td>{st.cas.who}</td>
                  <td>{st.cas.expected}</td>
                  <td>{st.cas.actual}</td>
                  <td>{st.cas.ok ? 'swap' : 'fail'}</td>
                </tr>
              ) : null,
            )}
          </tbody>
        </table>
      }
    >
      <div className="viz-controls">
        <Button onClick={() => setStep(0)}>Replay from start</Button>
        <Button primary onClick={() => setStep((k) => Math.min(states.length - 1, Math.min(k, states.length - 1) + 1))} disabled={done}>
          Next step
        </Button>
        <Button onClick={() => setStep(99)} disabled={done}>
          Show all
        </Button>
      </div>
      <svg viewBox={`0 0 ${W} ${H}`} width={W} height={H} role="img" aria-label={`Lock-free stack after step ${i}: top ${word(s.top, s.tag, mode)}, stack ${v.ok ? 'intact' : 'corrupt'}`}>
        <text x={12} y={20} fontSize={11} fill="var(--viz-ink-2)">
          shared top-of-stack word
        </text>
        <rect x={12} y={30} width={130} height={40} rx={5} fill="var(--viz-surface)" stroke="var(--viz-ink-2)" strokeWidth={1.5} />
        <text x={77} y={55} fontSize={13} textAnchor="middle" fontWeight={600} fill="var(--viz-ink)">
          top = {word(s.top, s.tag, mode)}
        </text>
        {s.cas ? (
          <g>
            <rect x={160} y={30} width={300} height={40} rx={5} fill="var(--viz-surface)" stroke={s.cas.ok ? (s.cas.who === 'T1' && mode === 'plain' && reuse ? 'var(--viz-critical)' : 'var(--viz-good)') : 'var(--viz-warning)'} strokeWidth={2} />
            <text x={172} y={48} fontSize={11} fill="var(--viz-ink)">
              {s.cas.who}: CAS expects {s.cas.expected}
            </text>
            <text x={172} y={63} fontSize={11} fill="var(--viz-ink)">
              finds {s.cas.actual} → {s.cas.ok ? 'swap' : 'fail'}
            </text>
          </g>
        ) : null}

        {s.t1 ? (
          <g>
            <rect x={480} y={30} width={188} height={56} rx={5} fill="var(--viz-surface)" stroke="var(--viz-2)" strokeWidth={2} strokeDasharray="5 3" />
            <text x={492} y={48} fontSize={11} fontWeight={600} fill="var(--viz-ink)">
              {i >= 5 ? 'T1 (snapshot is stale)' : 'T1 (stalled before CAS)'}
            </text>
            <text x={492} y={64} fontSize={11} fill="var(--viz-ink)">
              expected top = {word(s.t1.ptr, s.t1.tag, mode)}
            </text>
            <text x={492} y={79} fontSize={11} fill="var(--viz-ink)">
              new top = {s.t1.next}
            </text>
          </g>
        ) : null}

        <text x={12} y={112} fontSize={11} fill="var(--viz-ink-2)">
          reachable from top
        </text>
        {reach.map((id, k) => {
          const x = 12 + k * 116;
          return (
            <g key={id}>
              {box(id, x, 122, s.nodes[id].freed)}
              {k < reach.length - 1 ? (
                <path d={`M ${x + 88} 142 L ${x + 112} 142`} stroke="var(--viz-ink-2)" strokeWidth={1.5} markerEnd="url(#stack-arrow)" />
              ) : null}
            </g>
          );
        })}
        {reach.length === 0 ? (
          <text x={12} y={146} fontSize={12} fill="var(--viz-ink-2)">
            (empty)
          </text>
        ) : null}

        <text x={12} y={205} fontSize={11} fill="var(--viz-ink-2)">
          free list
        </text>
        {others.length === 0 ? (
          <text x={80} y={205} fontSize={11} fill="var(--viz-ink-muted)">
            empty
          </text>
        ) : (
          others.map((id, k) => box(id, 80 + k * 100, 186, false))
        )}
        <defs>
          <marker id="stack-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
            <path d="M 0 0 L 10 5 L 0 10 z" fill="var(--viz-ink-2)" />
          </marker>
        </defs>
      </svg>
    </VizPanel>
  );
}
