import { useMemo, useState } from 'react';
import { VizPanel, Segmented, Slider, Button, Legend, Stats, Note, fmtNum, makeRng } from './Viz';

type P = '0.5' | '0.25';
const MAX_HEIGHT = 12; // RocksDB's InlineSkipList and LevelDB both default to 12

/** Height of a new node: keep flipping a p-weighted coin. Deterministic per (seed, index). */
function heightsFor(n: number, p: number, seed: number) {
  const rng = makeRng(seed);
  return Array.from({ length: n }, () => {
    let h = 1;
    while (h < MAX_HEIGHT && rng() < p) h++;
    return h;
  });
}

type Node = { key: number; h: number };

/** Standard skip-list search: go right while the next key is < target, else drop a level. */
function search(nodes: Node[], target: number) {
  const top = Math.max(1, ...nodes.map((x) => x.h));
  const path: { level: number; at: number; cmp: number | null; move: 'right' | 'down' }[] = [];
  let pos = -1; // -1 = head
  let comparisons = 0;
  for (let level = top; level >= 1; level--) {
    for (;;) {
      let next = pos + 1;
      while (next < nodes.length && nodes[next].h < level) next++;
      if (next >= nodes.length) {
        path.push({ level, at: pos, cmp: null, move: 'down' });
        break;
      }
      comparisons++;
      if (nodes[next].key < target) {
        path.push({ level, at: next, cmp: nodes[next].key, move: 'right' });
        pos = next;
      } else {
        path.push({ level, at: pos, cmp: nodes[next].key, move: 'down' });
        break;
      }
    }
  }
  const found = pos + 1 < nodes.length && nodes[pos + 1].key === target;
  return { path, comparisons, found, landed: pos + 1 };
}

/** Pugh's expected search cost: about log_{1/p}(n) / p comparisons. */
const expectedCmp = (n: number, p: number) => Math.log(n) / Math.log(1 / p) / p;
const expectedPointers = (p: number) => 1 / (1 - p);

export default function SkipListStaircaseLab() {
  const [pStr, setPStr] = useState<P>('0.25');
  const [count, setCount] = useState(24);
  const [seed, setSeed] = useState(7);
  const [target, setTarget] = useState(0); // index into keys; 0 means "pick default"
  const [statN, setStatN] = useState(4);

  const p = Number(pStr);
  const nodes: Node[] = useMemo(() => {
    const hs = heightsFor(count, p, seed * 97 + 13);
    return hs.map((h, i) => ({ key: (i + 1) * 10, h }));
  }, [count, p, seed]);

  const targetKey = target || nodes[Math.floor(nodes.length * 0.7)]?.key || 10;
  const run = useMemo(() => search(nodes, targetKey), [nodes, targetKey]);

  const top = Math.max(1, ...nodes.map((x) => x.h));
  const pointers = nodes.reduce((a, x) => a + x.h, 0);

  // Empirical average over many lists and searches, against the analytic curve.
  // Uses real per-level forward pointers so each search costs what a skip-list search costs.
  const empirical = useMemo(() => {
    const n = 10 ** statN;
    const trials = 20;
    let total = 0;
    let heights = 0;
    for (let tr = 0; tr < trials; tr++) {
      const hs = heightsFor(n, p, 1000 + tr);
      const topH = Math.max(...hs);
      // next[l][i] = index of the first node after i with height > l (or n); slot n is the head.
      const next: Int32Array[] = [];
      for (let l = 0; l < topH; l++) {
        const arr = new Int32Array(n + 1);
        let nxt = n;
        for (let i = n - 1; i >= 0; i--) {
          arr[i] = nxt;
          if (hs[i] > l) nxt = i;
        }
        arr[n] = nxt; // head's successor on this level
        next.push(arr);
      }
      const probe = (tr * 7919) % n; // keys are 0..n-1
      let pos = n; // head
      let cmp = 0;
      for (let l = topH - 1; l >= 0; l--) {
        for (;;) {
          const nx = next[l][pos];
          if (nx >= n) break;
          cmp++;
          if (nx < probe) pos = nx;
          else break;
        }
      }
      total += cmp;
      heights += topH;
    }
    return { cmp: total / trials, height: heights / trials, n, trials };
  }, [statN, p]);

  const colW = 30;
  const laneH = 22;
  const W = 60 + nodes.length * colW + 30;
  const H = 30 + Math.min(top, MAX_HEIGHT) * laneH + 40;
  const onPath = new Set(run.path.filter((s) => s.move === 'right').map((s) => `${s.level}:${s.at}`));
  const dropAt = run.path.filter((s) => s.move === 'down').map((s) => ({ level: s.level, at: s.at }));
  const xOf = (i: number) => (i < 0 ? 22 : 60 + i * colW + colW / 2);
  const yOf = (level: number) => 30 + (top - level) * laneH + laneH / 2;

  return (
    <VizPanel
      title="A skip list, and the staircase a search walks"
      subtitle="Each node's height comes from repeated coin flips with probability p. A search runs right along the highest lane while the next key is smaller, then drops a level. No node is ever moved to rebalance anything."
      controls={
        <>
          <Segmented label="p (probability of growing a level)" value={pStr} onChange={setPStr} options={[{ value: '0.5', label: '½ (classic)' }, { value: '0.25', label: '¼ (RocksDB, LevelDB)' }]} />
          <Slider label="Keys" min={8} max={40} value={count} onChange={(n) => { setCount(n); setTarget(0); }} />
          <Slider label="Search for" min={1} max={count} value={Math.max(1, Math.round(targetKey / 10))} onChange={(i) => setTarget(i * 10)} format={(i) => `key ${i * 10}`} />
          <Button onClick={() => setSeed((s) => s + 1)} title="Re-flip every node's coins">Re-flip coins</Button>
        </>
      }
      legend={
        <Legend
          items={[
            { label: 'Node tower (one pointer per level)', color: 'var(--viz-1)' },
            { label: 'Step right along a lane', color: 'var(--viz-2)', shape: 'line' },
            { label: 'Drop down a level', color: 'var(--viz-ink)', shape: 'line' },
            { label: 'Target key', color: 'var(--viz-good)' },
          ]}
        />
      }
      stats={
        <Stats
          items={[
            { label: 'Comparisons for this search', value: fmtNum(run.comparisons) },
            { label: `Expected at n=${count}, p=${pStr}`, value: fmtNum(expectedCmp(count, p), 1), hint: 'log₁/ₚ(n) / p' },
            { label: 'Pointers per node', value: `${fmtNum(pointers / nodes.length, 2)} (expected ${fmtNum(expectedPointers(p), 2)})` },
            { label: 'Tallest tower', value: `${top} of max ${MAX_HEIGHT}` },
          ]}
        />
      }
      note={
        <Note>
          <strong>
            Search for {targetKey}: {run.comparisons} comparisons, {run.found ? 'found' : 'not present'}.
          </strong>{' '}
          It started on lane {top} at the head and stepped right {run.path.filter((s) => s.move === 'right').length} times and down {dropAt.length} times.
          Press <em>Re-flip coins</em> to rebuild the same keys with new heights: the comparison count changes run to run, because balance is only probable, not guaranteed.
        </Note>
      }
      table={
        <table className="viz-table">
          <thead>
            <tr>
              <th>n</th>
              <th>p</th>
              <th>Expected comparisons</th>
              <th>Measured (avg of {empirical.trials} lists)</th>
              <th>Avg tallest tower</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td>{fmtNum(empirical.n)}</td>
              <td>{pStr}</td>
              <td>{fmtNum(expectedCmp(empirical.n, p), 1)}</td>
              <td>{fmtNum(empirical.cmp, 1)}</td>
              <td>{fmtNum(empirical.height, 1)}</td>
            </tr>
          </tbody>
        </table>
      }
    >
      <svg width={W} height={H} role="img" aria-label={`Skip list with ${count} keys and search path to ${targetKey}`}>
        {Array.from({ length: top }, (_, k) => {
          const level = top - k;
          return (
            <text key={level} x={0} y={yOf(level) + 4} fontSize={10}>
              L{level}
            </text>
          );
        })}
        <rect x={14} y={24} width={16} height={top * laneH + 12} rx={3} fill="var(--viz-plane)" stroke="var(--viz-border)" />
        <text x={22} y={H - 8} textAnchor="middle" fontSize={9}>head</text>

        {/* lane pointers */}
        {Array.from({ length: top }, (_, k) => {
          const level = top - k;
          const members = [-1, ...nodes.map((x, i) => (x.h >= level ? i : null)).filter((i): i is number => i !== null)];
          return members.slice(0, -1).map((from, j) => {
            const to = members[j + 1];
            const hot = onPath.has(`${level}:${to}`);
            return <line key={`${level}-${from}`} x1={xOf(from)} x2={xOf(to)} y1={yOf(level)} y2={yOf(level)} stroke={hot ? 'var(--viz-2)' : 'var(--viz-grid)'} strokeWidth={hot ? 3 : 1} />;
          });
        })}

        {/* drops */}
        {dropAt.map((d, i) => (d.level > 1 ? <line key={i} x1={xOf(d.at)} x2={xOf(d.at)} y1={yOf(d.level)} y2={yOf(d.level - 1)} stroke="var(--viz-ink)" strokeWidth={2} /> : null))}

        {/* towers */}
        {nodes.map((x, i) => (
          <g key={x.key}>
            {Array.from({ length: x.h }, (_, k) => (
              <rect key={k} x={xOf(i) - 6} y={yOf(k + 1) - 6} width={12} height={12} rx={2} fill={x.key === targetKey ? 'var(--viz-good)' : 'var(--viz-1)'} stroke="var(--viz-surface)" strokeWidth={1.5} />
            ))}
            <text x={xOf(i)} y={H - 8} textAnchor="middle" fontSize={9} fill={x.key === targetKey ? 'var(--viz-ink)' : undefined}>
              {x.key}
            </text>
          </g>
        ))}
      </svg>

      <div style={{ marginTop: 10 }}>
        <Slider label="Measure at n =" min={2} max={5} value={statN} onChange={setStatN} format={(e) => fmtNum(10 ** e)} />
        <p style={{ margin: '0.35rem 0 0', fontSize: '0.78rem', color: 'var(--viz-ink-2)' }}>
          At n = {fmtNum(empirical.n)} and p = {pStr}: expected ≈ {fmtNum(expectedCmp(empirical.n, p), 1)} comparisons, measured {fmtNum(empirical.cmp, 1)} averaged over {empirical.trials} independently built lists; the tallest tower averaged {fmtNum(empirical.height, 1)} levels (log₁/ₚ n = {fmtNum(Math.log(empirical.n) / Math.log(1 / p), 1)}).
        </p>
      </div>
    </VizPanel>
  );
}
