import { useState } from 'react';
import {
  VizPanel,
  Check,
  Button,
  Legend,
  Stats,
  Note,
  TooltipHost,
  useTip,
  fmtTime,
  makeRng,
  useSize,
} from './Viz';

/**
 * What one eviction costs when the mapping lives in the pages themselves.
 *
 * A hash-table pool evicts by deleting one entry from one table. A swizzling pool has
 * put the frame address inside the *parent* page, so eviction is a four-part protocol:
 * sample a victim, refuse it if it still has swizzled children, latch the parent and
 * rewrite its swip back into a page id, then let the page age through a cooling FIFO
 * before the frame is actually taken. The cooling stage is what makes the unswizzle
 * revocable: a reader that arrives in time finds the page through the cooling hash
 * table and swizzles it straight back with no I/O.
 */

type NodeState = 'hot' | 'cooling' | 'evicted';

type Node = { id: number; label: string; parent: number; x: number; y: number; kind: 'root' | 'inner' | 'leaf' };

const NW = 64;
const NH = 34;

const NODES: Node[] = [
  { id: 0, label: 'P1', parent: -1, x: 160, y: 12, kind: 'root' },
  { id: 1, label: 'P2', parent: 0, x: 60, y: 92, kind: 'inner' },
  { id: 2, label: 'P3', parent: 0, x: 260, y: 92, kind: 'inner' },
  { id: 3, label: 'P4', parent: 1, x: 10, y: 172, kind: 'leaf' },
  { id: 4, label: 'P5', parent: 1, x: 110, y: 172, kind: 'leaf' },
  { id: 5, label: 'P6', parent: 2, x: 210, y: 172, kind: 'leaf' },
  { id: 6, label: 'P7', parent: 2, x: 310, y: 172, kind: 'leaf' },
];

const childrenOf = (id: number) => NODES.filter((n) => n.parent === id).map((n) => n.id);

type Phase = 'idle' | 'sampled' | 'rejected' | 'parent' | 'unswizzled' | 'aged' | 'written' | 'evicted';

type S = {
  state: NodeState[];
  fifo: number[]; // index 0 is the tail: the next page to leave the pool
  victim: number | null;
  phase: Phase;
  seed: number;
  steps: number;
  ios: number;
  writes: number;
  head: string;
  body: string;
};

const INITIAL: S = {
  state: ['hot', 'hot', 'hot', 'hot', 'hot', 'hot', 'cooling'],
  fifo: [6],
  victim: null,
  phase: 'idle',
  seed: 31,
  steps: 0,
  ios: 0,
  writes: 0,
  head: 'The pool is full and one frame has to go.',
  body:
    'Every parent here reaches its children through a swizzled swip — a word holding the frame address, ' +
    'not a page id — except P7, which has already been cooled. Sample a victim, or click any page, and ' +
    'step through what the buffer manager has to do before it can take that frame.',
};

type Action = { type: 'sample' } | { type: 'pick'; id: number } | { type: 'step' } | { type: 'touch' } | { type: 'reset' };

const IO_NS = 20_000;
const WRITE_NS = 24_000;
const DEREF_NS = 7;
const COOL_PROBE_NS = 52;

function reduce(s: S, a: Action, dirty: boolean): S {
  if (a.type === 'reset') return INITIAL;
  const next: S = { ...s, state: [...s.state], fifo: [...s.fifo], steps: s.steps + 1 };

  if (a.type === 'sample' || a.type === 'pick') {
    let id: number;
    if (a.type === 'pick') {
      id = a.id;
    } else {
      const cands = NODES.filter((n) => s.state[n.id] === 'hot').map((n) => n.id);
      if (cands.length === 0) return { ...s, head: 'Nothing left to sample.', body: 'Every page is cooling or evicted. Touch one to bring it back.' };
      const rng = makeRng(s.seed);
      id = cands[Math.floor(rng() * cands.length)];
      next.seed = s.seed + 7;
    }
    next.victim = id;
    next.phase = 'sampled';
    const n = NODES[id];
    next.head = `Candidate: ${n.label}.`;
    next.body =
      'LeanStore keeps no LRU list and no per-access counter — that bookkeeping is exactly the cost it is ' +
      'trying to avoid — so it picks replacement candidates by sampling a handful of frames at random. ' +
      'Whether this one can actually be evicted is decided next.';
    return next;
  }

  if (a.type === 'touch') {
    const v = s.victim;
    if (v === null) return { ...s, head: 'Pick a page first.', body: 'Click a page in the tree, or sample one.' };
    const st = s.state[v];
    const n = NODES[v];
    if (st === 'hot') {
      next.phase = s.phase;
      next.head = `${n.label} is swizzled: one dereference, ${fmtTime(DEREF_NS)}.`;
      next.body =
        'The parent already holds the frame address. The reader tests the tag bit, follows the pointer and ' +
        'validates the page’s version counter optimistically — no hash table is consulted, no latch is ' +
        'written, and nothing in shared memory is modified by a read.';
      return next;
    }
    if (st === 'cooling') {
      next.state[v] = 'hot';
      next.fifo = next.fifo.filter((x) => x !== v);
      next.phase = 'idle';
      next.victim = v;
      next.head = `${n.label} was rescued from the cooling stage — no I/O.`;
      next.body =
        `The parent’s swip was a page id, so this access cost a probe of the cooling hash table ` +
        `(${fmtTime(COOL_PROBE_NS)}): the page was found still resident, unlinked from the FIFO and swizzled ` +
        'back into the parent. This is why cooling exists — the decision to evict stays revocable right up ' +
        'until the frame is reused.';
      return next;
    }
    next.state[v] = 'hot';
    next.ios = s.ios + 1;
    next.phase = 'idle';
    next.head = `${n.label} had to be read back: ${fmtTime(IO_NS)}.`;
    next.body =
      'The swip held a page id, the cooling table did not have it, so this is a real miss. The page is read ' +
      'into a free frame and the parent’s swip is swizzled — overwritten with the new frame address — so the ' +
      'next reader pays a dereference again.';
    return next;
  }

  /* ------------------------------------------------------------------ step */
  const v = s.victim;
  if (v === null) return { ...s, head: 'No candidate yet.', body: 'Sample a victim, or click a page, to start the protocol.' };
  const n = NODES[v];
  const parent = n.parent >= 0 ? NODES[n.parent] : null;

  switch (s.phase) {
    case 'sampled': {
      const hotKids = childrenOf(v).filter((k) => s.state[k] === 'hot');
      if (hotKids.length > 0) {
        next.phase = 'rejected';
        next.head = `${n.label} cannot be evicted: ${hotKids.length} of its children are still swizzled.`;
        next.body =
          `Its child slots hold RAM addresses. Evicting ${n.label} would write those addresses to disk, where ` +
          'they mean nothing, and would leave those children with no reachable reference at all. The ' +
          'invariant the whole design rests on is that every page is referenced by exactly one swip — so an ' +
          'inner page may only leave after its swizzled children have. Step again to descend to one of them.';
        return next;
      }
      next.phase = 'parent';
      next.head = parent ? `Found the reference: it lives inside ${parent.label}.` : `${n.label} is the root — its swip lives in the tree’s metadata page.`;
      next.body =
        (parent
          ? `There is no central table to update. The only thing pointing at this frame is one word inside ` +
            `${parent.label}, so the parent must be found and latched exclusively before anything changes. ` +
            'LeanStore finds it with a parent pointer hint and falls back to a top-down traversal when the ' +
            'hint is stale. '
          : 'Even the root is reached through a swip; it just lives in the per-tree metadata page instead of ' +
            'in another node. ') +
        'This is the price of distributing the mapping: eviction is a write to another page.';
      return next;
    }

    case 'rejected': {
      const hotKids = childrenOf(v).filter((k) => s.state[k] === 'hot');
      const rng = makeRng(s.seed);
      const pickId = hotKids[Math.floor(rng() * hotKids.length)];
      next.seed = s.seed + 7;
      next.victim = pickId;
      next.phase = 'sampled';
      next.head = `Descending to ${NODES[pickId].label}.`;
      next.body =
        'Rather than throw the sample away, the buffer manager follows one of the swizzled children and ' +
        'considers it instead. The effect is a replacement policy with a bias towards leaves — which is what ' +
        'you want, because inner pages are the hot ones.';
      return next;
    }

    case 'parent': {
      next.state[v] = 'cooling';
      next.fifo = [...s.fifo, v];
      next.phase = 'unswizzled';
      next.head = `Unswizzled: ${parent ? parent.label : 'the metadata page'} now holds a page id, not a pointer.`;
      next.body =
        `The swip’s tag bit flips and the word is overwritten with ${n.label}’s page id. The page has not ` +
        'moved and not been written; it is simply no longer reachable by pointer. It goes to the head of the ' +
        'cooling FIFO and into the cooling hash table, which together hold roughly a tenth of the pool.';
      return next;
    }

    case 'unswizzled': {
      // Everything queued ahead of the victim leaves the pool first.
      const ahead = next.fifo.slice(0, next.fifo.indexOf(v));
      for (const q of ahead) next.state[q] = 'evicted';
      next.fifo = next.fifo.filter((x) => !ahead.includes(x));
      next.phase = 'aged';
      next.head = ahead.length
        ? `${ahead.map((q) => NODES[q].label).join(', ')} reached the end of the FIFO first.`
        : `${n.label} is now at the end of the FIFO.`;
      next.body =
        'The FIFO is the clock: a cooled page survives for as long as it takes the queue to drain past it. ' +
        'Nothing about this is per-access — no counter was incremented by any reader — which is the whole ' +
        'trick. A page that gets touched during this window is rescued for free; one that does not, leaves.';
      return next;
    }

    case 'aged': {
      if (dirty) {
        next.writes = s.writes + 1;
        next.phase = 'written';
        next.head = `${n.label} is dirty: ${fmtTime(WRITE_NS)} to write it out first.`;
        next.body =
          'A dirty page cannot be dropped. Engines that log write-ahead must also make sure the log records ' +
          'covering this page are durable before the page is — the same WAL-before-data rule the classic pool ' +
          'obeys, unchanged by swizzling. The write is issued asynchronously so the FIFO does not stall.';
        return next;
      }
      next.state[v] = 'evicted';
      next.fifo = next.fifo.filter((x) => x !== v);
      next.phase = 'evicted';
      next.head = `${n.label} evicted. The frame is free.`;
      next.body =
        'Clean, so there was nothing to write. No dangling reference is possible: the parent stopped pointing ' +
        'at this frame several steps ago, which is precisely why unswizzling had to happen first and not last.';
      return next;
    }

    case 'written': {
      next.state[v] = 'evicted';
      next.fifo = next.fifo.filter((x) => x !== v);
      next.phase = 'evicted';
      next.head = `${n.label} evicted after its write completed.`;
      next.body =
        'The frame goes back on the free list. Total cost of reclaiming it: one parent latch, one word ' +
        'rewritten, one page written. Compare that with a hash-table pool, which deletes one entry from one ' +
        'table — and then pays a probe on every access, forever.';
      return next;
    }

    case 'evicted':
    case 'idle':
    default: {
      next.victim = null;
      next.phase = 'idle';
      next.head = 'Ready for the next victim.';
      next.body = 'Sample again, or touch an evicted page to watch it fault back in and get re-swizzled.';
      return next;
    }
  }
}

/* ------------------------------------------------------------------- drawing */

const NODE_FILL: Record<NodeState, string> = {
  hot: 'var(--viz-clean)',
  cooling: 'var(--viz-warning)',
  evicted: 'var(--viz-neutral)',
};

const READER_COST: Record<NodeState, string> = {
  hot: `pointer dereference, ${fmtTime(DEREF_NS)}`,
  cooling: `cooling-table probe + re-swizzle, ${fmtTime(COOL_PROBE_NS)}`,
  evicted: `page fault, ${fmtTime(IO_NS)}`,
};

const PROTOCOL: { phase: Phase; what: string; latch: string }[] = [
  { phase: 'sampled', what: 'Sample random frames for a replacement candidate', latch: 'none — no list to protect' },
  { phase: 'rejected', what: 'Refuse a candidate with swizzled children; descend to one', latch: 'optimistic read of the candidate' },
  { phase: 'parent', what: 'Locate the single swip referencing the page (parent hint)', latch: 'parent, exclusive' },
  { phase: 'unswizzled', what: 'Rewrite the swip: frame address → page id; enqueue in the cooling FIFO', latch: 'parent + page, exclusive' },
  { phase: 'aged', what: 'Age through the FIFO; a reader may still rescue it', latch: 'cooling-table partition' },
  { phase: 'written', what: 'Write the page if dirty (WAL first)', latch: 'page, shared' },
  { phase: 'evicted', what: 'Return the frame to the free list', latch: 'none' },
];

export default function SwizzleEvictionWalk() {
  const [dirty, setDirty] = useState(true);
  const [s, setS] = useState<S>(INITIAL);
  const [ref, width] = useSize(720);
  const tip = useTip();

  const run = (a: Action) => setS((cur) => reduce(cur, a, dirty));

  const treeW = 386;
  const fifoX = treeW + 40;
  const svgW = Math.max(width, fifoX + 190);
  const diskY = 250;
  const height = diskY + 62;

  const resident = s.state.filter((x) => x !== 'evicted').length;
  const swizzled = NODES.filter((n) => s.state[n.id] === 'hot').length;

  const edge = (n: Node) => {
    const p = NODES[n.parent];
    const x1 = p.x + NW / 2;
    const y1 = p.y + NH;
    const x2 = n.x + NW / 2;
    const y2 = n.y;
    return { x1, y1, x2, y2, mx: (x1 + x2) / 2, my: (y1 + y2) / 2 };
  };

  return (
    <VizPanel
      title="One eviction, when the mapping lives in the pages"
      subtitle="A three-level tree in a full pool. Sample a victim — or click any page — and step through what has to happen before its frame can be reused."
      controls={
        <>
          <Button onClick={() => run({ type: 'sample' })} primary>
            Sample a victim
          </Button>
          <Button onClick={() => run({ type: 'step' })} disabled={s.victim === null && s.phase === 'idle'}>
            Step
          </Button>
          <Button onClick={() => run({ type: 'touch' })} disabled={s.victim === null} title="A concurrent reader arrives at this page right now">
            Touch this page
          </Button>
          <Check label="Victim is dirty" checked={dirty} onChange={setDirty} />
          <Button onClick={() => run({ type: 'reset' })}>Reset</Button>
        </>
      }
      legend={
        <Legend
          items={[
            { label: 'swizzled — parent holds a frame address', color: NODE_FILL.hot },
            { label: 'cooling — parent holds a page id, page still resident', color: NODE_FILL.cooling },
            { label: 'evicted — on disk only (dashed outline)', color: NODE_FILL.evicted },
            { label: 'pointer reference', color: 'var(--viz-1)', shape: 'line' },
            { label: 'page-id reference (dashed)', color: 'var(--viz-ink-muted)', shape: 'line' },
          ]}
        />
      }
      stats={
        <Stats
          items={[
            { label: 'Frames in use', value: `${resident} / ${NODES.length}`, hint: 'Cooling pages still occupy a frame' },
            { label: 'Swizzled references', value: `${swizzled}`, hint: 'Accesses that cost no lookup at all' },
            { label: 'Cooling stage', value: `${s.fifo.length}`, hint: 'Unswizzled but still rescuable' },
            { label: 'Reads faulted back', value: `${s.ios}`, hint: `${fmtTime(IO_NS)} each` },
            { label: 'Pages written', value: `${s.writes}`, hint: `${fmtTime(WRITE_NS)} each` },
          ]}
        />
      }
      note={
        <Note>
          <strong>{s.head}</strong> {s.body}
        </Note>
      }
      table={
        <>
          <table className="viz-table">
            <thead>
              <tr>
                <th>Page</th>
                <th>Referenced from</th>
                <th>Swip encoding</th>
                <th>State</th>
                <th>What a reader pays now</th>
              </tr>
            </thead>
            <tbody>
              {NODES.map((n) => (
                <tr key={n.id}>
                  <td>{n.label}</td>
                  <td>{n.parent >= 0 ? NODES[n.parent].label : 'tree metadata page'}</td>
                  <td>{s.state[n.id] === 'hot' ? 'frame address, tag bit clear' : `page id ${n.id + 1}, tag bit set`}</td>
                  <td>{s.state[n.id]}</td>
                  <td>{READER_COST[s.state[n.id]]}</td>
                </tr>
              ))}
            </tbody>
          </table>
          <table className="viz-table">
            <thead>
              <tr>
                <th>Step</th>
                <th>What the buffer manager does</th>
                <th>Latch held</th>
              </tr>
            </thead>
            <tbody>
              {PROTOCOL.map((p) => (
                <tr key={p.phase}>
                  <td>{p.phase}</td>
                  <td>{p.what}</td>
                  <td>{p.latch}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </>
      }
    >
      <div ref={ref}>
        <TooltipHost>
          <svg width={svgW} height={height} role="img" aria-label="A B-tree whose parent-to-child references are pointers, and one page being unswizzled and evicted">
            <defs>
              <marker id="swz-arrow" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto">
                <path d="M0,0 L8,4 L0,8 z" fill="var(--viz-1)" />
              </marker>
            </defs>

            {NODES.filter((n) => n.parent >= 0).map((n) => {
              const e = edge(n);
              const hot = s.state[n.id] === 'hot';
              return (
                <g key={`e${n.id}`}>
                  <line
                    x1={e.x1}
                    y1={e.y1}
                    x2={e.x2}
                    y2={e.y2}
                    stroke={hot ? 'var(--viz-1)' : 'var(--viz-ink-muted)'}
                    strokeWidth={hot ? 2 : 1.4}
                    strokeDasharray={hot ? undefined : '4 3'}
                    markerEnd={hot ? 'url(#swz-arrow)' : undefined}
                  />
                  {!hot ? (
                    <g {...tip(<span>The parent stores {NODES[n.id].label}’s page id. Reaching it costs a lookup — of the cooling table, or of the disk.</span>)}>
                      <rect x={e.mx - 22} y={e.my - 9} width={44} height={18} rx={4} fill="var(--viz-surface)" stroke="var(--viz-border)" />
                      <text x={e.mx} y={e.my + 4} textAnchor="middle" fill="var(--viz-ink-2)">
                        PID {n.id + 1}
                      </text>
                    </g>
                  ) : null}
                </g>
              );
            })}

            {NODES.map((n) => {
              const st = s.state[n.id];
              const isVictim = s.victim === n.id;
              return (
                <g
                  key={n.id}
                  onClick={() => run({ type: 'pick', id: n.id })}
                  style={{ cursor: 'pointer' }}
                  {...tip(
                    <span>
                      <strong>{n.label}</strong> — {n.kind}, {st}. {READER_COST[st]}. Click to make it the eviction candidate.
                    </span>,
                  )}
                >
                  <rect
                    x={n.x}
                    y={n.y}
                    width={NW}
                    height={NH}
                    rx={7}
                    fill={st === 'evicted' ? 'var(--viz-plane)' : NODE_FILL[st]}
                    stroke={isVictim ? 'var(--viz-ink)' : st === 'evicted' ? 'var(--viz-axis)' : 'var(--viz-surface)'}
                    strokeWidth={isVictim ? 2.5 : 1.5}
                    strokeDasharray={st === 'evicted' ? '4 3' : undefined}
                  />
                  <text
                    x={n.x + NW / 2}
                    y={n.y + 22}
                    textAnchor="middle"
                    fontWeight={600}
                    fill={st === 'hot' ? 'var(--viz-surface)' : st === 'cooling' ? 'var(--viz-ink)' : 'var(--viz-ink-muted)'}
                  >
                    {n.label}
                  </text>
                  {isVictim && dirty && st !== 'evicted' ? (
                    <circle cx={n.x + NW - 9} cy={n.y + 9} r={5} fill="var(--viz-dirty)" stroke="var(--viz-surface)" strokeWidth={1.2} />
                  ) : null}
                </g>
              );
            })}

            {s.victim !== null ? (
              <text x={NODES[s.victim].x + NW / 2} y={NODES[s.victim].y - 6} textAnchor="middle" fill="var(--viz-ink-2)">
                candidate
              </text>
            ) : null}

            <text x={fifoX} y={16} fill="var(--viz-ink)" fontWeight={600}>
              Cooling FIFO
            </text>
            <text x={fifoX} y={32} fill="var(--viz-ink-muted)">
              ~10% of the pool
            </text>
            {[0, 1, 2, 3].map((slot) => {
              const idx = s.fifo.length - 1 - slot; // newest at the top
              const id = idx >= 0 ? s.fifo[idx] : null;
              const y = 44 + slot * 30;
              return (
                <g key={slot}>
                  <rect x={fifoX} y={y} width={112} height={24} rx={6} fill={id === null ? 'var(--viz-plane)' : 'var(--viz-warning)'} stroke="var(--viz-border)" />
                  {id !== null ? (
                    <text x={fifoX + 56} y={y + 16} textAnchor="middle" fill="var(--viz-ink)" fontWeight={600}>
                      {NODES[id].label}
                    </text>
                  ) : null}
                </g>
              );
            })}
            <text x={fifoX + 120} y={60} fill="var(--viz-ink-muted)">
              head
            </text>
            <text x={fifoX + 120} y={150} fill="var(--viz-ink-muted)">
              tail → evict
            </text>

            <rect x={0} y={diskY} width={svgW - 4} height={48} rx={8} fill="var(--viz-plane)" stroke="var(--viz-border)" strokeDasharray="5 4" />
            <text x={12} y={diskY + 20} fill="var(--viz-ink-2)" fontWeight={600}>
              Disk
            </text>
            <text x={12} y={diskY + 38} fill="var(--viz-ink-muted)">
              reachable only by page id
            </text>
            {NODES.filter((n) => s.state[n.id] === 'evicted').map((n, i) => (
              <g key={`d${n.id}`} {...tip(<span>{n.label} holds no frame. Its parent’s swip is a page id; the next reader pays {fmtTime(IO_NS)}.</span>)}>
                <rect x={150 + i * 58} y={diskY + 12} width={48} height={24} rx={5} fill="var(--viz-neutral)" stroke="var(--viz-axis)" />
                <text x={150 + i * 58 + 24} y={diskY + 28} textAnchor="middle" fill="var(--viz-ink-2)">
                  {n.label}
                </text>
              </g>
            ))}
          </svg>
        </TooltipHost>
      </div>
    </VizPanel>
  );
}
