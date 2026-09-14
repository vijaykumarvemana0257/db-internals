import { useMemo, useState } from 'react';
import {
  VizPanel,
  Segmented,
  Slider,
  Button,
  Legend,
  Stats,
  Note,
  TooltipHost,
  useTip,
  makeRng,
  useSize,
} from './Viz';

/**
 * Replacement policies on one trace.
 *
 * Every policy here is the textbook algorithm, run step by step on the same access
 * trace, so the differences you see are the algorithms' and not the workload's:
 *   LRU     exact recency list, move-to-front on every hit
 *   CLOCK   PostgreSQL's clock sweep: usage_count capped at 5, decremented per visit
 *   LRU-K   K = 2, victim = largest backward K-distance, history retained for
 *           non-resident pages (no correlated-reference window — see the page)
 *   2Q      full version: A1in FIFO (Kin = c/4), A1out ghost FIFO (Kout = c/2), Am LRU
 *   ARC     T1/T2 resident lists, B1/B2 ghosts, adaptive target p
 *   OPT     Belady's MIN, offline: evict the page whose next use is furthest away
 */

type Policy = 'lru' | 'clock' | 'lruk' | 'twoq' | 'arc';
type Alg = Policy | 'opt';
type Cls = 'hot' | 'loop' | 'rand';
type Access = { page: number; cls: Cls };

const TRACE_LEN = 192;
const MAX_USAGE = 5; // BM_MAX_USAGE_COUNT

const POLICIES: { value: Policy; label: string; title: string }[] = [
  { value: 'lru', label: 'LRU', title: 'Exact recency: a list splice on every hit' },
  { value: 'clock', label: 'CLOCK', title: "PostgreSQL's clock sweep with usage_count" },
  { value: 'lruk', label: 'LRU-K', title: 'K = 2: evict the largest backward 2-distance' },
  { value: 'twoq', label: '2Q', title: 'A1in FIFO + A1out ghosts + Am LRU' },
  { value: 'arc', label: 'ARC', title: 'Adaptive T1/T2 with B1/B2 ghost lists' },
];

const ALG_NAME: Record<Alg, string> = {
  lru: 'LRU',
  clock: 'CLOCK',
  lruk: 'LRU-2',
  twoq: '2Q',
  arc: 'ARC',
  opt: 'OPT',
};

const CLS_COLOR: Record<Cls, string> = {
  hot: 'var(--viz-1)',
  loop: 'var(--viz-2)',
  rand: 'var(--viz-3)',
};

const CLS_TAG: Record<Cls, string> = { hot: 'H', loop: 'L', rand: 'R' };

function classOf(page: number): Cls {
  return page < 100 ? 'hot' : page < 900 ? 'loop' : 'rand';
}

function pageLabel(page: number) {
  const c = classOf(page);
  return `${CLS_TAG[c]}${c === 'hot' ? page : c === 'loop' ? page - 100 : page - 900}`;
}

/* ------------------------------------------------------------------- trace */

/** Deterministic mix of a re-referenced hot set, a cyclic loop, and one-touch pages. */
function buildTrace(hotPages: number, loopLen: number, randPct: number): Access[] {
  const rng = makeRng(19930824); // LRU-K, VLDB '93
  const out: Access[] = [];
  let cursor = 0;
  let alt = 0;
  for (let i = 0; i < TRACE_LEN; i++) {
    const r = rng() * 100;
    if (r < randPct) {
      out.push({ page: 900 + Math.floor(rng() * 240), cls: 'rand' });
    } else if (loopLen > 0 && alt++ % 2 === 0) {
      out.push({ page: 100 + (cursor++ % loopLen), cls: 'loop' });
    } else {
      out.push({ page: 1 + Math.floor(rng() * hotPages), cls: 'hot' });
    }
  }
  return out;
}

/* -------------------------------------------------------------- simulation */

type GhostRow = { label: string; hint: string; items: number[] };

type Sim = {
  slots: (number | null)[];
  meta: Map<number, string>;
  ghosts: GhostRow[];
  hand: number | null;
  hits: number;
  misses: number;
  evictions: number;
  examined: number;
  allocs: number;
  hotHits: number;
  hotAcc: number;
  curve: number[];
  hitAt: boolean[];
  last: { i: number; page: number; cls: Cls; hit: boolean; victim: number | null; why: string } | null;
};

function simulate(trace: Access[], nFrames: number, alg: Alg, upto: number): Sim {
  const n = Math.min(upto, trace.length);
  const slots: (number | null)[] = new Array(nFrames).fill(null);
  const slotOf = new Map<number, number>();

  let hits = 0;
  let misses = 0;
  let evictions = 0;
  let examined = 0;
  let allocs = 0;
  let hotHits = 0;
  let hotAcc = 0;
  const curve: number[] = [];
  const hitAt: boolean[] = [];
  let last: Sim['last'] = null;

  // LRU
  const lruOrder: number[] = []; // MRU first
  // CLOCK
  const usage = new Map<number, number>();
  let hand = 0;
  // LRU-K
  const K = 2;
  const hist = new Map<number, number[]>(); // newest first, at most K entries
  // 2Q
  const Kin = Math.max(1, Math.round(nFrames * 0.25));
  const Kout = Math.max(1, Math.round(nFrames * 0.5));
  const a1in: number[] = []; // FIFO, newest first
  const a1out: number[] = []; // ghost FIFO, newest first
  const am: number[] = []; // LRU, MRU first
  // ARC
  const T1: number[] = [];
  const T2: number[] = [];
  const B1: number[] = [];
  const B2: number[] = [];
  let p = 0;
  // OPT
  const nextIdx = new Array<number>(trace.length).fill(Infinity);
  if (alg === 'opt') {
    const seen = new Map<number, number>();
    for (let i = trace.length - 1; i >= 0; i--) {
      const nx = seen.get(trace[i].page);
      nextIdx[i] = nx === undefined ? Infinity : nx;
      seen.set(trace[i].page, i);
    }
  }
  const lastIdx = new Map<number, number>();

  const place = (pg: number) => {
    let s = slots.indexOf(null);
    if (s < 0) s = 0;
    slots[s] = pg;
    slotOf.set(pg, s);
  };
  const evict = (pg: number) => {
    const s = slotOf.get(pg);
    if (s !== undefined) {
      slots[s] = null;
      slotOf.delete(pg);
    }
    evictions++;
  };

  for (let i = 0; i < n; i++) {
    const a = trace[i];
    const t = i + 1;
    if (a.cls === 'hot') hotAcc++;
    const resident = slotOf.has(a.page);
    if (resident) {
      hits++;
      if (a.cls === 'hot') hotHits++;
    } else {
      misses++;
    }
    let victim: number | null = null;
    let why = '';

    switch (alg) {
      case 'lru': {
        if (resident) {
          lruOrder.splice(lruOrder.indexOf(a.page), 1);
          lruOrder.unshift(a.page);
          why = 'Hit — and the hit is not free: the page is unlinked and relinked at the head of the LRU list, under the lock that protects it.';
        } else {
          if (slotOf.size >= nFrames) {
            victim = lruOrder.pop() ?? null;
            if (victim !== null) evict(victim);
          }
          place(a.page);
          lruOrder.unshift(a.page);
          allocs++;
          why =
            victim === null
              ? 'Miss into a free frame — no victim needed yet.'
              : `Miss: the tail of the list, ${pageLabel(victim)}, is the victim. LRU has no idea how often it was touched, only when it was touched last.`;
        }
        break;
      }

      case 'clock': {
        if (resident) {
          const u = Math.min(MAX_USAGE, (usage.get(a.page) ?? 0) + 1);
          usage.set(a.page, u);
          why = `Hit — the whole cost is usage_count → ${u} (capped at BM_MAX_USAGE_COUNT = ${MAX_USAGE}). No list to splice, no global lock to take.`;
        } else {
          let target = slots.indexOf(null);
          allocs++;
          if (target < 0) {
            let steps = 0;
            const guard = nFrames * (MAX_USAGE + 2);
            for (;;) {
              const pg = slots[hand];
              const u = pg === null ? 0 : usage.get(pg) ?? 0;
              if (pg === null || u === 0 || steps > guard) {
                target = hand;
                if (pg !== null) {
                  victim = pg;
                  evict(pg);
                  usage.delete(pg);
                }
                hand = (hand + 1) % nFrames;
                steps++;
                break;
              }
              usage.set(pg, u - 1);
              hand = (hand + 1) % nFrames;
              steps++;
            }
            examined += steps;
            why = `Miss: the sweep examined ${steps} frame${steps === 1 ? '' : 's'}, decrementing usage_count as it went, and took the first one that reached zero${victim === null ? '' : ` — ${pageLabel(victim)}`}.`;
          } else {
            examined += 1;
            why = 'Miss into a free frame — Postgres pulls from the freelist before it sweeps at all.';
          }
          slots[target] = a.page;
          slotOf.set(a.page, target);
          usage.set(a.page, 1);
        }
        break;
      }

      case 'lruk': {
        const h = hist.get(a.page) ?? [];
        h.unshift(t);
        if (h.length > K) h.length = K;
        hist.set(a.page, h);

        if (resident) {
          why = 'Hit — the reference is appended to this page’s history. Nothing else in the pool moves.';
        } else {
          if (slotOf.size >= nFrames) {
            let best: number | null = null;
            let bestD = -1;
            let bestLast = Infinity;
            for (const pg of slotOf.keys()) {
              const ph = hist.get(pg) ?? [];
              const d = ph.length >= K ? t - ph[K - 1] : Infinity;
              const lr = ph[0] ?? 0;
              if (best === null || d > bestD || (d === bestD && lr < bestLast)) {
                best = pg;
                bestD = d;
                bestLast = lr;
              }
            }
            if (best !== null) {
              victim = best;
              evict(best); // history is kept: the retained information period
              why = `Miss: victim is the page with the largest backward 2-distance (${bestD === Infinity ? 'never referenced twice' : `${bestD} accesses`}) — ${pageLabel(best)}. A page seen once looks infinitely cold no matter how recently it was touched.`;
            }
          } else {
            why = 'Miss into a free frame.';
          }
          place(a.page);
          allocs++;
        }

        // Retained information period: history for non-resident pages is bounded.
        const limit = nFrames * 4;
        const nonRes: number[] = [];
        for (const pg of hist.keys()) if (!slotOf.has(pg)) nonRes.push(pg);
        if (nonRes.length > limit) {
          nonRes.sort((x, y) => (hist.get(x)?.[0] ?? 0) - (hist.get(y)?.[0] ?? 0));
          for (let j = 0; j < nonRes.length - limit; j++) hist.delete(nonRes[j]);
        }
        break;
      }

      case 'twoq': {
        const inAm = am.indexOf(a.page);
        const inA1in = a1in.indexOf(a.page);
        if (inAm >= 0) {
          am.splice(inAm, 1);
          am.unshift(a.page);
          why = 'Hit in Am: promoted to the head of the hot LRU queue.';
        } else if (inA1in >= 0) {
          why = 'Hit in A1in — and the page does not move. A second touch while still in the FIFO is treated as a correlated reference, not as proof of a hot page.';
        } else {
          if (slotOf.size >= nFrames) {
            if (a1in.length > Kin || am.length === 0) {
              const y = a1in.pop();
              if (y !== undefined) {
                victim = y;
                evict(y);
                a1out.unshift(y);
                if (a1out.length > Kout) a1out.pop();
              }
            } else {
              const y = am.pop();
              if (y !== undefined) {
                victim = y;
                evict(y);
              }
            }
          }
          const g = a1out.indexOf(a.page);
          if (g >= 0) {
            a1out.splice(g, 1);
            am.unshift(a.page);
            why = `Miss — but ${pageLabel(a.page)} was still remembered in A1out, so this is its second independent reference and it enters Am directly.`;
          } else {
            a1in.unshift(a.page);
            why = `Miss: a first-seen page enters the A1in FIFO (Kin = ${Kin} frames), where it will age out into A1out without ever displacing anything in Am.`;
          }
          place(a.page);
          allocs++;
        }
        break;
      }

      case 'arc': {
        const c = nFrames;
        const replace = (inB2: boolean) => {
          if (slotOf.size < nFrames) return;
          if (T1.length >= 1 && ((inB2 && T1.length === p) || T1.length > p)) {
            const y = T1.pop();
            if (y !== undefined) {
              victim = y;
              evict(y);
              B1.unshift(y);
              if (B1.length > c) B1.pop();
            }
          } else if (T2.length >= 1) {
            const y = T2.pop();
            if (y !== undefined) {
              victim = y;
              evict(y);
              B2.unshift(y);
              if (B2.length > c) B2.pop();
            }
          }
        };

        const i1 = T1.indexOf(a.page);
        const i2 = T2.indexOf(a.page);
        if (i1 >= 0) {
          T1.splice(i1, 1);
          T2.unshift(a.page);
          why = 'Hit in T1: a second reference promotes the page out of the recency list and into T2, the frequency list.';
        } else if (i2 >= 0) {
          T2.splice(i2, 1);
          T2.unshift(a.page);
          why = 'Hit in T2: it stays in the frequency list, at the head.';
        } else {
          const g1 = B1.indexOf(a.page);
          const g2 = B2.indexOf(a.page);
          if (g1 >= 0) {
            p = Math.min(c, p + Math.max(1, Math.floor(B2.length / Math.max(1, B1.length))));
            replace(false);
            B1.splice(g1, 1);
            T2.unshift(a.page);
            place(a.page);
            allocs++;
            why = `Ghost hit in B1 — ARC evicted this page from T1 too early, so the recency target p grows to ${p} of ${c} frames and T1 is allowed more room.`;
          } else if (g2 >= 0) {
            p = Math.max(0, p - Math.max(1, Math.floor(B1.length / Math.max(1, B2.length))));
            replace(true);
            B2.splice(g2, 1);
            T2.unshift(a.page);
            place(a.page);
            allocs++;
            why = `Ghost hit in B2 — a page that had been referenced twice was evicted too early, so p shrinks to ${p} and the frequency list T2 takes back space.`;
          } else {
            const l1 = T1.length + B1.length;
            const total = T1.length + T2.length + B1.length + B2.length;
            if (l1 === c) {
              if (T1.length < c) {
                B1.pop();
                replace(false);
              } else {
                const y = T1.pop();
                if (y !== undefined) {
                  victim = y;
                  evict(y);
                }
              }
            } else if (l1 < c && total >= c) {
              if (total >= 2 * c) B2.pop();
              replace(false);
            }
            T1.unshift(a.page);
            place(a.page);
            allocs++;
            why = `Miss on a page no list remembers: it enters T1 at the head. One-touch pages live and die in T1 without ever touching T2 (p = ${p} of ${c}).`;
          }
        }
        break;
      }

      case 'opt': {
        if (!resident) {
          if (slotOf.size >= nFrames) {
            let best: number | null = null;
            let bestNext = -1;
            for (const pg of slotOf.keys()) {
              const li = lastIdx.get(pg);
              const nx = li === undefined ? Infinity : nextIdx[li];
              if (best === null || nx > bestNext) {
                best = pg;
                bestNext = nx;
              }
            }
            if (best !== null) {
              victim = best;
              evict(best);
              why = `Miss: evict ${pageLabel(best)}, whose next use is ${bestNext === Infinity ? 'never' : `${bestNext - i} accesses away`}. Only an offline algorithm can know this.`;
            }
          }
          place(a.page);
          allocs++;
        } else {
          why = 'Hit.';
        }
        break;
      }
    }

    lastIdx.set(a.page, i);
    curve.push(hits / t);
    hitAt.push(resident);
    last = { i, page: a.page, cls: a.cls, hit: resident, victim, why };
  }

  /* ---- per-frame annotation and ghost rows, read off the final structures ---- */

  const meta = new Map<number, string>();
  const ghosts: GhostRow[] = [];
  const resident = [...slotOf.keys()];

  if (alg === 'lru') {
    for (const pg of resident) {
      const r = lruOrder.indexOf(pg) + 1;
      meta.set(pg, r === 1 ? 'MRU' : r === lruOrder.length ? 'LRU ▸ next victim' : `#${r}`);
    }
  } else if (alg === 'clock') {
    for (const pg of resident) meta.set(pg, `usage_count ${usage.get(pg) ?? 0}`);
  } else if (alg === 'lruk') {
    const t = n + 1;
    for (const pg of resident) {
      const ph = hist.get(pg) ?? [];
      meta.set(pg, ph.length >= K ? `b₂ = ${t - ph[K - 1]}` : 'b₂ = ∞');
    }
    const kept = [...hist.keys()].filter((pg) => !slotOf.has(pg));
    kept.sort((x, y) => (hist.get(y)?.[0] ?? 0) - (hist.get(x)?.[0] ?? 0));
    ghosts.push({
      label: 'history',
      hint: 'The retained information period: reference history kept for pages that are no longer resident, so a second reference can still be recognised.',
      items: kept.slice(0, 14),
    });
  } else if (alg === 'twoq') {
    for (const pg of resident) {
      const ai = a1in.indexOf(pg);
      meta.set(pg, ai >= 0 ? `A1in #${ai + 1}` : `Am #${am.indexOf(pg) + 1}`);
    }
    ghosts.push({
      label: `A1out (${Kout})`,
      hint: 'Page identifiers only, no frames: a hit here is the second reference that promotes a page into Am.',
      items: a1out.slice(0, 14),
    });
  } else if (alg === 'arc') {
    for (const pg of resident) {
      const i1 = T1.indexOf(pg);
      meta.set(pg, i1 >= 0 ? `T1 #${i1 + 1}` : `T2 #${T2.indexOf(pg) + 1}`);
    }
    ghosts.push({
      label: 'B1 ghosts',
      hint: 'Evicted from T1. A hit here means recency was starved, and pushes the target p up.',
      items: B1.slice(0, 14),
    });
    ghosts.push({
      label: 'B2 ghosts',
      hint: 'Evicted from T2. A hit here means frequency was starved, and pulls the target p down.',
      items: B2.slice(0, 14),
    });
  } else {
    for (const pg of resident) {
      const li = lastIdx.get(pg);
      const nx = li === undefined ? Infinity : nextIdx[li];
      meta.set(pg, nx === Infinity ? 'never used again' : `next use +${nx - n + 1}`);
    }
  }

  return {
    slots,
    meta,
    ghosts,
    hand: alg === 'clock' ? hand : null,
    hits,
    misses,
    evictions,
    examined,
    allocs,
    hotHits,
    hotAcc,
    curve,
    hitAt,
    last,
  };
}

/* --------------------------------------------------------------- component */

export default function ReplacementPolicyPlayground() {
  const [policy, setPolicy] = useState<Policy>('lru');
  const [frames, setFrames] = useState(8);
  const [hotPages, setHotPages] = useState(5);
  const [loopLen, setLoopLen] = useState(10);
  const [randPct, setRandPct] = useState(10);
  const [step, setStep] = useState(48);
  const [ref, width] = useSize(760);
  const tip = useTip();

  const trace = useMemo(() => buildTrace(hotPages, loopLen, randPct), [hotPages, loopLen, randPct]);
  const cur = useMemo(() => simulate(trace, frames, policy, step), [trace, frames, policy, step]);
  const full = useMemo(() => simulate(trace, frames, policy, TRACE_LEN), [trace, frames, policy]);
  const optFull = useMemo(() => simulate(trace, frames, 'opt', TRACE_LEN), [trace, frames]);
  const allAlgs = useMemo(
    () =>
      (['lru', 'clock', 'lruk', 'twoq', 'arc', 'opt'] as Alg[]).map((a) => ({
        alg: a,
        sim: simulate(trace, frames, a, TRACE_LEN),
      })),
    [trace, frames],
  );

  const done = Math.min(step, TRACE_LEN);
  const rate = done === 0 ? 0 : cur.hits / done;
  const optRate = done === 0 ? 0 : optFull.curve[done - 1];
  const hotRate = cur.hotAcc === 0 ? 0 : cur.hotHits / cur.hotAcc;

  /* ---- pool drawing ---- */
  const labelW = 96;
  const boxW = 78;
  const boxH = 58;
  const gap = 10;
  const poolW = Math.max(width, labelW + frames * (boxW + gap) + 12);
  const ghostTop = 34 + boxH + 16;
  const ghostH = 40;
  const poolH = ghostTop + Math.max(1, cur.ghosts.length) * ghostH + 6;

  /* ---- trace strip ---- */
  const stripW = Math.max(360, width - 8);
  const tickW = (stripW - labelW - 8) / TRACE_LEN;
  const stripH = 62;

  /* ---- hit-rate curve ---- */
  const chartW = Math.max(360, width - 8);
  const chartH = 156;
  const cl = 44;
  const cr = 14;
  const ct = 12;
  const cb = 26;
  const px = (i: number) => cl + ((i + 1) / TRACE_LEN) * (chartW - cl - cr);
  const py = (v: number) => ct + (1 - v) * (chartH - ct - cb);
  const line = (vals: number[]) =>
    vals
      .slice(0, done)
      .map((v, i) => `${i === 0 ? 'M' : 'L'}${px(i).toFixed(1)},${py(v).toFixed(1)}`)
      .join(' ');

  const last = cur.last;
  const loopFloods = loopLen >= frames && loopLen > 0;
  const loopCrowds = loopLen > 0 && !loopFloods && loopLen + hotPages > frames;

  return (
    <VizPanel
      title="Replacement policy playground"
      subtitle="One trace, six policies. Compose the workload, step the trace, and watch the internal state each policy keeps — reference bits and the clock hand, backward K-distances, queue membership, ghost lists — while its hit rate is drawn against the offline optimum."
      controls={
        <>
          <Segmented label="Policy" value={policy} onChange={setPolicy} options={POLICIES} />
          <Slider label="Frames" min={4} max={12} value={frames} onChange={setFrames} format={(n) => `${n} pages`} />
          <Slider label="Hot set" min={2} max={10} value={hotPages} onChange={setHotPages} format={(n) => `${n} pages`} />
          <Slider
            label="Loop length"
            min={0}
            max={20}
            value={loopLen}
            onChange={setLoopLen}
            format={(n) => (n === 0 ? 'off' : `${n} pages`)}
          />
          <Slider label="One-touch" min={0} max={40} value={randPct} onChange={setRandPct} format={(n) => `${n}%`} />
          <Button onClick={() => setStep((s) => Math.min(TRACE_LEN, s + 1))} disabled={step >= TRACE_LEN} primary>
            Step
          </Button>
          <Button onClick={() => setStep((s) => Math.min(TRACE_LEN, s + 10))} disabled={step >= TRACE_LEN}>
            Step ×10
          </Button>
          <Button onClick={() => setStep(TRACE_LEN)} disabled={step >= TRACE_LEN}>
            Run to end
          </Button>
          <Button onClick={() => setStep(0)} disabled={step === 0}>
            Reset
          </Button>
        </>
      }
      legend={
        <Legend
          items={[
            { label: 'Hot set (re-referenced)', color: CLS_COLOR.hot },
            { label: 'Loop (cyclic scan)', color: CLS_COLOR.loop },
            { label: 'One-touch page', color: CLS_COLOR.rand },
            { label: 'Ghost / history entry (no frame)', color: 'var(--viz-ink-muted)' },
            { label: `${ALG_NAME[policy]} hit rate`, color: 'var(--viz-1)', shape: 'line' },
            { label: 'OPT (offline optimum)', color: 'var(--viz-ink-2)', shape: 'line' },
          ]}
        />
      }
      stats={
        <Stats
          items={[
            { label: 'Accesses', value: `${done} / ${TRACE_LEN}`, hint: 'How far through the trace you have stepped' },
            { label: `${ALG_NAME[policy]} hit rate`, value: `${(rate * 100).toFixed(1)}%` },
            { label: 'OPT hit rate', value: `${(optRate * 100).toFixed(1)}%`, hint: "Belady's MIN on the same trace and the same number of frames" },
            { label: 'Hot-set hit rate', value: `${(hotRate * 100).toFixed(1)}%`, hint: 'Hits on the re-referenced working set only — the number the loop destroys' },
            policy === 'clock'
              ? {
                  label: 'Frames per sweep',
                  value: cur.allocs === 0 ? '—' : (cur.examined / cur.allocs).toFixed(2),
                  hint: 'Buffers examined per allocation: the real cost of the clock sweep',
                }
              : { label: 'Evictions', value: String(cur.evictions) },
          ]}
        />
      }
      note={
        <Note>
          {last ? (
            <>
              <strong>
                #{last.i + 1} → {pageLabel(last.page)} · {last.hit ? 'hit' : 'miss'}
                {last.victim !== null ? ` · evicted ${pageLabel(last.victim)}` : ''}
              </strong>{' '}
              {last.why}{' '}
            </>
          ) : (
            <>
              <strong>Press Step.</strong> The pool is empty; every access is a compulsory miss until the frames
              fill.{' '}
            </>
          )}
          {loopFloods
            ? `The loop alone is ${loopLen} pages and the pool holds ${frames}: under pure recency each loop page is evicted an access or two before it comes round again, so the loop hits almost nothing and takes the hot set down with it.`
            : loopCrowds
              ? `The ${loopLen}-page loop fits in ${frames} frames on its own, but not next to a ${hotPages}-page hot set — the two compete for the same frames, and only the policies that can tell a re-referenced page from a one-touch page keep the hot set resident.`
              : loopLen > 0
                ? `The ${loopLen}-page loop and the ${hotPages}-page hot set both fit in ${frames} frames, so every policy holds the whole working set. Push the loop past the frame count to find the cliff.`
                : 'With the loop off, the workload is a hot set plus one-touch noise — the case where frequency-aware policies protect the hot set and pure recency does not.'}
        </Note>
      }
      table={
        <table className="viz-table">
          <thead>
            <tr>
              <th>Policy</th>
              <th>Hits</th>
              <th>Misses</th>
              <th>Hit rate</th>
              <th>Hot-set hit rate</th>
              <th>Evictions</th>
            </tr>
          </thead>
          <tbody>
            {allAlgs.map(({ alg, sim }) => (
              <tr key={alg}>
                <td>{ALG_NAME[alg]}</td>
                <td>{sim.hits}</td>
                <td>{sim.misses}</td>
                <td>{((sim.hits / TRACE_LEN) * 100).toFixed(1)}%</td>
                <td>{sim.hotAcc === 0 ? '—' : `${((sim.hotHits / sim.hotAcc) * 100).toFixed(1)}%`}</td>
                <td>{sim.evictions}</td>
              </tr>
            ))}
          </tbody>
        </table>
      }
    >
      <TooltipHost>
        <div ref={ref}>
          {/* ---------------------------------------------------------- pool */}
          <svg width={poolW} height={poolH} role="img" aria-label={`${ALG_NAME[policy]} buffer pool state`}>
            <text x={0} y={30} fill="var(--viz-ink-2)">
              frames
            </text>
            {cur.slots.map((pg, i) => {
              const x = labelW + i * (boxW + gap);
              const cls = pg === null ? null : classOf(pg);
              return (
                <g key={i} {...tip(
                  pg === null ? (
                    <>frame {i}: free</>
                  ) : (
                    <>
                      <strong>frame {i} · page {pageLabel(pg)}</strong>
                      <br />
                      {cls === 'hot' ? 'Hot set — referenced again and again.' : cls === 'loop' ? 'Loop page — referenced once per cycle.' : 'One-touch page.'}
                      <br />
                      {cur.meta.get(pg) ?? ''}
                    </>
                  ),
                )}>
                  <rect
                    x={x}
                    y={14}
                    width={boxW}
                    height={boxH}
                    rx={6}
                    fill={pg === null ? 'var(--viz-plane)' : CLS_COLOR[cls as Cls]}
                    stroke={pg === null ? 'var(--viz-axis)' : 'var(--viz-surface)'}
                    strokeWidth={pg === null ? 1 : 2}
                    strokeDasharray={pg === null ? '3 3' : undefined}
                  />
                  {pg === null ? (
                    <text x={x + boxW / 2} y={44} textAnchor="middle" fill="var(--viz-ink-muted)">
                      free
                    </text>
                  ) : (
                    <>
                      <text x={x + boxW / 2} y={38} textAnchor="middle" fill="var(--viz-surface)" style={{ fontWeight: 600 }}>
                        {pageLabel(pg)}
                      </text>
                      <text x={x + boxW / 2} y={52} textAnchor="middle" fill="var(--viz-surface)" style={{ fontSize: 10 }}>
                        {cur.meta.get(pg)}
                      </text>
                    </>
                  )}
                  {cur.hand === i ? (
                    <>
                      <path
                        d={`M${x + boxW / 2 - 6},4 L${x + boxW / 2 + 6},4 L${x + boxW / 2},13 Z`}
                        fill="var(--viz-ink)"
                      />
                      <text x={x + boxW / 2 + 10} y={11} fill="var(--viz-ink-2)" style={{ fontSize: 10 }}>
                        hand
                      </text>
                    </>
                  ) : null}
                </g>
              );
            })}

            {cur.ghosts.length === 0 ? (
              <text x={0} y={ghostTop + 20} fill="var(--viz-ink-muted)">
                {policy === 'clock'
                  ? 'no ghost lists — CLOCK remembers nothing about pages it has evicted'
                  : 'no ghost lists — LRU remembers nothing about pages it has evicted'}
              </text>
            ) : (
              cur.ghosts.map((row, gi) => (
                <g key={row.label}>
                  <text x={0} y={ghostTop + gi * ghostH + 22} fill="var(--viz-ink-2)" style={{ fontSize: 11 }}>
                    {row.label}
                  </text>
                  {row.items.length === 0 ? (
                    <text x={labelW} y={ghostTop + gi * ghostH + 22} fill="var(--viz-ink-muted)" style={{ fontSize: 11 }}>
                      empty
                    </text>
                  ) : (
                    row.items.map((pg, j) => {
                      const x = labelW + j * 52;
                      return (
                        <g key={`${row.label}-${pg}`} {...tip(<>{row.hint}</>)}>
                          <rect
                            x={x}
                            y={ghostTop + gi * ghostH + 6}
                            width={46}
                            height={22}
                            rx={5}
                            fill="var(--viz-neutral)"
                            stroke={CLS_COLOR[classOf(pg)]}
                            strokeWidth={1.5}
                            strokeDasharray="3 2"
                          />
                          <text
                            x={x + 23}
                            y={ghostTop + gi * ghostH + 21}
                            textAnchor="middle"
                            fill="var(--viz-ink-2)"
                            style={{ fontSize: 11 }}
                          >
                            {pageLabel(pg)}
                          </text>
                        </g>
                      );
                    })
                  )}
                </g>
              ))
            )}
          </svg>

          {/* --------------------------------------------------------- trace */}
          <svg width={stripW} height={stripH} role="img" aria-label="Access trace">
            <text x={0} y={22} fill="var(--viz-ink-2)">
              trace
            </text>
            <text x={0} y={40} fill="var(--viz-ink-muted)" style={{ fontSize: 10 }}>
              tall = hit
            </text>
            {trace.map((a, i) => {
              const x = labelW + i * tickW;
              const past = i < done;
              const hit = past && full.hitAt[i];
              const h = hit ? 30 : past ? 14 : 22;
              return (
                <rect
                  key={i}
                  x={x}
                  y={38 - h}
                  width={Math.max(1.2, tickW - 0.8)}
                  height={h}
                  fill={CLS_COLOR[a.cls]}
                  opacity={past ? (hit ? 1 : 0.55) : 0.16}
                />
              );
            })}
            {done > 0 ? (
              <line
                x1={labelW + (done - 0.5) * tickW}
                x2={labelW + (done - 0.5) * tickW}
                y1={4}
                y2={46}
                stroke="var(--viz-ink)"
                strokeWidth={1.5}
              />
            ) : null}
            <text x={labelW} y={56} fill="var(--viz-ink-muted)" style={{ fontSize: 10 }}>
              access 1
            </text>
            <text x={labelW + TRACE_LEN * tickW} y={56} textAnchor="end" fill="var(--viz-ink-muted)" style={{ fontSize: 10 }}>
              access {TRACE_LEN}
            </text>
          </svg>

          {/* --------------------------------------------------------- chart */}
          <svg width={chartW} height={chartH} role="img" aria-label="Cumulative hit rate versus OPT">
            {[0, 0.25, 0.5, 0.75, 1].map((v) => (
              <g key={v}>
                <line className="viz-grid-line" x1={cl} x2={chartW - cr} y1={py(v)} y2={py(v)} />
                <text x={cl - 6} y={py(v) + 4} textAnchor="end" fill="var(--viz-ink-muted)" style={{ fontSize: 10 }}>
                  {v * 100}%
                </text>
              </g>
            ))}
            <line className="viz-axis-line" x1={cl} x2={chartW - cr} y1={py(0)} y2={py(0)} />
            {done > 1 ? (
              <>
                <path d={line(optFull.curve)} fill="none" stroke="var(--viz-ink-2)" strokeWidth={1.6} strokeDasharray="5 4" />
                <path d={line(full.curve)} fill="none" stroke="var(--viz-1)" strokeWidth={2.2} />
                <circle cx={px(done - 1)} cy={py(full.curve[done - 1])} r={3.5} fill="var(--viz-1)" />
                <text
                  x={px(done - 1) - 8}
                  y={py(full.curve[done - 1]) - 8}
                  textAnchor="end"
                  fill="var(--viz-ink)"
                  style={{ fontSize: 11 }}
                >
                  {ALG_NAME[policy]} {(full.curve[done - 1] * 100).toFixed(0)}%
                </text>
                <text
                  x={px(done - 1) - 8}
                  y={py(optFull.curve[done - 1]) + 14}
                  textAnchor="end"
                  fill="var(--viz-ink-2)"
                  style={{ fontSize: 11 }}
                >
                  OPT {(optFull.curve[done - 1] * 100).toFixed(0)}%
                </text>
              </>
            ) : (
              <text x={cl + 8} y={py(0.5)} fill="var(--viz-ink-muted)" style={{ fontSize: 11 }}>
                step the trace to draw the cumulative hit rate
              </text>
            )}
            <text x={chartW - cr} y={chartH - 6} textAnchor="end" fill="var(--viz-ink-muted)" style={{ fontSize: 10 }}>
              cumulative hit rate over the trace
            </text>
          </svg>
        </div>
      </TooltipHost>
    </VizPanel>
  );
}
