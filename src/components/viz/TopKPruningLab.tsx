import { useMemo, useState, type ReactNode } from 'react';
import { VizPanel, Segmented, Choice, Check, Slider, Button, Legend, Stats, Note, makeRng, fmtNum } from './Viz';

/**
 * Safe top-k pruning over an inverted index, and Fagin's threshold algorithm.
 *
 * Doc-ID-ordered panel — four document-at-a-time scorers over the same postings, following PISA's
 * implementations (include/pisa/query/algorithm/*.hpp):
 *  - Exhaustive OR scores every document in the union of the query terms' postings.
 *  - MaxScore (Turtle & Flood 1995, as in PISA's maxscore_query): terms ordered by max score; the
 *    lowest-bound terms whose summed bounds cannot beat the heap become "non-essential" — they are only
 *    probed for documents found through the essential terms, with an early exit.
 *  - WAND (Broder et al. 2003): cursors sorted by current doc ID; the pivot is the first cursor where the
 *    running sum of per-term max scores would enter the heap. Docs before the pivot are skipped.
 *  - Block-max WAND (Ding & Suel 2011): WAND's pivot, then a check against the maxima of the blocks that
 *    contain the pivot doc (shallow moves, no decoding). If the block bound fails, jump past the nearest
 *    block end instead of scoring.
 * The heap admits a score only if it is strictly greater than the current k-th score (PISA's would_enter);
 * until the heap holds k entries the threshold is 0.
 *
 * Model assumptions (labelled in the UI where numbers appear):
 *  - A synthetic collection of 2,000 documents. Scores are Lucene's BM25 (k1 = 1.2, b = 0.75,
 *    idf = ln(1 + (N − n + 0.5)/(n + 0.5)), no (k1+1) factor), computed from real term frequencies and
 *    document lengths. Documents are not length-quantized into one byte as Lucene's norms are.
 *  - Postings are cut into fixed-size blocks; every block stores its maximum score (Lucene stores
 *    (freq, norm) impacts instead and derives the bound at query time — same effect for BM25).
 *  - "Postings decoded" counts every posting in a block the scorer had to decompress: a deep move
 *    (next / nextGEQ) that lands in a block decodes the whole block; a shallow move only reads skip data.
 *  - Production twists: a per-user boost multiplies each document's score by 0.5–3.0, a value the index
 *    never saw; a post-filter drops documents after scoring (or, pushed into the query, lets cursors leap to
 *    the next passing document); exact hit counting keeps the collector from ever raising the threshold, as
 *    Lucene's TopScoreDocCollector does until totalHitsThreshold is passed; deleted documents stay in
 *    the postings and the block maxima until a merge, and are skipped when collected.
 *
 * Score-ordered panel — Fagin, Lotem & Naor's TA and NRA over the same query terms, each list sorted by
 * that term's contribution (missing grade = 0). TA does one random access per list for each distinct
 * object the first time it is seen (the paper re-accesses on every sighting to keep buffers bounded; the
 * lab memoizes). NRA does no random access and halts when no object outside its top k — including unseen
 * objects, bounded by the threshold — can still beat the k-th lower bound.
 */

/* -------------------------------------------------------------- the index */

export const N_DOCS = 2000;
export const K1 = 1.2;
export const B_PARAM = 0.75;

export type Posting = { doc: number; tf: number; s: number };
export type Block = { start: number; end: number; first: number; last: number; max: number };
export type TermList = { term: string; df: number; idf: number; postings: Posting[]; blocks: Block[]; maxScore: number };
export type Index = { terms: TermList[]; total: number; blockSize: number; avgdl: number };

type TermSpec = { term: string; df: number; topical: number; seed: number };
const TERM_SPECS: Record<string, TermSpec> = {
  raft: { term: 'raft', df: 40, topical: 0.9, seed: 11 },
  election: { term: 'election', df: 300, topical: 0.8, seed: 23 },
  leader: { term: 'leader', df: 700, topical: 0.85, seed: 37 },
  the: { term: 'the', df: 1900, topical: 1, seed: 41 },
  index: { term: 'index', df: 600, topical: 0.7, seed: 53 },
};

export const QUERIES = [
  { value: 'raft-leader-election', label: 'raft leader election', terms: ['raft', 'leader', 'election'] },
  { value: 'raft-election', label: 'raft election', terms: ['raft', 'election'] },
  { value: 'the-index', label: 'the index', terms: ['the', 'index'] },
] as const;
export type QueryId = (typeof QUERIES)[number]['value'];

function docLengths() {
  const rng = makeRng(97);
  const dl: number[] = [];
  for (let d = 0; d < N_DOCS; d++) {
    const r = rng();
    dl.push(Math.round(40 + 520 * r * r));
  }
  return dl;
}
const DL = docLengths();
const AVGDL = DL.reduce((a, b) => a + b, 0) / N_DOCS;

/** About one document in 33 (3%) is "about" distributed consensus: query terms co-occur there with higher tf. */
const TOPICAL: boolean[] = (() => {
  const rng = makeRng(5);
  return Array.from({ length: N_DOCS }, () => rng() < 0.03);
})();

export const bm25 = (tf: number, dl: number, idf: number) => (idf * tf) / (tf + K1 * (1 - B_PARAM + (B_PARAM * dl) / AVGDL));

function buildTerm(spec: TermSpec, blockSize: number): TermList {
  const rng = makeRng(spec.seed);
  const chosen = new Set<number>();
  for (let d = 0; d < N_DOCS && chosen.size < spec.df; d++) if (TOPICAL[d] && rng() < spec.topical) chosen.add(d);
  let guard = 0;
  while (chosen.size < spec.df && guard++ < 100_000) chosen.add(Math.floor(rng() * N_DOCS));
  const docs = [...chosen].sort((a, b) => a - b);
  const idf = Math.log(1 + (N_DOCS - docs.length + 0.5) / (docs.length + 0.5));
  const postings = docs.map((doc) => {
    const r = rng();
    const tf = TOPICAL[doc] ? 2 + Math.floor(r * 9) : 1 + Math.floor(-Math.log(1 - r * 0.999) * 0.9);
    return { doc, tf, s: bm25(tf, DL[doc], idf) };
  });
  const blocks: Block[] = [];
  for (let start = 0; start < postings.length; start += blockSize) {
    const end = Math.min(postings.length, start + blockSize);
    let max = 0;
    for (let i = start; i < end; i++) max = Math.max(max, postings[i].s);
    blocks.push({ start, end, first: postings[start].doc, last: postings[end - 1].doc, max });
  }
  return { term: spec.term, df: docs.length, idf, postings, blocks, maxScore: Math.max(...postings.map((p) => p.s)) };
}

export function buildIndex(query: QueryId, blockSize: number): Index {
  const q = QUERIES.find((x) => x.value === query) ?? QUERIES[0];
  const terms = q.terms.map((t) => buildTerm(TERM_SPECS[t], blockSize));
  return { terms, total: terms.reduce((a, t) => a + t.df, 0), blockSize, avgdl: AVGDL };
}

/* ---------------------------------------------------------- twists */

export type Twist = 'none' | 'boost' | 'filter' | 'count' | 'deleted';
export type Algo = 'exhaustive' | 'maxscore' | 'wand' | 'bmw';

export type Scenario = {
  k: number;
  twist: Twist;
  /** boost: keep pruning with the stored (now stale) bounds instead of falling back to a full scan */
  pruneStale: boolean;
  /** filter: fraction of documents that pass */
  selectivity: number;
  /** filter: pushed into the query as a required clause instead of a post-filter */
  filterAsClause: boolean;
  /** deleted: fraction of documents deleted but not yet merged away */
  deletedFraction: number;
};

export const DEFAULT_SCENARIO: Scenario = { k: 10, twist: 'none', pruneStale: false, selectivity: 0.03, filterAsClause: false, deletedFraction: 0.3 };

const perDoc = (seed: number) => {
  const rng = makeRng(seed);
  return Array.from({ length: N_DOCS }, () => rng());
};
const BOOST_R = perDoc(71);
const FILTER_R = perDoc(83);
const DELETE_R = perDoc(89);
export const boostOf = (d: number) => 0.5 + 2.5 * BOOST_R[d];

function twistFns(sc: Scenario) {
  const boost = sc.twist === 'boost' ? boostOf : () => 1;
  const passes = sc.twist === 'filter' ? (d: number) => FILTER_R[d] < sc.selectivity : () => true;
  const deleted = sc.twist === 'deleted' ? (d: number) => DELETE_R[d] < sc.deletedFraction : () => false;
  const clause = sc.twist === 'filter' && sc.filterAsClause;
  const nextPass = (d: number) => {
    let x = d;
    while (x < N_DOCS && !passes(x)) x++;
    return x;
  };
  /** An honest engine cannot bound a score it cannot see: the stored maxima become +infinity. */
  const boundsUsable = !(sc.twist === 'boost' && !sc.pruneStale);
  const countUpTo = sc.twist === 'count' ? Infinity : 0;
  return { boost, passes, deleted, clause, nextPass, boundsUsable, countUpTo };
}

/* ---------------------------------------------------------- top-k heap */

class TopK {
  items: { doc: number; score: number }[] = [];
  constructor(public k: number) {}
  get full() {
    return this.items.length >= this.k;
  }
  theta() {
    return this.full ? this.items[this.items.length - 1].score : 0;
  }
  insert(doc: number, score: number) {
    if (!(score > this.theta())) return false;
    this.items.push({ doc, score });
    this.items.sort((a, b) => b.score - a.score || a.doc - b.doc);
    if (this.items.length > this.k) this.items.pop();
    return true;
  }
}

/* ---------------------------------------------------------- trace */

export type EventKind = 'start' | 'score' | 'advance' | 'block-skip' | 'filter-leap' | 'deleted' | 'filtered' | 'partial' | 'stop';
export type TraceEvent = {
  kind: EventKind;
  text: string;
  doc: number;
  cursors: number[];
  theta: number;
  heap: { doc: number; score: number }[];
  bound?: number;
  score?: number;
  entered?: boolean;
  essential?: number;
  decoded: number;
  blocksDecoded: number;
  scored: number;
  evaluated: number;
};

export type RunResult = {
  algo: Algo;
  events: TraceEvent[];
  /** event index at which each block was first decoded, or -1 */
  decodedAt: number[][];
  /** event index at which each posting's score was computed, or -1 */
  scoredAt: number[][];
  top: { doc: number; score: number }[];
  decoded: number;
  blocksDecoded: number;
  blocksTotal: number;
  scored: number;
  evaluated: number;
  total: number;
  missed: number[];
  truth: { doc: number; score: number }[];
};

const fmt = (x: number) => (x === Infinity ? '∞' : x.toFixed(2));
/** Format a > b with enough decimals that the two printed values differ (2 decimals can print '5.53 > 5.53'). */
const fmtGt = (a: number, b: number): [string, string] => {
  for (let dp = 2; dp < 6; dp++) if (fmt(a) === '∞' || a.toFixed(dp) !== b.toFixed(dp)) return [dp === 2 ? fmt(a) : a.toFixed(dp), b.toFixed(dp)];
  return [a.toFixed(6), b.toFixed(6)];
};

export function run(index: Index, algo: Algo, sc: Scenario): RunResult {
  const tw = twistFns(sc);
  const m = index.terms.length;
  const B = index.blockSize;
  const heap = new TopK(sc.k);
  const events: TraceEvent[] = [];
  const decodedAt = index.terms.map((t) => t.blocks.map(() => -1));
  const scoredAt = index.terms.map((t) => t.postings.map(() => -1));
  let decoded = 0;
  let blocksDecoded = 0;
  let scored = 0;
  let evaluated = 0;
  let hits = 0;

  type Cur = { t: number; idx: number; blk: number };
  const cursors: Cur[] = index.terms.map((_, t) => ({ t, idx: 0, blk: 0 }));
  const docOf = (c: Cur) => (c.idx < index.terms[c.t].postings.length ? index.terms[c.t].postings[c.idx].doc : N_DOCS);
  const decode = (c: Cur) => {
    const tl = index.terms[c.t];
    if (c.idx >= tl.postings.length) return;
    const b = Math.floor(c.idx / B);
    if (b > c.blk) c.blk = b;
    if (decodedAt[c.t][b] < 0) {
      decodedAt[c.t][b] = events.length;
      decoded += tl.blocks[b].end - tl.blocks[b].start;
      blocksDecoded++;
    }
  };
  const nextGEQ = (c: Cur, target: number) => {
    const tl = index.terms[c.t];
    if (docOf(c) >= target) return;
    while (c.idx < tl.postings.length && tl.postings[c.idx].doc < target) c.idx++;
    decode(c);
  };
  const next = (c: Cur) => nextGEQ(c, docOf(c) + 1);
  const nextShallow = (c: Cur, target: number) => {
    const tl = index.terms[c.t];
    while (c.blk < tl.blocks.length && tl.blocks[c.blk].last < target) c.blk++;
  };
  const blockLast = (c: Cur) => (c.blk < index.terms[c.t].blocks.length ? index.terms[c.t].blocks[c.blk].last : N_DOCS - 1);
  const blockMax = (c: Cur) => (!tw.boundsUsable ? Infinity : c.blk < index.terms[c.t].blocks.length ? index.terms[c.t].blocks[c.blk].max : 0);
  const termMax = (c: Cur) => (tw.boundsUsable ? index.terms[c.t].maxScore : Infinity);
  const scoreOf = (c: Cur) => {
    scoredAt[c.t][c.idx] = events.length;
    scored++;
    return index.terms[c.t].postings[c.idx].s;
  };
  /** The threshold the scorer may prune against: the collector withholds it while it still has to count hits. */
  const pruneTheta = () => (hits > tw.countUpTo ? heap.theta() : 0);
  const wouldEnter = (x: number) => x > pruneTheta();

  const push = (kind: EventKind, text: string, extra: Partial<TraceEvent> = {}) => {
    events.push({
      kind,
      text,
      doc: -1,
      cursors: cursors.map((c) => docOf(c)),
      theta: heap.theta(),
      heap: heap.items.map((x) => ({ ...x })),
      decoded,
      blocksDecoded,
      scored,
      evaluated,
      ...extra,
    });
  };

  for (const c of cursors) decode(c);
  push('start', `Open a cursor on each of the ${m} posting lists and decode its first block.`);

  /** Collect a fully scored candidate: deleted and post-filtered documents never reach the heap. */
  const collect = (d: number, raw: number, extraText = '') => {
    evaluated++;
    const score = raw * tw.boost(d);
    if (!tw.passes(d)) {
      push('filtered', `Doc ${d} scored ${fmt(score)} but fails the post-filter: dropped, the threshold does not move.${extraText}`, { doc: d, score });
      return;
    }
    hits++;
    const entered = heap.insert(d, score);
    push(
      'score',
      entered
        ? `Score doc ${d}: ${fmt(score)} enters the top ${sc.k}${heap.full ? `; threshold θ is now ${fmt(heap.theta())}` : ''}.${extraText}`
        : `Score doc ${d}: ${fmt(score)} does not beat θ = ${fmt(heap.theta())}.${extraText}`,
      { doc: d, score, entered },
    );
  };

  const leapFilter = (d: number, movable: Cur[]) => {
    const nf = tw.nextPass(d);
    for (const c of movable) if (docOf(c) < nf) nextGEQ(c, nf);
    push('filter-leap', `Doc ${d} fails the required filter clause: every cursor still before doc ${nf >= N_DOCS ? 'the end' : nf} leaps there.`, { doc: d });
  };

  let guard = 0;
  const LIMIT = 6000;

  if (algo === 'exhaustive') {
    while (guard++ < LIMIT) {
      const d = Math.min(...cursors.map(docOf));
      if (d >= N_DOCS) break;
      if (tw.clause && !tw.passes(d)) {
        leapFilter(d, cursors);
        continue;
      }
      const at = cursors.filter((c) => docOf(c) === d);
      if (tw.deleted(d)) {
        at.forEach(next);
        push('deleted', `Doc ${d} is deleted (still in the postings until a merge): skip it.`, { doc: d });
        continue;
      }
      let raw = 0;
      for (const c of at) raw += scoreOf(c);
      at.forEach(next);
      collect(d, raw);
    }
  } else if (algo === 'wand' || algo === 'bmw') {
    while (guard++ < LIMIT) {
      const ord = cursors.slice().sort((a, b) => docOf(a) - docOf(b));
      let acc = 0;
      let p = -1;
      for (let i = 0; i < m; i++) {
        if (docOf(ord[i]) >= N_DOCS) break;
        acc += termMax(ord[i]);
        if (wouldEnter(acc)) {
          p = i;
          break;
        }
      }
      if (p < 0) {
        push('stop', `No prefix of the cursors has max scores summing past θ = ${fmt(pruneTheta())}: nothing left can enter the top ${sc.k}. Stop.`);
        break;
      }
      const d = docOf(ord[p]);
      const pivotTerm = index.terms[ord[p].t].term;
      if (algo === 'bmw') while (p + 1 < m && docOf(ord[p + 1]) === d) p++;
      if (tw.clause && !tw.passes(d)) {
        leapFilter(d, ord.slice(0, p + 1));
        continue;
      }
      let blockUB = acc;
      if (algo === 'bmw') {
        blockUB = 0;
        for (let i = 0; i <= p; i++) {
          if (blockLast(ord[i]) < d) nextShallow(ord[i], d);
          blockUB += blockMax(ord[i]);
        }
        if (!wouldEnter(blockUB)) {
          let nextList = p;
          let maxW = termMax(ord[p]);
          for (let i = 0; i < p; i++) if (termMax(ord[i]) > maxW) (nextList = i), (maxW = termMax(ord[i]));
          let nxt = N_DOCS;
          for (let i = 0; i <= p; i++) nxt = Math.min(nxt, blockLast(ord[i]));
          nxt += 1;
          if (p + 1 < m && docOf(ord[p + 1]) < nxt) nxt = docOf(ord[p + 1]);
          if (nxt <= d) nxt = d + 1;
          nextGEQ(ord[nextList], nxt);
          push(
            'block-skip',
            `Pivot doc ${d}: global bounds pass, but the blocks holding it max out at ${fmt(blockUB)} ≤ θ = ${fmt(pruneTheta())}. Jump '${index.terms[ord[nextList].t].term}' to doc ${nxt >= N_DOCS ? 'the end' : nxt} without decoding.`,
            { doc: d, bound: blockUB },
          );
          continue;
        }
      }
      if (docOf(ord[0]) === d) {
        const at = ord.filter((c) => docOf(c) === d);
        if (tw.deleted(d)) {
          at.forEach(next);
          push('deleted', `Pivot doc ${d} is deleted (its postings and block maxima remain until a merge): skip it.`, { doc: d, bound: blockUB });
          continue;
        }
        let raw = 0;
        let ub = blockUB;
        let partial = false;
        for (const c of at) {
          const s = scoreOf(c);
          raw += s;
          if (algo === 'bmw' && tw.boundsUsable) {
            ub -= blockMax(c) - s;
            if (!wouldEnter(ub) && c !== at[at.length - 1]) {
              partial = true;
              break;
            }
          }
        }
        at.forEach(next);
        if (partial) {
          evaluated++;
          push('partial', `Doc ${d}: after scoring some of its terms the remaining block bound ${fmt(ub)} cannot beat θ. Stop scoring it.`, { doc: d, bound: blockUB, score: raw });
          continue;
        }
        // WAND's pivot prefix may stop before later cursors on the same doc; report the bound over every list holding it.
        const docBound = algo === 'bmw' ? blockUB : at.reduce((a, c) => a + termMax(c), 0);
        collect(d, raw, ` (upper bound ${fmt(docBound)})`);
        events[events.length - 1].bound = docBound;
      } else {
        let nl = p;
        while (nl > 0 && docOf(ord[nl]) === d) nl--;
        nextGEQ(ord[nl], d);
        const [accS, thetaS] = fmtGt(acc, pruneTheta());
        push('advance', `Pivot is doc ${d}: max scores up to '${pivotTerm}' sum to ${accS} > θ = ${thetaS}. Docs before it cannot enter, so advance '${index.terms[ord[nl].t].term}' to doc ${d}.`, {
          doc: d,
          bound: acc,
        });
      }
    }
  } else {
    // MaxScore, PISA's maxscore_query
    const ord = cursors.slice().sort((a, b) => termMax(b) - termMax(a) || a.t - b.t);
    const ub: number[] = new Array(m).fill(0);
    {
      let acc = 0;
      for (let i = m - 1; i >= 0; i--) {
        acc += termMax(ord[i]);
        ub[i] = acc;
      }
    }
    let firstLookup = m;
    const update = () => {
      while (firstLookup > 0 && !wouldEnter(ub[firstLookup - 1])) {
        firstLookup--;
        if (firstLookup === 0) return false;
      }
      return true;
    };
    const essentialNames = () => ord.slice(0, firstLookup).map((c) => `'${index.terms[c.t].term}'`).join(', ');
    let nextDoc = Math.min(...cursors.map(docOf));
    let stopped = false;
    while (guard++ < LIMIT && !stopped) {
      if (nextDoc >= N_DOCS) break;
      const d = nextDoc;
      if (tw.clause && !tw.passes(d)) {
        leapFilter(d, ord.slice(0, firstLookup));
        nextDoc = Math.min(...ord.slice(0, firstLookup).map(docOf));
        continue;
      }
      nextDoc = N_DOCS;
      const del = tw.deleted(d);
      let raw = 0;
      for (let i = 0; i < firstLookup; i++) {
        const c = ord[i];
        if (docOf(c) === d) {
          if (!del) raw += scoreOf(c);
          next(c);
        }
        nextDoc = Math.min(nextDoc, docOf(c));
      }
      if (del) {
        push('deleted', `Doc ${d} is deleted (still in the postings until a merge): skip it.`, { doc: d, essential: firstLookup });
        continue;
      }
      let insert = true;
      let failedBound = 0;
      for (let i = firstLookup; i < m; i++) {
        if (!wouldEnter(raw + ub[i])) {
          insert = false;
          failedBound = raw + ub[i];
          break;
        }
        nextGEQ(ord[i], d);
        if (docOf(ord[i]) === d) raw += scoreOf(ord[i]);
      }
      if (!insert) {
        evaluated++;
        push('partial', `Doc ${d} (found through ${essentialNames()}): its score so far plus the non-essential terms' bounds, ${fmt(failedBound)}, cannot beat θ = ${fmt(pruneTheta())}. Skip the lookups.`, {
          doc: d,
          score: raw,
          essential: firstLookup,
        });
        continue;
      }
      const before = firstLookup;
      collect(d, raw);
      events[events.length - 1].essential = firstLookup;
      if (!update()) {
        push('stop', `Even the sum of every term's max score cannot beat θ = ${fmt(pruneTheta())}. Stop.`, { essential: 0 });
        stopped = true;
        break;
      }
      if (firstLookup < before) {
        const e = events[events.length - 1];
        e.text += ` Terms with the smallest max scores are now non-essential: only ${essentialNames()} drive${firstLookup === 1 ? 's' : ''} the iteration.`;
        e.essential = firstLookup;
      }
    }
  }
  if (events[events.length - 1].kind !== 'stop') push('stop', `Every cursor is exhausted. The top ${sc.k} is final.`);

  const truth = trueTopK(index, sc);
  const got = new Set(heap.items.map((x) => x.doc));
  const missed = truth.filter((x) => !got.has(x.doc)).map((x) => x.doc);
  return {
    algo,
    events,
    decodedAt,
    scoredAt,
    top: heap.items,
    decoded,
    blocksDecoded,
    blocksTotal: index.terms.reduce((a, t) => a + t.blocks.length, 0),
    scored,
    evaluated,
    total: index.total,
    missed,
    truth,
  };
}

/** The answer an exhaustive evaluation returns under the same scoring, filter and deletions. */
export function trueTopK(index: Index, sc: Scenario) {
  const tw = twistFns(sc);
  const acc = new Map<number, number>();
  for (const t of index.terms) for (const p of t.postings) acc.set(p.doc, (acc.get(p.doc) ?? 0) + p.s);
  const all = [...acc.entries()]
    .filter(([d]) => tw.passes(d) && !tw.deleted(d))
    .map(([doc, s]) => ({ doc, score: s * tw.boost(doc) }))
    .sort((a, b) => b.score - a.score || a.doc - b.doc);
  return all.slice(0, sc.k);
}

/* ------------------------------------------------ Fagin: TA and NRA */

export type Agg = 'sum' | 'min' | 'boost';
export type FaginAlgo = 'ta' | 'nra';
export type FaginStep = {
  depth: number;
  bottoms: number[];
  tau: number;
  seen: number;
  sorted: number;
  random: number;
  top: { doc: number; grade: number; lo: number; hi: number }[];
  kth: number;
  bestOutside: number;
  halted: boolean;
};
export type FaginResult = { lists: Posting[][]; steps: FaginStep[]; haltDepth: number; universe: number; truth: number[]; missed: number[]; maxLen: number };

export function fagin(index: Index, algo: FaginAlgo, agg: Agg, k: number): FaginResult {
  const m = index.terms.length;
  const lists = index.terms.map((t) => t.postings.slice().sort((a, b) => b.s - a.s || a.doc - b.doc));
  const grade = index.terms.map((t) => new Map(t.postings.map((p) => [p.doc, p.s])));
  const universe = new Set(index.terms.flatMap((t) => t.postings.map((p) => p.doc)));
  const combine = (xs: number[], doc: number) => {
    if (agg === 'min') return Math.min(...xs);
    const s = xs.reduce((a, b) => a + b, 0);
    return agg === 'boost' ? s * boostOf(doc) : s;
  };
  /** The bound on an unseen object is computed from the lists alone: the lists know nothing about a per-user boost. */
  const listBound = (xs: number[]) => (agg === 'min' ? Math.min(...xs) : xs.reduce((a, b) => a + b, 0));
  const exact = [...universe].map((doc) => ({ doc, g: combine(index.terms.map((_, i) => grade[i].get(doc) ?? 0), doc) })).sort((a, b) => b.g - a.g || a.doc - b.doc);
  const truth = exact.slice(0, k).map((x) => x.doc);
  const maxLen = Math.max(...lists.map((l) => l.length));

  const steps: FaginStep[] = [];
  const seenFields = new Map<number, Map<number, number>>();
  const graded = new Map<number, number>();
  let sorted = 0;
  let random = 0;
  let haltDepth = maxLen;
  const bottoms = new Array(m).fill(0).map((_, i) => lists[i][0]?.s ?? 0);
  steps.push({ depth: 0, bottoms: bottoms.slice(), tau: listBound(bottoms), seen: 0, sorted: 0, random: 0, top: [], kth: 0, bestOutside: listBound(bottoms), halted: false });

  for (let depth = 1; depth <= maxLen; depth++) {
    for (let i = 0; i < m; i++) {
      const e = lists[i][depth - 1];
      if (!e) {
        bottoms[i] = 0;
        continue;
      }
      sorted++;
      bottoms[i] = e.s;
      if (!seenFields.has(e.doc)) seenFields.set(e.doc, new Map());
      seenFields.get(e.doc)!.set(i, e.s);
      if (algo === 'ta' && !graded.has(e.doc)) {
        random += m - 1;
        graded.set(e.doc, combine(index.terms.map((_, j) => grade[j].get(e.doc) ?? 0), e.doc));
      }
    }
    const tau = listBound(bottoms);
    let top: FaginStep['top'] = [];
    let kth = 0;
    let bestOutside = 0;
    let halted = false;
    if (algo === 'ta') {
      top = [...graded.entries()]
        .map(([doc, g]) => ({ doc, grade: g, lo: g, hi: g }))
        .sort((a, b) => b.grade - a.grade || a.doc - b.doc)
        .slice(0, k);
      kth = top.length >= k ? top[k - 1].grade : 0;
      bestOutside = tau;
      halted = top.length >= k && kth >= tau;
    } else {
      const rows = [...seenFields.entries()].map(([doc, f]) => {
        const lo = combine(index.terms.map((_, j) => f.get(j) ?? 0), doc);
        const hi = combine(index.terms.map((_, j) => f.get(j) ?? bottoms[j]), doc);
        return { doc, grade: lo, lo, hi };
      });
      rows.sort((a, b) => b.lo - a.lo || b.hi - a.hi || a.doc - b.doc);
      top = rows.slice(0, k);
      kth = top.length >= k ? top[k - 1].lo : 0;
      const outside = rows.slice(k).map((r) => r.hi);
      if (seenFields.size < universe.size) outside.push(tau);
      bestOutside = outside.length ? Math.max(...outside) : 0;
      halted = top.length >= k && bestOutside <= kth;
    }
    steps.push({ depth, bottoms: bottoms.slice(), tau, seen: seenFields.size, sorted, random, top, kth, bestOutside, halted });
    if (halted) {
      haltDepth = depth;
      break;
    }
  }
  const final = steps[steps.length - 1];
  const got = new Set(final.top.map((t) => t.doc));
  const exactOf = new Map(exact.map((x) => [x.doc, x.g]));
  // Ties are not mistakes: a result is wrong only if it left out a document that scores strictly higher than one it kept.
  const worstKept = Math.min(...final.top.map((t) => exactOf.get(t.doc) ?? 0));
  const missed = truth.filter((d) => !got.has(d) && (exactOf.get(d) ?? 0) > worstKept + 1e-9);
  return { lists, steps, haltDepth, universe: universe.size, truth, missed, maxLen };
}

/* ------------------------------------------------------------------ UI */

const ALGO_OPTIONS: { value: Algo; label: string }[] = [
  { value: 'exhaustive', label: 'Exhaustive OR (score everything)' },
  { value: 'maxscore', label: 'MaxScore' },
  { value: 'wand', label: 'WAND' },
  { value: 'bmw', label: 'Block-max WAND' },
];
const ALGO_NAME: Record<Algo, string> = { exhaustive: 'Exhaustive OR', maxscore: 'MaxScore', wand: 'WAND', bmw: 'Block-max WAND' };

const TWIST_OPTIONS: { value: Twist; label: string }[] = [
  { value: 'none', label: 'None: plain BM25' },
  { value: 'boost', label: 'Per-user boost the index never saw' },
  { value: 'filter', label: 'Selective filter' },
  { value: 'count', label: 'Exact total hit count' },
  { value: 'deleted', label: 'Deleted docs not yet merged away' },
];

const W = 680;
const LEFT = 118;
const RIGHT = 12;
const LANE_H = 58;
const LANE_GAP = 12;
const TOP = 26;

const pct = (a: number, b: number) => (b > 0 ? `${fmtNum((100 * a) / b)}%` : '—');

export default function TopKPruningLab() {
  const [panel, setPanel] = useState<'daat' | 'fagin'>('daat');
  const [query, setQuery] = useState<QueryId>('raft-leader-election');
  const [k, setK] = useState(10);

  // doc-ID-ordered panel
  const [algo, setAlgo] = useState<Algo>('bmw');
  const [blockSize, setBlockSize] = useState(8);
  const [twist, setTwist] = useState<Twist>('none');
  const [pruneStale, setPruneStale] = useState(false);
  const [selPct, setSelPct] = useState(3);
  const [asClause, setAsClause] = useState(false);
  const [delPct, setDelPct] = useState(30);
  const [stepState, setStepState] = useState<{ key: string; i: number } | null>(null);

  // score-ordered panel
  const [fAlgo, setFAlgo] = useState<FaginAlgo>('ta');
  const [agg, setAgg] = useState<Agg>('sum');
  const [fStepState, setFStepState] = useState<{ key: string; i: number } | null>(null);

  const scenario: Scenario = useMemo(
    () => ({ k, twist, pruneStale, selectivity: selPct / 100, filterAsClause: asClause, deletedFraction: delPct / 100 }),
    [k, twist, pruneStale, selPct, asClause, delPct],
  );
  const index = useMemo(() => buildIndex(query, blockSize), [query, blockSize]);
  const all = useMemo(() => {
    const out = {} as Record<Algo, RunResult>;
    for (const a of ['exhaustive', 'maxscore', 'wand', 'bmw'] as Algo[]) out[a] = run(index, a, scenario);
    return out;
  }, [index, scenario]);
  const r = all[algo];
  const unionSize = useMemo(() => new Set(index.terms.flatMap((t) => t.postings.map((p) => p.doc))).size, [index]);

  const key = `${query}|${blockSize}|${algo}|${JSON.stringify(scenario)}`;
  const last = r.events.length - 1;
  const cur = stepState && stepState.key === key ? Math.max(0, Math.min(last, stepState.i)) : last;
  const setCur = (i: number) => setStepState({ key, i: Math.max(0, Math.min(last, i)) });
  const ev = r.events[cur];
  const done = cur === last;

  const effAgg: Agg = fAlgo === 'nra' && agg === 'boost' ? 'sum' : agg;
  const f = useMemo(() => fagin(index, fAlgo, effAgg, k), [index, fAlgo, effAgg, k]);
  const fKey = `${query}|${fAlgo}|${effAgg}|${k}`;
  const fLast = f.steps.length - 1;
  const fCur = fStepState && fStepState.key === fKey ? Math.max(0, Math.min(fLast, fStepState.i)) : fLast;
  const setFCur = (i: number) => setFStepState({ key: fKey, i: Math.max(0, Math.min(fLast, i)) });
  const fs = f.steps[fCur];

  const controls = (
    <>
      <Segmented
        label="View"
        value={panel}
        onChange={setPanel}
        options={[
          { value: 'daat', label: 'Doc-ID order' },
          { value: 'fagin', label: 'Score order (Fagin)' },
        ]}
      />
      <Choice label="Query" value={query} onChange={setQuery} options={QUERIES.map((q) => ({ value: q.value, label: q.label }))} />
      <Slider label="k" min={1} max={50} value={k} onChange={setK} />
      {panel === 'daat' ? (
        <>
          <Choice label="Scorer" value={algo} onChange={setAlgo} options={ALGO_OPTIONS} />
          <Segmented
            label="Postings per block"
            value={String(blockSize) as '4' | '8' | '16' | '32'}
            onChange={(v) => setBlockSize(Number(v))}
            options={[
              { value: '4', label: '4' },
              { value: '8', label: '8' },
              { value: '16', label: '16' },
              { value: '32', label: '32' },
            ]}
          />
          <Choice label="Production twist" value={twist} onChange={setTwist} options={TWIST_OPTIONS} />
          {twist === 'boost' ? <Check label="Keep pruning with the stored bounds" checked={pruneStale} onChange={setPruneStale} /> : null}
          {twist === 'filter' ? (
            <>
              <Slider label="Docs passing the filter" min={1} max={100} value={selPct} onChange={setSelPct} format={(v) => `${v}%`} />
              <Check label="Push the filter into the query" checked={asClause} onChange={setAsClause} />
            </>
          ) : null}
          {twist === 'deleted' ? <Slider label="Deleted, not yet merged" min={0} max={90} step={5} value={delPct} onChange={setDelPct} format={(v) => `${v}%`} /> : null}
        </>
      ) : (
        <>
          <Segmented
            label="Algorithm"
            value={fAlgo}
            onChange={setFAlgo}
            options={[
              { value: 'ta', label: 'TA' },
              { value: 'nra', label: 'NRA' },
            ]}
          />
          <Choice
            label="Aggregation"
            value={effAgg}
            onChange={setAgg}
            options={
              fAlgo === 'ta'
                ? [
                    { value: 'sum', label: 'sum (BM25)' },
                    { value: 'min', label: 'min (fuzzy AND)' },
                    { value: 'boost', label: 'sum × per-user boost' },
                  ]
                : [
                    { value: 'sum', label: 'sum (BM25)' },
                    { value: 'min', label: 'min (fuzzy AND)' },
                  ]
            }
          />
        </>
      )}
    </>
  );

  if (panel === 'fagin') return <FaginView f={f} fs={fs} fCur={fCur} fLast={fLast} setFCur={setFCur} index={index} fAlgo={fAlgo} agg={effAgg} k={k} controls={controls} />;

  /* ---------------------------------------------------------- doc-ID view */
  const m = index.terms.length;
  const laneY = (t: number) => TOP + t * (LANE_H + LANE_GAP);
  const heapY = laneY(m) + 4;
  const H = heapY + 40;
  const X = (d: number) => LEFT + (d / N_DOCS) * (W - LEFT - RIGHT);
  const barH = (s: number, t: number) => Math.max(1.5, (s / index.terms[t].maxScore) * (LANE_H - 16));

  const blockState = (t: number, b: number) => {
    const at = r.decodedAt[t][b];
    if (at >= 0 && at <= cur) return 'decoded';
    if (index.terms[t].blocks[b].last < ev.cursors[t] || done) return 'skipped';
    return 'ahead';
  };
  let skipped = 0;
  for (let t = 0; t < m; t++) for (let b = 0; b < index.terms[t].blocks.length; b++) if (blockState(t, b) === 'skipped') skipped++;

  const heapDocs = new Set(ev.heap.map((h) => h.doc));
  const showMissed = done && r.missed.length > 0;
  const scale = Math.max(
    1e-9,
    ...r.events.map((e) => Math.max(e.heap[0]?.score ?? 0, Number.isFinite(e.bound ?? NaN) ? (e.bound as number) : 0, e.score ?? 0)),
    ...r.truth.map((x) => x.score),
  );

  const twistNote = (() => {
    if (twist === 'boost')
      return pruneStale
        ? `The stored block and term maxima describe BM25 alone, but every score is multiplied by up to 3×. Pruning against them discards documents that would have won: ${r.missed.length} of the true top ${k} ${done ? 'are' : 'will be'} missing, and nothing reports an error.`
        : 'The engine cannot bound a score it cannot see, so every stored maximum becomes +∞: no pivot is ever skipped and every posting is decoded — the same work as Exhaustive OR.';
    if (twist === 'filter')
      return asClause
        ? `As a required clause the filter drives iteration: cursors leap to the next passing document, so only ${fmtNum(r.scored)} postings are scored.`
        : `Only ${selPct}% of documents survive the post-filter, and only survivors enter the heap. ${r.top.length < k ? `The heap never fills (${r.top.length} of ${k}), so the threshold stays at 0 and nothing can be skipped.` : `The threshold settles at ${ev.theta.toFixed(2)}, lower than the unfiltered one, so fewer blocks can be skipped.`}`;
    if (twist === 'count') return 'The collector must count every match, so it never hands the scorer a minimum competitive score: the threshold the scorer prunes against stays 0 and every posting is decoded.';
    if (twist === 'deleted')
      return `${delPct}% of documents are deleted but still sit in the postings and in the block maxima computed before the deletes. Bounds stay valid but loose, and deleted winners never raise the threshold — results stay correct while the work grows.`;
    return '';
  })();

  return (
    <VizPanel
      title="A top-k scorer with a live threshold"
      subtitle="The query's posting lists laid out by doc ID. Each rectangle is a block of postings, as tall as its stored maximum BM25 contribution relative to the list's own maximum. The scorer keeps the best k scores in a heap and skips anything whose upper bound cannot beat the k-th."
      controls={controls}
      legend={
        <Legend
          items={[
            { label: 'Block decoded', color: 'var(--viz-1)' },
            { label: 'Block skipped without decoding', color: 'var(--viz-stale)' },
            { label: 'Block not reached yet', color: 'var(--viz-axis)' },
            { label: 'Posting scored', color: 'var(--viz-2)' },
            { label: 'Upper bound of the candidate', color: 'var(--viz-3)' },
            { label: 'Current candidate doc', color: 'var(--viz-7)' },
            { label: 'In the top-k heap', color: 'var(--viz-good)' },
            ...(showMissed ? [{ label: 'True top-k doc the scorer lost', color: 'var(--viz-critical)' }] : []),
          ]}
        />
      }
      stats={
        <Stats
          items={[
            { label: 'Postings decoded', value: `${fmtNum(ev.decoded)} / ${fmtNum(r.total)}`, hint: 'Every posting in a block the scorer had to decompress. Skipping a block reads only its skip entry.' },
            { label: 'Blocks skipped', value: `${fmtNum(skipped)} / ${fmtNum(r.blocksTotal)}` },
            { label: 'Docs scored', value: `${fmtNum(ev.evaluated)} / ${fmtNum(unionSize)}`, hint: 'Documents whose score was computed (or partially computed and abandoned), out of every document containing a query term.' },
            { label: 'Threshold θ (k-th score)', value: ev.heap.length >= k ? ev.theta.toFixed(2) : `0 (heap ${ev.heap.length}/${k})` },
            { label: 'Correct top-k', value: done ? `${Math.min(k, r.truth.length) - r.missed.length} / ${Math.min(k, r.truth.length)}` : '…', hint: 'Compared with an exhaustive evaluation under the same scoring, filter and deletions.' },
          ]}
        />
      }
      note={
        <Note>
          <strong>
            {ALGO_NAME[algo]}
            {done ? ' finished' : ` at event ${cur + 1} of ${last + 1}`}: {fmtNum(ev.decoded)} of {fmtNum(r.total)} postings decoded ({pct(ev.decoded, r.total)}), {fmtNum(ev.evaluated)} of {fmtNum(unionSize)} documents scored.
          </strong>{' '}
          {twistNote ||
            (algo === 'exhaustive'
              ? 'Every document that contains any query term is scored, whatever the threshold says.'
              : algo === 'wand'
                ? 'WAND skips with one bound per term. When every list has a document near the pivot, those bounds are too loose to skip much.'
                : algo === 'maxscore'
                  ? 'MaxScore stops iterating the lists whose combined maxima cannot beat θ and only probes them for documents the other lists produce.'
                  : 'Block-max WAND confirms each pivot against the maxima of the blocks that hold it, and jumps past a block boundary when they cannot beat θ.')}
          {' '}
          <span style={{ color: 'var(--viz-ink-2)' }}>Model: 2,000 synthetic documents scored with Lucene's BM25 (k1 = 1.2, b = 0.75). Skip rates are illustrative, not benchmarks.</span>
        </Note>
      }
      table={
        <table className="viz-table">
          <thead>
            <tr>
              <th>Scorer</th>
              <th>Postings decoded</th>
              <th>Blocks decoded</th>
              <th>Docs scored</th>
              <th>Correct top-k</th>
            </tr>
          </thead>
          <tbody>
            {(['exhaustive', 'maxscore', 'wand', 'bmw'] as Algo[]).map((a) => (
              <tr key={a}>
                <td>{ALGO_NAME[a]}</td>
                <td>
                  {fmtNum(all[a].decoded)} ({pct(all[a].decoded, all[a].total)})
                </td>
                <td>
                  {fmtNum(all[a].blocksDecoded)} / {fmtNum(all[a].blocksTotal)} (
                  {index.terms.map((tl, t) => `${tl.term} ${fmtNum(all[a].decodedAt[t].filter((x) => x >= 0).length)}/${fmtNum(tl.blocks.length)}`).join(', ')})
                </td>
                <td>{fmtNum(all[a].evaluated)}</td>
                <td>
                  {Math.min(k, all[a].truth.length) - all[a].missed.length} / {Math.min(k, all[a].truth.length)}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      }
    >
      <div className="viz-controls">
        <Button onClick={() => setCur(0)} disabled={cur === 0}>
          Restart
        </Button>
        <Button primary onClick={() => setCur(cur + 1)} disabled={done}>
          Step
        </Button>
        <Button onClick={() => setCur(cur + 25)} disabled={done}>
          +25
        </Button>
        <Button onClick={() => setCur(last)} disabled={done}>
          Run to end
        </Button>
        <Slider label="Event" min={0} max={last} value={cur} onChange={setCur} format={(v) => `${v + 1} / ${last + 1}`} />
      </div>
      <p style={{ margin: '0 0 0.4rem', fontSize: '0.8rem', color: 'var(--viz-ink)', minHeight: '2.4em' }}>{ev.text}</p>
      <svg viewBox={`0 0 ${W} ${H}`} width={W} height={H} role="img" aria-label={`${ALGO_NAME[algo]} over ${m} posting lists: ${ev.decoded} of ${r.total} postings decoded, ${skipped} blocks skipped`}>
        {[0, 500, 1000, 1500, 2000].map((d) => (
          <g key={d}>
            <line x1={X(d)} x2={X(d)} y1={TOP - 6} y2={heapY + 22} className="viz-grid-line" />
            <text x={X(d)} y={TOP - 10} textAnchor={d === 0 ? 'start' : d === 2000 ? 'end' : 'middle'} fontSize={10}>
              {d === 0 ? 'doc 0' : fmtNum(d)}
            </text>
          </g>
        ))}
        {index.terms.map((tl, t) => {
          const y = laneY(t);
          const base = y + LANE_H - 8;
          const cd = ev.cursors[t];
          return (
            <g key={tl.term}>
              <text x={4} y={y + 18} fontSize={13} fontWeight={600} fill="var(--viz-ink)">
                {tl.term}
              </text>
              <text x={4} y={y + 33} fontSize={10}>
                df {fmtNum(tl.df)} · idf {tl.idf.toFixed(2)}
              </text>
              <text x={4} y={y + 46} fontSize={10}>
                max score {tl.maxScore.toFixed(2)}
              </text>
              <line x1={LEFT} x2={W - RIGHT} y1={base} y2={base} className="viz-axis-line" />
              {tl.blocks.map((b, bi) => {
                const st = blockState(t, bi);
                const x0 = X(b.first);
                const w = Math.max(1.2, X(b.last + 1) - x0 - 0.6);
                const h = barH(b.max, t);
                return (
                  <rect
                    key={bi}
                    x={x0}
                    y={base - h}
                    width={w}
                    height={h}
                    fill={st === 'decoded' ? 'var(--viz-1)' : st === 'skipped' ? 'var(--viz-stale)' : 'none'}
                    fillOpacity={st === 'decoded' ? 0.3 : st === 'skipped' ? 0.6 : 0}
                    stroke={st === 'decoded' ? 'var(--viz-1)' : st === 'skipped' ? 'none' : 'var(--viz-axis)'}
                    strokeWidth={0.8}
                  />
                );
              })}
              {tl.postings.map((p, pi) => {
                const at = r.scoredAt[t][pi];
                if (at < 0 || at > cur) return null;
                return <line key={pi} x1={X(p.doc)} x2={X(p.doc)} y1={base} y2={base - barH(p.s, t)} stroke="var(--viz-2)" strokeWidth={index.total > 1500 ? 0.7 : 1.1} strokeOpacity={0.85} />;
              })}
              {cd < N_DOCS ? <path d={`M ${X(cd)} ${base + 1} l -4 7 h 8 z`} fill="var(--viz-ink)" /> : null}
            </g>
          );
        })}
        {ev.doc >= 0 ? (
          <g>
            <line x1={X(ev.doc)} x2={X(ev.doc)} y1={TOP - 4} y2={heapY + 20} stroke="var(--viz-7)" strokeWidth={1.4} strokeDasharray="4 3" />
          </g>
        ) : null}
        <text x={4} y={heapY + 18} fontSize={11} fill="var(--viz-ink)">
          top-{k} heap
        </text>
        <line x1={LEFT} x2={W - RIGHT} y1={heapY + 14} y2={heapY + 14} className="viz-axis-line" />
        {ev.heap.map((h) => (
          <path key={h.doc} d={`M ${X(h.doc)} ${heapY + 7} l 5 7 l -5 7 l -5 -7 z`} fill="var(--viz-good)" />
        ))}
        {showMissed
          ? r.missed.map((d) => (
              <path key={`m${d}`} d={`M ${X(d) - 4} ${heapY + 10} l 8 8 M ${X(d) + 4} ${heapY + 10} l -8 8`} stroke="var(--viz-critical)" strokeWidth={2} />
            ))
          : null}
        {heapDocs.size === 0 ? (
          <text x={LEFT + 6} y={heapY + 30} fontSize={10}>
            empty
          </text>
        ) : null}
      </svg>
      <CandidateGauge ev={ev} k={k} scale={scale} />
    </VizPanel>
  );
}

function CandidateGauge({ ev, k, scale }: { ev: TraceEvent; k: number; scale: number }) {
  const GW = W;
  const L = 170;
  const span = GW - L - 90;
  const bw = (v: number) => Math.max(0, Math.min(1, v / scale)) * span;
  const full = ev.heap.length >= k;
  const boundInf = ev.bound === Infinity;
  const rows = [
    { label: 'Candidate upper bound', v: ev.bound, color: 'var(--viz-3)' },
    { label: 'Candidate score', v: ev.score, color: 'var(--viz-2)' },
  ];
  return (
    <svg viewBox={`0 0 ${GW} 70`} width={GW} height={70} role="img" aria-label={`Threshold ${full ? ev.theta.toFixed(2) : 0}; candidate bound ${ev.bound ?? 'none'}; candidate score ${ev.score ?? 'none'}`} style={{ marginTop: 6 }}>
      {rows.map((row, i) => {
        const y = 8 + i * 24;
        const has = row.v !== undefined;
        return (
          <g key={row.label}>
            <text x={4} y={y + 12} fontSize={11} fill="var(--viz-ink)">
              {row.label}
            </text>
            {has ? <rect x={L} y={y} width={boundInf && i === 0 ? span : bw(row.v as number)} height={16} rx={3} fill={row.color} fillOpacity={boundInf && i === 0 ? 0.35 : 1} /> : null}
            <text x={GW - 4} y={y + 12} fontSize={11} textAnchor="end" fill="var(--viz-ink)">
              {has ? (row.v === Infinity ? '∞ (no usable bound)' : (row.v as number).toFixed(2)) : '—'}
            </text>
          </g>
        );
      })}
      {full ? (
        <g>
          <line x1={L + bw(ev.theta)} x2={L + bw(ev.theta)} y1={2} y2={56} stroke="var(--viz-ink)" strokeWidth={1.5} strokeDasharray="5 3" />
          <text x={L + bw(ev.theta)} y={67} textAnchor="middle" fontSize={10} fill="var(--viz-ink)">
            θ {ev.theta.toFixed(2)}
          </text>
        </g>
      ) : (
        <text x={L} y={67} fontSize={10}>
          θ = 0 until the heap holds {k}
        </text>
      )}
    </svg>
  );
}

/* ---------------------------------------------------------- Fagin view */

function FaginView({
  f,
  fs,
  fCur,
  fLast,
  setFCur,
  index,
  fAlgo,
  agg,
  k,
  controls,
}: {
  f: FaginResult;
  fs: FaginStep;
  fCur: number;
  fLast: number;
  setFCur: (i: number) => void;
  index: Index;
  fAlgo: FaginAlgo;
  agg: Agg;
  k: number;
  controls: ReactNode;
}) {
  const m = index.terms.length;
  const D = Math.min(f.maxLen, Math.max(40, Math.ceil(f.haltDepth * 1.25)));
  const laneY = (t: number) => TOP + t * (LANE_H + LANE_GAP);
  const H = laneY(m) + 6;
  const bw = (W - LEFT - RIGHT) / D;
  const gmax = Math.max(...index.terms.map((t) => t.maxScore));
  const barH = (s: number) => Math.max(0.8, (s / gmax) * (LANE_H - 16));
  const ranks = useMemo(() => f.lists.map((l) => new Map(l.map((p, i) => [p.doc, i]))), [f]);
  const done = fCur === fLast;
  const halted = fs.halted;
  const name = fAlgo === 'ta' ? 'TA' : 'NRA';
  const kthLabel = fAlgo === 'ta' ? `k-th best grade` : `k-th best lower bound W`;
  const outLabel = fAlgo === 'ta' ? `threshold τ` : `best upper bound B outside`;
  const scale = Math.max(1e-9, ...f.steps.map((s) => Math.max(s.kth, Number.isFinite(s.bestOutside) ? s.bestOutside : 0, s.tau)));
  const GL = 190;
  const gw = (v: number) => Math.max(0, Math.min(1, v / scale)) * (W - GL - 70);

  return (
    <VizPanel
      title="Top-k over score-sorted lists: Fagin's TA and NRA"
      subtitle="The same query terms, but each list now holds one term's BM25 contributions sorted from highest to lowest. Sorted access reads one row deeper in every list per step; TA also looks up each new document's grade in the other lists (random access)."
      controls={controls}
      legend={
        <Legend
          items={[
            { label: 'Read by sorted access', color: 'var(--viz-1)' },
            { label: 'Not read yet', color: 'var(--viz-axis)' },
            { label: fAlgo === 'ta' ? 'Current top-k document' : 'Current top-k document (hollow: grade in this list still unknown)', color: 'var(--viz-good)' },
            { label: kthLabel, color: 'var(--viz-2)' },
            { label: fAlgo === 'ta' ? 'Threshold τ (bound on every unseen document)' : 'Best upper bound outside the top k', color: 'var(--viz-3)' },
          ]}
        />
      }
      stats={
        <Stats
          items={[
            { label: 'Depth', value: `${fmtNum(fs.depth)}${halted ? ' (halted)' : ''}` },
            { label: 'Sorted accesses', value: fmtNum(fs.sorted) },
            { label: 'Random accesses', value: fmtNum(fs.random), hint: 'One lookup in each other list per distinct document, the first time it is seen.' },
            { label: 'Documents seen', value: `${fmtNum(fs.seen)} / ${fmtNum(f.universe)}` },
            { label: 'Correct top-k', value: done ? `${k - f.missed.length} / ${k}` : '…', hint: 'Compared with grading every document. Ties count as correct.' },
          ]}
        />
      }
      note={
        <Note>
          <strong>
            {name}, depth {fs.depth}: {kthLabel} {fs.kth.toFixed(2)} vs {outLabel} {fs.bestOutside.toFixed(2)}.
          </strong>{' '}
          {halted
            ? fAlgo === 'ta'
              ? `${k} documents have grades at or above τ, the best any unseen document could score, so TA halts after ${fmtNum(fs.sorted)} sorted and ${fmtNum(fs.random)} random accesses, having seen ${fmtNum(fs.seen)} of ${fmtNum(f.universe)} documents.`
              : `No document outside the top ${k}, seen or unseen, can still beat the k-th lower bound, so NRA halts after ${fmtNum(fs.sorted)} sorted accesses and no random ones. It knows the top ${k} without knowing all their grades.`
            : fAlgo === 'ta'
              ? `τ is ${agg === 'min' ? 'the minimum' : 'the sum'} of the last grade read from each list. TA keeps reading until the k-th best grade reaches it.`
              : 'NRA knows only the fields it has read: a lower bound W fills the rest with 0, an upper bound B fills them with each list’s last grade read.'}{' '}
          {done && f.missed.length > 0
            ? `But ${f.missed.length} of the true top ${k} are missing: the per-user boost is not a function of the lists, so τ was never a bound on unseen documents.`
            : ''}
        </Note>
      }
      table={
        <table className="viz-table">
          <thead>
            <tr>
              <th>Rank</th>
              <th>Doc</th>
              {fAlgo === 'ta' ? <th>Grade</th> : <th>W (lower)</th>}
              {fAlgo === 'nra' ? <th>B (upper)</th> : null}
            </tr>
          </thead>
          <tbody>
            {fs.top.map((t, i) => (
              <tr key={t.doc}>
                <td>{i + 1}</td>
                <td>{t.doc}</td>
                <td>{t.lo.toFixed(3)}</td>
                {fAlgo === 'nra' ? <td>{t.hi.toFixed(3)}</td> : null}
              </tr>
            ))}
          </tbody>
        </table>
      }
    >
      <div className="viz-controls">
        <Button onClick={() => setFCur(0)} disabled={fCur === 0}>
          Restart
        </Button>
        <Button primary onClick={() => setFCur(fCur + 1)} disabled={done}>
          Read one row
        </Button>
        <Button onClick={() => setFCur(fCur + 10)} disabled={done}>
          +10
        </Button>
        <Button onClick={() => setFCur(fLast)} disabled={done}>
          Run to halt
        </Button>
        <Slider label="Depth" min={0} max={fLast} value={fCur} onChange={setFCur} />
      </div>
      <svg viewBox={`0 0 ${W} ${H}`} width={W} height={H} role="img" aria-label={`${name} at depth ${fs.depth} over ${m} score-sorted lists`}>
        <text x={LEFT} y={TOP - 10} fontSize={10}>
          rank 1
        </text>
        <text x={W - RIGHT} y={TOP - 10} fontSize={10} textAnchor="end">
          rank {fmtNum(D)} (first {fmtNum(D)} rows shown)
        </text>
        {index.terms.map((tl, t) => {
          const y = laneY(t);
          const base = y + LANE_H - 8;
          const list = f.lists[t];
          let seenPath = '';
          let restPath = '';
          for (let r = 0; r < Math.min(D, list.length); r++) {
            const x = LEFT + r * bw;
            const h = barH(list[r].s);
            const seg = `M${x.toFixed(2)} ${base}V${(base - h).toFixed(2)}H${(x + Math.max(0.6, bw * 0.85)).toFixed(2)}V${base}Z`;
            if (r < fs.depth) seenPath += seg;
            else restPath += seg;
          }
          const bottom = fs.bottoms[t];
          return (
            <g key={tl.term}>
              <text x={4} y={y + 18} fontSize={13} fontWeight={600} fill="var(--viz-ink)">
                {tl.term}
              </text>
              <text x={4} y={y + 33} fontSize={10}>
                {fmtNum(list.length)} postings
              </text>
              <text x={4} y={y + 46} fontSize={10}>
                {fs.depth > list.length ? 'exhausted' : `last read ${fs.depth === 0 ? '—' : bottom.toFixed(2)}`}
              </text>
              <line x1={LEFT} x2={W - RIGHT} y1={base} y2={base} className="viz-axis-line" />
              <path d={restPath} fill="var(--viz-axis)" />
              <path d={seenPath} fill="var(--viz-1)" />
              {list.length < D ? (
                <text x={LEFT + list.length * bw + 4} y={base - 4} fontSize={10}>
                  end of list
                </text>
              ) : null}
              {fs.top.map((td) => {
                const rk = ranks[t].get(td.doc);
                if (rk === undefined || rk >= D) return null;
                const known = fAlgo === 'ta' || rk < fs.depth;
                const x = LEFT + rk * bw + bw / 2;
                return <path key={td.doc} d={`M ${x} ${base + 2} l -4 7 h 8 z`} fill={known ? 'var(--viz-good)' : 'var(--viz-surface)'} stroke="var(--viz-good)" strokeWidth={1.2} />;
              })}
            </g>
          );
        })}
        {(() => {
          const x = LEFT + Math.min(fs.depth, D) * bw;
          return (
            <g>
              <line x1={x} x2={x} y1={TOP - 4} y2={H - 4} stroke="var(--viz-ink)" strokeDasharray="4 3" />
            </g>
          );
        })()}
      </svg>
      <svg viewBox={`0 0 ${W} 60`} width={W} height={60} role="img" aria-label={`${kthLabel} ${fs.kth.toFixed(2)}, ${outLabel} ${fs.bestOutside.toFixed(2)}`} style={{ marginTop: 6 }}>
        {[
          { label: kthLabel, v: fs.kth, color: 'var(--viz-2)' },
          { label: outLabel, v: fs.bestOutside, color: 'var(--viz-3)' },
        ].map((row, i) => {
          const y = 6 + i * 26;
          return (
            <g key={row.label}>
              <text x={4} y={y + 13} fontSize={11} fill="var(--viz-ink)">
                {row.label}
              </text>
              <rect x={GL} y={y} width={gw(row.v)} height={18} rx={3} fill={row.color} />
              <text x={GL + gw(row.v) + 6} y={y + 13} fontSize={11}>
                {row.v.toFixed(2)}
              </text>
            </g>
          );
        })}
      </svg>
    </VizPanel>
  );
}
