import { useDeferredValue, useMemo, useState } from 'react';
import { VizPanel, Segmented, Choice, Check, Slider, Button, Legend, Stats, Note, SERIES, makeRng, fmtNum, useSize } from './Viz';

/**
 * Querying a prebuilt full-text index: term dictionary lookup, then posting-list cursors.
 *
 * What is modelled faithfully:
 * - Postings are sorted doc IDs cut into fixed-size blocks (Lucene: 128 up to 10.3, 256 from 10.4;
 *   Tantivy: 128). A trailing partial block is a VInt block with no skip entry. How a block is encoded
 *   (bit-packed gaps, or since Lucene 10.2 a bit set when that is smaller) is not modelled: counters count postings.
 * - Skip data follows Lucene 9.12+: a level-0 entry (the block's last doc ID) before every packed block and a
 *   level-1 entry before every 32 packed blocks. advance(target) reads entries until it finds the first block
 *   whose last doc is >= target and decodes only that block. nextDoc() entering a new packed block reads that
 *   block's level-0 entry (and a level-1 entry at a group boundary), as Lucene's moveToNextLevel0Block does.
 * - "Leapfrog" is Lucene's ConjunctionDISI.doNext: iterators sorted by cost (docFreq), lead1/lead2/others, and any
 *   iterator that overshoots pulls lead1 forward to its doc.
 * - "Lead drives" is the shape of Postgres GIN's keyGetItem for an AND key: the minimum over the required entries
 *   (the lead) picks each candidate, and every additional entry is advanced to it; nothing pulls the lead forward.
 * - OR / prefix use a doc-ID heap (Lucene's DisjunctionDISIApproximation): every posting of every list is visited.
 * - Phrases check positions with Lucene's ExactPhraseMatcher position leapfrog, or (GIN) recheck the heap row.
 *
 * Model assumptions (labelled in the UI):
 * - One segment of 100,000 documents; postings are sampled independently per term, except that "ahead" is planted
 *   right after "write" in a fixed subset of documents so the phrase has real matches.
 * - Cost counters are counts, not times: postings decoded (whole blocks), skip entries read, positions read.
 * - The dictionary has 64 terms in blocks of 8 (Lucene's BlockTree uses 25-48). The trie indexes every term, as
 *   Tantivy's FST does; Lucene's trie only indexes block prefixes and then scans the block.
 */

export const N_DOCS = 100_000;
export const NO_MORE = 2147483647;
export const DICT_BLOCK = 8;

export const VOCAB = 'abort ahead alter analyze append array backup btree buffer cache checkpoint checksum cluster column commit compact compacted compacting compaction compactions compress cursor delete durable flush fsync heap index insert journal latch ledger lock merge mvcc page partition postgres query raft read recovery replica rollback row scan schema segment snapshot sort split sstable table tombstone trigger tuple undo update vacuum view wal write xmin zone'.split(' ');

export const FSYNC_DFS = [10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000] as const;

const FIXED_DF: Record<string, number> = {
  postgres: 40_000,
  checkpoint: 30_000,
  write: 15_000,
  compact: 3_000,
  compacted: 700,
  compacting: 400,
  compaction: 5_000,
  compactions: 1_100,
};
const AHEAD_RANDOM_DF = 6_000;
const PLANT_RATE = 0.12;

/* ------------------------------------------------------------------ hashing */

function fnv(s: string) {
  let h = 0x811c9dc5;
  for (let i = 0; i < s.length; i++) {
    h ^= s.charCodeAt(i);
    h = Math.imul(h, 0x01000193) >>> 0;
  }
  return h >>> 0;
}
function mix(a: number, b: number) {
  let h = Math.imul((a ^ Math.imul(b + 0x9e3779b9, 0x27d4eb2d)) >>> 0, 0x85ebca6b);
  h ^= h >>> 13;
  h = Math.imul(h, 0xc2b2ae35);
  h ^= h >>> 16;
  return h >>> 0;
}
const unit = (h: number) => h / 4294967296;

/* ------------------------------------------------------------------ corpus */

function sampleDocs(df: number, seed: number): number[] {
  const out: number[] = [];
  if (df <= 0) return out;
  const p = Math.min(0.95, df / N_DOCS);
  const r = makeRng(mix(seed, 0x51ed) | 1);
  r();
  r();
  const lg = Math.log(1 - p);
  let d = -1;
  for (let guard = 0; guard < N_DOCS + 1; guard++) {
    const u = Math.min(r(), 0.999999);
    d += 1 + Math.floor(Math.log(1 - u) / lg);
    if (d >= N_DOCS) break;
    out.push(d);
  }
  return out;
}

const cache = new Map<string, Int32Array>();
let plantedSet: Set<number> | null = null;

function planted() {
  if (!plantedSet) {
    const write = docsFor('write', 0);
    plantedSet = new Set<number>();
    for (const d of write) if (unit(mix(d, 0x0a4ead)) < PLANT_RATE) plantedSet.add(d);
  }
  return plantedSet;
}

/** Doc IDs containing a term. fsyncDf only affects "fsync". */
export function docsFor(term: string, fsyncDf: number): Int32Array {
  const key = term === 'fsync' ? `fsync:${fsyncDf}` : term;
  const hit = cache.get(key);
  if (hit) return hit;
  let docs: number[];
  if (term === 'fsync') docs = sampleDocs(fsyncDf, fnv(key));
  else if (term === 'ahead') {
    const set = new Set<number>(planted());
    for (const d of sampleDocs(AHEAD_RANDOM_DF, fnv('ahead'))) set.add(d);
    docs = [...set].sort((a, b) => a - b);
  } else docs = sampleDocs(FIXED_DF[term] ?? 50 + (mix(fnv(term), 1) % 20_000), fnv(term));
  const arr = Int32Array.from(docs);
  cache.set(key, arr);
  return arr;
}

export const docLen = (doc: number) => 60 + (mix(doc, 0x1e2) % 340);

/** Token positions of a term inside a document (sorted, unique). */
export function positionsOf(term: string, doc: number): number[] {
  const r = makeRng(mix(fnv(term), doc) | 1);
  r();
  const L = docLen(doc);
  const f = 1 + Math.floor(r() * 3);
  const set = new Set<number>();
  for (let i = 0; i < f; i++) set.add(Math.floor(r() * (L - 1)));
  if (term === 'ahead' && planted().has(doc)) set.add(positionsOf('write', doc)[0] + 1);
  return [...set].sort((a, b) => a - b);
}

/** Display-only docFreq for dictionary terms that no query uses. */
export function dictDf(term: string, fsyncDf: number) {
  if (term === 'fsync' || term === 'ahead' || term in FIXED_DF) return docsFor(term, fsyncDf).length;
  return 50 + (mix(fnv(term), 1) % 20_000);
}

/* ------------------------------------------------------------- dictionary */

export type BinaryLookup = { probes: number[]; block: number; scanned: number[]; found: number; comparisons: number };

/** Binary search over the in-memory index of block-leading terms, then a linear scan inside the block. */
export function lookupBinary(term: string): BinaryLookup {
  const nb = Math.ceil(VOCAB.length / DICT_BLOCK);
  let lo = 0;
  let hi = nb - 1;
  const probes: number[] = [];
  while (lo < hi) {
    const mid = (lo + hi + 1) >> 1;
    probes.push(mid);
    if (VOCAB[mid * DICT_BLOCK] <= term) lo = mid;
    else hi = mid - 1;
  }
  const scanned: number[] = [];
  let found = -1;
  for (let i = lo * DICT_BLOCK; i < Math.min(VOCAB.length, (lo + 1) * DICT_BLOCK); i++) {
    scanned.push(i);
    if (VOCAB[i] >= term) {
      if (VOCAB[i] === term) found = i;
      break;
    }
  }
  return { probes, block: lo, scanned, found, comparisons: probes.length + scanned.length };
}

export type PrefixBinary = BinaryLookup & { range: number[]; blocksRead: number };

/** Prefix as a dictionary range scan: seek to the first term >= prefix, then read forward while terms share it. */
export function prefixBinary(prefix: string): PrefixBinary {
  const seek = lookupBinary(prefix);
  const scanned = seek.scanned.slice();
  let i = scanned[scanned.length - 1];
  const range: number[] = [];
  const blocks = new Set<number>([seek.block]);
  if (VOCAB[i] < prefix) i++;
  while (i < VOCAB.length) {
    blocks.add(Math.floor(i / DICT_BLOCK));
    if (!scanned.includes(i)) scanned.push(i);
    if (!VOCAB[i].startsWith(prefix)) break;
    range.push(i);
    i++;
  }
  return { ...seek, scanned, range, blocksRead: blocks.size, comparisons: seek.probes.length + scanned.length };
}

type TrieNode = { children: Map<string, TrieNode>; term: number };
let trieRoot: TrieNode | null = null;
function trie() {
  if (!trieRoot) {
    trieRoot = { children: new Map(), term: -1 };
    VOCAB.forEach((t, idx) => {
      let n = trieRoot!;
      for (const ch of t) {
        let c = n.children.get(ch);
        if (!c) {
          c = { children: new Map(), term: -1 };
          n.children.set(ch, c);
        }
        n = c;
      }
      n.term = idx;
    });
  }
  return trieRoot;
}

export type TrieWalk = { steps: { ch: string; arcsOut: number }[]; found: number; failedAt: number; below: number[]; nodesVisited: number };

/** Walk one arc per character. For a prefix, then enumerate every term below the node reached. */
export function walkTrie(key: string, asPrefix: boolean): TrieWalk {
  let n = trie();
  const steps: { ch: string; arcsOut: number }[] = [];
  for (let i = 0; i < key.length; i++) {
    const next = n.children.get(key[i]);
    steps.push({ ch: key[i], arcsOut: n.children.size });
    if (!next) return { steps, found: -1, failedAt: i, below: [], nodesVisited: 0 };
    n = next;
  }
  const below: number[] = [];
  let nodesVisited = 0;
  if (asPrefix) {
    const stack = [n];
    while (stack.length) {
      const x = stack.pop()!;
      nodesVisited++;
      if (x.term >= 0) below.push(x.term);
      stack.push(...[...x.children.values()].reverse());
    }
    below.sort((a, b) => a - b);
  }
  return { steps, found: n.term, failedAt: -1, below, nodesVisited };
}

/* ------------------------------------------------------------- postings */

export type QueryKind = 'and2' | 'and3' | 'phrase' | 'prefix' | 'or';
export type Strategy = 'leapfrog' | 'drive';
export type Order = 'rare' | 'common';
export type PhraseMode = 'positions' | 'recheck';

export const QUERIES: Record<QueryKind, { label: string; terms: string[]; prefix?: string }> = {
  and2: { label: 'fsync AND postgres', terms: ['fsync', 'postgres'] },
  and3: { label: 'fsync AND checkpoint AND postgres', terms: ['fsync', 'checkpoint', 'postgres'] },
  phrase: { label: '"write ahead" (phrase)', terms: ['write', 'ahead'] },
  prefix: { label: 'compact* (prefix)', terms: VOCAB.filter((t) => t.startsWith('compact')), prefix: 'compact' },
  or: { label: 'fsync OR postgres', terms: ['fsync', 'postgres'] },
};

export type AdvEv = {
  kind: 'adv';
  list: number;
  next: boolean;
  target: number;
  from: number;
  to: number;
  jumpFrom: number;
  jumpTo: number;
  entries: number;
  decFrom: number;
  decTo: number;
  decoded: number;
};
export type MatchEv = { kind: 'match'; doc: number };
export type PhraseEv = { kind: 'phrase' | 'recheck'; doc: number; ok: boolean; posRead: number; lists: { term: string; pos: number[] }[] };
export type Ev = AdvEv | MatchEv | PhraseEv;

export type ListMeta = { term: string; color: string; docs: Int32Array; df: number; B: number; nFull: number; nBlocks: number };

type Cur = { m: ListMeta; idx: number; pos: number; block: number; doc: number };

const lastDoc = (m: ListMeta, b: number) => m.docs[Math.min(m.df, (b + 1) * m.B) - 1];
const blockLen = (m: ListMeta, b: number) => Math.min(m.df, (b + 1) * m.B) - b * m.B;

export type RunOpts = { query: QueryKind; strategy: Strategy; order: Order; skip: boolean; blockSize: number; fsyncDf: number; phraseMode: PhraseMode; record?: boolean };

export type Totals = {
  decoded: number;
  postings: number;
  entries: number;
  blocksDecoded: number;
  blocks: number;
  matches: number;
  candidates: number;
  rejected: number;
  posRead: number;
  rechecks: number;
  perList: { decoded: number; entries: number; blocksDecoded: number }[];
};

const MAX_EVENTS = 400_000;

export function runQuery(o: RunOpts) {
  const q = QUERIES[o.query];
  const record = o.record !== false;
  const base: ListMeta[] = q.terms.map((term, i) => {
    const docs = docsFor(term, o.fsyncDf);
    const df = docs.length;
    return { term, color: SERIES[i], docs, df, B: o.blockSize, nFull: Math.floor(df / o.blockSize), nBlocks: Math.ceil(df / o.blockSize) };
  });
  const usesOrder = o.query !== 'or' && o.query !== 'prefix';
  const lists = usesOrder ? base.slice().sort((a, b) => (o.order === 'rare' ? a.df - b.df : b.df - a.df)) : base;
  const curs: Cur[] = lists.map((m, idx) => ({ m, idx, pos: -1, block: -1, doc: -1 }));
  const events: Ev[] = [];
  const t: Totals = {
    decoded: 0,
    postings: lists.reduce((s, m) => s + m.df, 0),
    entries: 0,
    blocksDecoded: 0,
    blocks: lists.reduce((s, m) => s + m.nBlocks, 0),
    matches: 0,
    candidates: 0,
    rejected: 0,
    posRead: 0,
    rechecks: 0,
    perList: lists.map(() => ({ decoded: 0, entries: 0, blocksDecoded: 0 })),
  };
  const push = (e: Ev) => {
    if (record && events.length < MAX_EVENTS) events.push(e);
  };

  /** Move a cursor to the first doc >= target. */
  const adv = (c: Cur, target: number, next: boolean): number => {
    if (c.doc >= target) return c.doc;
    const m = c.m;
    const pl = t.perList[c.idx];
    const from = c.doc;
    let b = c.block;
    let jumpFrom = -1;
    let jumpTo = -1;
    let entries = 0;
    let decFrom = -1;
    let decTo = -1;
    let decoded = 0;
    const finish = (to: number) => {
      c.doc = to;
      t.entries += entries;
      pl.entries += entries;
      t.decoded += decoded;
      pl.decoded += decoded;
      const nb = decFrom < 0 ? 0 : decTo - decFrom + 1;
      t.blocksDecoded += nb;
      pl.blocksDecoded += nb;
      push({ kind: 'adv', list: c.idx, next, target, from, to, jumpFrom, jumpTo, entries, decFrom, decTo, decoded });
      return to;
    };
    const decode = (lo: number, hi: number) => {
      decFrom = lo;
      decTo = hi;
      for (let k = lo; k <= hi; k++) decoded += blockLen(m, k);
    };
    if (target >= NO_MORE) {
      c.pos = m.df;
      c.block = m.nBlocks;
      return finish(NO_MORE);
    }
    if (!(b >= 0 && b < m.nBlocks && lastDoc(m, b) >= target)) {
      b += 1;
      if (o.skip && !next) {
        jumpFrom = b;
        // level 1: the entry before each group of 32 packed blocks holds the group's last doc
        for (;;) {
          const g = Math.floor(b / 32);
          const gEnd = (g + 1) * 32;
          if (gEnd > m.nFull) break;
          if (b === g * 32) entries++;
          if (lastDoc(m, gEnd - 1) >= target) break;
          b = gEnd;
        }
        // level 0: one entry before every packed block
        while (b < m.nFull) {
          entries++;
          if (lastDoc(m, b) >= target) break;
          b++;
        }
        jumpTo = b;
        if (b >= m.nBlocks) {
          c.pos = m.df;
          c.block = m.nBlocks;
          return finish(NO_MORE);
        }
        // the tail (VInt) block has no skip entry: decode it to find out
        if (b >= m.nFull && lastDoc(m, b) < target) {
          decode(b, b);
          c.pos = m.df;
          c.block = m.nBlocks;
          return finish(NO_MORE);
        }
        decode(b, b);
      } else {
        const lo = b;
        while (b < m.nBlocks && lastDoc(m, b) < target) b++;
        if (o.skip) {
          // nextDoc() still reads the entry in front of each packed block it enters (and a level-1 entry when it
          // crosses into a complete group of 32), but uses it only to find the block's length.
          for (let k = lo; k <= Math.min(b, m.nBlocks - 1) && k < m.nFull; k++) {
            entries++;
            if (k % 32 === 0 && k + 32 <= m.nFull) entries++;
          }
        }
        if (b >= m.nBlocks) {
          if (lo < m.nBlocks) decode(lo, m.nBlocks - 1);
          c.pos = m.df;
          c.block = m.nBlocks;
          return finish(NO_MORE);
        }
        decode(lo, b);
      }
      c.block = b;
    }
    let p = Math.max(c.pos + 1, c.block * m.B);
    while (m.docs[p] < target) p++;
    c.pos = p;
    return finish(m.docs[p]);
  };

  const phraseOffsets: Record<string, number> = { write: 0, ahead: 1 };
  const checkPhrase = (doc: number) => {
    // ExactPhraseMatcher orders its terms by phrase position (PostingsAndFreq.compareTo), not by docFreq:
    // the first phrase term leads the position check whatever order the doc-level conjunction used.
    const P = curs
      .map((c) => ({ term: c.m.term, off: phraseOffsets[c.m.term] ?? 0, list: positionsOf(c.m.term, doc), upTo: 0, pos: -1 }))
      .sort((a, b) => a.off - b.off);
    let posRead = 0;
    const read = (x: (typeof P)[number]) => {
      x.pos = x.list[x.upTo++];
      posRead++;
    };
    const advPos = (x: (typeof P)[number], target: number) => {
      while (x.pos < target) {
        if (x.upTo === x.list.length) return false;
        read(x);
      }
      return true;
    };
    const lead = P[0];
    let ok = false;
    if (lead.upTo < lead.list.length) {
      read(lead);
      head: for (;;) {
        const phrasePos = lead.pos - lead.off;
        for (let j = 1; j < P.length; j++) {
          const x = P[j];
          const expected = phrasePos + x.off;
          if (!advPos(x, expected)) break head;
          if (x.pos !== expected) {
            if (advPos(lead, x.pos - x.off + lead.off)) continue head;
            break head;
          }
        }
        ok = true;
        break;
      }
    }
    return { ok, posRead, lists: P.map((x) => ({ term: x.term, pos: x.list })) };
  };

  const accept = (doc: number) => {
    if (o.query === 'phrase') {
      t.candidates++;
      const r = checkPhrase(doc);
      if (o.phraseMode === 'positions') t.posRead += r.posRead;
      else t.rechecks++;
      push({ kind: o.phraseMode === 'positions' ? 'phrase' : 'recheck', doc, ok: r.ok, posRead: o.phraseMode === 'positions' ? r.posRead : 0, lists: r.lists });
      if (!r.ok) {
        t.rejected++;
        return;
      }
    }
    t.matches++;
    push({ kind: 'match', doc });
  };

  if (!usesOrder) {
    for (const c of curs) adv(c, 0, true);
    for (let guard = 0; guard < N_DOCS + 1; guard++) {
      let min = NO_MORE;
      for (const c of curs) if (c.doc < min) min = c.doc;
      if (min >= NO_MORE) break;
      accept(min);
      for (const c of curs) if (c.doc === min) adv(c, min + 1, true);
    }
  } else {
    const [l1, l2, ...others] = curs;
    let doc = adv(l1, 0, true);
    for (let guard = 0; doc < NO_MORE && guard < 4 * N_DOCS; guard++) {
      if (o.strategy === 'leapfrog') {
        let agreed = false;
        head: for (;;) {
          const n2 = adv(l2, doc, false);
          if (n2 !== doc) {
            doc = adv(l1, n2, false);
            if (doc >= NO_MORE) break;
            if (n2 !== doc) continue;
          }
          for (const x of others) {
            if (x.doc < doc) {
              const n = adv(x, doc, false);
              if (n > doc) {
                doc = adv(l1, n, false);
                if (doc >= NO_MORE) break head;
                continue head;
              }
            }
          }
          agreed = true;
          break;
        }
        if (!agreed) break;
        accept(doc);
      } else {
        let all = true;
        for (const x of [l2, ...others]) {
          if (x.doc < doc) adv(x, doc, false);
          if (x.doc !== doc) all = false;
        }
        if (all) accept(doc);
      }
      doc = adv(l1, doc + 1, true);
    }
  }
  return { lists, events, totals: t, usesOrder, truncated: events.length >= MAX_EVENTS };
}

export type Run = ReturnType<typeof runQuery>;

const BINS = 490;

/** State after the first k events: per-block states (0 untouched, 1 skipped, 2 decoded), cursors and counters. */
export function replay(run: Run, k: number) {
  const n = Math.min(k, run.events.length);
  const states = run.lists.map((m) => new Uint8Array(m.nBlocks));
  const cur = run.lists.map(() => -1);
  const lastAdv: (AdvEv | null)[] = run.lists.map(() => null);
  const matchBins = new Uint16Array(BINS);
  const rejectBins = new Uint16Array(BINS);
  let decoded = 0;
  let entries = 0;
  let blocksDecoded = 0;
  let matches = 0;
  let candidates = 0;
  let posRead = 0;
  let rechecks = 0;
  let lastPhrase: PhraseEv | null = null;
  let lastIdx = -1;
  const bin = (doc: number) => Math.min(BINS - 1, Math.floor((doc / N_DOCS) * BINS));
  for (let i = 0; i < n; i++) {
    const e = run.events[i];
    if (e.kind === 'adv') {
      const s = states[e.list];
      if (e.jumpFrom >= 0) for (let b = e.jumpFrom; b < Math.min(e.jumpTo, s.length); b++) if (s[b] === 0) s[b] = 1;
      if (e.decFrom >= 0) for (let b = e.decFrom; b <= e.decTo; b++) s[b] = 2;
      cur[e.list] = e.to;
      lastAdv[e.list] = e;
      decoded += e.decoded;
      entries += e.entries;
      blocksDecoded += e.decFrom >= 0 ? e.decTo - e.decFrom + 1 : 0;
    } else if (e.kind === 'match') {
      matches++;
      matchBins[bin(e.doc)]++;
    } else {
      candidates++;
      posRead += e.posRead;
      if (e.kind === 'recheck') rechecks++;
      if (!e.ok) rejectBins[bin(e.doc)]++;
      lastPhrase = e;
    }
    lastIdx = i;
  }
  return { states, cur, lastAdv, matchBins, rejectBins, decoded, entries, blocksDecoded, matches, candidates, posRead, rechecks, lastPhrase, lastIdx, shown: n };
}

export function describe(run: Run, e: Ev): string {
  if (e.kind === 'match') return run.usesOrder ? `Every list is on doc ${fmtNum(e.doc)}: match.` : `Doc ${fmtNum(e.doc)} is the smallest doc ID in the heap: emit it, then step every list sitting on it.`;
  if (e.kind === 'phrase' || e.kind === 'recheck') {
    const ps = e.lists.map((l) => `${l.term} @ ${l.pos.join(', ')}`).join(' · ');
    return e.kind === 'phrase'
      ? `Doc ${fmtNum(e.doc)} has both terms. Position lists ${ps} → ${e.ok ? 'adjacent: phrase matches' : 'never adjacent: rejected'} (${e.posRead} positions read).`
      : `Doc ${fmtNum(e.doc)} has both terms, but the index holds no positions: fetch the heap row and recheck its tsvector → ${e.ok ? 'phrase matches' : 'not adjacent, row discarded'}.`;
  }
  if (e.kind !== 'adv') return '';
  const m = run.lists[e.list];
  const to = e.to >= NO_MORE ? 'exhausted' : fmtNum(e.to);
  const call = e.next ? `${m.term}.nextDoc()` : `${m.term}.advance(${e.target >= NO_MORE ? 'NO_MORE_DOCS' : fmtNum(e.target)})`;
  const dec = e.decFrom < 0 ? (e.to >= NO_MORE ? (e.target >= NO_MORE ? 'reads nothing' : 'has no blocks left') : 'stays inside the block already decoded') : e.decFrom === e.decTo ? `decodes block ${e.decFrom} (${fmtNum(e.decoded)} postings)` : `decodes blocks ${e.decFrom}–${e.decTo} (${fmtNum(e.decoded)} postings)`;
  if (e.entries > 0) {
    const jumped = Math.max(0, e.jumpTo - e.jumpFrom);
    const jumps = jumped > 0 ? `, jumps ${jumped} block${jumped === 1 ? '' : 's'} without decoding them` : '';
    return `${call}: reads ${e.entries} skip entr${e.entries === 1 ? 'y' : 'ies'}${jumps}, ${dec} → ${to}.`;
  }
  return `${call}: ${dec} → ${to}.`;
}

/* ------------------------------------------------------------------ view */

export default function TermDictPostingsLeapfrogLab() {
  const [query, setQuery] = useState<QueryKind>('and2');
  const [strategy, setStrategy] = useState<Strategy>('leapfrog');
  const [order, setOrder] = useState<Order>('rare');
  const [skip, setSkip] = useState(true);
  const [blockSize, setBlockSize] = useState<'128' | '256'>('128');
  const [dfIdx, setDfIdx] = useState(2);
  const [phraseMode, setPhraseMode] = useState<PhraseMode>('positions');
  const [lookup, setLookup] = useState<'binary' | 'trie'>('binary');
  const [dictTerm, setDictTerm] = useState(0);
  const [step, setStep] = useState(Number.POSITIVE_INFINITY);
  const [sizeRef, width] = useSize(680);

  const fsyncDf = FSYNC_DFS[useDeferredValue(dfIdx)];
  const opts: RunOpts = { query, strategy, order, skip, blockSize: Number(blockSize), fsyncDf, phraseMode };
  const deps = [query, strategy, order, skip, blockSize, fsyncDf, phraseMode];
  const run = useMemo(() => runQuery(opts), deps);
  const alt = useMemo(() => (run.usesOrder ? runQuery({ ...opts, order: order === 'rare' ? 'common' : 'rare', record: false }) : null), deps);
  const noSkip = useMemo(() => (skip ? runQuery({ ...opts, skip: false, record: false }) : null), deps);
  const r = useMemo(() => replay(run, step), [run, step]);
  const done = r.shown >= run.events.length;
  const t = run.totals;
  const q = QUERIES[query];

  const resetStep = () => setStep(Number.POSITIVE_INFINITY);
  const change =
    <T,>(fn: (v: T) => void) =>
    (v: T) => {
      fn(v);
      resetStep();
    };
  const nextMatch = () => {
    let i = Math.min(r.shown, run.events.length);
    while (i < run.events.length && run.events[i].kind === 'adv') i++;
    // a phrase check that succeeds is followed by its match event: show both in one press
    const e = run.events[i];
    if (e && e.kind !== 'adv' && e.kind !== 'match' && e.ok && run.events[i + 1]?.kind === 'match') i++;
    setStep(Math.min(run.events.length, i + 1));
  };

  // ---- layout
  const W = Math.round(Math.max(300, Math.min(680, width)));
  const narrow = W < 520;

  // ---- dictionary
  const dictTerms = q.prefix ? [q.prefix] : q.terms;
  const dIdx = Math.min(dictTerm, dictTerms.length - 1);
  const dKey = dictTerms[dIdx];
  const bin = q.prefix ? prefixBinary(q.prefix) : lookupBinary(dKey);
  const walk = walkTrie(dKey, !!q.prefix);
  const hits = q.prefix ? (lookup === 'binary' ? (bin as PrefixBinary).range : walk.below) : [lookup === 'binary' ? bin.found : walk.found];
  const termColor = (vi: number) => {
    const i = q.terms.indexOf(VOCAB[vi]);
    return i >= 0 ? SERIES[i] : 'var(--viz-ink-2)';
  };
  const nDictBlocks = Math.ceil(VOCAB.length / DICT_BLOCK);
  const cols = W < 580 ? 4 : 8;
  const colW = (W - 4) / cols;
  const blockH = 22 + DICT_BLOCK * 16 + 4;
  const dictH = Math.ceil(nDictBlocks / cols) * (blockH + 4);
  const trieStep = Math.min(54, (W - 30) / Math.max(1, walk.steps.length));

  // ---- lanes
  const LANE_X = narrow ? 78 : 112;
  const READ_W = narrow ? 0 : 86;
  const LANE_W = W - LANE_X - READ_W - 4;
  const X = (doc: number) => LANE_X + (Math.min(doc, N_DOCS) / N_DOCS) * LANE_W;
  const laneH = 30;
  const nL = run.lists.length;
  const lanesH = 12 + (nL + 1) * laneH + 22;
  const leadIdx = run.usesOrder ? 0 : -1;
  const lastEv = r.lastIdx >= 0 ? run.events[r.lastIdx] : null;
  const recentFrom = Math.max(0, r.shown - 4);
  const recent = run.events.slice(recentFrom, r.shown);

  const laneMarks = useMemo(
    () =>
      run.lists.map((m, i) => {
        const s = r.states[i];
        const runs: { from: number; to: number; st: number }[] = [];
        const ticks = new Uint8Array(BINS);
        for (let b = 0; b < m.nBlocks; b++) {
          const st = s[b];
          if (st === 2) for (let k = b * m.B; k < Math.min(m.df, (b + 1) * m.B); k++) ticks[Math.min(BINS - 1, Math.floor((m.docs[k] / N_DOCS) * BINS))] = 1;
          if (st === 0) continue;
          const last = runs[runs.length - 1];
          if (last && last.st === st && last.to === b - 1) last.to = b;
          else runs.push({ from: b, to: b, st });
        }
        return { runs, ticks };
      }),
    [r],
  );

  const orderWord = order === 'rare' ? 'rarest term first' : 'most common term first';
  const altDecoded = alt ? alt.totals.decoded : 0;
  const ratio = (a: number, b: number) => fmtNum(a / Math.max(1, b), 1);

  return (
    <VizPanel
      title="Answering a query from a prebuilt inverted index"
      subtitle="Each query term is looked up in the term dictionary. Its postings are sorted doc IDs packed into blocks, with a skip entry before each block. Cursors advance over the lists; the counters show how many postings had to be decoded to find the matches."
      controls={
        <>
          <Choice label="Query" value={query} onChange={(v) => { setQuery(v); setDictTerm(0); resetStep(); }} options={(Object.keys(QUERIES) as QueryKind[]).map((k) => ({ value: k, label: QUERIES[k].label }))} />
          {run.usesOrder ? (
            <>
              <Segmented label="Intersection" value={strategy} onChange={change(setStrategy)} options={[{ value: 'leapfrog', label: 'Leapfrog (Lucene)', title: 'Any list that overshoots pulls the lead forward (ConjunctionDISI)' }, { value: 'drive', label: 'Lead drives (GIN)', title: 'Only the lead proposes candidates; every other list is advanced to each one (GIN required/additional entries)' }]} />
              <Segmented label="Evaluation order" value={order} onChange={change(setOrder)} options={[{ value: 'rare', label: 'Rarest first' }, { value: 'common', label: 'Most common first' }]} />
            </>
          ) : null}
          {query === 'phrase' ? <Segmented label="Positions" value={phraseMode} onChange={change(setPhraseMode)} options={[{ value: 'positions', label: 'In the index (Lucene)' }, { value: 'recheck', label: 'Heap recheck (GIN)' }]} /> : null}
          <Segmented label="Postings per block" value={blockSize} onChange={change(setBlockSize)} options={[{ value: '128', label: '128', title: 'Lucene 9.x–10.3 and Tantivy' }, { value: '256', label: '256', title: 'Lucene 10.4 and later' }]} />
          {q.terms.includes('fsync') ? <Slider label="docFreq of fsync" min={0} max={FSYNC_DFS.length - 1} value={dfIdx} onChange={change(setDfIdx)} format={(i) => fmtNum(docsFor('fsync', FSYNC_DFS[i]).length)} /> : null}
          <Check label="Skip data" checked={skip} onChange={change(setSkip)} />
        </>
      }
      legend={
        <Legend
          items={[
            ...q.terms.map((term, i) => ({ label: term, color: SERIES[i] })),
            { label: 'Shaded block with ticks: decoded', color: 'var(--viz-ink-2)' },
            { label: 'Thin line: jumped, only its skip entry read', color: 'var(--viz-ink-2)', shape: 'line' as const },
            { label: 'Matching document', color: 'var(--viz-good)' },
            ...(query === 'phrase' ? [{ label: 'Candidate rejected by the phrase check', color: 'var(--viz-critical)' }] : []),
          ]}
        />
      }
      stats={
        <Stats
          items={[
            { label: `Postings decoded (of ${fmtNum(t.postings)})`, value: fmtNum(r.decoded), hint: 'Whole blocks are decoded at once, as Lucene and Tantivy do.' },
            { label: `Blocks decoded (of ${fmtNum(t.blocks)})`, value: fmtNum(r.blocksDecoded) },
            { label: 'Skip entries read', value: fmtNum(r.entries), hint: 'A level-0 entry before each packed block, a level-1 entry before each 32 blocks. advance() reads entries until it finds its block; nextDoc() reads only the entry of the block it enters.' },
            ...(query === 'phrase'
              ? [
                  { label: 'Candidates (both terms)', value: fmtNum(r.candidates) },
                  phraseMode === 'positions' ? { label: 'Positions read', value: fmtNum(r.posRead) } : { label: 'Heap rows rechecked', value: fmtNum(r.rechecks) },
                ]
              : []),
            { label: 'Documents matched', value: fmtNum(r.matches) },
          ]}
        />
      }
      note={
        <Note>
          {!done ? <strong>Stepping: operation {fmtNum(r.shown)} of {fmtNum(run.events.length)}. </strong> : null}
          {run.usesOrder ? (
            <>
              <strong>
                {strategy === 'leapfrog' ? 'Leapfrog' : 'Lead drives'}, {orderWord}: {fmtNum(t.decoded)} postings decoded to find {fmtNum(t.matches)} match{t.matches === 1 ? '' : 'es'}.
              </strong>{' '}
              The reverse order decodes {fmtNum(altDecoded)}
              {altDecoded > t.decoded * 1.3 ? ` (${ratio(altDecoded, t.decoded)}× as many)` : altDecoded * 1.3 < t.decoded ? ` (${ratio(t.decoded, altDecoded)}× fewer)` : ' (about the same)'}
              {strategy === 'leapfrog' ? ': a list that overshoots pulls the lead forward, whichever list leads. ' : ': only the lead proposes candidates, and every other list is advanced to each one. '}
              {noSkip ? `Without skip data this query decodes ${fmtNum(noSkip.totals.decoded)}.` : 'Skip data is off, so every block up to each target is decoded.'}
              {query === 'phrase' ? (phraseMode === 'positions' ? ` Of ${fmtNum(t.candidates)} documents with both terms, ${fmtNum(t.rejected)} fail the position check.` : ` Every one of the ${fmtNum(t.candidates)} documents with both terms costs a heap fetch; ${fmtNum(t.rejected)} are discarded.`) : ''}
            </>
          ) : (
            <>
              <strong>
                A union decodes every posting of every list: {fmtNum(t.decoded)} of {fmtNum(t.postings)}, for {fmtNum(t.matches)} matching documents.
              </strong>{' '}
              {query === 'prefix' ? `The prefix expanded to ${q.terms.length} terms, one posting list each, merged by a doc-ID heap. ` : 'A doc-ID heap merges the lists; nothing tells a cursor where to jump, so skip entries are read only to step into each next block. '}
              {q.terms.includes('fsync') ? 'Raise the docFreq of fsync and the cost rises one for one.' : ''}
            </>
          )}
        </Note>
      }
      table={
        <table className="viz-table">
          <thead>
            <tr>
              <th>List (evaluation order)</th>
              <th>docFreq</th>
              <th>Blocks</th>
              <th>Blocks decoded</th>
              <th>Postings decoded</th>
              <th>Skip entries read</th>
            </tr>
          </thead>
          <tbody>
            {run.lists.map((m, i) => (
              <tr key={m.term}>
                <td>{m.term}</td>
                <td>{fmtNum(m.df)}</td>
                <td>{fmtNum(m.nBlocks)}</td>
                <td>{fmtNum(t.perList[i].blocksDecoded)}</td>
                <td>{fmtNum(t.perList[i].decoded)}</td>
                <td>{fmtNum(t.perList[i].entries)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      }
    >
      <div ref={sizeRef} style={{ width: '100%' }}>
        <p className="viz-sub" style={{ margin: '0 0 .4rem' }}>
          <strong>1. Term dictionary.</strong> 64 terms in blocks of {DICT_BLOCK} (a model: Lucene’s blocks hold 25–48 terms).
        </p>
        <div className="viz-controls">
          <Segmented label="Lookup" value={lookup} onChange={setLookup} options={[{ value: 'binary', label: 'Binary search' }, { value: 'trie', label: 'Trie walk' }]} />
          {dictTerms.length > 1 ? <Segmented label="Term" value={String(dIdx)} onChange={(v) => setDictTerm(Number(v))} options={dictTerms.map((d, i) => ({ value: String(i), label: d }))} /> : null}
        </div>
        <svg viewBox={`0 0 ${W} ${dictH}`} width={W} height={dictH} role="img" aria-label={`Term dictionary: ${lookup === 'binary' ? `binary search probes blocks ${bin.probes.join(', ')}, then scans ${bin.scanned.length} entries` : `trie walk of ${walk.steps.length} arcs`} for ${dKey}`}>
          {Array.from({ length: nDictBlocks }, (_, b) => {
            const x = 2 + (b % cols) * colW;
            const y0 = Math.floor(b / cols) * (blockH + 4);
            const probeNo = lookup === 'binary' ? bin.probes.indexOf(b) : -1;
            const chosen = lookup === 'binary' && (bin.block === b || (!!q.prefix && hits.some((vi) => Math.floor(vi / DICT_BLOCK) === b)));
            return (
              <g key={b} data-block={b}>
                <rect x={x + 1} y={y0 + 1} width={colW - 2} height={blockH - 2} rx={4} fill="var(--viz-surface)" stroke={chosen ? 'var(--viz-ink-2)' : 'var(--viz-border)'} strokeWidth={chosen ? 1.5 : 1} strokeDasharray={probeNo >= 0 && !chosen ? '4 3' : undefined} />
                <text x={x + 6} y={y0 + 14} fontSize={10} fill="var(--viz-ink-2)">
                  block {b}
                </text>
                {probeNo >= 0 ? (
                  <g>
                    <circle cx={x + colW - 12} cy={y0 + 10} r={7} fill="var(--viz-surface)" stroke="var(--viz-ink-2)" />
                    <text x={x + colW - 12} y={y0 + 13.5} fontSize={9} textAnchor="middle" fill="var(--viz-ink)">
                      {probeNo + 1}
                    </text>
                  </g>
                ) : null}
                {VOCAB.slice(b * DICT_BLOCK, (b + 1) * DICT_BLOCK).map((term, row) => {
                  const vi = b * DICT_BLOCK + row;
                  const y = y0 + 20 + row * 16;
                  const hit = hits.includes(vi);
                  const scanned = lookup === 'binary' && bin.scanned.includes(vi);
                  const leading = row === 0 && probeNo >= 0;
                  return (
                    <g key={term}>
                      {scanned || hit ? <rect x={x + 3} y={y} width={colW - 6} height={15} rx={2} fill={scanned ? 'var(--viz-neutral)' : 'var(--viz-surface)'} stroke={hit ? termColor(vi) : 'none'} strokeWidth={2} /> : null}
                      <text x={x + 7} y={y + 11} style={{ fontSize: Math.min(11, (colW - 13) / (term.length * (hit || leading ? 0.62 : 0.58))) }} fill={hit || leading ? 'var(--viz-ink)' : 'var(--viz-ink-2)'} fontWeight={hit || leading ? 600 : 400}>
                        {term}
                      </text>
                    </g>
                  );
                })}
              </g>
            );
          })}
        </svg>
        {lookup === 'trie' ? (
          <svg viewBox={`0 0 ${W} 48`} width={W} height={48} role="img" aria-label={`Trie walk for ${dKey}: ${walk.steps.length} arcs, one per character`} style={{ marginTop: 4 }}>
            {[{ ch: '', arcsOut: 0 }, ...walk.steps].map((s, i) => {
              const x = 12 + i * trieStep;
              return (
                <g key={i}>
                  {i > 0 ? (
                    <>
                      <line x1={x - trieStep + 7} x2={x - 7} y1={22} y2={22} stroke="var(--viz-ink-2)" />
                      <text x={x - trieStep / 2} y={17} fontSize={11} textAnchor="middle" fill="var(--viz-ink)" fontWeight={600}>
                        {s.ch}
                      </text>
                    </>
                  ) : null}
                  <circle cx={x} cy={22} r={6} fill={i === walk.steps.length ? 'var(--viz-ink-2)' : 'var(--viz-surface)'} stroke="var(--viz-ink-2)" />
                  {i < walk.steps.length ? (
                    <text x={x} y={42} fontSize={9} textAnchor="middle" fill="var(--viz-ink-muted)">
                      {walk.steps[i].arcsOut}
                    </text>
                  ) : null}
                </g>
              );
            })}
          </svg>
        ) : null}
        <p style={{ fontSize: '0.8rem', margin: '0.3rem 0 0.9rem', color: 'var(--viz-ink-2)' }} data-testid="dict-caption">
          {lookup === 'binary'
            ? q.prefix
              ? `Seek to the first term ≥ “${q.prefix}”: ${bin.probes.length} probes of the in-memory index of block-leading terms, then read forward while terms still start with it — ${hits.length} terms across ${(bin as PrefixBinary).blocksRead} blocks, ${bin.scanned.length} entries compared.`
              : `“${dKey}”: ${bin.probes.length} probes (numbered) of the in-memory index of block-leading terms pick block ${bin.block}; ${bin.scanned.length} entr${bin.scanned.length === 1 ? 'y is' : 'ies are'} scanned inside it. The entry holds docFreq ${fmtNum(dictDf(dKey, fsyncDf))} and a file pointer to the postings.`
            : q.prefix
              ? `Follow ${walk.steps.length} arcs for “${q.prefix}”, then enumerate the subtree below that node (${walk.nodesVisited} nodes): ${walk.below.length} terms, each becoming one posting list in a union.`
              : `“${dKey}”: ${walk.steps.length} arcs, one per byte, however large the vocabulary. The number under each node is how many arcs leave it; a missing arc proves the term is absent without reading any block.`}
        </p>

        <p className="viz-sub" style={{ margin: '0 0 .4rem' }}>
          <strong>2. Posting lists.</strong> Doc IDs 0–{fmtNum(N_DOCS)} run left to right; {run.usesOrder ? 'the lead is on top.' : 'a doc-ID heap merges the lists.'}
        </p>
        <div className="viz-controls">
          <Button onClick={() => setStep(1)}>Step from the start</Button>
          <Button primary onClick={() => setStep(Math.min(run.events.length, r.shown + 1))} disabled={done}>
            Next operation
          </Button>
          <Button onClick={nextMatch} disabled={done}>
            Next match
          </Button>
          <Button onClick={resetStep} disabled={done}>
            Show result
          </Button>
        </div>
        <svg viewBox={`0 0 ${W} ${lanesH}`} width={W} height={lanesH} role="img" aria-label={`Posting lists: ${fmtNum(r.blocksDecoded)} of ${fmtNum(t.blocks)} blocks decoded, ${fmtNum(r.matches)} matches`}>
          {run.lists.map((m, i) => {
            const y = 12 + i * laneH;
            const { runs, ticks } = laneMarks[i];
            const c = r.cur[i];
            const la = r.lastAdv[i];
            const isLast = !!lastEv && lastEv.kind === 'adv' && lastEv.list === i;
            return (
              <g key={m.term} data-lane={m.term}>
                <text x={0} y={y + 12} fontSize={11} fill="var(--viz-ink)" fontWeight={i === leadIdx ? 600 : 400}>
                  {m.term}
                  {i === leadIdx ? ' (lead)' : ''}
                </text>
                <text x={0} y={y + 24} fontSize={9} fill="var(--viz-ink-muted)">
                  df {fmtNum(m.df)}{narrow ? '' : ` · ${fmtNum(m.nBlocks)} blk`}
                </text>
                <line x1={LANE_X} x2={LANE_X + LANE_W} y1={y + 20} y2={y + 20} stroke="var(--viz-grid)" />
                {runs.map((rn) => {
                  const x1 = X(m.docs[rn.from * m.B]);
                  const x2 = X(lastDoc(m, rn.to));
                  return rn.st === 2 ? (
                    <rect key={`d${rn.from}`} x={x1} y={y + 5} width={Math.max(1.2, x2 - x1)} height={15} fill={m.color} opacity={0.2} />
                  ) : (
                    <rect key={`s${rn.from}`} x={x1} y={y + 18.5} width={Math.max(1.2, x2 - x1)} height={2} fill={m.color} />
                  );
                })}
                {Array.from(ticks).map((v, bi) => (v ? <rect key={`t${bi}`} x={LANE_X + (bi * LANE_W) / BINS} y={y + 5} width={Math.max(1, LANE_W / BINS)} height={15} fill={m.color} /> : null))}
                {isLast && la && la.from >= 0 && la.to < NO_MORE && X(la.to) - X(la.from) > 2 ? (
                  <path d={`M ${X(la.from)} ${y + 5} Q ${(X(la.from) + X(la.to)) / 2} ${y - 7} ${X(la.to)} ${y + 5}`} fill="none" stroke="var(--viz-ink)" strokeWidth={1.3} />
                ) : null}
                {c >= 0 && c < NO_MORE ? <path d={`M ${X(c)} ${y + 21} l -4 7 h 8 z`} fill="var(--viz-ink)" /> : null}
                {READ_W > 0 ? (
                  <text x={LANE_X + LANE_W + 8} y={y + 16} fontSize={10} fill="var(--viz-ink-2)">
                    {c < 0 ? 'not started' : c >= NO_MORE ? 'exhausted' : `at ${fmtNum(c)}`}
                  </text>
                ) : null}
              </g>
            );
          })}
          {(() => {
            const y = 12 + nL * laneH;
            return (
              <g>
                <text x={0} y={y + 14} fontSize={11} fill="var(--viz-ink)">
                  matches
                </text>
                <line x1={LANE_X} x2={LANE_X + LANE_W} y1={y + 20} y2={y + 20} stroke="var(--viz-grid)" />
                {Array.from(r.matchBins).map((v, bi) => (v ? <rect key={`m${bi}`} x={LANE_X + (bi * LANE_W) / BINS} y={y + 4} width={Math.max(1, LANE_W / BINS)} height={9} fill="var(--viz-good)" /> : null))}
                {Array.from(r.rejectBins).map((v, bi) => (v ? <rect key={`r${bi}`} x={LANE_X + (bi * LANE_W) / BINS} y={y + 14} width={Math.max(1, LANE_W / BINS)} height={6} fill="var(--viz-critical)" /> : null))}
                {READ_W > 0 ? (
                  <text x={LANE_X + LANE_W + 8} y={y + 16} fontSize={10} fill="var(--viz-ink-2)">
                    {fmtNum(r.matches)}
                  </text>
                ) : null}
                {[0, 25_000, 50_000, 75_000, 100_000].map((d) => (
                  <text key={d} x={X(d)} y={y + 38} fontSize={9} textAnchor={d === 0 ? 'start' : d === 100_000 ? 'end' : 'middle'} fill="var(--viz-ink-muted)">
                    {d === 0 ? 'doc 0' : `${d / 1000}k`}
                  </text>
                ))}
              </g>
            );
          })()}
        </svg>
        {query === 'phrase' && r.lastPhrase ? (
          <svg viewBox={`0 0 ${W} 62`} width={W} height={62} role="img" aria-label={`Positions of both terms in doc ${r.lastPhrase.doc}: ${r.lastPhrase.ok ? 'adjacent' : 'not adjacent'}`} style={{ marginTop: 6 }}>
            {(() => {
              const e = r.lastPhrase!;
              const L = docLen(e.doc);
              const PX = (p: number) => LANE_X + (p / L) * LANE_W;
              const w = e.lists.find((l) => l.term === 'write')?.pos ?? [];
              const a = e.lists.find((l) => l.term === 'ahead')?.pos ?? [];
              const pairs = w.filter((p) => a.includes(p + 1));
              return (
                <>
                  <text x={0} y={10} fontSize={10} fill="var(--viz-ink-2)">
                    positions in doc {fmtNum(e.doc)} (0–{L - 1})
                  </text>
                  {[
                    { term: 'write', pos: w, y: 20, color: SERIES[0] },
                    { term: 'ahead', pos: a, y: 42, color: SERIES[1] },
                  ].map((row) => (
                    <g key={row.term}>
                      <text x={0} y={row.y + 10} fontSize={11} fill="var(--viz-ink)">
                        {row.term}
                      </text>
                      <line x1={LANE_X} x2={LANE_X + LANE_W} y1={row.y + 6} y2={row.y + 6} stroke="var(--viz-grid)" />
                      {row.pos.map((p) => (
                        <rect key={p} x={PX(p) - 1.5} y={row.y} width={3} height={12} fill={row.color} />
                      ))}
                    </g>
                  ))}
                  {pairs.map((p) => (
                    <rect key={`pair${p}`} x={PX(p) - 5} y={17} width={PX(p + 1) - PX(p) + 10} height={39} rx={3} fill="none" stroke="var(--viz-good)" strokeWidth={2} />
                  ))}
                  {READ_W > 0 ? (
                    <text x={LANE_X + LANE_W + 8} y={36} fontSize={10} fill="var(--viz-ink-2)">
                      {e.ok ? 'adjacent' : 'never adjacent'}
                    </text>
                  ) : null}
                </>
              );
            })()}
          </svg>
        ) : null}
        <div style={{ margin: '0.5rem 0 0', fontSize: '0.78rem', display: 'grid', gap: 3 }} data-testid="oplog">
          {recent.map((e, i) => (
            <div key={recentFrom + i} style={{ color: 'var(--viz-ink)', margin: 0 }}>
              <span style={{ color: 'var(--viz-ink-muted)' }}>op {fmtNum(recentFrom + i + 1)} · </span>
              {describe(run, e)}
            </div>
          ))}
        </div>
      </div>
    </VizPanel>
  );
}
