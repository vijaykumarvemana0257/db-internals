import { useMemo, useState, type KeyboardEvent } from 'react';
import { VizPanel, Segmented, Choice, Slider, Button, Legend, Stats, Note, fmtNum } from './Viz';

/**
 * Extendible and linear hashing, replayed from the list of inserted keys (so every render is deterministic).
 *
 * Hash: MurmurHash3's 32-bit finalizer (fmix32) applied to an integer key. Any well-mixed hash behaves the same.
 *
 * Extendible hashing follows Fagin, Nievergelt, Pippenger and Strong (ACM TODS 1979):
 * - a directory of 2^G pointers indexed by the FIRST G bits of the hash (the paper's choice);
 * - each bucket has a local depth d′ and is pointed to by 2^(G−d′) adjacent directory entries;
 * - a full bucket with d′ < G splits on bit d′+1 and only its directory entries are rewired;
 * - a full bucket with d′ = G first doubles the directory (G+1), then splits;
 * - after a split the key is inserted again, so one insert can split (and double) repeatedly.
 * Model assumption: the directory stops at MAX_GLOBAL_DEPTH = 10 (1,024 entries). A full bucket that already uses 10
 * bits cannot split, so it chains an overflow page — the only way identical hashes can be stored.
 *
 * Linear hashing follows Litwin (VLDB 1980) and PostgreSQL's hash index:
 * - N initial buckets, level i, split pointer `next`; bucket count M = N·2^i + next;
 * - h_i(k) = hash mod N·2^i reads the LOW bits; address = h_i(k), or h_{i+1}(k) if h_i(k) < next;
 * - a split always takes bucket `next` (not the bucket that overflowed), moves the keys h_{i+1} sends to next + N·2^i,
 *   and advances the pointer; when it reaches N·2^i the level goes up and the pointer returns to 0;
 * - triggers: never (static hashing), on every insert into a full bucket (Litwin's uncontrolled splits), or when
 *   keys > threshold × capacity × M (controlled; PostgreSQL's hashm_ntuples > hashm_ffactor × (hashm_maxbucket + 1)).
 * Model assumptions: at most one split per insert (as in PostgreSQL); a bucket's pages = ceil(keys / capacity), so an
 * overflow page appears as soon as a bucket holds more keys than one page can.
 */

export type Scheme = 'extendible' | 'linear';
export type Pattern = 'random' | 'skewed' | 'duplicate';
export type Trigger = 'never' | 'overflow' | 'load';

export const MAX_KEYS = 48;
export const MAX_GLOBAL_DEPTH = 10;
export const DEFAULT_KEYS = 10;
export const DUPLICATE_KEY = 42;
export const SKEW_BITS = 4;
const MAX_LINEAR_BUCKETS = 128;

export function hash32(key: number): number {
  let h = key >>> 0;
  h ^= h >>> 16;
  h = Math.imul(h, 0x85ebca6b) >>> 0;
  h ^= h >>> 13;
  h = Math.imul(h, 0xc2b2ae35) >>> 0;
  h ^= h >>> 16;
  return h >>> 0;
}

/** The first n bits of a 32-bit hash, as an integer. */
export const leadingBits = (h: number, n: number) => (n <= 0 ? 0 : h >>> (32 - n));
/** The last n bits of a 32-bit hash, as an integer. */
export const lowBits = (h: number, n: number) => (n <= 0 ? 0 : h % 2 ** n);
export const bitString = (v: number, width: number) => (width <= 0 ? '' : v.toString(2).padStart(width, '0'));
export const hashBits = (h: number) => h.toString(2).padStart(32, '0');
const log2 = (n: number) => Math.round(Math.log2(n));
const pagesFor = (keys: number, capacity: number) => Math.max(1, Math.ceil(keys / capacity));

/* ------------------------------------------------------------------ key streams */

function isSkewed(h: number, scheme: Scheme) {
  return scheme === 'extendible' ? leadingBits(h, SKEW_BITS) === 0 : lowBits(h, SKEW_BITS) === 0;
}

/** Next key of a pattern that has not been used yet (duplicates always return the same key). */
export function nextPatternKey(pattern: Pattern, scheme: Scheme, used: Set<number>): number {
  if (pattern === 'duplicate') return DUPLICATE_KEY;
  if (pattern === 'random') {
    for (let j = 0; j < 100_000; j++) {
      const k = 1000 + (hash32(j * 7919 + 17) % 9000);
      if (!used.has(k)) return k;
    }
    return -1;
  }
  for (let k = 1; k < 5_000_000; k++) if (!used.has(k) && isSkewed(hash32(k), scheme)) return k;
  return -1;
}

export function patternKeys(pattern: Pattern, scheme: Scheme, count: number): number[] {
  const out: number[] = [];
  const used = new Set<number>();
  for (let i = 0; i < Math.min(count, MAX_KEYS); i++) {
    const k = nextPatternKey(pattern, scheme, used);
    out.push(k);
    used.add(k);
  }
  return out;
}

/* ------------------------------------------------------------------ extendible hashing */

export type ExtBucket = { id: number; localDepth: number; keys: number[] };
export type ExtEvent = {
  key: number;
  hash: number;
  depthBefore: number;
  depthAfter: number;
  doublings: number;
  splits: number;
  /** Local depth of the bucket before each split this insert performed. */
  splitDepths: number[];
  /** Keys moved into new sibling buckets by this insert's splits. */
  moved: number;
  /** Bucket ids that were split or created by this insert. */
  touched: number[];
  landed: number;
  /** True when the key had to go on an overflow page because the bucket could not split any further. */
  overflow: boolean;
};
export type ExtState = { globalDepth: number; dir: number[]; buckets: ExtBucket[]; events: ExtEvent[]; doublings: number; splits: number };

export function extendibleReplay(keys: number[], capacity: number): ExtState {
  let G = 0;
  let dir: number[] = [0];
  const buckets: ExtBucket[] = [{ id: 0, localDepth: 0, keys: [] }];
  const events: ExtEvent[] = [];
  let doublings = 0;
  let splits = 0;
  for (const key of keys.slice(0, MAX_KEYS)) {
    const h = hash32(key);
    const ev: ExtEvent = { key, hash: h, depthBefore: G, depthAfter: G, doublings: 0, splits: 0, splitDepths: [], moved: 0, touched: [], landed: -1, overflow: false };
    for (let guard = 0; guard <= 2 * MAX_GLOBAL_DEPTH + 2; guard++) {
      const b = buckets[dir[leadingBits(h, G)]];
      if (b.keys.length < capacity) {
        b.keys.push(key);
        ev.landed = b.id;
        break;
      }
      if (b.localDepth >= MAX_GLOBAL_DEPTH) {
        b.keys.push(key);
        ev.landed = b.id;
        ev.overflow = true;
        break;
      }
      if (b.localDepth === G) {
        // Double the directory: with prefix addressing, entry j becomes entries 2j and 2j+1.
        const nd: number[] = new Array(dir.length * 2);
        for (let j = 0; j < dir.length; j++) {
          nd[2 * j] = dir[j];
          nd[2 * j + 1] = dir[j];
        }
        dir = nd;
        G++;
        ev.doublings++;
        doublings++;
      }
      // Split on the next bit: hash bit d′ (0-based from the top).
      const d = b.localDepth;
      const sib: ExtBucket = { id: buckets.length, localDepth: d + 1, keys: [] };
      buckets.push(sib);
      b.localDepth = d + 1;
      const stay: number[] = [];
      for (const k of b.keys) {
        if ((hash32(k) >>> (31 - d)) & 1) sib.keys.push(k);
        else stay.push(k);
      }
      b.keys = stay;
      for (let j = 0; j < dir.length; j++) if (dir[j] === b.id && (j >>> (G - 1 - d)) & 1) dir[j] = sib.id;
      ev.moved += sib.keys.length;
      ev.splits++;
      splits++;
      ev.splitDepths.push(d);
      if (!ev.touched.includes(b.id)) ev.touched.push(b.id);
      ev.touched.push(sib.id);
    }
    ev.depthAfter = G;
    events.push(ev);
  }
  return { globalDepth: G, dir, buckets, events, doublings, splits };
}

export type ExtRow = { bucket: ExtBucket; start: number; span: number; prefix: string; pages: number };

/** Buckets in directory order, each with the contiguous run of directory entries that point to it. */
export function extendibleView(s: ExtState, capacity: number) {
  const rows: ExtRow[] = [];
  let j = 0;
  while (j < s.dir.length) {
    const b = s.buckets[s.dir[j]];
    const span = 2 ** (s.globalDepth - b.localDepth);
    rows.push({ bucket: b, start: j, span, prefix: bitString(j >>> (s.globalDepth - b.localDepth), b.localDepth), pages: pagesFor(b.keys.length, capacity) });
    j += span;
  }
  const keys = rows.reduce((a, r) => a + r.bucket.keys.length, 0);
  const pages = rows.reduce((a, r) => a + r.pages, 0);
  const invariantOk = s.globalDepth === rows.reduce((a, r) => Math.max(a, r.bucket.localDepth), 0) && rows.every((r) => r.bucket.keys.every((k) => bitString(leadingBits(hash32(k), r.bucket.localDepth), r.bucket.localDepth) === r.prefix));
  return {
    rows,
    keys,
    buckets: rows.length,
    directory: s.dir.length,
    overflowPages: pages - rows.length,
    avgChain: pages / rows.length,
    load: keys / (rows.length * capacity),
    invariantOk,
  };
}

export function keyForExtendibleBucket(s: ExtState, bucketId: number, used: Set<number>): number {
  const b = s.buckets[bucketId];
  const start = s.dir.indexOf(bucketId);
  const prefix = start >>> (s.globalDepth - b.localDepth);
  for (let k = 1; k < 5_000_000; k++) if (!used.has(k) && leadingBits(hash32(k), b.localDepth) === prefix) return k;
  return -1;
}

/* ------------------------------------------------------------------ linear hashing */

export type LinSplit = { from: number; to: number; moved: number; pagesBefore: number; pagesAfter: number; levelUp: boolean };
export type LinEvent = {
  key: number;
  hash: number;
  levelBefore: number;
  nextBefore: number;
  /** h_i(key) before the pointer comparison. */
  first: number;
  bucket: number;
  usedNextLevel: boolean;
  keysBefore: number;
  /** The key did not fit on the bucket's primary page. */
  overflow: boolean;
  split: LinSplit | null;
  loadAfterInsert: number;
};
export type LinState = { N: number; level: number; next: number; buckets: number[][]; events: LinEvent[]; splits: number };

/** Litwin's addressing: h_i, then h_{i+1} for buckets the pointer has already split this round. */
export function linearAddress(h: number, N: number, level: number, next: number) {
  const lo = N * 2 ** level;
  const first = h % lo;
  if (first < next) return { first, bucket: h % (2 * lo), usedNextLevel: true };
  return { first, bucket: first, usedNextLevel: false };
}

/** PostgreSQL's _hash_hashkey2bucket: bucket = hash & highmask; if bucket > maxbucket, bucket &= lowmask. */
export function postgresHashkey2bucket(h: number, maxbucket: number, highmask: number, lowmask: number) {
  let bucket = (h & highmask) >>> 0;
  if (bucket > maxbucket) bucket = (bucket & lowmask) >>> 0;
  return bucket;
}

export function linearReplay(keys: number[], capacity: number, N: number, trigger: Trigger, threshold: number): LinState {
  let level = 0;
  let next = 0;
  let total = 0;
  let splits = 0;
  const buckets: number[][] = Array.from({ length: N }, () => []);
  const events: LinEvent[] = [];
  for (const key of keys.slice(0, MAX_KEYS)) {
    const h = hash32(key);
    const a = linearAddress(h, N, level, next);
    const keysBefore = buckets[a.bucket].length;
    buckets[a.bucket].push(key);
    total++;
    const M = buckets.length;
    const ev: LinEvent = { key, hash: h, levelBefore: level, nextBefore: next, first: a.first, bucket: a.bucket, usedNextLevel: a.usedNextLevel, keysBefore, overflow: keysBefore >= capacity, split: null, loadAfterInsert: total / (M * capacity) };
    // Integer comparison in percent, so a load exactly at the threshold never splits because of float rounding.
    const wantSplit = trigger === 'overflow' ? keysBefore >= capacity : trigger === 'load' ? total * 100 > Math.round(threshold * 100) * capacity * M : false;
    if (wantSplit && M < MAX_LINEAR_BUCKETS) {
      const lo = N * 2 ** level;
      const from = next;
      const to = next + lo;
      const old = buckets[from];
      const stay: number[] = [];
      const move: number[] = [];
      for (const k of old) (hash32(k) % (2 * lo) === from ? stay : move).push(k);
      buckets[from] = stay;
      buckets.push(move);
      next++;
      let levelUp = false;
      if (next === lo) {
        level++;
        next = 0;
        levelUp = true;
      }
      ev.split = { from, to, moved: move.length, pagesBefore: pagesFor(old.length, capacity), pagesAfter: pagesFor(stay.length, capacity), levelUp };
      splits++;
    }
    events.push(ev);
  }
  return { N, level, next, buckets, events, splits };
}

export function linearView(s: LinState, capacity: number) {
  const lo = s.N * 2 ** s.level;
  const baseBits = log2(lo);
  const rows = s.buckets.map((keys, b) => {
    const hi = b < s.next || b >= lo;
    const bits = hi ? baseBits + 1 : baseBits;
    return { bucket: b, keys, hi, bits, label: bitString(b, bits), pages: pagesFor(keys.length, capacity) };
  });
  const keys = rows.reduce((a, r) => a + r.keys.length, 0);
  const pages = rows.reduce((a, r) => a + r.pages, 0);
  const invariantOk = rows.every((r) => r.keys.every((k) => linearAddress(hash32(k), s.N, s.level, s.next).bucket === r.bucket));
  return {
    rows,
    lo,
    baseBits,
    keys,
    buckets: rows.length,
    overflowPages: pages - rows.length,
    avgChain: pages / rows.length,
    longest: rows.reduce((a, r) => Math.max(a, r.pages), 0),
    load: keys / (rows.length * capacity),
    invariantOk,
  };
}

export function keyForLinearBucket(s: LinState, bucket: number, used: Set<number>): number {
  for (let k = 1; k < 5_000_000; k++) if (!used.has(k) && linearAddress(hash32(k), s.N, s.level, s.next).bucket === bucket) return k;
  return -1;
}

/* ------------------------------------------------------------------ narration */

const plural = (n: number, word: string, many = `${word}s`) => `${fmtNum(n)} ${n === 1 ? word : many}`;

export function describeExtendible(s: ExtState, capacity: number): string {
  const ev = s.events[s.events.length - 1];
  if (!ev) return 'The directory has one entry (global depth 0) pointing at one empty bucket. Insert a key, or click a bucket to insert a key that hashes into it.';
  const G = s.globalDepth;
  if (ev.overflow) {
    const doubled = ev.doublings > 0 ? `after doubling the directory ${plural(ev.doublings, 'time')} ` : '';
    const b = s.buckets[ev.landed];
    if (b.keys.every((k) => hash32(k) === ev.hash)) {
      return `Key ${ev.key} landed in a full bucket whose keys all have exactly the same hash. No split can separate identical bits, so ${doubled}the lab stops at global depth ${MAX_GLOBAL_DEPTH} and chains an overflow page.`;
    }
    return `Key ${ev.key} landed in a full bucket whose keys all share their first ${MAX_GLOBAL_DEPTH} hash bits. A real directory would keep doubling until a later bit separated them; ${doubled}the lab stops at its cap of global depth ${MAX_GLOBAL_DEPTH} and chains an overflow page instead.`;
  }
  if (ev.doublings > 0) {
    return `Key ${ev.key} hit a full bucket whose local depth equalled the global depth ${ev.depthBefore}, so the directory doubled${ev.doublings > 1 ? ` ${fmtNum(ev.doublings)} times, to ${fmtNum(2 ** G)} entries,` : ` to ${fmtNum(2 ** G)} entries`} (global depth ${G}) before the split succeeded. ${ev.splits > 1 ? `${plural(ev.splits, 'split')} ran because the first ${ev.doublings > 1 ? 'ones sent' : 'one sent'} every key the same way.` : `Only the one overflowing bucket was split; every other bucket page is untouched, now pointed at by twice as many entries.`}`;
  }
  if (ev.splits > 0) {
    const d = ev.splitDepths[0];
    return `Key ${ev.key} hit a full bucket with local depth ${d}, below the global depth ${G}. It split into two buckets of local depth ${d + 1} and ${plural(ev.moved, 'key')} moved to the new one. The ${fmtNum(2 ** (G - d))} directory entries that shared it now divide between the two; the directory did not grow.`;
  }
  const b = s.buckets[ev.landed];
  return `Key ${ev.key}: the first ${G} hash bits pick directory entry ${bitString(leadingBits(ev.hash, G), G) || '(the only one)'}, which points at a bucket with local depth ${b.localDepth}. It had a free slot (capacity ${capacity}), so nothing else changed.`;
}

export function describeLinear(s: LinState, capacity: number, trigger: Trigger, threshold: number): string {
  const ev = s.events[s.events.length - 1];
  if (!ev) return `${plural(s.N, 'bucket')}, level 0, split pointer at bucket 0. Insert a key, or click a bucket to insert a key that hashes into it.`;
  const lo = s.N * 2 ** ev.levelBefore;
  const addr = ev.usedNextLevel
    ? `h${ev.levelBefore} gives bucket ${ev.first}, which the pointer (at ${ev.nextBefore}) has already split this round, so h${ev.levelBefore + 1} decides: bucket ${ev.bucket}.`
    : `h${ev.levelBefore} gives bucket ${ev.bucket}.`;
  const placed = ev.overflow ? `Its primary page was full, so the key went onto ${ev.keysBefore % capacity === 0 ? 'a new overflow page' : 'an overflow page'}.` : 'It fit on the primary page.';
  if (!ev.split) {
    const why = trigger === 'never' ? 'Static hashing never splits: chains only grow.' : trigger === 'load' ? `Load ${fmtNum(ev.loadAfterInsert * 100)}% is not above ${fmtNum(threshold * 100)}%, so no split.` : 'The bucket had room, so no split.';
    return `Key ${ev.key}: ${addr} ${placed} ${why}`;
  }
  const sp = ev.split;
  const other = ev.overflow && sp.from !== ev.bucket && sp.to !== ev.bucket ? ` The pointer, not the overflowing bucket, chooses what splits: bucket ${ev.bucket} keeps its chain until the pointer reaches it.` : '';
  const wrap = sp.levelUp ? ` That finished round ${ev.levelBefore}: all ${fmtNum(2 * lo)} buckets now use h${ev.levelBefore + 1}, and the pointer returns to 0.` : ` The pointer moves to ${sp.from + 1}.`;
  return `Key ${ev.key}: ${addr} ${placed} ${trigger === 'load' ? `Load passed ${fmtNum(threshold * 100)}%` : 'An overflow occurred'}, so bucket ${sp.from} split into ${sp.from} and ${sp.to} with h${ev.levelBefore + 1}, moving ${plural(sp.moved, 'key')} (${plural(sp.pagesBefore, 'page')} → ${plural(sp.pagesAfter, 'page')}).${other}${wrap}`;
}

/* ------------------------------------------------------------------ rendering helpers */

const BUCKET_X = 196;
const EXT_LABEL_W = 80;
const LIN_LABEL_W = 8;
const MAX_OVERFLOW_BOXES = 3;

/** SVG width: room for the widest bucket, its visible overflow pages and the "+k more" note. */
function figureWidth(labelW: number, capacity: number, maxPages: number) {
  const shown = Math.min(Math.max(0, maxPages - 1), MAX_OVERFLOW_BOXES);
  const tail = shown > 0 ? 14 + shown * 72 + (maxPages - 1 > shown ? 70 : 0) : 0;
  return Math.max(560, BUCKET_X + labelW + capacity * cellWidth(capacity) + 6 + tail + 12);
}

function cellWidth(capacity: number) {
  return capacity <= 3 ? 60 : capacity === 4 ? 50 : 42;
}
function bitsThatFit(capacity: number) {
  return capacity <= 3 ? 6 : capacity === 4 ? 5 : 4;
}

function onActivate(fn: () => void) {
  return {
    onClick: fn,
    onKeyDown: (e: KeyboardEvent) => {
      if (e.key === 'Enter' || e.key === ' ') {
        e.preventDefault();
        fn();
      }
    },
  };
}

function BitsLine({ x, y, hash, emphasis, from, label }: { x: number; y: number; hash: number; emphasis: number; from: 'start' | 'end'; label: string }) {
  const bits = hashBits(hash);
  const shown = from === 'start' ? bits.slice(0, 16) : bits.slice(16);
  const groups: { text: string; strong: boolean }[] = [];
  for (let i = 0; i < shown.length; i++) {
    const pos = from === 'start' ? i : shown.length - 1 - i;
    const strong = pos < emphasis;
    const ch = shown[i] + (i % 4 === 3 && i < shown.length - 1 ? ' ' : '');
    const last = groups[groups.length - 1];
    if (last && last.strong === strong) last.text += ch;
    else groups.push({ text: ch, strong });
  }
  return (
    <text x={x} y={y} fontSize={12}>
      <tspan fill="var(--viz-ink-2)">{label} </tspan>
      {from === 'end' ? <tspan fill="var(--viz-ink-2)">… </tspan> : null}
      {groups.map((g, i) => (
        <tspan key={i} fill={g.strong ? 'var(--viz-ink)' : 'var(--viz-ink-muted)'} fontWeight={g.strong ? 700 : 400}>
          {g.text}
        </tspan>
      ))}
      {from === 'start' ? <tspan fill="var(--viz-ink-2)"> …</tspan> : null}
    </text>
  );
}

function Pages({ x, yMid, keys, capacity, cellH, lastKey, lastIsNew, bitsOf, labelW, label, sublabel }: { x: number; yMid: number; keys: number[]; capacity: number; cellH: number; lastKey: number | null; lastIsNew: boolean; bitsOf: (k: number) => string; labelW: number; label: string; sublabel?: string }) {
  const cw = cellWidth(capacity);
  const primaryW = labelW + capacity * cw + 6;
  const pages = pagesFor(keys.length, capacity);
  const showBits = cellH >= 26;
  const y = yMid - cellH / 2;
  const lastIndex = lastIsNew && lastKey !== null ? keys.lastIndexOf(lastKey) : -1;
  const overflowShown = Math.min(pages - 1, MAX_OVERFLOW_BOXES);
  // Overflow boxes stay tall enough for their "k keys" label even when rows are packed tightly.
  const ovH = Math.max(cellH - 2, 14);
  const ovSmall = ovH < 20;
  return (
    <g>
      {sublabel && cellH >= 24 ? (
        <>
          <text x={x + 6} y={yMid - 2} fontSize={label.length > 8 ? 9 : 10} fill="var(--viz-ink)">
            {label}
          </text>
          <text x={x + 6} y={yMid + 10} fontSize={10} fill="var(--viz-ink-2)">
            {sublabel}
          </text>
        </>
      ) : sublabel ? (
        <text x={x + 6} y={yMid + 4} fontSize={10} fill="var(--viz-ink-2)">
          {sublabel}
        </text>
      ) : (
        <text x={x + 6} y={yMid + 4} fontSize={10} fill="var(--viz-ink)">
          {label}
        </text>
      )}
      {Array.from({ length: capacity }, (_, i) => {
        const k = keys[i];
        const cx = x + labelW + i * cw;
        const isLast = i === lastIndex;
        return (
          <g key={i}>
            <rect x={cx + 1} y={y + 1} width={cw - 4} height={cellH - 2} rx={3} fill={k === undefined ? 'none' : 'var(--viz-plane)'} stroke={isLast ? 'var(--viz-1)' : 'var(--viz-border)'} strokeWidth={isLast ? 2.5 : 1} strokeDasharray={k === undefined ? '3 2' : undefined} />
            {k !== undefined ? (
              <>
                <text x={cx + cw / 2 - 1} y={showBits ? y + cellH / 2 - 1 : y + cellH / 2 + 4} textAnchor="middle" fontSize={11} fill="var(--viz-ink)">
                  {k}
                </text>
                {showBits ? (
                  <text x={cx + cw / 2 - 1} y={y + cellH / 2 + 10} textAnchor="middle" fontSize={9} fill="var(--viz-ink-2)">
                    {bitsOf(k)}
                  </text>
                ) : null}
              </>
            ) : null}
          </g>
        );
      })}
      {Array.from({ length: overflowShown }, (_, p) => {
        const ox = x + primaryW + 14 + p * 72;
        const count = Math.min(capacity, keys.length - (p + 1) * capacity);
        const holdsLast = lastIndex >= (p + 1) * capacity && lastIndex < (p + 2) * capacity;
        return (
          <g key={`o${p}`}>
            <line x1={ox - 14} x2={ox} y1={yMid} y2={yMid} stroke="var(--viz-serious)" strokeWidth={1.5} />
            <rect x={ox} y={yMid - ovH / 2} width={58} height={ovH} rx={3} fill="var(--viz-surface)" stroke={holdsLast ? 'var(--viz-1)' : 'var(--viz-serious)'} strokeWidth={holdsLast ? 2.5 : 1.5} strokeDasharray={holdsLast ? undefined : '4 2'} />
            <text x={ox + 29} y={yMid + (ovSmall ? 3 : 4)} textAnchor="middle" fontSize={ovSmall ? 9 : 10} fill="var(--viz-ink)">
              {plural(count, 'key')}
            </text>
          </g>
        );
      })}
      {pages - 1 > overflowShown ? (
        <text x={x + primaryW + 14 + overflowShown * 72} y={yMid + 4} fontSize={10} fill="var(--viz-ink-2)">
          +{fmtNum(pages - 1 - overflowShown)} more
        </text>
      ) : null}
    </g>
  );
}

/* ------------------------------------------------------------------ component */

const patternOptions = (scheme: Scheme) =>
  [
    { value: 'random', label: 'Distinct random keys' },
    { value: 'skewed', label: scheme === 'extendible' ? `Skewed: hashes share their first ${SKEW_BITS} bits` : `Skewed: hashes share their last ${SKEW_BITS} bits` },
    { value: 'duplicate', label: `One key value, over and over (${DUPLICATE_KEY})` },
  ] as const;

export default function ExtendibleLinearHashLab() {
  const [scheme, setScheme] = useState<Scheme>('extendible');

  const [extPattern, setExtPattern] = useState<Pattern>('random');
  const [extKeys, setExtKeys] = useState<number[]>(() => patternKeys('random', 'extendible', DEFAULT_KEYS));
  const [extCap, setExtCap] = useState(3);

  const [linPattern, setLinPattern] = useState<Pattern>('random');
  const [linKeys, setLinKeys] = useState<number[]>(() => patternKeys('random', 'linear', DEFAULT_KEYS));
  const [linCap, setLinCap] = useState(3);
  const [linN, setLinN] = useState<'1' | '2' | '4'>('2');
  const [trigger, setTrigger] = useState<Trigger>('load');
  const [thresholdPct, setThresholdPct] = useState(75);

  const ext = useMemo(() => extendibleReplay(extKeys, extCap), [extKeys, extCap]);
  const ev = useMemo(() => extendibleView(ext, extCap), [ext, extCap]);
  const N = Number(linN);
  const lin = useMemo(() => linearReplay(linKeys, linCap, N, trigger, thresholdPct / 100), [linKeys, linCap, N, trigger, thresholdPct]);
  const lv = useMemo(() => linearView(lin, linCap), [lin, linCap]);

  const isExt = scheme === 'extendible';
  const keys = isExt ? extKeys : linKeys;
  const setKeys = isExt ? setExtKeys : setLinKeys;
  const pattern = isExt ? extPattern : linPattern;
  const full = keys.length >= MAX_KEYS;

  const insertNext = (count: number) => {
    const out = keys.slice();
    const used = new Set(out);
    for (let i = 0; i < count && out.length < MAX_KEYS; i++) {
      const k = nextPatternKey(pattern, scheme, used);
      if (k < 0) break;
      out.push(k);
      used.add(k);
    }
    setKeys(out);
  };
  const choosePattern = (p: Pattern) => {
    if (isExt) setExtPattern(p);
    else setLinPattern(p);
    setKeys(patternKeys(p, scheme, keys.length));
  };
  const insertInto = (target: number) => {
    if (full) return;
    const used = new Set(keys);
    const k = isExt ? keyForExtendibleBucket(ext, target, used) : keyForLinearBucket(lin, target, used);
    if (k > 0) setKeys([...keys, k]);
  };

  const lastExt = ext.events[ext.events.length - 1];
  const lastLin = lin.events[lin.events.length - 1];

  /* ---------------- extendible figure geometry */
  const G = ext.globalDepth;
  const perEntry = 2 ** G <= 16;
  const nb = ev.buckets;
  const extBase = nb <= 12 ? 36 : Math.max(22, Math.floor(480 / nb));
  const extBands = ev.rows.map((r) => (perEntry ? Math.max(extBase, r.span * 18) : extBase));
  const headerH = 58;
  const extTop = headerH + 22;
  const extYs: number[] = [];
  extBands.reduce((y, h) => {
    extYs.push(y);
    return y + h;
  }, extTop);
  const extH = extTop + extBands.reduce((a, b) => a + b, 0) + 10;
  const extW = figureWidth(EXT_LABEL_W, extCap, ev.rows.reduce((a, r) => Math.max(a, r.pages), 1));
  const extCellH = Math.min(30, extBase - 6);
  const extBitsN = Math.max(1, Math.min(bitsThatFit(extCap), Math.max(G, 1)));
  const lastDirEntry = lastExt ? leadingBits(lastExt.hash, G) : -1;
  const extLastLanded = lastExt ? lastExt.landed : -1;

  /* ---------------- linear figure geometry */
  const M = lv.buckets;
  const linBand = M <= 12 ? 38 : Math.max(22, Math.floor(500 / M));
  const linTop = headerH + 22;
  const linH = linTop + M * linBand + 14;
  const linW = figureWidth(LIN_LABEL_W, linCap, lv.longest);
  const linCellH = Math.min(30, linBand - 6);
  const linBitsN = Math.max(1, Math.min(bitsThatFit(linCap), lv.baseBits + 1));

  const extTable = (
    <table className="viz-table">
      <thead>
        <tr>
          <th>Bucket (hash prefix)</th>
          <th>Local depth d′</th>
          <th>Directory entries</th>
          <th>Keys: first 12 hash bits</th>
          <th>Pages</th>
        </tr>
      </thead>
      <tbody>
        {ev.rows.map((r) => (
          <tr key={r.bucket.id}>
            <td>{r.prefix === '' ? '(all)' : `${r.prefix}…`}</td>
            <td>{r.bucket.localDepth}</td>
            <td>{fmtNum(r.span)}</td>
            <td>{r.bucket.keys.length ? r.bucket.keys.map((k) => `${k}: ${hashBits(hash32(k)).slice(0, 12)}`).join(', ') : '—'}</td>
            <td>{r.pages}</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
  const linTable = (
    <table className="viz-table">
      <thead>
        <tr>
          <th>Bucket</th>
          <th>Addressed by</th>
          <th>Keys: last 12 hash bits</th>
          <th>Pages</th>
        </tr>
      </thead>
      <tbody>
        {lv.rows.map((r) => (
          <tr key={r.bucket}>
            <td>
              {r.bucket} ({r.label})
            </td>
            <td>h{r.hi ? lin.level + 1 : lin.level}: low {r.bits} bits</td>
            <td>{r.keys.length ? r.keys.map((k) => `${k}: …${hashBits(hash32(k)).slice(20)}`).join(', ') : '—'}</td>
            <td>{r.pages}</td>
          </tr>
        ))}
      </tbody>
    </table>
  );

  return (
    <VizPanel
      title={isExt ? 'Extendible hashing: a directory over buckets' : trigger === 'never' ? 'Static hashing: a fixed set of buckets and their overflow chains' : 'Linear hashing: buckets split in a fixed order'}
      subtitle={
        isExt
          ? 'The first G bits of each key’s hash index a directory of 2^G pointers. A bucket with local depth d′ is shared by 2^(G−d′) entries. When a bucket overflows it splits; the directory doubles only if the bucket already used all G bits.'
          : trigger === 'never'
            ? 'N buckets, fixed for good: the address is hash mod N. Nothing ever splits, so once a bucket’s primary page is full every further key goes onto an overflow page chained to it.'
            : 'No directory: the bucket count M alone decides the address, using the low bits of the hash. A split pointer walks the buckets in order and splits the one it points at, whichever bucket overflowed; keys wait on overflow pages until the pointer arrives.'
      }
      controls={
        <Segmented
          label="Scheme"
          value={scheme}
          onChange={setScheme}
          options={[
            { value: 'extendible', label: 'Extendible hashing' },
            { value: 'linear', label: 'Linear hashing' },
          ]}
        />
      }
      legend={
        <Legend
          items={
            isExt
              ? [
                  { label: 'Last key inserted, and the directory entry it used', color: 'var(--viz-1)' },
                  { label: 'Bucket split by the last insert', color: 'var(--viz-3)' },
                  { label: 'Directory doubled by the last insert', color: 'var(--viz-7)' },
                  { label: 'Overflow page', color: 'var(--viz-serious)' },
                ]
              : trigger === 'never'
                ? [
                    { label: 'Last key inserted', color: 'var(--viz-1)' },
                    { label: 'Overflow page', color: 'var(--viz-serious)' },
                  ]
                : [
                    { label: 'Last key inserted', color: 'var(--viz-1)' },
                    { label: 'Bucket split by the last insert, and its new bucket', color: 'var(--viz-3)' },
                    { label: 'Split pointer', color: 'var(--viz-7)' },
                    { label: 'Overflow page', color: 'var(--viz-serious)' },
                  ]
          }
        />
      }
      stats={
        isExt ? (
          <Stats
            items={[
              { label: 'Keys', value: fmtNum(ev.keys) },
              { label: 'Global depth G', value: fmtNum(G) },
              { label: 'Directory entries', value: fmtNum(ev.directory), hint: 'Always 2^G. Fits in memory in practice; one address computation finds the entry.' },
              { label: 'Buckets', value: fmtNum(nb) },
              { label: 'Overflow pages', value: fmtNum(ev.overflowPages), hint: `Only when a bucket already uses all ${MAX_GLOBAL_DEPTH} bits the lab allows and still overflows.` },
              { label: 'Avg chain length', value: `${ev.avgChain.toFixed(2)} pages`, hint: 'Pages per bucket, primary plus overflow.' },
              { label: 'Bucket fill', value: `${fmtNum(ev.load * 100)}%`, hint: 'Keys ÷ (buckets × capacity).' },
              { label: 'Doublings / splits', value: `${fmtNum(ext.doublings)} / ${fmtNum(ext.splits)}` },
            ]}
          />
        ) : (
          <Stats
            items={[
              { label: 'Level i · pointer', value: `${lin.level} · ${lin.next}`, hint: `h${lin.level} reads the low ${lv.baseBits} bits; h${lin.level + 1} reads ${lv.baseBits + 1}.` },
              { label: 'Buckets M', value: fmtNum(M), hint: 'M = N·2^i + pointer' },
              { label: 'Directory entries', value: '0', hint: 'Linear hashing needs no directory: the address is computed from M.' },
              { label: 'Overflow pages', value: fmtNum(lv.overflowPages) },
              { label: 'Avg chain length', value: `${lv.avgChain.toFixed(2)} pages`, hint: 'Pages per bucket, primary plus overflow.' },
              { label: 'Longest chain', value: plural(lv.longest, 'page') },
              { label: 'Load', value: `${fmtNum(lv.load * 100)}%`, hint: 'Keys ÷ (M × capacity). The controlled trigger compares this with the threshold.' },
              { label: 'Splits', value: fmtNum(lin.splits) },
            ]}
          />
        )
      }
      note={
        <Note>
          {keys.length === 0 ? null : <strong>{isExt ? `${fmtNum(ev.keys)} keys in ${plural(nb, 'bucket')}, directory of ${fmtNum(ev.directory)}. ` : `${fmtNum(lv.keys)} keys in ${plural(M, 'bucket')}, ${plural(lv.overflowPages, 'overflow page')}. `}</strong>}
          {isExt ? describeExtendible(ext, extCap) : describeLinear(lin, linCap, trigger, thresholdPct / 100)}
          {full ? ` The lab stops at ${MAX_KEYS} keys: Undo or Reset to continue.` : ''}
        </Note>
      }
      table={isExt ? extTable : linTable}
    >
      <div className="viz-controls">
        <Choice label="Keys" value={pattern} onChange={choosePattern} options={patternOptions(scheme)} />
        {isExt ? (
          <Slider label="Bucket capacity" min={2} max={6} value={extCap} onChange={setExtCap} format={(v) => `${v} keys`} />
        ) : (
          <>
            <Slider label="Page capacity" min={2} max={6} value={linCap} onChange={setLinCap} format={(v) => `${v} keys`} />
            <Segmented
              label="Initial buckets N"
              value={linN}
              onChange={setLinN}
              options={[
                { value: '1', label: '1' },
                { value: '2', label: '2' },
                { value: '4', label: '4' },
              ]}
            />
            <Choice
              label="Split when"
              value={trigger}
              onChange={setTrigger}
              options={[
                { value: 'load', label: 'Load exceeds a threshold (controlled)' },
                { value: 'overflow', label: 'Any insert overflows (uncontrolled)' },
                { value: 'never', label: 'Never (static hashing)' },
              ]}
            />
            {trigger === 'load' ? <Slider label="Load threshold" min={50} max={100} step={5} value={thresholdPct} onChange={setThresholdPct} format={(v) => `${v}%`} /> : null}
          </>
        )}
      </div>
      <div className="viz-controls">
        <Button primary onClick={() => insertNext(1)} disabled={full}>
          Insert next key
        </Button>
        <Button onClick={() => insertNext(10)} disabled={full}>
          Insert 10 keys
        </Button>
        <Button onClick={() => setKeys(keys.slice(0, -1))} disabled={keys.length === 0}>
          Undo
        </Button>
        <Button onClick={() => setKeys(patternKeys(pattern, scheme, DEFAULT_KEYS))}>Reset</Button>
      </div>

      {isExt ? (
        <svg viewBox={`0 0 ${extW} ${extH}`} width={extW} height={extH} role="img" aria-label={`Extendible hashing: global depth ${G}, ${ev.directory} directory entries, ${nb} buckets, ${ev.overflowPages} overflow pages`}>
          {lastExt ? (
            <>
              <text x={8} y={18} fontSize={12} fill="var(--viz-ink)">
                Key {lastExt.key}
                {lastExt.doublings > 0 ? ` · directory doubled ${lastExt.doublings > 1 ? `${lastExt.doublings}× ` : ''}to ${fmtNum(2 ** G)} entries` : lastExt.splits > 0 ? ' · local split, directory unchanged' : ''}
                {lastExt.overflow ? ' · overflow page' : lastExt.splits === 0 ? ' · fits' : ''}
              </text>
              <BitsLine x={8} y={38} hash={lastExt.hash} emphasis={G} from="start" label="hash:" />
              <text x={8} y={54} fontSize={11} fill="var(--viz-ink-2)">
                {G === 0 ? 'Global depth 0: one directory entry for every key.' : `First ${G} bit${G === 1 ? '' : 's'} (bold) → directory entry ${bitString(lastDirEntry, G)}.`} Click a bucket to insert into it.
              </text>
            </>
          ) : (
            <text x={8} y={24} fontSize={12} fill="var(--viz-ink)">
              No keys yet. Insert a key, or click the bucket.
            </text>
          )}
          <text x={8} y={extTop - 8} fontSize={11} fill="var(--viz-ink)" fontWeight={600}>
            Directory · G = {G} · {fmtNum(ev.directory)} entr{ev.directory === 1 ? 'y' : 'ies'}
          </text>
          <text x={BUCKET_X} y={extTop - 8} fontSize={11} fill="var(--viz-ink)" fontWeight={600}>
            Buckets · local depth d′ · capacity {extCap}
          </text>
          {ev.rows.map((r, i) => {
            const y = extYs[i];
            const h = extBands[i];
            const mid = y + h / 2;
            const entryH = h / r.span;
            const doubled = !!lastExt && lastExt.doublings > 0;
            const split = !!lastExt && lastExt.touched.includes(r.bucket.id);
            const cw = cellWidth(extCap);
            const boxW = EXT_LABEL_W + extCap * cw + 6;
            const boxH = Math.min(h - 4, extCellH + 8);
            const holdsLastEntry = lastDirEntry >= r.start && lastDirEntry < r.start + r.span;
            return (
              <g key={r.bucket.id}>
                {entryH >= 14 ? (
                  Array.from({ length: r.span }, (_, e) => {
                    const idx = r.start + e;
                    const used = lastExt && idx === lastDirEntry;
                    return (
                      <g key={e}>
                        <rect x={8} y={y + e * entryH + 1} width={118} height={entryH - 2} rx={3} fill="var(--viz-plane)" stroke={used ? 'var(--viz-1)' : doubled ? 'var(--viz-7)' : 'var(--viz-border)'} strokeWidth={used ? 2.5 : doubled ? 1.5 : 1} />
                        <text x={16} y={y + e * entryH + entryH / 2 + 4} fontSize={11} fill="var(--viz-ink)">
                          {G === 0 ? 'entry 0 (no bits)' : bitString(idx, G)}
                        </text>
                      </g>
                    );
                  })
                ) : (
                  <g>
                    <rect x={8} y={y + 1} width={118} height={h - 2} rx={3} fill="var(--viz-plane)" stroke={holdsLastEntry && lastExt ? 'var(--viz-1)' : doubled ? 'var(--viz-7)' : 'var(--viz-border)'} strokeWidth={holdsLastEntry && lastExt ? 2.5 : doubled ? 1.5 : 1} />
                    <text x={14} y={mid + 4} fontSize={10} fill="var(--viz-ink)">
                      {r.prefix}
                      {'*'.repeat(Math.min(G - r.bucket.localDepth, 10))} ×{fmtNum(r.span)}
                    </text>
                  </g>
                )}
                <line x1={128} x2={BUCKET_X - 6} y1={mid} y2={mid} stroke="var(--viz-ink-muted)" strokeWidth={1} />
                <path d={`M ${BUCKET_X - 6} ${mid - 4} L ${BUCKET_X} ${mid} L ${BUCKET_X - 6} ${mid + 4} Z`} fill="var(--viz-ink-muted)" />
                <g role="button" tabIndex={full ? -1 : 0} aria-label={`Insert a key whose hash starts with ${r.prefix || 'anything'} into this bucket`} style={{ cursor: full ? 'default' : 'pointer' }} {...onActivate(() => insertInto(r.bucket.id))}>
                  <rect x={BUCKET_X} y={mid - boxH / 2} width={boxW} height={boxH} rx={5} fill="var(--viz-surface)" stroke={split ? 'var(--viz-3)' : r.bucket.id === extLastLanded ? 'var(--viz-ink-2)' : 'var(--viz-ink-muted)'} strokeWidth={split ? 2.5 : 1} />
                  <Pages x={BUCKET_X} yMid={mid} keys={r.bucket.keys} capacity={extCap} cellH={Math.min(extCellH, boxH - 6)} lastKey={lastExt ? lastExt.key : null} lastIsNew={r.bucket.id === extLastLanded} bitsOf={(k) => bitString(leadingBits(hash32(k), extBitsN), extBitsN) + (extBitsN < 32 ? '…' : '')} labelW={EXT_LABEL_W} label={r.prefix === '' ? '(all)' : `${r.prefix}…`} sublabel={`d′=${r.bucket.localDepth}`} />
                </g>
              </g>
            );
          })}
        </svg>
      ) : (
        <svg viewBox={`0 0 ${linW} ${linH}`} width={linW} height={linH} role="img" aria-label={`Linear hashing: level ${lin.level}, split pointer ${lin.next}, ${M} buckets, ${lv.overflowPages} overflow pages`}>
          {lastLin ? (
            <>
              <text x={8} y={18} fontSize={12} fill="var(--viz-ink)">
                Key {lastLin.key} → bucket {lastLin.bucket}
                {lastLin.usedNextLevel ? ` (h${lastLin.levelBefore} gave ${lastLin.first}, already split: h${lastLin.levelBefore + 1})` : ` (h${lastLin.levelBefore})`}
                {lastLin.overflow ? ' · overflow page' : ''}
                {lastLin.split ? ` · pointer split ${lastLin.split.from} → ${lastLin.split.from} + ${lastLin.split.to}` : ''}
              </text>
              <BitsLine x={8} y={38} hash={lastLin.hash} emphasis={log2(lin.N * 2 ** lastLin.levelBefore) + (lastLin.usedNextLevel ? 1 : 0)} from="end" label="hash:" />
              <text x={8} y={54} fontSize={11} fill="var(--viz-ink-2)">
                {trigger === 'never' ? 'Bold low bits = bucket number (hash mod N).' : 'Bold low bits = bucket number; “(new)” = added this round.'} Click a bucket to insert into it.
              </text>
            </>
          ) : (
            <text x={8} y={24} fontSize={12} fill="var(--viz-ink)">
              No keys yet. Insert a key, or click a bucket.
            </text>
          )}
          <text x={8} y={linTop - 8} fontSize={11} fill="var(--viz-ink)" fontWeight={600}>
            Bucket · bits · reads
          </text>
          <text x={BUCKET_X} y={linTop - 8} fontSize={11} fill="var(--viz-ink)" fontWeight={600}>
            Primary page → overflow pages · capacity {linCap}
          </text>
          {lv.rows.map((r) => {
            const y = linTop + r.bucket * linBand;
            const mid = y + linBand / 2;
            const split = !!lastLin?.split && (lastLin.split.from === r.bucket || lastLin.split.to === r.bucket);
            const cw = cellWidth(linCap);
            const boxW = LIN_LABEL_W + linCap * cw + 6;
            const boxH = Math.min(linBand - 4, linCellH + 8);
            const isPointer = trigger !== 'never' && r.bucket === lin.next;
            return (
              <g key={r.bucket}>
                {r.bucket === lv.lo && trigger !== 'never' ? (
                  <>
                    <line x1={8} x2={linW - 8} y1={y} y2={y} stroke="var(--viz-ink-muted)" strokeDasharray="5 4" />
                  </>
                ) : null}
                {isPointer ? (
                  <g>
                    <path d={`M 10 ${mid - 7} L 40 ${mid - 7} L 40 ${mid - 11} L 52 ${mid} L 40 ${mid + 11} L 40 ${mid + 7} L 10 ${mid + 7} Z`} fill="var(--viz-surface)" stroke="var(--viz-7)" strokeWidth={2} />
                    <text x={14} y={mid + 4} fontSize={10} fill="var(--viz-ink)">
                      next
                    </text>
                  </g>
                ) : null}
                <text x={60} y={mid + 4} fontSize={11} fill="var(--viz-ink)">
                  {r.bucket}
                </text>
                <text x={84} y={mid + 4} fontSize={10} fill="var(--viz-ink-2)">
                  {r.label || '—'}
                </text>
                <text x={136} y={mid + 4} fontSize={10} fill="var(--viz-ink-2)">
                  {trigger === 'never' ? 'h0' : `h${r.hi ? lin.level + 1 : lin.level}${r.bucket >= lv.lo ? ' (new)' : ''}`}
                </text>
                <g role="button" tabIndex={full ? -1 : 0} aria-label={`Insert a key that hashes into bucket ${r.bucket}`} style={{ cursor: full ? 'default' : 'pointer' }} {...onActivate(() => insertInto(r.bucket))}>
                  <rect x={BUCKET_X} y={mid - boxH / 2} width={boxW} height={boxH} rx={5} fill="var(--viz-surface)" stroke={split ? 'var(--viz-3)' : 'var(--viz-ink-muted)'} strokeWidth={split ? 2.5 : 1} />
                  <Pages x={BUCKET_X} yMid={mid} keys={r.keys} capacity={linCap} cellH={Math.min(linCellH, boxH - 6)} lastKey={lastLin ? lastLin.key : null} lastIsNew={!!lastLin && lastLin.split === null ? lastLin.bucket === r.bucket : r.keys.includes(lastLin ? lastLin.key : -1)} bitsOf={(k) => '…' + bitString(lowBits(hash32(k), linBitsN), linBitsN)} labelW={LIN_LABEL_W} label="" />
                </g>
              </g>
            );
          })}
        </svg>
      )}
    </VizPanel>
  );
}
