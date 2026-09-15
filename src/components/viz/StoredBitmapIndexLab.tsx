import { useDeferredValue, useMemo, useState } from 'react';
import { VizPanel, Segmented, Slider, Check, Legend, Stats, Note, makeRng, fmtNum, fmtBytes } from './Viz';

/**
 * A stored bitmap index: one bitmap per distinct value, compressed, combined word by word.
 *
 * Encodings follow their primary sources:
 * - WAH (Wu, Otoo, Shoshani, LBNL-49626): 32-bit words. MSB 0 = literal carrying 31 bitmap bits, the
 *   first bitmap bit in bit 30. MSB 1 = fill: the next bit is the fill value and the low 30 bits count
 *   31-bit groups. The trailing partial group needs an active word plus a word holding its bit count.
 * - Roaring (CRoaring containers): the row space is cut into 2^16-row chunks; each non-empty chunk is a
 *   run container (2 + 4 bytes per run) if that is no larger than both alternatives, otherwise an array
 *   of 16-bit values (2 bytes each, up to 4,096 values) or an 8 KB bitset. Sizes are container payload
 *   only; per-container headers are ignored.
 * - Locking (Oracle Database Concepts, "Bitmap Storage Structure"): an update of the indexed column needs
 *   exclusive access to the index entries for the old and the new value, which blocks DML on the rows those
 *   entries cover until commit. Oracle splits large bitmaps into pieces with their own rowid ranges; this
 *   lab keeps every value in one piece, so its blocked-row count is the upper bound.
 *
 * Model assumptions: 262,144 rows; region values are equally frequent; channel = 'web' for about a third of
 * rows, independent of region and position.
 */

export const N = 1 << 18;
export const CHUNK = 1 << 16;
const GROUP = 31;
const SEGS = 128;
const SEG_ROWS = N / SEGS;

export type Order = 'clustered' | 'arrival';
export type Op = 'and' | 'or' | 'andnot';

export const CARDINALITIES = [2, 4, 8, 16, 64, 256, 1024];

let arrivalU: Float64Array | null = null;
let webBits: Uint8Array | null = null;
function base() {
  if (!arrivalU || !webBits) {
    const r1 = makeRng(7);
    const r2 = makeRng(99);
    arrivalU = new Float64Array(N);
    webBits = new Uint8Array(N);
    for (let i = 0; i < N; i++) {
      arrivalU[i] = r1();
      webBits[i] = r2() < 1 / 3 ? 1 : 0;
    }
  }
  return { arrivalU, webBits };
}

export function regionColumn(card: number, order: Order) {
  const { arrivalU: u } = base();
  const col = new Uint16Array(N);
  for (let i = 0; i < N; i++) col[i] = order === 'clustered' ? Math.floor((i * card) / N) : Math.min(card - 1, Math.floor(u[i] * card));
  return col;
}

type WahWord = { hex: string; kind: string };

/** Encode a bitmap given as ascending set positions. Returns the word count and the first few words. */
export function wahEncode(positions: ArrayLike<number>, nbits = N, keep = 6) {
  const ngroups = Math.floor(nbits / GROUP);
  const shown: WahWord[] = [];
  let words = 0;
  let fillBit = -1;
  let fillLen = 0;
  const hex = (v: number) => (v >>> 0).toString(16).toUpperCase().padStart(8, '0');
  const pushWord = (v: number, kind: () => string) => {
    words++;
    if (shown.length < keep) shown.push({ hex: hex(v), kind: kind() });
  };
  const literal = () => 'literal';
  const flushFill = () => {
    if (fillLen === 0) return;
    const len = fillLen;
    const bit = fillBit;
    if (len === 1) pushWord(bit ? 0x7fffffff : 0, literal);
    else pushWord((0x80000000 | (bit << 30) | len) >>> 0, () => `${bit}-fill × ${fmtNum(len)} groups`);
    fillBit = -1;
    fillLen = 0;
  };
  const emitGroup = (value: number, count: number) => {
    if (value === 0 || value === 0x7fffffff) {
      const b = value === 0 ? 0 : 1;
      if (fillBit !== b) {
        flushFill();
        fillBit = b;
      }
      fillLen += count;
    } else {
      flushFill();
      pushWord(value, literal);
    }
  };
  let g = 0; // next group not yet emitted
  let cur = 0;
  let curGroup = -1;
  let activeBits = 0;
  for (let i = 0; i < positions.length; i++) {
    const p = positions[i];
    const pg = Math.floor(p / GROUP);
    if (pg >= ngroups) {
      activeBits++;
      continue;
    }
    if (pg !== curGroup) {
      if (curGroup >= 0) {
        emitGroup(cur >>> 0, 1);
        g = curGroup + 1;
      }
      if (pg > g) emitGroup(0, pg - g);
      curGroup = pg;
      cur = 0;
    }
    cur = (cur | (1 << (GROUP - 1 - (p % GROUP)))) >>> 0;
  }
  if (curGroup >= 0) {
    emitGroup(cur >>> 0, 1);
    g = curGroup + 1;
  }
  if (ngroups > g) emitGroup(0, ngroups - g);
  flushFill();
  words += 2; // active word + its bit count
  return { words, bytes: words * 4, shown, activeBits };
}

export type Container = { type: 'run' | 'array' | 'bitset'; card: number; runs: number; bytes: number };

export function roaringEncode(positions: ArrayLike<number>) {
  const containers: (Container | null)[] = [];
  const nchunks = Math.ceil(N / CHUNK);
  const card = new Array(nchunks).fill(0);
  const runs = new Array(nchunks).fill(0);
  let prev = -2;
  for (let i = 0; i < positions.length; i++) {
    const p = positions[i];
    const c = Math.floor(p / CHUNK);
    card[c]++;
    if (p !== prev + 1 || Math.floor(prev / CHUNK) !== c) runs[c]++;
    prev = p;
  }
  let bytes = 0;
  for (let c = 0; c < nchunks; c++) {
    if (card[c] === 0) {
      containers.push(null);
      continue;
    }
    const runBytes = 2 + 4 * runs[c];
    const arrayBytes = 2 * card[c];
    const bitsetBytes = 8192;
    let ct: Container;
    if (runBytes <= Math.min(arrayBytes, bitsetBytes)) ct = { type: 'run', card: card[c], runs: runs[c], bytes: runBytes };
    else if (card[c] <= 4096) ct = { type: 'array', card: card[c], runs: runs[c], bytes: arrayBytes };
    else ct = { type: 'bitset', card: card[c], runs: runs[c], bytes: bitsetBytes };
    containers.push(ct);
    bytes += ct.bytes;
  }
  return { containers, bytes };
}

function positionsOf(bits: Uint8Array) {
  const out: number[] = [];
  for (let i = 0; i < bits.length; i++) if (bits[i]) out.push(i);
  return Int32Array.from(out);
}

function density(bits: Uint8Array) {
  const d = new Float64Array(SEGS);
  for (let i = 0; i < N; i++) if (bits[i]) d[Math.floor(i / SEG_ROWS)]++;
  return Array.from(d, (x) => x / SEG_ROWS);
}

export function runStoredIndex(card: number, order: Order, op: Op) {
  const { webBits: web } = base();
  const region = regionColumn(card, order);
  const target = 1 % card;

  // every value's bitmap, as position lists (counting sort)
  const counts = new Int32Array(card);
  for (let i = 0; i < N; i++) counts[region[i]]++;
  const starts = new Int32Array(card + 1);
  for (let v = 0; v < card; v++) starts[v + 1] = starts[v] + counts[v];
  const fillPos = starts.slice(0, card);
  const all = new Int32Array(N);
  for (let i = 0; i < N; i++) all[fillPos[region[i]]++] = i;
  let wahWords = 0;
  let roaringBytes = 0;
  const containerMix = { run: 0, array: 0, bitset: 0 };
  for (let v = 0; v < card; v++) {
    const pos = all.subarray(starts[v], starts[v + 1]);
    wahWords += wahEncode(pos, N, 0).words;
    const ro = roaringEncode(pos);
    roaringBytes += ro.bytes;
    for (const c of ro.containers) if (c) containerMix[c.type]++;
  }

  const rBits = new Uint8Array(N);
  for (let i = 0; i < N; i++) rBits[i] = region[i] === target ? 1 : 0;
  const res = new Uint8Array(N);
  for (let i = 0; i < N; i++) res[i] = op === 'and' ? rBits[i] & web[i] : op === 'or' ? rBits[i] | web[i] : rBits[i] & (1 - web[i]);

  const describe = (label: string, bits: Uint8Array) => {
    const pos = positionsOf(bits);
    return { label, setBits: pos.length, wah: wahEncode(pos), roaring: roaringEncode(pos), density: density(bits) };
  };
  const bitmaps = [describe(`region = 'r${target}'`, rBits), describe(`channel = 'web'`, web), describe('result', res)];

  // one uncommitted UPDATE moves a row from r<target> to r<other>: both entries are locked
  const other = card > 2 ? 2 : 0;
  const lockedRows = counts[target] + counts[other];

  return {
    card,
    order,
    op,
    target,
    other,
    counts: Array.from(counts),
    uncompressedBytes: (card * N) / 8,
    wahBytes: wahWords * 4,
    roaringBytes,
    containerMix,
    bitmaps,
    lockedRows,
  };
}

export type StoredResult = ReturnType<typeof runStoredIndex>;

const OP_TEXT: Record<Op, string> = { and: 'AND', or: 'OR', andnot: 'AND NOT' };

export default function StoredBitmapIndexLab() {
  const [cardIdx, setCardIdx] = useState(3);
  const [order, setOrder] = useState<Order>('clustered');
  const [op, setOp] = useState<Op>('and');
  const [lock, setLock] = useState(false);
  const inputsNow = useMemo(() => ({ card: CARDINALITIES[cardIdx], order, op }), [cardIdx, order, op]);
  const inputs = useDeferredValue(inputsNow);
  const r = useMemo(() => runStoredIndex(inputs.card, inputs.order, inputs.op), [inputs]);

  const W = 704;
  const LEFT = 104;
  const SEGW = 4;
  const STRIP = 20;
  const ROW = 58;
  const colors = ['var(--viz-1)', 'var(--viz-2)', 'var(--viz-7)'];
  const lockTop = 3 * ROW + 26;
  const MAXKEYS = 12;
  const shownKeys = Math.min(r.card, MAXKEYS);
  const keyW = (W - LEFT - 8) / MAXKEYS;
  const H = lockTop + 62;

  const wahText = r.bitmaps
    .map((b) => `${b.label.padEnd(18)} ${b.wah.shown.map((w) => `${w.hex} (${w.kind})`).join('  ')}${b.wah.words > b.wah.shown.length ? `  … ${fmtNum(b.wah.words)} words` : ''}`)
    .join('\n');

  const query = `region = 'r${r.target}' ${OP_TEXT[r.op]} channel = 'web'`;

  return (
    <VizPanel
      title="A stored bitmap index: one compressed bitmap per value"
      subtitle={`262,144 rows. Every distinct region has its own bitmap, stored compressed. The query ${query} is answered by combining two bitmaps word by word, before any row is read.`}
      controls={
        <>
          <Slider label="Distinct regions" min={0} max={CARDINALITIES.length - 1} value={cardIdx} onChange={setCardIdx} format={(i) => fmtNum(CARDINALITIES[i])} />
          <Segmented
            label="Physical row order"
            value={order}
            onChange={setOrder}
            options={[
              { value: 'clustered', label: 'Loaded sorted by region' },
              { value: 'arrival', label: 'Arrival order' },
            ]}
          />
          <Segmented
            label="Combine"
            value={op}
            onChange={setOp}
            options={[
              { value: 'and', label: 'AND' },
              { value: 'or', label: 'OR' },
              { value: 'andnot', label: 'AND NOT' },
            ]}
          />
          <Check label="Session 1 updates one row's region (uncommitted)" checked={lock} onChange={setLock} />
        </>
      }
      legend={
        <Legend
          items={[
            { label: `Bitmap region = 'r${r.target}' (share of rows set)`, color: colors[0] },
            { label: "Bitmap channel = 'web'", color: colors[1] },
            { label: 'Result bitmap', color: colors[2] },
            ...(lock ? [{ label: 'Index entry locked by session 1', color: 'var(--viz-critical)' }] : []),
          ]}
        />
      }
      stats={
        <Stats
          items={[
            { label: 'Bitmaps in the region index', value: fmtNum(r.card) },
            { label: 'Uncompressed', value: fmtBytes(r.uncompressedBytes), hint: 'One bit per row per distinct value.' },
            { label: 'WAH', value: fmtBytes(r.wahBytes), hint: '32-bit words, including each bitmap’s active word and bit count.' },
            { label: 'Roaring', value: fmtBytes(r.roaringBytes), hint: `Container payload only. ${r.containerMix.run} run, ${r.containerMix.array} array, ${r.containerMix.bitset} bitset containers.` },
            { label: 'Rows in the result', value: fmtNum(r.bitmaps[2].setBits) },
            { label: 'Rows whose DML now waits', value: lock ? fmtNum(r.lockedRows) : '—', hint: 'Rows covered by the two locked index entries (one piece per value in this lab). With a B-tree index, only the updated row is locked.' },
          ]}
        />
      }
      note={
        <Note>
          <strong>
            {r.order === 'clustered'
              ? `Sorted by region, each value is one contiguous run: ${fmtNum(r.card)} bitmaps compress from ${fmtBytes(r.uncompressedBytes)} to ${fmtBytes(r.wahBytes)} of WAH words.`
              : `In arrival order each region bitmap is scattered: ${fmtNum(r.card)} bitmaps take ${fmtBytes(r.wahBytes)} in WAH and ${fmtBytes(r.roaringBytes)} in Roaring, against ${fmtBytes(r.uncompressedBytes)} uncompressed.`}
          </strong>{' '}
          {r.order === 'arrival' && r.card <= 4
            ? 'A value on a quarter or more of the rows leaves almost no all-zero 31-bit group, so WAH stores nearly every group as a literal and costs slightly more than the raw bits. '
            : r.order === 'arrival' && r.card >= 256
              ? 'Sparse bitmaps are where compression pays: WAH spends about two words per set bit, and Roaring stores each chunk as a short array of 16-bit values. '
              : ''}
          {lock
            ? `Session 1 moved one row from r${r.target} to r${r.other}. It holds the index entries for both values until it commits, so DML that touches either bitmap — ${fmtNum(r.lockedRows)} rows here — waits. `
            : 'Tick the update to see what one uncommitted row change locks. '}
        </Note>
      }
      table={
        <table className="viz-table">
          <thead>
            <tr>
              <th>Bitmap</th>
              <th>Rows set</th>
              <th>WAH words</th>
              <th>WAH size</th>
              <th>Roaring containers (per 65,536-row chunk)</th>
              <th>Roaring size</th>
            </tr>
          </thead>
          <tbody>
            {r.bitmaps.map((b) => (
              <tr key={b.label}>
                <td>{b.label}</td>
                <td>{fmtNum(b.setBits)}</td>
                <td>{fmtNum(b.wah.words)}</td>
                <td>{fmtBytes(b.wah.bytes)}</td>
                <td>{b.roaring.containers.map((c) => (c ? `${c.type}(${fmtNum(c.card)})` : 'none')).join(', ')}</td>
                <td>{fmtBytes(b.roaring.bytes)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      }
    >
      <svg viewBox={`0 0 ${W} ${H}`} width={W} height={H} role="img" aria-label={`Bitmaps for ${query}: ${r.bitmaps[2].setBits} rows in the result`}>
        {r.bitmaps.map((b, bi) => {
          const y = 8 + bi * ROW;
          return (
            <g key={b.label}>
              <text x={LEFT - 8} y={y + 14} fontSize={11} textAnchor="end" fill="var(--viz-ink)">
                {bi === 2 ? OP_TEXT[r.op] + ' →' : b.label}
              </text>
              <rect x={LEFT} y={y} width={SEGS * SEGW} height={STRIP} fill="var(--viz-plane)" stroke="var(--viz-border)" />
              {b.density.map((d, i) =>
                d > 0 ? <rect key={i} x={LEFT + i * SEGW} y={y} width={SEGW + 0.6} height={STRIP} fill={colors[bi]} fillOpacity={Math.max(0.12, d)} shapeRendering="crispEdges" /> : null,
              )}
              <text x={LEFT + SEGS * SEGW + 8} y={y + 14} fontSize={10} fill="var(--viz-ink-2)">
                {fmtNum(b.setBits)} rows
              </text>
              {b.roaring.containers.map((c, ci) => (
                <g key={ci}>
                  {ci > 0 ? <line x1={LEFT + ci * (SEGS / 4) * SEGW} x2={LEFT + ci * (SEGS / 4) * SEGW} y1={y - 3} y2={y + STRIP + 16} stroke="var(--viz-ink-muted)" strokeDasharray="3 2" /> : null}
                  <text x={LEFT + ci * (SEGS / 4) * SEGW + 4} y={y + STRIP + 13} fontSize={9.5} fill="var(--viz-ink-2)">
                    {c ? `${c.type} · ${fmtBytes(c.bytes)}` : 'no container'}
                  </text>
                </g>
              ))}
            </g>
          );
        })}
        <text x={LEFT - 8} y={3 * ROW + 6} fontSize={10} textAnchor="end" fill="var(--viz-ink-muted)">
          row 0
        </text>
        <text x={LEFT + SEGS * SEGW} y={3 * ROW + 6} fontSize={10} textAnchor="end" fill="var(--viz-ink-muted)">
          row 262,143 · dashed lines: 65,536-row Roaring chunks
        </text>

        <text x={LEFT - 8} y={lockTop + 14} fontSize={11} textAnchor="end" fill="var(--viz-ink)">
          index entries
        </text>
        {Array.from({ length: shownKeys }, (_, v) => {
          const isLocked = lock && (v === r.target || v === r.other);
          const x = LEFT + v * keyW;
          const last = r.card > MAXKEYS && v === MAXKEYS - 1;
          return (
            <g key={v}>
              <rect x={x + 1} y={lockTop} width={keyW - 3} height={34} rx={3} fill="var(--viz-surface)" stroke={isLocked ? 'var(--viz-critical)' : 'var(--viz-border)'} strokeWidth={isLocked ? 2.5 : 1} />
              <text x={x + keyW / 2} y={lockTop + 14} fontSize={10} textAnchor="middle" fill="var(--viz-ink)">
                {last ? '…' : `r${v}`}
              </text>
              <text x={x + keyW / 2} y={lockTop + 27} fontSize={10} textAnchor="middle" fill="var(--viz-ink-2)">
                {last ? `+${fmtNum(r.card - (MAXKEYS - 1))} more` : fmtNum(r.counts[v])}
              </text>
            </g>
          );
        })}
        <text x={LEFT} y={lockTop + 52} fontSize={10} fill="var(--viz-ink-2)">
          {lock ? `locked: r${r.target} and r${r.other} — ${fmtNum(r.lockedRows)} rows' index changes wait for session 1's commit` : 'each entry: a key and its bitmap; the number is rows with that value'}
        </text>
      </svg>
      <pre style={{ margin: '0.5rem 0 0', fontSize: '0.72rem', lineHeight: 1.45, overflowX: 'auto', color: 'var(--viz-ink)' }} aria-label="First WAH words of each bitmap">
        {`First WAH words (hex)\n${wahText}`}
      </pre>
    </VizPanel>
  );
}
