import { useDeferredValue, useMemo, useState } from 'react';
import { VizPanel, Segmented, Choice, Check, Slider, Button, Legend, Stats, Note, fmtNum, fmtTime } from './Viz';
import { genSample, genJsonText, delta8, scramble, lz4Compress, utf8, pointReadUs, type SampleKind, type Lz4Seq } from './blockCompressionModel';
import { FIXTURE_SIZES, FIXTURE_BLOCK_SIZES, THROUGHPUT_DEFAULTS, type FixtureBlockSize, type FixtureCodec, type FixtureKind } from './blockCompressionFixture';

/**
 * A 16 KB block compression bench.
 *
 * "Inside an LZ4 block" runs a port of LZ4_compress_generic() (lz4 1.10.0, the independent-block byU16
 * path that LZ4_compress_default takes for inputs under 64 KB) on the block in the browser, so edits change
 * the real sequence stream: tokens, literal runs, 2-byte offsets and match lengths. Its output was checked
 * byte-for-byte against the lz4 1.10.0 CLI.
 *
 * Everything else is a clearly labelled fixture (blockCompressionFixture.ts): zstd 1.5.6 levels, lz4 -9,
 * zlib level 6 and a trained zstd dictionary, measured offline on the unedited sample blocks at five block
 * sizes. The read-cost view is an explicit model: cold point read = latency + compressed bytes / read speed
 * + whole block / decompression speed, with the throughput constants editable on screen (defaults measured
 * on one laptop). A block that saves less than 12.5% is stored raw, which is RocksDB's default
 * max_compressed_bytes_per_kb = 896.
 */

type View = 'block' | 'cost';
const BLOCK = 16384;
const DEFAULT_JSON = genJsonText(BLOCK);

const CODECS: { id: FixtureCodec; label: string; family: 'lz4' | 'zstd' | 'zlib' }[] = [
  { id: 'lz4', label: 'LZ4 (level 1)', family: 'lz4' },
  { id: 'lz4hc', label: 'LZ4HC -9', family: 'lz4' },
  { id: 'zstd1', label: 'ZSTD -1', family: 'zstd' },
  { id: 'zstd3', label: 'ZSTD -3', family: 'zstd' },
  { id: 'zstd9', label: 'ZSTD -9', family: 'zstd' },
  { id: 'zstd19', label: 'ZSTD -19', family: 'zstd' },
  { id: 'zstd3dict', label: 'ZSTD -3 + dictionary', family: 'zstd' },
  { id: 'zlib6', label: 'zlib -6', family: 'zlib' },
];
const codecLabel = (id: FixtureCodec | 'none') => (id === 'none' ? 'No compression' : (CODECS.find((c) => c.id === id)?.label ?? id));

export const SAMPLE_LABEL: Record<SampleKind, string> = {
  timestamps: 'INT64 timestamp column chunk',
  json: 'JSON rows',
  random: 'Random bytes (encrypted / already compressed)',
};

const SAMPLE_NOUN: Record<SampleKind, string> = { timestamps: 'INT64 timestamps', json: 'JSON rows', random: 'random bytes' };

export function fixtureKind(sample: SampleKind, delta: boolean): FixtureKind {
  return sample === 'timestamps' ? (delta ? 'timestamps-delta' : 'timestamps') : sample;
}

/** Bytes a block occupies on disk under a codec, applying the optional "not worth it, store raw" rule. */
export function storedBytes(blockBytes: number, compressed: number, rawRule: boolean) {
  const limit = rawRule ? (blockBytes * 896) / 1024 : blockBytes;
  return compressed > limit || compressed >= blockBytes ? { bytes: blockBytes, raw: true } : { bytes: compressed, raw: false };
}

export type CostRow = { id: FixtureCodec | 'none'; stored: number; raw: boolean; io: number; cpu: number; total: number; compressSecPerGB: number };

export function costTable(kind: FixtureKind, blockBytes: FixtureBlockSize, tp: Record<FixtureCodec, { c: number; d: number }>, latencyUs: number, readMBps: number, rawRule: boolean): CostRow[] {
  const rows: CostRow[] = [];
  const none = pointReadUs(blockBytes, blockBytes, null, latencyUs, readMBps);
  rows.push({ id: 'none', stored: blockBytes, raw: true, io: none.io, cpu: 0, total: none.total, compressSecPerGB: 0 });
  for (const c of CODECS) {
    const cell = FIXTURE_SIZES[kind][blockBytes][c.id];
    const s = storedBytes(blockBytes, cell.bytes, rawRule);
    const t = pointReadUs(blockBytes, s.bytes, s.raw ? null : tp[c.id].d, latencyUs, readMBps);
    rows.push({ id: c.id, stored: s.bytes, raw: s.raw, io: t.io, cpu: t.cpu, total: t.total, compressSecPerGB: 1000 / tp[c.id].c });
  }
  return rows;
}

const printable = (b: Uint8Array, from: number, len: number, max = 22) => {
  let s = '';
  for (let i = from; i < from + Math.min(len, max); i++) {
    const c = b[i];
    s += c >= 32 && c < 127 ? String.fromCharCode(c) : `\\x${c.toString(16).padStart(2, '0')}`;
  }
  return len > max ? `${s}…` : s;
};
const hex2 = (n: number) => `0x${n.toString(16).padStart(2, '0')}`;

export default function BlockCompressionBench() {
  const [view, setView] = useState<View>('block');
  const [sample, setSample] = useState<SampleKind>('json');
  const [delta, setDelta] = useState(false);
  const [scr, setScr] = useState(0);
  const [jsonText, setJsonText] = useState(DEFAULT_JSON);
  const [seqPick, setSeqPick] = useState(0);
  const [blockSize, setBlockSize] = useState<FixtureBlockSize>(16384);
  const [latencyUs, setLatencyUs] = useState(100);
  const [readMBps, setReadMBps] = useState(1500);
  const [cached, setCached] = useState(false);
  const [tp, setTp] = useState(THROUGHPUT_DEFAULTS);

  const dText = useDeferredValue(jsonText);
  const dScr = useDeferredValue(scr);

  const block = useMemo(() => {
    const base = sample === 'json' ? utf8(dText.slice(0, BLOCK)).subarray(0, BLOCK) : genSample(sample, BLOCK);
    const filtered = sample === 'timestamps' && delta ? delta8(base) : base;
    return scramble(filtered, dScr / 100);
  }, [sample, delta, dText, dScr]);
  const res = useMemo(() => lz4Compress(block), [block]);
  const kind = fixtureKind(sample, delta);
  const edited = scr > 0 || (sample === 'json' && jsonText !== DEFAULT_JSON);
  const ratio = block.length / Math.max(1, res.out.length);
  const matchSeqs = res.seqs.length - 1;
  const seqIdx = Math.min(seqPick, Math.max(0, res.seqs.length - 1));
  const sel: Lz4Seq | undefined = res.seqs[seqIdx];

  const costRows = useMemo(() => costTable(kind, blockSize, tp, cached ? 0 : latencyUs, cached ? Infinity : readMBps, true), [kind, blockSize, tp, latencyUs, readMBps, cached]);

  /* ------------------------------------------------ byte map geometry */
  const W = 680;
  const LEFT = 44;
  const PER_ROW = 1024;
  const nRows = Math.max(1, Math.ceil(block.length / PER_ROW));
  const ROW_H = 11;
  const ROW_GAP = 4;
  const px = (W - LEFT - 8) / PER_ROW;
  const rowY = (r: number) => 8 + r * (ROW_H + ROW_GAP);
  const mapH = rowY(nRows) + 6;

  const paths = useMemo(() => {
    const lit: string[] = [];
    const mat: string[] = [];
    const add = (arr: string[], start: number, len: number) => {
      let s = start;
      let remaining = len;
      while (remaining > 0) {
        const r = Math.floor(s / PER_ROW);
        const inRow = Math.min(remaining, PER_ROW - (s % PER_ROW));
        const x = LEFT + (s % PER_ROW) * px;
        arr.push(`M${x.toFixed(2)} ${rowY(r)}h${Math.max(0.6, inRow * px).toFixed(2)}v${ROW_H}h${(-Math.max(0.6, inRow * px)).toFixed(2)}z`);
        s += inRow;
        remaining -= inRow;
      }
    };
    for (const q of res.seqs) {
      if (q.litLen) add(lit, q.litStart, q.litLen);
      if (q.matchLen) add(mat, q.matchStart, q.matchLen);
    }
    return { lit: lit.join(''), mat: mat.join('') };
  }, [res, px]);

  const outline = (start: number, len: number) => {
    const out: { x: number; y: number; w: number }[] = [];
    let s = start;
    let remaining = len;
    while (remaining > 0) {
      const r = Math.floor(s / PER_ROW);
      const inRow = Math.min(remaining, PER_ROW - (s % PER_ROW));
      out.push({ x: LEFT + (s % PER_ROW) * px, y: rowY(r), w: Math.max(2, inRow * px) });
      s += inRow;
      remaining -= inRow;
    }
    return out;
  };

  const inspect = res.seqs.slice(seqIdx, seqIdx + 4);

  /* ------------------------------------------------ codec bars (block view) */
  const fixtureRows = CODECS.map((c) => ({ ...c, cell: FIXTURE_SIZES[kind][16384][c.id] }));
  const barMax = BLOCK * 1.01;
  const BAR_LEFT = 190;
  const barW = (bytes: number) => (Math.min(bytes, barMax) / barMax) * (W - BAR_LEFT - 110);
  const barsH = 18 + (fixtureRows.length + 1) * 20;

  /* ------------------------------------------------ cost view geometry */
  const maxTotal = Math.max(...costRows.map((r) => r.total));
  const COST_LEFT = 150;
  const colA = 180; // width of the stored-size column
  const costH = 24 + costRows.length * 24;
  const timeX0 = COST_LEFT + colA + 70;
  const timeW = W - timeX0 - 70;
  const best = costRows.slice(1).reduce((a, b) => (b.stored < a.stored ? b : a));
  const fastest = costRows.slice(1).reduce((a, b) => (b.total < a.total ? b : a));
  const z3 = FIXTURE_SIZES[kind][blockSize].zstd3.bytes;
  const zd = FIXTURE_SIZES[kind][blockSize].zstd3dict.bytes;
  const dictGain = 1 - zd / z3;
  const lz4Row = costRows.find((r) => r.id === 'lz4')!;
  const z19Row = costRows.find((r) => r.id === 'zstd19')!;

  const sampleControls = (
    <>
      <Choice
        label="Sample block"
        value={sample}
        onChange={(v) => {
          setSample(v);
          setSeqPick(0);
        }}
        options={[
          { value: 'json', label: SAMPLE_LABEL.json },
          { value: 'timestamps', label: SAMPLE_LABEL.timestamps },
          { value: 'random', label: SAMPLE_LABEL.random },
        ]}
      />
      {sample === 'timestamps' ? <Check label="Delta(8) first, as in CODEC(Delta, …)" checked={delta} onChange={setDelta} /> : null}
    </>
  );

  return (
    <VizPanel
      title={view === 'block' ? 'Inside an LZ4 block' : 'Codec, block size and the cost of one point read'}
      subtitle={
        view === 'block'
          ? 'LZ4 compresses this 16 KB block live, in your browser: every byte ends up either as a literal copied verbatim or inside a match that repeats earlier bytes. Edit the block and watch the sequence stream change. The other codecs’ sizes are a measured fixture for the unedited block.'
          : 'A reader that wants one row must fetch and decompress its whole block. Pick a block size, edit the throughput constants, and see the storage/CPU/latency trade-off that decides block size. Blocks that save less than 12.5% are stored raw, as RocksDB does. Sizes are measured; times are a model.'
      }
      controls={
        <>
          <Segmented
            label="View"
            value={view}
            onChange={setView}
            options={[
              { value: 'block', label: 'Inside an LZ4 block' },
              { value: 'cost', label: 'Block size and read cost' },
            ]}
          />
          {sampleControls}
          {view === 'block' ? (
            <Slider label="Overwrite bytes with random data" min={0} max={100} step={1} value={scr} onChange={setScr} format={(n) => `${n}%`} />
          ) : (
            <>
              <Segmented
                label="Block size"
                value={String(blockSize)}
                onChange={(v) => setBlockSize(Number(v) as FixtureBlockSize)}
                options={FIXTURE_BLOCK_SIZES.map((b) => ({ value: String(b), label: `${b / 1024} KB` }))}
              />
              <Check label="Compressed block already in the OS page cache" checked={cached} onChange={setCached} />
              <Slider label="Random read latency" min={10} max={2000} step={10} value={latencyUs} onChange={setLatencyUs} format={(n) => (cached ? 'n/a (cached)' : `${n} µs`)} disabled={cached} />
              <Slider label="Read throughput" min={100} max={7000} step={100} value={readMBps} onChange={setReadMBps} format={(n) => (cached ? 'n/a (cached)' : `${fmtNum(n)} MB/s`)} disabled={cached} />
            </>
          )}
        </>
      }
      legend={
        view === 'block' ? (
          <Legend
            items={[
              { label: 'Literal bytes (stored verbatim)', color: 'var(--viz-2)' },
              { label: 'Bytes covered by a match (2-byte offset + length)', color: 'var(--viz-1)' },
              { label: 'LZ4, computed live on this block', color: 'var(--viz-3)' },
              { label: 'Measured fixture, unedited block', color: 'var(--viz-4)' },
            ]}
          />
        ) : (
          <Legend
            items={[
              { label: 'Stored bytes per block', color: 'var(--viz-5)' },
              { label: 'I/O: latency + compressed bytes', color: 'var(--viz-6)' },
              { label: 'CPU: decompress the whole block', color: 'var(--viz-7)' },
            ]}
          />
        )
      }
      stats={
        view === 'block' ? (
          <Stats
            items={[
              { label: 'Block', value: `${fmtNum(block.length)} B` },
              { label: 'LZ4 output', value: `${fmtNum(res.out.length)} B`, hint: 'A compressed block the caller must store with its size; engines keep raw bytes instead when this does not pay.' },
              { label: 'Ratio', value: `${fmtNum(ratio, 2)}×` },
              { label: 'Sequences', value: fmtNum(res.seqs.length), hint: 'Each sequence = token + literals + (offset + match length), except the last, which is literals only.' },
              { label: 'Literal bytes', value: `${fmtNum((100 * res.literalBytes) / Math.max(1, block.length), 1)}%` },
              { label: 'Tokens + lengths + offsets', value: `${fmtNum(res.tokenBytes + res.lengthBytes + res.offsetBytes)} B`, hint: 'Per-sequence overhead. LZ4 has no entropy coder, so this never shrinks below 3 bytes per match.' },
            ]}
          />
        ) : (
          <Stats
            items={[
              { label: 'Smallest block', value: best.raw ? 'none: every codec stores it raw' : `${codecLabel(best.id)}: ${fmtNum(blockSize / best.stored, 2)}×` },
              { label: cached ? 'Fastest read, block in page cache' : 'Fastest cold read', value: `${codecLabel(fastest.id)}: ${fmtTime(fastest.total * 1000)}` },
              { label: 'Trained dictionary vs ZSTD -3', value: `${dictGain >= 0 ? '−' : '+'}${fmtNum(Math.abs(dictGain) * 100, 0)}% bytes`, hint: 'Size of zstd -3 with a 16 KB trained dictionary relative to zstd -3 alone, at this block size.' },
              { label: 'LZ4 read: decompress share', value: lz4Row.total > 0 ? `${fmtNum((100 * lz4Row.cpu) / lz4Row.total, 0)}%` : '—' },
              { label: 'ZSTD -19: CPU to compress 1 GB', value: `${fmtNum(z19Row.compressSecPerGB, 0)} s`, hint: '1000 MB divided by the compress throughput constant.' },
            ]}
          />
        )
      }
      note={
        <Note>
          {view === 'block' ? (
            sample === 'random' || scr >= 60 ? (
              <>
                <strong>
                  {fmtNum(res.literalBytes)} of {fmtNum(block.length)} bytes are literals; the output is {res.out.length >= block.length ? `${fmtNum(res.out.length - block.length)} bytes larger than the input` : `${fmtNum(ratio, 2)}× smaller`}.
                </strong>{' '}
                Random bytes contain almost no repeated 4-byte sequence for the hash table to find, so LZ4 emits long literal runs plus their length bytes.{' '}
                {sample === 'random'
                  ? 'No codec or level fixes that: every measured ZSTD frame for this block is 10 to 14 bytes larger than the input. RocksDB (max_compressed_bytes_per_kb) and InnoDB page compression both notice when compression does not pay and write the raw bytes instead.'
                  : 'The more of the block is noise, the less any codec can do; the fixture bars describe the clean block, not this one.'}
              </>
            ) : (
              <>
                <strong>
                  {fmtNum(matchSeqs)} matches cover {fmtNum((100 * res.matchedBytes) / Math.max(1, block.length), 1)}% of the block; LZ4 writes {fmtNum(res.out.length)} bytes ({fmtNum(ratio, 2)}×).
                </strong>{' '}
                {sample === 'timestamps' && !delta
                  ? 'Only the constant high bytes of each 8-byte timestamp repeat, so LZ4 alternates short literals with short matches. Tick Delta(8): the deltas are small numbers with six zero bytes each, and the same block becomes long matches.'
                  : sample === 'timestamps' && delta
                    ? `After Delta(8) almost everything is a match, yet LZ4 still spends at least 3 bytes per sequence — ${fmtNum(res.tokenBytes + res.offsetBytes)} bytes of tokens and offsets. ZSTD entropy-codes those sequence fields with FSE and gets the same block to ${fmtNum(FIXTURE_SIZES['timestamps-delta'][16384].zstd1.bytes)} bytes at level 1.`
                    : `Keys like "user":"u_ repeat in every row, so matches dominate; literals are mostly digits that change row to row. ZSTD's Huffman stage squeezes those literals and its FSE stage the match fields, which is why it reaches ${fmtNum(FIXTURE_SIZES.json[16384].zstd3.bytes)} bytes at level 3 on the unedited block.`}
                {edited ? ' The fixture bars below still describe the unedited block.' : ''}
              </>
            )
          ) : (
            <>
              <strong>
                {blockSize / 1024} KB blocks of {SAMPLE_NOUN[sample]}
                {sample === 'timestamps' && delta ? ' after Delta(8)' : ''}: LZ4 stores {fmtNum(lz4Row.stored)} B and reads in {fmtTime(lz4Row.total * 1000)}; ZSTD -19 stores {fmtNum(z19Row.stored)} B and reads in {fmtTime(z19Row.total * 1000)}.
              </strong>{' '}
              {z19Row.raw && lz4Row.raw
                ? 'Nothing compresses, so every codec stores the raw block and pays no decompression.'
                : `Decompression is linear in the block, not the row: a point read here decompresses ${fmtNum(blockSize)} bytes to return one record. `}
              {!(z19Row.raw && lz4Row.raw) && sample !== 'random'
                ? dictGain > 0.05
                  ? `The trained dictionary cuts ZSTD -3's output by ${fmtNum(dictGain * 100, 0)}% here — small blocks have little history of their own.`
                  : 'The trained dictionary buys nothing at this size: the block has become its own dictionary.'
                : ''}
            </>
          )}
        </Note>
      }
      table={
        view === 'block' ? (
          <table className="viz-table">
            <thead>
              <tr>
                <th>Codec</th>
                <th>Source</th>
                <th>Bytes for this 16 KB block</th>
                <th>Ratio</th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <td>LZ4 (level 1)</td>
                <td>live, current block</td>
                <td>{fmtNum(res.out.length)}</td>
                <td>{fmtNum(ratio, 2)}×</td>
              </tr>
              {fixtureRows.map((r) => (
                <tr key={r.id}>
                  <td>{r.label}</td>
                  <td>fixture, unedited block</td>
                  <td>{r.cell.raw ? `${fmtNum(BLOCK)} (stored raw)` : fmtNum(r.cell.bytes)}</td>
                  <td>{fmtNum(BLOCK / (r.cell.raw ? BLOCK : r.cell.bytes), 2)}×</td>
                </tr>
              ))}
            </tbody>
          </table>
        ) : (
          <table className="viz-table">
            <thead>
              <tr>
                <th>Codec</th>
                {FIXTURE_BLOCK_SIZES.map((b) => (
                  <th key={b}>{b / 1024} KB block</th>
                ))}
                <th>{cached ? 'Cached' : 'Cold'} read at {blockSize / 1024} KB</th>
              </tr>
            </thead>
            <tbody>
              {CODECS.map((c) => (
                <tr key={c.id}>
                  <td>{c.label}</td>
                  {FIXTURE_BLOCK_SIZES.map((b) => {
                    const cell = FIXTURE_SIZES[kind][b][c.id];
                    return <td key={b}>{cell.raw ? 'raw' : `${fmtNum(cell.bytes)} (${fmtNum(b / cell.bytes, 2)}×)`}</td>;
                  })}
                  <td>{fmtTime((costRows.find((r) => r.id === c.id)?.total ?? 0) * 1000)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        )
      }
    >
      {view === 'block' ? (
        <>
          <svg viewBox={`0 0 ${W} ${mapH}`} width={W} height={mapH} role="img" aria-label={`Byte map of the block: ${fmtNum(res.literalBytes)} literal bytes and ${fmtNum(res.matchedBytes)} matched bytes in ${fmtNum(res.seqs.length)} LZ4 sequences`}>
            {Array.from({ length: nRows }, (_, r) => (
              <g key={r}>
                <rect x={LEFT} y={rowY(r)} width={PER_ROW * px} height={ROW_H} fill="var(--viz-plane)" />
                {r % 4 === 0 ? (
                  <text x={LEFT - 6} y={rowY(r) + 9} textAnchor="end" fontSize={10}>
                    {r} KB
                  </text>
                ) : null}
              </g>
            ))}
            <path d={paths.lit} fill="var(--viz-2)" />
            <path d={paths.mat} fill="var(--viz-1)" />
            {sel && sel.matchLen > 0
              ? outline(sel.matchStart - sel.offset, sel.matchLen).map((o, i) => <rect key={`s${i}`} x={o.x - 1} y={o.y - 2} width={o.w + 2} height={ROW_H + 4} fill="none" stroke="var(--viz-ink-2)" strokeWidth={1.5} strokeDasharray="3 2" />)
              : null}
            {sel
              ? outline(sel.litStart, sel.litLen + sel.matchLen || 1).map((o, i) => <rect key={`q${i}`} x={o.x - 1} y={o.y - 2} width={o.w + 2} height={ROW_H + 4} fill="none" stroke="var(--viz-ink)" strokeWidth={2} />)
              : null}
          </svg>

          <div className="viz-controls" style={{ marginTop: 8 }}>
            <Slider label="Inspect sequence" min={0} max={Math.max(0, res.seqs.length - 1)} value={seqIdx} onChange={setSeqPick} format={(n) => `#${n + 1} of ${fmtNum(res.seqs.length)}`} />
          </div>
          <div style={{ overflowX: 'auto' }}>
            <table className="viz-table" style={{ marginTop: 0, fontSize: '0.76rem', whiteSpace: 'nowrap' }}>
              <thead>
                <tr>
                  <th>#</th>
                  <th>Token</th>
                  <th>Literal length</th>
                  <th>Literal bytes</th>
                  <th>Offset</th>
                  <th>Match length</th>
                </tr>
              </thead>
              <tbody>
                {inspect.map((q, i) => {
                  const litNib = Math.min(15, q.litLen);
                  const mlNib = q.matchLen ? Math.min(15, q.matchLen - 4) : 0;
                  const extraLit = q.litLen >= 15 ? Math.floor((q.litLen - 15) / 255) + 1 : 0;
                  const extraMl = q.matchLen && q.matchLen - 4 >= 15 ? Math.floor((q.matchLen - 19) / 255) + 1 : 0;
                  return (
                    <tr key={seqIdx + i} style={{ fontWeight: i === 0 ? 600 : 400 }}>
                      <td>{seqIdx + i + 1}</td>
                      <td style={{ fontFamily: 'var(--sl-font-mono, monospace)' }}>{hex2((litNib << 4) | mlNib)}</td>
                      <td>{q.litLen >= 15 ? `15 + ${q.litLen - 15} (${extraLit} extra byte${extraLit > 1 ? 's' : ''})` : q.litLen}</td>
                      <td style={{ fontFamily: 'var(--sl-font-mono, monospace)' }}>{q.litLen ? printable(block, q.litStart, q.litLen, 14) : '—'}</td>
                      <td>{q.matchLen ? fmtNum(q.offset) : 'none: last sequence'}</td>
                      <td>{q.matchLen ? `4 + ${q.matchLen - 4} = ${q.matchLen}${extraMl ? ` (${extraMl} extra byte${extraMl > 1 ? 's' : ''})` : ''}` : '—'}</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>

          <svg viewBox={`0 0 ${W} ${barsH}`} width={W} height={barsH} role="img" aria-label="Compressed size of the 16 KB block under each codec" style={{ marginTop: 10 }}>
            <text x={10} y={12} fontSize={11} fill="var(--viz-ink-2)">
              Bytes for this 16 KB block (bar = share of 16,384)
            </text>
            {[{ id: 'live', label: 'LZ4 (live, this block)', bytes: res.out.length, raw: false, live: true }, ...fixtureRows.map((r) => ({ id: r.id, label: r.label, bytes: r.cell.raw ? BLOCK : r.cell.bytes, raw: !!r.cell.raw, live: false }))].map((r, i) => {
              const y = 20 + i * 20;
              const dim = !r.live && edited;
              return (
                <g key={r.id} opacity={dim ? 0.45 : 1}>
                  <text x={10} y={y + 12} fontSize={11} fill="var(--viz-ink)">
                    {r.label}
                  </text>
                  <rect x={BAR_LEFT} y={y + 2} width={Math.max(2, barW(r.bytes))} height={13} rx={2} fill={r.live ? 'var(--viz-3)' : 'var(--viz-4)'} />
                  <text x={BAR_LEFT + Math.max(2, barW(r.bytes)) + 6} y={y + 12} fontSize={11} fill="var(--viz-ink-2)">
                    {r.raw ? 'stored raw' : `${fmtNum(r.bytes)} B · ${fmtNum(BLOCK / r.bytes, 2)}×`}
                  </text>
                </g>
              );
            })}
          </svg>

          {sample === 'json' ? (
            <details style={{ marginTop: 8 }}>
              <summary style={{ fontSize: '0.8rem', color: 'var(--viz-ink-2)', cursor: 'pointer' }}>Edit the block’s JSON rows ({fmtNum(jsonText.length)} of 16,384 bytes)</summary>
              <textarea
                aria-label="Block contents: JSON rows"
                value={jsonText}
                maxLength={BLOCK}
                onChange={(e) => setJsonText(e.currentTarget.value)}
                rows={6}
                spellCheck={false}
                style={{ fontFamily: 'var(--sl-font-mono, monospace)', fontSize: '0.72rem', background: 'var(--viz-plane)', color: 'var(--viz-ink)', border: '1px solid var(--viz-border)', borderRadius: 6, padding: 6, width: '100%', boxSizing: 'border-box', marginTop: 6 }}
              />
              <Button onClick={() => setJsonText(DEFAULT_JSON)} disabled={jsonText === DEFAULT_JSON}>
                Restore the original rows
              </Button>
            </details>
          ) : null}
        </>
      ) : (
        <>
          <svg viewBox={`0 0 ${W} ${costH}`} width={W} height={costH} role="img" aria-label={`Stored size and cold point-read time for ${blockSize / 1024} KB blocks under each codec`}>
            <text x={COST_LEFT} y={12} fontSize={11}>
              Stored per {blockSize / 1024} KB block
            </text>
            <text x={timeX0} y={12} fontSize={11}>
              {cached ? 'Point read, block in page cache (model)' : 'Cold point read (model)'}
            </text>
            {costRows.map((r, i) => {
              const y = 22 + i * 24;
              const sw = (r.stored / blockSize) * colA;
              const scale = timeW / Math.max(1, maxTotal);
              return (
                <g key={r.id}>
                  <text x={10} y={y + 13} fontSize={11} fill="var(--viz-ink)">
                    {codecLabel(r.id)}
                  </text>
                  <rect x={COST_LEFT} y={y + 3} width={colA} height={14} fill="var(--viz-plane)" />
                  <rect x={COST_LEFT} y={y + 3} width={Math.max(1.5, sw)} height={14} fill="var(--viz-5)" />
                  <text x={COST_LEFT + colA + 6} y={y + 14} fontSize={10} fill="var(--viz-ink-2)">
                    {r.raw ? (r.id === 'none' ? '1.00×' : 'raw') : `${fmtNum(blockSize / r.stored, 2)}×`}
                  </text>
                  <rect x={timeX0} y={y + 3} width={Math.max(1, r.io * scale)} height={14} fill="var(--viz-6)" />
                  <rect x={timeX0 + r.io * scale} y={y + 3} width={r.cpu * scale} height={14} fill="var(--viz-7)" />
                  <text x={timeX0 + r.total * scale + 6} y={y + 14} fontSize={10} fill="var(--viz-ink-2)">
                    {fmtTime(r.total * 1000)}
                  </text>
                </g>
              );
            })}
          </svg>
          <div style={{ overflowX: 'auto', marginTop: 8 }}>
            <table className="viz-table" style={{ marginTop: 0, fontSize: '0.78rem' }}>
              <thead>
                <tr>
                  <th>Throughput constants (MB/s, editable)</th>
                  <th>Compress</th>
                  <th>Decompress</th>
                  <th>CPU to compress 1 GB</th>
                </tr>
              </thead>
              <tbody>
                {CODECS.map((c) => (
                  <tr key={c.id}>
                    <td>{c.label}</td>
                    {(['c', 'd'] as const).map((k) => (
                      <td key={k}>
                        <input
                          type="number"
                          min={0.1}
                          step="any"
                          aria-label={`${c.label} ${k === 'c' ? 'compress' : 'decompress'} MB/s`}
                          value={tp[c.id][k]}
                          onChange={(e) => {
                            const v = Number(e.currentTarget.value);
                            if (Number.isFinite(v) && v > 0) setTp((t) => ({ ...t, [c.id]: { ...t[c.id], [k]: v } }));
                          }}
                          style={{ width: '6.5em', font: 'inherit', background: 'var(--viz-plane)', color: 'var(--viz-ink)', border: '1px solid var(--viz-border)', borderRadius: 4, padding: '1px 4px' }}
                        />
                      </td>
                    ))}
                    <td>{fmtNum(1000 / tp[c.id].c, 1)} s</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <p style={{ fontSize: '0.75rem', color: 'var(--viz-ink-2)', margin: '6px 0' }}>
            Defaults were measured once on an Apple M3 Pro with the zstd 1.5.6 and lz4 1.10.0 benchmark modes on 16 KB chunks of the JSON sample, and Node’s zlib for zlib -6. Replace them with numbers from your own hardware and data. <Button onClick={() => setTp(THROUGHPUT_DEFAULTS)}>Reset constants</Button>
          </p>
        </>
      )}
    </VizPanel>
  );
}
