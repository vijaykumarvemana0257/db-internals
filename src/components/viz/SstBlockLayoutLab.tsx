import { useMemo, useState } from 'react';
import { VizPanel, Slider, Choice, Legend, Stats, Note, fmtNum, TooltipHost, useTip } from './Viz';

/**
 * A block-based SSTable built from the learner's keys, with the byte sizes the
 * LevelDB/RocksDB format actually produces: varint entry headers, prefix compression
 * against the previous key, a full key every `restart interval` entries, a trailing
 * uint32 restart array, a 5-byte trailer (compression type + CRC32C) per block,
 * shortest-separator index keys, a full Bloom filter at 10 bits/key, and a 53-byte footer.
 */
const TRAILER = 5;
const FOOTER = 53;
const PROPERTIES = 180; // rough size of the properties block; only its existence matters here

const varintLen = (n: number) => (n < 128 ? 1 : n < 16384 ? 2 : n < 2097152 ? 3 : 4);
const commonPrefix = (a: string, b: string) => {
  let i = 0;
  while (i < a.length && i < b.length && a[i] === b[i]) i++;
  return i;
};

/** LevelDB BytewiseComparator::FindShortestSeparator, on strings. */
export function shortestSeparator(start: string, limit: string) {
  const d = commonPrefix(start, limit);
  if (d >= Math.min(start.length, limit.length)) return start;
  const c = start.charCodeAt(d);
  if (c < 0xff && c + 1 < limit.charCodeAt(d)) return start.slice(0, d) + String.fromCharCode(c + 1);
  return start;
}
/** LevelDB FindShortSuccessor: the shortest string >= key. */
function shortSuccessor(key: string) {
  for (let i = 0; i < key.length; i++) {
    const c = key.charCodeAt(i);
    if (c !== 0xff) return key.slice(0, i) + String.fromCharCode(c + 1);
  }
  return key;
}

type RegionKind = 'data' | 'filter' | 'props' | 'meta' | 'index' | 'footer';
type Region = { kind: RegionKind; label: string; offset: number; size: number };
type Entry = { key: string; shared: number; unshared: number; vlen: number; size: number; restart: boolean };
type Block = { entries: Entry[]; restarts: number; payload: number; size: number; offset: number; first: string; last: string };

export function build(keys: string[], valueLen: number, restartInterval: number, blockSize: number) {
  const blocks: Block[] = [];
  let cur: Entry[] = [];
  let prev = '';
  let payload = 0;
  const flush = () => {
    if (!cur.length) return;
    const restarts = cur.filter((e) => e.restart).length;
    const size = payload + restarts * 4 + 4;
    blocks.push({ entries: cur, restarts, payload, size, offset: 0, first: cur[0].key, last: cur[cur.length - 1].key });
    cur = [];
    payload = 0;
    prev = '';
  };
  for (const key of keys) {
    const restart = cur.length % restartInterval === 0;
    const shared = restart ? 0 : commonPrefix(prev, key);
    const unshared = key.length - shared;
    const size = varintLen(shared) + varintLen(unshared) + varintLen(valueLen) + unshared + valueLen;
    cur.push({ key, shared, unshared, vlen: valueLen, size, restart });
    payload += size;
    prev = key;
    if (payload >= blockSize) flush();
  }
  flush();

  let offset = 0;
  for (const b of blocks) {
    b.offset = offset;
    offset += b.size + TRAILER;
  }
  const dataEnd = offset;
  const index = blocks.map((b, i) => {
    const sep = i + 1 < blocks.length ? shortestSeparator(b.last, blocks[i + 1].first) : shortSuccessor(b.last);
    const handle = varintLen(b.offset) * 2; // offset + size varints (size rarely needs more)
    return { sep, block: i, size: varintLen(0) + varintLen(sep.length) + varintLen(handle) + sep.length + handle };
  });
  const filterBytes = Math.ceil((keys.length * 10) / 8) + 5; // 10 bits/key plus a small header
  const indexSize = index.reduce((a, x) => a + x.size, 0) + index.length * 4 + 4;
  const metaindex = 60;
  const regions: Region[] = [
    ...blocks.map((b, i) => ({ kind: 'data' as const, label: `data block ${i}`, offset: b.offset, size: b.size + TRAILER })),
    { kind: 'filter' as const, label: 'filter block', offset: dataEnd, size: filterBytes + TRAILER },
    { kind: 'props' as const, label: 'properties block', offset: dataEnd + filterBytes + TRAILER, size: PROPERTIES + TRAILER },
    { kind: 'meta' as const, label: 'metaindex block', offset: dataEnd + filterBytes + PROPERTIES + 2 * TRAILER, size: metaindex + TRAILER },
    { kind: 'index' as const, label: 'index block', offset: dataEnd + filterBytes + PROPERTIES + metaindex + 3 * TRAILER, size: indexSize + TRAILER },
  ];
  const last = regions[regions.length - 1];
  regions.push({ kind: 'footer' as const, label: 'footer', offset: last.offset + last.size, size: FOOTER });
  const fileSize = regions[regions.length - 1].offset + FOOTER;
  const rawKeyBytes = keys.reduce((a, k) => a + k.length, 0);
  return { blocks, index, regions, fileSize, filterBytes, rawKeyBytes, dataEnd };
}

/** Get(target): footer → index (binary search) → filter → data block → restart binary search → linear decode. */
export function lookup(t: ReturnType<typeof build>, target: string, present: boolean) {
  const steps: string[] = ['Read the footer (last 53 bytes): it holds the handles of the metaindex and index blocks.'];
  let lo = 0;
  let hi = t.index.length - 1;
  let indexCmp = 0;
  let found = -1;
  while (lo <= hi) {
    const mid = (lo + hi) >> 1;
    indexCmp++;
    if (t.index[mid].sep >= target) {
      found = mid;
      hi = mid - 1;
    } else lo = mid + 1;
  }
  steps.push(`Binary-search the index block: ${indexCmp} comparison${indexCmp === 1 ? '' : 's'} over ${t.index.length} separator keys.`);
  if (found < 0) {
    steps.push('Every separator is smaller than the key, so it is past the end of this file. No data block is read.');
    return { steps, indexCmp, restartCmp: 0, decodes: 0, blockRead: false };
  }
  if (!present) {
    steps.push('Check the filter: the key was never added, so with ~1% probability of a false positive the filter answers “definitely not here”. No data block is read.');
    return { steps, indexCmp, restartCmp: 0, decodes: 0, blockRead: false };
  }
  const b = t.blocks[found];
  steps.push(`The filter says “maybe”. Read data block ${found} (${fmtNum(b.size + TRAILER)} bytes) and verify its CRC32C.`);
  const restartKeys = b.entries.map((e, i) => ({ e, i })).filter((x) => x.e.restart);
  let rlo = 0;
  let rhi = restartKeys.length - 1;
  let restartCmp = 0;
  let start = 0;
  while (rlo <= rhi) {
    const mid = (rlo + rhi) >> 1;
    restartCmp++;
    if (restartKeys[mid].e.key < target) {
      start = mid;
      rlo = mid + 1;
    } else rhi = mid - 1;
  }
  let decodes = 0;
  for (let i = restartKeys[start].i; i < b.entries.length; i++) {
    decodes++;
    if (b.entries[i].key >= target) break;
  }
  steps.push(`Binary-search the block's ${restartKeys.length} restart points: ${restartCmp} comparison${restartCmp === 1 ? '' : 's'} on full keys.`);
  steps.push(`Decode forward from that restart point, rebuilding each key from the previous one: ${decodes} entr${decodes === 1 ? 'y' : 'ies'} until the key is reached.`);
  return { steps, indexCmp, restartCmp, decodes, blockRead: true };
}

const KIND_COLOR: Record<string, string> = {
  data: 'var(--viz-1)',
  filter: 'var(--viz-2)',
  props: 'var(--viz-3)',
  meta: 'var(--viz-4)',
  index: 'var(--viz-5)',
  footer: 'var(--viz-7)',
};

const PRESETS: Record<string, string[]> = {
  users: Array.from({ length: 120 }, (_, i) => `user:${String(1000 + i * 7).padStart(6, '0')}`),
  events: Array.from({ length: 120 }, (_, i) => `evt/2026-09-${String(1 + (i % 28)).padStart(2, '0')}/${String(i).padStart(4, '0')}`),
  urls: Array.from({ length: 120 }, (_, i) => `https://example.com/${['docs', 'blog', 'api', 'shop'][i % 4]}/${['a', 'b', 'c'][i % 3]}${i}`),
};

function ByteMap({ regions, fileSize, width }: { regions: ReturnType<typeof build>['regions']; fileSize: number; width: number }) {
  const tip = useTip();
  return (
    <svg width={width} height={46} role="img" aria-label="File layout by byte offset">
      {regions.map((r) => {
        const x = (r.offset / fileSize) * width;
        const w = Math.max(1.5, (r.size / fileSize) * width - 1);
        return (
          <rect
            key={r.label}
            x={x}
            y={6}
            width={w}
            height={26}
            rx={2}
            fill={KIND_COLOR[r.kind]}
            {...tip(
              <>
                <strong>{r.label}</strong>
                <br />
                offset {fmtNum(r.offset)} · {fmtNum(r.size)} bytes{r.kind !== 'footer' ? ` (includes the ${TRAILER}-byte trailer)` : ''}
              </>,
            )}
          />
        );
      })}
      <text x={0} y={44} fontSize={10}>
        0
      </text>
      <text x={width} y={44} fontSize={10} textAnchor="end">
        {fmtNum(fileSize)} bytes
      </text>
    </svg>
  );
}

export default function SstBlockLayoutLab() {
  const [preset, setPreset] = useState('users');
  const [text, setText] = useState(PRESETS.users.slice(0, 60).join('\n'));
  const [restart, setRestart] = useState(16);
  const [blockSize, setBlockSize] = useState(512);
  const [valueLen, setValueLen] = useState(24);
  const [probe, setProbe] = useState(0.45);
  const [present, setPresent] = useState<'present' | 'absent'>('present');
  const [shownBlock, setShownBlock] = useState(0);

  const keys = useMemo(() => [...new Set(text.split('\n').map((s) => s.trim()).filter(Boolean))].sort(), [text]);
  const table = useMemo(() => build(keys, valueLen, restart, blockSize), [keys, valueLen, restart, blockSize]);
  const targetKey = present === 'present' ? keys[Math.min(keys.length - 1, Math.floor(probe * keys.length))] ?? '' : `${keys[Math.floor(probe * keys.length)] ?? 'a'}~absent`;
  const get = useMemo(() => (keys.length ? lookup(table, targetKey, present === 'present') : null), [table, targetKey, present, keys.length]);

  // The trade-off across restart intervals for the same keys.
  const sweep = useMemo(
    () =>
      [1, 2, 4, 8, 16, 32].map((ri) => {
        const t = build(keys, valueLen, ri, blockSize);
        let decodes = 0;
        let rcmp = 0;
        const sample = keys.filter((_, i) => i % Math.max(1, Math.floor(keys.length / 25)) === 0);
        for (const k of sample) {
          const g = lookup(t, k, true);
          decodes += g.decodes;
          rcmp += g.restartCmp;
        }
        return { ri, size: t.dataEnd, decodes: decodes / Math.max(1, sample.length), rcmp: rcmp / Math.max(1, sample.length) };
      }),
    [keys, valueLen, blockSize],
  );

  const blk = table.blocks[Math.min(shownBlock, table.blocks.length - 1)];
  const W = 680;

  return (
    <VizPanel
      title="Inside a block-based SSTable"
      subtitle="Edit the keys or pick a preset. The file is rebuilt with real format sizes: prefix-compressed entries, restart points, a trailer on every block, a shortest-separator index, a filter and the footer. Then run a lookup."
      controls={
        <>
          <Choice
            label="Keys"
            value={preset}
            onChange={(v) => {
              setPreset(v);
              setText(PRESETS[v].slice(0, 60).join('\n'));
            }}
            options={[
              { value: 'users', label: 'user IDs' },
              { value: 'events', label: 'time-ordered events' },
              { value: 'urls', label: 'URLs' },
            ]}
          />
          <Slider label="Restart interval" min={1} max={32} value={restart} onChange={setRestart} />
          <Slider label="Block size" min={128} max={2048} step={64} value={blockSize} onChange={setBlockSize} format={(n) => `${n} B`} />
          <Slider label="Value size" min={0} max={128} step={4} value={valueLen} onChange={setValueLen} format={(n) => `${n} B`} />
        </>
      }
      legend={
        <Legend
          items={[
            { label: 'Data blocks', color: KIND_COLOR.data },
            { label: 'Filter', color: KIND_COLOR.filter },
            { label: 'Properties', color: KIND_COLOR.props },
            { label: 'Metaindex', color: KIND_COLOR.meta },
            { label: 'Index', color: KIND_COLOR.index },
            { label: 'Footer', color: KIND_COLOR.footer },
          ]}
        />
      }
      stats={
        <Stats
          items={[
            { label: 'Keys', value: fmtNum(keys.length) },
            { label: 'Data blocks', value: fmtNum(table.blocks.length) },
            { label: 'File size', value: `${fmtNum(table.fileSize)} B` },
            { label: 'Key bytes stored vs raw', value: `${fmtNum(table.blocks.reduce((a, b) => a + b.entries.reduce((c, e) => c + e.unshared, 0), 0))} / ${fmtNum(table.rawKeyBytes)}`, hint: 'Unshared key bytes written after prefix compression' },
            { label: 'Lookup cost', value: get ? `${get.indexCmp} + ${get.restartCmp} cmp, ${get.decodes} decodes` : '—', hint: 'Index comparisons + restart comparisons, then entries decoded' },
          ]}
        />
      }
      note={
        <Note>
          <strong>
            Get({targetKey.length > 40 ? targetKey.slice(0, 40) + '…' : targetKey}):
          </strong>{' '}
          {get ? get.steps.join(' ') : 'Add some keys.'}
        </Note>
      }
      table={
        <table className="viz-table">
          <thead>
            <tr>
              <th>Restart interval</th>
              <th>Data section bytes</th>
              <th>Avg restart comparisons</th>
              <th>Avg entries decoded</th>
            </tr>
          </thead>
          <tbody>
            {sweep.map((s) => (
              <tr key={s.ri}>
                <td>
                  {s.ri}
                  {s.ri === 16 ? ' (default)' : ''}
                </td>
                <td>{fmtNum(s.size)}</td>
                <td>{fmtNum(s.rcmp, 1)}</td>
                <td>{fmtNum(s.decodes, 1)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      }
    >
      <div style={{ display: 'grid', gap: 10 }}>
        <label className="viz-control">
          <span>Keys, one per line (sorted and de-duplicated for you)</span>
          <textarea
            value={text}
            onChange={(e) => setText(e.currentTarget.value)}
            rows={4}
            spellCheck={false}
            style={{ font: 'inherit', fontFamily: 'var(--sl-font-mono, monospace)', fontSize: '0.75rem', background: 'var(--viz-plane)', color: 'var(--viz-ink)', border: '1px solid var(--viz-border)', borderRadius: 6, padding: 6, width: '100%', maxWidth: W }}
          />
        </label>

        <TooltipHost>
          <ByteMap regions={table.regions} fileSize={table.fileSize} width={W} />
        </TooltipHost>

        <div className="viz-controls" style={{ marginBottom: 0 }}>
          <Slider label="Look up key at position" min={0} max={0.99} step={0.01} value={probe} onChange={setProbe} format={(n) => `${Math.round(n * 100)}%`} />
          <Choice label="Lookup" value={present} onChange={setPresent} options={[{ value: 'present', label: 'a key that exists' }, { value: 'absent', label: 'a key that does not' }]} />
          <Slider label="Show data block" min={0} max={Math.max(0, table.blocks.length - 1)} value={Math.min(shownBlock, table.blocks.length - 1)} onChange={setShownBlock} />
        </div>

        {blk ? (
          <div style={{ overflowX: 'auto' }}>
            <table className="viz-table" style={{ marginTop: 0 }}>
              <thead>
                <tr>
                  <th>#</th>
                  <th>shared</th>
                  <th>unshared</th>
                  <th>value_len</th>
                  <th>key delta written</th>
                  <th>full key</th>
                  <th>entry bytes</th>
                </tr>
              </thead>
              <tbody>
                {blk.entries.slice(0, 20).map((e, i) => (
                  <tr key={e.key} style={{ fontWeight: e.restart ? 700 : 400 }}>
                    <td>
                      {i}
                      {e.restart ? ' ◆' : ''}
                    </td>
                    <td>{e.shared}</td>
                    <td>{e.unshared}</td>
                    <td>{e.vlen}</td>
                    <td style={{ fontFamily: 'var(--sl-font-mono, monospace)' }}>{e.key.slice(e.shared)}</td>
                    <td style={{ fontFamily: 'var(--sl-font-mono, monospace)', color: 'var(--viz-ink-2)' }}>{e.key}</td>
                    <td>{e.size}</td>
                  </tr>
                ))}
              </tbody>
            </table>
            <p style={{ margin: '0.3rem 0 0', fontSize: '0.75rem', color: 'var(--viz-ink-2)' }}>
              ◆ = restart point (full key, shared = 0). Block {Math.min(shownBlock, table.blocks.length - 1)}: {blk.entries.length} entries, {blk.restarts} restart offsets × 4 B + a 4 B count + a {TRAILER} B trailer = {fmtNum(blk.size + TRAILER)} bytes.
              {blk.entries.length > 20 ? ` Showing the first 20 entries.` : ''}
            </p>
          </div>
        ) : null}
      </div>
    </VizPanel>
  );
}
