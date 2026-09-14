import { useMemo, useState, type ReactNode } from 'react';
import {
  VizPanel,
  Slider,
  Choice,
  Segmented,
  Button,
  Legend,
  Stats,
  Note,
  TooltipHost,
  useTip,
  fmtBytes,
  fmtNum,
  makeRng,
  useSize,
} from './Viz';

/**
 * The LevelDB/RocksDB block-based table, built in front of the learner.
 *
 * Everything here is the real encoding: an entry is
 * varint(shared) varint(non_shared) varint(value_len) key_delta value; every
 * `block_restart_interval` entries one entry stores shared = 0 so its key is whole, and the
 * offsets of those entries are appended as a u32 array plus a u32 count so a reader can
 * binary-search a block it has not decoded. Each block carries a 5-byte trailer (1 byte
 * compression type + 4 byte CRC32C). The index stores the shortest separator between
 * adjacent blocks (LevelDB's FindShortestSeparator / FindShortSuccessor), and the fixed
 * footer at the end of the file is the only structure whose offset is known a priori.
 *
 * Deterministic: keys and value sizes come from makeRng, never Math.random.
 */

/* -------------------------------------------------------------- encoding */

const TRAILER = 5; // 1 byte compression type + 4 byte CRC32C
const FOOTER = 53; // RocksDB block-based footer; LevelDB's is 48
const INTERNAL_SUFFIX = 8; // 7-byte sequence number + 1-byte value type
const PROPS_BYTES = 384; // a properties block is a few hundred bytes of name/value pairs
const METAINDEX_ENTRY = 46; // name + varint offset + varint size, per meta block

function varint(n: number) {
  return n < 128 ? 1 : n < 16384 ? 2 : n < 2097152 ? 3 : 4;
}

function commonPrefix(a: string, b: string) {
  const n = Math.min(a.length, b.length);
  let i = 0;
  while (i < n && a.charCodeAt(i) === b.charCodeAt(i)) i++;
  return i;
}

/** LevelDB BytewiseComparator::FindShortestSeparator -- the key the index actually stores. */
function shortestSeparator(a: string, b: string) {
  const n = Math.min(a.length, b.length);
  let i = 0;
  while (i < n && a[i] === b[i]) i++;
  if (i >= n) return a;
  const c = a.charCodeAt(i);
  if (c < 0xff && c + 1 < b.charCodeAt(i)) return a.slice(0, i) + String.fromCharCode(c + 1);
  return a;
}

/** LevelDB BytewiseComparator::FindShortSuccessor -- the separator for the very last block. */
function shortSuccessor(a: string) {
  for (let i = 0; i < a.length; i++) {
    const c = a.charCodeAt(i);
    if (c !== 0xff) return a.slice(0, i) + String.fromCharCode(c + 1);
  }
  return a;
}

/* ------------------------------------------------------------- key sets */

type Shape = 'user' | 'ts' | 'hash';

const SHAPES: { value: Shape; label: string }[] = [
  { value: 'user', label: 'user:000000123|profile' },
  { value: 'ts', label: 'm|host-03|2026-09-13T...' },
  { value: 'hash', label: '9f3c...  (hashed ids)' },
];

function pad(n: number, w: number) {
  return String(n).padStart(w, '0');
}

const HEX = '0123456789abcdef';

function makeKeys(shape: Shape, n: number): string[] {
  const rng = makeRng(shape === 'user' ? 7919 : shape === 'ts' ? 104729 : 15485863);
  const out: string[] = [];
  if (shape === 'user') {
    let id = 100000000;
    for (let i = 0; i < n; i++) {
      id += 1 + Math.floor(rng() * 9);
      out.push(`user:${pad(id, 9)}|profile`);
    }
  } else if (shape === 'ts') {
    let t = 0;
    for (let i = 0; i < n; i++) {
      t += 1 + Math.floor(rng() * 40);
      const host = pad(1 + Math.floor(rng() * 8), 2);
      const hh = pad(Math.floor(t / 3600) % 24, 2);
      const mm = pad(Math.floor(t / 60) % 60, 2);
      const ss = pad(t % 60, 2);
      out.push(`m|host-${host}|2026-09-13T${hh}:${mm}:${ss}Z`);
    }
  } else {
    for (let i = 0; i < n; i++) {
      let s = '';
      for (let j = 0; j < 16; j++) s += HEX[Math.floor(rng() * 16)];
      out.push(s);
    }
  }
  out.sort();
  return out.filter((k, i) => i === 0 || k !== out[i - 1]);
}

function makeValueLens(n: number, target: number) {
  const rng = makeRng(2654435761);
  return Array.from({ length: n }, () => Math.max(4, Math.round(target * (0.7 + rng() * 0.6))));
}

/* ------------------------------------------------------------- the table */

type Entry = {
  i: number;
  key: string;
  shared: number;
  nonShared: number;
  valueLen: number;
  hdr: number;
  bytes: number;
  restart: boolean;
};

type Block = {
  idx: number;
  offset: number;
  entries: Entry[];
  restarts: number[];
  raw: number; // entry bytes only
  restartBytes: number; // 4 per restart point + 4 for the count
  size: number; // raw + restartBytes + TRAILER
  firstKey: string;
  lastKey: string;
  sep: string; // the index key that routes to this block
  sepBytes: number;
};

type Built = {
  keys: string[];
  blocks: Block[];
  dataBytes: number;
  indexBytes: number;
  filterBytes: number;
  metaindexBytes: number;
  propsBytes: number;
  total: number;
  rawKeyBytes: number; // every internal key in full, if nothing were elided
  storedKeyBytes: number; // what the data blocks actually hold
  restartBytes: number;
  filterK: number;
  fp: number;
};

function buildBlocks(keys: string[], vlens: number[], blockTarget: number, interval: number): Block[] {
  const blocks: Block[] = [];
  let entries: Entry[] = [];
  let restarts: number[] = [];
  let raw = 0;
  let counter = 0;
  let prev = '';
  let offset = 0;

  const flush = () => {
    if (entries.length === 0) return;
    const restartBytes = restarts.length * 4 + 4;
    const size = raw + restartBytes + TRAILER;
    blocks.push({
      idx: blocks.length,
      offset,
      entries,
      restarts,
      raw,
      restartBytes,
      size,
      firstKey: entries[0].key,
      lastKey: entries[entries.length - 1].key,
      sep: '',
      sepBytes: 0,
    });
    offset += size;
    entries = [];
    restarts = [];
    raw = 0;
    counter = 0;
    prev = '';
  };

  keys.forEach((k, i) => {
    const isRestart = counter === 0;
    const ikLen = k.length + INTERNAL_SUFFIX;
    const shared = isRestart ? 0 : commonPrefix(prev, k);
    const nonShared = ikLen - shared;
    const valueLen = vlens[i] ?? 64;
    const hdr = varint(shared) + varint(nonShared) + varint(valueLen);
    const bytes = hdr + nonShared + valueLen;
    if (isRestart) restarts.push(entries.length);
    entries.push({ i, key: k, shared, nonShared, valueLen, hdr, bytes, restart: isRestart });
    raw += bytes;
    prev = k;
    counter = (counter + 1) % interval;
    if (raw + restarts.length * 4 + 4 >= blockTarget) flush();
  });
  flush();
  return blocks;
}

function build(
  keys: string[],
  vlens: number[],
  blockTarget: number,
  interval: number,
  bitsPerKey: number,
  hasFilter: boolean,
): Built {
  const blocks = buildBlocks(keys, vlens, blockTarget, interval);

  // Index: one entry per data block, keyed by the shortest separator to the next block.
  let indexBytes = 0;
  blocks.forEach((b, i) => {
    const next = blocks[i + 1];
    const sep = next ? shortestSeparator(b.lastKey, next.firstKey) : shortSuccessor(b.lastKey);
    const sepLen = sep.length + INTERNAL_SUFFIX;
    const handle = varint(b.offset) + varint(b.size);
    // index_block_restart_interval = 1: every index entry is a restart, so shared = 0.
    b.sep = sep;
    b.sepBytes = varint(0) + varint(sepLen) + varint(handle) + sepLen + handle;
    indexBytes += b.sepBytes;
  });
  indexBytes += blocks.length * 4 + 4 + TRAILER;

  const dataBytes = blocks.reduce((a, b) => a + b.size, 0);
  const filterBytes = hasFilter ? Math.ceil((bitsPerKey * keys.length) / 8) + 5 + TRAILER : 0;
  const propsBytes = PROPS_BYTES + TRAILER;
  const metaindexBytes = METAINDEX_ENTRY * (hasFilter ? 2 : 1) + 8 + TRAILER;

  const rawKeyBytes = keys.reduce((a, k) => a + k.length + INTERNAL_SUFFIX, 0);
  const storedKeyBytes = blocks.reduce((a, b) => a + b.entries.reduce((x, e) => x + e.nonShared, 0), 0);
  const restartBytes = blocks.reduce((a, b) => a + b.restartBytes, 0);

  const k = Math.max(1, Math.round(bitsPerKey * Math.LN2));
  const fp = hasFilter && bitsPerKey > 0 ? Math.pow(1 - Math.exp(-k / bitsPerKey), k) : 1;

  return {
    keys,
    blocks,
    dataBytes,
    indexBytes,
    filterBytes,
    metaindexBytes,
    propsBytes,
    total: dataBytes + filterBytes + indexBytes + propsBytes + metaindexBytes + FOOTER,
    rawKeyBytes,
    storedKeyBytes,
    restartBytes,
    filterK: k,
    fp,
  };
}

/* ----------------------------------------------------------- the lookup */

type FilterMode = 'full' | 'block' | 'none';

type Step = {
  region: string;
  title: string;
  detail: string;
  bytes: number;
  block?: number;
  from?: number; // entry slot within the block where the linear decode starts
  to?: number;
};

type Probe = { id: string; key: string; label: string; present: boolean; forceFp?: boolean };

function makeProbes(keys: string[]): Probe[] {
  const n = keys.length;
  if (n === 0) return [];
  const mid = Math.floor(n * 0.37);
  const q = Math.floor(n * 0.62);
  return [
    { id: 'first', key: keys[0], label: 'present - first key', present: true },
    { id: 'mid', key: keys[mid], label: `present - key #${mid + 1}`, present: true },
    { id: 'last', key: keys[n - 1], label: 'present - last key', present: true },
    { id: 'gap', key: keys[q] + '~', label: `absent - between #${q + 1} and #${q + 2}`, present: false },
    { id: 'fp', key: keys[mid] + '~', label: 'absent - filter false positive', present: false, forceFp: true },
    { id: 'past', key: keys[n - 1] + '~', label: 'absent - just past the last key', present: false },
    { id: 'way', key: '~~', label: "absent - above the file's whole range", present: false },
  ];
}

function log2Steps(n: number) {
  return n <= 1 ? 1 : Math.ceil(Math.log2(n + 1));
}

function hash01(s: string) {
  let h = 2166136261 >>> 0;
  for (let i = 0; i < s.length; i++) {
    h ^= s.charCodeAt(i);
    h = Math.imul(h, 16777619) >>> 0;
  }
  return (h >>> 8) / 16777216;
}

type Search = {
  steps: Step[];
  found: boolean;
  bytesRead: number;
  bytesWarm: number; // what it costs once index and filter are pinned in the block cache
  indexCmp: number;
  restartCmp: number;
  decoded: number;
  keyBytes: number;
};

function runGet(f: Built, probe: Probe, mode: FilterMode): Search {
  const steps: Step[] = [];
  const target = probe.key;

  steps.push({
    region: 'footer',
    title: 'read the footer',
    detail:
      `The last ${FOOTER} bytes sit at a known offset - file_size minus ${FOOTER}. They hold the metaindex handle, ` +
      'the index handle, the checksum type, the format version and the 8-byte magic number that proves this is a ' +
      'block-based table and not something else with an .sst extension.',
    bytes: FOOTER,
  });
  steps.push({
    region: 'metaindex',
    title: 'read the metaindex',
    detail:
      'A tiny sorted block of name to BlockHandle: "rocksdb.properties", and the filter under its policy name. ' +
      'This is how a reader discovers blocks whose existence is optional.',
    bytes: f.metaindexBytes,
  });

  let bytesRead = FOOTER + f.metaindexBytes;
  let bytesWarm = 0;

  const filterMiss = mode === 'full' && !probe.present && !probe.forceFp && hash01(target) >= f.fp;
  if (mode === 'full') {
    steps.push({
      region: 'filter',
      title: 'probe the full-file filter',
      detail: filterMiss
        ? `${f.filterK} hash probes, at least one bit is 0, so the key is definitively not in this file. The index ` +
          'and the data block are never touched - this is the whole reason a point read does not cost one block ' +
          'read per file in the tree.'
        : probe.forceFp
          ? `${f.filterK} probes all land on bits that other keys set. A false positive costs a full index descent ` +
            `and a data-block read that returns nothing, on about ${fmtNum(f.fp * 100, 2)} % of absent lookups.`
          : `${f.filterK} probes, every bit set, so "may contain". A filter can only ever rule a key out.`,
      bytes: f.filterBytes,
    });
    bytesRead += f.filterBytes;
    if (filterMiss) {
      return { steps, found: false, bytesRead, bytesWarm, indexCmp: 0, restartCmp: 0, decoded: 0, keyBytes: 0 };
    }
  }

  const blocks = f.blocks;
  const indexCmp = log2Steps(blocks.length);
  const hitRaw = blocks.findIndex((b) => b.sep >= target);
  const past = hitRaw < 0;
  steps.push({
    region: 'index',
    title: 'binary-search the index',
    detail: past
      ? `${indexCmp} comparisons against separator keys: the target sorts above the last separator ` +
        `"${blocks[blocks.length - 1].sep}", so no data block in this file can hold it. Not found, and zero data I/O.`
      : `${indexCmp} comparisons over ${blocks.length} separators find the first block whose separator is >= the ` +
        `target: block ${hitRaw}, at offset ${fmtNum(blocks[hitRaw].offset)}, ${fmtNum(blocks[hitRaw].size)} bytes. ` +
        'The separator is a truncated key that may not exist in the file at all.',
    bytes: f.indexBytes,
  });
  bytesRead += f.indexBytes;
  if (past) {
    return { steps, found: false, bytesRead, bytesWarm, indexCmp, restartCmp: 0, decoded: 0, keyBytes: 0 };
  }
  const hit = Math.min(hitRaw, blocks.length - 1);
  const b = blocks[hit];

  if (mode === 'block') {
    const share = Math.max(16, Math.ceil(f.filterBytes / Math.max(1, blocks.length)));
    steps.push({
      region: 'filter',
      title: "probe this block's filter",
      detail:
        'LevelDB builds one filter per 2 KB of data blocks, keyed by block offset, so the filter can only be ' +
        'consulted once the index has already picked a block. RocksDB deprecated this layout precisely because it ' +
        'cannot skip the index descent.',
      bytes: share,
    });
    bytesRead += share;
  }

  steps.push({
    region: `d${hit}`,
    title: `read + verify block ${hit}`,
    detail:
      `${fmtNum(b.size)} bytes off disk, or out of the block cache, then CRC32C over the block plus its ` +
      'compression-type byte is compared against the 4-byte trailer. Decompression, if the block is compressed, ' +
      'happens here - the whole block, not the one entry you asked for.',
    bytes: b.size,
    block: hit,
  });
  bytesRead += b.size;
  bytesWarm += b.size;

  const restartCmp = log2Steps(b.restarts.length);
  let r = 0;
  for (let i = 0; i < b.restarts.length; i++) {
    if (b.entries[b.restarts[i]].key <= target) r = i;
    else break;
  }
  const start = b.restarts[r];
  const end = r + 1 < b.restarts.length ? b.restarts[r + 1] : b.entries.length;
  steps.push({
    region: `d${hit}`,
    title: 'binary-search the restarts',
    detail:
      `The last ${b.restartBytes} bytes of the block are ${b.restarts.length} u32 offsets plus a u32 count. The ` +
      'entries at those offsets store shared = 0, so their keys are whole and comparable without decoding anything ' +
      `before them - ${restartCmp} comparisons land on restart ${r}, entry #${b.entries[start].i}.`,
    bytes: b.restartBytes,
    block: hit,
    from: start,
    to: start,
  });

  let decoded = 0;
  let keyBytes = 0;
  let foundAt = -1;
  let cur = '';
  for (let i = start; i < end; i++) {
    const e = b.entries[i];
    cur = cur.slice(0, e.shared) + e.key.slice(e.shared);
    decoded++;
    keyBytes += cur.length + INTERNAL_SUFFIX;
    if (e.key >= target) {
      foundAt = i;
      break;
    }
  }
  const found = foundAt >= 0 && b.entries[foundAt].key === target;
  steps.push({
    region: `d${hit}`,
    title: found ? 'linear decode: hit' : 'linear decode: miss',
    detail: found
      ? `Forward from the restart point, each entry is rebuilt as previous_key[0 .. shared] + delta: ${decoded} ` +
        `entr${decoded === 1 ? 'y' : 'ies'} decoded and ${keyBytes} key bytes materialised to return a ` +
        `${b.entries[foundAt].valueLen}-byte value. This scan is exactly what block_restart_interval bounds.`
      : `${decoded} entries decoded before the keys ran past the target. At the SSTable level that is "not in this ` +
        'file", and the LSM read path moves on to the next one.',
    bytes: 0,
    block: hit,
    from: start,
    to: foundAt >= 0 ? foundAt : Math.max(start, end - 1),
  });

  return { steps, found, bytesRead, bytesWarm, indexCmp, restartCmp, decoded, keyBytes };
}

/* ------------------------------------------------------------- painting */

type RegionKind = 'data' | 'filter' | 'index' | 'props' | 'metaindex' | 'footer';

const KIND_FILL: Record<RegionKind, string> = {
  data: 'var(--viz-1)',
  filter: 'var(--viz-6)',
  index: 'var(--viz-2)',
  props: 'var(--viz-5)',
  metaindex: 'var(--viz-7)',
  footer: 'var(--viz-8)',
};

type Region = { id: string; kind: RegionKind; label: string; bytes: number; hint: ReactNode };

function regionsOf(f: Built, mode: FilterMode): Region[] {
  const out: Region[] = f.blocks.map((b) => ({
    id: `d${b.idx}`,
    kind: 'data' as RegionKind,
    label: `${b.idx}`,
    bytes: b.size,
    hint: (
      <>
        <strong>Data block {b.idx}</strong>
        <br />
        offset {fmtNum(b.offset)}, {fmtNum(b.size)} B = {fmtNum(b.raw)} entry bytes + {b.restartBytes} restart array +{' '}
        {TRAILER} trailer
        <br />
        {b.entries.length} entries, {b.restarts.length} restart points
        <br />
        first: {b.firstKey}
        <br />
        last: {b.lastKey}
        <br />
        index separator: {b.sep}
      </>
    ),
  }));
  if (mode !== 'none') {
    out.push({
      id: 'filter',
      kind: 'filter',
      label: 'filter',
      bytes: f.filterBytes,
      hint: (
        <>
          <strong>Filter block</strong>
          <br />
          {fmtNum(f.filterBytes)} B, {f.filterK} hash probes per key, about {fmtNum(f.fp * 100, 2)} % false positives
          <br />
          {mode === 'full'
            ? 'RocksDB full filter: one filter over every key in the file, consulted before the index.'
            : 'LevelDB block-based filter: one filter per 2 KB of data blocks, consulted after the index.'}
        </>
      ),
    });
  }
  out.push({
    id: 'index',
    kind: 'index',
    label: 'index',
    bytes: f.indexBytes,
    hint: (
      <>
        <strong>Index block</strong>
        <br />
        {fmtNum(f.indexBytes)} B, one entry per data block: shortest separator + BlockHandle(offset, size)
        <br />
        index_block_restart_interval = 1, so every index entry is a restart and nothing here is prefix-compressed
      </>
    ),
  });
  out.push({
    id: 'props',
    kind: 'props',
    label: 'properties',
    bytes: f.propsBytes,
    hint: (
      <>
        <strong>Properties block</strong>
        <br />
        rocksdb.num.entries, rocksdb.raw.key.size, rocksdb.data.size, rocksdb.index.size, rocksdb.filter.size,
        rocksdb.deleted.keys - what compaction and sst_dump --show_properties read without touching a data block
      </>
    ),
  });
  out.push({
    id: 'metaindex',
    kind: 'metaindex',
    label: 'metaindex',
    bytes: f.metaindexBytes,
    hint: (
      <>
        <strong>Metaindex block</strong>
        <br />
        name to BlockHandle for every optional block: the filter under its policy name, the properties block, a
        compression dictionary, the range-deletion block
      </>
    ),
  });
  out.push({
    id: 'footer',
    kind: 'footer',
    label: 'footer',
    bytes: FOOTER,
    hint: (
      <>
        <strong>Footer - fixed size, read first</strong>
        <br />
        metaindex handle + index handle, padded, then the checksum type, the format version and the magic number
        <br />
        LevelDB: 48 B ending 0xdb4775248b80fb57. RocksDB: 53 B ending 0x88e241b785f4cff7
      </>
    ),
  });
  return out;
}

/* ------------------------------------------------------------ component */

const MAX_ROWS = 16;

export default function SstableBlockFormatExplorer() {
  const [shape, setShape] = useState<Shape>('user');
  const [nKeys, setNKeys] = useState(128);
  const [valueLen, setValueLen] = useState(192);
  const [blockTarget, setBlockTarget] = useState(4096);
  const [interval, setIntervalKnob] = useState(16);
  const [bits, setBits] = useState(10);
  const [mode, setMode] = useState<FilterMode>('full');
  const [probeId, setProbeId] = useState('mid');
  const [step, setStep] = useState(0);
  const [sel, setSel] = useState(0);
  const [ref, width] = useSize(760);
  const tip = useTip();

  const keys = useMemo(() => makeKeys(shape, nKeys), [shape, nKeys]);
  const vlens = useMemo(() => makeValueLens(keys.length, valueLen), [keys.length, valueLen]);
  const f = useMemo(
    () => build(keys, vlens, blockTarget, interval, bits, mode !== 'none'),
    [keys, vlens, blockTarget, interval, bits, mode],
  );
  const probes = useMemo(() => makeProbes(keys), [keys]);
  const probe = probes.find((p) => p.id === probeId) ?? probes[0];
  const search = useMemo(() => runGet(f, probe, mode), [f, probe, mode]);

  const sweep = useMemo(
    () =>
      [1, 2, 4, 8, 16, 32].map((iv) => {
        const g = build(keys, vlens, blockTarget, iv, bits, mode !== 'none');
        const avgRestarts = g.blocks.reduce((a, b) => a + b.restarts.length, 0) / Math.max(1, g.blocks.length);
        return {
          iv,
          total: g.total,
          restartBytes: g.restartBytes,
          stored: g.storedKeyBytes,
          cmp: log2Steps(Math.round(avgRestarts)),
          scan: (iv + 1) / 2,
        };
      }),
    [keys, vlens, blockTarget, bits, mode],
  );

  const regions = useMemo(() => regionsOf(f, mode), [f, mode]);
  const activeStep = step > 0 && step <= search.steps.length ? search.steps[step - 1] : null;
  const selBlock = f.blocks[Math.min(sel, f.blocks.length - 1)];
  const showBlock = activeStep?.block != null ? f.blocks[activeStep.block] : selBlock;

  const reset = (fn: () => void) => {
    fn();
    setStep(0);
  };

  /* file map */
  const mapPad = 8;
  const avail = Math.max(320, width - 2 * mapPad);
  const pxPerByte = avail / Math.max(1, f.total);
  const widths = regions.map((r) => Math.max(30, r.bytes * pxPerByte));
  const mapW = Math.max(width, widths.reduce((a, b) => a + b, 0) + 2 * mapPad);
  const mapH = 108;

  /* block detail */
  const rows = showBlock ? showBlock.entries.slice(0, MAX_ROWS) : [];
  const maxEntry = Math.max(1, ...rows.map((e) => e.hdr + e.shared + e.nonShared + e.valueLen));
  const keyCol = 250;
  const stripX = keyCol + 26;
  const stripW = Math.max(220, Math.min(430, width - stripX - 16));
  const rowH = 21;
  const detH = 54 + rows.length * rowH + 80;
  const detW = Math.max(width, stripX + stripW + 16);
  const byteScale = stripW / maxEntry;
  const scanFrom = activeStep && activeStep.from != null && showBlock && activeStep.block === showBlock.idx ? activeStep.from : -1;
  const scanTo = activeStep && activeStep.to != null ? activeStep.to : -1;

  /* read path */
  const pathH = 88;
  const stepW = Math.max(118, Math.min(190, (width - 16) / Math.max(1, search.steps.length)));
  const pathW = Math.max(width, search.steps.length * stepW + 16);

  return (
    <VizPanel
      title="Build an SSTable, then Get() a key out of it"
      subtitle="Pick a key shape and the builder knobs; the file below is drawn to scale, byte for byte. Then pick a probe key and step the read path: footer, metaindex, filter, index, block, restart array, linear decode."
      controls={
        <>
          <Choice label="Key shape" value={shape} options={SHAPES} onChange={(v) => reset(() => setShape(v))} />
          <Slider label="Keys" min={32} max={320} step={8} value={nKeys} onChange={(v) => reset(() => setNKeys(v))} />
          <Slider
            label="Value size"
            min={16}
            max={512}
            step={16}
            value={valueLen}
            onChange={(v) => reset(() => setValueLen(v))}
            format={(n) => `${n} B`}
          />
          <Slider
            label="block_size"
            min={1024}
            max={8192}
            step={1024}
            value={blockTarget}
            onChange={(v) => reset(() => setBlockTarget(v))}
            format={(n) => `${n / 1024} KB`}
          />
          <Slider
            label="block_restart_interval"
            min={1}
            max={32}
            value={interval}
            onChange={(v) => reset(() => setIntervalKnob(v))}
          />
          <Slider
            label="Filter bits/key"
            min={4}
            max={20}
            value={bits}
            onChange={(v) => reset(() => setBits(v))}
            format={(n) => `${n} bits`}
          />
          <Segmented
            label="Filter layout"
            value={mode}
            onChange={(v) => reset(() => setMode(v))}
            options={[
              { value: 'full', label: 'full file', title: 'RocksDB: one filter for the whole SSTable, checked before the index' },
              { value: 'block', label: 'per block', title: 'LevelDB: one filter per 2 KB of data, checked after the index' },
              { value: 'none', label: 'none', title: 'No filter block at all' },
            ]}
          />
          <Choice
            label="Get(key)"
            value={probeId}
            options={probes.map((p) => ({ value: p.id, label: p.label }))}
            onChange={(v) => reset(() => setProbeId(v))}
          />
          <Button
            onClick={() => setStep((s) => Math.min(s + 1, search.steps.length))}
            primary
            disabled={step >= search.steps.length}
          >
            Step
          </Button>
          <Button onClick={() => setStep(search.steps.length)} disabled={step >= search.steps.length}>
            Run all
          </Button>
          <Button onClick={() => setStep(0)} disabled={step === 0}>
            Reset
          </Button>
        </>
      }
      legend={
        <Legend
          items={[
            { label: 'data block', color: KIND_FILL.data },
            { label: 'filter block', color: KIND_FILL.filter },
            { label: 'index block', color: KIND_FILL.index },
            { label: 'properties', color: KIND_FILL.props },
            { label: 'metaindex', color: KIND_FILL.metaindex },
            { label: 'footer', color: KIND_FILL.footer },
            { label: 'varint header (shared, non_shared, value_len)', color: 'var(--viz-7)' },
            { label: 'key bytes elided by the shared prefix', color: 'var(--viz-stale)' },
            { label: 'R - restart point, whole key stored', color: 'var(--viz-warning)' },
          ]}
        />
      }
      stats={
        <Stats
          items={[
            {
              label: 'File size',
              value: fmtBytes(f.total),
              hint: `${fmtNum(f.total)} bytes across ${f.blocks.length} data blocks`,
            },
            {
              label: 'Key bytes stored',
              value: `${fmtNum((f.storedKeyBytes / Math.max(1, f.rawKeyBytes)) * 100, 0)} %`,
              hint: `${fmtNum(f.storedKeyBytes)} of ${fmtNum(f.rawKeyBytes)} internal-key bytes survive prefix compression`,
            },
            {
              label: 'Restart arrays',
              value: fmtBytes(f.restartBytes),
              hint: '4 bytes per restart point plus a 4-byte count, in every block',
            },
            {
              label: 'Index + filter',
              value: fmtBytes(f.indexBytes + f.filterBytes),
              hint: 'The metadata a reader wants resident - what cache_index_and_filter_blocks decides where to keep',
            },
            { label: 'Index comparisons', value: search.indexCmp || '-' },
            { label: 'Restart comparisons', value: search.restartCmp || '-' },
            { label: 'Entries decoded', value: search.decoded || '-', hint: 'The linear scan inside one restart interval' },
            {
              label: 'Bytes read, cold',
              value: fmtBytes(search.bytesRead),
              hint: `${fmtBytes(search.bytesWarm)} once the index and filter blocks are pinned in the block cache`,
            },
          ]}
        />
      }
      note={
        <Note>
          {activeStep ? (
            <>
              <strong>
                {step}/{search.steps.length} - {activeStep.title}.
              </strong>{' '}
              {activeStep.detail}
            </>
          ) : (
            <>
              <strong>
                Get("{probe.key}") - {search.steps.length} steps.
              </strong>{' '}
              Press Step. Every read starts at the footer, because it is the only structure in the file whose offset is
              known without reading something else first.
            </>
          )}
        </Note>
      }
      table={
        <>
          <table className="viz-table">
            <thead>
              <tr>
                <th>Block</th>
                <th>Offset</th>
                <th>Bytes</th>
                <th>Entries</th>
                <th>Restarts</th>
                <th>Restart array</th>
                <th>First key</th>
                <th>Last key</th>
                <th>Index separator</th>
              </tr>
            </thead>
            <tbody>
              {f.blocks.map((b) => (
                <tr key={b.idx}>
                  <td>{b.idx}</td>
                  <td>{fmtNum(b.offset)}</td>
                  <td>{fmtNum(b.size)}</td>
                  <td>{b.entries.length}</td>
                  <td>{b.restarts.length}</td>
                  <td>{b.restartBytes} B</td>
                  <td>{b.firstKey}</td>
                  <td>{b.lastKey}</td>
                  <td>{b.sep}</td>
                </tr>
              ))}
            </tbody>
          </table>
          <table className="viz-table">
            <thead>
              <tr>
                <th>block_restart_interval</th>
                <th>File size (B)</th>
                <th>Restart arrays (B)</th>
                <th>Key bytes stored</th>
                <th>Binary-search comparisons</th>
                <th>Entries decoded, average</th>
              </tr>
            </thead>
            <tbody>
              {sweep.map((s) => (
                <tr key={s.iv}>
                  <td>{s.iv}</td>
                  <td>{fmtNum(s.total)}</td>
                  <td>{fmtNum(s.restartBytes)}</td>
                  <td>{fmtNum(s.stored)}</td>
                  <td>{s.cmp}</td>
                  <td>{fmtNum(s.scan, 1)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </>
      }
    >
      <div ref={ref}>
        <TooltipHost>
          <svg
            width={mapW}
            height={mapH}
            role="img"
            aria-label="The SSTable laid out to scale, from the first data block to the footer"
          >
            <text x={0} y={12} fill="var(--viz-ink-2)">
              file layout - {fmtNum(f.total)} bytes; click a data block to open it
            </text>
            {regions.map((r, i) => {
              const x = mapPad + widths.slice(0, i).reduce((a, b) => a + b, 0);
              const w = widths[i];
              const on = activeStep?.region === r.id;
              const opened = r.kind === 'data' && showBlock && r.id === `d${showBlock.idx}`;
              return (
                <g
                  key={r.id}
                  {...tip(r.hint)}
                  onClick={() => {
                    if (r.kind === 'data') setSel(Number(r.id.slice(1)));
                  }}
                  style={{ cursor: r.kind === 'data' ? 'pointer' : 'help' }}
                >
                  <rect
                    x={x}
                    y={24}
                    width={Math.max(2, w - 2)}
                    height={44}
                    rx={4}
                    fill={KIND_FILL[r.kind]}
                    opacity={on ? 1 : opened ? 0.95 : 0.7}
                    stroke={on ? 'var(--viz-warning)' : opened ? 'var(--viz-ink)' : 'var(--viz-surface)'}
                    strokeWidth={on ? 3 : opened ? 2 : 1}
                  />
                  {w > 34 ? (
                    <text x={x + w / 2 - 1} y={51} textAnchor="middle" fill="var(--viz-surface)" fontWeight={600}>
                      {r.label}
                    </text>
                  ) : null}
                  {w > 56 ? (
                    <text x={x + w / 2 - 1} y={84} textAnchor="middle" fill="var(--viz-ink-2)">
                      {fmtBytes(r.bytes)}
                    </text>
                  ) : null}
                  {r.kind === 'data' && r.idx === 0 ? null : null}
                </g>
              );
            })}
            <text x={mapPad} y={100} fill="var(--viz-ink-muted)">
              data blocks, in key order
            </text>
          </svg>

          <svg
            width={detW}
            height={detH}
            role="img"
            aria-label={`Byte layout of data block ${showBlock ? showBlock.idx : 0}`}
          >
            <text x={0} y={14} fill="var(--viz-ink)" fontWeight={600}>
              data block {showBlock ? showBlock.idx : 0}
            </text>
            <text x={112} y={14} fill="var(--viz-ink-2)">
              {showBlock
                ? `${fmtNum(showBlock.raw)} B of entries + ${showBlock.restartBytes} B restart array + ${TRAILER} B trailer = ${fmtNum(showBlock.size)} B`
                : ''}
            </text>
            <text x={0} y={32} fill="var(--viz-ink-muted)">
              decoded key (grey = shared prefix, never written)
            </text>
            <text x={stripX} y={32} fill="var(--viz-ink-muted)">
              bytes on disk, to scale
            </text>

            {rows.map((e, i) => {
              const y = 42 + i * rowH;
              const inScan = scanFrom >= 0 && i >= scanFrom && i <= Math.max(scanFrom, scanTo);
              const sharedW = e.shared * byteScale;
              const hdrW = Math.max(3, e.hdr * byteScale);
              const deltaW = Math.max(2, e.nonShared * byteScale);
              const valW = Math.max(2, e.valueLen * byteScale);
              const keyTxt = e.key.length > 30 ? `${e.key.slice(0, 29)}...` : e.key;
              const sh = Math.min(e.shared, keyTxt.length);
              return (
                <g
                  key={e.i}
                  {...tip(
                    <>
                      <strong>entry #{e.i}</strong> - {e.key}
                      <br />
                      shared {e.shared} | non_shared {e.nonShared} | value_len {e.valueLen}
                      <br />
                      {e.hdr} B varint header + {e.nonShared} B key delta + {e.valueLen} B value = {e.bytes} B
                      <br />
                      {e.restart
                        ? 'Restart point: shared = 0, so the whole internal key is stored here and a reader can compare it without decoding anything before it.'
                        : `Only the last ${e.nonShared} bytes of the internal key are on disk; the first ${e.shared} come from the previous entry.`}
                    </>,
                  )}
                  style={{ cursor: 'help' }}
                >
                  {inScan ? (
                    <rect x={-2} y={y - 2} width={detW - 12} height={rowH - 2} rx={3} fill="var(--viz-neutral)" />
                  ) : null}
                  <text x={0} y={y + 12} fill="var(--viz-ink-muted)">
                    {e.i}
                  </text>
                  <text x={32} y={y + 12}>
                    <tspan fill="var(--viz-stale)">{keyTxt.slice(0, sh)}</tspan>
                    <tspan fill="var(--viz-ink)" fontWeight={600}>
                      {keyTxt.slice(sh)}
                    </tspan>
                  </text>
                  {e.restart ? (
                    <>
                      <rect x={keyCol} y={y + 1} width={16} height={14} rx={3} fill="var(--viz-warning)" />
                      <text x={keyCol + 8} y={y + 12} textAnchor="middle" fill="var(--viz-ink)" fontWeight={700}>
                        R
                      </text>
                    </>
                  ) : null}
                  <rect
                    x={stripX}
                    y={y + 2}
                    width={Math.max(1, sharedW)}
                    height={12}
                    rx={2}
                    fill="var(--viz-stale)"
                    opacity={0.22}
                    stroke="var(--viz-stale)"
                    strokeDasharray="2 2"
                  />
                  <rect x={stripX + sharedW} y={y + 2} width={hdrW} height={12} fill="var(--viz-7)" />
                  <rect x={stripX + sharedW + hdrW} y={y + 2} width={deltaW} height={12} fill="var(--viz-1)" />
                  <rect
                    x={stripX + sharedW + hdrW + deltaW}
                    y={y + 2}
                    width={valW}
                    height={12}
                    rx={2}
                    fill="var(--viz-2)"
                    opacity={0.8}
                  />
                </g>
              );
            })}

            {showBlock && showBlock.entries.length > MAX_ROWS ? (
              <text x={32} y={42 + rows.length * rowH + 14} fill="var(--viz-ink-muted)">
                +{showBlock.entries.length - MAX_ROWS} more entries in this block
              </text>
            ) : null}

            {showBlock ? (
              <g
                {...tip(
                  <>
                    <strong>Block tail</strong>
                    <br />
                    {showBlock.restarts.length} u32 restart offsets + a u32 count, then the 5-byte trailer: 1 byte
                    compression type and a 4-byte CRC32C over the block contents plus that type byte.
                  </>,
                )}
                style={{ cursor: 'help' }}
              >
                <rect
                  x={0}
                  y={detH - 50}
                  width={Math.max(200, stripW * 0.6)}
                  height={20}
                  rx={4}
                  fill="var(--viz-4)"
                  opacity={0.85}
                  stroke={activeStep?.title.includes('restarts') ? 'var(--viz-warning)' : 'var(--viz-surface)'}
                  strokeWidth={activeStep?.title.includes('restarts') ? 3 : 1}
                />
                <text x={8} y={detH - 36} fill="var(--viz-ink)" fontWeight={600}>
                  restart array: {showBlock.restarts.length} x u32 + count = {showBlock.restartBytes} B
                </text>
                <rect
                  x={Math.max(208, stripW * 0.6 + 8)}
                  y={detH - 50}
                  width={150}
                  height={20}
                  rx={4}
                  fill="var(--viz-8)"
                  opacity={0.85}
                />
                <text x={Math.max(216, stripW * 0.6 + 16)} y={detH - 36} fill="var(--viz-surface)" fontWeight={600}>
                  type byte + CRC32C = 5 B
                </text>
                <text x={0} y={detH - 12} fill="var(--viz-ink-muted)">
                  the restart array is what makes a block binary-searchable without decoding it first
                </text>
              </g>
            ) : null}
          </svg>

          <svg width={pathW} height={pathH} role="img" aria-label="The Get() path through the file, step by step">
            <text x={4} y={12} fill="var(--viz-ink-2)">
              Get("{probe.key}") - {search.found ? 'found' : 'not present in this file'}
            </text>
            {search.steps.map((s, i) => {
              const x = 4 + i * stepW;
              const on = i === step - 1;
              const done = i < step;
              return (
                <g key={`${s.title}-${i}`}>
                  <rect
                    x={x}
                    y={20}
                    width={stepW - 10}
                    height={40}
                    rx={6}
                    fill={done ? 'var(--viz-neutral)' : 'var(--viz-plane)'}
                    stroke={on ? 'var(--viz-warning)' : 'var(--viz-border)'}
                    strokeWidth={on ? 3 : 1}
                  />
                  <text x={x + 10} y={36} fill="var(--viz-ink)" fontWeight={on ? 700 : 500}>
                    {i + 1}. {s.title.length > 22 ? `${s.title.slice(0, 21)}...` : s.title}
                  </text>
                  <text x={x + 10} y={52} fill="var(--viz-ink-2)">
                    {s.bytes > 0 ? `${fmtBytes(s.bytes)} read` : 'in memory'}
                  </text>
                  {i < search.steps.length - 1 ? (
                    <line x1={x + stepW - 9} x2={x + stepW - 1} y1={40} y2={40} stroke="var(--viz-axis)" strokeWidth={2} />
                  ) : null}
                </g>
              );
            })}
            <text x={4} y={pathH - 6} fill="var(--viz-ink-muted)">
              {mode === 'full'
                ? 'Full filter first: a negative answer costs one filter probe and no index or data I/O at all.'
                : mode === 'block'
                  ? 'Per-block filter: the index descent happens first, so even a negative answer pays for it.'
                  : 'No filter: every candidate file is searched to its data blocks before it can be ruled out.'}
            </text>
          </svg>
        </TooltipHost>
      </div>
    </VizPanel>
  );
}
