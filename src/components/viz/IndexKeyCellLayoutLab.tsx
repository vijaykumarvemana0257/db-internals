import { useId, useMemo, useState } from 'react';
import {
  VizPanel,
  Choice,
  Segmented,
  Check,
  Slider,
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
 * What a key costs, byte by byte, and what that does to fanout.
 *
 * The learner picks a realistic key shape and toggles the three space techniques a
 * B+tree has for the bytes in a cell:
 *
 *   suffix truncation  — a separator in an internal page only has to *separate*, so
 *                        nbtree's _bt_truncate keeps attributes up to and including the
 *                        first one where the boundary keys differ, and drops the rest
 *                        (including the implicit heap-TID column and any INCLUDE columns).
 *   prefix compression — front coding: a cell stores only the bytes it does not share
 *                        with the preceding cell, plus a length byte, with a full key
 *                        re-anchored every RESTART cells so binary search stays possible.
 *                        (WiredTiger, MyISAM PACK_KEYS, RocksDB block restarts.)
 *   deduplication      — PostgreSQL 13: a run of equal keys collapses into one posting
 *                        list tuple holding the key once and an array of 6-byte TIDs,
 *                        bounded by the same ~1/3-page item limit as any index tuple.
 *
 * Everything downstream — cells per leaf, separator width, internal fanout, level count,
 * index size — is derived from those byte counts, so the toggles move the tree, not a
 * decoration. Sizes follow PostgreSQL nbtree: 24 B PageHeaderData, 16 B BTPageOpaqueData,
 * 8 B IndexTupleData, MAXALIGN 8, a 4 B ItemIdData per item, one high key per page,
 * leaves packed to fillfactor 90 and internal pages packed full, as _bt_load does. A leaf
 * tuple's heap TID is t_tid inside that 8 B header, so only a posting list tuple pays extra
 * ItemPointerData bytes.
 */

/* ------------------------------------------------------------------ fixtures */

type Shape = 'composite' | 'lowcard' | 'uuid' | 'widepk';

type ShapeSpec = {
  id: Shape;
  label: string;
  attrNames: string[];
  /** Bytes of row pointer stored per leaf entry (6 = heap TID; 0 = the PK is in the key). */
  pointerBytes: number;
  pointerLabel: string;
  build: () => string[][];
};

const pad = (n: number, w: number) => String(n).padStart(w, '0');

function uuidFrom(rng: () => number) {
  const h = '0123456789abcdef';
  let s = '';
  for (let i = 0; i < 32; i++) {
    s += h[Math.floor(rng() * 16)];
    if (i === 7 || i === 11 || i === 15 || i === 19) s += '-';
  }
  return s;
}

const TENANTS = [
  'acme-logistics-eu-west-1',
  'acme-logistics-us-east-1',
  'bluewave-media-ap-southeast-2',
  'globex-retail-eu-central-1',
  'initech-payments-us-west-2',
  'umbrella-health-us-east-1',
];

const EVENTS = [
  'checkout.session.completed',
  'checkout.session.expired',
  'invoice.payment_failed',
  'invoice.payment_succeeded',
  'order.fulfilled',
  'order.refunded',
  'subscription.updated',
];

function buildComposite(): string[][] {
  const rng = makeRng(90210);
  const out: string[][] = [];
  for (const tenant of TENANTS) {
    for (const event of EVENTS) {
      let sec = 3600 + Math.floor(rng() * 600);
      const n = 9 + Math.floor(rng() * 5);
      for (let i = 0; i < n; i++) {
        sec += 20 + Math.floor(rng() * 900);
        const day = 14 + Math.floor(sec / 86400);
        const rest = sec % 86400;
        const ts = `2026-03-${pad(day, 2)}T${pad(Math.floor(rest / 3600), 2)}:${pad(
          Math.floor((rest % 3600) / 60),
          2,
        )}:${pad(rest % 60, 2)}.${pad(Math.floor(rng() * 1000), 3)}Z`;
        out.push([tenant, event, ts, uuidFrom(rng)]);
      }
    }
  }
  return out;
}

function buildLowCard(): string[][] {
  const rng = makeRng(4242);
  const statuses: [string, number][] = [
    ['active', 146],
    ['archived', 88],
    ['cancelled', 37],
    ['delivered', 121],
    ['pending', 63],
    ['processing', 29],
    ['refunded', 16],
  ];
  const out: string[][] = [];
  for (const [s, n] of statuses) for (let i = 0; i < n; i++) out.push([s]);
  // touch the rng so the fixture is generated the same way as the others
  rng();
  return out;
}

function buildUuid(): string[][] {
  const rng = makeRng(777);
  const out: string[] = [];
  for (let i = 0; i < 500; i++) out.push(uuidFrom(rng));
  out.sort();
  return out.map((u) => [u]);
}

const FIRST = ['ada', 'bram', 'chen', 'dara', 'esi', 'fiona', 'gus', 'hana', 'ivo', 'juno', 'kai', 'lena'];
const LAST = ['abbott', 'bhatt', 'cortez', 'duarte', 'engel', 'fontaine', 'gupta', 'hollis', 'ingram', 'jarvis'];
const DOMAIN = ['globex.example', 'initech.example', 'umbrella.example'];

function buildWidePk(): string[][] {
  const rng = makeRng(31337);
  const seen = new Set<string>();
  const emails: string[] = [];
  for (const l of LAST)
    for (const f of FIRST)
      for (const d of DOMAIN) {
        const e = `${f}.${l}${Math.floor(rng() * 90) + 10}@${d}`;
        if (!seen.has(e)) {
          seen.add(e);
          emails.push(e);
        }
      }
  emails.sort();
  const rng2 = makeRng(6161);
  return emails.map((e) => [e, uuidFrom(rng2)]);
}

const SHAPES: Record<Shape, ShapeSpec> = {
  composite: {
    id: 'composite',
    label: 'Composite: (tenant, event_type, occurred_at, id)',
    attrNames: ['tenant', 'event_type', 'occurred_at', 'id'],
    pointerBytes: 6,
    pointerLabel: 'heap TID (6 B)',
    build: buildComposite,
  },
  lowcard: {
    id: 'lowcard',
    label: 'Low cardinality: (status) — long duplicate runs',
    attrNames: ['status'],
    pointerBytes: 6,
    pointerLabel: 'heap TID (6 B)',
    build: buildLowCard,
  },
  uuid: {
    id: 'uuid',
    label: 'Random: (uuid v4 as text)',
    attrNames: ['id'],
    pointerBytes: 6,
    pointerLabel: 'heap TID (6 B)',
    build: buildUuid,
  },
  widepk: {
    id: 'widepk',
    label: 'InnoDB-style secondary: (email) + 36 B clustered PK',
    attrNames: ['email', 'pk'],
    pointerBytes: 0,
    pointerLabel: 'clustered PK is the last key column',
    build: buildWidePk,
  },
};

/* --------------------------------------------------------------------- model */

const ALIGN = 8;
const HEADER = 8; // IndexTupleData: 6 B t_tid + 2 B t_info
const LP = 4; // ItemIdData
const PAGE_HDR = 24; // PageHeaderData
const OPAQUE = 16; // BTPageOpaqueData
const FILL = 0.9; // nbtree default fillfactor for leaves
const RESTART = 16; // front-coding anchor interval (RocksDB's block_restart_interval)
const MIN_PREFIX = 4; // minimum shared bytes before front coding is worth a length byte

const maxalign = (n: number) => Math.ceil(n / ALIGN) * ALIGN;
const maxalignDown = (n: number) => Math.floor(n / ALIGN) * ALIGN;

/** BTMaxItemSize: one index tuple may not exceed ~1/3 of a page, less room for a heap TID. */
function maxItemSize(pageSize: number) {
  return maxalignDown((pageSize - maxalign(PAGE_HDR + 3 * LP) - OPAQUE) / 3) - ALIGN;
}

function sharedPrefix(a: string, b: string) {
  const n = Math.min(a.length, b.length);
  let i = 0;
  while (i < n && a.charCodeAt(i) === b.charCodeAt(i)) i++;
  return i;
}

type Cell = {
  keyBytes: number; // full key bytes this cell represents
  stored: number; // key bytes actually written into the cell
  elided: number; // key bytes the shared prefix saved
  tids: number; // row pointers packed into this cell
  total: number; // maxalign(HEADER + stored + tids*ptr) + LP
  anchor: boolean;
};

type Model = {
  cells: Cell[];
  avgLeafCell: number;
  avgSepCell: number;
  avgSepKeyBytes: number;
  avgFullKeyBytes: number;
  cellsPerRow: number;
  keysPerLeaf: number;
  fanout: number;
  leafPages: number;
  internalPages: number;
  totalPages: number;
  height: number;
  bytes: number;
  bytesPerRow: number;
  sepAttrsKept: number;
};

type Opts = { truncate: boolean; prefix: boolean; dedup: boolean; pageSize: number; rows: number };

function compute(spec: ShapeSpec, keys: string[][], o: Opts): Model {
  const strs = keys.map((k) => k.join('|'));
  const ptr = spec.pointerBytes;
  const maxItem = maxItemSize(o.pageSize);

  /* ---- runs of equal keys in the sample, scaled up to the real row count ---- */
  const n = strs.length;
  type Run = { key: string; count: number; prevKey: string | null };
  const runs: Run[] = [];
  for (let i = 0; i < n; i++) {
    const last = runs[runs.length - 1];
    if (last && last.key === strs[i]) last.count++;
    else runs.push({ key: strs[i], count: 1, prevKey: i > 0 ? strs[i - 1] : null });
  }

  /* Front coding: a cell stores only what it does not share with its predecessor, minus
     one byte for the shared-length field, and one cell in RESTART re-anchors at full length. */
  const elideOf = (sp: number) => (o.prefix && sp >= MIN_PREFIX ? (sp - 1) * (1 - 1 / RESTART) : 0);

  let tuples = 0;
  let totalBytes = 0;
  const cells: Cell[] = [];
  for (const r of runs) {
    const scaled = (r.count / n) * o.rows; // rows this key covers at full scale
    const keyBytes = r.key.length;
    // A run of length 1 in the sample stands for that many *distinct* neighbouring keys;
    // only a genuine duplicate run can be deduplicated or share its whole key with itself.
    const dup = r.count > 1;
    // A posting list tuple is bounded by BTMaxItemSize like any other index tuple.
    const cap =
      o.dedup && ptr > 0 && dup ? Math.max(1, Math.floor((maxItem - maxalign(HEADER + keyBytes)) / ptr)) : 1;
    const t = Math.max(1, Math.ceil(scaled / cap));
    const tids = Math.max(1, Math.round(scaled / t));
    const eFirst = elideOf(r.prevKey ? sharedPrefix(r.prevKey, r.key) : 0);
    const eRest = dup ? elideOf(keyBytes) : eFirst; // a repeated key shares all of itself
    // A plain tuple's row pointer is t_tid, already inside HEADER; only a posting list
    // tuple appends an explicit ItemPointerData array after the key data.
    const sizeOf = (e: number) =>
      maxalign(HEADER + (keyBytes - e)) + (tids > 1 ? tids * ptr : 0) + LP;
    totalBytes += sizeOf(eFirst) + (t - 1) * sizeOf(eRest);
    tuples += t;

    const showMax = Math.min(t, dup ? 3 : 1); // keep the picture varied: a few cells per run
    for (let k = 0; k < showMax && cells.length < 7; k++) {
      const anchor = cells.length % RESTART === 0;
      const elided = anchor ? 0 : Math.round(k === 0 ? eFirst : eRest);
      cells.push({
        keyBytes,
        stored: keyBytes - elided,
        elided,
        tids,
        total: maxalign(HEADER + keyBytes - elided) + (tids > 1 ? tids * ptr : 0) + LP,
        anchor,
      });
    }
  }

  const avgLeafCell = totalBytes / tuples;
  const avgFullKeyBytes = strs.reduce((s, k) => s + k.length, 0) / n;

  /* ---- separators: what the parent has to store for each page boundary ---- */
  const sepStrs: string[] = [];
  let keptTotal = 0;
  for (let i = 1; i < keys.length; i++) {
    const left = keys[i - 1];
    const right = keys[i];
    if (!o.truncate) {
      sepStrs.push(strs[i]);
      keptTotal += right.length;
      continue;
    }
    let d = -1;
    for (let a = 0; a < right.length; a++)
      if (left[a] !== right[a]) {
        d = a;
        break;
      }
    if (d === -1) {
      // identical in every column: nbtree falls back on the heap TID as the tiebreaker
      sepStrs.push(strs[i]);
      keptTotal += right.length;
    } else {
      sepStrs.push(right.slice(0, d + 1).join('|'));
      keptTotal += d + 1;
    }
  }
  const sepExtra = o.truncate ? 0 : ptr; // an untruncated separator carries the row pointer too
  let sepBytes = 0;
  for (let i = 0; i < sepStrs.length; i++) {
    let elided = 0;
    if (o.prefix && i > 0 && i % RESTART !== 0) {
      const sp = sharedPrefix(sepStrs[i - 1], sepStrs[i]);
      if (sp >= MIN_PREFIX) elided = sp - 1;
    }
    sepBytes += sepStrs[i].length - elided;
  }
  const avgSepKeyBytes = sepBytes / sepStrs.length;
  const avgSepCell = maxalign(HEADER + avgSepKeyBytes + sepExtra) + LP;

  /* ---- pack the pages ---- */
  const leafUsable = o.pageSize - PAGE_HDR - OPAQUE - avgSepCell; // every page carries a high key
  const keysPerLeaf = Math.max(2, Math.floor((leafUsable * FILL) / avgLeafCell));
  const internalUsable = o.pageSize - PAGE_HDR - OPAQUE - avgSepCell;
  const fanout = Math.max(2, Math.floor(internalUsable / avgSepCell));

  const cellsPerRow = tuples / o.rows;
  const leafPages = Math.max(1, Math.ceil(tuples / keysPerLeaf));
  let pages = leafPages;
  let internalPages = 0;
  let height = 1;
  while (pages > 1) {
    pages = Math.ceil(pages / fanout);
    internalPages += pages;
    height++;
  }
  const totalPages = leafPages + internalPages + 1; // + the nbtree metapage
  const bytes = totalPages * o.pageSize;

  return {
    cells,
    avgLeafCell,
    avgSepCell,
    avgSepKeyBytes,
    avgFullKeyBytes,
    cellsPerRow,
    keysPerLeaf,
    fanout,
    leafPages,
    internalPages,
    totalPages,
    height,
    bytes,
    bytesPerRow: bytes / o.rows,
    sepAttrsKept: keptTotal / (keys.length - 1),
  };
}

/* ----------------------------------------------------------------- component */

const C_KEY = 'var(--viz-1)';
const C_PTR = 'var(--viz-2)';
const C_OVH = 'var(--viz-3)';
const C_GONE = 'var(--viz-stale)';

export default function IndexKeyCellLayoutLab() {
  const [shape, setShape] = useState<Shape>('composite');
  const [pageSize, setPageSize] = useState('8192');
  const [rowsExp, setRowsExp] = useState(7);
  const [truncate, setTruncate] = useState(true);
  const [prefix, setPrefix] = useState(false);
  const [dedup, setDedup] = useState(false);
  const [ref, width] = useSize(760);
  const tip = useTip();
  const hatchId = useId().replace(/:/g, '_');

  const spec = SHAPES[shape];
  const keys = useMemo(() => spec.build(), [spec]);
  const rows = Math.round(10 ** rowsExp);
  const ps = Number(pageSize);
  const opts: Opts = { truncate, prefix, dedup, pageSize: ps, rows };

  const m = useMemo(() => compute(spec, keys, opts), [spec, keys, truncate, prefix, dedup, ps, rows]);
  const base = useMemo(
    () => compute(spec, keys, { ...opts, truncate: false, prefix: false, dedup: false }),
    [spec, keys, ps, rows],
  );

  /* the boundary pair used for the separator picture: the first pair that actually differs */
  const pair = useMemo(() => {
    for (let i = 1; i < keys.length; i++) {
      for (let a = 0; a < keys[i].length; a++)
        if (keys[i - 1][a] !== keys[i][a]) return { left: keys[i - 1], right: keys[i], d: a };
    }
    return { left: keys[0], right: keys[1] ?? keys[0], d: keys[0].length - 1 };
  }, [keys]);

  const W = Math.max(680, Math.min(width, 980));
  const LABEL = 118;
  const RIGHT = 78;
  const BAR = W - LABEL - RIGHT;

  const shown = m.cells.slice(0, 7);
  const maxCell = Math.max(...shown.map((c) => c.total + c.elided), 1);
  const px = BAR / maxCell;

  const rowH = 21;
  const aTop = 30;
  const bTop = aTop + shown.length * rowH + 34;
  const cTop = bTop + 2 * 26 + 44;
  const stackOf = (leafPages: number, fanout: number) => {
    let p = leafPages;
    const stack: { label: string; pages: number }[] = [{ label: 'leaves', pages: p }];
    let lvl = 1;
    while (p > 1) {
      p = Math.ceil(p / fanout);
      stack.push({ label: p === 1 ? 'root' : `internal L${lvl}`, pages: p });
      lvl++;
    }
    return stack; // index 0 = leaves
  };
  const stack = stackOf(m.leafPages, m.fanout);
  const baseStack = stackOf(base.leafPages, base.fanout);
  const levels = [...stack].reverse();
  const maxPages = Math.max(m.leafPages, base.leafPages);
  const lscale = (p: number) => Math.max(3, (Math.log10(p + 1) / Math.log10(maxPages + 1)) * BAR);
  const height = cTop + levels.length * 22 + 26;

  const sepKeyPx = Math.min(BAR / Math.max(m.avgFullKeyBytes, 1), 7);
  const delta = (m.bytes - base.bytes) / base.bytes;

  return (
    <VizPanel
      title="What a key costs, and what that costs the tree"
      subtitle="Toggle the three byte-level techniques and watch cell width, separator width, internal fanout and total pages move together."
      controls={
        <>
          <Choice
            label="Key shape"
            value={shape}
            onChange={(v) => setShape(v as Shape)}
            options={Object.values(SHAPES).map((s) => ({ value: s.id, label: s.label }))}
          />
          <Segmented
            label="Page size"
            value={pageSize}
            onChange={setPageSize}
            options={[
              { value: '4096', label: '4 KB' },
              { value: '8192', label: '8 KB' },
              { value: '16384', label: '16 KB' },
            ]}
          />
          <Slider
            label="Rows indexed"
            min={5}
            max={9}
            step={0.5}
            value={rowsExp}
            onChange={setRowsExp}
            format={() => fmtNum(rows)}
          />
          <Check label="Suffix truncation" checked={truncate} onChange={setTruncate} />
          <Check label="Prefix compression" checked={prefix} onChange={setPrefix} />
          <Check label="Deduplication" checked={dedup} onChange={setDedup} />
        </>
      }
      legend={
        <Legend
          items={[
            {
              label: spec.pointerBytes
                ? 'Key bytes stored in the cell'
                : 'Key bytes stored in the cell (the clustered PK is the last key column)',
              color: C_KEY,
            },
            ...(spec.pointerBytes
              ? [{ label: 'Posting list of heap TIDs (6 B each)', color: C_PTR }]
              : []),
            { label: 'Tuple header incl. t_tid + line pointer (12 B)', color: C_OVH },
            { label: 'Bytes not stored (elided prefix / truncated suffix)', color: C_GONE },
          ]}
        />
      }
      stats={
        <Stats
          items={[
            { label: 'Index size', value: fmtBytes(m.bytes), hint: `${fmtNum(m.totalPages)} pages of ${fmtBytes(ps)}` },
            { label: 'vs no compression', value: `${delta >= 0 ? '+' : ''}${(delta * 100).toFixed(0)}%` },
            { label: 'Avg leaf cell', value: `${m.avgLeafCell.toFixed(0)} B`, hint: 'MAXALIGN(8 + key + TIDs) + 4' },
            { label: 'Avg separator cell', value: `${m.avgSepCell.toFixed(0)} B` },
            { label: 'Internal fanout', value: fmtNum(m.fanout) },
            { label: 'Height', value: `${m.height} level${m.height === 1 ? '' : 's'}` },
          ]}
        />
      }
      note={
        <Note>
          <strong>
            {m.avgSepCell < base.avgSepCell
              ? `Separators are ${(base.avgSepCell / m.avgSepCell).toFixed(1)}× narrower than the full key, so one internal page routes ${fmtNum(m.fanout)} children instead of ${fmtNum(base.fanout)}.`
              : `Separators carry the whole key, so one internal page routes only ${fmtNum(m.fanout)} children.`}
          </strong>{' '}
          {m.height === base.height
            ? `The tree is still ${m.height} levels deep — fanout has to change by a large factor before a level disappears, which is why the win shows up as ${fmtBytes(base.bytes - m.bytes)} of pages and cache residency rather than as a shorter descent.`
            : `That is enough to drop a whole level: ${base.height} → ${m.height}, one fewer page read on every point lookup.`}{' '}
          {dedup && m.cellsPerRow < 0.9
            ? `Deduplication folded ${(1 / m.cellsPerRow).toFixed(0)} rows per posting list tuple, so a leaf holds ${fmtNum(Math.round(m.keysPerLeaf / m.cellsPerRow))} rows instead of ${fmtNum(base.keysPerLeaf)}.`
            : ''}
        </Note>
      }
      table={
        <table className="viz-table">
          <caption style={{ captionSide: 'top', textAlign: 'left', paddingBottom: '.4rem' }}>
            Same {fmtNum(rows)} rows, {fmtBytes(ps)} pages, each technique applied on its own and together.
          </caption>
          <thead>
            <tr>
              <th>Configuration</th>
              <th>Leaf cell</th>
              <th>Separator cell</th>
              <th>Cells/leaf</th>
              <th>Fanout</th>
              <th>Leaf pages</th>
              <th>Height</th>
              <th>Index size</th>
            </tr>
          </thead>
          <tbody>
            {(
              [
                ['None (textbook full keys)', { truncate: false, prefix: false, dedup: false }],
                ['Suffix truncation', { truncate: true, prefix: false, dedup: false }],
                ['Prefix compression', { truncate: false, prefix: true, dedup: false }],
                ['Deduplication', { truncate: false, prefix: false, dedup: true }],
                ['All three', { truncate: true, prefix: true, dedup: true }],
              ] as [string, Partial<Opts>][]
            ).map(([label, over]) => {
              const r = compute(spec, keys, { ...opts, ...over });
              return (
                <tr key={label}>
                  <td>{label}</td>
                  <td>{r.avgLeafCell.toFixed(0)} B</td>
                  <td>{r.avgSepCell.toFixed(0)} B</td>
                  <td>{fmtNum(r.keysPerLeaf)}</td>
                  <td>{fmtNum(r.fanout)}</td>
                  <td>{fmtNum(r.leafPages)}</td>
                  <td>{r.height}</td>
                  <td>{fmtBytes(r.bytes)}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      }
    >
      <TooltipHost>
        <div ref={ref}>
          <svg width={W} height={height} role="img" aria-label="Index cell layout, separator truncation and resulting tree shape">
            <defs>
              <pattern id={hatchId} width={6} height={6} patternUnits="userSpaceOnUse" patternTransform="rotate(45)">
                <rect width={6} height={6} fill="var(--viz-plane)" />
                <line x1={0} y1={0} x2={0} y2={6} stroke={C_GONE} strokeWidth={2} />
              </pattern>
            </defs>

            {/* ---------------------------------------- A: cells in a leaf page */}
            <text x={0} y={12} fill="var(--viz-ink)" fontWeight={600}>
              First cells of one leaf page, at byte scale
            </text>
            <text x={0} y={26} fill="var(--viz-ink-muted)">
              {m.cellsPerRow < 1
                ? `${fmtNum(m.keysPerLeaf)} cells per page covering ${fmtNum(Math.round(m.keysPerLeaf / m.cellsPerRow))} rows`
                : `${fmtNum(m.keysPerLeaf)} cells per page`}
            </text>
            {shown.map((c, i) => {
              const y = aTop + 8 + i * rowH;
              const ovh = LP + HEADER;
              let x = LABEL;
              const segs: { w: number; fill: string; name: string }[] = [
                {
                  w: ovh * px,
                  fill: C_OVH,
                  name: `tuple header (t_tid + t_info) + line pointer, ${ovh} B`,
                },
              ];
              if (c.elided > 0)
                segs.push({ w: c.elided * px, fill: `url(#${hatchId})`, name: `${c.elided} B shared with the previous cell — not stored` });
              segs.push({ w: c.stored * px, fill: C_KEY, name: `${c.stored} B of key` });
              if (spec.pointerBytes > 0 && c.tids > 1)
                segs.push({
                  w: c.tids * spec.pointerBytes * px,
                  fill: C_PTR,
                  name: `posting list of ${c.tids} heap TIDs, ${c.tids * spec.pointerBytes} B`,
                });
              return (
                <g key={i}>
                  <text x={0} y={y + 12} fill="var(--viz-ink-2)">
                    {c.tids > 1 ? `posting ×${c.tids}` : `cell ${i}`}
                    {c.anchor && prefix ? ' ⚓' : ''}
                  </text>
                  {segs.map((s, j) => {
                    const el = (
                      <rect
                        key={j}
                        x={x}
                        y={y}
                        width={Math.max(1, s.w)}
                        height={15}
                        rx={2}
                        fill={s.fill}
                        stroke="var(--viz-border)"
                        {...tip(
                          <>
                            <strong>{s.name}</strong>
                            <br />
                            cell total {c.total} B
                          </>,
                        )}
                      />
                    );
                    x += s.w;
                    return el;
                  })}
                  <text x={LABEL + (c.total + c.elided) * px + 8} y={y + 12} fill="var(--viz-ink-muted)">
                    {c.total} B
                  </text>
                </g>
              );
            })}

            {/* ------------------------------ B: the separator promoted upwards */}
            <text x={0} y={bTop - 12} fill="var(--viz-ink)" fontWeight={600}>
              The separator this boundary promotes into the parent
            </text>
            {[0, 1].map((row) => {
              const y = bTop + row * 26;
              let x = LABEL;
              return (
                <g key={row}>
                  <text x={0} y={y + 12} fill="var(--viz-ink-2)">
                    {row === 0 ? 'boundary key' : 'separator'}
                  </text>
                  {pair.right.map((a, ai) => {
                    const kept = !truncate || ai <= pair.d;
                    const w = Math.max(10, (a.length + 1) * sepKeyPx);
                    const dim = row === 1 && !kept;
                    const el = (
                      <g key={ai}>
                        <rect
                          x={x}
                          y={y}
                          width={w}
                          height={15}
                          rx={2}
                          fill={dim ? `url(#${hatchId})` : C_KEY}
                          stroke="var(--viz-border)"
                          {...tip(
                            <>
                              <strong>{spec.attrNames[ai]}</strong> — {a.length} B
                              <br />
                              {dim ? 'truncated away: it cannot change the routing decision' : 'kept: needed to separate the two pages'}
                              <br />
                              <code>{a.length > 28 ? `${a.slice(0, 28)}…` : a}</code>
                            </>,
                          )}
                        />
                        {w > 46 ? (
                          <text x={x + 4} y={y + 12} fill={dim ? 'var(--viz-ink-2)' : 'var(--viz-surface)'}>
                            {spec.attrNames[ai]}
                          </text>
                        ) : null}
                      </g>
                    );
                    x += w + 2;
                    return el;
                  })}
                  {row === 1 && spec.pointerBytes > 0 ? (
                    <rect
                      x={x}
                      y={y}
                      width={Math.max(8, spec.pointerBytes * sepKeyPx)}
                      height={15}
                      rx={2}
                      fill={truncate ? `url(#${hatchId})` : C_PTR}
                      stroke="var(--viz-border)"
                      {...tip(<strong>{truncate ? 'heap TID truncated away too' : 'heap TID kept in the separator'}</strong>)}
                    />
                  ) : null}
                  <text x={W - RIGHT + 4} y={y + 12} fill="var(--viz-ink-muted)">
                    {row === 0
                      ? `${pair.right.join('|').length} B`
                      : `${m.avgSepKeyBytes.toFixed(0)} B avg`}
                  </text>
                </g>
              );
            })}

            {/* -------------------------------------------- C: resulting levels */}
            <text x={0} y={cTop - 12} fill="var(--viz-ink)" fontWeight={600}>
              Resulting tree — bars are pages per level (log scale), dashed outline is the same index with no compression
            </text>
            {levels.map((lv, i) => {
              const y = cTop + i * 22;
              const baseLevelPages = baseStack[levels.length - 1 - i]?.pages ?? 0;
              return (
                <g key={lv.label}>
                  <text x={0} y={y + 12} fill="var(--viz-ink-2)">
                    {lv.label}
                  </text>
                  <rect
                    x={LABEL}
                    y={y}
                    width={lscale(lv.pages)}
                    height={15}
                    rx={2}
                    fill={lv.label === 'leaves' ? C_KEY : C_OVH}
                    {...tip(
                      <>
                        <strong>{fmtNum(lv.pages)} pages</strong>
                        <br />
                        {fmtBytes(lv.pages * ps)} at this level
                      </>,
                    )}
                  />
                  {baseLevelPages > 0 ? (
                    <rect
                      x={LABEL}
                      y={y - 2}
                      width={lscale(baseLevelPages)}
                      height={19}
                      rx={2}
                      fill="none"
                      stroke="var(--viz-ink-muted)"
                      strokeDasharray="3 3"
                    />
                  ) : null}
                  <text
                    x={LABEL + Math.max(lscale(lv.pages), lscale(Math.max(1, baseLevelPages))) + 8}
                    y={y + 12}
                    fill="var(--viz-ink-muted)"
                  >
                    {fmtNum(lv.pages)}
                    {baseLevelPages > lv.pages ? ` (was ${fmtNum(baseLevelPages)})` : ''}
                  </text>
                </g>
              );
            })}
          </svg>
        </div>
      </TooltipHost>
    </VizPanel>
  );
}
