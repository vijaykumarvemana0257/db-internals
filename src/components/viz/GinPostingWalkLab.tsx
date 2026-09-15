import { useMemo, useRef, useState, type ReactNode } from 'react';
import { VizPanel, Segmented, Choice, Slider, Button, Legend, Stats, Note, fmtNum, useTicker } from './Viz';
import {
  type Json,
  type Kind,
  type Opclass,
  extractKeys,
  rowKeys,
  rowValue,
  tidCode,
  tidLabel,
  postingListBytes,
  postingTreeShape,
  showValue,
  HEAP_TUPLES_PER_PAGE,
  INLINE_LIMIT_BYTES,
  POSTING_LEAF_TIDS,
  POSTING_SEGMENT_TIDS,
} from './GinPostingListLab';

/**
 * How a GIN scan answers @>, && and ? — following gingetbitmap / startScanKey / keyGetItem / entryGetItem /
 * entryLoadMoreItems in src/backend/access/gin/ginget.c.
 *
 * - The pending list is scanned FIRST, row by row (each row's sorted entries binary-searched for the query keys),
 *   and matches go straight into the TID bitmap.
 * - Each query key is a scan entry. An inline posting list is decoded whole when the scan starts; a posting tree
 *   loads one leaf page at a time.
 * - startScanKey sorts entries by estimated size and calls triConsistent with the rarest entries FALSE and the rest
 *   MAYBE: the entries needed before a match is possible are "required", the rest "additional". For @> (AND) that
 *   is the single rarest entry; for && (OR) every entry is required.
 * - keyGetItem takes the smallest current TID among required entries, advances additional entries just up to it
 *   (skipping everything smaller), and calls consistent. A posting tree that must move past its loaded page steps
 *   right if the next TID is adjacent, otherwise it re-descends from the posting-tree root to the leaf holding the
 *   target and decodes only the segments from there on (GinDataLeafPageGetItems).
 * - Results are exact for array @> and &&; jsonb_ops and jsonb_path_ops always set recheck, so each candidate heap
 *   row is fetched and the operator re-evaluated.
 * Lab scaling: posting-tree leaf pages hold 16 TIDs in segments of 4, inner pages fan out 8, and a list over
 * 48 bytes becomes a posting tree. The main structure holds rows [0, rows − pending); the rest sit in the
 * pending list.
 */

export type Op = 'contains' | 'overlap' | 'exists';
export type QueryPreset = { id: string; kind: Kind; op: Op; sql: string; value: string[] | Json; blurb: string };

export const QUERIES: QueryPreset[] = [
  { id: 'rare-common', kind: 'tags', op: 'contains', sql: "tags @> '{gin,postgres}'", value: ['gin', 'postgres'], blurb: 'rare AND common' },
  { id: 'common-common', kind: 'tags', op: 'contains', sql: "tags @> '{linux,postgres}'", value: ['linux', 'postgres'], blurb: 'common AND common' },
  { id: 'recent-common', kind: 'tags', op: 'contains', sql: "tags @> '{rust,postgres}'", value: ['rust', 'postgres'], blurb: 'recent AND common' },
  { id: 'overlap', kind: 'tags', op: 'overlap', sql: "tags && '{gin,rust}'", value: ['gin', 'rust'], blurb: 'OR of two rare tags' },
  { id: 'jsonb-contains', kind: 'jsonb', op: 'contains', sql: `doc @> '{"status":"active","owner":{"team":"db"}}'`, value: { status: 'active', owner: { team: 'db' } }, blurb: 'containment' },
  { id: 'jsonb-exists', kind: 'jsonb', op: 'exists', sql: "doc ? 'team'", value: 'team', blurb: 'top-level key exists' },
];

export function jsonbContains(doc: Json, q: Json): boolean {
  if (q !== null && typeof q === 'object' && !Array.isArray(q)) {
    if (doc === null || typeof doc !== 'object' || Array.isArray(doc)) return false;
    return Object.keys(q).every((k) => k in doc && jsonbContains(doc[k], q[k]));
  }
  if (Array.isArray(q)) {
    if (!Array.isArray(doc)) return false;
    return q.every((qe) => doc.some((de) => jsonbContains(de, qe)));
  }
  return doc === q;
}

/** The real operator, evaluated on the heap row: the recheck, and the ground truth. */
export function heapMatches(q: QueryPreset, row: number) {
  const v = rowValue(q.kind, row);
  if (q.kind === 'tags') {
    const tags = v as string[];
    const qs = q.value as string[];
    return q.op === 'overlap' ? qs.some((t) => tags.includes(t)) : qs.every((t) => tags.includes(t));
  }
  if (q.op === 'exists') return Object.prototype.hasOwnProperty.call(v as object, q.value as string);
  return jsonbContains(v as Json, q.value as Json);
}

const MIN = 0;
const MAX = Number.MAX_SAFE_INTEGER;

type Decode = { event: number; from: number; to: number };
type PageRead = { event: number; page: number; how: 'descend' | 'right' };

export type EntryInfo = {
  id: string;
  label: string;
  detail: string;
  rows: number[];
  codes: number[];
  bytes: number;
  tree: boolean;
  pages: number;
  height: number;
  predict: number;
  required: boolean;
  decodes: Decode[];
  pageReads: PageRead[];
};

type Runtime = {
  e: EntryInfo;
  list: number[]; // codes of the loaded batch
  base: number; // index into e.codes of list[0]
  offset: number;
  cur: number;
  finished: boolean;
  released: boolean;
  page: number;
};

export type TraceEvent = {
  kind: 'start' | 'consistent' | 'done';
  row: number; // candidate row (-1 for start)
  match: boolean;
  cursors: (number | null)[]; // current row per entry, null when finished
  res: boolean[];
};

export type ScanResult = {
  unsupported: string | null;
  searchAll: boolean;
  entries: EntryInfo[];
  trace: TraceEvent[];
  pending: { rows: number; entries: number; matches: number[] };
  mainMatches: number[];
  recheck: boolean;
  candidates: number[];
  confirmed: number[];
  discarded: number[];
  truth: number[];
  totals: Totals;
};
export type Totals = { tids: number; decoded: number; leafPages: number; leafReads: number; descents: number; consistentCalls: number };

function triConsistent(op: Op, res: ('T' | 'F' | 'M')[]) {
  if (op === 'overlap') return res.includes('T') ? 'T' : res.every((r) => r === 'F') ? 'F' : 'M';
  return res.includes('F') ? 'F' : res.every((r) => r === 'T') ? 'T' : 'M';
}

export function runScan(q: QueryPreset, opclass: Opclass, rows: number, pendingRows: number): ScanResult {
  const mainRows = Math.max(0, rows - pendingRows);
  const oc: Opclass = q.kind === 'tags' ? 'array_ops' : opclass;
  const empty: ScanResult = {
    unsupported: null,
    searchAll: false,
    entries: [],
    trace: [],
    pending: { rows: 0, entries: 0, matches: [] },
    mainMatches: [],
    recheck: false,
    candidates: [],
    confirmed: [],
    discarded: [],
    truth: [],
    totals: { tids: 0, decoded: 0, leafPages: 0, leafReads: 0, descents: 0, consistentCalls: 0 },
  };
  const truth: number[] = [];
  for (let r = 0; r < rows; r++) if (heapMatches(q, r)) truth.push(r);
  if (q.op === 'exists' && oc === 'jsonb_path_ops') {
    return { ...empty, truth, unsupported: 'jsonb_path_ops has no ? operator, so the planner cannot use this index: the query needs a sequential scan or a jsonb_ops index.' };
  }

  // extractQuery
  const qkeys = q.op === 'exists' ? [{ id: `1${q.value as string}`, label: `key:${q.value as string}`, detail: 'key (JGINFLAG_KEY)' }] : extractKeys(oc, q.value);
  const recheck = q.kind === 'jsonb';

  // posting lists of the main structure
  const lists = new Map<string, number[]>(qkeys.map((k) => [k.id, []]));
  for (let r = 0; r < mainRows; r++) for (const k of rowKeys(q.kind, oc, r)) lists.get(k.id)?.push(r);

  const infos: EntryInfo[] = qkeys.map((k) => {
    const rs = lists.get(k.id)!;
    const bytes = postingListBytes(rs);
    const tree = bytes > INLINE_LIMIT_BYTES;
    const shape = postingTreeShape(rs.length);
    const first = Math.min(POSTING_LEAF_TIDS, rs.length);
    return {
      id: k.id,
      label: k.label,
      detail: k.detail,
      rows: rs,
      codes: rs.map(tidCode),
      bytes,
      tree,
      pages: tree ? shape.leaves : 0,
      height: tree ? shape.height : 0,
      // predictNumberResult: list length, or (estimated leaf pages) x (items on the first leaf)
      predict: tree ? shape.leaves * first : rs.length,
      required: false,
      decodes: [],
      pageReads: [],
    };
  });

  // pending list first: every pending entry of every pending row is compared
  const qset = new Set(qkeys.map((k) => k.id));
  const pendingMatches: number[] = [];
  let pendingEntries = 0;
  for (let r = mainRows; r < rows; r++) {
    const ks = rowKeys(q.kind, oc, r);
    pendingEntries += ks.length;
    const present = qkeys.map((k) => ks.some((x) => x.id === k.id));
    const ok = q.op === 'overlap' ? present.some(Boolean) : present.every(Boolean);
    if (ok && qset.size) pendingMatches.push(r);
  }

  // startScanKey: required / additional split
  const order = infos.map((_, i) => i).sort((a, b) => infos[a].predict - infos[b].predict || a - b);
  const res: ('T' | 'F' | 'M')[] = infos.map(() => 'M');
  let i = 0;
  if (infos.length > 1) {
    for (i = 0; i < infos.length - 1; i++) {
      res[order[i]] = 'F';
      if (triConsistent(q.op, res) === 'F') break;
    }
  }
  const nrequired = infos.length > 1 ? i + 1 : infos.length;
  order.forEach((idx, j) => (infos[idx].required = j < nrequired));
  const required = order.slice(0, nrequired);
  const additional = order.slice(nrequired);

  const trace: TraceEvent[] = [];
  let event = 0;
  const rts: Runtime[] = infos.map((e) => ({ e, list: [], base: 0, offset: 0, cur: MIN, finished: e.rows.length === 0, released: false, page: 0 }));

  const segStartFor = (pageStart: number, pageEnd: number, codes: number[], advancePast: number) => {
    let s = pageStart;
    while (s + POSTING_SEGMENT_TIDS < pageEnd && codes[s + POSTING_SEGMENT_TIDS] <= advancePast) s += POSTING_SEGMENT_TIDS;
    return s;
  };
  const loadPage = (rt: Runtime, page: number, advancePast: number) => {
    const { codes } = rt.e;
    const start = page * POSTING_LEAF_TIDS;
    const end = Math.min(codes.length, start + POSTING_LEAF_TIDS);
    const seg = segStartFor(start, end, codes, advancePast);
    rt.e.decodes.push({ event, from: seg, to: end });
    rt.list = codes.slice(seg, end);
    rt.base = seg;
    rt.offset = 0;
    rt.page = page;
    if (page === rt.e.pages - 1) rt.released = true;
  };

  // startScanEntry
  for (const rt of rts) {
    if (rt.finished) continue;
    if (!rt.e.tree) {
      rt.e.decodes.push({ event, from: 0, to: rt.e.codes.length });
      rt.list = rt.e.codes;
    } else {
      rt.e.pageReads.push({ event, page: 0, how: 'descend' });
      loadPage(rt, 0, MIN);
    }
  }

  const loadMore = (rt: Runtime, advancePast: number) => {
    const { codes, pages } = rt.e;
    let stepright: boolean;
    if (rt.cur === advancePast) stepright = true;
    else {
      let target = pages - 1;
      for (let p = 0; p < pages; p++) {
        const last = codes[Math.min(codes.length, (p + 1) * POSTING_LEAF_TIDS) - 1];
        if (last > advancePast) {
          target = p;
          break;
        }
      }
      rt.e.pageReads.push({ event, page: target, how: 'descend' });
      rt.page = target;
      stepright = false;
    }
    for (;;) {
      if (stepright) {
        if (rt.page >= pages - 1) {
          rt.finished = true;
          return;
        }
        rt.page++;
        rt.e.pageReads.push({ event, page: rt.page, how: 'right' });
      }
      stepright = true;
      const last = codes[Math.min(codes.length, (rt.page + 1) * POSTING_LEAF_TIDS) - 1];
      if (rt.page < pages - 1 && advancePast >= last) continue;
      loadPage(rt, rt.page, advancePast);
      const idx = rt.list.findIndex((c) => c > advancePast);
      if (idx >= 0) {
        rt.offset = idx;
        return;
      }
    }
  };

  const entryGetItem = (rt: Runtime, advancePast: number) => {
    if (rt.finished) return;
    for (;;) {
      if (rt.offset >= rt.list.length) {
        if (!rt.e.tree || rt.released) {
          rt.finished = true;
          return;
        }
        while (rt.offset >= rt.list.length) {
          loadMore(rt, advancePast);
          if (rt.finished) return;
        }
      }
      rt.cur = rt.list[rt.offset++];
      if (rt.cur <= advancePast) continue;
      return;
    }
  };

  trace.push({ kind: 'start', row: -1, match: false, cursors: rts.map(() => null), res: rts.map(() => false) });
  const mainMatches: number[] = [];
  let advancePast = MIN;
  let consistentCalls = 0;
  const codeToRow = (c: number) => {
    const blk = Math.floor(c / 2048);
    return blk * HEAP_TUPLES_PER_PAGE + (c - blk * 2048) - 1;
  };
  if (infos.length) {
    for (let guard = 0; guard < 100000; guard++) {
      event = trace.length;
      let minItem = MAX;
      let allFinished = true;
      for (const ri of required) {
        const rt = rts[ri];
        if (rt.finished) continue;
        if (rt.cur <= advancePast) {
          entryGetItem(rt, advancePast);
          if (rt.finished) continue;
        }
        allFinished = false;
        if (rt.cur < minItem) minItem = rt.cur;
      }
      if (allFinished) break;
      const past = minItem - 1;
      for (const ai of additional) {
        const rt = rts[ai];
        if (rt.finished) continue;
        if (rt.cur <= past) entryGetItem(rt, past);
      }
      const r = rts.map((rt) => !rt.finished && rt.cur === minItem);
      const match = q.op === 'overlap' ? r.some(Boolean) : r.every(Boolean);
      consistentCalls++;
      const row = codeToRow(minItem);
      trace.push({ kind: 'consistent', row, match, cursors: rts.map((rt) => (rt.finished ? null : codeToRow(rt.cur))), res: r });
      if (match) mainMatches.push(row);
      advancePast = minItem;
    }
  }
  event = trace.length;
  trace.push({ kind: 'done', row: -1, match: false, cursors: rts.map(() => null), res: rts.map(() => false) });

  const candidates = [...pendingMatches, ...mainMatches].sort((a, b) => a - b);
  const confirmed = candidates.filter((r) => heapMatches(q, r));
  const discarded = candidates.filter((r) => !heapMatches(q, r));
  const tids = infos.reduce((a, e) => a + e.rows.length, 0);
  const decoded = infos.reduce((a, e) => a + e.decodes.reduce((s, d) => s + d.to - d.from, 0), 0);
  const leafPages = infos.reduce((a, e) => a + e.pages, 0);
  const leafReads = infos.reduce((a, e) => a + e.pageReads.length, 0);
  const descents = infos.reduce((a, e) => a + e.pageReads.filter((p) => p.how === 'descend').length, 0);
  return {
    unsupported: null,
    searchAll: false,
    entries: infos,
    trace,
    pending: { rows: rows - mainRows, entries: pendingEntries, matches: pendingMatches },
    mainMatches,
    recheck,
    candidates,
    confirmed,
    discarded,
    truth,
    totals: { tids, decoded, leafPages, leafReads, descents, consistentCalls },
  };
}

/** Totals up to and including trace event `step`. */
export function progressAt(s: ScanResult, step: number) {
  const decoded = s.entries.map((e) => e.decodes.filter((d) => d.event <= step).reduce((a, d) => a + d.to - d.from, 0));
  const leafReads = s.entries.map((e) => e.pageReads.filter((p) => p.event <= step).length);
  const descents = s.entries.map((e) => e.pageReads.filter((p) => p.event <= step && p.how === 'descend').length);
  const calls = s.trace.slice(0, step + 1).filter((t) => t.kind === 'consistent');
  const sum = (xs: number[]) => xs.reduce((a, b) => a + b, 0);
  return {
    decoded,
    leafReads,
    descents,
    totalDecoded: sum(decoded),
    totalLeafReads: sum(leafReads),
    totalDescents: sum(descents),
    calls: calls.length,
    matches: calls.filter((t) => t.match).map((t) => t.row),
  };
}

const STEP_MS = 170;

export default function GinPostingWalkLab() {
  const [queryId, setQueryId] = useState('rare-common');
  const [opclass, setOpclass] = useState<Opclass>('jsonb_ops');
  const [rows, setRows] = useState(400);
  const [pendingRows, setPendingRows] = useState(0);
  const [step, setStep] = useState(1e9);
  const [playing, setPlaying] = useState(false);
  const acc = useRef(0);

  const q = QUERIES.find((x) => x.id === queryId) ?? QUERIES[0];
  const scan = useMemo(() => runScan(q, opclass, rows, pendingRows), [q, opclass, rows, pendingRows]);
  const last = Math.max(0, scan.trace.length - 1);
  const cur = Math.min(step, last);
  const done = cur >= last;
  const prog = useMemo(() => progressAt(scan, cur), [scan, cur]);
  const ev = scan.trace[cur];

  useTicker((dt) => {
    acc.current += dt;
    if (acc.current < STEP_MS) return;
    acc.current = 0;
    const next = Math.min(last, cur + 1);
    setStep(next);
    if (next >= last) setPlaying(false);
  }, playing && !done);

  const reset = (fn: () => void) => {
    fn();
    setPlaying(false);
    setStep(1e9);
  };

  const W = 680;
  const LX = 196;
  const RX = 668;
  const x = (row: number) => LX + ((row + 0.5) * (RX - LX)) / Math.max(1, rows);
  const Y0 = 36;
  const TH = 56;
  const n = scan.entries.length;
  const mainRows = rows - scan.pending.rows;
  const yBitmap = Y0 + n * TH + 8;
  const yRecheck = yBitmap + 38;
  const H = yRecheck + 40;

  const decodedMask = scan.entries.map((e) => {
    const m = new Uint8Array(e.rows.length);
    for (const d of e.decodes) if (d.event <= cur) m.fill(1, d.from, d.to);
    return m;
  });
  const readMask = scan.entries.map((e) => {
    const m = new Uint8Array(e.pages);
    for (const pr of e.pageReads) if (pr.event <= cur) m[pr.page] = 1;
    return m;
  });
  const currentReads = scan.entries.map((e) => e.pageReads.filter((pr) => pr.event === cur && cur > 0));
  const bitmapMain = new Set(prog.matches);
  const requiredEntries = scan.entries.filter((e) => e.required);
  const additionalEntries = scan.entries.filter((e) => !e.required);
  const shortLabel = (e: EntryInfo) => (q.kind === 'jsonb' && opclass === 'jsonb_path_ops' && q.op !== 'exists' ? e.detail.replace(' = ', '=').replace(/"/g, '') : e.label);
  const names = (es: EntryInfo[]) => es.map((e) => shortLabel(e)).join(', ');
  const kept = scan.confirmed.length;

  let note: ReactNode;
  if (scan.unsupported) {
    note = (
      <>
        <strong>Not indexable with jsonb_path_ops.</strong> {scan.unsupported} {fmtNum(scan.truth.length)} of {fmtNum(rows)} rows have a top-level <code>team</code> key.
      </>
    );
  } else if (!ev || ev.kind === 'start') {
    note = (
      <>
        <strong>Scan start.</strong>{' '}
        {scan.pending.rows > 0
          ? `The pending list is read first: all ${fmtNum(scan.pending.entries)} entries of its ${fmtNum(scan.pending.rows)} rows are read, each row’s sorted keys searched for the query keys, and ${fmtNum(scan.pending.matches.length)} rows go straight into the bitmap. `
          : 'The pending list is empty. '}
        startScanKey sorted the {n} entries by size: {names(requiredEntries)} {requiredEntries.length === 1 ? 'is' : 'are'} required
        {additionalEntries.length ? `; ${names(additionalEntries)} ${additionalEntries.length === 1 ? 'is' : 'are'} additional` : ''}. Inline posting lists are decoded in full now; each posting tree loads only its first leaf.
      </>
    );
  } else if (ev.kind === 'consistent') {
    const have = scan.entries.filter((_, i) => ev.res[i]);
    const lack = scan.entries.filter((_, i) => !ev.res[i]);
    note = (
      <>
        <strong>
          Candidate {tidLabel(ev.row)}: consistent() → {ev.match ? (scan.recheck ? 'TRUE, recheck' : 'TRUE') : 'FALSE'}.
        </strong>{' '}
        It is the smallest current TID among the required entries. {additionalEntries.length ? 'Additional entries skipped forward to it without testing the TIDs in between. ' : ''}
        {have.length ? `Has it: ${names(have)}. ` : ''}
        {lack.length ? `Lacks it: ${names(lack)}.` : ''}
      </>
    );
  } else {
    note = (
      <>
        <strong>
          consistent() ran {fmtNum(scan.totals.consistentCalls)} times for {fmtNum(scan.totals.tids)} TIDs in these posting lists.
        </strong>{' '}
        The scan decoded {fmtNum(scan.totals.decoded)} TIDs and loaded {fmtNum(scan.totals.leafReads)} of {fmtNum(scan.totals.leafPages)} posting-tree leaf pages ({fmtNum(scan.totals.descents)} root-to-leaf descents).{' '}
        {fmtNum(scan.candidates.length)} rows are in the bitmap
        {scan.recheck ? `; the heap recheck keeps ${fmtNum(kept)} and throws away ${fmtNum(scan.discarded.length)}.` : ', and array @> and && are exact, so every one is a real match.'}
        {scan.discarded.length ? ` First discarded row ${tidLabel(scan.discarded[0])}: ${showValue(rowValue(q.kind, scan.discarded[0]))}.` : ''}
      </>
    );
  }

  return (
    <VizPanel
      title="Answering a query from posting lists"
      subtitle="Each query key is a posting list of heap TIDs. GIN walks them in TID order: the rarest keys drive, the others skip forward to each candidate, and consistent() decides. Step through it."
      controls={
        <>
          <Choice label="Query" value={queryId} onChange={(v) => reset(() => setQueryId(v))} options={QUERIES.map((o) => ({ value: o.id, label: `${o.sql}  (${o.blurb})` }))} />
          {q.kind === 'jsonb' ? (
            <Segmented
              label="Operator class"
              value={opclass}
              onChange={(v) => reset(() => setOpclass(v))}
              options={[
                { value: 'jsonb_ops', label: 'jsonb_ops' },
                { value: 'jsonb_path_ops', label: 'jsonb_path_ops' },
              ]}
            />
          ) : null}
          <Slider label="Rows in the table" min={40} max={400} step={20} value={rows} onChange={(v) => reset(() => { setRows(v); setPendingRows((p) => Math.min(p, v)); })} format={fmtNum} />
          <Slider label="Newest rows still in the pending list" min={0} max={60} step={5} value={pendingRows} onChange={(v) => reset(() => setPendingRows(Math.min(v, rows)))} />
        </>
      }
      legend={
        <Legend
          items={[
            { label: 'TID decoded', color: 'var(--viz-1)' },
            { label: 'TID never decoded', color: 'var(--viz-ink-muted)' },
            { label: 'Posting-tree leaf page loaded', color: 'var(--viz-7)' },
            ...(scan.pending.rows > 0 ? [{ label: 'Pending-list rows', color: 'var(--viz-2)' }] : []),
            { label: 'TID in the bitmap', color: 'var(--viz-ink)' },
            ...(scan.recheck ? [{ label: 'Kept by heap recheck', color: 'var(--viz-good)' }, { label: 'Discarded by recheck', color: 'var(--viz-critical)' }] : []),
          ]}
        />
      }
      stats={
        <Stats
          items={[
            { label: 'Entries: required / additional', value: scan.unsupported ? '—' : `${requiredEntries.length} / ${additionalEntries.length}`, hint: 'startScanKey puts the rarest entries in the required set until triConsistent says the rest cannot match on their own.' },
            { label: 'consistent() calls', value: scan.unsupported ? '—' : fmtNum(prog.calls) },
            { label: 'TIDs decoded', value: scan.unsupported ? '—' : `${fmtNum(prog.totalDecoded)} of ${fmtNum(scan.totals.tids)}`, hint: 'Inline posting lists are decoded whole at scan start; posting-tree pages are decoded from the segment holding the target onward.' },
            { label: 'Leaf pages loaded', value: scan.unsupported ? '—' : `${fmtNum(prog.totalLeafReads)} of ${fmtNum(scan.totals.leafPages)}`, hint: 'Posting-tree leaf pages. A skip past the loaded page re-descends from the posting tree root. Lab leaves hold 16 TIDs; real 8 KB leaves hold thousands.' },
            { label: 'Pending entries read', value: fmtNum(scan.unsupported ? 0 : scan.pending.entries), hint: 'The pending list is not organised by key, so every scan reads every pending page and binary-searches each row’s sorted entries for the query keys.' },
            { label: scan.recheck ? 'Bitmap rows → kept by recheck' : 'Rows in bitmap', value: scan.unsupported ? '—' : !done ? '…' : scan.recheck ? `${fmtNum(scan.candidates.length)} → ${fmtNum(kept)}` : fmtNum(scan.candidates.length) },
          ]}
        />
      }
      note={<Note>{note}</Note>}
      table={
        <table className="viz-table">
          <thead>
            <tr>
              <th>Entry</th>
              <th>Role</th>
              <th>TIDs</th>
              <th>Storage</th>
              <th>Decoded</th>
              <th>Leaf pages loaded</th>
              <th>Descents</th>
            </tr>
          </thead>
          <tbody>
            {scan.entries.map((e) => (
              <tr key={e.id}>
                <td>
                  {e.label}
                  {e.label !== e.detail ? ` (${e.detail})` : ''}
                </td>
                <td>{e.required ? 'required' : 'additional'}</td>
                <td>{fmtNum(e.rows.length)}</td>
                <td>{e.tree ? `posting tree, ${e.pages} leaves, height ${e.height}` : `inline, ${e.bytes} B`}</td>
                <td>{fmtNum(e.decodes.reduce((a, d) => a + d.to - d.from, 0))}</td>
                <td>{e.tree ? `${e.pageReads.length} of ${e.pages}` : '—'}</td>
                <td>{e.pageReads.filter((p) => p.how === 'descend').length}</td>
              </tr>
            ))}
            <tr>
              <td>Pending list</td>
              <td>scanned first</td>
              <td>{fmtNum(scan.pending.rows)} rows</td>
              <td>{fmtNum(scan.pending.entries)} entries</td>
              <td>all</td>
              <td>—</td>
              <td>—</td>
            </tr>
          </tbody>
        </table>
      }
    >
      <div className="viz-controls">
        <Button onClick={() => { setPlaying(false); setStep(0); }} disabled={!!scan.unsupported}>
          Restart scan
        </Button>
        <Button primary onClick={() => { setPlaying(false); setStep(Math.min(last, cur + 1)); }} disabled={done || !!scan.unsupported}>
          Next consistent() call
        </Button>
        <Button onClick={() => { if (done) setStep(0); acc.current = 0; setPlaying((p) => !p); }} disabled={!!scan.unsupported}>
          {playing ? 'Pause' : 'Play'}
        </Button>
        <Button onClick={() => { setPlaying(false); setStep(1e9); }} disabled={done || !!scan.unsupported}>
          Show result
        </Button>
      </div>
      <p className="viz-sub" style={{ margin: '0.25rem 0 0.4rem' }}>
        <code>{q.sql}</code>
      </p>
      {scan.unsupported ? (
        <svg viewBox={`0 0 ${W} 60`} width={W} height={60} role="img" aria-label="jsonb_path_ops cannot answer the ? operator">
          <rect x={1} y={1} width={W - 2} height={58} rx={6} fill="var(--viz-surface)" stroke="var(--viz-border)" />
          <text x={W / 2} y={26} textAnchor="middle" fontSize={13} fill="var(--viz-ink)">
            jsonb_path_ops stores only hashes of path + value
          </text>
          <text x={W / 2} y={44} textAnchor="middle" fontSize={11} fill="var(--viz-ink-2)">
            there is no entry for “key team exists”, so this operator is not in the opclass
          </text>
        </svg>
      ) : (
        <svg viewBox={`0 0 ${W} ${H}`} width={W} height={H} role="img" aria-label={`GIN scan of ${q.sql}: ${prog.calls} consistent calls, ${prog.totalDecoded} of ${scan.totals.tids} TIDs decoded`}>
          <text x={LX} y={14} fontSize={10} fill="var(--viz-ink-2)">
            heap TID order → {tidLabel(0)}
          </text>
          <text x={RX} y={14} fontSize={10} textAnchor="end" fill="var(--viz-ink-2)">
            {tidLabel(rows - 1)}
          </text>
          {scan.pending.rows > 0 ? (
            <g>
              <rect x={x(mainRows) - (RX - LX) / rows / 2} y={20} width={RX - x(mainRows) + (RX - LX) / rows / 2} height={yRecheck + 26 - 20} fill="var(--viz-2)" fillOpacity={0.12} />
              <text x={RX - 4} y={30} fontSize={10} textAnchor="end" fill="var(--viz-ink)">
                pending list: {fmtNum(scan.pending.rows)} rows
              </text>
            </g>
          ) : null}
          {scan.entries.map((e, i) => {
            const y = Y0 + i * TH;
            const base = y + 28;
            const cursor = ev?.kind === 'consistent' ? ev.cursors[i] : null;
            const pagesLoaded = readMask[i];
            return (
              <g key={e.id}>
                <text x={6} y={y + 15} fontSize={13} fill="var(--viz-ink)">
                  {shortLabel(e).length > 24 ? `${shortLabel(e).slice(0, 23)}…` : shortLabel(e)}
                </text>
                <text x={6} y={y + 30} fontSize={11} fill="var(--viz-ink-2)">
                  {fmtNum(e.rows.length)} TIDs · {e.tree ? `${e.pages}-leaf tree` : e.rows.length ? `inline ${e.bytes} B` : 'no entry'}
                </text>
                <text x={6} y={y + 45} fontSize={11} fill="var(--viz-ink)" fontWeight={e.required ? 600 : 400}>
                  {e.required ? 'required' : 'additional'}
                  {cursor === null && ev?.kind === 'consistent' ? ' · finished' : ''}
                </text>
                <line x1={LX} x2={RX} y1={base} y2={base} stroke="var(--viz-grid)" />
                {e.tree
                  ? Array.from({ length: e.pages }, (_, p) => {
                      const first = e.rows[p * POSTING_LEAF_TIDS];
                      const lastRow = e.rows[Math.min(e.rows.length, (p + 1) * POSTING_LEAF_TIDS) - 1];
                      const loaded = pagesLoaded[p] === 1;
                      const now = currentReads[i].some((pr) => pr.page === p);
                      return (
                        <rect
                          key={p}
                          x={x(first) - 2}
                          y={base - 13}
                          width={Math.max(4, x(lastRow) - x(first) + 4)}
                          height={26}
                          rx={3}
                          fill={loaded ? 'var(--viz-7)' : 'none'}
                          fillOpacity={loaded ? 0.16 : 0}
                          stroke="var(--viz-7)"
                          strokeOpacity={loaded ? 0.9 : 0.45}
                          strokeWidth={now ? 2.5 : 1}
                          strokeDasharray={loaded ? undefined : '3 2'}
                        />
                      );
                    })
                  : null}
                {e.rows.map((r, j) => (
                  <line key={r} x1={x(r)} x2={x(r)} y1={base - 8} y2={base + 8} stroke={decodedMask[i][j] ? 'var(--viz-1)' : 'var(--viz-ink-muted)'} strokeOpacity={decodedMask[i][j] ? 1 : 0.45} strokeWidth={1.3} />
                ))}
                {ev?.kind === 'consistent' ? (
                  <circle cx={x(ev.row)} cy={base} r={4} fill={ev.res[i] ? 'var(--viz-ink)' : 'var(--viz-surface)'} stroke="var(--viz-ink)" strokeWidth={1.5} />
                ) : null}
                {cursor !== null && cursor !== undefined && cursor !== ev?.row ? (
                  <path d={`M ${x(cursor)} ${base + 10} l -4 7 h 8 z`} fill="var(--viz-ink)" />
                ) : null}
              </g>
            );
          })}
          {ev?.kind === 'consistent' ? <line x1={x(ev.row)} x2={x(ev.row)} y1={Y0 - 2} y2={yBitmap + 24} stroke="var(--viz-ink-2)" strokeDasharray="4 3" /> : null}

          <text x={6} y={yBitmap + 14} fontSize={13} fill="var(--viz-ink)">
            TID bitmap
          </text>
          <text x={6} y={yBitmap + 28} fontSize={11} fill="var(--viz-ink-2)">
            {fmtNum(scan.pending.matches.length + prog.matches.length)} rows so far
          </text>
          <line x1={LX} x2={RX} y1={yBitmap + 12} y2={yBitmap + 12} stroke="var(--viz-grid)" />
          {scan.pending.matches.map((r) => (
            <line key={`p${r}`} x1={x(r)} x2={x(r)} y1={yBitmap + 3} y2={yBitmap + 21} stroke="var(--viz-2)" strokeWidth={1.6} />
          ))}
          {[...bitmapMain].map((r) => (
            <line key={`m${r}`} x1={x(r)} x2={x(r)} y1={yBitmap + 3} y2={yBitmap + 21} stroke="var(--viz-ink)" strokeWidth={1.6} />
          ))}

          <text x={6} y={yRecheck + 14} fontSize={13} fill="var(--viz-ink)">
            Heap recheck
          </text>
          {scan.recheck ? (
            done ? (
              <>
                <text x={6} y={yRecheck + 28} fontSize={11} fill="var(--viz-ink-2)">
                  kept {fmtNum(kept)}, discarded {fmtNum(scan.discarded.length)}
                </text>
                <line x1={LX} x2={RX} y1={yRecheck + 12} y2={yRecheck + 12} stroke="var(--viz-grid)" />
                {scan.confirmed.map((r) => (
                  <line key={`k${r}`} x1={x(r)} x2={x(r)} y1={yRecheck + 3} y2={yRecheck + 21} stroke="var(--viz-good)" strokeWidth={1.6} />
                ))}
                {scan.discarded.map((r) => (
                  <path key={`d${r}`} d={`M ${x(r) - 3} ${yRecheck + 8} l 6 8 m 0 -8 l -6 8`} stroke="var(--viz-critical)" strokeWidth={1.5} />
                ))}
              </>
            ) : (
              <text x={LX} y={yRecheck + 16} fontSize={11} fill="var(--viz-ink-2)">
                after the bitmap is complete: fetch each row, re-run the operator
              </text>
            )
          ) : (
            <text x={LX} y={yRecheck + 16} fontSize={11} fill="var(--viz-ink-2)">
              {'none needed: array_ops sets recheck = false for @> and &&'}
            </text>
          )}
        </svg>
      )}
    </VizPanel>
  );
}
