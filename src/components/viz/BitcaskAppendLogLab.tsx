import { useState } from 'react';
import {
  VizPanel,
  Choice,
  Segmented,
  Button,
  Slider,
  Legend,
  Stats,
  Note,
  TooltipHost,
  useTip,
  fmtBytes,
  fmtTime,
  fmtNum,
  makeRng,
  useSize,
} from './Viz';

/**
 * Bitcask: an append-only log plus an in-RAM hash directory.
 *
 * Every write is a record appended to the single active file; the keydir maps key ->
 * (file_id, value_pos, value_sz, tstamp) and is re-pointed at the newest offset. Older
 * records for the same key are not touched — they are simply unreachable, which is what
 * merge later reclaims. Merge rewrites the closed files keeping only the record the
 * keydir currently points at, and emits a hint file so a restart can rebuild the keydir
 * from key-sized entries instead of scanning every value byte.
 *
 * Byte model (Riak's bitcask): 14-byte data header = crc32(4) + tstamp(4) + ksz(2) +
 * valsz(4); 18-byte hint header = tstamp(4) + ksz(2) + total_sz(4) + value_pos(8).
 */

const HEADER = 14;
const HINT_HEADER = 18;
const TOMB_VALUE = 'bitcask_tombstone'; // 17 bytes
const ENTRY_FIXED = 20; // file_id 4 + value_sz 4 + value_pos 8 + tstamp 4
const ENTRY_OVERHEAD = 24; // hash bucket + allocator overhead (modelled)
const SEQ_BPS = 600e6; // 600 MB/s sequential read while rebuilding
const INSERT_NS = 250; // cost of one keydir insert during rebuild
const HASH_NS = 90; // one hash-table probe
const READ_NS = 95_000; // one random read of a value, NVMe, incl. syscall
const MAX_ACTIVE = 5; // records before the active file rolls (stands in for max_file_size)
const MAX_SEGS = 6;

const KEYS = ['u:101', 'u:102', 'u:103', 'c:88', 's:a1', 'f:x'] as const;
type Key = (typeof KEYS)[number];

type Rec = { id: number; key: Key; kind: 'put' | 'tomb'; sz: number; off: number; ts: number };
type Seg = { id: number; recs: Rec[]; bytes: number; closed: boolean; hint: boolean; merged: boolean };
type Entry = { file: number; off: number; sz: number; ts: number; tomb: boolean };

type LogRow = { n: number; op: string; detail: string; cost: number; live: number; disk: number; dead: number };

type S = {
  segs: Seg[];
  keydir: Record<string, Entry>;
  nextFile: number;
  nextRec: number;
  clock: number;
  script: number;
  down: boolean; // crashed: the keydir is gone, the files are not
  head: string;
  body: string;
  log: LogRow[];
};

function recSize(key: string, kind: 'put' | 'tomb', vsz: number) {
  return HEADER + key.length + (kind === 'tomb' ? TOMB_VALUE.length : vsz);
}
function hintBytes(seg: Seg) {
  return seg.recs.reduce((a, r) => a + HINT_HEADER + r.key.length, 0);
}
function entryRam(key: string) {
  return key.length + ENTRY_FIXED + ENTRY_OVERHEAD;
}

const INITIAL: S = {
  segs: [{ id: 1, recs: [], bytes: 0, closed: false, hint: false, merged: false }],
  keydir: {},
  nextFile: 2,
  nextRec: 1,
  clock: 1,
  script: 0,
  down: false,
  head: 'One open file, one empty hash table.',
  body:
    'Put a key: the record is appended to the tail of the active file and the keydir entry for that key ' +
    'is re-pointed at the new offset. Nothing is ever overwritten in place, so a write costs one ' +
    'sequential append and a read costs one hash probe plus one seek.',
  log: [],
};

/** Deterministic op script for the "Run 6 ops" button. */
const SCRIPT = (() => {
  const rng = makeRng(0xb17ca5);
  return Array.from({ length: 60 }, () => {
    const key = KEYS[Math.floor(rng() * KEYS.length) % KEYS.length];
    const r = rng();
    return { key, kind: r > 0.82 ? ('tomb' as const) : ('put' as const), vsz: 32 + Math.floor(rng() * 12) * 48 };
  });
})();

function tally(s: S) {
  let disk = 0;
  let dead = 0;
  for (const seg of s.segs) {
    disk += seg.bytes;
    for (const r of seg.recs) {
      const e = s.keydir[r.key];
      const livePtr = e && e.file === seg.id && e.off === r.off;
      if (!livePtr) dead += r.sz;
      else if (r.kind === 'tomb') dead += r.sz; // a live tombstone is still garbage, it just cannot be dropped yet
    }
  }
  const live = Object.values(s.keydir).filter((e) => !e.tomb).length;
  const ram = Object.entries(s.keydir).reduce((a, [k]) => a + entryRam(k), 0);
  return { disk, dead, live, ram, keys: Object.keys(s.keydir).length };
}

type Op = 'put' | 'del' | 'get' | 'scan' | 'roll' | 'merge' | 'crash' | 'restart' | 'traffic';
type Cfg = { key: Key; vsz: number; rebuild: 'hint' | 'scan' };

function append(s: S, key: Key, kind: 'put' | 'tomb', vsz: number): S {
  const segs = s.segs.map((x) => ({ ...x, recs: [...x.recs] }));
  const active = segs[segs.length - 1];
  const sz = recSize(key, kind, vsz);
  const rec: Rec = { id: s.nextRec, key, kind, sz, off: active.bytes, ts: s.clock };
  active.recs.push(rec);
  active.bytes += sz;
  const keydir = { ...s.keydir, [key]: { file: active.id, off: rec.off, sz, ts: s.clock, tomb: kind === 'tomb' } };
  let nextFile = s.nextFile;
  if (active.recs.length >= MAX_ACTIVE && segs.length < MAX_SEGS) {
    active.closed = true;
    segs.push({ id: nextFile++, recs: [], bytes: 0, closed: false, hint: false, merged: false });
  }
  return { ...s, segs, keydir, nextRec: s.nextRec + 1, clock: s.clock + 1, nextFile };
}

function rebuild(s: S, mode: 'hint' | 'scan') {
  // Replay every file oldest-first; the newest record for a key wins, a tombstone removes it.
  const keydir: Record<string, Entry> = {};
  let bytes = 0;
  let entries = 0;
  for (const seg of s.segs) {
    const useHint = mode === 'hint' && seg.hint;
    bytes += useHint ? hintBytes(seg) : seg.bytes;
    for (const r of seg.recs) {
      entries++;
      if (r.kind === 'tomb') delete keydir[r.key];
      else keydir[r.key] = { file: seg.id, off: r.off, sz: r.sz, ts: r.ts, tomb: false };
    }
  }
  const cost = (bytes / SEQ_BPS) * 1e9 + entries * INSERT_NS;
  return { keydir, bytes, entries, cost };
}

function apply(s: S, op: Op, cfg: Cfg): S {
  if (s.down && op !== 'restart') {
    return { ...s, head: 'The process is not running.', body: 'Press Restart to rebuild the keydir from the files on disk.' };
  }
  const t0 = tally(s);
  let next: S = s;
  let head = '';
  let body = '';
  let detail = '';
  let cost = 0;
  let label = op as string;

  switch (op) {
    case 'put':
    case 'traffic':
    case 'del': {
      const steps =
        op === 'traffic'
          ? Array.from({ length: 6 }, (_, i) => SCRIPT[(s.script + i) % SCRIPT.length])
          : [{ key: cfg.key, kind: op === 'del' ? ('tomb' as const) : ('put' as const), vsz: cfg.vsz }];
      const full = s.segs.length >= MAX_SEGS && s.segs[s.segs.length - 1].recs.length >= MAX_ACTIVE;
      if (full) {
        return {
          ...s,
          head: 'Out of room in this model.',
          body: 'Merge the closed files (or reset) before appending more — the real thing just keeps opening new files.',
        };
      }
      next = steps.reduce((acc, st) => append(acc, st.key, st.kind, st.vsz), s);
      if (op === 'traffic') next = { ...next, script: (s.script + 6) % SCRIPT.length };
      const wrote = steps.reduce((a, st) => a + recSize(st.key, st.kind, st.vsz), 0);
      cost = (wrote / SEQ_BPS) * 1e9;
      detail = steps.map((st) => `${st.kind === 'tomb' ? 'del ' : 'put '}${st.key}`).join(', ');
      label = op === 'del' ? 'delete' : op === 'traffic' ? '6 ops' : 'put';
      if (op === 'del') {
        head = `delete ${cfg.key} appended a tombstone — it did not erase anything.`;
        body =
          `A delete is an ordinary record whose value is the sentinel ${TOMB_VALUE} (later bitcask versions ` +
          `write a versioned tombstone carrying the file id and offset of the record it kills). The keydir entry ` +
          `now says "deleted", so a get returns not_found, and every older record for ${cfg.key} is unreachable ` +
          `garbage that only a merge will reclaim.`;
      } else if (op === 'traffic') {
        head = 'Six writes appended at the tail.';
        body =
          'Watch the arrows: each put re-points one keydir entry at the newest offset and instantly orphans the ' +
          'record it used to point at. The write cost never changes with the size of the dataset — it is one ' +
          'sequential append, which is the entire reason this design is fast.';
      } else {
        head = `put ${cfg.key} appended ${fmtBytes(recSize(cfg.key, 'put', cfg.vsz))} at the tail.`;
        body =
          `The record framing is crc32(4) + tstamp(4) + ksz(2) + valsz(4) + key + value = ` +
          `${HEADER} + ${cfg.key.length} + ${cfg.vsz} bytes. The keydir entry for ${cfg.key} was overwritten in ` +
          `place — one pointer move in RAM — and the previous record for that key is now dead weight on disk.`;
      }
      break;
    }

    case 'get': {
      const e = s.keydir[cfg.key];
      cost = HASH_NS + (e && !e.tomb ? READ_NS : 0);
      detail = `get ${cfg.key}`;
      label = 'get';
      if (!e) {
        head = `get ${cfg.key} → not_found, with zero disk I/O.`;
        body =
          'The key is not in the keydir, and the keydir is authoritative: every live key on disk has an entry. ' +
          'A miss costs one hash probe and never touches a file.';
      } else if (e.tomb) {
        head = `get ${cfg.key} → not_found, from the keydir alone.`;
        body = 'The entry is a tombstone. No read is issued; the record on disk is never consulted.';
      } else {
        head = `get ${cfg.key} → one hash probe, then one read at file ${e.file}, offset ${e.off}.`;
        body =
          `The entry gives (file_id ${e.file}, value_pos ${e.off}, value_sz ${e.sz}), so the read is a single ` +
          `pread of exactly the right bytes — no index descent, no tree, and exactly one disk access no matter ` +
          `how large the dataset has grown. That flat, predictable read path is Bitcask's other selling point.`;
      }
      break;
    }

    case 'scan': {
      const keys = Object.keys(s.keydir);
      const matches = keys.filter((k) => k.startsWith('u:') && !s.keydir[k].tomb);
      cost = keys.length * HASH_NS + matches.length * READ_NS;
      detail = `scan u:* (${matches.length}/${keys.length})`;
      label = 'range scan';
      head = `Range scan u:* had to look at all ${keys.length} keydir entries to find ${matches.length}.`;
      body =
        'A hash table has no order. There is no "next key after u:101" — the only way to answer a range query ' +
        'is to enumerate the entire keydir, filter in the application, and then issue one random read per hit. ' +
        'Cost is O(total keys), not O(matches). This limitation, not RAM, is what sorted runs and SSTables exist to fix.';
      break;
    }

    case 'roll': {
      const segs = s.segs.map((x) => ({ ...x }));
      const active = segs[segs.length - 1];
      if (active.recs.length === 0) {
        return { ...s, head: 'The active file is empty.', body: 'Nothing to close — append something first.' };
      }
      if (segs.length >= MAX_SEGS) {
        return { ...s, head: 'Too many files in this model.', body: 'Merge first.' };
      }
      active.closed = true;
      segs.push({ id: s.nextFile, recs: [], bytes: 0, closed: false, hint: false, merged: false });
      next = { ...s, segs, nextFile: s.nextFile + 1 };
      label = 'roll';
      detail = `close file ${active.id}`;
      head = `File ${active.id} is closed and immutable; file ${s.nextFile} is now the write target.`;
      body =
        'Exactly one file is ever open for writing. When it passes bitcask.max_file_size it is closed, and from ' +
        'that moment it is read-only — which is what makes it safe to merge it in the background while readers ' +
        'and writers carry on untouched.';
      break;
    }

    case 'merge': {
      const closed = s.segs.filter((x) => x.closed);
      const active = s.segs[s.segs.length - 1];
      if (closed.length === 0) {
        return {
          ...s,
          head: 'Nothing to merge.',
          body: 'Merge only ever touches closed files. Roll the active file (or keep writing until it rolls) first.',
        };
      }
      const before = closed.reduce((a, x) => a + x.bytes, 0);
      // Keep only the record the keydir currently points at, and only if it lives in a closed file.
      const survivors = Object.entries(s.keydir)
        .filter(([, e]) => closed.some((c) => c.id === e.file))
        .sort((a, b) => a[1].ts - b[1].ts);
      const dropped = Object.entries(s.keydir).filter(([, e]) => e.tomb && closed.some((c) => c.id === e.file)).length;
      const merged: Seg = { id: s.nextFile, recs: [], bytes: 0, closed: true, hint: true, merged: true };
      const keydir = { ...s.keydir };
      let rid = s.nextRec;
      for (const [key, e] of survivors) {
        if (e.tomb) {
          // Every file that could hold an older value for this key is in the merge set, so the
          // tombstone has nothing left to suppress and can finally be dropped.
          delete keydir[key];
          continue;
        }
        const rec: Rec = { id: rid++, key: key as Key, kind: 'put', sz: e.sz, off: merged.bytes, ts: e.ts };
        merged.recs.push(rec);
        merged.bytes += e.sz;
        keydir[key] = { ...e, file: merged.id, off: rec.off };
      }
      let nf = s.nextFile + 1;
      const segs: Seg[] = [merged];
      if (active.closed) segs.push({ id: nf++, recs: [], bytes: 0, closed: false, hint: false, merged: false });
      else segs.push(active);
      next = { ...s, segs, keydir, nextFile: nf, nextRec: rid };
      cost = ((before + merged.bytes) / SEQ_BPS) * 1e9;
      label = 'merge';
      detail = `${closed.length} files → 1 + hint`;
      head = `Merged ${closed.length} closed files: ${fmtBytes(before)} → ${fmtBytes(merged.bytes)}, plus a ${fmtBytes(
        hintBytes(merged),
      )} hint file.`;
      body =
        `Merge reads the closed files, writes out only the record each live key currently points at, and ` +
        `re-points the keydir at the new offsets — so it is a read-write pass over the data, not an in-place edit. ` +
        (dropped > 0
          ? `${dropped} tombstone${dropped === 1 ? '' : 's'} disappeared entirely: because every file that could ` +
            `hold an older value for those keys was in the merge set, there is nothing left for them to suppress. `
          : 'Tombstones can only be dropped when every file that might still hold an older value is in the merge set. ') +
        `The hint file it emits carries tstamp + ksz + total_sz + value_pos + key for each record — no values — ` +
        `which is the whole point of the next button.`;
      break;
    }

    case 'crash': {
      next = { ...s, keydir: {}, down: true };
      label = 'crash';
      detail = 'kill -9';
      head = 'Process gone. Every byte on disk survived; the entire index did not.';
      body =
        'The keydir was never persisted — it is a pure in-memory derivative of the files, so every record has gone ' +
        'grey: not lost, just unreachable until something rebuilds the index. A partially written trailing record is ' +
        'the only on-disk damage possible, and the crc32 in each header catches it, so startup truncates the tail and ' +
        'carries on. The cost of that simplicity is paid entirely at startup.';
      break;
    }

    case 'restart': {
      const r = rebuild(s, cfg.rebuild);
      const scanOnly = rebuild(s, 'scan');
      const segs = s.segs.map((x) => ({ ...x }));
      let nf = s.nextFile;
      if (segs.length < MAX_SEGS) {
        segs.forEach((x) => {
          x.closed = true;
        });
        segs.push({ id: nf++, recs: [], bytes: 0, closed: false, hint: false, merged: false });
      }
      next = { ...s, segs, keydir: r.keydir, down: false, nextFile: nf };
      cost = r.cost;
      label = cfg.rebuild === 'hint' ? 'restart (hints)' : 'restart (full scan)';
      detail = `read ${fmtBytes(r.bytes)}, ${r.entries} entries`;
      const speedup = r.bytes > 0 ? scanOnly.cost / r.cost : 1;
      head = `Keydir rebuilt: ${r.entries} records replayed, ${fmtBytes(r.bytes)} read, ${fmtTime(r.cost)}.`;
      body =
        (cfg.rebuild === 'hint'
          ? `Files with a hint file were rebuilt from key-sized entries; files without one — the active file, and ` +
            `any closed file not yet merged — had to be scanned in full, values and all. That is ${speedup.toFixed(1)}× ` +
            `faster than scanning everything here, and the ratio in production is the ratio of value size to key size: ` +
            `1 KB values and 20-byte keys make it roughly 30×. `
          : `Hints ignored: every byte of every file was read just to recover keys and offsets, and the value bytes ` +
            `were thrown away as soon as they were parsed. `) +
        `Startup time scales with bytes on disk, which is why a multi-hundred-gigabyte Bitcask node takes minutes to ` +
        `come back and why merge is an availability concern, not just a space one. A new active file was opened; the ` +
        `file that was being written before the crash is now read-only.`;
      break;
    }
  }

  const t = tally(next);
  return {
    ...next,
    head,
    body,
    log: [
      ...s.log,
      { n: s.log.length + 1, op: label, detail, cost, live: t.live, disk: t.disk, dead: t.dead },
    ].slice(-14),
  };
}

/* -------------------------------------------------------------- the drawing */

export default function BitcaskAppendLogLab() {
  const [key, setKey] = useState<Key>('u:101');
  const [vsz, setVsz] = useState(256);
  const [rebuildMode, setRebuildMode] = useState<'hint' | 'scan'>('hint');
  const [exp, setExp] = useState(7);
  const [s, setS] = useState<S>(INITIAL);
  const [ref, width] = useSize(760);
  const tip = useTip();

  const cfg: Cfg = { key, vsz, rebuild: rebuildMode };
  const run = (op: Op) => setS((cur) => apply(cur, op, cfg));
  const t = tally(s);

  const labelW = 152;
  const svgW = Math.max(width, labelW + KEYS.length * 104 + 24);
  const availW = svgW - labelW - 16;
  const chipTop = 24;
  const chipH = 26;
  const lanesTop = 74;
  const laneH = 54;
  const laneGap = 10;
  const height = lanesTop + s.segs.length * (laneH + laneGap) + 8;
  const laneY = (i: number) => lanesTop + i * (laneH + laneGap);

  const maxBytes = Math.max(1, ...s.segs.map((x) => x.bytes));
  const bScale = availW / maxBytes;

  type Box = { rec: Rec; x: number; w: number; live: boolean; seg: Seg; y: number };
  const boxes: Box[] = [];
  s.segs.forEach((seg, i) => {
    const gap = 4;
    const raw = seg.recs.map((r) => Math.max(26, r.sz * bScale));
    const total = raw.reduce((a, b) => a + b, 0) + gap * Math.max(0, raw.length - 1);
    const shrink = total > availW ? availW / total : 1;
    let x = labelW + 8;
    seg.recs.forEach((r, k) => {
      const w = raw[k] * shrink;
      const e = s.keydir[r.key];
      boxes.push({ rec: r, x, w, live: !!e && e.file === seg.id && e.off === r.off, seg, y: laneY(i) + 14 });
      x += w + gap * shrink;
    });
  });

  const chipX = (k: number) => labelW + 8 + k * ((availW - 8) / KEYS.length);
  const chipW = Math.min(96, (availW - 8) / KEYS.length - 8);

  const projected = Math.pow(10, exp);
  const avgEntry = entryRam('u:101');

  return (
    <VizPanel
      title="Bitcask: an append-only log under an in-RAM hash directory"
      subtitle="Every write appends at the tail and re-points one keydir entry. Merge rewrites the closed files and emits hint files; a crash throws the whole index away and startup has to rebuild it from what is on disk."
      controls={
        <>
          <Choice label="Key" value={key} onChange={setKey} options={KEYS.map((k) => ({ value: k, label: k }))} />
          <Slider label="Value size" min={16} max={2048} step={16} value={vsz} onChange={setVsz} format={fmtBytes} />
          <Button onClick={() => run('put')} primary>
            put
          </Button>
          <Button onClick={() => run('del')} title="Append a tombstone record">
            delete
          </Button>
          <Button onClick={() => run('get')}>get</Button>
          <Button onClick={() => run('scan')} title="Enumerate every keydir entry with the prefix u:">
            scan u:*
          </Button>
          <Button onClick={() => run('traffic')} title="Six deterministic puts and deletes">
            Run 6 ops
          </Button>
          <Button onClick={() => run('roll')} title="Close the active file and open a new one">
            Roll file
          </Button>
          <Button onClick={() => run('merge')} title="Rewrite the closed files, keeping only live records, and emit hint files">
            Merge
          </Button>
          <Segmented
            label="Restart reads"
            value={rebuildMode}
            onChange={setRebuildMode}
            options={[
              { value: 'hint', label: 'hint files', title: 'Key-sized entries: no value bytes are read' },
              { value: 'scan', label: 'full scan', title: 'Every record header and value, just to recover offsets' },
            ]}
          />
          <Button onClick={() => run('crash')} disabled={s.down}>
            Crash
          </Button>
          <Button onClick={() => run('restart')} disabled={!s.down}>
            Restart
          </Button>
          <Slider
            label="Projected keyspace"
            min={5}
            max={9}
            step={1}
            value={exp}
            onChange={setExp}
            format={(n) => fmtNum(Math.pow(10, n))}
          />
          <Button onClick={() => setS(INITIAL)}>Reset</Button>
        </>
      }
      legend={
        <Legend
          items={[
            { label: 'live record — the keydir points here', color: 'var(--viz-1)' },
            { label: 'tombstone (delete marker)', color: 'var(--viz-2)' },
            { label: 'dead — superseded, reclaimed only by merge', color: 'var(--viz-stale)' },
            { label: 'hint file (keys + offsets, no values)', color: 'var(--viz-7)' },
          ]}
        />
      }
      stats={
        <Stats
          items={[
            { label: 'Live keys', value: fmtNum(t.live), hint: 'Keydir entries that resolve to a value' },
            { label: 'Bytes on disk', value: fmtBytes(t.disk) },
            {
              label: 'Dead bytes',
              value: `${fmtBytes(t.dead)} (${t.disk ? Math.round((t.dead / t.disk) * 100) : 0}%)`,
              hint: 'Superseded records and tombstones — space a merge would reclaim',
            },
            { label: 'Keydir RAM', value: fmtBytes(t.ram), hint: 'key + file_id 4 + value_sz 4 + value_pos 8 + tstamp 4 + ~24 B of table overhead' },
            {
              label: `Keydir at ${fmtNum(projected)} keys`,
              value: fmtBytes(projected * avgEntry),
              hint: 'Every key must fit in RAM on every node, all the time — this is the hard ceiling',
            },
            { label: 'Last op', value: s.log.length ? fmtTime(s.log[s.log.length - 1].cost) : '—' },
          ]}
        />
      }
      note={
        <Note>
          <strong>{s.head}</strong> {s.body}
        </Note>
      }
      table={
        <>
          <table className="viz-table">
            <thead>
              <tr>
                <th>File</th>
                <th>Offset</th>
                <th>Size</th>
                <th>Key</th>
                <th>Record</th>
                <th>State</th>
              </tr>
            </thead>
            <tbody>
              {boxes.length === 0 ? (
                <tr>
                  <td colSpan={6}>No records yet.</td>
                </tr>
              ) : (
                boxes.map((b) => (
                  <tr key={`${b.seg.id}-${b.rec.off}-${b.rec.id}`}>
                    <td>{String(b.seg.id).padStart(6, '0')}</td>
                    <td>{b.rec.off}</td>
                    <td>{b.rec.sz}</td>
                    <td>{b.rec.key}</td>
                    <td>{b.rec.kind === 'tomb' ? 'tombstone' : 'value'}</td>
                    <td>{b.live ? (b.rec.kind === 'tomb' ? 'live tombstone' : 'live') : 'dead'}</td>
                  </tr>
                ))
              )}
            </tbody>
          </table>
          <table className="viz-table">
            <thead>
              <tr>
                <th>Keydir key</th>
                <th>file_id</th>
                <th>value_pos</th>
                <th>value_sz</th>
                <th>tstamp</th>
                <th>RAM</th>
              </tr>
            </thead>
            <tbody>
              {Object.keys(s.keydir).length === 0 ? (
                <tr>
                  <td colSpan={6}>{s.down ? 'Empty — the process is down.' : 'Empty.'}</td>
                </tr>
              ) : (
                Object.entries(s.keydir).map(([k, e]) => (
                  <tr key={k}>
                    <td>{k}</td>
                    <td>{e.file}</td>
                    <td>{e.off}</td>
                    <td>{e.tomb ? '— (tombstone)' : e.sz}</td>
                    <td>{e.ts}</td>
                    <td>{entryRam(k)} B</td>
                  </tr>
                ))
              )}
            </tbody>
          </table>
          <table className="viz-table">
            <thead>
              <tr>
                <th>File</th>
                <th>Bytes</th>
                <th>Records</th>
                <th>Hint file</th>
                <th>Restart read, hints</th>
                <th>Restart read, full scan</th>
              </tr>
            </thead>
            <tbody>
              {s.segs.map((seg) => (
                <tr key={seg.id}>
                  <td>
                    {String(seg.id).padStart(6, '0')}
                    {seg.closed ? '' : ' (active)'}
                  </td>
                  <td>{seg.bytes}</td>
                  <td>{seg.recs.length}</td>
                  <td>{seg.hint ? `${hintBytes(seg)} B` : 'none'}</td>
                  <td>{seg.hint ? fmtBytes(hintBytes(seg)) : fmtBytes(seg.bytes)}</td>
                  <td>{fmtBytes(seg.bytes)}</td>
                </tr>
              ))}
            </tbody>
          </table>
          {s.log.length > 0 ? (
            <table className="viz-table">
              <thead>
                <tr>
                  <th>#</th>
                  <th>Op</th>
                  <th>Detail</th>
                  <th>Modelled cost</th>
                  <th>Live keys</th>
                  <th>On disk</th>
                  <th>Dead</th>
                </tr>
              </thead>
              <tbody>
                {s.log.map((r) => (
                  <tr key={r.n}>
                    <td>{r.n}</td>
                    <td>{r.op}</td>
                    <td>{r.detail}</td>
                    <td>{r.cost > 0 ? fmtTime(r.cost) : '—'}</td>
                    <td>{r.live}</td>
                    <td>{fmtBytes(r.disk)}</td>
                    <td>{fmtBytes(r.dead)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          ) : null}
        </>
      }
    >
      <div ref={ref}>
        <TooltipHost>
          <svg width={svgW} height={height} role="img" aria-label="Append-only segment files with an in-memory keydir pointing at the newest record for each key">
            <defs>
              <marker id="bc-arrow" viewBox="0 0 8 8" refX="6" refY="4" markerWidth="6" markerHeight="6" orient="auto">
                <path d="M0,0 L8,4 L0,8 z" fill="var(--viz-1)" />
              </marker>
            </defs>

            <text x={0} y={16} fill="var(--viz-ink)" fontWeight={600}>
              keydir (RAM)
            </text>
            <text x={0} y={chipTop + 18} fill="var(--viz-ink-muted)">
              {s.down ? 'gone — process down' : `${t.keys} entries · ${fmtBytes(t.ram)}`}
            </text>

            {KEYS.map((k, i) => {
              const e = s.keydir[k];
              const x = chipX(i);
              return (
                <g
                  key={k}
                  {...tip(
                    e ? (
                      <>
                        <strong>{k}</strong>
                        <br />
                        file_id {e.file} · value_pos {e.off} · value_sz {e.tomb ? 'tombstone' : e.sz} · tstamp {e.ts}
                        <br />
                        {entryRam(k)} bytes of RAM, held for as long as the key exists.
                      </>
                    ) : (
                      <>
                        <strong>{k}</strong>
                        <br />
                        No entry: the key has never been written, or a merge dropped its tombstone.
                      </>
                    ),
                  )}
                  style={{ cursor: 'help' }}
                >
                  <rect
                    x={x}
                    y={chipTop}
                    width={chipW}
                    height={chipH}
                    rx={5}
                    fill={e ? (e.tomb ? 'var(--viz-2)' : 'var(--viz-1)') : 'var(--viz-plane)'}
                    stroke={e ? 'var(--viz-surface)' : 'var(--viz-border)'}
                    strokeWidth={e ? 2 : 1}
                    strokeDasharray={e ? undefined : '3 2'}
                    opacity={e ? 1 : 0.7}
                  />
                  <text
                    x={x + chipW / 2}
                    y={chipTop + 17}
                    textAnchor="middle"
                    fill={e ? 'var(--viz-surface)' : 'var(--viz-ink-muted)'}
                    fontWeight={600}
                  >
                    {k}
                  </text>
                  {e ? (
                    <text x={x + chipW / 2} y={chipTop + chipH + 12} textAnchor="middle" fill="var(--viz-ink-2)">
                      {e.tomb ? 'tombstone' : `f${e.file}@${e.off}`}
                    </text>
                  ) : null}
                </g>
              );
            })}

            {s.segs.map((seg, i) => (
              <g key={seg.id}>
                <rect
                  x={labelW}
                  y={laneY(i)}
                  width={svgW - labelW - 8}
                  height={laneH}
                  rx={8}
                  fill="var(--viz-plane)"
                  stroke="var(--viz-border)"
                />
                <text x={0} y={laneY(i) + 20} fill="var(--viz-ink)" fontWeight={600}>
                  {String(seg.id).padStart(6, '0')}.data
                </text>
                <text x={0} y={laneY(i) + 36} fill="var(--viz-ink-muted)">
                  {seg.closed ? (seg.merged ? 'merged, read-only' : 'closed, read-only') : 'active — append here'}
                </text>
                {seg.hint ? (
                  <g {...tip(<>Hint file: tstamp + ksz + total_sz + value_pos + key for every record, {hintBytes(seg)} bytes — no values.</>)}>
                    <rect x={0} y={laneY(i) + 40} width={46} height={13} rx={3} fill="var(--viz-7)" />
                    <text x={52} y={laneY(i) + 50} fill="var(--viz-ink-2)">
                      hint {hintBytes(seg)} B
                    </text>
                  </g>
                ) : null}
                {seg.recs.length === 0 ? (
                  <text x={labelW + 12} y={laneY(i) + 32} fill="var(--viz-ink-muted)">
                    empty
                  </text>
                ) : null}
              </g>
            ))}

            {boxes.map((b) => {
              const fill = !b.live ? 'var(--viz-stale)' : b.rec.kind === 'tomb' ? 'var(--viz-2)' : 'var(--viz-1)';
              return (
                <g
                  key={`b${b.seg.id}-${b.rec.off}-${b.rec.id}`}
                  {...tip(
                    <>
                      <strong>
                        {b.rec.key} · {b.rec.kind === 'tomb' ? 'tombstone' : 'value'}
                      </strong>
                      <br />
                      file {b.seg.id}, offset {b.rec.off}, {b.rec.sz} bytes (header {HEADER} + key {b.rec.key.length} +
                      value {b.rec.sz - HEADER - b.rec.key.length})
                      <br />
                      {b.live
                        ? b.rec.kind === 'tomb'
                          ? 'The keydir points here, so the key reads as deleted. Still garbage — it just cannot be dropped until every older file is merged with it.'
                          : 'The keydir points here. This is the current value.'
                        : 'Superseded: a newer record for this key exists further along the log. Unreachable, and still occupying disk until a merge.'}
                    </>,
                  )}
                  style={{ cursor: 'help' }}
                >
                  <rect
                    x={b.x}
                    y={b.y}
                    width={b.w}
                    height={26}
                    rx={4}
                    fill={fill}
                    opacity={b.live ? 1 : 0.55}
                    stroke="var(--viz-surface)"
                    strokeWidth={1.5}
                  />
                  {b.w >= 52 ? (
                    <text x={b.x + b.w / 2} y={b.y + 17} textAnchor="middle" fill="var(--viz-surface)" fontWeight={600}>
                      {b.rec.kind === 'tomb' ? `${b.rec.key} ✕` : b.rec.key}
                    </text>
                  ) : null}
                  <text x={b.x + 1} y={b.y + 38} fill="var(--viz-ink-muted)">
                    {b.rec.off}
                  </text>
                </g>
              );
            })}

            {KEYS.map((k, i) => {
              const e = s.keydir[k];
              if (!e) return null;
              const target = boxes.find((b) => b.seg.id === e.file && b.rec.off === e.off);
              if (!target) return null;
              const x1 = chipX(i) + chipW / 2;
              const y1 = chipTop + chipH + 16;
              const x2 = target.x + target.w / 2;
              const y2 = target.y - 3;
              return (
                <path
                  key={`a${k}`}
                  d={`M ${x1} ${y1} C ${x1} ${y1 + 22}, ${x2} ${y2 - 22}, ${x2} ${y2}`}
                  fill="none"
                  stroke="var(--viz-1)"
                  strokeWidth={1.4}
                  opacity={0.85}
                  markerEnd="url(#bc-arrow)"
                />
              );
            })}
          </svg>
        </TooltipHost>
      </div>
    </VizPanel>
  );
}
