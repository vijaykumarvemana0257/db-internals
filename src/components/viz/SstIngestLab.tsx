import { useMemo, useState } from 'react';
import { VizPanel, Segmented, Choice, Check, Slider, Button, Legend, Stats, Note, fmtBytes, fmtNum } from './Viz';

/**
 * External SST ingestion, following RocksDB's ExternalSstFileIngestionJob:
 * - Foreground writes are stopped for the whole job. If the file's range overlaps the memtable, the
 *   memtable is flushed first (still with writes stopped), or ingestion fails when allow_blocking_flush=false.
 * - Normal ingestion walks down from L0. A level whose keys overlap the file ends the walk; every level
 *   above it where the file fits (no file-range overlap; L0 always fits) is a candidate, and the deepest wins.
 * - The file gets sequence number 0 unless it overlaps existing data or a snapshot is open
 *   (snapshot_consistency), in which case it gets last sequence + 1.
 * - ingest_behind needs a column family created with cf_allow_ingest_behind (universal compaction),
 *   which reserves the last level. The file must fit there, and keeps sequence number 0.
 * Files here are dense key ranges, so "keys overlap" and "ranges overlap" coincide.
 */
export type Range = [number, number];
export type Mode = 'normal' | 'behind';

const LEVELED: Range[][] = [
  [[300, 420]],
  [[0, 140], [230, 480], [520, 760]],
  [[0, 90], [100, 180], [240, 500], [510, 640], [650, 800]],
  [[0, 120], [130, 260], [270, 390], [400, 560], [570, 740], [750, 880]],
  [[0, 100], [100, 250], [250, 400], [400, 560], [560, 720], [720, 900]],
];
const UNIVERSAL_BEHIND: Range[][] = [
  [[300, 420]],
  [[0, 140], [230, 480], [520, 760]],
  [[0, 260], [270, 560], [570, 880]],
  [[0, 300], [300, 600], [600, 900]],
  [[0, 180]], // reserved last level: only earlier ingest_behind files
];
export const MEMTABLE: Range = [440, 470];
export const LAST_SEQ = 18_204;
export const ROW_BYTES = 100;
/** Bytes per key unit in L1, for sizing the L0→L1 compaction an L0 landing owes (L1 ≈ 160 MB). */
const L1_BYTES_PER_UNIT = 250_000;

export const treeFor = (mode: Mode) => (mode === 'behind' ? UNIVERSAL_BEHIND : LEVELED);
const overlap = (a: Range, b: Range) => a[0] <= b[1] && b[0] <= a[1];

export type Step = { kind: 'writes' | 'memtable' | 'level' | 'seq' | 'edit' | 'fail'; level?: number; bad?: boolean; text: string };

export function ingest(file: Range, mode: Mode, allowBlockingFlush: boolean, snapshot: boolean) {
  const tree = treeFor(mode).map((l) => l.slice());
  const last = tree.length - 1;
  const steps: Step[] = [{ kind: 'writes', text: 'Stop foreground writes: the ingestion job becomes the only writer until it commits.' }];
  const memOverlap = overlap(file, MEMTABLE);
  let flushed = false;
  const fail = (text: string, level?: number) => {
    steps.push({ kind: 'fail', bad: true, level, text });
    return { ok: false as const, level: -1, seq: -1, steps, flushed, memOverlap, blockedAt: level ?? -1, tree };
  };

  if (memOverlap) {
    if (!allowBlockingFlush) return fail(`Memtable holds keys in [${MEMTABLE[0]}, ${MEMTABLE[1]}], inside the file's range, and allow_blocking_flush is off: fail with "External file requires flush".`);
    flushed = true;
    tree[0] = [...tree[0], MEMTABLE];
    steps.push({ kind: 'memtable', bad: true, text: `Memtable overlaps the file, so flush it to a new L0 file first — writes stay stopped while the flush runs.` });
  } else {
    steps.push({ kind: 'memtable', text: 'Memtable does not overlap the file: no flush.' });
  }

  if (mode === 'behind') {
    if (tree[last].some((r) => overlap(file, r))) return fail(`L${last} is reserved for ingested files, and an earlier one already covers part of [${file[0]}, ${file[1]}]: fail with "Can't ingest_behind file as it doesn't fit at the last level".`, last);
    steps.push({ kind: 'level', level: last, text: `ingest_behind skips the walk: the file fits in the reserved last level, L${last}.` });
    steps.push({ kind: 'seq', text: 'Sequence number 0 — older than every existing version, so any key already present keeps its current value.' });
    steps.push({ kind: 'edit', level: last, text: `Commit a VersionEdit adding the file at L${last} to the MANIFEST, then resume writes.` });
    return { ok: true as const, level: last, seq: 0, steps, flushed, memOverlap, blockedAt: -1, tree };
  }

  let target = 0;
  let blockedAt = -1;
  for (let lvl = 0; lvl <= last; lvl++) {
    if (tree[lvl].some((r) => overlap(file, r))) {
      blockedAt = lvl;
      steps.push({ kind: 'level', level: lvl, bad: true, text: lvl === 0 ? 'L0 has keys in this range: stop. The file can only go to L0.' : `L${lvl} has keys in this range: stop. The file must sit above them, so L${target} is the deepest choice.` });
      break;
    }
    target = lvl;
    steps.push({ kind: 'level', level: lvl, text: lvl === last ? `L${lvl}, the last level, has no keys in this range: the file can go all the way down.` : lvl === 0 ? 'L0 has no keys in this range (and L0 accepts any file): keep walking.' : `L${lvl} has no keys in this range, and the file fits between its files: keep walking.` });
  }
  const overlapsDb = blockedAt >= 0;
  const seq = overlapsDb || snapshot ? LAST_SEQ + 1 : 0;
  steps.push({
    kind: 'seq',
    text: overlapsDb
      ? `Existing keys overlap, so stamp sequence number ${fmtNum(seq)} (last + 1): the file's values win over older versions, and snapshots taken earlier do not see them.`
      : snapshot
        ? `Nothing overlaps, but a snapshot is open: stamp ${fmtNum(seq)} anyway, so the snapshot does not suddenly see the new keys.`
        : 'Nothing overlaps and no snapshot is open: the file keeps sequence number 0.',
  });
  steps.push({ kind: 'edit', level: target, text: `Commit a VersionEdit adding the file at L${target} to the MANIFEST, then resume writes.` });
  return { ok: true as const, level: target, seq, steps, flushed, memOverlap, blockedAt, tree };
}

/** Bytes the database writes for these rows' own bytes. Merges also rewrite overlapping data already in the next level. */
export function costs(rows: number, mode: Mode, r: ReturnType<typeof ingest>) {
  const d = rows * ROW_BYTES;
  const dataBottom = mode === 'behind' ? treeFor(mode).length - 2 : treeFor(mode).length - 1;
  const stream = { wal: d, flush: d, compaction: d * dataBottom };
  const moves = !r.ok || mode === 'behind' ? 0 : dataBottom - r.level;
  const ingestCost = { wal: 0, flush: 0, compaction: d * moves };
  return { d, stream, ingest: ingestCost, moves, dataBottom };
}

export function l0CompactionOwed(fileRange: Range, rows: number, mode: Mode) {
  const l1 = treeFor(mode)[1].filter((f) => overlap(fileRange, f));
  const l1Bytes = l1.reduce((s, f) => s + (f[1] - f[0]) * L1_BYTES_PER_UNIT, 0);
  return { l1Files: l1.length, bytes: rows * ROW_BYTES + l1Bytes, l1Bytes };
}

const PRESETS: Record<Mode, { value: string; label: string; range: Range }[]> = {
  normal: [
    { value: 'gap', label: 'Fits a gap in L1, blocked at L2', range: [145, 175] },
    { value: 'fresh', label: 'Brand-new keys past the end', range: [910, 990] },
    { value: 'memtable', label: 'Overlaps the memtable', range: [445, 460] },
    { value: 'l0', label: 'Overlaps an L0 file', range: [350, 380] },
    { value: 'wide', label: 'A whole table in one file', range: [0, 1000] },
  ],
  behind: [
    { value: 'backfill', label: 'Backfill under live data', range: [200, 420] },
    { value: 'clash', label: 'Collides with an earlier backfill', range: [150, 250] },
    { value: 'fresh', label: 'Brand-new keys past the end', range: [910, 990] },
  ],
};

export default function SstIngestLab() {
  const [mode, setMode] = useState<Mode>('normal');
  const [preset, setPreset] = useState('gap');
  const [lo, setLo] = useState(145);
  const [hi, setHi] = useState(175);
  const [blocking, setBlocking] = useState(true);
  const [snapshot, setSnapshot] = useState(false);
  const [rows, setRows] = useState(2_000_000);
  const [step, setStep] = useState(99);

  const file: Range = [Math.min(lo, hi), Math.max(lo, hi)];
  const r = useMemo(() => ingest(file, mode, blocking, snapshot), [file[0], file[1], mode, blocking, snapshot]);
  const c = costs(rows, mode, r);
  const owed = l0CompactionOwed(file, rows, mode);
  const shown = r.steps.slice(0, Math.min(step, r.steps.length));
  const done = shown.length === r.steps.length;
  const current = shown[shown.length - 1];
  const flushShown = r.flushed && shown.some((s) => s.kind === 'memtable');
  const placed = r.ok && done;

  const choosePreset = (m: Mode, v: string) => {
    const p = PRESETS[m].find((x) => x.value === v) ?? PRESETS[m][0];
    setPreset(p.value);
    setLo(p.range[0]);
    setHi(p.range[1]);
    setStep(99);
  };

  const W = 680;
  const LEFT = 92;
  const X = (k: number) => LEFT + (k / 1000) * (W - LEFT - 16);
  const tree = r.tree;
  const rowY = (lvl: number) => 48 + lvl * 34;
  const H = rowY(tree.length) + 34;
  const levelName = (lvl: number) => `L${lvl}${lvl === tree.length - 1 ? (mode === 'behind' ? ' reserved' : ' last') : ''}`;

  const maxBar = Math.max(c.stream.wal + c.stream.flush + c.stream.compaction, c.d + c.ingest.compaction, 1);
  const bar = (v: number) => (v / maxBar) * (W - 230);

  return (
    <VizPanel
      title="Ingesting an externally built SST file"
      subtitle="Choose a key range for a file built offline and drop it into a populated tree. The engine checks the memtable, walks down the levels, stamps a sequence number and commits one VersionEdit — no WAL records, no memtable inserts."
      controls={
        <>
          <Segmented
            label="Mode"
            value={mode}
            onChange={(m) => {
              setMode(m);
              choosePreset(m, PRESETS[m][0].value);
            }}
            options={[
              { value: 'normal', label: 'Normal ingestion (leveled)' },
              { value: 'behind', label: 'ingest_behind (reserved last level)' },
            ]}
          />
          <Choice label="File key range" value={preset} onChange={(v) => choosePreset(mode, v)} options={PRESETS[mode].map((p) => ({ value: p.value, label: p.label }))} />
          <Slider label="Smallest key" min={0} max={1000} step={5} value={lo} onChange={(v) => { setLo(v); setStep(99); }} />
          <Slider label="Largest key" min={0} max={1000} step={5} value={hi} onChange={(v) => { setHi(v); setStep(99); }} />
          <Check label="allow_blocking_flush" checked={blocking} onChange={(v) => { setBlocking(v); setStep(99); }} />
          {mode === 'normal' ? <Check label="A snapshot is open" checked={snapshot} onChange={(v) => { setSnapshot(v); setStep(99); }} /> : null}
          <Slider label="Rows in the file" min={100_000} max={10_000_000} step={100_000} value={rows} onChange={setRows} format={fmtNum} />
        </>
      }
      legend={
        <Legend
          items={[
            { label: 'Existing SST file', color: 'var(--viz-clean)' },
            { label: 'Memtable', color: 'var(--viz-dirty)' },
            { label: 'Ingested file', color: 'var(--viz-7)' },
            { label: 'Overlap that stops the walk', color: 'var(--viz-critical)' },
          ]}
        />
      }
      stats={
        <Stats
          items={[
            { label: 'Result', value: !done ? '…' : r.ok ? `lands in L${r.level}` : 'ingestion fails' },
            { label: 'Sequence number', value: !done || !r.ok ? '—' : fmtNum(r.seq) },
            { label: 'Memtable flushed', value: r.flushed ? 'yes' : 'no', hint: 'The flush runs while foreground writes are stopped.' },
            { label: 'L0 files after', value: fmtNum(tree[0].length + (r.ok && r.level === 0 ? 1 : 0)), hint: 'Every L0 file whose range covers a key is a file a point read may have to check.' },
            {
              label: 'DB writes: streamed vs ingested',
              value: r.ok ? `${fmtNum((c.stream.wal + c.stream.flush + c.stream.compaction) / c.d)}× vs ${fmtNum(c.ingest.compaction / c.d)}×` : '—',
              hint: 'Multiples of the rows’ size, counting only rewrites of these rows. The ingested file itself is written once by SstFileWriter, outside the database.',
            },
          ]}
        />
      }
      note={
        <Note>
          {!r.ok ? (
            <strong>Ingestion refused. </strong>
          ) : r.level === 0 ? (
            <>
              <strong>The file lands in L0.</strong> It is one more overlapping file there, and before L0 is healthy again the engine owes an L0→L1 compaction of about {fmtBytes(owed.bytes)} — the file plus {fmtBytes(owed.l1Bytes)} of overlapping L1 data in {owed.l1Files} file{owed.l1Files === 1 ? '' : 's'}.{' '}
            </>
          ) : mode === 'behind' ? (
            <>
              <strong>The file lands beneath everything, in L{r.level}.</strong> Keys it shares with live data are hidden by the newer versions; compaction never rewrites the reserved level.{' '}
            </>
          ) : (
            <>
              <strong>The file lands in L{r.level}.</strong> {c.moves === 0 ? 'It is already in the last level: no compaction will ever need to push these rows down.' : `Compaction will push these rows down ${c.moves} more level${c.moves === 1 ? '' : 's'}.`}{' '}
            </>
          )}
          Streaming the same {fmtNum(rows)} rows ({fmtBytes(c.d)}) through Put would write {fmtBytes(c.stream.wal)} of WAL, flush {fmtBytes(c.stream.flush)} and rewrite them in {c.dataBottom} compactions on the way to the {mode === 'behind' ? 'deepest data level' : 'last level'}{mode === 'behind' ? ' — and would overwrite the newer values instead of sitting beneath them' : ''}.
        </Note>
      }
      table={
        <table className="viz-table">
          <thead>
            <tr>
              <th>Path</th>
              <th>WAL</th>
              <th>Flush</th>
              <th>Compaction rewrites of these rows</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td>Streamed through Put</td>
              <td>{fmtBytes(c.stream.wal)}</td>
              <td>{fmtBytes(c.stream.flush)}</td>
              <td>{fmtBytes(c.stream.compaction)}</td>
            </tr>
            <tr>
              <td>Ingested {r.ok ? `into L${r.level}` : '(failed)'}</td>
              <td>0</td>
              <td>0</td>
              <td>{fmtBytes(c.ingest.compaction)}</td>
            </tr>
          </tbody>
        </table>
      }
    >
      <div className="viz-controls">
        <Button onClick={() => setStep(1)}>Ingest step by step</Button>
        <Button primary onClick={() => setStep((s) => Math.min(r.steps.length, s + 1))} disabled={done}>
          Next step
        </Button>
        <Button onClick={() => setStep(99)} disabled={done}>
          Show result
        </Button>
      </div>
      <svg viewBox={`0 0 ${W} ${H}`} width={W} height={H} role="img" aria-label={done ? (r.ok ? `Ingested file lands in level ${r.level}` : 'Ingestion fails') : 'Ingestion in progress'}>
        {current && (current.kind === 'level' || current.kind === 'edit' || current.kind === 'fail') && current.level !== undefined ? (
          <rect x={4} y={rowY(current.level) - 6} width={W - 8} height={32} rx={4} fill="none" stroke="var(--viz-ink-2)" strokeDasharray="4 3" />
        ) : null}
        {current && current.kind === 'memtable' ? <rect x={4} y={6} width={W - 8} height={28} rx={4} fill="none" stroke="var(--viz-ink-2)" strokeDasharray="4 3" /> : null}

        <text x={10} y={25} fontSize={12} fill="var(--viz-ink)">
          memtable
        </text>
        <rect
          x={X(MEMTABLE[0])}
          y={12}
          width={X(MEMTABLE[1]) - X(MEMTABLE[0])}
          height={16}
          rx={3}
          fill={flushShown ? 'var(--viz-surface)' : r.memOverlap ? 'var(--viz-critical)' : 'var(--viz-dirty)'}
          stroke={flushShown ? 'var(--viz-ink-muted)' : 'none'}
          strokeDasharray="3 2"
        />
        {flushShown ? (
          <text x={X(MEMTABLE[1]) + 6} y={25} fontSize={10} fill="var(--viz-ink-2)">
            flushed to L0
          </text>
        ) : null}

        {tree.map((files, lvl) => {
          const y = rowY(lvl);
          const walkReached = shown.some((s) => (s.kind === 'level' || s.kind === 'fail') && s.level === lvl);
          return (
            <g key={lvl}>
              <text x={10} y={y + 15} fontSize={12} fill="var(--viz-ink)">
                {levelName(lvl)}
              </text>
              <line x1={LEFT} x2={W - 16} y1={y + 10} y2={y + 10} stroke="var(--viz-ink-muted)" strokeOpacity={0.35} />
              {files.map((f, i) => {
                const isFlushed = lvl === 0 && r.flushed && f === MEMTABLE;
                if (isFlushed && !flushShown) return null;
                const blocks = walkReached && r.blockedAt === lvl && overlap(file, f);
                const clash = mode === 'behind' && !r.ok && lvl === tree.length - 1 && walkReached && overlap(file, f);
                return (
                  <rect
                    key={i}
                    x={X(f[0]) + 1}
                    y={y}
                    width={Math.max(3, X(f[1]) - X(f[0]) - 2)}
                    height={20}
                    rx={3}
                    fill={blocks || clash ? 'var(--viz-critical)' : 'var(--viz-clean)'}
                    opacity={0.9}
                  />
                );
              })}
              {placed && r.level === lvl ? (
                <rect x={X(file[0]) + 1} y={y - 4} width={Math.max(4, X(file[1]) - X(file[0]) - 2)} height={28} rx={3} fill="var(--viz-7)" fillOpacity={0.3} stroke="var(--viz-7)" strokeWidth={2.5} />
              ) : null}
            </g>
          );
        })}
        <line x1={X(file[0])} x2={X(file[0])} y1={8} y2={rowY(tree.length) - 4} stroke="var(--viz-7)" strokeDasharray="4 3" />
        <line x1={X(file[1])} x2={X(file[1])} y1={8} y2={rowY(tree.length) - 4} stroke="var(--viz-7)" strokeDasharray="4 3" />
        <text x={Math.min(W - 90, Math.max(LEFT + 70, (X(file[0]) + X(file[1])) / 2))} y={rowY(tree.length) + 12} textAnchor="middle" fontSize={11} fill="var(--viz-ink-2)">
          file range [{file[0]}, {file[1]}]
        </text>
      </svg>

      <ol style={{ margin: '0.5rem 0 0', paddingLeft: '1.2rem', fontSize: '0.8rem', display: 'grid', gap: 4 }}>
        {shown.map((s, i) => (
          <li key={i} style={{ color: 'var(--viz-ink)', borderLeft: s.bad ? '3px solid var(--viz-critical)' : '3px solid transparent', paddingLeft: 6 }}>
            {s.text}
          </li>
        ))}
      </ol>

      <svg viewBox={`0 0 ${W} 84`} width={W} height={84} role="img" aria-label="Bytes written by streaming versus ingesting the same rows" style={{ marginTop: 10 }}>
        {[
          { label: 'Streamed via Put', parts: [c.stream.wal, c.stream.flush, c.stream.compaction], y: 10 },
          { label: 'Ingested file', parts: [0, 0, c.ingest.compaction], y: 46, file: c.d },
        ].map((row) => {
          let x = 150;
          const colors = ['var(--viz-3)', 'var(--viz-4)', 'var(--viz-5)'];
          return (
            <g key={row.label}>
              <text x={10} y={row.y + 15} fontSize={12} fill="var(--viz-ink)">
                {row.label}
              </text>
              {row.file !== undefined ? (
                <rect x={x} y={row.y} width={Math.max(2, bar(row.file))} height={20} rx={3} fill="var(--viz-surface)" stroke="var(--viz-7)" strokeWidth={2} strokeDasharray="4 2" />
              ) : null}
              {(() => {
                if (row.file !== undefined) x += Math.max(2, bar(row.file)) + 2;
                return row.parts.map((p, i) => {
                  if (p <= 0) return null;
                  const w = Math.max(2, bar(p));
                  const el = <rect key={i} x={x} y={row.y} width={w - 2} height={20} rx={3} fill={colors[i]} />;
                  x += w;
                  return el;
                });
              })()}
              <text x={x + 6} y={row.y + 15} fontSize={11} fill="var(--viz-ink-2)">
                {fmtBytes(row.parts.reduce((s, p) => s + p, 0) + (row.file ?? 0))}
              </text>
            </g>
          );
        })}
      </svg>
      <Legend
        items={[
          { label: 'WAL', color: 'var(--viz-3)' },
          { label: 'Flush', color: 'var(--viz-4)' },
          { label: 'Compaction rewrites', color: 'var(--viz-5)' },
          { label: 'Building the file (SstFileWriter, outside the DB)', color: 'var(--viz-7)' },
        ]}
      />
    </VizPanel>
  );
}
