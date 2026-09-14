import { useMemo, useState } from 'react';
import { VizPanel, Choice, Button, Legend, Stats, Note } from './Viz';

/**
 * RocksDB-style metadata and recovery. Every step below is an event that either creates a file,
 * appends a record to a WAL, or appends a synced VersionEdit to a MANIFEST. A crash freezes the
 * disk at some event; recovery is computed from what is on disk, exactly as the engine would.
 */
type Ev =
  | { kind: 'wal-open'; wal: number; text: string }
  | { kind: 'put'; wal: number; key: string; val: string; text: string }
  | { kind: 'sst'; file: number; keys: Record<string, string>; text: string }
  | { kind: 'edit'; manifest: number; add?: { file: number; level: number }[]; del?: number[]; logNumber?: number; text: string }
  | { kind: 'manifest-new'; manifest: number; text: string }
  | { kind: 'current'; manifest: number; text: string };

const EVENTS: Ev[] = [
  { kind: 'manifest-new', manifest: 5, text: 'MANIFEST-000005 exists; CURRENT names it. The live set is empty; log_number is 7.' },
  { kind: 'wal-open', wal: 7, text: 'WAL 000007 is the active log.' },
  { kind: 'put', wal: 7, key: 'a', val: '1', text: 'Put(a, 1) appended to WAL 7 and inserted into the memtable.' },
  { kind: 'put', wal: 7, key: 'b', val: '2', text: 'Put(b, 2) appended to WAL 7.' },
  { kind: 'wal-open', wal: 8, text: 'The memtable is full: it becomes immutable and a new WAL 000008 opens for new writes.' },
  { kind: 'sst', file: 9, keys: { a: '1', b: '2' }, text: 'Flush writes SST 000009 containing a and b, and syncs it.' },
  { kind: 'edit', manifest: 5, add: [{ file: 9, level: 0 }], logNumber: 8, text: 'VersionEdit appended to MANIFEST-5 and synced: add file 9 at L0, log_number = 8 (WAL 7 is now fully persisted in SSTs).' },
  { kind: 'put', wal: 8, key: 'c', val: '3', text: 'Put(c, 3) appended to WAL 8.' },
  { kind: 'sst', file: 10, keys: { a: '1', b: '2' }, text: 'A compaction rewrites file 9 into SST 000010 at L1, and syncs it.' },
  { kind: 'edit', manifest: 5, add: [{ file: 10, level: 1 }], del: [9], text: 'VersionEdit synced: add file 10 at L1, delete file 9.' },
  { kind: 'put', wal: 8, key: 'd', val: '4', text: 'Put(d, 4) appended to WAL 8.' },
  { kind: 'manifest-new', manifest: 11, text: 'MANIFEST-5 is large: MANIFEST-000011 is written, beginning with a snapshot of the whole current Version, and synced.' },
  { kind: 'current', manifest: 11, text: 'CURRENT is atomically replaced (write a temporary file, sync, rename) to name MANIFEST-11.' },
];

type Crash = { id: string; label: string; after: number; torn?: boolean; partialSst?: number };
export const CRASHES: Crash[] = [
  { id: 'mid-flush', label: 'During the flush (SST 9 half written)', after: 4, partialSst: 9 },
  { id: 'sst-no-edit', label: 'SST 9 written, VersionEdit not logged', after: 5 },
  { id: 'after-flush-edit', label: 'Just after the flush’s VersionEdit', after: 6 },
  { id: 'mid-compaction', label: 'Compaction output written, edit not logged', after: 8 },
  { id: 'torn-wal', label: 'Torn WAL tail: Put(d) only half written', after: 10, torn: true },
  { id: 'mid-rollover', label: 'New MANIFEST written, CURRENT not yet switched', after: 11 },
  { id: 'after-rollover', label: 'After CURRENT is switched', after: 12 },
];

type Mode = 'point-in-time' | 'absolute';

export function recover(crash: Crash, mode: Mode) {
  const disk = EVENTS.slice(0, crash.after + 1);
  const steps: string[] = [];

  // 1. CURRENT
  const currentEv = [...disk].reverse().find((e) => e.kind === 'current') as Extract<Ev, { kind: 'current' }> | undefined;
  const manifest = currentEv ? currentEv.manifest : 5;
  steps.push(`Read CURRENT → MANIFEST-${String(manifest).padStart(6, '0')}.`);

  // 2. replay that manifest's edits
  const live = new Map<number, number>(); // file -> level
  let logNumber = 7;
  if (manifest === 11) {
    // MANIFEST-11 begins with a snapshot of the Version as of its creation: file 10 at L1, log_number 8.
    live.set(10, 1);
    logNumber = 8;
    steps.push('Replay MANIFEST-11: its first record is a full snapshot — file 10 at L1, log_number 8.');
  } else {
    for (const e of disk) {
      if (e.kind === 'edit' && e.manifest === manifest) {
        e.add?.forEach((a) => live.set(a.file, a.level));
        e.del?.forEach((d) => live.delete(d));
        if (e.logNumber) logNumber = e.logNumber;
        steps.push(`Replay edit: ${e.add ? `add ${e.add.map((a) => `file ${a.file}@L${a.level}`).join(', ')}` : ''}${e.del ? `${e.add ? '; ' : ''}delete file ${e.del.join(', ')}` : ''}${e.logNumber ? `; log_number ${e.logNumber}` : ''}.`);
      }
    }
    if (!disk.some((e) => e.kind === 'edit')) steps.push('MANIFEST-5 has no edits yet: the live set is empty.');
  }

  // 3. orphans: SSTs and manifests on disk that the recovered state does not reference
  const onDiskSst = disk.filter((e): e is Extract<Ev, { kind: 'sst' }> => e.kind === 'sst').map((e) => e.file);
  if (crash.partialSst) onDiskSst.push(crash.partialSst);
  const orphanSst = [...new Set(onDiskSst)].filter((f) => !live.has(f));
  const orphanManifest = disk.some((e) => e.kind === 'manifest-new' && e.manifest === 11) && manifest !== 11 ? [11] : [];
  if (orphanSst.length) steps.push(`Delete orphan SST${orphanSst.length > 1 ? 's' : ''} ${orphanSst.join(', ')}: on disk, but in no recovered Version.`);
  if (orphanManifest.length) steps.push('Delete orphan MANIFEST-11: CURRENT never pointed to it.');

  // 4. replay WALs >= log_number
  const memtable = new Map<string, string>();
  let failed = false;
  const puts = disk.filter((e): e is Extract<Ev, { kind: 'put' }> => e.kind === 'put');
  const walsReplayed = [...new Set(puts.map((p) => p.wal))].filter((w) => w >= logNumber);
  steps.push(walsReplayed.length ? `Replay WAL${walsReplayed.length > 1 ? 's' : ''} ${walsReplayed.join(', ')} (numbers ≥ log_number ${logNumber}); WALs below it are already in SSTs and are skipped.` : `No WAL at or above log_number ${logNumber} holds records.`);
  for (const p of puts) {
    if (p.wal < logNumber) continue;
    const isTorn = crash.torn && p === puts[puts.length - 1];
    if (isTorn) {
      if (mode === 'absolute') {
        failed = true;
        steps.push(`WAL ${p.wal}: the record for Put(${p.key}) fails its checksum. kAbsoluteConsistency treats any corruption as fatal — the database refuses to open.`);
      } else {
        steps.push(`WAL ${p.wal}: the record for Put(${p.key}) fails its checksum. kPointInTimeRecovery stops replay at the last valid record.`);
      }
      break;
    }
    memtable.set(p.key, p.val);
  }

  // 5. visible data
  const visible = new Map<string, string>();
  for (const e of disk) if (e.kind === 'sst' && live.has(e.file)) Object.entries(e.keys).forEach(([k, v]) => visible.set(k, v));
  memtable.forEach((v, k) => visible.set(k, v));
  const written = new Map(puts.filter((p) => !(crash.torn && p === puts[puts.length - 1])).map((p) => [p.key, p.val] as const));
  const lost = [...written.keys()].filter((k) => !visible.has(k));

  return { manifest, live: [...live.entries()], logNumber, orphanSst, orphanManifest, memtable: [...memtable.entries()], visible: failed ? [] : [...visible.entries()].sort(), lost: failed ? [] : lost, failed, steps, disk };
}

export default function ManifestRecoveryLab() {
  const [crashId, setCrashId] = useState('sst-no-edit');
  const [mode, setMode] = useState<Mode>('point-in-time');
  const [step, setStep] = useState(99);
  const crash = CRASHES.find((c) => c.id === crashId)!;
  const r = useMemo(() => recover(crash, mode), [crash, mode]);
  const shownSteps = r.steps.slice(0, Math.min(step, r.steps.length));

  return (
    <VizPanel
      title="Crash, then recover from CURRENT, the MANIFEST and the WAL"
      subtitle="Pick the moment the machine loses power. The disk is frozen at that event, and recovery is worked out from nothing but what is on it."
      controls={
        <>
          <Choice label="Crash point" value={crashId} onChange={(v) => { setCrashId(v); setStep(99); }} options={CRASHES.map((c) => ({ value: c.id, label: c.label }))} />
          <Choice label="wal_recovery_mode" value={mode} onChange={(v) => { setMode(v); setStep(99); }} options={[{ value: 'point-in-time', label: 'kPointInTimeRecovery (default)' }, { value: 'absolute', label: 'kAbsoluteConsistency' }]} />
          <Button onClick={() => setStep(1)}>Replay step by step</Button>
          <Button primary onClick={() => setStep((s) => Math.min(r.steps.length, s + 1))} disabled={step >= r.steps.length}>
            Next recovery step
          </Button>
        </>
      }
      legend={
        <Legend
          items={[
            { label: 'On disk before the crash', color: 'var(--viz-clean)' },
            { label: 'Lost at the crash / never durable', color: 'var(--viz-stale)' },
            { label: 'Orphan, deleted by recovery', color: 'var(--viz-critical)' },
          ]}
        />
      }
      stats={
        <Stats
          items={[
            { label: 'CURRENT names', value: `MANIFEST-${r.manifest}` },
            { label: 'Live files', value: r.live.length ? r.live.map(([f, l]) => `${f}@L${l}`).join(', ') : 'none' },
            { label: 'Replay WALs from', value: `log_number ${r.logNumber}` },
            { label: 'Recovered keys', value: r.failed ? 'open failed' : r.visible.map(([k, v]) => `${k}=${v}`).join(' ') || 'none' },
            { label: 'Acknowledged writes lost', value: r.failed ? '—' : r.lost.length ? r.lost.join(', ') : 'none' },
          ]}
        />
      }
      note={
        <Note>
          <strong>{crash.label}.</strong>{' '}
          {r.failed
            ? 'Recovery refused to open the database: kAbsoluteConsistency tolerates no corruption at all, even a torn final record.'
            : r.orphanSst.length || r.orphanManifest.length
              ? `Files written before the crash but never recorded in the MANIFEST (${[...r.orphanSst.map((f) => `SST ${f}`), ...r.orphanManifest.map((m) => `MANIFEST-${m}`)].join(', ')}) are simply deleted. Their data is still safe, because the WAL that covered it was not retired.`
              : 'Every file on disk is part of the recovered Version, and the WAL replay restores everything written after the last flush.'}{' '}
          {crash.torn && !r.failed ? 'The torn record belonged to a write that never became durable; everything before it is recovered.' : ''}
        </Note>
      }
      table={
        <table className="viz-table">
          <thead>
            <tr>
              <th>#</th>
              <th>Event before the crash</th>
              <th>On disk?</th>
            </tr>
          </thead>
          <tbody>
            {EVENTS.map((e, i) => (
              <tr key={i}>
                <td>{i + 1}</td>
                <td>{e.text}</td>
                <td>{i <= crash.after ? (crash.torn && i === crash.after ? 'torn' : crash.partialSst && i === crash.after + 1 ? 'partial' : 'yes') : 'no — after the crash'}</td>
              </tr>
            ))}
          </tbody>
        </table>
      }
    >
      <div style={{ display: 'grid', gridTemplateColumns: 'minmax(14rem, 1fr) minmax(14rem, 1fr)', gap: '0.9rem' }}>
        <div>
          <div style={{ fontSize: '0.75rem', color: 'var(--viz-ink-2)', marginBottom: 6 }}>Disk at the moment of the crash</div>
          <ol style={{ margin: 0, paddingLeft: '1.1rem', fontSize: '0.76rem', display: 'grid', gap: 3 }}>
            {EVENTS.map((e, i) => {
              const done = i <= crash.after;
              const torn = crash.torn && i === crash.after;
              const orphan = (e.kind === 'sst' && r.orphanSst.includes(e.file)) || (e.kind === 'manifest-new' && r.orphanManifest.includes(e.manifest));
              return (
                <li key={i} style={{ color: done ? 'var(--viz-ink)' : 'var(--viz-ink-muted)', textDecoration: done ? 'none' : 'line-through', borderLeft: `3px solid ${!done || torn ? 'var(--viz-stale)' : orphan ? 'var(--viz-critical)' : 'var(--viz-clean)'}`, paddingLeft: 6 }}>
                  {torn ? '(torn) ' : ''}
                  {orphan ? '(orphan) ' : ''}
                  {e.text}
                </li>
              );
            })}
            {crash.partialSst ? (
              <li style={{ borderLeft: '3px solid var(--viz-critical)', paddingLeft: 6, color: 'var(--viz-ink)' }}>(orphan) A partial SST 000009, cut off mid-write.</li>
            ) : null}
          </ol>
        </div>
        <div>
          <div style={{ fontSize: '0.75rem', color: 'var(--viz-ink-2)', marginBottom: 6 }}>Recovery, in order</div>
          <ol style={{ margin: 0, paddingLeft: '1.1rem', fontSize: '0.78rem', display: 'grid', gap: 5 }}>
            {shownSteps.map((s, i) => (
              <li key={i} style={{ color: 'var(--viz-ink)' }}>
                {s}
              </li>
            ))}
          </ol>
        </div>
      </div>
    </VizPanel>
  );
}
