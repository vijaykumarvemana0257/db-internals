import { useMemo, useState } from 'react';
import { VizPanel, Segmented, Slider, Check, Legend, Stats, Note, fmtNum } from './Viz';

type Panel = 'compact' | 'scan' | 'zombie';

/* ------------------------------------------------------ compaction + snapshots */
type Version = { seq: number; kind: 'put' | 'del'; val?: string; level: string };
const HISTORY: Version[] = [
  { seq: 30, kind: 'del', level: 'L1' },
  { seq: 20, kind: 'put', val: 'v2', level: 'L1' },
  { seq: 10, kind: 'put', val: 'v1', level: 'L2' },
];

/**
 * What compaction of these versions may keep, following RocksDB's rules:
 * - a version survives if it is the newest version visible to the latest read or to some live snapshot;
 * - a tombstone is dropped only at the bottommost level AND when its sequence number is at or below the
 *   earliest live snapshot (so no snapshot can still see anything it hides); older versions go with it.
 */
export function compact(snapshots: number[], bottommost: boolean) {
  const snaps = [...snapshots].sort((a, b) => a - b);
  const earliest = snaps.length ? snaps[0] : Infinity;
  const readers = [...snaps, Infinity]; // Infinity = a read at the latest sequence number
  const newestTomb = HISTORY.find((v) => v.kind === 'del');
  const verdict = HISTORY.map((v) => {
    const visibleTo = readers.filter((r) => HISTORY.find((x) => x.seq <= r)?.seq === v.seq);
    if (v.kind === 'del') {
      if (bottommost && v.seq <= earliest) return { v, keep: false, why: `dropped: bottommost level, and no live snapshot is older than seq ${v.seq}` };
      if (!bottommost) return { v, keep: true, why: 'kept: lower levels may still hold older versions it has to hide' };
      return { v, keep: true, why: `kept: snapshot ${snaps.find((s) => s < v.seq)} is older and could still read what this hides` };
    }
    if (newestTomb && bottommost && newestTomb.seq > v.seq && newestTomb.seq <= earliest) {
      return { v, keep: false, why: 'dropped together with the tombstone that hides it' };
    }
    if (visibleTo.length) {
      const who = visibleTo.map((r) => (r === Infinity ? 'the latest read' : `snapshot ${r}`)).join(' and ');
      return { v, keep: true, why: `kept: newest version visible to ${who}` };
    }
    return { v, keep: false, why: 'dropped: a newer version is visible to every reader that could see it' };
  });
  const read = (at: number) => {
    const surviving = verdict.filter((x) => x.keep).map((x) => x.v);
    const hit = surviving.find((x) => x.seq <= at);
    return hit ? (hit.kind === 'del' ? 'not found' : hit.val!) : 'not found';
  };
  return { verdict, read };
}

/* ------------------------------------------------------ queue scan */
export function queueScan(enqueued: number, consumed: number, compacted: boolean, snapshotHeld: boolean) {
  const tombstonesLeft = compacted && !snapshotHeld ? 0 : consumed;
  const live = enqueued - consumed;
  return { tombstonesLeft, live, skippedPerLiveRow: live > 0 ? tombstonesLeft : tombstonesLeft };
}

/* ------------------------------------------------------ gc_grace zombies */
export function zombie(downDays: number, graceDays: number) {
  // Day 0: DELETE reaches replicas A and C; B is down and still holds the old row.
  // Compaction purges A's and C's tombstone once gc_grace has passed. B returns after downDays.
  const purged = downDays > graceDays;
  return {
    purged,
    outcome: purged ? 'zombie' : 'deleted',
    events: [
      { day: 0, text: 'DELETE succeeds on A and C (a quorum). B is down and misses it.' },
      ...(purged ? [{ day: graceDays, text: `gc_grace passes; compaction on A and C purges the tombstone.` }] : []),
      { day: downDays, text: purged ? 'B returns with the old row. Repair finds B has data A and C lack — and copies it back.' : 'B returns; repair finds the tombstone on A and C and applies it to B.' },
    ],
  };
}

export default function TombstoneZombieLab() {
  const [panel, setPanel] = useState<Panel>('compact');

  const [snap15, setSnap15] = useState(false);
  const [snap25, setSnap25] = useState(true);
  const [bottommost, setBottommost] = useState(true);
  const snapshots = [snap15 ? 15 : null, snap25 ? 25 : null].filter((x): x is number => x !== null);
  const c = useMemo(() => compact(snapshots, bottommost), [snap15, snap25, bottommost]);

  const [enqueued, setEnqueued] = useState(20000);
  const [consumedPct, setConsumedPct] = useState(95);
  const [compacted, setCompacted] = useState(false);
  const [snapHeld, setSnapHeld] = useState(false);
  const consumed = Math.round((enqueued * consumedPct) / 100);
  const q = queueScan(enqueued, consumed, compacted, snapHeld);

  const [downDays, setDownDays] = useState(8);
  const [graceDays, setGraceDays] = useState(10);
  const z = zombie(downDays, graceDays);

  const W = 660;
  return (
    <VizPanel
      title="Deletes are writes, and they stay until compaction can prove they are safe to forget"
      subtitle="Pick a scenario. Each one is a rule about when a tombstone — the record that says “this key is deleted” — may finally be dropped, and what goes wrong when it is dropped too early or kept too long."
      controls={
        <Segmented
          label="Scenario"
          value={panel}
          onChange={setPanel}
          options={[
            { value: 'compact', label: 'Compaction and snapshots' },
            { value: 'scan', label: 'A queue full of tombstones' },
            { value: 'zombie', label: 'gc_grace and zombies' },
          ]}
        />
      }
      legend={
        panel === 'compact' ? (
          <Legend
            items={[
              { label: 'Put', color: 'var(--viz-clean)' },
              { label: 'Tombstone', color: 'var(--viz-critical)' },
              { label: 'Kept by this compaction', color: 'var(--viz-ink)' },
              { label: 'Dropped (dashed)', color: 'var(--viz-stale)' },
            ]}
          />
        ) : panel === 'scan' ? (
          <Legend
            items={[
              { label: 'Tombstone a scan must step over', color: 'var(--viz-stale)' },
              { label: 'Live row', color: 'var(--viz-clean)' },
            ]}
          />
        ) : (
          <Legend
            items={[
              { label: 'Row present', color: 'var(--viz-clean)' },
              { label: 'Tombstone', color: 'var(--viz-critical)' },
              { label: 'Replica down', color: 'var(--viz-stale)' },
            ]}
          />
        )
      }
      stats={
        panel === 'compact' ? (
          <Stats
            items={[
              { label: 'Versions kept', value: `${c.verdict.filter((x) => x.keep).length} of ${HISTORY.length}` },
              { label: 'Latest read returns', value: c.read(Infinity) },
              ...snapshots.map((s) => ({ label: `Snapshot ${s} reads`, value: c.read(s) })),
            ]}
          />
        ) : panel === 'scan' ? (
          <Stats
            items={[
              { label: 'Live messages', value: fmtNum(q.live) },
              { label: 'Tombstones still on disk', value: fmtNum(q.tombstonesLeft) },
              { label: 'Stepped over to find the first live row', value: fmtNum(q.tombstonesLeft) },
              { label: 'vs Cassandra warn / fail thresholds', value: q.tombstonesLeft >= 100000 ? 'fails (≥ 100,000)' : q.tombstonesLeft >= 1000 ? 'warns (≥ 1,000)' : 'below warning' },
            ]}
          />
        ) : (
          <Stats
            items={[
              { label: 'Replica B down for', value: `${downDays} days` },
              { label: 'gc_grace_seconds', value: `${graceDays} days`, hint: 'Cassandra default: 864,000 s = 10 days' },
              { label: 'Outcome', value: z.outcome === 'zombie' ? '⚠ deleted row returns' : '✓ stays deleted' },
            ]}
          />
        )
      }
      note={
        <Note>
          {panel === 'compact' ? (
            <>
              <strong>
                {bottommost ? 'Compacting into the bottommost level' : 'Compacting into a level with more data below it'}, with {snapshots.length ? `snapshot${snapshots.length > 1 ? 's' : ''} ${snapshots.join(' and ')} held` : 'no snapshots held'}:
              </strong>{' '}
              {c.verdict.map((x) => `${x.v.kind === 'del' ? 'tombstone' : x.v.val}@${x.v.seq} ${x.why}`).join('; ')}.
            </>
          ) : panel === 'scan' ? (
            <>
              <strong>
                To return the first live message, the scan steps over {fmtNum(q.tombstonesLeft)} tombstone{q.tombstonesLeft === 1 ? '' : 's'}.
              </strong>{' '}
              {compacted && snapHeld
                ? 'Compaction ran, but an open snapshot older than the deletes keeps every tombstone alive.'
                : compacted
                  ? 'Compaction reached the bottom level with no older snapshot, so the tombstones — and the messages they hid — are gone.'
                  : 'The consumer deleted messages from the head of the key range, so every delete is a tombstone sitting exactly where the next scan starts.'}
            </>
          ) : (
            <>
              <strong>{z.outcome === 'zombie' ? 'Zombie.' : 'Deleted everywhere.'}</strong> {z.events.map((e) => `Day ${e.day}: ${e.text}`).join(' ')}
            </>
          )}
        </Note>
      }
    >
      {panel === 'compact' ? (
        <>
          <div className="viz-controls">
            <Check label="Hold snapshot at seq 15" checked={snap15} onChange={setSnap15} />
            <Check label="Hold snapshot at seq 25" checked={snap25} onChange={setSnap25} />
            <Check label="Output level is the bottommost (nothing overlaps below)" checked={bottommost} onChange={setBottommost} />
          </div>
          <svg width={W} height={150} role="img" aria-label="Versions of one key and what compaction keeps">
            <text x={0} y={16} fill="var(--viz-ink)">Key “k”, newest version first</text>
            {c.verdict.map((x, i) => {
              const xx = 20 + i * 200;
              const color = x.v.kind === 'del' ? 'var(--viz-critical)' : 'var(--viz-clean)';
              return (
                <g key={x.v.seq}>
                  <rect x={xx} y={30} width={170} height={44} rx={8} fill="var(--viz-surface)" stroke={x.keep ? color : 'var(--viz-stale)'} strokeWidth={x.keep ? 3 : 2} strokeDasharray={x.keep ? undefined : '6 4'} />
                  <text x={xx + 85} y={50} textAnchor="middle" fill={x.keep ? 'var(--viz-ink)' : 'var(--viz-ink-muted)'} fontWeight={600}>
                    {x.v.kind === 'del' ? 'tombstone' : `put ${x.v.val}`} @ seq {x.v.seq}
                  </text>
                  <text x={xx + 85} y={66} textAnchor="middle" fontSize={10}>
                    {x.keep ? 'kept' : 'dropped'} · was in {x.v.level}
                  </text>
                </g>
              );
            })}
            <text x={0} y={104} fill="var(--viz-ink)">Readers</text>
            {[...snapshots, Infinity].map((r, i) => (
              <text key={String(r)} x={20 + i * 200} y={124} fontSize={11}>
                {r === Infinity ? 'latest read' : `snapshot @ ${r}`} → {c.read(r)}
              </text>
            ))}
          </svg>
        </>
      ) : panel === 'scan' ? (
        <>
          <div className="viz-controls">
            <Slider label="Messages enqueued" min={1000} max={200000} step={1000} value={enqueued} onChange={setEnqueued} format={fmtNum} />
            <Slider label="Already consumed (deleted)" min={0} max={99} value={consumedPct} onChange={setConsumedPct} format={(v) => `${v}%`} />
            <Check label="Compaction has reached the bottom level" checked={compacted} onChange={setCompacted} />
            <Check label="A long-running snapshot is open" checked={snapHeld} onChange={setSnapHeld} />
          </div>
          <svg width={W} height={70} role="img" aria-label="Queue key range: tombstones before the first live message">
            {Array.from({ length: 100 }, (_, i) => {
              const isTomb = i < Math.round((q.tombstonesLeft / enqueued) * 100);
              const isGone = !isTomb && i < consumedPct;
              return <rect key={i} x={i * 6.5} y={10} width={5.5} height={30} rx={1} fill={isGone ? 'var(--viz-plane)' : isTomb ? 'var(--viz-stale)' : 'var(--viz-clean)'} />;
            })}
            <text x={0} y={60} fontSize={10}>
              head of the queue (oldest key)
            </text>
            <text x={650} y={60} fontSize={10} textAnchor="end">
              tail (newest key)
            </text>
          </svg>
        </>
      ) : (
        <>
          <div className="viz-controls">
            <Slider label="Replica B down for" min={1} max={30} value={downDays} onChange={setDownDays} format={(v) => `${v} days`} />
            <Slider label="gc_grace_seconds" min={1} max={30} value={graceDays} onChange={setGraceDays} format={(v) => `${v} days`} />
          </div>
          <svg width={W} height={150} role="img" aria-label={`Replica timeline: outcome ${z.outcome}`}>
            {['A', 'B', 'C'].map((rep, ri) => {
              const y = 14 + ri * 36;
              const scale = (d: number) => 60 + (d / 30) * 580;
              const bDown = rep === 'B';
              return (
                <g key={rep}>
                  <text x={0} y={y + 16} fill="var(--viz-ink)">
                    Replica {rep}
                  </text>
                  <rect x={60} y={y} width={580} height={22} rx={4} fill="var(--viz-plane)" stroke="var(--viz-border)" />
                  {bDown ? <rect x={scale(0)} y={y} width={scale(downDays) - scale(0)} height={22} fill="var(--viz-stale)" opacity={0.6} /> : null}
                  {!bDown ? (
                    <rect x={scale(0)} y={y + 4} width={(z.purged ? scale(graceDays) : scale(30)) - scale(0)} height={14} rx={3} fill="var(--viz-critical)" />
                  ) : null}
                  {bDown ? <rect x={scale(downDays)} y={y + 4} width={scale(30) - scale(downDays)} height={14} rx={3} fill={z.purged ? 'var(--viz-clean)' : 'var(--viz-critical)'} /> : null}
                  {!bDown && z.purged ? <rect x={scale(downDays)} y={y + 4} width={scale(30) - scale(downDays)} height={14} rx={3} fill="var(--viz-clean)" /> : null}
                </g>
              );
            })}
            <line x1={60 + (graceDays / 30) * 580} x2={60 + (graceDays / 30) * 580} y1={8} y2={122} stroke="var(--viz-ink)" strokeDasharray="4 3" />
            <text x={62 + (graceDays / 30) * 580} y={136} fontSize={10}>
              gc_grace ends (day {graceDays})
            </text>
            <line x1={60 + (downDays / 30) * 580} x2={60 + (downDays / 30) * 580} y1={8} y2={122} stroke="var(--viz-ink-2)" />
            <text x={62 + (downDays / 30) * 580} y={148} fontSize={10}>
              B returns (day {downDays})
            </text>
          </svg>
        </>
      )}
    </VizPanel>
  );
}
