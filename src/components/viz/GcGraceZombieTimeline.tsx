import { useState } from 'react';
import { VizPanel, Slider, Check, Button, Legend, Stats, Note, TooltipHost, useTip, useSize } from './Viz';

/**
 * Why a distributed store cannot purge a tombstone on its own schedule.
 *
 * Cassandra's gc_grace_seconds (default 864000 = 10 days) is not a space knob, it is the
 * deadline by which anti-entropy must have carried the delete to every replica. A replica
 * that is offline past that deadline comes back holding the pre-delete value, and if the
 * tombstone has already been purged on the other replicas there is nothing left to say the
 * row is dead — so read repair happily copies it back. Hinted handoff covers only the first
 * max_hint_window (3 hours by default), which is far too short to be the safety net.
 */

const HINT_WINDOW_D = 0.125; // max_hint_window, 3 hours, expressed in days

type Ev = { day: number; lane: number; kind: 'delete' | 'purge' | 'down' | 'up' | 'repair' | 'zombie'; text: string };

const LANES = ['Replica A (coordinator)', 'Replica B', 'Replica C (goes offline)'];

export default function GcGraceZombieTimeline() {
  const [grace, setGrace] = useState(10);
  const [offline, setOffline] = useState(12);
  const [repairEvery, setRepairEvery] = useState(7);
  const [compacted, setCompacted] = useState(true);
  const [replaced, setReplaced] = useState(false);
  const [ref, width] = useSize(760);
  const tip = useTip();

  /* ----------------------------------------------------------------- model */

  const hinted = offline <= HINT_WINDOW_D;
  const firstRepair =
    repairEvery > 0 ? Math.ceil(Math.max(offline, 0.001) / repairEvery) * repairEvery : Infinity;
  // When the delete actually reaches C: on hint replay at rejoin, or at the next repair.
  const deliver = replaced ? offline : hinted ? offline : firstRepair;
  const purgedAt = compacted ? grace : Infinity;
  const zombie = !replaced && purgedAt <= deliver;
  const end = Math.max(
    grace,
    offline,
    Number.isFinite(deliver) ? deliver : 0,
    Number.isFinite(firstRepair) ? firstRepair : 0,
    12,
  ) + 3;

  const events: Ev[] = [
    { day: 0, lane: 0, kind: 'delete', text: 'DELETE at CL=QUORUM: a tombstone with this timestamp is written to A and B. Nothing is searched for and nothing is overwritten.' },
    { day: 0, lane: 1, kind: 'delete', text: 'B takes the same tombstone. Two of three replicas acknowledge, so the client is told the delete succeeded.' },
    { day: 0, lane: 2, kind: 'down', text: `C is down and never sees the delete. The coordinator stores a hint, which is discarded after max_hint_window (3 hours).` },
    { day: offline, lane: 2, kind: 'up', text: replaced
      ? 'C is rebuilt from scratch (nodetool removenode, then a fresh bootstrap / replace). It streams current data from A and B and carries no stale row of its own.'
      : hinted
        ? 'C rejoins inside the hint window, so the stored hint is replayed and C learns about the delete immediately.'
        : 'C rejoins carrying the pre-delete value, and the hint that would have told it about the delete expired long ago.' },
  ];
  if (compacted) {
    events.push({
      day: grace,
      lane: 0,
      kind: 'purge',
      text: `gc_grace_seconds has elapsed, so the next compaction that touches this SSTable may drop the tombstone — provided no other SSTable holds older data for the partition. After this instant, A has no record that the row was ever deleted.`,
    });
  }
  if (Number.isFinite(firstRepair) && !replaced && !hinted) {
    events.push({
      day: firstRepair,
      lane: 2,
      kind: 'repair',
      text: `nodetool repair reconciles A, B and C by Merkle tree. If the tombstone is still there it wins on timestamp and C's stale value dies. If it has already been purged, the only surviving version of this row is C's live value — and repair pushes it back to A and B.`,
    });
  }
  if (zombie) {
    events.push({
      day: Math.min(Number.isFinite(deliver) ? deliver : end - 1, end - 1),
      lane: 0,
      kind: 'zombie',
      text: 'Resurrection: the deleted row is live on all three replicas again, with its original write timestamp. No error was raised anywhere; the client simply sees a row it deleted weeks ago.',
    });
  }

  /* -------------------------------------------------------------- geometry */

  const labelW = 168;
  const svgW = Math.max(width, 620);
  const plotW = svgW - labelW - 24;
  const x = (d: number) => labelW + (Math.min(d, end) / end) * plotW;
  const laneH = 54;
  const top = 30;
  const svgH = top + LANES.length * laneH + 46;
  const laneY = (i: number) => top + i * laneH;

  const ticks = Array.from({ length: 7 }, (_, i) => Math.round((end / 6) * i));

  return (
    <VizPanel
      title="gc_grace_seconds against a replica that stayed down too long"
      subtitle="A row is deleted while one of three replicas is offline. gc_grace is the deadline for anti-entropy to carry that delete everywhere; miss it and the delete is what gets forgotten."
      controls={
        <>
          <Slider label="gc_grace_seconds" min={0} max={20} value={grace} onChange={setGrace} format={(n) => `${n} d`} />
          <Slider label="Replica C offline" min={0} max={20} step={1} value={offline} onChange={setOffline} format={(n) => `${n} d`} />
          <Slider
            label="nodetool repair every"
            min={0}
            max={30}
            value={repairEvery}
            onChange={setRepairEvery}
            format={(n) => (n === 0 ? 'never' : `${n} d`)}
          />
          <Check label="A compaction purges the tombstone once it may" checked={compacted} onChange={setCompacted} />
          <Check label="Rebuild C instead of rejoining it" checked={replaced} onChange={setReplaced} />
          <Button
            onClick={() => {
              setGrace(10);
              setOffline(12);
              setRepairEvery(7);
              setCompacted(true);
              setReplaced(false);
            }}
          >
            Defaults
          </Button>
        </>
      }
      legend={
        <Legend
          items={[
            { label: 'live value', color: 'var(--viz-clean)' },
            { label: 'tombstone present', color: 'var(--viz-8)' },
            { label: 'replica offline', color: 'var(--viz-stale)' },
            { label: 'gc_grace window', color: 'var(--viz-warning)' },
            { label: 'resurrected row', color: 'var(--viz-critical)', shape: 'dot' },
          ]}
        />
      }
      stats={
        <Stats
          items={[
            { label: 'Hint window', value: '3 h', hint: 'max_hint_window — the only automatic repair, and it expires almost immediately' },
            { label: 'C rejoins', value: `day ${offline}` },
            { label: 'Delete reaches C', value: replaced ? `day ${offline} (rebuilt)` : hinted ? `day ${offline} (hint)` : Number.isFinite(firstRepair) ? `day ${firstRepair} (repair)` : 'never' },
            { label: 'Tombstone purgeable', value: compacted ? `day ${grace}` : 'never (no compaction)' },
            { label: 'Repair inside gc_grace?', value: Number.isFinite(deliver) && deliver <= grace ? 'yes' : 'no' },
            { label: 'Read at the end returns', value: zombie ? 'the deleted row' : 'not found' },
          ]}
        />
      }
      note={
        <Note>
          {zombie ? (
            <>
              <strong>Zombie.</strong> C came back on day {offline} still holding the value, and the tombstone had
              already been purged on day {grace}. With no tombstone anywhere, C&apos;s copy is the newest thing the
              cluster knows about this row, so{' '}
              {Number.isFinite(firstRepair) ? 'repair' : 'the next read repair'} copies it back to A and B. The
              delete is gone; the row is not. This is exactly what gc_grace_seconds exists to prevent, and why
              lowering it to reclaim space is the most dangerous knob in the file.
            </>
          ) : replaced ? (
            <>
              <strong>Safe, the correct way.</strong> C was removed and rebuilt rather than rejoined, so it streamed
              its data from replicas that agree the row is dead. This is the documented procedure for a node that has
              been down longer than gc_grace_seconds — rejoining it is not.
            </>
          ) : hinted ? (
            <>
              <strong>Safe, by luck.</strong> C was back inside the 3-hour hint window, so hinted handoff replayed the
              delete. Hints are a convenience for a reboot, not a correctness mechanism: they are capped by
              max_hint_window and dropped when the coordinator runs out of hint space.
            </>
          ) : (
            <>
              <strong>Safe, by repair.</strong> The delete reached C on day{' '}
              {Number.isFinite(deliver) ? deliver : '—'}, before the tombstone was purged on day{' '}
              {compacted ? grace : '—'}. That ordering is the entire contract: every replica must be repaired within
              gc_grace_seconds, which is why the default is a generous 10 days and why an operator who skips repair
              is running on borrowed time.
            </>
          )}
        </Note>
      }
      table={
        <table className="viz-table">
          <thead>
            <tr>
              <th>Day</th>
              <th>Replica</th>
              <th>Event</th>
              <th>Row state afterwards</th>
            </tr>
          </thead>
          <tbody>
            {[...events]
              .sort((a, b) => a.day - b.day)
              .map((e, i) => (
                <tr key={`${e.kind}-${e.lane}-${i}`}>
                  <td>{Number.isFinite(e.day) ? e.day : '—'}</td>
                  <td>{LANES[e.lane]}</td>
                  <td>{e.kind}</td>
                  <td>
                    {e.kind === 'delete'
                      ? 'tombstone'
                      : e.kind === 'purge'
                        ? 'nothing at all'
                        : e.kind === 'zombie'
                          ? 'live value, resurrected'
                          : e.kind === 'up'
                            ? replaced || hinted
                              ? 'tombstone'
                              : 'stale live value'
                            : e.kind === 'repair'
                              ? zombie
                                ? 'live value pushed back to A and B'
                                : 'tombstone'
                              : 'offline'}
                  </td>
                </tr>
              ))}
          </tbody>
        </table>
      }
    >
      <div ref={ref}>
        <TooltipHost>
          <svg width={svgW} height={svgH} role="img" aria-label="Timeline of a delete, a replica outage, gc_grace expiry and a resurrected row">
            {/* gc_grace window */}
            {compacted ? (
              <rect x={x(0)} y={top - 14} width={x(grace) - x(0)} height={LANES.length * laneH + 6} fill="var(--viz-warning)" opacity={0.12} />
            ) : null}
            <line x1={x(grace)} y1={top - 16} x2={x(grace)} y2={top + LANES.length * laneH - 6} stroke="var(--viz-warning)" strokeWidth={2} />
            <text x={x(grace) + 5} y={top - 18} fontSize={11} fill="var(--viz-warning)">
              gc_grace ends (day {grace})
            </text>

            {LANES.map((ln, i) => {
              const y = laneY(i);
              const isC = i === 2;
              return (
                <g key={ln}>
                  <text x={0} y={y + 18} fill="var(--viz-ink)" fontSize={12} fontWeight={600}>
                    {ln}
                  </text>
                  <line x1={labelW} y1={y + 24} x2={svgW - 16} y2={y + 24} stroke="var(--viz-grid)" />

                  {isC ? (
                    <>
                      {/* C holds the live value until the delete reaches it */}
                      <rect
                        x={x(0)}
                        y={y + 12}
                        width={Math.max(2, x(Number.isFinite(deliver) ? deliver : end) - x(0))}
                        height={12}
                        rx={3}
                        fill="var(--viz-clean)"
                        opacity={0.9}
                      />
                      <rect x={x(0)} y={y + 4} width={Math.max(2, x(offline) - x(0))} height={8} rx={3} fill="var(--viz-stale)" />
                      <text x={x(0) + 4} y={y + 1} fontSize={11} fill="var(--viz-ink-muted)">
                        offline
                      </text>
                      {Number.isFinite(deliver) ? (
                        <rect x={x(deliver)} y={y + 12} width={Math.max(2, x(end) - x(deliver))} height={12} rx={3} fill={zombie ? 'var(--viz-clean)' : 'var(--viz-8)'} />
                      ) : null}
                    </>
                  ) : (
                    <>
                      {/* A and B: tombstone from day 0 until it is purged */}
                      <rect x={x(0)} y={y + 12} width={Math.max(2, x(compacted ? grace : end) - x(0))} height={12} rx={3} fill="var(--viz-8)" />
                      {compacted ? (
                        <rect
                          x={x(grace)}
                          y={y + 12}
                          width={Math.max(2, x(end) - x(grace))}
                          height={12}
                          rx={3}
                          fill={zombie ? 'var(--viz-clean)' : 'var(--viz-neutral)'}
                          stroke="var(--viz-border)"
                        />
                      ) : null}
                      {zombie && Number.isFinite(deliver) ? (
                        <line
                          x1={x(deliver)}
                          y1={laneY(2) + 12}
                          x2={x(deliver)}
                          y2={y + 24}
                          stroke="var(--viz-critical)"
                          strokeWidth={2}
                          strokeDasharray="4 3"
                        />
                      ) : null}
                    </>
                  )}
                </g>
              );
            })}

            {events.map((e, i) => (
              <g key={`${e.kind}-${e.lane}-${i}`} {...tip(<><strong>Day {Number.isFinite(e.day) ? e.day : '—'} · {LANES[e.lane]}</strong><br />{e.text}</>)}>
                <circle
                  cx={x(e.day)}
                  cy={laneY(e.lane) + 18}
                  r={6}
                  fill={
                    e.kind === 'delete'
                      ? 'var(--viz-8)'
                      : e.kind === 'purge'
                        ? 'var(--viz-warning)'
                        : e.kind === 'zombie'
                          ? 'var(--viz-critical)'
                          : e.kind === 'repair'
                            ? 'var(--viz-6)'
                            : 'var(--viz-ink-2)'
                  }
                  stroke="var(--viz-surface)"
                  strokeWidth={1.5}
                />
                <text x={x(e.day)} y={laneY(e.lane) + 21} fontSize={9} textAnchor="middle" fill="var(--viz-surface)" fontWeight={700}>
                  {e.kind === 'delete' ? 'D' : e.kind === 'purge' ? '✕' : e.kind === 'repair' ? 'R' : e.kind === 'zombie' ? 'Z' : e.kind === 'up' ? '↑' : '↓'}
                </text>
              </g>
            ))}

            {ticks.map((d) => (
              <g key={d}>
                <line x1={x(d)} y1={top + LANES.length * laneH - 6} x2={x(d)} y2={top + LANES.length * laneH} stroke="var(--viz-axis)" />
                <text x={x(d)} y={top + LANES.length * laneH + 14} fontSize={11} textAnchor="middle" fill="var(--viz-ink-muted)">
                  day {d}
                </text>
              </g>
            ))}
          </svg>
        </TooltipHost>
      </div>
    </VizPanel>
  );
}
