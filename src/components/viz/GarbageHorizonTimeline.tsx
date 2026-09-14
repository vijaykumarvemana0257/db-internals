import { useState } from 'react';
import {
  VizPanel,
  Segmented,
  Slider,
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
 * Versions, snapshots and the garbage horizon.
 *
 * One logical row, many physical versions. A version created by transaction m and
 * superseded by transaction d occupies the half-open interval [m, d): a snapshot taken
 * at S (meaning "every id below S is finished") sees it iff m < S <= d. The horizon is
 * one scalar — the smallest S among all open readers, or the next id if there are none —
 * and a version is collectable iff d < horizon, i.e. iff its whole box lies to the left
 * of the horizon line. Everything else the collector must keep, including versions that
 * no reader can actually see: the horizon does not ask which reader wanted which version.
 */

const START_XID = 742;
const MAX_VERSIONS = 8;
const MAX_SPAN = 15; // ids on the axis before writes are cut off
const BYTES_PER_VERSION = 128; // modelled, and labelled as such in the stats

/* ----------------------------------------------------------------- engines */

type EngineId = 'pg' | 'innodb' | 'lsm';

type Engine = {
  id: EngineId;
  label: string;
  unit: string; // what the axis counts
  liveNoun: string; // where the current version lives
  deadNoun: string; // where a superseded version lives
  reclaim: string; // what the collector is called
  horizon: string; // what the engine calls the horizon
  who: string; // who runs the collector
};

const ENGINES: Record<EngineId, Engine> = {
  pg: {
    id: 'pg',
    label: 'PostgreSQL heap',
    unit: 'xid',
    liveNoun: 'heap tuple',
    deadNoun: 'dead heap tuple',
    reclaim: 'VACUUM',
    horizon: 'OldestXmin',
    who: 'autovacuum',
  },
  innodb: {
    id: 'innodb',
    label: 'InnoDB undo',
    unit: 'trx_id',
    liveNoun: 'clustered-index row',
    deadNoun: 'undo record',
    reclaim: 'purge',
    horizon: 'oldest read view',
    who: 'the purge threads',
  },
  lsm: {
    id: 'lsm',
    label: 'LSM (RocksDB)',
    unit: 'seq',
    liveNoun: 'newest entry for the key',
    deadNoun: 'older entry for the key',
    reclaim: 'compact',
    horizon: 'oldest snapshot seq',
    who: 'compaction',
  },
};

/* ------------------------------------------------------------------- state */

type Version = {
  vid: number;
  xmin: number;
  xmax: number | null; // null = still the current version
  tomb: boolean; // this version was ended by a DELETE, not an UPDATE
};

type Reader = { rid: number; snap: number };

type S = {
  next: number; // the next id to be handed out
  versions: Version[];
  readers: Reader[];
  nextVid: number;
  nextRid: number;
  reclaimed: number;
  deleted: boolean; // the row currently has no live version
  head: string;
  body: string;
};

const INITIAL: S = {
  next: 743,
  versions: [{ vid: 1, xmin: 742, xmax: null, tomb: false }],
  readers: [{ rid: 1, snap: 743 }],
  nextVid: 2,
  nextRid: 2,
  reclaimed: 0,
  deleted: false,
  head: 'One row, one version, one open reader.',
  body:
    'R1 took its snapshot at 743, so it sees every id below 743 as finished. Press UPDATE: the row is ' +
    'not overwritten — v1 gets an end id and a new version is appended, and R1 keeps reading v1.',
};

/** Deterministic physical addresses, so the tooltips read like a real page layout. */
const CTIDS = (() => {
  const rng = makeRng(19990625); // Postgres 6.5, the release that introduced MVCC
  return Array.from({ length: 40 }, () => ({ page: Math.floor(rng() * 6), off: 1 + Math.floor(rng() * 38) }));
})();

const ctid = (vid: number) => CTIDS[vid % CTIDS.length];

/* -------------------------------------------------------------- the rules */

/** A snapshot at S sees every id below S as finished. */
function visibleTo(v: Version, snap: number) {
  return v.xmin < snap && (v.xmax === null || v.xmax >= snap);
}

function horizonOf(s: S) {
  return s.readers.length === 0 ? s.next : Math.min(...s.readers.map((r) => r.snap));
}

type State = 'current' | 'pinned' | 'retained' | 'collectable';

function classify(v: Version, s: S, h: number): State {
  if (v.xmax === null) return 'current';
  if (v.xmax < h) return 'collectable';
  return s.readers.some((r) => visibleTo(v, r.snap)) ? 'pinned' : 'retained';
}

const STATE_STYLE: Record<State, { color: string; glyph: string; word: string }> = {
  current: { color: 'var(--viz-clean)', glyph: '●', word: 'current' },
  pinned: { color: 'var(--viz-warning)', glyph: '◆', word: 'dead, pinned by a reader' },
  retained: { color: 'var(--viz-serious)', glyph: '▲', word: 'dead to everyone, held by the horizon' },
  collectable: { color: 'var(--viz-stale)', glyph: '✕', word: 'collectable' },
};

/* ----------------------------------------------------------------- actions */

type Op = 'update' | 'delete' | 'insert' | 'open' | 'close' | 'reclaim';

function apply(s: S, op: Op, e: Engine): S {
  const next = { ...s, versions: [...s.versions], readers: [...s.readers] };
  const id = s.next;

  switch (op) {
    case 'update':
    case 'delete': {
      const i = next.versions.findIndex((v) => v.xmax === null);
      if (i < 0) return s;
      next.versions[i] = { ...next.versions[i], xmax: id, tomb: op === 'delete' };
      if (op === 'update') {
        next.versions.push({ vid: s.nextVid, xmin: id, xmax: null, tomb: false });
        next.nextVid = s.nextVid + 1;
      }
      next.deleted = op === 'delete';
      next.next = id + 1;
      const h = horizonOf(next);
      const stillSeen = next.versions[i].xmax! >= h;
      next.head =
        op === 'update'
          ? `${e.unit} ${id} UPDATE: v${next.versions[i].vid} ended, v${s.nextVid} appended.`
          : `${e.unit} ${id} DELETE: v${next.versions[i].vid} ended, nothing replaces it.`;
      next.body = stillSeen
        ? `The old version is dead, but its end id ${id} is at or past the horizon (${h}), so nobody may ` +
          `remove it yet. This is the backlog: every write while a reader sits open adds one more ` +
          `${e.deadNoun} that ${e.who} has to skip over.`
        : `Its end id ${id} is already below the horizon (${h}), so no current or future snapshot can ` +
          `reach it — ${e.reclaim} may take it on the next pass.`;
      return next;
    }
    case 'insert': {
      next.versions.push({ vid: s.nextVid, xmin: id, xmax: null, tomb: false });
      next.nextVid = s.nextVid + 1;
      next.deleted = false;
      next.next = id + 1;
      next.head = `${e.unit} ${id} INSERT: v${s.nextVid} is the current version again.`;
      next.body =
        'A re-inserted row is a brand-new version; the deleted one is still on disk until the collector ' +
        'gets to it.';
      return next;
    }
    case 'open': {
      next.readers.push({ rid: s.nextRid, snap: s.next });
      next.nextRid = s.nextRid + 1;
      const h = horizonOf(next);
      next.head = `R${s.nextRid} opened with a snapshot at ${s.next}.`;
      next.body =
        h === s.next
          ? `It is now the oldest reader, so the horizon is ${h}: everything ended before ${h} is garbage, ` +
            `everything ended at or after it must be kept.`
          : `An older reader (horizon ${h}) still sets the horizon — a new reader never lets the collector ` +
            `catch up, only the oldest one leaving does.`;
      return next;
    }
    case 'close': {
      if (next.readers.length === 0) return s;
      const oldest = next.readers.reduce((a, b) => (a.snap <= b.snap ? a : b));
      next.readers = next.readers.filter((r) => r.rid !== oldest.rid);
      const h = horizonOf(next);
      next.head = `R${oldest.rid} committed; the horizon moves from ${oldest.snap} to ${h}.`;
      next.body =
        `Nothing was rewritten and no lock was released — the only thing that changed is one number, and ` +
        `a batch of versions became collectable all at once.`;
      return next;
    }
    case 'reclaim': {
      const h = horizonOf(next);
      const dead = next.versions.filter((v) => v.xmax !== null);
      const gone = dead.filter((v) => v.xmax! < h);
      const kept = dead.length - gone.length;
      next.versions = next.versions.filter((v) => v.xmax === null || v.xmax! >= h);
      next.reclaimed = s.reclaimed + gone.length;
      const live = next.versions.filter((v) => v.xmax === null).length;
      next.head =
        e.id === 'pg'
          ? `tuples: ${gone.length} removed, ${live} remain, ${kept} are dead but not yet ` +
            `removable, oldest xmin: ${h}`
          : e.id === 'innodb'
            ? `purge removed ${gone.length} undo record${gone.length === 1 ? '' : 's'}; ${kept} stay on the ` +
              `history list behind the oldest read view (${h}).`
            : `compaction dropped ${gone.length} entr${gone.length === 1 ? 'y' : 'ies'}; ${kept} survived ` +
              `because a snapshot (${h}) falls between them and the newer value.`;
      next.body =
        kept > 0
          ? 'The collector ran to completion and still left work behind. It is not slow and it is not ' +
            'behind: it is forbidden from touching anything ending at or after the horizon.'
          : 'Nothing is left behind: no reader is older than any dead version, so the whole chain collapsed ' +
            'to the current version.';
      return next;
    }
  }
}

/* --------------------------------------------------------------- component */

export default function GarbageHorizonTimeline() {
  const [engineId, setEngineId] = useState<EngineId>('pg');
  const [s, setS] = useState<S>(INITIAL);
  const [ref, width] = useSize(760);
  const tip = useTip();

  const e = ENGINES[engineId];
  const h = horizonOf(s);
  const run = (op: Op) => setS((cur) => apply(cur, op, e));

  const domainStart = Math.min(START_XID, ...s.versions.map((v) => v.xmin));
  const domainEnd = s.next + 1;
  const span = domainEnd - domainStart + 1;
  const atLimit = s.next - domainStart >= MAX_SPAN || s.versions.length >= MAX_VERSIONS;

  const rows = s.versions.map((v) => ({ v, state: classify(v, s, h) }));
  const nCollect = rows.filter((r) => r.state === 'collectable').length;
  const nPinned = rows.filter((r) => r.state === 'pinned').length;
  const nRetained = rows.filter((r) => r.state === 'retained').length;
  const oldest = s.readers.length ? s.readers.reduce((a, b) => (a.snap <= b.snap ? a : b)) : null;

  /* geometry */
  const gutter = 176;
  const colW = 54;
  const svgW = Math.max(width, gutter + span * colW + 90);
  const axisH = 34;
  const rowH = 44;
  const readerH = 40;
  const gap = 14;
  const versTop = axisH + 8;
  const readTop = versTop + rows.length * rowH + gap;
  const height = readTop + Math.max(1, s.readers.length) * readerH + 16;

  /** left edge of an id's slot — where a snapshot line sits */
  const lineX = (x: number) => gutter + (x - domainStart) * colW;
  /** middle of an id's slot — where a version begins or ends */
  const midX = (x: number) => gutter + (x - domainStart + 0.5) * colW;
  const rightEdge = gutter + span * colW;

  return (
    <VizPanel
      title="Versions, snapshots and the garbage horizon"
      subtitle="One logical row. Write to it and the old version stays; open a reader and the collector loses the right to remove anything it might still see. A version is garbage exactly when its box ends to the left of the horizon line."
      controls={
        <>
          <Segmented
            label="Engine"
            value={engineId}
            onChange={setEngineId}
            options={[
              { value: 'pg', label: 'Postgres', title: 'Old versions stay in the heap; VACUUM reclaims them' },
              { value: 'innodb', label: 'InnoDB', title: 'Old versions live in undo logs; purge threads reclaim them' },
              { value: 'lsm', label: 'LSM', title: 'Old versions are older entries for the key; compaction drops them' },
            ]}
          />
          <Button onClick={() => run('update')} disabled={atLimit || s.deleted} primary>
            UPDATE
          </Button>
          <Button onClick={() => run('delete')} disabled={atLimit || s.deleted}>
            DELETE
          </Button>
          <Button onClick={() => run('insert')} disabled={atLimit || !s.deleted}>
            INSERT
          </Button>
          <Button onClick={() => run('open')} disabled={s.readers.length >= 4 || s.next - domainStart >= MAX_SPAN}>
            Open reader
          </Button>
          <Button onClick={() => run('close')} disabled={s.readers.length === 0}>
            Close oldest reader
          </Button>
          <Slider
            label="Oldest reader's snapshot"
            min={domainStart}
            max={s.next}
            value={oldest ? oldest.snap : s.next}
            disabled={!oldest}
            onChange={(n) =>
              setS((cur) => {
                if (cur.readers.length === 0) return cur;
                const o = cur.readers.reduce((a, b) => (a.snap <= b.snap ? a : b));
                const readers = cur.readers.map((r) => (r.rid === o.rid ? { ...r, snap: n } : r));
                const nh = Math.min(...readers.map((r) => r.snap));
                return {
                  ...cur,
                  readers,
                  head: `R${o.rid} re-reads with a snapshot at ${n}; the horizon is now ${nh}.`,
                  body:
                    'Dragging the oldest reader forward is what finishing a long query does to the horizon: ' +
                    'the backlog in front of it drains without anything being rewritten.',
                };
              })
            }
            format={(n) => `${e.unit} ${n}`}
          />
          <Button onClick={() => run('reclaim')} disabled={s.versions.every((v) => v.xmax === null)}>
            Run {e.reclaim}
          </Button>
          <Button onClick={() => setS(INITIAL)}>Reset</Button>
        </>
      }
      legend={
        <Legend
          items={[
            { label: '● current version', color: STATE_STYLE.current.color },
            { label: '◆ dead, a reader still sees it', color: STATE_STYLE.pinned.color },
            { label: '▲ dead to everyone, held by the horizon', color: STATE_STYLE.retained.color },
            { label: '✕ collectable', color: STATE_STYLE.collectable.color },
            { label: 'reader snapshot', color: 'var(--viz-7)', shape: 'line' },
            { label: `garbage horizon (${e.horizon})`, color: 'var(--viz-critical)', shape: 'line' },
          ]}
        />
      }
      stats={
        <Stats
          items={[
            { label: `Next ${e.unit}`, value: s.next, hint: 'The id the next write will take' },
            {
              label: `Horizon (${e.horizon})`,
              value: h,
              hint: 'The oldest open snapshot — or the next id if no reader is open',
            },
            { label: 'Versions of the row', value: rows.length, hint: 'Physical versions currently stored' },
            {
              label: 'Dead, not yet removable',
              value: nPinned + nRetained,
              hint: `${nPinned} a reader can still see, ${nRetained} nobody can see but the horizon protects anyway`,
            },
            { label: `Collectable by ${e.reclaim}`, value: nCollect, hint: 'Ended strictly before the horizon' },
            {
              label: 'Dead space held',
              value: fmtBytes((nPinned + nRetained) * BYTES_PER_VERSION),
              hint: `Modelled at ${BYTES_PER_VERSION} B per row version — the real number scales with row width`,
            },
            { label: 'Reclaimed so far', value: fmtNum(s.reclaimed), hint: 'Versions the collector has taken' },
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
                <th>Version</th>
                <th>Stored as</th>
                <th>ctid</th>
                <th>xmin (created by)</th>
                <th>xmax (ended by)</th>
                <th>Visible to</th>
                <th>State</th>
              </tr>
            </thead>
            <tbody>
              {rows.map(({ v, state }) => (
                <tr key={v.vid}>
                  <td>v{v.vid}</td>
                  <td>{state === 'current' ? e.liveNoun : e.deadNoun}</td>
                  <td>
                    ({ctid(v.vid).page},{ctid(v.vid).off})
                  </td>
                  <td>{v.xmin}</td>
                  <td>{v.xmax === null ? '—' : `${v.xmax}${v.tomb ? ' (DELETE)' : ''}`}</td>
                  <td>
                    {s.readers
                      .filter((r) => visibleTo(v, r.snap))
                      .map((r) => `R${r.rid}`)
                      .join(', ') || (v.xmax === null ? 'new transactions' : 'nobody')}
                  </td>
                  <td>{STATE_STYLE[state].word}</td>
                </tr>
              ))}
            </tbody>
          </table>
          <table className="viz-table">
            <thead>
              <tr>
                <th>Reader</th>
                <th>Snapshot</th>
                <th>Sees version</th>
                <th>Sets the horizon?</th>
              </tr>
            </thead>
            <tbody>
              {s.readers.length === 0 ? (
                <tr>
                  <td colSpan={4}>
                    No reader open — the horizon is the next {e.unit} ({s.next}), so every dead version is
                    collectable.
                  </td>
                </tr>
              ) : (
                s.readers.map((r) => {
                  const seen = rows.find(({ v }) => visibleTo(v, r.snap));
                  return (
                    <tr key={r.rid}>
                      <td>R{r.rid}</td>
                      <td>{r.snap}</td>
                      <td>{seen ? `v${seen.v.vid}` : 'row not visible (deleted or not yet inserted)'}</td>
                      <td>{oldest && r.rid === oldest.rid ? 'yes' : 'no'}</td>
                    </tr>
                  );
                })
              )}
            </tbody>
          </table>
        </>
      }
    >
      <div ref={ref}>
        <TooltipHost>
          <svg
            width={svgW}
            height={height}
            role="img"
            aria-label={`Version chain of one row along a ${e.unit} axis, with reader snapshots and the garbage horizon`}
          >
            {/* axis */}
            {Array.from({ length: span }, (_, i) => domainStart + i).map((x) => (
              <g key={`ax${x}`}>
                <line
                  x1={lineX(x)}
                  x2={lineX(x)}
                  y1={axisH - 8}
                  y2={height - 8}
                  stroke="var(--viz-grid)"
                  strokeWidth={1}
                />
                <text x={midX(x)} y={axisH - 14} textAnchor="middle" fill="var(--viz-ink-muted)">
                  {x <= s.next ? x : ''}
                </text>
              </g>
            ))}
            <text x={0} y={axisH - 14} fill="var(--viz-ink-2)" fontWeight={600}>
              {e.unit} →
            </text>

            {/* version rows */}
            {rows.map(({ v, state }, i) => {
              const y = versTop + i * rowH;
              const x0 = midX(v.xmin);
              const x1 = v.xmax === null ? rightEdge : midX(v.xmax);
              const st = STATE_STYLE[state];
              const c = ctid(v.vid);
              return (
                <g key={v.vid}>
                  <text x={0} y={y + 14} fill="var(--viz-ink)" fontWeight={600}>
                    v{v.vid}
                  </text>
                  <text x={0} y={y + 30} fill="var(--viz-ink-muted)">
                    {state === 'current' ? e.liveNoun : e.deadNoun} ({c.page},{c.off})
                  </text>
                  <g
                    {...tip(
                      <>
                        <strong>
                          v{v.vid} — {st.word}
                        </strong>
                        <br />
                        xmin {v.xmin}, xmax {v.xmax === null ? 'none (current)' : v.xmax}
                        <br />
                        {state === 'current'
                          ? 'Every new snapshot reads this one.'
                          : state === 'pinned'
                            ? `A reader's snapshot falls inside (${v.xmin}, ${v.xmax}] — removing it would break that read.`
                            : state === 'retained'
                              ? `No open snapshot falls inside (${v.xmin}, ${v.xmax}], but xmax ${v.xmax} is not below the horizon ${h}, and the collector only compares against that one number.`
                              : `xmax ${v.xmax} < horizon ${h}: no current or future snapshot can reach it.`}
                      </>,
                    )}
                    style={{ cursor: 'help' }}
                  >
                    <rect
                      x={x0}
                      y={y}
                      width={Math.max(10, x1 - x0)}
                      height={20}
                      rx={5}
                      fill={st.color}
                      stroke="var(--viz-surface)"
                      strokeWidth={1.5}
                      opacity={state === 'collectable' ? 0.55 : 1}
                    />
                    <text x={x0 + 7} y={y + 15} fill="var(--viz-surface)" fontWeight={700}>
                      {st.glyph}
                    </text>
                  </g>
                  <text x={x0 + 2} y={y + 36} fill="var(--viz-ink-2)">
                    xmin {v.xmin} · xmax {v.xmax === null ? '—' : v.xmax} · {st.word}
                    {v.tomb ? ' (DELETE)' : ''}
                  </text>
                </g>
              );
            })}

            {/* reader lanes */}
            {s.readers.length === 0 ? (
              <text x={0} y={readTop + 16} fill="var(--viz-ink-muted)">
                no reader open — horizon = next {e.unit}
              </text>
            ) : (
              s.readers.map((r, i) => {
                const y = readTop + i * readerH;
                const isOldest = oldest !== null && r.rid === oldest.rid;
                const seen = rows.find(({ v }) => visibleTo(v, r.snap));
                return (
                  <g key={r.rid}>
                    <text x={0} y={y + 14} fill="var(--viz-ink)" fontWeight={600}>
                      R{r.rid} {isOldest ? '(oldest)' : ''}
                    </text>
                    <text x={0} y={y + 30} fill="var(--viz-ink-muted)">
                      snapshot @ {r.snap}
                    </text>
                    <g
                      {...tip(
                        <>
                          <strong>
                            R{r.rid} — snapshot at {r.snap}
                          </strong>
                          <br />
                          Sees every {e.unit} below {r.snap} as finished, so it reads{' '}
                          {seen ? `v${seen.v.vid}` : 'no version of this row'}.
                          {isOldest ? ' It is the oldest open reader, so it sets the horizon.' : ''}
                        </>,
                      )}
                      style={{ cursor: 'help' }}
                    >
                      <rect
                        x={lineX(r.snap)}
                        y={y + 4}
                        width={Math.max(10, rightEdge - lineX(r.snap))}
                        height={10}
                        rx={4}
                        fill="var(--viz-7)"
                        opacity={isOldest ? 1 : 0.55}
                      />
                    </g>
                    <text x={lineX(r.snap) + 4} y={y + 32} fill="var(--viz-ink-2)">
                      reads {seen ? `v${seen.v.vid}` : 'nothing'}
                    </text>
                  </g>
                );
              })
            )}

            {/* snapshot lines */}
            {s.readers.map((r) => (
              <line
                key={`sn${r.rid}`}
                x1={lineX(r.snap)}
                x2={lineX(r.snap)}
                y1={axisH - 8}
                y2={height - 8}
                stroke="var(--viz-7)"
                strokeWidth={1.5}
                strokeDasharray="4 3"
              />
            ))}

            {/* the horizon */}
            <line
              x1={lineX(h)}
              x2={lineX(h)}
              y1={axisH - 10}
              y2={height - 6}
              stroke="var(--viz-critical)"
              strokeWidth={2.5}
            />
            <text x={lineX(h) + 6} y={height - 10} fill="var(--viz-critical)" fontWeight={600}>
              horizon {h}
            </text>
          </svg>
        </TooltipHost>
      </div>
    </VizPanel>
  );
}
