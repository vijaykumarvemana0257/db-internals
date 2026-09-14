import { useState } from 'react';
import {
  VizPanel,
  Choice,
  Segmented,
  Slider,
  Legend,
  Stats,
  Note,
  TooltipHost,
  useTip,
  fmtNum,
  useSize,
} from './Viz';

/**
 * Instants, civil time and the two DST discontinuities.
 *
 * One event is typed as a *civil* wall-clock time in a venue zone (America/Los_Angeles)
 * and then written four ways: as timestamptz (an instant: microseconds since the
 * Postgres epoch 2000-01-01 UTC), as timestamp without time zone (the wall-clock string,
 * no zone at all), as a bigint of epoch milliseconds, and as the pair
 * (civil timestamp, IANA zone id). The session TimeZone then decides only how each one
 * is *rendered* — and where the day boundary that date_trunc('day') and a BETWEEN range
 * land on the UTC axis.
 *
 * All time arithmetic here is hand-rolled against fixed 2026 US transition instants so
 * the figure is identical on the server, in the browser, and on a host whose tzdata is
 * three years stale. Date.UTC is used only as a pure civil→epoch function.
 */

const HOUR = 3600;

/* -------------------------------------------------------------- zone model */

/** Real 2026 US transitions for America/Los_Angeles, as UTC instants. */
const LA_SPRING = Date.UTC(2026, 2, 8, 10) / 1000; // 02:00 PST -> 03:00 PDT
const LA_FALL = Date.UTC(2026, 10, 1, 9) / 1000; // 02:00 PDT -> 01:00 PST

type Zone = {
  id: string;
  short: string;
  offsetAt: (t: number) => number; // seconds east of UTC
  abbrevAt: (t: number) => string;
};

const LA: Zone = {
  id: 'America/Los_Angeles',
  short: 'LA',
  offsetAt: (t) => (t >= LA_SPRING && t < LA_FALL ? -7 * HOUR : -8 * HOUR),
  abbrevAt: (t) => (t >= LA_SPRING && t < LA_FALL ? 'PDT' : 'PST'),
};

const ZONES: Zone[] = [
  { id: 'UTC', short: 'UTC', offsetAt: () => 0, abbrevAt: () => 'UTC' },
  LA,
  { id: 'Asia/Kolkata', short: 'Kolkata', offsetAt: () => 5.5 * HOUR, abbrevAt: () => 'IST' },
];

type ZoneId = 'UTC' | 'America/Los_Angeles' | 'Asia/Kolkata';
const zoneById = (id: ZoneId) => ZONES.find((z) => z.id === id)!;

/* ------------------------------------------------------------ civil <-> utc */

type Civil = { y: number; mo: number; d: number; h: number; mi: number };

const civilToNaive = (c: Civil) => Date.UTC(c.y, c.mo - 1, c.d, c.h, c.mi) / 1000;

function toCivil(t: number, offset: number): Civil {
  const d = new Date((t + offset) * 1000);
  return {
    y: d.getUTCFullYear(),
    mo: d.getUTCMonth() + 1,
    d: d.getUTCDate(),
    h: d.getUTCHours(),
    mi: d.getUTCMinutes(),
  };
}

const p2 = (n: number) => String(n).padStart(2, '0');
const dateStr = (c: Civil) => `${c.y}-${p2(c.mo)}-${p2(c.d)}`;
const civilStr = (c: Civil) => `${dateStr(c)} ${p2(c.h)}:${p2(c.mi)}:00`;
const hhmm = (c: Civil) => `${p2(c.h)}:${p2(c.mi)}`;

function offsetStr(off: number) {
  const sign = off < 0 ? '-' : '+';
  const a = Math.abs(off);
  const h = Math.floor(a / HOUR);
  const m = Math.round((a % HOUR) / 60);
  return m === 0 ? `${sign}${p2(h)}` : `${sign}${p2(h)}:${p2(m)}`;
}

/** Render an instant the way a client would print it under a session TimeZone. */
function renderTz(t: number, z: Zone) {
  const off = z.offsetAt(t);
  return `${civilStr(toCivil(t, off))}${offsetStr(off)}`;
}

/**
 * Every instant a civil time in `z` could mean. Zero candidates = the local time does
 * not exist (spring-forward gap); two = it happened twice (fall-back overlap).
 */
function candidates(c: Civil, z: Zone) {
  const naive = civilToNaive(c);
  const out: { off: number; t: number; abbrev: string }[] = [];
  for (const off of [-8 * HOUR, -7 * HOUR, 0, 5.5 * HOUR]) {
    const t = naive - off;
    if (z.offsetAt(t) === off && !out.some((o) => o.t === t)) {
      out.push({ off, t, abbrev: z.abbrevAt(t) });
    }
  }
  out.sort((a, b) => a.t - b.t);
  return out;
}

/**
 * Turn a civil time into an instant the way a driver must: pick an offset. `rule` is
 * which side of the transition the offset comes from — the choice every stack makes
 * silently, and the only reason a gap or an overlap does not raise an error.
 */
function toInstant(c: Civil, z: Zone, rule: 'before' | 'after') {
  const cands = candidates(c, z);
  if (cands.length === 1) return { t: cands[0].t, kind: 'valid' as const, cands };
  const naive = civilToNaive(c);
  if (cands.length === 0) {
    // Gap: neither offset is self-consistent. Applying the pre-transition offset
    // shifts the event forward past the gap; the post-transition offset pulls it back.
    const t = rule === 'before' ? naive + 8 * HOUR : naive + 7 * HOUR;
    return { t, kind: 'gap' as const, cands };
  }
  return { t: rule === 'before' ? cands[0].t : cands[1].t, kind: 'ambiguous' as const, cands };
}

/** Local midnight of a civil date in `z`, and the local midnight after it. */
function dayBounds(date: Civil, z: Zone, rule: 'before' | 'after') {
  const start = toInstant({ ...date, h: 0, mi: 0 }, z, rule).t;
  const nextCivil = toCivil(civilToNaive({ ...date, h: 12, mi: 0 }) + 24 * HOUR, 0);
  const end = toInstant({ ...nextCivil, h: 0, mi: 0 }, z, rule).t;
  return { start, end };
}

/* -------------------------------------------------------------- scenarios */

type ScenId = 'fall' | 'spring' | 'plain';

const SCENARIOS: { id: ScenId; label: string; date: Civil; blurb: string }[] = [
  {
    id: 'fall',
    label: 'Fall back — 2026-11-01',
    date: { y: 2026, mo: 11, d: 1, h: 0, mi: 0 },
    blurb: '01:00–02:00 happens twice in America/Los_Angeles; the local day is 25 hours long.',
  },
  {
    id: 'spring',
    label: 'Spring forward — 2026-03-08',
    date: { y: 2026, mo: 3, d: 8, h: 0, mi: 0 },
    blurb: '02:00–03:00 never happens in America/Los_Angeles; the local day is 23 hours long.',
  },
  {
    id: 'plain',
    label: 'No transition — 2026-06-14',
    date: { y: 2026, mo: 6, d: 14, h: 0, mi: 0 },
    blurb: 'A 24-hour local day: every wall-clock time means exactly one instant.',
  },
];

/* ------------------------------------------------------------- the component */

const EPOCH_2000 = Date.UTC(2000, 0, 1) / 1000; // the Postgres timestamptz epoch

export default function InstantCivilTimeLab() {
  const [scenId, setScenId] = useState<ScenId>('fall');
  const [mins, setMins] = useState(90); // wall-clock minutes after local midnight, as typed
  const [sessionId, setSessionId] = useState<ZoneId>('UTC');
  const [rule, setRule] = useState<'before' | 'after'>('before');
  const [ref, width] = useSize(880);
  const tip = useTip();

  const scen = SCENARIOS.find((s) => s.id === scenId)!;
  const session = zoneById(sessionId);

  const typed: Civil = { ...scen.date, h: Math.floor(mins / 60), mi: mins % 60 };
  const ev = toInstant(typed, LA, rule);
  const laDay = dayBounds(scen.date, LA, rule);
  const dayHours = (laDay.end - laDay.start) / HOUR;

  // The query's day, as the session zone defines it: DATE '2026-11-01' cast to
  // timestamptz is midnight *in the session TimeZone*, not in the venue's.
  const qDay = dayBounds(scen.date, session, rule);
  const bands = [
    { key: 'h24', label: "ts >= D AND ts < D + interval '24 hours'", a: qDay.start, b: qDay.start + 24 * HOUR, closed: false },
    { key: 'd1', label: "ts >= D AND ts < D + interval '1 day'", a: qDay.start, b: qDay.end, closed: false },
    { key: 'btw', label: "ts BETWEEN D AND D + interval '1 day'", a: qDay.start, b: qDay.end, closed: true },
  ];
  const inBand = (b: (typeof bands)[number]) => ev.t >= b.a && (b.closed ? ev.t <= b.b : ev.t < b.b);

  // date_trunc('day', ts) in the session zone
  const bucket = toCivil(ev.t, session.offsetAt(ev.t));
  const bucketStart = toInstant({ ...bucket, h: 0, mi: 0 }, session, rule).t;

  const micros2000 = (ev.t - EPOCH_2000) * 1e6;
  const millis = ev.t * 1000;
  const msAsMicros = toCivil(Math.floor(millis / 1e6), 0);

  /* -------------------------------------------------------------- geometry */

  const padL = 172;
  const padR = 18;
  const svgW = Math.max(width, 880);
  const t0 = laDay.start - 2 * HOUR;
  const t1 = laDay.end + 2 * HOUR;
  const ax0 = padL;
  const ax1 = svgW - padR;
  const x = (t: number) => ax0 + ((t - t0) / (t1 - t0)) * (ax1 - ax0);
  /** Clamped: a day boundary in a distant session zone can sit off the axis entirely. */
  const xc = (t: number) => Math.min(ax1, Math.max(ax0, x(t)));
  const onAxis = (t: number) => t >= t0 && t <= t1;

  const yUtcAxis = 44;
  const yLocal = 76;
  const localH = 22;
  const yLocal2 = yLocal + localH + 4;
  const yBands = 142;
  const bandH = 15;
  const bandGap = 8;
  const yBucket = yBands + bands.length * (bandH + bandGap) + 10;
  const timelineBottom = yBucket + 20;

  const matrixTop = timelineBottom + 34;
  const rowH = 27;
  const colW = (svgW - padL - padR) / 3;

  const ROWS = [
    {
      key: 'tstz',
      name: 'event_at timestamptz',
      sub: 'int64 µs since 2000-01-01 UTC',
      cell: (z: Zone) => renderTz(ev.t, z),
      same: false,
    },
    {
      key: 'ts',
      name: 'event_at timestamp',
      sub: 'wall clock, no zone stored',
      cell: () => civilStr(typed),
      same: true,
    },
    {
      key: 'ms',
      name: 'event_ms bigint',
      sub: 'epoch milliseconds',
      cell: () => fmtNum(millis),
      same: true,
    },
    {
      key: 'civil',
      name: '(event_local, zone)',
      sub: 'civil time + IANA id',
      cell: () => `${civilStr(typed)} @ LA`,
      same: true,
    },
  ];

  // UTC hour ticks every two hours
  const ticks: number[] = [];
  for (let t = Math.ceil(t0 / (2 * HOUR)) * 2 * HOUR; t <= t1; t += 2 * HOUR) ticks.push(t);

  // Local-hour ticks on the venue ruler: walk instants, label with the LA wall clock.
  const localTicks: { t: number; c: Civil; second: boolean }[] = [];
  for (let t = Math.ceil(t0 / HOUR) * HOUR; t <= t1; t += HOUR) {
    const off = LA.offsetAt(t);
    const c = toCivil(t, off);
    const second = scenId === 'fall' && t >= LA_FALL && t < LA_FALL + HOUR;
    localTicks.push({ t, c, second });
  }

  const stateColor =
    ev.kind === 'valid' ? 'var(--viz-good)' : ev.kind === 'ambiguous' ? 'var(--viz-warning)' : 'var(--viz-critical)';
  const BAND_COLORS = ['var(--viz-1)', 'var(--viz-2)', 'var(--viz-3)'];

  const other = ev.kind === 'ambiguous' ? ev.cands.find((c) => c.t !== ev.t) : undefined;

  const head =
    ev.kind === 'valid'
      ? `${civilStr(typed)} America/Los_Angeles is one instant.`
      : ev.kind === 'ambiguous'
        ? `${hhmm(typed)} America/Los_Angeles happens twice on this date.`
        : `${hhmm(typed)} America/Los_Angeles never happens on this date.`;

  const body =
    ev.kind === 'valid'
      ? `The wall clock maps to exactly one UTC instant, ${renderTz(ev.t, ZONES[0])}. timestamptz stores that ` +
        `instant and forgets everything else — the zone you typed is not in the column. Every rendering below is ` +
        `the same eight bytes read through a different session TimeZone.`
      : ev.kind === 'ambiguous'
        ? `The clocks went back at 02:00 PDT, so ${hhmm(typed)} occurs once at ${offsetStr(ev.cands[0].off)} (${ev.cands[0].abbrev}) ` +
          `and again an hour later at ${offsetStr(ev.cands[1].off)} (${ev.cands[1].abbrev}). No error is raised anywhere: the ` +
          `driver silently picks one. With the ${rule === 'before' ? 'pre' : 'post'}-transition offset it stored ` +
          `${renderTz(ev.t, ZONES[0])}; the other reading is ${other ? renderTz(other.t, ZONES[0]) : '—'}, one hour away.`
        : `The clocks jumped 02:00 PST straight to 03:00 PDT, so no instant renders as ${hhmm(typed)} here. Postgres does ` +
          `not reject it — it applies an offset anyway. With the ${rule === 'before' ? 'pre' : 'post'}-transition offset ` +
          `the stored instant is ${renderTz(ev.t, ZONES[0])}, which reads back in LA as ${hhmm(toCivil(ev.t, LA.offsetAt(ev.t)))} ` +
          `— an hour from what the user typed, and the row now says something they never entered.`;

  return (
    <VizPanel
      title="One moment, four columns, three session zones"
      subtitle={`Type a wall-clock time in America/Los_Angeles and watch it become an instant — or fail to. ${scen.blurb} The UTC axis never moves; the local ruler, the day boundary and the range predicates all do.`}
      controls={
        <>
          <Choice
            label="Date"
            value={scenId}
            onChange={(v) => setScenId(v)}
            options={SCENARIOS.map((s) => ({ value: s.id, label: s.label }))}
          />
          <Slider
            label="Event local time (typed)"
            min={0}
            max={1425}
            step={15}
            value={mins}
            onChange={setMins}
            format={(n) => `${p2(Math.floor(n / 60))}:${p2(n % 60)}`}
          />
          <Choice
            label="SET TimeZone"
            value={sessionId}
            onChange={(v) => setSessionId(v)}
            options={ZONES.map((z) => ({ value: z.id as ZoneId, label: z.id }))}
          />
          <Segmented
            label="Offset for a gap/overlap"
            value={rule}
            onChange={setRule}
            options={[
              { value: 'before', label: 'pre-transition', title: 'The offset in effect before the transition — what Postgres and java.time do' },
              { value: 'after', label: 'post-transition', title: 'The offset in effect after the transition — Python fold=1 for an overlap' },
            ]}
          />
        </>
      }
      legend={
        <Legend
          items={[
            { label: "ts >= D AND ts < D + interval '24 hours'", color: BAND_COLORS[0] },
            { label: "ts >= D AND ts < D + interval '1 day' (calendar day)", color: BAND_COLORS[1] },
            { label: 'ts BETWEEN D AND D + 1 day (closed upper bound)', color: BAND_COLORS[2] },
            { label: '▼ stored instant', color: stateColor, shape: 'dot' },
            { label: '▽ the offset not chosen', color: 'var(--viz-stale)', shape: 'dot' },
            { label: 'repeated / missing local hour', color: 'var(--viz-warning)' },
          ]}
        />
      }
      stats={
        <Stats
          items={[
            {
              label: 'Local time is',
              value: ev.kind === 'valid' ? 'unambiguous' : ev.kind === 'ambiguous' ? 'ambiguous ×2' : 'nonexistent',
              hint: 'How many instants this wall-clock string can mean in America/Los_Angeles',
            },
            { label: 'Stored instant (UTC)', value: renderTz(ev.t, ZONES[0]), hint: 'What timestamptz actually holds' },
            {
              label: 'On disk (µs since 2000-01-01)',
              value: fmtNum(micros2000),
              hint: 'A timestamptz is an 8-byte signed integer of microseconds from the Postgres epoch — no zone, no offset',
            },
            {
              label: `Renders in ${session.short}`,
              value: renderTz(ev.t, session),
              hint: 'The session TimeZone GUC changes only this string',
            },
            {
              label: "date_trunc('day') bucket",
              value: `${dateStr(bucket)} ${session.short}`,
              hint: `Bucket starts at ${renderTz(bucketStart, ZONES[0])} — the boundary moves with the session zone`,
            },
            {
              label: 'Local day length',
              value: `${dayHours} h`,
              hint: "Why + interval '1 day' and + interval '24 hours' are different operations on timestamptz",
            },
            {
              label: 'ms read as µs',
              value: dateStr(msAsMicros),
              hint: 'The same bigint decoded with the wrong scale — epoch millis interpreted as microseconds',
            },
          ]}
        />
      }
      note={
        <Note>
          <strong>{head}</strong> {body}
        </Note>
      }
      table={
        <>
          <table className="viz-table">
            <thead>
              <tr>
                <th>Column</th>
                {ZONES.map((z) => (
                  <th key={z.id}>SET TimeZone = {z.id}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {ROWS.map((r) => (
                <tr key={r.key}>
                  <td>{r.name}</td>
                  {ZONES.map((z) => (
                    <td key={z.id}>{r.cell(z)}</td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
          <table className="viz-table">
            <thead>
              <tr>
                <th>Predicate (D = DATE '{dateStr(scen.date)}' in the session zone)</th>
                <th>From (UTC)</th>
                <th>To (UTC)</th>
                <th>Width</th>
                <th>Row matched?</th>
              </tr>
            </thead>
            <tbody>
              {bands.map((b) => (
                <tr key={b.key}>
                  <td>{b.label}</td>
                  <td>{renderTz(b.a, ZONES[0])}</td>
                  <td>
                    {renderTz(b.b, ZONES[0])} {b.closed ? '(inclusive)' : '(exclusive)'}
                  </td>
                  <td>{(b.b - b.a) / HOUR} h</td>
                  <td>{inBand(b) ? 'yes' : 'no'}</td>
                </tr>
              ))}
            </tbody>
          </table>
          <table className="viz-table">
            <thead>
              <tr>
                <th>Candidate offset for {civilStr(typed)} America/Los_Angeles</th>
                <th>Instant (UTC)</th>
                <th>Reads back as</th>
              </tr>
            </thead>
            <tbody>
              {ev.cands.length === 0 ? (
                <tr>
                  <td colSpan={3}>none — this local time does not exist in this zone</td>
                </tr>
              ) : (
                ev.cands.map((c) => (
                  <tr key={c.off}>
                    <td>
                      {offsetStr(c.off)} ({c.abbrev})
                    </td>
                    <td>{renderTz(c.t, ZONES[0])}</td>
                    <td>{renderTz(c.t, LA)}</td>
                  </tr>
                ))
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
            height={matrixTop + ROWS.length * rowH + 16}
            role="img"
            aria-label="A civil time mapped onto the UTC axis, with the DST transition, day-boundary and range predicates drawn against it"
          >
            {/* ------------------------------------------------ UTC instant axis */}
            <text x={0} y={yUtcAxis - 12} fill="var(--viz-ink)" fontWeight={600}>
              UTC instant axis
            </text>
            <text x={0} y={yUtcAxis + 4} fill="var(--viz-ink-muted)">
              monotonic, no gaps, no repeats
            </text>
            <line className="viz-axis-line" x1={padL} x2={svgW - padR} y1={yUtcAxis} y2={yUtcAxis} />
            {ticks.map((t) => (
              <g key={t}>
                <line className="viz-axis-line" x1={x(t)} x2={x(t)} y1={yUtcAxis} y2={yUtcAxis + 5} />
                <text x={x(t)} y={yUtcAxis + 18} textAnchor="middle">
                  {hhmm(toCivil(t, 0))}Z
                </text>
              </g>
            ))}

            {/* --------------------------------------------- venue local ruler */}
            <text x={0} y={yLocal + 14} fill="var(--viz-ink)" fontWeight={600}>
              America/Los_Angeles
            </text>
            <text x={0} y={yLocal + 30} fill="var(--viz-ink-muted)">
              the wall clock people read
            </text>
            <rect
              x={padL}
              y={yLocal}
              width={svgW - padL - padR}
              height={localH}
              rx={5}
              fill="var(--viz-plane)"
              stroke="var(--viz-border)"
            />
            {scenId === 'fall' ? (
              <>
                <rect x={x(LA_FALL - HOUR)} y={yLocal} width={x(LA_FALL) - x(LA_FALL - HOUR)} height={localH} fill="var(--viz-warning)" opacity={0.3} />
                <rect
                  x={x(LA_FALL)}
                  y={yLocal2}
                  width={x(LA_FALL + HOUR) - x(LA_FALL)}
                  height={localH}
                  rx={5}
                  fill="var(--viz-warning)"
                  opacity={0.3}
                  stroke="var(--viz-warning)"
                  strokeDasharray="4 3"
                />
                <text x={x(LA_FALL + HOUR) + 6} y={yLocal2 + 15} fill="var(--viz-ink)">
                  01:00–02:00 happens a second time, now at -08 (PST)
                </text>
              </>
            ) : null}
            {scenId === 'spring' ? (
              <>
                <line x1={x(LA_SPRING)} x2={x(LA_SPRING)} y1={yLocal - 6} y2={yLocal + localH + 6} stroke="var(--viz-critical)" strokeWidth={2} />
                <text x={x(LA_SPRING) + 6} y={yLocal2 + 15} fill="var(--viz-critical)">
                  02:00 → 03:00: the hour 02:00–02:59 has no instant
                </text>
              </>
            ) : null}
            {localTicks.map((lt) => (
              <g key={`${lt.t}-${lt.second ? 'b' : 'a'}`}>
                <line
                  className="viz-grid-line"
                  x1={x(lt.t)}
                  x2={x(lt.t)}
                  y1={lt.second ? yLocal2 : yLocal}
                  y2={(lt.second ? yLocal2 : yLocal) + localH}
                />
                {lt.c.h % 2 === 0 ? (
                  <text x={x(lt.t) + 3} y={(lt.second ? yLocal2 : yLocal) + 15}>
                    {hhmm(lt.c)}
                  </text>
                ) : null}
              </g>
            ))}

            {/* ------------------------------------------------- range predicates */}
            {bands.map((b, i) => {
              const y = yBands + i * (bandH + bandGap);
              const hit = inBand(b);
              return (
                <g key={b.key} {...tip(
                  <>
                    <strong>{b.label}</strong>
                    <br />
                    {renderTz(b.a, ZONES[0])} → {renderTz(b.b, ZONES[0])} ({(b.b - b.a) / HOUR} h,{' '}
                    {b.closed ? 'inclusive' : 'exclusive'} upper bound)
                    <br />
                    This event: {hit ? 'matched' : 'missed'}
                  </>,
                )}>
                  <rect
                    x={xc(b.a)}
                    y={y}
                    width={Math.max(2, xc(b.b) - xc(b.a))}
                    height={bandH}
                    rx={3}
                    fill={BAND_COLORS[i]}
                    opacity={hit ? 0.85 : 0.3}
                    stroke={BAND_COLORS[i]}
                    strokeDasharray={hit ? undefined : '4 3'}
                  />
                  <text x={0} y={y + 12} fill="var(--viz-ink-2)">
                    {i === 0 ? "+ interval '24 hours'" : i === 1 ? "+ interval '1 day'" : 'BETWEEN … AND'}
                  </text>
                  {b.closed && onAxis(b.b) ? <circle cx={xc(b.b)} cy={y + bandH / 2} r={3.5} fill={BAND_COLORS[i]} /> : null}
                  <text
                    x={Math.min(xc(b.b) + 7, ax1 - 66)}
                    y={y + 12}
                    fill={hit ? 'var(--viz-good)' : 'var(--viz-critical)'}
                  >
                    {hit ? 'row matched' : 'row missed'}
                  </text>
                </g>
              );
            })}

            {/* ------------------------------------- date_trunc bucket boundaries */}
            <text x={0} y={yBucket + 12} fill="var(--viz-ink-2)">
              date_trunc('day') in {session.short}
            </text>
            {[qDay.start, qDay.end].map((t) =>
              onAxis(t) ? (
                <g key={`b${t}`}>
                  <line
                    x1={x(t)}
                    x2={x(t)}
                    y1={yUtcAxis}
                    y2={yBucket + 14}
                    stroke="var(--viz-ink-2)"
                    strokeWidth={1}
                    strokeDasharray="3 4"
                  />
                  <text x={x(t) + 4} y={yBucket + 12}>
                    {dateStr(toCivil(t, session.offsetAt(t)))} 00:00 {session.short}
                  </text>
                </g>
              ) : (
                <text key={`b${t}`} x={t < t0 ? padL + 4 : ax1 - 4} y={yBucket + 12} textAnchor={t < t0 ? 'start' : 'end'}>
                  {t < t0 ? '◀ ' : ''}
                  {dateStr(toCivil(t, session.offsetAt(t)))} 00:00 {session.short} = {hhmm(toCivil(t, 0))}Z, off this axis
                  {t < t0 ? '' : ' ▶'}
                </text>
              ),
            )}

            {/* --------------------------------------------------- event markers */}
            {other ? (
              <g {...tip(<>The instant the other offset would have stored: {renderTz(other.t, ZONES[0])}</>)}>
                <line x1={x(other.t)} x2={x(other.t)} y1={yUtcAxis - 8} y2={timelineBottom} stroke="var(--viz-stale)" strokeDasharray="3 3" />
                <path
                  d={`M ${x(other.t) - 6} ${yUtcAxis - 18} L ${x(other.t) + 6} ${yUtcAxis - 18} L ${x(other.t)} ${yUtcAxis - 8} Z`}
                  fill="var(--viz-surface)"
                  stroke="var(--viz-stale)"
                />
              </g>
            ) : null}
            <g
              {...tip(
                <>
                  <strong>{civilStr(typed)} as typed</strong>
                  <br />
                  stored as {renderTz(ev.t, ZONES[0])}
                  <br />
                  reads back in LA as {renderTz(ev.t, LA)}
                </>,
              )}
            >
              <line x1={x(ev.t)} x2={x(ev.t)} y1={yUtcAxis - 8} y2={timelineBottom} stroke={stateColor} strokeWidth={2} />
              <path
                d={`M ${x(ev.t) - 7} ${yUtcAxis - 20} L ${x(ev.t) + 7} ${yUtcAxis - 20} L ${x(ev.t)} ${yUtcAxis - 8} Z`}
                fill={stateColor}
              />
              <text x={x(ev.t) + 10} y={yUtcAxis - 20} fill={stateColor}>
                {ev.kind === 'gap' ? 'shifted' : ev.kind === 'ambiguous' ? 'one of two' : 'stored'}
              </text>
            </g>

            {/* ------------------------------------------------ rendering matrix */}
            <line className="viz-axis-line" x1={0} x2={svgW - padR} y1={matrixTop - 24} y2={matrixTop - 24} />
            <text x={0} y={matrixTop - 8} fill="var(--viz-ink)" fontWeight={600}>
              Same row, read under
            </text>
            {ZONES.map((z, i) => (
              <text key={z.id} x={padL + i * colW + 6} y={matrixTop - 8} fill="var(--viz-ink)" fontWeight={600}>
                SET TimeZone = {z.id}
              </text>
            ))}
            {ROWS.map((r, ri) => {
              const y = matrixTop + ri * rowH;
              return (
                <g key={r.key}>
                  <rect x={0} y={y} width={svgW - padR} height={rowH - 4} rx={5} fill={ri % 2 ? 'var(--viz-plane)' : 'transparent'} />
                  <text x={0} y={y + 12} fill="var(--viz-ink)">
                    {r.name}
                  </text>
                  <text x={0} y={y + 23} fill="var(--viz-ink-muted)">
                    {r.sub}
                  </text>
                  {ZONES.map((z, ci) => (
                    <text
                      key={z.id}
                      x={padL + ci * colW + 6}
                      y={y + 17}
                      fill={r.same || z.id === session.id ? 'var(--viz-ink)' : 'var(--viz-ink-2)'}
                      fontWeight={z.id === session.id ? 600 : 400}
                    >
                      {r.cell(z)}
                    </text>
                  ))}
                </g>
              );
            })}
          </svg>
        </TooltipHost>
      </div>
    </VizPanel>
  );
}
