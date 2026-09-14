import { useMemo, useState } from 'react';
import { VizPanel, Segmented, Choice, Check, Legend, Stats, Note, fmtNum } from './Viz';
import {
  graph,
  bySlug,
  closure,
  dependents,
  evaluate,
  hours,
  MINUTES_PER_PAGE,
  ROUTES,
  GOALS,
  NAIVE_ROUTE,
  type Route,
} from './curriculumRoutes';

type Audience = 'engineering' | 'roles';

const EDGE = {
  step: 'var(--viz-1)',
  added: 'var(--viz-2)',
  extra: 'var(--viz-3)',
  flagged: 'var(--viz-critical)',
} as const;

const fmtH = (h: number) => (h < 10 ? h.toFixed(1) : fmtNum(h));

export default function RouteMapExplorer({ audience = 'engineering' }: { audience?: Audience }) {
  const presets = useMemo(() => {
    const own = ROUTES.filter((r) => r.audience === audience);
    return audience === 'roles' ? [...own, ...ROUTES.filter((r) => r.audience === 'engineering')] : [...own, NAIVE_ROUTE, ...GOALS];
  }, [audience]);

  const [routeId, setRouteId] = useState(presets[0].id);
  const [secondGoal, setSecondGoal] = useState('none');
  const [unwrittenOnly, setUnwrittenOnly] = useState(false);
  const [focus, setFocus] = useState<string | null>(null);

  const route = presets.find((r) => r.id === routeId)!;
  const base = useMemo(() => evaluate([route]), [route]);
  const withGoal = useMemo(() => {
    const g = GOALS.find((x) => x.id === secondGoal);
    return g ? evaluate([route, g]) : null;
  }, [route, secondGoal]);
  const newlyLit = useMemo(
    () => (withGoal ? new Set([...withGoal.lit].filter((s) => !base.lit.has(s))) : new Set<string>()),
    [withGoal, base],
  );

  // Pages added by each leg = modules first lit by that leg (its steps plus prerequisites not yet lit).
  const legRows = useMemo(() => {
    const seen = new Set<string>();
    let cumulative = 0;
    return route.legs.map((leg) => {
      const lit = [...closure(leg.modules)].filter((s) => !seen.has(s));
      lit.forEach((s) => seen.add(s));
      const pages = lit.reduce((n, s) => n + bySlug.get(s)!.pages, 0);
      const written = lit.reduce((n, s) => n + bySlug.get(s)!.builtPages, 0);
      cumulative += pages;
      return { name: leg.name, modules: lit.length, pages, written, cumulativeHours: hours(cumulative) };
    });
  }, [route]);

  const focusNode = focus ? bySlug.get(focus) : null;
  const focusPrereqs = new Set(focusNode?.prerequisites ?? []);

  const stateOf = (slug: string) => {
    if (base.outOfOrder.has(slug)) return 'flagged';
    if (base.steps.includes(slug)) return 'step';
    if (base.added.has(slug)) return 'added';
    if (newlyLit.has(slug)) return 'extra';
    return 'outside';
  };

  const presetOptions = presets.map((r) => ({ value: r.id, label: r.label }));
  const goalOptions = [{ value: 'none', label: 'None' }, ...GOALS.filter((g) => g.id !== routeId).map((g) => ({ value: g.id, label: g.label }))];

  return (
    <VizPanel
      title={audience === 'roles' ? 'Routes by role' : 'Routes through the curriculum'}
      subtitle={`Pick a route or a goal. Its modules light up, the prerequisites it would otherwise skip light up in a second colour, and totals are computed live from the curriculum's own dependency data. Estimates assume ${MINUTES_PER_PAGE} minutes per page.`}
      controls={
        <>
          <Choice label={audience === 'roles' ? 'Route' : 'Route or goal'} value={routeId} onChange={(v) => { setRouteId(v); setSecondGoal('none'); }} options={presetOptions} />
          <Choice label="Then also learn…" value={secondGoal} onChange={setSecondGoal} options={goalOptions} />
          <Check label="Highlight pages not written yet" checked={unwrittenOnly} onChange={setUnwrittenOnly} />
        </>
      }
      legend={
        <Legend
          items={[
            { label: 'On the route', color: EDGE.step },
            { label: 'Prerequisite the route pulls in', color: EDGE.added },
            ...(withGoal ? [{ label: '+ Added by the second goal', color: EDGE.extra }] : []),
            { label: '⚠ Listed before its prerequisites', color: EDGE.flagged },
            { label: 'Dashed = not written yet', color: 'var(--viz-ink-muted)' },
          ]}
        />
      }
      stats={
        <Stats
          items={[
            { label: 'Modules needed', value: fmtNum(base.lit.size), hint: `${base.steps.length} listed on the route + ${base.added.size} prerequisites` },
            { label: 'Pages', value: fmtNum(base.pages) },
            { label: 'Written so far', value: `${fmtNum(base.builtPages)} of ${fmtNum(base.pages)}` },
            { label: 'Estimated time', value: `${fmtH(hours(base.pages))} h`, hint: `${base.pages} pages × ${MINUTES_PER_PAGE} min` },
          ]}
        />
      }
      note={
        <Note>
          {focusNode ? (
            <>
              <strong>{focusNode.index} {focusNode.title}</strong> —{' '}
              {focusNode.prerequisites.length
                ? `needs ${focusNode.prerequisites.map((p) => bySlug.get(p)?.index ?? p).join(', ')} (ringed)`
                : 'has no prerequisites'}
              ; {fmtNum(dependents(focusNode.slug).size)} later modules build on it. {focusNode.builtPages}/{focusNode.pages} pages written.
            </>
          ) : withGoal ? (
            <>
              <strong>Adding “{GOALS.find((g) => g.id === secondGoal)!.label}” pulls in {newlyLit.size} more module{newlyLit.size === 1 ? '' : 's'}</strong>
              {newlyLit.size ? ` (${fmtNum(withGoal.pages - base.pages)} pages)` : ' — this route already covers it'}. A goal is never just its own module: it drags in everything that module assumes.
            </>
          ) : base.outOfOrder.size ? (
            <>
              <strong>⚠ This route lists {base.outOfOrder.size} module{base.outOfOrder.size === 1 ? '' : 's'} before {base.outOfOrder.size === 1 ? 'its' : 'their'} own prerequisites:</strong>{' '}
              {[...base.outOfOrder].map(([s, needs]) => `${bySlug.get(s)!.index} needs ${needs.map((n) => bySlug.get(n)!.index).join(', ')}`).join('; ')}. Read in the listed order, those pages lean on material the route only reaches later. The valid order starts {base.readingOrder.slice(0, 3).map((s) => bySlug.get(s)!.index).join(' → ')}…
            </>
          ) : (
            <>
              <strong>{route.label}:</strong> {route.blurb || 'the minimal set of modules for this goal.'}{' '}
              {base.added.size
                ? `The route names ${base.steps.length} modules, but their declared prerequisites pull in ${base.added.size} more.`
                : 'Every prerequisite is already on the route.'}
              {route.stopAfter ? ` A reasonable stopping point is after “${route.stopAfter}”.` : ''} Hover any module to see what it depends on.
            </>
          )}
          {base.unknown.length ? ` (Route names unknown modules: ${base.unknown.join(', ')}.)` : ''}
        </Note>
      }
      table={
        <table className="viz-table">
          <thead>
            <tr>
              <th>Leg</th>
              <th>Modules first needed</th>
              <th>Pages</th>
              <th>Written</th>
              <th>Cumulative time</th>
            </tr>
          </thead>
          <tbody>
            {legRows.map((r) => (
              <tr key={r.name}>
                <td>
                  {r.name}
                  {route.stopAfter === r.name ? ' — stopping point' : ''}
                </td>
                <td>{r.modules}</td>
                <td>{r.pages}</td>
                <td>{r.written}</td>
                <td>{fmtH(r.cumulativeHours)} h</td>
              </tr>
            ))}
          </tbody>
        </table>
      }
    >
      <div style={{ display: 'grid', gap: '0.5rem', minWidth: 560 }} onMouseLeave={() => setFocus(null)}>
        {graph.parts.map((part) => {
          const mods = graph.modules.filter((m) => m.part === part.n);
          const litHere = mods.filter((m) => base.lit.has(m.slug) || newlyLit.has(m.slug)).length;
          return (
            <div key={part.n} style={{ display: 'grid', gridTemplateColumns: '8.5rem 1fr', gap: '0.5rem', alignItems: 'start' }}>
              <div style={{ fontSize: '0.72rem', color: 'var(--viz-ink-2)', paddingTop: '0.2rem' }}>
                <strong style={{ color: 'var(--viz-ink)' }}>Part {part.n}</strong>
                <br />
                {litHere}/{mods.length} on route
              </div>
              <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.3rem' }}>
                {mods.map((m) => {
                  const st = stateOf(m.slug);
                  const written = m.builtPages > 0;
                  const dimmed = st === 'outside' || (unwrittenOnly && written);
                  const ringed = focusPrereqs.has(m.slug);
                  const edge = st === 'outside' ? 'transparent' : EDGE[st];
                  const label = `${st === 'flagged' ? '⚠ ' : st === 'extra' ? '+ ' : ''}${m.index} ${m.title}`;
                  const style: React.CSSProperties = {
                    display: 'inline-block',
                    maxWidth: '15rem',
                    overflow: 'hidden',
                    textOverflow: 'ellipsis',
                    whiteSpace: 'nowrap',
                    fontSize: '0.72rem',
                    lineHeight: 1.35,
                    padding: '0.18rem 0.45rem 0.18rem 0.4rem',
                    borderRadius: 6,
                    background: 'var(--viz-plane)',
                    color: dimmed ? 'var(--viz-ink-muted)' : 'var(--viz-ink)',
                    border: `1px ${written ? 'solid' : 'dashed'} ${ringed ? 'var(--viz-ink)' : 'var(--viz-border)'}`,
                    boxShadow: ringed ? '0 0 0 2px var(--viz-ink)' : undefined,
                    borderLeft: `4px solid ${edge}`,
                    opacity: dimmed && !ringed ? 0.55 : 1,
                    textDecoration: 'none',
                  };
                  const title = `${m.index} ${m.title} — ${m.builtPages}/${m.pages} pages written${m.prerequisites.length ? `; needs ${m.prerequisites.map((p) => bySlug.get(p)?.index ?? p).join(', ')}` : ''}`;
                  const common = {
                    style,
                    title,
                    onMouseEnter: () => setFocus(m.slug),
                    onFocus: () => setFocus(m.slug),
                  };
                  return m.url ? (
                    <a key={m.slug} href={m.url} {...common}>
                      {label}
                    </a>
                  ) : (
                    <span key={m.slug} tabIndex={0} {...common}>
                      {label}
                    </span>
                  );
                })}
              </div>
            </div>
          );
        })}

        {route.entryPoints?.length ? (
          <div style={{ marginTop: '0.4rem', fontSize: '0.75rem', color: 'var(--viz-ink-2)' }}>
            <strong style={{ color: 'var(--viz-ink)' }}>Sideways entry, mid-incident:</strong>{' '}
            {route.entryPoints.map((s, i) => {
              const m = bySlug.get(s);
              if (!m) return null;
              return (
                <span key={s}>
                  {i ? ' · ' : ''}
                  {m.url ? <a href={m.url}>{m.title}</a> : <span title="Not written yet">{m.title} (not written yet)</span>}
                </span>
              );
            })}
          </div>
        ) : null}
      </div>
    </VizPanel>
  );
}

export type { Route };
