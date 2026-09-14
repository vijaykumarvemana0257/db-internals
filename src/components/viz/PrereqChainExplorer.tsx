import { useMemo, useState } from 'react';
import { VizPanel, Choice, Legend, Stats, Note, fmtNum } from './Viz';
import { graph, bySlug, closure, dependents, hours, MINUTES_PER_PAGE } from './curriculumRoutes';

const LEVEL_ORDER = ['foundations', 'core', 'advanced', 'expert'] as const;
const DEPTH_COLOR: Record<string, string> = {
  basics: 'var(--viz-seq-250)',
  intermediate: 'var(--viz-seq-400)',
  deep: 'var(--viz-seq-700)',
};

/**
 * Longest-path layering of a module's prerequisite closure: layer 0 is the module,
 * layer k holds prerequisites whose longest route to it is k edges.
 */
function layers(target: string): string[][] {
  const chain = closure([target]);
  const dist = new Map<string, number>([[target, 0]]);
  // Process in reverse curriculum order so every module is finalized before its prerequisites.
  const ordered = [...chain].sort((a, b) => graph.modules.findIndex((m) => m.slug === b) - graph.modules.findIndex((m) => m.slug === a));
  for (const s of ordered) {
    const d = dist.get(s);
    if (d === undefined) continue;
    for (const p of bySlug.get(s)!.prerequisites) if (chain.has(p)) dist.set(p, Math.max(dist.get(p) ?? 0, d + 1));
  }
  const out: string[][] = [];
  for (const [s, d] of dist) (out[d] ??= []).push(s);
  return out;
}

/** Pages before the first 'deep' subtopic: the data-derived "enough for most engineers" point. */
function enoughFor(slug: string) {
  const subs = bySlug.get(slug)!.subtopics;
  const firstDeep = subs.findIndex((s) => s.depth === 'deep');
  return firstDeep === -1 ? subs.length : firstDeep;
}

export default function PrereqChainExplorer() {
  const choices = useMemo(
    () =>
      graph.modules
        .filter((m) => m.prerequisites.length && m.slug !== 'start-here-routes-and-method')
        .map((m) => ({ value: m.slug, label: `${m.index} ${m.title}` })),
    [],
  );
  const [target, setTarget] = useState('b-trees-and-variants');
  const [skip, setSkip] = useState('none');

  const node = bySlug.get(target)!;
  const tiers = useMemo(() => layers(target), [target]);
  const chain = useMemo(() => new Set(tiers.flat()), [tiers]);
  const skipOptions = [{ value: 'none', label: 'Nothing' }, ...[...chain].filter((s) => s !== target).sort().map((s) => ({ value: s, label: `${bySlug.get(s)!.index} ${bySlug.get(s)!.title}` }))];
  const skipValid = skip !== 'none' && chain.has(skip) && skip !== target;

  // Skipping a prerequisite strands every module in this chain that (transitively) depends on it.
  const stranded = useMemo(() => (skipValid ? new Set([...dependents(skip)].filter((s) => chain.has(s))) : new Set<string>()), [skip, skipValid, chain]);
  const strandedSiteWide = skipValid ? dependents(skip).size : 0;

  const pages = [...chain].reduce((n, s) => n + bySlug.get(s)!.pages, 0);
  const depthMix = node.subtopics.reduce<Record<string, number>>((acc, s) => ((acc[s.depth] = (acc[s.depth] ?? 0) + 1), acc), {});
  const enough = enoughFor(target);

  const chip = (s: string) => {
    const m = bySlug.get(s)!;
    const isTarget = s === target;
    const isSkipped = s === skip;
    const isStranded = stranded.has(s);
    const label = `${isSkipped ? '✕ ' : isStranded ? '⚠ ' : ''}${m.index} ${m.title}`;
    const style: React.CSSProperties = {
      display: 'block',
      fontSize: '0.72rem',
      lineHeight: 1.35,
      padding: '0.3rem 0.45rem',
      borderRadius: 6,
      background: 'var(--viz-plane)',
      color: isSkipped || isStranded ? 'var(--viz-ink-muted)' : 'var(--viz-ink)',
      border: `1px ${m.builtPages ? 'solid' : 'dashed'} var(--viz-border)`,
      borderLeft: `4px solid ${isSkipped ? 'var(--viz-ink-muted)' : isStranded ? 'var(--viz-critical)' : isTarget ? 'var(--viz-1)' : 'var(--viz-2)'}`,
      textDecoration: isSkipped ? 'line-through' : 'none',
      marginBottom: '0.3rem',
    };
    const title = `${m.index} ${m.title} — ${m.level}, ${m.builtPages}/${m.pages} pages written`;
    return m.url ? (
      <a key={s} href={m.url} style={style} title={title}>
        {label}
      </a>
    ) : (
      <span key={s} style={style} title={`${title} (not written yet)`}>
        {label}
      </span>
    );
  };

  return (
    <VizPanel
      title="What a module stands on"
      subtitle="Pick a module to lay out everything it assumes, nearest prerequisites first. Then skip one of them and see which modules in the chain are left standing on nothing."
      controls={
        <>
          <Choice label="Module" value={target} onChange={(v) => { setTarget(v); setSkip('none'); }} options={choices} />
          <Choice label="Skip a prerequisite" value={skipValid ? skip : 'none'} onChange={setSkip} options={skipOptions} />
        </>
      }
      legend={
        <Legend
          items={[
            { label: 'The module', color: 'var(--viz-1)' },
            { label: 'A prerequisite', color: 'var(--viz-2)' },
            { label: '✕ Skipped', color: 'var(--viz-ink-muted)' },
            { label: '⚠ Stands on the skipped module', color: 'var(--viz-critical)' },
            { label: 'Dashed = not written yet', color: 'var(--viz-ink-muted)' },
          ]}
        />
      }
      stats={
        <Stats
          items={[
            { label: 'Level', value: node.level, hint: `One of ${LEVEL_ORDER.join(' → ')}` },
            { label: 'Modules in its chain', value: fmtNum(chain.size) },
            { label: 'Pages to reach it', value: fmtNum(pages), hint: `≈ ${hours(pages).toFixed(0)} h at ${MINUTES_PER_PAGE} min/page` },
            { label: 'Enough for most engineers', value: enough === 0 ? 'starts deep' : enough === node.pages ? `all ${node.pages}` : `first ${enough} of ${node.pages}`, hint: enough === 0 ? 'Its first page is already marked deep: a specialist module' : 'Pages before the first one marked deep' },
          ]}
        />
      }
      note={
        <Note>
          {skipValid ? (
            <>
              <strong>Skipping {bySlug.get(skip)!.index} {bySlug.get(skip)!.title}</strong> strands {stranded.size} module{stranded.size === 1 ? '' : 's'} in this chain
              {stranded.size ? ` (${[...stranded].map((s) => bySlug.get(s)!.index).join(', ')})` : ''}, including the one you picked
              {strandedSiteWide > stranded.size ? `, and ${fmtNum(strandedSiteWide - stranded.size)} more elsewhere in the curriculum` : ''}. Those pages still read — they just lean on material you never covered.
            </>
          ) : (
            <>
              <strong>
                {node.index} {node.title}
              </strong>{' '}
              is <em>{node.level}</em>, but its own pages run {Object.entries(depthMix).map(([d, n]) => `${n} ${d}`).join(', ')}. Module level says where it sits in the curriculum; page depth says how far into the mechanism each page goes.
              {node.seeAlso?.length ? ` It also points forward to ${node.seeAlso.map((s) => bySlug.get(s)?.index ?? s).join(', ')} as recommended later reading — not a prerequisite.` : ''}
            </>
          )}
        </Note>
      }
      table={
        <table className="viz-table">
          <thead>
            <tr>
              <th>Page</th>
              <th>Depth</th>
              <th>Written</th>
            </tr>
          </thead>
          <tbody>
            {node.subtopics.map((s, i) => (
              <tr key={s.title}>
                <td>
                  {i + 1}. {s.title}
                  {i === enough - 1 && enough < node.pages ? ' — enough for most engineers' : ''}
                </td>
                <td>{s.depth}</td>
                <td>{s.url ? 'yes' : 'not yet'}</td>
              </tr>
            ))}
          </tbody>
        </table>
      }
    >
      <div style={{ display: 'grid', gridTemplateColumns: `repeat(${tiers.length}, minmax(10rem, 1fr))`, gap: '0.75rem', minWidth: tiers.length * 170 }}>
        {tiers.map((tier, i) => (
          <div key={i}>
            <div style={{ fontSize: '0.7rem', color: 'var(--viz-ink-2)', marginBottom: '0.35rem' }}>
              {i === 0 ? 'The module' : i === 1 ? 'Needs directly' : `${i} steps back`}
            </div>
            {[...tier].sort((a, b) => bySlug.get(a)!.index.localeCompare(bySlug.get(b)!.index, undefined, { numeric: true })).map(chip)}
          </div>
        ))}
      </div>

      <div style={{ marginTop: '0.75rem' }}>
        <div style={{ fontSize: '0.7rem', color: 'var(--viz-ink-2)', marginBottom: '0.3rem' }}>
          Page depth inside {node.index}, in reading order
        </div>
        <div style={{ display: 'flex', gap: 3 }} role="img" aria-label={`Depth of each page in ${node.title}`}>
          {node.subtopics.map((s, i) => (
            <div
              key={s.title}
              title={`${i + 1}. ${s.title} — ${s.depth}`}
              style={{
                flex: 1,
                height: 14,
                borderRadius: 3,
                background: DEPTH_COLOR[s.depth] ?? 'var(--viz-ink-muted)',
                outline: i === enough - 1 && enough < node.pages ? '2px solid var(--viz-ink)' : undefined,
                outlineOffset: 1,
              }}
            />
          ))}
        </div>
        <Legend
          items={[
            { label: 'basics', color: DEPTH_COLOR.basics },
            { label: 'intermediate', color: DEPTH_COLOR.intermediate },
            { label: 'deep', color: DEPTH_COLOR.deep },
          ]}
        />
      </div>
    </VizPanel>
  );
}
