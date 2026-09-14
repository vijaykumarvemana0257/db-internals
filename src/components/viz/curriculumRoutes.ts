/**
 * Curriculum graph helpers + the named routes through it.
 *
 * Everything here reads src/data/curriculum-graph.json, which scripts/gen-sidebar.mjs
 * regenerates on every build. Routes are lists of real module slugs; nothing is
 * hardcoded that the graph can compute (counts, prerequisite edges, written status).
 */
import graphData from '../../data/curriculum-graph.json';
import { siteHref } from './Viz';

export type Subtopic = { title: string; depth: string; url: string | null };
export type ModuleNode = {
  slug: string;
  title: string;
  index: string;
  part: number;
  level: string;
  prerequisites: string[];
  /** Recommended later reading: a cross-part link the module is written to stand without. */
  seeAlso?: string[];
  pages: number;
  builtPages: number;
  url: string | null;
  subtopics: Subtopic[];
};
export type Part = { n: number; title: string; summary: string };
export type Graph = {
  totals: { parts: number; modules: number; pages: number; builtPages: number };
  parts: Part[];
  modules: ModuleNode[];
};

const raw = graphData as Graph;
const withSite = (u: string | null) => (u ? siteHref(u) : null);
/** The generated graph stores base-less URLs; resolve them against the deployed base path here. */
export const graph: Graph = {
  ...raw,
  modules: raw.modules.map((m) => ({ ...m, url: withSite(m.url), subtopics: m.subtopics.map((s) => ({ ...s, url: withSite(s.url) })) })),
};
export const bySlug = new Map(graph.modules.map((m) => [m.slug, m]));
export const order = new Map(graph.modules.map((m, i) => [m.slug, i]));

/**
 * The single assumption behind every time estimate on the Start Here pages:
 * ~12 minutes to read a ~2,800-word page plus ~8 minutes with its simulator.
 */
export const MINUTES_PER_PAGE = 20;
export const hours = (pages: number) => (pages * MINUTES_PER_PAGE) / 60;

/** Transitive prerequisite closure of a set of modules (including the modules themselves). */
export function closure(slugs: Iterable<string>): Set<string> {
  const out = new Set<string>();
  const stack = [...slugs];
  while (stack.length) {
    const s = stack.pop()!;
    if (out.has(s) || !bySlug.has(s)) continue;
    out.add(s);
    for (const p of bySlug.get(s)!.prerequisites) stack.push(p);
  }
  return out;
}

/** Every module that transitively depends on `slug` (not including it). */
export function dependents(slug: string): Set<string> {
  const out = new Set<string>();
  let frontier = [slug];
  while (frontier.length) {
    const next: string[] = [];
    for (const m of graph.modules) {
      if (out.has(m.slug) || m.slug === slug) continue;
      if (m.prerequisites.some((p) => frontier.includes(p))) {
        out.add(m.slug);
        next.push(m.slug);
      }
    }
    frontier = next;
  }
  return out;
}

export const partModules = (n: number) =>
  graph.modules.filter((m) => m.part === n && m.slug !== 'start-here-routes-and-method').map((m) => m.slug);

export type Route = {
  id: string;
  label: string;
  audience: 'engineering' | 'roles' | 'goal';
  blurb: string;
  /** Ordered legs, each a list of module slugs. Order is the order the reader walks. */
  legs: { name: string; modules: string[] }[];
  /** Where the route can legitimately stop, by leg name. */
  stopAfter?: string;
  /** Sideways entry points for someone arriving mid-incident (Part 12). */
  entryPoints?: string[];
};

export const ROUTES: Route[] = [
  {
    id: 'app',
    label: 'Application engineer',
    audience: 'engineering',
    blurb: 'Make queries fast, keep data correct under concurrency, change schemas safely.',
    legs: [
      { name: 'Foundations', modules: ['data-models-and-sql-primer', 'sql-semantics-types-and-schema-design', 'anatomy-of-a-database-engine'] },
      { name: 'Reading plans', modules: ['sql-to-logical-plan', 'physical-operators', 'cost-based-optimization', 'explain-and-query-tuning'] },
      { name: 'Indexing', modules: ['index-selection-strategy'] },
      { name: 'Correctness under concurrency', modules: ['transactions-and-isolation', 'lock-based-concurrency-control', 'mvcc-storage-vacuum-gc'] },
      { name: 'Using it well', modules: ['advanced-sql', 'application-access', 'connections-pooling-and-process-models', 'application-data-patterns', 'schema-evolution-and-migrations'] },
      { name: 'Running it', modules: ['operating-observability-tuning-capacity'] },
    ],
    stopAfter: 'Using it well',
    entryPoints: ['triage-playbooks', 'decision-guides'],
  },
  {
    id: 'storage',
    label: 'Storage internals',
    audience: 'engineering',
    blurb: 'How a single-node engine stores, indexes, plans, isolates and recovers.',
    legs: [
      { name: 'Foundations', modules: partModules(1) },
      { name: 'Storage engines', modules: partModules(2) },
      { name: 'Query processing', modules: partModules(3) },
      { name: 'Transactions and recovery', modules: partModules(4) },
      { name: 'Real engines', modules: ['case-studies-relational-embedded'] },
    ],
    stopAfter: 'Transactions and recovery',
  },
  {
    id: 'dist',
    label: 'Distributed systems',
    audience: 'engineering',
    blurb: 'Replication, partitioning, consensus and distributed transactions — built on single-node depth.',
    legs: [
      { name: 'Single-node minimum', modules: ['anatomy-of-a-database-engine', 'transactions-and-isolation', 'application-access'] },
      { name: 'Transactions and recovery', modules: partModules(4) },
      { name: 'Distributed foundations', modules: partModules(6) },
      { name: 'Replication and consensus', modules: partModules(7) },
      { name: 'Distributed transactions and architectures', modules: partModules(8) },
      { name: 'Real systems', modules: partModules(9) },
    ],
    stopAfter: 'Replication and consensus',
  },
  {
    id: 'sre',
    label: 'SRE / DBA on call',
    audience: 'roles',
    blurb: 'Keep a database up: recovery, replication, vacuum, backups, incidents.',
    legs: [
      { name: 'The engine', modules: ['anatomy-of-a-database-engine'] },
      { name: 'MVCC, vacuum and horizons', modules: ['mvcc-storage-vacuum-gc', 'vacuum-freezing-and-horizons'] },
      { name: 'Durability', modules: ['write-ahead-logging-and-recovery', 'crash-recovery-passes-and-durability'] },
      { name: 'Replication', modules: ['replication', 'replication-operations'] },
      { name: 'Operating', modules: ['operating-observability-tuning-capacity', 'failover-stacks-poolers-and-upgrades', 'backups-incidents-disaster-recovery', 'database-incident-response'] },
    ],
    stopAfter: 'Replication',
    entryPoints: ['triage-playbooks', 'triage-playbooks-data-and-forensics'],
  },
  {
    id: 'data',
    label: 'Data engineer',
    audience: 'roles',
    blurb: 'Columnar storage, lakehouse tables, change data capture and warehouse pipelines.',
    legs: [
      { name: 'Models', modules: ['data-models-and-sql-primer'] },
      { name: 'Columnar storage', modules: ['column-stores-analytical-storage'] },
      { name: 'Analytical engines and lakehouse', modules: ['analytical-engines-and-lakehouse-storage', 'lakehouse-table-formats-and-catalogs'] },
      { name: 'Logs and CDC', modules: ['logs-streams-derived-data'] },
      { name: 'Pipelines and delivery', modules: ['data-engineering-warehouse-pipelines', 'analytics-delivery-semantics-and-activation'] },
    ],
    stopAfter: 'Logs and CDC',
    entryPoints: ['decision-guides', 'comparison-matrices'],
  },
  {
    id: 'platform',
    label: 'Platform engineer',
    audience: 'roles',
    blurb: 'Provision, secure and upgrade a fleet of databases others depend on.',
    legs: [
      { name: 'Provisioning', modules: ['provisioning-platform-and-fleet-lifecycle'] },
      { name: 'Security and tenancy', modules: ['security-identity-and-access-control', 'data-protection-and-tenant-isolation'] },
      { name: 'Observability and capacity', modules: ['operating-observability-tuning-capacity'] },
      { name: 'Managed services, cost and upgrades', modules: ['running-databases-capacity-platforms-upgrades', 'failover-stacks-poolers-and-upgrades'] },
    ],
    stopAfter: 'Observability and capacity',
    entryPoints: ['lifecycle-checklists-and-review-gates', 'decision-guides'],
  },
];

/**
 * A deliberately mis-sequenced route: how people often describe learning distributed
 * databases ("start with replication and consensus"). It exists so the validator on the
 * Start Here page has a genuine mistake to flag.
 */
export const NAIVE_ROUTE: Route = {
  id: 'naive',
  label: 'Distributed-first (a common mistake)',
  audience: 'engineering',
  blurb: 'Jump straight to replication and consensus, then circle back to transactions and logging.',
  legs: [
    { name: 'The exciting part', modules: ['replication', 'consensus-and-coordination'] },
    { name: 'Then the basics', modules: ['transactions-and-isolation', 'write-ahead-logging-and-recovery'] },
  ],
};

/** Single-goal routes: "I need to do X" → the minimal set of modules that teaches it. */
export const GOALS: Route[] = [
  { id: 'g-fast', label: 'Make this query fast', audience: 'goal', blurb: '', legs: [{ name: 'Goal', modules: ['explain-and-query-tuning', 'index-selection-strategy'] }] },
  { id: 'g-crash', label: 'Survive a crash without losing data', audience: 'goal', blurb: '', legs: [{ name: 'Goal', modules: ['write-ahead-logging-and-recovery', 'crash-recovery-passes-and-durability'] }] },
  { id: 'g-shard', label: 'Design a sharded store', audience: 'goal', blurb: '', legs: [{ name: 'Goal', modules: ['partitioning-and-sharding', 'querying-and-modeling-partitioned-stores'] }] },
  { id: 'g-iso', label: 'Stop a race condition in my app', audience: 'goal', blurb: '', legs: [{ name: 'Goal', modules: ['transactions-and-isolation', 'serializability-without-blocking'] }] },
];

export type Evaluated = {
  steps: string[]; // route steps in walk order, de-duplicated
  lit: Set<string>; // steps + everything their prerequisites pull in
  added: Set<string>; // lit but not a written step: prerequisites pulled in automatically
  outOfOrder: Map<string, string[]>; // step -> prerequisites the route lists LATER than it
  readingOrder: string[]; // lit modules in a valid prerequisite order
  unknown: string[]; // slugs a route names that do not exist in the graph
  pages: number;
  builtPages: number;
};

/**
 * Walk a route: compute everything it needs, and flag genuine sequencing mistakes.
 *
 * A prerequisite the route never mentions is fine — it is pulled in automatically and
 * read first. The real authoring error is listing a module BEFORE one of its own
 * (transitive) prerequisites that the route also lists, later.
 */
export function evaluate(routes: Route[]): Evaluated {
  const steps: string[] = [];
  const unknown: string[] = [];
  for (const r of routes)
    for (const leg of r.legs)
      for (const s of leg.modules) {
        if (!bySlug.has(s)) unknown.push(s);
        else if (!steps.includes(s)) steps.push(s);
      }

  const lit = closure(steps);
  const added = new Set([...lit].filter((s) => !steps.includes(s)));

  const outOfOrder = new Map<string, string[]>();
  steps.forEach((s, i) => {
    const later = new Set(steps.slice(i + 1));
    const needsLater = [...closure([s])].filter((p) => p !== s && later.has(p));
    if (needsLater.length) outOfOrder.set(s, needsLater);
  });

  // The curriculum's own order has no forward prerequisite edges, so sorting by it is a valid topological order.
  const readingOrder = [...lit].sort((a, b) => order.get(a)! - order.get(b)!);

  let pages = 0;
  let builtPages = 0;
  for (const s of lit) {
    pages += bySlug.get(s)!.pages;
    builtPages += bySlug.get(s)!.builtPages;
  }
  return { steps, lit, added, outOfOrder, readingOrder, unknown, pages, builtPages };
}
