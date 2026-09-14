// Generates src/sidebar.json and _planning/page-manifest.json from the approved curriculum.
// Run: npm run sidebar
import { readFileSync, writeFileSync, mkdirSync, existsSync, readdirSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const root = resolve(dirname(fileURLToPath(import.meta.url)), '..');
// Generated URLs are base-less ("/pNN-part/module/NN-page/"). The base path is only known at build
// time (CI sets BASE for `astro build`, not for this script), so consumers add it: components via
// siteHref / import.meta.env.BASE_URL, and curriculum.md via page-relative links.
const curriculum = JSON.parse(
  readFileSync(resolve(root, '_planning/curriculum-result.json'), 'utf8'),
).curriculum;

const pad = (n) => String(n).padStart(2, '0');
const slugify = (s) =>
  s
    .toLowerCase()
    .replace(/[’']/g, '')
    .replace(/[^a-z0-9]+/g, '-')
    .replace(/^-+|-+$/g, '')
    .slice(0, 60);

const sidebar = [];
const manifest = [];

curriculum.parts.forEach((part, pi) => {
  const partDir = `p${pad(pi + 1)}-${slugify(part.title.split(':')[0])}`;
  const items = [];

  part.modules.forEach((mod, mi) => {
    const moduleDir = `${partDir}/${mod.slug}`;
    // Only list a module once it has pages on disk — the nav grows as parts ship,
    // instead of advertising links that 404.
    const abs = resolve(root, 'src/content/docs', moduleDir);
    const built = existsSync(abs) && readdirSync(abs).some((f) => f.endsWith('.mdx'));
    if (built) {
      items.push({
        label: `${pi + 1}.${mi + 1} ${mod.title}`,
        collapsed: true,
        items: [{ autogenerate: { directory: moduleDir } }],
      });
    }

    mod.subtopics.forEach((sub, si) => {
      manifest.push({
        part: pi + 1,
        partTitle: part.title,
        partSummary: part.summary,
        module: mi + 1,
        moduleSlug: mod.slug,
        moduleTitle: mod.title,
        moduleWhy: mod.why,
        moduleLevel: mod.level,
        prerequisites: mod.prerequisites,
        page: si + 1,
        title: sub.title,
        covers: sub.covers,
        visual: sub.visual,
        depth: sub.depth,
        path: `src/content/docs/${moduleDir}/${pad(si + 1)}-${slugify(sub.title)}.mdx`,
        url: `/${moduleDir}/${pad(si + 1)}-${slugify(sub.title)}/`,
      });
    });
  });

  if (items.length) {
    sidebar.push({
      label: `Part ${pi + 1} — ${part.title}`,
      collapsed: true,
      items,
    });
  }
});

mkdirSync(resolve(root, 'src'), { recursive: true });
writeFileSync(resolve(root, 'src/sidebar.json'), JSON.stringify(sidebar, null, 2) + '\n');
writeFileSync(
  resolve(root, '_planning/page-manifest.json'),
  JSON.stringify(manifest, null, 1) + '\n',
);

// ---- lint: content pages must not use root-absolute internal links ----
// Starlight prefixes `base` onto its own navigation but not onto links inside page content,
// so "/p02-…" 404s on a project site (base "/db-internals/"). Pages are three levels deep
// (pNN/module/page/), so write "../../../p02-…" instead, which resolves under any base.
{
  const offenders = [];
  const walk = (dir) => {
    for (const f of readdirSync(dir, { withFileTypes: true })) {
      const full = resolve(dir, f.name);
      if (f.isDirectory()) walk(full);
      else if (/\.mdx?$/.test(f.name) && !/^(index|curriculum)\.mdx?$/.test(f.name)) {
        const hits = readFileSync(full, 'utf8').match(/\]\(\/(p\d{2}-|curriculum)[^)]*\)/g);
        if (hits) offenders.push(`${full.replace(root + '/', '')}: ${hits.length} root-absolute link(s), e.g. ${hits[0]}`);
      }
    }
  };
  walk(resolve(root, 'src/content/docs'));
  // Components render links too: they must wrap internal paths in siteHref() from Viz.tsx.
  for (const f of readdirSync(resolve(root, 'src/components/viz'))) {
    if (!f.endsWith('.tsx')) continue;
    const src = readFileSync(resolve(root, 'src/components/viz', f), 'utf8');
    const hits = src.match(/href=["']\/(p\d{2}-|curriculum)[^"']*["']/g);
    if (hits) offenders.push(`src/components/viz/${f}: ${hits.length} root-absolute href(s) — wrap in siteHref(), e.g. ${hits[0]}`);
  }
  if (offenders.length) {
    console.error(`\nRoot-absolute internal links break under a base path. Use ../../../pNN-… instead:\n  ${offenders.join('\n  ')}\n`);
    process.exit(1);
  }
}

// ---- /curriculum/ : the full map, with built pages linked and the rest listed ----
const esc = (s) => s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
const built = (p) => existsSync(resolve(root, p.path));
const lines = [
  '---',
  'title: The full curriculum',
  `description: Every part, module and page on this site — ${curriculum.parts.length} parts, ${curriculum.parts.reduce((n, p) => n + p.modules.length, 0)} modules, ${manifest.length} pages.`,
  'tableOfContents: false',
  '---',
  '',
  `${curriculum.parts.length} parts · ${manifest.reduce((s, m) => s + (m.page === 1 ? 1 : 0), 0)} modules · ${manifest.length} pages.`,
  `**${manifest.filter(built).length}** are written so far; the rest are specified and queued.`,
  '',
];
curriculum.parts.forEach((part, pi) => {
  lines.push(`## Part ${pi + 1} — ${esc(part.title)}`, '', esc(part.summary), '');
  part.modules.forEach((mod, mi) => {
    lines.push(`### ${pi + 1}.${mi + 1} ${esc(mod.title)}`, '', `*${mod.level}* — ${esc(mod.why)}`, '');
    manifest
      .filter((m) => m.part === pi + 1 && m.module === mi + 1)
      .forEach((m) => {
        lines.push(built(m) ? `- [${esc(m.title)}](..${m.url})` : `- ${esc(m.title)} <span class="pending">soon</span>`);
      });
    lines.push('');
  });
});
writeFileSync(resolve(root, 'src/content/docs/curriculum.md'), lines.join('\n'));

// ---- src/data/curriculum-graph.json : module dependency graph for the Start Here visuals ----
// Regenerated on every build (CI runs `npm run sidebar` first), so `built` never goes stale.
const graph = {
  totals: {
    parts: curriculum.parts.length,
    modules: curriculum.parts.reduce((n, p) => n + p.modules.length, 0),
    pages: manifest.length,
    builtPages: manifest.filter(built).length,
  },
  parts: curriculum.parts.map((p, pi) => ({ n: pi + 1, title: p.title, summary: p.summary })),
  modules: curriculum.parts.flatMap((part, pi) =>
    part.modules.map((mod, mi) => {
      const pages = manifest.filter((m) => m.part === pi + 1 && m.module === mi + 1);
      const first = pages.find(built);
      return {
        slug: mod.slug,
        title: mod.title,
        index: `${pi + 1}.${mi + 1}`,
        part: pi + 1,
        level: mod.level,
        prerequisites: mod.prerequisites,
        seeAlso: mod.seeAlso ?? [],
        pages: pages.length,
        builtPages: pages.filter(built).length,
        url: first ? first.url : null,
        subtopics: pages.map((m) => ({ title: m.title, depth: m.depth, url: built(m) ? m.url : null })),
      };
    }),
  ),
};
mkdirSync(resolve(root, 'src/data'), { recursive: true });
writeFileSync(resolve(root, 'src/data/curriculum-graph.json'), JSON.stringify(graph) + '\n');

console.log(
  `sidebar: ${sidebar.length} parts, ${sidebar.reduce((n, p) => n + p.items.length, 0)} modules listed; ${manifest.filter(built).length}/${manifest.length} pages written`,
);
