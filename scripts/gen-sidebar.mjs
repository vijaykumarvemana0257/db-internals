// Generates src/sidebar.json and _planning/page-manifest.json from the approved curriculum.
// Run: npm run sidebar
import { readFileSync, writeFileSync, mkdirSync, existsSync, readdirSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const root = resolve(dirname(fileURLToPath(import.meta.url)), '..');
// Must match astro.config.mjs's BASE resolution, so generated links land under
// the same base path the site is actually served from (e.g. GitHub Pages project sites).
const base = process.env.BASE ?? '/';
const withBase = (path) => `${base.replace(/\/$/, '')}${path}`;
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
        url: withBase(`/${moduleDir}/${pad(si + 1)}-${slugify(sub.title)}/`),
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

// ---- /curriculum/ : the full map, with built pages linked and the rest listed ----
const esc = (s) => s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
const built = (p) => existsSync(resolve(root, p.path));
const lines = [
  '---',
  'title: The full curriculum',
  'description: Every part, module and page on this site — 11 parts, 45 modules, 396 pages.',
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
        lines.push(built(m) ? `- [${esc(m.title)}](${m.url})` : `- ${esc(m.title)} <span class="pending">soon</span>`);
      });
    lines.push('');
  });
});
writeFileSync(resolve(root, 'src/content/docs/curriculum.md'), lines.join('\n'));

console.log(
  `sidebar: ${sidebar.length} parts, ${sidebar.reduce((n, p) => n + p.items.length, 0)} modules listed; ${manifest.filter(built).length}/${manifest.length} pages written`,
);
