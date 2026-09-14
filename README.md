# db-internals

**How Databases Work** — databases and distributed data explained visually, from first
principles to the depth a senior engineer needs. Every page runs the mechanism it teaches.

- `CURRICULUM.md` — the full plan: 11 parts, 45 modules, 396 pages, each with a build spec
- `_planning/` — curriculum JSON and the generated page manifest

## Develop

```bash
npm install
npm run dev        # http://localhost:4321
npm run build      # static site → dist/
npm run preview    # serve the build
npm run sidebar    # regenerate nav + /curriculum from _planning/curriculum-result.json
```

Run `npm run sidebar` after adding a module's first page — the nav only lists modules that
have pages on disk, so it grows as parts ship.

## Stack

Astro + Starlight (sidebar, search, dark mode, mobile — all built in) with React islands
for the interactive visuals. Output is fully static: no server, no database, no API keys.

- `src/content/docs/` — pages, one `.mdx` per subtopic
- `src/components/viz/` — interactive visuals, built on `viz/Viz.tsx` primitives
- `src/styles/custom.css` — the design system, including the validated visualization palette

## Publish

The build is plain static files in `dist/`, so any static host works.

**Cloudflare Pages / Netlify / Vercel** — connect the repo, build command `npm run build`,
output directory `dist`. Set `SITE` to the final URL.

**GitHub Pages** — push to `main` and the included workflow
(`.github/workflows/deploy.yml`) builds and deploys. For a project site (not a custom
domain) also set `BASE` to `/<repo-name>/` in the workflow env, so asset paths resolve.
