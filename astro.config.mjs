// @ts-check
import { defineConfig } from 'astro/config';
import starlight from '@astrojs/starlight';
import react from '@astrojs/react';

import sidebar from './src/sidebar.json' with { type: 'json' };

// Deployment: set SITE (and BASE for GitHub Pages project sites, e.g. "/db-internals/").
// Cloudflare Pages / Netlify / Vercel / a custom domain need SITE only.
const site = process.env.SITE ?? 'https://db-internals.pages.dev';
const base = process.env.BASE ?? undefined;

export default defineConfig({
  site,
  base,
  integrations: [
    starlight({
      title: 'How Databases Work',
      description:
        'Databases and distributed data explained visually — from bytes on disk to planet-scale consensus.',
      sidebar,
      customCss: ['./src/styles/custom.css'],
      lastUpdated: true,
      pagination: true,
      tableOfContents: { minHeadingLevel: 2, maxHeadingLevel: 3 },
      social: [
        {
          icon: 'github',
          label: 'GitHub',
          href: 'https://github.com/',
        },
      ],
    }),
    react(),
  ],
});
