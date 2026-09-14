import { defineCollection } from 'astro:content';
import { docsLoader } from '@astrojs/starlight/loaders';
import { docsSchema } from '@astrojs/starlight/schema';
import { z } from 'astro:content';

export const collections = {
  docs: defineCollection({
    loader: docsLoader(),
    schema: docsSchema({
      extend: z.object({
        // Page-level metadata used by the curriculum tooling and page chrome.
        depth: z.enum(['basics', 'intermediate', 'deep']).optional(),
        covers: z.string().optional(),
      }),
    }),
  }),
};
