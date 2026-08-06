// SPDX-License-Identifier: AGPL-3.0-only

import adapter from '@sveltejs/adapter-static';
import { vitePreprocess } from '@sveltejs/vite-plugin-svelte';

/** @type {import('@sveltejs/kit').Config} */
const config = {
  preprocess: vitePreprocess(),
  kit: {
    // Fully static output: `build/` drops straight into an nginx docroot.
    adapter: adapter({
      pages: 'build',
      assets: 'build',
      fallback: undefined,
      precompress: false,
      strict: true
    }),
    // SvelteKit compiles src/service-worker.ts and registers it automatically.
    serviceWorker: {
      register: true
    },
    prerender: {
      entries: ['*'],
      handleHttpError: 'fail'
    }
  }
};

export default config;
