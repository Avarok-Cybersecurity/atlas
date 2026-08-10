// SPDX-License-Identifier: AGPL-3.0-only

import { sveltekit } from '@sveltejs/kit/vite';
import { defineConfig } from 'vite';

export default defineConfig({
  plugins: [sveltekit()],
  build: {
    // The whole point is a small, auditable bundle. Fail loudly if it grows.
    chunkSizeWarningLimit: 200
  },
  server: {
    fs: { strict: true }
  }
});
