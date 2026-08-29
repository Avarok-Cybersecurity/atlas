import { sveltekit } from '@sveltejs/kit/vite';
import { defineConfig } from 'vite';
import { resolve } from 'node:path';
import { glslStrip } from '../web-shared/glsl-strip.js';

export default defineConfig({
  plugins: [glslStrip(), sveltekit()],
  server: {
    // app.css imports ../../web-shared/atlas-tokens.css, which is outside this
    // app's root. The build resolves it regardless; the dev server needs to be
    // told the path is allowed.
    fs: { allow: [resolve(import.meta.dirname, '..', 'web-shared')] }
  }
});
