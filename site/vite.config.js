import { sveltekit } from '@sveltejs/kit/vite';
import { defineConfig } from 'vite';
import { execFileSync } from 'node:child_process';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { glslStrip } from '../web-shared/glsl-strip.js';

const here = dirname(fileURLToPath(import.meta.url));

// Regenerate src/lib/*.generated.json from their SSOTs on every build (and dev
// server start). Env (AVAROK_RECIPES_ROOT / AVAROK_BASELINES_ROOT / GH_TOKEN) is
// passed through so CI and local hosts resolve their sources identically.
function avarokGenerators() {
  const run = (script) =>
    execFileSync(process.execPath, [resolve(here, 'scripts', script)], {
      cwd: here,
      stdio: 'inherit',
      env: process.env
    });
  return {
    name: 'avarok-generators',
    apply: () => true, // build + serve
    buildStart() {
      // Structural generators: a nonzero exit is a hard, loud build failure.
      // (gen-gates degrades its cross-branch leg internally; its committed-
      // records leg is structural.)
      run('gen-models.mjs');
      run('gen-benchmarks.mjs');
      run('gen-gates.mjs');
      run('gen-ladder.mjs');
      // The product updates page renders CHANGELOG.md. Structural, like gen-gates.
      run('gen-changelog.mjs');
      // Reduces gates and models to the few counts the marketing pages print, so
      // none of them has to load the 1 MB gate record set. After gen-gates and
      // gen-models, which it reads.
      run('gen-live.mjs');
      // Depends on the four above: llms.txt restates their numbers for answer
      // engines, so it must be written after they are.
      run('gen-llms.mjs');
      // Vendors the sha256-pinned LatticeDB wasm into static/lattice/; a pin
      // mismatch or no-cache-and-offline is a hard failure by design.
      run('gen-lattice.mjs');
      // Best-effort: gh/network flakiness must never fail the build.
      try {
        run('gen-stars.mjs');
      } catch (err) {
        this.warn(`gen-stars failed (non-fatal): ${err && err.message ? err.message : err}`);
      }
      // The contributors page renders the GitHub contributor list. Best-effort,
      // like gen-stars: the committed file stands in when the API is unreachable.
      try {
        run('gen-contributors.mjs');
      } catch (err) {
        this.warn(`gen-contributors failed (non-fatal): ${err && err.message ? err.message : err}`);
      }
    }
  };
}

export default defineConfig({
  plugins: [glslStrip(), avarokGenerators(), sveltekit()],
  server: {
    // app.css and the field import from web-shared/, outside this app's root.
    // The build resolves it regardless; the dev server has to be told.
    fs: { allow: [resolve(import.meta.dirname, '..', 'web-shared')] }
  }
});
