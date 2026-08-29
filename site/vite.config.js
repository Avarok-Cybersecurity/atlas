import { sveltekit } from '@sveltejs/kit/vite';
import { defineConfig } from 'vite';
import { execFileSync } from 'node:child_process';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const here = dirname(fileURLToPath(import.meta.url));

// Regenerate src/lib/*.generated.json from their SSOTs on every build (and dev
// server start). Env (ATLAS_RECIPES_ROOT / ATLAS_BASELINES_ROOT / GH_TOKEN) is
// passed through so CI and local hosts resolve their sources identically.
function atlasGenerators() {
  const run = (script) =>
    execFileSync(process.execPath, [resolve(here, 'scripts', script)], {
      cwd: here,
      stdio: 'inherit',
      env: process.env
    });
  return {
    name: 'atlas-generators',
    apply: () => true, // build + serve
    buildStart() {
      // Structural generators: a nonzero exit is a hard, loud build failure.
      // (gen-gates degrades its cross-branch leg internally; its committed-
      // records leg is structural.)
      run('gen-models.mjs');
      run('gen-benchmarks.mjs');
      run('gen-gates.mjs');
      run('gen-ladder.mjs');
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
    }
  };
}

export default defineConfig({
  plugins: [atlasGenerators(), sveltekit()],
  server: {
    // app.css and the field import from web-shared/, outside this app's root.
    // The build resolves it regardless; the dev server has to be told.
    fs: { allow: [resolve(import.meta.dirname, '..', 'web-shared')] }
  }
});
