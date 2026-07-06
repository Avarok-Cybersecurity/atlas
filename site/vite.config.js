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
      run('gen-models.mjs');
      run('gen-benchmarks.mjs');
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
  plugins: [atlasGenerators(), sveltekit()]
});
