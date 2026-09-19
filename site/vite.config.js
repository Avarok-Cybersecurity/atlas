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
  build: {
    rolldownOptions: {
      output: {
        // Two named chunks instead of one chunk per component.
        //
        // The bundler gives every module its own chunk when a different set of
        // routes shares it, and the marketing site is thirty routes sharing
        // forty small components in different combinations. The front page
        // came out at 50 requests where main's made 27. Lighthouse serves over
        // HTTP/1.1, six connections at a time, so each extra small file is a
        // queued round trip ahead of the first paint: that alone held every
        // page at 99 against a gate that demands 100.
        //
        //   av-chrome   what every route loads: the header, the footer, the
        //               names and the nav tree, the page registry, and the
        //               rule that keeps the two design systems apart
        //   av-ui       the marketing components
        //
        // Page copy (src/lib/content/home.js and friends) is deliberately NOT
        // grouped: each stays its own file, loaded only by the pages that
        // print it.
        //
        // A group also takes its modules' DEPENDENCIES. So anything that both a
        // marketing component and a developer page import lands in av-ui, and
        // the developer page then has to download all of av-ui, and inline all
        // of its CSS, to read it. For a while /engine loaded 130 KB of marketing
        // components for the sake of a star count. The shared things are named
        // in av-chrome instead, which every route loads anyway: data.js, the
        // install helpers, the route group rule, and the two small generated
        // files both sides print (stars, the ladder). Never the big ones.
        // e2e/page-weight.spec.js fails if a developer page loads av-ui again,
        // or if any page grows a request.
        codeSplitting: {
          groups: [
            {
              name: 'av-chrome',
              priority: 30,
              test: /[\\/](src[\\/]lib[\\/]components[\\/]avarok[\\/](SiteNav|SiteFooter)\.svelte|src[\\/]lib[\\/]content[\\/](brand|index|faq)\.js|src[\\/]lib[\\/]route-groups\.js|src[\\/]lib[\\/]data\.js|src[\\/]lib[\\/]install[\\/]|src[\\/]lib[\\/](stars|ladder)\.generated\.json|web-shared[\\/]components[\\/](AtlasLockup|ThemeToggle)\.svelte|web-shared[\\/]theme\.js)/
            },
            {
              name: 'av-ui',
              priority: 20,
              test: /[\\/]src[\\/]lib[\\/](components[\\/]avarok[\\/]|reveal\.js)/
            }
          ]
        }
      }
    }
  },
  server: {
    // app.css and the field import from web-shared/, outside this app's root.
    // The build resolves it regardless; the dev server has to be told.
    fs: { allow: [resolve(import.meta.dirname, '..', 'web-shared')] },
    // `build/` is the static adapter's output, which the dev server never serves.
    // Left watched, every `vite build` beside a running dev server deletes and
    // rewrites a few hundred watched files, and every open tab is told to reload
    // once per page written. Vite adds this to its own ignore list.
    watch: { ignored: ['**/build/**'] }
  }
});
