import { plugin } from 'bun';
import { compileModule } from 'svelte/compiler';
import { readFileSync } from 'fs';
import { join, dirname } from 'path';
import { fileURLToPath } from 'url';

// `.svelte.js` modules are RUNE modules: `$state` and friends are compiler
// constructs, not runtime functions, so `bun test` cannot import them as-is.
// That is why none of the six rune modules in this repo has ever had a test —
// and why a latching-state regression in `fleet.svelte.js` reached main and had
// to be found by reading the call graph instead.
//
// Two things are needed: compile the runes, and resolve SvelteKit's `$lib`
// alias, which vite supplies during a real build and bun does not.
const LIB = join(dirname(fileURLToPath(import.meta.url)), 'src', 'lib');

plugin({
  name: 'svelte-runes',
  setup(build) {
    build.onResolve({ filter: /^\$lib(\/|$)/ }, (args) => ({
      path: join(LIB, args.path.slice('$lib'.length)),
    }));

    // `$app/environment` is SvelteKit's, supplied by vite at build time and
    // absent here. `chat/state.svelte.js` imports it, so without this a test of
    // that module fails with "Cannot find module '$app/environment'" — an error
    // about the harness, dressed as an error about the code.
    //
    // `browser: false` is the truthful answer, not a convenient one: bun test
    // has no DOM, so code guarding DOM access with `if (browser)` SHOULD skip.
    // Claiming true here would run those branches against globals that are not
    // there and fail somewhere less obvious.
    build.onLoad({ filter: /\.svelte\.js$/ }, (args) => {
      const src = readFileSync(args.path, 'utf8');
      const { js } = compileModule(src, { filename: args.path, generate: 'client' });
      // `$app/environment` is rewritten here rather than resolved in `onResolve`,
      // which does not fire for this specifier -- `$lib` does, and removing that
      // resolver demonstrably breaks the suite, so the difference is the specifier
      // and not the hook. `chat/state.svelte.js` imports it, and without this a
      // test of that module fails with "Cannot find module": an error about the
      // harness wearing the costume of an error about the code.
      const stub = join(dirname(fileURLToPath(import.meta.url)), 'test-stubs', 'app-environment.js');
      const code = js.code.replaceAll("'$app/environment'", JSON.stringify(stub));
      return { contents: code, loader: 'js' };
    });
  },
});
