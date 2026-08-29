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
    build.onLoad({ filter: /\.svelte\.js$/ }, (args) => {
      const src = readFileSync(args.path, 'utf8');
      const { js } = compileModule(src, { filename: args.path, generate: 'client' });
      return { contents: js.code, loader: 'js' };
    });
  },
});
