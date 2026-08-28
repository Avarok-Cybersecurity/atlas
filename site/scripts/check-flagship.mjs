// SPDX-License-Identifier: AGPL-3.0-only
//
// The front page prints `atlasctl run <flagshipRecipe>` as its headline
// instruction. Nothing tied that name to the recipe corpus, so retiring or
// renaming a recipe in atlas-recipes would leave the site confidently
// advertising a command that fails on the visitor's machine — silently, and
// only for them.
//
// A standalone check run by the BUILD job, not a line inside gen-models.mjs.
// That generator's output is committed and it runs only when someone
// regenerates by hand, which is precisely not the moment a recipe gets
// retired: the guard would have missed the event it exists for.
//
// It reads the corpus CI actually ships against — the atlas-recipes checkout
// the build already makes for install.sh — rather than whatever branch a local
// mirror happens to be sitting on.

import { readdirSync, statSync } from 'node:fs';
import { resolve, dirname, join } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

const here = dirname(fileURLToPath(import.meta.url));
const root = process.env.ATLAS_RECIPES_ROOT || '/workspace/atlas-recipes/recipes';

// Imported, not scraped. A regex over data.js has no word boundary and takes
// the first match anywhere in the file, so a comment or an `oldFlagshipRecipe`
// above it would validate the wrong name and pass — failing open, which is the
// one way a guard must never fail.
const { flagshipRecipe, runCommandRaw } = await import(
  pathToFileURL(resolve(here, '..', 'src', 'lib', 'data.js')).href
);

/** Every `<name>.yaml` under the corpus, by stem. */
function recipeStems(dir) {
  const out = [];
  for (const entry of readdirSync(dir)) {
    const p = join(dir, entry);
    if (statSync(p).isDirectory()) out.push(...recipeStems(p));
    else if (entry.endsWith('.yaml') || entry.endsWith('.yml')) out.push(entry.replace(/\.ya?ml$/, ''));
  }
  return out;
}

let stems;
try {
  stems = recipeStems(root);
} catch (e) {
  console.error(`could not read the recipe corpus at ${root}: ${e.message}`);
  console.error('Set ATLAS_RECIPES_ROOT to a checkout of atlas-recipes/recipes.');
  process.exit(1);
}

if (stems.length === 0) {
  // An empty corpus would make every name "missing", which is a different
  // failure and must not be reported as a retired recipe.
  console.error(`no recipes found under ${root} — the corpus path is wrong, not the name`);
  process.exit(1);
}

const problems = [];
if (!stems.includes(flagshipRecipe)) {
  problems.push(
    `data.js advertises \`atlasctl run ${flagshipRecipe}\`, and no such recipe exists in the corpus.`
  );
}
// The pasteable command is checked too, because it is a second place the name
// can be written and the two have already drifted once.
if (typeof runCommandRaw === 'string' && !runCommandRaw.includes(flagshipRecipe)) {
  problems.push(
    `runCommandRaw (${runCommandRaw}) does not name the flagship recipe ${flagshipRecipe}.`
  );
}

if (problems.length > 0) {
  for (const p of problems) console.error(p);
  console.error('The front page would print a command that fails.');
  process.exit(1);
}
console.log(`flagship recipe ${flagshipRecipe} is in the corpus (${stems.length} recipes)`);
