#!/usr/bin/env node
// =============================================================================
// gen-live.mjs — the few counts the marketing pages print, in a file they can
// afford to load
// -----------------------------------------------------------------------------
// The marketing pages print two counts that come from large generated files:
//   how many concurrency gate records passed   (gates.generated.json, 1 MB)
//   how many model recipes the engine ships    (models.generated.json)
//
// Importing gates.generated.json to read one integer put a 1 MB chunk on the
// front page and eleven others. This writes the integers to
// src/lib/live.generated.json instead, a few hundred bytes, and
// src/lib/content/live.js reads that. The numbers are still generated, never
// typed, and still come from the same sources as the developer pages.
//
// Runs after gen-gates.mjs and gen-models.mjs (vite.config.js, package.json).
// Hard-fails on a missing or empty source, because a silently zero count on
// the front page looks like a fact.
//
// Regenerate with:   node site/scripts/gen-live.mjs
// No third-party deps: Node builtins only.
// =============================================================================

import { existsSync, readFileSync, writeFileSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const here = dirname(fileURLToPath(import.meta.url));
const lib = resolve(here, '..', 'src', 'lib');
const read = (name) => {
  const p = resolve(lib, name);
  if (!existsSync(p)) throw new Error(`gen-live: ${p} is missing, run its generator first`);
  return JSON.parse(readFileSync(p, 'utf8'));
};

const gates = read('gates.generated.json');
const models = read('models.generated.json');

const concurrency = Object.entries(gates.benchmarks ?? {})
  .filter(([id]) => id.startsWith('concurrency'))
  .flatMap(([, b]) => b.records ?? []);
if (concurrency.length === 0) throw new Error('gen-live: gates.generated.json holds no concurrency gate records');

const recipes = models.reduce((n, v) => n + v.subfamilies.reduce((m, f) => m + f.recipes.length, 0), 0);
if (recipes === 0) throw new Error('gen-live: models.generated.json produced no recipes');

const out = {
  gates: {
    sha: gates.generated_sha,
    date: gates.generated_date,
    registered: gates.registered?.length ?? 0,
    concurrencyRecords: concurrency.length,
    concurrencyPass: concurrency.filter((r) => r.verdict === 'PASS').length
  },
  recipes
};

const target = resolve(lib, 'live.generated.json');
const text = JSON.stringify(out, null, 2) + '\n';
if (existsSync(target) && readFileSync(target, 'utf8').replace(/\r\n/g, '\n') === text) {
  console.log(`gen-live: ${target} unchanged (${out.gates.concurrencyPass}/${out.gates.concurrencyRecords} concurrency gates, ${recipes} recipes)`);
} else {
  writeFileSync(target, text);
  console.log(`gen-live: wrote ${target} (${out.gates.concurrencyPass}/${out.gates.concurrencyRecords} concurrency gates, ${recipes} recipes)`);
}
