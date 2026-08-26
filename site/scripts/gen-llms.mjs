#!/usr/bin/env node
// =============================================================================
// gen-llms.mjs — generate static/llms.txt from the same SSOTs as the page
// -----------------------------------------------------------------------------
// llms.txt is what an answer engine reads when it wants the short version of a
//   site. That makes it a claim surface, so it is generated rather than typed:
//   the model list comes from models.generated.json (itself generated from
//   atlas-recipes), the competitive numbers from ladder.generated.json, and the
//   MLPerf status from mlperf.json. Nothing here can drift from the page,
//   because there is no second copy to drift.
//
// Prose that is genuinely editorial (what Atlas is, what it is not) is read out
//   of src/lib/data.js, the same file the page renders — so a copy change lands
//   in both places at once.
//
// Hard-fails on a missing source, because a silently truncated llms.txt still
//   looks like a complete one (PCND).
//
// Regenerate with:   node site/scripts/gen-llms.mjs
// No third-party deps: Node builtins only.
// =============================================================================

import { readFileSync, writeFileSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const here = dirname(fileURLToPath(import.meta.url));
const site = resolve(here, '..');
const read = (p) => JSON.parse(readFileSync(resolve(site, p), 'utf8'));

const models = read('src/lib/models.generated.json');
const ladder = read('src/lib/ladder.generated.json');
const mlperf = read('src/lib/mlperf.json');
const bench = read('src/lib/benchmarks.generated.json');

// data.js is an ES module of plain exports; importing it keeps the copy in one
// place instead of restating it here.
const data = await import(resolve(site, 'src/lib/data.js'));
const { tagline, hero, githubUrl, recipesUrl, discordUrl, xUrl, guideUrl, hardware } = data;

const recipes = models.flatMap((v) =>
  v.subfamilies.flatMap((f) => f.recipes.map((r) => ({ vendor: v.vendor, family: f.name, ...r })))
);
if (recipes.length === 0) throw new Error('gen-llms: models.generated.json produced no recipes');
if (!ladder.rows?.length) throw new Error('gen-llms: ladder.generated.json has no rungs');

const w = ladder.workload;
const s = ladder.summary;
const fmt = (n) => n.toFixed(3);

const lines = [];
const push = (...l) => lines.push(...l);

push(
  '# Atlas Inference Engine',
  '',
  `> ${tagline}`,
  '',
  `${hero.sub}`,
  '',
  'Written in pure Rust and CUDA and licensed AGPL-3.0-only. One codebase covers',
  'the range, from edge-class accelerators through workstations to expert-parallel',
  'deployments across nodes.',
  '',
  '## What it runs on',
  ''
);
for (const c of hardware.cards) {
  push(`- ${c.name} (${c.chip}) — ${c.statusText}. ${c.body}`);
}

push(
  '',
  '## Measured performance',
  '',
  `${ladder.title}. ${ladder.subtitle}.`,
  `Aggregate: ${ladder.aggregate}. Box: ${ladder.box.name}, ${ladder.box.gpu}.`,
  `Workload: ISL ${w.isl_tokens} / OSL ${w.osl_tokens} tokens, temperature ${w.temperature},`,
  `seed ${w.seed}, ${w.reps} timed reps after ${w.warmup} warmup. ${w.sampling_parity}.`,
  '',
  `Result: Atlas wins ${s.won} of ${s.rungs} rungs, margin ${fmt(s.min_ratio)}x to ${fmt(s.max_ratio)}x`,
  'against whichever vLLM configuration is faster at that concurrency.',
  '',
  '| concurrency | Atlas tok/s | best vLLM tok/s | ratio |',
  '| --- | --- | --- | --- |'
);
for (const r of ladder.rows) {
  const best = r.baselines.find((b) => b.id === r.best_baseline_id);
  push(`| ${r.c} | ${r.atlas.toFixed(2)} | ${best.tok_s.toFixed(2)} (${best.label}) | ${fmt(r.ratio_vs_best)}x |`);
}
push(
  '',
  `Full campaign log including every rung lost on the way: ${ladder.results_doc_url}`,
  `MLPerf Inference v6.1: ${mlperf.status}, closed edge division, on both GB10 and gfx1151.`,
  `Release gate: ${bench.methodology}`,
  `Reproduce: ${bench.repro_cmd}`,
  ''
);

push('## Install', '', '```sh', data.runCommand, '```', '', 'Or without piping to a shell:', '', '```sh', data.quickInstall, data.runCommandRaw, '```', '');

push(
  `## Models (${recipes.length} recipes)`,
  '',
  'Every model below maps to one recipe in atlas-recipes; the site cannot list a',
  'model that has no recipe. Run any of them with `atlasctl run <id>`.',
  ''
);
for (const vendor of [...new Set(recipes.map((r) => r.vendor))]) {
  push(`### ${vendor}`, '');
  for (const r of recipes.filter((x) => x.vendor === vendor)) {
    push(`- \`${r.recipeId}\` — ${r.displayName}, ${r.params ?? 'n/a'} ${r.quant}, ${r.topology}, \`${r.hfId}\``);
  }
  push('');
}

push(
  '## Links',
  '',
  `- Engine repo: ${githubUrl}`,
  `- Recipes (model SSOT): ${recipesUrl}`,
  `- Deployment guide: ${guideUrl}`,
  `- Benchmark results: ${ladder.results_doc_url}`,
  `- Discord: ${discordUrl}`,
  `- X: ${xUrl}`,
  '- Site: https://atlasinference.io',
  '',
  '## License',
  '',
  'AGPL-3.0-only for the Community Edition. Contributions are covered by a CLA',
  'that permits Enterprise re-licensing.',
  ''
);

const out = resolve(site, 'static/llms.txt');
writeFileSync(out, lines.join('\n'));
console.log(`gen-llms: wrote ${out} (${lines.length} lines, ${recipes.length} recipes)`);
