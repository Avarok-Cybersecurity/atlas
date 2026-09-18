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
// Prose that is genuinely editorial (what the engine is, what it is not) is read
//   out of src/lib/data.js, the same file the developer page renders, and the
//   company, platform, page list and pricing come from src/lib/content/, the
//   same modules the marketing pages render — so a copy change lands in both
//   places at once.
//
// Hard-fails on a missing source, because a silently truncated llms.txt still
//   looks like a complete one (PCND).
//
// Regenerate with:   node site/scripts/gen-llms.mjs
// No third-party deps: Node builtins only.
// =============================================================================

import { readFileSync, writeFileSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

const here = dirname(fileURLToPath(import.meta.url));
const site = resolve(here, '..');
const read = (p) => JSON.parse(readFileSync(resolve(site, p), 'utf8'));
// A file URL, not a path: node on Windows refuses to import 'C:\...'.
const load = (p) => import(pathToFileURL(resolve(site, p)).href);

const models = read('src/lib/models.generated.json');
const ladder = read('src/lib/ladder.generated.json');
const mlperf = read('src/lib/mlperf.json');
const bench = read('src/lib/benchmarks.generated.json');

// data.js is an ES module of plain exports; importing it keeps the copy in one
// place instead of restating it here.
const data = await load('src/lib/data.js');
const { hero, githubUrl, recipesUrl, discordUrl, xUrl, guideUrl, hardware } = data;

// The marketing copy. These modules import nothing but each other, so they
// load here exactly as the pages load them.
const { pages, company, SITE } = await load('src/lib/content/index.js');
const home = await load('src/lib/content/home.js');
const pricing = await load('src/lib/content/pricing.js');
if (!pages?.length) throw new Error('gen-llms: the page registry is empty');

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
  `# ${company.name}`,
  '',
  `> ${home.hero.title.join(' ')}`,
  '',
  home.hero.lede,
  '',
  `${company.name} was named Atlas until September 2026. The engine, the repository and the`,
  'domain are the same ones. The legal entity is Atlas Cybernetics Corp.',
  '',
  '## The platform',
  '',
  `- ${company.engine}: the open source inference engine, pure Rust and CUDA, AGPL-3.0-only.`,
  `- ${company.control}: the governance and control plane. Signed recipes, canary rollouts, routing, fleet policy.`,
  `- ${company.economics}: cost per workload, chargeback, stranded capacity and payback, from runtime telemetry.`,
  '',
  '## Pages',
  ''
);
for (const pg of pages.filter((x) => !x.noindex)) {
  push(`- [${pg.title}](${SITE}${pg.path === '/' ? '' : pg.path}): ${pg.description}`);
}
push(
  '',
  '## Pricing',
  '',
  'Proposed list prices, subject to contract. The pricing page carries the payback model.',
  ''
);
for (const t of pricing.tiers) {
  push(`- ${t.name}: ${t.price}${t.per ? ` ${t.per}` : ''}`);
}
push(
  '',
  `## ${company.engine}`,
  '',
  `${hero.sub}`,
  '',
  'Written in pure Rust and CUDA and licensed AGPL-3.0-only. One codebase covers',
  'the range, from edge-class accelerators through workstations to expert-parallel',
  'deployments across nodes.',
  '',
  '### What it runs on',
  ''
);
for (const c of hardware.cards) {
  push(`- ${c.name} (${c.chip}) — ${c.statusText}. ${c.body}`);
}

push(
  '',
  '### Measured performance',
  '',
  `${ladder.title}. ${ladder.subtitle}.`,
  `Aggregate: ${ladder.aggregate}. Box: ${ladder.box.name}, ${ladder.box.gpu}.`,
  `Workload: ISL ${w.isl_tokens} / OSL ${w.osl_tokens} tokens, temperature ${w.temperature},`,
  `seed ${w.seed}, ${w.reps} timed reps after ${w.warmup} warmup. ${w.sampling_parity}.`,
  '',
  `Result: ${company.engine} wins ${s.won} of ${s.rungs} rungs, margin ${fmt(s.min_ratio)}x to ${fmt(s.max_ratio)}x`,
  'against the matched vLLM + MTP configuration at each concurrency.',
  '',
  `| concurrency | ${company.engine} tok/s | matched vLLM tok/s | ratio |`,
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

push('### Install', '', '```sh', data.runCommand, '```', '', 'Or without piping to a shell:', '', '```sh', data.quickInstall, data.runCommandRaw, '```', '');

push(
  `### Models (${recipes.length} recipes)`,
  '',
  'Every model below maps to one recipe in atlas-recipes; the site cannot list a',
  'model that has no recipe. Run any of them with `atlasctl run <id>`.',
  ''
);
for (const vendor of [...new Set(recipes.map((r) => r.vendor))]) {
  push(`#### ${vendor}`, '');
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
  '- Site: https://atlascybernetics.ai',
  `- Developer page: https://atlascybernetics.ai${data.ENGINE}`,
  '- Documentation: https://docs.atlascybernetics.ai — full book, also at /llms.txt',
  '- Engineering blog: https://blog.atlascybernetics.ai — also at /llms.txt',
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
