// SPDX-License-Identifier: AGPL-3.0-only

// Deck copy and derived figures. Same rule as the rest of the site: nothing
// numeric is typed here. Every ratio, rung count and command string is read out
// of ladder.generated.json / benchmarks.generated.json, so a re-run of
// `npm run generate` moves the slides too, and a lost rung changes the claim
// instead of leaving a stale one on a page shown to an investor.

import ladder from '$lib/ladder.generated.json';
import bench from '$lib/benchmarks.generated.json';
import gates from '$lib/gates.generated.json';
import { githubUrl } from '$lib/data.js';

const subject = ladder.series.find((s) => s.role === 'subject');
const matched = ladder.series.find((s) => s.id === 'vllm-mtp');
const unmatched = ladder.series.find((s) => s.id === 'vllm-nospec');

const pct = (r) => `${((r - 1) * 100).toFixed(1)}%`;
const x = (r) => `${r.toFixed(3)}×`;

export const stamp = `atlas ${bench.generated_sha} · ${ladder.generated_utc.slice(0, 10)}`;

export const claim = {
  engine: subject.engine,
  build: subject.build,
  buildPublic: subject.build_public,
  baseline: matched.engine ?? 'vLLM 0.27.1 + MTP',
  checkpoint: ladder.workload.checkpoint,
  box: ladder.box.gpu,
  boxName: ladder.box.name,
  rungs: ladder.summary.rungs,
  won: ladder.summary.won,
  allWon: ladder.summary.all_won,
  min: x(ladder.summary.min_ratio),
  max: x(ladder.summary.max_ratio),
  minPct: pct(ladder.summary.min_ratio),
  maxPct: pct(ladder.summary.max_ratio),
  concurrencies: ladder.concurrencies.join(', '),
  aggregate: ladder.aggregate,
  isl: ladder.workload.isl_tokens,
  osl: ladder.workload.osl_tokens,
  seed: ladder.workload.seed,
  temperature: ladder.workload.temperature,
  reps: ladder.workload.reps,
  warmup: ladder.workload.warmup,
  harnessFile: ladder.workload.harness,
  resultsUrl: ladder.results_doc_url,
  resultsDoc: ladder.results_doc
};

// The rungs won by a margin small enough that ordinary run-to-run drift could
// swallow them. Derived, not typed: the earlier hand-written sentence said
// "1.007x to 1.03x" while the record said 1.012x to 1.032x -- the exact staleness
// the generated-copy rule exists to prevent. 1.05 is the cutoff because the
// campaign's own re-measurements moved rungs by ~2-4%.
const FRAGILE_MAX_RATIO = 1.05;
const fragileRows = ladder.rows.filter((r) => r.ratio_vs_best < FRAGILE_MAX_RATIO);

export const fragile = {
  count: fragileRows.length,
  rungs: fragileRows.map((r) => `C=${r.c}`).join(', '),
  min: x(Math.min(...fragileRows.map((r) => r.ratio_vs_best))),
  max: x(Math.max(...fragileRows.map((r) => r.ratio_vs_best)))
};

// The fingerprint an analyst needs before they can start. Third element is the
// reason the value matters, not a restatement of it.
export const fingerprint = [
  ['box', `${ladder.box.name} — ${ladder.box.gpu}`, ladder.box.note],
  ['checkpoint', ladder.workload.checkpoint, ladder.workload.checkpoint_note],
  ['harness', ladder.workload.harness, `campaign driver for the published ladder, sha256 pinned per leg; the gate's own instrument is \`spark benchmark run concurrency-sweep\``],
  ['atlas', `${subject.engine} @ ${subject.build}`, subject.build_note],
  ['baseline', matched.engine, 'container digest pinned in RESULTS.md'],
  ['aggregate', ladder.aggregate, `${ladder.workload.warmup} warmup discarded`]
];

// Parity. `false` in the third slot means the axis is deliberately not pinned,
// and the value column has to carry the reason.
export const parity = [
  ['context', '2048 both', true],
  ['batch cap', '128 both', true],
  ['gpu util', '0.85 both', true],
  ['kv cache', 'fp8 both', true],
  ['prefix cache', 'on both', true],
  ['speculation', 'MTP K=4 both', true],
  ['thinking', ladder.workload.thinking, true],
  ['sampling', ladder.workload.sampling_parity, true],
  ['prompts', `ISL ${ladder.workload.isl_tokens} / OSL ${ladder.workload.osl_tokens}, seed ${ladder.workload.seed}, temp ${ladder.workload.temperature}`, true],
  ['harness', 'one script, both legs, back to back', true]
];

// Heiser's benchmarking-crimes taxonomy, answered. Rows that are still open
// stay open — see Audit.svelte.
export const audit = [
  {
    state: 'clear',
    risk: 'Unfairly tuned competitor',
    answer: `Baseline is vLLM with MTP K=4 and fp8 KV, not vLLM at defaults. The default-config number (${unmatched.label}) is published beside it and is not what we claim against.`
  },
  {
    state: 'clear',
    risk: 'Selective data range',
    answer: `All ${ladder.summary.rungs} rungs published, C=${ladder.concurrencies[0]} to ${ladder.concurrencies.at(-1)}, including the ones we win by ${ladder.summary.min_ratio.toFixed(3)}×.`
  },
  {
    state: 'clear',
    risk: 'No statistical treatment',
    answer: `${ladder.workload.reps} timed reps per rung with the spread printed; the C=32 A/B publishes every rep so the distributions can be checked for overlap.`
  },
  {
    state: 'clear',
    risk: 'Relative numbers only',
    answer: 'Absolute tok/s for both engines at every rung, two decimals, matching the repo record byte for byte.'
  },
  {
    state: 'clear',
    risk: 'Incomplete platform spec',
    answer: 'GPU SKU, driver, container digest, harness sha256, engine build SHA and full launch flags for both legs.'
  },
  {
    state: 'clear',
    risk: 'Microbenchmark as end-to-end',
    answer: 'Every number is served over the OpenAI HTTP path with a real client. No kernel-level timings are claimed as serving throughput.'
  },
  {
    state: 'clear',
    risk: 'Best-of instead of representative',
    answer: 'Where a rung was re-measured higher, the older, lower number is published. C=2 is the documented case.'
  },
  {
    state: 'open',
    risk: 'Single hardware, single model',
    answer: 'One GB10 box, one 27B NVFP4 checkpoint. We publish what we measured there and label it as such — no figure here is an extrapolation to another SKU or model class.'
  },
  {
    state: 'open',
    risk: 'Fleet drift over time',
    answer: 'A fleet-wide shift cost Atlas 4.2% and vLLM 2.2% at C=32. The margin narrowed and held; the differential is published rather than the favourable snapshot.'
  },
  {
    state: 'open',
    risk: 'Not third-party audited',
    answer: 'Everything here is self-run. That is exactly why the artifacts are pinned to the level an external auditor would ask for, and why we would rather you re-ran it than took it.'
  }
];

export const links = {
  results: ladder.results_doc_url,
  repo: githubUrl,
  gateDoc: bench.gate_doc,
  repro: bench.repro_cmd
};

export const gateFacts = {
  registered: gates.registered.length,
  committed: gates.sources.committed,
  branches: gates.sources.branches_scanned,
  fromBranches: gates.sources.from_branches,
  methodology: bench.methodology
};
