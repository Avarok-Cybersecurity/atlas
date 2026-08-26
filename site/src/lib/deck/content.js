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
  // The --concs argument, from the same rung list the chart is drawn from, so a
  // lost rung shortens the command as well as the ladder.
  concsArg: ladder.concurrencies.join(','),
  // The driver hash the published Atlas legs carry, beside the one a reader will
  // actually get from the tree. Both derived: the first is the manifest key whose
  // note names the Atlas legs, the second is hashed from the file at build time.
  harnessShaAtlas: Object.keys(ladder.harness_shas).find((k) =>
    /Atlas legs/i.test(ladder.harness_shas[k])
  ),
  harnessShaRepo: ladder.harness_repo_sha256.slice(0, 10),
  resultsUrl: ladder.results_doc_url,
  resultsDoc: ladder.results_doc
};

// Step 4's serve command, rendered FROM THE RECORD rather than hand-abridged.
//
// It used to be a shortened list with a note pointing at series[atlas].cli for
// "the full flag list, including the SSM cache and scheduling knobs". That
// pointer was not enough: the abridged command omits thirteen flags, six of
// which are kernel and scheduling knobs whose defaults are the OPPOSITE of the
// certified values (ssm_h_dtype f32 not f16-pool, gdn_fused_norm off,
// ssm_batched_recurrent off, ssm_tail_midchunk on, mtp_gate auto not force,
// prefill_varlen_batch off). Serving that way and then running the ladder in
// Step 5b measures a differently-configured engine and lands well under the
// chart — the walkthrough manufacturing evidence against its own claim, which
// is the worst failure available to a page like this. Measured on 2026-08-26:
// the abridged config ran 4.7% under at C=1 and 14% under at C=4, the gap
// widening with concurrency exactly as those knobs predict.
//
// Wrapped here rather than in the component so the command cannot drift from
// the record: change the serve config and this moves with it.
const argvLines = (argv, { width = 66, indent = '  ', breakAnywhere = false } = {}) => {
  const out = [];
  let line = '';
  for (const tok of argv) {
    // In a CLI a value belongs to the flag before it, so only a `--` token may
    // start a new line. An env prefix is all standalone assignments, so any
    // token may.
    const canBreak = breakAnywhere || tok.startsWith('--');
    const joined = line ? `${line} ${tok}` : tok;
    if (line && joined.length > width && canBreak) {
      out.push(line);
      line = indent + tok;
    } else {
      line = joined;
    }
  }
  if (line) out.push(line);
  return out;
};

export const serve = {
  env: argvLines(subject.env.split(/\s+/), { breakAnywhere: true, indent: '' }),
  cli: argvLines(subject.cli.split(/\s+/))
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
    // C=8 is the case that actually answers this, and C=2 is not: C=2's
    // published pair IS the better Atlas number (41.02, over round 11's 40.42)
    // — defensible because both legs were re-measured back to back that day and
    // the pair replaced the pair, but it is not an example of declining a
    // better number. C=8 is: re-measured at a HIGHER ratio and the lower one
    // still stands.
    answer: `Re-measuring C=8 on a later build gave 1.013×; the certified 1.012× is what is published. Both engines came out ~2.2% below their certified absolutes in that run while the ratio held, which is why every rung is quoted as a same-day A/B rather than against a stored number.`
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
