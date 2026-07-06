#!/usr/bin/env node
// =============================================================================
// gen-benchmarks.mjs — generate src/lib/benchmarks.generated.json from baselines
// -----------------------------------------------------------------------------
// SSOT: the atlas test harness writes throughput baselines to tests/baselines/
//   (one *.json per gated model). Those files are the single source of truth for
//   the "verified" throughput a shipped Atlas image is held to. An EMPTY
//   tests/baselines/ (only .gitkeep + README.md) is the EXPECTED "pending"
//   state before the first gate run — it is NOT a build failure.
//
// Regenerate with:   node site/scripts/gen-benchmarks.mjs
//
// Output is consumed by the benchmarks UI:
//   { status, generated_sha, generated_date, methodology, gate_doc,
//     repro_cmd, entries: [{ label, model, quant, hardware, tps,
//                            source_path, repro_cmd }] }
//   status: "pending" when zero baselines exist, else "verified".
//
// MLPerf numbers are deliberately NOT sourced here — those live in the
// hand-edited src/lib/mlperf.json and only appear once officially published,
// in MLCommons citation format.
//
// No third-party deps: Node builtins + `git` (via child_process) for the stamp.
// =============================================================================

import { readdirSync, readFileSync, writeFileSync, existsSync } from 'node:fs';
import { basename, dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { execFileSync } from 'node:child_process';

const here = dirname(fileURLToPath(import.meta.url));
const REPO = resolve(here, '..', '..');
const BASELINES_ROOT =
  process.env.ATLAS_BASELINES_ROOT || resolve(REPO, 'tests', 'baselines');
const OUT = resolve(here, '..', 'src', 'lib', 'benchmarks.generated.json');

// --- git stamp (sha + committer date) ---------------------------------------
// Date comes from git, NOT Date.now() (unavailable / non-reproducible here).
function git(args) {
  try {
    return execFileSync('git', ['-C', REPO, ...args], {
      encoding: 'utf8',
      stdio: ['ignore', 'pipe', 'ignore']
    }).trim();
  } catch {
    return '';
  }
}
const sha = git(['rev-parse', '--short', 'HEAD']);
const date = git(['log', '-1', '--format=%cs']);

// --- quant derivation from the baseline filename stem ------------------------
function deriveQuant(stem) {
  return /fp8/i.test(stem) ? 'fp8' : 'nvfp4';
}

// --- main --------------------------------------------------------------------
if (!existsSync(BASELINES_ROOT)) {
  console.error(
    `gen-benchmarks: baselines directory does not exist: ${BASELINES_ROOT}\n` +
      `  Set ATLAS_BASELINES_ROOT or ensure tests/baselines/ is checked out.`
  );
  process.exit(1);
}

const files = readdirSync(BASELINES_ROOT)
  .filter((f) => f.endsWith('.json'))
  .sort();

let status;
let entries = [];
if (files.length === 0) {
  status = 'pending';
} else {
  status = 'verified';
  entries = files.map((filename) => {
    const stem = basename(filename).replace(/\.json$/, '');
    const obj = JSON.parse(readFileSync(resolve(BASELINES_ROOT, filename), 'utf8'));
    const model = obj.model || '';
    return {
      label: stem,
      model,
      quant: deriveQuant(stem),
      hardware: 'DGX Spark (GB10)',
      tps: obj.tps ?? null,
      source_path: 'tests/baselines/' + filename,
      repro_cmd:
        `python3 tests/single_gpu_suite.py --model ${model} ` +
        `--output tests/all_models_results/${stem}.json`
    };
  });
}

const obj = {
  status,
  generated_sha: sha,
  generated_date: date,
  methodology:
    'An Atlas image ships only after the serve matrix passes: every model ' +
    'boots, stays coherent (greedy determinism, no token leakage, tool ' +
    'reliability), and holds throughput within 10% of its committed baseline.',
  gate_doc:
    'https://github.com/Avarok-Cybersecurity/atlas/blob/main/docs/GB10_DEPLOYMENT_GUIDE.md#8-what-verified-means-so-you-can-trust-an-image',
  repro_cmd:
    'python3 tests/run_all_models.py && python3 tests/gate_results.py --update-baselines',
  entries
};

writeFileSync(OUT, JSON.stringify(obj, null, 2) + '\n');
console.log(`gen-benchmarks: status=${status}, ${entries.length} entries -> ${OUT}`);
