#!/usr/bin/env node
// =============================================================================
// gen-gates.mjs — generate src/lib/gates.generated.json from .benchmarks/
// -----------------------------------------------------------------------------
// SSOT: the PR gate commits one record per (benchmark, run) at
//   .benchmarks/<bench>/<YYYY-MM-DD>-<sha10>.json — on the branch the gate ran
//   against. The newest records therefore often sit on an UNMERGED branch, so
//   this generator unions records across:
//     1. the checked-out working tree (committed data — can never flake), then
//     2. every remote head, via git plumbing (fetch + ls-tree + cat-file).
//   Leg 2 is best-effort: git is authenticated by the checkout itself, uses no
//   HTTP API (zero rate-limit exposure), and on any failure the output simply
//   degrades to worktree-only — it must NEVER clobber good data with less
//   (see gen-stars.mjs for the bug that rule comes from).
//
// The registered benchmark list is derived from the descriptor SSOT
// (crates/avarok-plugin/src/benchmarks/**: `id: "<bench-id>"`), so the UI can
// name gated-but-not-yet-published benchmarks without hardcoding them.
//
// Records are slimmed for the page: `closure` (per-kernel hashes, ~10x the
// payload), `hardware_state.before/after` and `summary` are dropped; every
// field the dashboard's metadata card shows is kept verbatim. `command` is
// kept VERBATIM too — the point card's "reproduction steps" panel shows what
// was run, and a command reconstructed on the page from `params` would be a
// command nobody ran. `perf_env`, `dirty_paths`, `dataset_fingerprint`, the
// record `path`, its `.sig` signer and a `box_state` subset ride along for
// the same panel (see src/lib/repro-steps.js).
//
// Regenerate with:   node site/scripts/gen-gates.mjs
// No third-party deps: Node builtins + `git` via child_process.
// =============================================================================

import { readdirSync, readFileSync, writeFileSync, existsSync, statSync } from 'node:fs';
import { writeStable } from './lib/write-stable.mjs';
import { dirname, resolve, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { execFileSync } from 'node:child_process';
import { assignTrendPredecessors } from '../src/lib/gate-lineage.js';

const here = dirname(fileURLToPath(import.meta.url));
const REPO = resolve(here, '..', '..');
const RECORDS_ROOT = resolve(REPO, '.benchmarks');
const DESCRIPTOR_ROOT = resolve(REPO, 'crates', 'avarok-plugin', 'src', 'benchmarks');
const OUT = resolve(here, '..', 'src', 'lib', 'gates.generated.json');

function git(args, opts = {}) {
  return execFileSync('git', ['-C', REPO, ...args], {
    encoding: 'utf8',
    stdio: ['ignore', 'pipe', 'pipe'],
    maxBuffer: 64 * 1024 * 1024,
    ...opts
  }).trim();
}
function gitSoft(args) {
  try {
    return git(args);
  } catch {
    return '';
  }
}

const ancestryCache = new Map();
const resolvedCommitCache = new Map();
let commitParents;

function loadCommitParents() {
  if (commitParents) return commitParents;
  commitParents = new Map();
  for (const line of gitSoft(['rev-list', '--parents', '--all']).split('\n')) {
    const [sha, ...parents] = line.trim().split(/\s+/);
    if (sha) commitParents.set(sha, parents);
  }
  return commitParents;
}

function resolveCommit(sha) {
  if (!sha) return '';
  if (resolvedCommitCache.has(sha)) return resolvedCommitCache.get(sha);
  const parents = loadCommitParents();
  let resolved = parents.has(sha) ? sha : '';
  if (!resolved) {
    const matches = [...parents.keys()].filter((candidate) => candidate.startsWith(sha));
    if (matches.length === 1) resolved = matches[0];
  }
  resolvedCommitCache.set(sha, resolved);
  return resolved;
}

function gitIsAncestor(older, newer) {
  if (!older || !newer) return false;
  if (older === newer) return true;
  const key = `${older}>${newer}`;
  if (ancestryCache.has(key)) return ancestryCache.get(key);
  let yes = false;
  const parents = loadCommitParents();
  const olderCommit = resolveCommit(older);
  const newerCommit = resolveCommit(newer);
  const pending = newerCommit ? [newerCommit] : [];
  const seen = new Set();
  while (pending.length > 0) {
    const sha = pending.pop();
    if (sha === olderCommit) {
      yes = true;
      break;
    }
    if (seen.has(sha)) continue;
    seen.add(sha);
    pending.push(...(parents.get(sha) ?? []));
  }
  ancestryCache.set(key, yes);
  return yes;
}

function gitCommitKnown(sha) {
  return Boolean(resolveCommit(sha));
}

// --- registered suite from the descriptor SSOT -------------------------------
// Each `BenchmarkDescriptor { id: "…", … expected_secs: N, … sensitivity:
// Sensitivity::X }` literal also yields the planning cost and class the
// reproduction panel quotes, read from the text between one `id:` and the
// next so they cannot be attributed to a neighbouring descriptor.
function registeredBenchmarks() {
  const ids = new Set();
  const meta = {};
  const walk = (dir) => {
    for (const name of readdirSync(dir)) {
      const p = join(dir, name);
      if (statSync(p).isDirectory()) walk(p);
      else if (name.endsWith('.rs') && !name.includes('test')) {
        const src = readFileSync(p, 'utf8');
        const hits = [...src.matchAll(/^\s*id: "([a-z0-9-]+)"/gm)];
        hits.forEach((m, k) => {
          ids.add(m[1]);
          const body = src.slice(m.index, hits[k + 1]?.index ?? src.length);
          const secs = /^\s*expected_secs: (\d+)/m.exec(body);
          const sens = /^\s*sensitivity: Sensitivity::(\w+)/m.exec(body);
          if (secs && sens) meta[m[1]] = { expected_secs: Number(secs[1]), sensitivity: sens[1] };
        });
      }
    }
  };
  if (existsSync(DESCRIPTOR_ROOT)) walk(DESCRIPTOR_ROOT);
  return { ids: [...ids].sort(), meta };
}

// The serve allowance every self-served gate may spend before its first
// sample, from the hardware limits SSOT. One key, so a TOML parser is not
// worth a dependency; absent file or key → null, and the panel says so.
function serveAllowanceSecs() {
  const p = resolve(REPO, 'kernels', 'gb10', 'HARDWARE.toml');
  if (!existsSync(p)) return null;
  const m = /^serve_allowance_s\s*=\s*(\d+)/m.exec(readFileSync(p, 'utf8'));
  return m ? Number(m[1]) : null;
}

// --- record slimming ---------------------------------------------------------
// Keep exactly the fields the dashboard shows; `branch` is provenance added
// here (empty string = committed on the current checkout), `path` is the
// record's repo path and `signer` the key fingerprint from its `.sig`
// sidecar (null = unsigned).
//
// `dirty_paths`, `perf_env` and `dataset_fingerprint` are decoded the way
// record.rs serialises them: `skip_serializing_if` empty/none, so an ABSENT
// key is an empty list / map / no fingerprint — a faithful read of the
// record, not a default invented here.
function slim(raw, branch, path, signer) {
  const hs = raw.hardware_state;
  return {
    path,
    signer,
    benchmark_id: raw.benchmark_id,
    benchmark_name: raw.benchmark_name,
    git_sha: raw.git_sha,
    recorded_at: raw.recorded_at,
    target_model: raw.target_model,
    served_by: raw.served_by,
    atlas_version: raw.atlas_version,
    hardware: raw.hardware,
    perf_class: raw.hardware_state?.perf_class ?? '',
    machine_id:
      raw.hardware_state?.before?.machine?.machine_id ??
      raw.hardware_state?.after?.machine?.machine_id ??
      '',
    params: raw.params,
    serve_overrides: raw.serve_overrides,
    metrics: raw.metrics,
    frame_status: raw.frame_status,
    verdict: raw.verdict,
    verdict_reason: raw.verdict_reason,
    command: raw.command,
    perf_env: raw.perf_env ?? {},
    dirty_paths: raw.dirty_paths ?? [],
    dataset_fingerprint: raw.dataset_fingerprint ?? null,
    // The box-state subset the panel reads; null = no hardware check recorded.
    // `sensitivity` is NOT repeated here: it is a property of the benchmark
    // and rides in `registered_meta` from the descriptor SSOT.
    box_state: hs
      ? {
          validity: hs.postcheck?.validity ?? null,
          concerns: hs.postcheck?.concerns ?? [],
          gpu_temp_delta_c: hs.delta?.gpu_temp_delta_c ?? null,
          hottest_chassis_delta_c: hs.delta?.hottest_chassis_delta_c ?? null,
          elapsed_s: hs.delta?.elapsed_s ?? null
        }
      : null,
    branch
  };
}

/** The signer fingerprint inside a `.sig` sidecar's JSON, or null. */
function signerOf(sigText) {
  if (!sigText) return null;
  try {
    const key = JSON.parse(sigText).key;
    return typeof key === 'string' && key !== '' ? key : null;
  } catch {
    return null;
  }
}

// --- leg 1: working tree (committed data — structural) -----------------------
const records = new Map(); // ".benchmarks/<bench>/<file>" -> slim record
if (existsSync(RECORDS_ROOT)) {
  for (const bench of readdirSync(RECORDS_ROOT)) {
    const dir = join(RECORDS_ROOT, bench);
    if (!statSync(dir).isDirectory()) continue;
    for (const f of readdirSync(dir).filter((f) => f.endsWith('.json'))) {
      const raw = JSON.parse(readFileSync(join(dir, f), 'utf8'));
      const p = `.benchmarks/${bench}/${f}`;
      const sig = join(dir, `${f}.sig`);
      records.set(p, slim(raw, '', p, signerOf(existsSync(sig) ? readFileSync(sig, 'utf8') : '')));
    }
  }
}
const committedCount = records.size;

// --- leg 2: every remote head (best-effort) ----------------------------------
let branchesScanned = 0;
let fromBranches = 0;
try {
  const remote = gitSoft(['remote']).split('\n')[0];
  if (remote) {
    // Shallow-refresh all heads; tolerable if it fails (offline build).
    try {
      git(['fetch', '--quiet', '--depth=1', remote, `+refs/heads/*:refs/remotes/${remote}/*`], {
        timeout: 120_000
      });
    } catch (err) {
      console.error(`gen-gates: fetch degraded (${String(err.message || err).split('\n')[0]})`);
    }
    const refs = gitSoft(['for-each-ref', '--format=%(refname:short)', `refs/remotes/${remote}`])
      .split('\n')
      .filter((r) => r && !r.endsWith('/HEAD'));
    for (const ref of refs) {
      branchesScanned += 1;
      const paths = gitSoft(['ls-tree', '-r', '--name-only', ref, '--', '.benchmarks'])
        .split('\n')
        .filter((p) => p.endsWith('.json'));
      for (const p of paths) {
        if (records.has(p)) continue;
        try {
          const raw = JSON.parse(git(['show', `${ref}:${p}`]));
          records.set(p, slim(raw, ref.replace(`${remote}/`, ''), p, signerOf(gitSoft(['show', `${ref}:${p}.sig`]))));
          fromBranches += 1;
        } catch {
          /* unreadable blob on a foreign branch — skip, never fail the build */
        }
      }
    }
  }
} catch (err) {
  console.error(`gen-gates: branch scan degraded (${String(err.message || err).split('\n')[0]})`);
}

// --- assemble ---------------------------------------------------------------
const benchmarks = {};
const generatedHead = gitSoft(['rev-parse', 'HEAD']);
for (const rec of records.values()) {
  const b = (benchmarks[rec.benchmark_id] ??= { name: rec.benchmark_name, records: [] });
  b.records.push(rec);
}
for (const b of Object.values(benchmarks)) {
  b.records.sort((x, y) => x.recorded_at - y.recorded_at);
  assignTrendPredecessors(b.records, gitIsAncestor);
  for (const rec of b.records) {
    rec.generated_ancestry = !gitCommitKnown(rec.git_sha)
      ? 'unknown'
      : gitIsAncestor(rec.git_sha, generatedHead)
        ? 'yes'
        : 'no';
  }
}

const registered = registeredBenchmarks();
const obj = {
  generated_sha: gitSoft(['rev-parse', '--short', 'HEAD']),
  generated_date: gitSoft(['log', '-1', '--format=%cs']),
  registered: registered.ids,
  registered_meta: registered.meta,
  limits: { serve_allowance_s: serveAllowanceSecs() },
  sources: { committed: committedCount, branches_scanned: branchesScanned, from_branches: fromBranches },
  benchmarks
};
writeStable(OUT, obj, ['generated_sha', 'generated_date'], (o) => JSON.stringify(o) + '\n');
console.log(
  `gen-gates: ${records.size} records (${committedCount} committed, +${fromBranches} from ${branchesScanned} branches) -> ${OUT}`
);
