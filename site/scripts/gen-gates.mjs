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
// (crates/atlas-plugin/src/benchmarks/**: `id: "<bench-id>"`), so the UI can
// name gated-but-not-yet-published benchmarks without hardcoding them.
//
// Records are slimmed for the page: `closure` (per-kernel hashes, ~10x the
// payload) and `command` (reconstructible from params) are dropped; every
// field the dashboard's metadata card shows is kept verbatim.
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
const DESCRIPTOR_ROOT = resolve(REPO, 'crates', 'atlas-plugin', 'src', 'benchmarks');
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
function registeredBenchmarks() {
  const ids = new Set();
  const walk = (dir) => {
    for (const name of readdirSync(dir)) {
      const p = join(dir, name);
      if (statSync(p).isDirectory()) walk(p);
      else if (name.endsWith('.rs') && !name.includes('test')) {
        for (const m of readFileSync(p, 'utf8').matchAll(/^\s*id: "([a-z0-9-]+)"/gm)) ids.add(m[1]);
      }
    }
  };
  if (existsSync(DESCRIPTOR_ROOT)) walk(DESCRIPTOR_ROOT);
  return [...ids].sort();
}

// --- record slimming ---------------------------------------------------------
// Keep exactly the fields the dashboard shows; `branch` is provenance added
// here (empty string = committed on the current checkout).
function slim(raw, branch) {
  return {
    benchmark_id: raw.benchmark_id,
    benchmark_name: raw.benchmark_name,
    git_sha: raw.git_sha,
    recorded_at: raw.recorded_at,
    target_model: raw.target_model,
    served_by: raw.served_by,
    atlas_version: raw.atlas_version,
    hardware: raw.hardware,
    perf_class: raw.hardware_state?.perf_class ?? '',
    params: raw.params,
    serve_overrides: raw.serve_overrides,
    metrics: raw.metrics,
    frame_status: raw.frame_status,
    verdict: raw.verdict,
    verdict_reason: raw.verdict_reason,
    branch
  };
}

// --- leg 1: working tree (committed data — structural) -----------------------
const records = new Map(); // ".benchmarks/<bench>/<file>" -> slim record
if (existsSync(RECORDS_ROOT)) {
  for (const bench of readdirSync(RECORDS_ROOT)) {
    const dir = join(RECORDS_ROOT, bench);
    if (!statSync(dir).isDirectory()) continue;
    for (const f of readdirSync(dir).filter((f) => f.endsWith('.json'))) {
      const raw = JSON.parse(readFileSync(join(dir, f), 'utf8'));
      records.set(`.benchmarks/${bench}/${f}`, slim(raw, ''));
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
          records.set(p, slim(raw, ref.replace(`${remote}/`, '')));
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

const obj = {
  generated_sha: gitSoft(['rev-parse', '--short', 'HEAD']),
  generated_date: gitSoft(['log', '-1', '--format=%cs']),
  registered: registeredBenchmarks(),
  sources: { committed: committedCount, branches_scanned: branchesScanned, from_branches: fromBranches },
  benchmarks
};
writeStable(OUT, obj, ['generated_sha', 'generated_date'], (o) => JSON.stringify(o) + '\n');
console.log(
  `gen-gates: ${records.size} records (${committedCount} committed, +${fromBranches} from ${branchesScanned} branches) -> ${OUT}`
);
