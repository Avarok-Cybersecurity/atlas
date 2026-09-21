#!/usr/bin/env node
// =============================================================================
// gen-ladder.mjs — generate src/lib/ladder.generated.json and
//                  src/lib/ladders.generated.json from the subjects' manifests
// -----------------------------------------------------------------------------
// SSOT: src/lib/concurrency-subjects.json names, per subject, the published
//   manifest (or null); each manifest names, per series and per rung, the raw
//   harness output file that backs the published number. Every figure the
//   page shows is COMPUTED by lib/ladder-build.mjs from those files — none is
//   transcribed. Change a measurement and the site changes with it; delete
//   one and the build fails.
//
// Two outputs from one build:
//   ladder.generated.json   the dense ladder (PUBLISHED_SUBJECT), shape
//                           unchanged — Verified, Receipt, the deck, llms.txt
//                           and the marketing claim all read it;
//   ladders.generated.json  every subject that has a manifest, keyed by
//                           subject id. A bun test pins that the dense entry
//                           deep-equals ladder.generated.json, so the two
//                           cannot drift: both are build products of one
//                           manifest.
//
// A subject with `published_manifest: null` is a state, not an error, and is
//   absent from ladders.generated.json. A manifest that names a raw file which
//   does not exist, or fails any guard in lib/ladder-build.mjs, is a hard
//   build failure: a silently-dropped rung would render as a shorter ladder
//   that still looks complete (PCND — no implicit defaults on a published
//   claim).
//
// Regenerate with:   node site/scripts/gen-ladder.mjs
// No third-party deps: Node builtins only.
// =============================================================================

import { createHash } from 'node:crypto';
import { readFileSync } from 'node:fs';
import { writeStable } from './lib/write-stable.mjs';
import { buildLadder } from './lib/ladder-build.mjs';
import { dirname, resolve, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const here = dirname(fileURLToPath(import.meta.url));
const REPO = resolve(here, '..', '..');
const LIB = resolve(here, '..', 'src', 'lib');
const SUBJECTS = resolve(LIB, 'concurrency-subjects.json');
const OUT = resolve(LIB, 'ladder.generated.json');
const OUT_ALL = resolve(LIB, 'ladders.generated.json');

// The subject whose ladder is ALSO written as ladder.generated.json: the
// marketing claim on Verified/Home/the deck is the dense campaign. Explicit,
// not "the first subject" — a reordered list must not move the claim.
export const PUBLISHED_SUBJECT = 'qwen38-27b';

const die = (msg) => {
  console.error(`gen-ladder: ${msg}`);
  process.exit(1);
};

const readJson = (path, what) => {
  try {
    return JSON.parse(readFileSync(path, 'utf8'));
  } catch (err) {
    die(`cannot read ${what} at ${path}: ${err.message}`);
  }
};

// The campaign driver stamps a sha256 of its own source into every record it
// writes, and the published rungs carry the hash of the copy that produced
// them. Hash the copy that ships in the tree so the deck can put the two side
// by side: a reader who runs the repo's driver gets THIS hash in their output.
// Computed, never typed — if the file is ever edited this moves with it.
const sha256Of = (path) => createHash('sha256').update(readFileSync(path)).digest('hex');

const subjects = readJson(SUBJECTS, 'concurrency-subjects.json');
const generated_utc = new Date().toISOString().replace(/\.\d+Z$/, 'Z');
const ladders = {};

for (const subject of subjects) {
  if (subject.published_manifest === null) continue;
  const manifestPath = resolve(REPO, subject.published_manifest);
  const dir = dirname(manifestPath);
  const manifest = readJson(manifestPath, `${subject.id} manifest`);
  const rawOf = (file) => readJson(join(dir, file), `${subject.id} raw source ${file}`);
  let harnessRepoSha256;
  try {
    harnessRepoSha256 = sha256Of(resolve(REPO, manifest.workload.harness));
  } catch (err) {
    die(`${subject.id}: cannot hash workload.harness ${manifest.workload?.harness}: ${err.message}`);
  }
  try {
    ladders[subject.id] = buildLadder(manifest, { subject, rawOf, harnessRepoSha256 });
  } catch (err) {
    die(`${subject.id}: ${err.message.replace(/^gen-ladder: /, '')}`);
  }
}

const dense = ladders[PUBLISHED_SUBJECT];
if (!dense) die(`subject ${PUBLISHED_SUBJECT} has no generated ladder — the published claim has no data`);
if (!dense.rows?.length) die(`subject ${PUBLISHED_SUBJECT} has no subject series — the published claim has no pair`);

const serialize = (o) => `${JSON.stringify(o, null, 2)}\n`;
writeStable(OUT, { generated_utc, ...dense }, ['generated_utc'], serialize);
writeStable(OUT_ALL, { generated_utc, subjects: ladders }, ['generated_utc'], serialize);

for (const [id, l] of Object.entries(ladders)) {
  const what = l.summary
    ? `${l.summary.won}/${l.summary.rungs} rungs won (${l.summary.min_ratio}x..${l.summary.max_ratio}x vs matched baseline)`
    : `baseline only, C=${l.concurrencies.join(',')} — no subject series yet`;
  console.log(`gen-ladder: ${id}: ${what}`);
}
console.log(`gen-ladder: -> ${OUT} (${PUBLISHED_SUBJECT}), ${OUT_ALL} (${Object.keys(ladders).length} subjects)`);
