#!/usr/bin/env node
// =============================================================================
// gen-stars.mjs — generate src/lib/stars.generated.json from the GitHub API
// -----------------------------------------------------------------------------
// SSOT: the live GitHub star count + star history for Avarok-Cybersecurity/atlas
//   fetched via the `gh` CLI (uses ambient auth — GH_TOKEN in CI, the logged-in
//   user locally). This script is BEST-EFFORT: it MUST NEVER fail the build.
//   On any error it re-emits the existing generated file unchanged, or writes a
//   safe fallback, and always exits 0.
//
// Regenerate with:   node site/scripts/gen-stars.mjs
//
// Output is consumed by the star-count / growth UI:
//   { count, url, history: [{ date: "YYYY-MM", stars: <cumulative> }], generated_date }
//
// No third-party deps: Node builtins + `gh`/`git` via child_process.
// =============================================================================

import { readFileSync, writeFileSync, existsSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { execFileSync } from 'node:child_process';

const here = dirname(fileURLToPath(import.meta.url));
const REPO_DIR = resolve(here, '..', '..');
const OUT = resolve(here, '..', 'src', 'lib', 'stars.generated.json');

const REPO = 'Avarok-Cybersecurity/atlas';
const URL = 'https://github.com/Avarok-Cybersecurity/atlas';
const FALLBACK_COUNT = 546;

function gh(args) {
  // stderr is piped (not discarded) so a failing leg names its HTTP error in
  // the build log instead of degrading invisibly.
  return execFileSync('gh', args, {
    encoding: 'utf8',
    stdio: ['ignore', 'pipe', 'pipe'],
    maxBuffer: 64 * 1024 * 1024
  }).trim();
}
function git(args) {
  try {
    return execFileSync('git', ['-C', REPO_DIR, ...args], {
      encoding: 'utf8',
      stdio: ['ignore', 'pipe', 'ignore']
    }).trim();
  } catch {
    return '';
  }
}

// Never throw past here: re-emit existing file, else fallback. Always exit 0.
function emitFallback() {
  if (existsSync(OUT)) {
    try {
      writeFileSync(OUT, readFileSync(OUT, 'utf8'));
      console.log(`gen-stars: degraded — re-emitted existing ${OUT}`);
      return;
    } catch {
      /* fall through to hard fallback */
    }
  }
  const obj = { count: FALLBACK_COUNT, url: URL, history: [], generated_date: '' };
  writeFileSync(OUT, JSON.stringify(obj, null, 2) + '\n');
  console.log(`gen-stars: degraded — wrote fallback (count=${FALLBACK_COUNT})`);
}

// Bucket ISO starred_at timestamps into a CUMULATIVE DAILY series, then
// downsample to <=60 points so the growth curve has real shape and the SVG
// stays light. Always keeps the first + last point; terminates at live count.
function buildHistory(isoLines, count) {
  const days = isoLines
    .map((s) => s.trim())
    .filter(Boolean)
    .map((s) => s.slice(0, 10)) // YYYY-MM-DD
    .sort();
  if (days.length === 0) return [];
  const perDay = new Map();
  for (const d of days) perDay.set(d, (perDay.get(d) || 0) + 1);
  const ordered = [...perDay.keys()].sort();
  let cum = 0;
  let series = ordered.map((d) => {
    cum += perDay.get(d);
    return { date: d, stars: cum };
  });
  if (series.length) series[series.length - 1].stars = count; // authoritative

  const MAX = 60;
  if (series.length > MAX) {
    const step = (series.length - 1) / (MAX - 1);
    const ds = [];
    for (let i = 0; i < MAX; i++) ds.push(series[Math.round(i * step)]);
    ds[ds.length - 1] = series[series.length - 1];
    series = ds;
  }
  return series;
}

// The history leg fails in environments whose token cannot list stargazers
// (the endpoint rejects anonymous and restricted tokens). When that happens,
// the fresh history is empty but the count leg (plain repo metadata) still
// succeeds — and writing { count, history: [] } would CLOBBER the committed
// history and blank the star chart on the deployed site. Salvage instead:
// keep the previously generated curve and pin its terminal point to the live
// count so the figure and the curve agree.
function salvageHistory(count, generatedDate) {
  if (!existsSync(OUT)) return [];
  try {
    const prev = JSON.parse(readFileSync(OUT, 'utf8'));
    const hist = (Array.isArray(prev.history) ? prev.history : []).filter(
      (p) => p && typeof p.date === 'string' && Number.isFinite(p.stars)
    );
    if (hist.length === 0) return [];
    const last = hist[hist.length - 1];
    if (last.stars !== count) {
      if (generatedDate && generatedDate > last.date) hist.push({ date: generatedDate, stars: count });
      else last.stars = count;
    }
    console.log(`gen-stars: history leg degraded — salvaged ${hist.length} committed points`);
    return hist;
  } catch {
    return [];
  }
}

try {
  const count = parseInt(gh(['api', `repos/${REPO}`, '--jq', '.stargazers_count']), 10);
  if (!Number.isFinite(count)) throw new Error('non-numeric star count');
  const generatedDate = git(['log', '-1', '--format=%cs']);

  let history = [];
  try {
    const iso = gh([
      'api',
      `repos/${REPO}/stargazers`,
      '-H',
      'Accept: application/vnd.github.star+json',
      '--paginate',
      '--jq',
      '.[].starred_at'
    ]);
    history = buildHistory(iso.split('\n'), count);
  } catch (err) {
    // Name the failure — a silent catch here is what shipped an empty chart.
    const detail = ((err && (err.stderr || err.message)) || String(err)).toString().trim();
    console.error(`gen-stars: stargazers leg failed: ${detail}`);
  }
  if (history.length === 0) history = salvageHistory(count, generatedDate);

  const obj = { count, url: URL, history, generated_date: generatedDate };
  writeFileSync(OUT, JSON.stringify(obj, null, 2) + '\n');
  console.log(`gen-stars: count=${count}, ${history.length} history points -> ${OUT}`);
} catch (err) {
  console.error(`gen-stars: ${err && err.message ? err.message : err}`);
  emitFallback();
}

process.exit(0);
