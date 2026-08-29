#!/usr/bin/env node
// =============================================================================
// gen-ladder.mjs — generate src/lib/ladder.generated.json from bench/ladder38/
// -----------------------------------------------------------------------------
// SSOT: bench/ladder38/published.json names, per series and per rung, the raw
//   harness output file that backs the published number. Every figure the page
//   shows is COMPUTED here from those files — none is transcribed. Change a
//   measurement and the site changes with it; delete one and the build fails.
//
// The published statistic is the mean of the timed reps, which is what
//   RESULTS.md publishes (verified: round 11's means reproduce its final table
//   exactly). The median and the rep spread are emitted alongside so the page
//   can show how tight each rung was rather than asking for trust.
//
// Ratios are taken against the BEST vLLM configuration at each rung, not the
//   matched one. At C=128 vLLM's no-speculation leg (390.42) beats its own MTP
//   leg (358.57), and quoting the MTP number there would inflate our margin
//   from 1.22x to 1.33x. The matched-parity ratio is emitted too, labelled.
//
// Hard-fails on a missing file, a missing rung, or a rung whose reps are empty:
//   a silently-dropped rung would render as a shorter ladder that still looks
//   complete (PCND — no implicit defaults on a published claim).
//
// Regenerate with:   node site/scripts/gen-ladder.mjs
// No third-party deps: Node builtins only.
// =============================================================================

import { createHash } from 'node:crypto';
import { readFileSync, writeFileSync } from 'node:fs';
import { writeStable } from './lib/write-stable.mjs';
import { dirname, resolve, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const here = dirname(fileURLToPath(import.meta.url));
const REPO = resolve(here, '..', '..');
const LADDER_DIR = resolve(REPO, 'bench', 'ladder38');
const MANIFEST = resolve(LADDER_DIR, 'published.json');
const OUT = resolve(here, '..', 'src', 'lib', 'ladder.generated.json');

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

const mean = (xs) => xs.reduce((a, b) => a + b, 0) / xs.length;
const median = (xs) => {
  const s = [...xs].sort((a, b) => a - b);
  const m = s.length >> 1;
  return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2;
};
const r2 = (v) => Math.round(v * 100) / 100;
// Ratios keep a third decimal: at C=2 the margin is 1.004x, and rounding that
// to 1.00x would present a real (if slim) win as a tie.
const r3 = (v) => Math.round(v * 1000) / 1000;

// One rung of one series: the reps for concurrency `c` inside `file`.
function rungStats(seriesId, c, file) {
  const path = join(LADDER_DIR, file);
  const doc = readJson(path, `${seriesId} C=${c} source`);
  const rung = (doc.rungs ?? []).find((r) => r.concurrency === c);
  if (!rung) die(`${file} has no rung for C=${c} (series ${seriesId})`);
  const reps = rung.reps ?? [];
  if (reps.length === 0) die(`${file} rung C=${c} has no reps (series ${seriesId})`);

  const tok = reps.map((r) => r.tok_s);
  const ttft = reps.map((r) => r.ttft_p50_ms).filter((v) => typeof v === 'number');
  const tpot = reps.map((r) => r.tpot_p50_ms).filter((v) => typeof v === 'number');
  const errs = reps.reduce((a, r) => a + (r.n_err ?? 0), 0);
  if (errs > 0) die(`${file} rung C=${c} recorded ${errs} request errors — not publishable`);

  return {
    c,
    tok_s: r2(mean(tok)),
    tok_s_median: r2(median(tok)),
    // Spread as a share of the mean: how much the rung moved run to run. A
    // published number whose reps disagree by 10% deserves to say so.
    spread_pct: r2(((Math.max(...tok) - Math.min(...tok)) / mean(tok)) * 100),
    reps: reps.length,
    ttft_p50_ms: ttft.length ? r2(median(ttft)) : null,
    tpot_p50_ms: tpot.length ? r2(median(tpot)) : null,
    source: file,
    measured_utc: doc.started_utc ?? null,
    harness_sha256: (doc.driver_sha256 ?? '').slice(0, 10) || null
  };
}

const manifest = readJson(MANIFEST, 'published.json manifest');

const series = manifest.series.map((s) => {
  const rungs = Object.entries(s.sources)
    .map(([c, file]) => rungStats(s.id, Number(c), file))
    .sort((a, b) => a.c - b.c);
  return { ...s, sources: undefined, rungs };
});

// Guard-then-use, matching `rungStats` above. `die` is returnless, so
// `x ?? die(...)` statically assigns undefined and would only fail later, at a
// confusing place — and not at all if `die` ever stopped calling process.exit.
const subject = series.find((s) => s.role === 'subject');
if (!subject) die('manifest has no subject series');
const baselines = series.filter((s) => s.role === 'baseline');
const matched = baselines.find((s) => s.parity === 'matched');
if (!matched) die('no matched-parity baseline');

// `variant`: another configuration of the SUBJECT engine (e.g. a different
// drafter). Drawn on the chart, and deliberately absent from `rows`,
// `ratio_vs_best`, `wins` and `summary` below: the published claim is Atlas
// against the best vLLM at each rung, and admitting a second Atlas
// configuration would change what that number means. A variant is evidence
// about Atlas, not evidence about the comparison.
//
// It IS held to the same rung coverage as a baseline, for the same reason: a
// line that stops partway along a log2 axis reads as a measurement, not as a
// gap.
const variants = series.filter((s) => s.role === 'variant');
for (const v of variants) {
  for (const row of subject.rungs) {
    if (!v.rungs.some((r) => r.c === row.c))
      die(`variant ${v.id} is missing rung C=${row.c}`);
  }
}
const KNOWN_ROLES = new Set(['subject', 'baseline', 'variant']);
for (const s2 of series) {
  if (!KNOWN_ROLES.has(s2.role))
    die(`series ${s2.id} has unknown role ${JSON.stringify(s2.role)}`);
}

const at = (s, c) => s.rungs.find((r) => r.c === c);

// Every rung the subject measured must exist in every baseline, or the
// comparison silently changes shape partway along the x-axis.
const rows = subject.rungs.map((row) => {
  const perBaseline = baselines.map((b) => {
    const r = at(b, row.c);
    if (!r) die(`baseline ${b.id} is missing rung C=${row.c}`);
    return { id: b.id, label: b.label, parity: b.parity, tok_s: r.tok_s };
  });
  const best = perBaseline.reduce((a, b) => (b.tok_s > a.tok_s ? b : a));
  const m = perBaseline.find((b) => b.id === matched.id);
  return {
    c: row.c,
    atlas: row.tok_s,
    baselines: perBaseline,
    best_baseline_id: best.id,
    ratio_vs_best: r3(row.tok_s / best.tok_s),
    ratio_vs_matched: r3(row.tok_s / m.tok_s),
    wins: row.tok_s > best.tok_s
  };
});

// The campaign driver stamps a sha256 of its own source into every record it
// writes, and the published rungs carry the hash of the copy that produced
// them. Hash the copy that ships in the tree so the deck can put the two side
// by side: a reader who runs the repo's driver gets THIS hash in their output,
// and a page that quoted only the recorded one would be inviting them to
// compare against bytes they do not have. Computed, never typed — if the file
// is ever edited this moves with it.
const harnessRepoSha256 = createHash('sha256')
  .update(readFileSync(resolve(REPO, manifest.workload.harness)))
  .digest('hex');

const out = {
  generated_utc: new Date().toISOString().replace(/\.\d+Z$/, 'Z'),
  title: manifest.title,
  subtitle: manifest.subtitle,
  aggregate: manifest.aggregate,
  results_doc: manifest.results_doc,
  results_doc_url: `https://github.com/Avarok-Cybersecurity/atlas/blob/main/${manifest.results_doc}`,
  workload: manifest.workload,
  box: manifest.box,
  harness_shas: manifest.harness_shas,
  harness_repo_sha256: harnessRepoSha256,
  concurrencies: subject.rungs.map((r) => r.c),
  series,
  rows,
  // Claim strength is derived, never asserted: if a future measurement loses a
  // rung, the page says so instead of the copy going stale.
  summary: {
    rungs: rows.length,
    won: rows.filter((r) => r.wins).length,
    all_won: rows.every((r) => r.wins),
    min_ratio: r3(Math.min(...rows.map((r) => r.ratio_vs_best))),
    max_ratio: r3(Math.max(...rows.map((r) => r.ratio_vs_best)))
  }
};

writeStable(OUT, out, ['generated_utc'], (o) => `${JSON.stringify(o, null, 2)}\n`);
console.log(
  `gen-ladder: ${out.summary.won}/${out.summary.rungs} rungs won ` +
    `(${out.summary.min_ratio}x..${out.summary.max_ratio}x vs best baseline) -> ${OUT}`
);
