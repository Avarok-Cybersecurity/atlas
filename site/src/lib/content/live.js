// SPDX-License-Identifier: AGPL-3.0-only
//
// The live numbers. Every performance figure the marketing pages print comes
// through here, from the same generated JSON the developer page reads, so a
// regenerated ladder rewrites the front page rather than leaving prose that
// asserts a number the repository no longer holds.
//
// `fill(text)` replaces {placeholders} in copy strings. A placeholder with no
// value is left visible in braces on purpose, which is easier to see in review
// than a silent blank.
import ladder from '$lib/ladder.generated.json';
import counts from '$lib/live.generated.json';
import stars from '$lib/stars.generated.json';
import { benchmarkHighlight } from '$lib/marketing.js';
import { fleetModel, apiModel, paybackLabel } from '$lib/economics.js';
import { company } from './brand.js';

const highlight = benchmarkHighlight(ladder);
const rows = [...ladder.rows].sort((a, b) => a.c - b.c);
const top = rows[rows.length - 1];

// Two counts come from files too large to put on a marketing page: the 1 MB
// gate record set and the model catalogue. scripts/gen-live.mjs reduces them to
// the integers printed here, in live.generated.json. Do not import
// gates.generated.json from anything a marketing page loads: it becomes a 1 MB
// chunk on every one of them (bundle-budget.test.js holds that line).
const recipeCount = counts.recipes;
const gatePass = counts.gates.concurrencyPass;

const fleet = fleetModel();
const api = apiModel({ boxTokensPerSecond: top.atlas });

export const live = {
  engine: company.engine,
  ratio: `${highlight.ratio.toFixed(3)}×`,
  ratioPlain: highlight.ratio.toFixed(3),
  c: String(highlight.concurrency),
  atlasTop: top.atlas.toFixed(2),
  baselineTop: highlight.baseline.toFixed(2),
  baselineLabel: highlight.baselineLabel,
  won: String(ladder.summary?.won ?? rows.length),
  rungs: String(ladder.summary?.rungs ?? rows.length),
  checkpoint: ladder.workload.checkpoint,
  gpu: ladder.box.gpu,
  gatePass: String(gatePass),
  recipes: String(recipeCount),
  stars: String(stars.count),
  payback: paybackLabel(fleet.paybackMonths),
  apiSavings: `${Math.round(api.savingsPct)}%`,
  stamp: `${ladder.series?.[0]?.build_public ?? ''} · ${ladder.generated_utc?.slice(0, 10) ?? ''}`.trim()
};

export const ladderData = ladder;
export const highlightData = highlight;

export function fill(text) {
  if (typeof text !== 'string') return text;
  return text.replace(/\{(\w+)\}/g, (m, key) => (key in live ? live[key] : m));
}
