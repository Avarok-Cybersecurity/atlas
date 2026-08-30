// SPDX-License-Identifier: AGPL-3.0-only

// gate-band.js — collapse a pile of historical ladder runs into one region.
//
// The concurrency panel used to draw every run it had as its own polyline,
// with older ones at 28% opacity. At ~40 runs that is not a chart, it is a
// smear: the fade was trying to say "these are context", but forty faded
// strokes still add up to solid ink, and nothing about the shape survives.
//
// History becomes a REGION instead — the observed range at each rung — with
// only the newest run and the one before it left as lines. That is the
// standard systems-paper idiom, and it answers the question the fade was
// reaching for ("is today's line where it has been?") far better, because the
// eye can actually see the corridor.
//
// Pure and dependency-free so `bun test` can measure it directly.

import { percentile } from './gate-domain.js';

/**
 * Above this many historical runs the band shows the 10th-90th percentile
 * rather than the full min-max.
 *
 * With a handful of runs, min-max IS the honest summary — throwing away the
 * extremes of five samples hides most of what there is to know. Once there are
 * many, one freak run would otherwise inflate the corridor for every rung and
 * make the band claim more variance than the engine actually has.
 */
export const BAND_PERCENTILE_FROM = 8;

/**
 * The historical corridor, one entry per rung, ascending by concurrency.
 *
 * @param {Array<{pts: Array<{c: number, v: number}>}>} runs the OLDER runs only
 * @returns {Array<{c: number, lo: number, hi: number}>} rungs measured by at
 *   least two runs; a rung with a single observation has no range to draw and
 *   is skipped rather than rendered as a zero-height sliver.
 */
export function historyBand(runs) {
  const byRung = new Map();
  for (const run of runs) {
    for (const p of run.pts ?? []) {
      if (!Number.isFinite(p.v)) continue;
      if (!byRung.has(p.c)) byRung.set(p.c, []);
      byRung.get(p.c).push(p.v);
    }
  }
  const out = [];
  for (const [c, values] of byRung) {
    if (values.length < 2) continue;
    const sorted = [...values].sort((a, b) => a - b);
    const wide = sorted.length < BAND_PERCENTILE_FROM;
    out.push({
      c,
      lo: wide ? sorted[0] : percentile(sorted, 0.1),
      hi: wide ? sorted[sorted.length - 1] : percentile(sorted, 0.9)
    });
  }
  return out.sort((a, b) => a.c - b.c);
}

/**
 * Split one variant's runs into the two that stay lines and the rest.
 *
 * `runs` must be chronological. The previous run is kept as a line because it
 * is what makes "did the newest run move?" readable at a glance — the single
 * comparison the old fade was actually useful for.
 *
 * @param {Array<object>} runs chronological
 * @returns {{latest: object|null, previous: object|null, history: Array<object>}}
 */
export function splitHistory(runs) {
  const n = runs.length;
  return {
    latest: n > 0 ? runs[n - 1] : null,
    previous: n > 1 ? runs[n - 2] : null,
    history: n > 2 ? runs.slice(0, n - 2) : []
  };
}

/** An SVG path enclosing the corridor: upper edge left to right, then back. */
export function bandPath(band, x, y) {
  if (band.length < 2) return '';
  const top = band.map((b, i) => `${i ? 'L' : 'M'}${x(b.c).toFixed(1)} ${y(b.hi).toFixed(1)}`);
  const bottom = [...band].reverse().map((b) => `L${x(b.c).toFixed(1)} ${y(b.lo).toFixed(1)}`);
  return `${top.join(' ')} ${bottom.join(' ')} Z`;
}
