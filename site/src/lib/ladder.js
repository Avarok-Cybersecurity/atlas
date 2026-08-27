// SPDX-License-Identifier: AGPL-3.0-only

// Derived readings of the concurrency ladder.
//
// The site's claim policy forbids hand-typed performance numbers, and a
// percentage computed once and pasted into prose is exactly that: it stops
// tracking the artifact the moment the ladder is regenerated. Everything here
// is computed from `ladder.generated.json` at build time instead.

/** Percentage change from `a` to `b`, one decimal place. */
const change = (a, b) => Math.round(((b - a) / a) * 1000) / 10;

/**
 * How each engine scaled across the top two rungs.
 *
 * The baseline compared is whichever vLLM configuration *leads the lower
 * rung*, not whichever leads each rung independently. That matters: at C=128
 * vLLM's speculative build falls behind its own non-speculative one, so
 * "best at each rung" would silently switch configurations mid-comparison and
 * describe a scaling curve no single deployment ever had.
 *
 * Returns null when the ladder has fewer than two rungs or the lower rung's
 * leading baseline was not run at the upper rung, because a missing datapoint
 * is not a result and this must not invent one.
 */
export function headroom(rungs) {
  if (!Array.isArray(rungs) || rungs.length < 2) return null;

  const sorted = [...rungs].sort((a, b) => a.c - b.c);
  const from = sorted.at(-2);
  const to = sorted.at(-1);

  const id = from.best_baseline_id;
  const lower = from.baselines?.find((b) => b.id === id);
  const upper = to.baselines?.find((b) => b.id === id);
  if (!lower || !upper) return null;

  return {
    from: from.c,
    to: to.c,
    label: lower.label,
    atlas: change(from.atlas, to.atlas),
    baseline: change(lower.tok_s, upper.tok_s),
    ratio: to.ratio_vs_best
  };
}

/** `+23.7%` / `-0.8%`, for prose that must read the sign out loud. */
export const signed = (pct) => `${pct >= 0 ? '+' : ''}${pct.toFixed(1)}%`;
