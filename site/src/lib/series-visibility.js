// SPDX-License-Identifier: AGPL-3.0-only
//
// Which series of a ladder are drawn. The legend pills on the concurrency
// comparison toggle a series in and out of the chart and the table; this is
// the one place that rule lives, so the pills, the SVG and the table cannot
// disagree about it.
//
// One rule: at least one series stays drawn. Hiding the last visible series
// is REFUSED, and the refusal is returned as text for the page to print —
// a pill that does nothing when pressed is the defect this module replaces.

/** The series of `series` not listed in `hidden`, in their original order. */
export const visibleOf = (series, hidden) => series.filter((s) => !hidden.includes(s.id));

/**
 * Toggle `id` in the hidden set.
 *
 * @param {{ id: string, label: string }[]} series every series the ladder carries
 * @param {string[]} hidden ids currently hidden
 * @param {string} id the series the pill stands for
 * @returns {{ hidden: string[], refused: string | null }} the new hidden set,
 *   and the reason the request was refused when it was (then `hidden` is
 *   unchanged).
 */
export function toggleSeries(series, hidden, id) {
  const target = series.find((s) => s.id === id);
  if (!target) throw new Error(`toggleSeries: unknown series ${JSON.stringify(id)}`);
  if (hidden.includes(id)) return { hidden: hidden.filter((h) => h !== id), refused: null };
  const visible = visibleOf(series, hidden);
  if (visible.length === 1) {
    return {
      hidden,
      refused: `${target.label} stays drawn — it is the only series left. Show another series before hiding it.`
    };
  }
  return { hidden: [...hidden, id], refused: null };
}
