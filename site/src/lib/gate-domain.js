// SPDX-License-Identifier: AGPL-3.0-only

// gate-domain.js — axis policy for the gate dashboard's charts.
//
// Two problems this solves, both visible on `ttft-cold-gate`:
//
//   * ONE EXTREME POINT FLATTENS THE CHART. A raw min..max domain let a single
//     23,484 ms cold-start stretch the axis to ~29k and squash seventy normal
//     readings into a band a few pixels tall. The domain here is cut from
//     percentiles instead, and anything outside it is CLAMPED AND MARKED —
//     never dropped. A chart that quietly deletes a measurement to look tidy
//     is worse than one that is hard to read.
//
//   * A MILLISECOND AXIS STARTED BELOW ZERO (-81 ms, -2,330 ms), because the
//     padding was applied blind. The floor here is derived from the data: if
//     nothing observed is negative, the axis does not go negative either.
//
// Pure and dependency-free so `bun test` can measure it directly.

/**
 * Linear-interpolated percentile of an ASCENDING-sorted array.
 * @param {number[]} sorted
 * @param {number} p in [0, 1]
 */
export function percentile(sorted, p) {
  if (sorted.length === 0) return NaN;
  if (sorted.length === 1) return sorted[0];
  const idx = p * (sorted.length - 1);
  const lo = Math.floor(idx);
  const hi = Math.ceil(idx);
  return sorted[lo] + (sorted[hi] - sorted[lo]) * (idx - lo);
}

/**
 * How far past the robust band a tail has to reach before the axis stops
 * following it. Below this the true extreme is used and nothing is clipped, so
 * a clipped marker always means something unusual rather than being routine.
 */
const TAIL_DOMINATES = 0.75;

/** Headroom added on each side once the range is settled. */
const PAD = 0.08;

/**
 * Robust y-domain for one chart.
 *
 * @param {number[]} values every value that will be DRAWN (aggregated medians
 *   and singletons alike). Deliberately not the raw members: an extreme hidden
 *   inside a group must not re-inflate the axis that aggregation just settled.
 * @param {Array<{value: number}>} refLines panel floors and caps, read from
 *   records upstream — a cap is the gate's meaning and is never clipped out of
 *   view.
 * @returns {{v0: number, v1: number, clipHigh: boolean, clipLow: boolean}|null}
 *   null when there is nothing to draw.
 */
export function robustDomain(values, refLines = []) {
  const finite = values.filter((v) => Number.isFinite(v));
  if (finite.length === 0) return null;

  const sorted = [...finite].sort((a, b) => a - b);
  const q05 = percentile(sorted, 0.05);
  const q95 = percentile(sorted, 0.95);
  const min = sorted[0];
  const max = sorted[sorted.length - 1];

  // The yardstick for "how far out is far out". When the robust band is flat —
  // a series that sits on one number and spikes once, which is precisely the
  // shape that most needs clipping — fall back to a small fraction of the
  // level, so a degenerate band does not silently switch clipping off.
  const span = q95 - q05 || Math.abs(q95) * 0.02 || 1;

  const clipHigh = max - q95 > TAIL_DOMINATES * span;
  const clipLow = q05 - min > TAIL_DOMINATES * span;
  let hi = clipHigh ? q95 : max;
  let lo = clipLow ? q05 : min;

  // A reference line is part of the gate's claim; widen for it rather than
  // clipping it. Done before padding so the line never sits flush on the frame.
  const refs = refLines.map((l) => l.value).filter((v) => Number.isFinite(v));
  for (const r of refs) {
    hi = Math.max(hi, r);
    lo = Math.min(lo, r);
  }

  const pad = PAD * (hi - lo || Math.abs(hi) || 1);
  let v0 = lo - pad;
  const v1 = hi + pad;

  // Floor from the DATA, not from a unit string: a metric that never goes
  // negative gets an axis that never goes negative, and one that legitimately
  // does keeps its negative range.
  if (min >= 0 && refs.every((r) => r >= 0)) v0 = Math.max(0, v0);

  return { v0, v1, clipHigh, clipLow };
}

/**
 * Where to draw a value, and whether that position is a lie about its size.
 *
 * Never mutates its input: the true value stays on the node so the tooltip,
 * the aria label and the record card can all still report it.
 *
 * @param {number} v
 * @param {{v0: number, v1: number}} domain
 * @returns {{y: number, clamped: 'low'|'high'|null}}
 */
export function clampValue(v, domain) {
  if (v > domain.v1) return { y: domain.v1, clamped: 'high' };
  if (v < domain.v0) return { y: domain.v0, clamped: 'low' };
  return { y: v, clamped: null };
}

/**
 * Tick label for a clipped edge — `1,240+` reads as "and beyond" at a glance
 * and needs no glyph outside the mono font already in use.
 * @param {string} text already-formatted tick
 * @param {'high'|'low'|null} edge
 */
export const tickLabel = (text, edge) =>
  edge === 'high' ? `${text}+` : edge === 'low' ? `${text}−` : text;

/**
 * Spread end-of-series labels apart so none overlaps another.
 *
 * Pool-adjacent-violators: repeatedly merge clusters that are closer than one
 * label height and re-centre each cluster on the mean of its members' wanted
 * positions, which is the placement that minimises total displacement. Then
 * push any cluster that has left the field back inside and settle again.
 *
 * @param {number[]} desired wanted centre of each label, any order
 * @param {{height: number, top: number, bottom: number}} box
 * @returns {number[]} placed centres, in the SAME order as `desired`
 */
export function dodgeLabels(desired, { height, top, bottom }) {
  const order = desired.map((y, i) => ({ y, i })).sort((a, b) => a.y - b.y);
  // Each cluster: the labels it holds, and the mean position they want.
  let clusters = order.map((o) => ({ items: [o], want: o.y }));

  const settle = () => {
    let merged = true;
    while (merged) {
      merged = false;
      for (let i = 0; i < clusters.length - 1; i += 1) {
        const a = clusters[i];
        const b = clusters[i + 1];
        const aEnd = a.want + ((a.items.length - 1) / 2) * height;
        const bStart = b.want - ((b.items.length - 1) / 2) * height;
        if (bStart - aEnd < height) {
          const items = [...a.items, ...b.items];
          const want = items.reduce((s, it) => s + it.y, 0) / items.length;
          clusters.splice(i, 2, { items, want });
          merged = true;
          break;
        }
      }
    }
  };

  settle();
  // Keep every cluster inside the plot field, then re-settle in case shifting
  // one into range pushed it into its neighbour.
  for (let pass = 0; pass < clusters.length + 1; pass += 1) {
    let moved = false;
    for (const c of clusters) {
      const half = ((c.items.length - 1) / 2) * height;
      const lo = c.want - half;
      const hi = c.want + half;
      if (lo < top) {
        c.want += top - lo;
        moved = true;
      } else if (hi > bottom) {
        c.want -= hi - bottom;
        moved = true;
      }
    }
    if (!moved) break;
    settle();
  }

  const placed = new Array(desired.length);
  for (const c of clusters) {
    const start = c.want - ((c.items.length - 1) / 2) * height;
    c.items.forEach((it, k) => {
      placed[it.i] = start + k * height;
    });
  }
  return placed;
}
