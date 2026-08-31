// SPDX-License-Identifier: AGPL-3.0-only

// chart-marks.js — the marker vocabulary shared by the gate charts and their
// legend.
//
// Five things a point can be, and each is told apart by SHAPE rather than by
// colour alone, so the chart still reads under any colour deficiency and in
// print:
//
//   solid disc      a passing run
//   open ring       a failing run              (hollow = something is missing)
//   disc + halo     a group of runs            (it has company)
//   filled triangle a point with no trend line (a corner, not a curve)
//   open caret      a value beyond the axis    (no base: it is not a datum,
//                                               it is an arrow off the edge)
//
// The triangle and the caret are the pair most at risk of being confused, so
// they differ in two ways at once: the triangle is FILLED and sits wherever
// its value is; the caret is STROKED, has no base edge, and only ever appears
// touching the top or bottom of the plot field.
//
// The generators live here rather than inline in the components so the legend
// key and the plotted mark cannot drift apart — a legend that stops matching
// the chart is worse than no legend.

/** Radius of the circumscribed circle for the lone-point triangle. */
export const LONE_R = 4.5;

/** How far the caret's arms drop below its apex. */
export const CARET_RISE = 5;
/** Half-width of the caret. */
export const CARET_HALF = 4.5;

const n = (v) => Number(v.toFixed(2));

/**
 * An equilateral triangle, point up, centred on its circumcircle.
 * @param {number} cx
 * @param {number} cy
 * @param {number} [r]
 */
export function loneTriangle(cx, cy, r = LONE_R) {
  const dx = n(r * Math.sin(Math.PI / 3));
  const dy = n(r * 0.5);
  return `M${n(cx)} ${n(cy - r)} L${n(cx + dx)} ${n(cy + dy)} L${n(cx - dx)} ${n(cy + dy)} Z`;
}

/**
 * An open caret pinned to a clipped edge, apex pointing off the chart.
 * @param {number} cx
 * @param {number} y the clamp row (the axis edge the value ran past)
 * @param {'high'|'low'} edge
 */
export function clipCaret(cx, y, edge) {
  const dir = edge === 'low' ? -1 : 1;
  const apex = n(y + dir * -CARET_RISE * 0.4);
  const base = n(y + dir * CARET_RISE * 0.6);
  return `M${n(cx - CARET_HALF)} ${base} L${n(cx)} ${apex} L${n(cx + CARET_HALF)} ${base}`;
}
