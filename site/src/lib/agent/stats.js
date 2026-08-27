// SPDX-License-Identifier: AGPL-3.0-only

// Turning readings into something a person can read at a glance.
//
// **Absent is not zero, all the way to the pixel.** Every formatter here
// returns a dash for a missing value rather than "0", and every tile decides
// whether it has data before it decides what colour to be. A dashboard that
// renders 0 tok/s for "not measured yet" teaches an operator to distrust it,
// and the one time throughput really is zero they will not believe it.

/** How many samples the sparkline keeps. At ~1 Hz this is a few minutes. */
export const HISTORY = 180;

/**
 * Format a rate.
 *
 * @param {number|null|undefined} v
 * @returns {string}
 */
export function tokens(v) {
  if (!Number.isFinite(v)) return '—';
  if (v >= 100) return v.toFixed(0);
  if (v >= 10) return v.toFixed(1);
  return v.toFixed(2);
}

/**
 * Format a duration given in seconds, choosing the unit a human would.
 *
 * @param {number|null|undefined} v seconds
 * @returns {string}
 */
export function duration(v) {
  if (!Number.isFinite(v)) return '—';
  if (v < 1) return `${Math.round(v * 1000)} ms`;
  return `${v.toFixed(2)} s`;
}

/**
 * Format a 0..1 share as a percentage.
 *
 * @param {number|null|undefined} v
 * @returns {string}
 */
export function percent(v) {
  if (!Number.isFinite(v)) return '—';
  return `${Math.round(v * 100)}%`;
}

/**
 * Format a count.
 *
 * @param {number|null|undefined} v
 * @returns {string}
 */
export function count(v) {
  if (!Number.isFinite(v)) return '—';
  return Math.round(v).toLocaleString('en');
}

/**
 * Append a sample to a bounded history.
 *
 * A missing reading is pushed as `null` rather than skipped, so the sparkline
 * shows a gap where the model was not answering instead of drawing a straight
 * line across it as though nothing happened.
 *
 * @param {Array<number|null>} history
 * @param {number|null|undefined} value
 * @returns {Array<number|null>} a new array
 */
export function push(history, value) {
  const next = [...history, Number.isFinite(value) ? value : null];
  return next.length > HISTORY ? next.slice(next.length - HISTORY) : next;
}

/**
 * An SVG path across a history, scaled to a box.
 *
 * Returns `''` when there is nothing to draw. Gaps break the path rather than
 * being interpolated through, for the same reason `push` records them.
 *
 * @param {Array<number|null>} history
 * @param {number} w
 * @param {number} h
 * @returns {string}
 */
export function sparkline(history, w, h) {
  const points = history.filter((v) => v != null);
  if (points.length < 2) return '';
  const max = Math.max(...points, Number.EPSILON);
  const step = w / Math.max(history.length - 1, 1);
  let d = '';
  let pen = false;
  history.forEach((v, i) => {
    if (v == null) {
      pen = false;
      return;
    }
    // A flat zero line still sits on the floor rather than in the middle.
    const y = h - (v / max) * h;
    d += `${pen ? 'L' : 'M'}${(i * step).toFixed(2)} ${y.toFixed(2)} `;
    pen = true;
  });
  return d.trim();
}

/**
 * Whether a reading says anything at all.
 *
 * Used to tell "the model is still loading" from "the model is idle": an
 * engine that answers with every field absent is not the same as one that is
 * not answering, and the page says which.
 *
 * @param {object|null|undefined} stats
 * @returns {boolean}
 */
export function hasAnything(stats) {
  if (stats == null) return false;
  return Object.values(stats).some((v) => Number.isFinite(v));
}
