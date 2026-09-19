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

/**
 * Format an uptime given in seconds, coarsely — an uptime is read for its
 * order of magnitude, not its precision.
 *
 * @param {number|null|undefined} v seconds
 * @returns {string}
 */
export function uptime(v) {
  if (!Number.isFinite(v) || v < 0) return '—';
  const s = Math.floor(v);
  if (s < 60) return `${s}s`;
  const m = Math.floor(s / 60);
  if (m < 60) return `${m}m`;
  const h = Math.floor(m / 60);
  if (h < 48) return `${h}h ${m % 60}m`;
  return `${Math.floor(h / 24)}d ${h % 24}h`;
}

/**
 * Slots a held sparkline leaves empty at its right edge, so "the page stopped
 * asking" is visible as the line stopping short of now.
 */
export const HOLD_GAP = 12;

/**
 * A history pinned to the strip's fixed time axis of `HISTORY` slots.
 *
 * Samples fill from the left, so a young session's line ends short of the
 * right edge instead of being stretched across it — the x-axis means time,
 * not "however long we happen to have been looking".
 *
 * `held` is required: when the operator paused polling, or the launch stopped
 * answering, the line must end visibly short of the right edge — a line
 * touching "now" claims the last pixel is current, which is exactly what a
 * held reading is not. When the natural padding already leaves the gap,
 * nothing is moved.
 *
 * @param {Array<number|null>} history
 * @param {{held: boolean}} opts
 * @returns {Array<number|null>} exactly `HISTORY` entries
 */
export function timeline(history, opts) {
  if (opts?.held !== true && opts?.held !== false) {
    throw new TypeError('timeline must be told whether the reading is held');
  }
  const list = Array.isArray(history) ? history : [];
  const win = list.length > HISTORY ? list.slice(-HISTORY) : list;
  const padded =
    win.length < HISTORY ? [...win, ...new Array(HISTORY - win.length).fill(null)] : [...win];
  if (!opts.held) return padded;
  let trailing = 0;
  for (let i = padded.length - 1; i >= 0 && padded[i] == null; i--) trailing++;
  if (trailing >= HOLD_GAP) return padded;
  const shift = HOLD_GAP - trailing;
  return [...padded.slice(shift), ...new Array(shift).fill(null)];
}
