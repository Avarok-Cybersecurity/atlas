// SPDX-License-Identifier: AGPL-3.0-only

// tablist.js — keyboard movement for the record card's tabs.
//
// Split out of the component because it is decidable logic with edge cases
// (wrapping, single-tab lists, keys that are none of its business) and the
// repo keeps that kind of thing in a module a test can reach.

/**
 * Where a key press should move the selection in a tablist.
 *
 * Follows the ARIA authoring practice: left/right wrap around, Home and End
 * jump to the ends, and every other key is left alone so the dialog's own
 * handlers (Escape, Tab) still see it.
 *
 * @param {string} key
 * @param {number} index currently selected
 * @param {number} count
 * @returns {number|null} the new index, or null when the key is not ours
 */
export function moveTab(key, index, count) {
  if (count <= 1) return null;
  if (key === 'ArrowRight') return (index + 1) % count;
  if (key === 'ArrowLeft') return (index - 1 + count) % count;
  if (key === 'Home') return 0;
  if (key === 'End') return count - 1;
  return null;
}
