// SPDX-License-Identifier: AGPL-3.0-only

// What the page's single live region says, and when it says nothing.
//
// Pure and plain `.js` for the house reason: "announce severity CHANGES only"
// is a testable rule, and a file holding runes cannot be imported by the test
// runner. The bridge has exactly one aria-live region — a screen reader that
// hears every alert re-render narrates a dashboard instead of a change — so
// the decision of whether this render is worth interrupting someone lives
// here, beside its tests, and the svelte layer only owns the timer.

import { sanitize } from './ingest.js';

/**
 * How long the surface lets the fleet settle before speaking. An alert storm
 * that escalates twice within the window is announced once, at its worst.
 */
export const ANNOUNCE_DEBOUNCE_MS = 1500;

const SEVERITIES = new Set(['critical', 'warning', 'info']);

/**
 * What to announce given the worst severity last announced and the current
 * alerts (worst first, as `fleet.alerts` sorts them).
 *
 * Null means stay quiet: the worst severity has not changed, and re-reading
 * the same fact louder is noise. Any transition — first alert, escalation,
 * de-escalation, all clear — speaks once, verbatim from the alert.
 *
 * @param {string|null} prevSeverity what was last announced, null for none
 * @param {{severity: string, nodeName?: string, kind?: string, detail?: string}[]} alerts
 * @returns {{severity: string|null, text: string}|null}
 */
export function announcement(prevSeverity, alerts) {
  const prev = SEVERITIES.has(prevSeverity) ? prevSeverity : null;
  const worst = Array.isArray(alerts) ? (alerts[0] ?? null) : null;
  const next = worst && SEVERITIES.has(worst.severity) ? worst.severity : null;
  if (next === prev) return null;
  if (next === null) return { severity: null, text: 'All alerts cleared.' };
  const kind = typeof worst.kind === 'string' ? worst.kind.replaceAll('_', ' ') : '';
  const what = sanitize(worst.detail ?? '', 200) || kind || 'alert';
  const who = sanitize(worst.nodeName ?? '', 63) || 'a machine';
  return { severity: next, text: `${next}: ${who}: ${what}` };
}
