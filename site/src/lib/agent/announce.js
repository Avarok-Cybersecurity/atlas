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

/**
 * A debouncer that survives being re-run.
 *
 * The component's effect re-runs on every vitals event — `fleet.alerts` derives
 * from `fleet.nodes`, which is rebuilt about once a second — so the timer
 * cannot live in the effect's cleanup. It did, and the sequence was:
 *
 *   1. an alert is raised; the effect advances `prevSeverity` and arms a timer
 *   2. a vitals event arrives inside the debounce window
 *   3. the effect re-runs, and its cleanup CLEARS the pending timer
 *   4. the severity has already advanced, so `announcement` returns null and
 *      nothing re-arms it
 *
 * The live region then never fires — measured: still empty 5.7 s after a
 * critical alert, on a wire with ordinary 1 Hz telemetry. Quiet wires
 * announced fine, which is why it survived every hand test.
 *
 * So the timer is owned here. It is cleared only when a NEW announcement
 * supersedes a pending one, or when the caller disposes it.
 *
 * @param {(text: string) => void} emit called with the text to announce
 * @param {{setTimeout: Function, clearTimeout: Function}} [timers] injected for
 *   tests, which must not wait 1.5 real seconds. Wrapped, never destructured:
 *   a browser's `setTimeout` is a `Window` method and rejects a foreign `this`.
 */
export function makeAnnouncer(emit, timers) {
  const t = timers ?? {
    setTimeout: (fn, ms) => setTimeout(fn, ms),
    clearTimeout: (h) => clearTimeout(h)
  };
  let handle = null;
  let prevSeverity = null;

  return {
    /** Feed the current alerts. Safe to call on every render. */
    update(alerts) {
      const a = announcement(prevSeverity, alerts);
      if (!a) return;
      prevSeverity = a.severity;
      // A newer transition replaces a pending one: a storm that escalates
      // twice inside the window is read once, at its worst.
      t.clearTimeout(handle);
      handle = t.setTimeout(() => emit(a.text), ANNOUNCE_DEBOUNCE_MS);
    },
    /** Drop any pending announcement. For component teardown only. */
    dispose() {
      t.clearTimeout(handle);
      handle = null;
    }
  };
}
