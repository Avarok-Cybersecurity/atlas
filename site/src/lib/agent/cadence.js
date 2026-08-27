// SPDX-License-Identifier: AGPL-3.0-only

// When to ask which node for stats.
//
// Pure and plain `.js` for the house reason: the scheduling rules are the
// testable part, and a file holding runes cannot be imported by the test
// runner. Nothing here owns a timer — the caller injects the clock by passing
// `nowMs`, and the svelte layer merely does what `due` says.
//
// Two rules shape the schedule:
//
// **Only running launches are polled.** LaunchStats answers for a launch;
// a node with nothing serving has nothing to ask, and polling it anyway
// manufactures error chatter that then reads as a sick machine.
//
// **Offsets are staggered.** Stats for a remote node are forwarded through a
// relay, so a page that fires every poll on the same tick makes one machine
// fan out simultaneous forwarded requests. Every entry gets a distinct offset
// inside its period, so no two polls from this page are ever due on the same
// instant of the shared epoch.

/** What the segmented control offers. `ms: null` is Pause. */
export const CADENCES = [
  { id: 'pause', ms: null },
  { id: '1s', ms: 1000 },
  { id: '2s', ms: 2000 },
  { id: '5s', ms: 5000 }
];

/** How often an unselected running node is polled for the roster micro-columns. */
export const UNSELECTED_MS = 10_000;

/** Backoff on error never grows past this: a recovering agent is noticed within a minute. */
export const BACKOFF_MAX_MS = 60_000;

/** Whether a cadence value means "do not poll at all". */
export function isPaused(cadenceMs) {
  return cadenceMs === null;
}

/**
 * The valid cadence periods, derived from the control's own options so the
 * gate cannot drift from what the UI offers.
 */
const VALID_MS = new Set(CADENCES.map((c) => c.ms).filter((ms) => ms !== null));

/**
 * Build the poll plan: which node is polled at which period, at which offset.
 *
 * Paused means paused for everything — the operator asked the page to go
 * quiet, and background 10s polls that keep flowing would keep a relay busy
 * on behalf of a page claiming to be paused.
 *
 * @param {{id: string, selected: boolean, running: boolean}[]} nodes
 * @param {number|null} cadenceMs one of `CADENCES`; null pauses all polling
 * @returns {{id: string, periodMs: number, offsetMs: number}[]}
 */
export function plan(nodes, cadenceMs) {
  if (cadenceMs === null) return [];
  if (!VALID_MS.has(cadenceMs)) {
    // A typo'd 20ms here is a page hammering an agent 50 times a second, so
    // an unknown cadence is refused rather than obeyed.
    throw new TypeError(`not a cadence this page offers: ${String(cadenceMs)}`);
  }
  const list = (Array.isArray(nodes) ? nodes : []).filter((n) => n && n.running === true);
  const selected = list.filter((n) => n.selected === true);
  const rest = list.filter((n) => n.selected !== true).sort((a, b) => a.id.localeCompare(b.id));

  const out = [];
  // The selected node anchors the schedule at offset 0.
  for (const n of selected) out.push({ id: n.id, periodMs: cadenceMs, offsetMs: 0 });
  // The rest are spread over their own period, skipping offset 0 so none of
  // them coincides with the selected node's anchor tick. Sorted by id first,
  // so the same fleet always gets the same offsets and a re-plan does not
  // reshuffle everyone's phase.
  rest.forEach((n, i) => {
    out.push({
      id: n.id,
      periodMs: UNSELECTED_MS,
      offsetMs: Math.round(((i + 1) * UNSELECTED_MS) / (rest.length + 1))
    });
  });
  return out;
}

/**
 * When this entry should next be polled, in epoch ms.
 *
 * A node never polled is due at its offset from the schedule epoch — not
 * immediately, or eight nodes would all fire on the first tick and the
 * stagger would only exist from the second round onward.
 *
 * Consecutive failures back the entry off exponentially from its period,
 * capped: a node that is down stops being asked every second, and one that
 * comes back is noticed within `BACKOFF_MAX_MS`.
 *
 * @param {{periodMs: number, offsetMs: number}} entry
 * @param {{lastAt: number|null, failures: number}|undefined} state
 * @param {number} epochMs when this schedule began
 * @returns {number}
 */
export function nextDue(entry, state, epochMs) {
  if (!state || state.lastAt === null) return epochMs + entry.offsetMs;
  const failures = state.failures ?? 0;
  const delay =
    failures > 0 ? Math.min(entry.periodMs * 2 ** failures, BACKOFF_MAX_MS) : entry.periodMs;
  return state.lastAt + delay;
}

/**
 * Which nodes are due for a poll right now, soonest first.
 *
 * @param {{id: string, periodMs: number, offsetMs: number}[]} entries from `plan`
 * @param {Record<string, {lastAt: number|null, failures: number}>} states
 * @param {number} nowMs
 * @param {number} epochMs
 * @returns {string[]} node ids
 */
export function due(entries, states, nowMs, epochMs) {
  return entries
    .map((e) => ({ id: e.id, at: nextDue(e, states[e.id], epochMs) }))
    .filter((e) => e.at <= nowMs)
    .sort((a, b) => a.at - b.at || a.id.localeCompare(b.id))
    .map((e) => e.id);
}

/**
 * Record a successful poll. Returns a new state map; failures reset, because
 * backoff exists to protect a struggling agent, not to punish a recovered one.
 */
export function polled(states, id, nowMs) {
  return { ...states, [id]: { lastAt: nowMs, failures: 0 } };
}

/** Record a failed poll. Returns a new state map with the failure counted. */
export function failed(states, id, nowMs) {
  const failures = (states[id]?.failures ?? 0) + 1;
  return { ...states, [id]: { lastAt: nowMs, failures } };
}
