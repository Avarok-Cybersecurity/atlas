// SPDX-License-Identifier: AGPL-3.0-only

// The registry of everything the page admits it cannot do yet.
//
// Pure and plain `.js` for the house reason: the caps and the solo-mode
// collapse are testable rules, and a file holding runes cannot be imported by
// the test runner.
//
// A placeholder is the fourth honesty class: dashed outline, `soon` chip, and
// NO value — not even an em-dash, because the dash means "this hardware
// cannot answer", which would be a claim about a question that was never
// asked. Each entry names the specific missing capability in one sentence,
// shown in a popover, so "soon" is a statement rather than marketing.
//
// The per-region caps exist because placeholders multiply: each one alone is
// honest, and a console covered in them reads as a roadmap. The cap is
// enforced HERE, where entries are registered, so adding a fourth actions
// chip fails a test instead of quietly densifying the UI.
//
// Deliberately absent, even as a placeholder: remote shell, raw exec, agent
// restart, reboot. The control vocabulary is a closed enum; a "soon" chip for
// an open-proxy verb would promise exactly what the security model exists to
// prevent.

/** How many placeholders each region may show. */
export const CAPS = {
  iostrip: 2,
  actions: 3,
  dock: 1,
  command: 2,
  alerts: 3,
  // The Status tab's footer: the one admission that the action log is
  // session-memory, not an audit trail.
  status: 1
};

const REGISTRY = [
  {
    id: 'isl',
    region: 'iostrip',
    label: 'ISL',
    soon: 'Coming soon — the agent does not yet report a mean input length per request, and deriving one in the browser from counters would be biased by in-flight requests.'
  },
  {
    id: 'osl',
    region: 'iostrip',
    label: 'OSL',
    soon: 'Coming soon — the agent does not yet report a mean output length per request, and deriving one in the browser from counters would be biased by in-flight requests.'
  },
  {
    id: 'test-prompt',
    region: 'actions',
    label: 'Test prompt',
    soon: 'Coming soon — the agent has no prompt-proxy verb, and the model endpoint is cross-origin from this page.'
  },
  {
    id: 'model-cache',
    region: 'actions',
    label: 'Model cache',
    soon: 'Coming soon — the agent reports only free disk space, not what is in the model cache.'
  },
  {
    id: 'update-agent',
    region: 'actions',
    label: 'Update agent',
    soon: 'Coming soon — there is no update verb; shipping one is security-sensitive and will not be rushed.'
  },
  {
    id: 'requests-tab',
    region: 'dock',
    label: 'Requests',
    soon: 'Coming soon — the engine does not yet export a per-request table, KV-cache occupancy, or queue depth.'
  },
  {
    id: 'range-1h',
    region: 'command',
    label: '1h',
    soon: 'Coming soon — the agent keeps no history ring buffer and has no query verb for one; this page can only show what it saw this session.'
  },
  {
    id: 'range-24h',
    region: 'command',
    label: '24h',
    soon: 'Coming soon — the agent keeps no history ring buffer and has no query verb for one; this page can only show what it saw this session.'
  },
  {
    id: 'durable-audit',
    region: 'status',
    label: 'Durable audit',
    soon: 'Coming soon — the agent keeps no action history; this log is only what this page did this session, and it dies with the tab.'
  },
  {
    id: 'alert-ack',
    region: 'alerts',
    label: 'Ack',
    soon: 'Coming soon — alerts are live state only; the agent keeps no acknowledgement.'
  },
  {
    id: 'alert-silence',
    region: 'alerts',
    label: 'Silence',
    soon: 'Coming soon — alerts are live state only; the agent keeps no silencing rules.'
  },
  {
    id: 'alert-routing',
    region: 'alerts',
    label: 'Routing',
    soon: 'Coming soon — no notification configuration exists anywhere in the agent.'
  }
];

/**
 * The placeholders a region shows.
 *
 * `solo` is required: in solo mode the actions-bar chips collapse into one
 * `soon ▾` chip so a single-machine console never reads as a roadmap, and a
 * caller that does not say which mode it is in would silently pick one.
 *
 * The registry is injectable for the cap-enforcement tests only; production
 * callers use the built-in one.
 *
 * @param {string} region a key of `CAPS`
 * @param {{solo: boolean}} opts
 * @param {object[]} [registry]
 * @returns {object[]}
 */
export function placeholdersFor(region, opts, registry = REGISTRY) {
  if (!(region in CAPS)) throw new TypeError(`not a placeholder region: ${String(region)}`);
  if (opts?.solo !== true && opts?.solo !== false) {
    throw new TypeError('placeholdersFor must be told whether the fleet is solo');
  }
  const entries = registry.filter((e) => e.region === region);
  if (entries.length > CAPS[region]) {
    // Fail at the source of the creep, not in a review screenshot.
    throw new RangeError(
      `${region} carries ${entries.length} placeholders; its cap is ${CAPS[region]}`
    );
  }
  if (opts.solo && region === 'actions' && entries.length > 1) {
    return [
      {
        id: 'soon-menu',
        region: 'actions',
        label: 'soon ▾',
        collapsed: entries
      }
    ];
  }
  return entries;
}

/**
 * One entry by id, for the popover that opens when its button is pressed.
 * Throws on an unknown id: a popover with no sentence is a placeholder for a
 * placeholder.
 *
 * @param {string} id
 * @param {object[]} [registry]
 * @returns {object}
 */
export function placeholder(id, registry = REGISTRY) {
  const found = registry.find((e) => e.id === id);
  if (!found) throw new TypeError(`no placeholder registered as ${String(id)}`);
  return found;
}
