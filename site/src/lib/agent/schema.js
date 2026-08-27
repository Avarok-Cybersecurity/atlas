// SPDX-License-Identifier: AGPL-3.0-only

// Rendering and checking settings against the agent's schema.
//
// The schema arrives at the handshake rather than being baked into this page.
// That is deliberate: the agent is what validates, and a page that renders
// bounds the validator does not share is how "it looked fine and then the
// launch was rejected" happens. Everything here is a rendering hint; the agent
// decides.

/** Tabs, in the order they are shown. Labels are ours; keys come from the agent. */
export const GROUPS = [
  { key: 'server', label: 'Server' },
  { key: 'performance', label: 'Performance' },
  { key: 'memory_kv', label: 'Memory & KV' },
  { key: 'speculative', label: 'Speculative' },
  { key: 'tools_chat', label: 'Tools & chat' },
  { key: 'topology', label: 'Topology' }
];

/** Which groups actually have settings in this agent's schema. */
export function groupsPresent(schema, showAdvanced) {
  return GROUPS.filter((g) => schema.some((s) => s.group === g.key && (showAdvanced || !s.advanced)));
}

/** The settings for one tab. */
export function settingsIn(schema, group, showAdvanced) {
  return schema.filter((s) => s.group === group && (showAdvanced || !s.advanced));
}

/**
 * Check a value the way the agent will.
 *
 * Returns null when acceptable, or a message. This is UX only — the agent
 * re-checks everything, and its answer is the one that counts.
 */
export function checkValue(spec, value) {
  // `?? {}` rather than `spec.bound`: this schema arrives from the agent at
  // handshake, and a spec without a bound — an older agent, a newer one, a
  // field renamed — threw here and took the whole settings panel with it. The
  // default arm below already treats an unrecognised KIND as read-only; a
  // missing bound is the same situation and deserves the same answer, not a
  // TypeError.
  const b = spec?.bound ?? {};
  switch (b.kind) {
    case 'int': {
      if (!Number.isInteger(value)) return 'must be a whole number';
      if (value < b.min || value > b.max) return `must be between ${b.min} and ${b.max}`;
      return null;
    }
    case 'float': {
      if (typeof value !== 'number' || !Number.isFinite(value)) return 'must be a number';
      if (value < b.min || value > b.max) return `must be between ${b.min} and ${b.max}`;
      return null;
    }
    case 'enum': {
      // A variant list is what makes an enum checkable. Without one there is
      // nothing to check against, so say nothing rather than throw.
      if (!Array.isArray(b.variants)) return null;
      return b.variants.includes(value) ? null : `must be one of: ${b.variants.join(', ')}`;
    }
    case 'toggle':
    case 'bool_value':
      return typeof value === 'boolean' ? null : 'must be true or false';
    case 'int_or_auto': {
      if (value === 'auto') return null;
      if (!Number.isInteger(value)) return '`auto` or a whole number';
      if (value < b.min || value > b.max) return `must be between ${b.min} and ${b.max}`;
      return null;
    }
    default:
      // A kind this page does not know how to render. The setting is shown
      // read-only rather than hidden, so an agent newer than the site does not
      // silently lose a knob the user can see exists.
      return null;
  }
}

/** Whether this page can render an editor for a bound kind. */
export function isEditable(spec) {
  return ['int', 'float', 'enum', 'toggle', 'bool_value', 'int_or_auto'].includes(
    spec?.bound?.kind
  );
}

/**
 * Recipe values this page cannot offer an editor for.
 *
 * **Not the same as "will not be applied", and the distinction matters.** The
 * schema bounds what a *web page* may override; the agent applies a recipe's
 * own defaults through its flag table regardless. `host` is the clear case: it
 * is deliberately absent from the schema, because a page must not choose what
 * address a server binds to — and the recipe's `host: 0.0.0.0` still lands on
 * the command line. Telling an operator it "will not be applied" would be
 * false, and would send them looking for a version mismatch that is not there.
 *
 * The thing that genuinely reaches nothing is reported per rank by the agent
 * as `unmapped`, because only the machine that owns the flag table can know it.
 */
export function notEditableHere(schema, defaults) {
  const known = new Set(schema.map((s) => s.key));
  return Object.keys(defaults ?? {}).filter((k) => !known.has(k));
}
