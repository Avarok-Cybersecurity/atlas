// SPDX-License-Identifier: AGPL-3.0-only

// What the operator changed, kept apart from what the recipe already said.
//
// The state is a **sparse map of overrides over a read-only base of recipe
// defaults**, never a merged copy. That distinction is what makes version skew
// survivable: a setting this page cannot render is simply never overridden, and
// the recipe's own value applies on the agent. A merged copy would freeze
// whatever the page understood at the moment it loaded and send that back as if
// the operator had chosen it.
//
// It is also what keeps the wire honest. Only genuine differences are sent, so
// "changed 2 settings" means two, and a value typed back to its default stops
// counting as a change rather than travelling as one.

/**
 * The value in force for a setting: the override if there is one, else the
 * recipe's own default, else undefined.
 *
 * @param {string} key
 * @param {object} defaults recipe defaults, read-only
 * @param {object} overrides sparse
 * @returns {unknown}
 */
export function effective(key, defaults, overrides) {
  if (Object.hasOwn(overrides ?? {}, key)) return overrides[key];
  return (defaults ?? {})[key];
}

/**
 * Whether a setting currently differs from what the recipe says.
 *
 * @param {string} key
 * @param {object} defaults
 * @param {object} overrides
 * @returns {boolean}
 */
export function isChanged(key, defaults, overrides) {
  if (!Object.hasOwn(overrides ?? {}, key)) return false;
  return !same(overrides[key], (defaults ?? {})[key]);
}

/**
 * Record a value, dropping it again when it matches the recipe.
 *
 * Setting a value back to the default removes the override rather than storing
 * an equal one: otherwise the change count lies, and the wire carries a
 * decision the operator did not make.
 *
 * @param {object} overrides
 * @param {string} key
 * @param {unknown} value
 * @param {object} defaults
 * @returns {object} a new map
 */
export function set(overrides, key, value, defaults) {
  const next = { ...(overrides ?? {}) };
  if (same(value, (defaults ?? {})[key])) delete next[key];
  else next[key] = value;
  return next;
}

/**
 * Drop an override, returning the setting to the recipe's value.
 *
 * @param {object} overrides
 * @param {string} key
 * @returns {object}
 */
export function clear(overrides, key) {
  const next = { ...(overrides ?? {}) };
  delete next[key];
  return next;
}

/**
 * How many settings differ from the recipe.
 *
 * @param {object} overrides
 * @param {object} defaults
 * @returns {number}
 */
export function changedCount(overrides, defaults) {
  return Object.keys(overrides ?? {}).filter((k) => isChanged(k, defaults, overrides)).length;
}

/**
 * Exactly what to send: the differences, and nothing else.
 *
 * @param {object} overrides
 * @param {object} defaults
 * @returns {object}
 */
export function toWire(overrides, defaults) {
  const out = {};
  for (const [k, v] of Object.entries(overrides ?? {})) {
    if (isChanged(k, defaults, overrides)) out[k] = v;
  }
  return out;
}

/**
 * Parse what an input produced into the type the setting actually holds.
 *
 * Returns `{ value }` or `{ error }`. A string that does not parse is an error
 * rather than a silent zero: `NaN` reaching the agent as a float would be
 * rejected there with a message about bounds, which tells the operator nothing
 * about the empty box they left behind.
 *
 * @param {object} spec
 * @param {string|boolean} raw
 * @returns {{value?: unknown, error?: string}}
 */
export function parse(spec, raw) {
  const kind = spec?.bound?.kind;
  if (kind === 'toggle' || kind === 'bool_value') {
    return { value: raw === true || raw === 'true' };
  }
  if (kind === 'enum') return { value: String(raw) };
  if (kind === 'int_or_auto' && String(raw).trim() === 'auto') return { value: 'auto' };

  const text = String(raw).trim();
  if (text === '') return { error: 'enter a value, or reset it to the recipe default' };
  const n = Number(text);
  if (!Number.isFinite(n)) return { error: 'must be a number' };
  if (kind === 'int' || kind === 'int_or_auto') {
    if (!Number.isInteger(n)) return { error: 'must be a whole number' };
  }
  return { value: n };
}

/**
 * Equality that treats numbers written two ways as one value.
 *
 * `0.9` typed into a box arrives as `0.9`; a recipe may carry `0.90`. Comparing
 * their text would make an unchanged setting look changed.
 *
 * @param {unknown} a
 * @param {unknown} b
 * @returns {boolean}
 */
function same(a, b) {
  if (typeof a === 'number' && typeof b === 'number') return a === b;
  return a === b;
}
