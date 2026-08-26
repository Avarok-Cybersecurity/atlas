// SPDX-License-Identifier: AGPL-3.0-only

// What this browser remembers about its operator, so returning to the site
// resumes rather than restarts.
//
// SCOPE. A profile holds *preferences* — the recipe last chosen, the settings
// changed away from its defaults, which machine was serving the API. It is not
// a cache of the fleet and it is not a session: nothing here grants access to
// anything, and an attacker who reads it learns what someone likes to run, not
// how to run it. The browser-pairing token stays in its own key, owned by
// `protocol.js`, because it is a credential and has a different lifetime.
//
// NODE IDS ARE STORED, and that is a deliberate narrowing of the fleet
// module's "nothing but the token is written" rule. A head selection is
// worthless without naming the machine it refers to, and the id is a key
// fingerprint this machine already holds in its pin store. Names, addresses,
// vitals and alerts are still never written — only the fingerprint, and only
// for machines the operator themselves selected.
//
// STORAGE IS INJECTED rather than reached for, so every rule below is testable
// without a browser. That matters more than usual here: the failure modes are
// all in the reading — a hand-edited value, a profile written by a newer
// version of the site, a Safari private window where `setItem` throws — and
// none of them may take the page down.
//
// EVERY READ IS TOTAL. `load` cannot fail and cannot throw; it answers with a
// valid profile no matter what it finds. A corrupt profile costs the operator
// their preferences, never their page.

/** Where a profile lives. */
export const KEY = 'atlas.profile';

/** Schema version, so a future shape can migrate instead of exploding. */
export const VERSION = 1;

/** Bounds. Storage is attacker-writable — by hand, if nothing else. */
const MAX_RECIPE_ID = 128;
const MAX_NODE_ID = 128;
const MAX_NODES = 8;
const MAX_REMEMBERED_RECIPES = 24;
const MAX_OVERRIDES_PER_RECIPE = 64;
const MAX_SETTING_KEY = 64;
const MAX_STRING_VALUE = 128;
/** Refuse to write more than this, so a profile cannot fill the origin quota. */
const MAX_BYTES = 64 * 1024;

/** A profile with nothing in it. Fresh object each call — never a shared one. */
export function empty() {
  return { v: VERSION, recipe: null, head: null, selected: [], overrides: {} };
}

/**
 * Trim a string, or answer null when it is not one worth keeping.
 *
 * @param {unknown} raw
 * @param {number} max
 * @returns {string|null}
 */
function str(raw, max) {
  if (typeof raw !== 'string') return null;
  const s = raw.trim();
  if (s.length === 0 || s.length > max) return null;
  return s;
}

/**
 * Keep a setting value only if it is one of the three shapes the wire uses.
 *
 * Objects and arrays are dropped rather than descended into: a setting is a
 * scalar, so a nested value is either corruption or someone probing, and in
 * both cases the right answer is to forget it.
 *
 * @param {unknown} v
 * @returns {string|number|boolean|undefined}
 */
function scalar(v) {
  if (typeof v === 'boolean') return v;
  // NaN and infinities round-trip through JSON as null, so a non-finite number
  // here means the value was hand-written. It cannot be a valid setting.
  if (typeof v === 'number') return Number.isFinite(v) ? v : undefined;
  if (typeof v === 'string') return v.length <= MAX_STRING_VALUE ? v : undefined;
  return undefined;
}

/**
 * Normalise whatever was stored into a profile this app can use.
 *
 * @param {unknown} raw
 * @returns {object}
 */
export function sanitize(raw) {
  const out = empty();
  if (raw === null || typeof raw !== 'object' || Array.isArray(raw)) return out;

  // A profile from a *newer* site is not readable — its fields may mean
  // something else — so it is discarded rather than half-applied. An older one
  // has no fields this version does not understand, so it is read as-is.
  if (raw.v !== VERSION) return out;

  out.recipe = str(raw.recipe, MAX_RECIPE_ID);
  out.head = str(raw.head, MAX_NODE_ID);

  if (Array.isArray(raw.selected)) {
    const seen = new Set();
    for (const id of raw.selected) {
      const s = str(id, MAX_NODE_ID);
      if (s === null || seen.has(s)) continue;
      seen.add(s);
      out.selected.push(s);
      if (out.selected.length >= MAX_NODES) break;
    }
  }

  if (raw.overrides !== null && typeof raw.overrides === 'object' && !Array.isArray(raw.overrides)) {
    let recipes = 0;
    for (const [recipeId, map] of Object.entries(raw.overrides)) {
      if (recipes >= MAX_REMEMBERED_RECIPES) break;
      const id = str(recipeId, MAX_RECIPE_ID);
      if (id === null || map === null || typeof map !== 'object' || Array.isArray(map)) continue;
      const kept = {};
      let n = 0;
      for (const [k, v] of Object.entries(map)) {
        if (n >= MAX_OVERRIDES_PER_RECIPE) break;
        const key = str(k, MAX_SETTING_KEY);
        if (key === null) continue;
        const val = scalar(v);
        if (val === undefined) continue;
        kept[key] = val;
        n += 1;
      }
      // A recipe whose every override was rejected is not worth a key.
      if (n === 0) continue;
      out.overrides[id] = kept;
      recipes += 1;
    }
  }

  // The head must be one of the selected machines, or it means nothing. This
  // is the same rule the launch flow enforces; storage is not allowed to be a
  // way around it.
  if (out.head !== null && !out.selected.includes(out.head)) out.head = null;

  return out;
}

/**
 * Read the profile. Never throws, and always answers a usable profile.
 *
 * @param {object} storage a `localStorage`-shaped object, or null
 * @returns {object}
 */
export function load(storage) {
  try {
    const raw = storage?.getItem(KEY);
    if (typeof raw !== 'string') return empty();
    return sanitize(JSON.parse(raw));
  } catch {
    // Disabled storage, a quota error on read, or text that is not JSON. All
    // of them mean the same thing to the caller: no profile.
    return empty();
  }
}

/**
 * Write the profile. Answers whether it was actually stored.
 *
 * Sanitised on the way out as well as in, so a bug upstream cannot persist a
 * value that `load` would then refuse — which would look like the profile
 * silently not saving.
 *
 * @param {object} storage a `localStorage`-shaped object, or null
 * @param {object} profile
 * @returns {boolean}
 */
export function save(storage, profile) {
  if (!storage) return false;
  try {
    const body = JSON.stringify(sanitize(profile));
    if (body.length > MAX_BYTES) return false;
    storage.setItem(KEY, body);
    return true;
  } catch {
    // Private windows throw on write, and a full origin quota throws too.
    // Losing a preference is not worth an error the operator has to read.
    return false;
  }
}

/** Forget everything. Answers whether the key is now gone. */
export function clear(storage) {
  try {
    storage?.removeItem(KEY);
    return true;
  } catch {
    return false;
  }
}

/**
 * Apply a patch, returning a new profile. Does not touch storage.
 *
 * @param {object} profile
 * @param {object} patch
 * @returns {object}
 */
export function merge(profile, patch) {
  return sanitize({ ...sanitize(profile), ...patch, v: VERSION });
}

/**
 * Record the overrides for one recipe, leaving other recipes alone.
 *
 * An empty map removes the recipe's entry rather than storing `{}`: "changed
 * nothing" is the default state and does not need remembering, and keeping it
 * would let the remembered-recipe budget fill with nothing.
 *
 * @param {object} profile
 * @param {string} recipeId
 * @param {object} overrides
 * @returns {object}
 */
export function rememberOverrides(profile, recipeId, overrides) {
  const base = sanitize(profile);
  const next = { ...base.overrides };
  if (overrides && Object.keys(overrides).length > 0) next[recipeId] = overrides;
  else delete next[recipeId];
  return sanitize({ ...base, overrides: next });
}

/**
 * The overrides remembered for a recipe, as a fresh object.
 *
 * Always a copy: the caller binds this to an editor, and handing out the
 * stored object would let an edit mutate the profile without going through
 * `rememberOverrides`.
 *
 * @param {object} profile
 * @param {string|null} recipeId
 * @returns {object}
 */
export function overridesFor(profile, recipeId) {
  if (recipeId === null) return {};
  return { ...(sanitize(profile).overrides[recipeId] ?? {}) };
}
