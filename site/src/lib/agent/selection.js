// SPDX-License-Identifier: AGPL-3.0-only

// Which node the stage is showing.
//
// Pure and plain `.js` for the house reason: what happens when the selected
// node vanishes mid-session is exactly the kind of rule that must be
// testable, and a file holding runes cannot be imported by the test runner.
//
// The roster is the selection surface: number keys 1–8 jump, arrows rove,
// and the selection survives a page reload through `location.hash` — by node
// ID, never by index, because the roster reorders as machines come and go
// and "the third row" is a different machine after every reorder.

import { nodeId } from './ingest.js';

/** The roster caps hotkeys at 8 rows, matching the layout's own max. */
export const KEY_MAX = 8;

/** The hash parameter carrying the selection across a reload. */
const HASH_PREFIX = '#node=';

/** The ordered, keyable node list: ids only, holes refused. */
function roster(nodes) {
  return (Array.isArray(nodes) ? nodes : []).filter(
    (n) => n && typeof n.id === 'string' && n.id.length > 0
  );
}

/**
 * The node a number key selects, or null for "no change".
 *
 * Out-of-range is null rather than a clamp: an operator who presses 7 with
 * five machines listed meant a machine that is not there, and jumping them to
 * the last row would select something they did not name.
 *
 * @param {object[]} nodes roster order
 * @param {string} key the pressed key
 * @returns {string|null}
 */
export function selectByKey(nodes, key) {
  if (typeof key !== 'string' || !/^[1-8]$/.test(key)) return null;
  const list = roster(nodes);
  const index = Number(key) - 1;
  return index < Math.min(list.length, KEY_MAX) ? list[index].id : null;
}

/**
 * Arrow roving: one step up or down, clamped at the ends.
 *
 * Clamped, not wrapped — at the bottom of an 8-row roster, one extra ↓ that
 * teleports the stage to a different machine at the top is how an operator
 * stops the wrong launch. No current selection roves to the first node.
 *
 * @param {object[]} nodes roster order
 * @param {string|null} currentId
 * @param {-1|1} delta
 * @returns {string|null} the id to select; null when there is nothing to select
 */
export function move(nodes, currentId, delta) {
  if (delta !== 1 && delta !== -1) throw new TypeError('move roves one step at a time');
  const list = roster(nodes);
  if (list.length === 0) return null;
  const at = list.findIndex((n) => n.id === currentId);
  if (at === -1) return list[0].id;
  const next = Math.min(Math.max(at + delta, 0), list.length - 1);
  return list[next].id;
}

/**
 * Keep the selection valid as the fleet changes under it.
 *
 * A selected node that is gone falls back to the local node — the one machine
 * that is always a sensible thing to look at — then to the first row, then to
 * nothing. Silently keeping a dead id would leave the stage showing a
 * machine the roster no longer lists.
 *
 * @param {object[]} nodes
 * @param {string|null} currentId
 * @returns {string|null}
 */
export function reselect(nodes, currentId) {
  const list = roster(nodes);
  if (list.some((n) => n.id === currentId)) return currentId;
  return list.find((n) => n.isLocal === true)?.id ?? list[0]?.id ?? null;
}

/**
 * The roster rows as the view renders them: hotkey label for the first 8,
 * `aria-selected` computed here so the row markup cannot get it wrong.
 *
 * @param {object[]} nodes
 * @param {string|null} selectedId
 * @returns {{id: string, index: number, key: string|null, selected: boolean}[]}
 */
export function rosterVm(nodes, selectedId) {
  return roster(nodes).map((n, index) => ({
    id: n.id,
    index,
    key: index < KEY_MAX ? String(index + 1) : null,
    selected: n.id === selectedId
  }));
}

/**
 * The hash fragment that persists a selection, or '' for none.
 *
 * Only a validated node id may enter the URL bar: the hash round-trips
 * through history and copy-paste, which makes it wire input on the way back.
 *
 * @param {string|null} id
 * @returns {string}
 */
export function toHash(id) {
  const valid = nodeId(id);
  return valid === null ? '' : `${HASH_PREFIX}${valid}`;
}

/**
 * The selection a hash names, if that machine is actually in the fleet.
 *
 * A hash naming an unknown machine yields null rather than a phantom
 * selection: the stage would otherwise render a node the roster cannot show.
 *
 * @param {string|null|undefined} hash `location.hash`
 * @param {object[]} nodes
 * @returns {string|null}
 */
export function fromHash(hash, nodes) {
  if (typeof hash !== 'string' || !hash.startsWith(HASH_PREFIX)) return null;
  const id = nodeId(hash.slice(HASH_PREFIX.length));
  if (id === null) return null;
  return roster(nodes).some((n) => n.id === id) ? id : null;
}
