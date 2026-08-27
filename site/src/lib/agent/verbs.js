// SPDX-License-Identifier: AGPL-3.0-only

// The seven control verbs, and what the selected machine lets each one do.
//
// Pure and plain `.js` for the house reason: whether a verb is offered, and
// what the bar says about where it will run, are testable rules, and a file
// holding runes cannot be imported by the test runner.
//
// Two rules from the spec live here:
//
// **A verb a target cannot honour renders disabled with the stated reason,
// never hidden.** A missing button teaches the operator the page is broken; a
// disabled one with "nothing is serving on this node" teaches them the fleet.
//
// **Every action states where it will run before it runs.** "runs on dgx3 ·
// via dgx1" is two different machines to go look at when something goes
// wrong, and the moment to say so is before the click, not in the error.

import { nameOf } from './refusal.js';

/**
 * The seven ControlReq verbs, in bar order.
 *
 * `mutating` verbs interpose a confirm step (and a travel warning when the
 * target is reached through a peer). `needsRunning` verbs answer for a
 * launch, so a node with nothing serving has nothing to ask. `needsLaunch`
 * verbs are refused by a control-only machine, and the bar must say so with
 * the machine's own words.
 */
export const VERBS = [
  { id: 'recipes', label: 'Recipes', mutating: false, needsRunning: false, needsLaunch: false },
  { id: 'preview', label: 'Preview', mutating: false, needsRunning: false, needsLaunch: true },
  { id: 'launch', label: 'Launch', mutating: true, needsRunning: false, needsLaunch: true },
  { id: 'stop', label: 'Stop', mutating: true, needsRunning: true, needsLaunch: false },
  { id: 'status', label: 'Status', mutating: false, needsRunning: false, needsLaunch: false },
  { id: 'stats', label: 'Stats', mutating: false, needsRunning: true, needsLaunch: false },
  { id: 'logs', label: 'Logs', mutating: false, needsRunning: true, needsLaunch: false }
];

/**
 * Whether control can be aimed at this node at all.
 *
 * Local and paired are first-hand. Vouched is second-hand identity but
 * first-class control: the whole point of the relay is that the voucher
 * carries the verb. Discovered and mid-ceremony machines get nothing —
 * telemetry from an unpaired machine proves nothing, and control toward one
 * would be worse.
 *
 * @param {object|null} node
 * @returns {boolean}
 */
export function targetable(node) {
  if (!node) return false;
  return node.isLocal === true || node.pairing === 'paired' || node.pairing === 'vouched';
}

/**
 * What each verb may do against this node, with the reason when it may not.
 *
 * @param {object|null} node
 * @returns {Record<string, {enabled: boolean, reason: string|null}>}
 */
export function availability(node) {
  const out = {};
  for (const v of VERBS) {
    let reason = null;
    if (!node) {
      reason = 'No machine is selected.';
    } else if (node.pairing === 'unreachable' && !node.isLocal) {
      reason = 'This machine is not answering. It stays in your fleet; nothing can run on it until it comes back.';
    } else if (!targetable(node)) {
      reason = 'Not paired. Telemetry and control both need the pairing ceremony first.';
    } else if (v.needsLaunch && node.canLaunch !== true) {
      // The machine's own words, verbatim — the page invents no diagnosis.
      reason = node.cannotLaunchReason || 'This machine cannot run models.';
    } else if (v.needsRunning && !node.running) {
      reason = 'Nothing is serving on this node.';
    }
    out[v.id] = { enabled: reason === null, reason };
  }
  return out;
}

/**
 * Where a verb aimed at this node will run, stated before it runs.
 *
 * @param {object|null} node
 * @param {object[]} nodes for resolving the relay's name
 * @returns {string}
 */
export function route(node, nodes) {
  if (!node) return '';
  if (node.isLocal) return 'runs on this machine';
  const base = `runs on ${nameOf(node.id, nodes)}`;
  return node.reachedVia ? `${base} · via ${nameOf(node.reachedVia, nodes)}` : base;
}

/**
 * The travel warning a mutating verb interposes, or null for a direct target.
 *
 * @param {object|null} node
 * @param {object[]} nodes
 * @returns {string|null}
 */
export function travelWarning(node, nodes) {
  if (!node?.reachedVia) return null;
  return `This will travel through ${nameOf(node.reachedVia, nodes)}.`;
}

/**
 * The `on` target a verb against this node puts on the wire.
 *
 * Null means this machine; anything else is the node id. Centralised so a
 * call site cannot invent a third convention.
 *
 * @param {object} node
 * @returns {string|null}
 */
export function onTarget(node) {
  return node.isLocal ? null : node.id;
}
