// SPDX-License-Identifier: AGPL-3.0-only

// The session action log: what this page asked the fleet to do, and what came
// back — verb, target, route, outcome.
//
// Pure and plain `.js` for the house reason: what a log entry must carry, and
// how the log is bounded, are testable rules, and a file holding runes cannot
// be imported by the test runner.
//
// Deliberately labelled *this session* everywhere it renders: the log lives
// in page memory and dies with the tab. The agent keeps no action history —
// that gap is the `durable-audit` placeholder, not something this module
// papers over by pretending localStorage is an audit trail.

import { DETAIL_MAX, sanitize } from './ingest.js';

/** Entries kept. Enough to scroll back through a session, bounded on purpose. */
export const LOG_CAP = 200;

/**
 * One log entry, validated at the door.
 *
 * `ok` and `atMs` are required rather than defaulted: an outcome that does
 * not say whether it succeeded, or an entry that cannot say when, is exactly
 * the ambiguity an action log exists to remove.
 *
 * @param {{verb: string, target: string, route: string, outcome: string, ok: boolean}} fields
 * @param {number} atMs
 * @returns {{verb: string, target: string, route: string, outcome: string, ok: boolean, at: number}}
 */
export function entry(fields, atMs) {
  if (fields?.ok !== true && fields?.ok !== false) {
    throw new TypeError('a log entry must say whether the action succeeded');
  }
  if (!Number.isFinite(atMs)) {
    throw new TypeError('a log entry must carry the clock, not assume one');
  }
  const verb = sanitize(fields.verb, 32);
  if (!verb) throw new TypeError('a log entry must name its verb');
  return {
    verb,
    target: sanitize(fields.target, 63) || 'this machine',
    route: sanitize(fields.route, 140),
    outcome: sanitize(fields.outcome, DETAIL_MAX),
    ok: fields.ok,
    at: atMs
  };
}

/**
 * Append, newest first, bounded.
 *
 * Newest first because the log renders in a dock tab: the row an operator is
 * looking for is almost always the action they just took.
 *
 * @param {object[]} log
 * @param {object} e from `entry`
 * @returns {object[]} a new array
 */
export function append(log, e) {
  return [e, ...(Array.isArray(log) ? log : [])].slice(0, LOG_CAP);
}
