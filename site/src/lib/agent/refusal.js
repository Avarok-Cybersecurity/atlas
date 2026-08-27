// SPDX-License-Identifier: AGPL-3.0-only

// Saying no with a name on it.
//
// Pure and plain `.js` for the house reason: whether the page blames the
// right machine is testable logic, and a file holding runes cannot be
// imported by the test runner.
//
// A forwarded verb can fail in four distinct places, and each sends the
// operator to a different box: the TARGET can refuse ("dgx3 refused: not a
// controller" — go grant control on dgx3), the RELAY can fail to carry it
// ("dgx1 could not reach dgx3" — go look at dgx1 or the link), the LOCAL
// agent can find no route at all (pair something), or the TRANSPORT can
// simply not answer. `ControlRep::Refused.by` exists because a refusal that
// does not say whose it is teaches the operator to restart the wrong box —
// the same rule holds all the way to the rendered string.

import { DETAIL_MAX, sanitize } from './ingest.js';
import { describeError } from './protocol.js';

/**
 * A machine's display name, or enough fingerprint to find it.
 *
 * Falls back to the first 8 hex of the id — the roster's own short form — so
 * a refusal about a machine the fleet list has dropped still names it.
 *
 * @param {string|null|undefined} id
 * @param {{id: string, name?: string}[]} nodes
 * @returns {string}
 */
export function nameOf(id, nodes) {
  if (typeof id !== 'string' || id.length === 0) return 'an unknown machine';
  const found = (Array.isArray(nodes) ? nodes : []).find((n) => n?.id === id);
  const name = found?.name ? sanitize(found.name, 63) : '';
  return name || id.slice(0, 8);
}

/** Verbatim error text, capped and stripped of anything that could restyle the page. */
function verbatim(text) {
  return sanitize(text, DETAIL_MAX);
}

/**
 * Attribute a failed control reply to the machine that failed it.
 *
 * @param {{by?: string|null, error?: object|null, message?: string|null}|null} outcome
 *   `by` is `ControlRep::Refused.by` when the transport surfaced one;
 *   `error` is the `AgentError`; `message` is transport-level prose.
 * @param {{target: string|null, nodes: object[]}} ctx
 *   `target` is who the verb was aimed at (`on`), null for this machine.
 * @returns {{text: string, blame: 'target'|'relay'|'local'|'transport'}}
 */
export function refusal(outcome, ctx) {
  const nodes = ctx?.nodes ?? [];
  const target = ctx?.target ?? null;
  const error = outcome?.error ?? null;

  if (!error) {
    // Nothing answered. Blaming a machine here would be inventing a fact —
    // the honest statement is that no reply came back.
    const detail = verbatim(outcome?.message) || 'no reply';
    return { text: `No answer: ${detail}`, blame: 'transport' };
  }

  switch (error.code) {
    // The target itself said no: its pin of the sender lacks the controller
    // grant. The reason names the exact command to run, so it passes through
    // verbatim.
    case 'control_refused':
      return {
        text: `${nameOf(error.node, nodes)} refused: ${verbatim(error.reason)}`,
        blame: 'target'
      };

    // The relay declined or failed to carry it. `error.node` is the TARGET;
    // the refusing relay is named by `by` when the reply carried one.
    case 'relay_refused': {
      const relay = outcome?.by ? nameOf(outcome.by, nodes) : 'the relay';
      return {
        text: `${relay} could not reach ${nameOf(error.node, nodes)}: ${verbatim(error.detail)}`,
        blame: 'relay'
      };
    }

    // No relay was ever asked: this agent has no route. The fix is pairing or
    // waking a voucher, which is different from a relay's logs.
    case 'not_routable':
      return {
        text: `No route to ${nameOf(error.node, nodes)}: ${verbatim(error.reason)}`,
        blame: 'local'
      };

    default: {
      // An ordinary agent error. When `by` names a machine other than the one
      // we asked — or we asked a remote target at all — the error happened
      // over there, and the string must say so; an unattributed "no recipe
      // named x" reads as the local agent's problem.
      const by = outcome?.by ?? target;
      if (by) {
        return { text: `${nameOf(by, nodes)} refused: ${verbatim(describeError(error))}`, blame: 'target' };
      }
      return { text: verbatim(describeError(error)), blame: 'local' };
    }
  }
}
