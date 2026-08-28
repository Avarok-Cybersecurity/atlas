// SPDX-License-Identifier: AGPL-3.0-only

// How the page reads the agent's answers to the trust ceremony.
//
// Pure and separate from `fleet.svelte.js` for the reason the rest of this
// directory is split that way: a file holding runes cannot be imported by the
// test runner, so anything living in it is untestable by construction. And this
// is the last logic in the fleet that should be untestable — it decides whether
// the operator is told a machine is trusted.
//
// Under protocol 2 the reply shapes are:
//   pair_peer      -> { exchanged: bool, verification: string|null, detail }
//   pair_peer_at   -> { node: id|null, name, address, exchanged, verification, detail }
//   confirm/reject -> { trusted: bool, detail }
//   unpair_peer    -> { trusted: bool, detail }
//
// Every reader below tests the field EXPLICITLY against the value that means
// success, rather than trusting the transport to have sent the right message.
// A reply whose field is missing reads as `undefined`, which is neither `true`
// nor `false`, so an unexpected or mis-routed reply fails closed in the safe
// direction: the page says the operation did not take.

import { DETAIL_MAX, sanitize } from './ingest.js';

/**
 * Read a `pair_result`.
 *
 * @param {any} reply
 * @returns {{ok: boolean, verification: string|null, detail: string}}
 */
export function readExchange(reply) {
  return {
    ok: reply?.exchanged === true,
    verification: reply?.verification ?? null,
    detail: sanitize(reply?.detail, DETAIL_MAX)
  };
}

/**
 * Read a `pair_decision`, given the trust state that means success.
 *
 * `expectTrusted` is required rather than defaulted: confirm succeeds when the
 * node ends up trusted, reject and unpair succeed when it does not, and a
 * default would silently pick one of those for a caller that forgot to say.
 *
 * @param {any} reply
 * @param {boolean} expectTrusted
 * @returns {{ok: boolean, detail: string}}
 */
export function readDecision(reply, expectTrusted) {
  if (typeof expectTrusted !== 'boolean') {
    throw new TypeError('readDecision needs to be told which outcome means success');
  }
  return {
    ok: reply?.trusted === expectTrusted,
    detail: sanitize(reply?.detail, DETAIL_MAX)
  };
}

/**
 * Read a `pair_at_result`.
 *
 * Carries the identity that answered, because the operator typed an ADDRESS —
 * nothing was discovered, so this reply is the first moment anyone can say
 * which machine it was. They need that in front of them before they are asked
 * to trust it, alongside the words.
 *
 * `node` is null when nothing answered. That is not the same as a failure with
 * an identity, and collapsing the two would let the page name a machine it
 * never reached.
 *
 * @param {any} reply
 * @returns {{ok: boolean, node: string|null, name: string, address: string,
 *   verification: string|null, detail: string}}
 */
export function readExchangeAt(reply) {
  const ok = reply?.exchanged === true;
  return {
    ok,
    node: ok && typeof reply?.node === 'string' ? reply.node : null,
    name: sanitize(reply?.name, 63),
    address: sanitize(reply?.address, 63),
    verification: reply?.verification ?? null,
    detail: sanitize(reply?.detail, DETAIL_MAX)
  };
}
