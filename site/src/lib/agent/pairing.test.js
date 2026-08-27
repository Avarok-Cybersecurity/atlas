// SPDX-License-Identifier: AGPL-3.0-only

// The trust ceremony's contract with the agent, as of protocol 2.
//
// `fleet.pair` no longer means "trusted" — it means the exchange completed and
// there are words to compare. These pin the reply shapes, because the failure
// they guard against is silent: a page that read the old `paired` field would
// find it undefined, treat the exchange as failed, and show a pairing dialog
// that never advances — or, worse, a page that kept the old field name against
// a new agent would show a machine as trusted that the agent has not accepted.

import { test, expect } from 'bun:test';
import { PROTOCOL_VERSION } from './protocol.js';
import { readDecision, readExchange } from './pairing.js';

test('the page speaks the version that has the two-phase verbs', () => {
  // The agent enforces an exact match. If this drifts below the agent's
  // version the handshake is refused — which is the designed behaviour, but it
  // must drift deliberately rather than by being forgotten.
  expect(PROTOCOL_VERSION).toBe(2);
});

/** The reply shape `pair_peer` returns under protocol 2. */
const exchangeReply = (over = {}) => ({
  type: 'pair_result',
  node: 'a'.repeat(64),
  exchanged: true,
  verification: 'abcd-ef01',
  detail: '',
  ...over
});

test('an exchange reply carries words and does not claim trust', () => {
  const r = exchangeReply();
  expect(r.exchanged).toBe(true);
  expect(r.verification).toBeTruthy();
  // The old field must be gone, not merely unused: a page still reading it
  // would get undefined and silently treat every exchange as a failure.
  expect('paired' in r).toBe(false);
});

test('the protocol invariant holds: exchanged implies words', () => {
  // Documented on ServerMsg::PairResult. A reply claiming an exchange with no
  // words would leave the dialog with nothing for the human to compare, which
  // is the one thing this ceremony is for.
  for (const r of [exchangeReply(), exchangeReply({ exchanged: false, verification: null })]) {
    expect(r.exchanged).toBe(r.verification !== null);
  }
});

test('a decision reply says what is true about trust, not what a ceremony did', () => {
  const d = { type: 'pair_decision', node: 'b'.repeat(64), trusted: true, detail: '' };
  expect(d.trusted).toBe(true);
  expect('exchanged' in d).toBe(false);
  expect('verification' in d).toBe(false);
});

// ---- reading the replies -------------------------------------------------

test('an exchange is successful only when the agent says it exchanged', () => {
  expect(readExchange(exchangeReply()).ok).toBe(true);
  expect(readExchange(exchangeReply()).verification).toBe('abcd-ef01');

  // The failure cases all have to read as "not exchanged". `paired: true` is
  // the one that matters: it is what a protocol-1 agent would send, and reading
  // it as success would show trusted words for an exchange this page cannot
  // vouch for.
  expect(readExchange(exchangeReply({ exchanged: false, verification: null })).ok).toBe(false);
  expect(readExchange({ paired: true, verification: 'x' }).ok).toBe(false);
  expect(readExchange({}).ok).toBe(false);
  expect(readExchange(null).ok).toBe(false);
  expect(readExchange({ exchanged: 'true' }).ok).toBe(false); // not the string
  expect(readExchange({ exchanged: 1 }).ok).toBe(false); // not truthy-coerced
});

test('a confirm succeeds only when the node ended up trusted', () => {
  expect(readDecision({ trusted: true, detail: '' }, true).ok).toBe(true);
  expect(readDecision({ trusted: false, detail: 'nope' }, true).ok).toBe(false);
  // A reply that carries no verdict must not be read as one.
  expect(readDecision({}, true).ok).toBe(false);
  expect(readDecision(null, true).ok).toBe(false);
});

test('a reject succeeds only when the node ended up NOT trusted', () => {
  expect(readDecision({ trusted: false, detail: '' }, false).ok).toBe(true);
  // The case the old code could not see: the agent says the peer is STILL
  // trusted, and the page previously reported the refusal as successful
  // because it returned `ok: true` without reading the reply at all.
  expect(readDecision({ trusted: true, detail: '' }, false).ok).toBe(false);
  expect(readDecision({}, false).ok).toBe(false);
});

test('a decision reader must be told which outcome means success', () => {
  // Defaulting would silently pick confirm's polarity for a reject, which is
  // the exact inversion that reports a failed refusal as a success.
  expect(() => readDecision({ trusted: true }, undefined)).toThrow();
});

test('detail text is sanitized, because it is rendered', () => {
  const d = readDecision({ trusted: false, detail: 'a\u202eb' }, false).detail;
  expect(d).toBe('ab');
  expect(readDecision({ trusted: false, detail: 'x'.repeat(9999) }, false).detail.length)
    .toBeLessThan(9999);
});
