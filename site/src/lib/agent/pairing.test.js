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
import { readDecision, readExchange, readExchangeAt } from './pairing.js';

test('the page speaks the version the agent speaks', () => {
  // The agent enforces an exact match. If this drifts from the agent's version
  // the handshake is refused — which is the designed behaviour, but it must
  // drift deliberately rather than by being forgotten.
  //
  // This test has already earned its keep: the agent moved to 4 and the page
  // was left on 3, which the exact-match gate turns into "every control page is
  // refused" the moment both halves ship. 4 adds the `on`/`via` control
  // annotations; the page never sends `on`, and declaring the version it
  // speaks is what lets it connect at all.
  expect(PROTOCOL_VERSION).toBe(4);
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

// ---- pairing with a typed address ----------------------------------------

test('a successful address pairing names the machine that answered', () => {
  const r = readExchangeAt({
    type: 'pair_at_result',
    node: 'c'.repeat(64),
    name: 'spark-43fa',
    address: '10.10.10.2',
    exchanged: true,
    verification: 'amber-koala-drift',
    detail: ''
  });
  expect(r.ok).toBe(true);
  expect(r.node).toBe('c'.repeat(64));
  expect(r.name).toBe('spark-43fa');
  expect(r.address).toBe('10.10.10.2');
  expect(r.verification).toBe('amber-koala-drift');
});

test('nothing answering is not a machine with no name', () => {
  // The operator typed an address. If nothing answered there is no identity at
  // all, and the page must not present one — naming a machine it never reached
  // is worse than saying the attempt failed.
  const r = readExchangeAt({
    type: 'pair_at_result',
    node: null,
    name: '',
    address: '',
    exchanged: false,
    verification: null,
    detail: 'nothing answered at that address'
  });
  expect(r.ok).toBe(false);
  expect(r.node).toBeNull();
  expect(r.detail).toContain('nothing answered');
});

test('an identity is never taken from a reply that did not exchange', () => {
  // Defence in depth: even if a reply carried both a node and exchanged:false,
  // the node must not be adopted. The exchange is what makes the identity mean
  // anything.
  const r = readExchangeAt({ node: 'd'.repeat(64), exchanged: false, verification: null });
  expect(r.ok).toBe(false);
  expect(r.node).toBeNull();
});

test('a name from the wire is sanitized before it is rendered', () => {
  // It arrives from a machine that is not trusted yet — that is the whole point
  // of the step the operator is about to take.
  const r = readExchangeAt({
    node: 'e'.repeat(64),
    exchanged: true,
    verification: 'w',
    name: 'spark\u202e-evil'
  });
  expect(r.name).toBe('spark-evil');
});
