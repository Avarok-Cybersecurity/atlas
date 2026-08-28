// SPDX-License-Identifier: AGPL-3.0-only

import { describe, expect, test } from 'bun:test';
import { nameOf, refusal } from './refusal.js';

const DGX1 = '11'.repeat(32);
const DGX3 = '33'.repeat(32);
const NODES = [
  { id: DGX1, name: 'dgx1' },
  { id: DGX3, name: 'dgx3' }
];
const CTX = { target: DGX3, nodes: NODES };

describe('each failure names the machine it belongs to', () => {
  test('the target refusing control blames the target, verbatim', () => {
    const r = refusal(
      { error: { code: 'control_refused', node: DGX3, reason: 'not a controller — run `atlasctl peer grant-control`' } },
      CTX
    );
    expect(r.blame).toBe('target');
    expect(r.text).toBe('dgx3 refused: not a controller — run `atlasctl peer grant-control`');
  });

  test('a relay failure blames the relay and names both machines', () => {
    const r = refusal(
      { by: DGX1, error: { code: 'relay_refused', node: DGX3, detail: 'dial timed out' } },
      CTX
    );
    expect(r.blame).toBe('relay');
    expect(r.text).toBe('dgx1 could not reach dgx3: dial timed out');
  });

  test('a relay failure without a `by` does not invent one', () => {
    const r = refusal({ error: { code: 'relay_refused', node: DGX3, detail: 'answer budget elapsed' } }, CTX);
    expect(r.text).toBe('the relay could not reach dgx3: answer budget elapsed');
  });

  test('the relay is named from inside the error, where `by` never reaches', () => {
    // The browser frame the agent actually sends: no `by` at all, because
    // `session/remote.rs` keeps only the error. Before `via` existed this
    // read "the relay could not reach dgx3" — true, useless, and one box
    // short of telling the operator where to look.
    const r = refusal({ error: { code: 'relay_refused', node: DGX3, via: DGX1, detail: 'dial timed out' } }, CTX);
    expect(r.blame).toBe('relay');
    expect(r.text).toBe('dgx1 could not reach dgx3: dial timed out');
  });

  test('`via` outranks `by`: it is built where the failure happened', () => {
    // They agree on one hop, so a disagreement means something is wrong and
    // the structural field — written by the code that produced the error,
    // not by whoever forwarded it — is the one to believe.
    const r = refusal(
      { by: DGX3, error: { code: 'relay_refused', node: DGX3, via: DGX1, detail: 'dial timed out' } },
      CTX
    );
    expect(r.text).toBe('dgx1 could not reach dgx3: dial timed out');
  });

  test('no route is a local fact: the fix is pairing, not a relay log', () => {
    const r = refusal(
      { error: { code: 'not_routable', node: DGX3, reason: 'not pinned and no reachable voucher' } },
      CTX
    );
    expect(r.blame).toBe('local');
    expect(r.text).toBe('No route to dgx3: not pinned and no reachable voucher');
  });

  test('an ordinary error on a forwarded verb is attributed to the target', () => {
    const r = refusal({ by: DGX3, error: { code: 'already_running', recipe: 'r1' } }, CTX);
    expect(r.blame).toBe('target');
    expect(r.text).toBe('dgx3 refused: That recipe is already running.');
  });

  test('a local error stays unattributed prose', () => {
    const r = refusal(
      { error: { code: 'docker_unavailable', detail: 'socket missing' } },
      { target: null, nodes: NODES }
    );
    expect(r.blame).toBe('local');
    expect(r.text).toBe('Docker is not available on that machine: socket missing');
  });

  test('silence is silence — no machine is blamed for a missing reply', () => {
    const r = refusal({ message: 'The agent did not reply.' }, CTX);
    expect(r.blame).toBe('transport');
    expect(r.text).toBe('No answer: The agent did not reply.');
    expect(refusal(null, CTX).text).toBe('No answer: no reply');
  });
});

describe('the strings are safe and the names resolvable', () => {
  test('error detail passes through verbatim but sanitised', () => {
    // A refusal detail crosses a peer channel; control characters and bidi
    // marks must not survive into the DOM.
    const r = refusal(
      { error: { code: 'control_refused', node: DGX3, reason: 'bad\u0000 thing\u202e!' } },
      CTX
    );
    expect(r.text).toBe('dgx3 refused: bad thing!');
  });

  test('a machine the fleet list has dropped is still named by fingerprint', () => {
    const gone = 'ee'.repeat(32);
    expect(nameOf(gone, NODES)).toBe('eeeeeeee');
    expect(nameOf(null, NODES)).toBe('an unknown machine');
  });
});
