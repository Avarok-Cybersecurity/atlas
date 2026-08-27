// SPDX-License-Identifier: AGPL-3.0-only

// The decision, not the markup. Each case below is a state the launch dialog
// actually reaches, including the ones that used to send an operator to a
// settings form for a machine that was never going to run anything.

import { describe as suite, expect, test } from 'bun:test';
import * as P from './placement.js';

const node = (id, extra = {}) => ({
  id,
  name: id,
  isLocal: false,
  pairing: 'paired',
  canLaunch: true,
  os: '',
  accelerator: '',
  addresses: [],
  ...extra
});

suite('who could hold a rank', () => {
  test('a discovered stranger is not somewhere to send a workload', () => {
    const got = P.candidates([node('a', { pairing: 'discovered' }), node('b')]);
    expect(got.map((n) => n.id)).toEqual(['b']);
  });

  test('a paired machine that cannot launch is not a candidate either', () => {
    const got = P.candidates([node('a', { canLaunch: false }), node('b')]);
    expect(got.map((n) => n.id)).toEqual(['b']);
  });

  test('this machine sorts first, then by name, so vitals arriving do not reorder it', () => {
    const got = P.candidates([node('zeta'), node('alpha'), node('me', { isLocal: true })]);
    expect(got.map((n) => n.id)).toEqual(['me', 'alpha', 'zeta']);
  });

  test('junk in is an empty list, not a crash', () => {
    expect(P.candidates(null)).toEqual([]);
    expect(P.candidates([null, undefined, {}])).toEqual([]);
  });
});

suite('whether to ask at all', () => {
  test('one candidate skips the step — a chooser with one option is a nag', () => {
    const d = P.decide([node('me', { isLocal: true })], { nodes: 1 });
    expect(d.kind).toBe('only');
    expect(d.target.id).toBe('me');
  });

  test('two candidates is a real question', () => {
    const d = P.decide([node('me', { isLocal: true }), node('spark')], { nodes: 1 });
    expect(d.kind).toBe('ask');
    expect(d.options.map((n) => n.id)).toEqual(['me', 'spark']);
  });

  test('nothing to run on says so, and offers to onboard', () => {
    const d = P.decide([], { nodes: 1 });
    expect(d.kind).toBe('none');
    expect(d.canOnboard).toBe(true);
  });

  // The case this whole step exists for: the laptop.
  test('a control-only machine with no peers explains itself', () => {
    const d = P.decide([node('me', { isLocal: true, canLaunch: false })], { nodes: 1 }, false);
    expect(d.kind).toBe('none');
    expect(d.reason).toContain('cannot run models');
    expect(d.reason).toContain('no machine that can is paired');
  });

  test('before the agent has said, the wording does not accuse this machine', () => {
    // canLaunch tri-state: null is "not yet known", and must not render as a
    // statement that this machine cannot run models.
    const d = P.decide([], { nodes: 1 }, null);
    expect(d.reason).not.toContain('cannot run models');
  });
});

suite('multi-node recipes are a cluster plan, not a placement question', () => {
  test('enough machines routes to the cluster flow rather than a chooser', () => {
    const d = P.decide([node('a'), node('b')], { nodes: 2 });
    expect(d.kind).toBe('cluster');
    expect(d.need).toBe(2);
  });

  test('too few says how many short, in the plural it deserves', () => {
    const d = P.decide([node('a')], { nodes: 2 });
    expect(d.kind).toBe('none');
    expect(d.reason).toContain('needs 2 machines and 1 is available');
  });

  test('a missing or nonsense node count means one machine, never zero', () => {
    for (const recipe of [{}, { nodes: 0 }, { nodes: 'two' }, null]) {
      expect(P.required(recipe)).toBe(1);
    }
  });
});

suite('how a machine is described', () => {
  test('what distinguishes it, in the order an operator scans', () => {
    const d = P.describe(node('spark', { os: 'Linux', accelerator: 'GB10', addresses: [{ addr: '10.10.10.2', class: 'roce' }] }));
    expect(d).toBe('Linux · GB10 · 10.10.10.2');
  });

  test('unreported fields are omitted, never printed blank', () => {
    expect(P.describe(node('bare'))).toBe('');
    expect(P.describe(node('some', { os: 'macOS' }))).toBe('macOS');
  });

  test('this machine says so and does not need an address', () => {
    const d = P.describe(node('me', { isLocal: true, os: 'macOS', addresses: [{ addr: '1.2.3.4', class: 'ethernet' }] }));
    expect(d).toBe('this machine · macOS');
  });

  test('a loopback address is not an address anyone can use', () => {
    const d = P.describe(node('x', { addresses: [{ addr: '127.0.0.1', class: 'loopback' }] }));
    expect(d).toBe('');
  });
});

test('a launchable machine with an unreported fleet runs here, it does not despair', () => {
  // First launch on a brand-new Spark: the fleet session has not started, so
  // `nodes` is empty. The agent this dialog is connected to has already said it
  // can launch. Telling the operator "No machine here can run this yet" sent
  // them off to onboard a second machine to fix a machine that was fine.
  const d = P.decide([], { id: 'solo' }, true);
  expect(d.kind).toBe('here');
});

test('but an unknown local capability still asks rather than assuming', () => {
  // null means the agent has not said. Guessing "it can" would flash a
  // launchable UI at a control-only laptop — the same mistake pointing the
  // other way.
  const d = P.decide([], { id: 'solo' }, null);
  expect(d.kind).toBe('none');
  expect(d.canOnboard).toBe(true);
});

test('a machine that has said it cannot launch is still told so plainly', () => {
  const d = P.decide([], { id: 'solo' }, false);
  expect(d.kind).toBe('none');
  expect(d.reason).toMatch(/cannot run models/);
});
