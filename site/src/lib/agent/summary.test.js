// SPDX-License-Identifier: AGPL-3.0-only

import { describe, expect, test } from 'bun:test';
import * as U from './summary.js';

const node = (id, extra = {}) => ({
  id,
  isLocal: false,
  pairing: 'paired',
  running: null,
  alerts: [],
  ...extra,
});

describe('a visitor who has never installed anything sees nothing', () => {
  // Most visitors to this site are not customers. A widget reporting that it
  // cannot reach something they have never heard of is worse than no widget.
  test('every non-live mode is silence, not an error', () => {
    for (const mode of ['idle', 'probing', 'no_agent', 'browser_unpaired', 'reconnecting']) {
      expect(U.summarize({ mode, nodes: [] }).show).toBe(false);
    }
    expect(U.summarize(null).show).toBe(false);
    expect(U.summarize({}).show).toBe(false);
  });
});

describe('what a live fleet says', () => {
  test('a lone machine reads as one node, idle', () => {
    const s = U.summarize({ mode: 'live', nodes: [node('a', { isLocal: true })] });
    expect(s.show).toBe(true);
    expect(s.label).toBe('1 node');
    expect(s.detail).toBe('idle');
    expect(s.tone).toBe('ok');
  });

  test('a machine that is serving says so', () => {
    const s = U.summarize({
      mode: 'live',
      nodes: [node('a', { isLocal: true, running: 'qwen3.6-35b' }), node('b')],
    });
    expect(s.label).toBe('2 nodes');
    expect(s.detail).toBe('1 serving');
    expect(s.tone).toBe('serving');
  });

  // A machine that has only been seen on the network is not part of the fleet:
  // discovery grants no authority, and counting it would overstate what the
  // operator can actually use.
  test('a merely-discovered peer is not counted', () => {
    const s = U.summarize({
      mode: 'live',
      nodes: [node('a', { isLocal: true }), node('b', { pairing: 'discovered' })],
    });
    expect(s.label).toBe('1 node');
  });

  test('an unreachable paired peer is still part of the fleet', () => {
    const s = U.summarize({
      mode: 'live',
      nodes: [node('a', { isLocal: true }), node('b', { pairing: 'unreachable' })],
    });
    expect(s.label).toBe('1 node');
  });

  test('a problem outranks whatever else is happening', () => {
    const s = U.summarize({
      mode: 'live',
      nodes: [
        node('a', { isLocal: true, running: 'x' }),
        node('b', { alerts: [{ severity: 'warning' }] }),
      ],
    });
    expect(s.tone).toBe('warning');
  });

  test('critical outranks warning', () => {
    expect(
      U.worstSeverity([
        node('a', { alerts: [{ severity: 'warning' }] }),
        node('b', { alerts: [{ severity: 'critical' }] }),
      ]),
    ).toBe('critical');
    expect(U.worstSeverity([node('a', { alerts: [{ severity: 'info' }] })])).toBeNull();
    expect(U.worstSeverity([])).toBeNull();
  });
});

describe('a flapping peer does not flicker the count', () => {
  // Discovery is multicast and lossy. A count that oscillates between 2 and 3
  // teaches an operator to ignore it.
  test('one missing read holds the node', () => {
    const prev = [node('a'), node('b')];
    const r = U.settle(prev, [node('a')], {});
    expect(r.nodes.map((n) => n.id).sort()).toEqual(['a', 'b']);
    expect(r.misses.b).toBe(1);
  });

  test('two missing reads drop it', () => {
    const prev = [node('a'), node('b')];
    const one = U.settle(prev, [node('a')], {});
    const two = U.settle(one.nodes, [node('a')], one.misses);
    expect(two.nodes.map((n) => n.id)).toEqual(['a']);
    expect(two.misses.b).toBeUndefined();
  });

  test('a node that comes back has its count cleared', () => {
    const prev = [node('a'), node('b')];
    const one = U.settle(prev, [node('a')], {});
    const back = U.settle(one.nodes, [node('a'), node('b')], one.misses);
    expect(back.misses.b).toBeUndefined();
    // And it must survive the next miss rather than being evicted immediately.
    const after = U.settle(back.nodes, [node('a')], back.misses);
    expect(after.nodes.map((n) => n.id).sort()).toEqual(['a', 'b']);
  });

  test('fresh data always wins for a node that is present', () => {
    const prev = [node('a', { running: 'old' })];
    const r = U.settle(prev, [node('a', { running: 'new' })], {});
    expect(r.nodes).toHaveLength(1);
    expect(r.nodes[0].running).toBe('new');
  });
});

describe('a control-only machine says what it is', () => {
  // "idle" invites an operator to expect a model to start here. A laptop
  // running --client structurally cannot run one, and no amount of waiting
  // changes that, so the pill must not imply otherwise.
  test('a lone control-only machine reads as control only, not idle', () => {
    const s = U.summarize({
      mode: 'live',
      nodes: [node('a', { isLocal: true, canLaunch: false })],
    });
    expect(s.label).toBe('1 node');
    expect(s.detail).toBe('control only');
  });

  test('once something launchable is paired it is idle again — there is a wait that ends', () => {
    const s = U.summarize({
      mode: 'live',
      nodes: [node('a', { isLocal: true, canLaunch: false }), node('b', { canLaunch: true })],
    });
    expect(s.detail).toBe('idle');
  });

  test('a paired peer that also cannot launch does not make the fleet launchable', () => {
    const s = U.summarize({
      mode: 'live',
      nodes: [node('a', { isLocal: true, canLaunch: false }), node('b', { canLaunch: false })],
    });
    expect(s.detail).toBe('control only');
  });

  test('serving always wins over the control-only note', () => {
    // A control-only head can drive a peer that is serving; that is the whole
    // point of the mode, and it is the more useful thing to report.
    const s = U.summarize({
      mode: 'live',
      nodes: [
        node('a', { isLocal: true, canLaunch: false }),
        node('b', { canLaunch: true, running: 'qwen3.6-35b' }),
      ],
    });
    expect(s.detail).toBe('1 serving');
  });

  test('a machine whose agent has not answered yet is idle, not control only', () => {
    // canLaunch absent means "not yet known"; assuming false would flash the
    // wrong word at every operator on a Spark while the agent connects.
    const s = U.summarize({ mode: 'live', nodes: [node('a', { isLocal: true })] });
    expect(s.detail).toBe('idle');
  });
});
