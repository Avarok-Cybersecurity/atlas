// SPDX-License-Identifier: AGPL-3.0-only

import { describe, expect, test } from 'bun:test';
import { VERBS, availability, onTarget, route, targetable, travelWarning } from './verbs.js';

const DGX1 = '11'.repeat(32);
const DGX3 = '33'.repeat(32);
const NODES = [
  { id: DGX1, name: 'dgx1' },
  { id: DGX3, name: 'dgx3' }
];

const paired = (over = {}) => ({
  id: DGX3,
  name: 'dgx3',
  isLocal: false,
  pairing: 'paired',
  canLaunch: true,
  cannotLaunchReason: '',
  running: 'qwen3.8-27b-nvfp4',
  reachedVia: null,
  ...over
});

describe('the bar offers exactly the seven verbs, in order', () => {
  test('the vocabulary is closed', () => {
    expect(VERBS.map((v) => v.id)).toEqual([
      'recipes',
      'preview',
      'launch',
      'stop',
      'status',
      'stats',
      'logs'
    ]);
  });
});

describe('disabled with the stated reason, never hidden', () => {
  test('a healthy paired node enables everything', () => {
    const a = availability(paired());
    for (const v of VERBS) expect(a[v.id]).toEqual({ enabled: true, reason: null });
  });

  test('nothing serving disables the launch-scoped verbs only', () => {
    const a = availability(paired({ running: null }));
    for (const id of ['stop', 'stats', 'logs']) {
      expect(a[id].enabled).toBe(false);
      expect(a[id].reason).toBe('Nothing is serving on this node.');
    }
    for (const id of ['recipes', 'preview', 'launch', 'status']) {
      expect(a[id].enabled).toBe(true);
    }
  });

  test('a control-only machine refuses launch verbs in its own words', () => {
    const a = availability(
      paired({ canLaunch: false, cannotLaunchReason: 'no accelerator present' })
    );
    expect(a.launch).toEqual({ enabled: false, reason: 'no accelerator present' });
    expect(a.preview.enabled).toBe(false);
    // Reading its inventory and status is still allowed.
    expect(a.recipes.enabled).toBe(true);
    expect(a.status.enabled).toBe(true);
  });

  test('an unpaired machine gets nothing, with the ceremony named', () => {
    const a = availability(paired({ pairing: 'discovered' }));
    for (const v of VERBS) {
      expect(a[v.id].enabled).toBe(false);
      expect(a[v.id].reason).toContain('pairing ceremony');
    }
    expect(targetable(paired({ pairing: 'discovered' }))).toBe(false);
  });

  test('unreachable is its own reason — the machine is trusted, just absent', () => {
    const a = availability(paired({ pairing: 'unreachable' }));
    expect(a.status.enabled).toBe(false);
    expect(a.status.reason).toContain('not answering');
  });

  test('vouched machines are controllable — that is what the relay is for', () => {
    expect(targetable(paired({ pairing: 'vouched', reachedVia: DGX1 }))).toBe(true);
    expect(availability(paired({ pairing: 'vouched', reachedVia: DGX1 })).launch.enabled).toBe(
      true
    );
  });

  test('no selection disables everything', () => {
    const a = availability(null);
    for (const v of VERBS) expect(a[v.id].reason).toBe('No machine is selected.');
  });
});

describe('every action states where it will run', () => {
  test('local, direct remote, and relayed each read differently', () => {
    expect(route({ isLocal: true }, NODES)).toBe('runs on this machine');
    expect(route(paired(), NODES)).toBe('runs on dgx3');
    expect(route(paired({ reachedVia: DGX1 }), NODES)).toBe('runs on dgx3 · via dgx1');
  });

  test('mutating verbs on a relayed target warn about the travel', () => {
    expect(travelWarning(paired({ reachedVia: DGX1 }), NODES)).toBe(
      'This will travel through dgx1.'
    );
    expect(travelWarning(paired(), NODES)).toBeNull();
    expect(travelWarning(null, NODES)).toBeNull();
  });

  test('the wire target is null locally and the id remotely', () => {
    expect(onTarget({ isLocal: true, id: DGX1 })).toBeNull();
    expect(onTarget(paired())).toBe(DGX3);
  });
});
