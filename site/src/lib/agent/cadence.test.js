// SPDX-License-Identifier: AGPL-3.0-only

import { describe, expect, test } from 'bun:test';
import * as C from './cadence.js';

const node = (id, selected = false, running = true) => ({ id, selected, running });

describe('what gets polled at all', () => {
  test('only running launches are polled', () => {
    const p = C.plan([node('a', true, false), node('b', false, false), node('c')], 2000);
    expect(p.map((e) => e.id)).toEqual(['c']);
  });

  test('the selected node runs at the chosen cadence, the rest at 10s', () => {
    const p = C.plan([node('a', true), node('b'), node('c')], 1000);
    expect(p.find((e) => e.id === 'a').periodMs).toBe(1000);
    expect(p.find((e) => e.id === 'b').periodMs).toBe(C.UNSELECTED_MS);
    expect(p.find((e) => e.id === 'c').periodMs).toBe(C.UNSELECTED_MS);
  });

  test('an unknown cadence is refused rather than obeyed', () => {
    // A typo'd 20ms would hammer the agent 50 times a second.
    for (const bad of [20, 0, undefined, '1s', NaN]) {
      expect(() => C.plan([node('a', true)], bad)).toThrow(TypeError);
    }
  });
});

describe('pause means no due polls, for anyone', () => {
  test('a paused plan is empty and nothing ever comes due', () => {
    const p = C.plan([node('a', true), node('b')], null);
    expect(p).toEqual([]);
    expect(C.due(p, {}, 1e12, 0)).toEqual([]);
    expect(C.isPaused(null)).toBe(true);
    expect(C.isPaused(2000)).toBe(false);
  });
});

describe('stagger: a relay never fans out simultaneous forwarded Stats', () => {
  test('every entry has a distinct offset', () => {
    const nodes = [node('a', true), ...'bcdefgh'.split('').map((id) => node(id))];
    const offsets = C.plan(nodes, 2000).map((e) => e.offsetMs);
    expect(new Set(offsets).size).toBe(offsets.length);
  });

  test('first polls land at the offsets, not all on the first tick', () => {
    const p = C.plan([node('a', true), node('b'), node('c')], 2000);
    // At the epoch itself only the selected node (offset 0) is due.
    expect(C.due(p, {}, 0, 0)).toEqual(['a']);
    // The others become due at their own offsets, one at a time.
    const later = C.due(p, {}, C.UNSELECTED_MS - 1, 0);
    expect(later).toContain('b');
    expect(later).toContain('c');
  });

  test('offsets are stable across re-plans of the same fleet', () => {
    const nodes = [node('a', true), node('b'), node('c')];
    expect(C.plan(nodes, 2000)).toEqual(C.plan([...nodes].reverse(), 2000));
  });
});

describe('the poll cycle', () => {
  test('a polled node is not due again until its period has passed', () => {
    const p = C.plan([node('a', true)], 2000);
    let s = C.polled({}, 'a', 1000);
    expect(C.due(p, s, 2999, 0)).toEqual([]);
    expect(C.due(p, s, 3000, 0)).toEqual(['a']);
  });

  test('cadence change mid-session takes effect at the next poll, without a burst', () => {
    let s = C.polled({}, 'a', 10_000);
    const fast = C.plan([node('a', true)], 1000);
    // Re-planned at t=10.2s from 5s to 1s: not due instantly...
    expect(C.due(fast, s, 10_200, 0)).toEqual([]);
    // ...due at lastAt + the NEW period, not the old one.
    expect(C.due(fast, s, 11_000, 0)).toEqual(['a']);
    const slow = C.plan([node('a', true)], 5000);
    expect(C.due(slow, s, 11_000, 0)).toEqual([]);
    expect(C.due(slow, s, 15_000, 0)).toEqual(['a']);
  });
});

describe('backoff on error', () => {
  test('failures double the wait and a success resets it', () => {
    const e = { periodMs: 2000, offsetMs: 0 };
    let s = C.failed({}, 'a', 0);
    expect(C.nextDue(e, s.a, 0)).toBe(4000);
    s = C.failed(s, 'a', 4000);
    expect(C.nextDue(e, s.a, 0)).toBe(4000 + 8000);
    s = C.failed(s, 'a', 12_000);
    expect(C.nextDue(e, s.a, 0)).toBe(12_000 + 16_000);
    s = C.polled(s, 'a', 28_000);
    expect(C.nextDue(e, s.a, 0)).toBe(28_000 + 2000);
  });

  test('backoff is capped so a recovered node is noticed within a minute', () => {
    const e = { periodMs: 10_000, offsetMs: 0 };
    let s = {};
    for (let i = 0; i < 10; i++) s = C.failed(s, 'a', 0);
    expect(C.nextDue(e, s.a, 0)).toBe(C.BACKOFF_MAX_MS);
  });
});
