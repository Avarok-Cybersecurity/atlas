// SPDX-License-Identifier: AGPL-3.0-only

import { describe, expect, test } from 'bun:test';
import * as S from './stats.js';

describe('a missing number is never rendered as zero', () => {
  test('every formatter dashes rather than inventing a value', () => {
    for (const f of [S.tokens, S.duration, S.percent, S.count]) {
      for (const v of [null, undefined, NaN, Infinity, -Infinity, 'x']) {
        expect(f(v)).toBe('—');
      }
    }
  });

  test('a real zero is still shown as zero', () => {
    expect(S.tokens(0)).toBe('0.00');
    expect(S.percent(0)).toBe('0%');
    expect(S.count(0)).toBe('0');
  });
});

describe('formatting', () => {
  test('rates gain precision as they get smaller', () => {
    expect(S.tokens(1234.5)).toBe('1235');
    expect(S.tokens(39.62)).toBe('39.6');
    expect(S.tokens(8.34)).toBe('8.34');
  });

  test('durations pick the unit a human would', () => {
    expect(S.duration(0.175)).toBe('175 ms');
    expect(S.duration(2.5)).toBe('2.50 s');
  });

  test('shares read as percentages', () => {
    expect(S.percent(0.69)).toBe('69%');
    expect(S.percent(1)).toBe('100%');
  });
});

describe('history', () => {
  test('it is bounded, keeping the most recent samples', () => {
    let h = [];
    for (let i = 0; i < S.HISTORY + 50; i++) h = S.push(h, i);
    expect(h.length).toBe(S.HISTORY);
    expect(h.at(-1)).toBe(S.HISTORY + 49);
  });

  // A gap is where the model was not answering. Skipping it would draw a
  // straight line across the outage as though nothing had happened.
  test('a missing reading is recorded as a gap, not dropped', () => {
    const h = S.push(S.push(S.push([], 1), null), 3);
    expect(h).toEqual([1, null, 3]);
  });
});

describe('sparkline', () => {
  test('nothing to draw yields no path at all', () => {
    expect(S.sparkline([], 100, 20)).toBe('');
    expect(S.sparkline([5], 100, 20)).toBe('');
    expect(S.sparkline([null, null], 100, 20)).toBe('');
  });

  test('a gap breaks the path instead of being interpolated through', () => {
    const d = S.sparkline([1, null, 3, 4], 100, 20);
    // Two pen-downs: one before the gap is not enough to draw, one after.
    expect((d.match(/M/g) ?? []).length).toBe(2);
  });

  test('the largest sample touches the top and the path stays in the box', () => {
    const d = S.sparkline([0, 5, 10], 100, 20);
    const ys = [...d.matchAll(/[ML]\S+ (\S+)/g)].map((m) => Number(m[1]));
    expect(Math.min(...ys)).toBeCloseTo(0, 5);
    expect(Math.max(...ys)).toBeCloseTo(20, 5);
  });

  /// A flat zero line must sit on the floor rather than float in the middle.
  test('an all-zero history draws along the bottom', () => {
    const d = S.sparkline([0, 0, 0], 100, 20);
    const ys = [...d.matchAll(/[ML]\S+ (\S+)/g)].map((m) => Number(m[1]));
    expect(ys.every((y) => y === 20)).toBe(true);
  });
});

describe('telling loading from idle', () => {
  test('a reading with any number in it says something', () => {
    expect(S.hasAnything({ requests_total: 0 })).toBe(true);
    expect(S.hasAnything({ decode_tokens_per_s: 39.6 })).toBe(true);
  });

  test('a reading with nothing in it says nothing', () => {
    expect(S.hasAnything(null)).toBe(false);
    expect(S.hasAnything({})).toBe(false);
    expect(S.hasAnything({ a: null, b: undefined })).toBe(false);
  });
});

describe('uptime reads at its order of magnitude', () => {
  test('each band formats coarsely', () => {
    expect(S.uptime(42)).toBe('42s');
    expect(S.uptime(90)).toBe('1m');
    expect(S.uptime(3660)).toBe('1h 1m');
    expect(S.uptime(3 * 86400 + 2 * 3600)).toBe('3d 2h');
  });

  test('absent is the dash, never zero', () => {
    expect(S.uptime(null)).toBe('—');
    expect(S.uptime(undefined)).toBe('—');
    expect(S.uptime(-5)).toBe('—');
  });
});

describe('timeline pins the sparkline to a fixed time axis', () => {
  test('always exactly HISTORY slots, samples at the left', () => {
    const t = S.timeline([1, 2, 3], { held: false });
    expect(t.length).toBe(S.HISTORY);
    expect(t.slice(0, 3)).toEqual([1, 2, 3]);
    expect(t[3]).toBeNull();
  });

  test('a full history is windowed to the newest HISTORY samples', () => {
    const long = Array.from({ length: S.HISTORY + 10 }, (_, i) => i);
    const t = S.timeline(long, { held: false });
    expect(t.length).toBe(S.HISTORY);
    expect(t[S.HISTORY - 1]).toBe(S.HISTORY + 9);
  });

  test('held leaves a visible gap at the right edge', () => {
    const full = Array.from({ length: S.HISTORY }, () => 5);
    const t = S.timeline(full, { held: true });
    expect(t.slice(-S.HOLD_GAP).every((v) => v === null)).toBe(true);
    // The samples slid left rather than being dropped from the count shown.
    expect(t[S.HISTORY - S.HOLD_GAP - 1]).toBe(5);
  });

  test('held does not move a line that already ends short', () => {
    const t = S.timeline([1, 2], { held: true });
    expect(t.slice(0, 2)).toEqual([1, 2]);
  });

  test('held must be said, not assumed', () => {
    expect(() => S.timeline([1])).toThrow(TypeError);
    expect(() => S.timeline([1], {})).toThrow(TypeError);
  });
});
