// SPDX-License-Identifier: AGPL-3.0-only

import { describe, expect, test } from 'bun:test';
import { clampValue, dodgeLabels, percentile, robustDomain, tickLabel } from './gate-domain.js';

const ramp = (n, f = (i) => i) => Array.from({ length: n }, (_, i) => f(i));

describe('percentile', () => {
  test('interpolates between neighbours', () => {
    const s = ramp(11, (i) => i * 10); // 0..100
    expect(percentile(s, 0)).toBe(0);
    expect(percentile(s, 1)).toBe(100);
    expect(percentile(s, 0.5)).toBe(50);
    expect(percentile(s, 0.95)).toBeCloseTo(95, 6);
    // an off-by-one on the index, or nearest-rank instead of interpolation,
    // both miss this one
    expect(percentile(s, 0.27)).toBeCloseTo(27, 6);
  });

  test('degenerate inputs', () => {
    expect(percentile([], 0.5)).toBeNaN();
    expect(percentile([7], 0.95)).toBe(7);
  });
});

describe('robustDomain', () => {
  test('one extreme point does not blow up the axis', () => {
    // 40 readings around 100 and a single 10,000 — the ttft-cold shape.
    const values = [...ramp(40, () => 100), 10000];
    const d = robustDomain(values);
    expect(d.v1).toBeLessThan(500);
    expect(d.clipHigh).toBe(true);
  });

  test('the real ttft-cold shape: a flat series with one 23,484 ms spike', () => {
    // Regression guard for a hole this module had: when the robust band is
    // flat (q95 === q05), a `span > 0` test switched clipping off entirely —
    // exactly on the series that needs it most.
    const values = [...ramp(70, (i) => 1620 + (i % 7)), 23484];
    const d = robustDomain(values);
    expect(d.clipHigh).toBe(true);
    expect(d.v1).toBeLessThan(2000);
    expect(clampValue(23484, d).clamped).toBe('high');
  });

  test('a genuinely constant series is not treated as having an outlier', () => {
    const d = robustDomain(ramp(30, () => 1500));
    expect(d.clipHigh).toBe(false);
    expect(d.clipLow).toBe(false);
  });

  test('NEGATIVE CONTROL: a well-behaved series is not clipped at all', () => {
    // Without this, the test above would pass on an implementation that always
    // clipped to q95 and threw away real headroom.
    const d = robustDomain(ramp(40, (i) => 100 + i));
    expect(d.clipHigh).toBe(false);
    expect(d.clipLow).toBe(false);
    expect(d.v1).toBeGreaterThan(139); // the true max is inside the domain
  });

  test('clamping reports the position without touching the value', () => {
    const d = robustDomain([...ramp(40, () => 100), 10000]);
    const node = { v: 10000 };
    const { y, clamped } = clampValue(node.v, d);
    expect(clamped).toBe('high');
    expect(y).toBe(d.v1);
    expect(node.v).toBe(10000); // the true value survives for the tooltip
  });

  test('a low extreme clamps to the floor', () => {
    const d = robustDomain([...ramp(40, () => 1000), 1]);
    expect(d.clipLow).toBe(true);
    expect(clampValue(1, d)).toEqual({ y: d.v0, clamped: 'low' });
  });

  test('a value inside the domain is left exactly where it is', () => {
    const d = robustDomain(ramp(40, (i) => 100 + i));
    expect(clampValue(120, d)).toEqual({ y: 120, clamped: null });
  });

  test('a reference line is never clipped out of view', () => {
    // Values sit near 600; the budget cap is 1800. The axis must reach it.
    const d = robustDomain(ramp(30, (i) => 600 + i), [{ value: 1800 }]);
    expect(d.v1).toBeGreaterThan(1800);
    // and a floor below the data pulls the other end down
    const f = robustDomain(ramp(30, (i) => 600 + i), [{ value: 90 }]);
    expect(f.v0).toBeLessThan(90);
  });

  test('a positive-only metric never gets a negative axis', () => {
    // This is the -81 ms / -2,330 ms bug.
    for (const values of [ramp(30, (i) => 5 + i), [1000], ramp(40, () => 100)]) {
      expect(robustDomain(values).v0).toBeGreaterThanOrEqual(0);
    }
  });

  test('NEGATIVE CONTROL: the floor is data-derived, so real negatives survive', () => {
    // A hardcoded max(0, ...) would clip these away and this test would fail.
    const d = robustDomain(ramp(30, (i) => -50 + i));
    expect(d.v0).toBeLessThan(0);
  });

  test('degenerate inputs do not produce a zero-width or NaN axis', () => {
    expect(robustDomain([])).toBeNull();
    for (const values of [[5], [5, 5, 5, 5]]) {
      const d = robustDomain(values);
      expect(d.v1).toBeGreaterThan(d.v0);
      expect(Number.isFinite(d.v0) && Number.isFinite(d.v1)).toBe(true);
    }
  });

  test('non-finite values are ignored rather than poisoning the axis', () => {
    const d = robustDomain([100, NaN, 110, Infinity, 120]);
    expect(Number.isFinite(d.v0) && Number.isFinite(d.v1)).toBe(true);
  });
});

describe('tickLabel', () => {
  test('marks a clipped edge and leaves an unclipped one alone', () => {
    expect(tickLabel('1,240', 'high')).toBe('1,240+');
    expect(tickLabel('12', 'low')).toBe('12−');
    expect(tickLabel('1,240', null)).toBe('1,240');
  });
});

describe('dodgeLabels', () => {
  const box = { height: 13, top: 7, bottom: 202 };
  const minGap = (ys) => {
    const s = [...ys].sort((a, b) => a - b);
    return s.length < 2 ? Infinity : Math.min(...s.slice(1).map((y, i) => y - s[i]));
  };

  test('labels that already fit are not moved', () => {
    const want = [20, 60, 120];
    expect(dodgeLabels(want, box)).toEqual(want);
  });

  test('colliding labels are pushed apart to a full label height', () => {
    // the observed bug: 2,052 printed on top of 907.97
    const placed = dodgeLabels([100, 103], box);
    expect(minGap(placed)).toBeGreaterThanOrEqual(13);
  });

  test('output order matches input order', () => {
    const placed = dodgeLabels([103, 100], box);
    expect(placed[0]).toBeGreaterThan(placed[1]);
  });

  test('a whole pile separates and stays inside the field', () => {
    const placed = dodgeLabels([100, 100, 100, 100, 100, 100], box);
    expect(minGap(placed)).toBeGreaterThanOrEqual(13 - 1e-9);
    expect(Math.min(...placed)).toBeGreaterThanOrEqual(box.top - 1e-9);
    expect(Math.max(...placed)).toBeLessThanOrEqual(box.bottom + 1e-9);
  });

  test('a pile at the top edge is pushed down, not off the chart', () => {
    const placed = dodgeLabels([0, 1, 2], box);
    expect(Math.min(...placed)).toBeGreaterThanOrEqual(box.top - 1e-9);
    expect(minGap(placed)).toBeGreaterThanOrEqual(13 - 1e-9);
  });

  test('a pile at the bottom edge is pushed up', () => {
    const placed = dodgeLabels([210, 211, 212], box);
    expect(Math.max(...placed)).toBeLessThanOrEqual(box.bottom + 1e-9);
    expect(minGap(placed)).toBeGreaterThanOrEqual(13 - 1e-9);
  });

  test('displacement is shared rather than dumped on one label', () => {
    // Two labels wanting the same spot should move apart symmetrically; an
    // implementation that only ever pushes the later one down fails this.
    const placed = dodgeLabels([100, 100], box);
    expect(placed[0]).toBeCloseTo(93.5, 6);
    expect(placed[1]).toBeCloseTo(106.5, 6);
  });

  test('empty input', () => {
    expect(dodgeLabels([], box)).toEqual([]);
  });
});
