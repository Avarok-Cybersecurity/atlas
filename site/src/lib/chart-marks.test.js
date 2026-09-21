// SPDX-License-Identifier: AGPL-3.0-only

import { describe, expect, test } from 'bun:test';
import { LONE_R, clipCaret, loneTriangle } from './chart-marks.js';

const nums = (d) => d.match(/-?\d+(\.\d+)?/g).map(Number);

describe('loneTriangle', () => {
  test('is closed, point-up, and centred on its circumcircle', () => {
    const d = loneTriangle(100, 50);
    expect(d.endsWith('Z')).toBe(true); // closed => it reads as filled
    const [ax, ay, bx, by, cx2, cy2] = nums(d);
    expect(ax).toBeCloseTo(100, 6);
    expect(ay).toBeCloseTo(50 - LONE_R, 1); // apex above centre
    expect(by).toBeCloseTo(cy2, 6); // flat base
    expect(bx + cx2).toBeCloseTo(200, 1); // symmetric about cx
    expect(by).toBeGreaterThan(ay);
  });
});

describe('clipCaret', () => {
  test('has no base edge, so it cannot be mistaken for the triangle', () => {
    expect(clipCaret(100, 14, 'high')).not.toContain('Z');
  });

  test('points off the top edge when clipped high', () => {
    const [, ay, , by] = nums(clipCaret(100, 14, 'high'));
    expect(by).toBeLessThan(ay); // apex is above the arms
  });

  test('points off the bottom edge when clipped low', () => {
    const [, ay, , by] = nums(clipCaret(100, 200, 'low'));
    expect(by).toBeGreaterThan(ay); // apex is below the arms
  });

  test('the two directions are mirror images, not the same glyph', () => {
    expect(clipCaret(100, 100, 'high')).not.toBe(clipCaret(100, 100, 'low'));
  });
});
