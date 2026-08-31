// SPDX-License-Identifier: AGPL-3.0-only

import { describe, expect, test } from 'bun:test';
import { BAND_PERCENTILE_FROM, bandPath, historyBand, splitHistory } from './gate-band.js';

const run = (...vals) => ({ pts: vals.map((v, i) => ({ c: 2 ** i, v })) });

describe('historyBand', () => {
  test('a few runs give the full observed range', () => {
    const band = historyBand([run(10, 20), run(14, 26), run(12, 22)]);
    expect(band).toEqual([
      { c: 1, lo: 10, hi: 14 },
      { c: 2, lo: 20, hi: 26 }
    ]);
  });

  test('many runs trim to the 10th-90th percentile so one freak run cannot widen it', () => {
    const runs = [...Array(9)].map(() => run(100));
    const withFreak = [...runs, run(9999)];
    const band = historyBand(withFreak);
    expect(withFreak.length).toBeGreaterThanOrEqual(BAND_PERCENTILE_FROM);
    expect(band[0].hi).toBeLessThan(9999);
  });

  test('NEGATIVE CONTROL: below the threshold the same freak run IS included', () => {
    // Proves the percentile branch is actually selected by the run count and
    // not applied unconditionally.
    const band = historyBand([run(100), run(100), run(9999)]);
    expect(band[0].hi).toBe(9999);
  });

  test('a rung measured once has no range and is skipped', () => {
    // A zero-height sliver would draw a hairline that reads as data.
    const band = historyBand([{ pts: [{ c: 1, v: 10 }, { c: 2, v: 20 }] }, { pts: [{ c: 1, v: 12 }] }]);
    expect(band.map((b) => b.c)).toEqual([1]);
  });

  test('rungs come out ascending regardless of input order', () => {
    const a = { pts: [{ c: 64, v: 5 }, { c: 2, v: 1 }, { c: 8, v: 3 }] };
    const b = { pts: [{ c: 8, v: 4 }, { c: 64, v: 6 }, { c: 2, v: 2 }] };
    expect(historyBand([a, b]).map((x) => x.c)).toEqual([2, 8, 64]);
  });

  test('empty and malformed input', () => {
    expect(historyBand([])).toEqual([]);
    expect(historyBand([{ pts: [] }, {}])).toEqual([]);
    expect(historyBand([run(NaN), run(NaN)])).toEqual([]);
  });
});

describe('splitHistory', () => {
  test('keeps the newest two as lines and the rest as history', () => {
    const runs = [1, 2, 3, 4, 5].map((n) => ({ n }));
    const { latest, previous, history } = splitHistory(runs);
    expect(latest.n).toBe(5);
    expect(previous.n).toBe(4);
    expect(history.map((h) => h.n)).toEqual([1, 2, 3]);
  });

  test('short histories degrade without inventing runs', () => {
    expect(splitHistory([])).toEqual({ latest: null, previous: null, history: [] });
    const one = splitHistory([{ n: 1 }]);
    expect(one.latest.n).toBe(1);
    expect(one.previous).toBeNull();
    expect(one.history).toEqual([]);
    const two = splitHistory([{ n: 1 }, { n: 2 }]);
    expect(two.previous.n).toBe(1);
    expect(two.history).toEqual([]);
  });
});

describe('bandPath', () => {
  test('is a closed region, not a line', () => {
    const d = bandPath([{ c: 1, lo: 1, hi: 2 }, { c: 2, lo: 2, hi: 4 }], (c) => c * 10, (v) => v * 10);
    expect(d.endsWith('Z')).toBe(true);
    expect(d.startsWith('M')).toBe(true);
  });

  test('a band that cannot enclose anything draws nothing', () => {
    expect(bandPath([], (c) => c, (v) => v)).toBe('');
    expect(bandPath([{ c: 1, lo: 1, hi: 2 }], (c) => c, (v) => v)).toBe('');
  });
});
