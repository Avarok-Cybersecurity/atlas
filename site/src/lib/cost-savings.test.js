// SPDX-License-Identifier: AGPL-3.0-only
import { describe, expect, test } from 'bun:test';
import { savingsOverTime, SAVINGS_HORIZONS, DEFAULT_TOKENS_PER_DAY, costPerMillion } from './cost.js';

// One measured pair, so the arithmetic below is checkable by hand:
// baseline 2.0 J/tok, Atlas 1.2 J/tok, 1e9 tokens/day, $0.15/kWh, PUE 1.0.
//   saved per day = (2.0 - 1.2) * 1e9 / 3.6e6 kWh = 222.22... kWh
//   at $0.15                                      = $33.33...
const base = { atlasJPerTok: 1.2, baseJPerTok: 2.0, tokensPerDay: 1e9, usdPerKwh: 0.15, pue: 1.0, days: 30 };

describe('savingsOverTime', () => {
  test('the per-day figure is the joule difference, converted once', () => {
    const r = savingsOverTime(base);
    expect(r.state).toBe('measured');
    expect(r.perDayKwh).toBeCloseTo((0.8 * 1e9) / 3.6e6, 6);
    expect(r.perDayUsd).toBeCloseTo(r.perDayKwh * 0.15, 9);
  });

  test('it agrees with costPerMillion — the two must not drift apart', () => {
    // Savings per day must equal (baseline $/M − Atlas $/M) × millions per day.
    const perM = (j) => costPerMillion(j, base.usdPerKwh, base.pue);
    const expected = (perM(base.baseJPerTok) - perM(base.atlasJPerTok)) * (base.tokensPerDay / 1e6);
    expect(savingsOverTime(base).perDayUsd).toBeCloseTo(expected, 9);
  });

  test('cumulative is linear and the horizon endpoint is exact', () => {
    const r = savingsOverTime(base);
    expect(r.points[0]).toEqual({ day: 0, kwh: 0, usd: 0 });
    expect(r.points[r.points.length - 1].day).toBe(30);
    expect(r.totalUsd).toBeCloseTo(r.perDayUsd * 30, 9);
    // linear: the midpoint is half the total
    const mid = r.points.find((p) => p.day === 15);
    expect(mid.usd).toBeCloseTo(r.totalUsd / 2, 9);
  });

  test('a year is sampled, not 365 points, and still ends exactly on the horizon', () => {
    const r = savingsOverTime({ ...base, days: 365 });
    expect(r.points.length).toBeLessThanOrEqual(62);
    expect(r.points[r.points.length - 1].day).toBe(365);
    expect(r.points[r.points.length - 1].usd).toBeCloseTo(r.totalUsd, 9);
  });

  test('PUE multiplies the saving, because the cooling is saved too', () => {
    const one = savingsOverTime({ ...base, pue: 1.0 });
    const half = savingsOverTime({ ...base, pue: 1.5 });
    expect(half.perDayUsd).toBeCloseTo(one.perDayUsd * 1.5, 9);
  });

  test('a LOSS is reported as a loss, not clamped to zero', () => {
    // The whole point of the guard: if Atlas spends more per token, say so.
    const r = savingsOverTime({ ...base, atlasJPerTok: 2.5 });
    expect(r.state).toBe('measured');
    expect(r.perDayUsd).toBeLessThan(0);
    expect(r.totalUsd).toBeLessThan(0);
  });

  test('every missing input is refused BY NAME, not as a generic failure', () => {
    const why = (over) => savingsOverTime({ ...base, ...over });
    expect(why({ atlasJPerTok: null }).why).toContain('Atlas energy');
    expect(why({ baseJPerTok: undefined }).why).toContain('baseline');
    expect(why({ tokensPerDay: 0 }).why).toContain('demand');
    expect(why({ usdPerKwh: -1 }).why).toContain('electricity');
    expect(why({ pue: 0.5 }).why).toContain('PUE');
    expect(why({ days: 0 }).why).toContain('horizon');
    for (const over of [{ atlasJPerTok: null }, { days: 0 }]) expect(why(over).state).toBe('unavailable');
  });

  test('the horizons are real and the default demand is positive', () => {
    expect(SAVINGS_HORIZONS.map((h) => h.days)).toEqual([1, 30, 365]);
    expect(DEFAULT_TOKENS_PER_DAY).toBeGreaterThan(0);
  });
});
