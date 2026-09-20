// SPDX-License-Identifier: AGPL-3.0-only

import { describe, expect, test } from 'bun:test';
import {
  declaredLimit,
  fmtLimit,
  governingParam,
  limitFor,
  limitLabel,
  recordLimit,
  rungFloors,
  rungSpans,
  stepPath,
  timeSpans,
  violationOf
} from './gate-limits.js';

const MODEL = 'unsloth/Qwen3.8-27B-NVFP4';
// The shape gates.generated.json#gate_limits carries (gen-gates.mjs).
const TABLE = {
  'concurrency-sweep': {
    [MODEL]: { c1_aggregate_tok_s: { min: 20.4 }, c2_aggregate_tok_s: { min: 25 }, peak_aggregate_tok_s: { min: 109.5 } }
  },
  'decode-floor': { [MODEL]: { server_decode_tok_s: { min: 25 } } },
  'agentic-webserver': { [MODEL]: { sum_wall_s: { max: 5000 }, iterations: { min: 10, max: 10 } } }
};
// Every params value is a STRING: that is how the harness records them.
const rec = (over = {}) => ({ benchmark_id: 'concurrency-sweep', target_model: MODEL, params: {}, metrics: {}, ...over });

describe('which recorded param governs a metric', () => {
  test('a ladder rung maps to its min_c<C>, the rest to the descriptor names', () => {
    expect(governingParam('c8_aggregate_tok_s')).toEqual({ param: 'min_c8', bound: 'min' });
    expect(governingParam('c128_aggregate_tok_s')).toEqual({ param: 'min_c128', bound: 'min' });
    expect(governingParam('peak_aggregate_tok_s')).toEqual({ param: 'min_peak', bound: 'min' });
    expect(governingParam('sum_wall_s')).toEqual({ param: 'wall_budget_s', bound: 'max' });
  });
  test('NEGATIVE CONTROL: a metric with no recorded threshold has no governing param', () => {
    expect(governingParam('median_ms')).toBeNull();
    expect(governingParam('c8_ttft_p50_ms')).toBeNull();
  });
});

describe('the limit a record says judged it', () => {
  test('a recorded floor is read as a number, from the string the harness wrote', () => {
    expect(recordLimit(rec({ params: { min_c8: '45.7' } }), 'c8_aggregate_tok_s')).toEqual({ min: 45.7, max: null });
    expect(recordLimit(rec({ params: { wall_budget_s: '1800' } }), 'sum_wall_s')).toEqual({ min: null, max: 1800 });
  });
  test('0 is the OFF state, not a floor of zero', () => {
    expect(recordLimit(rec({ params: { min_c128: '0' } }), 'c128_aggregate_tok_s')).toEqual({ min: null, max: null });
  });
  test('an older BFCL record carries the floor only in its verdict text', () => {
    const r = rec({ benchmark_id: 'bfcl-subset', params: {}, verdict_reason: 'overall 84.1 >= 82.6 (floor 82.6)' });
    expect(recordLimit(r, 'overall_accuracy')).toEqual({ min: 82.6, max: null });
  });
  test('NEGATIVE CONTROL: the verdict text is read for overall_accuracy alone, and only when the param is absent', () => {
    const r = rec({ params: {}, verdict_reason: 'peak 100 >= 90 (floor 90)' });
    expect(recordLimit(r, 'peak_aggregate_tok_s')).toEqual({ min: null, max: null });
    const both = rec({ benchmark_id: 'bfcl-subset', params: { min_overall: '83' }, verdict_reason: '(floor 82.6)' });
    expect(recordLimit(both, 'overall_accuracy').min).toBe(83);
  });
  test('a param that is not a number is no limit', () => {
    expect(recordLimit(rec({ params: { min_c1: 'inherit' } }), 'c1_aggregate_tok_s')).toEqual({ min: null, max: null });
  });
});

describe('the declared limit and the merge', () => {
  test('declared is keyed by gate, checkpoint and metric', () => {
    expect(declaredLimit(TABLE, 'agentic-webserver', MODEL, 'iterations')).toEqual({ min: 10, max: 10 });
    expect(declaredLimit(TABLE, 'agentic-webserver', MODEL, 'sum_wall_s')).toEqual({ min: null, max: 5000 });
  });
  test('NEGATIVE CONTROL: another checkpoint, another gate, or an undeclared metric is no limit', () => {
    const none = { min: null, max: null };
    expect(declaredLimit(TABLE, 'agentic-webserver', 'Qwen/Qwen3.6-35B-A3B-FP8', 'sum_wall_s')).toEqual(none);
    expect(declaredLimit(TABLE, 'ttft-warm-gate', MODEL, 'sum_wall_s')).toEqual(none);
    expect(declaredLimit(TABLE, 'agentic-webserver', MODEL, 's_per_turn')).toEqual(none);
  });
  test('the record wins over the declaration, bound by bound', () => {
    // The record was judged at 24.5; BENCH.toml has since ratcheted to 25.
    expect(limitFor(rec({ benchmark_id: 'decode-floor', params: { min_tok_s: '24.5' } }), 'server_decode_tok_s', TABLE)).toEqual({
      min: 24.5,
      max: null
    });
    // No recorded ceiling: the declared one stands in.
    expect(limitFor(rec({ benchmark_id: 'agentic-webserver' }), 'sum_wall_s', TABLE)).toEqual({ min: null, max: 5000 });
    // A recorded ceiling beats the declared one.
    expect(limitFor(rec({ benchmark_id: 'agentic-webserver', params: { wall_budget_s: '1800' } }), 'sum_wall_s', TABLE).max).toBe(1800);
  });
  test('NEGATIVE CONTROL: a recorded OFF (0) falls through to the declaration', () => {
    expect(limitFor(rec({ params: { min_c2: '0' } }), 'c2_aggregate_tok_s', TABLE).min).toBe(25);
  });
});

describe('violations', () => {
  test('below the floor, over the ceiling; on the line passes', () => {
    expect(violationOf(24.9, { min: 25, max: null })).toBe('floor');
    expect(violationOf(25, { min: 25, max: null })).toBeNull();
    expect(violationOf(5001, { min: null, max: 5000 })).toBe('ceiling');
    expect(violationOf(5000, { min: null, max: 5000 })).toBeNull();
  });
  test('NEGATIVE CONTROL: no limit, no violation; a non-number is not judged', () => {
    expect(violationOf(-1e9, { min: null, max: null })).toBeNull();
    expect(violationOf(NaN, { min: 25, max: null })).toBeNull();
  });
});

describe('per-rung floors of a ladder record', () => {
  test('one floor per measured rung, in rung order, each from its own min_c<C>', () => {
    const r = rec({
      metrics: { c8_aggregate_tok_s: 50, c1_aggregate_tok_s: 21, c2_aggregate_tok_s: 26, peak_aggregate_tok_s: 50 },
      params: { min_c1: '20.099999999999998', min_c2: '24.05', min_c8: '45.7', min_peak: '50' }
    });
    expect(rungFloors(r, {})).toEqual([
      { c: 1, value: 20.099999999999998 },
      { c: 2, value: 24.05 },
      { c: 8, value: 45.7 }
    ]);
  });
  test('an OFF rung is omitted, and a rung without a recorded floor reads the declaration', () => {
    const r = rec({ metrics: { c1_aggregate_tok_s: 21, c2_aggregate_tok_s: 26 }, params: { min_c1: '0' } });
    expect(rungFloors(r, TABLE)).toEqual([
      { c: 1, value: 20.4 },
      { c: 2, value: 25 }
    ]);
    expect(rungFloors(r, {})).toEqual([]);
  });
});

describe('step geometry', () => {
  const y = (v) => 200 - v;
  test('time spans reach from the left edge to the right edge and merge equal neighbours', () => {
    const L = (min) => ({ min, max: null });
    const spans = timeSpans(
      [
        { x: 100, limit: L(24.5) },
        { x: 200, limit: L(24.5) },
        { x: 300, limit: L(26) }
      ],
      56,
      704
    );
    expect(spans).toEqual([
      { x0: 56, x1: 300, min: 24.5, max: null },
      { x0: 300, x1: 704, min: 26, max: null }
    ]);
    expect(stepPath(spans, 'min', y)).toBe('M56.0 175.5 H300.0 V174.0 H704.0');
  });
  test('a bound absent in the middle breaks the path into two sub-paths', () => {
    const spans = [
      { x0: 0, x1: 10, min: 5, max: null },
      { x0: 10, x1: 20, min: null, max: null },
      { x0: 20, x1: 30, min: 5, max: null }
    ];
    expect(stepPath(spans, 'min', y)).toBe('M0.0 195.0 H10.0 M20.0 195.0 H30.0');
  });
  test('NEGATIVE CONTROL: no span carries the bound, no path', () => {
    expect(stepPath([{ x0: 0, x1: 10, min: 5, max: null }], 'max', y)).toBe('');
    expect(stepPath([], 'min', y)).toBe('');
  });
  test('rung spans split at the geometric midpoint of a log axis and extend the ends by a half step', () => {
    const x = (c) => 100 * Math.log2(c); // C=1,2,4 at 0, 100, 200
    // log2(sqrt 2) carries float noise; the geometry is asserted to 1e-9.
    const tidy = (sp) => ({ ...sp, x0: +sp.x0.toFixed(9), x1: +sp.x1.toFixed(9) });
    const spans = rungSpans(
      [
        { c: 1, value: 20 },
        { c: 2, value: 25 },
        { c: 4, value: 25 }
      ],
      x
    ).map(tidy);
    expect(spans).toEqual([
      { x0: -50, x1: 50, min: 20, max: null, c: 1 },
      { x0: 50, x1: 250, min: 25, max: null, c: 2 }
    ]);
    expect(rungSpans([{ c: 8, value: 3 }], x, 18)).toEqual([{ x0: 282, x1: 318, min: 3, max: null, c: 8 }]);
  });
});

describe('labels', () => {
  test('float noise is trimmed, thousands are grouped, the bound is named', () => {
    expect(fmtLimit(20.099999999999998)).toBe('20.1');
    expect(fmtLimit(1800)).toBe('1,800');
    expect(limitLabel('min', 24.05, 'tok/s')).toBe('floor 24.05 tok/s');
    expect(limitLabel('max', 5000, 's')).toBe('ceiling 5,000 s');
    expect(limitLabel('min', 83)).toBe('floor 83');
  });
});
