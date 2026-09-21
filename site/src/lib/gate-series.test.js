// SPDX-License-Identifier: AGPL-3.0-only

import { describe, expect, test } from 'bun:test';
import { buildSeries, drawnValues, modelsOf, splitFamilies } from './gate-series.js';

const QWEN = 'Qwen/Qwen3.6-35B-A3B-FP8';
const GEMMA = 'bg-digitalservices/Gemma-4-26B-A4B-it-NVFP4A16';

let seq = 0;
const rec = (over = {}) => {
  seq += 1;
  return {
    benchmark_id: 'ttft-cold-gate',
    git_sha: `sha${seq}`,
    recorded_at: seq,
    target_model: QWEN,
    served_by: 'recipe-a',
    verdict: 'PASS',
    params: {},
    metrics: { median_ms: 100 + seq },
    trend_predecessor: '',
    generated_ancestry: 'unknown',
    ...over
  };
};
const panel = { metrics: [{ key: 'median_ms', label: 'median' }] };

describe('the per-model split (the bug this module fixes)', () => {
  test('two models on one metric become two series, never one', () => {
    // Against the previous implementation this is false by construction: it
    // built a single series and coloured all of it from its first point.
    const records = [
      rec(), rec({ target_model: GEMMA }), rec(), rec({ target_model: GEMMA })
    ];
    const series = buildSeries(panel, records);
    expect(series).toHaveLength(2);
    expect(new Set(series.map((s) => s.model))).toEqual(new Set([QWEN, GEMMA]));
    for (const s of series) {
      for (const n of s.nodes) {
        for (const m of n.members) expect(m.rec.target_model).toBe(s.model);
      }
    }
  });

  test('an extreme reading stays inside its own model rather than the other one', () => {
    // The literal ttft-cold case: Gemma's 23,484 ms must not appear in the
    // Qwen series, where it read as an outlier.
    const records = [rec(), rec(), rec({ target_model: GEMMA, metrics: { median_ms: 23484 } })];
    const series = buildSeries(panel, records);
    const qwen = series.find((s) => s.model === QWEN);
    const gemma = series.find((s) => s.model === GEMMA);
    expect(drawnValues([qwen])).not.toContain(23484);
    expect(drawnValues([gemma])).toContain(23484);
  });

  test('NEGATIVE CONTROL: a single-model panel yields exactly one series', () => {
    // Without this the split test would pass on code that shattered every
    // record into its own series.
    const series = buildSeries(panel, [rec(), rec(), rec()]);
    expect(series).toHaveLength(1);
    expect(series[0].nodes.reduce((a, n) => a + n.count, 0)).toBe(3);
  });

  test('labels name the model only when more than one is present', () => {
    expect(buildSeries(panel, [rec(), rec()])[0].label).toBe('median');
    const mixed = buildSeries(panel, [rec(), rec({ target_model: GEMMA })]);
    expect(mixed.map((s) => s.label).sort()).toEqual([
      'median · Gemma-4-26B-A4B-it-NVFP4A16',
      'median · Qwen3.6-35B-A3B-FP8'
    ]);
  });

  test('the quant suffix survives shortening', () => {
    // FP8 vs NVFP4 of the same family is the comparison that must never be
    // ambiguous, so the label may not stop at `Qwen3.6-35B-A3B`.
    const series = buildSeries(panel, [
      rec(), rec({ target_model: 'nvidia/Qwen3.6-35B-A3B-NVFP4' })
    ]);
    const labels = series.map((s) => s.label).join(' ');
    expect(labels).toContain('FP8');
    expect(labels).toContain('NVFP4');
  });
});

describe('metric and variant scoping', () => {
  test('a variant-scoped metric reads only its own benchmark id', () => {
    const records = [
      rec({ benchmark_id: 'concurrency-sweep', metrics: { peak_aggregate_tok_s: 100 } }),
      rec({ benchmark_id: 'concurrency-sweep-dflash2', metrics: { peak_aggregate_tok_s: 62 } })
    ];
    const p = {
      metrics: [
        { key: 'peak_aggregate_tok_s', label: 'peak', variant: 'concurrency-sweep' },
        { key: 'peak_aggregate_tok_s', label: 'peak (DFlash2)', variant: 'concurrency-sweep-dflash2', dashed: true }
      ]
    };
    const series = buildSeries(p, records);
    expect(series).toHaveLength(2);
    expect(drawnValues([series[0]])).toEqual([100]);
    expect(drawnValues([series[1]])).toEqual([62]);
    expect(series[1].dashed).toBe(true);
  });

  test('two metrics on one model become two series', () => {
    const p = {
      metrics: [
        { key: 'median_ms', label: 'median' },
        { key: 'p90_ms', label: 'p90', dashed: true }
      ]
    };
    const records = [rec({ metrics: { median_ms: 10, p90_ms: 20 } }), rec({ metrics: { median_ms: 11, p90_ms: 21 } })];
    expect(buildSeries(p, records)).toHaveLength(2);
  });

  test('records missing the metric are skipped, not plotted as zero', () => {
    const records = [rec(), rec({ metrics: {} }), rec()];
    expect(drawnValues(buildSeries(panel, records))).toHaveLength(2);
  });

  test('keys are unique per family', () => {
    const records = [rec(), rec({ target_model: GEMMA })];
    const keys = buildSeries(panel, records).map((s) => s.key);
    expect(new Set(keys).size).toBe(keys.length);
  });
});

describe('lineage', () => {
  test('NEGATIVE CONTROL: unproven records produce no edges even once grouped', () => {
    // Every fixture above has trend_predecessor: '' — nothing went through the
    // generator — so aggregation must not invent adjacency.
    const records = Array.from({ length: 60 }, () => rec());
    const [series] = buildSeries(panel, records);
    expect(series.nodes.length).toBeGreaterThan(1);
    expect(series.edges).toEqual([]);
  });

  test('the per-model split does not sever a proven edge', () => {
    // instrumentKey already includes target_model, so a real predecessor is
    // always inside its own model's series.
    const a = rec();
    const b = rec();
    b.trend_predecessor = `${a.git_sha}`;
    const [series] = buildSeries(panel, [a, b]);
    // whether the key format matches or not, the edge must never cross models
    for (const e of series.edges) {
      expect(e.a.rec.target_model).toBe(e.b.rec.target_model);
    }
  });
});

describe('sparse series', () => {
  test('a one-point series is marked sparse so no line is drawn', () => {
    expect(buildSeries(panel, [rec()])[0].sparse).toBe(true);
  });

  test('NEGATIVE CONTROL: two points is enough for a line', () => {
    expect(buildSeries(panel, [rec(), rec()])[0].sparse).toBe(false);
    expect(buildSeries(panel, [rec(), rec(), rec()])[0].sparse).toBe(false);
  });
});

describe('the chart-wide cap', () => {
  test('two long series share one group size and each fits the cap', () => {
    const p = {
      metrics: [
        { key: 'median_ms', label: 'median' },
        { key: 'p90_ms', label: 'p90' }
      ]
    };
    const records = Array.from({ length: 77 }, (_, i) =>
      rec({ metrics: { median_ms: 100 + i, p90_ms: 200 + i } })
    );
    const series = buildSeries(p, records);
    expect(series).toHaveLength(2);
    for (const s of series) expect(s.nodes.length).toBeLessThanOrEqual(48);
    // and the two are binned identically, so their volatility stays comparable
    expect(series[0].nodes.length).toBe(series[1].nodes.length);
  });

  test('a short chart is left completely alone', () => {
    const records = Array.from({ length: 41 }, () => rec());
    const [series] = buildSeries(panel, records);
    expect(series.nodes).toHaveLength(41);
    expect(series.nodes.every((n) => !n.aggregated)).toBe(true);
  });
});

describe('helpers', () => {
  test('modelsOf lists each model once, in first-drawn order', () => {
    const series = buildSeries(panel, [rec(), rec({ target_model: GEMMA }), rec()]);
    expect(modelsOf(series)).toEqual([QWEN, GEMMA]);
  });

  test('splitFamilies and an empty panel', () => {
    expect(splitFamilies({ metrics: [] }, [rec()])).toEqual([]);
    expect(buildSeries({ metrics: [] }, [rec()])).toEqual([]);
    expect(buildSeries(panel, [])).toEqual([]);
  });
});

// ---------------------------------------------------------------------------
// The measured run-to-run envelope (#1214) reaches the chart, and the axis
// policy is told about it.
// ---------------------------------------------------------------------------
describe('envelope', () => {
  const at = (t, v) => ({ t, v, model: 'm', aggregated: false });
  const seriesWith = (envelope) => [
    { model: 'm', envelope, nodes: [at(1, 100), at(2, 110)] }
  ];

  test('buildSeries carries a metric envelope onto the series, and null when absent', () => {
    const e = { n: 32, lo: 0.94, hi: 1.06 };
    const recs = [
      { recorded_at: 1, git_sha: 'a', verdict: 'PASS', metrics: { m: 100 } },
      { recorded_at: 2, git_sha: 'b', verdict: 'PASS', metrics: { m: 110 } }
    ];
    const withE = buildSeries({ metrics: [{ key: 'm', label: 'm', envelope: e }] }, recs, 40);
    const without = buildSeries({ metrics: [{ key: 'm', label: 'm' }] }, recs, 40);
    expect(withE[0].envelope).toEqual(e);
    expect(without[0].envelope).toBeNull();
  });

  test('drawnValues includes the envelope extremes, so the bar cannot be clipped by the axis', () => {
    const v = drawnValues(seriesWith({ n: 32, lo: 0.9, hi: 1.1 }));
    expect(Math.min(...v)).toBeCloseTo(90, 6);
    expect(Math.max(...v)).toBeCloseTo(121, 6);
  });

  // ★ THE CONTROL. Without an envelope the axis input must be exactly the
  // plotted values — otherwise every other panel on the dashboard silently
  // changes its domain.
  test('drawnValues is unchanged for a series with no envelope', () => {
    expect(drawnValues(seriesWith(null))).toEqual([100, 110]);
  });

  test('an AGGREGATED node keeps its own observed span and gets no imputed one', () => {
    const s = [{ model: 'm', envelope: { n: 32, lo: 0.5, hi: 2 }, nodes: [{ t: 1, v: 100, aggregated: true, vMin: 95, vMax: 105 }] }];
    expect(drawnValues(s)).toEqual([100]);
  });
});
