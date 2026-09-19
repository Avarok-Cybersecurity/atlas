// SPDX-License-Identifier: AGPL-3.0-only

// gate-series.js — turn a panel spec plus a pile of records into drawable series.
//
// The defect this module exists to fix: `GateChart` used to build ONE series
// per panel metric and colour the whole thing with
// `colorFor(series.points[0].rec.target_model)` — the FIRST point's model. On
// `ttft-cold-gate`, which holds four different checkpoints, that painted 77
// records in one model's colour, so a perfectly good 23,484 ms cold start on
// Gemma-4-26B read as an absurd outlier belonging to Qwen3.6-35B.
//
// The repo's rule was never wrong, only misapplied: colour identifies the
// MODEL, dash identifies the VARIANT. So a series here is one
// (metric, variant, model) family — never a mixture — and each gets its own
// colour, its own line and its own end label.
//
// Order of operations is load-bearing and is asserted by the tests:
//   split by family -> lineage on the RAW records -> aggregate -> lift edges.
// Lineage is a claim about commits. Running it after aggregation would mean
// inventing a commit identity for a group of them.

import { aggregateSeries, chartGroupSize, liftEdges } from './gate-aggregate.js';
import { trendEdges } from './gate-lineage.js';
import { shortModel } from './series-colors.js';

/**
 * A series with fewer than this many points gets no connecting line: two dots
 * joined by a segment assert a trend that two observations cannot support.
 * Those points are drawn as standalone marks instead.
 */
export const MIN_POINTS_FOR_A_LINE = 3;

/**
 * @typedef {object} Series
 * @property {string} key       stable identity, `${metric}|${variant}|${model}`
 * @property {string} metricKey the record metric this series reads
 * @property {string} label     what the legend shows
 * @property {string} metricLabel the metric's name alone, without the model
 * @property {string} model     target_model, the colour key
 * @property {string|null} variant benchmark id when the metric is variant-scoped
 * @property {boolean} dashed
 * @property {boolean} sparse   too short to justify a line
 * @property {Array<object>} nodes  plotted points (see gate-aggregate)
 * @property {Array<object>} edges  lifted lineage edges
 */

/**
 * Split one panel's records into per-(metric, variant, model) point lists.
 * Exported so a test can assert the split without the aggregation on top.
 *
 * @param {{metrics: Array<{key: string, label: string, variant?: string, dashed?: boolean}>}} panel
 * @param {Array<object>} records chronological
 * @returns {Array<{metric: object, model: string, pts: Array<object>}>}
 */
export function splitFamilies(panel, records) {
  const out = [];
  for (const metric of panel.metrics ?? []) {
    const byModel = new Map();
    for (const rec of records) {
      if (metric.variant && rec.benchmark_id !== metric.variant) continue;
      const v = rec.metrics?.[metric.key];
      if (!Number.isFinite(v)) continue;
      const model = rec.target_model ?? '';
      if (!byModel.has(model)) byModel.set(model, []);
      byModel.get(model).push({ t: rec.recorded_at, v, rec });
    }
    for (const [model, pts] of byModel) out.push({ metric, model, pts });
  }
  return out;
}

/**
 * Build every drawable series for a panel.
 *
 * @param {object} panel
 * @param {Array<object>} records chronological
 * @param {number} [cap] override for `MAX_VISIBLE_POINTS_PER_CHART`
 * @returns {Series[]}
 */
export function buildSeries(panel, records, cap) {
  const families = splitFamilies(panel, records);
  if (families.length === 0) return [];

  // One group size for the whole chart. Two series on one axis binned at
  // different resolutions would misstate their relative volatility.
  const g = chartGroupSize(families.map((f) => f.pts), cap);

  // Only disambiguate labels by model when the panel actually shows more than
  // one — "median · Qwen3.6-35B-A3B-FP8" on a single-model panel is noise.
  const models = new Set(families.map((f) => f.model));
  const manyModels = models.size > 1;

  return families.map(({ metric, model, pts }) => {
    const nodes = aggregateSeries(pts, g);
    return {
      key: `${metric.key}|${metric.variant ?? ''}|${model}`,
      metricKey: metric.key,
      label: manyModels ? `${metric.label} · ${shortModel(model)}` : metric.label,
      // The metric's own name, without the model suffix. The legend needs it
      // for the dash key, which describes the statistic and must not repeat
      // itself once per model.
      metricLabel: metric.label,
      model,
      variant: metric.variant ?? null,
      dashed: Boolean(metric.dashed),
      sparse: nodes.length < MIN_POINTS_FOR_A_LINE,
      nodes,
      edges: liftEdges(nodes, trendEdges(pts))
    };
  });
}

/** Every value that will actually be drawn — the input to the axis policy. */
export const drawnValues = (series) => series.flatMap((s) => s.nodes.map((n) => n.v));

/** The models present, in first-drawn order, for the legend. */
export const modelsOf = (series) => [...new Set(series.map((s) => s.model))];
