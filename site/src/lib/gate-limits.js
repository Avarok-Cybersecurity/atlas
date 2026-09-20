// SPDX-License-Identifier: AGPL-3.0-only

// gate-limits.js — which floor or ceiling governs a plotted metric, and how
// that limit is drawn.
//
// A gate limit has two homes, and the chart must read both:
//
//   * the RECORD. When a gate runs, the harness writes the threshold it
//     enforced into the record's `params` under the gate's own name for it —
//     `min_c8` for `c8_aggregate_tok_s`, `min_overall` for `overall_accuracy`,
//     `wall_budget_s` for `sum_wall_s`. That is the rule that judged THAT run,
//     so it is the one drawn under that point — a floor that has been ratcheted
//     three times draws as three steps, not as today's value painted over
//     history it never judged.
//
//   * BENCH.toml. `[benchmarks.metrics.<name>] min / max` is the declaration
//     in force now, carried into gates.generated.json by scripts/gen-gates.mjs
//     as `gate_limits[gate][checkpoint][metric]`. It is the only source for a
//     metric whose record carries no absolute (the TTFT gates record a
//     percentage against a baseline), and the fallback for older records
//     written before the harness recorded thresholds.
//
// Per record wins over declared, bound by bound. A per-rung floor (`min_c*`)
// is a per-record value by construction, which is the "prefer per-rung" rule.
//
// `0` in a recorded threshold is the gate's OFF state (see ladder-baselines.js
// on `min_c*`), never a floor of zero. A declared `max = 0` is a real ceiling
// (`vacuous_cells`), so the OFF rule applies to recorded params only.
//
// Pure and dependency-free so `bun test` can measure it directly.

/** One rung of the concurrency ladder — same shape as gates.js#LADDER_KEY. */
export const RUNG_METRIC = /^c(\d+)_aggregate_tok_s$/;

/**
 * Metric → the param the harness records its threshold under, and which bound
 * that threshold is. The names are the gate descriptors' (crates/avarok-plugin
 * /src/benchmarks); they are spelled once, here.
 */
const GOVERNING_PARAM = Object.freeze({
  peak_aggregate_tok_s: ['min_peak', 'min'],
  overall_accuracy: ['min_overall', 'min'],
  normalized_single_turn_score: ['min_normalized', 'min'],
  server_decode_tok_s: ['min_tok_s', 'min'],
  sum_wall_s: ['wall_budget_s', 'max'],
  s_per_turn: ['s_per_turn_budget', 'max']
});

/**
 * @param {string} metricKey
 * @returns {{param: string, bound: 'min'|'max'}|null}
 */
export function governingParam(metricKey) {
  const rung = RUNG_METRIC.exec(metricKey);
  if (rung) return { param: `min_c${rung[1]}`, bound: 'min' };
  const g = GOVERNING_PARAM[metricKey];
  return g ? { param: g[0], bound: g[1] } : null;
}

/**
 * @typedef {object} Limit
 * @property {number|null} min the floor, or null when none governs
 * @property {number|null} max the ceiling, or null when none governs
 */

const NONE = Object.freeze({ min: null, max: null });

const asThreshold = (raw) => {
  if (raw === undefined || raw === null || raw === '') return null;
  const v = typeof raw === 'number' ? raw : Number(String(raw).trim());
  if (!Number.isFinite(v)) return null;
  return v === 0 ? null : v;
};

/**
 * The limit a record says judged it, for one metric.
 *
 * Older BFCL records predate the `min_overall` param and carry the floor only
 * in the verdict text (`… (floor 82.6)`); that text is read for
 * `overall_accuracy` alone, and only when the param is absent.
 *
 * @param {object} record a gate record
 * @param {string} metricKey
 * @returns {Limit}
 */
export function recordLimit(record, metricKey) {
  const g = governingParam(metricKey);
  if (!g) return NONE;
  const v = asThreshold(record?.params?.[g.param]);
  if (v !== null) return { min: g.bound === 'min' ? v : null, max: g.bound === 'max' ? v : null };
  if (metricKey === 'overall_accuracy') {
    const m = /floor ([0-9.]+)/.exec(record?.verdict_reason ?? '');
    if (m) return { min: +m[1], max: null };
  }
  return NONE;
}

/**
 * The limit BENCH.toml declares for (gate, checkpoint, metric) right now.
 *
 * @param {Record<string, Record<string, Record<string, {min?: number, max?: number}>>>} table
 *   gates.generated.json#gate_limits
 * @param {string} benchmarkId
 * @param {string} checkpoint the record's target_model
 * @param {string} metricKey
 * @returns {Limit}
 */
export function declaredLimit(table, benchmarkId, checkpoint, metricKey) {
  const row = table?.[benchmarkId]?.[checkpoint]?.[metricKey];
  if (!row) return NONE;
  return { min: Number.isFinite(row.min) ? row.min : null, max: Number.isFinite(row.max) ? row.max : null };
}

/**
 * The limit to draw under a record's point: recorded first, declared second,
 * bound by bound.
 *
 * @param {object} record
 * @param {string} metricKey
 * @param {object} table gate_limits
 * @returns {Limit}
 */
export function limitFor(record, metricKey, table) {
  const rec = recordLimit(record, metricKey);
  const dec = declaredLimit(table, record?.benchmark_id, record?.target_model, metricKey);
  return { min: rec.min ?? dec.min, max: rec.max ?? dec.max };
}

export const sameLimit = (a, b) => a.min === b.min && a.max === b.max;
export const hasLimit = (l) => l.min !== null || l.max !== null;

/**
 * Which bound a value breaks, if any. A value ON the line passes: the gate's
 * own rule is `v < min` / `v > max`.
 *
 * @param {number} v
 * @param {Limit} limit
 * @returns {'floor'|'ceiling'|null}
 */
export function violationOf(v, limit) {
  if (!Number.isFinite(v)) return null;
  if (limit.min !== null && v < limit.min) return 'floor';
  if (limit.max !== null && v > limit.max) return 'ceiling';
  return null;
}

/**
 * @typedef {object} Span
 * @property {number} x0
 * @property {number} x1
 * @property {number|null} min
 * @property {number|null} max
 */

/**
 * Spans for a time series: each point's limit holds from its x until the next
 * point's x; the first reaches back to the plot's left edge and the last runs
 * to the right edge, so the rule is visible across the whole field it judged.
 * Adjacent spans with one limit merge.
 *
 * @param {Array<{x: number, limit: Limit}>} points ascending by x
 * @param {number} xStart the plot field's left edge
 * @param {number} xEnd the plot field's right edge
 * @returns {Span[]}
 */
export function timeSpans(points, xStart, xEnd) {
  const out = [];
  points.forEach((p, i) => {
    const x0 = i === 0 ? xStart : p.x;
    const x1 = i + 1 < points.length ? points[i + 1].x : xEnd;
    const last = out[out.length - 1];
    if (last && sameLimit(last, p.limit)) last.x1 = x1;
    else out.push({ x0, x1, min: p.limit.min, max: p.limit.max });
  });
  return out;
}

/**
 * Spans for a per-rung floor on a log2 concurrency axis: each rung's floor
 * holds from the geometric midpoint with its left neighbour to the one with
 * its right neighbour, and the end rungs extend by the same half step (a
 * fixed half step when there is only one rung, since it has no neighbour to
 * measure against). Adjacent rungs with one floor merge into one span.
 *
 * @param {Array<{c: number, value: number}>} floors ascending by c
 * @param {(c: number) => number} x the chart's concurrency scale
 * @param {number} [loneHalfStep] px, used only for a single rung
 * @returns {Array<Span & {c: number}>}
 */
export function rungSpans(floors, x, loneHalfStep = 18) {
  const mid = (a, b) => x(Math.sqrt(a * b));
  const out = [];
  floors.forEach((f, i) => {
    const prev = floors[i - 1];
    const next = floors[i + 1];
    const left = prev ? mid(prev.c, f.c) : next ? 2 * x(f.c) - mid(f.c, next.c) : x(f.c) - loneHalfStep;
    const right = next ? mid(f.c, next.c) : prev ? 2 * x(f.c) - mid(prev.c, f.c) : x(f.c) + loneHalfStep;
    const last = out[out.length - 1];
    if (last && last.min === f.value) last.x1 = right;
    else out.push({ x0: left, x1: right, min: f.value, max: null, c: f.c });
  });
  return out;
}

/**
 * The per-rung floors a ladder record was judged against, one per rung the
 * record measured, in rung order. Rungs with no floor (OFF, or undeclared)
 * are omitted — an absent floor is not a floor of zero.
 *
 * @param {object} record a concurrency-sweep record
 * @param {object} table gate_limits
 * @returns {Array<{c: number, value: number}>}
 */
export function rungFloors(record, table) {
  return Object.keys(record?.metrics ?? {})
    .map((k) => RUNG_METRIC.exec(k))
    .filter(Boolean)
    .map((m) => ({ c: +m[1], value: limitFor(record, m[0], table).min }))
    .filter((f) => f.value !== null)
    .sort((a, b) => a.c - b.c);
}

const r1 = (n) => n.toFixed(1);

/**
 * One SVG path through the spans' `bound` values as a step line: horizontal
 * along each span, vertical where the value changes, and a fresh sub-path
 * after any span with no such bound. Empty when no span carries the bound.
 *
 * @param {Span[]} spans in x order
 * @param {'min'|'max'} bound
 * @param {(v: number) => number} y the chart's value scale
 * @returns {string}
 */
export function stepPath(spans, bound, y) {
  let d = '';
  let pen = null;
  for (const s of spans) {
    const v = s[bound];
    if (v === null || v === undefined) {
      pen = null;
      continue;
    }
    const py = y(v);
    if (pen === null) d += `${d ? ' ' : ''}M${r1(s.x0)} ${r1(py)}`;
    else if (pen !== py) d += ` V${r1(py)}`;
    d += ` H${r1(s.x1)}`;
    pen = py;
  }
  return d;
}

/**
 * The value as the label prints it: thousands grouped, otherwise at most two
 * decimals with float noise (`20.099999999999998`) removed.
 */
export const fmtLimit = (v) =>
  Math.abs(v) >= 1000 ? Math.round(v).toLocaleString('en-US') : String(+v.toFixed(2));

/**
 * @param {'min'|'max'} bound
 * @param {number} value
 * @param {string} [unit] appended when it is a real unit (`ms`, `s`, `tok/s`)
 */
export const limitLabel = (bound, value, unit = '') =>
  `${bound === 'min' ? 'floor' : 'ceiling'} ${fmtLimit(value)}${unit ? ` ${unit}` : ''}`;
