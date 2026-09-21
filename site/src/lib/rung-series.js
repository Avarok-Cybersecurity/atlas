// SPDX-License-Identifier: AGPL-3.0-only

// rung-series.js — everything needed to draw ONE concurrency rung's history
// (plan-concurrency-ia §3). Pure: records in, a panel spec plus per-point
// facts out. The component draws what this says and adds nothing.
//
// x is TIME (`recorded_at`), never commit order: commits are unorderable
// across squash-merged PR branches and the campaign is not a recorded field
// (§3.2). `recorded_at` is epoch SECONDS, an integer, on every record in this
// repo, and it is read strictly — `Date.parse` on an integer yields NaN and
// silently collapses every record onto one instant (bfcl-partition.js has
// the same guard for the same reason).
//
// Three things a reader must never be able to misread:
//   * a floor is drawn only where a record DECLARED one — no floor, no line,
//     and a declared `0` is the gate's documented OFF state
//     (concurrency_verdict.rs: "0.0 is the documented OFF state"), not a
//     floor at zero;
//   * a baseline (one-shot vLLM) is drawn only when ladder-baselines.js says
//     its instrument is comparable with the latest point's; otherwise the
//     page gets the differing axes to print, not a line;
//   * two adjacent points on different instruments are never joined: the
//     first point of a new regime carries the diff for a vertical rule.
//
// Pure and dependency-free so `bun test` measures it directly.

import { comparable, describeDiffers, instrumentKey } from './ladder-baselines.js';

/**
 * The sweep's per-rung metric. gates.js#LADDER_KEY is the SSOT for this
 * shape; it cannot be imported here without dragging `$lib/*.generated.json`
 * into a pure module, so the test pins this builder against LADDER_KEY's
 * source text instead.
 */
export const rungMetricKey = (c) => `c${c}_aggregate_tok_s`;

/** A record's measurement time, strictly. See the header for why. */
const at = (rec) => {
  const t = rec?.recorded_at;
  if (typeof t !== 'number' || !Number.isFinite(t)) {
    throw new TypeError(`record ${rec?.git_sha ?? '?'} has no numeric recorded_at: ${JSON.stringify(t)}`);
  }
  return t;
};

/**
 * The floor a record declares for rung `c`, or null when it declares none.
 *
 * `min_c<N>` is a string in `params` (`"107.6"`, `"20.099999999999998"`).
 * Absent → null. `0` → null, because the gate itself treats a zero floor as
 * unpopulated (its verdict says "every populated floor met"). Anything
 * unreadable is a broken record and is refused by sha, never coerced to 0.
 */
export function floorOf(rec, c) {
  const raw = rec?.params?.[`min_c${c}`];
  if (raw === undefined || raw === null) return null;
  const v = Number(raw);
  if (typeof raw !== 'string' && typeof raw !== 'number') {
    throw new TypeError(`record ${rec.git_sha}: min_c${c} is ${JSON.stringify(raw)}, not a number`);
  }
  if (raw === '' || !Number.isFinite(v)) {
    throw new TypeError(`record ${rec.git_sha}: min_c${c} ${JSON.stringify(raw)} is not a number`);
  }
  return v > 0 ? v : null;
}

/** Floors print as the gate prints them in verdict_reason (`{floor:.1}`). */
const fmtFloor = (v) => v.toFixed(1);

/**
 * ISO date of a baseline's `measured_to`: an ISO-8601 string or epoch
 * seconds. A baseline with no date cannot be labelled "one-shot · <date>"
 * and is refused — the label is the reader's only cue that the line is a
 * snapshot, not a live series.
 */
function baselineDate(baseline) {
  const m = baseline.measured_to;
  if (typeof m === 'number' && Number.isFinite(m)) return new Date(m * 1000).toISOString().slice(0, 10);
  if (typeof m === 'string' && /^\d{4}-\d{2}-\d{2}/.test(m)) return m.slice(0, 10);
  throw new TypeError(`baseline ${baseline.engine ?? '?'} has no measured_to date: ${JSON.stringify(m)}`);
}

/**
 * @typedef {object} RungPoint
 * @property {number} t          recorded_at, epoch seconds
 * @property {number} v          c<N>_aggregate_tok_s
 * @property {object} rec        the record, by reference
 * @property {string} verdict    verbatim
 * @property {boolean} pass      verdict === 'PASS'
 * @property {number|null} floor the floor this record declared for the rung
 * @property {boolean} belowFloor v < floor (only ever true on a non-PASS)
 * @property {boolean} newLow    a PASS below every earlier same-instrument PASS
 * @property {string} instrument box-agnostic instrument key (ladder-baselines)
 */

/**
 * @typedef {object} RungPanel
 * @property {number} c
 * @property {string} key       the metric key GateChart reads
 * @property {string} title     `C=64`
 * @property {string} unit      `tok/s`
 * @property {boolean} measured any record measured this rung
 * @property {Array<{key: string, label: string}>} metrics GateChart panel shape
 * @property {Array<{value: number, label: string, from: number, to: number}>} floors
 *   distinct declared floors in order of first appearance, each with the
 *   time span over which it applied — a step when it changed
 * @property {Array<object>} caps always empty (a rung has no budget)
 * @property {Array<{kind: 'baseline', value: number, label: string, engine: string, date: string, baseline: object}>} refs
 *   one per baseline that is comparable with the latest point's instrument
 * @property {Array<{engine: string, kind: 'not-comparable'|'no-rung', differs?: object[], label: string}>} notes
 *   baselines that could not be drawn, and exactly why
 * @property {RungPoint[]} points chronological
 * @property {Array<{index: number, t: number, rec: object, differs: object[], label: string}>} regimes
 *   the first point of each new instrument regime, with what changed
 * @property {RungPoint|null} latest
 * @property {number} runs
 */

/**
 * Build the panel for rung `c`.
 *
 * @param {number} c
 * @param {object[]} records the subject's records, any order
 * @param {{baselines?: object[]}} [opts] `baselines` are the subject's
 *   one-shot runs from ladders.generated.json; omitted means none published,
 *   which is a state the page prints (§5.2), not a default.
 * @returns {RungPanel}
 */
export function rungPanel(c, records, opts = {}) {
  if (!Number.isInteger(c) || c <= 0) throw new TypeError(`rung must be a positive integer, got ${c}`);
  const key = rungMetricKey(c);
  const baselines = opts.baselines ?? [];

  const points = records
    .filter((rec) => Number.isFinite(rec?.metrics?.[key]))
    .map((rec) => ({
      t: at(rec),
      v: rec.metrics[key],
      rec,
      verdict: rec.verdict,
      pass: rec.verdict === 'PASS',
      floor: floorOf(rec, c),
      belowFloor: false,
      newLow: false,
      instrument: instrumentKey(rec)
    }))
    .sort((p, q) => p.t - q.t);

  // New low: below every EARLIER PASS on the SAME instrument, wherever in the
  // history that instrument was in use. A lower number on another instrument
  // is a different measurement, not a low. The first PASS on an instrument
  // has nothing earlier to be below.
  const lowSoFar = new Map();
  for (const p of points) {
    if (p.floor !== null) p.belowFloor = p.v < p.floor;
    if (!p.pass) continue;
    const prior = lowSoFar.get(p.instrument);
    p.newLow = prior !== undefined && p.v < prior;
    if (prior === undefined || p.v < prior) lowSoFar.set(p.instrument, p.v);
  }

  const regimes = [];
  for (let i = 1; i < points.length; i += 1) {
    if (points[i].instrument === points[i - 1].instrument) continue;
    const { differs } = comparable(points[i - 1].rec, points[i].rec);
    regimes.push({ index: i, t: points[i].t, rec: points[i].rec, differs, label: describeDiffers(differs) });
  }

  const floorSpans = new Map();
  for (const p of points) {
    if (p.floor === null) continue;
    const span = floorSpans.get(p.floor);
    if (span) span.to = p.t;
    else floorSpans.set(p.floor, { value: p.floor, label: `gate floor ${fmtFloor(p.floor)}`, from: p.t, to: p.t });
  }

  const latest = points.length > 0 ? points[points.length - 1] : null;
  const refs = [];
  const notes = [];
  for (const baseline of baselines) {
    const engine = String(baseline.engine ?? baseline.label ?? 'baseline');
    const rung = (baseline.rungs ?? []).find((r) => r?.c === c);
    if (!rung) {
      notes.push({ engine, kind: 'no-rung', label: `${engine} was not run at C=${c}` });
      continue;
    }
    if (!Number.isFinite(rung.tok_s)) {
      throw new TypeError(`baseline ${engine} rung C=${c} has no numeric tok_s: ${JSON.stringify(rung.tok_s)}`);
    }
    // No latest point means nothing to compare with; the note says so rather
    // than drawing a lone reference line over an empty panel.
    if (!latest) {
      notes.push({ engine, kind: 'no-rung', label: `nothing measured at C=${c} to compare ${engine} with` });
      continue;
    }
    const cmp = comparable(latest.rec, baseline);
    if (!cmp.ok) {
      notes.push({
        engine,
        kind: 'not-comparable',
        differs: cmp.differs,
        label: `${engine} — not comparable on this instrument (${describeDiffers(cmp.differs)})`
      });
      continue;
    }
    const date = baselineDate(baseline);
    refs.push({
      kind: 'baseline',
      value: rung.tok_s,
      label: `${engine} one-shot · ${date} · ${rung.tok_s.toFixed(1)}`,
      engine,
      date,
      baseline
    });
  }

  return {
    c,
    key,
    title: `C=${c}`,
    unit: 'tok/s',
    measured: points.length > 0,
    metrics: [{ key, label: 'aggregate tok/s' }],
    floors: [...floorSpans.values()],
    caps: [],
    refs,
    notes,
    points,
    regimes,
    latest,
    runs: points.length
  };
}
