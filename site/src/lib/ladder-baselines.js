// SPDX-License-Identifier: AGPL-3.0-only

// ladder-baselines.js — may two throughput measurements share one axis?
//
// The confusion this module exists to make impossible: the published ladder
// (bench/ladder38) measured vLLM at ISL 128 / OSL 1024, ctx 2048, batch 128,
// and until 2026-09-21 the live gate measured Atlas at ISL 512 / OSL 320, ctx
// 4096, batch 128. Same checkpoint, ~478 vs ~116 tok/s, and readers who saw
// both on one chart concluded the engine was inconsistent. The gate has since
// been re-pointed onto the published axes (#1220), so the two DO pair now --
// but only because this module says so axis by axis, and a record from the
// retired instrument still in .benchmarks/ must keep failing to. Nothing here
// is allowed to say "same
// instrument" from a name or a checkpoint: it is decided axis by axis, and a
// mismatch names every axis that differs so the page can print them.
//
// Two kinds of measurement are fingerprinted:
//   * a GATE RECORD (gates.generated.json) — instrument in `params` and
//     `serve_overrides`, both string-valued as the harness recorded them;
//   * a BASELINE (plan-concurrency-ia §2.3, `ladders.generated.json`) — a
//     one-shot vLLM run whose `instrument` object carries the same axes.
//
// The fingerprint is BOX-AGNOSTIC (plan §3.2): `machine_id`, `perf_class`,
// `hardware.driver` and the thermal snapshot are the box, not the workload —
// the six Speed-class boxes are one population under BENCH.toml's
// SPEED_SPREAD equivalence, and the driver is one-per-box in this record set,
// so keying on either would fragment 23 widened runs into six trails. Gate
// THRESHOLDS (`min_c*`, `min_peak`) are excluded too: a floor is the gate's
// verdict rule, not the instrument, and it steps over time on one instrument.
//
// gate-lineage.js keys its trend edges on the same params/serve_overrides
// PLUS the box (a lineage edge is a stronger claim than "may share an axis").
// The test pins that this fingerprint disagrees with lineage only on the box,
// the thresholds and `served_by` — never on a workload axis. `served_by`
// names the recipe (engine configuration, speculation included); lineage
// keys on it because a trend edge is a claim about one code line, while a
// shared axis is a claim about one WORKLOAD — the two gate variants are
// meant to share an axis (gate-variants.js), and so is a vLLM one-shot.
//
// Pure and dependency-free so `bun test` measures it directly.

/**
 * The axes every measurement must declare before it may be drawn against
 * another. Absent on either side is a difference, never a match: a baseline
 * that forgot to say its KV dtype is not comparable to anything.
 */
export const REQUIRED_AXES = Object.freeze([
  'isl',
  'osl',
  'prompt_mode',
  'max_model_len',
  'max_batch_size',
  'kv_cache_dtype'
]);

/** A gate floor: verdict rule, not instrument. `0` is the floor's OFF state. */
export const THRESHOLD_PARAM = /^min_(c\d+|peak)$/;

// The harness spells the ISL list in the plural (`isls: "512"`); a baseline
// manifest spells the single value `isl`. One name on the fingerprint.
const PARAM_AXIS = Object.freeze({ isls: 'isl' });

/**
 * One scalar axis value as a string, so `"4096"` from a gate record and
 * `4096` from a manifest compare equal. `null` means undeclared.
 */
const norm = (axis, value) => {
  if (value === null || value === undefined) return null;
  if (typeof value === 'string') return value.trim();
  if (typeof value === 'number') {
    if (!Number.isFinite(value)) throw new TypeError(`axis ${axis} is not finite: ${value}`);
    return String(value);
  }
  if (typeof value === 'boolean') return String(value);
  throw new TypeError(`axis ${axis} must be a scalar, got ${JSON.stringify(value)}`);
};

const setAxis = (axes, axis, value) => {
  if (axis in axes) throw new TypeError(`axis ${axis} declared twice (params and serve_overrides)`);
  const v = norm(axis, value);
  if (v !== null) axes[axis] = v;
};

function gateAxes(record) {
  const axes = {};
  setAxis(axes, 'checkpoint', record.target_model);
  setAxis(axes, 'gpu', record.hardware?.gpu);
  for (const [k, v] of Object.entries(record.params ?? {})) {
    if (THRESHOLD_PARAM.test(k)) continue;
    setAxis(axes, PARAM_AXIS[k] ?? k, v);
  }
  for (const [k, v] of Object.entries(record.serve_overrides ?? {})) setAxis(axes, k, v);
  return axes;
}

function baselineAxes(baseline) {
  const axes = {};
  setAxis(axes, 'checkpoint', baseline.checkpoint);
  setAxis(axes, 'gpu', baseline.gpu);
  for (const [k, v] of Object.entries(baseline.instrument)) setAxis(axes, PARAM_AXIS[k] ?? k, v);
  return axes;
}

/**
 * @typedef {object} Instrument
 * @property {'gate'|'baseline'} kind
 * @property {Record<string, string>} axes box-agnostic, threshold-free,
 *   every value a trimmed string; undeclared axes are simply absent
 */

/**
 * The box-agnostic instrument fingerprint of a gate record or a baseline.
 *
 * Kind is decided by shape, not by a flag the caller could get wrong: a gate
 * record carries `params`, a baseline carries `instrument`. Both or neither
 * is refused — guessing would let a malformed manifest fingerprint as a gate
 * record and pass every check below.
 *
 * @param {object} x
 * @returns {Instrument}
 */
export function instrumentOf(x) {
  const isGate = x?.params !== undefined && x.params !== null;
  const isBaseline = x?.instrument !== undefined && x.instrument !== null;
  if (isGate === isBaseline) {
    throw new TypeError(
      `instrumentOf: expected a gate record (params) or a baseline (instrument), got ${JSON.stringify(
        x && Object.keys(x)
      )}`
    );
  }
  if (isGate && typeof x.params !== 'object') throw new TypeError('gate record params must be an object');
  if (isBaseline && typeof x.instrument !== 'object') throw new TypeError('baseline instrument must be an object');
  return isGate ? { kind: 'gate', axes: gateAxes(x) } : { kind: 'baseline', axes: baselineAxes(x) };
}

/**
 * A stable string for "same instrument" equality — what rung-series.js uses
 * to decide whether two adjacent points may be joined by a line.
 */
export const instrumentKey = (x) => {
  const { axes } = instrumentOf(x);
  return JSON.stringify(Object.keys(axes).sort().map((k) => [k, axes[k]]));
};

/**
 * @typedef {object} Difference
 * @property {string} axis
 * @property {string|null} a value on the first side, null when undeclared
 * @property {string|null} b value on the second side, null when undeclared
 */

/**
 * Decide whether `a` and `b` may be drawn on one axis, naming every axis that
 * says no.
 *
 * Rules, in order:
 *   1. Every REQUIRED axis must be declared on both sides and equal.
 *   2. Same kind (gate vs gate): every remaining axis must match, presence
 *      included — an override present on one record and absent on the other
 *      is a different serve configuration (`dflash=true` vs the recipe's
 *      default), and `concurrencies 1,4,8,16` vs `1…128` is the regime
 *      change the chart must draw a rule at.
 *   3. Different kinds (gate vs baseline): remaining axes are compared only
 *      where BOTH declare them. A vLLM manifest has no `ssm_cache_slots` and
 *      a gate record has no `reps`/`seed`; neither absence is a mismatch.
 *      The engine, its speculation and its build are deliberately not axes:
 *      a baseline is a different engine by definition, and it is comparable
 *      by workload, not by identity.
 *
 * @param {object} a gate record or baseline
 * @param {object} b gate record or baseline
 * @returns {{ok: boolean, differs: Difference[]}} differs is empty iff ok;
 *   required axes come first in REQUIRED_AXES order, then the rest by name
 */
export function comparable(a, b) {
  const A = instrumentOf(a);
  const B = instrumentOf(b);
  const differs = [];

  for (const axis of REQUIRED_AXES) {
    const va = A.axes[axis] ?? null;
    const vb = B.axes[axis] ?? null;
    if (va === null || vb === null || va !== vb) differs.push({ axis, a: va, b: vb });
  }

  const required = new Set(REQUIRED_AXES);
  const extras = [...new Set([...Object.keys(A.axes), ...Object.keys(B.axes)])]
    .filter((axis) => !required.has(axis))
    .sort();
  const crossKind = A.kind !== B.kind;
  for (const axis of extras) {
    const va = A.axes[axis] ?? null;
    const vb = B.axes[axis] ?? null;
    if (crossKind && (va === null || vb === null)) continue;
    if (va !== vb) differs.push({ axis, a: va, b: vb });
  }

  return { ok: differs.length === 0, differs };
}

/**
 * The differences as the page prints them: `isl 512 → 128, osl 320 → 1024`.
 * An undeclared side prints as `undeclared` so the reader sees a missing
 * manifest field for what it is rather than an empty string.
 */
export const describeDiffers = (differs) =>
  differs.map(({ axis, a, b }) => `${axis} ${a ?? 'undeclared'} → ${b ?? 'undeclared'}`).join(', ');
