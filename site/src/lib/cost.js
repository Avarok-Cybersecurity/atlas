// SPDX-License-Identifier: AGPL-3.0-only

// cost.js — what a token costs in electricity, and every honesty rule around
// that number. Pure and dependency-free (apart from the two instrument
// modules) so `bun test` measures it directly.
//
// # Store joules and tokens, derive every ratio
//
// The producer (crates/avarok-plugin/src/hardware/energy.rs) stores JOULES and
// the TOKEN COUNT of the window they span, and nothing else: joules and tokens
// are additive and survive re-aggregation, a ratio does not. So J/token,
// tokens/Wh and dollars are computed HERE, from that pair, and none of them is
// ever read from a record.
//
//   J/token   = energy_j / tokens
//   tok/Wh    = tokens / energy_j × 3600
//   W         = energy_j / window_s
//   $/1M tok  = J/token × ($/kWh) / 3.6
//
// tokens/W and tokens/J are the SAME quantity — `(tok/s) / (J/s) = tok/J` — so
// only one efficiency unit appears on an axis anywhere in the section.
//
// # The rail is in every key name, and it is a LOWER BOUND
//
// On GB10 `nvidia-smi` exposes the GPU rail only: `power.limit`, Module Power
// Readings and GPU Memory Power all answer N/A, so Grace's cores and the
// LPDDR5X are OUTSIDE every number here. The producer puts `gpu_rail` in each
// key name for exactly that reason and this module keeps the name; the page
// states the bound ON the chart, not in a footnote.
//
// # Four things a reader must never be able to misread
//
//   * an ABSENT key is "not measured", never 0 — a zero joule count reads as
//     FREE, which is the one thing this section must not claim;
//   * an UNDER-SAMPLED window is marked and left out of every tile, verdict
//     and trend line, but stays visible — silently averaging it would launder
//     a guess into a measurement;
//   * two measurements on different instruments are never joined (the
//     `ladder-baselines.js` rule, plus the rail and the sampler cadence);
//   * a LOSING rung is drawn exactly like a winning one. There is no filter
//     and no default view that hides one.

import { comparable, describeDiffers, instrumentKey } from './ladder-baselines.js';
import { baselineSeriesOf, comparisonStateOf, ladderFor, liveRecordOf } from './concurrency-comparison.js';

/**
 * The per-window energy keys, spelled exactly as `EnergyWindow::metrics`
 * emits them. A cell of the concurrency sweep carries each under a `c{C}_`
 * prefix; a decode-floor run carries them unprefixed. `cost-keys.test.js`
 * reads the Rust source and proves these strings are the producer's.
 */
export const KEY = Object.freeze({
  energyJ: 'gpu_rail_energy_j',
  tokens: 'gpu_rail_energy_window_tokens',
  windowS: 'gpu_rail_energy_window_s',
  samples: 'gpu_rail_power_samples',
  meanW: 'gpu_rail_mean_power_w',
  maxW: 'gpu_rail_max_power_w',
  aboveIdleJ: 'gpu_rail_energy_above_idle_j',
  swCapFrac: 'gpu_rail_sw_power_cap_frac',
  hwBrakeFrac: 'gpu_rail_hw_power_brake_frac'
});

/** The run-level keys, from `EnergySampler::metrics` / `SamplerCost::metrics`. */
export const RUN_KEY = Object.freeze({
  idleW: 'gpu_rail_idle_power_w',
  idleSamples: 'gpu_rail_idle_power_samples',
  periodMs: 'gpu_rail_sample_period_ms'
});

/** One cell's key, `c8_gpu_rail_energy_j`. */
export const cellKey = (c, name) => `c${c}_${name}`;

/**
 * The electricity price is the reader's, not a measurement: 0.15 $/kWh is a
 * round retail-commercial placeholder. It lives in localStorage per viewer and
 * NEVER in the URL — a deep link must not carry someone else's assumption.
 */
export const DEFAULT_USD_PER_KWH = 0.15;
export const PRICE_STORAGE_KEY = 'atlas.cost.usd_per_kwh';

/**
 * POWER USAGE EFFECTIVENESS — the reader's second input, and like the price it
 * is an assumption rather than a measurement.
 *
 * PUE is total facility energy divided by IT energy: cooling, power
 * conversion, lighting and everything else the building spends to keep the
 * machine running. A figure on this page is GPU-rail joules, so multiplying by
 * PUE is what turns "what the part drew" into "what the meter in the building
 * reads".
 *
 * ★ THE DEFAULT IS 1.0, AND THAT IS NOT A GUESS — it is the identity. At 1.0
 * nothing is applied and the chart is exactly the rail reading it has always
 * been. Every other value is the reader's own facility, typed in. The page
 * never assumes a multiplier on a number that is already a lower bound.
 *
 * ★ PUE CANNOT CHANGE WHO WINS. It multiplies both engines equally, so every
 * ratio, verdict and tile on this tab is invariant under it. That is asserted
 * in cost.test.js rather than merely stated here, because it is the property
 * that makes offering the control safe: a reader can explore their own
 * facility's overhead without being able to move the comparison.
 *
 * The LOW/HIGH pair exists because a facility is a range, not a number, and a
 * reader who does not know theirs to two decimals should not have to pretend.
 * Equal low and high is one curve; a gap is a band, and the band is the honest
 * shape of "somewhere in here".
 */
export const DEFAULT_PUE = 1.0;
/** Below 1.0 a building would be producing energy. A hard floor, not a taste. */
export const PUE_MIN = 1.0;
/**
 * A typo guard, NOT a physical bound. Real facilities above 2.0 exist (and
 * have existed at 3.0); this stops a stray keystroke rescaling the axis by a
 * thousand, and the input says so rather than implying 3.0 is impossible.
 */
export const PUE_MAX = 3.0;
/**
 * The band offered as a preset, and the one figure pair this module asserts
 * about the outside world: 1.1-1.5 is the range this tab's own disclosure has
 * quoted for a datacentre since it was written. Offered, never applied.
 */
export const PUE_TYPICAL = Object.freeze({ low: 1.1, high: 1.5 });
export const PUE_LOW_STORAGE_KEY = 'atlas.cost.pue_low';
export const PUE_HIGH_STORAGE_KEY = 'atlas.cost.pue_high';

/** A usable PUE: finite and in [PUE_MIN, PUE_MAX]. Anything else is not one. */
export const isPue = (v) => Number.isFinite(v) && v >= PUE_MIN && v <= PUE_MAX;

/**
 * The pair the charts draw, from whatever the two boxes currently hold.
 *
 * Invalid input does not silently become a default that then looks measured:
 * an unusable box falls back to `DEFAULT_PUE` (the identity), and the caller
 * is told which box was refused so the page can say so next to it. A high
 * below the low is not an error either — it is two numbers a reader typed in
 * the order that suited them, so they are ordered rather than rejected.
 *
 * @param {number} low @param {number} high
 * @returns {{low: number, high: number, band: boolean, lowOk: boolean, highOk: boolean}}
 */
export function pueRange(low, high) {
  const lowOk = isPue(low);
  const highOk = isPue(high);
  const a = lowOk ? low : DEFAULT_PUE;
  const b = highOk ? high : DEFAULT_PUE;
  return { low: Math.min(a, b), high: Math.max(a, b), band: a !== b, lowOk, highOk };
}

/**
 * Trust rules for one window, calibrated against the producer's own pinned
 * cadence (`SAMPLE_PERIOD_MS = 250`, four readings a second): ten readings is
 * 2.5 s of evidence, and a sampler that covered less than 90% of the window
 * integrated over a gap it did not watch.
 */
export const MIN_POWER_SAMPLES = 10;
export const MIN_SAMPLE_COVERAGE = 0.9;

// ★ THE FLOOR UNDER A SPREAD, AND WHY IT IS TEN.
// The range of n i.i.d. draws covers about (n-1)/(n+1) of the population it is
// drawn from: 82% at n=10, 89% at n=17. Below ten, the observed min-max is a
// truncated view of the real spread, and a truncated spread makes "this is a
// real fall" too easy to claim -- the one direction this page must never make
// easier. Ten is the smallest n where the envelope is more honest than silence.
export const MIN_SPREAD_RUNS = 10;

/**
 * `tokens / window_s` must agree with the cell's own `c{C}_aggregate_tok_s`:
 * the producer derives both from ONE batch (`throughput = tokens / wall`, the
 * energy window being that batch's first-sent → last-complete), so a
 * disagreement means the joules and the tokens are not from the same window.
 * Such a record is refused by sha and named, never coerced.
 */
export const CROSS_CHECK_TOLERANCE = 0.05;

/** The sweep's per-rung throughput key. gates.js#LADDER_KEY is the SSOT for the shape. */
export const RUNG_THROUGHPUT = /^c(\d+)_aggregate_tok_s$/;
export const rungThroughputKey = (c) => `c${c}_aggregate_tok_s`;

/** The derived trend metric. Never stored on a record — `costTrend` adds it to a copy. */
export const trendMetricKey = (c) => `c${c}_tok_per_wh`;

// ---- derivations ------------------------------------------------------------

export const jPerToken = (energyJ, tokens) => energyJ / tokens;
export const wattsOf = (energyJ, windowS) => energyJ / windowS;
export const tokPerWh = (tokens, energyJ) => (tokens / energyJ) * 3600;

/**
 * Dollars per million tokens, at a price and a facility overhead.
 *
 *   1e6 tokens × J/token = J;  J / 3.6e6 = kWh;  × PUE × $/kWh = $
 *   ⇒ $ = J/token × ($/kWh) × PUE / 3.6
 *
 * Worked: 80 W at 25 tok/s is 3.2 J/token; a million tokens is 11.1 h and
 * 0.89 kWh, which at $0.15/kWh and PUE 1.0 is $0.13, and at PUE 1.5 is $0.20.
 *
 * ★ `pue` IS REQUIRED, deliberately. It was tempting to default it to 1.0 —
 * the identity, so nothing would break — but a defaulted overhead is exactly
 * the kind of assumption that later becomes invisible: every call site must
 * say which facility multiplier its number carries, and a caller that forgets
 * gets NaN rather than a figure that silently means something else.
 */
export const costPerMillion = (jPerTok, usdPerKwh, pue) => (jPerTok * usdPerKwh * pue) / 3.6;

// ---- savings over time ------------------------------------------------------

/**
 * Horizons the savings view offers. Days, not months, because a month is not a
 * fixed length and a cumulative line that jumps at month boundaries is a lie
 * about the rate.
 */
export const SAVINGS_HORIZONS = Object.freeze([
  Object.freeze({ days: 1, label: 'a day' }),
  Object.freeze({ days: 30, label: '30 days' }),
  Object.freeze({ days: 365, label: 'a year' })
]);

/** Demand the savings view assumes until the reader sets their own. */
export const DEFAULT_TOKENS_PER_DAY = 1e9;
export const TOKENS_PER_DAY_STORAGE_KEY = 'atlas.cost.tokens_per_day';

/**
 * Cumulative energy and money saved by serving a FIXED token demand on Atlas
 * instead of the baseline.
 *
 * ★ THE COMPARISON IS SAME-WORKLOAD, AND THAT CHOICE IS THE WHOLE MEANING.
 * Two honest questions exist: "the same N tokens, which engine spends less?"
 * and "the same box flat out, which delivers more?". Only the first is a
 * SAVING — the second is a throughput win and would be double-counted if it
 * were folded in here. So demand is held fixed at `tokensPerDay` and the only
 * difference is joules per token. A reader who wants the throughput story has
 * the ladder chart directly above.
 *
 * ★ A NEGATIVE RESULT IS RETURNED, NOT CLAMPED. If Atlas spends more per token
 * at some rung, this says so with a negative number. A savings calculator that
 * cannot express a loss is marketing, not measurement.
 *
 * ★ `pue` IS REQUIRED, for the same reason `costPerMillion` requires it.
 *
 * @returns {{state:'measured', perDayKwh:number, perDayUsd:number,
 *            points:{day:number,kwh:number,usd:number}[],
 *            totalKwh:number, totalUsd:number}
 *          |{state:'unavailable', why:string}}
 */
export function savingsOverTime({ atlasJPerTok, baseJPerTok, tokensPerDay, usdPerKwh, pue, days }) {
  // Each missing input is named individually: "cannot compute" sends a reader
  // looking at the wrong thing, and the usual absence here is one specific
  // one -- Atlas energy that has not been measured on this instrument yet.
  if (!Number.isFinite(atlasJPerTok)) return { state: 'unavailable', why: 'Atlas energy is not measured on this instrument' };
  if (!Number.isFinite(baseJPerTok)) return { state: 'unavailable', why: 'the baseline carries no energy-instrumented run' };
  if (!Number.isFinite(tokensPerDay) || tokensPerDay <= 0) return { state: 'unavailable', why: 'daily token demand must be a positive number' };
  if (!Number.isFinite(usdPerKwh) || usdPerKwh < 0) return { state: 'unavailable', why: 'electricity price must be zero or more' };
  if (!isPue(pue)) return { state: 'unavailable', why: `PUE must be between ${PUE_MIN} and ${PUE_MAX}` };
  if (!Number.isFinite(days) || days <= 0) return { state: 'unavailable', why: 'the horizon must be a positive number of days' };

  // J -> kWh is 3.6e6. Facility overhead multiplies the whole draw, so it
  // multiplies the DIFFERENCE too -- a datacentre saves the cooling it would
  // have spent on the joules it no longer burns.
  const perDayKwh = ((baseJPerTok - atlasJPerTok) * tokensPerDay * pue) / 3.6e6;
  const perDayUsd = perDayKwh * usdPerKwh;
  // One point per day up to 60, then sampled, so a year does not ship 365
  // points into an SVG path that is 200px wide.
  const step = days <= 60 ? 1 : Math.ceil(days / 60);
  const points = [];
  for (let d = 0; d <= days; d += step) points.push({ day: d, kwh: perDayKwh * d, usd: perDayUsd * d });
  if (points[points.length - 1].day !== days) points.push({ day: days, kwh: perDayKwh * days, usd: perDayUsd * days });
  return { state: 'measured', perDayKwh, perDayUsd, points, totalKwh: perDayKwh * days, totalUsd: perDayUsd * days };
}

/** Two significant figures below a dollar, two decimals above: `0.13`, `0.0054`, `12.40`. */
export const fmtUsd = (v) => (v >= 1 ? v.toFixed(2) : Number(v.toPrecision(2)).toString());

// ---- reading one window -----------------------------------------------------

const num = (v) => (typeof v === 'number' && Number.isFinite(v) ? v : null);

/**
 * @typedef {object} EnergyCell
 * @property {'absent'|'refused'|'measured'} state
 * @property {string} reason        why it is absent or refused; `''` when measured
 * @property {number} [energyJ]     as recorded
 * @property {number} [tokens]      delivered inside the same window
 * @property {number} [windowS]
 * @property {number|null} [samples]
 * @property {number} [jPerToken] @property {number} [tokPerWh] @property {number} [watts]
 * @property {number|null} [aboveIdleJ] @property {number|null} [idleW]
 * @property {number|null} [swCapFrac] @property {number|null} [hwBrakeFrac]
 * @property {number|null} [periodMs]
 * @property {boolean} [trusted]    false ⇒ out of every tile, verdict and trend
 * @property {boolean} [throttled]  HW power brake asserted ⇒ out of the verdicts
 * @property {string[]} [concerns]  each printed verbatim in the tooltip
 */

/**
 * Read one energy window out of a flat map of numbers.
 *
 * @param {object} m         the record's `metrics`, or a ladder rung
 * @param {string} prefix    `'c8_'` for a sweep cell, `''` for a rung
 * @param {object} run       where the run-level keys live (idle, sampler period)
 * @param {string} id        what a refusal names — a sha, or `series C=8`
 * @param {number|null} expectTokS  the cell's own tok/s for the cross-check
 * @returns {EnergyCell}
 */
export function readEnergy(m, prefix, run, id, expectTokS) {
  const at = (k) => m?.[`${prefix}${k}`];
  const raw = at(KEY.energyJ);
  // Absent is NOT zero. Nothing downstream may turn this into a number.
  if (raw === undefined || raw === null) return { state: 'absent', reason: 'not measured' };

  const refuse = (reason) => ({ state: 'refused', reason: `${id} — ${reason}` });
  const energyJ = num(raw);
  const tokens = num(at(KEY.tokens));
  const windowS = num(at(KEY.windowS));
  if (energyJ === null) return refuse(`${prefix}${KEY.energyJ} is ${JSON.stringify(raw)}, not a number`);
  if (energyJ <= 0)
    return refuse(`${prefix}${KEY.energyJ} is ${energyJ} — a window cannot cost nothing (or less)`);
  if (tokens === null || tokens <= 0)
    return refuse(`${energyJ.toFixed(1)} J with no token count (${prefix}${KEY.tokens}) — joules with no denominator`);
  if (windowS === null || windowS <= 0) return refuse(`${prefix}${KEY.windowS} is not a positive duration`);

  // The joules and the tokens must describe ONE window. See CROSS_CHECK_TOLERANCE.
  const expect = num(expectTokS);
  if (expect !== null && expect > 0) {
    const off = Math.abs(tokens / windowS - expect) / expect;
    if (off > CROSS_CHECK_TOLERANCE)
      return refuse(
        `${tokens} tok over ${windowS.toFixed(1)} s is ${(tokens / windowS).toFixed(1)} tok/s, ` +
          `but the cell recorded ${expect.toFixed(1)} tok/s (${(off * 100).toFixed(1)}% apart) — ` +
          `the joules and the tokens are not from the same window`
      );
  }

  const samples = num(at(KEY.samples));
  const periodMs = num(run?.[RUN_KEY.periodMs]);
  const concerns = [];
  if (samples === null) concerns.push('no sample count recorded — the joule count is unauditable');
  else if (samples < MIN_POWER_SAMPLES)
    concerns.push(`under-sampled: ${samples} readings over ${windowS.toFixed(1)} s`);
  else if (periodMs !== null && periodMs > 0) {
    const covered = (samples * periodMs) / 1000 / windowS;
    if (covered < MIN_SAMPLE_COVERAGE)
      concerns.push(
        `the sampler covered ${(covered * 100).toFixed(0)}% of the ${windowS.toFixed(1)} s window ` +
          `(${samples} readings at ${periodMs} ms)`
      );
  }
  // ★ ABSENT IS NOT ZERO, APPLIED TO THE GUARD ITSELF. Coverage is
  // `samples x period / window`, so with no recorded cadence it cannot be
  // computed -- and the chain above used to simply fall off its end: a record
  // with ample samples and no period collected NO concern at all, was marked
  // trusted, drawn solid, and counted in every tile, verdict and trend line
  // with its coverage never checked. The window could have been sampled at a
  // cadence leaving most of it unobserved and the page would have shown a full
  // measurement. Only 2 of the 72 committed concurrency-sweep records carry the
  // key, so this was inert almost everywhere it mattered.
  else
    concerns.push(
      `sampler cadence not recorded (${RUN_KEY.periodMs}) — coverage of the ` +
        `${windowS.toFixed(1)} s window cannot be verified`
    );
  // SW Power Cap is this box's NORMAL steady state under load (energy.rs), so
  // it is reported and never disqualifying. The HW power brake is not normal.
  const hwBrakeFrac = num(at(KEY.hwBrakeFrac));
  const throttled = hwBrakeFrac !== null && hwBrakeFrac > 0;
  if (throttled) concerns.push(`HW power brake asserted on ${(hwBrakeFrac * 100).toFixed(0)}% of readings`);

  return {
    state: 'measured',
    reason: '',
    energyJ,
    tokens,
    windowS,
    samples,
    meanW: num(at(KEY.meanW)),
    maxW: num(at(KEY.maxW)),
    aboveIdleJ: num(at(KEY.aboveIdleJ)),
    idleW: num(run?.[RUN_KEY.idleW]),
    swCapFrac: num(at(KEY.swCapFrac)),
    hwBrakeFrac,
    periodMs,
    jPerToken: jPerToken(energyJ, tokens),
    tokPerWh: tokPerWh(tokens, energyJ),
    watts: wattsOf(energyJ, windowS),
    trusted: concerns.length === 0 || (concerns.length === 1 && throttled),
    throttled,
    concerns
  };
}

/** One rung of one gate record. */
export function energyOf(record, c) {
  const m = record?.metrics ?? {};
  return readEnergy(m, `c${c}_`, m, `record ${record?.git_sha ?? '?'} C=${c}`, m[rungThroughputKey(c)]);
}

/** One rung of a published/one-shot ladder series. */
export function energyOfRung(seriesLabel, rung) {
  return readEnergy(rung, '', rung, `${seriesLabel} C=${rung?.c}`, rung?.tok_s);
}

/** The rungs a record measured throughput at, ascending. */
export const rungsOf = (record) =>
  Object.keys(record?.metrics ?? {})
    .map((k) => RUNG_THROUGHPUT.exec(k))
    .filter(Boolean)
    .map((m) => +m[1])
    .sort((a, b) => a - b);

/**
 * "Over time" is a sequence on ONE cost instrument. A cost instrument is the
 * workload instrument (ladder-baselines.js) PLUS the rail and the sampler
 * cadence: a joule count taken at another cadence is not the same measurement,
 * and an unrecorded cadence is its own regime rather than a match with one.
 */
export const costInstrumentKey = (record) =>
  `${instrumentKey(record)}|gpu_rail|${num(record?.metrics?.[RUN_KEY.periodMs]) ?? 'unrecorded-period'}`;

// ---- chart A: cost against the concurrency ladder ---------------------------

/**
 * @typedef {object} CostPoint
 * @property {number} c
 * @property {EnergyCell} energy
 */

const pointsOfRecord = (record) => rungsOf(record).map((c) => ({ c, energy: energyOf(record, c) }));
const pointsOfSeries = (series) =>
  (series.rungs ?? []).map((r) => ({ c: r.c, energy: energyOfRung(series.label ?? series.id, r) }));
const anyMeasured = (points) => points.some((p) => p.energy.state === 'measured');

/**
 * What chart A draws for one subject, before a price is applied — the series
 * are J/token; dollars are a relabelling.
 *
 * The Atlas/vLLM pairing decision is `comparisonStateOf`'s, the SAME decision
 * the throughput comparison makes, intersected with "carries energy". A
 * baseline whose instrument differs is refused by ladder-baselines.js and
 * NAMED; one that is comparable but carries no joules is named too, as
 * exactly that — neither is drawn and neither is a zero.
 *
 * @param {object} subject a concurrency-subjects.json entry
 * @param {object[]} records the subject's gate records, chronological
 * @param {object} ladders ladders.generated.json
 */
export function costLadder(subject, records, ladders) {
  const state = comparisonStateOf(subject, records, ladders);
  const ladder = ladderFor(subject, ladders);
  const live = liveRecordOf(records);
  const refused = [];
  const noEnergy = [];

  let atlas = null;
  if (state === 'published') {
    const s = ladder.series.find((x) => x.role === 'subject');
    const points = pointsOfSeries(s);
    atlas = { label: s.label, source: 'published campaign', build: s.build, points };
  } else if (live) {
    atlas = {
      label: 'Atlas',
      source: 'latest gate run on main',
      build: live.git_sha,
      record: live,
      points: pointsOfRecord(live)
    };
  }

  const candidates = ladder ? baselineSeriesOf(ladder) : [];
  const baselines = [];
  for (const b of candidates) {
    // In the live state the vLLM leg must fingerprint as the gate record's
    // instrument; in the published state the manifest already pairs them.
    if (state === 'live' && live) {
      const cmp = comparable(live, { checkpoint: ladder.workload.checkpoint, instrument: b.instrument });
      if (!cmp.ok) {
        refused.push({ label: b.label, why: describeDiffers(cmp.differs) });
        continue;
      }
    }
    const points = pointsOfSeries(b);
    if (!anyMeasured(points)) {
      noEnergy.push({ label: b.label, why: 'no energy-instrumented run on this instrument' });
      continue;
    }
    baselines.push({ id: b.id, label: b.label, series: b, points });
  }

  const verdicts = atlas ? rungVerdicts(atlas.points, baselines) : { rungs: [], k: 0, n: 0 };
  const atlasHasEnergy = Boolean(atlas) && anyMeasured(atlas.points);
  return {
    state,
    energyState: !atlasHasEnergy ? 'none' : baselines.length === 0 ? 'atlas-only' : 'paired',
    atlas,
    baselines,
    refused,
    noEnergy,
    verdicts,
    idle: idleAvailability(atlas, baselines),
    refusals: [atlas, ...baselines]
      .filter(Boolean)
      .flatMap((s) => s.points.filter((p) => p.energy.state === 'refused').map((p) => p.energy.reason))
  };
}

/**
 * `above idle` is offered only when idle was recorded on BOTH sides drawn —
 * subtracting a baseline one engine measured and the other did not would
 * flatter whichever one recorded it.
 */
export function idleAvailability(atlas, baselines) {
  const sides = [atlas, ...baselines].filter(Boolean);
  if (sides.length < 2) return { both: false, why: 'only one side is drawn' };
  const missing = sides
    .filter((s) => !s.points.some((p) => p.energy.state === 'measured' && p.energy.idleW !== null))
    .map((s) => s.label);
  return missing.length === 0
    ? { both: true, why: '' }
    : { both: false, why: `idle not recorded for ${missing.join(', ')}` };
}

/**
 * Per rung: is Atlas cheaper, and by what factor. Every rung with a
 * measurement on both sides is returned — there is no filter — but only a
 * rung whose BOTH cells are trusted and unthrottled is COUNTED in k of n.
 *
 * @returns {{rungs: Array<object>, k: number, n: number}}
 */
export function rungVerdicts(atlasPoints, baselines) {
  const rungs = [];
  for (const p of atlasPoints) {
    if (p.energy.state !== 'measured') continue;
    const rivals = baselines
      .map((b) => ({ label: b.label, point: b.points.find((q) => q.c === p.c) }))
      .filter((r) => r.point && r.point.energy.state === 'measured');
    if (rivals.length === 0) continue;
    const best = rivals.reduce((a, b) => (b.point.energy.jPerToken < a.point.energy.jPerToken ? b : a));
    const ratio = p.energy.jPerToken / best.point.energy.jPerToken;
    const excluded = [p.energy, best.point.energy].some((e) => !e.trusted || e.throttled);
    rungs.push({
      c: p.c,
      atlas: p.energy,
      rival: best.point.energy,
      rivalLabel: best.label,
      ratio,
      atlasCheaper: ratio < 1,
      counted: !excluded,
      // The label drawn under a losing point, derived and never typed.
      loseLabel: ratio >= 1 ? `vLLM cheaper ×${ratio.toFixed(2)}` : null
    });
  }
  const counted = rungs.filter((r) => r.counted);
  return { rungs, k: counted.filter((r) => r.atlasCheaper).length, n: counted.length };
}

/**
 * The headline tile. It flips to naming the loser at k = 0 — the section has
 * no wording in which losing every rung reads as a win.
 */
export function verdictTile({ k, n }) {
  if (n === 0) return 'no rung measured on both engines';
  return k === 0 ? `vLLM cheaper at ${n} of ${n} rungs` : `cheaper at ${k} of ${n} rungs`;
}

/** `best`/`worst` by cost ratio, over the counted rungs only. */
export function extremeRungs({ rungs }) {
  const counted = rungs.filter((r) => r.counted);
  if (counted.length === 0) return { best: null, worst: null };
  return {
    best: counted.reduce((a, b) => (b.ratio < a.ratio ? b : a)),
    worst: counted.reduce((a, b) => (b.ratio > a.ratio ? b : a))
  };
}

// ---- chart B: efficiency over time ------------------------------------------

/**
 * The tokens-per-Wh trend for one rung, as a GateChart panel plus the records
 * it reads — the derived key is added to a SHALLOW COPY, never to the stored
 * record.
 *
 * Points on different cost instruments get DIFFERENT metric keys, so GateChart
 * draws them as separate series and no line ever crosses a regime change. The
 * newest instrument keeps the plain key and is drawn solid; each earlier one
 * is dashed and labelled with what changed after it.
 *
 * Untrusted and refused cells are excluded here (they are named in `excluded`
 * and stay visible on chart A): a trend line through a window the sampler
 * barely watched is a guess with a line through it.
 */
const median = (xs) => {
  const v = [...xs].sort((a, b) => a - b);
  const m = v.length >> 1;
  return v.length % 2 ? v[m] : (v[m - 1] + v[m]) / 2;
};

/**
 * The measured run-to-run spread of rung `c` on one instrument, as a relative
 * multiplicative envelope around a plotted tokens-per-Wh point.
 *
 * ★ THE POPULATION IS KEYED ON `instrumentKey`, NEVER `costInstrumentKey`.
 * `costInstrumentKey` appends `gpu_rail_sample_period_ms`, and almost no record
 * carries that key -- keying on it collapses the population to the same two
 * points the envelope is supposed to judge, and a band drawn from the points it
 * judges is circular. The denominator's spread is a property of the SERVE
 * instrument, so it is measured over every PASS run of that instrument,
 * including the many that carry no joules at all.
 *
 * tok/Wh = (tok/s) / W, so the envelope is multiplicative: the throughput
 * factor comes from the whole population, the rail factor only from this
 * generation's measured points (usually one, in which case it is 1).
 *
 * ★ IT CAN ONLY EVER SUPPRESS A CLAIM. Real movement inside the window widens
 * the envelope; nothing narrows it. So this cannot manufacture a trend, only
 * refuse one -- which is the only direction it is safe to be wrong in here.
 *
 * @returns {{n:number,lo:number,hi:number,tMin:number,tMax:number,tMedian:number,powerMeasured:number}|null}
 *   null when the instrument has fewer than `MIN_SPREAD_RUNS` usable runs.
 */
export function rungSpread(c, key, records) {
  const tk = rungThroughputKey(c);
  const t = [];
  for (const rec of records || []) {
    if (rec?.verdict !== 'PASS') continue;          // never average a failed run
    if (instrumentKey(rec) !== key) continue;
    const v = num(rec?.metrics?.[tk]);
    if (v !== null && v > 0) t.push(v);
  }
  if (t.length < MIN_SPREAD_RUNS) return null;
  const tMed = median(t);
  const w = [];
  for (const rec of records || []) {
    if (instrumentKey(rec) !== key) continue;
    const v = num(rec?.metrics?.[cellKey(c, KEY.meanW)]);
    if (v !== null && v > 0) w.push(v);
  }
  const wMed = w.length ? median(w) : null;
  const wLoF = wMed ? Math.max(...w) / wMed : 1;    // more watts => fewer tok/Wh
  const wHiF = wMed ? Math.min(...w) / wMed : 1;
  return {
    n: t.length,
    lo: Math.min(...t) / tMed / wLoF,
    hi: Math.max(...t) / tMed / wHiF,
    tMin: Math.min(...t),
    tMax: Math.max(...t),
    tMedian: tMed,
    powerMeasured: w.length
  };
}

/**
 * Can two consecutive plotted points be told apart, given the envelope?
 * Only when their bars do not overlap. Returns 'overlap' (the honest default),
 * 'separated', or 'unmeasured' when there is no envelope to judge against.
 */
export function spreadVerdict(values, envelope) {
  const v = (values || []).filter((x) => num(x) !== null && x > 0);
  if (!envelope) return { state: 'unmeasured', n: 0 };
  if (v.length < 2) return { state: 'unmeasured', n: envelope.n };
  const a = v[v.length - 2];
  const b = v[v.length - 1];
  const rose = b * envelope.lo > a * envelope.hi;
  const fell = a * envelope.lo > b * envelope.hi;
  return { state: rose || fell ? 'separated' : 'overlap', direction: rose ? 'up' : fell ? 'down' : null, n: envelope.n };
}

export function costTrend(c, records) {
  const excluded = [];
  const usable = [];
  for (const rec of records) {
    const e = energyOf(rec, c);
    if (e.state === 'absent') continue;
    if (e.state === 'refused') {
      excluded.push({ sha: rec.git_sha, reason: e.reason });
      continue;
    }
    if (!e.trusted) {
      excluded.push({ sha: rec.git_sha, reason: `${rec.git_sha} C=${c} — ${e.concerns.join('; ')}` });
      continue;
    }
    usable.push({ rec, e });
  }
  usable.sort((a, b) => a.rec.recorded_at - b.rec.recorded_at);

  const gens = [];
  for (const u of usable) {
    const key = costInstrumentKey(u.rec);
    const last = gens[gens.length - 1];
    if (last && last.instrument === key) last.members.push(u);
    else gens.push({ instrument: key, members: [u] });
  }

  const base = trendMetricKey(c);
  const metrics = [];
  const derived = [];
  gens.forEach((g, i) => {
    const newest = i === gens.length - 1;
    g.key = newest ? base : `${base}__g${i + 1}`;
    if (!newest) {
      const { differs } = comparable(g.members[g.members.length - 1].rec, gens[i + 1].members[0].rec);
      g.differs = describeDiffers(differs) || 'sampler cadence';
    }
    g.label = newest ? 'tokens per Wh' : `tokens per Wh · earlier instrument (${g.differs})`;
    // Recompute the key from the member rather than splitting it back out of
    // `g.instrument`: costInstrumentKey is `${instrumentKey}|gpu_rail|${period}`
    // and instrumentKey is a JSON array string, so a `|` inside any axis value
    // would make that parse silently wrong.
    g.spread = rungSpread(c, instrumentKey(g.members[0].rec), records);
    g.verdict = spreadVerdict(g.members.map((u) => u.e.tokPerWh), g.spread);
    metrics.push({ key: g.key, label: g.label, dashed: !newest, envelope: g.spread });
    for (const u of g.members) derived.push({ ...u.rec, metrics: { ...u.rec.metrics, [g.key]: u.e.tokPerWh } });
  });

  return {
    c,
    key: base,
    title: `tokens per Wh · C=${c}`,
    unit: 'tok/Wh',
    metrics,
    records: derived,
    generations: gens,
    excluded,
    runs: usable.length
  };
}

/**
 * The vLLM figure for the same rung: a DATED SNAPSHOT, never a series. It is
 * printed beside chart B with its date and "not re-run" — a one-shot joined to
 * an Atlas trend would read as a vLLM that moved when it did not.
 */
export function baselineSnapshots(c, baselines) {
  return baselines
    .map((b) => ({ b, point: b.points.find((p) => p.c === c) }))
    .filter((x) => x.point && x.point.energy.state === 'measured')
    .map(({ b, point }) => ({
      label: b.label,
      date: String(b.series?.rungs?.find((r) => r.c === c)?.measured_utc ?? '').slice(0, 10),
      tokPerWh: point.energy.tokPerWh,
      trusted: point.energy.trusted
    }));
}

// ---- the empty state --------------------------------------------------------

/**
 * Why a subject has no cost chart, in the order a reader needs it: what it has,
 * what the other engine has, what fills the gap. Every line is DERIVED from the
 * data — none of it is typed, and none of it is a zero.
 */
/** The first few differing axes, so one line stays readable without hiding the rest. */
const summariseDiffers = (differs, show = 3) =>
  differs.length <= show
    ? describeDiffers(differs)
    : `${describeDiffers(differs.slice(0, show))}, +${differs.length - show} more axes`;

export function emptyStateOf(subject, records, ladders) {
  const ladder = ladderFor(subject, ladders);
  const withEnergy = records.filter((r) => rungsOf(r).some((c) => energyOf(r, c).state !== 'absent'));
  const atlas =
    records.length === 0
      ? `no ${subject.gate} record for ${subject.checkpoint} yet — 0 records`
      : withEnergy.length === 0
        ? `${records.length} ${subject.gate} records, none carries ${cellKey('{C}', KEY.energyJ)}`
        : `${withEnergy.length} of ${records.length} records carry GPU-rail joules`;

  // Whether a published vLLM series could even be drawn against this gate is
  // half the answer: a one-shot on another instrument is not "a comparison
  // waiting for joules", it is a different measurement. So the line says both.
  const series = ladder ? baselineSeriesOf(ladder) : [];
  const against = records[records.length - 1] ?? null;
  const vllm =
    series.length === 0
      ? 'no vLLM series is published for this subject'
      : series
          .map((b) => {
            const measured = pointsOfSeries(b).some((p) => p.energy.state === 'measured');
            const cmp = against
              ? comparable(against, { checkpoint: ladder.workload.checkpoint, instrument: b.instrument })
              : null;
            const where =
              cmp === null
                ? 'no Atlas run to fingerprint it against'
                : cmp.ok
                  ? 'the gate instrument'
                  : `another instrument (${summariseDiffers(cmp.differs)})`;
            return `${b.label} · ISL ${b.instrument?.isl} / OSL ${b.instrument?.osl} · ${where} · ${
              measured ? 'carries joules' : 'no energy recorded'
            }`;
          })
          .join('; ');

  return {
    title: 'Cost · not yet measured',
    atlas,
    vllm,
    fills:
      `one ${subject.gate} run on main with the energy sampler → the Atlas cost curve; ` +
      `one energy-instrumented vLLM run on the same instrument, filed under ${subject.baselines_dir}/ → the comparison`
  };
}

// ---- the disclosures, rendered under chart A and in the tab footer ----------

export const DISCLOSURES = Object.freeze([
  {
    head: 'GPU rail only — a lower bound on cost.',
    body:
      'GB10 exposes one power reading, nvidia-smi power.draw.average on the GPU rail. Module power — ' +
      'Grace CPU cores and LPDDR5X memory — is not readable: power.limit, Module Power Readings and GPU ' +
      'Memory Power all report N/A. On a unified-memory part whose decode is bandwidth-bound, a large ' +
      'share of the real energy is outside this number, for BOTH engines. Every figure here is therefore ' +
      'a floor on the real bill, and the ratio between engines assumes the unread share is similar — an ' +
      'assumption this hardware cannot test.'
  },
  {
    head: 'Comparable only same-instrument, same-rail, same-cadence.',
    body:
      'A vLLM line is drawn only when its run matched the Atlas run on every workload axis (ISL, OSL, ' +
      'prompt mode, context, batch cap, KV dtype) and was measured on the same rail at the same sampler ' +
      'cadence; otherwise it is named and not drawn. A one-shot is a dated snapshot, never a trend.'
  },
  {
    head: 'Absent is not zero; under-sampled is not averaged.',
    body:
      'A missing key renders as "not measured". A window with fewer than ' +
      `${MIN_POWER_SAMPLES} readings, or one the sampler covered less than ` +
      `${Math.round(MIN_SAMPLE_COVERAGE * 100)}% of, is drawn hollow and left out of every tile, ` +
      'verdict and trend line — visible, but never counted.'
  },
  {
    head: 'Facility overhead is yours to set, and it starts at none.',
    body:
      `PUE — total facility energy over IT energy — defaults to ${DEFAULT_PUE}×, the identity: nothing is ` +
      'applied and the figures are the GPU rail as measured. Set a low and a high (a datacentre is ' +
      `typically ${PUE_TYPICAL.low}–${PUE_TYPICAL.high}×) and the chart draws the band between them. It ` +
      'multiplies both engines equally, so it moves every absolute figure and changes no ratio, no ' +
      'verdict and no winner — which is why it can be offered at all. It is still stacked on a reading ' +
      'that is already a lower bound.'
  },
  {
    head: '$/kWh is your input.',
    body:
      'The only assumption on this page is the electricity price you type; the default ' +
      `${DEFAULT_USD_PER_KWH} $/kWh is a round retail-commercial placeholder, not a measurement.`
  }
]);
