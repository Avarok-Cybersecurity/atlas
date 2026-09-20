// SPDX-License-Identifier: AGPL-3.0-only
//
// The payback model behind /pricing#payback and the savings tile on the front
// page. Pure functions over plain inputs so the same arithmetic is unit tested,
// rendered in the browser, and quotable in a sales conversation.
//
// Three scenarios, because the buyers are different:
//
//   fleet  — "get more out of the GPUs you own": the uplift frees capacity,
//            and freed capacity is deferred capex or rental plus power. The
//            license is the cost. This is the enterprise conversation.
//   api    — "stop renting tokens": replace a metered API bill with boxes you
//            own running the engine at its measured throughput. This is the
//            workstation, SMB and edge conversation, and where the 70% claim
//            on the front page comes from.
//   energy — "get more tokens per watt": the same comparison in a unit that
//            owes nothing to a price list. This is the conversation with
//            whoever owns the power budget. The units are spelled out above
//            `efficiency()` because a benchmark has to produce them.
//
// Every input is labeled with its evidence class in the UI: MEASURED comes
// from ladder.generated.json, PROPOSED is a list price the team can change,
// USER is whatever the visitor types. Nothing here rounds a claim up.

export const HOURS_PER_YEAR = 8760;
export const JOULES_PER_KWH = 3_600_000;
const SECONDS_PER_YEAR = 31_536_000;

/** Defaults for the enterprise fleet scenario. */
export const FLEET_DEFAULTS = Object.freeze({
  gpus: 256,
  gpuCostPerYear: 40_000, // amortized purchase or rental, per GPU, USER
  utilization: 0.6, // USER
  uplift: 1.2, // conservative, below the measured GB10 ratio at C=128
  licensePerGpuYear: 2_400, // PROPOSED, realized at fleet scale
  wattsPerGpu: 700, // H100 class, USER
  pue: 1.3, // USER
  usdPerKwh: 0.12, // USER
  replacedSoftwarePerGpuYear: 0 // e.g. 4500 for NVIDIA AI Enterprise, USER
});

/** Defaults for the API replacement scenario. */
export const API_DEFAULTS = Object.freeze({
  monthlySpend: 20_000, // USER
  usdPerMillionTokens: 0.9, // blended in plus out for a 27B class open model on a hosted API, USER
  boxTokensPerSecond: 478, // MEASURED, replaced at render time by the ladder top rung
  boxCapex: 4_000, // DGX Spark class box, USER
  amortMonths: 36, // USER
  utilization: 0.6, // USER
  wattsPerBox: 240, // USER
  pue: 1.2, // USER
  usdPerKwh: 0.12, // USER
  licensePerBoxMonth: 50 // PROPOSED workstation license
});

/** Defaults for the tokens per watt scenario. */
export const ENERGY_DEFAULTS = Object.freeze({
  tokensPerSecond: 478, // MEASURED, replaced at render time by the ladder top rung
  baselineTokensPerSecond: 359, // MEASURED, the matched baseline on the same rung
  // Draw under load. 240 W is the peak total system power NVIDIA states for a
  // DGX Spark, so it is a ceiling and the efficiency it gives is a floor. USER
  // until the ladder records the draw, then MEASURED (see efficiencyLadder).
  watts: 240,
  baselineWatts: 240, // USER. Equal draw is an assumption, and the visitor can break it
  pue: 1.2, // USER
  usdPerKwh: 0.12, // USER
  millionTokensPerMonth: 1000 // USER
});

const round = (n, d = 1) => Math.round(n * 10 ** d) / 10 ** d;

/** Annual power cost for one device at a steady draw. */
export function powerCostPerYear({ watts, pue, usdPerKwh }) {
  return ((watts * pue * HOURS_PER_YEAR) / 1000) * usdPerKwh;
}

/**
 * The enterprise fleet scenario.
 *
 * With uplift u, the same work needs gpus / u GPUs, so the capacity freed is
 * gpus * (1 - 1/u). Its value is what those GPUs cost to keep, plus what they
 * burn, plus any per GPU software they were carrying. The license is the
 * price of the uplift. Payback is license over monthly savings.
 */
export function fleetModel(input = {}) {
  const i = { ...FLEET_DEFAULTS, ...input };
  const freedGpus = i.uplift > 0 ? i.gpus * (1 - 1 / i.uplift) : 0;
  const capacityValue = freedGpus * i.gpuCostPerYear;
  const power = powerCostPerYear({ watts: i.wattsPerGpu, pue: i.pue, usdPerKwh: i.usdPerKwh });
  const powerValue = freedGpus * power;
  const replaced = i.gpus * i.replacedSoftwarePerGpuYear;
  const grossSavings = capacityValue + powerValue + replaced;
  const license = i.gpus * i.licensePerGpuYear;
  const net = grossSavings - license;
  const paybackMonths = grossSavings > 0 ? (license / grossSavings) * 12 : Infinity;
  const spend = i.gpus * i.gpuCostPerYear;
  return {
    freedGpus: round(freedGpus, 1),
    capacityValue: Math.round(capacityValue),
    powerValue: Math.round(powerValue),
    replaced: Math.round(replaced),
    grossSavings: Math.round(grossSavings),
    license: Math.round(license),
    net: Math.round(net),
    threeYearNet: Math.round(net * 3),
    paybackMonths: Number.isFinite(paybackMonths) ? round(paybackMonths, 1) : null,
    savingsPct: spend > 0 ? round((net / spend) * 100, 1) : 0
  };
}

/**
 * The API replacement scenario.
 *
 * Tokens per month come from the bill. Boxes needed come from measured
 * throughput at the stated utilization. The new monthly cost is amortized
 * capex plus power plus license for that many boxes. Savings percent is
 * against the old bill, which is where "70% or less" is checked.
 */
export function apiModel(input = {}) {
  const i = { ...API_DEFAULTS, ...input };
  const tokensPerMonth = i.usdPerMillionTokens > 0 ? (i.monthlySpend / i.usdPerMillionTokens) * 1e6 : 0;
  const boxTokensPerMonth = i.boxTokensPerSecond * i.utilization * (SECONDS_PER_YEAR / 12);
  const boxesExact = boxTokensPerMonth > 0 ? tokensPerMonth / boxTokensPerMonth : 0;
  const boxes = Math.max(1, Math.ceil(boxesExact));
  const capexPerMonth = i.amortMonths > 0 ? (boxes * i.boxCapex) / i.amortMonths : 0;
  const powerPerMonth = (boxes * powerCostPerYear({ watts: i.wattsPerBox, pue: i.pue, usdPerKwh: i.usdPerKwh })) / 12;
  const licensePerMonth = boxes * i.licensePerBoxMonth;
  const newMonthly = capexPerMonth + powerPerMonth + licensePerMonth;
  const monthlySavings = i.monthlySpend - newMonthly;
  const costPerMillion = tokensPerMonth > 0 ? newMonthly / (tokensPerMonth / 1e6) : 0;
  const paybackMonths = monthlySavings > 0 ? (boxes * i.boxCapex) / monthlySavings : null;
  return {
    tokensPerMonth: Math.round(tokensPerMonth),
    boxes,
    capexPerMonth: Math.round(capexPerMonth),
    powerPerMonth: Math.round(powerPerMonth),
    licensePerMonth: Math.round(licensePerMonth),
    newMonthly: Math.round(newMonthly),
    monthlySavings: Math.round(monthlySavings),
    savingsPct: i.monthlySpend > 0 ? round((monthlySavings / i.monthlySpend) * 100, 1) : 0,
    costPerMillion: round(costPerMillion, 3),
    paybackMonths: paybackMonths === null ? null : round(paybackMonths, 1)
  };
}

// The units, exactly, because a benchmark has to produce them:
//
//   throughput   tok/s      completion tokens over the timed window
//   power        W = J/s    mean draw over the same window, at a named boundary
//                           (the GPU rail, or the wall)
//   efficiency   tok/J      throughput over power. "Tokens per watt" is the trade
//                           name: tokens per second, per watt. The seconds cancel.
//   energy       J/tok      one over efficiency
//                kWh/Mtok   J/tok divided by 3.6  (1 kWh = 3.6e6 J, 1 Mtok = 1e6 tok)
//
// PUE is not part of efficiency. Efficiency is a property of the box and what
// runs on it. PUE is applied where joules become a facility's bill.

/** One engine on one box: tokens per joule, and the energy a token costs. */
export function efficiency({ tokensPerSecond, watts }) {
  const ok = tokensPerSecond > 0 && watts > 0;
  const joulesPerToken = ok ? watts / tokensPerSecond : 0;
  return {
    tokensPerJoule: ok ? tokensPerSecond / watts : 0,
    joulesPerToken,
    kwhPerMillion: (joulesPerToken * 1e6) / JOULES_PER_KWH
  };
}

/**
 * The tokens per watt scenario.
 *
 * Two engines on the same box, each with a measured throughput and a draw.
 * Efficiency is one over the other. A month of work, in millions of tokens,
 * turns the difference into kilowatt hours, and PUE and the tariff turn those
 * into dollars. There is no license line: this tab prices energy and nothing
 * else, so it cannot be read as a payback claim.
 *
 * It counts the energy of the tokens served, which is what tokens per joule
 * means. It does not count idle draw. A box that finishes sooner and is left on
 * still draws its idle power for the hours it saved, so on a box that is never
 * switched off or given other work the real saving is smaller by the ratio of
 * idle draw to draw under load. The page says so.
 */
export function energyModel(input = {}) {
  const i = { ...ENERGY_DEFAULTS, ...input };
  const ours = efficiency({ tokensPerSecond: i.tokensPerSecond, watts: i.watts });
  const theirs = efficiency({ tokensPerSecond: i.baselineTokensPerSecond, watts: i.baselineWatts });
  const monthKwh = (e) => e.kwhPerMillion * i.millionTokensPerMonth * i.pue;
  const usdPerMillion = (e) => e.kwhPerMillion * i.pue * i.usdPerKwh;
  const kwhSaved = monthKwh(theirs) - monthKwh(ours);
  return {
    tokensPerJoule: round(ours.tokensPerJoule, 3),
    baselineTokensPerJoule: round(theirs.tokensPerJoule, 3),
    ratio: theirs.tokensPerJoule > 0 ? round(ours.tokensPerJoule / theirs.tokensPerJoule, 3) : null,
    joulesPerToken: round(ours.joulesPerToken, 3),
    baselineJoulesPerToken: round(theirs.joulesPerToken, 3),
    kwhPerMillion: round(ours.kwhPerMillion, 4),
    baselineKwhPerMillion: round(theirs.kwhPerMillion, 4),
    usdPerMillion: round(usdPerMillion(ours), 4),
    baselineUsdPerMillion: round(usdPerMillion(theirs), 4),
    energyCutPct: theirs.joulesPerToken > 0 ? round((1 - ours.joulesPerToken / theirs.joulesPerToken) * 100, 1) : 0,
    kwhPerMonth: Math.round(monthKwh(ours)),
    baselineKwhPerMonth: Math.round(monthKwh(theirs)),
    kwhSavedPerMonth: Math.round(kwhSaved),
    usdSavedPerYear: Math.round(kwhSaved * i.usdPerKwh * 12)
  };
}

// The baseline a rung is published against: the matched one, never the fastest.
const matchedBaseline = (row) => row.baselines?.find((b) => b.id === row.best_baseline_id) ?? row.baselines?.[0];

/**
 * Tokens per joule on every rung of a published ladder, for the graph.
 *
 * Throughput is measured on every rung. Power is not published yet, so a rung
 * falls back to the draw the visitor typed, held flat across the rungs. A box
 * draws less when it is doing less, so a flat draw understates the low rungs.
 *
 * When the ladder records the mean draw over a rung's timed window, as
 * `atlas_watts` on the row and `watts` on a baseline, that rung uses it, and
 * `measured` says whether every rung did. That is the only change the tab needs
 * to go from USER to MEASURED.
 */
export function efficiencyLadder(rows = [], { watts = ENERGY_DEFAULTS.watts, baselineWatts = ENERGY_DEFAULTS.baselineWatts } = {}) {
  const perJoule = (tokensPerSecond, w) => round(efficiency({ tokensPerSecond, watts: w }).tokensPerJoule, 4);
  const points = [...rows]
    .sort((a, b) => a.c - b.c)
    .map((r) => {
      const base = matchedBaseline(r) ?? { tok_s: 0 };
      const measured = r.atlas_watts > 0 && base.watts > 0;
      return {
        c: r.c,
        avarok: perJoule(r.atlas, measured ? r.atlas_watts : watts),
        baseline: perJoule(base.tok_s, measured ? base.watts : baselineWatts),
        measured
      };
    });
  return { points, measured: points.length > 0 && points.every((p) => p.measured) };
}

/**
 * Where the scenario starts, from a published ladder: the top rung's throughput
 * for both engines, and that rung's recorded draw when it has one. So the big
 * number and the last point on the graph are the same measurement.
 */
export function energyInputsFrom(rows = []) {
  const top = [...rows].sort((a, b) => a.c - b.c).at(-1);
  const base = top && matchedBaseline(top);
  if (!top || !base) return { ...ENERGY_DEFAULTS };
  const measured = top.atlas_watts > 0 && base.watts > 0;
  return { ...ENERGY_DEFAULTS, tokensPerSecond: top.atlas, baselineTokensPerSecond: base.tok_s, ...(measured ? { watts: top.atlas_watts, baselineWatts: base.watts } : {}) };
}

/**
 * A y axis for the graph: evenly spaced ticks on round numbers, up to the first
 * one at or above `max`, so every gridline sits on the value its label prints.
 */
export function axisFor(max, target = 4) {
  if (!(max > 0) || !Number.isFinite(max)) return { top: 1, ticks: [0, 0.25, 0.5, 0.75, 1], digits: 2 };
  const raw = max / target;
  const pow = 10 ** Math.floor(Math.log10(raw));
  const n = raw / pow;
  const step = round((n <= 1 ? 1 : n <= 2 ? 2 : n <= 2.5 ? 2.5 : n <= 5 ? 5 : 10) * pow, 9);
  const count = Math.ceil(max / step - 1e-9);
  const ticks = Array.from({ length: count + 1 }, (_, k) => round(k * step, 9));
  const digits = Math.min(6, (String(step).split('.')[1] ?? '').length);
  return { top: ticks.at(-1), ticks, digits };
}

/** Whole months, human. "4.2" becomes "4 months", "0.6" becomes "3 weeks". */
export function paybackLabel(months) {
  if (months === null || !Number.isFinite(months)) return 'no payback';
  if (months < 1) return `${Math.max(1, Math.round(months * 4.33))} weeks`;
  const m = Math.round(months);
  return `${m} ${m === 1 ? 'month' : 'months'}`;
}

export const usd = (n, digits = 0) =>
  new Intl.NumberFormat('en-US', { style: 'currency', currency: 'USD', maximumFractionDigits: digits }).format(n);
export const num = (n, digits = 0) => new Intl.NumberFormat('en-US', { maximumFractionDigits: digits }).format(n);
