// SPDX-License-Identifier: AGPL-3.0-only
//
// The payback model behind /pricing#payback and the savings tile on the front
// page. Pure functions over plain inputs so the same arithmetic is unit tested,
// rendered in the browser, and quotable in a sales conversation.
//
// Two scenarios, because the two buyers are different:
//
//   fleet  — "get more out of the GPUs you own": the uplift frees capacity,
//            and freed capacity is deferred capex or rental plus power. The
//            license is the cost. This is the enterprise conversation.
//   api    — "stop renting tokens": replace a metered API bill with boxes you
//            own running the engine at its measured throughput. This is the
//            workstation, SMB and edge conversation, and where the 70% claim
//            on the front page comes from.
//
// Every input is labeled with its evidence class in the UI: MEASURED comes
// from ladder.generated.json, PROPOSED is a list price the team can change,
// USER is whatever the visitor types. Nothing here rounds a claim up.

export const HOURS_PER_YEAR = 8760;
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
