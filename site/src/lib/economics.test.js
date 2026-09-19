// SPDX-License-Identifier: AGPL-3.0-only
import { expect, test } from 'bun:test';
import { fleetModel, apiModel, energyModel, energyInputsFrom, efficiency, efficiencyLadder, axisFor, paybackLabel, FLEET_DEFAULTS, API_DEFAULTS, ENERGY_DEFAULTS, JOULES_PER_KWH, powerCostPerYear } from './economics.js';
import ladder from './ladder.generated.json';

test('power cost is watts times hours times PUE at the tariff', () => {
  // 1 kW at PUE 1.0 for a year at $0.10 is 8760 kWh = $876.
  expect(powerCostPerYear({ watts: 1000, pue: 1, usdPerKwh: 0.1 })).toBeCloseTo(876, 5);
});

test('an uplift of 1.0 frees nothing and never pays back', () => {
  const r = fleetModel({ uplift: 1 });
  expect(r.freedGpus).toBe(0);
  expect(r.grossSavings).toBe(0);
  expect(r.paybackMonths).toBeNull();
});

test('the fleet defaults pay back inside a year with a positive three year net', () => {
  const r = fleetModel();
  expect(r.paybackMonths).toBeGreaterThan(0);
  expect(r.paybackMonths).toBeLessThan(12);
  expect(r.threeYearNet).toBeGreaterThan(0);
  // 256 GPUs at 1.2x frees 256 * (1 - 1/1.2) = 42.67 GPUs.
  expect(r.freedGpus).toBeCloseTo(42.7, 1);
});

test('the fleet defaults are the ones the copy describes', () => {
  expect(FLEET_DEFAULTS.gpus).toBe(256);
  expect(FLEET_DEFAULTS.uplift).toBe(1.2);
});

test('the API defaults clear the 70% claim on the front page', () => {
  const r = apiModel();
  expect(r.savingsPct).toBeGreaterThanOrEqual(70);
  expect(r.boxes).toBeGreaterThanOrEqual(1);
  expect(r.paybackMonths).toBeGreaterThan(0);
});

test('a cheaper API erodes the savings honestly', () => {
  const cheap = apiModel({ usdPerMillionTokens: 0.2 });
  const dear = apiModel({ usdPerMillionTokens: 2 });
  expect(cheap.savingsPct).toBeLessThan(dear.savingsPct);
});

test('a bill of zero produces no division by zero', () => {
  const r = apiModel({ monthlySpend: 0 });
  expect(r.savingsPct).toBe(0);
  expect(Number.isFinite(r.newMonthly)).toBe(true);
});

test('measured throughput is the API default the render replaces', () => {
  expect(API_DEFAULTS.boxTokensPerSecond).toBeGreaterThan(0);
});

// --- tokens per watt ----------------------------------------------------------

test('tokens per watt is tokens per joule: throughput over power, and the seconds cancel', () => {
  // 500 tok/s at 250 W is 2 tokens for every joule, so a token costs half a joule.
  const e = efficiency({ tokensPerSecond: 500, watts: 250 });
  expect(e.tokensPerJoule).toBe(2);
  expect(e.joulesPerToken).toBe(0.5);
  // A million tokens at 0.5 J each is 500 kJ, and a kWh is 3.6 MJ.
  expect(e.kwhPerMillion).toBeCloseTo(500_000 / JOULES_PER_KWH, 10);
  expect(JOULES_PER_KWH).toBe(1000 * 3600);
});

test('no draw or no throughput is no efficiency, not a division by zero', () => {
  for (const bad of [{ tokensPerSecond: 0, watts: 240 }, { tokensPerSecond: 478, watts: 0 }]) {
    const e = efficiency(bad);
    expect(e.tokensPerJoule).toBe(0);
    expect(Number.isFinite(e.kwhPerMillion)).toBe(true);
  }
  const r = energyModel({ baselineTokensPerSecond: 0 });
  expect(r.ratio).toBeNull();
  expect(r.energyCutPct).toBe(0);
});

test('at the same draw the efficiency ratio is the throughput ratio, and the energy cut follows from it', () => {
  const r = energyModel({ tokensPerSecond: 400, baselineTokensPerSecond: 300, watts: 200, baselineWatts: 200 });
  expect(r.ratio).toBeCloseTo(400 / 300, 3);
  // 1 - 300/400: a quarter less energy for the same token.
  expect(r.energyCutPct).toBe(25);
});

test('an engine that draws more for its extra tokens gives the advantage back', () => {
  // A third more tokens for a third more watts is no gain at all.
  const r = energyModel({ tokensPerSecond: 400, baselineTokensPerSecond: 300, watts: 320, baselineWatts: 240 });
  expect(r.ratio).toBe(1);
  expect(r.energyCutPct).toBe(0);
  expect(r.kwhSavedPerMonth).toBe(0);
  const worse = energyModel({ tokensPerSecond: 400, baselineTokensPerSecond: 300, watts: 400, baselineWatts: 240 });
  expect(worse.usdSavedPerYear).toBeLessThan(0);
});

test('a month of tokens becomes kilowatt hours at the wall, then dollars', () => {
  // 1 J per token: a billion tokens is 1e9 J = 277.78 kWh of IT load. PUE 1.5
  // makes it 416.67 at the meter. The baseline at 2 J per token is twice that.
  const r = energyModel({ tokensPerSecond: 100, watts: 100, baselineTokensPerSecond: 50, baselineWatts: 100, pue: 1.5, usdPerKwh: 0.1, millionTokensPerMonth: 1000 });
  expect(r.kwhPerMonth).toBe(417);
  expect(r.baselineKwhPerMonth).toBe(833);
  expect(r.kwhSavedPerMonth).toBe(417);
  expect(r.usdSavedPerYear).toBe(500); // 416.67 kWh x $0.10 x 12
  expect(r.usdPerMillion).toBeCloseTo(0.0417, 4);
});

test('PUE changes the bill and never the efficiency', () => {
  const lean = energyModel({ pue: 1 });
  const heavy = energyModel({ pue: 2 });
  expect(heavy.tokensPerJoule).toBe(lean.tokensPerJoule);
  expect(heavy.kwhPerMillion).toBe(lean.kwhPerMillion);
  expect(heavy.kwhPerMonth).toBeCloseTo(lean.kwhPerMonth * 2, -1);
});

test('the energy defaults assume the same draw for both engines, at the box ceiling', () => {
  expect(ENERGY_DEFAULTS.watts).toBe(ENERGY_DEFAULTS.baselineWatts);
  expect(ENERGY_DEFAULTS.watts).toBe(API_DEFAULTS.wattsPerBox);
});

test('the graph has a point for every published rung, from measured throughput', () => {
  const { points, measured } = efficiencyLadder(ladder.rows, { watts: 240, baselineWatts: 240 });
  expect(points.map((p) => p.c)).toEqual([...ladder.concurrencies].sort((a, b) => a - b));
  const top = ladder.rows.find((r) => r.c === points.at(-1).c);
  expect(points.at(-1).avarok).toBeCloseTo(top.atlas / 240, 3);
  expect(points.at(-1).baseline).toBeCloseTo(top.baselines.find((b) => b.id === top.best_baseline_id).tok_s / 240, 3);
  // The ladder publishes no power today, so nothing on the graph may claim it.
  expect(measured).toBe(false);
  expect(points.some((p) => p.measured)).toBe(false);
});

test('a rung that records its draw uses it, and only a fully recorded ladder is called measured', () => {
  const rows = [
    { c: 1, atlas: 20, atlas_watts: 40, best_baseline_id: 'v', baselines: [{ id: 'v', tok_s: 10, watts: 50 }] },
    { c: 8, atlas: 100, best_baseline_id: 'v', baselines: [{ id: 'v', tok_s: 80 }] }
  ];
  const mixed = efficiencyLadder(rows, { watts: 200, baselineWatts: 200 });
  expect(mixed.points[0]).toEqual({ c: 1, avarok: 0.5, baseline: 0.2, measured: true });
  expect(mixed.points[1]).toEqual({ c: 8, avarok: 0.5, baseline: 0.4, measured: false });
  expect(mixed.measured).toBe(false);
  expect(efficiencyLadder([rows[0]]).measured).toBe(true);
  expect(efficiencyLadder([]).measured).toBe(false);
});

test('the scenario starts from the top rung, against the matched baseline and never the fastest', () => {
  const start = energyInputsFrom(ladder.rows);
  const top = [...ladder.rows].sort((a, b) => a.c - b.c).at(-1);
  expect(start.tokensPerSecond).toBe(top.atlas);
  expect(start.baselineTokensPerSecond).toBe(top.baselines.find((b) => b.id === top.best_baseline_id).tok_s);
  expect(start.watts).toBe(ENERGY_DEFAULTS.watts); // no recorded draw yet
  // The big number and the last point on the graph are one measurement.
  const { points } = efficiencyLadder(ladder.rows, { watts: start.watts, baselineWatts: start.baselineWatts });
  expect(points.at(-1).avarok).toBeCloseTo(energyModel(start).tokensPerJoule, 3);
  // A recorded draw on the top rung becomes the starting draw.
  const recorded = energyInputsFrom([{ c: 4, atlas: 90, atlas_watts: 60, best_baseline_id: 'v', baselines: [{ id: 'fast', tok_s: 99 }, { id: 'v', tok_s: 70, watts: 55 }] }]);
  expect(recorded).toMatchObject({ tokensPerSecond: 90, baselineTokensPerSecond: 70, watts: 60, baselineWatts: 55 });
  expect(energyInputsFrom([])).toEqual({ ...ENERGY_DEFAULTS });
});

test('the graph axis puts every gridline on the round number its label prints', () => {
  expect(axisFor(1.992)).toEqual({ top: 2, ticks: [0, 0.5, 1, 1.5, 2], digits: 1 });
  expect(axisFor(4.78)).toEqual({ top: 6, ticks: [0, 2, 4, 6], digits: 0 });
  expect(axisFor(0.083)).toEqual({ top: 0.1, ticks: [0, 0.025, 0.05, 0.075, 0.1], digits: 3 });
  expect(axisFor(2).top).toBe(2); // a maximum on a tick needs no extra band
  for (const max of [0.0137, 0.4, 3.3, 7.9, 12, 96, 250]) {
    const { top, ticks } = axisFor(max);
    expect(top).toBeGreaterThanOrEqual(max);
    expect(ticks.length).toBeGreaterThanOrEqual(3);
    expect(ticks.length).toBeLessThanOrEqual(7);
  }
  expect(axisFor(0).top).toBe(1);
  expect(axisFor(NaN).top).toBe(1);
});

test('payback labels read like a person wrote them', () => {
  expect(paybackLabel(null)).toBe('no payback');
  expect(paybackLabel(0.5)).toBe('2 weeks');
  expect(paybackLabel(1.2)).toBe('1 month');
  expect(paybackLabel(4.4)).toBe('4 months');
});
