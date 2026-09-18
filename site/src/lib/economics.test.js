// SPDX-License-Identifier: AGPL-3.0-only
import { expect, test } from 'bun:test';
import { fleetModel, apiModel, paybackLabel, FLEET_DEFAULTS, API_DEFAULTS, powerCostPerYear } from './economics.js';

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

test('payback labels read like a person wrote them', () => {
  expect(paybackLabel(null)).toBe('no payback');
  expect(paybackLabel(0.5)).toBe('2 weeks');
  expect(paybackLabel(1.2)).toBe('1 month');
  expect(paybackLabel(4.4)).toBe('4 months');
});
