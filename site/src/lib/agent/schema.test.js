// SPDX-License-Identifier: AGPL-3.0-only

// The schema is sent by the agent at handshake, which means this page can be
// handed one it was not built for — an older agent, a newer one, a field
// renamed between them. The module's stated intent is that an unrecognised
// bound kind degrades to read-only rather than hiding the setting; these cover
// the case it did not consider, which is a bound that is not there at all.

import { test, expect } from 'bun:test';
import { checkValue, isEditable } from './schema.js';

test('a spec with no bound is read-only, not a crash', () => {
  // Reachable from SettingsEditor and LaunchModal, both of which render every
  // spec the agent sends. A throw here blanks the whole settings panel.
  expect(() => checkValue({ key: 'x' }, 1)).not.toThrow();
  expect(checkValue({ key: 'x' }, 1)).toBeNull();
  expect(isEditable({ key: 'x' })).toBe(false);
});

test('an unknown bound kind stays read-only, as the module intends', () => {
  const spec = { key: 'x', bound: { kind: 'colour_wheel' } };
  expect(checkValue(spec, 'red')).toBeNull();
  expect(isEditable(spec)).toBe(false);
});

test('an enum with no variants says nothing rather than throwing', () => {
  expect(() => checkValue({ key: 'x', bound: { kind: 'enum' } }, 'a')).not.toThrow();
  expect(checkValue({ key: 'x', bound: { kind: 'enum' } }, 'a')).toBeNull();
});

test('null and undefined specs are survivable', () => {
  expect(() => checkValue(null, 1)).not.toThrow();
  expect(() => checkValue(undefined, 1)).not.toThrow();
  expect(isEditable(null)).toBe(false);
});

test('real bounds still check as before', () => {
  const int = { key: 'n', bound: { kind: 'int', min: 1, max: 8 } };
  expect(checkValue(int, 4)).toBeNull();
  expect(checkValue(int, 9)).toMatch(/between 1 and 8/);
  expect(checkValue(int, 1.5)).toMatch(/whole number/);
  expect(isEditable(int)).toBe(true);

  const en = { key: 'p', bound: { kind: 'enum', variants: ['fifo', 'slai'] } };
  expect(checkValue(en, 'fifo')).toBeNull();
  expect(checkValue(en, 'fcfs')).toMatch(/must be one of: fifo, slai/);
});
