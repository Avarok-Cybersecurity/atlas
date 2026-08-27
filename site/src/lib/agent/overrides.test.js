// SPDX-License-Identifier: AGPL-3.0-only

import { describe, expect, test } from 'bun:test';
import * as O from './overrides.js';

const DEFAULTS = { port: 8888, gpu_memory_utilization: 0.9, kv_cache_dtype: 'bf16', enable_prefix_caching: true };
const FLOAT = { key: 'gpu_memory_utilization', bound: { kind: 'float', min: 0.1, max: 0.95 } };
const INT = { key: 'port', bound: { kind: 'int', min: 1024, max: 49151 } };
const ENUM = { key: 'kv_cache_dtype', bound: { kind: 'enum', variants: ['bf16', 'fp8', 'nvfp4'] } };
const TOGGLE = { key: 'enable_prefix_caching', bound: { kind: 'toggle' } };
const AUTO = { key: 'mtp_gate', bound: { kind: 'int_or_auto', min: 0, max: 64 } };

describe('what is in force', () => {
  test('an override wins, otherwise the recipe speaks', () => {
    expect(O.effective('port', DEFAULTS, {})).toBe(8888);
    expect(O.effective('port', DEFAULTS, { port: 9000 })).toBe(9000);
    expect(O.effective('nothing', DEFAULTS, {})).toBeUndefined();
  });

  // A setting the page cannot render is never overridden, so the recipe's own
  // value applies on the agent. That is what makes version skew survivable.
  test('a setting with no override and no default is simply absent', () => {
    expect(O.toWire({}, DEFAULTS)).toEqual({});
  });
});

describe('only real differences travel', () => {
  test('setting a value back to the recipe default drops the override', () => {
    let o = O.set({}, 'gpu_memory_utilization', 0.85, DEFAULTS);
    expect(O.changedCount(o, DEFAULTS)).toBe(1);
    o = O.set(o, 'gpu_memory_utilization', 0.9, DEFAULTS);
    expect(o).toEqual({});
    expect(O.changedCount(o, DEFAULTS)).toBe(0);
  });

  test('0.90 and 0.9 are the same number, not a change', () => {
    const o = O.set({}, 'gpu_memory_utilization', 0.90, DEFAULTS);
    expect(o).toEqual({});
  });

  test('the wire carries the differences and nothing else', () => {
    const o = O.set(O.set({}, 'port', 9000, DEFAULTS), 'kv_cache_dtype', 'bf16', DEFAULTS);
    expect(O.toWire(o, DEFAULTS)).toEqual({ port: 9000 });
  });

  test('clearing returns a setting to the recipe', () => {
    const o = O.clear(O.set({}, 'port', 9000, DEFAULTS), 'port');
    expect(O.effective('port', DEFAULTS, o)).toBe(8888);
    expect(O.changedCount(o, DEFAULTS)).toBe(0);
  });

  test('a setting the recipe never mentions counts as changed once set', () => {
    const o = O.set({}, 'max_model_len', 4096, DEFAULTS);
    expect(O.isChanged('max_model_len', DEFAULTS, o)).toBe(true);
    expect(O.toWire(o, DEFAULTS)).toEqual({ max_model_len: 4096 });
  });
});

describe('parsing what an input produced', () => {
  test('numbers keep their type', () => {
    expect(O.parse(INT, '9000')).toEqual({ value: 9000 });
    expect(O.parse(FLOAT, '0.85')).toEqual({ value: 0.85 });
  });

  // NaN reaching the agent would come back as a complaint about bounds, which
  // tells the operator nothing about the empty box they left behind.
  test('an empty box is an error, not a silent zero', () => {
    expect(O.parse(INT, '').error).toContain('enter a value');
    expect(O.parse(FLOAT, '   ').error).toContain('enter a value');
  });

  test('a non-number is refused before it reaches the agent', () => {
    expect(O.parse(FLOAT, 'abc').error).toBe('must be a number');
    expect(O.parse(INT, '80.5').error).toBe('must be a whole number');
  });

  test('enums and toggles keep their own shapes', () => {
    expect(O.parse(ENUM, 'fp8')).toEqual({ value: 'fp8' });
    expect(O.parse(TOGGLE, true)).toEqual({ value: true });
    expect(O.parse(TOGGLE, 'true')).toEqual({ value: true });
    expect(O.parse(TOGGLE, false)).toEqual({ value: false });
  });

  test('int_or_auto keeps the word auto rather than turning it into NaN', () => {
    expect(O.parse(AUTO, 'auto')).toEqual({ value: 'auto' });
    expect(O.parse(AUTO, ' auto ')).toEqual({ value: 'auto' });
    expect(O.parse(AUTO, '8')).toEqual({ value: 8 });
  });
});

// SettingField carried its own `Number(raw)` parser until 2026-08-27. These pin
// the behaviour it now shares, because the divergence was invisible: the same
// empty box meant "reset to 0" on the homepage and "that is not a value" on the
// control page, and only one of those is right.
test('an emptied numeric field is an error, not zero', () => {
  const spec = { key: 'gpu', bound: { kind: 'float', min: 0, max: 1 } };
  const r = O.parse(spec, '');
  expect(r.value).toBeUndefined();
  expect(r.error).toMatch(/enter a value/);
});

test('whitespace is not a value either', () => {
  const spec = { key: 'n', bound: { kind: 'int', min: 0, max: 8 } };
  expect(O.parse(spec, '   ').error).toMatch(/enter a value/);
});

test('a real zero still parses, so the guard is about emptiness not falsiness', () => {
  const spec = { key: 'n', bound: { kind: 'int', min: 0, max: 8 } };
  const r = O.parse(spec, '0');
  expect(r.value).toBe(0);
  expect(r.error).toBeUndefined();
});
