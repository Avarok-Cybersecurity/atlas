// SPDX-License-Identifier: AGPL-3.0-only

// The interesting part of a profile is the reading, not the writing: the value
// is attacker-writable by hand, may have been written by a different version of
// this site, and lives in an API that throws outright in a private window. Each
// test below is one of those, because each one is a way the control page could
// have been taken down by a preference.

import { test, expect } from 'bun:test';
import * as P from './profile.js';

/** A localStorage-shaped fake. `fail` makes every method throw, like Safari private mode. */
function storage(initial = null, fail = false) {
  let value = initial;
  return {
    getItem(k) {
      if (fail) throw new Error('storage disabled');
      return k === P.KEY ? value : null;
    },
    setItem(k, v) {
      if (fail) throw new Error('quota exceeded');
      if (k === P.KEY) value = v;
    },
    removeItem(k) {
      if (fail) throw new Error('storage disabled');
      if (k === P.KEY) value = null;
    },
    get raw() {
      return value;
    }
  };
}

test('a fresh browser loads an empty profile rather than failing', () => {
  expect(P.load(storage())).toEqual(P.empty());
});

test('empty() hands back a fresh object each time, never a shared one', () => {
  const a = P.empty();
  a.selected.push('x');
  expect(P.empty().selected).toEqual([]);
});

test('absent storage is not an error', () => {
  expect(P.load(null)).toEqual(P.empty());
  expect(P.save(null, P.empty())).toBe(false);
});

test('storage that throws on every call costs preferences, not the page', () => {
  const s = storage(null, true);
  expect(P.load(s)).toEqual(P.empty());
  expect(P.save(s, P.empty())).toBe(false);
  expect(P.clear(s)).toBe(false);
});

test('text that is not JSON is read as no profile', () => {
  expect(P.load(storage('}{ not json'))).toEqual(P.empty());
});

test('JSON that is not an object is read as no profile', () => {
  for (const raw of ['null', '42', '"a string"', '[1,2,3]']) {
    expect(P.load(storage(raw))).toEqual(P.empty());
  }
});

test('a profile from a newer site is discarded, not half-applied', () => {
  // Half-applying is the dangerous option: a field that changed meaning between
  // versions would be read under its old meaning.
  const raw = JSON.stringify({ v: P.VERSION + 1, recipe: 'qwen3.6-27b', head: 'aa' });
  expect(P.load(storage(raw))).toEqual(P.empty());
});

test('a profile with no version at all is discarded', () => {
  expect(P.load(storage(JSON.stringify({ recipe: 'qwen3.6-27b' })))).toEqual(P.empty());
});

test('a stored head that is not among the stored machines is dropped', () => {
  // Otherwise storage becomes a way around the launch flow's own rule that the
  // head must be one of the selected machines.
  const raw = JSON.stringify({ v: P.VERSION, selected: ['aa', 'bb'], head: 'cc' });
  const p = P.load(storage(raw));
  expect(p.selected).toEqual(['aa', 'bb']);
  expect(p.head).toBeNull();
});

test('a head that is among the machines survives', () => {
  const raw = JSON.stringify({ v: P.VERSION, selected: ['aa', 'bb'], head: 'bb' });
  expect(P.load(storage(raw)).head).toBe('bb');
});

test('duplicate machines collapse and the list is capped', () => {
  const many = Array.from({ length: 50 }, (_, i) => `node-${i}`);
  const raw = JSON.stringify({ v: P.VERSION, selected: ['aa', 'aa', ...many] });
  const p = P.load(storage(raw));
  expect(p.selected.length).toBeLessThanOrEqual(8);
  expect(new Set(p.selected).size).toBe(p.selected.length);
});

test('absurdly long ids are refused rather than rendered', () => {
  const raw = JSON.stringify({ v: P.VERSION, recipe: 'r'.repeat(5000), selected: ['n'.repeat(5000)] });
  const p = P.load(storage(raw));
  expect(p.recipe).toBeNull();
  expect(p.selected).toEqual([]);
});

test('non-string ids are refused', () => {
  const raw = JSON.stringify({ v: P.VERSION, recipe: 42, head: {}, selected: [1, null, true, 'ok'] });
  const p = P.load(storage(raw));
  expect(p.recipe).toBeNull();
  expect(p.selected).toEqual(['ok']);
});

test('override values keep scalars and drop everything else', () => {
  const raw = JSON.stringify({
    v: P.VERSION,
    overrides: {
      'a-recipe': {
        port: 8888,
        kv_cache_dtype: 'fp8',
        enable_mtp: true,
        nested: { evil: 1 },
        list: [1, 2],
        nothing: null
      }
    }
  });
  expect(P.load(storage(raw)).overrides['a-recipe']).toEqual({
    port: 8888,
    kv_cache_dtype: 'fp8',
    enable_mtp: true
  });
});

test('a non-finite number is refused', () => {
  // NaN and Infinity become null through JSON, so seeing one means the value
  // was hand-written. It cannot be a setting the editor produced.
  const p = P.sanitize({ v: P.VERSION, overrides: { r: { a: Number.NaN, b: Infinity, c: 1 } } });
  expect(p.overrides.r).toEqual({ c: 1 });
});

test('a recipe whose every override was refused does not take a slot', () => {
  const p = P.sanitize({ v: P.VERSION, overrides: { r: { nested: {} }, ok: { port: 1 } } });
  expect(Object.keys(p.overrides)).toEqual(['ok']);
});

test('the number of remembered recipes and of overrides per recipe are both capped', () => {
  const overrides = {};
  for (let i = 0; i < 200; i += 1) {
    const map = {};
    for (let k = 0; k < 200; k += 1) map[`k${k}`] = k;
    overrides[`recipe-${i}`] = map;
  }
  const p = P.sanitize({ v: P.VERSION, overrides });
  expect(Object.keys(p.overrides).length).toBeLessThanOrEqual(24);
  for (const map of Object.values(p.overrides)) {
    expect(Object.keys(map).length).toBeLessThanOrEqual(64);
  }
});

test('what save writes is what load reads back', () => {
  const s = storage();
  const p = P.merge(P.empty(), {
    recipe: 'qwen3.6-27b',
    selected: ['aa', 'bb'],
    head: 'aa',
    overrides: { 'qwen3.6-27b': { port: 8890 } }
  });
  expect(P.save(s, p)).toBe(true);
  expect(P.load(s)).toEqual(p);
});

test('save sanitises on the way out, so nothing is stored that load would refuse', () => {
  // Without this, a bug upstream persists a value that never comes back, which
  // reads to the operator as "the profile does not save".
  const s = storage();
  P.save(s, { v: P.VERSION, selected: ['aa'], head: 'not-selected', recipe: 'r' });
  expect(P.load(s).head).toBeNull();
  expect(JSON.parse(s.raw).head).toBeNull();
});

test('an oversized profile is refused rather than filling the origin quota', () => {
  const s = storage();
  const huge = { v: P.VERSION, overrides: {} };
  for (let i = 0; i < 24; i += 1) {
    const map = {};
    for (let k = 0; k < 64; k += 1) map[`k${k}`] = 'v'.repeat(128);
    huge.overrides[`recipe-${i}`] = map;
  }
  expect(P.save(s, huge)).toBe(false);
  expect(s.raw).toBeNull();
});

test('clear forgets everything', () => {
  const s = storage();
  P.save(s, P.merge(P.empty(), { recipe: 'r' }));
  expect(P.clear(s)).toBe(true);
  expect(P.load(s)).toEqual(P.empty());
});

test('rememberOverrides leaves other recipes alone', () => {
  let p = P.rememberOverrides(P.empty(), 'a', { port: 1 });
  p = P.rememberOverrides(p, 'b', { port: 2 });
  expect(p.overrides).toEqual({ a: { port: 1 }, b: { port: 2 } });
});

test('remembering an empty map removes the recipe rather than storing nothing', () => {
  let p = P.rememberOverrides(P.empty(), 'a', { port: 1 });
  p = P.rememberOverrides(p, 'a', {});
  expect(p.overrides).toEqual({});
});

test('overridesFor hands back a copy, so editing it cannot mutate the profile', () => {
  const p = P.rememberOverrides(P.empty(), 'a', { port: 1 });
  const got = P.overridesFor(p, 'a');
  got.port = 9999;
  expect(P.overridesFor(p, 'a')).toEqual({ port: 1 });
});

test('overridesFor answers an empty map for an unknown or absent recipe', () => {
  expect(P.overridesFor(P.empty(), null)).toEqual({});
  expect(P.overridesFor(P.empty(), 'never-seen')).toEqual({});
});

test('merge cannot be used to write a foreign version', () => {
  expect(P.merge(P.empty(), { v: 99, recipe: 'r' }).v).toBe(P.VERSION);
});

test('a stored __proto__ key cannot set the prototype of what load returns', () => {
  // JSON.parse makes __proto__ an ordinary own property and Object.entries
  // hands it over, but writing it back with obj[key] = value goes through the
  // inherited setter and swaps the prototype instead of adding a key. The
  // object then reads as empty while `overrides.anything` resolves through the
  // attacker's value — load() is documented as total, and a return value whose
  // prototype came from localStorage is not that.
  const raw = '{"v":1,"overrides":{"__proto__":{"gpu":0.9},"real":{"gpu":0.8}}}';
  const p = P.load({ getItem: () => raw });
  expect(Object.getPrototypeOf(p.overrides)).toBe(Object.prototype);
  expect(p.overrides.gpu).toBeUndefined();
  expect(Object.keys(p.overrides)).toEqual(['real']);
  // And nothing leaked onto every other object in the page.
  expect(/** @type {any} */ ({}).gpu).toBeUndefined();
});

test('constructor and prototype are refused as recipe ids too', () => {
  const raw = '{"v":1,"overrides":{"constructor":{"a":1},"prototype":{"b":2},"ok":{"c":3}}}';
  const p = P.load({ getItem: () => raw });
  expect(Object.keys(p.overrides)).toEqual(['ok']);
});

// This one passes with or without the key guard, and that is the point of
// writing it down: the inner map is safe for a DIFFERENT reason — `scalar()`
// refuses objects, and assigning a non-object to `__proto__` is a no-op per
// spec. So the safety here rests on a coupling nobody states out loud. Widen
// `scalar()` to accept objects and this map becomes pollutable; this test is
// what fails when someone does.
test('the inner map stays clean because scalar() refuses objects', () => {
  const raw = '{"v":1,"overrides":{"r":{"__proto__":{"x":1},"real":5}}}';
  const p = P.load({ getItem: () => raw });
  expect(Object.getPrototypeOf(p.overrides.r)).toBe(Object.prototype);
  expect(Object.keys(p.overrides.r)).toEqual(['real']);
});
