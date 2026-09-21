// SPDX-License-Identifier: AGPL-3.0-only

import { test, expect } from 'bun:test';
import { copyText, selectText, copyOrSelect, copyLabel } from './clipboard.js';

const ok = () => {
  const seen = [];
  return { nav: { clipboard: { writeText: async (t) => void seen.push(t) } }, seen };
};

test('a successful copy reports copied and writes the text', async () => {
  const { nav, seen } = ok();
  expect(await copyText('curl x | sh', nav)).toBe('copied');
  expect(seen).toEqual(['curl x | sh']);
});

test('a refusal reports denied rather than throwing', async () => {
  // This is the case every previous copy of this code swallowed. A caller that
  // sees `denied` can show the text; one that sees an exception it never
  // catches shows a button that flashed nothing.
  const nav = { clipboard: { writeText: async () => { throw new Error('NotAllowedError'); } } };
  expect(await copyText('x', nav)).toBe('denied');
});

test('no clipboard at all is denied, not a crash', async () => {
  // Plain http on a LAN address has no `navigator.clipboard` — which is exactly
  // where this control page runs.
  expect(await copyText('x', {})).toBe('denied');
  expect(await copyText('x', { clipboard: {} })).toBe('denied');
  expect(await copyText('x', undefined)).toBe('denied');
  expect(await copyText('x', { clipboard: { writeText: 'not a function' } })).toBe('denied');
});

test('a synchronous throw is caught too', async () => {
  // `writeText` is not required to return a promise before it fails.
  const nav = { clipboard: { writeText: () => { throw new Error('boom'); } } };
  expect(await copyText('x', nav)).toBe('denied');
});

test('null and undefined are copied as empty, never as the string "null"', async () => {
  const { nav, seen } = ok();
  await copyText(null, nav);
  await copyText(undefined, nav);
  expect(seen).toEqual(['', '']);
});

test('selecting nothing is false, not an exception', () => {
  expect(selectText(null)).toBe(false);
  expect(selectText(undefined)).toBe(false);
});

test('a refused copy never reports success', () => {
  // The whole point: every state except `copied` must read as "not yet".
  for (const state of ['idle', 'manual', 'blocked']) {
    expect(copyLabel(state)).not.toBe('Copied');
  }
  expect(copyLabel('copied')).toBe('Copied');
});

test('a refusal tells the operator what to do next, not just that it failed', () => {
  // "Press ⌘/Ctrl+C" is actionable because the text has just been selected.
  expect(copyLabel('manual')).toContain('C');
  expect(copyLabel('blocked').length).toBeGreaterThan(0);
});

test('the idle label is the caller\'s, so one component can say more', () => {
  expect(copyLabel('idle', 'Copy command')).toBe('Copy command');
  expect(copyLabel('copied', 'Copy command')).toBe('Copied');
});

test('copyOrSelect falls through to selection when the clipboard refuses', async () => {
  // No document in this runner, so selectText cannot succeed: the honest
  // answer is `blocked`, never `copied`.
  const refusing = { clipboard: { writeText: () => Promise.reject(new Error('denied')) } };
  const orig = globalThis.navigator;
  try {
    Object.defineProperty(globalThis, 'navigator', { value: refusing, configurable: true });
    expect(await copyOrSelect('x', null)).toBe('blocked');
  } finally {
    Object.defineProperty(globalThis, 'navigator', { value: orig, configurable: true });
  }
});
