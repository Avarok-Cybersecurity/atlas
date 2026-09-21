// SPDX-License-Identifier: AGPL-3.0-only

import { describe, expect, test } from 'bun:test';
import { SHORTCUTS, shortcut } from './shortcuts.js';

const FREE = { typing: false, overlayOpen: false, modified: false };

describe('the map itself', () => {
  test('every spec key is present, in the spec order', () => {
    // §3's keyboard map, verbatim. A row disappearing from the table would
    // silently disappear from the sheet too — this is the tripwire.
    expect(SHORTCUTS.map((s) => s.keys)).toEqual([
      '1–8',
      '↑ ↓',
      'l',
      'n',
      's',
      'a',
      'c',
      'p',
      '?',
      'Esc'
    ]);
  });

  test('every row carries prose for the sheet', () => {
    for (const s of SHORTCUTS) {
      expect(s.label.length).toBeGreaterThan(3);
    }
  });
});

describe('dispatch', () => {
  test.each([
    ['1', { kind: 'select', key: '1' }],
    ['8', { kind: 'select', key: '8' }],
    ['l', { kind: 'tab', tab: 'logs' }],
    ['n', { kind: 'tab', tab: 'launch' }],
    ['s', { kind: 'stop' }],
    ['a', { kind: 'alerts' }],
    ['c', { kind: 'cluster' }],
    ['p', { kind: 'pause' }],
    ['?', { kind: 'sheet' }]
  ])('%s dispatches', (key, action) => {
    expect(shortcut(key, FREE)).toEqual(action);
  });

  test('caps lock does not disable the console', () => {
    expect(shortcut('L', FREE)).toEqual({ kind: 'tab', tab: 'logs' });
    expect(shortcut('S', FREE)).toEqual({ kind: 'stop' });
  });

  test('9 and 0 select nothing — the roster caps hotkeys at 8', () => {
    expect(shortcut('9', FREE)).toBeNull();
    expect(shortcut('0', FREE)).toBeNull();
  });

  test('arrows and Esc are documented but never dispatched globally', () => {
    // Arrows rove only inside the roster (its own handler); a global arrow
    // would steal keyboard scrolling from every tabindex-0 scroll region.
    expect(shortcut('ArrowDown', FREE)).toBeNull();
    expect(shortcut('ArrowUp', FREE)).toBeNull();
    expect(shortcut('Escape', FREE)).toBeNull();
  });

  test('unknown keys do nothing', () => {
    expect(shortcut('x', FREE)).toBeNull();
    expect(shortcut('Enter', FREE)).toBeNull();
    expect(shortcut(undefined, FREE)).toBeNull();
  });
});

describe('suppression', () => {
  test('typing wins — every key is text while an input has focus', () => {
    for (const k of ['1', 'l', 'n', 's', 'a', 'c', 'p', '?']) {
      expect(shortcut(k, { ...FREE, typing: true })).toBeNull();
    }
  });

  test('an open overlay owns the keyboard', () => {
    for (const k of ['1', 's', 'c', '?']) {
      expect(shortcut(k, { ...FREE, overlayOpen: true })).toBeNull();
    }
  });

  test('modifier chords belong to the browser', () => {
    expect(shortcut('1', { ...FREE, modified: true })).toBeNull();
    expect(shortcut('l', { ...FREE, modified: true })).toBeNull();
  });

  test('context must be said, not assumed', () => {
    expect(() => shortcut('s', {})).toThrow(TypeError);
    expect(() => shortcut('s', { typing: false, overlayOpen: false })).toThrow(TypeError);
    expect(() => shortcut('s')).toThrow(TypeError);
  });
});
