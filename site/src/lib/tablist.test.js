// SPDX-License-Identifier: AGPL-3.0-only

import { describe, expect, test } from 'bun:test';
import { moveTab } from './tablist.js';

describe('moveTab', () => {
  test('moves and wraps in both directions', () => {
    expect(moveTab('ArrowRight', 0, 3)).toBe(1);
    expect(moveTab('ArrowRight', 2, 3)).toBe(0);
    expect(moveTab('ArrowLeft', 2, 3)).toBe(1);
    expect(moveTab('ArrowLeft', 0, 3)).toBe(2);
  });

  test('Home and End jump to the ends', () => {
    expect(moveTab('Home', 2, 3)).toBe(0);
    expect(moveTab('End', 0, 3)).toBe(2);
  });

  test('keys that belong to the dialog are not swallowed', () => {
    // Escape must reach the modal's own handler and Tab must reach the focus
    // trap; returning an index for them would break both.
    for (const key of ['Escape', 'Tab', 'Enter', ' ', 'a', 'ArrowUp']) {
      expect(moveTab(key, 1, 3)).toBeNull();
    }
  });

  test('a single tab never moves', () => {
    expect(moveTab('ArrowRight', 0, 1)).toBeNull();
    expect(moveTab('End', 0, 1)).toBeNull();
    expect(moveTab('ArrowRight', 0, 0)).toBeNull();
  });
});
