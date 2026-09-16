// SPDX-License-Identifier: AGPL-3.0-only
import { expect, test } from 'bun:test';
import { THEME_DARK_BG, THEME_KEY, THEME_LIGHT_BG, readTheme } from '../../../web-shared/theme.js';

test('ground colours are the brand pair, not a third canvas', () => {
  expect(THEME_DARK_BG).toBe('#0F1216');
  expect(THEME_LIGHT_BG).toBe('#FFFFFF');
  expect(THEME_KEY).toBe('avarok-theme');
});

test('without a document, readTheme reports dark rather than throwing', () => {
  expect(readTheme()).toBe('dark');
});
