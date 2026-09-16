// SPDX-License-Identifier: AGPL-3.0-only
//
// One theme for atlascybernetics.ai and blog.atlascybernetics.ai.
// The blocking boot script in each app.html must stay in lockstep with KEY
// and the two ground colours here — theme-color.test.js pins the colours.

export const THEME_KEY = 'avarok-theme';
export const THEME_DARK_BG = '#0F1216';
export const THEME_LIGHT_BG = '#FFFFFF';

export function readTheme() {
  if (typeof document === 'undefined') return 'dark';
  return document.documentElement.getAttribute('data-theme') === 'light' ? 'light' : 'dark';
}

export function applyTheme(theme) {
  if (typeof document === 'undefined') return;
  const next = theme === 'light' ? 'light' : 'dark';
  document.documentElement.setAttribute('data-theme', next);
  try {
    localStorage.setItem(THEME_KEY, next);
  } catch {
    /* private mode */
  }
  const color = next === 'light' ? THEME_LIGHT_BG : THEME_DARK_BG;
  for (const meta of document.querySelectorAll('meta[name="theme-color"]')) {
    if (!meta.hasAttribute('media')) meta.setAttribute('content', color);
  }
}

export function toggleTheme() {
  applyTheme(readTheme() === 'light' ? 'dark' : 'light');
  return readTheme();
}
