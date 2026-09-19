// SPDX-License-Identifier: AGPL-3.0-only
import { expect, test } from 'bun:test';
import { readFileSync } from 'node:fs';
import { THEME_DARK_BG, THEME_KEY, THEME_LIGHT_BG, readTheme } from '../../../web-shared/theme.js';

test('ground colours are the brand pair, not a third canvas', () => {
  expect(THEME_DARK_BG).toBe('#0F1216');
  expect(THEME_LIGHT_BG).toBe('#FFFFFF');
  expect(THEME_KEY).toBe('avarok-theme');
});

test('without a document, readTheme reports dark rather than throwing', () => {
  expect(readTheme()).toBe('dark');
});

/**
 * The blocking boot script in app.html does two things now: it settles
 * data-theme, and on the home page it queues the one hero render that theme
 * calls for. The hero poster is the largest paint on the page, so if the preload stops being
 * emitted the largest paint on the page loses its head start and there is
 * nothing else in the markup to recover it.
 *
 * There is no DOM here, so the script runs against the handful of globals it
 * actually touches. That keeps the test honest about the source in app.html
 * rather than restating the URLs somewhere a copy could drift.
 */
function runBootScript({ stored = null, prefersLight = false, pathname = '/', storageThrows = false } = {}) {
  const html = readFileSync(new URL('../app.html', import.meta.url), 'utf8');
  const source = html.match(/<script>([\s\S]*?)<\/script\s*>/i)?.[1];
  expect(source, 'no inline boot script found in app.html').toBeTruthy();

  const appended = [];
  const documentElement = {
    attributes: {},
    setAttribute(name, value) {
      this.attributes[name] = value;
    }
  };
  const document = {
    documentElement,
    head: {
      appendChild(node) {
        appended.push(node);
      }
    },
    createElement(tag) {
      return {
        tag,
        setAttribute(name, value) {
          this[name] = value;
        }
      };
    }
  };
  const localStorage = {
    getItem() {
      if (storageThrows) throw new Error('storage unavailable');
      return stored;
    }
  };
  const matchMedia = query => ({ matches: query.includes('light') ? prefersLight : !prefersLight });

  new Function('document', 'localStorage', 'matchMedia', 'location', source)(
    document,
    localStorage,
    matchMedia,
    { pathname }
  );
  return { theme: documentElement.attributes['data-theme'], preloads: appended };
}

test('the home page preloads the hero poster when the theme is dark', () => {
  const { theme, preloads } = runBootScript({ stored: 'dark' });
  expect(theme).toBe('dark');
  expect(preloads).toHaveLength(1);
  expect(preloads[0].rel).toBe('preload');
  expect(preloads[0].as).toBe('image');
  expect(preloads[0].fetchpriority).toBe('high');
  expect(preloads[0].href).toBe('/media/console-ask.webp');
});

test('the home page preloads the same poster when the theme is light', () => {
  const { theme, preloads } = runBootScript({ stored: 'light' });
  expect(theme).toBe('light');
  expect(preloads).toHaveLength(1);
  expect(preloads[0].href).toBe('/media/console-ask.webp');
});

test('an unset preference follows the media query, poster and all', () => {
  expect(runBootScript({ prefersLight: true }).preloads[0].href).toBe('/media/console-ask.webp');
  expect(runBootScript({ prefersLight: false }).preloads[0].href).toBe('/media/console-ask.webp');
});

test('the prerendered /index.html counts as the home page', () => {
  expect(runBootScript({ stored: 'dark', pathname: '/index.html' }).preloads).toHaveLength(1);
});

test('pages without the hero preload nothing', () => {
  const { theme, preloads } = runBootScript({ stored: 'dark', pathname: '/roadmap' });
  expect(theme).toBe('dark');
  expect(preloads).toHaveLength(0);
});

test('when storage is unreadable the page still falls to dark and preloads its hero', () => {
  const { theme, preloads } = runBootScript({ storageThrows: true });
  expect(theme).toBe('dark');
  expect(preloads).toHaveLength(1);
  expect(preloads[0].href).toBe('/media/console-ask.webp');
});
