import { test, expect, describe } from 'bun:test';
import { readFileSync } from 'node:fs';
import { MODEL_COLORS, MODEL_COLORS_LIGHT, MODEL_SLUGS, UNKNOWN_MODEL_COLOR, colorFor } from './series-colors.js';

/**
 * The benchmark chart's series colours were hand-derived against specific
 * surfaces, and the comment above them asks for re-derivation whenever the
 * palette moves. This makes that non-optional: the surfaces are read from the
 * token file rather than retyped, so a change to --bg or --card is measured
 * against the series the moment it lands.
 *
 * Extended 2026-09-19 to measure BOTH themes. One hex set used to serve both
 * grounds, and on the light theme's white three of the six were under the
 * floor (teal 1.96, citron 1.33, sky 1.45). Each theme now declares its own
 * `--series-<slug>` tokens; this reads each block separately and measures it
 * against its own grounds, so a token that is only right for one theme cannot
 * pass by being measured on the other.
 *
 * >=3:1 is the floor the palette has always held to — WCAG's non-text contrast
 * minimum, which is what a chart line is.
 */
const FLOOR = 3;
/** The bar the 2026-08-30 extension set for itself, and which the light set holds too. */
const LIGHT_BAR = 4;

const tokens = readFileSync(new URL('../../../web-shared/avarok-tokens.css', import.meta.url), 'utf8');

/**
 * Both themes declare the same token names, so a match has to be scoped to one
 * block. Neither block nests braces, which is what makes the lazy match to the
 * first column-zero `}` safe. (Same helper as light-text-contrast.test.js.)
 */
const block = (selector) => {
  const m = tokens.match(new RegExp(`${selector}\\s*\\{([\\s\\S]*?)\\n\\}`));
  if (!m) throw new Error(`${selector} is not in web-shared/avarok-tokens.css`);
  return m[1];
};
const DARK = block(':root');
const LIGHT = block('\\[data-theme="light"\\]');

const hexIn = (src, where, name) => {
  const m = src.match(new RegExp(`--${name}:\\s*([^;]+);`));
  if (!m) throw new Error(`--${name} is not in the ${where} block of web-shared/avarok-tokens.css`);
  const v = m[1].trim();
  if (!/^#[0-9a-fA-F]{6}$/.test(v)) throw new Error(`--${name} is "${v}" in the ${where} block, not a hex literal`);
  return v;
};

const srgb = (hex) => [1, 3, 5].map((i) => parseInt(hex.slice(i, i + 2), 16) / 255);
const lin = (c) => (c <= 0.04045 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4);
const relLum = (hex) => {
  const v = srgb(hex);
  return 0.2126 * lin(v[0]) + 0.7152 * lin(v[1]) + 0.0722 * lin(v[2]);
};
const contrast = (a, b) => {
  const [hi, lo] = [relLum(a), relLum(b)].sort((x, y) => y - x);
  return (hi + 0.05) / (lo + 0.05);
};

/**
 * The grounds a series line can land on, per theme. Light omits --card because
 * it is #FFFFFF, the same value as --bg, and adds --bg2 because the dashboard's
 * alt sections use it; --card-2 is the darkest light ground and so the binding
 * one.
 */
const THEMES = [
  { name: 'dark', src: DARK, grounds: ['bg', 'card'], table: MODEL_COLORS, bar: FLOOR },
  { name: 'light', src: LIGHT, grounds: ['bg', 'bg2', 'card-2'], table: MODEL_COLORS_LIGHT, bar: LIGHT_BAR }
];

const MODELS = Object.keys(MODEL_SLUGS);

test('the series palette is not empty (so the loops below are not vacuous)', () => {
  expect(MODELS.length).toBeGreaterThanOrEqual(4);
  expect(Object.keys(MODEL_COLORS)).toEqual(MODELS);
  expect(Object.keys(MODEL_COLORS_LIGHT)).toEqual(MODELS);
  expect(new Set(Object.values(MODEL_SLUGS)).size).toBe(MODELS.length); // slugs are distinct
  expect(UNKNOWN_MODEL_COLOR).toMatch(/^#[0-9a-f]{6}$/i);
});

describe.each(THEMES)('$name theme', ({ name, src, grounds, table, bar }) => {
  const surfaces = grounds.map((g) => [`--${g}`, hexIn(src, name, g)]);

  for (const model of MODELS) {
    const slug = MODEL_SLUGS[model];
    // The CSS declaration is what the browser paints, so it is what gets
    // measured; the JS table is the var() fallback and must match it.
    const hex = hexIn(src, name, `series-${slug}`);
    test(`--series-${slug} equals the JS table entry for ${model}`, () => {
      expect(hex).toBe(table[model]);
    });
    for (const [surfaceName, surface] of surfaces) {
      test(`--series-${slug} (${hex}) clears ${bar}:1 on ${surfaceName} ${surface}`, () => {
        expect(contrast(hex, surface)).toBeGreaterThanOrEqual(bar);
      });
    }
  }

  for (const [surfaceName, surface] of surfaces) {
    test(`the fallback ${UNKNOWN_MODEL_COLOR} clears ${FLOOR}:1 on ${surfaceName} ${surface}`, () => {
      expect(contrast(UNKNOWN_MODEL_COLOR, surface)).toBeGreaterThanOrEqual(FLOOR);
    });
  }
});

/**
 * The dark set is the 2026-08-30 palette and is deliberately unchanged by the
 * light-theme work: its separations were searched under dichromat simulation
 * and are documented in series-colors.js. A literal pin, not a derivation, so
 * that editing the JS table and the CSS block together still fails here.
 */
const DARK_PINNED = {
  copper: '#ee6f2f',
  steel: '#2f88ee',
  teal: '#51cdb0',
  rose: '#cd517a',
  citron: '#d5e88a',
  sky: '#a1e0f7'
};
test('the dark series values are the shipped 2026-08-30 set, unchanged', () => {
  expect(Object.fromEntries(MODELS.map((m) => [MODEL_SLUGS[m], MODEL_COLORS[m]]))).toEqual(DARK_PINNED);
  for (const [slug, hex] of Object.entries(DARK_PINNED)) expect(hexIn(DARK, 'dark', `series-${slug}`)).toBe(hex);
});

test('the light block declares exactly the same series tokens as the dark block', () => {
  const names = (src) => [...src.matchAll(/--series-([a-z0-9-]+):/g)].map((m) => m[1]).sort();
  expect(names(LIGHT)).toEqual(names(DARK));
  expect(names(DARK)).toEqual(Object.values(MODEL_SLUGS).sort());
});

test('colorFor resolves through the theme token with the dark hex as fallback', () => {
  for (const model of MODELS) {
    expect(colorFor(model)).toBe(`var(--series-${MODEL_SLUGS[model]}, ${MODEL_COLORS[model]})`);
  }
  expect(colorFor('nobody/Not-A-Model')).toBe(UNKNOWN_MODEL_COLOR);
  expect(colorFor(undefined)).toBe(UNKNOWN_MODEL_COLOR);
});

test('the check can distinguish a passing colour from a failing one', () => {
  // Without this, every assertion above would still pass if `contrast` were
  // returning a constant. #1a1d22 is a near-ground grey: a plausible-looking
  // series colour that is invisible on the dark canvas.
  expect(contrast('#1a1d22', hexIn(DARK, 'dark', 'bg'))).toBeLessThan(FLOOR);
  expect(contrast('#ffffff', hexIn(DARK, 'dark', 'bg'))).toBeGreaterThan(FLOOR);
  // And on the light ground: the dark theme's citron, which is what used to be
  // drawn there, reads 1.33:1 on white.
  expect(contrast(DARK_PINNED.citron, hexIn(LIGHT, 'light', 'bg'))).toBeLessThan(FLOOR);
});
