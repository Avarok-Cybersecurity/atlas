import { test, expect } from 'bun:test';
import { readFileSync } from 'node:fs';
import { MODEL_COLORS, UNKNOWN_MODEL_COLOR } from './series-colors.js';

/**
 * The benchmark chart's series colours were hand-derived against specific
 * surfaces, and the comment above them asks for re-derivation whenever the
 * palette moves. This makes that non-optional: the surfaces are read from the
 * token file rather than retyped, so a change to --bg or --card is measured
 * against the series the moment it lands.
 *
 * >=3:1 is the floor the palette has always held to — WCAG's non-text contrast
 * minimum, which is what a chart line is.
 */
const FLOOR = 3;

const tokens = readFileSync(new URL('../../../web-shared/atlas-tokens.css', import.meta.url), 'utf8');
const token = (name) => {
  const m = tokens.match(new RegExp(`--${name}:\\s*(#[0-9a-fA-F]{6})`));
  if (!m) throw new Error(`--${name} is not in web-shared/atlas-tokens.css`);
  return m[1];
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

const SURFACES = [['--bg', token('bg')], ['--card', token('card')]];
const SERIES = [...Object.entries(MODEL_COLORS), ['fallback', UNKNOWN_MODEL_COLOR]];

test('the series palette is not empty (so the loops below are not vacuous)', () => {
  expect(SERIES.length).toBeGreaterThanOrEqual(4);
  for (const [, hex] of SERIES) expect(hex).toMatch(/^#[0-9a-f]{6}$/i);
});

for (const [surfaceName, surface] of SURFACES) {
  for (const [model, hex] of SERIES) {
    test(`${model} (${hex}) clears ${FLOOR}:1 on ${surfaceName} ${surface}`, () => {
      expect(contrast(hex, surface)).toBeGreaterThanOrEqual(FLOOR);
    });
  }
}

test('the check can distinguish a passing colour from a failing one', () => {
  // Without this, every assertion above would still pass if `contrast` were
  // returning a constant. #1a1d22 is a near-ground grey: a plausible-looking
  // series colour that is invisible on the canvas.
  expect(contrast('#1a1d22', token('bg'))).toBeLessThan(FLOOR);
  expect(contrast('#ffffff', token('bg'))).toBeGreaterThan(FLOOR);
});
