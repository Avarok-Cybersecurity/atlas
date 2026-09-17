import { test, expect } from 'bun:test';
import { readFileSync } from 'node:fs';

/**
 * Why this file exists: the Lighthouse accessibility gate is 100 on every page
 * and it audits the LIGHT theme, where `color-contrast` is the only check the
 * palette can fail on its own. It did — `--t3` was 4.2:1 on white, and the four
 * chevron hues used as text ran 1.88 to 2.54, because they are the mark's own
 * colours and the mark was drawn for a dark ground.
 *
 * The fix was a second set of tokens rather than a repainted mark: --ch-*-text
 * is the same hue with its lightness lowered, so a fill still carries the brand
 * colour and only glyphs read from the darkened twin. That split is easy to
 * undo by accident — someone "restores" a -text token to the fill value, or
 * nudges a light ground lighter — and nothing in the unit suite would notice
 * until CI failed on a rendered page. This measures it here instead.
 *
 * 4.5:1 is WCAG AA for small text, which is what every failing node was:
 * .ledger-no, details.topo-table > summary, the blog's table headers, and the
 * chevron-coloured .ledger-name and strong.scale-up.
 */
const FLOOR = 4.5;

const tokens = readFileSync(new URL('../../../web-shared/avarok-tokens.css', import.meta.url), 'utf8');

/**
 * Both themes declare the same token names, so a match has to be scoped to one
 * block. Neither block nests braces, which is what makes the lazy match to the
 * first column-zero `}` safe.
 */
const block = (selector) => {
  const m = tokens.match(new RegExp(`${selector}\\s*\\{([\\s\\S]*?)\\n\\}`));
  if (!m) throw new Error(`${selector} is not in web-shared/avarok-tokens.css`);
  return m[1];
};
const DARK = block(':root');
const LIGHT = block('\\[data-theme="light"\\]');

const declaration = (src, where, name) => {
  const m = src.match(new RegExp(`--${name}:\\s*([^;]+);`));
  if (!m) throw new Error(`--${name} is not in the ${where} block of web-shared/avarok-tokens.css`);
  return m[1].trim();
};
const lightHex = (name) => {
  const v = declaration(LIGHT, 'light', name);
  if (!/^#[0-9a-fA-F]{6}$/.test(v)) throw new Error(`--${name} is "${v}" in the light block, not a hex literal`);
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
 * The three light grounds small text can land on. --card is omitted because it
 * is #FFFFFF, the same value as --bg; --card-2 is the darkest of the three and
 * therefore the binding one.
 */
const GROUNDS = ['bg', 'bg2', 'card-2'].map((n) => [`--${n}`, lightHex(n)]);

/** Every token that is only ever used as text, and so owes AA rather than 3:1. */
const TEXT_TOKENS = ['t3', 'ch-violet-text', 'ch-cyan-text', 'ch-green-text', 'ch-gold-text'];

/** The -text twins, paired with the fill they are derived from. */
const TWINS = [
  ['ch-violet-text', 'ch-violet'],
  ['ch-cyan-text', 'ch-cyan'],
  ['ch-green-text', 'ch-green'],
  ['ch-gold-text', 'ch-gold']
];

test('the grounds and the text tokens are all present (so the loops below are not vacuous)', () => {
  expect(GROUNDS).toHaveLength(3);
  for (const [, hex] of GROUNDS) expect(hex).toMatch(/^#[0-9a-f]{6}$/i);
  expect(TEXT_TOKENS).toHaveLength(5);
  for (const name of TEXT_TOKENS) expect(lightHex(name)).toMatch(/^#[0-9a-f]{6}$/i);
});

for (const [groundName, ground] of GROUNDS) {
  for (const name of TEXT_TOKENS) {
    test(`light --${name} (${lightHex(name)}) clears ${FLOOR}:1 on ${groundName} ${ground}`, () => {
      expect(contrast(lightHex(name), ground)).toBeGreaterThanOrEqual(FLOOR);
    });
  }
}

for (const [textName, fillName] of TWINS) {
  test(`dark --${textName} is --${fillName}, not a second value to keep in sync`, () => {
    // On the dark ground the mark's own hue already clears AA as text, so the
    // twin must resolve THROUGH the fill rather than restating it: a hex here
    // would be a copy that drifts the next time the fill moves.
    expect(declaration(DARK, 'dark', textName)).toBe(`var(--${fillName})`);
  });
}

test('the light theme darkens the twins rather than inheriting them', () => {
  // Without this the block above would still pass if the light theme simply
  // stopped overriding the -text tokens, which is the exact regression that
  // puts 2.08:1 cyan back on white.
  for (const [textName, fillName] of TWINS) {
    expect(lightHex(textName).toLowerCase()).not.toBe(declaration(DARK, 'dark', fillName).toLowerCase());
    expect(relLum(lightHex(textName))).toBeLessThan(relLum(declaration(DARK, 'dark', fillName)));
  }
});

test('--sx and --sx-text are separate channels', () => {
  // The section accent drives fills and text from one custom property each.
  // If --sx-text collapsed back onto --sx, every `color: var(--sx-text)` in
  // site/src would quietly go back to the fill hue.
  expect(declaration(DARK, 'dark', 'sx')).toBe('var(--ch-violet)');
  expect(declaration(DARK, 'dark', 'sx-text')).toBe('var(--ch-violet-text)');
});

test('the check can distinguish a passing colour from a failing one', () => {
  // Without this, every assertion above would still pass if `contrast` were
  // returning a constant. #9AA0A8 is the kind of mid grey that looks like a
  // plausible metadata token and fails AA on white at 2.8:1.
  const white = lightHex('bg');
  expect(contrast('#9AA0A8', white)).toBeLessThan(FLOOR);
  expect(contrast('#000000', white)).toBeGreaterThan(FLOOR);
});
