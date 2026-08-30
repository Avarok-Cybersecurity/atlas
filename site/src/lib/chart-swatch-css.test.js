// SPDX-License-Identifier: AGPL-3.0-only

// The legend swatches on the gate dashboard once rendered as giant filled
// pills with their labels shoved off the edge. The cause was CSS specificity,
// not markup: a panel-wide `.gate-panel svg { width: 100% }` rule (0,1,1) beat
// the swatch's own `width: 18px` rule (0,1,0), so every 18x8 legend glyph was
// stretched to the width of the chart.
//
// `ladder.css` had already met and documented this trap, and fixed it with a
// `:not()` guard — but the fix was never carried across to `dashboard.css`.
// This test pins BOTH, so the next chart panel that adds a sizing rule cannot
// quietly reintroduce it in either file.

import { describe, expect, test } from 'bun:test';
import { readFileSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const styles = resolve(dirname(fileURLToPath(import.meta.url)), '..', 'styles');
const css = (f) => readFileSync(resolve(styles, f), 'utf8');

/**
 * Every rule whose selector sizes an svg inside a chart panel.
 *
 * Comments are stripped FIRST, and deliberately so: the first version of this
 * helper matched against the raw text, which carries the comment sitting above
 * each rule into the "selector". The comment above the real rule explains the
 * `:not()` guard and therefore contains the literal `:not(.gl-swatch)` — so the
 * assertion passed on the prose while the rule underneath was unguarded, and
 * the negative control below came back green. A check that reads its own
 * documentation instead of the code measures nothing.
 */
const svgSizingSelectors = (text, panelClass) =>
  [...text.replace(/\/\*[\s\S]*?\*\//g, '').matchAll(/([^}{]*\bsvg\b[^}{]*)\{([^}]*)\}/g)]
    .filter(([, sel, body]) => sel.includes(panelClass) && /width\s*:\s*100%/.test(body))
    .map(([, sel]) => sel.trim());

describe.each([
  ['dashboard.css', '.gate-panel', '.gl-swatch'],
  ['ladder.css', '.cl-panel', '.cl-swatch']
])('%s', (file, panelClass, swatchClass) => {
  test('the panel-wide svg sizing rule exempts the legend swatch', () => {
    const rules = svgSizingSelectors(css(file), panelClass);
    expect(rules.length).toBeGreaterThan(0); // the rule exists at all
    for (const sel of rules) {
      expect(sel).toContain(`:not(${swatchClass})`);
    }
  });

  test('the swatch still has its own explicit size', () => {
    expect(css(file)).toMatch(new RegExp(`\\${swatchClass}\\s*\\{[^}]*width\\s*:\\s*\\d+px`));
  });
});

describe('the check can fail', () => {
  test('NEGATIVE CONTROL: an unguarded sizing rule is detected', () => {
    // Exactly the shape that shipped the bug.
    const bad = '.gate-panel svg { width: 100%; height: auto; display: block; }';
    const rules = svgSizingSelectors(bad, '.gate-panel');
    expect(rules).toHaveLength(1);
    expect(rules[0]).not.toContain(':not(');
  });

  test('NEGATIVE CONTROL: a rule that does not size is ignored', () => {
    // Guards the filter itself — otherwise the test would trivially pass on a
    // stylesheet with no sizing rule at all.
    const harmless = '.gate-panel svg { opacity: 1; }';
    expect(svgSizingSelectors(harmless, '.gate-panel')).toHaveLength(0);
  });
});
