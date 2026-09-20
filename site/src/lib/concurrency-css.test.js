// SPDX-License-Identifier: AGPL-3.0-only
//
// The concurrency tab's comparison card once rendered inside `<div class="cc">`
// with `<span class="cc-chip">` pills — and chat.css owns both names for the
// chat modal: `.cc` is a fixed-width flex column with `max-height` and
// `overflow: hidden` (the table stopped at C=2 with no scrollbar), a mono
// face and a receipt-tear `::before` over the heading; `.cc-chip` is an
// accent pill with `cursor: pointer` (three spans that looked pressable and
// did nothing). chat.css loads after dashboard.css, so it won.
//
// Two guards, both over the real stylesheets and the real components:
//   1. no class a Concurrency* component emits is selected by chat.css;
//   2. no rule in any stylesheet that selects one of those classes may clip
//      it (`overflow: hidden`, `max-height`) or pin its width.
// The test renders nothing — it reads the class attributes the components
// carry and the selectors the stylesheets declare — so it cannot say the
// page LOOKS right; it says the cause of the screenshot is gone and stays gone.
import { describe, expect, test } from 'bun:test';
import { readdirSync, readFileSync } from 'node:fs';
import { join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const LIB = fileURLToPath(new URL('./', import.meta.url));
const STYLES = resolve(LIB, '..', 'styles');
const COMPONENTS = resolve(LIB, 'components');

/** Every static class name a component's markup carries (`class="a b"` and `class:x`). */
const classesOf = (src) => {
  const out = new Set();
  for (const m of src.matchAll(/class="([^"{}]*)"/g)) for (const c of m[1].split(/\s+/)) if (c) out.add(c);
  for (const m of src.matchAll(/class:([A-Za-z0-9_-]+)/g)) out.add(m[1]);
  return out;
};

/** `[selector, body]` for every rule, comments stripped first (see chart-swatch-css.test.js). */
const rulesOf = (css) =>
  [...css.replace(/\/\*[\s\S]*?\*\//g, '').matchAll(/([^{}]+)\{([^{}]*)\}/g)]
    .map(([, sel, body]) => [sel.trim(), body])
    .filter(([sel]) => !sel.startsWith('@'));

/** The class names a selector list names as simple selectors. */
const classesIn = (selector) => new Set([...selector.matchAll(/\.([A-Za-z0-9_-]+)/g)].map((m) => m[1]));

const CLIPPING = /overflow\s*:\s*hidden|max-height\s*:|(?<![a-z-])width\s*:\s*min\(/;

/** Rules whose selector names one of `classes` and whose body clips or pins it. */
const clippingRules = (css, classes) =>
  rulesOf(css).filter(([sel, body]) => [...classesIn(sel)].some((c) => classes.has(c)) && CLIPPING.test(body));

const concurrencyClasses = new Set();
for (const f of readdirSync(COMPONENTS).filter((f) => /^Concurrency.*\.svelte$/.test(f)))
  for (const c of classesOf(readFileSync(join(COMPONENTS, f), 'utf8'))) concurrencyClasses.add(c);

const sheets = Object.fromEntries(
  [...readdirSync(STYLES).filter((f) => f.endsWith('.css')).map((f) => join(STYLES, f)), resolve(LIB, '..', 'app.css')].map(
    (p) => [p.slice(p.lastIndexOf('/') + 1), readFileSync(p, 'utf8')]
  )
);

describe('the concurrency components and chat.css', () => {
  test('the scan found the components and their classes', () => {
    expect(concurrencyClasses.has('cl-table')).toBe(true);
    expect(concurrencyClasses.has('gate-panel')).toBe(true);
    expect(Object.keys(sheets)).toContain('chat.css');
  });

  test('no class a Concurrency* component emits is selected by chat.css', () => {
    const chat = new Set(rulesOf(sheets['chat.css']).flatMap(([sel]) => [...classesIn(sel)]));
    expect([...concurrencyClasses].filter((c) => chat.has(c))).toEqual([]);
  });

  test('no stylesheet clips, caps the height of, or pins the width of a class the concurrency components emit', () => {
    for (const [name, css] of Object.entries(sheets)) {
      // `.cl-tablewrap { overflow-x: auto }` is a scroll, not a clip, and is not matched.
      const bad = clippingRules(css, concurrencyClasses).map(([sel, body]) => `${name}: ${sel} { ${body.trim()} }`);
      expect(bad, name).toEqual([]);
    }
  });
});

describe('the check can fail', () => {
  test('NEGATIVE CONTROL: the exact chat.css rule that clipped the table is detected', () => {
    const shipped = `.cc {
      position: relative; width: min(860px, 100%);
      max-height: calc(100dvh - 5.5rem);
      display: flex; flex-direction: column; overflow: hidden;
    }`;
    expect(clippingRules(shipped, new Set(['cc']))).toHaveLength(1);
    expect(clippingRules(shipped, new Set(['cmp']))).toHaveLength(0);
  });

  test('NEGATIVE CONTROL: a comment naming the class is not a rule, and a scroll is not a clip', () => {
    expect(clippingRules('/* .cmp { overflow: hidden } */ .cmp { display: grid; }', new Set(['cmp']))).toHaveLength(0);
    expect(clippingRules('.cl-tablewrap { overflow-x: auto; }', new Set(['cl-tablewrap']))).toHaveLength(0);
  });

  test('NEGATIVE CONTROL: the component scan reads class attributes, not prose', () => {
    expect([...classesOf('<!-- cc --> <div class="cmp a"><b class:cl-best={x}></b></div>')]).toEqual(['cmp', 'a', 'cl-best']);
  });
});
