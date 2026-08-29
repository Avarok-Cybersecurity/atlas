import { test, expect } from 'bun:test';
import { readFileSync } from 'node:fs';

/**
 * `<meta name="theme-color">` paints the browser chrome and the mobile status
 * bar. It cannot read a CSS custom property, so it is the one place the ground
 * colour has to be written out by hand — and therefore the one place it can
 * silently disagree with the page.
 *
 * It did: the palette move swept every stylesheet and missed both app.html
 * files, leaving the chrome the old violet #14111f above a #0F1216 page.
 */
const tokens = readFileSync(new URL('../../../web-shared/atlas-tokens.css', import.meta.url), 'utf8');
const bg = tokens.match(/--bg:\s*(#[0-9a-fA-F]{6})/)?.[1];

test('the token file defines --bg (so the comparisons below are not vacuous)', () => {
  expect(bg).toMatch(/^#[0-9a-fA-F]{6}$/);
});

for (const [label, rel] of [
  ['marketing site', '../../src/app.html'],
  ['blog', '../../../blog/src/app.html']
]) {
  test(`${label}: theme-color equals --bg`, () => {
    const html = readFileSync(new URL(rel, import.meta.url), 'utf8');
    const m = html.match(/<meta\s+name="theme-color"\s+content="(#[0-9a-fA-F]{6})"/);
    expect(m, `${label}: no theme-color meta found`).not.toBeNull();
    expect(m[1].toLowerCase()).toBe(bg.toLowerCase());
  });
}
