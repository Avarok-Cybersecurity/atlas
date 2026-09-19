import { test, expect } from 'bun:test';
import { readFileSync } from 'node:fs';

/**
 * The lockup component draws the brand by hand, in markup, next to the code
 * that colours it — which is exactly the shape a redraw takes. It has already
 * happened once: the rebrand replaced the second wordmark path and the viewBox
 * in the component and in four `assets/brand/*.svg` masters at the same time,
 * so nothing in the tree disagreed and nothing failed.
 *
 * So this file pins the component's artwork to the master files it claims to be
 * lifted from. A path edited in one place and not the other is now a failing
 * test rather than a logo that is subtly not the logo.
 */
const read = (p) => readFileSync(new URL(p, import.meta.url), 'utf8');
const lockup = read('../../../web-shared/components/AtlasLockup.svelte');
const full = read('../../../assets/brand/logo-full.svg');
const corp = read('../../../assets/brand/logo-full-corp.svg');
const tokens = read('../../../web-shared/avarok-tokens.css');

/** Every `d="…"` in document order. */
const paths = (svg) => [...svg.matchAll(/\bd="([^"]+)"/g)].map((m) => m[1]);

/** The `d` of the one path whose fill is `hex` (the masters pin colours literally). */
const pathFilledWith = (svg, hex) =>
  svg.match(new RegExp(`<path d="([^"]+)" fill="${hex}"`))?.[1] ??
  svg.match(new RegExp(`<path[^>]*fill="${hex}"[^>]*\\bd="([^"]+)"`))?.[1];

test('the masters are the artwork this test thinks they are (not a vacuous pin)', () => {
  // Three chevrons, two wordmark outlines, one tagline: six paths, or the
  // comparisons below would pass by comparing nothing.
  expect(paths(full)).toHaveLength(6);
  expect(paths(corp)).toHaveLength(6);
  expect(paths(full)[0]).toBe('M38 38L358 318L38 598');
});

test('the mark in the component is the mark in the masters', () => {
  for (const chevron of paths(full).slice(0, 3)) {
    expect(lockup).toContain(`d="${chevron}"`);
  }
});

test('the wordmark is the masters’, in both lockups', () => {
  const [w0, w1] = paths(full).slice(3, 5);
  expect(lockup).toContain(`d="${w0}"`);
  expect(lockup).toContain(`d="${w1}"`);
  // The corporate lockup must carry the SAME wordmark, not a second cut of it.
  expect(paths(corp).slice(3, 5)).toEqual([w0, w1]);
});

test('both taglines come from their master file', () => {
  const engine = paths(full)[5];
  const cybernetics = pathFilledWith(corp, '#BDC0C5');
  expect(cybernetics).toBeTruthy();
  expect(engine).not.toBe(cybernetics);
  expect(lockup).toContain(`id="atlas-tagline" fill="var(--logo-tagline)" d="${engine}"`);
  expect(lockup).toContain(`id="atlas-tagline-corp" fill="var(--logo-tagline)" d="${cybernetics}"`);
});

test('the lockup reads the logo greys, never the text ramp', () => {
  // The guidelines call the light-ground greys logo colours, and the light
  // theme darkens --t2/--t3 to clear WCAG AA on paper. Artwork bound to the
  // text ramp would recolour itself the next time the ramp moved.
  expect(lockup).toContain('id="atlas-word" fill="var(--logo-word)"');
  expect(lockup).not.toMatch(/id="atlas-(word|tagline[^"]*)" fill="var\(--t[123]\)"/);

  const block = (sel) => tokens.match(new RegExp(`${sel}\\s*\\{[\\s\\S]*?\\n\\}`))?.[0] ?? '';
  const dark = block(':root');
  const light = block('\\[data-theme="light"\\]');
  // The four values are the guidelines' own colour table.
  expect(dark).toContain('--logo-word: #C9CCD4;');
  expect(dark).toContain('--logo-tagline: #82868F;');
  expect(light).toContain('--logo-word: #9397A0;');
  expect(light).toContain('--logo-tagline: #BDC0C5;');
  // And they are the greys the two master cuts differ by.
  expect(corp).toContain('fill="#9397A0"');
  expect(read('../../../assets/brand/logo-full-corp-ondark.svg')).toContain('fill="#C9CCD4"');
});

test('the corporate lockup is never rendered below its legibility floor', () => {
  // 220px is where the tagline's x-height reaches 8px; the component may only
  // ever be wider, and the narrow twin is the horizontal cut, not a small corp.
  const style = lockup.match(/<style>[\s\S]*<\/style>/)[0];
  const width = Number(style.match(/\.logo-c \{ width: (\d+)px/)[1]);
  expect(width).toBeGreaterThanOrEqual(220);
  expect(style).toMatch(/\.logo-c-narrow \{ display: none;/);
  expect(lockup).toContain('viewBox="0 0 1230.82 335.8" aria-hidden="true"');
});
