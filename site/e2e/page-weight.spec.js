// =============================================================================
// page-weight.spec.js — what each page makes a browser fetch before it paints
// -----------------------------------------------------------------------------
// The Lighthouse gate demands 100 and serves over HTTP/1.1, six connections at a
// time, so every extra small file is a queued round trip ahead of first paint.
// Twice a change that looked free cost every page a request: forty components
// each becoming a chunk, and one `import` in the root layout leaving a 22 byte
// facade file behind. Neither failed anything. This reads the build the suite
// has just made and holds the line, in numbers, with no browser needed.
// vite.config.js explains the two named chunks these budgets depend on.
// =============================================================================
import { test, expect } from '@playwright/test';
import { readFileSync } from 'node:fs';
import { join } from 'node:path';

const BUILD = 'build';
const preloads = (page) => [...readFileSync(join(BUILD, page), 'utf8').matchAll(/href="[^"]*?(_app\/immutable\/[^"]+\.js)" rel="modulepreload"/g)].map((m) => m[1]);
// A string only the marketing components contain.
const isMarketingChunk = (file) => readFileSync(join(BUILD, file), 'utf8').includes('av-tourstack');

test.describe('page weight', () => {
  test.beforeEach(({}, testInfo) => test.skip(testInfo.project.name !== 'chromium', 'reads files, one run is enough'));

  test('a developer page never loads the marketing components', () => {
    for (const page of ['engine.html', 'control.html', 'diligence.html']) {
      expect(preloads(page).filter(isMarketingChunk), `${page} loads the marketing chunk`).toEqual([]);
      expect(readFileSync(join(BUILD, page), 'utf8').includes('.av-receipt'), `${page} inlines marketing CSS`).toBe(false);
    }
  });

  // Measured when this was written: marketing 8 or 9, /engine 13, /control and
  // /diligence 11. One spare each. Raising a number here is a decision to make
  // every visit to that page slower, so say why in the commit.
  const BUDGET = { 'index.html': 10, 'why-avarok.html': 9, 'pricing.html': 9, 'company.html': 9, 'engine.html': 14, 'control.html': 12, 'diligence.html': 12 };
  for (const [page, max] of Object.entries(BUDGET)) {
    test(`${page} preloads at most ${max} scripts`, () => {
      expect(preloads(page).length).toBeLessThanOrEqual(max);
    });
  }
});
