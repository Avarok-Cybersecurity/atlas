// =============================================================================
// site-guide.spec.js — SITE-GUIDE.md matches the site that was just built
// -----------------------------------------------------------------------------
// The guide lists every page, every button and where it goes. It is generated
// from build/, which this suite has just produced, so this is the one place the
// check can run against the real thing. No browser is needed: it runs the
// generator in --check mode and fails with what the generator says.
// =============================================================================
import { test, expect } from '@playwright/test';
import { execFileSync } from 'node:child_process';

test('the site guide is current: pages, buttons, links, facts and assets', async ({}, testInfo) => {
  test.skip(testInfo.project.name !== 'chromium', 'one run is enough, it does not use a browser');
  let output = '';
  let code = 0;
  try {
    output = execFileSync('node', ['scripts/gen-site-guide.mjs', '--check'], { encoding: 'utf8', stdio: ['ignore', 'pipe', 'pipe'] });
  } catch (e) {
    code = e.status ?? 1;
    output = `${e.stdout ?? ''}${e.stderr ?? ''}`;
  }
  expect(output.trim() + ` (exit ${code})`).toContain('guide: current');
});
