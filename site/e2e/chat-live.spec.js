// =============================================================================
// chat-live.spec.js — @live tests that touch the real published corpus and
// (optionally) real OpenRouter. Excluded by default (grepInvert in the
// config); run with `bun run test:e2e:live`. The OpenRouter leg needs
// OPENROUTER_API_KEY in the environment and skips cleanly without it —
// the key value is never printed or persisted.
// =============================================================================

import { test, expect } from '@playwright/test';

const LIVE_KEY = process.env.OPENROUTER_API_KEY;

test.describe('@live real corpus', () => {
  test('the published corpus downloads, indexes, and reaches ready', async ({ page }) => {
    test.setTimeout(300_000); // real download + wasm indexing on real hardware
    await page.goto('/');
    if (page.viewportSize().width <= 860) await page.locator('.nav-toggle').click();
    await page.locator('.nav-chat-btn:visible').first().click();
    await expect(page.locator('.cc[role="dialog"]:not(.cc-skeleton)')).toBeVisible();
    const status = page.locator('.cc-status-text');
    await expect(status).toContainText('ready ·', { timeout: 280_000 });
    // Real stats: a commit short-sha, non-zero chunks, and a real dim.
    await expect(status).toHaveText(
      /^ready · [0-9a-f]{7} · [1-9]\d* chunks · dim [1-9]\d*$/
    );
  });

  test('a real question comes back cited when a key is provided', async ({ page }) => {
    test.skip(!LIVE_KEY, 'OPENROUTER_API_KEY not set — skipping the real-key leg');
    test.setTimeout(300_000);
    await page.addInitScript((k) => localStorage.setItem('atlas-openrouter-key', k), LIVE_KEY);
    await page.goto('/');
    if (page.viewportSize().width <= 860) await page.locator('.nav-toggle').click();
    await page.locator('.nav-chat-btn:visible').first().click();
    await expect(page.locator('.cc[role="dialog"]:not(.cc-skeleton)')).toBeVisible();
    await expect(page.locator('.cc-status-text')).toContainText('ready ·', { timeout: 280_000 });

    await page.locator('.cc-input').fill('Where does the engine decide which requests join a decode batch?');
    await page.locator('.cc-ask').click();
    const card = page.locator('.cm-card');
    await expect(card).toBeVisible({ timeout: 120_000 });
    // The card appears with the thinking trace alone; .cm-body only exists once
    // answer tokens start, and a reasoning model can think for tens of seconds.
    await expect(card.locator('.cm-body')).not.toBeEmpty({ timeout: 180_000 });
    // Free-tier models can rate-limit; sources are the part the site controls.
    await expect(card.locator('.cm-src').first()).toBeVisible({ timeout: 120_000 });
    expect(await card.locator('.cm-src').first().getAttribute('href')).toMatch(
      /^https:\/\/github\.com\/Avarok-Cybersecurity\/atlas\/blob\/[0-9a-f]{40}\/.+#L\d+-L\d+$/
    );
  });
});
