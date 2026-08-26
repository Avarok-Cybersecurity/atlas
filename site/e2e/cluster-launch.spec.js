// SPDX-License-Identifier: AGPL-3.0-only

// The cluster-launch flow against a real local agent.
//
// @live because it needs `atlasctl agent run` on this machine with at least one
// paired peer. Without that there is nothing to launch across, and a mocked
// agent would only prove the mock agrees with itself.

import { expect, test } from '@playwright/test';

const TOKEN = process.env.ATLASCTL_TOKEN ?? '';

test.describe('@live cluster launch', () => {
  test.skip(!TOKEN, 'needs ATLASCTL_TOKEN and a running agent');

  test.beforeEach(async ({ page }) => {
    await page.addInitScript((t) => {
      window.localStorage.setItem('atlasctl.token', t);
    }, TOKEN);
  });

  test('previews a two-node launch with a command per machine', async ({ page }) => {
    await page.goto('/control');

    const launch = page.locator('#launch');
    await expect(launch).toBeVisible();

    // The recipe list comes from the agent, so its arrival proves the socket is
    // live rather than that the page rendered.
    const select = launch.locator('select');
    await expect(select).toBeEnabled({ timeout: 20_000 });
    const twoNode = launch.locator('option', { hasText: '2 machines' }).first();
    await expect(twoNode).toBeAttached({ timeout: 20_000 });
    await select.selectOption(await twoNode.getAttribute('value'));

    // Pick every machine the recipe needs.
    const boxes = launch.locator('.lc-node input[type=checkbox]');
    await expect(boxes.first()).toBeVisible();
    const n = await boxes.count();
    for (let i = 0; i < Math.min(n, 2); i++) await boxes.nth(i).check();

    await expect(launch.locator('.lc-blocker')).toHaveCount(0);
    await launch.getByRole('button', { name: /what will run/i }).click();

    // One rendered command per rank, each from the machine that would run it.
    const ranks = launch.locator('.lc-rank');
    await expect(ranks).toHaveCount(2, { timeout: 30_000 });
    await expect(ranks.first().locator('pre')).toContainText('docker run');
    await expect(launch.getByText('serves the API')).toBeVisible();
  });

  test('reserves both machines and then releases them', async ({ page }) => {
    await page.goto('/control');
    const launch = page.locator('#launch');

    const select = launch.locator('select');
    await expect(select).toBeEnabled({ timeout: 20_000 });
    const twoNode = launch.locator('option', { hasText: '2 machines' }).first();
    await expect(twoNode).toBeAttached({ timeout: 20_000 });
    await select.selectOption(await twoNode.getAttribute('value'));

    const boxes = launch.locator('.lc-node input[type=checkbox]');
    const n = await boxes.count();
    for (let i = 0; i < Math.min(n, 2); i++) await boxes.nth(i).check();

    await launch.getByRole('button', { name: /what will run/i }).click();
    await expect(launch.locator('.lc-rank')).toHaveCount(2, { timeout: 30_000 });

    await launch.getByRole('button', { name: /reserve these machines/i }).click();
    const answers = launch.locator('.lc-prepared li');
    await expect(answers).toHaveCount(2, { timeout: 30_000 });
    await expect(launch.locator('.lc-ok')).toHaveCount(2);

    // Nothing may have started: prepare reserves, it does not launch.
    await expect(launch.locator('.lc-running')).toHaveCount(0);

    // And the reservations must be releasable, or the fleet stays stuck.
    await launch.getByRole('button', { name: /release the reservations/i }).click();
    await expect(launch.locator('.lc-prepared')).toHaveCount(0, { timeout: 15_000 });
  });
});
