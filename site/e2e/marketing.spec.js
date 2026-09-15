// SPDX-License-Identifier: AGPL-3.0-only
import { test, expect } from '@playwright/test';

test('homepage possibilities work by pointer and keyboard', async ({ page }) => {
  await page.goto('/');
  const privateTab = page.getByRole('tab', { name: 'Private intelligence' });
  await privateTab.click();
  await expect(privateTab).toHaveAttribute('aria-selected', 'true');
  await expect(page.getByRole('heading', { name: 'Keep your intelligence close.' })).toBeVisible();
  await privateTab.press('ArrowRight');
  const research = page.getByRole('tab', { name: 'Unrestricted curiosity' });
  await expect(research).toBeFocused();
  await expect(page.getByRole('heading', { name: 'Make space for the next discovery.' })).toBeVisible();
  await research.press('Home');
  await expect(page.getByRole('tab', { name: 'Agentic experiences' })).toBeFocused();
  await expect(page.getByRole('heading', { name: 'Give your agents room to think.' })).toBeVisible();
});

test('developers can reach the complete engine and return home', async ({ page }) => {
  await page.goto('/');
  await page.getByRole('link', { name: 'Developers', exact: true }).click();
  await expect(page).toHaveURL(/\/engine\.html$/);
  await expect(page.locator('#verified')).toBeVisible();
  await page.getByRole('link', { name: 'Atlas home', exact: true }).first().click();
  await expect(page.getByRole('heading', { name: 'Intelligence, on your terms.' })).toBeVisible();
});

test('technical bookmarks keep their section while homepage summaries stay on home', async ({ page }) => {
  await page.goto('/?ref=bookmark');
  await page.evaluate(() => { window.location.hash = 'faq'; });
  await expect(page).toHaveURL(/\/engine\.html\?ref=bookmark#faq$/);
  await expect(page.locator('#faq')).toBeInViewport();
  await page.goto('/#verified');
  await expect(page).toHaveURL(/\/#verified$/);
  await expect(page.getByRole('heading', { name: 'Confidence, built right in.' })).toBeVisible();
});
