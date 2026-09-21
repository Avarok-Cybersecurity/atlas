// =============================================================================
// chat-helpers.js — shared plumbing for the chat E2E specs (chat.spec.js and
// chat-streaming.spec.js): fixture corpus constants, corpus routing, and the
// open-modal / wait-ready / ask UI drivers. Extracted so both specs code to
// one copy (SSOT) and neither spec file balloons.
// =============================================================================

import { expect } from '@playwright/test';
import { readFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { CORPUS_GZ_URL, CORPUS_META_URL, LS_OPENROUTER_KEY } from '../../src/lib/chat/config.js';
import { CORS_HEADERS } from './openrouter.js';

const FIX = dirname(fileURLToPath(import.meta.url));
export const META = JSON.parse(readFileSync(join(FIX, 'corpus-small.meta.json'), 'utf8'));
export const GZ = readFileSync(join(FIX, 'corpus-small.jsonl.gz'));
export const COMMIT = META.commit_sha;
export const SHORT = COMMIT.slice(0, 7);
export const READY_LINE = `ready · ${SHORT} · ${META.points} chunks · dim ${META.dim}`;
export const TEST_KEY = 'sk-or-v1-e2e-fixture-key';

export const JSON_HEADERS = { ...CORS_HEADERS, 'content-type': 'application/json' };
export const GZ_HEADERS = { ...CORS_HEADERS, 'content-type': 'application/gzip' };

/** Serve meta + corpus from fixtures, counting hits per URL. */
export async function routeCorpus(context) {
  const hits = { meta: 0, gz: 0 };
  await context.route(CORPUS_META_URL, async (route) => {
    hits.meta++;
    await route.fulfill({ status: 200, headers: JSON_HEADERS, body: JSON.stringify(META) });
  });
  await context.route(CORPUS_GZ_URL, async (route) => {
    hits.gz++;
    await route.fulfill({ status: 200, headers: GZ_HEADERS, body: GZ });
  });
  return hits;
}

// --- UI drivers --------------------------------------------------------------

export const isMobile = (page) => page.viewportSize().width <= 860;

export async function clickChatTrigger(page) {
  if (isMobile(page)) {
    await page.locator('.nav-toggle').click();
    await page.locator('#nav-drawer .nav-chat-btn').click();
  } else {
    await page.locator('.nav-links .nav-chat-btn').click();
  }
}

/** Open the modal and wait for the real dialog (past the lazy-load skeleton). */
export async function openChat(page) {
  await clickChatTrigger(page);
  const dialog = page.locator('.cc[role="dialog"]:not(.cc-skeleton)');
  await expect(dialog).toBeVisible();
  return dialog;
}

export const statusText = (page) => page.locator('.cc-status-text');

export async function waitReady(page) {
  await expect(statusText(page)).toContainText('ready ·', { timeout: 30_000 });
}

export function withKey(page) {
  return page.addInitScript(
    ([k, v]) => localStorage.setItem(k, v),
    [LS_OPENROUTER_KEY, TEST_KEY]
  );
}

export async function askQuestion(page, question) {
  await page.locator('.cc-input').fill(question);
  await page.locator('.cc-ask').click();
}
