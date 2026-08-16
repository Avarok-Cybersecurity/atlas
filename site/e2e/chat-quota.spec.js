// =============================================================================
// The daily free-model allowance is not a momentary rate limit. It has a known
// reset, retrying cannot clear it, and the remedy is a different model — so it
// gets its own card, its own (absent) retry policy, and a one-click switch to
// the paid twin. These tests pin all three.
// =============================================================================
import { test, expect } from '@playwright/test';
import {
  OR_EMBEDDINGS,
  OR_RERANK,
  OR_CHAT,
  embeddingsHandler,
  rerankHandler,
  dailyQuota429Handler,
  freeQuotaPaidOkHandler,
  QUOTA_RESET_AT
} from './fixtures/openrouter.js';
import {
  META,
  routeCorpus,
  openChat,
  waitReady,
  withKey,
  askQuestion
} from './fixtures/chat-helpers.js';

const FREE_MODEL = 'nvidia/nemotron-3-ultra-550b-a55b:free';
const PAID_MODEL = 'nvidia/nemotron-3-ultra-550b-a55b';

async function routeRetrieval(context) {
  await context.route(OR_EMBEDDINGS, embeddingsHandler({ dim: META.dim }));
  await context.route(OR_RERANK, rerankHandler());
}

test.describe('@quota daily allowance', () => {
  test('a spent daily allowance shows the quota card with its reset, and never retries', async ({
    page,
    context
  }) => {
    await routeCorpus(context);
    await routeRetrieval(context);
    const attempts = [];
    await context.route(OR_CHAT, dailyQuota429Handler({ log: attempts }));

    await withKey(page);
    await page.goto('/');
    await openChat(page);
    await waitReady(page);
    // Shrink the backoff: if the engine wrongly retried, this test would still
    // finish fast and the attempt count below would catch the mistake.
    await page.evaluate(() => window.__atlasChatSetRetryBaseMs(1));

    await askQuestion(page, 'how does the scheduler batch decode?');

    const card = page.locator('.cc-error[role="alert"]');
    await expect(card).toBeVisible({ timeout: 20_000 });
    await expect(card.locator('.cc-error-tag')).toHaveText('daily limit reached');
    // The momentary-rate copy must never appear here: waiting seconds is a lie.
    await expect(card.locator('.cc-error-body')).not.toContainText('catching their breath');

    const expected = new Date(QUOTA_RESET_AT).toLocaleTimeString([], {
      hour: 'numeric',
      minute: '2-digit'
    });
    await expect(card.locator('.cc-error-reset')).toContainText(expected);

    expect(attempts.length).toBe(1); // no backoff loop on a per-day cap
  });

  test('the paid-model button re-asks on the paid twin and persists the choice', async ({
    page,
    context
  }) => {
    await routeCorpus(context);
    await routeRetrieval(context);
    const calls = [];
    await context.route(
      OR_CHAT,
      freeQuotaPaidOkHandler('The scheduler batches decode in `batch.rs` [1].', { log: calls })
    );

    await withKey(page);
    await page.goto('/');
    await openChat(page);
    await waitReady(page);
    await page.evaluate(() => window.__atlasChatSetRetryBaseMs(1));

    await askQuestion(page, 'how does the scheduler batch decode?');
    const card = page.locator('.cc-error[role="alert"]');
    await expect(card).toBeVisible({ timeout: 20_000 });
    expect(calls[0].model).toBe(FREE_MODEL);

    await card.locator('.cc-error-paid').click();

    // The answer now arrives, and it came from the paid twin.
    await expect(page.locator('.cm-card .cm-body').last()).toContainText('batch.rs', {
      timeout: 20_000
    });
    expect(calls[calls.length - 1].model).toBe(PAID_MODEL);
    await expect(page.locator('.cc-model-id')).toHaveText(PAID_MODEL);

    // The choice survives a reload, and can be handed back to the free default.
    await page.reload();
    await openChat(page);
    await expect(page.locator('.cc-model-id')).toHaveText(PAID_MODEL);
    await page.locator('.cc-model-reset').click();
    await expect(page.locator('.cc-model-id')).toHaveText(FREE_MODEL);
  });
});
