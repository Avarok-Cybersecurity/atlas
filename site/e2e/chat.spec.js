// =============================================================================
// chat.spec.js — the "Ask the codebase" E2E suite (no network beyond
// localhost: corpus + OpenRouter are served from checked-in fixtures via
// Playwright routes; the wasm engine is the real vendored build).
// Runs in both projects: chromium (desktop) and mobile (390x844 sheet).
// =============================================================================

import { test, expect } from '@playwright/test';
import { CORPUS_GZ_URL, CORPUS_META_URL, LS_OPENROUTER_KEY } from '../src/lib/chat/config.js';
import {
  OR_EMBEDDINGS,
  OR_RERANK,
  OR_CHAT,
  CORS_HEADERS,
  embeddingsHandler,
  rerankHandler,
  chatHandler,
  http429Handler,
  ok200ErrorBodyHandler
} from './fixtures/openrouter.js';
import {
  META,
  GZ,
  COMMIT,
  READY_LINE,
  TEST_KEY,
  JSON_HEADERS,
  GZ_HEADERS,
  routeCorpus,
  isMobile,
  openChat,
  statusText,
  waitReady,
  withKey,
  askQuestion
} from './fixtures/chat-helpers.js';
import { startSlowServer } from './fixtures/slow-server.mjs';

// --- route helpers -----------------------------------------------------------

/** Redirect the corpus URL to a local server that streams the gz slowly. */
async function routeSlowCorpus(context, hits, opts) {
  const slow = await startSlowServer(GZ, opts);
  await context.unroute(CORPUS_GZ_URL);
  await context.route(CORPUS_GZ_URL, async (route) => {
    hits.gz++;
    await route.fulfill({ status: 302, headers: { ...CORS_HEADERS, location: slow.url } });
  });
  return slow;
}

async function restoreFastCorpus(context, hits) {
  await context.unroute(CORPUS_GZ_URL);
  await context.route(CORPUS_GZ_URL, async (route) => {
    hits.gz++;
    await route.fulfill({ status: 200, headers: GZ_HEADERS, body: GZ });
  });
}

/** Mock the full OpenRouter pipeline for a happy-path answer. */
async function routeOpenRouter(context, answer, { embedDim = META.dim } = {}) {
  await context.route(OR_EMBEDDINGS, embeddingsHandler({ dim: embedDim }));
  await context.route(OR_RERANK, rerankHandler());
  await context.route(OR_CHAT, chatHandler(answer));
}

// --- UI helpers --------------------------------------------------------------
// (shared drivers live in fixtures/chat-helpers.js)

const opfsFiles = (page) =>
  page.evaluate(async () => {
    const dir = await navigator.storage.getDirectory();
    const names = [];
    for await (const [name] of dir) names.push(name);
    return names.sort();
  });

// =============================================================================
// nav presence / aria
// =============================================================================

test.describe('nav trigger', () => {
  test('is present with dialog aria wiring', async ({ page }) => {
    await page.goto('/');
    if (isMobile(page)) {
      await expect(page.locator('.nav-links .nav-chat-btn')).toBeHidden();
      const toggle = page.locator('.nav-toggle');
      await expect(toggle).toHaveAttribute('aria-expanded', 'false');
      await toggle.click();
      await expect(toggle).toHaveAttribute('aria-expanded', 'true');
      const btn = page.locator('#nav-drawer .nav-chat-btn');
      await expect(btn).toBeVisible();
      await expect(btn).toHaveAttribute('aria-label', 'Ask the codebase');
      await expect(btn).toHaveAttribute('aria-haspopup', 'dialog');
    } else {
      const btn = page.locator('.nav-links .nav-chat-btn');
      await expect(btn).toBeVisible();
      await expect(btn).toHaveAttribute('aria-label', 'Ask the codebase');
      await expect(btn).toHaveAttribute('aria-haspopup', 'dialog');
      await expect(page.locator('.nav-toggle')).toBeHidden();
    }
  });
});

// =============================================================================
// modal shell: open/close/escape/backdrop/focus/scroll lock
// =============================================================================

test.describe('modal shell', () => {
  test('opens with dialog aria, locks scroll, Escape closes and returns focus', async ({
    page,
    context
  }) => {
    await routeCorpus(context);
    await page.goto('/');
    const dialog = await openChat(page);

    await expect(dialog).toHaveAttribute('aria-modal', 'true');
    await expect(dialog).toHaveAttribute('aria-label', 'Ask the codebase');
    await expect
      .poll(() => page.evaluate(() => document.body.style.overflow))
      .toBe('hidden');

    await page.keyboard.press('Escape');
    await expect(dialog).toBeHidden();
    await expect
      .poll(() => page.evaluate(() => document.body.style.overflow))
      .toBe('');
    // Focus returns to the opener — except on the phone, where the opener sits
    // in the (now hidden) drawer, so it lands on the drawer toggle instead.
    const focusClass = isMobile(page) ? 'nav-toggle' : 'nav-chat-btn';
    await expect
      .poll(() =>
        page.evaluate(
          (cls) => document.activeElement?.classList?.contains(cls) ?? false,
          focusClass
        )
      )
      .toBe(true);
  });

  test('close button and backdrop click both close', async ({ page, context }) => {
    await routeCorpus(context);
    await page.goto('/');
    let dialog = await openChat(page);
    await dialog.locator('.cc-close').click();
    await expect(dialog).toBeHidden();

    if (!isMobile(page)) {
      // Backdrop is only exposed around the dialog on desktop (the mobile
      // sheet is full-bleed).
      dialog = await openChat(page);
      await page.locator('.cc-backdrop').click({ position: { x: 8, y: 8 } });
      await expect(dialog).toBeHidden();
    }
  });
});

// =============================================================================
// lazy corpus: nothing downloads before the modal opens
// =============================================================================

test('no corpus or manifest request before the modal opens', async ({ page, context }) => {
  const hits = await routeCorpus(context);
  await page.goto('/');
  if (!isMobile(page)) {
    // Hovering warms the lazy chunk + wasm, which must NOT touch the corpus.
    await page.locator('.nav-links .nav-chat-btn').hover();
  }
  await page.waitForTimeout(3000);
  expect(hits.meta).toBe(0);
  expect(hits.gz).toBe(0);

  await openChat(page);
  await expect.poll(() => hits.meta, { timeout: 15_000 }).toBeGreaterThan(0);
  await expect.poll(() => hits.gz, { timeout: 15_000 }).toBeGreaterThan(0);
});

// =============================================================================
// init walk: manifest -> downloading (MB progress) -> indexing -> ready
// =============================================================================

test('first open walks download with MB progress to ready with fixture stats', async ({
  page,
  context
}) => {
  const hits = await routeCorpus(context);
  const slow = await routeSlowCorpus(context, hits, { chunkSize: 512, delayMs: 60 });
  try {
    await page.goto('/');
    // Record every status-line change so sub-second phases are still assertable.
    await page.evaluate(() => {
      window.__statusLog = [];
      const observer = new MutationObserver(() => {
        const el = document.querySelector('.cc-status-text');
        if (!el) return;
        const t = el.textContent;
        const log = window.__statusLog;
        if (log[log.length - 1] !== t) log.push(t);
      });
      observer.observe(document.body, { subtree: true, childList: true, characterData: true });
    });

    await openChat(page);
    // The loader panel is up with the determinate bar while downloading.
    await expect(page.locator('.cc-load')).toBeVisible();
    await expect(page.locator('.cc-bar[role="progressbar"]')).toBeVisible();
    await waitReady(page);

    const log = await page.evaluate(() => window.__statusLog);
    expect(log.some((t) => /downloading corpus \d+\.\d of \d+\.\d MB/.test(t))).toBe(true);
    expect(log.some((t) => /^indexing|^writing local cache/.test(t))).toBe(true);
    await expect(statusText(page)).toHaveText(READY_LINE);
  } finally {
    await slow.close();
  }
});

// =============================================================================
// OPFS cache: second open = manifest only; stale corpora are pruned
// =============================================================================

test('second open serves the corpus from OPFS with only a manifest request', async ({
  page,
  context
}) => {
  const hits = await routeCorpus(context);
  await page.goto('/');
  await openChat(page);
  await waitReady(page);
  expect(hits).toEqual({ meta: 1, gz: 1 });
  await expect.poll(() => opfsFiles(page)).toContain(`lattice-db-${COMMIT}.jsonl`);

  // Plant a stale corpus; the next load must prune it.
  await page.evaluate(async () => {
    const dir = await navigator.storage.getDirectory();
    const handle = await dir.getFileHandle('lattice-db-00stale00.jsonl', { create: true });
    const w = await handle.createWritable();
    await w.write('stale corpus');
    await w.close();
  });

  await page.reload();
  await openChat(page);
  await waitReady(page);
  await expect(statusText(page)).toHaveText(READY_LINE);
  expect(hits).toEqual({ meta: 2, gz: 1 }); // corpus came from OPFS, not the network
  expect(await opfsFiles(page)).toEqual([`lattice-db-${COMMIT}.jsonl`]);
});

// =============================================================================
// offline manifest -> cached fallback badge
// =============================================================================

test('manifest failure falls back to the cached corpus with the offline badge', async ({
  page,
  context
}) => {
  const hits = await routeCorpus(context);
  await page.goto('/');
  await openChat(page);
  await waitReady(page);

  await context.unroute(CORPUS_META_URL);
  await context.route(CORPUS_META_URL, (route) => route.abort('failed'));

  await page.reload();
  await openChat(page);
  await waitReady(page);
  await expect(statusText(page)).toContainText('cached · offline');
  await expect(page.locator('.cc-log .cc-offline')).toBeVisible();
  expect(hits.gz).toBe(1); // never re-downloaded
});

// =============================================================================
// abort on close mid-download: no partial cache, clean re-run
// =============================================================================

test('closing mid-download aborts cleanly and leaves no partial corpus', async ({
  page,
  context
}) => {
  const hits = await routeCorpus(context);
  const slow = await routeSlowCorpus(context, hits, { chunkSize: 256, delayMs: 100 });
  try {
    await page.goto('/');
    const dialog = await openChat(page);
    await expect(statusText(page)).toContainText('downloading corpus');
    await page.keyboard.press('Escape');
    await expect(dialog).toBeHidden();

    // Nothing partial may remain in OPFS after the abort settles.
    await expect.poll(() => opfsFiles(page)).toEqual([]);

    // Reopening goes back through manifest + full download and succeeds.
    await restoreFastCorpus(context, hits);
    await openChat(page);
    await waitReady(page);
    await expect(statusText(page)).toHaveText(READY_LINE);
    expect(hits.meta).toBe(2);
    expect(hits.gz).toBe(2);
  } finally {
    await slow.close();
  }
});

// =============================================================================
// key gate
// =============================================================================

test('asking is gated on an OpenRouter key that persists across visits', async ({
  page,
  context
}) => {
  await routeCorpus(context);
  await page.goto('/');
  await openChat(page);
  await waitReady(page);

  // Ready but keyless: the question input works, the ask button does not.
  await expect(page.locator('.cc-input')).toBeEnabled();
  await page.locator('.cc-input').fill('what schedules decode batches?');
  await expect(page.locator('.cc-ask')).toBeDisabled();
  await expect(page.locator('.cc-hint')).toContainText('connect your OpenRouter key');

  const keyInput = page.locator('.cc-key-input');
  await expect(keyInput).toHaveAttribute('type', 'password');
  await page.locator('.cc-key-toggle').click();
  await expect(keyInput).toHaveAttribute('type', 'text');
  await keyInput.fill(TEST_KEY);
  await page.locator('.cc-key-save').click();

  await expect(page.locator('.cc-connected')).toBeVisible();
  await expect(page.locator('.cc-ask')).toBeEnabled();
  expect(await page.evaluate((k) => localStorage.getItem(k), LS_OPENROUTER_KEY)).toBe(TEST_KEY);

  // The key survives a reload (localStorage persistence).
  await page.reload();
  await openChat(page);
  await waitReady(page);
  await expect(page.locator('.cc-connected')).toBeVisible();
  await expect(page.locator('.cc-key-input')).toBeHidden();
});

// =============================================================================
// mocked chat round-trip
// =============================================================================

const ANSWER = [
  'The verifier keeps the longest accepted prefix of the draft tokens [1][2].',
  '',
  '```rust',
  'let kept = draft.iter().zip(verified).take_while(|(d, v)| v.argmax() == **d).count();',
  '```',
  '',
  'Probe: <img src=x onerror="window.__xss=1"> must print as text, never run.'
].join('\n');

test('mocked round-trip prints prompt, receipt, markdown, and real source links', async ({
  page,
  context
}) => {
  await routeCorpus(context);
  await routeOpenRouter(context, ANSWER);
  await withKey(page);
  await page.goto('/');
  await openChat(page);
  await waitReady(page);

  // Ask via the first starter chip (also proves the chips submit).
  const starter = await page.locator('.cc-chip').first().textContent();
  await page.locator('.cc-chip').first().click();
  await expect(page.locator('.cm-user')).toContainText(starter.trim());

  const card = page.locator('.cm-card');
  await expect(card).toBeVisible({ timeout: 15_000 });

  // Markdown: citations as <sup>, the fence as <pre><code>.
  await expect(card.locator('sup.cc-cite')).toHaveText(['[1]', '[2]']);
  await expect(card.locator('pre.cc-fence code.language-rust')).toContainText('take_while');

  // This mock answers with a plain JSON completion (not SSE) — the non-stream
  // fallback path. It carries no reasoning, so no thinking trace is printed.
  expect(await card.locator('.cm-think').count()).toBe(0);

  // XSS probe arrived escaped: rendered as text, no element, no execution.
  await expect(card.locator('.cm-body')).toContainText('<img src=x onerror=');
  expect(await card.locator('.cm-body img').count()).toBe(0);
  expect(await page.evaluate(() => window.__xss)).toBeUndefined();

  // Source receipts: path, line range, and a blob link pinned to the corpus commit.
  const sources = card.locator('.cm-src');
  await expect(sources).toHaveCount(3);
  const hrefPattern = new RegExp(
    `^https://github\\.com/Avarok-Cybersecurity/atlas/blob/${COMMIT}/.+#L\\d+-L\\d+$`
  );
  for (const src of await sources.all()) {
    expect(await src.getAttribute('href')).toMatch(hrefPattern);
    await expect(src.locator('.cm-src-path')).not.toBeEmpty();
    await expect(src.locator('.cm-src-lines')).toHaveText(/^L\d+–\d+$/);
    await expect(src.locator('.cm-src-pct')).toHaveText(/^\d+%$/);
  }
  // Retrieval is deterministic (fixture vectors + fixture embedder): the MTP
  // question must surface the MTP chunks.
  const paths = await card.locator('.cm-src-path').allTextContents();
  expect(paths.some((p) => p.includes('mtp.rs'))).toBe(true);

  // Sources ship open on desktop and collapsed on the phone sheet.
  const details = card.locator('.cm-sources');
  if (isMobile(page)) {
    await expect(details).not.toHaveAttribute('open', '');
  } else {
    await expect(details).toHaveAttribute('open', '');
  }
});

// =============================================================================
// error states
// =============================================================================

test.describe('error states', () => {
  test('corpus 404 shows the download-failed card and retry recovers', async ({
    page,
    context
  }) => {
    const hits = { meta: 0, gz: 0 };
    await context.route(CORPUS_META_URL, (route) =>
      route.fulfill({ status: 200, headers: JSON_HEADERS, body: JSON.stringify(META) })
    );
    await context.route(CORPUS_GZ_URL, (route) =>
      route.fulfill({ status: 404, headers: JSON_HEADERS, body: 'not found' })
    );
    await page.goto('/');
    await openChat(page);

    const card = page.locator('.cc-error[role="alert"]');
    await expect(card).toBeVisible({ timeout: 20_000 });
    await expect(card.locator('.cc-error-tag')).toHaveText('download failed');

    await restoreFastCorpus(context, hits);
    await card.locator('.cc-error-retry').click();
    await waitReady(page);
    await expect(statusText(page)).toHaveText(READY_LINE);
  });

  test('embedding 429s exhaust retries into the rate card', async ({ page, context }) => {
    await routeCorpus(context);
    const attempts = [];
    await context.route(OR_EMBEDDINGS, http429Handler({ log: attempts }));
    await withKey(page);
    await page.goto('/');
    await openChat(page);
    await waitReady(page);
    await page.evaluate(() => window.__atlasChatSetRetryBaseMs(1));

    await askQuestion(page, 'what schedules decode batches?');
    const card = page.locator('.cc-error[role="alert"]');
    await expect(card).toBeVisible({ timeout: 20_000 });
    await expect(card.locator('.cc-error-tag')).toHaveText('rate limited');
    expect(attempts.length).toBe(3); // OR_MAX_ATTEMPTS, all POSTs
  });

  test('chat 200-with-error-body surfaces the rate card', async ({ page, context }) => {
    await routeCorpus(context);
    const attempts = [];
    await context.route(OR_EMBEDDINGS, embeddingsHandler({ dim: META.dim }));
    await context.route(OR_RERANK, rerankHandler());
    await context.route(OR_CHAT, ok200ErrorBodyHandler({ log: attempts }));
    await withKey(page);
    await page.goto('/');
    await openChat(page);
    await waitReady(page);
    await page.evaluate(() => window.__atlasChatSetRetryBaseMs(1));

    await askQuestion(page, 'how do NVFP4 kernels dispatch?');
    const card = page.locator('.cc-error[role="alert"]');
    await expect(card).toBeVisible({ timeout: 20_000 });
    await expect(card.locator('.cc-error-tag')).toHaveText('rate limited');
    expect(attempts.length).toBe(3); // 200-with-error-body is retried as transient
  });

  test('embedding dimension mismatch prints a legible sync error', async ({ page, context }) => {
    await routeCorpus(context);
    await context.route(OR_EMBEDDINGS, embeddingsHandler({ dim: 16 }));
    await withKey(page);
    await page.goto('/');
    await openChat(page);
    await waitReady(page);

    await askQuestion(page, 'where is the kv pool?');
    const card = page.locator('.cc-error[role="alert"]');
    await expect(card).toBeVisible({ timeout: 20_000 });
    await expect(card.locator('.cc-error-body')).toContainText(
      `returned 16 dimensions but this corpus was built with ${META.dim}`
    );
    // A chat-time fault must not knock the corpus out of ready.
    await expect(statusText(page)).toContainText('ready ·');
  });

  // The rerank API returns positions into the documents WE sent. rag.js used
  // them to index `candidates` directly, so an index the response invented
  // produced `undefined` in `picked` and the next line read `.payload` off it —
  // a bare TypeError that took the whole answer down instead of degrading.
  test('a rerank index that points nowhere does not take the answer down', async ({
    page,
    context
  }) => {
    await routeCorpus(context);
    await context.route(OR_EMBEDDINGS, embeddingsHandler({ dim: META.dim }));
    await context.route(OR_CHAT, chatHandler('The KV pool lives in `kv_pool.rs` [1].'));
    await context.route(OR_RERANK, async (route) => {
      const body = route.request().postDataJSON();
      // One real position and two that do not exist.
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          // Scores matter: rag.js takes .slice(0, TOP_K) of the RANKED list,
          // so a bad index only reaches the guard if it scores into the top
          // few. The first version of this test put them last, and passed
          // against a guard that did not hold.
          results: [
            { index: 'length', relevance_score: 0.99 },
            { index: 'map', relevance_score: 0.98 },
            { index: 0, relevance_score: 0.97 },
            { index: body.documents.length + 5, relevance_score: 0.8 },
            { index: -1, relevance_score: 0.7 },
            { index: 1.5, relevance_score: 0.4 },
            { index: null, relevance_score: 0.3 }
          ]
        })
      });
    });
    await withKey(page);
    await page.goto('/');
    await openChat(page);
    await waitReady(page);

    await askQuestion(page, 'where is the kv pool?');
    // The answer still arrives, built from the position that did resolve.
    await expect(page.locator('.cm-body').first()).toBeVisible({ timeout: 20_000 });
    await expect(statusText(page)).toContainText('ready ·');
  });
});

// =============================================================================
// mobile sheet behavior
// =============================================================================

test('the modal is a full-bleed sheet on a phone', async ({ page, context }) => {
  test.skip(!isMobile(page), 'mobile project only');
  await routeCorpus(context);
  await page.goto('/');
  const dialog = await openChat(page);

  const viewport = page.viewportSize();
  const box = await dialog.boundingBox();
  expect(box.width).toBe(viewport.width);
  expect(box.height).toBe(viewport.height);
  expect(box.x).toBe(0);
  expect(box.y).toBe(0);
  const styles = await dialog.evaluate((el) => {
    const s = getComputedStyle(el);
    return { radius: s.borderRadius, inputRow: getComputedStyle(el.querySelector('.cc-input-row')).position };
  });
  expect(styles.radius).toBe('0px');
  expect(styles.inputRow).toBe('sticky'); // input stays pinned while the log scrolls
});
