// =============================================================================
// chat-streaming.spec.js — streamed answers in the "Ask the codebase" modal:
// the thinking trace streams in first, collapses to the "reasoned for N.Ns"
// disclosure when the answer starts, the answer grows as progressive markdown,
// and a mid-stream fault surfaces without a retry. Shares route/UI plumbing
// with chat.spec.js via fixtures/chat-helpers.js.
// =============================================================================

import { test, expect } from '@playwright/test';
import {
  OR_EMBEDDINGS,
  OR_RERANK,
  OR_CHAT,
  embeddingsHandler,
  rerankHandler,
  sseChatHandler,
  sseMidStreamErrorHandler,
  installPacedChat
} from './fixtures/openrouter.js';
import {
  META,
  routeCorpus,
  openChat,
  statusText,
  waitReady,
  withKey,
  askQuestion
} from './fixtures/chat-helpers.js';

// Reasoning deltas: even indices stream as `delta.reasoning`, odd indices as
// `delta.reasoning_details` (see sseChatFrames) — both shapes must land in
// the trace. The DETAILS-SHAPE marker rides an odd index on purpose.
const REASONING = [
  'The question asks how the verifier keeps draft tokens. ',
  'Context [1] shows the MTP verifier zipping drafts with verified logits. ',
  'The take_while call keeps the longest accepted prefix. ',
  'DETAILS-SHAPE-DELTA arrives through reasoning_details and must land in the trace too. ',
  'I should cite [1] and include the rust line. ',
  'Checking the other context blocks for a better excerpt. ',
  'Nothing closer than [1] in the retrieved set. ',
  'Now writing the final answer.'
];

// Content deltas include a mid-stream XSS probe split across two deltas — the
// joined text forms an onerror attribute and must still print as text.
const CONTENT = [
  'The verifier keeps the longest ',
  'accepted prefix of the draft tokens [1].',
  '\n\n',
  '```rust\n',
  'let kept = draft.iter().zip(verified).take_while(|(d, v)| v.argmax() == **d).count();\n',
  '```\n\n',
  // The attribute name is split across two deltas so escaping is proven on
  // partially-streamed markup. The split point sits after "on" because the
  // repo-wide typos gate reads fixture strings too and flags the trailing
  // fragment of the attribute name as a misspelling when the cut lands one
  // letter later.
  'Probe: <img src=x on',
  'error="window.__xss=1"> must print as text, never run.'
];

async function routeRetrieval(context) {
  await context.route(OR_EMBEDDINGS, embeddingsHandler({ dim: META.dim }));
  await context.route(OR_RERANK, rerankHandler());
}

test('the thinking trace streams, collapses to the disclosure, and the answer streams below', async ({
  page,
  context
}) => {
  await routeCorpus(context);
  await routeRetrieval(context);
  await withKey(page);
  await installPacedChat(page, { reasoning: REASONING, content: CONTENT, delayMs: 200 });
  await page.goto('/');
  await openChat(page);
  await waitReady(page);

  await askQuestion(page, 'how does the verifier keep draft tokens?');

  // Thinking first: the live card comes up in its thinking state with the
  // `reasoning` slabel and an open trace.
  const card = page.locator('.cm-card');
  await expect(card).toHaveAttribute('data-streaming', 'thinking');
  await expect(card.locator('.cm-think-label')).toHaveText('reasoning');
  await expect(card.locator('.cm-think')).toHaveAttribute('data-open', 'true');
  await expect(statusText(page)).toHaveText('reasoning');

  // The trace streams: its text grows while reasoning deltas arrive.
  const traceText = card.locator('.cm-think-text');
  const grew = (await traceText.textContent()).length;
  await expect
    .poll(async () => (await traceText.textContent()).length, { timeout: 15_000 })
    .toBeGreaterThan(grew);

  // First answer token: the trace collapses to the one-line disclosure and
  // the pill flips to writing.
  await expect(card).toHaveAttribute('data-streaming', 'writing', { timeout: 15_000 });
  await expect(card.locator('.cm-think')).toHaveAttribute('data-open', 'false');
  const toggle = card.locator('.cm-think-toggle');
  await expect(toggle).toContainText(/reasoned for \d+\.\ds/);
  await expect(toggle).toContainText('show');
  await expect(statusText(page)).toHaveText('writing');

  // The answer grows progressively as markdown.
  const body = card.locator('.cm-body');
  const bodyLen = (await body.textContent()).length;
  await expect
    .poll(async () => (await body.textContent()).length, { timeout: 15_000 })
    .toBeGreaterThan(bodyLen);

  // Completion: the printed message carries the full markdown answer, the
  // persistent disclosure, and the sources footer.
  await expect(card.locator('.cm-src')).toHaveCount(3, { timeout: 15_000 });
  await expect.poll(() => card.getAttribute('data-streaming')).toBeNull();
  await expect(card.locator('sup.cc-cite').first()).toHaveText('[1]');
  await expect(card.locator('pre.cc-fence code.language-rust')).toContainText('take_while');

  // The split XSS probe arrived across two deltas and still prints as text.
  await expect(body).toContainText('<img src=x onerror=');
  expect(await body.locator('img').count()).toBe(0);
  expect(await page.evaluate(() => window.__xss)).toBeUndefined();

  // Expanding the disclosure brings back the full trace, both delta shapes
  // included, and the toggle now offers hide.
  await toggle.click();
  await expect(card.locator('.cm-think')).toHaveAttribute('data-open', 'true');
  await expect(traceText).toContainText('The question asks how the verifier');
  await expect(traceText).toContainText('DETAILS-SHAPE-DELTA');
  await expect(traceText).toContainText('Now writing the final answer.');
  await expect(toggle).toContainText('hide');
});

test('a one-shot SSE response settles into the full answer with the reasoning disclosure', async ({
  page,
  context
}) => {
  await routeCorpus(context);
  await routeRetrieval(context);
  await context.route(OR_CHAT, sseChatHandler(REASONING, CONTENT));
  await withKey(page);
  await page.goto('/');
  await openChat(page);
  await waitReady(page);

  await askQuestion(page, 'how does the verifier keep draft tokens?');

  const card = page.locator('.cm-card');
  await expect(card.locator('.cm-src')).toHaveCount(3, { timeout: 15_000 });
  await expect.poll(() => card.getAttribute('data-streaming')).toBeNull();

  // [DONE] respected, full answer assembled from the content deltas.
  await expect(card.locator('.cm-body')).toContainText(
    'The verifier keeps the longest accepted prefix of the draft tokens'
  );
  await expect(card.locator('pre.cc-fence code.language-rust')).toContainText('take_while');
  await expect(card.locator('.cm-body')).toContainText('<img src=x onerror=');
  expect(await page.evaluate(() => window.__xss)).toBeUndefined();

  // The reasoning survives as the collapsed disclosure and expands on click.
  const toggle = card.locator('.cm-think-toggle');
  await expect(toggle).toContainText(/reasoned for \d+\.\ds/);
  await expect(card.locator('.cm-think')).toHaveAttribute('data-open', 'false');
  await toggle.click();
  await expect(card.locator('.cm-think-text')).toContainText('DETAILS-SHAPE-DELTA');
});

test('a mid-stream error surfaces the rate card without any retry', async ({ page, context }) => {
  await routeCorpus(context);
  await routeRetrieval(context);
  const attempts = [];
  await context.route(OR_CHAT, sseMidStreamErrorHandler({ log: attempts }));
  await withKey(page);
  await page.goto('/');
  await openChat(page);
  await waitReady(page);
  // Backoff shrunk to ~0: if the engine wrongly retried after first byte, all
  // three attempts would land before the card shows and the count would say so.
  await page.evaluate(() => window.__atlasChatSetRetryBaseMs(1));

  await askQuestion(page, 'how do NVFP4 kernels dispatch?');
  const card = page.locator('.cc-error[role="alert"]');
  await expect(card).toBeVisible({ timeout: 20_000 });
  await expect(card.locator('.cc-error-tag')).toHaveText('rate limited');
  expect(attempts.length).toBe(1); // first byte was emitted — no retry allowed

  // The half-streamed card is gone; only the prompt line remains in the log.
  expect(await page.locator('.cm-card').count()).toBe(0);
  // A chat-time fault must not knock the corpus out of ready.
  await expect(statusText(page)).toContainText('ready ·');
});
