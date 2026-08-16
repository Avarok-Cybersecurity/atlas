// =============================================================================
// openrouter.js — Playwright route-handler factories that stand in for the
// OpenRouter API in the non-@live E2E suite. Every handler answers the CORS
// preflight itself (the page calls api from another origin with Authorization
// + Content-Type headers, so the browser preflights every POST).
// =============================================================================

import { embedText } from './embed.mjs';

export const OR_EMBEDDINGS = 'https://openrouter.ai/api/v1/embeddings';
export const OR_RERANK = 'https://openrouter.ai/api/v1/rerank';
export const OR_CHAT = 'https://openrouter.ai/api/v1/chat/completions';

export const CORS_HEADERS = {
  'access-control-allow-origin': '*',
  'access-control-allow-methods': 'GET, POST, OPTIONS',
  'access-control-allow-headers': 'authorization, content-type, http-referer, x-title'
};

/** Answer the OPTIONS preflight; returns true when the route was consumed. */
async function handlePreflight(route) {
  if (route.request().method() !== 'OPTIONS') return false;
  await route.fulfill({ status: 204, headers: CORS_HEADERS });
  return true;
}

const json = (body, status = 200) => ({
  status,
  headers: { ...CORS_HEADERS, 'content-type': 'application/json' },
  body: JSON.stringify(body)
});

/**
 * /embeddings — deterministic embeddings via the shared fake embedder, same
 * function the corpus generator used, so retrieval ranks semantically.
 * `dim` other than the corpus dim exercises the dim-mismatch guard.
 */
export function embeddingsHandler({ dim = 8, log } = {}) {
  return async (route) => {
    if (await handlePreflight(route)) return;
    log?.push(route.request().postDataJSON());
    const input = route.request().postDataJSON().input;
    const texts = Array.isArray(input) ? input : [input];
    await route.fulfill(
      json({
        object: 'list',
        model: 'nvidia/llama-nemotron-embed-vl-1b-v2',
        data: texts.map((t, index) => ({ object: 'embedding', index, embedding: embedText(String(t), dim) }))
      })
    );
  };
}

/** /rerank — scores documents by shared-word overlap with the query. */
export function rerankHandler({ log } = {}) {
  return async (route) => {
    if (await handlePreflight(route)) return;
    const body = route.request().postDataJSON();
    log?.push(body);
    const queryWords = new Set(String(body.query).toLowerCase().split(/\W+/));
    const results = body.documents
      .map((doc, index) => {
        const words = String(doc).toLowerCase().split(/\W+/);
        const hits = words.filter((w) => w.length > 2 && queryWords.has(w)).length;
        return { index, relevance_score: Math.min(0.99, hits / 8 + 0.01) };
      })
      .sort((a, b) => b.relevance_score - a.relevance_score);
    await route.fulfill(json({ results }));
  };
}

/** /chat/completions — returns `answer` as the assistant message. */
export function chatHandler(answer, { log } = {}) {
  return async (route) => {
    if (await handlePreflight(route)) return;
    log?.push(route.request().postDataJSON());
    await route.fulfill(
      json({
        id: 'gen-e2e',
        choices: [{ index: 0, message: { role: 'assistant', content: answer }, finish_reason: 'stop' }]
      })
    );
  };
}

/**
 * Build the SSE frame list for a streamed chat completion: one frame per
 * reasoning delta, then one per content delta, then `[DONE]`. Reasoning
 * frames alternate between the `delta.reasoning` string shape and the
 * `delta.reasoning_details` array shape so both parser paths stay covered,
 * with a keepalive comment sprinkled in like the real feed.
 */
export function sseChatFrames(reasoningDeltas, contentDeltas) {
  const frames = [': OPENROUTER PROCESSING\n\n'];
  reasoningDeltas.forEach((text, i) => {
    const delta =
      i % 2 === 1
        ? { content: '', reasoning_details: [{ type: 'reasoning.text', text }] }
        : { content: '', reasoning: text };
    frames.push(`data: ${JSON.stringify({ id: 'gen-e2e-sse', choices: [{ index: 0, delta }] })}\n\n`);
  });
  contentDeltas.forEach((text) => {
    frames.push(
      `data: ${JSON.stringify({ id: 'gen-e2e-sse', choices: [{ index: 0, delta: { content: text } }] })}\n\n`
    );
  });
  frames.push('data: [DONE]\n\n');
  return frames;
}

/**
 * /chat/completions — a streamed SSE answer delivered in one fulfill. The
 * page's parser still walks it delta by delta, so this covers the SSE path's
 * end state; use installPacedChat when a test must OBSERVE mid-stream states.
 */
export function sseChatHandler(reasoningDeltas, contentDeltas, { log } = {}) {
  return async (route) => {
    if (await handlePreflight(route)) return;
    log?.push(route.request().postDataJSON());
    await route.fulfill({
      status: 200,
      headers: { ...CORS_HEADERS, 'content-type': 'text/event-stream' },
      body: sseChatFrames(reasoningDeltas, contentDeltas).join('')
    });
  };
}

/**
 * /chat/completions — one reasoning delta, then a mid-stream error chunk.
 * Exercises the after-first-byte contract: the error must surface as a stream
 * fault with NO retry (`log` counts the POST attempts).
 */
export function sseMidStreamErrorHandler({ log } = {}) {
  return async (route) => {
    if (await handlePreflight(route)) return;
    log?.push(route.request().method());
    const frames = [
      `data: ${JSON.stringify({ id: 'gen-e2e-sse', choices: [{ index: 0, delta: { reasoning: 'Considering the retrieved context ' } }] })}\n\n`,
      `data: ${JSON.stringify({ error: { code: 429, message: 'Upstream error from Nvidia: ResourceExhausted mid stream' } })}\n\n`
    ];
    await route.fulfill({
      status: 200,
      headers: { ...CORS_HEADERS, 'content-type': 'text/event-stream' },
      body: frames.join('')
    });
  };
}

/**
 * Paced streaming for tests that must SEE the thinking → writing walk.
 * Playwright's route.fulfill delivers a body atomically, which collapses the
 * whole stream into one paint, so this overrides window.fetch for the chat
 * URL inside the page and hands back a Response whose ReadableStream emits
 * one SSE frame per delay tick.
 *
 * Mock justification (per project rules): Playwright cannot stream a
 * fulfilled body and cross-origin redirects to a local SSE server would trip
 * preflight rules. Only the chat POST is faked; every other request still
 * goes through the real fetch and route stack.
 */
export function installPacedChat(page, { reasoning, content, delayMs = 150 }) {
  return page.addInitScript(
    ({ url, frames, delayMs }) => {
      const orig = window.fetch.bind(window);
      window.fetch = (input, init) => {
        const target = typeof input === 'string' ? input : input.url;
        if (target !== url || (init?.method ?? 'GET') !== 'POST') return orig(input, init);
        const encoder = new TextEncoder();
        const stream = new ReadableStream({
          async start(controller) {
            for (const frame of frames) {
              await new Promise((r) => setTimeout(r, delayMs));
              controller.enqueue(encoder.encode(frame));
            }
            controller.close();
          }
        });
        return Promise.resolve(
          new Response(stream, { status: 200, headers: { 'content-type': 'text/event-stream' } })
        );
      };
    },
    { url: OR_CHAT, frames: sseChatFrames(reasoning, content), delayMs }
  );
}

/** Factory: plain HTTP 429 for every POST (counts attempts into `log`). */
export function http429Handler({ log } = {}) {
  return async (route) => {
    if (await handlePreflight(route)) return;
    log?.push(route.request().method());
    await route.fulfill(json({ error: { code: 429, message: 'Rate limit exceeded: free tier' } }, 429));
  };
}

/**
 * Factory: the OpenRouter quirk — HTTP 200 whose body is an error envelope
 * (transient upstream saturation). The engine must treat it as retryable.
 */
export function ok200ErrorBodyHandler({ log } = {}) {
  return async (route) => {
    if (await handlePreflight(route)) return;
    log?.push(route.request().method());
    await route.fulfill(
      json({ error: { code: 429, message: 'Upstream error from Nvidia: ResourceExhausted — please retry' } })
    );
  };
}
