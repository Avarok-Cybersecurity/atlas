// =============================================================================
// openrouter.js — ALL OpenRouter network I/O for the chat feature (SBIO).
// Port of lattice-db/examples/rag-example/src/openrouter.ts, including its
// 200-with-error-body handling and transient-aware retry policy.
// =============================================================================

import {
  OPENROUTER_API_URL,
  EMBEDDING_MODEL,
  RERANK_MODEL,
  CHAT_MODEL,
  APP_TITLE,
  SITE_ORIGIN,
  OR_MAX_ATTEMPTS,
  OR_RETRY_BASE_MS
} from './config.js';

/** Error that knows whether retrying could plausibly help. */
export class OpenRouterError extends Error {
  constructor(message, transient) {
    super(message);
    this.name = 'OpenRouterError';
    this.transient = transient;
  }
}

const headersFor = (apiKey) => ({
  Authorization: `Bearer ${apiKey}`,
  'Content-Type': 'application/json',
  'HTTP-Referer': typeof window !== 'undefined' ? window.location.origin : SITE_ORIGIN,
  'X-Title': APP_TITLE
});

/**
 * Parse an OpenRouter response, failing fast with a readable message.
 *
 * OpenRouter can report upstream failures as **HTTP 200 with an `error` body**
 * (e.g. `{"error":{"message":"Upstream error from Nvidia: ResourceExhausted…"}}`),
 * which is common on the free tier when a provider is saturated. Without this
 * check the caller reads a missing field and throws an opaque TypeError, so
 * every request funnels through here.
 */
async function parseResponse(response, what) {
  const raw = await response.text();

  if (!response.ok) {
    throw new OpenRouterError(
      `${what} failed: ${response.status} - ${raw}`,
      response.status === 429 || response.status >= 500
    );
  }

  let data;
  try {
    data = JSON.parse(raw);
  } catch {
    throw new OpenRouterError(`${what} returned invalid JSON: ${raw.slice(0, 200)}`, false);
  }

  const maybeError = data?.error;
  if (maybeError) {
    const code = maybeError.code ?? 0;
    const message = maybeError.message ?? 'unknown error';
    throw new OpenRouterError(
      `${what} failed${code ? ` (${code})` : ''}: ${message}`,
      code === 429 || code >= 500 || /ResourceExhausted|rate.?limit|overloaded/i.test(message)
    );
  }

  return data;
}

// The free tier shares provider capacity, so requests intermittently come back
// as "ResourceExhausted". A couple of short retries turn most of those into a
// successful call instead of a failed answer.
let retryBaseMs = OR_RETRY_BASE_MS;

/** Test hook: shrink the backoff so retry paths are E2E-testable in seconds. */
export function _setRetryBaseMs(ms) {
  retryBaseMs = ms;
}
if (typeof window !== 'undefined') {
  window.__atlasChatSetRetryBaseMs = _setRetryBaseMs;
}

async function withRetry(operation) {
  let lastError;

  for (let attempt = 1; attempt <= OR_MAX_ATTEMPTS; attempt++) {
    try {
      return await operation();
    } catch (error) {
      lastError = error;
      const transient = error instanceof OpenRouterError && error.transient;
      if (!transient || attempt === OR_MAX_ATTEMPTS) break;
      // Exponential backoff: 700ms, 1400ms (at the default base).
      await new Promise((resolve) => setTimeout(resolve, retryBaseMs * 2 ** (attempt - 1)));
    }
  }

  throw lastError;
}

/**
 * Embed a batch of texts in a single request. Results are returned in the same
 * order as `texts` (the API may return them out of order, so we sort by index).
 */
export async function getEmbeddings(texts, apiKey, model = EMBEDDING_MODEL) {
  if (texts.length === 0) return [];

  const data = await withRetry(async () => {
    const response = await fetch(`${OPENROUTER_API_URL}/embeddings`, {
      method: 'POST',
      headers: headersFor(apiKey),
      body: JSON.stringify({ model, input: texts })
    });
    return parseResponse(response, 'Embedding request');
  });

  return data.data
    .slice()
    .sort((a, b) => a.index - b.index)
    .map((d) => d.embedding);
}

export async function getEmbedding(text, apiKey, model = EMBEDDING_MODEL) {
  const [embedding] = await getEmbeddings([text], apiKey, model);
  return embedding;
}

/**
 * Chat completion. `system` is the complete system-message content (the caller
 * builds it — this module stays free of prompt policy).
 */
export async function chat(messages, system, apiKey, model = CHAT_MODEL) {
  const allMessages = [{ role: 'system', content: system }, ...messages];

  const data = await withRetry(async () => {
    const response = await fetch(`${OPENROUTER_API_URL}/chat/completions`, {
      method: 'POST',
      headers: headersFor(apiKey),
      body: JSON.stringify({ model, messages: allMessages })
    });
    return parseResponse(response, 'Chat request');
  });

  const content = data.choices?.[0]?.message?.content;
  if (content === undefined) {
    throw new Error('Chat request returned no message content');
  }
  return content;
}

/**
 * Rerank `documents` against `query` with a cross-encoder reranker.
 * Returns document indices (into the input array) ordered most- to
 * least-relevant, each with its relevance score. Callers map indices back to
 * their own records — we do not rely on the response echoing document text.
 */
export async function rerank(query, documents, apiKey, topN, model = RERANK_MODEL) {
  const data = await withRetry(async () => {
    const response = await fetch(`${OPENROUTER_API_URL}/rerank`, {
      method: 'POST',
      headers: headersFor(apiKey),
      body: JSON.stringify({
        model,
        query,
        documents,
        top_n: topN,
        return_documents: false
      })
    });
    return parseResponse(response, 'Rerank request');
  });

  return data.results
    .slice()
    .sort((a, b) => b.relevance_score - a.relevance_score)
    .map((r) => ({ index: r.index, score: r.relevance_score }));
}
