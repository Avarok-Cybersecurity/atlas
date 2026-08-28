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
  constructor(message, transient, { quota = false, resetAt = null } = {}) {
    super(message);
    this.name = 'OpenRouterError';
    this.transient = transient;
    // A per-day allowance that no amount of backoff will clear before its
    // reset, as opposed to a provider being momentarily saturated.
    this.quota = quota;
    this.resetAt = resetAt;
  }
}

// OpenRouter returns 429 for two very different situations. A saturated
// provider clears in seconds, so it is worth retrying. A per-day free-model
// allowance does not clear until its reset stamp, so retrying only burns time
// and then lies to the reader about what went wrong. The daily case announces
// itself through metadata.limit_source (e.g. "openrouter_free_tier_daily") or
// the message text (e.g. "Rate limit exceeded: free-models-per-day-…").
const DAILY_LIMIT_TEXT = /free-models-per-day|per-?day|daily/i;

function dailyQuota(errorBody, response) {
  const meta = errorBody?.metadata ?? {};
  const source = String(meta.limit_source ?? '');
  const message = String(errorBody?.message ?? '');
  if (!DAILY_LIMIT_TEXT.test(source) && !DAILY_LIMIT_TEXT.test(message)) return null;

  const stamp = meta.headers?.['X-RateLimit-Reset'] ?? response?.headers?.get?.('X-RateLimit-Reset');
  const resetAt = Number(stamp);
  return { resetAt: Number.isFinite(resetAt) && resetAt > 0 ? resetAt : null };
}

/** Non-retryable, carries the reset stamp so the UI can name a real time. */
function quotaError(what, message, quota) {
  return new OpenRouterError(`${what} failed: ${message}`, false, {
    quota: true,
    resetAt: quota.resetAt
  });
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
    let body = null;
    try {
      body = JSON.parse(raw);
    } catch {
      /* not JSON — fall through to the generic message below */
    }
    const quota = response.status === 429 ? dailyQuota(body?.error, response) : null;
    if (quota) throw quotaError(what, body.error.message ?? 'daily limit reached', quota);

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
    const quota = dailyQuota(maybeError, response);
    if (quota) throw quotaError(what, message, quota);
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

  // parseResponse guarantees a 2xx and an `error`-free body, not a shaped one.
  // A 200 whose body is not the documented envelope used to die here as an
  // opaque TypeError -- the exact failure parseResponse exists to prevent.
  if (!Array.isArray(data?.data)) {
    throw new OpenRouterError('Embedding request returned no embeddings.', false);
  }

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
 * Thinking tokens arrive on `delta.reasoning` (a plain string) while
 * `delta.content` stays empty; some providers ship a `delta.reasoning_details`
 * array of typed parts instead. Prefer the string, fall back to concatenating
 * the parts' text — defensively, since the part shapes vary by provider.
 */
function reasoningFromDelta(delta) {
  if (typeof delta.reasoning === 'string' && delta.reasoning) return delta.reasoning;
  if (Array.isArray(delta.reasoning_details)) {
    let out = '';
    for (const part of delta.reasoning_details) {
      if (typeof part?.text === 'string') out += part.text;
      else if (typeof part?.summary === 'string') out += part.summary;
    }
    if (out) return out;
  }
  return undefined;
}

/**
 * Consume an OpenRouter SSE body: `data:` lines carrying JSON chunks,
 * `:`-prefixed keepalive comments, and a final `data: [DONE]`. Reasoning
 * models emit thinking deltas first, then switch to `delta.content` for the
 * answer. An upstream failure can also arrive as an `{ "error": … }` chunk
 * mid-stream — surfaced as a throw from here.
 */
async function readSseStream(bodyStream, emit) {
  const reader = bodyStream.getReader();
  const decoder = new TextDecoder();
  let buffer = '';

  try {
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });

      // SSE frames are newline-delimited; process every complete line and
      // keep any partial remainder in the buffer.
      let nl;
      while ((nl = buffer.indexOf('\n')) !== -1) {
        const line = buffer.slice(0, nl).trim();
        buffer = buffer.slice(nl + 1);

        if (line === '' || line.startsWith(':')) continue; // blank / keepalive
        if (!line.startsWith('data:')) continue;

        const payload = line.slice(5).trim();
        if (payload === '[DONE]') return;

        let chunk;
        try {
          chunk = JSON.parse(payload);
        } catch {
          continue; // tolerate any non-JSON line
        }

        if (chunk.error) {
          const message = chunk.error.message ?? 'unknown error';
          const quota = dailyQuota(chunk.error, null);
          if (quota) throw quotaError('Chat request', message, quota);
          throw new OpenRouterError(
            `Chat request failed: ${message}`,
            chunk.error.code === 429 ||
              (chunk.error.code ?? 0) >= 500 ||
              /ResourceExhausted|rate.?limit|overloaded/i.test(message)
          );
        }

        const delta = chunk.choices?.[0]?.delta;
        if (!delta) continue;
        const reasoning = reasoningFromDelta(delta);
        const answer =
          typeof delta.content === 'string' && delta.content ? delta.content : undefined;
        if (reasoning || answer) emit(reasoning, answer);
      }
    }
  } finally {
    reader.releaseLock?.();
  }
}

/**
 * Chat completion, streamed. `system` is the complete system-message content
 * (the caller builds it — this module stays free of prompt policy).
 *
 * Requests `stream: true` and invokes `onDelta({ reasoning, content })` as
 * thinking and answer tokens arrive; resolves with the full answer text
 * (reasoning excluded). A response that is not SSE falls back to the plain
 * completion shape (and to parseResponse's 200-with-error-body handling).
 * Retries stay confined to errors BEFORE the first emitted token — once
 * anything has been rendered, restarting the stream would duplicate it, so a
 * mid-stream failure surfaces as a stream error instead.
 */
export async function chat(messages, system, apiKey, { onDelta, model = CHAT_MODEL } = {}) {
  const body = JSON.stringify({
    model,
    messages: [{ role: 'system', content: system }, ...messages],
    stream: true
  });

  let content = '';
  let emitted = false;

  const emit = (reasoning, answer) => {
    emitted = true;
    if (answer) content += answer;
    onDelta?.({ reasoning, content: answer });
  };

  const runOnce = async () => {
    const response = await fetch(`${OPENROUTER_API_URL}/chat/completions`, {
      method: 'POST',
      headers: headersFor(apiKey),
      body
    });

    const type = response.headers.get('content-type') ?? '';
    if (!response.ok || !type.includes('text/event-stream')) {
      // Non-SSE: a plain JSON completion, or an error envelope — parseResponse
      // owns !ok, invalid-JSON, and the 200-with-error-body quirk.
      const data = await parseResponse(response, 'Chat request');
      const full = data.choices?.[0]?.message?.content;
      if (full === undefined) {
        throw new Error('Chat request returned no message content');
      }
      emit(undefined, full);
      return;
    }
    if (!response.body) {
      throw new OpenRouterError('Chat request returned no response stream', true);
    }
    await readSseStream(response.body, emit);
  };

  for (let attempt = 1; ; attempt++) {
    try {
      await runOnce();
      return content;
    } catch (error) {
      const transient = error instanceof OpenRouterError && error.transient;
      if (!transient || emitted || attempt >= OR_MAX_ATTEMPTS) throw error;
      await new Promise((resolve) => setTimeout(resolve, retryBaseMs * 2 ** (attempt - 1)));
    }
  }
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

  // See getEmbeddings: a 200 with an undocumented body is a shape failure, not
  // a transient one, so retrying it would only repeat the same answer.
  if (!Array.isArray(data?.results)) {
    throw new OpenRouterError('Rerank request returned no results.', false);
  }

  return data.results
    .slice()
    .sort((a, b) => b.relevance_score - a.relevance_score)
    .map((r) => ({ index: r.index, score: r.relevance_score }));
}
