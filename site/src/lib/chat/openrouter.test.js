// SPDX-License-Identifier: AGPL-3.0-only

// parseResponse exists, in its own words, so that "the caller reads a missing
// field and throws an opaque TypeError" cannot happen. It covers non-2xx and
// the HTTP-200-with-an-error-body case that OpenRouter's free tier produces.
// It did not cover a 200 whose body simply is not the documented envelope, and
// the two non-streaming unpacks read straight through it.
//
// Only fetch is stubbed — the I/O boundary. Everything under test (retry,
// parseResponse, the shape guards) is the production path.

import { expect, test, afterEach } from 'bun:test';
import { getEmbeddings, rerank, OpenRouterError } from './openrouter.js';

const realFetch = globalThis.fetch;
afterEach(() => {
  globalThis.fetch = realFetch;
});

function respondWith(body, { ok = true, status = 200 } = {}) {
  globalThis.fetch = async () => ({
    ok,
    status,
    text: async () => (typeof body === 'string' ? body : JSON.stringify(body))
  });
}

// A 200 carrying something other than the documented envelope. Retrying cannot
// change a shape, so these must fail fast rather than burn the retry budget.
const MALFORMED = [{}, { data: null }, { data: 'not-an-array' }, { data: { 0: 'x' } }, []];

test('an embedding response with no embeddings names itself instead of throwing a TypeError', async () => {
  for (const body of MALFORMED) {
    respondWith(body);
    let caught;
    try {
      await getEmbeddings(['hello'], 'k');
    } catch (e) {
      caught = e;
    }
    expect(caught).toBeInstanceOf(OpenRouterError);
    expect(caught.message).toBe('Embedding request returned no embeddings.');
    expect(caught.transient).toBe(false);
  }
});

test('a rerank response with no results does the same', async () => {
  for (const body of [{}, { results: null }, { results: 'nope' }]) {
    respondWith(body);
    let caught;
    try {
      await rerank('q', ['a', 'b'], 'k');
    } catch (e) {
      caught = e;
    }
    expect(caught).toBeInstanceOf(OpenRouterError);
    expect(caught.message).toBe('Rerank request returned no results.');
    expect(caught.transient).toBe(false);
  }
});

// The guards must not swallow the good path.
test('a well-formed response still comes back, ordered by index', async () => {
  respondWith({
    data: [
      { index: 1, embedding: [0.2] },
      { index: 0, embedding: [0.1] }
    ]
  });
  expect(await getEmbeddings(['a', 'b'], 'k')).toEqual([[0.1], [0.2]]);

  respondWith({
    results: [
      { index: 0, relevance_score: 0.1 },
      { index: 2, relevance_score: 0.9 }
    ]
  });
  expect(await rerank('q', ['a', 'b', 'c'], 'k')).toEqual([
    { index: 2, score: 0.9 },
    { index: 0, score: 0.1 }
  ]);
});

test('an empty input list never reaches the network', async () => {
  globalThis.fetch = async () => {
    throw new Error('should not have been called');
  };
  expect(await getEmbeddings([], 'k')).toEqual([]);
});
