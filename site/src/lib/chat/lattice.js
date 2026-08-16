// =============================================================================
// lattice.js — thin wrapper around the vendored LatticeDB wasm build.
// Loads /lattice/lattice_server.js (emitted by scripts/gen-lattice.mjs) at
// runtime via dynamic import, so none of the 763 KB wasm lands in the app
// bundle. All wasm-binding quirks (ApiResponse envelopes, the listCollections
// shape) are normalized here and nowhere else.
// =============================================================================

import { WASM_JS_URL, WASM_BIN_URL, COLLECTION } from './config.js';

let db = null;
let initPromise = null;

/**
 * Initialize the wasm module and the (singleton) LatticeDB instance.
 * Idempotent; safe to re-call after a failure (the promise is dropped so a
 * retry re-runs the load). Browser-only — never called during prerender.
 */
export function initWasm() {
  if (db) return Promise.resolve(db);
  if (initPromise) return initPromise;
  initPromise = (async () => {
    // @vite-ignore: the module is a static asset served from /lattice/, not a
    // bundle-graph import — Vite must leave this as a plain runtime import().
    const mod = await import(/* @vite-ignore */ WASM_JS_URL);
    try {
      await mod.default({ module_or_path: WASM_BIN_URL });
    } catch {
      // Origins that serve .wasm with a wrong MIME type break
      // instantiateStreaming — fall back to fetch + ArrayBuffer instantiate.
      const res = await fetch(WASM_BIN_URL);
      if (!res.ok) throw new Error(`wasm fetch failed: HTTP ${res.status}`);
      await mod.default({ module_or_path: await res.arrayBuffer() });
    }
    db = new mod.LatticeDB();
    return db;
  })();
  initPromise.catch(() => {
    initPromise = null;
  });
  return initPromise;
}

// The raw wasm binding returns { status, result, error } envelopes.
function unwrap(res, what) {
  if (res && res.status === 'ok' && res.result !== undefined) return res.result;
  throw new Error(`${what} failed: ${(res && res.error) || 'unknown wasm error'}`);
}

function requireDb() {
  if (!db) throw new Error('LatticeDB wasm not initialized — call initWasm() first');
  return db;
}

// The binding returns collection info as `{ collections: [{ name }] }` at
// runtime even though its .d.ts claims `string[]` — normalize both shapes.
function collectionNames() {
  const raw = unwrap(requireDb().listCollections(), 'listCollections');
  if (Array.isArray(raw)) return raw.map((c) => (typeof c === 'string' ? c : c?.name));
  return (raw?.collections ?? []).map((c) => c.name);
}

/**
 * (Re)create the corpus collection for `dim`-sized cosine vectors. Any
 * existing collection is dropped first so a retried load never indexes on top
 * of a partial one.
 */
export function createCorpusCollection(dim) {
  const handle = requireDb();
  if (collectionNames().includes(COLLECTION)) {
    handle.deleteCollection(COLLECTION);
  }
  const res = handle.createCollection(COLLECTION, {
    vectors: { size: dim, distance: 'Cosine' }
  });
  if (!res || res.status !== 'ok' || res.result !== true) {
    throw new Error(`createCollection failed: ${(res && res.error) || 'unknown wasm error'}`);
  }
}

/** Upsert a batch of `{ id, vector, payload }` points into the corpus. */
export function upsertBatch(points) {
  return unwrap(requireDb().upsert(COLLECTION, points), 'upsert');
}

/**
 * Nearest-neighbor search. Returned `score` is a cosine DISTANCE
 * (0.0 = identical, lower = better).
 */
export function searchVectors(vector, limit) {
  const res = requireDb().search(COLLECTION, new Float32Array(vector), limit, {
    with_payload: true
  });
  return unwrap(res, 'search');
}

/**
 * Yield the main thread between upsert batches so indexing never holds a long
 * task. Uses requestIdleCallback when present, else a macrotask.
 */
export function idleYield() {
  return new Promise((resolve) => {
    if (typeof requestIdleCallback === 'function') {
      requestIdleCallback(() => resolve(), { timeout: 100 });
    } else {
      setTimeout(resolve, 0);
    }
  });
}
