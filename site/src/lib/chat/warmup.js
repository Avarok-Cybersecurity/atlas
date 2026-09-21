// =============================================================================
// warmup.js — cheap pre-warming only. The corpus NEVER downloads here (user
// requirement: no corpus bytes until the modal opens). Two levers:
//   1) preloadChat(): memoized dynamic import of the lazy modal chunk, fired
//      on nav-button pointerenter/focus (Hero lazy-load pattern).
//   2) prefetchWasmOnIdle(): optionally warms the 763 KB wasm binary into the
//      Cache API at low priority — skipped on save-data/slow connections.
// =============================================================================

import { WASM_BIN_URL, WASM_PREFETCH_CACHE } from './config.js';

let chunkPromise = null;

/** Preload the lazy CodeChat chunk. Memoized; a failed load allows a retry. */
export function preloadChat() {
  if (!chunkPromise) {
    chunkPromise = import('../components/CodeChat.svelte').catch((err) => {
      chunkPromise = null;
      throw err;
    });
  }
  return chunkPromise;
}

// Respect constrained connections: never spend a visitor's metered/slow
// bandwidth on a speculative wasm fetch.
function connectionAllowsPrefetch() {
  const conn = typeof navigator !== 'undefined' ? navigator.connection : undefined;
  if (!conn) return true;
  if (conn.saveData) return false;
  if (typeof conn.effectiveType === 'string' && /(^|-)2g$/.test(conn.effectiveType)) return false;
  return true;
}

let wasmPrefetched = false;

/**
 * Idle-time, low-priority prefetch of the wasm binary into the Cache API so
 * the modal's wasm-init phase is warm. Best-effort: any failure is swallowed.
 */
export function prefetchWasmOnIdle() {
  if (wasmPrefetched) return;
  if (typeof window === 'undefined' || typeof caches === 'undefined') return;
  if (!connectionAllowsPrefetch()) return;
  wasmPrefetched = true;

  const run = async () => {
    try {
      const cache = await caches.open(WASM_PREFETCH_CACHE);
      const hit = await cache.match(WASM_BIN_URL);
      if (hit) return;
      const res = await fetch(WASM_BIN_URL, { priority: 'low' });
      if (res.ok) await cache.put(WASM_BIN_URL, res);
    } catch {
      // Best-effort only — the modal load path fetches the wasm itself.
      wasmPrefetched = false;
    }
  };

  if (typeof requestIdleCallback === 'function') {
    requestIdleCallback(() => void run(), { timeout: 4000 });
  } else {
    setTimeout(() => void run(), 2000);
  }
}
