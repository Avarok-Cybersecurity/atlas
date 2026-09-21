// =============================================================================
// state.svelte.js — the "Ask the codebase" state machine (Svelte 5 runes).
// The single API surface the UI imports (plus markdown.js renderMarkdown).
// Orchestrates: wasm init -> manifest preflight -> OPFS-cached load OR
// streamed download (fetch -> DecompressionStream -> tee to OPFS + parser) ->
// batched indexing -> ready. All fetches are browser-only; the engine never
// touches the DOM.
// =============================================================================

import { browser } from '$app/environment';
import {
  CORPUS_GZ_URL,
  CORPUS_META_URL,
  UPSERT_BATCH,
  LS_OPENROUTER_KEY,
  LS_CHAT_MODEL,
  CHAT_MODEL
} from './config.js';
import { initWasm, createCorpusCollection, upsertBatch, idleYield } from './lattice.js';
import { getCachedCorpus, createCorpusWriter, pruneStale, listCached } from './opfs.js';
import { askCodebase } from './rag.js';

export const chat = $state({
  status: 'idle', // 'idle'|'wasm-init'|'manifest'|'loading-cached'|'downloading'|'caching'|'indexing'|'ready'|'error'
  progress: { loadedBytes: 0, totalBytes: 0, indexed: 0, totalPoints: 0 },
  corpus: null, // { commit, files, chunks, dim, generatedAt, repo }
  offline: false, // serving a cached corpus because the manifest fetch failed
  error: null, // { kind: 'wasm'|'manifest'|'corpus'|'decompress'|'rate'|'quota'|'key', message, resetAt? }
  keyState: 'missing', // 'missing'|'set'
  chatModel: CHAT_MODEL, // answer model, overridable by the visitor
  msgPhase: null, // null|'retrieving'|'reranking'|'thinking'|'writing'
  // In-flight streamed message, non-null only while ask() runs. The UI renders
  // its growing texts per animation frame; reasoningMs is stamped when the
  // first answer token arrives (how long the model reasoned).
  stream: null // null | { reasoningText, answerText, reasoningMs }
});

if (browser) {
  chat.keyState = localStorage.getItem(LS_OPENROUTER_KEY) ? 'set' : 'missing';
  chat.chatModel = localStorage.getItem(LS_CHAT_MODEL) || CHAT_MODEL;
}

// --- answer model ------------------------------------------------------------

/**
 * Point the answer model somewhere else. Spending a visitor's credits is their
 * decision alone, so nothing here is ever called automatically — only from an
 * explicit click.
 */
export function setChatModel(model) {
  if (!browser) return;
  const trimmed = String(model ?? '').trim();
  if (!trimmed) return;
  localStorage.setItem(LS_CHAT_MODEL, trimmed);
  chat.chatModel = trimmed;
  if (chat.error?.kind === 'quota') chat.error = null;
}

export function resetChatModel() {
  if (!browser) return;
  localStorage.removeItem(LS_CHAT_MODEL);
  chat.chatModel = CHAT_MODEL;
}

// --- key management ----------------------------------------------------------

export function setKey(key) {
  if (!browser) return;
  const trimmed = String(key ?? '').trim();
  if (!trimmed) return;
  localStorage.setItem(LS_OPENROUTER_KEY, trimmed);
  chat.keyState = 'set';
  if (chat.error?.kind === 'key') chat.error = null;
}

export function clearKey() {
  if (!browser) return;
  localStorage.removeItem(LS_OPENROUTER_KEY);
  chat.keyState = 'missing';
}

function getKey() {
  if (!browser) return null;
  return localStorage.getItem(LS_OPENROUTER_KEY);
}

// --- load orchestration ------------------------------------------------------

let loadPromise = null; // in-flight ensureReady(); cleared on error/abort
let controller = null; // AbortController for the corpus download
let writer = null; // OPFS corpus writer (partial file cleanup on abort)
let loadToken = null; // { aborted } — lets abortLoad() stop non-fetch phases

class AbortedError extends Error {
  constructor() {
    super('load aborted');
    this.name = 'AbortError';
  }
}

function throwIfAborted(token) {
  if (token.aborted) throw new AbortedError();
}

function resetProgress() {
  chat.progress.loadedBytes = 0;
  chat.progress.totalBytes = 0;
  chat.progress.indexed = 0;
  chat.progress.totalPoints = 0;
}

function fail(kind, err) {
  chat.status = 'error';
  chat.error = { kind, message: err?.message ? String(err.message) : String(err) };
}

/**
 * Drive the machine to 'ready'. Idempotent: concurrent calls share one load;
 * re-calling after an error retries from scratch. Resolves when the machine
 * settles ('ready' or 'error' — errors are reported via chat.error, not by
 * rejecting, so callers never need their own catch for state display).
 */
export function ensureReady() {
  if (!browser) return Promise.resolve();
  if (chat.status === 'ready') return Promise.resolve();
  if (loadPromise) return loadPromise;

  const token = { aborted: false };
  loadToken = token;
  loadPromise = load(token)
    .catch((err) => {
      if (token.aborted || err?.name === 'AbortError') {
        // abortLoad() already restored 'idle' — not an error state.
        return;
      }
      if (!chat.error) fail('corpus', err);
    })
    .finally(() => {
      if (chat.status !== 'ready') loadPromise = null;
      if (loadToken === token) loadToken = null;
      controller = null;
      writer = null;
    });
  return loadPromise;
}

/** Abort an in-flight load: cancel the fetch, drop any partial OPFS file. */
export function abortLoad() {
  if (chat.status === 'ready' || chat.status === 'idle') return;
  if (loadToken) loadToken.aborted = true;
  controller?.abort();
  const w = writer;
  writer = null;
  if (w) w.discard().catch(() => undefined);
  loadPromise = null;
  chat.error = null;
  chat.offline = false;
  resetProgress();
  chat.status = 'idle';
}

async function load(token) {
  chat.error = null;
  chat.offline = false;
  resetProgress();

  // 1) wasm
  chat.status = 'wasm-init';
  try {
    await initWasm();
  } catch (err) {
    fail('wasm', err);
    return;
  }
  throwIfAborted(token);

  // 2) manifest preflight (tiny meta.json: sha/size/dim)
  chat.status = 'manifest';
  let meta = null;
  try {
    const res = await fetch(CORPUS_META_URL, { cache: 'no-cache' });
    if (!res.ok) throw new Error(`manifest fetch failed: HTTP ${res.status}`);
    meta = await res.json();
    // `commit_sha` must be a STRING, not merely truthy. A numeric one passes a
    // truthiness check, is coerced into a filename (`lattice-db-12345.jsonl`),
    // and then fails `pruneStale`'s `typeof` precondition — which returns
    // early, so pruning is silently disabled and cached corpora accumulate in
    // OPFS with nothing reporting it.
    if (typeof meta?.commit_sha !== 'string' || meta.commit_sha === '') {
      throw new Error('manifest is missing commit_sha/dim');
    }
    if (!Number.isFinite(meta?.dim)) {
      throw new Error('manifest is missing commit_sha/dim');
    }
  } catch (err) {
    throwIfAborted(token);
    // Manifest unreachable — fall back to any cached corpus (offline path).
    const cached = await listCached();
    if (cached.length === 0) {
      fail('manifest', err);
      return;
    }
    const file = await getCachedCorpus(cached[0].sha);
    if (!file) {
      fail('manifest', err);
      return;
    }
    chat.offline = true;
    await loadFromFile(token, file, cached[0].sha, null);
    return;
  }
  throwIfAborted(token);

  // 3) OPFS hit? Serve the corpus with zero network.
  const cachedFile = await getCachedCorpus(meta.commit_sha, meta.bytes);
  throwIfAborted(token);
  if (cachedFile) {
    await loadFromFile(token, cachedFile, meta.commit_sha, meta);
    return;
  }

  // 4) Download: fetch .gz -> count bytes -> gunzip -> tee (OPFS + parser).
  await loadFromNetwork(token, meta);
}

// Serve a previously cached decompressed corpus from OPFS.
async function loadFromFile(token, file, sha, meta) {
  chat.status = 'loading-cached';
  chat.progress.totalBytes = file.size;
  const { header, repo } = await indexByteStream(token, file.stream(), {
    meta,
    countBytes: true,
    onDrained: () => pruneStale(sha)
  });
  throwIfAborted(token);
  finishReady(header, repo, sha, meta);
}

// Stream-download the corpus, decompressing and indexing as bytes arrive,
// while teeing the decompressed stream into an OPFS write.
async function loadFromNetwork(token, meta) {
  chat.status = 'downloading';
  controller = new AbortController();

  let res;
  try {
    res = await fetch(CORPUS_GZ_URL, { signal: controller.signal });
    if (!res.ok) throw new Error(`corpus fetch failed: HTTP ${res.status}`);
    if (!res.body) throw new Error('corpus fetch returned no body stream');
  } catch (err) {
    throwIfAborted(token);
    if (err?.name === 'AbortError') throw err;
    fail('corpus', err);
    return;
  }

  // Determinate progress: Content-Length when the server provides it for the
  // gz transfer, else the manifest's recorded gz size.
  const contentLength = Number(res.headers.get('content-length'));
  chat.progress.totalBytes =
    Number.isFinite(contentLength) && contentLength > 0 ? contentLength : (meta.gz_bytes ?? 0);

  if (typeof DecompressionStream !== 'function') {
    fail('decompress', new Error('this browser lacks DecompressionStream (gzip) support'));
    return;
  }

  // Count compressed bytes as they arrive (drives the download progress bar).
  const counted = res.body.pipeThrough(
    new TransformStream({
      transform(chunk, ctrl) {
        chat.progress.loadedBytes += chunk.byteLength;
        ctrl.enqueue(chunk);
      }
    })
  );

  const decompressed = counted.pipeThrough(new DecompressionStream('gzip'));

  // OPFS writer for the decompressed corpus; null when OPFS is unavailable
  // (private browsing) — we then stream without caching.
  writer = await createCorpusWriter(meta.commit_sha);
  if (token.aborted) {
    // abortLoad() ran while the writer was being created (it saw writer ===
    // null and had nothing to discard) — discard here or the create:true file
    // and its open-writable swap leak in OPFS.
    const w = writer;
    writer = null;
    if (w) await w.discard().catch(() => undefined);
    throw new AbortedError();
  }

  // Runs once the byte stream is fully drained (all bytes teed), BEFORE the
  // trailing 'indexing' flush — preserving the contract's state order
  // downloading -> caching -> indexing.
  const commitCache = async () => {
    chat.status = 'caching';
    const w = writer;
    writer = null;
    if (w) {
      try {
        await w.finalize();
      } catch {
        // Cache write failure is non-fatal — the corpus is already indexed.
        await w.discard().catch(() => undefined);
      }
    }
    await pruneStale(meta.commit_sha);
  };

  let header;
  let repo;
  try {
    ({ header, repo } = await indexByteStream(token, decompressed, {
      meta,
      tee: writer,
      countBytes: false,
      onDrained: commitCache
    }));
  } catch (err) {
    const w = writer;
    writer = null;
    if (w) await w.discard().catch(() => undefined);
    if (token.aborted || err?.name === 'AbortError') throw err;
    // A gzip stream that fails mid-flight surfaces here.
    if (!chat.error) {
      fail(/gzip|decompress|inflate|corrupt/i.test(String(err?.message)) ? 'decompress' : 'corpus', err);
    }
    return;
  }

  throwIfAborted(token);
  finishReady(header, repo, meta.commit_sha, meta);
}

/**
 * Consume a byte stream of lattice-jsonl: parse the header line, create the
 * collection, then upsert points in UPSERT_BATCH batches with idle yields so
 * the main thread never holds a long task. Optionally tees raw bytes to an
 * OPFS writer and/or counts bytes into progress.loadedBytes.
 */
async function indexByteStream(token, byteStream, { meta, tee, countBytes, onDrained }) {
  const reader = byteStream.getReader();
  const decoder = new TextDecoder();
  let buf = '';
  let header = null;
  let repo = null;
  let batch = [];

  const flush = async () => {
    if (batch.length === 0) return;
    // Mid-stream flushes keep the current status ('downloading' or
    // 'loading-cached'); the trailing flush after onDrained runs as 'indexing'.
    upsertBatch(batch);
    chat.progress.indexed += batch.length;
    batch = [];
    await idleYield();
    throwIfAborted(token);
  };

  const handleLine = (line) => {
    const trimmed = line.trim();
    if (trimmed === '') return;
    if (!header) {
      const h = JSON.parse(trimmed);
      if (h?.t !== 'header' || h?.format !== 'lattice-jsonl') {
        throw new Error('corpus is not lattice-jsonl (missing header line)');
      }
      if (meta && Number.isFinite(meta.dim) && h.dim !== meta.dim) {
        throw new Error(
          `corpus header says ${h.dim} dimensions but the manifest says ${meta.dim} — ` +
            `the published corpus and manifest disagree`
        );
      }
      header = h;
      chat.progress.totalPoints = h.points ?? 0;
      createCorpusCollection(h.dim);
      return;
    }
    const p = JSON.parse(trimmed);
    if (p?.t !== 'point') return;
    if (repo === null && p.payload?.repo) repo = p.payload.repo;
    batch.push({ id: p.id, vector: p.vector, payload: p.payload });
  };

  try {
    for (;;) {
      throwIfAborted(token);
      const { done, value } = await reader.read();
      if (done) break;
      if (tee) await tee.write(value);
      if (countBytes) chat.progress.loadedBytes += value.byteLength;
      buf += decoder.decode(value, { stream: true });
      let nl;
      while ((nl = buf.indexOf('\n')) !== -1) {
        handleLine(buf.slice(0, nl));
        buf = buf.slice(nl + 1);
        if (batch.length >= UPSERT_BATCH) await flush();
      }
    }
    buf += decoder.decode();
    if (buf.trim() !== '') handleLine(buf);
  } finally {
    reader.releaseLock?.();
  }

  if (!header) throw new Error('corpus stream ended before a header line');

  throwIfAborted(token);
  await onDrained?.();
  throwIfAborted(token);

  // Trailing points that arrived after the last mid-stream flush.
  chat.status = 'indexing';
  await flush();

  return { header, repo };
}

function finishReady(header, repo, sha, meta) {
  chat.corpus = {
    commit: sha,
    files: meta?.files ?? null,
    chunks: header.points ?? chat.progress.indexed,
    dim: header.dim,
    generatedAt: meta?.generated_at ?? null,
    repo: repo ?? 'Avarok-Cybersecurity/atlas'
  };
  chat.status = 'ready';
}

// --- asking ------------------------------------------------------------------

/**
 * Ask a question against the ready corpus. `history` is prior turns as
 * [{ role, content }]. While in flight, streamed thinking/answer tokens grow
 * chat.stream (batched to one state write per animation frame so a fast token
 * burst costs one re-render) and chat.msgPhase walks
 * retrieving -> reranking -> thinking -> writing. Resolves
 * { answer, sources, reasoning, reasoningMs } (see rag.js for answer/sources).
 * Rejects on failure after recording a chat.error of kind 'key'
 * (missing/rejected key) or 'rate' (transient upstream saturation).
 */
export async function ask(question, history = []) {
  if (chat.status !== 'ready') {
    throw new Error('the corpus is not ready yet');
  }
  const apiKey = getKey();
  if (!apiKey) {
    const err = new Error('no OpenRouter key set — add one to start asking');
    failAsk('key', err);
    throw err;
  }

  chat.stream = { reasoningText: '', answerText: '', reasoningMs: 0 };
  let reasoningAll = ''; // full trace for the resolved message (survives chat.stream reset)
  let pendingReasoning = '';
  let pendingAnswer = '';
  let rafId = 0;
  let thinkStart = 0;

  const flush = () => {
    rafId = 0;
    if (!chat.stream) return;
    if (pendingReasoning) {
      chat.stream.reasoningText += pendingReasoning;
      pendingReasoning = '';
    }
    if (pendingAnswer) {
      chat.stream.answerText += pendingAnswer;
      pendingAnswer = '';
    }
  };
  const schedule = () => {
    if (!rafId) rafId = requestAnimationFrame(flush);
  };

  const onDelta = ({ reasoning, content }) => {
    if (reasoning) {
      if (!thinkStart) thinkStart = performance.now();
      reasoningAll += reasoning;
      pendingReasoning += reasoning;
      schedule();
    }
    if (content) {
      // First answer token: phase flips immediately (not on the rAF tick) so
      // the trace collapse and the pill react without a frame of lag.
      if (chat.msgPhase !== 'writing') {
        chat.msgPhase = 'writing';
        if (thinkStart && chat.stream && chat.stream.reasoningMs === 0) {
          chat.stream.reasoningMs = performance.now() - thinkStart;
        }
      }
      pendingAnswer += content;
      schedule();
    }
  };

  try {
    const result = await askCodebase(question, history, {
      apiKey,
      corpus: chat.corpus,
      chatModel: chat.chatModel,
      onPhase: (phase) => {
        chat.msgPhase = phase;
      },
      onDelta
    });
    if (['rate', 'quota', 'key'].includes(chat.error?.kind)) chat.error = null;
    const reasoningMs =
      chat.stream?.reasoningMs || (thinkStart ? performance.now() - thinkStart : 0);
    return { ...result, reasoning: reasoningAll, reasoningMs };
  } catch (err) {
    const msg = String(err?.message ?? err);
    // A spent daily allowance is not a momentary rate limit: it has a known
    // reset and its own remedy, so it must never wear the "try again in a few
    // seconds" copy.
    if (err?.quota) failAsk('quota', err, { resetAt: err.resetAt ?? null });
    else if (/\b401\b|\b403\b|invalid.{0,20}key|no auth/i.test(msg)) failAsk('key', err);
    else if (/dimensions but this corpus/.test(msg)) failAsk('corpus', err);
    else failAsk('rate', err);
    throw err;
  } finally {
    if (rafId) cancelAnimationFrame(rafId);
    chat.msgPhase = null;
    chat.stream = null;
  }
}

// Chat-time errors must not knock the machine out of 'ready' — the corpus is
// still indexed and usable; only the message failed.
function failAsk(kind, err, extra = {}) {
  chat.error = {
    kind,
    message: err?.message ? String(err.message) : String(err),
    ...extra
  };
}
