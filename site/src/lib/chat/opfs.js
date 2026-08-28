// =============================================================================
// opfs.js — Origin Private File System cache for the decompressed corpus.
// All OPFS I/O for the chat feature lives here (SBIO); callers never touch
// navigator.storage directly. Layout: one file per corpus at the OPFS root,
// named by latticeFileName(commitSha).
//
// Crash safety: createWritable() writes to a browser-managed swap file that is
// only committed on close(), so an interrupted write never leaves a partial
// corpus behind — abort()/discard() (or a tab crash) simply drops the swap.
// =============================================================================

import { latticeFileName, LATTICE_FILE_RE } from './config.js';

/** True when the browser exposes OPFS (absent in some private-browsing modes). */
export function isSupported() {
  return typeof navigator !== 'undefined' && !!navigator.storage?.getDirectory;
}

async function root() {
  if (!isSupported()) return null;
  try {
    return await navigator.storage.getDirectory();
  } catch {
    return null;
  }
}

/**
 * Return the cached decompressed corpus for `sha` as a File, or null when
 * absent/unsupported. When `expectedBytes` is provided (from the manifest),
 * a size mismatch is treated as a miss so a wrong-sized file is never served.
 */
export async function getCachedCorpus(sha, expectedBytes) {
  const dir = await root();
  if (!dir) return null;
  try {
    const handle = await dir.getFileHandle(latticeFileName(sha));
    const file = await handle.getFile();
    if (Number.isFinite(expectedBytes) && expectedBytes > 0 && file.size !== expectedBytes) {
      return null;
    }
    return file;
  } catch {
    return null;
  }
}

/**
 * Streaming writer for the corpus of `sha`. Returns null when OPFS is
 * unavailable (caller degrades to no-cache streaming). The file only becomes
 * visible to getCachedCorpus() after finalize(); discard() drops everything.
 */
export async function createCorpusWriter(sha) {
  const dir = await root();
  if (!dir) return null;
  let handle;
  let writable;
  try {
    handle = await dir.getFileHandle(latticeFileName(sha), { create: true });
    writable = await handle.createWritable({ keepExistingData: false });
  } catch {
    return null;
  }
  let open = true;
  return {
    async write(chunk) {
      if (!open) return;
      await writable.write(chunk);
    },
    async finalize() {
      if (!open) return;
      open = false;
      await writable.close();
    },
    async discard() {
      if (!open) return;
      open = false;
      await writable.abort().catch(() => undefined);
      // Remove any pre-existing file the create:true call may have created
      // empty, so a later cache probe cannot see a zero-byte corpus.
      await dir.removeEntry(latticeFileName(sha)).catch(() => undefined);
    }
  };
}

/** List cached corpora as [{ sha, name, size, lastModified }], newest first. */
export async function listCached() {
  const dir = await root();
  if (!dir) return [];
  const out = [];
  try {
    for await (const [name, handle] of dir) {
      if (handle.kind !== 'file') continue;
      const m = LATTICE_FILE_RE.exec(name);
      if (!m) continue;
      try {
        const file = await handle.getFile();
        out.push({ sha: m[1], name, size: file.size, lastModified: file.lastModified });
      } catch {
        // Unreadable entry: skip rather than break the listing.
      }
    }
  } catch {
    return out;
  }
  return out.sort((a, b) => b.lastModified - a.lastModified);
}

/**
 * Delete every cached corpus except the one for `keepSha`.
 *
 * Refuses a falsy or non-string sha. Both callers pass `meta.commit_sha`, which
 * `ensureReady` has already rejected the manifest for lacking, so this cannot
 * fire today — but the failure mode if it ever did is not "prunes the wrong
 * file", it is "deletes ALL of them": `latticeFileName(undefined)` is
 * `lattice-db-undefined.jsonl`, which matches nothing, so every real corpus
 * takes the delete branch. A precondition is cheap; re-downloading a corpus
 * because a caller passed the wrong thing is not.
 */
export async function pruneStale(keepSha) {
  if (typeof keepSha !== 'string' || keepSha === '') return;
  const dir = await root();
  if (!dir) return;
  const keep = latticeFileName(keepSha);
  const cached = await listCached();
  for (const entry of cached) {
    if (entry.name === keep) continue;
    await dir.removeEntry(entry.name).catch(() => undefined);
  }
}
