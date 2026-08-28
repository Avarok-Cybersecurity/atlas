// SPDX-License-Identifier: AGPL-3.0-only

// Which caching strategy a same-origin GET gets.
//
// Extracted from the service worker so it can be tested. The worker itself
// imports `$service-worker`, a SvelteKit virtual module that only exists
// during a build, so nothing in it was reachable from a test — and one of
// these decisions is security-adjacent: someone piping install.sh into a shell
// must get what the server has at that moment, never a copy this worker
// happened to keep. That rule had no test, and it is one line to lose.

/** Paths that must never be served or stored by the worker. */
export const NEVER_CACHED = ['/install.sh', '/install.ps1', '/quickstart.sh'];

/** Assets whose filenames carry a content hash, so their bytes never change. */
const IMMUTABLE_PREFIX = '/_app/immutable/';

/** @typedef {'bypass' | 'cache-first' | 'network-first'} Strategy */

/**
 * @param {string} pathname
 * @returns {Strategy}
 */
export function strategyFor(pathname) {
  // Checked FIRST, and by exact path. An install script that fell through to
  // network-first would still be written to the cache on success, and served
  // from it the next time the network hiccuped — which is the one outcome
  // this list exists to prevent.
  if (NEVER_CACHED.includes(pathname)) return 'bypass';
  // Content-hashed: the URL changes when the bytes do, so cache-first can
  // never serve something stale.
  if (pathname.startsWith(IMMUTABLE_PREFIX)) return 'cache-first';
  // Everything else — documents, images, the manifest. Network-first so a
  // deploy is picked up on the very next load, with the cache as the offline
  // answer rather than the usual one.
  return 'network-first';
}

/**
 * Whether a file belongs in the install-time precache.
 *
 * @param {string} path
 * @returns {boolean}
 */
export function shouldPrecache(path) {
  // The og-image is fetched by social scrapers, never by visitors. /lattice/
  // is the 763 KB LatticeDB wasm for the chat modal and must load when the
  // feature is used, not at install for everyone.
  if (path.includes('og-image')) return false;
  if (path.startsWith('/lattice/')) return false;
  return !NEVER_CACHED.includes(path);
}
