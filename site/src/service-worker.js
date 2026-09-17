/// <reference types="@sveltejs/kit" />
// PRPL "pre-cache": after first paint, cache the full app shell (including the
// lazily-imported dashboard chunk) so repeat visits and the on-click dashboard
// are served from disk, and the site survives offline.
//
// Update path (deliberate, do not weaken): the cache name is keyed to the
// build `version`, `activate` deletes every other cache, and documents are
// network-first — so a deploy is picked up on the very next load and the cache
// only answers when the network can't. A stale-forever worker is worse than none.
import { build, files, version } from '$service-worker';
// The routing decisions live in $lib so they can be tested: this file imports
// `$service-worker`, a virtual module that exists only during a build, so
// nothing declared here was reachable from a test.
import { shouldPrecache, strategyFor } from '$lib/sw/strategy.js';

const CACHE = `atlas-site-${version}`;

// The og-image is fetched by social scrapers, never by visitors — don't spend
// visitors' disk/bandwidth pre-caching it. /lattice/ (the 763 KB LatticeDB
// wasm + loader for the chat modal) is excluded too: it must load only when
// the feature is used, not at SW install for every visitor.
//
// The install scripts are excluded and stay excluded: someone piping
// install.sh into a shell must get exactly what the server has at that moment,
// never a copy this worker happened to keep.
const PRECACHE = ['/', ...build, ...files.filter(shouldPrecache)];

self.addEventListener('install', (event) => {
  event.waitUntil(
    caches
      .open(CACHE)
      .then((cache) => cache.addAll(PRECACHE))
      .then(() => self.skipWaiting())
  );
});

self.addEventListener('activate', (event) => {
  event.waitUntil(
    caches
      .keys()
      .then((keys) => Promise.all(keys.filter((key) => key !== CACHE).map((key) => caches.delete(key))))
      .then(() => self.clients.claim())
  );
});

self.addEventListener('fetch', (event) => {
  const { request } = event;
  if (request.method !== 'GET') return;
  const url = new URL(request.url);
  if (url.origin !== location.origin) return;
  const strategy = strategyFor(url.pathname);
  // Not handled at all: the request goes to the network as if no worker were
  // installed, which is what an install script must get.
  if (strategy === 'bypass') return;

  if (strategy === 'cache-first') {
    event.respondWith(
      caches.match(request).then(
        (hit) =>
          hit ??
          fetch(request).then((res) => {
            if (res.ok) {
              const copy = res.clone();
              caches.open(CACHE).then((cache) => cache.put(request, copy));
            }
            return res;
          })
      )
    );
    return;
  }

  // Documents and static files: network-first, cache as offline fallback.
  event.respondWith(
    fetch(request)
      .then((res) => {
        if (res.ok) {
          const copy = res.clone();
          caches.open(CACHE).then((cache) => cache.put(request, copy));
        }
        return res;
      })
      .catch(async () => {
        const hit = await caches.match(request);
        if (hit) return hit;
        if (request.mode === 'navigate') {
          const shell = await caches.match('/');
          if (shell) return shell;
        }
        return Response.error();
      })
  );
});
