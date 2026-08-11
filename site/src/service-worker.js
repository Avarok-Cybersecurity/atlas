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

const CACHE = `atlas-site-${version}`;

// The og-image is fetched by social scrapers, never by visitors — don't spend
// visitors' disk/bandwidth pre-caching it.
const PRECACHE = ['/', ...build, ...files.filter((f) => !f.includes('og-image'))];

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

  // Content-hashed build assets are immutable: cache-first is always correct.
  if (url.pathname.includes('/_app/immutable/')) {
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
