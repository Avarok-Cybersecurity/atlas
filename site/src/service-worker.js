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
// visitors' disk/bandwidth pre-caching it. /lattice/ (the 763 KB LatticeDB
// wasm + loader for the chat modal) is excluded too: it must load only when
// the feature is used, not at SW install for every visitor.
//
// The install scripts are excluded and stay excluded: someone piping
// install.sh into a shell must get exactly what the server has at that moment,
// never a copy this worker happened to keep.
const INSTALL_SCRIPTS = ['/install.sh', '/install.ps1', '/quickstart.sh'];
const PRECACHE = [
  '/',
  ...build,
  ...files.filter(
    (f) => !f.includes('og-image') && !f.startsWith('/lattice/') && !INSTALL_SCRIPTS.includes(f)
  )
];

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
  // Never serve or store an install script: it must come from the network.
  if (INSTALL_SCRIPTS.includes(url.pathname)) return;

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
