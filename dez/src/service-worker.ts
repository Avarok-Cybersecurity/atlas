// SPDX-License-Identifier: AGPL-3.0-only

/// <reference types="@sveltejs/kit" />
/// <reference lib="webworker" />

import { build, files, prerendered, version } from '$service-worker';

const sw = self as unknown as ServiceWorkerGlobalScope;

/** One cache per build; `version` changes whenever the output changes. */
const CACHE = `dez-cache-${version}`;

/**
 * The full offline shell: hashed JS/CSS (`build`), everything in static/
 * (`files`), and the prerendered HTML pages (`prerendered`). Precaching all
 * three is what lets the site load with the network switched off, which is the
 * whole claim the product makes.
 */
const PRECACHE: readonly string[] = [...build, ...files, ...prerendered];

sw.addEventListener('install', (event) => {
  event.waitUntil(
    (async () => {
      const cache = await caches.open(CACHE);
      await cache.addAll(PRECACHE);
      await sw.skipWaiting();
    })()
  );
});

sw.addEventListener('activate', (event) => {
  event.waitUntil(
    (async () => {
      for (const key of await caches.keys()) {
        if (key !== CACHE) await caches.delete(key);
      }
      await sw.clients.claim();
    })()
  );
});

sw.addEventListener('fetch', (event) => {
  const request = event.request;
  if (request.method !== 'GET') return;

  const url = new URL(request.url);
  // Never touch anything off-origin. There are no third-party requests by
  // design; if one ever appears, it must not silently land in our cache.
  if (url.origin !== sw.location.origin) return;

  event.respondWith(
    (async () => {
      const cache = await caches.open(CACHE);

      // Build artefacts are content-hashed and therefore immutable.
      if (build.includes(url.pathname)) {
        const hit = await cache.match(url.pathname);
        if (hit) return hit;
      }

      try {
        const response = await fetch(request);
        // Opaque/error responses must not poison the offline shell.
        if (response.status === 200 && response.type === 'basic') {
          await cache.put(request, response.clone());
        }
        return response;
      } catch (error) {
        const hit = (await cache.match(request)) ?? (await cache.match('/'));
        if (hit) return hit;
        throw error;
      }
    })()
  );
});
