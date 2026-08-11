import adapter from '@sveltejs/adapter-static';

/** @type {import('@sveltejs/kit').Config} */
const config = {
  kit: {
    adapter: adapter({
      pages: 'build',
      assets: 'build',
      fallback: undefined,
      precompress: false,
      strict: true
    }),
    // PRPL "render the initial route ASAP": inline the (single) stylesheet into
    // the prerendered document so first paint needs no CSS round trip. The 38 KB
    // sheet is the only render-blocking resource; 48 KiB covers it with headroom
    // while leaving any future oversized sheet external rather than bloating the
    // document unboundedly. Repeat-visit caching is handled by the service worker.
    inlineStyleThreshold: 48 * 1024,
    prerender: {
      entries: ['*'],
      handleHttpError: ({ path, referrer, message }) => {
        if (path === '/favicon.png') return;
        throw new Error(message);
      }
    }
  }
};

export default config;
