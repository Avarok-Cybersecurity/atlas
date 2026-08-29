import adapter from '@sveltejs/adapter-static';

/** @type {import('@sveltejs/kit').Config} */
const config = {
  kit: {
    adapter: adapter({
      pages: 'build',
      assets: 'build',
      // A real prerendered 404 document. nginx serves it via `error_page 404
      // /404.html`, so a mistyped URL gets the site's own chrome and a way back
      // rather than nginx's stock page.
      fallback: undefined,
      precompress: false,
      strict: true
    }),
    prerender: {
      entries: ['*'],
      // Every internal link must resolve at build time. A blog's whole failure
      // mode is a post linking to a slug that was renamed, and prerendering is
      // the only place that is cheap to catch.
      handleHttpError: 'fail',
      handleMissingId: 'fail'
    }
  }
};

export default config;
