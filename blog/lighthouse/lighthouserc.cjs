// SPDX-License-Identifier: AGPL-3.0-only
//
// The blog's Lighthouse contract: 100 in every category, on every page.
//
// The bar is not invented here — site/lighthouse/lighthouserc.json already
// holds `minScore: 1` on all four categories for the marketing index, WITH a
// WebGL background, and it holds in this CI. The blog is smaller and fully
// prerendered; it has no excuse. This config extends that precedent to a
// property Lighthouse did not cover at all.
//
// `.cjs` rather than `.json` so the URL list is COMPUTED. A hand-listed set of
// slugs rots the day a post is renamed, and it rots silently — the audit would
// keep passing while quietly testing fewer pages than it names.

const { readdirSync } = require('node:fs');
const { join } = require('node:path');

const build = join(__dirname, '..', 'build');
const posts = readdirSync(join(build, 'posts'))
  .filter((f) => f.endsWith('.html'))
  .map((f) => `http://localhost/posts/${f}`);

module.exports = {
  ci: {
    collect: {
      staticDistDir: './blog/build',
      url: ['http://localhost/index.html', 'http://localhost/404.html', ...posts],
      // Five runs, not three. The performance assertion below takes the best
      // of them, and five samples make "best" robust against one noisy
      // neighbour on a shared runner without slowing the categories that do
      // not vary at all.
      numberOfRuns: 5,
      settings: { preset: 'desktop', chromeFlags: '--no-sandbox --headless=new' }
    },
    assert: {
      assertions: {
        // Accessibility, best-practices and SEO are DOM audits: the same
        // markup scores the same every time, so the median run is exact.
        'categories:accessibility': ['error', { minScore: 1, aggregationMethod: 'median-run' }],
        'categories:best-practices': ['error', { minScore: 1, aggregationMethod: 'median-run' }],
        'categories:seo': ['error', { minScore: 1, aggregationMethod: 'median-run' }],
        // Performance is the only timing-derived category, and the only place
        // a 100 could become a coin flip. Best-of-five is what makes it a gate
        // instead: a real regression — a bigger bundle, a render-blocking
        // resource, a layout shift — depresses EVERY run, while runner
        // contention depresses only some. A flaky required gate is worse than
        // no gate, so the aggregation is chosen to be sensitive to the first
        // and blind to the second.
        'categories:performance': ['error', { minScore: 1, aggregationMethod: 'optimistic' }],
        // Not a score — a determinism guard. The moment a third-party origin
        // appears (a font CDN, an analytics tag, a video thumbnail host),
        // performance stops being a property of this repo and starts being
        // network weather. Holding the count at zero is what keeps the audit
        // hermetic, and it is why the fonts are self-hosted and the video
        // embed is a facade.
        'resource-summary:third-party:count': ['error', { maxNumericValue: 0 }]
      }
    },
    upload: { target: 'temporary-public-storage' }
  }
};
