// SPDX-License-Identifier: AGPL-3.0-only
// =============================================================================
// Blog SSOT. Posts, hrefs, and the writing index copy live here so the pages,
// llms.txt, and the sitemap cannot drift. Performance numbers do not. Those
// are derived at render time from ladder.js, the same helper Verified.svelte
// uses, because a percentage pasted into an article is a hand-typed claim.
//
// Canonical hrefs carry `.html`. adapter-static writes a route to `<name>.html`
// and the deploy target serves files literally.
//
// VOICE and CLAIM POLICY are the same as data.js. Visible copy has no colons,
// em dashes, semicolons, or exclamation marks. Vision is labeled vision.
// =============================================================================

export const SITE_ORIGIN = 'https://atlasinference.io';

export const writing = {
  label: '// writing',
  title: 'Writing.',
  sub: 'Longer pages that stay on the ledger. Numbers still come from the repo.',
  indexHref: '/blog.html',
  crumb: 'Writing'
};

/** `/blog/fleets.html`, the href a static host will actually answer. */
export const postHref = (slug) => `/blog/${slug}.html`;

export const postUrl = (slug) => `${SITE_ORIGIN}${postHref(slug)}`;

export const indexUrl = `${SITE_ORIGIN}${writing.indexHref}`;

export const getPost = (slug) => posts.find((p) => p.slug === slug) ?? null;

export const posts = [
  {
    slug: 'fleets',
    date: '2026-08-27',
    displayDate: 'August 2026',
    kicker: 'concurrency',
    title: 'C=1 is a conversation. C=128 is the work.',
    description:
      'The concurrency thesis at a length an answer engine can cite. A swarm is a fleet. A context bus keeps the story on hardware you own. The motherboard is vision.',
    blocks: [
      {
        type: 'p',
        text: 'The homepage already says it. Agentic work does not arrive as one conversation at a time. It arrives as fleets of tool calling agents sharing a context bus, fanning out and rejoining. The engine underneath them is judged where the requests pile up rather than at a single stream.'
      },
      {
        type: 'p',
        text: 'This page is that sentence at the length an answer engine can cite. Every figure of merit still comes from the committed concurrency ladder. If a number is not in the repo, it is not on this page.'
      },
      { type: 'h2', text: 'C=1 is a conversation.' },
      {
        type: 'p',
        text: 'A single stream is how people demo an engine. It is not how agentic systems use one. One agent calling tools, waiting, calling again, is already more than one stream. A fleet of them is a pile-up. The interesting load is the pile-up.'
      },
      { type: 'h2', text: 'C=128 is the work.' },
      {
        type: 'p',
        text: 'The published ladder runs the same box, the same checkpoint, the same client, and the same prompts, from C=1 to C=128. Greedy sampling with matched penalties. The margin that matters is the top of the ladder.'
      },
      { type: 'scale' },
      {
        type: 'p',
        text: 'That gap is the whole thesis. An engine that flattens under load caps how many agents you can actually run, on any hardware you put it on. Holding the curve is what turns one accelerator into a swarm, and it is why the same engine is worth running on a rack.'
      },
      { type: 'log' },
      { type: 'h2', text: 'Swarm.' },
      {
        type: 'p',
        text: 'Atlas uses swarm for that fleet, not as a product name. A swarm is concurrent tool calling agents sharing work on hardware you own. The published ladder is the receipt. C=1 is a conversation. C=128 is the work.'
      },
      { type: 'h2', text: 'Context bus.' },
      {
        type: 'p',
        text: "A context bus is shared context that stays on the machines running the swarm, without a hop through someone else's cloud. Threads between desks, floors, and partner stacks that already sit on the same story. The homepage already uses the phrase. This names it so it can be cited."
      },
      {
        type: 'figure',
        src: '/blog/customers.webp',
        width: 1920,
        height: 1080,
        alt: 'Three photographs side by side. A field workshop with a desk GPU, a home garage with a local box, and an office tower at dusk. Captioned field, home, and enterprise.',
        caption: 'Field, home, and enterprise. Three customers of the same binary.'
      },
      { type: 'h2', text: 'Three places, one engine.' },
      {
        type: 'p',
        text: 'Field work, a home GPU, and an enterprise floor are three customers of the same binary. Verified silicon today is NVIDIA GB10. AMD gfx1151 runs the same CUDA source through SCALE and is submitted to MLPerf Inference v6.1 in the closed edge division. Numbers wait until MLCommons publishes them.'
      },
      {
        type: 'p',
        text: 'The range is the product. A swarm on a Spark is why the same engine is worth running on a rack, and later on a phone.'
      },
      { type: 'h2', text: 'What this is not.' },
      {
        type: 'p',
        text: 'The AI motherboard is a vision, not a product you can order. The next abstraction Atlas is pointing at is connectors, volatile memory, a context bus, a swarm, and duty, sitting together the way a motherboard sits under a CPU. Verified silicon and the concurrency ladder are what ships today. The motherboard is the direction those pieces travel, with its status attached.'
      },
      {
        type: 'figure',
        src: '/blog/motherboard.webp',
        width: 1920,
        height: 1080,
        alt: 'An office tower at dusk with five labeled stages across the top. Connectors, volatile memory, context bus, swarm, and duty. Marked as vision, not a shipped product.',
        caption:
          'Connectors, volatile memory, a context bus, a swarm, and duty. Labeled vision, not a shipped product.'
      },
      {
        type: 'p',
        text: 'Enterprise is the commercial license path for teams that need different terms than AGPL-3.0-only. The Community Edition stays AGPL. Contributions are covered by a CLA that permits re-licensing. Enterprise is on-prem inference as a B2B API on hardware you own, not a hosted cloud hop.'
      },
      { type: 'h2', text: 'Read the receipts.' },
      {
        type: 'p',
        text: 'The ladder, the methodology, and the reproduce command sit on the verified section of the homepage. This page does not restate throughput figures that the ladder already owns.'
      },
      { type: 'verified' }
    ]
  }
];

/** Paths the sitemap must name. Diligence is noindex and stays off this list. */
export function sitemapPaths() {
  return ['/', '/control.html', writing.indexHref, ...posts.map((p) => postHref(p.slug))];
}
