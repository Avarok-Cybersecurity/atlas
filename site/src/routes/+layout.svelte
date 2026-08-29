<script>
  // Load order matters. app.css is the desktop-first design system, news.css adds
  // the news band, dashboard.css then chat.css add the two modals (chat reuses
  // dashboard pieces so it loads after), ladder.css styles the concurrency
  // section (it reuses dashboard's chart primitives, so it loads after those),
  // mobile.css is the SSOT for every viewport rule and must land last.
  import '../app.css';
  import '../styles/news.css';
  import '../styles/dashboard.css';
  import '../styles/chat.css';
  import '../styles/ladder.css';
  import '../styles/control.css';
  import '../styles/bridge.css';
  import '../styles/stage.css';
  import '../styles/overlays.css';
  import '../styles/mobile.css';
  import { page } from '$app/state';
  import { tagline, hero, faq, githubUrl, recipesUrl, discordUrl, xUrl } from '$lib/data.js';
  import { onMount } from 'svelte';
  import { detectHost } from '$lib/install/host.svelte.js';
  let { children } = $props();

  // Once, here, rather than in each surface that prints an install line: three
  // components sniffing separately is how the hero and the control page end up
  // disagreeing about which machine the visitor is on. In `onMount` because
  // the prerendered HTML has no navigator, and rendering a guess into it would
  // make the page rewrite itself for every visitor it guessed wrong about.
  onMount(detectHost);

  // Route-aware: a hardcoded canonical meant /control emitted two of them,
  // which is the same as emitting none.
  //
  // The `.html` matters. adapter-static writes a sub-page to `<name>.html`, and
  // the deploy target serves files literally — no extension guessing, no
  // directory index outside the document root. A canonical of `/control` named
  // a URL that answers 500, which is worse than naming none.
  const canonical = $derived(
    page.url.pathname === '/'
      ? SITE
      : `${SITE.replace(/\/$/, '')}${page.url.pathname}.html`
  );

  const SITE = 'https://atlasinference.io/';

  // One @graph rather than three separate blocks, so the entities can reference
  // each other by @id — that is what lets a search or answer engine tie the
  // software, the publisher and the answers together instead of treating them
  // as unrelated fragments.
  //
  // Every field restates something rendered on the page. The FAQ entities come
  // from the same `faq` export the FAQ section renders, because marking up an
  // answer a visitor cannot see is a policy violation, not an optimisation.
  const graph = {
    '@context': 'https://schema.org',
    '@graph': [
      {
        '@type': 'Organization',
        '@id': `${SITE}#org`,
        name: 'Atlas Inference',
        url: SITE,
        logo: `${SITE}icon-512.png`,
        description: tagline,
        sameAs: [githubUrl, recipesUrl, discordUrl, xUrl]
      },
      {
        '@type': 'WebSite',
        '@id': `${SITE}#site`,
        url: SITE,
        name: 'Atlas Inference',
        description: tagline,
        inLanguage: 'en',
        publisher: { '@id': `${SITE}#org` }
      },
      {
        '@type': 'SoftwareApplication',
        '@id': `${SITE}#app`,
        name: 'Atlas Inference Engine',
        alternateName: 'Atlas',
        applicationCategory: 'DeveloperApplication',
        applicationSubCategory: 'LLM inference engine',
        operatingSystem: 'Linux',
        processorRequirements: 'NVIDIA GB10 (DGX Spark) or AMD gfx1151 (Strix Halo)',
        description: hero.sub,
        url: SITE,
        downloadUrl: githubUrl,
        softwareHelp: recipesUrl,
        programmingLanguage: ['Rust', 'CUDA'],
        license: 'https://spdx.org/licenses/AGPL-3.0-only.html',
        isAccessibleForFree: true,
        // Schema.org wants a price even for free software; omitting the offer
        // entirely reads as "price unknown" rather than "free".
        offers: { '@type': 'Offer', price: '0', priceCurrency: 'USD' },
        publisher: { '@id': `${SITE}#org` }
      },
      {
        '@type': 'FAQPage',
        '@id': `${SITE}#faq`,
        isPartOf: { '@id': `${SITE}#site` },
        mainEntity: faq.items.map((item) => ({
          '@type': 'Question',
          name: item.q,
          acceptedAnswer: { '@type': 'Answer', text: item.a }
        }))
      }
    ]
  };

  // JSON.stringify does not escape the less-than character, so a closing
  // script tag appearing anywhere in the FAQ copy would end the emitted block
  // early and spill the rest of the page's markup into the document. Escaping
  // it as \u003c keeps the JSON parseable and identical in value to consumers.
  // (Writing that tag literally in this comment would end THIS block, too.)
  const ldjson = JSON.stringify(graph).replace(/</g, '\\u003c');
</script>

<svelte:head>
  <!-- No <title> here. A layout title and a page title compete for the single
       head slot and the LAYOUT's wins, so /control shipped with the homepage's
       title while its own two <meta>s came through -- the page head renders,
       only its title is discarded. Every route owns its own title instead. -->
  <link rel="canonical" href={canonical} />
  {@html `<script type="application/ld+json">${ldjson}<\/script>`}
</svelte:head>

{@render children()}
