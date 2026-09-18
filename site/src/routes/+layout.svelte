<script>
  // The root layout: what every route shares. Design tokens and the brand
  // typeface, the two lockup definition blocks, the canonical URL, and the
  // structured data. Stylesheets that belong to one route group live in that
  // group's layout: app.css in (engine), avarok.css in (marketing), so the
  // developer pages and the marketing pages never load each other's rules.
  import '../../../web-shared/avarok-tokens.css';
  import '../styles/fonts.css';
  import AtlasLockup from '$shared/components/AtlasLockup.svelte';
  import { page } from '$app/state';
  import { onMount } from 'svelte';
  import { detectHost } from '$lib/install/host.svelte.js';
  import { faq as engineFaq, githubUrl, recipesUrl, discordUrl, xUrl, hero } from '$lib/data.js';
  import { pages, SITE, company, links } from '$lib/content/index.js';
  import { faqFor } from '$lib/content/faq.js';
  let { children } = $props();

  // Once, here, rather than in each surface that prints an install line.
  onMount(detectHost);

  // Canonicals are extensionless. Cloudflare Pages pretty-URLs /engine (200)
  // and 308s /engine.html → /engine, so a canonical ending in .html names a
  // redirect. adapter-static still writes engine.html as the file.
  const path = $derived(page.url.pathname.replace(/\.html$/, '').replace(/\/index$/, '/'));
  const canonical = $derived(path === '/' ? SITE + '/' : `${SITE}${path.replace(/\/$/, '')}`);
  const enginePage = $derived(path === '/engine');
  const registry = $derived(pages.find((p) => p.path === (path === '/' ? '/' : path.replace(/\/$/, ''))));

  // One @graph rather than separate blocks, so the entities can reference
  // each other by @id. Every field restates something rendered on a page. The
  // FAQ entities come from the same lists the FAQ sections render, because
  // marking up an answer a visitor cannot see is a policy violation.
  const faqItems = $derived(enginePage ? engineFaq.items : registry?.faq ? faqFor(registry.faq) : []);
  const graph = $derived({
    '@context': 'https://schema.org',
    '@graph': [
      {
        '@type': 'Organization',
        '@id': `${SITE}/#org`,
        name: company.name,
        legalName: company.legal,
        url: `${SITE}/`,
        logo: `${SITE}/icon-512.png`,
        description: company.short,
        sameAs: [githubUrl, recipesUrl, discordUrl, xUrl, links.blog]
      },
      {
        '@type': 'WebSite',
        '@id': `${SITE}/#site`,
        url: `${SITE}/`,
        name: company.name,
        description: company.short,
        inLanguage: 'en',
        publisher: { '@id': `${SITE}/#org` }
      },
      {
        '@type': 'SoftwareApplication',
        '@id': `${SITE}/#app`,
        name: company.engine,
        alternateName: 'Atlas Inference Engine',
        applicationCategory: 'DeveloperApplication',
        applicationSubCategory: 'LLM inference engine',
        operatingSystem: 'Linux',
        processorRequirements: 'NVIDIA GB10 (DGX Spark) or AMD gfx1151 (Strix Halo)',
        description: hero.sub,
        url: `${SITE}/engine`,
        downloadUrl: githubUrl,
        softwareHelp: links.docs,
        programmingLanguage: ['Rust', 'CUDA'],
        license: 'https://spdx.org/licenses/AGPL-3.0-only.html',
        isAccessibleForFree: true,
        offers: { '@type': 'Offer', price: '0', priceCurrency: 'USD' },
        publisher: { '@id': `${SITE}/#org` }
      },
      ...(faqItems.length
        ? [
            {
              '@type': 'FAQPage',
              '@id': `${canonical}#faq`,
              isPartOf: { '@id': `${SITE}/#site` },
              mainEntity: faqItems.map((item) => ({
                '@type': 'Question',
                name: item.q,
                acceptedAnswer: { '@type': 'Answer', text: item.a }
              }))
            }
          ]
        : [])
    ]
  });

  // JSON.stringify does not escape the less-than character, so a closing
  // script tag anywhere in the copy would end the emitted block early.
  const ldjson = $derived(JSON.stringify(graph).replace(/</g, '\\u003c'));
</script>

<svelte:head>
  <!-- No <title> here. A layout title and a page title compete for the single
       head slot and the layout's wins. Every route owns its own title. -->
  <link rel="canonical" href={canonical} />
  <meta property="og:url" content={canonical} />
  {@html `<script type="application/ld+json">${ldjson}<\/script>`}
</svelte:head>

<!-- The brand vector, defined once and <use>d by every lockup on the page. -->
<AtlasLockup kind="defs" />

{@render children()}
