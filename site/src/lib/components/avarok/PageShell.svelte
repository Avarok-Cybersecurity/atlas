<!--
  Every marketing page renders inside this: head tags from the page registry,
  the global nav, the content, the global footer. `path` is the registry key
  (one of `routes`), so a page cannot mount without a title and description.
-->
<script>
  import { pages, SITE } from '$lib/content/index.js';
  import SiteNav from './SiteNav.svelte';
  import SiteFooter from './SiteFooter.svelte';
  import { reveal } from '$lib/reveal.js';

  let { path, ogImage = '/og-image.png', children } = $props();
  // Derived, so the head follows `path` if a parent ever changes it. A page
  // that is not in the registry fails the build here instead of shipping
  // without a title.
  const meta = $derived.by(() => {
    const m = pages.find((p) => p.path === path);
    if (!m) throw new Error(`PageShell: ${path} is not in the page registry (src/lib/content/index.js)`);
    return m;
  });
</script>

<svelte:head>
  <title>{meta.title}</title>
  <meta name="description" content={meta.description} />
  <meta property="og:title" content={meta.title} />
  <meta property="og:description" content={meta.description} />
  <meta property="og:image" content={`${SITE}${ogImage}`} />
  <meta name="twitter:title" content={meta.title} />
  <meta name="twitter:description" content={meta.description} />
  {#if meta.noindex}<meta name="robots" content="noindex" />{/if}
</svelte:head>

<div class="av" id="top" use:reveal>
  <a class="av-skip" href="#main">Skip to content</a>
  <SiteNav />
  <main id="main">{@render children()}</main>
  <SiteFooter />
</div>
