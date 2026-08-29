<script>
  import '../app.css';
  import { page } from '$app/state';
  import ChevronField from '$shared/components/ChevronField.svelte';
  import AtlasLockup from '$shared/components/AtlasLockup.svelte';
  import Header from '$lib/components/Header.svelte';
  import Footer from '$lib/components/Footer.svelte';
  import { SITE, blog, cleanSlug } from '$lib/content.js';

  let { children } = $props();

  /* adapter-static writes a sub-page to `<name>.html` and nginx resolves the
     extensionless URL onto it, so the canonical is the extensionless form —
     the one a visitor can actually paste. */
  const canonical = $derived(
    SITE + (page.url.pathname === '/' ? '/' : cleanSlug(page.url.pathname))
  );
</script>

<svelte:head>
  <!-- No <title> here on purpose. A layout title and a page title compete for
       the single head slot and the LAYOUT's wins, which is how a site ships
       every article under the homepage's name. Each route owns its own. -->
  <link rel="canonical" href={canonical} />
  <meta property="og:url" content={canonical} />
</svelte:head>

<!-- The canvas must stay a direct child of the layout root. A `transform`,
     `filter`, `perspective`, `will-change` or `contain: paint` on any ancestor
     would make that ancestor the containing block for fixed-position
     descendants, and the background would silently start scrolling with the
     content instead of staying put. -->
<ChevronField />

<!-- The brand vector, defined once and <use>d by every lockup below. -->
<AtlasLockup kind="defs" />

<a class="skip" href="#main">Skip to content</a>
<div class="page">
  <Header />
  <main id="main">{@render children()}</main>
  <Footer />
</div>
