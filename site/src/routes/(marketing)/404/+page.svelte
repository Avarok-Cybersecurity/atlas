<!--
  The PRERENDERED 404 document. adapter-static writes this route to
  `build/404.html`, and both hosts reach for that exact filename: nginx through
  `error_page 404`, Cloudflare Pages by convention. Without it, Pages answers
  every unmatched path with index.html and a 200.
-->
<script>
  import PageShell from '$lib/components/avarok/PageShell.svelte';
  import { routes, links } from '$lib/content/brand.js';
</script>

<PageShell path="/404">
  <section class="av-section" style="min-height:60vh">
    <div class="av-container av-narrow">
      <p class="av-eyebrow">404</p>
      <h1 class="av-h1">That page is not here.</h1>
      <p class="av-lede">It may have been renamed in the facelift, or it may never have existed. The <a class="av-link" href={routes.home}>front page</a> has the pitch, the <a class="av-link" href={routes.benchmarks}>benchmarks</a> have the numbers, and the <a class="av-link" href={routes.openSource}>open source page</a> has the install command. The source is on <a class="av-link" href={links.github}>GitHub</a>.</p>
    </div>
  </section>
</PageShell>
