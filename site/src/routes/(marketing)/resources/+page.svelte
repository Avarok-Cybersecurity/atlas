<script>
  import PageShell from '$lib/components/avarok/PageShell.svelte';
  import PageHero from '$lib/components/avarok/PageHero.svelte';
  import CtaBand from '$lib/components/avarok/CtaBand.svelte';
  import { resourcesHub as r, openSourceCallout as o } from '$lib/content/resources.js';
  import { routes } from '$lib/content/brand.js';
</script>

<PageShell path={routes.resources}>
  <PageHero eyebrow={r.eyebrow} title={r.title} lede={r.lede} color="gold" />
  <section class="av-section av-section-alt">
    <div class="av-container">
      <h2 class="av-sr">All resources</h2>
      <div class="av-grid av-grid-3 av-reveal">
        {#each r.cards as c, i}
          <a class="av-card av-card-accent av-sx-{['violet', 'cyan', 'green', 'gold'][i % 4]}" href={c.href} target={c.external ? '_blank' : undefined} rel={c.external ? 'noopener' : undefined}>
            <h3>{c.title}</h3>
            <p>{c.body}</p>
            <span class="av-link">{c.cta} {c.external ? '↗' : '→'}</span>
          </a>
        {/each}
      </div>
    </div>
  </section>
  <CtaBand eyebrow={o.eyebrow} title={o.title} body={o.body} primary={o.primary} secondary={o.secondary} />
</PageShell>
