<script>
  import PageShell from '$lib/components/avarok/PageShell.svelte';
  import PageHero from '$lib/components/avarok/PageHero.svelte';
  import CtaBand from '$lib/components/avarok/CtaBand.svelte';
  import { labs as l } from '$lib/content/resources.js';
  import { routes } from '$lib/content/brand.js';
</script>

<PageShell path={routes.labs}>
  <PageHero eyebrow={l.eyebrow} title={l.title} lede={l.lede} primary={l.cta} secondary={l.cta2} color="violet" />
  <section class="av-section av-section-alt">
    <div class="av-container">
      <h2 class="av-sr">Research tracks</h2>
      <div class="av-grid av-grid-4 av-reveal">
        {#each l.tracks as t}
          <div class="av-card av-card-accent av-sx-{t.color}"><h3 style="font-size:1rem">{t.title}</h3><p style="font-size:0.9rem">{t.body}</p></div>
        {/each}
      </div>
    </div>
  </section>
  <section class="av-section">
    <div class="av-container av-narrow">
      <div class="av-status-note av-reveal" style="margin:0 0 1.5rem">Direction, not a shipped product. Everything below is labeled as where the work is going.</div>
      <div class="av-reveal">
        <p class="av-eyebrow">{l.vision.title}</p>
        <p class="av-lede av-lede-lg">{l.vision.body}</p>
      </div>
    </div>
  </section>
  <CtaBand title="Working on something adjacent?" body="Kernels, compilers, protocols, benchmarks, safety evaluation. If it advances inference on hardware people own, we want the conversation." primary={l.cta} secondary={{ text: 'Open roles', href: routes.careers }} />
</PageShell>
