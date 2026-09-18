<script>
  import PageShell from '$lib/components/avarok/PageShell.svelte';
  import PageHero from '$lib/components/avarok/PageHero.svelte';
  import ArchDiagram from '$lib/components/avarok/ArchDiagram.svelte';
  import CtaBand from '$lib/components/avarok/CtaBand.svelte';
  import { platformOverview as p } from '$lib/content/platform.js';
  import { routes } from '$lib/content/brand.js';
</script>

<PageShell path={routes.platform}>
  <PageHero eyebrow={p.eyebrow} title={p.title} lede={p.lede} primary={p.primary} secondary={p.secondary} />

  <section class="av-section av-section-alt">
    <div class="av-container">
      <div class="av-grid av-grid-3 av-reveal">
        {#each p.tiles as t}
          <a class="av-card av-card-accent av-sx-{t.color}" href={t.href}>
            <p class="av-card-tag">{t.tag}</p>
            <h3>{t.name}</h3>
            <p>{t.body}</p>
            <span class="av-link">Explore <span class="av-arrow">→</span></span>
          </a>
        {/each}
      </div>
    </div>
  </section>

  <section class="av-section">
    <div class="av-container">
      <div class="av-head av-reveal">
        <p class="av-eyebrow">Architecture</p>
        <h2 class="av-h2">One request path. One ledger.</h2>
      </div>
      <div class="av-reveal"><ArchDiagram /></div>
      <div class="av-split av-reveal" style="margin-top:3rem;align-items:start">
        <div>
          <h3 class="av-h3">{p.surfaces.title}</h3>
          <p class="av-lede" style="font-size:1rem">{p.surfaces.body}</p>
          <ul class="av-row" style="margin-top:1rem">{#each p.surfaces.items as s}<li class="av-chip">{s}</li>{/each}</ul>
        </div>
        <div class="av-bridge" style="margin-top:0">
          <h3>{p.band.title}</h3>
          <p>{p.band.body}</p>
        </div>
      </div>
    </div>
  </section>

  <CtaBand title="See the platform on your workload." body="A working session with the console, the ladder and the payback model, on demo data or yours." primary={{ text: 'Book a demo', href: routes.demo }} secondary={{ text: 'Run the payback model', href: `${routes.pricing}#payback` }} />
</PageShell>
