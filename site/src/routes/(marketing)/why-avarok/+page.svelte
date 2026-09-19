<script>
  import PageShell from '$lib/components/avarok/PageShell.svelte';
  import PageHero from '$lib/components/avarok/PageHero.svelte';
  import ValueBand from '$lib/components/avarok/home/ValueBand.svelte';
  import Chain from '$lib/components/avarok/home/Chain.svelte';
  import Deliveries from '$lib/components/avarok/home/Deliveries.svelte';
  import FaqList from '$lib/components/avarok/FaqList.svelte';
  import CtaBand from '$lib/components/avarok/CtaBand.svelte';
  import { whyHero, pillars, pov } from '$lib/content/why.js';
  import { faqFor } from '$lib/content/faq.js';
  import { routes } from '$lib/content/brand.js';
</script>

<PageShell path={routes.why}>
  <PageHero eyebrow={whyHero.eyebrow} title={whyHero.title} lede={whyHero.lede} primary={whyHero.primary} secondary={whyHero.secondary}>
    <div class="av-grid av-grid-3 av-why-pillars">
      {#each whyHero.pillars as p}
        <div class="av-card av-card-accent av-sx-{p.color}">
          <p class="av-card-tag">{p.name}</p>
          <p>{p.body}</p>
        </div>
      {/each}
    </div>
  </PageHero>

  {#each pillars as p, i}
    <section class="av-section av-sx-{p.color}" class:av-section-alt={i % 2 === 0} id={p.name.toLowerCase()}>
      <div class="av-container">
        <div class="av-head av-reveal">
          <p class="av-eyebrow">{p.n} · {p.name}</p>
          <h2 class="av-h2" style="max-width:24ch">{p.title}</h2>
          <p class="av-lede">{p.lede}</p>
        </div>
        <div class="av-grid av-grid-2 av-reveal">
          {#each p.blocks as b}
            <div class="av-card">
              <h3>{b.h}</h3>
              <p>{b.p}</p>
            </div>
          {/each}
        </div>
        <div class="av-reveal">
          <p class="av-question">{p.question}</p>
          <a class="av-link" style="margin-top:1rem" href={p.proofHref}>{p.proofText} <span class="av-arrow">→</span></a>
        </div>
      </div>
    </section>
  {/each}

  <Chain />
  <ValueBand />
  <Deliveries />
  <FaqList items={faqFor('why')} />

  <section class="av-section av-section-tight">
    <div class="av-container">
      <div class="av-band av-reveal">
        <div class="av-band-grid">
          <div>
            <p class="av-eyebrow">{pov.eyebrow}</p>
            <h2 class="av-h2">{pov.title}</h2>
            <p class="av-lede">{pov.body}</p>
            <ul class="av-list-check" style="margin-top:1.2rem">{#each pov.bullets as b}<li>{b}</li>{/each}</ul>
          </div>
          <div class="av-row" style="justify-content:flex-end">
            <a class="av-btn av-btn-primary av-btn-lg" href={pov.primary.href}>{pov.primary.text} <span class="av-arrow">→</span></a>
            <a class="av-btn av-btn-ghost av-btn-lg" href={pov.secondary.href} target="_blank" rel="noopener">{pov.secondary.text} ↗</a>
          </div>
        </div>
      </div>
    </div>
  </section>
</PageShell>

<style>
  .av-why-pillars { margin-top: 2.25rem; max-width: 1180px; }
</style>
