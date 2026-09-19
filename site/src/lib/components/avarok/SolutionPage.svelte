<script>
  import PageShell from './PageShell.svelte';
  import PageHero from './PageHero.svelte';
  import CtaBand from './CtaBand.svelte';
  import { solutionCta } from '$lib/content/solutions.js';
  import { routes, industries, solutionHref } from '$lib/content/brand.js';

  let { path, s } = $props();
  const others = industries.filter((i) => i.slug !== s.slug);
</script>

<PageShell {path}>
  <PageHero eyebrow={s.eyebrow} title={s.title} lede={s.lede} primary={{ text: s.cta, href: routes.demoForm }} secondary={{ text: 'See pricing', href: routes.pricing }} color="cyan" />

  <section class="av-section av-section-alt">
    <div class="av-container">
      <h2 class="av-sr">Where it fits</h2>
      <div class="av-grid av-grid-3 av-reveal">
        <div class="av-card av-sx-violet">
          <p class="av-card-tag">Why it fits</p>
          <ul class="av-list-check">{#each s.fit as f}<li>{f}</li>{/each}</ul>
        </div>
        <div class="av-card av-sx-cyan">
          <p class="av-card-tag">Workloads that move first</p>
          <ul class="av-list-check">{#each s.workloads as w}<li>{w}</li>{/each}</ul>
        </div>
        <div class="av-card av-sx-green">
          <p class="av-card-tag">How it deploys</p>
          <p>{s.deployment}</p>
        </div>
      </div>
      <div class="av-bridge av-reveal">
        <h3>The proof we bring</h3>
        <p>{s.proof}</p>
      </div>
    </div>
  </section>

  <section class="av-section av-section-tight">
    <div class="av-container">
      <p class="av-eyebrow">Other industries</p>
      <ul class="av-row">
        {#each others as o}<li><a class="av-btn av-btn-sm av-btn-ghost" href={solutionHref(o.slug)}>{o.name}</a></li>{/each}
      </ul>
    </div>
  </section>

  <CtaBand eyebrow={solutionCta.eyebrow} title={solutionCta.title} body={solutionCta.body} primary={{ text: s.cta, href: routes.demoForm }} secondary={solutionCta.secondary} />
</PageShell>
