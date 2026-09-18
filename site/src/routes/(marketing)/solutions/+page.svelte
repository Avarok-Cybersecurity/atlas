<script>
  import PageShell from '$lib/components/avarok/PageShell.svelte';
  import PageHero from '$lib/components/avarok/PageHero.svelte';
  import CtaBand from '$lib/components/avarok/CtaBand.svelte';
  import { solutionsIndex as s, solutions, solutionCta } from '$lib/content/solutions.js';
  import { routes, industries, solutionHref } from '$lib/content/brand.js';
</script>

<PageShell path={routes.solutions}>
  <PageHero eyebrow={s.eyebrow} title={s.title} lede={s.lede} color="cyan" />
  <section class="av-section av-section-alt">
    <div class="av-container">
      <p class="av-eyebrow av-reveal">By industry</p>
      <div class="av-grid av-grid-3 av-reveal">
        {#each industries as i, k}
          <a class="av-card av-card-accent av-sx-{['violet', 'cyan', 'green', 'gold'][k % 4]}" href={solutionHref(i.slug)}>
            <h3>{i.name}</h3>
            <p>{solutions[i.slug].title}</p>
            <span class="av-link">Read the solution <span class="av-arrow">→</span></span>
          </a>
        {/each}
      </div>
    </div>
  </section>
  <section class="av-section">
    <div class="av-container">
      <p class="av-eyebrow av-reveal">{s.deploymentsTitle}</p>
      <div class="av-grid av-grid-4 av-reveal">
        {#each s.deployments as d}
          <a class="av-card" href={d.href}><h3 style="font-size:1rem">{d.name}</h3><p>{d.body}</p><span class="av-link">Open <span class="av-arrow">→</span></span></a>
        {/each}
      </div>
    </div>
  </section>
  <CtaBand eyebrow={solutionCta.eyebrow} title={solutionCta.title} body={solutionCta.body} primary={solutionCta.primary} secondary={solutionCta.secondary} />
</PageShell>
