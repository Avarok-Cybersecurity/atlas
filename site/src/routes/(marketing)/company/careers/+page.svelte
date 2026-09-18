<script>
  import PageShell from '$lib/components/avarok/PageShell.svelte';
  import PageHero from '$lib/components/avarok/PageHero.svelte';
  import CtaBand from '$lib/components/avarok/CtaBand.svelte';
  import { careers as c } from '$lib/content/company.js';
  import { routes, links } from '$lib/content/brand.js';
</script>

<PageShell path={routes.careers}>
  <PageHero eyebrow={c.eyebrow} title={c.title} lede={c.lede} primary={c.cta} secondary={{ text: 'The repository', href: links.github, external: true }} color="gold" />
  <section class="av-section av-section-alt">
    <div class="av-container">
      <h2 class="av-sr">How we work</h2>
      <div class="av-grid av-grid-4 av-reveal">
        {#each c.benefits as b, i}<div class="av-card av-card-accent av-sx-{['violet', 'cyan', 'green', 'gold'][i]}"><h3>{b.title}</h3><p>{b.body}</p></div>{/each}
      </div>
    </div>
  </section>
  <section class="av-section" id="roles">
    <div class="av-container">
      <div class="av-head-split av-reveal">
        <div><p class="av-eyebrow">{c.rolesTitle}</p><h2 class="av-h2">{c.rolesHeading}</h2></div>
        <p class="av-lede" style="max-width:40ch">{c.how}</p>
      </div>
      <div class="av-accordion av-reveal">
        {#each c.roles as r}
          <details>
            <summary><span>{r.title} <span class="av-small" style="font-weight:500;margin-left:0.6rem">{r.location} · {r.team}</span></span><span class="av-plus" aria-hidden="true">+</span></summary>
            <div class="av-answer">
              <p>{r.body}</p>
              <a class="av-btn av-btn-secondary av-btn-sm" style="margin-top:1rem" href={`mailto:${c.cta.href.replace('mailto:', '').split('?')[0]}?subject=${encodeURIComponent(r.title)}`}>Apply by email</a>
            </div>
          </details>
        {/each}
      </div>
    </div>
  </section>
  <CtaBand title="Do not see your role?" body="Send the note anyway. A pull request against the engine is the best introduction there is." primary={c.cta} secondary={{ text: 'Good first issues', href: `${links.github}/labels/good%20first%20issue`, external: true }} />
</PageShell>
