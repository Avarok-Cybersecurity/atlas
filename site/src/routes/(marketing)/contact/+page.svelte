<script>
  import PageShell from '$lib/components/avarok/PageShell.svelte';
  import PageHero from '$lib/components/avarok/PageHero.svelte';
  import { contact as c } from '$lib/content/company.js';
  import { routes, links } from '$lib/content/brand.js';
</script>

<PageShell path={routes.contact}>
  <PageHero eyebrow={c.eyebrow} title={c.title} lede={c.lede} primary={{ text: 'Book a demo', href: routes.demoForm }} />
  <section class="av-section av-section-alt">
    <div class="av-container">
      <h2 class="av-sr">Ways to reach us</h2>
      <div class="av-grid av-grid-3 av-reveal">
        {#each c.paths as p, i}
          <div class="av-card av-card-accent av-sx-{['violet', 'cyan', 'green', 'gold'][i % 4]}">
            <h3>{p.title}</h3>
            <p>{p.body}</p>
            <div class="av-stack" style="margin-top:1rem;gap:0.4rem">
              <a class="av-link" href={`mailto:${p.email}`}>{p.cta} · {p.email}</a>
              {#if p.demo}<a class="av-link" href={routes.demoForm}>Or book a working session <span class="av-arrow">→</span></a>{/if}
              {#if p.discord}<a class="av-link" href={links.discord} target="_blank" rel="noopener">Or find us in Discord ↗</a>{/if}
              {#if p.href}<a class="av-link" href={p.href} target="_blank" rel="noopener">Disclosure policy ↗</a>{/if}
            </div>
          </div>
        {/each}
      </div>
    </div>
  </section>
</PageShell>
