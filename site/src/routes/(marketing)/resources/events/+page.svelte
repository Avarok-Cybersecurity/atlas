<script>
  import PageShell from '$lib/components/avarok/PageShell.svelte';
  import PageHero from '$lib/components/avarok/PageHero.svelte';
  import CtaBand from '$lib/components/avarok/CtaBand.svelte';
  import { events as e } from '$lib/content/resources.js';
  import { routes, links } from '$lib/content/brand.js';

  const WHEN = { daily: 'Every day', request: 'By request', pending: 'Date set by MLCommons' };
  const fmt = (d) => (WHEN[d] ? WHEN[d] : new Date(d + 'T00:00:00Z').toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric', timeZone: 'UTC' }));
</script>

<PageShell path={routes.events}>
  <PageHero eyebrow={e.eyebrow} title={e.title} lede={e.lede} primary={{ text: 'Join the Discord', href: links.discord, external: true }} color="gold" />
  <section class="av-section av-section-alt">
    <div class="av-container">
      <h2 class="av-sr">Where to find us</h2>
      <div class="av-grid av-grid-3 av-reveal">
        {#each e.items as it, i}
          <div class="av-card av-card-accent av-sx-{['gold', 'violet', 'cyan'][i % 3]}">
            <p class="av-card-tag">{fmt(it.date)} · {it.place}</p>
            <h3>{it.title}</h3>
            <p>{it.body}</p>
            <a class="av-link" href={it.href} target={it.href.startsWith('http') ? '_blank' : undefined} rel={it.href.startsWith('http') ? 'noopener' : undefined}>{it.cta} {it.href.startsWith('http') ? '↗' : '→'}</a>
          </div>
        {/each}
      </div>
    </div>
  </section>
  <CtaBand title="Want us at your event?" body="Conference talks, customer dinners, a working session for your platform team. Tell us where." primary={{ text: 'Contact', href: routes.contact }} />
</PageShell>
