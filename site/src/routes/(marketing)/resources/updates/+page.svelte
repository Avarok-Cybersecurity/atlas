<script>
  import PageShell from '$lib/components/avarok/PageShell.svelte';
  import PageHero from '$lib/components/avarok/PageHero.svelte';
  import CtaBand from '$lib/components/avarok/CtaBand.svelte';
  import { updates as u } from '$lib/content/resources.js';
  import { routes, links } from '$lib/content/brand.js';
  import changelog from '$lib/changelog.generated.json';
</script>

<PageShell path={routes.updates}>
  <PageHero eyebrow={u.eyebrow} title={u.title} lede={u.lede} primary={{ text: u.more.text, href: u.more.href, external: true }} color="green" />
  <section class="av-section av-section-alt">
    <div class="av-container av-narrow">
      <h2 class="av-sr">Releases</h2>
      <p class="av-kicker av-reveal" style="margin-bottom:1.5rem"><span class="av-dot"></span> Generated from CHANGELOG.md at {changelog.generated_sha} · {changelog.generated_date}</p>
      <div class="av-updates">
        {#each changelog.releases as rel}
          <section class="av-update av-reveal">
            <div>
              <p class="av-update-kind">{rel.version === 'Unreleased' ? 'On main, unreleased' : `Release ${rel.version}`}</p>
              {#if rel.date}<p class="av-small">{rel.date}</p>{/if}
            </div>
            <div>
              {#each rel.sections as s}
                <h3 class="av-h4 av-sx-{s.kind === 'Added' ? 'green' : s.kind === 'Fixed' ? 'gold' : 'cyan'}" style="margin:0.3rem 0 0.5rem">{s.kind}</h3>
                <ul class="av-sx-{s.kind === 'Added' ? 'green' : s.kind === 'Fixed' ? 'gold' : 'cyan'}">
                  {#each s.items as it}<li>{it}</li>{/each}
                </ul>
              {/each}
            </div>
          </section>
        {/each}
      </div>
    </div>
  </section>
  <CtaBand title="Want the next one on your fleet?" body="Every release passes the serve matrix before it ships. Enterprise customers pin a channel and get the receipt with the upgrade." primary={{ text: 'Book a demo', href: routes.demoForm }} secondary={{ text: 'Subscribe on GitHub', href: `${links.github}/releases`, external: true }} />
</PageShell>
