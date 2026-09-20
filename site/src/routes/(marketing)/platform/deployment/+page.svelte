<script>
  import PlatformPage from '$lib/components/avarok/PlatformPage.svelte';
  import { deploymentPage as p } from '$lib/content/platform.js';
  import { routes } from '$lib/content/brand.js';
</script>

<PlatformPage path={routes.deployment} page={p} color="gold">
  {#snippet before()}
    <section class="av-section av-section-alt">
      <div class="av-container">
        <h2 class="av-sr">Deployment models</h2>
        <div class="av-grid av-grid-3 av-reveal">
          {#each p.models as m, i}
            <div class="av-card av-card-accent av-sx-{['violet', 'cyan', 'green', 'gold'][i % 4]}">
              <h3>{m.name}</h3>
              <p>{m.body}</p>
              <p class="av-card-tag" style="margin-top:1rem;margin-bottom:0">Fits · {m.fit}</p>
            </div>
          {/each}
        </div>
      </div>
    </section>
    <section class="av-section">
      <div class="av-container">
        <div class="av-split" style="align-items:start">
          <div class="av-reveal">
            <p class="av-eyebrow">Onboarding</p>
            <h2 class="av-h2">{p.onboarding.title}</h2>
            <ol class="av-steps" style="margin-top:1.5rem">
              {#each p.onboarding.steps as s, i}
                <li class="av-step"><span class="av-step-n">0{i + 1}</span><p style="color:var(--t1);font-weight:500;align-self:center">{s}</p></li>
              {/each}
            </ol>
            <p class="av-lede" style="font-size:1rem">{p.onboarding.foot}</p>
          </div>
          <div class="av-reveal">
            <p class="av-eyebrow av-sx-cyan">Isolation</p>
            <h2 class="av-h2">{p.isolation.title}</h2>
            <table class="av-table" style="margin-top:1.5rem">
              <thead><tr><th>Tier</th><th>Isolation</th></tr></thead>
              <tbody>{#each p.isolation.rows as [tier, iso]}<tr><td>{tier}</td><td>{iso}</td></tr>{/each}</tbody>
            </table>
          </div>
        </div>
      </div>
    </section>
  {/snippet}
</PlatformPage>
