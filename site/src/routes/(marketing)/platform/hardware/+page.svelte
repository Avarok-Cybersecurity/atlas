<script>
  import PlatformPage from '$lib/components/avarok/PlatformPage.svelte';
  import { hardwarePage as p } from '$lib/content/platform.js';
  import { routes } from '$lib/content/brand.js';
  // The model catalogue is this page's alone, so it is imported here and not
  // through live.js, which every marketing page loads.
  import modelData from '$lib/models.generated.json';

  let vendor = $state(modelData[0]?.vendor ?? '');
  const current = $derived(modelData.find((v) => v.vendor === vendor));
</script>

<PlatformPage path={routes.hardware} page={p} color="cyan">
  {#snippet before()}
    <section class="av-section av-section-alt">
      <div class="av-container">
        <div class="av-grid av-grid-2 av-reveal">
          {#each p.verified as h}
            <div class="av-card av-card-accent av-sx-green">
              <div class="av-row" style="justify-content:space-between;margin-bottom:0.6rem">
                <p class="av-card-tag" style="margin:0">{h.chip}</p>
                <span class="av-chip av-chip-green">{h.status}</span>
              </div>
              <h3>{h.name}</h3>
              <p>{h.body}</p>
              <a class="av-link" href={h.href} target="_blank" rel="noopener">{h.cta} ↗</a>
            </div>
          {/each}
        </div>
        <div class="av-grid av-grid-4 av-reveal" style="margin-top:1.25rem">
          {#each p.bringup as h}
            <div class="av-card av-sx-gold">
              <div class="av-row" style="justify-content:space-between;margin-bottom:0.6rem">
                <p class="av-card-tag" style="margin:0">{h.chip}</p>
                <span class="av-chip av-chip-gold">{h.status}</span>
              </div>
              <h3 style="font-size:1rem">{h.name}</h3>
              <p style="font-size:0.88rem">{h.body}</p>
            </div>
          {/each}
        </div>
      </div>
    </section>
    <section class="av-section" id="models">
      <div class="av-container">
        <div class="av-head av-reveal">
          <p class="av-eyebrow">Models</p>
          <h2 class="av-h2">{p.modelsTitle}</h2>
          <p class="av-lede">{p.modelsLede}</p>
        </div>
        <div class="av-tabs av-reveal" role="tablist" aria-label="Model vendors">
          {#each modelData as v}
            <button type="button" role="tab" class="av-tab" aria-selected={vendor === v.vendor} onclick={() => (vendor = v.vendor)}>{v.vendor} <span class="av-mono" style="opacity:.6">{v.subfamilies.reduce((n, f) => n + f.recipes.length, 0)}</span></button>
          {/each}
        </div>
        {#if current}
          <div class="av-reveal">
            {#each current.subfamilies as f}
              <h3 class="av-h4" style="margin:1.4rem 0 0.7rem">{f.name}</h3>
              <div class="av-model-list">
                {#each f.recipes as r}
                  <div class="av-model">
                    <b>{r.displayName}</b>
                    <div class="av-row" style="gap:0.35rem">
                      <span class="av-chip av-chip-violet">{r.quant}</span>
                      <span class="av-chip">{r.topology}</span>
                      {#if r.params}<span class="av-chip">{r.params}</span>{/if}
                    </div>
                    <code>{r.command}</code>
                  </div>
                {/each}
              </div>
            {/each}
          </div>
        {/if}
      </div>
    </section>
  {/snippet}
</PlatformPage>
