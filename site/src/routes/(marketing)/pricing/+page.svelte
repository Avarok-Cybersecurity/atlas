<script>
  import PageShell from '$lib/components/avarok/PageShell.svelte';
  import PageHero from '$lib/components/avarok/PageHero.svelte';
  import PaybackCalculator from '$lib/components/avarok/PaybackCalculator.svelte';
  import FaqList from '$lib/components/avarok/FaqList.svelte';
  import CtaBand from '$lib/components/avarok/CtaBand.svelte';
  import { pricingHero, tiers, anchors, contractEconomics, paybackCopy, pricingCta } from '$lib/content/pricing.js';
  import { faqFor } from '$lib/content/faq.js';
  import { routes } from '$lib/content/brand.js';
</script>

<PageShell path={routes.pricing}>
  <PageHero eyebrow={pricingHero.eyebrow} title={pricingHero.title} lede={pricingHero.lede} color="green">
    <p class="av-kicker" style="margin-top:1.4rem"><span class="av-dot"></span> {pricingHero.stamp}</p>
  </PageHero>

  <section class="av-section av-section-alt av-section-tight" id="tiers">
    <div class="av-container">
      <div class="av-grid av-grid-4 av-reveal av-tiers">
        {#each tiers as t}
          <div class="av-card av-tier" class:is-featured={t.featured}>
            <p class="av-card-tag">{#if t.featured}<span class="av-chip av-chip-violet av-tier-flag">Most fleets start here</span><br />{/if}{#if t.badge}<span class="av-chip av-chip-gold av-tier-flag">{t.badge}</span><br />{/if}{t.name}{#if t.proposed}<span class="av-evidence is-proposed">PROPOSED</span>{/if}</p>
            <div class="av-tier-price"><span class="av-num av-num-plain">{t.price}</span><span class="av-small">{t.per}</span></div>
            <p>{t.blurb}</p>
            <ul class="av-list-check av-sx-{t.featured ? 'violet' : 'green'}" style="margin-top:1rem">{#each t.includes as i}<li>{i}</li>{/each}</ul>
            <a class="av-btn {t.featured ? 'av-btn-primary' : 'av-btn-secondary'}" style="margin-top:1.2rem;width:100%" href={t.cta.href}>{t.cta.text}</a>
          </div>
        {/each}
      </div>
    </div>
  </section>

  <section class="av-section">
    <div class="av-container">
      <div class="av-split" style="align-items:start">
        <div class="av-reveal">
          <p class="av-eyebrow">{anchors.eyebrow}</p>
          <h2 class="av-h2">{anchors.title}</h2>
          <p class="av-lede">{anchors.body}</p>
          <table class="av-table" style="margin-top:1.5rem">
            <thead><tr><th>Product</th><th>List</th><th>Basis</th></tr></thead>
            <tbody>
              {#each anchors.rows as r}
                <tr class:is-accent={r.accent}><td>{r.name}<br /><span class="av-small">{r.note}</span></td><td class="av-mono">{r.price}</td><td>{r.per}</td></tr>
              {/each}
            </tbody>
          </table>
          <p class="av-small" style="margin-top:0.8rem">{anchors.foot}</p>
        </div>
        <div class="av-reveal">
          <p class="av-eyebrow av-sx-green">{contractEconomics.eyebrow}</p>
          <h2 class="av-h2">{contractEconomics.title}</h2>
          <p class="av-lede">{contractEconomics.body}</p>
          <div class="av-stack" style="margin-top:1.5rem">
            {#each contractEconomics.rows as r, i}
              <div class="av-tile av-acv" style="--w:{[34, 60, 100][i]}%">
                <div class="av-row" style="justify-content:space-between"><span class="av-tile-label">{r.gpus}</span><span class="av-num av-num-plain" style="font-size:1.6rem">{r.acv}</span></div>
                <div class="av-acv-bar"><span></span></div>
              </div>
            {/each}
          </div>
        </div>
      </div>
    </div>
  </section>

  <section class="av-section av-section-alt" id="payback-section">
    <div class="av-container">
      <div class="av-head av-reveal">
        <p class="av-eyebrow av-sx-green">{paybackCopy.eyebrow}</p>
        <h2 class="av-h2">{paybackCopy.title}</h2>
        <p class="av-lede">{paybackCopy.lede}</p>
      </div>
      <div class="av-reveal"><PaybackCalculator /></div>
    </div>
  </section>

  <FaqList items={faqFor('pricing')} />
  <CtaBand eyebrow={pricingCta.eyebrow} title={pricingCta.title} body={pricingCta.body} primary={pricingCta.primary} secondary={pricingCta.secondary} />
</PageShell>

<style>
  .av-tier { display: flex; flex-direction: column; }
  .av-tier p:not(.av-card-tag) { flex: 0 0 auto; }
  .av-tier .av-list-check { flex: 1 0 auto; }
  .av-tier.is-featured { border-color: var(--accent); box-shadow: 0 0 0 1px var(--accent), var(--av-shadow); }
  .av-tier-flag { display: inline-flex; margin-bottom: 0.6rem; }
  .av-tier-price { display: grid; gap: 0.2rem; margin: 0.4rem 0 0.9rem; }
  .av-tier-price .av-num { font-size: 2.2rem; }
  .av-acv-bar { height: 8px; border-radius: 4px; background: var(--bg2); margin-top: 0.7rem; overflow: hidden; }
  .av-acv-bar span { display: block; height: 100%; width: var(--w); background: linear-gradient(90deg, var(--ch-violet), var(--ch-cyan)); border-radius: 4px; }
  @media (max-width: 1080px) { .av-tiers { grid-template-columns: repeat(2, minmax(0, 1fr)); } }
  @media (max-width: 640px) { .av-tiers { grid-template-columns: minmax(0, 1fr); } }
</style>
