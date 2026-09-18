<!--
  The generic platform subpage: opener, feature grid, stat strip, optional
  quote, optional status note, FAQ subset, closing band. Custom blocks a page
  needs (the economics columns, the deployment models, the hardware cards)
  come in through the `extra` snippet and render between features and stats.
-->
<script>
  import PageShell from './PageShell.svelte';
  import PageHero from './PageHero.svelte';
  import PageClip from './PageClip.svelte';
  import FaqList from './FaqList.svelte';
  import CtaBand from './CtaBand.svelte';
  import { faqFor } from '$lib/content/faq.js';
  import { fill } from '$lib/content/live.js';
  import { routes } from '$lib/content/brand.js';

  let { path, page, color = 'violet', extra = null, before = null } = $props();
</script>

<PageShell {path}>
  <PageHero eyebrow={page.eyebrow} title={page.title} lede={page.lede} who={page.who} {color} />
  <PageClip {path} />

  {#if before}{@render before()}{/if}

  {#if page.features}
    <section class="av-section av-section-alt">
      <div class="av-container">
        <div class="av-grid av-grid-3 av-reveal">
          {#each page.features as f, i}
            <div class="av-card av-sx-{['violet', 'cyan', 'green', 'gold'][i % 4]}">
              <div class="av-card-icon" aria-hidden="true"><span class="av-mono">0{i + 1}</span></div>
              <h3>{f.title}</h3>
              <p>{f.body}</p>
            </div>
          {/each}
        </div>
        {#if page.status}<p class="av-status-note av-reveal">{page.status}</p>{/if}
      </div>
    </section>
  {/if}

  {#if extra}{@render extra()}{/if}

  {#if page.stats}
    <section class="av-section av-section-tight">
      <div class="av-container">
        <div class="av-grid av-grid-4 av-reveal">
          {#each page.stats as s, i}
            <div class="av-tile av-sx-{['violet', 'cyan', 'green', 'gold'][i % 4]}">
              <div class="av-num" style="font-size:2.2rem">{fill(s.value)}{#if s.unit}<span style="font-size:0.5em;font-weight:500"> {s.unit}</span>{/if}</div>
              <p>{fill(s.label)}</p>
              {#if s.href}<a class="av-link" href={s.href}>Open it <span class="av-arrow">→</span></a>{/if}
            </div>
          {/each}
        </div>
      </div>
    </section>
  {/if}

  {#if page.quote}
    <section class="av-section av-section-tight">
      <div class="av-container av-narrow">
        <figure class="av-quote av-reveal">
          <blockquote>“{page.quote.text}”</blockquote>
          <cite>{page.quote.who}{#if page.quote.href} · <a href={page.quote.href} target="_blank" rel="noopener">source</a>{/if}</cite>
        </figure>
      </div>
    </section>
  {/if}

  {#if page.disclosure}
    <section class="av-section av-section-tight">
      <div class="av-container av-narrow av-reveal">
        <a class="av-link" href={page.disclosure.href} target="_blank" rel="noopener">{page.disclosure.text} ↗</a>
      </div>
    </section>
  {/if}

  {#if page.faqTag}<FaqList items={faqFor(page.faqTag)} />{/if}

  <CtaBand title={page.ctaTitle ?? 'See it against your own workload.'} body={page.ctaBody ?? 'A side by side ladder on your hardware inside 48 hours. Your models, your criteria, your receipt.'} primary={page.cta ?? { text: 'Book a demo', href: routes.demo }} secondary={page.cta2 ?? null} />
</PageShell>
