<script>
  // Ledger entry 09. Rendered from `faq` in data.js, the same export that feeds
  // the FAQPage structured data in +layout.svelte — so the markup can only ever
  // claim answers a visitor can actually read.
  //
  // <details>/<summary> rather than a JS accordion: keyboard and screen-reader
  // behaviour is native, it costs no runtime, and the answers are in the
  // prerendered HTML whether or not a crawler executes script.
  import { faq } from '$lib/data.js';
  import SectionHead from './SectionHead.svelte';
</script>

<section id="faq" class="sx-cyan">
  <div class="container">
    <SectionHead label={faq.label} title={faq.title} sub={faq.sub} />

    <div class="faq-list">
      {#each faq.items as item}
        <details class="faq-item">
          <summary>
            <span class="faq-q">{item.q}</span>
            <span class="faq-mark" aria-hidden="true"></span>
          </summary>
          <p class="faq-a">{item.a}</p>
        </details>
      {/each}
    </div>
  </div>
</section>
