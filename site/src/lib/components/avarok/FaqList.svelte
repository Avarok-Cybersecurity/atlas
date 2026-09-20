<!--
  Native details/summary so the answers are in the prerendered HTML and the
  keyboard behaviour is the browser's. The same `items` feed the FAQPage
  structured data in the root layout, so markup and schema cannot drift.
-->
<script>
  import { faqSection } from '$lib/content/faq.js';
  let { items, eyebrow = faqSection.eyebrow, title = faqSection.title, lede = faqSection.lede, id = 'faq' } = $props();
</script>

<section class="av-section" {id}>
  <div class="av-container av-narrow">
    <div class="av-head av-reveal">
      <p class="av-eyebrow">{eyebrow}</p>
      <h2 class="av-h2">{title}</h2>
      <p class="av-lede">{lede}</p>
    </div>
    <div class="av-accordion av-reveal">
      {#each items as item}
        <details>
          <summary><span>{item.q}</span><span class="av-plus" aria-hidden="true">+</span></summary>
          <p class="av-answer">{item.a}</p>
        </details>
      {/each}
    </div>
    <p class="av-small" style="margin-top:1.25rem">
      <a class="av-link" href={faqSection.more.href}>{faqSection.more.text}</a>, or read the
      <a class="av-link" href={faqSection.repo.href} target="_blank" rel="noopener">{faqSection.repo.text} ↗</a>.
    </p>
  </div>
</section>
