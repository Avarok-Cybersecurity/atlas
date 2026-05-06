<script>
  import { testimonials, firstPostUrl } from '$lib/data.js';
</script>

<section id="community" style="background: var(--bg2);">
  <div class="container">
    <div class="slabel">Community</div>
    <h2 class="stitle">What people are saying</h2>
    <p class="ssub">
      Real feedback from DGX Spark owners running Atlas. Check out our
      <a href={firstPostUrl} class="ssub-link" target="_blank" rel="noopener">first post on r/LocalLLaMA</a>
      that started it all.
    </p>
    <div class="tgrid">
      {#each testimonials as t}
        <div class="tcard">
          <blockquote>{t.quote}</blockquote>
          <div class="attr">
            <strong>{t.author}</strong>, <a href={t.sourceUrl}>{t.source}</a>
          </div>
        </div>
      {/each}
    </div>
  </div>
</section>
