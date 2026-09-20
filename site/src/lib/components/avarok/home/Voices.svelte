<script>
  import { voices } from '$lib/content/home.js';
</script>

<section class="av-section av-section-tight" id="voices">
  <div class="av-container">
    <div class="av-head-split av-reveal">
      <div>
        <p class="av-eyebrow av-sx-gold">{voices.eyebrow}</p>
        <h2 class="av-h3">{voices.title}</h2>
      </div>
    </div>
    <div class="av-grid av-grid-3 av-reveal">
      {#each voices.items as v}
        <figure class="av-quote">
          <blockquote>“{v.quote}”</blockquote>
          <cite>{v.who}, <a href={v.href} target="_blank" rel="noopener">{v.where}</a></cite>
        </figure>
      {/each}
    </div>
    <p class="av-fineprint av-reveal" style="margin-top:1.25rem">{voices.note}</p>
  </div>
</section>
