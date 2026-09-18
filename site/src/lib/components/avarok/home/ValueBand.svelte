<script>
  import { value } from '$lib/content/home.js';
  import { fill } from '$lib/content/live.js';
</script>

<section class="av-section av-section-alt" id="value">
  <div class="av-container">
    <div class="av-head av-center av-reveal">
      <p class="av-eyebrow" style="justify-content:center">{value.eyebrow}</p>
      <h2 class="av-h2">{value.title}</h2>
      <p class="av-lede">{value.lede}</p>
    </div>
    <div class="av-grid av-grid-3 av-reveal">
      {#each value.tiles as t, i}
        <div class="av-tile {['av-sx-violet', 'av-sx-cyan', 'av-sx-green'][i]}">
          <span class="av-tile-label">{t.label}</span>
          <div class="av-num">{fill(t.value)}</div>
          <p>{fill(t.body)}</p>
          <a class="av-link" href={t.href}>{t.cta} <span class="av-arrow">→</span></a>
        </div>
      {/each}
    </div>
  </div>
</section>
