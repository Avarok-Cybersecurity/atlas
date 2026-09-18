<script>
  import { recognition } from '$lib/content/home.js';
</script>

<section class="av-section" id="recognition">
  <div class="av-container">
    <div class="av-head av-center av-reveal">
      <p class="av-eyebrow av-sx-gold" style="justify-content:center">{recognition.eyebrow}</p>
      <h2 class="av-h2">{recognition.title}</h2>
      <p class="av-lede">{recognition.lede}</p>
    </div>
    <div class="av-grid av-grid-4 av-reveal">
      {#each recognition.cards as c}
        <a class="av-card av-card-link" href={c.href} target="_blank" rel="noopener">
          <p class="av-card-tag">{c.org} · {c.date}</p>
          <h3 style="font-size:1rem;line-height:1.35">{c.title}</h3>
          <span class="av-link">{c.cta} ↗</span>
        </a>
      {/each}
    </div>
  </div>
</section>
