<script>
  import { hero } from '$lib/content/home.js';
  import { media } from '$lib/content/media.js';
  import VideoClip from '../VideoClip.svelte';
</script>

<section class="av-hero">
  <div class="av-glow av-glow-a" aria-hidden="true"></div>
  <div class="av-glow av-glow-b" aria-hidden="true"></div>
  <div class="av-container">
    <div class="av-hero-grid">
      <div class="av-hero-copy">
        <p class="av-eyebrow av-hero-kicker">{hero.kicker}</p>
        <ul class="av-pillars" aria-label="What Avarok delivers">
          {#each hero.pillars as p, i}
            <li class="av-pill {['av-sx-violet', 'av-sx-cyan', 'av-sx-green'][i]}"><i aria-hidden="true"></i>{p}</li>
          {/each}
        </ul>
        <h1 class="av-h1">{hero.title[0]}<br /><span class="av-soft">{hero.title[1]}</span></h1>
        <p class="av-lede av-lede-lg">{hero.lede}</p>
        <div class="av-hero-actions">
          <a class="av-btn av-btn-primary av-btn-lg" href={hero.primary.href}>{hero.primary.text} <span class="av-arrow">→</span></a>
          <a class="av-btn av-btn-ghost av-btn-lg" href={hero.secondary.href}>{hero.secondary.text}</a>
        </div>
        <p class="av-hero-claim"><strong>{hero.claim}</strong> <a class="av-link" href={hero.claimHref}>{hero.claimCta} <span class="av-arrow">→</span></a></p>
      </div>
      <div class="av-hero-art">
        <div class="av-frame">
          <div><VideoClip clip={media.hero} eager /></div>
        </div>
        <p class="av-video-caption">{hero.videoCaption} <a class="av-link" href={hero.film.href}>{hero.film.text} <span class="av-arrow">→</span></a></p>
      </div>
    </div>
  </div>
</section>
