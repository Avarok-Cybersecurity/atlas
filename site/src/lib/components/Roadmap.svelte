<script>
  import { roadmap } from '$lib/data.js';
</script>

<section id="roadmap">
  <div class="container">
    <div class="slabel">{roadmap.label}</div>
    <h2 class="stitle">{roadmap.title}</h2>
    <p class="ssub">{roadmap.sub}</p>

    <div class="road-grid">
      {#each roadmap.items as item}
        <div class="road-card">
          <div class="road-top">
            <h3>{#if item.gift}<span class="gift-emoji" title="Gifted hardware">🎁</span> {/if}{item.title}</h3>
            <span class="road-status">{item.status}</span>
          </div>
          <p>{item.body}</p>
          <a href={item.url} target="_blank" rel="noopener">{item.cta} →</a>
        </div>
      {/each}
    </div>
  </div>
</section>
