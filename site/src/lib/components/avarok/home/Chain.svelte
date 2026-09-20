<script>
  import { chain } from '$lib/content/home.js';
</script>

<section class="av-section" id="chain">
  <div class="av-container">
    <div class="av-head av-center av-reveal">
      <p class="av-eyebrow av-sx-cyan" style="justify-content:center">{chain.eyebrow}</p>
      <h2 class="av-h2">{chain.title}</h2>
    </div>
    <div class="av-chain av-reveal">
      {#each chain.steps as s}
        <div class="av-chain-step"><h3>{s.title}</h3><p>{s.body}</p></div>
      {/each}
    </div>
    <p class="av-lede av-center av-reveal" style="margin: 2rem auto 0; max-width: 60ch">{chain.foot}</p>
  </div>
</section>
