<script>
  import { contribute } from '$lib/data.js';
</script>

<section id="contribute" class="section-alt">
  <div class="container">
    <div class="slabel">{contribute.label}</div>
    <h2 class="stitle">{contribute.title}</h2>
    <p class="ssub">{contribute.sub}</p>

    <div class="contrib-grid">
      {#each contribute.paths as p}
        <div class="contrib-card">
          <h3>{p.title}</h3>
          <p>{p.body}</p>
          <a href={p.url} target="_blank" rel="noopener">{p.cta} →</a>
        </div>
      {/each}
    </div>
    <p class="cla-line">{contribute.cla}</p>
  </div>
</section>
