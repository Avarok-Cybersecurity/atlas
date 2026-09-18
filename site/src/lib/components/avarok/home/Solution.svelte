<script>
  import { solution } from '$lib/content/home.js';
  import ArchDiagram from '../ArchDiagram.svelte';
</script>

<section class="av-section" id="platform">
  <div class="av-container">
    <div class="av-head av-reveal">
      <p class="av-eyebrow">{solution.eyebrow}</p>
      <h2 class="av-h2" style="max-width:24ch">{solution.title}</h2>
    </div>
    <div class="av-layers av-reveal">
      {#each solution.layers as l}
        <a class="av-card av-card-accent av-layer" class:is-cyan={l.color === 'cyan'} class:is-green={l.color === 'green'} href={l.href}>
          <p class="av-card-tag">{l.tag}</p>
          <h3>{l.name}</h3>
          <p class="av-layer-title">{l.title}</p>
          <p>{l.body}</p>
          <span class="av-link">Explore {l.name} <span class="av-arrow">→</span></span>
        </a>
      {/each}
    </div>
    <div class="av-reveal"><ArchDiagram /></div>
    <div class="av-solution-foot av-reveal">
      <h3 class="av-h3">{solution.foot.title}</h3>
      <p class="av-lede">{solution.foot.body}</p>
    </div>
  </div>
</section>

<style>
  .av-layer-title { font-weight: 600; color: var(--t1) !important; margin-bottom: 0.5rem; }
  .av-solution-foot { margin-top: 2.5rem; text-align: center; }
  .av-solution-foot .av-lede { margin-inline: auto; }
</style>
