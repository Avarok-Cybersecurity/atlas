<script>
  import { hardware } from '$lib/data.js';
  import SectionHead from './SectionHead.svelte';
</script>

<section id="hardware" class="section-alt sx-cyan">
  <div class="container">
    <SectionHead
      label={hardware.label}
      title={hardware.title}
      sub={hardware.sub}
    />

    <div class="hw-grid">
      {#each hardware.cards as c}
        <div class="hw-card">
          <div class="hw-top">
            <div>
              <h3>{c.name}</h3>
              <div class="hw-chip mono">{c.chip}</div>
            </div>
            <span class="hw-status {c.status}"><span class="dot"></span>{c.statusText}</span>
          </div>
          <p>{c.body}</p>
          <div class="hw-links">
            <a href={c.cta.url} target="_blank" rel="noopener">{c.cta.text} →</a>
            {#if c.scale}
              <a class="hw-scale" href={c.scale.url} target="_blank" rel="noopener">{c.scale.text} ↗</a>
            {/if}
          </div>
        </div>
      {/each}
    </div>
  </div>
</section>
