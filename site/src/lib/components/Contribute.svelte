<script>
  // Ledger entry 08. Contribute and the roadmap were two sections that shared
  // card grammar, shared an audience, and both ended in a repo link. They are
  // one entry with two card rows: how to join, then what is being built.
  import { contribute, roadmap } from '$lib/data.js';
  import SectionHead from './SectionHead.svelte';
</script>

<section id="contribute" class="section-alt sx-violet">
  <div class="container">
    <SectionHead label={contribute.label} title={contribute.title} sub={contribute.sub} />

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

    <div class="entry-row" id="roadmap">
      <h3 class="entry-row-title">{roadmap.rowTitle}</h3>
      <p class="entry-row-sub">{roadmap.rowSub}</p>
    </div>

    <div class="road-grid">
      {#each roadmap.items as item}
        <div class="road-card">
          <div class="road-top">
            <h3>{item.title}</h3>
            <span class="road-status">{item.status}</span>
          </div>
          <p>{item.body}</p>
          <a href={item.url} target="_blank" rel="noopener">{item.cta} →</a>
        </div>
      {/each}
    </div>
  </div>
</section>
