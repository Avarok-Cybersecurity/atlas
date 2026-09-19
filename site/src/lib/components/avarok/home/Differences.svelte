<script>
  import { differences } from '$lib/content/home.js';
</script>

<section class="av-section av-section-alt" id="why">
  <div class="av-container">
    <div class="av-head av-reveal">
      <p class="av-eyebrow">{differences.eyebrow}</p>
      <h2 class="av-h2">{differences.title}</h2>
      <p class="av-lede">{differences.lede}</p>
    </div>
    <div class="av-diffs">
      {#each differences.items as d}
        <article class="av-diff av-reveal av-sx-{d.color}">
          <div class="av-diff-n"><span class="av-step-n">{d.n}</span><span class="av-role">{d.pillar}</span></div>
          <div>
            <h3 class="av-h3">{d.title}</h3>
            <p class="av-body">{d.body}</p>
            <p class="av-question">{d.question}</p>
            <a class="av-link" style="margin-top:1rem" href={d.href}>Go deeper on {d.pillar.toLowerCase()} <span class="av-arrow">→</span></a>
          </div>
        </article>
      {/each}
    </div>
  </div>
</section>

<style>
  .av-diffs { display: grid; gap: 1.5rem; }
  .av-diff { display: grid; grid-template-columns: 140px 1fr; gap: 2rem; padding: 2rem 0; border-top: 1px solid var(--border); }
  .av-diff-n { display: grid; gap: 0.5rem; align-content: start; }
  .av-diff-n .av-step-n { font-size: 3rem; }
  @media (max-width: 720px) { .av-diff { grid-template-columns: minmax(0, 1fr); gap: 0.8rem; } .av-diff-n { grid-auto-flow: column; align-items: baseline; justify-content: start; gap: 1rem; } .av-diff-n .av-step-n { font-size: 2rem; } }
</style>
