<script>
  import { problem } from '$lib/content/home.js';
</script>

<section class="av-section av-section-alt" id="problem">
  <div class="av-container">
    <div class="av-head av-reveal">
      <p class="av-eyebrow">{problem.eyebrow}</p>
      <h2 class="av-h2" style="max-width:22ch">{problem.title}</h2>
      <p class="av-lede">{problem.lede}</p>
    </div>
    <div class="av-cols av-reveal">
      <div class="av-col">
        <h3>{problem.left.title}</h3>
        <ul>{#each problem.left.items as it}<li>{it}</li>{/each}</ul>
      </div>
      <div class="av-cols-mid" aria-hidden="true"><span>≠</span></div>
      <div class="av-col av-col-right">
        <h3>{problem.right.title}</h3>
        <ul>{#each problem.right.items as it}<li>{it}</li>{/each}</ul>
      </div>
    </div>
    <div class="av-bridge av-reveal">
      <h3>{problem.bridge.title}</h3>
      <p>{problem.bridge.body}</p>
    </div>
  </div>
</section>
