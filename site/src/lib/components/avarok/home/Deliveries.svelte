<script>
  import { deliveries } from '$lib/content/home.js';
</script>

<section class="av-section av-section-alt" id="deliveries">
  <div class="av-container">
    <div class="av-split" style="align-items:start">
      <div class="av-reveal">
        <p class="av-eyebrow av-sx-green">{deliveries.eyebrow}</p>
        <h2 class="av-h2">{deliveries.title}</h2>
        <p class="av-lede">{deliveries.lede}</p>
      </div>
      <div class="av-steps av-reveal">
        {#each deliveries.steps as s, i}
          <div class="av-step av-sx-green">
            <span class="av-step-n">0{i + 1}</span>
            <div>
              <p class="av-step-when">{s.when}</p>
              <h3>{s.title}</h3>
              <p>{s.body}</p>
            </div>
          </div>
        {/each}
      </div>
    </div>
  </div>
</section>
