<script>
  import { roadmap } from '$lib/data.js';
</script>

<section id="roadmap">
  <div class="container">
    <div class="slabel">Roadmap</div>
    <h2 class="stitle">Built for the community</h2>
    <p class="ssub">
      We don't chase every architecture at once. We do each one properly, with kernels that hit the
      hardware ceiling rather than emulate around it.
    </p>
    <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); gap: 1.25rem;">
      {#each roadmap as item}
        <div class="card">
          <div class="card-icon">{item.icon}</div>
          <h3>{item.title}</h3>
          <p>{item.body}</p>
        </div>
      {/each}
    </div>
  </div>
</section>
