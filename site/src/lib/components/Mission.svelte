<script>
  import { mission } from '$lib/data.js';
</script>

<section id="mission">
  <div class="container">
    <div class="slabel">{mission.label}</div>
    <h2 class="stitle">{mission.title}</h2>
    <div class="mission-body">
      {#each mission.body as para}
        <p>{para}</p>
      {/each}
    </div>
  </div>
</section>
