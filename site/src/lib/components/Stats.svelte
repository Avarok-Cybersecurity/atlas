<script>
  import { heroStats } from '$lib/data.js';
</script>

<section style="padding-top: 0;">
  <div class="container">
    <div class="stats">
      {#each heroStats as stat}
        <div class="stat">
          <div class="stat-value">{stat.value}</div>
          <div class="stat-label">{stat.label}</div>
        </div>
      {/each}
    </div>
  </div>
</section>
