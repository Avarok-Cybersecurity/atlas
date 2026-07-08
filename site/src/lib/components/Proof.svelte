<script>
  import { proof } from '$lib/data.js';
</script>

<section id="proof" class="proof">
  <div class="container">
    <div class="proof-strip">
      {#each proof.items as item}
        <a class="proof-item" href={item.url} target="_blank" rel="noopener">
          <span class="proof-tick" aria-hidden="true">✓</span>
          <span>{item.text}</span>
        </a>
      {/each}
    </div>
  </div>
</section>
