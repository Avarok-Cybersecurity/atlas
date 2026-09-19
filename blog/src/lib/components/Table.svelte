<!-- Wraps a table so it scrolls inside its own box instead of widening the
     reading column on a phone. -->
<script>
  let { caption = '', children } = $props();
</script>

<figure>
  <div class="tbl-wrap">
    <table>{@render children()}</table>
  </div>
  {#if caption}<figcaption>{caption}</figcaption>{/if}
</figure>
