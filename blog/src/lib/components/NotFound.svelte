<!-- The body of both 404 surfaces: the prerendered document nginx serves and
     the runtime error page the client router renders. One copy. -->
<script>
  let { status = 404, message = '' } = $props();
</script>

<div class="shell">
  <div class="band nf">
    <div class="mono-label">{status}</div>
    <h1>{status === 404 ? 'That page is not here.' : 'Something went wrong.'}</h1>
    <p>
      {#if status === 404}
        It may have been renamed, or it may never have existed. The
        <a href="/">front page</a> lists everything published, and
        <a href="/rss.xml">the feed</a> has the same set in reverse order.
      {:else}
        {message || 'No further detail was reported.'} Try the <a href="/">front page</a>.
      {/if}
    </p>
  </div>
</div>

<style>
  .nf { border-bottom: 0; padding-bottom: 120px; }
  .nf a { color: var(--accent); }
</style>
