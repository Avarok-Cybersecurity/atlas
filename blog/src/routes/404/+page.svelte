<script>
  import { blog } from '$lib/content.js';
</script>

<svelte:head>
  <title>Not found — {blog.name}</title>
  <meta name="robots" content="noindex" />
</svelte:head>

<div class="shell">
  <div class="band nf">
    <div class="mono-label">404</div>
    <h1>That page is not here.</h1>
    <p>
      It may have been renamed, or it may never have existed. The
      <a href="/">front page</a> lists everything published, and
      <a href="/rss.xml">the feed</a> has the same set in reverse order.
    </p>
  </div>
</div>

<style>
  .nf { border-bottom: 0; padding-bottom: 120px; }
  .nf a { color: var(--accent); }
</style>
