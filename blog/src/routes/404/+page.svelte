<!--
  The PRERENDERED 404 document. adapter-static writes this route to
  `build/404.html`, which is the file the vhost names in `error_page 404`, so
  it is what nginx returns for a URL that never reaches the app at all. The
  runtime counterpart is +error.svelte; they are separate because they are
  reached by different mechanisms, and the copy is shared through NotFound.
-->
<script>
  import NotFound from '$lib/components/NotFound.svelte';
  import { blog } from '$lib/content.js';
</script>

<svelte:head>
  <title>Not found — {blog.name}</title>
  <meta name="robots" content="noindex" />
</svelte:head>

<NotFound />
