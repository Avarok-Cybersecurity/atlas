<!--
  The RUNTIME error page — distinct from the prerendered /404 document nginx
  serves via `error_page`. This one renders when the client router resolves a
  route that throws: a stale in-app link, a renamed slug. Without it,
  SvelteKit's default renders unstyled inside the layout, flush to the viewport
  edge with the site's chrome around it, which reads as a broken page rather
  than a missing one.
-->
<script>
  import { page } from '$app/state';
  import { blog } from '$lib/content.js';
  import NotFound from '$lib/components/NotFound.svelte';
</script>

<svelte:head>
  <title>{page.status === 404 ? 'Not found' : 'Something went wrong'} — {blog.name}</title>
  <meta name="robots" content="noindex" />
</svelte:head>

<NotFound status={page.status} message={page.error?.message ?? ''} />
