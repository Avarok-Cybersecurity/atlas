<script>
  import PostList from '$lib/components/PostList.svelte';
  import { blog, SITE } from '$lib/content.js';
  let { data } = $props();
  const title = $derived(`${data.author.name} — ${blog.name}`);
</script>

<svelte:head>
  <title>{title}</title>
  <meta name="description" content={data.author.bio} />
  <meta property="og:title" content={title} />
  <meta property="og:description" content={data.author.bio} />
  <meta property="og:image" content="{SITE}/og-image.png" />
</svelte:head>

<div class="shell">
  <div class="band">
    <div class="tag-h">
      <span class="avatar" aria-hidden="true">{data.author.initials}</span>
      <h1>{data.author.name}</h1>
    </div>
    <p class="role mono-label">{data.author.role}</p>
    <p>{data.author.bio}</p>
  </div>
  <PostList items={data.items} />
</div>

<style>
  .role { margin-top: 10px; }
</style>
