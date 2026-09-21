<script>
  import PostList from '$lib/components/PostList.svelte';
  import { blog, tags, SITE } from '$lib/content.js';
  let { data } = $props();
  const title = $derived(`${data.tag.name} — ${blog.name}`);
</script>

<svelte:head>
  <title>{title}</title>
  <meta name="description" content={data.tag.blurb} />
  <meta property="og:title" content={title} />
  <meta property="og:description" content={data.tag.blurb} />
  <meta property="og:image" content="{SITE}/og-image.png" />
</svelte:head>

<div class="shell">
  <div class="band">
    <div class="tag-h">
      <svg width="17" height="27" viewBox="0 0 396 636" fill="none" stroke-width="76" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
        <path d="M38 38L358 318L38 598" stroke={data.tag.color} />
      </svg>
      <h1>{data.tag.name}</h1>
    </div>
    <p>{data.tag.blurb}</p>
  </div>

  <div class="chips" role="navigation" aria-label="Filter by category">
    <a class="chip" href="/">All</a>
    {#each Object.entries(tags) as [slug, t]}
      <a class="chip" href="/tags/{slug}" aria-current={slug === data.slug ? 'page' : undefined}>
        <span class="dot" style="--tag-c: {t.color}"></span>{t.name}
      </a>
    {/each}
  </div>

  {#if data.items.length}
    <PostList items={data.items} showTag={false} />
  {:else}
    <p class="empty">Nothing published under {data.tag.name} yet. <a href="/">Everything else</a> is on the front page.</p>
  {/if}
</div>

<style>
  .empty { padding: 56px 0; color: var(--text-3); }
</style>
