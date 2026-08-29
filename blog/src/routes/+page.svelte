<script>
  import { posts, formatDate } from '$lib/posts.js';
  import { tags, authors, blog, SITE } from '$lib/content.js';
  import PostList from '$lib/components/PostList.svelte';

  const [featured, ...rest] = posts;
  const title = `${blog.name} — ${blog.title}`;
</script>

<svelte:head>
  <title>{title}</title>
  <meta name="description" content={blog.description} />
  <meta property="og:title" content={title} />
  <meta property="og:description" content={blog.description} />
  <meta property="og:image" content="{SITE}/og-image.png" />
  <meta name="twitter:title" content={title} />
  <meta name="twitter:description" content={blog.description} />
  <meta name="twitter:image" content="{SITE}/og-image.png" />
</svelte:head>

<div class="shell">
  <div class="masthead">
    <div class="mono-label">{blog.kicker}</div>
    <h1>{blog.title}</h1>
    <p class="lede">{blog.lede}</p>
  </div>

  <div class="chips" role="navigation" aria-label="Filter by category">
    <a class="chip" href="/" aria-current="page">All</a>
    {#each Object.entries(tags) as [slug, t]}
      <a class="chip" href="/tags/{slug}"><span class="dot" style="--tag-c: {t.color}"></span>{t.name}</a>
    {/each}
  </div>

  {#if featured}
    <a class="featured" href={featured.href}>
      <div class="feat-kicker">
        <span class="mono-label" style="color: {tags[featured.tag].color}">{tags[featured.tag].name}</span>
        <span class="sep" aria-hidden="true"></span>
        <span class="mono-label">Featured</span>
      </div>
      <h2>{featured.title}</h2>
      <p class="dek">{featured.dek}</p>
      <div class="meta">
        <span>{authors[featured.author].name}</span>
        <span class="sep" aria-hidden="true"></span>
        <span>{formatDate(featured.date)}</span>
        <span class="sep" aria-hidden="true"></span>
        <span>{featured.readingMinutes} min read</span>
      </div>
    </a>
  {/if}

  {#if rest.length}
    <PostList items={rest} />
  {/if}
</div>
