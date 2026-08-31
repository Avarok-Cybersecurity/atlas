<script>
  import { onMount } from 'svelte';
  import { tags, authors, SITE, blog } from '$lib/content.js';
  import { formatDate } from '$lib/posts.js';

  let { data } = $props();
  const post = $derived(data.post);
  const author = $derived(authors[post.author]);
  const tag = $derived(tags[post.tag]);
  const Body = $derived(post.Component);

  /* The table of contents is read out of the rendered article rather than
     declared in each post's front matter. Two lists of the same headings drift
     the first time someone edits one — and this one cannot, because there is
     only ever one. */
  let article = $state(null);
  let toc = $state([]);
  let activeId = $state('');

  onMount(() => {
    if (!article) return;
    const hs = [...article.querySelectorAll('h2[id]')];
    toc = hs.map((h) => ({ id: h.id, text: h.textContent.trim() }));
    if (!hs.length) return;

    /* Highlight the heading the reader is under, not the one nearest the
       middle of the screen: a top margin of the header height and a bottom
       margin that leaves only the top slice of the viewport live means the
       active entry advances exactly when a heading crosses under the bar. */
    const io = new IntersectionObserver(
      (entries) => {
        const onscreen = entries.filter((e) => e.isIntersecting);
        if (onscreen.length) activeId = onscreen[0].target.id;
      },
      { rootMargin: '-70px 0px -75% 0px', threshold: 0 }
    );
    hs.forEach((h) => io.observe(h));
    return () => io.disconnect();
  });

  const url = $derived(post.canonical ?? `${SITE}/posts/${post.slug}`);
  /* A post may ship its own social card; an empty `og-image` means "use the
     site's". Resolved here rather than in each meta tag so the two cannot
     disagree about which image this page advertises. */
  const cardUrl = $derived(`${SITE}${post.ogImage || '/og-image.png'}`);
  const ldjson = $derived(
    JSON.stringify({
      '@context': 'https://schema.org',
      '@type': 'BlogPosting',
      headline: post.title,
      description: post.dek,
      datePublished: post.date,
      author: { '@type': 'Person', name: author.name },
      publisher: { '@type': 'Organization', name: 'Atlas Inference', url: 'https://atlasinference.io/' },
      mainEntityOfPage: url,
      url,
      dateModified: post.updated ?? post.date,
      articleSection: tag.name,
      keywords: post.keywords,
      image: `${SITE}${post.ogImage || '/og-image.png'}`,
      timeRequired: `PT${post.readingMinutes}M`,
      isAccessibleForFree: true
      // JSON.stringify does not escape "<", so a closing script tag anywhere in
      // a dek would end the emitted block early and spill markup into the page.
    }).replace(/</g, '\\u003c')
  );
</script>

<svelte:head>
  <title>{post.title} — {blog.name}</title>
  <meta name="description" content={post.dek} />
  <meta property="og:type" content="article" />
  <meta property="og:title" content={post.title} />
  <meta property="og:description" content={post.dek} />
  <meta property="og:image" content={cardUrl} />
  <meta property="og:image:alt" content={post.title} />
  <meta property="article:published_time" content={post.date} />
  {#if post.updated}<meta property="article:modified_time" content={post.updated} />{/if}
  <meta property="article:author" content={author.name} />
  <meta property="article:section" content={tag.name} />
  {#each post.categories as c}<meta property="article:tag" content={tags[c].name} />{/each}
  {#if post.keywords.length}<meta name="keywords" content={post.keywords.join(', ')} />{/if}
  <meta name="twitter:title" content={post.title} />
  <meta name="twitter:description" content={post.dek} />
  <meta name="twitter:image" content={cardUrl} />
  <!-- KaTeX's stylesheet is ~23 KB and render-blocking. Loading it globally
       would charge every post — including every Svelte post, which has no
       math at all — for a feature it does not use. `hasMath` is computed from
       the source at build time, so this link appears on exactly the pages
       that need it. -->
  {#if post.hasMath}<link rel="stylesheet" href="/katex/katex.min.css" />{/if}
  {@html `<script type="application/ld+json">${ldjson}<\/script>`}
</svelte:head>

<div class="shell">
  <div class="post-head">
    <div class="breadcrumb mono-label">
      <a href="/">Blog</a>
      <span aria-hidden="true">/</span>
      <a href="/tags/{post.tag}" style="color: {tag.color}">{tag.name}</a>
    </div>
    <h1 class="post-title">{post.title}</h1>
    <p class="post-dek">{post.dek}</p>
    <div class="byline">
      <span class="avatar" aria-hidden="true">{author.initials}</span>
      <a href="/authors/{post.author}">{author.name}</a>
      <span class="sep" aria-hidden="true"></span>
      <time class="mono-label byline-meta" datetime={post.date}>{formatDate(post.date)}</time>
      <span class="sep" aria-hidden="true"></span>
      <span class="mono-label byline-meta">{post.readingMinutes} min read</span>
    </div>
  </div>

  <div class="post-grid">
    <article class="prose" bind:this={article}>
      <Body />
    </article>

    <!-- DOM-after the article, so it never precedes it for a screen reader or
         on a narrow viewport. -->
    <aside class="rail">
      <div class="rail-in">
        {#if toc.length}
          <div class="mono-label">On this page</div>
          <ul class="toc">
            {#each toc as h (h.id)}
              <li><a href="#{h.id}" data-active={h.id === activeId}>{h.text}</a></li>
            {/each}
          </ul>
        {/if}
        <div class="rail-foot">
          <a href="/tags/{post.tag}">More in {tag.name}</a>
          <a href="/authors/{post.author}">More by {author.name}</a>
          <a href="/rss.xml">RSS feed</a>
        </div>
      </div>
    </aside>
  </div>

  {#if data.newer || data.older}
    <nav class="postnav" aria-label="Adjacent posts">
      {#if data.older}
        <a class="prev" href={data.older.href}>
          <span class="postnav-label mono-label">
            <svg class="pn-chev back" viewBox="0 0 396 636" fill="none" stroke-width="76"
              stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
              <path d="M38 38L358 318L38 598" stroke="var(--ch-violet)" />
            </svg>Previous
          </span>
          <span class="postnav-title">{data.older.title}</span>
        </a>
      {:else}<span></span>{/if}
      {#if data.newer}
        <a class="next" href={data.newer.href}>
          <span class="postnav-label mono-label">
            Next<svg class="pn-chev" viewBox="0 0 396 636" fill="none" stroke-width="76"
              stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
              <path d="M38 38L358 318L38 598" stroke="var(--ch-violet)" />
            </svg>
          </span>
          <span class="postnav-title">{data.newer.title}</span>
        </a>
      {:else}<span></span>{/if}
    </nav>
  {/if}
</div>

<style>
  /* The byline's date and read time are metadata, not labels: they keep the
     mono face but drop the uppercase tracking that would shout them. */
  .byline-meta { letter-spacing: .04em; text-transform: none; font-size: 12.5px; }

  /* The same chevron the section headings draw, pointing the way the link
     goes. Decorative — the label already says Previous or Next, so the mark is
     aria-hidden and adds nothing for a screen reader to read twice. */
  .postnav-label { display: inline-flex; align-items: center; gap: 6px; }
  .pn-chev { width: 7px; height: auto; overflow: visible; flex: none; }
  .pn-chev.back { transform: rotate(180deg); }
</style>
