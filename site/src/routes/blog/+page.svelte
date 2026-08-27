<!-- SPDX-License-Identifier: AGPL-3.0-only -->
<script>
  import Nav from '$lib/components/Nav.svelte';
  import Footer from '$lib/components/Footer.svelte';
  import SectionHead from '$lib/components/SectionHead.svelte';
  import { writing, posts, postHref, indexUrl } from '$lib/blog.js';

  const graph = {
    '@context': 'https://schema.org',
    '@type': 'Blog',
    '@id': `${indexUrl}#blog`,
    url: indexUrl,
    name: 'Atlas Inference writing',
    description: writing.sub,
    publisher: { '@id': 'https://atlasinference.io/#org' },
    blogPost: posts.map((p) => ({
      '@type': 'BlogPosting',
      headline: p.title,
      description: p.description,
      datePublished: p.date,
      url: `https://atlasinference.io${postHref(p.slug)}`
    }))
  };
</script>

<svelte:head>
  <title>Writing · Atlas</title>
  <meta name="description" content={writing.sub} />
  <meta property="og:type" content="website" />
  <meta property="og:title" content="Writing · Atlas" />
  <meta property="og:description" content={writing.sub} />
  <meta property="og:url" content={indexUrl} />
  <meta name="twitter:title" content="Writing · Atlas" />
  <meta name="twitter:description" content={writing.sub} />
  {@html `<script type="application/ld+json">${JSON.stringify(graph)}<\/script>`}
</svelte:head>

<Nav />

<main class="blog">
  <section class="sx-violet">
    <div class="container">
      <SectionHead level={1} label={writing.label} title={writing.title} sub={writing.sub} />

      <div class="blog-index">
        {#each posts as post}
          <article class="blog-card">
            <div class="blog-card-top">
              <span class="news-tag">{post.kicker}</span>
              <span class="news-date mono">{post.displayDate}</span>
            </div>
            <h2><a href={postHref(post.slug)}>{post.title}</a></h2>
            <p>{post.description}</p>
            <a class="blog-more" href={postHref(post.slug)}>Read the page</a>
          </article>
        {/each}
      </div>
    </div>
  </section>
</main>

<Footer />
