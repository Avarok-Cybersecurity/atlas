<!-- SPDX-License-Identifier: AGPL-3.0-only -->
<script>
  import Nav from '$lib/components/Nav.svelte';
  import Footer from '$lib/components/Footer.svelte';
  import SectionHead from '$lib/components/SectionHead.svelte';
  import { writing, postUrl } from '$lib/blog.js';
  import { headroom, signed } from '$lib/ladder.js';
  import ladder from '$lib/ladder.generated.json';

  let { data } = $props();
  const post = $derived(data.post);
  const top = headroom(ladder.rows);
  const url = $derived(postUrl(post.slug));
  const ogImage = $derived(
    post.blocks.find((b) => b.type === 'figure')?.src
      ? `https://atlasinference.io${post.blocks.find((b) => b.type === 'figure').src}`
      : 'https://atlasinference.io/og-image.png'
  );

  const graph = $derived({
    '@context': 'https://schema.org',
    '@type': 'BlogPosting',
    headline: post.title,
    description: post.description,
    datePublished: post.date,
    dateModified: post.date,
    url,
    mainEntityOfPage: url,
    image: ogImage,
    isPartOf: { '@id': 'https://atlasinference.io/#site' },
    publisher: { '@id': 'https://atlasinference.io/#org' }
  });
</script>

<svelte:head>
  <title>{post.title} · Atlas</title>
  <meta name="description" content={post.description} />
  <meta property="og:type" content="article" />
  <meta property="og:title" content={`${post.title} · Atlas`} />
  <meta property="og:description" content={post.description} />
  <meta property="og:url" content={url} />
  <meta property="og:image" content={ogImage} />
  <meta name="twitter:card" content="summary_large_image" />
  <meta name="twitter:title" content={`${post.title} · Atlas`} />
  <meta name="twitter:description" content={post.description} />
  <meta name="twitter:image" content={ogImage} />
  {@html `<script type="application/ld+json">${JSON.stringify(graph)}<\/script>`}
</svelte:head>

<Nav />

<main class="blog">
  <article class="sx-violet">
    <div class="container">
      <p class="blog-crumb"><a href={writing.indexHref}>{writing.crumb}</a></p>
      <SectionHead
        level={1}
        label={writing.label}
        title={post.title}
        sub={post.description}
        prov={post.displayDate}
      />

      {#each post.blocks as block}
        {#if block.type === 'p'}
          <p class="blog-p">{block.text}</p>
        {:else if block.type === 'h2'}
          <h2 class="blog-h2">{block.text}</h2>
        {:else if block.type === 'figure'}
          <figure class="blog-fig">
            <img
              src={block.src}
              alt={block.alt}
              width={block.width}
              height={block.height}
              loading="lazy"
            />
            <figcaption>{block.caption}</figcaption>
          </figure>
        {:else if block.type === 'scale' && top}
          <div class="scale-note">
            <p class="scale-figure">
              From C={top.from} to C={top.to}, Atlas adds
              <strong class="scale-up">{signed(top.atlas)}</strong> throughput while
              {top.label} adds <strong class="scale-flat">{signed(top.baseline)}</strong>.
            </p>
          </div>
        {:else if block.type === 'log'}
          <p class="blog-p">
            <a class="link" href={ladder.results_doc_url} target="_blank" rel="noopener"
              >The campaign log, including the rungs lost on the way</a
            >
          </p>
        {:else if block.type === 'verified'}
          <p class="blog-p blog-end">
            <a class="link" href="/#verified">The verified section of the homepage</a>
          </p>
        {/if}
      {/each}
    </div>
  </article>
</main>

<Footer />
