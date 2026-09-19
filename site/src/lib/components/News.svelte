<script>
  // News band. Presentation only, every card comes from `news` in data.js so the
  // page can never carry an announcement that is not in the SSOT.
  import { news } from '$lib/data.js';
  import SectionHead from './SectionHead.svelte';
</script>

<section id="news" class="sx-gold">
  <div class="container">
    <SectionHead
      label={news.label}
      title={news.title}
      sub={news.sub}
      prov="every card links its primary source"
    />

    <div class="news-grid">
      {#each news.items as item}
        <article class="news-card" class:is-featured={item.featured}>
          {#if item.featured}<span class="news-accent" aria-hidden="true"></span>{/if}
          <div class="news-top">
            <span class="news-tag">{item.tag}</span>
            <span class="news-date mono">{item.date}</span>
          </div>
          <h3>{item.title}</h3>
          <p>{item.body}</p>
          <a class="news-link" href={item.url} target="_blank" rel="noopener">{item.cta} →</a>
        </article>
      {/each}
    </div>
  </div>
</section>
