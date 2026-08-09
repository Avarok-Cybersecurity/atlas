<script>
  // News band. Presentation only, every card comes from `news` in data.js so the
  // page can never carry an announcement that is not in the SSOT.
  import { news } from '$lib/data.js';
</script>

<section id="news">
  <div class="container">
    <div class="slabel">{news.label}</div>
    <h2 class="stitle">{news.title}</h2>
    <p class="ssub">{news.sub}</p>

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
