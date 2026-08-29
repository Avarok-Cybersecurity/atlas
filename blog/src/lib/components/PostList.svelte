<script>
  import { tags, authors } from '$lib/content.js';
  import { formatDate } from '$lib/posts.js';
  let { items, showTag = true } = $props();
</script>

<ul class="list">
  {#each items as p (p.slug)}
    <li>
      <a class="entry" href={p.href} style="--tag-c: {tags[p.tag].color}">
        <span class="entry-date">{formatDate(p.date)}</span>
        <span class="entry-body">
          <h3>{p.title}</h3>
          <p>{p.dek}</p>
        </span>
        <span class="entry-side">
          {#if showTag}<span class="entry-tag">{tags[p.tag].name}</span>{/if}
          <span class="entry-read">{p.readingMinutes} min · {authors[p.author].initials}</span>
        </span>
      </a>
    </li>
  {/each}
</ul>

<style>
  .entry-side { display: flex; flex-direction: column; align-items: flex-end; gap: 6px; }
  @media (max-width: 820px) {
    .entry-side { flex-direction: row; align-items: baseline; gap: 10px; }
  }
</style>
