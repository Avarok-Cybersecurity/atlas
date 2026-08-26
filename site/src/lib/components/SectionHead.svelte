<script>
  // The ledger rail. Every section header on the page is one entry in a chain
  // of custody: index, chevron in the section's colour, entry name, then the
  // provenance stamp for whatever the section claims.
  //
  // The index and the name are parsed out of the single `label` string that
  // already lives in data.js ("// 03 · verified"), so the numbering keeps one
  // source. A label with no number still renders — it just has no index.
  // `level` lets a page make its first entry the document's h1. Sections on
  // the home page stay h2 under the hero's h1; a standalone page has no hero,
  // and a document whose headings start at h2 fails heading-order.
  let { label, title, sub = '', prov = '', provUrl = '', level = 2 } = $props();

  const parsed = /^\s*\/\/\s*(?:(\d+)\s*·\s*)?(.+?)\s*$/.exec(label);
  const index = parsed?.[1] ?? '';
  const name = parsed?.[2] ?? label;
</script>

<div class="ledger-rail">
  {#if index}<span class="ledger-no">{index}</span>{/if}
  <svg class="ledger-chev" viewBox="0 0 26 44" width="13" height="22" aria-hidden="true">
    <path
      d="M6 6L20 22L6 38"
      fill="none"
      stroke="currentColor"
      stroke-width="8"
      stroke-linecap="round"
      stroke-linejoin="round"
    />
  </svg>
  <span class="ledger-name">{name}</span>
  <span class="ledger-line"></span>
  {#if prov}
    {#if provUrl}
      <a class="prov" href={provUrl} target="_blank" rel="noopener">{prov}</a>
    {:else}
      <span class="prov">{prov}</span>
    {/if}
  {/if}
</div>

<svelte:element this={level === 1 ? 'h1' : 'h2'} class="stitle">{title}</svelte:element>
{#if sub}<p class="ssub">{sub}</p>{/if}
