<script>
  // Energy glyph set for the Cost tab. House icon convention (see
  // GithubIcon.svelte): 24-unit viewBox, `currentColor` so the light/dark swap
  // is inherited rather than re-declared, `aria-hidden` because every one of
  // these sits beside text that already says the same thing — an icon that
  // needs a label is a label.
  //
  // Stroke-only, 1.5 weight, round caps: it matches the chart line weight, and
  // it stays legible at 16px where a filled glyph turns into a blob.
  //
  // ★ THIS MODULE IS LOADED LAZILY. It is reached only through LazyIcon.svelte,
  // which imports it on intersection, so these paths are a separate chunk and
  // cost nothing on a page that never scrolls to the cost panel. Importing it
  // directly defeats that — use LazyIcon.
  let { name = 'bolt', size = 18 } = $props();

  // viewBox 0 0 24 24 throughout, so the set is drop-in interchangeable.
  const PATHS = {
    // energy / power draw
    bolt: 'M13 2 4.5 13.5H11l-1 8.5L19.5 10.5H13z',
    // efficiency — a gauge needle past centre
    gauge: 'M3.5 18a9 9 0 1 1 17 0M12 18l4.2-5.6',
    // savings — a leaf, used only where the figure is a saving
    leaf: 'M20 4C10 4 4 9 4 16c0 2.2.8 3.4.8 3.4M4.8 19.4C12 20 20 15 20 4M4.8 19.4C7 14 11 10.5 15.5 8.5',
    // over time
    clock: 'M12 21a9 9 0 1 1 0-18 9 9 0 0 1 0 18M12 7v5.2l3.4 2'
  };
  const d = $derived(PATHS[name] ?? PATHS.bolt);
</script>

<svg
  width={size}
  height={size}
  viewBox="0 0 24 24"
  fill="none"
  aria-hidden="true"
  focusable="false"
  class="energy-icon"
>
  <path d={d} stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round" />
</svg>
