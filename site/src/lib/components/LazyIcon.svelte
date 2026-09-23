<script>
  // Loads an icon only once it is about to be seen.
  //
  // ★ WHY THIS EXISTS RATHER THAN A PLAIN IMPORT. The glyphs are decoration on
  // a panel most visitors never scroll to, and a static import would fold them
  // into the entry chunk for everyone. `import()` on intersection puts them in
  // their own chunk, fetched once, shared by every instance — the same trick
  // ChatLatticeIcon's artwork already uses for the chat modal.
  //
  // ★ IT RESERVES ITS OWN BOX. The placeholder is the exact final size, so the
  // icon arriving cannot reflow the line it sits in. A lazy image that shifts
  // its neighbours is worse than an eager one.
  //
  // ★ IT DEGRADES TO NOTHING, DELIBERATELY. No observer (older browser, SSR),
  // or a chunk that fails to load, leaves the reserved box empty — never a
  // broken-image glyph or an error. These are decorative; the text beside them
  // carries the meaning, so absence costs a reader nothing.
  import { onMount } from 'svelte';

  let { name = 'bolt', size = 18, rootMargin = '200px' } = $props();

  let Icon = $state(null);
  let host = $state(null);

  onMount(() => {
    if (typeof IntersectionObserver !== 'function') return;
    const io = new IntersectionObserver(
      (entries) => {
        if (!entries.some((e) => e.isIntersecting)) return;
        io.disconnect();
        import('./EnergyIcon.svelte')
          .then((m) => (Icon = m.default))
          .catch(() => {});   // decorative: a failed chunk is simply no icon
      },
      { rootMargin }
    );
    if (host) io.observe(host);
    return () => io.disconnect();
  });
</script>

<span
  bind:this={host}
  class="lazy-icon"
  style="width:{size}px;height:{size}px"
  aria-hidden="true"
>
  {#if Icon}<Icon {name} {size} />{/if}
</span>
