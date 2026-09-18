<!--
  A product clip. Poster first, video only when it scrolls near the viewport
  and the visitor has not asked for reduced motion or reduced data. The poster
  is the LCP candidate on the front page, so it is a plain <img> with explicit
  dimensions and no lazy attribute when `eager` is set.

  `preload="none"` on purpose: a page with four clips must not fetch four
  videos on load. Lighthouse scores this page against a perfect budget.
-->
<script>
  import { onMount } from 'svelte';
  let { clip, eager = false, controls = false, class: klass = '' } = $props();
  let el = $state(null);
  let ready = $state(false);
  let playing = $state(false);

  onMount(() => {
    const reduced = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
    const saveData = navigator.connection && navigator.connection.saveData;
    if (reduced || saveData) return;
    const io = new IntersectionObserver(
      (entries) => {
        for (const e of entries) {
          if (e.isIntersecting) {
            ready = true;
            io.disconnect();
          }
        }
      },
      { rootMargin: '200px 0px' }
    );
    io.observe(el);
    return () => io.disconnect();
  });

  function onCanPlay(e) {
    const v = e.currentTarget;
    v.play().then(() => (playing = true)).catch(() => {});
  }
</script>

<div class="av-video {klass}" bind:this={el}>
  <img src={clip.poster} alt={clip.alt} width={clip.width} height={clip.height} loading={eager ? 'eager' : 'lazy'} fetchpriority={eager ? 'high' : undefined} decoding="async" class:is-hidden={playing} />
  {#if ready}
    <video muted playsinline loop={clip.loop} preload="none" {controls} aria-label={clip.alt} oncanplay={onCanPlay}>
      <!-- mp4 first on purpose. For flat interface footage H.264 comes out
           smaller than VP9 at the same legibility (measured: 0.40 MB against
           0.60 MB for the hero), and every browser that plays the webm plays
           the mp4. The webm stays for the few builds that ship without H.264. -->
      <source src={clip.mp4} type="video/mp4" />
      <source src={clip.webm} type="video/webm" />
    </video>
  {/if}
</div>

<style>
  img { transition: opacity 0.4s; z-index: 1; }
  img.is-hidden { opacity: 0; pointer-events: none; }
</style>
