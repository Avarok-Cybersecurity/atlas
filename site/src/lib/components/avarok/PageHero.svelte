<!--
  The interior page opener: eyebrow, title, lede, up to two calls to action,
  and a picture beside them when the page has one.

  What the picture is comes from src/lib/content/media.js, keyed by the page's
  path, so giving a page a picture is a data change:
    - artFor(path): a still installed from the prompt pack (art.json).
    - heroClipFor(path): a clip. Product footage on the platform pages, ambient
      footage where the page is about an idea. It plays muted, in the same
      frame the front page uses, and loads nothing until it is in view.
  A still that was approved for the page keeps the hero. If the page also has
  a clip, PageClip.svelte shows it in a band directly underneath, so neither
  goes unused. A page with a clip and no still shows the clip here. A page with
  neither is text only.
-->
<script>
  import { page } from '$app/state';
  import { onMount } from 'svelte';
  import { artsFor, heroClipFor } from '$lib/content/media.js';
  import VideoClip from './VideoClip.svelte';
  let { eyebrow, title, lede = '', primary = null, secondary = null, who = '', color = 'violet', children } = $props();
  const here = $derived(page.url.pathname.replace(/\.html$/, '').replace(/\/$/, '') || '/');
  const arts = $derived(artsFor(here));
  const art = $derived(arts[0] ?? null);
  const clip = $derived(art ? null : heroClipFor(here));

  // A still that fails to arrive is asked for again, twice, a little later each
  // time. One dropped request on a busy connection otherwise leaves a hole where
  // the page's picture should be until someone thinks to refresh.
  // A page with several stills turns through them, slowly, by opacity alone, and
  // only while the frame is on screen and the visitor has not asked for less motion.
  let shown = $state(0);
  let frame = $state(null);
  onMount(() => {
    if (window.matchMedia('(prefers-reduced-motion: reduce)').matches) return;
    const timer = setInterval(() => {
      if (arts.length < 2 || document.hidden || !frame?.classList.contains('is-live')) return;
      shown = (shown + 1) % arts.length;
    }, 6000);
    return () => clearInterval(timer);
  });
  $effect(() => {
    here;
    shown = 0;
  });

  let tries = 0;
  function retry(e) {
    const img = e.currentTarget;
    if (tries >= 2 || !art) return;
    tries += 1;
    setTimeout(() => {
      img.src = `${art.src}?retry=${tries}`;
    }, 600 * tries);
  }
</script>

<section class="av-hero av-page-hero" class:has-art={!!clip || !!art}>
  <div class="av-glow av-glow-a" aria-hidden="true"></div>
  <div class="av-container av-page-hero-grid">
    <div class="av-page-hero-in av-sx-{color}">
      <p class="av-eyebrow">{eyebrow}</p>
      <h1 class="av-h1">{title}</h1>
      {#if lede}<p class="av-lede av-lede-lg">{lede}</p>{/if}
      {#if primary || secondary}
        <div class="av-hero-actions">
          {#if primary}<a class="av-btn av-btn-primary av-btn-lg" href={primary.href} target={primary.external ? '_blank' : undefined} rel={primary.external ? 'noopener' : undefined}>{primary.text} <span class="av-arrow">→</span></a>{/if}
          {#if secondary}<a class="av-btn av-btn-ghost av-btn-lg" href={secondary.href} target={secondary.external ? '_blank' : undefined} rel={secondary.external ? 'noopener' : undefined}>{secondary.text}</a>{/if}
        </div>
      {/if}
      {#if who}<p class="av-who"><span class="av-kicker">Who this is for</span>{who}</p>{/if}
      {#if children}{@render children()}{/if}
    </div>
    {#if clip}
      <div class="av-page-hero-art">
        <div class="av-frame"><div><VideoClip {clip} eager /></div></div>
        {#if clip.name.startsWith('console-')}<p class="av-video-caption">Avarok Console. Demo data, recorded from the product mockup.</p>{/if}
      </div>
    {:else if art}
      <div class="av-frame av-page-hero-art" class:is-turning={arts.length > 1} bind:this={frame}>
        <div>
          {#each arts as a, i (a.slot)}
            <img class:is-on={i === shown} src={a.src} alt={i === shown ? a.alt : ''} aria-hidden={i === shown ? undefined : 'true'} width={a.width} height={a.height} decoding="async" fetchpriority={i === 0 ? 'high' : undefined} loading={i === 0 ? undefined : 'lazy'} onerror={i === 0 ? retry : undefined} />
          {/each}
        </div>
      </div>
    {/if}
  </div>
</section>

<style>
  .av-page-hero { padding: 4rem 0 3rem; }
  .av-page-hero-in { max-width: 820px; }
  .has-art .av-page-hero-grid { display: grid; grid-template-columns: minmax(0, 1.05fr) minmax(0, 0.95fr); gap: 3rem; align-items: center; }
  .av-page-hero-art img { display: block; width: 100%; height: auto; }
  /* Every still is dark. A dark ground means a late one is a shadow, not a white hole. */
  .av-page-hero-art > div { background: #0f1216; position: relative; }
  /* Several stills: the first gives the box its height, the rest lie over it, and
     only opacity changes. */
  .is-turning img { opacity: 0; transition: opacity 1.4s var(--av-ease); }
  .is-turning img.is-on { opacity: 1; }
  .is-turning img:not(:first-child) { position: absolute; inset: 0; height: 100%; object-fit: cover; }
  /* The hero ground is tinted, and the quiet caption grey measured 4.46:1 on the
     violet tint, under the 4.5 it needs. One step darker clears it on every tint. */
  .av-page-hero-art :global(.av-video-caption) { color: var(--t2); }
  @media (max-width: 1000px) {
    .has-art .av-page-hero-grid { grid-template-columns: minmax(0, 1fr); gap: 2rem; }
  }
  .av-page-hero .av-h1 { font-size: clamp(2.2rem, 4.6vw, 3.6rem); margin-top: 0; }
  .has-art .av-h1 { font-size: clamp(2rem, 3.6vw, 3rem); }
  .av-who { margin-top: 1.8rem; padding-top: 1.2rem; border-top: 1px dashed var(--border-strong); color: var(--t2); font-size: 0.95rem; display: grid; gap: 0.4rem; max-width: 62ch; }
</style>
