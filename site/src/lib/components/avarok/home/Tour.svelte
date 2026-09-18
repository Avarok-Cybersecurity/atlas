<!--
  The console tour. Five tabs, each a panel of copy beside a recorded screen.

  A tab change has to paint at once. So the panels are not display:none. They
  are stacked in one grid cell and the ones not chosen are visibility:hidden,
  which keeps their posters fetched and decoded. The posters of the tabs nobody
  has opened are held back until the page has loaded and gone idle (`primed`),
  so they cost the first paint nothing, and then released together.

  Videos are dearer than posters, so only the chosen tab plays. A tab's video
  starts loading when the pointer or the focus reaches its tab, and the tab
  after the chosen one loads too, because people click through in order.
-->
<script>
  import { onMount } from 'svelte';
  import { tour } from '$lib/content/home.js';
  import { media } from '$lib/content/media.js';
  import { moveTab } from '$lib/tablist.js';
  import VideoClip from '../VideoClip.svelte';

  let selected = $state(0);
  let primed = $state(false);
  let warmed = $state([]);

  function warm(i) {
    if (!warmed.includes(i)) warmed = [...warmed, i];
  }
  function select(i) {
    selected = i;
    warm((i + 1) % tour.tabs.length);
  }
  function onKey(e, i) {
    const next = moveTab(e.key, i, tour.tabs.length);
    if (next === null) return;
    e.preventDefault();
    select(next);
    e.currentTarget.parentElement.querySelectorAll('[role="tab"]')[next].focus();
  }

  // Release the held posters once the page is idle AND the tour is within a
  // couple of screens, so a visitor who never scrolls this far fetches nothing.
  let section = $state(null);
  onMount(() => {
    let idle = false;
    let near = false;
    const sync = () => {
      if (idle && near) primed = true;
    };
    const rest = () => {
      idle = true;
      sync();
    };
    const go = () => {
      if (window.requestIdleCallback) requestIdleCallback(rest, { timeout: 2000 });
      else setTimeout(rest, 300);
    };
    if (document.readyState === 'complete') go();
    else addEventListener('load', go, { once: true });

    const io = new IntersectionObserver(
      (entries) => {
        if (!entries.some((e) => e.isIntersecting)) return;
        near = true;
        io.disconnect();
        sync();
      },
      { rootMargin: '1400px 0px' }
    );
    io.observe(section);
    return () => io.disconnect();
  });
</script>

<section class="av-section av-section-alt" id="tour" bind:this={section}>
  <div class="av-container">
    <div class="av-head av-reveal">
      <p class="av-eyebrow">{tour.eyebrow}</p>
      <h2 class="av-h2">{tour.title}</h2>
      <p class="av-lede">{tour.lede}</p>
    </div>
    <div class="av-tabs av-reveal" role="tablist" aria-label="Console workflows">
      {#each tour.tabs as t, i}
        <button
          type="button"
          role="tab"
          id={`tour-tab-${t.id}`}
          class="av-tab"
          aria-selected={selected === i}
          aria-controls={`tour-panel-${t.id}`}
          tabindex={selected === i ? 0 : -1}
          onclick={() => select(i)}
          onkeydown={(e) => onKey(e, i)}
          onpointerenter={() => warm(i)}
          onfocus={() => warm(i)}>{t.label}</button
        >
      {/each}
    </div>
    <div class="av-tourstack av-reveal">
      {#each tour.tabs as t, i}
        <div class="av-tabpanel" class:is-off={selected !== i} role="tabpanel" id={`tour-panel-${t.id}`} aria-labelledby={`tour-tab-${t.id}`} inert={selected !== i}>
          <div>
            <h3 class="av-h3">{t.title}</h3>
            <p class="av-lede" style="font-size:1rem">{t.body}</p>
            <a class="av-link" style="margin-top:1.25rem" href={tour.cta.href}>{tour.cta.text} <span class="av-arrow">→</span></a>
          </div>
          <div class="av-frame">
            <div><VideoClip clip={media.tour[t.id]} active={selected === i} warm={warmed.includes(i)} hold={selected !== i && !primed} instant={i !== 0} /></div>
          </div>
        </div>
      {/each}
    </div>
  </div>
</section>
