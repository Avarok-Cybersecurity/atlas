<script>
  import { tour } from '$lib/content/home.js';
  import { media } from '$lib/content/media.js';
  import { moveTab } from '$lib/tablist.js';
  import VideoClip from '../VideoClip.svelte';

  let selected = $state(0);
  function onKey(e, i) {
    const next = moveTab(e.key, i, tour.tabs.length);
    if (next === null) return;
    e.preventDefault();
    selected = next;
    e.currentTarget.parentElement.querySelectorAll('[role="tab"]')[next].focus();
  }
</script>

<section class="av-section av-section-alt" id="tour">
  <div class="av-container">
    <div class="av-head av-reveal">
      <p class="av-eyebrow">{tour.eyebrow}</p>
      <h2 class="av-h2">{tour.title}</h2>
      <p class="av-lede">{tour.lede}</p>
    </div>
    <div class="av-tabs av-reveal" role="tablist" aria-label="Console workflows">
      {#each tour.tabs as t, i}
        <button type="button" role="tab" id={`tour-tab-${t.id}`} class="av-tab" aria-selected={selected === i} aria-controls={`tour-panel-${t.id}`} tabindex={selected === i ? 0 : -1} onclick={() => (selected = i)} onkeydown={(e) => onKey(e, i)}>{t.label}</button>
      {/each}
    </div>
    {#each tour.tabs as t, i}
      <div class="av-tabpanel av-reveal" role="tabpanel" id={`tour-panel-${t.id}`} aria-labelledby={`tour-tab-${t.id}`} hidden={selected !== i}>
        <div>
          <h3 class="av-h3">{t.title}</h3>
          <p class="av-lede" style="font-size:1rem">{t.body}</p>
          <a class="av-link" style="margin-top:1.25rem" href={tour.cta.href}>{tour.cta.text} <span class="av-arrow">→</span></a>
        </div>
        <div class="av-frame"><div><VideoClip clip={media.tour[t.id]} /></div></div>
      </div>
    {/each}
  </div>
</section>
