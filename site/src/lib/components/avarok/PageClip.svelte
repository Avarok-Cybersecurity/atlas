<!--
  The product clip for a page whose hero is already taken by a still.

  A page can have both: a generated still from the prompt pack (art.json) and
  product footage (heroClips in media.js). The still keeps the hero, because it
  was approved for that page, and the footage gets this band directly under it,
  so neither asset goes unused. On a page with footage and no still the hero
  shows the footage and this renders nothing. `path` is the page's route.
-->
<script>
  import { artFor, heroClipFor } from '$lib/content/media.js';
  import VideoClip from './VideoClip.svelte';
  let { path } = $props();
  const clip = $derived(artFor(path) ? heroClipFor(path) : null);
</script>

{#if clip}
  <section class="av-section av-section-tight av-page-clip">
    <div class="av-container">
      <div class="av-split av-split-wide av-reveal" style="align-items:center">
        <div>
          <p class="av-eyebrow">In the console</p>
          <p class="av-lede" style="margin-top:0.6rem">{clip.alt}</p>
        </div>
        <div>
          <div class="av-frame"><div><VideoClip {clip} /></div></div>
          <p class="av-video-caption">Avarok Console. Demo data, recorded from the product mockup.</p>
        </div>
      </div>
    </div>
  </section>
{/if}
