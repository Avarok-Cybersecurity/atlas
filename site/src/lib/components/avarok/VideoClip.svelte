<!--
  A product clip. Two behaviours, chosen by `controls`:

  A loop (the default): poster first, then the clip plays by itself, muted, once
  it is on screen, and pauses when it scrolls away. The poster is the largest
  paint on the front page, so it is a plain <img> with explicit dimensions and
  no lazy attribute when `eager` is set.

  A film (`controls`): the native player with the poster as its poster frame,
  and nothing laid over it, so the play button can be pressed. It never starts
  by itself.

  Nothing is fetched until the clip is near the viewport AND the page has
  finished loading and gone idle. A page with six clips must not fetch six
  videos on load, and Lighthouse scores these pages against a perfect budget.
  Visitors who ask for reduced motion or reduced data keep the poster.

  Three props exist for clips that share one place on the page, like the tabs
  of the console tour (home/Tour.svelte):
    active   false keeps the clip from playing. When it turns false the clip
             rewinds and the poster comes back, so the next visit starts clean.
    warm     fetch the video now, paused, because it is likely to be asked for.
    hold     render no poster yet. Tour holds the posters of the tabs nobody
             has opened until the page is idle, then releases them all so a
             tab change paints at once.
    instant  the poster is fetched right away and decoded in the same frame it
             is first painted in, so a tab change never shows an empty screen.

  History, because it cost a day: the first version set preload="none" and
  started playback from the canplay event. With preload="none" nothing loads,
  so canplay never fires, so no clip on the site ever played. The poster also
  sat on top of the film's controls. e2e/marketing.spec.js now checks that a
  clip's currentTime actually advances.
-->
<script>
  import { onMount } from 'svelte';
  let { clip, eager = false, controls = false, active = true, warm = false, hold = false, instant = false, class: klass = '' } = $props();
  let el = $state(null);
  let video = $state(null);
  let allowed = $state(false); // motion and data are fine with this visitor
  let idle = $state(false); // the page has loaded and gone quiet
  let visible = $state(false); // the clip is on or near the screen
  let ready = $state(false); // the <video> element is in the DOM, and stays
  let playing = $state(false); // frames are on screen, the poster can go

  onMount(() => {
    const reduced = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
    const saveData = navigator.connection && navigator.connection.saveData;
    // A film only moves when the visitor presses play, so it is always allowed.
    allowed = controls || !(reduced || saveData);
    if (!allowed) return;

    // After load and at idle, so a clip never competes with the first paint.
    const whenIdle = () => {
      const go = () => (idle = true);
      if (window.requestIdleCallback) requestIdleCallback(go, { timeout: 1500 });
      else setTimeout(go, 200);
    };
    if (document.readyState === 'complete') whenIdle();
    else addEventListener('load', whenIdle, { once: true });

    const io = new IntersectionObserver(
      (entries) => {
        for (const e of entries) visible = e.isIntersecting;
      },
      { rootMargin: '200px 0px' }
    );
    io.observe(el);
    return () => io.disconnect();
  });

  // Mount once, never unmount: scrolling away and back must not fetch twice.
  $effect(() => {
    if (allowed && idle && ((visible && active) || warm)) ready = true;
  });

  // muted is set as a property as well as an attribute: browsers only allow
  // play() without a gesture on an element whose muted PROPERTY is true, and an
  // element created from script does not always pick that up from the attribute.
  $effect(() => {
    if (!video || controls) return;
    video.muted = true;
    video.defaultMuted = true;
    if (visible && active) {
      video.play().catch(() => {});
      return;
    }
    video.pause();
    if (!active) {
      playing = false;
      if (video.currentTime > 0) video.currentTime = 0;
    }
  });
</script>

<div class="av-video {klass}" bind:this={el}>
  {#if !hold && !(ready && controls)}
    <img
      src={clip.poster}
      alt={clip.alt}
      width={clip.width}
      height={clip.height}
      loading={eager || instant ? 'eager' : 'lazy'}
      fetchpriority={eager ? 'high' : undefined}
      decoding={instant ? 'sync' : 'async'}
      class:is-hidden={playing}
    />
  {/if}
  {#if ready}
    <!-- mp4 first on purpose. For flat interface footage H.264 comes out smaller
         than VP9 at the same legibility (measured: 0.40 MB against 0.60 MB for
         the hero), and every browser that plays the webm plays the mp4. The
         webm stays for the builds that ship without H.264. -->
    {#if controls}
      <video bind:this={video} controls playsinline preload="metadata" poster={clip.poster} aria-label={clip.alt}>
        <source src={clip.mp4} type="video/mp4" />
        <source src={clip.webm} type="video/webm" />
      </video>
    {:else}
      <video bind:this={video} muted playsinline loop={clip.loop} preload="auto" aria-label={clip.alt} onplaying={() => (playing = true)}>
        <source src={clip.mp4} type="video/mp4" />
        <source src={clip.webm} type="video/webm" />
      </video>
    {/if}
  {/if}
</div>

<style>
  img { transition: opacity 0.4s; z-index: 1; }
  img.is-hidden { opacity: 0; pointer-events: none; }
</style>
