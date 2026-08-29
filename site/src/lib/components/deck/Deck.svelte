<script>
  // The deck engine. Slides register themselves through context in DOM order,
  // so an act can be split across files without a slide manifest to keep in
  // sync — the composition IS the manifest.
  //
  // No presentation library. The three things one would have bought (a scaled
  // canvas, slide-to-slide motion, syntax highlighting) are, in 2026, a
  // container query, a CSS transition, and a <pre>. What a library would add
  // instead is a second design system to override and a router conflict:
  // reveal.js writes the URL with a bare history.replaceState, which nulls the
  // history metadata SvelteKit stores there and breaks Back. Deep-linking here
  // goes through $app/navigation, which does not.
  import { setContext } from 'svelte';
  import { browser } from '$app/environment';
  import { replaceState } from '$app/navigation';
  import Chevrons from './Chevrons.svelte';

  let { title = 'Verification steps', stamp = '', children } = $props();

  let index = $state(0);
  let step = $state(0);
  let stepsOnSlide = $state(0);
  let total = $state(0);
  let acts = $state([]);

  // Registration happens during child init, which Svelte runs in DOM order.
  let seq = 0;
  setContext('deck', {
    register(act) {
      const n = seq++;
      total = seq;
      acts[n] = act;
      return n;
    },
    // Read by the active slide so it can publish its fragment count upward.
    current: () => index,
    step: () => step,
    setSteps(n) {
      stepsOnSlide = n;
    }
  });

  const act = $derived(acts[index] ?? 'violet');

  // The hash, not a query param: this route is prerendered, and touching
  // url.searchParams during prerender is a build error. The hash never reaches
  // the prerenderer at all.
  $effect(() => {
    if (!browser) return;
    const n = Number(location.hash.slice(1));
    if (Number.isInteger(n) && n >= 1 && n <= total) index = n - 1;
  });

  function go(n) {
    const next = Math.max(0, Math.min(total - 1, n));
    if (next === index) return;
    index = next;
    step = 0;
    replaceState(`#${next + 1}`, {});
  }

  function advance() {
    if (step < stepsOnSlide) step += 1;
    else go(index + 1);
  }

  function retreat() {
    if (step > 0) step -= 1;
    else go(index - 1);
  }

  function onkeydown(e) {
    if (e.metaKey || e.ctrlKey || e.altKey) return;
    const tag = document.activeElement?.tagName;
    if (tag === 'INPUT' || tag === 'TEXTAREA') return;
    switch (e.key) {
      case 'ArrowRight':
      case 'ArrowDown':
      case 'PageDown':
      case ' ':
        advance();
        break;
      case 'ArrowLeft':
      case 'ArrowUp':
      case 'PageUp':
        retreat();
        break;
      case 'Home':
        go(0);
        break;
      case 'End':
        go(total - 1);
        break;
      case 'f':
        document.documentElement.requestFullscreen?.();
        break;
      case 'Escape':
        document.exitFullscreen?.();
        break;
      default:
        return;
    }
    e.preventDefault();
  }
</script>

<svelte:window {onkeydown} />
<svelte:head><title>{title} · Atlas</title></svelte:head>

<div class="dk" style="--sx: var(--ch-{act})">
  <p class="dk-live" aria-live="polite">Slide {index + 1} of {total}</p>

  <div class="dk-stage" style="--step: {step}">
    <div class="dk-track" style="--i: {index}">
      {@render children()}
    </div>
  </div>

  <button
    type="button"
    class="dk-edge dk-edge-prev"
    onclick={retreat}
    disabled={index === 0 && step === 0}
    aria-label="Previous slide"
  >
    <svg viewBox="0 0 396 636" aria-hidden="true">
      <path d="M358 38L38 318L358 598" />
    </svg>
  </button>
  <button
    type="button"
    class="dk-edge dk-edge-next"
    onclick={advance}
    disabled={index === total - 1 && step === stepsOnSlide}
    aria-label="Next slide"
  >
    <svg viewBox="0 0 396 636" aria-hidden="true">
      <path d="M38 38L358 318L38 598" />
    </svg>
  </button>

  <!-- One segment per slide, coloured by act: the reader can see the three
       movements of the deck and where they are inside the current one. -->
  <div class="dk-rail" aria-hidden="true">
    {#each acts as a, n}
      <i class="dk-seg" class:on={n <= index} style="--c: var(--ch-{a ?? 'violet'})"></i>
    {/each}
  </div>

  <div class="dk-chrome">
    <a class="dk-mark" href="/" aria-label="Atlas home"><Chevrons /></a>
    <span class="dk-stamp mono">{stamp}</span>
    <span class="dk-count mono">{String(index + 1).padStart(2, '0')} / {String(total).padStart(2, '0')}</span>
  </div>
</div>

<style>
  /* The deck fills the window rather than letterboxing a fixed 16:9 canvas
     inside it: a black margin on a projector or an ultrawide monitor reads as
     a broken page, not as a design. The container is here, on the element
     that IS the viewport, because container-query units on the container
     element itself resolve against its ancestor, not itself. */
  .dk {
    position: fixed;
    inset: 0;
    container-type: size;
    container-name: stage;
    background: var(--bg);
    color: var(--t1);
    overflow: hidden;
  }

  /* A wash of the current act's colour, low enough to read as depth rather
     than decoration. It moves through violet, cyan, green and gold as the deck
     changes act, which is the only ambient cue that a movement has ended. */
  .dk::before {
    content: '';
    position: absolute;
    inset: 0;
    pointer-events: none;
    background:
      radial-gradient(120% 80% at 8% -10%, color-mix(in oklab, var(--sx) 13%, transparent), transparent 62%),
      radial-gradient(90% 70% at 100% 108%, color-mix(in oklab, var(--sx) 8%, transparent), transparent 60%);
    transition: background 620ms ease;
  }

  /* One scale unit for the whole deck, taken from whichever viewport dimension
     binds. At 16:9 the two arms are equal and --u is 1% of the width; on a
     shorter or wider window the height arm wins and everything shrinks
     together, so a slide never grows out of the frame. Type stays real text —
     no transform — so it selects, prints and reads by a screen reader. */
  .dk-stage {
    position: absolute;
    inset: 0;
    --u: min(1cqw, 1.778cqh);
    font-size: calc(1.25 * var(--u));
    overflow: hidden;
  }

  /* Slides sit side by side on one track and the track slides. A crossfade
     reads as a dissolve between unrelated pictures; a horizontal move reads as
     turning a page, which is what an argument in sequence actually is. */
  .dk-track {
    display: flex;
    height: 100%;
    transform: translate3d(calc(var(--i) * -100%), 0, 0);
    transition: transform 620ms cubic-bezier(0.66, 0, 0.24, 1);
    will-change: transform;
  }

  /* Edge navigation, drawn as one chevron of the mark. Quiet until the pointer
     is near, because a control that shouts on every slide becomes furniture. */
  .dk-edge {
    position: absolute;
    top: 50%;
    translate: 0 -50%;
    width: 3.2rem;
    height: 5.5rem;
    display: grid;
    place-items: center;
    border: 0;
    background: none;
    cursor: pointer;
    opacity: 0.24;
    transition: opacity 200ms ease, translate 200ms ease;
  }
  .dk-edge svg {
    width: 0.85rem;
    height: auto;
    fill: none;
    stroke: var(--t1);
    stroke-width: 76;
    stroke-linecap: round;
    stroke-linejoin: round;
  }
  .dk-edge-prev {
    left: 0.4rem;
  }
  .dk-edge-next {
    right: 0.4rem;
  }
  .dk-edge:hover:not(:disabled) {
    opacity: 1;
  }
  .dk-edge-prev:hover:not(:disabled) {
    translate: -0.25rem -50%;
  }
  .dk-edge-next:hover:not(:disabled) {
    translate: 0.25rem -50%;
  }
  .dk-edge:hover:not(:disabled) svg {
    stroke: var(--sx);
  }
  .dk-edge:disabled {
    opacity: 0.06;
    cursor: default;
  }
  .dk-edge:focus-visible {
    opacity: 1;
    outline: 2px solid var(--sx);
    outline-offset: -6px;
    border-radius: 8px;
  }

  .dk-rail {
    position: absolute;
    left: 0;
    right: 0;
    bottom: 0;
    height: 3px;
    display: flex;
    gap: 2px;
    background: transparent;
  }
  .dk-seg {
    flex: 1;
    background: var(--border);
    transition: background 320ms ease, opacity 320ms ease;
    opacity: 0.55;
  }
  .dk-seg.on {
    background: var(--c);
    opacity: 1;
  }

  .dk-chrome {
    position: absolute;
    left: 1.6rem;
    right: 1.6rem;
    bottom: 1rem;
    display: flex;
    align-items: center;
    gap: 1rem;
    font-size: 0.72rem;
    color: var(--t3);
  }
  .dk-mark {
    display: block;
    width: 34px;
    opacity: 0.9;
  }
  .dk-stamp {
    flex: 1;
    letter-spacing: 0.02em;
  }
  .dk-count {
    letter-spacing: 0.06em;
    color: var(--t2);
  }

  .dk-live {
    position: absolute;
    width: 1px;
    height: 1px;
    overflow: hidden;
    clip-path: inset(50%);
    white-space: nowrap;
  }

  /* Print is the PDF export. Every slide is already in the DOM — that is the
     whole reason slides are hidden with visibility rather than {#if} — so this
     is a stylesheet, not a feature. */
  @media print {
    @page {
      size: 1600px 900px landscape;
      margin: 0;
    }
    .dk {
      position: static;
      container-type: normal;
      display: block;
      overflow: visible;
    }
    .dk-stage {
      position: static;
      overflow: visible;
      --u: 16px;
      font-size: 21px;
    }
    .dk-track {
      display: block;
      transform: none;
    }
    .dk::before,
    .dk-rail,
    .dk-edge,
    .dk-chrome {
      display: none;
    }
  }

  /* Slide.svelte in this same directory already honours this; the deck that
     MOVES the slides did not. `.dk-track` carries a 620ms full-viewport
     translation — the exact motion the preference exists for — and `.dk-edge`
     slides its arrows in.

     Only motion is dropped. The background and opacity fades stay: they cause
     none of what the setting is about, and removing them would make the deck
     snap rather than settle. */
  @media (prefers-reduced-motion: reduce) {
    .dk-track {
      transition: none;
    }
    .dk-edge {
      transition: opacity 200ms ease;
      translate: none;
    }
  }
</style>
