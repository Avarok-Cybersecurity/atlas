<script>
  // One slide. It takes its index from the deck at init, which means slides can
  // be written in whatever file the act lives in and reordered by moving the
  // markup — there is no numbering to maintain.
  //
  // `steps` is the number of fragments this slide reveals. The active slide
  // reports it to the deck so the arrow keys walk fragments before moving on.
  import { getContext } from 'svelte';

  let { act = 'violet', eyebrow = '', title = '', lede = '', steps = 0, wide = false, children } =
    $props();

  const deck = getContext('deck');
  const n = deck.register(act);

  const active = $derived(deck.current() === n);

  $effect(() => {
    if (active) deck.setSteps(steps);
  });
</script>

<section
  class="sl"
  class:sl-active={active}
  class:sl-wide={wide}
  inert={!active}
  aria-roledescription="slide"
  aria-label={title || `Slide ${n + 1}`}
  style="--sx: var(--ch-{act})"
>
  <header class="sl-head">
    {#if eyebrow}<p class="sl-eyebrow mono">{eyebrow}</p>{/if}
    {#if title}<h2 class="sl-title">{title}</h2>{/if}
    {#if lede}<p class="sl-lede">{lede}</p>{/if}
  </header>

  <div class="sl-body">
    {@render children?.()}
  </div>
</section>

<style>
  /* One panel on the deck's track. Off-screen slides are dimmed and eased back
     a hair, so the one in view is the one with weight — the difference is only
     visible for the ~600 ms the track is moving, which is the point. */
  .sl {
    flex: 0 0 100%;
    min-width: 0;
    height: 100%;
    position: relative;
    display: grid;
    grid-template-rows: auto 1fr;
    gap: 1.3em;
    padding: calc(3.6 * var(--u)) calc(5 * var(--u)) calc(5.4 * var(--u));
    align-content: start;
    opacity: 0.28;
    scale: 0.985;
    transition: opacity 480ms ease, scale 620ms cubic-bezier(0.66, 0, 0.24, 1);
  }
  .sl-active {
    opacity: 1;
    scale: 1;
  }
  .sl-wide {
    padding-inline: calc(3.6 * var(--u));
  }

  .sl-head {
    display: grid;
    gap: 0.5em;
    max-width: 46ch;
  }
  .sl-wide .sl-head {
    max-width: none;
  }
  .sl-eyebrow {
    font-size: 0.78em;
    font-weight: 600;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: var(--sx);
  }
  .sl-eyebrow::before {
    content: '';
    display: inline-block;
    width: 1.5em;
    height: 1px;
    background: var(--sx);
    vertical-align: 0.35em;
    margin-right: 0.7em;
  }
  .sl-title {
    font-size: 2.2em;
    font-weight: 800;
    letter-spacing: -0.03em;
    line-height: 1.08;
    text-wrap: balance;
  }
  .sl-lede {
    font-size: 1.05em;
    color: var(--t2);
    line-height: 1.6;
    max-width: 62ch;
  }

  .sl-body {
    min-height: 0;
    font-size: 0.98em;
  }

  /* Fragments. A child opts in with class="at" and style="--n: 1"; the deck
     sets --step on the stage. Numeric attr() would be tidier and is Chromium
     only, so this stays a custom property. */
  .sl :global(.at) {
    opacity: 0;
    transform: translateY(0.35em);
    transition: opacity 220ms ease, transform 220ms ease;
  }
  .sl-active :global(.at) {
    opacity: clamp(0, calc(var(--step, 0) - var(--n) + 1), 1);
    transform: translateY(calc((1 - clamp(0, calc(var(--step, 0) - var(--n) + 1), 1)) * 0.35em));
  }

  @media (prefers-reduced-motion: reduce) {
    .sl,
    .sl :global(.at) {
      transition: none;
    }
    .sl {
      scale: 1;
    }
  }

  @media print {
    .sl {
      display: grid;
      width: 1600px;
      height: 900px;
      opacity: 1 !important;
      scale: 1 !important;
      break-after: page;
      border-bottom: 1px solid var(--border);
    }
    .sl :global(.at) {
      opacity: 1 !important;
      transform: none !important;
    }
  }
</style>
