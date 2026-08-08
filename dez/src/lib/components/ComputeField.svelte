<!-- SPDX-License-Identifier: AGPL-3.0-only -->
<!--
  Decorative backdrop for the hero. CSS only — no canvas, no JS, no timers, and
  nothing that can block the main thread. Two soft accent fields drift behind a
  masked grid. With prefers-reduced-motion the drift stops and the composition
  is rendered statically, which is the intended still frame rather than a
  degraded one.
-->
<div class="field" aria-hidden="true">
  <div class="grid"></div>
  <div class="glow glow-a"></div>
  <div class="glow glow-b"></div>
  <div class="scan"></div>
</div>

<style>
  .field {
    position: absolute;
    inset: 0;
    overflow: hidden;
    pointer-events: none;
    /* Fade the whole composition out towards the section edges. */
    mask-image: radial-gradient(115% 90% at 50% 5%, #000 30%, transparent 78%);
  }

  .grid {
    position: absolute;
    inset: -2px;
    background-image:
      linear-gradient(to right, var(--border) 1px, transparent 1px),
      linear-gradient(to bottom, var(--border) 1px, transparent 1px);
    background-size: 56px 56px;
    opacity: 0.55;
  }

  .glow {
    position: absolute;
    width: min(46rem, 120vw);
    aspect-ratio: 1;
    border-radius: 50%;
    filter: blur(60px);
    opacity: 0.5;
    will-change: transform;
  }

  .glow-a {
    top: -22rem;
    left: -10rem;
    background: radial-gradient(circle, var(--accent-wash) 0%, transparent 62%);
    animation: drift-a 26s var(--ease) infinite alternate;
  }

  .glow-b {
    top: -16rem;
    right: -14rem;
    background: radial-gradient(
      circle,
      color-mix(in srgb, var(--accent) 9%, transparent) 0%,
      transparent 60%
    );
    animation: drift-b 34s var(--ease) infinite alternate;
  }

  /* A single accent hairline sweeping the top edge — the one moving accent. */
  .scan {
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    height: 1px;
    background: linear-gradient(
      90deg,
      transparent 0%,
      var(--accent) 45%,
      var(--accent) 55%,
      transparent 100%
    );
    opacity: 0.65;
    transform: translateX(-100%);
    animation: sweep 11s linear infinite;
  }

  @keyframes drift-a {
    from {
      transform: translate3d(0, 0, 0) scale(1);
    }
    to {
      transform: translate3d(6rem, 3rem, 0) scale(1.12);
    }
  }

  @keyframes drift-b {
    from {
      transform: translate3d(0, 0, 0) scale(1.08);
    }
    to {
      transform: translate3d(-5rem, 4rem, 0) scale(1);
    }
  }

  @keyframes sweep {
    from {
      transform: translateX(-100%);
    }
    to {
      transform: translateX(100%);
    }
  }

  @media (prefers-reduced-motion: reduce) {
    .glow-a,
    .glow-b {
      animation: none;
    }

    /* Nothing sweeps; leave a static accent rule instead of a frozen ghost. */
    .scan {
      animation: none;
      transform: none;
      opacity: 0.4;
    }
  }
</style>
