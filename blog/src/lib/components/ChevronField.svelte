<!--
  The ambient Atlas chevron field: one fullscreen triangle, one fragment
  shader. No 3D library, because there is no scene — no graph, no camera
  motion, no materials, no loaders, no picking. ~2.5 KB gzipped including
  the shader; three.js drawing the identical pixels costs ~151 KB. The
  measurement is in lib/gl/FIELD-NOTES.md.

  The canvas paints the ground colour ITSELF (the shader's last line is
  `u_ground + col * amt`), which is why the five colours are read out of the
  cascade here rather than passed as literals: if they were duplicated, a
  token change would show up as a visible seam between the canvas and the
  rest of the page, silently, with nothing failing.

  Put it once in the root layout, above the page content.
-->
<script>
  import { createChevronField } from '$lib/gl/chevron-field.js';
  import FRAG from '$lib/gl/chevron-field.glsl?raw';

  let {
    /** 0-1, multiplies the field. The default in chevron-field.js is the
     *  contrast-derived ceiling; `.contrast-check.mjs` fails if it is raised. */
    density = undefined,
    parallax = true
  } = $props();

  let canvas = $state(null);
  let ready = $state(false);

  /* Design tokens come from web-shared/atlas-tokens.css. Reading them from the
     cascade rather than hardcoding them is what keeps the canvas and the page
     the same colour. `#rgb` shorthand is expanded because the shader wants six
     digits; anything else is rejected by the runtime and the CSS field stays. */
  function tokens(el) {
    const cs = getComputedStyle(el);
    const t = (n) => {
      const v = cs.getPropertyValue(n).trim();
      return /^#[0-9a-fA-F]{3}$/.test(v) ? '#' + [...v.slice(1)].map((c) => c + c).join('') : v;
    };
    return {
      ground: t('--bg'),
      c1: t('--ch-violet'),
      c2: t('--ch-cyan'),
      c3u: t('--ch-green'),
      c3l: t('--ch-gold')
    };
  }

  /* No `browser` guard: $effect bodies are not emitted into the SSR bundle at
     all, so this cannot run during prerender. The <canvas> IS rendered
     server-side on purpose — that reserves its box and avoids a hydration pop. */
  $effect(() => {
    if (!canvas) return; // bind:this is null on the first run

    let field = null;
    let onScroll = null;
    let idle = 0;

    /* Shader compile and link is a SYNCHRONOUS main-thread stall: single-digit
       ms on a desktop GPU, 50-300ms on low-end mobile. Landing that on a user
       interaction is one catastrophic INP sample, so it happens during idle. */
    const boot = () => {
      const opts = tokens(document.documentElement);
      if (density !== undefined) opts.density = density;
      field = createChevronField(canvas, FRAG, opts);
      if (!field) return; // no WebGL2, or a token did not resolve: CSS field stays
      ready = true;
      if (onScroll) onScroll();
    };
    idle = 'requestIdleCallback' in window
      ? requestIdleCallback(boot, { timeout: 1200 })
      : setTimeout(boot, 1);

    if (parallax) {
      onScroll = () => {
        if (!field) return;
        const max = document.documentElement.scrollHeight - innerHeight;
        field.setScroll(max > 0 ? Math.min(1, Math.max(0, scrollY / max)) : 0);
      };
      addEventListener('scroll', onScroll, { passive: true });
    }

    return () => {
      // The idle callback is a suspension point: uncancelled, a fast unmount
      // leaves a renderer that nothing holds a reference to.
      if ('cancelIdleCallback' in window) cancelIdleCallback(idle); else clearTimeout(idle);
      if (onScroll) removeEventListener('scroll', onScroll);
      if (!field) { ready = false; return; }
      /* Non-negotiable on a client-routed site. Without destroy() the rAF loop
         keeps drawing to a detached canvas after every navigation, and the
         leaked GL contexts walk into the browser's 16-context cap — at which
         point an unrelated canvas elsewhere on the site goes black. */
      field.destroy();
      ready = false;
    };
  });
</script>

<!-- The CSS dot field is the floor: it renders for everyone, including no-JS
     and no-WebGL, and fades out once the canvas takes over. -->
<div class="cf-dots" class:cf-muted={ready} aria-hidden="true"></div>
<canvas bind:this={canvas} class="cf" class:cf-on={ready} aria-hidden="true"></canvas>

<style>
  .cf,
  .cf-dots {
    position: fixed;
    inset: 0;
    z-index: 0;
    pointer-events: none; /* never swallow a click or break text selection */
  }
  .cf {
    width: 100%;
    height: 100%;
    display: block;
    opacity: 0;
    transition: opacity 600ms cubic-bezier(0.4, 0, 0.2, 1);
  }
  .cf-on { opacity: 1; }
  .cf-dots {
    background-image: radial-gradient(color-mix(in srgb, var(--t2) 9%, transparent) 1px, transparent 0);
    background-size: 8px 8px;
    -webkit-mask-image: radial-gradient(120% 70% at 50% 0, #000 0%, transparent 62%);
    mask-image: radial-gradient(120% 70% at 50% 0, #000 0%, transparent 62%);
    transition: opacity 600ms cubic-bezier(0.4, 0, 0.2, 1);
  }
  .cf-muted { opacity: 0; }
  @media (prefers-reduced-motion: reduce) {
    .cf, .cf-dots { transition: none; }
  }
</style>
