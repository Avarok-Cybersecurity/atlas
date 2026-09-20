<!--
  A full-viewport canvas running one of the procedural loops in scenes.js.
  Opened by the recorder at /broll?scene=<name>. The logical frame is
  1280 by 720 and is scaled to cover whatever the window is, so a 1280 by 720
  recording is pixel exact and a browser window just gets a bigger picture.

  window.__sceneStart is stamped on the first frame and window.__sceneDone is
  set after one full loop plus a beat, which is what scripts/media/record.mjs
  waits for. Nothing else on the site imports this component.
-->
<script>
  import { onMount } from 'svelte';
  import { H, LOOP, TAU, W, mulberry32, scenes, seedFrom } from './scenes.js';

  let { scene = 'field' } = $props();
  let canvas = $state(null);
  const def = $derived(scenes[scene] ?? scenes.field);

  onMount(() => {
    const ctx = canvas.getContext('2d');
    const layout = def.setup(mulberry32(seedFrom(scene)));
    let raf = 0;
    let t0 = -1;
    let done = false;

    const fit = () => {
      const dpr = Math.min(2, window.devicePixelRatio || 1);
      canvas.width = Math.round(window.innerWidth * dpr);
      canvas.height = Math.round(window.innerHeight * dpr);
    };
    const frame = (now) => {
      if (t0 < 0) {
        t0 = now;
        window.__sceneStart = Date.now();
      }
      const t = (now - t0) / 1000;
      const s = Math.max(canvas.width / W, canvas.height / H);
      ctx.setTransform(s, 0, 0, s, (canvas.width - W * s) / 2, (canvas.height - H * s) / 2);
      def.draw(ctx, layout, ((t % LOOP) / LOOP) * TAU);
      if (!done && t >= LOOP + 0.4) {
        done = true;
        window.__sceneDone = true;
      }
      raf = requestAnimationFrame(frame);
    };

    fit();
    window.addEventListener('resize', fit);
    raf = requestAnimationFrame(frame);
    return () => {
      cancelAnimationFrame(raf);
      window.removeEventListener('resize', fit);
    };
  });
</script>

<canvas bind:this={canvas} class="broll" aria-label={def.alt}></canvas>

<style>
  .broll {
    position: fixed;
    inset: 0;
    width: 100vw;
    height: 100vh;
    display: block;
    background: #0f1216;
  }
</style>
