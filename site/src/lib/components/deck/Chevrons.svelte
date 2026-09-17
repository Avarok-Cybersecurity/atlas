<script>
  // The mark, drawn from the geometry rather than shipped as an asset: stroke
  // 76, arm 320 x 280 (the rise is exactly 7/8 of the run), gap 280. Colours
  // come from the --ch-* tokens in app.css so the mark and the page cannot
  // drift apart.
  //
  // The third chevron changes colour at its elbow. That is one path with a
  // hard-stop gradient at offset 0.5 in objectBoundingBox units — the box runs
  // cap-centre to cap-centre, so 0.5 IS the elbow at any size, under any
  // transform. Two overlapping strokes would put a seam through the round join.
  let { id = 'dk-cut' } = $props();
</script>

<svg viewBox="0 0 956 636" role="img" aria-label="Atlas">
  <defs>
    <linearGradient {id} x1="0" y1="0" x2="0" y2="1">
      <stop offset="0.5" stop-color="var(--ch-green)" />
      <stop offset="0.5" stop-color="var(--ch-gold)" />
    </linearGradient>
  </defs>
  <g fill="none" stroke-width="76" stroke-linecap="round" stroke-linejoin="round">
    <path d="M38 38L358 318L38 598" stroke="var(--ch-violet)" />
    <path d="M318 38L638 318L318 598" stroke="var(--ch-cyan)" />
    <path d="M598 38L918 318L598 598" stroke="url(#{id})" />
  </g>
</svg>

<style>
  svg {
    display: block;
    width: 100%;
    height: auto;
  }
</style>
