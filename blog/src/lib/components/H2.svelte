<!--
  A section heading with its chevron in the rail.

  The chevron encodes POSITION, not mood: sections run through the brand
  sequence violet -> cyan -> green -> gold and then repeat, so the rail tells a
  reader how far through the piece they are at a glance. It is the mark's own
  geometry (arm 320x280, stroke 76) — the same path the background field draws
  as an SDF, so the two never disagree about what a chevron looks like.
-->
<script>
  const SEQUENCE = ['--ch-violet', '--ch-cyan', '--ch-green', '--ch-gold'];
  let { id, index = 0, children } = $props();
  const stroke = $derived(`var(${SEQUENCE[index % SEQUENCE.length]})`);
</script>

<h2 {id}>
  <svg class="chev" viewBox="0 0 396 636" fill="none" stroke-width="76" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
    <path d="M38 38L358 318L38 598" {stroke} />
  </svg>{@render children()}
</h2>
