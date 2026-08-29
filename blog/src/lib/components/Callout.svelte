<!-- An aside that must not be skimmed past. `tone` picks the chevron colour
     that already means this: gold = community/notice, green = verified,
     cyan = technical, violet = engine. -->
<script>
  const TONES = {
    notice: 'var(--ch-gold)',
    verified: 'var(--ch-green)',
    technical: 'var(--ch-cyan)',
    engine: 'var(--ch-violet)'
  };
  let { label, tone = 'notice', children } = $props();
  const color = $derived(TONES[tone] ?? TONES.notice);
</script>

<aside class="callout" style="--co: {color}">
  <svg class="ci" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" aria-hidden="true">
    <path d="M12 9v4" /><path d="M12 17h.01" /><circle cx="12" cy="12" r="9" />
  </svg>
  <p><span class="ct">{label}</span>{@render children()}</p>
</aside>
