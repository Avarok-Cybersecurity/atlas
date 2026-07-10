<script>
  // Self-hosted star-history chart. No external star-history.com embed (privacy +
  // CSP + perf). Data regenerated from the GitHub API every deploy by gen-stars.mjs.
  import stars from '$lib/stars.generated.json';

  const W = 640, H = 264, PL = 46, PR = 18, PT = 18, PB = 34;
  const pts = stars.history ?? [];
  const MON = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'];

  const maxY = Math.max(100, Math.ceil((stars.count || 1) / 100) * 100);
  const xFor = (i) => PL + (pts.length <= 1 ? 0 : (i / (pts.length - 1)) * (W - PL - PR));
  const yFor = (v) => PT + (1 - v / maxY) * (H - PT - PB);

  const xy = pts.map((p, i) => ({ x: xFor(i), y: yFor(p.stars), ...p }));
  const line = xy.map((p, i) => `${i ? 'L' : 'M'}${p.x.toFixed(1)} ${p.y.toFixed(1)}`).join(' ');
  const area = xy.length ? `${line} L${xy[xy.length-1].x.toFixed(1)} ${H-PB} L${xy[0].x.toFixed(1)} ${H-PB} Z` : '';
  const last = xy[xy.length - 1];

  const yTicks = [0, maxY / 2, maxY];
  const label = (d) => { const m = +String(d).slice(5, 7) - 1; return MON[m] ?? d; };
  const xTicks = xy.length
    ? [xy[0], xy[Math.floor(xy.length / 2)], xy[xy.length - 1]].map((p) => ({ x: p.x, t: label(p.date) }))
    : [];
</script>

<div class="star-chart">
  <svg viewBox="0 0 {W} {H}" role="img" aria-label="GitHub star history reaching {stars.count} stars">
    <defs>
      <linearGradient id="starfill" x1="0" y1="0" x2="0" y2="1">
        <stop offset="0%" stop-color="var(--accent)" stop-opacity="0.20" />
        <stop offset="100%" stop-color="var(--accent)" stop-opacity="0.01" />
      </linearGradient>
    </defs>

    {#each yTicks as t}
      <line class="grid" x1={PL} y1={yFor(t)} x2={W - PR} y2={yFor(t)} />
      <text class="axis-y" x={PL - 8} y={yFor(t) + 3.5} text-anchor="end">{t}</text>
    {/each}

    {#if area}<path d={area} fill="url(#starfill)" />{/if}
    {#if line}<path d={line} fill="none" stroke="var(--accent)" stroke-width="2.5" stroke-linejoin="round" stroke-linecap="round" />{/if}

    {#if last}
      <circle cx={last.x} cy={last.y} r="4.5" fill="var(--accent)" />
      <text class="axis-val" x={last.x - 6} y={last.y - 10} text-anchor="end">{stars.count} ★</text>
    {/if}

    {#each xTicks as t}
      <text class="axis-x" x={t.x} y={H - 10} text-anchor="middle">{t.t}</text>
    {/each}
  </svg>
</div>
