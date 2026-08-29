<script>
  // The concurrency ladder: aggregate tok/s vs concurrency C, one polyline
  // per run, newest run full-strength and older runs faded so the curve's
  // drift stays visible without overloading the panel. X is log2-spaced —
  // the sweep's rungs are 1,2,4,8,16,32 and linear spacing would crush the
  // low-C half where the single-stream story lives. Same hand-rolled SVG
  // dialect and classes as GateChart.svelte — no chart library.
  import { colorFor, fmtDate, ladderPoints } from '$lib/gates.js';
  import { dashFor, isLatestOfVariant, variantLabel } from '$lib/gate-variants.js';

  let { records, panel, onselect } = $props();

  const W = 720, H = 232, PL = 56, PR = 16, PT = 14, PB = 30;

  // Only runs that actually published ladder keys chart; the panel spec is
  // only emitted when at least one exists, but a mixed history (older runs
  // from before the metrics map) must not blank the chart.
  // `latest` is per VARIANT, not the last element. With two gate ids on one
  // axis the newest DFlash2 run is usually not the newest run overall, so a
  // single global latest would leave a whole variant permanently faded and
  // reading as stale. Dash carries the variant; colour still follows the
  // model, because both variants serve the same checkpoint.
  const runs = $derived(
    records
      .map((rec) => ({
        rec,
        pts: ladderPoints(rec),
        dash: dashFor(rec.benchmark_id),
        variant: variantLabel(rec.benchmark_id),
        latest: isLatestOfVariant(rec, records)
      }))
      .filter((r) => r.pts.length > 0)
  );
  // One legend entry per variant actually drawn, so a reader can tell which
  // line is which without clicking a point.
  const legend = $derived(
    [...new Map(runs.filter((r) => r.variant).map((r) => [r.variant, r])).values()]
  );

  const ext = $derived.by(() => {
    const cs = runs.flatMap((r) => r.pts.map((p) => p.c));
    const vs = runs.flatMap((r) => r.pts.map((p) => p.v));
    const [c0, c1] = [Math.min(...cs), Math.max(...cs)];
    let [v0, v1] = [Math.min(...vs), Math.max(...vs)];
    const pad = (v1 - v0 || Math.abs(v1) || 1) * 0.12;
    return { l0: Math.log2(c0), l1: Math.log2(c1), v0: v0 - pad, v1: v1 + pad };
  });

  const x = (c) =>
    PL + (ext.l1 === ext.l0 ? 0.5 : (Math.log2(c) - ext.l0) / (ext.l1 - ext.l0)) * (W - PL - PR);
  const y = (v) => PT + (1 - (v - ext.v0) / (ext.v1 - ext.v0 || 1)) * (H - PT - PB);
  const path = (pts) => pts.map((p, i) => `${i ? 'L' : 'M'}${x(p.c).toFixed(1)} ${y(p.v).toFixed(1)}`).join(' ');

  const fmtV = (v) => (Math.abs(v) >= 1000 ? Math.round(v).toLocaleString('en-US') : +v.toFixed(1));
  const cTicks = $derived([...new Set(runs.flatMap((r) => r.pts.map((p) => p.c)))].sort((a, b) => a - b));
  const yTicks = $derived([ext.v0, (ext.v0 + ext.v1) / 2, ext.v1]);
</script>

<figure class="gate-panel">
  <figcaption class="gate-panel-head">
    <span class="gate-panel-title">{panel.title}</span>
    <span class="gate-panel-unit">{panel.unit}</span>
    {#if legend.length > 0}
      <span class="gate-legend">
        {#each legend as l}
          <span class="gate-legend-item">
            <svg class="gate-legend-line" viewBox="0 0 18 8" aria-hidden="true">
              <line x1="1" y1="4" x2="17" y2="4" stroke={colorFor(l.rec.target_model)}
                stroke-width="2" stroke-dasharray={l.dash} stroke-linecap="round" />
            </svg>
            {l.variant}
          </span>
        {/each}
      </span>
    {:else if runs.length > 1}
      <span class="gate-legend">
        <span class="gate-legend-item">latest run solid · older runs faded</span>
      </span>
    {/if}
  </figcaption>

  <svg viewBox="0 0 {W} {H}" role="img" aria-label="{panel.title} for the latest gate runs">
    {#each yTicks as t}
      <line class="gc-grid" x1={PL} y1={y(t)} x2={W - PR} y2={y(t)} />
      <text class="gc-axis" x={PL - 8} y={y(t) + 3.5} text-anchor="end">{fmtV(t)}</text>
    {/each}
    {#each cTicks as c}
      <text class="gc-axis" x={x(c)} y={H - 8} text-anchor="middle">C={c}</text>
    {/each}

    {#each runs as r}
      {@const latest = r.latest}
      {@const col = colorFor(r.rec.target_model)}
      <g opacity={latest ? 1 : 0.28}>
        <path d={path(r.pts)} fill="none" stroke={col} stroke-width={latest ? 2 : 1.5}
          stroke-dasharray={r.dash} stroke-linejoin="round" stroke-linecap="round" />
        {#each r.pts as p}
          <g
            class="gc-pt"
            role="button"
            tabindex="0"
            aria-label="C={p.c}: {fmtV(p.v)} tok/s{r.variant ? ', ' + r.variant : ''} on {fmtDate(r.rec.recorded_at)}, {r.rec.verdict} — details"
            onclick={() => onselect(r.rec)}
            onkeydown={(e) => (e.key === 'Enter' || e.key === ' ') && (e.preventDefault(), onselect(r.rec))}
          >
            <title>C={p.c} · {fmtV(p.v)} tok/s{r.variant ? ' · ' + r.variant : ''} · {fmtDate(r.rec.recorded_at)} · {r.rec.verdict} · click for record</title>
            <circle class="gc-hit" cx={x(p.c)} cy={y(p.v)} r="11" />
            {#if r.rec.verdict === 'PASS'}
              <circle class="gc-mark" cx={x(p.c)} cy={y(p.v)} r="3.5" fill={col} />
            {:else}
              <circle class="gc-mark gc-fail" cx={x(p.c)} cy={y(p.v)} r="4.5" fill="var(--card)" stroke={col} stroke-width="2" />
            {/if}
          </g>
        {/each}
        {#if latest && r.pts.length}
          {@const lp = r.pts[r.pts.length - 1]}
          <text class="gc-val" x={Math.min(x(lp.c), W - PR - 4)} y={Math.max(y(lp.v) - 9, 13)} text-anchor="end" fill={col}>
            {fmtV(lp.v)}
          </text>
        {/if}
      </g>
    {/each}
  </svg>
</figure>
