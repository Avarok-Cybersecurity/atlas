<script>
  // One metric panel of the benchmark dashboard: a time-scaled line per metric,
  // reference lines for budgets/floors read from the records, and a click on
  // any point raising the record for the metadata card. Same hand-rolled SVG
  // dialect as StarChart.svelte — no chart library.
  import { colorFor, fmtDate } from '$lib/gates.js';
  import { trendEdges } from '$lib/gate-lineage.js';

  let { records, panel, onselect } = $props();

  const W = 720, H = 232, PL = 56, PR = 16, PT = 14, PB = 30;

  const series = $derived(
    panel.metrics
      .map((m) => {
        const pts = records
          // `m.variant` narrows a series to one gate id, so two variants of
          // one instrument never share a polyline (see gate-variants.js).
          // Filtering BEFORE trendEdges also keeps lineage honest: an edge
          // between two different gates' records is not a trend.
          .filter((r) => !m.variant || r.benchmark_id === m.variant)
          .filter((r) => Number.isFinite(r.metrics?.[m.key]))
          .map((r) => ({ t: r.recorded_at, v: r.metrics[m.key], rec: r }));
        return { ...m, pts, edges: trendEdges(pts) };
      })
      .filter((s) => s.pts.length > 0)
  );

  const refLines = $derived([...(panel.caps ?? []), ...(panel.floors ?? [])]);

  const ext = $derived.by(() => {
    const ts = series.flatMap((s) => s.pts.map((p) => p.t));
    const vs = [...series.flatMap((s) => s.pts.map((p) => p.v)), ...refLines.map((l) => l.value)];
    const [t0, t1] = [Math.min(...ts), Math.max(...ts)];
    let [v0, v1] = panel.domain ?? [Math.min(...vs), Math.max(...vs)];
    if (!panel.domain) {
      const pad = (v1 - v0 || Math.abs(v1) || 1) * 0.12;
      v0 -= pad; v1 += pad;
    }
    return { t0, t1, v0, v1 };
  });

  const x = (t) => PL + (ext.t1 === ext.t0 ? 0.5 : (t - ext.t0) / (ext.t1 - ext.t0)) * (W - PL - PR);
  const y = (v) => PT + (1 - (v - ext.v0) / (ext.v1 - ext.v0 || 1)) * (H - PT - PB);
  const path = (pts) => pts.map((p, i) => `${i ? 'L' : 'M'}${x(p.t).toFixed(1)} ${y(p.v).toFixed(1)}`).join(' ');

  const fmtV = (v) => (Math.abs(v) >= 1000 ? Math.round(v).toLocaleString('en-US') : +v.toFixed(2));
  // Axis ticks read as landmarks, not data — round them (fine values live on
  // the points, their tooltips and the metadata card).
  const fmtTick = (v) => {
    const range = ext.v1 - ext.v0;
    if (range >= 50) return Math.round(v).toLocaleString('en-US');
    return +v.toFixed(range >= 5 ? 0 : 1);
  };
  const yTicks = $derived([ext.v0, (ext.v0 + ext.v1) / 2, ext.v1]);
  const xTicks = $derived.by(() => {
    const ts = [...new Set(series.flatMap((s) => s.pts.map((p) => p.t)))].sort((a, b) => a - b);
    const picked = ts.length <= 2 ? ts : [ts[0], ts[Math.floor(ts.length / 2)], ts[ts.length - 1]];
    return picked.map((t, i) => ({
      t,
      anchor: i === 0 ? 'start' : i === picked.length - 1 ? 'end' : 'middle',
      ax: i === 0 ? PL : i === picked.length - 1 ? W - PR : x(t)
    }));
  });
  const hasFail = $derived(series.some((s) => s.pts.some((p) => p.rec.verdict !== 'PASS')));
</script>

<figure class="gate-panel">
  <figcaption class="gate-panel-head">
    <span class="gate-panel-title">{panel.title}</span>
    <span class="gate-panel-unit">{panel.unit}</span>
    {#if series.length > 1 || hasFail}
      <span class="gate-legend">
        {#each series.length > 1 ? series : [] as s}
          <span class="gate-legend-item">
            <svg width="18" height="8" aria-hidden="true"
              ><line x1="1" y1="4" x2="17" y2="4" stroke={colorFor(s.pts[0].rec.target_model)} stroke-width="2"
                stroke-dasharray={s.dashed ? '4 3' : 'none'} /></svg>
            {s.label}
          </span>
        {/each}
        {#if hasFail}
          <span class="gate-legend-item">
            <svg width="12" height="10" aria-hidden="true"
              ><circle cx="6" cy="5" r="3.5" fill="none" stroke="currentColor" stroke-width="1.8" /></svg>
            FAIL
          </span>
        {/if}
      </span>
    {/if}
  </figcaption>

  <svg viewBox="0 0 {W} {H}" role="img" aria-label="{panel.title} across gate runs">
    {#each yTicks as t}
      <line class="gc-grid" x1={PL} y1={y(t)} x2={W - PR} y2={y(t)} />
      <text class="gc-axis" x={PL - 8} y={y(t) + 3.5} text-anchor="end">{fmtTick(t)}</text>
    {/each}
    {#each xTicks as t}
      <text class="gc-axis" x={t.ax} y={H - 8} text-anchor={t.anchor}>{fmtDate(t.t)}</text>
    {/each}

    {#each refLines as l}
      <line class="gc-ref" x1={PL} y1={y(l.value)} x2={W - PR} y2={y(l.value)} />
      <text class="gc-ref-label" x={W - PR} y={y(l.value) - 5} text-anchor="end">{l.label}</text>
    {/each}

    {#each series as s}
      {@const c = colorFor(s.pts[0].rec.target_model)}
      {#each s.edges as edge}
        <path d={path(edge)} fill="none" stroke={c} stroke-width="2"
          stroke-dasharray={s.dashed ? '5 4' : 'none'} stroke-linejoin="round" stroke-linecap="round" />
      {/each}
      {#each s.pts as p}
        <g
          class="gc-pt"
          role="button"
          tabindex="0"
          aria-label="{s.label} {fmtV(p.v)} on {fmtDate(p.t)}, {p.rec.verdict}, {p.rec.target_model} — details"
          onclick={() => onselect(p.rec)}
          onkeydown={(e) => (e.key === 'Enter' || e.key === ' ') && (e.preventDefault(), onselect(p.rec))}
        >
          <title>{s.label} {fmtV(p.v)}{panel.unit === 'ms' || panel.unit === 's' ? ` ${panel.unit}` : ''} · {fmtDate(p.t)} · {p.rec.verdict} · click for record</title>
          <circle class="gc-hit" cx={x(p.t)} cy={y(p.v)} r="11" />
          {#if p.rec.verdict === 'PASS'}
            <circle class="gc-mark" cx={x(p.t)} cy={y(p.v)} r="3.5" fill={c} />
          {:else}
            <circle class="gc-mark gc-fail" cx={x(p.t)} cy={y(p.v)} r="4.5" fill="var(--card)" stroke={c} stroke-width="2" />
          {/if}
        </g>
      {/each}
      {#if s.pts.length}
        {@const lp = s.pts[s.pts.length - 1]}
        <text class="gc-val" x={Math.min(x(lp.t), W - PR - 4)} y={Math.max(y(lp.v) - 9, 13)} text-anchor="end" fill={c}>
          {fmtV(lp.v)}
        </text>
      {/if}
    {/each}
  </svg>
</figure>
