<script>
  // The concurrency ladder: aggregate tok/s vs concurrency C. X is log2-spaced —
  // the sweep's rungs are 1,2,4,8,16,32,64,128 and linear spacing would crush
  // the low-C half where the single-stream story lives. Same hand-rolled SVG
  // dialect and classes as GateChart.svelte — no chart library.
  //
  // Run history is a REGION, not forty polylines. Only the newest run per
  // variant and the one before it are drawn as lines; everything older becomes
  // the corridor they sit in (gate-band.js). Dash carries the variant; colour
  // still follows the model, because both variants serve the same checkpoint.
  import { colorFor, fmtDate, ladderPoints } from '$lib/gates.js';
  import { dashFor, variantLabel } from '$lib/gate-variants.js';
  import { bandPath, historyBand, splitHistory } from '$lib/gate-band.js';
  import { dodgeLabels } from '$lib/gate-domain.js';

  let { records, panel, onselect } = $props();

  const W = 720, H = 232, PL = 56, PR = 16, PT = 14, PB = 30;
  const LABEL_H = 13;

  const runs = $derived(
    records
      .map((rec) => ({
        rec,
        pts: ladderPoints(rec),
        dash: dashFor(rec.benchmark_id),
        variant: variantLabel(rec.benchmark_id)
      }))
      .filter((r) => r.pts.length > 0)
  );

  // One lane per (benchmark id, model): a lane is the thing that has a history,
  // and mixing two variants' runs into one corridor would claim a spread that
  // no single configuration ever showed.
  const lanes = $derived.by(() => {
    const by = new Map();
    for (const r of runs) {
      const k = `${r.rec.benchmark_id}|${r.rec.target_model}`;
      if (!by.has(k)) by.set(k, []);
      by.get(k).push(r);
    }
    return [...by.values()].map((laneRuns) => {
      const { latest, previous, history } = splitHistory(laneRuns);
      return {
        latest,
        previous,
        history,
        band: historyBand(history),
        color: colorFor(latest.rec.target_model),
        dash: latest.dash,
        variant: latest.variant,
        model: latest.rec.target_model
      };
    });
  });

  const ext = $derived.by(() => {
    const cs = runs.flatMap((r) => r.pts.map((p) => p.c));
    const vs = [
      ...lanes.flatMap((l) => [l.latest, l.previous].filter(Boolean).flatMap((r) => r.pts.map((p) => p.v))),
      ...lanes.flatMap((l) => l.band.flatMap((b) => [b.lo, b.hi]))
    ];
    const [c0, c1] = [Math.min(...cs), Math.max(...cs)];
    const [lo, hi] = [Math.min(...vs), Math.max(...vs)];
    const pad = (hi - lo || Math.abs(hi) || 1) * 0.12;
    // tok/s is never negative; a padded floor below zero would be a lie about
    // the scale.
    return { l0: Math.log2(c0), l1: Math.log2(c1), v0: Math.max(0, lo - pad), v1: hi + pad };
  });

  const x = (c) =>
    PL + (ext.l1 === ext.l0 ? 0.5 : (Math.log2(c) - ext.l0) / (ext.l1 - ext.l0)) * (W - PL - PR);
  const y = (v) => PT + (1 - (v - ext.v0) / (ext.v1 - ext.v0 || 1)) * (H - PT - PB);
  const path = (pts) => pts.map((p, i) => `${i ? 'L' : 'M'}${x(p.c).toFixed(1)} ${y(p.v).toFixed(1)}`).join(' ');

  const fmtV = (v) => (Math.abs(v) >= 1000 ? Math.round(v).toLocaleString('en-US') : +v.toFixed(1));
  const cTicks = $derived([...new Set(runs.flatMap((r) => r.pts.map((p) => p.c)))].sort((a, b) => a - b));
  const yTicks = $derived([ext.v0, (ext.v0 + ext.v1) / 2, ext.v1]);

  const models = $derived([...new Set(lanes.map((l) => l.model))]);
  const variants = $derived([...new Map(lanes.filter((l) => l.variant).map((l) => [l.variant, l])).values()]);
  const hasHistory = $derived(lanes.some((l) => l.band.length > 1));

  const endLabels = $derived.by(() => {
    const ends = lanes
      .filter((l) => l.latest?.pts.length)
      .map((l) => ({ l, p: l.latest.pts[l.latest.pts.length - 1] }));
    const placed = dodgeLabels(
      ends.map((e) => y(e.p.v) - 9),
      { height: LABEL_H, top: PT + 7, bottom: H - PB - 6 }
    );
    return ends.map((e, i) => ({
      text: fmtV(e.p.v),
      color: e.l.color,
      x: Math.min(x(e.p.c), W - PR - 4),
      y: placed[i]
    }));
  });
</script>

<figure class="gate-panel">
  <figcaption class="gate-panel-head">
    <span class="gate-panel-title">{panel.title}</span>
    <span class="gate-panel-unit">{panel.unit}</span>
    <span class="gate-legend">
      {#if models.length > 1}
        <span class="gl-group">
          {#each models as m}
            <span class="gate-legend-item">
              <svg class="gl-swatch" viewBox="0 0 20 10" aria-hidden="true">
                <line x1="1" y1="5" x2="19" y2="5" stroke={colorFor(m)} stroke-width="2" />
                <circle cx="10" cy="5" r="3" fill={colorFor(m)} />
              </svg>{m.split('/').pop()}
            </span>
          {/each}
        </span>
      {/if}
      {#if variants.length > 0}
        {#if models.length > 1}<span class="gl-sep" aria-hidden="true"></span>{/if}
        <span class="gl-group">
          {#each variants as v}
            <span class="gate-legend-item">
              <svg class="gl-swatch" viewBox="0 0 20 10" aria-hidden="true">
                <line x1="1" y1="5" x2="19" y2="5" stroke="currentColor" stroke-width="2"
                  stroke-dasharray={v.dash} stroke-linecap="round" />
              </svg>{v.variant}
            </span>
          {/each}
        </span>
      {/if}
      {#if hasHistory}
        <span class="gl-sep" aria-hidden="true"></span>
        <span class="gl-group">
          <span class="gate-legend-item">
            <svg class="gl-swatch" viewBox="0 0 20 10" aria-hidden="true">
              <rect x="1" y="2" width="18" height="6" fill="currentColor" opacity="0.16" />
            </svg>earlier runs
          </span>
        </span>
      {/if}
    </span>
  </figcaption>

  <svg viewBox="0 0 {W} {H}" role="img" aria-label="{panel.title} for the latest gate runs">
    <defs>
      <!-- Two variants share a hue, so their corridors cannot be told apart by
           opacity alone. Texture is the channel that survives that, and it also
           survives printing and any colour deficiency. -->
      <pattern id="gl-hatch" width="6" height="6" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">
        <line x1="0" y1="0" x2="0" y2="6" stroke="currentColor" stroke-width="1.4" opacity="0.5" />
      </pattern>
    </defs>

    {#each yTicks as t}
      <line class="gc-grid" x1={PL} y1={y(t)} x2={W - PR} y2={y(t)} />
      <text class="gc-axis" x={PL - 8} y={y(t) + 3.5} text-anchor="end">{fmtV(t)}</text>
    {/each}
    {#each cTicks as c}
      <text class="gc-axis" x={x(c)} y={H - 8} text-anchor="middle">C={c}</text>
    {/each}

    {#each lanes as lane}
      {#if lane.band.length > 1}
        <path
          class="gl-band"
          d={bandPath(lane.band, x, y)}
          fill={lane.dash ? 'url(#gl-hatch)' : lane.color}
          fill-opacity={lane.dash ? 1 : 0.1}
          style={lane.dash ? `color:${lane.color}` : undefined}
          stroke="none"
        >
          <title>range over {lane.history.length} earlier runs{lane.variant ? ` · ${lane.variant}` : ''} · {fmtDate(lane.history[0].rec.recorded_at)} – {fmtDate(lane.history[lane.history.length - 1].rec.recorded_at)}</title>
        </path>
      {/if}
    {/each}

    {#each lanes as lane}
      {#each [lane.previous, lane.latest].filter(Boolean) as r, i}
        {@const isLatest = i === [lane.previous, lane.latest].filter(Boolean).length - 1}
        <g opacity={isLatest ? 1 : 0.45}>
          <path d={path(r.pts)} fill="none" stroke={lane.color} stroke-width={isLatest ? 2 : 1.25}
            stroke-dasharray={lane.dash} stroke-linejoin="round" stroke-linecap="round" />
          {#each r.pts as p}
            <g
              class="gc-pt"
              role="button"
              tabindex="0"
              aria-label="C={p.c}: {fmtV(p.v)} tok/s{lane.variant ? ', ' + lane.variant : ''} on {fmtDate(r.rec.recorded_at)}, {r.rec.verdict} — details"
              onclick={() => onselect([r.rec])}
              onkeydown={(e) => (e.key === 'Enter' || e.key === ' ') && (e.preventDefault(), onselect([r.rec]))}
            >
              <title>C={p.c} · {fmtV(p.v)} tok/s{lane.variant ? ' · ' + lane.variant : ''} · {fmtDate(r.rec.recorded_at)} · {r.rec.verdict} · click for record</title>
              <circle class="gc-hit" cx={x(p.c)} cy={y(p.v)} r="11" />
              {#if r.rec.verdict === 'PASS'}
                <circle class="gc-mark" cx={x(p.c)} cy={y(p.v)} r={isLatest ? 3.5 : 2.5} fill={lane.color}
                  stroke="var(--card)" stroke-width="1" />
              {:else}
                <circle class="gc-mark gc-fail" cx={x(p.c)} cy={y(p.v)} r="4.5" fill="var(--card)"
                  stroke={lane.color} stroke-width="2" />
              {/if}
            </g>
          {/each}
        </g>
      {/each}
    {/each}

    {#each endLabels as l}
      <text class="gc-val" x={l.x} y={l.y + 3} text-anchor="end" fill={l.color}>{l.text}</text>
    {/each}
  </svg>
</figure>
