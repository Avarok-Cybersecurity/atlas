<script>
  // One metric panel of the benchmark dashboard: a time-scaled line per
  // (metric, variant, MODEL), reference lines for budgets/floors read from the
  // records, and a click on any point raising the underlying record(s) for the
  // metadata card. Same hand-rolled SVG dialect as StarChart.svelte — no chart
  // library.
  //
  // Everything decidable lives in pure modules so it can be tested directly:
  //   gate-series.js   which points belong to which line
  //   gate-domain.js   where the axis starts and stops, and label dodging
  //   gate-aggregate.js how many runs one drawn point stands for
  //   chart-marks.js   the marker glyphs, shared with the legend
  import { colorFor, fmtDate, shortModel } from '$lib/gates.js';
  import { buildSeries, drawnValues, modelsOf } from '$lib/gate-series.js';
  import { clampValue, dodgeLabels, robustDomain, tickLabel } from '$lib/gate-domain.js';
  import { clipCaret, loneTriangle } from '$lib/chart-marks.js';

  let { records, panel, onselect } = $props();

  const W = 720, H = 232, PL = 56, PR = 16, PT = 14, PB = 30;
  const LABEL_H = 13;

  const series = $derived(buildSeries(panel, records));
  const refLines = $derived([...(panel.caps ?? []), ...(panel.floors ?? [])]);

  const ext = $derived.by(() => {
    const ts = series.flatMap((s) => s.nodes.map((n) => n.t));
    const [t0, t1] = [Math.min(...ts), Math.max(...ts)];
    // An explicit panel domain still wins: `webserver_ok` is bounded by the
    // iteration count, and nothing can exceed it, so there is nothing to clip.
    if (panel.domain) {
      return { t0, t1, v0: panel.domain[0], v1: panel.domain[1], clipHigh: false, clipLow: false };
    }
    const d = robustDomain(drawnValues(series), refLines) ?? { v0: 0, v1: 1, clipHigh: false, clipLow: false };
    return { t0, t1, ...d };
  });

  const x = (t) => PL + (ext.t1 === ext.t0 ? 0.5 : (t - ext.t0) / (ext.t1 - ext.t0)) * (W - PL - PR);
  const y = (v) => PT + (1 - (v - ext.v0) / (ext.v1 - ext.v0 || 1)) * (H - PT - PB);
  /** Plot position of a node, and whether that position understates its value. */
  const at = (n) => {
    const { y: cy, clamped } = clampValue(n.v, ext);
    return { px: x(n.t), py: y(cy), clamped };
  };
  const seg = (a, b) => `M${x(a.t).toFixed(1)} ${at(a).py.toFixed(1)}L${x(b.t).toFixed(1)} ${at(b).py.toFixed(1)}`;

  const fmtV = (v) => (Math.abs(v) >= 1000 ? Math.round(v).toLocaleString('en-US') : +v.toFixed(2));
  // Axis ticks read as landmarks, not data — round them (fine values live on
  // the points, their tooltips and the metadata card).
  const fmtTick = (v) => {
    const range = ext.v1 - ext.v0;
    if (range >= 50) return Math.round(v).toLocaleString('en-US');
    return +v.toFixed(range >= 5 ? 0 : 1);
  };
  const yTicks = $derived([
    { v: ext.v0, edge: ext.clipLow ? 'low' : null },
    { v: (ext.v0 + ext.v1) / 2, edge: null },
    { v: ext.v1, edge: ext.clipHigh ? 'high' : null }
  ]);
  const xTicks = $derived.by(() => {
    const ts = [...new Set(series.flatMap((s) => s.nodes.map((n) => n.t)))].sort((a, b) => a - b);
    const picked = ts.length <= 2 ? ts : [ts[0], ts[Math.floor(ts.length / 2)], ts[ts.length - 1]];
    return picked.map((t, i) => ({
      t,
      anchor: i === 0 ? 'start' : i === picked.length - 1 ? 'end' : 'middle',
      ax: i === 0 ? PL : i === picked.length - 1 ? W - PR : x(t)
    }));
  });

  // End-of-series value labels, spread apart so two series ending at similar
  // values cannot print on top of each other.
  const endLabels = $derived.by(() => {
    const ends = series
      .filter((s) => s.nodes.length > 0)
      .map((s) => ({ s, node: s.nodes[s.nodes.length - 1] }));
    const placed = dodgeLabels(
      ends.map((e) => at(e.node).py - 9),
      { height: LABEL_H, top: PT + 7, bottom: H - PB - 6 }
    );
    return ends.map((e, i) => ({
      text: fmtV(e.node.v),
      color: colorFor(e.s.model),
      x: Math.min(at(e.node).px, W - PR - 4),
      y: placed[i],
      // A label pushed far from its point needs a thread back to it.
      leader: Math.abs(placed[i] - (at(e.node).py - 9)) > 14 ? at(e.node) : null
    }));
  });

  const models = $derived(modelsOf(series));
  const hasFail = $derived(series.some((s) => s.nodes.some((n) => !n.allPass)));
  const hasAgg = $derived(series.some((s) => s.nodes.some((n) => n.aggregated)));
  const hasClip = $derived(ext.clipHigh || ext.clipLow);
  const hasLone = $derived(series.some((s) => s.sparse));
  const variantKeys = $derived(
    [...new Map(series.map((s) => [`${s.metricKey}|${s.variant ?? ''}`, s])).values()]
  );
  const showVariantKey = $derived(new Set(series.map((s) => s.metricKey)).size > 1 || series.some((s) => s.variant));

  const describe = (s, n) => {
    const when = n.aggregated ? `${fmtDate(n.tMin)} – ${fmtDate(n.tMax)}` : fmtDate(n.t);
    const what = n.aggregated ? `median of ${n.count} runs` : s.label;
    const unit = panel.unit === 'ms' || panel.unit === 's' ? ` ${panel.unit}` : '';
    const off = at(n).clamped ? ' · beyond the axis' : '';
    return `${what} ${fmtV(n.v)}${unit}${off} · ${when} · ${n.allPass ? 'PASS' : 'has a failure'}`;
  };
</script>

<figure class="gate-panel">
  <figcaption class="gate-panel-head">
    <span class="gate-panel-title">{panel.title}</span>
    <span class="gate-panel-unit">{panel.unit}</span>
    {#if models.length > 1 || showVariantKey || hasFail || hasAgg || hasClip}
      <span class="gate-legend">
        {#if models.length > 1}
          <span class="gl-group">
            {#each models as m}
              <span class="gate-legend-item">
                <svg class="gl-swatch" viewBox="0 0 20 10" aria-hidden="true">
                  <line x1="1" y1="5" x2="19" y2="5" stroke={colorFor(m)} stroke-width="2" />
                  <circle cx="10" cy="5" r="3" fill={colorFor(m)} />
                </svg>{shortModel(m)}
              </span>
            {/each}
          </span>
        {/if}
        {#if showVariantKey && variantKeys.length > 1}
          <span class="gl-sep" aria-hidden="true"></span>
          <span class="gl-group">
            {#each variantKeys as s}
              <span class="gate-legend-item">
                <svg class="gl-swatch" viewBox="0 0 20 10" aria-hidden="true">
                  <line x1="1" y1="5" x2="19" y2="5" stroke="currentColor" stroke-width="2"
                    stroke-dasharray={s.dashed ? '5 4' : 'none'} />
                </svg>{s.metricLabel}
              </span>
            {/each}
          </span>
        {/if}
        {#if hasFail || hasAgg || hasClip || hasLone}
          <span class="gl-sep" aria-hidden="true"></span>
          <span class="gl-group">
            {#if hasFail}
              <span class="gate-legend-item">
                <svg class="gl-swatch" viewBox="0 0 12 10" aria-hidden="true"
                  ><circle cx="6" cy="5" r="3.5" fill="none" stroke="currentColor" stroke-width="1.8" /></svg>
                fail
              </span>
            {/if}
            {#if hasAgg}
              <span class="gate-legend-item">
                <svg class="gl-swatch" viewBox="0 0 12 10" aria-hidden="true"
                  ><circle cx="6" cy="5" r="2.2" fill="currentColor" /><circle cx="6" cy="5" r="4.4"
                    fill="none" stroke="currentColor" stroke-width="1" opacity="0.55" /></svg>
                median of N
              </span>
            {/if}
            {#if hasLone}
              <span class="gate-legend-item">
                <svg class="gl-swatch" viewBox="0 0 12 10" aria-hidden="true"
                  ><path d={loneTriangle(6, 5, 4)} fill="currentColor" /></svg>
                single run
              </span>
            {/if}
            {#if hasClip}
              <span class="gate-legend-item">
                <svg class="gl-swatch" viewBox="0 0 12 10" aria-hidden="true"
                  ><path d={clipCaret(6, 5, 'high')} fill="none" stroke="currentColor" stroke-width="1.8"
                    stroke-linecap="round" stroke-linejoin="round" /></svg>
                off scale
              </span>
            {/if}
          </span>
        {/if}
      </span>
    {/if}
  </figcaption>

  <svg viewBox="0 0 {W} {H}" role="img" aria-label="{panel.title} across gate runs">
    {#each yTicks as t}
      <line class="gc-grid" class:gc-grid-clipped={t.edge} x1={PL} y1={y(t.v)} x2={W - PR} y2={y(t.v)} />
      <text class="gc-axis" x={PL - 8} y={y(t.v) + 3.5} text-anchor="end">{tickLabel(fmtTick(t.v), t.edge)}</text>
    {/each}
    {#each xTicks as t}
      <text class="gc-axis" x={t.ax} y={H - 8} text-anchor={t.anchor}>{fmtDate(t.t)}</text>
    {/each}

    {#each refLines as l}
      <line class="gc-ref" x1={PL} y1={y(l.value)} x2={W - PR} y2={y(l.value)} />
      <text class="gc-ref-label" x={W - PR} y={y(l.value) - 5} text-anchor="end">{l.label}</text>
    {/each}

    <!-- Spread of each aggregated group, beneath the lines so it reads as
         context rather than as data of its own. -->
    {#each series as s}
      {@const c = colorFor(s.model)}
      {#each s.nodes.filter((n) => n.aggregated) as n}
        {@const hi = y(clampValue(n.vMax, ext).y)}
        {@const lo = y(clampValue(n.vMin, ext).y)}
        {#if Math.abs(lo - hi) >= 3}
          <line class="gc-spread" x1={x(n.t)} y1={hi} x2={x(n.t)} y2={lo} stroke={c} />
        {/if}
      {/each}
    {/each}

    {#each series as s}
      {@const c = colorFor(s.model)}
      {#if !s.sparse}
        {#each s.edges as e}
          <path class="gc-line" d={seg(e.a, e.b)} fill="none" stroke={c} stroke-width="2"
            stroke-dasharray={s.dashed ? '5 4' : 'none'} stroke-linejoin="round" stroke-linecap="round" />
        {/each}
      {/if}
      {#each s.nodes as n}
        {@const p = at(n)}
        <g
          class="gc-pt"
          role="button"
          tabindex="0"
          aria-label="{describe(s, n)} — {n.count > 1 ? `${n.count} records` : 'details'}"
          onclick={() => onselect(n.members.map((m) => m.rec))}
          onkeydown={(e) => (e.key === 'Enter' || e.key === ' ') && (e.preventDefault(), onselect(n.members.map((m) => m.rec)))}
        >
          <title>{describe(s, n)} · click for {n.count > 1 ? `the ${n.count} records` : 'the record'}</title>
          <circle class="gc-hit" cx={p.px} cy={p.py} r="11" />
          {#if p.clamped}
            <path class="gc-mark gc-clip" d={clipCaret(p.px, p.py, p.clamped)} fill="none"
              stroke={c} stroke-width="2" stroke-linecap="round" stroke-linejoin="round" />
          {:else if !n.allPass}
            <circle class="gc-mark gc-fail" cx={p.px} cy={p.py} r="4.5" fill="var(--card)" stroke={c} stroke-width="2" />
          {:else if n.aggregated}
            <g class="gc-mark gc-agg">
              <circle cx={p.px} cy={p.py} r="3" fill={c} />
              <circle cx={p.px} cy={p.py} r="5.5" fill="none" stroke={c} stroke-width="1.25" opacity="0.55" />
            </g>
          {:else if s.sparse}
            <path class="gc-mark gc-lone" d={loneTriangle(p.px, p.py)} fill={c} stroke="var(--card)" stroke-width="1" />
          {:else}
            <circle class="gc-mark" cx={p.px} cy={p.py} r="3.5" fill={c} stroke="var(--card)" stroke-width="1" />
          {/if}
        </g>
      {/each}
    {/each}

    {#each endLabels as l}
      {#if l.leader}
        <line class="gc-leader" x1={l.x} y1={l.y} x2={l.leader.px} y2={l.leader.py} stroke={l.color} />
      {/if}
      <text class="gc-val" x={l.x} y={l.y + 3} text-anchor="end" fill={l.color}>{l.text}</text>
    {/each}
  </svg>
</figure>
