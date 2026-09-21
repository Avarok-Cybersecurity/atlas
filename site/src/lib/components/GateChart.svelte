<script>
  // One metric panel of the benchmark dashboard: a time-scaled line per
  // (metric, variant, MODEL), the gate's floor/ceiling stepped under it as it
  // was in force at each run, and a click on any point raising the underlying
  // record(s) for the metadata card. Same hand-rolled SVG dialect as
  // StarChart.svelte — no chart library.
  //
  // Everything decidable lives in pure modules so it can be tested directly:
  //   gate-series.js   which points belong to which line
  //   gate-limits.js   which floor/ceiling judged a point, and the step geometry
  //   gate-domain.js   where the axis starts and stops, and label dodging
  //   gate-aggregate.js how many runs one drawn point stands for
  //   chart-marks.js   the marker glyphs, shared with the legend
  import { colorFor, fmtDate, latestLimitChange, limitAsOf, limitFor, shortModel } from '$lib/gates.js';
  import { buildSeries, drawnValues, modelsOf } from '$lib/gate-series.js';
  import { fmtLimit, limitLabel, stepPath, timeSpans, violationOf } from '$lib/gate-limits.js';
  import { clampValue, dodgeLabels, robustDomain, tickLabel } from '$lib/gate-domain.js';
  import { clipCaret, loneTriangle } from '$lib/chart-marks.js';

  let { records, panel, onselect } = $props();

  const W = 720, H = 232, PL = 56, PR = 16, PT = 14, PB = 30;
  const LABEL_H = 13;

  const series = $derived(buildSeries(panel, records));
  // The limit that judged a node is its representative record's — read per
  // node, so a floor ratcheted mid-history steps exactly where it stepped.
  const limitOf = (s, n) => limitFor(n.rec, s.metricKey);
  // Every bound in play widens the axis: a rule is part of the gate's claim —
  // including one re-cut after the newest record (drawn at its own date).
  const refLines = $derived(
    series.flatMap((s) => {
      const limits = s.nodes.map((n) => limitOf(s, n));
      const recut = recutAfter(s);
      if (recut !== null) limits.push(limitAsOf(s.nodes[s.nodes.length - 1].rec, s.metricKey, recut));
      return limits.flatMap((l) => [l.min, l.max].filter((v) => v !== null).map((value) => ({ value })));
    })
  );

  // A re-cut that post-dates the newest record: the time axis reaches out to
  // it so the step is drawn on its day, in an otherwise empty stretch, rather
  // than not at all. Per series, because each has its own declaration.
  const recutAfter = (s) => {
    const last = s.nodes[s.nodes.length - 1];
    const since = last ? latestLimitChange(last.rec, s.metricKey) : null;
    return since !== null && since > last.t ? since : null;
  };
  const ext = $derived.by(() => {
    const ts = series.flatMap((s) => s.nodes.map((n) => n.t));
    const recuts = series.map(recutAfter).filter((t) => t !== null);
    const t0 = Math.min(...ts);
    const recut = recuts.length ? Math.max(...recuts) : null;
    // A little room past the re-cut, so the new bar has a stretch to be seen on.
    const t1 = recut === null ? Math.max(...ts) : recut + (recut - t0) * 0.04;
    // An explicit panel domain still wins: `webserver_ok` is bounded by the
    // iteration count, and nothing can exceed it, so there is nothing to clip.
    if (panel.domain) {
      return { t0, t1, recut, v0: panel.domain[0], v1: panel.domain[1], clipHigh: false, clipLow: false };
    }
    const d = robustDomain(drawnValues(series), refLines) ?? { v0: 0, v1: 1, clipHigh: false, clipLow: false };
    return { t0, t1, recut, ...d };
  });

  const x = (t) => PL + (ext.t1 === ext.t0 ? 0.5 : (t - ext.t0) / (ext.t1 - ext.t0)) * (W - PL - PR);
  const y = (v) => PT + (1 - (v - ext.v0) / (ext.v1 - ext.v0 || 1)) * (H - PT - PB);
  /** Plot position of a node, and whether that position understates its value. */
  const at = (n) => {
    const { y: cy, clamped } = clampValue(n.v, ext);
    return { px: x(n.t), py: y(cy), clamped };
  };
  // One line through a series' nodes in time order; a clipped node's segment
  // meets it at the clamp row, where its caret sits.
  const linePath = (nodes) =>
    nodes
      .map((n, i) => {
        const p = at(n);
        return `${i ? 'L' : 'M'}${p.px.toFixed(1)} ${p.py.toFixed(1)}`;
      })
      .join(' ');

  const unitSuffix = $derived(['ms', 's', 'tok/s'].includes(panel.unit) ? panel.unit : '');
  // A floor's label hangs under its rule and a ceiling's rides above it, so
  // the passing side is the one the text is not on; both stay in the field.
  const labelY = (bound, py) =>
    bound === 'min' ? Math.min(py + 11, H - PB - 2) : Math.max(py - 4, PT + 8);
  const MIN_LABELLED_SPAN = 48;
  // The stepped floor/ceiling per series: one span per stretch of history one
  // bound judged, labelled where the value changes and at the right edge.
  // Two series under one rule (the two concurrency variants share a model
  // and a peak floor) draw it once.
  const limitLines = $derived.by(() => {
    const seen = new Set();
    const out = [];
    for (const s of series) {
      const points = s.nodes.map((n) => ({ x: x(n.t), limit: limitOf(s, n) }));
      const recut = recutAfter(s);
      if (recut !== null) {
        const last = s.nodes[s.nodes.length - 1];
        points.push({ x: x(recut), limit: limitAsOf(last.rec, s.metricKey, recut) });
      }
      const spans = timeSpans(points, PL, W - PR);
      for (const bound of ['min', 'max']) {
        const d = stepPath(spans, bound, (v) => y(clampValue(v, ext).y));
        if (!d || seen.has(d)) continue;
        seen.add(d);
        const carried = spans.filter((sp) => sp[bound] !== null);
        // The rule in force now is labelled at the right edge, where every
        // reference line on this dashboard has always been read; an earlier
        // stretch (a bound since ratcheted) is labelled where it began, if it
        // is wide enough to hold the text.
        const labels = carried.flatMap((sp, i) => {
          const last = i === carried.length - 1;
          if (!last && sp.x1 - sp.x0 < MIN_LABELLED_SPAN) return [];
          return [
            {
              text: limitLabel(bound, sp[bound], unitSuffix),
              color: colorFor(s.model),
              x: last ? W - PR : sp.x0 + 3,
              anchor: last ? 'end' : 'start',
              y: labelY(bound, y(clampValue(sp[bound], ext).y))
            }
          ];
        });
        out.push({ d, color: colorFor(s.model), labels });
      }
    }
    return out;
  });
  const stepLabels = $derived(limitLines.flatMap((l) => l.labels.filter((t) => t.anchor === 'start')));

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
    // The axis may end on a re-cut rather than a record; the edge tick says so.
    if (ext.recut !== null) ts.push(ext.recut);
    const picked = ts.length <= 2 ? ts : [ts[0], ts[Math.floor(ts.length / 2)], ts[ts.length - 1]];
    return picked.map((t, i) => ({
      t,
      anchor: i === 0 ? 'start' : i === picked.length - 1 ? 'end' : 'middle',
      ax: i === 0 ? PL : i === picked.length - 1 ? W - PR : x(t)
    }));
  });

  // Everything labelled at the right edge — each series' end value and the
  // rule now in force — dodged apart in ONE pass, so a floor cannot print on
  // a final value, nor a median ceiling on another model's p90 ceiling.
  const edgeLabels = $derived.by(() => {
    const ends = series
      .filter((s) => s.nodes.length > 0)
      .map((s) => ({ s, node: s.nodes[s.nodes.length - 1] }));
    const rules = limitLines.flatMap((l) => l.labels.filter((t) => t.anchor === 'end'));
    const placed = dodgeLabels(
      [...ends.map((e) => at(e.node).py - 9), ...rules.map((t) => t.y)],
      { height: LABEL_H, top: PT + 7, bottom: H - PB - 6 }
    );
    return {
      end: ends.map((e, i) => ({
        text: fmtV(e.node.v),
        color: colorFor(e.s.model),
        x: Math.min(at(e.node).px, W - PR - 4),
        y: placed[i],
        // A label pushed far from its point needs a thread back to it.
        leader: Math.abs(placed[i] - (at(e.node).py - 9)) > 14 ? at(e.node) : null
      })),
      limit: rules.map((t, i) => ({ ...t, y: placed[ends.length + i] }))
    };
  });

  const models = $derived(modelsOf(series));
  const hasFail = $derived(series.some((s) => s.nodes.some((n) => !n.allPass)));
  const hasAgg = $derived(series.some((s) => s.nodes.some((n) => n.aggregated)));
  const hasClip = $derived(ext.clipHigh || ext.clipLow);
  const hasLone = $derived(series.some((s) => s.sparse));
  const hasLimitLine = $derived(limitLines.length > 0);
  const hasViol = $derived(series.some((s) => s.nodes.some((n) => violationOf(n.v, limitOf(s, n)))));
  const variantKeys = $derived(
    [...new Map(series.map((s) => [`${s.metricKey}|${s.variant ?? ''}`, s])).values()]
  );
  const showVariantKey = $derived(new Set(series.map((s) => s.metricKey)).size > 1 || series.some((s) => s.variant));

  const describe = (s, n) => {
    const when = n.aggregated ? `${fmtDate(n.tMin)} – ${fmtDate(n.tMax)}` : fmtDate(n.t);
    const what = n.aggregated ? `median of ${n.count} runs` : s.label;
    const unit = unitSuffix ? ` ${unitSuffix}` : '';
    const off = at(n).clamped ? ' · beyond the axis' : '';
    const limit = limitOf(s, n);
    const viol = violationOf(n.v, limit);
    const broke = viol === 'floor' ? ` · below floor ${fmtLimit(limit.min)}` : viol === 'ceiling' ? ` · over ceiling ${fmtLimit(limit.max)}` : '';
    return `${what} ${fmtV(n.v)}${unit}${off}${broke} · ${when} · ${n.allPass ? 'PASS' : 'has a failure'}`;
  };
</script>

<figure class="gate-panel">
  <figcaption class="gate-panel-head">
    <span class="gate-panel-title">{panel.title}</span>
    <span class="gate-panel-unit">{panel.unit}</span>
    {#if models.length > 1 || showVariantKey || hasFail || hasAgg || hasClip || hasLimitLine || hasViol}
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
        {#if hasFail || hasAgg || hasClip || hasLone || hasLimitLine || hasViol}
          <span class="gl-sep" aria-hidden="true"></span>
          <span class="gl-group">
            {#if hasLimitLine}
              <span class="gate-legend-item">
                <svg class="gl-swatch" viewBox="0 0 20 10" aria-hidden="true">
                  <path class="gc-limit" d="M1 5 H19" fill="none" stroke="currentColor" />
                </svg>
                gate floor / ceiling
              </span>
            {/if}
            {#if hasViol}
              <span class="gate-legend-item">
                <svg class="gl-swatch" viewBox="0 0 12 10" aria-hidden="true"
                  ><circle class="gc-viol" cx="6" cy="5" r="4.2" /></svg>
                past its limit
              </span>
            {/if}
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

    <!-- The gate's rule, stepped as it was in force. Under the data: it is
         what the points are judged against, not a reading of its own. -->
    {#each limitLines as l}
      <path class="gc-limit" d={l.d} fill="none" stroke={l.color} />
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
      <!-- A single run gets the MEASURED run-to-run envelope of its instrument
           instead. Only on non-aggregated nodes: an aggregated group already
           shows the span of its own members, and an observed span must never be
           replaced by an imputed one. -->
      {#if s.envelope}
        {#each s.nodes.filter((n) => !n.aggregated) as n}
          {@const hi = y(clampValue(n.v * s.envelope.hi, ext).y)}
          {@const lo = y(clampValue(n.v * s.envelope.lo, ext).y)}
          {#if Math.abs(lo - hi) >= 3}
            <line class="gc-spread" x1={x(n.t)} y1={hi} x2={x(n.t)} y2={lo} stroke={c} />
          {/if}
        {/each}
      {/if}
    {/each}

    {#each series as s}
      {@const c = colorFor(s.model)}
      {#if !s.sparse}
        <path class="gc-line" d={linePath(s.nodes)} fill="none" stroke={c} stroke-width="2"
          stroke-dasharray={s.dashed ? '5 4' : 'none'} stroke-linejoin="round" stroke-linecap="round" />
      {/if}
      {#each s.nodes as n}
        {@const p = at(n)}
        {@const viol = violationOf(n.v, limitOf(s, n))}
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
          {#if viol}
            <circle class="gc-viol" cx={p.px} cy={p.py} r="7.5" data-limit={viol} />
          {/if}
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

    <!-- Rule labels ride above the marks, like the end values: a label under
         a cluster of points is a label nobody can read. -->
    {#each [...stepLabels, ...edgeLabels.limit] as t}
      <text class="gc-limit-label" x={t.x} y={t.y} text-anchor={t.anchor} fill={t.color}>{t.text}</text>
    {/each}
    {#each edgeLabels.end as l}
      {#if l.leader}
        <line class="gc-leader" x1={l.x} y1={l.y} x2={l.leader.px} y2={l.leader.py} stroke={l.color} />
      {/if}
      <text class="gc-val" x={l.x} y={l.y + 3} text-anchor="end" fill={l.color}>{l.text}</text>
    {/each}
  </svg>
</figure>
