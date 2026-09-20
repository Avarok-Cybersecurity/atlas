<script>
  // Chart A of the Cost section: dollars per million tokens against the
  // concurrency rung, Atlas and every comparable vLLM series on one axis.
  //
  // The SVG dialect, the W/H/PL constants and the log2 rung axis are
  // ConcurrencyComparison's, copied rather than imported: that component is
  // the throughput comparison and owns its own decisions, and a shared base
  // would couple two charts that are allowed to diverge.
  //
  // The y axis is LOGARITHMIC and says so. J/token spans ~27x across the
  // ladder (3.5 at C=1 to 0.13 at C=128 on the committed proxy); on a linear
  // axis every wide rung collapses onto the baseline and the very ratio this
  // chart exists to show becomes unreadable. On a log axis a cost ratio is a
  // constant vertical offset, which is what a reader is here to compare.
  //
  // Three honesty rules are IN THE MARKS, not in a footnote:
  //   * a losing rung is drawn exactly like a winning one, plus the factor by
  //     which it loses — there is no filter and no default view that hides it;
  //   * an under-sampled window is HOLLOW and is in no tile, verdict or trend;
  //   * the rail is named inside the plot area, because every figure on this
  //     axis is a lower bound on system energy.
  import { colorFor } from '$lib/gates.js';
  import { dashFor } from '$lib/gate-variants.js';
  import { costPerMillion, fmtUsd } from '$lib/cost.js';

  /**
   * @type {{
   *   subject: object,
   *   cost: object,        the costLadder() result
   *   usdPerKwh: number,
   *   rungs: number[],
   *   title: string,
   *   aboveIdle: boolean,
   *   onselect: (recs: object[]) => void
   * }}
   */
  let { subject, cost, usdPerKwh, rungs, title, aboveIdle = false, onselect } = $props();

  const W = 720, H = 260, PL = 62, PR = 16, PT = 26, PB = 44;

  // The dollar value of one cell at the price now in the box. `above idle`
  // subtracts the resident-model idle draw, and is only ever offered when
  // both sides recorded one (cost.js#idleAvailability).
  const usdOf = (e) => {
    const j = aboveIdle && e.aboveIdleJ !== null ? e.aboveIdleJ : e.energyJ;
    return costPerMillion(j / e.tokens, usdPerKwh);
  };
  const measured = (s) => s.points.filter((p) => p.energy.state === 'measured');
  const seriesList = $derived([
    ...(cost.atlas ? [{ ...cost.atlas, role: 'atlas' }] : []),
    ...cost.baselines.map((b) => ({ ...b, role: 'baseline' }))
  ]);
  const values = $derived(seriesList.flatMap((s) => measured(s).map((p) => usdOf(p.energy))));

  // Log axis, padded by an eighth of a decade at each end so no mark sits on
  // a rule. A single value gets a decade around it rather than a zero span.
  const lo = $derived(values.length ? Math.log10(Math.min(...values)) - 0.125 : 0);
  const hi = $derived(values.length ? Math.log10(Math.max(...values)) + 0.125 : 1);
  const span = $derived(hi - lo < 0.2 ? 0.2 : hi - lo);
  const y = (v) => PT + (1 - (Math.log10(v) - lo) / span) * (H - PT - PB);
  const l0 = $derived(Math.log2(Math.min(...rungs)));
  const l1 = $derived(Math.log2(Math.max(...rungs)));
  const x = (c) => PL + (l1 === l0 ? 0.5 : (Math.log2(c) - l0) / (l1 - l0)) * (W - PL - PR);

  const yTicks = $derived(values.length ? [0, 0.5, 1].map((f) => 10 ** (lo + f * span)) : []);
  const pathOf = (s) =>
    measured(s)
      .sort((a, b) => a.c - b.c)
      .map((p, i) => `${i ? 'L' : 'M'}${x(p.c).toFixed(1)} ${y(usdOf(p.energy)).toFixed(1)}`)
      .join(' ');

  const color = $derived(colorFor(subject.checkpoint));
  const dash = $derived(dashFor(subject.gate));
  // A rung the record reached but the sampler did not: named, never a zero.
  const atlasMeasured = $derived(new Set(cost.atlas ? measured(cost.atlas).map((p) => p.c) : []));
  const atlasAbsent = $derived(
    cost.atlas ? cost.atlas.points.filter((p) => p.energy.state !== 'measured') : []
  );
  const verdictAt = (c) => cost.verdicts.rungs.find((r) => r.c === c) ?? null;
  // A losing rung's label must stay INSIDE the plot: at the widest rung the
  // centred text ran past the viewBox and the factor was clipped away, which
  // is precisely the statement this chart exists to make. The MARK keeps its
  // position; only the label's anchor moves at the two ends.
  const LOSE_HALF_W = 56; // half of ~18 monospace characters at 10px
  const loseAnchor = (c) =>
    x(c) - LOSE_HALF_W < PL ? 'start' : x(c) + LOSE_HALF_W > W - PR ? 'end' : 'middle';
  const loseX = (c) => ({ start: PL, end: W - PR, middle: x(c) })[loseAnchor(c)];
  const hasHollow = $derived(seriesList.some((s) => measured(s).some((p) => !p.energy.trusted)));
  const hasRing = $derived(seriesList.some((s) => measured(s).some((p) => p.energy.throttled)));

  const tip = (label, p) => {
    const e = p.energy;
    const parts = [
      `${label} · C=${p.c}`,
      `$${fmtUsd(usdOf(e))} per 1M tokens at ${usdPerKwh} $/kWh`,
      `${e.jPerToken.toFixed(3)} J/token`,
      `${e.tokPerWh.toFixed(0)} tok/Wh`,
      `${e.watts.toFixed(1)} W over the window`,
      `${e.tokens} tokens in ${e.windowS.toFixed(1)} s`,
      e.samples === null ? 'no sample count' : `${e.samples} power readings`
    ];
    if (e.swCapFrac !== null) parts.push(`sw power cap on ${(e.swCapFrac * 100).toFixed(0)}% of readings`);
    for (const c of e.concerns) parts.push(c);
    return parts.join(' · ');
  };
</script>

<figure class="gate-panel cost-chart">
  <figcaption class="gate-panel-head">
    <span class="gate-panel-title">{title}</span>
    <span class="gate-panel-unit">$ per 1M tokens · log scale</span>
    <span class="gate-legend">
      <span class="gl-group">
        {#if cost.atlas}
          <span class="gate-legend-item">
            <svg class="gl-swatch" viewBox="0 0 20 10" aria-hidden="true">
              <line x1="1" y1="5" x2="19" y2="5" stroke={color} stroke-width="2" stroke-dasharray={dash} />
              <circle cx="10" cy="5" r="3" fill={color} />
            </svg>{cost.atlas.label} · {cost.atlas.source} · {cost.atlas.build}
          </span>
        {/if}
        {#each cost.baselines as b}
          <span class="gate-legend-item">
            <svg class="gl-swatch" viewBox="0 0 20 10" aria-hidden="true">
              <line x1="1" y1="5" x2="19" y2="5" stroke="var(--t2)" stroke-width="1.5" />
              <rect x="6.5" y="1.5" width="7" height="7" fill="var(--t2)" />
            </svg>{b.label} · one-shot · not re-run
          </span>
        {/each}
        {#each cost.noEnergy as n}
          <span class="gate-legend-item">
            <svg class="gl-swatch" viewBox="0 0 20 10" aria-hidden="true">
              <rect x="6.5" y="1.5" width="7" height="7" fill="none" stroke="var(--t2)" stroke-width="1.5" />
            </svg>{n.label} · {n.why}
          </span>
        {/each}
      </span>
      {#if hasHollow || hasRing}
        <span class="gl-sep" aria-hidden="true"></span>
        <span class="gl-group">
          {#if hasHollow}
            <span class="gate-legend-item">
              <svg class="gl-swatch" viewBox="0 0 12 10" aria-hidden="true"
                ><circle cx="6" cy="5" r="3.5" fill="none" stroke="currentColor" stroke-width="1.8" /></svg>
              under-sampled — drawn, never counted
            </span>
          {/if}
          {#if hasRing}
            <span class="gate-legend-item">
              <svg class="gl-swatch" viewBox="0 0 12 10" aria-hidden="true"
                ><circle class="gc-viol" cx="6" cy="5" r="4.2" /></svg>
              HW power brake in the window
            </span>
          {/if}
        </span>
      {/if}
    </span>
  </figcaption>

  <svg viewBox="0 0 {W} {H}" role="img"
    aria-label="Cost per million tokens against concurrency for {subject.label}, GPU rail only, {cost.baselines.length ? 'Atlas against vLLM' : 'Atlas alone — no comparable vLLM energy'}">
    {#each yTicks as t}
      <line class="gc-grid" x1={PL} y1={y(t)} x2={W - PR} y2={y(t)} />
      <text class="gc-axis" x={PL - 8} y={y(t) + 3.5} text-anchor="end">${fmtUsd(t)}</text>
    {/each}
    {#each rungs as c}
      <text class="gc-axis" x={x(c)} y={H - 22} text-anchor="middle">C={c}</text>
    {/each}

    <!-- The rail, ON the chart. Every mark above it is a lower bound on the
         energy the machine actually drew. -->
    <text class="cost-rail" x={PL} y={PT - 12}>
      GPU rail only · Grace CPU and LPDDR5X are outside these numbers · lower bound
    </text>

    {#each atlasAbsent as p}
      <g class="cmp-absent">
        <title>C={p.c} · {p.energy.state === 'refused' ? p.energy.reason : 'no GPU-rail energy recorded for this rung'}</title>
        <line class="gc-grid gc-grid-clipped" x1={x(p.c)} y1={PT} x2={x(p.c)} y2={H - PB} />
        <text class="gc-ref-label" x={x(p.c)} y={PT + 10} text-anchor="middle">not measured</text>
      </g>
    {/each}

    {#each cost.baselines as b}
      <path d={pathOf(b)} fill="none" stroke="var(--t2)" stroke-width="1.5" stroke-linejoin="round" />
      {#each measured(b) as p}
        {@const py = y(usdOf(p.energy))}
        <g>
          <title>{tip(b.label, p)}</title>
          {#if p.energy.throttled}
            <circle class="gc-viol" cx={x(p.c)} cy={py} r="7.5" />
          {/if}
          <rect class="cmp-sq" x={x(p.c) - 3.5} y={py - 3.5} width="7" height="7"
            fill={p.energy.trusted ? 'var(--t2)' : 'none'} stroke="var(--t2)" stroke-width="1.5" />
        </g>
      {/each}
    {/each}

    {#if cost.atlas && atlasMeasured.size}
      <path d={pathOf(cost.atlas)} fill="none" stroke={color} stroke-width="2" stroke-dasharray={dash}
        stroke-linejoin="round" stroke-linecap="round" />
      {#each measured(cost.atlas) as p}
        {@const py = y(usdOf(p.energy))}
        {@const v = verdictAt(p.c)}
        <g
          class="gc-pt"
          role="button"
          tabindex="0"
          aria-label="C={p.c}: ${fmtUsd(usdOf(p.energy))} per million tokens{v?.loseLabel ? `, ${v.loseLabel}` : ''}{p.energy.trusted ? '' : ', under-sampled'} — details"
          onclick={() => cost.atlas.record && onselect([cost.atlas.record])}
          onkeydown={(e) =>
            (e.key === 'Enter' || e.key === ' ') &&
            (e.preventDefault(), cost.atlas.record && onselect([cost.atlas.record]))}
        >
          <title>{tip(cost.atlas.label, p)}{v ? ` · ${v.atlasCheaper ? `Atlas cheaper ×${(1 / v.ratio).toFixed(2)}` : v.loseLabel} vs ${v.rivalLabel}` : ''}</title>
          <circle class="gc-hit" cx={x(p.c)} cy={py} r="11" />
          {#if p.energy.throttled}
            <circle class="gc-viol" cx={x(p.c)} cy={py} r="7.5" />
          {/if}
          <circle class="gc-mark" cx={x(p.c)} cy={py} r="3.5"
            fill={p.energy.trusted ? color : 'var(--card)'} stroke={color} stroke-width="1.8" />
        </g>
        <!-- The losing label. Drawn under the point, in the same ink as every
             other mark: a loss is stated, not styled as an alarm. -->
        {#if v?.loseLabel}
          <text class="cost-lose" x={loseX(p.c)} y={Math.min(py + 15, H - PB - 2)} text-anchor={loseAnchor(p.c)}
            >{v.loseLabel}</text>
        {/if}
      {/each}
    {/if}
  </svg>
</figure>
