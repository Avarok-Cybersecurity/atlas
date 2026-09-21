<script module>
  // The comparison chart is the first block of every subject tab: Atlas
  // against vLLM on ONE instrument. Which chart that is depends on what the
  // subject has, and the decision is made in concurrency-comparison.js — pure
  // and exported — so the panel around it (header tiles, the bridging note
  // over the gate sweep) cannot disagree with the chart about what is drawn:
  //
  //   published  the frozen campaign pair from the subject's manifest (dense)
  //   live       the newest passing gate record on main, Atlas on the gate's
  //              instrument, plus every one-shot baseline whose instrument
  //              fingerprint equals the record's (ladder-baselines.js decides,
  //              axis by axis — a mismatch is named, never drawn)
  //   baseline   a vLLM one-shot on the published instrument, no Atlas leg at
  //              it yet (MoE): the vLLM curve and an honest empty Atlas state
  //   none       nothing to draw yet
  //
  // The generated ladders are keyed by subject id; `ladders` is a prop with
  // that file as its default so a test can prove the pair branch draws with
  // a baseline the real data does not yet contain.
  import LADDERS from '$lib/ladders.generated.json';
  import publishedLadder from '$lib/ladder.generated.json';
  import {
    baselineOnlyFor as _baselineOnlyFor,
    baselineTileOf as _baselineTileOf,
    batchCapOf,
    comparisonStateOf as _comparisonStateOf,
    instrumentLabel,
    ladderFor,
    liveRecordOf,
    measuredRange,
    oneShotChip,
    pairWith,
    publishedFor as _publishedFor
  } from '$lib/concurrency-comparison.js';

  export const publishedFor = (subject, ladders = LADDERS) => _publishedFor(subject, ladders);
  export const baselineOnlyFor = (subject, ladders = LADDERS) => _baselineOnlyFor(subject, ladders);
  export const comparisonStateOf = (subject, records, ladders = LADDERS) =>
    _comparisonStateOf(subject, records, ladders);
  export const baselineTileOf = (subject, records, ladders = LADDERS) => _baselineTileOf(subject, records, ladders);
  export { batchCapOf, instrumentLabel, liveRecordOf, measuredRange };
</script>

<script>
  import ConcurrencyLadder from './ConcurrencyLadder.svelte';
  import ConcurrencyBaseline from './ConcurrencyBaseline.svelte';
  import { colorFor, fmtDate, ladderPoints, rungFloors } from '$lib/gates.js';
  import { dashFor } from '$lib/gate-variants.js';
  import { fmtLimit, limitLabel, rungSpans, stepPath, violationOf } from '$lib/gate-limits.js';
  import { toggleSeries } from '$lib/series-visibility.js';

  let { subject, records, rungs, onselect, ladders = LADDERS } = $props();

  const ladder = $derived(ladderFor(subject, ladders));
  const published = $derived(publishedFor(subject, ladders));
  const live = $derived(liveRecordOf(records));
  const state = $derived(comparisonStateOf(subject, records, ladders));

  // -- published pair: legend chips and caption, all read from the ladder ----
  // ★ THE SERIES THIS VIEW DRAWS, not every leg in the manifest. A leg with
  // `scope: 'cost'` belongs to the Cost tab (vllm-mtp-energy: the same engine,
  // checkpoint and instrument as vllm-mtp, re-measured with power sampling so
  // that tab has a vLLM curve carrying joules). Filtering HERE is what keeps
  // the pills, the ladder and the table agreeing about how many series exist
  // -- this component owns the pill list and ConcurrencyLadder owns the lines,
  // and they must not disagree. cost.js reads the leg through
  // `baselineSeriesOf`, which is deliberately unfiltered.
  const series = $derived(published ? published.series.filter((s) => s.scope !== 'cost') : []);
  const atlas = $derived(series.find((s) => s.role === 'subject'));
  const baselines = $derived(series.filter((s) => s.role === 'baseline'));
  const baselineRange = $derived(measuredRange(baselines.flatMap((b) => b.rungs)));
  const engines = $derived([...new Set(baselines.map((b) => `${b.engine} (${b.build})`))]);
  // The legend pills toggle a series in and out of the ladder below. The
  // rule (never the last one) is series-visibility.js's; a refused press is
  // printed, not swallowed.
  let hidden = $state([]);
  let refused = $state(null);
  const shown = (id) => !hidden.includes(id);
  const toggle = (id) => {
    const next = toggleSeries(series, hidden, id);
    hidden = next.hidden;
    refused = next.refused;
  };
  // The published ladder is the nearest vLLM number a live-only tab without
  // a manifest has, and the caption must say why it is not drawn here:
  // different ISL/OSL and, if its matched leg recorded one, a different
  // batch cap.
  const ref = publishedLadder;
  const refIsSameCheckpoint = $derived(ref.workload.checkpoint === subject.checkpoint);
  const refCap = batchCapOf(ref.series.find((s) => s.parity === 'matched')?.cli);

  // -- live, Atlas on the gate instrument + comparable one-shots -------------
  // A baseline whose fingerprint differs is refused and NAMED: the page
  // prints the axes, it never draws the line.
  const pair = $derived(live && ladder ? pairWith(live, ladder) : { drawn: [], refused: [] });
  const W = 720, H = 232, PL = 56, PR = 16, PT = 14, PB = 30;
  const pts = $derived(live ? ladderPoints(live) : []);
  const present = $derived(new Set(pts.map((p) => p.c)));
  // "Not run at this rung" is a statement about a record that exists and
  // stops short; with no record every rung is simply empty chrome.
  const absent = $derived(live ? rungs.filter((c) => !present.has(c)) : []);
  const drawnMax = $derived(Math.max(0, ...pair.drawn.flatMap((b) => b.rungs.map((r) => r.tok_s))));
  // The per-rung floors that judged the live record (gate-limits.js): drawn
  // under the Atlas curve, and held by the axis like any other claim.
  const floors = $derived(live ? rungFloors(live) : []);
  const vMax = $derived(
    pts.length ? Math.max(drawnMax, ...pts.map((p) => p.v), ...floors.map((f) => f.value)) * 1.12 : 1
  );
  const l0 = $derived(Math.log2(Math.min(...rungs)));
  const l1 = $derived(Math.log2(Math.max(...rungs)));
  const x = (c) => PL + (l1 === l0 ? 0.5 : (Math.log2(c) - l0) / (l1 - l0)) * (W - PL - PR);
  const y = (v) => PT + (1 - v / vMax) * (H - PT - PB);
  const path = $derived(pts.map((p, i) => `${i ? 'L' : 'M'}${x(p.c).toFixed(1)} ${y(p.v).toFixed(1)}`).join(' '));
  const floorSpans = $derived(rungSpans(floors, x));
  const floorPath = $derived(stepPath(floorSpans, 'min', y));
  const floorLabels = $derived(
    floorSpans.map((sp, i) => ({
      text: i === 0 ? limitLabel('min', sp.min, 'tok/s') : fmtLimit(sp.min),
      x: (sp.x0 + sp.x1) / 2,
      y: Math.min(y(sp.min) + 11, H - PB - 2)
    }))
  );
  const floorAt = (c) => ({ min: floors.find((f) => f.c === c)?.value ?? null, max: null });
  const violAt = (p) => violationOf(p.v, floorAt(p.c));
  const hasViol = $derived(pts.some(violAt));
  const bPath = (b) => b.rungs.map((r, i) => `${i ? 'L' : 'M'}${x(r.c).toFixed(1)} ${y(r.tok_s).toFixed(1)}`).join(' ');
  const yTicks = $derived(pts.length ? [0, vMax / 2, vMax] : []);
  const fmtV = (v) => +v.toFixed(1);
  const fmtB = (v) => v.toFixed(2);
  const color = $derived(colorFor(subject.checkpoint));
  const dash = $derived(dashFor(subject.gate));
  const maxRung = $derived(pts.length ? Math.max(...pts.map((p) => p.c)) : null);
  // Why a rung is absent, from the record: the gate stops where its batch cap
  // stops. Without a recorded cap the reason is the declared rung list itself.
  const absentReason = (c) => {
    const cap = live?.serve_overrides?.max_batch_size;
    const why = cap
      ? `the ${subject.gate} gate stops at C=${maxRung} by design: max_batch_size = "${cap}"`
      : `the ${subject.gate} gate declares concurrencies = "${live.params?.concurrencies ?? 'not recorded'}"`;
    return `C=${c} · not run at this rung — ${why}`;
  };
  const title = $derived(
    state === 'published'
      ? `Atlas vs vLLM · published campaign · ISL ${published.workload.isl_tokens} / OSL ${published.workload.osl_tokens}`
      : state === 'live'
        ? `Atlas vs vLLM · gate instrument · ${instrumentLabel(live)}`
        : 'Atlas vs vLLM · not yet measured'
  );
</script>

{#if state === 'published'}
  <!-- The same chrome as the gate sweep below (figure.gate-panel), so the
       two cards share one width and one inset. -->
  <figure class="gate-panel cmp">
    <figcaption class="gate-panel-head">
      <span class="gate-panel-title">{title}</span>
      <span class="gate-panel-unit">tok/s</span>
    </figcaption>
    <!-- Generated from the series, never typed: a flat vLLM line must read as
         a dated snapshot, and the dates are the proof. Each pill is a real
         button that shows or hides its series in the chart and the table. -->
    <div class="cmp-keys" role="group" aria-label="Series drawn">
      <button type="button" class="cmp-chip" style="border-color:{color}"
        aria-pressed={shown(atlas.id)} onclick={() => toggle(atlas.id)}
        >Atlas · published campaign · {measuredRange(atlas.rungs)}</button>
      {#each baselines as b}
        <button type="button" class="cmp-chip" aria-pressed={shown(b.id)} onclick={() => toggle(b.id)}
          >{oneShotChip(b)}</button>
      {/each}
    </div>
    {#if refused}
      <p class="cmp-refused" role="status">{refused}</p>
    {/if}
    <ConcurrencyLadder embedded ladder={published} {hidden} />
    <p class="cmp-caption">
      vLLM was measured <strong>once</strong>, on {baselineRange}, with
      {#each engines as e, i}{i ? '; ' : ''}<code>{e}</code>{/each} on {published.box.name}, and is
      not re-measured when Atlas moves. Atlas on this chart is the published campaign run of
      {measuredRange(atlas.rungs)} at <code>{atlas.build}</code>; the live series is in
      "Latest gate sweep" below.
    </p>
  </figure>
{:else if state === 'baseline'}
  <ConcurrencyBaseline {subject} ladder={baselineOnlyFor(subject, ladders)} {rungs} />
{:else}
  <figure class="gate-panel cmp">
    <figcaption class="gate-panel-head">
      <span class="gate-panel-title">{title}</span>
      <span class="gate-panel-unit">tok/s</span>
      {#if state === 'live'}
        <span class="gate-legend">
          <span class="gl-group">
            <span class="gate-legend-item">
              <svg class="gl-swatch" viewBox="0 0 20 10" aria-hidden="true">
                <line x1="1" y1="5" x2="19" y2="5" stroke={color} stroke-width="2" stroke-dasharray={dash} />
                <circle cx="10" cy="5" r="3" fill={color} />
              </svg>Atlas · live · latest gate {fmtDate(live.recorded_at)} · {live.git_sha}
            </span>
          </span>
          <span class="gl-sep" aria-hidden="true"></span>
          <span class="gl-group">
            {#each pair.drawn as b}
              <span class="gate-legend-item">
                <svg class="gl-swatch" viewBox="0 0 20 10" aria-hidden="true">
                  <line x1="1" y1="5" x2="19" y2="5" stroke="var(--t2)" stroke-width="1.5" />
                  <rect x="6.5" y="1.5" width="7" height="7" fill="var(--t2)" />
                </svg>{oneShotChip(b)}
              </span>
            {/each}
            <!-- A HOLLOW square: the one-shot baseline's glyph is a filled
                 square, so an empty one is the same key with nothing in it. -->
            {#each pair.refused as r}
              <span class="gate-legend-item">
                <svg class="gl-swatch" viewBox="0 0 20 10" aria-hidden="true">
                  <rect x="6.5" y="1.5" width="7" height="7" fill="none" stroke="var(--t2)" stroke-width="1.5" />
                </svg>{r.series.label} · other instrument · not drawn
              </span>
            {/each}
            {#if pair.drawn.length === 0 && pair.refused.length === 0}
              <span class="gate-legend-item">
                <svg class="gl-swatch" viewBox="0 0 20 10" aria-hidden="true">
                  <rect x="6.5" y="1.5" width="7" height="7" fill="none" stroke="var(--t2)" stroke-width="1.5" />
                </svg>vLLM · not measured on this instrument
              </span>
            {/if}
          </span>
          {#if floorPath || hasViol}
            <span class="gl-sep" aria-hidden="true"></span>
            <span class="gl-group">
              {#if floorPath}
                <span class="gate-legend-item">
                  <svg class="gl-swatch" viewBox="0 0 20 10" aria-hidden="true">
                    <path class="gc-limit" d="M1 7 H8 V3 H19" fill="none" stroke="currentColor" />
                  </svg>gate floor per rung
                </span>
              {/if}
              {#if hasViol}
                <span class="gate-legend-item">
                  <svg class="gl-swatch" viewBox="0 0 12 10" aria-hidden="true"
                    ><circle class="gc-viol" cx="6" cy="5" r="4.2" /></svg>
                  below its floor
                </span>
              {/if}
            </span>
          {/if}
        </span>
      {/if}
    </figcaption>

    <svg viewBox="0 0 {W} {H}" role="img"
      aria-label={state === 'live'
        ? `Atlas throughput versus concurrency on the ${subject.gate} instrument; ${pair.drawn.length ? 'vLLM one-shot on the same instrument' : 'no vLLM series'}`
        : `Empty comparison axes for ${subject.checkpoint}; no run yet`}>
      {#each yTicks as t}
        <line class="gc-grid" x1={PL} y1={y(t)} x2={W - PR} y2={y(t)} />
        <text class="gc-axis" x={PL - 8} y={y(t) + 3.5} text-anchor="end">{fmtV(t)}</text>
      {/each}
      {#if yTicks.length === 0}
        <line class="gc-grid" x1={PL} y1={H - PB} x2={W - PR} y2={H - PB} />
      {/if}
      {#each rungs as c}
        <text class="gc-axis" x={x(c)} y={H - 8} text-anchor="middle">C={c}</text>
      {/each}
      {#each absent as c}
        <g class="cmp-absent">
          <title>{absentReason(c)}</title>
          <line class="gc-grid gc-grid-clipped" x1={x(c)} y1={PT} x2={x(c)} y2={H - PB} />
          <text class="gc-ref-label" x={x(c)} y={PT + 10} text-anchor="middle">not run</text>
        </g>
      {/each}
      {#each pair.drawn as b}
        <path d={bPath(b)} fill="none" stroke="var(--t2)" stroke-width="1.5" stroke-linejoin="round" stroke-linecap="square" />
        {#each b.rungs as r}
          <rect class="cmp-sq" x={x(r.c) - 3.5} y={y(r.tok_s) - 3.5} width="7" height="7" fill="var(--t2)">
            <title>{b.label} · C={r.c} · {fmtB(r.tok_s)} tok/s · mean of {r.reps} reps · spread {r.spread_pct}% · {r.source}</title>
          </rect>
        {/each}
      {/each}
      {#if floorPath}
        <path class="gc-limit" d={floorPath} fill="none" stroke={color} />
        {#each floorLabels as t}
          <text class="gc-limit-label" x={t.x} y={t.y} text-anchor="middle" fill={color}>{t.text}</text>
        {/each}
      {/if}
      {#if pts.length}
        <path d={path} fill="none" stroke={color} stroke-width="2" stroke-dasharray={dash}
          stroke-linejoin="round" stroke-linecap="round" />
        {#each pts as p}
          {@const broke = violAt(p) ? ` · below floor ${fmtLimit(floorAt(p.c).min)}` : ''}
          <g
            class="gc-pt"
            role="button"
            tabindex="0"
            aria-label="C={p.c}: {fmtV(p.v)} tok/s{broke}, gate record {fmtDate(live.recorded_at)} — details"
            onclick={() => onselect([live])}
            onkeydown={(e) => (e.key === 'Enter' || e.key === ' ') && (e.preventDefault(), onselect([live]))}
          >
            <title>Atlas · C={p.c} · {fmtV(p.v)} tok/s{broke} · gate record {fmtDate(live.recorded_at)} · {live.git_sha} · click for record</title>
            <circle class="gc-hit" cx={x(p.c)} cy={y(p.v)} r="11" />
            {#if broke}
              <circle class="gc-viol" cx={x(p.c)} cy={y(p.v)} r="7.5" data-limit="floor" />
            {/if}
            <circle class="gc-mark" cx={x(p.c)} cy={y(p.v)} r="3.5" fill={color} stroke="var(--card)" stroke-width="1" />
          </g>
        {/each}
      {/if}
    </svg>

    {#if state === 'live' && pair.drawn.length}
      <p class="cmp-caption">
        vLLM was measured <strong>once</strong>, on {measuredRange(pair.drawn.flatMap((b) => b.rungs))}, with
        {#each [...new Set(pair.drawn.map((b) => `${b.engine} (${b.build})`))] as e, i}{i ? '; ' : ''}<code>{e}</code>{/each}
        on {ladder.box.name}, on this instrument ({instrumentLabel(live)}), and is not re-measured
        when Atlas moves. Atlas is the newest passing run on main ({fmtDate(live.recorded_at)} ·
        <code>{live.git_sha}</code>).
        {#each pair.refused as r}
          The {r.series.label} one-shot of {measuredRange(r.series.rungs)} is on another instrument
          and is not drawn: {r.why}.
        {/each}
      </p>
    {:else if state === 'live'}
      <p class="cmp-caption">
        <strong>vLLM has not been run on this instrument</strong> ({instrumentLabel(live)}).
        {#each pair.refused as r}
          The {r.series.label} one-shot of {measuredRange(r.series.rungs)} is on another
          instrument and is not comparable: {r.why}.
        {/each}
        {#if pair.refused.length === 0 && refIsSameCheckpoint}
          The published ladder's vLLM legs are ISL {ref.workload.isl_tokens} /
          OSL {ref.workload.osl_tokens}{refCap ? ` at batch cap ${refCap}` : ''}
          and are not comparable in either direction.
        {/if}
        One run of the reference harness at these settings, filed under
        <code>{subject.baselines_dir}/</code>, fills the comparison.
      </p>
    {:else}
      <!-- What is missing, why, what fills it — in that order, and never a zero. -->
      <div class="cmp-empty">
        <p><strong>No concurrency run on main yet for <code>{subject.checkpoint}</code>.</strong></p>
        <p>
          The gate <code>{subject.gate}</code> is declared for this checkpoint with
          <code>status = "unmeasured"</code> and no metrics table — thresholds follow the first
          run on main, by design.
        </p>
        <p>What fills this chart:</p>
        <ul>
          <li>one passing <code>{subject.gate}</code> run for this checkpoint merged to main → the Atlas series</li>
          <li>one vLLM run on the gate's instrument, filed under <code>{subject.baselines_dir}/</code> → the comparison</li>
        </ul>
        {#if /fp8/i.test(subject.checkpoint)}
          <p>
            Note: the chart prints the checkpoint id, which is what a record carries — an FP8
            checkpoint here, whatever a <code>quant</code> line elsewhere says.
          </p>
        {/if}
        {#if subject.published_manifest !== null}
          <p>
            Note: <code>{subject.published_manifest}</code> is declared for this subject but the
            site has no generated ladder for it; run <code>node site/scripts/gen-ladder.mjs</code>.
          </p>
        {/if}
      </div>
    {/if}
  </figure>
{/if}
