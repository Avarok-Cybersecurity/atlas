<script module>
  // The comparison chart is the first block of every subject tab: Atlas
  // against vLLM on ONE instrument. Which chart that is depends on what the
  // subject has, and the decision is made here — in one place, exported — so
  // the panel around it (header tiles, the bridging note over the gate sweep)
  // cannot disagree with the chart about what is being drawn.
  //
  //   published  the frozen campaign pair from the subject's manifest (dense)
  //   live       the newest passing gate record on main, Atlas only (DFlash)
  //   none       nothing to draw yet (MoE)
  //
  // A fourth state, the live PAIR (gate record + a one-shot vLLM run on the
  // gate's own instrument), needs the baseline files under `baselines_dir` and
  // the instrument fingerprint in ladder-baselines.js; neither exists yet, so
  // it is not decidable here and is not pretended.
  import publishedLadder from '$lib/ladder.generated.json';

  /**
   * The published ladder for a subject, or null. A manifest is only honoured
   * when the generated ladder is for THIS checkpoint — the site has one
   * generated ladder, and drawing it under a different subject's label would
   * put a 27B curve on a 35B tab.
   */
  export function publishedFor(subject) {
    return subject.published_manifest !== null && publishedLadder.workload.checkpoint === subject.checkpoint
      ? publishedLadder
      : null;
  }

  /** Newest passing gate record on main — the one the live series is drawn from. */
  export const liveRecordOf = (records) =>
    records.filter((r) => r.verdict === 'PASS' && !r.branch).at(-1) ?? null;

  export function comparisonStateOf(subject, records) {
    if (publishedFor(subject)) return 'published';
    if (liveRecordOf(records)) return 'live';
    return 'none';
  }

  /**
   * The gate's instrument, read off a record, for chart titles and captions.
   * ISL/OSL are `params` (the harness pins them); batch cap and KV dtype are
   * `serve_overrides`. Every field is printed only if the record carries it —
   * a title must not claim a setting the run did not record.
   */
  export function instrumentLabel(record) {
    const p = record.params ?? {};
    const o = record.serve_overrides ?? {};
    const parts = [];
    if (p.isls !== undefined && p.osl !== undefined) parts.push(`ISL ${p.isls} / OSL ${p.osl}`);
    if (p.prompt_mode) parts.push(`${p.prompt_mode} fixture`);
    if (o.max_batch_size) parts.push(`batch cap ${o.max_batch_size}`);
    if (o.kv_cache_dtype) parts.push(`${o.kv_cache_dtype} KV`);
    return parts.join(' · ');
  }

  /** `2026-08-17 → 2026-08-18`, or one date when a series was measured in a day. */
  export function measuredRange(rungs) {
    const days = rungs.map((r) => r.measured_utc.slice(0, 10)).sort();
    const [from, to] = [days[0], days[days.length - 1]];
    return from === to ? from : `${from} → ${to}`;
  }

  /** The batch cap a baseline's serve command pinned, if its CLI records one. */
  export function batchCapOf(cli) {
    const m = /--(?:max-num-seqs|max-batch-size)\s+(\d+)/.exec(cli ?? '');
    return m ? m[1] : null;
  }
</script>

<script>
  import ConcurrencyLadder from './ConcurrencyLadder.svelte';
  import { colorFor, fmtDate, ladderPoints } from '$lib/gates.js';
  import { dashFor } from '$lib/gate-variants.js';

  let { subject, records, rungs, onselect } = $props();

  const published = $derived(publishedFor(subject));
  const live = $derived(liveRecordOf(records));
  const state = $derived(comparisonStateOf(subject, records));

  // -- published pair: legend chips and caption, all read from the ladder ----
  const series = $derived(published ? published.series : []);
  const atlas = $derived(series.find((s) => s.role === 'subject'));
  const baselines = $derived(series.filter((s) => s.role === 'baseline'));
  const baselineRange = $derived(measuredRange(baselines.flatMap((b) => b.rungs)));
  const engines = $derived([...new Set(baselines.map((b) => `${b.engine} (${b.build})`))]);
  // The published ladder is the nearest vLLM number a live-only tab has, and
  // the caption must say why it is not drawn here: different ISL/OSL and, if
  // its matched leg recorded one, a different batch cap.
  const refIsSameCheckpoint = $derived(publishedLadder.workload.checkpoint === subject.checkpoint);
  const refCap = $derived(batchCapOf(publishedLadder.series.find((s) => s.parity === 'matched')?.cli));

  // -- live, Atlas only: same SVG dialect as GateLadderChart -----------------
  const W = 720, H = 232, PL = 56, PR = 16, PT = 14, PB = 30;
  const pts = $derived(live ? ladderPoints(live) : []);
  const present = $derived(new Set(pts.map((p) => p.c)));
  // "Not run at this rung" is a statement about a record that exists and
  // stops short; with no record every rung is simply empty chrome.
  const absent = $derived(live ? rungs.filter((c) => !present.has(c)) : []);
  const vMax = $derived(pts.length ? Math.max(...pts.map((p) => p.v)) * 1.12 : 1);
  const l0 = $derived(Math.log2(Math.min(...rungs)));
  const l1 = $derived(Math.log2(Math.max(...rungs)));
  const x = (c) => PL + (l1 === l0 ? 0.5 : (Math.log2(c) - l0) / (l1 - l0)) * (W - PL - PR);
  const y = (v) => PT + (1 - v / vMax) * (H - PT - PB);
  const path = $derived(pts.map((p, i) => `${i ? 'L' : 'M'}${x(p.c).toFixed(1)} ${y(p.v).toFixed(1)}`).join(' '));
  const yTicks = $derived(pts.length ? [0, vMax / 2, vMax] : []);
  const fmtV = (v) => +v.toFixed(1);
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
  <div class="cc">
    <div class="gate-panel-head">
      <span class="gate-panel-title">{title}</span>
      <span class="gate-panel-unit">tok/s</span>
    </div>
    <!-- Generated from the series, never typed: a flat vLLM line must read as
         a dated snapshot, and the dates are the proof. -->
    <p class="cc-keys">
      <span class="cc-chip" style="border-color:{color}">Atlas · published campaign · {measuredRange(atlas.rungs)}</span>
      {#each baselines as b}
        <span class="cc-chip">{b.label} · one-shot · measured {measuredRange(b.rungs)} · not re-run</span>
      {/each}
    </p>
    <ConcurrencyLadder embedded ladder={published} />
    <p class="cc-caption">
      vLLM was measured <strong>once</strong>, on {baselineRange}, with
      {#each engines as e, i}{i ? '; ' : ''}<code>{e}</code>{/each} on {published.box.name}, and is
      not re-measured when Atlas moves. Atlas on this chart is the published campaign run of
      {measuredRange(atlas.rungs)} at <code>{atlas.build}</code>; the live series is in
      "Latest gate sweep" below.
    </p>
  </div>
{:else}
  <figure class="gate-panel cc">
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
            <!-- A HOLLOW square: the one-shot baseline's glyph is a filled
                 square, so an empty one is the same key with nothing in it. -->
            <span class="gate-legend-item">
              <svg class="gl-swatch" viewBox="0 0 20 10" aria-hidden="true">
                <rect x="6.5" y="1.5" width="7" height="7" fill="none" stroke="var(--t2)" stroke-width="1.5" />
              </svg>vLLM · not measured on this instrument
            </span>
          </span>
        </span>
      {/if}
    </figcaption>

    <svg viewBox="0 0 {W} {H}" role="img"
      aria-label={state === 'live'
        ? `Atlas throughput versus concurrency on the ${subject.gate} instrument; no vLLM series`
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
        <g class="cc-absent">
          <title>{absentReason(c)}</title>
          <line class="gc-grid gc-grid-clipped" x1={x(c)} y1={PT} x2={x(c)} y2={H - PB} />
          <text class="gc-ref-label" x={x(c)} y={PT + 10} text-anchor="middle">not run</text>
        </g>
      {/each}
      {#if pts.length}
        <path d={path} fill="none" stroke={color} stroke-width="2" stroke-dasharray={dash}
          stroke-linejoin="round" stroke-linecap="round" />
        {#each pts as p}
          <g
            class="gc-pt"
            role="button"
            tabindex="0"
            aria-label="C={p.c}: {fmtV(p.v)} tok/s, gate record {fmtDate(live.recorded_at)} — details"
            onclick={() => onselect([live])}
            onkeydown={(e) => (e.key === 'Enter' || e.key === ' ') && (e.preventDefault(), onselect([live]))}
          >
            <title>Atlas · C={p.c} · {fmtV(p.v)} tok/s · gate record {fmtDate(live.recorded_at)} · {live.git_sha} · click for record</title>
            <circle class="gc-hit" cx={x(p.c)} cy={y(p.v)} r="11" />
            <circle class="gc-mark" cx={x(p.c)} cy={y(p.v)} r="3.5" fill={color} stroke="var(--card)" stroke-width="1" />
          </g>
        {/each}
      {/if}
    </svg>

    {#if state === 'live'}
      <p class="cc-caption">
        <strong>vLLM has not been run on this instrument</strong> ({instrumentLabel(live)}).
        {#if refIsSameCheckpoint}
          The published ladder's vLLM legs are ISL {publishedLadder.workload.isl_tokens} /
          OSL {publishedLadder.workload.osl_tokens}{refCap ? ` at batch cap ${refCap}` : ''}
          and are not comparable in either direction.
        {/if}
        One run of the reference harness at these settings, filed under
        <code>{subject.baselines_dir}/</code>, fills the comparison.
      </p>
    {:else}
      <!-- What is missing, why, what fills it — in that order, and never a zero. -->
      <div class="cc-empty">
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
            site's generated ladder is for <code>{publishedLadder.workload.checkpoint}</code>; the
            manifest has not been generated for this checkpoint.
          </p>
        {/if}
      </div>
    {/if}
  </figure>
{/if}
