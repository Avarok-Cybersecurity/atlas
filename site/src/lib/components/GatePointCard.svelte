<script>
  import { modal } from '$lib/modal.js';
  // The per-point metadata card ("mini modal"). Every row is a verbatim field
  // of the gate record — nothing synthesized. Raised by clicking a chart point.
  import { GH_COMMIT, shortModel, colorFor, fmtDateTime, sampleCount } from '$lib/gates.js';

  let { record, onclose } = $props();
  const r = $derived(record);

  function onkeydown(e) {
    if (e.key === 'Escape') { e.stopPropagation(); onclose(); }
  }
</script>

<svelte:window {onkeydown} />

<div class="gpc-backdrop" onclick={onclose} role="presentation">
  <article
    class="gpc receipt"
    role="dialog"
    aria-modal="true"
    aria-label="Gate record {r.git_sha}"
    use:modal
    onclick={(e) => e.stopPropagation()}
  >
    <div class="receipt-body">
      <div class="receipt-head">
        <span class="receipt-title">{r.benchmark_name}</span>
        <span class="gpc-verdict" data-verdict={r.verdict}>{r.verdict === 'PASS' ? '✓ PASS' : '✗ ' + r.verdict}</span>
      </div>

      <div class="gpc-model-row">
        <span class="gpc-swatch" style="background:{colorFor(r.target_model)}" aria-hidden="true"></span>
        <span class="gpc-model">{r.target_model}</span>
      </div>

      <dl class="gpc-rows">
        <dt>commit</dt>
        <dd>
          <a href="{GH_COMMIT}{r.git_sha}" target="_blank" rel="noopener">{r.git_sha}</a>
          {#if r.branch}<span class="gpc-branch"> · source {r.branch}</span>{/if}
          {#if r.generated_ancestry === 'no'}<span class="gpc-branch"> · not in dashboard commit history</span>{/if}
          {#if r.generated_ancestry === 'unknown'}<span class="gpc-branch"> · commit history unavailable</span>{/if}
        </dd>
        <dt>recorded</dt>
        <dd>{fmtDateTime(r.recorded_at)}</dd>
        <dt>served by</dt>
        <dd>{r.served_by}</dd>
        <dt>box</dt>
        <dd>{r.hardware?.gpu} · driver {r.hardware?.driver}</dd>
        <dt>sm clock</dt>
        <dd>{r.hardware?.sm_clock_mhz} MHz <span class="gpc-fine">({r.hardware?.source})</span></dd>
        {#if sampleCount(r) !== null}
          <dt>n</dt>
          <dd>{sampleCount(r)}</dd>
        {/if}
        <dt>atlas</dt>
        <dd>{r.atlas_version}</dd>
      </dl>

      <div class="gpc-section">metrics</div>
      <dl class="gpc-rows">
        {#each Object.entries(r.metrics ?? {}) as [k, v]}
          <dt>{k}</dt>
          <dd>{Math.abs(v) >= 1000 ? Math.round(v).toLocaleString('en-US') : +(+v).toFixed(2)}</dd>
        {/each}
      </dl>

      <div class="gpc-section">verdict · thresholds</div>
      <p class="gpc-reason">{r.verdict_reason}</p>

      <div class="gpc-section">run parameters (the draw)</div>
      <dl class="gpc-rows gpc-params">
        {#each Object.entries(r.params ?? {}) as [k, v]}
          <dt>{k}</dt>
          <dd>{v}</dd>
        {/each}
      </dl>

      <div class="receipt-foot">
        <span>{shortModel(r.target_model)}</span>
        <button type="button" class="gpc-close" onclick={onclose}>close</button>
      </div>
    </div>
  </article>
</div>
