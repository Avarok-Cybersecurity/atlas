<!-- SPDX-License-Identifier: AGPL-3.0-only -->
<script>
  // One measurement, in one of five states that must be distinguishable at a
  // glance because they mean different things:
  //
  //   value    a reading
  //   n/a      this hardware cannot answer — permanent, and NOT a zero. On a
  //            GB10 the framebuffer fields are genuinely unanswerable because
  //            Grace-Blackwell is unified memory, and drawing 0 GB there would
  //            be reporting a measurement nobody took.
  //   pending  the capability exists, the first sample has not arrived
  //   alerting a reading that has tripped a threshold
  //   paused   the OPERATOR stopped the updates — the last value stays, in
  //            --t3 with a 'paused' sublabel, so an old number is never
  //            dressed as a live one. Distinct from stale, whose tilde blames
  //            the machine for going quiet; here the page went quiet on
  //            purpose, so the tilde is suppressed.
  //
  // Fixed height in every state, so a card never resizes as telemetry arrives.

  let {
    label,
    /** `{ state: 'reading', value } | { state: 'unsupported' } | null` */
    metric = null,
    unit = '',
    /** 0..1, drawn as a meter under the value when present. */
    fraction = null,
    /** Set when this reading has tripped a threshold. */
    alert = null,
    /** Renders the last known value greyed out. */
    stale = false,
    /** The operator paused updates; the last value is shown as held, not live. */
    paused = false,
    /** A word the agent raised about this reading (e.g. 'clamped'). The page
        invents no thresholds: a badge renders only when an alert names it. */
    badge = null,
    format = (v) => v.toFixed(0)
  } = $props();

  const unsupported = $derived(metric?.state === 'unsupported');
  const pending = $derived(metric == null);
  const value = $derived(metric?.state === 'reading' ? metric.value : null);
</script>

<div
  class="vt-tile"
  class:vt-na={unsupported}
  class:vt-stale={stale && !paused}
  class:vt-paused={paused}
  class:vt-alert={Boolean(alert)}
  class:vt-alert-crit={alert === 'critical'}
>
  <span class="vt-label"
    >{label}{#if badge}<span class="vt-badge">{badge}</span>{/if}</span
  >

  {#if unsupported}
    <span class="vt-val vt-dash" aria-hidden="true">—</span>
    <span class="vt-sub">not on this hw</span>
    <span class="visually-hidden">{label}: not available on this hardware</span>
  {:else if pending}
    <span class="vt-skeleton" aria-hidden="true"></span>
    <span class="visually-hidden">{label}: waiting for the first sample</span>
  {:else}
    <span class="vt-val">
      {#if stale && !paused}<span class="vt-tilde" aria-hidden="true">~</span>{/if}{format(
        value
      )}<span class="vt-unit">{unit}</span>
    </span>
    {#if paused}
      <span class="vt-sub">paused</span>
    {:else if fraction !== null}
      <span class="vt-meter" aria-hidden="true">
        <span class="vt-meter-fill" style="width: {Math.max(0, Math.min(1, fraction)) * 100}%"
        ></span>
      </span>
    {:else}
      <span class="vt-sub"></span>
    {/if}
  {/if}
</div>
