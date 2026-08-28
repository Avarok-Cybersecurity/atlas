<!-- SPDX-License-Identifier: AGPL-3.0-only -->
<script>
  // Region C2: eight vitals in a fixed 2×4 grid — GPU util, SM clock,
  // temperature, power, unified memory, disk free, docker health, agent
  // uptime. Every tile is a VitalTile, so the five-state grammar (reading /
  // unsupported / pending / alerting / paused) is one implementation.
  //
  // The grid is fixed geometry: telemetry arriving, going stale or being
  // paused never reflows it. The SM clock's `clamped` badge appears only when
  // the agent raises `sm_clock_clamped` — this page invents no thresholds,
  // because a clamped clock costing whole benchmark campaigns was diagnosed
  // by the agent, not by a UI heuristic.
  //
  // The vitals toggle is DISPLAY ONLY for now: the agent discards
  // `WatchFleet{vitals}` (session.rs:219), so samples keep arriving and this
  // component holds the last snapshot instead of rendering the new ones.
  // The command strip says so next to the toggle.

  import VitalTile from './VitalTile.svelte';
  import { isStale } from '$lib/agent/fleet.svelte.js';
  import { nowMs, useClock } from '$lib/agent/clock.svelte.js';
  import * as S from '$lib/agent/stats.js';

  let { node, paused = false } = $props();

  $effect(() => useClock());

  const trusted = $derived(
    node.isLocal || node.pairing === 'paired' || node.pairing === 'vouched' || node.pairing === 'unreachable'
  );
  const offline = $derived(node.pairing === 'unreachable');
  const stale = $derived(trusted && isStale(node, nowMs()));
  const staleFor = $derived(Math.max(0, Math.round((nowMs() - node.lastSeen) / 1000)));

  // Paused holds the snapshot that was current when the operator flipped the
  // toggle. New samples keep arriving (see the header comment); they are
  // simply not shown until the toggle flips back.
  let held = $state(null);
  $effect(() => {
    if (!paused) {
      held = null;
      return;
    }
    // Re-snapshot on a node switch too: while paused, the newly selected
    // machine must hold from the moment it appears, not keep updating live.
    if (held === null || held.forNode !== node.id) {
      held = { forNode: node.id, vitals: node.vitals };
    }
  });
  const v = $derived(paused && held?.forNode === node.id ? held.vitals : node.vitals);

  /**
   * The severity the AGENT raised for a kind, or null if it raised none.
   *
   * Not a severity this file chose. Each of these tiles used to hardcode one —
   * `sm_clock_clamped` was always drawn critical, the rest always warning —
   * which contradicts this page's own rule that it invents no thresholds. An
   * agent that raises a clamp at `warning` must not be redrawn as critical
   * here; the machine that measured it is the one entitled to say how bad it is.
   */
  const severityOf = (alerts, kind) => alerts.find((a) => a.kind === kind)?.severity ?? null;


  const clockAlert = $derived(severityOf(node.alerts, 'sm_clock_clamped'));
  const memAlert = $derived(severityOf(node.alerts, 'memory_pressure'));
  const tempAlert = $derived(severityOf(node.alerts, 'thermal_throttle'));
  const diskAlert = $derived(severityOf(node.alerts, 'disk_low'));

  const gb = (bytes) => (bytes / 1e9).toFixed(0);
  // Plain values become the metric shape here so VitalTile stays the one
  // renderer: a boolean health and a counter are still readings.
  const dockerMetric = $derived(v ? { state: 'reading', value: v.docker_ok ? 1 : 0 } : null);
  const uptimeMetric = $derived(
    v && Number.isFinite(v.agent_uptime_s) ? { state: 'reading', value: v.agent_uptime_s } : null
  );
</script>

<div class="vg" aria-label="Vitals">
  {#if !trusted}
    <p class="vg-note">
      Telemetry from an unpaired machine proves nothing, so none is shown.
      Pair it and these tiles fill in.
    </p>
  {:else if offline}
    <p class="vg-note">
      Paired, but not answering right now — last seen {staleFor}s ago. It stays
      in your fleet; switch it on and these tiles come back on their own.
    </p>
  {:else}
    <div class="vg-grid">
      <VitalTile
        label="GPU UTIL"
        metric={v?.accelerator_util}
        unit="%"
        fraction={v?.accelerator_util?.value != null ? v.accelerator_util.value / 100 : null}
        {stale}
        {paused}
      />
      <VitalTile
        label="SM CLOCK"
        metric={v?.sm_clock_mhz}
        unit=" MHz"
        alert={clockAlert}
        badge={clockAlert ? 'clamped' : null}
        {stale}
        {paused}
      />
      <VitalTile label="TEMP" metric={v?.temperature_c} unit="°C" alert={tempAlert} {stale} {paused} />
      <VitalTile label="POWER" metric={v?.power_w} unit=" W" {stale} {paused} />
      <VitalTile
        label="MEMORY"
        metric={v?.memory_used_frac}
        unit="%"
        fraction={v?.memory_used_frac?.value ?? null}
        format={(x) => (x * 100).toFixed(0)}
        alert={memAlert}
        {stale}
        {paused}
      />
      <VitalTile label="DISK FREE" metric={v?.disk_free_bytes} unit=" GB" format={gb} alert={diskAlert} {stale} {paused} />
      <VitalTile
        label="DOCKER"
        metric={dockerMetric}
        format={(x) => (x ? 'ok' : 'down')}
        {stale}
        {paused}
      />
      <VitalTile label="AGENT UP" metric={uptimeMetric} format={S.uptime} {stale} {paused} />
    </div>
  {/if}
</div>
