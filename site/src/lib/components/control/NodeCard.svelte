<!-- SPDX-License-Identifier: AGPL-3.0-only -->
<script>
  // One machine.
  //
  // The hostname is the least trustworthy thing on this card — Sparks ship with
  // colliding names like spark-256a, and the name arrives over an
  // unauthenticated beacon. So the fingerprint is shown beside it, and every
  // action keys on the fingerprint rather than the name.
  //
  // A discovered node shows no vitals at all. Telemetry from something you have
  // not paired is not evidence about your fleet, and rendering it would suggest
  // otherwise.

  import VitalTile from './VitalTile.svelte';
  import { preferredAddress, linkWarns, isStale } from '$lib/agent/fleet.svelte.js';
  import { nowMs, useClock } from '$lib/agent/clock.svelte.js';

  let { node, onpair, onunpair, ondetails } = $props();

  // Staleness is the passage of time, not an event. Without a reactive clock
  // the derived below only re-ran when `node` changed — and a node that has
  // stopped reporting never changes, so the badge never appeared and the
  // counter froze at whatever it read when the last update arrived.
  $effect(() => useClock());

  const addr = $derived(preferredAddress(node));
  const stale = $derived(isStale(node, nowMs()));
  const staleFor = $derived(Math.max(0, Math.round((nowMs() - node.lastSeen) / 1000)));
  // 'unreachable' is a PAIRED node that is not answering, so it keeps its
  // identity, its address and its unpair action. Only a genuinely unpaired
  // node gets the "pair me" treatment.
  const trusted = $derived(
    node.isLocal || node.pairing === 'paired' || node.pairing === 'unreachable'
  );
  const offline = $derived(node.pairing === 'unreachable');
  const v = $derived(node.vitals);

  const short = $derived(
    node.id ? `${node.id.slice(0, 4)}·${node.id.slice(4, 8)}` : '????·????'
  );

  const trustLabel = $derived(
    node.isLocal
      ? 'this machine'
      : node.pairing === 'paired'
        ? 'paired'
        : node.pairing === 'pairing'
          ? 'pairing…'
          : node.pairing === 'unreachable'
            ? 'unreachable'
            : 'discovered'
  );

  // A clamped SM clock has cost whole benchmark campaigns: every throughput
  // number reads 2.5-2.9x low while every correctness gate still passes. It is
  // the agent that decides "clamped", never this page.
  const clockAlert = $derived(
    node.alerts.find((a) => a.kind === 'sm_clock_clamped') ? 'critical' : null
  );
  const memAlert = $derived(node.alerts.find((a) => a.kind === 'memory_pressure') ? 'warning' : null);
  const tempAlert = $derived(
    node.alerts.find((a) => a.kind === 'thermal_throttle') ? 'warning' : null
  );
  const diskAlert = $derived(node.alerts.find((a) => a.kind === 'disk_low') ? 'warning' : null);

  const gb = (bytes) => (bytes / 1e9).toFixed(0);
</script>

<article
  class="fl-card"
  class:fl-card-discovered={!trusted}
  class:fl-card-stale={stale && trusted}
>
  <header class="fl-head">
    <div class="fl-head-name">
      <h2 title={node.name}>{node.name}</h2>
      <button
        type="button"
        class="fp-chip mono"
        onclick={() => ondetails?.(node)}
        title="Full fingerprint and interfaces"
        aria-label={`Identity of ${node.name}: fingerprint ${short}. Show details.`}
      >
        <svg viewBox="0 0 12 14" width="9" height="11" aria-hidden="true">
          <path
            d="M6 .8 11.2 3.9v6.2L6 13.2.8 10.1V3.9Z"
            fill="none"
            stroke="currentColor"
            stroke-width="1.4"
            stroke-linejoin="round"
          />
        </svg>
        {short}
      </button>
    </div>
    <span class="trust-chip trust-{node.pairing}" class:trust-this={node.isLocal}>
      <span class="dot" aria-hidden="true"></span>{trustLabel}
    </span>
  </header>

  <p class="fl-sub">
    {#if node.isLocal}this machine{:else}peer{/if}
    {#if node.os}· {node.os}{/if}
    {#if node.accelerator}· {node.accelerator}{/if}
    {#if node.agentVersion}· agent {node.agentVersion}{/if}
    {#if !node.canLaunch}· <span class="fl-controlonly">control only</span>{/if}
  </p>

  {#if !trusted}
    <p class="fl-untrusted">
      Seen on the network. Nothing is trusted until you pair it — vitals stay
      hidden because telemetry from an unpaired machine proves nothing.
    </p>
    <div class="fl-actions">
      <button type="button" class="btn btn-secondary fl-btn-sm" onclick={() => onpair?.(node)}>
        Pair this node…
      </button>
    </div>
  {:else}
    <div class="fl-ident">
      {#if addr}
        <span class="mono fl-addr">{addr.addr}</span>
        <span
          class="chip chip-link"
          class:chip-link-warn={linkWarns(addr.class)}
          class:chip-link-unknown={addr.class === 'unverified'}
          title={addr.class === 'unverified'
            ? 'This machine told us where it is, but that message is not authenticated — so its link is not taken on trust until we reach it over the paired channel.'
            : undefined}
        >
          {addr.class === 'roce'
            ? 'RoCE'
            : addr.class === 'infini_band'
              ? 'InfiniBand'
              : addr.class === 'wireless'
                ? 'Wi-Fi'
                : addr.class === 'unverified'
                  ? 'link unverified'
                  : 'Ethernet'}
          {#if addr.speedMbps}· {Math.round(addr.speedMbps / 1000)}G{/if}
        </span>
      {:else}
        <span class="fl-addr fl-addr-none">no usable network link</span>
      {/if}
    </div>

    {#if trusted && !node.isLocal && !v}
      <p class="fl-pendingvitals">
        Vitals arrive over the paired channel. Nothing is shown here rather than
        a placeholder that would never fill.
      </p>
    {:else if offline}
      <p class="fl-offline">
        Paired, but not answering right now. It stays in your fleet — switch it
        on and it comes back on its own.
      </p>
    {:else if !node.canLaunch}
      <p class="fl-untrusted">{node.cannotLaunchReason || 'This node cannot run models.'}</p>
    {:else}
      <div class="vt-row">
        <VitalTile
          label="ACCEL"
          metric={v?.accelerator_util}
          unit="%"
          fraction={v?.accelerator_util?.value != null ? v.accelerator_util.value / 100 : null}
          {stale}
        />
        <VitalTile
          label="SM CLOCK"
          metric={v?.sm_clock_mhz}
          unit=" MHz"
          alert={clockAlert}
          {stale}
        />
        <VitalTile
          label="MEMORY"
          metric={v?.memory_used_frac}
          unit="%"
          fraction={v?.memory_used_frac?.value ?? null}
          format={(x) => (x * 100).toFixed(0)}
          alert={memAlert}
          {stale}
        />
        <VitalTile label="TEMP" metric={v?.temperature_c} unit="°C" alert={tempAlert} {stale} />
        <VitalTile label="POWER" metric={v?.power_w} unit=" W" {stale} />
        <VitalTile
          label="DISK"
          metric={v?.disk_free_bytes}
          unit=" GB"
          format={gb}
          alert={diskAlert}
          {stale}
        />
      </div>
    {/if}

    {#if node.alerts.length > 0}
      <ul class="fl-badges">
        {#each node.alerts.slice(0, 3) as a (a.kind)}
          <li class="al-badge al-{a.severity}" title={a.detail}>{a.kind.replaceAll('_', ' ')}</li>
        {/each}
        {#if node.alerts.length > 3}
          <li class="al-badge">+{node.alerts.length - 3}</li>
        {/if}
      </ul>
    {/if}

    <footer class="fl-activity">
      {#if node.running}
        <span class="fl-running"><span class="dot" aria-hidden="true"></span>{node.running}</span>
      {:else}
        <span class="fl-idle">idle</span>
      {/if}
      {#if stale}<span class="fl-stale-note">last seen {staleFor}s ago</span>{/if}
      {#if !node.isLocal}
        <button type="button" class="fl-unpair" onclick={() => onunpair?.(node)}>Unpair…</button>
      {/if}
    </footer>
  {/if}
</article>
