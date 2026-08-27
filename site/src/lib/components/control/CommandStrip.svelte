<!-- SPDX-License-Identifier: AGPL-3.0-only -->
<script>
  // Region A of the bridge: 48px of who-and-how-bad.
  //
  // Left, identity and trust: the Atlas mark (the way back to the site the
  // rest of this surface deliberately is not), the connection chip, and the
  // trust counts — vouched counted apart from paired, because second-hand
  // identity must never wear a verified pin's clothes.
  //
  // Center, the worst live alert verbatim. Clicking it selects the node, so
  // the loudest problem is one action from its machine.
  //
  // Right, the telemetry range chip: what this page can show is this session,
  // and the 1h/24h segments are registered placeholders that say exactly what
  // is missing (an agent ring buffer and a query verb).
  //
  // This strip also hosts THE page's one aria-live region. Everything else on
  // the surface renders silently; only a change in worst severity is worth
  // interrupting a screen-reader user for, and `announce.js` decides that.
  // The timer here only lets the fleet settle before the sentence is read.

  import { ANNOUNCE_DEBOUNCE_MS, announcement } from '$lib/agent/announce.js';
  import { CADENCES } from '$lib/agent/cadence.js';
  import { placeholdersFor } from '$lib/agent/placeholders.js';
  import ComingSoon from './ComingSoon.svelte';

  let { fleet, onselect, cadence, oncadence, vitals = true, onvitals } = $props();

  const cadenceLabel = (c) => (c.ms === null ? 'Pause' : c.id);

  const worst = $derived(fleet.alerts[0] ?? null);
  const more = $derived(Math.max(0, fleet.alerts.length - 1));

  const counts = $derived.by(() => {
    const by = {};
    for (const n of fleet.peers) by[n.pairing] = (by[n.pairing] ?? 0) + 1;
    return by;
  });

  const solo = $derived(fleet.peers.length === 0);
  const rangeSegments = $derived(placeholdersFor('command', { solo }));

  // The one live region. `announced` is written only when announce.js says a
  // severity transition happened, and only after the debounce window — so a
  // storm that escalates twice inside it is read once, at its worst.
  let announced = $state('');
  let prevSeverity = null;
  $effect(() => {
    const a = announcement(prevSeverity, fleet.alerts);
    if (!a) return;
    prevSeverity = a.severity;
    const timer = setTimeout(() => (announced = a.text), ANNOUNCE_DEBOUNCE_MS);
    return () => clearTimeout(timer);
  });
</script>

<header class="cmd" aria-label="Fleet command strip">
  <div class="cmd-left">
    <a class="cmd-mark" href="/" aria-label="Atlas home">
      <img src="/favicon.svg" alt="" width="20" height="20" />
      <span>Atlas</span>
    </a>
    <span class="cmd-chip" class:cmd-chip-amber={fleet.controlOnly}>
      {fleet.controlOnly ? 'control-only' : 'live'}
    </span>
    <span class="cmd-counts">
      {#if solo}
        solo
      {:else}
        {#each ['paired', 'vouched', 'pairing', 'unreachable', 'discovered'] as state (state)}
          {#if counts[state]}
            <span class="cmd-count cmd-count-{state}">{counts[state]} {state}</span>
          {/if}
        {/each}
      {/if}
    </span>
  </div>

  <div class="cmd-center">
    {#if worst}
      <button
        type="button"
        class="cmd-alert al-{worst.severity}"
        onclick={() => onselect?.(worst.node)}
        aria-label={`Worst alert, ${worst.severity} on ${worst.nodeName}: ${
          worst.detail || worst.kind.replaceAll('_', ' ')
        }. Select that machine.`}
      >
        <strong>{worst.nodeName}:</strong>
        <span class="cmd-alert-text">{worst.detail || worst.kind.replaceAll('_', ' ')}</span>
        {#if more > 0}<span class="cmd-alert-more">+{more}</span>{/if}
      </button>
    {:else}
      <span class="cmd-quiet">no alerts</span>
    {/if}
    <span class="visually-hidden" role="status">{announced}</span>
  </div>

  <div class="cmd-right">
    <!-- The poll cadence: what the page asks LaunchStats for, and how often.
         Pause means paused for everything — background 10s polls that kept
         flowing would keep a relay busy on behalf of a page claiming quiet. -->
    <span class="cmd-seg" role="group" aria-label="Poll cadence">
      {#each CADENCES as c (c.id)}
        <button
          type="button"
          class="cmd-seg-btn"
          class:cmd-seg-on={cadence === c.id}
          aria-pressed={cadence === c.id}
          onclick={() => oncadence?.(c.id)}
        >
          {cadenceLabel(c)}
        </button>
      {/each}
    </span>

    <!-- Display only, and it says so: the agent discards WatchFleet{vitals}
         (session.rs:219), so flipping this holds the tiles rather than
         quieting the wire. The caption comes off when the agent honours it. -->
    <span class="cmd-vitals">
      <button
        type="button"
        role="switch"
        aria-checked={vitals}
        class="cmd-vitals-btn"
        class:cmd-vitals-on={vitals}
        onclick={() => onvitals?.(!vitals)}
      >
        Vitals
      </button>
      <span class="cmd-vitals-cap">display only</span>
    </span>

    <span class="cmd-range" aria-label="Telemetry range">
      <span class="cmd-range-live" aria-current="true">this session</span>
      {#each rangeSegments as seg (seg.id)}
        <ComingSoon id={seg.id} kind="chip" />
      {/each}
    </span>
  </div>
</header>
