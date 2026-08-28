<!-- SPDX-License-Identifier: AGPL-3.0-only -->
<script>
  // Region B1: the fleet Σ, honestly captioned.
  //
  // The sum comes from aggregate.js, which also builds the caption — the two
  // are computed side by side so "Σ of latest per-node readings · windows
  // differ" can never drift away from the number it disclaims. Stale nodes
  // are excluded from the Σ and NAMED in the caption, because "3 excluded"
  // sends an operator hunting and a name sends them to the machine.
  //
  // The sparkline advances one slot per second on the reactive clock — its
  // x-axis is time, not "whenever a poll happened to land" — and it records a
  // gap when there is no fresh Σ, so a paused page or a wedged fleet shows as
  // the line stopping rather than flatlining at its old value.

  import { untrack } from 'svelte';
  import { aggregate } from '$lib/agent/aggregate.js';
  import { nowMs, useClock } from '$lib/agent/clock.svelte.js';
  import * as S from '$lib/agent/stats.js';

  let {
    /** `{id, name, at, reading}[]` — one per node with a running launch. */
    entries = [],
    /** The operator paused polling: the line must stop short of now. */
    paused = false
  } = $props();

  // Staleness exclusion is the passage of time, not an event.
  $effect(() => useClock());
  const agg = $derived(aggregate(entries, nowMs()));

  let hist = $state([]);
  $effect(() => {
    // One slot per clock tick, and ONLY per clock tick: everything but the
    // clock is read untracked, so polls landing between ticks cannot add
    // slots and the write to `hist` cannot retrigger the effect.
    nowMs();
    untrack(() => {
      hist = S.push(hist, paused ? null : agg.decode);
    });
  });
  const path = $derived(S.sparkline(S.timeline(hist, { held: paused }), 220, 22));
</script>

<div class="fa" aria-label="Fleet aggregate">
  <div class="fa-nums">
    <span class="fa-stat">
      <span class="fa-label">Σ decode</span>
      <span class="fa-val mono">{S.tokens(agg.decode)}<span class="fa-unit"> tok/s</span></span>
    </span>
    <span class="fa-stat">
      <span class="fa-label">Σ rq active</span>
      <span class="fa-val mono">{S.count(agg.active)}</span>
    </span>
    {#if path}
      <!-- Decorative: the Σ it encodes is printed beside it. -->
      <svg class="fa-spark" viewBox="0 0 220 22" preserveAspectRatio="none" aria-hidden="true">
        <path d={path} fill="none" stroke="currentColor" stroke-width="1.2"
              stroke-linejoin="round" stroke-linecap="round" />
      </svg>
    {/if}
  </div>
  <p class="fa-caption">{agg.caption}</p>
</div>
