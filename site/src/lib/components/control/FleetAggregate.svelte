<!-- SPDX-License-Identifier: AGPL-3.0-only -->
<script>
  // Region B1: the fleet Σ, honestly captioned.
  //
  // The sum comes from aggregate.js, which also builds the caption — the two
  // are computed side by side so "Σ of latest per-node readings · windows
  // differ" can never drift away from the number it disclaims. Until the
  // stats pollers land, `entries` is empty and both numerals read the
  // formatter's em-dash: no reading is not a fleet doing 0 tok/s.

  import { aggregate } from '$lib/agent/aggregate.js';
  import { nowMs, useClock } from '$lib/agent/clock.svelte.js';
  import * as S from '$lib/agent/stats.js';

  let {
    /** `{id, name, at, reading}[]` — one per node with a running launch. */
    entries = []
  } = $props();

  // Staleness exclusion is the passage of time, not an event.
  $effect(() => useClock());
  const agg = $derived(aggregate(entries, nowMs()));
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
  </div>
  <p class="fa-caption">{agg.caption}</p>
</div>
