<!-- SPDX-License-Identifier: AGPL-3.0-only -->
<script>
  // Region B: the roster — the selection surface and the comparative fleet
  // view in one column. Row order, hotkeys, aria-selected bookkeeping and
  // what happens when the selected machine vanishes are all selection.js
  // rules; this file draws rows and forwards keys.
  //
  // Arrow keys rove (clamped, never wrapped — one extra ↓ that teleports to
  // the top is how an operator stops the wrong launch) and 1–8 jump. Both are
  // handled on the list, so they work whenever focus is anywhere in it, and
  // both move focus WITH selection so the keyboard never loses its place.

  import { move, rosterVm, selectByKey } from '$lib/agent/selection.js';
  import * as S from '$lib/agent/stats.js';
  import FleetAggregate from './FleetAggregate.svelte';
  import RosterRow from './RosterRow.svelte';

  let { fleet, selectedId, poller = null, paused = false, onselect, onadd, onpair } = $props();

  const nodes = $derived(fleet.nodes);
  const rows = $derived(rosterVm(nodes, selectedId));

  // The micro-columns and the fleet Σ read the same poller entries; a dash
  // means "unknown", which must never render as a number.
  const readings = $derived(poller?.byNode ?? {});
  const entries = $derived(
    nodes
      .filter((n) => n.running)
      .map((n) => ({
        id: n.id,
        name: n.name,
        at: readings[n.id]?.at ?? null,
        reading: readings[n.id]?.reading ?? null
      }))
  );
  const decodeOf = (id) => {
    const v = readings[id]?.reading?.decode_tokens_per_s;
    return Number.isFinite(v) ? S.tokens(v) : null;
  };
  const requestsOf = (id) => {
    const v = readings[id]?.reading?.requests_active;
    return Number.isFinite(v) ? S.count(v) : null;
  };

  let listEl = $state(null);

  function focusRow(id) {
    listEl?.querySelector(`[data-node="${CSS.escape(id)}"]`)?.focus();
  }

  function onKeys(ev) {
    if (ev.target instanceof HTMLInputElement) return;
    let next = null;
    if (ev.key === 'ArrowDown') next = move(nodes, selectedId, 1);
    else if (ev.key === 'ArrowUp') next = move(nodes, selectedId, -1);
    else next = selectByKey(nodes, ev.key);
    if (next === null) return;
    ev.preventDefault();
    onselect?.(next);
    focusRow(next);
  }
</script>

<!-- Not a <nav>: app.css fixes bare nav to the viewport for the site header,
     which would tear this column out of the grid. A labelled region is the
     honest role anyway — this is a selection surface, not site navigation. -->
<div class="roster" id="fleet" role="region" aria-label="Fleet roster">
  <FleetAggregate {entries} {paused} />

  <div class="roster-hdr" aria-hidden="true">
    <span class="roster-hdr-node">node</span>
    <span class="roster-hdr-col">tok/s</span>
    <span class="roster-hdr-col">rq</span>
  </div>

  <!-- svelte-ignore a11y_no_noninteractive_element_interactions -->
  <ul class="roster-rows" onkeydown={onKeys} bind:this={listEl}>
    {#each rows as vm (vm.id)}
      <RosterRow
        node={nodes.find((n) => n.id === vm.id)}
        {vm}
        {nodes}
        {onselect}
        {onpair}
        decode={decodeOf(vm.id)}
        requests={requestsOf(vm.id)}
      />
    {:else}
      <li class="roster-empty">No machines yet.</li>
    {/each}
    {#if fleet.peers.length === 0 && rows.length > 0}
      <li class="roster-solo">
        Solo fleet. A second machine gives this column something to compare —
        and unlocks the EP=2 recipes, which need exactly two nodes.
      </li>
    {/if}
  </ul>

  <div class="roster-foot">
    <button type="button" class="btn btn-secondary roster-add" onclick={() => onadd?.()}>
      Add machine
    </button>
  </div>
</div>
