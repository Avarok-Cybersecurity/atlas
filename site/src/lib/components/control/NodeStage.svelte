<!-- SPDX-License-Identifier: AGPL-3.0-only -->
<script>
  // Region C: the stage, recomposed to the spec's fixed rows —
  //
  //   C1 identity header   48px
  //   C2 vitals grid      144px
  //   C3 serving I/O      120px
  //   C4 actions bar       44px
  //   C5 console dock      the remainder, the only scroll region
  //
  // 48+144+120+44 = 356px of fixed rows; at 1366×768 the dock keeps
  // 719−356 = 363px ≥ its 320px minimum, so the page never scrolls at
  // desktop widths. All tiles are fixed geometry — telemetry arriving never
  // reflows the stage.
  //
  import ActionsBar from './ActionsBar.svelte';
  import ConsoleDock from './ConsoleDock.svelte';
  import IdentityHeader from './IdentityHeader.svelte';
  import IoStrip from './IoStrip.svelte';
  import VitalsGrid from './VitalsGrid.svelte';

  let {
    fleet,
    node,
    poller,
    paused = false,
    vitalsOn = true,
    // The dock tab is the PAGE's state, not this component's: the keyboard
    // map ('l' Logs, 'n' Launch) has to reach it from outside the stage.
    tab = 'launch',
    ontab,
    log = [],
    onlog,
    onpair,
    onunpair,
    ondetails
  } = $props();

  const nodes = $derived(fleet.nodes);
  const solo = $derived(fleet.peers.length === 0);
  const entry = $derived(node ? (poller?.byNode[node.id] ?? null) : null);

  let bar = $state(null);

  /** The 's' hotkey: same two-step arm/confirm as pressing Stop in the bar. */
  export function armStop() {
    bar?.armStop();
  }

  function onverb(verb) {
    // The bar's verbs land in the dock: the tab does the work and shows the
    // reply where there is room to read it.
    if (verb === 'logs') ontab?.('logs');
    else if (verb === 'status') ontab?.('status');
    else ontab?.('launch');
  }
</script>

<section class="stage" aria-label={node ? `Node stage: ${node.name}` : 'Node stage'}>
  {#if node}
    <IdentityHeader {node} {nodes} {onpair} {onunpair} {ondetails} />
    <VitalsGrid {node} paused={!vitalsOn} />
    <IoStrip {node} {entry} {paused} {nodes} />
    <ActionsBar
      bind:this={bar}
      {fleet}
      {node}
      {nodes}
      {solo}
      {onverb}
      {onlog}
      onstats={() => poller?.pollNow(node.id)}
    />
    <ConsoleDock {fleet} {node} {nodes} {tab} {ontab} {log} {onlog} />
  {:else}
    <p class="stage-none">No machine selected. Pick one from the roster.</p>
  {/if}
</section>
