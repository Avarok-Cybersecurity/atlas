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
  // Panels whose final home is a step-7 overlay (cluster launch, fleet scan,
  // the solo join guide) are hosted at the bottom of the dock's scroll for
  // now, so every flow stays reachable while the rows above stay fixed.

  import ActionsBar from './ActionsBar.svelte';
  import ClusterLaunch from './ClusterLaunch.svelte';
  import ConsoleDock from './ConsoleDock.svelte';
  import FleetScan from './FleetScan.svelte';
  import IdentityHeader from './IdentityHeader.svelte';
  import IoStrip from './IoStrip.svelte';
  import JoinGuide from './JoinGuide.svelte';
  import VitalsGrid from './VitalsGrid.svelte';

  let {
    fleet,
    node,
    poller,
    paused = false,
    vitalsOn = true,
    addOpen = false,
    log = [],
    onlog,
    onpair,
    onunpair,
    ondetails
  } = $props();

  const nodes = $derived(fleet.nodes);
  const solo = $derived(fleet.peers.length === 0);
  const remoteCount = $derived(fleet.remoteLaunchable.length);
  const entry = $derived(node ? (poller?.byNode[node.id] ?? null) : null);

  let tab = $state('launch');

  function onverb(verb) {
    // The bar's verbs land in the dock: the tab does the work and shows the
    // reply where there is room to read it.
    if (verb === 'logs') tab = 'logs';
    else if (verb === 'status') tab = 'status';
    else tab = 'launch';
  }
</script>

<section class="stage" aria-label={node ? `Node stage: ${node.name}` : 'Node stage'}>
  {#if node}
    <IdentityHeader {node} {nodes} {onpair} {onunpair} {ondetails} />
    <VitalsGrid {node} paused={!vitalsOn} />
    <IoStrip {node} {entry} {paused} {nodes} />
    <ActionsBar
      {fleet}
      {node}
      {nodes}
      {solo}
      {onverb}
      {onlog}
      onstats={() => poller?.pollNow(node.id)}
    />
    <ConsoleDock {fleet} {node} {nodes} {tab} ontab={(t) => (tab = t)} {log} {onlog}>
      {#snippet extra()}
        {#if fleet.controlOnly && remoteCount === 0}
          <p class="fl-co-why">
            This machine drives the fleet; it does not run models itself. Pair a
            machine that can and everything on this page applies to it.
          </p>
          <JoinGuide {fleet} />
        {/if}

        {#if solo}
          {#if !fleet.controlOnly}
            <p class="fl-solo-note">
              No peers yet. Pairing a second machine also unlocks the EP=2
              recipes, which need exactly two nodes.
            </p>
            <JoinGuide {fleet} />
          {/if}
          <FleetScan {fleet} />
        {:else if addOpen}
          <div class="stage-add">
            <h3 class="stage-h">Add a machine</h3>
            <FleetScan {fleet} />
          </div>
        {/if}

        <!-- Keeps the pre-bridge anchor: deep links and the @live e2e spec
             target #launch, and moving the panel must not break either. -->
        <div class="stage-launch" id="launch">
          <h3 class="stage-h">Cluster launch</h3>
          <p class="stage-sub">
            Two phases, because one cannot fail cleanly: every machine validates
            and reserves, and nothing starts until all of them have agreed.
          </p>
          <ClusterLaunch {fleet} />
        </div>
      {/snippet}
    </ConsoleDock>
  {:else}
    <p class="stage-none">No machine selected. Pick one from the roster.</p>
  {/if}
</section>
