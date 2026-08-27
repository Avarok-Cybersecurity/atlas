<!-- SPDX-License-Identifier: AGPL-3.0-only -->
<script>
  // Region C: the stage — everything about the selected machine, plus (until
  // the launch overlay lands) the cluster launch flow, which needs more room
  // than the 336px rail can honestly give a Prepare→Commit ceremony.
  //
  // Shell stage: the existing panels are HOSTED here unmodified, so for now
  // the whole stage is one scroll region with a focus ring. The final stage
  // scrolls only its console dock (48+144+120+44 fixed rows; at 1366×768 the
  // dock keeps 720−356 = 364px ≥ its 320px minimum) — that recomposition is
  // the next step, and this container is where those rows land.

  import ClusterLaunch from './ClusterLaunch.svelte';
  import FleetScan from './FleetScan.svelte';
  import JoinGuide from './JoinGuide.svelte';
  import NodeCard from './NodeCard.svelte';

  let { fleet, node, addOpen = false, onpair, onunpair, ondetails } = $props();

  const solo = $derived(fleet.peers.length === 0);
  const remoteCount = $derived(fleet.remoteLaunchable.length);
</script>

<!-- A scroll region must be reachable by keyboard or its overflow is
     mouse-only content; region-with-label plus tabindex is the WCAG technique
     for exactly this, and the spec requires it on every internal scroller. -->
<!-- svelte-ignore a11y_no_noninteractive_tabindex -->
<section
  class="stage"
  aria-label={node ? `Node stage: ${node.name}` : 'Node stage'}
  tabindex="0"
>
  {#if node}
    <NodeCard nodes={fleet.nodes} {node} {onpair} {onunpair} {ondetails} />
  {:else}
    <p class="stage-none">No machine selected. Pick one from the roster.</p>
  {/if}

  {#if fleet.controlOnly}
    <div class="fl-control-only">
      <p class="fl-co-head">
        <span class="fl-co-chip">Control only</span>
        This machine drives the fleet; it does not run models itself.
      </p>
      <p class="fl-co-why">
        {#if remoteCount === 0}
          Pair a machine that can run models and everything on this page —
          telemetry, launching, alerts — applies to it.
        {:else}
          Everything on this page — telemetry, launching, alerts — applies
          to the {remoteCount === 1 ? 'machine' : `${remoteCount} machines`}
          you have paired.
        {/if}
      </p>
      {#if remoteCount === 0}
        <JoinGuide {fleet} />
      {/if}
    </div>
  {/if}

  {#if solo}
    {#if !fleet.controlOnly}
      <p class="fl-solo-note">
        No peers yet. Pairing a second machine also unlocks the EP=2
        recipes, which need exactly two nodes.
      </p>
      <!-- Rendered here rather than in both places so it appears exactly once
           — the control-only panel above already carries it, and two "Show me
           how" buttons on one screen is two live join codes and a question
           about which is real. -->
      <JoinGuide {fleet} />
    {/if}
    <FleetScan {fleet} />
  {:else if addOpen}
    <!-- The roster's "Add machine" opened this. In solo mode FleetScan is
         already on the stage, so this branch exists only for a fleet that
         outgrew it and wants another member. -->
    <div class="stage-add">
      <h3 class="stage-h">Add a machine</h3>
      <FleetScan {fleet} />
    </div>
  {/if}

  <!-- Keeps the pre-bridge anchor: deep links and the @live e2e spec target
       #launch, and moving the panel must not break either. -->
  <div class="stage-launch" id="launch">
    <h3 class="stage-h">Cluster launch</h3>
    <p class="stage-sub">
      Two phases, because one cannot fail cleanly: every machine validates and
      reserves, and nothing starts until all of them have agreed.
    </p>
    <ClusterLaunch {fleet} />
  </div>
</section>
