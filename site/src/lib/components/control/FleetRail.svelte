<!-- SPDX-License-Identifier: AGPL-3.0-only -->
<script>
  // Region D: the fleet rail — the live alert feed, then the fabric.
  //
  // D1 is every alert on every node, worst first, exactly as the old alerts
  // section listed them — except each row is now a button that selects its
  // machine, because a feed you cannot act from is a log. No aria-live here:
  // the command strip owns the page's one live region, and a feed that
  // narrated every repaint would bury the severity change it exists to
  // surface.
  //
  // D2 hosts the existing ReachMap and TopologyMap unmodified — two graphs,
  // because "how do I reach dgx3" and "can these machines talk to each other"
  // are different questions — plus the head-picking and pair/unpair actions
  // that lived beside them.

  import { linkWarns, preferredAddress } from '$lib/agent/fleet.svelte.js';
  import ComingSoon from './ComingSoon.svelte';
  import ReachMap from './ReachMap.svelte';
  import TopologyMap from './TopologyMap.svelte';

  let {
    fleet,
    head,
    /** The page's cluster flow — this rail only ever summarizes it. */
    clusterFlow = null,
    oncluster,
    onmakehead,
    onselect,
    onpair,
    onunpair
  } = $props();

  // D3 is a summary, never the ceremony: the epoch-pinned Prepare→Commit
  // flow with its always-visible Abort lives in the overlay, because it must
  // never scroll inside a 180px sub-panel.
  const clusterState = $derived.by(() => {
    const f = clusterFlow;
    if (!f) return { line: 'No cluster.', members: [] };
    if (f.started?.length > 0) {
      return {
        line: `running · ${f.recipe}`,
        members: f.started.map((r) => ({ name: r.name, rank: r.rank, ok: true }))
      };
    }
    if (f.epoch != null && f.phase === 'prepared') {
      return {
        line: `prepared · epoch ${f.epoch}`,
        members: f.answers.map((r) => ({ name: r.name, rank: r.rank, ok: r.prepared === true }))
      };
    }
    return { line: 'No cluster.', members: [] };
  });

  const nodes = $derived(fleet.nodes);
  // The amber the placement machinery raises when a cluster would fall back
  // to ethernet: several times slower while every correctness check passes.
  const fallback = $derived(
    nodes.some((n) => {
      const a = preferredAddress(n);
      return a ? linkWarns(a.class) : false;
    })
  );
</script>

<!-- Same rule as the stage: a keyboard user must be able to scroll the
     rail. Region-with-label plus tabindex is the WCAG technique for it. -->
<!-- svelte-ignore a11y_no_noninteractive_tabindex -->
<aside class="rail" aria-label="Fleet rail" tabindex="0">
  <section class="rail-sec rail-alerts" id="alerts" aria-label="Alerts">
    <h3 class="rail-h">Alerts</h3>
    {#if fleet.alerts.length === 0}
      <p class="rail-quiet">
        Nothing to report. This lane stays here so you never wonder where
        alerts would appear.
      </p>
    {:else}
      <!-- An internal scroll region, so keyboard-reachable like every other. -->
      <!-- svelte-ignore a11y_no_noninteractive_tabindex -->
      <ul class="rail-al-list" id="alert-lane" tabindex="0" aria-label="Alert feed">
        {#each fleet.alerts as a (a.node + a.kind)}
          <li class="rail-al-item">
            <button
              type="button"
              class="rail-al-row"
              onclick={() => onselect?.(a.node)}
              aria-label={`${a.severity} on ${a.nodeName}: ${
                a.detail || a.kind.replaceAll('_', ' ')
              }. Select that machine.`}
            >
              <span class="al-sev al-{a.severity}" aria-hidden="true">{a.severity}</span>
              <span class="rail-al-body" aria-hidden="true">
                <span class="rail-al-node">{a.nodeName}</span>
                <span class="rail-al-kind">{a.kind.replaceAll('_', ' ')}</span>
                {#if a.detail}<span class="rail-al-detail">{a.detail}</span>{/if}
              </span>
            </button>
            <span class="rail-al-acts">
              <ComingSoon id="alert-ack" kind="chip" />
              <ComingSoon id="alert-silence" kind="chip" />
            </span>
          </li>
        {/each}
      </ul>
    {/if}
    <p class="rail-al-foot">
      <ComingSoon id="alert-routing" kind="chip" />
    </p>
  </section>

  <section class="rail-sec" id="topology" aria-label="Fabric">
    <h3 class="rail-h">
      Fabric
      {#if fallback}<span class="rail-warn">Ethernet fallback</span>{/if}
    </h3>
    {#if nodes.length > 0}
      <ReachMap {nodes} />
    {/if}
    <TopologyMap {nodes} {head} />

    <div class="rail-nodes">
      {#each nodes as node (node.id)}
        <div class="topo-act-group">
          <p class="topo-act-name">
            {node.name}
            <span class="mono topo-act-fp">{node.id.slice(0, 8)}</span>
          </p>
          {#if node.isLocal || node.pairing === 'paired'}
            {#if node.canLaunch}
              <button
                type="button"
                class="topo-act-btn"
                disabled={head === node.id}
                onclick={() => onmakehead?.(node.id)}
              >
                {head === node.id ? 'Head (rank 0)' : 'Make head'}
              </button>
            {:else}
              <span class="topo-act-note">Control only — cannot hold a rank</span>
            {/if}
            {#if !node.isLocal}
              <button
                type="button"
                class="topo-act-btn topo-act-danger"
                onclick={() => onunpair?.(node)}>Unpair…</button
              >
            {/if}
          {:else}
            <button type="button" class="topo-act-btn" onclick={() => onpair?.(node)}>
              Pair…
            </button>
          {/if}
        </div>
      {:else}
        <p class="topo-act-empty">No machines yet.</p>
      {/each}
    </div>
  </section>

  <section class="rail-sec rail-cluster" aria-label="Cluster">
    <h3 class="rail-h">Cluster</h3>
    <p class="rail-cl-state">{clusterState.line}</p>
    {#if clusterState.members.length > 0}
      <ul class="rail-cl-members">
        {#each clusterState.members as m (m.name + m.rank)}
          <li class="rail-cl-member" class:rail-cl-refused={!m.ok}>
            <span class="mono rail-cl-rank">r{m.rank}</span>
            {m.name}{#if !m.ok}<span class="rail-cl-no">refused</span>{/if}
          </li>
        {/each}
      </ul>
    {:else}
      <p class="rail-quiet">
        Recipes that span two machines launch from the overlay — every rank
        previews its own exact command before anything reserves.
      </p>
    {/if}
    {#if clusterFlow?.linkWarning}
      <p class="rail-cl-warn">{clusterFlow.linkWarning}</p>
    {/if}
    <button type="button" class="btn btn-primary rail-cl-btn" onclick={() => oncluster?.()}>
      Cluster launch…
    </button>
  </section>
</aside>
