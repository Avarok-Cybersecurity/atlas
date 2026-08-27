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
  import ReachMap from './ReachMap.svelte';
  import TopologyMap from './TopologyMap.svelte';

  let { fleet, head, onmakehead, onselect, onpair, onunpair } = $props();

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
      <ul class="rail-al-list">
        {#each fleet.alerts as a (a.node + a.kind)}
          <li>
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
          </li>
        {/each}
      </ul>
    {/if}
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
</aside>
