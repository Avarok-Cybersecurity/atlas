<!-- SPDX-License-Identifier: AGPL-3.0-only -->
<script>
  // Region C5: the console dock — the stage's only scroll region.
  //
  // Four tabs: Logs, Launch, Status, and Requests, which is the dock's one
  // permitted placeholder (placeholders.js caps it). The Requests tab is a
  // real tab that opens a designed coming-soon panel naming the missing
  // engine exports, rather than a dead label — a placeholder is a statement,
  // not a decoration.

  import LogsTab from './LogsTab.svelte';
  import LaunchTab from './LaunchTab.svelte';
  import StatusTab from './StatusTab.svelte';
  import { placeholder, placeholdersFor } from '$lib/agent/placeholders.js';

  let { fleet, node, nodes = [], tab = 'launch', ontab, log = [], onlog } = $props();

  // The registry enforces the dock's cap of one placeholder tab; reading it
  // here (rather than hardcoding a fourth tab) means adding a fifth fails a
  // test instead of quietly densifying the dock.
  const soonTabs = $derived(placeholdersFor('dock', { solo: false }));
  const requests = $derived(placeholder('requests-tab'));

  const TABS = [
    { id: 'logs', label: 'Logs' },
    { id: 'launch', label: 'Launch' },
    { id: 'status', label: 'Status' }
  ];
</script>

<section class="dock" aria-label="Console">
  <div class="dock-tabs" role="tablist" aria-label="Console tabs">
    {#each TABS as t (t.id)}
      <button
        type="button"
        role="tab"
        aria-selected={tab === t.id}
        class="dock-tab"
        class:dock-tab-on={tab === t.id}
        onclick={() => ontab?.(t.id)}
      >
        {t.label}
      </button>
    {/each}
    {#each soonTabs as t (t.id)}
      <button
        type="button"
        role="tab"
        aria-selected={tab === t.id}
        class="dock-tab dock-tab-soon"
        class:dock-tab-on={tab === t.id}
        onclick={() => ontab?.(t.id)}
      >
        {t.label} <span class="cs-chip">soon</span>
      </button>
    {/each}
  </div>

  <!-- The one scroll region the stage's fixed rows leave. A scroll region
       must be keyboard-reachable or its overflow is mouse-only content. -->
  <!-- svelte-ignore a11y_no_noninteractive_tabindex -->
  <div class="dock-body" role="tabpanel" tabindex="0">
    {#if tab === 'logs'}
      <LogsTab {fleet} {node} {nodes} />
    {:else if tab === 'launch'}
      <LaunchTab {fleet} {node} {nodes} {onlog} />
    {:else if tab === 'status'}
      <StatusTab {fleet} {node} {nodes} {log} />
    {:else if tab === 'requests-tab'}
      <div class="dock-soon">
        <p class="dock-soon-head">Requests <span class="cs-chip">soon</span></p>
        <p class="dock-soon-text">{requests.soon}</p>
        <p class="dock-soon-text">
          When the engine exports them, this tab becomes a per-request table
          with KV-cache occupancy and queue depth — in this footprint, with
          nothing else moving.
        </p>
      </div>
    {/if}
  </div>
</section>
