<!-- SPDX-License-Identifier: AGPL-3.0-only -->
<script>
  // The cluster launch overlay: the Prepare→Commit ceremony, promoted out of
  // the console dock into the centred dialog the spec draws.
  //
  // Two structural decisions live here rather than in ClusterLaunch:
  //
  // **Abort is pinned in the footer, always visible.** The ceremony's body
  // scrolls — recipe, machines, per-rank commands, reservations — but its
  // escape hatch may never scroll out of view. When nothing is reserved the
  // button renders disabled with the stated reason, never hidden, exactly
  // like the actions bar's verbs.
  //
  // **The flow state is the PAGE's, bound through.** Closing this overlay is
  // not abandoning the prepare: the reservations stay held, the rail's
  // cluster summary keeps saying so, and reopening lands exactly where the
  // operator left. Only Abort releases.
  //
  // The dialog carries id="launch": the anchor the pre-bridge page put on its
  // launch section, kept so deep links and the @live e2e spec keep working —
  // arriving with #launch in the URL opens this overlay.

  import Overlay from './Overlay.svelte';
  import ClusterLaunch from './ClusterLaunch.svelte';

  let { fleet, flow = $bindable(), onclose } = $props();

  let launcher = $state(null);

  const held = $derived(flow.epoch != null && flow.phase === 'prepared');
  // Anything reserved, or answered: 'prepared' covers both the all-agreed
  // and the some-refused outcomes, and both hold state worth releasing.
  const abortable = $derived(flow.phase === 'prepared');
</script>

<Overlay label="Cluster launch" id="launch" wide {onclose}>
  <p class="stage-sub">
    Two phases, because one cannot fail cleanly: every machine validates and
    reserves, and nothing starts until all of them have agreed.
  </p>
  <ClusterLaunch {fleet} bind:flow bind:this={launcher} />

  {#snippet footer()}
    {#if held}
      <span class="ov-epoch mono">epoch {flow.epoch} held</span>
    {/if}
    <button
      type="button"
      class="btn ov-abort"
      class:ov-abort-live={abortable}
      aria-disabled={!abortable}
      aria-describedby={abortable ? undefined : 'ov-abort-why'}
      title={abortable ? undefined : 'Nothing is reserved — there is nothing to abort.'}
      onclick={() => abortable && launcher?.abandon()}
    >
      Abort — release the reservations
    </button>
    {#if !abortable}
      <span class="visually-hidden" id="ov-abort-why">
        Nothing is reserved — there is nothing to abort.
      </span>
    {/if}
    <button type="button" class="btn ov-close" onclick={() => onclose?.()}>
      Close{held ? ' — reservations stay held' : ''}
    </button>
  {/snippet}
</Overlay>
