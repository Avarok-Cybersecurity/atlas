<script>
  import { modal } from '$lib/modal.js';
  // The per-point metadata card ("mini modal"). Raised by clicking a chart
  // point.
  //
  // A chart point may stand for more than one run: once a series exceeds the
  // per-chart point budget, adjacent runs are grouped and the chart plots one
  // of them. Clicking such a point must still reach EVERY run inside it, so
  // this card takes an array and gives a group one tab per commit. The plotted
  // run is not privileged over the others — they are all real receipts.
  import { shortModel, fmtDate } from '$lib/gates.js';
  import { moveTab } from '$lib/tablist.js';
  import GateRecordBody from './GateRecordBody.svelte';

  let { records, onclose } = $props();

  // Newest first inside a group: the chart emphasises the latest value, so the
  // tab that opens should be the one a reader is most likely to be after.
  let active = $state(0);
  $effect(() => {
    records;
    active = records.length - 1;
  });

  const r = $derived(records[Math.min(active, records.length - 1)]);
  const many = $derived(records.length > 1);

  function onkeydown(e) {
    if (e.key === 'Escape') { e.stopPropagation(); onclose(); }
  }
  function ontabkey(e) {
    const to = moveTab(e.key, active, records.length);
    if (to === null) return;
    e.preventDefault();
    active = to;
    e.currentTarget.parentElement?.querySelectorAll('[role="tab"]')[to]?.focus();
  }
</script>

<svelte:window {onkeydown} />

<div class="gpc-backdrop" onclick={onclose} role="presentation">
  <article
    class="gpc receipt"
    role="dialog"
    aria-modal="true"
    aria-label={many
      ? `Gate records, ${records.length} runs from ${fmtDate(records[0].recorded_at)} to ${fmtDate(records[records.length - 1].recorded_at)}`
      : `Gate record ${r.git_sha}`}
    use:modal
    onclick={(e) => e.stopPropagation()}
  >
    <div class="receipt-body">
      {#if many}
        <!-- Roving tabindex: only the active tab is reachable by Tab, so the
             dialog's focus trap cycles through the tablist as one stop and the
             arrow keys move within it. -->
        <div class="gpc-tabs" role="tablist" aria-label="runs grouped into this point">
          {#each records as rec, i}
            <button
              type="button"
              role="tab"
              id="gpc-tab-{i}"
              aria-controls="gpc-panel"
              aria-selected={i === active}
              tabindex={i === active ? 0 : -1}
              class:is-active={i === active}
              onclick={() => (active = i)}
              onkeydown={ontabkey}
            >
              <span class="gpc-tab-sha">{rec.git_sha.slice(0, 9)}</span>
              <span class="gpc-tab-date">{fmtDate(rec.recorded_at)}</span>
              {#if rec.verdict !== 'PASS'}<span class="gpc-tab-fail" aria-label="failed">●</span>{/if}
            </button>
          {/each}
        </div>
      {/if}

      <div id="gpc-panel" role={many ? 'tabpanel' : undefined} aria-labelledby={many ? `gpc-tab-${active}` : undefined}>
        <GateRecordBody record={r} />
      </div>

      <div class="receipt-foot">
        <span>{shortModel(r.target_model)}{#if many} · run {active + 1} of {records.length}{/if}</span>
        <button type="button" class="gpc-close" onclick={onclose}>close</button>
      </div>
    </div>
  </article>
</div>
