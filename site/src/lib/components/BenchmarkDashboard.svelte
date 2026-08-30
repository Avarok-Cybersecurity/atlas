<script>
  import { modal } from '$lib/modal.js';
  // The hero's benchmark dashboard modal. One tab per benchmark family, a
  // model switcher that filters but never relabels, and a metadata card on
  // every chart point. Data: gates.generated.json — the union of gate records
  // across ALL branches at build time, so the newest run shows even before its
  // PR merges (provenance shown per point and in the footer).
  import ConcurrencyLadder from './ConcurrencyLadder.svelte';
  import GateBenchSection from './GateBenchSection.svelte';
  import GatePointCard from './GatePointCard.svelte';
  import { gateData, tabs, unpublished, models, recordsFor, benchName, shortModel, colorFor } from '$lib/gates.js';
  import { groupFor, groupRecords, groupedBenches } from '$lib/gate-variants.js';

  let { onclose } = $props();

  let activeTab = $state(tabs[0]?.id);
  let modelFilter = $state('all');
  let selected = $state(null);
  let dialogEl = $state(null);

  const tab = $derived(tabs.find((t) => t.id === activeTab) ?? tabs[0]);
  const keep = (r) => modelFilter === 'all' || r.target_model === modelFilter;
  // Grouped benches (see gate-variants.js) collapse into ONE section drawn
  // under the group's primary id, so the concurrency ladder renders as two
  // lines on one axis instead of two panels that cannot be read against each
  // other. Everything else keeps the one-bench-one-section shape.
  const sections = $derived.by(() => {
    const out = [];
    const done = new Set();
    for (const b of tab?.benches ?? []) {
      if (done.has(b)) continue;
      const group = groupFor(b);
      if (group) {
        group.members.forEach((m) => done.add(m.bench));
        const records = groupRecords(group, recordsFor).filter(keep);
        if (records.length > 0)
          out.push({ benchId: group.primary, name: benchName(group.primary), records });
      } else {
        done.add(b);
        const records = recordsFor(b).filter(keep);
        if (records.length > 0) out.push({ benchId: b, name: benchName(b), records });
      }
    }
    return out;
  });
  // A grouped member is never "hidden": its records are drawn inside the
  // group's section under the primary's id, so matching on benchId alone
  // would accuse the DFlash2 gate of being filtered out on every render.
  const hiddenByFilter = $derived(
    (tab?.benches ?? []).filter(
      (b) =>
        recordsFor(b).length > 0 &&
        !groupedBenches.has(b) &&
        !sections.some((s) => s.benchId === b)
    )
  );
  const src = gateData.sources;

  // Focus-in, the Tab trap and focus-return all live in `use:modal` below.
  // This effect used to call `dialogEl?.focus()` and stop there — a dialog
  // that claims `aria-modal="true"` while Tab still walks the page behind it,
  // which is the half of the contract that actually keeps a keyboard operator
  // inside. Body-scroll lock stays here: it is this dialog's own concern.
  $effect(() => {
    document.body.style.overflow = 'hidden';
    return () => (document.body.style.overflow = '');
  });

  function onkeydown(e) {
    if (e.key === 'Escape' && !selected) onclose();
  }
</script>

<svelte:window {onkeydown} />

<div class="bd-backdrop" onclick={onclose} role="presentation">
  <div
    class="bd"
    role="dialog"
    aria-modal="true"
    aria-label="Atlas benchmark dashboard"
    tabindex="-1"
    bind:this={dialogEl}
    use:modal
    onclick={(e) => e.stopPropagation()}
  >
    <header class="bd-head">
      <div class="bd-head-titles">
        <span class="slabel bd-label">gate receipts</span>
        <h2 class="bd-title">Benchmark dashboard</h2>
      </div>
      <button type="button" class="bd-close" onclick={onclose} aria-label="Close dashboard">✕</button>
    </header>

    <div class="bd-controls">
      <!-- div, not <nav>: app.css styles the bare nav element (position:fixed). -->
      <div class="bd-tabs" role="tablist" aria-label="Benchmarks">
        {#each tabs as t}
          <button
            type="button"
            role="tab"
            aria-selected={t.id === activeTab}
            class="bd-tab"
            class:is-active={t.id === activeTab}
            onclick={() => (activeTab = t.id)}>{t.label}</button>
        {/each}
      </div>
      <label class="bd-model">
        <span class="bd-model-label">model</span>
        <select bind:value={modelFilter} aria-label="Filter by model">
          <option value="all">all models</option>
          {#each models as m}
            <option value={m}>{shortModel(m)}</option>
          {/each}
        </select>
      </label>
    </div>

    <div class="bd-body">
      {#if activeTab === 'concurrency'}
        <ConcurrencyLadder />
      {/if}
      {#each sections as s (s.benchId)}
        <GateBenchSection {...s} onselect={(rec) => (selected = rec)} />
      {/each}
      {#if sections.length === 0 && activeTab !== 'concurrency'}
        <p class="bd-empty">No records for this model in this benchmark family.</p>
      {/if}
      {#each hiddenByFilter as b}
        {@const rs = recordsFor(b)}
        <p class="bd-filtered-note">
          <span class="gpc-swatch" style="background:{colorFor(rs[0].target_model)}" aria-hidden="true"></span>
          {benchName(b)} runs on {shortModel(rs[0].target_model)} — switch the model filter to see its {rs.length} records.
        </p>
      {/each}
      {#if activeTab === 'bfcl'}
        <p class="bd-footnote">
          The two BFCL charts use different models AND different sample draws (see a point's run
          parameters) — scores are comparable within a chart, not across them.
        </p>
      {/if}
    </div>

    <footer class="bd-foot">
      <span>
        {src.committed + src.from_branches} records · {src.branches_scanned} branches scanned
        {#if src.from_branches > 0}· {src.from_branches} from remote branch heads{/if}
        · as of {gateData.generated_date} ({gateData.generated_sha})
      </span>
      {#if unpublished.length > 0}
        <span class="bd-unpublished">gated, not yet published: {unpublished.join(', ')}</span>
      {/if}
    </footer>
  </div>
</div>

{#if selected}
  <GatePointCard record={selected} onclose={() => (selected = null)} />
{/if}
