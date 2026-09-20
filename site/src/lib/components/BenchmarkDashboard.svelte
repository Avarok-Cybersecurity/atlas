<script>
  import { modal } from '$lib/modal.js';
  // The hero's benchmark dashboard modal. One tab per benchmark family, a
  // model switcher that filters but never relabels, and a metadata card on
  // every chart point. Data: gates.generated.json — the union of gate records
  // across ALL branches at build time, so the newest run shows even before its
  // PR merges (provenance shown per point and in the footer).
  import ConcurrencyTab from './ConcurrencyTab.svelte';
  import GateBenchSection from './GateBenchSection.svelte';
  import GatePointCard from './GatePointCard.svelte';
  import TabStrip from './TabStrip.svelte';
  import { browser } from '$app/environment';
  import { replaceState } from '$app/navigation';
  import { gateData, tabs, unpublished, models, recordsFor, benchName, shortModel, colorFor } from '$lib/gates.js';
  import { SUBJECTS, rungsDeclared } from '$lib/concurrency-subjects.js';
  import { formatDashboardHash, isDeepLink, parseDashboardHash } from '$lib/dashboard-link.js';

  let { onclose } = $props();

  // What a hash may name: the tabs that earned one, the subjects in the SSOT,
  // and every rung any subject's gate declares — so `c=64` is a valid link even
  // on a subject whose gate stops at C=16 (it lands on a labelled "not run at
  // this rung" panel), while `c=3` never is.
  const known = {
    tabIds: tabs.map((t) => t.id),
    subjectIds: SUBJECTS.map((s) => s.id),
    rungs: [...new Set(SUBJECTS.flatMap((s) => rungsDeclared(s, recordsFor)))].sort((a, b) => a - b)
  };
  const readLink = () => parseDashboardHash(browser ? location.hash : '', known);
  const initial = readLink();

  // A deep link picks the tab; a plain open lands on the first tab as before.
  let activeTab = $state(initial.tab ?? tabs[0]?.id);
  // `subject` drives the concurrency tab's inner strip (bound both ways, so a
  // click and a pasted link agree). `rung` is held and written back so a
  // `c=64` link survives until the rung navigator mounts (a later step).
  let subject = $state(initial.subject);
  let rung = $state(initial.c);
  let modelFilter = $state('all');
  // The record(s) behind the clicked chart point. An array because one plotted
  // point can stand for several grouped runs — see GatePointCard.
  let selected = $state(null);
  let dialogEl = $state(null);

  const tab = $derived(tabs.find((t) => t.id === activeTab) ?? tabs[0]);
  const onConcurrency = $derived(activeTab === 'concurrency');
  const keep = (r) => modelFilter === 'all' || r.target_model === modelFilter;
  // One bench, one section — except on the concurrency tab, where the subject
  // tabs own every record (ConcurrencyTab) and the model select is hidden:
  // the subject strip is the model filter there, and a second filter on top
  // of it could empty a tab that still has records.
  const sections = $derived(
    onConcurrency
      ? []
      : (tab?.benches ?? [])
          .map((b) => ({ benchId: b, name: benchName(b), records: recordsFor(b).filter(keep) }))
          .filter((s) => s.records.length > 0)
  );
  const hiddenByFilter = $derived(
    onConcurrency
      ? []
      : (tab?.benches ?? []).filter((b) => recordsFor(b).length > 0 && !sections.some((s) => s.benchId === b))
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

  // The URL is always the deep link to what is on screen. `replaceState` from
  // $app/navigation, because a bare history.replaceState nulls the history
  // metadata SvelteKit keeps there and breaks Back; replace rather than push so
  // tab flips do not pile up entries. Subject and rung travel only with the
  // concurrency tab — a TTFT link has no subject.
  $effect(() => {
    const hash = formatDashboardHash({
      tab: activeTab,
      subject: onConcurrency ? subject : null,
      c: onConcurrency ? rung : null
    });
    replaceState(hash ? `#${hash}` : location.pathname + location.search, {});
  });
  // Cleared on close — but only while the hash is still ours, so a route
  // change mid-open cannot eat the next page's own hash (the deck uses one).
  $effect(() => () => {
    if (isDeepLink(readLink())) replaceState(location.pathname + location.search, {});
  });

  // A URL pasted in place, or Back/Forward between two deep links, re-syncs.
  // replaceState does not fire this, so the write-back above cannot loop.
  function onhashchange() {
    const link = readLink();
    if (!isDeepLink(link)) return;
    activeTab = link.tab;
    subject = link.subject;
    rung = link.c;
  }

  function onkeydown(e) {
    if (e.key === 'Escape' && !selected) onclose();
  }
</script>

<svelte:window {onkeydown} {onhashchange} />

<div class="bd-backdrop" onclick={onclose} role="presentation">
  <div
    class="bd"
    role="dialog"
    aria-modal="true"
    aria-label="Avarok benchmark dashboard"
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
      <TabStrip prefix="bd" label="Benchmarks" {tabs} bind:active={activeTab} />
      {#if !onConcurrency}
        <label class="bd-model">
          <span class="bd-model-label">model</span>
          <select bind:value={modelFilter} aria-label="Filter by model">
            <option value="all">all models</option>
            {#each models as m}
              <option value={m}>{shortModel(m)}</option>
            {/each}
          </select>
        </label>
      {/if}
    </div>

    <!-- The outer tabpanel. Nested tablists (subjects, rungs) live INSIDE this
         panel, never inside a tab, so each stays its own roving group. -->
    <div
      class="bd-body"
      id="bd-panel-{activeTab}"
      role="tabpanel"
      aria-labelledby="bd-tab-{activeTab}"
      tabindex="-1"
    >
      {#if onConcurrency}
        <ConcurrencyTab
          bind:subject
          rungs={known.rungs}
          benches={tab.benches}
          {recordsFor}
          onselect={(recs) => (selected = recs)}
        />
      {/if}
      {#each sections as s (s.benchId)}
        <GateBenchSection {...s} onselect={(recs) => (selected = recs)} />
      {/each}
      {#if sections.length === 0 && !onConcurrency}
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
  <GatePointCard records={selected} onclose={() => (selected = null)} />
{/if}
