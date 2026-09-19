<script>
  // The Concurrency tab: one inner tab per subject (the SSOT list in
  // concurrency-subjects.json), one ConcurrencySubjectPanel for the active
  // one. The subject strip IS the model filter on this tab — the dashboard's
  // global model select is hidden while this tab is up.
  //
  // `subject` is bound to the dashboard so the deep link
  // (#bench=concurrency&subject=…) both selects a tab and follows a click.
  import TabStrip from './TabStrip.svelte';
  import ConcurrencySubjectPanel from './ConcurrencySubjectPanel.svelte';
  import { SUBJECTS, formatUnassigned, recordsOf, unassignedRecords } from '$lib/concurrency-subjects.js';

  /**
   * @type {{
   *   subject: string,
   *   rungs: number[],
   *   benches: string[],
   *   recordsFor: (benchId: string) => object[],
   *   onselect: (recs: object[]) => void
   * }}
   */
  let { subject = $bindable(), rungs, benches, recordsFor, onselect } = $props();

  const bySubject = $derived(SUBJECTS.map((s) => ({ s, records: recordsOf(s, recordsFor) })));
  // A subject with no records still gets a tab — the owner wants the empty
  // state shown — and the chip is inside the button, so the state is heard
  // when the tab is reached by arrow key, not only seen.
  const tabs = $derived(
    bySubject.map(({ s, records }) => ({
      id: s.id,
      label: s.label,
      chip: records.length === 0 ? 'not yet measured' : undefined
    }))
  );
  const active = $derived.by(() => {
    const found = bySubject.find((b) => b.s.id === subject);
    // The dashboard resolves the hash through resolveSubject before this
    // mounts, so an unknown id here is a wiring bug, not user input.
    if (!found) throw new Error(`ConcurrencyTab: unknown subject ${JSON.stringify(subject)}`);
    return found;
  });
  // Listed, never dropped: a new checkpoint's runs must be visible somewhere
  // until it gets a subject.
  const unassigned = $derived(unassignedRecords(benches, recordsFor));
</script>

<div class="ct">
  <TabStrip prefix="cs" label="Concurrency subjects" {tabs} bind:active={subject} />
  <ConcurrencySubjectPanel subject={active.s} records={active.records} {rungs} {onselect} />
  {#if unassigned.length > 0}
    <p class="bd-footnote">
      unassigned: {unassigned.map(formatUnassigned).join(' · ')}
    </p>
  {/if}
</div>
