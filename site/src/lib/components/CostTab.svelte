<script>
  // The Cost tab: one inner tab per subject, one CostSubjectPanel for the
  // active one. The subject list is concurrency-subjects.json — the SAME SSOT
  // the Concurrency tab reads, so a subject cannot exist on one tab and not
  // the other, and no cost-specific model list is minted here.
  //
  // `subject` and `rung` are bound up to the dashboard so the deep link
  // (#bench=cost&subject=…&c=…) both selects and follows a click.
  import TabStrip from './TabStrip.svelte';
  import CostSubjectPanel from './CostSubjectPanel.svelte';
  import { SUBJECTS, formatUnassigned, recordsOf, unassignedRecords } from '$lib/concurrency-subjects.js';
  import { energyOf, rungsOf } from '$lib/cost.js';

  /**
   * @type {{
   *   subject: string,
   *   rung: string|number,
   *   benches: string[],
   *   recordsFor: (benchId: string) => object[],
   *   onselect: (recs: object[]) => void
   * }}
   */
  let { subject = $bindable(), rung = $bindable(), benches, recordsFor, onselect } = $props();

  const bySubject = $derived(SUBJECTS.map((s) => ({ s, records: recordsOf(s, recordsFor) })));
  // The chip states what the tab holds BEFORE it is opened, and it is derived:
  // a subject with records but no joules is a different state from one with no
  // records at all, and neither is "0".
  const tabs = $derived(
    bySubject.map(({ s, records }) => {
      const withEnergy = records.some((r) => rungsOf(r).some((c) => energyOf(r, c).state !== 'absent'));
      return {
        id: s.id,
        label: s.label,
        chip: withEnergy ? undefined : records.length === 0 ? 'no runs yet' : 'energy not yet measured'
      };
    })
  );
  const active = $derived.by(() => {
    const found = bySubject.find((b) => b.s.id === subject);
    // The dashboard resolves the hash through resolveSubject before this
    // mounts, so an unknown id here is a wiring bug, not user input.
    if (!found) throw new Error(`CostTab: unknown subject ${JSON.stringify(subject)}`);
    return found;
  });
  const unassigned = $derived(unassignedRecords(benches, recordsFor));
</script>

<div class="ct">
  <TabStrip prefix="co" label="Cost subjects" {tabs} bind:active={subject} />
  <CostSubjectPanel subject={active.s} records={active.records} bind:rung {onselect} />
  {#if unassigned.length > 0}
    <p class="bd-footnote">
      unassigned: {unassigned.map(formatUnassigned).join(' · ')}
    </p>
  {/if}
</div>
