<script>
  // One subject of the Concurrency tab: header tiles, then three blocks in a
  // fixed order — the comparison chart, the per-rung navigator (later), the
  // latest gate sweep. The order is the argument: "how do we compare" before
  // "how are we moving".
  //
  // The two charts share ONE instrument as of #1220 (2026-09-21/22): the
  // gate was re-pointed onto the published ladder's own axes (ISL 128 / OSL
  // 1024 / essay / ctx 2048 / batch 128 / fp8 KV) and its throughput recipe.
  // Until then they were different instruments (the gate ran ISL 512 / OSL
  // 320 on the agentic recipe) and the top chart read ~4x the bottom one — a
  // sentence that is now FALSE, which is why this comment says so rather than
  // leaving a reader to reconcile two charts that finally agree. Every title
  // still carries its instrument, and the note between the charts branches on
  // `state`: "Same instrument as the chart above" once a live record pairs,
  // "Not the chart above's instrument" while the gate still shows the
  // retired one.
  import ConcurrencyComparison, {
    baselineTileOf,
    comparisonStateOf,
    instrumentLabel,
    liveRecordOf,
    publishedFor
  } from './ConcurrencyComparison.svelte';
  import GateLadderChart from './GateLadderChart.svelte';
  import { colorFor, fmtDate, ladderPoints } from '$lib/gates.js';
  import { rungsDeclared } from '$lib/concurrency-subjects.js';

  /** @type {{ subject: object, records: object[], rungs: number[], onselect: (recs: object[]) => void }} */
  let { subject, records, rungs, onselect } = $props();

  const latest = $derived(records[records.length - 1] ?? null);
  const live = $derived(liveRecordOf(records));
  const state = $derived(comparisonStateOf(subject, records));
  const published = $derived(publishedFor(subject));
  // The vLLM baseline tile follows the chart's own decision (a pair, a dated
  // one-shot, one on another instrument, none); the date is read from the
  // one-shot's rungs, never typed.
  const baselineTile = $derived(baselineTileOf(subject, records));
  // `records` is already this subject's, so the lookup is the identity.
  const declared = $derived(rungsDeclared(subject, () => records));

  // Header tiles, every value read from a record or the subject list. The
  // verdict tile exists only when there is a verdict: an empty subject shows
  // its state, never a red or a zero.
  const tiles = $derived.by(() => {
    if (!latest) {
      return [
        { value: '0', label: 'records' },
        { value: 'declared, unmeasured', label: `gate ${subject.gate}` },
        { value: baselineTile, label: 'vLLM baseline' }
      ];
    }
    const out = [];
    const pts = ladderPoints(latest);
    if (pts.length > 0) {
      const peak = pts.reduce((a, b) => (b.v > a.v ? b : a));
      out.push({ value: `${peak.v.toFixed(1)} tok/s`, label: `peak (C=${peak.c}) · ${fmtDate(latest.recorded_at)}` });
    }
    out.push({ value: `${declared.length} of ${rungs.length}`, label: 'rungs declared' });
    out.push({ value: baselineTile, label: 'vLLM baseline' });
    return out;
  });

  const gatePanel = $derived(
    latest ? { title: `latest gate sweep · ${instrumentLabel(latest)}`, unit: 'tok/s' } : null
  );
</script>

<div
  class="csp"
  id="cs-panel-{subject.id}"
  role="tabpanel"
  aria-labelledby="cs-tab-{subject.id}"
  tabindex="-1"
>
  <header class="gbs-head">
    <div class="gbs-title-row">
      <h3 class="gbs-name">{subject.label}</h3>
      <!-- The checkpoint id, verbatim: an FP8 must never read as an NVFP4. -->
      <span class="gbs-model" style="border-color:{colorFor(subject.checkpoint)}; color:{colorFor(subject.checkpoint)}">
        {subject.checkpoint}
      </span>
    </div>
    <div class="gbs-tiles">
      {#each tiles as tile}
        <div class="gbs-tile">
          <span class="gbs-tile-val">{tile.value}</span>
          <span class="gbs-tile-label">{tile.label}</span>
        </div>
      {/each}
      {#if latest}
        <div class="gbs-tile">
          <span class="gbs-tile-val gbs-verdict" data-verdict={latest.verdict}
            >{latest.verdict === 'PASS' ? '✓ PASS' : '✗ ' + latest.verdict}</span>
          <span class="gbs-tile-label">latest · {fmtDate(latest.recorded_at)} · {latest.git_sha}</span>
        </div>
      {/if}
    </div>
  </header>

  <ConcurrencyComparison {subject} {records} {rungs} {onselect} />

  <!-- RungNavigator mounts here (plan-concurrency-ia.md §3): the "All · C=1 …
       C=128" strip over RungGrid / one GateChart per rung, fed by
       rung-series.js#rungPanel. It is the block between the comparison and
       the gate sweep by design, so it is reserved here rather than appended
       later. Nothing is stubbed: an empty navigator would be a fourth chart
       with no data behind it. -->

  {#if latest && gatePanel}
    <p class="cmp-bridge">
      {#if state === 'published'}
        Not the chart above's instrument: this is the gate's
        <code>{instrumentLabel(latest)}</code>, against the published pair's
        ISL {published.workload.isl_tokens} / OSL {published.workload.osl_tokens} with speculation pinned —
        so its tok/s are not the ladder's tok/s. Read each chart against its own history, never one
        against the other.
      {:else if live}
        Same instrument as the chart above, which draws the newest passing run on main
        ({fmtDate(live.recorded_at)} · {live.git_sha}); this chart adds the run history around it.
      {:else}
        No passing run on main yet to compare against; the runs here are from branches or did not
        pass.
      {/if}
    </p>
    <GateLadderChart {records} panel={gatePanel} {onselect} />
  {/if}
</div>
