<script>
  // One benchmark inside a dashboard tab: the latest-run stat tiles, then a
  // GateChart per panel spec. The model chip is ALWAYS rendered — a 27B number
  // must never be readable as a 35B one.
  import GateChart from './GateChart.svelte';
  import GateLadderChart from './GateLadderChart.svelte';
  import { panelsFor, colorFor, shortModel, fmtDate, sampleCount, ladderPoints } from '$lib/gates.js';
  import { variantLabel } from '$lib/gate-variants.js';

  let { benchId, name, records, onselect } = $props();

  const latest = $derived(records[records.length - 1]);
  const panels = $derived(panelsFor(benchId, records));
  const headline = $derived.by(() => {
    const m = latest?.metrics ?? {};
    if (benchId === 'agentic-webserver')
      return [
        { label: 'webserver_ok', value: `${m.webserver_ok}/${m.iterations}` },
        { label: 'Σ wall', value: `${Math.round(m.sum_wall_s).toLocaleString('en-US')} s` }
      ];
    if (benchId.startsWith('bfcl'))
      return [
        { label: 'overall', value: m.overall_accuracy },
        { label: 'normalized', value: m.normalized_single_turn_score }
      ];
    if (benchId.startsWith('ttft'))
      return [
        { label: 'median', value: `${Math.round(m.median_ms).toLocaleString('en-US')} ms` },
        { label: 'p90', value: `${Math.round(m.p90_ms).toLocaleString('en-US')} ms` }
      ];
    if (benchId === 'concurrency-sweep') {
      // Guarded per key: the tiles must survive a record that predates the
      // metrics map (or one whose cells were all vacuous and published no
      // comparable curve).
      const tiles = [];
      // The latest record in a grouped section can belong to either variant,
      // so the tile has to say which one it is. Unlabelled, a DFlash2 peak
      // would read as the plain gate's under the plain gate's heading.
      const variant = variantLabel(latest.benchmark_id);
      if (Number.isFinite(m.peak_aggregate_tok_s))
        tiles.push({
          label: variant ? `peak · ${variant}` : 'peak',
          value: `${(+m.peak_aggregate_tok_s).toFixed(1)} tok/s`
        });
      const rungs = ladderPoints(latest).length;
      if (rungs > 0) tiles.push({ label: 'rungs', value: rungs });
      if (Number.isFinite(m.vacuous_cells) && m.vacuous_cells > 0)
        tiles.push({ label: 'vacuous cells', value: m.vacuous_cells });
      return tiles;
    }
    return [];
  });
</script>

<!-- article, not <section>: app.css pads the bare section element (5.5rem+). -->
<article class="gbs" aria-label="{name} results">
  <header class="gbs-head">
    <div class="gbs-title-row">
      <h3 class="gbs-name">{name}</h3>
      <span class="gbs-model" style="border-color:{colorFor(latest.target_model)}; color:{colorFor(latest.target_model)}">
        {shortModel(latest.target_model)}
      </span>
    </div>
    <div class="gbs-tiles">
      {#each headline as tile}
        <div class="gbs-tile">
          <span class="gbs-tile-val">{tile.value}</span>
          <span class="gbs-tile-label">{tile.label}</span>
        </div>
      {/each}
      <div class="gbs-tile">
        <span class="gbs-tile-val gbs-verdict" data-verdict={latest.verdict}
          >{latest.verdict === 'PASS' ? '✓ PASS' : '✗ ' + latest.verdict}</span>
        <span class="gbs-tile-label">latest · {fmtDate(latest.recorded_at)}{#if sampleCount(latest) !== null}&nbsp;· n={sampleCount(latest)}{/if}</span>
      </div>
    </div>
  </header>

  {#each panels as panel}
    {#if panel.kind === 'ladder'}
      <GateLadderChart {records} {panel} {onselect} />
    {:else}
      <GateChart {records} {panel} {onselect} />
    {/if}
  {/each}
</article>
