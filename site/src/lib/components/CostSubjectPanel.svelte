<script>
  // One subject of the Cost tab, in the order a reader needs it: what it
  // costs against the other engine (chart A), then whether that is moving
  // (chart B), then every assumption the two rest on.
  //
  // Nothing here is a filter. Every rung Atlas and vLLM both measured is on
  // chart A whether Atlas wins it or loses it, and the headline tile names
  // the loser when there is one — `vLLM cheaper at n of n rungs` is a state
  // this section is built to render, not an edge case it avoids.
  import CostLadderChart from './CostLadderChart.svelte';
  import GateChart from './GateChart.svelte';
  import LADDERS from '$lib/ladders.generated.json';
  import { colorFor, fmtDate } from '$lib/gates.js';
  import { instrumentLabel } from '$lib/concurrency-comparison.js';
  import {
    DEFAULT_USD_PER_KWH,
    DISCLOSURES,
    PRICE_STORAGE_KEY,
    baselineSnapshots,
    costLadder,
    costPerMillion,
    costTrend,
    emptyStateOf,
    extremeRungs,
    fmtUsd,
    rungsOf,
    verdictTile
  } from '$lib/cost.js';

  /**
   * @type {{
   *   subject: object,
   *   records: object[],
   *   rung: string|number,
   *   onselect: (recs: object[]) => void,
   *   ladders?: object
   * }}
   */
  let { subject, records, rung = $bindable(), onselect, ladders = LADDERS } = $props();

  const cost = $derived(costLadder(subject, records, ladders));
  const empty = $derived(emptyStateOf(subject, records, ladders));
  const latest = $derived(records[records.length - 1] ?? null);

  // -- the price: the reader's input, never a measurement, never in the URL --
  let priceInput = $state(DEFAULT_USD_PER_KWH);
  const priceValid = $derived(Number.isFinite(priceInput) && priceInput > 0);
  const usdPerKwh = $derived(priceValid ? priceInput : DEFAULT_USD_PER_KWH);
  // localStorage is per viewer and per origin, and every access is guarded:
  // a private window, cleared site data or a browser set to block storage
  // throws on the accessor itself, and the page must still render.
  $effect(() => {
    try {
      const raw = localStorage.getItem(PRICE_STORAGE_KEY);
      const v = Number(raw);
      if (raw !== null && Number.isFinite(v) && v > 0) priceInput = v;
    } catch {
      /* no stored price: the default stands, and it is labelled as one */
    }
  });
  $effect(() => {
    if (!priceValid) return;
    try {
      localStorage.setItem(PRICE_STORAGE_KEY, String(usdPerKwh));
    } catch {
      /* storage refused: the price still applies to this view */
    }
  });

  // -- total vs above idle ---------------------------------------------------
  let aboveIdle = $state(false);
  const idle = $derived(cost.idle);
  const showAboveIdle = $derived(aboveIdle && idle.both);

  // -- tiles, every one derived ---------------------------------------------
  const extremes = $derived(extremeRungs(cost.verdicts));
  // One place computes dollars from joules (cost.js); this only chooses which
  // joule count the current view is showing.
  const usdAt = (e) =>
    fmtUsd(costPerMillion((showAboveIdle && e.aboveIdleJ !== null ? e.aboveIdleJ : e.energyJ) / e.tokens, usdPerKwh));
  const tiles = $derived.by(() => {
    if (cost.energyState === 'none') {
      return [
        { value: records.length === 0 ? '0' : String(records.length), label: `${subject.gate} records` },
        { value: 'not yet measured', label: 'GPU-rail energy' },
        { value: 'lower bound', label: 'GPU rail only' }
      ];
    }
    const out = [{ value: verdictTile(cost.verdicts), label: 'cost per 1M tokens vs vLLM' }];
    if (extremes.best) {
      const b = extremes.best;
      out.push({
        value: `C=${b.c} · $${usdAt(b.atlas)} vs $${usdAt(b.rival)}`,
        label: `best rung · ${b.atlasCheaper ? 'Atlas' : 'vLLM'} cheaper ×${(b.atlasCheaper ? 1 / b.ratio : b.ratio).toFixed(2)}`
      });
    }
    if (extremes.worst && extremes.worst !== extremes.best) {
      const w = extremes.worst;
      out.push({
        value: `C=${w.c} · $${usdAt(w.atlas)} vs $${usdAt(w.rival)}`,
        label: `worst rung · ${w.atlasCheaper ? 'Atlas' : 'vLLM'} cheaper ×${(w.atlasCheaper ? 1 / w.ratio : w.ratio).toFixed(2)}`
      });
    }
    out.push({ value: 'lower bound', label: 'GPU rail only' });
    return out;
  });

  const chartTitle = $derived(
    `$ per 1M tokens · ${latest ? instrumentLabel(latest) : subject.gate} · ${
      showAboveIdle ? 'above idle (marginal)' : 'total'
    }`
  );

  // -- chart B: one rung at a time -------------------------------------------
  // The rungs any record measured, and the default: the WIDEST one the newest
  // record reached — the operating point a deployment runs at, and the
  // cheapest per token. The deep link's `c` wins when it names a rung this
  // subject measured; `all` is not a cost view (see the section comment).
  const rungsMeasured = $derived([...new Set(records.flatMap(rungsOf))].sort((a, b) => a - b));
  const defaultRung = $derived(rungsMeasured.length ? rungsMeasured[rungsMeasured.length - 1] : null);
  const selectedRung = $derived(
    typeof rung === 'number' && rungsMeasured.includes(rung) ? rung : defaultRung
  );
  const trend = $derived(selectedRung === null ? null : costTrend(selectedRung, records));
  const snapshots = $derived(selectedRung === null ? [] : baselineSnapshots(selectedRung, cost.baselines));
</script>

<div
  class="csp cost-panel"
  id="co-panel-{subject.id}"
  role="tabpanel"
  aria-labelledby="co-tab-{subject.id}"
  tabindex="-1"
>
  <header class="gbs-head">
    <div class="gbs-title-row">
      <h3 class="gbs-name">{subject.label}</h3>
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
    </div>
  </header>

  <!-- The one input on this page, and it is labelled as an input. -->
  <div class="cost-price">
    <label class="cost-price-field">
      <span class="cost-price-label">electricity price</span>
      <span class="cost-price-box">
        <span aria-hidden="true">$</span>
        <input type="number" min="0" step="0.01" inputmode="decimal"
          bind:value={priceInput} aria-label="Electricity price in dollars per kilowatt-hour" />
        <span aria-hidden="true">/ kWh</span>
      </span>
    </label>
    <p class="cost-price-note">
      <strong>Your input, not a measurement.</strong> The default {DEFAULT_USD_PER_KWH} $/kWh is a round
      retail-commercial figure — change it to your own tariff. It is kept in this browser and is never
      written into the link.
      {#if !priceValid}
        <span class="cost-warn">That is not a price; the chart is drawn at {DEFAULT_USD_PER_KWH} $/kWh until it is one.</span>
      {/if}
    </p>
  </div>

  {#if cost.energyState === 'none'}
    <!-- What is missing, what the other engine has, what fills it — in that
         order, each line derived, and never a zero. -->
    <figure class="gate-panel cmp">
      <figcaption class="gate-panel-head">
        <span class="gate-panel-title">{empty.title}</span>
        <span class="gate-panel-unit">$ per 1M tokens</span>
      </figcaption>
      <div class="cmp-empty">
        <p><strong>No GPU-rail energy for <code>{subject.checkpoint}</code> yet.</strong></p>
        <p>Atlas: {empty.atlas}.</p>
        <p>vLLM: {empty.vllm}.</p>
        <p>What fills this: {empty.fills}.</p>
        <p>
          Until then nothing on this tab is drawn at zero: a missing joule count is
          <em>not measured</em>, and a zero would read as free.
        </p>
      </div>
    </figure>
  {:else}
    <div class="cost-toggles">
      <button type="button" class="cmp-chip" aria-pressed={showAboveIdle} disabled={!idle.both}
        onclick={() => (aboveIdle = !aboveIdle)}>
        above idle (marginal)
      </button>
      <span class="cost-toggle-why">
        {#if idle.both}
          showing {showAboveIdle ? 'joules above the resident-model idle draw' : 'total joules — what the meter reads'}
        {:else}
          total only — {idle.why}
        {/if}
      </span>
    </div>

    <CostLadderChart {subject} {cost} {usdPerKwh} rungs={rungsMeasured} title={chartTitle}
      aboveIdle={showAboveIdle} {onselect} />

    <p class="cmp-caption">
      Per-token cost is throughput divided by power, so Atlas is cheaper only where its throughput lead
      exceeds its power cost. The tiles above count the rungs where it does; where it does not, the chart
      says <code>vLLM cheaper ×r</code> under the point and the count flips to name vLLM.
      {#each cost.refused as r}
        The {r.label} one-shot is on another instrument and is not drawn: {r.why}.
      {/each}
      {#each cost.noEnergy as n}
        {n.label} is comparable but carries no joules, so it has no cost curve.
      {/each}
    </p>
  {/if}

  <!-- Chart B: efficiency over time, through GateChart so it inherits the
       dated limit lines, the aggregation marks and the violation rings. -->
  {#if trend && trend.metrics.length > 0}
    <div class="cost-rungs" role="group" aria-label="Rung for the efficiency trend">
      {#each rungsMeasured as c}
        <button type="button" class="cmp-chip" aria-pressed={c === selectedRung} onclick={() => (rung = c)}>
          C={c}
        </button>
      {/each}
    </div>
    <GateChart records={trend.records} panel={{ title: trend.title, unit: trend.unit, metrics: trend.metrics }}
      {onselect} />
    <p class="cmp-caption">
      One point is one certified run: <code>tokens ÷ joules × 3600</code> for that rung's batch. Runs on
      different cost instruments are drawn as separate lines and are never joined —
      {trend.generations.length > 1
        ? `${trend.generations.length} instruments here (${trend.generations
            .filter((g) => g.differs)
            .map((g) => g.differs)
            .join('; ')})`
        : `one instrument, ${trend.runs} ${trend.runs === 1 ? 'run' : 'runs'}`}.
      {#each snapshots as s}
        {s.label} measured {s.tokPerWh.toFixed(0)} tok/Wh at C={selectedRung} on {s.date} — a one-shot,
        not re-run, and not a line.
      {/each}
      {#each trend.excluded as e}
        Excluded from this trend: {e.reason}.
      {/each}
    </p>
  {:else if cost.energyState !== 'none'}
    <p class="cmp-caption">
      No trustworthy tokens-per-Wh point yet at
      {selectedRung === null ? 'any rung' : `C=${selectedRung}`}: the trend needs a certified run whose
      energy window is fully sampled.
      {#each trend?.excluded ?? [] as e}
        {e.reason}.
      {/each}
    </p>
  {/if}

  {#if cost.refusals.length > 0}
    <p class="bd-footnote cost-refused">
      {#each cost.refusals as r}refused: {r}. {/each}
    </p>
  {/if}

  <div class="cost-disclosure">
    {#each DISCLOSURES as d}
      <p><strong>{d.head}</strong> {d.body}</p>
    {/each}
    {#if latest}
      <p class="cost-provenance">
        Latest {subject.gate} run: {fmtDate(latest.recorded_at)} · <code>{latest.git_sha}</code> ·
        <code>{instrumentLabel(latest)}</code>.
      </p>
    {/if}
  </div>
</div>
