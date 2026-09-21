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
  import CostSavings from './CostSavings.svelte';
  import LazyIcon from './LazyIcon.svelte';
  import GateChart from './GateChart.svelte';
  import LADDERS from '$lib/ladders.generated.json';
  import { colorFor, fmtDate } from '$lib/gates.js';
  import { instrumentLabel } from '$lib/concurrency-comparison.js';
  import {
    DEFAULT_PUE,
    DEFAULT_USD_PER_KWH,
    DISCLOSURES,
    PRICE_STORAGE_KEY,
    PUE_HIGH_STORAGE_KEY,
    PUE_LOW_STORAGE_KEY,
    PUE_MAX,
    PUE_MIN,
    PUE_TYPICAL,
    baselineSnapshots,
    costLadder,
    costPerMillion,
    costTrend,
    emptyStateOf,
    extremeRungs,
    fmtUsd,
    pueRange,
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

  // -- facility overhead: the reader's SECOND input, and it starts at none ---
  //
  // Two boxes, not one, because a building is a range. Both default to
  // DEFAULT_PUE (1.0, the identity), so a reader who ignores this control sees
  // exactly the chart that existed before it — the marks do not move and no
  // band is drawn. `pueRange` orders the pair and reports which box it had to
  // refuse, so an unusable value is SAID rather than silently defaulted.
  let pueLowInput = $state(DEFAULT_PUE);
  let pueHighInput = $state(DEFAULT_PUE);
  const pue = $derived(pueRange(pueLowInput, pueHighInput));
  const pueTypical = () => {
    pueLowInput = PUE_TYPICAL.low;
    pueHighInput = PUE_TYPICAL.high;
  };
  const pueNone = () => {
    pueLowInput = DEFAULT_PUE;
    pueHighInput = DEFAULT_PUE;
  };
  // Same storage contract as the price: per viewer, per origin, every access
  // guarded, and never in the URL — a deep link must not carry someone else's
  // facility any more than it carries their tariff.
  $effect(() => {
    for (const [key, set] of [
      [PUE_LOW_STORAGE_KEY, (v) => (pueLowInput = v)],
      [PUE_HIGH_STORAGE_KEY, (v) => (pueHighInput = v)]
    ]) {
      try {
        const raw = localStorage.getItem(key);
        const v = Number(raw);
        if (raw !== null && Number.isFinite(v) && v >= PUE_MIN && v <= PUE_MAX) set(v);
      } catch {
        /* no stored PUE: 1.0 stands, and 1.0 applies nothing */
      }
    }
  });
  $effect(() => {
    if (!pue.lowOk || !pue.highOk) return;
    try {
      localStorage.setItem(PUE_LOW_STORAGE_KEY, String(pueLowInput));
      localStorage.setItem(PUE_HIGH_STORAGE_KEY, String(pueHighInput));
    } catch {
      /* storage refused: the overhead still applies to this view */
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
  // Tiles carry the LOW edge — the same number the marks sit on, and at the
  // default PUE the rail reading itself. A tile is not the place to show a
  // range; the chart is, and the tile must agree with the mark beside it.
  const usdAt = (e) =>
    fmtUsd(
      costPerMillion(
        (showAboveIdle && e.aboveIdleJ !== null ? e.aboveIdleJ : e.energyJ) / e.tokens,
        usdPerKwh,
        pue.low
      )
    );
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
  // The newest generation is the live line; older ones are dashed history and
  // carry their own (unused) envelopes.
  const newest = $derived(trend?.generations?.[trend.generations.length - 1] ?? null);
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

  <!-- The two inputs on this page, and both are labelled as inputs. -->
  <div class="cost-price">
    <div class="cost-inputs">
      <label class="cost-price-field">
        <span class="cost-price-label">electricity price</span>
        <span class="cost-price-box">
          <span aria-hidden="true">$</span>
          <input type="number" min="0" step="0.01" inputmode="decimal"
            bind:value={priceInput} aria-label="Electricity price in dollars per kilowatt-hour" />
          <span aria-hidden="true">/ kWh</span>
        </span>
      </label>

      <!-- Facility overhead. Two boxes because a building is a range, and the
           default is the identity so the control applies nothing until it is
           touched. -->
      <div class="cost-price-field" role="group" aria-labelledby="pue-label-{subject.id}">
        <span class="cost-price-label" id="pue-label-{subject.id}">facility overhead (PUE)</span>
        <span class="cost-price-box cost-pue-box">
          <input type="number" min={PUE_MIN} max={PUE_MAX} step="0.05" inputmode="decimal"
            bind:value={pueLowInput} aria-label="Lowest power usage effectiveness to draw" />
          <span aria-hidden="true">–</span>
          <input type="number" min={PUE_MIN} max={PUE_MAX} step="0.05" inputmode="decimal"
            bind:value={pueHighInput} aria-label="Highest power usage effectiveness to draw" />
          <span aria-hidden="true">×</span>
        </span>
        <span class="cost-pue-presets">
          <button type="button" class="cmp-chip" aria-pressed={!pue.band && pue.low === DEFAULT_PUE}
            onclick={pueNone}>none ({DEFAULT_PUE}×)</button>
          <button type="button" class="cmp-chip"
            aria-pressed={pue.low === PUE_TYPICAL.low && pue.high === PUE_TYPICAL.high}
            onclick={pueTypical}>datacentre ({PUE_TYPICAL.low}–{PUE_TYPICAL.high}×)</button>
        </span>
      </div>
    </div>

    <p class="cost-price-note">
      <strong>Your input, not a measurement.</strong> The default {DEFAULT_USD_PER_KWH} $/kWh is a round
      retail-commercial figure — change it to your own tariff. It is kept in this browser and is never
      written into the link.
      {#if !priceValid}
        <span class="cost-warn">That is not a price; the chart is drawn at {DEFAULT_USD_PER_KWH} $/kWh until it is one.</span>
      {/if}
    </p>
    <p class="cost-price-note">
      <strong>PUE starts at {DEFAULT_PUE}×, which applies nothing.</strong> Set a low and a high — a
      datacentre is typically {PUE_TYPICAL.low}–{PUE_TYPICAL.high}× — and the chart shades the band
      between them for <em>both</em> engines. Because it multiplies both equally it moves every
      absolute figure and changes no ratio, no verdict and no winner.
      {#if pue.band}
        Drawing {pue.low}–{pue.high}×; the line is the rail, the band is the building.
      {/if}
      {#if !pue.lowOk || !pue.highOk}
        <span class="cost-warn"
          >{!pue.lowOk && !pue.highOk ? 'Neither box is' : !pue.lowOk ? 'The low box is not' : 'The high box is not'}
          a PUE between {PUE_MIN} and {PUE_MAX}; {!pue.lowOk && !pue.highOk ? 'both are' : 'it is'} drawn at
          {DEFAULT_PUE}× until {!pue.lowOk && !pue.highOk ? 'they are' : 'it is'}.</span>
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

    <CostLadderChart {subject} {cost} {usdPerKwh} {pue} rungs={rungsMeasured} title={chartTitle}
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

    <!-- What the per-token figure above is worth cumulatively. It follows
         `selectedRung` and `showAboveIdle` so it can never disagree with the
         chart it sits under. -->
    <CostSavings {cost} rung={selectedRung} {usdPerKwh} {pue} aboveIdle={showAboveIdle} />
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
    <h3 class="cost-trend-h"><LazyIcon name="gauge" size={16} /> {trend.title}</h3>
    <GateChart records={trend.records} panel={{ title: trend.title, unit: trend.unit, metrics: trend.metrics }}
      {onselect} />
    <p class="cmp-caption">
      One point is one certified run: <code>tokens ÷ joules × 3600</code> for that rung's batch. This
      line shows the CURRENT cost instrument only — {`${trend.runs} ${trend.runs === 1 ? 'run' : 'runs'}`}.
      <!-- ★ THE CAPTION MOVED WITH THE CODE. It used to say earlier instruments
           "are drawn as separate lines and are never joined", which stopped
           being true the moment they stopped being drawn. A caption describing
           behaviour the component no longer has is a harder bug to find than a
           wrong number, because nothing fails. -->
      {#if trend.superseded}
        {trend.superseded.runs} earlier {trend.superseded.runs === 1 ? 'run is' : 'runs are'} not
        drawn: measured on a different instrument ({trend.superseded.differs}), so their tok/Wh is a
        different measurement rather than an earlier value of this one. They stay in
        <code>.benchmarks/</code> as certification evidence.
      {/if}
      <!-- ★ THE VERDICT IS CARRIED IN WORDS, UNCONDITIONALLY. GateChart
           suppresses a spread bar shorter than 3 px, so at the narrow rungs the
           bar may not be visible at all — and a chart that looks like a clean
           fall while the sentence that would contradict it is conditional on
           pixel height is exactly the misreading this exists to prevent. -->
      {#if newest?.spread}
        The bar on each point is the measured run-to-run spread of this rung on this instrument:
        across the {newest.spread.n} certified runs that share it, C={selectedRung} throughput spanned
        {newest.spread.tMin.toFixed(2)}–{newest.spread.tMax.toFixed(2)} tok/s. It is measured on
        throughput because only {newest.spread.powerMeasured} of those runs carry joules, and widened
        by the spread of the rail across those — so it is a <strong>floor</strong> on this line's
        spread, not a confidence interval.
        {newest.verdict?.state === 'separated'
          ? `The latest step is larger than that spread, so it is a real ${newest.verdict.direction === 'down' ? 'fall' : 'rise'}.`
          : newest.verdict?.state === 'overlap'
            ? 'Consecutive points here differ by less than that spread, so this line does not yet show a token getting cheaper.'
            : 'There is only one point on this instrument so far, so there is no step to judge.'}
      {:else}
        The run-to-run spread of C={selectedRung} on this instrument is not yet measured
        ({trend.runs} {trend.runs === 1 ? 'run' : 'runs'}), so this line cannot be told from noise.
      {/if}
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
