<!--
  The payback model, interactive. The three scenarios from src/lib/economics.js,
  every input labeled with its evidence class. The measured throughput comes
  from the ladder at render time; everything else is the visitor's to change.

  The third tab counts in tokens per joule and draws that on every rung of the
  ladder. The ladder publishes no power yet, so the draw is a USER field. When a
  rung carries its recorded draw, efficiencyLadder() uses it, and the labels and
  the caption here turn to MEASURED with no edit to this file.
-->
<script>
  import { fleetModel, apiModel, energyModel, energyInputsFrom, efficiencyLadder, axisFor, paybackLabel, usd, num, FLEET_DEFAULTS, API_DEFAULTS } from '$lib/economics.js';
  import { paybackCopy } from '$lib/content/pricing.js';
  import { live, ladderData } from '$lib/content/live.js';
  import { moveTab } from '$lib/tablist.js';

  let tab = $state('fleet');
  let f = $state({ ...FLEET_DEFAULTS });
  let a = $state({ ...API_DEFAULTS, boxTokensPerSecond: Number(live.atlasTop) });
  let e = $state(energyInputsFrom(ladderData.rows));
  const fr = $derived(fleetModel(f));
  const ar = $derived(apiModel(a));
  const er = $derived(energyModel(e));
  const measuredRatio = Number(live.ratioPlain);
  const box = ladderData.box.gpu.split(',')[0];

  // The graph: tokens per joule on each rung, both engines, on one scale. Its
  // text is sized in the drawing's own units, so a phone gets a narrower drawing
  // instead of the wide one shrunk until the labels cannot be read.
  const curve = $derived(efficiencyLadder(ladderData.rows, { watts: Number(e.watts), baselineWatts: Number(e.baselineWatts) }));
  let chartWidth = $state(0);
  const W = $derived(chartWidth > 0 && chartWidth < 460 ? 340 : 560);
  const H = 230, PL = 40, PR = 22, PT = 14, PB = 30;
  const maxC = $derived(Math.max(...curve.points.map((p) => p.c), 2));
  const axis = $derived(axisFor(Math.max(0, ...curve.points.map((p) => Math.max(p.avarok, p.baseline)))));
  const x = (c) => PL + (Math.log2(c) / Math.log2(maxC)) * (W - PL - PR);
  const y = (v) => PT + (1 - v / axis.top) * (H - PT - PB);
  const path = (key) => curve.points.map((p, i) => `${i ? 'L' : 'M'}${x(p.c).toFixed(1)} ${y(p[key]).toFixed(1)}`).join(' ');
  const last = $derived(curve.points.at(-1));
  // The narrow drawing has no room for the unit in its key. The title beside it names the unit.
  const keyUnit = $derived(W === 340 || !last ? '' : ` tok/J at C=${last.c}`);

  const TABS = ['fleet', 'api', 'energy'];
  function onKey(ev, i) {
    const next = moveTab(ev.key, i, TABS.length);
    if (next === null) return;
    ev.preventDefault();
    tab = TABS[next];
    ev.currentTarget.parentElement.querySelectorAll('[role="tab"]')[next].focus();
  }
</script>

<div class="av-calc" id="payback">
  <div class="av-tabs" role="tablist" aria-label="Payback scenarios">
    {#each TABS as t, i}
      <!-- Only the chosen panel is in the document, so only its tab may point at one. -->
      <button type="button" role="tab" class="av-tab" id={`payback-tab-${t}`} aria-controls={tab === t ? `payback-panel-${t}` : undefined} aria-selected={tab === t} tabindex={tab === t ? 0 : -1} onclick={() => (tab = t)} onkeydown={(ev) => onKey(ev, i)}>{paybackCopy[t].title}</button>
    {/each}
  </div>

  {#if tab === 'fleet'}
    <div class="av-calc-grid" role="tabpanel" id="payback-panel-fleet" aria-labelledby="payback-tab-fleet">
      <div class="av-calc-inputs">
        <p class="av-body">{paybackCopy.fleet.body}</p>
        <label class="av-field"><span>GPUs in the fleet <span class="av-evidence">USER</span></span><input type="number" min="1" step="1" bind:value={f.gpus} /></label>
        <label class="av-field"><span>Cost per GPU per year, amortized purchase or rental <span class="av-evidence">USER</span></span><input type="number" min="0" step="500" bind:value={f.gpuCostPerYear} /></label>
        <label class="av-field">
          <span>Throughput uplift on your workload <span class="av-evidence is-measured">MEASURED {live.ratio} at C={live.c} on GB10</span></span>
          <span class="av-range"><input type="range" min="1" max={Math.max(1.5, measuredRatio).toFixed(2)} step="0.01" bind:value={f.uplift} /><output>{Number(f.uplift).toFixed(2)}×</output></span>
        </label>
        <label class="av-field"><span>License per GPU per year <span class="av-evidence is-proposed">PROPOSED</span></span><input type="number" min="0" step="100" bind:value={f.licensePerGpuYear} /></label>
        <div class="av-field-row">
          <label class="av-field"><span>Watts per GPU <span class="av-evidence">USER</span></span><input type="number" min="0" step="10" bind:value={f.wattsPerGpu} /></label>
          <label class="av-field"><span>PUE <span class="av-evidence">USER</span></span><input type="number" min="1" step="0.05" bind:value={f.pue} /></label>
        </div>
        <div class="av-field-row">
          <label class="av-field"><span>Dollars per kWh <span class="av-evidence">USER</span></span><input type="number" min="0" step="0.01" bind:value={f.usdPerKwh} /></label>
          <label class="av-field"><span>Per GPU software replaced, per year <span class="av-evidence">USER</span></span><input type="number" min="0" step="100" bind:value={f.replacedSoftwarePerGpuYear} /></label>
        </div>
        <p class="av-small">{paybackCopy.fleet.note}</p>
      </div>
      <div class="av-calc-out">
        <div class="av-calc-hero">
          <span class="av-tile-label">Payback period</span>
          <span class="av-num">{paybackLabel(fr.paybackMonths)}</span>
          <span class="av-small">then {usd(fr.net)} a year is upside</span>
        </div>
        <dl class="av-calc-rows">
          <div><dt>GPUs freed by the uplift</dt><dd>{num(fr.freedGpus, 1)}</dd></div>
          <div><dt>Deferred purchase or rental</dt><dd>{usd(fr.capacityValue)}</dd></div>
          <div><dt>Power no longer burned</dt><dd>{usd(fr.powerValue)}</dd></div>
          {#if fr.replaced}<div><dt>Software replaced</dt><dd>{usd(fr.replaced)}</dd></div>{/if}
          <div><dt>Gross savings per year</dt><dd>{usd(fr.grossSavings)}</dd></div>
          <div><dt>Avarok license per year</dt><dd>{usd(fr.license)}</dd></div>
          <div class="is-total"><dt>Net per year</dt><dd>{usd(fr.net)}</dd></div>
          <div><dt>Three year net</dt><dd>{usd(fr.threeYearNet)}</dd></div>
        </dl>
      </div>
    </div>
  {:else if tab === 'api'}
    <div class="av-calc-grid" role="tabpanel" id="payback-panel-api" aria-labelledby="payback-tab-api">
      <div class="av-calc-inputs">
        <p class="av-body">{paybackCopy.api.body}</p>
        <label class="av-field"><span>Current monthly API bill <span class="av-evidence">USER</span></span><input type="number" min="0" step="500" bind:value={a.monthlySpend} /></label>
        <label class="av-field"><span>Blended price per million tokens, in plus out <span class="av-evidence">USER</span></span><input type="number" min="0.01" step="0.05" bind:value={a.usdPerMillionTokens} /></label>
        <label class="av-field"><span>Tokens per second per box <span class="av-evidence is-measured">MEASURED at C={live.c}, {ladderData.box.gpu.split(',')[0]}</span></span><input type="number" min="1" step="1" bind:value={a.boxTokensPerSecond} /></label>
        <div class="av-field-row">
          <label class="av-field"><span>Box cost <span class="av-evidence">USER</span></span><input type="number" min="0" step="100" bind:value={a.boxCapex} /></label>
          <label class="av-field"><span>Amortize over, months <span class="av-evidence">USER</span></span><input type="number" min="1" step="1" bind:value={a.amortMonths} /></label>
        </div>
        <div class="av-field-row">
          <label class="av-field"><span>Utilization <span class="av-evidence">USER</span></span><span class="av-range"><input type="range" min="0.1" max="1" step="0.05" bind:value={a.utilization} /><output>{Math.round(a.utilization * 100)}%</output></span></label>
          <label class="av-field"><span>License per box per month <span class="av-evidence is-proposed">PROPOSED</span></span><input type="number" min="0" step="5" bind:value={a.licensePerBoxMonth} /></label>
        </div>
        <div class="av-field-row">
          <label class="av-field"><span>Watts per box <span class="av-evidence">USER</span></span><input type="number" min="0" step="10" bind:value={a.wattsPerBox} /></label>
          <label class="av-field"><span>Dollars per kWh <span class="av-evidence">USER</span></span><input type="number" min="0" step="0.01" bind:value={a.usdPerKwh} /></label>
        </div>
        <p class="av-small">{paybackCopy.api.note}</p>
      </div>
      <div class="av-calc-out">
        <div class="av-calc-hero">
          <span class="av-tile-label">New bill against the old</span>
          <span class="av-num">{ar.savingsPct > 0 ? `${Math.round(100 - ar.savingsPct)}%` : 'more'}</span>
          <span class="av-small">{ar.savingsPct > 0 ? `${Math.round(ar.savingsPct)}% lower, payback in ${paybackLabel(ar.paybackMonths)}` : 'owning does not beat renting at these inputs'}</span>
        </div>
        <dl class="av-calc-rows">
          <div><dt>Tokens per month on the bill</dt><dd>{num(ar.tokensPerMonth / 1e6)} M</dd></div>
          <div><dt>Boxes needed at measured throughput</dt><dd>{ar.boxes}</dd></div>
          <div><dt>Amortized hardware per month</dt><dd>{usd(ar.capexPerMonth)}</dd></div>
          <div><dt>Power per month</dt><dd>{usd(ar.powerPerMonth)}</dd></div>
          <div><dt>License per month</dt><dd>{usd(ar.licensePerMonth)}</dd></div>
          <div class="is-total"><dt>New monthly cost</dt><dd>{usd(ar.newMonthly)}</dd></div>
          <div><dt>Cost per million tokens</dt><dd>{usd(ar.costPerMillion, 3)}</dd></div>
          <div><dt>Monthly savings</dt><dd>{usd(ar.monthlySavings)}</dd></div>
        </dl>
      </div>
    </div>
  {:else}
    <div class="av-calc-grid" role="tabpanel" id="payback-panel-energy" aria-labelledby="payback-tab-energy">
      <div class="av-calc-inputs">
        <p class="av-body">{paybackCopy.energy.body}</p>
        <div class="av-field-row">
          <label class="av-field"><span>Avarok, tokens per second <span class="av-evidence is-measured">MEASURED C={maxC}</span></span><input type="number" min="1" step="1" bind:value={e.tokensPerSecond} /></label>
          <label class="av-field"><span>{live.baselineLabel}, tokens per second <span class="av-evidence is-measured">MEASURED C={maxC}</span></span><input type="number" min="1" step="1" bind:value={e.baselineTokensPerSecond} /></label>
        </div>
        <div class="av-field-row">
          <label class="av-field"><span>Avarok, watts under load <span class="av-evidence" class:is-measured={curve.measured}>{curve.measured ? 'MEASURED' : 'USER'}</span></span><input type="number" min="1" step="5" bind:value={e.watts} /></label>
          <label class="av-field"><span>{live.baselineLabel}, watts under load <span class="av-evidence" class:is-measured={curve.measured}>{curve.measured ? 'MEASURED' : 'USER'}</span></span><input type="number" min="1" step="5" bind:value={e.baselineWatts} /></label>
        </div>
        <label class="av-field"><span>Tokens served per month, millions <span class="av-evidence">USER</span></span><input type="number" min="0" step="100" bind:value={e.millionTokensPerMonth} /></label>
        <div class="av-field-row">
          <label class="av-field"><span>PUE <span class="av-evidence">USER</span></span><input type="number" min="1" step="0.05" bind:value={e.pue} /></label>
          <label class="av-field"><span>Dollars per kWh <span class="av-evidence">USER</span></span><input type="number" min="0" step="0.01" bind:value={e.usdPerKwh} /></label>
        </div>
        <p class="av-small">{paybackCopy.energy.note}</p>
      </div>
      <div class="av-calc-out">
        <div class="av-calc-hero">
          <span class="av-tile-label">Tokens per watt</span>
          <span class="av-num">{num(er.tokensPerJoule, 2)}<small> tok/J</small></span>
          <span class="av-small">{#if er.ratio === null}no baseline to compare{:else if er.energyCutPct > 0}against {num(er.baselineTokensPerJoule, 2)}, so {num(er.energyCutPct, 1)}% less energy for the same token{:else if er.energyCutPct < 0}against {num(er.baselineTokensPerJoule, 2)}, so the baseline is the more efficient at these inputs{:else}against {num(er.baselineTokensPerJoule, 2)}, no difference at these inputs{/if}</span>
        </div>
        <p class="av-small">{paybackCopy.energy.unit}</p>
        <table class="av-calc-table">
          <thead>
            <tr><td></td><th scope="col">Avarok</th><th scope="col">{live.baselineLabel}</th></tr>
          </thead>
          <tbody>
            <tr><th scope="row">Tokens per joule</th><td>{num(er.tokensPerJoule, 2)}</td><td>{num(er.baselineTokensPerJoule, 2)}</td></tr>
            <tr><th scope="row">Joules per token</th><td>{num(er.joulesPerToken, 3)}</td><td>{num(er.baselineJoulesPerToken, 3)}</td></tr>
            <tr><th scope="row">kWh per million tokens</th><td>{num(er.kwhPerMillion, 3)}</td><td>{num(er.baselineKwhPerMillion, 3)}</td></tr>
            <tr><th scope="row">Energy cost per million tokens</th><td>{usd(er.usdPerMillion, 4)}</td><td>{usd(er.baselineUsdPerMillion, 4)}</td></tr>
            <tr><th scope="row">kWh per month at your volume</th><td>{num(er.kwhPerMonth)}</td><td>{num(er.baselineKwhPerMonth)}</td></tr>
          </tbody>
        </table>
        <dl class="av-calc-rows">
          <div class="is-total"><dt>kWh not burned per month</dt><dd>{num(er.kwhSavedPerMonth)}</dd></div>
          <div><dt>Energy bill saved per year</dt><dd>{usd(er.usdSavedPerYear)}</dd></div>
        </dl>
      </div>
      <figure class="av-calc-chart">
        <div bind:clientWidth={chartWidth}>
          <svg viewBox={`0 0 ${W} ${H}`} class="av-effchart" role="img" aria-label={`Tokens per joule from C=1 to C=${maxC}. At C=${last?.c}, Avarok ${num(last?.avarok ?? 0, 2)} and ${live.baselineLabel} ${num(last?.baseline ?? 0, 2)}.`}>
            {#each axis.ticks as t}
              <line x1={PL} x2={W - PR} y1={y(t)} y2={y(t)} class="grid" />
              <text x={PL - 6} y={y(t) + 4} text-anchor="end" class="axis">{t.toFixed(axis.digits)}</text>
            {/each}
            {#each curve.points as p}
              <text x={x(p.c)} y={H - 8} text-anchor="middle" class="axis">C={p.c}</text>
            {/each}
            <path d={path('baseline')} class="base" />
            <path d={path('avarok')} class="atlas" />
            {#each curve.points as p}
              <circle cx={x(p.c)} cy={y(p.avarok)} r="4" class="atlas-dot" />
            {/each}
            <!-- The key sits top left, the one corner two rising lines never reach. -->
            {#if last}
              <line x1={PL + 10} x2={PL + 32} y1={PT + 10} y2={PT + 10} class="atlas" />
              <circle cx={PL + 21} cy={PT + 10} r="4" class="atlas-dot" />
              <text x={PL + 40} y={PT + 14} class="lab atlas-lab">Avarok {num(last.avarok, 2)}{keyUnit}</text>
              <line x1={PL + 10} x2={PL + 32} y1={PT + 30} y2={PT + 30} class="base" />
              <text x={PL + 40} y={PT + 34} class="lab base-lab">{live.baselineLabel} {num(last.baseline, 2)}{keyUnit}</text>
            {/if}
          </svg>
        </div>
        <figcaption>
          <span class="av-kicker"><span class="av-dot"></span> {paybackCopy.energy.chartTitle}</span>
          <span class="av-evidence" class:is-measured={curve.measured}>{curve.measured ? `MEASURED throughput and draw, ${box}` : `MEASURED throughput, ${box}. USER draw`}</span>
          <span class="av-small">{curve.measured ? paybackCopy.energy.chartMeasured : paybackCopy.energy.chart}</span>
        </figcaption>
      </figure>
    </div>
  {/if}
  <p class="av-small av-calc-foot">{paybackCopy.disclaimer} Evidence classes: <b>MEASURED</b> {paybackCopy.classes.MEASURED.toLowerCase()}. <b>PROPOSED</b> {paybackCopy.classes.PROPOSED.toLowerCase()}. <b>USER</b> {paybackCopy.classes.USER.toLowerCase()}.</p>
</div>

<style>
  .av-calc { background: var(--card); border: 1px solid var(--border); border-radius: var(--av-radius); padding: 1.6rem; box-shadow: var(--av-shadow); }
  .av-calc-grid { display: grid; grid-template-columns: minmax(0, 1.1fr) minmax(0, 0.9fr); gap: 2rem; }
  .av-calc-inputs { display: grid; gap: 0.9rem; align-content: start; }
  .av-calc-inputs .av-field > span:first-child { font-size: 0.84rem; font-weight: 600; color: var(--t1); }
  .av-calc-out { background: var(--bg2); border: 1px solid var(--border); border-radius: var(--av-radius-sm); padding: 1.4rem; align-self: start; position: sticky; top: 90px; }
  .av-calc-hero { display: grid; gap: 0.35rem; padding-bottom: 1.1rem; border-bottom: 1px dashed var(--border-strong); margin-bottom: 1rem; }
  .av-calc-hero .av-num { font-size: 2.6rem; }
  .av-calc-rows { margin: 0; display: grid; gap: 0.45rem; }
  .av-calc-rows > div { display: flex; justify-content: space-between; gap: 1rem; font-size: 0.9rem; }
  .av-calc-rows dt { color: var(--t2); }
  .av-calc-rows dd { margin: 0; font-family: var(--font-mono); color: var(--t1); font-variant-numeric: tabular-nums; }
  .av-calc-rows .is-total { padding-top: 0.5rem; border-top: 1px solid var(--border-strong); font-weight: 700; }
  /* A chip in a half width field moves to the next line whole, never split across two. */
  .av-field-row .av-evidence { white-space: nowrap; }
  .av-calc-hero .av-num small { margin-left: 0.4rem; font-size: 1rem; font-weight: 500; color: var(--t2); letter-spacing: 0; }
  /* Two engines, side by side: a table, so a screen reader names the column. */
  .av-calc-table { width: 100%; border-collapse: collapse; margin: 0.9rem 0 0.7rem; font-size: 0.9rem; }
  .av-calc-table th, .av-calc-table td { padding: 0.28rem 0; text-align: right; vertical-align: baseline; }
  .av-calc-table th[scope='row'] { text-align: left; font-weight: 400; color: var(--t2); padding-right: 0.75rem; }
  .av-calc-table th[scope='col'] { font-family: var(--font-mono); font-size: 0.7rem; font-weight: 600; letter-spacing: 0.04em; text-transform: uppercase; color: var(--t2); padding-left: 0.75rem; white-space: nowrap; }
  .av-calc-table td { font-family: var(--font-mono); color: var(--t1); font-variant-numeric: tabular-nums; padding-left: 0.75rem; white-space: nowrap; }
  .av-calc-table thead tr { border-bottom: 1px solid var(--border); }
  /* The graph sits under both columns, the drawing beside what it says. */
  .av-calc-chart { grid-column: 1 / -1; margin: 0; display: grid; grid-template-columns: minmax(0, 1.1fr) minmax(0, 0.9fr); gap: 1rem 2rem; align-items: center; padding-top: 1.4rem; border-top: 1px dashed var(--border-strong); }
  .av-calc-chart figcaption { display: grid; gap: 0.6rem; justify-items: start; }
  .av-effchart { width: 100%; height: auto; }
  .av-effchart .grid { stroke: var(--border); stroke-width: 1; }
  .av-effchart .axis { fill: var(--t3); font-family: var(--font-mono); font-size: 10px; }
  .av-effchart .base { fill: none; stroke: var(--t3); stroke-width: 2; stroke-dasharray: 5 4; }
  .av-effchart .atlas { fill: none; stroke: var(--accent); stroke-width: 3; }
  .av-effchart .atlas-dot { fill: var(--accent); }
  .av-effchart .lab { font-family: var(--font-mono); font-size: 11px; font-weight: 700; }
  .av-effchart .atlas-lab { fill: var(--accent); }
  .av-effchart .base-lab { fill: var(--t3); }
  .av-calc-foot { margin-top: 1.25rem; }
  .av-calc-foot b { color: var(--t2); font-weight: 700; }
  @media (max-width: 900px) { .av-calc-grid, .av-calc-chart { grid-template-columns: minmax(0, 1fr); } .av-calc-out { position: static; } }
  @media (max-width: 720px) { .av-calc { padding: 1rem; } }
</style>
