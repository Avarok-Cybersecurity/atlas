<!--
  The payback model, interactive. Both scenarios from src/lib/economics.js,
  every input labeled with its evidence class. The measured throughput comes
  from the ladder at render time; everything else is the visitor's to change.
-->
<script>
  import { fleetModel, apiModel, paybackLabel, usd, num, FLEET_DEFAULTS, API_DEFAULTS } from '$lib/economics.js';
  import { paybackCopy } from '$lib/content/pricing.js';
  import { live, ladderData } from '$lib/content/live.js';

  let tab = $state('fleet');
  let f = $state({ ...FLEET_DEFAULTS });
  let a = $state({ ...API_DEFAULTS, boxTokensPerSecond: Number(live.atlasTop) });
  const fr = $derived(fleetModel(f));
  const ar = $derived(apiModel(a));
  const measuredRatio = Number(live.ratioPlain);
</script>

<div class="av-calc" id="payback">
  <div class="av-tabs" role="tablist" aria-label="Payback scenarios">
    <button type="button" role="tab" class="av-tab" aria-selected={tab === 'fleet'} onclick={() => (tab = 'fleet')}>{paybackCopy.fleet.title}</button>
    <button type="button" role="tab" class="av-tab" aria-selected={tab === 'api'} onclick={() => (tab = 'api')}>{paybackCopy.api.title}</button>
  </div>

  {#if tab === 'fleet'}
    <div class="av-calc-grid" role="tabpanel">
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
  {:else}
    <div class="av-calc-grid" role="tabpanel">
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
  .av-calc-foot { margin-top: 1.25rem; }
  .av-calc-foot b { color: var(--t2); font-weight: 700; }
  @media (max-width: 900px) { .av-calc-grid { grid-template-columns: minmax(0, 1fr); } .av-calc-out { position: static; } }
  @media (max-width: 720px) { .av-calc { padding: 1rem; } }
</style>
