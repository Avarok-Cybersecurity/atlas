<script>
  // Ledger entry 02. This is the second screen on purpose: the concurrency
  // ladder is the strongest verified artifact the project owns, and it used to
  // be hidden behind a click on the hero receipt. The methodology that makes
  // the chart credible sits directly under it rather than in a modal.
  import { copyLabel, copyOrSelect } from '$lib/clipboard.js';
  import {
    verified, mlperfCopy, mlperfTrademark, mlcommons, verifiedAnchor, gateSrcUrl, recipesUrl
  } from '$lib/data.js';
  import bench from '$lib/benchmarks.generated.json';
  import ladder from '$lib/ladder.generated.json';
  import mlperf from '$lib/mlperf.json';
  import Receipt from './Receipt.svelte';
  import SectionHead from './SectionHead.svelte';
  import ConcurrencyLadder from './ConcurrencyLadder.svelte';
  import { headroom, signed } from '$lib/ladder.js';

  // Derived from the committed ladder, so regenerating it rewrites this
  // sentence rather than leaving prose asserting a gap that closed.
  const top = headroom(ladder.rows);

  const mlperfLine = mlperfCopy[mlperf.status] ?? mlperfCopy.preparing;
  const stamp = `atlas ${bench.generated_sha} · ${bench.generated_date}`;

  let copyState = $state('idle'); // idle | copied | manual | blocked
  let copyTimer;
  // The flash outlives the component on navigation without this.
  $effect(() => () => clearTimeout(copyTimer));
  let cmdEl = $state(null);
  async function copyRepro() {
    clearTimeout(copyTimer);
    // A reproduce command someone retypes by eye is a reproduction that does
    // not reproduce; a refusal has to say so.
    copyState = await copyOrSelect(bench.repro_cmd, cmdEl);
    copyTimer = setTimeout(() => (copyState = 'idle'), 2400);
  }
</script>

<section id="verified" class="sx-green">
  <div class="container">
    <SectionHead
      label={verified.label}
      title={verified.title}
      sub={verified.sub}
      prov={stamp}
      provUrl={ladder.results_doc_url}
    />

    <ConcurrencyLadder embedded />

    {#if top}
      <div class="scale-note">
        <h3>{verified.scale.title}</h3>
        <p>{verified.scale.lead}</p>
        <p class="scale-figure">
          From C={top.from} to C={top.to}, Atlas adds
          <strong class="scale-up">{signed(top.atlas)}</strong> throughput while
          {top.label} adds <strong class="scale-flat">{signed(top.baseline)}</strong>.
        </p>
        <p>{verified.scale.tail}</p>
      </div>
    {/if}

    <div class="verified-grid">
      <div>
        <div class="method-card">
          <h3>What the gate checks</h3>
          <p>{bench.methodology} <a class="link" href={verifiedAnchor} target="_blank" rel="noopener">What “verified” means</a> · <a class="link" href={gateSrcUrl} target="_blank" rel="noopener">gate_results.py</a></p>
        </div>

        <p class="mech-line">{verified.mechanism}</p>

        <p class="mlperf-note">{mlperfLine}</p>
        <p class="mlperf-note">{mlcommons.line} <a class="link" href={mlcommons.url} target="_blank" rel="noopener">{mlcommons.linkText}</a>.</p>
        <p class="trademark">{mlperfTrademark}</p>

        <div class="repro" aria-label="Reproduce command">
          <code bind:this={cmdEl}>{bench.repro_cmd}</code>
          <button type="button" class="copy-btn" onclick={copyRepro} aria-label="Copy reproduce command">{copyLabel(copyState)}</button>
        </div>
        <p class="mlperf-note" style="font-weight:650;color:var(--t1)">{verified.challengeLine}</p>
        <p class="mlperf-note" style="font-size:0.84rem">Every model card comes from a recipe in <a class="link" href={recipesUrl} target="_blank" rel="noopener">atlas-recipes</a>.</p>
      </div>

      <Receipt source="gate" />
    </div>
  </div>
</section>
