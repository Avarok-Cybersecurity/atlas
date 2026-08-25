<script>
  // Ledger entry 02. This is the second screen on purpose: the concurrency
  // ladder is the strongest verified artifact the project owns, and it used to
  // be hidden behind a click on the hero receipt. The methodology that makes
  // the chart credible sits directly under it rather than in a modal.
  import {
    verified, mlperfCopy, mlperfTrademark, mlcommons, verifiedAnchor, gateSrcUrl, recipesUrl
  } from '$lib/data.js';
  import bench from '$lib/benchmarks.generated.json';
  import ladder from '$lib/ladder.generated.json';
  import mlperf from '$lib/mlperf.json';
  import Receipt from './Receipt.svelte';
  import SectionHead from './SectionHead.svelte';
  import ConcurrencyLadder from './ConcurrencyLadder.svelte';

  const mlperfLine = mlperfCopy[mlperf.status] ?? mlperfCopy.preparing;
  const stamp = `atlas ${bench.generated_sha} · ${bench.generated_date}`;

  let copied = $state(false);
  async function copyRepro() {
    try {
      await navigator.clipboard.writeText(bench.repro_cmd);
      copied = true;
      setTimeout(() => (copied = false), 1600);
    } catch {}
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
          <code>{bench.repro_cmd}</code>
          <button type="button" class="copy-btn" onclick={copyRepro} aria-label="Copy reproduce command">{copied ? 'Copied' : 'Copy'}</button>
        </div>
        <p class="mlperf-note" style="font-weight:650;color:var(--t1)">{verified.challengeLine}</p>
        <p class="mlperf-note" style="font-size:0.84rem">Every model card comes from a recipe in <a class="link" href={recipesUrl} target="_blank" rel="noopener">atlas-recipes</a>.</p>
      </div>

      <Receipt source="gate" />
    </div>
  </div>
</section>
