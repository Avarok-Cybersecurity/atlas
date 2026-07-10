<script>
  // The signature element. Renders the release-gate state as a printed receipt.
  // Data is generated from tests/baselines/ via scripts/gen-benchmarks.mjs.
  // status 'pending' (no committed baselines yet) -> honest "submission" panel;
  // status 'verified' -> ✓ rows. Same UI, driven entirely by the data file.
  import bench from '$lib/benchmarks.generated.json';
  import { verified } from '$lib/data.js';

  let { compact = false } = $props();
  const rows = bench.entries ?? [];
  const isVerified = bench.status === 'verified' && rows.length > 0;
</script>

<div class="receipt receipt-print" role="figure" aria-label="Atlas release-gate receipt">
  <div class="receipt-body">
    <div class="receipt-head">
      <span class="receipt-title">serve matrix</span>
      <span class="receipt-hw">DGX Spark · GB10</span>
    </div>

    {#if isVerified}
      {#each rows as r}
        <div class="receipt-row">
          <span class="ok">✓</span>
          <span class="name">{r.label}</span>
          <span class="val">{r.quant} · {r.tps} tok/s</span>
        </div>
      {/each}
    {:else}
      <div class="receipt-pending">
        <span class="tag">▷ {verified.pendingHeadline}</span>
        {#if !compact}<p>{verified.pendingBody}</p>{/if}
      </div>
      <div class="receipt-row" style="margin-top:0.6rem">
        <span class="ok">✓</span><span class="name">liveness + coherence</span><span class="val">enforced</span>
      </div>
      <div class="receipt-row">
        <span class="ok" style="color:var(--amber)">◷</span><span class="name">throughput baselines</span><span class="val">awaiting submission</span>
      </div>
    {/if}

    <div class="receipt-foot">
      <span>atlas {bench.generated_sha}</span>
      <span>{bench.generated_date}</span>
    </div>
  </div>
</div>
