<script>
  // The signature element. Renders the release-gate state as a printed receipt.
  //
  // Three data-driven states, in order of preference, so the receipt always
  // prints something true rather than an empty form:
  //   1. serve-matrix baselines committed  -> ✓ rows per model
  //   2. no baselines, but a concurrency ladder exists -> ✓ rows per rung
  //   3. neither -> the honest "submitted, awaiting publication" panel
  // State 2 is the live case today: benchmarks.generated.json is still
  // `pending`, while ladder.generated.json is eight rungs and eight wins.
  import bench from '$lib/benchmarks.generated.json';
  import ladder from '$lib/ladder.generated.json';
  import { verified } from '$lib/data.js';

  // `source`: 'auto' prefers whichever data exists (serve matrix, then ladder).
  // 'gate' pins the receipt to the release-gate state — used where the ladder
  // is already rendered in full on the page and reprinting it would duplicate.
  let { compact = false, source = 'auto' } = $props();

  const rows = bench.entries ?? [];
  const isVerified = bench.status === 'verified' && rows.length > 0;

  const rungs = ladder.rows ?? [];
  const isLadder = source === 'auto' && !isVerified && rungs.length > 0;
  // Compact (hero) prints the four rungs that bracket the range — single
  // stream, small batch, mid, and the widest published rung — because eight
  // rows would push the receipt past the fold on a laptop.
  const COMPACT_RUNGS = [1, 8, 32, 128];
  const shown = compact ? rungs.filter((r) => COMPACT_RUNGS.includes(r.c)) : rungs;
  const bestLabel = (r) =>
    r.baselines.find((b) => b.id === r.best_baseline_id)?.label ?? 'baseline';
</script>

<div class="receipt receipt-print" role="figure" aria-label="Atlas release-gate receipt">
  <div class="receipt-body">
    <div class="receipt-head">
      <span class="receipt-title">{isLadder ? 'concurrency ladder' : 'serve matrix'}</span>
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
    {:else if isLadder}
      <div class="receipt-sub">
        {ladder.workload.checkpoint} vs {bestLabel(rungs[0])}
      </div>
      {#each shown as r}
        <div class="receipt-row">
          <span class="ok">{r.wins ? '✓' : '×'}</span>
          <span class="name">C={r.c}</span>
          <span class="val">{r.ratio_vs_best.toFixed(3)}×</span>
        </div>
      {/each}
      <div class="receipt-total">
        <span class="ok">✓</span>
        <span class="name">{ladder.summary.won}/{ladder.summary.rungs} rungs</span>
        <span class="val">{ladder.summary.min_ratio.toFixed(3)}–{ladder.summary.max_ratio.toFixed(3)}×</span>
      </div>
      <div class="receipt-row">
        <span class="ok pending">◷</span>
        <span class="name">MLPerf v6.1</span>
        <span class="val">submitted</span>
      </div>
    {:else}
      <div class="receipt-pending">
        <span class="tag">▷ {verified.pendingHeadline}</span>
        {#if !compact}<p>{verified.pendingBody}</p>{/if}
      </div>
      <div class="receipt-row">
        <span class="ok">✓</span><span class="name">liveness + coherence</span><span class="val">enforced</span>
      </div>
      <div class="receipt-row">
        <span class="ok pending">◷</span><span class="name">throughput baselines</span><span class="val">awaiting submission</span>
      </div>
    {/if}

    <div class="receipt-foot">
      <span>atlas {bench.generated_sha}</span>
      <span>{bench.generated_date}</span>
    </div>
  </div>
</div>
