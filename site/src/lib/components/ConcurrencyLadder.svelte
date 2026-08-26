<script>
  // Atlas vs vLLM across the published concurrency ladder, C=1..128.
  //
  // Everything rendered here comes from ladder.generated.json, which
  // gen-ladder.mjs computes from the raw harness output in bench/ladder38/.
  // Nothing on this page is a hand-typed number, including the claim in the
  // heading — `summary.all_won` is derived, so a lost rung changes the copy
  // instead of leaving it stale.
  //
  // Same hand-rolled SVG dialect as GateLadderChart.svelte (log2 X, no chart
  // library) because the rungs double: linear spacing would crush C=1..8,
  // which is where single-stream latency lives.
  import ladder from '$lib/ladder.generated.json';

  // `embedded`: render as a block inside a section that already has a heading
  // and a container (the Verified entry). Default is the standalone section the
  // benchmark dashboard mounts.
  //
  // `compact`: chart and table only. The slide deck at /due-diligence gives the
  // claim its own headline and spends the rest of the slide on the evidence, so
  // it needs the instrument without the surrounding prose or the provenance
  // disclosure — which it reaches on its own slides instead.
  let { embedded = false, compact = false } = $props();

  const W = 760, H = 300, PL = 62, PR = 20, PT = 18, PB = 34;

  const subject = ladder.series.find((s) => s.role === 'subject');
  const baselines = ladder.series.filter((s) => s.role === 'baseline');
  const plotted = [subject, ...baselines];

  const COLOR = {
    atlas: 'var(--accent)',
    'vllm-mtp': 'var(--t2)',
    'vllm-nospec': 'var(--t3, var(--t2))'
  };
  const DASH = { atlas: null, 'vllm-mtp': null, 'vllm-nospec': '5 4' };

  const cs = ladder.concurrencies;
  const allV = plotted.flatMap((s) => s.rungs.map((r) => r.tok_s));
  const vMax = Math.max(...allV) * 1.08;

  const x = (c) => PL + (Math.log2(c) / Math.log2(Math.max(...cs))) * (W - PL - PR);
  const y = (v) => PT + (1 - v / vMax) * (H - PT - PB);
  const path = (rungs) =>
    rungs.map((r, i) => `${i ? 'L' : 'M'}${x(r.c).toFixed(1)} ${y(r.tok_s).toFixed(1)}`).join(' ');

  const yTicks = [0, vMax / 4, vMax / 2, (vMax * 3) / 4, vMax];
  // Two decimals everywhere, which is exactly how RESULTS.md publishes these
  // numbers — the site and the repo record should be diffable by eye.
  const fmtV = (v) => v.toFixed(2);
  // Always three decimals: the rungs span 1.004x to 1.225x, and switching
  // precision by magnitude would print "1.20x" next to "1.004x".
  const ratio = (r) => `${r.toFixed(3)}×`;
</script>

<svelte:element this={embedded ? 'div' : 'section'} id="concurrency"
  class={embedded ? 'cl-embed' : 'section-alt'}>
  <div class={embedded ? 'cl-embed-inner' : 'container'}>
    {#if !embedded}
      <div class="slabel">Concurrency</div>
    {/if}
    {#if !compact}
    <svelte:element this={embedded ? 'h3' : 'h2'} class={embedded ? 'cl-h' : 'stitle'}>
      {#if ladder.summary.all_won}
        Faster than vLLM at every concurrency, C=1 to 128
      {:else}
        Atlas vs vLLM, C=1 to 128 — {ladder.summary.won} of {ladder.summary.rungs} rungs
      {/if}
    </svelte:element>
    <p class="cl-sub">
      {ladder.workload.checkpoint} on one GB10. {ladder.aggregate}. The matched baseline
      runs vLLM's own MTP speculative decoding at K=4, same as Atlas, on the same box,
      checkpoint, client and prompts. Margin ranges
      {ratio(ladder.summary.min_ratio)}–{ratio(ladder.summary.max_ratio)} against whichever
      vLLM configuration is faster at that rung.
    </p>
    {/if}

    <figure class="cl-panel">
      <figcaption class="cl-legend">
        {#each plotted as s}
          <span class="cl-key">
            <svg class="cl-swatch" viewBox="0 0 22 8" aria-hidden="true">
              <line x1="1" y1="4" x2="21" y2="4" stroke={COLOR[s.id]} stroke-width="2.5"
                stroke-dasharray={DASH[s.id]} stroke-linecap="round" />
            </svg>
            <span>{s.label}</span>
            {#if s.parity === 'unmatched'}<span class="cl-tag">config differs</span>{/if}
          </span>
        {/each}
      </figcaption>

      <svg viewBox="0 0 {W} {H}" role="img"
        aria-label="Throughput in tokens per second versus concurrency, Atlas compared with two vLLM configurations">
        {#each yTicks as t}
          <line class="gc-grid" x1={PL} y1={y(t)} x2={W - PR} y2={y(t)} />
          <text class="gc-axis" x={PL - 8} y={y(t) + 3.5} text-anchor="end">{Math.round(t)}</text>
        {/each}
        <text class="gc-axis cl-ylab" text-anchor="middle" x={13} y={(PT + H - PB) / 2 + 4}
          transform="rotate(-90 13 {(PT + H - PB) / 2 + 4})">tok/s</text>
        {#each cs as c}
          <text class="gc-axis" x={x(c)} y={H - 9} text-anchor="middle">{c}</text>
        {/each}
        <text class="gc-axis cl-xlab" x={(PL + W - PR) / 2} y={H - 24} text-anchor="middle">
          concurrent requests
        </text>

        {#each plotted as s}
          <path d={path(s.rungs)} fill="none" stroke={COLOR[s.id]}
            stroke-width={s.role === 'subject' ? 2.6 : 1.8}
            stroke-dasharray={DASH[s.id]} stroke-linejoin="round" stroke-linecap="round"
            opacity={s.role === 'subject' ? 1 : 0.75} />
          {#each s.rungs as r}
            <circle cx={x(r.c)} cy={y(r.tok_s)} r={s.role === 'subject' ? 3.6 : 2.6}
              fill={COLOR[s.id]} opacity={s.role === 'subject' ? 1 : 0.75}>
              <title>{s.label} · C={r.c} · {fmtV(r.tok_s)} tok/s · mean of {r.reps} reps</title>
            </circle>
          {/each}
        {/each}
      </svg>
    </figure>

    <div class="cl-tablewrap">
      <table class="cl-table">
        <caption class="cl-caption">
          Throughput in tok/s. Ratio is Atlas over the faster vLLM configuration at that
          rung; where the two disagree the losing one is not silently dropped.
        </caption>
        <thead>
          <tr>
            <th scope="col">C</th>
            <th scope="col">Atlas</th>
            {#each baselines as b}<th scope="col">{b.label}</th>{/each}
            <th scope="col">Ratio</th>
          </tr>
        </thead>
        <tbody>
          {#each ladder.rows as row}
            <tr>
              <th scope="row" class="mono">{row.c}</th>
              <td class="mono cl-win">{fmtV(row.atlas)}</td>
              {#each row.baselines as b}
                <td class="mono" class:cl-best={b.id === row.best_baseline_id}>{fmtV(b.tok_s)}</td>
              {/each}
              <td class="mono cl-ratio">{ratio(row.ratio_vs_best)}</td>
            </tr>
          {/each}
        </tbody>
      </table>
    </div>

    {#if !compact}
    <details class="cl-details">
      <summary class="cl-toggle">Exact configuration and provenance</summary>
      <div class="cl-meta">
        <dl class="cl-facts">
          <div><dt>Checkpoint</dt><dd class="mono">{ladder.workload.checkpoint}</dd></div>
          <div><dt>Hardware</dt><dd>{ladder.box.gpu} — {ladder.box.name}</dd></div>
          <div><dt>Workload</dt><dd>
            ISL {ladder.workload.isl_tokens} / OSL {ladder.workload.osl_tokens},
            {ladder.workload.reps} timed reps + {ladder.workload.warmup} warmup,
            temperature {ladder.workload.temperature}, seed {ladder.workload.seed}
          </dd></div>
          <div><dt>Parity</dt><dd>
            {ladder.workload.thinking}; {ladder.workload.sampling_parity}
          </dd></div>
        </dl>

        {#each ladder.series as s}
          <article class="cl-series">
            <h3>{s.label} <span class="cl-eng">{s.engine}</span></h3>
            {#if s.parity === 'unmatched'}
              <p class="cl-warn">
                Not matched to Atlas: {s.parity_deltas.join('; ')}. Shown because
                it is the faster vLLM configuration at C=128.
              </p>
            {/if}
            <p class="cl-note">{s.source_note}</p>
            <div class="cl-kv"><span>Build</span><code>{s.build}</code></div>
            {#if s.build_note}<p class="cl-note">{s.build_note}</p>{/if}
            <div class="cl-kv"><span>Speculation</span><code>{s.speculation}</code></div>
            {#if s.env}<div class="cl-kv"><span>Env</span><code>{s.env}</code></div>{/if}
            <div class="cl-kv cl-kv-block"><span>Command</span><code>{s.cli}</code></div>
          </article>
        {/each}

        <article class="cl-series">
          <h3>Per-rung detail</h3>
          <div class="cl-tablewrap">
            <table class="cl-table cl-table-dense">
              <thead>
                <tr>
                  <th scope="col">Series</th><th scope="col">C</th><th scope="col">tok/s</th>
                  <th scope="col">median</th><th scope="col">spread</th>
                  <th scope="col">TTFT p50</th><th scope="col">TPOT p50</th>
                  <th scope="col">source file</th>
                </tr>
              </thead>
              <tbody>
                {#each ladder.series as s}
                  {#each s.rungs as r}
                    <tr>
                      <td>{s.label}</td>
                      <td class="mono">{r.c}</td>
                      <td class="mono">{fmtV(r.tok_s)}</td>
                      <td class="mono">{fmtV(r.tok_s_median)}</td>
                      <td class="mono">{r.spread_pct}%</td>
                      <td class="mono">{r.ttft_p50_ms ?? '—'}{r.ttft_p50_ms ? ' ms' : ''}</td>
                      <td class="mono">{r.tpot_p50_ms ?? '—'}{r.tpot_p50_ms ? ' ms' : ''}</td>
                      <td class="mono cl-src">{r.source}</td>
                    </tr>
                  {/each}
                {/each}
              </tbody>
            </table>
          </div>
          <p class="cl-note">
            Harness {ladder.workload.harness}. Two harness revisions appear above:
            {#each Object.entries(ladder.harness_shas).filter(([k]) => k !== 'equivalence') as [sha, what], i}
              {i ? '; ' : ''}<code>{sha}</code> — {what}
            {/each}
            {ladder.harness_shas.equivalence}
          </p>
          <p class="cl-note">
            Full campaign log, including every rung we lost on the way and the three
            claims we retracted: <a href={ladder.results_doc_url}>{ladder.results_doc}</a>.
            Generated {ladder.generated_utc} from the committed measurements.
          </p>
        </article>
      </div>
    </details>
    {/if}
  </div>
</svelte:element>
