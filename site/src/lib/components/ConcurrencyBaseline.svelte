<script>
  // ConcurrencyBaseline.svelte — the comparison block for a subject that has
  // a vLLM one-shot on the PUBLISHED instrument (the ladder harness: ISL 128 /
  // OSL 1024, ctx 2048) and no Atlas leg at that instrument yet.
  //
  // Everything here is read from the subject's generated ladder, which
  // gen-ladder.mjs computes from the raw harness file the manifest names.
  // Nothing is typed: the values, the date on the chip and the in-plot stamp,
  // the harness hash, the reason a rung is absent — all from the data.
  //
  // Same hand-rolled SVG dialect as ConcurrencyComparison's live figure (log2
  // X over the page's declared rungs) so the MoE tab's axes match the other
  // two tabs'. The vLLM series is drawn in neutral ink with SQUARE marks: a
  // one-shot is never drawn in a series colour, and a square is the glyph the
  // legend's hollow "not measured" key is the empty form of.
  //
  // A rung the baseline did not measure is a dotted guide and the recorded
  // reason, where the point would be. Never a mark at zero.
  import { absentReasonOf, baselineSeriesOf, measuredRange, oneShotChip } from '$lib/concurrency-comparison.js';

  /** @type {{ subject: object, ladder: object, rungs: number[] }} */
  let { subject, ladder, rungs } = $props();

  const baselines = $derived(baselineSeriesOf(ladder));
  const w = $derived(ladder.workload);
  const engines = $derived([...new Set(baselines.map((b) => `${b.engine} (${b.build})`))]);
  const range = $derived(measuredRange(baselines.flatMap((b) => b.rungs)));
  const overridden = $derived(baselines.filter((b) => b.kernel_override));
  const unmeasured = $derived(baselines.filter((b) => b.unmeasured));
  const revisions = $derived(Object.entries(ladder.harness_shas).filter(([k]) => k !== 'equivalence'));

  const W = 720, H = 232, PL = 56, PR = 16, PT = 14, PB = 30;
  const measured = $derived(new Set(baselines.flatMap((b) => b.rungs.map((r) => r.c))));
  // The page's declared rungs plus the baseline's own, so a rung the gate
  // declares and the baseline skipped is answered where its point would be.
  const axis = $derived([...new Set([...rungs, ...measured])].sort((a, b) => a - b));
  const absent = $derived(axis.filter((c) => !measured.has(c)));
  const vMax = $derived(Math.max(...baselines.flatMap((b) => b.rungs.map((r) => r.tok_s))) * 1.12);
  const l0 = $derived(Math.log2(Math.min(...axis)));
  const l1 = $derived(Math.log2(Math.max(...axis)));
  const x = (c) => PL + (l1 === l0 ? 0.5 : (Math.log2(c) - l0) / (l1 - l0)) * (W - PL - PR);
  const y = (v) => PT + (1 - v / vMax) * (H - PT - PB);
  const path = (b) => b.rungs.map((r, i) => `${i ? 'L' : 'M'}${x(r.c).toFixed(1)} ${y(r.tok_s).toFixed(1)}`).join(' ');
  const yTicks = $derived([0, vMax / 2, vMax]);
  const fmtV = (v) => v.toFixed(2);
  const fmtTick = (v) => +v.toFixed(1);
  const last = (b) => b.rungs[b.rungs.length - 1];
  const title = $derived(
    `Atlas vs vLLM · published instrument · ISL ${w.isl_tokens} / OSL ${w.osl_tokens} · vLLM one-shot, no Atlas run yet`
  );
</script>

<figure class="gate-panel cmp">
  <figcaption class="gate-panel-head">
    <span class="gate-panel-title">{title}</span>
    <span class="gate-panel-unit">tok/s</span>
    <span class="gate-legend">
      <span class="gl-group">
        <!-- A HOLLOW circle: the Atlas mark is a filled disc, so an empty one
             is the same key with nothing in it. -->
        <span class="gate-legend-item">
          <svg class="gl-swatch" viewBox="0 0 20 10" aria-hidden="true">
            <circle cx="10" cy="5" r="3.5" fill="none" stroke="var(--t2)" stroke-width="1.5" />
          </svg>Atlas · no run at this instrument yet
        </span>
      </span>
      <span class="gl-sep" aria-hidden="true"></span>
      <span class="gl-group">
        {#each baselines as b}
          <span class="gate-legend-item">
            <svg class="gl-swatch" viewBox="0 0 20 10" aria-hidden="true">
              <line x1="1" y1="5" x2="19" y2="5" stroke="var(--t2)" stroke-width="1.5" />
              <rect x="6.5" y="1.5" width="7" height="7" fill="var(--t2)" />
            </svg>{oneShotChip(b)}
          </span>
        {/each}
      </span>
    </span>
  </figcaption>

  <svg viewBox="0 0 {W} {H}" role="img"
    aria-label="vLLM throughput versus concurrency on the published instrument for {subject.checkpoint}; no Atlas series yet">
    {#each yTicks as t}
      <line class="gc-grid" x1={PL} y1={y(t)} x2={W - PR} y2={y(t)} />
      <text class="gc-axis" x={PL - 8} y={y(t) + 3.5} text-anchor="end">{fmtTick(t)}</text>
    {/each}
    {#each axis as c}
      <text class="gc-axis" x={x(c)} y={H - 8} text-anchor="middle">C={c}</text>
    {/each}
    {#each absent as c}
      <g class="cmp-absent">
        <title>{absentReasonOf(ladder, c)}</title>
        <line class="gc-grid gc-grid-clipped" x1={x(c)} y1={PT} x2={x(c)} y2={H - PB} />
        <text class="gc-ref-label" x={x(c)} y={PT + 10} text-anchor="middle">not measured</text>
      </g>
    {/each}
    {#each baselines as b}
      <path d={path(b)} fill="none" stroke="var(--t2)" stroke-width="1.5" stroke-linejoin="round" stroke-linecap="square" />
      {#each b.rungs as r}
        <rect class="cmp-sq" x={x(r.c) - 3.5} y={y(r.tok_s) - 3.5} width="7" height="7" fill="var(--t2)">
          <title>{b.label} · C={r.c} · {fmtV(r.tok_s)} tok/s · mean of {r.reps} reps · spread {r.spread_pct}% · {r.source}</title>
        </rect>
      {/each}
      <!-- The dated stamp is INSIDE the plot so a cropped screenshot still
           says this is a snapshot, not a live series. -->
      <text class="gc-ref-label cl-stamp" x={x(last(b).c)} y={y(last(b).tok_s) - 9} text-anchor="end">
        {b.engine} · {last(b).measured_utc.slice(0, 10)}
      </text>
    {/each}
  </svg>

  <p class="cmp-caption">
    vLLM was measured <strong>once</strong>, on {range}, with
    {#each engines as e, i}{i ? '; ' : ''}<code>{e}</code>{/each} on {ladder.box.name}, at this
    ladder's own instrument — ISL {w.isl_tokens} / OSL {w.osl_tokens}, {w.reps} timed reps +
    {w.warmup} warmup, temperature {w.temperature}, seed {w.seed} — and is not re-measured when
    Atlas moves.
    {#each overridden as b}
      <strong>{b.label} is not running vLLM's default kernel here.</strong>
      {b.kernel_override.what} {b.kernel_override.why}
      Forced with <code>{b.kernel_override.env}</code>; the crash it avoids:
      <code>{b.kernel_override.error}</code>.
    {/each}
    {#each unmeasured as b}
      <strong>C={b.unmeasured.rungs.join(', ')} not measured.</strong> {b.unmeasured.reason}
    {/each}
  </p>

  <!-- What is missing, why, what fills it — in that order, and never a zero. -->
  <div class="cmp-empty">
    <p><strong>No Atlas run at this instrument yet for <code>{subject.checkpoint}</code>.</strong></p>
    <p>
      What fills the comparison: one Atlas leg of <code>{w.harness}</code> at these settings, filed
      as a series with role <code>subject</code> in <code>{ladder.manifest}</code>, whose sources
      name its raw files. A <code>{subject.gate}</code> gate run on main appears as the live series
      below, on the gate's own instrument; it is drawn against this one-shot only if its instrument
      fingerprint matches, and the ladder harness declares no prompt fixture where the gate does,
      so the chart will name the axes that differ rather than draw them on one axis.
    </p>
  </div>

  <details class="cl-details">
    <summary class="cl-toggle">Exact configuration and provenance</summary>
    <div class="cl-meta">
      {#each baselines as s}
        <article class="cl-series">
          <h3>{s.label} <span class="cl-eng">{s.engine}</span></h3>
          <p class="cl-note">{s.source_note}</p>
          <div class="cl-kv"><span>Build</span><code>{s.build}</code></div>
          {#if s.build_note}<p class="cl-note">{s.build_note}</p>{/if}
          <div class="cl-kv"><span>Speculation</span><code>{s.speculation}</code></div>
          {#if s.env}<div class="cl-kv"><span>Env</span><code>{s.env}</code></div>{/if}
          {#if s.kernel_override}
            <div class="cl-kv"><span>Kernel</span><code>{s.kernel_override.env}</code></div>
            <p class="cl-warn">{s.kernel_override.what} {s.kernel_override.why} {s.kernel_override.scope}</p>
            <div class="cl-kv cl-kv-block"><span>Crash avoided</span><code>{s.kernel_override.error}</code></div>
          {/if}
          <div class="cl-kv cl-kv-block"><span>Command</span><code>{s.cli}</code></div>
          <div class="cl-kv cl-kv-block"><span>Instrument</span><code>{JSON.stringify(s.instrument)}</code></div>
          {#if s.instrument_note}<p class="cl-note">{s.instrument_note}</p>{/if}
          {#if s.parity_note}<p class="cl-note">Parity: {s.parity_note}</p>{/if}
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
                <th scope="col">measured</th><th scope="col">source file</th>
              </tr>
            </thead>
            <tbody>
              {#each baselines as s}
                {#each s.rungs as r}
                  <tr>
                    <td>{s.label}</td>
                    <td class="mono">{r.c}</td>
                    <td class="mono">{fmtV(r.tok_s)}</td>
                    <td class="mono">{fmtV(r.tok_s_median)}</td>
                    <td class="mono">{r.spread_pct}%</td>
                    <td class="mono">{r.ttft_p50_ms ?? '—'}{r.ttft_p50_ms ? ' ms' : ''}</td>
                    <td class="mono">{r.tpot_p50_ms ?? '—'}{r.tpot_p50_ms ? ' ms' : ''}</td>
                    <td class="mono">{r.measured_utc}</td>
                    <td class="mono cl-src">{r.source}</td>
                  </tr>
                {/each}
              {/each}
            </tbody>
          </table>
        </div>
        <p class="cl-note">
          Harness {w.harness}, sha256 of the committed copy <code>{ladder.harness_repo_sha256.slice(0, 10)}</code>.
          {#each revisions as [sha, what], i}{i ? '; ' : ''}<code>{sha}</code> — {what}{/each}
          {ladder.harness_shas.equivalence}
        </p>
        <p class="cl-note">
          Manifest and raw receipt: <a href={ladder.results_doc_url}>{ladder.results_doc}</a>.
        </p>
      </article>
    </div>
  </details>
</figure>
