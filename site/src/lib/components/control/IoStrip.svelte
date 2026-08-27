<!-- SPDX-License-Identifier: AGPL-3.0-only -->
<script>
  // Region C3: the serving I/O strip — 120px, two fixed rows.
  //
  // Row A: decode and prompt tok/s (numeral + gap-honest sparkline), requests
  // total, requests in flight. Row B: TTFT p50/p90, draft accept, prefix
  // cache — and ISL/OSL, which were placeholders until the agent learned to
  // emit `isl_mean`/`osl_mean`. They kept the same footprint throughout, so
  // going live changed a border and a value and nothing reflowed.
  //
  // Every rule about what a tile shows lives in iostrip.js, tested: the four
  // kinds of absence (reading / pending / absent / placeholder) must never
  // collapse, the first poll surrenders its rates, and a window wider than
  // 60s is a fabricated rate. This file only draws what tiles() says.

  import * as IO from '$lib/agent/iostrip.js';
  import * as S from '$lib/agent/stats.js';
  import { nameOf } from '$lib/agent/refusal.js';

  let {
    node,
    /** Poller entry for this node: {reading, at, failure, via, decodeHist, promptHist}. */
    entry = null,
    paused = false,
    nodes = []
  } = $props();

  const trusted = $derived(
    node.isLocal || node.pairing === 'paired' || node.pairing === 'vouched' || node.pairing === 'unreachable'
  );
  const serving = $derived(Boolean(node.running));
  const reading = $derived(entry?.reading ?? null);
  const mode = $derived(
    trusted ? IO.mode({ serving, reading, failure: entry?.failure ?? null }) : 'off'
  );

  const viaId = $derived(entry?.via ?? node.reachedVia ?? null);
  const viaName = $derived(viaId ? nameOf(viaId, nodes) : null);
  // The strip is fixed at 120px, so the trouble states live in the caption
  // slot rather than adding a row the budget does not have.
  const caption = $derived.by(() => {
    const base = IO.caption(reading, { via: viaName });
    if (mode === 'unanswered') {
      const why = entry?.failure ? ` — ${entry.failure}` : '';
      return [base, `not answering${why}`].filter(Boolean).join(' · ');
    }
    if (mode === 'quiet') {
      return [base, 'engine answered with no fields yet — loading, not idle']
        .filter(Boolean)
        .join(' · ');
    }
    return base;
  });

  const held = $derived(paused || mode === 'unanswered');
  const tiles = $derived(IO.tiles(reading, { paused }));
  const byId = $derived(Object.fromEntries(tiles.map((t) => [t.id, t])));

  const decodePath = $derived(
    S.sparkline(S.timeline(entry?.decodeHist ?? [], { held }), 220, 30)
  );
  const promptPath = $derived(
    S.sparkline(S.timeline(entry?.promptHist ?? [], { held }), 220, 30)
  );
</script>

{#snippet tile(t)}
  <div class="io-tile" class:io-paused={t.paused} class:io-absent={t.kind === 'absent'}>
    <span class="io-label">{t.label}</span>
    {#if t.kind === 'pending'}
      <span class="vt-skeleton" aria-hidden="true"></span>
      <span class="visually-hidden">{t.label}: waiting for the first sample</span>
    {:else if t.kind === 'absent'}
      <span class="io-val io-dash" aria-hidden="true">{t.text}</span>
      <span class="io-note">{t.note}</span>
      <span class="visually-hidden">{t.label}: {t.note}</span>
    {:else}
      <span class="io-val">{t.text}{#if t.unit}<span class="io-unit"> {t.unit}</span>{/if}</span>
      {#if t.paused}<span class="io-note">paused</span>{/if}
    {/if}
  </div>
{/snippet}

{#snippet wide(t, path)}
  <div class="io-tile io-wide" class:io-paused={t.paused} class:io-absent={t.kind === 'absent'}>
    <span class="io-label"
      >{t.label}{#if t.id === 'decode' && caption}<span class="io-caption"> · {caption}</span>{/if}</span
    >
    <div class="io-wide-body">
      {#if t.kind === 'pending'}
        <span class="vt-skeleton" aria-hidden="true"></span>
        <span class="visually-hidden">{t.label}: waiting for the first sample</span>
      {:else if t.kind === 'absent'}
        <span class="io-val io-dash" aria-hidden="true">—</span>
        <span class="io-note">{t.note}</span>
        <span class="visually-hidden">{t.label}: {t.note}</span>
      {:else}
        <span class="io-val">{t.text}<span class="io-unit"> {t.unit}</span></span>
        {#if t.paused}<span class="io-note">paused</span>{/if}
      {/if}
      {#if path}
        <!-- Decorative: every number it encodes is printed beside it. -->
        <svg class="io-spark" viewBox="0 0 220 30" preserveAspectRatio="none" aria-hidden="true">
          <path d={path} fill="none" stroke="currentColor" stroke-width="1.5"
                stroke-linejoin="round" stroke-linecap="round" />
        </svg>
      {/if}
    </div>
  </div>
{/snippet}

<div class="io" aria-label="Serving I/O">
  {#if !trusted}
    <p class="io-off">Nothing is known about serving on an unpaired machine.</p>
  {:else if mode === 'off'}
    <!-- One sentence, never a wall of dashes. -->
    <p class="io-off">Nothing serving on this node.</p>
  {:else}
    <div class="io-rowa">
      {@render wide(byId['decode'], decodePath)}
      {@render wide(byId['prompt'], promptPath)}
      {@render tile(byId['requests-total'])}
      {@render tile(byId['requests-active'])}
    </div>
    <div class="io-rowb">
      {@render tile(byId['ttft-p50'])}
      {@render tile(byId['ttft-p90'])}
      {@render tile(byId['accept'])}
      {@render tile(byId['prefix'])}
      {@render tile(byId['isl'])}
      {@render tile(byId['osl'])}
    </div>
  {/if}
</div>
