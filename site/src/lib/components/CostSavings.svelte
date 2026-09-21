<script>
  // Energy savings over time: what serving the SAME token demand on Atlas
  // instead of the baseline is worth, cumulatively.
  //
  // ★ IT FOLLOWS THE PANEL'S RUNG RATHER THAN OWNING ONE. A second rung
  // selector would let this figure disagree with the chart directly above it
  // while both looked authoritative. One selection, two views.
  //
  // ★ IT USES THE SAME JOULES THE CHART PLOTS, including the above-idle
  // toggle, for the same reason: `costPerMillion` and `savingsOverTime` are
  // cross-checked against each other in cost-savings.test.js, and that check
  // is worthless if the two views feed them different numbers.
  import {
    savingsOverTime, SAVINGS_HORIZONS, DEFAULT_TOKENS_PER_DAY,
    TOKENS_PER_DAY_STORAGE_KEY, fmtUsd
  } from '$lib/cost.js';
  import LazyIcon from './LazyIcon.svelte';

  let { cost, rung, usdPerKwh, pue, aboveIdle = false } = $props();

  // localStorage is a per-viewer convenience here, never a source of truth:
  // a private window, cleared site data or a browser that throws on access all
  // land on the declared default rather than on an error.
  let tokensInput = $state(DEFAULT_TOKENS_PER_DAY);
  $effect(() => {
    try {
      const v = Number(localStorage.getItem(TOKENS_PER_DAY_STORAGE_KEY));
      if (Number.isFinite(v) && v > 0) tokensInput = v;
    } catch { /* storage unavailable — keep the default */ }
  });
  function onTokens(e) {
    const v = Number(e.currentTarget.value);
    tokensInput = v;
    try { if (Number.isFinite(v) && v > 0) localStorage.setItem(TOKENS_PER_DAY_STORAGE_KEY, String(v)); } catch { /* ignore */ }
  }

  let days = $state(30);

  const jOf = (s) => {
    const p = (s?.points ?? []).find((x) => x.c === rung && x.energy?.state === 'measured');
    if (!p) return null;
    const e = p.energy;
    const j = aboveIdle && e.aboveIdleJ !== null && e.aboveIdleJ !== undefined ? e.aboveIdleJ : e.energyJ;
    return j / e.tokens;
  };

  // The baseline compared against is the first that HAS energy at this rung —
  // the same one the chart draws. If none does, `savingsOverTime` says so.
  const baseline = $derived((cost?.baselines ?? []).find((b) => jOf(b) !== null) ?? null);
  const result = $derived(
    savingsOverTime({
      atlasJPerTok: jOf(cost?.atlas),
      baseJPerTok: jOf(baseline),
      tokensPerDay: tokensInput,
      usdPerKwh,
      pue: pue.low,
      days
    })
  );

  const W = 320, H = 96, PAD = 4;
  const path = $derived.by(() => {
    if (result.state !== 'measured') return '';
    const pts = result.points;
    const peak = Math.max(...pts.map((p) => Math.abs(p.usd))) || 1;
    const x = (d) => PAD + (d / days) * (W - 2 * PAD);
    const y = (v) => H - PAD - (Math.abs(v) / peak) * (H - 2 * PAD);
    return pts.map((p, i) => `${i ? 'L' : 'M'}${x(p.day).toFixed(1)} ${y(p.usd).toFixed(1)}`).join(' ');
  });
  const losing = $derived(result.state === 'measured' && result.perDayUsd < 0);
</script>

<section class="cost-savings" aria-labelledby="cost-savings-h">
  <h3 id="cost-savings-h"><LazyIcon name="bolt" size={16} /> Energy savings over time</h3>

  {#if result.state !== 'measured'}
    <p class="cost-savings-none">
      Not available yet — {result.why}.
    </p>
  {:else}
    <div class="cost-savings-inputs">
      <label>
        tokens per day
        <input type="number" min="1" step="1e8" value={tokensInput} oninput={onTokens}
               aria-describedby="cost-savings-h" />
      </label>
      <div class="cost-savings-horizons" role="group" aria-label="horizon">
        {#each SAVINGS_HORIZONS as h}
          <button type="button" class="cost-horizon" class:is-on={days === h.days}
                  aria-pressed={days === h.days} onclick={() => (days = h.days)}>{h.label}</button>
        {/each}
      </div>
    </div>

    <p class="cost-savings-figure" class:is-loss={losing}>
      {#if losing}
        Atlas costs <strong>${fmtUsd(Math.abs(result.totalUsd))}</strong> more
      {:else}
        <!-- The leaf appears ONLY on a saving. An icon that means "good" must
             never sit beside a number that is bad. -->
        <LazyIcon name="leaf" size={17} /> <strong>${fmtUsd(result.totalUsd)}</strong> saved
      {/if}
      over {days === 1 ? 'a day' : days === 365 ? 'a year' : `${days} days`}
      · {Math.abs(result.totalKwh).toFixed(Math.abs(result.totalKwh) < 10 ? 1 : 0)} kWh
      <span class="cost-savings-sub">
        at C={rung} vs {baseline.label}, {tokensInput.toLocaleString()} tokens/day, PUE {pue.low}×
      </span>
    </p>

    <svg class="cost-savings-spark" viewBox="0 0 {W} {H}" role="img"
         aria-label="cumulative savings, {days} days">
      <path d={path} fill="none" stroke="currentColor" stroke-width="2" />
    </svg>
  {/if}
</section>
