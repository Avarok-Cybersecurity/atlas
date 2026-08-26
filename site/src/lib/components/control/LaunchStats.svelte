<!-- SPDX-License-Identifier: AGPL-3.0-only -->
<script>
  // How a running launch is doing.
  //
  // Polled, not streamed. A rate is a difference between two scrapes, so the
  // page asking is what sets the window it gets — and a page that navigates
  // away simply stops asking, which costs the agent nothing.
  //
  // Absent is never zero, all the way to the pixel. A tile with no reading
  // shows a dash and says why; it does not show 0 and let the operator work
  // out for themselves that the model has not finished loading.
  import * as S from '$lib/agent/stats.js';

  let { fleet, recipe, every = 2000 } = $props();

  let stats = $state(null);
  let decodeHistory = $state([]);
  let problem = $state(null);
  let stopped = false;

  const live = $derived(S.hasAnything(stats));

  $effect(() => {
    stopped = false;
    let timer = null;

    async function tick() {
      if (stopped) return;
      try {
        const res = await fleet.agent.launchStats(recipe);
        if (res?.ok) {
          stats = res.reply.stats;
          decodeHistory = S.push(decodeHistory, stats.decode_tokens_per_s);
          problem = null;
        } else {
          // A model still loading its weights is not answering yet. That is an
          // ordinary state, so it is described rather than shown as a fault.
          problem = res?.error?.reason ?? res?.message ?? 'No reading yet.';
          decodeHistory = S.push(decodeHistory, null);
        }
      } catch (e) {
        problem = e?.message ?? 'No reading yet.';
        decodeHistory = S.push(decodeHistory, null);
      }
      if (!stopped) timer = setTimeout(tick, every);
    }

    tick();
    return () => {
      stopped = true;
      if (timer) clearTimeout(timer);
    };
  });

  const path = $derived(S.sparkline(decodeHistory, 220, 34));
</script>

<div class="ls">
  <div class="ls-head">
    <h3>Live</h3>
    {#if stats?.window_s}
      <span class="ls-window">measured over {S.duration(stats.window_s)}</span>
    {/if}
  </div>

  {#if !live && problem}
    <p class="ls-waiting" role="status">{problem}</p>
  {/if}

  <div class="ls-grid">
    <div class="ls-tile ls-tile-wide">
      <span class="ls-label">Decode</span>
      <span class="ls-value">{S.tokens(stats?.decode_tokens_per_s)}<em>tok/s</em></span>
      {#if path}
        <!-- Decorative: every number it encodes is printed beside it, so a
             screen reader loses nothing by skipping the drawing. -->
        <svg class="ls-spark" viewBox="0 0 220 34" preserveAspectRatio="none" aria-hidden="true">
          <path d={path} fill="none" stroke="currentColor" stroke-width="1.5"
                stroke-linejoin="round" stroke-linecap="round" />
        </svg>
      {/if}
    </div>

    <div class="ls-tile">
      <span class="ls-label">Prompt</span>
      <span class="ls-value">{S.tokens(stats?.prompt_tokens_per_s)}<em>tok/s</em></span>
    </div>

    <div class="ls-tile">
      <span class="ls-label">In flight</span>
      <span class="ls-value">{S.count(stats?.requests_active)}</span>
    </div>

    <div class="ls-tile">
      <span class="ls-label">Requests</span>
      <span class="ls-value">{S.count(stats?.requests_total)}</span>
    </div>

    <div class="ls-tile">
      <span class="ls-label">TTFT median</span>
      <span class="ls-value">{S.duration(stats?.ttft_p50_s)}</span>
    </div>

    <div class="ls-tile">
      <span class="ls-label">TTFT p90</span>
      <span class="ls-value">{S.duration(stats?.ttft_p90_s)}</span>
    </div>

    <div class="ls-tile">
      <span class="ls-label">Draft accepted</span>
      <span class="ls-value">{S.percent(stats?.accept_rate)}</span>
      {#if stats?.accept_rate == null}
        <span class="ls-note">not speculating</span>
      {/if}
    </div>

    <div class="ls-tile">
      <span class="ls-label">Prefix cache</span>
      <span class="ls-value">{S.percent(stats?.prefix_hit_rate)}</span>
    </div>
  </div>
</div>
