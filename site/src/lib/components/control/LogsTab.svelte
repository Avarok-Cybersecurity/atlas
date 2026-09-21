<!-- SPDX-License-Identifier: AGPL-3.0-only -->
<script>
  // Dock tab: the tail of the selected node's launch log, local or forwarded.
  //
  // Diagnosis only — every number on this page comes from the engine's own
  // /metrics; nothing is parsed out of log text. Shown for a container that
  // has exited as well: a rank that died a second after starting is exactly
  // the launch whose last words an operator needs.

  import { nameOf } from '$lib/agent/refusal.js';

  let { fleet, node, nodes = [], every = 3000 } = $props();

  let lines = $state([]);
  let container = $state('');
  let running = $state(true);
  let problem = $state(null);
  let follow = $state(true);
  let via = $state(null);
  let pane = $state(null);

  // Value-stable deriveds: the fleet list is replaced wholesale on every
  // vitals event, so `node` is a NEW OBJECT every second. An effect that read
  // it directly would tear down and restart the log poll once a second; these
  // recompute then, but their VALUES only change when the fact does, so the
  // effect below re-runs only on a real target or recipe change.
  const recipe = $derived(node?.running ?? null);
  const on = $derived(node && !node.isLocal ? node.id : null);

  $effect(() => {
    const r = recipe;
    const target = on;
    if (!r) return;
    lines = [];
    problem = null;
    let stopped = false;
    let timer = null;

    async function tick() {
      if (stopped) return;
      try {
        const res = await fleet.agent.launchLogs(r, 200, target);
        if (stopped) return;
        if (res?.ok) {
          lines = res.reply.lines;
          container = res.reply.container;
          running = res.reply.running;
          via = res.reply.via ?? null;
          problem = null;
          if (follow && pane) pane.scrollTop = pane.scrollHeight;
        } else {
          problem = res?.error?.reason ?? res?.message ?? 'No log yet.';
        }
      } catch (e) {
        problem = e?.message ?? 'No log yet.';
      }
      // An exited container will not say anything else.
      if (!stopped && running) timer = setTimeout(tick, every);
    }

    tick();
    return () => {
      stopped = true;
      if (timer) clearTimeout(timer);
    };
  });
</script>

<div class="dt">
  {#if !recipe}
    <p class="dt-quiet">Nothing is serving on this node, so there is no launch log to tail.</p>
  {:else}
    <p class="dt-cap mono">
      {container || recipe}{running ? '' : ' · exited'}
      {#if via}· via {nameOf(via, nodes)}{/if}
    </p>
    {#if problem}
      <p class="dt-problem">{problem}</p>
    {/if}
    {#if !running && lines.length > 0}
      <p class="dt-note">This container has exited. These are the last lines it wrote.</p>
    {/if}
    <!-- svelte-ignore a11y_no_noninteractive_tabindex -->
    <div
      class="dt-log mono"
      role="region"
      aria-label="Launch log tail"
      tabindex="0"
      bind:this={pane}
      onscroll={(e) => {
        const el = e.currentTarget;
        // Following means pinned to the bottom; scrolling up turns it off so
        // reading an earlier line is not fought by the next poll.
        follow = el.scrollHeight - el.scrollTop - el.clientHeight < 24;
      }}
    >
      {#if lines.length === 0 && !problem}
        <p class="dt-quiet">Nothing yet.</p>
      {/if}
      {#each lines as line, i (i)}
        <div class="dt-line">{line}</div>
      {/each}
    </div>
    {#if running && !follow}
      <button
        type="button"
        class="dt-follow"
        onclick={() => {
          follow = true;
          if (pane) pane.scrollTop = pane.scrollHeight;
        }}
      >
        Jump to newest
      </button>
    {/if}
  {/if}
</div>
