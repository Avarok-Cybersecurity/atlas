<!-- SPDX-License-Identifier: AGPL-3.0-only -->
<script>
  // The tail of a launch's log.
  //
  // Diagnosis only. Every number on this page comes from the engine's own
  // /metrics; nothing here is parsed out of log text, because a log line is
  // prose that changes whenever someone rewords it.
  //
  // Shown for a container that has *exited* as well as one that is running —
  // that is the case where it matters most. A rank that died a second after
  // starting is exactly the launch whose last words an operator needs, and the
  // agent keeps the container precisely so they still exist.

  let { fleet, recipe, every = 3000 } = $props();

  let lines = $state([]);
  let container = $state('');
  let running = $state(true);
  let problem = $state(null);
  let open = $state(false);
  let follow = $state(true);
  let pane;

  $effect(() => {
    if (!open) return;
    let stopped = false;
    let timer = null;

    async function tick() {
      if (stopped) return;
      try {
        const res = await fleet.agent.launchLogs(recipe, 200);
        if (res?.ok) {
          lines = res.reply.lines;
          container = res.reply.container;
          running = res.reply.running;
          problem = null;
          if (follow && pane) pane.scrollTop = pane.scrollHeight;
        } else {
          problem = res?.error?.reason ?? res?.message ?? 'No log yet.';
        }
      } catch (e) {
        problem = e?.message ?? 'No log yet.';
      }
      // A container that has exited will not say anything else, so there is
      // nothing to poll for.
      if (!stopped && running) timer = setTimeout(tick, every);
    }

    tick();
    return () => {
      stopped = true;
      if (timer) clearTimeout(timer);
    };
  });
</script>

<div class="lg">
  <button class="lg-disclose" onclick={() => (open = !open)} aria-expanded={open}>
    {open ? 'Hide log' : 'Log'}
    {#if open && container}<span class="lg-name">{container}{running ? '' : ' · exited'}</span>{/if}
  </button>

  {#if open}
    {#if problem}
      <p class="lg-problem" role="status">{problem}</p>
    {/if}
    {#if !running && lines.length > 0}
      <p class="lg-exited" role="status">
        This container has exited. These are the last lines it wrote.
      </p>
    {/if}
    <div class="lg-pane" bind:this={pane} onscroll={(e) => {
      const el = e.currentTarget;
      // Following means "pinned to the bottom". Scrolling up turns it off, so
      // reading an earlier line is not fought by the next poll.
      follow = el.scrollHeight - el.scrollTop - el.clientHeight < 24;
    }}>
      {#if lines.length === 0 && !problem}
        <p class="lg-empty">Nothing yet.</p>
      {/if}
      {#each lines as line, i (i)}
        <div class="lg-line">{line}</div>
      {/each}
    </div>
    {#if running && !follow}
      <button class="lg-follow" onclick={() => { follow = true; if (pane) pane.scrollTop = pane.scrollHeight; }}>
        Jump to newest
      </button>
    {/if}
  {/if}
</div>
