<!-- SPDX-License-Identifier: AGPL-3.0-only -->
<script>
  // Launching one recipe across several machines.
  //
  // The rules live in `launch.js`, which is pure and tested; this file is the
  // surface. It shows the operator four things in order — what will run, on
  // which machines, whether every machine agreed, and what started — and it
  // never lets them skip a step, because each one is what makes the next safe.
  //
  // Every command shown here was rendered by the machine that would run it.
  // The head does not know another machine's recipe revision or hardware, so a
  // preview it invented would be a guess presented as the thing that executes.
  import * as L from '$lib/agent/launch.js';
  import { copyLabel, copyOrSelect } from '$lib/clipboard.js';
  import * as O from '$lib/agent/overrides.js';
  import * as Prof from '$lib/agent/profile.js';
  import SettingsEditor from './SettingsEditor.svelte';
  import LaunchStats from './LaunchStats.svelte';
  import LaunchLogs from './LaunchLogs.svelte';

  let { fleet } = $props();

  let flow = $state(L.initial());
  let overrides = $state({});
  let showSettings = $state(false);
  let copied = $state('');
  let copyState = $state('idle'); // idle | copied | manual | blocked
  let copyTimer;
  // A launch can navigate away while the flash is pending.
  $effect(() => () => clearTimeout(copyTimer));

  // The operator's remembered preferences. Loaded at init rather than in an
  // effect: there is no storage during prerender, so this is `empty()` on the
  // server and the real profile on the client — and this component only
  // renders once an agent has answered, which never happens while
  // prerendering, so the two cannot disagree on screen.
  const store = typeof localStorage === 'undefined' ? null : localStorage;
  let profile = $state(Prof.load(store));

  // Deliberately NOT `$state`. An effect that reads a flag it also assigns
  // depends on itself, which is the loop this file's own comments warn about;
  // a plain variable is a latch the effect can check without subscribing to.
  let recipeRestored = false;
  let selectionRestored = false;

  const recipes = $derived(
    (fleet.agent?.recipes ?? []).filter((r) => r.runnable).slice().sort((a, b) => a.id.localeCompare(b.id)),
  );
  const recipe = $derived(recipes.find((r) => r.id === flow.recipe) ?? null);
  const candidates = $derived(fleet.launchable);
  const blocker = $derived(L.blocker(flow, recipe, candidates.length));
  const busy = $derived(L.BUSY.includes(flow.phase));
  const held = $derived(flow.epoch != null && flow.phase === 'prepared');

  // Whatever this machine is serving, however it was launched — including a
  // launch from a previous run of the agent, which it re-adopts rather than
  // forgetting.
  const runningHere = $derived(fleet.local?.running ?? null);

  const defaults = $derived(recipe?.defaults ?? {});
  const changed = $derived(O.changedCount(overrides, defaults));
  const wire = $derived(O.toWire(overrides, defaults));

  // Pick up where the operator left off. Two steps, because the two halves
  // become checkable at different moments: the recipe list arrives with the
  // agent's handshake, while the machines that can hold a rank arrive as
  // discovery and pairing resolve. Restoring both on the first signal meant a
  // fleet that had not finished loading discarded the remembered machines
  // silently and for good.
  //
  // Neither restores blind. A remembered choice is a preference, never an
  // assertion about what is on the network, so a recipe this agent does not
  // carry and a machine that is not currently able to hold a rank are both
  // simply not applied.
  $effect(() => {
    if (recipeRestored || recipes.length === 0) return;
    recipeRestored = true;
    const id = profile.recipe;
    if (id == null || !recipes.some((r) => r.id === id)) return;
    flow = L.setRecipe(L.initial(), id);
    overrides = Prof.overridesFor(profile, id);
  });

  $effect(() => {
    // `recipe` gates this as well as `candidates`: selecting machines needs
    // the recipe's own node count to know how many the operator may pick.
    if (selectionRestored || candidates.length === 0 || recipe == null) return;
    // Only latch once something was actually offered to restore against; an
    // operator who never selected anything has nothing to restore, and
    // latching on their behalf costs nothing either way.
    selectionRestored = true;
    if (profile.selected.length === 0) return;

    const live = new Set(candidates.map((n) => n.id));
    let next = flow;
    for (const id of profile.selected) {
      if (live.has(id) && !next.selected.includes(id)) {
        next = L.toggleNode(next, id, recipe);
      }
    }
    if (profile.head != null && next.selected.includes(profile.head)) {
      next = L.setHead(next, profile.head);
    }
    flow = next;
  });

  /** Persist the current plan. Failure is silent by design — see profile.js. */
  function remember(patch) {
    profile = Prof.merge(profile, patch);
    Prof.save(store, profile);
  }

  function chooseRecipe(id) {
    flow = L.setRecipe(flow, id);
    // Bounds and defaults belong to a recipe; carrying one recipe's overrides
    // onto another would apply values the operator chose for something else.
    // A recipe's *own* remembered overrides are a different matter: those are
    // what this operator last chose for this recipe, so they come back.
    overrides = Prof.overridesFor(profile, id);
    remember({ recipe: id });
  }

  function nameOf(id) {
    return fleet.nodes.find((n) => n.id === id)?.name ?? id.slice(0, 12);
  }

  // The client answers with an {ok, reply} envelope rather than the frame, and
  // reading the envelope as the frame produced an empty preview with no error —
  // the silent failure this whole surface exists to avoid. Unwrapped in one
  // place so no verb can get it wrong on its own.
  async function run(begin, call, land) {
    flow = begin(flow);
    try {
      const res = await call();
      if (!res?.ok) {
        flow = L.failed(flow, res?.error ?? res?.message ?? 'The agent did not reply.');
        return;
      }
      flow = land(flow, res.reply);
    } catch (e) {
      flow = L.failed(flow, e);
    }
  }

  const doPreview = () =>
    run(L.beginPreview, () => fleet.agent.previewCluster(flow.recipe, flow.selected, flow.head, wire), L.previewed);

  const doPrepare = () =>
    run(L.beginPrepare, () => fleet.agent.prepareCluster(flow.recipe, flow.selected, flow.head, wire), L.prepared);

  const doCommit = () => run(L.beginCommit, () => fleet.agent.commitCluster(flow.epoch), L.started);

  const doStop = () => run(L.beginStop, () => fleet.agent.stopCluster(), L.stopped);

  async function doAbandon() {
    const epoch = flow.epoch;
    // Optimistic: the reservations are released whether or not this reply
    // arrives, and leaving the operator staring at a spinner would tempt them
    // into a second prepare while the first is still held.
    flow = L.abandoned(flow);
    try {
      await fleet.agent.abortCluster(epoch);
    } catch {
      // The agent releases on its next prepare regardless, and a failure here
      // must not replace whatever the operator was actually doing.
    }
  }

  // A preview describes one exact plan. Changing a setting after it was
  // rendered would leave commands on screen that no longer match what would
  // run, so the plan drops back to unpreviewed. Driven by an explicit callback
  // rather than an effect watching the map: an effect that assigns state it
  // also depends on is a loop waiting to be introduced.
  function settingsChanged() {
    flow = L.settingsChanged(flow);
    if (flow.recipe != null) {
      profile = Prof.rememberOverrides(profile, flow.recipe, overrides);
      Prof.save(store, profile);
    }
  }

  function pickNode(id) {
    flow = L.toggleNode(flow, id, recipe);
    remember({ selected: flow.selected, head: flow.head });
  }

  function pickHead(id) {
    flow = L.setHead(flow, id);
    remember({ selected: flow.selected, head: flow.head });
  }

  // The `try/catch` here was load-bearing for the wrong reason: `copyText`
  // already cannot throw. What was missing is the refusal REPORT.
  async function copy(text, key, el) {
    clearTimeout(copyTimer);
    copied = key;
    copyState = await copyOrSelect(text, el);
    copyTimer = setTimeout(() => {
      if (copied === key) {
        copied = '';
        copyState = 'idle';
      }
    }, 2400);
  }
</script>

<div class="lc">
  {#if runningHere && flow.started.length === 0}
    <div class="lc-here">
      <p class="lc-here-head">
        <strong>{runningHere}</strong> is running on {fleet.local?.name ?? 'this machine'}.
      </p>
      <LaunchStats {fleet} recipe={runningHere} />
      <LaunchLogs {fleet} recipe={runningHere} />
    </div>
  {/if}

  <div class="lc-pick">
    <label class="lc-field">
      <span>Recipe</span>
      <select
        value={flow.recipe ?? ''}
        disabled={busy || held}
        onchange={(e) => chooseRecipe(e.currentTarget.value || null)}
      >
        <option value="">Choose a recipe…</option>
        {#each recipes as r (r.id)}
          <option value={r.id}>{r.id} · {r.nodes === 1 ? '1 machine' : `${r.nodes} machines`}</option>
        {/each}
      </select>
    </label>

    {#if recipe}
      <p class="lc-model">Serves <strong>{recipe.model}</strong></p>
    {/if}
  </div>

  {#if recipe}
    <fieldset class="lc-nodes" disabled={busy || held}>
      <legend>Machines · pick {L.required(recipe)}</legend>
      {#if candidates.length === 0}
        {#if fleet.controlOnly}
          <p class="lc-empty">
            This machine is control only, so it cannot hold a rank itself. Pair a
            machine that can run models and it will appear here.
          </p>
        {:else}
          <p class="lc-empty">No machine here can hold a rank yet. Pair one first.</p>
        {/if}
      {/if}
      {#each candidates as n (n.id)}
        {@const on = flow.selected.includes(n.id)}
        <div class="lc-node" class:lc-node-on={on}>
          <label class="lc-node-pick">
            <input type="checkbox" checked={on} onchange={() => pickNode(n.id)} />
            <span class="lc-node-name">{n.name}</span>
            <span class="lc-node-sub">{n.isLocal ? 'this machine' : (n.addresses[0]?.addr ?? '')}</span>
          </label>
          <label class="lc-head" class:lc-head-off={!on}>
            <input
              type="radio"
              name="cluster-head"
              checked={flow.head === n.id}
              disabled={!on}
              onchange={() => pickHead(n.id)}
            />
            <span>serves the API</span>
          </label>
        </div>
      {/each}
    </fieldset>
  {/if}

  {#if recipe}
    <div class="lc-settings">
      <button class="lc-disclose" onclick={() => (showSettings = !showSettings)} aria-expanded={showSettings}>
        {showSettings ? 'Hide settings' : 'Settings'}
        <span class="lc-changed">{changed === 0 ? 'recipe defaults' : `${changed} changed`}</span>
      </button>
      {#if showSettings}
        <SettingsEditor
          schema={fleet.agent.schema}
          {defaults}
          bind:overrides
          onchange={settingsChanged}
          disabled={busy || held || flow.phase === 'running'}
        />
      {/if}
    </div>
  {/if}

  <div class="lc-actions">
    {#if blocker}
      <p class="lc-blocker">{blocker}</p>
    {:else if flow.phase === 'choosing' || flow.phase === 'failed'}
      <button class="btn btn-primary" onclick={doPreview} disabled={busy}>Show me what will run</button>
    {/if}

    {#if flow.phase === 'previewing'}<p class="lc-wait">Asking each machine what it would run…</p>{/if}
    {#if flow.phase === 'preparing'}<p class="lc-wait">Asking each machine to reserve…</p>{/if}
    {#if flow.phase === 'committing'}<p class="lc-wait">Starting every rank…</p>{/if}
    {#if flow.phase === 'stopping'}<p class="lc-wait">Stopping every rank…</p>{/if}
  </div>

  {#if flow.reason}
    <p class="lc-error" role="alert">{flow.reason}</p>
  {/if}

  {#if flow.linkWarning}
    <p class="lc-warn" role="status">{flow.linkWarning}</p>
  {/if}

  {#if flow.ranks.length > 0}
    <div class="lc-preview">
      <h3>What each machine will run</h3>
      <p class="lc-note">
        Each command below was rendered by the machine that will run it, from its own copy of the recipe — not
        composed here.
      </p>
      {#each flow.ranks as r (r.node)}
        <article class="lc-rank">
          <header>
            <span class="lc-rank-no">rank {r.rank}</span>
            <strong>{r.name}</strong>
            {#if r.rank === 0}<span class="lc-tag">serves the API</span>{/if}
          </header>
          <pre>{r.command}</pre>
          {#if r.unmapped?.length}
            <!-- Per rank, because the machines can be running different
                 revisions: a value that lands on rank 0 can be dropped on
                 rank 1, and that asymmetry is what needs seeing. -->
            <p class="lc-unmapped" role="status">
              {r.name} does not understand {r.unmapped.join(', ')}, so {r.unmapped.length === 1 ? 'it reaches' : 'they reach'}
              nothing on that machine.
            </p>
          {/if}
          <button
            class="lc-copy"
            onclick={(ev) =>
              copy(r.command, r.node, ev.currentTarget.closest('article')?.querySelector('pre'))}
          >
            {copied === r.node ? copyLabel(copyState) : 'Copy'}
          </button>
        </article>
      {/each}

      {#if flow.phase === 'previewed'}
        <button class="btn btn-primary" onclick={doPrepare}>Reserve these machines</button>
        <p class="lc-note">Nothing starts yet. Every machine checks it can run this, and holds its place.</p>
      {/if}
    </div>
  {/if}

  {#if flow.answers.length > 0}
    <div class="lc-prepared">
      <h3>Reservations</h3>
      <ul>
        {#each flow.answers as a (a.node)}
          <li class:lc-refused={!a.prepared}>
            <strong>{a.name}</strong>
            <span>rank {a.rank}</span>
            {#if a.prepared}
              <span class="lc-ok">ready</span>
            {:else}
              <span class="lc-no">{a.reason}</span>
            {/if}
          </li>
        {/each}
      </ul>

      {#if L.mayCommit(flow)}
        <div class="lc-commit">
          <button class="btn btn-primary" onclick={doCommit}>Start the cluster</button>
          <button class="btn" onclick={doAbandon}>Release the reservations</button>
        </div>
      {:else if held}
        <p class="lc-note">
          Not every machine agreed, so nothing can start. The reservations that were taken have already been
          released.
        </p>
        <button class="btn" onclick={doAbandon}>Back to the plan</button>
      {/if}
    </div>
  {/if}

  {#if flow.started.length > 0}
    <div class="lc-running">
      <h3>Running</h3>
      <ul>
        {#each flow.started as r (r.node)}
          <li>
            <strong>{r.name}</strong>
            <span>rank {r.rank}</span>
            <code>{r.container}</code>
            {#if r.endpoint}
              <a href={r.endpoint} rel="noreferrer noopener" target="_blank">{r.endpoint}</a>
            {/if}
          </li>
        {/each}
      </ul>
      <LaunchStats {fleet} recipe={flow.recipe} />
      <LaunchLogs {fleet} recipe={flow.recipe} />

      <div class="lc-commit">
        <button class="btn btn-primary" onclick={doStop} disabled={busy}>Stop the cluster</button>
        <button class="btn" onclick={() => (flow = L.initial())} disabled={busy}>Plan another launch</button>
      </div>
      <p class="lc-note">
        Stopping reaches every machine, not just this one. Leaving a worker running would hold its GPU with
        nothing to serve.
      </p>
    </div>
  {/if}
</div>
