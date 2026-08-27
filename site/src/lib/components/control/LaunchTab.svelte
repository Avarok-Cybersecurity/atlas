<!-- SPDX-License-Identifier: AGPL-3.0-only -->
<script>
  // Dock tab: launching one recipe on the selected node, local or forwarded.
  //
  // The ceremony is deliberate: inventory → settings → Preview → Launch.
  // Preview is mandatory — the Launch button stays disabled until the exact
  // command has been rendered by the agent's own renderer, and any settings
  // change voids the preview, because a command an operator has not seen is
  // a command they did not approve. A launch toward a machine reached
  // through a peer interposes the travel warning before it goes.
  //
  // The settings schema is the LOCAL agent's (it arrived in `ready`). The
  // target re-validates every value and its answer is the one that counts;
  // skew between the two surfaces as the agent's own error, never silently.

  import { untrack } from 'svelte';
  import SettingsEditor from './SettingsEditor.svelte';
  import { onTarget, route, travelWarning } from '$lib/agent/verbs.js';
  import { nameOf, refusal } from '$lib/agent/refusal.js';
  import { copyText } from '$lib/clipboard.js';

  let { fleet, node, nodes = [], onlog } = $props();

  let recipes = $state(null);
  let listProblem = $state(null);
  let picked = $state(null);
  let overrides = $state({});
  let preview = $state(null);
  let unapplied = $state([]);
  let busy = $state(false);
  let problem = $state(null);
  let confirming = $state(false);
  let started = $state(null);
  let copied = $state(false);

  const trusted = $derived(
    Boolean(node && (node.isLocal || node.pairing === 'paired' || node.pairing === 'vouched'))
  );
  const travel = $derived(travelWarning(node, nodes));
  const info = $derived(recipes?.find((r) => r.id === picked) ?? null);
  // Value-stable: `node` is a new object on every 1Hz vitals event; keying the
  // reset effect on these primitives keeps it from wiping the form each second.
  const nodeId = $derived(node?.id ?? null);
  const on = $derived(node && !node.isLocal ? node.id : null);
  const routeText = $derived(route(node, nodes));

  // Log entries carry who the verb was aimed at, not who clicked: a launch
  // that ran on dgx3 attributed to "this machine" is a lie in the audit line.
  function logIt(verb, ok, outcome) {
    onlog?.({
      verb,
      ok,
      outcome,
      target: node?.isLocal ? 'this machine' : (node?.name ?? ''),
      route: routeText
    });
  }

  function fail(res, verb) {
    const r = refusal(
      { error: res.error ?? null, message: res.message ?? null },
      { target: node ? onTarget(node) : null, nodes }
    );
    problem = r.text;
    logIt(verb, false, r.text);
    return r.text;
  }

  $effect(() => {
    const id = nodeId;
    const target = on;
    if (!id || !trusted) return;
    // New target, new inventory; everything downstream of the old one is void.
    recipes = null;
    listProblem = null;
    picked = null;
    overrides = {};
    preview = null;
    unapplied = [];
    problem = null;
    confirming = false;
    started = null;
    let stale = false;
    untrack(() => {
      (async () => {
        const res = await fleet.agent.listRecipes(target);
        if (stale) return;
        if (res.ok) {
          recipes = res.reply.recipes ?? [];
        } else {
          const r = refusal(
            { error: res.error ?? null, message: res.message ?? null },
            { target, nodes }
          );
          listProblem = r.text;
        }
      })();
    });
    return () => (stale = true);
  });

  function pick(id) {
    picked = id;
    overrides = {};
    preview = null;
    unapplied = [];
    problem = null;
    confirming = false;
    started = null;
  }

  async function doPreview() {
    if (!picked) return;
    busy = true;
    problem = null;
    const res = await fleet.agent.preview(picked, overrides, onTarget(node));
    busy = false;
    if (res.ok) {
      preview = res.reply.command;
      unapplied = res.reply.unapplied ?? [];
      logIt('preview', true, `previewed ${picked}`);
    } else {
      preview = null;
      fail(res, 'preview');
    }
  }

  async function doLaunch() {
    if (!picked || preview === null) return;
    if (travel && !confirming) {
      // The travel warning is interposed, not implied: the first press arms,
      // the second sends. See §3 — mutating verbs through a relay confirm.
      confirming = true;
      return;
    }
    confirming = false;
    busy = true;
    problem = null;
    const res = await fleet.agent.launch(picked, overrides, onTarget(node));
    busy = false;
    if (res.ok) {
      started = {
        recipe: res.reply.recipe,
        container: res.reply.container,
        endpoint: res.reply.endpoint ?? null,
        on: res.reply.on ?? null,
        via: res.reply.via ?? null
      };
      logIt('launch', true, `launched ${res.reply.recipe}`);
    } else {
      fail(res, 'launch');
    }
  }

  async function copyEndpoint() {
    copied = (await copyText(started.endpoint)) === 'copied';
    if (copied) setTimeout(() => (copied = false), 1500);
  }
</script>

<div class="dt">
  {#if !node}
    <p class="dt-quiet">No machine is selected.</p>
  {:else if !trusted}
    <p class="dt-quiet">
      Launching needs a paired machine. Pair this one first — nothing can be
      asked of a machine that has only been seen on the network.
    </p>
  {:else if node.canLaunch !== true}
    <p class="dt-quiet">
      <span class="fl-co-chip">Control only</span>
      {node.cannotLaunchReason || 'This machine reports it cannot run models.'}
      It can still drive machines that can — select one in the roster and
      launch there.
    </p>
  {:else}
    {#if listProblem}
      <p class="dt-problem">{listProblem}</p>
    {:else if recipes === null}
      <p class="dt-quiet">Asking for the recipe inventory…</p>
    {:else if recipes.length === 0}
      <p class="dt-quiet">This agent lists no recipes. Updating atlasctl adds the current set.</p>
    {:else}
      <div class="dt-recipes" role="radiogroup" aria-label="Recipe">
        {#each recipes as r (r.id)}
          <button
            type="button"
            role="radio"
            aria-checked={picked === r.id}
            class="dt-recipe"
            class:dt-recipe-on={picked === r.id}
            class:dt-recipe-off={!r.runnable}
            aria-disabled={!r.runnable}
            title={r.runnable ? undefined : r.reason}
            onclick={() => r.runnable && pick(r.id)}
          >
            <span class="mono">{r.id}</span>
            <span class="dt-recipe-model">{r.model}</span>
            {#if !r.runnable}<span class="dt-recipe-why">{r.reason}</span>{/if}
          </button>
        {/each}
      </div>

      {#if picked}
        <SettingsEditor
          schema={fleet.agent.schema}
          defaults={info?.defaults ?? {}}
          bind:overrides
          onchange={() => {
            preview = null;
            confirming = false;
          }}
        />

        <div class="dt-launchrow">
          <button type="button" class="btn btn-secondary dt-act" onclick={doPreview} disabled={busy}>
            {busy && preview === null ? 'Rendering…' : 'Preview'}
          </button>
          <button
            type="button"
            class="btn btn-primary dt-act"
            onclick={doLaunch}
            disabled={busy || preview === null}
            title={preview === null ? 'Preview the exact command first.' : undefined}
          >
            {confirming ? 'Confirm launch' : busy ? 'Working…' : 'Launch'}
          </button>
          {#if confirming}
            <span class="ab-warn">{travel}</span>
            <button type="button" class="dt-cancel" onclick={() => (confirming = false)}>Cancel</button>
          {/if}
        </div>

        {#if preview}
          <p class="dt-cap">This is the command that will run. Nothing else is sent.</p>
          <pre class="dt-cmd mono">{preview}</pre>
          {#if unapplied.length > 0}
            <p class="dt-warn">
              The target does not understand {unapplied.length} setting{unapplied.length === 1
                ? ''
                : 's'} this recipe carries, so they will <strong>not</strong> be applied:
              <code class="mono">{unapplied.join(', ')}</code>. Updating atlasctl may fix this.
            </p>
          {/if}
        {/if}

        {#if problem}
          <p class="dt-problem">{problem}</p>
        {/if}

        {#if started}
          <div class="dt-started">
            <p>
              <strong>{started.recipe}</strong> is starting in
              <code class="mono">{started.container}</code>
              <span class="dt-route-badge">
                on {started.on ? nameOf(started.on, nodes) : 'this machine'}{started.via
                  ? ` · via ${nameOf(started.via, nodes)}`
                  : ''}
              </span>
            </p>
            {#if started.endpoint}
              <p class="dt-endpoint">
                <code class="mono">{started.endpoint}</code>
                <button type="button" class="dt-copy" onclick={copyEndpoint}>
                  {copied ? 'copied' : 'copy'}
                </button>
              </p>
            {:else}
              <p class="dt-cap">The endpoint appears once the model is serving.</p>
            {/if}
          </div>
        {/if}
      {/if}
    {/if}
  {/if}
</div>
