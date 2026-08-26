<!-- SPDX-License-Identifier: AGPL-3.0-only -->
<script>
  // Launch settings for one recipe.
  //
  // The modal shows the recipe's own values as a read-only base and tracks only
  // what the user changed. Start sends that sparse diff, so an untouched modal
  // launches byte-identically to `atlasctl run <recipe>`.
  //
  // The Review tab shows the exact command, rendered by the agent's own
  // renderer — the same function the launch will use. Preview and execution
  // cannot drift, because they are one code path.

  import { groupsPresent, settingsIn, checkValue, notEditableHere } from '$lib/agent/schema.js';
  import SettingField from './SettingField.svelte';

  let { agent, recipeId, onclose, onstarted } = $props();

  const info = $derived(agent.recipes.find((r) => r.id === recipeId));
  const defaults = $derived(info?.defaults ?? {});

  // This agent may be a control node — installed on a laptop to drive headless
  // machines, and structurally unable to run a model itself. Without this the
  // modal offered the whole settings form and a Start button, and the refusal
  // only arrived after the form was filled in. Say it before the work, not
  // after.
  const controlOnly = $derived(agent.canLaunch === false);

  // `.html`, not `/control`: adapter-static writes this route to control.html
  // and the deploy target serves files literally. Same reasoning as FleetPill.
  const CONTROL = '/control.html';

  let overrides = $state({});
  let showAdvanced = $state(false);
  let tab = $state('server');
  let preview = $state(null);
  let unapplied = $state([]);
  let starting = $state(false);
  let failure = $state('');

  const tabs = $derived([...groupsPresent(agent.schema, showAdvanced), { key: 'review', label: 'Review' }]);
  const changedCount = $derived(Object.keys(overrides).length);
  // Not editable from a page is not the same as not applied: the schema
  // bounds what a web page may override, and the agent applies the recipe's own
  // defaults through its flag table regardless. `host` is the clear case — it is
  // deliberately absent from the schema, and `host: 0.0.0.0` still reaches the
  // command line.
  const fixedByRecipe = $derived(notEditableHere(agent.schema, defaults));

  // Every invalid field at once, so a form can be fixed in one pass.
  const invalid = $derived(
    Object.entries(overrides)
      .map(([key, value]) => {
        const spec = agent.schema.find((s) => s.key === key);
        return spec ? { key, message: checkValue(spec, value) } : null;
      })
      .filter((x) => x && x.message)
  );

  function valueOf(spec) {
    return spec.key in overrides ? overrides[spec.key] : defaults[spec.key];
  }

  function setValue(key, value) {
    if (value === undefined) {
      const { [key]: _drop, ...rest } = overrides;
      overrides = rest;
    } else {
      overrides = { ...overrides, [key]: value };
    }
    preview = null;
  }

  async function refreshPreview() {
    const result = await agent.preview(recipeId, overrides);
    if (result.ok) {
      preview = result.reply.command;
      unapplied = result.reply.unapplied ?? [];
      failure = '';
    } else {
      preview = null;
      failure = result.message;
    }
  }

  $effect(() => {
    if (tab === 'review' && preview === null && !failure) refreshPreview();
  });

  async function start() {
    starting = true;
    failure = '';
    const result = await agent.launch(recipeId, overrides);
    starting = false;
    if (result.ok) {
      onstarted?.(result.reply);
    } else {
      failure = result.message;
      // The agent is authoritative, so send the user back to fix the fields it
      // named rather than leaving them on a Review tab that now lies.
      if (result.error?.code === 'bad_settings') tab = 'server';
    }
  }
</script>

<div class="lm" role="dialog" aria-modal="true" aria-label={`Launch settings for ${recipeId}`}>
  <header class="lm-head">
    <div>
      <h3 class="lm-title mono">{recipeId}</h3>
      {#if info}<p class="lm-sub mono">{info.model}</p>{/if}
    </div>
    <button type="button" class="lm-close" onclick={onclose} aria-label="Close">×</button>
  </header>

  {#if controlOnly}
    <div class="lm-body">
      <p class="lm-co-lead">
        <span class="fl-co-chip">Control only</span>
        This machine cannot run models, so there is nothing to configure here.
      </p>
      <p class="lm-co-why">
        {agent.canLaunchReason || 'The agent on this machine reports it cannot run models.'}
      </p>
      <p class="lm-co-why">
        It can still drive machines that do. Pair one from the control plane and
        launch <code class="mono">{recipeId}</code> onto it from there.
      </p>
    </div>
    <footer class="lm-foot">
      <button type="button" class="cmd-copy" onclick={onclose}>Close</button>
      <a class="cmd-run lm-co-go" href={CONTROL}>Open the control plane</a>
    </footer>
  {:else}
  <div class="lm-tabs" role="tablist">
    {#each tabs as t (t.key)}
      <button
        type="button"
        role="tab"
        aria-selected={tab === t.key}
        class="lm-tab"
        class:lm-tab-on={tab === t.key}
        onclick={() => (tab = t.key)}
      >{t.label}</button>
    {/each}
  </div>

  <div class="lm-body">
    {#if tab === 'review'}
      <p class="set-help">This is the command that will run. Nothing else is sent.</p>
      {#if preview}
        <pre class="lm-cmd mono">{preview}</pre>
      {:else if failure}
        <p class="set-error">{failure}</p>
      {:else}
        <p class="set-help">Rendering…</p>
      {/if}

      {#if changedCount > 0}
        <h4 class="lm-h4">Changed from the recipe</h4>
        <ul class="lm-diff mono">
          {#each Object.entries(overrides) as [key, value] (key)}
            <li>{key}: {String(defaults[key] ?? '—')} → {String(value)}</li>
          {/each}
        </ul>
      {/if}

      {#if unapplied.length > 0}
        <p class="lm-warn">
          Your agent does not understand {unapplied.length} setting(s) this recipe
          carries, so they will <strong>not</strong> be applied:
          <code class="mono">{unapplied.join(', ')}</code>.
          Updating atlasctl may fix this.
        </p>
      {/if}
    {:else}
      {#each settingsIn(agent.schema, tab, showAdvanced) as spec (spec.key)}
        <SettingField
          {spec}
          value={valueOf(spec)}
          isDefault={!(spec.key in overrides)}
          onchange={(v) => setValue(spec.key, v)}
        />
      {/each}
      {#if fixedByRecipe.length > 0 && tab === 'server'}
        <p class="lm-warn">
          The recipe also sets <code class="mono">{fixedByRecipe.join(', ')}</code>.
          {fixedByRecipe.length === 1 ? 'It is' : 'They are'} applied as written and cannot be changed from a
          web page.
        </p>
      {/if}
    {/if}
  </div>

  <footer class="lm-foot">
    <label class="lm-adv">
      <input type="checkbox" bind:checked={showAdvanced} /> Show advanced
    </label>
    <span class="lm-count">{changedCount} changed</span>
    {#if invalid.length > 0}
      <span class="set-error">{invalid.length} invalid</span>
    {/if}
    <button type="button" class="cmd-copy" onclick={onclose}>Cancel</button>
    <button
      type="button"
      class="cmd-run"
      onclick={start}
      disabled={starting || invalid.length > 0}
    >{starting ? 'Starting…' : 'Start'}</button>
  </footer>
  {/if}
</div>
