<!-- SPDX-License-Identifier: AGPL-3.0-only -->
<script>
  // Editing the settings a launch will use.
  //
  // The recipe's own values are the read-only base; this edits a sparse map of
  // differences over them. Only those differences are sent, so a setting this
  // page is too old to render is never overridden and the recipe's value
  // applies on the agent — the skew is survivable instead of silent.
  //
  // Bounds shown here are rendering hints. The agent re-checks every value and
  // its answer is the one that counts; showing a bound the validator does not
  // share is how "it looked fine and then the launch was rejected" happens.
  import * as S from '$lib/agent/schema.js';
  import * as O from '$lib/agent/overrides.js';

  let {
    schema = [],
    defaults = {},
    overrides = $bindable({}),
    onchange = () => {},
    disabled = false,
  } = $props();

  let advanced = $state(false);
  let group = $state('server');
  let errors = $state({});

  const groups = $derived(S.groupsPresent(schema, advanced));
  const shown = $derived(S.settingsIn(schema, group, advanced));
  const changed = $derived(O.changedCount(overrides, defaults));
  // Recipe values with no editor here. They are still applied by the agent —
  // the schema bounds what a *page* may override, not what a recipe may set —
  // so this says "not editable", never "not applied".
  const fixed = $derived(S.notEditableHere(schema, defaults));

  $effect(() => {
    if (groups.length > 0 && !groups.some((g) => g.key === group)) group = groups[0].key;
  });

  function edit(spec, raw) {
    const parsed = O.parse(spec, raw);
    if (parsed.error) {
      errors = { ...errors, [spec.key]: parsed.error };
      return;
    }
    const complaint = S.checkValue(spec, parsed.value);
    errors = { ...errors, [spec.key]: complaint ?? undefined };
    if (complaint) return;
    overrides = O.set(overrides, spec.key, parsed.value, defaults);
    onchange();
  }

  function reset(spec) {
    errors = { ...errors, [spec.key]: undefined };
    overrides = O.clear(overrides, spec.key);
    onchange();
  }
</script>

<div class="se">
  <div class="se-bar">
    <div class="se-tabs" role="tablist" aria-label="Setting groups">
      {#each groups as g (g.key)}
        <button
          role="tab"
          aria-selected={group === g.key}
          class="se-tab"
          class:se-tab-on={group === g.key}
          onclick={() => (group = g.key)}
        >{g.label}</button>
      {/each}
    </div>
    <label class="se-adv">
      <input type="checkbox" bind:checked={advanced} />
      <span>Show advanced</span>
    </label>
  </div>

  <p class="se-count">
    {changed === 0 ? 'Using the recipe’s own settings.' : `${changed} setting${changed === 1 ? '' : 's'} changed from the recipe.`}
  </p>

  {#if shown.length === 0}
    <p class="se-empty">Nothing to set here.</p>
  {/if}

  {#each shown as spec (spec.key)}
    {@const value = O.effective(spec.key, defaults, overrides)}
    {@const isSet = O.isChanged(spec.key, defaults, overrides)}
    <div class="se-row" class:se-row-changed={isSet}>
      <div class="se-label">
        <label for={`set-${spec.key}`}>{spec.label ?? spec.key}</label>
        <code>{spec.key}</code>
        {#if spec.help}<p class="se-help">{spec.help}</p>{/if}
      </div>

      <div class="se-input">
        {#if !S.isEditable(spec)}
          <!-- A kind this page cannot render is shown read-only rather than
               hidden: an agent newer than the site must not silently lose a
               knob the operator can see exists. -->
          <span class="se-ro">{String(value ?? '—')}</span>
          <span class="se-note">editing needs a newer site version</span>
        {:else if spec.bound.kind === 'toggle' || spec.bound.kind === 'bool_value'}
          <input
            id={`set-${spec.key}`}
            type="checkbox"
            {disabled}
            checked={value === true}
            onchange={(e) => edit(spec, e.currentTarget.checked)}
          />
        {:else if spec.bound.kind === 'enum'}
          <select id={`set-${spec.key}`} {disabled} value={String(value ?? '')} onchange={(e) => edit(spec, e.currentTarget.value)}>
            {#each spec.bound.variants as v (v)}<option value={v}>{v}</option>{/each}
          </select>
        {:else}
          <input
            id={`set-${spec.key}`}
            type="text"
            inputmode="decimal"
            {disabled}
            value={String(value ?? '')}
            onchange={(e) => edit(spec, e.currentTarget.value)}
          />
          <span class="se-bounds">
            {spec.bound.kind === 'int_or_auto' ? 'auto, or ' : ''}{spec.bound.min}–{spec.bound.max}
          </span>
        {/if}

        {#if isSet}
          <button class="se-reset" {disabled} onclick={() => reset(spec)}>reset</button>
        {/if}
      </div>

      {#if errors[spec.key]}
        <p class="se-err" role="alert">{errors[spec.key]}</p>
      {/if}
    </div>
  {/each}

  {#if fixed.length > 0}
    <p class="se-fixed" role="status">
      The recipe also sets <code>{fixed.join(', ')}</code>. {fixed.length === 1 ? 'It is' : 'They are'} applied as
      written and cannot be changed from a web page.
    </p>
  {/if}
</div>
