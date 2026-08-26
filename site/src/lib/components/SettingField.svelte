<!-- SPDX-License-Identifier: AGPL-3.0-only -->
<script>
  // One editable setting.
  //
  // Rendered entirely from the agent's schema — there is no per-key code here,
  // which is what lets an agent newer than this page still present its settings.
  import { checkValue, isEditable } from '$lib/agent/schema.js';

  let { spec, value, isDefault, onchange } = $props();

  const error = $derived(value === undefined ? null : checkValue(spec, value));
  const editable = $derived(isEditable(spec));

  function num(raw) {
    const n = Number(raw);
    return Number.isNaN(n) ? raw : n;
  }
</script>

<div class="set-row" class:set-changed={!isDefault}>
  <label class="set-label" for={`set-${spec.key}`}>
    {spec.label}
    {#if !isDefault}<span class="set-dot" title="changed from the recipe's value">●</span>{/if}
  </label>

  <div class="set-control">
    {#if !editable}
      <!-- A bound kind this page version cannot render. Shown, not hidden: the
           user should know the setting exists and will use its default. -->
      <code class="mono">{String(value)}</code>
      <span class="set-note">this page cannot edit this setting; its default applies</span>
    {:else if spec.bound.kind === 'toggle' || spec.bound.kind === 'bool_value'}
      <input
        id={`set-${spec.key}`}
        type="checkbox"
        checked={value === true}
        onchange={(e) => onchange(e.currentTarget.checked)}
      />
    {:else if spec.bound.kind === 'enum'}
      <select id={`set-${spec.key}`} value={String(value)} onchange={(e) => onchange(e.currentTarget.value)}>
        {#each spec.bound.variants as v (v)}
          <option value={v}>{v}</option>
        {/each}
      </select>
    {:else if spec.bound.kind === 'int_or_auto'}
      <input
        id={`set-${spec.key}`}
        class="mono"
        value={String(value)}
        onchange={(e) => {
          const raw = e.currentTarget.value.trim();
          onchange(raw === 'auto' ? 'auto' : num(raw));
        }}
      />
    {:else}
      <input
        id={`set-${spec.key}`}
        class="mono"
        type="number"
        value={value}
        min={spec.bound.min}
        max={spec.bound.max}
        step={spec.bound.kind === 'float' ? 0.01 : 1}
        onchange={(e) => onchange(num(e.currentTarget.value))}
      />
    {/if}
    {#if spec.unit}<span class="set-unit">{spec.unit}</span>{/if}
    {#if !isDefault}
      <button type="button" class="set-reset" onclick={() => onchange(undefined)}>reset</button>
    {/if}
  </div>

  <p class="set-help">{spec.help}</p>
  {#if error}<p class="set-error">{error}</p>{/if}
</div>
