<script>
  // The one tablist. Roving tabindex: only the active tab is a Tab stop, so a
  // strip costs exactly one stop inside the dialog's focus trap and the arrow
  // keys move within it. Selection follows focus (automatic activation, the
  // choice GatePointCard already made). Left/Right/Home/End are the whole
  // vocabulary — Up/Down stay unbound so nested strips never feel like one
  // list, and Escape/Tab fall through to the dialog untouched.
  //
  // `prefix` names both the element ids (`{prefix}-tab-{id}`, controlling
  // `{prefix}-panel-{id}`) and the CSS classes (`{prefix}-tabs`, `{prefix}-tab`,
  // `{prefix}-chip`), so the outer strip keeps its existing `bd-*` rules and
  // each nested strip gets its own size step.
  //
  // A div, not <nav>: app.css styles the bare nav element (position:fixed).
  import { moveTab } from '$lib/tablist.js';

  /** @type {{ tabs: Array<{id:string,label:string,chip?:string}>, active: string, label: string, prefix: string }} */
  let { tabs, active = $bindable(), label, prefix } = $props();

  let listEl = $state(null);

  function onkeydown(e) {
    const from = tabs.findIndex((t) => t.id === active);
    const to = moveTab(e.key, from, tabs.length);
    if (to === null) return;
    e.preventDefault();
    active = tabs[to].id;
    listEl?.querySelectorAll('[role="tab"]')[to]?.focus();
  }
</script>

<div class="{prefix}-tabs" role="tablist" aria-label={label} bind:this={listEl}>
  {#each tabs as t (t.id)}
    <button
      type="button"
      role="tab"
      id="{prefix}-tab-{t.id}"
      aria-controls="{prefix}-panel-{t.id}"
      aria-selected={t.id === active}
      tabindex={t.id === active ? 0 : -1}
      class="{prefix}-tab"
      class:is-active={t.id === active}
      onclick={() => (active = t.id)}
      {onkeydown}
    >
      {t.label}
      <!-- Inside the button, so the chip is part of the accessible name and
           is read when the tab is reached by arrow key. -->
      {#if t.chip}<span class="{prefix}-chip">{t.chip}</span>{/if}
    </button>
  {/each}
</div>
