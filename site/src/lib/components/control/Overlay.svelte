<!-- SPDX-License-Identifier: AGPL-3.0-only -->
<script>
  // The shell every bridge overlay shares: centred dialog, scrim, focus trap
  // (modal.js), Esc closes, focus returns to the opener. Overlays are never
  // navigation — closing one always lands exactly where the operator was.
  //
  // The optional `footer` snippet renders OUTSIDE the scrolling body, pinned.
  // That is a spec requirement, not a flourish: a Prepare→Commit ceremony
  // must never scroll its own escape hatch out of view, so the Cluster
  // overlay's Abort lives in this slot and cannot leave the screen.

  import { modal } from '$lib/modal.js';

  let {
    /** Accessible name; also the visible heading. */
    label,
    /** Optional DOM id on the dialog — the Cluster overlay carries #launch. */
    id = undefined,
    wide = false,
    onclose,
    children,
    footer = undefined
  } = $props();

  function onKey(ev) {
    // Window-level, bubble phase: a step change inside can unmount the
    // focused control and drop focus to <body>, where a dialog-scoped
    // listener would go deaf. Bubble phase still lets a popover inside
    // (ComingSoon, HelpDot) stop propagation of the Esc that closes IT, so
    // one press closes one layer.
    if (ev.key === 'Escape') onclose?.();
  }
</script>

<svelte:window onkeydown={onKey} />

<div class="ld-backdrop" role="presentation" onclick={() => onclose?.()}></div>
<!-- svelte-ignore a11y_no_noninteractive_element_to_interactive_role -->
<div
  class="ld ov"
  class:ld-wide={wide}
  role="dialog"
  aria-modal="true"
  aria-label={label}
  tabindex="-1"
  {id}
  use:modal
>
  <header class="ld-head">
    <h3 class="ld-title">{label}</h3>
    <button type="button" class="ld-close" onclick={() => onclose?.()} aria-label="Close">
      ×
    </button>
  </header>

  <!-- The one scroll region; keyboard-reachable like every other one. -->
  <!-- svelte-ignore a11y_no_noninteractive_tabindex -->
  <div class="ld-body ov-body" tabindex="0">
    {@render children()}
  </div>

  {#if footer}
    <footer class="ov-foot">
      {@render footer()}
    </footer>
  {/if}
</div>
