<!-- SPDX-License-Identifier: AGPL-3.0-only -->
<script>
  // A capability the page admits it does not have yet.
  //
  // The placeholder grammar (the fourth honesty class): dashed outline, no
  // fill, a `soon` chip, and NO value — not even an em-dash, because the dash
  // means "this hardware cannot answer", a different fact. Which capabilities
  // may appear, and what each one's missing piece is, lives in the tested
  // registry (`placeholders.js`); this file only draws one entry and runs the
  // popover ceremony.
  //
  // The whole tile/chip is a real <button aria-haspopup="dialog">: click,
  // Enter, Space and tap all open the popover; Esc and click-out close it;
  // focus returns to the button. Never title=-only, never hover-only — a
  // tooltip nobody can reach on a touch screen is a secret, not an
  // explanation.

  import { placeholder } from '$lib/agent/placeholders.js';

  let {
    /** A registered placeholder id — unknown ids throw in `placeholder()`. */
    id,
    /** 'chip' (actions bar, command strip, alert lane) or 'tile' (I/O strip). */
    kind = 'chip'
  } = $props();

  const entry = $derived(placeholder(id));

  let open = $state(false);
  let btn = $state(null);
  let pop = $state(null);
  let closeBtn = $state(null);

  function close(refocus) {
    if (!open) return;
    open = false;
    // Focus returns to the button — except on click-out, where the operator
    // has already put their attention somewhere else and yanking it back
    // would fight them.
    if (refocus) btn?.focus();
  }

  $effect(() => {
    if (!open) return;
    pop?.focus();
    const onDown = (ev) => {
      if (pop?.contains(ev.target) || btn?.contains(ev.target)) return;
      close(false);
    };
    window.addEventListener('pointerdown', onDown, true);
    return () => window.removeEventListener('pointerdown', onDown, true);
  });

  function onBtnKey(ev) {
    if (open && ev.key === 'Escape') {
      ev.stopPropagation();
      close(true);
    }
  }

  function onPopKey(ev) {
    if (ev.key === 'Escape') {
      ev.stopPropagation();
      close(true);
    } else if (ev.key === 'Tab') {
      // The dialog holds exactly one focusable control, so the trap is a
      // single stop: Tab in any direction lands on Close.
      ev.preventDefault();
      closeBtn?.focus();
    }
  }
</script>

<span class="cs-wrap">
  <button
    type="button"
    class="cs-btn cs-{kind}"
    aria-haspopup="dialog"
    aria-expanded={open}
    bind:this={btn}
    onclick={() => (open ? close(true) : (open = true))}
    onkeydown={onBtnKey}
  >
    <span class="cs-label">{entry.label}</span>
    <span class="cs-chip">soon</span>
  </button>

  {#if open}
    <div
      class="cs-pop"
      role="dialog"
      aria-label="{entry.label} — coming soon"
      tabindex="-1"
      bind:this={pop}
      onkeydown={onPopKey}
    >
      <p class="cs-text">{entry.soon}</p>
      <button type="button" class="cs-close" bind:this={closeBtn} onclick={() => close(true)}>
        Close
      </button>
    </div>
  {/if}
</span>
