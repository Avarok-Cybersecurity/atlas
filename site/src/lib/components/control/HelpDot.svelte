<!-- SPDX-License-Identifier: AGPL-3.0-only -->
<script>
  // A "?" that explains something without spending permanent screen space on it.
  //
  // A button, not a `title=` and not a hover-only div. Three reasons, all of
  // them things that break for real people: `title` never appears on a
  // touchscreen and is unstyleable; hover-only never appears for anyone
  // navigating by keyboard; and a non-focusable tooltip is invisible to a
  // screen reader. This opens on hover, on focus, and on click, and closes on
  // Escape or on leaving.
  //
  // The panel is `role="note"` rather than `role="tooltip"` because it holds a
  // paragraph of guidance rather than a label for the control it hangs off.

  let { label = 'What is this?', children } = $props();

  let open = $state(false);
  let id = `help-${Math.random().toString(36).slice(2, 9)}`;

  function onKey(e) {
    if (e.key === 'Escape' && open) {
      open = false;
      e.stopPropagation();
    }
  }
</script>

<span
  class="hd"
  onmouseenter={() => (open = true)}
  onmouseleave={() => (open = false)}
  role="presentation"
>
  <button
    type="button"
    class="hd-btn"
    aria-label={label}
    aria-expanded={open}
    aria-describedby={open ? id : undefined}
    onclick={() => (open = !open)}
    onfocus={() => (open = true)}
    onblur={() => (open = false)}
    onkeydown={onKey}
  >
    ?
  </button>
  {#if open}
    <span class="hd-panel" {id} role="note">
      {@render children?.()}
    </span>
  {/if}
</span>
