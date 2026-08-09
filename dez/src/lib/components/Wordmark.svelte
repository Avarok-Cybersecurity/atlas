<!-- SPDX-License-Identifier: AGPL-3.0-only -->
<script lang="ts">
  interface Props {
    /** Edge length of the glyph in px. */
    size?: number;
  }

  const { size = 26 }: Props = $props();
</script>

<span class="wordmark">
  <svg
    class="mark"
    width={size}
    height={size}
    viewBox="0 0 32 32"
    fill="none"
    aria-hidden="true"
    focusable="false"
  >
    <rect x="0.75" y="0.75" width="30.5" height="30.5" rx="7.25" class="plate" />
    <!-- A bracket opening onto a caret: an editor, running something. -->
    <path d="M11 9.5 L6.5 16 L11 22.5" class="stroke" />
    <path d="M15.5 22.5 L20 9.5" class="stroke dim" />
    <path d="M23.5 13.5 L26 16 L23.5 18.5" class="stroke accent" />
  </svg>
  <span class="name">Dez</span>
</span>

<style>
  .wordmark {
    display: inline-flex;
    align-items: center;
    gap: 0.55rem;
  }

  .plate {
    fill: var(--bg-sunken);
    stroke: var(--border-strong);
    stroke-width: 1.5;
  }

  .stroke {
    stroke: var(--fg);
    stroke-width: 2.1;
    stroke-linecap: round;
    stroke-linejoin: round;
    fill: none;
  }

  .stroke.dim {
    stroke: var(--fg-muted);
    stroke-width: 1.9;
  }

  .stroke.accent {
    stroke: var(--accent);
  }

  .name {
    font-weight: 640;
    font-size: 1.05rem;
    letter-spacing: -0.015em;
  }
</style>
