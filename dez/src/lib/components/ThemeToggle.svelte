<!-- SPDX-License-Identifier: AGPL-3.0-only -->
<script lang="ts">
  import { theme } from '$lib/theme.svelte';

  $effect(() => theme.init());

  const next = $derived(theme.current === 'dark' ? 'light' : 'dark');
</script>

<button
  type="button"
  class="toggle"
  aria-pressed={theme.current === 'light'}
  aria-label="Switch to {next} theme"
  title="Switch to {next} theme"
  onclick={() => theme.toggle()}
>
  <svg viewBox="0 0 24 24" aria-hidden="true" focusable="false" width="18" height="18">
    {#if theme.current === 'dark'}
      <path
        d="M20 14.2A8.2 8.2 0 0 1 9.8 4a8.4 8.4 0 1 0 10.2 10.2Z"
        fill="none"
        stroke="currentColor"
        stroke-width="1.7"
        stroke-linejoin="round"
      />
    {:else}
      <g fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round">
        <circle cx="12" cy="12" r="4.2" />
        <path d="M12 2.6v2.2M12 19.2v2.2M2.6 12h2.2M19.2 12h2.2M5.4 5.4l1.6 1.6M17 17l1.6 1.6M18.6 5.4 17 7M7 17l-1.6 1.6" />
      </g>
    {/if}
  </svg>
</button>

<style>
  .toggle {
    display: inline-grid;
    place-items: center;
    width: 2.35rem;
    height: 2.35rem;
    border: 1px solid var(--border);
    border-radius: var(--radius-sm);
    background: var(--bg-raised);
    color: var(--fg-muted);
    cursor: pointer;
    transition:
      color 0.15s var(--ease),
      border-color 0.15s var(--ease),
      background-color 0.15s var(--ease);
  }

  .toggle:hover {
    color: var(--fg);
    border-color: var(--border-strong);
  }
</style>
