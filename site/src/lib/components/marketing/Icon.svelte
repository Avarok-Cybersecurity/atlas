<!-- SPDX-License-Identifier: AGPL-3.0-only -->
<script>
  import { icons } from './icons.js';
  let { name, size = 18, strokeWidth = 1.8 } = $props();
</script>
<svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width={strokeWidth} stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
  {#each icons[name] as [tag, attributes]}
    <svelte:element this={tag} {...attributes} />
  {/each}
</svg>
