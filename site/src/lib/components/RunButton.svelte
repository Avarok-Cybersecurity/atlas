<!-- SPDX-License-Identifier: AGPL-3.0-only -->
<script>
  // The Run button on a recipe card.
  //
  // Deliberately thin: it asks the page-level session to open, and owns no
  // dialog of its own. A dialog rendered here would sit inside `.subcard`,
  // which applies a transform on hover — and a transformed ancestor becomes the
  // containing block for `position: fixed`, so the dialog would position
  // against the card rather than the viewport.
  import { launch } from '$lib/agent/session.svelte.js';

  let { recipeId, runnable = true } = $props();

  const isOpen = $derived(launch.openRecipe === recipeId);
  const isConnecting = $derived(isOpen && launch.busy);
</script>

<button
  type="button"
  class="cmd-run"
  onclick={() => launch.open(recipeId)}
  disabled={!runnable || isConnecting}
  aria-haspopup="dialog"
  aria-expanded={isOpen}
  title={runnable
    ? 'Run this recipe on your own machine'
    : 'This recipe needs more than one machine — use the command instead'}
>
  {#if isConnecting}Checking…{:else}Run{/if}
</button>
