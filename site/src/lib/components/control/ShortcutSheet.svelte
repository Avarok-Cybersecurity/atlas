<!-- SPDX-License-Identifier: AGPL-3.0-only -->
<script>
  // The "?" sheet: the keyboard map, rendered from the same table the
  // dispatcher runs (shortcuts.js), so the sheet can never document a key
  // the page does not honour.

  import Overlay from './Overlay.svelte';
  import { SHORTCUTS } from '$lib/agent/shortcuts.js';

  let { onclose } = $props();
</script>

<Overlay label="Keyboard shortcuts" {onclose}>
  <dl class="ks-list">
    {#each SHORTCUTS as s (s.keys)}
      <dt class="ks-keys mono">{s.keys}</dt>
      <dd class="ks-what">{s.label}</dd>
    {/each}
  </dl>
  <p class="ks-note">
    Everything here is equally reachable by click and touch — the keys are a
    faster door, never the only one.
  </p>
</Overlay>
