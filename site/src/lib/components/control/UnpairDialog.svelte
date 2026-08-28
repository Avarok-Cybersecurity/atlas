<!-- SPDX-License-Identifier: AGPL-3.0-only -->
<script>
  // The unpair confirmation, moved verbatim out of +page.svelte when the
  // bridge shell landed. The rules are unchanged: the operator types the
  // fingerprint prefix, and a refusal keeps the dialog open — closing it
  // regardless used to leave a machine trusted while the interface implied it
  // had been removed, the worst of the three possible outcomes.

  import { modal } from '$lib/modal.js';

  let { fleet, node, onclose } = $props();

  let confirm = $state('');
  let error = $state('');
  let el = $state(null);

  const ready = $derived(confirm.trim().toLowerCase() === node.id.slice(0, 8));

  // Declaring aria-modal="true" and then leaving the dialog unfocused with no
  // Escape handler tells an assistive-technology user they are in a modal and
  // gives them no way out.
  // Focus in, Tab trap and focus-return live in modal.js (use:modal below).
  $effect(() => {
    const onKey = (ev) => {
      if (ev.key === 'Escape') onclose?.();
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  });

  async function doUnpair() {
    if (!ready) return;
    error = '';
    const res = await fleet.unpair(node.id);
    if (!res.ok) {
      error = res.detail || 'The agent refused to remove this pairing.';
      return;
    }
    onclose?.();
  }
</script>

<div class="ld-backdrop" role="presentation" onclick={() => onclose?.()}></div>
<!-- svelte-ignore a11y_no_noninteractive_element_to_interactive_role -->
<div
  class="ld"
  role="dialog"
  aria-modal="true"
  aria-labelledby="unpair-title"
  tabindex="-1"
  bind:this={el}
  use:modal
>
  <header class="ld-head">
    <h3 class="ld-title" id="unpair-title">Unpair {node.name}?</h3>
    <button type="button" class="ld-close" onclick={() => onclose?.()} aria-label="Close">×</button>
  </header>
  <div class="ld-body">
    <p>
      {node.name} will stop trusting this machine, any cluster launch that
      includes it will be stopped, and pairing again needs someone at that machine
      to read a new code.
    </p>
    <p class="unpair-confirm-note">
      Type <code class="mono">{node.id.slice(0, 8)}</code> to confirm.
    </p>
    <input
      class="pair-code mono"
      bind:value={confirm}
      aria-label="Type the fingerprint prefix to confirm"
      placeholder={node.id.slice(0, 8)}
    />
    {#if error}
      <p class="ld-error" role="alert">{error}</p>
    {/if}
    <div class="ld-actions">
      <button type="button" class="btn btn-ghost" onclick={() => onclose?.()}>Cancel</button>
      <button type="button" class="btn btn-danger" disabled={!ready} onclick={doUnpair}>
        Unpair
      </button>
    </div>
  </div>
</div>
