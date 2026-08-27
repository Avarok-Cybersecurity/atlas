<!-- SPDX-License-Identifier: AGPL-3.0-only -->
<script>
  // The trust ceremony.
  //
  // The code always originates on the TARGET machine. This browser never
  // generates one and there is no protocol verb that would let it — that is
  // what stops a hostile page pairing anything on its own, and it is why the
  // copy tells the user to go and run a command on the other box.
  //
  // The fingerprint comparison is the part people skip, so it is the part with
  // the most room on screen and the plainest consequence attached to it.
  //
  // IMPORTANT about when trust actually happens. `pair_peer` completes the
  // SPAKE2 ceremony and writes the pin BEFORE these words can be shown — the
  // words are derived from the exchange, so there is no earlier moment to show
  // them. The dialog therefore cannot ask "may I pair?"; it can only ask "was
  // that the right machine?" and undo it if the answer is no. Every path out of
  // the confirm step that is not the confirm button unpairs, because a screen
  // that says "if they differ, cancel" while cancel does nothing is worse than
  // no ceremony at all.

  import { fleet } from '$lib/agent/fleet.svelte.js';

  let { node, onclose } = $props();

  /** 'code' | 'verifying' | 'confirm' | 'rejecting' | 'paired' | 'failed' */
  let phase = $state('code');
  let code = $state('');
  let detail = $state('');
  let verification = $state('');
  let dialogEl = $state(null);

  const START = 'atlasctl agent pair';
  const digitsOnly = $derived(code.replaceAll(/\D/g, '').slice(0, 8));
  const ready = $derived(digitsOnly.length === 8);

  /**
   * Leave the confirm step without trusting the node.
   *
   * Removes the pin the ceremony already wrote. If that fails the dialog stays
   * open and says so rather than closing on a promise it did not keep.
   */
  async function reject() {
    if (phase !== 'confirm') {
      onclose?.();
      return;
    }
    phase = 'rejecting';
    const res = await fleet.unpair(node.id);
    if (res.ok) {
      onclose?.();
      return;
    }
    detail =
      `${node.name} could not be un-paired: ${res.detail || 'the agent refused'}. ` +
      'It is still trusted — remove it with Unpair before using this fleet.';
    phase = 'confirm';
  }

  const short = (id, n = 4) =>
    id ? `${id.slice(0, n)} ${id.slice(n, 8)} ${id.slice(8, 12)} ${id.slice(12, 16)}` : '';

  async function submit(e) {
    e?.preventDefault();
    if (!ready || phase === 'verifying') return;
    phase = 'verifying';
    detail = '';
    const res = await fleet.pair(node.id, digitsOnly);
    if (res.ok) {
      verification = res.verification ?? '';
      phase = 'confirm';
    } else {
      detail = res.detail || 'The code was not accepted.';
      phase = 'failed';
    }
  }

  $effect(() => {
    dialogEl?.focus();
    const onKey = (ev) => {
      if (ev.key === 'Escape') void reject();
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  });
</script>

<div class="ld-backdrop" role="presentation" onclick={() => void reject()}></div>
<!-- svelte-ignore a11y_no_noninteractive_element_to_interactive_role -->
<div
  class="ld ld-wide"
  role="dialog"
  aria-modal="true"
  aria-labelledby="pair-title"
  tabindex="-1"
  bind:this={dialogEl}
>
  <header class="ld-head">
    <h3 class="ld-title" id="pair-title">
      {#if phase === 'confirm' || phase === 'rejecting'}Confirm you are joining the right machines
      {:else if phase === 'paired'}Paired
      {:else}Pair {node.name}{/if}
    </h3>
    <p class="ld-sub mono">{node.id.slice(0, 16)}</p>
    <button type="button" class="ld-close" onclick={() => void reject()} aria-label="Close">×</button>
  </header>

  <div class="ld-body">
    {#if phase === 'code' || phase === 'verifying' || phase === 'failed'}
      <p>
        Pairing proves you control both machines. The code is shown on
        <strong>{node.name}</strong>, so only someone who can reach that machine can
        join it to your fleet.
      </p>

      <ol class="ld-steps">
        <li>
          <span class="ld-step-n">1</span>
          <div>
            <p class="ld-step-t">On {node.name}, run</p>
            <div class="ld-cmd"><code class="mono">{START}</code></div>
          </div>
        </li>
        <li>
          <span class="ld-step-n">2</span>
          <div>
            <p class="ld-step-t">Type the eight digits it prints</p>
            <form onsubmit={submit}>
              <input
                class="pair-code mono"
                inputmode="numeric"
                autocomplete="one-time-code"
                aria-label="Pairing code, eight digits"
                placeholder="1234 5678"
                bind:value={code}
                disabled={phase === 'verifying'}
              />
              <button type="submit" class="btn btn-primary" disabled={!ready || phase === 'verifying'}>
                {phase === 'verifying' ? 'Checking…' : 'Pair'}
              </button>
            </form>
          </div>
        </li>
      </ol>

      {#if phase === 'failed'}
        <p class="ld-error" role="alert">{detail}</p>
      {/if}
    {:else if phase === 'confirm' || phase === 'rejecting'}
      <p>
        The code was accepted and <strong>{node.name}</strong> is paired. Check that it
        is showing the same words. If it is not, cancel and the pairing is removed.
      </p>

      <div class="pair-fps">
        <div class="pair-words mono">{verification || '—'}</div>
        <p class="pair-fp-note">
          <code class="mono">{START}</code> on {node.name} is printing these same words.
          If they differ, something is sitting between your machines — cancel, and
          this node is un-paired again.
        </p>
      </div>

      <p class="pair-consequence">
        Once paired, {node.name} can run models on this fleet and this machine can run
        models on it. You can undo it at any time with Unpair.
      </p>

      {#if detail}
        <p class="ld-error" role="alert">{detail}</p>
      {/if}

      <div class="ld-actions">
        <button
          type="button"
          class="btn btn-ghost"
          disabled={phase === 'rejecting'}
          onclick={() => void reject()}
        >
          {phase === 'rejecting' ? 'Removing…' : "They differ — cancel"}
        </button>
        <button
          type="button"
          class="btn btn-primary"
          disabled={phase === 'rejecting'}
          onclick={() => {
            phase = 'paired';
            onclose?.(true);
          }}
        >
          They match — trust this node
        </button>
      </div>
    {/if}
  </div>
</div>
