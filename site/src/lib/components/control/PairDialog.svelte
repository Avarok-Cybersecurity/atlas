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
  // When trust actually happens, as of protocol 2. `pair_peer` runs the SPAKE2
  // exchange and writes NOTHING. The words are derived from that exchange, so
  // this is the earliest they can exist — and now the pin does not exist yet
  // either. Confirming writes it; every other way out of this step discards it.
  //
  // That is why the copy can finally say "not trusted yet" and mean it. Before
  // this, the pin was written first and the dialog could only undo — so a
  // refusal was a REMOVAL that could itself fail, leaving a machine trusted
  // that the operator had explicitly rejected.

  import { fleet } from '$lib/agent/fleet.svelte.js';

  let { node, onclose } = $props();

  /** 'code' | 'verifying' | 'confirm' | 'accepting' | 'rejecting' | 'paired' | 'failed' */
  let phase = $state('code');
  let code = $state('');
  let detail = $state('');
  let verification = $state('');
  let dialogEl = $state(null);
  /** Set when the operator dismissed while the exchange was still in flight. */
  let dismissed = $state(false);

  /**
   * Accept the words and trust the node.
   *
   * The pin is written HERE, by the agent, on this click — which is the whole
   * point of the two-phase change. A failure leaves nothing trusted, and the
   * dialog stays open to say so rather than closing on an unkept promise.
   */
  async function accept() {
    if (phase !== 'confirm') return;
    phase = 'accepting';
    const res = await fleet.confirm(node.id);
    if (res.ok) {
      phase = 'paired';
      onclose?.(true);
      return;
    }
    detail = res.detail || 'The agent did not accept the pairing.';
    phase = 'confirm';
  }

  const START = 'atlasctl agent pair';
  const digitsOnly = $derived(code.replaceAll(/\D/g, '').slice(0, 8));
  const ready = $derived(digitsOnly.length === 8);

  /**
   * Leave the confirm step without trusting the node.
   *
   * Discards the exchange. Nothing was written, so unlike the old unpair-based
   * refusal there is nothing that can fail and leave the peer trusted.
   */
  async function reject() {
    // Dismissing while the exchange is still running: `pair_peer` is in flight
    // and will complete. Under protocol 2 it writes nothing, so the pending
    // exchange would die with the socket anyway — but the dismissal is still
    // remembered and turned into an explicit reject when the reply lands,
    // because relying on a socket closing is relying on a side effect, and the
    // operator said no.
    if (phase === 'verifying') {
      dismissed = true;
      return;
    }
    // A decision is already in flight. Re-entering would send a second one for
    // an exchange the agent has already taken, and the reply would say "no
    // exchange waiting" — true, confusing, and entirely self-inflicted.
    if (phase === 'rejecting' || phase === 'accepting') return;
    if (phase !== 'confirm') {
      onclose?.();
      return;
    }
    phase = 'rejecting';
    const res = await fleet.reject(node.id);
    if (res.ok) {
      onclose?.();
      return;
    }
    // The agent could not even discard it. Nothing was trusted either way —
    // say that plainly rather than implying the operator must go clean up.
    detail =
      `The agent did not acknowledge the refusal: ${res.detail || 'no reason given'}. ` +
      `${node.name} was not trusted.`;
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
      // They walked away mid-ceremony. The pin exists now, and they never saw
      // the words, so the only honest reading is that they did not approve it.
      if (dismissed) {
        dismissed = false;
        void reject();
      }
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
      {#if phase === 'confirm' || phase === 'rejecting' || phase === 'accepting'}Confirm you are joining the right machines
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
    {:else if phase === 'confirm' || phase === 'rejecting' || phase === 'accepting'}
      <p>
        The code was accepted. <strong>{node.name}</strong> is <strong>not trusted
        yet</strong> — check that it is showing the same words first.
      </p>

      <div class="pair-fps">
        <div class="pair-words mono">{verification || '—'}</div>
        <p class="pair-fp-note">
          <code class="mono">{START}</code> on {node.name} is printing these same words.
          If they differ, something is sitting between your machines. Cancel and
          nothing is trusted; no pairing was written.
        </p>
      </div>

      <p class="pair-consequence">
        If you confirm, {node.name} can run models on this fleet and this machine can
        run models on it. You can undo it later with Unpair.
      </p>

      {#if detail}
        <p class="ld-error" role="alert">{detail}</p>
      {/if}

      <div class="ld-actions">
        <button
          type="button"
          class="btn btn-ghost"
          disabled={phase === 'rejecting' || phase === 'accepting'}
          onclick={() => void reject()}
        >
          {phase === 'rejecting' ? 'Removing…' : "They differ — cancel"}
        </button>
        <button
          type="button"
          class="btn btn-primary"
          disabled={phase === 'rejecting' || phase === 'accepting'}
          onclick={() => void accept()}
        >
          {phase === 'accepting' ? 'Trusting…' : 'They match — trust this node'}
        </button>
      </div>
    {/if}
  </div>
</div>
