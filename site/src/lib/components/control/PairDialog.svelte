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
  import { modal } from '$lib/modal.js';

  let { node, onclose } = $props();

  /** 'code' | 'verifying' | 'confirm' | 'accepting' | 'rejecting' | 'paired' | 'failed' */
  let phase = $state('code');
  let code = $state('');
  let detail = $state('');
  let verification = $state('');
  /** Whether the trusted peer may drive this machine. Off until said. */
  let allowControl = $state(false);
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
    const res = await fleet.confirm(node.id, allowControl);
    if (res.ok) {
      phase = 'paired';
      onclose?.(true);
      return;
    }
    detail = res.detail || 'The agent did not accept the pairing.';
    phase = 'confirm';
  }

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
      // They walked away mid-ceremony. Nothing is written — under protocol 2 the
      // exchange holds no pin — but the exchange IS live on the agent, and the
      // operator never saw the words. The only honest reading of a dismissal is
      // that they did not approve it, so the exchange is spent explicitly rather
      // than left to lapse: relying on a socket closing is relying on a side
      // effect, and they said no.
      if (dismissed) {
        dismissed = false;
        void reject();
      }
    } else {
      detail = res.detail || 'The code was not accepted.';
      phase = 'failed';
    }
  }

  // Focus in, Tab trap and focus-return live in modal.js (use:modal below);
  // Esc stays here because in this dialog it is a rejection, not a dismissal.
  $effect(() => {
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
  use:modal
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
            <p class="ld-step-t">On {node.name}, open its control page and mint a code</p>
            <!-- NOT `atlasctl agent pair`. That command binds the peer port,
                 and {node.name}'s agent is already holding it — which is the
                 only reason this machine can see it at all. The code has to
                 come from the running agent, and its control page is what asks
                 for one. -->
            <p class="pair-hint">
              Use “Show me how” there, or the add-a-machine panel — either opens a
              join window and shows the eight digits.
            </p>
          </div>
        </li>
        <li>
          <span class="ld-step-n">2</span>
          <div>
            <p class="ld-step-t">Type those eight digits here</p>
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
          {node.name} logs these same words when it accepts. If they differ, something
          is sitting between your machines. Cancel and nothing is trusted; no pairing
          was written.
        </p>
      </div>

      <p class="pair-consequence">
        If you confirm, this machine will trust {node.name} and can launch on it
        wherever it has granted control. You can undo it later with Unpair.
      </p>

      <label class="jg-grant">
        <input
          type="checkbox"
          bind:checked={allowControl}
          disabled={phase === 'rejecting' || phase === 'accepting'}
        />
        <span>
          Let {node.name} control this machine.
          <span class="jg-grant-why">
            Ticked, it can launch and stop models here. Unticked, control runs one
            way — from here toward it.
          </span>
        </span>
      </label>

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
          <!-- Not "Removing…": nothing was written, so there is nothing to
               remove. Saying otherwise implies a pin existed, which is exactly
               the pre-protocol-2 behaviour this dialog stopped having. -->
          {phase === 'rejecting' ? 'Discarding…' : "They differ — cancel"}
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
