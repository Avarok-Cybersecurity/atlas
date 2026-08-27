<!-- SPDX-License-Identifier: AGPL-3.0-only -->
<script>
  // "Show me how" — the card that gets a GPU machine into the fleet.
  //
  // Inline and persistent rather than a modal, because the operator physically
  // WALKS AWAY in the middle of this: the next step happens on a different
  // computer. They must be able to leave, come back, and find the same state —
  // a modal that closed while they were gone loses the code, and an operator
  // who has to mint a second one has learned that the first was a trap.
  //
  // Behind a button rather than always open, because opening it mints a
  // one-use credential and every unattended control page would otherwise be
  // sitting on a live one.
  //
  // Every rule is in `joinwindow.js` (plain `.js`, tested). This file renders
  // whichever state that module names.

  import { tick } from 'svelte';
  import { joinState } from '$lib/agent/joinstate.svelte.js';
  import { nowMs, useClock } from '$lib/agent/clock.svelte.js';
  import { joinCommand } from '$lib/agent/joincommand.js';
  import * as JW from '$lib/agent/joinwindow.js';
  import CommandRow from '../CommandRow.svelte';
  import InstallSteps from '../InstallSteps.svelte';

  let { fleet } = $props();

  let open = $state(false);
  let heading = $state(null);
  let shownAt = $state(0);
  let arrived = $state(null);
  let detailsOpen = $state(false);
  let touchedDetails = $state(false);

  const join = $derived(joinState.current);
  const command = $derived(join ? joinCommand(join) : '');
  const kind = $derived(JW.offerKind(join));
  const left = $derived(
    join ? JW.remaining(JW.deadlineMs(join.mintedAtMs, join.expiresInS), nowMs()) : null
  );
  const stalled = $derived(shownAt > 0 && JW.stalled(shownAt, nowMs()));

  // Hold the shared clock only while the card is open, so a closed guide costs
  // no timer at all.
  $effect(() => {
    if (open) return useClock();
  });

  // The command's arrival time is what the stall clock measures from — not the
  // mint, which can precede it when an offer is reused after a collapse.
  $effect(() => {
    if (open && command && shownAt === 0) shownAt = Date.now();
    if (!open) shownAt = 0;
  });

  // Auto-open the troubleshooting once, unless the operator has already made
  // their own choice about it. Overriding that would be the page arguing.
  $effect(() => {
    if (stalled && !touchedDetails) detailsOpen = true;
  });

  // The moment a peer appears, say so — the operator may have walked back to a
  // page that otherwise looks exactly as they left it.
  $effect(() => {
    const peers = fleet.peers;
    if (open && !arrived && peers.length > 0) arrived = peers[0];
  });

  async function toggle() {
    open = !open;
    if (!open) return;
    // Reuse a live offer: re-minting on every open would invalidate the command
    // the operator may already be carrying to the other machine.
    if (!join || left?.expired) await joinState.mint(fleet.agent);
    // The card is rendered by this state change, so the heading does not exist
    // until Svelte has flushed. Focusing before that silently does nothing —
    // and a keyboard operator is then left where they were, with a card that
    // opened somewhere below them.
    await tick();
    heading?.focus();
  }

  async function remint() {
    shownAt = 0;
    arrived = null;
    await joinState.mint(fleet.agent);
  }

  async function cancel() {
    await joinState.revoke(fleet.agent);
    shownAt = 0;
  }
</script>

<p class="fl-co-next">
  <strong>Next:</strong> add the machine with the GPU. One command, run on that machine,
  installs the agent, starts it, and pairs it back to here.
  <button
    type="button"
    class="btn jg-toggle"
    aria-expanded={open}
    aria-controls="join-guide"
    onclick={toggle}
  >
    {open ? 'Hide the guide' : 'Show me how'}
  </button>
</p>

{#if open}
  <div class="jg" id="join-guide">
    <h3 class="jg-h" tabindex="-1" bind:this={heading}>Add your GPU machine</h3>

    {#if arrived}
      <p class="fs-good" role="status">
        {arrived.name || 'The machine'} joined your fleet. It is in the panel above.
      </p>
    {:else}
      <p class="jg-lead">
        The next step happens on that machine, not this one. Everything you need to
        carry over is on this card.
      </p>

      {#if kind === 'command' && left && !left.expired}
        <ol class="ld-steps">
          <li>
            <span class="ld-step-n">1</span>
            <div>
              <p class="ld-step-t">Open a terminal on the GPU machine</p>
              <p class="jg-body">
                SSH to it from here if you can: you'll see what the installer says
                without walking back. Walking over works too.
              </p>
            </div>
          </li>
          <li>
            <span class="ld-step-n">2</span>
            <div>
              <p class="ld-step-t">Paste this line there</p>
              <CommandRow {command} />
              <p class="jg-body">
                It installs the agent, starts it in the background, and pairs the
                machine back to this fleet — all three. Watch that terminal until it
                finishes: if the join fails, the error prints there, not here.
              </p>
              <p class="jg-facts">
                Code <span class="mono">{JW.groupedCode(join.code)}</span> · good for one
                machine, once · expires in <span class="mono">{left.label}</span>
                <button type="button" class="jg-link" onclick={cancel}>Cancel this code</button>
              </p>
              {#if left.warning}
                <p class="jg-warn">
                  This code expires in {left.label}. Not at the machine yet? Mint a fresh
                  one from this page when you're ready — minting again costs nothing.
                </p>
              {/if}
              <p class="jg-body">
                Can't paste on that machine? Only the tail of the line is yours:
                <span class="mono">{JW.shortForm(join)}</span> — eight digits and this
                machine's address. Short enough to photograph, read aloud, or retype;
                the rest is the standard installer from this site.
              </p>
            </div>
          </li>
          <li>
            <span class="ld-step-n">3</span>
            <div>
              <p class="ld-step-t">Come back to this page</p>
              <p class="jg-body">
                Nothing to click. The machine appears in the panel above on its own. You
                can close this tab while you go; the invitation lives in the agent on
                this machine, not in the browser.
              </p>
            </div>
          </li>
        </ol>

        <p class="ld-watching" aria-live="polite">
          <span class="ld-pulse" aria-hidden="true"></span>
          {#if stalled}
            Still watching. If you already ran the command, one of the causes below is
            usually why — the far machine's terminal has the exact error.
          {:else}
            Watching for the new machine — this page updates on its own. While the
            install runs, the progress is in that machine's terminal, not here.
          {/if}
        </p>
      {:else if kind === 'command' && left?.expired}
        <p class="jg-expired" role="status">
          This code expired — nothing ran, and there is nothing to undo on that machine.
          Codes go stale quickly on purpose, so one pasted in the wrong place dies fast.
        </p>
        <button type="button" class="btn btn-primary" onclick={remint}>Mint a new code</button>
      {:else if kind === 'no_address'}
        <p class="jg-body">
          This machine has no network address another machine could dial — only loopback
          or virtual interfaces are up. Connect it to the network you want the fleet on,
          then reopen this guide. Failing that, install the agent by hand:
        </p>
        <InstallSteps />
      {:else}
        <p class="jg-body">
          This agent cannot hand out an invitation. Install the agent on the other
          machine and pair it from the form below.
        </p>
        <InstallSteps />
      {/if}

      <p class="ld-caution">
        Anyone who runs that line joins your fleet and can use its hardware. Send it to a
        machine you own, not a chat. It is fine in that machine's shell history
        afterwards — the code works once, then it is dead.
      </p>
      <p class="ctl-safety">
        Any web page can show you an install command. Check the address bar says
        <strong>atlasinference.io</strong> before running one.
      </p>

      <details class="jg-tshoot" bind:open={detailsOpen} ontoggle={() => (touchedDetails = true)}>
        <summary>Ran it and nothing appeared?</summary>
        <ul>
          <li>Give it a minute — the installer downloads the agent before it can pair.</li>
          <li>
            The GPU machine has to be able to reach this one
            {#if JW.dialHost(join)}
              at <span class="mono">{JW.dialHost(join)}:{JW.JOIN_PORT}</span>{/if}. A
            different network, a VPN, or a firewall on that port blocks the join — the
            install still succeeds and only the pairing step fails, with the error in that
            machine's terminal. If the two machines can't see each other, add it by
            address instead, using the form below this card.
          </li>
          <li>
            If the code expired before you ran the line, mint a fresh one here and run the
            new command there. The old line is dead; nothing to clean up.
          </li>
        </ul>
      </details>
    {/if}
  </div>
{/if}
