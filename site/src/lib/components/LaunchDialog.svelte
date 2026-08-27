<!-- SPDX-License-Identifier: AGPL-3.0-only -->
<script>
  // The launch dialog, rendered once at the page root.
  //
  // It must live here rather than inside a recipe card: `.subcard` applies a
  // transform on hover, and a transformed ancestor becomes the containing block
  // for `position: fixed`, which made the dialog position against the card and
  // vanish when the hover ended.
  //
  // Most visitors have no agent, so "no agent" is a designed state rather than
  // an error. While that state is shown the dialog keeps probing, so starting
  // the agent in a terminal advances the dialog on its own — the user never has
  // to come back and click retry.

  import { launch } from '$lib/agent/session.svelte.js';
  import { describe } from '$lib/agent/placement.js';
  import { joinCommand } from '$lib/agent/joincommand.js';
  import LaunchModal from './LaunchModal.svelte';

  let tokenInput = $state('');
  let copied = $state('');
  let dialogEl = $state(null);

  const INSTALL = 'curl -fsSL https://atlasinference.io/install.sh | sh';
  // `install`, not `run`: `run` holds the terminal and the agent dies with it.
  const START = 'atlasctl agent install';

  async function copy(text) {
    try {
      await navigator.clipboard.writeText(text);
      copied = text;
      setTimeout(() => { if (copied === text) copied = ''; }, 1600);
    } catch {
      /* clipboard blocked; the command is on screen to select */
    }
  }

  // A loopback connection either answers or is refused within a few
  // milliseconds, so rendering the "looking for your agent" panel the instant
  // the dialog opens puts it on screen and tears it down inside one frame —
  // a flash, not information. Show it only once the attempt has lasted long
  // enough to be worth reporting; below that threshold the dialog simply opens
  // on its real answer.
  const PROBE_VISIBLE_AFTER_MS = 350;
  let showProbe = $state(false);
  $effect(() => {
    if (launch.phase !== 'connecting') {
      showProbe = false;
      return;
    }
    const timer = setTimeout(() => { showProbe = true; }, PROBE_VISIBLE_AFTER_MS);
    return () => clearTimeout(timer);
  });

  // Keep probing while we are waiting for the user to start an agent, so the
  // dialog advances by itself the moment one appears.
  $effect(() => {
    if (launch.phase !== 'guide' || launch.openRecipe === null) return;
    let cancelled = false;
    let delay = 1200;
    const tick = async () => {
      if (cancelled) return;
      await launch.probe();
      if (cancelled || launch.phase !== 'guide') return;
      // Back off so a dialog left open does not poll forever at full rate.
      delay = Math.min(delay * 1.4, 8000);
      timer = setTimeout(tick, delay);
    };
    let timer = setTimeout(tick, delay);
    return () => { cancelled = true; clearTimeout(timer); };
  });

  // Escape closes, and focus moves into the dialog when it opens.
  $effect(() => {
    if (launch.openRecipe === null) return;
    dialogEl?.focus();
    const onKey = (e) => { if (e.key === 'Escape') launch.close(); };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  });

  async function submitToken(e) {
    e.preventDefault();
    if (await launch.pair(tokenInput)) tokenInput = '';
  }
</script>

{#if launch.openRecipe !== null}
  <div class="ld-backdrop" role="presentation" onclick={() => launch.close()}></div>

  {#if launch.phase === 'settings'}
    <LaunchModal
      agent={launch.agent}
      recipeId={launch.openRecipe}
      onclose={() => launch.close()}
      onstarted={(reply) => launch.started(reply)}
    />
  {:else}
    <!-- svelte-ignore a11y_no_noninteractive_element_to_interactive_role -->
    <div
      class="ld"
      role="dialog"
      aria-modal="true"
      aria-labelledby="ld-title"
      tabindex="-1"
      bind:this={dialogEl}
    >
      <header class="ld-head">
        <h3 class="ld-title" id="ld-title">
          {#if launch.phase === 'connecting' && showProbe}Looking for your agent
          {:else if launch.phase === 'connecting'}Run this on your own machine
          {:else if launch.phase === 'guide'}Run this on your own machine
          {:else if launch.phase === 'placement' && launch.placement?.kind === 'none'}Add a machine
          {:else if launch.phase === 'placement'}Where should this run?
          {:else if launch.phase === 'pairing'}Pair this browser
          {:else if launch.phase === 'running'}Running
          {:else}That didn’t work{/if}
        </h3>
        <p class="ld-sub mono">{launch.openRecipe}</p>
        <button type="button" class="ld-close" onclick={() => launch.close()} aria-label="Close">×</button>
      </header>

      <div class="ld-body">
        {#if launch.phase === 'connecting' && showProbe}
          <div class="ld-probe" aria-live="polite">
            <span class="ld-spinner" aria-hidden="true"></span>
            <span>Checking <code class="mono">127.0.0.1:34333</code> …</span>
          </div>

        {:else if launch.phase === 'connecting'}
          <!-- Sub-threshold: hold the frame rather than paint something that is
               about to be replaced. Same height as the guide, so nothing jumps
               when the real answer lands a few milliseconds from now. -->
          <div class="ld-settle" aria-hidden="true"></div>

          {:else if launch.phase === 'placement' && launch.placement?.kind === 'none'}
          <p class="ld-place-lead">{launch.placement.reason}</p>
          <p class="ld-place-lead">
            Add a machine that can. Run this on it — the code is good for one
            machine, once, for {Math.round((launch.join?.expiresInS ?? 600) / 60)} minutes.
          </p>
          {#if launch.join}
            <div class="ld-cmd ld-place-cmd">
              <code class="mono">{joinCommand(launch.join)}</code>
              <button type="button" class="cmd-copy" onclick={() => copy(joinCommand(launch.join))}>
                {copied === joinCommand(launch.join) ? 'Copied' : 'Copy'}
              </button>
            </div>
            <p class="ld-watching">
              <span class="ld-dot" aria-hidden="true"></span>
              Watching for it — this dialog will continue on its own.
            </p>
          {:else}
            <p class="ld-place-sub">
              This agent cannot invite machines. Install the agent on the other
              machine and pair it from the control plane.
            </p>
          {/if}
          <p class="ld-caution">
            Anyone who runs that command joins your fleet and can use its
            hardware. Send it to a machine you own, not a chat.
          </p>

        {:else if launch.phase === 'placement'}
          <p class="ld-place-lead">
            More than one of your machines can run this. Pick one.
          </p>
          <ul class="ld-place">
            {#each launch.placement?.options ?? [] as n (n.id)}
              <li>
                <button type="button" class="ld-place-btn" onclick={() => launch.chooseTarget(n)}>
                  <span class="ld-place-name">{n.name}</span>
                  <span class="ld-place-sub">{describe(n)}</span>
                  {#if n.running}
                    <span class="ld-place-busy">running {n.running}</span>
                  {/if}
                </button>
              </li>
            {/each}
          </ul>

        {:else if launch.phase === 'guide'}
          <p>
            Atlas runs on your hardware, not ours. This page can start a model for
            you once a small local agent is listening.
          </p>
          <ol class="ld-steps">
            <li>
              <span class="ld-step-n">1</span>
              <div>
                <p class="ld-step-t">Install the launcher</p>
                <div class="ld-cmd">
                  <code class="mono">{INSTALL}</code>
                  <button type="button" class="cmd-copy" onclick={() => copy(INSTALL)}>
                    {copied === INSTALL ? 'Copied' : 'Copy'}
                  </button>
                </div>
              </div>
            </li>
            <li>
              <span class="ld-step-n">2</span>
              <div>
                <p class="ld-step-t">Start the agent in the background</p>
                <div class="ld-cmd">
                  <code class="mono">{START}</code>
                  <button type="button" class="cmd-copy" onclick={() => copy(START)}>
                    {copied === START ? 'Copied' : 'Copy'}
                  </button>
                </div>
              </div>
            </li>
          </ol>
          <p class="ld-watching" aria-live="polite">
            <span class="ld-pulse" aria-hidden="true"></span>
            Watching for it — this will continue on its own.
          </p>
          <p class="ld-caution">
            Any web page can show you an install command. Check the address bar
            says <strong>atlasinference.io</strong> before running one.
          </p>

        {:else if launch.phase === 'pairing'}
          <p>
            The agent prints a token when it starts. Paste it once so it knows
            this browser is yours.
          </p>
          <div class="ld-cmd">
            <code class="mono">atlasctl agent token</code>
            <button type="button" class="cmd-copy" onclick={() => copy('atlasctl agent token')}>
              {copied === 'atlasctl agent token' ? 'Copied' : 'Copy'}
            </button>
          </div>
          <form onsubmit={submitToken}>
            <input
              class="mono ld-token"
              bind:value={tokenInput}
              placeholder="64 hexadecimal characters"
              aria-label="Pairing token"
              autocomplete="off"
              spellcheck="false"
            />
            <button type="submit" class="cmd-run" disabled={launch.busy}>
              {launch.busy ? 'Pairing…' : 'Pair'}
            </button>
          </form>
          {#if launch.detail}<p class="ld-error" role="alert">{launch.detail}</p>{/if}

        {:else if launch.phase === 'running'}
          {#if launch.endpoint}
            <p>
              Serving at <code class="mono">{launch.endpoint}</code>. The model
              takes a few minutes to load before it answers.
            </p>
          {:else}
            <p>Started. The model takes a few minutes to load before it answers.</p>
          {/if}

        {:else}
          <p class="ld-error" role="alert">{launch.detail || 'The agent reported a problem.'}</p>
          <button type="button" class="cmd-copy" onclick={() => launch.retry()}>Try again</button>
        {/if}
      </div>
    </div>
  {/if}
{/if}
