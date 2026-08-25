<!-- SPDX-License-Identifier: AGPL-3.0-only -->
<script>
  // The Run button on a recipe card.
  //
  // It only ever names a recipe. It cannot send an image, a command, a flag, or
  // a path — the agent's message surface has nowhere to put one — so the worst
  // this button can do is start a recipe the agent already ships.

  import { AgentClient } from '$lib/agent/client.svelte.js';
  import { looksLikeToken, storeToken } from '$lib/agent/protocol.js';
  import LaunchModal from './LaunchModal.svelte';

  let { recipeId, runnable = true } = $props();

  const agent = new AgentClient();

  // 'idle' | 'probing' | 'guide' | 'pairing' | 'settings' | 'launching' | 'running' | 'failed'
  let phase = $state('idle');
  let detail = $state('');
  let endpoint = $state('');
  let tokenInput = $state('');

  async function onRun() {
    if (phase === 'running') return;
    phase = 'probing';
    detail = '';

    const connected = await agent.connect();
    if (!connected) {
      // Every failure to connect is explained rather than reported as an error:
      // most visitors simply do not have the agent yet.
      if (agent.phase === 'unpaired') {
        phase = 'pairing';
      } else if (agent.phase === 'error') {
        phase = 'failed';
        detail = agent.message;
      } else {
        phase = 'guide';
      }
      return;
    }
    await start();
  }

  async function submitToken(event) {
    event.preventDefault();
    const token = tokenInput.trim();
    if (!looksLikeToken(token)) {
      detail = 'That does not look like a pairing token. It is 64 hex characters.';
      return;
    }
    storeToken(token);
    tokenInput = '';
    phase = 'probing';
    detail = '';
    if (await agent.connect(token)) {
      await start();
    } else {
      phase = agent.phase === 'unpaired' ? 'pairing' : 'failed';
      detail = agent.message;
    }
  }

  async function start() {
    if (!agent.runnable(recipeId)) {
      phase = 'failed';
      detail = agent.canLaunch
        ? 'Your agent does not have this recipe, or it needs more than one machine. Update atlasctl, or use the command above.'
        : (agent.canLaunchReason ?? 'That machine cannot launch recipes.');
      return;
    }
    // Settings first, and a chance to read the command before it runs.
    phase = 'settings';
  }

  function onStarted(reply) {
    endpoint = reply.endpoint ?? '';
    phase = 'running';
  }

  async function onStop() {
    const result = await agent.stop(recipeId);
    phase = result.ok ? 'ready' : 'failed';
    if (!result.ok) detail = result.message;
  }

  function dismiss() {
    phase = 'idle';
    detail = '';
  }
</script>

<button
  type="button"
  class="cmd-run"
  onclick={onRun}
  disabled={!runnable || phase === 'probing' || phase === 'launching'}
  title={runnable
    ? 'Run this recipe on your machine'
    : 'This recipe needs more than one machine; use the command instead'}
>
  {#if phase === 'probing'}Checking…
  {:else if phase === 'launching'}Starting…
  {:else if phase === 'running'}Running
  {:else}Run{/if}
</button>

{#if phase === 'settings'}
  <LaunchModal
    {agent}
    {recipeId}
    onclose={() => (phase = 'idle')}
    onstarted={onStarted}
  />
{/if}

{#if phase !== 'idle' && phase !== 'probing' && phase !== 'settings'}
  <div class="run-panel" role="status">
    {#if phase === 'guide'}
      <p class="run-panel-title">No local agent found</p>
      <p>
        Atlas runs on your machine, not ours. Install the launcher, then start the
        agent:
      </p>
      <pre class="mono">curl -fsSL https://atlasinference.io/install.sh | sh
atlasctl agent run</pre>
      <p class="run-panel-note">
        Any web page can show you an install command. Check the address bar says
        <strong>atlasinference.io</strong> before running it.
      </p>
      <button type="button" class="cmd-copy" onclick={onRun}>Try again</button>
    {:else if phase === 'pairing'}
      <p class="run-panel-title">Pair this browser</p>
      <p>
        The agent prints a token when it starts. Paste it once, so it knows this
        browser is yours.
      </p>
      <pre class="mono">atlasctl agent token</pre>
      <form onsubmit={submitToken}>
        <!-- svelte-ignore a11y_autofocus -->
        <input
          class="mono run-token"
          bind:value={tokenInput}
          placeholder="64 hex characters"
          aria-label="Pairing token"
          autocomplete="off"
          spellcheck="false"
        />
        <button type="submit" class="cmd-copy">Pair</button>
      </form>
      {#if detail}<p class="run-panel-error">{detail}</p>{/if}
    {:else if phase === 'running'}
      <p class="run-panel-title">Running</p>
      {#if endpoint}
        <p>
          Serving at <code class="mono">{endpoint}</code>. The model takes a few
          minutes to load before it answers.
        </p>
      {/if}
      <button type="button" class="cmd-copy" onclick={onStop}>Stop</button>
    {:else if phase === 'failed'}
      <p class="run-panel-title">That did not work</p>
      <p class="run-panel-error">{detail}</p>
      <button type="button" class="cmd-copy" onclick={dismiss}>Dismiss</button>
    {/if}
  </div>
{/if}
