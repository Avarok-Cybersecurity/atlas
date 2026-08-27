<!-- SPDX-License-Identifier: AGPL-3.0-only -->
<script>
  // The control plane.
  //
  // Prerendered in its no-agent state, which is what most visitors get, what
  // crawlers get, and what Lighthouse measures. It also means the shipped HTML
  // contains no fleet data at all — the privacy property falls out of the
  // architecture rather than being maintained by hand.
  //
  // Hydration then probes ws://127.0.0.1:34333 and the page advances in place,
  // the same way the launch dialog does. A background probe never repaints what
  // the reader is looking at; it may only move the page forward.

  import Nav from '$lib/components/Nav.svelte';
  import Footer from '$lib/components/Footer.svelte';
  import SectionHead from '$lib/components/SectionHead.svelte';
  import NodeCard from '$lib/components/control/NodeCard.svelte';
  import ClusterLaunch from '$lib/components/control/ClusterLaunch.svelte';
  import TopologyMap from '$lib/components/control/TopologyMap.svelte';
  import PairDialog from '$lib/components/control/PairDialog.svelte';
  import NodeDetails from '$lib/components/control/NodeDetails.svelte';
  import { fleet } from '$lib/agent/fleet.svelte.js';
  import { runCommand } from '$lib/data.js';

  // `install`, not `run`. `run` holds the terminal and the agent dies with it,
  // which turns a fleet into a demo: close the window and the page this
  // command was meant to light up goes dark again.
  const START_AGENT = 'atlasctl agent install';

  let pairing = $state(null);
  let details = $state(null);
  let unpairing = $state(null);
  let unpairConfirm = $state('');
  let head = $state(null);
  let copied = $state('');

  const nodes = $derived(fleet.nodes);
  const solo = $derived(fleet.mode === 'live' && fleet.peers.length === 0);
  const remoteCount = $derived(fleet.remoteLaunchable.length);

  async function copy(text) {
    try {
      await navigator.clipboard.writeText(text);
      copied = text;
      setTimeout(() => {
        if (copied === text) copied = '';
      }, 1600);
    } catch {
      /* clipboard blocked; the command is on screen */
    }
  }

  // Start only. The session is an app-wide singleton the nav indicator shares,
  // so tearing it down when this effect re-runs would kill a connection some
  // other caller is still using — and did: the page connected, then lost its
  // event listener mid-open and rendered an empty fleet.
  $effect(() => {
    fleet.start({ watch: true });
  });

  // Default the head to this machine — but only if this machine can actually
  // hold rank 0. On a control-only node it cannot, and defaulting to it drew
  // the laptop as rank 0 in the topology while its own "Make head" button sat
  // disabled, which is a picture of a cluster that could never start.
  $effect(() => {
    if (head !== null) return;
    const first = fleet.launchable;
    if (first.length === 0) return;
    head = (fleet.local?.canLaunch ? fleet.local : first[0]).id;
  });

  // If the machine acting as head stops being able to hold a rank — it was
  // unpaired, or it went away — the selection is stale and must not linger.
  $effect(() => {
    if (head === null) return;
    if (!fleet.launchable.some((n) => n.id === head)) head = null;
  });

  const unpairReady = $derived(
    unpairing ? unpairConfirm.trim().toLowerCase() === unpairing.id.slice(0, 8) : false
  );

  /** Set when an unpair was refused, so the dialog can say so. */
  let unpairError = $state('');

  async function doUnpair() {
    if (!unpairReady) return;
    unpairError = '';
    const res = await fleet.unpair(unpairing.id);
    if (!res.ok) {
      // The dialog used to close here regardless. The node stayed trusted and
      // the interface implied it had been removed, which is the worst of the
      // three possible outcomes: the operator believes a machine is out of
      // their fleet when it is still in it.
      unpairError = res.detail || 'The agent refused to remove this pairing.';
      return;
    }
    unpairing = null;
    unpairConfirm = '';
  }
</script>

<svelte:head>
  <title>Control plane — Atlas</title>
  <meta
    name="description"
    content="Manage the Atlas agents on your own machines: node health, pairing, topology and multi-node launches. Everything stays on your LAN."
  />
</svelte:head>

<Nav />

<main class="control">
  <section id="fleet" class="sx-cyan">
    <div class="container">
      <SectionHead
        level={1}
        label="// 01 · fleet"
        title="Your machines, one panel."
        sub="This page talks only to an agent on the computer you are using. That agent finds the others. Nothing here leaves your network, and none of it is in the page you downloaded."
        prov={fleet.mode === 'live' ? 'local agent · 127.0.0.1:34333' : 'no agent connected'}
      />

      <div class="ctl-modes">
      {#if fleet.mode === 'live'}
        {#if fleet.alerts.length > 0}
          <div class="al-strip al-{fleet.worstSeverity}" role="status">
            <strong>{fleet.alerts[0].nodeName}:</strong>
            {fleet.alerts[0].detail || fleet.alerts[0].kind.replaceAll('_', ' ')}
            {#if fleet.alerts.length > 1}
              <a href="#alerts">and {fleet.alerts.length - 1} more</a>
            {/if}
          </div>
        {/if}

        <div class="fl-grid">
          {#each nodes as node (node.id)}
            <NodeCard
              {node}
              ondetails={(n) => (details = n)}
              onpair={(n) => (pairing = n)}
              onunpair={(n) => {
                unpairing = n;
                unpairConfirm = '';
              }}
            />
          {/each}
        </div>

        {#if fleet.controlOnly}
          <div class="fl-control-only" role="status">
            <p class="fl-co-head">
              <span class="fl-co-chip">Control only</span>
              This machine drives the fleet; it does not run models itself.
            </p>
            <!-- The agent's own reason is on the card above; repeating it
                 here verbatim read as a stutter. This says what it means. -->
            <p class="fl-co-why">
              {#if remoteCount === 0}
                Pair a machine that can run models and everything on this page —
                topology, launching, alerts — applies to it.
              {:else}
                Everything on this page — topology, launching, alerts — applies
                to the {remoteCount === 1 ? 'machine' : `${remoteCount} machines`}
                you have paired.
              {/if}
            </p>
            {#if remoteCount === 0}
              <p class="fl-co-next">
                <strong>Next:</strong> install the agent on a machine with a GPU and
                pair it. It will appear here on its own once it is running.
              </p>
            {/if}
          </div>
        {/if}

        {#if solo && !fleet.controlOnly}
          <p class="fl-solo-note">
            No peers yet. Start an agent on another machine on this network and it
            will appear here on its own — then pair it to unlock the EP=2 recipes,
            which need exactly two nodes.
          </p>
        {/if}
      {:else if fleet.mode === 'reconnecting'}
        <!-- The agent was here a moment ago and went away — a restart, a
             reboot, an ssh session closing. Falling through to the "install the
             agent" invitation below would tell someone who plainly HAS one to
             go and get one. -->
        <div class="ctl-setup">
          <h2>Lost the agent</h2>
          <p>
            The connection to the local agent dropped. This page is trying again on
            its own, and will pick up where it left off as soon as the agent answers.
          </p>
          <p class="ld-watching">Reconnecting…</p>
          <p>
            If it does not come back, the agent may have stopped. Check it with
            <code class="mono">atlasctl agent status</code>, or start it again with
            <code class="mono">{START_AGENT}</code>.
          </p>
        </div>
      {:else if fleet.mode === 'browser_unpaired'}
        <div class="ctl-setup">
          <h2>Pair this browser with your agent</h2>
          <p>
            An agent is running, but it has not seen this browser before. Run
            <code class="mono">atlasctl agent token</code> and paste the value the
            launch dialog asks for. This is separate from pairing machines to each
            other.
          </p>
        </div>
      {:else}
        <!-- Prerendered. Most visitors see this, and it must read as an
             invitation rather than an error. -->
        <div class="ctl-setup">
          <h2>Nothing is running here yet</h2>
          <p>
            Atlas runs on your hardware, not ours. Install the agent on a machine
            and this page becomes its control panel.
          </p>
          <ol class="ld-steps">
            <li>
              <span class="ld-step-n">1</span>
              <div>
                <p class="ld-step-t">Install the launcher</p>
                <div class="ld-cmd">
                  <code class="mono">{runCommand}</code>
                  <button type="button" class="cmd-copy" onclick={() => copy(runCommand)}>
                    {copied === runCommand ? 'Copied' : 'Copy'}
                  </button>
                </div>
              </div>
            </li>
            <li>
              <span class="ld-step-n">2</span>
              <div>
                <p class="ld-step-t">Start the agent in the background</p>
                <div class="ld-cmd">
                  <code class="mono">{START_AGENT}</code>
                  <button type="button" class="cmd-copy" onclick={() => copy(START_AGENT)}>
                    {copied === START_AGENT ? 'Copied' : 'Copy'}
                  </button>
                </div>
              </div>
            </li>
          </ol>
          <p class="ld-watching">
            <span class="ld-dot" aria-hidden="true"></span>
            Watching for it — this page will continue on its own.
          </p>
          <p class="ctl-safety">
            Any web page can show you an install command. Check the address bar says
            <strong>atlasinference.io</strong> before running one.
          </p>
        </div>
      {/if}
      </div>
    </div>
  </section>

  <section id="topology" class="section-alt sx-cyan">
    <div class="container">
      <SectionHead
        label="// 02 · topology"
        title="How they reach each other."
        sub="Multi-node decode is all-reduce bound, so the link between two machines decides the throughput. A cluster that falls back to ethernet still runs — several times slower — while every correctness check keeps passing, so the fabric is called out here rather than left to be discovered in a benchmark."
      />

      <div class="topo-wrap">
        <TopologyMap {nodes} {head} />

        <div class="topo-actions">
          <h3 class="topo-actions-title">Nodes</h3>
          {#each nodes as node (node.id)}
            <div class="topo-act-group">
              <p class="topo-act-name">
                {node.name}
                <span class="mono topo-act-fp">{node.id.slice(0, 8)}</span>
              </p>
              {#if node.isLocal || node.pairing === 'paired'}
                {#if node.canLaunch}
                  <button
                    type="button"
                    class="topo-act-btn"
                    disabled={head === node.id}
                    onclick={() => (head = node.id)}
                  >
                    {head === node.id ? 'Head (rank 0)' : 'Make head'}
                  </button>
                {:else}
                  <!-- A disabled "Make head" reads as something broken. Say
                       what this machine is instead: rank 0 serves the API, so
                       a machine that cannot run models can never hold it. -->
                  <span class="topo-act-note">Control only — cannot hold a rank</span>
                {/if}
                {#if !node.isLocal}
                  <button
                    type="button"
                    class="topo-act-btn topo-act-danger"
                    onclick={() => {
                      unpairing = node;
                      unpairConfirm = '';
                    }}>Unpair…</button
                  >
                {/if}
              {:else}
                <button type="button" class="topo-act-btn" onclick={() => (pairing = node)}>
                  Pair…
                </button>
              {/if}
            </div>
          {:else}
            <p class="topo-act-empty">No machines yet.</p>
          {/each}
        </div>
      </div>
    </div>
  </section>

  <section id="launch" class="sx-cyan">
    <div class="container">
      <SectionHead
        label="// 03 · launch"
        title="Run one model across them."
        sub="Two phases, because one cannot fail cleanly. Every machine checks it can run this and holds its place; nothing starts until all of them have agreed. If one refuses, the reservations the others took are released — so a cluster is either whole or absent, never a half that hangs waiting on a rendezvous."
      />

      {#if fleet.mode === 'live'}
        <ClusterLaunch {fleet} />
      {:else}
        <p class="lc-offline">Connect an agent to launch anything.</p>
      {/if}
    </div>
  </section>

  <section id="alerts" class="section-alt sx-cyan">
    <div class="container">
      <SectionHead
        label="// 04 · alerts"
        title="What needs looking at."
        sub="Idle machines matter as much as busy ones. A clamped clock, a failing fan or a full cache filesystem is something to know before a launch, not after a benchmark comes back wrong."
      />

      {#if fleet.alerts.length === 0}
        <h3 class="al-empty-title">Nothing to report</h3>
        <p class="al-empty">
          No alerts. This section stays here so you never have to wonder where they
          would appear.
        </p>
      {:else}
        <h3 class="visually-hidden">Active alerts</h3>
        <ul class="al-list">
          {#each fleet.alerts as a (a.node + a.kind)}
            <li class="al-row">
              <span class="al-sev al-{a.severity}">{a.severity}</span>
              <span class="al-node">{a.nodeName}</span>
              <span class="al-kind">{a.kind.replaceAll('_', ' ')}</span>
              <span class="al-detail">{a.detail}</span>
            </li>
          {/each}
        </ul>
      {/if}
    </div>
  </section>
</main>

<Footer />

{#if details}
  <NodeDetails node={details} onclose={() => (details = null)} />
{/if}

{#if pairing}
  <PairDialog node={pairing} onclose={() => (pairing = null)} />
{/if}

{#if unpairing}
  <div class="ld-backdrop" role="presentation" onclick={() => { unpairing = null; unpairError = ''; }}></div>
  <!-- svelte-ignore a11y_no_noninteractive_element_to_interactive_role -->
  <div class="ld" role="dialog" aria-modal="true" aria-labelledby="unpair-title" tabindex="-1">
    <header class="ld-head">
      <h3 class="ld-title" id="unpair-title">Unpair {unpairing.name}?</h3>
      <button type="button" class="ld-close" onclick={() => { unpairing = null; unpairError = ''; }} aria-label="Close"
        >×</button
      >
    </header>
    <div class="ld-body">
      <p>
        {unpairing.name} will stop trusting this machine, any cluster launch that
        includes it will be stopped, and pairing again needs someone at that machine
        to read a new code.
      </p>
      <p class="unpair-confirm-note">
        Type <code class="mono">{unpairing.id.slice(0, 8)}</code> to confirm.
      </p>
      <input
        class="pair-code mono"
        bind:value={unpairConfirm}
        aria-label="Type the fingerprint prefix to confirm"
        placeholder={unpairing.id.slice(0, 8)}
      />
      {#if unpairError}
        <p class="ld-error" role="alert">{unpairError}</p>
      {/if}
      <div class="ld-actions">
        <button type="button" class="btn btn-ghost" onclick={() => { unpairing = null; unpairError = ''; }}>Cancel</button>
        <button type="button" class="btn btn-danger" disabled={!unpairReady} onclick={doUnpair}>
          Unpair
        </button>
      </div>
    </div>
  </div>
{/if}
