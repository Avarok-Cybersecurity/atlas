<!-- SPDX-License-Identifier: AGPL-3.0-only -->
<script>
  // The control plane, as one surface: the bridge.
  //
  // Prerendered in its no-agent state, which is what most visitors get, what
  // crawlers get, and what Lighthouse measures. It also means the shipped HTML
  // contains no fleet data at all — the privacy property falls out of the
  // architecture rather than being maintained by hand.
  //
  // Hydration then probes ws://127.0.0.1:34333 and the page advances in place.
  // Once the agent answers, the marketing chrome steps aside and the viewport
  // becomes the bridge: command strip across the top, roster left, stage
  // center, rail right — one screen, no page scroll at desktop widths. The
  // no-agent, lost-agent and unpaired-browser states keep their full-page
  // invitations, because a person with no fleet needs prose, not panels.
  //
  // This file is composition only: which panel goes where, and which machine
  // is selected. Every rule — row order, hotkeys, what happens when the
  // selected machine vanishes — lives in selection.js, where it is tested.

  import { browser } from '$app/environment';
  import { replaceState } from '$app/navigation';
  import Nav from '$lib/components/Nav.svelte';
  import Footer from '$lib/components/Footer.svelte';
  import SectionHead from '$lib/components/SectionHead.svelte';
  import InstallSteps from '$lib/components/InstallSteps.svelte';
  import CommandStrip from '$lib/components/control/CommandStrip.svelte';
  import Roster from '$lib/components/control/Roster.svelte';
  import NodeStage from '$lib/components/control/NodeStage.svelte';
  import FleetRail from '$lib/components/control/FleetRail.svelte';
  import NodeDetails from '$lib/components/control/NodeDetails.svelte';
  import PairDialog from '$lib/components/control/PairDialog.svelte';
  import UnpairDialog from '$lib/components/control/UnpairDialog.svelte';
  import PairingOverlay from '$lib/components/control/PairingOverlay.svelte';
  import ClusterOverlay from '$lib/components/control/ClusterOverlay.svelte';
  import ShortcutSheet from '$lib/components/control/ShortcutSheet.svelte';
  import TopologyMap from '$lib/components/control/TopologyMap.svelte';
  import { fleet } from '$lib/agent/fleet.svelte.js';
  import { fromHash, reselect, selectByKey, toHash } from '$lib/agent/selection.js';
  import { shortcut } from '$lib/agent/shortcuts.js';
  import { storedToken } from '$lib/agent/protocol.js';
  import { CADENCES, isPaused } from '$lib/agent/cadence.js';
  import { StatsPoller } from '$lib/agent/statspoll.svelte.js';
  import * as A from '$lib/agent/actionlog.js';
  import * as L from '$lib/agent/launch.js';
  import { startAgentCommand } from '$lib/data.js';

  // `install`, not `run`. `run` holds the terminal and the agent dies with it,
  // which turns a fleet into a demo: close the window and the page this
  // command was meant to light up goes dark again.

  let pairing = $state(null);
  let details = $state(null);
  let unpairing = $state(null);
  let head = $state(null);
  let pairingOpen = $state(false);
  let clusterOpen = $state(false);
  let sheetOpen = $state(false);

  // The cluster flow lives HERE, not in the overlay: closing the overlay is
  // not abandoning a prepare, so the reservations survive it, the rail's D3
  // summary reads the same state, and reopening lands mid-ceremony.
  let clusterFlow = $state(L.initial());

  // The dock tab is page state so the keyboard map can reach it.
  let tab = $state('launch');
  let stageEl = $state(null);

  // ---- telemetry ---------------------------------------------------------

  // 2s: fast enough to feel live, half the load of 1s on a relayed fleet.
  // The operator owns it from the command strip; it is a starting position,
  // not a policy.
  let cadenceId = $state('2s');
  let vitalsOn = $state(true);
  const cadenceMs = $derived(CADENCES.find((c) => c.id === cadenceId).ms);
  const paused = $derived(isPaused(cadenceMs));

  // The one stats poller. Schedule rules live in cadence.js; this page only
  // tells it what the fleet looks like and which node is selected.
  const poller = new StatsPoller(fleet.agent);
  $effect(() => {
    if (fleet.mode !== 'live') return;
    poller.start();
    return () => poller.stop();
  });
  $effect(() => {
    poller.configure(
      fleet.nodes.map((n) => ({
        id: n.id,
        selected: n.id === selectedId,
        running: n.running !== null,
        recipe: n.running,
        on: n.isLocal ? null : n.id
      })),
      cadenceMs
    );
  });

  // The session action log: what this page asked the fleet to do. Page
  // memory, dead with the tab — the Status tab says so.
  let log = $state([]);
  function record(fields) {
    log = A.append(log, A.entry(fields, Date.now()));
  }

  /** Whether a connection to the local agent has been attempted yet. */
  let attempted = $state(false);

  // Start only. The session is an app-wide singleton the nav indicator shares,
  // so tearing it down when this effect re-runs would kill a connection some
  // other caller is still using.
  //
  // **Only if this browser has paired before.** Opening a loopback socket makes
  // the browser ask for "access other apps and services on this device", and
  // asking that of someone who has just arrived is asking for a permission
  // that is not yet needed. A stored token is proof this browser was paired,
  // so re-dialing prompts nobody. Without one, the page renders its install
  // invitation and waits for the operator to press Connect below.
  $effect(() => {
    if (attempted || !storedToken()) return;
    attempted = true;
    fleet.start({ watch: true });
  });

  /** Dial the local agent because the operator asked. */
  function connectNow() {
    attempted = true;
    fleet.start({ watch: true });
  }

  // ---- selection ----------------------------------------------------------

  let selectedId = $state(null);
  /** A #node= hash captured at arrival, honoured once its machine appears. */
  let wantedHash = browser ? location.hash : '';

  // The pre-bridge #launch anchor: deep links and the @live e2e spec target
  // it, and the panel it named now lives in the cluster overlay — so the
  // anchor opens the overlay instead of scrolling to a section.
  if (wantedHash === '#launch') {
    clusterOpen = true;
    wantedHash = '';
  }

  $effect(() => {
    const nodes = fleet.nodes;
    if (wantedHash) {
      const id = fromHash(wantedHash, nodes);
      if (id) {
        wantedHash = '';
        selectedId = id;
        return;
      }
    }
    // The reselect rule: keep a valid selection, else local, else first row.
    selectedId = reselect(nodes, selectedId);
  });

  // Persist by node id, never by index — the roster reorders as machines come
  // and go, and "the third row" is a different machine after every reorder.
  $effect(() => {
    if (!browser || fleet.mode !== 'live') return;
    const h = toHash(selectedId);
    replaceState(h || location.pathname + location.search, {});
  });

  const selectedNode = $derived(fleet.nodes.find((n) => n.id === selectedId) ?? null);
  const select = (id) => (selectedId = id);

  // ---- keyboard map -------------------------------------------------------

  // Which key does what — and when a key must do nothing — is shortcuts.js's
  // tested rule; this handler only carries out the verdict. Arrows are
  // deliberately not here: they rove only while the roster has focus
  // (Roster.svelte), so they never steal scrolling from the dock or rail.
  let lastCadence = '2s';

  function onKeydown(e) {
    if (fleet.mode !== 'live') return;
    // The roster's own handler (arrows, 1–8 while focused) runs first and
    // prevents default; dispatching again would double-move the selection.
    if (e.defaultPrevented) return;
    const t = e.target;
    const typing = Boolean(
      t instanceof Element && t.closest('input, textarea, select, [contenteditable="true"]')
    );
    const overlayOpen = Boolean(
      pairingOpen || clusterOpen || sheetOpen || pairing || unpairing || details
    );
    const act = shortcut(e.key, {
      typing,
      overlayOpen,
      modified: e.ctrlKey || e.metaKey || e.altKey
    });
    if (!act) return;
    e.preventDefault();
    if (act.kind === 'select') {
      const id = selectByKey(fleet.nodes, act.key);
      if (id) selectedId = id;
    } else if (act.kind === 'tab') {
      tab = act.tab;
    } else if (act.kind === 'stop') {
      stageEl?.armStop();
    } else if (act.kind === 'alerts') {
      document.getElementById('alert-lane')?.focus();
    } else if (act.kind === 'cluster') {
      clusterOpen = true;
    } else if (act.kind === 'pause') {
      if (cadenceId === 'pause') {
        cadenceId = lastCadence;
      } else {
        lastCadence = cadenceId;
        cadenceId = 'pause';
      }
    } else if (act.kind === 'sheet') {
      sheetOpen = !sheetOpen;
    }
  }

  // ---- cluster head -------------------------------------------------------

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
</script>

<svelte:head>
  <title>Control plane — Atlas</title>
  <meta
    name="description"
    content="Manage the Atlas agents on your own machines: node health, pairing, topology and multi-node launches. Everything stays on your LAN."
  />
</svelte:head>

<svelte:window onkeydown={onKeydown} />

{#if fleet.mode === 'live'}
  <div class="bridge">
    <CommandStrip
      {fleet}
      onselect={select}
      cadence={cadenceId}
      oncadence={(id) => (cadenceId = id)}
      vitals={vitalsOn}
      onvitals={(v) => (vitalsOn = v)}
      onhelp={() => (sheetOpen = true)}
    />
    <Roster
      {fleet}
      {selectedId}
      {poller}
      {paused}
      onselect={select}
      onadd={() => (pairingOpen = true)}
      onpair={(n) => (pairing = n)}
    />
    <NodeStage
      bind:this={stageEl}
      {fleet}
      node={selectedNode}
      {poller}
      {paused}
      {vitalsOn}
      {tab}
      ontab={(t) => (tab = t)}
      {log}
      onlog={record}
      onpair={(n) => (pairing = n)}
      onunpair={(n) => (unpairing = n)}
      ondetails={(n) => (details = n)}
    />
    <FleetRail
      {fleet}
      {head}
      {clusterFlow}
      oncluster={() => (clusterOpen = true)}
      onmakehead={(id) => (head = id)}
      onselect={select}
      onpair={(n) => (pairing = n)}
      onunpair={(n) => (unpairing = n)}
    />
  </div>
{:else}
  <Nav />

  <main class="control">
    <section id="fleet" class="sx-cyan">
      <div class="container">
        <SectionHead
          level={1}
          label="// 01 · fleet"
          title="Your machines, one panel."
          sub="This page talks only to an agent on the computer you are using. That agent finds the others. Nothing here leaves your network, and none of it is in the page you downloaded."
          prov="no agent connected"
        />

        <div class="ctl-modes">
          {#if fleet.mode === 'reconnecting'}
            <!-- The agent was here a moment ago and went away — a restart, a
                 reboot, an ssh session closing. Falling through to the
                 "install the agent" invitation below would tell someone who
                 plainly HAS one to go and get one. -->
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
                <code class="mono">{startAgentCommand}</code>.
              </p>
            </div>
          {:else if fleet.mode === 'browser_unpaired'}
            <div class="ctl-setup">
              <h2>Pair this browser with your agent</h2>
              <!-- The reason, when the handshake gave one. `#scheduleProbe` sets
                   `detail` on this path too, and it was rendered nowhere: the
                   `{#if fleet.detail}` added with the no_agent panel covered only
                   that branch, so an operator here was told to paste a token
                   without being told what the agent actually said. -->
              {#if fleet.detail}
                <p class="ld-error" role="alert">{fleet.detail}</p>
              {/if}
              <p>
                An agent is running, but it has not seen this browser before. Run
                <code class="mono">atlasctl agent token</code> and paste the value into the
                launch dialog on the <a href="/">home page</a> — that is where the token
                field lives, not here. This page notices on its own once you have; it keeps
                looking. This is separate from pairing machines to each other.
              </p>
            </div>
          {:else}
            <!-- Prerendered. Most visitors see this, and it must read as an
                 invitation rather than an error. -->
            <div class="ctl-setup">
              <h2>Nothing is running here yet</h2>
              <!-- A refused handshake is not an empty machine. The probe now carries
                   the reason here (a protocol mismatch names the installer to run),
                   and without showing it this panel told an operator whose agent WAS
                   running that nothing was -- then promised to continue on its own
                   while retrying a handshake that fails the same way every time. -->
              {#if fleet.detail}
                <p class="ld-error" role="alert">{fleet.detail}</p>
              {/if}
              <p>
                Atlas runs on your hardware, not ours. Install the agent on a machine
                and this page becomes its control panel.
              </p>
              <InstallSteps />
              {#if attempted}
                <p class="ld-watching">
                  <span class="ld-pulse" aria-hidden="true"></span>
                  Watching for it — this page will continue on its own.
                </p>
              {:else}
                <!-- Not "watching": nothing is being watched until the operator
                     asks. Saying otherwise would be a claim about behaviour that
                     is deliberately not happening yet. -->
                <button type="button" class="btn btn-primary" onclick={connectNow}>
                  Connect to the agent on this machine
                </button>
                <p class="ctl-safety">
                  Your browser will ask permission to reach other apps on this
                  device. That is this page opening a connection to the agent on
                  127.0.0.1, and nothing else — it is asked now, rather than on
                  arrival, because until now there was nothing to connect to.
                </p>
              {/if}
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
          sub="Two views: how you reach each machine, and how the machines reach each other. Multi-node decode is all-reduce bound, so the link between two machines decides the throughput. A cluster that falls back to ethernet still runs — several times slower — while every correctness check keeps passing, so the fabric is called out here rather than left to be discovered in a benchmark."
        />
        <!-- Without an agent there are no machines to draw, so this section is
             the promise of the picture rather than the picture. The live
             topology, with its head-picking and pair actions, lives on the
             bridge's fleet rail. -->
        <TopologyMap nodes={fleet.nodes} {head} />
      </div>
    </section>

    <section id="launch" class="sx-cyan">
      <div class="container">
        <SectionHead
          label="// 03 · launch"
          title="Run one model across them."
          sub="Two phases, because one cannot fail cleanly. Every machine checks it can run this and holds its place; nothing starts until all of them have agreed. If one refuses, the reservations the others took are released — so a cluster is either whole or absent, never a half that hangs waiting on a rendezvous."
        />
        <p class="lc-offline">Connect an agent to launch anything.</p>
      </div>
    </section>

    <section id="alerts" class="section-alt sx-cyan">
      <div class="container">
        <SectionHead
          label="// 04 · alerts"
          title="What needs looking at."
          sub="Idle machines matter as much as busy ones. A clamped clock, a failing fan or a full cache filesystem is something to know before a launch, not after a benchmark comes back wrong."
        />
        <h3 class="al-empty-title">Nothing to report</h3>
        <p class="al-empty">
          No alerts. This section stays here so you never have to wonder where they
          would appear.
        </p>
      </div>
    </section>
  </main>

  <Footer />
{/if}

{#if fleet.mode === 'live' && pairingOpen}
  <PairingOverlay {fleet} onclose={() => (pairingOpen = false)} />
{/if}

{#if fleet.mode === 'live' && clusterOpen}
  <ClusterOverlay {fleet} bind:flow={clusterFlow} onclose={() => (clusterOpen = false)} />
{/if}

{#if sheetOpen}
  <ShortcutSheet onclose={() => (sheetOpen = false)} />
{/if}

{#if details}
  <NodeDetails node={details} onclose={() => (details = null)} />
{/if}

{#if pairing}
  <PairDialog node={pairing} onclose={() => (pairing = null)} />
{/if}

{#if unpairing}
  <UnpairDialog {fleet} node={unpairing} onclose={() => (unpairing = null)} />
{/if}
