<!-- SPDX-License-Identifier: AGPL-3.0-only -->
<script>
  // What the operator sees while the fleet is still being assembled.
  //
  // Three things, because they are three halves of one question — "where are my
  // machines?":
  //   1. that discovery is running, and is passive rather than stalled
  //   2. which networks THIS machine is on, since that is what discovery can
  //      reach and it is the first thing to check when nothing appears
  //   3. a way to name a machine directly, for when discovery cannot reach it
  //
  // (3) is not a fallback for a broken (1). Discovery is mDNS, which is
  // link-local by design: it does not cross a router, and plenty of managed
  // networks drop multicast outright. A machine one hop away is invisible to it
  // and perfectly reachable by address.
  //
  // House rule, as in ClusterLaunch: the rules live in tested modules
  // (`network.js`), and this file is only a surface.
  import { checkTarget, networksOf } from '$lib/agent/network.js';

  let { fleet } = $props();

  const networks = $derived(networksOf(fleet.local ?? { addresses: [] }));

  let target = $state('');
  let code = $state('');
  let phase = $state('idle'); // idle | dialling | confirm | done
  let detail = $state('');
  let answered = $state(null); // {node, name, address, verification}

  const targetProblem = $derived(target.trim() ? (checkTarget(target).why ?? '') : '');
  const ready = $derived(checkTarget(target).ok && code.trim().length > 0);

  async function dial() {
    if (!ready || phase === 'dialling') return;
    phase = 'dialling';
    detail = '';
    const res = await fleet.pairAt(target.trim(), code.trim());
    if (!res.ok) {
      detail = res.detail || 'Nothing answered at that address.';
      phase = 'idle';
      return;
    }
    answered = res;
    phase = 'confirm';
  }

  async function accept() {
    if (phase !== 'confirm' || !answered?.node) return;
    const res = await fleet.confirm(answered.node);
    if (res.ok) {
      phase = 'done';
      target = '';
      code = '';
      return;
    }
    detail = res.detail || 'The agent did not accept the pairing.';
  }

  async function refuse() {
    if (phase !== 'confirm' || !answered?.node) return;
    await fleet.reject(answered.node);
    answered = null;
    phase = 'idle';
    detail = 'Nothing was trusted.';
  }
</script>

<div class="fs">
  <p class="ld-watching fs-scan">
    <span class="ld-pulse" aria-hidden="true"></span>
    Scanning for available nodes…
  </p>

  {#if networks.length > 0}
    <p class="fs-lead">This machine is on:</p>
    <ul class="fs-nets">
      {#each networks as n (n.addr)}
        <li>
          <span class="mono fs-addr">{n.addr}</span>
          {#if n.subnet}<span class="fs-sub mono">{n.subnet}</span>{/if}
          <span class="fs-meta">{n.iface} · {n.detail}</span>
        </li>
      {/each}
    </ul>
    <p class="fs-note">
      Machines on these networks appear here on their own. Discovery is
      link-local, so anything past a router has to be named below.
    </p>
  {:else}
    <!-- Not "no networks found". The agent reports its own interfaces, so an
         empty list means it enumerated none — which on a machine that plainly
         has a network is a fault worth naming, not a quiet blank. -->
    <p class="fs-note">
      This agent reported no network interfaces, so it cannot discover anything.
      A machine can still be added by address below.
    </p>
  {/if}

  {#if phase === 'confirm' && answered}
    <div class="fs-confirm">
      <p class="fs-lead">
        <strong>{answered.name || 'A machine'}</strong> answered at
        <span class="mono">{answered.address}</span>.
      </p>
      <p class="fs-words mono">{answered.verification}</p>
      <p class="fs-note">
        Check those words match what that machine is showing. Nothing is trusted
        yet — no pairing has been written.
      </p>
      <div class="fs-actions">
        <button type="button" class="btn btn-primary" onclick={accept}>
          They match — trust this node
        </button>
        <button type="button" class="btn" onclick={refuse}>They differ — cancel</button>
      </div>
    </div>
  {:else}
    <form
      class="fs-form"
      onsubmit={(e) => {
        e.preventDefault();
        dial();
      }}
    >
      <p class="fs-lead">Or add one by address:</p>
      <div class="fs-row">
        <label class="fs-field">
          <span>Address</span>
          <input
            class="mono"
            bind:value={target}
            placeholder="10.10.10.2"
            autocomplete="off"
            spellcheck="false"
          />
        </label>
        <label class="fs-field fs-field-code">
          <span>Pairing code</span>
          <input
            class="mono"
            bind:value={code}
            placeholder="12345678"
            inputmode="numeric"
            autocomplete="off"
          />
        </label>
        <button type="submit" class="btn btn-primary" disabled={!ready || phase === 'dialling'}>
          {phase === 'dialling' ? 'Connecting…' : 'Add'}
        </button>
      </div>
      <p class="fs-note">
        The port is optional — 34334 is assumed. Run
        <code class="mono">atlasctl agent pair</code> on that machine to get one.
      </p>
      {#if targetProblem}<p class="fs-bad">{targetProblem}</p>{/if}
    </form>
  {/if}

  {#if detail}<p class="fs-bad">{detail}</p>{/if}
</div>
