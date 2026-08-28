<!-- SPDX-License-Identifier: AGPL-3.0-only -->
<script>
  // Everything known about one machine.
  //
  // The card shows what an operator scans; this shows what they check. The
  // difference matters most for the fingerprint: the card shows eight
  // characters because that is enough to tell two machines apart at a glance,
  // and this shows all of it because that is what you compare when you are
  // deciding whether to trust something.
  //
  // Every interface is listed, not only the preferred one. A cluster that
  // silently fell back to ethernet is diagnosed by seeing which links exist,
  // and the card cannot show that without becoming unreadable.

  import { linkWarns } from '$lib/agent/fleet.svelte.js';
  import { modal } from '$lib/modal.js';

  let { node, onclose } = $props();

  let dialogEl = $state(null);

  // The same two lines PairDialog and LaunchDialog carry. Without them this
  // markup claims `role="dialog" aria-modal="true"` while Tab walks straight
  // into the page behind it and Escape does nothing — an assistive-technology
  // user is told they are in a modal and then handed no way out of it.
  // Focus in, Tab trap and focus-return live in modal.js (use:modal below);
  // Esc keeps its window listener so it works wherever focus wanders.
  $effect(() => {
    const onKey = (ev) => {
      if (ev.key === 'Escape') onclose?.();
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  });

  const linkName = {
    roce: 'RoCE',
    infini_band: 'InfiniBand',
    ethernet: 'Ethernet',
    wireless: 'Wi-Fi',
    virtual: 'virtual',
    loopback: 'loopback',
    unverified: 'unverified'
  };

  function speed(a) {
    return a.speedMbps ? `${Math.round(a.speedMbps / 1000)} Gb/s` : '—';
  }
</script>

<div class="ld-backdrop" role="presentation" onclick={onclose}></div>
<!-- svelte-ignore a11y_no_noninteractive_element_to_interactive_role -->
<div
  class="ld nd"
  role="dialog"
  aria-modal="true"
  aria-labelledby="nd-title"
  tabindex="-1"
  bind:this={dialogEl}
  use:modal
>
  <header class="ld-head">
    <h3 class="ld-title" id="nd-title">{node.name}</h3>
    <button type="button" class="ld-close" onclick={onclose} aria-label="Close">×</button>
  </header>

  <div class="ld-body">
    <dl class="nd-facts">
      <dt>Fingerprint</dt>
      <dd class="mono nd-fp">{node.id}</dd>

      <dt>Trust</dt>
      <dd>{node.isLocal ? 'this machine' : node.pairing}</dd>

      <dt>Operating system</dt>
      <dd>{node.os || 'not reported'}</dd>

      <dt>Accelerator</dt>
      <dd>{node.accelerator || 'not reported'}</dd>

      <dt>Agent</dt>
      <dd>{node.agentVersion || 'not reported'}</dd>

      <dt>Can run models</dt>
      <dd>
        {#if node.canLaunch}yes{:else}no — {node.cannotLaunchReason || 'no reason given'}{/if}
      </dd>

      <dt>Running</dt>
      <dd>{node.running || 'nothing'}</dd>
    </dl>

    <h4 class="nd-h4">Interfaces</h4>
    {#if node.addresses.length === 0}
      <p class="nd-none">None reported. A machine with no usable link cannot hold a rank.</p>
    {:else}
      <table class="nd-links">
        <thead>
          <tr>
            <th scope="col">Interface</th><th scope="col">Address</th>
            <th scope="col">Link</th><th scope="col">Speed</th>
          </tr>
        </thead>
        <tbody>
          {#each node.addresses as a (a.addr + a.iface)}
            <tr>
              <td class="mono">{a.iface || '—'}</td>
              <td class="mono">{a.addr}</td>
              <td class:nd-warn={linkWarns(a.class)}>
                {linkName[a.class] ?? a.class}{#if a.rdma} · RDMA{/if}
              </td>
              <td class="mono">{speed(a)}</td>
            </tr>
          {/each}
        </tbody>
      </table>
      <p class="nd-note">
        Multi-node decode is all-reduce bound, so the slowest link between two
        machines decides the throughput — not the fastest one either of them has.
      </p>
    {/if}
  </div>
</div>
