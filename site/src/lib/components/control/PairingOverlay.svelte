<!-- SPDX-License-Identifier: AGPL-3.0-only -->
<script>
  // The pairing & trust overlay — where machines are added, in every way the
  // protocol offers, promoted out of the console dock where these panels
  // were parked while the bridge landed.
  //
  // Two paths, in the order an operator actually uses them:
  //
  //   1. JoinGuide — mint a one-use code here, run one line on the new
  //      machine. This is the path for a machine with no agent yet, and it
  //      carries both explicit control grants: `--grant-control` on the
  //      pasted line (that machine letting this fleet drive it) and the
  //      mint-time `allow_control` (this machine letting the joiner drive
  //      it back). Both default to the direction the copy states.
  //
  //   2. FleetScan — discovery status, plus add-by-address for the machine
  //      mDNS cannot see; its word-comparison confirm carries the same
  //      explicit `allow_control` decision.
  //
  // Discovered machines ALSO keep their Pair… buttons on the roster and rail
  // — this overlay adds machines; it does not become the only door.

  import Overlay from './Overlay.svelte';
  import FleetScan from './FleetScan.svelte';
  import JoinGuide from './JoinGuide.svelte';

  let { fleet, onclose } = $props();
</script>

<Overlay label="Add a machine" wide {onclose}>
  {#if fleet.controlOnly}
    <p class="fl-co-why">
      This machine drives the fleet; it does not run models itself. Pair a
      machine that can and everything on the bridge applies to it.
    </p>
  {/if}

  <JoinGuide {fleet} />

  <div class="po-scan">
    <FleetScan {fleet} />
  </div>

  <p class="po-note">
    A machine another peer told us about arrives as <strong>vouched</strong> —
    second-hand identity, never shown as paired. Running the ceremony against
    it directly is what upgrades it.
  </p>
</Overlay>
