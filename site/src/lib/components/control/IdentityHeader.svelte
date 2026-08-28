<!-- SPDX-License-Identifier: AGPL-3.0-only -->
<script>
  // Region C1: who the stage is about, in one 48px line.
  //
  // The hostname is the least trustworthy thing on it — Sparks ship with
  // colliding names — so the fingerprint sits beside it, copyable in full,
  // and every action keys on the fingerprint. Provenance renders as two
  // separate facts ("identity vouched by dgx1 · control routed via dgx1")
  // because they ARE two facts: who claimed this machine exists, and who
  // carries control to it. They are usually the same peer and still not the
  // same statement.

  import { copyOrSelect } from '$lib/clipboard.js';
  import { nameOf } from '$lib/agent/refusal.js';
  import { isStale } from '$lib/agent/fleet.svelte.js';
  import { nowMs, useClock } from '$lib/agent/clock.svelte.js';
  import * as S from '$lib/agent/stats.js';

  let { node, nodes = [], onpair, onunpair, ondetails } = $props();

  $effect(() => useClock());

  const short = $derived(`${node.id.slice(0, 4)}·${node.id.slice(4, 8)}`);
  const trusted = $derived(
    node.isLocal || node.pairing === 'paired' || node.pairing === 'vouched' || node.pairing === 'unreachable'
  );
  const trustLabel = $derived(
    node.isLocal
      ? 'this machine'
      : node.pairing === 'paired'
        ? 'paired'
        : node.pairing === 'pairing'
          ? 'pairing…'
          : node.pairing === 'unreachable'
            ? 'unreachable'
            : node.pairing === 'vouched'
              ? 'vouched'
              : 'discovered'
  );
  const stale = $derived(isStale(node, nowMs()) && trusted);
  const staleFor = $derived(Math.max(0, Math.round((nowMs() - node.lastSeen) / 1000)));

  const provenance = $derived.by(() => {
    const parts = [];
    if (node.vouchedBy) parts.push(`identity vouched by ${nameOf(node.vouchedBy, nodes)}`);
    if (node.reachedVia) parts.push(`control routed via ${nameOf(node.reachedVia, nodes)}`);
    return parts.join(' · ');
  });

  let copyState = $state('idle'); // idle | copied | blocked
  let copyTimer;
  $effect(() => () => clearTimeout(copyTimer));

  async function copyFp() {
    clearTimeout(copyTimer);
    // No element to fall back to ON PURPOSE. This copies the FULL fingerprint
    // while only the short form is rendered, so selecting what is on screen
    // would hand over a truncated value that looks like the real one — worse
    // than the refusal it was covering for.
    copyState = await copyOrSelect(node.id, null);
    copyTimer = setTimeout(() => (copyState = 'idle'), 2400);
  }
</script>

<header class="ih" class:ih-vouched={node.pairing === 'vouched'}>
  <h2 class="ih-name" title={node.name}>{node.name}</h2>
  <button
    type="button"
    class="fp-chip mono"
    onclick={copyFp}
    aria-label={`Fingerprint ${short}. Copy the full fingerprint.`}
  >
    <svg viewBox="0 0 12 14" width="9" height="11" aria-hidden="true">
      <path
        d="M6 .8 11.2 3.9v6.2L6 13.2.8 10.1V3.9Z"
        fill="none"
        stroke="currentColor"
        stroke-width="1.4"
        stroke-linejoin="round"
      />
    </svg>
    <!-- Not "select it manually": the full fingerprint is not on screen to
         select. `ondetails` is where it lives. -->
    {copyState === 'copied' ? 'copied' : copyState === 'idle' ? short : 'copy failed'}
  </button>
  <span class="trust-chip trust-{node.pairing}" class:trust-this={node.isLocal}>
    <span class="dot" aria-hidden="true"></span>{trustLabel}
  </span>

  <span class="ih-facts">
    {#if node.os}{node.os}{/if}
    {#if node.accelerator}· {node.accelerator}{/if}
    {#if node.agentVersion}· agent {node.agentVersion}{/if}
    {#if node.vitals?.agent_uptime_s != null}· up {S.uptime(node.vitals.agent_uptime_s)}{/if}
    {#if stale}· <span class="ih-stale">last seen {staleFor}s ago</span>{/if}
  </span>

  {#if provenance}
    <span class="ih-prov">{provenance}</span>
  {/if}

  {#if trusted && node.canLaunch !== true}
    <!-- The machine's own words, verbatim. Clamped to the header's two-line
         budget; the full sentence is one hover (title) and one tab (Launch
         tab) away, stated rather than truncated silently. -->
    <span class="ih-reason" title={node.cannotLaunchReason || 'This machine cannot run models.'}>
      {node.cannotLaunchReason || 'cannot run models'}
    </span>
  {/if}

  <span class="ih-actions">
    {#if !trusted}
      <button type="button" class="rr-pair" onclick={() => onpair?.(node)}>Pair…</button>
    {:else if !node.isLocal}
      <button type="button" class="fl-unpair" onclick={() => onunpair?.(node)}>Unpair…</button>
    {/if}
    <button
      type="button"
      class="ih-details"
      onclick={() => ondetails?.(node)}
      aria-label={`Full identity and interfaces of ${node.name}`}
    >
      details
    </button>
  </span>
</header>
