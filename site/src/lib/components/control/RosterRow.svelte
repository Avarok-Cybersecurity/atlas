<!-- SPDX-License-Identifier: AGPL-3.0-only -->
<script>
  // One roster row: identity, trust, worst severity, what it is running, how
  // control reaches it — and the two mono micro-columns that make the roster
  // the comparative fleet I/O view. Values arrive with the stats pollers;
  // until then (and whenever a node is simply not serving) they are dashes,
  // because "unknown" must never render as a number.
  //
  // A discovered node gets identity and a Pair button only. Telemetry from an
  // unpaired machine proves nothing, and a row that ranked it alongside
  // verified machines would suggest otherwise.

  let { node, vm, nodes = [], onselect, onpair, decode = null, requests = null } = $props();

  const short = $derived(node.id.slice(0, 8));
  const discovered = $derived(!node.isLocal && (node.pairing === 'discovered' || node.pairing === 'pairing'));
  const worst = $derived.by(() => {
    const weight = { critical: 0, warning: 1, info: 2 };
    return node.alerts.reduce(
      (w, a) => ((weight[a.severity] ?? 9) < (weight[w?.severity] ?? 9) ? a : w),
      null
    );
  });
  const viaName = $derived(
    node.reachedVia ? (nodes.find((n) => n.id === node.reachedVia)?.name ?? 'a peer') : null
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

  const label = $derived.by(() => {
    const parts = [node.name, trustLabel];
    if (viaName) parts.push(`reached via ${viaName}`);
    if (node.running) parts.push(`running ${node.running}`);
    if (worst) parts.push(`worst alert ${worst.severity}`);
    return parts.join(', ');
  });
</script>

<li class="rr" class:rr-selected={vm.selected} aria-current={vm.selected ? 'true' : undefined}>
  <button
    type="button"
    class="rr-main"
    data-node={node.id}
    aria-label={label}
    onclick={() => onselect?.(node.id)}
  >
    <span class="rr-body" aria-hidden="true">
      <span class="rr-id">
        <span class="rr-name">{node.name}</span>
        <span class="rr-fp mono">{short}</span>
      </span>
      <span class="rr-meta">
        <span class="rr-trust rr-trust-{node.isLocal ? 'local' : node.pairing}">{trustLabel}</span>
        {#if worst}<span class="rr-dot rr-dot-{worst.severity}"></span>{/if}
        {#if viaName}<span class="rr-via">via {viaName}</span>{/if}
        {#if node.running}<span class="rr-recipe mono">{node.running}</span>{/if}
      </span>
    </span>
    {#if !discovered}
      <span class="rr-cols mono" aria-hidden="true">
        <span class="rr-col">{decode ?? '—'}</span>
        <span class="rr-col">{requests ?? '—'}</span>
      </span>
    {/if}
  </button>
  {#if discovered}
    <button type="button" class="rr-pair" onclick={() => onpair?.(node)}>Pair…</button>
  {/if}
</li>
