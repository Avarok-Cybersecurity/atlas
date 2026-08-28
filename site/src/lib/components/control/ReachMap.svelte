<!-- SPDX-License-Identifier: AGPL-3.0-only -->
<script>
  // How the fleet is connected, from where the operator is sitting.
  //
  // The other map answers "can these machines talk to each other", which is the
  // cluster question and is drawn as a mesh. This answers "how do I get to
  // this one", which is the question someone asks when a machine is missing or
  // when they are deciding what to pair next. Those are different graphs and
  // conflating them made the second unanswerable.
  //
  // Same house rules as TopologyMap: layout is computed and never dragged, the
  // SVG is role="img" with a generated sentence for its label, and every
  // machine also appears in the list beside it so nothing here needs to be
  // focusable.

  import { flip } from 'svelte/animate';
  import * as Hops from '$lib/agent/hops.js';
  import HelpDot from './HelpDot.svelte';

  let { nodes = [] } = $props();

  const W = 640;
  const H = 210;
  const R = 21;

  const tiers = $derived(Hops.tiers(nodes));
  const edges = $derived(Hops.reach(nodes));

  /** Position every node by its tier (a column) and its place within it. */
  const placed = $derived.by(() => {
    const cols = tiers.length;
    const map = new Map();
    tiers.forEach((t, ci) => {
      const x = cols === 1 ? W / 2 : 70 + (ci * (W - 140)) / (cols - 1);
      t.nodes.forEach((n, ri) => {
        const rows = t.nodes.length;
        const y = rows === 1 ? H / 2 : H / 2 - ((rows - 1) * 62) / 2 + ri * 62;
        map.set(n.id, { node: n, x, y, tier: t.tier });
      });
    });
    return map;
  });

  const lines = $derived(
    edges
      .map((e) => ({ e, a: placed.get(e.from), b: placed.get(e.to) }))
      .filter((l) => l.a && l.b)
  );

  const shown = $derived([...placed.values()]);

  const caption = $derived.by(() => {
    const vouched = nodes.filter((n) => n?.via).length;
    const direct = nodes.filter((n) => n && !n.isLocal && !n.via).length;
    if (!nodes.some((n) => n?.isLocal)) return 'No agent is connected to this page.';
    const parts = [`This browser reaches the agent on this machine`];
    if (direct) parts.push(`${direct} machine${direct === 1 ? '' : 's'} directly`);
    if (vouched) parts.push(`${vouched} more through another machine`);
    return `${parts.join(', then ')}.`;
  });

  const short = (n) => (n.kind === 'browser' ? 'You' : n.name || n.id.slice(0, 6));
</script>

<figure class="rm">
  <figcaption class="rm-cap">
    How your fleet is connected
    <HelpDot label="How to reach machines on another network">
      <p>
        This graph is drawn from where you are: your browser, the agent on this
        machine, and everything reachable outwards from it.
      </p>
      <p>
        A machine drawn one column further out is reached <em>through</em> the
        one before it, not directly. To add machines on a network this one
        cannot see, install the agent on a machine that sits on both — one
        interface on each — and pair with that. It becomes the middle node, and
        its neighbours appear behind it.
      </p>
      <p>Pairing stays between neighbours; a middle node vouches, it does not share keys.</p>
    </HelpDot>
  </figcaption>

  <svg viewBox="0 0 {W} {H}" role="img" aria-label={caption} class="rm-svg">
    {#each lines as l (l.e.from + l.e.to)}
      <line
        class="rm-edge"
        class:rm-edge-vouched={l.e.kind === 'vouched'}
        class:rm-edge-browser={l.e.kind === 'browser'}
        x1={l.a.x}
        y1={l.a.y}
        x2={l.b.x}
        y2={l.b.y}
      />
      {#if l.e.kind === 'vouched'}
        <text class="rm-edge-label" x={(l.a.x + l.b.x) / 2} y={(l.a.y + l.b.y) / 2 - 8} text-anchor="middle">
          via
        </text>
      {/if}
    {/each}

    {#each shown as p (p.node.id)}
      <g class="rm-node" animate:flip={{ duration: 380 }}>
        {#if p.node.kind === 'browser'}
          <rect class="rm-you" x={p.x - R} y={p.y - R + 4} width={R * 2} height={R * 2 - 8} rx="6" />
        {:else}
          <circle
            class="rm-dot"
            class:rm-dot-local={p.node.isLocal}
            class:rm-dot-vouched={p.tier === 3}
            cx={p.x}
            cy={p.y}
            r={R}
          />
        {/if}
        <text class="rm-label" x={p.x} y={p.y + R + 15} text-anchor="middle">{short(p.node)}</text>
      </g>
    {/each}
  </svg>
</figure>
