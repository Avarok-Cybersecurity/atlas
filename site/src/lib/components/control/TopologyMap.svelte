<!-- SPDX-License-Identifier: AGPL-3.0-only -->
<script>
  // The fleet as a picture.
  //
  // The rules live in `$lib/agent/topology.js`, which is pure and tested; this
  // file is the surface. Layout is COMPUTED, never dragged — the same fleet
  // draws the same pixels every time, so nothing shifts under a cursor and
  // nothing depends on the order machines happened to be discovered in.
  //
  // Nothing inside this SVG is focusable or clickable. It carries role="img"
  // and a generated sentence as its label; every action lives in the button
  // list beside it. That gives full keyboard operability without inventing
  // focus semantics for SVG, and it means a screen reader gets a description
  // rather than a maze of unlabelled shapes.

  import { flip } from 'svelte/animate';
  import { scale } from 'svelte/transition';
  import { preferredAddress } from '$lib/agent/fleet.svelte.js';
  import * as Topo from '$lib/agent/topology.js';

  let { nodes = [], head = null, selected = new Set() } = $props();

  const W = Topo.W;
  const H = Topo.H;
  const R = Topo.R;

  // A pattern id must be unique per instance: two graphs on one page sharing a
  // hardcoded id makes the second one reference the first one's fill.
  const hatchId = `topoHatch-${Math.random().toString(36).slice(2, 9)}`;

  const ordered = $derived(Topo.ordered(nodes));
  const points = $derived(Topo.points(nodes));
  const edges = $derived(Topo.edges(points));

  const label = $derived(
    (() => {
      if (ordered.length === 0) return 'No nodes.';
      const parts = ordered.map((n) => {
        const a = preferredAddress(n);
        const role = n.isLocal ? 'this machine' : n.pairing;
        const link = a ? `reachable over ${a.class.replace('_', ' ')}` : 'no usable link';
        const isHead = head === n.id ? ', head' : '';
        return `${n.name} (${role}${isHead}), ${link}`;
      });
      const warned = edges.filter((e) => e.warn).length;
      const tail = warned
        ? ` ${warned} link${warned === 1 ? '' : 's'} would not use RDMA, which is several times slower for multi-node decode.`
        : '';
      return `Fleet of ${ordered.length} node${ordered.length === 1 ? '' : 's'}. ${parts.join('. ')}.${tail}`;
    })()
  );

  /** How a link class is written on an edge. */
  function linkText(cls) {
    if (cls === 'none') return 'no link';
    if (cls === 'roce') return 'RoCE';
    if (cls === 'infini_band') return 'IB';
    if (cls === 'wireless') return 'Wi-Fi';
    if (cls === 'unverified') return 'unverified';
    return 'eth';
  }
</script>

<figure class="topo-fig">
  {#if ordered.length === 0}
    <!-- An empty picture is worse than no picture: it reads as broken. The
         placeholder keeps the figure's height so the section does not resize
         when the first node arrives. -->
    <div class="topo-empty">
      <svg viewBox="0 0 120 60" width="120" height="60" aria-hidden="true" class="topo-empty-mark">
        <circle cx="24" cy="30" r="14" fill="none" stroke="currentColor" stroke-width="2" stroke-dasharray="4 4" />
        <circle cx="96" cy="30" r="14" fill="none" stroke="currentColor" stroke-width="2" stroke-dasharray="4 4" />
        <line x1="38" y1="30" x2="82" y2="30" stroke="currentColor" stroke-width="2" stroke-dasharray="5 5" />
      </svg>
      <p>Machines appear here once an agent is running.</p>
      <p class="topo-empty-sub">
        Two of them, on the same network, are what the EP=2 recipes need.
      </p>
    </div>
  {:else}
  <svg class="topo-svg" viewBox="0 0 {W} {H}" role="img" aria-label={label}>
    <defs>
      <pattern id={hatchId} width="8" height="8" patternTransform="rotate(45)" patternUnits="userSpaceOnUse">
        <line x1="0" y1="0" x2="0" y2="8" stroke="var(--border-strong)" stroke-width="3" />
      </pattern>
    </defs>

    {#each edges as e (e.a.node.id + e.b.node.id)}
      <line
        in:scale={{ duration: 260 }}
        class="topo-edge"
        class:topo-edge-warn={e.warn}
        x1={e.a.x}
        y1={e.a.y}
        x2={e.b.x}
        y2={e.b.y}
      />
      <text class="topo-edge-label" x={(e.a.x + e.b.x) / 2} y={(e.a.y + e.b.y) / 2 - 10} text-anchor="middle">
        {linkText(e.cls)}{#if e.speed} {Math.round(e.speed / 1000)}G{/if}
      </text>
    {/each}

    {#each points as p (p.node.id)}
      {@const trusted = p.node.isLocal || p.node.pairing === 'paired'}
      <!-- A discovered machine used to pop in and shove every other node
           sideways, because positions are index-derived. Keyed + flipped, they
           slide to their new places instead.

           Both are CSS-driven, so the global `prefers-reduced-motion` rule in
           app.css — `animation-duration: 0.001ms !important` on `*` — zeroes
           them. `!important` in a stylesheet outranks the inline shorthand
           Svelte writes, which is what makes one killswitch enough rather than
           each animation needing its own guard. -->
      <g
        class="topo-node"
        class:topo-node-sel={selected.has(p.node.id)}
        animate:flip={{ duration: 420 }}
        in:scale={{ duration: 260, start: 0.6 }}
      >
        <circle
          cx={p.x}
          cy={p.y}
          r={R}
          fill={trusted ? 'var(--card)' : `url(#${hatchId})`}
          stroke={selected.has(p.node.id) ? 'var(--sx)' : 'var(--border-strong)'}
          stroke-width={selected.has(p.node.id) ? 2.5 : 1.5}
        />
        <!-- The fingerprint, not the hostname: this file's own header warns
             that Sparks ship with colliding names, and the last four characters
             of a hostname collide for exactly the machines an operator most
             needs to tell apart. -->
        <text class="topo-node-name" x={p.x} y={p.y + 4} text-anchor="middle">
          {Topo.label(p.node)}
        </text>
        {#if head === p.node.id}
          <g transform="translate({p.x - 10}, {p.y + R + 6})">
            <rect width="20" height="14" rx="3" fill="var(--ch-cyan)" />
            <text class="topo-head-tag" x="10" y="10.5" text-anchor="middle">H</text>
          </g>
        {/if}
        <!-- Inside the animated group, not beside it: `animate:` requires the
             element to be the only child of its keyed block, and a hostname
             that stayed put while its circle slid would be worse than no
             animation at all. -->
        <text class="topo-node-label" x={p.x} y={p.y - R - 10} text-anchor="middle">
          {p.node.name}
        </text>
      </g>
    {/each}
  </svg>

  <figcaption class="visually-hidden">{label}</figcaption>
  {/if}
</figure>

<!-- The chart's text equivalent, same policy as the ladder chart on the home
     page: every number drawn is also available as text. -->
<details class="topo-table">
  <summary>Links as text</summary>
  <table>
    <thead>
      <tr><th scope="col">From</th><th scope="col">To</th><th scope="col">Link</th><th scope="col">Speed</th></tr>
    </thead>
    <tbody>
      {#each edges as e (e.a.node.id + e.b.node.id + 't')}
        <tr>
          <td>{e.a.node.name}</td>
          <td>{e.b.node.name}</td>
          <td>{e.cls === 'none' ? 'none' : e.cls.replace('_', ' ')}</td>
          <td class="mono">{e.speed ? `${Math.round(e.speed / 1000)} Gb/s` : '—'}</td>
        </tr>
      {:else}
        <tr><td colspan="4">No links — pair a second machine to form a cluster.</td></tr>
      {/each}
    </tbody>
  </table>
</details>
