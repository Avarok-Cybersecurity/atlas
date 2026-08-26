<!-- SPDX-License-Identifier: AGPL-3.0-only -->
<script>
  // The fleet as a picture.
  //
  // Layout is COMPUTED, never dragged: node positions are a pure function of
  // the sorted node list, so the same fleet draws the same pixels every time
  // and nothing shifts. Fleets here are 1-8 machines and multi-node recipes are
  // exactly two, so a force simulation would be a dependency and a source of
  // jitter in exchange for nothing.
  //
  // Nothing inside this SVG is focusable or clickable. It carries role="img"
  // and a generated sentence as its label; every action lives in the button
  // list beside it. That gives full keyboard operability without inventing
  // focus semantics for SVG, and it means a screen reader gets a description
  // rather than a maze of unlabelled shapes.

  import { preferredAddress, linkWarns } from '$lib/agent/fleet.svelte.js';

  let { nodes = [], head = null, selected = new Set() } = $props();

  const W = 760;
  const H = 300;
  const R = 34;

  // Local node first, then paired, then discovered — and by fingerprint within
  // each group, so the order cannot depend on arrival timing.
  const ordered = $derived(
    nodes.slice().sort((a, b) => {
      const rank = (n) => (n.isLocal ? 0 : n.pairing === 'paired' ? 1 : 2);
      const r = rank(a) - rank(b);
      return r !== 0 ? r : a.id.localeCompare(b.id);
    })
  );

  const points = $derived(
    ordered.map((n, i) => {
      const count = ordered.length;
      if (count === 1) return { node: n, x: W / 2, y: H / 2 - 10 };
      // A single row, equally spaced, with a gentle arc so edges have room.
      const span = Math.min(W - 160, count * 190);
      const x = W / 2 - span / 2 + (span / Math.max(1, count - 1)) * i;
      const lift = count > 2 ? Math.sin((i / (count - 1)) * Math.PI) * 26 : 0;
      return { node: n, x, y: H / 2 - 10 - lift };
    })
  );

  // Edges only between trusted, launch-capable nodes: an unpaired node has no
  // link to draw because there is no relationship yet.
  const edges = $derived(
    points
      .filter((p) => p.node.isLocal || p.node.pairing === 'paired')
      .flatMap((a, i, arr) =>
        arr.slice(i + 1).map((b) => {
          const aa = preferredAddress(a.node);
          const bb = preferredAddress(b.node);
          const cls =
            !aa || !bb ? 'none' : (aa.class ?? 'ethernet') === (bb.class ?? 'ethernet') ? aa.class : 'ethernet';
          const speed = Math.min(aa?.speedMbps ?? 0, bb?.speedMbps ?? 0);
          return { a, b, cls, speed, warn: cls === 'none' || linkWarns(cls) };
        })
      )
  );

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
      <pattern id="topoHatch" width="8" height="8" patternTransform="rotate(45)" patternUnits="userSpaceOnUse">
        <line x1="0" y1="0" x2="0" y2="8" stroke="var(--border-strong)" stroke-width="3" />
      </pattern>
    </defs>

    {#each edges as e (e.a.node.id + e.b.node.id)}
      <line
        class="topo-edge"
        class:topo-edge-warn={e.warn}
        x1={e.a.x}
        y1={e.a.y}
        x2={e.b.x}
        y2={e.b.y}
      />
      <text class="topo-edge-label" x={(e.a.x + e.b.x) / 2} y={(e.a.y + e.b.y) / 2 - 10} text-anchor="middle">
        {#if e.cls === 'none'}no link{:else}{e.cls === 'roce'
            ? 'RoCE'
            : e.cls === 'infini_band'
              ? 'IB'
              : e.cls === 'wireless'
                ? 'Wi-Fi'
                : 'eth'}{#if e.speed}
            {Math.round(e.speed / 1000)}G{/if}{/if}
      </text>
    {/each}

    {#each points as p (p.node.id)}
      {@const trusted = p.node.isLocal || p.node.pairing === 'paired'}
      <g class="topo-node" class:topo-node-sel={selected.has(p.node.id)}>
        <circle
          cx={p.x}
          cy={p.y}
          r={R}
          fill={trusted ? 'var(--card)' : 'url(#topoHatch)'}
          stroke={selected.has(p.node.id) ? 'var(--sx)' : 'var(--border-strong)'}
          stroke-width={selected.has(p.node.id) ? 2.5 : 1.5}
        />
        <text class="topo-node-name" x={p.x} y={p.y + 4} text-anchor="middle">
          {p.node.name.slice(-4)}
        </text>
        {#if head === p.node.id}
          <g transform="translate({p.x - 10}, {p.y + R + 6})">
            <rect width="20" height="14" rx="3" fill="var(--ch-cyan)" />
            <text class="topo-head-tag" x="10" y="10.5" text-anchor="middle">H</text>
          </g>
        {/if}
      </g>
      <text class="topo-node-label" x={p.x} y={p.y - R - 10} text-anchor="middle">
        {p.node.name}
      </text>
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
