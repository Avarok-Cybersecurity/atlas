<!--
  The platform in one picture: the request path across the top and down into
  the engine nodes, the control plane beside it (never on that path), the
  economics ledger underneath collecting what the nodes report. Inline SVG so
  it follows the theme and needs no image request.

  Sizing a box: the type is IBM Plex Mono, so a line is (characters x 0.6 x font
  size) wide. Labels are 13 units, sub lines are 11. Keep 12 units clear on each
  side. The longest line in a node is 40 characters, 264 units, which is why a
  node is 288 wide and the canvas is 1100. e2e/marketing.spec.js measures every
  line against its box, so a longer label fails a test instead of a review.
-->
<script>
  import { company } from '$lib/content/brand.js';

  const nodes = [
    { x: 160, hw: 'GB10 · NVFP4' },
    { x: 460, hw: 'H100 · FP8 · bring up' },
    { x: 760, hw: 'gfx1151 · SCALE' }
  ];
  const duties = ['rollout', 'canary', 'policy', 'repair', 'scale'];
</script>

<figure class="av-diagram" aria-label="Avarok platform architecture">
  <svg viewBox="0 0 1100 400" role="img">
    <title>Requests flow from your applications through the router to Avarok Engine nodes on your GPUs. Avarok Control manages rollout, policy and repair out of band. Avarok Economics collects telemetry from every node into a ledger.</title>
    <!-- request path -->
    <rect class="box" x="6" y="40" width="264" height="70" />
    <text class="lbl" x="138" y="70" text-anchor="middle">Your applications and agents</text>
    <text class="sub" x="138" y="92" text-anchor="middle">OpenAI · Anthropic · Responses APIs</text>

    <path class="wire hot" d="M270 75 H338" />
    <rect class="box violet" x="338" y="40" width="244" height="70" />
    <text class="lbl" x="460" y="70" text-anchor="middle">GPU aware router</text>
    <text class="sub" x="460" y="92" text-anchor="middle">KV reuse · queue · VRAM pressure</text>

    <path class="wire hot" d="M460 110 V175" />
    <path class="wire hot" d="M460 142 H160 V175" />
    <path class="wire hot" d="M460 142 H760 V175" />

    <!-- the fleet: what the control plane manages as one thing -->
    <rect class="fleet" x="6" y="163" width="908" height="108" />
    {#each nodes as n, i}
      <rect class="box violet" x={n.x - 144} y="175" width="288" height="84" />
      <text class="lbl" x={n.x} y="203" text-anchor="middle">{company.engine} · node {i + 1}</text>
      <text class="sub" x={n.x} y="224" text-anchor="middle">signed recipe · kernels for this silicon</text>
      <text class="sub" x={n.x} y="243" text-anchor="middle">{n.hw}</text>
      <path class="wire" d={`M${n.x} 259 V300`} />
    {/each}

    <!-- control plane, out of band: it reaches the fleet, never the request path -->
    <rect class="box cyan" x="954" y="40" width="140" height="231" />
    <text class="lbl" x="1024" y="70" text-anchor="middle">{company.control}</text>
    <text class="sub" x="1024" y="92" text-anchor="middle">out of band</text>
    {#each duties as w, i}
      <text class="sub" x="1024" y={128 + i * 24} text-anchor="middle">{w}</text>
    {/each}
    <path class="wire oob" d="M954 217 H914" />

    <!-- economics ledger -->
    <rect class="box green" x="6" y="300" width="1088" height="80" />
    <text class="lbl" x="28" y="330">{company.economics}</text>
    <text class="sub" x="28" y="352">workload × model × runtime × configuration × GPU × cluster</text>
    <text class="sub" x="1072" y="330" text-anchor="end">$ per million tokens · $ per workload at SLO · productive GPU hours</text>
    <text class="sub" x="1072" y="352" text-anchor="end">stranded capacity · chargeback by business unit · payback</text>
  </svg>
  <figcaption class="av-small" style="margin-top:0.9rem">The request path never touches the control plane. The ledger reads what the engine measured at the source.</figcaption>
</figure>
