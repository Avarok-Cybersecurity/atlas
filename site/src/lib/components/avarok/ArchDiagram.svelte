<!--
  The platform in one picture: the request path across the top and down into
  the engine nodes, the control plane beside it (never on that path), the
  economics ledger underneath collecting what the nodes report. Inline SVG so
  it follows the theme and needs no image request.
-->
<script>
  import { company } from '$lib/content/brand.js';
</script>

<figure class="av-diagram" aria-label="Avarok platform architecture">
  <svg viewBox="0 0 1000 400" role="img">
    <title>Requests flow from your applications through the router to Avarok Engine nodes on your GPUs. Avarok Control manages rollout, policy and repair out of band. Avarok Economics collects telemetry from every node into a ledger.</title>
    <!-- request path -->
    <rect class="box" x="20" y="40" width="220" height="70" />
    <text class="lbl" x="130" y="70" text-anchor="middle">Your applications and agents</text>
    <text class="sub" x="130" y="92" text-anchor="middle">OpenAI · Anthropic · Responses APIs</text>

    <path class="wire hot" d="M240 75 H330" />
    <rect class="box violet" x="330" y="40" width="170" height="70" />
    <text class="lbl" x="415" y="70" text-anchor="middle">GPU aware router</text>
    <text class="sub" x="415" y="92" text-anchor="middle">KV reuse · queue · VRAM pressure</text>

    <path class="wire hot" d="M415 110 V150" />
    <path class="wire hot" d="M415 150 H120 V175" />
    <path class="wire hot" d="M415 150 V175" />
    <path class="wire hot" d="M415 150 H710 V175" />

    {#each [120, 415, 710] as x, i}
      <rect class="box violet" x={x - 130} y="175" width="260" height="84" />
      <text class="lbl" x={x} y="203" text-anchor="middle">{company.engine} · node {i + 1}</text>
      <text class="sub" x={x} y="224" text-anchor="middle">signed recipe · kernels for this silicon</text>
      <text class="sub" x={x} y="243" text-anchor="middle">{['GB10 · NVFP4', 'H100 · FP8 · bring up', 'gfx1151 · SCALE'][i]}</text>
      <path class="wire" d={`M${x} 259 V300`} />
    {/each}

    <!-- control plane, out of band -->
    <rect class="box cyan" x="860" y="40" width="120" height="219" />
    <text class="lbl" x="920" y="70" text-anchor="middle">{company.control}</text>
    <text class="sub" x="920" y="92" text-anchor="middle">out of band</text>
    {#each ['rollout', 'canary', 'policy', 'repair', 'scale'] as w, i}
      <text class="sub" x="920" y={122 + i * 22} text-anchor="middle">{w}</text>
    {/each}
    <path class="wire" d="M860 120 H840 V217 H840" stroke-dasharray="4 5" />
    <path class="wire" d="M840 217 H840" />
    <path class="wire" d="M840 217 H840" />
    <path class="wire" d="M840 217 L840 217" />
    <path class="wire" d="M840 217 H840" />
    <path class="wire" d="M840 217 H840" />
    <path class="wire" d="M840 120 L840 300" stroke-dasharray="4 5" />
    <path class="wire" d="M840 300 H120" stroke-dasharray="4 5" />

    <!-- economics ledger -->
    <rect class="box green" x="20" y="300" width="960" height="80" />
    <text class="lbl" x="40" y="330">{company.economics}</text>
    <text class="sub" x="40" y="352">workload × model × runtime × configuration × GPU × cluster</text>
    <text class="sub" x="960" y="330" text-anchor="end">$ per million tokens · $ per workload at SLO · productive GPU hours</text>
    <text class="sub" x="960" y="352" text-anchor="end">stranded capacity · chargeback by business unit · payback</text>
  </svg>
  <figcaption class="av-small" style="margin-top:0.9rem">The request path never touches the control plane. The ledger reads what the engine measured at the source.</figcaption>
</figure>
