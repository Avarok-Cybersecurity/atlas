<script>
  import { models } from '$lib/data.js';

  const badgeClass = { MTP: 'badge-mtp', FP8: 'badge-fp8', EP2: 'badge-ep2' };
  const badgeText = { MTP: 'MTP', FP8: 'FP8', EP2: 'EP=2' };
</script>

<section id="models" style="background: var(--bg2);">
  <div class="container">
    <div class="slabel">Supported Models</div>
    <h2 class="stitle">Model matrix</h2>
    <p class="ssub">
      Every model gets hand-tuned CUDA kernels. We expand based on what the community runs. All models
      ship with OpenAI-compatible tool calling.
    </p>
    <div class="mtable-scroll">
    <table class="mtable">
      <thead>
        <tr>
          <th>Model</th>
          <th>Parameters</th>
          <th>Quantization</th>
          <th>Architecture</th>
          <th>Throughput</th>
        </tr>
      </thead>
      <tbody>
        {#each models as m}
          <tr>
            <td class="mn">
              {m.name}
              {#each m.badges as b}
                <span class="badge {badgeClass[b]}">{badgeText[b]}</span>
              {/each}
            </td>
            <td>{m.params}</td>
            <td>{m.quant}</td>
            <td>{m.arch}</td>
            <td class="sp">{m.tps}</td>
          </tr>
        {/each}
      </tbody>
    </table>
    </div>
    <div style="font-size: 0.72rem; color: var(--t3); margin-top: 1rem; font-style: italic;">
      All benchmarks on single DGX Spark (GB10) unless noted. EP=2 = Expert Parallelism across two nodes.
      MTP = Multi-Token Prediction speculative decoding.
    </div>
  </div>
</section>
