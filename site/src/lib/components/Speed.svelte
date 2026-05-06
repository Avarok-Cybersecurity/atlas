<script>
  import { advantage, pillars, benchmarks } from '$lib/data.js';
</script>

<section id="speed">
  <div class="container">
    <div class="slabel">Faster by Design</div>
    <h2 class="stitle">Clean architecture beats bloat</h2>
    <p class="ssub">
      vLLM ships 20+ GB of Python, PyTorch, and 200+ dependencies. Atlas ships a single ~2.5 GB
      binary. That simplicity is the speed.
    </p>

    <div class="advantage-grid">
      <div class="advantage-col hl">
        <h3 style="color: var(--cyan);">Atlas</h3>
        {#each advantage.atlas as row}
          <div class="adv-row">
            <span class="l">{row.label}</span>
            <span class="v {row.tone}">{row.value}</span>
          </div>
        {/each}
      </div>
      <div class="advantage-col">
        <h3 style="color: #64748b;">vLLM</h3>
        {#each advantage.vllm as row}
          <div class="adv-row">
            <span class="l">{row.label}</span>
            <span class="v" class:bad={row.tone === 'bad'} style={row.tone === 'neutral' ? 'color: var(--t2)' : ''}>
              {row.value}
            </span>
          </div>
        {/each}
      </div>
    </div>

    <div class="pillars">
      {#each pillars as p}
        <div class="card">
          <div class="card-icon">{p.icon}</div>
          <h3>{p.title}</h3>
          <p>{p.body}</p>
        </div>
      {/each}
    </div>

    {#each benchmarks as bench}
      <div class="chart">
        <div class="chart-head">
          <div>
            <div class="chart-title">{bench.title}</div>
            <div class="chart-sub">{bench.sub}</div>
          </div>
          <div class="chart-legend">
            <div class="chart-legend-item"><div class="swatch a"></div> Atlas</div>
            <div class="chart-legend-item"><div class="swatch v"></div> vLLM</div>
          </div>
        </div>
        {#each bench.pairs as pair}
          <div class="cpair">
            <div class="cpair-label">{pair.label}</div>
            <div class="crow">
              <span class="crow-tag">Atlas</span>
              <div class="crow-track">
                <div class="crow-fill a" style="width:{pair.atlas.width}%">{pair.atlas.value}</div>
              </div>
              <span class="crow-x">{pair.atlas.speedup}</span>
            </div>
            <div class="crow">
              <span class="crow-tag">vLLM</span>
              <div class="crow-track">
                <div class="crow-fill v" style="width:{pair.vllm.width}%">{pair.vllm.value}</div>
              </div>
              <span class="crow-x"></span>
            </div>
          </div>
        {/each}
      </div>
    {/each}
  </div>
</section>
