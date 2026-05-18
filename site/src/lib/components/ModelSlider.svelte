<script>
  // Data is SSOT-derived from github.com/Avarok-Cybersecurity/atlas-recipes
  // via site/scripts/gen-models.mjs -> models.generated.json.
  import families from '$lib/models.generated.json';

  let index = $state(0);
  let copied = $state('');

  const total = families.length;

  function go(i) {
    index = ((i % total) + total) % total;
  }
  function prev() { go(index - 1); }
  function next() { go(index + 1); }

  async function copyCmd(cmd) {
    try {
      await navigator.clipboard.writeText(cmd);
      copied = cmd;
      setTimeout(() => { if (copied === cmd) copied = ''; }, 1600);
    } catch {}
  }

  const quantClass = (q) => {
    const k = (q || '').toLowerCase();
    if (k === 'nvfp4') return 'chip chip-nvfp4';
    if (k === 'fp8') return 'chip chip-fp8';
    if (k === 'bf16') return 'chip chip-bf16';
    return 'chip';
  };
  const quantLabel = (q) => (q && q !== 'none' ? q.toUpperCase() : 'BF16');
  const topoClass = (t) => (t === 'EP=2' ? 'chip chip-ep2' : t === 'TP=2' ? 'chip chip-tp2' : 'chip chip-single');
</script>

<section id="models" style="background: var(--bg2);">
  <div class="container">
    <div class="slabel">Supported Models</div>
    <h2 class="stitle">Model matrix</h2>
    <p class="ssub">
      Every model gets hand-tuned CUDA kernels. Each card is one model family;
      every recipe maps to a single
      <a href="https://github.com/Avarok-Cybersecurity/atlas-recipes" class="ssub-link">sparkrun recipe</a>
      you can copy and run as-is.
    </p>

    <div class="ms-carousel">
      <div class="ms-controls">
        <button type="button" class="ms-arrow" onclick={prev} aria-label="Previous family">‹</button>
        <div class="ms-dots" role="tablist" aria-label="Model families">
          {#each families as f, i}
            <button
              type="button"
              class="ms-dot {i === index ? 'is-active' : ''}"
              role="tab"
              aria-selected={i === index}
              aria-label={f.displayName}
              onclick={() => go(i)}
            ></button>
          {/each}
        </div>
        <button type="button" class="ms-arrow" onclick={next} aria-label="Next family">›</button>
      </div>

      <div class="ms-viewport mtable-scroll">
        <div class="ms-track" style="transform: translateX(-{index * 100}%);">
          {#each families as f}
            <div class="ms-slide">
              <div class="card ms-famcard">
                <div class="ms-accent" aria-hidden="true"></div>
                <div class="ms-famhead">
                  <h3>{f.displayName}</h3>
                  <span class="ms-count">{f.recipes.length} recipe{f.recipes.length === 1 ? '' : 's'}</span>
                </div>
                <div class="ms-grid">
                  {#each f.recipes as r}
                    <div class="subcard">
                      <div class="subcard-label">{r.displayName}</div>
                      <div class="subcard-meta">
                        <span class={quantClass(r.quant)}>{quantLabel(r.quant)}</span>
                        <span class={topoClass(r.topology)}>{r.topology}</span>
                        {#if r.params}<span class="chip chip-params">{r.params}</span>{/if}
                      </div>
                      <div class="subcard-hf mono" title={r.hfId}>{r.hfId}</div>
                      <div class="cmd-pill">
                        <code class="cmd-text mono">{r.command}</code>
                        <button
                          type="button"
                          class="cmd-copy"
                          onclick={() => copyCmd(r.command)}
                          aria-label={`Copy ${r.command}`}
                        >
                          {copied === r.command ? 'Copied' : 'Copy'}
                        </button>
                      </div>
                    </div>
                  {/each}
                </div>
              </div>
            </div>
          {/each}
        </div>
      </div>
    </div>

    <div class="ms-foot">
      All recipes are the single source of truth in
      <a href="https://github.com/Avarok-Cybersecurity/atlas-recipes" class="ssub-link">atlas-recipes</a>.
      Run any of them with
      <a href="https://sparkrun.dev/runtimes/atlas/" class="ssub-link">sparkrun</a>.
      EP=2 = Expert Parallelism across two GB10 nodes.
    </div>
  </div>
</section>
