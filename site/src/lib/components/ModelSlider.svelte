<script>
  // Data is SSOT-derived from github.com/Avarok-Cybersecurity/atlas-recipes
  // via site/scripts/gen-models.mjs -> models.generated.json.
  // 3-level tree: vendor (brand) -> subfamily (recipe dir) -> recipes.
  import vendors from '$lib/models.generated.json';

  // --- inline brand/model marks (no external URL/CDN — static site) ---------
  // Monochrome, viewBox 0 0 24 24, fill=currentColor -> inherits the dark
  // theme + accent. Resolved by the `icon` key emitted in models.generated.json.
  const ICONS = {
    qwen:
      'M12.604 1.34c.393.69.784 1.382 1.174 2.075a.18.18 0 00.157.091h5.552c.174 0 .322.11.446.327l1.454 2.57c.19.337.24.478.024.837-.26.43-.513.864-.76 1.3l-.367.658c-.106.196-.223.28-.04.512l2.652 4.637c.172.301.111.494-.043.77-.437.785-.882 1.564-1.335 2.34-.159.272-.352.375-.68.37-.777-.016-1.552-.01-2.327.016a.099.099 0 00-.081.05 575.097 575.097 0 01-2.705 4.74c-.169.293-.38.363-.725.364-.997.003-2.002.004-3.017.002a.537.537 0 01-.465-.271l-1.335-2.323a.09.09 0 00-.083-.049H4.982c-.285.03-.553-.001-.805-.092l-1.603-2.77a.543.543 0 01-.002-.54l1.207-2.12a.198.198 0 000-.197 550.951 550.951 0 01-1.875-3.272l-.79-1.395c-.16-.31-.173-.496.095-.965.465-.813.927-1.625 1.387-2.436.132-.234.304-.334.584-.335a338.3 338.3 0 012.589-.001.124.124 0 00.107-.063l2.806-4.895a.488.488 0 01.422-.246c.524-.001 1.053 0 1.583-.006L11.704 1c.341-.003.724.032.9.34zm-3.432.403a.06.06 0 00-.052.03L6.254 6.788a.157.157 0 01-.135.078H3.253c-.056 0-.07.025-.041.074l5.81 10.156c.025.042.013.062-.034.063l-2.795.015a.218.218 0 00-.2.116l-1.32 2.31c-.044.078-.021.118.068.118l5.716.008c.046 0 .08.02.104.061l1.403 2.454c.046.081.092.082.139 0l5.006-8.76.783-1.382a.055.055 0 01.096 0l1.424 2.53a.122.122 0 00.107.062l2.763-.02a.04.04 0 00.035-.02.041.041 0 000-.04l-2.9-5.086a.108.108 0 010-.113l.293-.507 1.12-1.977c.024-.041.012-.062-.035-.062H9.2c-.059 0-.073-.026-.043-.077l1.434-2.505a.107.107 0 000-.114L9.225 1.774a.06.06 0 00-.053-.031z',
    gemma:
      'M12.34 5.953a8.233 8.233 0 01-.247-1.125V3.72a8.25 8.25 0 015.562 2.232H12.34zm-.69 0c.113-.373.199-.755.257-1.145V3.72a8.25 8.25 0 00-5.562 2.232h5.304zm-5.433.187h5.373a7.98 7.98 0 01-.267.696 8.41 8.41 0 01-1.76 2.65L6.216 6.14zm-.264-.187H2.977v.187h2.915a8.436 8.436 0 00-2.357 5.767H0v.186h3.535a8.436 8.436 0 002.357 5.767H2.977v.186h2.976v2.977h.187v-2.915a8.436 8.436 0 005.767 2.357V24h.186v-3.535a8.436 8.436 0 005.767-2.357v2.915h.186v-2.977h2.977v-.186h-2.915a8.436 8.436 0 002.357-5.767H24v-.186h-3.535a8.436 8.436 0 00-2.357-5.767h2.915v-.187h-2.977V2.977h-.186v2.915a8.436 8.436 0 00-5.767-2.357V0h-.186v3.535A8.436 8.436 0 006.14 5.892V2.977h-.187v2.976zm6.14 14.326a8.25 8.25 0 005.562-2.233H12.34c-.108.367-.19.743-.247 1.126v1.107zm-.186-1.087a8.015 8.015 0 00-.258-1.146H6.345a8.25 8.25 0 005.562 2.233v-1.087zm-8.186-7.285h1.107a8.23 8.23 0 001.125-.247V6.345a8.25 8.25 0 00-2.232 5.562zm1.087.186H3.72a8.25 8.25 0 002.232 5.562v-5.304a8.012 8.012 0 00-1.145-.258zm15.47-.186a8.25 8.25 0 00-2.232-5.562v5.315c.367.108.743.19 1.126.247h1.107zm-1.086.186c-.39.058-.772.144-1.146.258v5.304a8.25 8.25 0 002.233-5.562h-1.087zm-1.332 5.69V12.41a7.97 7.97 0 00-.696.267 8.409 8.409 0 00-2.65 1.76l3.346 3.346zm0-6.18v-5.45l-.012-.013h-5.451c.076.235.162.468.26.696a8.698 8.698 0 001.819 2.688 8.698 8.698 0 002.688 1.82c.228.097.46.183.696.259zM6.14 17.848V12.41c.235.078.468.167.696.267a8.403 8.403 0 012.688 1.799 8.404 8.404 0 011.799 2.688c.1.228.19.46.267.696H6.152l-.012-.012zm0-6.245V6.326l3.29 3.29a8.716 8.716 0 01-2.594 1.728 8.14 8.14 0 01-.696.259zm6.257 6.257h5.277l-3.29-3.29a8.716 8.716 0 00-1.728 2.594 8.135 8.135 0 00-.259.696zm-2.347-7.81a9.435 9.435 0 01-2.88 1.96 9.14 9.14 0 012.88 1.94 9.14 9.14 0 011.94 2.88 9.435 9.435 0 011.96-2.88 9.14 9.14 0 012.88-1.94 9.435 9.435 0 01-2.88-1.96 9.434 9.434 0 01-1.96-2.88 9.14 9.14 0 01-1.94 2.88z',
    nemotron:
      'M10.212 8.976V7.62c.127-.01.256-.017.388-.021 3.596-.117 5.957 3.184 5.957 3.184s-2.548 3.647-5.282 3.647a3.227 3.227 0 01-1.063-.175v-4.109c1.4.174 1.681.812 2.523 2.258l1.873-1.627a4.905 4.905 0 00-3.67-1.846 6.594 6.594 0 00-.729.044m0-4.476v2.025c.13-.01.259-.019.388-.024 5.002-.174 8.261 4.226 8.261 4.226s-3.743 4.69-7.643 4.69c-.338 0-.675-.031-1.007-.092v1.25c.278.038.558.057.838.057 3.629 0 6.253-1.91 8.794-4.169.421.347 2.146 1.193 2.501 1.564-2.416 2.083-8.048 3.763-11.24 3.763-.308 0-.603-.02-.894-.048V19.5H24v-15H10.21zm0 9.756v1.068c-3.356-.616-4.287-4.21-4.287-4.21a7.173 7.173 0 014.287-2.138v1.172h-.005a3.182 3.182 0 00-2.502 1.178s.615 2.276 2.507 2.931m-5.961-3.3c1.436-1.935 3.604-3.148 5.961-3.336V6.523C5.81 6.887 2 10.723 2 10.723s2.158 6.427 8.21 7.015v-1.166C5.77 16 4.25 10.958 4.25 10.958h-.002z',
    mistral:
      'M3.428 3.4h3.429v3.428h3.429v3.429h-.002 3.431V6.828h3.427V3.4h3.43v13.714H24v3.429H13.714v-3.428h-3.428v-3.429h-3.43v3.428h3.43v3.429H0v-3.429h3.428V3.4zm10.286 13.715h3.428v-3.429h-3.427v3.429z',
    minimax:
      'M16.278 2c1.156 0 2.093.927 2.093 2.07v12.501a.74.74 0 00.744.709.74.74 0 00.743-.709V9.099a2.06 2.06 0 012.071-2.049A2.06 2.06 0 0124 9.1v6.561a.649.649 0 01-.652.645.649.649 0 01-.653-.645V9.1a.762.762 0 00-.766-.758.762.762 0 00-.766.758v7.472a2.037 2.037 0 01-2.048 2.026 2.037 2.037 0 01-2.048-2.026v-12.5a.785.785 0 00-.788-.753.785.785 0 00-.789.752l-.001 15.904A2.037 2.037 0 0113.441 22a2.037 2.037 0 01-2.048-2.026V18.04c0-.356.292-.645.652-.645.36 0 .652.289.652.645v1.934c0 .263.142.506.372.638.23.131.514.131.744 0a.734.734 0 00.372-.638V4.07c0-1.143.937-2.07 2.093-2.07zm-5.674 0c1.156 0 2.093.927 2.093 2.07v11.523a.648.648 0 01-.652.645.648.648 0 01-.652-.645V4.07a.785.785 0 00-.789-.78.785.785 0 00-.789.78v14.013a2.06 2.06 0 01-2.07 2.048 2.06 2.06 0 01-2.071-2.048V9.1a.762.762 0 00-.766-.758.762.762 0 00-.766.758v3.8a2.06 2.06 0 01-2.071 2.049A2.06 2.06 0 010 12.9v-1.378c0-.357.292-.646.652-.646.36 0 .653.29.653.646V12.9c0 .418.343.757.766.757s.766-.339.766-.757V9.099a2.06 2.06 0 012.07-2.048 2.06 2.06 0 012.071 2.048v8.984c0 .419.343.758.767.758.423 0 .766-.339.766-.758V4.07c0-1.143.937-2.07 2.093-2.07z'
  };

  // selected vendor + per-vendor selected subfamily (Svelte 5 runes)
  let selectedVendor = $state(0);
  let selectedSub = $state(vendors.map(() => 0));
  let copied = $state('');

  const vendor = $derived(vendors[selectedVendor]);
  const sub = $derived(vendor.subfamilies[selectedSub[selectedVendor]] ?? vendor.subfamilies[0]);

  function pickVendor(i) {
    selectedVendor = i;
  }
  function pickSub(i) {
    selectedSub[selectedVendor] = i;
  }

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
      Every model gets hand-tuned CUDA kernels. Pick a vendor, then a model
      family; every recipe maps to a single
      <a href="https://github.com/Avarok-Cybersecurity/atlas-recipes" class="ssub-link">sparkrun recipe</a>
      you can copy and run as-is.
    </p>

    <div class="mnav">
      <!-- Level 1: vendor brand tabs -->
      <div class="mnav-vendors" role="tablist" aria-label="Model vendors">
        {#each vendors as v, i}
          <button
            type="button"
            class="mnav-vendor {i === selectedVendor ? 'is-active' : ''}"
            role="tab"
            aria-selected={i === selectedVendor}
            onclick={() => pickVendor(i)}
          >
            <svg class="mnav-ico" viewBox="0 0 24 24" aria-hidden="true" fill="currentColor">
              <path d={ICONS[v.icon]} />
            </svg>
            <span>{v.vendor}</span>
          </button>
        {/each}
      </div>

      <!-- Level 2: subfamily sub-tabs (within the selected vendor) -->
      <div class="mnav-subs" role="tablist" aria-label={`${vendor.vendor} model families`}>
        {#each vendor.subfamilies as sf, i}
          <button
            type="button"
            class="mnav-sub {i === (selectedSub[selectedVendor] ?? 0) ? 'is-active' : ''}"
            role="tab"
            aria-selected={i === (selectedSub[selectedVendor] ?? 0)}
            onclick={() => pickSub(i)}
          >
            {sf.name}
            <span class="mnav-sub-count">{sf.recipes.length}</span>
          </button>
        {/each}
      </div>

      <!-- Level 3: recipe sub-cards (within the selected subfamily) -->
      <div class="card ms-famcard">
        <div class="ms-accent" aria-hidden="true"></div>
        <div class="ms-famhead">
          <h3>{vendor.vendor} · {sub.name}</h3>
          <span class="ms-count">{sub.recipes.length} recipe{sub.recipes.length === 1 ? '' : 's'}</span>
        </div>
        <div class="ms-grid">
          {#each sub.recipes as r (r.recipeStem)}
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

    <div class="ms-foot">
      All recipes are the single source of truth in
      <a href="https://github.com/Avarok-Cybersecurity/atlas-recipes" class="ssub-link">atlas-recipes</a>.
      Run any of them with
      <a href="https://sparkrun.dev/runtimes/atlas/" class="ssub-link">sparkrun</a>.
      EP=2 = Expert Parallelism across two GB10 nodes.
    </div>
  </div>
</section>
