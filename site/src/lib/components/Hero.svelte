<script>
  import { modal } from '$lib/modal.js';
  import { hero, githubUrl, discordUrl } from '$lib/data.js';
  import { currentInstall } from '$lib/install/host.svelte.js';

  // Windows visitors were shown `curl … | sh`, which cannot run there.
  const install = $derived(currentInstall());
  import { copyLabel, copyOrSelect } from '$lib/clipboard.js';
  import Receipt from './Receipt.svelte';
  import DiscordIcon from './DiscordIcon.svelte';

  let copyState = $state('idle'); // idle | copied | manual | blocked
  let copyTimer;
  // The flash outlives the component on navigation without this.
  $effect(() => () => clearTimeout(copyTimer));
  let cmdEl = $state(null);
  let dashboardOpen = $state(false);

  // PRPL "lazy-load": the dashboard is on-click only, so its component (and the
  // gate-records JSON it pulls in) stays out of the initial bundle. Hovering or
  // focusing the receipt starts the fetch early, so by click time the module is
  // usually already here; the skeleton in dashboard.css covers the rest.
  let Dashboard = $state(null);
  let dashboardError = $state(false);
  let dashboardModule = null;
  function preloadDashboard() {
    dashboardModule ??= import('./BenchmarkDashboard.svelte');
    return dashboardModule;
  }
  async function openDashboard() {
    dashboardOpen = true;
    try {
      Dashboard = (await preloadDashboard()).default;
    } catch {
      dashboardModule = null; // allow retry on next open
      dashboardError = true;
    }
  }
  function closeDashboard() {
    dashboardOpen = false;
    dashboardError = false;
  }

  async function copy() {
    clearTimeout(copyTimer);
    // Was a silent `return` on refusal: the button did not change, which is
    // indistinguishable from success to the person who clicked it.
    copyState = await copyOrSelect(install.command, cmdEl);
    copyTimer = setTimeout(() => (copyState = 'idle'), 2400);
  }
</script>

<!-- The loaded dashboard handles Escape itself; this covers the skeleton phase. -->
<svelte:window onkeydown={(e) => { if (e.key === 'Escape' && dashboardOpen && !Dashboard) closeDashboard(); }} />

<section class="hero">

  <div class="hero-inner">
    <div class="hero-copy">
      <span class="hero-badge"><span class="dot"></span> {hero.badge}</span>
      <h1>{hero.headline[0]} <span class="lede2">{hero.headline[1]}</span></h1>
      <p class="hero-sub">{hero.sub}</p>

      <div class="hero-cmd" role="group" aria-label="Run Atlas">
        <span class="prompt">{install.prompt}</span>
        <code bind:this={cmdEl}>{install.command}</code>
        <button type="button" class="copy-btn" onclick={copy} aria-label="Copy run command">
          {copyLabel(copyState)}
        </button>
      </div>

      <div class="hero-buttons">
        <a class="btn btn-primary" href={githubUrl} target="_blank" rel="noopener">{hero.primaryCta}</a>
        <a class="btn btn-discord" href={discordUrl} target="_blank" rel="noopener"><DiscordIcon size={17} /> {hero.discordCta}</a>
        <a class="btn btn-ghost" href="#run">{hero.secondaryCta}</a>
      </div>

      <p class="hero-challenge">
        {hero.challenge.lead} <strong>{hero.challenge.claim}</strong>
        <span class="fine">{hero.challenge.fine}</span>
      </p>
    </div>

    <div class="hero-receipt">
      <div class="receipt-hit">
        <span class="receipt-chip" aria-hidden="true">⤢ expand</span>
        <Receipt compact={true} />
        <button
          type="button"
          class="receipt-open-hint"
          onclick={openDashboard}
          onpointerenter={preloadDashboard}
          onfocus={preloadDashboard}
          aria-haspopup="dialog"
        >
          <span class="rh-glyph" aria-hidden="true">⤢</span>
          view the benchmark dashboard
        </button>
      </div>
    </div>
  </div>
</section>

{#if dashboardOpen}
  {#if Dashboard}
    <Dashboard onclose={closeDashboard} />
  {:else}
    <!-- Same .bd-backdrop/.bd classes as the real dialog: identical dimensions,
         so the swap from skeleton to dashboard causes zero layout shift. -->
    <div class="bd-backdrop" onclick={closeDashboard} role="presentation">
      <div
        class="bd bd-skeleton"
        role="dialog"
        aria-modal="true"
        aria-label="Benchmark dashboard, loading"
        aria-busy="true"
        onclick={(e) => e.stopPropagation()}
        use:modal
      >
        {#if dashboardError}
          <p class="bd-skeleton-error">Couldn’t load the dashboard (network?). Close and try again.</p>
        {:else}
          <div class="bd-skeleton-bar" style="width: 38%"></div>
          <div class="bd-skeleton-bar" style="width: 62%"></div>
          <div class="bd-skeleton-chart"></div>
          <div class="bd-skeleton-chart"></div>
        {/if}
        <button type="button" class="bd-close bd-skeleton-close" onclick={closeDashboard} aria-label="Close dashboard">✕</button>
      </div>
    </div>
  {/if}
{/if}
