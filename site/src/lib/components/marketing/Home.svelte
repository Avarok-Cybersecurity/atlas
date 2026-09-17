<!-- SPDX-License-Identifier: AGPL-3.0-only -->
<script>
  import { onMount } from 'svelte';
  import { afterNavigate } from '$app/navigation';
  import { githubUrl as REPO, guideUrl as GUIDE, discordUrl, blogUrl, contactEmails, CONTROL } from '$lib/data.js';
  import ladder from '$lib/ladder.generated.json';
  import models from '$lib/models.generated.json';
  import { benchmarkHighlight, legacyEngineDestination, legacySections, brandStyle, ENGINE } from '$lib/marketing.js';
  import Icon from './Icon.svelte';
  import Possibilities from './Possibilities.svelte';
  import AtlasLockup from '$shared/components/AtlasLockup.svelte';
  import ThemeToggle from '$shared/components/ThemeToggle.svelte';
  import '../../../styles/marketing.css';

  const benchmark = benchmarkHighlight(ladder);
  // The hero is one image. Its source is picked from data-theme rather than
  // from CSS, so a page load fetches the render it is going to show and not
  // both of them. The light render is multiplied onto a near-white ground and
  // the dark one is a re-toned derivative levelled to the dark --bg.
  const HERO_LIGHT = '/brand/atlas-hero.webp';
  const HERO_DARK = '/brand/atlas-hero-dark.webp';
  const heroSrc = theme => (theme === 'dark' ? HERO_DARK : HERO_LIGHT);
  // Svelte has no way to put a raw script element in markup, so the parse time
  // pick ships as literal html. On a prerendered load it runs while the parser
  // is still on the hero, which is why the image carries no src of its own and
  // the head preload is the only fetch. On a client side navigation the html
  // arrives through innerHTML and the script does not run, so onMount below
  // sets the same source a second time.
  const heroPick =
    `<script>(function(){var i=document.getElementById('m-hero-img');` +
    `if(i)i.src=document.documentElement.getAttribute('data-theme')==='dark'` +
    `?'${HERO_DARK}':'${HERO_LIGHT}';})()<\/script>`;
  onMount(() => {
    const img = document.getElementById('m-hero-img');
    if (!img) return;
    // A theme flip after load swaps the source. The second render is fetched
    // at that moment, which is the cost of never fetching it on a load that
    // does not show it.
    const sync = () => {
      const next = heroSrc(document.documentElement.getAttribute('data-theme'));
      if (!img.src.endsWith(next)) img.src = next;
    };
    sync();
    // theme.js flips the attribute without announcing it, so the attribute is
    // what we watch.
    const watcher = new MutationObserver(sync);
    watcher.observe(document.documentElement, { attributeFilter: ['data-theme'] });
    return () => watcher.disconnect();
  });
  const featuredModels = models.filter(model => ['Qwen', 'Gemma', 'Mistral', 'Nemotron'].includes(model.vendor));
  function forwardLegacyFragment({ hash, search }) {
    if (hash === '#why-avarok') {
      window.location.replace(`${window.location.pathname}${search}#why-avarok`);
      return;
    }
    const destination = legacyEngineDestination(hash, search);
    if (destination) window.location.replace(destination);
  }
  afterNavigate(({ to }) => { if (to) forwardLegacyFragment(to.url); });
  function goHome(e) {
    if (window.location.pathname !== '/' && window.location.pathname !== '/index.html') return;
    e.preventDefault();
    if (window.location.hash) history.replaceState(null, '', window.location.pathname);
    window.scrollTo(0, 0);
  }
</script>

<svelte:window onhashchange={() => forwardLegacyFragment(window.location)} />

<div class="marketing" style={brandStyle}>
    <a class="m-skip-link" href="#main">Skip to content</a>
    <header class="m-header">
      <a class="m-brand" href="/" aria-label="Avarok home" onclick={goHome}>
        <AtlasLockup kind="corp" />
      </a>
      <nav aria-label="Main navigation"><a href="#why-avarok">Why Avarok</a><a href="#possibilities">Possibilities</a><a href={ENGINE} data-sveltekit-reload>Developers <Icon name="ArrowUpRight" size={13}/></a></nav>
      <div class="m-header-actions">
        <ThemeToggle />
        <a class="m-button m-button-dark m-nav-cta" href="#start">Start building <Icon name="ArrowUpRight" size={16}/></a>
      </div>
    </header>
    <main id="main">
      <section class="m-hero">
        <div class="m-hero-art"><img id="m-hero-img" class="m-hero-img" alt="" width="1536" height="1024" fetchpriority="high" decoding="async" />{@html heroPick}<noscript><img class="m-hero-img" src="/brand/atlas-hero-dark.webp" alt="" width="1536" height="1024" /></noscript></div>
        <div class="m-hero-copy">
          <a class="m-eyebrow m-hero-announcement" href={REPO} target="_blank" rel="noreferrer"><span class="m-status-dot"/> OPEN SOURCE. OPEN POSSIBILITIES. <Icon name="ArrowUpRight" size={13}/></a>
          <h1>Intelligence,<br/>on your <em>terms.</em></h1>
          <p class="m-hero-description">Your ideas deserve room to run. Avarok brings powerful AI to your own hardware, so you can build freely, move faster, and stay in control.</p>
          <div class="m-hero-actions"><a class="m-button m-button-primary" href="#start">Build with Avarok <Icon name="ArrowUpRight" size={18}/></a><a class="m-text-link" href="#why-avarok">Meet your engine <Icon name="ArrowRight" size={17}/></a></div>
          <div class="m-hero-note"><span/> Pure Rust. GPU native. Yours to run.</div>
        </div>
        <div class="m-art-caption"><span class="m-crosshair">+</span><span>LESS BETWEEN<br/>IDEA AND INTELLIGENCE.</span><span class="m-caption-index">AVAROK / 01</span></div>
      </section>
      <div id="models" class="m-ecosystem"><p>OPEN MODELS.<br/><strong>WIDE-OPEN POSSIBILITIES.</strong></p>{#each featuredModels as model}<div class="m-model-name">{model.vendor}</div>{/each}<a href={`${ENGINE}#models`} data-sveltekit-reload>Explore supported models <Icon name="ArrowUpRight" size={15}/></a></div>

      <section id="why-avarok" class="m-section m-why-section">
        <div class="m-section-heading"><div><p class="m-eyebrow m-section-label"><span>01 / WHY AVAROK</span></p><h2>Big ideas.<br/><span class="m-soft-text">No permission needed.</span></h2></div><p class="m-section-intro">The next wave of AI belongs to the people building it. We’re putting the engine in your hands.</p></div>
        <div class="m-benefits">
          <article><div class="m-benefit-top"><Icon name="Zap" strokeWidth={1.4} size={26}/><span>01</span></div><h3>Stay in your flow.</h3><p>Ideas move fast. Your engine should keep up. Rust and custom GPU kernels bring your models closer to the hardware.</p><a href={`${ENGINE}#verified`} target="_blank" rel="noreferrer">Explore the performance <Icon name="ArrowUpRight" size={16}/></a></article>
          <article><div class="m-benefit-top"><Icon name="Fingerprint" strokeWidth={1.4} size={28}/><span>02</span></div><h3>Own your next move.</h3><p>Your models, on your infrastructure. Keep control of where intelligence runs and how it fits into your world.</p><a href={GUIDE} target="_blank" rel="noreferrer">Find your setup <Icon name="ArrowUpRight" size={16}/></a></article>
          <article><div class="m-benefit-top"><Icon name="Code2" strokeWidth={1.4} size={27}/><span>03</span></div><h3>Build without the black box.</h3><p>See how it works. Shape what comes next. Avarok is open source, with the code, recipes, and benchmarks out in the open.</p><a href={REPO} target="_blank" rel="noreferrer">Get to know the code <Icon name="ArrowUpRight" size={16}/></a></article>
        </div>
      </section>
      <section id="possibilities" class="m-section m-possibilities m-dark">
        <div class="m-section-heading"><div><p class="m-eyebrow m-section-label">02 / OPEN POSSIBILITIES</p><h2>What will you<br/><span class="m-soft-text">set in motion?</span></h2></div><p class="m-section-intro">A better assistant. A bolder experiment.<br/>That thing you can’t stop thinking about.<br/>Give it an engine.</p></div>
        <Possibilities/>
        <div class="m-possibilities-bottom"><span>YOUR APPLICATION</span><Icon name="ArrowRight" size={16}/><span>AVAROK</span><Icon name="ArrowRight" size={16}/><span>YOUR HARDWARE</span><span class="m-bottom-note">A direct line to possibility.</span></div>
      </section>
      <section id="verified" class="m-section m-evidence">
        <div class="m-evidence-copy"><p class="m-eyebrow m-section-label">03 / REAL ENGINE. REAL EVIDENCE.</p><h2>Confidence,<br/><span class="m-soft-text">built right in.</span></h2><p>Big promises need something solid underneath. Avarok publishes its benchmarks, names the hardware, and checks every release against a committed baseline.</p><a class="m-text-link" href={ladder.results_doc_url} target="_blank" rel="noreferrer">See the work behind the numbers <Icon name="ArrowUpRight" size={17}/></a></div>
        <div class="m-evidence-card"><div class="m-evidence-kicker"><span class="m-status-dot"/> PUBLISHED GB10 BENCHMARK</div><div class="m-big-number">{benchmark.ratio.toFixed(3)}<span>×</span></div><h3>{benchmark.improved ? "More throughput" : "Relative throughput"} at {benchmark.concurrency} concurrent requests.</h3><div class="m-benchmark-bars"><div><span>Avarok</span><b style:width={`${benchmark.atlasWidth}%`}>{benchmark.avarok.toFixed(2)} tok/s</b></div><div><span>vLLM</span><b style:width={`${benchmark.baselineWidth}%`}>{benchmark.baseline.toFixed(2)} tok/s</b></div></div><p class="m-benchmark-note">{ladder.workload.checkpoint} · {ladder.box.gpu} · {ladder.aggregate} · {ladder.workload.isl_tokens} input / {ladder.workload.osl_tokens.toLocaleString("en-US")} output tokens. Compared with the matched vLLM + MTP configuration at C={benchmark.concurrency}. <a href={`${ENGINE}#verified`} target="_blank" rel="noreferrer">Full methodology ↗</a></p></div>
      </section>
      <section id="run" class="m-start-section"><span id="start"></span>
        <div class="m-start-top"><p class="m-eyebrow">THE FUTURE IS OPEN. MAKE IT YOURS.</p><span class="m-large-plus">+</span></div>
        <h2>Your next big thing<br/>starts <span>here.</span></h2>
        <div class="m-start-bottom"><p>Bring your curiosity.<br/>We’ll bring the engine.</p><div><a class="m-button m-button-white" href={GUIDE} target="_blank" rel="noreferrer">Get started with Avarok <Icon name="ArrowUpRight" size={20}/></a><a class="m-github-link" href={REPO} target="_blank" rel="noreferrer"><Icon name="Code2" size={17}/> Explore on GitHub <Icon name="ArrowUpRight" size={15}/></a></div></div>
        <div class="m-start-fineprint"><span><Icon name="Check" size={14}/> Open source under AGPL-3.0</span><span><Icon name="Check" size={14}/> Verified on NVIDIA DGX Spark</span><span><Icon name="Check" size={14}/> OpenAI-compatible API</span></div>
      </section>
    </main>
    <footer class="m-footer">
      <div class="m-footer-main"><div><a class="m-brand" href="/" aria-label="Avarok home" onclick={goHome}><AtlasLockup kind="corp" /></a><p>Intelligence, on your terms.</p></div><div class="m-footer-links"><div><span>BUILD</span><a href={ENGINE} data-sveltekit-reload>Engine and benchmarks</a><a href={CONTROL} data-sveltekit-reload>Fleet Manager</a><a href={GUIDE} target="_blank" rel="noreferrer">Documentation</a><a href={REPO} target="_blank" rel="noreferrer">GitHub</a><a href={`${ENGINE}#verified`} target="_blank" rel="noreferrer">Benchmarks</a></div><div><span>CONNECT</span><a href={discordUrl} target="_blank" rel="noreferrer">Discord</a><a href={blogUrl} target="_blank" rel="noreferrer">The Avarok blog</a><a href={`mailto:${contactEmails[0]}`}>Let’s talk <Icon name="ArrowUpRight" size={13}/></a></div></div></div>
      <div class="m-footer-bottom"><span>Avarok · Built in the open.</span><span>Community Edition · AGPL-3.0</span><a href="#main">Back to top ↑</a></div>
    </footer>
  <noscript>
    <nav class="m-legacy-links" aria-label="Avarok engine sections">
      {#each legacySections as section}
        <a id={section.id} href={`${ENGINE}#${section.id}`}>{section.label} ↗</a>
      {/each}
    </nav>
  </noscript>
</div>
