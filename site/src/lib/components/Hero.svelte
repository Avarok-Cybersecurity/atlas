<script>
  import { discordUrl, quickInstall } from '$lib/data.js';
  import DiscordIcon from './DiscordIcon.svelte';

  let copied = $state(false);
  async function copy() {
    try {
      await navigator.clipboard.writeText(quickInstall);
      copied = true;
      setTimeout(() => (copied = false), 1600);
    } catch {}
  }
</script>

<section class="hero">
  <div class="hero-content">
    <div class="hero-badge">
      <span class="dot"></span> Pure CUDA + Rust. Zero python dependencies, zero complex recipes.
    </div>
    <h1>
      Inference at<br />
      <span class="gradient">unimaginable speeds</span>
    </h1>
    <p>
      An LLM inference engine written from scratch in Rust and CUDA. No PyTorch. No Python. Just a
      ~2.5 GB image that runs 3x faster than the status quo.
    </p>

    <div class="hero-install" role="group" aria-label="Install sparkrun">
      <span class="hero-install-prompt mono">$</span>
      <code class="hero-install-cmd mono">{quickInstall}</code>
      <button type="button" class="hero-install-copy" onclick={copy} aria-label="Copy sparkrun install command">
        {copied ? 'Copied' : 'Copy'}
      </button>
    </div>
    <div class="hero-install-sub">
      <a href="https://sparkrun.dev/runtimes/atlas/" class="hero-install-link">sparkrun</a>
      pulls &amp; runs the ~2.5 GB Atlas image for you. Run command
      <a href="#try" class="hero-install-link">below</a>.
    </div>

    <div class="hero-buttons">
      <a href={discordUrl} class="btn btn-primary">
        <DiscordIcon />
        Join Discord
      </a>
      <a href="#try" class="btn btn-secondary">Run it now</a>
    </div>
  </div>
</section>
