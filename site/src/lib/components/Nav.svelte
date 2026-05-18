<script>
  import { discordUrl } from '$lib/data.js';
  import GithubIcon from './GithubIcon.svelte';
</script>

<nav>
  <div class="nav-inner">
    <a class="nav-logo" href="/">
      <img src="/favicon.svg" alt="" class="nav-mark" />
      <span class="nav-wordmark">Atlas</span>
    </a>
    <div class="nav-links">
      <a href="#speed">Why Atlas</a>
      <a href="#models">Models</a>
      <a href="#try">Try It</a>
      <a href="https://sparkrun.dev/runtimes/atlas/" target="_blank" rel="noopener">sparkrun</a>
      <a href="#community">Community</a>
      <a href="#roadmap">Roadmap</a>
      <a
        href="https://github.com/Avarok-Cybersecurity/atlas"
        class="nav-icon-link"
        target="_blank"
        rel="noopener"
        aria-label="Atlas on GitHub"
      >
        <GithubIcon size={20} />
      </a>
      <a href={discordUrl} class="nav-cta">Join Discord</a>
    </div>
  </div>
</nav>
