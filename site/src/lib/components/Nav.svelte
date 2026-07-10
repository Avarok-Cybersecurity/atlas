<script>
  import { githubUrl, discordUrl, xUrl } from '$lib/data.js';
  import stars from '$lib/stars.generated.json';
  import GithubIcon from './GithubIcon.svelte';
  import DiscordIcon from './DiscordIcon.svelte';
  import XIcon from './XIcon.svelte';
</script>

<nav>
  <div class="nav-inner">
    <a class="nav-logo" href="/" aria-label="Atlas home">
      <img class="nav-mark" src="/favicon.svg" alt="" width="34" height="34" />
      <span class="nav-wordmark">Atlas</span>
    </a>
    <div class="nav-links">
      <a href="#verified">Verified</a>
      <a href="#hardware">Hardware</a>
      <a href="#models">Models</a>
      <a href="#run">Get running</a>
      <a href="#contribute">Contribute</a>
      <a class="nav-icon-link" href={discordUrl} aria-label="Discord" target="_blank" rel="noopener"><DiscordIcon size={18} /></a>
      <a class="nav-icon-link" href={xUrl} aria-label="X / Twitter" target="_blank" rel="noopener"><XIcon size={16} /></a>
      <a class="nav-star-btn" href={githubUrl} target="_blank" rel="noopener">
        <GithubIcon size={15} /> Star <span class="nav-star-count">{stars.count}</span>
      </a>
    </div>
  </div>
</nav>
