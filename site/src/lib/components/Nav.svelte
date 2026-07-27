<script>
  // Desktop bar + mobile drawer render from the SAME `nav.links` in data.js.
  // Below the drawer breakpoint (styles/mobile.css) the bar hides and the
  // toggle appears, so phones keep every link the desktop has.
  import { nav, githubUrl, discordUrl, xUrl } from '$lib/data.js';
  import stars from '$lib/stars.generated.json';
  import GithubIcon from './GithubIcon.svelte';
  import DiscordIcon from './DiscordIcon.svelte';
  import XIcon from './XIcon.svelte';

  let open = $state(false);

  // No body scroll lock on purpose. Measured on this page, body.overflow hidden
  // is a no-op because scrolling lives on the viewport and body already carries
  // overflow-x clip, so the page keeps scrolling anyway. The drawer is a short
  // panel pinned under the bar rather than a full screen sheet, so there is
  // nothing to lock. The scrim absorbs stray taps and closes the menu.
</script>

<svelte:window onkeydown={(e) => { if (e.key === 'Escape') open = false; }} />

<nav>
  <div class="nav-inner">
    <a class="nav-logo" href="/" aria-label="Atlas home">
      <img class="nav-mark" src="/favicon.svg" alt="" width="34" height="34" />
      <span class="nav-wordmark">Atlas</span>
    </a>
    <div class="nav-links">
      {#each nav.links as l}
        <a href={l.href}>{l.text}</a>
      {/each}
      <a class="nav-icon-link" href={discordUrl} aria-label="Discord" target="_blank" rel="noopener"><DiscordIcon size={18} /></a>
      <a class="nav-icon-link" href={xUrl} aria-label="X / Twitter" target="_blank" rel="noopener"><XIcon size={16} /></a>
      <a class="nav-star-btn" href={githubUrl} target="_blank" rel="noopener">
        <GithubIcon size={15} /> Star <span class="nav-star-count">{stars.count}</span>
      </a>
    </div>

    <button
      type="button"
      class="nav-toggle"
      aria-expanded={open}
      aria-controls="nav-drawer"
      aria-label={open ? nav.closeLabel : nav.menuLabel}
      onclick={() => (open = !open)}
    >
      <span class="nav-burger" class:is-x={open} aria-hidden="true"><span></span><span></span><span></span></span>
    </button>
  </div>

  <div id="nav-drawer" class="nav-drawer" class:is-open={open}>
    {#each nav.links as l}
      <a class="nav-drawer-link" href={l.href} tabindex={open ? 0 : -1} onclick={() => (open = false)}>{l.text}</a>
    {/each}
    <div class="nav-drawer-foot">
      <a class="nav-star-btn" href={githubUrl} target="_blank" rel="noopener" tabindex={open ? 0 : -1}>
        <GithubIcon size={15} /> Star <span class="nav-star-count">{stars.count}</span>
      </a>
      <a class="nav-icon-link" href={discordUrl} aria-label="Discord" target="_blank" rel="noopener" tabindex={open ? 0 : -1}><DiscordIcon size={22} /></a>
      <a class="nav-icon-link" href={xUrl} aria-label="X / Twitter" target="_blank" rel="noopener" tabindex={open ? 0 : -1}><XIcon size={19} /></a>
    </div>
  </div>
</nav>

<button
  type="button"
  class="nav-scrim"
  class:is-on={open}
  aria-label={nav.closeLabel}
  tabindex={open ? 0 : -1}
  onclick={() => (open = false)}
></button>
