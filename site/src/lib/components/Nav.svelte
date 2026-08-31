<script>
  import { modal } from '$lib/modal.js';
  import AtlasLockup from '$shared/components/AtlasLockup.svelte';
  // Desktop bar + mobile drawer render from the SAME `nav.links` in data.js.
  // Below the drawer breakpoint (styles/mobile.css) the bar hides and the
  // toggle appears, so phones keep every link the desktop has.
  import { nav, githubUrl, discordUrl, xUrl, codeChat } from '$lib/data.js';
  import stars from '$lib/stars.generated.json';
  import GithubIcon from './GithubIcon.svelte';
  import DiscordIcon from './DiscordIcon.svelte';
  import XIcon from './XIcon.svelte';
  import ChatLatticeIcon from './ChatLatticeIcon.svelte';
  import FleetPill from './FleetPill.svelte';
  import { preloadChat, prefetchWasmOnIdle } from '../chat/warmup.js';

  let open = $state(false);

  // "Ask the codebase" modal, Hero's lazy pattern. The chunk (engine, wasm
  // glue, chat copy) stays out of the initial bundle. Hover or focus on the
  // trigger warms the import via chat/warmup.js (the preload SSOT, which also
  // idle-prefetches the wasm binary at low priority), so by click time the
  // module is usually here and the skeleton in chat.css covers the rest.
  // chatReady lights the status dot once the corpus has mounted this visit
  // (CodeChat reports it, so the engine module itself is never imported
  // eagerly). The corpus itself NEVER downloads before the modal opens.
  let chatOpen = $state(false);
  let Chat = $state(null);
  let chatError = $state(false);
  let chatReady = $state(false);
  function warmChat() {
    preloadChat().catch(() => {}); // memoized; a failed warm retries on open
    prefetchWasmOnIdle();
  }
  async function openChat() {
    open = false;
    chatOpen = true;
    chatError = false;
    try {
      Chat = (await preloadChat()).default;
    } catch {
      chatError = true; // preloadChat dropped its memo, next open retries
    }
  }
  function closeChat() {
    chatOpen = false;
    chatError = false;
  }

  // No body scroll lock on purpose. Measured on this page, body.overflow hidden
  // is a no-op because scrolling lives on the viewport and body already carries
  // overflow-x clip, so the page keeps scrolling anyway. The drawer is a short
  // panel pinned under the bar rather than a full screen sheet, so there is
  // nothing to lock. The scrim absorbs stray taps and closes the menu.
</script>

<!-- The loaded CodeChat handles Escape itself; this covers the drawer and the skeleton phase. -->
<svelte:window onkeydown={(e) => { if (e.key === 'Escape') { open = false; if (chatOpen && !Chat) closeChat(); } }} />

<nav>
  <div class="nav-inner">
    <a class="nav-logo" href="/" aria-label="Atlas home">
      <AtlasLockup kind="horizontal" width={122} label="Atlas" />
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
      <button
        type="button"
        class="nav-chat-btn"
        aria-label={codeChat.navLabel}
        aria-haspopup="dialog"
        onpointerenter={warmChat}
        onfocus={warmChat}
        onclick={openChat}
      >
        <ChatLatticeIcon size={18} />
        {#if chatReady}<span class="nav-chat-dot" aria-hidden="true"></span>{/if}
      </button>
      <!-- Fleet status chip. Last on purpose: status reads as chrome, not as a
           destination, so it anchors the far right of the bar instead of
           sitting between the links and the icons. Renders nothing for
           visitors without a paired local agent (see FleetPill.svelte). -->
      <FleetPill />
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
      <button
        type="button"
        class="nav-chat-btn"
        aria-label={codeChat.navLabel}
        aria-haspopup="dialog"
        tabindex={open ? 0 : -1}
        onfocus={warmChat}
        onclick={openChat}
      >
        <ChatLatticeIcon size={20} />
        {#if chatReady}<span class="nav-chat-dot" aria-hidden="true"></span>{/if}
      </button>
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

{#if chatOpen}
  {#if Chat}
    <Chat onclose={closeChat} onready={() => (chatReady = true)} />
  {:else}
    <!-- Same .cc-backdrop/.cc classes as the real dialog: identical geometry,
         so the swap from skeleton to chat causes zero layout shift. -->
    <div class="cc-backdrop" onclick={closeChat} role="presentation">
      <div
        class="cc cc-skeleton"
        role="dialog"
        aria-modal="true"
        aria-label="{codeChat.navLabel}, loading"
        use:modal
        aria-busy="true"
        onclick={(e) => e.stopPropagation()}
      >
        {#if chatError}
          <p class="cc-skeleton-error">{codeChat.loadFail}</p>
        {:else}
          <div class="bd-skeleton-bar" style="width: 34%"></div>
          <div class="bd-skeleton-bar" style="width: 58%"></div>
          <div class="bd-skeleton-chart"></div>
        {/if}
        <button type="button" class="bd-close cc-skeleton-close" onclick={closeChat} aria-label={codeChat.closeLabel}>✕</button>
      </div>
    </div>
  {/if}
{/if}
