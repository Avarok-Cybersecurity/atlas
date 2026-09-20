<!--
  Makes every email link on the page do something a visitor can see.

  A `mailto:` link hands the address to the operating system and returns. With
  no mail app registered, which is most people on a work laptop in a browser,
  the button does nothing at all and the visitor concludes it is broken. There
  is no way for a page to find out whether a mail app answered.

  So on any click of a `mailto:` link, the link still opens the mail app if
  there is one, and this also copies the address and says so, with the address
  in view. Whatever the machine does, the visitor leaves with the address.

  Mounted once, by PageShell. It listens on the document, so it covers links in
  content, cards and buttons without any of them knowing about it.
-->
<script>
  import { onMount } from 'svelte';

  let address = $state('');
  let copied = $state(false);
  let open = $state(false);
  let timer;

  async function copy(text) {
    try {
      await navigator.clipboard.writeText(text);
      return true;
    } catch {
      return false;
    }
  }

  onMount(() => {
    const onClick = async (e) => {
      const a = e.target instanceof Element ? e.target.closest('a[href^="mailto:"]') : null;
      if (!a) return;
      address = decodeURIComponent(a.getAttribute('href').slice(7).split('?')[0]);
      copied = await copy(address);
      open = true;
      clearTimeout(timer);
      timer = setTimeout(() => (open = false), 9000);
    };
    document.addEventListener('click', onClick);
    return () => {
      document.removeEventListener('click', onClick);
      clearTimeout(timer);
    };
  });
</script>

<div class="av-mailtoast" class:is-open={open} role="status" aria-live="polite">
  {#if open}
    <p>
      <strong>{copied ? 'Address copied.' : 'Write to this address.'}</strong>
      Your mail app should open. If it does not, paste this into a new message.
    </p>
    <p class="av-mailtoast-row">
      <span class="av-mono">{address}</span>
      <button type="button" class="av-btn av-btn-secondary av-btn-sm" onclick={async () => (copied = await copy(address))}>Copy</button>
      <button type="button" class="av-mailtoast-x" aria-label="Dismiss" onclick={() => (open = false)}>×</button>
    </p>
  {/if}
</div>

<style>
  .av-mailtoast {
    position: fixed; z-index: 120; left: 50%; bottom: 20px; transform: translate(-50%, 24px);
    /* Closed means gone, not parked below the fold: an empty box slid 140% of its
       own small height still showed a sliver along the bottom of every page. */
    visibility: hidden; opacity: 0;
    width: min(440px, calc(100vw - 32px)); padding: 0.9rem 1rem;
    background: var(--card); color: var(--t1); border: 1px solid var(--border-strong); border-radius: var(--av-radius-sm);
    box-shadow: var(--av-shadow); font-size: 0.86rem; line-height: 1.5;
    transition: transform 0.28s var(--av-ease), opacity 0.2s, visibility 0s linear 0.28s; pointer-events: none;
  }
  .av-mailtoast.is-open { transform: translate(-50%, 0); visibility: visible; opacity: 1; pointer-events: auto; transition: transform 0.28s var(--av-ease), opacity 0.2s, visibility 0s; }
  .av-mailtoast p { margin: 0; color: var(--t2); }
  .av-mailtoast strong { color: var(--t1); }
  .av-mailtoast-row { display: flex; align-items: center; gap: 0.6rem; margin-top: 0.55rem !important; }
  .av-mailtoast-row .av-mono { flex: 1; min-width: 0; overflow-wrap: anywhere; color: var(--t1); font-size: 0.84rem; }
  .av-mailtoast-x { border: 0; background: none; color: var(--t3); font-size: 1.2rem; line-height: 1; cursor: pointer; padding: 0.2rem 0.35rem; border-radius: 6px; }
  .av-mailtoast-x:hover { color: var(--t1); }
</style>
