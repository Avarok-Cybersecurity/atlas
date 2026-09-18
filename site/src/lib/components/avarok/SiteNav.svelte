<!--
  The global header. Desktop: a bar with mega menus that open on hover, focus
  or click. Mobile: a drawer with the same tree as accordions. Both render
  from `nav` in src/lib/content/brand.js, so a link exists once.

  Scoped styles on purpose: this component mounts on the marketing routes and
  on the developer routes (/engine, /control), which load different global
  stylesheets. Nothing here depends on either.
-->
<script>
  import { page } from '$app/state';
  import { nav, routes } from '$lib/content/brand.js';
  import AtlasLockup from '$shared/components/AtlasLockup.svelte';
  import ThemeToggle from '$shared/components/ThemeToggle.svelte';

  let open = $state(null); // label of the open mega menu
  let drawer = $state(false);
  let expanded = $state({}); // drawer accordion state
  let closeTimer;

  const path = $derived(page.url.pathname.replace(/\.html$/, '').replace(/\/$/, '') || '/');
  const isCurrent = (href) => href && !href.startsWith('http') && (href === '/' ? path === '/' : path === href || path.startsWith(href + '/'));
  const groupCurrent = (g) => (g.columns ?? []).some((c) => c.items.some((i) => isCurrent(i.href)));

  const canHover = () => typeof window !== 'undefined' && window.matchMedia('(hover: hover) and (min-width: 1101px)').matches;
  function enter(label) {
    if (!canHover()) return;
    clearTimeout(closeTimer);
    open = label;
  }
  function leave() {
    if (!canHover()) return;
    closeTimer = setTimeout(() => (open = null), 140);
  }
  // A mouse user hovers before they click, so by the time the click lands the
  // hover has already opened the menu, and a plain toggle would close it under
  // the cursor. On a device that hovers, a mouse click only ever opens; moving
  // away closes. A keyboard activation (detail 0) and a touch both toggle.
  function toggle(label, e) {
    const mouse = e && e.detail !== 0 && canHover();
    if (mouse && open === label) return;
    open = open === label ? null : label;
  }
  function closeAll() {
    open = null;
    drawer = false;
  }
  function onKey(e) {
    if (e.key === 'Escape') closeAll();
  }
  function onDocClick(e) {
    if (!e.target.closest('.av-header')) open = null;
  }
</script>

<svelte:window onkeydown={onKey} />
<svelte:document onclick={onDocClick} />

<header class="av-header" class:is-drawer={drawer}>
  <div class="av-header-in">
    <a class="av-brand" href={routes.home} aria-label="Avarok home" data-sveltekit-reload>
      <AtlasLockup kind="horizontal" width={152} />
    </a>

    <nav class="av-nav" aria-label="Main">
      <ul>
        {#each nav.groups as g}
          {#if g.columns}
            <li class="av-nav-item" class:is-open={open === g.label} onpointerenter={() => enter(g.label)} onpointerleave={leave}>
              <button
                type="button"
                class="av-nav-btn"
                class:is-current={groupCurrent(g)}
                aria-expanded={open === g.label}
                aria-controls={`av-mega-${g.label}`}
                onclick={(e) => toggle(g.label, e)}
              >
                {g.label}
                <svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="3" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="m6 9 6 6 6-6" /></svg>
              </button>
              <div class="av-mega" id={`av-mega-${g.label}`} hidden={open !== g.label}>
                <div class="av-mega-in" class:is-wide={g.columns.length > 1}>
                  {#each g.columns as col}
                    <div class="av-mega-col">
                      <h3>{col.heading}</h3>
                      {#each col.items as it}
                        <a
                          href={it.href}
                          class:is-accent={it.accent}
                          class:is-current={isCurrent(it.href)}
                          target={it.external ? '_blank' : undefined}
                          rel={it.external ? 'noopener' : undefined}
                          onclick={closeAll}
                        >
                          <span class="av-mega-t">{it.text}{#if it.external}<span aria-hidden="true"> ↗</span>{/if}</span>
                          {#if it.blurb}<span class="av-mega-b">{it.blurb}</span>{/if}
                        </a>
                      {/each}
                    </div>
                  {/each}
                </div>
              </div>
            </li>
          {:else}
            <li><a class="av-nav-link" class:is-current={isCurrent(g.href)} href={g.href} aria-current={isCurrent(g.href) ? 'page' : undefined}>{g.label}</a></li>
          {/if}
        {/each}
      </ul>
    </nav>

    <div class="av-header-actions">
      <ThemeToggle />
      <a class="av-cta" href={nav.cta.href}>{nav.cta.text}</a>
      <button type="button" class="av-burger" aria-expanded={drawer} aria-controls="av-drawer" aria-label={drawer ? 'Close menu' : 'Open menu'} onclick={() => (drawer = !drawer)}>
        <span class:is-x={drawer}><i></i><i></i><i></i></span>
      </button>
    </div>
  </div>

  <div id="av-drawer" class="av-drawer" hidden={!drawer}>
    {#each nav.groups as g}
      {#if g.columns}
        <div class="av-drawer-group">
          <button type="button" class="av-drawer-head" aria-expanded={!!expanded[g.label]} onclick={() => (expanded = { ...expanded, [g.label]: !expanded[g.label] })}>
            {g.label}
            <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="3" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="m6 9 6 6 6-6" /></svg>
          </button>
          <div class="av-drawer-items" hidden={!expanded[g.label]}>
            {#each g.columns as col}
              {#each col.items as it}
                <a href={it.href} target={it.external ? '_blank' : undefined} rel={it.external ? 'noopener' : undefined} onclick={closeAll}>{it.text}</a>
              {/each}
            {/each}
          </div>
        </div>
      {:else}
        <a class="av-drawer-link" href={g.href} onclick={closeAll}>{g.label}</a>
      {/if}
    {/each}
    <a class="av-cta av-cta-block" href={nav.cta.href} onclick={closeAll}>{nav.cta.text}</a>
  </div>
</header>
{#if drawer}
  <button type="button" class="av-scrim" aria-label="Close menu" onclick={closeAll}></button>
{/if}

<style>
  .av-header {
    /* 100 is the band the old nav used. Every modal on the developer pages sits
       at 110 or above and has to cover this bar, so do not raise it. */
    position: sticky; top: 0; z-index: 100;
    background: color-mix(in srgb, var(--bg) 84%, transparent);
    backdrop-filter: blur(14px) saturate(140%); -webkit-backdrop-filter: blur(14px) saturate(140%);
    border-bottom: 1px solid var(--border);
    font-family: var(--font-sans);
  }
  .av-header-in { max-width: 1180px; margin: 0 auto; padding: 0 24px; height: 68px; display: flex; align-items: center; gap: 1.5rem; }
  .av-brand { display: flex; align-items: center; text-decoration: none; color: inherit; flex-shrink: 0; }
  /* The bar's own padding is the clear space, so the lockup's margin is dropped here. */
  .av-brand :global(.logo) { margin: 0; }
  .av-nav { flex: 1; display: flex; justify-content: center; }
  .av-nav ul { display: flex; gap: 0.25rem; margin: 0; padding: 0; list-style: none; align-items: center; }
  .av-nav-btn, .av-nav-link {
    display: inline-flex; align-items: center; gap: 0.35rem; padding: 0.55rem 0.8rem; border-radius: 999px;
    background: none; border: 0; font: inherit; font-weight: 550; font-size: 0.92rem; color: var(--t2); cursor: pointer; text-decoration: none;
    transition: color 0.15s, background 0.15s;
  }
  .av-nav-btn:hover, .av-nav-link:hover, .av-nav-item.is-open .av-nav-btn { color: var(--t1); background: var(--bg2); }
  .av-nav-btn.is-current, .av-nav-link.is-current { color: var(--t1); }
  .av-nav-btn svg { transition: transform 0.15s; opacity: 0.7; }
  .av-nav-item.is-open .av-nav-btn svg { transform: rotate(180deg); }
  .av-nav-item { position: relative; }
  .av-mega { position: absolute; top: calc(100% + 10px); left: 50%; transform: translateX(-50%); z-index: 130; }
  .av-mega-in {
    display: grid; grid-template-columns: 1fr; gap: 1.5rem; min-width: 300px; padding: 1.1rem;
    background: var(--card); border: 1px solid var(--border-strong); border-radius: 18px;
    box-shadow: 0 30px 70px -30px rgba(0, 0, 0, 0.55), 0 1px 2px rgba(0, 0, 0, 0.08);
  }
  .av-mega-in.is-wide { grid-template-columns: 1fr 1fr; min-width: 640px; }
  .av-mega-col h3 { margin: 0 0 0.4rem; padding: 0 0.7rem; font-family: var(--font-mono); font-size: 0.66rem; letter-spacing: 0.12em; text-transform: uppercase; color: var(--t3); font-weight: 600; }
  .av-mega-col a { display: flex; flex-direction: column; gap: 0.15rem; padding: 0.6rem 0.7rem; border-radius: 10px; text-decoration: none; color: inherit; transition: background 0.15s; }
  .av-mega-col a:hover, .av-mega-col a.is-current { background: var(--bg2); }
  .av-mega-t { font-size: 0.92rem; font-weight: 600; color: var(--t1); }
  .av-mega-col a.is-accent .av-mega-t { color: var(--accent); }
  .av-mega-b { font-size: 0.78rem; color: var(--t3); line-height: 1.4; }
  .av-header-actions { display: flex; align-items: center; gap: 0.6rem; flex-shrink: 0; }
  .av-cta {
    display: inline-flex; align-items: center; padding: 0.6rem 1.1rem; border-radius: 999px; background: var(--accent-fill); color: #fff;
    font-weight: 600; font-size: 0.9rem; text-decoration: none; white-space: nowrap; transition: background 0.15s, transform 0.15s;
  }
  .av-cta:hover { background: var(--accent-fill-hover); transform: translateY(-1px); }
  .av-cta-block { justify-content: center; margin-top: 0.8rem; }
  .av-burger { display: none; width: 42px; height: 42px; border-radius: 10px; border: 1px solid var(--border-strong); background: var(--card); color: var(--t1); cursor: pointer; place-items: center; }
  .av-burger span { position: relative; display: block; width: 18px; height: 12px; }
  .av-burger i { position: absolute; left: 0; width: 100%; height: 2px; background: currentColor; border-radius: 2px; transition: transform 0.2s, opacity 0.15s, top 0.2s; }
  .av-burger i:nth-child(1) { top: 0; } .av-burger i:nth-child(2) { top: 5px; } .av-burger i:nth-child(3) { top: 10px; }
  .av-burger .is-x i:nth-child(1) { top: 5px; transform: rotate(45deg); }
  .av-burger .is-x i:nth-child(2) { opacity: 0; }
  .av-burger .is-x i:nth-child(3) { top: 5px; transform: rotate(-45deg); }
  .av-drawer { display: none; padding: 0.5rem 16px 1.25rem; border-top: 1px solid var(--border); background: var(--bg); max-height: calc(100dvh - 68px); overflow-y: auto; }
  .av-drawer-group { border-bottom: 1px solid var(--border); }
  .av-drawer-head { width: 100%; display: flex; justify-content: space-between; align-items: center; padding: 0.95rem 0.25rem; background: none; border: 0; font: inherit; font-weight: 600; font-size: 1rem; color: var(--t1); cursor: pointer; }
  .av-drawer-head[aria-expanded='true'] svg { transform: rotate(180deg); }
  .av-drawer-items { display: grid; padding: 0 0 0.6rem; }
  .av-drawer-items a { padding: 0.55rem 0.9rem; color: var(--t2); text-decoration: none; font-size: 0.95rem; border-radius: 8px; }
  .av-drawer-items a:hover { background: var(--bg2); color: var(--t1); }
  .av-drawer-link { display: block; padding: 0.95rem 0.25rem; border-bottom: 1px solid var(--border); font-weight: 600; color: var(--t1); text-decoration: none; }
  .av-scrim { position: fixed; inset: 68px 0 0; z-index: 95; background: rgba(0, 0, 0, 0.35); border: 0; cursor: pointer; }
  /* Seven items need about 1080 px beside the lockup and the actions. Below
     that the drawer takes over, before anything can crowd or wrap. */
  @media (max-width: 1100px) {
    .av-nav { display: none; }
    .av-burger { display: grid; }
    .av-header.is-drawer .av-drawer { display: block; }
    .av-header-actions .av-cta { display: none; }
  }
  @media (max-width: 480px) {
    .av-header-in { padding: 0 16px; height: 62px; }
    .av-scrim { top: 62px; }
    .av-drawer { max-height: calc(100dvh - 62px); }
  }
</style>
