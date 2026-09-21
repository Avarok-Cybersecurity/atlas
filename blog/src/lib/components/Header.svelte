<script>
  import { page } from '$app/state';
  import { nav, githubUrl, MAIN_SITE, navCurrent } from '$lib/content.js';
  import GithubIcon from './GithubIcon.svelte';
  import AtlasLockup from '$shared/components/AtlasLockup.svelte';
  import ThemeToggle from '$shared/components/ThemeToggle.svelte';

  const current = (href) => navCurrent(page.url.pathname, href);
  // Local review runs the marketing site on :5173. Production always uses MAIN_SITE.
  const landing = import.meta.env.DEV ? 'http://127.0.0.1:5173/' : MAIN_SITE;
</script>

<header class="hdr">
  <div class="hdr-in">
    <div class="brand">
      <a class="brand-mark" href={landing} aria-label="Atlas home">
        <AtlasLockup kind="horizontal" />
      </a>
      <span class="brand-div" aria-hidden="true"></span>
      <a class="brand-sub" href="/" aria-current={current('/') ? 'page' : undefined}>Blog</a>
    </div>

    <nav class="nav" aria-label="Categories">
      {#each nav as l}
        <a href={l.href} aria-current={current(l.href) ? 'page' : undefined}>{l.label}</a>
      {/each}
    </nav>

    <div class="hdr-right">
      <ThemeToggle />
      <!-- The label is display:none below 460px, which removes it from the
           accessibility tree as well as the page — so the name has to be on the
           element, or the link has no accessible name at exactly the widths
           where it is only an arrow. -->
      <a class="btn-ghost" href={MAIN_SITE} aria-label="atlascybernetics.ai">
        <span class="lbl">atlascybernetics.ai</span>
        <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
          <path d="M7 17L17 7M17 7H8M17 7v9" />
        </svg>
      </a>
      <a class="btn-ghost" href={githubUrl} target="_blank" rel="noopener" aria-label="Atlas on GitHub">
        <GithubIcon size={14} />
      </a>
    </div>
  </div>
</header>
