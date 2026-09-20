<!-- The global footer. Columns from `footer` in src/lib/content/brand.js. -->
<script>
  import { footer, links, company, routes } from '$lib/content/brand.js';
  import AtlasLockup from '$shared/components/AtlasLockup.svelte';
</script>

<footer class="av-footer">
  <div class="av-footer-in">
    <div class="av-footer-brand">
      <a href={routes.home} aria-label="Avarok home" data-sveltekit-reload><AtlasLockup kind="full" width={244} /></a>
      <p class="av-footer-slogan">{footer.slogan}</p>
      <p class="av-footer-lic">{footer.license}</p>
      <div class="av-footer-social">
        <a href={links.github} aria-label="Avarok on GitHub" target="_blank" rel="noopener">
          <svg width="18" height="18" viewBox="0 0 24 24" fill="currentColor" aria-hidden="true"><path d="M12 .5a12 12 0 0 0-3.8 23.4c.6.1.8-.3.8-.6v-2.2c-3.3.7-4-1.4-4-1.4-.6-1.4-1.4-1.8-1.4-1.8-1.1-.7.1-.7.1-.7 1.2.1 1.8 1.2 1.8 1.2 1.1 1.8 2.8 1.3 3.5 1 .1-.8.4-1.3.8-1.6-2.7-.3-5.5-1.3-5.5-5.9 0-1.3.5-2.4 1.2-3.2-.1-.3-.5-1.5.1-3.2 0 0 1-.3 3.3 1.2a11.5 11.5 0 0 1 6 0c2.3-1.5 3.3-1.2 3.3-1.2.7 1.7.3 2.9.1 3.2.8.8 1.2 1.9 1.2 3.2 0 4.6-2.8 5.6-5.5 5.9.4.4.8 1.1.8 2.2v3.3c0 .3.2.7.8.6A12 12 0 0 0 12 .5Z"/></svg>
        </a>
        <a href={links.discord} aria-label="Avarok on Discord" target="_blank" rel="noopener">
          <svg width="20" height="20" viewBox="0 0 24 24" fill="currentColor" aria-hidden="true"><path d="M20.3 4.4A19.8 19.8 0 0 0 15.4 3l-.2.4a13.3 13.3 0 0 1 4.5 2.3 15.9 15.9 0 0 0-15.4 0A13.3 13.3 0 0 1 8.8 3.4L8.6 3a19.8 19.8 0 0 0-4.9 1.4C.6 9 0 13.4.3 17.8a20 20 0 0 0 6 3l1.3-2a12.5 12.5 0 0 1-2-1l.5-.4a14.2 14.2 0 0 0 11.8 0l.5.4a12.5 12.5 0 0 1-2 1l1.3 2a20 20 0 0 0 6-3c.4-5-.7-9.4-3.4-13.4ZM8.5 15.2c-1.2 0-2.1-1.1-2.1-2.4s1-2.4 2.1-2.4 2.2 1.1 2.1 2.4c0 1.3-.9 2.4-2.1 2.4Zm7 0c-1.2 0-2.1-1.1-2.1-2.4s1-2.4 2.1-2.4 2.2 1.1 2.1 2.4c0 1.3-.9 2.4-2.1 2.4Z"/></svg>
        </a>
        <a href={links.x} aria-label="Avarok on X" target="_blank" rel="noopener">
          <svg width="18" height="18" viewBox="0 0 24 24" fill="currentColor" aria-hidden="true"><path d="M18.2 2h3.4l-7.4 8.5L23 22h-6.8l-5.3-7-6.1 7H1.4l7.9-9L1 2h7l4.8 6.4L18.2 2Zm-1.2 18h1.9L7.1 3.9H5.1L17 20Z"/></svg>
        </a>
      </div>
      <a class="av-footer-badge" href={links.inception} target="_blank" rel="noopener noreferrer">
        <img src="/nvidia-inception.webp" alt="NVIDIA Inception Program member" width="280" height="102" loading="lazy" />
      </a>
    </div>
    {#each footer.cols as col}
      <div class="av-footer-col">
        <h2>{col.heading}</h2>
        {#each col.links as l}
          <a href={l.href} target={l.external ? '_blank' : undefined} rel={l.external ? 'noopener' : undefined}>{l.text}{#if l.external}<span aria-hidden="true"> ↗</span>{/if}</a>
        {/each}
      </div>
    {/each}
  </div>
  <div class="av-footer-legal">
    <div class="av-footer-legal-row">
      <span>{footer.legal}</span>
      <span class="av-footer-links"><a href={routes.trust}>Trust center</a><a href={links.license} target="_blank" rel="noopener">License</a><a href={links.securityPolicy} target="_blank" rel="noopener">Security</a><a href="#top">Back to top ↑</a></span>
    </div>
    <p class="av-footer-tm">{footer.mlperf}</p>
  </div>
</footer>

<style>
  .av-footer { border-top: 1px solid var(--border); background: var(--bg2); font-family: var(--font-sans); color: var(--t2); }
  .av-footer-in { max-width: 1180px; margin: 0 auto; padding: 3.5rem 24px 2.5rem; display: grid; grid-template-columns: 1.6fr repeat(4, 1fr); gap: 2rem; }
  /* The child combinator matters. Without it this also matched the first social
     link, turned its centring grid into an inline block, and left the GitHub mark
     ten pixels above its neighbours. */
  .av-footer-brand > a:first-child { display: inline-block; text-decoration: none; color: inherit; }
  .av-footer-brand :global(.logo) { margin: 0; }
  .av-footer-slogan { margin: 1rem 0 0.4rem; font-size: 0.95rem; color: var(--t1); font-weight: 500; max-width: 26ch; }
  .av-footer-lic { margin: 0; font-size: 0.78rem; color: var(--t3); max-width: 32ch; line-height: 1.5; }
  .av-footer-social { display: flex; gap: 0.4rem; margin-top: 1rem; margin-left: -10px; }
  .av-footer-social a { display: inline-grid; place-items: center; width: 40px; height: 40px; color: var(--t2); border-radius: 10px; transition: color 0.15s, background 0.15s; }
  .av-footer-social a:hover { color: var(--accent); background: var(--card); }
  .av-footer-badge { display: inline-block; margin-top: 1.2rem; line-height: 0; opacity: 0.75; transition: opacity 0.15s; }
  .av-footer-badge:hover { opacity: 1; }
  .av-footer-badge img { width: 132px; height: auto; }
  :global(html:not([data-theme='light'])) .av-footer-badge img { filter: grayscale(1) invert(1) brightness(1.15); }
  /* h2, not h4. The footer follows a page whose last heading is an h2, and a
     jump to h4 is an invalid heading order. The size is set here, not by level. */
  .av-footer-col h2 { margin: 0 0 0.8rem; font-family: var(--font-mono); font-size: 0.66rem; letter-spacing: 0.12em; text-transform: uppercase; color: var(--t3); font-weight: 600; }
  .av-footer-col a { display: block; font-size: 0.86rem; color: var(--t2); text-decoration: none; margin-bottom: 0.5rem; transition: color 0.15s; }
  .av-footer-col a:hover { color: var(--accent); }
  .av-footer-legal { border-top: 1px solid var(--border); }
  .av-footer-legal-row { max-width: 1180px; margin: 0 auto; padding: 1.25rem 24px 0; display: flex; justify-content: space-between; gap: 1rem; flex-wrap: wrap; font-size: 0.76rem; color: var(--t3); }
  .av-footer-links { display: flex; gap: 1.1rem; flex-wrap: wrap; }
  .av-footer-links a { color: var(--t3); text-decoration: none; }
  .av-footer-links a:hover { color: var(--accent); }
  .av-footer-tm { max-width: 1180px; margin: 0.8rem auto 0; padding: 0 24px 1.5rem; font-size: 0.66rem; color: var(--t3); line-height: 1.5; }
  @media (max-width: 1000px) { .av-footer-in { grid-template-columns: 1fr 1fr 1fr; } .av-footer-brand { grid-column: 1 / -1; } }
  @media (max-width: 600px) { .av-footer-in { grid-template-columns: 1fr 1fr; padding: 2.5rem 16px 2rem; } .av-footer-legal-row, .av-footer-tm { padding-inline: 16px; } }
</style>
