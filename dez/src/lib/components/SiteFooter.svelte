<!-- SPDX-License-Identifier: AGPL-3.0-only -->
<script lang="ts">
  import { LINKS, NAME } from '$lib/site';
  import Wordmark from './Wordmark.svelte';
</script>

<footer class="footer">
  <div class="container inner">
    <div class="col">
      <Wordmark size={22} />
      <p class="note">
        {NAME} is an in-development IDE built on the Atlas Inference Engine. Nothing on this page
        is a shipping product.
      </p>
    </div>

    <nav class="col links" aria-label="Project links">
      <h2 class="eyebrow">Atlas</h2>
      <a href={LINKS.atlasRepo} rel="noopener noreferrer">GitHub repository</a>
      <a href={LINKS.atlasSite} rel="noopener noreferrer">atlasinference.io</a>
      <a href={LINKS.discord} rel="noopener noreferrer">Community Discord</a>
    </nav>

    <nav class="col links" aria-label="Reference">
      <h2 class="eyebrow">Reference</h2>
      <a href={LINKS.atlasLicense} rel="noopener noreferrer">Licence (AGPL-3.0)</a>
      <a href={LINKS.webgpu} rel="noopener noreferrer">WebGPU specification</a>
    </nav>
  </div>

  <div class="container legal">
    <p>Licensed AGPL-3.0-only, like Atlas. No trackers, no analytics, no third-party requests.</p>
  </div>
</footer>

<style>
  .footer {
    border-top: 1px solid var(--border);
    background: var(--bg-sunken);
    padding-block: clamp(2.5rem, 6vw, 4rem) 2rem;
  }

  .inner {
    display: grid;
    gap: 2rem;
    grid-template-columns: 1fr;
  }

  .col {
    display: flex;
    flex-direction: column;
    gap: 0.7rem;
    align-items: flex-start;
  }

  .note {
    max-width: 34ch;
    color: var(--fg-muted);
    font-size: var(--step--1);
  }

  .links a {
    color: var(--fg-muted);
    font-size: var(--step--1);
    text-decoration: none;
    transition: color 0.15s var(--ease);
  }

  .links a:hover {
    color: var(--fg);
    text-decoration: underline;
  }

  .links h2 {
    margin-bottom: 0.1rem;
  }

  .legal {
    margin-top: 2.5rem;
    padding-top: 1.25rem;
    border-top: 1px solid var(--border);
    color: var(--fg-muted);
    font-size: var(--step--1);
  }

  @media (min-width: 46rem) {
    .inner {
      grid-template-columns: 2fr 1fr 1fr;
      gap: 3rem;
    }
  }
</style>
