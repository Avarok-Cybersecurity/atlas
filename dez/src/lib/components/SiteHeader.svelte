<!-- SPDX-License-Identifier: AGPL-3.0-only -->
<script lang="ts">
  import { LINKS, STATUS } from '$lib/site';
  import ThemeToggle from './ThemeToggle.svelte';
  import Wordmark from './Wordmark.svelte';
</script>

<header class="header">
  <div class="container bar">
    <a class="brand" href="#top" aria-label="Dez — home">
      <Wordmark />
    </a>

    <nav class="nav" aria-label="Sections">
      <a href="#what">What it is</a>
      <a href="#features">Features</a>
      <a href="#how">How it works</a>
    </nav>

    <div class="actions">
      <span class="status" title={STATUS.detail}>
        <span class="dot" aria-hidden="true"></span>
        {STATUS.label}
      </span>
      <ThemeToggle />
      <a class="ghost" href={LINKS.atlasRepo} rel="noopener noreferrer">
        GitHub<span class="visually-hidden"> — Atlas repository (opens on github.com)</span>
      </a>
    </div>
  </div>
</header>

<style>
  .header {
    position: sticky;
    top: 0;
    z-index: 20;
    background: color-mix(in srgb, var(--bg) 86%, transparent);
    backdrop-filter: blur(10px);
    border-bottom: 1px solid var(--border);
  }

  .bar {
    display: flex;
    align-items: center;
    gap: 1rem;
    min-height: 3.75rem;
  }

  .brand {
    color: var(--fg);
    text-decoration: none;
    flex: 0 0 auto;
  }

  .nav {
    display: none;
    gap: 1.4rem;
    margin-inline: auto;
    font-size: var(--step--1);
  }

  .nav a {
    color: var(--fg-muted);
    text-decoration: none;
    transition: color 0.15s var(--ease);
  }

  .nav a:hover {
    color: var(--fg);
  }

  .actions {
    display: flex;
    align-items: center;
    gap: 0.6rem;
    margin-left: auto;
  }

  .status {
    display: none;
    align-items: center;
    gap: 0.45rem;
    padding: 0.25rem 0.65rem;
    border: 1px solid var(--border);
    border-radius: 999px;
    background: var(--accent-wash);
    color: var(--fg-muted);
    font-family: var(--font-mono);
    font-size: 0.72rem;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    white-space: nowrap;
  }

  .dot {
    width: 6px;
    height: 6px;
    border-radius: 50%;
    background: var(--accent);
    animation: pulse 2.8s var(--ease) infinite;
  }

  .ghost {
    padding: 0.42rem 0.85rem;
    border: 1px solid var(--border-strong);
    border-radius: var(--radius-sm);
    color: var(--fg);
    font-size: var(--step--1);
    text-decoration: none;
    white-space: nowrap;
    transition: border-color 0.15s var(--ease), background-color 0.15s var(--ease);
  }

  .ghost:hover {
    background: var(--bg-raised);
    border-color: var(--accent);
  }

  @keyframes pulse {
    0%,
    100% {
      opacity: 1;
    }
    50% {
      opacity: 0.35;
    }
  }

  @media (min-width: 46rem) {
    .nav,
    .status {
      display: flex;
    }
  }
</style>
