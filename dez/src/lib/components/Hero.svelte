<!-- SPDX-License-Identifier: AGPL-3.0-only -->
<script lang="ts">
  import { LINKS, STATUS } from '$lib/site';
  import ComputeField from './ComputeField.svelte';
</script>

<section class="hero" id="top">
  <ComputeField />

  <div class="container inner">
    <p class="badge">
      <span class="dot" aria-hidden="true"></span>
      {STATUS.label} — no release yet
    </p>

    <!--
      The tagline verbatim, including the colon. Only the markup is split, for
      typographic emphasis — the text content and accessible name are exact.
    -->
    <h1>
      <span class="brandline">Dez<span class="colon">:</span></span>
      <span class="rest">
        The free and open-source IDE for the
        <span class="accent">local inference-first</span> paradigm
      </span>
    </h1>

    <p class="lede">
      An editor built on the Atlas Inference Engine, WebGPU and Rust compiled to WebAssembly. The
      model runs in your browser, on your GPU. No server round-trip. No API key. Your code and your
      weights never leave the machine.
    </p>

    <div class="cta">
      <a class="btn primary" href={LINKS.atlasRepo} rel="noopener noreferrer">
        Follow development on GitHub
      </a>
      <a class="btn" href="#what">What is Dez?</a>
    </div>

    <ul class="stack" aria-label="Built on">
      <li>Atlas Inference Engine</li>
      <li>Rust</li>
      <li>WebGPU</li>
      <li>100% WebAssembly</li>
    </ul>
  </div>
</section>

<style>
  .hero {
    position: relative;
    isolation: isolate;
    padding-block: clamp(3.5rem, 11vw, 8rem) clamp(3rem, 8vw, 6rem);
    overflow: hidden;
  }

  .inner {
    position: relative;
    z-index: 1;
  }

  .badge {
    display: inline-flex;
    align-items: center;
    gap: 0.5rem;
    padding: 0.3rem 0.8rem;
    margin-bottom: 1.6rem;
    border: 1px solid var(--border-strong);
    border-radius: 999px;
    background: var(--bg-raised);
    color: var(--fg-muted);
    font-family: var(--font-mono);
    font-size: 0.74rem;
    letter-spacing: 0.07em;
    text-transform: uppercase;
  }

  .dot {
    width: 6px;
    height: 6px;
    border-radius: 50%;
    background: var(--accent);
  }

  h1 {
    max-width: 20ch;
    margin-bottom: 1.5rem;
  }

  .brandline {
    display: block;
    color: var(--fg);
  }

  .colon {
    color: var(--fg-muted);
  }

  .rest {
    display: block;
    color: var(--fg-muted);
    font-weight: 500;
  }

  .accent {
    color: var(--accent);
  }

  .lede {
    margin-bottom: 2.25rem;
  }

  .cta {
    display: flex;
    flex-wrap: wrap;
    gap: 0.75rem;
    margin-bottom: 3rem;
  }

  .btn {
    display: inline-flex;
    align-items: center;
    padding: 0.7rem 1.2rem;
    border: 1px solid var(--border-strong);
    border-radius: var(--radius-sm);
    background: var(--bg-raised);
    color: var(--fg);
    font-size: var(--step--1);
    font-weight: 520;
    text-decoration: none;
    transition:
      background-color 0.15s var(--ease),
      border-color 0.15s var(--ease),
      transform 0.15s var(--ease);
  }

  .btn:hover {
    border-color: var(--accent);
    transform: translateY(-1px);
  }

  .btn.primary {
    background: var(--accent);
    border-color: var(--accent);
    color: var(--on-accent);
  }

  .btn.primary:hover {
    background: var(--accent-quiet);
    border-color: var(--accent-quiet);
  }

  .stack {
    display: flex;
    flex-wrap: wrap;
    gap: 0.5rem 1.5rem;
    padding: 0;
    margin: 0;
    list-style: none;
    color: var(--fg-muted);
    font-family: var(--font-mono);
    font-size: var(--step--1);
  }

  .stack li {
    position: relative;
    padding-left: 0.95rem;
  }

  .stack li::before {
    content: '';
    position: absolute;
    left: 0;
    top: 0.72em;
    width: 5px;
    height: 5px;
    border-radius: 1px;
    background: var(--accent);
    opacity: 0.75;
  }

  @media (min-width: 46rem) {
    h1 {
      max-width: 22ch;
    }
  }
</style>
