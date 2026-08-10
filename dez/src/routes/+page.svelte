<!-- SPDX-License-Identifier: AGPL-3.0-only -->
<script lang="ts">
  import Hero from '$lib/components/Hero.svelte';
  import { DESCRIPTION, FEATURES, LINKS, PIPELINE, STATUS, TAGLINE } from '$lib/site';
</script>

<svelte:head>
  <title>{TAGLINE}</title>
  <meta name="description" content={DESCRIPTION} />
  <meta property="og:title" content={TAGLINE} />
  <meta property="og:description" content={DESCRIPTION} />
  <meta property="og:type" content="website" />
  <meta name="twitter:card" content="summary" />
</svelte:head>

<Hero />

<section class="section" id="what" aria-labelledby="what-h">
  <div class="container split">
    <div class="split-head">
      <p class="eyebrow">What it is</p>
      <h2 id="what-h">An editor that does its own thinking</h2>
    </div>
    <div class="split-body">
      <p>
        Dez is an IDE whose language model runs where your files already are. It is built on the
        <a href={LINKS.atlasRepo} rel="noopener noreferrer">Atlas Inference Engine</a> — a pure-Rust
        LLM inference stack — compiled to WebAssembly and dispatched to your GPU through WebGPU.
        Open the page, load a model, and the editor works with no account and no network.
      </p>
      <p>
        The reason to do this is not benchmark bragging. It is that “local-first” stops being a
        setting you can forget to turn on. There is no inference endpoint to leak to, no key to
        rotate, and no vendor between you and your own source code.
      </p>
      <p class="honest">
        <strong>{STATUS.label}.</strong>
        {STATUS.detail}
      </p>
    </div>
  </div>
</section>

<section class="section" id="features" aria-labelledby="features-h">
  <div class="container">
    <p class="eyebrow">Principles</p>
    <h2 id="features-h">Four commitments, not four features</h2>
    <ul class="grid">
      {#each FEATURES as feature (feature.id)}
        <li class="card">
          <h3>{feature.title}</h3>
          <p>{feature.body}</p>
        </li>
      {/each}
    </ul>
  </div>
</section>

<section class="section" id="how" aria-labelledby="how-h">
  <div class="container">
    <p class="eyebrow">How it works</p>
    <h2 id="how-h">Atlas, retargeted at the browser</h2>
    <p class="lede intro">
      The pipeline is deliberately short. Every stage below is a component that already exists in
      the open — the work of Dez is joining them into an editor.
    </p>

    <ol class="pipeline">
      {#each PIPELINE as stage (stage.step)}
        <li>
          <span class="step" aria-hidden="true">{stage.step}</span>
          <div>
            <h3>{stage.title}</h3>
            <p>{stage.body}</p>
          </div>
        </li>
      {/each}
    </ol>

    <p class="footnote">
      Atlas is open source under the AGPL-3.0 and ships hardware- and model-specific kernels behind
      swappable backend traits — the property that makes a WebGPU backend a port rather than a
      rewrite. Read the engine at
      <a href={LINKS.atlasRepo} rel="noopener noreferrer">github.com/Avarok-Cybersecurity/atlas</a>.
    </p>
  </div>
</section>

<section class="section cta-band" aria-labelledby="cta-h">
  <div class="container">
    <h2 id="cta-h">There is nothing to download yet</h2>
    <p class="lede">
      Dez is early. Rather than a fake install button, here is the honest one: watch the engine it
      is built on, and you will see Dez arrive as it is built.
    </p>
    <div class="cta-row">
      <a class="btn primary" href={LINKS.atlasRepo} rel="noopener noreferrer">
        Watch Atlas on GitHub
      </a>
      <a class="btn" href={LINKS.discord} rel="noopener noreferrer">Join the Discord</a>
    </div>
  </div>
</section>

<style>
  .split {
    display: grid;
    gap: 1.5rem;
    align-items: start;
  }

  .split-body {
    display: flex;
    flex-direction: column;
    gap: 1.1rem;
    max-width: var(--measure);
    color: var(--fg-muted);
  }

  .honest {
    padding: 1rem 1.1rem;
    border: 1px solid var(--border);
    border-left: 2px solid var(--accent);
    border-radius: 0 var(--radius-sm) var(--radius-sm) 0;
    background: var(--accent-wash);
    color: var(--fg);
    font-size: var(--step--1);
  }

  h2 {
    margin-block: 0.4rem 0;
  }

  .grid {
    display: grid;
    gap: 1rem;
    grid-template-columns: 1fr;
    padding: 0;
    margin-top: 2.5rem;
    list-style: none;
  }

  .card {
    padding: 1.4rem;
    border: 1px solid var(--border);
    border-radius: var(--radius);
    background: var(--bg-raised);
    transition: border-color 0.18s var(--ease);
  }

  .card:hover {
    border-color: var(--border-strong);
  }

  .card h3 {
    margin-bottom: 0.55rem;
  }

  .card p {
    color: var(--fg-muted);
    font-size: var(--step--1);
  }

  .intro {
    margin-top: 1.25rem;
  }

  .pipeline {
    display: grid;
    gap: 0;
    padding: 0;
    margin-top: 2.5rem;
    list-style: none;
    counter-reset: stage;
  }

  .pipeline li {
    display: grid;
    grid-template-columns: 3.25rem 1fr;
    gap: 1rem;
    padding-block: 1.4rem;
    border-top: 1px solid var(--border);
  }

  .pipeline li:last-child {
    border-bottom: 1px solid var(--border);
  }

  .step {
    font-family: var(--font-mono);
    font-size: var(--step--1);
    color: var(--accent);
    padding-top: 0.2rem;
  }

  .pipeline h3 {
    margin-bottom: 0.4rem;
  }

  .pipeline p {
    max-width: var(--measure);
    color: var(--fg-muted);
    font-size: var(--step--1);
  }

  .footnote {
    max-width: var(--measure);
    margin-top: 1.75rem;
    color: var(--fg-muted);
    font-size: var(--step--1);
  }

  .cta-band {
    background: var(--bg-sunken);
  }

  .cta-band .lede {
    margin-top: 1rem;
  }

  .cta-row {
    display: flex;
    flex-wrap: wrap;
    gap: 0.75rem;
    margin-top: 2rem;
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
    transition: border-color 0.15s var(--ease), transform 0.15s var(--ease);
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

  @media (min-width: 46rem) {
    .split {
      grid-template-columns: minmax(0, 20rem) minmax(0, 1fr);
      gap: 3rem;
    }

    .grid {
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 1.25rem;
    }

    .pipeline li {
      grid-template-columns: 5rem 1fr;
    }
  }
</style>
