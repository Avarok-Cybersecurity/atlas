<script>
  // Act I — what is being claimed, and what is not. The order is deliberate:
  // the limits slide comes before any evidence, because a claim whose edges are
  // stated first is read differently from one whose edges have to be dug out.
  import Slide from '../Slide.svelte';
  import Chevrons from '../Chevrons.svelte';
  import Kv from '../Kv.svelte';
  import { claim, fragile, stamp } from '$lib/deck/content.js';
</script>

<Slide act="violet" wide>
  <div class="cover">
    <div class="cover-mark"><Chevrons id="dk-cover" /></div>
    <div>
      <p class="cover-kicker mono">Verification steps</p>
      <h1 class="cover-title">Reproduce the ladder<br />before you believe it.</h1>
      <p class="cover-sub">
        Everything the front page claims about concurrency, batch sizing and vLLM, restated as
        commands you can run on your own box. {claim.rungs} rungs, {claim.min} to {claim.max}.
      </p>
      <p class="cover-stamp mono">{stamp}</p>
    </div>
  </div>
</Slide>

<Slide
  act="violet"
  eyebrow="How to read this"
  title="Three questions, in order"
  lede="Diligence on an inference engine is not a code review. It answers three things, and the
        third is the one that decides the round."
  steps={3}
>
  <ol class="q">
    <li class="at" style="--n: 1">
      <span class="q-n mono">01</span>
      <strong>Is the claim true?</strong>
      <span>Not "is the chart real" — can a stranger produce the same numbers on their own
      hardware, from the artifacts, without talking to us. Act II is that walkthrough.</span>
    </li>
    <li class="at" style="--n: 2">
      <span class="q-n mono">02</span>
      <strong>Is it durable?</strong>
      <span>A configuration gap closes in six weeks when upstream ships. A mechanism does not.
      Act III separates the two and shows what defends the number between releases.</span>
    </li>
    <li class="at" style="--n: 3">
      <span class="q-n mono">03</span>
      <strong>What does it cost to keep true?</strong>
      <span>Gate machinery, licence posture, contributor provenance, and the bus factor on the
      parts that produce the win.</span>
    </li>
  </ol>
</Slide>

<Slide
  act="violet"
  eyebrow="The claim"
  title="Stated so it can be falsified"
  lede="A performance claim without every axis named is not yet a claim. This is the whole of ours."
>
  <blockquote class="claim">
    <p>
      <strong>{claim.engine}</strong> (build <code class="mono">{claim.build}</code>) sustains higher
      mean decode throughput than <strong>{claim.baseline}</strong> on
      <code class="mono">{claim.checkpoint}</code>, served on one {claim.box}, at
      <strong>every</strong> concurrency C = {claim.concurrencies} — ISL {claim.isl}, OSL
      {claim.osl}, temperature {claim.temperature}, seed {claim.seed}, {claim.aggregate}.
      Margins run {claim.min} to {claim.max}.
    </p>
  </blockquote>
  <p class="claim-note">
    Every noun in that sentence is a knob someone could have turned to flatter us. The next act
    hands you each of them.
  </p>
</Slide>

<Slide
  act="violet"
  eyebrow="Scope"
  title="What we are not claiming"
  lede="The fastest way to waste your week is to test something we never said."
  steps={4}
>
  <div class="cols">
    <Kv
      rows={[
        ['measured on', 'one GB10 box', 'DGX Spark. Every figure here was taken there, not extrapolated from it.'],
        ['not claimed', 'other model classes', 'One dense 27B hybrid at NVFP4. MoE and long-context behave differently.'],
        ['not claimed', 'multi-node', 'Single box. No TP/PP/EP story is being told here.']
      ]}
    />
    <aside class="warn at" style="--n: 4">
      <p class="warn-h mono">Fragile rungs, named</p>
      <p>
        {fragile.count} rungs are won by margins inside plausible run-to-run drift — {fragile.rungs}
        sit between {fragile.min} and {fragile.max}. We flag them rather than rounding them into the
        headline, and they are the rungs we re-measure first when anything changes.
      </p>
    </aside>
  </div>
</Slide>

<Slide
  act="violet"
  eyebrow="Why the desktop is the right instrument"
  title="The Spark is the on-ramp, not the destination"
  lede="Measuring on a DGX Spark is not a smaller claim than measuring in a datacenter. It is the
        rung NVIDIA built for exactly this, and the path off it is theirs, not our extrapolation."
  steps={2}
>
  <div class="cols">
    <div class="at" style="--n: 1">
      <blockquote class="jhq">
        <p>
          “AI has transformed every layer of the computing stack. It stands to reason a new class of
          computers would emerge — designed for AI-native developers and to run AI-native
          applications. With these new DGX personal AI computers, AI can span from cloud services to
          desktop and edge applications.”
        </p>
        <footer class="mono">Jensen Huang, NVIDIA — DGX Spark announcement, 18 March 2025</footer>
      </blockquote>
      <p class="jhn">
        NVIDIA's own framing in the same release: the full-stack platform lets DGX Spark users
        “seamlessly move their models from their desktops to DGX Cloud or any accelerated cloud or
        data center infrastructure — with virtually no code changes.”
      </p>
    </div>
    <aside class="warn at" style="--n: 2">
      <p class="warn-h mono">Where Atlas sits on that path</p>
      <p>
        Same Blackwell architecture, same CUDA stack, same OpenAI-compatible surface — a workload
        validated on GB10 moves up the line rather than starting over. Atlas ships its GB10 kernel
        target today, and the kernel system is already three-dimensional (hardware × model ×
        quant): another hardware arm is a target to add, not an engine to rewrite.
      </p>
    </aside>
  </div>
</Slide>

<style>
  .cover {
    display: grid;
    grid-template-columns: calc(12 * var(--u)) 1fr;
    gap: calc(3.4 * var(--u));
    align-items: center;
    height: 100%;
    padding-bottom: calc(4 * var(--u));
  }
  .cover-mark {
    width: 100%;
  }
  .cover-kicker {
    font-size: 0.85em;
    letter-spacing: 0.22em;
    text-transform: uppercase;
    color: var(--sx);
    margin-bottom: 0.9em;
  }
  .cover-title {
    font-size: 3.5em;
    font-weight: 800;
    letter-spacing: -0.035em;
    line-height: 1.02;
  }
  .cover-sub {
    margin-top: 1em;
    font-size: 1.08em;
    color: var(--t2);
    max-width: 52ch;
    line-height: 1.6;
  }
  .cover-stamp {
    margin-top: 2.2em;
    font-size: 0.72em;
    color: var(--t3);
    letter-spacing: 0.04em;
  }

  .q {
    list-style: none;
    display: grid;
    gap: 1.1em;
    max-width: 74ch;
  }
  .q li {
    display: grid;
    grid-template-columns: 3.2em 1fr;
    gap: 0 1em;
    padding-bottom: 1em;
    border-bottom: 1px solid var(--border);
  }
  .q-n {
    grid-row: span 2;
    font-size: 1.5em;
    font-weight: 700;
    color: var(--sx);
    opacity: 0.75;
  }
  .q strong {
    font-size: 1.15em;
    font-weight: 700;
  }
  .q span:last-child {
    color: var(--t2);
    line-height: 1.6;
  }

  .claim {
    border-left: 3px solid var(--sx);
    background: var(--card);
    padding: 1.2em 1.5em;
    border-radius: 0 8px 8px 0;
    max-width: 82ch;
  }
  .claim p {
    font-size: 1.12em;
    line-height: 1.65;
  }
  .claim code {
    font-size: 0.9em;
    color: var(--accent-deep);
  }
  .claim-note {
    margin-top: 1.2em;
    color: var(--t3);
    font-size: 0.92em;
  }

  .cols {
    display: grid;
    grid-template-columns: 1.35fr 1fr;
    gap: 2.4em;
    align-items: start;
  }
  .warn {
    border: 1px solid var(--border-strong);
    border-top: 2px solid var(--amber);
    background: var(--card);
    border-radius: 6px;
    padding: 1em 1.1em;
  }
  .jhq {
    border-left: 3px solid var(--sx);
    padding: 0.2em 0 0.2em 1.1em;
    margin-bottom: 1em;
  }
  .jhq p {
    font-size: 0.98em;
    line-height: 1.55;
    color: var(--t1);
  }
  .jhq footer {
    margin-top: 0.7em;
    font-size: 0.74em;
    color: var(--t3);
  }
  .jhn {
    color: var(--t2);
    font-size: 0.92em;
    line-height: 1.6;
    max-width: 62ch;
  }

  .warn-h {
    font-size: 0.74em;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: var(--amber);
    margin-bottom: 0.6em;
  }
  .warn p {
    color: var(--t2);
    line-height: 1.6;
    font-size: 0.92em;
  }
</style>
