<script>
  // Act II, second half — measurement. Two instruments appear here and the
  // distinction between them is the point of the act:
  //
  //   * `spark benchmark run concurrency-sweep` is what CI runs. It is the only
  //     one that can mint a gate record, and it measures the GATE's workload.
  //   * `bench/ladder38/harness_w55_conc_ladder.py` is what produced the chart
  //     on this page. Nothing else reproduces the published ladder, so the deck
  //     has to hand it over or the walkthrough stops short of its own headline.
  //
  // Setup (fingerprint, parity, Steps 1-4) is Reproduce.svelte, which precedes
  // this act in the route.
  import Slide from '../Slide.svelte';
  import Cmd from '../Cmd.svelte';
  import ConcurrencyLadder from '../../ConcurrencyLadder.svelte';
  import { claim } from '$lib/deck/content.js';
</script>

<Slide
  act="cyan"
  eyebrow="Step 5"
  title="The gate's instrument, pointed at both engines"
  lede="The subcommand drives an endpoint that is already serving — it neither loads a model nor
        touches the GPU. So the binary that gates our own pull requests is the binary that measures
        vLLM. This measures the GATE's workload, not the chart's; Step 5b is the chart."
  steps={2}
>
  <div class="wide2">
    <div class="at" style="--n: 1">
    <Cmd
      label="the baseline leg, then the subject leg"
      lines={[
        `spark benchmark run concurrency-sweep \\`,
        `  --url http://127.0.0.1:8000 --model ${claim.checkpoint} \\`,
        `  --param concurrencies=1,4,8,16 --param isls=512 --param osl=320 \\`,
        `  --skip-coherence-probe --format json > vllm.json`,
        `spark benchmark run concurrency-sweep \\`,
        `  --url http://127.0.0.1:8888 --model ${claim.checkpoint} \\`,
        `  --param concurrencies=1,4,8,16 --param isls=512 --param osl=320 \\`,
        `  --format json > atlas.json`
      ]}
      note="http:// only. stdout carries the record and stderr the progress, so the redirect gives a clean file. Those --param values are the ones the gate pins; an unknown key is an error, never a silent no-op."
    />
    </div>
    <aside class="side at" style="--n: 2">
      <p class="side-h mono">Two instruments, not interchangeable</p>
      <p>
        The published C=1…128 ladder came from <code class="mono">{claim.harnessFile}</code>, a
        campaign driver with <code class="mono">--reps</code>; the gate's sweep runs one measured
        batch per cell at pinned parameters.
      </p>
      <p>
        Their prompt corpus is byte-identical, which makes a cell's throughput comparable across
        the two. Their percentile rules are <em>not</em> — the driver interpolates, the gate takes a
        nearest rank — so compare the aggregate tok/s the ladder publishes, never one instrument's
        p50 TTFT against the other's. Only the gate's produces a record CI accepts.
      </p>
    </aside>
  </div>
</Slide>

<Slide
  act="cyan"
  eyebrow="Step 5b"
  title="The command that reproduces the chart"
  lede="Step 5 measures what CI gates. This measures what this page publishes — the same driver, at
        the same shape, that produced every number in the ladder overleaf."
  steps={2}
>
  <div class="wide2">
    <div class="at" style="--n: 1">
      <Cmd
        label="the published ladder, Atlas leg"
        lines={[
          `python3 -m venv .venv && .venv/bin/pip install aiohttp`,
          ``,
          `.venv/bin/python ${claim.harnessFile} \\`,
          `  --url http://127.0.0.1:8888 --model ${claim.checkpoint} \\`,
          `  --label atlas --out atlas_ladder.json \\`,
          `  --concs ${claim.concsArg} \\`,
          `  --reps ${claim.reps} --isl ${claim.isl} --osl ${claim.osl} --warmup ${claim.warmup}`
        ]}
        note={`Every knob is a required argument — the driver defaults nothing silently, so the command IS the methodology. Swap --url and --label for the vLLM leg and run them back to back. Budget about an hour per leg on a GB10: C=128 alone streams ${128 * claim.osl} tokens per rep, and there are ${claim.reps} timed reps plus ${claim.warmup} warmup at every rung.`}
      />
    </div>
    <aside class="side at" style="--n: 2">
      <p class="side-h mono">Check the driver hash first</p>
      <p>
        The driver prints its own <code class="mono">sha256</code> on the first line and writes it
        into the output as <code class="mono">driver_sha256</code>. The published Atlas legs carry
        <code class="mono">{claim.harnessShaAtlas}</code>; the copy in the repository today hashes
        <code class="mono">{claim.harnessShaRepo}</code>.
      </p>
      <p>
        Same sampling behaviour — both send <code class="mono">presence_penalty</code> and
        <code class="mono">frequency_penalty</code> at 0.0, which is what the recorded pair differed
        over — but they are not the same bytes, so record the hash you actually ran rather than
        quoting ours.
      </p>
    </aside>
  </div>
</Slide>

<Slide
  act="cyan"
  eyebrow="Step 6"
  title="The command that gates every pull request"
  lede="The same subcommand in its other mode. This one starts a server: it serves the benchmark's
        own recipe on a free port, waits up to 900 s for a cold NVFP4 load, and tears it down."
  steps={2}
>
  <div class="at" style="--n: 1">
    <Cmd
      label="scripts/queue-perf-pr.sh — the campaign it prints"
      lines={[
        `for g in ttft-cold-gate ttft-warm-gate vision-fidelity \\`,
        `         ssm-state-poisoning-gate decode-floor concurrency-sweep \\`,
        `         video-fidelity bfcl-subset bfcl-subset-echolp agentic-webserver; do`,
        `  timeout 21600 ./target/release/spark benchmark run "$g" \\`,
        `      --pull-request-gate --yes`,
        `done`,
        ``,
        `spark benchmark --pull-request-gate-check    # what CI then runs`
      ]}
      note="One gate per process, which is why it is a loop. Each run writes .benchmarks/&lt;id&gt;/&lt;date&gt;-&lt;sha&gt;.json carrying the metrics, the verdict, the hardware fingerprint, the exact command and the commit sha."
    />
  </div>
  <p class="after at" style="--n: 2">
    <code class="mono">--url</code> and <code class="mono">--pull-request-gate</code> are mutually
    exclusive by design: a run pointed at someone else's server can be measured and argued about,
    but it can never become a record. The CLI draws the line between an experiment and evidence,
    so a reviewer does not have to.
  </p>
</Slide>

<Slide
  act="cyan"
  eyebrow="Result"
  title="The ladder"
  lede="{claim.won} of {claim.rungs} rungs, {claim.min} to {claim.max}. Log2 X, because the rungs double and linear spacing would crush the band where single-stream latency lives."
  wide
>
  <div class="chart">
    <ConcurrencyLadder embedded compact />
  </div>
</Slide>

<Slide
  act="cyan"
  eyebrow="Batch sizing"
  title="The batch cap is part of the comparison, not a free knob"
  lede="The trap that caught the person who wrote the note warning about it — recorded as a
        correction rather than quietly fixed."
  steps={3}
>
  <div class="two">
    <div class="at" style="--n: 1">
      <table class="tb">
        <thead>
          <tr><th>C=32, same box, same day</th><th>cap 32</th><th>cap 128</th><th>effect</th></tr>
        </thead>
        <tbody>
          <tr><td>Atlas</td><td class="mono">277.31</td><td class="mono">278.93</td><td>flat</td></tr>
          <tr><td>vLLM + MTP</td><td class="mono">284.54</td><td class="mono">277.12</td><td class="up">+2.7% at cap 32</td></tr>
        </tbody>
      </table>
      <p class="tb-note">
        Lowering the cap to match the rung looks neutral. It is not: vLLM sizes its KV blocks and
        scheduler budget from <code class="mono">max_num_seqs</code>, so a smaller cap changes
        allocation, preemption and prefix reuse — and materially favours it.
      </p>
    </div>
    <div class="at" style="--n: 2">
      <p class="lead">
        A matched cap-32 pair, adopted in good faith to dodge a hardware hazard, produced
        <strong>0.975×</strong> and inverted the true ordering. Measured again at the certified
        cap-128 configuration on the same box the same day: <strong>1.007×</strong>, with
        non-overlapping distributions — Atlas's worst rep beat vLLM's best.
      </p>
      <p class="lead">
        The certified table pins cap 128 on both engines at <em>every</em> rung, independently of
        the concurrency being driven. That pin is load-bearing.
      </p>
    </div>
  </div>
  <p class="pull at" style="--n: 3">
    If you re-cap to survive a hazard on your own box: say so beside the number, and re-pin both
    engines to the same value. Do not fold it into the certified column.
  </p>
</Slide>

<Slide
  act="cyan"
  eyebrow="Mechanism"
  title="Where the margin actually comes from"
  lede="Ask this before the numbers. A speedup with no stated mechanism and no known ceiling is a
        configuration artefact waiting to be found."
  steps={3}
>
  <div class="mech">
    <article class="at" style="--n: 1">
      <h3>MTP speculation, width-laddered</h3>
      <p>
        K is chosen per concurrency (<code class="mono">1:3,2:1,4:2,8:2,16:1</code>) rather than
        fixed, and self-disables above 32 concurrent sequences where verify cost exceeds the win.
      </p>
    </article>
    <article class="at" style="--n: 2">
      <h3>Prefill co-dispatch + fp8 row-wise</h3>
      <p>
        Prefill overlaps decode rather than stalling it; row-wise fp8 moves fewer bytes on the
        bandwidth-bound path. Decode at this scale is a memory-traffic problem, so this is where a
        real win has to come from.
      </p>
    </article>
    <article class="at" style="--n: 3">
      <h3>Why C=128 is the widest rung</h3>
      <p>
        It is not that Atlas gets faster — it is that vLLM's C=128 falls <em>below</em> its own
        C=64 when speculation stays on at high concurrency. Atlas's ladder has already switched it
        off. The margin is a scheduling decision, and it is reproducible for that reason.
      </p>
    </article>
  </div>
</Slide>

<style>
  .wide2 {
    display: grid;
    grid-template-columns: 1.55fr 1fr;
    gap: 2em;
    align-items: start;
  }
  .side {
    border: 1px solid var(--border-strong);
    border-top: 2px solid var(--sx);
    background: var(--card);
    border-radius: 6px;
    padding: 1em 1.1em;
  }
  .side-h {
    font-size: 0.74em;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: var(--sx);
    margin-bottom: 0.6em;
  }
  .side p {
    color: var(--t2);
    line-height: 1.6;
    font-size: 0.88em;
    margin-bottom: 0.7em;
  }
  .side p:last-child {
    margin-bottom: 0;
  }
  .two {
    display: grid;
    grid-template-columns: 1.4fr 1fr;
    gap: 2.4em;
    align-items: start;
  }
  .lead {
    color: var(--t2);
    line-height: 1.65;
    margin-bottom: 0.9em;
    max-width: 60ch;
  }
  .lead strong {
    color: var(--t1);
  }
  .after {
    margin-top: 1em;
    color: var(--t3);
    font-size: 0.85em;
    max-width: 74ch;
  }

  /* The ladder component stacks chart over table, which is a page layout. On a
     slide they sit side by side and the table carries the exact figures the
     chart only implies. */
  .chart :global(.cl-embed-inner) {
    display: grid;
    grid-template-columns: 1.5fr 1fr;
    gap: 1.8em;
    align-items: center;
    max-width: none;
  }
  .chart :global(.cl-panel) {
    margin: 0;
  }
  .chart :global(.cl-table),
  .chart :global(.cl-caption) {
    font-size: 0.62em;
  }
  .chart :global(.cl-tablewrap) {
    margin: 0;
  }

  .tb {
    border-collapse: collapse;
    width: 100%;
    font-size: 0.9em;
  }
  .tb th,
  .tb td {
    text-align: right;
    padding: 0.45em 0.7em;
    border-bottom: 1px solid var(--border);
  }
  .tb th:first-child,
  .tb td:first-child {
    text-align: left;
  }
  .tb th {
    font-size: 0.8em;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: var(--t3);
    font-weight: 600;
  }
  .up {
    color: var(--amber);
  }
  .tb-note {
    margin-top: 0.9em;
    color: var(--t3);
    font-size: 0.85em;
    line-height: 1.6;
  }

  .pull {
    margin-top: 1em;
    padding: 0.8em 1.1em;
    border: 1px dashed var(--border-strong);
    border-radius: 6px;
    color: var(--t2);
    font-size: 0.92em;
    max-width: 90ch;
  }

  .mech {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 1.4em;
  }
  .mech article {
    border-top: 2px solid var(--sx);
    padding-top: 0.9em;
  }
  .mech h3 {
    font-size: 1.02em;
    font-weight: 700;
    margin-bottom: 0.5em;
  }
  .mech p {
    color: var(--t2);
    line-height: 1.6;
    font-size: 0.92em;
  }
  .mech code {
    color: var(--accent-deep);
    font-size: 0.88em;
  }
</style>
