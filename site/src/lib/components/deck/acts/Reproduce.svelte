<script>
  // Act II — the walkthrough. Four commands and a chart. The commands are the
  // ones in bench/ladder38/RESULTS.md, not a simplified retelling of them:
  // a reproduction that needs a translation step is not a reproduction.
  import Slide from '../Slide.svelte';
  import Cmd from '../Cmd.svelte';
  import Kv from '../Kv.svelte';
  import ConcurrencyLadder from '../../ConcurrencyLadder.svelte';
  import { claim, fingerprint, parity } from '$lib/deck/content.js';

  const VLLM_DIGEST = 'sha256:0a51ea5b4ae2dc5d81890e5173f54203d2a3ae0cfffe51b8fd2afd4391bfd967';
</script>

<Slide
  act="cyan"
  eyebrow="Reference"
  title="The fingerprint"
  lede="Six lines that decide whether anything after them is comparable. If your box differs on
        any of them, you are measuring something else — fine, but say so."
>
  <Kv rows={fingerprint} />
</Slide>

<Slide
  act="cyan"
  eyebrow="Reference"
  title="Every axis pinned on both engines"
  lede="The commonest way to manufacture a speedup is to leave one of these unmatched. Ten axes,
        ten pins — driven by one script, not two."
>
  <div class="parity">
    <Kv rows={parity} mark cols={2} />
  </div>
</Slide>

<Slide
  act="cyan"
  eyebrow="Reference"
  title="We benchmark against vLLM at its best, not its defaults"
  lede="vLLM 0.27.1 registers Qwen3_5MTP and this checkpoint ships mtp.* weights, so vLLM can
        speculate here. Running it without would have been the easy 2×, and a fabricated one."
  steps={2}
>
  <div class="two">
    <div class="at" style="--n: 1">
      <p class="lead">
        The earlier reference in this campaign ran speculative decoding <em>off</em>. It understated
        vLLM badly, so it was replaced and the old column kept in view rather than deleted.
      </p>
      <p class="lead">
        The published table therefore carries two baselines: the matched one we claim against, and
        the unmatched one, labelled as such. At C=128 the unmatched configuration is actually
        <em>faster</em> than the matched one — vLLM's speculation costs it throughput at high
        concurrency — so we claim against whichever is stronger at each rung.
      </p>
    </div>
    <aside class="quote at" style="--n: 2">
      <p>
        “Inadequate competitor tuning is scientific misconduct.”
      </p>
      <footer class="mono">Heiser, <em>Systems Benchmarking Crimes</em></footer>
    </aside>
  </div>
</Slide>

<Slide
  act="cyan"
  eyebrow="Step 1"
  title="Prove the box before you trust a number"
  lede="Four checks, in this order. Every one of them has been the reason a run was thrown away in
        this campaign, so none of them is ceremony."
  steps={2}
>
  <div class="wide2">
    <div class="at" style="--n: 1">
      <Cmd
        label="preflight"
        lines={[
          `nvidia-smi                       # GB10, driver 580+, 119.7 GB free`,
          `nvcc --version                   # must report CUDA 13.0`,
          `docker run --rm --gpus all \\`,
          `  nvidia/cuda:13.0.0-base-ubuntu24.04 nvidia-smi`,
          `df -h ~/.cache/huggingface       # weights land here, tens of GB`
        ]}
        note="The third line is the one people skip: it proves the NVIDIA Container Toolkit is wired up, not just installed. Without it the image builds and then serves nothing."
      />
    </div>
    <aside class="side at" style="--n: 2">
      <p class="side-h mono">What a shared box costs you</p>
      <p>
        The gate refuses to self-start below 85% free host memory, and the hardware precheck
        tolerates at most one foreign compute process. Measure on an idle box or the run will be
        declined — which is the correct behaviour, and a surprise the first time.
      </p>
    </aside>
  </div>
</Slide>

<Slide
  act="cyan"
  eyebrow="Step 2"
  title="Build both artefacts"
  lede="The container serves models; the binary measures them. You need both, and the binary has to
        come from the same tree as the commit you are testing."
  steps={2}
>
  <div class="wide2">
    <div class="at" style="--n: 1">
      <Cmd
        label="clone, image, binary"
        lines={[
          `git clone https://github.com/Avarok-Cybersecurity/atlas.git`,
          `cd atlas && git checkout ${claim.build}`,
          ``,
          `docker build -f docker/gb10/Dockerfile -t atlas-gb10 .`,
          ``,
          `sudo apt-get install -y build-essential pkg-config \\`,
          `  cmake clang libclang-dev`,
          `cargo build --release -p spark-server --bin spark`
        ]}
        note="Both builds run from the repository root. The multi-target image compiles PTX for every supported model; the first cargo build takes 15–30 minutes for the same reason and leaves 3–5 GB under target/."
      />
    </div>
    <div class="at" style="--n: 2">
      <Cmd
        label="verify before going further"
        lines={[
          `./target/release/spark --version`,
          `./target/release/spark benchmark list`,
          `./target/release/spark benchmark list concurrency-sweep`
        ]}
        note="The last line prints every parameter of the sweep with its default — the schema the next steps override. If it prints, the toolchain is sound and the rest of this deck will run."
      />
      <p class="after">
        The gate's self-start also reads a cached recipe index at
        <code class="mono">~/.atlas/atlas-recipes/index.json</code>. Open the TUI library once to
        populate it, or Step 6 stops with exactly that message.
      </p>
    </div>
  </div>
</Slide>

<Slide act="cyan" eyebrow="Step 3" title="Bring up the baseline leg" lede="Pinned by digest, not by tag — “latest” is not a version.">
  <Cmd
    label="vLLM 0.27.1 + MTP, fp8 KV"
    lines={[
      `docker run --rm --gpus all --network host \\`,
      `  vllm/vllm-openai@${VLLM_DIGEST} \\`,
      `  --model ${claim.checkpoint} \\`,
      `  --max-model-len 2048 --max-num-seqs 128 \\`,
      `  --gpu-memory-utilization 0.85 \\`,
      `  --kv-cache-dtype fp8 --enable-prefix-caching \\`,
      `  --speculative-config '{"method":"mtp","num_speculative_tokens":3}'`
    ]}
    note="num_speculative_tokens 3 is K=4 — the same draft width Atlas runs. Context 2048 and batch cap 128 are the pinned pair; changing either invalidates the comparison in both directions."
  />
</Slide>

<Slide act="cyan" eyebrow="Step 4" title="Bring up the subject leg" lede="Same box, same checkpoint, same client. Back to back, not from memory.">
  <Cmd
    label="Atlas — round-11 flags"
    lines={[
      `ATLAS_PREFILL_CODISPATCH=1 ATLAS_FP8_ROWWISE=1 \\`,
      `ATLAS_MTP_DCUT_RATIO=1.0 ATLAS_MTP_K_LADDER=1:3,2:1,4:2,8:2,16:1 \\`,
      `spark serve ${claim.checkpoint} \\`,
      `  --host 0.0.0.0 --port 8888 --max-seq-len 2048 --max-batch-size 128 \\`,
      `  --gpu-memory-utilization 0.85 --kv-cache-dtype fp8 \\`,
      `  --enable-prefix-caching true --speculative --num-drafts 3 \\`,
      `  --mtp-quantization bf16 --disable-thinking --no-tui`
    ]}
    note="The full flag list, including the SSM cache and scheduling knobs, is in ladder.generated.json under series[atlas].cli — the site renders it from the same record the harness wrote."
  />
</Slide>

<Slide
  act="cyan"
  eyebrow="Step 5"
  title="One instrument, pointed at both engines"
  lede="The subcommand drives an endpoint that is already serving — it neither loads a model nor
        touches the GPU. So the binary that gates our own pull requests is the binary that measures
        vLLM, on the same parameters and the same prompt fixture."
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
        campaign driver with <code class="mono">--reps</code> and a per-rep series. The gate's sweep
        runs one measured batch per cell at pinned parameters.
      </p>
      <p>
        They share a byte-identical prompt corpus and a bit-compatible percentile function on
        purpose, so the numbers are comparable — but only the gate's instrument produces a record
        CI will accept.
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
  .parity {
    max-width: 92%;
  }
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
  .quote {
    border-left: 3px solid var(--sx);
    padding: 0.4em 0 0.4em 1.1em;
  }
  .quote p {
    font-size: 1.15em;
    line-height: 1.5;
    color: var(--t1);
  }
  .quote footer {
    margin-top: 0.7em;
    font-size: 0.75em;
    color: var(--t3);
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
