<script>
  // Act II, first half — the reference frame and the setup: fingerprint,
  // parity, and the four steps that get an outsider to two serving engines.
  // Measuring them is Ladder.svelte, which follows this act in the route.
  // The commands are the ones in bench/ladder38/RESULTS.md, not a simplified
  // retelling: a reproduction that needs a translation step is not one.
  import Slide from '../Slide.svelte';
  import Cmd from '../Cmd.svelte';
  import Kv from '../Kv.svelte';
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
  lede="Five checks, in this order. Every one of them has been the reason a run was thrown away in
        this campaign, so none of them is ceremony."
  steps={2}
>
  <div class="wide2">
    <div class="at" style="--n: 1">
      <Cmd
        label="preflight"
        lines={[
          `nvidia-smi                       # GB10, driver 580+`,
          `free -g                          # ~121 GB unified, not nvidia-smi`,
          ``,
          `export PATH=/usr/local/cuda/bin:$PATH`,
          `nvcc --version                   # must report CUDA 13.0`,
          ``,
          `docker run --rm --gpus all \\`,
          `  nvidia/cuda:13.0.0-base-ubuntu24.04 nvidia-smi`,
          `df -h ~/.cache/huggingface       # weights land here, tens of GB`
        ]}
        note="Two GB10 particulars, both of which have cost this campaign time. nvidia-smi reports memory as `Not Supported` — the 121 GB is a unified LPDDR5X pool, so `free` is the instrument. And CUDA ships outside PATH: without that export, `nvcc --version` says command-not-found and the cargo build in Step 2 dies in cudarc's build script rather than anywhere informative. The docker line is the one people skip: it proves the NVIDIA Container Toolkit is wired up, not just installed."
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
          `cd atlas && git checkout ${claim.buildPublic}`,
          ``,
          `docker build -f docker/gb10/Dockerfile -t atlas-gb10 .`,
          ``,
          `sudo apt-get install -y build-essential pkg-config \\`,
          `  cmake clang libclang-dev`,
          `cargo build --release -p spark-server --bin spark`
        ]}
        note={`Both builds run from the repository root, with CUDA still on PATH from Step 1. The multi-target image compiles PTX for every supported model; the first cargo build takes 15–30 minutes for the same reason and leaves 3–5 GB under target/. ${claim.buildPublic} is the certified sha rather than ${claim.build}, the tree the numbers were measured on: that one was a local merge and was never pushed, so it does not exist in your clone. The two differ only in doc comments and gate machinery — no executable change.`}
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
</style>
