<script>
  // Act III — everything that decides whether the number survives contact with
  // time: the self-audit, the artifact kit, the notebook, the gates, and the
  // licence and provenance posture behind them.
  import Slide from '../Slide.svelte';
  import Audit from '../Audit.svelte';
  import Kv from '../Kv.svelte';
  import Cmd from '../Cmd.svelte';
  import { audit, claim, gateFacts, links } from '$lib/deck/content.js';
</script>

<Slide
  act="green"
  eyebrow="Self-audit"
  title="Ten ways to fake this, answered"
  lede="Heiser's benchmarking-crimes taxonomy, run against our own campaign. Three rows stay open,
        because a checklist with nothing open is marketing."
  steps={3}
  wide
>
  <Audit rows={audit} />
</Slide>

<Slide
  act="green"
  eyebrow="Artifacts"
  title="The reproduction kit"
  lede="A reproduction that needs a conversation with us is not one. Everything below is already
        in the repository, at the commit the numbers were taken on."
>
  <div class="grid2">
    <Kv
      rows={[
        ['code', `Atlas ${claim.build}`, 'plus the certified SHA and the merge commit, all three named'],
        ['baseline', 'container digest', 'sha256, not a tag — the same digest across every leg'],
        ['harness', 'sha256 in every output', 'the script hashes its own source into the JSON it writes'],
        ['weights', claim.checkpoint, 'HF repo, pinned revision'],
        ['seeds', `seed ${claim.seed}, temp ${claim.temperature}`, 'constants in the harness, not flags'],
        ['raw data', 'per-rung JSON', 'every rep, not the aggregate — in bench/ladder38/'],
        ['record', claim.resultsDoc, 'the lab notebook, including what failed']
      ]}
    />
    <aside class="ask">
      <p class="ask-h mono">What we would ask of you</p>
      <p>
        Rent your own GB10 hour. Run the two legs back to back. If a rung disagrees with our table
        by more than its published spread, that is a finding and we want it — the fragile rungs
        (C=8, C=16, C=32) are where to spend the budget first.
      </p>
    </aside>
  </div>
</Slide>

<Slide
  act="green"
  eyebrow="Method"
  title="The notebook is the evidence, not the chart"
  lede="RESULTS.md runs past 1,500 lines and reads as a lab notebook: dated rounds, hypotheses,
        and the ones that died."
  steps={4}
>
  <ul class="notes">
    <li class="at" style="--n: 1">
      <span class="nt mono">retraction</span>
      <span>An “agentic wall regression” was withdrawn once it turned out to be a cross-box
      comparison. The withdrawal is in the file, above the claim it replaced.</span>
    </li>
    <li class="at" style="--n: 2">
      <span class="nt mono">negative</span>
      <span>Four hypotheses closed as negative results — D-Cut pruning net-negative at the
      contested rungs, the K-ladder A/B, a fixed-cost audit, and <code class="mono">decode_tps</code>
      rejected as a gate.</span>
    </li>
    <li class="at" style="--n: 3">
      <span class="nt mono">discarded</span>
      <span>Two completed runs thrown away for being measured on the wrong box, and a bf16-KV
      attempt discarded for not matching the reference — rather than kept as the better number.</span>
    </li>
    <li class="at" style="--n: 4">
      <span class="nt mono">excluded</span>
      <span>Driver version excluded as an explanation by measuring three boxes across two
      drivers. Thermals excluded by re-baselining after a physical move.</span>
    </li>
  </ul>
</Slide>

<Slide
  act="green"
  eyebrow="Durability"
  title="What defends the number between releases"
  lede="A margin that is only ever measured by hand decays silently. These are the mechanisms that
        make a regression loud."
  steps={3}
>
  <div class="grid2">
    <div class="at" style="--n: 1">
      <p class="lead">
        Ten gates are required for a pull request to land, and thresholds are not a percentage
        band: they are absolute per-metric floors committed in
        <code class="mono">kernels/gb10/&lt;model&gt;/BENCH.toml</code>, each with an explicit
        <code class="mono">noise</code> slack. CI refuses a slack above 5% of its own bound — larger
        than that is a threshold change wearing a measurement-noise costume.
      </p>
      <p class="lead">
        A record is voided by <em>content</em>, not ancestry. Eight paths invalidate one:
        <code class="mono">crates/</code>, <code class="mono">kernels/</code>,
        <code class="mono">Cargo.toml</code>, <code class="mono">Cargo.lock</code>,
        <code class="mono">vendor/</code>, <code class="mono">3rdparty_patches/</code>,
        <code class="mono">rust-toolchain.toml</code> — and <code class="mono">jinja-templates/</code>,
        which is runtime input rather than build input: the server loads one over the checkpoint's own
        chat template, so editing it changes the bytes every prompt renders to. A dirty tree fails, and
        an entry declaring no thresholds fails rather than passes.
      </p>
    </div>
    <div class="at" style="--n: 2">
      <table class="tb2">
        <thead>
          <tr><th>concurrency-sweep floor</th><th>min tok/s</th><th>noise</th></tr>
        </thead>
        <tbody>
          <tr><td>c1_aggregate_tok_s</td><td class="mono">17.0</td><td class="mono">0.8</td></tr>
          <tr><td>c4_aggregate_tok_s</td><td class="mono">35.0</td><td class="mono">1.5</td></tr>
          <tr><td>c8_aggregate_tok_s</td><td class="mono">52.0</td><td class="mono">1.5</td></tr>
          <tr><td>c16_aggregate_tok_s</td><td class="mono">73.5</td><td class="mono">1.5</td></tr>
          <tr><td>peak_aggregate_tok_s</td><td class="mono">73.5</td><td class="mono">1.5</td></tr>
          <tr><td>vacuous_cells</td><td class="mono">max 0</td><td class="mono">—</td></tr>
        </tbody>
      </table>
      <p class="note2">
        Calibrated at mean minus max(3σ, ~5%) from three fresh reps on the same instrument, never
        from a best rep — the derivation is a comment beside each floor.
      </p>
    </div>
  </div>
  <div class="stats at" style="--n: 3">
    <div><b class="mono">{gateFacts.registered}</b><span>benchmarks registered</span></div>
    <div><b class="mono">{gateFacts.committed}</b><span>committed results</span></div>
    <div><b class="mono">{gateFacts.branches}</b><span>branches scanned</span></div>
    <div><b class="mono">{gateFacts.fromBranches}</b><span>harvested from branches</span></div>
  </div>
</Slide>

<Slide
  act="gold"
  eyebrow="IP and licensing"
  title="AGPL-3.0-only, enforced rather than declared"
  lede="The first question an investor's counsel asks about a serving engine is the licence, so
        here it is with the machinery that keeps it honest."
  steps={2}
>
  <div class="grid2">
    <Kv
      rows={[
        ['licence', 'AGPL-3.0-only', 'network copyleft, chosen deliberately — an open-core position, not an accident'],
        ['headers', 'SPDX line 1, every source file', 'CI-enforced via skywalking-eyes against .licenserc.yaml'],
        ['dependencies', 'deny.toml allowlist', 'licence policy is a lockfile, not a policy document'],
        ['contributors', 'CLA workflow', 'cla.yml gates every pull request'],
        ['provenance', 'signed commits, merge ancestry check', 'merge-ancestry.yml rejects unrecorded history']
      ]}
    />
    <aside class="ask at" style="--n: 2">
      <p class="ask-h mono">The question behind the question</p>
      <p>
        AGPL is a red flag when it is <em>found</em> in a proprietary serving path during
        diligence. It is a position when it is the licence of the whole work, with a CLA that
        keeps relicensing possible. We are the second case, and the CLA is why.
      </p>
    </aside>
  </div>
</Slide>

<Slide
  act="gold"
  eyebrow="Engineering"
  title="The invariants CI actually enforces"
  lede="House rules only count if a machine says no. These do."
  steps={2}
>
  <div class="grid3">
    <article class="at" style="--n: 1">
      <h3>Structure</h3>
      <ul>
        <li>500-LoC cap per Rust source file, enforced by <code class="mono">file-size-cap.yml</code></li>
        <li>SSOT — every datum has one authoritative source and the rest derive</li>
        <li>No implicit defaults in production paths; fail fast instead</li>
      </ul>
    </article>
    <article class="at" style="--n: 1">
      <h3>Correctness</h3>
      <ul>
        <li>Serve matrix per image: boot, coherence, greedy determinism, tool reliability</li>
        <li>Kernel compile and coverage workflows on every change</li>
        <li>Clippy at deny-warnings, workspace-wide</li>
      </ul>
    </article>
    <article class="at" style="--n: 2">
      <h3>Supply chain</h3>
      <ul>
        <li>Dedicated security workflow; disclosure policy in <code class="mono">SECURITY.md</code></li>
        <li>Release and install-canary pipelines, separate from dev builds</li>
        <li>Lighthouse budget on this very site, in CI</li>
      </ul>
    </article>
  </div>
</Slide>

<Slide
  act="gold"
  eyebrow="Over to you"
  title="Run it, and tell us where we are wrong"
  lede="The strongest thing we can hand a technical analyst is not a chart. It is the box, the
        commands, and a file that already records what we got wrong."
>
  <div class="grid2">
    <Cmd
      label="start here"
      lines={[
        `spark benchmark list concurrency-sweep`,
        `spark benchmark run concurrency-sweep --url YOUR_VLLM_URL --model CHECKPOINT \\`,
        `    --param concurrencies=1,4,8,16 --param isls=512 --param osl=320`,
        `spark benchmark run concurrency-sweep --pull-request-gate --yes`
      ]}
      note="The first line prints every parameter and its default. The second measures whatever you already have serving. The third is the one CI runs. Expect ~2 hours on a single GB10 including the model download."
    />
    <ul class="links">
      <li><span class="mono">results</span><a class="link" href={links.results} target="_blank" rel="noopener">{claim.resultsDoc}</a></li>
      <li><span class="mono">source</span><a class="link" href={links.repo} target="_blank" rel="noopener">github.com/Avarok-Cybersecurity/atlas</a></li>
      <li><span class="mono">gates</span><a class="link" href={links.gateDoc} target="_blank" rel="noopener">what “verified” means</a></li>
    </ul>
  </div>
  <p class="close">
    Every number in this deck is read from the same generated records the front page renders. If
    the ladder is re-run and a rung is lost, these slides say so on the next build.
  </p>
</Slide>

<style>
  .grid2 {
    display: grid;
    grid-template-columns: 1.5fr 1fr;
    gap: 2.4em;
    align-items: start;
  }
  .grid3 {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 1.6em;
  }
  .lead {
    color: var(--t2);
    line-height: 1.65;
    margin-bottom: 0.9em;
    max-width: 58ch;
  }

  .ask {
    border: 1px solid var(--border-strong);
    border-top: 2px solid var(--sx);
    background: var(--card);
    border-radius: 6px;
    padding: 1em 1.1em;
  }
  .ask-h {
    font-size: 0.74em;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: var(--sx);
    margin-bottom: 0.6em;
  }
  .ask p {
    color: var(--t2);
    line-height: 1.6;
    font-size: 0.92em;
  }

  .notes {
    list-style: none;
    display: grid;
    gap: 0;
    max-width: 96ch;
  }
  .notes li {
    display: grid;
    grid-template-columns: 11ch 1fr;
    gap: 1.2em;
    padding: 0.55em 0;
    border-bottom: 1px solid var(--border);
    align-items: baseline;
  }
  .nt {
    font-size: 0.74em;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: var(--sx);
  }
  .notes span:last-child {
    color: var(--t2);
    line-height: 1.6;
  }
  .notes code {
    color: var(--t1);
    font-size: 0.9em;
  }

  .tb2 {
    border-collapse: collapse;
    width: 100%;
    font-size: 0.82em;
  }
  .tb2 th,
  .tb2 td {
    text-align: right;
    padding: 0.32em 0.7em;
    border-bottom: 1px solid var(--border);
  }
  .tb2 th:first-child,
  .tb2 td:first-child {
    text-align: left;
    font-family: var(--font-mono);
    font-size: 0.94em;
  }
  .tb2 th {
    font-size: 0.82em;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: var(--t3);
    font-weight: 600;
    font-family: var(--font-sans);
  }
  .note2 {
    margin-top: 0.8em;
    color: var(--t3);
    font-size: 0.82em;
    line-height: 1.55;
  }

  .stats {
    margin-top: 1.2em;
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 1px;
    background: var(--border);
    border: 1px solid var(--border);
    border-radius: 6px;
    overflow: hidden;
  }
  .stats div {
    background: var(--card);
    padding: 0.6em 0.9em;
    display: grid;
    gap: 0.2em;
  }
  .stats b {
    font-size: 1.35em;
    font-weight: 700;
    color: var(--sx);
    line-height: 1;
  }
  .stats span {
    font-size: 0.78em;
    color: var(--t3);
  }

  .grid3 h3 {
    font-size: 1em;
    font-weight: 700;
    margin-bottom: 0.6em;
    padding-bottom: 0.4em;
    border-bottom: 2px solid var(--sx);
  }
  .grid3 ul {
    list-style: none;
    display: grid;
    gap: 0.55em;
  }
  .grid3 li {
    color: var(--t2);
    font-size: 0.88em;
    line-height: 1.55;
    padding-left: 1em;
    position: relative;
  }
  .grid3 li::before {
    content: '›';
    position: absolute;
    left: 0;
    color: var(--sx);
  }
  .grid3 code {
    color: var(--t1);
    font-size: 0.88em;
  }

  .links {
    list-style: none;
    display: grid;
    gap: 0.55em;
    align-content: start;
  }
  .links li {
    display: grid;
    grid-template-columns: 9ch 1fr;
    gap: 0.9em;
    align-items: baseline;
    padding-bottom: 0.5em;
    border-bottom: 1px solid var(--border);
  }
  .links span {
    font-size: 0.74em;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: var(--t3);
  }
  .links a {
    font-size: 0.92em;
  }

  .close {
    margin-top: 1.6em;
    color: var(--t3);
    font-size: 0.9em;
    max-width: 92ch;
  }
</style>
