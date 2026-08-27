// =============================================================================
// Central data source for ALL site copy + links (SSOT).
// Components are presentation only. Generated data (models, benchmarks, stars)
// lives in *.generated.json and is imported by components directly.
//
// VOICE: confident and plain, builder-to-builder. No colons, no em dashes, no
// semicolons in the visible copy. Commas and periods only. No exclamation
// marks, no emoji in prose, and nothing that reads as a novelty. An operator
// evaluating an engine for a rack and a developer evaluating it for a desk are
// reading the same page.
//
// SCOPE: Atlas spans a range, from edge class accelerators through workstations
// to multi node deployments. Copy must not narrow that to "desk machines", and
// must not claim a tier we have not verified. Verified silicon is named. The
// rest is stated as direction, with its status attached.
//
// CLAIM POLICY: every performance number is generated-from-repo or mechanically
// true. No hand-typed tok/s. No bare "fastest / best / #1". Third-party names
// carry a live artifact link and a status-true tense.
// =============================================================================

// --- canonical links ---------------------------------------------------------
export const githubUrl = 'https://github.com/Avarok-Cybersecurity/atlas';
export const discordUrl = 'https://discord.gg/RQcGakU2jW';
export const xUrl = 'https://x.com/AtlasInferenceX';
export const xHandle = '@AtlasInferenceX';
export const redditUrl = 'https://www.reddit.com/r/LocalLLaMA/comments/1rmvxo3/';
export const firstPostUrl =
  'https://www.reddit.com/r/LocalLLaMA/comments/1rkefjw/solved_the_dgx_spark_102_stable_toks_qwen3535ba3b/';
export const recipesUrl = 'https://github.com/Avarok-Cybersecurity/atlas-recipes';
export const guideUrl =
  'https://github.com/Avarok-Cybersecurity/atlas/blob/main/docs/GB10_DEPLOYMENT_GUIDE.md';
export const verifiedAnchor =
  'https://github.com/Avarok-Cybersecurity/atlas/blob/main/docs/GB10_DEPLOYMENT_GUIDE.md#8-what-verified-means-so-you-can-trust-an-image';
export const gateSrcUrl =
  'https://github.com/Avarok-Cybersecurity/atlas/blob/main/tests/gate_results.py';
export const discussionsUrl = 'https://github.com/Avarok-Cybersecurity/atlas/discussions';
export const goodFirstIssuesUrl =
  'https://github.com/Avarok-Cybersecurity/atlas/labels/good%20first%20issue';
// Single source of truth for contact addresses (footer + reach-out section).
export const contactEmails = ['thomas@atlasinference.io'];

// third-party artifacts (link-or-cut, each verified live July 2026)
export const transformersPrUrl = 'https://github.com/huggingface/transformers/pull/46423';
export const hubKernelUrl = 'https://huggingface.co/kernels/Atlas-Inference/gdn';
export const scaleUrl = 'https://docs.scale-lang.com/stable/';
export const qwenAmbassadorUrl = 'https://qwen.ai/ambassador';
export const strixPrUrl = 'https://github.com/Avarok-Cybersecurity/atlas/pull/187';
export const mlperfResultsUrl = 'https://mlcommons.org/benchmarks/inference-datacenter/';
export const mlcommonsEndpointsPrUrl = 'https://github.com/mlcommons/endpoints/pull/346';
export const mlcommonsArticleUrl =
  'https://mlcommons.org/2026/07/mlperf-inference-v61-edge-agentic/';
export const nvidiaInceptionUrl = 'https://www.nvidia.com/en-us/startups/';

// --- brand -------------------------------------------------------------------
export const tagline =
  'Pure Rust inference, from the device in your hand to the datacenter rack.';

// --- commands (one flagship recipe, kept in lockstep with static/install.sh)
export const flagshipRecipe = 'qwen3.6-35b-a3b-fp8-mtp';
export const quickInstall = 'cargo install atlasctl';
/// Where install.sh is served from. One authority: the join one-liner in
/// `joincommand.js` builds on this too, and a second copy is how the two drift.
export const installerUrl = 'https://atlasinference.io/install.sh';
export const runCommand = `curl -fsSL ${installerUrl} | sh`;
/// Install the agent as a service — deliberately `install`, not `run`: a bare
/// `run` dies with the terminal that started it, and the machine silently
/// leaves the fleet the next time someone closes an ssh session.
export const startAgentCommand = 'atlasctl agent install';
export const runCommandRaw =
  'atlasctl run qwen3.6-35b-a3b-fp8-mtp';

// --- hardware acknowledgment (modest banner) ---------------------------------
// --- announcement band (the strip above the hero) ----------------------------
export const announcement = {
  line: 'Atlas is partnering with Avarok Cybersecurity.',
  sub: 'Post quantum security and high performance inference across edge, workstation, and datacenter deployments. The model and your data stay on hardware you own.',
  ctaText: 'avarok.net',
  ctaUrl: 'https://avarok.net'
};

// --- nav (SSOT for both the desktop bar and the mobile drawer) ---------------
export const nav = {
  links: [
    { text: 'Verified', href: '/#verified' },
    { text: 'News', href: '/#news' },
    { text: 'Hardware', href: '/#hardware' },
    { text: 'Models', href: '/#models' },
    { text: 'Get running', href: '/#run' },
    // `.html`, not `/control`. adapter-static writes this route to
    // control.html, and the deploy target serves files literally: no extension
    // guessing, and no directory index outside the document root. /control is
    // the SPA fallback at best and a 500 at worst. If the server ever gains
    // `try_files $uri $uri.html`, this becomes '/control'.
    { text: 'Control', href: '/control.html' }
  ],
  menuLabel: 'Menu',
  closeLabel: 'Close menu'
};

// --- hero --------------------------------------------------------------------
export const hero = {
  badge: 'Open source. Pure Rust and CUDA. Verified on GB10.',
  headline: ['One inference engine, from the device in your hand', 'to the datacenter rack.'],
  sub:
    'Atlas is an open source LLM engine written in Rust and CUDA. One ~75 MB binary, no Python, no PyTorch. It runs on edge class accelerators today, scales across nodes with expert parallelism, and holds throughput at the concurrency a datacenter serves. What ships is what we verify, and we bench every release.',
  challenge: {
    claim: 'First token in under 90 seconds on a DGX Spark.',
    lead: 'Do not take our word for it.',
    fine:
      'Median of our GB10 runs, model cached, atlas 59616dc, Jul 2026. Same command below, run it and time it yourself.'
  },
  primaryCta: 'Star on GitHub',
  secondaryCta: 'Get running',
  discordCta: 'Join the Discord'
};

// --- proof strip (prominent, right under the hero) ---------------------------
export const proof = {
  label: '// receipts, not adjectives',
  items: [
    { text: 'Merged into Hugging Face Transformers', url: transformersPrUrl },
    { text: 'Qwen Dev Ambassadors', url: qwenAmbassadorUrl },
    { text: 'MLPerf Edge Agentic task force', url: mlcommonsArticleUrl },
    { text: 'Built with SCALE by Spectral Compute', url: scaleUrl }
  ]
};

// --- news band ----------------------------------------------------------------
// Newest first. Every card points at a primary source, no numbers before
// MLCommons publishes. See CLAIM POLICY at the top of this file.
export const news = {
  label: '// 03 · news',
  title: 'What just happened.',
  sub:
    'Three things landed this month and every card links straight to the primary source.',
  items: [
    {
      tag: 'MLCommons',
      date: 'July 2026',
      featured: true,
      title: 'We helped build the MLPerf Edge Agentic benchmark',
      body:
        'MLCommons published the new MLPerf Inference v6.1 edge agentic benchmark and Atlas Inference is named as a contributor alongside NVIDIA. It measures multi turn agentic LLMs on a single edge accelerator, BFCL v4 for accuracy and replayed agentic coding trajectories for performance. We helped shape it because it is the benchmark that actually looks like the work.',
      cta: 'Read the MLCommons announcement',
      url: mlcommonsArticleUrl
    },
    {
      tag: 'AMD',
      date: 'July 2026',
      title: 'Atlas running on AMD Strix Halo',
      body:
        'AMD provided a Strix Halo desktop and we brought Atlas to it through SCALE, custom kernels and all. That machine is the box we ran and submitted MLPerf on. One codebase now covers both vendors with no HIP port and no second kernel tree.',
      cta: 'See the post on X',
      url: xUrl
    },
    {
      tag: 'MLPerf v6.1',
      date: 'Submitted',
      title: 'Our MLPerf submission is in',
      body:
        'Atlas is submitted to MLPerf Inference v6.1 in the closed edge division, the same CUDA source across NVIDIA GB10 and AMD gfx1151. Results stay under embargo until MLCommons publishes.',
      cta: 'Follow along in Discord',
      url: discordUrl
    }
  ]
};

// --- star / social proof -----------------------------------------------------
export const stars = {
  label: '// 07 · community',
  title: 'Built in the open, starred in the open.',
  sub:
    'Atlas went from one Reddit post to a community running it on their own hardware. The curve below is live, regenerated from the GitHub API on every deploy.',
  cta: 'Star the repo'
};

export const testimonials = [
  {
    quote:
      'Night and day compared to the 10 minute torch.compile cycle. Startup in about 15 seconds and it just stays coherent in an agentic loop.',
    author: 'ronald_15496',
    source: '#general',
    sourceUrl: discordUrl
  },
  {
    quote:
      'Testing Atlas on a DGX Spark in an agentic workflow for over an hour. Super impressed. Spark is actually awesome with Atlas.',
    author: 'PersonWhoThinks',
    source: 'r/LocalLLaMA',
    sourceUrl: redditUrl
  },
  {
    quote:
      'I had grown tired of the usual stack and was hoping for something like this. Really surprised and impressed. So glad I bought a Spark.',
    author: 'tetsuro59',
    source: '#general',
    sourceUrl: discordUrl
  }
];

// --- community / discord push ------------------------------------------------
export const community = {
  label: '// come build with us',
  title: 'The action is in Discord.',
  body:
    'Hundreds of builders are running Atlas on their own hardware right now. We are in there every day, shipping fixes, taking model requests, and tuning kernels in the open. Your machine is the test fleet and your voice sets the roadmap.',
  cta: 'Join the Discord',
  sub: 'Active every day.'
};

// --- verified performance (the gate receipt) ---------------------------------
export const verified = {
  label: '// 02 · verified',
  title: 'Every number is a receipt.',
  sub:
    'The website is a build artifact of the repo. Models come from recipes, performance comes from committed gate enforced baselines, stamped with commit and date. If a number is not in the repo, it is not on this page.',
  pendingHeadline: 'MLPerf v6.1 submitted',
  pendingBody:
    'Our MLPerf Inference v6.1 submission is in, closed edge division, on both GB10 and gfx1151. The numbers render right here in this receipt the moment MLCommons publishes them, gate enforced, reproducible, stamped. Until then the release gate holds every image to liveness and coherence, and you can reproduce any run yourself.',
  mechanism:
    'A release that ships slower than the committed baseline fails our gate. That one sentence is the whole positioning.',
  reproLead: 'Reproduce the matrix',
  challengeLine: 'Beat these numbers or catch a regression, open an issue and we will feature it.',
  // Rendered directly under the ladder chart. The numbers in this block are
  // derived from ladder.generated.json by lib/ladder.js, never typed here.
  scale: {
    title: 'The top of the ladder is the part that matters.',
    lead:
      'Agentic work does not arrive as one conversation at a time. It arrives as fleets of tool calling agents sharing a context bus, fanning out and rejoining, and the engine underneath them is judged where the requests pile up rather than at a single stream.',
    tail:
      'That gap is the whole thesis. An engine that flattens under load caps how many agents you can actually run, on any hardware you put it on. Holding the curve is what turns one accelerator into a swarm, and it is why the same engine is worth running on a rack.'
  }
};

export const mlperfCopy = {
  preparing:
    'We are prepping a submission to MLPerf Inference v6.1, the same CUDA source submitted across NVIDIA GB10 and AMD gfx1151. Aiming to be the first to run identical CUDA on both.',
  submitted:
    'Submitted to MLPerf Inference v6.1 in the closed edge division, the same CUDA source across NVIDIA GB10 and AMD gfx1151. Results are under embargo until MLCommons publishes them.',
  published: 'Published in MLPerf Inference v6.1 across NVIDIA GB10 and AMD gfx1151.'
};

export const mlcommons = {
  line:
    'Atlas is a member of MLCommons and sits on the Edge LLM taskforce, where we helped shape the new v6.1 edge agentic benchmark. MLCommons names Atlas Inference as a contributor in the announcement.',
  linkText: 'read the announcement',
  url: mlcommonsArticleUrl
};

export const mlperfTrademark =
  'The MLPerf name and logo are registered and unregistered trademarks of MLCommons Association in the United States and other countries. All rights reserved. Unauthorized use strictly prohibited. See mlcommons.org for more information.';

// --- hardware ----------------------------------------------------------------
export const hardware = {
  label: '// 04 · hardware',
  title: 'One engine, every tier.',
  sub:
    'The same Rust and CUDA source runs on both platforms below, compiles for NVIDIA and AMD without a second kernel tree, and scales from a single accelerator to expert parallel across nodes. These are the parts we have verified. The range is the design, and the list grows.',
  cards: [
    {
      name: 'NVIDIA DGX Spark',
      chip: 'GB10 · SM121',
      status: 'verified',
      statusText: 'Verified today',
      body:
        'One multi model binary serves a full matrix of hand tuned targets on a single GB10. NVFP4 and FP8, MTP speculative decoding, EP=2 across two Sparks. Every target passes the serve matrix before we cut an image.',
      cta: { text: 'Read the deployment guide', url: guideUrl }
    },
    {
      name: 'AMD Strix Halo',
      chip: 'gfx1151 · RDNA 3.5',
      status: 'verified',
      statusText: 'MLPerf submitted',
      body:
        'One codebase, both vendors. Our CUDA kernels compile straight for AMD gfx1151 with SCALE by Spectral Compute. No HIP port, no second kernel tree. AMD provided the Strix Halo desktop we ran and submitted our MLPerf Inference v6.1 numbers on.',
      cta: { text: 'Join the bring up, PR #187', url: strixPrUrl },
      scale: { text: 'Built with SCALE by Spectral Compute', url: scaleUrl }
    }
  ]
};

// --- models ------------------------------------------------------------------
export const models = {
  label: '// 05 · models',
  title: 'Every model here has a recipe.',
  sub:
    'Pick a vendor, then a family. Every card maps to one recipe in atlas-recipes, so the site cannot list a model we do not ship. Copy the command and run it as is. Qwen3.6 leads because it is our flagship.',
  qwen: {
    kernel: 'Our fused Qwen3.6 Gated DeltaNet kernel ships in Hugging Face Transformers.',
    kernelUrl: transformersPrUrl,
    hubText: 'kernel repo on the Hub',
    hubUrl: hubKernelUrl,
    ambassador: 'We are Qwen Dev Ambassadors and we ship a recipe for every Qwen release.',
    ambassadorUrl: qwenAmbassadorUrl
  }
};

// --- get running -------------------------------------------------------------
export const getRunning = {
  label: '// 06 · get running',
  title: 'Up and running in one command.',
  sub:
    'This is the first 60 seconds. Everything after, per model recipes, EP=2, tuning, lives in the docs.',
  inspectNote:
    'Rather not pipe curl to a shell. Install atlasctl from crates.io, then run the flagship recipe direct.',
  docsCta: 'Read the deployment guide',
  quickstartHint:
    'The script downloads a prebuilt atlasctl, verifies its checksum, and installs it to ~/.local/bin. No Python, no Rust toolchain. Run it with --uninstall to reverse it.'
};

// --- mission -----------------------------------------------------------------
export const mission = {
  title: 'AI worth having, on hardware you own.',
  statement:
    'AI worth having should run on hardware you own, whether that is an accelerator at the edge, the workstation under your desk, or a rack you operate. We build one engine for the whole range, and we verify it on the silicon we can put our hands on.',
  // Shown small, under the statement — the reasoning behind it rather than a
  // second full-size claim.
  footnote:
    'Pure Rust because the whole stack should be inspectable by one person, HTTP to kernel dispatch, no interpreter in the hot path. That is also what makes one binary portable across the range, from a part measured in watts to a node measured in kilowatts. We develop on machines provided by NVIDIA and AMD, and the test fleet is the community running it. If a model matters to you, it matters to us.'
};

// --- contribute --------------------------------------------------------------
export const contribute = {
  label: '// 08 · build with us',
  title: 'Your machine is the test fleet.',
  sub:
    'Atlas grows from the machines it runs on. Every path below is real and linked. Contributions ship in the Community Edition under AGPLv3, and the CLA lets us re license for the Enterprise Edition.',
  paths: [
    {
      title: 'Run the serve matrix',
      body: 'Boot the matrix on your own GB10 and report what you see. Regressions and wins both get featured.',
      cta: 'Deployment guide',
      url: guideUrl
    },
    {
      title: 'Add or tune a recipe',
      body: 'Recipes are the model SSOT. Add a model, tune a quant, open a PR against atlas-recipes.',
      cta: 'atlas-recipes',
      url: recipesUrl
    },
    {
      title: 'Kernels in Rust and CUDA',
      body: 'Hand tuned attention, MoE, GDN, Mamba-2 for Blackwell. Register level work, no generic fallbacks.',
      cta: 'Good first issues',
      url: goodFirstIssuesUrl
    },
    {
      title: 'Docs, triage, ideas',
      body: 'Improve the guide, triage issues, or just tell us what you are running in Discord.',
      cta: 'Discussions',
      url: discussionsUrl
    }
  ],
  cla: 'Contributions are AGPLv3 and the CLA permits Enterprise re licensing. See CONTRIBUTING.md.'
};

// --- roadmap (next up + artifact-linked) -------------------------------------
export const roadmap = {
  rowTitle: 'What we are building next.',
  rowSub: 'Everything shipped links to an issue, a PR, or the Discord where the work happens. Anything not yet committed carries its status, and we do not round it up.',
  items: [
    {
      title: 'Three node GB10 topology',
      status: 'Next up',
      body: 'Three GB10s in one rig for models that will not fit across two. More memory, more experts, more concurrency headroom. We are wiring up the topology now.',
      cta: 'Discuss the topology in Discord',
      url: discordUrl
    },
    {
      title: 'Intel Arc Pro B70',
      status: 'In talks',
      body: 'Active conversations with Intel about bringing Atlas to the Arc Pro B70. Nothing is signed yet, and this card will say so until it is.',
      cta: 'Follow along in Discord',
      url: discordUrl
    },
    {
      title: 'AMD Strix Halo',
      status: 'MLPerf submitted',
      body: 'Native gfx1151 through SCALE. AMD provided a Strix Halo desktop and we brought Atlas to it, custom kernels and all.',
      cta: 'PR #187',
      url: strixPrUrl
    },
    {
      title: 'MLPerf Inference v6.1',
      status: 'Submitted',
      body: 'The same CUDA source submitted across GB10 and gfx1151, closed edge division. No numbers until MLCommons publishes.',
      cta: 'Read the benchmark announcement',
      url: mlcommonsArticleUrl
    },
    {
      title: 'Qwen GDN kernel upstream',
      status: 'Merged',
      body: 'Our fused Gated DeltaNet kernel for Qwen3.6 landed in Hugging Face Transformers.',
      cta: 'transformers #46423',
      url: transformersPrUrl
    },
    {
      title: 'Bigger model support',
      status: 'Tracking',
      body: 'Large MoE NVFP4 ports across EP topologies, DeepSeek and Kimi class, tracked in the open.',
      cta: 'Open issues',
      url: 'https://github.com/Avarok-Cybersecurity/atlas/issues'
    }
  ]
};

// --- FAQ ---------------------------------------------------------------------
// Rendered on the page AND emitted as FAQPage structured data. Both come from
// here, which is what keeps the markup answering the same questions the page
// answers — marking up an answer a visitor cannot see is a search-policy
// violation, not a shortcut.
//
// CLAIM POLICY applies: every answer below restates something already shown
// elsewhere on this page or in a linked artifact. No new numbers.
export const faq = {
  label: '// 09 · questions',
  title: 'The questions we actually get asked.',
  sub: 'Short answers, each one backed by something on this page or in the repo.',
  items: [
    {
      q: 'What is Atlas?',
      a: 'An open source LLM inference engine written in pure Rust and CUDA. It serves an OpenAI-compatible API from a single binary, with no Python and no PyTorch in the serving path. One codebase covers the range, from edge-class accelerators through workstations to expert-parallel deployments across nodes.'
    },
    {
      q: 'What hardware does Atlas run on?',
      a: 'NVIDIA DGX Spark (GB10) is verified today, and AMD Strix Halo (gfx1151) runs the same CUDA source compiled through SCALE by Spectral Compute — one codebase, no HIP port. Both were submitted to MLPerf Inference v6.1 in the closed edge division.'
    },
    {
      q: 'Is Atlas faster than vLLM on a DGX Spark?',
      a: 'On the published concurrency ladder, yes at every rung from C=1 to C=128, by 1.012x to 1.225x against whichever vLLM configuration is faster at that concurrency. The margin is widest at the top, because between C=64 and C=128 Atlas keeps scaling and the vLLM configuration that leads the mid-ladder stops. Same box, same checkpoint, same client, same prompts, greedy sampling with matched penalties. The full campaign log, including the rungs we lost on the way, is in the repo.'
    },
    {
      q: 'How do I install it?',
      a: 'One command: curl -fsSL https://atlasinference.io/install.sh | sh. It downloads a prebuilt atlasctl, verifies its checksum, and installs to ~/.local/bin. If you would rather not pipe curl to a shell, cargo install atlasctl does the same thing from source.'
    },
    {
      q: 'Which models can I run?',
      a: 'Every model on this page maps to a recipe in the atlas-recipes repository, which is the single source of truth — the site cannot list a model that has no recipe. Qwen3.6 is the flagship family, alongside Gemma, Nemotron, Mistral, MiniMax and DeepSeek.'
    },
    {
      q: 'What does “verified” mean here?',
      a: 'An image ships only after the serve matrix passes: every model boots, stays coherent under greedy determinism with no token leakage and reliable tool calls, and holds throughput within 10% of its committed baseline. A release that ships slower than its baseline fails the gate.'
    },
    {
      q: 'Why does concurrency matter more than single-stream speed?',
      a: 'Because agentic systems do not send one request at a time. A fleet of tool-calling agents sharing a context bus arrives as many concurrent streams, so the engine is judged where the requests pile up. On the published ladder Atlas keeps gaining throughput from C=64 to C=128 while the leading vLLM configuration does not, and an engine that flattens under load caps how many agents a given box can actually run.'
    },
    {
      q: 'What license is Atlas under, and can I use it commercially?',
      a: 'The Community Edition is AGPL-3.0-only. Contributions are covered by a CLA that permits re-licensing for the Enterprise Edition. If you are running Atlas in production or need different terms, email us.'
    },
    {
      q: 'Does Atlas run multi-node?',
      a: 'Yes. EP=2 expert parallelism across two DGX Sparks is supported and shipped as recipes; those cards are marked EP=2 in the model list. A three-node GB10 topology is being wired up now.'
    }
  ]
};

// --- reach out ---------------------------------------------------------------
export const reachout = {
  label: '// 10 · reach out',
  title: 'Come work with us.',
  sub:
    'Building on Spark or Strix, deploying at rack scale, bringing hardware to the table, or wanting to partner. We want to hear from you.',
  cards: [
    {
      emoji: '💼',
      title: 'Business',
      body: 'Running Atlas in production or evaluating the Enterprise Edition. Tell us what you need and we will scope it with you.'
    },
    {
      emoji: '🤝',
      title: 'Partnerships',
      body: 'Frameworks, benchmarks, standards bodies. If it advances inference on hardware people own, we want the conversation.'
    },
    {
      emoji: '🔧',
      title: 'Hardware',
      body: 'Silicon you want Atlas running on. Tell us about it and we will scope a bring up.'
    }
  ],
  emails: contactEmails,
  discordCta: 'Or find us in Discord'
};

// --- ask the codebase (chat modal) -------------------------------------------
// Copy SSOT for the CodeChat modal. Retrieval runs locally in wasm from the
// repo corpus, answers come from free OpenRouter models with the visitor's own
// key. Same voice rules as everything above.
export const codeChat = {
  navLabel: 'Ask the codebase',
  closeLabel: 'Close ask the codebase',
  label: '// 11 \u00b7 ask the codebase',
  title: 'Ask the codebase.',
  sub:
    'The whole Atlas repo is embedded into a vector lattice that runs right here in your browser. Ask a question, get an answer with file and line receipts.',
  boot: [
    'atlas code lattice online',
    'retrieval runs locally in wasm, only the model call leaves this page',
    'pick a question or type your own'
  ],
  starters: [
    'How does MTP speculative decoding pick which draft tokens to keep?',
    'Where does the scheduler decide which requests join a decode batch?',
    'How do the NVFP4 GEMM kernels get dispatched on GB10?'
  ],
  key: {
    tag: 'openrouter key',
    lead:
      'Answers come from free models on OpenRouter, so you bring your own key. It stays in this browser and we never see it.',
    linkText: 'grab a free key at openrouter.ai/keys',
    url: 'https://openrouter.ai/keys',
    placeholder: 'sk-or-v1-...',
    inputLabel: 'OpenRouter API key',
    reveal: 'show',
    conceal: 'hide',
    save: 'connect',
    connectedTag: 'connected',
    connectedNote: 'key stored in this browser only',
    change: 'swap key'
  },
  status: {
    idle: 'standby',
    'wasm-init': 'starting engine',
    manifest: 'fetching manifest',
    'loading-cached': 'reading local cache',
    downloading: 'downloading corpus',
    caching: 'writing local cache',
    indexing: 'indexing',
    ready: 'ready',
    error: 'fault'
  },
  offlineBadge: 'cached \u00b7 offline',
  phase: {
    retrieving: 'searching the lattice',
    reranking: 'reranking matches',
    thinking: 'reasoning',
    writing: 'writing'
  },
  trace: {
    label: 'reasoning',
    reasonedPrefix: 'reasoned for',
    show: 'show',
    hide: 'hide'
  },
  loader: {
    title: 'mounting the code lattice',
    commitLabel: 'commit',
    sizeLabel: 'download',
    chunksLabel: 'chunks',
    stages: {
      'wasm-init': 'start the wasm engine',
      manifest: 'fetch the corpus manifest',
      corpus: 'load the corpus',
      indexing: 'index the chunks'
    },
    cancelNote: 'close anytime, the download cancels cleanly and nothing partial is kept'
  },
  input: {
    placeholder: 'ask about kernels, scheduling, quantization, anything in the repo',
    ask: 'ask',
    hintLoading: 'the corpus is still mounting, hang tight',
    hintNoKey: 'connect your OpenRouter key above to ask',
    fine: 'answers are generated and can be wrong, the source links are real so read them before you trust them'
  },
  answerTag: 'answer',
  sourcesHeading: 'source receipts',
  sourcesOne: 'source',
  sourcesMany: 'sources',
  loadFail: 'The chat window did not load, maybe the network blinked. Close and try again.',
  model: {
    label: 'answer model',
    reset: 'back to free'
  },

  errors: {
    wasm: {
      tag: 'engine fault',
      body: 'The wasm engine failed to start. Usually a one off, a retry brings it right up.',
      retry: 'restart engine'
    },
    manifest: {
      tag: 'manifest unreachable',
      body: 'Could not reach the corpus manifest. Check your connection and retry.',
      retry: 'retry'
    },
    corpus: {
      tag: 'download failed',
      body: 'The corpus download did not finish. Nothing partial was kept, retry whenever.',
      retry: 'retry download'
    },
    decompress: {
      tag: 'unpack failed',
      body: 'Your browser could not unpack the corpus stream. Any current Chrome, Edge, Firefox or Safari handles it.',
      retry: 'retry'
    },
    rate: {
      tag: 'rate limited',
      body: 'The free OpenRouter models are catching their breath. Give it a few seconds.',
      retry: 'ask again'
    },
    quota: {
      tag: 'daily limit reached',
      body: 'Your OpenRouter key has spent its free model allowance for today. Retrieval still runs locally so the corpus stays loaded and ready.',
      reset: 'The free pool refills at',
      paid: 'switch to the paid model and spend my own credits',
      retry: 'ask again'
    },
    key: {
      tag: 'key rejected',
      body: 'OpenRouter did not accept that key. Paste a fresh one and reconnect.',
      retry: 'swap key'
    },
    generic: {
      tag: 'hiccup',
      body: 'That one did not go through. Retry in a moment.',
      retry: 'retry'
    }
  }
};

// --- footer ------------------------------------------------------------------
export const footer = {
  tagline: 'Pure Rust and CUDA inference, from the device in your hand to the datacenter rack.',
  license: 'Dual licensed. Community Edition under AGPLv3, Enterprise Edition commercial.',
  cols: [
    {
      heading: 'Project',
      links: [
        { text: 'GitHub', url: githubUrl },
        { text: 'Deployment guide', url: guideUrl },
        { text: 'Recipes (SSOT)', url: recipesUrl },
        { text: 'License AGPLv3', url: githubUrl + '/blob/main/LICENSE' }
      ]
    },
    {
      heading: 'Community',
      links: [
        { text: 'Discord', url: discordUrl },
        { text: 'Discussions', url: discussionsUrl },
        { text: 'Good first issues', url: goodFirstIssuesUrl },
        { text: 'r/LocalLLaMA', url: redditUrl }
      ]
    }
  ]
};
