// =============================================================================
// Central data source for ALL site copy + links (SSOT).
// Components are presentation only. Generated data (models, benchmarks, stars)
// lives in *.generated.json and is imported by components directly.
//
// VOICE: enthusiastic, builder-to-builder. No colons, no em dashes, no
// semicolons in the visible copy. Commas and periods only.
//
// CLAIM POLICY: every performance number is generated-from-repo or mechanically
// true. No hand-typed tok/s. No bare "fastest / best / #1". Third-party names
// carry a live artifact link and a status-true tense.
// =============================================================================

// --- canonical links ---------------------------------------------------------
export const githubUrl = 'https://github.com/Avarok-Cybersecurity/atlas';
export const discordUrl = 'https://discord.gg/6vDbKaKrKD';
export const xUrl = 'https://x.com/atlasinference';
export const xHandle = '@atlasinference';
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
export const contactEmails = ['debaterishaqui@gmail.com', 'thomas@avarok.net'];

// third-party artifacts (link-or-cut, each verified live July 2026)
export const transformersPrUrl = 'https://github.com/huggingface/transformers/pull/46423';
export const hubKernelUrl = 'https://huggingface.co/kernels/Atlas-Inference/gdn';
export const scaleUrl = 'https://docs.scale-lang.com/stable/';
export const qwenAmbassadorUrl = 'https://qwen.ai/ambassador';
export const strixPrUrl = 'https://github.com/Avarok-Cybersecurity/atlas/pull/187';
export const mlperfResultsUrl = 'https://mlcommons.org/benchmarks/inference-datacenter/';
export const mlcommonsEndpointsPrUrl = 'https://github.com/mlcommons/endpoints/pull/346';

// --- brand -------------------------------------------------------------------
export const tagline = 'Pure Rust inference, tuned for the machine on your desk.';

// --- commands (one flagship recipe, kept in lockstep with static/quickstart.sh)
export const flagshipRecipe = 'qwen3.6-35b-a3b-fp8-mtp';
export const quickInstall = 'uvx sparkrun setup install';
export const runCommand = 'curl -fsSL https://atlasinference.io/quickstart.sh | sh';
export const runCommandRaw =
  'uvx sparkrun setup install && sparkrun run @atlas/qwen3.6-35b-a3b-fp8-mtp --hosts localhost';

// --- hardware acknowledgment (modest banner) ---------------------------------
export const gifts = {
  line: 'Thank you NVIDIA and AMD.',
  sub: 'DGX Spark gifted by NVIDIA, Strix Halo gifted by AMD. Both camps handed us silicon and we intend to contnue proving to the world the raw power of these machines.'
};

// --- hero --------------------------------------------------------------------
export const hero = {
  badge: 'Open source. Pure Rust and CUDA. Verified on GB10.',
  headline: ['The inference engine for the', 'machine on your desk.'],
  sub:
    'Atlas is an open source LLM engine we hand tuned for NVIDIA DGX Spark. One 2.5 GB binary, no Python, no PyTorch. What ships is what we verify, and we bench it every single release.',
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
    { text: 'MLPerf Agentic Edge task force', url: mlcommonsEndpointsPrUrl },
    { text: 'Built with SCALE by Spectral Compute', url: scaleUrl }
  ]
};

// --- star / social proof -----------------------------------------------------
export const stars = {
  label: '// 01 · momentum',
  title: 'Built in the open, starred in the open.',
  sub:
    'Atlas went from one Reddit post to a whole crew of builders running it on their own Sparks. The curve below is live, regenerated from the GitHub API on every deploy.',
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
    'Hundreds of builders are running Atlas on their own Sparks right now. We are in there every single day, shipping fixes, taking model requests, and tuning kernels live. Your machine is the test fleet and your voice sets the roadmap. Pull up.',
  cta: 'Join the Discord',
  sub: 'Active every day. Bring your Spark.'
};

// --- verified performance (the gate receipt) ---------------------------------
export const verified = {
  label: '// 02 · verified',
  title: 'Every number is a receipt.',
  sub:
    'The website is a build artifact of the repo. Models come from recipes, performance comes from committed gate enforced baselines, stamped with commit and date. If a number is not in the repo, it is not on this page.',
  pendingHeadline: 'MLPerf submission in progress',
  pendingBody:
    'We are prepping our numbers for a public MLPerf Inference submission. When they land they render right here in this receipt, gate enforced, reproducible, stamped. Until then the release gate holds every image to liveness and coherence, and you can reproduce any run yourself.',
  mechanism:
    'A release that ships slower than the committed baseline fails our gate. That one sentence is the whole positioning.',
  reproLead: 'Reproduce the matrix',
  challengeLine: 'Beat these numbers or catch a regression, open an issue and we will feature it.'
};

export const mlperfCopy = {
  preparing:
    'We are prepping a submission to MLPerf Inference v6.1, the same CUDA source submitted across NVIDIA GB10 and AMD gfx1151. Aiming to be the first to run identical CUDA on both.',
  submitted:
    'Submitted to MLPerf Inference v6.1 across NVIDIA GB10 and AMD gfx1151. Results are under embargo until MLCommons publishes.',
  published: 'Published in MLPerf Inference v6.1 across NVIDIA GB10 and AMD gfx1151.'
};

export const mlcommons = {
  line:
    'Atlas is a member of MLCommons and sits on the MLPerf Agentic Edge task force, where we helped shape the BFCL-v4 edge agentic benchmark.',
  linkText: 'the edge agentic benchmark work',
  url: mlcommonsEndpointsPrUrl
};

export const mlperfTrademark =
  'The MLPerf name and logo are registered and unregistered trademarks of MLCommons Association in the United States and other countries. All rights reserved. Unauthorized use strictly prohibited. See mlcommons.org for more information.';

// --- hardware ----------------------------------------------------------------
export const hardware = {
  label: '// 03 · hardware',
  title: 'Prosumer first. Desk machines, not clusters.',
  cards: [
    {
      name: 'NVIDIA DGX Spark',
      chip: 'GB10 · SM121',
      status: 'verified',
      statusText: 'Verified today',
      gift: true,
      body:
        'One multi model binary serves a full matrix of hand tuned targets on a single GB10. NVFP4 and FP8, MTP speculative decoding, EP=2 across two Sparks. Every target passes the serve matrix before we cut an image.',
      cta: { text: 'Read the deployment guide', url: guideUrl }
    },
    {
      name: 'AMD Strix Halo',
      chip: 'gfx1151 · RDNA 3.5',
      status: 'bringup',
      statusText: 'In bring up',
      gift: true,
      body:
        'One codebase, both camps. Our CUDA kernels compile straight for AMD gfx1151 with SCALE by Spectral Compute. No HIP port, no second kernel tree. Serving Qwen at NVFP4 quality on a dev branch and stabilizing now.',
      cta: { text: 'Join the bring up, PR #187', url: strixPrUrl },
      scale: { text: 'Built with SCALE by Spectral Compute', url: scaleUrl }
    }
  ]
};

// --- models ------------------------------------------------------------------
export const models = {
  label: '// 04 · models',
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
  label: '// 05 · get running',
  title: 'Up and running in one command.',
  sub:
    'This is the first 60 seconds. Everything after, per model recipes, EP=2, tuning, lives in the docs.',
  inspectNote: 'Rather not pipe curl to a shell. Install sparkrun, then run the flagship recipe direct.',
  docsCta: 'Read the deployment guide',
  quickstartHint:
    'The script checks for sparkrun, installs it with uvx if missing, then runs the flagship Qwen3.6 recipe.'
};

// --- mission -----------------------------------------------------------------
export const mission = {
  label: '// 06 · mission',
  title: 'Local AI worth having, open to all.',
  body: [
    'AI worth having should run on hardware you own. Prosumer machines like DGX Spark and Strix Halo are the first generation that makes that real, and we build for them first.',
    'Pure Rust because the whole stack should be inspectable by one person, HTTP to kernel dispatch, no interpreter in the hot path. We develop on machines granted by NVIDIA and AMD. Both camps handed us silicon and we intend to contnue proving to the world the raw power of these machines.',
    'Open to all. The test fleet is the community desks. If a model matters to you, it matters to us.'
  ]
};

// --- contribute --------------------------------------------------------------
export const contribute = {
  label: '// 07 · contribute',
  title: 'Your machine is the test fleet.',
  sub:
    'Atlas grows from the desks it runs on. Every path below is real and linked. Contributions ship in the Community Edition under AGPLv3, and the CLA lets us re license for the Enterprise Edition.',
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
  label: '// 08 · next up',
  title: 'What we are building next.',
  sub: 'Everything real links to an issue, a PR, or the Discord where the work happens. The teasers are teasers, and we say so.',
  items: [
    {
      title: 'Trifecta, three Sparks',
      status: 'Next up',
      gift: true,
      body: 'Three GB10s in one rig for the really big models. More memory, more experts, more headroom. We are wiring up the topology now.',
      cta: 'Talk trifecta in Discord',
      url: discordUrl
    },
    {
      title: 'Intel Arc Pro B70',
      status: 'In talks',
      body: 'Active conversations with Intel about bringing Atlas to the Arc Pro B70. The email chain is live and we are waiting on confirmation. Nothing signed yet, but we are fired up about it.',
      cta: 'Follow along in Discord',
      url: discordUrl
    },
    {
      title: 'AMD Strix Halo',
      status: 'In bring up',
      gift: true,
      body: 'Native gfx1151 through SCALE. Serving Qwen at NVFP4 quality on a branch and stabilizing CI.',
      cta: 'PR #187',
      url: strixPrUrl
    },
    {
      title: 'MLPerf Inference v6.1',
      status: 'Prepping',
      body: 'The same CUDA source submitted across GB10 and gfx1151. No numbers until MLCommons publishes.',
      cta: 'MLCommons',
      url: mlperfResultsUrl
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

// --- reach out ---------------------------------------------------------------
export const reachout = {
  label: '// 09 · reach out',
  title: 'Come work with us.',
  sub:
    'Building on Spark or Strix, bringing hardware to the table, or wanting to partner or talk business. We want to hear from you and we move fast.',
  cards: [
    {
      emoji: '💼',
      title: 'Business',
      body: 'Running Atlas in production or eyeing the Enterprise Edition. Tell us what you need and we will get you sorted.'
    },
    {
      emoji: '🤝',
      title: 'Partnerships',
      body: 'Frameworks, benchmarks, standards bodies. If it makes local AI better we are all in.'
    },
    {
      emoji: '🎁',
      title: 'Hardware',
      body: 'Got silicon you want Atlas running on. Send it our way and watch what we do with it.'
    }
  ],
  emails: ['thomas@avarok.net', 'debaterishaqui@gmail.com'],
  discordCta: 'Or pull up in Discord'
};

// --- footer ------------------------------------------------------------------
export const footer = {
  tagline: 'Pure Rust and CUDA inference for the machine on your desk.',
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
