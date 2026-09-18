// =============================================================================
// The front page. Read top to bottom, this is the pitch: what we sell, who
// built it, the problem, the platform, the proof, the console, the three
// differences, the delivery path, the questions, the ask.
//
// Placeholders in braces ({ratio}, {c}, {won}) are filled at render time from
// the generated ladder and gate data by src/lib/content/live.js. Nothing
// numeric about performance is typed here.
// =============================================================================
import { routes, links, company } from './brand.js';

export const announcement = {
  tag: 'New',
  text: '{engine} serves {ratio} the matched vLLM configuration at C={c} on the same GB10, with every rung published.',
  cta: 'See the ladder',
  href: routes.benchmarks
};

export const hero = {
  // The product statement, asked for in so many words in the brief. It sits
  // above the headline and on the reel's title card (media-brief/REEL.md).
  kicker: 'Inference economics, reimagined.',
  pillars: ['Speed', 'Security', 'Governance'],
  title: ['Faster inference. Stronger governance.', 'A fraction of what you pay today.'],
  lede:
    'Avarok is the inference economics platform for the GPUs you already own. It runs your models faster on the same silicon, keeps every prompt inside your perimeter, and shows your CFO what each workload costs.',
  claim: 'Modeled at 70% or less of your current inference spend when the silicon is yours.',
  claimCta: 'Run the model',
  claimHref: `${routes.pricing}#payback`,
  primary: { text: 'Book a demo', href: routes.demo },
  secondary: { text: 'See how it works', href: routes.why },
  videoCaption: `${company.console}. Demo data, recorded from the product mockup.`,
  film: { text: 'Watch the one minute film', href: `${routes.demo}#film` },
  videoAlt: 'The Avarok Console. A model selector, a compute grid selector, a streaming answer, and the cost per request beside it.'
};

export const logoWall = {
  label: 'Built by people who have stood up operations for',
  // `file` names an SVG under static/logos. `kind: text` renders a set
  // wordmark instead, which is how the two government entities appear:
  // their official seals are restricted insignia and stay off the page.
  items: [
    { name: 'U.S. Cyber Command', kind: 'text' },
    { name: 'Naval Special Warfare', kind: 'text' },
    { name: 'Kraken', file: 'kraken' },
    { name: 'Beyond Gravity', file: 'beyondgravity' },
    { name: 'Synopsys', file: 'synopsys' },
    { name: 'Anaconda', file: 'anaconda' },
    { name: 'Google', file: 'google' }
  ],
  note: 'Prior roles of the founding team and core contributors. Listed for background, not as customers or endorsements.',
  programsLabel: 'Programs and partners',
  programs: [
    { name: 'NVIDIA Inception', src: '/nvidia-inception.webp', href: links.inception, blurb: 'Program member. DGX Spark hardware provided.' },
    { name: 'AMD', kind: 'text', blurb: 'Strix Halo hardware provided. MLPerf submitted on it.' },
    { name: 'MLCommons', kind: 'text', href: links.mlcommons, blurb: 'Named contributor to the MLPerf edge agentic benchmark.' },
    { name: 'Hugging Face', kind: 'text', href: links.transformersPr, blurb: 'Fused Qwen GDN kernel merged into Transformers.' },
    { name: 'Qwen', kind: 'text', href: links.qwenAmbassador, blurb: 'Dev Ambassadors. A recipe for every release.' },
    { name: 'SCALE by Spectral Compute', kind: 'text', href: links.scale, blurb: 'One CUDA source, NVIDIA and AMD.' }
  ]
};

export const problem = {
  eyebrow: 'The problem',
  title: 'Your GPUs report tokens per second. Your CFO pays in dollars per workload.',
  lede:
    'Every inference engine gives you operator statistics. Every FinOps tool gives you a bill. Nothing connects the two, so the fastest engine in the rack still cannot say what a workload cost, which cluster stranded capacity, or whether last quarter’s upgrade paid for itself.',
  left: {
    title: 'What the infrastructure sees',
    items: ['TTFT and TPOT', 'Tokens per second', 'Queue depth', 'KV cache hit rate', 'GPU utilization', 'Kernel and driver versions']
  },
  right: {
    title: 'What the enterprise pays for',
    items: [
      'Dollars per million tokens',
      'Dollars per successful workload at SLO',
      'Productive GPU hours',
      'Stranded capacity',
      'Cost by model, cluster and business unit',
      'Savings against the production baseline'
    ]
  },
  bridge: {
    title: 'Avarok is the correlation layer between the two.',
    body:
      'Every unit of work is tied to the workload, model, runtime, configuration, GPU and cluster that produced it. An operator number becomes a finance number without a spreadsheet in between.'
  }
};

export const solution = {
  eyebrow: 'The platform',
  title: 'One platform. Three layers. Every GPU, every workload, every dollar.',
  layers: [
    {
      key: 'engine',
      name: company.engine,
      tag: 'Inference layer',
      title: 'Runs your models faster on the same silicon.',
      body:
        'A compiled inference stack in Rust and CUDA. Hand tuned kernels per hardware, model and quantization, speculative decoding, prefix caching and expert parallelism across nodes. OpenAI and Anthropic compatible APIs from one 75 MB binary.',
      href: routes.engine,
      color: 'violet'
    },
    {
      key: 'control',
      name: company.control,
      tag: 'Governance and control plane',
      title: 'Deploys, governs and repairs the fleet.',
      body:
        'Signed recipes, release channels, canary rollouts that roll back on telemetry, GPU aware routing, autoscaling and node repair. Runs in your cloud account, on your metal, or air gapped. Never on the inference path.',
      href: routes.control,
      color: 'cyan'
    },
    {
      key: 'economics',
      name: company.economics,
      tag: 'Economics layer',
      title: 'Turns telemetry into accountability.',
      body:
        'Cost per million tokens, cost per successful workload, productive GPU hours and stranded capacity, by model, cluster and business unit, measured against the baseline you ran before.',
      href: routes.economics,
      color: 'green'
    }
  ],
  foot: {
    title: 'Started on a single box. Built to run a fleet.',
    body:
      'Avarok began as local inference on a DGX Spark. The same pinned stack scales to datacenter GPU fleets with the qualification record, routing and observability that enterprises need in production.'
  }
};

export const value = {
  eyebrow: 'The value',
  title: 'Faster inference. Stronger governance. Higher savings.',
  lede: 'Measured on the box we can put our hands on. Modeled where the box is yours, with the inputs on the page.',
  tiles: [
    { key: 'speed', label: 'Faster inference', value: '{ratio}', body: 'the throughput of the matched vLLM configuration at C={c}, same GB10, same checkpoint, same client.', href: routes.benchmarks, cta: 'Read the ladder' },
    { key: 'security', label: 'Stronger governance', value: '1 binary', body: 'about 75 MB of Rust and CUDA. No Python or PyTorch in the request path. Signed, gated and replayable.', href: routes.security, cta: 'See the posture' },
    { key: 'savings', label: 'Higher savings', value: '{payback}', body: 'to pay back the license on a 256 GPU fleet at a 1.20x uplift. Every month after is upside. Edit the inputs yourself.', href: `${routes.pricing}#payback`, cta: 'Open the model' }
  ]
};

export const proof = {
  eyebrow: 'Proof, not a pitch',
  title: 'Same GB10, same checkpoint, same client. Eight rungs, eight wins.',
  body:
    'We publish the concurrency ladder against the matched vLLM configuration from C=1 to C=128, with every rung we lost on the way to it. The margin is widest at the top, which is where fleets of tool calling agents actually run.',
  stats: [
    { value: '{atlasTop}', unit: 'tok/s', label: 'at C={c}, {checkpoint}' },
    { value: '{won}/{rungs}', unit: '', label: 'rungs won against matched vLLM + MTP' },
    { value: '{gatePass}', unit: '', label: 'concurrency gate records passing across the repo' }
  ],
  primary: { text: 'Read the full campaign log', href: links.ladderLog, external: true },
  secondary: { text: 'Every benchmark', href: routes.benchmarks }
};

export const tour = {
  eyebrow: 'The console',
  title: 'One console for every inference workflow.',
  lede: 'Recorded from the product mockup with demo data. The shipped product will differ. The workflow will not.',
  tabs: [
    {
      id: 'console',
      label: 'Ask',
      title: 'Pick a model. Pick a grid. Start.',
      body:
        'Choose the model, choose the compute grid it runs on, and start getting answers from an air gapped fleet with the tokens per second and the cost per request on the same screen.'
    },
    {
      id: 'queue',
      label: 'Queue',
      title: 'Bring up the links. Queue the project.',
      body:
        'Launch encrypted links to the grids you lease offsite, then queue a large project across all of them. It is sized, priced against your baseline, and placed only where its data class allows.'
    },
    {
      id: 'fleet',
      label: 'Fleet',
      title: 'Every node, every kernel, every rollout.',
      body:
        'Nodes report health, kernel versions and utilization. Canary a release to five percent, watch TTFT, roll back on a breach without waking anyone up.'
    },
    {
      id: 'economics',
      label: 'Economics',
      title: 'Dollars per workload, not tokens per second.',
      body:
        'Chargeback by business unit, productive GPU hours, stranded capacity and the payback clock, against the baseline you ran before Avarok.'
    },
    {
      id: 'governance',
      label: 'Governance',
      title: 'Policy, provenance and the audit trail.',
      body:
        'Data residency, model allowlists and redaction as policy. Every response traces to a signed recipe, a kernel build and a gate record.'
    }
  ],
  cta: { text: 'See it on your workload', href: routes.demo }
};

export const recognition = {
  eyebrow: 'Recognition',
  title: 'Receipts, not adjectives.',
  lede: 'Every card links to the primary source.',
  cards: [
    { org: 'MLCommons', date: 'July 2026', title: 'Named contributor to the MLPerf Inference v6.1 edge agentic benchmark', cta: 'Read the announcement', href: links.mlcommons },
    { org: 'Hugging Face', date: 'July 2026', title: 'Fused Qwen Gated DeltaNet kernel merged into Transformers', cta: 'See the pull request', href: links.transformersPr },
    { org: 'AMD', date: 'July 2026', title: 'Strix Halo hardware provided. MLPerf v6.1 submitted from the same CUDA source', cta: 'See the post', href: links.x },
    { org: 'NVIDIA Inception', date: 'Member', title: 'Program member. DGX Spark hardware provided for the GB10 bring up', cta: 'About the program', href: links.inception }
  ]
};

export const differences = {
  eyebrow: 'Why Avarok',
  title: 'Three differences you can test.',
  lede:
    'The market is loud. Every runtime claims speed and every dashboard claims visibility. Here is what is actually different, and the question to put to anyone else.',
  items: [
    {
      n: '01',
      pillar: 'Speed',
      title: 'Same silicon, more tokens.',
      body:
        'Avarok compiles a kernel set per hardware, model and quantization instead of shipping one generic path. On the published GB10 ladder it wins every rung against the matched vLLM configuration and keeps scaling from C=64 to C=128 while the baseline flattens.',
      question: 'Ask for the throughput curve at C=128 on your workload, not a single stream number on theirs.',
      href: routes.engine,
      color: 'violet'
    },
    {
      n: '02',
      pillar: 'Security',
      title: 'Nothing leaves your perimeter.',
      body:
        'One signed Rust binary with no Python or PyTorch in the request path. Prompts, weights and telemetry stay on hardware you own, in your cloud account, or on an air gapped network. The control plane never sits on the inference path.',
      question: 'Ask what is in the request path, and who audits the two hundred dependencies behind it.',
      href: routes.security,
      color: 'cyan'
    },
    {
      n: '03',
      pillar: 'Governance',
      title: 'Every token has a receipt.',
      body:
        'Every response traces to a signed recipe, a kernel build and a gate record. Every GPU hour is attributed to a model, a cluster and a business unit. Governance is a ledger finance can sign, not a dashboard operators tolerate.',
      question: 'Ask what a workload cost last Tuesday, by business unit. A tokens per second chart is not an answer.',
      href: routes.economics,
      color: 'green'
    }
  ]
};

export const chain = {
  eyebrow: 'Only Avarok',
  title: 'Each layer earns the next.',
  steps: [
    { title: 'The engine earns the deployment', body: 'Nobody installs a governance layer for its own sake. Avarok gets into the fleet by making the same GPUs produce more inference.' },
    { title: 'The deployment earns the telemetry', body: 'Once the engine owns the request path, every kernel, cache and queue decision is observed at the source instead of inferred from a proxy.' },
    { title: 'The telemetry earns the economics', body: 'With source data in hand, cost per workload, stranded capacity and payback stop being estimates and become a ledger.' }
  ],
  foot: 'One architectural decision, made at the start. Own the request path, then measure what it is worth. Everything on this page is downstream of that.'
};

export const deliveries = {
  eyebrow: 'Getting there',
  title: 'Complement or replace your serving stack, at the pace you choose.',
  lede:
    'Avarok runs beside vLLM, SGLang or llama.cpp on day one. Nothing is turned off and nothing is migrated until your own numbers say so.',
  steps: [
    {
      when: 'Week one',
      title: 'Side by side on your workload',
      body:
        'One binary per node, one signed recipe per model, next to what you run today. A side by side ladder against your current engine, on your hardware, inside 48 hours. The economics baseline starts recording the same day.'
    },
    {
      when: 'Months one to six',
      title: 'Traffic moves workload by workload',
      body:
        'Production traffic shifts one model family at a time. The control plane takes over rollout, canary, rollback and fleet policy. Economics reports every cluster against what it cost before, by business unit.'
    },
    {
      when: 'At renewal',
      title: 'Your call, on evidence',
      body:
        'Some customers keep the old engine for one model family. Others move off it entirely. You make that call having run both, on your own receipts rather than a vendor timeline.'
    }
  ]
};

export const cta = {
  eyebrow: 'Next step',
  title: 'Ready to get the most out of the datacenter you already paid for?',
  body: 'Book a working session. We run the ladder on your workload, on your hardware, and hand you the receipt.',
  primary: { text: 'Book a demo', href: routes.demo },
  secondary: { text: 'Run the payback model', href: `${routes.pricing}#payback` }
};

// The testimonials are real quotes from the community, already on the
// developer page. Names are handles, sources link to the thread. They are
// verbatim, old product name included: an attributed quote is never edited,
// not even to follow a rebrand. The note says why the name differs.
export const voices = {
  eyebrow: 'From the fleet',
  title: 'Operators running it on their own hardware.',
  note: "Quotes are verbatim. Atlas was the engine's name until the September 2026 rebrand to Avarok.",
  items: [
    { quote: 'Night and day compared to the 10 minute torch.compile cycle. Startup in about 15 seconds and it just stays coherent in an agentic loop.', who: 'ronald_15496', where: 'Discord, #general', href: links.discord },
    { quote: 'Testing Atlas on a DGX Spark in an agentic workflow for over an hour. Super impressed. Spark is actually awesome with Atlas.', who: 'PersonWhoThinks', where: 'r/LocalLLaMA', href: 'https://www.reddit.com/r/LocalLLaMA/comments/1rmvxo3/' },
    { quote: 'I had grown tired of the usual stack and was hoping for something like this. Really surprised and impressed. So glad I bought a Spark.', who: 'tetsuro59', where: 'Discord, #general', href: links.discord }
  ]
};
