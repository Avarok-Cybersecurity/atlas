// =============================================================================
// Frequently asked questions. One list, tagged, so the home page can show the
// buyer set, /why-avarok the deployment set, /pricing the commercial set, and
// the FAQPage structured data in the layout can restate exactly what is
// visible on each page and nothing more.
// =============================================================================
import { routes, links } from './brand.js';

export const faq = [
  {
    tags: ['home', 'why'],
    q: 'What is Avarok?',
    a: 'Avarok is an inference economics platform for GPUs you own or rent by the hour. Avarok Engine runs open models faster on the same silicon, Avarok Control deploys and governs the fleet, and Avarok Economics turns the telemetry into cost per workload, chargeback and payback. The engine is open source under AGPL-3.0. The platform is licensed per GPU.'
  },
  {
    tags: ['home', 'why', 'deploy'],
    q: 'Do I have to replace vLLM, SGLang or llama.cpp to use it?',
    a: 'No. Avarok deploys beside your current engine and takes traffic one model family at a time. Many teams keep the old engine for a family we do not ship a recipe for yet. You decide what moves, on your own side by side numbers.'
  },
  {
    tags: ['home', 'why', 'deploy'],
    q: 'How long does a deployment take?',
    a: 'One binary per node and one signed recipe per model. A single box runs in minutes from one install command. A fleet pilot has a side by side ladder against your current engine in week one. Production cutover is workload by workload over the following weeks, at your pace.'
  },
  {
    tags: ['home', 'why', 'security'],
    q: 'Who owns the data?',
    a: 'You do. Prompts, weights, outputs and telemetry stay on hardware you own, in your cloud account, or on an air gapped network. The control plane manages configuration, licensing, versions and aggregate metrics, and it never sits on the request path. In a bring your own cloud deployment no inference request leaves your account.'
  },
  {
    tags: ['home', 'hardware'],
    q: 'What hardware does it run on?',
    a: 'NVIDIA DGX Spark (GB10) is verified today, and AMD Strix Halo (gfx1151) runs the same CUDA source compiled through SCALE, with both submitted to MLPerf Inference v6.1. Hopper and Blackwell datacenter targets are in active bring up with receipts in the changelog. Expert parallelism across two nodes ships as recipes and a three node topology is being wired up.'
  },
  {
    tags: ['home', 'security', 'deploy'],
    q: 'Can it run air gapped?',
    a: 'Yes. The engine is one binary with no runtime download and no Python environment to resolve. Recipes, models and kernels are delivered as signed artifacts and installed from local media. Telemetry can stay entirely inside the network and export on your schedule, or never.'
  },
  {
    tags: ['home', 'hardware'],
    q: 'Which models can I run?',
    a: 'Every model on this site maps to a recipe in the atlas-recipes repository, which is the single source of truth, so the site cannot list a model without one. Qwen leads with the most recipes, alongside Gemma, Nemotron, Mistral, MiniMax and DeepSeek. Bring your own weights and we scope the bring up.'
  },
  {
    tags: ['home', 'why', 'benchmarks'],
    q: 'What does verified mean on this site?',
    a: 'An image ships only after the serve matrix passes. Every model boots, stays coherent under greedy determinism with no token leakage and reliable tool calls, and holds throughput within ten percent of its committed baseline. A release that ships slower than its baseline fails the gate. Every number on the benchmarks page is generated from a record in the repository.'
  },
  {
    tags: ['home', 'pricing'],
    q: 'How is it priced?',
    a: 'Per GPU per year for the Enterprise Edition, with volume tiers as the fleet grows, and a per box license for workstation and edge deployments. Support and forward deployed engineering are priced separately. The Community Edition is free under AGPL-3.0. The pricing page lists the proposed sheet and a payback model with editable inputs.'
  },
  {
    tags: ['pricing', 'why'],
    q: 'How do you measure savings?',
    a: 'Against your own baseline. Economics records what each cluster cost per workload before Avarok takes traffic, then reports the delta as traffic moves. The pilot ends with a receipt in dollars per million tokens and dollars per successful workload, not a slide.'
  },
  {
    tags: ['pricing'],
    q: 'What is the payback period?',
    a: 'It depends on your fleet, your utilization and the uplift we measure on your workload. The model on the pricing page computes it from inputs you control. Every month after payback is upside, which is why we talk about payback rather than a percentage.'
  },
  {
    tags: ['deploy', 'security'],
    q: 'Does it run in my cloud account?',
    a: 'Yes. Bring your own cloud deploys the engine and router into your AWS, Azure or GCP account, on your GPU node pools, through Terraform or Helm. The control plane sees configuration, licensing, versions and aggregate telemetry, and nothing else. Regulated buyers can run the customer pull GitOps mode, where Avarok never holds credentials to your account.'
  },
  {
    tags: ['security', 'pricing'],
    q: 'What about SOC 2 and compliance?',
    a: 'The architecture is built for regulated buyers, and SOC 2 readiness documentation, model risk documentation and pinned recipe governance packs are part of the first SLA engagements. Ask for the current state of the audit program when you book. We will tell you exactly where it is.'
  },
  {
    tags: ['home', 'benchmarks'],
    q: 'Why does concurrency matter more than single stream speed?',
    a: 'Because agentic systems do not send one request at a time. A fleet of tool calling agents sharing a context bus arrives as many concurrent streams, so an engine is judged where the requests pile up. On the published ladder Avarok keeps gaining throughput from C=64 to C=128 while the matched vLLM configuration does not, and an engine that flattens under load caps how many agents a box can run.'
  },
  {
    tags: ['deploy'],
    q: 'Which APIs does it expose?',
    a: 'OpenAI compatible chat and completions, the Anthropic Messages API and the Responses API, from the same binary, so existing SDKs, agents and gateways point at Avarok without code changes.'
  },
  {
    tags: ['pricing', 'deploy'],
    q: 'What support comes with it?',
    a: 'Community support in Discord for the Community Edition. Enterprise includes a named engineer, a response SLA and a shared channel. Forward deployed engineering for the pilot and the cutover is scoped per engagement and credited against the first year on conversion.'
  }
];

export const faqFor = (tag) => faq.filter((f) => f.tags.includes(tag));

export const faqSection = {
  eyebrow: 'Questions',
  title: 'The questions we actually get asked.',
  lede: 'Short answers. Each one is backed by something on this site or in the repository.',
  more: { text: 'Ask the rest in a working session', href: routes.demo },
  repo: { text: 'deployment guide', href: links.guide }
};
