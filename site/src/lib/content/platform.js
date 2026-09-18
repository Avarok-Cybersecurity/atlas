// =============================================================================
// /platform and its six subpages. Each page is one object with the same
// shape, so PlatformPage.svelte renders all of them and adding a seventh is a
// new entry here plus one route file that imports it.
// =============================================================================
import { routes, links, company } from './brand.js';

export const platformOverview = {
  eyebrow: 'Platform',
  title: 'One platform for the whole inference lifecycle, from the kernel to the invoice.',
  lede:
    'Avarok is three layers that share one request path. The engine makes the GPUs faster, the control plane keeps the fleet honest, and the economics layer turns what they report into numbers finance can sign.',
  primary: { text: 'Book a demo', href: routes.demo },
  secondary: { text: 'Watch the console tour', href: `${routes.home}#tour` },
  tiles: [
    { name: company.engine, tag: 'Inference layer', color: 'violet', body: 'Compiled per hardware, model and quantization. More tokens per GPU, verified on the box before it ships.', href: routes.engine },
    { name: company.control, tag: 'Governance and control plane', color: 'cyan', body: 'Signed recipes, canary rollouts, GPU aware routing, autoscaling, node repair and policy. Never on the request path.', href: routes.control },
    { name: company.economics, tag: 'Economics layer', color: 'green', body: 'Cost per workload, chargeback, stranded capacity and payback, against the baseline you ran before.', href: routes.economics },
    { name: 'Security', tag: 'Posture', color: 'gold', body: 'One signed binary, no interpreter in the request path, links designed for a hostile network, nothing leaves.', href: routes.security },
    { name: 'Deployment', tag: 'Surfaces', color: 'violet', body: 'Hosted with private connectivity, your cloud account, on premises or air gapped. Same binary, same recipes.', href: routes.deployment },
    { name: 'Hardware and models', tag: 'Compatibility', color: 'cyan', body: 'Verified silicon, targets in bring up, and every model recipe we ship, generated from the repository.', href: routes.hardware }
  ],
  surfaces: {
    title: 'Deployment surfaces',
    body: 'Private datacenters, AWS, Azure, GCP and neocloud GPU pools. Hosted with private connectivity where you want us to run it, bring your own cloud where you do not.',
    items: ['Private datacenter', 'Air gapped network', 'AWS', 'Azure', 'GCP', 'Neocloud GPU pools', 'Workstation and edge']
  },
  band: {
    title: 'Started on a single box. Built to run a fleet.',
    body: 'Avarok began as local inference on a DGX Spark. The same pinned stack scales to datacenter GPU fleets with the qualification record, routing and observability that enterprises need in production.'
  }
};

export const enginePage = {
  eyebrow: `Platform · ${company.engine}`,
  title: 'Not a generic runtime. A compiled inference stack.',
  lede:
    'Avarok Engine is a pure Rust and CUDA inference engine. It picks a workload, model plus quantization plus hardware target, compiles the kernel set for it, ships a tested build, and serves it behind the APIs your applications already speak.',
  who: 'Platform teams that own GPUs and are judged on tokens per GPU, and the developers who have to keep those GPUs busy at 128 concurrent agents rather than one chat window.',
  features: [
    { title: 'Selected workload', body: 'Model, quantization and hardware target chosen together. NVFP4 and FP8 with per target kernels, not one generic path with flags.' },
    { title: 'Optimized for the silicon', body: 'Hand tuned attention, MoE, Gated DeltaNet and Mamba class kernels per hardware target, register level work with no generic fallback on the hot path.' },
    { title: 'Speculative decoding', body: 'MTP draft heads and DFlash block diffusion, with a resolver that picks the verify width itself. The scheduler decides, not a flag.' },
    { title: 'Prefix cache and KV', body: 'Radix tree prefix caching so a shared system prompt is prefilled once, not per agent. Paged KV with tiered offload across host RAM, NVMe and RDMA peers.' },
    { title: 'Expert parallel across nodes', body: 'EP=2 across two DGX Sparks ships as recipes today. Three node topology is being wired up. The same binary, the same recipes.' },
    { title: 'Familiar APIs', body: 'OpenAI compatible chat and completions, the Anthropic Messages API and the Responses API from one binary, so agents, SDKs and gateways point at it unchanged.' }
  ],
  stats: [
    { value: '{ratio}', label: 'the matched vLLM configuration at C={c}, same GB10' },
    { value: '{atlasTop}', unit: 'tok/s', label: 'aggregate at C={c}, {checkpoint}' },
    { value: '~75 MB', label: 'one binary, no Python, no PyTorch, no runtime compilation' },
    { value: '{recipes}', label: 'model recipes, every one verified before it is listed' }
  ],
  quote: {
    text: 'Night and day compared to the 10 minute torch.compile cycle. Startup in about 15 seconds and it just stays coherent in an agentic loop.',
    who: 'ronald_15496, Discord',
    href: links.discord
  },
  faqTag: 'hardware',
  cta: { text: 'Install the Community Edition', href: routes.openSource },
  cta2: { text: 'Read the ladder', href: routes.benchmarks }
};

export const controlPage = {
  eyebrow: `Platform · ${company.control}`,
  title: 'Deploys, governs and repairs the fleet. Never on the inference path.',
  lede:
    'Avarok Control is the governance and control plane. It turns a signed recipe into a running, routed, observed workload on the GPUs you own, then keeps it that way without a human in the loop. A control plane outage never stops inference that is already running.',
  who: 'Infrastructure and platform engineering teams responsible for a GPU fleet, and the security and compliance owners who have to sign off on what runs on it.',
  features: [
    { title: 'Signed recipes and release channels', body: 'Every model is a recipe. Every recipe is signed. Edge, beta, stable and LTS channels, promoted by digest and never rebuilt between environments.' },
    { title: 'Canary rollouts that roll back on telemetry', body: 'Five percent, twenty five, one hundred. A canary that regresses TTFT, throughput or error rate fails the release gate and rolls back on its own.' },
    { title: 'GPU aware routing', body: 'Workers are chosen on projected prefill and decode cost, queue depth, VRAM pressure and KV cache reuse. A request stays on the worker that started it.' },
    { title: 'Autoscaling and node repair', body: 'Pods scale on queue wait, TTFT and active sequences, nodes scale on demand, and a failed GPU is cordoned, drained and replaced, not patched by hand.' },
    { title: 'Fleet policy', body: 'Data residency, model allowlists, redaction and tenant quotas as policy objects, enforced at the router and recorded in the audit log.' },
    { title: 'Your account, your metal, or air gapped', body: 'Runs in your AWS, Azure or GCP account through Terraform or Helm, on premises, or on an isolated network. Regulated buyers can use the customer pull GitOps mode where Avarok never holds credentials.' }
  ],
  stats: [
    { value: 'Live', label: 'the LAN fleet manager ships today as atlasctl, in early access', href: routes.controlPlane },
    { value: '4', label: 'release channels, promoted by digest' },
    { value: '0', label: 'inference requests that pass through the control plane' },
    { value: '5 → 25 → 100', label: 'percent, the canary ladder every release climbs' }
  ],
  status: 'The single operator fleet manager is live at /control and in active development. Multi tenant control plane, RBAC and SLA enforcement are the next milestones and this page will say so until they ship.',
  faqTag: 'deploy',
  cta: { text: 'Open the live control plane', href: routes.controlPlane },
  cta2: { text: 'Book a demo', href: routes.demo }
};

export const economicsPage = {
  eyebrow: `Platform · ${company.economics}`,
  title: 'Every GPU. Every workload. Every dollar.',
  lede:
    'Avarok Economics turns runtime telemetry into enterprise economic accountability across private and cloud GPU fleets. The industry measures inference in tokens per second. Enterprises pay for it in dollars per workload. This is the layer connecting the two.',
  who: 'CFOs, FinOps and platform leaders who have to explain an inference bill, and the operators who want the budget conversation to be about numbers they can stand behind.',
  columns: {
    left: { title: 'Operator telemetry', items: ['TTFT and TPOT', 'Tokens per second', 'Queue depth', 'KV cache hit rate', 'GPU utilization'] },
    middle: { title: 'Correlation layer', body: 'Workload × model × runtime × configuration × GPU × cluster. Tie each unit of work to the infrastructure and configuration that produced it.' },
    right: { title: 'Economic accountability', items: ['Dollars per million tokens', 'Dollars per successful workload at SLO', 'Productive GPU hours', 'Stranded capacity', 'Cost by model, cluster and business unit', 'Savings against the production baseline'] }
  },
  features: [
    { title: 'Baseline first', body: 'Before a single request moves, Economics records what each cluster costs per workload today. Every later number is a delta against that receipt, not a vendor estimate.' },
    { title: 'Chargeback by business unit', body: 'Cost attributed to the team, the application and the model that consumed it. Exportable to your FinOps tooling as CSV, Prometheus or OpenTelemetry.' },
    { title: 'Stranded capacity', body: 'GPU hours that were paid for and produced nothing, by cluster and by hour, so idle capacity becomes a scheduling decision instead of a surprise.' },
    { title: 'Power and cooling', body: 'Watts, PUE and tariff per site, so cost per million tokens includes the electricity and the savings include the kilowatt hours you no longer burn.' },
    { title: 'The payback clock', body: 'License cost against measured savings, recomputed monthly from production data. The number the decision maker walks to the CFO with.' },
    { title: 'Provenance on every figure', body: 'Every cost line traces to the signed recipe, kernel build and gate record that produced the tokens. Finance can audit it, not just read it.' }
  ],
  stats: [
    { value: '{payback}', label: 'modeled payback on a 256 GPU fleet at a 1.20x uplift, defaults shown on the pricing page' },
    { value: '{apiSavings}', label: 'modeled savings replacing a metered API with owned boxes at measured throughput' },
    { value: '6', label: 'ledger dimensions per unit of work' },
    { value: '1', label: 'baseline per cluster, recorded before traffic moves' }
  ],
  faqTag: 'pricing',
  cta: { text: 'Run the payback model', href: `${routes.pricing}#payback` },
  cta2: { text: 'Book a demo', href: routes.demo }
};

export const securityPage = {
  eyebrow: 'Platform · Security',
  title: 'One signed binary. No interpreter in the request path. Nothing leaves.',
  lede:
    'Security is a property of the request path, not a feature beside it. Avarok Engine is one binary of about 75 MB with no Python, no PyTorch and no runtime compilation. What runs is what was signed, and what it touches stays inside your perimeter.',
  who: 'CISOs, security architects and compliance owners who have to approve an inference stack for regulated data, and the engineers who have to keep it approved.',
  features: [
    { title: 'A request path you can read', body: 'HTTP to kernel dispatch in Rust. No Python environment to resolve, no two hundred transitive dependencies, no runtime download. Cargo deny audits every dependency on every pull request.' },
    { title: 'Signed artifacts, promoted by digest', body: 'Recipes, models and kernels arrive as signed artifacts. Kernel targets are content hashed down to their transitive include closure, so a build proves what it was compiled from.' },
    { title: 'Nothing leaves your perimeter', body: 'Prompts, weights, outputs and telemetry stay on hardware you own, in your cloud account, or on an air gapped network. Prompt and output logging is configurable and off by default in enterprise deployments.' },
    { title: 'Links designed for a hostile network', body: 'Node to node transport assumes every link is untrusted, so the mesh spans racks, sites and edges you already have with no private backbone and no VPN to babysit.' },
    { title: 'The control plane is out of band', body: 'It manages configuration, licensing, versions and aggregate telemetry. It is never on the synchronous inference path, and an outage never stops running workers.' },
    { title: 'Threat model on the page', body: 'CUDA kernel memory safety, API input validation, weight loading and unsafe FFI are the named review targets. The security policy says what is automated and what is human review, and claims nothing it does not run.' }
  ],
  stats: [
    { value: '0', label: 'Python or PyTorch in the request path' },
    { value: '~75 MB', label: 'the whole signed binary' },
    { value: '48 h', label: 'acknowledgement window for a reported vulnerability' },
    { value: 'AGPL + commercial', label: 'dual licensed, so legal knows exactly what it signed' }
  ],
  disclosure: { text: 'Read the security policy and disclosure process', href: links.securityPolicy },
  faqTag: 'security',
  cta: { text: 'Visit the trust center', href: routes.trust },
  cta2: { text: 'Book a security review', href: routes.demo }
};

export const deploymentPage = {
  eyebrow: 'Platform · Deployment',
  title: 'Hosted, your cloud account, on premises or air gapped. Same binary, same recipes.',
  lede:
    'Every deployment model runs the same signed artifacts through the same control plane. The difference is where the GPUs are and who holds the keys, and that is a policy decision you make once.',
  who: 'Infrastructure leads choosing where inference should live, and procurement teams that need the options on one page.',
  models: [
    { name: 'Hosted with private connectivity', body: 'Avarok operates the GPUs. Your applications reach them over private connectivity from your VPC with no public endpoint. Lowest friction, strongest protection for the kernel binaries.', fit: 'Fast start, no GPU estate of your own' },
    { name: 'Dedicated', body: 'A dedicated GPU pool, cluster or VPC operated by Avarok for one customer. Promote from shared to dedicated without changing the API.', fit: 'Enterprise isolation without an ops team' },
    { name: 'Bring your own cloud', body: 'The engine and router deploy into your AWS, Azure or GCP account on your GPU node pools through Terraform or Helm. No inference request leaves your account.', fit: 'Regulated data, existing cloud commitments' },
    { name: 'On premises', body: 'Your datacenter, your racks, your network. One binary per node, signed recipes, the control plane inside your perimeter.', fit: 'Owned GPU fleets, sovereign requirements' },
    { name: 'Air gapped', body: 'Signed artifacts installed from local media. Telemetry stays inside the network and exports on your schedule, or never.', fit: 'Defense, classified and isolated networks' },
    { name: 'Workstation and edge', body: 'A DGX Spark or Strix Halo class box under a desk or in a branch office, licensed per box, managed by the same control plane.', fit: 'SMB, branch offices, field deployments' }
  ],
  onboarding: {
    title: 'What onboarding looks like',
    steps: [
      'Choose hosted, your cloud, on premises or air gapped',
      'Select region, GPU profile, models and capacity limits',
      'Avarok validates the environment and generates the deployment',
      'GPUs provision, the runtime selects the kernels for the silicon it finds',
      'Models warm, health checks pass, you receive an endpoint',
      'Routing, observation, repair, scaling, canary and rollback run from then on'
    ],
    foot: 'You never have to understand CUDA compute capability, driver compatibility, Kubernetes GPU plugins or kernel rollout. Those are our problems.'
  },
  isolation: {
    title: 'Isolation is a setting',
    rows: [
      ['Shared', 'Namespace and logical tenant'],
      ['Enhanced', 'Dedicated GPU node pool'],
      ['Enterprise', 'Dedicated cluster'],
      ['Regulated', 'Dedicated VPC and cluster'],
      ['Bring your own cloud', 'Your account, your keys']
    ]
  },
  faqTag: 'deploy',
  cta: { text: 'Talk through your deployment', href: routes.demo },
  cta2: { text: 'Read the deployment guide', href: links.guide, external: true }
};

export const hardwarePage = {
  eyebrow: 'Platform · Hardware and models',
  title: 'Verified silicon, targets in bring up, and every recipe we ship.',
  lede:
    'The same Rust and CUDA source runs on NVIDIA and AMD without a second kernel tree. This page names what is verified, what is in bring up, and every model recipe, generated from the repository at build time.',
  who: 'Engineers checking whether their hardware and their models are covered before a pilot.',
  verified: [
    { name: 'NVIDIA DGX Spark', chip: 'GB10 · Blackwell SM121', status: 'Verified', body: 'One multi model binary serves a full matrix of hand tuned targets on a single GB10. NVFP4 and FP8, MTP speculative decoding, EP=2 across two Sparks. Every target passes the serve matrix before an image is cut.', href: links.guide, cta: 'Deployment guide' },
    { name: 'AMD Strix Halo', chip: 'gfx1151 · RDNA 3.5', status: 'MLPerf submitted', body: 'One codebase, both vendors. The CUDA kernels compile straight for gfx1151 through SCALE by Spectral Compute. AMD provided the Strix Halo desktop the MLPerf Inference v6.1 numbers were submitted on.', href: links.scale, cta: 'About SCALE' }
  ],
  bringup: [
    { name: 'NVIDIA H100 and H200', chip: 'Hopper · SM90', status: 'Bring up', body: 'Hopper owned decode and prefill kernels with published receipts in the changelog, bit identical to the reference on the production shapes. Not yet a verified target, and this page will say so until it is.' },
    { name: 'NVIDIA B200 and GB200', chip: 'Blackwell · SM100', status: 'Campaign', body: 'Datacenter Blackwell is the next objective after Hopper, built around the same principle the hardware demonstrates, inference economics improve when the software is designed for the accelerator.' },
    { name: 'AMD Radeon AI PRO R9700', chip: 'gfx1201 · RDNA 4', status: 'In review', body: 'A SCALE target serving Qwen3.8-27B in an open pull request. The compatibility list grows in the open.' },
    { name: 'Intel Arc Pro B70', chip: 'Battlemage', status: 'In talks', body: 'Active conversations with Intel. Nothing is signed and this card will say so until it is.' }
  ],
  modelsTitle: 'Every model here has a recipe.',
  modelsLede: 'Pick a vendor, then a family. Every card maps to one recipe in atlas-recipes, so the site cannot list a model it does not ship.',
  faqTag: 'hardware',
  cta: { text: 'Bring us your hardware', href: routes.contact },
  cta2: { text: 'Every recipe on GitHub', href: links.recipes, external: true }
};
