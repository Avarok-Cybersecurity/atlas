// =============================================================================
// /solutions. Industry pages exist to create familiarity: a banker clicks the
// bank page, a neocloud clicks the neocloud page, and each reads the same
// platform through their own constraints. Every page ends at the demo.
//
// Keep claims to what the platform does. No customer names appear here until
// there are customers, and the copy is written so it will not need to change
// when they arrive.
// =============================================================================
import { routes, industries } from './brand.js';

export const solutionsIndex = {
  eyebrow: 'Solutions',
  title: 'Built for the people who own the GPUs.',
  lede:
    'Three kinds of buyer open their wallets for inference economics. Operators who resell GPU time, enterprises that run AI on their own estate, and teams that need a box under a desk. Pick your industry.',
  deploymentsTitle: 'By deployment',
  deployments: [
    { name: 'Enterprise datacenter', body: 'Owned GPU fleet, existing serving stack, a CFO who wants the bill explained.', href: `${routes.solutions}/enterprise-datacenter` },
    { name: 'Neocloud and GPU provider', body: 'Tokens are cost of goods sold. More tokens per GPU is margin.', href: `${routes.solutions}/neoclouds` },
    { name: 'Air gapped and sovereign', body: 'Nothing leaves. Signed artifacts, local install, telemetry that stays home.', href: `${routes.solutions}/government-defense` },
    { name: 'Workstation and SMB', body: 'One box, one license, the same engine. Stop renting tokens.', href: `${routes.solutions}/smb-edge` }
  ]
};

const base = Object.fromEntries(industries.map((i) => [i.slug, i]));

export const solutions = {
  neoclouds: {
    ...base.neoclouds,
    eyebrow: 'Solutions · Neoclouds and GPU providers',
    title: 'You sell GPU time. Avarok makes every hour of it produce more tokens.',
    lede:
      'When tokens are cost of goods sold, throughput per GPU is margin. Avarok lifts the throughput of the fleet you already bought, on NVIDIA and AMD from one codebase, and gives your customers a per workload cost they can plan around.',
    fit: [
      'Tokens per watt is literally your value proposition to your own customers',
      'Mixed NVIDIA and AMD pools with one engine and no second kernel tree',
      'Multi tenant routing with per tenant quotas and isolation tiers',
      'A ladder against your current engine on your own hardware in week one'
    ],
    workloads: ['Open weight model serving at scale', 'Agentic workloads at high concurrency', 'Serverless endpoints with fast cold start from a single binary'],
    deployment: 'Bring your own cloud or on premises. Enterprise license per GPU, volume tiers as the fleet grows. Co marketing of the results is on the table.',
    proof: 'On the published GB10 ladder Avarok wins every rung against the matched vLLM configuration and keeps climbing from C=64 to C=128 while the baseline flattens. That headroom is capacity you sell.',
    cta: 'Run the ladder on your fleet'
  },
  'enterprise-datacenter': {
    ...base['enterprise-datacenter'],
    eyebrow: 'Solutions · Enterprise datacenters',
    title: 'You invested in the datacenter. Now get the most out of it.',
    lede:
      'Enterprises are buying accelerators faster than their serving software can use them. Avarok runs your models faster on the GPUs you own, governs what runs on them, and reports every business unit’s inference cost against what it was before.',
    fit: [
      'A GPU estate that is measured in utilization and blamed in budget meetings',
      'Several business units sharing one fleet with no chargeback',
      'A serving stack that takes a team to keep running',
      'A CFO who wants payback, not a percentage'
    ],
    workloads: ['Internal assistants and copilots', 'Document and knowledge workloads over private data', 'Agent fleets for engineering and operations'],
    deployment: 'On premises or in your cloud account. Complement your current engine on day one, move traffic workload by workload, decide at renewal on your own receipts.',
    proof: 'The payback model on the pricing page defaults to a 256 GPU fleet at a conservative 1.20x uplift and pays the license back in months. Change the inputs to your fleet.',
    cta: 'Model your fleet'
  },
  'financial-services': {
    ...base['financial-services'],
    eyebrow: 'Solutions · Financial services',
    title: 'Inference that stays inside the perimeter, with a ledger the auditors can read.',
    lede:
      'Banks, insurers and asset managers cannot send prompts to a black box and cannot deploy a serving stack nobody can audit. Avarok is one signed binary, running in your account or your datacenter, with provenance on every token.',
    fit: [
      'Model risk management wants to know exactly what produced an output',
      'Data residency and retention are policy, not configuration flags',
      'Procurement needs a license they can read and a vendor they can audit',
      'Cost allocation across desks, business lines and regions'
    ],
    workloads: ['Research and document intelligence over confidential data', 'Client service copilots with strict data handling', 'Risk, compliance and surveillance workloads'],
    deployment: 'Bring your own cloud with the customer pull GitOps mode, or on premises. Prompt and output logging off by default. SOC 2 readiness documentation as part of the engagement.',
    proof: 'Every response traces to a signed recipe, a kernel build and a gate record. Every GPU hour is attributed to a business unit. That is the audit trail, not a report generated after the fact.',
    cta: 'Book a security and compliance review'
  },
  healthcare: {
    ...base.healthcare,
    eyebrow: 'Solutions · Healthcare',
    title: 'Clinical grade inference on hardware the hospital owns.',
    lede:
      'Protected health information should not leave the building to be summarized. Avarok runs open models on GPUs inside your network, with residency, redaction and access as policy, and with the cost of every workload attributed to the department that ran it.',
    fit: [
      'PHI that must stay on premises or in a controlled cloud account',
      'Departments that share GPUs and need chargeback',
      'Clinical and administrative workloads with different risk profiles',
      'IT teams that cannot staff a Python serving stack'
    ],
    workloads: ['Clinical documentation and summarization', 'Prior authorization and coding assistance', 'Research over de identified records'],
    deployment: 'On premises or bring your own cloud. Air gapped where required. One binary per node, signed recipes, no runtime downloads.',
    proof: 'One signed binary with no interpreter in the request path means the security review is about the stack you can read, not two hundred dependencies you cannot.',
    cta: 'Talk to us about a pilot'
  },
  'government-defense': {
    ...base['government-defense'],
    eyebrow: 'Solutions · Government and defense',
    title: 'Air gapped by design. Signed by default. Nothing leaves.',
    lede:
      'Avarok was built by people who have stood up operations for cyber and special operations commands. It installs from local media, runs on an isolated network, and exports telemetry on your schedule or never.',
    fit: [
      'Classified and isolated networks with no route to the internet',
      'Supply chain requirements that reject runtime dependency resolution',
      'Sovereign and on premises deployments with domestic control',
      'Mission workloads that need high concurrency on small footprints'
    ],
    workloads: ['Analyst assistants over classified corpora', 'Agentic tooling at the tactical edge on Spark class hardware', 'Translation, triage and summarization at scale'],
    deployment: 'Air gapped or on premises. Signed artifacts, content hashed kernel builds, links designed for a hostile network, and a control plane that lives inside the enclave.',
    proof: 'The mesh assumes every link is untrusted, so it spans sites and edges you already have with no private backbone. The control plane is never on the inference path and never needs to phone home.',
    cta: 'Request the air gapped brief'
  },
  legal: {
    ...base.legal,
    eyebrow: 'Solutions · Legal and professional services',
    title: 'Privileged work product stays privileged.',
    lede:
      'Law firms and professional services firms are burning through annual AI budgets in a quarter, on per seat subscriptions that send client matter data to someone else’s cloud. A box under the desk running Avarok serves the whole office, with the matter data never leaving it.',
    fit: [
      'Client confidentiality that rules out hosted APIs',
      'Per seat subscription costs that scale with headcount',
      'Small IT teams that need set it and forget it',
      'Matter based cost allocation'
    ],
    workloads: ['Drafting, review and summarization', 'Research over the firm’s own precedent', 'Intake, triage and internal knowledge assistants'],
    deployment: 'Workstation and edge license per box, on DGX Spark or Strix Halo class hardware, managed by the control plane from the same console. Enterprise license for firms running a rack.',
    proof: 'The API replacement model on the pricing page defaults to a metered API bill replaced by owned boxes at measured throughput and lands at 70% or less of the old bill. Edit the inputs to your own.',
    cta: 'Price a box for the office'
  },
  hyperscalers: {
    ...base.hyperscalers,
    eyebrow: 'Solutions · Hyperscalers and cloud platforms',
    title: 'More effective capacity from the fleet you already bought.',
    lede:
      'Hyperscalers and cloud platforms are measured on tokens delivered per dollar of capital. Avarok is a specialized runtime that outperforms the generic serving baseline on the same silicon, with a vendor neutral kernel path across NVIDIA and AMD.',
    fit: [
      'Every percent of effective capacity is reportable',
      'Mixed silicon estates that need one engine and one qualification record',
      'Partner programs that want an optimized runtime to recommend',
      'Design partnerships gated on datacenter class receipts, which is how we prefer to start'
    ],
    workloads: ['Managed inference endpoints', 'Marketplace runtimes for GPU instances', 'Internal platform teams serving open weight models'],
    deployment: 'Design partnership first. Free or discounted Enterprise access in exchange for production telemetry and a co published benchmark, then a commercial license at fleet scale.',
    proof: 'Hopper decode and prefill kernels already carry published receipts in the changelog, bit identical to the reference on production shapes. Datacenter class verification is the next artifact and this page will say when it lands.',
    cta: 'Open a design partnership'
  },
  research: {
    ...base.research,
    eyebrow: 'Solutions · Research labs and AI safety',
    title: 'Run the new weights the week they drop, on hardware the lab controls.',
    lede:
      'Independent labs and safety teams pick between renting cloud GPUs and eating the price, access and data exposure, or building local and losing weeks to setup. Avarok makes open models run fast on machines researchers own, with a written setup path and results you can reproduce.',
    fit: [
      'Evaluation, red team and research workloads that must stay private',
      'A workstation or small cluster instead of a cloud account',
      'Reproducibility as a requirement, not a preference',
      'Grant budgets that cannot absorb metered token pricing'
    ],
    workloads: ['Evaluation and red teaming', 'Reproducing published results', 'Long running research jobs on owned GPUs'],
    deployment: 'The open source engine on owned hardware, with the Enterprise control plane when the lab grows into a cluster. We publish how each setup was configured and what it measured.',
    proof: 'Every number on this site is generated from a record in the repository and comes with a reproduce command. The verification walkthrough shows every step.',
    cta: 'Set up a lab deployment'
  },
  'smb-edge': {
    ...base['smb-edge'],
    eyebrow: 'Solutions · SMB and edge',
    title: 'One box. One license. Stop renting tokens.',
    lede:
      'A DGX Spark or Strix Halo class box running Avarok serves an entire small business, branch or field team from under a desk. The same engine as the datacenter, licensed per box, managed from the same console.',
    fit: [
      'A metered API bill that grew past the cost of owning the hardware',
      'Data that should not leave the office',
      'No engineer to babysit a serving stack',
      'A branch or field site with unreliable connectivity'
    ],
    workloads: ['Office assistants over company documents', 'Customer service and intake', 'Field and branch deployments that must work offline'],
    deployment: 'Workstation and edge license per box, installed in one command, updated from a signed channel. Add boxes and the console sees them.',
    proof: 'The workstation scenario on the pricing page replaces a metered API bill with owned boxes at measured throughput and lands at 70% or less of the old bill, with payback in months.',
    cta: 'Price a box'
  }
};

export const solutionCta = {
  eyebrow: 'Next step',
  title: 'See it against your own workload.',
  body: 'A side by side ladder on your hardware in week one. Your models, your criteria, your receipt.',
  primary: { text: 'Book a demo', href: routes.demoForm },
  secondary: { text: 'See pricing', href: routes.pricing }
};
