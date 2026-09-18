// =============================================================================
// /pricing. Transparent on purpose: the anchors are public list prices, the
// Avarok numbers are the proposed sheet from the September 2026 deck, and the
// payback model shows its inputs. Everything marked PROPOSED is the team's to
// change here and nowhere else.
// =============================================================================
import { routes, contacts } from './brand.js';

export const pricingHero = {
  eyebrow: 'Pricing',
  title: 'Priced against productive GPU capacity, not seats.',
  lede:
    'Enterprise AI infrastructure software already prices per GPU per year. Avarok sits inside that range, ships with the control plane and the economics layer, and shows its payback on this page. Proposed list prices, September 2026.',
  stamp: 'Proposed sheet · September 2026'
};

export const tiers = [
  {
    key: 'community',
    name: 'Community Edition',
    price: '$0',
    per: 'AGPL-3.0, forever',
    blurb: 'The engine, every recipe, one install command. For developers, labs and anyone running open models on hardware they own.',
    includes: [
      'Avarok Engine, full source',
      'Every model recipe in atlas-recipes',
      'OpenAI, Anthropic and Responses APIs',
      'LAN fleet manager, early access',
      'Community support in Discord'
    ],
    cta: { text: 'Install in one command', href: routes.openSource },
    tone: 'plain'
  },
  {
    key: 'workstation',
    name: 'Workstation and edge',
    price: '$50',
    per: 'per box per month, billed annually',
    blurb: 'A DGX Spark or Strix Halo class box serving an office, a branch or a field team. Commercial license, signed update channel, managed from the console.',
    includes: [
      'Commercial license per box',
      'Signed stable and LTS channels',
      'Console access for every licensed box',
      'Email support, next business day',
      'Volume pricing from 25 boxes'
    ],
    cta: { text: 'Price a fleet of boxes', href: routes.contact },
    tone: 'plain',
    proposed: true
  },
  {
    key: 'enterprise',
    name: 'Enterprise',
    price: '$3,000',
    per: 'per GPU per year, list',
    blurb: 'The full platform for GPU fleets. Realized pricing at fleet scale runs $1,800 to $2,400 per GPU per year. Support and forward deployed engineering priced separately.',
    includes: [
      'Avarok Engine, commercial license',
      'Avarok Control, rollouts, routing, policy, repair',
      'Avarok Economics, chargeback and payback',
      'Named engineer and response SLA',
      'Hosted, your cloud, on premises or air gapped'
    ],
    cta: { text: 'Book a demo', href: routes.demo },
    tone: 'accent',
    proposed: true,
    featured: true
  },
  {
    key: 'pilot',
    name: 'Proof of value',
    price: 'Fixed fee',
    per: 'four weeks, credited on conversion',
    blurb: 'One model, one hardware target, one workload. A side by side ladder inside 48 hours and a receipt in dollars per workload at the end.',
    includes: [
      'Scoped success criteria, yours or ours',
      'Side by side against your current engine',
      'Economics baseline of the target cluster',
      'Forward deployed engineer for the four weeks',
      'Fee credited against the first year on conversion'
    ],
    cta: { text: 'Scope a pilot', href: routes.demo },
    tone: 'plain'
  }
];

export const anchors = {
  eyebrow: 'Market anchor',
  title: 'Where it sits.',
  body: 'Established enterprise AI infrastructure software already prices against GPU capacity. Avarok lists inside the range and includes the layers the others sell separately.',
  rows: [
    { name: 'Red Hat AI Inference Server', price: '~$2,500', per: 'per accelerator per year', note: 'Hardened vLLM, published list price' },
    { name: 'NVIDIA AI Enterprise', price: '$4,500', per: 'per GPU per year', note: 'Broad platform, OEM backed' },
    { name: 'Avarok Enterprise', price: '~$3,000', per: 'per GPU per year, list', note: 'Engine, control plane and economics, realized $1,800 to $2,400 at scale', accent: true }
  ],
  foot: 'Third party prices are public list prices at the time of writing and belong to their owners. Avarok prices are proposed and subject to contract.'
};

export const contractEconomics = {
  eyebrow: 'Illustrative contract economics',
  title: 'What a fleet costs to license.',
  body: 'At realized fleet scale pricing. Illustrative, not a forecast.',
  rows: [
    { gpus: '64 GPUs', acv: '≈ $175K' },
    { gpus: '256 GPUs', acv: '≈ $550K' },
    { gpus: '1,000 GPUs', acv: '≈ $2.0M' }
  ]
};

export const paybackCopy = {
  eyebrow: 'Payback',
  title: 'Find your payback period.',
  lede:
    'If a thing costs three thousand dollars and makes you a thousand a month, it pays for itself in three months, and everything after is upside. That is the number to walk to the CFO with. Two scenarios, every input editable, evidence class on every field.',
  fleet: {
    title: 'Get more out of the fleet you own',
    body: 'The uplift frees GPUs. Freed GPUs are deferred purchases or rentals plus the power they burned. The license is what the uplift costs.',
    note: 'Uplift defaults to 1.20x, below the measured ratio on the GB10 ladder at C=128, because a datacenter part is not a Spark until we publish the receipt.'
  },
  api: {
    title: 'Stop renting tokens',
    body: 'Take the API bill, count the tokens, and run them on boxes you own at measured throughput. This is where the 70% claim on the front page comes from.',
    note: 'Throughput defaults to the top rung of the published ladder. Blended API price defaults to a hosted rate for a 27B class open model, which you should replace with your own invoice.'
  },
  classes: {
    MEASURED: 'From ladder.generated.json, the published concurrency ladder',
    PROPOSED: 'A proposed list price from this page, the team can change it',
    USER: 'Yours to edit, the model recomputes as you type'
  },
  disclaimer: 'A model, not a quote. Savings depend on your workload, your utilization and the uplift measured on your hardware during the pilot.'
};

export const pricingFaqTag = 'pricing';

export const pricingCta = {
  eyebrow: 'Next step',
  title: 'Get the sheet, or get the receipt.',
  body: `Email ${contacts.sales} for the full price sheet, or book a working session and we run the ladder on your workload.`,
  primary: { text: 'Book a demo', href: routes.demo },
  secondary: { text: 'Email sales', href: `mailto:${contacts.sales}?subject=Avarok%20pricing%20sheet` }
};
