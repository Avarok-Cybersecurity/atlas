// =============================================================================
// Avarok marketing site: brand, links, navigation, footer.
//
// This directory is the single source of truth for every word on the
// marketing routes (everything except /engine, /control and /diligence, which
// still read src/lib/data.js). Components under src/lib/components/avarok are
// presentation only. Change copy here, never in markup.
//
// VOICE: plain, confident, buyer to buyer. Commas and periods. No em dashes,
// no semicolons, no exclamation marks. Numbers that describe performance are
// never typed here, they are computed from the generated JSON at build time
// (see src/lib/content/live.js). Numbers that describe a model or a price are
// labeled as modeled or proposed wherever they render.
// =============================================================================

// The domain is unchanged by the rebrand (see PR #1101). When DNS moves, this
// is the one constant to change. Everything absolute is built from it.
export const SITE = 'https://atlascybernetics.ai';

export const company = {
  name: 'Avarok',
  legal: 'Atlas Cybernetics Corp.',
  // The product family. One masterbrand, descriptive product names beneath it,
  // the way Salesforce and Microsoft do it, so every layer sells the others.
  engine: 'Avarok Engine',
  control: 'Avarok Control',
  economics: 'Avarok Economics',
  console: 'Avarok Console',
  category: 'The inference economics platform',
  tagline: 'Same silicon. Smarter inference. Stronger scalability.',
  short: 'Avarok is the inference economics platform for GPUs you already own.',
  // Only what is sourced. The engine started in January 2026. Where the company
  // is based and how it works are the company's to state, not the site's to guess.
  founded: 'Started in 2026.',
  x: '@AtlasInferenceX'
};

// For a local demo of both apps. The live blog deploys from main, so until this
// branch merges it shows main's header. Run the blog beside the site and point
// the site's Blog links at it:
//   (in blog/)  bun run dev -- --port 5174
//   (in site/)  VITE_BLOG_ORIGIN=http://127.0.0.1:5174 bun x --bun vite dev
// Unset, which is every real build, the links go to the live blog.
const localBlog = (import.meta.env ?? {}).VITE_BLOG_ORIGIN;

export const links = {
  github: 'https://github.com/Avarok-Cybersecurity/atlas',
  recipes: 'https://github.com/Avarok-Cybersecurity/atlas-recipes',
  discord: 'https://discord.gg/RQcGakU2jW',
  blog: localBlog || 'https://blog.atlascybernetics.ai',
  docs: 'https://docs.atlascybernetics.ai',
  x: 'https://x.com/AtlasInferenceX',
  guide: 'https://github.com/Avarok-Cybersecurity/atlas/blob/main/docs/GB10_DEPLOYMENT_GUIDE.md',
  ladderLog: 'https://github.com/Avarok-Cybersecurity/atlas/blob/main/bench/ladder38/RESULTS.md',
  transformersPr: 'https://github.com/huggingface/transformers/pull/46423',
  mlcommons: 'https://mlcommons.org/2026/07/mlperf-inference-v61-edge-agentic/',
  inception: 'https://www.nvidia.com/en-us/startups/',
  scale: 'https://docs.scale-lang.com/stable/',
  qwenAmbassador: 'https://qwen.ai/ambassador',
  llamaCppPr: 'https://github.com/ggml-org/llama.cpp/pull/18680',
  securityPolicy: 'https://github.com/Avarok-Cybersecurity/atlas/blob/main/SECURITY.md',
  license: 'https://github.com/Avarok-Cybersecurity/atlas/blob/main/LICENSE',
  contributing: 'https://github.com/Avarok-Cybersecurity/atlas/blob/main/CONTRIBUTING.md',
  changelog: 'https://github.com/Avarok-Cybersecurity/atlas/blob/main/CHANGELOG.md',
  issues: 'https://github.com/Avarok-Cybersecurity/atlas/issues',
  sequoiaPatel: 'https://sequoiacap.com/podcast/dylan-patel-of-semianalysis-why-hardware-software-co-design-is-ais-real-100x'
};

// Where the forms post. Empty means each form drafts an email in the visitor's own
// mail app, which loses every visitor who does not press send. The endpoint is
// the Worker in deploy/cloudflare/forms-worker: deploy it (its README has the
// ten minutes of setup), paste its /lead address here, then `bun run guide`.
// Any endpoint that accepts a JSON POST works, a hosted form service included.
export const formEndpoint = '';

// Who answers what. One address per job, and the job is the key, so a change of
// person is a change of one line. Company mail is @atlascybernetics.ai.
//   sales          Eric     sales and pilots. The demo form and the waitlist land here.
//   business       Kyle     business and design partners, investors, press.
//   technical      Thomas   engineering, open source, careers.
//   operations     Peter    operations, and the second door on the security card.
//   collaboration  Tom      public collaboration: silicon, frameworks, benchmarks.
// The security address is the exception on purpose: it is the one SECURITY.md
// publishes, and a vulnerability report must never go to a mailbox nobody set
// up. Change both together.
export const contacts = {
  sales: 'eric@atlascybernetics.ai',
  business: 'kyle@atlascybernetics.ai',
  technical: 'thomas@atlascybernetics.ai',
  operations: 'peter@atlascybernetics.ai',
  collaboration: 'tom@atlascybernetics.ai',
  security: 'security@atlas.net'
};

// Routes. Every internal href on the site comes from here so a rename is one
// edit and routes.test.js can prove each one has a page.
export const routes = {
  home: '/',
  why: '/why-avarok',
  platform: '/platform',
  engine: '/platform/engine',
  control: '/platform/control',
  economics: '/platform/economics',
  security: '/platform/security',
  deployment: '/platform/deployment',
  hardware: '/platform/hardware',
  benchmarks: '/benchmarks',
  solutions: '/solutions',
  pricing: '/pricing',
  demo: '/demo',
  // The booking form itself. In-page calls to action land here, on the form,
  // with the first field focused. The header button keeps the top of the page.
  demoForm: '/demo#book',
  waitlist: '/waitlist',
  broll: '/broll',
  resources: '/resources',
  updates: '/resources/updates',
  events: '/resources/events',
  contributors: '/resources/contributors',
  labs: '/labs',
  company: '/company',
  careers: '/company/careers',
  contact: '/contact',
  trust: '/trust',
  openSource: '/engine',
  controlPlane: '/control',
  diligence: '/diligence'
};

export const industries = [
  { slug: 'neoclouds', name: 'Neoclouds and GPU providers', short: 'GPU clouds' },
  { slug: 'enterprise-datacenter', name: 'Enterprise datacenters', short: 'Enterprise' },
  { slug: 'financial-services', name: 'Financial services', short: 'Banks' },
  { slug: 'healthcare', name: 'Healthcare', short: 'Hospitals' },
  { slug: 'government-defense', name: 'Government and defense', short: 'Gov and defense' },
  { slug: 'legal', name: 'Legal and professional services', short: 'Law firms' },
  { slug: 'hyperscalers', name: 'Hyperscalers and cloud platforms', short: 'Hyperscale' },
  { slug: 'research', name: 'Research labs and AI safety', short: 'Research' },
  { slug: 'smb-edge', name: 'SMB and edge', short: 'SMB and edge' }
];

export const solutionHref = (slug) => `${routes.solutions}/${slug}`;

// The desktop mega menu and the mobile drawer render from the same tree.
export const nav = {
  cta: { text: 'Book a demo', href: routes.demo },
  groups: [
    {
      label: 'Platform',
      columns: [
        {
          heading: 'The platform',
          items: [
            { text: 'Overview', blurb: 'One platform, three layers, every GPU dollar accounted for', href: routes.platform },
            { text: company.engine, blurb: 'Compiled inference in Rust and CUDA, more tokens on the same silicon', href: routes.engine },
            { text: company.control, blurb: 'Signed rollouts, GPU aware routing, fleet policy and self repair', href: routes.control },
            { text: company.economics, blurb: 'Cost per workload, chargeback and payback, against your baseline', href: routes.economics }
          ]
        },
        {
          heading: 'Trust and proof',
          items: [
            { text: 'Security', blurb: 'One signed binary, no Python in the request path, nothing leaves', href: routes.security },
            { text: 'Deployment', blurb: 'Hosted, your cloud account, on premises or air gapped', href: routes.deployment },
            { text: 'Hardware and models', blurb: 'Verified silicon and every recipe we ship', href: routes.hardware },
            { text: 'Benchmarks', blurb: 'The concurrency ladder and every gate record, live from the repo', href: routes.benchmarks }
          ]
        }
      ]
    },
    {
      label: 'Solutions',
      columns: [
        {
          heading: 'By industry',
          items: industries.slice(0, 5).map((i) => ({ text: i.name, href: solutionHref(i.slug) }))
        },
        {
          heading: 'More',
          items: [
            ...industries.slice(5).map((i) => ({ text: i.name, href: solutionHref(i.slug) })),
            { text: 'All solutions', href: routes.solutions, accent: true }
          ]
        }
      ]
    },
    { label: 'Why Avarok', href: routes.why },
    { label: 'Pricing', href: routes.pricing },
    // The other door. Buyers go to pricing and the demo, developers go to the
    // open source engine, and each page points at the other.
    { label: 'Developers', href: routes.openSource },
    {
      label: 'Resources',
      columns: [
        {
          heading: 'Learn',
          items: [
            { text: 'Blog', blurb: 'Kernel work, measured benchmarks, product notes', href: links.blog, external: true },
            { text: 'Documentation', blurb: 'The Avarok book, install to fleet', href: links.docs, external: true },
            { text: 'Product updates', blurb: 'What shipped, rendered from the changelog', href: routes.updates },
            { text: 'Events', blurb: 'Where to meet the team', href: routes.events }
          ]
        },
        {
          heading: 'Build',
          items: [
            { text: 'Open source', blurb: 'The engine, AGPL-3.0, running today', href: routes.openSource },
            { text: 'Community Edition', blurb: 'Not released yet. Join the waitlist', href: routes.waitlist },
            { text: 'Contributors', blurb: 'Everyone who has landed code, called out by name', href: routes.contributors },
            { text: 'Avarok Labs', blurb: 'The research arm and what it is working on', href: routes.labs },
            { text: 'Verification walkthrough', blurb: 'Reproduce the ladder yourself, step by step', href: routes.diligence }
          ]
        }
      ]
    },
    {
      label: 'Company',
      columns: [
        {
          heading: 'Avarok',
          items: [
            { text: 'About Avarok', blurb: 'The team and the two words that started it', href: routes.company },
            { text: 'Careers', blurb: 'The first hires on the founding team', href: routes.careers },
            { text: 'Trust center', blurb: 'Security posture, licensing and disclosure', href: routes.trust },
            { text: 'Contact', blurb: 'Sales, partnerships, hardware and press', href: routes.contact }
          ]
        }
      ]
    }
  ]
};

export const footer = {
  slogan: 'Same silicon. Smarter inference. Stronger scalability.',
  cols: [
    {
      heading: 'Platform',
      links: [
        { text: 'Overview', href: routes.platform },
        { text: company.engine, href: routes.engine },
        { text: company.control, href: routes.control },
        { text: company.economics, href: routes.economics },
        { text: 'Security', href: routes.security },
        { text: 'Deployment', href: routes.deployment },
        { text: 'Benchmarks', href: routes.benchmarks }
      ]
    },
    {
      heading: 'Solutions',
      links: industries.map((i) => ({ text: i.name, href: solutionHref(i.slug) }))
    },
    {
      heading: 'Resources',
      links: [
        { text: 'Blog', href: links.blog, external: true },
        { text: 'Documentation', href: links.docs, external: true },
        { text: 'Open source', href: routes.openSource },
        { text: 'Community Edition waitlist', href: routes.waitlist },
        { text: 'Contributors', href: routes.contributors },
        { text: 'Product updates', href: routes.updates },
        { text: 'Avarok Labs', href: routes.labs },
        { text: 'Control plane, live', href: routes.controlPlane }
      ]
    },
    {
      heading: 'Company',
      links: [
        { text: 'About', href: routes.company },
        { text: 'Careers', href: routes.careers },
        { text: 'Pricing', href: routes.pricing },
        { text: 'Trust center', href: routes.trust },
        { text: 'Contact', href: routes.contact },
        { text: 'Book a demo', href: routes.demo }
      ]
    }
  ],
  legal: `© 2026 ${company.legal} Avarok and Avarok Engine are products of ${company.legal}`,
  license: 'Community Edition under AGPL-3.0. Enterprise Edition under a commercial license.',
  mlperf:
    'The MLPerf name and logo are registered and unregistered trademarks of MLCommons Association in the United States and other countries. All rights reserved. Unauthorized use strictly prohibited. See mlcommons.org for more information.'
};
