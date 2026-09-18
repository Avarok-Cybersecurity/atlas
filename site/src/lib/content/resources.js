// =============================================================================
// /resources, /resources/updates, /resources/events, /resources/contributors,
// /labs. The hub, the events list, the research arm, and the copy around the
// two generated lists (updates from CHANGELOG.md, contributors from GitHub).
// =============================================================================
import { routes, links } from './brand.js';

export const resourcesHub = {
  eyebrow: 'Resources',
  title: 'Everything we publish, and where to find us.',
  lede: 'Engineering notes, the book, product updates rendered from the changelog, the open source project, and the people behind it.',
  cards: [
    { title: 'Blog', body: 'Kernel work, measured benchmarks and what it takes to run frontier models on hardware you own. Everything reproducible from a commit.', href: links.blog, external: true, cta: 'Read the blog' },
    { title: 'Documentation', body: 'The Avarok book. Install, recipes, deployment, the control plane and the gates, with an llms.txt for answer engines.', href: links.docs, external: true, cta: 'Open the book' },
    { title: 'Product updates', body: 'What shipped, what changed and what it does not change, rendered from the repository changelog on every build.', href: routes.updates, cta: 'See what shipped' },
    { title: 'Benchmarks', body: 'The concurrency ladder and every gate record, generated from the repository. Reproduce any of them with the command on the page.', href: routes.benchmarks, cta: 'See the numbers' },
    { title: 'Open source', body: 'The Community Edition. One install command, every recipe, the deployment guide and the Discord where the work happens.', href: routes.openSource, cta: 'Install the engine' },
    { title: 'Contributors', body: 'Everyone who has landed code in the repository, called out by name, with the core team’s roles.', href: routes.contributors, cta: 'Meet the contributors' },
    { title: 'Events', body: 'Where to meet the team, in person and online.', href: routes.events, cta: 'Find us' },
    { title: 'Avarok Labs', body: 'The research arm. Kernels, compression, speculative decoding, protocols and the day zero model bring ups.', href: routes.labs, cta: 'See the research' },
    { title: 'Verification walkthrough', body: 'Reproduce the ladder yourself. Fingerprint, parity, commands and the artifacts behind the vLLM comparison.', href: routes.diligence, cta: 'Walk through it' }
  ]
};

export const updates = {
  eyebrow: 'Product updates',
  title: 'What shipped.',
  lede: 'Rendered from CHANGELOG.md in the repository on every build. Kernel level wins, defaults that changed, and fixes, in the words of the engineers who landed them.',
  more: { text: 'The full changelog on GitHub', href: links.changelog }
};

export const events = {
  eyebrow: 'Events',
  title: 'Where to meet the team.',
  lede: 'In person when the calendar fills, in Discord every day. Add an event here and it appears on the site.',
  // Add entries with a date, a title, a place and a link. Past events sort
  // below upcoming ones automatically.
  items: [
    {
      date: '2026-09-22',
      title: 'Working session, Avarok Console walkthrough',
      place: 'Online, by request',
      body: 'A thirty minute walkthrough of the console on demo data, the published ladder, and the payback model with your inputs.',
      href: routes.demo,
      cta: 'Book a session'
    },
    {
      date: '2026-10-01',
      title: 'MLPerf Inference v6.1 results',
      place: 'Published by MLCommons',
      body: 'Our submission is in the closed edge division on both GB10 and gfx1151 from the same CUDA source. The numbers render on the benchmarks page the moment MLCommons publishes them.',
      href: links.mlcommons,
      cta: 'The benchmark announcement'
    },
    {
      date: 'daily',
      title: 'Discord, every day',
      place: 'discord.gg/RQcGakU2jW',
      body: 'Hundreds of builders running the engine on their own hardware. We are in there every day, shipping fixes, taking model requests and tuning kernels in the open.',
      href: links.discord,
      cta: 'Join the Discord'
    }
  ]
};

export const contributors = {
  eyebrow: 'Contributors',
  title: 'Everyone who has landed code, called out by name.',
  lede: 'Generated from the GitHub contributors API on every build. The core team is annotated with their role. Everyone else is the reason the test fleet keeps growing.',
  // GitHub logins with a role. Anyone not listed here renders as a contributor.
  core: {
    tbraun96: 'Founder, CTO, engine architecture',
    TheTom: 'Core contributor, kernels and quantization, advisor',
    rsafier: 'Core contributor, EXL3 and multi node',
    rrstesiak: 'Founding engineer, speculative decoding and single Spark records',
    SeedSource: 'Core contributor, grammar and prefix cache',
    DrRainbows: 'Systems engineering, site and information path'
  },
  cla: 'Contributions ship in the Community Edition under AGPL-3.0. The CLA permits Enterprise relicensing.',
  cta: { text: 'Good first issues', href: `${links.github}/labels/good%20first%20issue` },
  cta2: { text: 'How to contribute', href: links.contributing }
};

export const labs = {
  eyebrow: 'Avarok Labs',
  title: 'The research arm.',
  lede:
    'Labs is where the engine gets its next order of magnitude. Kernels and compression, speculative decoding, memory and context, compilers and languages, protocols, agentic benchmarks, and the day zero bring up of every model that matters. The work lands in the open repository as pull requests with certified benchmarks.',
  tracks: [
    { title: 'Kernels and quantization', body: 'Hand tuned attention, MoE, GDN and quantized GEMM per hardware target. NVFP4, FP8 and K quant expert kernels on raw blocks. TurboQuant+ KV cache compression.', color: 'violet' },
    { title: 'Speculative decoding', body: 'MTP draft heads, DFlash block diffusion, lookup drafts into a wide verify, and a resolver that picks the verify width itself.', color: 'cyan' },
    { title: 'Memory and context', body: 'Tiered KV and SSM state across host RAM, NVMe and RDMA peers. Prefix caches that are prefilled once per fleet, not once per node. The context bus between agents.', color: 'green' },
    { title: 'Compilers and languages', body: 'One CUDA source compiled for NVIDIA and AMD through SCALE. Rust as the systems language. Interest in massively parallel functional runtimes, Bend and HVM among them, for the next abstraction.', color: 'gold' },
    { title: 'Protocols and transport', body: 'Node to node links designed for a hostile network, one sided RDMA primitives shared by every tier, and the Citadel protocol lineage the founder brought to the company.', color: 'violet' },
    { title: 'Agentic benchmarks', body: 'Contributor to the MLPerf edge agentic benchmark. BFCL and replayed agentic trajectories as gates, because a benchmark should look like the work.', color: 'cyan' },
    { title: 'Day zero model bring up', body: 'DeepSeek V4.1 Flash, Kimi K3, GLM 5.3, Qwen 3.8 Flash Next and Gemma 4 in open pull requests. The goal is that model vendors check Avarok the same week they check vLLM.', color: 'green' },
    { title: 'Inference economics', body: 'The correlation layer between operator telemetry and the ledger. Living benchmarks on the latest hardware and the latest models, because costs for equivalent quality keep falling and the measurement has to keep up.', color: 'gold' }
  ],
  vision: {
    title: 'What we are building toward',
    body:
      'Labeled as direction, not as a shipped product. A fleet where every idle machine on the network joins the mesh overnight as cache and compute. A repository that merges, corrects and versions itself around the clock with certified gates. A tuner that takes any hardware in any configuration, any model in any version, and has it running fast and stable on day zero. Each of these is a step we are taking now because the things we have to build today are on the way there.'
  },
  cta: { text: 'Open a research collaboration', href: routes.contact },
  cta2: { text: 'Open pull requests', href: `${links.github}/pulls`, external: true }
};

export const openSourceCallout = {
  eyebrow: 'Open source',
  title: 'The engine is free. The platform pays for the people who keep it that way.',
  body: 'Install the Community Edition in one command, run any recipe, and bring your machine to the test fleet. The Enterprise Edition adds the control plane, the economics layer and a named engineer.',
  primary: { text: 'Install the engine', href: routes.openSource },
  secondary: { text: 'Star on GitHub', href: links.github, external: true }
};
