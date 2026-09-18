// =============================================================================
// /company, /company/careers, /contact, /trust, /demo. The story, the team
// (present in data, hidden by a flag until the founders are full time), the
// roles, the contact paths and the trust posture.
// =============================================================================
import { routes, links, contacts, company } from './brand.js';

export const story = {
  eyebrow: 'About Avarok',
  title: 'It started with two words.',
  lede:
    'In January 2026 a working improvement to llama.cpp was closed because it had been written with AI. The author wrote a short appeal to common sense, and when it went over everyone’s heads, answered the room with a question. Then he went and built the engine from scratch.',
  beats: [
    {
      when: 'January 7, 2026',
      title: 'The pull request',
      body: 'A loop attention model ran on a DGX Spark. The pull request adding it to llama.cpp was closed as containing AI generated code without disclosure. The reply argued that whether AI or a compiler, both translate one language to another, and that a community building AI tooling should not hold contempt for AI written code.',
      href: links.llamaCppPr,
      cta: 'Read the thread'
    },
    {
      when: 'January 8, 2026',
      title: '“Your point?”',
      body: 'Asked why the pull request looked entirely AI generated, the author answered with two words. They became the first principle of the repository that followed. AI authored is the default. A human who writes code by hand explains why they were better than the machine.'
    },
    {
      when: 'Winter 2026',
      title: 'From scratch, in Rust',
      body: 'Months of trying to improve vLLM on the Spark had shown that the feedback loop from a kernel change to a number was too slow to learn from. The serving stack was rewritten in Rust with hand tuned CUDA, no Python, and a build that takes a minute instead of forty.'
    },
    {
      when: 'May 2026',
      title: 'One Reddit post',
      body: 'A stable 102 tokens per second on a DGX Spark, posted to r/LocalLLaMA. The star count went from a few dozen to a few hundred in a week and the Discord became the test fleet.'
    },
    {
      when: 'July 2026',
      title: 'Receipts',
      body: 'The fused Qwen Gated DeltaNet kernel merged into Hugging Face Transformers. MLCommons named the project a contributor to the new MLPerf edge agentic benchmark. AMD provided a Strix Halo desktop and the MLPerf submission went in from the same CUDA source on both vendors.'
    },
    {
      when: 'August 2026',
      title: 'The ladder',
      body: 'The concurrency ladder against the matched vLLM configuration was published with every rung lost on the way. Eight rungs, eight wins, and the margin widest at C=128.'
    },
    {
      when: 'September 2026',
      title: 'Avarok',
      body: 'The company took its name from the protocol work its founder had done before the engine, brought in commercial leadership that had scaled Anaconda, and set out to sell what the engine had proved. The industry measures inference in tokens per second. Avarok measures what that performance is worth.'
    }
  ]
};

export const mission = {
  title: 'Same silicon. Smarter inference. Stronger scalability.',
  body:
    'AI worth having should run on hardware you own, whether that is an accelerator at the edge, the workstation under your desk, or a rack you operate. We build one engine for the whole range, verify it on the silicon we can put our hands on, and make what it produces accountable to the people paying for it.',
  principles: [
    { title: 'Receipts, not adjectives', body: 'Every performance number on this site is generated from a record in the repository. If it is not in the repo, it is not on the page.' },
    { title: 'AI first, human accountable', body: 'AI authored is the default in the repository. Certified benchmarks gate every kernel change. People decide what ships.' },
    { title: 'Own the request path', body: 'Security, governance and economics are only exact when the engine is under the workload. Everything we build follows from that.' },
    { title: 'Open at the core', body: 'The Community Edition is AGPL-3.0 and always will be. The enterprise platform pays for the people who keep it that way.' }
  ]
};

// The founders section is off, on instruction: "skip the founders section for
// now". The markup is in place on the company page and renders when showTeam
// is true and `people` has entries ({ name, role, focus, bio }). The people and
// their bios are deliberately not in this repository. They are real people's
// details and go in when each of them has approved their own entry.
export const showTeam = false;
export const team = {
  eyebrow: 'Team',
  title: 'Deep technical roots. Proven commercial leadership.',
  people: []
};

export const careers = {
  eyebrow: 'Careers',
  title: 'Build the layer between the GPU and the invoice.',
  lede:
    'We are a small founding team with an AI first repository, hardware from NVIDIA and AMD on the bench, and buyers who want receipts. The first hires shape the company.',
  how: `Send a note and a link to something you built to ${contacts.technical}. Code beats resumes. A pull request against the engine beats both.`,
  // How the work is done, all of it visible in the repository. Deliberately not
  // here: equity, equipment, location or remote policy. Those are terms of
  // employment, only the company can state them, and a candidate would be
  // entitled to rely on anything this page promised.
  benefits: [
    { title: 'An AI first repository', body: 'Agents write the first draft, certified benchmarks gate the merge, you decide what ships.' },
    { title: 'Real hardware on the bench', body: 'DGX Spark and Strix Halo today, from NVIDIA and AMD. Hopper and Blackwell are the next campaigns.' },
    { title: 'Receipts, not adjectives', body: 'Every claim ships with the measurement behind it, including the rungs we lost on the way.' },
    { title: 'A founding team', body: 'Small enough that the first hires decide how the company works.' }
  ],
  // These are the first hires the company plans. Nothing says a search is open
  // for any of them yet, so the page does not say "openings".
  rolesTitle: 'The first hires',
  rolesHeading: 'Four roles we are building toward',
  roles: [
    { title: 'Kernel Engineer', location: 'Location by agreement', team: 'Engine', body: 'CUDA and Rust. Attention, MoE, GDN and quantized GEMM kernels per hardware target, with a certified benchmark on every merge. Hopper and Blackwell are the next campaigns.' },
    { title: 'Founding Platform and Distributed Systems Engineer', location: 'Location by agreement', team: 'Control', body: 'The control plane. Routing, rollout, autoscaling, repair and policy across mixed NVIDIA and AMD fleets, in Rust, on Kubernetes, in the customer’s account and ours.' },
    { title: 'Founding Infrastructure and Forward Deployed Engineer', location: 'Location by agreement, travel', team: 'Customers', body: 'You stand up the pilot, run the ladder on the customer’s hardware, and hand them the receipt. The person the customer calls, and the person who tells engineering what broke.' },
    { title: 'Strategic Product and Marketing Advisor', location: 'Part time', team: 'Go to market', body: 'Former product, sales or GTM leadership at an inference or GPU infrastructure company. Monthly working sessions, pricing and positioning review, select enterprise calls.' }
  ],
  cta: { text: 'Email the team', href: `mailto:${contacts.technical}?subject=Avarok%20careers` }
};

export const contact = {
  eyebrow: 'Contact',
  title: 'Someone should know how to reach us immediately.',
  lede: 'Pick the path that fits. Every one of them lands with a founder.',
  paths: [
    { title: 'Sales and pilots', body: 'Enterprise, datacenter, air gapped or a fleet of boxes. Tell us the hardware and the workload and we scope the ladder.', email: contacts.sales, cta: 'Email sales', demo: true },
    { title: 'Technical and open source', body: 'Running the Community Edition, bringing hardware to the table, or want to contribute. Discord is fastest, email works.', email: contacts.technical, cta: 'Email engineering', discord: true },
    { title: 'Partnerships', body: 'Silicon vendors, clouds, frameworks, benchmarks and standards bodies. If it advances inference on hardware people own, we want the conversation.', email: contacts.sales, cta: 'Email partnerships' },
    { title: 'Security', body: 'Report a vulnerability privately. We acknowledge within 48 hours and give an initial assessment within seven days.', email: contacts.security, cta: 'Report privately', href: links.securityPolicy },
    { title: 'Press and investors', body: 'Deck, receipts and the verification walkthrough on request.', email: contacts.press, cta: 'Email the CEO' }
  ]
};

export const demoPage = {
  eyebrow: 'Book a demo',
  title: 'See it against your own workload.',
  lede:
    'A working session, not a slideshow. Tell us the hardware and the workload, and we come with the ladder ready to run on it. Thirty minutes to decide whether a four week proof of value is worth your time.',
  bullets: [
    'Live walkthrough of the console on demo data',
    'The published ladder, and what it would look like on your fleet',
    'The payback model with your inputs',
    'A scoped proof of value if it makes sense'
  ],
  form: {
    title: 'Tell us about your fleet',
    fields: [
      { name: 'name', label: 'Your name', type: 'text', required: true, autocomplete: 'name' },
      { name: 'email', label: 'Work email', type: 'email', required: true, autocomplete: 'email' },
      { name: 'company', label: 'Company', type: 'text', required: true, autocomplete: 'organization' },
      { name: 'segment', label: 'You are', type: 'select', options: ['Enterprise datacenter', 'Neocloud or GPU provider', 'Air gapped or sovereign', 'Workstation or SMB', 'Research lab', 'Investor or press', 'Something else'] },
      { name: 'hardware', label: 'Hardware you run inference on', type: 'text', placeholder: 'e.g. 64 H100, 2 DGX Spark, mixed NVIDIA and AMD' },
      { name: 'engine', label: 'Serving stack today', type: 'select', options: ['vLLM', 'SGLang', 'llama.cpp', 'TensorRT-LLM or NIM', 'A hosted API', 'None yet', 'Other'] },
      { name: 'notes', label: 'What you want to see', type: 'textarea', placeholder: 'The workload, the concurrency, the question you need answered' }
    ],
    submit: 'Request a working session',
    fallbackNote: 'Submitting opens a prefilled email to the founding team. Nothing is stored on this site.',
    thanks: 'Thank you. A founder will reply.'
  },
  aside: {
    title: 'Prefer email',
    body: `Write to ${contacts.sales} with the hardware and the workload. Same people, same answer.`,
    discord: 'Or find us in Discord, we are in there every day.'
  }
};

// /waitlist. The Community Edition is announced, not released, so every button
// that used to say "install" comes here. The engine underneath it IS released
// and open source, and the page says so, because a developer who wants it today
// should not be told to wait. When the edition ships: point `routes.waitlist`
// users back at the install page and delete this block and its route.
export const waitlistPage = {
  eyebrow: 'Community Edition',
  title: 'The Community Edition is not out yet. Be first when it is.',
  lede:
    'The Community Edition is the free edition of Avarok, under AGPL-3.0. Leave an address and the hardware you run, and we will write to you when it is released.',
  bullets: [
    'One note when the Community Edition is released',
    'Tell us the hardware you run, so we know what people are waiting on',
    'Need it for a business today? A working session is the faster road'
  ],
  today: {
    title: 'Want to run something today',
    body: 'The engine underneath is open source and running now. The developer page has the install command, the recipes and the numbers.',
    cta: { text: 'Go to the developer page', href: routes.openSource },
    discord: 'The people building it are in Discord every day.'
  },
  form: {
    title: 'Join the waitlist',
    fields: [
      { name: 'email', label: 'Email', type: 'email', required: true, autocomplete: 'email' },
      { name: 'name', label: 'Your name', type: 'text', autocomplete: 'name' },
      { name: 'hardware', label: 'Hardware you would run it on', type: 'text', placeholder: 'e.g. DGX Spark, Strix Halo, 2 RTX 5090, 8 H100' },
      { name: 'use', label: 'You are', type: 'select', options: ['A developer', 'A research lab', 'A small business', 'An enterprise team', 'Something else'] },
      { name: 'notes', label: 'What you would run', type: 'textarea', placeholder: 'The models, the workload, anything we should know' }
    ],
    submit: 'Join the waitlist',
    fallbackNote: 'Submitting opens a prefilled email to the team. Nothing is stored on this site.',
    thanks: 'You are on the list. We will write when the Community Edition is released.'
  },
  cta: {
    title: 'Running inference for a business?',
    body: 'The Enterprise Edition is available now, with the control plane, the economics layer and a named engineer.',
    primary: { text: 'Book a demo', href: routes.demoForm },
    secondary: { text: 'See pricing', href: routes.pricing }
  }
};

export const trust = {
  eyebrow: 'Trust center',
  title: 'What we run, what we claim, and what we do not.',
  lede:
    'The security policy in the repository says which controls are automated and which are human review, and claims nothing it does not run. This page is the same posture, for the people who sign.',
  sections: [
    {
      title: 'Architecture',
      items: [
        'One signed binary of about 75 MB in Rust and CUDA. No Python, no PyTorch, no runtime compilation in the request path.',
        'Recipes, models and kernels delivered as signed artifacts. Kernel targets content hashed down to their transitive include closure.',
        'Release images promoted by digest from staging to canary to production, never rebuilt between environments.',
        'The control plane is out of band. It never sits on the synchronous inference path.'
      ]
    },
    {
      title: 'Data handling',
      items: [
        'Prompts, weights, outputs and telemetry stay on hardware you own, in your cloud account, or on an air gapped network.',
        'Prompt and output logging is configurable and off by default in enterprise deployments.',
        'In bring your own cloud, no inference request leaves your account. The control plane sees configuration, licensing, versions and aggregate telemetry.',
        'Air gapped installs from local media, with telemetry exported on your schedule or never.'
      ]
    },
    {
      title: 'Assurance',
      items: [
        'Cargo deny audits dependencies for advisories, license compliance and banned crates on every pull request and weekly.',
        'Every kernel change carries a certified benchmark before it merges. Every release passes the serve matrix on the real box.',
        'There is no automated static analysis of CUDA kernel sources. Kernel memory safety is human review plus the runtime kernel audit, and we say so.',
        'SOC 2 readiness documentation, model risk documentation and pinned recipe governance packs are part of the first SLA engagements. Ask for the current state of the audit program.'
      ]
    },
    {
      title: 'Licensing',
      items: [
        'Community Edition under AGPL-3.0-only. Contributions are covered by a CLA that permits Enterprise relicensing.',
        'Enterprise Edition under a commercial license, per GPU per year, with terms your legal team can read in one sitting.',
        'Third party names on this site belong to their owners and appear for background or as primary sources, never as endorsements.'
      ]
    }
  ],
  disclosure: {
    title: 'Responsible disclosure',
    body: `Do not open a public issue for a vulnerability. Email ${contacts.security} with a description, reproduction steps, environment and the affected component. We acknowledge within 48 hours and give an initial assessment within seven days, then merge the fix, tag a release and credit the reporter unless anonymity is requested.`,
    href: links.securityPolicy,
    cta: 'Read the policy'
  }
};

export const companyCta = {
  eyebrow: 'Next step',
  title: 'Come build with us, or come buy from us.',
  body: 'Both conversations start the same way. Tell us what you run.',
  primary: { text: 'Book a demo', href: routes.demoForm },
  secondary: { text: 'The first hires', href: routes.careers }
};

export { company };
