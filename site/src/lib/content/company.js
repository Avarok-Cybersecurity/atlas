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

// The team, from the company's own team slide: names, titles and one line each,
// cut to the highest signal. `photo` is a square WebP under static/team, `hue`
// one of the four brand colours, `linkedin` the profile each person gave.
// Every entry is a real person's public face: each of them should confirm their
// own line before this merges, and a line changes only on their word.
// `showTeam = false` takes the whole section off the page.
export const showTeam = true;
export const team = {
  eyebrow: 'Team',
  title: 'Deep technical roots. Proven commercial leadership.',
  people: [
    {
      name: 'Kyle Croll',
      role: 'CEO, Co-Founder',
      focus: 'Strategy and finance',
      bio: 'Fortune 30 security leadership. Eight years in cyber intelligence, incident response and M&A diligence. Navy veteran. MBA, Texas McCombs.',
      photo: 'kyle-croll',
      hue: 'violet',
      linkedin: 'https://www.linkedin.com/in/kylecroll/'
    },
    {
      name: 'Thomas Braun',
      role: 'CTO, Founder',
      focus: 'Engineering',
      bio: 'Started the engine and owns its architecture. Veteran Rust and systems engineer. Created the patent allowed Citadel encryption protocol.',
      photo: 'thomas-braun',
      hue: 'cyan',
      linkedin: 'https://www.linkedin.com/in/tpbraun/'
    },
    {
      name: 'Eric Gonzalez',
      role: 'CRO, Co-Founder',
      focus: 'Revenue',
      bio: 'Ran Americas and APAC for Anaconda from $50M to $156M ARR. Early Oracle NetSuite. Two exits, twice a founder. MBA, Texas McCombs.',
      photo: 'eric-gonzalez',
      hue: 'gold',
      linkedin: 'https://www.linkedin.com/in/ericgonzalez/'
    },
    {
      name: 'Peter Drybrough',
      role: 'Founding CPO',
      focus: 'Operations and product',
      bio: 'Ten years leading cyber intelligence, security operations and security engineering in the Fortune 30. M.S. in Intelligence Analysis, Johns Hopkins.',
      photo: 'peter-drybrough',
      hue: 'green',
      linkedin: 'https://www.linkedin.com/in/c1ph3rp13rr3/'
    },
    {
      name: 'Tom Turney',
      role: 'Technical Advisor and Investor',
      focus: 'Kernels and compression',
      bio: 'Nine and a half years at Google as a Staff TLM. Created TurboQuant+, open source KV cache compression. CEO and founder of PsyGuard.AI. Core contributor to the engine.',
      photo: 'tom-turney',
      hue: 'violet',
      linkedin: 'https://www.linkedin.com/in/tturney/'
    }
  ],
  // The deck. `file` is empty on purpose: the repository is public, so a file
  // committed here is published the moment it is pushed, before anyone reviews
  // it. With no file the button asks for the deck by email. To offer a download
  // instead, put the PDF under static/ and set `file` to its path.
  deck: {
    body: 'The deck, the receipts and the verification walkthrough are available to investors and design partners.',
    file: '',
    request: { text: 'Request the deck', href: `mailto:${contacts.business}?subject=Avarok%20deck` },
    download: 'Download the deck'
  }
};

// "The exchange" on the company page: the real comments, captured from the public
// pull request in both themes, with their words as alt text so nothing is locked
// inside a picture. Captured in UTC, so the dates in the picture are the dates
// the timeline cites. scripts/media/capture-thread.mjs takes them again.
export const exchange = {
  eyebrow: 'The exchange',
  image: { light: '/media/about/exchange-light.webp', dark: '/media/about/exchange-dark.webp', width: 1240, height: 710 },
  alt: 'Two comments on GitHub. XMR13 asks: why are you so aggressive? Your PR looks completely 100% AI generated. tbraun96, the author, replies: Your point?',
  before: {
    summary: 'What came before it',
    image: { light: '/media/about/exchange-before-light.webp', dark: '/media/about/exchange-before-dark.webp', width: 1240, height: 1424 },
    alt: 'Two earlier comments. A maintainer writes that the pull request appears to contain substantial AI generated code without disclosure and cannot be accepted in its current form. The author replies that it works on the DGX Spark, that the requirement is like the 1960s when people thought compilers were sketchy, and that whether AI or a compiler, both translate one language to another.'
  },
  caption: 'llama.cpp pull request 18680, January 7 and 8, 2026 (UTC). Captured from the public thread.',
  link: { text: 'The whole thread', href: links.llamaCppPr }
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
    { title: 'Agents draft, you decide', body: 'The repository is AI first. Agents write the first pass, certified benchmarks gate the merge, and a person decides what ships.' },
    { title: 'Hardware you can touch', body: 'DGX Spark and Strix Halo are on the bench today, from NVIDIA and AMD. Hopper and Blackwell are the next campaigns.' },
    { title: 'The number is the argument', body: 'Every claim ships with the measurement behind it, including the rungs we lost on the way. Nobody wins a review by seniority.' },
    { title: 'Early enough to matter', body: 'The team is small enough that the first hires decide how the company works, not only what it builds.' }
  ],
  // Short on purpose. It states how the work is done, which the repository
  // shows, and nothing about terms of employment.
  culture: {
    eyebrow: 'How we work',
    lines: ['Ship the number, then talk about it.', 'AI writes the first draft. A person owns the result.', 'If it is not measured, it is an opinion.', 'Small team. Real hardware. No theatre.'],
    body: 'We started because a working change was turned away for how it was written, not for what it did. So we judge work by what it does. Bring a result and the measurement behind it and you will be heard, whoever you are and however you made it.'
  },
  // The fastest way in. No money is promised here: a bounty is a term the company
  // would have to set, fund and honour, and this page cannot do that for it.
  fastTrack: {
    eyebrow: 'The fast track',
    title: 'Open a pull request.',
    body: 'The repository is public. Pick an issue, land a change, and mention it in your note. It is the introduction we read first, and it tells us more than an interview does.',
    primary: { text: 'Good first issues', href: `${links.github}/labels/good%20first%20issue`, external: true },
    secondary: { text: 'How to contribute', href: links.contributing, external: true }
  },
  // These are the first hires the company plans. Nothing says a search is open
  // for any of them yet, so the page does not say "openings".
  rolesTitle: 'The first hires',
  rolesHeading: 'Four roles we are building toward',
  roles: [
    {
      title: 'Kernel Engineer',
      location: 'Location by agreement',
      team: 'Engine',
      body: 'CUDA and Rust. Attention, MoE, GDN and quantized GEMM kernels per hardware target, with a certified benchmark on every merge. Hopper and Blackwell are the next campaigns.',
      does: ['Write and tune kernels for one hardware target at a time', 'Land every change behind a certified benchmark', 'Open the Hopper and Blackwell campaigns'],
      signal: 'A kernel you wrote, and the number it moved.'
    },
    {
      title: 'Founding Platform and Distributed Systems Engineer',
      location: 'Location by agreement',
      team: 'Control',
      body: 'The control plane. Routing, rollout, autoscaling, repair and policy across mixed NVIDIA and AMD fleets, in Rust, on Kubernetes, in the customer’s account and ours.',
      does: ['Build routing, rollout, autoscaling, repair and policy in Rust', 'Run it on Kubernetes, in the customer’s account and in ours', 'Keep the control plane off the inference path'],
      signal: 'A scheduler or control plane you ran in production.'
    },
    {
      title: 'Founding Infrastructure and Forward Deployed Engineer',
      location: 'Location by agreement, travel',
      team: 'Customers',
      body: 'You stand up the pilot, run the ladder on the customer’s hardware, and hand them the receipt. The person the customer calls, and the person who tells engineering what broke.',
      does: ['Stand up the pilot on the customer’s hardware', 'Run the ladder and hand over the receipt', 'Bring what broke back to engineering'],
      signal: 'A deployment you carried from the first call to production.'
    },
    {
      title: 'Strategic Product and Marketing Advisor',
      location: 'Part time',
      team: 'Go to market',
      body: 'Former product, sales or GTM leadership at an inference or GPU infrastructure company. Monthly working sessions, pricing and positioning review, select enterprise calls.',
      does: ['Monthly working sessions with the founders', 'Review pricing and positioning', 'Join select enterprise calls'],
      signal: 'Product, sales or go to market leadership at an inference or GPU infrastructure company.'
    }
  ],
  cta: { text: 'Email the team', href: `mailto:${contacts.technical}?subject=Avarok%20careers` },
  apply: { text: 'Register interest', href: '#apply' },
  // The interest form. Same component as the demo form, so it behaves the same:
  // it posts to `formEndpoint` when there is one, and drafts an email when not.
  form: {
    title: 'Register interest',
    fields: [
      { name: 'name', label: 'Your name', type: 'text', required: true, autocomplete: 'name' },
      { name: 'email', label: 'Email', type: 'email', required: true, autocomplete: 'email' },
      { name: 'role', label: 'Role', type: 'select', options: ['Kernel Engineer', 'Platform and Distributed Systems Engineer', 'Infrastructure and Forward Deployed Engineer', 'Product and Marketing Advisor', 'Something else'] },
      { name: 'work', label: 'Something you built', type: 'url', placeholder: 'A repository, a pull request, a write up' },
      { name: 'notes', label: 'Anything else', type: 'textarea', placeholder: 'What you want to work on, and why here' }
    ],
    submit: 'Send it',
    fallbackNote: 'Submitting opens a prefilled email to the engineering team. Nothing is stored on this site.',
    thanks: 'Thank you. We read every one.',
    composed: 'Your note is drafted in your mail app. Press send and it reaches us.'
  }
};

export const contact = {
  eyebrow: 'Contact',
  title: 'Someone should know how to reach us immediately.',
  lede: 'Pick the path that fits. Every one of them lands with a founder.',
  paths: [
    // `doors` are the addresses on a card, each with the job it is for. The label is
    // the job, not the person, so the card stays true when the person changes.
    { title: 'Sales and pilots', body: 'Enterprise, datacenter, air gapped or a fleet of boxes. Tell us the hardware and the workload and we scope the ladder.', doors: [{ label: 'Email sales', email: contacts.sales }], demo: true },
    { title: 'Technical and open source', body: 'Running the open source engine, bringing hardware to the table, or want to contribute. Discord is fastest, email works.', doors: [{ label: 'Email engineering', email: contacts.technical }], discord: true },
    {
      title: 'Partnerships',
      body: 'Silicon vendors, clouds, frameworks, benchmarks and standards bodies. If it advances inference on hardware people own, we want the conversation.',
      doors: [
        { label: 'Business and design partners', email: contacts.business },
        { label: 'Public collaboration', email: contacts.collaboration }
      ]
    },
    {
      title: 'Security',
      body: 'Report a vulnerability privately. We acknowledge within 48 hours and give an initial assessment within seven days.',
      doors: [
        { label: 'Report privately', email: contacts.security },
        { label: 'Email operations', email: contacts.operations }
      ],
      href: links.securityPolicy
    },
    { title: 'Press and investors', body: 'Deck, receipts and the verification walkthrough on request.', doors: [{ label: 'Email the CEO', email: contacts.business }] }
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
    thanks: 'Thank you. A founder will reply.',
    // Shown instead of `thanks` while there is no form endpoint: nothing has been
    // sent yet, the visitor still has to press send in their own mail app.
    composed: 'Your request is drafted in your mail app. Press send and it reaches us.'
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
    thanks: 'You are on the list. We will write when the Community Edition is released.',
    composed: 'Your note is drafted in your mail app. Press send and you are on the list.'
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
