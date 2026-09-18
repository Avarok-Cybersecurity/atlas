// =============================================================================
// The page registry. One entry per marketing route: its title, description
// and which FAQ set it shows. The sitemap, the JSON-LD in the layout, the
// route test and the Seo component all read this, so a page cannot exist
// without a title and cannot be missing from the sitemap.
// =============================================================================
import { routes, industries, solutionHref, company, SITE, links } from './brand.js';

export { SITE, company, routes, industries, solutionHref, links };

const T = (t) => `${t} · Avarok`;
// Sentence case for a name used mid sentence, without flattening acronyms:
// 'Neoclouds and GPU providers' becomes 'neoclouds and GPU providers'.
const ACRONYMS = new Set(['GPU', 'AI', 'SMB']);
const lower = (s) => s.split(' ').map((w) => (ACRONYMS.has(w) ? w : w.charAt(0).toLowerCase() + w.slice(1))).join(' ');

export const pages = [
  { path: routes.home, title: 'Avarok, the inference economics platform', description: 'Faster inference, stronger governance, a fraction of what you pay today. Avarok runs your models faster on GPUs you own, keeps every prompt inside your perimeter, and shows what each workload costs.', faq: 'home', priority: 1.0 },
  { path: routes.why, title: T('Why Avarok'), description: 'You invested in the datacenter. Now get the most out of it. Speed, security and governance, each explained, tested and delivered at your pace.', faq: 'why', priority: 0.9 },
  { path: routes.platform, title: T('Platform'), description: 'One platform for the whole inference lifecycle. Avarok Engine, Avarok Control and Avarok Economics share one request path.', priority: 0.9 },
  { path: routes.engine, title: T(company.engine), description: 'A compiled inference stack in Rust and CUDA. Hand tuned kernels per hardware, model and quantization. More tokens on the same silicon.', faq: 'hardware', priority: 0.8 },
  { path: routes.control, title: T(company.control), description: 'The governance and control plane. Signed recipes, canary rollouts, GPU aware routing, autoscaling, node repair and fleet policy, never on the inference path.', faq: 'deploy', priority: 0.8 },
  { path: routes.economics, title: T(company.economics), description: 'Every GPU, every workload, every dollar. Turn runtime telemetry into cost per workload, chargeback, stranded capacity and payback.', faq: 'pricing', priority: 0.8 },
  { path: routes.security, title: T('Security'), description: 'One signed binary, no interpreter in the request path, links designed for a hostile network, nothing leaves your perimeter.', faq: 'security', priority: 0.8 },
  { path: routes.deployment, title: T('Deployment'), description: 'Hosted with private connectivity, your cloud account, on premises or air gapped. Same binary, same recipes, same control plane.', faq: 'deploy', priority: 0.8 },
  { path: routes.hardware, title: T('Hardware and models'), description: 'Verified silicon, targets in bring up, and every model recipe we ship, generated from the repository.', faq: 'hardware', priority: 0.8 },
  { path: routes.benchmarks, title: T('Benchmarks'), description: 'The concurrency ladder against vLLM and every gate record, generated from the repository. Reproduce any of them.', faq: 'benchmarks', priority: 0.9 },
  { path: routes.solutions, title: T('Solutions'), description: 'Built for the people who own the GPUs. Neoclouds, enterprises, banks, hospitals, government, law firms, hyperscalers, labs and SMB.', priority: 0.8 },
  ...industries.map((i) => ({ path: solutionHref(i.slug), title: T(i.name), description: `Avarok for ${lower(i.name)}. Faster inference, stronger governance and a payback the CFO can read, on hardware you own.`, priority: 0.7 })),
  { path: routes.pricing, title: T('Pricing'), description: 'Priced against productive GPU capacity, not seats. Community, workstation, enterprise and proof of value, with a payback model you can edit.', faq: 'pricing', priority: 0.9 },
  { path: routes.demo, title: T('Book a demo'), description: 'A working session on your workload. The console on demo data, the published ladder, the payback model with your inputs, and a scoped proof of value.', priority: 0.9 },
  { path: routes.broll, title: T('B-roll'), description: 'Procedural ambient loops drawn from the brand palette for the product videos. The media pipeline records this page. Nothing links here.', priority: 0.1, noindex: true, sitemap: false },
  { path: routes.resources, title: T('Resources'), description: 'Blog, documentation, product updates, benchmarks, open source, contributors, events and Avarok Labs.', priority: 0.7 },
  { path: routes.updates, title: T('Product updates'), description: 'What shipped, rendered from the repository changelog on every build.', priority: 0.7 },
  { path: routes.events, title: T('Events'), description: 'Where to meet the Avarok team, in person and online.', priority: 0.5 },
  { path: routes.contributors, title: T('Contributors'), description: 'Everyone who has landed code in the Avarok repository, called out by name.', priority: 0.5 },
  { path: routes.labs, title: T('Avarok Labs'), description: 'The research arm. Kernels, compression, speculative decoding, memory, compilers, protocols, agentic benchmarks and day zero model bring ups.', priority: 0.7 },
  { path: routes.company, title: T('About'), description: 'It started with two words. The story, the mission and the principles behind Avarok.', priority: 0.7 },
  { path: routes.careers, title: T('Careers'), description: 'Build the layer between the GPU and the invoice. Open roles on the founding team.', priority: 0.6 },
  { path: routes.contact, title: T('Contact'), description: 'Sales, technical, partnerships, security and press. Every path lands with a founder.', priority: 0.7 },
  { path: routes.trust, title: T('Trust center'), description: 'Architecture, data handling, assurance and licensing. What we run, what we claim, and what we do not.', priority: 0.7 },
  { path: '/404', title: T('Not found'), description: 'That page is not here. The front page has the pitch, the benchmarks have the numbers, and the open source page has the install command.', noindex: true, sitemap: false }
];

export const pageFor = (path) => pages.find((p) => p.path === path.replace(/\.html$/, '').replace(/\/$/, '') || (path === '/' && p.path === '/'));
