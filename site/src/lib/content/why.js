// =============================================================================
// /why-avarok. The lead page a buyer reads after the pitch. The same three
// pillars, each with what it is, how it is built, why it changes the
// outcome and why it is hard to copy, then the chain, the delivery path and
// the proof of value.
// =============================================================================
import { routes, links, company } from './brand.js';

export const whyHero = {
  eyebrow: 'Why Avarok',
  title: 'You invested in the datacenter. Now get the most out of it.',
  lede:
    'Avarok models your inference economics first, then runs, governs and measures every workload against that model directly. A fundamentally different approach to getting more inference, and more accountability, from silicon you already own.',
  primary: { text: 'Book a demo', href: routes.demo },
  secondary: { text: 'Watch the console', href: `${routes.home}#tour` },
  pillars: [
    { name: 'Speed', color: 'violet', body: 'Compiled per hardware, model and quantization. More tokens per GPU, and the curve keeps climbing where agent fleets run.' },
    { name: 'Security', color: 'cyan', body: 'One signed Rust binary, no interpreter in the request path, nothing leaves your perimeter.' },
    { name: 'Governance', color: 'green', body: 'Every token has a receipt. Every GPU hour has an owner. Finance can sign it.' }
  ]
};

export const pillars = [
  {
    n: '01',
    name: 'Speed',
    color: 'violet',
    title: 'The same GPUs produce more inference, and the curve holds where it matters.',
    lede: 'Generic runtimes ship one code path for every accelerator and tune it in Python. Avarok compiles the path for the silicon in front of it.',
    blocks: [
      {
        h: 'What it is',
        p: 'A pure Rust and CUDA inference engine with hand tuned kernels per hardware, model and quantization target, NVFP4 and FP8 quantization, speculative decoding with MTP draft heads and DFlash block diffusion, radix tree prefix caching and expert parallelism across nodes.'
      },
      {
        h: 'How it is built',
        p: 'Every kernel target has a compatibility manifest and a gate. A release ships only after the serve matrix passes on the real box, and a release that ships slower than its committed baseline fails the gate. Numbers on this site are generated from those records, never typed.'
      },
      {
        h: 'Why it changes the outcome',
        p: 'Agentic work arrives as fleets of tool calling agents sharing a system prompt, not one conversation at a time. On the published GB10 ladder Avarok wins every rung from C=1 to C=128 and keeps scaling at the top while the matched vLLM configuration flattens. That gap is how many agents one box can run, and it compounds across a rack.'
      },
      {
        h: 'Why it is hard to build',
        p: 'It is a decision made at the start, not a feature added later. Leaving the Python and PyTorch ecosystem meant rewriting the serving stack from scratch in Rust and running an AI first repository where every kernel change must carry a certified benchmark before it merges.'
      }
    ],
    question: 'Ask for the throughput curve at C=128 on your workload, not a single stream number on theirs.',
    proofHref: routes.benchmarks,
    proofText: 'Read the ladder and every gate record'
  },
  {
    n: '02',
    name: 'Security',
    color: 'cyan',
    title: 'Nothing leaves your perimeter, and there is nothing in the request path you did not sign.',
    lede: 'A serving stack that pulls two hundred dependencies at startup is a supply chain you did not audit. Avarok is one binary.',
    blocks: [
      {
        h: 'What it is',
        p: 'One signed binary of about 75 MB with no Python, no PyTorch and no runtime compilation. Recipes, models and kernels arrive as signed artifacts. Node to node links are designed for a hostile network, so the mesh can span racks, sites and edges you already have with no private backbone.'
      },
      {
        h: 'How it is built',
        p: 'Cargo deny audits every dependency on every pull request. Kernel targets are content hashed down to their transitive include closure, so a build can prove what it was compiled from. Release images promote by digest from staging to canary to production, and are never rebuilt between environments.'
      },
      {
        h: 'Why it changes the outcome',
        p: 'Regulated buyers cannot move at hyperscaler speed because every deployment re proves its own compliance. Pinned, signed, repeatable deployment primitives turn that overhead from a blocker into the reason to buy. Prompts, weights and telemetry stay where the policy says they stay.'
      },
      {
        h: 'Why it is hard to build',
        p: 'Security is a property of the request path, not a feature beside it. It is only cheap when the engine, the control plane and the deployment primitives were designed by the same people at the same time, which is the team Avarok started with.'
      }
    ],
    question: 'Ask what is in the request path, and who audits the two hundred dependencies behind it.',
    proofHref: routes.security,
    proofText: 'Read the security posture'
  },
  {
    n: '03',
    name: 'Governance',
    color: 'green',
    title: 'Every token has a receipt. Every GPU hour has an owner.',
    lede: 'The industry measures inference in tokens per second. Enterprises pay for it in dollars per workload. Avarok owns the layer between the two.',
    blocks: [
      {
        h: 'What it is',
        p: 'A correlation layer that ties each unit of work to the workload, model, runtime, configuration, GPU and cluster that produced it, then a ledger that reports cost per million tokens, cost per successful workload at SLO, productive GPU hours, stranded capacity and savings against the baseline, by business unit.'
      },
      {
        h: 'How it is built',
        p: 'The engine owns the request path, so utilization, cache behaviour and queue pressure are measured at the source rather than inferred from a proxy. The control plane records every rollout, canary, rollback and policy decision with a signed recipe and a gate record attached.'
      },
      {
        h: 'Why it changes the outcome',
        p: 'The decision maker walks to the CFO with a payback period rather than a percentage. Everything after payback is upside, and every month it is recomputed from production data rather than a vendor estimate.'
      },
      {
        h: 'Why it is hard to build',
        p: 'You cannot attribute cost to a workload you did not serve. Platforms that sit beside the engine can only estimate. Avarok is under the workload, which is the only place the ledger can be exact.'
      }
    ],
    question: 'Ask what a workload cost last Tuesday, by business unit. A tokens per second chart is not an answer.',
    proofHref: routes.economics,
    proofText: `See ${company.economics}`
  }
];

export const pov = {
  eyebrow: 'Next step',
  title: 'Run the tests against your own fleet.',
  body: 'A proof of value runs four weeks against success criteria you set. Not a scripted demo on someone else’s hardware.',
  bullets: [
    'Four weeks, start to finish',
    'A side by side ladder on your workload in week one',
    'Your hardware, your models, your criteria',
    'We supply five criteria if you do not have your own'
  ],
  primary: { text: 'Book a demo', href: routes.demo },
  secondary: { text: 'Read the deployment guide', href: links.guide, external: true }
};
