---
title: Seven Tenets Powering Atlas Inference Accelerated Workloads
dek: Atlas Inference is a free and open source LLM inference engine written from scratch in Rust. These are the seven philosophical tenets we started it on, and why we left the Python vLLM stack to do it.
categories: [engineering, design]
date: 2026-08-31
keywords: [rust inference engine, vllm alternative, open source llm inference, monorepo, ai friendly repository, hardware specific kernels, sbio, atlas inference]
og-image: /images/og/seven-tenets-powering-atlas-inference.webp
author: thomas-braun
draft: false
---
_Special note to readers: this is our first post. As a long-time writer, before AI became notorious for using em dashes, I wrote at great length on topics ranging from philosophy, programming, theology, cybernetics, and more. As a writer, I use em dashes relatively frequently to help make sentences flow better and draw attention to nuance. If something is AI-generated, we will let you know. We believe the human element is special; the spirit of writing ought not to be rendered vanquished by AI._

**[Atlas Inference](https://atlasinference.io)**

![A polished silicon wafer leaning against a block of rusted iron.](/images/posts/seven-tenets-powering-atlas-inference/00-rust-silicon.webp)

_Images generated with AI (ironically)._

Earlier this year, we began developing an inference engine in order to take an ecosystem — often widely fragmented — from a proof of concept to a professional software product. Given that many data scientists and researchers worked on vLLM, naturally, they used the language they were most _comfortable_ with: Python. We applaud their contribution to the world, proving we can run AI almost anywhere (given sufficient hardware resources). Given the ecosystem is external-dependency heavy, as well as lacking the rigor of professional software architecture practices, we decided to start from scratch, embracing several philosophical tenets.

**Seven Tenets Powering Atlas Inference Accelerated Workloads**

_The First as a Note on Philosophy_

![A cut of rock showing its layers, the darkest one at the bottom.](/images/posts/seven-tenets-powering-atlas-inference/01-strata.webp)

Philosophy is a "lowest level" first principle of abstractive reasoning: it underpins biology, chemistry, physics, mathematics, logic, and beyond. Given we were working at the crossroads of AI and software engineering, the first philosophical tenet we decided to embrace was to make our inference engine **free and open source**, _always_. We believe the best software, similar to Linux, should be open to iterations of public inspection, community improvement, and, enable modification to let both the author and community learn rapidly about the end-users' needs.

![A cleaved quartz crystal with the lattice inside it visible.](/images/posts/seven-tenets-powering-atlas-inference/02-crystal.webp)

The second tenet is to be **Community First**. A project without a community is just a pet project. This is not to say pet projects are negative, but rather, that these so-called "pet projects" are, by themselves, providing little to no external value to the world other than the maturation of the author. As developers, many of you have numerous pet projects that can be considered the product of your craft and hobby, and indeed, the skills you gain from creating such projects strengthen you, and in turn provide value by proxy to the external world. However, if the goal is to make excellent software supported and driven by a community, then, efforts must be made to make the software accessible to the community where free exchange increases the inherent limits of each party. Without community, we limit the potential of software to ourselves.

![A long table of notebooks and cups around one core sample, nobody in the frame.](/images/posts/seven-tenets-powering-atlas-inference/03-table.webp)

_The Third as a Standards of Practice_

The third tenet is to ensure the repository is, by design, a monorepo. By taking the monorepo approach in the age of AI, agents do not have to submit downstream PRs and wait for another to merge. Furthermore, given agents now frequently have long context windows that can absorb an entire codebase or effectively index it, having knowledge all in one place rather than scattered in a dependency web allows for fast iteration. Indeed, compilation time and building a docker image takes only a minute or two, whereas with vLLM, around 40+ minutes.

![One tree trunk and its whole root system held in a single volume of earth.](/images/posts/seven-tenets-powering-atlas-inference/04-roots.webp)

Why would this be important? Well, last October, when the DGX Sparks were released, I spent months trying to improve vLLM. The time it took to observe a kernel's effect on the accuracy and performance was simply too long to make for an effective way to improve kernels fast enough. By abandoning the data-scientist ecosystem around December/January, and, starting over from scratch in my native language (i.e., Rust), I was able to very quickly iterate and improve kernels. Months later, research articles came out on AI-generated kernels using self-improving iterative loops. Recursion. The realm of academia — in part — had caught up to where we, at Atlas Cybernetics, were at.

_The fourth tenet_ is to design kernels to be **specific per hardware per model**. While many kernels may be re-used across the same hardware set for various models, designing the kernel selection mechanism to allow one kernel to _shadow_ another... enabled optimization techniques specific to one model's architecture. Squeezing out more performance like this gets you more data center per data center.

![Glass plates set at an offset so each one is seen through the next.](/images/posts/seven-tenets-powering-atlas-inference/05-glass.webp)

_The Fifth as Embracing the Tsunami_

The fifth tenet is being an **AI-friendly repo**. While some developers bicker about whether or not to use AI, we simply moved on and embraced the future. This choice came with challenges that make hand-written code look like a walk in the park. We wanted to prove to the world that AI could lower the barrier to entry (for making meaningful contributions) to one of the most technically challenging fields in software and data science. With the community's help, we succeeded. Managing an increased amount of nondeterminism was an engineering challenge overcome by strict PR checks beyond the common PR checks found in most open source repositories.

For example, if code for a kernel is changed, we mandate that multiple benchmarks be run and certified before allowing a merge. But, if code for just a website page is changed, we don't mandate benchmark certification. We had to design a taxonomic system whereby the nodes in the taxonomy tree — each representing a category of the code — implied an associated benchmark. Then, using static code analysis, our PR gate generates the taxonomy tree, looks at the code changes, maps the taxonomy tree to a subset tree, and then takes this subset tree and aggregates each remaining node's associated benchmarks. Finally, the PR cannot pass until the latest commit provides a certified run of each required benchmark. There are more components than this benchmarking system, but, this gives us an idea of how we had to design the AI-repo in such a way that we gracefully allow and guide chaos rather than downright rejecting it.

![A single beam split by a prism, each path continuing through its own filter.](/images/posts/seven-tenets-powering-atlas-inference/06-prism.webp)

_The sixth tenet_ flips the legacy paradigm on its head. Before it is declared, I want to make one thing clear: we are not anti-human. **We require AI-generated PRs**. If a human manually adds code (i.e., "legacy coding"), they must explain why they believe they needed to do so, and, why they are better than the AI. The process is not meant to disparage humans; this is to collect data on the remaining gaps between AI and humans. With the right controls, we can safely store each gap in LatticeDB (i.e., a WASM-compatible hybrid graph/vector DB) and later analyze the data to help improve LLMs with our partners. Human authoring is a fundamental point of purpose and we see this as elevating the contribution in a sea of abstracted (but very sophisticated) agent contributions.

![A lattice of nodes, most of them dark, a few lit gold.](/images/posts/seven-tenets-powering-atlas-inference/07-lattice.webp)

_The Seventh as Pipelining Improvement_

The seventh and final tenet is taking on an **abstraction-first modular design** set-based solution space. After one writes code for many years, one may begin to see underlying patterns that can be generalized and abstracted-away using meta-narratives (e.g., interfaces, traits, etc). By declaring abstractions, you set boundary conditions and discover required pattern languages that emerge for your application, helping guide existing and future code additions in the right direction.

A concrete example, I actively apply what I uncovered and now call the "separation of business logic and I/O" principle, or SBIO. By keeping your business logic separate from your method of I/O, you can easily test business logic by hot-swapping the I/O method, which is also useful when one is programming an application that is WASM-compatible. Relevantly, this approach creates a repository that makes it easier for an AI to understand, as well as compel it to "fall in line" with the patterns of the code. Given the occasional tendency for AI to go "off the rails" (an issue which is now much less common than it used to be even just a year ago), by laying the boundaries of the track, the train will stay on the rails. Paradoxically, you get more out than is taken away by a given restriction: a key property of open systems in a closed loop.

![Two copper circuit planes held apart with air between them.](/images/posts/seven-tenets-powering-atlas-inference/08-layers.webp)

Overall, given the philosophical foundation we have embraced, experienced, struggled, improved, and use at scale to safely merge many PRs, coupled with the beautiful nurture and guidance from the community, we have succeeded. From here on out, we can apply the very foundations highlighted herein to make future AI repositories, iterate, and continually improve. We appreciate the community, and look forward to the next large unveil of a product the world has not yet imagined in its entirety!
