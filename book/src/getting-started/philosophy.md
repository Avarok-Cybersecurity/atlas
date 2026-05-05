# Philosophy

Atlas is built for **broad hardware and model support** — the kind of matrix that today includes GB10, tomorrow includes H100, B200, Apple Silicon, AMD MI300X and Intel GPUs, and a long tail of model architectures on top of each. The question the project starts from is the one every inference engine eventually has to answer:

> How do you cover a large matrix of `(Hardware, Model_q)` targets *and* run each one at the hardware's theoretical peak?

Existing general-purpose engines answer by absorbing genericity into the kernel: templated CUDA, JIT compilation, runtime shape branching. Atlas answers the opposite way — **specialize per target, but design the abstractions so that specialization can scale**.

This is our version of **AI Kernel HyperCompiling**. Every chapter of this book is a consequence of taking that answer seriously.

## The trade general frameworks make

vLLM, TensorRT-LLM, SGLang, and the other mainstream engines support thousands of models across dozens of GPU generations with a single binary. That is a real, useful thing. The cost of doing it well — a cost they pay every release — is a layer of abstraction *between* the kernel and the hardware: a templating engine, a just-in-time compiler, or a broadly-parameterized CUDA kernel that branches on shape, dtype, and arch.

Every one of those layers trims a few percent off peak throughput. Added up, on a specific hybrid-SSM/attention/MoE model running on a specific GPU, those trims are how a 3.6× gap opens up.

Atlas does not try to close that gap inside a general framework. We reject the framing — and we refuse to let specialization shrink the scope of what we support. Both at once.

## Abstractions designed for many targets

The way Atlas gets broad support is by putting the genericity *above* the kernel layer, not inside it. Three traits do the load-bearing work:

- **`ComputeTarget`** (`atlas-core/src/compute.rs`) — the *build-time* trait. Given a hardware vendor and an architecture flag, it knows how to invoke the right compiler (`nvcc`, `hipcc`, `xcrun metal`, …) to turn a source file into a binary module. Adding a new hardware vendor is one `impl ComputeTarget`.
- **`GpuBackend`** (`spark-runtime/src/gpu.rs`) — the *runtime* trait. 27 methods cover memory, kernel launch, streams, events, and graphs. The model code, the scheduler, the HTTP server — none of them know whether they're running on CUDA, Metal, or HIP. They hold a `&dyn GpuBackend`. A new backend is one `impl GpuBackend`.
- **`CommBackend`** (`spark-comm/src/lib.rs`) — the multi-GPU trait. `all_reduce`, `all_gather`, `reduce_scatter`, broadcast. NCCL ships today; HIP's RCCL and Metal's MPS collective ops would drop into the same shape.

The `Vendor` enum in `atlas-core` already enumerates `Nvidia`, `Amd`, `Apple`, `Intel`. Nothing about the engine above the trait layer is NVIDIA-specific. The multi-vendor design is in the code today; the first `(hw, model, quant)` set we happened to ship was for GB10 because that's the hardware on our desks. Porting to H100, B200, MI300X, M4 Ultra, or Arc A770 means implementing two traits and writing the kernel source — the full walkthrough is in the [Adding a new hardware target](https://github.com/Avarok-Cybersecurity/atlas/blob/main/README.md#adding-a-new-hardware-target) guide.

## How specialization scales

The question this framing forces — and the question the rest of the codebase answers — is: *how do you keep adding targets without turning into the general framework you rejected?*

The Atlas answer has two parts:

1. **Hyperoptimize in isolation.** Every `(H, M_q)` target lives in its own directory: `kernels/<hw>/<model>/<quant>/`. The kernels there can use any tiling strategy, any shared-memory layout, any MMA instruction mix — they cannot accidentally slow down another target because they share no code with one. This is the opposite of the templating approach: instead of one kernel that branches, we have N kernels that each do exactly one thing. Adding a new target is a new directory; it is *physically impossible* for it to regress an existing one.
2. **Share abstractions, not kernels.** What *is* shared — the `GpuBackend` trait, the `ComputeTarget` build-time trait, the layer factory in `spark-model` — is abstraction *above* the kernel level. The shared code knows about "launch a kernel" and "allocate GPU memory"; it does not know, and does not want to know, what the kernel inside is doing. New hardware plugs in at the trait layer; new models plug in at the `ModelWeightLoader` trait — neither disturbs the other axis.

Adding a new model works the same way. One new `ModelWeightLoader` impl, one match arm in `spark-model/src/factory.rs`. The KV cache, buffer arena, scheduler, and HTTP server are model-agnostic — they do not need to change to support a fundamentally different architecture. Qwen3.5 hybrid SSM+attention+MoE, Nemotron-H Mamba-2, MiniMax 256-expert sigmoid MoE, Gemma-4 sliding+full alternating attention, Qwen3-VL vision all coexist in one binary today, sharing zero kernel code with each other.

## Why specialization is finally practical

Writing specialized kernels *was* prohibitively expensive. One-off work by a human CUDA engineer, non-transferable, bit-rots as the hardware changes. The reason general frameworks won for a decade is that the specialist approach had no path to scale — you could not afford to write a new kernel for every GPU generation and every quantization scheme.

That has changed. AI-assisted kernel engineering — profiling a kernel, proposing tiling experiments, verifying correctness against a reference — is now good enough that we can dedicate real effort to every target. One reason Atlas exists is to prove this at scale: every kernel in the repo is AI-written, human-reviewed, benchmark-verified. We explicitly want new PRs to be AI-generated. Human-only contributions get reviewed by AI.

The specialization thesis does not require a superhuman human sitting behind every kernel. It requires a pipeline where specializing is cheap enough to do twelve times, then fifty, then a hundred — across hardware *and* across models.

## What this means for you

- **If you're an operator** — expect Atlas to be fast on every target we've shipped. The matrix grows with each release; if your model or GPU is not in the matrix yet, it is a well-scoped piece of work to add it, not an architectural impossibility.
- **If you're a model author** — the trait you need to implement is `ModelWeightLoader`. Everything downstream is model-agnostic and will not change when you add a new architecture. See [spark-model](../crates/spark-model.md).
- **If you're a kernel engineer** — the directory you'll live in is `kernels/<hw>/<model>/<quant>/`. The abstractions above you exist to stay out of your way. Your job is to make that one `(H, M_q)` tuple run as close to silicon peak as you can. See [CUDA Kernel Engineering](../deep-dives/kernels.md).
- **If you're a hardware vendor** — the abstractions that let a new GPU family plug in are two traits (`ComputeTarget`, `GpuBackend`) plus kernel source. See [Adding a new hardware target](https://github.com/Avarok-Cybersecurity/atlas/blob/main/README.md#adding-a-new-hardware-target) in the README.

The next chapter gets you running on the hardware we've shipped. The rest of the book earns the claim that specialization, done with the right abstractions, scales — across GPUs, across models, and across quantization schemes — without giving up a single percent of peak throughput.

## A formal lens

If you want the same argument in the language of category theory — the target matrix as a product, the kernel registry as a coproduct, the trait layer as an algebraic theory, general frameworks as a factoring Atlas refuses — see the appendix [A Category-Theoretic Perspective](../appendix/category-theory.md). It is optional reading; nothing else in the book depends on it.
