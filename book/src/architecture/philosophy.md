# Philosophy: AI Kernel HyperCompiling

Part I's Philosophy chapter answered *why* Atlas specializes. This chapter answers *how the specialization thesis forces specific design choices in the code*, and what you should see — or, if you're writing a PR, what you should preserve — when you read the codebase.

The single design rule that every choice below derives from is:

> **Specialization is a directory, not a template.** Everything that varies across `(Hardware, Model_q)` targets lives in its own directory. Everything shared across them is abstraction *above* the kernel layer, not parameters *inside* one.

## Consequence 1: the kernel tree is a coordinate system

```
kernels/
  gb10/                               # Hardware
    HARDWARE.toml                     # vendor, arch, memory specs
    common/
      KERNEL.toml
      *.cu                            # the shared baseline — 160 files today
    qwen3-next-80b-a3b/               # Model
      MODEL.toml                      # layer counts, sampling defaults
      nvfp4/                          # Quantization
        KERNEL.toml                   # compiler flags
        *.cu                          # only the files this target overrides
```

Three levels of directory, over a `common/` baseline. The shape is deliberate: a
leaf directory owns *exactly one* `(H, M_q)` target, and a leaf file **shadows**
its same-stem namesake in `common/` (`atlas-kernels/build.rs::collect_cu_files`).
Shadowing is whole-file, not per-symbol.

So a leaf holds *only its divergences*, not a full kernel set — `qwen3.6-27b/nvfp4`
carries 11 `.cu`, `qwen3.6-35b-a3b/nvfp4` carries 5, `qwen3-next-80b-a3b/nvfp4`
carries 3, over the 160 in `common/`. Two leaves therefore **do** share source for
everything neither of them overrides; what is guaranteed is that where a target
*does* diverge, it diverges in a file nothing else compiles. When we say "Atlas
ships N targets," we mean N independent leaves — 22 of them on GB10 today.

The corollary is a real failure class: because shadowing is whole-file, a shadow
that has quietly become identical to `common/` overrides nothing while masking
every later `common/` improvement, and two models keeping private byte-identical
copies of the same shadow will drift apart. CI's `kernel-structure` job
(`scripts/check_kernel_shadows.py`) rejects both; the sanctioned way to share one
file between leaves is a relative symlink.

The three `.toml` files are the only metadata the build system consumes. `HARDWARE.toml` tells `atlas-kernels/build.rs` which `ComputeTarget` impl to use (nvidia, amd, apple, intel) and what arch flag to pass the compiler. `MODEL.toml` is the per-model behavior SSOT — sampling presets, thinking budgets, tool-call parser defaults. `KERNEL.toml` overrides compiler flags and module names.

Adding a model or a hardware target is, at the file-system level, *creating a new directory*. No code elsewhere in the repository needs to move.

## Consequence 2: the runtime crate structure mirrors the axis split

Read the workspace `Cargo.toml` and you'll see nineteen workspace members. Group them by what axis of variation they insulate:

| Axis they insulate | Crates |
|---|---|
| *Hardware vendor* | `atlas-core` (`ComputeTarget`, `Vendor` enum, `KernelTarget`), `spark-runtime` (`GpuBackend`), `spark-comm` (`CommBackend`) |
| *Model architecture* | `spark-model` (`ModelWeightLoader` trait, `TransformerLayer` trait, per-family loaders) |
| *Quantization format* | `spark-model/src/quant_format/` (per-format modules + runtime dispatch), `atlas-core/src/numeric.rs` (host-side FP8/BF16 conversions) |
| *Compiled kernels (one artifact per axis combination)* | `atlas-kernels` (embedded PTX modules, auto-generated from the kernel tree) |
| *Request serving* | `spark-server` (HTTP, tokenizer, tool parsing) |
| *Measurement* | `atlas-spark-bench` |

Each crate has exactly one reason to change. A new GPU vendor never touches `spark-model`. A new model family never touches `spark-runtime`. A new quantization scheme touches `spark-model`'s format modules and `atlas-kernels`, but not the layer code. This orthogonality is not a happy accident of the crate layout — it *is* the architectural consequence of the specialization thesis.

## Consequence 3: SBIO — business logic never touches I/O

Business logic — the layer code in `spark-model`, the scheduler in `spark-server` — never calls CUDA APIs, never opens a socket, never reads a file. Every such operation goes through a trait:

- GPU memory, launches, graphs → `GpuBackend`
- Collective comms → `CommBackend`
- Weight loading I/O → `WeightStore` (wraps safetensors + `O_DIRECT`)
- HTTP responses → `axum` handlers, tested against a mock channel

This is what the user instructions call **SBIO** (Separation of Business logic from I/O). The payoff is that 80%+ of the codebase is unit-testable without a GPU. `MockGpuBackend` records launches but does not execute them. `SingleGpuBackend` is a no-op `CommBackend` for single-GPU runs. The [SBIO chapter](./sbio.md) shows the pattern in detail.

## Consequence 4: zero runtime compilation

Every general-purpose framework has, somewhere, a codepath that compiles kernels at runtime. PyTorch has `torch.compile`. vLLM has Triton JIT. TensorRT-LLM has TRT engine builds. Each of those is a slow path the first time you hit a new shape, and an ongoing operational surface the ops team has to manage (cache directories, warm-up scripts, cold-start budgets).

Atlas has none of it. `atlas-kernels/build.rs` enumerates every `(H, M_q)` target matching the `ATLAS_TARGET_*` env vars, compiles every `.cu` file for every matching target, and emits one auto-generated `target_ptx.rs` that is `include!`'d into the crate. The release binary contains every PTX module we ship. Startup is "mmap the binary, upload PTX to the GPU, capture CUDA graphs for a handful of batch sizes, done".

This is what "embedded in the binary" means throughout the book. It is the concrete mechanism by which specialization does not cost operator pain.

## Consequence 5: one binary per installation, N kernel sets

You deploy one Docker image. It contains one `spark-server` binary. It contains 22 (today) `(gb10, model, quant)` PTX sets embedded in that binary. At startup, the binary reads the model's `config.json`, computes the canonical `model_type`, looks up the matching `KernelTarget`, and uses that set.

The knobs that let this scale:

- **`ATLAS_TARGET_*=*` at build time** — compiles every matching target. The default image sets everything to `*` and ships the lot.
- **`ATLAS_TARGET_HW=gb10 ATLAS_TARGET_MODEL=qwen3.5-35b-a3b ATLAS_TARGET_QUANT=nvfp4`** — compiles exactly one target. Used for per-model slim images in `docker/gb10/<model>/`.
- **`ATLAS_SKIP_BUILD=1`** — emits a stub `target_ptx.rs` so that `clippy`, `fmt`, `check`, and any non-GPU test can run on a vanilla Linux host.

The same image works across all supported targets. The startup dispatcher picks the right kernels. Operators don't manage a kernel cache; they don't warm a JIT; they don't think about it.

## What the rest of this book is about

Every subsequent chapter is an elaboration of one of the consequences above. The [workspace layout chapter](./workspace.md) walks the directory tree. The [dispatch chapter](./dispatch.md) traces a single request from HTTP to kernel launch. The [SBIO chapter](./sbio.md) shows how the testability claim actually holds.

The deep-dive chapters in Part IV show what the kernels look like — what a hand-tuned kernel set per target buys you, and how you'd write new ones when you're porting Atlas to your own `(H, M_q)` target.

## Reading the architecture categorically

The design choices above have precise names in category theory. The target set `𝒯 = Hw × Mod × Quant` is a categorical **product**; the crate split is that product made syntactically real, which is why orthogonality of axes is a structural fact and not a convention. The kernel registry is a **coproduct** (disjoint union of per-target PTX sets), which is why adding a summand cannot regress existing summands. The `GpuBackend` trait defines an **algebraic theory** with two ship-worthy models — `AtlasCudaBackend` and `MockGpuBackend` — and that is what makes the test suite runnable without a GPU. A general framework is, in this vocabulary, an engine that factors `Kernels : 𝒯 → 𝐒𝐞𝐭` through a smaller "essence" category; Atlas refuses the factoring, and the 3.6× gap against vLLM is the cost of the factoring that Atlas does not pay.

The appendix [A Category-Theoretic Perspective](../appendix/category-theory.md) works through each of these structures at appendix length. It is a design reference, not a prerequisite.
