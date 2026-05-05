# A Category-Theoretic Perspective

The Atlas book argues its case in prose. The prose carries the claim: for every `(Hardware, Model, Quantization)` target, there exists a kernel configuration that runs at the hardware's theoretical peak; general frameworks cannot reach that peak because they pay a genericity tax; Atlas refuses the tax by specializing per target while keeping abstractions *above* the kernel layer.

Category theory gives precise names for the structures that claim leans on. This appendix names them. It is not a proof of performance, not a tutorial in category theory, and not required reading for anyone wanting to run or extend Atlas. It is a lens. Read it if you want to see the same design with fewer words.

Standard references for the underlying mathematics: Saunders Mac Lane, *Categories for the Working Mathematician* (second edition); Emily Riehl, *Category Theory in Context* (freely available). Everything below uses only the first two chapters of either.

---

## 1. The target category `𝒯`

A **category** is a collection of objects together with arrows (morphisms) between them, closed under composition and equipped with an identity arrow on every object. In symbols: `ob(𝒯)` is a class, and for every ordered pair `A, B ∈ ob(𝒯)` there is a set `𝒯(A, B)` of arrows.

Atlas's target category `𝒯` has one object per supported `(H, M, q)` triple. In code, these objects are `atlas_core::target::KernelTarget` values — `GB10_QWEN35_NVFP4`, `GB10_QWEN3_NVFP4`, `GB10_QWEN35_122B_NVFP4`, and nine siblings. The `const` declarations in `crates/atlas-core/src/target.rs` are a literal list of `ob(𝒯)`.

The non-obvious choice is the morphism set: **for every distinct pair `A ≠ B`, `𝒯(A, B) = ∅`**. The only arrows are identities. `𝒯` is a *discrete* category.

This choice matters. A non-identity arrow `f : A → B` would mean "a canonical way to go from kernel set `A` to kernel set `B`" — a declared compatibility. Such compatibilities are temptations that collapse specialization: the moment you posit `f : (GB10, Qwen3.5-35B, NVFP4) → (GB10, Qwen3-Next-80B, NVFP4)`, you have committed to a kernel set that serves both, or at least to a shared essence that both factor through. That is the shape of vLLM. Atlas refuses by making `𝒯` discrete.

The specialization thesis, in one sentence: **`𝒯` is discrete, and all performance claims are local to an object**.

## 2. `𝒯` as a product

The three axes decompose: `𝒯 ≅ Hw × Mod × Quant`, where each factor is itself a discrete category (one object per supported value). The product comes with projection functors — `π_Hw : 𝒯 → Hw`, `π_Mod : 𝒯 → Mod`, `π_Quant : 𝒯 → Quant` — that read off one coordinate.

A **functor** is a structure-preserving map between categories: it sends objects to objects and arrows to arrows, respecting identities and composition. On discrete categories a functor is just an object-to-object function.

The product decomposition is visible in three places in the repo:

- The **directory tree**: `kernels/<hw>/<model>/<quant>/` mirrors the three-factor product exactly. A leaf is an object of `𝒯`.
- The **build-time wildcards**: `ATLAS_TARGET_HW`, `ATLAS_TARGET_MODEL`, `ATLAS_TARGET_QUANT` in `atlas-kernels/build.rs` select subsets of each factor independently.
- The **workspace crate split**: `spark-runtime`/`spark-comm` insulate the Hw axis, `spark-model` insulates the Mod axis, `atlas-quant` + `atlas-kernels` insulate the Quant axis.

Orthogonality of axes is not a lucky accident — it is the defining property of a categorical product. Adding an object to `Hw` does not touch `Mod × Quant`; the projection `π_{Mod×Quant}` is unchanged. This is exactly the empirical fact that "adding a new hardware vendor is two trait impls and a directory".

## 3. Kernels as a functor

The primary structure over `𝒯` is the kernel assignment:

```text
Kernels : 𝒯 → 𝐒𝐞𝐭
```

`𝐒𝐞𝐭` is the category of sets. `Kernels` sends each target to its set of compiled PTX modules. The auto-generated file `atlas-kernels/src/target_ptx.rs` is this functor materialised in code. `ptx_modules(target: &KernelTarget) -> Option<&'static [PtxModule]>` is the functor applied to an object.

Because `𝒯` is discrete, there are no naturality squares to draw — `Kernels` has complete freedom per object, which is the whole point. The image `Kernels(H, M, q)` in the default multi-model image has ~30–40 elements; no two targets share an element by construction.

## 4. Build-to-runtime as a composition of functors

Three categories and two functors sit in a line:

```text
Sources  ──[ComputeTarget.compile]──►  Binaries  ──[embed + load]──►  KernelHandles
```

`Sources` has one object per `(H, M, q)` leaf directory whose underlying data is the set of `.cu` / `.metal` / `.hip` files inside. `Binaries` has one object per leaf whose underlying data is the set of compiled PTX / metallib / HSACO byte blobs. `KernelHandles` holds the runtime-resident entries returned by `GpuBackend::kernel(module, function)`.

The first arrow is the `ComputeTarget` trait in `crates/atlas-core/src/compute.rs`. It is a **vendor-indexed family of functors** — one concrete functor per `Vendor` (`Nvidia`, `Amd`, `Apple`, `Intel`). Adding a new hardware vendor means adding a new member to the family. The rest of the diagram commutes unchanged: `Binaries → KernelHandles` doesn't care how the binaries were produced.

This is the categorical reading of "the abstractions sit above the kernel layer, not inside it". The abstractions *are* the arrows in the diagram. The kernels *are* elements of the objects. Arrows and elements live at different levels; only arrows need to be generic.

## 5. The `GpuBackend` trait as an algebraic theory

An **algebraic theory** is a signature (operation symbols with arities) plus equations that any implementation must satisfy. A **model** of the theory is a set together with operations that satisfy the equations. Different sets can be different models of the same theory — this is the mathematical name for "multiple implementations of the same trait".

The `GpuBackend` trait in `crates/spark-runtime/src/gpu.rs` is such a theory. Its operations are `alloc`, `free`, `kernel`, `launch`, `synchronize`, `copy_h2d`, and so on (27 methods). The (unwritten, but real) equations include "`free` after `alloc` returns memory to the pool", "`synchronize` serialises previously-launched work on the given stream", and "`launch` of a kernel with pointer arguments passes the addresses unchanged to the kernel".

Two models ship:

- `AtlasCudaBackend` — implements the theory by delegating to the CUDA driver API.
- `MockGpuBackend` — implements the theory by recording launches and returning the opaque successes the equations demand.

The business logic — scheduler, engine, layer code — is **polymorphic over the choice of model**. In category-theoretic language, business logic is an arrow in the category of `GpuBackend`-algebras, and evaluating it requires picking a model. The `cargo test` suite evaluates in `MockGpuBackend`; production evaluates in `AtlasCudaBackend`. Both evaluations agree on all facts that depend only on the algebraic theory — sequence of launches, argument correctness, allocation hygiene. This is why ~80% of the test surface runs without a GPU.

This is the formal meaning of [SBIO](../architecture/sbio.md): business logic never directly performs I/O because it never commits to a model. Commitment happens at the top of `main`.

## 6. The kernel registry as a coproduct

A **coproduct** (or disjoint union) in `𝐒𝐞𝐭` is the set-theoretic union of pairwise-disjoint copies:

```text
all_ptx  ≅  ∐_{(H,M,q) ∈ 𝒯}  Kernels(H, M, q)
```

In code, `all_ptx_sets()` in `atlas-kernels/src/lib.rs` returns this coproduct. Each `(H, M, q)` contributes a summand; the summands share no elements by construction, because different leaf directories produce different PTX blobs with different module names.

The coproduct has a universal property that is worth stating because it matches the design discipline: for any set `S` and family of functions `f_{H,M,q} : Kernels(H, M, q) → S`, there is a unique function `f : all_ptx → S` that restricts to each `f_{H,M,q}`. The registry dispatch at runtime — "given a target, return the right PTX set" — is the inverse construction: a function out of `all_ptx` that *factors through* the target index.

Adding a new target adds a new summand. The universal property says the existing `f_{H,M,q}` for other targets don't need to change. This is the formal meaning of "specialization is a directory, not a template".

## 7. Where general frameworks sit in this picture

A general framework — call one `𝒢` — offers a kernel assignment `Kernels_𝒢 : 𝒯 → 𝐒𝐞𝐭` that factors through a smaller "essence" category `ℰ`:

```text
Kernels_𝒢  :  𝒯  ──F──►  ℰ  ──G──►  𝐒𝐞𝐭
```

`ℰ` has richer morphisms than `𝒯`. Examples of non-identity arrows in `ℰ`:

- Shape polymorphism (a single templated kernel covers `seq_len = 128` and `seq_len = 256` via a compile-time branch).
- Dtype dispatch (a single kernel handles BF16 and FP16 via a runtime tag).
- Just-in-time specialisation (a single source file JITs per shape on first call).

The factoring is attractive because the image of `F` can be small: you write one kernel in `ℰ` and cover many objects of `𝒯`. The cost is paid by `G`: every time `G` realises a morphism from the `ℰ`-image down to a specific `𝒯`-object, real work happens — a branch, a dispatch, a JIT compilation, a dequant-to-BF16 fallback. Those costs are the **genericity tax**.

Atlas refuses the factoring. There is no `ℰ`. `Kernels : 𝒯 → 𝐒𝐞𝐭` is defined directly, object by object, with no intermediate. This is why `atlas-kernels` has no runtime compilation and no dispatch branching: there is nothing to branch over.

The 3.6× gap on Qwen3.5-35B against NVIDIA's vLLM is the cost of NVIDIA's `G` on that particular object. The benchmarks in [Benchmarks](../operations/benchmarks.md) report what the cost is, per kernel and end-to-end, across the whole matrix.

## 8. Reading the book through this lens

The rest of the book, re-read categorically:

- The [Part I philosophy chapter](../getting-started/philosophy.md) argues the refusal of `ℰ` in operator terms: the abstractions that enable scaling to new targets live above the kernel layer, not inside kernels themselves.
- The [Part II philosophy chapter](../architecture/philosophy.md) shows how the refusal forces specific code structure: the kernel tree *is* the coordinate system, the crate split *is* the product decomposition.
- The [workspace chapter](../architecture/workspace.md) is a walk through `ob(𝒯)` and the trait layer above it.
- The [dispatch chapter](../architecture/dispatch.md) traces a single request through the functor composition of Section 4.
- The [SBIO chapter](../architecture/sbio.md) is the operational version of Section 5 — two models of one theory.
- The [crate chapters](../crates/atlas-core.md) describe one vertex of the diagram each.
- The [deep-dive chapters](../deep-dives/kernels.md) describe `Kernels(H, M, q)` at a single object each — the inside of one summand of the coproduct in Section 6.

Nothing in the book changes when you put on the categorical lens. What changes is the vocabulary you have for arguing about proposals — "does this preserve the product structure of `𝒯`?", "does this force a factoring through `ℰ`?", "is this an arrow between models of the algebraic theory, or an operation inside one model?" These are questions a code review benefits from asking aloud.

## 9. What this perspective does not prove

Category theory names structures. It does not measure throughput, does not verify kernel correctness, does not port Atlas to a new hardware vendor, and does not write tool-call parsers. Everything the formalism claims follows from the code already being organised along these lines; the formalism is a mirror, not an engine.

In particular:

- **Performance** is empirical. See [Benchmarking](../operations/benchmarks.md).
- **Correctness** is tested. See [Contributing](../project/contributing.md).
- **Porting a vendor** is design work, not paperwork. The categorical answer ("one new `ComputeTarget` impl, one new `GpuBackend` impl, kernel source") names the files but not the effort.

The formalism earns its keep when it helps catch a design drift early. When a PR proposes a cross-cutting trait that couples two axes of the product — say, a method on `GpuBackend` that only makes sense for one model family — the categorical reading surfaces it immediately: this proposal introduces an arrow between factors of `Hw × Mod × Quant`, breaking the product. That reading has saved review time in the past and will again. It is why the appendix is worth writing down.

---

**Further reading.** For the mathematics used above: Mac Lane, *Categories for the Working Mathematician*, chapters I–III; Riehl, *Category Theory in Context*, chapters 1–4. For the engineering the formalism describes: [Philosophy (Part I)](../getting-started/philosophy.md), [Philosophy (Part II)](../architecture/philosophy.md), [Kernel Dispatch Pipeline](../architecture/dispatch.md), [SBIO](../architecture/sbio.md).
