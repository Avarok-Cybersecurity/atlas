# Workspace Layout

Atlas is a twelve-member Cargo workspace plus a build-time kernel tree. This chapter maps every top-level directory to its role, and the twelve crates to the axes of variation they each insulate.

## Repository tree (top level)

```
atlas/
├── README.md                     headline, benchmarks, porting guides
├── QUICKSTART.md                 per-model Docker recipes
├── CONTRIBUTING.md, AGENTS.md    contributor workflow
├── SECURITY.md                   disclosure
├── CLA.md                        contributor license agreement
├── LICENSE                       AGPL-3.0-only
├── Cargo.toml                    workspace root (12 members)
├── Cargo.lock
├── rust-toolchain.toml           pins stable
├── deny.toml                     cargo-deny allow/deny lists
├── crates/                       Rust source for every crate
├── kernels/                      CUDA source, organized as (hw, model, quant)
├── docker/                       per-hardware Dockerfiles
├── scripts/                      bench, model-sweep, release helpers
├── tests/                        cross-crate integration tests (run_all_models.py lives here)
├── docs/                         design notes, history, release notes
├── paper/                        LaTeX paper (ArXiv)
├── jinja-templates/              chat templates for models that need custom ones
├── bench/                        stable benchmark harness outputs (tracked)
├── book/                         this book (mdBook source)
└── vendor/                       vendored deps (e.g. xgrammar-rs)
```

## The twelve workspace members

`Cargo.toml` lists:

```toml
members = [
    "crates/atlas-core",
    "crates/atlas-quant",
    "crates/atlas-norm",
    "crates/atlas-activation",
    "crates/atlas-embed",
    "crates/atlas-reduce",
    "crates/atlas-kernels",
    "crates/spark-runtime",
    "crates/spark-comm",
    "crates/spark-model",
    "crates/spark-server",
    "crates/atlas-spark-bench",
]
```

Each is its own crate with its own `Cargo.toml`, its own unit tests, and its own responsibility:

| Crate | Role | Consumed by |
|---|---|---|
| `atlas-core` | Traits & types used by every crate below: `ComputeTarget` (build-time compiler abstraction), `KernelTarget` (runtime dispatch key), `Vendor`, `Dtype`, `Tensor`, `ModelConfig` parsing | everyone |
| `atlas-quant` | Quantization traits + kernels: NVFP4 (4-bit E2M1 + FP8 scales), FP8 native | `spark-model`, `atlas-kernels` |
| `atlas-norm`, `atlas-activation`, `atlas-embed`, `atlas-reduce` | Small primitive-op trait crates (RMSNorm, SiLU, RoPE, argmax). Keeps the trait-only surface area clean | `spark-model`, `spark-runtime` |
| `atlas-kernels` | Auto-generated Rust glue over compiled PTX. `build.rs` enumerates `kernels/<hw>/<model>/<quant>/*.cu`, compiles each through the matching `ComputeTarget`, emits one `target_ptx.rs` that `include!()`s back into this crate | `spark-runtime` |
| `spark-runtime` | `GpuBackend` trait (27 methods) + CUDA impl (`cuda_backend.rs`). KV cache, prefix cache (radix tree), paged FP8 cache, buffer arena, sampler, `WeightStore` (`O_DIRECT` + pipelined safetensors loader). Everything that touches the GPU goes through here. | `spark-model`, `spark-server` |
| `spark-comm` | `CommBackend` trait (collective ops) + NCCL impl. `SingleGpuBackend` is the no-op impl for single-GPU runs. | `spark-model`, `spark-server` |
| `spark-model` | Model assembly: layers (`Qwen3Attention`, `Qwen3Ssm`, `NemotronMamba2`, `MoeLayer`, `VisionEncoder`), per-family weight loaders, `TransformerLayer` trait, the inference engine (`engine.rs`), speculative decoding, vision preprocessing | `spark-server` |
| `spark-server` | Binary. HTTP server (axum), OpenAI + Anthropic compatible endpoints, tool-call parsing (Hermes / Qwen3-coder / Mistral / XGrammar), tokenizer wrapper, rate limiter, CLI | n/a — the deliverable |
| `atlas-spark-bench` | Criterion benchmark client. Targets a live server, records per-endpoint throughput + TTFT. The numbers in `bench/` come from here. | bench runs only |

The dependency graph runs strictly downward in the table above — `atlas-core` has no internal deps, every crate above it builds on crates below. There are no cycles.

## The kernel tree

```
kernels/
└── gb10/                                        # One directory per hardware target
    ├── HARDWARE.toml                            # vendor, arch, memory specs
    ├── qwen3-next-80b-a3b/                      # One directory per model target
    │   ├── MODEL.toml                           # layer counts, sampling presets, behavior
    │   └── nvfp4/                               # One directory per quantization target
    │       ├── KERNEL.toml                      # compile flags, module name overrides
    │       └── *.cu                             # ~35 hand-written CUDA kernels
    ├── qwen3.5-35b-a3b/
    │   └── nvfp4/
    │       └── *.cu
    ├── qwen3.6-35b-a3b/
    │   └── fp8/
    ├── nemotron-3-nano-30b-a3b/
    │   └── nvfp4/
    ├── mistral-small-4-119b/
    │   └── nvfp4/
    ├── minimax-m2-229b/
    │   └── nvfp4/
    └── ... (one leaf per supported model, twelve leaves today)
```

Every leaf directory is a fully self-contained `(gb10, model, quant)` target. The kernels inside a leaf can use any tile shape, any register budget, any shared-memory layout — they are physically incapable of regressing a different target.

This is the mechanism that makes `kernels/` a scalable structure. Adding a new GPU is `kernels/<new-hw>/`. Adding a new model is `kernels/<hw>/<new-model>/`. Adding a new quantization is `kernels/<hw>/<model>/<new-quant>/`. Nothing else moves.

## Docker layout

```
docker/
├── gb10/
│   ├── Dockerfile                         multi-model image — compiles every target
│   ├── qwen3-next-80b-a3b/nvfp4/          per-model slim image
│   ├── qwen3.5-35b-a3b/nvfp4/
│   └── ... (one slim Dockerfile per supported model)
└── docker-guide.md                        build + run instructions
```

The multi-model `Dockerfile` at `docker/gb10/Dockerfile` is what ships as `avarok/atlas-gb10:latest`. Per-model Dockerfiles exist for operators who want a smaller image containing only one target — the kernel registry still uses `KernelTarget` at runtime, but only one target set is baked in.

## Docs, design, history, releases

Inside `docs/`:

- `design/` — authoritative design docs for in-flight features (NVFP4 coherence, FP8 native serving, xgrammar integration, MixKVQ, tool-calling gap analysis, chunked prefill, etc.). Treat these as the long-form rationale behind code changes; commit messages are deliberately terse and point here.
- `history/` — benchmark journeys and retrospectives (alpha-2 release, SM100 notes, Qwen3.5-35B journey, SSM catastrophic forgetting TODO, multi-GPU optimizations). Useful context, but not a contract.
- `releases/` — human-readable release notes per alpha version (`alpha-2.43.md`, `alpha-2.44.md`).

The book you're reading in `book/` synthesises all of this into a single narrative — it is *not* a canonical rewrite of those documents. The design docs in `docs/design/` remain the authoritative reference and the book links to them directly from the deep-dive chapters.

## What changes when you add a…

| You added | You touched |
|---|---|
| A new quantization (e.g. MXFP4) | `atlas-quant/src/<scheme>.rs`, `kernels/<hw>/<model>/<scheme>/*.cu`, runtime dispatch in `spark-model/src/quant_format.rs` |
| A new model family (e.g. Phi-4) | `spark-model/src/weight_loader/<family>.rs`, one arm in `spark-model/src/factory.rs`, `kernels/<hw>/<family>/<quant>/MODEL.toml`, optional `jinja-templates/<family>.j2` |
| A new hardware vendor (e.g. MI300X) | `atlas-core/src/compute.rs` (new `ComputeTarget` impl), `atlas-kernels/build.rs::resolve_compute_target()` arm, `spark-runtime/src/<vendor>_backend.rs` (new `GpuBackend` impl), `spark-comm/src/<vendor>_backend.rs` if the vendor needs its own collective impl, `kernels/<hw>/HARDWARE.toml`, kernel source under `kernels/<hw>/<model>/<quant>/` |
| A new CLI flag | `spark-server/src/cli.rs`, plumbing wherever it lands |
| A new tool-call format | `spark-server/src/tool_parser.rs` |

Each row touches a small, bounded set of files. That bounded-ness is the architectural payoff of the workspace being split along axes of variation. Read [Kernel Dispatch Pipeline](./dispatch.md) next to see the runtime side, or [SBIO](./sbio.md) to see how the trait layering makes the whole thing testable without a GPU.
