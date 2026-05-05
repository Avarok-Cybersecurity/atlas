# ADR-0009: `build.rs` PTX compilation per (hardware, model, quant) tuple

**Status:** Accepted
**Date:** 2026-04-17

## Context

Atlas's CUDA kernels are written per **(hardware, model, quant)** tuple.
A NVFP4 grouped-GEMM tuned for Qwen3.6-35B on GB10 is not the same code
as the FP8 dense GEMM for Mistral-Small-4 on the same hardware, even
though they share device code style.

The candidate distribution shapes:

1. **Pre-compiled `.fatbin` blobs**, checked into the repo or downloaded
   on first run. Fastest startup; binary-size bloat; weak provenance
   (which exact `.cu` was this fatbin compiled from?).
2. **Just-in-Time at startup** via `nvrtc`. Slow first run; great for
   experiments; runtime CUDA toolchain dependency.
3. **Compile to PTX at build time, embed PTX bytes in the binary,
   load via cudarc at startup.** PTX is small (vs SASS), human-
   inspectable, lets the driver re-JIT to the running GPU's exact SM.

Within (3), the question becomes "how do we organize the kernel tree
and let users build subset binaries (one model + one quant) versus
sweep binaries (everything)?".

## Decision

Atlas uses approach (3) with a directory tree:

```
kernels/
└── <hardware>/                    e.g. gb10
    ├── HARDWARE.toml              arch, sm, fp32-residual flag
    ├── <quant>/                   shared kernels for hw + quant
    │   └── *.cu                   e.g. nvfp4/dense_gemm.cu
    └── <model>/                   per-model overrides
        ├── MODEL.toml             model_type list, sampling, behavior
        └── <quant>/               per-(model, quant) overrides
            └── *.cu               e.g. qwen3.6-35b-a3b/nvfp4/inferspark_prefill_h128.cu
```

A single `build.rs` (`crates/atlas-kernels/build.rs`) walks the tree,
reads `HARDWARE.toml` and `MODEL.toml`, and compiles every `.cu` for
the **selected tuple** to PTX via `nvcc`. The build script:

- Picks the tuple via env vars: `ATLAS_TARGET_HW`, `ATLAS_TARGET_MODEL`,
  `ATLAS_TARGET_QUANT`. Each accepts `*` to wildcard ("all known
  models", "all known quants").
- Resolves model-specific overrides over shared kernels at the file-name
  level (per-model `dense_gemm.cu` wins over shared `dense_gemm.cu`).
- Emits a generated `target_ptx.rs` containing
  `static PTX_BYTES: &[(&str, &[u8])]` arrays of (kernel-module-name,
  PTX-string) pairs.
- Marks `cargo:rerun-if-changed=` on every consumed file so incremental
  builds work.
- Provides an `ATLAS_SKIP_BUILD=1` short-circuit that emits a stub
  registry — used by CI for GPU-free `cargo check` / `clippy` runs.

At runtime, `spark-runtime::gpu` loads the embedded PTX into the CUDA
driver and exposes kernel handles to the rest of the stack.

## Consequences

**Better:**
- One static binary per deployment shape. No first-run JIT, no fatbin
  blobs to ship separately.
- The build script is the single source of truth for "which kernels
  exist." Adding a new (hw, model, quant) tuple is a directory
  addition, not a code change.
- PTX is small enough to embed without bloating the binary
  appreciably (~tens of MB for a sweep build).
- The driver re-JITs PTX → SASS for the actual running SM, so a binary
  built for "blackwell, sm_120-or-up" runs on sm_121 without a rebuild.

**Worse:**
- Every (model, quant) added is more `nvcc` invocations at build
  time. A full sweep build is several minutes. CI uses
  `ATLAS_SKIP_BUILD=1` to dodge this for non-build steps.
- The override-by-name resolution is implicit. A misnamed per-model
  override silently doesn't override anything; the build script logs
  shadows but the warning is easy to miss.
- `MODEL.toml` is now a load-bearing configuration surface that lives
  outside Rust's type system. Schema drift is a real risk; we mitigate
  with serde-derived parsers in `atlas-core`.

**New problems we created:**
- Cross-tuple kernel sharing is awkward. If two models legitimately
  want the same kernel with different `#define` constants, the choice
  is "duplicate the file" or "introduce a templating layer." We've
  taken duplication (cheap to read) so far; large families may force
  a templating step.
- Adding a new hardware target requires a new `HARDWARE.toml` + a
  per-quant kernel directory; all kernels must recompile against the
  new SM target with potentially different tile shapes. See
  `docs/HARDWARE.md` for the recipe.
