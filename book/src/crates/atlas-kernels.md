# atlas-kernels

**Role:** the bridge between the CUDA source tree and the Rust workspace. Every PTX module every other crate launches is defined here.
**Key file:** `src/lib.rs` (hand-written glue, ~50 lines) + `build.rs` (auto-generates the heavy file).

## The trick: auto-generated PTX embedding

`atlas-kernels/src/lib.rs` ends with a single line:

```rust
include!(concat!(env!("OUT_DIR"), "/target_ptx.rs"));
```

Everything inside `target_ptx.rs` — per-target PTX byte constants, the `ptx_modules()` lookup function, the `all_ptx_sets()` multi-target registry — is produced by `build.rs` at every Cargo build. You will not find `target_ptx.rs` in the repository; it is generated fresh into `OUT_DIR` each time.

The generated file looks roughly like:

```rust
pub static PTX_GB10_QWEN3_NVFP4: &[PtxModule] = &[
    PtxModule { name: "prefill_attn_v47", ptx: include_bytes!("...sm121/prefill_v47.ptx") },
    PtxModule { name: "decode_attn",       ptx: include_bytes!("...sm121/decode_attn.ptx") },
    PtxModule { name: "moe_w4a16",         ptx: include_bytes!("...sm121/moe_w4a16.ptx") },
    // ~35 modules per target
];
pub static PTX_GB10_QWEN35_NVFP4: &[PtxModule] = &[ /* ... */ ];

pub fn ptx_modules(target: &KernelTarget) -> Option<&'static [PtxModule]> {
    match (target.arch, target.model, target.quant) {
        ("sm_121", "qwen3-next-80b-a3b", "nvfp4") => Some(PTX_GB10_QWEN3_NVFP4),
        ("sm_121", "qwen3.5-35b-a3b",   "nvfp4") => Some(PTX_GB10_QWEN35_NVFP4),
        // ...
        _ => None,
    }
}
```

At runtime, `spark-server::main` resolves the current model's `KernelTarget`, calls `ptx_modules(&target)`, and passes the resulting slice to `AtlasCudaBackend::new` which uploads each PTX module to the GPU via `cuModuleLoadData`.

## What `build.rs` actually does

1. **Read `ATLAS_TARGET_*`** — three env vars (`HW`, `MODEL`, `QUANT`). Wildcards (`*`) expand to "every matching directory".
2. **Walk `kernels/<hw>/<model>/<quant>/`** — for each leaf that matches the wildcards, read `HARDWARE.toml`, `MODEL.toml`, `KERNEL.toml`.
3. **Resolve the compiler** — `resolve_compute_target(vendor)` returns a `Box<dyn ComputeTarget>`. Today always `NvidiaTarget { nvcc }`.
4. **Compile every source file** — for each `*.cu` in the leaf, call `compute_target.compile(src, out, arch, flags)`. Flags come from `KERNEL.toml`'s `extra_nvcc_flags = [...]` plus the arch-specific ones from `HARDWARE.toml`.
5. **Apply module-name overrides** — `KERNEL.toml`'s `[modules]` section lets kernels with different file stems (`e2m1_branchless.cu`) expose themselves under shorter module names (`e2m1`). This is cosmetic but keeps `GpuBackend::kernel("e2m1", "convert_f32_to_e2m1")` readable at the call site.
6. **Parse `MODEL.toml` → `SamplingPresets` + `ModelBehavior`** — non-kernel metadata that the server consumes directly (default `temperature`, `thinking_budget`, etc.). Emitted as Rust constants alongside the PTX.
7. **Write `target_ptx.rs`** — one `PtxModule` array per target, the dispatch match, and a `const ALL_TARGETS: &[KernelTarget]` listing everything that got compiled.

The whole phase is idempotent — `rerun-if-changed` directives on the kernel tree mean `cargo` only recompiles what changed.

## `ATLAS_SKIP_BUILD=1` — the escape hatch

On a Linux laptop with no `nvcc`, the crate would fail to build without this. The escape hatch is a single env var. When set, `build.rs`:

- Does not invoke any compiler.
- Emits a stub `target_ptx.rs` with an empty `ALL_TARGETS` and `ptx_modules` returning `None` for everything.
- The crate compiles cleanly.

`ci.yml` uses this. So does local `cargo clippy`. The [Kernel Dispatch](../architecture/dispatch.md) chapter covers the broader flow.

## Per-target PTX bytes — sizes and counts

Rough numbers for the default multi-model build at `avarok/atlas-gb10:latest`:

| Target | # kernels | PTX bytes (approx) |
|---|---:|---:|
| GB10 / Qwen3.5-35B-A3B / NVFP4 | 35 | ~5.4 MB |
| GB10 / Qwen3-Next-80B-A3B / NVFP4 | 35 | ~5.5 MB |
| GB10 / Qwen3.5-122B-A10B / NVFP4 | 35 | ~5.5 MB |
| GB10 / Nemotron-3-Nano / NVFP4 | 33 | ~4.8 MB |
| GB10 / Nemotron-3-Super / NVFP4 | 33 | ~4.8 MB |
| GB10 / Mistral-Small-4 / NVFP4 | 31 | ~4.2 MB |
| GB10 / MiniMax-M2.7 / NVFP4 | 38 | ~6.1 MB |
| GB10 / Qwen3.6 / FP8 | 34 | ~5.2 MB |
| GB10 / Gemma-4 / NVFP4 (×2 flavors) | 29 | ~4.0 MB ea |
| GB10 / Qwen3-VL / NVFP4 | 40 (incl. ViT) | ~6.6 MB |

Total embedded PTX in the default multi-model binary: ~65 MB. The binary itself lands at ~200 MB in release builds. This is why the Docker image is ~8 GB once you add the CUDA userspace (`libcudart`, `libnvrtc`), tokenizer deps, and Ubuntu base — the actual Atlas footprint is small.

## What gets added to this crate when you…

- **…add a new `(hw, model, quant)` leaf?** Nothing in the `atlas-kernels/src/` directory. `build.rs` picks it up automatically on the next `cargo build`. You *do* need to add a matching `KernelTarget` const in `atlas-core::target::KernelTarget` so downstream code can refer to it by name.
- **…add a new kernel to an existing leaf?** Drop the `.cu` in the leaf directory; `build.rs` picks it up. If you want a non-stem module name, add an entry to the leaf's `KERNEL.toml`.
- **…add a new hardware vendor?** Extend `resolve_compute_target(vendor)` in `build.rs` to return your new `ComputeTarget` impl. Everything else flows from there.

The rest of the kernel-engineering story is in the [CUDA Kernel Engineering](../deep-dives/kernels.md) deep dive.
