# atlas-core

**Role:** the type + trait vocabulary every other Atlas crate builds on.
**Key traits:** `ComputeTarget` (build-time compiler abstraction), `Vendor`, `DType`, `KernelTarget`, `TensorRef`, `ModelConfig`.
**Dependencies:** none from the workspace — this is the bottom of the stack.

## Module map

```text
crates/atlas-core/src/
├── lib.rs            re-exports every public module
├── compute.rs        ComputeTarget trait + Vendor enum (Nvidia/Amd/Apple/Intel)
├── target.rs         KernelTarget — the (arch, model, quant) dispatch key
├── dtype.rs          DType enum: E2M1, FP8E4M3, FP8E5M2, BF16, FP16, FP32
├── tensor.rs         TensorRef — zero-copy handle (ptr, shape, strides, dtype)
├── config.rs         ModelConfig — deserialized HF config.json (SSOT for model dims)
├── device.rs         Device {ordinal, total/free memory}
├── stream.rs         Stream abstraction (u64 wrapping CUstream or Metal queue)
├── kernel.rs         Kernel launch descriptor types
├── registry.rs       Kernel module registry shared by atlas-kernels codegen
├── capabilities.rs   Hardware capability flags (tensor cores, async copy, graphs)
└── error.rs          anyhow re-export + Result alias
```

## `ComputeTarget`: the hardware-vendor abstraction

```rust
pub trait ComputeTarget {
    fn source_extension(&self) -> &str;       // "cu", "metal", "hip", "cl"
    fn output_extension(&self) -> &str;       // "ptx", "metallib", "hsaco"
    fn output_is_text(&self) -> bool;         // PTX is text; metallib is binary
    fn find_compiler(&self) -> Option<PathBuf>; // nvcc, xcrun, hipcc, icpx
    fn compile(&self, src: &Path, out: &Path, arch: &str, flags: &[String]) -> Result<(), String>;
    fn vendor(&self) -> Vendor;
}
```

The trait is consumed at **build time** by `atlas-kernels/build.rs`, which reads `kernels/<hw>/HARDWARE.toml`, instantiates the matching `Box<dyn ComputeTarget>`, and calls `compile()` on every source file in every matching `(model, quant)` leaf. Today only `NvidiaTarget { nvcc }` is implemented; `AppleTarget`, `AmdTarget`, `IntelTarget` are the planned impls.

## `Vendor` and the matrix

```rust
pub enum Vendor { Nvidia, Amd, Apple, Intel }
```

Parsed from `HARDWARE.toml`'s `vendor = "nvidia"` / `"amd"` / `"apple"` / `"intel"` field. `Vendor::from_str` lives here, used by both `build.rs` (to pick `ComputeTarget`) and `spark-server::main` (to pick the runtime `GpuBackend`).

## `KernelTarget`: the runtime dispatch key

```rust
pub struct KernelTarget {
    pub arch: &'static str,   // "sm_121", "sm_100a", ...
    pub model: &'static str,  // "qwen3-next-80b-a3b"
    pub quant: &'static str,  // "nvfp4", "fp8", "bf16"
}
```

One `const` per supported target (`KernelTarget::GB10_QWEN3_NVFP4`, `GB10_QWEN35_NVFP4`, `GB10_QWEN35_122B_NVFP4`, …). Equality + hashing work from the three string components; the struct is `Copy`.

The `atlas-kernels` crate's auto-generated `target_ptx.rs` emits one `pub static PTX_<TARGET>: &[PtxModule]` per known `KernelTarget`. Startup dispatch in `spark-server::main` calls `atlas_kernels::select_target(&ptx_sets, &target)` to pick the right module set.

## `DType`: the supported numeric formats

```rust
pub enum DType {
    E2M1,        // 4-bit float: {0, 0.5, 1, 1.5, 2, 3, 4, 6} — NVFP4 weights
    FP8E4M3,     // 8-bit float, 4e/3m — block scales, FP8 weights
    FP8E5M2,     // 8-bit float, 5e/2m — alternative FP8
    BF16,        // brain float — activations, residual stream
    FP16,        // IEEE 754 half — rarely used
    FP32,        // full precision — accumulation
}
```

`DType::element_size_bits()` returns the per-element bit count (not bytes — E2M1 is sub-byte). Buffer sizing in `spark-runtime::buffers` uses this.

## `TensorRef`: the kernel-argument view

```rust
pub struct TensorRef {
    pub ptr: u64,              // CUdeviceptr / MTLBuffer / ...
    pub shape: Vec<usize>,
    pub strides: Vec<usize>,   // in elements, not bytes
    pub dtype: DType,
}
```

Every kernel argument that points at GPU memory is a `TensorRef`. The layer code in `spark-model` builds `TensorRef`s from `DevicePtr`s owned by `BufferArena` (in `spark-runtime`), passes them to primitive op traits (`Normalize::rms_norm`, `Activation::silu_mul`, `Reduce::topk`), and those impls extract the raw `ptr: u64` to pass to `GpuBackend::launch()`.

`TensorRef` itself does not own memory and is `Clone` — it's always a view.

## `ModelConfig`: the single source of truth for model shape

`ModelConfig::from_hf(&path)` reads `config.json` (plus nested text/vision configs for VLMs) and returns a struct that every downstream buffer/KV-cache size derives from. Fields include `hidden_size`, `num_hidden_layers`, `num_attention_heads`, `num_key_value_heads`, `intermediate_size`, `num_experts`, `num_experts_per_token`, `vocab_size`, MoE-specific `moe_intermediate_size`, SSM-specific `ssm_state_size`, vision tower sub-config, and layer-type arrays for hybrid models.

Per the PCND principle, there are no implicit defaults for fields that materially affect correctness — missing critical fields produce a `bail!` at load time.

## `Capabilities`: hardware feature flags

`HardwareCapabilities` exposes booleans like `tensor_cores`, `async_copy`, `cuda_graphs`, `native_fp4_mma`. `HARDWARE.toml` populates them; kernel-selection code in some layers reads them to pick between code paths (e.g., GB10 has no native FP4 MMA so the NVFP4 kernels fall back to software E2M1 conversion — see the [NVFP4 deep dive](../deep-dives/nvfp4.md)).

## What's explicitly not here

- **No actual compilation.** `ComputeTarget::compile()` is a trait method; the impls live in the callers (or in `atlas-kernels/build.rs` for the bootstrap).
- **No actual GPU allocation.** That's `spark-runtime`.
- **No weight loader types.** Those are `spark-model`.
- **No HTTP types.** Those are `spark-server`.

`atlas-core` is small on purpose. It contains only the vocabulary that *every* crate downstream needs.

## Adding a vendor

Implementing a new hardware vendor starts here:

1. Extend `Vendor` if needed (all four major ones are already enumerated).
2. Write your `struct XxxTarget` in your own crate or inline, implement `ComputeTarget`.
3. Register it in `atlas-kernels/build.rs::resolve_compute_target()`.
4. Continue to the [spark-runtime chapter](./spark-runtime.md) for the runtime trait (`GpuBackend`).
