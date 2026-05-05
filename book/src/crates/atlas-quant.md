# atlas-quant

**Role:** quantization traits + dispatch glue for NVFP4 and FP8 weight/KV formats.
**Key traits:** `Quantize`, `Dequantize`.
**Modules:** `nvfp4`, `fp8`, `traits`.

## What `atlas-quant` owns

Quantization lives at three levels in Atlas, and this crate owns the middle level:

1. **Format definitions + runtime dispatch** — this crate. `NvFp4Quantizer`, `Fp8Format`, scale-tensor layout, block sizes, per-tensor vs per-block conventions.
2. **The actual quantization kernels** — live in `kernels/<hw>/<model>/<quant>/*.cu`. `atlas-quant` does not compile CUDA; `atlas-kernels` does.
3. **Per-request dispatch** (which format to use given a loaded checkpoint) — lives in `spark-model/src/quant_format.rs`, a thin wrapper that picks the right `atlas-quant` impl from the model's config.

This separation is deliberate. Adding a new quantization scheme (e.g. MXFP4, int4+zp) is a new module here plus new `.cu` files under the kernel tree — it does not touch the scheduler, the engine, or the HTTP layer.

## The `Quantize` / `Dequantize` traits

```rust
pub trait Quantize {
    fn quantize(
        &self,
        input: &TensorRef,    // FP32 / BF16 activations
        output: &TensorRef,   // target dtype (E2M1 / FP8)
        scale: &TensorRef,    // per-block scale factors
        stream_ptr: u64,
    ) -> Result<()>;
}

pub trait Dequantize {
    fn dequantize(
        &self,
        input: &TensorRef,
        output: &TensorRef,
        scale: &TensorRef,
        stream_ptr: u64,
    ) -> Result<()>;
}
```

Both are GPU-side operations — the trait methods take a `stream_ptr` so the caller can pipeline quantization against MoE dispatch or attention. The impls call into `atlas-kernels`'s loaded PTX via `GpuBackend::launch`.

## NVFP4 — the Atlas headline format

`NVFP4` = **4-bit E2M1 weights** + **FP8 E4M3 per-block scales** (block size = 16 elements). For a `[N, K]` weight matrix, the storage is:

- `weight` — `[N, K/2]` `u8` (two E2M1 nibbles per byte)
- `weight_scale` — `[N, K/16]` `f8e4m3` (one scale per 16-element block along K)

`NvFp4Quantizer` (in `nvfp4.rs`) is the runtime binding for the `e2m1_branchless.cu` kernel. "Branchless" refers to the conversion from FP32 to E2M1 — the kernel uses 7 unsigned integer comparisons on the IEEE-754 bit pattern instead of branching on the float's magnitude. This matters on SM121 because the hardware lacks the `cvt.rn.satfinite.e2m1x2.f32` PTX instruction that later Blackwell variants expose; the software path costs ~3 extra ALU ops per conversion but avoids a silicon limitation.

The [NVFP4 deep dive](../deep-dives/nvfp4.md) walks the actual numeric conversion, the NVFP4 value set (`{0, ±0.5, ±1, ±1.5, ±2, ±3, ±4, ±6}` after sign), and the SM121-specific conversion kernel in detail.

## FP8 — two checkpoint shapes

`Fp8Format` in `fp8.rs` describes two different in-checkpoint layouts Atlas has to handle:

```rust
pub struct Fp8Format {
    pub block_size: usize,         // e.g. 128
    pub scale_dtype: ScaleDtype,   // Fp32 or Bf16
}
```

- **Per-tensor scaled** — `weight` (FP8) + `weight_scale` (a single `f32`). Used by many vLLM-exported FP8 checkpoints.
- **Block-scaled** — `weight` (FP8) + `weight_scale_inv` (BF16, one scale per `block_size × block_size` tile). Used by `compressed-tensors` FP8 checkpoints from Qwen and Nemotron.

Both layouts are valid on-disk formats. `Fp8Format` unifies them at load time so that the layer code sees a single dequant trait and does not care about the disk layout.

The crate also ships a 256-entry **FP8 E4M3 → f32 lookup table** for CPU-side sanity checks and weight-loader arithmetic (e.g. shape validation, scale inversion). The fast GPU path does not use the LUT — it uses hardware FP8 conversions or the `cvt.rn.bf16.e4m3` PTX instruction.

See the [FP8 deep dive](../deep-dives/fp8.md) for the full story, including why FP8 native serving (FP8 all the way to the tensor cores, no BF16 upcast) is the primary performance path for Qwen3.6 and Nemotron checkpoints.

## Why it's tiny

The crate is tiny — `traits.rs` is ~30 lines, `nvfp4.rs` is a handful, `fp8.rs` is larger because of the checkpoint-format enum and the LUT. That's on purpose. The complexity lives in the CUDA source under `kernels/<hw>/<model>/<quant>/`. This crate exists to give the layer code a typed, vendor-agnostic, testable handle on quantization — not to do quantization.

## What's explicitly not here

- **The actual CUDA kernels.** They live in the kernel tree.
- **The per-layer quantization decision** (bf16 vs fp8 KV, which layers stay at BF16). That is `spark-runtime::kv_cache::KvCacheDtype` + `spark-server::cli` flags.
- **Weight-loader plumbing** (reading an FP8 checkpoint off disk). That is `spark-model::weight_map::dequant_fp8_to_bf16` etc.

## Adding a new quantization scheme

Checklist (also in the repo's [Adding a new hardware target](https://github.com/Avarok-Cybersecurity/atlas/blob/main/README.md#adding-a-new-hardware-target) guide, though the scheme axis is model-orthogonal):

- [ ] `atlas-quant/src/<scheme>.rs` — format struct + `impl Quantize` / `impl Dequantize`.
- [ ] `kernels/<hw>/<model>/<scheme>/*.cu` — the actual kernels.
- [ ] `kernels/<hw>/<model>/<scheme>/KERNEL.toml` — compile flags.
- [ ] `spark-model/src/quant_format.rs` — runtime detection (sniff the checkpoint shape, return the right `Box<dyn Quantize>` / `Box<dyn Dequantize>`).
- [ ] A match arm in `spark-model::factory::loader_for_config` *if* the scheme has loader-visible effects.
- [ ] `spark-server::cli::ServeArgs::kv_cache_dtype` enum extension if the scheme is KV-cache-eligible.
