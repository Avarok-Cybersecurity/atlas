# NVFP4 Support for Qwen3.6-27B Dense (Standard/modelopt format)

## Status

**In progress** — weight loading works for GDN and attention layers; end-to-end inference under test.

## Background

Atlas already supports NVFP4 for the 80B MoE model via the `CompressedTensors` format
(Sehyo/compressed-tensors convention: `weight_packed`, `weight_global_scale`).

The Qwen3.6-27B dense model exists in a second NVFP4 format used by NVIDIA's modelopt toolchain
(Standard/modelopt convention: `weight` as U8, `weight_scale` as FP8E4M3, `weight_scale_2` as F32).
This format was unsupported — Atlas would error immediately on weight load.

Target checkpoint: `sakamakismile/Qwen3.6-27B-Text-NVFP4-MTP` (~19 GB, single safetensors shard).

## NVFP4 Format Comparison

| Field | CompressedTensors (Sehyo) | Standard (modelopt) |
|---|---|---|
| Packed weights | `weight_packed` (U8) | `weight` (U8) |
| Block scales | `weight_scale` (FP8 E4M3) | `weight_scale` (FP8 E4M3) |
| Global scale | `weight_global_scale` (F32, **reciprocal**) | `weight_scale_2` (F32, direct multiplier) |
| Input scale | `input_global_scale` (F32) | `input_scale` (F32) |

The existing `dequant_nvfp4_to_bf16` in `fp8_lut.rs` already auto-detects both formats
(checks for `weight_packed` key; falls back to `weight`). The gap was in the weight loaders
calling it.

## Changes

### `crates/spark-model/src/weight_loader/qwen35_dense.rs`

**1. GDN SSM projections (`load_ssm_proj` closure)**

Added U8 dtype detection so Standard NVFP4 GDN projections dequant correctly:

```rust
let load_ssm_proj =
    |name: &str, rows: usize, cols: usize| -> Result<DenseWeight> {
        if store.contains(&format!("{name}.weight_packed")) {
            // CompressedTensors: weight_packed + weight_global_scale
            dequant_nvfp4_to_bf16(store, name, rows, cols, gpu)
        } else if matches!(
            store.get(&format!("{name}.weight")).map(|w| w.dtype),
            Ok(WeightDtype::UInt8)
        ) {
            // Standard/modelopt: weight (U8) + weight_scale (FP8) + weight_scale_2 (F32)
            dequant_nvfp4_to_bf16(store, name, rows, cols, gpu)
        } else {
            dense_auto(store, &format!("{name}.weight"), gpu)
        }
    };
```

**2. `in_proj_a` / `in_proj_b` routing**

The original code used `dense()` (BF16-only). Changed to route through `load_ssm_proj`.

Important: both `in_proj_a` and `in_proj_b` have `nv` rows (= `linear_num_value_heads` = 48),
not `nk` rows. `nk` is the GDN key-head count used only for interleaving logic, not the
projection output dimension.

```rust
let in_proj_a = load_ssm_proj(&format!("{la}.in_proj_a"), nv, h)?;
let in_proj_b = load_ssm_proj(&format!("{la}.in_proj_b"), nv, h)?;  // nv, not nk
```

**3. Full-attention Q/K/V/O projections (`load_bf16_then_nvfp4` closure)**

The Standard NVFP4 variant path assumed BF16/FP8 weights and called `dense_auto`, which
fails on U8. For pre-quantized Standard NVFP4 attention weights, load directly as
`QuantizedWeight` using `quantized_auto(Standard)`:

```rust
let weight_key = format!("{p}.{name}.weight");
if matches!(
    store.get(&weight_key).map(|w| w.dtype),
    Ok(WeightDtype::UInt8)
) {
    let null_dense = DenseWeight { weight: spark_runtime::gpu::DevicePtr::NULL };
    let qw = quantized_auto(store, &format!("{p}.{name}"), gpu, Nvfp4Variant::Standard)?;
    return Ok((null_dense, qw));
}
let src = dense_auto(store, &weight_key, gpu)?;
// ... existing BF16 → quantize_to_nvfp4 path
```

`gpu.free(NULL)` is safe (returns `Ok(())` in `cuda_backend/gpu_impl.rs`).

This avoids double-quantization (NVFP4 → BF16 → NVFP4) for checkpoints that already
have native NVFP4 attention weights.

## Bugs Found and Fixed

### Bug 1: `dense_auto` rejects U8 dtype
`dense_auto` had no handler for `WeightDtype::UInt8` → `anyhow::bail!`. Fixed by routing
U8 tensors through `dequant_nvfp4_to_bf16` before calling `dense_auto`.

### Bug 2: `in_proj_b` passed wrong row count
Originally `load_ssm_proj("{la}.in_proj_b", nk, h)` — but the tensor has `nv` rows (48),
not `nk` rows (16). This caused `dequant_nvfp4_to_bf16` to allocate a 3× undersized BF16
output buffer, and then `interleave_ba` triggered `cuMemcpyDtoHAsync_v2 status=1`
(CUDA_ERROR_INVALID_VALUE) when trying to D2H copy more bytes than the buffer held.

### Bug 3: Kernel target mismatch
Default build uses `ATLAS_TARGET_MODEL=qwen3-next-80b-a3b`. Must build with
`ATLAS_TARGET_MODEL=qwen3.6-27b ATLAS_TARGET_QUANT=nvfp4 ATLAS_TARGET_HW=gb10`.

## Build Command

```bash
export CUDA_HOME=/usr/local/cuda
export PATH=$PATH:/usr/local/cuda/bin
ATLAS_TARGET_MODEL=qwen3.6-27b \
ATLAS_TARGET_QUANT=nvfp4 \
ATLAS_TARGET_HW=gb10 \
cargo build --release -p spark-server --no-default-features --features cuda
```

Note: requires no NCCL (`--no-default-features --features cuda` drops the `nccl` default).
Multi-GPU EP/TP still requires NCCL; single-GPU inference works without it.

## Test Checkpoint

```
sakamakismile/Qwen3.6-27B-Text-NVFP4-MTP
```

Weight stats: 2354 tensors, dtype breakdown: U8×496, FP8E4M3×496, BF16×370, F32×992.
All 496 linear projection weights (Q/K/V/O for attention + QKV/Z/out/A/B for GDN) are U8.

## Benchmark Results (GB10, sakamakismile/Qwen3.6-27B-Text-NVFP4-MTP)

| Metric | Value |
|---|---|
| Decode speed | ~13.5 tok/s |
| TTFR (prefill) | ~212 ms |
| Model size on disk | 18.3 GB |
| GPU memory after load | ~44.5 GB |
| KV cache budget | 51.1 GB (1,675,824 max tokens) |

**Why not faster than FP8 (~14.7 tok/s)?**

GDN SSM projections (`in_proj_a/b`, `in_proj_qkv`, `in_proj_z`, `out_proj`) are converted
NVFP4→BF16 at load time (CPU dequant). Decode kernels see BF16 weights for GDN layers —
same bandwidth as FP8 path. Attention and FFN layers use native w4a16 NVFP4 kernels.

For a true speed improvement over FP8, native NVFP4 GDN decode kernels are needed (Step 3).

## What's Next

- [ ] Test with `unsloth/Qwen3.6-27B-NVFP4` (partial quant: attention=Standard NVFP4, GDN A/B=BF16)
- [ ] Test with `AEON-7/Qwen3.6-27B-AEON-Ultimate-Uncensored-Multimodal-NVFP4-MTP-XS` (CompressedTensors)
- [ ] Native NVFP4 GDN decode kernels — keep GDN projections as QuantizedWeight and dispatch w4a16_gemv instead of BF16 GEMV. Expected: ~2× bandwidth reduction for GDN layers → meaningful tok/s gain.
- [ ] TP sharding for pre-quantized Standard NVFP4 attention weights (currently tp_size=1 only)
- [ ] Open PR to Atlas upstream

## Hardware

NVIDIA GB10 (Grace Blackwell), sm_121a, 121 GB unified RAM, CUDA 13.0.
