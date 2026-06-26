// SPDX-License-Identifier: AGPL-3.0-only

//! Auto-extracted from `weight_map.rs` during refactor wave 4a.

#![allow(unused_imports)]

use anyhow::{Context, Result, bail, ensure};
use spark_runtime::gpu::{DevicePtr, GpuBackend};
use spark_runtime::weights::{WeightDtype, WeightStore};

use super::*;

/// Shared CPU-side FP8 E4M3 → BF16 conversion.
pub(super) fn dequant_fp8_bytes_to_bf16(fp8_buf: &[u8], scale: f32) -> Vec<u8> {
    fp8_buf
        .iter()
        .flat_map(|&byte| {
            let val = fp8_e4m3_to_f32(byte) * scale;
            f32_to_bf16(val).to_le_bytes()
        })
        .collect()
}

/// Dequantize FP8 E4M3 block-scaled weight → BF16, entirely on the GPU.
///
/// Block-scaled FP8 (e.g. `quant_method: "fp8"` with `weight_block_size: [128, 128]`):
///   - `{prefix}.weight`: FP8E4M3 tensor of shape `[N, K]`
///   - `{prefix}.weight_scale_inv`: BF16 (Qwen/DeepSeek) or FP32 (MiniMax) of shape `[N/block, K/block]`
///   - Dequant: `bf16[i,j] = E4M3_LUT[fp8[i,j]] * scale_inv[i/block, j/block]`
///
/// The FP8 weight and scale tensors already live on the GPU (loaded by the
/// fast weight loader). This launches `dequant_fp8_blockscaled_bf16` to do
/// the conversion in-place on device — no D2H download, no host CPU loop,
/// no H2D upload. Replaces the old per-element CPU loop that dominated load
/// time for FP8-MoE models under ATLAS_FP8_DEQUANT_MOE_TO_BF16=1 (~30k calls,
/// ~22 min total → ~seconds).
///
/// Returns a BF16 DenseWeight on GPU.
pub(crate) fn dequant_fp8_blockscaled_to_bf16(
    store: &WeightStore,
    prefix: &str,
    gpu: &dyn GpuBackend,
) -> Result<DenseWeight> {
    use spark_runtime::kernel_args::{KernelLaunch, div_ceil};

    let w = store.get(&format!("{prefix}.weight"))?;
    ensure!(
        w.dtype == WeightDtype::FP8E4M3,
        "Expected FP8E4M3 for {prefix}.weight, got {:?}",
        w.dtype,
    );
    ensure!(
        w.shape.len() == 2,
        "Expected 2D weight for {prefix}, got {:?}",
        w.shape
    );
    let n = w.shape[0];
    let k = w.shape[1];
    let total = n * k;
    let byte_size = w.byte_size();
    ensure!(
        total == byte_size,
        "FP8 size mismatch: total={total} byte_size={byte_size}"
    );

    let s = store.get(&format!("{prefix}.weight_scale_inv"))?;
    ensure!(
        s.dtype == WeightDtype::BF16 || s.dtype == WeightDtype::FP32,
        "Expected BF16 or FP32 for {prefix}.weight_scale_inv, got {:?}",
        s.dtype,
    );
    let sn = s.shape[0];
    let sk = s.shape[1];
    let block_n = (n / sn) as u32;
    let block_k = (k / sk) as u32;
    let scale_is_f32 = s.dtype == WeightDtype::FP32;

    // Allocate BF16 output on device (2 bytes/element).
    let out = gpu.alloc(total * 2)?;

    // GPU dequant: bf16_out[n,k] = E4M3_LUT[fp8[n,k]] * scale_inv[n/block_n, k/block_k].
    // Block (64, 4, 1) → each thread does one element; grid covers [K, N].
    let stream = gpu.default_stream();
    let kernel = gpu.kernel(
        "dequant_fp8_blockscaled_bf16",
        "dequant_fp8_blockscaled_bf16",
    )?;
    KernelLaunch::new(gpu, kernel)
        .grid([div_ceil(k as u32, 64), div_ceil(n as u32, 4), 1])
        .block([64, 4, 1])
        .arg_ptr(w.ptr)
        .arg_ptr(s.ptr)
        .arg_ptr(out)
        .arg_u32(n as u32)
        .arg_u32(k as u32)
        .arg_u32(block_n)
        .arg_u32(block_k)
        .arg_u32(sk as u32)
        .arg_u32(scale_is_f32 as u32)
        .launch(stream)?;
    gpu.synchronize(stream).with_context(|| {
        format!("GPU dequant_fp8_blockscaled_bf16 failed for {prefix} [{n},{k}]")
    })?;

    tracing::debug!(
        "GPU-dequanted FP8 blockscaled {prefix}: [{n}, {k}] block=[{block_n}, {block_k}] → BF16",
    );
    Ok(DenseWeight { weight: out })
}

/// Convert BF16 bytes (little-endian) to f32.
pub(super) fn bf16_bytes_to_f32(bytes: [u8; 2]) -> f32 {
    let bits = u16::from_le_bytes(bytes);
    f32::from_bits((bits as u32) << 16)
}

/// Load a dense weight, auto-detecting FP8 block-scaled vs BF16/FP32.
///
/// If the tensor is FP8E4M3 and a `{name_without_.weight}.weight_scale_inv` key exists,
/// performs block-scaled dequantization to BF16. FP32 dense tensors are converted
/// to BF16 because Atlas dense kernels consume BF16.
pub(crate) fn dense_auto(
    store: &WeightStore,
    name: &str,
    gpu: &dyn GpuBackend,
) -> Result<DenseWeight> {
    let w = store.get(name)?;
    match w.dtype {
        WeightDtype::BF16 => Ok(DenseWeight { weight: w.ptr }),
        WeightDtype::FP32 => dense_f32_safe(store, name, gpu),
        WeightDtype::FP8E4M3 => {
            // Derive prefix: "foo.q_proj.weight" -> "foo.q_proj".
            let prefix = name
                .strip_suffix(".weight")
                .ok_or_else(|| anyhow::anyhow!("FP8 tensor {name} doesn't end with .weight"))?;
            // Two FP8 scale conventions: block-scaled (DeepSeek / Qwen native
            // FP8) ships `weight_scale_inv` (2D), while per-tensor FP8 (nvidia
            // MIXED_PRECISION checkpoints, e.g. Qwen3.6-35B-A3B-NVFP4's attn +
            // linear_attn projections) ships a scalar `weight_scale`. Pick by
            // which one is present so MIXED_PRECISION loads instead of erroring
            // on the absent `weight_scale_inv` (issue #107).
            if store.contains(&format!("{prefix}.weight_scale_inv")) {
                dequant_fp8_blockscaled_to_bf16(store, prefix, gpu)
            } else {
                dequant_fp8_to_bf16(store, prefix, gpu)
            }
        }
        other => anyhow::bail!("dense_auto: unsupported dtype {:?} for {name}", other),
    }
}

/// Build a QuantizedWeight from Sehyo/compressed-tensors NVFP4 naming convention.
///
/// Sehyo quantization uses: weight_packed, weight_scale, weight_global_scale, input_global_scale
/// (vs standard: weight, weight_scale, weight_scale_2, input_scale).
///
/// **Scale convention difference**: compressed-tensors stores `weight_global_scale`
/// as the reciprocal of Atlas/TRT-LLM's `scale2`. Verified empirically:
///   - nvidia 80B `weight_scale_2` ≈ 7.01e-5 (small)
///   - Sehyo 35B `weight_global_scale` = 29568 → `1/29568` ≈ 3.38e-5 (same order)
///
/// Atlas GEMV dequant: `w = E2M1_val * fp8_scale * scale2` requires the small value.
pub(crate) fn quantized_v2(
    store: &WeightStore,
    prefix: &str,
    gpu: &dyn GpuBackend,
) -> Result<QuantizedWeight> {
    let raw_global_scale = scalar_f32(store, &format!("{prefix}.weight_global_scale"), gpu)?;
    // Guard against degenerate / corrupted checkpoints where
    // weight_global_scale is 0 — the unconditional 1/x would store
    // +inf into weight_scale_2 and silently NaN every dequant. Treat
    // it as a hard load error so the operator notices.
    if !raw_global_scale.is_finite() || raw_global_scale.abs() < f32::MIN_POSITIVE {
        anyhow::bail!(
            "{prefix}.weight_global_scale is non-finite or zero ({raw_global_scale}); \
             checkpoint likely corrupted"
        );
    }
    Ok(QuantizedWeight {
        weight: ptr(store, &format!("{prefix}.weight_packed"))?,
        weight_scale: ptr(store, &format!("{prefix}.weight_scale"))?,
        weight_scale_2: 1.0 / raw_global_scale,
        input_scale: ptr(store, &format!("{prefix}.input_global_scale"))?,
        weight_scale_2_vec: DevicePtr::NULL,
    })
}
