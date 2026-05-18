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

/// Dequantize FP8 E4M3 block-scaled weight → BF16.
///
/// Block-scaled FP8 (e.g. `quant_method: "fp8"` with `weight_block_size: [128, 128]`):
///   - `{prefix}.weight`: FP8E4M3 tensor of shape `[N, K]`
///   - `{prefix}.weight_scale_inv`: BF16 tensor of shape `[N/block, K/block]`
///   - Dequant: `bf16[i,j] = fp8[i,j] * scale_inv[i/block, j/block]`
///
/// Returns a BF16 DenseWeight on GPU.
pub(crate) fn dequant_fp8_blockscaled_to_bf16(
    store: &WeightStore,
    prefix: &str,
    gpu: &dyn GpuBackend,
) -> Result<DenseWeight> {
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
    tracing::debug!(
        "FP8 blockscaled dequant: {prefix} shape=[{n},{k}] total={total} byte_size={byte_size} ptr={}",
        w.ptr.0,
    );

    // Sync to flush any pending CUDA errors from prior operations
    gpu.synchronize(gpu.default_stream())?;

    // Download FP8 weight bytes (1 byte per element)
    ensure!(
        total == byte_size,
        "FP8 size mismatch: total={total} byte_size={byte_size}"
    );
    let mut fp8_buf = vec![0u8; byte_size];
    gpu.copy_d2h(w.ptr, &mut fp8_buf).with_context(|| {
        let free = gpu.free_memory().unwrap_or(0);
        format!(
            "D2H failed for {prefix}.weight: ptr={}, size={byte_size}, free={:.1} GB",
            w.ptr.0,
            free as f64 / (1024.0 * 1024.0 * 1024.0),
        )
    })?;

    // Download block scale_inv. DeepSeek-V3 / Qwen FP8 ship BF16 here;
    // MiniMax-M2 ships FP32. Handle both — the subsequent dequant reads
    // a scalar `f32` scale per (row, col) either way.
    let s = store.get(&format!("{prefix}.weight_scale_inv"))?;
    ensure!(
        s.dtype == WeightDtype::BF16 || s.dtype == WeightDtype::FP32,
        "Expected BF16 or FP32 for {prefix}.weight_scale_inv, got {:?}",
        s.dtype,
    );
    let sn = s.shape[0];
    let sk = s.shape[1];
    let block_n = n / sn;
    let block_k = k / sk;
    let scale_is_f32 = s.dtype == WeightDtype::FP32;
    let scale_bytes_per = if scale_is_f32 { 4 } else { 2 };
    let mut scale_buf = vec![0u8; sn * sk * scale_bytes_per];
    gpu.copy_d2h(s.ptr, &mut scale_buf).with_context(|| {
        format!(
            "D2H failed for {prefix}.weight_scale_inv: ptr={}, size={}",
            s.ptr.0,
            sn * sk * scale_bytes_per
        )
    })?;

    // CPU dequant: bf16_out[i,j] = fp8[i,j] * scale_inv[i/block_n, j/block_k]
    let mut bf16_out = vec![0u8; total * 2];
    for row in 0..n {
        let scale_row = row / block_n;
        for col in 0..k {
            let scale_col = col / block_k;
            let scale_idx = scale_row * sk + scale_col;
            let scale_f32 = if scale_is_f32 {
                let b = [
                    scale_buf[scale_idx * 4],
                    scale_buf[scale_idx * 4 + 1],
                    scale_buf[scale_idx * 4 + 2],
                    scale_buf[scale_idx * 4 + 3],
                ];
                f32::from_le_bytes(b)
            } else {
                let b = [scale_buf[scale_idx * 2], scale_buf[scale_idx * 2 + 1]];
                bf16_bytes_to_f32(b)
            };

            let fp8_byte = fp8_buf[row * k + col];
            let val = fp8_e4m3_to_f32(fp8_byte) * scale_f32;
            let bf16_val = f32_to_bf16(val);

            let out_idx = (row * k + col) * 2;
            let [lo, hi] = bf16_val.to_le_bytes();
            bf16_out[out_idx] = lo;
            bf16_out[out_idx + 1] = hi;
        }
    }

    // Diagnostic: print weight statistics for first few dequants
    {
        use std::sync::atomic::{AtomicUsize, Ordering};
        static DIAG_COUNT: AtomicUsize = AtomicUsize::new(0);
        let count = DIAG_COUNT.fetch_add(1, Ordering::Relaxed);
        if count < 3 {
            let mut min_val = f32::MAX;
            let mut max_val = f32::MIN;
            let mut sum = 0.0f64;
            let mut zeros = 0usize;
            for i in 0..total {
                let lo = bf16_out[i * 2];
                let hi = bf16_out[i * 2 + 1];
                let v = bf16_bytes_to_f32([lo, hi]);
                if v == 0.0 {
                    zeros += 1;
                }
                if v < min_val {
                    min_val = v;
                }
                if v > max_val {
                    max_val = v;
                }
                sum += v as f64;
            }
            let mean = sum / total as f64;
            tracing::info!(
                "FP8 dequant stats for {prefix}: min={min_val:.6}, max={max_val:.6}, mean={mean:.6}, zeros={zeros}/{total}"
            );
            // First 8 values
            let vals: Vec<f32> = (0..8.min(total))
                .map(|i| bf16_bytes_to_f32([bf16_out[i * 2], bf16_out[i * 2 + 1]]))
                .collect();
            tracing::info!("  First 8 BF16 values: {:?}", vals);
        }
    }

    let ptr = gpu.alloc(bf16_out.len())?;
    gpu.copy_h2d(&bf16_out, ptr)?;

    // Diagnostic: readback first 8 BF16 values from GPU and compare with CPU
    {
        use std::sync::atomic::{AtomicUsize, Ordering};
        static VERIFY_COUNT: AtomicUsize = AtomicUsize::new(0);
        if VERIFY_COUNT.fetch_add(1, Ordering::Relaxed) < 3 {
            let check_len = 16.min(bf16_out.len());
            let mut readback = vec![0u8; check_len];
            if gpu.copy_d2h(ptr, &mut readback).is_ok() {
                let match_ok = readback[..check_len] == bf16_out[..check_len];
                if !match_ok {
                    tracing::error!(
                        "BF16 GPU readback MISMATCH for {prefix}: cpu={:?} gpu={:?}",
                        &bf16_out[..check_len],
                        &readback[..check_len],
                    );
                } else {
                    tracing::info!("BF16 GPU readback verified OK for {prefix}");
                }
            }
        }
    }

    tracing::debug!(
        "Dequanted FP8 blockscaled {prefix}: [{n}, {k}] block=[{block_n}, {block_k}] → BF16",
    );
    Ok(DenseWeight { weight: ptr })
}

/// Convert BF16 bytes (little-endian) to f32.
pub(super) fn bf16_bytes_to_f32(bytes: [u8; 2]) -> f32 {
    let bits = u16::from_le_bytes(bytes);
    f32::from_bits((bits as u32) << 16)
}

/// Load a dense weight, auto-detecting FP8 block-scaled vs BF16.
///
/// If the tensor is FP8E4M3 and a `{name_without_.weight}.weight_scale_inv` key exists,
/// performs block-scaled dequantization to BF16. Otherwise returns the raw pointer (BF16).
pub(crate) fn dense_auto(
    store: &WeightStore,
    name: &str,
    gpu: &dyn GpuBackend,
) -> Result<DenseWeight> {
    let w = store.get(name)?;
    if w.dtype == WeightDtype::FP8E4M3 {
        // Derive prefix: "foo.q_proj.weight" → "foo.q_proj"
        let prefix = name
            .strip_suffix(".weight")
            .ok_or_else(|| anyhow::anyhow!("FP8 tensor {name} doesn't end with .weight"))?;
        dequant_fp8_blockscaled_to_bf16(store, prefix, gpu)
    } else {
        Ok(DenseWeight { weight: w.ptr })
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
    })
}
