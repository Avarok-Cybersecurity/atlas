// SPDX-License-Identifier: AGPL-3.0-only

//! FP8-E4M3 → BF16 per-tensor dequantization helpers.
//!
//! Extracted from `model_a.rs` (Wave: ARM-2 native-MXFP4) to keep it under the
//! 500-LoC cap. The GPU fast-path kernel (`dequant_fp8_blockscaled_bf16` as the
//! single-block degenerate case) + the BF16/FP32-scalar host fallback used by
//! the SSM/dense FP8 loaders (nemotron, quant_helpers, …). Re-exported from
//! `weight_map.rs` (`pub use fp8_dequant::*`), so all call sites are unchanged.

#![allow(unused_imports)]

use anyhow::{Context, Result, bail, ensure};
use spark_runtime::gpu::{DevicePtr, GpuBackend};
use spark_runtime::weights::{WeightDtype, WeightStore};

use super::*;

/// GPU-dequant an FP8 E4M3 + per-tensor scalar-scale weight into `out` (BF16).
///
/// Reuses the byte-exact `dequant_fp8_blockscaled_bf16` kernel: a per-tensor
/// scalar is the single-block degenerate case (`block_n=N, block_k=K, sk=1`),
/// so every element reads `weight_scale[0]`. The scalar `weight_scale` already
/// lives on-device as an FP32 element, so its `.ptr` is passed straight in —
/// no D2H download, no host loop, no H2D upload.
///
/// Replaces the old `copy_d2h` + single-threaded per-byte CPU `flat_map`
/// (`dequant_fp8_bytes_to_bf16`) that dominated cold load: ~5 SSM projections
/// × 30 layers over multi-million-element tensors ≈ 80s. Math matches the CPU
/// path 1:1 (`E4M3_LUT[b] * scale` in f32 → RNE → bf16), validated
/// token-for-token against the copies path.
fn gpu_dequant_fp8_pertensor(
    store: &WeightStore,
    prefix: &str,
    gpu: &dyn GpuBackend,
    out: DevicePtr,
) -> Result<()> {
    use spark_runtime::kernel_args::{KernelLaunch, div_ceil};

    let w = store.get(&format!("{prefix}.weight"))?;
    ensure!(
        w.dtype == WeightDtype::FP8E4M3,
        "Expected FP8E4M3 for {prefix}.weight, got {:?}",
        w.dtype,
    );
    // Any factorization N*K = total works (the scale is constant across the
    // whole tensor); use the real 2D shape for occupancy, else flatten.
    let total = w.num_elements();
    let (n, k) = if w.shape.len() == 2 {
        (w.shape[0], w.shape[1])
    } else {
        (1usize, total)
    };
    ensure!(
        n * k == total,
        "FP8 shape mismatch for {prefix}: {n}*{k} != {total}"
    );

    let s = store.get(&format!("{prefix}.weight_scale"))?;
    ensure!(
        s.dtype == WeightDtype::FP32 && s.num_elements() == 1,
        "Expected FP32 scalar weight_scale for {prefix}, got {:?} ({} elems)",
        s.dtype,
        s.num_elements(),
    );

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
        .arg_u32(n as u32) // block_n = N → sn_idx always 0
        .arg_u32(k as u32) // block_k = K → sk_idx always 0
        .arg_u32(1) // sk = 1 (single scale element)
        .arg_u32(1) // scale_is_fp32 = 1
        .launch(stream)?;
    // No per-call sync: kernel runs async on the load stream; `out` is consumed
    // by later same-stream ops (CUDA orders them). A fault surfaces at the next
    // real sync. (Syncing here is the per-layer stall the cold-load fix removed.)
    Ok(())
}

/// Read a per-tensor FP8 `weight_scale` scalar (FP32 or BF16 — RedHatAI re-quants
/// ship BF16) from the store as `f32`. Shared host fallback for the per-tensor
/// dequant paths below.
fn read_scalar_weight_scale(
    store: &WeightStore,
    prefix: &str,
    gpu: &dyn GpuBackend,
) -> Result<f32> {
    let scale_key = format!("{prefix}.weight_scale");
    let s = store.get(&scale_key)?;
    ensure!(
        s.num_elements() == 1,
        "Expected scalar for {scale_key}, got {} elements",
        s.num_elements()
    );
    match s.dtype {
        WeightDtype::FP32 => {
            let mut buf = [0u8; 4];
            gpu.copy_d2h(s.ptr, &mut buf)?;
            Ok(f32::from_le_bytes(buf))
        }
        WeightDtype::BF16 => {
            let mut buf = [0u8; 2];
            gpu.copy_d2h(s.ptr, &mut buf)?;
            Ok(bf16_bytes_to_f32(buf))
        }
        other => bail!("Expected FP32 or BF16 for {scale_key}, got {:?}", other),
    }
}

/// Dequantize FP8 E4M3 + per-tensor scale → BF16, returning a DenseWeight.
///
/// Allocates a new GPU buffer for the result. Use `dequant_fp8_to_bf16_into` to
/// write into a pre-allocated scratch buffer instead (avoids gpu.free on GB10 UVM).
pub(crate) fn dequant_fp8_to_bf16(
    store: &WeightStore,
    prefix: &str,
    gpu: &dyn GpuBackend,
) -> Result<DenseWeight> {
    let w = store.get(&format!("{prefix}.weight"))?;
    // Per-tensor FP8 dequant. Fast path: GPU `gpu_dequant_fp8_pertensor` — but it
    // only supports an FP32 scalar scale. RedHatAI re-quants (e.g.
    // DeepSeek-V4-Flash) ship a BF16 scalar; the host fallback below handles both
    // FP32 and BF16 scalars.
    let scale_is_fp32_scalar = store
        .get(&format!("{prefix}.weight_scale"))
        .map(|s| s.dtype == WeightDtype::FP32 && s.num_elements() == 1)
        .unwrap_or(false);
    if scale_is_fp32_scalar {
        let total = w.num_elements();
        let out = gpu.alloc(total * 2)?;
        gpu_dequant_fp8_pertensor(store, prefix, gpu, out)?;
        Ok(DenseWeight { weight: out })
    } else {
        let n_bytes = w.num_elements();
        let mut fp8_buf = vec![0u8; n_bytes];
        gpu.copy_d2h(w.ptr, &mut fp8_buf)?;

        // RedHatAI re-quant checkpoints store per-tensor scale as BF16, not FP32.
        let scale = read_scalar_weight_scale(store, prefix, gpu)?;
        let bf16_buf = dequant_fp8_bytes_to_bf16(&fp8_buf, scale);
        let ptr = gpu.alloc(bf16_buf.len())?;
        gpu.copy_h2d(&bf16_buf, ptr)?;
        Ok(DenseWeight { weight: ptr })
    }
}

/// Dequantize FP8 E4M3 → BF16, writing into a pre-allocated destination buffer.
///
/// Avoids gpu.alloc/free for the intermediate BF16 data. The caller provides a
/// reusable scratch buffer that is overwritten each call. Safe to reuse after
/// `quantize_to_nvfp4` returns (it synchronizes internally).
pub(crate) fn dequant_fp8_to_bf16_into(
    store: &WeightStore,
    prefix: &str,
    gpu: &dyn GpuBackend,
    dest: DevicePtr,
) -> Result<DenseWeight> {
    // FP32-scalar → GPU fast path; BF16-scalar (RedHatAI re-quant) → host fallback.
    let scale_is_fp32_scalar = store
        .get(&format!("{prefix}.weight_scale"))
        .map(|s| s.dtype == WeightDtype::FP32 && s.num_elements() == 1)
        .unwrap_or(false);
    if scale_is_fp32_scalar {
        gpu_dequant_fp8_pertensor(store, prefix, gpu, dest)?;
    } else {
        let w = store.get(&format!("{prefix}.weight"))?;
        let n_bytes = w.num_elements();
        let mut fp8_buf = vec![0u8; n_bytes];
        gpu.copy_d2h(w.ptr, &mut fp8_buf)?;

        let scale = read_scalar_weight_scale(store, prefix, gpu)?;
        let bf16_buf = dequant_fp8_bytes_to_bf16(&fp8_buf, scale);
        gpu.copy_h2d(&bf16_buf, dest)?;
    }
    Ok(DenseWeight { weight: dest })
}
