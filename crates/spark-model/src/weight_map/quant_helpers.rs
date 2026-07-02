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

    // Fast GPU path (merged main + V4): take the device dequant when the scale
    // is GPU-kernel-compatible — `weight_scale_inv` (DeepSeek-V3 / Qwen / MiniMax
    // native, 2-D BF16/FP32), OR a 2-D `weight_scale` (ModelOpt / compressed-
    // tensors mixed-precision, e.g. lovedheart AgentWorld-35B). Both are the
    // per-block multiplier `dequant_fp8_blockscaled_bf16` consumes identically.
    // V4 (`.scale`, F8_E8M0) and 1-D / scalar `weight_scale` are NOT taken here
    // (the GPU kernel handles neither E8M0 nor 1-D) — they fall through to the
    // CPU host-dequant below. NOTE: this must stay an `if let`/fall-through, not
    // an unconditional `store.get(scale_key)?` — V4 ships no `weight_scale*` at
    // all, so a hard get would error instead of reaching the CPU path.
    let gpu_scale = store
        .get(&format!("{prefix}.weight_scale_inv"))
        .ok()
        .or_else(|| {
            store
                .get(&format!("{prefix}.weight_scale"))
                .ok()
                .filter(|s| s.shape.len() == 2)
        });
    if let Some(s) = gpu_scale {
        ensure!(
            s.dtype == WeightDtype::BF16 || s.dtype == WeightDtype::FP32,
            "Expected BF16 or FP32 for {prefix} GPU block scale, got {:?}",
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
        return Ok(DenseWeight { weight: out });
    }

    // V4 / RedHatAI CPU fallback: checkpoints that ship `.weight_scale`
    // (RedHatAI compressed-tensors, BF16/FP32 2-D block) or `.scale`
    // (DeepSeek-V4 original, F8_E8M0 block scales). These layouts are not
    // handled by the GPU `dequant_fp8_blockscaled_bf16` kernel (no E8M0
    // support, different key), so dequant on the host. Download the FP8
    // weight once, apply the per-block scale, upload BF16.
    enum ScaleDtype {
        Fp32,
        Bf16,
        E8M0,
    }
    let mut fp8_buf = vec![0u8; total];
    gpu.copy_d2h(w.ptr, &mut fp8_buf).with_context(|| {
        format!(
            "D2H failed for {prefix}.weight: ptr={}, size={total}",
            w.ptr.0
        )
    })?;

    let (scale_buf, _sn, sk, block_n, block_k, scale_dtype) = if let Ok(s) =
        store.get(&format!("{prefix}.weight_scale"))
    {
        // RedHatAI / compressed-tensors block-scaled BF16/FP32.
        ensure!(
            s.dtype == WeightDtype::BF16 || s.dtype == WeightDtype::FP32,
            "Expected BF16 or FP32 2-D block scale for {prefix}.weight_scale, got {:?}",
            s.dtype,
        );
        let rank = s.shape.len();
        let (sn, sk) = if rank == 2 {
            (s.shape[0], s.shape[1])
        } else if rank == 1 {
            // Treat 1-D as per-row with single column block
            (s.shape[0], 1)
        } else {
            bail!(
                "Expected 1-D or 2-D scale for {prefix}.weight_scale, got shape {:?}",
                s.shape
            );
        };
        let block_n = if sn > 1 { n / sn } else { n };
        let block_k = if sk > 1 { k / sk } else { k };
        let scale_is_f32 = s.dtype == WeightDtype::FP32;
        let scale_bytes_per = if scale_is_f32 { 4 } else { 2 };
        let mut buf = vec![0u8; sn * sk * scale_bytes_per];
        gpu.copy_d2h(s.ptr, &mut buf).with_context(|| {
            format!(
                "D2H failed for {prefix}.weight_scale: ptr={}, size={}",
                s.ptr.0,
                sn * sk * scale_bytes_per
            )
        })?;
        let sd = if scale_is_f32 {
            ScaleDtype::Fp32
        } else {
            ScaleDtype::Bf16
        };
        (buf, sn, sk, block_n, block_k, sd)
    } else if let Ok(s) = store.get(&format!("{prefix}.scale")) {
        // DeepSeek-V4 block-scaled FP8 uses `.scale` with F8_E8M0 dtype.
        let rank = s.shape.len();
        let (sn, sk) = if rank == 2 {
            (s.shape[0], s.shape[1])
        } else if rank == 1 {
            // Treat 1-D scale as per-row (N) with single column
            (s.shape[0], 1)
        } else {
            bail!(
                "Expected 1-D or 2-D scale for {prefix}.scale, got shape {:?}",
                s.shape
            );
        };
        let block_n = if sn > 1 { n / sn } else { n };
        let block_k = if sk > 1 { k / sk } else { k };
        let sd = match s.dtype {
            WeightDtype::FP32 => ScaleDtype::Fp32,
            WeightDtype::BF16 => ScaleDtype::Bf16,
            WeightDtype::FP8E8M0 => ScaleDtype::E8M0,
            other => bail!(
                "Expected FP32, BF16, or FP8E8M0 for {prefix}.scale, got {:?}",
                other,
            ),
        };
        let scale_bytes_per = s.dtype.byte_size();
        let mut buf = vec![0u8; sn * sk * scale_bytes_per];
        gpu.copy_d2h(s.ptr, &mut buf).with_context(|| {
            format!(
                "D2H failed for {prefix}.scale: ptr={}, size={}",
                s.ptr.0,
                sn * sk * scale_bytes_per
            )
        })?;
        (buf, sn, sk, block_n, block_k, sd)
    } else {
        bail!(
            "FP8 tensor {prefix}: no .weight_scale_inv, .weight_scale, or .scale found for dequant"
        );
    };

    // CPU dequant: bf16_out[i,j] = fp8[i,j] * scale[i/block_n, j/block_k]
    let mut bf16_out = vec![0u8; total * 2];
    for row in 0..n {
        let scale_row = row / block_n;
        for col in 0..k {
            let scale_col = col / block_k;
            let scale_idx = scale_row * sk + scale_col;
            let scale_f32 = match scale_dtype {
                ScaleDtype::E8M0 => fp8_e8m0_to_f32(scale_buf[scale_idx]),
                ScaleDtype::Fp32 => {
                    let b = [
                        scale_buf[scale_idx * 4],
                        scale_buf[scale_idx * 4 + 1],
                        scale_buf[scale_idx * 4 + 2],
                        scale_buf[scale_idx * 4 + 3],
                    ];
                    f32::from_le_bytes(b)
                }
                ScaleDtype::Bf16 => {
                    let b = [scale_buf[scale_idx * 2], scale_buf[scale_idx * 2 + 1]];
                    bf16_bytes_to_f32(b)
                }
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

    let out = gpu.alloc(bf16_out.len())?;
    gpu.copy_h2d(&bf16_out, out)?;
    tracing::debug!(
        "CPU-dequanted FP8 blockscaled {prefix}: [{n}, {k}] block=[{block_n}, {block_k}] → BF16",
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
            // FP8 scale conventions (merged main + V4):
            // 1. block-scaled (DeepSeek-V3 / Qwen native FP8): `weight_scale_inv` (2D)
            // 2. per-tensor (nvidia MIXED_PRECISION, e.g. Qwen3.6-35B-A3B-NVFP4's
            //    attn + linear_attn projections): scalar `weight_scale` (1 element)
            // 3. block-scaled compressed-tensors (RedHatAI re-quant): 2-D / 1-D
            //    multi-element `weight_scale` (incl. ModelOpt mixed-precision, e.g.
            //    lovedheart AgentWorld-35B) / `.scale` (DeepSeek-V4, F8_E8M0)
            // Pick by which one is present so MIXED_PRECISION loads instead of
            // erroring on the absent `weight_scale_inv` (issue #107), while V4 /
            // RedHatAI block-scaled checkpoints route to the block dequant.
            // `num_elements() > 1` is the superset of main's `shape.len() == 2`
            // (also catches 1-D per-row scales); `has_v4_scale` keeps V4 on the
            // block path (main alone routed V4's `.scale` to the scalar path).
            let has_blockscale = store.contains(&format!("{prefix}.weight_scale_inv"));
            let has_per_row_scale = store
                .get(&format!("{prefix}.weight_scale"))
                .map(|s| s.num_elements() > 1)
                .unwrap_or(false);
            let has_v4_scale = store.contains(&format!("{prefix}.scale"));
            if has_blockscale || has_per_row_scale || has_v4_scale {
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
    })
}
