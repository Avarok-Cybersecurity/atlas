// SPDX-License-Identifier: AGPL-3.0-only

//! Auto-extracted from `weight_map.rs` during refactor wave 4a.

#![allow(unused_imports)]

use anyhow::{Context, Result, bail, ensure};
use spark_runtime::gpu::{DevicePtr, GpuBackend};
use spark_runtime::weights::{WeightDtype, WeightStore};

use super::*;

/// All model weights organized by layer.
pub struct ModelWeights {
    /// Embedding table: [vocab_size, hidden_size] BF16.
    pub embed_tokens: DenseWeight,
    /// Final RMS norm: `[hidden_size]` BF16.
    pub final_norm: DenseWeight,
    /// LM head: [hidden_size, vocab_size] BF16.
    pub lm_head: DenseWeight,
    /// Per-layer weights.
    pub layers: Vec<LayerWeights>,
}

/// Extract a DevicePtr from the weight store by name.
pub(crate) fn ptr(store: &WeightStore, name: &str) -> Result<DevicePtr> {
    Ok(store.get(name)?.ptr)
}

/// Extract a scalar f32 from a single-element weight tensor via D2H copy.
pub(crate) fn scalar_f32(store: &WeightStore, name: &str, gpu: &dyn GpuBackend) -> Result<f32> {
    let w = store.get(name)?;
    ensure!(
        w.dtype == WeightDtype::FP32,
        "Expected FP32 for {name}, got {:?}",
        w.dtype
    );
    ensure!(
        w.num_elements() == 1,
        "Expected scalar for {name}, got {} elements",
        w.num_elements()
    );
    let mut buf = [0u8; 4];
    gpu.copy_d2h(w.ptr, &mut buf)?;
    Ok(f32::from_le_bytes(buf))
}

/// Load FP8 KV cache quantization scales from a checkpoint.
///
/// Searches for `{attn_prefix}.k_proj.k_scale` and `{attn_prefix}.v_proj.v_scale`
/// tensors in the weight store. When found, returns the scalar f32 values from the
/// checkpoint (calibrated per-tensor scales for FP8 KV cache quantization).
///
/// When not found, returns (1.0, 1.0) with a debug-level log. This fallback is
/// correct for BF16 KV cache (scales are unused) and acceptable as uncalibrated
/// default for FP8 KV cache (Workstream 4C will add proper calibration).
pub(crate) fn load_kv_scales(
    store: &WeightStore,
    attn_prefix: &str,
    gpu: &dyn GpuBackend,
) -> (f32, f32) {
    let k_key = format!("{attn_prefix}.k_proj.k_scale");
    let v_key = format!("{attn_prefix}.v_proj.v_scale");

    let k_scale = if store.contains(&k_key) {
        match scalar_f32(store, &k_key, gpu) {
            Ok(v) => {
                tracing::debug!("Loaded k_scale={v:.6} from {k_key}");
                v
            }
            Err(e) => {
                tracing::warn!("Failed to load {k_key}: {e:#}, using 1.0");
                1.0
            }
        }
    } else {
        tracing::debug!("No {k_key} in checkpoint, using k_scale=1.0");
        1.0
    };

    let v_scale = if store.contains(&v_key) {
        match scalar_f32(store, &v_key, gpu) {
            Ok(v) => {
                tracing::debug!("Loaded v_scale={v:.6} from {v_key}");
                v
            }
            Err(e) => {
                tracing::warn!("Failed to load {v_key}: {e:#}, using 1.0");
                1.0
            }
        }
    } else {
        tracing::debug!("No {v_key} in checkpoint, using v_scale=1.0");
        1.0
    };

    (k_scale, v_scale)
}

/// Build a QuantizedWeight from the store using the standard NVFP4 naming.
///
/// `weight_scale_2` is a single FP32 scalar — extracted from GPU via D2H copy.
pub(crate) fn quantized(
    store: &WeightStore,
    prefix: &str,
    gpu: &dyn GpuBackend,
) -> Result<QuantizedWeight> {
    let input_scale_key = format!("{prefix}.input_scale");
    Ok(QuantizedWeight {
        weight: ptr(store, &format!("{prefix}.weight"))?,
        weight_scale: ptr(store, &format!("{prefix}.weight_scale"))?,
        weight_scale_2: scalar_f32(store, &format!("{prefix}.weight_scale_2"), gpu)?,
        input_scale: if store.contains(&input_scale_key) {
            ptr(store, &input_scale_key)?
        } else {
            DevicePtr::NULL
        },
        weight_scale_2_vec: DevicePtr::NULL,
    })
}

/// Native MXFP4 routed expert: land the on-disk bytes device-resident
/// **UNCHANGED** — no dequant, no re-quantize, no dtype coercion (the
/// transcode-free path, contrast `quantized_from_fp8` / the old E8M0
/// `dequant→quantize_to_nvfp4` arm that cost TWO lossy 4-bit conversions).
///
/// On-disk layout (DeepSeek-V4-Flash ORIGINAL routed format):
///   - `.weight` = 4-bit E2M1 nibbles, 2 per byte, stored U8/I8, shape `[n, k/2]`
///   - `.scale`  = F8_E8M0 per-block (biased exponent; scale = `2^(byte-127)`),
///     `GROUP_SIZE=32`, NO per-tensor global.
///
/// The buffer is tagged `WeightQuantFormat::Mxfp4E8m0` at the MoE-layer level
/// (`MoeWeights::experts_scale_kind`); the E8M0 kernel variants (Phase-K)
/// consume `weight_scale` as E8M0 bytes and ignore `weight_scale_2`. Asserts
/// the inferred block size is 32 so a non-MX checkpoint can't slip through.
pub(crate) fn quantized_mxfp4_e8m0(store: &WeightStore, prefix: &str) -> Result<QuantizedWeight> {
    let w = store.get(&format!("{prefix}.weight"))?;
    let n = w.shape[0];
    let k_packed = w.shape[1];
    let total_nibbles = n * k_packed * 2;
    let scale_t = store.get(&format!("{prefix}.scale"))?;
    let num_groups = scale_t.num_elements();
    ensure!(
        num_groups > 0 && total_nibbles.is_multiple_of(num_groups),
        "{prefix}: MXFP4 weight nibbles {total_nibbles} not divisible by E8M0 scale groups {num_groups}"
    );
    let block = total_nibbles / num_groups;
    ensure!(
        block == 32,
        "{prefix}: native MXFP4 expects GROUP_SIZE=32, inferred {block} (scale groups {num_groups}) \
         — refusing to land a non-MX checkpoint on the transcode-free path"
    );
    Ok(QuantizedWeight {
        weight: ptr(store, &format!("{prefix}.weight"))?,
        weight_scale: ptr(store, &format!("{prefix}.scale"))?,
        weight_scale_2: 1.0, // native MXFP4 has no per-tensor global
        input_scale: DevicePtr::NULL,
        // native MXFP4 uses the scalar `weight_scale_2` (E8M0 per-group), not
        // the per-output-row scale2 vector added by #257 → NULL.
        weight_scale_2_vec: DevicePtr::NULL,
    })
}

pub(crate) fn dense(store: &WeightStore, name: &str) -> Result<DenseWeight> {
    let w = store.get(name)?;
    Ok(DenseWeight { weight: w.ptr })
}

// REMOVED: `dense_minus_one` — the offset-from-1 norm loader.
//
// It pre-subtracted 1.0 and stored `bf16(w - 1)` so the offset-from-1 `rms_norm`
// kernel would recover `1 + (w - 1) = w`. That round-trip is only lossless when
// `w ≈ 1`. DeepSeek-V4 — its ONLY caller — ships HF-vanilla norm weights of
// ≈ 0.03, so it stored ≈ −0.97 and BF16's rounding error there (~1.9e-3 absolute)
// became a 1.8–3.4 % RELATIVE error on the weight once 1 was added back
// (catastrophic cancellation; up to 19 % on `q_norm`, 100 % with sign flips on the
// compressor norms — measured over all 249 V4 norm tensors, 2026-07-13).
//
// V4 now loads its norm weights exactly (`dense_auto`) and dispatches
// `rms_norm_vanilla`. See `crate::ships_vanilla_norm_weights`. Models whose norm
// weights are genuinely stored as an offset (Qwen3-Next, init 0) use `dense` and
// keep the offset kernel — they never needed this function.

/// Load a weight, auto-dequanting FP8 block-scaled to BF16 when needed.
///
/// Used for models with mixed-precision layers — Qwen3.6's ViT, for
/// example, keeps the first four blocks in BF16 but stores the rest as
/// FP8. Callers pass a prefix (e.g. `"model.visual.blocks.5.attn.qkv"`)
/// and get back a BF16 GPU buffer either way.
pub(crate) fn dense_auto_fp8_or_bf16(
    store: &WeightStore,
    prefix: &str,
    gpu: &dyn GpuBackend,
) -> Result<DenseWeight> {
    let w = store.get(&format!("{prefix}.weight"))?;
    match w.dtype {
        WeightDtype::BF16 => Ok(DenseWeight { weight: w.ptr }),
        WeightDtype::FP8E4M3 => dequant_fp8_blockscaled_to_bf16(store, prefix, gpu),
        other => anyhow::bail!(
            "dense_auto_fp8_or_bf16: unsupported dtype {:?} for {prefix}.weight",
            other
        ),
    }
}

/// Load a dense weight, converting FP32 → BF16 if needed. Used for norm weights
/// that may be FP32 in some checkpoints (e.g. Qwen FP8 `linear_attn.norm.weight`).
pub(crate) fn dense_f32_safe(
    store: &WeightStore,
    name: &str,
    gpu: &dyn GpuBackend,
) -> Result<DenseWeight> {
    let w = store.get(name)?;
    if w.dtype == WeightDtype::FP32 {
        // On-device FP32→BF16 truncation — ONE async kernel on the load stream,
        // no D2H/CPU/H2D round-trip. The old path did copy_d2h→CPU-truncate→
        // copy_h2d per weight, each with 2 cuStreamSynchronize on the busy load
        // stream → ~104s across 635 FP32 weights (the dominant cold-load cost).
        // The kernel reads the high 2 bytes of each f32 → bit-identical to the
        // prior CPU truncation. Ordered after the weight's upload (same stream),
        // so no sync needed.
        let n = w.num_elements();
        let ptr = gpu.alloc(n * 2)?;
        let trunc = gpu.kernel("quantize_nvfp4", "f32_to_bf16_trunc")?;
        let blocks = (n.div_ceil(256) as u32).max(1);
        spark_runtime::kernel_args::KernelLaunch::new(gpu, trunc)
            .grid([blocks, 1, 1])
            .block([256, 1, 1])
            .arg_ptr(w.ptr)
            .arg_ptr(ptr)
            .arg_u32(n as u32)
            .launch(gpu.default_stream())?;
        Ok(DenseWeight { weight: ptr })
    } else {
        Ok(DenseWeight { weight: w.ptr })
    }
}

/// Load a weight and ensure it's FP32 on GPU, regardless of source dtype.
///
/// Used for SSM gate parameters (A_log, dt_bias) where BF16 precision
/// causes exponential error amplification in the recurrent state at
/// long context (8k+ tokens). A 1-ULP BF16 error in the decay gate
/// produces (g_correct/g_error)^8000 ≈ 3000x magnitude divergence.
///
/// - FP32 in safetensors: keep as-is (no conversion)
/// - BF16 in safetensors: convert BF16 → FP32 via zero-extension
pub(crate) fn dense_keep_f32(
    store: &WeightStore,
    name: &str,
    gpu: &dyn GpuBackend,
) -> Result<DenseWeight> {
    let w = store.get(name)?;
    match w.dtype {
        WeightDtype::FP32 => {
            // Already FP32 — use directly, no conversion needed
            Ok(DenseWeight { weight: w.ptr })
        }
        WeightDtype::BF16 => {
            // Convert BF16 → FP32 to preserve precision
            tracing::info!(
                "dense_keep_f32: promoting {name} from BF16 to FP32 ({:?})",
                w.shape
            );
            let n = w.num_elements();
            let mut bf16_buf = vec![0u8; n * 2];
            gpu.copy_d2h(w.ptr, &mut bf16_buf)?;
            let f32_buf: Vec<u8> = bf16_buf
                .chunks_exact(2)
                .flat_map(|c| {
                    let bits = u16::from_le_bytes([c[0], c[1]]);
                    let f32_bits = (bits as u32) << 16;
                    f32_bits.to_le_bytes()
                })
                .collect();
            let ptr = gpu.alloc(f32_buf.len())?;
            gpu.copy_h2d(&f32_buf, ptr)?;
            Ok(DenseWeight { weight: ptr })
        }
        other => {
            bail!("dense_keep_f32: unsupported dtype {:?} for {name}", other);
        }
    }
}

/// Load a BF16 tensor and convert to F32 on-device via CPU roundtrip.
///
/// Used for Nemotron-H SSM parameters (A_log, D, dt_bias, conv1d.bias)
/// which are stored as BF16 in safetensors but consumed as F32 by CUDA kernels.
pub(crate) fn dense_bf16_as_f32(
    store: &WeightStore,
    name: &str,
    gpu: &dyn GpuBackend,
) -> Result<DenseWeight> {
    let w = store.get(name)?;
    ensure!(
        w.dtype == WeightDtype::BF16,
        "Expected BF16 for {name}, got {:?}",
        w.dtype
    );
    let n = w.num_elements();
    let mut bf16_buf = vec![0u8; n * 2];
    gpu.copy_d2h(w.ptr, &mut bf16_buf)?;
    let f32_buf: Vec<u8> = bf16_buf
        .chunks_exact(2)
        .flat_map(|c| {
            let bits = u16::from_le_bytes([c[0], c[1]]);
            let f32_bits = (bits as u32) << 16;
            f32_bits.to_le_bytes()
        })
        .collect();
    let ptr = gpu.alloc(f32_buf.len())?;
    gpu.copy_h2d(&f32_buf, ptr)?;
    Ok(DenseWeight { weight: ptr })
}

/// Load an F32 tensor and convert to BF16 on-device via CPU roundtrip.
///
/// Used for Nemotron-H gate weights (F32 in safetensors, consumed as BF16).
pub(crate) fn dense_f32_as_bf16(
    store: &WeightStore,
    name: &str,
    gpu: &dyn GpuBackend,
) -> Result<DenseWeight> {
    let w = store.get(name)?;
    ensure!(
        w.dtype == WeightDtype::FP32,
        "Expected FP32 for {name}, got {:?}",
        w.dtype
    );
    let n = w.num_elements();
    let mut f32_buf = vec![0u8; n * 4];
    gpu.copy_d2h(w.ptr, &mut f32_buf)?;
    let bf16_buf: Vec<u8> = f32_buf
        .chunks_exact(4)
        .flat_map(|c| {
            let val = f32::from_le_bytes([c[0], c[1], c[2], c[3]]);
            f32_to_bf16(val).to_le_bytes()
        })
        .collect();
    let ptr = gpu.alloc(bf16_buf.len())?;
    gpu.copy_h2d(&bf16_buf, ptr)?;
    Ok(DenseWeight { weight: ptr })
}
