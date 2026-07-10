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

pub(crate) fn dense(store: &WeightStore, name: &str) -> Result<DenseWeight> {
    let w = store.get(name)?;
    Ok(DenseWeight { weight: w.ptr })
}

/// Load a BF16 norm weight and subtract 1.0 from every element.
///
/// Atlas's `rms_norm` kernel uses the Qwen3-Next "offset-from-1" convention
/// (`out = x * (1 + weight)`). Models with STANDARD RMSNorm (`out = x * weight`,
/// e.g. DeepSeek-V4: `DeepseekV4RMSNorm` = T5LayerNorm) must pre-subtract 1.0 so
/// the kernel computes `1 + (w - 1) = w`. Without this, every norm is scaled
/// wrong (e.g. kv_norm 2.6x, attn_norm ~30x too large) → attention overflow.
pub(crate) fn dense_minus_one(
    store: &WeightStore,
    name: &str,
    gpu: &dyn GpuBackend,
) -> Result<DenseWeight> {
    let w = store.get(name)?;
    let n = w.num_elements();
    let mut bf16_buf = vec![0u8; n * 2];
    gpu.copy_d2h(w.ptr, &mut bf16_buf)?;
    let adjusted: Vec<u8> = bf16_buf
        .chunks_exact(2)
        .flat_map(|c| {
            let bits = u16::from_le_bytes([c[0], c[1]]);
            let v = f32::from_bits((bits as u32) << 16) - 1.0;
            // round-to-nearest-even bf16 truncation
            let u = v.to_bits();
            let round_bit = (u >> 15) & 1;
            let sticky = (u & 0x7FFF != 0) as u32;
            let bf =
                ((u >> 16) as u16).wrapping_add((round_bit & (sticky | ((u >> 16) & 1))) as u16);
            bf.to_le_bytes()
        })
        .collect();
    let ptr = gpu.alloc(adjusted.len())?;
    gpu.copy_h2d(&adjusted, ptr)?;
    Ok(DenseWeight { weight: ptr })
}

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
