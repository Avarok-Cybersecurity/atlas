// SPDX-License-Identifier: AGPL-3.0-only

//! Auto-extracted from `weight_map.rs` during refactor wave 4a.

#![allow(unused_imports)]

use anyhow::{Context, Result, bail, ensure};
use spark_runtime::gpu::{DevicePtr, GpuBackend};
use spark_runtime::weights::{WeightDtype, WeightStore};

use super::*;

/// Dequantize an NVFP4 weight to BF16 on CPU, then upload to GPU.
///
/// Used at load time when projections are NVFP4-quantized on disk but need
/// BF16 format for dense GEMV/GEMM. One-time cost, not on hot path.
///
/// Auto-detects format:
/// - **compressed-tensors**: `weight_packed`, `weight_scale`, `weight_global_scale` (reciprocal)
/// - **Standard (modelopt)**: `weight`, `weight_scale`, `weight_scale_2` (direct multiplier)
pub(crate) fn dequant_nvfp4_to_bf16(
    store: &WeightStore,
    prefix: &str,
    n: usize,
    k: usize,
    gpu: &dyn GpuBackend,
) -> Result<DenseWeight> {
    let total = n * k;

    // Auto-detect format: compressed-tensors vs Standard
    let (packed_ptr, scale_ptr, global_scale, is_reciprocal) =
        if store.contains(&format!("{prefix}.weight_packed")) {
            // compressed-tensors: global_scale is reciprocal
            let pp = ptr(store, &format!("{prefix}.weight_packed"))?;
            let sp = ptr(store, &format!("{prefix}.weight_scale"))?;
            let gs = scalar_f32(store, &format!("{prefix}.weight_global_scale"), gpu)?;
            (pp, sp, gs, true)
        } else {
            // Standard/modelopt: weight_scale_2 is direct multiplier
            let pp = ptr(store, &format!("{prefix}.weight"))?;
            let sp = ptr(store, &format!("{prefix}.weight_scale"))?;
            let gs = scalar_f32(store, &format!("{prefix}.weight_scale_2"), gpu)?;
            (pp, sp, gs, false)
        };

    // Fold the global-scale convention into a single MULTIPLY for the kernel:
    // compressed-tensors stores a RECIPROCAL global (val = E2M1 * fp8_scale /
    // global), ModelOpt a direct multiplier (val = E2M1 * fp8_scale * global).
    let combined_global = if is_reciprocal {
        if global_scale != 0.0 {
            1.0 / global_scale
        } else {
            0.0
        }
    } else {
        global_scale
    };

    // GPU dequant — replaces the former D2H(packed+scales) + 83M-element
    // single-threaded CPU loop + H2D (the real cost of the NVFP4→BF16→NVFP4
    // fused-qkvz round-trip: ~8s per dense-27B SSM layer). Same math, on-device.
    // One sync so the BF16 is ready for the caller (gpu_concat_rows / requant).
    let out = gpu.alloc(total * 2)?;
    let kernel = gpu.kernel("dequant_nvfp4_bf16", "dequant_nvfp4_to_bf16")?;
    let stream = gpu.default_stream();
    spark_runtime::kernel_args::KernelLaunch::new(gpu, kernel)
        .grid([n as u32, 1, 1])
        .block([256, 1, 1])
        .arg_ptr(packed_ptr)
        .arg_ptr(scale_ptr)
        .arg_ptr(out)
        .arg_f32(combined_global)
        .arg_u32(n as u32)
        .arg_u32(k as u32)
        .launch(stream)?;
    gpu.synchronize(stream)?;
    Ok(DenseWeight { weight: out })
}

/// Dequantize an NVFP4 weight with **E8M0** (power-of-2) per-block scales and
/// **no global scale** to BF16 on CPU, then upload. This is DeepSeek-V4's
/// ORIGINAL microscaling format (used by the MTP module's routed experts):
/// `.weight` = 4-bit-packed E2M1 (2 per byte, stored U8/I8) + `.scale` =
/// F8_E8M0 per block. The block size is inferred from the scale element count
/// (`total / num_scale_elems`, e.g. 32) rather than hardcoded. One-time load cost.
// ARM-2: no longer called on the load path — the native MXFP4 arm
// (`quantized_mxfp4_e8m0`) lands E8M0 bytes device-resident transcode-free
// instead of dequant→requantize. RETAINED as the correct host-side E8M0→BF16
// reference for Phase-K Leg-2 (kernel dequant numeric check on synthetic tiles).
#[allow(dead_code)]
pub(crate) fn dequant_nvfp4_e8m0_to_bf16(
    store: &WeightStore,
    prefix: &str,
    n: usize,
    k: usize,
    gpu: &dyn GpuBackend,
) -> Result<DenseWeight> {
    let total = n * k;
    let packed_bytes = total / 2;
    let packed_ptr = ptr(store, &format!("{prefix}.weight"))?;
    let scale_t = store.get(&format!("{prefix}.scale"))?;
    let num_groups = scale_t.num_elements();
    ensure!(
        num_groups > 0 && total.is_multiple_of(num_groups),
        "{prefix}: weight elems {total} not divisible by E8M0 scale groups {num_groups}"
    );
    let block = total / num_groups;

    let mut packed = vec![0u8; packed_bytes];
    let mut scales = vec![0u8; num_groups]; // FP8 E8M0, 1 byte each
    gpu.copy_d2h(packed_ptr, &mut packed)?;
    gpu.copy_d2h(scale_t.ptr, &mut scales)?;

    let e2m1_table: [f32; 16] = [
        0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
    ];
    // Row-major weight [n,k] and scale [n, k/block] → scale group `g` covers
    // weight flat indices `g*block .. (g+1)*block` (same nibble convention as
    // dequant_nvfp4_to_bf16: even flat index = low nibble).
    let mut bf16_out = vec![0u16; total];
    for group in 0..num_groups {
        let block_scale = fp8_e8m0_to_f32(scales[group]);
        for elem in 0..block {
            let flat_idx = group * block + elem;
            let byte_idx = flat_idx / 2;
            let nibble = if flat_idx.is_multiple_of(2) {
                packed[byte_idx] & 0x0F
            } else {
                (packed[byte_idx] >> 4) & 0x0F
            };
            bf16_out[flat_idx] = f32_to_bf16(e2m1_table[nibble as usize] * block_scale);
        }
    }

    let buf = gpu.alloc(total * 2)?;
    // SAFETY: `bf16_out` is `vec![0u16; total]`, so `bf16_out.len() == total` and
    // every element is initialised (zeroed at construction, then overwritten by the
    // `group`/`elem` dequant loop above). `total * 2 == bf16_out.len() *
    // size_of::<u16>()`, so the span is exactly the Vec's buffer. Shared borrow
    // only; `buf` was allocated at `total * 2` bytes so the H2D destination matches.
    let bf16_bytes: &[u8] =
        unsafe { std::slice::from_raw_parts(bf16_out.as_ptr() as *const u8, total * 2) };
    gpu.copy_h2d(bf16_bytes, buf)?;
    Ok(DenseWeight { weight: buf })
}

/// FP8 E4M3 decode and the f32 → BF16 cast live in `atlas_core::numeric`.
///
/// They used to live here AND in `atlas-quant/src/fp8.rs`, with the
/// byte-exactness tests attached to the copy that had zero dependents and
/// therefore never ran on any serving path. Both crates already depended
/// on `atlas-core`, so the arithmetic moved down there together with the
/// RNE and PyTorch-parity vectors. Re-exported under the old names so
/// every call site in this module is unchanged.
pub(super) use atlas_core::numeric::{FP8_E4M3_LUT, f32_to_bf16, fp8_e4m3_to_f32};

/// FP8 E8M0 → f32 lookup table (256 entries).
///
/// E8M0 format: unsigned 8-bit exponent, 0 mantissa, bias=127.
/// Value = 2^(exp - 127). exp=0 → 0, exp=255 → NaN (stored as 0.0).
const FP8_E8M0_LUT: [f32; 256] = {
    let mut table = [0.0f32; 256];
    let mut i: u32 = 0;
    while i < 256 {
        let exp = i as u8;
        table[i as usize] = if exp == 0 {
            0.0f32
        } else if exp == 255 {
            0.0f32 // NaN weight-scales should not appear in practice
        } else {
            f32::from_bits((exp as u32) << 23)
        };
        i += 1;
    }
    table
};

/// Convert FP8 E8M0 byte to f32 via LUT (branchless, single array lookup).
#[inline(always)]
pub(super) fn fp8_e8m0_to_f32(bits: u8) -> f32 {
    FP8_E8M0_LUT[bits as usize]
}

/// Load dense FFN weights (gate_proj, up_proj, down_proj) as NVFP4.
///
/// Used by non-MoE models (e.g. Qwen3.5-27B) where the MLP is a standard
/// SwiGLU FFN instead of a mixture of experts.
pub(crate) fn load_dense_ffn(
    store: &WeightStore,
    prefix: &str,
    gpu: &dyn GpuBackend,
    variant: Nvfp4Variant,
    absmax_k: spark_runtime::gpu::KernelHandle,
    quantize_k: spark_runtime::gpu::KernelHandle,
    stream: u64,
    config: &atlas_core::config::ModelConfig,
) -> Result<crate::layers::dense_ffn::DenseFfnWeights> {
    use crate::layers::dense_ffn::DenseFfnWeights;
    match variant {
        Nvfp4Variant::Fp8Dequanted => {
            // Dense FFN uses `intermediate_size` (the standard SwiGLU FFN width).
            // `moe_intermediate_size` is the per-expert width for MoE models and
            // is unset (=0) for dense Qwen3.6-27B-FP8 — using it would request a
            // 0-byte allocation in `quantize_to_nvfp4`. Fall back to
            // `moe_intermediate_size` when it's set and `intermediate_size` is
            // not, to preserve compatibility with prior MoE-style configs.
            let inter = if config.intermediate_size > 0 {
                config.intermediate_size
            } else {
                config.moe_intermediate_size
            };
            let h = config.hidden_size;
            let gate = quantized_from_fp8(
                store,
                &format!("{prefix}.mlp.gate_proj"),
                inter,
                h,
                gpu,
                absmax_k,
                quantize_k,
                stream,
            )?;
            let up = quantized_from_fp8(
                store,
                &format!("{prefix}.mlp.up_proj"),
                inter,
                h,
                gpu,
                absmax_k,
                quantize_k,
                stream,
            )?;
            let down = quantized_from_fp8(
                store,
                &format!("{prefix}.mlp.down_proj"),
                h,
                inter,
                gpu,
                absmax_k,
                quantize_k,
                stream,
            )?;
            // Transposed copies for the fast w4a16_gemm_t_m128 prefill kernel.
            Ok(DenseFfnWeights {
                gate_proj: gate,
                up_proj: up,
                down_proj: down,
                gate_proj_t: Some(gate.transpose_for_gemm(gpu, inter, h)?),
                up_proj_t: Some(up.transpose_for_gemm(gpu, inter, h)?),
                down_proj_t: Some(down.transpose_for_gemm(gpu, h, inter)?),
            })
        }
        Nvfp4Variant::Bf16Raw => {
            // Raw BF16 fine-tune (e.g. Holo-3.1-0.8B / Ornith dense): the dense
            // FFN ships un-quantized, so runtime-quantize BF16→NVFP4 via
            // `quantized_any` (the `quantized_auto` path used below for
            // Standard/CompressedTensors `unreachable!`s on Bf16Raw — it lacks
            // the dims + quant kernels the runtime-quant needs).
            let inter = if config.intermediate_size > 0 {
                config.intermediate_size
            } else {
                config.moe_intermediate_size
            };
            let h = config.hidden_size;
            let qctx = QuantizeCtx {
                absmax_k,
                quantize_k,
                stream,
            };
            let gate = quantized_any(
                store,
                &format!("{prefix}.mlp.gate_proj"),
                inter,
                h,
                gpu,
                variant,
                qctx,
            )?;
            let up = quantized_any(
                store,
                &format!("{prefix}.mlp.up_proj"),
                inter,
                h,
                gpu,
                variant,
                qctx,
            )?;
            let down = quantized_any(
                store,
                &format!("{prefix}.mlp.down_proj"),
                h,
                inter,
                gpu,
                variant,
                qctx,
            )?;
            Ok(DenseFfnWeights {
                gate_proj: gate,
                up_proj: up,
                down_proj: down,
                gate_proj_t: Some(gate.transpose_for_gemm(gpu, inter, h)?),
                up_proj_t: Some(up.transpose_for_gemm(gpu, inter, h)?),
                down_proj_t: Some(down.transpose_for_gemm(gpu, h, inter)?),
            })
        }
        _ => {
            // `quantized_any`, not `quantized_auto`: mixed-precision
            // compressed-tensors checkpoints (unsloth Qwen3.6-*-NVFP4,
            // re-quantized 2026-07-10) leave a tail of dense-FFN layers as FP8
            // E4M3 + per-row `weight_scale` inside an otherwise-NVFP4 net.
            // `quantized_auto` would take the declared variant at face value and
            // die on the absent `weight_global_scale`; `quantized_any` detects
            // the per-key layout and dequant→requantizes those keys. It
            // dispatches identically for genuinely-NVFP4 keys.
            let inter_ = if config.intermediate_size > 0 {
                config.intermediate_size
            } else {
                config.moe_intermediate_size
            };
            let h_ = config.hidden_size;
            let qctx = QuantizeCtx {
                absmax_k,
                quantize_k,
                stream,
            };
            let gate = quantized_any(
                store,
                &format!("{prefix}.mlp.gate_proj"),
                inter_,
                h_,
                gpu,
                variant,
                qctx,
            )?;
            let up = quantized_any(
                store,
                &format!("{prefix}.mlp.up_proj"),
                inter_,
                h_,
                gpu,
                variant,
                qctx,
            )?;
            let down = quantized_any(
                store,
                &format!("{prefix}.mlp.down_proj"),
                h_,
                inter_,
                gpu,
                variant,
                qctx,
            )?;
            let inter = if config.intermediate_size > 0 {
                config.intermediate_size
            } else {
                config.moe_intermediate_size
            };
            let h = config.hidden_size;
            Ok(DenseFfnWeights {
                gate_proj: gate,
                up_proj: up,
                down_proj: down,
                gate_proj_t: Some(gate.transpose_for_gemm(gpu, inter, h)?),
                up_proj_t: Some(up.transpose_for_gemm(gpu, inter, h)?),
                down_proj_t: Some(down.transpose_for_gemm(gpu, h, inter)?),
            })
        }
    }
}

/// Load MTP head weights for Qwen3.5.
/// Same key patterns as 80B MTP but with 256 experts.
#[allow(dead_code)]
pub(crate) fn load_mtp_qwen35(
    store: &WeightStore,
    num_experts: usize,
    gpu: &dyn GpuBackend,
    variant: Nvfp4Variant,
) -> Result<MtpWeights> {
    load_mtp(store, num_experts, gpu, variant)
}

/// GPU-concatenate two weight matrices row-wise: [A; B] → [A_rows + B_rows, K].
///
/// Both inputs must be contiguous BF16 matrices with the same K dimension.
/// Returns a new DenseWeight on GPU with the concatenated data.
pub(crate) fn gpu_concat_rows(
    a: &DenseWeight,
    a_rows: usize,
    b: &DenseWeight,
    b_rows: usize,
    k: usize,
    gpu: &dyn GpuBackend,
) -> Result<DenseWeight> {
    let a_bytes = a_rows * k * 2; // BF16
    let b_bytes = b_rows * k * 2;
    let total = a_bytes + b_bytes;
    let buf = gpu.alloc(total)?;
    tracing::debug!(
        target: "spark_model::weight_map::concat",
        a_rows, b_rows, k, a_bytes, b_bytes, total,
        a = a.weight.0, b = b.weight.0, buf = buf.0,
        "gpu concat rows [a;b]"
    );
    gpu.copy_d2d(a.weight, buf, a_bytes)?;
    gpu.copy_d2d(b.weight, buf.offset(a_bytes), b_bytes)?;
    Ok(DenseWeight { weight: buf })
}

/// CPU-side interleave A and B weight rows into BA format for dense_gemv_ba_gates.
///
/// Expected output format per GQA group: [b_vh0, b_vh1, a_vh0, a_vh1] (vpg betas, then vpg alphas).
/// A: [nv, K] BF16 (alpha rows, one per value head)
/// B: [nv, K] BF16 (beta rows, one per value head)
/// Returns: [2*nv, K] BF16 on GPU in interleaved format.
pub(crate) fn interleave_ba(
    a_weight: &DenseWeight,
    b_weight: &DenseWeight,
    nv: usize,
    nk: usize,
    k: usize,
    gpu: &dyn GpuBackend,
) -> Result<DenseWeight> {
    let vpg = nv / nk; // values per group (2)
    let row_bytes = k * 2; // BF16
    let ba_size = nv * 2;

    // Download A and B to CPU
    let mut a_cpu = vec![0u8; nv * row_bytes];
    let mut b_cpu = vec![0u8; nv * row_bytes];
    gpu.copy_d2h(a_weight.weight, &mut a_cpu)?;
    gpu.copy_d2h(b_weight.weight, &mut b_cpu)?;

    // Interleave: for each group g, write [b_vpg_heads, a_vpg_heads]
    let mut ba_cpu = vec![0u8; ba_size * row_bytes];
    for g in 0..nk {
        for v in 0..vpg {
            let vh = g * vpg + v;
            // Beta (B) rows first in each group
            let dst_row = g * (2 * vpg) + v;
            ba_cpu[dst_row * row_bytes..(dst_row + 1) * row_bytes]
                .copy_from_slice(&b_cpu[vh * row_bytes..(vh + 1) * row_bytes]);
            // Alpha (A) rows second in each group
            let dst_row = g * (2 * vpg) + vpg + v;
            ba_cpu[dst_row * row_bytes..(dst_row + 1) * row_bytes]
                .copy_from_slice(&a_cpu[vh * row_bytes..(vh + 1) * row_bytes]);
        }
    }

    // Upload to GPU
    let buf = gpu.alloc(ba_size * row_bytes)?;
    gpu.copy_h2d(&ba_cpu, buf)?;
    Ok(DenseWeight { weight: buf })
}
