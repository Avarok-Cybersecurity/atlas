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
    let packed_bytes = total / 2;
    let num_groups = total / 16;

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

    let mut packed = vec![0u8; packed_bytes];
    let mut scales = vec![0u8; num_groups]; // FP8 E4M3, 1 byte each
    gpu.copy_d2h(packed_ptr, &mut packed)?;
    gpu.copy_d2h(scale_ptr, &mut scales)?;

    // E2M1 lookup table: 4-bit nibble → float value
    // Bits: [sign(1)][exp(2)][mantissa(1)]
    let e2m1_table: [f32; 16] = [
        0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
    ];

    // Dequant to f32, then convert to BF16
    let mut bf16_out = vec![0u16; total];
    for group in 0..num_groups {
        let fp8_byte = scales[group];
        let block_scale = fp8_e4m3_to_f32(fp8_byte);
        // compressed-tensors: weight_global_scale is reciprocal → val = E2M1 * fp8_scale / global_scale
        // Standard/modelopt: weight_scale_2 is direct multiplier → val = E2M1 * fp8_scale * global_scale
        let combined_scale = if is_reciprocal {
            block_scale / global_scale
        } else {
            block_scale * global_scale
        };

        for elem in 0..16 {
            let flat_idx = group * 16 + elem;
            let byte_idx = flat_idx / 2;
            let nibble = if flat_idx % 2 == 0 {
                packed[byte_idx] & 0x0F
            } else {
                (packed[byte_idx] >> 4) & 0x0F
            };
            let val = e2m1_table[nibble as usize] * combined_scale;
            bf16_out[flat_idx] = f32_to_bf16(val);
        }
    }

    // Upload BF16 to GPU
    let buf = gpu.alloc(total * 2)?;
    let bf16_bytes: &[u8] =
        unsafe { std::slice::from_raw_parts(bf16_out.as_ptr() as *const u8, total * 2) };
    gpu.copy_h2d(bf16_bytes, buf)?;
    Ok(DenseWeight { weight: buf })
}

/// FP8 E4M3 → f32 lookup table (256 entries, one per byte value).
///
/// OCP FP8 E4M3FN format: sign(1) | exponent(4) | mantissa(3), bias=7.
/// Special values: 0x7F / 0xFF = NaN (no infinities).
/// Max finite: ±448.0 (exp=15, mant=6).
///
/// Generated at compile time — eliminates all branching from the hot dequant loop.
pub(super) static FP8_E4M3_LUT: [f32; 256] = {
    let mut table = [0.0f32; 256];
    let mut i: u32 = 0;
    while i < 256 {
        let bits = i as u8;
        let sign = (bits >> 7) & 1;
        let exp = (bits >> 3) & 0x0F;
        let mantissa = bits & 0x07;

        // NaN: exp=15, mantissa=7
        // We store 0.0 for NaN entries — NaN weights should not appear in practice,
        // and 0.0 is safer than propagating NaN through the dequant pipeline.
        let val = if exp == 0 && mantissa == 0 {
            0.0f32
        } else if exp == 0x0F && mantissa == 0x07 {
            0.0f32
        } else if exp == 0 {
            // Subnormal: 2^(-6) * (mantissa / 8)
            // 2^(-6) = 0.015625, /8 = 0.001953125 per mantissa unit
            (mantissa as f32) * (0.015625f32 / 8.0)
        } else {
            // Normal: 2^(exp-7) * (1 + mantissa/8)
            // Use bit manipulation to construct f32 directly:
            //   f32 exponent = fp8_exp - 7 + 127 = fp8_exp + 120
            //   f32 mantissa = fp8_mant << 20  (3 bits → 23 bits, left-aligned)
            let f32_exp = (exp as u32 + 120) << 23;
            let f32_mant = (mantissa as u32) << 20;
            f32::from_bits(f32_exp | f32_mant)
        };

        table[i as usize] = if sign == 1 { -val } else { val };
        i += 1;
    }
    table
};

/// Convert FP8 E4M3 byte to f32 via LUT (branchless, single array lookup).
#[inline(always)]
pub(super) fn fp8_e4m3_to_f32(bits: u8) -> f32 {
    FP8_E4M3_LUT[bits as usize]
}

/// Convert f32 to BF16 (truncation, no rounding).
pub(super) fn f32_to_bf16(val: f32) -> u16 {
    (val.to_bits() >> 16) as u16
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
            Ok(DenseFfnWeights {
                gate_proj: gate,
                up_proj: up,
                down_proj: down,
            })
        }
        _ => {
            let gate = quantized_auto(store, &format!("{prefix}.mlp.gate_proj"), gpu, variant)?;
            let up = quantized_auto(store, &format!("{prefix}.mlp.up_proj"), gpu, variant)?;
            let down = quantized_auto(store, &format!("{prefix}.mlp.down_proj"), gpu, variant)?;
            Ok(DenseFfnWeights {
                gate_proj: gate,
                up_proj: up,
                down_proj: down,
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
