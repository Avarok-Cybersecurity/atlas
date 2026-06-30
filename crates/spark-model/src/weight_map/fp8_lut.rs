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
/// Kept (allow dead_code) as the SSOT CPU reference for the FP8 E4M3 decode now
/// that `dequant_nvfp4_to_bf16` runs on the GPU (`dequant_nvfp4_bf16.cu`).
#[inline(always)]
#[allow(dead_code)]
pub(super) fn fp8_e4m3_to_f32(bits: u8) -> f32 {
    FP8_E4M3_LUT[bits as usize]
}

/// FP8 E8M0 → f32 lookup table (256 entries).
///
/// E8M0 format: unsigned 8-bit exponent, 0 mantissa, bias=127.
/// Value = 2^(exp - 127). exp=0 → 0, exp=255 → NaN (stored as 0.0).
static FP8_E8M0_LUT: [f32; 256] = {
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

/// Convert f32 to BF16 with IEEE-754 round-to-nearest-even.
///
/// SSOT-paired with `atlas_quant::fp8::f32_to_bf16`: both implement the
/// same RNE algorithm and must stay byte-identical to PyTorch's
/// `torch.float32 -> torch.bfloat16` cast. The CUDA-side mirror is
/// `__float2bfloat16_rn` in
/// `kernels/gb10/common/moe_fp8_grouped_gemm.cu`.
///
/// Phase 2b (Atlas FP8 dequant audit, 2026-05-24): replaced the
/// truncation `(bits >> 16) as u16` with proper ties-to-even rounding.
/// Phase 2a measurement showed Atlas-vs-canonical-dequant mean cos =
/// 0.969 driven primarily by this rounding bias accumulating across
/// 31745 dequanted tensors of Qwen3.6-35B-FP8.
///
/// Called by `dequant_fp8_blockscaled_to_bf16` (load-time shared-expert
/// dequant) AND `dequant_nvfp4_to_bf16` (NVFP4 -> BF16 path), so the
/// fix applies uniformly across all quantization formats that route
/// through this helper.
#[inline(always)]
pub(super) fn f32_to_bf16(val: f32) -> u16 {
    // Phase 2c day-2 bisect: ATLAS_DISABLE_RNE=1 reverts the Phase 2b
    // round-to-nearest-even patch back to truncation. Used to isolate
    // whether RNE accounts for the observed cosine regression vs the
    // May 23 rne baseline.
    if std::env::var("ATLAS_DISABLE_RNE").is_ok() {
        return (val.to_bits() >> 16) as u16;
    }
    let bits = val.to_bits();
    if val.is_nan() {
        let sign = ((bits >> 16) & 0x8000) as u16;
        return sign | 0x7FC0;
    }
    let lsb = (bits >> 16) & 1;
    let rounding_bias = 0x7FFFu32 + lsb;
    (bits.wrapping_add(rounding_bias) >> 16) as u16
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
        _ => {
            let gate = quantized_auto(store, &format!("{prefix}.mlp.gate_proj"), gpu, variant)?;
            let up = quantized_auto(store, &format!("{prefix}.mlp.up_proj"), gpu, variant)?;
            let down = quantized_auto(store, &format!("{prefix}.mlp.down_proj"), gpu, variant)?;
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
