// SPDX-License-Identifier: AGPL-3.0-only

//! FP8 E4M3 lookup table and FP32/BF16 conversion primitives.

// ── FP8 E4M3 LUT ──

/// FP8 E4M3 -> f32 lookup table (256 entries, one per byte value).
///
/// OCP FP8 E4M3FN format: sign(1) | exponent(4) | mantissa(3), bias=7.
/// Special values: 0x7F / 0xFF = NaN (mapped to 0.0 for safety).
/// Max finite: +/-448.0 (exp=15, mant=6).
#[allow(clippy::if_same_then_else)]
static FP8_E4M3_LUT: [f32; 256] = {
    let mut table = [0.0f32; 256];
    let mut i: u32 = 0;
    while i < 256 {
        let bits = i as u8;
        let sign = (bits >> 7) & 1;
        let exp = (bits >> 3) & 0x0F;
        let mantissa = bits & 0x07;

        let val = if exp == 0 && mantissa == 0 {
            0.0f32
        } else if exp == 0x0F && mantissa == 0x07 {
            0.0f32 // NaN -> 0.0
        } else if exp == 0 {
            // Subnormal: 2^(-6) * (mantissa / 8)
            (mantissa as f32) * (0.015625f32 / 8.0)
        } else {
            // Normal: 2^(exp-7) * (1 + mantissa/8)
            let f32_exp = (exp as u32 + 120) << 23;
            let f32_mant = (mantissa as u32) << 20;
            f32::from_bits(f32_exp | f32_mant)
        };

        table[i as usize] = if sign == 1 { -val } else { val };
        i += 1;
    }
    table
};

/// Convert a single FP8 E4M3 byte to f32 via LUT (branchless, single array lookup).
#[inline(always)]
pub fn fp8_e4m3_to_f32(bits: u8) -> f32 {
    FP8_E4M3_LUT[bits as usize]
}

/// Convert f32 to BF16 with IEEE-754 round-to-nearest-even.
///
/// SSOT for the FP32 -> BF16 rounding used by all FP8 dequant paths
/// (`dequant_fp8_pertensor_to_bf16`, `dequant_fp8_block_to_bf16`, and
/// transitively `quant_helpers::dequant_fp8_blockscaled_to_bf16`). The
/// CUDA-side mirror is `__float2bfloat16_rn` in
/// `kernels/gb10/common/moe_fp8_grouped_gemm.cu`.
///
/// Phase 2b (Atlas FP8 dequant audit, 2026-05-24): replaced truncation
/// `(bits >> 16) as u16` with proper ties-to-even rounding. The
/// truncation bias accumulated to ~3% per-layer cosine loss over the
/// 31k+ block-scaled tensors in Qwen3.6-35B-FP8 (Phase 2a measurement
/// C mean = 0.969); RNE matches PyTorch's `float32 -> bfloat16` cast
/// byte-exact.
///
/// NaN is mapped to the canonical quiet-NaN bit pattern preserving the
/// sign, matching PyTorch's `torch.float32 -> torch.bfloat16` behavior
/// (Phase 2a's dequanted reference snapshot was produced this way).
#[inline(always)]
pub(super) fn f32_to_bf16(val: f32) -> u16 {
    // Phase 2c day-2 bisect: ATLAS_DISABLE_RNE=1 reverts to truncation.
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

/// Convert BF16 bytes (little-endian) to f32.
#[inline(always)]
pub(super) fn bf16_bytes_to_f32(bytes: [u8; 2]) -> f32 {
    let bits = u16::from_le_bytes(bytes);
    f32::from_bits((bits as u32) << 16)
}
