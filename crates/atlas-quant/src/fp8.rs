// SPDX-License-Identifier: AGPL-3.0-only

//! FP8 E4M3 quantization and dequantization utilities.
//!
//! Supports two FP8 checkpoint formats:
//!   1. **Per-tensor scaled**: `weight` (FP8) + `weight_scale` (f32 scalar).
//!   2. **Block-scaled**: `weight` (FP8) + `weight_scale_inv` (BF16 per-block).
//!
//! FP8 E4M3FN: sign(1) | exponent(4) | mantissa(3), bias=7, range [-448, 448].

use atlas_core::error::Result;
use atlas_core::tensor::TensorRef;

use crate::traits::Quantize;

// ── Format descriptors ──

/// Scale factor precision for block-scaled FP8.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScaleDtype {
    Fp32,
    Bf16,
}

/// FP8 E4M3 block-scaled format descriptor.
///
/// Describes the layout of compressed-tensors FP8 checkpoints:
/// - Weight tensor: FP8 E4M3 bytes, shape [N, K].
/// - Scale tensor: per-block scales, shape [N/block_size, K/block_size].
#[derive(Debug, Clone)]
pub struct Fp8Format {
    /// Block size for block-scaled FP8 (e.g., 128 elements per scale in each dim).
    pub block_size: usize,
    /// Precision of the per-block scale factors.
    pub scale_dtype: ScaleDtype,
}

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
fn f32_to_bf16(val: f32) -> u16 {
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
fn bf16_bytes_to_f32(bytes: [u8; 2]) -> f32 {
    let bits = u16::from_le_bytes(bytes);
    f32::from_bits((bits as u32) << 16)
}

// ── Detection ──

/// Check if a safetensors dtype string represents an FP8 tensor.
///
/// Recognizes: `"F8_E4M3"`, `"float8_e4m3fn"`, `"float8_e4m3fnuz"`.
pub fn is_fp8_tensor(dtype: &str) -> bool {
    matches!(dtype, "F8_E4M3" | "float8_e4m3fn" | "float8_e4m3fnuz")
}

// ── Per-tensor dequant ──

/// Dequantize FP8 E4M3 per-tensor-scaled data to BF16 bytes.
///
/// Each FP8 byte is converted to f32, multiplied by the per-tensor scale,
/// then truncated to BF16. Returns a Vec of BF16 bytes (2 bytes per element).
///
/// This matches the on-disk layout used by `compressed-tensors` FP8 with a
/// single `weight_scale` scalar per weight tensor.
pub fn dequant_fp8_pertensor_to_bf16(fp8_data: &[u8], scale: f32) -> Vec<u8> {
    fp8_data
        .iter()
        .flat_map(|&byte| {
            let val = fp8_e4m3_to_f32(byte) * scale;
            f32_to_bf16(val).to_le_bytes()
        })
        .collect()
}

// ── Block-scaled dequant ──

/// Dequantize FP8 E4M3 block-scaled tensor to BF16 bytes.
///
/// Layout:
///   - `fp8_data`: FP8 E4M3 weight bytes, row-major [N, K] (N*K bytes total).
///   - `scales`: Per-block scale_inv values. Format depends on `scale_dtype`.
///   - `n`, `k`: Logical weight dimensions.
///   - `block_size`: Block size along each dimension (e.g. 128 for [128, 128] blocks).
///   - `scale_dtype`: Precision of scale values (BF16 or FP32).
///
/// Dequantization formula: `bf16[i,j] = fp8[i,j] * scale_inv[i/block, j/block]`
///
/// Returns a Vec of BF16 bytes (2 bytes per element, N*K*2 total).
pub fn dequant_fp8_block_to_bf16(
    fp8_data: &[u8],
    scales: &[u8],
    n: usize,
    k: usize,
    block_size: usize,
    scale_dtype: ScaleDtype,
) -> Vec<u8> {
    assert_eq!(
        fp8_data.len(),
        n * k,
        "FP8 data length mismatch: expected {}, got {}",
        n * k,
        fp8_data.len()
    );

    let sn = n.div_ceil(block_size);
    let sk = k.div_ceil(block_size);

    let scale_elem_bytes = match scale_dtype {
        ScaleDtype::Bf16 => 2,
        ScaleDtype::Fp32 => 4,
    };
    let expected_scale_bytes = sn * sk * scale_elem_bytes;
    assert_eq!(
        scales.len(),
        expected_scale_bytes,
        "Scale buffer length mismatch: expected {expected_scale_bytes}, got {}",
        scales.len(),
    );

    let read_scale = |scale_idx: usize| -> f32 {
        let offset = scale_idx * scale_elem_bytes;
        match scale_dtype {
            ScaleDtype::Bf16 => bf16_bytes_to_f32([scales[offset], scales[offset + 1]]),
            ScaleDtype::Fp32 => f32::from_le_bytes([
                scales[offset],
                scales[offset + 1],
                scales[offset + 2],
                scales[offset + 3],
            ]),
        }
    };

    let total = n * k;
    let mut bf16_out = vec![0u8; total * 2];

    for row in 0..n {
        let scale_row = row / block_size;
        for col in 0..k {
            let scale_col = col / block_size;
            let scale_idx = scale_row * sk + scale_col;
            let scale_val = read_scale(scale_idx);

            let fp8_byte = fp8_data[row * k + col];
            let val = fp8_e4m3_to_f32(fp8_byte) * scale_val;
            let bf16_val = f32_to_bf16(val);

            let out_idx = (row * k + col) * 2;
            let [lo, hi] = bf16_val.to_le_bytes();
            bf16_out[out_idx] = lo;
            bf16_out[out_idx + 1] = hi;
        }
    }

    bf16_out
}

// ── GPU quantizer (stub for future 4B GEMM dispatch) ──

/// FP8 E4M3 quantization: FP32/BF16 -> FP8 with per-tensor or per-token scale.
pub struct Fp8Quantizer;

impl Quantize for Fp8Quantizer {
    fn quantize(
        &self,
        _input: &TensorRef,
        _output: &TensorRef,
        _scale: &TensorRef,
        _stream_ptr: u64,
    ) -> Result<()> {
        // TODO: Launch fp8_quant.cu kernel (Workstream 4B)
        todo!("FP8 quantization kernel launch")
    }
}

// ── Tests ──

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fp8_lut_reference_values() {
        assert_eq!(fp8_e4m3_to_f32(0x00), 0.0); // +0
        assert_eq!(fp8_e4m3_to_f32(0x80), -0.0); // -0
        assert_eq!(fp8_e4m3_to_f32(0x38), 1.0); // exp=7, mant=0
        assert_eq!(fp8_e4m3_to_f32(0xB8), -1.0); // -1.0
        assert_eq!(fp8_e4m3_to_f32(0x3C), 1.5); // exp=7, mant=4
        assert_eq!(fp8_e4m3_to_f32(0x7E), 448.0); // max finite
        assert_eq!(fp8_e4m3_to_f32(0xFE), -448.0); // min finite
        assert_eq!(fp8_e4m3_to_f32(0x7F), 0.0); // NaN -> 0
        assert_eq!(fp8_e4m3_to_f32(0xFF), 0.0); // -NaN -> 0

        // Subnormals: 2^(-6) * mant/8
        let eps = 1e-10;
        assert!((fp8_e4m3_to_f32(0x01) - 0.001953125).abs() < eps);
        assert!((fp8_e4m3_to_f32(0x07) - 0.013671875).abs() < eps);
    }

    #[test]
    #[allow(clippy::if_same_then_else)]
    fn test_fp8_lut_exhaustive() {
        for i in 0u16..256 {
            let bits = i as u8;
            let sign = (bits >> 7) & 1;
            let exp = (bits >> 3) & 0x0F;
            let mant = bits & 0x07;

            let expected = if exp == 0x0F && mant == 0x07 {
                0.0f32
            } else if exp == 0 && mant == 0 {
                0.0f32
            } else if exp == 0 {
                let v = (mant as f32 / 8.0) * 2.0f32.powi(-6);
                if sign == 1 { -v } else { v }
            } else {
                let v = (1.0 + mant as f32 / 8.0) * 2.0f32.powi(exp as i32 - 7);
                if sign == 1 { -v } else { v }
            };
            let actual = fp8_e4m3_to_f32(bits);
            assert!(
                (actual - expected).abs() < 1e-10 || (actual == 0.0 && expected == 0.0),
                "LUT mismatch at {i:#04x}: expected {expected}, got {actual}"
            );
        }
    }

    #[test]
    fn test_is_fp8_tensor() {
        assert!(is_fp8_tensor("F8_E4M3"));
        assert!(is_fp8_tensor("float8_e4m3fn"));
        assert!(is_fp8_tensor("float8_e4m3fnuz"));
        assert!(!is_fp8_tensor("BF16"));
        assert!(!is_fp8_tensor("F32"));
        assert!(!is_fp8_tensor(""));
    }

    #[test]
    fn test_dequant_pertensor_identity() {
        // FP8 byte 0x38 = 1.0, scale=1.0 -> BF16 1.0
        let fp8 = vec![0x38u8];
        let result = dequant_fp8_pertensor_to_bf16(&fp8, 1.0);
        assert_eq!(result.len(), 2);
        let bf16_val = u16::from_le_bytes([result[0], result[1]]);
        // BF16 1.0 = 0x3F80
        assert_eq!(bf16_val, 0x3F80);
    }

    #[test]
    fn test_dequant_pertensor_with_scale() {
        // FP8 byte 0x38 = 1.0, scale=2.0 -> BF16 2.0
        let fp8 = vec![0x38u8];
        let result = dequant_fp8_pertensor_to_bf16(&fp8, 2.0);
        let bf16_val = u16::from_le_bytes([result[0], result[1]]);
        // BF16 2.0 = 0x4000
        assert_eq!(bf16_val, 0x4000);
    }

    #[test]
    fn test_dequant_pertensor_negative() {
        // FP8 byte 0xB8 = -1.0, scale=3.0 -> BF16 -3.0
        let fp8 = vec![0xB8u8];
        let result = dequant_fp8_pertensor_to_bf16(&fp8, 3.0);
        let bf16_val = u16::from_le_bytes([result[0], result[1]]);
        // BF16 -3.0 = 0xC040
        assert_eq!(bf16_val, 0xC040);
    }

    #[test]
    fn test_dequant_pertensor_zero() {
        let fp8 = vec![0x00u8];
        let result = dequant_fp8_pertensor_to_bf16(&fp8, 42.0);
        let bf16_val = u16::from_le_bytes([result[0], result[1]]);
        assert_eq!(bf16_val, 0x0000); // +0 * anything = +0
    }

    #[test]
    fn test_dequant_pertensor_multiple() {
        // 4 elements: [1.0, -1.0, 0.0, 448.0] with scale=0.5
        let fp8 = vec![0x38, 0xB8, 0x00, 0x7E];
        let result = dequant_fp8_pertensor_to_bf16(&fp8, 0.5);
        assert_eq!(result.len(), 8); // 4 * 2 bytes

        let vals: Vec<f32> = result
            .chunks_exact(2)
            .map(|c| bf16_bytes_to_f32([c[0], c[1]]))
            .collect();

        assert!((vals[0] - 0.5).abs() < 0.01);
        assert!((vals[1] - (-0.5)).abs() < 0.01);
        assert_eq!(vals[2], 0.0);
        assert!((vals[3] - 224.0).abs() < 1.0);
    }

    #[test]
    fn test_dequant_block_bf16_scales() {
        // 2x2 matrix, block_size=1 (each element has its own scale)
        // FP8: [[1.0, 2.0], [-1.0, 0.5]]
        // 1.0 = 0x38, 2.0 = 0x40, -1.0 = 0xB8, 0.5 = 0x30
        let fp8_data = vec![0x38, 0x40, 0xB8, 0x30];

        // Scales (BF16): [2.0, 0.5, 1.0, 3.0] per block
        // BF16 2.0 = 0x4000, 0.5 = 0x3F00, 1.0 = 0x3F80, 3.0 = 0x4040
        let scale_bf16: Vec<u8> = [0x4000u16, 0x3F00, 0x3F80, 0x4040]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();

        let result = dequant_fp8_block_to_bf16(
            &fp8_data,
            &scale_bf16,
            2,
            2, // n=2, k=2
            1, // block_size=1
            ScaleDtype::Bf16,
        );
        assert_eq!(result.len(), 8);

        let vals: Vec<f32> = result
            .chunks_exact(2)
            .map(|c| bf16_bytes_to_f32([c[0], c[1]]))
            .collect();

        // [0] = 1.0 * 2.0 = 2.0
        assert!((vals[0] - 2.0).abs() < 0.01, "val[0] = {}", vals[0]);
        // [1] = 2.0 * 0.5 = 1.0
        assert!((vals[1] - 1.0).abs() < 0.01, "val[1] = {}", vals[1]);
        // [2] = -1.0 * 1.0 = -1.0
        assert!((vals[2] - (-1.0)).abs() < 0.01, "val[2] = {}", vals[2]);
        // [3] = 0.5 * 3.0 = 1.5
        assert!((vals[3] - 1.5).abs() < 0.01, "val[3] = {}", vals[3]);
    }

    #[test]
    fn test_dequant_block_fp32_scales() {
        // 4x4 matrix, block_size=2 -> scale shape [2, 2]
        // All FP8 bytes = 0x38 (1.0)
        let fp8_data = vec![0x38u8; 16];

        // Scales (FP32): [1.0, 2.0, 3.0, 4.0]
        let scale_f32: Vec<u8> = [1.0f32, 2.0, 3.0, 4.0]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();

        let result = dequant_fp8_block_to_bf16(&fp8_data, &scale_f32, 4, 4, 2, ScaleDtype::Fp32);
        assert_eq!(result.len(), 32);

        let vals: Vec<f32> = result
            .chunks_exact(2)
            .map(|c| bf16_bytes_to_f32([c[0], c[1]]))
            .collect();

        // Row 0, Col 0-1: block [0,0] scale=1.0 -> 1.0 * 1.0 = 1.0
        assert!((vals[0] - 1.0).abs() < 0.01);
        assert!((vals[1] - 1.0).abs() < 0.01);
        // Row 0, Col 2-3: block [0,1] scale=2.0 -> 1.0 * 2.0 = 2.0
        assert!((vals[2] - 2.0).abs() < 0.01);
        assert!((vals[3] - 2.0).abs() < 0.01);
        // Row 2, Col 0-1: block [1,0] scale=3.0 -> 1.0 * 3.0 = 3.0
        assert!((vals[8] - 3.0).abs() < 0.01);
        // Row 2, Col 2-3: block [1,1] scale=4.0 -> 1.0 * 4.0 = 4.0
        assert!((vals[10] - 4.0).abs() < 0.01);
    }

    #[test]
    fn test_dequant_block_128_stride() {
        // Realistic block_size=128: 128x128 matrix, single block, scale=0.5
        let n = 128;
        let k = 128;
        let fp8_data = vec![0x38u8; n * k]; // All 1.0

        // BF16 scale = 0.5 = 0x3F00
        let scale_bf16: Vec<u8> = 0x3F00u16.to_le_bytes().to_vec();

        let result = dequant_fp8_block_to_bf16(&fp8_data, &scale_bf16, n, k, 128, ScaleDtype::Bf16);
        assert_eq!(result.len(), n * k * 2);

        // Every element should be 1.0 * 0.5 = 0.5
        for i in 0..n * k {
            let val = bf16_bytes_to_f32([result[i * 2], result[i * 2 + 1]]);
            assert!(
                (val - 0.5).abs() < 0.01,
                "element {i}: expected 0.5, got {val}"
            );
        }
    }

    /// Phase 2b RNE byte-exact regression: cases that distinguish
    /// round-to-nearest-even from truncation-toward-zero.
    /// Truncation would FAIL all "round up" assertions below.
    #[test]
    fn test_f32_to_bf16_rne_byte_exact() {
        // Helper: invoke the private converter via the public dequant path
        // with scale=1.0; same arithmetic, byte-identical output.
        fn convert(bits: u32) -> u16 {
            super::f32_to_bf16(f32::from_bits(bits))
        }

        // (1) Below half-ULP: round DOWN. Truncation also rounds down.
        assert_eq!(convert(0x3F80_0800), 0x3F80, "1.0 + below-half-ULP -> 1.0");
        // (2) Exactly half-ULP, LSB=0: tie -> round to EVEN (down).
        //     Both truncation and RNE produce 0x3F80; doesn't distinguish.
        assert_eq!(convert(0x3F80_8000), 0x3F80, "1.0 + exact-half-ULP, LSB=0 -> 1.0 (even)");
        // (3) Above half-ULP: round UP. Truncation would FAIL (gives 0x3F80).
        assert_eq!(
            convert(0x3F80_8001),
            0x3F81,
            "1.0 + above-half-ULP -> next bf16 (truncation would give 0x3F80)"
        );
        // (4) Exactly half-ULP, LSB=1: tie -> round to EVEN (up).
        //     Truncation would FAIL (gives 0x3F81, RNE gives 0x3F82).
        assert_eq!(
            convert(0x3F81_8000),
            0x3F82,
            "1.0078125 + exact-half-ULP, LSB=1 -> 1.015625 (truncation would give 0x3F81)"
        );
        // (5) Negative parity: -1.0 + (-above-half-ULP) -> bigger magnitude.
        assert_eq!(convert(0xBF80_8001), 0xBF81, "negative round up");
        // (6) Zero: exact, no rounding.
        assert_eq!(convert(0x0000_0000), 0x0000, "+0.0");
        assert_eq!(convert(0x8000_0000), 0x8000, "-0.0");
        // (7) Smallest subnormal f32 (2^-149) -> nearest bf16 = 0 (LSB=0 tie).
        assert_eq!(convert(0x0000_0001), 0x0000, "tiny subnormal -> 0");
        // (8) f32 +inf preserves +inf.
        assert_eq!(convert(0x7F80_0000), 0x7F80, "+inf");
        assert_eq!(convert(0xFF80_0000), 0xFF80, "-inf");
        // (9) Max-finite f32 rounds UP to +inf bf16 (closest representable).
        //     PyTorch does the same.
        assert_eq!(convert(0x7F7F_FFFF), 0x7F80, "max-finite f32 rounds to +inf bf16");
        // (10) NaN -> canonical quiet NaN, sign preserved.
        assert_eq!(convert(0x7FC0_0000), 0x7FC0, "qnan +");
        assert_eq!(convert(0xFFC0_0000), 0xFFC0, "qnan -");
        assert_eq!(convert(0x7F80_0001), 0x7FC0, "snan + -> qnan +");
    }

    /// Phase 2b: byte-exact match against the canonical reference values
    /// PyTorch's `torch.float32 -> torch.bfloat16` cast produces. The
    /// (f32_bits, bf16_bits) pairs below were captured directly from
    /// PyTorch 2.9 via `torch.tensor([x], dtype=torch.float32).bfloat16()`.
    /// If this test fails after a math change, the converter has drifted
    /// from PyTorch's RNE and the FP8 dequant ceiling work is broken.
    #[test]
    fn test_f32_to_bf16_pytorch_parity() {
        let cases: &[(u32, u16, &str)] = &[
            (0x3F80_0000, 0x3F80, "1.0"),
            (0x4000_0000, 0x4000, "2.0"),
            (0xC000_0000, 0xC000, "-2.0"),
            (0x3FC0_0000, 0x3FC0, "1.5"),
            (0x3DCC_CCCD, 0x3DCD, "0.1 -> RNE rounds UP to 0x3DCD (trunc=0x3DCC)"),
            (0x3F4C_CCCD, 0x3F4D, "0.8 -> RNE rounds UP to 0x3F4D"),
            (0x40C9_0FDB, 0x40C9, "pi -> truncates (next bit < half)"),
            (0x402D_F854, 0x402E, "e -> RNE rounds UP (next bit > half)"),
            (0x4490_0000, 0x4490, "1152.0"),
            (0x3727_C5AC, 0x3728, "1e-5 -> RNE rounds UP"),
        ];
        for (f32_bits, want, desc) in cases {
            let got = super::f32_to_bf16(f32::from_bits(*f32_bits));
            assert_eq!(
                got, *want,
                "f32={f32_bits:#010x} ({desc}): want bf16={want:#06x}, got {got:#06x}"
            );
        }
    }
}
