// SPDX-License-Identifier: AGPL-3.0-only

//! Tests for FP8 E4M3 LUT, conversions, and dequantization paths.

use super::lut::bf16_bytes_to_f32;
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
    // Helper: invoke the private converter directly.
    fn convert(bits: u32) -> u16 {
        super::lut::f32_to_bf16(f32::from_bits(bits))
    }

    // (1) Below half-ULP: round DOWN. Truncation also rounds down.
    assert_eq!(convert(0x3F80_0800), 0x3F80, "1.0 + below-half-ULP -> 1.0");
    // (2) Exactly half-ULP, LSB=0: tie -> round to EVEN (down).
    //     Both truncation and RNE produce 0x3F80; doesn't distinguish.
    assert_eq!(
        convert(0x3F80_8000),
        0x3F80,
        "1.0 + exact-half-ULP, LSB=0 -> 1.0 (even)"
    );
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
    assert_eq!(
        convert(0x7F7F_FFFF),
        0x7F80,
        "max-finite f32 rounds to +inf bf16"
    );
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
        (
            0x3DCC_CCCD,
            0x3DCD,
            "0.1 -> RNE rounds UP to 0x3DCD (trunc=0x3DCC)",
        ),
        (0x3F4C_CCCD, 0x3F4D, "0.8 -> RNE rounds UP to 0x3F4D"),
        (0x40C9_0FDB, 0x40C9, "pi -> truncates (next bit < half)"),
        (0x402D_F854, 0x402E, "e -> RNE rounds UP (next bit > half)"),
        (0x4490_0000, 0x4490, "1152.0"),
        (0x3727_C5AC, 0x3728, "1e-5 -> RNE rounds UP"),
    ];
    for (f32_bits, want, desc) in cases {
        let got = super::lut::f32_to_bf16(f32::from_bits(*f32_bits));
        assert_eq!(
            got, *want,
            "f32={f32_bits:#010x} ({desc}): want bf16={want:#06x}, got {got:#06x}"
        );
    }
}
