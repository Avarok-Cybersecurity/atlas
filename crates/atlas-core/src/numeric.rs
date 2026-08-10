// SPDX-License-Identifier: AGPL-3.0-only

//! Host-side numeric conversions shared by every weight loader: the FP8
//! E4M3 decode table and the f32 -> BF16 cast.
//!
//! This is the single copy. It used to exist twice — once in
//! `spark-model/src/weight_map/fp8_lut.rs` (live) and once in
//! `atlas-quant/src/fp8.rs` (unreachable, zero dependents) — with the
//! byte-exactness tests attached to the copy that never ran. Both crates
//! already depended on `atlas-core`, so the fix was to move the arithmetic
//! down here and bring the tests with it.
//!
//! Pure arithmetic: no CUDA, no allocation, compiles under every feature
//! combination. The CUDA-side mirrors are `E4M3_LUT_GMOE` and
//! `__float2bfloat16_rn` in `kernels/gb10/common/moe_fp8_grouped_gemm.cu`;
//! they must agree with this file element for element.

/// FP8 E4M3 -> f32 lookup table (256 entries, one per byte value).
///
/// OCP FP8 E4M3FN: sign(1) | exponent(4) | mantissa(3), bias = 7. There
/// are no infinities; `0x7F` / `0xFF` are NaN and max finite is +/-448.0
/// (exp = 15, mant = 6).
///
/// NaN entries decode to `0.0`. A NaN weight should not exist in a
/// checkpoint, and zero stops one bad byte from poisoning an entire
/// dequanted tensor — which is what propagating NaN through the loader
/// would do.
///
/// Built at compile time so the hot dequant loop is a single indexed load
/// with no branches.
#[allow(clippy::if_same_then_else)]
pub static FP8_E4M3_LUT: [f32; 256] = {
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
            // Subnormal: 2^(-6) * (mantissa / 8).
            (mantissa as f32) * (0.015625f32 / 8.0)
        } else {
            // Normal: 2^(exp-7) * (1 + mantissa/8), assembled directly in
            // f32 bits — f32 exponent = fp8_exp - 7 + 127 = fp8_exp + 120,
            // f32 mantissa = fp8_mant << 20 (3 bits left-aligned into 23).
            let f32_exp = (exp as u32 + 120) << 23;
            let f32_mant = (mantissa as u32) << 20;
            f32::from_bits(f32_exp | f32_mant)
        };

        table[i as usize] = if sign == 1 { -val } else { val };
        i += 1;
    }
    table
};

/// Decode one FP8 E4M3 byte to f32 (branchless, single array lookup).
#[inline(always)]
pub fn fp8_e4m3_to_f32(bits: u8) -> f32 {
    FP8_E4M3_LUT[bits as usize]
}

/// Convert f32 to BF16 with IEEE-754 round-to-nearest-even.
///
/// Must stay byte-identical to PyTorch's `torch.float32 -> torch.bfloat16`
/// cast: reference activations and the dequanted-weight snapshots Atlas is
/// scored against are produced that way, so any drift here shows up as an
/// accuracy regression with no other symptom.
///
/// Phase 2b (FP8 dequant audit, 2026-05-24) replaced truncation
/// (`bits >> 16`) with ties-to-even. Truncation is biased toward zero and
/// the bias accumulated across the 31745 dequanted tensors of
/// Qwen3.6-35B-FP8 to a mean per-layer cosine of 0.969.
///
/// NaN maps to the canonical quiet-NaN pattern with the sign preserved,
/// which is also what PyTorch does.
///
/// `ATLAS_DISABLE_RNE` is a bisect escape hatch that reverts to
/// truncation. It is a PRESENCE check, not a value check — `=0` disables
/// RNE just as `=1` does.
#[inline(always)]
pub fn f32_to_bf16(val: f32) -> u16 {
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

/// Widen little-endian BF16 bytes to f32. Exact — BF16 is the top 16 bits
/// of an f32, so this is a shift, never a rounding.
#[inline(always)]
pub fn bf16_bytes_to_f32(bytes: [u8; 2]) -> f32 {
    let bits = u16::from_le_bytes(bytes);
    f32::from_bits((bits as u32) << 16)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fp8_lut_reference_values() {
        assert_eq!(fp8_e4m3_to_f32(0x00), 0.0); // +0
        assert_eq!(fp8_e4m3_to_f32(0x80), -0.0); // -0
        assert_eq!(fp8_e4m3_to_f32(0x38), 1.0); // exp=7, mant=0
        assert_eq!(fp8_e4m3_to_f32(0xB8), -1.0);
        assert_eq!(fp8_e4m3_to_f32(0x3C), 1.5); // exp=7, mant=4
        assert_eq!(fp8_e4m3_to_f32(0x7E), 448.0); // max finite
        assert_eq!(fp8_e4m3_to_f32(0xFE), -448.0); // min finite
        assert_eq!(fp8_e4m3_to_f32(0x7F), 0.0); // NaN -> 0
        assert_eq!(fp8_e4m3_to_f32(0xFF), 0.0); // -NaN -> 0

        // Subnormals: 2^(-6) * mant/8.
        let eps = 1e-10;
        assert!((fp8_e4m3_to_f32(0x01) - 0.001953125).abs() < eps);
        assert!((fp8_e4m3_to_f32(0x07) - 0.013671875).abs() < eps);
    }

    #[test]
    #[allow(clippy::if_same_then_else)]
    fn fp8_lut_matches_the_spec_for_all_256_bytes() {
        // Re-derived from the OCP definition with float math, independently
        // of the bit-assembly the table is built with.
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

    /// The assertions that separate round-to-nearest-even from
    /// truncation-toward-zero. Truncation FAILS every "round up" case here.
    #[test]
    fn f32_to_bf16_is_rne_byte_exact() {
        fn convert(bits: u32) -> u16 {
            f32_to_bf16(f32::from_bits(bits))
        }

        // Below half-ULP: round DOWN. Truncation agrees.
        assert_eq!(convert(0x3F80_0800), 0x3F80, "1.0 + below-half-ULP -> 1.0");
        // Exactly half-ULP, LSB=0: tie -> round to EVEN (down). Does not
        // distinguish RNE from truncation; kept for the tie coverage.
        assert_eq!(
            convert(0x3F80_8000),
            0x3F80,
            "1.0 + exact-half-ULP, LSB=0 -> 1.0 (even)"
        );
        // Above half-ULP: round UP. Truncation would give 0x3F80.
        assert_eq!(
            convert(0x3F80_8001),
            0x3F81,
            "1.0 + above-half-ULP -> next bf16 (truncation would give 0x3F80)"
        );
        // Exactly half-ULP, LSB=1: tie -> round to EVEN (up). Truncation
        // would give 0x3F81.
        assert_eq!(
            convert(0x3F81_8000),
            0x3F82,
            "1.0078125 + exact-half-ULP, LSB=1 -> 1.015625"
        );
        // Negative parity: magnitude grows the same way.
        assert_eq!(convert(0xBF80_8001), 0xBF81, "negative round up");
        // Zero: exact, no rounding, sign preserved.
        assert_eq!(convert(0x0000_0000), 0x0000, "+0.0");
        assert_eq!(convert(0x8000_0000), 0x8000, "-0.0");
        // Smallest f32 subnormal (2^-149) -> nearest bf16 is 0 (LSB=0 tie).
        assert_eq!(convert(0x0000_0001), 0x0000, "tiny subnormal -> 0");
        // Infinities pass through.
        assert_eq!(convert(0x7F80_0000), 0x7F80, "+inf");
        assert_eq!(convert(0xFF80_0000), 0xFF80, "-inf");
        // Max-finite f32 rounds UP to +inf in bf16 — the closest
        // representable value. PyTorch does the same.
        assert_eq!(
            convert(0x7F7F_FFFF),
            0x7F80,
            "max-finite f32 rounds to +inf bf16"
        );
        // NaN -> canonical quiet NaN, sign preserved; signalling NaN is
        // quieted rather than passed through.
        assert_eq!(convert(0x7FC0_0000), 0x7FC0, "qnan +");
        assert_eq!(convert(0xFFC0_0000), 0xFFC0, "qnan -");
        assert_eq!(convert(0x7F80_0001), 0x7FC0, "snan + -> qnan +");
    }

    /// Byte-exact match against values captured from PyTorch 2.9 via
    /// `torch.tensor([x], dtype=torch.float32).bfloat16()`. If this fails
    /// after a math change, the converter has drifted from PyTorch's RNE
    /// and every dequanted weight in the engine is off by a bit.
    #[test]
    fn f32_to_bf16_matches_pytorch() {
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
            let got = f32_to_bf16(f32::from_bits(*f32_bits));
            assert_eq!(
                got, *want,
                "f32={f32_bits:#010x} ({desc}): want bf16={want:#06x}, got {got:#06x}"
            );
        }
    }

    #[test]
    fn bf16_widening_round_trips_every_bf16_value() {
        // BF16 is the top half of an f32, so widening then narrowing must
        // be the identity for all 65536 patterns except the NaN space,
        // which `f32_to_bf16` canonicalises on purpose.
        for bits in 0u32..=0xFFFF {
            let bf16 = bits as u16;
            let widened = bf16_bytes_to_f32(bf16.to_le_bytes());
            if widened.is_nan() {
                continue;
            }
            assert_eq!(
                f32_to_bf16(widened),
                bf16,
                "round trip failed for bf16 {bf16:#06x}"
            );
        }
    }
}
