// SPDX-License-Identifier: AGPL-3.0-only

//! The dequantization codebooks, and the group size they are indexed in.
//!
//! Split from the dequant routines that read them purely for the file-size
//! cap: the tables are data, the routines are the loops over that data, and
//! the seam falls between them cleanly. Every table matches its CUDA-side
//! counterpart byte-for-byte — see the parent module's table for which
//! kernel each mirrors.

/// Group size for per-group FP8 scales. Matches `NVFP4_GROUP_SIZE` in the
/// per-quant attention kernels.
pub const NVFP4_GROUP_SIZE: usize = 16;

/// E2M1 4-bit codebook (NVFP4). Matches
/// `kernels/gb10/common/paged_decode_attn_nvfp4.cu:118`.
pub const NVFP4_E2M1_LUT: [f32; 16] = [
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
];

/// Turbo4 16-level Lloyd-Max codebook. Matches
/// `kernels/gb10/common/paged_decode_attn_turbo4.cu:121`.
pub const TURBO4_LUT: [f32; 16] = [
    -2.7326, -2.0690, -1.6180, -1.2562, -0.9423, -0.6568, -0.3880, -0.1284, 0.1284, 0.3880, 0.6568,
    0.9423, 1.2562, 1.6180, 2.0690, 2.7326,
];

/// Turbo3 8-level Lloyd-Max codebook. Matches
/// `kernels/gb10/common/paged_decode_attn_turbo3.cu:137`.
pub const TURBO3_LUT: [f32; 8] = [
    -2.1520, -1.3440, -0.7560, -0.2451, 0.2451, 0.7560, 1.3440, 2.1520,
];

/// E4M3 → f32 LUT (256 entries).
///
/// A `const`, not a lazily-filled static: the table is pure arithmetic over a
/// fixed domain, so it is computed at compile time and there is no runtime
/// state to initialise, synchronise or invalidate.
const E4M3_LUT: [f32; 256] = {
    let mut lut = [0.0f32; 256];
    let mut byte = 0u32;
    while byte < 256 {
        let sign_bit = (byte >> 7) & 1;
        let exp = ((byte >> 3) & 0xF) as i32;
        let mant = byte & 0x7;
        let s: f32 = if sign_bit == 0 { 1.0 } else { -1.0 };
        lut[byte as usize] = if exp == 0 {
            if mant == 0 {
                s * 0.0
            } else {
                s * (mant as f32) * exp2(-9)
            }
        } else if exp == 0xF && mant == 0x7 {
            f32::NAN
        } else {
            s * exp2(exp - 7) * (1.0 + (mant as f32) / 8.0)
        };
        byte += 1;
    }
    lut
};

/// `2^e`, for the small integer exponents this table needs (`-9..=8`).
///
/// `f32::powi` is not a const fn. Repeated multiply/divide is EXACT here rather
/// than merely close: powers of two are representable exactly in binary
/// floating point, so every step is lossless and the result is bit-identical to
/// `powi`. A test pins that.
const fn exp2(e: i32) -> f32 {
    let mut v = 1.0f32;
    let mut i = 0i32;
    if e >= 0 {
        while i < e {
            v *= 2.0;
            i += 1;
        }
    } else {
        while i < -e {
            v /= 2.0;
            i += 1;
        }
    }
    v
}

/// Borrow the compile-time table.
pub fn e4m3_lut() -> &'static [f32; 256] {
    &E4M3_LUT
}

#[cfg(test)]
mod lut_tests {
    use super::*;

    /// The table moved from a lazily-filled `OnceLock` to a `const`, which
    /// meant replacing `f32::powi` (not const) with `exp2`. This proves the
    /// substitution changed no value: every entry must be bit-identical to the
    /// `powi` formula it replaced.
    #[test]
    fn the_const_table_is_bit_identical_to_the_powi_formula() {
        for byte in 0..256u32 {
            let sign_bit = (byte >> 7) & 1;
            let exp = ((byte >> 3) & 0xF) as i32;
            let mant = byte & 0x7;
            let s: f32 = if sign_bit == 0 { 1.0 } else { -1.0 };
            let expected: f32 = if exp == 0 {
                if mant == 0 {
                    s * 0.0
                } else {
                    s * (mant as f32) * 2.0f32.powi(-9)
                }
            } else if exp == 0xF && mant == 0x7 {
                f32::NAN
            } else {
                s * 2.0f32.powi(exp - 7) * (1.0 + (mant as f32) / 8.0)
            };
            let got = E4M3_LUT[byte as usize];
            if expected.is_nan() {
                assert!(got.is_nan(), "byte {byte}: expected NaN, got {got}");
            } else {
                assert_eq!(
                    got.to_bits(),
                    expected.to_bits(),
                    "byte {byte}: {got} != {expected}"
                );
            }
        }
    }

    #[test]
    fn exp2_matches_powi_across_the_domain() {
        for e in -9..=8i32 {
            assert_eq!(exp2(e).to_bits(), 2.0f32.powi(e).to_bits(), "2^{e}");
        }
    }
}
