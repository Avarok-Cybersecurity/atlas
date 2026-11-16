// SPDX-License-Identifier: AGPL-3.0-only

//! Responsibility: exact IEEE 754 binary16 conversions for the EXL3 reader: fp16 bits to `f64` (always
//! exact) and `f64` to fp16 bits with ONE round-to-nearest-even step.
//!
//! The EXL3 codebooks are defined as "compute in float64, then round once to fp16" (TensorFold
//! `docs/recipes/exl3.md`, the format paragraph; `zig/src/core/exl3_format.zig:35-52`). The `half` crate
//! cannot be used for the rounding step: its `f16::from_f64` converts through `f32` on x86 with F16C (a
//! double rounding) and its software fallback drops the low 32 bits of the `f64` mantissa before
//! rounding, so a value just above a tie can round the wrong way. This module rounds from the full
//! 53-bit significand.

/// fp16 bit pattern of +1.0.
pub const ONE: u16 = 0x3C00;
/// fp16 bit pattern of -1.0.
pub const MINUS_ONE: u16 = 0xBC00;

/// The value of an fp16 bit pattern as `f64` (exact: every fp16 value is an `f64` value). NaN payloads
/// are not preserved.
pub fn to_f64(bits: u16) -> f64 {
    let negative = bits & 0x8000 != 0;
    let exp = i32::from((bits >> 10) & 0x1F);
    let man = f64::from(bits & 0x03FF);
    let mag = match exp {
        0 => man * 2f64.powi(-24),
        31 => {
            if man != 0.0 {
                return f64::NAN;
            }
            f64::INFINITY
        }
        e => (1024.0 + man) * 2f64.powi(e - 25),
    };
    if negative { -mag } else { mag }
}

/// `x` rounded to the nearest fp16 value, ties to even, as a bit pattern. Overflow gives an infinity
/// of the right sign, NaN gives the canonical quiet NaN, and values below half the smallest fp16
/// subnormal give a signed zero.
pub fn from_f64(x: f64) -> u16 {
    let bits = x.to_bits();
    let sign = ((bits >> 48) & 0x8000) as u16;
    let exp = ((bits >> 52) & 0x7FF) as i32;
    let man = bits & 0x000F_FFFF_FFFF_FFFF;
    if exp == 0x7FF {
        return if man != 0 { 0x7E00 } else { sign | 0x7C00 };
    }
    if exp == 0 {
        // Zero, or an f64 subnormal (below 2^-1022): far below half of fp16's smallest subnormal (2^-25).
        return sign;
    }
    let e = exp - 1023;
    if e > 15 {
        return sign | 0x7C00;
    }
    // The significand with its implicit bit: 53 bits, value = sig * 2^(e - 52).
    let sig = man | (1u64 << 52);
    // Normal results keep 11 significant bits (shift 42); subnormal results are multiples of 2^-24.
    let (shift, biased) = if e >= -14 {
        (42u32, (e + 15) as u16)
    } else {
        (42 + (-14 - e) as u32, 0u16)
    };
    if shift >= 64 {
        return sign;
    }
    let mut q = sig >> shift;
    let rem = sig & ((1u64 << shift) - 1);
    let halfway = 1u64 << (shift - 1);
    if rem > halfway || (rem == halfway && q & 1 == 1) {
        q += 1;
    }
    let q = q as u16;
    if biased == 0 {
        // Subnormal (q <= 1024; q == 1024 is the smallest normal, whose bit pattern is also 1024).
        return sign | q;
    }
    // q is in [1024, 2048]; a carry to 2048 moves into the exponent field, up to 0x7C00 (infinity).
    sign | ((biased << 10) + (q - 1024))
}
