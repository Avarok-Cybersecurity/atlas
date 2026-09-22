// SPDX-License-Identifier: AGPL-3.0-only

//! Goldens for [`super::dequant_nvfp4_to_f32`].
//!
//! 🔴 Every expected number here is derived from the **OCP Microscaling
//! Formats (MX) v1.0** definitions of `E2M1` and `E8M0`/`E4M3`, written out
//! long-hand, NOT read back from the runtime tables the code under test uses.
//! A golden taken from `NVFP4_E2M1_LUT` would agree with a corrupted
//! `NVFP4_E2M1_LUT`.

use super::*;

/// OCP `E2M1`: 1 sign bit, 2 exponent bits (bias 1), 1 mantissa bit.
/// `e = 0` is subnormal (`2^-0 * m/2`), so the eight magnitudes are
/// `0, 0.5, 1, 1.5, 2, 3, 4, 6` and code `0b1xxx` negates.
const SPEC_E2M1: [f32; 16] = [
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
];

/// OCP `E4M3`: 1 sign, 4 exponent (bias 7), 3 mantissa; `e > 0` is
/// `2^(e-7) * (1 + m/8)`. Spelled out for the four codes this file uses.
const E4M3_HALF: u8 = 0x30; // 0 0110 000 -> 2^-1 * 1.0
const E4M3_ONE: u8 = 0x38; // 0 0111 000 -> 2^0  * 1.0
const E4M3_ONE_AND_A_HALF: u8 = 0x3C; // 0 0111 100 -> 2^0 * 1.5
const E4M3_MAX: u8 = 0x7E; // 0 1111 110 -> 2^8 * 1.75 = 448

/// Two `e2m1` codes to one byte, low nibble first — the ModelOpt packing.
fn pack(nibbles: &[u8]) -> Vec<u8> {
    nibbles
        .chunks_exact(2)
        .map(|c| (c[1] << 4) | (c[0] & 0x0F))
        .collect()
}

/// One block of 16, scale 1.0, global 1.0: the codebook itself must come back.
#[test]
fn one_block_reproduces_the_ocp_e2m1_codebook() {
    let nibbles: Vec<u8> = (0..16u8).collect();
    let packed = pack(&nibbles);
    assert_eq!(packed.len(), 8, "16 codes pack into 8 bytes");
    let out =
        dequant_nvfp4_to_f32("t", &packed, &[1, 8], &[E4M3_ONE], 1.0).expect("one clean block");
    assert_eq!(out.len(), 16);
    for (i, (got, want)) in out.iter().zip(SPEC_E2M1.iter()).enumerate() {
        assert_eq!(got.to_bits(), want.to_bits(), "code {i}");
    }
}

/// 🪤 The whole-tensor failure mode: reading the HIGH nibble first swaps every
/// column pair. Shape, finiteness and magnitudes all survive that, so only an
/// asymmetric byte catches it.
#[test]
fn even_column_is_the_low_nibble() {
    // byte 0x20 = high nibble 2 (-> 1.0), low nibble 0 (-> 0.0).
    let mut packed = vec![0x20u8];
    packed.extend_from_slice(&[0u8; 7]);
    let out = dequant_nvfp4_to_f32("t", &packed, &[1, 8], &[E4M3_ONE], 1.0).unwrap();
    assert_eq!(out[0], 0.0, "column 0 is the LOW nibble");
    assert_eq!(out[1], 1.0, "column 1 is the HIGH nibble");
}

/// Row stride, per-block scale selection and the global multiplier at once.
/// Every expected value is a product of exact binary fractions, so the
/// assertions are exact rather than approximate.
#[test]
fn scales_are_selected_per_row_and_per_block() {
    // 2 rows x 32 logical columns = 2 blocks per row, 16 packed bytes per row.
    let row: Vec<u8> = pack(&[2u8; 32]); // every code = 2 -> magnitude 1.0
    let mut packed = row.clone();
    packed.extend_from_slice(&row);
    // row 0: [1.0, 1.5]; row 1: [0.5, 448].
    let scales = [E4M3_ONE, E4M3_ONE_AND_A_HALF, E4M3_HALF, E4M3_MAX];
    let out = dequant_nvfp4_to_f32("t", &packed, &[2, 16], &scales, 0.25).unwrap();
    assert_eq!(out.len(), 64);
    assert_eq!(out[0], 1.0 * 1.0 * 0.25);
    assert_eq!(out[15], 1.0 * 1.0 * 0.25);
    assert_eq!(out[16], 1.0 * 1.5 * 0.25);
    assert_eq!(out[31], 1.0 * 1.5 * 0.25);
    assert_eq!(out[32], 1.0 * 0.5 * 0.25);
    assert_eq!(out[47], 1.0 * 0.5 * 0.25);
    assert_eq!(out[48], 1.0 * 448.0 * 0.25);
    assert_eq!(out[63], 1.0 * 448.0 * 0.25);
}

/// The real GLM shapes, reduced: the logical width is TWICE the on-disk one.
#[test]
fn the_logical_width_is_twice_the_on_disk_width() {
    let packed = vec![0u8; 3 * 8];
    let scales = vec![E4M3_ONE; 3]; // one 16-element block per row
    let out = dequant_nvfp4_to_f32("t", &packed, &[3, 8], &scales, 1.0).unwrap();
    assert_eq!(out.len(), 3 * 16, "[3, 8] U8 is a [3, 16] weight");
}

/// A short shard, a wrong scale count and a width that is not a whole number of
/// blocks are all errors, not quietly smaller tensors.
#[test]
fn malformed_inputs_are_errors() {
    let e = |r: anyhow::Result<Vec<f32>>| r.unwrap_err().to_string();

    let short = dequant_nvfp4_to_f32("t", &[0u8; 7], &[1, 8], &[E4M3_ONE], 1.0);
    assert!(e(short).contains("7 B of packed NVFP4"));

    let wrong_scales = dequant_nvfp4_to_f32("t", &[0u8; 8], &[1, 8], &[E4M3_ONE; 2], 1.0);
    assert!(e(wrong_scales).contains("block scales"));

    // 8 logical columns is half a block.
    let ragged = dequant_nvfp4_to_f32("t", &[0u8; 4], &[1, 4], &[E4M3_ONE], 1.0);
    assert!(e(ragged).contains("NVFP4 blocks"));

    let not_2d = dequant_nvfp4_to_f32("t", &[0u8; 8], &[1, 2, 4], &[E4M3_ONE], 1.0);
    assert!(e(not_2d).contains("must be 2-D"));
}

/// 🪤 The E4M3 LUT has all 256 entries, so a malformed block-scale byte
/// INDEXES rather than panicking: `0x7F` decodes to NaN and `0x80..` to a
/// negative. Both produce a well-formed tensor of the right shape — one full
/// of NaNs, one with a sign-flipped block — and neither has anything to assert
/// on until the logits are wrong. Range-check before the lookup.
#[test]
fn a_block_scale_byte_outside_the_finite_non_negative_range_is_refused() {
    // 0x7F = NaN, 0x80 = -0.0, 0xFF = -NaN, 0xB8 = -1.0.
    for bad in [0x7Fu8, 0x80, 0xB8, 0xFF] {
        let r = dequant_nvfp4_to_f32("t", &[0u8; 8], &[1, 8], &[bad], 1.0);
        let e = r.unwrap_err().to_string();
        assert!(
            e.contains(&format!("0x{bad:02X}")),
            "byte 0x{bad:02X} must be named in the error, got: {e}"
        );
        assert!(e.contains("E4M3"), "{e}");
    }
    // The boundary holds from the other side: 0x7E is 448.0, the largest
    // finite E4M3, and stays legal.
    assert!(dequant_nvfp4_to_f32("t", &[0u8; 8], &[1, 8], &[0x7E], 1.0).is_ok());
}
