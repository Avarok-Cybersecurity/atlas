// SPDX-License-Identifier: AGPL-3.0-only

//! Goldens and the round-trip property for [`super::quantize_to_nvfp4`].
//!
//! 🔴 As in `nvfp4_dequant_tests.rs`, every expected code and value is written
//! out from the **OCP Microscaling Formats (MX) v1.0** definitions of `E2M1`
//! and `E4M3`, not read back from the runtime tables the code under test uses.

use super::super::nvfp4_dequant::dequant_nvfp4_to_f32;
use super::*;

/// OCP `E2M1` magnitudes: `e = 0` subnormal `m/2`, else `2^(e-1) * (1 + m/2)`.
const SPEC_E2M1_MAG: [f32; 8] = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];

/// OCP `E4M3` codes, long-hand. `2^(e-7) * (1 + m/8)` for `e > 0`;
/// `2^-6 * m/8` for `e = 0`.
const E4M3_ZERO: u8 = 0x00;
const E4M3_MIN_SUBNORMAL: u8 = 0x01; // 2^-6 * 1/8 = 2^-9
const E4M3_ONE: u8 = 0x38; // 0 0111 000
const E4M3_ONE_EIGHTH_UP: u8 = 0x39; // 0 0111 001 -> 1.125
const E4M3_ONE_QUARTER_UP: u8 = 0x3A; // 0 0111 010 -> 1.25
const E4M3_MAX: u8 = 0x7E; // 0 1111 110 -> 2^8 * 1.75 = 448

fn nibble(packed: &[u8], flat: usize) -> u8 {
    let b = packed[flat / 2];
    if flat.is_multiple_of(2) {
        b & 0x0F
    } else {
        b >> 4
    }
}

/// A block whose 16 values are the signed `E2M1` ladder times `k`.
fn ladder_block(k: f32) -> Vec<f32> {
    SPEC_E2M1_MAG
        .iter()
        .map(|m| m * k)
        .chain(SPEC_E2M1_MAG.iter().map(|m| -m * k))
        .collect()
}

/// Deterministic, reproducible, and spread over several binades — no `rand`
/// dependency and no seed that drifts between runs.
fn pseudo_random(n: usize) -> Vec<f32> {
    let mut s = 0x2545_F491_4F6C_DD1Du64;
    (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            let unit = ((s >> 40) as f32) / 16_777_216.0 - 0.5; // [-0.5, 0.5)
            let binade = 1u32 << (s as u32 % 8);
            unit * binade as f32
        })
        .collect()
}

/// Data that is EXACTLY representable comes back bit-for-bit. Both blocks are
/// built so the two roundings are identities: the tensor amax is `6 * 448`, so
/// `weight_scale_2` is exactly 1, and each block's amax over 6 is an exact
/// `E4M3` value.
#[test]
fn exactly_representable_data_round_trips_exactly() {
    let mut values = ladder_block(448.0); // amax 2688 = 6 * 448
    values.extend(ladder_block(1.0)); // amax 6
    let q = quantize_to_nvfp4("t", &values, 1, 32).unwrap();

    assert_eq!(
        q.scale_2, 1.0,
        "amax = 6 * 448 makes the global scale exactly 1"
    );
    assert_eq!(q.scales, vec![E4M3_MAX, E4M3_ONE], "448 then 1");

    let back = dequant_nvfp4_to_f32("t", &q.packed, &[1, 16], &q.scales, q.scale_2).unwrap();
    assert_eq!(
        back, values,
        "an exactly representable tensor must be a fixed point"
    );
}

/// 🪤 The bias test. Round-to-nearest-even puts every tie on the even code;
/// truncation would put all seven on the code below and shrink the tensor.
#[test]
fn e2m1_ties_go_to_the_even_code() {
    // Midpoints of consecutive E2M1 magnitudes: 0|0.5, 0.5|1, 1|1.5, 1.5|2,
    // 2|3, 3|4, 4|6. The trailing 6.0 fixes the block amax so the effective
    // scale is exactly 1 and the codes are read against the ladder itself.
    let mut block = vec![0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0, 6.0];
    block.extend([0.0; 8]);
    let mut values = block;
    values.extend(ladder_block(448.0)); // pins the tensor amax at 6 * 448
    let q = quantize_to_nvfp4("t", &values, 1, 32).unwrap();
    assert_eq!(q.scales[0], E4M3_ONE, "block amax 6 / 6 / 1 = 1.0");

    // Even codes: 0 (0.0), 2 (1.0), 2 (1.0), 4 (2.0), 4 (2.0), 6 (4.0),
    // 6 (4.0); then the exact 6.0 -> 7.
    let want = [0u8, 2, 2, 4, 4, 6, 6, 7];
    for (i, w) in want.iter().enumerate() {
        assert_eq!(nibble(&q.packed, i), *w, "tie {i}");
    }
}

/// The `E4M3` scale encoder, against spec values and both tie directions.
#[test]
fn e4m3_scales_round_to_nearest_even() {
    assert_eq!(encode_e4m3_rne(0.0), E4M3_ZERO);
    assert_eq!(encode_e4m3_rne(1.0), E4M3_ONE);
    assert_eq!(encode_e4m3_rne(448.0), E4M3_MAX);
    assert_eq!(
        encode_e4m3_rne(1.0e9),
        E4M3_MAX,
        "saturates, never wraps to NaN"
    );
    assert_eq!(encode_e4m3_rne(0.001_953_125), E4M3_MIN_SUBNORMAL, "2^-9");
    // 1.0625 is exactly between 1.0 (0x38, even) and 1.125 (0x39) -> 0x38.
    assert_eq!(encode_e4m3_rne(1.0625), E4M3_ONE);
    // 1.1875 is exactly between 1.125 (0x39) and 1.25 (0x3A, even) -> 0x3A.
    assert_eq!(encode_e4m3_rne(1.1875), E4M3_ONE_QUARTER_UP);
    // Not a tie: nearest wins outright.
    assert_eq!(encode_e4m3_rne(1.1), E4M3_ONE_EIGHTH_UP);
}

/// The round-trip error is bounded by what NVFP4 can do, not merely "small".
///
/// Per block the effective step is `e4m3(amax/6) * scale_2`, within `2^-4` of
/// `amax/6` because `E4M3` carries 3 mantissa bits. The widest gap in the
/// `E2M1` ladder is `4 -> 6`, so a nearest-value quantiser errs by at most
/// half of it — one effective step — and a scaled value pushed past 6 by the
/// scale's own rounding clamps with even less. Hence
/// `|err| <= (1 + 2^-4) / 6 * amax ~= 0.178 * amax`; the gate is 0.20.
///
/// 🔴 The gate discriminates: truncation toward zero would err by a FULL gap,
/// `~0.354 * amax`, and fail here.
#[test]
fn round_trip_error_is_within_the_nvfp4_bound() {
    const ROWS: usize = 7;
    const COLS: usize = 64;
    let values = pseudo_random(ROWS * COLS);
    let q = quantize_to_nvfp4("t", &values, ROWS, COLS).unwrap();
    let back =
        dequant_nvfp4_to_f32("t", &q.packed, &[ROWS, COLS / 2], &q.scales, q.scale_2).unwrap();
    assert_eq!(back.len(), values.len());

    let mut worst = 0.0f32;
    for r in 0..ROWS {
        for g in 0..COLS / NVFP4_GROUP_SIZE {
            let base = r * COLS + g * NVFP4_GROUP_SIZE;
            let block = &values[base..base + NVFP4_GROUP_SIZE];
            let amax = block.iter().fold(0.0f32, |m, v| m.max(v.abs()));
            for i in 0..NVFP4_GROUP_SIZE {
                let err = (back[base + i] - values[base + i]).abs();
                assert!(
                    err <= 0.20 * amax,
                    "element {} err {err} against block amax {amax}",
                    base + i
                );
                worst = worst.max(if amax > 0.0 { err / amax } else { 0.0 });
            }
        }
    }
    // The bound must be TIGHT enough to be a test: if the data never came
    // close to it, the assertion above would pass for a broken quantiser too.
    assert!(
        worst > 0.05,
        "worst relative error {worst} — the gate is not exercised"
    );
}

/// Per-block scaling is the point: one loud block must not flatten a quiet one.
#[test]
fn a_quiet_block_survives_beside_a_loud_one() {
    let mut values = ladder_block(1024.0);
    values.extend(ladder_block(1.0));
    let q = quantize_to_nvfp4("t", &values, 1, 32).unwrap();
    let back = dequant_nvfp4_to_f32("t", &q.packed, &[1, 16], &q.scales, q.scale_2).unwrap();
    assert!(
        back[16..].iter().any(|v| *v != 0.0),
        "a block 1024x quieter than the tensor amax must not be flattened"
    );
    for i in 16..32 {
        let err = (back[i] - values[i]).abs();
        assert!(
            err <= 0.20 * 6.0,
            "quiet element {i}: {} vs {}",
            back[i],
            values[i]
        );
    }
}

/// 🪤 NVFP4's dynamic range is FINITE and this is where it ends: the block
/// scale is `E4M3`, whose smallest positive value is the subnormal `2^-9`, so
/// a block whose amax is more than `448 / 2^-9 = 229376` times below the
/// TENSOR's amax rounds its scale to zero and quantises to all zeros.
///
/// Recorded as a test rather than a comment because it is a real property of
/// the format that ModelOpt shares — not a defect to fix here — and because a
/// future "improvement" that clamps the scale up instead would then stop
/// matching the checkpoint-quantised experts it has to be indistinguishable
/// from.
#[test]
fn a_block_far_below_the_tensor_amax_underflows_to_zero() {
    let mut values = ladder_block(1.0e6);
    values.extend(ladder_block(1.0));
    let q = quantize_to_nvfp4("t", &values, 1, 32).unwrap();
    assert_eq!(q.scales[1], E4M3_ZERO, "the quiet block's scale underflows");
    let back = dequant_nvfp4_to_f32("t", &q.packed, &[1, 16], &q.scales, q.scale_2).unwrap();
    assert!(
        back[16..].iter().all(|v| *v == 0.0),
        "an underflowed scale must give zeros, never NaN"
    );
}

/// An all-zero tensor is a legal weight, not a division by zero.
#[test]
fn an_all_zero_tensor_quantises_to_zeros() {
    let q = quantize_to_nvfp4("t", &[0.0f32; 16], 1, 16).unwrap();
    assert_eq!(q.scale_2, 1.0);
    assert!(q.scale_2.is_finite());
    assert_eq!(q.packed, vec![0u8; 8]);
    assert_eq!(q.scales, vec![E4M3_ZERO]);
    let back = dequant_nvfp4_to_f32("t", &q.packed, &[1, 8], &q.scales, q.scale_2).unwrap();
    assert!(back.iter().all(|v| *v == 0.0), "no NaN from a 0 * 0 scale");
}

/// 🪤 Sign survives a magnitude that rounds to zero — `-0.0` is code `0x8`.
#[test]
fn the_sign_survives_a_magnitude_that_rounds_to_zero() {
    assert_eq!(encode_e2m1_rne(-0.0), 0x8);
    assert_eq!(encode_e2m1_rne(0.0), 0x0);
    // A tiny negative next to a large positive: scaled below half an LSB.
    assert_eq!(encode_e2m1_rne(-1.0e-9), 0x8);
    assert_eq!(
        encode_e2m1_rne(-7.0),
        0xF,
        "saturates to -6, keeping the sign"
    );
}

/// 🪤 The sweep is split into row bands across the cores, so a band boundary
/// must not change a single byte. It cannot — `scale_2` is computed before the
/// split and no block spans a row — and this pins that: the banded result and
/// a deliberately single-band one are byte-identical.
#[test]
fn the_row_banding_changes_nothing() {
    const ROWS: usize = 37; // prime, so no band count divides it evenly
    const COLS: usize = 48;
    let values = pseudo_random(ROWS * COLS);
    let banded = quantize_to_nvfp4("t", &values, ROWS, COLS).unwrap();

    let mut packed = vec![0u8; ROWS * COLS / 2];
    let mut scales = vec![0u8; ROWS * COLS / NVFP4_GROUP_SIZE];
    quantize_rows(&values, &mut packed, &mut scales, COLS, banded.scale_2);
    assert_eq!(banded.packed, packed);
    assert_eq!(banded.scales, scales);
}

/// Shape and finiteness are checked, not assumed.
#[test]
fn malformed_inputs_are_errors() {
    let e = |r: Result<Nvfp4Blob>| r.unwrap_err().to_string();
    assert!(e(quantize_to_nvfp4("t", &[0.0; 15], 1, 16)).contains("15 elements"));
    assert!(e(quantize_to_nvfp4("t", &[0.0; 8], 1, 8)).contains("NVFP4 blocks"));
    assert!(e(quantize_to_nvfp4("t", &[], 4, 0)).contains("NVFP4 blocks"));
    let mut nan = vec![1.0f32; 16];
    nan[3] = f32::NAN;
    assert!(e(quantize_to_nvfp4("t", &nan, 1, 16)).contains("non-finite"));
    let mut inf = vec![1.0f32; 16];
    inf[3] = f32::INFINITY;
    assert!(e(quantize_to_nvfp4("t", &inf, 1, 16)).contains("non-finite"));
}

/// The blob's three pieces are exactly the sizes `bind_expert` will hand the
/// kernel: `[rows, cols/2]` packed and `[rows, cols/16]` scales.
#[test]
fn the_blob_has_the_shapes_the_kernel_indexes() {
    let q = quantize_to_nvfp4("t", &pseudo_random(4 * 64), 4, 64).unwrap();
    assert_eq!(q.packed.len(), 4 * 32);
    assert_eq!(q.scales.len(), 4 * 4);
}
