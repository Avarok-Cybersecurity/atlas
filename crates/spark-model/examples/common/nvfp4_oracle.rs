// SPDX-License-Identifier: AGPL-3.0-only
//! Responsibility: an independent CPU oracle for the NVFP4 W4A16 GEMV, so the
//! `w4a16_ada_microtest` timing is only ever reported for a kernel whose math
//! and launch ABI were just checked.
//!
//! Format (read from `kernels/gb10/common/w4a16_gemv.cu:13-19`): `B_packed` is
//! `[N, K/2]` bytes, the byte at `[n, j]` holding `W[n, 2j]` in its low nibble
//! and `W[n, 2j+1]` in its high nibble, each nibble an E2M1 code; `B_scale` is
//! `[N, K/16]` E4M3 bytes, one per group of 16 values; `scale2` is a per-tensor
//! FP32 scale applied on top. `out[n] = sum_k A[k] * E2M1[nibble] * E4M3[scale] * scale2`.
//!
//! The oracle accumulates in FP64 and the kernel in FP32 in a different order,
//! so the comparison is a bound, not equality: 2 BF16 ulps of the reference plus
//! 2^-13 of the output's term magnitude (`sum |a * w|`), the same form as the FP8
//! W8A8 gate of `fp8_ada_microtest`.
#![allow(dead_code)]

/// E2M1 code to value (sign in the high bit of the nibble).
pub const E2M1: [f64; 16] = [
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
];

/// Accumulation error allowed per unit of term magnitude: 2^-13.
pub const ACC_TOL: f64 = 1.0 / 8192.0;
/// BF16's machine epsilon, 2^-7.
pub const BF16_EPS: f64 = 1.0 / 128.0;

/// Standard E4M3 (1-4-3, bias 7); the NaN encodings decode to NaN.
pub fn e4m3_to_f64(b: u8) -> f64 {
    let sign = if b & 0x80 != 0 { -1.0 } else { 1.0 };
    let e = i32::from((b >> 3) & 0xF);
    let m = f64::from(b & 7);
    if e == 0 {
        sign * m * 2f64.powi(-9)
    } else if e == 15 && (b & 7) == 7 {
        f64::NAN
    } else {
        sign * (1.0 + m / 8.0) * 2f64.powi(e - 7)
    }
}

/// BF16 bytes (little endian) to f32.
pub fn bf16_bytes_to_f32(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(2)
        .map(|c| half::bf16::from_bits(u16::from_le_bytes([c[0], c[1]])).to_f32())
        .collect()
}

/// The oracle output and each output's term magnitude. `swap_nibbles` is the
/// planted defect: high and low nibbles exchanged.
pub fn oracle(
    a: &[f32],
    packed: &[u8],
    scales: &[u8],
    scale2: f64,
    n: usize,
    k: usize,
    swap_nibbles: bool,
) -> (Vec<f64>, Vec<f64>) {
    let (mut out, mut mag) = (vec![0.0; n], vec![0.0; n]);
    for row in 0..n {
        let (mut acc, mut m) = (0.0f64, 0.0f64);
        for col in 0..k {
            let byte = packed[row * (k / 2) + col / 2];
            let odd = (col % 2 == 1) != swap_nibbles;
            let nib = if odd { byte >> 4 } else { byte & 0xF };
            let w =
                E2M1[usize::from(nib)] * e4m3_to_f64(scales[row * (k / 16) + col / 16]) * scale2;
            let term = f64::from(a[col]) * w;
            acc += term;
            m += term.abs();
        }
        out[row] = acc;
        mag[row] = m;
    }
    (out, mag)
}

/// How many BF16 outputs are outside the bound, and the worst error as a
/// fraction of its bound (above 1.0 is outside).
pub fn outside_bound(got_bf16: &[u8], want: &[f64], mag: &[f64]) -> (usize, f64) {
    let got = bf16_bytes_to_f32(got_bf16);
    let (mut bad, mut worst) = (0usize, 0.0f64);
    for ((g, w), m) in got.iter().zip(want).zip(mag) {
        let bound = 2.0 * BF16_EPS * w.abs() + ACC_TOL * m;
        let err = (f64::from(*g) - w).abs();
        let ratio = if bound > 0.0 {
            err / bound
        } else if err > 0.0 {
            f64::INFINITY
        } else {
            0.0
        };
        // A NaN output (err is NaN) is as wrong as an infinite one.
        let ratio = if ratio.is_nan() { f64::INFINITY } else { ratio };
        if ratio > 1.0 {
            bad += 1;
        }
        worst = worst.max(ratio);
    }
    (bad + want.len().saturating_sub(got.len()), worst)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn e4m3_known_values() {
        assert_eq!(e4m3_to_f64(0x38), 1.0);
        assert_eq!(e4m3_to_f64(0x30), 0.5);
        assert_eq!(e4m3_to_f64(0x46), 3.5);
        assert_eq!(e4m3_to_f64(0xB8), -1.0);
        assert_eq!(e4m3_to_f64(0x01), 2f64.powi(-9));
        assert!(e4m3_to_f64(0x7F).is_nan() && e4m3_to_f64(0xFF).is_nan());
        assert_eq!(e4m3_to_f64(0x00), 0.0);
    }

    /// One output, K = 16: activations 1, 2, 1, 2, ...; every byte 0x21 (low
    /// nibble 1 = 0.5, high nibble 2 = 1.0); scale byte 0x38 = 1.0.
    #[test]
    fn hand_computed_dot_and_planted_nibble_swap() {
        let a: Vec<f32> = (0..16).map(|k| 1.0 + (k % 2) as f32).collect();
        let packed = vec![0x21u8; 8];
        let scales = vec![0x38u8];
        let (out, mag) = oracle(&a, &packed, &scales, 1.0, 1, 16, false);
        // Even k: 0.5 * 1 = 0.5; odd k: 1.0 * 2 = 2.0; eight pairs: 20.
        assert_eq!(out[0], 20.0);
        assert_eq!(mag[0], 20.0);
        let (swapped, _) = oracle(&a, &packed, &scales, 1.0, 1, 16, true);
        // Even k: 1.0 * 1; odd k: 0.5 * 2: eight pairs of 2.0 = 16.
        assert_eq!(swapped[0], 16.0);
        let (doubled, _) = oracle(&a, &packed, &scales, 2.0, 1, 16, false);
        assert_eq!(doubled[0], 40.0);
    }

    #[test]
    fn sign_nibbles_and_group_scales() {
        // K = 32, two groups: scale bytes 0x38 (1.0) and 0x40 (2.0).
        // Nibble 0x9 is -0.5, nibble 0x1 is +0.5; a = 1.0 everywhere.
        let packed = vec![0x19u8; 16];
        let scales = vec![0x38u8, 0x40];
        let a = vec![1.0f32; 32];
        let (out, mag) = oracle(&a, &packed, &scales, 1.0, 1, 32, false);
        // Even k: -0.5, odd k: +0.5, so each pair cancels.
        assert_eq!(out[0], 0.0);
        // Group 0: 16 terms of 0.5; group 1: 16 terms of 1.0.
        assert_eq!(mag[0], 8.0 + 16.0);
    }

    fn bf16_bytes(v: &[f32]) -> Vec<u8> {
        v.iter()
            .flat_map(|x| half::bf16::from_f32(*x).to_bits().to_le_bytes())
            .collect()
    }

    #[test]
    fn the_bound_accepts_rounding_and_rejects_a_wrong_answer() {
        let want = [20.0f64, -3.5];
        let mag = [25.0f64, 4.0];
        let exact = bf16_bytes(&[20.0, -3.5]);
        assert_eq!(outside_bound(&exact, &want, &mag).0, 0);
        // One BF16 ulp at 20 is 0.125: inside the 2-ulp allowance.
        let one_ulp = bf16_bytes(&[20.125, -3.5]);
        assert_eq!(outside_bound(&one_ulp, &want, &mag).0, 0);
        // 10% off is far outside.
        let wrong = bf16_bytes(&[22.0, -3.5]);
        let (bad, worst) = outside_bound(&wrong, &want, &mag);
        assert_eq!(bad, 1);
        assert!(worst > 1.0);
        // NaN output is outside the bound.
        let nan = bf16_bytes(&[f32::NAN, -3.5]);
        assert_eq!(outside_bound(&nan, &want, &mag).0, 1);
        // A short output counts the missing elements.
        assert_eq!(outside_bound(&bf16_bytes(&[20.0]), &want, &mag).0, 1);
    }

    #[test]
    fn bf16_bytes_round_trip() {
        let v = bf16_bytes_to_f32(&bf16_bytes(&[1.0, -2.0, 0.5]));
        assert_eq!(v, vec![1.0, -2.0, 0.5]);
    }
}
