// SPDX-License-Identifier: AGPL-3.0-only

//! Split out of `native_fp8_ffn_w8a8_microtest.rs` to keep it under the 500-LoC
//! cap. The rule and its comparator are the half of that example that needs no
//! GPU, which is also the half worth testing on its own — moving them here puts
//! the CPU tests beside the code they pin.
//!
//! Test-only harness code; no serving path runs any of it.

use half::bf16;

/// cuBLASLt-vs-kernel acceptance. The two implementations consume the SAME FP8
/// bytes and the SAME FP32 scales, so the only licensed difference between them
/// is the ORDER of the FP32 accumulation (tile shape, split-K, epilogue
/// association). That is a real difference and it has a bounded consequence:
/// each FP32 partial sum moves by a few parts in 1e7, which is far under one
/// BF16 step (2^-8 relative), so an output lands on a different BF16 value only
/// when the exact sum sits within that sliver of a rounding boundary — and then
/// it moves by exactly ONE step, never two. A layout bug does not look like
/// that; it permutes scales, which moves elements by whole factors.
///
/// Hence the acceptance below is a **tolerance on every element**, not a
/// bit-identity check:
///
///   * `|a - b| <= 1 BF16 ULP at max(|a|, |b|)` — one rounding step, measured
///     at the larger operand's magnitude so a pair straddling a binade boundary
///     is judged by the coarser grid it actually shares.
///   * OR both values are under [`CUBLAS_SMALL_MAGNITUDE`]. Near zero the
///     output is what is left after cancellation between O(K) terms of much
///     larger magnitude, so its remaining bits — including its SIGN — are set
///     by accumulation order and carry no information. 0.05 is ~1.7x the
///     largest sign-flipped magnitude measured on H100 (0.0148, 0.0167, 0.0237,
///     0.0297 on 2026-09-11) and ~2e-4 of the shapes' output range, i.e. below
///     anything the downstream SiLU/BF16 store can distinguish.
///
/// The count of elements outside that bound is the gate and must be **0**.
/// `sign_flips`, `unequal_bf16` and `max_ulp` are REPORTED, not gated.
///
/// WHY `max_ulp` IS NO LONGER A GATE. The previous bound was `max_ulp <= 2` on
/// an ORDINAL ULP distance — `|ord(a) - ord(b)|` over sign-magnitude BF16 — and
/// that metric scores a sign flip on a value of magnitude `v` as `2 * ord(v)`,
/// a number in the tens of thousands however tiny `v` is. The 2026-09-11 H100
/// run measured exactly that failure mode: cosine 0.9999996, rel_rms
/// 9.1e-4-9.3e-4, `max_abs` equal to one BF16 ULP at the largest outputs
/// (1.000 and 2.000), and `max_ulp ~= 31 000` decoding to sign flips at
/// |v| <= 0.03. A bound no correctly-ordered FP32 accumulation can satisfy is
/// a bit-identity gate wearing a tolerance's clothes; this is the tolerance it
/// was pretending to be.
///
/// `rel_rms` moves 1e-3 -> 2e-3 for headroom over the same run's 9.29e-4 while
/// staying an order of magnitude under the ~2.5e-2 E4M3 floor the W8A16
/// comparison sits on — which is what keeps these a LAYOUT test and not a
/// precision test. The row-major control fails all three: rel_rms 1.1e-2-8.8e-2,
/// cosine down to 0.9961, and `max_abs` 55.6 against a one-ULP bound of 2.
pub(crate) const CUBLAS_SMALL_MAGNITUDE: f64 = 0.05;
pub(crate) const CUBLAS_COSINE_GATE: f64 = 0.99999;
pub(crate) const CUBLAS_REL_RMS_GATE: f64 = 2e-3;

fn to_f64(bits: &[u16]) -> Vec<f64> {
    bits.iter()
        .map(|b| bf16::from_bits(*b).to_f32() as f64)
        .collect()
}

/// BF16 bits → a monotonically ordered integer, so `|ord(a) - ord(b)|` is the
/// ULP distance (the standard sign-magnitude → two's-complement remap).
fn ord(bits: u16) -> i32 {
    if bits & 0x8000 != 0 {
        -((bits & 0x7fff) as i32)
    } else {
        bits as i32
    }
}

/// One BF16 ULP at `bits`' magnitude — the gap between it and its neighbour.
/// BF16 stores 7 mantissa bits, so a normal value with unbiased exponent `e`
/// steps by `2^(e - 7)`; every subnormal steps by the smallest normal's step,
/// `2^-133`. Inf/NaN report an infinite step so nothing is judged "close" to
/// them by arithmetic accident (the `<=` below still rejects a NaN difference).
fn bf16_ulp(bits: u16) -> f64 {
    let biased_exp = ((bits >> 7) & 0xff) as i32;
    match biased_exp {
        0xff => f64::INFINITY,
        0 => (-133.0_f64).exp2(),
        e => ((e - 127 - 7) as f64).exp2(),
    }
}

/// The per-element acceptance described on [`CUBLAS_SMALL_MAGNITUDE`]: one BF16
/// rounding step at the larger magnitude, or both values under the small-value
/// escape.
fn within_one_bf16_ulp(a_bits: u16, b_bits: u16) -> bool {
    let a = bf16::from_bits(a_bits).to_f32() as f64;
    let b = bf16::from_bits(b_bits).to_f32() as f64;
    if a.abs() < CUBLAS_SMALL_MAGNITUDE && b.abs() < CUBLAS_SMALL_MAGNITUDE {
        return true;
    }
    let ulp = if a.abs() >= b.abs() {
        bf16_ulp(a_bits)
    } else {
        bf16_ulp(b_bits)
    };
    (a - b).abs() <= ulp
}

/// Opposite signs with neither value zero. Reported for diagnosis: on these
/// shapes every flip seen so far has been a cancellation residue under the
/// small-magnitude escape, and a flip on a LARGE output would fail the
/// one-ULP bound anyway.
fn is_sign_flip(a_bits: u16, b_bits: u16) -> bool {
    let a = bf16::from_bits(a_bits).to_f32();
    let b = bf16::from_bits(b_bits).to_f32();
    a != 0.0 && b != 0.0 && (a < 0.0) != (b < 0.0)
}

pub(crate) struct Compare {
    pub(crate) max_abs: f64,
    pub(crate) cosine: f64,
    pub(crate) rel_rms: f64,
    pub(crate) max_ulp: i32,
    pub(crate) unequal: usize,
    /// Elements failing [`within_one_bf16_ulp`] — the cuBLASLt-vs-kernel gate.
    pub(crate) over_bound: usize,
    pub(crate) sign_flips: usize,
}

pub(crate) fn compare(a_bits: &[u16], b_bits: &[u16]) -> Compare {
    let (a, b) = (to_f64(a_bits), to_f64(b_bits));
    let (mut dot, mut na, mut nb, mut max_abs, mut sq_diff, mut sq_ref) =
        (0.0, 0.0, 0.0, 0.0_f64, 0.0, 0.0);
    for (x, y) in a.iter().zip(&b) {
        dot += x * y;
        na += x * x;
        nb += y * y;
        max_abs = max_abs.max((x - y).abs());
        sq_diff += (x - y) * (x - y);
        sq_ref += y * y;
    }
    Compare {
        max_abs,
        cosine: dot / (na.sqrt() * nb.sqrt()),
        rel_rms: (sq_diff / sq_ref.max(f64::MIN_POSITIVE)).sqrt(),
        max_ulp: a_bits
            .iter()
            .zip(b_bits)
            .map(|(x, y)| (ord(*x) - ord(*y)).abs())
            .max()
            .unwrap_or(0),
        unequal: a_bits.iter().zip(b_bits).filter(|(x, y)| x != y).count(),
        over_bound: a_bits
            .iter()
            .zip(b_bits)
            .filter(|(x, y)| !within_one_bf16_ulp(**x, **y))
            .count(),
        sign_flips: a_bits
            .iter()
            .zip(b_bits)
            .filter(|(x, y)| is_sign_flip(**x, **y))
            .count(),
    }
}

/// CPU tests for the cuBLASLt-vs-kernel acceptance. They need no GPU, no
/// kernels and no model: the point of pulling the bound out into
/// [`within_one_bf16_ulp`] is that the rule can be pinned on synthetic vectors
/// instead of only being exercised by the H100 run it gates.
///
/// Enabled by `test = true` on this example's `[[example]]` stanza, so
/// `cargo test -p spark-model --example native_fp8_ffn_w8a8_microtest
/// --features cuda,gpu-examples` runs them.
#[cfg(test)]
mod tests {
    use super::*;

    /// The BF16 value one representable step above `x`, by construction rather
    /// than by arithmetic: increment the significand of the stored bits.
    fn next_bf16_up(x: f32) -> u16 {
        let b = bf16::from_f32(x).to_bits();
        assert!(b & 0x8000 == 0, "helper is for positive values");
        b + 1
    }

    #[test]
    fn ulp_is_the_gap_to_the_next_bf16() {
        // 1.0 sits at the bottom of its binade: 7 stored mantissa bits -> 2^-7.
        let one = bf16::from_f32(1.0).to_bits();
        assert_eq!(bf16_ulp(one), 2.0_f64.powi(-7));
        // And the step really is the distance to the neighbour.
        let up = bf16::from_bits(next_bf16_up(1.0)).to_f32() as f64;
        assert!((up - 1.0 - bf16_ulp(one)).abs() < 1e-12);
        // The H100 run's max_abs values are exactly one ULP at their magnitude:
        // 1.000 in [128, 256) and 2.000 in [256, 512).
        assert_eq!(bf16_ulp(bf16::from_f32(200.0).to_bits()), 1.0);
        assert_eq!(bf16_ulp(bf16::from_f32(400.0).to_bits()), 2.0);
    }

    #[test]
    fn one_ulp_apart_passes() {
        for v in [1.0_f32, 3.5, 128.0, 200.0, 17408.0, 0.0625] {
            let a = bf16::from_f32(v).to_bits();
            let b = next_bf16_up(v);
            assert!(
                within_one_bf16_ulp(a, b),
                "one step at {v} should be accepted"
            );
            assert!(within_one_bf16_ulp(b, a), "the bound is symmetric at {v}");
        }
    }

    #[test]
    fn two_ulps_apart_fails() {
        for v in [1.0_f32, 3.5, 128.0, 200.0, 17408.0, 0.0625] {
            let a = bf16::from_f32(v).to_bits();
            let b = next_bf16_up(v) + 1;
            assert!(
                !within_one_bf16_ulp(a, b),
                "two steps at {v} must be rejected — this is the resolution the \
                 gate exists to have"
            );
        }
    }

    #[test]
    fn near_zero_sign_flip_passes() {
        // The H100 residual: cancellation noise whose sign is set by the
        // accumulation order. Magnitudes are the four decoded from the
        // 2026-09-11 run, plus the escape's own boundary.
        for v in [0.0148_f32, 0.0167, 0.0237, 0.0297, 0.049] {
            let a = bf16::from_f32(v).to_bits();
            let b = bf16::from_f32(-v).to_bits();
            assert!(
                within_one_bf16_ulp(a, b),
                "a sign flip at |v|={v} is cancellation residue, not a layout bug"
            );
            assert!(is_sign_flip(a, b), "and it is still COUNTED as a flip");
        }
    }

    #[test]
    fn sign_flip_above_the_escape_fails() {
        // Nothing large gets the escape: a flip at 0.06 is 0.12 apart against a
        // one-ULP bound of 2^-11, and a flip at 200.0 is 400 apart against 1.0.
        for v in [0.06_f32, 1.0, 200.0] {
            let a = bf16::from_f32(v).to_bits();
            let b = bf16::from_f32(-v).to_bits();
            assert!(
                !within_one_bf16_ulp(a, b),
                "a sign flip at |v|={v} is outside the small-value escape"
            );
        }
    }

    #[test]
    fn equal_values_and_zero_pass_and_nan_does_not() {
        let x = bf16::from_f32(7.25).to_bits();
        assert!(within_one_bf16_ulp(x, x));
        assert!(!is_sign_flip(x, x));
        let zero = bf16::from_f32(0.0).to_bits();
        let neg_zero = bf16::from_f32(-0.0).to_bits();
        assert!(within_one_bf16_ulp(zero, neg_zero));
        assert!(!is_sign_flip(zero, neg_zero), "+-0 is not a sign flip");
        let nan = bf16::from_f32(f32::NAN).to_bits();
        assert!(!within_one_bf16_ulp(nan, x), "NaN must never be accepted");
        assert!(!within_one_bf16_ulp(nan, nan));
    }

    /// A pair straddling a binade boundary is judged at the COARSER grid, which
    /// is the larger magnitude's — otherwise 127.5 and 128.0, genuine BF16
    /// neighbours, would be scored as a violation. The deliberate cost is that
    /// just under a boundary the rule admits two steps of the finer grid below
    /// it (127.0 vs 128.0); at the boundary the two values' shared resolution
    /// IS the coarse one, and a layout bug is never a boundary-sized error.
    #[test]
    fn straddling_a_binade_uses_the_larger_magnitude() {
        let hi = bf16::from_f32(128.0).to_bits(); // ULP 1.0
        let lo = hi - 1; // 127.5, the largest BF16 below 128 (ULP 0.5)
        assert_eq!(bf16::from_bits(lo).to_f32(), 127.5);
        assert!(within_one_bf16_ulp(lo, hi));
        assert!(within_one_bf16_ulp(hi, lo));
        assert!(
            within_one_bf16_ulp(lo - 1, hi),
            "127.0 vs 128.0 is one step of the coarser grid — accepted by design"
        );
        assert!(
            !within_one_bf16_ulp(lo - 2, hi),
            "126.5 vs 128.0 is 1.5 ULP even at the coarser grid"
        );
    }

    /// The whole-vector `compare` must agree with the per-element rule, and
    /// count rather than gate the flips.
    #[test]
    fn compare_counts_over_bound_and_sign_flips() {
        let a: Vec<u16> = vec![
            bf16::from_f32(1.0).to_bits(),    // equal
            bf16::from_f32(3.5).to_bits(),    // one ulp apart
            bf16::from_f32(0.0148).to_bits(), // near-zero sign flip
            bf16::from_f32(64.0).to_bits(),   // two ulps apart
        ];
        let b: Vec<u16> = vec![
            bf16::from_f32(1.0).to_bits(),
            next_bf16_up(3.5),
            bf16::from_f32(-0.0148).to_bits(),
            next_bf16_up(64.0) + 1,
        ];
        let c = compare(&a, &b);
        assert_eq!(c.over_bound, 1, "only the two-ULP element is out of bound");
        assert_eq!(c.sign_flips, 1);
        assert_eq!(c.unequal, 3);
    }
}
