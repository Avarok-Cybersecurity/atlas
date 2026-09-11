// SPDX-License-Identifier: AGPL-3.0-only

//! HOST SIMULATION of the ACCUMULATION FLOOR — the test that settles round 9's
//! `native_bf16_lm_head_m16_microtest` red cell.
//!
//! ROUND 9 (1xH100, 2026-09-11, the real `[N=248077, K=5120]` BF16 head): the
//! speed target was MET (M=16 best arm 0.914 ms / 2,779 GB/s against
//! `dense_gemv_bf16_batchm`'s 3.486 ms, 3.67x) and `rel_rms` was 1.1e-4 against
//! a 1e-3 budget — a 9x margin — but the PER-ELEMENT gate failed at every M:
//!
//! | M | `max_ulp` | `over_budget` | `sign_flips` | `n64_over_budget` |
//! |---|---|---|---|---|
//! | 5 | 32 | 10 | 0 | 10 |
//! | 8 | 100 | 16 | 1 | 16 |
//! | 13 | 100 | 25 | 1 | 25 |
//! | 16 | 100 | 37 | 1 | 37 |
//!
//! THREE facts in that table kill the defect hypotheses before any simulation:
//!
//! 1. **`over_budget` is linear in M** — 10/16/25/37 is 2.0/2.0/1.9/2.3 per
//!    row. A per-element statistical tail scales with the number of elements; a
//!    boundary defect does not. It also already fails at M=5, so it is not the
//!    second `m16n8k16` row group (rows 8..15).
//! 2. **`n64` rejects the IDENTICAL set at every M.** The 32- and 64-wide CTAs
//!    put their tile edges at different columns, so no tile-edge defect can
//!    produce the same rejected elements on both.
//! 3. **Every rejected element is a cancelled logit**: the receipt's printed
//!    `|ref|/rms` runs 4.9e-6..2.6e-4, and the worst (`ulp=100`) is
//!    `reference=-1.173019409e-4` against a block RMS of 23.80.
//!
//! [`the_round9_lm_head_outlier_is_a_cancelled_logit`] reconstructs that exact
//! element and shows the shipped `2^-20 * rms` floor rejects it while
//! [`m16_tc_acc_floor`] admits it. The remaining tests show WHY the old
//! constant was wrong — it did not scale with the reduction depth — and that
//! the new one has not become a rubber stamp.
//!
//! WHY THE REDUCTION MODEL HERE IS CHEAP. The index math (staging map, fragment
//! gather, store map) is pinned at the real geometry in
//! `dense_gemm_m16_bf16_tests.rs`. What is under test HERE is the METRIC, and
//! the metric depends only on the two REDUCTION ORDERS over K — so these
//! fixtures carry the real orders and a sampled N. Running the real
//! `[248077, 5120]` here would be a 2.0-GMAC unit test that still could not
//! reproduce the H100's exact tail, because the `m16n8k16`'s internal
//! 16-product order is unspecified hardware and any host model of it is
//! quieter than the part. The host model's own `rel_rms` at the real shape is
//! 3.3e-5 against the H100's 1.107e-4, i.e. the part is ~11x noisier in
//! variance — which is exactly why the floor cannot be fitted to a host run.

use crate::layers::dense_ffn::m16_tc::{
    M16_TC_ACC_FLOOR_MARGIN, m16_tc_acc_floor, within_m16_tc_budget,
};
use half::bf16;

/// Round 6's fixed floor, kept here ONLY so the round-9 pin can show what it
/// did to the LM head. It is `2^-20` of the block RMS, independent of K.
const ROUND6_FIXED_FLOOR: f64 = 9.536_743_164_062_5e-7;

/// The head's reduction depth.
const K_HEAD: usize = 5120;

/// The reference RMS round 9 measured on the real head at M=16.
const HEAD_RMS: f64 = 23.8027;

/// `dense_gemv_bf16`'s reduction order: 64 lanes each walking 8-wide chunks at
/// a stride of 64, lo-then-hi inside a chunk, then the 32-lane shfl butterfly
/// per warp and one add across the two warps. The REFERENCE arm.
fn gemv_order(a: &[f32], b: &[f32]) -> f32 {
    let k = a.len();
    let mut lanes = [0.0_f32; 64];
    for (lane, acc) in lanes.iter_mut().enumerate() {
        let mut kv = lane;
        while kv < k / 8 {
            for i in 0..8 {
                *acc += a[kv * 8 + i] * b[kv * 8 + i];
            }
            kv += 64;
        }
    }
    let mut warps = [0.0_f32; 2];
    for (w, out) in warps.iter_mut().enumerate() {
        let mut v = [0.0_f32; 32];
        v.copy_from_slice(&lanes[w * 32..(w + 1) * 32]);
        let mut off = 16;
        while off > 0 {
            for l in 0..off {
                v[l] += v[l + off];
            }
            off >>= 1;
        }
        *out = v[0];
    }
    warps[0] + warps[1]
}

/// `dense_gemm_m16_bf16`'s reduction order: ONE uninterrupted FP32 accumulator
/// stepped `K/16` times, each step adding one `m16n8k16`'s 16 products. The
/// MMA's internal order is unspecified hardware, so this models it as the
/// balanced tree — the quietest plausible realisation, which is the
/// conservative choice for a test that must hold whichever one the part picks.
fn mma_order(a: &[f32], b: &[f32]) -> f32 {
    let mut acc = 0.0_f32;
    for s in 0..a.len() / 16 {
        let mut p = [0.0_f32; 16];
        for (i, slot) in p.iter_mut().enumerate() {
            *slot = a[s * 16 + i] * b[s * 16 + i];
        }
        let mut w = 8;
        while w > 0 {
            for i in 0..w {
                p[i] += p[i + w];
            }
            w >>= 1;
        }
        acc += p[0];
    }
    acc
}

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u32 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1);
        (self.0 >> 32) as u32
    }
    /// The oracles' alphabet: a BF16-representable value in [-1, 1].
    fn bf16(&mut self) -> f32 {
        bf16::from_f32(((self.next() % 2049) as f32 - 1024.0) / 1024.0).to_f32()
    }
}

/// Weight columns per measurement of the order gap. Four is enough for an RMS
/// stable to the factor-of-two margins asserted below, and the whole sweep
/// stays under ~9 MMAC so this runs in CI on a laptop.
const DRAWS: usize = 4;

/// RMS of `mma_order - gemv_order` over [`DRAWS`] random weight columns: the
/// ABSOLUTE FP32 accumulation noise, measured rather than modelled.
///
/// 🔴 Measured on ORDINARY random columns, not on a hand-built cancelling one.
/// A row engineered to cancel term by term (`w[2i] = a[2i+1]`, `w[2i+1] =
/// -a[2i]`) is degenerate here: FP32 negation is exact and BOTH reduction
/// orders keep those pairs symmetric, so the two agree BIT-FOR-BIT and the gap
/// measures exactly zero. The gap is a property of the REDUCTION and does not
/// care whether the output cancelled; cancellation only decides whether an
/// ORDINAL budget can still see it.
fn order_gap_rms(rng: &mut Rng, a: &[f32]) -> f64 {
    let mut sum = 0.0_f64;
    for _ in 0..DRAWS {
        let b: Vec<f32> = (0..a.len()).map(|_| rng.bf16()).collect();
        let g = f64::from(mma_order(a, &b)) - f64::from(gemv_order(a, &b));
        sum += g * g;
    }
    (sum / DRAWS as f64).sqrt()
}

/// The OUTPUT row RMS an activation row produces against this alphabet's
/// weights: `o_n = dot(a, b_n)` has `E[o^2] = ||a||^2 * var(b)`, and the
/// alphabet's per-element variance is 1/3 — so the observable the floor is
/// scaled by is `||a|| / sqrt(3)`. Stated as arithmetic rather than measured,
/// so the fixture cannot drift from the scale the predicate is fed.
fn row_rms_of(row: &[f32]) -> f64 {
    let sum: f64 = row.iter().map(|v| f64::from(*v) * f64::from(*v)).sum();
    sum.sqrt() / 3.0_f64.sqrt()
}

/// THE ROUND-9 PIN — the element the H100 rejected, reconstructed from the
/// receipt and put through both floors.
///
/// `reference = -1.173019409e-4` is the value the microtest printed for the
/// worst cell (`ulp=100`) at M=8, 13 and 16; `HEAD_RMS` is the block RMS it
/// printed alongside. One hundred ordinal BF16 ULP from that reference is an
/// absolute error of 4.8e-5 (upward) or 9.1e-5 (downward) — both are checked,
/// because the receipt prints the distance and not the direction.
#[test]
fn the_round9_lm_head_outlier_is_a_cancelled_logit() {
    let reference = bf16::from_f32(-1.173_019_4e-4);
    let rb = reference.to_bits();
    assert_eq!(rb, 0xB8F6, "the receipt's reference must be BF16-exact");
    let relative = f64::from(reference.to_f32()).abs() / HEAD_RMS;
    assert!(
        (4.8e-6..5.0e-6).contains(&relative),
        "|ref|/rms is {relative:.3e}; the receipt printed 4.9e-6"
    );

    for direction in [-100_i32, 100] {
        // Ordinal +/-100 from the reference, back through the monotone map.
        let ord = -(i32::from(rb & 0x7FFF)) + direction;
        let ab = if ord < 0 {
            ((-ord) as u16) | 0x8000
        } else {
            ord as u16
        };
        let err = f64::from((bf16::from_bits(ab).to_f32() - reference.to_f32()).abs());
        assert!(
            (4.0e-5..1.0e-4).contains(&err),
            "100 ordinal ULP at this magnitude is {err:.3e}, not 4.8e-5..9.1e-5"
        );
        // What SHIPPED did: a fixed 2^-20 of the block RMS = 2.27e-5, BELOW the
        // accumulation noise it was supposed to forgive. This is the red cell.
        assert!(
            err > ROUND6_FIXED_FLOOR * HEAD_RMS,
            "round 6's fixed floor must be the thing that rejected this element"
        );
        // What the K-aware, per-row floor does: 8.1e-4, 9x above the worst
        // error the receipt exhibits.
        assert!(
            within_m16_tc_budget(ab, rb, K_HEAD, HEAD_RMS),
            "a logit cancelled to 4.9e-6 of its row has no relative accuracy to grade"
        );
    }
}

/// WHY the fixed constant was wrong: it does not scale with the reduction, and
/// the noise does. The SAME measurement at three depths shows the two orders
/// diverging while `2^-20 * row_rms` stands still — so there is always a K at
/// which a fixed floor sits under the noise, and `2^-20` (which is `u32 * 16`)
/// is the right size only for a reduction of a few hundred terms.
///
/// This is the mechanism behind the LM head's red cell, stated as an inequality
/// rather than as a story. The host model is QUIETER than the part (its
/// `rel_rms` at the real head shape is 3.3e-5 against the H100's 1.107e-4), so
/// the crossing happens at a larger K here than it did on the part at K=5120.
/// The DIRECTION is the claim; the crossing point is hardware.
#[test]
fn the_fixed_floor_does_not_track_the_reduction_depth_and_the_new_one_does() {
    let mut rng = Rng(0x0927_167C_2026);
    let mut previous = 0.0_f64;
    for k in [4096_usize, 65536, 1_048_576] {
        let a: Vec<f32> = (0..k).map(|_| rng.bf16()).collect();
        let row_rms = row_rms_of(&a);
        let gap = order_gap_rms(&mut rng, &a);
        assert!(
            gap > previous,
            "k={k}: the order gap must grow with the reduction depth, \
             {gap:.3e} <= {previous:.3e}"
        );
        previous = gap;
        let floor = m16_tc_acc_floor(k, row_rms);
        assert!(
            floor > 50.0 * gap,
            "k={k}: the K-aware floor {floor:.3e} must keep a real margin over the \
             measured noise {gap:.3e}"
        );
        let fixed = ROUND6_FIXED_FLOOR * row_rms;
        if k <= 4096 {
            // Shallow: the fixed constant sat comfortably ABOVE the noise, which
            // is why round 6 could get away with it on its own kernel pair.
            assert!(
                gap < fixed,
                "k={k}: {gap:.3e} vs the fixed floor {fixed:.3e}"
            );
        } else {
            // Deep: the fixed constant is BELOW the noise it exists to forgive.
            // That is the LM head's failure mode, on the host, with no GPU and
            // no defect anywhere in the index math.
            assert!(
                gap > fixed,
                "k={k}: the fixed floor {fixed:.3e} was supposed to be under the \
                 noise {gap:.3e} here"
            );
        }
    }
}

/// THE PER-ROW SCALE. A logit block whose rows have very different norms — the
/// "a few large rows" case the LM head actually serves, where one token's
/// activation is far hotter than another's — is the case a BLOCK-wide RMS gets
/// wrong in both directions at once: it under-scales the hot row's floor and
/// over-scales every quiet row's.
#[test]
fn the_floor_follows_the_row_and_not_the_block() {
    let mut rng = Rng(0x0927_1670_0009);
    let quiet: Vec<f32> = (0..K_HEAD).map(|_| rng.bf16() * 0.0625).collect();
    let hot: Vec<f32> = (0..K_HEAD).map(|_| rng.bf16()).collect();
    let (quiet_rms, hot_rms) = (row_rms_of(&quiet), row_rms_of(&hot));
    assert!(
        hot_rms / quiet_rms > 8.0,
        "the fixture must actually have a hot row"
    );
    let quiet_gap = order_gap_rms(&mut rng, &quiet);
    let hot_gap = order_gap_rms(&mut rng, &hot);

    // THE POINT: the noise itself scales with the ROW, so the floor must too.
    assert!(
        hot_gap > 4.0 * quiet_gap,
        "the hot row's own accumulation noise {hot_gap:.3e} must dwarf the quiet \
         row's {quiet_gap:.3e}"
    );
    for (label, gap, rms) in [("quiet", quiet_gap, quiet_rms), ("hot", hot_gap, hot_rms)] {
        assert!(
            gap < m16_tc_acc_floor(K_HEAD, rms),
            "{label}: the row's own floor must cover the row's own noise"
        );
    }

    // ...and a block RMS over one quiet and one hot row is dominated by the hot
    // one, so grading the quiet row at the block scale hands it >8x the slack it
    // has earned. That slack is what a real head's quiet tokens would collect.
    let block_rms = ((quiet_rms * quiet_rms + hot_rms * hot_rms) / 2.0).sqrt();
    assert!(
        m16_tc_acc_floor(K_HEAD, block_rms) > 8.0 * m16_tc_acc_floor(K_HEAD, quiet_rms),
        "a block RMS dominated by the hot row over-forgives the quiet one"
    );
}

#[test]
fn the_k_aware_floor_still_rejects_every_structural_error() {
    let floor = m16_tc_acc_floor(K_HEAD, HEAD_RMS);
    assert!(
        (8.0e-4..8.3e-4).contains(&floor),
        "the head's floor is 8.1e-4; got {floor:.4e}"
    );
    // 1/29,000 of the block's own RMS: four orders below any misplaced output.
    assert!(HEAD_RMS / floor > 29_000.0, "the floor must stay tiny");
    for (label, r, a) in [
        ("a misplaced output row", 40.0_f32, -12.0_f32),
        ("three BF16 quanta at the top of the range", 96.0, 97.5),
        (
            "a cancelled logit moved by a matrix-scale error",
            1.0e-4,
            0.5,
        ),
        (
            "the microtest's three-ordinal mutation on |value| > 1",
            1.0,
            1.023_437_5,
        ),
    ] {
        let (rb, ab) = (bf16::from_f32(r).to_bits(), bf16::from_f32(a).to_bits());
        assert!(
            !within_m16_tc_budget(ab, rb, K_HEAD, HEAD_RMS),
            "{label}: ref={r} actual={a} must still fail the budget"
        );
    }
    // The margin the constant buys, as a number rather than a comment: the
    // weakest negative control above is 28x the floor.
    assert!(
        (1.0 - 1.023_437_5_f64).abs() > 28.0 * floor,
        "the three-ordinal control must keep a real margin over the floor"
    );
    assert!(
        (M16_TC_ACC_FLOOR_MARGIN - 8.0).abs() < 1e-12,
        "the margin above is stated for M16_TC_ACC_FLOOR_MARGIN = 8"
    );
}
