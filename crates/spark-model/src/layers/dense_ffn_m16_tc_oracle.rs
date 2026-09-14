// SPDX-License-Identifier: AGPL-3.0-only

//! The numerics contract for the tensor-core decode tiers (`w8a16_gemm_m16`,
//! #927; `dense_gemm_m16_bf16`, #927/#928) — ONE comparison, evaluated by the
//! GPU oracles (`examples/native_fp8_ffn_m16_tc_microtest.rs`,
//! `examples/native_bf16_lm_head_m16_microtest.rs`) and by the host simulations
//! (`dense_ffn_m16_tc_m32_tests.rs`, `ops/dense_gemm_m16_bf16_tests.rs`,
//! `ops/dense_gemm_m16_bf16_floor_tests.rs`), so a
//! receipt and a unit test cannot drift into grading different things.
//!
//! The kernel REASSOCIATES the K reduction relative to the scalar GEMV (an
//! m16n8k16 MMA reduces 16 K-products in the tensor core's own order before
//! they reach the FP32 accumulator), so the contract is a tolerance and always
//! was. What changed in round 6 is WHICH tolerance; what changed in round 9 is
//! how the tolerance's absolute half is SCALED — see [`m16_tc_acc_floor`].

/// BF16 ordinal-ULP budget this tier is held to, against the scalar GEMV.
/// Unchanged since #927.
pub const M16_TC_MAX_ULP: i32 = 2;

/// FP32 unit roundoff, `2^-24`. Every rounding the accumulator performs is a
/// multiple of this times the magnitude being rounded, so it is the only
/// constant in [`m16_tc_acc_floor`] that is not a judgement call.
pub const F32_UNIT_ROUNDOFF: f64 = 5.960_464_477_539_063e-8;

/// Margin over the `u * sqrt(K)` accumulation scale that [`m16_tc_acc_floor`]
/// admits.
///
/// 🔴 CALIBRATED, NOT DERIVED, and the round-9 fix turns on it. The SHAPE of
/// the floor (`u * sqrt(K) * row_rms`) follows from the arithmetic; the O(1)
/// constant in front of it does not, because the m16n8k16's internal 16-product
/// order is unspecified hardware and the H100 turns out to be noisier than any
/// host model of it. So the constant is fixed by a receipt at both ends:
///
/// * **Above the noise.** H100 round 9 (`native_bf16_lm_head_m16_microtest`,
///   1xH100, 2026-09-11, the real `[N=248077, K=5120]` BF16 head) rejected a
///   worst cell of `reference=-1.173019409e-4` at `max_ulp=100` against a block
///   RMS of 23.80. One hundred ordinal BF16 ULP at that magnitude is an
///   absolute error of 4.8e-5..9.1e-5. `8 * u * sqrt(5120) * 23.80` =
///   **8.1e-4**, i.e. 9x above the worst error the receipt exhibits. (A host
///   run of both reduction orders at the real shape reproduces that reference
///   value bit-for-bit at row 7, column 228,041 — an INTERIOR column of CTA
///   7,126 of 7,753, nowhere near the 13-column tail. The reference arm is
///   `dense_gemv_bf16`, which the host reproduces exactly, so the element's
///   identity carries even though its ULP distance does not.)
/// * **Below anything structural.** The same floor is 1/29,000 of that block's
///   own RMS, and 28x below the microtest's weakest negative control (a
///   three-ordinal mutation on a `|value| > 1`, which moves it by 0.0234). A
///   row, pitch, offset or tail-mask defect misplaces whole outputs and lands
///   errors of order the RMS itself — four orders above the floor.
///
/// A future widening has to argue with assertions, not with this comment:
/// `the_accumulation_floor_still_rejects_a_structural_error` in
/// `dense_ffn_m16_tc_m32_tests.rs`, and
/// `the_round9_lm_head_outlier_is_a_cancelled_logit` plus
/// `the_k_aware_floor_still_rejects_every_structural_error` in
/// `ops/dense_gemm_m16_bf16_floor_tests.rs`.
pub const M16_TC_ACC_FLOOR_MARGIN: f64 = 8.0;

/// Absolute error below which an ordinal-ULP budget says nothing, for one
/// output of a length-`k` FP32 reduction whose row has RMS `row_rms`.
///
/// 🔴 THIS IS THE ROUND-9 `lm_head` FIX, and it replaces round 6's fixed
/// `2^-20 * block_rms`. Two things were wrong with that constant:
///
/// 1. **It did not scale with the reduction.** An ordinal BF16 ULP is a
///    RELATIVE unit, and an output that has catastrophically cancelled has no
///    relative accuracy left to measure — but the absolute noise it is measured
///    against is a property of the REDUCTION, not of the output. Two FP32
///    orders over the same `k` terms differ by a random walk of `k` roundings,
///    i.e. `~u * sqrt(k)` times the scale of the terms; the terms' scale is
///    `row_rms / sqrt(k)` when they are zero-mean and independent, so the
///    difference scales as `u * sqrt(k) * row_rms`. `2^-20` is `u * 16`, so it
///    happens to be the right size only near `k = 256`.
/// 2. **It was fitted to ONE kernel pair.** Round 6 chose `2^-20` as "~3.5x
///    above the worst error observed" on `w8a16_gemm_m16` vs `w8a16_gemv`,
///    where BOTH arms fold a 128-wide FP8 scale block onto an outer FP32
///    accumulator — a two-level reduction. `dense_gemm_m16_bf16` has no block
///    scale, so its accumulator is ONE uninterrupted 320-step chain at
///    K=5120 and is several times noisier. Round 9 measured the consequence:
///    `over_budget` 10/16/25/37 at M=5/8/13/16 on the LM head — exactly linear
///    in M, i.e. ~2 rejected elements per row out of 248,077 columns, the
///    signature of a per-element statistical tail and not of a defect.
///
/// `row_rms` is the RMS of the REFERENCE row, not of the whole block: an
/// element's accumulation noise is proportional to `||a_m||`, the norm of the
/// activation row that produced it, and the row's output RMS is the only
/// observable that tracks `||a_m||`. A block-wide RMS under-scales the floor
/// for a hot row and over-scales it for a quiet one; the LM head's rows are
/// different tokens, so that is a real difference in service even though the
/// microtest's uniform fixture cannot show it.
///
/// The row's MAX `|ref|` was considered and rejected: it is a single order
/// statistic, so it moves with the block width and with one outlier, whereas
/// the reduction's noise depends on the row's L2 norm. RMS is that norm.
pub fn m16_tc_acc_floor(k: usize, row_rms: f64) -> f64 {
    M16_TC_ACC_FLOOR_MARGIN * F32_UNIT_ROUNDOFF * (k as f64).sqrt() * row_rms
}

/// BF16 bits -> a monotone integer, so `|ord(a) - ord(b)|` is the ULP distance
/// and +0/-0 are the same point.
pub fn bf16_ord(bits: u16) -> i32 {
    if bits & 0x8000 != 0 {
        -((bits & 0x7FFF) as i32)
    } else {
        bits as i32
    }
}

/// The tier's numerics contract, as ONE predicate every oracle and host
/// simulation evaluates, so they cannot drift.
///
/// An element passes if EITHER it is within [`M16_TC_MAX_ULP`] ordinal BF16 ULP
/// of the reference, OR its absolute error is under [`m16_tc_acc_floor`] for
/// the reduction depth `k` and the reference ROW's RMS.
///
/// `k` is the reduction depth the kernel actually ran, not the output width —
/// passing `n` here would make the floor grow with the vocabulary, which is the
/// one axis it must not depend on.
pub fn within_m16_tc_budget(actual_bits: u16, reference_bits: u16, k: usize, row_rms: f64) -> bool {
    if (bf16_ord(actual_bits) - bf16_ord(reference_bits)).abs() <= M16_TC_MAX_ULP {
        return true;
    }
    let a = f64::from(half::bf16::from_bits(actual_bits).to_f32());
    let b = f64::from(half::bf16::from_bits(reference_bits).to_f32());
    (a - b).abs() <= m16_tc_acc_floor(k, row_rms)
}

/// One element the comparison rejected, with everything needed to say WHERE it
/// is and WHY it failed.
///
/// Round 6 reported `over_budget 5` and nothing else, so the five elements
/// could not be located without re-running the H100 — which is why the
/// diagnosis took a host simulation. The report now names them, and round 9
/// added `row_rms` because the floor is per-row: without it the printed
/// `|ref|/rms` cannot be checked against the floor that actually rejected the
/// element.
#[derive(Debug, Clone, Copy)]
pub struct M16TcOutlier {
    pub row: usize,
    pub col: usize,
    pub reference: f32,
    pub actual: f32,
    pub ulp: i32,
    /// RMS of the reference ROW this element sits in — the scale
    /// [`m16_tc_acc_floor`] was evaluated at.
    pub row_rms: f64,
}

/// The result of comparing an `m x n` BF16 block against the scalar reference.
#[derive(Debug, Clone, Default)]
pub struct M16TcDiff {
    /// Largest ordinal BF16 ULP distance, sign flips excluded and counted.
    pub max_ulp: i32,
    /// Elements the FULL criterion rejected — ordinal budget AND the
    /// accumulation floor. This is the number that gates a cell.
    pub over_budget: Vec<M16TcOutlier>,
    /// Elements the ordinal budget alone would have rejected. Reported so a
    /// round-6-style cell can be read at a glance as "cancellation tail" rather
    /// than investigated as a defect.
    pub over_ulp_only: usize,
    pub sign_flips: usize,
    pub max_abs: f64,
    pub rel_rms: f64,
    /// RMS of the whole REFERENCE block. The floor is per-ROW, but this is the
    /// number the receipts print and the one that makes an outlier's magnitude
    /// interpretable at a glance.
    pub rms: f64,
    /// Per-row reference RMS, in row order — the scales the floor was actually
    /// evaluated at. Printed by the oracles next to each outlier.
    pub row_rms: Vec<f64>,
}

/// Magnitude below which a SIGN change carries no information: one ULP across
/// zero is a full sign flip, so those are counted separately rather than
/// graded. Unchanged from #927.
pub const M16_TC_SIGN_FLIP_BAND: f64 = 0.05;

/// RMS of a BF16 slice, in the f64 the floor is expressed in.
fn bf16_rms(block: &[u8]) -> f64 {
    let count = block.len() / 2;
    if count == 0 {
        return 0.0;
    }
    let sum: f64 = block
        .chunks_exact(2)
        .map(|b| {
            let v = f64::from(half::bf16::from_bits(u16::from_le_bytes([b[0], b[1]])).to_f32());
            v * v
        })
        .sum();
    (sum / count as f64).sqrt()
}

/// Compare an `m x n` BF16 block against the scalar reference under the tier's
/// contract. Both slices are `m * n` little-endian BF16 elements; `k` is the
/// reduction depth the kernel ran.
///
/// Two passes, because the absolute floor is expressed against the reference
/// row RMS and no row's RMS is known until it has been walked once.
pub fn compare_m16_tc_block(actual: &[u8], reference: &[u8], n: usize, k: usize) -> M16TcDiff {
    let bits = |b: &[u8]| u16::from_le_bytes([b[0], b[1]]);
    let val = |b: u16| f64::from(half::bf16::from_bits(b).to_f32());
    let row_bytes = n * 2;
    let row_rms: Vec<f64> = if row_bytes == 0 {
        Vec::new()
    } else {
        reference.chunks(row_bytes).map(bf16_rms).collect()
    };
    let mut d = M16TcDiff {
        rms: bf16_rms(reference),
        row_rms,
        ..Default::default()
    };
    let (mut err_sq, mut ref_sq) = (0.0_f64, 0.0_f64);
    for (i, (a, b)) in actual
        .chunks_exact(2)
        .zip(reference.chunks_exact(2))
        .enumerate()
    {
        let (ab, bb) = (bits(a), bits(b));
        let (av, bv) = (val(ab), val(bb));
        err_sq += (av - bv) * (av - bv);
        ref_sq += bv * bv;
        d.max_abs = d.max_abs.max((av - bv).abs());
        let ulp = (bf16_ord(ab) - bf16_ord(bb)).abs();
        if av.signum() != bv.signum() && bv.abs() < M16_TC_SIGN_FLIP_BAND {
            d.sign_flips += 1;
            continue;
        }
        d.max_ulp = d.max_ulp.max(ulp);
        if ulp > M16_TC_MAX_ULP {
            d.over_ulp_only += 1;
        }
        let row = if n == 0 { 0 } else { i / n };
        let scale = d.row_rms.get(row).copied().unwrap_or(d.rms);
        if !within_m16_tc_budget(ab, bb, k, scale) {
            d.over_budget.push(M16TcOutlier {
                row,
                col: if n == 0 { 0 } else { i % n },
                reference: bv as f32,
                actual: av as f32,
                ulp,
                row_rms: scale,
            });
        }
    }
    d.rel_rms = if ref_sq > 0.0 {
        (err_sq / ref_sq).sqrt()
    } else {
        err_sq.sqrt()
    };
    d
}
