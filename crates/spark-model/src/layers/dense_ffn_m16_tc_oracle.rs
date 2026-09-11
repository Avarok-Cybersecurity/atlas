// SPDX-License-Identifier: AGPL-3.0-only

//! The numerics contract for the tensor-core W8A16 decode tier
//! (`w8a16_gemm_m16`, #927) — ONE comparison, evaluated by both the GPU oracle
//! (`examples/native_fp8_ffn_m16_tc_microtest.rs`) and the host simulation
//! (`dense_ffn_m16_tc_m32_tests.rs`), so a receipt and a unit test cannot drift
//! into grading different things.
//!
//! The kernel REASSOCIATES the K reduction relative to the scalar `w8a16_gemv`
//! (an m16n8k16 MMA reduces 16 K-products in the tensor core's own order before
//! they reach the FP32 accumulator), so the contract is a tolerance and always
//! was. What changed in round 6 is WHICH tolerance — see [`M16_TC_ACC_FLOOR`].

/// BF16 ordinal-ULP budget this tier is held to, against the scalar
/// `w8a16_gemv`. Unchanged since #927.
pub const M16_TC_MAX_ULP: i32 = 2;

/// Absolute error, as a fraction of the reference matrix's RMS, below which an
/// ordinal-ULP budget says nothing.
///
/// 🔴 THIS IS THE ROUND-6 `gate/up M=32` FIX. An ordinal BF16 ULP is a
/// RELATIVE unit, and a GEMM output that has catastrophically cancelled has no
/// relative accuracy left to measure: at K=5120 the FP32 accumulator's absolute
/// noise is ~eps32 * sqrt(K) * rms(term), which is a fixed fraction of the
/// MATRIX scale and completely independent of how small any one output came
/// out. Round 6's five red elements had |reference| between 5.7e-6 and 1.6e-4
/// against a reference RMS of 39.1 — they had cancelled to 1e-7..4e-6 of the
/// matrix scale — and their absolute errors were 3.8e-6..1.05e-5, i.e. 1e-7 to
/// 2.7e-7 of that RMS. Judged in ordinal ULP that reads as 28 ULP and a FAIL;
/// judged in absolute terms it is the accumulator's own floor.
///
/// 2^-20 of the RMS is ~3.5x above the worst error observed and ~13,500x BELOW
/// the BF16 quantum at the top of the same matrix (max_abs was 0.500), so the
/// floor can only ever forgive elements that have cancelled to under ~6e-5 of
/// the RMS. A real row/pitch/offset defect misplaces whole outputs and lands
/// errors of order the RMS itself — four to seven orders of magnitude above
/// this — so the gate keeps all of its teeth. `m16_tc_m32_tests.rs` pins both
/// directions.
pub const M16_TC_ACC_FLOOR: f64 = 9.536_743_164_062_5e-7;

/// BF16 bits -> a monotone integer, so `|ord(a) - ord(b)|` is the ULP distance
/// and +0/-0 are the same point.
pub fn bf16_ord(bits: u16) -> i32 {
    if bits & 0x8000 != 0 {
        -((bits & 0x7FFF) as i32)
    } else {
        bits as i32
    }
}

/// The tier's numerics contract, as ONE predicate both the GPU oracle
/// (`examples/native_fp8_ffn_m16_tc_microtest.rs`) and the host simulation
/// evaluate, so the two cannot drift.
///
/// An element passes if EITHER it is within [`M16_TC_MAX_ULP`] ordinal BF16 ULP
/// of the reference, OR its absolute error is under [`M16_TC_ACC_FLOOR`] of
/// `rms` — the reference matrix's RMS, which is the only scale at which the
/// FP32 accumulation floor is expressible. `rms` must be the RMS of the whole
/// compared block, not of one element.
pub fn within_m16_tc_budget(actual_bits: u16, reference_bits: u16, rms: f64) -> bool {
    if (bf16_ord(actual_bits) - bf16_ord(reference_bits)).abs() <= M16_TC_MAX_ULP {
        return true;
    }
    let a = f64::from(half::bf16::from_bits(actual_bits).to_f32());
    let b = f64::from(half::bf16::from_bits(reference_bits).to_f32());
    (a - b).abs() <= M16_TC_ACC_FLOOR * rms
}

/// One element the comparison rejected, with everything needed to say WHERE it
/// is and WHY it failed.
///
/// Round 6 reported `over_budget 5` and nothing else, so the five elements
/// could not be located without re-running the H100 — which is why the
/// diagnosis took a host simulation. The report now names them.
#[derive(Debug, Clone, Copy)]
pub struct M16TcOutlier {
    pub row: usize,
    pub col: usize,
    pub reference: f32,
    pub actual: f32,
    pub ulp: i32,
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
    /// RMS of the REFERENCE block — the scale the absolute floor is expressed
    /// in, and worth printing because it is what makes an outlier's magnitude
    /// interpretable.
    pub rms: f64,
}

/// Magnitude below which a SIGN change carries no information: one ULP across
/// zero is a full sign flip, so those are counted separately rather than
/// graded. Unchanged from #927.
pub const M16_TC_SIGN_FLIP_BAND: f64 = 0.05;

/// Compare an `m x n` BF16 block against the scalar reference under the tier's
/// contract. Both slices are `m * n` little-endian BF16 elements.
///
/// Two passes, because the absolute floor is expressed against the reference
/// RMS and the RMS is not known until the block has been walked once.
pub fn compare_m16_tc_block(actual: &[u8], reference: &[u8], n: usize) -> M16TcDiff {
    let bits = |b: &[u8]| u16::from_le_bytes([b[0], b[1]]);
    let val = |b: u16| f64::from(half::bf16::from_bits(b).to_f32());
    let rms = {
        let sum: f64 = reference
            .chunks_exact(2)
            .map(|b| {
                let v = val(bits(b));
                v * v
            })
            .sum();
        let count = reference.len() / 2;
        if count == 0 {
            0.0
        } else {
            (sum / count as f64).sqrt()
        }
    };
    let mut d = M16TcDiff {
        rms,
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
        if !within_m16_tc_budget(ab, bb, rms) {
            d.over_budget.push(M16TcOutlier {
                row: i / n,
                col: i % n,
                reference: bv as f32,
                actual: av as f32,
                ulp,
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
