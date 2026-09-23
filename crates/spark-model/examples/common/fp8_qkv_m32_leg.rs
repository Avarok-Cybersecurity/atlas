// SPDX-License-Identifier: AGPL-3.0-only

//! The 17..=32-row leg of `native_fp8_qkv_batch_microtest.rs` (G18 lever A):
//! `w8a16_gemm_pipelined_m32_strided` against the per-row scalar loop.
//!
//! The bar here is a TOLERANCE, not bit-equality, and that is the point of
//! keeping it in its own file: the GEMV legs of the parent are byte-exact by
//! contract, while the 32-row M-tile twin reduces K on the tensor core
//! (m16n8k16 sums 16 products in its own order before the FP32 accumulator
//! sees them), so it is held to the contract every tensor-core decode tier
//! is held to — ONE predicate, `layers::dense_ffn::m16_tc::oracle`, shared
//! with the `m16` receipts and their host simulations: within 2 ordinal BF16
//! ULP of the scalar OR under the K-scaled accumulation floor, plus a block
//! `rel_rms` gate. What stays byte-exact is everything the kernel must NOT
//! touch: the rows past M, the other projections' slots inside the strided
//! row, and the guard bands — the parent's `check_untouched`.
//!
//! Test-only harness code; no serving path runs any of it.

use anyhow::{Result, ensure};
use spark_model::layers::dense_ffn::m16_tc::oracle::{M16_TC_MAX_ULP, compare_m16_tc_block};

use super::{GUARD, check_untouched, values};

/// Block-level relative-RMS gate, the value the `m16` oracle uses
/// (`examples/common/m16_tc_compare.rs`). Per-element budget: the lib.
const REL_RMS_GATE: f64 = 1e-3;

/// Worst-case numbers over the live extents of one case, for the receipt.
#[derive(Debug, Default, Clone, Copy)]
pub(crate) struct M32Verdict {
    pub(crate) max_ulp: i32,
    pub(crate) over_ulp_only: usize,
    pub(crate) rel_rms: f64,
    pub(crate) max_abs: f64,
}

/// Grade one case: every byte outside `live` must equal the sentinel on both
/// runs (the strided-write contract, byte-exact), and every live extent — one
/// contiguous run of BF16 outputs per row, all reduced over `k` — must pass
/// the tensor-core budget against the scalar baseline.
pub(crate) fn check_m32(
    observed: &[u8],
    baseline: &[u8],
    sentinel: &[u8],
    live: &[(usize, usize)],
    k: usize,
) -> Result<M32Verdict> {
    check_untouched(observed, baseline, sentinel, live)?;
    let mut worst = M32Verdict::default();
    for &(start, len) in live {
        let a = &observed[GUARD + start..GUARD + start + len];
        let b = &baseline[GUARD + start..GUARD + start + len];
        ensure!(
            values(a)
                .iter()
                .chain(values(b).iter())
                .all(|x| x.is_finite()),
            "nonfinite projection output"
        );
        let d = compare_m16_tc_block(a, b, len / 2, k);
        if let Some(o) = d.over_budget.first() {
            anyhow::bail!(
                "M32 tile output exceeds the {M16_TC_MAX_ULP}-ULP / accumulation-floor \
                 budget: {} element(s), first at col {} (scalar {:e}, tile {:e}, {} ULP, \
                 row_rms {:.4})",
                d.over_budget.len(),
                o.col,
                o.reference,
                o.actual,
                o.ulp,
                o.row_rms
            );
        }
        ensure!(
            d.rel_rms <= REL_RMS_GATE,
            "M32 tile rel_rms {:.3e} exceeds the {REL_RMS_GATE:.0e} block gate",
            d.rel_rms
        );
        worst.max_ulp = worst.max_ulp.max(d.max_ulp);
        worst.over_ulp_only += d.over_ulp_only;
        worst.rel_rms = worst.rel_rms.max(d.rel_rms);
        worst.max_abs = worst.max_abs.max(d.max_abs);
    }
    Ok(worst)
}

/// Known-bad controls for [`check_m32`], run once so a green M32 leg cannot
/// be a vacuous one. A tolerance oracle has one more way to be vacuous than
/// a byte oracle — a budget wide enough to admit a wrong answer — so the
/// first control is a NUMERIC error well outside the budget (an exponent
/// bit flip: ~2x the value, hundreds of ordinal ULP and far above the floor
/// for any element of ordinary magnitude), not a one-ULP nudge.
pub(crate) fn known_bad_controls(
    baseline: &[u8],
    sentinel: &[u8],
    live: &[(usize, usize)],
    gap_byte: usize,
    k: usize,
) -> Result<()> {
    // Pick a live element the flip cannot land near zero: the largest |value|
    // in the first extent.
    let (start, len) = live[0];
    let row = &baseline[GUARD + start..GUARD + start + len];
    let (idx, _) = values(row)
        .iter()
        .enumerate()
        .fold((0usize, 0.0f64), |acc, (i, v)| {
            if v.abs() > acc.1 { (i, v.abs()) } else { acc }
        });
    for mutation in ["exponent-bit", "gap", "guard", "nonfinite"] {
        let mut bad = baseline.to_vec();
        match mutation {
            "exponent-bit" => bad[GUARD + start + idx * 2 + 1] ^= 0x40,
            "gap" => bad[GUARD + gap_byte] ^= 1,
            "guard" => bad[0] ^= 1,
            _ => bad[GUARD + start..GUARD + start + 2].copy_from_slice(&0x7fc0_u16.to_le_bytes()),
        }
        let err = check_m32(&bad, baseline, sentinel, live, k)
            .err()
            .ok_or_else(|| {
                anyhow::anyhow!("known-bad `{mutation}` was admitted by the M32 oracle")
            })?;
        println!("KNOWN_BAD m32 {mutation}: refused: {err}");
    }
    Ok(())
}
