// SPDX-License-Identifier: AGPL-3.0-only
//! Prefill-MoE sub-leg 4: the FUSED tier ABOVE the legacy 128-row cap.
//!
//! The row cap (`Exl3MoePrefillScratch::rows_per_expert`; serving default
//! 1024, legacy 128) is the fused `exl3_moe` kernel's temp-slab height. The
//! kernel uses it ONLY as the per-group slab stride and the
//! `token_count > max_tokens_per_expert` skip predicate — an expert's 16-row
//! tile walk and its `output_slots` plain-store epilogue depend on
//! `token_count` alone. That is a code-reading argument; this sub-leg is the
//! evidence. The SAME T=192 skewed batch that sub-leg 3 ran at cap 128
//! (expert 0 at ~460 rows on the OVERFLOW tier) is re-run on slabs sized at
//!  * `WIDE_CAP` = 512 — 460 <= 512 < S = 768: the host-sync tier keeps
//!    EVERY expert fused (asserted: overflow_experts == 0 and num_active ==
//!    the host count of non-empty experts), expert 0 walking ~29
//!    sixteen-row passes through the fused kernel; and
//!  * cap = S = 768 — the S <= cap NO-SYNC shortcut (asserted: num_active ==
//!    -1, overflow 0) with the same ~460-row expert inside it.
//! Both gate against the f64 reference at the SAME rel 8e-3 / z 8e-2 the
//! overflow arm uses — a looser gate would hide a tier-dependent seam.
//!
//! The "one-time fp32-order change" claim is then QUANTIFIED, not asserted:
//! each wide run's bf16 output bits are diffed against sub-leg 3's cap-128
//! bits on identical inputs, split by token class —
//!  * tokens with NO slot on expert 0: every expert they touch was fused in
//!    both runs (same kernel, same grid, same tile walk, same per-slot
//!    store, same fixed-order reduce) -> ASSERTED bit-identical;
//!  * tokens with >= 1 slot on expert 0: the class whose accumulation order
//!    changed (overflow exl3_gemm -> MoE tile) -> mismatch fraction,
//!    max |delta| and max bf16-ulp distance are printed for the record.
//! The two wide runs are diffed against each other as well: when the
//! shortcut's default grid (8 SMs x C groups) equals the host-sync tier's
//! narrowed grid (GB10: C = 6 over 48 SMs -> 8 SMs/group either way) they
//! are ASSERTED bit-identical; on any other grid the count is INFO only.

use anyhow::{Result, ensure};
use spark_model::layers::ops::{Exl3MoeOverflowCtx, Exl3MoeProj};

use crate::legs_moe_prefill::{
    E, H, MOE_MAX_Z, MOE_REL_RMS, ROWS_PER_EXPERT, T_MAX, TOP_K, alloc_slabs_with_cap, run_native,
};
use crate::util::{Ctx, gate_leg};

/// The host-sync wide cap: above the skew count (~460), below S = 768 so the
/// host D2H still runs and the returned `num_active` is the real expert count.
pub const WIDE_CAP: usize = 512;

/// Sub-leg 3's skewed batch — inputs, routing, its cap-128 output bits and
/// the f64 reference — shared with the wide runs so every bit diff below is
/// over IDENTICAL inputs.
pub struct SkewCase<'a> {
    pub input_bf16: &'a [u16],
    pub ids: &'a [u32],
    pub probs: &'a [f32],
    pub bits_128: &'a [u16],
    pub y64: &'a [f64],
}

struct ClassDiff {
    clean_mismatch: usize,
    clean_total: usize,
    hot_mismatch: usize,
    hot_total: usize,
    hot_max_delta: f64,
    hot_max_ulp: u32,
}

/// Monotonic integer key over bf16 bit patterns (both zeros map to the same
/// key), so |ka - kb| is the ulp distance between two finite values.
fn bf16_key(bits: u16) -> i64 {
    if bits & 0x8000 != 0 {
        0x8000 - (bits & 0x7fff) as i64
    } else {
        0x8000 + bits as i64
    }
}

fn diff_by_class(a: &[u16], b: &[u16], touches_hot: &[bool]) -> ClassDiff {
    let mut d = ClassDiff {
        clean_mismatch: 0,
        clean_total: 0,
        hot_mismatch: 0,
        hot_total: 0,
        hot_max_delta: 0.0,
        hot_max_ulp: 0,
    };
    for (tok, &hot) in touches_hot.iter().enumerate() {
        let ra = &a[tok * H..(tok + 1) * H];
        let rb = &b[tok * H..(tok + 1) * H];
        let mism = ra.iter().zip(rb).filter(|(x, y)| x != y).count();
        if hot {
            d.hot_total += H;
            d.hot_mismatch += mism;
            for (&x, &y) in ra.iter().zip(rb) {
                if x != y {
                    let dv = (half::bf16::from_bits(x).to_f64()
                        - half::bf16::from_bits(y).to_f64())
                    .abs();
                    d.hot_max_delta = d.hot_max_delta.max(dv);
                    d.hot_max_ulp = d
                        .hot_max_ulp
                        .max((bf16_key(x) - bf16_key(y)).unsigned_abs() as u32);
                }
            }
        } else {
            d.clean_total += H;
            d.clean_mismatch += mism;
        }
    }
    d
}

pub fn subleg_wide(
    ctx: &Ctx,
    full: &[Exl3MoeProj; 3],
    ov: &Exl3MoeOverflowCtx,
    case: &SkewCase,
) -> Result<bool> {
    let g = ctx.g;
    let t = T_MAX;
    let s = t * TOP_K;
    let mut ok = true;
    ensure!(case.ids.len() == s && case.bits_128.len() == t * H && case.y64.len() == t * H);

    let mut counts = [0usize; E];
    for &e in case.ids {
        counts[e as usize] += 1;
    }
    let hot = counts[0];
    let expect_active = counts.iter().filter(|&&c| c > 0).count() as i64;
    ensure!(
        hot > ROWS_PER_EXPERT && hot <= WIDE_CAP,
        "wide sub-leg needs {ROWS_PER_EXPERT} < expert0 rows ({hot}) <= {WIDE_CAP}"
    );
    let touches_hot: Vec<bool> = (0..t)
        .map(|tok| case.ids[tok * TOP_K..(tok + 1) * TOP_K].contains(&0))
        .collect();
    let n_clean = touches_hot.iter().filter(|&&h| !h).count();
    println!(
        "moe-prefill WIDE T={t} skew: expert0 rows={hot}, non-empty experts={expect_active}, \
         tokens touching expert0={} / clean={n_clean}",
        t - n_clean
    );

    let mut runs: Vec<(usize, Vec<u16>)> = Vec::new();
    for cap in [WIDE_CAP, s] {
        let sl = alloc_slabs_with_cap(ctx, cap)?;
        let res = run_native(
            ctx,
            &sl,
            full,
            ov,
            case.input_bf16,
            case.ids,
            case.probs,
            t,
            0,
            E,
        );
        for p in sl.owned.iter() {
            g.free(*p).ok();
        }
        let (y, bits, num_active, n_ov) = res?;
        let (label, want_active) = if cap < s {
            ("host-sync fused", expect_active)
        } else {
            ("no-sync shortcut", -1)
        };
        ok &= gate_leg(
            &format!(
                "moe-prefill WIDE cap={cap} {label} T={t} (expert0 rows={hot} > {ROWS_PER_EXPERT})"
            ),
            &y,
            case.y64,
            MOE_REL_RMS,
            MOE_MAX_Z,
        );
        let tier_ok = n_ov == 0 && num_active == want_active;
        println!(
            "moe-prefill WIDE cap={cap} {label} tier stats (num_active={num_active} want \
             {want_active}, overflow_experts={n_ov} want 0) = {tier_ok}"
        );
        ok &= tier_ok;

        let d = diff_by_class(&bits, case.bits_128, &touches_hot);
        let hot_frac = d.hot_mismatch as f64 / d.hot_total.max(1) as f64;
        println!(
            "moe-prefill WIDE cap={cap} vs cap={ROWS_PER_EXPERT} bf16 bits: clean-token \
             mismatches {}/{} ; expert0-token mismatch fraction {hot_frac:.4e} ({}/{}), \
             max|delta| {:.3e}, max bf16-ulp {}",
            d.clean_mismatch,
            d.clean_total,
            d.hot_mismatch,
            d.hot_total,
            d.hot_max_delta,
            d.hot_max_ulp
        );
        if n_clean > 0 {
            let clean_identical = d.clean_mismatch == 0;
            println!(
                "moe-prefill WIDE cap={cap} ASSERT clean tokens (all experts fused in BOTH \
                 runs) bit-identical to cap={ROWS_PER_EXPERT} = {clean_identical}"
            );
            ok &= clean_identical;
        } else {
            println!(
                "moe-prefill WIDE cap={cap}: no clean tokens in this draw — class assertion \
                 NOT evaluated"
            );
        }
        runs.push((cap, bits));
    }

    // Host-sync (narrowed grid) vs shortcut (default grid): identical grids
    // on GB10, so the two fused runs must agree bit for bit there.
    let c = (ctx.sms as usize / 8).clamp(1, 64);
    let default_groups = c.min(64);
    let narrowed_groups = default_groups.min(expect_active as usize);
    let narrowed_group_size = (ctx.sms as usize / narrowed_groups).min(32);
    let same_grid = narrowed_groups == default_groups && narrowed_group_size == 8;
    let (a, b) = (&runs[0], &runs[1]);
    let mism = a.1.iter().zip(&b.1).filter(|(x, y)| x != y).count();
    if same_grid {
        let identical = mism == 0;
        println!(
            "moe-prefill WIDE ASSERT cap={} host-sync vs cap={} shortcut bit-identical (same \
             grid {narrowed_groups}x{narrowed_group_size} SMs): mismatches={mism} = {identical}",
            a.0, b.0
        );
        ok &= identical;
    } else {
        println!(
            "moe-prefill WIDE INFO cap={} vs cap={}: grids differ ({narrowed_groups}x\
             {narrowed_group_size} vs {default_groups}x8 SMs) — mismatches={mism}, not gated",
            a.0, b.0
        );
    }
    Ok(ok)
}
