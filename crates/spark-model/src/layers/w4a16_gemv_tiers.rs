// SPDX-License-Identifier: AGPL-3.0-only

//! SSOT for the narrow `w4a16_gemv_batch{M}` tier family (M = 4..8).
//!
//! # Why this module exists
//!
//! `w4a16_gemv_batchm_impl<MAX_M>` sizes `acc[]`, `s_vl[]` and `smem[]` by
//! `MAX_M`, and because its row loop is `#pragma unroll`ed, `MAX_M` also sizes
//! the CODE: at 80 static SASS instructions per row on sm_121f, the MAX_M=8
//! tier is 760 instructions against MAX_M=4's 440. The `t >= M` guard skips a
//! dead row's WORK at run time but not its instructions, and the template is
//! issue-bound (not DRAM-bound) at these M — so running M=5 on the MAX_M=8
//! tier pays for three rows that are not there.
//!
//! Measured on the real 27B qkv/o shape (N=5120 K=5120, cold-cycled weights,
//! 273 GB/s peak) before the exact-M tiers existed:
//!
//! | tier   | M | time     | eff BW     | % peak |
//! |--------|---|----------|------------|--------|
//! | batch4 | 4 |  70.5 us | 209.2 GB/s | 76.6%  |
//! | batch8 | 4 |  74.8 us | 197.3 GB/s | 72.3%  |
//! | batch8 | 5 |  89.0 us | 165.7 GB/s | 60.7%  |
//! | batch8 | 6 |  91.5 us | 161.2 GB/s | 59.0%  |
//! | batch8 | 8 | 106.4 us | 138.5 GB/s | 50.7%  |
//!
//! The `batch8 @ M=4` row is the argument: same rows, same weight stream,
//! +6.1% for nothing. It is NOT occupancy — batch4 lands on 48 registers /
//! 5 CTA per SM with no `__launch_bounds__` at all, which is exactly what the
//! pragma pins batch8 to.
//!
//! # Why a shared table instead of a `match` per call site
//!
//! Before this module, FIVE structs each carried a `w4a16_gemv_batch4_k` /
//! `w4a16_gemv_batch8_k` pair and each re-derived `1..=4 => batch4,
//! 5..=8 => batch8` inline (dense_ffn, qwen3_ssm x2, qwen3_attention, mtp_head,
//! model). Adding three tiers would have meant five more copies of a widening
//! decision. The decision now lives here once, as a PURE function over which
//! tiers the loaded target actually resolved.
//!
//! # Kill switch
//!
//! `ATLAS_NO_GEMV_EXACT_M_TIERS=1` (presence-checked per the house convention;
//! `=0` is NOT off) hides widths 5/6/7 from the decision, restoring exactly the
//! batch4/batch8 dispatch that shipped before them. It does not unload the
//! kernels — it only removes them from selection, so an A/B needs no rebuild.

use spark_runtime::gpu::{GpuBackend, KernelHandle};

/// Tier widths in this family, narrowest first. Parallel to the `handles`
/// field of [`W4a16BatchmTiers`] and to the `present` array of
/// [`select_tier`].
///
/// Deliberately stops at 8. `w4a16_gemv_batch16`/`_batch32` exist but are NOT
/// in this table: only two call sites can use them, they are wide-tier trades
/// with their own (measured) `__launch_bounds__`-free codegen, and folding them
/// in here would silently widen every site that today caps at 8.
pub const W4A16_BATCHM_WIDTHS: [u32; 5] = [4, 5, 6, 7, 8];

/// Index into [`W4A16_BATCHM_WIDTHS`] of the first EXACT-M tier (width 5).
/// Widths below this index shipped before the exact-M tiers and are never
/// hidden by the kill switch.
const FIRST_EXACT_M: usize = 1;

/// Widths hidden by `ATLAS_NO_GEMV_EXACT_M_TIERS=1`: the tiers added by this
/// change. 8 is NOT hidden — it is the pre-existing M=5..8 tier.
const EXACT_M_LAST: usize = 3;

/// Are the exact-M tiers (5/6/7) allowed in the dispatch decision?
///
/// PRESENCE check, read once per process: this predicate sits on the decode
/// path and is consulted per projection launch.
pub fn exact_m_tiers_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("ATLAS_NO_GEMV_EXACT_M_TIERS").is_none())
}

/// PURE tier decision: index into [`W4A16_BATCHM_WIDTHS`] of the narrowest
/// tier that both COVERS `m` rows and is present in the loaded target, or
/// `None` when this family cannot serve `m` (caller falls back to the tile
/// GEMMs / the wide tiers).
///
/// `present[i]` is whether `W4A16_BATCHM_WIDTHS[i]` resolved. `exact_m` is
/// [`exact_m_tiers_enabled`], threaded in rather than read here so both
/// polarities are testable without touching process env or a latched
/// `OnceLock`.
///
/// Narrowest-that-covers, not exact-only: a target missing a tier must still
/// dispatch, and the next wider tier is bit-identical at the same `m` (same
/// template, `MAX_M`-independent per-row FMA chain) — only slower.
pub fn select_tier(
    m: u32,
    present: [bool; W4A16_BATCHM_WIDTHS.len()],
    exact_m: bool,
) -> Option<usize> {
    if m == 0 {
        return None;
    }
    W4A16_BATCHM_WIDTHS
        .iter()
        .enumerate()
        .find(|&(i, &w)| {
            w >= m && present[i] && (exact_m || !(FIRST_EXACT_M..=EXACT_M_LAST).contains(&i))
        })
        .map(|(i, _)| i)
}

/// Resolved handles for the narrow `w4a16_gemv_batch{M}` family.
///
/// A zero handle means "this target did not load that tier"; every consumer
/// must gate on `.0 != 0` exactly as it did with the individual fields, and
/// [`Self::kernel`] returns a zero handle rather than panicking when nothing
/// in the family covers `m`.
#[derive(Clone, Copy, Debug)]
pub struct W4a16BatchmTiers {
    /// Parallel to [`W4A16_BATCHM_WIDTHS`].
    handles: [KernelHandle; W4A16_BATCHM_WIDTHS.len()],
}

/// `KernelHandle` has no `Default`, so the "no NVFP4 kernels" state is spelled
/// out: an all-zero table, which every consumer already treats as "decline".
impl Default for W4a16BatchmTiers {
    fn default() -> Self {
        Self {
            handles: [KernelHandle(0); W4A16_BATCHM_WIDTHS.len()],
        }
    }
}

impl W4a16BatchmTiers {
    /// Resolve every tier in the family. Misses are silent zero handles —
    /// tiers 5/6/7 are absent from any target built before they existed, and
    /// dispatch degrades to the pre-existing batch4/batch8 decision.
    pub fn resolve(gpu: &dyn GpuBackend) -> Self {
        let mut handles = [KernelHandle(0); W4A16_BATCHM_WIDTHS.len()];
        for (h, w) in handles.iter_mut().zip(W4A16_BATCHM_WIDTHS) {
            // Width 8 goes through `batch8_kernel`, which prefers the
            // register-tiled rt2 variant ported from PR #648 and falls back to
            // the classic entry point. Resolved HERE rather than at the call
            // sites because this loop is the single place every M<=8 consumer
            // (dense FFN, both SSM paths, attention, MTP head) gets its
            // handle — the upstream branch had to patch four sites for the
            // same effect, and a fifth consumer added later would have missed
            // it. Launch geometry is identical, so nothing downstream changes.
            *h = if w == 8 {
                super::batch8_kernel(gpu)
            } else {
                super::try_kernel(gpu, "w4a16_gemv", &format!("w4a16_gemv_batch{w}"))
            };
        }
        Self { handles }
    }

    /// Which tiers this target resolved — the `present` argument of
    /// [`select_tier`].
    fn present(&self) -> [bool; W4A16_BATCHM_WIDTHS.len()] {
        self.handles.map(|h| h.0 != 0)
    }

    /// Narrowest resolved tier covering `m` rows, or `KernelHandle(0)` when
    /// this family cannot serve `m`.
    pub fn kernel(&self, m: u32) -> KernelHandle {
        select_tier(m, self.present(), exact_m_tiers_enabled())
            .map_or(KernelHandle(0), |i| self.handles[i])
    }

    /// Width of the tier [`Self::kernel`] would pick, for logging and tests.
    pub fn width(&self, m: u32) -> Option<u32> {
        select_tier(m, self.present(), exact_m_tiers_enabled()).map(|i| W4A16_BATCHM_WIDTHS[i])
    }

    /// Is the BASE (`w4a16_gemv_batch4`) tier resolved? Capability probes that
    /// ask "can this build do NVFP4 batched decode at all" mean this one
    /// specifically — it is the tier every target has carried.
    pub fn has_base(&self) -> bool {
        self.handles[0].0 != 0
    }
}

#[cfg(test)]
#[path = "w4a16_gemv_tiers_tests.rs"]
mod w4a16_gemv_tiers_tests;
