// SPDX-License-Identifier: AGPL-3.0-only

//! M-row kernel selection for the batched cross-sequence drafter propose.
//!
//! Pure decision function + its two env levers, split out of
//! [`super::forward_batch`] so the M→kernel table is testable without a GPU
//! (and so the file stays under the 500-line cap).
//!
//! ## The defect this exists to fix
//!
//! nsys on dgx2 (C=1 vs C=2 decode, 27B, MTP K=4) attributed the drafter at
//! **14.22 ms/step at C=1** and **21.46 ms/step at C=2** — +7.24 ms of the
//! +51 ms/step cost of a second sequence, the second-largest term after the
//! main projection GEMV. Per draft position:
//!
//! | M | dispatch today | cost |
//! |---|---|---|
//! | 1 | `dense_gemv_bf16` (per-seq [`super::forward`]) | 3.57 ms |
//! | 2 | `dense_gemm_bf16_pipelined` ([`super::forward_batch`]) | 5.43 ms |
//!
//! One draft position reads ~849 MB of BF16 drafter weights (the eight
//! projections below), so M=1 at 3.57 ms is **238 GB/s** — essentially the
//! GB10 roofline (273 GB/s LPDDR5X). M=2 at 5.43 ms is 156 GB/s: **1.52x for
//! two rows** on a path where the weights are streamed once and the second
//! row should be nearly free.
//!
//! `dense_gemv_bf16_batchm` is exactly that "stream once, M accumulators"
//! kernel and already exists — it is simply not wired into the drafter. Its
//! gating microbench (commit 250931e97, `examples/dense_gemv_bf16_batchm_microtest`,
//! real decode shapes, K=3072) measured one batchm launch against M separate
//! M=1 launches:
//!
//! | shape | M | N | Mx M=1 | batchm | speedup |
//! |---|---|---|---|---|---|
//! | q_proj | 2 | 9216 | 0.4130 ms | 0.2059 ms | 2.01x |
//! | q_proj | 4 | 9216 | 0.8398 ms | 0.2122 ms | 3.96x |
//! | o_proj | 4 | 3072 | 0.0449 ms | 0.0199 ms | 2.26x |
//! | k/v    | 4 | 1024 | 0.0208 ms | 0.0115 ms | 1.81x |
//!
//! The load-bearing number is `q_proj`: batchm at M=4 costs 0.2122 ms against
//! 0.2100 ms for a single M=1 launch — **+1% for 4x the rows**. batchm's cost
//! is flat in M, so on the drafter it should land at the M=1 3.57 ms rather
//! than the pipelined GEMM's 5.1-5.4 ms, at every M it covers.
//!
//! ## Where the crossover comes from
//!
//! Not invented here. `dense_gemv_bf16_batchm` has a compile-time `MAX_M 8`
//! ([`DENSE_GEMV_BATCHM_MAX_M`]) and **clamps silently** above it, so 8 is a
//! hard ceiling, not a tuning choice. The floor is 2 because M=1 already has
//! a dedicated kernel and never reaches this path. The main model runs the
//! identical kernel over the identical `(2..=8)` band at three sites
//! (`multi_seq/qkv.rs`, `multi_seq/attn/o_proj.rs` x2), measured +6% at C=2
//! and +24% at C=4 (commit 84d5b763c). Above 8 the batched-GEMV family was
//! measured NEGATIVE against the tile GEMM (-14.4% at C=16, -29.4% at C=32,
//! commit 78d276832), which is why this tier stops at 8 instead of growing a
//! wider kernel.
//!
//! No drafter-specific microbench exists for these shapes (`batchm_bench` is
//! the w4a16 family, not the BF16 one), so the band is mirrored from the
//! main-model tiering rather than re-measured — recorded here as the reason.
//!
//! ## Numerics
//!
//! `dense_gemv_bf16_batchm` is **bit-identical to M separate `dense_gemv_bf16`
//! calls** (same per-row K-iteration order and reduction tree; the kernel dir
//! builds with `--fmad=false`), verified across all 9 shape/M combinations of
//! `examples/dense_gemv_bf16_batchm_microtest`. So:
//!
//! * vs the **per-row GEMV loop** it replaces (small N, m in 2..=7 today):
//!   bit-exact.
//! * vs the **pipelined tile GEMM** it replaces (N >= 4096, and small N at
//!   m == 8): NOT bit-exact — a tensor-core tile GEMM has a different
//!   accumulation order. The drafter output moves in the last bits.
//! * vs the **single-sequence propose** ([`super::forward`], which runs
//!   `ops::dense_gemv` for all eight projections): bit-exact, at every M this
//!   tier covers. The batched propose becomes numerically identical to the
//!   C=1 propose it is supposed to be a batched sibling of — today it is not.
//!
//! Drafts are proposals: every one is checked by the main model's verify, so
//! accepted output is unaffected by construction. Only the ACCEPT RATE can
//! move, and it moves toward the C=1 path.

use crate::layers::ops::DENSE_GEMV_BATCHM_MAX_M;

/// N at or above which the pipelined tile GEMM fills its 128-wide tile well
/// enough to beat the per-row GEMV loop. Pre-existing threshold, unchanged —
/// on the 27B drafter it separates {fc 5120, q 12288, o 5120, gate/up 17408,
/// down 5120} from {k 1024, v 1024}.
const TILE_N_MIN: u32 = 4096;

/// Which kernel one `gemm_rows` projection dispatches to.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum RowKernel {
    /// `dense_gemv_bf16_batchm`: ONE weight pass, M row accumulators.
    Batchm,
    /// `dense_gemm_bf16_pipelined`: tensor-core tile GEMM.
    Pipelined,
    /// `dense_gemv_bf16`, once per row.
    GemvLoop,
}

/// Select the kernel for an `[m, k] x [n, k]^T` drafter projection.
///
/// Pure: every input is an explicit argument, including both env levers, so
/// the whole table is a unit test.
///
/// * `batchm_ready` — the `dense_gemv_bf16_batchm` handle resolved
///   (`try_kernel` misses are a silent 0; older kernel sets fall back).
/// * `kv_gemv_pinned` — `ATLAS_MTP_KV_GEMV` present: the PRE-EXISTING lever
///   that pins the small-N (K/V) projections to the per-row GEMV loop. It
///   keeps that meaning here rather than going inert at m == 8, the one width
///   where it used to bite and the new tier would otherwise swallow it.
/// * `small_m_tier_off` — `ATLAS_NO_DRAFTER_SMALL_M_TIER=1`: restores the
///   pre-tier dispatch exactly, for a same-session A/B control.
pub(crate) fn drafter_row_kernel(
    m: usize,
    n: u32,
    k: u32,
    batchm_ready: bool,
    kv_gemv_pinned: bool,
    small_m_tier_off: bool,
) -> RowKernel {
    let small_n = n < TILE_N_MIN;
    // `k & 7`: `dense_gemv_bf16_batchm` casts each `A + t*K` and `B + n*K` row
    // base to `uint4*` (16-byte loads), so a K that is not a multiple of 8
    // misaligns every row past the first. The whole BF16 GEMV family shares
    // that assumption ("never hits for model dims"), but the tier must not be
    // the one that newly reaches an unaligned shape — it stays on the arms
    // that already served it. Every 27B drafter K (10240, 5120, 6144, 17408)
    // is a multiple of 8.
    let k_vec8 = (k & 7) == 0;
    if batchm_ready
        && !small_m_tier_off
        && k_vec8
        && (2..=DENSE_GEMV_BATCHM_MAX_M as usize).contains(&m)
        && !(small_n && kv_gemv_pinned)
    {
        return RowKernel::Batchm;
    }

    // ── Pre-tier dispatch, verbatim. Reached at m == 1, at m > MAX_M (the
    // C=16/32 propose widths, where the tile GEMM measured better anyway),
    // when the kernel set lacks batchm, and under either kill switch.
    let small_n_tile = m >= 8 && k_vec8 && !kv_gemv_pinned;
    if (!small_n || small_n_tile) && k_vec8 {
        RowKernel::Pipelined
    } else {
        RowKernel::GemvLoop
    }
}

/// `ATLAS_NO_DRAFTER_SMALL_M_TIER=1` — restore the pre-tier dispatch.
/// Read once (this is on the per-draft-position path: 8 projections x K
/// draft positions x every step).
pub(crate) fn small_m_tier_off() -> bool {
    static OFF: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *OFF.get_or_init(|| {
        std::env::var("ATLAS_NO_DRAFTER_SMALL_M_TIER")
            .ok()
            .as_deref()
            == Some("1")
    })
}

/// `ATLAS_MTP_KV_GEMV` (presence) — pin the small-N K/V projections to the
/// per-row GEMV loop. Pre-existing lever; hoisted out of the hot path into a
/// `OnceLock` alongside the new one.
pub(crate) fn kv_gemv_pinned() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("ATLAS_MTP_KV_GEMV").is_some())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The eight weight-bearing projections of ONE drafter draft position on
    /// Qwen3.6-27B (h 5120, nq 24, nkv 4, head_dim 256, inter 17408), in
    /// forward order, as `(label, N, K)`. Their BF16 weight bytes sum to
    /// 849.4 MB — the "~850 MB per forward" the module header divides by.
    const DRAFTER_SHAPES: &[(&str, u32, u32)] = &[
        ("fc", 5120, 10240),
        ("q_proj", 12288, 5120),
        ("k_proj", 1024, 5120),
        ("v_proj", 1024, 5120),
        ("o_proj", 5120, 6144),
        ("ffn_gate", 17408, 5120),
        ("ffn_up", 17408, 5120),
        ("ffn_down", 5120, 17408),
    ];

    /// The dispatch as it stood before this tier, transcribed from
    /// `forward_batch.rs::gemm_rows` at b08802cf1. Independent of the
    /// production expression on purpose: the kill-switch test below is only
    /// worth anything if it compares against a SEPARATE statement of the old
    /// behaviour.
    fn legacy(m: usize, n: u32, k: u32, kv_gemv_pinned: bool) -> RowKernel {
        let small_n_tile = m >= 8 && (k & 7) == 0 && !kv_gemv_pinned;
        if (n >= 4096 || small_n_tile) && (k & 7) == 0 {
            RowKernel::Pipelined
        } else {
            RowKernel::GemvLoop
        }
    }

    /// The whole point: at the widths the batched propose actually runs
    /// (C=2 -> m=2, C=4 -> m=4, C=8 -> m=8), EVERY projection takes the
    /// batched GEMV — including the two N=1024 K/V ones, which is the
    /// `small_n_tile = m >= 8` sub-defect subsumed rather than re-tuned.
    #[test]
    fn every_projection_batches_across_the_covered_widths() {
        for m in 2..=DENSE_GEMV_BATCHM_MAX_M as usize {
            for &(label, n, k) in DRAFTER_SHAPES {
                assert_eq!(
                    drafter_row_kernel(m, n, k, true, false, false),
                    RowKernel::Batchm,
                    "{label} at m={m} must take the batched GEMV"
                );
            }
        }
    }

    /// m=1 is unreachable in the batched propose (the scheduler only groups
    /// `group.len() >= 2`), and m > MAX_M is where the batchm family measured
    /// NEGATIVE. Both must be byte-for-byte the old dispatch, or the C=1 and
    /// C=16/32/64/128 ladder rungs move under a change that claims not to
    /// touch them.
    #[test]
    fn widths_outside_the_tier_are_untouched() {
        let outside = [1usize, 9, 10, 12, 16, 17, 24, 32, 64];
        for m in outside {
            for &(label, n, k) in DRAFTER_SHAPES {
                assert_eq!(
                    drafter_row_kernel(m, n, k, true, false, false),
                    legacy(m, n, k, false),
                    "{label} at m={m} must keep the pre-tier dispatch"
                );
            }
        }
    }

    /// `ATLAS_NO_DRAFTER_SMALL_M_TIER=1` must reproduce the pre-tier decision
    /// for EVERY (m, shape, other-lever) combination — that is what makes it
    /// a valid same-session A/B control for the ladder.
    #[test]
    fn kill_switch_restores_the_pre_tier_dispatch_exactly() {
        for m in 1..=64usize {
            for &(label, n, k) in DRAFTER_SHAPES {
                for kv_pin in [false, true] {
                    assert_eq!(
                        drafter_row_kernel(m, n, k, true, kv_pin, true),
                        legacy(m, n, k, kv_pin),
                        "{label} m={m} kv_pin={kv_pin} under the kill switch"
                    );
                }
            }
        }
    }

    /// A 0 handle (kernel absent from this target's set) must behave exactly
    /// like the kill switch — `try_kernel` misses are silent, so the fallback
    /// has to be total, not partial.
    #[test]
    fn missing_kernel_handle_falls_back_like_the_kill_switch() {
        for m in 1..=16usize {
            for &(_, n, k) in DRAFTER_SHAPES {
                assert_eq!(
                    drafter_row_kernel(m, n, k, false, false, false),
                    legacy(m, n, k, false),
                );
            }
        }
    }

    /// `ATLAS_MTP_KV_GEMV` keeps meaning "small-N K/V on the per-row loop" at
    /// every width, including m=8 where the new tier would otherwise silently
    /// swallow it. Large-N projections still batch.
    #[test]
    fn kv_gemv_lever_still_pins_small_n_at_every_width() {
        for m in 2..=DENSE_GEMV_BATCHM_MAX_M as usize {
            // N=1024 K/V.
            assert_eq!(
                drafter_row_kernel(m, 1024, 5120, true, true, false),
                RowKernel::GemvLoop,
                "K/V at m={m} under ATLAS_MTP_KV_GEMV"
            );
            // N=17408 FFN is not small-N; the lever does not reach it.
            assert_eq!(
                drafter_row_kernel(m, 17408, 5120, true, true, false),
                RowKernel::Batchm,
                "ffn_gate at m={m} is unaffected by ATLAS_MTP_KV_GEMV"
            );
        }
    }

    /// The tier must never hand `dense_gemv_batchm` an m the kernel clamps:
    /// `MAX_M` is a compile-time cap that silently truncates, so rows
    /// `MAX_M..m` would be left as stale bytes.
    #[test]
    fn never_selects_batchm_above_the_kernel_row_cap() {
        for m in 1..=128usize {
            for &(_, n, k) in DRAFTER_SHAPES {
                for (batchm, kv_pin, off) in [
                    (true, false, false),
                    (true, true, false),
                    (true, false, true),
                    (false, false, false),
                ] {
                    if drafter_row_kernel(m, n, k, batchm, kv_pin, off) == RowKernel::Batchm {
                        assert!(
                            (2..=DENSE_GEMV_BATCHM_MAX_M as usize).contains(&m),
                            "batchm selected at m={m}, outside 2..={DENSE_GEMV_BATCHM_MAX_M}"
                        );
                    }
                }
            }
        }
    }

    /// A K that is not a multiple of 8 misaligns the 16-byte row loads of
    /// BOTH batched kernels, so the tier must leave it exactly where it was —
    /// on the per-row GEMV loop — rather than becoming the first arm to hand
    /// an unaligned shape to a vectorized kernel.
    #[test]
    fn unaligned_k_stays_on_the_per_row_loop() {
        for m in [1usize, 2, 4, 8, 16] {
            for &(_, n, _) in DRAFTER_SHAPES {
                assert_eq!(
                    drafter_row_kernel(m, n, 5123, true, false, false),
                    RowKernel::GemvLoop,
                    "m={m} n={n} with an unaligned K"
                );
                assert_eq!(
                    drafter_row_kernel(m, n, 5123, true, false, false),
                    legacy(m, n, 5123, false),
                    "m={m} n={n}: unaligned K must match the pre-tier dispatch"
                );
            }
        }
    }
}
