// SPDX-License-Identifier: AGPL-3.0-only

//! Log-once route lines for the `ATLAS_ATTN_M16_TC` tensor-core tier (#927).
//!
//! H100 round 9 (2026-09-11) ran cell W (`ATLAS_ATTN_M16_TC=1`) and found the
//! lever moves the measurement — C=16 aggregate 235.47 -> 247.85 tok/s
//! (+5.26%), TPOT 53.42 -> 50.01 ms (-6.38%), against a 0.15% rep spread —
//! with NO confirmation in the boot log. Quote from the operator report:
//! "Grepping the whole boot log for `m16`, `tensor` or an attention-decode
//! route returns only the prefill notice. The lever is live in
//! `/proc/<pid>/environ` and it moves the measurement by 25x the rep spread,
//! so it plainly armed — but unlike `ATLAS_LM_HEAD_M16_TC` (which logs a
//! detailed line) and unlike the W8A8 decode families, there is no log-once
//! confirmation that an operator could check." This module is that
//! confirmation.
//!
//! TWO keys, not one: the lever arms two independent tiers — the strided Q/K/V
//! GEMM in `qkv_fp8_batch.rs` and the contiguous o_proj GEMM in
//! `attn/o_proj.rs` — and an operator needs to see both arm independently, the
//! same reasoning `w8a8_decode.rs` uses for its `log:attn_qkv_w8a8_decode` /
//! `log:attn_o_proj_w8a8_decode` split.
//!
//! Lives beside `attn_ncol_gemv.rs` rather than inside either call site's
//! file: both multi-seq call sites need the SAME wording and the SAME
//! receipt, and a shared home keeps them from drifting apart the way two
//! independently-edited copies would.

use crate::layers::ops::ModelStats;

/// `ctx.stats.once` key for the strided Q/K/V tier (`qkv_fp8_batch.rs`).
pub(crate) const QKV_M16_TC_ROUTE_KEY: &str = "log:attn_qkv_m16_tc_decode";
/// `ctx.stats.once` key for the contiguous o_proj tier (`attn/o_proj.rs`).
pub(crate) const O_PROJ_M16_TC_ROUTE_KEY: &str = "log:attn_o_proj_m16_tc_decode";

/// The Q/K/V tier's line. Names the tier, the lever, the kernel, the fixed
/// N_TILE (attention has no wide `n64` twin — see `dense_ffn_m16_tc.rs`'s
/// module doc: "the wide tile has no strided twin and its N is already
/// CTA-starved"), the row band, the two kernels it displaces (it is checked
/// BEFORE both the bit-exact N-column tier and the batch16 GEMV), the numeric
/// consequence and the off switch.
///
/// 🔴 The numeric parenthetical states the CONTRACT, not a bare "<= 2 BF16
/// ULP". Round 9's LM-head receipt showed why that bound is the wrong thing to
/// print: the same MMA reassociation reaches 100 ordinal ULP on outputs that
/// have catastrophically cancelled, and the tier is held to
/// `within_m16_tc_budget` (2 ordinal ULP OR the FP32 accumulation floor), which
/// is what `w8a16_gemm_m16`'s own `gate/up M=32` cell already leaned on.
pub(crate) const QKV_M16_TC_ROUTE_MSG: &str = "\
[atlas] attention decode q/k/v: ATLAS_ATTN_M16_TC — tensor-core \
w8a16_gemm_m16_strided N_TILE=32 (fixed; the strided tier has no wide n64 \
twin) for 5..=16 rows, checked ahead of the bit-exact N-column tier \
(ATLAS_ATTN_NCOL_GEMV) and w8a16_gemv_batch16_strided. One weight pass, \
m16n8k16 MMA, so outputs are REASSOCIATED vs the scalar w8a16_gemv — within \
2 ordinal BF16 ULP, OR the FP32 accumulation floor for outputs that have \
catastrophically cancelled (the contract is \
layers::dense_ffn::m16_tc::within_m16_tc_budget, not a bare 2-ULP bound) — \
unlike the batch16 tier. Unset it to restore the bit-exact tier (#927; H100 \
round 9 cell W: C=16 TPOT 53.42 -> 50.01 ms, +5.26% aggregate).";

/// The o_proj tier's line. Same shape as [`QKV_M16_TC_ROUTE_MSG`], naming the
/// contiguous kernel and the two arms IT displaces.
pub(crate) const O_PROJ_M16_TC_ROUTE_MSG: &str = "\
[atlas] attention decode o_proj: ATLAS_ATTN_M16_TC — tensor-core \
w8a16_gemm_m16 N_TILE=32 (fixed; the contiguous tier has no wide n64 twin) \
for 5..=16 rows, checked ahead of the bit-exact N-column tier \
(ATLAS_ATTN_NCOL_GEMV) and w8a16_gemv_batch16. One weight pass, m16n8k16 \
MMA, so outputs are REASSOCIATED vs the scalar w8a16_gemv — within 2 ordinal \
BF16 ULP, OR the FP32 accumulation floor for outputs that have \
catastrophically cancelled (the contract is \
layers::dense_ffn::m16_tc::within_m16_tc_budget, not a bare 2-ULP bound) — \
unlike the batch16 tier. Unset it to restore the bit-exact tier (#927; H100 \
round 9 cell W: C=16 TPOT 53.42 -> 50.01 ms, +5.26% aggregate).";

/// Fire the Q/K/V route line, once per model. Called from inside the `tc`
/// arm of `ms_qkv_batchm_fp8_gemv` (`qkv_fp8_batch.rs`) — i.e. only on the
/// call that actually launches `w8a16_gemm_m16_strided`.
pub(crate) fn log_qkv_m16_tc_route(stats: &ModelStats) {
    if stats.once(QKV_M16_TC_ROUTE_KEY) {
        tracing::info!("{QKV_M16_TC_ROUTE_MSG}");
    }
}

/// Fire the o_proj route line, once per model. Called from inside the `tc`
/// arm of `ms_phase_o_proj` (`attn/o_proj.rs`) — i.e. only on the call that
/// actually launches `w8a16_gemm_m16`.
pub(crate) fn log_o_proj_m16_tc_route(stats: &ModelStats) {
    if stats.once(O_PROJ_M16_TC_ROUTE_KEY) {
        tracing::info!("{O_PROJ_M16_TC_ROUTE_MSG}");
    }
}

#[cfg(test)]
#[path = "attn_m16_tc_route_tests.rs"]
mod tests;
