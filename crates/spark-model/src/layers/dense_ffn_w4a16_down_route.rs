// SPDX-License-Identifier: AGPL-3.0-only

//! Log-once route line for which `down` arm the W4A16/NVFP4 M=1 decode step
//! took — the split-SiLU default (`w4a16_gemv` after `silu_mul`) or the fused
//! `w4a16_gemv_silu_input[_sw]` kernel `ATLAS_NO_DECODE_SPLIT_SILU` restores.
//!
//! Sibling to `dense_ffn_fp8_down.rs`'s `log_fp8_down_route`: the SAME lever
//! (`ModelLevers::decode_split_silu`) gates BOTH weight formats' down
//! projection, at two separate decision sites in `dense_ffn.rs::forward` (the
//! FP8 branch via `fp8_down::fp8_down_arm`, this one directly as
//! `let split_silu = ...`). H100 round 9 measured the FP8 sibling's numerics
//! and speed (`native_fp8_ffn_down_gemv_microtest`: 1.79x, 843 -> 1513 GB/s;
//! `unequal=2130 max_abs=0.5 max_ulp=31195` vs the fused kernel) — no W4A16-
//! specific microtest has run the same comparison, so this line states the
//! REPRESENTATION change (BF16-staged SwiGLU vs an FP32 accumulator) without
//! borrowing the FP8 arm's numbers. The operator complaint is the same either
//! way: "the log always says which `down` path is live."
//!
//! A separate small file rather than folded into `dense_ffn.rs` directly:
//! that file is already the crate's largest, and a route-log module belongs
//! beside its FP8 twin (`dense_ffn_fp8_down.rs`), not buried inside `forward`.

use crate::layers::ops::ModelStats;

/// `ctx.stats.once` key for the split-SiLU default arm.
pub(crate) const W4A16_DOWN_SPLIT_SILU_KEY: &str = "log:ffn_down_split_silu_w4a16";
/// `ctx.stats.once` key for the fused-kernel arm (`ATLAS_NO_DECODE_SPLIT_SILU`
/// restores it, or it is reached directly because activation/lora made the
/// split path unavailable — see the message text).
pub(crate) const W4A16_DOWN_FUSED_SILU_KEY: &str = "log:ffn_down_fused_silu_w4a16";

/// The split-SiLU default's line for the W4A16/NVFP4 weight format.
pub(crate) const W4A16_DOWN_SPLIT_SILU_MSG: &str = "\
[atlas] dense FFN decode down: split-SiLU + w4a16_gemv (default; \
ATLAS_NO_DECODE_SPLIT_SILU restores the fused kernel; also pinned whenever a \
LoRA adapter is installed, since the fused kernel never materialises \
silu(gate)*up for the down delta to contract over). Same lever as the \
native-FP8 down arm (layers::dense_ffn::fp8_down), which H100 round 9 \
measured at 1.79x on down (843 -> 1513 GB/s, \
native_fp8_ffn_down_gemv_microtest). NOT bit-identical to the fused \
w4a16_gemv_silu_input kernel it replaces: the SwiGLU product is materialised \
in BF16 rather than kept in the FP32 accumulator.";

/// The fused-kernel arm's line — the "other branch" a route log must also
/// cover, so an operator can tell the two down paths apart either way.
pub(crate) const W4A16_DOWN_FUSED_SILU_MSG: &str = "\
[atlas] dense FFN decode down: fused w4a16_gemv_silu_input \
(ATLAS_NO_DECODE_SPLIT_SILU set, or the split path is unavailable on this \
layer — restores bit-exact numerics, off the split-SiLU + w4a16_gemv \
default).";

/// Fire the split-SiLU-default line, once per model.
pub(crate) fn log_w4a16_down_split_silu_route(stats: &ModelStats) {
    if stats.once(W4A16_DOWN_SPLIT_SILU_KEY) {
        tracing::info!("{W4A16_DOWN_SPLIT_SILU_MSG}");
    }
}

/// Fire the fused-kernel line, once per model.
pub(crate) fn log_w4a16_down_fused_silu_route(stats: &ModelStats) {
    if stats.once(W4A16_DOWN_FUSED_SILU_KEY) {
        tracing::info!("{W4A16_DOWN_FUSED_SILU_MSG}");
    }
}

#[cfg(test)]
#[path = "dense_ffn_w4a16_down_route_tests.rs"]
mod tests;
