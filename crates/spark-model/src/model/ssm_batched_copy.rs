// SPDX-License-Identifier: AGPL-3.0-only

//! Batched execution of the SSM verify-state copy sets.
//!
//! Every speculative-verify state move — pre-verify checkpoint, full-reject
//! rollback, partial-accept commit — is the SAME shape: for each of the
//! model's SSM layers, copy one h blob and one conv blob between two pool
//! regions. On the 27B hybrid (48 GDN layers) that loop issued **96 eager
//! `copy_d2d_async` launches per sequence per verify step**, entirely outside
//! any captured graph; at n=8 sequences a single decode step spent 768
//! launches on it.
//!
//! The blobs are not adjacent within a layer (h and conv live in different
//! pool families), but the per-layer regions of ONE family are uniformly
//! strided — [`super::ssm_pool::SsmStatePool`] allocates each family as one
//! contiguous block and hands out `base + layer * stride` (see
//! `alloc_layer_pools` there). So the 48 h copies collapse to a single
//! pitched 2-D copy, and likewise the 48 conv copies:
//!
//! | | launches / sequence / step | at n=8 |
//! |---|---|---|
//! | per-layer loop | 96 | 768 |
//! | batched | 2 | 16 |
//!
//! **The bytes are identical.** [`copy_plan_as_strided_run`] only collapses a
//! plan whose rows it can reproduce exactly — same width, same monotone
//! pitch — and `copy_d2d_2d_async`'s own fallback (and its
//! `cudaMemcpy2DAsync` implementation) writes row `r` as `width_bytes` from
//! `src + r*src_pitch` to `dst + r*dst_pitch`, which is row `r` of the plan
//! by construction. Anything that does not collapse (a fragmented pool, a
//! ragged plan, a single-layer model) runs the original loop verbatim.
//!
//! Kill switch: `ATLAS_NO_BATCHED_SSM_ROLLBACK=1` forces the loop everywhere.

use anyhow::Result;
use spark_runtime::gpu::{DevicePtr, GpuBackend};

/// One device-to-device state blob move: `bytes` from `src` to `dst`.
///
/// A `Vec<StateCopy>` is the SSOT for "what this rollback does" — both
/// executors below consume the same plan, so the batched and looped forms
/// cannot drift apart in which bytes land where.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct StateCopy {
    pub src: DevicePtr,
    pub dst: DevicePtr,
    pub bytes: usize,
}

/// A copy plan that collapses to one pitched 2-D transfer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct StridedRun {
    pub src: DevicePtr,
    pub src_pitch: usize,
    pub dst: DevicePtr,
    pub dst_pitch: usize,
    pub width_bytes: usize,
    pub height: usize,
}

/// Can `plan` be issued as ONE pitched 2-D copy instead of `plan.len()`
/// separate ones? PURE — this is where the equivalence argument lives, so it
/// is testable without a GPU.
///
/// Requires, and checks, every condition `cudaMemcpy2DAsync` needs to
/// reproduce the plan row-for-row:
/// - at least two rows (one row is already one launch — nothing to gain, and
///   a pitch cannot be derived from a single element);
/// - every row the same `width_bytes` (a ragged plan is not a 2-D shape);
/// - a single FORWARD `src_pitch` and `dst_pitch` shared by every consecutive
///   pair (row `r` must sit at `base + r*pitch`);
/// - `pitch >= width_bytes` on both sides, so consecutive rows never overlap
///   — the CUDA precondition, and what makes row order irrelevant.
///
/// Returns `None` on anything else; the caller then runs the plan as-is.
pub(crate) fn copy_plan_as_strided_run(plan: &[StateCopy]) -> Option<StridedRun> {
    if plan.len() < 2 {
        return None;
    }
    let width_bytes = plan[0].bytes;
    if width_bytes == 0 || plan.iter().any(|c| c.bytes != width_bytes) {
        return None;
    }
    // Pitches are derived from the first pair and then VERIFIED against every
    // other pair — `checked_sub` on u64 rejects a descending family outright
    // (a negative pitch has no `cudaMemcpy2D` representation).
    let src_pitch = plan[1].src.0.checked_sub(plan[0].src.0)?;
    let dst_pitch = plan[1].dst.0.checked_sub(plan[0].dst.0)?;
    if src_pitch < width_bytes as u64 || dst_pitch < width_bytes as u64 {
        return None;
    }
    for (r, c) in plan.iter().enumerate() {
        let r = r as u64;
        if c.src.0 != plan[0].src.0 + r * src_pitch || c.dst.0 != plan[0].dst.0 + r * dst_pitch {
            return None;
        }
    }
    Some(StridedRun {
        src: plan[0].src,
        src_pitch: src_pitch as usize,
        dst: plan[0].dst,
        dst_pitch: dst_pitch as usize,
        width_bytes,
        height: plan.len(),
    })
}

/// Kill switch for the batched form: `ATLAS_NO_BATCHED_SSM_ROLLBACK=1`
/// restores the per-layer `copy_d2d_async` loop everywhere.
///
/// PRESENCE check per the house convention (`=0` is NOT off), read once per
/// process — this predicate sits on the verify path, which runs per sequence
/// per decode step.
pub(crate) fn batched_ssm_copy_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("ATLAS_NO_BATCHED_SSM_ROLLBACK").is_none())
}

/// Issue `plan` on `stream`, batched when it collapses.
///
/// `batched` is a PARAMETER, not a flag read, so both polarities are testable
/// without touching process env or the latching `OnceLock` above (which would
/// make such tests order-dependent). Production callers pass
/// [`batched_ssm_copy_enabled`].
pub(crate) fn run_state_copies_with(
    gpu: &dyn GpuBackend,
    plan: &[StateCopy],
    batched: bool,
    stream: u64,
) -> Result<()> {
    if batched && let Some(run) = copy_plan_as_strided_run(plan) {
        return gpu.copy_d2d_2d_async(
            run.src,
            run.src_pitch,
            run.dst,
            run.dst_pitch,
            run.width_bytes,
            run.height,
            stream,
        );
    }
    for c in plan {
        gpu.copy_d2d_async(c.src, c.dst, c.bytes, stream)?;
    }
    Ok(())
}

/// [`run_state_copies_with`] at the process-wide kill-switch setting.
pub(crate) fn run_state_copies(
    gpu: &dyn GpuBackend,
    plan: &[StateCopy],
    stream: u64,
) -> Result<()> {
    run_state_copies_with(gpu, plan, batched_ssm_copy_enabled(), stream)
}

/// Issue an h plan and a conv plan back to back.
///
/// The two families are issued as separate runs rather than one interleaved
/// loop because a 2-D copy needs ONE width, and h blobs and conv blobs have
/// different widths. Re-ordering across the families is sound: they address
/// DISJOINT pool allocations (`h_*_pools` vs `conv_*_pools` in
/// [`super::ssm_pool::SsmStatePool`]), so no h copy can read or write a byte
/// any conv copy touches, and both land on the same stream before any
/// consumer of either.
pub(crate) fn run_ssm_state_copies(
    gpu: &dyn GpuBackend,
    h_plan: &[StateCopy],
    conv_plan: &[StateCopy],
    stream: u64,
) -> Result<()> {
    let batched = batched_ssm_copy_enabled();
    run_state_copies_with(gpu, h_plan, batched, stream)?;
    run_state_copies_with(gpu, conv_plan, batched, stream)
}

#[cfg(test)]
#[path = "ssm_batched_copy_tests.rs"]
mod tests;
