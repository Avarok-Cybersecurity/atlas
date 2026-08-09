// SPDX-License-Identifier: AGPL-3.0-only

//! Sampled expert-union telemetry for MTP verify batches
//! (`ATLAS_MOE_UNION_STATS=1`, default off = zero cost).
//!
//! MoE verify cost scales with the UNION of experts activated across the
//! verify batch's tokens, not with token count (measured verify_multiplier
//! ~2.07 at K=1 on the 35B vs ~1.1 dense; cf. arXiv 2605.00342). This tap
//! measures that union in production so the expert-union-aware verify
//! batching work is grounded in data: mean unique experts per layer-step
//! vs the M*top_k worst case and the num_experts ceiling.
//!
//! Sampling: 1 in [`SAMPLE_EVERY`] calls per process (call sites are
//! per-MoE-layer per verify step), each sample a `M*top_k*4`-byte D2H
//! after a stream sync — nanoseconds amortized. Aggregate is logged every
//! [`LOG_EVERY`] samples.

use std::sync::atomic::Ordering;

use spark_runtime::gpu::DevicePtr;

const SAMPLE_EVERY: u64 = 64;
const LOG_EVERY: u64 = 64;

// The four counters that lived here are now `ModelStats::moe_union`, reached
// through `ForwardContext`. Expert-union density is a property of ONE model's
// router: summed across a swap the periodic aggregate reports a mean of two
// routers, which describes neither. The `OnceLock<bool>` that gated sampling
// is likewise `ModelLevers::moe_union_stats`, passed in by the caller.

/// Sample the expert-index union for one MoE layer's verify batch.
/// `indices_dev` = `[m * top_k]` u32 expert ids, already written on `stream`.
pub(super) fn maybe_sample_expert_union(
    // The whole context, not three fields off it: the lever, the counters and
    // the backend are all this model's, and taking them separately made the
    // call site three lines longer than the thing it gates.
    ctx: &crate::layer::ForwardContext<'_>,
    indices_dev: DevicePtr,
    m: usize,
    top_k: usize,
    stream: u64,
) {
    if !ctx.levers.moe_union_stats {
        return;
    }
    // NEVER sync/copy inside a CUDA-graph capture — it invalidates the
    // capture (CUDA 901) and wedges the serve (measured: 35B NVFP4
    // decode_verify_graphed, 2026-07-20). Graph REPLAYS run no host code at
    // all, so this tap inherently samples only eager verify steps; disable
    // graphs for full-fidelity measurement runs.
    let gpu = ctx.gpu;
    if gpu.stream_is_capturing(stream) {
        return;
    }
    let call = ctx.stats.moe_union.calls.fetch_add(1, Ordering::Relaxed);
    if !call.is_multiple_of(SAMPLE_EVERY) {
        return;
    }
    // Order the D2H after the topk kernel that produced the indices.
    if gpu.synchronize(stream).is_err() {
        return;
    }
    let n = m * top_k;
    let mut buf = vec![0u8; n * 4];
    if gpu.copy_d2h(indices_dev, &mut buf).is_err() {
        return;
    }
    let mut ids: Vec<u32> = buf
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    ids.sort_unstable();
    ids.dedup();
    let unique = ids.len() as u64;
    ctx.stats
        .moe_union
        .unique_sum
        .fetch_add(unique, Ordering::Relaxed);
    ctx.stats
        .moe_union
        .slots_sum
        .fetch_add(n as u64, Ordering::Relaxed);
    let s = ctx.stats.moe_union.samples.fetch_add(1, Ordering::Relaxed) + 1;
    if s.is_multiple_of(LOG_EVERY) {
        let uniq = ctx.stats.moe_union.unique_sum.load(Ordering::Relaxed) as f64 / s as f64;
        let slots = ctx.stats.moe_union.slots_sum.load(Ordering::Relaxed) as f64 / s as f64;
        tracing::info!(
            "moe-union-stats: samples={s} mean_unique_experts={uniq:.1} \
             mean_routed_slots={slots:.1} overlap_saving={:.0}% (m={m} top_k={top_k})",
            (1.0 - uniq / slots.max(1.0)) * 100.0,
        );
    }
}
