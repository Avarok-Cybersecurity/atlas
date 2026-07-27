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

use std::sync::atomic::{AtomicU64, Ordering};

use spark_runtime::gpu::{DevicePtr, GpuBackend};

const SAMPLE_EVERY: u64 = 64;
const LOG_EVERY: u64 = 64;

static CALLS: AtomicU64 = AtomicU64::new(0);
static SAMPLES: AtomicU64 = AtomicU64::new(0);
static UNIQUE_SUM: AtomicU64 = AtomicU64::new(0);
static SLOTS_SUM: AtomicU64 = AtomicU64::new(0);

fn enabled() -> bool {
    static CACHED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *CACHED.get_or_init(|| std::env::var("ATLAS_MOE_UNION_STATS").ok().as_deref() == Some("1"))
}

/// Sample the expert-index union for one MoE layer's verify batch.
/// `indices_dev` = `[m * top_k]` u32 expert ids, already written on `stream`.
pub(super) fn maybe_sample_expert_union(
    gpu: &dyn GpuBackend,
    indices_dev: DevicePtr,
    m: usize,
    top_k: usize,
    stream: u64,
) {
    if !enabled() {
        return;
    }
    // NEVER sync/copy inside a CUDA-graph capture — it invalidates the
    // capture (CUDA 901) and wedges the serve (measured: 35B NVFP4
    // decode_verify_graphed, 2026-07-20). Graph REPLAYS run no host code at
    // all, so this tap inherently samples only eager verify steps; disable
    // graphs for full-fidelity measurement runs.
    if gpu.stream_is_capturing(stream) {
        return;
    }
    let call = CALLS.fetch_add(1, Ordering::Relaxed);
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
    UNIQUE_SUM.fetch_add(unique, Ordering::Relaxed);
    SLOTS_SUM.fetch_add(n as u64, Ordering::Relaxed);
    let s = SAMPLES.fetch_add(1, Ordering::Relaxed) + 1;
    if s.is_multiple_of(LOG_EVERY) {
        let uniq = UNIQUE_SUM.load(Ordering::Relaxed) as f64 / s as f64;
        let slots = SLOTS_SUM.load(Ordering::Relaxed) as f64 / s as f64;
        tracing::info!(
            "moe-union-stats: samples={s} mean_unique_experts={uniq:.1} \
             mean_routed_slots={slots:.1} overlap_saving={:.0}% (m={m} top_k={top_k})",
            (1.0 - uniq / slots.max(1.0)) * 100.0,
        );
    }
}
