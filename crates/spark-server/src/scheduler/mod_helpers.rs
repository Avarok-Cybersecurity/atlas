// SPDX-License-Identifier: AGPL-3.0-only

//! Per-iteration helpers extracted from `scheduler::run` (refactor
//! wave-4e):
//!   • install_high_speed_swap — orchestrator install after CUDA bind
//!   • drain_pending_requests — pop policy-selected reqs off the queue
//!   • retire_finished_sequences — swap_remove + slot compaction

use parking_lot::{Condvar, Mutex};
use spark_model::traits::Model;
use std::sync::Arc;

use super::*;
use crate::api::InferenceRequest;
use crate::scheduling_policy::{ActiveSeqTiming, PendingRequestInfo, SchedulingPolicy};

/// Install --high-speed-swap orchestrator after bind_gpu_to_thread.
pub(super) fn install_high_speed_swap(
    model: &dyn Model,
    cfg: Option<spark_storage::HighSpeedSwapConfig>,
) {
    let Some(cfg) = cfg else { return };
    match model.high_speed_swap_dims() {
        Some(dims) => {
            tracing::info!(
                "--high-speed-swap installing: dir={}, scratch={} blocks, qd={}, rank={}, \
                 model: {} layers × {}/{} (q/kv) heads × hd={}, bs={}, max_blocks={}",
                cfg.dir.display(),
                cfg.resident_blocks,
                cfg.qd,
                cfg.rank,
                dims.num_layers,
                dims.num_q_heads,
                dims.num_kv_heads,
                dims.head_dim,
                dims.block_size,
                dims.max_blocks_per_layer,
            );
            // Use the model's default stream (cuMemcpyHtoDAsync(stream=0))
            // for orchestrator setup. The hot-path API takes its own stream.
            if let Err(e) = spark_storage::install_local(0, cfg, dims) {
                tracing::error!("--high-speed-swap install failed: {e:#}");
            } else {
                tracing::info!("--high-speed-swap orchestrator installed on scheduler thread");
                if std::env::var("ATLAS_HIGH_SPEED_SWAP_REPLACE").is_ok() {
                    tracing::warn!(
                        "ATLAS_HIGH_SPEED_SWAP_REPLACE=1: per-layer attention will route \
                         through HighSpeedSwap. UNTESTED on real models — requires real-load \
                         validation before production use."
                    );
                }
            }
        }
        None => {
            tracing::warn!(
                "--high-speed-swap requested but model does not expose high_speed_swap_dims; \
                 orchestrator NOT installed"
            );
        }
    }
}

/// Drain pending request queue and policy-select prefills to start.
pub(super) fn drain_pending_requests(
    pending: &Arc<(Mutex<PendingQueue>, Condvar)>,
    active: &[ActiveSeq],
    prefilling: &[PrefillInProgress],
    policy: &dyn SchedulingPolicy,
    max_batch_size: usize,
) -> Vec<InferenceRequest> {
    let (ref mtx, ref cv) = **pending;
    let mut g = mtx.lock();
    if active.is_empty() && prefilling.is_empty() {
        // Block until signalled (no busy-wait, no polling).
        while g.requests.is_empty() && !g.closed {
            cv.wait(&mut g);
        }
        if g.closed && g.requests.is_empty() {
            return Vec::new();
        }
    }

    // Ask policy whether to accept prefills this iteration.
    let timings: Vec<ActiveSeqTiming> = active
        .iter()
        .map(|a| ActiveSeqTiming {
            last_token_time: a.last_token_time,
        })
        .collect();

    if g.requests.is_empty() || !policy.should_prefill(&timings) {
        return Vec::new();
    }

    // Account for both active and in-progress prefilling sequences.
    let cap = max_batch_size.saturating_sub(active.len() + prefilling.len());

    let infos: Vec<PendingRequestInfo> = g
        .requests
        .iter()
        .enumerate()
        .map(|(i, req)| PendingRequestInfo {
            prompt_len: req.prompt_len(),
            index: i,
        })
        .collect();
    let selected = policy.select_prefills(&infos, cap);

    // Remove selected indices from pending (reverse order to preserve indices).
    let mut remove_indices = selected.clone();
    remove_indices.sort_unstable_by(|a, b| b.cmp(a));
    let mut taken: Vec<(usize, InferenceRequest)> = Vec::with_capacity(selected.len());
    for idx in remove_indices {
        taken.push((idx, g.requests.remove(idx)));
    }

    // Re-sort into policy-selected order.
    let mut result = Vec::with_capacity(selected.len());
    for &sel_idx in &selected {
        let pos = taken.iter().position(|(i, _)| *i == sel_idx).unwrap();
        let (_, req) = taken.swap_remove(pos);
        result.push(req);
    }
    result
}

/// Retire finished sequences. After swap_remove, the last element moves to
/// position i. Compact its SSM states to match its new slot index so CUDA
/// graph addresses remain valid (active sequences must occupy contiguous
/// slots [0..N)).
///
/// CRITICAL: compact_sequence MUST run BEFORE finish_sequence (BUG #35).
pub(super) fn retire_finished_sequences(model: &dyn Model, active: &mut Vec<ActiveSeq>) {
    let mut i = 0;
    while i < active.len() {
        if active[i].finished {
            let mut a = active.swap_remove(i);
            if i < active.len() && active[i].seq.slot_idx != i {
                // Compact the swapped-in sequence to reuse the retired
                // seq's slot. Mark the retired seq's slot as reused so
                // free_sequence doesn't double-release it.
                if let Err(e) = model.compact_sequence(&mut active[i].seq, i) {
                    tracing::error!("compact_sequence: {e:#}");
                }
                a.seq.slot_idx = usize::MAX; // sentinel: slot reused by compact
            }
            finish_sequence(model, &mut a);
        } else {
            i += 1;
        }
    }
}
