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

/// Co-dispatch admission window: `Some(duration)` when `ATLAS_PREFILL_CODISPATCH=1`,
/// else `None`. The window length is `ATLAS_PREFILL_CODISPATCH_WINDOW_MS` (default 5).
fn codispatch_window() -> Option<std::time::Duration> {
    let on = std::env::var("ATLAS_PREFILL_CODISPATCH")
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(false);
    if !on {
        return None;
    }
    let ms = std::env::var("ATLAS_PREFILL_CODISPATCH_WINDOW_MS")
        .ok()
        .and_then(|v| v.parse::<u64>().ok())
        .unwrap_or(10);
    Some(std::time::Duration::from_millis(ms))
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
        // Block until signalled (no busy-wait, no polling). Also wake on a
        // pending rotation: a quiescence-applied LoraCommand (Rotate / Promote /
        // PromoteDisk) is pushed onto `g.rotations` by the rotation forwarder and
        // notified on this same Condvar. Without `rotations.is_empty()` in the
        // predicate the notify would wake us but the loop would immediately
        // re-sleep (requests still empty), starving the quiescence-apply block —
        // an idle-scheduler deadlock for demand-driven promotion.
        while g.requests.is_empty() && g.rotations.is_empty() && !g.closed {
            cv.wait(&mut g);
        }
        if g.closed && g.requests.is_empty() {
            return Vec::new();
        }
        // Rotation-only wakeup: we exited the wait with no requests but a pending
        // rotation. Return an empty batch so the caller reaches the
        // quiescence-apply block (active/prefilling/new_reqs/swapped all empty)
        // and drains `g.rotations` at true quiescence. Do NOT fall into the
        // co-dispatch window with zero requests.
        if g.requests.is_empty() {
            return Vec::new();
        }
        // Co-dispatch micro-batch window (ATLAS_PREFILL_CODISPATCH=1): when idle,
        // gather a whole concurrent BURST into one forward (batched via
        // run_batched_prefill_step) rather than stopping at the 2nd request — a
        // 4-request burst used to split into 2+2 because the loop exited at len==2.
        // Keep collecting up to `max_batch_size`, bounded by `window`, and dispatch
        // EARLY once the burst settles (>=2 gathered and no new arrival within
        // SETTLE) so latency stays low. A lone request pays at most `window` TTFT.
        if g.requests.len() < max_batch_size
            && let Some(window) = codispatch_window()
        {
            const SETTLE: std::time::Duration = std::time::Duration::from_millis(2);
            let deadline = std::time::Instant::now() + window;
            while g.requests.len() < max_batch_size && !g.closed {
                let now = std::time::Instant::now();
                if now >= deadline {
                    break;
                }
                let wait = (deadline - now).min(SETTLE);
                let res = cv.wait_for(&mut g, wait);
                // Timed out with no new arrival in SETTLE → burst drained; dispatch
                // what we have (if it's batchable). Otherwise keep gathering.
                if res.timed_out() && g.requests.len() >= 2 {
                    break;
                }
            }
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
///
/// Under v2 EP (`ep_protocol_v2`) the worker pre-allocates every slot at
/// startup and the head-worker mirror is keyed by `slot_idx`, not by the
/// active-set position. Moving SSM states on the head only would leave
/// the worker's mirror at the original slot — the next op against that
/// seq would address different physical memory on each rank. The retired
/// seq also can't be tagged with `usize::MAX` because that sentinel
/// becomes `0xFFFFFFFF` when cast to a u32 seq_id and trips the worker's
/// bounds check on the next `0xFFFFFFF1` broadcast. So v2 skips both
/// the compaction and the sentinel and lets the active vec be
/// non-contiguous w.r.t. `slot_idx` — pre-allocated slots stay valid
/// in place across the swap_remove, and the per-slot CUDA graph cache
/// stays warm because the seq never moved.
pub(super) fn retire_finished_sequences(model: &dyn Model, active: &mut Vec<ActiveSeq>) {
    if model.ep_protocol_v2() {
        // v2 EP: slots are pre-allocated and kept in place (see doc above);
        // just drop finished seqs, no compaction.
        let mut i = 0;
        while i < active.len() {
            if active[i].finished {
                let mut a = active.swap_remove(i);
                finish_sequence(model, &mut a);
            } else {
                i += 1;
            }
        }
        return;
    }

    // ── Two-phase retirement (bug-2 fix) ──
    // The old per-removal "compact swapped-in seq to position i + detach the
    // retired seq" ASSUMED the active vec was contiguous (slot_idx == position).
    // When co-dispatch admission left it non-contiguous, that compacted a
    // survivor onto a slot still owned by another live seq (double-own) while
    // leaking the retired seq's real slot — two co-dispatched seqs then shared
    // one SSM slot → shared GDN h_state → cross-stream content bleed. The fix
    // is order-independent and exclusivity-safe:
    //   Phase 1: drop every finished seq, releasing ITS OWN slot to the pool.
    //   Phase 2: compact survivors into contiguous slots [0..n), each migration
    //            target CLAIMED exclusively from the free list (compact_sequence
    //            → claim_specific). No two live seqs can ever share a slot.

    // Phase 1.
    let mut survivors: Vec<ActiveSeq> = Vec::with_capacity(active.len());
    for mut a in active.drain(..) {
        if a.finished {
            finish_sequence(model, &mut a); // RAII guard releases a's own slot
        } else {
            survivors.push(a);
        }
    }

    // Phase 2: compact survivors back into contiguous slots [0..n).
    compact_survivors_into_range(model, &mut survivors);
    *active = survivors;
}

/// Compact live sequences into contiguous SSM slots `[0..n)` (n = the slice
/// length), claiming each migration target exclusively from the free list so
/// no two live sequences can ever share a slot.
///
/// This is the exclusivity-safe core shared by `retire_finished_sequences`
/// (Phase 2) and `swap_out_sequence`: every sequence whose `slot_idx` is out
/// of the `[0..n)` range is migrated onto a free slot in that range (a slot
/// not held by any surviving sequence — i.e. one freed by a just-retired /
/// swapped-out sequence, or never occupied). There are exactly as many free
/// targets as out-of-range survivors, so each lands on a unique slot.
///
/// PRECONDITION: any slot being vacated (a retired/swapped-out sequence's
/// slot) must already be released to the pool before this runs, so it is
/// available as a target. Never call this under `ep_protocol_v2()` — v2
/// keeps slots pinned in place (see `retire_finished_sequences`).
pub(super) fn compact_survivors_into_range(model: &dyn Model, survivors: &mut [ActiveSeq]) {
    let n = survivors.len();
    let occupied: std::collections::HashSet<usize> =
        survivors.iter().map(|a| a.seq.slot_idx).collect();
    let mut free_targets: Vec<usize> = (0..n).filter(|s| !occupied.contains(s)).collect();
    for a in survivors.iter_mut() {
        if a.seq.slot_idx >= n {
            match free_targets.pop() {
                Some(target) => {
                    if let Err(e) = model.compact_sequence(&mut a.seq, target) {
                        tracing::error!("compact_sequence: {e:#}");
                    }
                }
                None => tracing::error!(
                    "compact_survivors_into_range: no free target for out-of-range \
                     slot {} (n={n})",
                    a.seq.slot_idx
                ),
            }
        }
    }
}
