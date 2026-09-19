// SPDX-License-Identifier: AGPL-3.0-only

//! Scheduling policy trait (SDD: FIFO vs SLAI).
//!
//! Controls two decisions in the scheduler loop:
//! 1. Whether to accept new prefills or prioritize decode (TBT deadline).
//! 2. Which pending requests to prefill and in what order.
//!
//! Implementations:
//! - [`FifoPolicy`]: always prefill, take first N from queue (current behavior).
//! - [`SlaiPolicy`]: skip prefills when active sequences approach TBT deadline,
//!   select shortest prompts first from ALL pending (SLAI — arXiv:2407.08353).

use std::time::{Duration, Instant};

/// Metadata about a pending request for selection decisions.
pub struct PendingRequestInfo {
    /// Number of prompt tokens (determines prefill cost).
    pub prompt_len: usize,
    /// Index into the full pending requests vec.
    pub index: usize,
}

/// Per-sequence timing for decode urgency decisions.
pub struct ActiveSeqTiming {
    /// When the last token was emitted for this sequence.
    pub last_token_time: Instant,
}

/// Scheduling policy controlling prefill admission and ordering.
pub trait SchedulingPolicy: Send {
    /// Whether to accept new prefills this iteration.
    ///
    /// Returns `false` to skip prefill and proceed directly to decode
    /// (e.g., when active sequences approach their TBT deadline).
    fn should_prefill(&self, active_timings: &[ActiveSeqTiming]) -> bool;

    /// Select up to `capacity` requests from ALL pending, in prefill order.
    ///
    /// Returns indices into `requests` for the selected items, ordered
    /// by desired prefill execution order. FIFO takes the first N;
    /// SLAI picks the N shortest prompts.
    fn select_prefills(&self, requests: &[PendingRequestInfo], capacity: usize) -> Vec<usize>;

    /// Number of prefill tokens to inject this iteration when fusing a
    /// prefill chunk into a decode step ("always-mixed" path).
    ///
    /// Returns a token budget in `[0, full_chunk]`:
    /// - `full_chunk` when no decode is active, or under moderate decode
    ///   pressure — fuse the WHOLE chunk (measured: shrinking the slice does
    ///   not lower decode TBT because the fused step's full-forward floor
    ///   dominates; it only slows prefill).
    /// - `0` ONLY as a hard suppress when a decode has already blown its
    ///   TBT deadline — the caller must then run decode-only this tick.
    fn prefill_slice_budget(&self, active_timings: &[ActiveSeqTiming], full_chunk: usize) -> usize {
        // Default (FIFO / unaware): inject the full chunk — same as today.
        let _ = active_timings;
        full_chunk
    }

    /// Policy name for logging.
    fn name(&self) -> &str;
}

/// FIFO scheduling: always prefill, take first N from queue.
pub struct FifoPolicy;

impl SchedulingPolicy for FifoPolicy {
    fn should_prefill(&self, _active_timings: &[ActiveSeqTiming]) -> bool {
        true
    }

    fn select_prefills(&self, requests: &[PendingRequestInfo], capacity: usize) -> Vec<usize> {
        // First N in queue order (FIFO).
        (0..requests.len().min(capacity)).collect()
    }

    fn name(&self) -> &str {
        "fifo"
    }
}

/// SLO-aware scheduling (SLAI-inspired).
///
/// - Skips prefills when any active sequence waited > 80% of `tbt_deadline`
///   since its last token emission (decode-first priority).
/// - Selects the N shortest prompts from ALL pending (reduces median TTFT).
pub struct SlaiPolicy {
    tbt_deadline: Duration,
}

impl SlaiPolicy {
    pub fn new(tbt_deadline_ms: u64) -> Self {
        Self {
            tbt_deadline: Duration::from_millis(tbt_deadline_ms),
        }
    }
}

impl SchedulingPolicy for SlaiPolicy {
    fn should_prefill(&self, active_timings: &[ActiveSeqTiming]) -> bool {
        if active_timings.is_empty() {
            return true;
        }
        let now = Instant::now();
        let margin = self.tbt_deadline.mul_f64(0.8);
        for timing in active_timings {
            if now.duration_since(timing.last_token_time) >= margin {
                return false;
            }
        }
        true
    }

    /// Shortest-job-first over ALL pending, with ONE seat reserved for the
    /// head of the queue.
    ///
    /// ★ Why the reservation exists. Pure SJF has no wait-time term, and
    /// `requests` is rebuilt from the pending queue every tick, so a long
    /// prompt is re-sorted to the tail on every tick it loses. Under a
    /// steady arrival of shorter requests — the ordinary shape of a busy
    /// serve — it is never selected at all. That is an UNBOUNDED wait, not
    /// a slow one: no amount of elapsed time makes it eligible, because
    /// nothing in the ranking key ever changes.
    ///
    /// The queue is in ARRIVAL order (the scheduler enumerates
    /// `PendingQueue::requests` directly, and admission re-inserts its
    /// overflow at the FRONT preserving relative order — `admission.rs`),
    /// so index 0 is the OLDEST pending request. Always taking it gives a
    /// hard bound: every request advances at least one position per
    /// selecting tick, so a request at position `p` waits at most `p`
    /// ticks. Aging by wall-clock would need a timestamp plumbed through
    /// [`PendingRequestInfo`] and would make this decision clock-dependent
    /// (it is currently pure, and tested as such); the positional bound
    /// needs neither and cannot be tuned wrong.
    ///
    /// The remaining `capacity - 1` seats are still filled shortest-first,
    /// which is where SLAI's median-TTFT win comes from — the reservation
    /// costs one seat per tick, and only when a request is actually queued
    /// behind others.
    fn select_prefills(&self, requests: &[PendingRequestInfo], capacity: usize) -> Vec<usize> {
        if capacity == 0 || requests.is_empty() {
            return Vec::new();
        }
        // Seat 0: the oldest pending request, unconditionally.
        let mut indices: Vec<usize> = vec![0];
        // Remaining seats: shortest-first over everything else.
        let mut rest: Vec<usize> = (1..requests.len()).collect();
        rest.sort_by_key(|&i| requests[i].prompt_len);
        rest.truncate(capacity - 1);
        indices.extend(rest);
        indices
    }

    fn prefill_slice_budget(&self, active_timings: &[ActiveSeqTiming], full_chunk: usize) -> usize {
        // No decode active → no TBT pressure → inject the full chunk.
        if active_timings.is_empty() {
            return full_chunk;
        }

        // Hard suppress: if ANY decode has already blown its TBT deadline,
        // return 0 so the caller runs decode-only this tick — let the late
        // decode catch up (a fused step would make it wait a whole forward).
        let now = Instant::now();
        let worst = active_timings
            .iter()
            .map(|t| now.duration_since(t.last_token_time))
            .max()
            .unwrap_or_default();
        if worst >= self.tbt_deadline {
            return 0;
        }

        // Otherwise fuse the FULL chunk. Measured (varied-load burst A/B,
        // 2026-06-24): the fused step has a ~250ms full-forward floor that
        // DOMINATES, so SHRINKING the prefill slice does NOT lower decode TBT
        // — it only multiplies 250ms steps and slows prefill (slice=32 → 4.6×
        // slower prefill for no TBT gain). A full-chunk slice keeps prefill at
        // flag-off speed AND still halves the decode-freeze p99 (2529→1285ms)
        // by riding decode on every chunk. So: fuse decode into the normal
        // chunk, never shrink it. (The earlier cost-driven/EWMA shrink was the
        // wrong lever for this model and was removed.)
        full_chunk
    }

    fn name(&self) -> &str {
        "slai"
    }
}

#[cfg(test)]
#[path = "scheduling_policy_tests.rs"]
mod tests;
