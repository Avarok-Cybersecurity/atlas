// SPDX-License-Identifier: AGPL-3.0-only

//! Per-tick scheduler observability snapshot.
//!
//! The model + all scheduler state live on a dedicated OS thread and are
//! thread-local by design; nothing outside could observe active sequences, KV
//! pressure, or the MTP gate. This module publishes one small `Copy` struct
//! per loop tick — a single uncontended `parking_lot` lock + memcpy, which is
//! unmeasurable against a multi-millisecond decode step — for the TUI (and
//! any future observer) to read.
//!
//! Global-static publication (not a channel) matches the established pattern
//! of `scheduler/helpers.rs` and avoids widening `scheduler::run`'s already
//! huge signature.

use std::time::Instant;

use parking_lot::Mutex;

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum MtpModeSnap {
    Mtp,
    Serial,
    Probing,
    /// Speculation disabled/not applicable (no gate constructed).
    Off,
}

#[derive(Clone, Copy, Debug)]
pub struct SchedulerSnapshot {
    pub active_seqs: u32,
    pub prefilling_seqs: u32,
    pub swapped_seqs: u32,
    /// Requests drained-but-not-yet-admitted at the observation point.
    pub pending_len: u32,
    pub kv_blocks_free: u32,
    pub kv_blocks_total: u32,
    pub ssm_slots_used: u32,
    pub ssm_slots_total: u32,
    pub mtp_mode: MtpModeSnap,
    /// The MTP gate's delivered-throughput EWMA for the current mode (tok/s);
    /// 0.0 when unmeasured.
    pub delivered_tps: f32,
    pub steps_total: u64,
    /// In-flight prefill progress, SUMMED over the prefilling sequences:
    /// tokens already ingested vs tokens in their prompts. Both 0 when
    /// nothing is prefilling. Aggregated rather than per-sequence so the
    /// snapshot stays `Copy` — a `Vec` here would allocate on the scheduler
    /// thread once per tick, which is exactly the cost this must not have.
    pub prefill_tokens_done: u32,
    pub prefill_tokens_total: u32,
    /// Token count (M) of the most recent prefill dispatch, and whether it
    /// took a fused/batched large-M path. This is how you see from the UI
    /// whether the fused arm actually engaged, rather than inferring it from
    /// throughput.
    pub prefill_chunk_width: u32,
    pub prefill_fused: bool,
    /// For staleness detection: a snapshot older than a few seconds means the
    /// scheduler thread is wedged or the server is idle-parked.
    pub published_at: Instant,
}

/// The run's snapshot cell: the scheduler writes, the dashboard reads.
///
/// A latest-value mailbox — every tick overwrites the whole snapshot, so
/// nothing accumulates. It was a static, which meant the numbers survived the
/// run that produced them: after a swap, and before the new scheduler's first
/// tick, `/health` and the dashboard reported the dead run's queue depth and
/// KV occupancy as if live. `published_at` bounded that window but did not
/// close it.
///
/// Shared as an `Arc` from `serve` — the same route `SchedLevers` takes — so
/// the cell dies with the run and a reader attached to no run sees `None`.
#[derive(Default)]
pub struct SnapshotCell {
    cell: Mutex<Option<SchedulerSnapshot>>,
    /// Last prefill dispatch width, and whether it was the fused path.
    /// Written from the prefill dispatch sites and read once per tick by
    /// `publish`. Relaxed atomics: this is observability, a torn read across
    /// the two costs a single stale UI frame and nothing else — worth far
    /// less than making the prefill path synchronise.
    last_chunk_width: std::sync::atomic::AtomicU32,
    last_chunk_fused: std::sync::atomic::AtomicBool,
}

impl SnapshotCell {
    /// Record the width of a prefill dispatch. Two relaxed stores — ~1 ns
    /// against a chunk that takes milliseconds.
    pub fn note_prefill_chunk(&self, width: usize, fused: bool) {
        use std::sync::atomic::Ordering::Relaxed;
        self.last_chunk_width.store(width as u32, Relaxed);
        self.last_chunk_fused.store(fused, Relaxed);
    }

    /// The last recorded dispatch width + fused flag.
    pub fn last_prefill_chunk(&self) -> (u32, bool) {
        use std::sync::atomic::Ordering::Relaxed;
        (
            self.last_chunk_width.load(Relaxed),
            self.last_chunk_fused.load(Relaxed),
        )
    }

    /// Publish the latest snapshot (scheduler thread, once per loop tick).
    pub fn publish(&self, s: SchedulerSnapshot) {
        *self.cell.lock() = Some(s);
    }

    /// Read the latest snapshot, if the scheduler has published one yet.
    pub fn read(&self) -> Option<SchedulerSnapshot> {
        *self.cell.lock()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn publish_read_roundtrip() {
        let cell = SnapshotCell::default();
        let s = SchedulerSnapshot {
            active_seqs: 3,
            prefilling_seqs: 1,
            swapped_seqs: 0,
            pending_len: 2,
            kv_blocks_free: 100,
            kv_blocks_total: 200,
            ssm_slots_used: 4,
            ssm_slots_total: 128,
            mtp_mode: MtpModeSnap::Mtp,
            delivered_tps: 42.5,
            steps_total: 7,
            prefill_tokens_done: 0,
            prefill_tokens_total: 0,
            prefill_chunk_width: 0,
            prefill_fused: false,
            published_at: Instant::now(),
        };
        cell.publish(s);
        let r = cell.read().expect("published");
        assert_eq!(r.active_seqs, 3);
        assert_eq!(r.mtp_mode, MtpModeSnap::Mtp);
    }
}
