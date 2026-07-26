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
    /// For staleness detection: a snapshot older than a few seconds means the
    /// scheduler thread is wedged or the server is idle-parked.
    pub published_at: Instant,
}

static SNAP: Mutex<Option<SchedulerSnapshot>> = Mutex::new(None);

/// Publish the latest snapshot (scheduler thread, once per loop tick).
pub fn publish(s: SchedulerSnapshot) {
    *SNAP.lock() = Some(s);
}

/// Read the latest snapshot, if the scheduler has published one yet.
pub fn read() -> Option<SchedulerSnapshot> {
    *SNAP.lock()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn publish_read_roundtrip() {
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
            published_at: Instant::now(),
        };
        publish(s);
        let r = read().expect("published");
        assert_eq!(r.active_seqs, 3);
        assert_eq!(r.mtp_mode, MtpModeSnap::Mtp);
    }
}
