// SPDX-License-Identifier: AGPL-3.0-only

//! Responsibility: the `mtp_width_policy` row: what the MTP throughput gate
//! (`mtp_gate.rs`) remembers per batch width, and how it paces its probes of
//! the mode it is not running. Off by default on every target; with it off the
//! gate holds no [`WidthMemory`] and runs exactly as before.
//!
//! # What goes wrong without it
//!
//! The gate arbitrates MTP against plain decode on delivered tokens per
//! second. Three of its rules were written for one stream:
//!
//! * its probe intervals count AGGREGATE tokens (re-probe MTP every 256 while
//!   in Serial, refresh the serial baseline every 1024 while in Mtp), and a
//!   probe is one 16-step window whatever the width;
//! * a change of power-of-two batch width marks both baselines stale and
//!   pulls a probe forward, and nothing of the old width is kept;
//! * a probe that confirms the standing mode is repeated at the same interval
//!   for ever.
//!
//! At 8 streams a decode step emits 8 tokens, so 256 tokens is 32 steps. The
//! losing mode is then probed every 32 steps for 16 steps. Tier 0 on the RTX
//! 4060 Ti 16 GB (Qwen3.5-4B, FP8 FFN, one draft, 2026-11-12) shows exactly
//! that cycle: an MTP step at 8 streams costs 99 ms for 13.4 tokens (135
//! tok/s), a decode step 39 to 40 ms for 8 (200 tok/s), the gate does choose
//! Serial, and then spends 16 of every 48 steps (55% of the wall time)
//! re-measuring MTP. 256 tokens in 1.28 s plus 214 tokens in 1.58 s is 164
//! tok/s, which is what the serve delivered (157 to 165) against 197 with no
//! `--speculative` at all. Each wave of requests also walks the width
//! 1, 2, 4, 8 and back, and every step of that walk discards what the gate
//! knew.
//!
//! D-Cut does not cover this (v1 or the cost model, `mtp_dcut.rs`): it keeps
//! at least one draft per sequence by construction, so it can thin a deep
//! draft but never turn drafting off, and with one draft it has nothing to
//! cut. Zero drafts is the gate's decision alone.
//!
//! # The policy
//!
//! 1. **One memory per width.** The gate's standing mode, both baselines, the
//!    depth they were measured at and the probe pacing are saved under the
//!    power-of-two width bucket when the width changes, and restored when
//!    that width comes back. A width seen before resumes its decision without
//!    a probe; an unseen width behaves as before (stale, probe pulled
//!    forward). The depth rule still applies to a restored bucket, because
//!    its `measured_at_depth` is restored with it.
//! 2. **Intervals per stream.** The re-probe and refresh intervals are
//!    multiplied by the width bucket, so they count tokens per stream and the
//!    probe duty cycle at any width is the single-stream one.
//! 3. **Backoff.** A probe that confirms the standing mode doubles the next
//!    interval, up to [`MAX_BACKOFF_LOG2`] doublings; a probe that favors the
//!    other mode, a switch, or a new depth or width regime resets it. A
//!    resumed width keeps at most [`RESUME_BACKOFF_LOG2`].
//!
//! Draft depth above one stays where it is decided today: D-Cut ranks the
//! optional drafts of a batch by survival and, with `mtp_dcut_cost_model`,
//! keeps the number that maximizes expected tokens per second for the live
//! row count. Together the two cover drafts per stream from zero (this file)
//! to the configured depth (D-Cut) as a function of the live batch and the
//! measured acceptance.
//!
//! # What it does not change
//!
//! Both modes emit verified tokens, so outputs are those of whichever mode
//! runs. The interleaving of the two modes differs from the row off, so a
//! greedy run is not bit-identical to the row off wherever the two forwards
//! differ (the batch-K numerics floor the entry pin exists for).

use std::sync::OnceLock;

/// Width buckets remembered: the power-of-two regimes 1, 2, 4, ..., 2^15.
pub(super) const BUCKETS: usize = 16;
/// Most doublings of a probe interval that confirming probes can earn (8x).
pub(super) const MAX_BACKOFF_LOG2: u32 = 3;
/// Doublings a resumed width keeps (4x), so a width whose streams have all
/// changed since it was confirmed is looked at again sooner than one that
/// never left.
pub(super) const RESUME_BACKOFF_LOG2: u32 = 2;

/// Is the `mtp_width_policy` row on (`AVAROK_MTP_WIDTH_POLICY`)? Read once.
pub(super) fn row() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(spark_model::layers::ops::port_rows::mtp_width_policy::row)
}

/// The bucket index of a width regime (`width.next_power_of_two()`), clamped
/// to the last bucket for widths past 2^15.
pub(super) fn bucket(regime: usize) -> usize {
    (regime.max(1).trailing_zeros() as usize).min(BUCKETS - 1)
}

/// A probe interval in aggregate tokens: `base` tokens per stream at width
/// regime `regime`, doubled once per confirming probe up to the cap.
pub(super) fn interval(base: usize, regime: usize, confirms: u32) -> usize {
    base.saturating_mul(regime.max(1))
        .saturating_mul(1usize << confirms.min(MAX_BACKOFF_LOG2))
}

/// What the gate keeps per width bucket, and the backoff of the live one.
///
/// Generic over the snapshot so the gate's private state stays private to it
/// and this file stays testable on its own.
pub(super) struct WidthMemory<S> {
    slots: [Option<S>; BUCKETS],
    /// Consecutive probes that confirmed the standing mode at the live width.
    confirms: u32,
}

impl<S> WidthMemory<S> {
    pub(super) fn new() -> Self {
        Self {
            slots: std::array::from_fn(|_| None),
            confirms: 0,
        }
    }

    /// Keep `snapshot` as what is known about width regime `regime`.
    pub(super) fn save(&mut self, regime: usize, snapshot: S) {
        self.slots[bucket(regime)] = Some(snapshot);
    }

    /// What is known about width regime `regime`, removed from the memory
    /// (the gate holds it live until the width changes again).
    pub(super) fn take(&mut self, regime: usize) -> Option<S> {
        self.slots[bucket(regime)].take()
    }

    /// Doublings in force at the live width.
    pub(super) fn confirms(&self) -> u32 {
        self.confirms
    }

    /// Restore the live width's doublings from a snapshot, capped for a
    /// resumed width.
    pub(super) fn resume(&mut self, saved_confirms: u32) {
        self.confirms = saved_confirms.min(RESUME_BACKOFF_LOG2);
    }

    /// A probe ended: `confirmed` when both baselines were fresh and the
    /// standing mode stayed ahead.
    pub(super) fn note_probe(&mut self, confirmed: bool) {
        self.confirms = if confirmed {
            (self.confirms + 1).min(MAX_BACKOFF_LOG2)
        } else {
            0
        };
    }

    /// A switch or a new regime: the next probe comes at the base interval.
    pub(super) fn reset_backoff(&mut self) {
        self.confirms = 0;
    }
}

#[cfg(test)]
#[path = "mtp_width_policy_tests.rs"]
mod tests;
