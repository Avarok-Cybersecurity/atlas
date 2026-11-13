// SPDX-License-Identifier: AGPL-3.0-only

//! Responsibility: the process-wide accepted-prefix histogram and the
//! per-position acceptance counters of the MTP verify passes, with their
//! `/metrics` text.
//!
//! # Series
//!
//! * `atlas_mtp_accepted_prefix_total{len="0".."16"}`: one count per verify
//!   pass, in the bucket of the number of drafts the pass accepted (16 is "16
//!   or more"). The buckets sum to `atlas_mtp_verify_passes_total`, because
//!   both advance once per sequence per step that verified at least one draft
//!   (`mtp_stats::account_step`).
//! * `atlas_mtp_accept_position_total{pos,result}`: for draft position `pos`
//!   (1-based; 1 is the first draft), how often the verifier accepted it
//!   (`result="accept"`) and how often it was the first rejected draft
//!   (`result="reject"`). A position past the first rejection was never
//!   reached and counts as neither, so `accept / (accept + reject)` is the
//!   conditional acceptance of that depth, the quantity a depth sweep needs.
//!
//! Both are denominated in the drafts the pass actually VERIFIED, which D-Cut
//! may have pruned below the drafts proposed (`SeqDcut::verified`).
//!
//! Plain relaxed atomic adds on the scheduler thread; no sync, no allocation.

use std::sync::atomic::{AtomicU64, Ordering};

use crate::api::port_metrics as pm;

/// Longest accepted prefix counted individually; longer ones fold into it.
pub const MAX_PREFIX: usize = 16;

/// The histogram and the per-position counters.
#[derive(Debug, Default)]
pub struct AcceptHistogram {
    prefix: [AtomicU64; MAX_PREFIX + 1],
    accepted_at: [AtomicU64; MAX_PREFIX],
    rejected_at: [AtomicU64; MAX_PREFIX],
}

/// A consistent-enough copy of the counters (each read is atomic; the set is
/// not, which a monitoring scrape does not need).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HistSnapshot {
    pub prefix: [u64; MAX_PREFIX + 1],
    pub accepted_at: [u64; MAX_PREFIX],
    pub rejected_at: [u64; MAX_PREFIX],
}

impl HistSnapshot {
    /// Verify passes counted (the buckets' sum).
    pub fn passes(&self) -> u64 {
        self.prefix.iter().sum()
    }
}

impl AcceptHistogram {
    pub const fn new() -> Self {
        Self {
            prefix: [const { AtomicU64::new(0) }; MAX_PREFIX + 1],
            accepted_at: [const { AtomicU64::new(0) }; MAX_PREFIX],
            rejected_at: [const { AtomicU64::new(0) }; MAX_PREFIX],
        }
    }

    /// Count one verify pass that checked `verified` drafts and accepted
    /// `accepted` of them (clamped to `verified`).
    pub fn record(&self, verified: usize, accepted: usize) {
        let accepted = accepted.min(verified);
        self.prefix[accepted.min(MAX_PREFIX)].fetch_add(1, Ordering::Relaxed);
        for c in self.accepted_at.iter().take(accepted) {
            c.fetch_add(1, Ordering::Relaxed);
        }
        if accepted < verified && accepted < MAX_PREFIX {
            self.rejected_at[accepted].fetch_add(1, Ordering::Relaxed);
        }
    }

    pub fn snapshot(&self) -> HistSnapshot {
        let load = |c: &AtomicU64| c.load(Ordering::Relaxed);
        HistSnapshot {
            prefix: std::array::from_fn(|i| load(&self.prefix[i])),
            accepted_at: std::array::from_fn(|i| load(&self.accepted_at[i])),
            rejected_at: std::array::from_fn(|i| load(&self.rejected_at[i])),
        }
    }
}

static HIST: AcceptHistogram = AcceptHistogram::new();

/// Count one verify pass in the process-wide histogram.
pub fn record_pass(verified: usize, accepted: usize) {
    HIST.record(verified, accepted);
}

/// The process-wide counters.
pub fn snapshot() -> HistSnapshot {
    HIST.snapshot()
}

/// Append the histogram and per-position series to `out`.
pub fn render_prometheus(out: &mut String) {
    render_snapshot(&snapshot(), out);
}

/// [`render_prometheus`] over a snapshot (pure, so a test can grade the text).
pub fn render_snapshot(s: &HistSnapshot, out: &mut String) {
    let labels: Vec<String> = (0..=MAX_PREFIX).map(|n| format!("len=\"{n}\"")).collect();
    let series: Vec<(&str, u64)> = labels
        .iter()
        .map(String::as_str)
        .zip(s.prefix.iter().copied())
        .collect();
    pm::labeled_counter(
        out,
        "atlas_mtp_accepted_prefix_total",
        "MTP verify passes by the number of drafts accepted (16 = 16 or more)",
        &series,
    );
    let mut labels = Vec::with_capacity(2 * MAX_PREFIX);
    let mut values = Vec::with_capacity(2 * MAX_PREFIX);
    for pos in 0..MAX_PREFIX {
        labels.push(format!("pos=\"{}\",result=\"accept\"", pos + 1));
        values.push(s.accepted_at[pos]);
        labels.push(format!("pos=\"{}\",result=\"reject\"", pos + 1));
        values.push(s.rejected_at[pos]);
    }
    let series: Vec<(&str, u64)> = labels
        .iter()
        .map(String::as_str)
        .zip(values.iter().copied())
        .collect();
    pm::labeled_counter(
        out,
        "atlas_mtp_accept_position_total",
        "MTP drafts by 1-based position and verdict (accept, or the first reject)",
        &series,
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_buckets_sum_to_the_passes_recorded() {
        let h = AcceptHistogram::new();
        let passes = [(3, 3), (3, 0), (3, 1), (1, 1), (1, 0), (2, 2), (3, 2)];
        for &(v, a) in &passes {
            h.record(v, a);
        }
        let s = h.snapshot();
        assert_eq!(s.passes(), passes.len() as u64);
        assert_eq!(s.prefix[..4], [2, 2, 2, 1]);
        assert_eq!(s.prefix[4..].iter().sum::<u64>(), 0);
    }

    #[test]
    fn positions_count_accepts_and_the_first_reject_only() {
        let h = AcceptHistogram::new();
        // Verified 3, accepted 1: position 1 accepted, position 2 the first
        // reject, position 3 never reached.
        h.record(3, 1);
        let s = h.snapshot();
        assert_eq!(s.accepted_at[..3], [1, 0, 0]);
        assert_eq!(s.rejected_at[..3], [0, 1, 0]);
        // Fully accepted: no reject to grade.
        let h = AcceptHistogram::new();
        h.record(3, 3);
        let s = h.snapshot();
        assert_eq!(s.accepted_at[..4], [1, 1, 1, 0]);
        assert_eq!(s.rejected_at.iter().sum::<u64>(), 0);
        // First draft rejected.
        let h = AcceptHistogram::new();
        h.record(2, 0);
        let s = h.snapshot();
        assert_eq!(s.accepted_at.iter().sum::<u64>(), 0);
        assert_eq!(s.rejected_at[..2], [1, 0]);
    }

    #[test]
    fn accepted_is_clamped_to_verified_and_long_prefixes_fold() {
        let h = AcceptHistogram::new();
        h.record(2, 9);
        assert_eq!(h.snapshot().prefix[2], 1);
        // A DFlash block longer than the tracked range folds into the last bucket.
        let h = AcceptHistogram::new();
        h.record(40, 40);
        let s = h.snapshot();
        assert_eq!(s.prefix[MAX_PREFIX], 1);
        assert_eq!(s.accepted_at, [1; MAX_PREFIX]);
        assert_eq!(s.rejected_at.iter().sum::<u64>(), 0);
        // A reject beyond the tracked positions is not attributed anywhere.
        let h = AcceptHistogram::new();
        h.record(40, 20);
        let s = h.snapshot();
        assert_eq!(s.prefix[MAX_PREFIX], 1);
        assert_eq!(s.rejected_at.iter().sum::<u64>(), 0);
    }

    #[test]
    fn the_text_has_every_bucket_and_position() {
        let h = AcceptHistogram::new();
        h.record(3, 2);
        let mut out = String::new();
        render_snapshot(&h.snapshot(), &mut out);
        assert!(out.contains("# TYPE atlas_mtp_accepted_prefix_total counter"));
        assert!(out.contains("atlas_mtp_accepted_prefix_total{len=\"2\"} 1\n"));
        assert!(out.contains("atlas_mtp_accepted_prefix_total{len=\"16\"} 0\n"));
        assert!(out.contains("atlas_mtp_accept_position_total{pos=\"1\",result=\"accept\"} 1\n"));
        assert!(out.contains("atlas_mtp_accept_position_total{pos=\"3\",result=\"reject\"} 1\n"));
        let lines = out
            .lines()
            .filter(|l| l.starts_with("atlas_mtp_accepted_prefix_total{"));
        assert_eq!(lines.count(), MAX_PREFIX + 1);
    }
}
