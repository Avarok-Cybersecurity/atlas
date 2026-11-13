// SPDX-License-Identifier: AGPL-3.0-only

//! Responsibility: the bounded view's counters (port lane L32), process-wide
//! atomics read by `spark-server`'s `scheduler::kv_view::render_prometheus`
//! into the `atlas_kv_view_*` series. All zero while the view is off.

use std::sync::atomic::{AtomicU64, Ordering};

/// Why a selection did not produce a view (the `reason` label).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SkipReason {
    /// The context fits the budget: the view would be the identity.
    ShortContext,
    /// No query row was captured for the final user message.
    NoQuery,
    /// Every query-capture buffer was held by another prefill.
    CaptureBusy,
    /// The prefill hook could not tell which sequence a stream was.
    CaptureAmbiguous,
    /// The kernel module is not in this build (every target but Ada).
    NoKernel,
    /// The model's geometry is not served (per-layer KV head counts).
    Geometry,
    /// Allocating the view's device buffers failed.
    Alloc,
    /// The core failed to score or the result was unusable.
    ScoreError,
    /// A decode or verify shape did not fit the view table arena.
    ArenaShape,
    /// A graph was captured before the view's buffers existed.
    NotReady,
    /// The pick dropped no page.
    NothingDropped,
    /// Some of the sequence's K rows were written where the sketch could not
    /// follow (a batched chunk-0 prefill with no page tables).
    SketchGap,
}

/// Number of [`SkipReason`] values.
const N_REASONS: usize = 12;

impl SkipReason {
    pub const ALL: [SkipReason; N_REASONS] = [
        SkipReason::ShortContext,
        SkipReason::NoQuery,
        SkipReason::CaptureBusy,
        SkipReason::CaptureAmbiguous,
        SkipReason::NoKernel,
        SkipReason::Geometry,
        SkipReason::Alloc,
        SkipReason::ScoreError,
        SkipReason::ArenaShape,
        SkipReason::NotReady,
        SkipReason::NothingDropped,
        SkipReason::SketchGap,
    ];

    /// The metric label value.
    pub fn label(self) -> &'static str {
        match self {
            SkipReason::ShortContext => "short_context",
            SkipReason::NoQuery => "no_query",
            SkipReason::CaptureBusy => "capture_busy",
            SkipReason::CaptureAmbiguous => "capture_ambiguous",
            SkipReason::NoKernel => "no_kernel",
            SkipReason::Geometry => "geometry",
            SkipReason::Alloc => "alloc",
            SkipReason::ScoreError => "score_error",
            SkipReason::ArenaShape => "arena_shape",
            SkipReason::NotReady => "not_ready",
            SkipReason::NothingDropped => "nothing_dropped",
            SkipReason::SketchGap => "sketch_gap",
        }
    }

    fn index(self) -> usize {
        Self::ALL.iter().position(|r| *r == self).unwrap_or(0)
    }
}

/// The counters. `GLOBAL` is the serve's; tests build their own.
pub struct Counters {
    selections: AtomicU64,
    kept: AtomicU64,
    select_nanos: AtomicU64,
    view_tokens: AtomicU64,
    active: AtomicU64,
    device_bytes: AtomicU64,
    recall_sum_micros: AtomicU64,
    recall_count: AtomicU64,
    skipped: [AtomicU64; N_REASONS],
}

impl Counters {
    pub const fn new() -> Self {
        Self {
            selections: AtomicU64::new(0),
            kept: AtomicU64::new(0),
            select_nanos: AtomicU64::new(0),
            view_tokens: AtomicU64::new(0),
            active: AtomicU64::new(0),
            device_bytes: AtomicU64::new(0),
            recall_sum_micros: AtomicU64::new(0),
            recall_count: AtomicU64::new(0),
            skipped: [const { AtomicU64::new(0) }; N_REASONS],
        }
    }

    /// A fresh selection took `nanos` and kept `recall` of the score mass.
    pub fn selected(&self, nanos: u64, recall: f64) {
        self.selections.fetch_add(1, Ordering::Relaxed);
        self.select_nanos.fetch_add(nanos, Ordering::Relaxed);
        if recall.is_finite() {
            let micros = (recall.clamp(0.0, 1.0) * 1e6).round() as u64;
            self.recall_sum_micros.fetch_add(micros, Ordering::Relaxed);
            self.recall_count.fetch_add(1, Ordering::Relaxed);
        }
    }

    /// A continuation kept the conversation's earlier selection.
    pub fn kept(&self) {
        self.kept.fetch_add(1, Ordering::Relaxed);
    }

    pub fn skip(&self, r: SkipReason) {
        self.skipped[r.index()].fetch_add(1, Ordering::Relaxed);
    }

    /// The live views: how many and the tokens they attend (at selection).
    pub fn set_live(&self, views: u64, tokens: u64) {
        self.active.store(views, Ordering::Relaxed);
        self.view_tokens.store(tokens, Ordering::Relaxed);
    }

    pub fn set_device_bytes(&self, b: u64) {
        self.device_bytes.store(b, Ordering::Relaxed);
    }

    pub fn snapshot(&self) -> Snapshot {
        let ld = |a: &AtomicU64| a.load(Ordering::Relaxed);
        Snapshot {
            selections: ld(&self.selections),
            kept: ld(&self.kept),
            select_seconds: ld(&self.select_nanos) as f64 / 1e9,
            view_tokens: ld(&self.view_tokens),
            active: ld(&self.active),
            device_bytes: ld(&self.device_bytes),
            recall_sum: ld(&self.recall_sum_micros) as f64 / 1e6,
            recall_count: ld(&self.recall_count),
            skipped: SkipReason::ALL.map(|r| (r.label(), ld(&self.skipped[r.index()]))),
        }
    }
}

impl Default for Counters {
    fn default() -> Self {
        Self::new()
    }
}

/// The serve's counters.
pub static GLOBAL: Counters = Counters::new();

/// A read of every counter.
#[derive(Clone, Debug, PartialEq)]
pub struct Snapshot {
    pub selections: u64,
    pub kept: u64,
    pub select_seconds: f64,
    pub view_tokens: u64,
    pub active: u64,
    pub device_bytes: u64,
    pub recall_sum: f64,
    pub recall_count: u64,
    pub skipped: [(&'static str, u64); N_REASONS],
}

/// The serve's counters, read.
pub fn snapshot() -> Snapshot {
    GLOBAL.snapshot()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn counters_accumulate_and_label_every_reason() {
        let c = Counters::new();
        c.selected(2_000_000_000, 0.75);
        c.selected(500_000_000, f64::NAN);
        c.skip(SkipReason::NoQuery);
        c.skip(SkipReason::NoQuery);
        c.set_live(2, 30000);
        let s = c.snapshot();
        assert_eq!(s.selections, 2);
        assert!((s.select_seconds - 2.5).abs() < 1e-9);
        assert_eq!(s.recall_count, 1);
        assert!((s.recall_sum - 0.75).abs() < 1e-9);
        assert_eq!(s.active, 2);
        assert_eq!(s.view_tokens, 30000);
        let labels: Vec<&str> = s.skipped.iter().map(|(l, _)| *l).collect();
        let mut unique = labels.clone();
        unique.sort_unstable();
        unique.dedup();
        assert_eq!(unique.len(), SkipReason::ALL.len());
        assert_eq!(
            s.skipped.iter().find(|(l, _)| *l == "no_query").unwrap().1,
            2
        );
    }
}
