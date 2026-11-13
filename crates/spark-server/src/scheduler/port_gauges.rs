// SPDX-License-Identifier: AGPL-3.0-only

//! Responsibility: the scheduler's steady-state gauges for the benchmark
//! harness (`atlas_scheduler_running`, `atlas_scheduler_prefilling`,
//! `atlas_decode_rows_last_round`) and the one `/metrics` entry point for the
//! scheduler-side port series (`render_prometheus`).
//!
//! # Why
//!
//! A steady-state throughput number is only meaningful over intervals in which
//! the scheduler really held C running sequences, no prefill, and one decode
//! row per sequence. The harness scrapes these gauges once a second and counts
//! only such intervals (`steady.py`); without them a wave's ramp-up and
//! drain-down are averaged into the result.
//!
//! # Semantics
//!
//! * `running`: sequences in the decode batch at the end of the last scheduler
//!   iteration (after retirement). `prefilling`: prompts mid-prefill.
//! * `rows`: decode rows of the last decode round. A plain decode step has one
//!   row per sequence; an MTP step has `sum(verified drafts + 1)` (bootstrap
//!   sequences one), noted by `mtp_stats::account_step`.
//! * A gauge older than [`FRESH_MS`] reads 0: the scheduler blocks on its
//!   request condvar when idle, and a value frozen at the last busy tick would
//!   look like a steady interval to a scraper. Zero is the conservative error
//!   (that interval is simply not counted).
//!
//! Plain relaxed atomics on the scheduler thread; one `Instant` read per round.

use std::sync::LazyLock;
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use std::time::Instant;

use crate::api::port_metrics as pm;

/// A gauge not refreshed within this many milliseconds reads 0.
pub const FRESH_MS: u64 = 1_500;
/// `spec_rows` value meaning "no speculative round noted this iteration".
const NO_SPEC: u32 = u32::MAX;

static EPOCH: LazyLock<Instant> = LazyLock::new(Instant::now);

fn now_ms() -> u64 {
    u64::try_from(EPOCH.elapsed().as_millis()).unwrap_or(u64::MAX - 1) + 1
}

fn clamp(n: usize) -> u32 {
    u32::try_from(n).unwrap_or(u32::MAX - 1)
}

/// The gauge cells. The process has one ([`GAUGES`]); tests build their own.
pub struct Gauges {
    running: AtomicU32,
    prefilling: AtomicU32,
    rows: AtomicU32,
    spec_rows: AtomicU32,
    /// Milliseconds since `EPOCH` of the last update, plus one (0 = never).
    stamp_ms: AtomicU64,
}

impl Default for Gauges {
    fn default() -> Self {
        Self::new()
    }
}

impl Gauges {
    pub const fn new() -> Self {
        Self {
            running: AtomicU32::new(0),
            prefilling: AtomicU32::new(0),
            rows: AtomicU32::new(0),
            spec_rows: AtomicU32::new(NO_SPEC),
            stamp_ms: AtomicU64::new(0),
        }
    }

    /// A speculative (MTP) round ran with `rows` verify plus bootstrap rows.
    /// The next [`Self::observe_round`] reports it instead of the plain count.
    pub fn note_spec_rows(&self, rows: usize) {
        self.spec_rows.store(clamp(rows), Ordering::Relaxed);
    }

    /// One decode round finished. `rows_plain` is the row count if it was a
    /// plain decode step (one per sequence); a noted speculative round wins.
    pub fn observe_round(&self, running: usize, prefilling: usize, rows_plain: usize, now: u64) {
        let spec = self.spec_rows.swap(NO_SPEC, Ordering::Relaxed);
        let rows = if spec == NO_SPEC {
            clamp(rows_plain)
        } else {
            spec
        };
        self.running.store(clamp(running), Ordering::Relaxed);
        self.prefilling.store(clamp(prefilling), Ordering::Relaxed);
        self.rows.store(rows, Ordering::Relaxed);
        self.stamp_ms.store(now, Ordering::Relaxed);
    }

    /// End of a scheduler iteration, after retirement: refresh the occupancy.
    /// With nothing running the last round's rows are stale, so they clear.
    pub fn end_tick(&self, running: usize, prefilling: usize, now: u64) {
        self.running.store(clamp(running), Ordering::Relaxed);
        self.prefilling.store(clamp(prefilling), Ordering::Relaxed);
        if running == 0 {
            self.rows.store(0, Ordering::Relaxed);
            self.spec_rows.store(NO_SPEC, Ordering::Relaxed);
        }
        self.stamp_ms.store(now, Ordering::Relaxed);
    }

    /// `(running, prefilling, rows)` as a scrape at `now` sees them: zeros once
    /// stale (or never written).
    pub fn read(&self, now: u64) -> (u32, u32, u32) {
        let stamp = self.stamp_ms.load(Ordering::Relaxed);
        if stamp == 0 || now.saturating_sub(stamp) > FRESH_MS {
            return (0, 0, 0);
        }
        (
            self.running.load(Ordering::Relaxed),
            self.prefilling.load(Ordering::Relaxed),
            self.rows.load(Ordering::Relaxed),
        )
    }
}

static GAUGES: Gauges = Gauges::new();

/// A speculative round just ran with `rows` rows (see [`Gauges::note_spec_rows`]).
pub fn note_spec_rows(rows: usize) {
    GAUGES.note_spec_rows(rows);
}

/// One decode round finished (see [`Gauges::observe_round`]).
pub fn observe_round(running: usize, prefilling: usize, rows_plain: usize) {
    GAUGES.observe_round(running, prefilling, rows_plain, now_ms());
}

/// End of a scheduler iteration (see [`Gauges::end_tick`]).
pub fn end_tick(running: usize, prefilling: usize) {
    GAUGES.end_tick(running, prefilling, now_ms());
}

/// Append every scheduler-side port series to the `/metrics` text: these
/// gauges, the accepted-prefix histogram and per-position acceptance, the
/// device phase timer and the D-Cut planner. `api/port_metrics.rs` calls this
/// once.
pub fn render_prometheus(out: &mut String) {
    let (running, prefilling, rows) = GAUGES.read(now_ms());
    pm::gauge(
        out,
        "atlas_scheduler_running",
        "Sequences in the decode batch at the end of the last scheduler iteration",
        running,
    );
    pm::gauge(
        out,
        "atlas_scheduler_prefilling",
        "Prompts mid-prefill at the end of the last scheduler iteration",
        prefilling,
    );
    pm::gauge(
        out,
        "atlas_decode_rows_last_round",
        "Decode rows of the last decode round (one per sequence when not speculating)",
        rows,
    );
    super::mtp_stats::hist::render_prometheus(out);
    super::mtp_device_timing::render_prometheus(out);
    super::mtp_dcut::cost::render_prometheus(out);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rows_come_from_the_speculative_round_when_one_ran() {
        let g = Gauges::new();
        g.observe_round(8, 0, 8, 100);
        assert_eq!(
            g.read(100),
            (8, 0, 8),
            "a plain round is one row per sequence"
        );
        g.note_spec_rows(26);
        g.observe_round(8, 0, 8, 110);
        assert_eq!(
            g.read(110),
            (8, 0, 26),
            "an MTP round reports its verify rows"
        );
        g.observe_round(8, 1, 8, 120);
        assert_eq!(g.read(120), (8, 1, 8), "a note is consumed by one round");
    }

    #[test]
    fn nothing_running_clears_the_rows() {
        let g = Gauges::new();
        g.note_spec_rows(12);
        g.observe_round(4, 0, 4, 100);
        g.note_spec_rows(12);
        g.end_tick(0, 0, 110);
        assert_eq!(g.read(110), (0, 0, 0));
        // A note left over from before the drain must not leak into the next round.
        g.observe_round(2, 0, 2, 120);
        assert_eq!(g.read(120), (2, 0, 2));
    }

    #[test]
    fn a_stale_or_unwritten_gauge_reads_zero() {
        let g = Gauges::new();
        assert_eq!(g.read(10_000), (0, 0, 0), "never written");
        g.observe_round(4, 0, 4, 1_000);
        assert_eq!(
            g.read(1_000 + FRESH_MS),
            (4, 0, 4),
            "just inside the window"
        );
        assert_eq!(
            g.read(1_000 + FRESH_MS + 1),
            (0, 0, 0),
            "an idle scheduler reads zero"
        );
    }

    #[test]
    fn the_text_has_the_three_gauges_and_the_other_scheduler_series() {
        let mut s = String::new();
        render_prometheus(&mut s);
        for needle in [
            "# TYPE atlas_scheduler_running gauge",
            "# TYPE atlas_scheduler_prefilling gauge",
            "# TYPE atlas_decode_rows_last_round gauge",
            "# TYPE atlas_mtp_accepted_prefix_total counter",
            "# TYPE atlas_mtp_accept_position_total counter",
            "# TYPE atlas_mtp_device_phase_microseconds_total counter",
            "# TYPE atlas_mtp_dcut_plans_total counter",
            "# TYPE atlas_mtp_dcut_calibration gauge",
        ] {
            assert!(s.contains(needle), "missing {needle:?}");
        }
    }
}
