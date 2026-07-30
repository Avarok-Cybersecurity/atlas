// SPDX-License-Identifier: AGPL-3.0-only

//! Run telemetry — speculation acceptance and decode timing.
//!
//! Nineteen `AtomicU64` statics across `verify_k2_step`, `verify_k3_step` and
//! `verify_k4_step` counted draft acceptance and per-position draft matches,
//! and periodically logged a summary. They are model-scoped in the way that
//! matters for a diagnostic: acceptance rate is a property of *a* model's
//! drafter against *a* model's verifier. Accumulated in process globals they
//! would blend two models' rates after a swap, and the periodic summary would
//! report a number that describes neither.
//!
//! Nothing here changes generation, so a stale count is not wrong output — it
//! is a wrong *measurement*, which for telemetry is the same kind of failure.
//!
//! Kept as atomics because the counters are genuinely mutated from the decode
//! path through a shared `&SchedCtx`; the change is that they live inside the
//! run rather than the process.

use std::sync::atomic::{AtomicU64, Ordering};

/// How many verify steps between summary log lines.
pub const SUMMARY_PERIOD: u64 = 512;

/// Counters for one chain width. `k2` uses only `accepts`/`rejects`; the wider
/// chains also bucket by how many drafts were accepted.
#[derive(Debug, Default)]
pub struct SpecStats {
    /// One-shot log latches for this run, keyed by a `&'static str`.
    fired: std::sync::Mutex<std::collections::BTreeSet<&'static str>>,
    // ── K=2 ──
    pub k2_accepts: AtomicU64,
    pub k2_rejects: AtomicU64,

    // ── K=3: acceptance buckets + per-position draft matches ──
    pub k3_accept: [AtomicU64; 3],
    pub k3_steps: AtomicU64,
    pub k3_d1_match: AtomicU64,
    pub k3_d2_match_uncond: AtomicU64,
    pub k3_d2_match_cond: AtomicU64,

    // ── K=4 ──
    pub k4_accept: [AtomicU64; 4],
    pub k4_steps: AtomicU64,
    pub k4_d1: AtomicU64,
    pub k4_d2_uncond: AtomicU64,
    pub k4_d3_uncond: AtomicU64,
    pub k4_d2_cond: AtomicU64,
    pub k4_d3_cond: AtomicU64,

    /// B1 drift gauge: decode positions inside a parameter body whose
    /// top1-top2 gap is below the low-margin threshold. A count of one
    /// model's argmax-flip exposure — summed across a swap it describes
    /// neither, and the periodic WARN would name a number belonging to both.
    pub b1_low_margin: AtomicU64,

    // ── Decode timing (ATLAS_DECODE_TIMING) ──
    pub decode_copy_us: AtomicU64,
    pub decode_sample_us: AtomicU64,
    pub decode_count: AtomicU64,
}

/// Bump a counter. Free-standing so call sites read as
/// `spec_stats::bump(&sched.stats.k4_steps)`.
#[inline]
pub fn bump(counter: &AtomicU64) -> u64 {
    counter.fetch_add(1, Ordering::Relaxed)
}

#[inline]
pub fn get(counter: &AtomicU64) -> u64 {
    counter.load(Ordering::Relaxed)
}

#[inline]
pub fn reset(counter: &AtomicU64) {
    counter.store(0, Ordering::Relaxed);
}

impl SpecStats {
    pub fn new() -> Self {
        Self::default()
    }

    /// `true` the first time THIS run reaches `key`, `false` after.
    ///
    /// The scheduler's counterpart to `ModelStats::once`. The lines it gates
    /// say which decode path a run engaged; a `static Once` meant the second
    /// model in a process ran with no record of its own.
    pub fn once(&self, key: &'static str) -> bool {
        self.fired.lock().expect("run latches poisoned").insert(key)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_fresh_run_starts_at_zero() {
        let s = SpecStats::new();
        assert_eq!(get(&s.k4_steps), 0);
        assert_eq!(get(&s.k2_accepts), 0);
        assert!(s.k3_accept.iter().all(|c| get(c) == 0));
    }

    #[test]
    fn two_runs_count_independently() {
        // The property nineteen process globals could not have: after a model
        // swap, the new run's acceptance rate must describe the new model.
        let a = SpecStats::new();
        let b = SpecStats::new();
        for _ in 0..7 {
            bump(&a.k4_steps);
        }
        assert_eq!(get(&a.k4_steps), 7);
        assert_eq!(get(&b.k4_steps), 0, "a second run starts clean");
    }

    #[test]
    fn reset_clears_a_counter_for_the_next_summary_window() {
        let s = SpecStats::new();
        bump(&s.k3_steps);
        bump(&s.k3_steps);
        assert_eq!(get(&s.k3_steps), 2);
        reset(&s.k3_steps);
        assert_eq!(get(&s.k3_steps), 0);
    }
}
