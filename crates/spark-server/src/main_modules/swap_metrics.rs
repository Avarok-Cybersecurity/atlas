// SPDX-License-Identifier: AGPL-3.0-only

//! Responsibility: how long a hot-swap takes, step by step (port sprint lane
//! L41, row `model_swap_timing`): the timer `model_swap::swap` drives and the
//! counters and per-model history it fills. The `/v1/models` `x_atlas` objects
//! are in [`atlas`] and the `/metrics` text in [`render`].
//!
//! # What is measured
//!
//! `model_swap::swap` runs six steps (see its module doc). The timer groups
//! them into the phases a router cares about:
//!
//! | phase      | steps | covers                                                       |
//! |------------|-------|--------------------------------------------------------------|
//! | `drain`    | 1-3   | refuse new work, wait for in-flight holders, scheduler drain |
//! | `teardown` | 4     | stream sync, `Model::teardown`, the allocator sweep          |
//! | `load`     | 5     | `load_model` of the new checkpoint                           |
//! | `total`    | 1-6   | first step to the publish (the whole outage window)          |
//!
//! The scheduler thread owns step 4 and the join covers 3 and 4 together, so
//! the scheduler marks the start of its release with
//! [`mark_teardown_begin`] (one fragment line in `scheduler/mod.rs`). Without
//! the mark the join counts as `drain` and the `teardown` series stays empty;
//! with it `drain` ends at the mark and `teardown` runs from the mark to the
//! end of the join. A mark outside the join window (a stale one) is ignored.
//!
//! `total` is observed only for swaps that published the new model, so the
//! per-model estimate is never polluted by a failed attempt. Every swap that
//! reaches the timer is counted once in `atlas_model_swap_total{outcome}`:
//! `ok`, `restored` (the new model failed and the previous one is back),
//! `failed` (no model is loaded afterwards, or the scheduler panicked) and
//! `refused` (the drain window ran out and the old model kept serving).
//!
//! # What `/v1/models` says (only with the row on)
//!
//! Each entry gains `x_atlas`: `resident` (the model, or an adapter of it, is
//! serving now), `resident_bytes` (the base model entry only: the device bytes
//! live in the allocation ledger) and `swap_seconds_estimate` (0 for what is
//! resident, else the median of the last [`HISTORY`] timed swaps to that id,
//! absent when none was timed). The list gains a top-level `x_atlas` with
//! `resident_model` and a `swap_seconds_estimate` map of every id with
//! history, which is how a router prices a model that is not loaded. With the
//! row off none of this is added and the bytes are those of `52d6b0e9a`.
//!
//! The series (all zero with the row off): `atlas_model_swap_seconds{phase}`
//! (sum and count), `atlas_model_swap_total{outcome}`,
//! `atlas_model_swap_estimate_seconds{model}` and `atlas_model_resident_bytes`.

pub mod atlas;
pub mod render;

use std::collections::{BTreeMap, VecDeque};
use std::sync::OnceLock;
use std::sync::atomic::{AtomicU64, Ordering::Relaxed};
use std::time::{Duration, Instant};

use parking_lot::Mutex;

/// Swap durations kept per model for the estimate.
pub const HISTORY: usize = 8;
/// Models tracked at once; the least recently first-seen id is dropped past it.
pub const MAX_MODELS: usize = 32;

/// A timed phase of a swap.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Phase {
    Drain,
    Teardown,
    Load,
    Total,
}

impl Phase {
    pub const ALL: [Self; 4] = [Self::Drain, Self::Teardown, Self::Load, Self::Total];
    pub const fn label(self) -> &'static str {
        match self {
            Self::Drain => "drain",
            Self::Teardown => "teardown",
            Self::Load => "load",
            Self::Total => "total",
        }
    }
    const fn index(self) -> usize {
        self as usize
    }
}

/// How a swap that reached the timer ended.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Outcome {
    Ok,
    Restored,
    Failed,
    Refused,
}

impl Outcome {
    pub const ALL: [Self; 4] = [Self::Ok, Self::Restored, Self::Failed, Self::Refused];
    pub const fn label(self) -> &'static str {
        match self {
            Self::Ok => "ok",
            Self::Restored => "restored",
            Self::Failed => "failed",
            Self::Refused => "refused",
        }
    }
    const fn index(self) -> usize {
        self as usize
    }
}

const fn zeros() -> [AtomicU64; 4] {
    [
        AtomicU64::new(0),
        AtomicU64::new(0),
        AtomicU64::new(0),
        AtomicU64::new(0),
    ]
}

/// Everything the swap timer records. One process-wide instance ([`stats`]);
/// tests build their own so they never share state.
pub struct SwapStats {
    phase_ns: [AtomicU64; 4],
    phase_count: [AtomicU64; 4],
    outcomes: [AtomicU64; 4],
    history: Mutex<Vec<(String, VecDeque<f64>)>>,
    epoch: OnceLock<Instant>,
    /// Nanoseconds since `epoch` of the scheduler's release start, plus one;
    /// 0 means no mark.
    teardown_mark: AtomicU64,
}

/// A consistent read of the counters.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Snapshot {
    /// `(count, sum of seconds)` per [`Phase`], in `Phase::ALL` order.
    pub phases: [(u64, f64); 4],
    /// Count per [`Outcome`], in `Outcome::ALL` order.
    pub outcomes: [u64; 4],
}

impl SwapStats {
    pub const fn new() -> Self {
        Self {
            phase_ns: zeros(),
            phase_count: zeros(),
            outcomes: zeros(),
            history: Mutex::new(Vec::new()),
            epoch: OnceLock::new(),
            teardown_mark: AtomicU64::new(0),
        }
    }

    /// Add one observation of `phase`.
    pub fn observe(&self, phase: Phase, d: Duration) {
        let ns = u64::try_from(d.as_nanos()).unwrap_or(u64::MAX);
        self.phase_ns[phase.index()].fetch_add(ns, Relaxed);
        self.phase_count[phase.index()].fetch_add(1, Relaxed);
    }

    /// Observe `total` and remember it as the latest swap time of `model`.
    pub fn record_total(&self, model: &str, d: Duration) {
        self.observe(Phase::Total, d);
        let secs = d.as_secs_f64();
        let mut h = self.history.lock();
        if let Some((_, q)) = h.iter_mut().find(|(m, _)| m == model) {
            push_bounded(q, secs);
            return;
        }
        if h.len() >= MAX_MODELS {
            h.remove(0);
        }
        let mut q = VecDeque::with_capacity(HISTORY);
        q.push_back(secs);
        h.push((model.to_string(), q));
    }

    pub fn count(&self, outcome: Outcome) {
        self.outcomes[outcome.index()].fetch_add(1, Relaxed);
    }

    pub fn snapshot(&self) -> Snapshot {
        let mut s = Snapshot::default();
        for p in Phase::ALL {
            let i = p.index();
            let ns = self.phase_ns[i].load(Relaxed);
            s.phases[i] = (self.phase_count[i].load(Relaxed), ns as f64 / 1e9);
        }
        for o in Outcome::ALL {
            s.outcomes[o.index()] = self.outcomes[o.index()].load(Relaxed);
        }
        s
    }

    /// Median of the retained swap times to `model`, in seconds.
    pub fn estimate(&self, model: &str) -> Option<f64> {
        let h = self.history.lock();
        let (_, q) = h.iter().find(|(m, _)| m == model)?;
        median(q.iter().copied())
    }

    /// [`estimate`](Self::estimate) of every model with history, by id.
    pub fn estimates(&self) -> BTreeMap<String, f64> {
        let h = self.history.lock();
        h.iter()
            .filter_map(|(m, q)| Some((m.clone(), median(q.iter().copied())?)))
            .collect()
    }

    /// The retained swap times of `model`, oldest first.
    #[cfg(test)]
    pub fn durations(&self, model: &str) -> Vec<f64> {
        let h = self.history.lock();
        h.iter()
            .find(|(m, _)| m == model)
            .map(|(_, q)| q.iter().copied().collect())
            .unwrap_or_default()
    }

    fn epoch(&self) -> Instant {
        *self.epoch.get_or_init(Instant::now)
    }

    /// The scheduler is about to release its model (stream sync, teardown).
    pub fn mark_teardown(&self) {
        let ns = Instant::now()
            .saturating_duration_since(self.epoch())
            .as_nanos();
        let ns = u64::try_from(ns).unwrap_or(u64::MAX - 1);
        self.teardown_mark.store(ns + 1, Relaxed);
    }

    fn clear_mark(&self) {
        let _ = self.epoch();
        self.teardown_mark.store(0, Relaxed);
    }

    fn teardown_mark(&self) -> Option<Instant> {
        match self.teardown_mark.load(Relaxed) {
            0 => None,
            v => self.epoch().checked_add(Duration::from_nanos(v - 1)),
        }
    }
}

impl Default for SwapStats {
    fn default() -> Self {
        Self::new()
    }
}

fn push_bounded(q: &mut VecDeque<f64>, v: f64) {
    if q.len() >= HISTORY {
        q.pop_front();
    }
    q.push_back(v);
}

fn median(values: impl Iterator<Item = f64>) -> Option<f64> {
    let mut v: Vec<f64> = values.collect();
    if v.is_empty() {
        return None;
    }
    v.sort_by(f64::total_cmp);
    let n = v.len();
    Some(if n % 2 == 1 {
        v[n / 2]
    } else {
        (v[n / 2 - 1] + v[n / 2]) / 2.0
    })
}

/// Split the scheduler join `[start, end]` at the teardown mark into
/// `(drain, teardown)`. A missing or out-of-window mark leaves the whole join
/// as drain.
pub fn split_join(
    start: Instant,
    end: Instant,
    mark: Option<Instant>,
) -> (Duration, Option<Duration>) {
    match mark {
        Some(m) if (start..=end).contains(&m) => (
            m.saturating_duration_since(start),
            Some(end.saturating_duration_since(m)),
        ),
        _ => (end.saturating_duration_since(start), None),
    }
}

/// Times one `swap` call. Created just before the old model is released, so a
/// swap refused by a cheap check never reaches it. With `on == false` every
/// method is a no-op. A timer dropped without [`finish`](Self::finish) (an
/// early `?` return, a panic) counts as `failed`.
pub struct SwapTimer<'a> {
    stats: &'a SwapStats,
    on: bool,
    start: Instant,
    last: Instant,
    done: bool,
}

impl<'a> SwapTimer<'a> {
    pub fn begin(stats: &'a SwapStats, on: bool) -> Self {
        if on {
            // A mark left by a boot or an earlier swap must not leak in.
            stats.clear_mark();
        }
        let now = Instant::now();
        Self {
            stats,
            on,
            start: now,
            last: now,
            done: false,
        }
    }

    /// The scheduler has been joined (steps 1-4 are over).
    pub fn joined(&mut self) {
        if !self.on {
            return;
        }
        let end = Instant::now();
        let (drain, teardown) = split_join(self.last, end, self.stats.teardown_mark());
        self.stats.observe(Phase::Drain, drain);
        if let Some(t) = teardown {
            self.stats.observe(Phase::Teardown, t);
        }
        self.last = end;
    }

    /// The new model finished loading (step 5). Call on success only.
    pub fn loaded(&mut self) {
        if !self.on {
            return;
        }
        let end = Instant::now();
        self.stats
            .observe(Phase::Load, end.saturating_duration_since(self.last));
        self.last = end;
    }

    /// End the swap. `Ok` observes `total` (and the per-model history under
    /// `model`); every outcome is counted once.
    pub fn finish(&mut self, model: Option<&str>, outcome: Outcome) {
        if !self.on || self.done {
            return;
        }
        self.done = true;
        if outcome == Outcome::Ok {
            let total = self.start.elapsed();
            match model {
                Some(m) => self.stats.record_total(m, total),
                None => self.stats.observe(Phase::Total, total),
            }
        }
        self.stats.count(outcome);
    }
}

impl Drop for SwapTimer<'_> {
    fn drop(&mut self) {
        if self.on && !self.done {
            self.stats.count(Outcome::Failed);
        }
    }
}

static STATS: SwapStats = SwapStats::new();

/// The process-wide stats.
pub fn stats() -> &'static SwapStats {
    &STATS
}

/// The `model_swap_timing` row.
pub fn enabled() -> bool {
    spark_model::layers::ops::port_rows::model_swap_timing::row()
}

/// Called by the scheduler thread when it starts releasing its model (one
/// fragment line in `scheduler/mod.rs`). The hook is optional by design: if it
/// is absent the swap still times, with the whole join counted as drain, so
/// the build must not depend on it being applied.
#[allow(dead_code)]
pub fn mark_teardown_begin() {
    if enabled() {
        STATS.mark_teardown();
    }
}

#[cfg(test)]
#[path = "swap_metrics_tests.rs"]
mod tests;
