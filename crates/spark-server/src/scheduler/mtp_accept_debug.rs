// SPDX-License-Identifier: AGPL-3.0-only

//! Per-batch-width MTP acceptance telemetry (`ATLAS_MTP_ACCEPT_DEBUG`).
//!
//! # Why this exists
//!
//! The C=8 bar is arithmetic in ONE quantity: expected tokens per verify step
//! divided by the verify step's cost relative to a plain decode step. The
//! numerator is `1 + p1 + p1*p2c + ...`, i.e. `1 + mean_accepted`. Before this
//! module nothing reported it at the shipped operating point:
//! `k4_record_positional` (the only p1 source) is gated on `k_drafts == 3`,
//! and the default ladder runs `k_drafts == 2` for n in [5, 8]. The na
//! histogram (`k4_record_outcome`) is width-blind — it mixes every n in one
//! set of counters, so an accept A/B at C=8 could not be attributed.
//!
//! # What it reports
//!
//! One line per `PERIOD` recorded verifies PER BATCH WIDTH:
//! `p1` (fraction of steps whose FIRST draft matched the target — measured
//! before the accept chain short-circuits, so it is unconditional),
//! `mean_na` (mean accepted drafts) and `tok_step = 1 + mean_na`.
//!
//! Counters are relaxed atomics and the log fires off one thread at a time;
//! there is no D2H and no stream sync, so the only cost in a timed leg is the
//! periodic `tracing::info!`. Still gated: presence of `ATLAS_MTP_ACCEPT_DEBUG`.

use std::sync::atomic::{AtomicU64, Ordering};

/// Widths tracked individually; anything wider folds into the last bucket.
const MAX_N: usize = 17;
/// Verifies per flush. A flush is both a log line and one controller tick
/// for [`super::adaptive_rung`], so it sets the rung's reaction time: 128
/// verifies = 8 steps at n=16 ~ 1 s, which puts the probe interval (8 ticks)
/// inside a single short agentic burst. Was 200 before wave 28, when the
/// flush had no consumer but the log.
const PERIOD: u64 = 128;

/// A flush that took longer than this to accumulate its `PERIOD` verifies is
/// NOT a sample of current traffic and is dropped rather than fed to the rung
/// controller. Measured 2026-08-01: the width the batch actually sits at
/// fills a bucket in ~1.1 s, while a width the batch only grazes (n=15 at
/// C=16) took over TWO MINUTES — long enough to span a whole change of
/// workload. That stale bucket flushed prose-era statistics in the middle of
/// a tool-shaped leg and flipped the rung back to `k=1`, which is exactly the
/// failure this guard removes. Only the flush timestamp is read, so the cost
/// is one clock read per `PERIOD` verifies, not per verify.
const MAX_SAMPLE_SPAN_MS: u64 = 5_000;

struct Bucket {
    steps: AtomicU64,
    d1: AtomicU64,
    na: AtomicU64,
    k: AtomicU64,
    /// Millis since [`EPOCH`] when this bucket's CURRENT accumulation window
    /// opened (stamped when `steps` goes 0 -> 1), so the span below is the
    /// exact window the sample covers — correct for the first flush too.
    window_start_ms: AtomicU64,
}

const fn new_bucket() -> Bucket {
    Bucket {
        steps: AtomicU64::new(0),
        d1: AtomicU64::new(0),
        na: AtomicU64::new(0),
        k: AtomicU64::new(0),
        window_start_ms: AtomicU64::new(0),
    }
}

/// Process-start reference for the flush clock (monotonic, unaffected by
/// wall-clock steps).
static EPOCH: std::sync::LazyLock<std::time::Instant> =
    std::sync::LazyLock::new(std::time::Instant::now);

#[allow(clippy::declare_interior_mutable_const)]
const INIT: Bucket = new_bucket();
static BUCKETS: [Bucket; MAX_N] = [INIT; MAX_N];

/// Record one sequence's verify outcome at batch width `n`.
///
/// `d1_match` must be the UNCONDITIONAL first-position draft match
/// (`drafts[0] == verified[0]`), not `num_accepted >= 1` — they agree today
/// but the second form silently becomes conditional if a future verdict path
/// short-circuits before comparing.
///
/// `k_drafts` is the RETAINED depth of THIS sequence, which D-Cut makes ragged
/// within one batch; the reported value is the period's DEEPEST retained depth
/// at this width (identical to the uniform value when D-Cut is off, and never
/// an arbitrary last-writer). The full per-step shape is on the `MTP D-Cut`
/// line — `p1` stays unconditional and `mean_na`/`tok_step` are measured over
/// the shape that actually ran, which is the quantity the C=8 arithmetic wants.
/// ★ Accumulation is UNCONDITIONAL as of wave 28 — `ATLAS_MTP_ACCEPT_DEBUG`
/// now gates only the log line. The counters are the SSOT for accept
/// statistics and [`super::adaptive_rung`] steers the n=16 rung from them, so
/// gating the accounting would make the shipped rung depend on whether
/// telemetry happened to be switched on. The counters are relaxed atomics
/// with no D2H and no stream sync, so always-on costs nothing measurable.
pub(super) fn record(n: usize, k_drafts: usize, d1_match: bool, num_accepted: usize) {
    let b = &BUCKETS[n.min(MAX_N - 1)];
    b.k.fetch_max(k_drafts as u64, Ordering::Relaxed);
    b.na.fetch_add(num_accepted as u64, Ordering::Relaxed);
    if d1_match {
        b.d1.fetch_add(1, Ordering::Relaxed);
    }
    let prev_steps = b.steps.fetch_add(1, Ordering::Relaxed);
    if prev_steps == 0 {
        b.window_start_ms
            .store(EPOCH.elapsed().as_millis() as u64, Ordering::Relaxed);
    }
    if prev_steps + 1 >= PERIOD {
        let steps = b.steps.swap(0, Ordering::Relaxed).max(1);
        let d1 = b.d1.swap(0, Ordering::Relaxed);
        let na = b.na.swap(0, Ordering::Relaxed);
        let k = b.k.swap(0, Ordering::Relaxed);
        let mean_na = na as f64 / steps as f64;
        let p1 = d1 as f64 / steps as f64;
        let span_ms = (EPOCH.elapsed().as_millis() as u64)
            .saturating_sub(b.window_start_ms.load(Ordering::Relaxed));
        let fresh = span_ms <= MAX_SAMPLE_SPAN_MS;
        // ONE accounting path: the rung controller consumes this flush
        // rather than maintaining its own counters (SSOT). Stale flushes are
        // logged but NOT steered on — see MAX_SAMPLE_SPAN_MS.
        if fresh {
            super::adaptive_rung::observe(n, k as usize, p1, mean_na);
        }
        if spark_model::speculative::mtp_accept_debug() {
            tracing::info!(
                "MTP accept n={n} k_drafts={k} verifies={steps} p1={p1:.3} \
                 mean_na={mean_na:.3} tok_step={:.3} token_ratio={:.4} \
                 span_ms={span_ms} fresh={fresh}",
                1.0 + mean_na,
                super::adaptive_rung::token_ratio(
                    p1,
                    super::adaptive_rung::p2_cond_from(p1, mean_na).unwrap_or(0.0),
                ),
            );
        }
    }
}
