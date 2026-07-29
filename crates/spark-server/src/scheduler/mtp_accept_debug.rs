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
const PERIOD: u64 = 200;

struct Bucket {
    steps: AtomicU64,
    d1: AtomicU64,
    na: AtomicU64,
    k: AtomicU64,
}

const fn new_bucket() -> Bucket {
    Bucket {
        steps: AtomicU64::new(0),
        d1: AtomicU64::new(0),
        na: AtomicU64::new(0),
        k: AtomicU64::new(0),
    }
}

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
pub(super) fn record(n: usize, k_drafts: usize, d1_match: bool, num_accepted: usize) {
    if !spark_model::speculative::mtp_accept_debug() {
        return;
    }
    let b = &BUCKETS[n.min(MAX_N - 1)];
    b.k.fetch_max(k_drafts as u64, Ordering::Relaxed);
    b.na.fetch_add(num_accepted as u64, Ordering::Relaxed);
    if d1_match {
        b.d1.fetch_add(1, Ordering::Relaxed);
    }
    if b.steps.fetch_add(1, Ordering::Relaxed) + 1 >= PERIOD {
        let steps = b.steps.swap(0, Ordering::Relaxed).max(1);
        let d1 = b.d1.swap(0, Ordering::Relaxed);
        let na = b.na.swap(0, Ordering::Relaxed);
        let k = b.k.swap(0, Ordering::Relaxed);
        let mean_na = na as f64 / steps as f64;
        tracing::info!(
            "MTP accept n={n} k_drafts={k} verifies={steps} p1={:.3} mean_na={mean_na:.3} \
             tok_step={:.3}",
            d1 as f64 / steps as f64,
            1.0 + mean_na,
        );
    }
}
