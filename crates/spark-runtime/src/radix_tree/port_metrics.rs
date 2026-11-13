// SPDX-License-Identifier: AGPL-3.0-only

//! Responsibility: the radix prefix cache's port-sprint `/metrics` series
//! (`atlas_prefix_evict_*`, `atlas_prefix_cow_*`) and their Prometheus text.
//!
//! spark-runtime sits below spark-server, so it formats its own text with the
//! same `# HELP` / `# TYPE` lines the server's `port_metrics` helpers write;
//! `api/port_metrics.rs` calls [`render_prometheus`] once. Every series is a
//! plain atomic, and every one stays zero unless its row is on (or, for the
//! eviction timing, `AVAROK_PREFIX_EVICT_TIMING=1`, an observability-only
//! switch that lets the row-off scan arm of an A/B report its cost too), so a
//! target that declares the rows off renders zeros.

use std::fmt::Write;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

static EVICT_NANOS: AtomicU64 = AtomicU64::new(0);
static EVICT_VICTIMS: AtomicU64 = AtomicU64::new(0);
static EVICT_CALLS: AtomicU64 = AtomicU64::new(0);
static COW_COPIES: AtomicU64 = AtomicU64::new(0);
static COW_TOKENS: AtomicU64 = AtomicU64::new(0);
static COW_FALLBACKS: AtomicU64 = AtomicU64::new(0);

/// `AVAROK_PREFIX_EVICT_TIMING=1` times eviction on a tree whose LRU index
/// row is off (the A arm of the index A/B). Read once; any other value or
/// unset leaves timing to the row.
pub(super) fn evict_timing_env() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var("AVAROK_PREFIX_EVICT_TIMING").as_deref() == Ok("1"))
}

/// One `RadixTree::evict` call: its wall time under the tree lock and the
/// number of radix nodes it removed.
pub(super) fn record_evict(elapsed: Duration, victims: u64) {
    EVICT_NANOS.fetch_add(
        u64::try_from(elapsed.as_nanos()).unwrap_or(u64::MAX),
        Ordering::Relaxed,
    );
    EVICT_VICTIMS.fetch_add(victims, Ordering::Relaxed);
    EVICT_CALLS.fetch_add(1, Ordering::Relaxed);
}

/// One copy-on-write tail applied: a private block received `rows` matched
/// rows (prefill tokens not recomputed).
pub fn record_cow_copy(rows: usize) {
    COW_COPIES.fetch_add(1, Ordering::Relaxed);
    COW_TOKENS.fetch_add(rows as u64, Ordering::Relaxed);
}

/// One copy-on-write tail dropped (no block to copy into, or the source was
/// not live): the request fell back to the block-aligned match.
pub fn record_cow_fallback() {
    COW_FALLBACKS.fetch_add(1, Ordering::Relaxed);
}

/// The counters as `(evict_nanos, evict_victims, evict_calls, cow_copies,
/// cow_tokens, cow_fallbacks)`, for tests and the bench.
pub fn snapshot() -> (u64, u64, u64, u64, u64, u64) {
    (
        EVICT_NANOS.load(Ordering::Relaxed),
        EVICT_VICTIMS.load(Ordering::Relaxed),
        EVICT_CALLS.load(Ordering::Relaxed),
        COW_COPIES.load(Ordering::Relaxed),
        COW_TOKENS.load(Ordering::Relaxed),
        COW_FALLBACKS.load(Ordering::Relaxed),
    )
}

fn counter(out: &mut String, name: &str, help: &str, v: impl std::fmt::Display) {
    let _ = writeln!(
        out,
        "# HELP {name} {help}\n# TYPE {name} counter\n{name} {v}"
    );
}

/// Append the radix series to the `/metrics` text.
pub fn render_prometheus(out: &mut String) {
    let (nanos, victims, calls, copies, tokens, fallbacks) = snapshot();
    counter(
        out,
        "atlas_prefix_evict_seconds_total",
        "Wall seconds spent choosing and removing radix prefix-cache eviction victims \
         (timed with radix_lru_index on, or AVAROK_PREFIX_EVICT_TIMING=1).",
        nanos as f64 / 1e9,
    );
    counter(
        out,
        "atlas_prefix_evict_victims_total",
        "Radix prefix-cache nodes evicted while eviction timing was on.",
        victims,
    );
    counter(
        out,
        "atlas_prefix_evict_calls_total",
        "Radix prefix-cache evict calls while eviction timing was on.",
        calls,
    );
    counter(
        out,
        "atlas_prefix_cow_copies_total",
        "Copy-on-write prefix tails applied (prefix_cow_tail): a private block received the \
         matched rows of a shared one.",
        copies,
    );
    counter(
        out,
        "atlas_prefix_cow_tokens_total",
        "Prompt tokens reused through copy-on-write prefix tails.",
        tokens,
    );
    counter(
        out,
        "atlas_prefix_cow_fallbacks_total",
        "Copy-on-write prefix tails dropped for the block-aligned match (no free block or a \
         source that was not live).",
        fallbacks,
    );
}
