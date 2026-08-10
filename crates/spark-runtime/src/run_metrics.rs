// SPDX-License-Identifier: AGPL-3.0-only

//! Run mailboxes — the observability surfaces that stay process-global on
//! purpose, and the one call that keeps them honest across a model swap.
//!
//! Most model-derived state in Atlas is carried: `SchedCtx`, `ForwardContext`,
//! `ModelLevers`, `OpCache`. A handful of counters cannot be, because their
//! *readers* cannot be handed a carrier — `/metrics` answers from an HTTP
//! handler thread and the dashboard polls from the TUI thread, both while the
//! scheduler is mid-step and holding its own context. A process-global address
//! is what an observability surface is for.
//!
//! That leaves the scoping problem: after a swap the counters would describe
//! two models at once. [`reset_for_new_run`] solves it from the other end —
//! the values stay reachable at a fixed address, but they start clean when a
//! run does, so a reader asking "what is the prefix-cache hit rate" gets the
//! rate for the model now running. Prometheus reads the reset as a counter
//! restart, which it already handles.
//!
//! Called from `AtlasCudaBackend::new`, which is where a model's GPU state
//! begins. That is deliberately upstream of the first kernel lookup, so the
//! kernel audit records only this model's modules.

use std::sync::atomic::{AtomicU32, AtomicU64, AtomicUsize, Ordering};
use std::sync::{LazyLock, Mutex};

/// The process's single run mailbox.
///
/// Seven separate statics across three modules became one, because they were
/// always one thing: the numbers a reader gets when it asks what the running
/// model is doing. Splitting them meant `reset_for_new_run` had to reach into
/// three modules and could silently miss one — the failure being a counter
/// that keeps a dead model's value while its neighbours restart.
#[derive(Debug, Default)]
pub struct RunMetrics {
    // ── Prefix cache (one RadixTree per server) ──
    pub cache_hits: AtomicU64,
    pub cache_misses: AtomicU64,
    pub cache_hit_tokens: AtomicU64,

    // ── Sampler entropy ──
    /// Most recent per-token entropy, f32 bits for a lock-free read.
    pub last_entropy: AtomicU32,
    pub low_entropy_tokens: AtomicU64,
    pub total_sampled_tokens: AtomicU64,

    /// Free device memory at GPU-context init, before this run allocated
    /// anything. Lets KV sizing measure this process's own footprint as
    /// `baseline - free_now`, excluding co-tenants automatically. `0` =
    /// unset (the mock backend in tests) and callers fall back.
    pub baseline_free_bytes: AtomicUsize,
    /// `(module, func, loaded, dispatch site)` for every kernel lookup this run
    /// made. The site is the `Location` of the `.kernel(…)` / `try_kernel(…)`
    /// call, carried in through `#[track_caller]`: a bare `module::func` list
    /// is not actionable when the same module is looked up from a dozen
    /// constructors and the fix is always "go to that line".
    pub kernel_audit: Mutex<Vec<(String, String, bool, &'static std::panic::Location<'static>)>>,

    // ── Per-run baselines for the counters above ──
    //
    // `cache_hits`, `cache_misses` and `cache_hit_tokens` are exported on
    // /metrics as `atlas_prefix_cache_*_total`, declared `TYPE counter`, and a
    // counter must only ever climb. Zeroing them was harmless while a backend
    // was built exactly once per process — the reset happened before anything
    // could scrape. Hot-swap builds one per load, so the same call now resets
    // live counters mid-life, which reads to Prometheus as a restart and
    // corrupts `rate()` and `increase()` across the swap.
    //
    // The counters therefore stay cumulative, and "this run" is DERIVED by
    // subtracting a snapshot taken when the run began. One authoritative
    // number, two views of it.
    run_base_cache_hits: AtomicU64,
    run_base_cache_misses: AtomicU64,
    run_base_cache_hit_tokens: AtomicU64,
}

/// The mailbox. See the module doc for why this one is static.
static METRICS: LazyLock<RunMetrics> = LazyLock::new(RunMetrics::default);

/// Read the mailbox.
pub fn metrics() -> &'static RunMetrics {
    &METRICS
}

/// Begin a new run's accounting. Called when a new model's backend is built.
///
/// The monotonic counters are SNAPSHOTTED, not cleared — see the baseline
/// fields. Everything else here is per-run scratch that nothing exports, so it
/// is cleared outright.
pub fn reset_for_new_run() {
    let m = metrics();
    for (counter, base) in [
        (&m.cache_hits, &m.run_base_cache_hits),
        (&m.cache_misses, &m.run_base_cache_misses),
        (&m.cache_hit_tokens, &m.run_base_cache_hit_tokens),
    ] {
        base.store(counter.load(Ordering::Relaxed), Ordering::Relaxed);
    }
    for c in [&m.low_entropy_tokens, &m.total_sampled_tokens] {
        c.store(0, Ordering::Relaxed);
    }
    m.baseline_free_bytes.store(0, Ordering::Relaxed);
    m.last_entropy.store(0, Ordering::Relaxed);
    if let Ok(mut v) = m.kernel_audit.lock() {
        v.clear();
    }
    // The next model runs its own eager lookups and gets its own boot gate, so
    // the seal from the outgoing model must not outlive it — otherwise every
    // one of the incoming model's own lookups reads as a "late" one.
    crate::kernel_audit::unseal();
}

/// Prefix-cache activity since the CURRENT model was loaded.
///
/// The dashboard asks "how is this model doing", not "how has this process
/// done since boot"; after a swap those differ. Prometheus wants the opposite
/// and reads the cumulative counters directly.
pub fn cache_counts_this_run() -> (u64, u64, u64) {
    let m = metrics();
    let sub = |c: &AtomicU64, b: &AtomicU64| {
        c.load(Ordering::Relaxed)
            .saturating_sub(b.load(Ordering::Relaxed))
    };
    (
        sub(&m.cache_hits, &m.run_base_cache_hits),
        sub(&m.cache_misses, &m.run_base_cache_misses),
        sub(&m.cache_hit_tokens, &m.run_base_cache_hit_tokens),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Written as a threshold rather than an equality on purpose.
    ///
    /// The mailbox is process-global, and cargo runs this binary's tests in
    /// parallel threads — the `radix_tree` and `sampler` cases record into it
    /// while this one runs. An `assert_eq!(.., 0)` after the reset is
    /// therefore flaky by construction, which is a fair demonstration of what
    /// a process-global counter costs even when a global is the right shape.
    /// A run's worth of hits is orders of magnitude above the handful a
    /// concurrent test contributes, so the drop is unambiguous.
    /// The assertion here is INVERTED from what it was, deliberately.
    ///
    /// It used to require that `cache_hit_count()` itself dropped to near zero.
    /// That was correct while a backend was built exactly once per process: the
    /// reset ran before anything could observe the counter. Hot-swap builds one
    /// per load, and the same value is exported on /metrics as
    /// `atlas_prefix_cache_hits_total`, declared `TYPE counter` — so the old
    /// behaviour resets a live counter, which Prometheus reads as a restart.
    ///
    /// A new run still starts from the bottom; the counter is no longer the
    /// thing that moves.
    #[test]
    fn a_new_run_starts_from_the_bottom() {
        const RUN: u64 = 10_000;
        for _ in 0..RUN {
            crate::prefix_cache::record_cache_hit(1);
        }
        let cumulative = crate::prefix_cache::cache_hit_count();
        assert!(cumulative >= RUN, "the run accumulated");

        reset_for_new_run();

        assert!(
            crate::prefix_cache::cache_hit_count() >= cumulative,
            "the exported counter must never go backwards"
        );
        let (hits, _, tokens) = cache_counts_this_run();
        assert!(
            hits < RUN / 10,
            "but the next run does not inherit the previous run's hits"
        );
        assert!(tokens < RUN / 10);
    }
}

#[cfg(test)]
mod swap_counter_tests {
    use super::*;

    /// Deliberately expressed as `>=` against a value read at the start, not as
    /// an equality. Other tests in this binary record into the same global
    /// mailbox concurrently — they can only ADD, so a lower bound is immune to
    /// them, and an exact assertion here was flaky on the first run of four.
    #[test]
    fn a_new_run_does_not_move_the_counters_prometheus_exports() {
        // atlas_prefix_cache_hits_total is declared TYPE counter, and a counter
        // must only ever climb. Zeroing it was harmless when a backend was
        // built once per process; hot-swap builds one per load, so the same
        // call would reset a live counter and read to Prometheus as a restart.
        let m = metrics();
        let before = m.cache_hits.load(Ordering::Relaxed);
        m.cache_hits.fetch_add(7, Ordering::Relaxed);

        reset_for_new_run();

        assert!(
            m.cache_hits.load(Ordering::Relaxed) >= before + 7,
            "the cumulative counter must not go backwards across a swap"
        );
    }
}
