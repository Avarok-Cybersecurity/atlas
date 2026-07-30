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

/// Clear every run mailbox. Call once, when a new model's backend is built.
pub fn reset_for_new_run() {
    crate::prefix_cache::reset();
    crate::sampler::reset_entropy();
    crate::kernel_audit::reset();
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Written as a threshold rather than an equality on purpose.
    ///
    /// These counters are process-global, and cargo runs this binary's tests
    /// in parallel threads — the `radix_tree` and `sampler` cases record into
    /// the same counters while this one runs. An `assert_eq!(.., 0)` after the
    /// reset is therefore flaky by construction, which is a fair demonstration
    /// of what a process-global counter costs even when it is the right shape.
    /// A run's worth of hits is orders of magnitude above the handful a
    /// concurrent test contributes, so the drop is unambiguous.
    #[test]
    fn a_new_run_starts_from_the_bottom() {
        const RUN: u64 = 10_000;
        for _ in 0..RUN {
            crate::prefix_cache::record_cache_hit(1);
        }
        assert!(
            crate::prefix_cache::cache_hit_count() >= RUN,
            "the run accumulated"
        );

        reset_for_new_run();

        assert!(
            crate::prefix_cache::cache_hit_count() < RUN / 10,
            "the next run does not inherit the previous run's hit count"
        );
        assert!(crate::prefix_cache::cache_hit_tokens_total() < RUN / 10);
    }
}
