// SPDX-License-Identifier: AGPL-3.0-only

//! What a gate record discloses about the ENVIRONMENT its server ran under:
//! the three scheduler controls with their defaults filled in (`perf_env`),
//! and the whole `AVAROK_*` lever set the gate applied (`serve_env`).
//!
//! Split out of `record.rs` at the 500-line cap. The `PERF_CONTROLS` table
//! and `resolve_perf_env` are an exact piecewise move; `with_serve_env` is
//! new with #1242.

use std::collections::BTreeMap;

use super::record::GateRecord;

/// The scheduler performance controls a gate record discloses, each with the
/// default the scheduler applies when it is unset.
///
/// Kept beside the record rather than imported from the scheduler because
/// `avarok-plugin` does not depend on `spark-server`. That is a real duplication
/// and `perf_env_defaults_match_the_scheduler` pins it: if a default moves in
/// `scheduler::mod_helpers`, that test is what fails.
const PERF_CONTROLS: [(&str, &str); 3] = [
    ("AVAROK_PREFILL_CODISPATCH", "0"),
    ("AVAROK_PREFILL_CODISPATCH_WINDOW_MS", "100"),
    ("AVAROK_PREFILL_CODISPATCH_SETTLE_MS", "10"),
];

/// Resolve the `PERF_CONTROLS` table through `lookup`, substituting each
/// default for an unset or empty variable.
///
/// Pure over the lookup so it is testable without mutating the process
/// environment — `set_var` is unsafe and process-global, and a test that raced
/// another test's read would be exactly the kind of intermittent this file
/// exists to make impossible.
pub fn resolve_perf_env(lookup: impl Fn(&str) -> Option<String>) -> BTreeMap<String, String> {
    PERF_CONTROLS
        .iter()
        .map(|(key, default)| {
            let value = lookup(key).filter(|v| !v.trim().is_empty());
            (
                (*key).to_string(),
                value.unwrap_or_else(|| (*default).to_string()),
            )
        })
        .collect()
}

impl GateRecord {
    /// Attach the `AVAROK_*` serve levers the gate APPLIED to its server —
    /// `serve_env::Reconciled::env` — and re-resolve `perf_env` through them.
    ///
    /// `from_run` reads the co-dispatch controls off THIS process's
    /// environment, which was exact while every gate served in-process. A
    /// leased server is a child that was given the declared set on top of
    /// the inherited environment, so a declared `AVAROK_PREFILL_CODISPATCH=1`
    /// the harness does not itself carry would otherwise be disclosed as `0`
    /// — the record contradicting the server. The applied set answers first;
    /// the process environment still answers for anything it does not name,
    /// which after `serve_env::reconcile` can only be the defaults.
    #[must_use]
    pub fn with_serve_env(mut self, env: BTreeMap<String, String>) -> Self {
        self.perf_env = resolve_perf_env(|k| env.get(k).cloned().or_else(|| std::env::var(k).ok()));
        self.serve_env = env;
        self
    }
}

#[cfg(test)]
#[path = "record_env_tests.rs"]
mod tests;
