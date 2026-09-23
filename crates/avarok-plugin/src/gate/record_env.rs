// SPDX-License-Identifier: AGPL-3.0-only

//! What a gate record discloses about the ENVIRONMENT its server ran under:
//! the co-dispatch timing controls with their defaults filled in
//! (`perf_env`), and the whole `AVAROK_*` lever set the gate applied
//! (`serve_env`).
//!
//! Split out of `record.rs` at the 500-line cap. The `PERF_CONTROLS` table
//! and `resolve_perf_env` are an exact piecewise move; `with_serve_env` is
//! new with #1242.

use std::collections::BTreeMap;

use super::record::GateRecord;

/// The scheduler performance controls a gate record discloses, each with the
/// default the scheduler applies when it is unset.
///
/// ★ `AVAROK_PREFILL_CODISPATCH` itself is NOT here since G22 (2026-09-23).
/// It became the `--prefill-codispatch` flag, which a flag-driven serve never
/// writes back to the environment, so resolving it here disclosed the table
/// default `0` for serves running `--prefill-codispatch true` — every
/// concurrency-sweep record from 6c75c09da4 on. The flag is disclosed from the
/// resolved serve instead (`record_serve::PREFILL_CODISPATCH` in
/// `serve_resolved`), and a recipe that still declares the legacy variable has
/// it disclosed in `serve_env`. The two timing controls stay: the scheduler
/// still reads them from the environment only.
///
/// Kept beside the record rather than imported from the scheduler because
/// `avarok-plugin` does not depend on `spark-server`. That is a real duplication
/// and `perf_env_defaults_match_the_scheduler` pins it: if a default moves in
/// `scheduler::mod_helpers`, that test is what fails.
///
/// `AVAROK_NO_W4A16_TC` is a PRESENCE kill switch (any non-empty value, `0`
/// included, turns the tensor-core small-M NVFP4 GEMV off), so its default is
/// the literal `unset`, the one value under which the tensor-core path ran.
/// `AVAROK_W4A16_TC_WIDE` is the opposite polarity: an OPT-IN (any non-empty
/// value) that widens the tensor-core row edge from 8 to 16, so `unset` means
/// it did NOT run. `record_env_tests` pins both rules against
/// `layers/ops/gemv_tc.rs`. The W4A4 activation downcast is NOT here: it is
/// the `--w4a4-downcast` flag, disclosed in `serve_resolved`
/// (`record_serve::W4A4_DOWNCAST`).
///
/// `AVAROK_NO_MTP_TC` is the same kind of PRESENCE kill switch as
/// `AVAROK_NO_W4A16_TC`, for the MTP drafter's tensor-core BF16 GEMV: default
/// `unset` (the path ran), any set value disclosed verbatim.
/// `tc_mtp_switch_default_matches_the_lever` pins that rule against
/// `layers/ops/dense_gemv_tc.rs`.
const PERF_CONTROLS: [(&str, &str); 5] = [
    ("AVAROK_PREFILL_CODISPATCH_WINDOW_MS", "100"),
    ("AVAROK_PREFILL_CODISPATCH_SETTLE_MS", "10"),
    ("AVAROK_NO_W4A16_TC", "unset"),
    ("AVAROK_W4A16_TC_WIDE", "unset"),
    ("AVAROK_NO_MTP_TC", "unset"),
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
    /// the inherited environment, so a declared
    /// `AVAROK_PREFILL_CODISPATCH_WINDOW_MS=50` the harness does not itself
    /// carry would otherwise be disclosed as `100` — the record contradicting
    /// the server. The applied set answers first;
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
