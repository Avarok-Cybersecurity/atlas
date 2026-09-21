// SPDX-License-Identifier: AGPL-3.0-only

//! The serve knobs a gate record DISCLOSES about the server it measured.
//!
//! `served_by` names a recipe in another repository at whatever version was
//! synced, and `serve_overrides` names only what the operator changed. Neither
//! says what the server actually ran with, and for one knob that gap has cost
//! real diagnosis: `mtp_gate: auto` is the standing explanation for
//! `agentic-webserver`'s intermittent 9/10, `atlas-recipes#16` pinned it to
//! `force` on 2026-08-28, and no record proves which regime any run was in —
//! a `BENCH.toml` `default = true` flip silently unpins a gate (#1159,
//! `docs/gate-queue-protocol.md`). A future failure whose record says `force`
//! is a pinned failure, and MTP nondeterminism is ruled out for free.
//!
//! The keys are named HERE, in the crate that owns the record, so the server
//! that resolves them and the tests that read them cannot drift apart.

use std::collections::BTreeMap;

use super::record::GateRecord;

/// The `--mtp-gate` regime in force, as the record spells it.
pub const MTP_GATE: &str = "mtp_gate";
/// Whether `--speculative` was on at all — `mtp_gate` means nothing without it.
pub const SPECULATIVE: &str = "speculative";

/// The disclosure for a server whose rendered flags resolved to these.
///
/// `mtp_gate_force` is the server's own resolution of `--mtp-gate` (and
/// `--hermetic`): `Some(true)` is `force`, `Some(false)` is `auto`. `None`
/// means the flag was not given, so the SERVER's environment decides — and
/// that environment is not this process's for a leased server. It is
/// recorded as ABSENT rather than as the default the scheduler would apply:
/// "the recipe pinned nothing" is the finding a reader needs, and spelling it
/// `auto` would hide it.
pub fn disclosure(mtp_gate_force: Option<bool>, speculative: bool) -> BTreeMap<String, String> {
    let mut m = BTreeMap::new();
    m.insert(SPECULATIVE.to_string(), speculative.to_string());
    if let Some(force) = mtp_gate_force {
        m.insert(
            MTP_GATE.to_string(),
            if force { "force" } else { "auto" }.to_string(),
        );
    }
    m
}

impl GateRecord {
    /// Attach what the gate's serve resolved — see [`disclosure`].
    #[must_use]
    pub fn with_serve_resolved(mut self, resolved: BTreeMap<String, String>) -> Self {
        self.serve_resolved = resolved;
        self
    }
}

#[cfg(test)]
#[path = "record_serve_tests.rs"]
mod record_serve_tests;
