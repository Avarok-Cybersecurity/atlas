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
/// The `--prefill-codispatch` flag as the rendered serve gave it (G22). It
/// became a flag on 2026-09-22; before that `perf_env` disclosed it from the
/// environment, which a flag never touches, so a serve running
/// `--prefill-codispatch true` was recorded as `AVAROK_PREFILL_CODISPATCH=0`.
pub const PREFILL_CODISPATCH: &str = "prefill_codispatch";
/// `--w4a4-downcast` (activations quantised to FP4 on the small-M projection
/// and dense-FFN paths). Disclosed ONLY when true: the flag defaults to false
/// and has no environment fallback, so absent is exactly false and every
/// record before the flag existed reads correctly.
pub const W4A4_DOWNCAST: &str = "w4a4_downcast";

/// The disclosure for a server whose rendered flags resolved to these.
///
/// `mtp_gate_force` is the server's own resolution of `--mtp-gate` (and
/// `--hermetic`): `Some(true)` is `force`, `Some(false)` is `auto`. `None`
/// means the flag was not given, so the SERVER's environment decides — and
/// that environment is not this process's for a leased server. It is
/// recorded as ABSENT rather than as the default the scheduler would apply:
/// "the recipe pinned nothing" is the finding a reader needs, and spelling it
/// `auto` would hide it.
///
/// `prefill_codispatch` follows the same rule: `Some` is the flag as rendered,
/// `None` means the flag was absent and the legacy `AVAROK_PREFILL_CODISPATCH`
/// variable decides — which `serve_env` discloses when the recipe declared
/// it, and which is off when unset.
pub fn disclosure(
    mtp_gate_force: Option<bool>,
    speculative: bool,
    prefill_codispatch: Option<bool>,
    w4a4_downcast: bool,
) -> BTreeMap<String, String> {
    let mut m = BTreeMap::new();
    m.insert(SPECULATIVE.to_string(), speculative.to_string());
    if let Some(force) = mtp_gate_force {
        m.insert(
            MTP_GATE.to_string(),
            if force { "force" } else { "auto" }.to_string(),
        );
    }
    if let Some(on) = prefill_codispatch {
        m.insert(PREFILL_CODISPATCH.to_string(), on.to_string());
    }
    if w4a4_downcast {
        m.insert(W4A4_DOWNCAST.to_string(), "true".to_string());
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
