// SPDX-License-Identifier: AGPL-3.0-only

//! [`HardwareStateReport`] — the two captures, the delta and both verdicts, as
//! one value that travels with the run.
//!
//! This is what lands in `.benchmarks/<id>/<date>-<sha>.json` beside the serve
//! and param provenance. The repo's discipline is that a number without its
//! fingerprint may not be quoted later; the 2026-08-15 retraction is the case
//! that extended "fingerprint" from *which box* to *what state that box was
//! in*. Two identical `Hardware` fingerprints (NVIDIA GB10, driver
//! 580.126.09) produced 692 s and 1079 s on the same benchmark, and nothing in
//! the record said why.

use serde::{Deserialize, Serialize};

use super::policy::{self, Postcheck, Precheck, Sensitivity};
use super::state::{HardwareState, HardwareStateDelta};

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct HardwareStateReport {
    /// Which class of number this was, so a reader knows why the verdict is
    /// what it is without looking the benchmark up.
    pub sensitivity: Sensitivity,
    /// Captured before `load()`, before a single request was issued.
    pub before: HardwareState,
    /// Captured as the terminal frame is emitted. `None` when the run died
    /// before that point.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub after: Option<HardwareState>,
    /// The primary signal. `None` for the same reason as `after`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub delta: Option<HardwareStateDelta>,
    pub precheck: Precheck,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub postcheck: Option<Postcheck>,
    /// `"gb10@spark-256a"` at capture time.
    ///
    /// Copied out of `before.machine` so a reader — and a future per-class
    /// threshold lookup — does not have to know how it is derived. See the
    /// `wall_budget_s` follow-up in [`super::state::MachineIdentity`].
    pub perf_class: String,
}

impl HardwareStateReport {
    /// Open a report with the pre-run capture and its verdict.
    pub fn opened(sensitivity: Sensitivity, before: HardwareState) -> Self {
        let precheck = policy::precheck(sensitivity, &before, policy::PolicyOptions::from_env());
        Self {
            sensitivity,
            perf_class: before.machine.perf_class(),
            before,
            after: None,
            delta: None,
            precheck,
            postcheck: None,
        }
    }

    /// Close it with the post-run capture, the delta and the validity verdict.
    pub fn close(&mut self, after: HardwareState) {
        let delta = HardwareStateDelta::between(&self.before, &after);
        self.postcheck = Some(policy::postcheck(
            self.sensitivity,
            &delta,
            policy::PolicyOptions::from_env(),
        ));
        self.delta = Some(delta);
        self.after = Some(after);
    }

    /// True when the pre-state forbids starting.
    pub fn refuses(&self) -> bool {
        self.precheck.decision == policy::Decision::Refuse
    }

    /// True when the run completed but its numbers may not be quoted.
    ///
    /// A run that was never closed is NOT invalid — it is unmeasured, and
    /// saying otherwise would blame the box for a harness failure.
    pub fn invalidated(&self) -> bool {
        self.postcheck
            .as_ref()
            .is_some_and(|p| p.validity == policy::Validity::Invalid)
    }

    /// Every concern from both phases, in order, for the run log.
    pub fn concerns(&self) -> Vec<&str> {
        self.precheck
            .concerns
            .iter()
            .chain(self.postcheck.iter().flat_map(|p| p.concerns.iter()))
            .map(String::as_str)
            .collect()
    }
}

#[cfg(test)]
#[path = "report_tests.rs"]
mod tests;
