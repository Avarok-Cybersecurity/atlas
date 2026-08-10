// SPDX-License-Identifier: AGPL-3.0-only

//! The endpoint check that runs between pressing START and the benchmark
//! actually starting.
//!
//! A benchmark aimed at the wrong server is the cheapest mistake to make and
//! one of the most expensive to discover: BFCL scores a failed sample as "no
//! call", so a 12-hour run finishes and reports a near-zero accuracy that looks
//! exactly like a model regression.
//!
//! So the check happens first, and the user sees it happen. It takes two short
//! completions, which is long enough to need a spinner and short enough that a
//! progress bar would be a lie.
//!
//! **It cannot veto a run.** Benchmarking a base checkpoint, or pointing a
//! suite at a model it was not written for, are real things to do on purpose.
//! The modal reports what it saw and the user decides.

use std::sync::mpsc::{Receiver, TryRecvError, channel};

use atlas_plugin::coherence::{self, Report};

/// Where the pre-flight has got to.
#[derive(Debug, PartialEq, Eq)]
pub enum Phase {
    /// Asking. The modal shows an indeterminate spinner.
    Checking,
    /// Something is worth saying before committing; the user picks.
    Concern(String),
}

pub struct Preflight {
    pub phase: Phase,
    rx: Option<Receiver<Report>>,
}

impl Preflight {
    /// Begin checking `target` on `runtime`.
    ///
    /// The probe is async and the dashboard is not, so it runs as a task and
    /// answers over a channel the UI drains on its normal tick — no blocking
    /// call on the render thread.
    pub fn begin(
        runtime: &tokio::runtime::Handle,
        target: atlas_plugin::TargetEndpoint,
        expectation: Option<atlas_plugin::benchmark::ModelExpectation>,
        timeout: std::time::Duration,
    ) -> Self {
        let (tx, rx) = channel();
        runtime.spawn(async move {
            // A disconnected receiver means the user moved on; not an error.
            let _ = tx.send(coherence::probe_for(&target, expectation, timeout).await);
        });
        Self {
            phase: Phase::Checking,
            rx: Some(rx),
        }
    }

    /// Drain the check. Returns `Some(true)` when the run should start now,
    /// `Some(false)` when the user must be asked first, `None` while waiting.
    pub fn poll(&mut self, target: &atlas_plugin::TargetEndpoint) -> Option<bool> {
        let rx = self.rx.as_ref()?;
        match rx.try_recv() {
            Ok(report) => {
                self.rx = None;
                match report.concern(target) {
                    None => Some(true),
                    Some(concern) => {
                        self.phase = Phase::Concern(concern);
                        Some(false)
                    }
                }
            }
            Err(TryRecvError::Empty) => None,
            // The task vanished. Do not strand the user in a spinner: treat an
            // unanswerable check as nothing to report and let the run proceed.
            Err(TryRecvError::Disconnected) => {
                self.rx = None;
                Some(true)
            }
        }
    }

    /// A pre-flight still waiting, for tests and for rendering the spinner
    /// without a runtime.
    #[cfg(test)]
    pub fn pending() -> Self {
        let (_tx, rx) = channel();
        std::mem::forget(_tx);
        Self {
            phase: Phase::Checking,
            rx: Some(rx),
        }
    }

    /// A pre-flight that has already found something to say.
    #[cfg(test)]
    pub fn with_concern(text: String) -> Self {
        Self {
            phase: Phase::Concern(text),
            rx: None,
        }
    }

    pub fn is_checking(&self) -> bool {
        self.phase == Phase::Checking
    }
}

#[cfg(test)]
#[path = "bench_preflight_tests.rs"]
mod tests;
