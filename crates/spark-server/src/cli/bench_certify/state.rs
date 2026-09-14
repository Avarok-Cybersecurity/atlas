// SPDX-License-Identifier: AGPL-3.0-only

//! The campaign as a pure state machine: which unit runs next, what a
//! finished unit means for the rest, and when to stop.
//!
//! No I/O here. The driver (local or remote) asks [`Campaign::next_to_start`],
//! runs the unit, reports the outcome, and honours [`Campaign::stopped`].
//! Every policy — stop on the first FAIL unless told to keep going, one retry
//! for a retryable harness failure, none for a timeout, abort on drift — is a
//! branch in this file with a test on it.

use super::guard::Drift;
use super::plan::Unit;
use super::runner::RunOutcome;

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Phase {
    Pending,
    Running,
    Passed,
    MemberDone,
    Failed(String),
    Skipped(String),
}

pub struct Campaign {
    pub units: Vec<Unit>,
    pub phase: Vec<Phase>,
    pub keep_going: bool,
    retried: Vec<bool>,
    /// Set when the whole campaign is over before its units are: drift,
    /// cancel, or a FAIL without `keep_going`.
    aborted: Option<String>,
    fail_seen: bool,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Summary {
    pub passed: Vec<&'static str>,
    pub member_done: Vec<&'static str>,
    pub failed: Vec<(&'static str, String)>,
    pub skipped: Vec<(&'static str, String)>,
    pub aborted: Option<String>,
}

impl Campaign {
    pub fn new(units: Vec<Unit>, keep_going: bool) -> Self {
        let n = units.len();
        Self {
            units,
            phase: vec![Phase::Pending; n],
            keep_going,
            retried: vec![false; n],
            aborted: None,
            fail_seen: false,
        }
    }

    /// The next pending unit, or `None` when there is none or the campaign
    /// is stopped. Marks it running.
    pub fn next_to_start(&mut self) -> Option<usize> {
        if self.stopped() {
            return None;
        }
        let i = self.phase.iter().position(|p| *p == Phase::Pending)?;
        self.phase[i] = Phase::Running;
        Some(i)
    }

    /// Mark a chosen pending unit running (a scheduler picked it rather
    /// than the first pending one).
    pub fn start(&mut self, i: usize) {
        debug_assert_eq!(self.phase[i], Phase::Pending);
        self.phase[i] = Phase::Running;
    }

    /// Is any unit still to run or running?
    pub fn stopped(&self) -> bool {
        self.aborted.is_some()
            || self
                .phase
                .iter()
                .all(|p| !matches!(p, Phase::Pending | Phase::Running))
    }

    /// Record a unit's outcome. Returns `true` when the unit should be run
    /// again (one retry for a retryable harness failure).
    pub fn finished(&mut self, i: usize, outcome: RunOutcome) -> bool {
        match outcome {
            RunOutcome::Passed { .. } => self.phase[i] = Phase::Passed,
            RunOutcome::MemberDone { .. } => self.phase[i] = Phase::MemberDone,
            RunOutcome::VerdictFail { reason, .. } => {
                self.phase[i] = Phase::Failed(reason);
                self.fail_seen = true;
                if !self.keep_going {
                    self.stop_rest(format!(
                        "stopped after {} failed its verdict (pass --keep-going to run on)",
                        self.units[i].id
                    ));
                }
            }
            RunOutcome::Harness { reason, retryable } => {
                if retryable && !self.retried[i] {
                    self.retried[i] = true;
                    self.phase[i] = Phase::Pending;
                    return true;
                }
                self.phase[i] = Phase::Failed(format!("harness: {reason}"));
                self.fail_seen = true;
                if !self.keep_going {
                    self.stop_rest(format!(
                        "stopped after {} could not be run",
                        self.units[i].id
                    ));
                }
            }
            RunOutcome::TimedOut => {
                self.phase[i] = Phase::Failed("timed out".into());
                self.fail_seen = true;
                if !self.keep_going {
                    self.stop_rest(format!("stopped after {} timed out", self.units[i].id));
                }
            }
            RunOutcome::Cancelled => {
                self.phase[i] = Phase::Failed("cancelled".into());
                self.abort("cancelled".into());
            }
        }
        false
    }

    /// What the guard found. Drift on a perf path aborts; a guard that could
    /// not answer aborts too — "could not check" is never "safe".
    pub fn guard(&mut self, result: Result<Drift, String>) -> Option<&str> {
        match result {
            Ok(Drift::Unmoved) | Ok(Drift::MovedHarmlessly { .. }) => None,
            Ok(Drift::PerfPathMoved { head, paths }) => {
                self.abort(format!(
                    "a perf path moved on the guarded branch (now {}): {} — every record \
                     taken after this names a dead tree",
                    &head[..head.len().min(10)],
                    paths.join(", ")
                ));
                self.aborted.as_deref()
            }
            Err(e) => {
                self.abort(format!(
                    "the drift guard could not answer ({e}); not continuing blind"
                ));
                self.aborted.as_deref()
            }
        }
    }

    pub fn cancel(&mut self) {
        self.abort("cancelled".into());
    }

    fn abort(&mut self, why: String) {
        if self.aborted.is_none() {
            self.stop_rest(why.clone());
            self.aborted = Some(why);
        }
    }

    fn stop_rest(&mut self, why: String) {
        for p in &mut self.phase {
            if *p == Phase::Pending {
                *p = Phase::Skipped(why.clone());
            }
        }
    }

    pub fn summary(&self) -> Summary {
        let mut s = Summary {
            aborted: self.aborted.clone(),
            ..Default::default()
        };
        for (u, p) in self.units.iter().zip(&self.phase) {
            match p {
                Phase::Passed => s.passed.push(u.id),
                Phase::MemberDone => s.member_done.push(u.id),
                Phase::Failed(r) => s.failed.push((u.id, r.clone())),
                Phase::Skipped(r) => s.skipped.push((u.id, r.clone())),
                Phase::Pending | Phase::Running => {}
            }
        }
        s
    }

    /// `0` certified, `2` a verdict failed, `3` aborted. `certified` is the
    /// final gate check's word, which this cannot know on its own.
    pub fn exit_code(&self, certified: bool) -> i32 {
        if self.aborted.is_some() {
            3
        } else if self.fail_seen || !certified {
            2
        } else {
            0
        }
    }
}

#[cfg(test)]
#[path = "state_tests.rs"]
mod state_tests;
