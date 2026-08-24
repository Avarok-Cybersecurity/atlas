// SPDX-License-Identifier: AGPL-3.0-only

//! A known-answer probe, run once before a benchmark's measured work.
//!
//! [`crate::http::probe`] proves only that *something* answers `/v1/models`
//! with a 200. It opens a socket, reads the status line, and never parses the
//! body — so it cannot tell whether the server holds the model you named, nor
//! whether that model can generate at all.
//!
//! That gap has a concrete cost. A wrong `--model` passes the reachability
//! probe, and every subsequent request fails individually. The BFCL benchmarks
//! score a failed sample as "no call" on purpose (a transport failure is
//! honestly *not* a tool call), so a 12-hour run completes and reports a
//! near-zero accuracy that looks like a model regression rather than a typo.
//!
//! This module asks two questions a serving instruct model cannot get wrong and
//! checks the answers. It costs two short completions and turns that 12-hour
//! failure into a 2-second one.
//!
//! It is deliberately **not** a quality measurement. Passing means "the
//! endpoint is wired up and generating sense", nothing more.

use anyhow::Result;
use serde_json::json;
use std::time::Duration;

use crate::http;
use crate::plugin::TargetEndpoint;

/// Whether a run probes the endpoint before it starts.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum CoherencePolicy {
    /// Probe and report. A failed probe is a **warning, never a veto**:
    /// benchmarking a base checkpoint, or a model that phrases answers
    /// unusually, are legitimate things to do on purpose, and a check that
    /// cannot be overruled turns a useful signal into an obstacle.
    #[default]
    Probe,
    /// Do not probe at all — not even the two short completions.
    Skip,
}

/// A question whose answer is not a matter of opinion.
#[derive(Clone, Copy, Debug)]
pub struct Check {
    pub label: &'static str,
    pub prompt: &'static str,
    /// Lower-cased standalone terms; the answer must contain **one** of them. More
    /// than one entry means the same fact has several acceptable spellings, not
    /// that the check is lenient.
    pub accept: &'static [&'static str],
}

/// Two facts, from different faculties: one arithmetic, one recall.
///
/// A model that is loaded but mis-quantized typically fails both; one that is
/// serving the wrong checkpoint usually still passes, which is correct — this
/// probe is not trying to detect that.
pub const CHECKS: &[Check] = &[
    Check {
        label: "arithmetic",
        prompt: "What is 2+2? Reply with only the number.",
        accept: &["4", "four"],
    },
    Check {
        label: "recall",
        prompt: "What is the capital of France? Reply with only the city name.",
        accept: &["paris"],
    },
];

/// What one check produced, kept so a failure can quote it back.
#[derive(Clone, Debug)]
pub struct Answer {
    pub label: &'static str,
    pub answer: String,
    pub passed: bool,
}

/// The outcome of a probe: what was asked, what came back, and whether it fit.
#[derive(Clone, Debug, Default)]
pub struct Report {
    pub answers: Vec<Answer>,
    /// Set when the endpoint could not be reached or refused the request. This
    /// is a different diagnosis from a wrong answer and is worded differently.
    pub transport_error: Option<String>,
    /// Set when the served model is not one the benchmark is DEFINED on.
    /// Distinct from `served_instead`: the name may be exactly what was asked
    /// for and still be the wrong model for this particular gate.
    pub wrong_family: Option<String>,
    /// What `/v1/models` says it is serving, when the requested model is not
    /// among them. **This is the check that catches a wrong `--model`:** Atlas
    /// answers a completion whatever name you send, so the questions below
    /// cannot see the mistake — only the model list can.
    pub served_instead: Option<Vec<String>>,
}

impl Report {
    /// Did every check pass and nothing go wrong on the wire?
    pub fn is_clean(&self) -> bool {
        self.transport_error.is_none()
            && self.wrong_family.is_none()
            && self.served_instead.is_none()
            && self.answers.iter().all(|a| a.passed)
    }

    /// One line naming what is wrong, or `None` when nothing is.
    ///
    /// Deliberately describes rather than forbids: a benchmark aimed at a
    /// different model is a legitimate thing to run on purpose, and a probe
    /// that cannot be overruled would block it.
    pub fn concern(&self, target: &TargetEndpoint) -> Option<String> {
        if let Some(e) = &self.transport_error {
            return Some(format!(
                "{} did not answer a test request: {e}",
                target.base_url
            ));
        }
        // A model the gate is not defined on outranks an odd answer for the
        // same reason a wrong name does: it explains the numbers before they
        // are measured.
        if let Some(note) = &self.wrong_family {
            return Some(note.clone());
        }
        // Reported before the answers: a wrong model name explains everything
        // downstream of it, and leading with "recall answered oddly" would bury
        // the cause under a symptom.
        if let Some(served) = &self.served_instead {
            // Serving *nothing* is a different situation from serving the wrong
            // thing, and the wrong-model wording is actively false about it:
            // there is no model to answer to a different name, so the run will
            // not produce numbers at all. It will produce 503s.
            if served.is_empty() {
                return Some(format!(
                    "{} has no model loaded, so this run will produce no numbers — every \
                     request will be refused. Load a model first: in the dashboard open the \
                     Library (press 4), choose a model and a recipe, and start it.",
                    target.base_url
                ));
            }
            return Some(format!(
                "{} is serving {} — not {:?}, which this benchmark is set to request. \
                 Atlas answers whatever model name it is sent, so the run WILL produce \
                 numbers; they will just be for a different model than the one named.",
                target.base_url,
                served.join(", "),
                target.model
            ));
        }
        let failed: Vec<&Answer> = self.answers.iter().filter(|a| !a.passed).collect();
        if failed.is_empty() {
            return None;
        }
        let detail = failed
            .iter()
            .map(|a| match a.answer.trim() {
                "" => format!("{} answered nothing", a.label),
                text => format!("{} answered {:?}", a.label, truncate(text, 60)),
            })
            .collect::<Vec<_>>()
            .join("; ");
        Some(format!(
            "{} is serving {:?}, which did not answer as expected ({detail}). \
             This benchmark may be aimed at a different model, or the checkpoint \
             may be a base (non-instruct) one — the run is still valid, but read \
             the numbers with that in mind.",
            target.base_url, target.model
        ))
    }
}

/// Ask every [`CHECKS`] question and report what came back.
///
/// **Never fails the run.** A wrong answer is information, not a veto: pointing
/// a latency sweep at a base checkpoint, or a tool-calling benchmark at a model
/// that phrases things unusually, are both things people do on purpose. A
/// transport error is captured the same way rather than propagated, so the
/// caller has one thing to inspect.
pub async fn probe(target: &TargetEndpoint, timeout: Duration) -> Report {
    probe_for(target, None, timeout).await
}

/// As [`probe`], and additionally check the served model against what
/// `expectation` says this benchmark is defined on.
///
/// The two questions catch a broken endpoint; the model list catches a wrong
/// name; only this catches the case where the name is exactly what was asked
/// for and is still the wrong model for the gate being run — Gate A pointed at
/// the dense 27B, say, whose thresholds were measured on the 35B MoE.
pub async fn probe_for(
    target: &TargetEndpoint,
    expectation: Option<crate::benchmark::ModelExpectation>,
    timeout: Duration,
) -> Report {
    let mut report = Report::default();

    // The model list first. It is one cheap request and it is the only thing
    // that can catch a wrong name, because a completion succeeds regardless.
    match http::list_models(target, timeout).await {
        Ok(served) if !served.iter().any(|m| m == &target.model) => {
            report.served_instead = Some(served);
        }
        Ok(_) => {}
        // Unreadable list: not fatal, and not worth a warning of its own — the
        // questions below will fail too if the endpoint is genuinely broken.
        Err(e) => tracing::debug!("could not read the model list: {e:#}"),
    }

    // Check the family against what the server actually serves where possible,
    // falling back to the requested name: the served id is the truth, but an
    // unreadable list must not disable the check.
    if let Some(expect) = expectation {
        let actual = report
            .served_instead
            .as_ref()
            .and_then(|s| s.first().cloned())
            .unwrap_or_else(|| target.model.clone());
        if !expect.accepts(&actual) {
            report.wrong_family = Some(format!(
                "{} is serving {actual}, which this benchmark is not defined on. {}",
                target.base_url, expect.note
            ));
        }
    }

    for check in CHECKS {
        match ask(target, check, timeout).await {
            Ok(answer) => report.answers.push(answer),
            Err(e) => {
                report.transport_error = Some(one_line(&format!("{e:#}")));
                break;
            }
        }
    }
    report
}

/// Collapse an error chain to one flowing line, bounded.
///
/// The bound is not "what a line holds" — the pre-flight modal wraps this to
/// as many lines as it needs. It is a guard against an unbounded error chain
/// filling the screen. 140 was tight enough that adding the actionable half of
/// a message cut it off mid-clause ("choose a model and a…"), which is the one
/// part a reader most needs intact.
fn one_line(s: &str) -> String {
    let flat = s.lines().map(str::trim).collect::<Vec<_>>().join(" ");
    truncate(&flat, 280)
}

/// One question. A transport or HTTP error propagates rather than counting as a
/// failed answer: "the server rejected the request" and "the model said the
/// wrong thing" are different diagnoses and must not share a message.
async fn ask(target: &TargetEndpoint, check: &Check, timeout: Duration) -> Result<Answer> {
    let body = json!({
        "model": target.model,
        "stream": true,
        // Thinking OFF, and a budget with room to spare even if the model
        // ignores that. Both prompts say "reply with only the …", so on a
        // non-thinking model 32 was already plenty -- but on a THINKING one the
        // whole budget goes to reasoning and `text` comes back EMPTY, which
        // this probe used to report as "answered nothing" and blame on a
        // mis-quantized or base checkpoint. It measured the model's verbosity
        // and called it brain damage.
        "chat_template_kwargs": {"enable_thinking": false},
        "max_tokens": 96,
        "temperature": 0.0,
        "messages": [{"role": "user", "content": check.prompt}],
    });
    let outcome = http::chat_stream(target, &body, timeout).await?;
    // Accept the fact wherever it appears. A model that reasons its way to
    // "4" has demonstrated exactly what this probe tests -- that it is not
    // emitting garbage -- and a server that ignores `enable_thinking` must not
    // turn a healthy checkpoint into a warning about a broken one.
    let (passed, answer) = judge(&outcome.text, &outcome.reasoning, check.accept);
    Ok(Answer {
        label: check.label,
        passed,
        answer,
    })
}

/// Did the reply contain the expected fact, and what should be quoted back?
///
/// Searches BOTH halves. A model that reasons its way to "4" has demonstrated
/// exactly what this probe tests -- that it is not emitting garbage -- and a
/// server that ignores `enable_thinking` must not turn a healthy checkpoint
/// into a warning about a broken one. The quoted answer prefers `text` and
/// falls back to the reasoning, so a genuine failure stays legible instead of
/// being reported as an empty string.
fn judge(text: &str, reasoning: &str, accept: &[&str]) -> (bool, String) {
    let matched = |s: &str| {
        let lowered = s.to_lowercase();
        accept.iter().any(|term| {
            lowered.match_indices(term).any(|(at, _)| {
                let before = lowered[..at].chars().next_back();
                let after = lowered[at + term.len()..].chars().next();
                let continues_word = |ch: char| ch.is_alphanumeric() || ch == '_';
                !before.is_some_and(continues_word) && !after.is_some_and(continues_word)
            })
        })
    };
    let passed = matched(text) || matched(reasoning);
    let answer = if text.trim().is_empty() {
        reasoning.to_string()
    } else {
        text.to_string()
    };
    (passed, answer)
}

fn truncate(s: &str, max: usize) -> String {
    let trimmed = s.trim();
    if trimmed.chars().count() <= max {
        return trimmed.to_string();
    }
    let head: String = trimmed.chars().take(max).collect();
    format!("{head}…")
}

#[cfg(test)]
#[path = "coherence_tests.rs"]
mod tests;
