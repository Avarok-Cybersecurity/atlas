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

use anyhow::{Result, bail};
use serde_json::json;
use std::time::Duration;

use crate::http;
use crate::plugin::TargetEndpoint;

/// Whether a run insists on a coherent endpoint before it starts.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum CoherencePolicy {
    /// Probe, and refuse to start if it fails. The default: every benchmark in
    /// the suite targets an instruct model, and none produce a meaningful
    /// number against an endpoint that cannot answer.
    #[default]
    Require,
    /// Skip the probe. For a base (non-instruct) checkpoint, where a pure
    /// latency measurement is still valid but the answers would be nonsense.
    Skip,
}

/// A question whose answer is not a matter of opinion.
#[derive(Clone, Copy, Debug)]
pub struct Check {
    pub label: &'static str,
    pub prompt: &'static str,
    /// Lower-cased substrings; the answer must contain **one** of them. More
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

/// Ask every [`CHECKS`] question and require all of them.
///
/// Returns the answers on success so the caller can log what it saw — a probe
/// that passes silently teaches nobody what "passing" looked like.
pub async fn verify(target: &TargetEndpoint, timeout: Duration) -> Result<Vec<Answer>> {
    let mut answers = Vec::with_capacity(CHECKS.len());
    for check in CHECKS {
        answers.push(ask(target, check, timeout).await?);
    }
    let failed: Vec<&Answer> = answers.iter().filter(|a| !a.passed).collect();
    if !failed.is_empty() {
        let detail = failed
            .iter()
            .map(|a| format!("{} answered {:?}", a.label, truncate(&a.answer, 80)))
            .collect::<Vec<_>>()
            .join("; ");
        bail!(
            "{} is serving {:?}, but it failed the coherence probe: {detail}. \
             The endpoint is reachable, so this is the model — a wrong --model, \
             a broken quantization, or a base (non-instruct) checkpoint. \
             Pass --skip-coherence-probe to measure it anyway.",
            target.base_url,
            target.model
        );
    }
    Ok(answers)
}

/// One question. A transport or HTTP error propagates rather than counting as a
/// failed answer: "the server rejected the request" and "the model said the
/// wrong thing" are different diagnoses and must not share a message.
async fn ask(target: &TargetEndpoint, check: &Check, timeout: Duration) -> Result<Answer> {
    let body = json!({
        "model": target.model,
        "stream": true,
        "max_tokens": 32,
        "temperature": 0.0,
        "messages": [{"role": "user", "content": check.prompt}],
    });
    let outcome = http::chat_stream(target, &body, timeout).await?;
    let lowered = outcome.text.to_lowercase();
    Ok(Answer {
        label: check.label,
        passed: check.accept.iter().any(|a| lowered.contains(a)),
        answer: outcome.text,
    })
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
