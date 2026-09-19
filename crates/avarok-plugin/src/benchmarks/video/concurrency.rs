// SPDX-License-Identifier: AGPL-3.0-only

//! Does multimodal input survive concurrency?
//!
//! A small, deliberately modest leg: C = 1, 2, 4 of the SAME request in
//! flight, asserting that every reply comes back, comes back CORRECT, and
//! reports the same geometry as the single-stream case.
//!
//! # What this is and is not
//!
//! It is not a performance measurement — `concurrency-sweep` owns that. It is
//! a correctness and survival check, and the correctness half is the point.
//! The vision path has shared, request-spanning state that a single-stream
//! test cannot exercise: one packed encoder output buffer, a shared grid
//! vector indexed by per-request base offsets, and a co-dispatch path that
//! batches several requests' images into one ViT forward. A defect there does
//! not crash — it hands request A the embeddings of request B, which reads as
//! a plausible answer to the wrong question. This repo has shipped exactly
//! that class before (logits-row aliasing across mixed steps), which is why
//! the assertion is "every reply is still correct", not merely "nothing
//! errored".
//!
//! # On the expected timing
//!
//! Vision and video prefill serialize today, so wall time is expected to grow
//! roughly linearly with C. That is recorded, not asserted: turning "should be
//! linear" into a threshold would make an unrelated scheduler improvement fail
//! the run. Batched video encoding is a later question.

use std::time::{Duration, Instant};

use crate::http::{self, ChatOutcome};
use crate::plugin::PluginHandle;

/// The concurrency levels swept. Small on purpose: the goal is to cross the
/// boundary from "one request at a time" to "several sharing the encoder",
/// which C = 2 already does, with C = 4 as corroboration.
pub const LEVELS: &[usize] = &[1, 2, 4];

/// One level's outcome.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LevelResult {
    pub conc: usize,
    /// Replies that came back at all.
    pub returned: usize,
    /// Replies that were also CORRECT — the check that catches cross-request
    /// contamination, which "it responded" cannot.
    pub correct: usize,
    /// Distinct prompt-token counts seen. Must be exactly one: identical
    /// requests that disagree on geometry mean the shared vision buffers were
    /// indexed per-request incorrectly.
    pub distinct_token_counts: usize,
    /// The agreed prompt-token count when there was exactly one. Retained so
    /// later levels can be compared with the single-stream baseline rather
    /// than merely checked for internal agreement.
    pub prompt_tokens: Option<usize>,
    pub wall_ms: u128,
    pub errors: Vec<String>,
}

impl LevelResult {
    pub fn ok(&self) -> bool {
        self.correct == self.conc && self.distinct_token_counts == 1 && self.prompt_tokens.is_some()
    }

    /// A level is clean only when its internally consistent geometry also
    /// matches the C=1 observation.
    pub fn ok_against(&self, baseline_prompt_tokens: usize) -> bool {
        self.ok() && self.prompt_tokens == Some(baseline_prompt_tokens)
    }

    pub fn geometry_detail(&self, baseline_prompt_tokens: Option<usize>) -> String {
        match (self.prompt_tokens, baseline_prompt_tokens) {
            (Some(got), Some(want)) if got != want => {
                format!("{got} prompt tokens, C=1 baseline {want}")
            }
            (Some(got), _) => format!("one geometry ({got} prompt tokens)"),
            (None, _) => format!("{} distinct token counts", self.distinct_token_counts),
        }
    }
}

/// Score the complete configured sweep, including cross-level geometry.
pub fn sweep_ok(results: &[LevelResult]) -> bool {
    let Some(baseline_prompt_tokens) = results.first().and_then(|r| r.prompt_tokens) else {
        return false;
    };
    results.len() == LEVELS.len()
        && results
            .iter()
            .zip(LEVELS)
            .all(|(r, &level)| r.conc == level && r.ok_against(baseline_prompt_tokens))
}

/// Fire `conc` copies of `body` at once and score them with `is_correct`.
pub async fn run_level(
    handle: &PluginHandle,
    body: &serde_json::Value,
    conc: usize,
    timeout: Duration,
    // `Sync` as well as `Fn`: the futures are joined across await points
    // inside a `Send` benchmark future, so a bare `&dyn Fn` makes the whole
    // state machine non-Send.
    is_correct: &(dyn Fn(&str) -> bool + Sync),
) -> LevelResult {
    let start = Instant::now();
    let futures: Vec<_> = (0..conc)
        .map(|_| http::chat_stream(handle.target(), body, timeout))
        .collect();
    let outcomes: Vec<anyhow::Result<ChatOutcome>> = futures::future::join_all(futures).await;
    let wall_ms = start.elapsed().as_millis();

    let mut returned = 0usize;
    let mut correct = 0usize;
    let mut counts: Vec<usize> = Vec::new();
    let mut errors = Vec::new();
    for o in outcomes {
        match o {
            Ok(out) => {
                returned += 1;
                counts.push(out.prompt_tokens);
                if is_correct(out.text.trim()) {
                    correct += 1;
                }
            }
            Err(e) => errors.push(crate::benchmarks::one_line(format!("{e:#}"))),
        }
    }
    counts.sort_unstable();
    counts.dedup();
    let prompt_tokens = (counts.len() == 1).then(|| counts[0]);
    LevelResult {
        conc,
        returned,
        correct,
        distinct_token_counts: counts.len(),
        prompt_tokens,
        wall_ms,
        errors,
    }
}

#[cfg(test)]
#[path = "concurrency_tests.rs"]
mod concurrency_tests;
