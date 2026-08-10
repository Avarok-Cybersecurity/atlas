// SPDX-License-Identifier: AGPL-3.0-only

//! The warm-up sanity check, ported from `run_tier.sh:159-182`.
//!
//! The harness header states the reason plainly: a direct API probe asserts the
//! model answers `4` for `2+2` and "HALTS on failure — saves the operator from
//! waiting 25 min on a catastrophic regression."
//!
//! `/v1/models` returning 200 only proves a server is listening. It says
//! nothing about whether the checkpoint still decodes, which is exactly the
//! failure that costs a whole tier.

use std::time::Duration;

use anyhow::{Result, bail};
use serde_json::json;

use crate::http;
use crate::plugin::PluginHandle;

/// Verbatim from the harness's `warmup_endpoint` body.
pub const SANITY_PROMPT: &str = "What is 2+2? Respond with just the number.";

/// `run_tier.sh:168-170` merges `content` **and** `reasoning_content` before
/// grepping — "some configs route to reasoning", so grading the content field
/// alone would fail a thinking model that answered correctly.
pub fn answered(text: &str, reasoning: &str) -> bool {
    text.contains('4') || reasoning.contains('4')
}

/// Ask the endpoint 2+2 and fail the run if it cannot say 4.
pub async fn sanity_check(handle: &PluginHandle, timeout: Duration) -> Result<()> {
    let target = handle.target();
    let body = json!({
        "model": target.model,
        "messages": [{"role": "user", "content": SANITY_PROMPT}],
        // 80 tokens, as the harness allows. Thinking is disabled for THIS probe
        // only (and only here) because 80 tokens is not a thinking budget — the
        // Gate A trajectories themselves keep thinking on, per the module docs.
        // `chat_template_kwargs.enable_thinking` is the key that is honoured;
        // a bare `thinking` field is silently ignored.
        "chat_template_kwargs": {"enable_thinking": false},
        "max_tokens": 80,
        "temperature": 0.0,
        "stream": true,
    });
    let outcome = http::chat_stream(target, &body, timeout).await?;
    if outcome.text.trim().is_empty() && outcome.reasoning.trim().is_empty() {
        bail!(
            "warm-up: {} returned no parseable response",
            target.base_url
        );
    }
    if !answered(&outcome.text, &outcome.reasoning) {
        bail!(
            "warm-up: {} did not answer '4' to 2+2 — catastrophic regression, halting: {:?}",
            target.base_url,
            crate::benchmarks::one_line(format!("{} {}", outcome.text, outcome.reasoning))
        );
    }
    Ok(())
}

// **A repeat-until-settled warm-up was tried here on 2026-08-07, and REJECTED
// on the measurement.** It is written down because the reasoning that leads to
// it is sound and someone will reach for it again.
//
// The observation it was built on holds. Repeating ONE tool-attached request
// against a fresh 35B FP8 serve on the Gate A recipe:
//
// | regime | `--speculative` | identical replies |
// |---|---|---|
// | first 6 requests of a fresh serve | on  | 1/6 (3 distinct) |
// | first 6 requests of a fresh serve | off | 4/6 (2 distinct) |
// | next 6 on the same process        | on  | 6/6 |
// | next 6 on the same process        | off | 6/6 |
//
// So a cold endpoint is not repeatable and a warmed one is — and a tier starts
// a fresh serve and measures iterations 0, 1, 2 straight into the cold regime.
// Sending discardable probes until two replies matched (settling took 3) looks
// like the obvious fix.
//
// It made the gate WORSE, and not marginally: with the warm-up, N=10 scored
// **3/10 webserver_ok · 1/10 followed_directions**, with 8 turns of 90 ending
// in `finish_reason: length` — the model degenerating into a repetition loop
// mid-run. The same binary with the warm-up removed and nothing else changed
// scored **3/3 webserver_ok** with no degeneration at all. Probing leaves
// prefix-cache and SSM-snapshot state behind that the real requests then
// partially match, and a partially matched SSM snapshot is not a cheaper
// prefix — it is the wrong recurrent state.
//
// Two rules follow, both paid for: do not send this endpoint traffic the
// measurement does not need, and A/B any determinism fix against the score
// before shipping it.

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_prompt_is_the_harness_prompt() {
        assert_eq!(SANITY_PROMPT, "What is 2+2? Respond with just the number.");
    }

    #[test]
    fn either_channel_may_carry_the_answer() {
        assert!(answered("4", ""));
        assert!(answered("", "2+2 is 4"));
        assert!(answered("The answer is 4.", ""));
        // A reply that never says 4 is the catastrophic-regression signal.
        assert!(!answered("", ""));
        assert!(!answered("five", "let me think"));
    }
}
