// SPDX-License-Identifier: AGPL-3.0-only

//! The per-round probes, ported from `tests/single_gpu_suite.py`.
//!
//! I/O only: every probe turns one or two requests into a [`Signal`], and the
//! bars that read them live in `score.rs`. That split is what lets the verdict
//! be unit-tested with no endpoint at all.
//!
//! **`repetition_penalty` is not sent, by any probe, deliberately.** The Python
//! codegen probe passes `repetition_penalty=1.05`, which penalises `\n` — the
//! most repeated token in source code — and collapses the generated function
//! onto three lines instead of eight, i.e. into a syntax error, while the
//! control image is byte-identical in both conditions. A codegen probe whose
//! own sampling breaks the code it grades measures the harness. There is a
//! test asserting the request bodies stay penalty-free.

use std::time::Duration;

use anyhow::Result;
use serde_json::{Value, json};

use super::score::Signal;
use crate::benchmarks::{one_line, stats};
use crate::coherence;
use crate::http;
use crate::plugin::TargetEndpoint;

/// Needle for the long-context probe. Distinctive enough that a model cannot
/// produce it by chance and unusual enough that it will not be tokenized into
/// something the filler already contains.
pub const NEEDLE: &str = "PURPLE-DOLPHIN-42";

/// Sampling every probe shares: greedy, no penalties. Callers add `max_tokens`
/// and the messages.
fn base_body(model: &str, max_tokens: usize) -> Value {
    json!({
        "model": model,
        "stream": true,
        "temperature": 0.0,
        "max_tokens": max_tokens,
    })
}

fn with_user(mut body: Value, prompt: String) -> Value {
    body["messages"] = json!([{"role": "user", "content": prompt}]);
    body
}

/// What one round's coherence leg established.
pub struct Coherence {
    pub passed: usize,
    pub total: usize,
    /// Did `/v1/models` name the checkpoint this round asked for?
    ///
    /// **This is the round's identity, and it has to be a bar of its own.**
    /// Atlas answers a completion under whatever model name it is sent, so a
    /// swap that failed and auto-restored the previous checkpoint still
    /// answers "4" and "Paris" — it scores a clean 2/2 and the whole round is
    /// recorded under a checkpoint that was never loaded. Only the model list
    /// can see that, and only a bar can stop it being a PASS.
    pub identity: Signal,
}

/// Ask the two known-answer questions [`crate::coherence`] already owns, and
/// check the endpoint is serving what this round asked for.
///
/// Reused rather than restated: a second copy of "what is 2+2" drifts from the
/// first.
pub async fn coherence_probe(target: &TargetEndpoint, timeout: Duration) -> (Coherence, String) {
    let report = coherence::probe(target, timeout).await;
    let identity = match (&report.transport_error, &report.served_instead) {
        (Some(e), _) => Signal::Fail(one_line(e)),
        (None, Some(served)) if served.is_empty() => {
            Signal::Fail("the endpoint has no model loaded".into())
        }
        (None, Some(served)) => Signal::Fail(format!(
            "serving {} — not {}, which this round loaded",
            served.join(", "),
            target.model
        )),
        (None, None) => Signal::Pass,
    };
    let detail = report.concern(target).unwrap_or_default();
    (
        Coherence {
            passed: report.answers.iter().filter(|a| a.passed).count(),
            total: coherence::CHECKS.len(),
            identity,
        },
        one_line(detail),
    )
}

/// Structural codegen: does the reply contain a fibonacci function with a body?
///
/// A STRUCTURE check, not an execution check — running model-authored code is
/// what the agentic benchmark exists for and it is gated behind a confirmation
/// there. What this catches is the failure the Python probe was actually
/// finding: a reply whose newlines have collapsed, so the "code" is one line
/// and cannot parse.
pub async fn codegen_probe(target: &TargetEndpoint, timeout: Duration, budget: usize) -> Signal {
    let body = with_user(
        base_body(&target.model, budget),
        "Write a Python function `fib(n)` that returns the n-th Fibonacci number. \
         Reply with the code only, no explanation."
            .into(),
    );
    let outcome = match http::chat_stream(target, &body, timeout).await {
        Ok(o) => o,
        Err(e) => return Signal::Fail(one_line(format!("{e:#}"))),
    };
    match code_shape(&outcome.text) {
        Ok(()) => Signal::Pass,
        Err(why) => Signal::Fail(format!("{why}: {}", one_line(&outcome.text))),
    }
}

/// The structural test, split out so it can be tested without an endpoint.
pub fn code_shape(text: &str) -> Result<(), String> {
    let lines: Vec<&str> = text.lines().collect();
    let Some((start, header)) = lines.iter().enumerate().find(|(_, line)| {
        line.trim_start()
            .strip_prefix("def fib")
            .is_some_and(|rest| rest.trim_start().starts_with('('))
    }) else {
        return Err("no `def fib` in the reply".into());
    };
    if !header.contains(':') {
        return Err("`def fib` has no parameter list or colon".into());
    }
    // The body has to be on its own indented line. This single assertion is
    // what `repetition_penalty` was breaking: penalise `\n` and the function
    // arrives as `def fib(n): return ...` fragments with no structure left.
    let indented = lines
        .iter()
        .skip(start + 1)
        .any(|l| !l.trim().is_empty() && (l.starts_with(' ') || l.starts_with('\t')));
    if !indented {
        return Err("the function has no indented body — the reply collapsed onto one line".into());
    }
    Ok(())
}

/// The weather tool the Python suite uses, in OpenAI shape.
fn weather_tool() -> Value {
    json!([{
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather for a location",
            "parameters": {
                "type": "object",
                "properties": {"location": {"type": "string", "description": "City name"}},
                "required": ["location"],
            },
        },
    }])
}

/// Did the model emit a `get_weather` call naming the city?
///
/// A request the server REFUSES (no parser wired up for this architecture) is
/// `NotApplicable`, matching the Python gate's known-gap tolerance. A request
/// that succeeded and produced no call is a `Fail` — that is a regression.
pub async fn tool_call_probe(target: &TargetEndpoint, timeout: Duration, budget: usize) -> Signal {
    let mut body = with_user(
        base_body(&target.model, budget),
        "What is the weather in Paris?".into(),
    );
    body["tools"] = weather_tool();
    body["tool_choice"] = json!("auto");
    let outcome = match http::chat_stream(target, &body, timeout).await {
        Ok(o) => o,
        Err(e) => {
            // The endpoint rejecting the tools payload is a capability gap,
            // not a wrong answer; a transport failure is neither and must not
            // be excused as one.
            let msg = one_line(format!("{e:#}"));
            return if is_client_rejection(&msg) {
                Signal::NotApplicable(msg)
            } else {
                Signal::Fail(msg)
            };
        }
    };
    score_tool_call(&outcome)
}

fn score_tool_call(outcome: &http::ChatOutcome) -> Signal {
    let Some(call) = outcome.tool_calls.iter().find(|c| !c.name.is_empty()) else {
        return Signal::Fail("no tool call in the reply".into());
    };
    if call.name != "get_weather" {
        return Signal::Fail(format!("called {:?}, not get_weather", call.name));
    }
    match serde_json::from_str::<Value>(&call.arguments) {
        Ok(args) => match args.get("location").and_then(Value::as_str) {
            Some(location) if location.trim().eq_ignore_ascii_case("Paris") => Signal::Pass,
            Some(location) => Signal::Fail(format!("called for {location:?}, not Paris")),
            None => Signal::Fail(format!("arguments carry no location: {}", call.arguments)),
        },
        Err(e) => Signal::Fail(format!("arguments are not JSON ({e}): {}", call.arguments)),
    }
}

/// Did the SERVER reject the request as malformed for it (4xx), as opposed to
/// failing to answer at all?
///
/// The distinction decides between `NotApplicable` (no bar) and `Fail` (a
/// bar), so it must be exact. Matching loose substrings does not work: a
/// `request exceeded 400s` timeout carries "400", a target on port 8400
/// carries it in the URL, and a 502 whose body echoes the request carries
/// "tool" — all three would excuse a dead server as a known parser gap. Only
/// [`crate::http`]'s own non-200 wording counts, and only for a 4xx.
fn is_client_rejection(msg: &str) -> bool {
    let Some(rest) = msg.split_once("endpoint returned \"").map(|(_, r)| r) else {
        return false;
    };
    let status = rest.split('"').next().unwrap_or_default();
    status.split_whitespace().any(|word| {
        word.len() == 3 && word.starts_with('4') && word.chars().all(|c| c.is_ascii_digit())
    })
}

/// Needle-in-a-haystack at `tokens` of context.
pub async fn long_context_probe(
    target: &TargetEndpoint,
    timeout: Duration,
    tokens: usize,
    tag: &str,
) -> Signal {
    let filler = stats::make_prompt(tokens, stats::PromptMode::Natural, tag);
    // Mid-document: a needle at either end is found by a model that only
    // attends to the edges, which is the failure this is looking for.
    let split = filler.len() / 2;
    let cut = (0..=split)
        .rev()
        .find(|i| filler.is_char_boundary(*i))
        .unwrap_or(0);
    let prompt = format!(
        "{}\nThe secret code is {NEEDLE}.\n{}\n\nWhat is the secret code? Reply with the code only.",
        &filler[..cut],
        &filler[cut..]
    );
    let body = with_user(base_body(&target.model, 32), prompt);
    match http::chat_stream(target, &body, timeout).await {
        Ok(o) if o.text.contains(NEEDLE) => Signal::Pass,
        Ok(o) => Signal::Fail(format!("needle not recalled: {}", one_line(&o.text))),
        Err(e) => Signal::Fail(one_line(format!("{e:#}"))),
    }
}

/// Decode tokens/sec on a fixed output budget.
///
/// `None` when the endpoint sent the whole reply in one SSE delta — Atlas
/// batches short replies that way, so there is no inter-token interval to time
/// and reporting a number derived from end-to-end latency would silently mix
/// prefill into a decode figure.
pub async fn tps_probe(
    target: &TargetEndpoint,
    timeout: Duration,
    budget: usize,
    tag: &str,
) -> (Option<f64>, Option<String>) {
    let prompt = stats::make_prompt(128, stats::PromptMode::Count, tag);
    let body = with_user(base_body(&target.model, budget), prompt);
    match http::chat_stream(target, &body, timeout).await {
        Ok(o) => (o.tpot_ms.filter(|v| *v > 0.0).map(|ms| 1000.0 / ms), None),
        Err(e) => (Some(0.0), Some(one_line(format!("{e:#}")))),
    }
}

#[cfg(test)]
#[path = "probes_tests.rs"]
mod tests;
