// SPDX-License-Identifier: AGPL-3.0-only

//! Path-independence of TOOL CALLS: the same request, reached three ways.
//!
//! # The hole this closes
//!
//! The replay probe next door asserts "identical input → identical output" by
//! replaying ONE script twelve times. Every round therefore has the SAME
//! predecessors, so it can only see state that degrades across repetitions of
//! one prefix — the 2026-08-11 batch4 class. It is structurally blind to the
//! other half of the invariant: **an answer must not depend on WHICH OTHER
//! requests preceded it.**
//!
//! Two further properties keep the replay probe from seeing that half even by
//! accident, and both are deliberate over there:
//!
//! 1. Its script issues **no tool calls at all** — every turn is prose recall.
//! 2. It tolerates [`super::compare::RoundVerdict::Jittered`], because
//!    byte-identity is not achievable on a healthy engine: Marconi restores
//!    the same token from alternating anchors, so prose comes back reworded.
//!
//! A reworded sentence is benign. **A changed tool call is not** — it flips a
//! known-answer test's 0/1 score outright — and at the transcript level the two
//! are indistinguishable. So a defect that moves tool calls walks past the
//! replay probe twice over.
//!
//! # The measurement that motivated this (2026-09-07, issue #936)
//!
//! Running the golden BFCL draw whole, and again split four ways by stride at
//! the same commit against the same serve, **12 of 995 answers changed** — the
//! 4-sample score delta #936 reported is the net after flips cancel. Ten of the
//! twelve were `live_irrelevance`, the subset whose correct answer is to make
//! NO call, and four unrelated prompts emitted the *same* boilerplate
//! `get_current_weather{"location":"San Francisco, CA"}`. An identical canned
//! call appearing across four unrelated samples is a contamination signature,
//! not floating-point drift.
//!
//! `ssm-state-poisoning-gate` passed on both trees throughout.
//!
//! # What this probe does
//!
//! One target turn, whose correct answer is to call NOTHING, reached by three
//! predecessor paths that share the same long prefix:
//!
//! ```text
//! A  prefix → ack → TARGET
//! B  prefix → ack → a turn that calls get_current_weather → TARGET
//! C  prefix → ack → a turn that calls web_search          → TARGET
//! ```
//!
//! The paths differ ONLY in what ran before the target, which is exactly the
//! variable sharding changes. The target's `tool_calls` must be byte-identical
//! across all three. Prose may still differ — this compares the calls alone, so
//! it does not re-litigate the anchor jitter the replay probe already accepts.
//!
//! Path A is the reference. B and C are the perturbation, and they are the two
//! tools that actually appeared in the contamination.
//!
//! ★ WHY THE TARGET IS AN IRRELEVANCE PROMPT. Where the right answer is "no
//! call", any inherited call-shaped state shows up as a call — a signal that
//! cannot be confused with a differently-worded but equally correct answer.
//! It is also where the measured instability concentrated: 10 of 12.

use serde_json::{Value, json};

use crate::benchmarks::transcript::Transcript;

/// The tool schemas offered on every turn of this probe.
///
/// These two are not arbitrary: they are the tools the sharded BFCL run
/// hallucinated on samples the whole run answered correctly. Offering exactly
/// them keeps the probe pointed at the observed defect rather than at a
/// hypothetical one.
pub fn tools() -> Vec<Value> {
    vec![
        json!({
            "type": "function",
            "function": {
                "name": "get_current_weather",
                "description": "Get the current weather for a location.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {"type": "string", "description": "City and state, e.g. San Francisco, CA"}
                    },
                    "required": ["location"]
                }
            }
        }),
        json!({
            "type": "function",
            "function": {
                "name": "web_search",
                "description": "Search the public web for a query string.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "The search query"}
                    },
                    "required": ["query"]
                }
            }
        }),
    ]
}

/// The turn that must call `get_current_weather`, used only to put a tool call
/// into path B's history.
pub const CALLS_WEATHER: &str = "What is the current weather in San Francisco, CA? Use the tool.";

/// The turn that must call `web_search`, used only to put a different tool
/// call into path C's history.
pub const CALLS_SEARCH: &str =
    "Search the web for the VirusTotal API domain report endpoint. Use the tool.";

/// The TARGET. Neither offered tool can answer it, so the correct behaviour is
/// to call nothing and reply in prose.
///
/// Phrased so that no tool is even arguably applicable: the answer is a fact
/// stated in the document the conversation already carries, and neither a
/// weather lookup nor a web search bears on it.
pub const TARGET: &str = "From SYSTEM DOCUMENT 7741-C already in this \
     conversation, state in one sentence what closed membership means. Answer \
     from the document itself.";

/// A labelled predecessor path to [`TARGET`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Path {
    /// prefix → ack → target. The reference.
    Direct,
    /// prefix → ack → a weather call → target.
    AfterWeather,
    /// prefix → ack → a search call → target.
    AfterSearch,
}

impl Path {
    pub const ALL: [Path; 3] = [Path::Direct, Path::AfterWeather, Path::AfterSearch];

    pub fn label(self) -> &'static str {
        match self {
            Path::Direct => "direct",
            Path::AfterWeather => "after-weather",
            Path::AfterSearch => "after-search",
        }
    }

    /// The turn to interpose between the ack and the target, if any.
    pub fn interposed(self) -> Option<&'static str> {
        match self {
            Path::Direct => None,
            Path::AfterWeather => Some(CALLS_WEATHER),
            Path::AfterSearch => Some(CALLS_SEARCH),
        }
    }
}

/// The request body for a tool-offering turn.
///
/// Greedy and seeded exactly like the replay probe's body, for the same
/// reason: the equality asserted here only holds when the sampler cannot
/// introduce variation of its own. `tool_choice: "auto"` is the point of the
/// probe — the model decides whether to call, and that decision is what must
/// not depend on history.
pub fn request_body(model: &str, messages: &[Value], max_tokens: usize) -> Value {
    json!({
        "model": model,
        "stream": true,
        "stream_options": {"include_usage": true},
        "temperature": 0.0,
        "seed": 0,
        "max_tokens": max_tokens,
        "messages": messages,
        "tools": tools(),
        "tool_choice": "auto",
    })
}

/// One path's outcome: the target turn's transcript.
#[derive(Debug, Clone)]
pub struct PathResult {
    pub path: Path,
    pub target: Transcript,
}

impl PathResult {
    /// The comparison key: the tool calls alone, canonicalised.
    ///
    /// Prose is deliberately excluded. The replay probe already established
    /// that reworded prose is a healthy property of anchor selection, and
    /// re-asserting byte-identity over it here would make this probe fail for
    /// a reason it is not about.
    pub fn calls(&self) -> Vec<(String, String)> {
        self.target.tool_calls.clone()
    }
}

/// One path disagreeing with the reference.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Divergence {
    pub path: &'static str,
    pub reference_calls: Vec<(String, String)>,
    pub path_calls: Vec<(String, String)>,
}

impl Divergence {
    /// A one-line reading for the log and the verdict.
    pub fn describe(&self) -> String {
        let fmt = |c: &Vec<(String, String)>| {
            if c.is_empty() {
                "no call".to_string()
            } else {
                c.iter()
                    .map(|(n, a)| format!("{n}({a})"))
                    .collect::<Vec<_>>()
                    .join(" + ")
            }
        };
        format!(
            "{}: reference made {}, this path made {}",
            self.path,
            fmt(&self.reference_calls),
            fmt(&self.path_calls)
        )
    }
}

/// Compare every path against the reference (`Path::Direct`).
///
/// Returns one [`Divergence`] per path whose tool calls differ. An empty
/// result is the invariant holding: the target's decision to call — and what
/// it called — did not depend on what preceded it.
///
/// ★ ZERO TOLERANCE, deliberately, and it is a different rule from the replay
/// probe's. There is no "healthy shape" for a tool call: a different function
/// name or a different argument string is a different answer, and a KAT scores
/// it 0 where the reference scored 1. The jitter band next door exists because
/// prose can be reworded and still be right; a call cannot.
pub fn divergences(results: &[PathResult]) -> Vec<Divergence> {
    let Some(reference) = results.iter().find(|r| r.path == Path::Direct) else {
        return Vec::new();
    };
    let ref_calls = reference.calls();
    results
        .iter()
        .filter(|r| r.path != Path::Direct)
        .filter_map(|r| {
            let calls = r.calls();
            (calls != ref_calls).then(|| Divergence {
                path: r.path.label(),
                reference_calls: ref_calls.clone(),
                path_calls: calls,
            })
        })
        .collect()
}

/// Did the reference path itself hallucinate a call on the irrelevance target?
///
/// Reported separately from [`divergences`] because it is a different finding:
/// divergence says the answer moved with history, this says the answer was
/// wrong on the direct path too. The gate fails on divergence; this is
/// recorded so a reference that is itself calling does not silently become the
/// baseline every other path is compared against.
pub fn reference_called(results: &[PathResult]) -> bool {
    results
        .iter()
        .find(|r| r.path == Path::Direct)
        .is_some_and(|r| !r.calls().is_empty())
}

#[cfg(test)]
#[path = "toolcall_tests.rs"]
mod tests;
