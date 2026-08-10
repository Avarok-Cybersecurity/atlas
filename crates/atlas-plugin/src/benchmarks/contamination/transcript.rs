// SPDX-License-Identifier: AGPL-3.0-only

//! What a request emitted, reduced to the part that must not change.
//!
//! Equality here IS the invariant: at temperature 0 a request's stream must be
//! identical whether it ran alone or beside N others.

use crate::http::ChatOutcome;

/// The comparable part of one reply.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct Transcript {
    /// Kept SEPARATE from `text`, as `ChatOutcome` does. Folding reasoning in
    /// would let a change of thinking hide behind an identical answer.
    pub reasoning: String,
    pub text: String,
    /// `(name, raw arguments)`. Arguments stay RAW — re-serialising JSON
    /// normalises key order and whitespace, which would mask a real difference.
    pub tool_calls: Vec<(String, String)>,
    pub finish_reason: Option<String>,
    /// From `usage.completion_tokens`, never a delta count: Atlas ships a short
    /// reply as ONE SSE delta, so counting deltas under-counts silently.
    pub completion_tokens: usize,
    /// Diagnostic ONLY — never part of equality. It distinguishes "diverged
    /// with identical cache state" (contamination) from "diverged and the cache
    /// state also moved" (eviction under load turned a warm request cold).
    pub cached_prompt_tokens: usize,
}

impl From<&ChatOutcome> for Transcript {
    fn from(o: &ChatOutcome) -> Self {
        Self {
            reasoning: o.reasoning.clone(),
            text: o.text.clone(),
            tool_calls: o
                .tool_calls
                .iter()
                .map(|t| (t.name.clone(), t.arguments.clone()))
                .collect(),
            finish_reason: o.finish_reason.clone(),
            completion_tokens: o.completion_tokens,
            cached_prompt_tokens: o.cached_prompt_tokens,
        }
    }
}

impl Transcript {
    /// Everything a divergence check compares, concatenated in a fixed order.
    ///
    /// Used for longest-common-prefix localisation. `completion_tokens` and
    /// `cached_prompt_tokens` are deliberately absent — they are compared
    /// separately so a count mismatch is reported as its own finding rather
    /// than shifting every character index.
    pub fn canonical(&self) -> String {
        let mut s = String::with_capacity(self.reasoning.len() + self.text.len() + 64);
        s.push_str(&self.reasoning);
        s.push('\u{1}');
        s.push_str(&self.text);
        for (name, args) in &self.tool_calls {
            s.push('\u{2}');
            s.push_str(name);
            s.push('\u{3}');
            s.push_str(args);
        }
        s.push('\u{4}');
        s.push_str(self.finish_reason.as_deref().unwrap_or(""));
        s
    }

    /// Does this reply contain a marker belonging to a DIFFERENT request?
    ///
    /// ★ The absolute detector. A diff needs a reference; a foreign canary is
    /// contamination on its own evidence, which is what makes it worth the
    /// prompt real estate.
    pub fn carries_foreign_canary<'a>(&self, own: &str, all: &[&'a str]) -> Option<&'a str> {
        let hay = self.canonical();
        all.iter().find(|c| **c != own && hay.contains(*c)).copied()
    }
}

/// One request's result. An error is its OWN variant: a failed request must
/// never compare equal to another failed request and read as "identical".
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RequestOutcome {
    Ok(Box<Transcript>),
    Error(String),
}

impl RequestOutcome {
    pub fn transcript(&self) -> Option<&Transcript> {
        match self {
            Self::Ok(t) => Some(t),
            Self::Error(_) => None,
        }
    }
}
