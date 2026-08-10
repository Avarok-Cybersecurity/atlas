// SPDX-License-Identifier: AGPL-3.0-only

//! Generic, model- and client-agnostic loop detection for assistant
//! turns.
//!
//! Replaces what used to be four hand-rolled "layers" in `api.rs`,
//! each chasing a specific failure shape (identical tool args,
//! identical text+tool combo, identical text prefix, etc.). Every new
//! observed failure pattern grew the layer count and introduced
//! per-shape thresholds that interacted badly (a fix for one shape
//! triggered false positives on another). This module replaces all of
//! it with a single content-similarity measure.
//!
//! The contract: given the recent assistant messages of a
//! conversation, return a `LoopState` describing whether the model is
//! repeating itself and how strongly. The detector is oblivious to
//! the client (Claude Code, opencode, raw OpenAI) and to the tool
//! schema — it operates only on text + tool-arg JSON, treated as
//! opaque strings.
//!
//! ## Method
//!
//! 1. For each recent assistant message, build a multi-channel
//!    `Signature` of n-gram shingle hash sets:
//!      - `text`: 4-token shingles over the message's prose content
//!      - `tools`: 4-token shingles over the concatenated tool-call
//!        names + JSON arguments
//!      - `combined`: 4-token shingles over the union of the two
//!    Empty channels produce empty sets.
//!
//! 2. For the most recent K signatures, compute pairwise Jaccard
//!    similarity per channel. The detector reports the strongest
//!    consecutive-similarity run found in any channel — a model
//!    repeating only its prose intro across turns hits via the `text`
//!    channel; a model repeating identical bash commands hits via
//!    `tools`; a model emitting near-identical (text + slight tool
//!    drift) hits via `combined`.
//!
//! 3. The report is a single `LoopState` enum:
//!      - `None` — no significant similarity
//!      - `Hint` — moderate similarity (>= 0.55) over >= 3 turns;
//!        recommend injecting a "verify before retrying" notice but
//!        do NOT hard-suppress tool emission
//!      - `Suppress` — high similarity (>= 0.80) over >= 3 turns OR
//!        moderate similarity over >= 4 turns; recommend masking
//!        `<tool_call>` for one turn so the model breaks out
//!
//! Thresholds and the n-gram order are tuned globally; there is no
//! per-shape knob to drift.

use std::collections::HashSet;
use std::hash::{Hash, Hasher};

/// Number of consecutive tokens in each shingle. Order 4 gives
/// strong "the model said almost the same paragraph twice"
/// discrimination without firing on trivial 2-3 word echoes that
/// would naturally recur in code generation.
const SHINGLE_ORDER: usize = 4;

/// How many recent assistant messages to inspect.
const RECENT_WINDOW: usize = 5;

/// Below this length the message has too little signal to shingle
/// meaningfully. Channels shorter than this threshold are zeroed.
const MIN_CHANNEL_TOKENS: usize = 8;

/// Pairwise similarity above which a turn pair is considered "near
/// identical". Matches "same intro phrase + similar tool args".
///
/// Lowered 0.80 → 0.65 (2026-04-25): empirical Jaccard on the
/// claude-export.txt failure (dump seq=54) showed paraphrased intros
/// scoring 0.62–0.73 — clearly looping but never reaching 0.80. The
/// 0.80 bar required verbatim copying that real models don't produce
/// at non-zero temperature. 0.65 catches paraphrase loops without
/// firing on legitimate iterative refactors (which typically score
/// 0.45–0.60 across turns).
const HIGH_SIMILARITY: f64 = 0.65;

/// Lower band — "definitely on the same trajectory" but not
/// identical. Catches "same intent, slight variation".
///
/// Lowered 0.55 → 0.50 (2026-04-25): the seq=54 dump's third pair
/// scored 0.5192 — borderline, broke the run. 0.50 keeps that pair in
/// the run while still rejecting unrelated turns (which typically
/// score 0.10–0.30).
const MODERATE_SIMILARITY: f64 = 0.50;

/// Verdict produced by [`detect`]. The caller chooses how to act.
#[derive(Debug, Clone, PartialEq)]
pub enum LoopState {
    /// No repetitive pattern detected — proceed normally.
    None,
    /// Moderate repetition over the recent window. The caller should
    /// inject a soft hint but should NOT hard-suppress tool emission
    /// (the model often needs verification tools to escape).
    Hint {
        /// Maximum pairwise similarity observed.
        score: f64,
        /// How many consecutive turns are above
        /// [`MODERATE_SIMILARITY`]. >= 3 by definition.
        run_length: usize,
        /// Which channel(s) drove the hit, for logging.
        channel: SimilarityChannel,
    },
    /// Strong repetition — recommend hard-suppressing
    /// `<tool_call>` token emission for the next turn so the model
    /// is forced to produce different content.
    Suppress {
        score: f64,
        run_length: usize,
        channel: SimilarityChannel,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SimilarityChannel {
    Text,
    Tools,
    Combined,
}

impl SimilarityChannel {
    pub fn name(&self) -> &'static str {
        match self {
            Self::Text => "text",
            Self::Tools => "tools",
            Self::Combined => "combined",
        }
    }
}

/// Lightweight assistant-message signature used for similarity
/// comparison. Built once per message and cheap to hash-compare.
#[derive(Debug, Clone, Default)]
pub struct Signature {
    text: HashSet<u64>,
    tools: HashSet<u64>,
    combined: HashSet<u64>,
}

impl Signature {
    /// Build a signature from the message's text content and an
    /// iterator of tool-call (name, json_args) pairs. The arguments
    /// are treated as opaque strings; the detector does not parse
    /// them.
    pub fn build<'a, I>(text: &str, tool_calls: I) -> Self
    where
        I: IntoIterator<Item = (&'a str, &'a str)>,
    {
        let text_tokens = tokenise(text);
        let mut tool_string = String::new();
        for (name, args) in tool_calls {
            if !tool_string.is_empty() {
                tool_string.push('\n');
            }
            tool_string.push_str(name);
            tool_string.push(' ');
            tool_string.push_str(args);
        }
        let tool_tokens = tokenise(&tool_string);

        let text = if text_tokens.len() >= MIN_CHANNEL_TOKENS {
            shingles(&text_tokens, SHINGLE_ORDER)
        } else {
            HashSet::new()
        };
        let tools = if tool_tokens.len() >= MIN_CHANNEL_TOKENS {
            shingles(&tool_tokens, SHINGLE_ORDER)
        } else {
            HashSet::new()
        };
        let combined_tokens: Vec<&str> = text_tokens
            .iter()
            .chain(tool_tokens.iter())
            .copied()
            .collect();
        let combined = if combined_tokens.len() >= MIN_CHANNEL_TOKENS {
            shingles(&combined_tokens, SHINGLE_ORDER)
        } else {
            HashSet::new()
        };
        Self {
            text,
            tools,
            combined,
        }
    }

    /// True iff every channel is empty — the message contributed
    /// nothing measurable. Used to skip from the recent window so
    /// pure noise (heart-beats, pings) doesn't dilute the run.
    pub fn is_empty(&self) -> bool {
        self.text.is_empty() && self.tools.is_empty() && self.combined.is_empty()
    }
}

/// Run the detector over the most recent assistant signatures
/// (newest first). Returns a single verdict — the strongest signal
/// across all three channels wins.
///
/// Caller responsibilities:
///  - Pass signatures in newest-first order.
///  - Filter out signatures from non-assistant turns (system, user,
///    tool-result) before calling — the detector treats every input
///    as a candidate assistant message.
pub fn detect(recent_newest_first: &[Signature]) -> LoopState {
    let recent: Vec<&Signature> = recent_newest_first
        .iter()
        .filter(|s| !s.is_empty())
        .take(RECENT_WINDOW)
        .collect();
    if recent.len() < 3 {
        return LoopState::None;
    }

    // For each channel, compute the pairwise Jaccard similarity
    // between consecutive (newest-first) signatures, then count the
    // longest leading run that meets the moderate threshold.
    let channels = [
        SimilarityChannel::Combined,
        SimilarityChannel::Text,
        SimilarityChannel::Tools,
    ];

    let mut best: Option<(SimilarityChannel, f64, usize, bool)> = None;
    for ch in channels {
        let sims: Vec<f64> = (0..recent.len() - 1)
            .map(|i| {
                let (a, b) = (recent[i], recent[i + 1]);
                let (sa, sb) = match ch {
                    SimilarityChannel::Text => (&a.text, &b.text),
                    SimilarityChannel::Tools => (&a.tools, &b.tools),
                    SimilarityChannel::Combined => (&a.combined, &b.combined),
                };
                jaccard(sa, sb)
            })
            .collect();
        // Leading consecutive run above MODERATE_SIMILARITY.
        let mut run_length = 0;
        let mut max_score = 0.0_f64;
        for &s in &sims {
            if s >= MODERATE_SIMILARITY {
                run_length += 1;
                if s > max_score {
                    max_score = s;
                }
            } else {
                break;
            }
        }
        // run_length is the number of consecutive PAIRS — `n` pairs
        // means `n + 1` turns in the run.
        if run_length == 0 {
            continue;
        }
        let high_pairs = sims
            .iter()
            .take(run_length)
            .filter(|&&s| s >= HIGH_SIMILARITY)
            .count();
        let high_band = high_pairs >= 2 || (high_pairs >= 1 && run_length >= 3);
        let qualifies = run_length >= 2 || max_score >= HIGH_SIMILARITY;
        if !qualifies {
            continue;
        }
        match best {
            Some((_, prev_score, prev_run, prev_high)) => {
                let prefer_new = run_length > prev_run
                    || (run_length == prev_run && max_score > prev_score)
                    || (high_band && !prev_high);
                if prefer_new {
                    best = Some((ch, max_score, run_length, high_band));
                }
            }
            None => best = Some((ch, max_score, run_length, high_band)),
        }
    }

    match best {
        None => LoopState::None,
        Some((channel, score, run_length, high_band)) => {
            // run_length pairs == run_length+1 turns.
            let turns = run_length + 1;
            if high_band || (score >= HIGH_SIMILARITY && turns >= 3) || turns >= 4 {
                LoopState::Suppress {
                    score,
                    run_length: turns,
                    channel,
                }
            } else {
                LoopState::Hint {
                    score,
                    run_length: turns,
                    channel,
                }
            }
        }
    }
}

// ─── P1-5 (2026-07-09): exact-match failing-call fast path ──────────

/// P1-5 (2026-07-09): one recent assistant turn's raw tool-call unit
/// plus whether the tool results that followed it were error-shaped.
/// Built by the orchestrator (api/chat/loop_detect.rs), which sees the
/// interleaved role=="tool" messages that the `Signature`-based
/// detector never receives.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct CallOutcome {
    /// Byte-exact representation of the turn's tool calls
    /// (name + arguments, joined). `None` when the turn issued no
    /// tool calls.
    pub call_unit: Option<String>,
    /// True iff at least one tool result followed the call AND every
    /// one of them was error-shaped
    /// (`crate::hint_injector::looks_like_error`).
    pub failing: bool,
    /// Concatenated text of the tool results that followed the call
    /// (bounded by the orchestrator). `None` when no result followed.
    /// P1-5b: lets the Suppress gate distinguish a productive
    /// similar-call cycle (cargo check → fix → check: results DIFFER
    /// each round) from a true infinite loop (results identical).
    pub result_unit: Option<String>,
}

/// P1-5b (2026-07-09): true iff the newest `n` turns' tool RESULTS show
/// forward progress — i.e. at least one adjacent pair of results is
/// materially different (Jaccard < 0.9 over 4-gram shingles). A build-
/// fix-rebuild cycle has similar CALLS but different RESULTS each round
/// (new error list); masking `<tool_call>` there corners the model into
/// an empty stop ("..." EOS, 57k session 2026-07-09). A true infinite
/// loop repeats both. Missing results ⇒ NOT progressing (conservative:
/// legacy Suppress applies).
pub fn recent_results_progressing(newest_first: &[CallOutcome], n: usize) -> bool {
    if n < 2 || newest_first.len() < n {
        return false;
    }
    let window = &newest_first[..n];
    if window.iter().any(|c| c.result_unit.is_none()) {
        return false;
    }
    window.windows(2).any(|pair| {
        let a = pair[0].result_unit.as_deref().unwrap_or("");
        let b = pair[1].result_unit.as_deref().unwrap_or("");
        if a.trim() == b.trim() {
            return false;
        }
        jaccard(&shingle_set(a), &shingle_set(b)) < 0.9
    })
}

/// P1-5 (2026-07-09): exact-match fast path for the 45k-collapse
/// shape — the model retries a BYTE-IDENTICAL FAILING tool call whose
/// short (e.g. 3-token empty-args) unit is below MIN_CHANNEL_TOKENS,
/// so [`detect`] is blind and escalates ~4 turns late.
///
/// Returns `Some(run_turns)` when the newest >= 3 turns carry
/// byte-identical tool calls, each followed by error-shaped results.
/// Callers must NOT hard-suppress `<tool_call>` on this signal: the
/// repeated unit is FAILING, and masking `<tool_call>` removes the
/// model's only escape action. Apply only the soft bias decay
/// (tool_call_repeat_count).
pub fn detect_exact_failing_repeat(newest_first: &[CallOutcome]) -> Option<usize> {
    let newest_unit = newest_first.first().and_then(|c| c.call_unit.as_deref())?;
    let run = newest_first
        .iter()
        .take_while(|c| c.failing && c.call_unit.as_deref() == Some(newest_unit))
        .count();
    if run >= 3 { Some(run) } else { None }
}

/// P1-5 (2026-07-09): true iff the newest `n` assistant turns each
/// issued at least one tool call AND every following tool result was
/// error-shaped. Gates the Suppress hard-mask: a repeated FAILING
/// call must not have `<tool_call>` masked (it is the escape action),
/// while repeated SUCCEEDING calls keep the legacy Suppress behavior
/// (the BW1/SPINFIX true-infinite-loop case).
pub fn recent_calls_all_failing(newest_first: &[CallOutcome], n: usize) -> bool {
    n >= 1
        && newest_first.len() >= n
        && newest_first[..n]
            .iter()
            .all(|c| c.call_unit.is_some() && c.failing)
}

/// P1-6 (2026-07-09): shared shingle helper, exposed crate-wide so
/// duplicate-error masking (api/chat/msg_entry.rs) reuses the exact
/// similarity measure this detector uses (SSOT) instead of
/// duplicating tokenise/shingle logic.
pub(crate) fn shingle_set(text: &str) -> HashSet<u64> {
    shingles(&tokenise(text), SHINGLE_ORDER)
}

fn tokenise(s: &str) -> Vec<&str> {
    // Word tokens, lowercased on-the-fly via a fresh string would
    // require allocation per token. Instead we keep references and
    // hash them case-insensitively in `shingles`. Punctuation is
    // dropped — the goal is semantic similarity, not byte equality.
    s.split(|c: char| !c.is_alphanumeric())
        .filter(|t| !t.is_empty())
        .collect()
}

fn shingles(tokens: &[&str], order: usize) -> HashSet<u64> {
    if tokens.len() < order {
        return HashSet::new();
    }
    let mut out = HashSet::with_capacity(tokens.len());
    for window in tokens.windows(order) {
        let mut h = std::collections::hash_map::DefaultHasher::new();
        for tok in window {
            // Lowercase by hashing each lowercase char individually
            // (avoids allocating a lowercase string per token).
            for ch in tok.chars() {
                for lc in ch.to_lowercase() {
                    lc.hash(&mut h);
                }
            }
            // Separator so "ab cd" and "abcd" don't collide.
            0u8.hash(&mut h);
        }
        out.insert(h.finish());
    }
    out
}

// P1-6 (2026-07-09): pub(crate) so msg_entry.rs duplicate-error
// masking shares this measure (SSOT).
pub(crate) fn jaccard(a: &HashSet<u64>, b: &HashSet<u64>) -> f64 {
    if a.is_empty() || b.is_empty() {
        return 0.0;
    }
    let intersection = a.intersection(b).count() as f64;
    let union = a.union(b).count() as f64;
    if union == 0.0 {
        0.0
    } else {
        intersection / union
    }
}

#[cfg(test)]
mod tests;
