// SPDX-License-Identifier: AGPL-3.0-only

//! Round comparison: every replay round against the reference round.
//!
//! The invariant is byte-identity of the comparable transcript, exactly as
//! in the cross-contamination detector — but the QUESTION is different.
//! Contamination asks whether CONCURRENT requests move each other's output.
//! This gate asks whether PREVIOUS requests move a later one: same script,
//! replayed sequentially, must come back identical no matter how much
//! prefix-cache / SSM-snapshot state has accumulated on the server. Any
//! divergence is engine-state corruption by construction — at temperature 0
//! and batch 1 there is no stochastic term that could legitimately differ.

use crate::benchmarks::transcript::Transcript;

/// One compared turn: the reference transcript against one replay of it.
#[derive(Debug, Clone)]
pub struct TurnPair {
    pub turn: usize,
    pub reference: Transcript,
    pub replay: Transcript,
}

/// The outcome of comparing one replay round to the reference round.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RoundVerdict {
    /// Every turn byte-identical.
    Invariant,
    /// At least one turn diverged. Carries the 1-based turn numbers that
    /// differed, for the report.
    Diverged { turns: Vec<usize> },
    /// At least one turn failed to produce a transcript (transport error),
    /// so the round cannot speak to the invariant. Carries the error text
    /// of the first failed turn.
    Unmeasured { reason: String },
}

/// Compare a reference turn list against a replay turn list. A replay shorter
/// than the reference (a turn that errored or was cut) is Unmeasured, never
/// Invariant — a missing turn is not evidence the others held.
pub fn compare_round(reference: &[Transcript], replay: &[Transcript]) -> RoundVerdict {
    if replay.len() != reference.len() {
        return RoundVerdict::Unmeasured {
            reason: format!(
                "replay produced {} turn(s), reference has {}",
                replay.len(),
                reference.len()
            ),
        };
    }
    if reference.is_empty() {
        return RoundVerdict::Unmeasured {
            reason: "reference round has no turns".into(),
        };
    }
    let mut diverged = Vec::new();
    let mut unmeasured: Option<String> = None;
    for (i, (r, p)) in reference.iter().zip(replay).enumerate() {
        if r.completion_tokens == 0 && p.completion_tokens == 0 {
            // Two empty replies are "equal" and prove nothing — the same
            // Unmeasured rule the contamination scorer applies.
            unmeasured = Some(format!("turn {} returned no tokens", i + 1));
            continue;
        }
        if r.canonical() != p.canonical() {
            diverged.push(i + 1);
        }
    }
    if !diverged.is_empty() {
        return RoundVerdict::Diverged { turns: diverged };
    }
    if let Some(reason) = unmeasured {
        return RoundVerdict::Unmeasured { reason };
    }
    RoundVerdict::Invariant
}

/// Where the two transcripts first differ, for the report. Returns a
/// (common prefix length in chars, ref excerpt, replay excerpt) triple.
pub fn first_divergence(reference: &Transcript, replay: &Transcript) -> (usize, String, String) {
    let a = reference.canonical();
    let b = replay.canonical();
    let common_chars = a.chars().zip(b.chars()).take_while(|(x, y)| x == y).count();
    // The char count is NOT a byte offset — walk the indices to convert.
    let excerpt = |s: &str| {
        let byte_start = s
            .char_indices()
            .nth(common_chars)
            .map(|(i, _)| i)
            .unwrap_or(s.len());
        s[byte_start..].chars().take(60).collect()
    };
    (common_chars, excerpt(&a), excerpt(&b))
}

#[cfg(test)]
#[path = "compare_tests.rs"]
mod compare_tests;
