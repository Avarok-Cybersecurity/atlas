// SPDX-License-Identifier: AGPL-3.0-only

//! Round comparison: every replay round against the reference round.
//!
//! # Two failure classes, deliberately separated
//!
//! The first recorded run of this gate (2026-08-12, clean main, snapshot pool
//! pinned) proved that byte-identity is NOT achievable on a healthy engine:
//! Marconi restores the same target token from ALTERNATING anchors across
//! rounds (anchor 1040 then 1088 for one turn), so the SSM replay length —
//! and therefore the floating-point accumulation — differs between rounds,
//! and turns 2-4 come back merely *reworded*. Turn 1, the fresh prefill, is
//! byte-identical every round; only the restore path jitters.
//!
//! The bug this gate exists to police (batch4, 2026-08-11) is a different
//! class: a POISONED snapshot makes the restored state garbage, and the
//! generation COLLAPSES — early-EOS, a handful of turns where there should
//! be a full answer. So the comparison splits the classes:
//!
//! * [`RoundVerdict::Invariant`] — byte-identical.
//! * [`RoundVerdict::Jittered`] — different bytes, healthy shape: same
//!   finish reason, replay length within [`COLLAPSE_RATIO_FLOOR`]..
//!   [`COLLAPSE_RATIO_CEIL`] of the reference. Benign restore jitter.
//! * [`RoundVerdict::Collapsed`] — the poisoning signature: the replay is
//!   drastically shorter (or longer) than the reference, or ends for a
//!   different reason.
//!
//! The gate FAILS on any collapse and tolerates jitter — exactly the line
//! between the shipped bug and a healthy build's restore geometry.

use crate::benchmarks::transcript::Transcript;

/// A replay shorter than this fraction of the reference is a collapse.
/// batch4's poisoned replays produced early-EOS stubs; healthy jitter moves
/// length by a few percent, not by halves.
pub const COLLAPSE_RATIO_FLOOR: f64 = 0.5;
/// A replay longer than this multiple of the reference is a collapse too:
/// poisoning can also manifest as runaway generation that hits the token
/// budget instead of stopping.
pub const COLLAPSE_RATIO_CEIL: f64 = 2.0;

/// One compared turn's shape.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TurnDelta {
    pub turn: usize,
    pub ref_tokens: usize,
    pub replay_tokens: usize,
    pub ref_finish: Option<String>,
    pub replay_finish: Option<String>,
}

impl TurnDelta {
    /// Poisoning signature: drastically different length, or a different
    /// finish reason (an early-EOS replay ends where the reference did not,
    /// and vice versa).
    pub fn is_collapse(&self) -> bool {
        if self.ref_finish != self.replay_finish {
            return true;
        }
        if self.ref_tokens == 0 {
            // A nonzero replay against a zero-token reference is an infinite
            // length ratio — above any ceiling, a collapse by definition.
            // Returning false here (as this branch once did) let an unbounded
            // blowup pass as Jittered. The both-zero pair is the only shape
            // that genuinely cannot collapse; the empty-reply Unmeasured rule
            // upstream keeps it from counting as evidence either way.
            return self.replay_tokens != 0;
        }
        let ratio = self.replay_tokens as f64 / self.ref_tokens as f64;
        !(COLLAPSE_RATIO_FLOOR..=COLLAPSE_RATIO_CEIL).contains(&ratio)
    }
}

/// The outcome of comparing one replay round to the reference round.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RoundVerdict {
    /// Every turn byte-identical.
    Invariant,
    /// At least one turn differs, but every differing turn kept a healthy
    /// shape (same finish reason, length within bounds). Benign restore
    /// jitter — recorded, not failed.
    Jittered { turns: Vec<TurnDelta> },
    /// At least one turn collapsed: drastically different length or a
    /// different finish reason. The poisoning signature.
    Collapsed { turns: Vec<TurnDelta> },
    /// At least one turn failed to produce a transcript (transport error),
    /// so the round cannot speak to the invariant.
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
    let mut jittered = Vec::new();
    let mut collapsed = Vec::new();
    let mut unmeasured: Option<String> = None;
    for (i, (r, p)) in reference.iter().zip(replay).enumerate() {
        if r.completion_tokens == 0 && p.completion_tokens == 0 {
            // Two empty replies are "equal" and prove nothing — the same
            // Unmeasured rule the contamination scorer applies.
            unmeasured = Some(format!("turn {} returned no tokens", i + 1));
            continue;
        }
        if r.canonical() == p.canonical() && r.completion_tokens == p.completion_tokens {
            continue;
        }
        let delta = TurnDelta {
            turn: i + 1,
            ref_tokens: r.completion_tokens,
            replay_tokens: p.completion_tokens,
            ref_finish: r.finish_reason.clone(),
            replay_finish: p.finish_reason.clone(),
        };
        if delta.is_collapse() {
            collapsed.push(delta);
        } else {
            jittered.push(delta);
        }
    }
    if !collapsed.is_empty() {
        return RoundVerdict::Collapsed { turns: collapsed };
    }
    // Unmeasured outranks Jittered: the gate's rule is that ANY unmeasured
    // round fails, and Jittered is a pass. With the order reversed, a round
    // with one empty-pair turn and one jittered turn read as Jittered — the
    // unproven turn hid behind tolerated jitter. Collapsed still comes first:
    // both fail, and the poisoning signature is the more specific finding.
    if let Some(reason) = unmeasured {
        return RoundVerdict::Unmeasured { reason };
    }
    if !jittered.is_empty() {
        return RoundVerdict::Jittered { turns: jittered };
    }
    RoundVerdict::Invariant
}

#[cfg(test)]
#[path = "compare_tests.rs"]
mod compare_tests;
