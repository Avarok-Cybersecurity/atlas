// SPDX-License-Identifier: AGPL-3.0-only

//! The decision logic: collected round verdicts → a single Score → the
//! gate verdict. Pure over data, which is what the tests exercise without
//! a server.
//!
//! The line the verdict draws (and why) lives in `compare.rs`: restore
//! JITTER is a healthy engine property of Marconi's anchor selection and is
//! recorded but passed; restore POISONING collapses the output and fails the
//! gate. This gate exists because the collapsed class shipped once already.

use crate::result::Verdict;

use super::compare::{RoundVerdict, TurnDelta};

/// Everything the report and the gate record read.
#[derive(Debug, Clone)]
pub struct Score {
    pub rounds: usize,
    pub invariant: usize,
    pub jittered: usize,
    pub collapsed: usize,
    pub unmeasured: usize,
    /// Which rounds jittered, with the per-turn length ratios.
    pub jittered_rounds: Vec<(usize, Vec<TurnDelta>)>,
    /// Which rounds collapsed — the poisoning signature.
    pub collapsed_rounds: Vec<(usize, Vec<TurnDelta>)>,
}

/// Reduce the collected replay verdicts to a [`Score`].
pub(super) fn score(replays: &[(usize, RoundVerdict)]) -> Score {
    let count = |f: fn(&RoundVerdict) -> bool| replays.iter().filter(|(_, v)| f(v)).count();
    let jittered_rounds = replays
        .iter()
        .filter_map(|(n, v)| {
            if let RoundVerdict::Jittered { turns } = v {
                Some((*n, turns.clone()))
            } else {
                None
            }
        })
        .collect();
    let collapsed_rounds = replays
        .iter()
        .filter_map(|(n, v)| {
            if let RoundVerdict::Collapsed { turns } = v {
                Some((*n, turns.clone()))
            } else {
                None
            }
        })
        .collect();
    Score {
        rounds: replays.len(),
        invariant: count(|v| matches!(v, RoundVerdict::Invariant)),
        jittered: count(|v| matches!(v, RoundVerdict::Jittered { .. })),
        collapsed: count(|v| matches!(v, RoundVerdict::Collapsed { .. })),
        unmeasured: count(|v| matches!(v, RoundVerdict::Unmeasured { .. })),
        jittered_rounds,
        collapsed_rounds,
    }
}

fn turn_summary(turns: &[TurnDelta]) -> String {
    turns
        .iter()
        .map(|t| {
            format!(
                "turn {} ({} -> {} tokens, finish {:?} -> {:?})",
                t.turn, t.ref_tokens, t.replay_tokens, t.ref_finish, t.replay_finish
            )
        })
        .collect::<Vec<_>>()
        .join("; ")
}

/// The verdict rule, stated once:
/// * ANY collapsed round FAILS — that is the poisoning signature this gate
///   exists to catch (batch4: early-EOS stubs instead of full answers).
/// * ANY unmeasured round FAILS — a transport error means the invariant is
///   unproven for that round, and a gate that cannot prove its invariant
///   must not pass.
/// * Jittered rounds PASS but are recorded: clean main's restore anchor
///   selection jitters turn lengths by a few percent between rounds; that
///   is a healthy engine, and failing it would train people to override
///   the gate on every healthy build.
/// * `rounds` is the configured count, so a short run cannot pass by
///   running fewer replays.
pub(super) fn verdict(s: &Score, rounds: usize) -> Verdict {
    if s.rounds != rounds {
        return Verdict::fail(format!(
            "{} of {} replay rounds completed",
            s.rounds, rounds
        ));
    }
    if s.collapsed > 0 {
        let detail = s
            .collapsed_rounds
            .iter()
            .map(|(n, turns)| format!("round {n}: {}", turn_summary(turns)))
            .collect::<Vec<_>>()
            .join(" | ");
        return Verdict::fail(format!(
            "{} of {} replays COLLAPSED against the reference: {detail} — a restored prefix \
             produced degenerate output (early-EOS or runaway), the SSM state poisoning \
             signature",
            s.collapsed, rounds
        ));
    }
    if s.unmeasured > 0 {
        return Verdict::fail(format!(
            "{} of {} replays were unmeasurable (transport errors) — the replay invariant is \
             unproven for those rounds",
            s.unmeasured, rounds
        ));
    }
    if s.jittered > 0 {
        return Verdict::pass(format!(
            "{} of {} replays byte-identical, {} jittered within bounds (restore anchor \
             selection varies between rounds on a healthy engine), 0 collapsed",
            s.invariant, s.rounds, s.jittered
        ));
    }
    Verdict::pass(format!(
        "{} of {} replays byte-identical to the reference",
        s.invariant, rounds
    ))
}

#[cfg(test)]
#[path = "score_tests.rs"]
mod score_tests;
