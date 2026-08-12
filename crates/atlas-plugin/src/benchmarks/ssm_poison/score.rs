// SPDX-License-Identifier: AGPL-3.0-only

//! The decision logic: collected round verdicts → a single Score → the
//! gate verdict. Pure over data, which is what the tests exercise without
//! a server.

use crate::result::Verdict;

use super::compare::RoundVerdict;

/// Everything the report and the gate record read.
#[derive(Debug, Clone)]
pub struct Score {
    pub rounds: usize,
    pub invariant: usize,
    pub diverged: usize,
    pub unmeasured: usize,
    /// Which replay rounds diverged, and which turns inside them.
    pub diverged_rounds: Vec<(usize, Vec<usize>)>,
}

/// Reduce the collected replay verdicts to a [`Score`].
pub(super) fn score(replays: &[(usize, RoundVerdict)]) -> Score {
    let invariant = replays
        .iter()
        .filter(|(_, v)| *v == RoundVerdict::Invariant)
        .count();
    let diverged = replays
        .iter()
        .filter(|(_, v)| matches!(v, RoundVerdict::Diverged { .. }))
        .count();
    let unmeasured = replays
        .iter()
        .filter(|(_, v)| matches!(v, RoundVerdict::Unmeasured { .. }))
        .count();
    let diverged_rounds = replays
        .iter()
        .filter_map(|(n, v)| {
            if let RoundVerdict::Diverged { turns } = v {
                Some((*n, turns.clone()))
            } else {
                None
            }
        })
        .collect();
    Score {
        rounds: replays.len(),
        invariant,
        diverged,
        unmeasured,
        diverged_rounds,
    }
}

/// Zero tolerance, in both directions: a diverged round is corruption; an
/// unmeasured round means the invariant was NOT PROVEN for that round, and
/// a gate that cannot prove its invariant must fail. `rounds` is the
/// configured count, so a short run cannot pass by running fewer replays.
pub(super) fn verdict(s: &Score, rounds: usize) -> Verdict {
    if s.rounds != rounds {
        return Verdict::fail(format!(
            "{} of {} replay rounds completed",
            s.rounds, rounds
        ));
    }
    if s.diverged > 0 {
        let detail = s
            .diverged_rounds
            .iter()
            .map(|(n, turns)| format!("round {n} turns {:?}", turns))
            .collect::<Vec<_>>()
            .join("; ");
        return Verdict::fail(format!(
            "{} of {} replays diverged from the reference: {detail} — accumulated server state \
             changed the output of an identical request",
            s.diverged, rounds
        ));
    }
    if s.unmeasured > 0 {
        return Verdict::fail(format!(
            "{} of {} replays were unmeasurable (transport errors) — the replay invariant is \
             unproven for those rounds",
            s.unmeasured, rounds
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
