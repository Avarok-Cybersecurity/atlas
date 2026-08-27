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

/// One replay round as the driver collected it: the comparison verdict plus
/// the server-attested cache state of the round's first turn.
#[derive(Debug, Clone)]
pub struct RoundRecord {
    /// 1-based round number over the replays.
    pub round: usize,
    pub verdict: RoundVerdict,
    /// `usage.prompt_tokens_details.cached_tokens` from the round's FIRST
    /// turn — the cross-round prefix restore this gate exists to exercise.
    /// `None` when the replay errored before turn 1 completed (the round is
    /// Unmeasured and fails on that already). `Some(0)` is the finding that
    /// motivated this field: a replay whose turn 1 restored nothing ran the
    /// script against a cold engine, and its "invariant" verdict says nothing
    /// about the poisoning class.
    pub turn1_cached: Option<usize>,
}

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
    /// Which rounds were unmeasured, with the transport reason. Carried per
    /// round so the report attributes the failure to the round that actually
    /// failed, not to the earliest unlabeled row.
    pub unmeasured_rounds: Vec<(usize, String)>,
    /// Rounds whose first turn attested ZERO cached prompt tokens — replays
    /// that never exercised the restore path under test.
    pub vacuous_rounds: Vec<usize>,
    /// Minimum turn-1 cached-token attestation across the replays that
    /// produced one. `None` when no replay got that far.
    pub min_turn1_cached: Option<usize>,
}

/// Reduce the collected replay records to a [`Score`].
pub(super) fn score(replays: &[RoundRecord]) -> Score {
    let count = |f: fn(&RoundVerdict) -> bool| replays.iter().filter(|r| f(&r.verdict)).count();
    let jittered_rounds = replays
        .iter()
        .filter_map(|r| {
            if let RoundVerdict::Jittered { turns } = &r.verdict {
                Some((r.round, turns.clone()))
            } else {
                None
            }
        })
        .collect();
    let collapsed_rounds = replays
        .iter()
        .filter_map(|r| {
            if let RoundVerdict::Collapsed { turns } = &r.verdict {
                Some((r.round, turns.clone()))
            } else {
                None
            }
        })
        .collect();
    let unmeasured_rounds = replays
        .iter()
        .filter_map(|r| {
            if let RoundVerdict::Unmeasured { reason } = &r.verdict {
                Some((r.round, reason.clone()))
            } else {
                None
            }
        })
        .collect();
    let vacuous_rounds = replays
        .iter()
        .filter(|r| r.turn1_cached == Some(0))
        .map(|r| r.round)
        .collect();
    let min_turn1_cached = replays.iter().filter_map(|r| r.turn1_cached).min();
    Score {
        rounds: replays.len(),
        invariant: count(|v| matches!(v, RoundVerdict::Invariant)),
        jittered: count(|v| matches!(v, RoundVerdict::Jittered { .. })),
        collapsed: count(|v| matches!(v, RoundVerdict::Collapsed { .. })),
        unmeasured: count(|v| matches!(v, RoundVerdict::Unmeasured { .. })),
        jittered_rounds,
        collapsed_rounds,
        unmeasured_rounds,
        vacuous_rounds,
        min_turn1_cached,
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
/// * ANY replay whose first turn attested zero cached prompt tokens FAILS —
///   the gate polices the prefix-restore path, and a replay that restored
///   nothing ran against a cold engine. Before this rule, serving with
///   prefix caching disabled produced a green PASS that proved nothing.
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
        let detail = s
            .unmeasured_rounds
            .iter()
            .map(|(round, reason)| format!("round {round}: {reason}"))
            .collect::<Vec<_>>()
            .join(" | ");
        return Verdict::fail(format!(
            "{} of {} replays were unmeasurable (transport errors): {detail} — the replay \
             invariant is unproven for those rounds",
            s.unmeasured, rounds,
        ));
    }
    if !s.vacuous_rounds.is_empty() {
        return Verdict::fail(format!(
            "replay round(s) {:?} attested 0 cached prompt tokens on turn 1 — the prefix \
             restore path this gate polices was never exercised, so their transcripts prove \
             nothing about the poisoning class (is prefix caching enabled on the served \
             recipe?)",
            s.vacuous_rounds
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
