// SPDX-License-Identifier: AGPL-3.0-only
// provenance-id: 526f6e616c6420522e205374657369616b

//! The draft gate for the wide verify (#974): lookup first, MTP head second.
//!
//! Both sites that hand the wide verify its drafts (the bootstrap propose in
//! `mtp_step` and the re-propose at the end of `verify_mtp_wide::finish`) ask
//! the lookup index before running the MTP head. A hit fills
//! `pending_drafts` at the width the head would have drafted, marks the
//! sequence, and the head does not run that step. The verify, accept, commit
//! and rollback are the same code either way.
//!
//! What a lookup step does to the drafter: nothing. No drafter rows are
//! written for lookup drafts, so `finish` skips the proposer trim for that
//! step (the Flash-Next head rewinds `drafted - accepted`, and both are zero
//! here). The drafter's own KV then lacks the positions the lookup step
//! committed, the same gap a suspended step leaves today; the next MTP
//! propose reads the target's highway for the row it drafts from, as always.
//!
//! Off under DFlash (its own drafter and seam), under a grammar (the wide
//! path clamps drafts there), at widths the wide verify does not dispatch,
//! and whenever more than one sequence is active.

use super::sched_ctx::SchedCtx;
use super::types::ActiveSeq;

/// Widths the wide verify dispatches: K=3 and K=4 rows.
const WIDE_WIDTHS: std::ops::RangeInclusive<usize> = 2..=3;

/// Try the lookup index for `width` drafts. `true` means `pending_drafts` is
/// set and the caller skips the MTP propose this step.
pub(super) fn take_lookup_drafts(
    seq: &mut ActiveSeq,
    sched: &SchedCtx,
    width: usize,
    dflash: bool,
    ep: bool,
) -> bool {
    // Off under expert parallelism: the drafter runs on rank 0 only, and a
    // lookup step on rank 0 moved the Marconi prefix-cache anchors on both
    // ranks of a TP=2 x EP=2 build (Richard's bisect, 2026-09-12). Until
    // that is understood the gate stays single-rank.
    if !sched.levers.lookup_drafts
        || dflash
        || ep
        || seq.grammar_state.is_some()
        || !WIDE_WIDTHS.contains(&width)
    {
        return false;
    }
    let mut lookup = sched.lookup.borrow_mut();
    if !lookup.single_sequence() {
        return false;
    }
    // `seq.tokens` holds the rows the model has consumed; the newest token
    // is `last_token`, emitted and not yet fed. Drafts follow it.
    let drafts = lookup.propose(&seq.seq.tokens, seq.last_token, width);
    if drafts.is_empty() {
        return false;
    }
    tracing::debug!(
        "lookup drafts: seq_len={} drafts={drafts:?} (fired {} steps)",
        seq.seq.seq_len,
        lookup.fired
    );
    seq.pending_drafts = drafts;
    seq.pending_draft_conf.clear();
    seq.pending_drafts_lookup = true;
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ctx_with(min_match: usize) -> SchedCtx {
        let mut levers = crate::scheduler::levers::SchedLevers::defaults();
        levers.lookup_min_match = min_match;
        SchedCtx::new(
            crate::scheduler::vocab_masks::VocabMasks::default(),
            std::sync::Arc::new(levers),
            std::sync::Arc::new(crate::scheduler::snapshot::SnapshotCell::default()),
            crate::scheduler::limits::SchedLimits::NONE,
            crate::scheduler::helpers::WatchdogParams::default(),
        )
    }

    fn seq_with(tokens: &[u32]) -> ActiveSeq {
        let (mut a, _rx) = crate::scheduler::test_support::active_seq(0, 0);
        a.seq.tokens = tokens.to_vec();
        a.seq.seq_len = tokens.len();
        a
    }

    #[test]
    fn fires_at_the_wide_width_and_marks_the_sequence() {
        let sched = ctx_with(3);
        sched.lookup.borrow_mut().set_single_sequence(true);
        let mut a = seq_with(&[1, 2, 3, 4, 5, 9, 1, 2]);
        a.last_token = 3;
        assert!(take_lookup_drafts(&mut a, &sched, 2, false, false));
        assert_eq!(a.pending_drafts, vec![4, 5]);
        assert!(a.pending_drafts_lookup);
        assert!(a.pending_draft_conf.is_empty());
    }

    #[test]
    fn stays_out_of_dflash_grammar_narrow_and_multi_sequence() {
        let sched = ctx_with(3);
        let t = [1, 2, 3, 4, 5, 9, 1, 2];
        let mut a = seq_with(&t);
        a.last_token = 3;
        assert!(!take_lookup_drafts(&mut a, &sched, 2, false, false), "not armed single");
        sched.lookup.borrow_mut().set_single_sequence(true);
        assert!(!take_lookup_drafts(&mut a, &sched, 2, true, false), "dflash");
        assert!(!take_lookup_drafts(&mut a, &sched, 1, false, false), "K=2 lane");
        assert!(!take_lookup_drafts(&mut a, &sched, 4, false, false), "past the wide verify");
        assert!(!take_lookup_drafts(&mut a, &sched, 2, false, true), "expert parallel");
        assert!(a.pending_drafts.is_empty() && !a.pending_drafts_lookup);
    }

    #[test]
    fn the_kill_switch_restores_the_mtp_head() {
        let mut levers = crate::scheduler::levers::SchedLevers::defaults();
        levers.lookup_drafts = false;
        levers.lookup_min_match = 3;
        let sched = SchedCtx::new(
            crate::scheduler::vocab_masks::VocabMasks::default(),
            std::sync::Arc::new(levers),
            std::sync::Arc::new(crate::scheduler::snapshot::SnapshotCell::default()),
            crate::scheduler::limits::SchedLimits::NONE,
            crate::scheduler::helpers::WatchdogParams::default(),
        );
        sched.lookup.borrow_mut().set_single_sequence(true);
        let mut a = seq_with(&[1, 2, 3, 4, 5, 9, 1, 2]);
        a.last_token = 3;
        assert!(!take_lookup_drafts(&mut a, &sched, 2, false, false));
    }
}
