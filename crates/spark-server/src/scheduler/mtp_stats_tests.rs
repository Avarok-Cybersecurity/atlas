// SPDX-License-Identifier: AGPL-3.0-only

//! Accounting tests for [`super::MtpRequestStats`] and [`super::StageClock`].

use super::*;

fn st(d: u64, v: u64, c: u64) -> StageTimes {
    StageTimes {
        draft_us: d,
        verify_us: v,
        commit_us: c,
    }
}

#[test]
fn no_mtp_step_means_no_line() {
    let s = MtpRequestStats::default();
    assert!(!s.speculated());
    assert_eq!(s.summary_line(), None);
    assert_eq!(s.accept_rate(), 0.0);
    assert_eq!(s.tokens_per_pass(), 0.0);
    assert_eq!(s.gate_multiplier(), 0.0);
}

#[test]
fn one_draft_accept_and_reject_are_counted() {
    let mut s = MtpRequestStats::default();
    // Bootstrap: no pending draft, one token.
    s.record_step(0, 1, st(300, 0, 0), 2_000);
    // Seven accepts (2 tokens each) and three rejects (1 token each).
    for _ in 0..7 {
        s.record_step(1, 2, st(400, 25_000, 50), 27_000);
    }
    for _ in 0..3 {
        s.record_step(1, 1, st(400, 25_000, 80), 27_000);
    }
    assert_eq!(s.bootstrap_steps, 1);
    assert_eq!(s.verify_passes, 10);
    assert_eq!(s.proposed, 10);
    assert_eq!(s.accepted, 7);
    assert_eq!(s.committed, 17);
    assert!((s.accept_rate() - 0.7).abs() < 1e-12);
    assert!((s.tokens_per_pass() - 1.7).abs() < 1e-12);
    assert_eq!(s.draft_us, 300 + 10 * 400);
    assert_eq!(s.verify_us, 10 * 25_000);
    assert_eq!(s.commit_us, 7 * 50 + 3 * 80);
    assert_eq!(s.step_us, 2_000 + 10 * 27_000);
    let line = s.summary_line().expect("speculated");
    assert!(line.starts_with("MTP stats: verify_passes=10 bootstrap_steps=1 proposed=10"));
    assert!(line.contains("accepted=7 accept_rate=70.0% tokens_per_pass=1.700"));
    assert!(line.contains("verify_ms=250.0"));
    assert!(line.contains("step_ms=272.0"));
}

#[test]
fn multi_draft_accept_is_capped_at_the_drafts() {
    let mut s = MtpRequestStats::default();
    // Three drafts, all accepted plus the bonus token: 4 committed.
    s.record_step(3, 4, StageTimes::default(), 1);
    // Three drafts, one accepted: 2 committed.
    s.record_step(3, 2, StageTimes::default(), 1);
    // A malformed count (more tokens than drafts + 1) never over-credits.
    s.record_step(1, 5, StageTimes::default(), 1);
    assert_eq!(s.proposed, 7);
    assert_eq!(s.accepted, 3 + 1 + 1);
    assert_eq!(s.committed, 11);
}

#[test]
fn a_bootstrap_only_request_still_reports() {
    let mut s = MtpRequestStats::default();
    s.record_step(0, 1, StageTimes::default(), 10);
    assert!(s.speculated());
    let line = s.summary_line().expect("speculated");
    assert!(line.contains("verify_passes=0 bootstrap_steps=1 proposed=0 accepted=0"));
    assert!(line.contains("accept_rate=0.0%"));
}

#[test]
fn gate_decisions_and_multiplier() {
    let mut s = MtpRequestStats::default();
    s.note_gate_decision(true);
    s.note_gate_decision(false);
    s.note_gate_decision(true);
    assert_eq!((s.gate_disables, s.gate_enables), (2, 1));
    // Only one estimate known: no multiplier.
    s.note_gate_estimates(Some(45.0), None);
    assert_eq!(s.gate_multiplier(), 0.0);
    // A `None` never clears a known estimate.
    s.note_gate_estimates(None, Some(36.0));
    assert_eq!(s.gate_mtp_tps, 45.0);
    // 10 passes committing 17 tokens: 1.7 tokens per MTP step, so the
    // implied verify cost is 1.7 * 36 / 45 = 1.36 decode steps.
    for _ in 0..7 {
        s.record_step(1, 2, StageTimes::default(), 1);
    }
    for _ in 0..3 {
        s.record_step(1, 1, StageTimes::default(), 1);
    }
    assert!((s.gate_multiplier() - 1.36).abs() < 1e-9);
    let line = s.summary_line().expect("speculated");
    assert!(line.contains("gate_disable=2 gate_enable=1"));
    assert!(line.contains("gate_mult=1.360"));
}

#[test]
fn stage_clock_routes_phases_and_resets() {
    let c = StageClock::default();
    c.add_us(0, 5);
    c.add_us(1, 7);
    c.add_us(2, 11);
    assert_eq!(c.take(), st(5, 7, 11));
    assert_eq!(c.take(), StageTimes::default());
    assert_eq!(stage_of(Phase::Propose), Some(0));
    assert_eq!(stage_of(Phase::ProposeMask), Some(0));
    assert_eq!(stage_of(Phase::VerifyForward), Some(1));
    assert_eq!(stage_of(Phase::Commit), Some(2));
    assert_eq!(stage_of(Phase::D2h), None);
    assert_eq!(stage_of(Phase::StepTotal), None);
    // An ignored phase never touches the accumulators.
    c.note(Phase::Argmax, Instant::now());
    assert_eq!(c.take(), StageTimes::default());
}

#[test]
fn stage_clock_feeds_from_a_disarmed_run_timing() {
    // The per-request view must not depend on AVAROK_MTP_TIMING.
    let t = crate::scheduler::mtp_timing::RunTiming::new(false);
    let since = Instant::now() - std::time::Duration::from_millis(3);
    t.record(Phase::VerifyForward, since);
    let got = t.stages.take();
    assert!(got.verify_us >= 3_000, "verify_us={}", got.verify_us);
    assert_eq!(got.draft_us, 0);
}

#[test]
fn account_step_charges_each_sequence_and_returns_the_batch_total() {
    let (mut a, _ra) = crate::scheduler::test_support::active_seq(0, 3);
    let (mut b, _rb) = crate::scheduler::test_support::active_seq(1, 3);
    a.pending_drafts = vec![42]; // a verifies one draft
    b.pending_drafts.clear(); // b bootstraps
    let mut active = vec![a, b];
    let t = crate::scheduler::mtp_timing::RunTiming::new(false);
    let (_wall, total) = account_step(&mut active, &t, |act| {
        // Stand-in for step_mtp: a accepts its draft (+2), b decodes (+1),
        // and the verify forward reports 3 ms of stage time.
        t.record(
            Phase::VerifyForward,
            Instant::now() - std::time::Duration::from_millis(3),
        );
        act[0].seq.seq_len += 2;
        act[1].seq.seq_len += 1;
    });
    assert_eq!(total, 3);
    let sa = active[0].mtp_acct.spec;
    let sb = active[1].mtp_acct.spec;
    assert_eq!(
        (sa.verify_passes, sa.proposed, sa.accepted, sa.committed),
        (1, 1, 1, 2)
    );
    assert_eq!(
        (sb.verify_passes, sb.bootstrap_steps, sb.proposed),
        (0, 1, 0)
    );
    assert!(sa.verify_us >= 3_000 && sb.verify_us == sa.verify_us);
    // The existing Done-line counters still advance.
    assert_eq!(active[0].mtp_acct.accepted_total(), 1);
    // The clock was drained.
    assert_eq!(t.stages.take(), StageTimes::default());
}
