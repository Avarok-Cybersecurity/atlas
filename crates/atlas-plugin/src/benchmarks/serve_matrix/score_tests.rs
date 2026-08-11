// SPDX-License-Identifier: AGPL-3.0-only

//! The bars, and the coverage guarantee.
//!
//! The coverage tests are the point of this file. `tests/test_gate_results.py`
//! exists for the same reason: the failure mode being locked out is a
//! checkpoint that crashed at boot, wrote nothing, and was scored as absent
//! rather than failed — a green matrix over half a matrix.

use super::*;
use crate::benchmarks::serve_matrix::host::{Absence, ServeCandidate};

fn clean() -> Signals {
    Signals {
        identity: Signal::Pass,
        coherence_pass: 2,
        coherence_total: 2,
        codegen: Signal::Pass,
        tool_call: Signal::Pass,
        long_ctx: Signal::Pass,
        tps: Some(100.0),
    }
}

fn result(label: &str, signals: Signals) -> RoundResult {
    RoundResult {
        label: label.into(),
        outcome: Outcome::Probed(Box::new(signals)),
        baseline_tps: None,
    }
}

fn plan_of(labels: &[&str]) -> Plan {
    let roster: Vec<ServeCandidate> = labels
        .iter()
        .map(|l| ServeCandidate::ready(*l, ""))
        .collect();
    Plan::build(&roster, "")
}

#[test]
fn a_clean_round_clears_every_bar() {
    assert!(result("m", clean()).bars().is_empty());
}

#[test]
fn coherence_needs_every_probe_when_there_are_only_two() {
    let mut s = clean();
    s.coherence_pass = 1;
    assert_eq!(result("m", s).bars(), vec!["coherence(1/2)"]);
}

#[test]
fn a_known_gap_tool_parser_is_not_a_failure_but_a_missing_call_is() {
    let mut na = clean();
    na.tool_call = Signal::NotApplicable("no parser for this architecture".into());
    assert!(result("m", na).bars().is_empty());

    let mut missing = clean();
    missing.tool_call = Signal::Fail("no tool call in the reply".into());
    assert_eq!(result("m", missing).bars(), vec!["tool_call"]);
}

#[test]
fn long_context_is_reported_but_does_not_gate() {
    // gate_results.py never scored the long-context leg. Adding a bar here
    // would make this run's PASS incomparable with every recorded one.
    let mut s = clean();
    s.long_ctx = Signal::Fail("needle not recalled".into());
    assert!(result("m", s).bars().is_empty());
}

#[test]
fn a_dead_server_fails_the_throughput_bar_even_with_no_baseline() {
    let mut s = clean();
    s.tps = Some(0.0);
    assert_eq!(result("m", s).bars(), vec!["tps(0)"]);
}

#[test]
fn without_a_baseline_throughput_is_liveness_only_and_says_so() {
    let r = result("m", clean());
    assert!(r.bars().is_empty());
    assert_eq!(r.tps_note(), Some("no baseline — liveness only"));
}

#[test]
fn with_a_baseline_a_ten_percent_drop_is_a_regression() {
    let mut r = result("m", clean());
    r.baseline_tps = Some(120.0);
    // 100 < 120 * 0.9 = 108
    assert_eq!(r.bars(), vec!["tps(100.0<108.0)"]);
    assert_eq!(r.tps_note(), None, "a real bar is not a 'no baseline' note");

    r.baseline_tps = Some(105.0); // floor 94.5, 100 clears it
    assert!(r.bars().is_empty());
}

#[test]
fn unmeasurable_throughput_is_not_a_failure() {
    // One SSE delta means no inter-token interval. That is a measurement
    // limit, not a dead server, and must not be scored as one.
    let mut s = clean();
    s.tps = None;
    assert!(result("m", s).bars().is_empty());
}

#[test]
fn a_round_served_by_the_wrong_checkpoint_fails_however_well_it_answers() {
    // The failure mode this bar exists for: a swap that did not take, or that
    // failed and auto-restored the previous model. Atlas answers a completion
    // under whatever name it is sent, so every other probe still passes and
    // the numbers get filed under a checkpoint that was never loaded.
    let mut s = clean();
    s.identity = Signal::Fail("serving org/previous — not org/this-round".into());
    assert_eq!(result("m", s).bars(), vec!["wrong-model"]);
}

#[test]
fn an_unprobed_signal_fails_rather_than_scoring_clean() {
    // `Signals::default()` is all-`NotRun`. If that scored zero bars, any
    // early return between booting and probing — or any probe that becomes
    // conditional later — would manufacture a verified round.
    let bars = result("m", Signals::default()).bars();
    for expected in [
        "wrong-model(not-probed)",
        "codegen(not-probed)",
        "tool_call(not-probed)",
        "coherence(0/0)",
    ] {
        assert!(
            bars.iter().any(|b| b == expected),
            "{expected} missing from {bars:?}"
        );
    }
}

// ── coverage enforcement ────────────────────────────────────────────────

#[test]
fn a_checkpoint_that_failed_to_boot_is_a_failure_not_an_absence() {
    let plan = plan_of(&["a", "b"]);
    let results = vec![
        result("a", clean()),
        RoundResult {
            label: "b".into(),
            outcome: Outcome::BootFailed("CUDA out of memory".into()),
            baseline_tps: None,
        },
    ];
    let t = tally(&plan, &results);
    assert_eq!((t.verified, t.planned), (1, 2));
    assert!(!t.passed());
    assert_eq!(t.failures.len(), 1);
    assert!(
        t.failures[0].1[0].starts_with("did-not-boot"),
        "the reason has to survive into the verdict: {:?}",
        t.failures[0]
    );
    assert!(verdict_text(&t, &plan).contains("CUDA out of memory"));
}

#[test]
fn a_planned_round_that_produced_no_result_at_all_fails() {
    // THE false-green: score only what was written and a crashed model
    // disappears from the denominator entirely.
    let plan = plan_of(&["a", "b", "c"]);
    let t = tally(&plan, &[result("a", clean())]);
    assert_eq!((t.verified, t.planned), (1, 3));
    assert_eq!(t.failures.len(), 2);
    assert!(t.failures.iter().all(|(_, bars)| bars == &["no-result"]));
    assert!(!t.passed());
}

#[test]
fn a_checkpoint_the_box_cannot_serve_is_skipped_not_failed() {
    let roster = vec![
        ServeCandidate::ready("a", ""),
        ServeCandidate::absent("b", "", Absence::NoWeights),
    ];
    let plan = Plan::build(&roster, "");
    let t = tally(&plan, &[result("a", clean())]);
    assert!(t.passed(), "an absent checkpoint must not fail the matrix");
    assert_eq!((t.verified, t.planned, t.skipped), (1, 1, 1));
    let text = verdict_text(&t, &plan);
    assert!(
        text.contains("not runnable") && text.contains(Absence::NoWeights.reason()),
        "the skip has to be visible in the verdict, or 1/1 lies: {text}"
    );
}

#[test]
fn skipped_and_failed_to_boot_never_read_the_same() {
    let absent = Plan::build(&[ServeCandidate::absent("b", "", Absence::NoWeights)], "");
    let absent_tally = tally(&absent, &[]);

    let planned = plan_of(&["b"]);
    let crashed = tally(
        &planned,
        &[RoundResult {
            label: "b".into(),
            outcome: Outcome::BootFailed("no kernels".into()),
            baseline_tps: None,
        }],
    );
    // Absent: nothing was planned, so nothing is below bar — and nothing is
    // claimed either (`report::verdict` renders that as Info, not PASS).
    assert!(absent_tally.failures.is_empty());
    assert_eq!((absent_tally.planned, absent_tally.skipped), (0, 1));
    // Crashed: planned, attempted, gone. That is a FAIL with a reason.
    assert_eq!(crashed.failures.len(), 1);
    assert_eq!((crashed.planned, crashed.skipped), (1, 0));
    assert!(!crashed.passed());
}

#[test]
fn an_empty_matrix_is_not_a_pass() {
    let plan = Plan::build(&[], "");
    let t = tally(&plan, &[]);
    assert!(
        !t.passed(),
        "zero planned models is nothing measured, not everything verified"
    );
}
