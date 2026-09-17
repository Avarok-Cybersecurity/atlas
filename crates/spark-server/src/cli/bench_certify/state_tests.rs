// SPDX-License-Identifier: AGPL-3.0-only

use super::*;
use crate::cli::bench_certify::plan::Estimate;
use avarok_plugin::hardware::policy::Sensitivity;
use std::path::PathBuf;

fn unit(id: &'static str, group: Option<&'static str>) -> Unit {
    unit_shard(id, group, group.map(|_| (0, 4)))
}

fn unit_shard(
    id: &'static str,
    group: Option<&'static str>,
    shard: Option<(usize, usize)>,
) -> Unit {
    Unit {
        id,
        group,
        shard,
        class: Sensitivity::Correctness,
        estimate: Estimate::Declared(10),
        needs_confirmation: false,
        serve_allowance_s: 600,
    }
}

fn four() -> Vec<Unit> {
    vec![
        unit_shard("bfcl-subset", Some("bfcl-subset"), Some((0, 2))),
        unit_shard("bfcl-subset", Some("bfcl-subset"), Some((1, 2))),
        unit("decode-floor", None),
        unit("vision-fidelity", None),
    ]
}

fn passed() -> RunOutcome {
    RunOutcome::Passed {
        record: PathBuf::from("r"),
    }
}

#[test]
fn a_clean_campaign_runs_every_unit_in_order_and_exits_zero() {
    let mut c = Campaign::new(four(), false);
    let mut ran = Vec::new();
    while let Some(i) = c.next_to_start() {
        ran.push(c.units[i].label());
        let out = if c.units[i].group.is_some() {
            RunOutcome::MemberDone {
                record: PathBuf::from("r"),
            }
        } else {
            passed()
        };
        assert!(!c.finished(i, out));
    }
    assert_eq!(
        ran,
        [
            "bfcl-subset[0/2]",
            "bfcl-subset[1/2]",
            "decode-floor",
            "vision-fidelity"
        ]
    );
    let s = c.summary();
    assert_eq!(s.member_done, ["bfcl-subset[0/2]", "bfcl-subset[1/2]"]);
    assert_eq!(s.passed, ["decode-floor", "vision-fidelity"]);
    assert_eq!(c.exit_code(true), 0);
    // The final gate check has the last word: a clean run whose records the
    // coverage check still rejects is not certified.
    assert_eq!(c.exit_code(false), 2);
}

#[test]
fn a_verdict_fail_stops_the_rest_unless_keep_going() {
    let mut c = Campaign::new(four(), false);
    let i = c.next_to_start().unwrap();
    c.finished(
        i,
        RunOutcome::VerdictFail {
            record: None,
            reason: "no".into(),
        },
    );
    assert!(c.stopped());
    assert!(c.next_to_start().is_none());
    let s = c.summary();
    assert_eq!(s.failed.len(), 1);
    assert_eq!(s.skipped.len(), 3);
    assert!(s.skipped[0].1.contains("--keep-going"));
    assert_eq!(c.exit_code(false), 2);

    let mut c = Campaign::new(four(), true);
    let i = c.next_to_start().unwrap();
    c.finished(
        i,
        RunOutcome::VerdictFail {
            record: None,
            reason: "no".into(),
        },
    );
    assert!(!c.stopped());
    assert_eq!(c.next_to_start(), Some(1));
}

#[test]
fn a_retryable_harness_failure_is_retried_exactly_once() {
    let mut c = Campaign::new(four(), false);
    let i = c.next_to_start().unwrap();
    let harness = RunOutcome::Harness {
        reason: "no record".into(),
        retryable: true,
    };
    assert!(c.finished(i, harness.clone()), "first failure → retry");
    assert_eq!(c.next_to_start(), Some(i), "the same unit runs again");
    assert!(!c.finished(i, harness), "second failure → failed");
    assert!(c.stopped());
    assert!(c.summary().failed[0].1.contains("harness"));
}

/// NEGATIVE CONTROL: a non-retryable harness failure and a timeout are never
/// retried.
#[test]
fn non_retryable_failures_are_not_retried() {
    for out in [
        RunOutcome::Harness {
            reason: "wrong sha".into(),
            retryable: false,
        },
        RunOutcome::TimedOut,
    ] {
        let mut c = Campaign::new(four(), true);
        let i = c.next_to_start().unwrap();
        assert!(!c.finished(i, out.clone()), "{out:?}");
        assert!(matches!(c.phase[i], Phase::Failed(_)), "{out:?}");
    }
}

#[test]
fn drift_on_a_perf_path_aborts_with_the_paths_named() {
    let mut c = Campaign::new(four(), true);
    let _ = c.next_to_start();
    let why = c
        .guard(Ok(Drift::PerfPathMoved {
            head: "bbbbbbbbbbbb".into(),
            paths: vec!["crates/x.rs".into()],
        }))
        .unwrap()
        .to_string();
    assert!(why.contains("crates/x.rs"), "{why}");
    assert!(why.contains("bbbbbbbbbb"), "{why}");
    assert!(c.stopped());
    assert_eq!(c.exit_code(false), 3);
    assert_eq!(c.summary().skipped.len(), 3);
}

/// NEGATIVE CONTROL: a guard that cannot answer aborts; it is never "safe".
#[test]
fn a_guard_error_aborts() {
    let mut c = Campaign::new(four(), true);
    assert!(c.guard(Err("fetch failed".into())).is_some());
    assert_eq!(c.exit_code(true), 3);
}

#[test]
fn a_harmless_move_does_not_abort() {
    let mut c = Campaign::new(four(), true);
    assert!(c.guard(Ok(Drift::Unmoved)).is_none());
    assert!(
        c.guard(Ok(Drift::MovedHarmlessly { head: "b".into() }))
            .is_none()
    );
    assert!(!c.stopped());
}

#[test]
fn cancel_aborts_with_code_three() {
    let mut c = Campaign::new(four(), true);
    let i = c.next_to_start().unwrap();
    c.finished(i, RunOutcome::Cancelled);
    assert!(c.stopped());
    assert_eq!(c.summary().aborted.as_deref(), Some("cancelled"));
    assert_eq!(c.exit_code(true), 3);
}
