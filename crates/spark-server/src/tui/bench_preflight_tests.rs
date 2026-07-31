// SPDX-License-Identifier: AGPL-3.0-only

use super::*;
use atlas_plugin::TargetEndpoint;
use atlas_plugin::coherence::Answer;

fn target() -> TargetEndpoint {
    TargetEndpoint::local(8888, "m")
}

/// Build a Preflight whose answer is already waiting, without a runtime.
fn resolved(report: Report) -> Preflight {
    let (tx, rx) = channel();
    tx.send(report).expect("send");
    Preflight {
        phase: Phase::Checking,
        rx: Some(rx),
    }
}

#[test]
fn a_clean_check_starts_the_run_without_asking() {
    // The common case must cost nothing but a flicker.
    let mut pre = resolved(Report {
        answers: vec![Answer {
            label: "recall",
            answer: "Paris".into(),
            passed: true,
        }],
        transport_error: None,
        served_instead: None,
        wrong_family: None,
    });
    assert_eq!(pre.poll(&target()), Some(true));
}

#[test]
fn a_concern_stops_to_ask_and_keeps_the_reason() {
    let mut pre = resolved(Report {
        answers: vec![Answer {
            label: "recall",
            answer: String::new(),
            passed: false,
        }],
        transport_error: None,
        served_instead: None,
        wrong_family: None,
    });
    assert_eq!(pre.poll(&target()), Some(false));
    match &pre.phase {
        Phase::Concern(text) => {
            assert!(text.contains("answered nothing"), "{text}");
            assert!(text.contains("still valid"), "it is a warning: {text}");
        }
        other => panic!("expected a concern, got {other:?}"),
    }
    assert!(!pre.is_checking());
}

#[test]
fn waiting_reports_nothing_yet() {
    let (_tx, rx) = channel::<Report>();
    let mut pre = Preflight {
        phase: Phase::Checking,
        rx: Some(rx),
    };
    assert_eq!(pre.poll(&target()), None);
    assert!(pre.is_checking());
}

#[test]
fn a_dropped_check_lets_the_run_proceed_rather_than_stranding_it() {
    // If the task vanishes there is nothing to report, and leaving the user in
    // a spinner forever is worse than starting.
    let (tx, rx) = channel::<Report>();
    drop(tx);
    let mut pre = Preflight {
        phase: Phase::Checking,
        rx: Some(rx),
    };
    assert_eq!(pre.poll(&target()), Some(true));
}

#[test]
fn polling_after_the_answer_is_harmless() {
    let mut pre = resolved(Report::default());
    assert_eq!(pre.poll(&target()), Some(true));
    assert_eq!(pre.poll(&target()), None, "the receiver is spent");
}
