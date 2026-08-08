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

/// The modal shows exactly what `concern` produced, so each refusal reason has
/// to name a different situation and a different thing to do about it. These
/// drive the reasons through the real `poll`, not through the formatter.
fn concern_for(report: Report) -> String {
    let mut pre = resolved(report);
    assert_eq!(
        pre.poll(&target()),
        Some(false),
        "a concern must stop to ask"
    );
    match &pre.phase {
        Phase::Concern(text) => text.clone(),
        other => panic!("expected a concern, got {other:?}"),
    }
}

fn answers(passed: bool) -> Vec<Answer> {
    vec![Answer {
        label: "arithmetic",
        answer: if passed { "4".into() } else { "banana".into() },
        passed,
    }]
}

#[test]
fn an_unreachable_endpoint_names_the_url_and_what_it_said() {
    let text = concern_for(Report {
        transport_error: Some("connection refused".into()),
        ..Report::default()
    });
    assert!(text.contains("http://127.0.0.1:8888"), "{text}");
    assert!(text.contains("connection refused"), "{text}");
}

#[test]
fn a_server_with_nothing_loaded_says_so_and_says_how_to_load_one() {
    // The wrong-model wording is actively false here: there is no model to
    // answer to a different name, so the run produces 503s, not numbers.
    let text = concern_for(Report {
        served_instead: Some(Vec::new()),
        ..Report::default()
    });
    assert!(text.contains("no model loaded"), "{text}");
    assert!(text.contains("Library"), "names the remedy: {text}");
    assert!(
        !text.contains("is serving m"),
        "and not a served name: {text}"
    );
}

#[test]
fn a_server_holding_a_different_model_says_the_numbers_will_still_come() {
    // Atlas answers whatever model name it is sent, so this run WILL produce
    // plausible numbers for the wrong checkpoint — which is the trap.
    let text = concern_for(Report {
        served_instead: Some(vec!["org/other".into()]),
        ..Report::default()
    });
    assert!(text.contains("org/other"), "{text}");
    assert!(text.contains("\"m\""), "names what was requested: {text}");
    assert!(text.contains("different model"), "{text}");
}

#[test]
fn a_model_the_gate_is_not_defined_on_is_reported_in_the_gates_own_words() {
    let text = concern_for(Report {
        wrong_family: Some("gate A is defined on the 35B MoE".into()),
        ..Report::default()
    });
    assert_eq!(text, "gate A is defined on the 35B MoE");
}

#[test]
fn a_wrong_answer_quotes_it_back_and_still_calls_the_run_valid() {
    let text = concern_for(Report {
        answers: answers(false),
        ..Report::default()
    });
    assert!(text.contains("arithmetic answered"), "{text}");
    assert!(text.contains("banana"), "quotes what came back: {text}");
    assert!(text.contains("base (non-instruct)"), "{text}");
    assert!(
        text.contains("still valid"),
        "a warning, not a veto: {text}"
    );
}

#[test]
fn every_refusal_reason_reads_differently() {
    // Five situations with five fixes; two that share wording send the reader
    // to the wrong one.
    let reasons = [
        concern_for(Report {
            transport_error: Some("connection refused".into()),
            ..Report::default()
        }),
        concern_for(Report {
            wrong_family: Some("this gate is defined on the 35B".into()),
            ..Report::default()
        }),
        concern_for(Report {
            served_instead: Some(Vec::new()),
            ..Report::default()
        }),
        concern_for(Report {
            served_instead: Some(vec!["org/other".into()]),
            ..Report::default()
        }),
        concern_for(Report {
            answers: answers(false),
            ..Report::default()
        }),
    ];
    let distinct: std::collections::BTreeSet<&String> = reasons.iter().collect();
    assert_eq!(distinct.len(), reasons.len(), "{reasons:#?}");
    for r in &reasons {
        assert!(!r.is_empty());
    }
}

#[test]
fn the_cause_is_reported_ahead_of_the_symptom_it_explains() {
    // A wrong name explains every odd answer downstream of it; leading with the
    // symptom buries the cause.
    let everything = Report {
        answers: answers(false),
        transport_error: Some("connection refused".into()),
        wrong_family: Some("wrong family".into()),
        served_instead: Some(vec!["org/other".into()]),
    };
    assert!(concern_for(everything.clone()).contains("connection refused"));

    let no_transport = Report {
        transport_error: None,
        ..everything.clone()
    };
    assert_eq!(concern_for(no_transport), "wrong family");

    let served_only = Report {
        transport_error: None,
        wrong_family: None,
        ..everything
    };
    assert!(concern_for(served_only).contains("org/other"));
}

#[test]
fn the_probe_warns_and_never_vetoes() {
    // Every outcome is either "start now" or "ask" — there is no answer that
    // refuses. Benchmarking a base checkpoint is a real thing to do on purpose.
    for report in [
        Report {
            transport_error: Some("connection refused".into()),
            ..Report::default()
        },
        Report {
            wrong_family: Some("wrong family".into()),
            ..Report::default()
        },
        Report {
            served_instead: Some(Vec::new()),
            ..Report::default()
        },
        Report {
            answers: answers(false),
            ..Report::default()
        },
        Report {
            answers: answers(true),
            ..Report::default()
        },
    ] {
        let clean = report.is_clean();
        let mut pre = resolved(report);
        let decided = pre.poll(&target()).expect("an answered check decides");
        assert_eq!(decided, clean, "a clean report starts, a concern asks");
        // A concern replaces the spinner with the question. A clean check keeps
        // `Checking` and is never drawn again — `poll_preflight` drops the whole
        // pre-flight in the same tick.
        assert_eq!(pre.is_checking(), clean);
    }
}
