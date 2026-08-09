// SPDX-License-Identifier: AGPL-3.0-only

//! Pure-logic tests for the coherence probe. The socket-level behaviour is
//! covered end to end in `tests/coherence.rs`, against the mock endpoint.

use super::*;

/// Does the answer text satisfy the check? Mirrors `ask`'s decision so the
/// matching rule can be tested without a server.
fn accepts(check: &Check, answer: &str) -> bool {
    let lowered = answer.to_lowercase();
    check.accept.iter().any(|a| lowered.contains(a))
}

fn check(label: &str) -> &'static Check {
    CHECKS
        .iter()
        .find(|c| c.label == label)
        .expect("check exists")
}

#[test]
fn a_model_answering_correctly_passes_however_it_phrases_it() {
    let arith = check("arithmetic");
    for answer in ["4", "4.", " 4\n", "The answer is 4.", "Four", "FOUR"] {
        assert!(accepts(arith, answer), "should accept {answer:?}");
    }
    let recall = check("recall");
    for answer in ["Paris", "paris", "The capital of France is Paris."] {
        assert!(accepts(recall, answer), "should accept {answer:?}");
    }
}

#[test]
fn a_wrong_or_empty_answer_fails() {
    let arith = check("arithmetic");
    for answer in ["5", "", "I cannot help with that", "twenty-two"] {
        assert!(!accepts(arith, answer), "should reject {answer:?}");
    }
    // "22" contains no "4" — a garbled quantization producing digits is not a pass.
    assert!(!accepts(arith, "22"));
    assert!(!accepts(check("recall"), "London"));
}

#[test]
fn the_checks_cover_two_different_faculties() {
    // Arithmetic and recall fail independently; a probe made of two arithmetic
    // questions would be one signal counted twice.
    assert_eq!(CHECKS.len(), 2);
    assert!(CHECKS.iter().any(|c| c.label == "arithmetic"));
    assert!(CHECKS.iter().any(|c| c.label == "recall"));
}

#[test]
fn every_accept_pattern_is_lower_case() {
    // `accepts` lower-cases the answer, so an upper-case pattern could never
    // match and the check would silently always fail.
    for c in CHECKS {
        for pattern in c.accept {
            assert_eq!(
                *pattern,
                pattern.to_lowercase(),
                "{}: {pattern:?} must be lower-case",
                c.label
            );
        }
    }
}

#[test]
fn probing_is_the_default_but_it_only_ever_warns() {
    // On by default so a wrong --model is noticed; advisory so a benchmark
    // aimed at a different model is still allowed to run.
    assert_eq!(CoherencePolicy::default(), CoherencePolicy::Probe);
}

#[test]
fn an_empty_answer_reads_as_answered_nothing() {
    // A model that returns no text at all produced the useless message
    // `recall answered ""`. Say what actually happened.
    let report = Report {
        answers: vec![Answer {
            label: "recall",
            answer: String::new(),
            passed: false,
        }],
        transport_error: None,
        served_instead: None,
        wrong_family: None,
    };
    let target = TargetEndpoint::local(8888, "m");
    let concern = report.concern(&target).expect("a concern");
    assert!(concern.contains("answered nothing"), "{concern}");
    assert!(!concern.contains("\"\""), "no empty quotes: {concern}");
    assert!(!report.is_clean());
}

#[test]
fn the_concern_describes_rather_than_forbids() {
    let report = Report {
        answers: vec![Answer {
            label: "recall",
            answer: "London".into(),
            passed: false,
        }],
        transport_error: None,
        served_instead: None,
        wrong_family: None,
    };
    let concern = report
        .concern(&TargetEndpoint::local(8888, "m"))
        .expect("a concern");
    // The old wording called it a failure and told the user to pass a flag.
    assert!(!concern.contains("failed"), "not a verdict: {concern}");
    assert!(
        concern.contains("still valid"),
        "says the run may proceed: {concern}"
    );
    assert!(concern.contains("different model"), "{concern}");
}

#[test]
fn a_transport_error_is_worded_as_one() {
    let report = Report {
        answers: Vec::new(),
        transport_error: Some("connection refused".into()),
        served_instead: None,
        wrong_family: None,
    };
    let concern = report
        .concern(&TargetEndpoint::local(8888, "m"))
        .expect("a concern");
    assert!(concern.contains("did not answer"), "{concern}");
    assert!(
        !concern.contains("different model"),
        "a closed port is not a model problem: {concern}"
    );
}

#[test]
fn a_clean_report_has_nothing_to_say() {
    let report = Report {
        answers: vec![Answer {
            label: "recall",
            answer: "Paris".into(),
            passed: true,
        }],
        transport_error: None,
        served_instead: None,
        wrong_family: None,
    };
    assert!(report.is_clean());
    assert!(report.concern(&TargetEndpoint::local(8888, "m")).is_none());
}

#[test]
fn a_long_answer_is_truncated_for_the_error_message() {
    let long = "x".repeat(500);
    let out = truncate(&long, 80);
    assert_eq!(out.chars().count(), 81, "80 chars plus the ellipsis");
    assert!(out.ends_with('…'));
    // Short answers survive intact, trimmed.
    assert_eq!(truncate("  Paris\n", 80), "Paris");
}

#[test]
fn truncate_counts_characters_not_bytes() {
    // A byte-slicing implementation panics on a multi-byte boundary.
    let s = "é".repeat(200);
    let out = truncate(&s, 10);
    assert_eq!(out.chars().count(), 11);
}

#[test]
fn a_wrong_model_name_is_reported_ahead_of_the_answers() {
    // THE case this check exists for: Atlas answers a completion whatever
    // model name it is sent, so the questions cannot see the mistake. Only the
    // model list can — and it must lead, because a wrong name explains any
    // oddity downstream of it.
    let report = Report {
        answers: vec![Answer {
            label: "recall",
            answer: String::new(),
            passed: false,
        }],
        transport_error: None,
        served_instead: Some(vec!["nvidia/Qwen3.6-27B-NVFP4".into()]),
        wrong_family: None,
    };
    let target = TargetEndpoint::local(8888, "does/not-exist");
    let concern = report.concern(&target).expect("a concern");
    assert!(
        concern.contains("nvidia/Qwen3.6-27B-NVFP4"),
        "names what IS served: {concern}"
    );
    assert!(
        concern.contains("does/not-exist"),
        "and what was asked for: {concern}"
    );
    assert!(
        !concern.contains("answered nothing"),
        "the cause leads, not the symptom: {concern}"
    );
    assert!(!report.is_clean());
}

#[test]
fn a_server_serving_the_requested_model_is_clean() {
    let report = Report {
        answers: vec![Answer {
            label: "recall",
            answer: "Paris".into(),
            passed: true,
        }],
        transport_error: None,
        served_instead: None,
        wrong_family: None,
    };
    assert!(report.is_clean());
}

/// Gate A's thresholds were measured on the 35B MoE. Pointing it at the dense
/// 27B — a perfectly healthy server, serving exactly the model that was
/// requested — must still say something, because the numbers would compare to
/// nothing.
#[test]
fn a_gate_run_against_the_wrong_model_family_is_reported() {
    use crate::registry;
    let agentic = registry::find("agentic-webserver").expect("registered");
    let expect = agentic.intended_for.expect("gate A names its model");

    assert!(
        expect.accepts("Qwen/Qwen3.6-35B-A3B-FP8"),
        "the FP8 flagship"
    );
    assert!(
        expect.accepts("nvidia/Qwen3.6-35B-A3B-NVFP4"),
        "and the NVFP4 variant of the same family"
    );
    assert!(
        !expect.accepts("unsloth/Qwen3.6-27B-NVFP4"),
        "the dense 27B is a DIFFERENT gate"
    );
}

#[test]
fn the_bfcl_gates_accept_both_of_their_models() {
    use crate::registry;
    let expect = registry::find("bfcl-subset")
        .expect("registered")
        .intended_for
        .expect("names its models");
    // Gate D is the dense 27B, gate B the 35B MoE — both are legitimate.
    assert!(expect.accepts("unsloth/Qwen3.6-27B-NVFP4"));
    assert!(expect.accepts("Qwen/Qwen3.6-35B-A3B-FP8"));
    assert!(!expect.accepts("meta-llama/Llama-3.1-8B"));
}

#[test]
fn a_latency_sweep_constrains_nothing() {
    use crate::registry;
    // These measure whatever they are pointed at; a constraint here would be
    // an invention, not a fact about the benchmark.
    for id in ["concurrency-sweep", "ttft-warm-gate", "ttft-cold-gate"] {
        assert!(
            registry::find(id)
                .expect("registered")
                .intended_for
                .is_none(),
            "{id} must not claim a model it has no threshold for"
        );
    }
}

#[test]
fn the_wrong_family_note_outranks_an_odd_answer() {
    // A gate run on the wrong model explains the numbers before they are
    // measured, so it must lead.
    let report = Report {
        answers: vec![Answer {
            label: "recall",
            answer: String::new(),
            passed: false,
        }],
        transport_error: None,
        served_instead: None,
        wrong_family: Some("Gate A is defined on the 35B MoE flagship".into()),
    };
    let concern = report
        .concern(&TargetEndpoint::local(8888, "m"))
        .expect("a concern");
    assert!(concern.contains("35B MoE"), "{concern}");
    assert!(
        !concern.contains("answered nothing"),
        "the cause leads: {concern}"
    );
    assert!(!report.is_clean());
}

#[test]
fn a_transport_error_carrying_a_hint_is_not_cut_off_mid_clause() {
    // The pre-flight modal is where a benchmark run against a modelless server
    // gets explained, and the explanation now ends in an instruction. Bounding
    // it at 140 chars truncated that instruction to "choose a model and a…".
    let e = "endpoint returned \"HTTP/1.1 503 Service Unavailable\": no model is loaded — \
             open the Library (press 4 in the dashboard), choose a model and a recipe, \
             and start it; then retry this request";
    let out = super::one_line(e);
    assert!(
        !out.ends_with('…'),
        "the actionable half must survive the bound: {out}"
    );
    assert!(out.contains("retry this request"), "{out}");
}

#[test]
fn a_runaway_error_chain_is_still_bounded() {
    let out = super::one_line(&"boom ".repeat(500));
    assert!(out.chars().count() <= 281, "still bounded: {}", out.len());
    assert!(out.ends_with('…'));
}

#[test]
fn serving_nothing_does_not_promise_numbers() {
    // The wrong-model wording ("the run WILL produce numbers; they will just be
    // for a different model") is false when nothing is loaded: every request is
    // refused, so there are no numbers at all.
    let target = TargetEndpoint {
        base_url: "http://127.0.0.1:8123".into(),
        model: "x".into(),
    };
    let report = Report {
        served_instead: Some(Vec::new()),
        ..Default::default()
    };
    let c = report
        .concern(&target)
        .expect("serving nothing is a concern");
    assert!(
        !c.contains("WILL produce"),
        "must not promise numbers it cannot produce: {c}"
    );
    assert!(c.contains("no model loaded"), "{c}");
    assert!(c.contains("Library"), "and says how to fix it: {c}");
}

#[test]
fn a_thinking_model_that_reasons_to_the_answer_passes() {
    // Regression: the probe read only `text`. A thinking model spends the
    // whole budget on `reasoning_content`, so `text` came back empty and both
    // checks reported "answered nothing" -- which the message then blamed on a
    // mis-quantized or base checkpoint. It measured verbosity and called it
    // brain damage.
    let (passed, answer) = super::judge("", "2 + 2 = 4, so the answer is 4", &["4", "four"]);
    assert!(passed, "the fact is present, in the reasoning");
    assert!(
        answer.contains("4"),
        "a failure must still quote something legible, not an empty string: {answer:?}"
    );
}

#[test]
fn the_answer_is_preferred_over_the_reasoning_when_both_are_present() {
    let (passed, answer) = super::judge("4", "let me think about 4", &["4"]);
    assert!(passed);
    assert_eq!(answer, "4", "quote the reply, not the thinking");
}

#[test]
fn genuine_garbage_still_fails() {
    // The probe must keep catching what it exists to catch.
    let (passed, answer) = super::judge("zzz zzz zzz", "", &["4", "four"]);
    assert!(!passed);
    assert_eq!(answer, "zzz zzz zzz");
}
