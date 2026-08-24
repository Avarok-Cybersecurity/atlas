// SPDX-License-Identifier: AGPL-3.0-only

//! Pure-logic tests for the coherence probe. The socket-level behaviour is
//! covered end to end in `tests/coherence.rs`, against the mock endpoint.

use super::*;

/// Exercise the same decision seam as `ask`, without a server.
fn accepts(check: &Check, answer: &str) -> bool {
    super::judge(answer, "", check.accept).0
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
    for answer in [
        "5",
        "",
        "I cannot help with that",
        "twenty-two",
        "14",
        "404",
        "fourteen",
    ] {
        assert!(!accepts(arith, answer), "should reject {answer:?}");
    }
    assert!(!accepts(arith, "22"));
    assert!(!accepts(check("recall"), "London"));
}

#[test]
fn the_checks_cover_two_different_faculties() {
    assert_eq!(
        CHECKS
            .iter()
            .map(|check| (check.label, check.prompt, check.accept))
            .collect::<Vec<_>>(),
        vec![
            (
                "arithmetic",
                "What is 2+2? Reply with only the number.",
                &["4", "four"][..],
            ),
            (
                "recall",
                "What is the capital of France? Reply with only the city name.",
                &["paris"][..],
            ),
        ],
        "the two probes must retain distinct faculties and their measured prompts"
    );
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
    assert_eq!(
        concern,
        "http://127.0.0.1:8888 is serving \"m\", which did not answer as expected (recall answered nothing). \
         This benchmark may be aimed at a different model, or the checkpoint may be a base (non-instruct) one — \
         the run is still valid, but read the numbers with that in mind."
    );
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
    assert_eq!(
        concern,
        "http://127.0.0.1:8888 is serving \"m\", which did not answer as expected (recall answered \"London\"). \
         This benchmark may be aimed at a different model, or the checkpoint may be a base (non-instruct) one — \
         the run is still valid, but read the numbers with that in mind."
    );
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
    assert_eq!(
        concern,
        "http://127.0.0.1:8888 did not answer a test request: connection refused"
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
    let long = format!("{}{}", "a".repeat(80), "z".repeat(420));
    let out = truncate(&long, 80);
    assert_eq!(out, format!("{}…", "a".repeat(80)));
    assert_eq!(truncate("  Paris\n", 80), "Paris");
}

#[test]
fn truncate_counts_characters_not_bytes() {
    // A byte-slicing implementation panics on a multi-byte boundary.
    let s = "é".repeat(200);
    let out = truncate(&s, 10);
    assert_eq!(out, format!("{}…", "é".repeat(10)));
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
    assert_eq!(
        concern,
        "http://127.0.0.1:8888 is serving nvidia/Qwen3.6-27B-NVFP4 — not \"does/not-exist\", \
         which this benchmark is set to request. Atlas answers whatever model name it is sent, \
         so the run WILL produce numbers; they will just be for a different model than the one named."
    );
    assert!(!report.is_clean());
}

/// Gate A's thresholds were measured on the 35B MoE, which stays the DEFAULT
/// subject. Both dense 27Bs are registered non-default variants and are
/// accepted, for different reasons: Qwen3.6-27B is UNMEASURED
/// (kernels/gb10/qwen3.6-27b/BENCH.toml) so a run there baselines rather than
/// gates, while Qwen3.8-27B is MEASURED with its own thresholds
/// (kernels/gb10/qwen3.8-27b/BENCH.toml) and gates against them. That is the
/// point of per-variant baselines — neither is compared against the 35B's
/// numbers. A model outside every declared family must still be reported,
/// because its numbers would compare to nothing.
#[test]
fn the_agentic_gate_accepts_only_its_declared_model_families() {
    use crate::registry;
    let agentic = registry::find("agentic-webserver").expect("registered");
    let expect = agentic.intended_for.expect("gate A names its models");

    assert_eq!(
        expect.families,
        &["qwen3.6-35b-a3b", "qwen3.6-27b", "qwen3.8-27b"]
    );
    assert!(
        expect.accepts("Qwen/Qwen3.6-35B-A3B-FP8"),
        "the FP8 flagship"
    );
    assert!(
        expect.accepts("nvidia/Qwen3.6-35B-A3B-NVFP4"),
        "and the NVFP4 variant of the same family"
    );
    assert!(
        expect.accepts("unsloth/Qwen3.6-27B-NVFP4"),
        "the dense Qwen3.6-27B is a registered UNMEASURED baselining variant"
    );
    assert!(
        expect.accepts("unsloth/Qwen3.8-27B-NVFP4"),
        "the dense Qwen3.8-27B is a MEASURED variant with its own thresholds"
    );
    assert!(
        !expect.accepts("meta-llama/Llama-3.1-8B"),
        "an unrelated model still compares to nothing"
    );
    assert!(!expect.accepts("Qwen/Qwen3.5-27B"));
    assert!(!expect.accepts("Qwen/Qwen3.6-270B"));
}

#[test]
fn the_bfcl_gate_accepts_all_and_only_its_declared_model_families() {
    use crate::registry;
    let expect = registry::find("bfcl-subset")
        .expect("registered")
        .intended_for
        .expect("names its models");
    assert_eq!(
        expect.families,
        &["qwen3.6-27b", "qwen3.6-35b-a3b", "qwen3.8-27b"]
    );
    assert!(expect.accepts("unsloth/Qwen3.6-27B-NVFP4"));
    assert!(expect.accepts("Qwen/Qwen3.6-35B-A3B-FP8"));
    assert!(expect.accepts("unsloth/Qwen3.8-27B-NVFP4"));
    assert!(!expect.accepts("meta-llama/Llama-3.1-8B"));
    assert!(!expect.accepts("Qwen/Qwen3.5-27B"));
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
    assert_eq!(concern, "Gate A is defined on the 35B MoE flagship");
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
    assert_eq!(
        out,
        "endpoint returned \"HTTP/1.1 503 Service Unavailable\": no model is loaded — \
         open the Library (press 4 in the dashboard), choose a model and a recipe, \
         and start it; then retry this request"
    );
}

#[test]
fn a_runaway_error_chain_is_still_bounded() {
    let out = super::one_line(&"boom ".repeat(500));
    assert_eq!(out, format!("{}…", "boom ".repeat(56)));
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
    assert_eq!(
        c,
        "http://127.0.0.1:8123 has no model loaded, so this run will produce no numbers — \
         every request will be refused. Load a model first: in the dashboard open the Library \
         (press 4), choose a model and a recipe, and start it."
    );
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
    assert_eq!(answer, "2 + 2 = 4, so the answer is 4");
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
