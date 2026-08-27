// SPDX-License-Identifier: AGPL-3.0-only

//! Upstream-parity tests for the inline-scorer port.
//!
//! The expected values in `parity_fixtures.json` were produced by EXECUTING
//! `AgenticInferenceInlineScorer` at mlcommons/endpoints@7935df4 over the case
//! list in `gen_parity_fixtures.py` — nothing in the fixture file was typed by
//! hand. These tests therefore prove agreement with the upstream scorer on
//! every pinned case, which is the only parity claim a port can honestly make
//! while the official dataset (and hence an end-to-end cross-check against the
//! official client) does not exist.

use serde_json::Value;

use super::*;

const FIXTURES: &str = include_str!("../../../assets/mlperf-agentic/parity_fixtures.json");
const GENERATOR: &str = include_str!("../../../assets/mlperf-agentic/gen_parity_fixtures.py");

fn fixtures() -> Value {
    serde_json::from_str(FIXTURES).expect("parity_fixtures.json parses")
}

fn cases<'a>(fixtures: &'a Value, key: &str) -> &'a Vec<Value> {
    fixtures
        .get(key)
        .and_then(Value::as_array)
        .unwrap_or_else(|| panic!("fixture set {key} missing"))
}

/// The fixtures must state which upstream commit produced them; a regenerated
/// file with a new commit id fails here until this pin is consciously moved.
#[test]
fn fixtures_are_pinned_to_the_recorded_upstream_commit() {
    use sha2::{Digest, Sha256};

    let f = fixtures();
    assert_eq!(f["upstream_commit"], "7935df4");
    assert_eq!(
        f["upstream_file"],
        "src/inference_endpoint/evaluation/scoring.py"
    );
    assert_eq!(f["upstream_class"], "AgenticInferenceInlineScorer");
    for (key, count) in [
        ("intent_cases", 15),
        ("gt_intent_cases", 5),
        ("bash_cases", 18),
        ("alias_cases", 58),
        ("wrapper_cases", 6),
        ("turn_cases", 8),
        ("domain_cases", 8),
    ] {
        assert_eq!(cases(&f, key).len(), count, "{key} fixture count");
    }
    assert_eq!(
        format!("{:x}", Sha256::digest(FIXTURES.as_bytes())),
        "cb70bbda8d9da4883480f8d08c4fd12b826b6a0c83ed99b738bfde44916fbf9e"
    );
    assert_eq!(
        format!("{:x}", Sha256::digest(GENERATOR.as_bytes())),
        "d13c91fc257e1d198ce5a892827656e384dc79b4f01abb67dedeb7f0ce403dc5"
    );
}

#[test]
fn model_intent_matches_upstream_on_every_case() {
    let f = fixtures();
    for case in cases(&f, "intent_cases") {
        let got = model_intent(&case["turn"]);
        let want = case["expected"].as_str().map(str::to_string);
        assert_eq!(got, want, "turn {}", case["turn"]);
    }
}

#[test]
fn ground_truth_intents_match_upstream_on_every_case() {
    let f = fixtures();
    for case in cases(&f, "gt_intent_cases") {
        let got = ground_truth_intents(&case["turn"]);
        let want: Vec<String> = case["expected"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_str().unwrap().to_string())
            .collect();
        assert_eq!(got, want, "turn {}", case["turn"]);
    }
}

fn as_bash_turn(arguments: &Value) -> Value {
    serde_json::json!({
        "tool_calls": [{"function": {"name": "bash", "arguments": arguments}}]
    })
}

fn assert_bash_cases(f: &Value, key: &str) {
    for case in cases(f, key) {
        let got = bash_actions(&as_bash_turn(&case["arguments"]));
        let want: Vec<String> = case["expected"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_str().unwrap().to_string())
            .collect();
        assert_eq!(got, want, "arguments {}", case["arguments"]);
    }
}

#[test]
fn bash_actions_match_upstream_on_every_case() {
    assert_bash_cases(&fixtures(), "bash_cases");
}

/// One generated case per upstream alias entry: the whole table is pinned,
/// not spot-checked.
#[test]
fn every_alias_table_entry_matches_upstream() {
    assert_bash_cases(&fixtures(), "alias_cases");
}

#[test]
fn every_shell_wrapper_matches_upstream() {
    assert_bash_cases(&fixtures(), "wrapper_cases");
}

/// The port's table must hold EXACTLY the upstream keys — the per-entry cases
/// above cannot see an entry this port added that upstream does not have.
#[test]
fn the_alias_table_has_exactly_the_upstream_keys() {
    let f = fixtures();
    let upstream_keys: std::collections::BTreeSet<String> = cases(&f, "alias_cases")
        .iter()
        .map(|c| {
            let cmd = c["arguments"]["cmd"].as_str().unwrap();
            cmd.strip_suffix(" --arg").unwrap().to_string()
        })
        .collect();
    let port_keys: std::collections::BTreeSet<String> = alias_table()
        .iter()
        .map(|entry| entry.0.to_string())
        .collect();
    assert_eq!(port_keys, upstream_keys);
    assert_eq!(
        alias_table().len(),
        58,
        "no duplicate keys hiding in the table"
    );
}

#[test]
fn turn_scores_match_upstream_on_every_case() {
    let f = fixtures();
    for case in cases(&f, "turn_cases") {
        let domain = match case["domain"].as_str().unwrap() {
            "workflow" => Domain::Workflow,
            _ => Domain::Coding,
        };
        let got = score_turn(domain, &case["gt"], &case["model"]);
        let want = case["expected"].as_f64().unwrap();
        assert!(
            (got - want).abs() < 1e-9,
            "domain {domain:?}: got {got}, upstream {want} (gt {}, model {})",
            case["gt"],
            case["model"]
        );
    }
}

#[test]
fn domain_matches_upstream_on_every_case() {
    let f = fixtures();
    for case in cases(&f, "domain_cases") {
        let id = case["conversation_id"].as_str().unwrap();
        let want = if case["workflow"].as_bool().unwrap() {
            Domain::Workflow
        } else {
            Domain::Coding
        };
        assert_eq!(domain_of(id), want, "{id}");
    }
}

/// Denominator semantics, from the upstream `score()` body rather than a
/// fixture (they need a whole events pipeline to exercise upstream): a ground
/// truth with no scorable content is EXCLUDED; an issued turn whose model
/// output is missing scores 0 and STAYS.
#[test]
fn unscorable_ground_truth_is_excluded_but_missing_output_scores_zero() {
    let no_gt = serde_json::json!({"content": "just prose"});
    assert!(!has_ground_truth(Domain::Coding, &no_gt));
    assert!(!has_ground_truth(Domain::Workflow, &no_gt));

    let gt = as_bash_turn(&serde_json::json!({"cmd": "make"}));
    assert!(has_ground_truth(Domain::Coding, &gt));
    // The "missing output" model turn upstream synthesizes is a bare
    // assistant role with no fields.
    let missing = serde_json::json!({"role": "assistant"});
    assert_eq!(score_turn(Domain::Coding, &gt, &missing), 0.0);

    let wf_gt = serde_json::json!({"intent_codes": ["I001"]});
    assert!(has_ground_truth(Domain::Workflow, &wf_gt));
    assert_eq!(score_turn(Domain::Workflow, &wf_gt, &missing), 0.0);
}

#[test]
fn multiset_iou_counts_duplicates_on_both_sides() {
    let g = |v: &[&str]| v.iter().map(|s| s.to_string()).collect::<Vec<_>>();
    assert_eq!(multiset_iou(&g(&["grep", "grep"]), &g(&["grep"])), 0.5);
    assert_eq!(
        multiset_iou(&g(&["make"]), &g(&["make", "make", "make"])),
        1.0 / 3.0
    );
    assert_eq!(multiset_iou(&g(&["ls"]), &g(&[])), 0.0);
    assert_eq!(multiset_iou(&g(&["ls"]), &g(&["ls"])), 1.0);
}
