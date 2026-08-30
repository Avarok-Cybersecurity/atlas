// SPDX-License-Identifier: AGPL-3.0-only

//! Parsing and the vacuity pins for `usage.expert_activation`.
//!
//! Each rejection here is a way a run could otherwise produce a
//! plausible-looking category table from a broken instrument.

use super::*;
use serde_json::json;

fn report(layers: serde_json::Value, tokens_routed: u64, top_k: u64) -> serde_json::Value {
    json!({
        "scope": "prefill",
        "top_k": top_k,
        "num_experts": 8,
        "tokens_routed": tokens_routed,
        "unattributed_rows": 0,
        "layers": layers,
    })
}

fn one_layer() -> serde_json::Value {
    json!([{"layer": 0, "experts": [1, 4], "counts": [2, 2], "mass": [1.0, 0.5]}])
}

// ---------------------------------------------------------------- Path A

#[test]
fn parses_a_well_formed_report() {
    let a = parse(&report(one_layer(), 2, 2), "py-001").expect("valid report");
    assert_eq!(a.top_k, 2);
    assert_eq!(a.num_experts, 8);
    assert_eq!(a.tokens_routed, 2);
    assert_eq!(a.layers.len(), 1);
    assert_eq!(a.layers[0].experts, vec![(1, 2, 1.0), (4, 2, 0.5)]);
}

// ---------------------------------------------------------------- Path B
// The pins. Each of these would otherwise average silently into a category.

#[test]
fn counts_that_do_not_match_tokens_routed_are_rejected() {
    // The core vacuity pin: every routed position picks exactly top_k
    // experts, so these two numbers are one fact seen twice. Drift means a
    // double-fold, a silent drop, or a token total nobody measured.
    let bad = report(one_layer(), 3, 2); // Σcounts = 4, expected 6
    let err = parse(&bad, "py-001").unwrap_err().to_string();
    assert!(err.contains("Σcounts = 4"), "got: {err}");
    assert!(err.contains("tokens_routed × top_k = 6"), "got: {err}");
}

#[test]
fn an_empty_layer_list_is_rejected_not_averaged_as_zero() {
    // Instrumented but recorded nothing. Folding it would drag every
    // category's mass toward zero with no error anywhere.
    let err = parse(&report(json!([]), 0, 2), "py-001")
        .unwrap_err()
        .to_string();
    assert!(err.contains("no layers"), "got: {err}");
    assert!(err.contains("Not averaging a zero"), "got: {err}");
}

#[test]
fn an_expert_id_outside_the_expert_count_is_rejected() {
    let bad = json!([{"layer": 0, "experts": [9], "counts": [2], "mass": [1.0]}]);
    let err = parse(&report(bad, 1, 2), "py-001").unwrap_err().to_string();
    assert!(err.contains("expert 9 of 8"), "got: {err}");
}

#[test]
fn non_ascending_expert_ids_are_rejected() {
    // The aggregator and the emitted TOML both assume ascending ids; if the
    // server stopped guaranteeing it, silently keeping the last write per id
    // would drop routing mass.
    let bad = json!([{"layer": 0, "experts": [4, 1], "counts": [2, 2], "mass": [1.0, 0.5]}]);
    let err = parse(&report(bad, 2, 2), "py-001").unwrap_err().to_string();
    assert!(err.contains("not strictly ascending"), "got: {err}");
}

#[test]
fn duplicate_expert_ids_are_rejected() {
    // A duplicate is a non-ascending pair; it would otherwise double-count
    // one expert's mass and bias it into every budget.
    let bad = json!([{"layer": 0, "experts": [1, 1], "counts": [2, 2], "mass": [1.0, 0.5]}]);
    let err = parse(&report(bad, 2, 2), "py-001").unwrap_err().to_string();
    assert!(err.contains("not strictly ascending"), "got: {err}");
}

#[test]
fn mismatched_parallel_arrays_are_rejected() {
    let bad = json!([{"layer": 0, "experts": [1, 4], "counts": [2], "mass": [1.0, 0.5]}]);
    let err = parse(&report(bad, 2, 2), "py-001").unwrap_err().to_string();
    assert!(err.contains("mismatched arrays"), "got: {err}");
}

#[test]
fn a_nan_or_negative_mass_is_rejected() {
    // NaN sorts unpredictably in the budget comparison and would silently
    // move which experts a category keeps.
    let neg = json!([{"layer": 0, "experts": [1], "counts": [2], "mass": [-0.5]}]);
    let err = parse(&report(neg, 1, 2), "py-001").unwrap_err().to_string();
    assert!(err.contains("mass -0.5"), "got: {err}");
}

// ---------------------------------------------------------------- Path C

#[test]
fn a_dense_shaped_report_is_rejected() {
    let dense = json!({
        "scope": "prefill", "top_k": 0, "num_experts": 0,
        "tokens_routed": 0, "unattributed_rows": 0, "layers": [],
    });
    let err = parse(&dense, "py-001").unwrap_err().to_string();
    assert!(err.contains("not an MoE report"), "got: {err}");
}

#[test]
fn missing_fields_name_themselves() {
    let mut v = report(one_layer(), 2, 2);
    v.as_object_mut().unwrap().remove("tokens_routed");
    let err = parse(&v, "py-001").unwrap_err().to_string();
    assert!(err.contains("`tokens_routed`"), "got: {err}");
    assert!(err.contains("py-001"), "the error must name the row: {err}");
}

#[test]
fn the_absent_field_error_names_the_flag_that_fixes_it() {
    // The fix text IS the product here: an operator hitting this needs to
    // know it is a serve flag, not a request parameter.
    let err = missing_report_error("py-007").to_string();
    assert!(err.contains("--expert-telemetry"), "got: {err}");
    assert!(err.contains("py-007"), "got: {err}");
}
