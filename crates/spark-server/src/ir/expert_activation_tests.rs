// SPDX-License-Identifier: AGPL-3.0-only

//! Expert-activation report: multi-turn merging and the wire shape.
//!
//! Both are consumed by the `expert-categories` benchmark, which turns these
//! numbers into the expert set a category gets loaded with. A merge that
//! dropped a turn, or a wire field that vanished, would move that set without
//! any error anywhere.

use super::expert_activation::{ExpertActivationReport, ExpertLayerActivation};

fn layer(l: usize, experts: &[u32], counts: &[u32], mass: &[f32]) -> ExpertLayerActivation {
    ExpertLayerActivation {
        layer: l,
        experts: experts.to_vec(),
        counts: counts.to_vec(),
        mass: mass.to_vec(),
    }
}

fn report(layers: Vec<ExpertLayerActivation>, tokens: u64) -> ExpertActivationReport {
    ExpertActivationReport {
        scope: "prefill",
        top_k: 2,
        num_experts: 8,
        tokens_routed: tokens,
        unattributed_rows: 0,
        decode_tokens_routed: 0,
        decode_unattributed_rows: 0,
        layers,
    }
}

// ---------------------------------------------------------------- Path A

#[test]
fn merge_sums_shared_experts_and_keeps_ids_ascending() {
    let mut a = report(vec![layer(0, &[1, 5], &[3, 1], &[1.5, 0.5])], 2);
    let b = report(vec![layer(0, &[3, 5], &[2, 4], &[1.0, 2.0])], 3);
    a.merge(&b);

    let l = &a.layers[0];
    assert_eq!(
        l.experts,
        vec![1, 3, 5],
        "ids must stay ascending after merge"
    );
    assert_eq!(l.counts, vec![3, 2, 5]);
    assert_eq!(l.mass, vec![1.5, 1.0, 2.5]);
    assert_eq!(a.tokens_routed, 5, "token totals add across turns");
}

#[test]
fn merge_inserts_a_layer_the_first_turn_never_used() {
    // A second tool-loop turn can route through a layer the first prompt
    // did not reach; dropping it would under-report the category.
    let mut a = report(vec![layer(2, &[1], &[1], &[1.0])], 1);
    let b = report(
        vec![layer(0, &[4], &[1], &[1.0]), layer(5, &[2], &[1], &[1.0])],
        1,
    );
    a.merge(&b);
    assert_eq!(
        a.layers.iter().map(|l| l.layer).collect::<Vec<_>>(),
        vec![0, 2, 5],
        "layers stay ascending so consumers can binary-search"
    );
}

// ---------------------------------------------------------------- Path B

#[test]
fn merge_carries_unattributed_rows_forward() {
    // If ANY turn could not be fully attributed, the merged report must say
    // so — otherwise a partial record reads as a complete one.
    let mut a = report(vec![layer(0, &[1], &[1], &[1.0])], 1);
    let mut b = report(vec![layer(0, &[1], &[1], &[1.0])], 1);
    b.unattributed_rows = 9;
    a.merge(&b);
    assert_eq!(a.unattributed_rows, 9);
}

#[test]
fn merging_a_prefill_only_turn_weakens_the_scope() {
    // A serve restarted mid-conversation, or a turn that ran before decode
    // attribution existed, must not let the merged report claim coverage it
    // does not have. The weaker scope wins.
    let mut a = report(vec![layer(0, &[1], &[1], &[1.0])], 1);
    a.scope = "prefill+decode";
    a.decode_tokens_routed = 40;
    let mut b = report(vec![layer(0, &[1], &[1], &[1.0])], 1);
    b.scope = "prefill";
    a.merge(&b);
    assert_eq!(a.scope, "prefill", "a prefill-only turn weakens the merge");
    assert_eq!(a.decode_tokens_routed, 40);
}

#[test]
fn merging_two_decode_scoped_turns_keeps_the_scope_and_sums_decode() {
    let mut a = report(vec![layer(0, &[1], &[1], &[1.0])], 1);
    a.scope = "prefill+decode";
    a.decode_tokens_routed = 40;
    a.decode_unattributed_rows = 3;
    let mut b = report(vec![layer(0, &[2], &[1], &[1.0])], 1);
    b.scope = "prefill+decode";
    b.decode_tokens_routed = 25;
    b.decode_unattributed_rows = 1;
    a.merge(&b);
    assert_eq!(a.scope, "prefill+decode");
    assert_eq!(a.decode_tokens_routed, 65);
    assert_eq!(a.decode_unattributed_rows, 4);
}

#[test]
fn merging_an_empty_report_changes_nothing() {
    let before = report(vec![layer(0, &[1], &[1], &[1.0])], 1);
    let mut a = before.clone();
    a.merge(&report(Vec::new(), 0));
    assert_eq!(a, before);
}

// ---------------------------------------------------------------- Path C
// The wire shape. The benchmark reads these exact key names off `usage`.

#[test]
fn wire_shape_carries_every_field_the_benchmark_reads() {
    let r = report(vec![layer(3, &[1, 7], &[10, 2], &[4.0, 0.5])], 6);
    let wire = crate::openai::encode_expert_activation(&r);
    let json = serde_json::to_value(&wire).expect("serializable");

    assert_eq!(json["scope"], "prefill");
    assert_eq!(json["top_k"], 2);
    assert_eq!(json["num_experts"], 8);
    assert_eq!(json["tokens_routed"], 6);
    assert_eq!(json["unattributed_rows"], 0);

    let l = &json["layers"][0];
    assert_eq!(l["layer"], 3);
    assert_eq!(l["experts"], serde_json::json!([1, 7]));
    assert_eq!(l["counts"], serde_json::json!([10, 2]));
    assert_eq!(l["mass"], serde_json::json!([4.0, 0.5]));
}

#[test]
fn usage_omits_the_field_entirely_when_no_report_was_requested() {
    // Absent, not empty: a consumer must be able to tell "this serve is not
    // instrumented" from "this prompt used no experts". Same distinction
    // `accepted_prediction_tokens` makes for speculation.
    let usage = crate::openai::Usage {
        prompt_tokens: 1,
        completion_tokens: 1,
        total_tokens: 2,
        prompt_tokens_details: None,
        completion_tokens_details: None,
        time_to_first_token_ms: 0.0,
        response_tokens_per_second: 0.0,
        expert_activation: None,
    };
    let json = serde_json::to_value(&usage).expect("serializable");
    assert!(
        json.get("expert_activation").is_none(),
        "the key must be absent, got: {json}"
    );
}
