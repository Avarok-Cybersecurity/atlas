// SPDX-License-Identifier: AGPL-3.0-only

//! The emitted TOML and stats artifact.
//!
//! The TOML block is the product: it is pasted into MODEL.toml and read at
//! build time by `parse_expert_categories`, which PANICS the build on
//! anything malformed. So the test that matters is that what this emits is
//! what that parser accepts — asserted by parsing it, not by matching
//! strings.

use super::*;

/// `(layer, [(expert, count, mass)])` — one layer of a fixture response.
type LayerFixture = (usize, Vec<(u32, u32, f64)>);
use crate::benchmarks::expert_categories::aggregate::{Accumulator, CategoryTotals};
use crate::benchmarks::expert_categories::usage::{Activation, LayerActivation};

fn budgets() -> Vec<CategoryBudget> {
    let mut a = Accumulator::new();
    let act = |layers: Vec<LayerFixture>| Activation {
        top_k: 2,
        num_experts: 8,
        tokens_routed: 1,
        unattributed_rows: 0,
        layers: layers
            .into_iter()
            .map(|(layer, experts)| LayerActivation { layer, experts })
            .collect(),
    };
    a.feed(
        "code-python",
        &act(vec![(0, vec![(1, 1, 0.9), (4, 1, 0.1)])]),
    )
    .unwrap();
    a.feed("sql", &act(vec![(0, vec![(4, 1, 0.9), (6, 1, 0.1)])]))
        .unwrap();
    a.budgets(0.9)
}

// ---------------------------------------------------------------- Path A

#[test]
fn emitted_toml_round_trips_through_a_toml_parser() {
    let out = toml_block("Qwen/Test-MoE", "abc123", 48, &budgets());
    let doc: toml::Value = toml::from_str(&out).expect("emitted TOML must parse");
    let cats = doc
        .get("expert_categories")
        .and_then(|v| v.as_table())
        .expect("[expert_categories] table");

    let py = cats.get("code-python").expect("the category table");
    assert_eq!(
        py.get("coverage").and_then(toml::Value::as_float),
        Some(0.9)
    );
    assert_eq!(py.get("prompts").and_then(toml::Value::as_integer), Some(1));

    let layers = py
        .get("layers")
        .and_then(|v| v.as_table())
        .expect("layers table");
    let ids: Vec<i64> = layers["0"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_integer().unwrap())
        .collect();
    assert_eq!(ids, vec![1], "0.9 of the mass is expert 1 alone");
}

#[test]
fn emitted_toml_carries_its_own_provenance() {
    // Found in a MODEL.toml a year later, the block has to say what made it.
    let out = toml_block("Qwen/Test-MoE", "deadbeef", 48, &budgets());
    assert!(out.contains("Qwen/Test-MoE"));
    assert!(out.contains("deadbeef"));
    assert!(out.contains("max_tokens 48"));
    assert!(
        out.contains("REBUILD"),
        "must say the table is read at build time"
    );
    assert!(
        out.contains("closure hash"),
        "must warn that gate records are invalidated"
    );
}

// ---------------------------------------------------------------- Path B

#[test]
fn a_layer_never_emits_an_empty_id_list() {
    // An empty list would tell boot-time loading to load NO experts for that
    // layer. The aggregator drops zero-mass layers; this pins that the
    // emitter cannot reintroduce one.
    let out = toml_block("m", "h", 48, &budgets());
    assert!(!out.contains("= []"), "empty expert list emitted:\n{out}");
}

#[test]
fn stats_carry_the_full_distribution_not_just_the_budgeted_set() {
    // So a different coverage can be evaluated without re-running the
    // benchmark — the expensive part is the measurement, not the arithmetic.
    let mut a = Accumulator::new();
    a.feed(
        "code-python",
        &Activation {
            top_k: 2,
            num_experts: 8,
            tokens_routed: 1,
            unattributed_rows: 0,
            layers: vec![LayerActivation {
                layer: 0,
                experts: vec![(1, 1, 0.9), (4, 1, 0.1)],
            }],
        },
    )
    .unwrap();
    let b = a.budgets(0.9);
    let stats = stats_json("m", "h", 0.9, &a, &b, None);

    let layer0 = &stats["categories"]["code-python"]["layers"]["0"];
    assert_eq!(
        layer0["experts"].as_array().unwrap().len(),
        2,
        "both experts must appear, including the one the budget dropped"
    );
    assert_eq!(layer0["budgeted"], serde_json::json!([1]));
    assert!((layer0["total_mass"].as_f64().unwrap() - 1.0).abs() < 1e-9);
}

#[test]
fn stats_report_cross_category_overlap() {
    // The evidence that categorization is possible: the two fixtures share
    // one expert of two in layer 0 after budgeting at 1.0.
    let mut a = Accumulator::new();
    let act = |experts: Vec<(u32, u32, f64)>| Activation {
        top_k: 2,
        num_experts: 8,
        tokens_routed: 1,
        unattributed_rows: 0,
        layers: vec![LayerActivation { layer: 0, experts }],
    };
    a.feed("code-python", &act(vec![(1, 1, 0.5), (4, 1, 0.5)]))
        .unwrap();
    a.feed("sql", &act(vec![(4, 1, 0.5), (6, 1, 0.5)])).unwrap();
    let b = a.budgets(1.0);
    let stats = stats_json("m", "h", 1.0, &a, &b, None);
    let j = stats["jaccard_budgeted"]["code-python|sql"]
        .as_f64()
        .unwrap();
    assert!((j - 1.0 / 3.0).abs() < 1e-9, "one shared of three: {j}");
}

// ---------------------------------------------------------------- Path C

#[test]
fn mean_experts_of_an_empty_budget_is_zero_not_a_nan() {
    // A NaN here would render as "NaN experts/layer" in the run table and
    // poison any downstream comparison.
    let empty = CategoryBudget {
        category: "x".into(),
        coverage: 0.9,
        totals: CategoryTotals::default(),
        layers: Vec::new(),
    };
    assert_eq!(mean_experts(&empty), 0.0);
}

#[test]
fn corpus_hash_changes_when_a_single_row_changes() {
    // The hash is how two reports are compared without trusting their
    // labels; if it did not move with the corpus it would be worse than
    // nothing.
    let a = corpus_sha256("{\"id\":\"py-001\"}\n");
    let b = corpus_sha256("{\"id\":\"py-002\"}\n");
    assert_ne!(a, b);
    assert_eq!(a.len(), 64);
}
