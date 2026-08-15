// SPDX-License-Identifier: AGPL-3.0-only

//! Tests for the model-variant step. The reducer half is pure; `variants_for`
//! is exercised against the REAL tree (same doctrine as `taxon_tests`), so the
//! committed `BENCH.toml` wiring — not a fixture imitating it — is what these
//! prove.

use crossterm::event::{KeyCode, KeyEvent};

use super::*;

fn key(code: KeyCode) -> KeyEvent {
    KeyEvent::from(code)
}

fn agentic_state() -> BenchState {
    let mut s = BenchState::default();
    s.target = atlas_plugin::TargetEndpoint::local(8888, "test-model");
    let index = atlas_plugin::registry::all()
        .iter()
        .position(|d| d.id == "agentic-webserver")
        .expect("registered");
    s.select(index);
    s
}

/// A synthetic pair, so the reducer tests do not depend on the tree the test
/// happens to run in. The tree-backed assertions live in
/// `the_agentic_gate_declares_both_variants_in_this_tree`.
fn two_rows() -> Vec<VariantRow> {
    let bound = |max| gate::Bound {
        min: None,
        max: Some(max),
        noise: None,
    };
    vec![
        VariantRow {
            hardware: "gb10".into(),
            checkpoint: "Qwen/Qwen3.6-35B-A3B-FP8".into(),
            title: "35B MoE flagship".into(),
            recipe: Some("qwen3.6/qwen3.6-35b-a3b-fp8-bf16head".into()),
            is_default: true,
            note: String::new(),
            metrics: vec![("sum_wall_s".into(), bound(1000.0))],
        },
        VariantRow {
            hardware: "gb10".into(),
            checkpoint: "unsloth/Qwen3.8-27B-NVFP4".into(),
            title: "dense 27B".into(),
            recipe: Some("qwen3.8/qwen3.8-27b-nvfp4-unsloth".into()),
            is_default: false,
            note: String::new(),
            metrics: vec![("sum_wall_s".into(), bound(2500.0))],
        },
    ]
}

/// The committed tree is what the TUI reads, so pin it: the agentic gate is
/// defined on the 35B MoE (the default, listed first) AND the dense
/// Qwen3.8-27B, whose own Σ-wall ceiling is the provisional 2500 s.
#[test]
fn the_agentic_gate_declares_both_variants_in_this_tree() {
    let rows = variants_for("agentic-webserver");
    assert!(
        rows.len() >= 2,
        "expected both variants, got {:?}",
        rows.iter().map(|r| &r.checkpoint).collect::<Vec<_>>()
    );
    assert!(rows[0].is_default, "the declared subject leads");
    assert_eq!(rows[0].checkpoint, "Qwen/Qwen3.6-35B-A3B-FP8");
    let dense = rows
        .iter()
        .find(|r| r.checkpoint == "unsloth/Qwen3.8-27B-NVFP4")
        .expect("the dense variant is declared");
    assert!(!dense.is_default, "the required subject is unchanged");
    assert_eq!(
        dense
            .metrics
            .iter()
            .find(|(k, _)| k == "sum_wall_s")
            .and_then(|(_, b)| b.max),
        Some(5000.0),
        "the dense variant carries its own wall ceiling"
    );
    assert!(
        dense.note.contains("PROVISIONAL") || dense.note.contains("2026-08-14"),
        "the provenance travels with the threshold"
    );
}

/// Entering a benchmark with declared variants shows them; the labels carried
/// in BENCH.toml are what the rows are titled with.
#[test]
fn entering_the_agentic_benchmark_opens_the_variant_step() {
    let mut s = agentic_state();
    s.enter_selected();
    assert_eq!(s.view, View::Variants);
    assert!(
        s.variants.iter().any(|r| r.title.contains("dense")),
        "labels from BENCH.toml title the rows: {:?}",
        s.variants.iter().map(|r| &r.title).collect::<Vec<_>>()
    );
}

/// Choosing a variant adopts its checkpoint AND its thresholds together: the
/// target model is pinned and the Σ-wall budget becomes the variant's own
/// ceiling — the schema default is only right for the 35B.
#[test]
fn choosing_the_dense_variant_adopts_model_and_wall_budget() {
    let mut s = agentic_state();
    s.variants = two_rows();
    s.choose_variant(1);
    assert_eq!(s.view, View::Params);
    assert_eq!(s.target.model, "unsloth/Qwen3.8-27B-NVFP4");
    assert!(
        s.target_model_pinned,
        "follow_live_model must not undo this"
    );
    assert_eq!(s.values.float("wall_budget_s").unwrap(), 2500.0);
    let budget_row = s
        .specs
        .iter()
        .position(|p| p.key == "wall_budget_s")
        .expect("agentic declares the budget");
    assert_eq!(s.edit[budget_row], "2500", "the form shows what will run");
    let model_row = s.specs.len() + 1;
    assert_eq!(s.edit[model_row], "unsloth/Qwen3.8-27B-NVFP4");
}

/// The `min` arm of the bound selection, mirrored from
/// `bench_resolve::apply_threshold_params`: BFCL's verdict floors pair with
/// metrics whose baselines declare `min` bounds, and choosing a variant
/// adopts them exactly like the agentic ceiling adopts its `max`.
#[test]
fn choosing_a_bfcl_variant_adopts_its_baseline_floors() {
    let bound = |min, max| gate::Bound {
        min,
        max,
        noise: None,
    };
    let state_with = |overall: gate::Bound| {
        let mut s = BenchState::default();
        s.target = atlas_plugin::TargetEndpoint::local(8888, "test-model");
        let index = atlas_plugin::registry::all()
            .iter()
            .position(|d| d.id == "bfcl-subset")
            .expect("registered");
        s.select(index);
        s.variants = vec![VariantRow {
            hardware: "gb10".into(),
            checkpoint: "unsloth/Qwen3.8-27B-NVFP4".into(),
            title: "dense 3.8".into(),
            recipe: Some("qwen3.8/qwen3.8-27b-nvfp4-unsloth-bfcl".into()),
            is_default: true,
            note: String::new(),
            metrics: vec![
                ("overall_accuracy".into(), overall),
                (
                    "normalized_single_turn_score".into(),
                    bound(Some(83.72), None),
                ),
            ],
        }];
        s.choose_variant(0);
        s
    };

    let s = state_with(bound(Some(83.82), None));
    assert_eq!(s.values.float("min_overall").unwrap(), 83.82);
    assert_eq!(s.values.float("min_normalized").unwrap(), 83.72);

    // A metric declaring BOTH bounds is ambiguous: adopt neither, keep the
    // schema default (0 = non-gating) rather than guessing a direction.
    let s = state_with(bound(Some(83.82), Some(90.0)));
    assert_eq!(
        s.values.float("min_overall").unwrap(),
        0.0,
        "ambiguous bound adopts nothing"
    );
    assert_eq!(s.values.float("min_normalized").unwrap(), 83.72);
}

/// The default variant keeps the schema default — 1000 is both, so choosing it
/// must not perturb the recorded 35B behaviour.
#[test]
fn choosing_the_default_variant_keeps_the_35b_budget() {
    let mut s = agentic_state();
    s.variants = two_rows();
    s.choose_variant(0);
    assert_eq!(s.target.model, "Qwen/Qwen3.6-35B-A3B-FP8");
    assert_eq!(s.values.float("wall_budget_s").unwrap(), 1000.0);
}

/// The step uses the same j/k/Enter/Esc grammar as the lists around it, and
/// Esc from the form retraces through the variants rather than skipping them.
#[test]
fn variant_navigation_and_the_way_back() {
    let mut s = agentic_state();
    s.variants = two_rows();
    s.view = View::Variants;
    s.variants_key(key(KeyCode::Char('j')));
    assert_eq!(s.variant_row, 1);
    s.variants_key(key(KeyCode::Char('j')));
    assert_eq!(s.variant_row, 1, "clamped at the end");
    s.variants_key(key(KeyCode::Char('k')));
    assert_eq!(s.variant_row, 0);
    s.variants_key(key(KeyCode::Esc));
    assert_eq!(s.view, View::List);

    // Enter chooses; Esc from Params goes back to the variants.
    s.view = View::Variants;
    s.variants_key(key(KeyCode::Enter));
    assert_eq!(s.view, View::Params);
    use crate::tui::app::BenchSub;
    s.on_key(key(KeyCode::Esc), BenchSub::Suite);
    assert_eq!(s.view, View::Variants);
}

/// A variant pin is scoped to the benchmark it was chosen for. Before the
/// fix, choosing the dense variant and then selecting a variantless
/// benchmark kept `unsloth/Qwen3.8-27B-NVFP4` pinned as its target — with
/// `follow_live_model` permanently disabled — for the rest of the session.
#[test]
fn a_variant_pin_is_released_when_another_benchmark_is_selected() {
    let mut s = agentic_state();
    s.variants = two_rows();
    s.choose_variant(1);
    assert_eq!(s.target.model, "unsloth/Qwen3.8-27B-NVFP4");
    assert!(s.target_model_pinned && s.variant_pinned);

    let matrix = atlas_plugin::registry::all()
        .iter()
        .position(|d| d.id == "serve-matrix")
        .expect("registered");
    s.select(matrix);
    assert!(
        !s.target_model_pinned && !s.variant_pinned,
        "the variant pin must not survive the benchmark it was chosen for"
    );
    s.follow_live_model("live/model");
    assert_eq!(
        s.target.model, "live/model",
        "the form follows the live server again"
    );
    assert_eq!(
        s.edit.last().map(String::as_str),
        Some("live/model"),
        "the model row shows what the target now is"
    );
}

/// An operator-TYPED pin is the session's word, not the variant step's:
/// switching benchmarks keeps it, exactly as before the variant feature.
#[test]
fn an_operator_typed_pin_survives_benchmark_switches() {
    let mut s = agentic_state();
    // Type a model into the target field (last form row).
    let model_row = s.specs.len() + 1;
    s.edit[model_row] = "my/endpoint-model".into();
    s.commit_row(model_row);
    assert!(s.target_model_pinned && !s.variant_pinned);

    let matrix = atlas_plugin::registry::all()
        .iter()
        .position(|d| d.id == "serve-matrix")
        .expect("registered");
    s.select(matrix);
    assert!(s.target_model_pinned, "typed pin survives");
    s.follow_live_model("live/model");
    assert_eq!(s.target.model, "my/endpoint-model");
}

/// A benchmark with no declared variants keeps the old two-step flow — and
/// selecting another benchmark clears the previous one's variants, so a stale
/// row can never adopt the wrong checkpoint.
#[test]
fn a_variantless_benchmark_skips_the_step_and_selection_clears_rows() {
    let mut s = agentic_state();
    s.variants = two_rows();
    let matrix = atlas_plugin::registry::all()
        .iter()
        .position(|d| d.id == "serve-matrix")
        .expect("registered");
    s.select(matrix);
    assert!(
        s.variants.is_empty(),
        "another benchmark's variants cleared"
    );
    s.enter_selected();
    assert_eq!(
        s.view,
        View::Params,
        "no baseline entries -> straight to the form"
    );
}
