// SPDX-License-Identifier: AGPL-3.0-only

use super::*;
use crate::result::VerdictKind;

fn configured(variant: Variant) -> Bfcl {
    let mut b = Bfcl::new(variant);
    let v = ParamValues::defaults(&b.parameters());
    b.configure(&v).unwrap();
    b
}

#[test]
fn the_two_variants_are_distinct_benchmarks() {
    assert_ne!(SUBSET_DESCRIPTOR.id, FULL_DESCRIPTOR.id);
    assert_eq!(Bfcl::new(Variant::Subset).descriptor().id, "bfcl-subset");
    assert_eq!(Bfcl::new(Variant::Full).descriptor().id, "bfcl-full");
}

#[test]
fn subset_defaults_reproduce_the_golden_draw() {
    let b = configured(Variant::Subset);
    assert_eq!(b.spec, DrawSpec::golden());
}

#[test]
fn full_defaults_take_every_sample_of_the_scored_categories() {
    let b = configured(Variant::Full);
    // 100% with no floor is arithmetically the same selection as `full()`.
    assert_eq!(b.spec.subset_floor, None);
    assert_eq!(b.spec.take_count("simple_python", 400), 400);
    assert_eq!(b.spec.take_count("live_relevance", 16), 0);
}

#[test]
fn defaults_are_the_mlperf_generation_config() {
    let b = Bfcl::new(Variant::Subset);
    let v = ParamValues::defaults(&b.parameters());
    assert_eq!(v.usize("max_new_tokens").unwrap(), 1024);
    assert_eq!(v.float("temperature").unwrap(), 0.0);
    assert_eq!(v.usize("subset_floor").unwrap(), 25);
}

#[test]
fn a_changed_percentage_changes_the_draw() {
    let mut b = Bfcl::new(Variant::Subset);
    let mut v = ParamValues::defaults(&b.parameters());
    v.set("non_live_pct", ParamValue::Float(20.0));
    b.configure(&v).unwrap();
    assert_eq!(b.spec.take_count("simple_python", 400), 80);
    assert_ne!(b.spec, DrawSpec::golden());
}

fn scores(overall: f64, normalized: f64) -> Scores {
    Scores {
        overall_accuracy: overall,
        normalized_single_turn_score: normalized,
        category_scores: BTreeMap::new(),
        subset_scores: BTreeMap::new(),
        total_samples: 995,
        unmatched_responses: 0,
    }
}

#[test]
fn the_verdict_gates_on_both_mlperf_floors() {
    let mut b = configured(Variant::Subset);

    b.scores = Some(scores(87.44, 88.53));
    assert_eq!(b.verdict().kind, VerdictKind::Pass);

    // Just under the overall floor.
    b.scores = Some(scores(83.63, 90.0));
    let v = b.verdict();
    assert_eq!(v.kind, VerdictKind::Fail);
    assert!(
        v.reason.contains("BELOW THE MLPERF-EDGE FLOOR"),
        "{}",
        v.reason
    );

    // Overall fine, normalized under its own floor.
    b.scores = Some(scores(90.0, 85.31));
    assert_eq!(b.verdict().kind, VerdictKind::Fail);

    // Exactly on both floors passes — the thresholds are inclusive.
    b.scores = Some(scores(MLPERF_FLOOR_OVERALL, MLPERF_FLOOR_NORMALIZED));
    assert_eq!(b.verdict().kind, VerdictKind::Pass);
}

#[test]
fn the_verdict_always_states_the_measured_values_and_the_floor() {
    let mut b = configured(Variant::Subset);
    b.scores = Some(scores(87.44, 88.53));
    let reason = b.verdict().reason;
    assert!(reason.contains("87.44") && reason.contains("83.64"));
    assert!(reason.contains("n=995"));
}

#[test]
fn an_unscored_run_is_info_not_a_pass() {
    let b = configured(Variant::Subset);
    assert_eq!(b.verdict().kind, VerdictKind::Info);
}

#[test]
fn reconfiguring_clears_generated_responses() {
    let mut b = configured(Variant::Subset);
    b.responses.push(serde_json::json!({"sample_id": "x"}));
    b.cursor = 7;
    b.tool_call_samples = 3;
    let v = ParamValues::defaults(&b.parameters());
    b.configure(&v).unwrap();
    assert!(b.responses.is_empty() && b.cursor == 0 && b.tool_call_samples == 0);
}

/// The committed baseline pins the draw each variant actually makes.
///
/// ★ Three places state a draw size — the variant's `expected_samples`, the
/// arithmetic the parameter defaults produce (`draw_tests`), and the `samples`
/// bound in `.benchmarks/<id>/BASELINE.json`. The first two are tested against
/// each other; nothing tied either to the third, which is the only one that
/// actually FAILS a run. A baseline pinned to a draw the benchmark no longer
/// makes fails every honest run, and a baseline pinned to nothing accepts a
/// score from any draw at all — the failure this pin exists to catch, moved
/// one file over.
///
/// The pin must be EXACT (`min == max`). A one-sided `min` would accept the
/// full 3625-sample draw against subset thresholds.
#[test]
fn the_committed_baselines_pin_the_draw_each_variant_makes() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(|p| p.parent())
        .expect("repo root is two levels above the crate");

    for variant in [Variant::Subset, Variant::SubsetEcholp] {
        let id = variant.descriptor().id;
        let want = variant
            .expected_samples()
            .expect("a gated variant has a pinned draw") as f64;
        let baseline = crate::gate::read_baseline(root, id)
            .unwrap_or_else(|e| panic!("{id}: baseline does not load: {e:#}"));
        for (hw, entry) in &baseline.hardware {
            for (model, mb) in &entry.models {
                let bound = mb.metrics.get("samples").unwrap_or_else(|| {
                    panic!(
                        "{id}/{hw}/{model}: no `samples` bound — a score from any draw \
                         would be accepted against these thresholds"
                    )
                });
                assert_eq!(
                    (bound.min, bound.max),
                    (Some(want), Some(want)),
                    "{id}/{hw}/{model}: the draw must be pinned EXACTLY at {want}"
                );
                assert!(
                    bound.noise.is_none(),
                    "{id}/{hw}/{model}: a sample count is exact; noise would widen the pin"
                );
            }
        }
    }
}

#[test]
fn the_mlperf_floors_are_the_recorded_thresholds() {
    // 86.23 / 87.96 × 0.97, the `mlperf-edge-current` numbers for qwen3.6-27b.
    assert!((MLPERF_FLOOR_OVERALL - 86.23 * 0.97).abs() < 0.01);
    assert!((MLPERF_FLOOR_NORMALIZED - 87.96 * 0.97).abs() < 0.01);
}
