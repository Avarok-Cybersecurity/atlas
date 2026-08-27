// SPDX-License-Identifier: AGPL-3.0-only

//! The planned / skipped / excluded classification.

use super::*;
use crate::benchmarks::serve_matrix::host::{Absence, ServeCandidate};

fn roster() -> Vec<ServeCandidate> {
    vec![
        ServeCandidate::ready("Qwen/Qwen3.6-27B", "bf16"),
        ServeCandidate::ready("nvidia/Qwen3.6-27B-NVFP4", "nvfp4"),
        ServeCandidate::absent("Qwen/Qwen3.6-35B-A3B", "fp8", Absence::NoWeights),
        ServeCandidate::absent("facebook/nllb-200-3.3B", "-", Absence::NoKernels),
    ]
}

#[test]
fn a_quant_is_its_own_round_so_the_axis_is_model_by_quant() {
    let plan = Plan::build(
        &[
            ServeCandidate::ready("org/same-model", "bf16"),
            ServeCandidate::ready("org/same-model", "nvfp4"),
        ],
        "",
    );
    let labels: Vec<String> = plan.planned().map(Round::label).collect();
    assert_eq!(
        labels,
        vec!["org/same-model · bf16", "org/same-model · nvfp4"]
    );
}

#[test]
fn an_unservable_checkpoint_is_skipped_with_its_reason_not_dropped() {
    let plan = Plan::build(&roster(), "");
    let skipped: Vec<(&str, &str, Absence)> = plan
        .skipped()
        .map(|(round, why)| (round.model.as_str(), round.quant.as_str(), why))
        .collect();
    // Sorted by HF id: `Qwen/…-35B` (no weights) then `facebook/…` (no kernels).
    assert_eq!(
        skipped,
        vec![
            ("Qwen/Qwen3.6-35B-A3B", "fp8", Absence::NoWeights),
            ("facebook/nllb-200-3.3B", "-", Absence::NoKernels),
        ],
        "both skips survive into the plan, each carrying why"
    );
    assert_eq!(plan.planned_count(), 2);
    // Every candidate is still a row: the plan is the whole roster, classified.
    assert_eq!(plan.rounds.len(), 4);
}

#[test]
fn the_filter_excludes_without_pretending_the_box_cannot_serve_it() {
    let plan = Plan::build(&roster(), "  NVIDIA  ");
    assert_eq!(plan.planned_count(), 1);
    assert_eq!(
        plan.planned().map(Round::label).collect::<Vec<_>>(),
        ["nvidia/Qwen3.6-27B-NVFP4 · nvfp4"]
    );
    // Only the one the box COULD have served. The two unservable checkpoints
    // stay counted as unservable — the categories are disjoint, or the counts
    // in the verdict add up to more checkpoints than the box has.
    assert_eq!(plan.excluded_count(), 1);
    assert_eq!(plan.skipped().count(), 2);
    let excluded = plan
        .rounds
        .iter()
        .find(|r| r.excluded)
        .expect("one filtered out");
    assert_eq!(excluded.label(), "Qwen/Qwen3.6-27B · bf16");
    assert!(
        excluded.skipped.is_none(),
        "a filtered round must not masquerade as one the box cannot serve"
    );
}

#[test]
fn round_order_is_stable_across_runs() {
    let mut shuffled = roster();
    shuffled.reverse();
    assert_eq!(Plan::build(&roster(), ""), Plan::build(&shuffled, ""));
}

#[test]
fn a_label_without_a_quant_is_just_the_model() {
    for quant in ["-", "", "  "] {
        let plan = Plan::build(&[ServeCandidate::ready("org/m", quant)], "");
        assert_eq!(plan.rounds[0].label(), "org/m");
    }
}
